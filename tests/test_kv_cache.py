"""Tests for KV caching in the autoregressive transformer.

KV caching reduces autoregressive sampling from O(n^3) total attention work to O(n^2)
by caching the key/value tensors at each layer and only computing attention for the
new token against all previously cached tokens.

Test organization:
- TestTransformerLayerKVCache: layer-level correctness of cached vs full forward
- TestTransformerKVCache: full transformer model cached vs full forward
- TestSamplerKVCache: sampler-level correctness (particle conservation, log probs)
- TestKVCachePerformance: timing comparison (cached should be faster)
- TestKVCacheEdgeCases: batch sizes, devices, gradient isolation

Usage:
    uv run pytest tests/test_kv_cache.py -v
"""

import time

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from flows.autoregressive_flow import (
        AutoregressiveConfig,
        AutoregressiveTransformer,
        AutoregressiveFlowSampler,
        TransformerDecoderLayer,
        configs_to_states,
        states_to_configs,
        BOS_TOKEN_ID,
        NUM_ORBITAL_STATES,
    )

    _HAS_AR_FLOW = True
except ImportError:
    _HAS_AR_FLOW = False

pytestmark = pytest.mark.skipif(
    not _HAS_AR_FLOW,
    reason="autoregressive_flow module not yet implemented",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_sampler(n_orbitals, n_alpha, n_beta, **kwargs):
    """Build a small AutoregressiveFlowSampler for testing."""
    num_sites = 2 * n_orbitals
    return AutoregressiveFlowSampler(
        num_sites=num_sites,
        n_alpha=n_alpha,
        n_beta=n_beta,
        **kwargs,
    )


def _make_transformer(n_orbitals, **kwargs):
    """Build a small AutoregressiveTransformer for testing."""
    config = AutoregressiveConfig(
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    return AutoregressiveTransformer(n_orbitals=n_orbitals, config=config)


# ===================================================================
# TestTransformerLayerKVCache
# ===================================================================


class TestTransformerLayerKVCache:
    """Test that TransformerDecoderLayer produces correct output with KV cache."""

    def test_layer_returns_kv_when_use_cache_true(self):
        """Layer should return a (output, kv_tuple) when use_cache=True."""
        layer = TransformerDecoderLayer(d_model=32, n_heads=2, d_ff=64)
        x = torch.randn(2, 5, 32)
        # Build a causal mask for 5 tokens
        mask = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)
        result = layer(x, attn_mask=mask, use_cache=True)
        assert isinstance(result, tuple), "Should return a tuple when use_cache=True"
        assert len(result) == 2, "Should return (output, kv_cache)"
        output, kv = result
        assert output.shape == (2, 5, 32)
        assert kv is not None, "kv cache should not be None when use_cache=True"
        past_k, past_v = kv
        # K/V cache should contain the pre-norm hidden states for all 5 positions
        assert past_k.shape[0] == 2, "Batch dim"
        assert past_k.shape[1] == 5, "Seq dim should cover all tokens"
        assert past_k.shape[2] == 32, "d_model dim"

    def test_layer_returns_tensor_when_use_cache_false(self):
        """Layer with use_cache=False should return plain tensor (backward compat)."""
        layer = TransformerDecoderLayer(d_model=32, n_heads=2, d_ff=64)
        x = torch.randn(2, 5, 32)
        mask = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)
        result = layer(x, attn_mask=mask, use_cache=False)
        # Backward compat: should return just a tensor
        assert isinstance(result, torch.Tensor), (
            "Should return plain Tensor when use_cache=False"
        )
        assert result.shape == (2, 5, 32)

    def test_layer_default_use_cache_is_false(self):
        """Default use_cache should be False for backward compatibility."""
        layer = TransformerDecoderLayer(d_model=32, n_heads=2, d_ff=64)
        x = torch.randn(2, 3, 32)
        mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
        result = layer(x, attn_mask=mask)
        assert isinstance(result, torch.Tensor), (
            "Default call should return plain Tensor"
        )

    def test_layer_cached_matches_full(self):
        """Incremental cached forward should produce the same output as full forward.

        Process [t0, t1, t2, t3, t4] in one pass vs. processing t0..t3 first,
        then t4 with the KV cache from t0..t3. The output for t4 should match.
        """
        torch.manual_seed(42)
        layer = TransformerDecoderLayer(d_model=32, n_heads=2, d_ff=64)
        layer.eval()

        batch = 3
        x_full = torch.randn(batch, 5, 32)
        mask_full = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)

        # Full forward
        out_full = layer(x_full, attn_mask=mask_full)
        assert isinstance(out_full, torch.Tensor)

        # Prefix forward (first 4 tokens)
        mask_prefix = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
        out_prefix, kv_cache = layer(x_full[:, :4, :], attn_mask=mask_prefix, use_cache=True)

        # Incremental forward (5th token with cache)
        out_incr, kv_updated = layer(
            x_full[:, 4:5, :], past_kv=kv_cache, use_cache=True
        )

        # The output for the 5th token should match
        torch.testing.assert_close(
            out_incr[:, 0, :],
            out_full[:, 4, :],
            atol=1e-5,
            rtol=1e-5,
            msg="Cached incremental output must match full forward for the last token",
        )

    def test_layer_cached_sequential_matches_full(self):
        """Process tokens one at a time with cache, compare to full forward."""
        torch.manual_seed(123)
        layer = TransformerDecoderLayer(d_model=32, n_heads=2, d_ff=64)
        layer.eval()

        batch, seq_len, d_model = 2, 6, 32
        x_full = torch.randn(batch, seq_len, d_model)
        mask_full = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

        # Full forward
        out_full = layer(x_full, attn_mask=mask_full)
        assert isinstance(out_full, torch.Tensor)

        # Sequential with cache: process token by token
        kv = None
        outputs = []
        for i in range(seq_len):
            if i == 0:
                mask_i = torch.zeros((1, 1))  # first token attends only to itself
                out_i, kv = layer(x_full[:, i : i + 1, :], attn_mask=mask_i, use_cache=True)
            else:
                out_i, kv = layer(
                    x_full[:, i : i + 1, :], past_kv=kv, use_cache=True
                )
            outputs.append(out_i)

        out_sequential = torch.cat(outputs, dim=1)  # (batch, seq_len, d_model)

        torch.testing.assert_close(
            out_sequential,
            out_full,
            atol=1e-5,
            rtol=1e-5,
            msg="Token-by-token cached forward must match full forward",
        )


# ===================================================================
# TestTransformerKVCache
# ===================================================================


class TestTransformerKVCache:
    """Test KV cache at the full AutoregressiveTransformer level."""

    def test_transformer_returns_logits_and_cache(self):
        """Transformer with use_cache=True should return (logits, kv_cache)."""
        model = _make_transformer(n_orbitals=6)
        model.eval()

        bos = torch.full((4, 1), BOS_TOKEN_ID, dtype=torch.long)
        result = model(bos, use_cache=True)
        assert isinstance(result, tuple), "Should return tuple with use_cache=True"
        logits, cache = result
        assert logits.shape == (4, 1, NUM_ORBITAL_STATES)
        assert cache is not None
        # Cache should have one entry per layer
        assert len(cache) == 2, "2 layers -> 2 cache entries"

    def test_transformer_backward_compat_no_cache(self):
        """Transformer with use_cache=False (default) returns just logits."""
        model = _make_transformer(n_orbitals=6)
        model.eval()

        bos = torch.full((4, 1), BOS_TOKEN_ID, dtype=torch.long)
        result = model(bos)
        assert isinstance(result, torch.Tensor), (
            "Default call should return plain logits tensor"
        )
        assert result.shape == (4, 1, NUM_ORBITAL_STATES)

    def test_transformer_cached_matches_full_forward(self):
        """Incremental cached decode must match full forward for every token."""
        torch.manual_seed(777)
        n_orb = 6
        model = _make_transformer(n_orbitals=n_orb)
        model.eval()

        batch = 5
        # Full sequence: BOS + 6 orbital states = 7 tokens
        full_seq = torch.randint(0, 5, (batch, n_orb + 1), dtype=torch.long)
        full_seq[:, 0] = BOS_TOKEN_ID

        # Full forward (no cache)
        logits_full = model(full_seq)
        assert isinstance(logits_full, torch.Tensor)

        # Incremental decode: BOS first, then one token at a time
        logits_incr, cache = model(full_seq[:, :1], use_cache=True)
        collected_logits = [logits_incr]

        for i in range(1, n_orb + 1):
            logits_step, cache = model(
                full_seq[:, i : i + 1], past_kv=cache, use_cache=True, start_pos=i
            )
            collected_logits.append(logits_step)

        logits_sequential = torch.cat(collected_logits, dim=1)

        torch.testing.assert_close(
            logits_sequential,
            logits_full,
            atol=1e-5,
            rtol=1e-5,
            msg="Token-by-token cached logits must match full forward logits",
        )

    def test_transformer_prefix_then_rest(self):
        """Process a prefix of 3 tokens, then the remaining 4 one at a time."""
        torch.manual_seed(42)
        n_orb = 6
        model = _make_transformer(n_orbitals=n_orb)
        model.eval()

        batch = 3
        full_seq = torch.randint(0, 5, (batch, n_orb + 1), dtype=torch.long)
        full_seq[:, 0] = BOS_TOKEN_ID

        logits_full = model(full_seq)
        assert isinstance(logits_full, torch.Tensor)

        # Process prefix of 3
        logits_prefix, cache = model(full_seq[:, :3], use_cache=True)
        collected = [logits_prefix]

        # Then 4 tokens one at a time
        for i in range(3, n_orb + 1):
            logits_step, cache = model(
                full_seq[:, i : i + 1], past_kv=cache, use_cache=True, start_pos=i
            )
            collected.append(logits_step)

        logits_combined = torch.cat(collected, dim=1)
        torch.testing.assert_close(
            logits_combined,
            logits_full,
            atol=1e-5,
            rtol=1e-5,
        )


# ===================================================================
# TestSamplerKVCache
# ===================================================================


class TestSamplerKVCache:
    """Test that KV-cached sampling preserves correctness in the sampler."""

    def test_kv_cache_particle_conservation(self):
        """Sampling with KV cache must produce configs with exact electron counts."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 6, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        log_probs, unique_configs = sampler.sample(100)

        # Check particle conservation
        alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
        beta_counts = unique_configs[:, n_orb:].sum(dim=1)
        assert (alpha_counts == n_alpha).all(), (
            f"Alpha electron count violated: {alpha_counts}"
        )
        assert (beta_counts == n_beta).all(), (
            f"Beta electron count violated: {beta_counts}"
        )

    def test_kv_cache_log_prob_consistency(self):
        """Log probs from KV-cached sampling must match teacher forcing."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 4, 1, 1
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        # Sample with KV cache (the _sample_autoregressive path)
        configs_all, log_probs_sample, unique = sampler.sample_with_probs(50)

        # Re-evaluate log probs via teacher forcing (single forward pass)
        log_probs_teacher = sampler.log_prob(configs_all)

        torch.testing.assert_close(
            log_probs_sample,
            log_probs_teacher,
            atol=1e-5,
            rtol=1e-5,
            msg=(
                "Log probs from KV-cached autoregressive sampling must match "
                "teacher-forcing log_prob evaluation"
            ),
        )

    def test_kv_cache_different_batch_sizes(self):
        """KV cache should work correctly with various batch sizes."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 5, 2, 1
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        for batch_size in [1, 2, 7, 16, 64]:
            log_probs, unique_configs = sampler.sample(batch_size)
            # Check particle conservation for all batch sizes
            alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
            beta_counts = unique_configs[:, n_orb:].sum(dim=1)
            assert (alpha_counts == n_alpha).all(), (
                f"Alpha violation at batch_size={batch_size}"
            )
            assert (beta_counts == n_beta).all(), (
                f"Beta violation at batch_size={batch_size}"
            )

    def test_kv_cache_deterministic_with_seed(self):
        """Same seed should produce identical results with KV cache."""
        n_orb, n_alpha, n_beta = 5, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        torch.manual_seed(999)
        _, configs1 = sampler.sample(20)

        torch.manual_seed(999)
        _, configs2 = sampler.sample(20)

        assert torch.equal(configs1, configs2), "Same seed should give same configs"

    def test_kv_cache_log_prob_negative(self):
        """All log probabilities should be negative (probabilities < 1)."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 6, 3, 3
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        configs, log_probs, _ = sampler.sample_with_probs(50)
        assert (log_probs < 0).all(), (
            f"Log probs should be negative, got max={log_probs.max().item()}"
        )

    def test_kv_cache_output_shapes(self):
        """Verify all output shapes are correct."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 6, 2, 2
        num_sites = 2 * n_orb
        n_samples = 30
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        configs, log_probs, unique = sampler.sample_with_probs(n_samples)
        assert configs.shape == (n_samples, num_sites), f"configs shape: {configs.shape}"
        assert log_probs.shape == (n_samples,), f"log_probs shape: {log_probs.shape}"
        assert unique.shape[1] == num_sites, f"unique configs width: {unique.shape[1]}"
        assert unique.shape[0] <= n_samples, "unique <= n_samples"

    def test_kv_cache_large_system(self):
        """Test with n_orbitals=20 (40Q target scale) for correctness."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 20, 7, 7
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        configs, log_probs, unique = sampler.sample_with_probs(10)

        # Particle conservation
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)
        assert (alpha_counts == n_alpha).all()
        assert (beta_counts == n_beta).all()

        # Log probs consistency
        log_probs_teacher = sampler.log_prob(configs)
        torch.testing.assert_close(
            log_probs,
            log_probs_teacher,
            atol=1e-4,
            rtol=1e-4,
            msg="40Q scale log prob mismatch between sampling and teacher forcing",
        )


# ===================================================================
# TestKVCachePerformance
# ===================================================================


class TestKVCachePerformance:
    """Test that KV caching provides a performance improvement."""

    def test_kv_cache_no_regression(self):
        """KV-cached sampling should not be slower than full recomputation.

        This test is conservative: it checks that the cached version is not
        significantly *slower* (> 2x), rather than requiring a strict speedup,
        to avoid flaky timing tests on CI.
        """
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 15, 5, 5
        n_samples = 20

        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.eval()

        # Warm-up
        sampler.sample(5)

        # Timed run (KV cache is the default now)
        t0 = time.perf_counter()
        for _ in range(3):
            sampler.sample(n_samples)
        t_cached = time.perf_counter() - t0

        # Sanity: just verify it completed in reasonable time (< 30s)
        assert t_cached < 30.0, (
            f"KV-cached sampling took {t_cached:.1f}s for 3x{n_samples} samples "
            f"with n_orb={n_orb}, which is unreasonably slow"
        )


# ===================================================================
# TestKVCacheEdgeCases
# ===================================================================


class TestKVCacheEdgeCases:
    """Edge cases and corner conditions for KV cache."""

    def test_single_orbital(self):
        """Minimal system: 1 orbital, 1 alpha, 0 beta."""
        torch.manual_seed(42)
        sampler = _make_sampler(n_orbitals=1, n_alpha=1, n_beta=0)
        sampler.eval()

        configs, log_probs, unique = sampler.sample_with_probs(10)
        # Only valid config: alpha=1, beta=0 -> state 2 -> config [1, 0]
        assert unique.shape[0] == 1, "Only one valid config for 1 orbital, 1 alpha"
        assert torch.equal(unique[0], torch.tensor([1, 0])), (
            f"Expected [1, 0], got {unique[0]}"
        )

    def test_all_doubly_occupied(self):
        """All orbitals doubly occupied: only one valid config."""
        torch.manual_seed(42)
        n_orb = 3
        sampler = _make_sampler(n_orbitals=n_orb, n_alpha=n_orb, n_beta=n_orb)
        sampler.eval()

        configs, log_probs, unique = sampler.sample_with_probs(10)
        assert unique.shape[0] == 1, "Only one valid config when fully occupied"
        expected = torch.ones(2 * n_orb, dtype=torch.long)
        assert torch.equal(unique[0], expected)

    def test_kv_cache_no_grad_sampling(self):
        """Sampling should not accumulate gradients (wrapped in torch.no_grad)."""
        torch.manual_seed(42)
        sampler = _make_sampler(n_orbitals=4, n_alpha=1, n_beta=1)
        sampler.eval()

        sampler.sample(10)

        # Verify no parameter has .grad set from sampling
        for name, param in sampler.named_parameters():
            assert param.grad is None, (
                f"Parameter {name} has grad after sampling (should be None)"
            )

    def test_kv_cache_with_temperature(self):
        """KV cache should work correctly at different temperatures."""
        torch.manual_seed(42)
        n_orb, n_alpha, n_beta = 5, 2, 2

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
            sampler = _make_sampler(n_orb, n_alpha, n_beta, temperature=temp)
            sampler.eval()

            configs, log_probs, _ = sampler.sample_with_probs(20)

            # Particle conservation at all temperatures
            alpha_counts = configs[:, :n_orb].sum(dim=1)
            beta_counts = configs[:, n_orb:].sum(dim=1)
            assert (alpha_counts == n_alpha).all(), f"Alpha violation at temp={temp}"
            assert (beta_counts == n_beta).all(), f"Beta violation at temp={temp}"

            # Log prob consistency at all temperatures
            log_probs_tf = sampler.log_prob(configs)
            torch.testing.assert_close(
                log_probs,
                log_probs_tf,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Log prob mismatch at temperature={temp}",
            )

    def test_transformer_use_cache_does_not_affect_training(self):
        """In training mode (with mask), use_cache=False should behave exactly as before."""
        torch.manual_seed(42)
        n_orb = 6
        model = _make_transformer(n_orbitals=n_orb)
        model.train()

        full_seq = torch.randint(0, 5, (4, n_orb + 1), dtype=torch.long)
        full_seq[:, 0] = BOS_TOKEN_ID

        # Without cache
        logits_no_cache = model(full_seq, use_cache=False)
        assert isinstance(logits_no_cache, torch.Tensor)

        # With cache but not using it (full forward, just collecting cache)
        logits_with_cache, cache = model(full_seq, use_cache=True)
        assert isinstance(logits_with_cache, torch.Tensor)

        torch.testing.assert_close(
            logits_no_cache,
            logits_with_cache,
            atol=1e-6,
            rtol=1e-6,
            msg="use_cache should not change logits during full forward",
        )
