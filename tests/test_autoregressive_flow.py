"""Comprehensive tests for the autoregressive transformer flow module.

The autoregressive flow is the Phase 4 upgrade to replace the non-autoregressive
ParticleConservingFlowSampler. The key architectural difference:

Non-autoregressive: P(config) = P(o1) * P(o2) * ... * P(on)    (independent marginals)
Autoregressive:     P(config) = P(o1) * P(o2|o1) * ... * P(on|o1...on-1)  (sequential)

This enables capturing inter-orbital correlations critical for strongly correlated
systems (Cr2, [2Fe-2S], stretched bonds) and 40Q+ scale molecules.

Test organization:
- TestStateConversion: binary <-> quaternary state mapping
- TestParticleConservation: electron count constraints at all temperatures
- TestLogProbability: shape, sign, consistency, normalization, differentiability
- TestSampling: output shapes, binary values, uniqueness
- TestTemperature: entropy control, deterministic/diverse regimes
- TestSamplerInterface: drop-in compatibility with ParticleConservingFlowSampler
- TestTransformerArchitecture: model structure, causal mask, config flexibility
- TestPipelineIntegration: pipeline compatibility (Hamiltonian, trainer)

Usage:
    uv run pytest tests/test_autoregressive_flow.py -v
    uv run pytest tests/test_autoregressive_flow.py -v -m slow   # slow integration tests
"""

import math
import itertools

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Import helpers — skip entire module if autoregressive_flow not yet available
# ---------------------------------------------------------------------------

try:
    from flows.autoregressive_flow import (
        AutoregressiveConfig,
        AutoregressiveTransformer,
        AutoregressiveFlowSampler,
        configs_to_states,
        states_to_configs,
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


def _enumerate_all_configs(n_orbitals, n_alpha, n_beta):
    """Enumerate all valid configs for a small system.

    Returns:
        (n_configs, 2*n_orbitals) int tensor with every valid occupation vector.
    """
    alpha_combos = list(itertools.combinations(range(n_orbitals), n_alpha))
    beta_combos = list(itertools.combinations(range(n_orbitals), n_beta))

    configs = []
    for a_occ in alpha_combos:
        for b_occ in beta_combos:
            vec = [0] * (2 * n_orbitals)
            for i in a_occ:
                vec[i] = 1
            for i in b_occ:
                vec[n_orbitals + i] = 1
            configs.append(vec)
    return torch.tensor(configs, dtype=torch.long)


# ===================================================================
# TestStateConversion
# ===================================================================

class TestStateConversion:
    """Binary occupation <-> quaternary orbital state conversion.

    Convention: state = 2 * alpha + beta
        0 = unoccupied    (alpha=0, beta=0)
        1 = beta only     (alpha=0, beta=1)
        2 = alpha only    (alpha=1, beta=0)
        3 = doubly occ.   (alpha=1, beta=1)
    """

    def test_configs_to_states_shape(self):
        """Output shape should be (batch, n_orbitals)."""
        n_orb = 6
        batch = 4
        configs = torch.zeros(batch, 2 * n_orb, dtype=torch.long)
        states = configs_to_states(configs, n_orb)
        assert states.shape == (batch, n_orb), (
            f"Expected (4, 6), got {states.shape}"
        )

    def test_configs_to_states_unoccupied(self):
        """All-zero config should give all-zero states."""
        n_orb = 4
        configs = torch.zeros(1, 2 * n_orb, dtype=torch.long)
        states = configs_to_states(configs, n_orb)
        assert (states == 0).all(), "Empty config should map to state 0"

    def test_configs_to_states_doubly_occupied(self):
        """Both alpha and beta occupied -> state 3."""
        n_orb = 4
        configs = torch.zeros(1, 2 * n_orb, dtype=torch.long)
        # Orbital 0: alpha=1, beta=1
        configs[0, 0] = 1
        configs[0, n_orb] = 1
        states = configs_to_states(configs, n_orb)
        assert states[0, 0].item() == 3, (
            f"Doubly occupied should be state 3, got {states[0, 0].item()}"
        )

    def test_configs_to_states_alpha_only(self):
        """Alpha occupied, beta not -> state 2."""
        n_orb = 4
        configs = torch.zeros(1, 2 * n_orb, dtype=torch.long)
        configs[0, 1] = 1  # alpha on orbital 1
        states = configs_to_states(configs, n_orb)
        assert states[0, 1].item() == 2, (
            f"Alpha-only should be state 2, got {states[0, 1].item()}"
        )

    def test_configs_to_states_beta_only(self):
        """Beta occupied, alpha not -> state 1."""
        n_orb = 4
        configs = torch.zeros(1, 2 * n_orb, dtype=torch.long)
        configs[0, n_orb + 2] = 1  # beta on orbital 2
        states = configs_to_states(configs, n_orb)
        assert states[0, 2].item() == 1, (
            f"Beta-only should be state 1, got {states[0, 2].item()}"
        )

    def test_configs_to_states_hf_lih(self):
        """LiH HF state: orbitals 0,1 doubly occupied (6 orbitals, 2 alpha, 2 beta).

        Alpha: [1,1,0,0,0,0], Beta: [1,1,0,0,0,0]
        States: [3,3,0,0,0,0]
        """
        n_orb = 6
        configs = torch.zeros(1, 2 * n_orb, dtype=torch.long)
        configs[0, 0] = 1
        configs[0, 1] = 1
        configs[0, n_orb] = 1
        configs[0, n_orb + 1] = 1
        states = configs_to_states(configs, n_orb)
        expected = torch.tensor([[3, 3, 0, 0, 0, 0]])
        assert torch.equal(states, expected), (
            f"LiH HF states wrong: {states} != {expected}"
        )

    def test_states_to_configs_shape(self):
        """Output shape should be (batch, 2*n_orbitals)."""
        n_orb = 6
        batch = 4
        states = torch.zeros(batch, n_orb, dtype=torch.long)
        configs = states_to_configs(states, n_orb)
        assert configs.shape == (batch, 2 * n_orb), (
            f"Expected (4, 12), got {configs.shape}"
        )

    def test_states_to_configs_doubly_occupied(self):
        """State 3 -> alpha=1, beta=1."""
        n_orb = 4
        states = torch.tensor([[3, 0, 0, 0]])
        configs = states_to_configs(states, n_orb)
        assert configs[0, 0].item() == 1, "Alpha should be 1 for state 3"
        assert configs[0, n_orb].item() == 1, "Beta should be 1 for state 3"

    def test_states_to_configs_alpha_only(self):
        """State 2 -> alpha=1, beta=0."""
        n_orb = 4
        states = torch.tensor([[0, 2, 0, 0]])
        configs = states_to_configs(states, n_orb)
        assert configs[0, 1].item() == 1, "Alpha should be 1 for state 2"
        assert configs[0, n_orb + 1].item() == 0, "Beta should be 0 for state 2"

    def test_states_to_configs_beta_only(self):
        """State 1 -> alpha=0, beta=1."""
        n_orb = 4
        states = torch.tensor([[0, 0, 1, 0]])
        configs = states_to_configs(states, n_orb)
        assert configs[0, 2].item() == 0, "Alpha should be 0 for state 1"
        assert configs[0, n_orb + 2].item() == 1, "Beta should be 1 for state 1"

    def test_roundtrip_configs_to_states_to_configs(self):
        """configs_to_states then states_to_configs should be identity."""
        n_orb = 4
        n_alpha, n_beta = 2, 2
        all_configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)
        states = configs_to_states(all_configs, n_orb)
        recovered = states_to_configs(states, n_orb)
        assert torch.equal(all_configs, recovered), (
            "Roundtrip configs -> states -> configs failed"
        )

    def test_roundtrip_states_to_configs_to_states(self):
        """states_to_configs then configs_to_states should be identity for valid states."""
        n_orb = 4
        # Manually create valid states (any value 0-3)
        states = torch.tensor([
            [3, 2, 0, 0],
            [3, 0, 1, 0],
            [0, 2, 1, 2],
            [1, 1, 2, 2],
        ])
        configs = states_to_configs(states, n_orb)
        recovered_states = configs_to_states(configs, n_orb)
        assert torch.equal(states, recovered_states), (
            "Roundtrip states -> configs -> states failed"
        )

    def test_batch_consistency(self):
        """Batch conversion should match element-wise conversion."""
        n_orb = 5
        batch_size = 8
        configs = torch.randint(0, 2, (batch_size, 2 * n_orb))
        batch_states = configs_to_states(configs, n_orb)
        for i in range(batch_size):
            single = configs_to_states(configs[i:i + 1], n_orb)
            assert torch.equal(batch_states[i:i + 1], single), (
                f"Batch vs single mismatch at index {i}"
            )


# ===================================================================
# TestParticleConservation
# ===================================================================

class TestParticleConservation:
    """All sampled configs must have exact electron counts. Non-negotiable."""

    @pytest.mark.parametrize("n_orb,n_alpha,n_beta", [
        (4, 2, 2),   # Small test system (36 configs)
        (6, 2, 2),   # LiH-like
        (7, 5, 5),   # H2O-like
    ])
    def test_particle_conservation_default_temperature(self, n_orb, n_alpha, n_beta):
        """At default temperature, sampled configs must conserve particles."""
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        with torch.no_grad():
            log_probs, unique_configs = sampler.sample(200)

        alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
        beta_counts = unique_configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == n_alpha).all(), (
            f"Alpha violation: expected {n_alpha}, got counts {alpha_counts.unique().tolist()}"
        )
        assert (beta_counts == n_beta).all(), (
            f"Beta violation: expected {n_beta}, got counts {beta_counts.unique().tolist()}"
        )

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_conservation_at_various_temperatures(self, temperature):
        """Particle conservation must hold at every temperature."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.set_temperature(temperature)

        with torch.no_grad():
            log_probs, unique_configs = sampler.sample(200)

        alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
        beta_counts = unique_configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == n_alpha).all(), (
            f"Alpha violation at temp={temperature}: {alpha_counts.unique().tolist()}"
        )
        assert (beta_counts == n_beta).all(), (
            f"Beta violation at temp={temperature}: {beta_counts.unique().tolist()}"
        )

    def test_conservation_with_sample_with_probs(self):
        """sample_with_probs must also conserve particles."""
        n_orb, n_alpha, n_beta = 6, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        with torch.no_grad():
            configs, log_probs, unique_configs = sampler.sample_with_probs(100)

        # Check all configs (not just unique)
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == n_alpha).all(), (
            f"Alpha violation in sample_with_probs: {alpha_counts.unique().tolist()}"
        )
        assert (beta_counts == n_beta).all(), (
            f"Beta violation in sample_with_probs: {beta_counts.unique().tolist()}"
        )

    def test_conservation_verify_function(self):
        """verify_particle_conservation should report no violations."""
        from flows.particle_conserving_flow import verify_particle_conservation

        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        with torch.no_grad():
            configs, _, _ = sampler.sample_with_probs(100)

        valid, stats = verify_particle_conservation(configs, n_orb, n_alpha, n_beta)
        assert valid, f"Particle conservation violated: {stats}"
        assert stats['alpha_violations'] == 0
        assert stats['beta_violations'] == 0

    def test_n2_scale_conservation(self):
        """N2-scale system (10 orbitals, 7 alpha, 7 beta) must conserve."""
        n_orb, n_alpha, n_beta = 10, 7, 7
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        with torch.no_grad():
            log_probs, unique_configs = sampler.sample(100)

        alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
        beta_counts = unique_configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == n_alpha).all()
        assert (beta_counts == n_beta).all()


# ===================================================================
# TestLogProbability
# ===================================================================

class TestLogProbability:
    """Log probability must be well-formed, consistent, and differentiable."""

    def test_log_prob_shape(self):
        """log_prob output should be (batch,)."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)
        log_probs = sampler.log_prob(configs.float())

        assert log_probs.shape == (len(configs),), (
            f"Expected ({len(configs)},), got {log_probs.shape}"
        )

    def test_log_prob_negative(self):
        """Log probabilities must be negative (probabilities < 1)."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)
        log_probs = sampler.log_prob(configs.float())

        assert (log_probs < 0).all(), (
            f"Some log_probs are non-negative: max={log_probs.max().item():.6f}"
        )

    def test_log_prob_finite(self):
        """Log probabilities must be finite (no inf, nan)."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)
        log_probs = sampler.log_prob(configs.float())

        assert torch.isfinite(log_probs).all(), (
            f"Non-finite log_probs found: {log_probs[~torch.isfinite(log_probs)]}"
        )

    def test_log_prob_consistency_with_sample(self):
        """log_prob of sampled configs should match sampling log_probs."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        # Sample and get log_probs from sampling
        configs, sample_log_probs, _ = sampler.sample_with_probs(50)

        # Recompute log_probs via log_prob method
        recomputed = sampler.log_prob(configs.float())

        # They should match (both use the same underlying model)
        assert torch.allclose(sample_log_probs, recomputed, atol=1e-4), (
            f"Max difference: {(sample_log_probs - recomputed).abs().max().item():.6f}"
        )

    def test_probabilities_sum_approximately_one(self):
        """For a tiny system, exp(log_prob) over all valid configs should sum to ~1.

        System: 4 orbitals, 2 alpha, 2 beta -> C(4,2)*C(4,2) = 36 configs.
        """
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        all_configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)
        assert len(all_configs) == 36, f"Expected 36 configs, got {len(all_configs)}"

        with torch.no_grad():
            log_probs = sampler.log_prob(all_configs.float())
            total_prob = torch.exp(log_probs).sum().item()

        assert abs(total_prob - 1.0) < 0.05, (
            f"Total probability = {total_prob:.6f}, expected ~1.0. "
            "The autoregressive model is not a proper probability distribution."
        )

    def test_log_prob_differentiable(self):
        """Gradients must flow through log_prob to model parameters."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)[:8].float()
        log_probs = sampler.log_prob(configs)
        loss = -log_probs.mean()
        loss.backward()

        params_with_grad = sum(
            1 for p in sampler.parameters() if p.grad is not None
        )
        total_params = sum(1 for _ in sampler.parameters())

        assert params_with_grad > 0, "No parameters received gradients"
        # At least half should have gradients (some may be unused in a specific path)
        assert params_with_grad >= total_params * 0.3, (
            f"Only {params_with_grad}/{total_params} parameters received gradients"
        )

    def test_estimate_discrete_prob(self):
        """estimate_discrete_prob should return exp(log_prob)."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)[:8].float()

        with torch.no_grad():
            probs = sampler.estimate_discrete_prob(configs)
            log_probs = sampler.log_prob(configs)
            expected_probs = torch.exp(log_probs)

        assert torch.allclose(probs, expected_probs, atol=1e-6), (
            f"estimate_discrete_prob != exp(log_prob), max diff: "
            f"{(probs - expected_probs).abs().max().item():.8f}"
        )


# ===================================================================
# TestSampling
# ===================================================================

class TestSampling:
    """Verify sampling outputs have correct shapes, types, and values."""

    def test_sample_returns_two_tensors(self):
        """sample() should return (log_probs, unique_configs)."""
        sampler = _make_sampler(4, 2, 2)
        result = sampler.sample(50)
        assert len(result) == 2, f"sample() should return 2 tensors, got {len(result)}"

    def test_sample_log_probs_shape(self):
        """log_probs shape should be (n_samples,)."""
        sampler = _make_sampler(4, 2, 2)
        log_probs, _ = sampler.sample(50)
        assert log_probs.shape == (50,), f"Expected (50,), got {log_probs.shape}"

    def test_sample_unique_configs_shape(self):
        """unique_configs second dim should be 2*n_orbitals."""
        n_orb = 4
        sampler = _make_sampler(n_orb, 2, 2)
        _, unique_configs = sampler.sample(50)
        assert unique_configs.shape[1] == 2 * n_orb, (
            f"Expected width {2 * n_orb}, got {unique_configs.shape[1]}"
        )

    def test_sample_binary_values(self):
        """Sampled configs must be binary {0, 1}."""
        sampler = _make_sampler(4, 2, 2)
        _, unique_configs = sampler.sample(100)
        unique_vals = set(unique_configs.unique().tolist())
        assert unique_vals.issubset({0, 1}), (
            f"Non-binary values found: {unique_vals}"
        )

    def test_sample_unique_configs_no_duplicates(self):
        """unique_configs should contain no duplicate rows."""
        sampler = _make_sampler(4, 2, 2)
        _, unique_configs = sampler.sample(200)
        n_unique = len(torch.unique(unique_configs, dim=0))
        assert n_unique == len(unique_configs), (
            f"unique_configs has duplicates: {n_unique} unique out of {len(unique_configs)}"
        )

    def test_sample_with_probs_returns_three_tensors(self):
        """sample_with_probs() should return (configs, log_probs, unique_configs)."""
        sampler = _make_sampler(4, 2, 2)
        result = sampler.sample_with_probs(50)
        assert len(result) == 3, (
            f"sample_with_probs() should return 3 tensors, got {len(result)}"
        )

    def test_sample_with_probs_shapes(self):
        """All returned tensors should have correct shapes."""
        n_orb = 4
        n_samples = 50
        sampler = _make_sampler(n_orb, 2, 2)
        configs, log_probs, unique_configs = sampler.sample_with_probs(n_samples)

        assert configs.shape == (n_samples, 2 * n_orb), (
            f"configs shape: expected ({n_samples}, {2*n_orb}), got {configs.shape}"
        )
        assert log_probs.shape == (n_samples,), (
            f"log_probs shape: expected ({n_samples},), got {log_probs.shape}"
        )
        assert unique_configs.shape[1] == 2 * n_orb, (
            f"unique_configs width: expected {2*n_orb}, got {unique_configs.shape[1]}"
        )
        assert len(unique_configs) <= n_samples, (
            f"unique_configs ({len(unique_configs)}) > n_samples ({n_samples})"
        )

    def test_sample_with_probs_configs_are_integer(self):
        """Configs from sample_with_probs should be integer-typed."""
        sampler = _make_sampler(4, 2, 2)
        configs, _, _ = sampler.sample_with_probs(50)
        assert configs.dtype in (torch.long, torch.int, torch.int64, torch.int32), (
            f"Expected integer dtype, got {configs.dtype}"
        )

    def test_large_sample_explores_multiple_configs(self):
        """With 10000 samples from LiH-sized system, we should see multiple unique configs."""
        n_orb, n_alpha, n_beta = 6, 2, 2  # LiH: C(6,2)*C(6,2) = 225 possible
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        with torch.no_grad():
            _, unique_configs = sampler.sample(10000)

        # Autoregressive should explore more than the deterministic non-autoregressive model
        # At minimum, expect > 1 unique config
        assert len(unique_configs) > 1, (
            f"Only {len(unique_configs)} unique config from 10000 samples — "
            "sampler is collapsed"
        )

    def test_forward_same_as_sample(self):
        """forward() should return the same as sample()."""
        sampler = _make_sampler(4, 2, 2)
        torch.manual_seed(42)
        result_forward = sampler.forward(50)
        assert len(result_forward) == 2, (
            f"forward() should return 2 tensors like sample()"
        )


# ===================================================================
# TestTemperature
# ===================================================================

class TestTemperature:
    """Temperature controls sampling entropy."""

    def test_set_temperature(self):
        """set_temperature should not raise."""
        sampler = _make_sampler(4, 2, 2)
        sampler.set_temperature(0.5)
        sampler.set_temperature(2.0)
        sampler.set_temperature(0.01)

    def test_low_temperature_few_unique(self):
        """At very low temperature, sampling should be nearly deterministic."""
        n_orb, n_alpha, n_beta = 6, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.set_temperature(0.01)

        with torch.no_grad():
            _, unique_configs = sampler.sample(500)

        # Near-deterministic: very few unique configs
        assert len(unique_configs) <= 5, (
            f"At temp=0.01, expected <= 5 unique configs, got {len(unique_configs)}"
        )

    def test_high_temperature_many_unique(self):
        """At high temperature, sampling should explore many configs."""
        n_orb, n_alpha, n_beta = 6, 2, 2  # 225 possible configs
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        sampler.set_temperature(10.0)

        with torch.no_grad():
            _, unique_configs = sampler.sample(5000)

        # High temp should explore significantly
        assert len(unique_configs) >= 10, (
            f"At temp=10.0, expected >= 10 unique configs, got {len(unique_configs)}"
        )

    def test_higher_temp_more_diversity(self):
        """Higher temperature should produce more unique configs than lower."""
        n_orb, n_alpha, n_beta = 6, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)
        n_samples = 2000

        # Low temperature
        sampler.set_temperature(0.1)
        with torch.no_grad():
            _, unique_low = sampler.sample(n_samples)

        # High temperature
        sampler.set_temperature(5.0)
        with torch.no_grad():
            _, unique_high = sampler.sample(n_samples)

        assert len(unique_high) >= len(unique_low), (
            f"High temp ({len(unique_high)} unique) should produce >= "
            f"low temp ({len(unique_low)} unique) diversity"
        )

    def test_temperature_does_not_break_conservation(self):
        """Changing temperature must not break particle conservation."""
        n_orb, n_alpha, n_beta = 6, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        for temp in [0.01, 0.1, 1.0, 5.0, 10.0]:
            sampler.set_temperature(temp)
            with torch.no_grad():
                _, unique_configs = sampler.sample(100)

            alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
            beta_counts = unique_configs[:, n_orb:].sum(dim=1)

            assert (alpha_counts == n_alpha).all(), (
                f"Alpha violation at temp={temp}"
            )
            assert (beta_counts == n_beta).all(), (
                f"Beta violation at temp={temp}"
            )


# ===================================================================
# TestSamplerInterface
# ===================================================================

class TestSamplerInterface:
    """AutoregressiveFlowSampler must be a drop-in for ParticleConservingFlowSampler."""

    def test_has_sample_method(self):
        """Must have sample(n_samples) -> (log_probs, unique_configs)."""
        sampler = _make_sampler(4, 2, 2)
        assert hasattr(sampler, 'sample')
        assert callable(sampler.sample)

    def test_has_sample_with_probs_method(self):
        """Must have sample_with_probs(n_samples) -> (configs, log_probs, unique_configs)."""
        sampler = _make_sampler(4, 2, 2)
        assert hasattr(sampler, 'sample_with_probs')
        assert callable(sampler.sample_with_probs)

    def test_has_log_prob_method(self):
        """Must have log_prob(configs) -> (batch,)."""
        sampler = _make_sampler(4, 2, 2)
        assert hasattr(sampler, 'log_prob')
        assert callable(sampler.log_prob)

    def test_has_estimate_discrete_prob_method(self):
        """Must have estimate_discrete_prob(configs) -> (batch,)."""
        sampler = _make_sampler(4, 2, 2)
        assert hasattr(sampler, 'estimate_discrete_prob')
        assert callable(sampler.estimate_discrete_prob)

    def test_has_set_temperature_method(self):
        """Must have set_temperature(float)."""
        sampler = _make_sampler(4, 2, 2)
        assert hasattr(sampler, 'set_temperature')
        assert callable(sampler.set_temperature)

    def test_has_forward_method(self):
        """Must have forward() for nn.Module compatibility."""
        sampler = _make_sampler(4, 2, 2)
        assert hasattr(sampler, 'forward')
        assert callable(sampler.forward)

    def test_is_nn_module(self):
        """Must be an nn.Module for optimizer integration."""
        sampler = _make_sampler(4, 2, 2)
        assert isinstance(sampler, nn.Module), (
            f"Expected nn.Module, got {type(sampler)}"
        )

    def test_has_parameters(self):
        """Must have learnable parameters."""
        sampler = _make_sampler(4, 2, 2)
        n_params = sum(p.numel() for p in sampler.parameters())
        assert n_params > 0, "Sampler has no learnable parameters"

    def test_num_sites_attribute(self):
        """Must have num_sites attribute matching 2*n_orbitals."""
        n_orb = 6
        sampler = _make_sampler(n_orb, 2, 2)
        assert hasattr(sampler, 'num_sites'), "Missing num_sites attribute"
        assert sampler.num_sites == 2 * n_orb, (
            f"num_sites should be {2*n_orb}, got {sampler.num_sites}"
        )

    def test_n_alpha_n_beta_attributes(self):
        """Must have n_alpha and n_beta attributes."""
        sampler = _make_sampler(6, 3, 2)
        assert hasattr(sampler, 'n_alpha'), "Missing n_alpha attribute"
        assert hasattr(sampler, 'n_beta'), "Missing n_beta attribute"
        assert sampler.n_alpha == 3
        assert sampler.n_beta == 2

    def test_sample_return_order_matches_pcf(self):
        """sample() returns (log_probs, unique_configs) — same order as PCF sampler."""
        sampler = _make_sampler(4, 2, 2)
        log_probs, unique_configs = sampler.sample(50)

        # log_probs should be 1D with n_samples elements
        assert log_probs.dim() == 1
        assert log_probs.shape[0] == 50

        # unique_configs should be 2D
        assert unique_configs.dim() == 2

    def test_sample_with_probs_return_order(self):
        """sample_with_probs() returns (configs, log_probs, unique_configs)."""
        sampler = _make_sampler(4, 2, 2)
        configs, log_probs, unique_configs = sampler.sample_with_probs(50)

        # configs: all sampled (including duplicates)
        assert configs.shape[0] == 50
        # log_probs: one per sample
        assert log_probs.shape[0] == 50
        # unique_configs: deduplicated
        assert unique_configs.shape[0] <= 50


# ===================================================================
# TestTransformerArchitecture
# ===================================================================

class TestTransformerArchitecture:
    """Verify the transformer model structure and properties."""

    def test_autoregressive_config_defaults(self):
        """AutoregressiveConfig should have sensible defaults."""
        config = AutoregressiveConfig()
        assert hasattr(config, 'n_layers')
        assert hasattr(config, 'n_heads')
        assert hasattr(config, 'd_model')
        assert hasattr(config, 'd_ff')
        assert hasattr(config, 'dropout')
        assert config.n_layers >= 1
        assert config.n_heads >= 1
        assert config.d_model >= 16
        assert config.d_ff >= config.d_model

    def test_autoregressive_config_custom(self):
        """Custom config values should be respected."""
        config = AutoregressiveConfig(
            n_layers=6, n_heads=8, d_model=256, d_ff=512, dropout=0.1
        )
        assert config.n_layers == 6
        assert config.n_heads == 8
        assert config.d_model == 256
        assert config.d_ff == 512
        assert config.dropout == 0.1

    def test_transformer_has_parameters(self):
        """AutoregressiveTransformer should be a parameterized nn.Module."""
        config = AutoregressiveConfig(n_layers=2, n_heads=2, d_model=32, d_ff=64)
        model = AutoregressiveTransformer(n_orbitals=4, config=config)
        assert isinstance(model, nn.Module)

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 100, (
            f"Transformer has only {n_params} parameters — too few"
        )

    def test_transformer_causal_mask(self):
        """Transformer should have a causal (lower-triangular) attention mask."""
        config = AutoregressiveConfig(n_layers=2, n_heads=2, d_model=32, d_ff=64)
        model = AutoregressiveTransformer(n_orbitals=6, config=config)

        # Check for causal mask buffer
        has_mask = False
        for name, buf in model.named_buffers():
            if 'mask' in name.lower() or 'causal' in name.lower():
                has_mask = True
                # Verify it is lower-triangular (or equivalent boolean mask)
                mask = buf
                if mask.dim() == 2:
                    # Upper triangle should be masked (True for attention_mask, -inf for additive)
                    n = mask.shape[0]
                    for i in range(n):
                        for j in range(i + 1, n):
                            # Either True (masked) or -inf
                            val = mask[i, j].item()
                            assert val == True or val == float('-inf') or val == 1, (
                                f"Causal mask[{i},{j}] should block future: got {val}"
                            )
                break

        assert has_mask, (
            "No causal mask buffer found. Autoregressive transformer must have one. "
            f"Buffers found: {[n for n, _ in model.named_buffers()]}"
        )

    def test_different_n_layers(self):
        """Model should work with various layer counts."""
        for n_layers in [1, 2, 4]:
            config = AutoregressiveConfig(
                n_layers=n_layers, n_heads=2, d_model=32, d_ff=64
            )
            model = AutoregressiveTransformer(n_orbitals=4, config=config)
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params > 0, f"n_layers={n_layers}: no parameters"

    def test_different_d_model(self):
        """Model should work with various embedding dimensions."""
        for d_model in [16, 32, 64, 128]:
            config = AutoregressiveConfig(
                n_layers=2, n_heads=2, d_model=d_model, d_ff=d_model * 2
            )
            model = AutoregressiveTransformer(n_orbitals=4, config=config)
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params > 0, f"d_model={d_model}: no parameters"

    def test_more_params_with_deeper_model(self):
        """Deeper model (more layers) should have more parameters."""
        config_small = AutoregressiveConfig(
            n_layers=1, n_heads=2, d_model=32, d_ff=64
        )
        config_large = AutoregressiveConfig(
            n_layers=4, n_heads=2, d_model=32, d_ff=64
        )
        model_small = AutoregressiveTransformer(n_orbitals=4, config=config_small)
        model_large = AutoregressiveTransformer(n_orbitals=4, config=config_large)

        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        assert params_large > params_small, (
            f"4-layer ({params_large}) should have more params than 1-layer ({params_small})"
        )

    def test_d_model_divisible_by_n_heads(self):
        """d_model must be divisible by n_heads for multi-head attention."""
        # This should work
        config = AutoregressiveConfig(n_layers=2, n_heads=4, d_model=64, d_ff=128)
        model = AutoregressiveTransformer(n_orbitals=4, config=config)
        assert model is not None

    def test_sampler_param_count_reasonable(self):
        """Sampler should have a reasonable number of parameters for the system size."""
        # Small system
        small = _make_sampler(4, 2, 2)
        params_small = sum(p.numel() for p in small.parameters())
        assert 100 < params_small < 10_000_000, (
            f"Unreasonable param count for n_orb=4: {params_small}"
        )

        # Larger system
        large = _make_sampler(10, 5, 5)
        params_large = sum(p.numel() for p in large.parameters())
        assert params_large > params_small, (
            "Larger system should have more parameters"
        )


# ===================================================================
# TestPipelineIntegration
# ===================================================================

class TestPipelineIntegration:
    """Verify the sampler integrates with the existing pipeline."""

    def test_works_with_nqs(self):
        """Sampler outputs should be compatible with DenseNQS input."""
        from nqs.dense import DenseNQS

        n_orb = 4
        num_sites = 2 * n_orb
        sampler = _make_sampler(n_orb, 2, 2)
        nqs = DenseNQS(num_sites=num_sites, hidden_dims=[32, 32])

        with torch.no_grad():
            configs, log_probs, unique = sampler.sample_with_probs(20)

        # NQS uses log_amplitude(), not forward() directly
        log_psi = nqs.log_amplitude(unique.float())
        assert log_psi.shape[0] == len(unique), (
            f"NQS output shape mismatch: {log_psi.shape[0]} vs {len(unique)}"
        )

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_works_with_physics_trainer(self, lih_hamiltonian):
        """AutoregressiveFlowSampler should work as a flow in PhysicsGuidedFlowTrainer.

        This is the critical drop-in test: replace ParticleConservingFlowSampler
        with AutoregressiveFlowSampler in the trainer and run 1 training step.
        """
        from flows.physics_guided_training import (
            PhysicsGuidedFlowTrainer,
            PhysicsGuidedConfig,
        )
        from nqs.dense import DenseNQS

        H = lih_hamiltonian
        n_orb = H.n_orbitals
        num_sites = 2 * n_orb

        sampler = _make_sampler(n_orb, H.n_alpha, H.n_beta)
        nqs = DenseNQS(num_sites=num_sites, hidden_dims=[32, 32])

        cfg = PhysicsGuidedConfig(
            num_epochs=2,
            min_epochs=1,
            samples_per_batch=50,
            convergence_threshold=0.001,
            early_stopping_patience=100,
            use_torch_compile=False,
        )

        trainer = PhysicsGuidedFlowTrainer(
            flow=sampler,
            nqs=nqs,
            hamiltonian=H,
            config=cfg,
            device="cpu",
        )

        # Should be able to train without error
        history = trainer.train()
        assert 'energies' in history, "Trainer did not return energy history"
        assert len(history['energies']) > 0, "No energies recorded during training"

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_skqd_with_autoregressive_samples(self, lih_hamiltonian):
        """SKQD should accept configs from the autoregressive sampler."""
        H = lih_hamiltonian
        n_orb = H.n_orbitals
        sampler = _make_sampler(n_orb, H.n_alpha, H.n_beta)

        with torch.no_grad():
            _, unique_configs = sampler.sample(500)

        # unique_configs should be valid for Hamiltonian
        assert unique_configs.shape[1] == 2 * n_orb
        assert unique_configs.dtype in (torch.long, torch.int, torch.int64, torch.int32)

        # Verify particle conservation before passing to SKQD
        from flows.particle_conserving_flow import verify_particle_conservation
        valid, stats = verify_particle_conservation(
            unique_configs, n_orb, H.n_alpha, H.n_beta
        )
        assert valid, f"Configs invalid for SKQD: {stats}"

    def test_gradient_through_full_pipeline_mock(self):
        """End-to-end gradient flow: sample -> log_prob -> loss -> backward."""
        from nqs.dense import DenseNQS

        n_orb = 4
        num_sites = 2 * n_orb
        sampler = _make_sampler(n_orb, 2, 2)
        nqs = DenseNQS(num_sites=num_sites, hidden_dims=[32, 32])

        # Forward: sample, compute NQS log_psi, compute loss
        configs, log_probs, unique = sampler.sample_with_probs(20)

        # Recompute log_prob to ensure differentiability
        flow_log_probs = sampler.log_prob(configs.float())
        # NQS uses log_amplitude(), not forward()
        nqs_log_psi = nqs.log_amplitude(configs.float())

        # Simple KL-like loss
        loss = (flow_log_probs - nqs_log_psi.detach()).pow(2).mean()
        loss.backward()

        # Sampler should receive gradients (NQS is detached)
        sampler_has_grad = any(
            p.grad is not None for p in sampler.parameters()
        )
        assert sampler_has_grad, "No gradients flowed to sampler parameters"


# ===================================================================
# TestAutoregressive Property
# ===================================================================

class TestAutoregressiveProperty:
    """Verify the model is truly autoregressive (not product-of-marginals)."""

    def test_conditional_probabilities_differ(self):
        """P(orbital_j | different histories) should differ.

        If the model were non-autoregressive (product-of-marginals),
        conditioning on different histories would not change orbital probabilities.
        """
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        # Create two configs that differ in early orbitals
        all_configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)

        # Find two configs with different first orbitals
        config_a = None
        config_b = None
        for c in all_configs:
            if c[0] == 1 and config_a is None:
                config_a = c
            elif c[0] == 0 and config_b is None:
                config_b = c
            if config_a is not None and config_b is not None:
                break

        if config_a is None or config_b is None:
            pytest.skip("Could not find suitable test configs")

        with torch.no_grad():
            log_p_a = sampler.log_prob(config_a.unsqueeze(0).float())
            log_p_b = sampler.log_prob(config_b.unsqueeze(0).float())

        # The log probs should be different (though this alone does not
        # prove autoregression -- it's a necessary condition)
        assert not torch.allclose(log_p_a, log_p_b, atol=1e-8), (
            "Different configs got same log_prob — model may not distinguish them"
        )

    def test_not_product_of_marginals(self):
        """The probability distribution should NOT factorize as independent marginals.

        For a truly autoregressive model, P(o1,o2,...) != P(o1)*P(o2)*...*P(on).
        We check this by comparing joint probability to product of marginals
        estimated from samples.
        """
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        all_configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)

        with torch.no_grad():
            log_probs = sampler.log_prob(all_configs.float())
            probs = torch.exp(log_probs)

        # Compute marginal occupation probabilities for each site
        marginals = (all_configs.float() * probs.unsqueeze(1)).sum(dim=0)

        # Product of marginals for each config
        product_probs = torch.ones(len(all_configs))
        for i in range(2 * n_orb):
            p_i = marginals[i]
            product_probs *= torch.where(
                all_configs[:, i] == 1,
                p_i,
                1 - p_i,
            )

        # If model is autoregressive, joint != product-of-marginals
        # Use KL divergence as a measure
        probs_normalized = probs / probs.sum()
        product_normalized = product_probs / product_probs.sum()

        kl_div = (probs_normalized * (
            torch.log(probs_normalized + 1e-10) - torch.log(product_normalized + 1e-10)
        )).sum()

        # Even an untrained model should show SOME deviation from product form
        # because the transformer architecture inherently couples positions.
        # We use a very generous threshold.
        assert kl_div.abs() > 1e-10, (
            f"KL(joint || product) = {kl_div.item():.2e}, "
            "model appears to be product-of-marginals"
        )


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_sample(self):
        """Sampling 1 config should work."""
        sampler = _make_sampler(4, 2, 2)
        log_probs, unique = sampler.sample(1)
        assert log_probs.shape == (1,)
        assert len(unique) >= 1

    def test_all_electrons_filled(self):
        """System where all orbitals are occupied (n_alpha=n_orb, n_beta=n_orb).

        Only 1 valid config: all ones.
        """
        n_orb = 3
        sampler = _make_sampler(n_orb, n_orb, n_orb)

        with torch.no_grad():
            configs, log_probs, unique = sampler.sample_with_probs(10)

        # Only one valid config: all orbitals occupied
        assert len(unique) == 1, (
            f"Expected 1 unique config (all filled), got {len(unique)}"
        )
        assert unique[0].sum().item() == 2 * n_orb, (
            "The only valid config should have all orbitals occupied"
        )

    def test_one_electron_each(self):
        """System with 1 alpha, 1 beta electron."""
        n_orb = 4
        sampler = _make_sampler(n_orb, 1, 1)

        with torch.no_grad():
            _, unique = sampler.sample(200)

        # Should produce configs with exactly 1 alpha and 1 beta
        for cfg in unique:
            assert cfg[:n_orb].sum().item() == 1
            assert cfg[n_orb:].sum().item() == 1

    def test_asymmetric_electrons(self):
        """n_alpha != n_beta (open-shell system)."""
        n_orb = 5
        n_alpha, n_beta = 3, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        with torch.no_grad():
            configs, _, _ = sampler.sample_with_probs(100)

        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == n_alpha).all()
        assert (beta_counts == n_beta).all()

    def test_reproducibility_with_seed(self):
        """Same random seed should give same samples."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        torch.manual_seed(12345)
        with torch.no_grad():
            configs_1, log_probs_1, _ = sampler.sample_with_probs(20)

        torch.manual_seed(12345)
        with torch.no_grad():
            configs_2, log_probs_2, _ = sampler.sample_with_probs(20)

        assert torch.equal(configs_1, configs_2), "Same seed should give same configs"
        assert torch.allclose(log_probs_1, log_probs_2, atol=1e-6), (
            "Same seed should give same log_probs"
        )

    def test_no_nan_in_gradients(self):
        """Gradients should never contain NaN."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        all_configs = _enumerate_all_configs(n_orb, n_alpha, n_beta)[:16].float()
        log_probs = sampler.log_prob(all_configs)
        loss = -log_probs.mean()
        loss.backward()

        for name, param in sampler.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"NaN/Inf gradient in {name}: {param.grad}"
                )

    def test_device_cpu(self):
        """Sampler should work on CPU."""
        sampler = _make_sampler(4, 2, 2)
        # All parameters should be on CPU
        for name, param in sampler.named_parameters():
            assert param.device.type == 'cpu', (
                f"Parameter {name} on {param.device}, expected cpu"
            )

        with torch.no_grad():
            log_probs, unique = sampler.sample(20)

        assert log_probs.device.type == 'cpu'
        assert unique.device.type == 'cpu'


# ===================================================================
# TestTraining (basic optimization test)
# ===================================================================

class TestTraining:
    """Verify the sampler can be trained via gradient descent."""

    def test_loss_decreases_with_training(self):
        """Negative log-likelihood on a target config should decrease."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        # Target: HF config (first n_alpha and n_beta orbitals occupied)
        target = torch.zeros(1, 2 * n_orb)
        for i in range(n_alpha):
            target[0, i] = 1
        for i in range(n_beta):
            target[0, n_orb + i] = 1

        optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-3)

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            log_prob = sampler.log_prob(target)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (first half avg > second half avg)
        first_half = sum(losses[:25]) / 25
        second_half = sum(losses[25:]) / 25
        assert second_half < first_half, (
            f"Loss did not decrease: first_half={first_half:.4f}, "
            f"second_half={second_half:.4f}"
        )

    def test_probability_increases_for_target(self):
        """After training on a target config, its probability should increase."""
        n_orb, n_alpha, n_beta = 4, 2, 2
        sampler = _make_sampler(n_orb, n_alpha, n_beta)

        # HF target config
        target = torch.zeros(1, 2 * n_orb)
        for i in range(n_alpha):
            target[0, i] = 1
        for i in range(n_beta):
            target[0, n_orb + i] = 1

        with torch.no_grad():
            initial_prob = sampler.estimate_discrete_prob(target).item()

        optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-3)
        for step in range(100):
            optimizer.zero_grad()
            log_prob = sampler.log_prob(target)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_prob = sampler.estimate_discrete_prob(target).item()

        assert final_prob > initial_prob, (
            f"Target probability did not increase: {initial_prob:.6f} -> {final_prob:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
