"""Tests for PR 2.3: SigmoidTopK — differentiable top-k via sigmoid threshold.

Replaces GumbelTopK with implicit-differentiation-based sigmoid selection.
Based on Thomas Ahle's approach (2022) with exact Jacobian backward.

References:
- Thomas Ahle, "A Differentiable Top-k Layer for PyTorch" (2022)
- Klas Wijk, "Top-k Sampling Beyond Gumbel Top-k" (2026)
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSigmoidTopKParticleConservation:
    """Hard selection must produce exactly k ones per row."""

    def test_exact_k_selected(self):
        """Every row must have exactly k ones."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(32, 10)

        for k in [1, 3, 5, 8]:
            selection = topk(logits, k, hard=True)
            counts = selection.sum(dim=-1)
            assert (counts == k).all(), (
                f"k={k}: expected {k} per row, got {counts.tolist()}"
            )

    def test_binary_values_hard(self):
        """Hard selection must be {0, 1} only."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(16, 8)
        selection = topk(logits, 3, hard=True)

        unique_vals = set(selection.unique().tolist())
        assert unique_vals.issubset({0.0, 1.0}), (
            f"Hard selection has non-binary values: {unique_vals}"
        )

    def test_soft_sums_to_k(self):
        """Soft selection values must sum to k per row."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(16, 10)
        soft = topk(logits, 4, hard=False)

        sums = soft.sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, 4.0), atol=1e-4), (
            f"Soft selection sums: {sums.tolist()}, expected 4.0"
        )

    def test_soft_values_in_unit_interval(self):
        """Soft selection values must be in [0, 1]."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(16, 10)
        soft = topk(logits, 3, hard=False)

        assert (soft >= -1e-6).all() and (soft <= 1 + 1e-6).all(), (
            f"Soft values out of [0,1]: min={soft.min()}, max={soft.max()}"
        )


class TestSigmoidTopKGradients:
    """Gradients must flow to ALL positions via implicit differentiation."""

    def test_gradients_flow_to_boundary_positions_soft(self):
        """Soft mode: positions near the decision boundary get gradients.

        Note: selection.sum() is always k (by constraint), so its gradient is 0.
        We use a weighted loss to test that gradients flow to individual positions.
        """
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = (torch.randn(32, 6) * 0.5).requires_grad_(True)
        selection = topk(logits, 2, hard=False)
        # Weighted loss breaks symmetry — gradient should be non-zero
        weights = torch.arange(1, 7, dtype=torch.float32)
        loss = (selection * weights).sum()
        loss.backward()

        assert logits.grad is not None
        # With weighted loss, most positions should get gradient
        nonzero_frac = (logits.grad.abs() > 1e-10).float().mean(dim=0)
        assert (nonzero_frac > 0.3).all(), (
            f"Too few positions got gradient: {nonzero_frac.tolist()}"
        )

    def test_gradients_flow_to_boundary_positions_hard(self):
        """Hard mode (STE): boundary positions still get gradients."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = (torch.randn(32, 6) * 0.5).requires_grad_(True)
        selection = topk(logits, 2, hard=True)
        # Weighted loss — selection.sum() is always k, gradient would be 0
        weights = torch.arange(1, 7, dtype=torch.float32)
        loss = (selection * weights).sum()
        loss.backward()

        assert logits.grad is not None
        nonzero_frac = (logits.grad.abs() > 1e-10).float().mean(dim=0)
        assert (nonzero_frac > 0.3).all(), (
            f"Hard STE: too few positions got gradient: {nonzero_frac.tolist()}"
        )

    def test_implicit_diff_jacobian_correctness(self):
        """Numerical gradient check: implicit diff Jacobian is correct."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)

        # Use soft mode for smooth gradients (hard has STE discontinuity)
        def func(x):
            return topk(x, 2, hard=False)

        passed = torch.autograd.gradcheck(func, (logits,), eps=1e-6, atol=1e-4)
        assert passed, "Implicit differentiation Jacobian failed gradcheck"

    def test_gradient_nonzero_at_moderate_temperature(self):
        """At moderate temperature, sigmoid still has meaningful gradients."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=0.5)
        logits = (torch.randn(16, 6) * 0.3).requires_grad_(True)
        selection = topk(logits, 2, hard=False)
        # Weighted loss — selection.sum() is always k, gradient would be 0
        weights = torch.arange(1, 7, dtype=torch.float32)
        loss = (selection * weights).sum()
        loss.backward()

        grad_norm = logits.grad.abs().mean().item()
        assert grad_norm > 1e-6, (
            f"Gradient vanished at temperature=0.5: norm={grad_norm}"
        )


class TestSigmoidTopKDropInCompatible:
    """SigmoidTopK must have same API as GumbelTopK."""

    def test_same_signature(self):
        """forward(logits, k, hard) -> (batch, n) tensor."""
        from flows.particle_conserving_flow import SigmoidTopK, GumbelTopK

        sigmoid_topk = SigmoidTopK(temperature=1.0)
        gumbel_topk = GumbelTopK(temperature=1.0)

        logits = torch.randn(8, 6)
        s_out = sigmoid_topk(logits, 2, hard=True)
        g_out = gumbel_topk(logits, 2, hard=True)

        assert s_out.shape == g_out.shape, (
            f"Shape mismatch: sigmoid={s_out.shape}, gumbel={g_out.shape}"
        )

    def test_works_in_particle_conserving_flow(self):
        """SigmoidTopK integrates with ParticleConservingFlow."""
        from flows.particle_conserving_flow import (
            ParticleConservingFlow, SigmoidTopK
        )

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )
        # Replace GumbelTopK with SigmoidTopK
        flow.gumbel_topk = SigmoidTopK(temperature=1.0)

        configs, log_probs = flow.sample(batch_size=16, hard=True)
        assert configs.shape == (16, 12)
        # Verify particle conservation
        alpha_counts = configs[:, :6].sum(dim=-1)
        beta_counts = configs[:, 6:].sum(dim=-1)
        assert (alpha_counts == 2).all(), f"Alpha violation: {alpha_counts}"
        assert (beta_counts == 2).all(), f"Beta violation: {beta_counts}"


class TestSigmoidTopKDeterminism:
    """SigmoidTopK is deterministic (no Gumbel noise)."""

    def test_soft_output_deterministic(self):
        """Same logits → same soft output (no stochastic noise)."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(4, 6)

        soft1 = topk(logits, 2, hard=False)
        soft2 = topk(logits, 2, hard=False)

        assert torch.allclose(soft1, soft2, atol=1e-10), (
            "SigmoidTopK soft output should be deterministic"
        )

    def test_hard_output_deterministic(self):
        """Same logits → same hard output."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(4, 6)

        hard1 = topk(logits, 2, hard=True)
        hard2 = topk(logits, 2, hard=True)

        assert torch.equal(hard1, hard2), (
            "SigmoidTopK hard output should be deterministic"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSigmoidTopKGPU:
    """SigmoidTopK must work on GPU without CPU-GPU sync issues."""

    def test_forward_backward_on_gpu(self):
        """Full forward+backward on GPU produces correct results."""
        from flows.particle_conserving_flow import SigmoidTopK

        device = torch.device("cuda")
        topk = SigmoidTopK(temperature=1.0)
        logits = torch.randn(32, 10, device=device, requires_grad=True)
        selection = topk(logits, 3, hard=True)

        assert selection.device.type == "cuda"
        assert (selection.sum(dim=-1) == 3).all()

        weights = torch.arange(1, 11, dtype=torch.float32, device=device)
        loss = (selection * weights).sum()
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.device.type == "cuda"

    def test_plackett_luce_on_gpu(self):
        """Plackett-Luce log_prob works on GPU."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        device = torch.device("cuda")
        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        ).to(device)

        configs, log_probs = flow.sample(batch_size=16, hard=True)
        assert configs.device.type == "cuda"
        assert torch.isfinite(log_probs).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
