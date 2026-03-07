"""Tests for Normalizing Flow implementations."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flows.particle_conserving_flow import (
    ParticleConservingFlowSampler,
    ParticleConservingFlow,
    GumbelTopK,
    verify_particle_conservation,
)


class TestGumbelTopK:
    """Test Gumbel-Softmax top-k selection."""

    def test_output_shape(self):
        logits = torch.randn(10, 8)
        topk = GumbelTopK(temperature=1.0)
        result = topk(logits, k=3)
        assert result.shape == (10, 8)

    def test_k_selected(self):
        logits = torch.randn(5, 6)
        topk = GumbelTopK(temperature=0.1)
        result = topk(logits, k=2)
        counts = result.sum(dim=1)
        assert torch.allclose(counts, torch.tensor([2.0] * 5))

    def test_gradient_flow(self):
        logits = torch.randn(4, 6, requires_grad=True)
        topk = GumbelTopK(temperature=1.0)
        result = topk(logits, k=2)
        loss = result.sum()
        loss.backward()
        assert logits.grad is not None


class TestParticleConservingFlow:
    """Test particle-conserving normalizing flow."""

    def test_construction(self):
        flow = ParticleConservingFlowSampler(
            num_sites=8, n_alpha=2, n_beta=2, hidden_dims=[32, 32]
        )
        assert flow.n_orbitals == 4
        assert flow.flow.n_alpha == 2
        assert flow.flow.n_beta == 2

    def test_sample_particle_conservation(self):
        flow = ParticleConservingFlowSampler(
            num_sites=8, n_alpha=2, n_beta=2, hidden_dims=[32, 32]
        )
        log_probs, unique_configs = flow.sample(n_samples=20)

        n_orb = 4
        alpha_counts = unique_configs[:, :n_orb].sum(dim=1)
        beta_counts = unique_configs[:, n_orb:].sum(dim=1)

        assert torch.all(alpha_counts == 2), "Alpha electron count violated"
        assert torch.all(beta_counts == 2), "Beta electron count violated"

    def test_sample_output_shape(self):
        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2, hidden_dims=[32, 32]
        )
        log_probs, unique_configs = flow.sample(n_samples=10)
        assert unique_configs.shape[1] == 12  # 2 * n_orbitals
        assert unique_configs.shape[0] <= 10  # unique <= n_samples
        assert log_probs.shape == (10,)

    def test_verify_particle_conservation_util(self):
        # Valid configs (n_orbitals=4)
        configs = torch.tensor([
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 1, 0],
        ])
        valid, stats = verify_particle_conservation(configs, n_orbitals=4, n_alpha=2, n_beta=2)
        assert valid

        # Invalid configs (3 alpha instead of 2)
        bad_configs = torch.tensor([
            [1, 1, 1, 0, 1, 1, 0, 0],
        ])
        valid, stats = verify_particle_conservation(bad_configs, n_orbitals=4, n_alpha=2, n_beta=2)
        assert not valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
