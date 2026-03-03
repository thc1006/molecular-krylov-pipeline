"""Tests for Normalizing Flow implementations."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flows.discrete_flow import DiscreteFlowSampler


class TestDiscreteFlowSampler:
    """Test cases for Discrete Flow Sampler."""

    def test_construction(self):
        """Test basic construction."""
        flow = DiscreteFlowSampler(
            num_sites=4,
            num_coupling_layers=2,
            hidden_dims=[32, 32],
        )
        assert flow.num_sites == 4

    def test_sample_continuous(self):
        """Test continuous sampling."""
        flow = DiscreteFlowSampler(num_sites=4, num_coupling_layers=2)

        samples = flow.sample_continuous(batch_size=10)

        # Should be in [-1, 1] due to tanh
        assert samples.shape == (10, 4)
        assert torch.all(samples >= -1)
        assert torch.all(samples <= 1)

    def test_discretize(self):
        """Test discretization."""
        flow = DiscreteFlowSampler(num_sites=4)

        y = torch.tensor([
            [0.5, -0.5, 0.1, -0.1],
            [-0.9, 0.9, -0.9, 0.9],
        ])

        configs = flow.discretize(y)

        # Should be {0, 1}
        assert torch.all((configs == 0) | (configs == 1))

        # Check specific values
        expected = torch.tensor([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
        assert torch.all(configs == expected)

    def test_sample_returns_unique(self):
        """Test that sampling returns unique configurations."""
        flow = DiscreteFlowSampler(num_sites=4)

        _, unique_configs = flow.sample(batch_size=100)

        # All rows should be unique
        n_unique = len(unique_configs)
        assert n_unique <= 100
        assert n_unique > 0

        # Check uniqueness
        unique_set = set(tuple(c.tolist()) for c in unique_configs)
        assert len(unique_set) == n_unique

    def test_log_prob_continuous(self):
        """Test continuous log probability computation."""
        flow = DiscreteFlowSampler(num_sites=4, num_coupling_layers=2)

        y = torch.randn(10, 4) * 0.5  # Stay away from boundaries

        log_probs = flow.log_prob_continuous(y)

        assert log_probs.shape == (10,)
        assert torch.all(torch.isfinite(log_probs))

    def test_estimate_discrete_prob(self):
        """Test discrete probability estimation."""
        flow = DiscreteFlowSampler(num_sites=4, num_coupling_layers=2)

        configs = torch.tensor([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ])

        probs = flow.estimate_discrete_prob(configs, n_mc_samples=10)

        assert probs.shape == (2,)
        assert torch.all(probs > 0)
        assert torch.all(torch.isfinite(probs))


class TestFlowTraining:
    """Test flow training functionality."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_training_step(self):
        """Test a single training step."""
        from flows.training import FlowNQSTrainer, TrainingConfig
        from nqs.dense import DenseNQS
        from hamiltonians.spin import TransverseFieldIsing

        # Small system for fast test
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)
        flow = DiscreteFlowSampler(num_sites=4, num_coupling_layers=2)
        nqs = DenseNQS(num_sites=4, hidden_dims=[32, 32])

        config = TrainingConfig(
            samples_per_batch=100,
            num_batches=2,
        )

        trainer = FlowNQSTrainer(
            flow=flow,
            nqs=nqs,
            hamiltonian=H,
            config=config,
            device="cpu",
        )

        # Run one step
        metrics = trainer.train_step()

        assert "energy" in metrics
        assert "unique_ratio" in metrics
        assert metrics["unique_ratio"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
