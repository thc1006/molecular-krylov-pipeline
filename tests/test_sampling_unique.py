"""Tests for P2.4: Sampling Without Replacement for AR Flow.

When the wavefunction distribution is peaked, standard i.i.d. sampling
produces many duplicate configurations. Sampling without replacement
(rejection-based unique accumulation) improves coverage efficiency.

Reference: Peaked Wavefunctions (arXiv:2408.07625), 118Q single GPU
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSampleUnique:
    """Test unique sampling functionality."""

    @pytest.fixture
    def small_flow(self):
        """Create a small AR flow for testing."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        flow = AutoregressiveFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1,
        )
        return flow

    def test_sample_unique_returns_unique(self, small_flow):
        """All returned configs should be unique."""
        torch.manual_seed(42)
        log_probs, configs = small_flow.sample_unique(n_unique_target=10)
        # Check uniqueness by converting to set of tuples
        config_set = set()
        for i in range(len(configs)):
            t = tuple(configs[i].cpu().tolist())
            assert t not in config_set, f"Duplicate config at index {i}"
            config_set.add(t)

    def test_sample_unique_respects_target(self, small_flow):
        """Should return at most n_unique_target configs."""
        torch.manual_seed(42)
        target = 5
        log_probs, configs = small_flow.sample_unique(n_unique_target=target)
        assert len(configs) <= target
        assert len(log_probs) == len(configs)

    def test_sample_unique_log_probs_shape(self, small_flow):
        """Log probs should match number of unique configs."""
        torch.manual_seed(42)
        log_probs, configs = small_flow.sample_unique(n_unique_target=8)
        assert log_probs.shape[0] == configs.shape[0]

    def test_sample_unique_log_probs_valid(self, small_flow):
        """Log probs should be finite and negative."""
        torch.manual_seed(42)
        log_probs, configs = small_flow.sample_unique(n_unique_target=8)
        assert torch.isfinite(log_probs).all()
        assert (log_probs <= 0).all()

    def test_sample_unique_more_unique_than_standard(self, small_flow):
        """Unique sampling should yield more unique configs for same effort."""
        torch.manual_seed(42)
        # Standard sampling: many duplicates likely
        _, standard_unique = small_flow.sample(50)
        n_standard = len(standard_unique)

        # Unique sampling: targets specific count
        torch.manual_seed(42)
        _, unique_configs = small_flow.sample_unique(n_unique_target=n_standard)
        n_unique = len(unique_configs)

        # Both should have similar unique count (unique sampling shouldn't do worse)
        assert n_unique >= n_standard - 1  # Allow small tolerance

    def test_sample_unique_particle_conservation(self, small_flow):
        """Unique samples should conserve particle number."""
        torch.manual_seed(42)
        _, configs = small_flow.sample_unique(n_unique_target=10)
        n_orb = small_flow.n_orbitals  # 2

        for i in range(len(configs)):
            alpha_count = configs[i, :n_orb].sum().item()
            beta_count = configs[i, n_orb:].sum().item()
            assert alpha_count == small_flow.n_alpha, (
                f"Config {i}: alpha={alpha_count}, expected {small_flow.n_alpha}"
            )
            assert beta_count == small_flow.n_beta, (
                f"Config {i}: beta={beta_count}, expected {small_flow.n_beta}"
            )

    def test_sample_unique_small_config_space(self):
        """For tiny systems, should find most valid configs."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        # H2: 2 orbitals, 1 alpha, 1 beta → C(2,1)×C(2,1) = 4 configs
        flow = AutoregressiveFlowSampler(num_sites=4, n_alpha=1, n_beta=1)
        torch.manual_seed(42)
        _, configs = flow.sample_unique(
            n_unique_target=4, max_attempts=20, batch_multiplier=5.0,
        )
        # Should find at least 3 of 4 valid configs (some may have near-zero prob)
        assert len(configs) >= 3

    def test_sample_unique_max_attempts_respected(self, small_flow):
        """Should stop after max_attempts even if target not reached."""
        # Request more than the config space size
        torch.manual_seed(42)
        log_probs, configs = small_flow.sample_unique(
            n_unique_target=1000, max_attempts=3, batch_multiplier=1.0,
        )
        # Should return whatever was found, not hang
        assert len(configs) > 0
        assert len(configs) < 1000

    def test_sample_unique_gradient_detached(self, small_flow):
        """Returned log_probs should be from teacher forcing (no grad by default)."""
        torch.manual_seed(42)
        log_probs, configs = small_flow.sample_unique(n_unique_target=5)
        # log_probs from torch.no_grad() context should not require grad
        assert not log_probs.requires_grad


class TestSampleUniqueIntegration:
    """Test unique sampling with pipeline-like usage."""

    @pytest.fixture
    def lih_flow(self):
        """Create AR flow for LiH-sized system."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        # LiH: 6 orbitals, 2 alpha, 2 beta → C(6,2)^2 = 225 configs
        flow = AutoregressiveFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
        )
        return flow

    def test_lih_unique_coverage(self, lih_flow):
        """Unique sampling should cover more of LiH config space."""
        torch.manual_seed(42)
        _, configs = lih_flow.sample_unique(n_unique_target=50)
        assert len(configs) >= 20  # Should find at least 20 unique

    def test_lih_configs_are_valid(self, lih_flow):
        """All LiH configs should have correct electron count."""
        torch.manual_seed(42)
        _, configs = lih_flow.sample_unique(n_unique_target=30)
        n_orb = lih_flow.n_orbitals  # 6
        for i in range(len(configs)):
            assert configs[i, :n_orb].sum().item() == 2
            assert configs[i, n_orb:].sum().item() == 2
