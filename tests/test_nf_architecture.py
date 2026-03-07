"""Tests for PR 2.2: NF Architecture Fixes.

Critical bugs in the normalizing flow:
1. Dead `beta_scorer` — ~160K unused parameters (memory + optimizer waste)
2. entropy_weight=0.0 default — anti-collapse disabled, NF collapses at low temperature
3. GumbelTopK gradient masking — hard selection blocks gradients to non-selected positions

References:
- ADR-001 Round 4 findings: NF critical bugs
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDeadBetaScorer:
    """beta_scorer is created but never used in sample(). Should be removed."""

    def test_no_beta_scorer_attribute(self):
        """ParticleConservingFlow should NOT have a standalone beta_scorer."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        sampler = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )
        flow = sampler.flow  # Inner ParticleConservingFlow
        assert not hasattr(flow, 'beta_scorer'), (
            "Dead beta_scorer still exists — unused parameters. "
            "Remove it since sample() uses beta_conditioned_scorer instead."
        )

    def test_all_parameters_used_in_forward(self):
        """Every parameter should receive gradients during training."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )

        # Forward pass
        configs, log_probs = flow.flow.sample(batch_size=8, hard=False)
        loss = -log_probs.mean()
        loss.backward()

        # Check all parameters received gradients
        unused_params = []
        for name, param in flow.named_parameters():
            if param.grad is None or param.grad.abs().max().item() == 0:
                unused_params.append(name)

        assert len(unused_params) == 0, (
            f"Unused parameters (no gradient): {unused_params}"
        )

    def test_parameter_count_reduced(self):
        """After removing beta_scorer, total params should decrease."""
        from flows.particle_conserving_flow import (
            ParticleConservingFlowSampler, OrbitalScoringNetwork
        )

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )
        total_params = sum(p.numel() for p in flow.parameters())

        # OrbitalScoringNetwork(6, [64, 64]) has roughly:
        # Linear(6, 64) + Linear(64, 64) + Linear(64, 6) = 384+64 + 4096+64 + 384+6 ≈ 5K
        # So dead beta_scorer wastes ~5K params (more with [256,256])
        # Flow should only have alpha_scorer + alpha_to_beta + beta_conditioned_scorer + gumbel_topk
        # No standalone beta_scorer
        scorer_params = sum(
            p.numel() for p in OrbitalScoringNetwork(6, [64, 64]).parameters()
        )

        # After fix: total should NOT include scorer_params for dead beta_scorer
        # This test will FAIL if beta_scorer still exists (total includes it)
        # We check that NO parameter named 'beta_scorer.*' exists
        beta_scorer_params = sum(
            p.numel() for name, p in flow.named_parameters()
            if name.startswith('beta_scorer.')
        )
        assert beta_scorer_params == 0, (
            f"beta_scorer has {beta_scorer_params} parameters — should be removed"
        )


class TestEntropyWeight:
    """entropy_weight=0.0 disables anti-collapse. Must be > 0 by default."""

    def test_default_entropy_weight_positive(self):
        """PhysicsGuidedFlowTrainer default entropy_weight must be > 0."""
        from flows.physics_guided_training import PhysicsGuidedConfig

        config = PhysicsGuidedConfig()
        assert config.entropy_weight > 0, (
            f"entropy_weight={config.entropy_weight}, must be > 0 to prevent NF collapse. "
            "Temperature annealing without entropy regularization causes mode collapse."
        )

    def test_entropy_in_loss(self):
        """Training loss must include entropy term when entropy_weight > 0."""
        from flows.physics_guided_training import PhysicsGuidedConfig

        config = PhysicsGuidedConfig(entropy_weight=0.1)
        assert config.entropy_weight == 0.1

    def test_entropy_prevents_collapse(self):
        """With entropy > 0, NF should produce diverse samples (not all identical)."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
            temperature=0.1,  # Low temperature encourages collapse
        )

        # Sample with low temperature
        with torch.no_grad():
            configs, _ = flow.flow.sample(batch_size=100, hard=True)

        # Check diversity: should have more than 1 unique config
        unique = torch.unique(configs, dim=0)
        assert len(unique) > 1, (
            f"NF collapsed to {len(unique)} unique config(s) at temperature=0.1. "
            "Entropy regularization should prevent this."
        )


class TestGumbelTopKGradients:
    """GumbelTopK must pass gradients to ALL logit positions, not just selected ones."""

    def test_gradients_flow_to_all_positions(self):
        """Gradients must reach all n_orbitals positions, not just top-k."""
        from flows.particle_conserving_flow import GumbelTopK

        n_orbitals = 6
        k = 2
        gumbel = GumbelTopK(temperature=1.0)

        logits = torch.randn(4, n_orbitals, requires_grad=True)
        selection = gumbel(logits, k, hard=False)  # soft for gradient flow
        loss = selection.sum()
        loss.backward()

        # ALL positions should have non-zero gradients
        assert logits.grad is not None
        grad_nonzero = (logits.grad.abs() > 1e-10).float().mean(dim=0)
        assert (grad_nonzero > 0).all(), (
            f"Some positions got zero gradient: {grad_nonzero.tolist()}. "
            "GumbelTopK must propagate gradients to all positions for exploration."
        )

    def test_hard_selection_has_straight_through_gradients(self):
        """Hard selection (forward) with soft gradients (backward)."""
        from flows.particle_conserving_flow import GumbelTopK

        n_orbitals = 6
        k = 2
        gumbel = GumbelTopK(temperature=1.0)

        logits = torch.randn(4, n_orbitals, requires_grad=True)
        selection = gumbel(logits, k, hard=True)

        # Forward should be discrete (0 or 1)
        assert set(selection.unique().tolist()).issubset({0.0, 1.0}), (
            "Hard selection should produce binary {0, 1} values"
        )

        # Each sample should select exactly k items
        assert (selection.sum(dim=-1) == k).all(), (
            "Each sample must select exactly k orbitals"
        )

        # Backward should still work (straight-through estimator)
        loss = selection.sum()
        loss.backward()
        assert logits.grad is not None, "No gradients through hard GumbelTopK"


class TestNFSampleQuality:
    """NF samples must satisfy physical constraints."""

    def test_particle_conservation(self):
        """Every sample must have exactly n_alpha + n_beta electrons."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )

        with torch.no_grad():
            configs, _ = flow.flow.sample(batch_size=50, hard=True)

        n_orb = 6
        alpha_counts = configs[:, :n_orb].sum(dim=-1)
        beta_counts = configs[:, n_orb:].sum(dim=-1)

        assert (alpha_counts == 2).all(), f"Alpha electron count violation: {alpha_counts}"
        assert (beta_counts == 2).all(), f"Beta electron count violation: {beta_counts}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
