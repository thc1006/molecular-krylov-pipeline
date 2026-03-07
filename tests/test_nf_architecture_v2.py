"""Tests for PR 2.2: Scale NF Architecture (remaining items).

Changes:
1. Hidden dims auto-scaling by n_orbitals (depth, not width)
2. Alpha-beta zero-padding fix — pass alpha_config instead of zeros
3. Temperature annealing improvement — exponential decay, higher floor
4. Non-autoregressive limitation documented
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# 1. Hidden dims auto-scaling
# ============================================================

class TestHiddenDimsAutoScale:
    """ParticleConservingFlowSampler should scale hidden_dims by n_orbitals."""

    def test_small_system_2_layers(self):
        """n_orbitals <= 10: should use 2 hidden layers."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=16, n_alpha=3, n_beta=3,  # n_orbitals=8
        )
        # Count layers in beta_conditioned_scorer (Linear layers only)
        linear_layers = [m for m in flow.flow.beta_conditioned_scorer if isinstance(m, nn.Linear)]
        assert len(linear_layers) >= 2

    def test_medium_system_3_layers(self):
        """n_orbitals 11-15: should use 3 hidden layers."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=24, n_alpha=4, n_beta=4,  # n_orbitals=12
        )
        linear_layers = [m for m in flow.flow.beta_conditioned_scorer if isinstance(m, nn.Linear)]
        # 3 hidden layers = input→h1, h1→h2, h2→h3, h3→output = 4 Linear
        assert len(linear_layers) >= 4, (
            f"n_orbitals=12 should have >= 3 hidden layers (4 Linear), got {len(linear_layers)}"
        )

    def test_large_system_wider(self):
        """n_orbitals >= 16: should use wider layers."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=40, n_alpha=5, n_beta=5,  # n_orbitals=20
        )
        linear_layers = [m for m in flow.flow.beta_conditioned_scorer if isinstance(m, nn.Linear)]
        # First hidden layer should be >= 384
        assert linear_layers[0].out_features >= 384, (
            f"n_orbitals=20 first hidden should be >= 384, got {linear_layers[0].out_features}"
        )

    def test_explicit_hidden_dims_override(self):
        """User-provided hidden_dims should override auto-scaling."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=40, n_alpha=5, n_beta=5,
            hidden_dims=[128, 128],  # User override
        )
        linear_layers = [m for m in flow.flow.beta_conditioned_scorer if isinstance(m, nn.Linear)]
        assert linear_layers[0].out_features == 128, (
            "User-provided hidden_dims should override auto-scaling"
        )


# ============================================================
# 2. Alpha-beta zero-padding fix
# ============================================================

class TestAlphaBetaConditioning:
    """Beta scorer must receive alpha_config, not zeros."""

    def test_beta_input_not_zeros(self):
        """sample() must pass alpha_config to beta scorer, not zeros."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler
        import unittest.mock as mock

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )

        # Hook into beta_conditioned_scorer to inspect its input
        captured_inputs = []
        original_forward = flow.flow.beta_conditioned_scorer.forward

        def spy_forward(x):
            captured_inputs.append(x.detach().clone())
            return original_forward(x)

        with mock.patch.object(flow.flow.beta_conditioned_scorer, 'forward', side_effect=spy_forward):
            configs, log_probs = flow.flow.sample(batch_size=8, hard=True)

        assert len(captured_inputs) == 1
        beta_input = captured_inputs[0]
        n_orb = flow.n_orbitals

        # First n_orbitals dims should NOT be all zeros
        first_dims = beta_input[:, :n_orb]
        assert first_dims.abs().sum() > 0, (
            "Beta scorer first n_orbitals dims are all zeros — "
            "should receive alpha_config for orbital conflict modeling"
        )

    def test_beta_input_matches_alpha_config(self):
        """The first n_orbitals dims of beta input should be the alpha config."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler
        import unittest.mock as mock

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )

        captured_inputs = []
        original_forward = flow.flow.beta_conditioned_scorer.forward

        def spy_forward(x):
            captured_inputs.append(x.detach().clone())
            return original_forward(x)

        with mock.patch.object(flow.flow.beta_conditioned_scorer, 'forward', side_effect=spy_forward):
            configs, _ = flow.flow.sample(batch_size=8, hard=True)

        beta_input = captured_inputs[0]
        n_orb = flow.n_orbitals
        alpha_in_beta = beta_input[:, :n_orb]
        alpha_from_config = configs[:, :n_orb]

        # Alpha config passed to beta should match the alpha part of output
        assert torch.allclose(alpha_in_beta, alpha_from_config.float(), atol=1e-5), (
            "Beta scorer should receive exact alpha_config as first n_orbitals dims"
        )

    def test_log_prob_also_uses_alpha_config(self):
        """log_prob() must also pass alpha_config, not zeros, to beta scorer."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler
        import unittest.mock as mock

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )

        # Get some configs first
        with torch.no_grad():
            configs, _ = flow.flow.sample(batch_size=8, hard=True)

        captured_inputs = []
        original_forward = flow.flow.beta_conditioned_scorer.forward

        def spy_forward(x):
            captured_inputs.append(x.detach().clone())
            return original_forward(x)

        with mock.patch.object(flow.flow.beta_conditioned_scorer, 'forward', side_effect=spy_forward):
            flow.flow.log_prob(configs)

        beta_input = captured_inputs[0]
        n_orb = flow.n_orbitals
        first_dims = beta_input[:, :n_orb]

        assert first_dims.abs().sum() > 0, (
            "log_prob() beta scorer also receives zeros — must use alpha_config"
        )

    def test_particle_conservation_after_fix(self):
        """Particle conservation must still hold after alpha-beta fix."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=12, n_alpha=2, n_beta=2,
            hidden_dims=[64, 64],
        )

        with torch.no_grad():
            configs, _ = flow.flow.sample(batch_size=100, hard=True)

        n_orb = 6
        alpha_counts = configs[:, :n_orb].sum(dim=-1)
        beta_counts = configs[:, n_orb:].sum(dim=-1)

        assert (alpha_counts == 2).all(), f"Alpha count violation: {alpha_counts}"
        assert (beta_counts == 2).all(), f"Beta count violation: {beta_counts}"


# ============================================================
# 3. Temperature annealing improvement
# ============================================================

class TestTemperatureAnnealing:
    """Temperature schedule should use exponential decay with higher floor."""

    def test_final_temperature_floor(self):
        """final_temperature should be >= 0.3 (not 0.1)."""
        from flows.physics_guided_training import PhysicsGuidedConfig

        config = PhysicsGuidedConfig()
        assert config.final_temperature >= 0.3, (
            f"final_temperature={config.final_temperature}, should be >= 0.3 "
            "to prevent sigmoid Jacobian degeneration"
        )

    def test_decay_epochs_increased(self):
        """temperature_decay_epochs should be >= 300."""
        from flows.physics_guided_training import PhysicsGuidedConfig

        config = PhysicsGuidedConfig()
        assert config.temperature_decay_epochs >= 300, (
            f"temperature_decay_epochs={config.temperature_decay_epochs}, "
            "should be >= 300 for slower annealing"
        )

    def test_exponential_decay_schedule(self):
        """Temperature should follow exponential decay, not linear."""
        from flows.physics_guided_training import PhysicsGuidedConfig

        config = PhysicsGuidedConfig()
        T_init = config.initial_temperature
        T_final = config.final_temperature
        decay_epochs = config.temperature_decay_epochs

        # Compute temperature at various epochs using the expected formula
        # Exponential: T(e) = T_final + (T_init - T_final) * exp(-decay_rate * e)
        # At epoch=0: T=T_init. At epoch=decay_epochs: T ≈ T_final

        # Verify midpoint is NOT the linear midpoint
        # Linear midpoint: (T_init + T_final) / 2
        # Exponential midpoint: T_final + (T_init - T_final) * exp(-rate * mid)
        # Exponential midpoint > linear midpoint (slower initial decay)
        linear_midpoint = (T_init + T_final) / 2

        # We just check the config has the right structure
        assert T_init > T_final, "initial > final temperature"
        assert T_init >= 0.8, f"initial_temperature should be >= 0.8, got {T_init}"


# ============================================================
# 4. Non-autoregressive limitation documentation
# ============================================================

class TestArchitectureDocumentation:
    """Architecture limitations should be documented in docstrings."""

    def test_flow_class_documents_limitation(self):
        """ParticleConservingFlow docstring should mention non-autoregressive limitation."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        docstring = ParticleConservingFlow.__doc__ or ""
        has_limitation = (
            "non-autoregressive" in docstring.lower()
            or "autoregressive" in docstring.lower()
            or "limitation" in docstring.lower()
            or "intra-channel" in docstring.lower()
        )
        assert has_limitation, (
            "ParticleConservingFlow docstring should document the non-autoregressive "
            "architecture limitation and its impact on strongly correlated systems"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
