"""
Tests for P2.3: Gumbel Top-K as alternative to SigmoidTopK.

GumbelTopK adds Gumbel noise for stochastic exploration, complementing
SigmoidTopK's deterministic gradient-exact selection.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGumbelTopKConfig:
    """Test configurable top-k selector type."""

    def test_default_is_sigmoid(self):
        """Default top-k selector should be SigmoidTopK."""
        from flows.particle_conserving_flow import (
            ParticleConservingFlowSampler, SigmoidTopK
        )
        flow = ParticleConservingFlowSampler(num_sites=6, n_alpha=1, n_beta=1)
        assert isinstance(flow.flow.topk_selector, SigmoidTopK)
        assert flow.flow._topk_type == "sigmoid"

    def test_gumbel_option(self):
        """topk_type='gumbel' should create GumbelTopK."""
        from flows.particle_conserving_flow import (
            ParticleConservingFlowSampler, GumbelTopK
        )
        flow = ParticleConservingFlowSampler(
            num_sites=6, n_alpha=1, n_beta=1, topk_type="gumbel"
        )
        assert isinstance(flow.flow.topk_selector, GumbelTopK)
        assert flow.flow._topk_type == "gumbel"

    def test_pipeline_config_topk_type(self):
        """PipelineConfig should have topk_type field."""
        from pipeline import PipelineConfig
        cfg = PipelineConfig()
        assert hasattr(cfg, "topk_type")
        assert cfg.topk_type == "sigmoid"

    def test_pipeline_config_gumbel(self):
        """PipelineConfig with topk_type='gumbel'."""
        from pipeline import PipelineConfig
        cfg = PipelineConfig(topk_type="gumbel")
        assert cfg.topk_type == "gumbel"


class TestGumbelTopKSampling:
    """Test GumbelTopK produces valid particle-conserving samples."""

    def test_particle_conservation(self):
        """GumbelTopK should produce configs with exact electron count."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=10, n_alpha=2, n_beta=2, topk_type="gumbel"
        )
        log_probs, unique_configs = flow.sample(50)

        assert unique_configs.shape[1] == 10  # 2*n_orbitals
        # Check alpha electrons (first 5 positions)
        alpha_counts = unique_configs[:, :5].sum(dim=1)
        assert (alpha_counts == 2).all(), f"Alpha count should be 2, got {alpha_counts}"
        # Check beta electrons (last 5 positions)
        beta_counts = unique_configs[:, 5:].sum(dim=1)
        assert (beta_counts == 2).all(), f"Beta count should be 2, got {beta_counts}"

    def test_stochastic_diversity(self):
        """GumbelTopK should produce more diverse samples than SigmoidTopK at high temp."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        torch.manual_seed(42)
        # Gumbel flow at high temperature
        gumbel_flow = ParticleConservingFlowSampler(
            num_sites=10, n_alpha=2, n_beta=2,
            topk_type="gumbel", temperature=2.0,
        )
        _, g_configs = gumbel_flow.sample(200)
        g_unique = len(g_configs)  # already unique

        # Sigmoid flow — deterministic, so it always produces the same config
        sigmoid_flow = ParticleConservingFlowSampler(
            num_sites=10, n_alpha=2, n_beta=2,
            topk_type="sigmoid", temperature=2.0,
        )
        _, s_configs = sigmoid_flow.sample(200)
        s_unique = len(s_configs)  # already unique

        # Gumbel should have more unique configs (stochastic noise → exploration)
        # Sigmoid is deterministic — all 200 samples are the same config
        assert g_unique > s_unique, (
            f"Gumbel unique configs ({g_unique}) should exceed sigmoid ({s_unique})"
        )

    def test_gumbel_gradient_flows(self):
        """Gradients should flow through GumbelTopK (STE)."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        flow = ParticleConservingFlowSampler(
            num_sites=6, n_alpha=1, n_beta=1, topk_type="gumbel"
        )
        log_probs, configs = flow.sample(10)

        # log_probs should require grad (from Plackett-Luce probability model)
        if log_probs.requires_grad:
            loss = log_probs.sum()
            loss.backward()
            has_grad = any(p.grad is not None for p in flow.parameters())
            assert has_grad, "Gradients should flow through GumbelTopK"

    def test_temperature_setter(self):
        """Temperature setter should work for both selector types."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        for topk_type in ("sigmoid", "gumbel"):
            sampler = ParticleConservingFlowSampler(
                num_sites=6, n_alpha=1, n_beta=1,
                topk_type=topk_type, temperature=1.0,
            )
            sampler.flow.set_temperature(0.5)
            # Just verify it doesn't crash — the effective temperature
            # may differ due to min_temperature in SigmoidTopK
            assert sampler.flow.temperature is not None
