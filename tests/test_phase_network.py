"""
Tests for P2.1: Phase Network (e^{iφ}) replacing Sign Network.

The phase network outputs φ ∈ [0, 2π) so that ψ(x) = √p(x) × e^{iφ(x)}.
This subsumes the sign network (φ=0 → +1, φ=π → -1) and provides smooth
gradients everywhere, unlike tanh which has vanishing gradients at ±∞.

References: QiankunNet (Nature Comms 2025), VMC-SIGN-PROBLEM-RESEARCH.md §4.2.
"""

import sys
import os
import math
import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPhaseNetworkBasic:
    """Test PhaseNetwork architecture and output properties."""

    def test_output_range(self):
        """Phase values should be in [0, 2π)."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=10)
        configs = torch.rand(100, 10)
        phases = net(configs)

        assert phases.shape == (100,)
        assert (phases >= 0).all(), f"Min phase: {phases.min():.4f}"
        assert (phases < 2 * math.pi).all(), f"Max phase: {phases.max():.4f}"

    def test_phase_factor_on_unit_circle(self):
        """e^{iφ} should have magnitude 1."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=10)
        configs = torch.rand(50, 10)
        factors = net.phase_factor(configs)

        assert factors.dtype == torch.complex128
        magnitudes = torch.abs(factors)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-12)

    def test_auto_hidden_dims(self):
        """Hidden dims should auto-scale with system size."""
        from flows.sign_network import PhaseNetwork

        small = PhaseNetwork(num_sites=8)
        medium = PhaseNetwork(num_sites=16)
        large = PhaseNetwork(num_sites=30)

        n_small = sum(p.numel() for p in small.parameters())
        n_medium = sum(p.numel() for p in medium.parameters())
        n_large = sum(p.numel() for p in large.parameters())

        assert n_small < n_medium < n_large

    def test_gradient_exists(self):
        """Phase network should have valid gradients for backprop."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=10)
        configs = torch.rand(20, 10, requires_grad=False)
        phases = net(configs)
        loss = phases.sum()
        loss.backward()

        for p in net.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


class TestPhaseSubsumesSign:
    """Test that phase network can represent sign network functionality."""

    def test_phase_can_represent_plus_one(self):
        """φ=0 → e^{i0} = +1 (real part = 1)."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=4)
        # Force output near 0 by setting last layer bias very negative
        # (sigmoid(very_negative) → 0 → phase ≈ 0)
        with torch.no_grad():
            last_layer = list(net.net.children())[-1]
            last_layer.bias.fill_(-10.0)
            last_layer.weight.fill_(0.0)

        configs = torch.rand(10, 4)
        phases = net(configs)
        factors = net.phase_factor(configs)

        # Phase should be near 0, factor near +1
        assert torch.allclose(phases, torch.zeros_like(phases), atol=0.01)
        assert torch.allclose(factors.real.float(), torch.ones(10), atol=0.01)

    def test_phase_can_represent_minus_one(self):
        """φ=π → e^{iπ} = -1 (real part = -1)."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=4)
        # Force output near 0.5 by setting last layer bias to 0 and weight to 0
        # sigmoid(0) = 0.5 → phase = π
        with torch.no_grad():
            last_layer = list(net.net.children())[-1]
            last_layer.bias.fill_(0.0)
            last_layer.weight.fill_(0.0)

        configs = torch.rand(10, 4)
        phases = net(configs)
        factors = net.phase_factor(configs)

        # Phase should be near π, factor near -1
        assert torch.allclose(phases, torch.full_like(phases, math.pi), atol=0.01)
        assert torch.allclose(factors.real.float(), -torch.ones(10), atol=0.01)

    def test_phase_ratio_well_defined(self):
        """Phase ratio e^{i(φ(y)-φ(x))} is always well-defined (no div-by-zero)."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=6)
        x = torch.rand(5, 6)
        y = torch.rand(5, 6)

        phi_x = net(x)
        phi_y = net(y)
        phase_diff = phi_y - phi_x
        ratio = torch.cos(phase_diff)  # Real part of e^{i(φ(y)-φ(x))}

        assert not torch.isnan(ratio).any()
        assert not torch.isinf(ratio).any()
        assert (ratio.abs() <= 1.0 + 1e-6).all()


class TestPhaseGradientSmooth:
    """Test that phase network has smooth gradients."""

    def test_gradient_no_vanishing(self):
        """Phase gradient should not vanish (unlike tanh at ±∞)."""
        from flows.sign_network import PhaseNetwork, SignNetwork

        torch.manual_seed(42)
        n_sites = 10
        configs = torch.rand(50, n_sites)

        # Phase network gradient magnitude
        phase_net = PhaseNetwork(n_sites)
        phases = phase_net(configs)
        phases.sum().backward()
        phase_grad_norm = sum(
            p.grad.norm().item() for p in phase_net.parameters() if p.grad is not None
        )

        # Sign network gradient magnitude
        sign_net = SignNetwork(n_sites)
        signs = sign_net(configs)
        signs.sum().backward()
        sign_grad_norm = sum(
            p.grad.norm().item() for p in sign_net.parameters() if p.grad is not None
        )

        # Both should have non-zero gradients
        assert phase_grad_norm > 0, "Phase gradient should be non-zero"
        assert sign_grad_norm > 0, "Sign gradient should be non-zero"

    def test_sigmoid_bounded_gradient(self):
        """Sigmoid derivative is bounded in [0, 0.25] — no explosion."""
        from flows.sign_network import PhaseNetwork

        net = PhaseNetwork(num_sites=6)
        # Test with extreme inputs
        configs = torch.cat([
            torch.zeros(10, 6),
            torch.ones(10, 6),
            torch.rand(10, 6),
        ])
        phases = net(configs)
        phases.sum().backward()

        for p in net.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "Gradient should not be NaN"
                assert not torch.isinf(p.grad).any(), "Gradient should not be Inf"


class TestPhaseNetworkWithVMC:
    """Test phase network integration with VMC trainer."""

    @pytest.fixture
    def h2_setup(self):
        """Set up H2 system with AR flow + phase network."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        from flows.sign_network import PhaseNetwork
        from flows.vmc_training import VMCConfig, VMCTrainer
        from hamiltonians.molecular import create_h2_hamiltonian

        H = create_h2_hamiltonian()
        flow = AutoregressiveFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1,
        )
        phase_net = PhaseNetwork(num_sites=4)
        config = VMCConfig(n_samples=100, n_steps=5, optimizer_type="minsr")
        trainer = VMCTrainer(flow, H, config=config, device="cpu", sign_network=phase_net)
        return trainer, H

    def test_vmc_detects_phase_network(self, h2_setup):
        """VMCTrainer should detect PhaseNetwork via _is_phase_network flag."""
        trainer, _ = h2_setup
        assert trainer._is_phase_network is True

    def test_vmc_with_phase_runs(self, h2_setup):
        """VMC training with phase network should complete without error."""
        trainer, _ = h2_setup
        results = trainer.train(verbose=False)
        assert "best_energy" in results
        assert not math.isnan(results["best_energy"])
        assert not math.isinf(results["best_energy"])

    def test_vmc_phase_energy_finite(self, h2_setup):
        """Local energies with phase network should be finite."""
        trainer, _ = h2_setup
        # Run a few steps
        results = trainer.train(verbose=False)
        energies = results.get("energies", [])
        assert len(energies) > 0
        for e in energies:
            assert math.isfinite(e), f"Non-finite energy: {e}"

    def test_sign_network_not_phase(self):
        """VMCTrainer with SignNetwork should have _is_phase_network=False."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        from flows.sign_network import SignNetwork
        from flows.vmc_training import VMCConfig, VMCTrainer
        from hamiltonians.molecular import create_h2_hamiltonian

        H = create_h2_hamiltonian()
        flow = AutoregressiveFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1,
        )
        sign_net = SignNetwork(num_sites=4)
        config = VMCConfig(n_samples=100, n_steps=3)
        trainer = VMCTrainer(flow, H, config=config, device="cpu", sign_network=sign_net)
        assert trainer._is_phase_network is False


class TestPhaseNetworkPipelineIntegration:
    """Test pipeline configuration for phase network."""

    def test_pipeline_config_sign_architecture(self):
        """PipelineConfig should have sign_architecture field."""
        from pipeline import PipelineConfig
        cfg = PipelineConfig()
        assert hasattr(cfg, "sign_architecture")
        assert cfg.sign_architecture == "phase"

    def test_pipeline_config_sign_option(self):
        """sign_architecture='sign' should be valid."""
        from pipeline import PipelineConfig
        cfg = PipelineConfig(sign_architecture="sign")
        assert cfg.sign_architecture == "sign"

    def test_pipeline_creates_phase_network(self):
        """Pipeline with sign_architecture='phase' creates PhaseNetwork."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        from flows.sign_network import PhaseNetwork
        from hamiltonians.molecular import create_h2_hamiltonian

        H = create_h2_hamiltonian()
        cfg = PipelineConfig(
            use_vmc_training=True,
            use_sign_network=True,
            sign_architecture="phase",
            skip_nf_training=True,
            use_autoregressive_flow=True,
        )
        pipe = FlowGuidedKrylovPipeline(H, config=cfg, auto_adapt=False)
        assert isinstance(pipe.sign_network, PhaseNetwork)

    def test_pipeline_creates_sign_network_legacy(self):
        """Pipeline with sign_architecture='sign' creates SignNetwork."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        from flows.sign_network import SignNetwork, PhaseNetwork
        from hamiltonians.molecular import create_h2_hamiltonian

        H = create_h2_hamiltonian()
        cfg = PipelineConfig(
            use_vmc_training=True,
            use_sign_network=True,
            sign_architecture="sign",
            skip_nf_training=True,
            use_autoregressive_flow=True,
        )
        pipe = FlowGuidedKrylovPipeline(H, config=cfg, auto_adapt=False)
        assert isinstance(pipe.sign_network, SignNetwork)
        assert not isinstance(pipe.sign_network, PhaseNetwork)
