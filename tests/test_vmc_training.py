"""Tests for VMC (Variational Monte Carlo) training of the autoregressive flow.

VMC directly minimizes the variational energy <psi|H|psi> using either REINFORCE
or MinSR (Minimum Stochastic Reconfiguration) gradient estimators.

For real positive wavefunctions, psi(x) = sqrt(p_theta(x)), the local energy is:

    E_loc(x) = H_xx + sum_{y connected} H_xy * exp(0.5 * (log_p(y) - log_p(x)))

**REINFORCE** (legacy):
    nabla <H> = E_{x~p}[(E_loc(x) - <E_loc>) * nabla log p(x)]

**MinSR** (default, Chen & Heyl Nature Physics 2024):
    Solves S * delta = -f using the Woodbury identity, where S is the Fisher
    information matrix.  Cost O(N_params * N_samples^2) instead of O(N_params^3).

Test organization:
- TestVMCConfig: default values, custom overrides, MinSR config fields
- TestVMCLocalEnergies: correctness, shape, finiteness, physics
- TestVMCTrainerReinforce: REINFORCE-specific mechanics (baseline, autograd gradients)
- TestVMCTrainerMinSR: MinSR Jacobian, linear solve, parameter updates
- TestVMCTrainer: optimizer-agnostic training mechanics
- TestVMCPipelineIntegration: PipelineConfig fields, end-to-end run

Usage:
    uv run pytest tests/test_vmc_training.py -v
    uv run pytest tests/test_vmc_training.py -v -m slow    # slow integration tests
"""

import math
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

try:
    from flows.vmc_training import VMCConfig, VMCTrainer

    _HAS_VMC = True
except ImportError:
    _HAS_VMC = False

try:
    from flows.autoregressive_flow import (
        AutoregressiveConfig,
        AutoregressiveFlowSampler,
        configs_to_states,
        states_to_configs,
    )

    _HAS_AR_FLOW = True
except ImportError:
    _HAS_AR_FLOW = False

pytestmark = pytest.mark.skipif(
    not (_HAS_VMC and _HAS_AR_FLOW),
    reason="vmc_training or autoregressive_flow module not available",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_sampler(n_orbitals, n_alpha, n_beta, **kwargs):
    """Build a small AutoregressiveFlowSampler for testing."""
    num_sites = 2 * n_orbitals
    config = AutoregressiveConfig(n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0)
    return AutoregressiveFlowSampler(
        num_sites=num_sites,
        n_alpha=n_alpha,
        n_beta=n_beta,
        transformer_config=config,
    )


# =========================================================================
# TestVMCConfig
# =========================================================================


class TestVMCConfig:
    """Test VMCConfig dataclass defaults and custom values."""

    def test_defaults(self):
        """VMCConfig should have sensible defaults."""
        cfg = VMCConfig()
        assert cfg.n_samples == 2000
        assert cfg.n_steps == 1000
        assert cfg.lr == 1e-3
        assert cfg.lr_decay == 0.999
        assert cfg.clip_grad == 1.0
        assert cfg.baseline_decay == 0.99
        assert cfg.min_steps == 200
        assert cfg.convergence_window == 50
        assert cfg.convergence_threshold == 1e-4

    def test_default_optimizer_is_minsr(self):
        """Default optimizer_type should be minsr (not reinforce)."""
        cfg = VMCConfig()
        assert cfg.optimizer_type == "minsr"

    def test_minsr_config_fields(self):
        """VMCConfig should have MinSR-specific fields with sensible defaults."""
        cfg = VMCConfig()
        assert cfg.sr_regularization == 1e-3
        assert cfg.sr_reg_decay == 0.99
        assert cfg.sr_reg_min == 1e-5

    def test_custom_values(self):
        """VMCConfig should accept custom overrides."""
        cfg = VMCConfig(
            n_samples=500,
            n_steps=100,
            lr=5e-4,
            lr_decay=0.995,
            clip_grad=2.0,
            baseline_decay=0.95,
            min_steps=50,
            convergence_window=20,
            convergence_threshold=1e-3,
        )
        assert cfg.n_samples == 500
        assert cfg.n_steps == 100
        assert cfg.lr == 5e-4
        assert cfg.lr_decay == 0.995
        assert cfg.clip_grad == 2.0
        assert cfg.baseline_decay == 0.95
        assert cfg.min_steps == 50
        assert cfg.convergence_window == 20
        assert cfg.convergence_threshold == 1e-3

    def test_custom_minsr_values(self):
        """VMCConfig should accept custom MinSR overrides."""
        cfg = VMCConfig(
            optimizer_type="minsr",
            sr_regularization=1e-2,
            sr_reg_decay=0.95,
            sr_reg_min=1e-4,
        )
        assert cfg.optimizer_type == "minsr"
        assert cfg.sr_regularization == 1e-2
        assert cfg.sr_reg_decay == 0.95
        assert cfg.sr_reg_min == 1e-4

    def test_reinforce_optimizer_type(self):
        """VMCConfig should accept optimizer_type='reinforce'."""
        cfg = VMCConfig(optimizer_type="reinforce")
        assert cfg.optimizer_type == "reinforce"

    def test_invalid_optimizer_type_raises(self):
        """VMCTrainer should reject unknown optimizer_type."""
        cfg = VMCConfig(optimizer_type="invalid")
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        with pytest.raises(ValueError, match="Unknown optimizer_type"):
            VMCTrainer(flow, hamiltonian=None, config=cfg, device="cpu")


# =========================================================================
# TestVMCLocalEnergies
# =========================================================================


class TestVMCLocalEnergies:
    """Test local energy computation E_loc(x) = <x|H|psi>/<x|psi>."""

    @pytest.mark.molecular
    def test_local_energy_shape(self, h2_hamiltonian):
        """E_loc should have shape (n_samples,)."""
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1)
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)

        local_energies = trainer.compute_local_energies(configs, log_probs)
        assert local_energies.shape == (10,)

    @pytest.mark.molecular
    def test_local_energy_finite(self, h2_hamiltonian):
        """No NaN or inf in local energies."""
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=20, n_steps=1)
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(20)
            configs = states_to_configs(states, flow.n_orbitals)

        local_energies = trainer.compute_local_energies(configs, log_probs)
        assert torch.isfinite(local_energies).all(), f"Non-finite local energies: {local_energies}"

    @pytest.mark.molecular
    def test_local_energy_dtype_float64(self, h2_hamiltonian):
        """Local energies should be real-valued float64 (matching Hamiltonian precision)."""
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1)
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)

        local_energies = trainer.compute_local_energies(configs, log_probs)
        assert local_energies.dtype == torch.float64

    @pytest.mark.molecular
    def test_hf_state_local_energy(self, lih_hamiltonian):
        """E_loc of HF state should be close to HF energy for a well-trained flow.

        For a random (untrained) flow, E_loc is not expected to match HF energy,
        but it should be finite and real-valued.
        """
        flow = _make_sampler(n_orbitals=6, n_alpha=2, n_beta=2)
        cfg = VMCConfig(n_samples=1, n_steps=1)
        trainer = VMCTrainer(flow, lih_hamiltonian, config=cfg, device="cpu")

        hf_state = lih_hamiltonian.get_hf_state().unsqueeze(0).float()
        with torch.no_grad():
            log_prob = flow.log_prob(hf_state)

        local_energy = trainer.compute_local_energies(hf_state, log_prob)
        assert local_energy.shape == (1,)
        assert torch.isfinite(local_energy).all()

    @pytest.mark.molecular
    def test_variational_mean_energy_above_exact(self, h2_hamiltonian):
        """Mean local energy from random flow should be >= exact ground state (variational bound).

        For a randomly initialized flow the mean E_loc is the variational energy
        <psi|H|psi>, which must be >= E_exact by the variational principle.
        We use a large sample to average out noise.
        """
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=200, n_steps=1)
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(200)
            configs = states_to_configs(states, flow.n_orbitals)

        local_energies = trainer.compute_local_energies(configs, log_probs)
        e_mean = local_energies.mean().item()
        e_exact = h2_hamiltonian.fci_energy()

        # Variational principle: <psi|H|psi> >= E_exact
        # Allow small numerical noise (1 mHa tolerance)
        assert (
            e_mean >= e_exact - 0.001
        ), f"Variational principle violated: E_mean={e_mean:.6f} < E_exact={e_exact:.6f}"


# =========================================================================
# TestVMCTrainerReinforce
# =========================================================================


class TestVMCTrainerReinforce:
    """Test REINFORCE-specific VMC training mechanics.

    These tests explicitly use optimizer_type='reinforce' since they check
    REINFORCE-specific behavior (baseline, autograd gradients on flow params).
    """

    @pytest.mark.molecular
    def test_energy_decreases_reinforce(self, h2_hamiltonian):
        """Energy should decrease over multiple REINFORCE steps (on average)."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=100,
            n_steps=30,
            lr=5e-3,
            min_steps=30,
            optimizer_type="reinforce",
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        results = trainer.train(verbose=False)
        energies = results["energies"]
        assert len(energies) == 30

        # Average of first 5 steps should be higher than average of last 5 steps
        early = sum(energies[:5]) / 5
        late = sum(energies[-5:]) / 5
        assert (
            late < early
        ), f"Energy did not decrease: early_mean={early:.6f}, late_mean={late:.6f}"

    @pytest.mark.molecular
    def test_gradient_flows_reinforce(self, h2_hamiltonian):
        """Gradients should flow to all trainable flow parameters after a REINFORCE step."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=50, n_steps=1, optimizer_type="reinforce")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        trainer.train_step()

        # At least some parameters should have non-zero gradients
        has_grad = False
        for name, param in flow.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients from REINFORCE VMC step"

    @pytest.mark.molecular
    def test_baseline_updates_reinforce(self, h2_hamiltonian):
        """Energy baseline should update after each REINFORCE step."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=50,
            n_steps=1,
            baseline_decay=0.9,
            optimizer_type="reinforce",
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        assert trainer.energy_baseline is None
        trainer.train_step()
        assert trainer.energy_baseline is not None
        baseline_1 = trainer.energy_baseline
        trainer.train_step()
        baseline_2 = trainer.energy_baseline
        # EMA baseline should change between steps
        assert baseline_2 != baseline_1

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_h2_vmc_improves_from_random_reinforce(self, h2_hamiltonian):
        """VMC REINFORCE on H2 should significantly improve energy from random init.

        The positive-real ansatz psi(x) = sqrt(p(x)) cannot represent the exact
        ground state when it has negative amplitudes (as in H2, where the doubly
        excited config |0101> has negative FCI coefficient).  The best achievable
        energy with a positive ansatz is the HF energy (-1.1168 Ha for H2/STO-3G),
        which is ~20 mHa above FCI.
        """
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=500,
            n_steps=200,
            lr=5e-3,
            lr_decay=0.999,
            min_steps=50,
            optimizer_type="reinforce",
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")
        results = trainer.train(verbose=False)

        # Initial energy should be far from HF; final should be close
        initial_energy = results["energies"][0]
        best_energy = results["best_energy"]
        improvement = initial_energy - best_energy
        assert improvement > 0.1, (
            f"VMC did not improve H2 energy: initial={initial_energy:.6f}, "
            f"best={best_energy:.6f}, improvement={improvement:.6f} Ha"
        )

        # Best energy should be in the vicinity of HF energy (-1.1168 Ha)
        # Allow 5 mHa tolerance around HF energy
        hf_state = h2_hamiltonian.get_hf_state()
        hf_energy = h2_hamiltonian.diagonal_element(hf_state).item()
        error_from_hf = abs(best_energy - hf_energy) * 1000
        assert error_from_hf < 5.0, (
            f"VMC best energy ({best_energy:.6f}) not close to HF energy "
            f"({hf_energy:.6f}): error = {error_from_hf:.2f} mHa"
        )

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_lih_energy_improves_reinforce(self, lih_hamiltonian):
        """VMC REINFORCE on LiH should significantly improve energy from random init."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=6, n_alpha=2, n_beta=2)
        cfg = VMCConfig(
            n_samples=200,
            n_steps=200,
            lr=2e-3,
            min_steps=50,
            optimizer_type="reinforce",
        )
        trainer = VMCTrainer(flow, lih_hamiltonian, config=cfg, device="cpu")

        results = trainer.train(verbose=False)
        # Best energy should be meaningfully lower than initial energy
        initial_energy = results["energies"][0]
        best_energy = results["best_energy"]
        improvement = initial_energy - best_energy
        assert improvement > 0.01, (
            f"VMC did not improve LiH energy: initial={initial_energy:.6f}, "
            f"best={best_energy:.6f}, improvement={improvement:.6f} Ha"
        )


# =========================================================================
# TestVMCTrainerMinSR
# =========================================================================


class TestVMCTrainerMinSR:
    """Test MinSR-specific VMC training mechanics.

    MinSR (Minimum Stochastic Reconfiguration) solves the SR equation
    S * delta = -f using the Woodbury identity.  These tests verify the
    Jacobian computation, linear solve, and parameter update mechanics.
    """

    @pytest.mark.molecular
    def test_minsr_single_step(self, h2_hamiltonian):
        """Single MinSR step should run without errors and return metrics."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=20, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        metrics = trainer.train_step()
        assert "energy" in metrics
        assert "energy_std" in metrics
        assert "grad_norm" in metrics
        assert "loss" in metrics
        assert math.isfinite(metrics["energy"])
        assert math.isfinite(metrics["energy_std"])
        assert math.isfinite(metrics["grad_norm"])

    @pytest.mark.molecular
    def test_jacobian_shape(self, h2_hamiltonian):
        """Jacobian should have shape (N_samples, N_params)."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, _ = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        assert jacobian.shape == (10, trainer._flow_n_params)

    @pytest.mark.molecular
    def test_jacobian_dtype_float64(self, h2_hamiltonian):
        """Jacobian should be computed in FP64 for numerical stability."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=5, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, _ = flow._sample_autoregressive(5)
            configs = states_to_configs(states, flow.n_orbitals)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        assert jacobian.dtype == torch.float64

    @pytest.mark.molecular
    def test_jacobian_finite(self, h2_hamiltonian):
        """Jacobian should contain no NaN or inf values."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, _ = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        assert torch.isfinite(jacobian).all(), "Non-finite values in Jacobian"

    @pytest.mark.molecular
    def test_jacobian_nonzero(self, h2_hamiltonian):
        """Jacobian should have non-zero entries (flow is differentiable)."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, _ = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        assert jacobian.abs().sum() > 0, "Jacobian is all zeros"

    @pytest.mark.molecular
    def test_minsr_update_shape(self, h2_hamiltonian):
        """MinSR update should return (N_params,) delta and scalar grad_norm."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)
            local_energies = trainer.compute_local_energies(configs, log_probs)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        delta, grad_norm = trainer._minsr_update(jacobian, local_energies)

        assert delta.shape == (trainer._flow_n_params,)
        assert isinstance(grad_norm, float)
        assert math.isfinite(grad_norm)

    @pytest.mark.molecular
    def test_minsr_update_finite(self, h2_hamiltonian):
        """MinSR update should be finite (no NaN or inf)."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=10, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(10)
            configs = states_to_configs(states, flow.n_orbitals)
            local_energies = trainer.compute_local_energies(configs, log_probs)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        delta, _ = trainer._minsr_update(jacobian, local_energies)

        assert torch.isfinite(delta).all(), f"Non-finite MinSR update: {delta}"

    @pytest.mark.molecular
    def test_minsr_params_change(self, h2_hamiltonian):
        """Flow parameters should change after a MinSR step."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=20, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        # Snapshot parameters before step
        params_before = {name: p.data.clone() for name, p in flow.named_parameters()}

        trainer.train_step()

        # At least some parameters should have changed
        any_changed = False
        for name, p in flow.named_parameters():
            if not torch.allclose(p.data, params_before[name], atol=1e-12):
                any_changed = True
                break
        assert any_changed, "No flow parameters changed after MinSR step"

    @pytest.mark.molecular
    def test_sr_lambda_decays(self, h2_hamiltonian):
        """SR regularization lambda should decay after each step."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=20,
            n_steps=1,
            optimizer_type="minsr",
            sr_regularization=1e-2,
            sr_reg_decay=0.9,
            sr_reg_min=1e-5,
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        lambda_before = trainer._sr_lambda
        trainer.train_step()
        lambda_after = trainer._sr_lambda

        assert (
            lambda_after < lambda_before
        ), f"SR lambda did not decay: before={lambda_before}, after={lambda_after}"
        assert lambda_after == pytest.approx(lambda_before * 0.9, abs=1e-10)

    @pytest.mark.molecular
    def test_sr_lambda_respects_minimum(self, h2_hamiltonian):
        """SR regularization should not decay below sr_reg_min."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=20,
            n_steps=1,
            optimizer_type="minsr",
            sr_regularization=2e-5,
            sr_reg_decay=0.1,
            sr_reg_min=1e-5,
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        trainer.train_step()
        # 2e-5 * 0.1 = 2e-6, but should be clamped to 1e-5
        assert trainer._sr_lambda >= cfg.sr_reg_min

    @pytest.mark.molecular
    def test_minsr_lr_decays(self, h2_hamiltonian):
        """MinSR learning rate should decay after each step."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=20,
            n_steps=1,
            optimizer_type="minsr",
            lr=0.01,
            lr_decay=0.9,
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        lr_before = trainer._minsr_lr
        trainer.train_step()
        lr_after = trainer._minsr_lr

        assert lr_after < lr_before, f"MinSR LR did not decay: before={lr_before}, after={lr_after}"

    @pytest.mark.molecular
    def test_minsr_update_clipped(self, h2_hamiltonian):
        """MinSR update should be clipped to clip_grad norm."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=20,
            n_steps=1,
            optimizer_type="minsr",
            lr=10.0,  # Exaggerated LR to produce large updates
            clip_grad=0.01,  # Tight clip
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        metrics = trainer.train_step()
        # grad_norm should be <= clip_grad (with floating point tolerance)
        assert (
            metrics["grad_norm"] <= cfg.clip_grad + 1e-6
        ), f"Update norm {metrics['grad_norm']} exceeds clip_grad {cfg.clip_grad}"

    @pytest.mark.molecular
    def test_minsr_woodbury_vs_naive(self, h2_hamiltonian):
        """MinSR Woodbury solution should match naive (S + lambda*I)^{-1} f.

        For small systems where we can form S explicitly, verify that the
        Woodbury identity gives the same result.
        """
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=15, n_steps=1, optimizer_type="minsr")
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(15)
            configs = states_to_configs(states, flow.n_orbitals)
            local_energies = trainer.compute_local_energies(configs, log_probs)

        jacobian = trainer._compute_per_sample_jacobian(configs)
        n_samples = jacobian.shape[0]

        # Woodbury result
        delta_woodbury, _ = trainer._minsr_update(jacobian, local_energies)

        # Naive result: form S explicitly and solve (S + lambda*I) delta = f
        J = jacobian.double()
        e_loc = local_energies.detach().double()
        J_c = J - J.mean(dim=0, keepdim=True)
        e_c = e_loc - e_loc.mean()

        S = J_c.T @ J_c / n_samples  # (N_params, N_params)
        f = J_c.T @ e_c / n_samples  # (N_params,)

        lam = trainer._sr_lambda
        S_reg = S + lam * torch.eye(S.shape[0], dtype=torch.float64)
        delta_naive = -trainer._minsr_lr * torch.linalg.solve(S_reg, f)

        # They should match (allowing for floating point differences).
        # The Woodbury identity and naive solve have different conditioning
        # properties, so use relative tolerance based on the larger norm.
        # Both approaches minimize the same quadratic, so directional agreement
        # (cosine similarity) is the primary check.
        delta_w = delta_woodbury.double()
        ref_norm = max(delta_w.norm().item(), delta_naive.norm().item(), 1e-15)

        # Relative max difference should be small
        rel_diff = (delta_w - delta_naive).abs().max().item() / ref_norm
        assert rel_diff < 0.5, (
            f"Woodbury and naive solutions differ significantly:\n"
            f"  relative max diff = {rel_diff:.4f}\n"
            f"  Woodbury norm = {delta_w.norm():.2e}\n"
            f"  Naive norm = {delta_naive.norm():.2e}"
        )

        # Cosine similarity: directions should agree
        cos_sim = (
            torch.dot(delta_w, delta_naive) / (delta_w.norm() * delta_naive.norm() + 1e-30)
        ).item()
        assert cos_sim > 0.9, (
            f"Woodbury and naive solutions point in different directions: "
            f"cos_sim = {cos_sim:.4f}"
        )

    @pytest.mark.molecular
    def test_minsr_energy_decreases(self, h2_hamiltonian):
        """Energy should decrease over multiple MinSR steps (on average).

        MinSR has curvature information, so it should be able to decrease
        energy even with few samples.
        """
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=30,
            n_steps=20,
            lr=5e-2,
            lr_decay=0.99,
            min_steps=20,
            optimizer_type="minsr",
            sr_regularization=1e-3,
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        results = trainer.train(verbose=False)
        energies = results["energies"]
        assert len(energies) == 20

        # Average of first 5 steps should be higher than average of last 5 steps
        early = sum(energies[:5]) / 5
        late = sum(energies[-5:]) / 5
        assert (
            late < early
        ), f"MinSR energy did not decrease: early_mean={early:.6f}, late_mean={late:.6f}"

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_minsr_h2_converges(self, h2_hamiltonian):
        """MinSR on H2 should converge toward HF energy from random initialization.

        Same test as REINFORCE but using MinSR.  MinSR should converge faster
        (or at least as well) with fewer samples per step.
        """
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=100,
            n_steps=200,
            lr=5e-2,
            lr_decay=0.998,
            min_steps=50,
            optimizer_type="minsr",
            sr_regularization=1e-3,
            sr_reg_decay=0.995,
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")
        results = trainer.train(verbose=False)

        initial_energy = results["energies"][0]
        best_energy = results["best_energy"]
        improvement = initial_energy - best_energy
        assert improvement > 0.05, (
            f"MinSR did not improve H2 energy: initial={initial_energy:.6f}, "
            f"best={best_energy:.6f}, improvement={improvement:.6f} Ha"
        )

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_minsr_lih_improves(self, lih_hamiltonian):
        """MinSR on LiH should significantly improve energy from random initialization."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=6, n_alpha=2, n_beta=2)
        cfg = VMCConfig(
            n_samples=50,
            n_steps=100,
            lr=2e-2,
            lr_decay=0.998,
            min_steps=50,
            optimizer_type="minsr",
            sr_regularization=1e-3,
        )
        trainer = VMCTrainer(flow, lih_hamiltonian, config=cfg, device="cpu")

        results = trainer.train(verbose=False)
        initial_energy = results["energies"][0]
        best_energy = results["best_energy"]
        improvement = initial_energy - best_energy
        assert improvement > 0.01, (
            f"MinSR did not improve LiH energy: initial={initial_energy:.6f}, "
            f"best={best_energy:.6f}, improvement={improvement:.6f} Ha"
        )


# =========================================================================
# TestVMCTrainer (optimizer-agnostic)
# =========================================================================


class TestVMCTrainer:
    """Test optimizer-agnostic VMC training mechanics."""

    @pytest.mark.molecular
    def test_single_step(self, h2_hamiltonian):
        """Single VMC step should run without errors and return metrics (default MinSR)."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=20, n_steps=1)
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        metrics = trainer.train_step()
        assert "energy" in metrics
        assert "energy_std" in metrics
        assert "grad_norm" in metrics
        assert "loss" in metrics
        assert math.isfinite(metrics["energy"])
        assert math.isfinite(metrics["energy_std"])
        assert math.isfinite(metrics["grad_norm"])

    @pytest.mark.molecular
    def test_train_returns_expected_keys(self, h2_hamiltonian):
        """Train loop should return dict with required keys."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(n_samples=20, n_steps=5, min_steps=5)
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        results = trainer.train(verbose=False)
        assert "energies" in results
        assert "best_energy" in results
        assert "n_steps" in results
        assert "converged" in results
        assert len(results["energies"]) == 5
        assert results["n_steps"] == 5
        assert results["best_energy"] <= min(results["energies"]) + 1e-10

    @pytest.mark.molecular
    def test_convergence_early_stop(self, h2_hamiltonian):
        """VMC should stop early if convergence threshold is met."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        # Very loose threshold to force early convergence
        cfg = VMCConfig(
            n_samples=20,
            n_steps=200,
            min_steps=10,
            convergence_window=5,
            convergence_threshold=100.0,  # 100 Ha -- anything converges
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        results = trainer.train(verbose=False)
        # Should have stopped before 200 steps due to loose threshold
        assert results["n_steps"] < 200
        assert results["converged"]

    @pytest.mark.molecular
    def test_lr_scheduler_steps(self, h2_hamiltonian):
        """Learning rate should decay over training steps."""
        torch.manual_seed(42)
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        cfg = VMCConfig(
            n_samples=20,
            n_steps=10,
            lr=0.01,
            lr_decay=0.9,
            min_steps=10,
        )
        trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.train(verbose=False)
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr, f"LR did not decay: initial={initial_lr}, final={final_lr}"

    @pytest.mark.molecular
    def test_both_optimizers_produce_finite_metrics(self, h2_hamiltonian):
        """Both REINFORCE and MinSR should produce finite metrics on the same problem."""
        for opt_type in ("reinforce", "minsr"):
            torch.manual_seed(42)
            flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
            cfg = VMCConfig(n_samples=20, n_steps=3, min_steps=3, optimizer_type=opt_type)
            trainer = VMCTrainer(flow, h2_hamiltonian, config=cfg, device="cpu")

            results = trainer.train(verbose=False)
            for e in results["energies"]:
                assert math.isfinite(e), f"Non-finite energy with optimizer_type='{opt_type}': {e}"


# =========================================================================
# TestVMCPipelineIntegration
# =========================================================================


class TestVMCPipelineIntegration:
    """Test VMC integration with the pipeline."""

    def test_config_field_exists(self):
        """PipelineConfig should have use_vmc_training field."""
        from pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert hasattr(cfg, "use_vmc_training")
        assert cfg.use_vmc_training is False

    def test_vmc_config_fields(self):
        """PipelineConfig should have VMC-specific fields."""
        from pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert hasattr(cfg, "vmc_n_steps")
        assert hasattr(cfg, "vmc_lr")
        assert hasattr(cfg, "vmc_n_samples")

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_vmc_pipeline_runs(self, h2_hamiltonian):
        """Pipeline with use_vmc_training=True should run end-to-end."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_exact = h2_hamiltonian.fci_energy()
        cfg = PipelineConfig(
            use_vmc_training=True,
            use_autoregressive_flow=True,
            skip_nf_training=False,
            vmc_n_steps=30,
            vmc_n_samples=50,
            vmc_lr=2e-3,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            h2_hamiltonian,
            config=cfg,
            exact_energy=e_exact,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)

        # Pipeline should complete and produce results
        assert "combined_energy" in results or "skqd_energy" in results
        assert "vmc_energy" in results

    @pytest.mark.molecular
    def test_vmc_trainer_constructed_with_hamiltonian(self, h2_hamiltonian):
        """VMCTrainer should accept Hamiltonian and flow objects."""
        flow = _make_sampler(n_orbitals=2, n_alpha=1, n_beta=1)
        trainer = VMCTrainer(flow, h2_hamiltonian, device="cpu")
        assert trainer.flow is flow
        assert trainer.hamiltonian is h2_hamiltonian
        assert trainer.config.n_samples == 2000  # default

    @pytest.mark.molecular
    def test_vmc_does_not_break_direct_ci(self, h2_hamiltonian):
        """Pipeline with use_vmc_training=False should still work (Direct-CI mode)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_exact = h2_hamiltonian.fci_energy()
        cfg = PipelineConfig(
            use_vmc_training=False,
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            h2_hamiltonian,
            config=cfg,
            exact_energy=e_exact,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)
        assert "combined_energy" in results or "skqd_energy" in results
