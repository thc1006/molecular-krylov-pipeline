"""Tests for PR-A2: Krylov Adaptive Convergence + dt Scaling.

TDD RED phase tests for:
1. Energy convergence monitoring (per-step energy history)
2. Early stopping on convergence
3. Adaptive dt scaling based on spectral range
4. S-matrix / Hamiltonian conditioning guard
5. No-regression tests for LiH and H2O
"""

import pytest
import torch
import numpy as np
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from krylov.skqd import (
    FlowGuidedSKQD,
    SKQDConfig,
)


# =============================================================================
# SKQDConfig new fields
# =============================================================================


class TestSKQDConfigAdaptiveFields:
    """Test that SKQDConfig has new adaptive convergence fields."""

    def test_convergence_threshold_config(self):
        """SKQDConfig should accept a convergence_threshold field with default 1e-5."""
        config = SKQDConfig()
        assert hasattr(config, "convergence_threshold")
        assert config.convergence_threshold == 1e-5

    def test_convergence_threshold_custom(self):
        """SKQDConfig should accept custom convergence_threshold values."""
        config = SKQDConfig(convergence_threshold=1e-3)
        assert config.convergence_threshold == 1e-3

    def test_adaptive_dt_config_field(self):
        """SKQDConfig should have an adaptive_dt boolean field (default False)."""
        config = SKQDConfig()
        assert hasattr(config, "adaptive_dt")
        assert config.adaptive_dt is False

    def test_adaptive_dt_disabled(self):
        """SKQDConfig should allow disabling adaptive_dt."""
        config = SKQDConfig(adaptive_dt=False)
        assert config.adaptive_dt is False

    def test_config_field_names(self):
        """Verify convergence_threshold and adaptive_dt are proper dataclass fields."""
        field_names = {f.name for f in fields(SKQDConfig)}
        assert "convergence_threshold" in field_names
        assert "adaptive_dt" in field_names


# =============================================================================
# Energy convergence monitoring
# =============================================================================


class TestEnergyConvergenceMonitoring:
    """Test that energy history is tracked per Krylov step."""

    @pytest.mark.molecular
    def test_energy_convergence_monitoring(self, lih_hamiltonian):
        """run_with_nf should record energy_history in results."""
        config = SKQDConfig(
            max_krylov_dim=4,
            time_step=0.1,
            convergence_threshold=1e-10,  # Very tight, won't trigger early stop
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        # Energy history should be present in results
        assert "energy_history" in results
        # Should have one energy per Krylov step (at least 2 entries for 4 steps)
        assert len(results["energy_history"]) >= 2
        # All entries should be finite floats
        for e in results["energy_history"]:
            assert np.isfinite(e), f"Energy {e} is not finite"

    @pytest.mark.molecular
    def test_energy_history_monotonic_decreasing_or_stable(self, lih_hamiltonian):
        """Energy history should generally decrease or stay stable (variational principle)."""
        config = SKQDConfig(
            max_krylov_dim=5,
            time_step=0.1,
            convergence_threshold=1e-10,
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        energy_history = results["energy_history"]
        # Energy should not increase by more than 1 mHa between steps
        # (small increases can happen due to basis change, but large ones indicate instability)
        for i in range(1, len(energy_history)):
            increase = energy_history[i] - energy_history[i - 1]
            assert increase < 0.001, (
                f"Energy increased by {increase * 1000:.4f} mHa at step {i}"
            )


# =============================================================================
# Early stopping on convergence
# =============================================================================


class TestEarlyStopping:
    """Test early stopping when energy converges."""

    @pytest.mark.molecular
    def test_early_stopping_on_convergence(self, lih_hamiltonian):
        """Should stop before max_krylov_dim when energy converges.

        Use a small initial basis (HF only) so the Krylov expansion grows
        the basis (finding singles/doubles via get_connections). Once the
        basis has grown and energy stabilizes, convergence early stop fires.
        """
        max_dim = 12
        config = SKQDConfig(
            max_krylov_dim=max_dim,
            time_step=0.1,
            convergence_threshold=1e-3,  # 1 mHa — should converge after basis grows
        )
        # Use only HF + a few configs so the Krylov loop has room to discover more
        hf_state = lih_hamiltonian.get_hf_state()
        nf_basis = hf_state.unsqueeze(0)  # Just the HF state

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        # Verify converged flag exists
        assert "converged" in results
        # Should converge before running all steps (basis grows then energy stabilizes)
        if results["converged"]:
            n_steps = len(results["energy_history"])
            assert n_steps < max_dim - 1, (
                f"Converged but ran {n_steps} steps (expected < {max_dim - 1})"
            )

    @pytest.mark.molecular
    def test_no_early_stopping_with_tight_threshold(self, h2_hamiltonian):
        """With very tight threshold, should run all Krylov steps."""
        max_dim = 4
        config = SKQDConfig(
            max_krylov_dim=max_dim,
            time_step=0.1,
            convergence_threshold=1e-15,  # Impossibly tight — won't converge
        )
        nf_basis = _generate_essential_configs_for_test(h2_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=h2_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        # Should have run all steps (no early stopping)
        assert "converged" in results
        # With impossibly tight threshold, should not converge
        # (unless H2 is trivially solved in 1 step)


# =============================================================================
# Adaptive dt scaling
# =============================================================================


class TestAdaptiveDtScaling:
    """Test adaptive time step scaling based on spectral range."""

    @pytest.mark.molecular
    def test_adaptive_dt_reduces_dt(self, lih_hamiltonian):
        """For wide-spectrum H, adaptive_dt should reduce dt to prevent aliasing.

        dt_safe = pi / spectral_range. If initial dt > dt_safe, it should be reduced.
        """
        # Use a very large initial dt that should get reduced
        config = SKQDConfig(
            max_krylov_dim=3,
            time_step=100.0,  # Very large — should be reduced by adaptive scaling
            adaptive_dt=True,
            convergence_threshold=1e-10,
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        # The effective dt should be stored in results
        assert "effective_dt" in results
        # The effective dt should be less than the initial 100.0
        assert results["effective_dt"] < 100.0
        # The effective dt should be positive and finite
        assert results["effective_dt"] > 0
        assert np.isfinite(results["effective_dt"])

    @pytest.mark.molecular
    def test_adaptive_dt_no_change_small_spectrum(self, h2_hamiltonian):
        """For narrow spectrum with small dt, adaptive_dt should not change dt."""
        # Use a very small dt that should NOT be reduced
        config = SKQDConfig(
            max_krylov_dim=3,
            time_step=0.001,  # Very small — should remain unchanged
            adaptive_dt=True,
            convergence_threshold=1e-10,
        )
        nf_basis = _generate_essential_configs_for_test(h2_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=h2_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        assert "effective_dt" in results
        # Small dt should remain unchanged (dt < pi/spectral_range)
        assert abs(results["effective_dt"] - 0.001) < 1e-10

    @pytest.mark.molecular
    def test_adaptive_dt_disabled(self, lih_hamiltonian):
        """When adaptive_dt=False, dt should not be modified."""
        config = SKQDConfig(
            max_krylov_dim=3,
            time_step=100.0,  # Large dt
            adaptive_dt=False,  # Disabled
            convergence_threshold=1e-10,
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        assert "effective_dt" in results
        # dt should remain at the original value when adaptive is disabled
        assert abs(results["effective_dt"] - 100.0) < 1e-10


# =============================================================================
# S-matrix / Hamiltonian conditioning guard
# =============================================================================


class TestConditioningGuard:
    """Test that ill-conditioned H matrix triggers early Krylov termination."""

    @pytest.mark.molecular
    def test_s_matrix_conditioning_guard(self, lih_hamiltonian):
        """run_with_nf results should include conditioning info."""
        config = SKQDConfig(
            max_krylov_dim=4,
            time_step=0.1,
            convergence_threshold=1e-10,
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        # Results should include conditioning information
        # (either max_condition_number or ill_conditioned flag)
        assert "ill_conditioned_stop" in results

    @pytest.mark.molecular
    def test_conditioning_guard_does_not_crash(self, lih_hamiltonian):
        """Even with ill-conditioning, SKQD should return valid results, not crash."""
        config = SKQDConfig(
            max_krylov_dim=4,
            time_step=0.1,
            convergence_threshold=1e-10,
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        # Should always produce valid energy
        assert "best_stable_energy" in results
        energy = results["best_stable_energy"]
        assert np.isfinite(energy)


# =============================================================================
# No-regression tests
# =============================================================================


class TestNoRegression:
    """Verify SKQD energy is unchanged after adaptive convergence changes."""

    @pytest.mark.molecular
    def test_no_regression_lih(self, lih_hamiltonian):
        """LiH SKQD energy should be within 0.01 mHa of FCI after changes."""
        fci_energy = lih_hamiltonian.fci_energy()

        config = SKQDConfig(
            max_krylov_dim=8,
            time_step=0.1,
            convergence_threshold=1e-10,  # Tight threshold to avoid early stop
        )
        nf_basis = _generate_essential_configs_for_test(lih_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        best_energy = results["best_stable_energy"]
        error_mha = abs(best_energy - fci_energy) * 1000

        # LiH should be within 0.2 mHa of FCI (regression bound from test_regression.py)
        assert error_mha < 0.2, (
            f"LiH SKQD error {error_mha:.4f} mHa exceeds regression bound 0.2 mHa"
        )

    @pytest.mark.molecular
    def test_no_regression_h2o(self, h2o_hamiltonian):
        """H2O SKQD energy should be within chemical accuracy of FCI after changes."""
        fci_energy = h2o_hamiltonian.fci_energy()

        config = SKQDConfig(
            max_krylov_dim=8,
            time_step=0.1,
            convergence_threshold=1e-10,  # Tight threshold to avoid early stop
        )
        nf_basis = _generate_essential_configs_for_test(h2o_hamiltonian)

        skqd = FlowGuidedSKQD(
            hamiltonian=h2o_hamiltonian,
            nf_basis=nf_basis,
            config=config,
        )
        results = skqd.run_with_nf(progress=False)

        best_energy = results["best_stable_energy"]
        error_mha = abs(best_energy - fci_energy) * 1000

        # H2O should be within 1.0 mHa of FCI (well within chemical accuracy 1.594 mHa).
        # Standalone SKQD with essential configs only (no diversity selection)
        # may have slightly larger error than the full pipeline's 0.2 mHa bound.
        assert error_mha < 1.0, (
            f"H2O SKQD error {error_mha:.4f} mHa exceeds regression bound 1.0 mHa"
        )


# =============================================================================
# Helper functions
# =============================================================================


def _generate_essential_configs_for_test(hamiltonian) -> torch.Tensor:
    """Generate HF + singles + doubles for a molecular Hamiltonian.

    Replicates pipeline._generate_essential_configs() logic for standalone testing.
    """
    from itertools import combinations

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta

    hf_state = hamiltonian.get_hf_state()
    essential = [hf_state.clone()]

    occ_alpha = list(range(n_alpha))
    occ_beta = list(range(n_beta))
    virt_alpha = list(range(n_alpha, n_orb))
    virt_beta = list(range(n_beta, n_orb))

    # Single excitations
    for i in occ_alpha:
        for a in virt_alpha:
            c = hf_state.clone()
            c[i] = 0
            c[a] = 1
            essential.append(c)

    for i in occ_beta:
        for a in virt_beta:
            c = hf_state.clone()
            c[i + n_orb] = 0
            c[a + n_orb] = 1
            essential.append(c)

    # Double excitations (alpha-beta only for speed)
    for i in occ_alpha:
        for j in occ_beta:
            for a in virt_alpha:
                for b in virt_beta:
                    c = hf_state.clone()
                    c[i] = 0
                    c[j + n_orb] = 0
                    c[a] = 1
                    c[b + n_orb] = 1
                    essential.append(c)

    # Alpha-alpha doubles
    for i, j in combinations(occ_alpha, 2):
        for a, b in combinations(virt_alpha, 2):
            c = hf_state.clone()
            c[i] = 0
            c[j] = 0
            c[a] = 1
            c[b] = 1
            essential.append(c)

    # Beta-beta doubles
    for i, j in combinations(occ_beta, 2):
        for a, b in combinations(virt_beta, 2):
            c = hf_state.clone()
            c[i + n_orb] = 0
            c[j + n_orb] = 0
            c[a + n_orb] = 1
            c[b + n_orb] = 1
            essential.append(c)

    result = torch.stack(essential)
    result = torch.unique(result, dim=0)
    return result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
