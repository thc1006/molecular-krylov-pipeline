"""Regression test suite for the Flow-Guided Krylov pipeline.

PR 0.1: These tests establish a regression gate before any code changes.
All tests use Direct-CI (skip_nf_training=True) + SKQD mode, matching
the "Pure SKQD" column from RESULTS.md.

Reference baselines from RESULTS.md (Pure SKQD errors in mHa):
    H2:   0.0000
    LiH:  0.0114
    H2O:  0.0090
    BeH2: 0.0297
    NH3:  0.1345
    CH4:  0.2818
    N2:   0.0629

Chemical accuracy threshold: 1.594 mHa (1.0 kcal/mol).
All 7 systems PASS chemical accuracy with Pure SKQD.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from hamiltonians.molecular import MolecularHamiltonian

# Chemical accuracy in Hartree
CHEMICAL_ACCURACY_HA = 1.594e-3

# Regression bounds (mHa) — 10x the RESULTS.md values as safety margin,
# with a floor of 0.1 mHa for numerically exact results.
REGRESSION_BOUNDS_MHA = {
    "h2": 0.1,
    "lih": 0.2,
    "h2o": 0.2,
    "beh2": 0.5,
    "nh3": 2.0,
    "ch4": 4.0,
    "n2": 1.0,
}



def _run_pipeline(hamiltonian: MolecularHamiltonian) -> dict:
    """Run the pipeline in Direct-CI + SKQD mode."""
    fci_energy = hamiltonian.fci_energy()
    config = PipelineConfig(
        subspace_mode="skqd",
        skip_nf_training=True,
        device="cpu",
    )
    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian, config=config, exact_energy=fci_energy
    )
    results = pipeline.run(progress=False)
    return {
        "fci_energy": fci_energy,
        "pipeline_energy": results.get(
            "combined_energy",
            results.get("skqd_energy", results.get("nf_nqs_energy")),
        ),
        "results": results,
    }


# =============================================================================
# Hamiltonian construction tests (fast, no pipeline run)
# =============================================================================


class TestHamiltonianConstruction:
    """Verify Hamiltonian factory functions produce valid objects."""

    @pytest.mark.molecular
    def test_h2_construction(self, h2_hamiltonian):
        H = h2_hamiltonian
        assert H.n_orbitals == 2
        assert H.n_alpha == 1 and H.n_beta == 1
        assert H.num_sites == 4

    @pytest.mark.molecular
    def test_lih_construction(self, lih_hamiltonian):
        H = lih_hamiltonian
        assert H.n_orbitals == 6
        assert H.n_alpha == 2 and H.n_beta == 2
        assert H.num_sites == 12

    @pytest.mark.molecular
    def test_h2o_construction(self, h2o_hamiltonian):
        H = h2o_hamiltonian
        assert H.n_orbitals == 7
        assert H.num_sites == 14

    @pytest.mark.molecular
    def test_beh2_construction(self, beh2_hamiltonian):
        H = beh2_hamiltonian
        assert H.n_orbitals == 7
        assert H.num_sites == 14

    @pytest.mark.molecular
    def test_nh3_construction(self, nh3_hamiltonian):
        H = nh3_hamiltonian
        assert H.n_orbitals == 8
        assert H.num_sites == 16

    @pytest.mark.molecular
    def test_ch4_construction(self, ch4_hamiltonian):
        H = ch4_hamiltonian
        assert H.n_orbitals == 9
        assert H.num_sites == 18

    @pytest.mark.molecular
    def test_n2_construction(self, n2_hamiltonian):
        H = n2_hamiltonian
        assert H.n_orbitals == 10
        assert H.num_sites == 20


# =============================================================================
# FCI energy sanity checks
# =============================================================================


class TestFCIEnergy:
    """Verify FCI energies are physically reasonable."""

    @pytest.mark.molecular
    def test_h2_fci_negative(self, h2_hamiltonian):
        e = h2_hamiltonian.fci_energy()
        assert e < 0, f"H2 FCI energy should be negative, got {e}"

    @pytest.mark.molecular
    def test_lih_fci_negative(self, lih_hamiltonian):
        e = lih_hamiltonian.fci_energy()
        assert e < 0, f"LiH FCI energy should be negative, got {e}"

    @pytest.mark.molecular
    def test_fci_below_hf(self, h2_hamiltonian):
        """FCI energy must be <= HF energy (correlation is always negative)."""
        H = h2_hamiltonian
        hf_config = H.get_hf_state()
        e_hf = H.diagonal_element(hf_config).item()
        e_fci = H.fci_energy()
        assert e_fci <= e_hf + 1e-10, (
            f"FCI ({e_fci:.8f}) should be <= HF ({e_hf:.8f})"
        )


# =============================================================================
# Particle conservation checks
# =============================================================================


class TestParticleConservation:
    """Verify pipeline preserves particle number."""

    @pytest.mark.molecular
    def test_hf_config_particle_count(self, h2_hamiltonian):
        """HF configuration has correct electron count."""
        H = h2_hamiltonian
        hf = H.get_hf_state()
        n_orb = H.n_orbitals
        alpha_count = hf[:n_orb].sum().item()
        beta_count = hf[n_orb:].sum().item()
        assert alpha_count == H.n_alpha
        assert beta_count == H.n_beta

    @pytest.mark.molecular
    def test_lih_hf_particle_count(self, lih_hamiltonian):
        H = lih_hamiltonian
        hf = H.get_hf_state()
        n_orb = H.n_orbitals
        alpha_count = hf[:n_orb].sum().item()
        beta_count = hf[n_orb:].sum().item()
        assert alpha_count == H.n_alpha
        assert beta_count == H.n_beta


# =============================================================================
# Variational principle checks
# =============================================================================


class TestVariationalPrinciple:
    """Verify E_computed >= E_FCI (variational principle)."""

    @pytest.mark.molecular
    def test_h2_variational(self, h2_hamiltonian):
        """H2 full diag in 4-config space should exactly match FCI."""
        H = h2_hamiltonian
        e_fci = H.fci_energy()

        # All 4 configs for H2: C(2,1) x C(2,1) = 4
        # Alpha orbitals: sites 0,1. Beta orbitals: sites 2,3.
        configs = torch.tensor([
            [1, 0, 1, 0],  # alpha=0, beta=0
            [1, 0, 0, 1],  # alpha=0, beta=1
            [0, 1, 1, 0],  # alpha=1, beta=0
            [0, 1, 0, 1],  # alpha=1, beta=1
        ], dtype=torch.long)
        H_mat = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_mat = 0.5 * (H_mat + H_mat.T)
        eigenvalues = np.linalg.eigvalsh(H_mat)

        assert eigenvalues[0] >= e_fci - 1e-8, "Variational principle violated"
        assert abs(eigenvalues[0] - e_fci) < 1e-6, (
            f"Full CI should match FCI: {eigenvalues[0]:.8f} vs {e_fci:.8f}"
        )


# =============================================================================
# Pipeline regression tests (Direct-CI + SKQD)
# =============================================================================


class TestPipelineRegression:
    """Regression gates: pipeline energy within bounds of RESULTS.md baselines."""

    @pytest.mark.molecular
    def test_h2_regression(self, h2_hamiltonian):
        result = _run_pipeline(h2_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["h2"], (
            f"H2 error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['h2']} mHa"
        )

    @pytest.mark.molecular
    def test_lih_regression(self, lih_hamiltonian):
        result = _run_pipeline(lih_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["lih"], (
            f"LiH error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['lih']} mHa"
        )

    @pytest.mark.molecular
    def test_h2o_regression(self, h2o_hamiltonian):
        result = _run_pipeline(h2o_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["h2o"], (
            f"H2O error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['h2o']} mHa"
        )

    @pytest.mark.molecular
    def test_beh2_regression(self, beh2_hamiltonian):
        result = _run_pipeline(beh2_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["beh2"], (
            f"BeH2 error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['beh2']} mHa"
        )

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_nh3_regression(self, nh3_hamiltonian):
        result = _run_pipeline(nh3_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["nh3"], (
            f"NH3 error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['nh3']} mHa"
        )

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_ch4_regression(self, ch4_hamiltonian):
        result = _run_pipeline(ch4_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["ch4"], (
            f"CH4 error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['ch4']} mHa"
        )

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_n2_regression(self, n2_hamiltonian):
        result = _run_pipeline(n2_hamiltonian)
        error_mha = abs(result["pipeline_energy"] - result["fci_energy"]) * 1000
        assert error_mha < REGRESSION_BOUNDS_MHA["n2"], (
            f"N2 error {error_mha:.4f} mHa exceeds bound {REGRESSION_BOUNDS_MHA['n2']} mHa"
        )


# =============================================================================
# Chemical accuracy checks
# =============================================================================


class TestChemicalAccuracy:
    """All 7 systems must achieve chemical accuracy with Pure SKQD."""

    @pytest.mark.molecular
    def test_h2_chemical_accuracy(self, h2_hamiltonian):
        result = _run_pipeline(h2_hamiltonian)
        error_ha = abs(result["pipeline_energy"] - result["fci_energy"])
        assert error_ha < CHEMICAL_ACCURACY_HA, (
            f"H2 fails chemical accuracy: {error_ha*1000:.4f} mHa"
        )

    @pytest.mark.molecular
    def test_lih_chemical_accuracy(self, lih_hamiltonian):
        result = _run_pipeline(lih_hamiltonian)
        error_ha = abs(result["pipeline_energy"] - result["fci_energy"])
        assert error_ha < CHEMICAL_ACCURACY_HA, (
            f"LiH fails chemical accuracy: {error_ha*1000:.4f} mHa"
        )
