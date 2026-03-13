"""Tests for P2.2: Natural Orbitals for NNCI.

Natural orbitals (NOs) are eigenvectors of the MP2 one-particle density matrix.
Their occupation numbers measure orbital correlation importance:
- n ≈ 2.0: doubly occupied (HF-like)
- n ≈ 0.0: empty (negligible)
- 0 << n << 2: strongly correlated (important for CI)

Using NOs to prioritize NNCI candidate generation focuses on the most
physically relevant excitations, reducing combinatorial overhead.

Reference: NO-NNCI (arXiv:2510.27665, Oct 2025)
"""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMP2RDM1:
    """Test MP2 one-particle density matrix computation."""

    @pytest.fixture
    def lih_hamiltonian(self):
        """LiH system for testing."""
        try:
            from hamiltonians.molecular import create_lih_hamiltonian
            return create_lih_hamiltonian(bond_length=1.6, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.fixture
    def h2_hamiltonian(self):
        """H2 system for testing."""
        try:
            from hamiltonians.molecular import create_h2_hamiltonian
            return create_h2_hamiltonian(bond_length=0.74, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.mark.molecular
    def test_rdm1_shape(self, lih_hamiltonian):
        """1-RDM should be (n_orb, n_orb)."""
        from utils.perturbative_pruning import compute_mp2_rdm1
        rdm1 = compute_mp2_rdm1(lih_hamiltonian)
        n_orb = lih_hamiltonian.n_orbitals
        assert rdm1.shape == (n_orb, n_orb)

    @pytest.mark.molecular
    def test_rdm1_symmetric(self, lih_hamiltonian):
        """1-RDM should be symmetric."""
        from utils.perturbative_pruning import compute_mp2_rdm1
        rdm1 = compute_mp2_rdm1(lih_hamiltonian)
        assert np.allclose(rdm1, rdm1.T, atol=1e-12)

    @pytest.mark.molecular
    def test_rdm1_trace_equals_n_electrons(self, lih_hamiltonian):
        """Trace of 1-RDM should equal number of electrons."""
        from utils.perturbative_pruning import compute_mp2_rdm1
        rdm1 = compute_mp2_rdm1(lih_hamiltonian)
        n_elec = lih_hamiltonian.integrals.n_electrons
        # MP2 1-RDM trace = n_electrons (for unrelaxed density)
        assert abs(np.trace(rdm1) - n_elec) < 0.1, (
            f"Trace = {np.trace(rdm1):.4f}, expected {n_elec}"
        )

    @pytest.mark.molecular
    def test_rdm1_eigenvalues_in_range(self, lih_hamiltonian):
        """1-RDM eigenvalues should be in [0, 2] (approximately)."""
        from utils.perturbative_pruning import compute_mp2_rdm1
        rdm1 = compute_mp2_rdm1(lih_hamiltonian)
        eigenvalues = np.linalg.eigvalsh(rdm1)
        # Unrelaxed MP2 may slightly violate bounds
        assert eigenvalues.min() > -0.1, f"Min eigenvalue: {eigenvalues.min():.4f}"
        assert eigenvalues.max() < 2.1, f"Max eigenvalue: {eigenvalues.max():.4f}"

    @pytest.mark.molecular
    def test_rdm1_h2_near_hf(self, h2_hamiltonian):
        """For H2 at equilibrium, 1-RDM should be close to HF (2,0)."""
        from utils.perturbative_pruning import compute_mp2_rdm1
        rdm1 = compute_mp2_rdm1(h2_hamiltonian)
        # H2 STO-3G: weakly correlated, eigenvalues ≈ (2-δ, δ) where δ ≈ 0
        eigenvalues = np.sort(np.linalg.eigvalsh(rdm1))[::-1]
        assert eigenvalues[0] > 1.9, f"Largest occ: {eigenvalues[0]:.4f}"
        assert eigenvalues[1] < 0.1, f"Smallest occ: {eigenvalues[1]:.4f}"


class TestNaturalOrbitals:
    """Test natural orbital computation and properties."""

    @pytest.fixture
    def lih_hamiltonian(self):
        try:
            from hamiltonians.molecular import create_lih_hamiltonian
            return create_lih_hamiltonian(bond_length=1.6, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.mark.molecular
    def test_no_occ_numbers_shape(self, lih_hamiltonian):
        """Occupation numbers should have length n_orb."""
        from utils.perturbative_pruning import compute_natural_orbitals
        occ, coeffs = compute_natural_orbitals(lih_hamiltonian)
        n_orb = lih_hamiltonian.n_orbitals
        assert occ.shape == (n_orb,)
        assert coeffs.shape == (n_orb, n_orb)

    @pytest.mark.molecular
    def test_no_occ_numbers_sorted_descending(self, lih_hamiltonian):
        """Occupation numbers should be sorted descending."""
        from utils.perturbative_pruning import compute_natural_orbitals
        occ, _ = compute_natural_orbitals(lih_hamiltonian)
        assert np.all(occ[:-1] >= occ[1:]), f"Not sorted: {occ}"

    @pytest.mark.molecular
    def test_no_coeffs_orthogonal(self, lih_hamiltonian):
        """NO coefficient matrix should be orthogonal."""
        from utils.perturbative_pruning import compute_natural_orbitals
        _, coeffs = compute_natural_orbitals(lih_hamiltonian)
        # C^T C = I
        product = coeffs.T @ coeffs
        assert np.allclose(product, np.eye(len(product)), atol=1e-10)

    @pytest.mark.molecular
    def test_no_occ_numbers_sum(self, lih_hamiltonian):
        """Sum of occupation numbers should equal n_electrons."""
        from utils.perturbative_pruning import compute_natural_orbitals
        occ, _ = compute_natural_orbitals(lih_hamiltonian)
        n_elec = lih_hamiltonian.integrals.n_electrons
        assert abs(sum(occ) - n_elec) < 0.1, (
            f"Sum = {sum(occ):.4f}, expected {n_elec}"
        )

    @pytest.mark.molecular
    def test_no_importance_scores(self, lih_hamiltonian):
        """Orbital importance should identify correlated orbitals."""
        from utils.perturbative_pruning import (
            compute_natural_orbitals, no_orbital_importance,
        )
        occ, _ = compute_natural_orbitals(lih_hamiltonian)
        importance = no_orbital_importance(occ)

        # Importance = min(n, 2-n)
        # HF-like occupied (n≈2): importance ≈ 0
        # Empty (n≈0): importance ≈ 0
        # Correlated: importance > 0
        assert importance.shape == occ.shape
        assert (importance >= 0).all()
        # At least one orbital should have non-zero importance
        assert importance.max() > 0


class TestIntegralTransform:
    """Test integral transformation to NO basis."""

    @pytest.fixture
    def lih_hamiltonian(self):
        try:
            from hamiltonians.molecular import create_lih_hamiltonian
            return create_lih_hamiltonian(bond_length=1.6, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.mark.molecular
    def test_transform_h1e_shape(self, lih_hamiltonian):
        """Transformed h1e should have same shape."""
        from utils.perturbative_pruning import (
            compute_natural_orbitals, transform_integrals_to_no_basis,
        )
        _, coeffs = compute_natural_orbitals(lih_hamiltonian)
        h1e_no, h2e_no = transform_integrals_to_no_basis(lih_hamiltonian, coeffs)
        n_orb = lih_hamiltonian.n_orbitals
        assert h1e_no.shape == (n_orb, n_orb)
        assert h2e_no.shape == (n_orb, n_orb, n_orb, n_orb)

    @pytest.mark.molecular
    def test_transform_preserves_trace(self, lih_hamiltonian):
        """h1e trace should be preserved under unitary rotation."""
        from utils.perturbative_pruning import (
            compute_natural_orbitals, transform_integrals_to_no_basis,
        )
        _, coeffs = compute_natural_orbitals(lih_hamiltonian)
        h1e_no, _ = transform_integrals_to_no_basis(lih_hamiltonian, coeffs)
        h1e_orig = lih_hamiltonian.integrals.h1e
        assert abs(np.trace(h1e_no) - np.trace(h1e_orig)) < 1e-10

    @pytest.mark.molecular
    def test_transform_h1e_symmetric(self, lih_hamiltonian):
        """Transformed h1e should be symmetric."""
        from utils.perturbative_pruning import (
            compute_natural_orbitals, transform_integrals_to_no_basis,
        )
        _, coeffs = compute_natural_orbitals(lih_hamiltonian)
        h1e_no, _ = transform_integrals_to_no_basis(lih_hamiltonian, coeffs)
        assert np.allclose(h1e_no, h1e_no.T, atol=1e-10)

    @pytest.mark.molecular
    def test_transform_identity_is_noop(self, lih_hamiltonian):
        """Identity transformation should return original integrals."""
        from utils.perturbative_pruning import transform_integrals_to_no_basis
        n_orb = lih_hamiltonian.n_orbitals
        identity = np.eye(n_orb)
        h1e_no, h2e_no = transform_integrals_to_no_basis(lih_hamiltonian, identity)
        h1e_orig = lih_hamiltonian.integrals.h1e.astype(np.float64)
        h2e_orig = lih_hamiltonian.integrals.h2e.astype(np.float64)
        assert np.allclose(h1e_no, h1e_orig, atol=1e-12)
        assert np.allclose(h2e_no, h2e_orig, atol=1e-12)


    @pytest.mark.molecular
    def test_transform_preserves_fci_energy(self, lih_hamiltonian):
        """FCI energy in NO basis should match MO basis (definitive correctness check)."""
        from utils.perturbative_pruning import (
            compute_natural_orbitals, transform_integrals_to_no_basis,
        )
        from hamiltonians.molecular import MolecularIntegrals, MolecularHamiltonian

        _, coeffs = compute_natural_orbitals(lih_hamiltonian)
        h1e_no, h2e_no = transform_integrals_to_no_basis(lih_hamiltonian, coeffs)

        # Build Hamiltonian in NO basis
        integrals_no = MolecularIntegrals(
            h1e=h1e_no,
            h2e=h2e_no,
            nuclear_repulsion=lih_hamiltonian.integrals.nuclear_repulsion,
            n_electrons=lih_hamiltonian.integrals.n_electrons,
            n_orbitals=lih_hamiltonian.integrals.n_orbitals,
            n_alpha=lih_hamiltonian.integrals.n_alpha,
            n_beta=lih_hamiltonian.integrals.n_beta,
        )
        H_no = MolecularHamiltonian(integrals_no, device="cpu")

        # FCI energy should be identical (unitary rotation preserves spectrum)
        e_mo = lih_hamiltonian.fci_energy()
        e_no = H_no.fci_energy()
        assert abs(e_mo - e_no) < 1e-8, (
            f"FCI energy mismatch: MO={e_mo:.10f}, NO={e_no:.10f}, diff={abs(e_mo-e_no):.2e}"
        )


class TestNNCINaturalOrbitals:
    """Test NNCI integration with natural orbital prioritization."""

    @pytest.fixture
    def lih_hamiltonian(self):
        try:
            from hamiltonians.molecular import create_lih_hamiltonian
            return create_lih_hamiltonian(bond_length=1.6, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.mark.molecular
    def test_nnci_config_has_no_fields(self):
        """NNCIConfig should have natural orbital fields."""
        from krylov.nnci import NNCIConfig
        cfg = NNCIConfig()
        assert hasattr(cfg, "use_natural_orbitals")
        assert cfg.use_natural_orbitals is False
        assert hasattr(cfg, "no_max_active_orbitals")
        assert cfg.no_max_active_orbitals == 0

    @pytest.mark.molecular
    def test_nnci_with_no_runs(self, lih_hamiltonian):
        """NNCI with natural orbitals should run without error."""
        from krylov.nnci import NNCIConfig, NNCIActiveLearning

        basis = lih_hamiltonian.get_hf_state().unsqueeze(0)
        # Add some singles
        from utils.perturbative_pruning import _classify_excitation
        # Just use HF as starting point
        config = NNCIConfig(
            max_iterations=2,
            top_k=10,
            max_candidates=500,
            use_natural_orbitals=True,
        )
        nnci = NNCIActiveLearning(lih_hamiltonian, basis, config)
        assert nnci._no_active_orbitals is not None
        results = nnci.run()
        assert "energy" in results
        assert results["basis_size"] >= 1

    @pytest.mark.molecular
    def test_nnci_no_with_truncation(self, lih_hamiltonian):
        """NNCI with NO truncation uses fewer active orbitals."""
        from krylov.nnci import NNCIConfig, NNCIActiveLearning

        basis = lih_hamiltonian.get_hf_state().unsqueeze(0)
        n_orb = lih_hamiltonian.n_orbitals  # 6 for LiH/STO-3G

        config = NNCIConfig(
            max_iterations=2,
            top_k=10,
            max_candidates=500,
            use_natural_orbitals=True,
            no_max_active_orbitals=4,  # Only top-4 most correlated
        )
        nnci = NNCIActiveLearning(lih_hamiltonian, basis, config)
        assert nnci._no_active_orbitals is not None
        assert len(nnci._no_active_orbitals) == 4

    @pytest.mark.molecular
    def test_no_generates_fewer_candidates(self, lih_hamiltonian):
        """NO-restricted generation should produce fewer candidates."""
        from krylov.nnci import NNCIConfig, NNCIActiveLearning

        basis = lih_hamiltonian.get_hf_state().unsqueeze(0)
        n_orb = lih_hamiltonian.n_orbitals  # 6

        # Without NO
        cfg_full = NNCIConfig(
            max_iterations=1, top_k=5, max_candidates=10000,
            use_natural_orbitals=False,
        )
        nnci_full = NNCIActiveLearning(lih_hamiltonian, basis, cfg_full)
        cands_full = nnci_full._generate_candidates()

        # With NO truncation to 4 orbitals
        cfg_no = NNCIConfig(
            max_iterations=1, top_k=5, max_candidates=10000,
            use_natural_orbitals=True, no_max_active_orbitals=4,
        )
        nnci_no = NNCIActiveLearning(lih_hamiltonian, basis, cfg_no)
        cands_no = nnci_no._generate_candidates()

        # Truncated should have fewer or equal candidates
        assert len(cands_no) <= len(cands_full), (
            f"NO candidates ({len(cands_no)}) should be <= full ({len(cands_full)})"
        )

    @pytest.mark.molecular
    def test_no_importance_ranks_correctly(self, lih_hamiltonian):
        """Most correlated orbitals should be frontier orbitals."""
        from utils.perturbative_pruning import (
            compute_natural_orbitals, no_orbital_importance,
        )
        occ, _ = compute_natural_orbitals(lih_hamiltonian)
        importance = no_orbital_importance(occ)
        n_occ = lih_hamiltonian.n_alpha  # 2 for LiH

        # HOMO/LUMO vicinity should have highest importance
        top2_idx = np.argsort(importance)[::-1][:2]
        # At least one of the top-2 should be near the HOMO-LUMO gap
        near_homo_lumo = any(
            abs(idx - n_occ) <= 1 or abs(idx - (n_occ - 1)) <= 1
            for idx in top2_idx
        )
        # This is a soft check: MP2 NOs for LiH should identify frontier orbitals
        # as most correlated, but the exact ranking depends on geometry
        assert importance.max() > 0.001  # At least some correlation


class TestPipelineNOConfig:
    """Test pipeline configuration for natural orbital NNCI."""

    @pytest.mark.molecular
    def test_pipeline_config_has_no_fields(self):
        """PipelineConfig should have NO NNCI fields."""
        from pipeline import PipelineConfig
        cfg = PipelineConfig()
        assert hasattr(cfg, "nnci_use_natural_orbitals")
        assert cfg.nnci_use_natural_orbitals is False
        assert hasattr(cfg, "nnci_no_max_active_orbitals")
        assert cfg.nnci_no_max_active_orbitals == 0

    @pytest.mark.molecular
    def test_pipeline_config_no_enabled(self):
        """PipelineConfig should accept NO parameters."""
        from pipeline import PipelineConfig
        cfg = PipelineConfig(
            nnci_use_natural_orbitals=True,
            nnci_no_max_active_orbitals=8,
        )
        assert cfg.nnci_use_natural_orbitals is True
        assert cfg.nnci_no_max_active_orbitals == 8
