"""Tests for cc-pVDZ basis set and CAS active space support."""

import os
import sys
from math import comb

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_no_pyscf():
    """Skip test if PySCF is not installed."""
    try:
        import pyscf  # noqa: F401
    except ImportError:
        pytest.skip("PySCF not available")


# ---------------------------------------------------------------------------
# Tests: basis parameter on existing factory functions
# ---------------------------------------------------------------------------


@pytest.mark.molecular
class TestBasisParameter:
    """Existing factory functions should accept an optional `basis` kwarg."""

    def test_h2_ccpvdz_more_orbitals_than_sto3g(self):
        """cc-pVDZ H2 should have more orbitals than STO-3G."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_h2_hamiltonian

        h2_sto = create_h2_hamiltonian(bond_length=0.74, device="cpu")
        h2_dz = create_h2_hamiltonian(bond_length=0.74, basis="cc-pvdz", device="cpu")

        # STO-3G H2: 2 orbitals. cc-pVDZ H2: 10 orbitals.
        assert h2_sto.n_orbitals == 2
        assert h2_dz.n_orbitals == 10
        assert h2_dz.n_orbitals > h2_sto.n_orbitals

    def test_lih_basis_parameter_works(self):
        """Factory function accepts basis parameter without error."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_lih_hamiltonian

        H = create_lih_hamiltonian(bond_length=1.6, basis="cc-pvdz", device="cpu")
        # cc-pVDZ LiH: 28 orbitals (Li: 14, H: 5 -> 14+5=19... actually
        # Li: 3s2p1d -> 9 functions, H: 2s1p -> 5 functions => 14 AOs total
        # But exact count depends on PySCF. Just check it is > STO-3G (6).
        assert H.n_orbitals > 6

    def test_basis_default_is_sto3g(self):
        """Default basis should still be sto-3g for backward compatibility."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_h2_hamiltonian

        h2_default = create_h2_hamiltonian(bond_length=0.74, device="cpu")
        h2_explicit = create_h2_hamiltonian(bond_length=0.74, basis="sto-3g", device="cpu")

        # Both should have same number of orbitals
        assert h2_default.n_orbitals == h2_explicit.n_orbitals
        assert h2_default.n_orbitals == 2  # STO-3G H2 = 2 MOs

    def test_n2_basis_parameter_works(self):
        """N2 factory accepts basis parameter."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_n2_hamiltonian

        H = create_n2_hamiltonian(bond_length=1.10, basis="sto-3g", device="cpu")
        assert H.n_orbitals == 10  # STO-3G N2

    def test_h2o_basis_parameter_works(self):
        """H2O factory accepts basis parameter."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_h2o_hamiltonian

        H = create_h2o_hamiltonian(basis="sto-3g", device="cpu")
        assert H.n_orbitals == 7  # STO-3G H2O

    def test_beh2_basis_parameter_works(self):
        """BeH2 factory accepts basis parameter."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_beh2_hamiltonian

        H = create_beh2_hamiltonian(basis="sto-3g", device="cpu")
        assert H.n_orbitals == 7  # STO-3G BeH2

    def test_nh3_basis_parameter_works(self):
        """NH3 factory accepts basis parameter."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_nh3_hamiltonian

        H = create_nh3_hamiltonian(basis="sto-3g", device="cpu")
        assert H.n_orbitals == 8  # STO-3G NH3

    def test_ch4_basis_parameter_works(self):
        """CH4 factory accepts basis parameter."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_ch4_hamiltonian

        H = create_ch4_hamiltonian(basis="sto-3g", device="cpu")
        assert H.n_orbitals == 9  # STO-3G CH4


# ---------------------------------------------------------------------------
# Tests: CAS integral computation
# ---------------------------------------------------------------------------


@pytest.mark.molecular
class TestCASIntegrals:
    """Tests for CAS (Complete Active Space) integral support."""

    def test_cas_reduces_orbital_count(self):
        """CAS(10,8) on N2/cc-pVDZ should give 8 orbitals, not 28."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
        integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))
        assert integrals.n_orbitals == 8

    def test_cas_h1e_shape(self):
        """h1e should be (ncas, ncas)."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
        integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))
        assert integrals.h1e.shape == (8, 8)

    def test_cas_h2e_shape(self):
        """h2e should be (ncas, ncas, ncas, ncas)."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
        integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))
        assert integrals.h2e.shape == (8, 8, 8, 8)

    def test_cas_electron_count(self):
        """CAS should report correct active electron counts."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
        # CAS(10, 8): 10 active electrons (5 alpha, 5 beta)
        integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))
        assert integrals.n_electrons == 10
        assert integrals.n_alpha == 5
        assert integrals.n_beta == 5

    def test_cas_energy_includes_core(self):
        """nuclear_repulsion should include frozen core energy, not just bare Enuc."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]

        # Full (no CAS) nuclear repulsion
        full_integrals = compute_molecular_integrals(geometry, basis="cc-pvdz")
        nuc_repulsion = full_integrals.nuclear_repulsion

        # CAS nuclear_repulsion = e_core = Enuc + frozen core electron energy
        cas_integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))
        # CAS(10,8) with 14 electrons: 2 core orbitals frozen.
        # e_core includes kinetic + nuclear attraction + Coulomb/exchange of
        # frozen core, so it is NOT equal to bare nuclear repulsion.
        assert cas_integrals.nuclear_repulsion != pytest.approx(nuc_repulsion, abs=1e-6)
        # The frozen core contribution is large and negative for N2
        # (2 core electrons in 1s orbitals), so e_core < Enuc
        assert cas_integrals.nuclear_repulsion < nuc_repulsion

    @pytest.mark.slow
    def test_cas_n2_small_fci_matches_casscf(self):
        """FCI on CAS(10,8) integrals should match PySCF CASSCF energy.

        This is the golden test: our pipeline extracts integrals from
        CASSCF-optimized orbitals, so diagonalizing in the active space
        must reproduce the CASSCF total energy. Tolerance: 0.1 mHa.
        """
        _skip_if_no_pyscf()
        from pyscf import gto, mcscf, scf

        from hamiltonians.molecular import MolecularHamiltonian, compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]

        # Get PySCF CASSCF reference energy
        mol = gto.Mole()
        mol.atom = geometry
        mol.basis = "cc-pvdz"
        mol.symmetry = True
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        mc = mcscf.CASSCF(mf, ncas=8, nelecas=10)
        mc.kernel()
        casscf_energy = mc.e_tot

        # Get CAS integrals from our pipeline (also runs CASSCF internally)
        cas_integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))

        # Build Hamiltonian and compute FCI on CAS subspace
        H = MolecularHamiltonian(cas_integrals, device="cpu")
        our_fci = H.fci_energy()

        # Should match CASSCF to within 0.1 mHa
        error_mha = abs(our_fci - casscf_energy) * 1000
        assert error_mha < 0.1, (
            f"FCI on CAS integrals ({our_fci:.8f}) differs from "
            f"PySCF CASSCF ({casscf_energy:.8f}) by {error_mha:.4f} mHa"
        )

    def test_cas_integrals_are_fp64(self):
        """CAS integrals should be float64 for numerical precision."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
        integrals = compute_molecular_integrals(geometry, basis="cc-pvdz", cas=(10, 8))
        assert integrals.h1e.dtype == np.float64
        assert integrals.h2e.dtype == np.float64

    def test_cas_without_cas_still_works(self):
        """cas=None (default) should produce full-space integrals unchanged."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import compute_molecular_integrals

        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]

        integrals_none = compute_molecular_integrals(geometry, basis="sto-3g", cas=None)
        integrals_default = compute_molecular_integrals(geometry, basis="sto-3g")

        assert integrals_none.n_orbitals == integrals_default.n_orbitals
        assert integrals_none.n_electrons == integrals_default.n_electrons
        np.testing.assert_allclose(integrals_none.h1e, integrals_default.h1e, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: CAS factory function
# ---------------------------------------------------------------------------


@pytest.mark.molecular
class TestCASFactory:
    """Tests for the create_n2_cas_hamiltonian factory function."""

    def test_create_n2_cas_hamiltonian_exists(self):
        """Factory function for N2/CAS should exist and work."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_n2_cas_hamiltonian

        H = create_n2_cas_hamiltonian(device="cpu")
        assert H is not None
        assert hasattr(H, "n_orbitals")
        assert hasattr(H, "n_alpha")
        assert hasattr(H, "n_beta")

    def test_n2_cas_10_8_config_space(self):
        """CAS(10,8): C(8,5)^2 = 3136 configs."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_n2_cas_hamiltonian

        H = create_n2_cas_hamiltonian(cas=(10, 8), device="cpu")
        expected_configs = comb(8, 5) ** 2
        assert expected_configs == 3136

        # Verify n_orbitals and electron count match
        assert H.n_orbitals == 8
        assert H.n_alpha == 5
        assert H.n_beta == 5

    def test_n2_cas_hamiltonian_diagonal(self):
        """Diagonal elements should be finite and negative (bound state)."""
        _skip_if_no_pyscf()
        import torch

        from hamiltonians.molecular import create_n2_cas_hamiltonian

        H = create_n2_cas_hamiltonian(cas=(10, 8), device="cpu")

        # Generate HF config (lowest orbitals occupied)
        n_orb = H.n_orbitals
        hf_config = torch.zeros(2 * n_orb, dtype=torch.long)
        for i in range(H.n_alpha):
            hf_config[i] = 1
        for i in range(H.n_beta):
            hf_config[n_orb + i] = 1

        diag_val = H.diagonal_element(hf_config)
        assert np.isfinite(diag_val.item()), "Diagonal element must be finite"
        assert diag_val.item() < 0, "N2 ground state energy must be negative"

    def test_n2_cas_default_basis_is_ccpvdz(self):
        """Default basis for CAS factory should be cc-pvdz."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_n2_cas_hamiltonian

        # Default cas=(10,8), basis="cc-pvdz"
        H = create_n2_cas_hamiltonian(device="cpu")
        assert H.n_orbitals == 8  # CAS(10,8) -> 8 active orbitals

    def test_n2_cas_custom_parameters(self):
        """Factory should accept custom bond_length, basis, cas."""
        _skip_if_no_pyscf()
        from hamiltonians.molecular import create_n2_cas_hamiltonian

        # CAS(6,6) on STO-3G for speed
        H = create_n2_cas_hamiltonian(bond_length=1.10, basis="sto-3g", cas=(6, 6), device="cpu")
        assert H.n_orbitals == 6
        assert H.n_alpha == 3
        assert H.n_beta == 3
