"""Tests for Hamiltonian implementations."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hamiltonians.spin import TransverseFieldIsing, HeisenbergHamiltonian
from hamiltonians.base import Hamiltonian


class TestTransverseFieldIsing:
    """Test cases for Transverse Field Ising model."""

    def test_construction(self):
        """Test basic construction."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)
        assert H.num_sites == 4
        assert H.V == 1.0
        assert H.h == 1.0

    def test_diagonal_element(self):
        """Test diagonal matrix elements."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.0, L=1, periodic=False)

        # All spins aligned: should give negative energy
        config_aligned = torch.tensor([0, 0, 0, 0])
        diag = H.diagonal_element(config_aligned)

        # -V * sum(sigma_i * sigma_j) for aligned spins
        # sigma_i = sigma_j = -1 for all zeros
        # E = -V * 3 * (-1)(-1) = -3
        assert abs(diag.item() + 3.0) < 1e-10

    def test_connections(self):
        """Test off-diagonal connections."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)

        config = torch.tensor([0, 0, 0, 0])
        connected, elements = H.get_connections(config)

        # Should have 4 connections (one per spin flip)
        assert len(connected) == 4
        assert len(elements) == 4

        # All elements should be -h = -1
        for elem in elements:
            assert abs(elem.item() + 1.0) < 1e-10

    def test_dense_matrix(self):
        """Test dense matrix construction for small system."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0, periodic=False)
        H_dense = H.to_dense()

        # Should be 2^4 x 2^4 = 16 x 16
        assert H_dense.shape == (16, 16)

        # Should be Hermitian
        assert torch.allclose(H_dense, H_dense.T, atol=1e-10)

    def test_exact_ground_state(self):
        """Test exact diagonalization."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.1, periodic=False)
        E0, psi0 = H.exact_ground_state()

        # Energy should be negative (ferromagnetic ground state)
        assert E0 < 0

        # Ground state should be normalized
        assert abs(torch.sum(torch.abs(psi0)**2) - 1.0) < 1e-10


class TestHeisenbergHamiltonian:
    """Test cases for Heisenberg model."""

    def test_construction(self):
        """Test basic construction."""
        H = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0)
        assert H.num_sites == 4

    def test_diagonal_antiferromagnetic(self):
        """Test diagonal for antiferromagnetic state."""
        H = HeisenbergHamiltonian(num_spins=4, Jx=0.0, Jy=0.0, Jz=1.0)

        # Néel state |0101⟩
        config = torch.tensor([0, 1, 0, 1])
        diag = H.diagonal_element(config)

        # ZZ interaction for alternating spins
        # Jz/4 * (σ_i^z σ_j^z) where σ = ±1
        # 3 bonds: (+1)(-1), (-1)(+1), (+1)(-1) = 3 * (-1) = -3
        # Energy = Jz/4 * (-3) = -0.75
        assert abs(diag.item() + 0.75) < 1e-10

    def test_exchange_connections(self):
        """Test XX+YY connections."""
        H = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=0.0)

        # State |0100⟩ - antiparallel spins at positions 1,2
        config = torch.tensor([0, 1, 0, 0])
        connected, elements = H.get_connections(config)

        # Should flip antiparallel pairs
        assert len(connected) > 0


class TestMatrixElements:
    """Test matrix element computation."""

    def test_matrix_elements_symmetric(self):
        """Test that H_ij = H_ji^* for Hermitian Hamiltonian."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)

        configs = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ])

        H_mat = H.matrix_elements(configs, configs)

        # Should be symmetric for real Hamiltonian
        assert torch.allclose(H_mat, H_mat.T, atol=1e-10)


class TestPauliExtraction:
    """Test Pauli string extraction."""

    def test_ising_paulis(self):
        """Test Pauli extraction for Ising model."""
        from hamiltonians.spin import extract_coeffs_and_paulis

        H = TransverseFieldIsing(num_spins=3, V=1.0, h=1.0, L=1, periodic=False)
        coeffs, paulis = extract_coeffs_and_paulis(H)

        # Should have ZZ terms and X terms
        assert len(coeffs) == len(paulis)
        assert any("ZZ" in p or "ZZI" in p or "IZZ" in p for p in paulis)
        assert any(p.count("X") == 1 for p in paulis)


class TestMolecularHamiltonian:
    """Test cases for Molecular Hamiltonian (second quantization + JW transform)."""

    @pytest.fixture
    def h2_hamiltonian(self):
        """Create H2 molecule Hamiltonian for testing."""
        try:
            from pyscf import gto, scf, ao2mo, fci
            from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
        except ImportError:
            pytest.skip("PySCF not available")

        # H2 molecule at equilibrium
        mol = gto.Mole()
        mol.atom = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
        mol.basis = "sto-3g"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        # Get integrals
        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        h2e = ao2mo.kernel(mol, mf.mo_coeff)
        h2e = ao2mo.restore(1, h2e, mol.nao)

        # Compute FCI energy for reference
        cisolver = fci.FCI(mf)
        e_fci, _ = cisolver.kernel()

        integrals = MolecularIntegrals(
            h1e=h1e,
            h2e=h2e,
            nuclear_repulsion=mol.energy_nuc(),
            n_electrons=mol.nelectron,
            n_orbitals=mol.nao,
            n_alpha=1,
            n_beta=1,
        )

        H = MolecularHamiltonian(integrals, device="cpu")
        return H, e_fci, mf.e_tot

    def test_hermitian_symmetry(self, h2_hamiltonian):
        """Test that H_ij = H_ji for real molecular Hamiltonian."""
        H, _, _ = h2_hamiltonian

        # Generate a few configurations manually
        configs = torch.tensor([
            [1, 0, 1, 0],  # HF state (alpha in 0, beta in 2)
            [1, 0, 0, 1],  # Single excitation
            [0, 1, 1, 0],  # Single excitation
            [0, 1, 0, 1],  # Double excitation
        ], dtype=torch.long)

        H_mat = H.matrix_elements(configs, configs)
        H_np = H_mat.cpu().numpy()

        # Should be symmetric
        assert np.allclose(H_np, H_np.T, atol=1e-10), \
            f"Matrix not symmetric! Max diff: {np.abs(H_np - H_np.T).max()}"

    def test_variational_principle(self, h2_hamiltonian):
        """Test that computed energy >= exact FCI energy (variational principle)."""
        H, e_fci, e_hf = h2_hamiltonian

        # Generate all valid configurations (small system, 2 orbitals, 2 electrons)
        # Alpha: 1 electron in 2 orbitals = C(2,1) = 2 configs
        # Beta: 1 electron in 2 orbitals = C(2,1) = 2 configs
        # Total: 4 configs
        configs = torch.tensor([
            [1, 0, 1, 0],  # alpha in 0, beta in 0+n_orb=2
            [1, 0, 0, 1],  # alpha in 0, beta in 1+n_orb=3
            [0, 1, 1, 0],  # alpha in 1, beta in 0+n_orb=2
            [0, 1, 0, 1],  # alpha in 1, beta in 1+n_orb=3
        ], dtype=torch.long)

        H_mat = H.matrix_elements(configs, configs)
        H_np = H_mat.cpu().numpy().astype(np.float64)

        # Symmetrize
        H_np = 0.5 * (H_np + H_np.T)

        # Diagonalize
        eigenvalues, _ = np.linalg.eigh(H_np)
        e_computed = eigenvalues[0]

        # Variational principle: E_computed >= E_exact
        assert e_computed >= e_fci - 1e-8, \
            f"Variational principle violated! Computed: {e_computed:.8f}, FCI: {e_fci:.8f}"

        # For full CI in small basis, should equal FCI exactly
        assert abs(e_computed - e_fci) < 1e-6, \
            f"Full CI energy doesn't match FCI! Computed: {e_computed:.8f}, FCI: {e_fci:.8f}"

    def test_jw_sign_consistency(self, h2_hamiltonian):
        """Test that JW signs are consistent: H[i,j] should equal H[j,i] before symmetrization."""
        H, _, _ = h2_hamiltonian

        # Create two configurations connected by a double excitation
        config1 = torch.tensor([1, 0, 1, 0], dtype=torch.long)  # HF
        config2 = torch.tensor([0, 1, 0, 1], dtype=torch.long)  # Double excitation

        # Get matrix element from config1 -> config2 direction
        connected_from_1, elements_from_1 = H.get_connections(config1)
        elem_1_to_2 = None
        for conn, elem in zip(connected_from_1, elements_from_1):
            if torch.all(conn == config2):
                elem_1_to_2 = elem.item()
                break

        # Get matrix element from config2 -> config1 direction
        connected_from_2, elements_from_2 = H.get_connections(config2)
        elem_2_to_1 = None
        for conn, elem in zip(connected_from_2, elements_from_2):
            if torch.all(conn == config1):
                elem_2_to_1 = elem.item()
                break

        # Both should be found (or both not found if not connected)
        if elem_1_to_2 is not None:
            assert elem_2_to_1 is not None, "Asymmetric connection found!"
            # For real Hamiltonian, H[i,j] = H[j,i]
            assert abs(elem_1_to_2 - elem_2_to_1) < 1e-10, \
                f"JW sign inconsistency! H[1,2]={elem_1_to_2}, H[2,1]={elem_2_to_1}"

    def test_hf_energy_matches(self, h2_hamiltonian):
        """Test that HF state diagonal element matches HF energy."""
        H, _, e_hf = h2_hamiltonian

        # HF configuration
        hf_config = torch.tensor([1, 0, 1, 0], dtype=torch.long)

        # Diagonal element
        e_diag = H.diagonal_element(hf_config).item()

        # Should match HF energy (nuclear repulsion is included)
        assert abs(e_diag - e_hf) < 1e-6, \
            f"HF diagonal energy mismatch! Computed: {e_diag:.8f}, PySCF: {e_hf:.8f}"


class TestJWSignDouble:
    """Specific tests for Jordan-Wigner sign in double excitations."""

    def test_jw_sign_example_1(self):
        """Test JW sign for a specific example configuration."""
        try:
            from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
        except ImportError:
            pytest.skip("Module not available")

        # Create a minimal Hamiltonian just to access JW sign method
        import numpy as np
        h1e = np.eye(3) * 0.1
        h2e = np.zeros((3, 3, 3, 3))

        integrals = MolecularIntegrals(
            h1e=h1e, h2e=h2e, nuclear_repulsion=0.0,
            n_electrons=2, n_orbitals=3, n_alpha=1, n_beta=1
        )
        H = MolecularHamiltonian(integrals, device="cpu")

        # Test case: config with sites 1 and 3 occupied
        # Operator: a+_0 a+_2 a_3 a_1 (annihilate 1,3; create 0,2)
        config = np.array([0, 1, 0, 1, 0, 0])
        p, r, q, s = 0, 2, 1, 3

        sign = H._jw_sign_double_np(config, p, r, q, s)

        # Manual calculation (corrected formula):
        # 1. a_q (q=1): count below 1 = config[:1].sum() = 0
        # 2. a_s (s=3): count below 3 - (1<3?1:0) = 1 - 1 = 0
        # 3. a+_r (r=2): count below 2 - (1<2?1:0) - (3<2?1:0) = 1 - 1 - 0 = 0
        # 4. a+_p (p=0): count below 0 - ... = 0
        # Total = 0, sign = +1

        assert sign == 1, f"JW sign should be +1, got {sign}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
