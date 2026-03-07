"""Tests for Hamiltonian implementations.

NOTE: Original tests referenced hamiltonians.spin (TransverseFieldIsing, HeisenbergHamiltonian)
      which no longer exists. The current codebase only has MolecularHamiltonian.
      These tests cover the actual molecular Hamiltonian API.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMolecularHamiltonian:
    """Test cases for Molecular Hamiltonian."""

    @pytest.fixture
    def h2_hamiltonian(self):
        try:
            from hamiltonians.molecular import create_h2_hamiltonian
        except ImportError:
            pytest.skip("PySCF not available")
        return create_h2_hamiltonian(bond_length=0.74, device="cpu")

    def test_construction(self, h2_hamiltonian):
        H = h2_hamiltonian
        assert H.n_orbitals == 2
        assert H.n_alpha == 1
        assert H.n_beta == 1
        assert H.num_sites == 4

    def test_hf_state(self, h2_hamiltonian):
        H = h2_hamiltonian
        hf = H.get_hf_state()
        assert hf.shape == (4,)
        assert hf[:H.n_orbitals].sum().item() == H.n_alpha
        assert hf[H.n_orbitals:].sum().item() == H.n_beta

    def test_diagonal_element(self, h2_hamiltonian):
        H = h2_hamiltonian
        hf = H.get_hf_state()
        e_diag = H.diagonal_element(hf)
        assert torch.isfinite(e_diag)
        assert e_diag.item() < 0  # H2 HF energy is negative

    def test_hermitian_symmetry(self, h2_hamiltonian):
        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        H_mat = H.matrix_elements(configs, configs)
        H_np = H_mat.cpu().numpy()
        assert np.allclose(H_np, H_np.T, atol=1e-10), \
            f"Matrix not symmetric! Max diff: {np.abs(H_np - H_np.T).max()}"

    def test_fci_energy(self, h2_hamiltonian):
        H = h2_hamiltonian
        e_fci = H.fci_energy()
        assert e_fci < 0
        # H2/STO-3G FCI energy should be around -1.137 Ha
        assert -1.2 < e_fci < -1.0

    def test_full_ci_matches_fci(self, h2_hamiltonian):
        """Full CI in complete basis should match PySCF FCI."""
        H = h2_hamiltonian
        e_fci = H.fci_energy()

        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        H_mat = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_mat = 0.5 * (H_mat + H_mat.T)
        eigenvalues = np.linalg.eigvalsh(H_mat)

        assert abs(eigenvalues[0] - e_fci) < 1e-6, \
            f"Full CI: {eigenvalues[0]:.8f} vs FCI: {e_fci:.8f}"

    def test_connections(self, h2_hamiltonian):
        H = h2_hamiltonian
        hf = H.get_hf_state()
        connected, elements = H.get_connections(hf)
        assert len(connected) > 0
        assert len(connected) == len(elements)

    def test_jw_sign_consistency(self, h2_hamiltonian):
        """H[i,j] should equal H[j,i] for real Hamiltonian."""
        H = h2_hamiltonian
        config1 = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        config2 = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        connected_from_1, elements_from_1 = H.get_connections(config1)
        elem_1_to_2 = None
        for conn, elem in zip(connected_from_1, elements_from_1):
            if torch.all(conn == config2):
                elem_1_to_2 = elem.item()
                break

        connected_from_2, elements_from_2 = H.get_connections(config2)
        elem_2_to_1 = None
        for conn, elem in zip(connected_from_2, elements_from_2):
            if torch.all(conn == config1):
                elem_2_to_1 = elem.item()
                break

        if elem_1_to_2 is not None:
            assert elem_2_to_1 is not None, "Asymmetric connection found!"
            assert abs(elem_1_to_2 - elem_2_to_1) < 1e-10, \
                f"JW sign inconsistency! H[1,2]={elem_1_to_2}, H[2,1]={elem_2_to_1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
