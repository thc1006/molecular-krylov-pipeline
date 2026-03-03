"""Base class for Hamiltonian representations."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
import torch
import numpy as np


class Hamiltonian(ABC):
    """
    Abstract base class for Hamiltonian representations.

    Supports both sparse (connection-based) and dense representations.
    Used for both training (local energy) and SKQD (projected Hamiltonian).

    Attributes:
        num_sites: Number of sites/qubits
        hilbert_dim: Dimension of Hilbert space (2^num_sites for qubits)
    """

    def __init__(self, num_sites: int, local_dim: int = 2):
        self.num_sites = num_sites
        self.local_dim = local_dim
        self.hilbert_dim = local_dim ** num_sites

    @abstractmethod
    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal matrix element ⟨x|H|x⟩.

        Args:
            config: Configuration, shape (num_sites,)

        Returns:
            Diagonal element (scalar)
        """
        pass

    @abstractmethod
    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get off-diagonal connections for a configuration.

        Returns connected configurations x' where ⟨x|H|x'⟩ ≠ 0,
        along with the corresponding matrix elements.

        Args:
            config: Configuration, shape (num_sites,)

        Returns:
            (connected_configs, matrix_elements):
                connected_configs: shape (n_connected, num_sites)
                matrix_elements: shape (n_connected,)
        """
        pass

    def matrix_element(
        self,
        config_i: torch.Tensor,
        config_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute single matrix element ⟨x_i|H|x_j⟩.

        Args:
            config_i: Bra configuration
            config_j: Ket configuration

        Returns:
            Matrix element (scalar)
        """
        # Check if diagonal
        if torch.all(config_i == config_j):
            return self.diagonal_element(config_i)

        # Check if connected
        connected, elements = self.get_connections(config_j)

        if len(connected) == 0:
            return torch.tensor(0.0, device=config_i.device)

        # Find matching configuration
        matches = torch.all(connected == config_i.unsqueeze(0), dim=1)

        if matches.any():
            idx = torch.where(matches)[0][0]
            return elements[idx]
        else:
            return torch.tensor(0.0, device=config_i.device)

    def matrix_elements(
        self,
        configs_bra: torch.Tensor,
        configs_ket: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matrix of elements H_ij = ⟨x_i|H|x_j⟩.

        Args:
            configs_bra: Bra configurations, shape (n_bra, num_sites)
            configs_ket: Ket configurations, shape (n_ket, num_sites)

        Returns:
            Matrix H, shape (n_bra, n_ket)
        """
        n_bra = configs_bra.shape[0]
        n_ket = configs_ket.shape[0]
        device = configs_bra.device

        H = torch.zeros(n_bra, n_ket, device=device)

        for j, config_j in enumerate(configs_ket):
            # Diagonal
            for i in range(n_bra):
                if torch.all(configs_bra[i] == config_j):
                    H[i, j] = self.diagonal_element(config_j)
                    break

            # Off-diagonal connections
            connected, elements = self.get_connections(config_j)

            if len(connected) > 0:
                for k, (conn, elem) in enumerate(zip(connected, elements)):
                    # Find if conn is in configs_bra
                    matches = torch.all(configs_bra == conn.unsqueeze(0), dim=1)
                    if matches.any():
                        i = torch.where(matches)[0][0]
                        H[i, j] = elem

        return H

    def to_dense(self, device: str = "cpu") -> torch.Tensor:
        """
        Convert to dense matrix representation.

        Warning: Exponential in system size! Only for small systems.

        Returns:
            Dense Hamiltonian matrix, shape (hilbert_dim, hilbert_dim)
        """
        if self.num_sites > 16:
            raise ValueError(
                f"Dense conversion not recommended for {self.num_sites} sites "
                f"(Hilbert space dimension {self.hilbert_dim})"
            )

        # Generate all basis states
        basis = self._generate_all_configs(device)

        return self.matrix_elements(basis, basis)

    def _generate_all_configs(self, device: str = "cpu") -> torch.Tensor:
        """Generate all computational basis configurations."""
        configs = []
        for i in range(self.hilbert_dim):
            config = []
            val = i
            for _ in range(self.num_sites):
                config.append(val % self.local_dim)
                val //= self.local_dim
            configs.append(config[::-1])

        return torch.tensor(configs, device=device, dtype=torch.long)

    def exact_ground_state(
        self, device: str = "cpu"
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute exact ground state by diagonalization.

        Warning: Exponential complexity!

        Returns:
            (ground_state_energy, ground_state_vector)
        """
        H = self.to_dense(device).cpu().numpy()

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        return E0, torch.from_numpy(psi0).to(device)

    def ground_state_sparse(
        self,
        k: int = 1,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ground state using sparse eigensolver.

        More efficient for larger systems with sparse Hamiltonians.

        Args:
            k: Number of lowest eigenvalues to compute

        Returns:
            (eigenvalues, eigenvectors)
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        H_sparse = self.to_sparse(device)
        eigenvalues, eigenvectors = eigsh(H_sparse, k=k, which="SA")

        return eigenvalues, eigenvectors

    def to_sparse(self, device: str = "cpu"):
        """
        Convert to sparse CSR matrix representation.

        More memory efficient than to_dense() for larger systems.
        Uses scipy.sparse.csr_matrix format.

        Returns:
            scipy.sparse.csr_matrix
        """
        from scipy.sparse import csr_matrix

        # Build sparse matrix
        basis = self._generate_all_configs(device)
        n = self.hilbert_dim

        rows, cols, data = [], [], []

        for j in range(n):
            config_j = basis[j]

            # Diagonal
            diag = self.diagonal_element(config_j).item()
            rows.append(j)
            cols.append(j)
            data.append(diag)

            # Off-diagonal
            connected, elements = self.get_connections(config_j)
            for conn, elem in zip(connected, elements):
                # Find index of connected config
                i = self._config_to_index(conn)
                rows.append(i)
                cols.append(j)
                data.append(elem.item() if hasattr(elem, 'item') else elem)

        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.complex128)

    def _config_to_index(self, config: torch.Tensor) -> int:
        """Convert configuration to basis index."""
        idx = 0
        for i, val in enumerate(config):
            idx = idx * self.local_dim + val.item()
        return idx

    def _index_to_config(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Convert basis index to configuration."""
        config = []
        for _ in range(self.num_sites):
            config.append(idx % self.local_dim)
            idx //= self.local_dim
        return torch.tensor(config[::-1], device=device, dtype=torch.long)


class PauliString:
    """
    Representation of a Pauli string operator.

    A Pauli string is a tensor product of single-site Pauli operators:
        P = c * σ_1^{p_1} ⊗ σ_2^{p_2} ⊗ ... ⊗ σ_n^{p_n}

    where p_i ∈ {I, X, Y, Z} and c is a coefficient.
    """

    # Pauli matrices
    PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(
        self,
        paulis: List[str],
        coefficient: complex = 1.0,
    ):
        """
        Initialize Pauli string.

        Args:
            paulis: List of Pauli operators, one per site ('I', 'X', 'Y', 'Z')
            coefficient: Coefficient c
        """
        self.paulis = paulis
        self.coefficient = coefficient
        self.num_sites = len(paulis)

    def apply(
        self,
        config: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], complex]:
        """
        Apply Pauli string to configuration.

        Returns the resulting configuration and coefficient.
        Returns (None, 0) if the result vanishes.

        Args:
            config: Input configuration {0, 1}^n

        Returns:
            (new_config, coefficient) or (None, 0)
        """
        new_config = config.clone()
        coeff = self.coefficient

        for i, pauli in enumerate(self.paulis):
            if pauli == "I":
                continue
            elif pauli == "X":
                new_config[i] = 1 - new_config[i]
            elif pauli == "Y":
                # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                if config[i] == 0:
                    coeff *= 1j
                else:
                    coeff *= -1j
                new_config[i] = 1 - new_config[i]
            elif pauli == "Z":
                # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                if config[i] == 1:
                    coeff *= -1

        return new_config, coeff

    def is_diagonal(self) -> bool:
        """Check if Pauli string is diagonal (only I and Z)."""
        return all(p in ["I", "Z"] for p in self.paulis)

    def __repr__(self) -> str:
        return f"PauliString({self.coefficient} * {''.join(self.paulis)})"
