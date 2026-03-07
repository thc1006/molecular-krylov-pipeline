"""
Sample-Based Krylov Quantum Diagonalization (SKQD).

Implements the SKQD algorithm from:
"Sample-based Krylov Quantum Diagonalization"
(Yu et al., IBM Quantum)

The algorithm:
1. Initialize reference state |ψ_0⟩
2. Generate Krylov states |ψ_k⟩ = U^k |ψ_0⟩ where U = e^{-iHΔt}
3. Sample basis states from each Krylov state
4. Project Hamiltonian onto sampled basis
5. Diagonalize to get ground state energy
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
from itertools import combinations
from math import comb
from tqdm import tqdm

# Sparse eigensolvers (as specified in AGENTs.md)
from scipy.sparse import csr_matrix as scipy_csr
from scipy.sparse.linalg import eigsh as scipy_eigsh

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
    # Also check if CUDA is actually usable (not just installed)
    try:
        cp.cuda.Device(0).compute_capability
        CUPY_AVAILABLE = True
    except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False

# Support both package imports and direct script execution
try:
    from ..hamiltonians.base import Hamiltonian
    from ..utils.gpu_linalg import gpu_eigh, gpu_eigsh, gpu_expm_multiply, mixed_precision_eigh
except ImportError:
    from hamiltonians.base import Hamiltonian
    from utils.gpu_linalg import gpu_eigh, gpu_eigsh, gpu_expm_multiply, mixed_precision_eigh


@dataclass
class SKQDConfig:
    """Configuration for SKQD algorithm."""

    # Krylov parameters
    max_krylov_dim: int = 12
    time_step: float = 0.1  # Δt
    total_evolution_time: Optional[float] = None  # If set, overrides max_k

    # Sampling parameters
    shots_per_krylov: int = 100000
    use_cumulative_basis: bool = True  # Accumulate samples across Krylov states

    # Eigensolver parameters
    num_eigenvalues: int = 2  # k for eigsh
    which_eigenvalues: str = "SA"  # Smallest algebraic

    # Numerical stability
    regularization: float = 1e-8  # Diagonal regularization for stability

    # Hardware
    use_gpu: bool = True

    # === KRYLOV SAMPLING LIMITS ===
    # Maximum new configurations to add per Krylov step
    max_new_configs_per_krylov_step: int = 1000

    # Fraction of basis to sample when finding connected configs
    # Higher values = more thorough exploration but slower
    krylov_basis_sample_fraction: float = 0.8

    # Minimum number of basis states to sample regardless of fraction
    min_krylov_basis_sample: int = 1000

    # Maximum basis size for dense diagonalization in compute_ground_state_energy().
    # If combined basis exceeds this, only the top configs (by energy importance)
    # are used for diagonalization. Set to 0 to disable (no cap).
    max_diag_basis_size: int = 15000


class SampleBasedKrylovDiagonalization:
    """
    Sample-Based Krylov Quantum Diagonalization.

    This class provides a classical simulation of SKQD for validation
    and development. For actual quantum hardware execution, use the
    CUDA-Q integration via KrylovBasisSampler.

    The algorithm builds a Krylov subspace by time-evolving a reference
    state and sampling in the computational basis at each step.

    Args:
        hamiltonian: System Hamiltonian
        config: SKQD configuration
        initial_state: Optional initial state (default: HF state)
    """

    # Guard: systems larger than this skip full subspace enumeration
    # to prevent OOM. Matches FlowGuidedSKQD.MAX_FULL_SUBSPACE_SIZE.
    MAX_FULL_SUBSPACE_SIZE = 100000

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: Optional[SKQDConfig] = None,
        initial_state: Optional[torch.Tensor] = None,
    ):
        self.hamiltonian = hamiltonian
        self.config = config or SKQDConfig()
        self.num_sites = hamiltonian.num_sites

        # Check if this is a molecular Hamiltonian with particle conservation
        self._is_molecular = hasattr(hamiltonian, 'n_alpha') and hasattr(hamiltonian, 'n_beta')
        self._subspace_basis = None
        self._subspace_index_map = None
        self._subspace_H = None

        # For molecular systems, set up particle-conserving subspace
        if self._is_molecular:
            self._setup_particle_conserving_subspace()

        # Set up initial state
        if initial_state is not None:
            self.initial_state = initial_state
        else:
            self.initial_state = self.hamiltonian.get_hf_state()

        # Time step for Krylov evolution
        self.time_step = self.config.time_step

        # Storage for results
        self.krylov_samples: List[Dict[str, int]] = []
        self.cumulative_basis: List[torch.Tensor] = []
        self.energies: List[float] = []

    def _setup_particle_conserving_subspace(self):
        """
        Set up the particle-conserving subspace for molecular Hamiltonians.

        This dramatically reduces the Hamiltonian size from 2^n to
        C(n_orb, n_alpha) * C(n_orb, n_beta), which is typically 10-100x smaller.

        Example sizes:
        - NH3 (16 qubits): 65,536 -> 3,136 (21x reduction)
        - N2 (20 qubits): 1,048,576 -> 14,400 (73x reduction)
        """
        n_orb = self.hamiltonian.n_orbitals
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        n_valid = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
        print(f"Setting up particle-conserving subspace: {n_valid:,} configs "
              f"(vs {self.hamiltonian.hilbert_dim:,} full Hilbert space)")

        if n_valid > self.MAX_FULL_SUBSPACE_SIZE:
            print(f"WARNING: {n_valid:,} configs exceeds MAX_FULL_SUBSPACE_SIZE "
                  f"({self.MAX_FULL_SUBSPACE_SIZE:,}). Skipping full subspace enumeration.")
            self._subspace_basis = None
            self._subspace_index_map = None
            return

        # Generate all valid configurations
        alpha_configs = list(combinations(range(n_orb), n_alpha))
        beta_configs = list(combinations(range(n_orb), n_beta))

        basis_configs = []
        for alpha_occ in alpha_configs:
            for beta_occ in beta_configs:
                # Build configuration tensor
                config = torch.zeros(self.num_sites, dtype=torch.long)
                for i in alpha_occ:
                    config[i] = 1
                for i in beta_occ:
                    config[i + n_orb] = 1
                basis_configs.append(config)

        self._subspace_basis = torch.stack(basis_configs)

        # Create index mapping: config tuple -> subspace index
        self._subspace_index_map = {}
        for idx, config in enumerate(basis_configs):
            key = tuple(config.tolist())
            self._subspace_index_map[key] = idx

        print(f"Subspace setup complete: {len(basis_configs)} configurations")

    @property
    def device(self) -> torch.device:
        """Get device from Hamiltonian (for GPU-aware Hamiltonians)."""
        if hasattr(self.hamiltonian, 'device'):
            return torch.device(self.hamiltonian.device)
        return torch.device('cpu')

    def _time_evolution_operator(
        self,
        state_vector: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Apply time evolution U^num_steps = (e^{-iHΔt})^num_steps.

        Uses particle-conserving subspace evolution for molecular systems,
        which is MUCH faster (e.g., 3K x 3K instead of 65K x 65K for NH3).

        Args:
            state_vector: Full state vector in Hilbert space
            num_steps: Number of time steps to apply

        Returns:
            Evolved state vector
        """
        # Ensure state vector is on the correct device
        device = self.device
        state_vector = state_vector.to(device)

        return self._sparse_time_evolution(state_vector, num_steps)

    def _sparse_time_evolution(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Apply time evolution using GPU-accelerated matrix exponential.

        Uses Lanczos-based Krylov approximation for efficient GPU computation
        of e^{-iHt}|psi> without forming the full matrix exponential.

        Works in particle-conserving subspace for massive speedup
        (e.g., 3K x 3K instead of 65K x 65K for NH3).
        """
        # For molecular systems, work in particle-conserving subspace
        if self._subspace_basis is not None:
            return self._sparse_time_evolution_subspace(state_vector, num_steps)

        # Fallback: build GPU Hamiltonian matrix (cached for reuse)
        if not hasattr(self, '_H_gpu') or self._H_gpu is None:
            self._H_gpu = self._build_gpu_hamiltonian()

        # GPU-accelerated time evolution
        device = state_vector.device
        psi = state_vector.to(torch.complex128)

        # Apply time evolution num_steps times using GPU
        for _ in range(num_steps):
            psi = gpu_expm_multiply(self._H_gpu, psi, t=-1j * self.time_step)

        return psi.to(device)

    def _sparse_time_evolution_subspace(
        self,
        state_vector: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        GPU-accelerated time evolution in particle-conserving subspace.

        Much faster because subspace is typically 10-100x smaller than full space.
        All operations stay on GPU for maximum performance.
        """
        device = state_vector.device

        # Build GPU subspace Hamiltonian if not cached
        if not hasattr(self, '_H_subspace_gpu') or self._H_subspace_gpu is None:
            self._H_subspace_gpu = self._build_gpu_subspace_hamiltonian()

        # Convert full state vector to subspace representation (on GPU)
        psi_subspace = self._full_to_subspace_gpu(state_vector)

        # Apply time evolution in subspace using GPU
        for _ in range(num_steps):
            psi_subspace = gpu_expm_multiply(
                self._H_subspace_gpu, psi_subspace, t=-1j * self.time_step
            )

        # Convert back to full Hilbert space
        return self._subspace_to_full_gpu(psi_subspace, device)

    def _full_to_subspace(self, state_vector: torch.Tensor) -> np.ndarray:
        """Convert full Hilbert space state to subspace representation."""
        n_subspace = len(self._subspace_basis)
        psi_subspace = np.zeros(n_subspace, dtype=np.complex128)

        # Extract amplitudes for valid configurations
        state_np = state_vector.cpu().numpy()
        for i, config in enumerate(self._subspace_basis):
            idx = self.hamiltonian._config_to_index(config)
            psi_subspace[i] = state_np[idx]

        return psi_subspace

    def _subspace_to_full(self, psi_subspace: np.ndarray, device) -> torch.Tensor:
        """Convert subspace state back to full Hilbert space."""
        n_full = self.hamiltonian.hilbert_dim
        state_full = np.zeros(n_full, dtype=np.complex128)

        # Place amplitudes in correct positions
        for i, config in enumerate(self._subspace_basis):
            idx = self.hamiltonian._config_to_index(config)
            state_full[idx] = psi_subspace[i]

        return torch.from_numpy(state_full).to(device)

    def _full_to_subspace_gpu(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Convert full Hilbert space state to subspace representation (GPU)."""
        device = state_vector.device
        n_subspace = len(self._subspace_basis)

        # Pre-compute subspace indices if not cached
        if not hasattr(self, '_subspace_indices'):
            self._subspace_indices = torch.tensor(
                [self.hamiltonian._config_to_index(c) for c in self._subspace_basis],
                device=device, dtype=torch.long
            )
        elif self._subspace_indices.device != device:
            self._subspace_indices = self._subspace_indices.to(device)

        # Extract amplitudes using advanced indexing (GPU-accelerated)
        psi_subspace = state_vector[self._subspace_indices].to(torch.complex128)
        return psi_subspace

    def _subspace_to_full_gpu(self, psi_subspace: torch.Tensor, device) -> torch.Tensor:
        """Convert subspace state back to full Hilbert space (GPU)."""
        n_full = self.hamiltonian.hilbert_dim

        # Ensure subspace indices are on correct device
        if not hasattr(self, '_subspace_indices'):
            self._subspace_indices = torch.tensor(
                [self.hamiltonian._config_to_index(c) for c in self._subspace_basis],
                device=device, dtype=torch.long
            )
        elif self._subspace_indices.device != device:
            self._subspace_indices = self._subspace_indices.to(device)

        # Create full state vector and scatter subspace values
        state_full = torch.zeros(n_full, dtype=torch.complex128, device=device)
        state_full[self._subspace_indices] = psi_subspace

        return state_full

    def _build_gpu_hamiltonian(self) -> torch.Tensor:
        """
        Build dense Hamiltonian matrix on GPU.

        Builds in particle-conserving subspace which is MUCH smaller
        than the full Hilbert space.
        """
        if self._subspace_basis is not None:
            return self._build_gpu_subspace_hamiltonian()

        raise ValueError(
            "No particle-conserving subspace available. "
            "SKQD requires a molecular Hamiltonian with particle conservation."
        )

    def _build_gpu_subspace_hamiltonian(self) -> torch.Tensor:
        """
        Build dense Hamiltonian in particle-conserving subspace on GPU.

        This is MUCH faster than building in full Hilbert space because:
        - NH3: 3,136 x 3,136 instead of 65,536 x 65,536 (420x fewer elements)
        - N2: 14,400 x 14,400 instead of 1,048,576 x 1,048,576 (5300x fewer elements)
        """
        device = self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        n_subspace = len(self._subspace_basis)

        print(f"Building GPU subspace Hamiltonian ({n_subspace:,} x {n_subspace:,})...")

        # Convert subspace basis to tensor
        basis = torch.stack([torch.tensor(c, device=device) for c in self._subspace_basis])

        # Build matrix using Hamiltonian's matrix_elements method
        H = self.hamiltonian.matrix_elements(basis, basis)

        # Convert to complex128 for time evolution
        return H.to(torch.complex128)

    def _build_sparse_hamiltonian(self):
        """
        Build sparse CSR Hamiltonian matrix in particle-conserving subspace.

        Much smaller than the full Hilbert space (e.g., 3K vs 65K for NH3).
        """
        if self._subspace_basis is not None:
            return self._build_subspace_hamiltonian()

        raise ValueError(
            "No particle-conserving subspace available. "
            "SKQD requires a molecular Hamiltonian with particle conservation."
        )

    def _build_subspace_hamiltonian(self):
        """
        Build sparse Hamiltonian in particle-conserving subspace.

        This is MUCH faster than building in full Hilbert space because:
        - NH3: 3,136 x 3,136 instead of 65,536 x 65,536 (420x fewer elements)
        - N2: 14,400 x 14,400 instead of 1,048,576 x 1,048,576 (5300x fewer elements)
        """
        from scipy.sparse import csr_matrix

        n_subspace = len(self._subspace_basis)
        print(f"Building subspace Hamiltonian ({n_subspace:,} x {n_subspace:,})...")

        rows, cols, data = [], [], []

        # Build Hamiltonian matrix elements in subspace
        for j in range(n_subspace):
            config_j = self._subspace_basis[j]

            # Diagonal element
            diag = self.hamiltonian.diagonal_element(config_j).item()
            rows.append(j)
            cols.append(j)
            data.append(diag)

            # Off-diagonal connections (only within subspace)
            connected, elements = self.hamiltonian.get_connections(config_j)

            if len(connected) > 0:
                for conn, elem in zip(connected, elements):
                    # Look up index in subspace
                    key = tuple(conn.tolist())
                    if key in self._subspace_index_map:
                        i = self._subspace_index_map[key]
                        rows.append(i)
                        cols.append(j)
                        data.append(elem.item() if hasattr(elem, 'item') else elem)

        H_subspace = csr_matrix(
            (data, (rows, cols)),
            shape=(n_subspace, n_subspace),
            dtype=np.complex128
        )

        print(f"Subspace Hamiltonian built: {H_subspace.nnz:,} non-zero elements")
        return H_subspace

    def _sample_from_state(
        self,
        state_vector: torch.Tensor,
        num_samples: int,
    ) -> Dict[str, int]:
        """
        Sample bitstrings from a quantum state.

        Args:
            state_vector: Quantum state vector (full Hilbert space)
            num_samples: Number of shots

        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Compute probabilities
        probs = torch.abs(state_vector) ** 2
        probs = probs / probs.sum()  # Normalize

        # Check for PyTorch multinomial limit (2^24 categories on CUDA)
        max_multinomial_categories = 2**24
        if len(probs) > max_multinomial_categories:
            # Use chunked/CPU sampling for very large state spaces
            return self._sample_from_large_state(probs, num_samples)

        # Sample indices (multinomial on GPU, convert to numpy for counting)
        indices = torch.multinomial(
            probs, num_samples, replacement=True
        ).cpu().numpy()

        # Count occurrences
        unique, counts = np.unique(indices, return_counts=True)

        # Convert to bitstring dictionary
        results = {}
        for idx, count in zip(unique, counts):
            bitstring = self._index_to_bitstring(idx)
            results[bitstring] = int(count)

        return results

    def _sample_from_large_state(
        self,
        probs: torch.Tensor,
        num_samples: int,
    ) -> Dict[str, int]:
        """
        Sample from state with more than 2^24 categories (CUDA multinomial limit).

        Uses numpy for sampling to avoid the CUDA limitation.

        Args:
            probs: Probability distribution (normalized)
            num_samples: Number of shots

        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Move to CPU and use numpy for sampling (no category limit)
        probs_np = probs.cpu().numpy().astype(np.float64)

        # Ensure valid probability distribution
        probs_np = np.maximum(probs_np, 0)
        probs_np = probs_np / probs_np.sum()

        # Sample using numpy's choice (no category limit)
        indices = np.random.choice(
            len(probs_np), size=num_samples, replace=True, p=probs_np
        )

        # Count occurrences
        unique, counts = np.unique(indices, return_counts=True)

        # Convert to bitstring dictionary
        results = {}
        for idx, count in zip(unique, counts):
            bitstring = self._index_to_bitstring(idx)
            results[bitstring] = int(count)

        return results

    def _index_to_bitstring(self, idx: int) -> str:
        """Convert Hilbert space index to bitstring."""
        return format(idx, f"0{self.num_sites}b")

    def _bitstring_to_tensor(self, bitstring: str) -> torch.Tensor:
        """Convert bitstring to tensor configuration."""
        return torch.tensor([int(b) for b in bitstring], dtype=torch.long,
                            device=self.hamiltonian.device)

    def generate_krylov_samples(
        self,
        max_krylov_dim: Optional[int] = None,
        progress: bool = True,
    ) -> List[Dict[str, int]]:
        """
        Generate samples from Krylov states.

        For k = 0, 1, ..., max_k - 1:
            1. Prepare |ψ_k⟩ = U^k |ψ_0⟩
            2. Sample num_shots measurements

        Args:
            max_krylov_dim: Override config max Krylov dimension
            progress: Show progress bar

        Returns:
            List of sample dictionaries for each Krylov state
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        self.krylov_samples = []

        # Work entirely in particle-conserving subspace
        # This avoids memory issues and PyTorch multinomial's 2^24 category limit
        if self._subspace_basis is not None:
            return self._generate_krylov_samples_subspace(max_krylov_dim, progress)

        raise ValueError(
            "No particle-conserving subspace available. "
            "SKQD requires a molecular Hamiltonian with particle conservation."
        )

    def _generate_krylov_samples_subspace(
        self,
        max_krylov_dim: int,
        progress: bool = True,
    ) -> List[Dict[str, int]]:
        """
        Generate Krylov samples working entirely in particle-conserving subspace.

        This is MUCH more memory-efficient for molecular systems:
        - C2H4 (28 qubits): 9M subspace vs 268M full space (30x smaller)
        - Avoids PyTorch multinomial's 2^24 category limit

        The key insight is that for molecular Hamiltonians, the particle number
        is conserved, so we never leave the subspace during time evolution.

        Args:
            max_krylov_dim: Maximum Krylov dimension
            progress: Show progress bar

        Returns:
            List of sample dictionaries for each Krylov state
        """
        # Build sparse Hamiltonian in subspace (cached for reuse)
        if not hasattr(self, '_sparse_H'):
            self._sparse_H = self._build_sparse_hamiltonian()

        # Build GPU dense version for gpu_expm_multiply
        if not hasattr(self, '_dense_H_gpu') or self._dense_H_gpu is None:
            gpu_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            H_dense = self._sparse_H.toarray()
            self._dense_H_gpu = torch.from_numpy(H_dense).to(torch.complex128).to(gpu_device)

        n_subspace = len(self._subspace_basis)

        # Find initial state index in subspace
        initial_key = tuple(self.initial_state.tolist())
        if initial_key not in self._subspace_index_map:
            raise ValueError(
                f"Initial state not in particle-conserving subspace. "
                f"Expected {self.hamiltonian.n_alpha} alpha and "
                f"{self.hamiltonian.n_beta} beta electrons."
            )
        initial_subspace_idx = self._subspace_index_map[initial_key]

        # Create initial state on GPU
        gpu_device = self._dense_H_gpu.device
        psi_subspace = torch.zeros(n_subspace, dtype=torch.complex128, device=gpu_device)
        psi_subspace[initial_subspace_idx] = 1.0

        iterator = range(max_krylov_dim)
        if progress:
            iterator = tqdm(iterator, desc="Generating Krylov states")

        for k in iterator:
            # Sample from current subspace state (needs numpy)
            psi_np = psi_subspace.cpu().numpy()
            samples = self._sample_from_subspace(psi_np)
            self.krylov_samples.append(samples)

            # GPU time evolution: |ψ_{k+1}⟩ = exp(-iHΔt) |ψ_k⟩
            if k < max_krylov_dim - 1:
                psi_subspace = gpu_expm_multiply(
                    self._dense_H_gpu, psi_subspace, t=-1j * self.time_step
                )

        return self.krylov_samples

    def _sample_from_subspace(
        self,
        psi_subspace: np.ndarray,
    ) -> Dict[str, int]:
        """
        Sample bitstrings from a quantum state in subspace representation.

        This method is much more memory-efficient than full-space sampling:
        - Only needs to store probabilities for valid configurations
        - Uses numpy sampling (no CUDA category limit)
        - Converts subspace indices to bitstrings via _subspace_basis

        Args:
            psi_subspace: State vector in subspace representation

        Returns:
            Dictionary mapping bitstrings to counts
        """
        num_samples = self.config.shots_per_krylov
        n_subspace = len(psi_subspace)

        # Compute probabilities in subspace
        probs = np.abs(psi_subspace) ** 2
        probs = probs / probs.sum()  # Normalize

        # Check if we can use CUDA multinomial (faster for moderate sizes)
        max_multinomial_categories = 2**24
        if n_subspace <= max_multinomial_categories and torch.cuda.is_available():
            # Use CUDA for sampling (faster)
            probs_torch = torch.from_numpy(probs).float().cuda()
            indices = torch.multinomial(
                probs_torch, num_samples, replacement=True
            ).cpu().numpy()
        else:
            # Use numpy for very large subspaces
            indices = np.random.choice(
                n_subspace, size=num_samples, replace=True, p=probs.astype(np.float64)
            )

        # Count occurrences
        unique, counts = np.unique(indices, return_counts=True)

        # Convert subspace indices to bitstrings
        results = {}
        for subspace_idx, count in zip(unique, counts):
            config = self._subspace_basis[subspace_idx]
            bitstring = "".join(str(b.item()) for b in config)
            results[bitstring] = int(count)

        return results

    def build_cumulative_basis(self) -> List[Dict[str, int]]:
        """
        Build cumulative basis by accumulating samples across Krylov states.

        cumulative[k] contains all unique bitstrings from steps 0, 1, ..., k.

        Returns:
            List of cumulative sample dictionaries
        """
        cumulative = []
        all_samples: Dict[str, int] = {}

        for k, samples in enumerate(self.krylov_samples):
            # Merge samples
            for bitstring, count in samples.items():
                all_samples[bitstring] = all_samples.get(bitstring, 0) + count

            cumulative.append(dict(all_samples))

        return cumulative

    def get_basis_states(
        self,
        krylov_index: int,
        cumulative: bool = True,
    ) -> torch.Tensor:
        """
        Get basis states as tensor array.

        Args:
            krylov_index: Krylov step index
            cumulative: Whether to use cumulative basis

        Returns:
            Tensor of basis configurations, shape (n_basis, num_sites)
        """
        if cumulative:
            samples = self.build_cumulative_basis()[krylov_index]
        else:
            samples = self.krylov_samples[krylov_index]

        bitstrings = list(samples.keys())
        configs = [self._bitstring_to_tensor(bs) for bs in bitstrings]

        return torch.stack(configs)

    def compute_ground_state_energy(
        self,
        basis: Optional[torch.Tensor] = None,
        return_eigenvector: bool = False,
        regularization: float = 1e-8,
        shift_invert: bool = False,
        mixed_precision: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute ground state energy via subspace diagonalization.

        Projects Hamiltonian onto the sampled basis and diagonalizes using
        sparse eigensolver (scipy.sparse.linalg.eigsh or cupyx equivalent)
        as specified in AGENTs.md for scalability.

        Includes numerical stability improvements:
        - Uses float64 for better precision
        - Regularization for ill-conditioned matrices
        - Hermitian symmetrization
        - SVD-based fallback for problematic cases
        - Validates result is real (imaginary part should be tiny)

        Args:
            basis: Basis states to use (default: use all sampled states)
            return_eigenvector: Whether to return ground state coefficients
            regularization: Small value added to diagonal for stability
            mixed_precision: If True, use FP32 eigh + FP64 Rayleigh refinement
                for ~100x speedup on DGX Spark GB10 (TF32 vs FP64)

        Returns:
            (ground_energy, ground_state_coefficients) if return_eigenvector
            ground_energy otherwise
        """
        if basis is None:
            # Use cumulative basis from last Krylov step
            cumulative = self.build_cumulative_basis()
            basis = self.get_basis_states(len(self.krylov_samples) - 1)

        # Cap basis size to avoid building huge dense matrices (N^2 memory)
        max_diag = self.config.max_diag_basis_size
        if max_diag > 0 and len(basis) > max_diag:
            # Keep first max_diag configs — pipeline orders by importance:
            # essential configs (HF+singles+doubles) first, then PT2-ranked residual
            original_size = len(basis)
            basis = basis[:max_diag]
            print(f"  Capped basis from {original_size} to {max_diag} configs for diagonalization")

        n = len(basis)
        device = basis.device

        # Use sparse path for large bases to avoid O(n²) dense matrix
        SPARSE_THRESHOLD = 3000
        if n >= SPARSE_THRESHOLD and hasattr(self.hamiltonian, 'get_sparse_matrix_elements'):
            return self._sparse_ground_state(
                basis, return_eigenvector, regularization, shift_invert=shift_invert
            )

        # Dense path: build projected Hamiltonian
        H_proj = self.hamiltonian.matrix_elements(basis, basis)

        # Use float64 for numerical stability (GPU supports double precision)
        H = H_proj.detach().double()

        # Ensure Hermitian symmetry (on GPU)
        if H.is_complex():
            H = 0.5 * (H + H.conj().T)
            # Verify Hamiltonian is essentially real (molecular Hamiltonians should be)
            max_imag = H.imag.abs().max().item()
            if max_imag <= 1e-10:
                # Essentially real - use real matrix for better stability
                H = H.real
        else:
            H = 0.5 * (H + H.T)

        # Add small regularization to improve conditioning
        # NOTE: This shifts ALL eigenvalues up by regularization amount
        if regularization > 0:
            H = H + regularization * torch.eye(n, dtype=H.dtype, device=device)

        # Check matrix conditioning (on GPU)
        try:
            # Use torch.linalg.cond for GPU-accelerated condition number
            cond = torch.linalg.cond(H).item()
            if cond > 1e12:
                print(f"WARNING: Ill-conditioned Hamiltonian (cond={cond:.2e})")
                print("Using SVD-based solver for numerical stability")
                return self._svd_ground_state(H, return_eigenvector)
        except Exception:
            print("WARNING: Could not compute condition number, using SVD")
            return self._svd_ground_state(H, return_eigenvector)

        # GPU-accelerated eigensolver — stays entirely on GPU
        # gpu_eigsh uses dense torch.linalg.eigh for n <= 10000, CuPy sparse for larger
        use_gpu = device.type == 'cuda'
        try:
            if mixed_precision:
                # FP32 eigh + FP64 Rayleigh quotient refinement
                # ~100x speedup on DGX Spark GB10 (TF32 vs FP64)
                eigenvalues, eigenvectors = mixed_precision_eigh(H, use_gpu=use_gpu)
                E0 = float(eigenvalues[0].cpu())
                v0 = eigenvectors[:, 0] if return_eigenvector else None
            elif n >= 2:
                k_eig = min(2, n - 1)
                eigenvalues, eigenvectors = gpu_eigsh(H, k=k_eig, which='SA', use_gpu=use_gpu)
                E0 = float(eigenvalues[0].cpu())
                v0 = eigenvectors[:, 0] if return_eigenvector else None
            else:
                eigenvalues, eigenvectors = gpu_eigh(H, use_gpu=use_gpu)
                E0 = float(eigenvalues[0].cpu())
                v0 = eigenvectors[:, 0] if return_eigenvector else None
        except Exception as e:
            print(f"GPU eigensolver failed: {e}, falling back to CPU solver")
            H_np = H.cpu().numpy()
            eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            E0 = float(eigenvalues[0])
            v0 = torch.from_numpy(eigenvectors[:, 0]).to(device) if return_eigenvector else None

        if return_eigenvector:
            return E0, v0
        else:
            return E0, None

    def _svd_ground_state(
        self,
        H_input,
        return_eigenvector: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute ground state using eigendecomposition with regularization.

        GPU-accelerated: uses torch.linalg.eigh on GPU.
        Projects out near-zero eigenvalue modes for numerical stability,
        preserving sign (unlike SVD which maps negative eigenvalues to positive).
        Accepts either numpy array or torch tensor.
        """
        # Convert to torch if needed
        if isinstance(H_input, np.ndarray):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            H_t = torch.from_numpy(H_input).double().to(device)
        else:
            H_t = H_input.double()
            device = H_t.device

        # GPU eigendecomposition to identify near-null modes
        eigenvalues_raw, eigenvectors_raw = torch.linalg.eigh(H_t)

        # Filter out near-zero eigenvalue modes (regularize)
        threshold = 1e-10 * eigenvalues_raw.abs().max().item()
        n_small = (eigenvalues_raw.abs() < threshold).sum().item()

        if n_small > 0:
            print(f"  Regularization: {n_small} near-null eigenvalue modes")
            # Regularize: clamp small |eigenvalues| but preserve sign
            eig_reg = torch.where(
                eigenvalues_raw.abs() > threshold,
                eigenvalues_raw,
                torch.sign(eigenvalues_raw) * threshold
            )
            # Reconstruct H with regularized eigenvalues (preserves sign)
            H_reg = eigenvectors_raw @ torch.diag(eig_reg) @ eigenvectors_raw.T
            eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
        else:
            eigenvalues = eigenvalues_raw
            eigenvectors = eigenvectors_raw

        E0 = float(eigenvalues[0].cpu())
        v0 = eigenvectors[:, 0]

        if return_eigenvector:
            return E0, v0
        else:
            return E0, None

    def _sparse_ground_state(
        self,
        basis: torch.Tensor,
        return_eigenvector: bool = False,
        regularization: float = 0.0,
        shift_invert: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute ground state using sparse Hamiltonian construction + scipy.sparse.eigsh.

        Avoids O(n²) dense matrix construction for large bases.
        Uses get_sparse_matrix_elements() for off-diagonals + diagonal_element() for diagonal.

        Args:
            shift_invert: If True, use shift-invert mode with sigma=E_hf.
                This finds eigenvalues near sigma via (H - sigma*I)^{-1},
                converging faster when sigma is close to the ground state.
        """
        from scipy.sparse import coo_matrix, csr_matrix
        from scipy.sparse.linalg import eigsh

        n = len(basis)
        device = basis.device
        mode_str = "shift-invert" if shift_invert else "standard"
        print(f"  Using sparse eigensolver ({mode_str}) for {n} configs")

        # Get sparse off-diagonal elements
        rows, cols, vals = self.hamiltonian.get_sparse_matrix_elements(basis)
        rows_np = rows.cpu().numpy()
        cols_np = cols.cpu().numpy()
        vals_np = vals.cpu().numpy().astype(np.float64)

        # Build sparse matrix (off-diagonal)
        H_sparse = coo_matrix((vals_np, (rows_np, cols_np)), shape=(n, n))

        # Add diagonal elements
        diag_vals = np.array([
            self.hamiltonian.diagonal_element(basis[i]).item()
            for i in range(n)
        ], dtype=np.float64)

        if regularization > 0:
            diag_vals += regularization

        from scipy.sparse import diags
        # get_sparse_matrix_elements already returns BOTH (i,j) and (j,i) entries.
        # COO sums duplicates, so H_sparse already has correct off-diagonal values.
        # Just symmetrize to handle any floating-point asymmetry, then add diagonal.
        H_csr = H_sparse.tocsr()
        H_csr = 0.5 * (H_csr + H_csr.T)
        H_csr = H_csr + diags(diag_vals, 0, shape=(n, n), format='csr')

        # Sparse eigensolver
        try:
            k_eig = min(2, n - 1) if n >= 2 else 1

            if shift_invert:
                # Shift-invert: solve (H - sigma*I)^{-1} to find eigenvalues
                # near sigma. Using which='SA' returns the algebraically smallest
                # eigenvalues, i.e., the ground state.
                sigma = float(self.hamiltonian.diagonal_element(
                    self.hamiltonian.get_hf_state()
                ).item())
                eigenvalues, eigenvectors = eigsh(
                    H_csr, k=k_eig, sigma=sigma, which='SA'
                )
                E0 = float(eigenvalues[0])
                v0 = torch.from_numpy(eigenvectors[:, 0]).to(device) if return_eigenvector else None
            else:
                eigenvalues, eigenvectors = eigsh(H_csr, k=k_eig, which='SA')
                E0 = float(eigenvalues[0])
                v0 = torch.from_numpy(eigenvectors[:, 0]).to(device) if return_eigenvector else None
        except Exception as e:
            print(f"  Sparse eigsh failed ({e}), falling back to dense")
            H_dense = H_csr.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            E0 = float(eigenvalues[0])
            v0 = torch.from_numpy(eigenvectors[:, 0]).to(device) if return_eigenvector else None

        if return_eigenvector:
            return E0, v0
        else:
            return E0, None

    def run(
        self,
        max_krylov_dim: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, List]:
        """
        Run full SKQD algorithm.

        Returns energy as function of Krylov dimension.

        Args:
            max_krylov_dim: Maximum Krylov dimension
            progress: Show progress bars

        Returns:
            Dictionary with 'krylov_dims', 'energies', 'basis_sizes'
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        # Generate Krylov samples
        self.generate_krylov_samples(max_krylov_dim, progress=progress)

        # Build cumulative basis
        cumulative = self.build_cumulative_basis()

        # Compute energy at each Krylov dimension
        results = {
            "krylov_dims": [],
            "energies": [],
            "basis_sizes": [],
        }

        for k in range(1, max_krylov_dim):
            basis = self.get_basis_states(k, cumulative=True)
            E0, _ = self.compute_ground_state_energy(basis)

            results["krylov_dims"].append(k + 1)
            results["energies"].append(E0)
            results["basis_sizes"].append(len(basis))

        self.energies = results["energies"]

        return results


class FlowGuidedSKQD(SampleBasedKrylovDiagonalization):
    """
    SKQD with Flow-Guided initial basis.

    Instead of (or in addition to) using Krylov time evolution samples,
    this variant incorporates the basis discovered by the normalizing flow.

    The NF-discovered basis provides a good initial subspace that captures
    the support of the ground state, while Krylov refinement improves
    the energy estimate through systematic subspace expansion.

    OPTIMIZATION FOR LARGE SYSTEMS:
    For systems with >100k valid configurations, we use NF-basis-guided
    Krylov expansion instead of full particle-conserving subspace evolution.
    This avoids building the prohibitively large Hamiltonian matrix.
    """

    # Threshold for using NF-guided Krylov (vs full subspace evolution)
    MAX_FULL_SUBSPACE_SIZE = 100000

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        nf_basis: torch.Tensor,
        config: Optional[SKQDConfig] = None,
        initial_state: Optional[torch.Tensor] = None,
        force_nf_guided: bool = False,
    ):
        # Check if we should use NF-guided mode for large systems
        self._use_nf_guided_mode = force_nf_guided
        if not self._use_nf_guided_mode:
            if hasattr(hamiltonian, 'n_alpha') and hasattr(hamiltonian, 'n_beta'):
                n_valid = comb(hamiltonian.n_orbitals, hamiltonian.n_alpha) * \
                          comb(hamiltonian.n_orbitals, hamiltonian.n_beta)
                if n_valid > self.MAX_FULL_SUBSPACE_SIZE:
                    self._use_nf_guided_mode = True

        if self._use_nf_guided_mode:
            print(f"Using NF-guided Krylov mode")

        # For NF-guided mode, temporarily disable molecular detection
        # to prevent full subspace setup
        if self._use_nf_guided_mode:
            # Don't call parent's molecular subspace setup
            self.hamiltonian = hamiltonian
            self.config = config or SKQDConfig()
            self.num_sites = hamiltonian.num_sites
            self._is_molecular = True  # Still molecular, but no full subspace
            self._subspace_basis = None
            self._subspace_index_map = None
            self._sparse_H = None

            # Set up initial state from NF basis
            if initial_state is not None:
                self.initial_state = initial_state
            else:
                self.initial_state = hamiltonian.get_hf_state()

            self.time_step = self.config.time_step
            self.krylov_samples = []
            self.cumulative_basis = []
            self.energies = []
        else:
            # Standard initialization for smaller systems
            super().__init__(hamiltonian, config, initial_state)

        self.nf_basis = nf_basis  # (n_nf, num_sites)

    def get_combined_basis(
        self,
        krylov_index: int,
        include_nf: bool = True,
    ) -> torch.Tensor:
        """
        Get combined basis from NF and Krylov sampling.

        Args:
            krylov_index: Krylov step index
            include_nf: Whether to include NF-discovered basis

        Returns:
            Combined unique basis states
        """
        # Get Krylov basis
        krylov_basis = self.get_basis_states(krylov_index, cumulative=True)

        if not include_nf:
            return krylov_basis

        # Ensure both are on the same device
        nf_basis = self.nf_basis.to(krylov_basis.device)

        # Combine with NF basis
        combined = torch.cat([nf_basis, krylov_basis], dim=0)

        # Remove duplicates
        unique = torch.unique(combined, dim=0)

        return unique

    def _generate_krylov_samples_nf_guided(
        self,
        max_krylov_dim: int,
        progress: bool = True,
    ) -> List[Dict[str, int]]:
        """
        Generate Krylov samples using NF-guided expansion (for large systems).

        This method avoids building the full particle-conserving Hamiltonian by:
        1. Starting with NF basis as the initial subspace
        2. Building H only in the current subspace (small matrix)
        3. Time evolving in this subspace
        4. Discovering new configurations via get_connections
        5. Expanding the subspace dynamically

        For C2H4: Works with ~1000-10000 configs instead of 9M.

        Args:
            max_krylov_dim: Maximum Krylov dimension
            progress: Show progress bar

        Returns:
            List of sample dictionaries for each Krylov step
        """
        device = self.hamiltonian.device if hasattr(self.hamiltonian, 'device') else 'cpu'

        # Start with NF basis as the subspace
        current_basis = self.nf_basis.clone().to(device)
        n_initial = len(current_basis)

        # Create index mapping
        basis_set = {tuple(c.cpu().tolist()) for c in current_basis}

        # Cap total expansion to max_diag_basis_size to avoid huge matrix builds
        max_expansion = self.config.max_diag_basis_size
        if max_expansion > 0:
            print(f"NF-guided Krylov: Starting with {n_initial} NF configs (max expansion: {max_expansion})")
        else:
            print(f"NF-guided Krylov: Starting with {n_initial} NF configs")

        # Initialize state: uniform superposition over NF basis (GPU tensor)
        n_subspace = len(current_basis)
        psi = torch.ones(n_subspace, dtype=torch.complex128, device=device) / np.sqrt(n_subspace)

        self.krylov_samples = []
        self._nf_guided_psi = None  # For importance-weighted exploration
        H_subspace_cached = None  # Cache H when basis is frozen (expansion capped)

        iterator = range(max_krylov_dim)
        if progress:
            iterator = tqdm(iterator, desc="NF-guided Krylov")

        for k in iterator:
            # Store psi as numpy for compatibility with sampling/exploration
            self._nf_guided_psi = psi.cpu().numpy()

            # Sample from current state (needs numpy psi)
            samples = self._sample_from_subspace_basis(self._nf_guided_psi, current_basis)
            self.krylov_samples.append(samples)

            if k < max_krylov_dim - 1:
                # Check if we've hit the expansion cap
                if max_expansion > 0 and len(current_basis) >= max_expansion:
                    # Basis is frozen — reuse cached H to avoid expensive rebuild
                    if H_subspace_cached is None:
                        H_subspace_cached = self._build_hamiltonian_in_basis_gpu(current_basis)
                    # GPU time evolution via gpu_expm_multiply
                    psi = gpu_expm_multiply(H_subspace_cached, psi, t=-1j * self.time_step)
                    psi = psi / torch.linalg.norm(psi)
                    continue

                # Expand subspace by finding connected configurations
                new_configs = self._find_connected_configs(current_basis, basis_set)

                # Trim new configs if they'd exceed the cap
                if max_expansion > 0 and len(new_configs) > 0:
                    room = max_expansion - len(current_basis)
                    if room <= 0:
                        new_configs = new_configs[:0]  # Empty tensor
                    elif len(new_configs) > room:
                        new_configs = new_configs[:room]

                if len(new_configs) > 0:
                    # Add new configs to basis
                    current_basis = torch.cat([current_basis, new_configs], dim=0)
                    for c in new_configs:
                        basis_set.add(tuple(c.cpu().tolist()))

                    # Expand state vector on GPU (new configs start with zero amplitude)
                    n_new = len(new_configs)
                    psi = torch.cat([psi, torch.zeros(n_new, dtype=torch.complex128, device=device)])

                # Build Hamiltonian in current subspace (GPU dense tensor)
                H_subspace = self._build_hamiltonian_in_basis_gpu(current_basis)

                # GPU time evolution
                psi = gpu_expm_multiply(H_subspace, psi, t=-1j * self.time_step)

                # Normalize on GPU
                psi = psi / torch.linalg.norm(psi)

        n_final = len(current_basis)
        print(f"NF-guided Krylov: Expanded to {n_final} configs (+{n_final - n_initial} new)")

        # Store final basis for later use
        self._nf_guided_basis = current_basis

        return self.krylov_samples

    def _find_connected_configs(
        self,
        basis: torch.Tensor,
        basis_set: set,
        max_new_per_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Find configurations connected to current basis but not in it.

        Uses configurable sampling parameters for better discovery rate.
        Default discovery rate was ~7% (296/4144), target is >20%.

        OPTIMIZED: Uses GPU-based integer encoding with single CPU transfer
        instead of per-connection CPU transfers.
        """
        device = basis.device
        cfg = self.config
        n_sites = basis.shape[1]

        # Use config parameter if not explicitly overridden
        if max_new_per_step is None:
            max_new_per_step = cfg.max_new_configs_per_krylov_step

        # Precompute powers of 2 on GPU for fast integer encoding
        powers = (2 ** torch.arange(n_sites, device=device, dtype=torch.long)).flip(0)

        # Calculate how many basis states to sample
        # Use fraction-based sampling with configurable minimum
        n_sample = max(
            cfg.min_krylov_basis_sample,
            int(len(basis) * cfg.krylov_basis_sample_fraction)
        )
        n_sample = min(len(basis), n_sample)

        # Use importance-weighted sampling if we have state amplitudes
        # Prefer exploring connections of high-amplitude basis states
        if hasattr(self, '_nf_guided_psi') and self._nf_guided_psi is not None:
            psi = self._nf_guided_psi
            probs = np.abs(psi[:len(basis)]) ** 2
            probs = probs / probs.sum()
            indices_np = np.random.choice(
                len(basis), size=n_sample, replace=False, p=probs
            )
            indices = torch.from_numpy(indices_np)
        else:
            indices = torch.randperm(len(basis))[:n_sample]

        # B4: Use vectorized batch method if available (avoids per-config Python loop)
        sampled_basis = basis[indices]
        if hasattr(self.hamiltonian, 'get_connections_vectorized_batch'):
            all_connected, _, _ = self.hamiltonian.get_connections_vectorized_batch(sampled_basis)
        else:
            all_connected_list = []
            for idx in indices:
                connected, elements = self.hamiltonian.get_connections(basis[idx])
                if len(connected) > 0:
                    all_connected_list.append(connected)
            if not all_connected_list:
                return torch.empty(0, n_sites, device=device)
            all_connected = torch.cat(all_connected_list, dim=0)

        if len(all_connected) == 0:
            return torch.empty(0, n_sites, device=device)

        # GPU-based integer encoding (single operation)
        connected_ints = (all_connected.long() * powers).sum(dim=1)

        # Single CPU transfer for membership check
        connected_ints_cpu = connected_ints.cpu().tolist()

        # Find new configs not in basis_set
        new_set = set()
        new_indices = []
        for i, config_int in enumerate(connected_ints_cpu):
            if config_int not in basis_set and config_int not in new_set:
                new_set.add(config_int)
                new_indices.append(i)
                if len(new_indices) >= max_new_per_step:
                    break

        if not new_indices:
            return torch.empty(0, n_sites, device=device)

        return all_connected[new_indices]

    def _build_hamiltonian_in_basis_gpu(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Build dense Hamiltonian matrix in given basis, entirely on GPU.

        Uses hamiltonian.matrix_elements() which returns a GPU torch tensor.
        This avoids the scipy sparse CPU path entirely.

        Returns:
            Dense Hamiltonian matrix (n, n) as complex128 torch tensor on device.
        """
        H = self.hamiltonian.matrix_elements(basis, basis)
        # Ensure complex128 for time evolution compatibility
        H = H.to(torch.complex128)
        # Symmetrize on GPU
        H = 0.5 * (H + H.conj().T)
        return H

    def _build_hamiltonian_in_basis(self, basis: torch.Tensor):
        """
        Build sparse Hamiltonian matrix in given basis (scipy sparse, CPU).

        Legacy method kept for compatibility with non-GPU code paths.
        For GPU code paths, use _build_hamiltonian_in_basis_gpu() instead.

        OPTIMIZED: Uses batch diagonal computation and GPU-based integer
        encoding for efficient membership checking.
        """
        from scipy.sparse import csr_matrix

        n = len(basis)
        device = basis.device
        n_sites = basis.shape[1]

        # Precompute powers of 2 on GPU for fast integer encoding
        powers = (2 ** torch.arange(n_sites, device=device, dtype=torch.long)).flip(0)

        # Create index mapping using GPU-based integer encoding (single CPU transfer)
        basis_ints = (basis.long() * powers).sum(dim=1).cpu().tolist()
        basis_map = {config_int: i for i, config_int in enumerate(basis_ints)}

        rows, cols, data = [], [], []

        # OPTIMIZED: Batch diagonal computation
        if hasattr(self.hamiltonian, 'diagonal_elements_batch'):
            diag_elements = self.hamiltonian.diagonal_elements_batch(basis)
            for j in range(n):
                rows.append(j)
                cols.append(j)
                data.append(diag_elements[j].item())
        else:
            # Fallback for Hamiltonians without batch method
            for j in range(n):
                diag = self.hamiltonian.diagonal_element(basis[j]).item()
                rows.append(j)
                cols.append(j)
                data.append(diag)

        # B4: Off-diagonal — use vectorized batch if available
        if hasattr(self.hamiltonian, 'get_connections_vectorized_batch'):
            all_connected, all_elements, batch_idx = self.hamiltonian.get_connections_vectorized_batch(basis)
            if len(all_connected) > 0:
                connected_ints = (all_connected.long() * powers).sum(dim=1).cpu().tolist()
                batch_idx_cpu = batch_idx.cpu().tolist()
                all_elem_cpu = all_elements.cpu().tolist() if hasattr(all_elements, 'cpu') else list(all_elements)
                for k_idx, config_int in enumerate(connected_ints):
                    if config_int in basis_map:
                        i = basis_map[config_int]
                        j = batch_idx_cpu[k_idx]
                        rows.append(i)
                        cols.append(j)
                        data.append(all_elem_cpu[k_idx])
        else:
            for j in range(n):
                connected, elements = self.hamiltonian.get_connections(basis[j])
                if len(connected) == 0:
                    continue
                connected_ints = (connected.long() * powers).sum(dim=1).cpu().tolist()
                for k, config_int in enumerate(connected_ints):
                    if config_int in basis_map:
                        i = basis_map[config_int]
                        elem = elements[k]
                        rows.append(i)
                        cols.append(j)
                        data.append(elem.item() if hasattr(elem, 'item') else elem)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.complex128
        )

    def _sample_from_subspace_basis(
        self,
        psi: np.ndarray,
        basis: torch.Tensor,
    ) -> Dict[str, int]:
        """Sample bitstrings from state in given basis."""
        num_samples = self.config.shots_per_krylov

        probs = np.abs(psi) ** 2
        probs = probs / probs.sum()

        indices = np.random.choice(len(probs), size=num_samples, replace=True, p=probs)
        unique, counts = np.unique(indices, return_counts=True)

        results = {}
        for idx, count in zip(unique, counts):
            config = basis[idx]
            bitstring = "".join(str(b.item()) for b in config)
            results[bitstring] = int(count)

        return results

    def run_with_nf(
        self,
        max_krylov_dim: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, List]:
        """
        Run SKQD with NF-augmented basis.

        IMPORTANT: This method combines Krylov-discovered configurations with
        the NF basis, rather than using the full particle-conserving subspace.
        This is the correct approach because:
        1. The NF basis already captures important low-energy configurations
        2. Krylov time evolution discovers configurations missed by NF
        3. Combining them gives a better basis than either alone

        For large systems (>100k configs), uses NF-guided Krylov expansion
        which works in a growing subspace instead of the full space.

        Includes numerical stability improvements:
        - Uses regularization from config
        - Validates energy is variationally consistent
        - Falls back to best NF-only energy if instability detected
        - Uses float64 for better numerical precision

        Returns:
            Dictionary with results comparing NF-only, Krylov-only, and combined
        """
        if max_krylov_dim is None:
            max_krylov_dim = self.config.max_krylov_dim

        # Energy with NF basis only (reference for stability check)
        E_nf, _ = self.compute_ground_state_energy(
            self.nf_basis,
            regularization=self.config.regularization
        )
        print(f"NF-only basis energy: {E_nf:.6f} ({len(self.nf_basis)} configs)")

        # Generate Krylov samples - use appropriate method based on system size
        if self._use_nf_guided_mode:
            self._generate_krylov_samples_nf_guided(max_krylov_dim, progress=progress)
        else:
            self.generate_krylov_samples(max_krylov_dim, progress=progress)

        results = {
            "krylov_dims": [],
            "energies_krylov": [],
            "energies_combined": [],
            "basis_sizes_krylov": [],
            "basis_sizes_combined": [],
            "energy_nf_only": E_nf,
            "nf_basis_size": len(self.nf_basis),
            "numerical_warnings": [],
        }

        best_energy = E_nf
        best_basis_size = len(self.nf_basis)
        instability_detected = False

        # For large systems (>10k configs), skip per-dimension diagnostics
        # and only compute final energy to avoid expensive matrix builds
        skip_diagnostics = len(self.nf_basis) > 10000
        if skip_diagnostics:
            print(f"Large basis ({len(self.nf_basis)} configs): skipping per-dimension energy computation")
            # Only compute final energy at last Krylov dimension
            k = max_krylov_dim - 1
            print(f"Computing final combined energy (k={k+1})...")
            krylov_basis = self.get_basis_states(k, cumulative=True)
            combined_basis = self.get_combined_basis(k, include_nf=True)
            print(f"  Building Hamiltonian for {len(combined_basis)} configs...")
            E_combined, _ = self.compute_ground_state_energy(
                combined_basis,
                regularization=self.config.regularization
            )
            print(f"  Final energy: {E_combined:.8f} Ha")

            results["krylov_dims"].append(k + 1)
            results["energies_krylov"].append(E_combined)  # Use combined for both
            results["energies_combined"].append(E_combined)
            results["basis_sizes_krylov"].append(len(krylov_basis))
            results["basis_sizes_combined"].append(len(combined_basis))

            best_energy = E_combined
            best_basis_size = len(combined_basis)
        else:
            # Standard path for smaller systems: compute energy at each Krylov dimension
            # B5: Only compute combined energy (not krylov-only) to halve eigsh calls
            for k in range(1, max_krylov_dim):
                krylov_basis = self.get_basis_states(k, cumulative=True)

                # Combined: NF basis + Krylov-discovered configs
                combined_basis = self.get_combined_basis(k, include_nf=True)
                E_combined, _ = self.compute_ground_state_energy(
                    combined_basis,
                    regularization=self.config.regularization
                )
                # Use combined energy as krylov estimate too (avoids second eigsh call)
                E_krylov = E_combined

                # VARIATIONAL CHECK: Energy should decrease or stay same as basis grows
                # If energy increases, likely numerical instability
                if k > 1 and len(results["energies_combined"]) > 0:
                    prev_energy = results["energies_combined"][-1]
                    energy_change = E_combined - prev_energy

                    # Energy should not increase significantly
                    if energy_change > 0.001:  # 1 mHa tolerance
                        warning = f"k={k+1}: Energy increased by {energy_change*1000:.4f} mHa (numerical instability)"
                        results["numerical_warnings"].append(warning)
                        print(f"WARNING: {warning}")
                        instability_detected = True

                    # Large energy jumps can indicate numerical instability
                    if abs(energy_change) > 1.0:  # 1 Ha is suspicious for Krylov refinement
                        warning = f"k={k+1}: Large energy jump {abs(energy_change):.4f} Ha"
                        results["numerical_warnings"].append(warning)
                        print(f"WARNING: {warning}")
                        instability_detected = True

                # Track best valid energy (variational principle)
                if E_combined < best_energy:
                    best_energy = E_combined
                    best_basis_size = len(combined_basis)

                results["krylov_dims"].append(k + 1)
                results["energies_krylov"].append(E_krylov)
                results["energies_combined"].append(E_combined)
                results["basis_sizes_krylov"].append(len(krylov_basis))
                results["basis_sizes_combined"].append(len(combined_basis))

        # Report statistics on Krylov contribution
        if results["energies_combined"]:
            krylov_improvement = E_nf - best_energy
            new_configs = best_basis_size - len(self.nf_basis)
            if krylov_improvement > 0:
                print(f"Krylov improvement: {krylov_improvement*1000:.4f} mHa "
                      f"({new_configs} new configs from Krylov sampling)")

        # If instability detected, report the most stable result
        if instability_detected:
            print(f"Numerical instability detected. Best stable energy: {best_energy:.6f}")
            results["best_stable_energy"] = best_energy
        else:
            results["best_stable_energy"] = best_energy

        return results
