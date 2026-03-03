"""
Eigenvalue solvers for projected Hamiltonians.

Provides interfaces to sparse eigensolvers for finding ground state
energies from projected Hamiltonian matrices.

Includes:
- Standard sparse eigensolvers (scipy/cupy eigsh)
- Davidson iterative solver for large subspaces
- Lanczos method for extreme eigenvalues
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
    from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from scipy.sparse import csr_matrix as csr_matrix_cpu
from scipy.sparse.linalg import eigsh as eigsh_cpu


def solve_generalized_eigenvalue(
    H: "csr_matrix",
    S: Optional["csr_matrix"] = None,
    k: int = 2,
    which: str = "SA",
    use_gpu: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenvalue problem Hv = ESv.

    For SKQD, S is typically the identity (standard eigenvalue problem)
    since the sampled basis states are orthonormal in the computational basis.

    For KQD with Krylov overlaps, S would be the overlap matrix.

    Args:
        H: Hamiltonian matrix (sparse)
        S: Overlap matrix (optional, default: identity)
        k: Number of eigenvalues to compute
        which: Which eigenvalues ("SA" = smallest algebraic)
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments for eigsh

    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue
    """
    use_gpu = use_gpu and CUPY_AVAILABLE

    if use_gpu:
        eigsh = eigsh_gpu
        # Ensure matrices are on GPU
        if not isinstance(H, csr_matrix_gpu):
            H = csr_matrix_gpu(H)
        if S is not None and not isinstance(S, csr_matrix_gpu):
            S = csr_matrix_gpu(S)
    else:
        eigsh = eigsh_cpu
        # Ensure matrices are on CPU
        if hasattr(H, 'get'):
            H = csr_matrix_cpu(H.get())
        if S is not None and hasattr(S, 'get'):
            S = csr_matrix_cpu(S.get())

    # Solve eigenvalue problem
    if S is None:
        eigenvalues, eigenvectors = eigsh(
            H, k=k, which=which, return_eigenvectors=True, **kwargs
        )
    else:
        eigenvalues, eigenvectors = eigsh(
            H, M=S, k=k, which=which, return_eigenvectors=True, **kwargs
        )

    # Convert to numpy if on GPU
    if use_gpu:
        eigenvalues = eigenvalues.get()
        eigenvectors = eigenvectors.get()

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_ground_state_energy(
    H: "csr_matrix",
    use_gpu: bool = True,
    **kwargs,
) -> float:
    """
    Compute ground state energy of a Hamiltonian.

    Args:
        H: Hamiltonian matrix (sparse)
        use_gpu: Whether to use GPU
        **kwargs: Additional arguments for eigsh

    Returns:
        Ground state energy
    """
    eigenvalues, _ = solve_generalized_eigenvalue(
        H, k=1, which="SA", use_gpu=use_gpu, **kwargs
    )
    return float(eigenvalues[0])


def analyze_spectrum(
    H: "csr_matrix",
    k: int = 10,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Analyze low-energy spectrum of Hamiltonian.

    Args:
        H: Hamiltonian matrix
        k: Number of low-lying states to compute
        use_gpu: Whether to use GPU

    Returns:
        Dictionary with spectral analysis:
            - eigenvalues: Array of k lowest eigenvalues
            - gaps: Energy gaps E_i - E_0
            - ground_state_energy: E_0
    """
    eigenvalues, eigenvectors = solve_generalized_eigenvalue(
        H, k=k, which="SA", use_gpu=use_gpu
    )

    E0 = eigenvalues[0]
    gaps = eigenvalues - E0

    return {
        "eigenvalues": eigenvalues,
        "gaps": gaps,
        "ground_state_energy": E0,
        "first_gap": gaps[1] if len(gaps) > 1 else None,
        "eigenvectors": eigenvectors,
    }


def regularize_overlap_matrix(
    S: "csr_matrix",
    threshold: float = 1e-10,
    use_gpu: bool = True,
) -> "csr_matrix":
    """
    Regularize an overlap matrix to ensure positive definiteness.

    For ill-conditioned Krylov overlap matrices, small eigenvalues
    can cause numerical instability. This function removes components
    with eigenvalues below a threshold.

    Args:
        S: Overlap matrix
        threshold: Minimum eigenvalue to retain
        use_gpu: Whether to use GPU

    Returns:
        Regularized overlap matrix
    """
    use_gpu = use_gpu and CUPY_AVAILABLE

    if use_gpu:
        S_dense = cp.asarray(S.toarray())
        eigenvalues, eigenvectors = cp.linalg.eigh(S_dense)

        # Zero out small eigenvalues
        eigenvalues = cp.maximum(eigenvalues, threshold)

        # Reconstruct
        S_reg = eigenvectors @ cp.diag(eigenvalues) @ eigenvectors.T
        return csr_matrix_gpu(S_reg)
    else:
        S_dense = np.asarray(S.toarray())
        eigenvalues, eigenvectors = np.linalg.eigh(S_dense)

        eigenvalues = np.maximum(eigenvalues, threshold)

        S_reg = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return csr_matrix_cpu(S_reg)


class DavidsonSolver:
    """
    Davidson iterative eigensolver for large subspaces.

    The Davidson method is an iterative Krylov-like method that efficiently
    finds extreme eigenvalues of large sparse matrices. It's particularly
    effective when:
    - Matrix is too large for full diagonalization
    - Only a few eigenvalues are needed
    - A good preconditioner is available

    Algorithm:
    1. Start with initial guess vectors
    2. Build and diagonalize subspace Hamiltonian
    3. Compute residual r = (H - E*I) * v
    4. If converged, return
    5. Apply preconditioner to residual
    6. Orthogonalize and add to subspace
    7. Repeat from step 2

    References:
    - Davidson, E.R., "The iterative calculation of a few of the lowest
      eigenvalues and corresponding eigenvectors of large real-symmetric
      matrices", J. Comput. Phys. 17, 87 (1975)
    """

    def __init__(
        self,
        max_subspace_size: int = 50,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-8,
        n_eigenvalues: int = 1,
        use_preconditioner: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            max_subspace_size: Maximum Davidson subspace dimension
            max_iterations: Maximum number of iterations
            convergence_threshold: Residual norm threshold for convergence
            n_eigenvalues: Number of eigenvalues to compute
            use_preconditioner: Whether to use diagonal preconditioner
            verbose: Print convergence information
        """
        self.max_subspace_size = max_subspace_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.n_eigenvalues = n_eigenvalues
        self.use_preconditioner = use_preconditioner
        self.verbose = verbose

    def solve(
        self,
        H: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        diagonal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve for lowest eigenvalues using Davidson method.

        Args:
            H: Hamiltonian matrix (dense or sparse)
            initial_guess: Initial guess vectors (n_dim, n_guess)
            diagonal: Diagonal elements for preconditioner

        Returns:
            eigenvalues: (n_eigenvalues,) lowest eigenvalues
            eigenvectors: (n_dim, n_eigenvalues) corresponding eigenvectors
            info: Dictionary with convergence information
        """
        n_dim = H.shape[0]
        n_eig = min(self.n_eigenvalues, n_dim)

        # Handle sparse matrices
        if hasattr(H, 'toarray'):
            H_dense = H.toarray()
        else:
            H_dense = np.asarray(H)

        # Get diagonal for preconditioner
        if diagonal is None:
            diagonal = np.diag(H_dense)

        # Initialize guess vectors
        if initial_guess is None:
            # Start with unit vectors at lowest diagonal elements
            sorted_indices = np.argsort(diagonal)
            initial_guess = np.zeros((n_dim, n_eig))
            for i in range(n_eig):
                initial_guess[sorted_indices[i], i] = 1.0

        # Orthonormalize initial guess
        V, _ = np.linalg.qr(initial_guess)

        converged = False
        iteration = 0
        history = {'residuals': [], 'eigenvalues': []}

        while not converged and iteration < self.max_iterations:
            iteration += 1
            subspace_size = V.shape[1]

            # Build subspace Hamiltonian
            H_sub = V.T @ H_dense @ V

            # Diagonalize subspace Hamiltonian
            sub_eigenvalues, sub_eigenvectors = np.linalg.eigh(H_sub)

            # Get Ritz vectors (approximate eigenvectors)
            ritz_vectors = V @ sub_eigenvectors[:, :n_eig]
            ritz_values = sub_eigenvalues[:n_eig]

            # Compute residuals
            residuals = H_dense @ ritz_vectors - ritz_vectors * ritz_values
            residual_norms = np.linalg.norm(residuals, axis=0)

            history['residuals'].append(residual_norms.copy())
            history['eigenvalues'].append(ritz_values.copy())

            if self.verbose:
                print(f"Iteration {iteration}: E = {ritz_values[0]:.8f}, "
                      f"residual = {residual_norms[0]:.2e}")

            # Check convergence
            if np.all(residual_norms < self.convergence_threshold):
                converged = True
                break

            # Add correction vectors to subspace
            new_vectors = []
            for i in range(n_eig):
                if residual_norms[i] >= self.convergence_threshold:
                    # Apply preconditioner
                    if self.use_preconditioner:
                        precond = 1.0 / (diagonal - ritz_values[i] + 1e-10)
                        precond = np.clip(precond, -1e10, 1e10)
                        correction = precond * residuals[:, i]
                    else:
                        correction = residuals[:, i]

                    new_vectors.append(correction)

            if not new_vectors:
                break

            # Orthogonalize new vectors against V
            new_V = np.column_stack(new_vectors)
            new_V = new_V - V @ (V.T @ new_V)

            # QR orthogonalization
            Q, R = np.linalg.qr(new_V)
            # Only keep vectors with sufficient norm
            keep = np.diag(R) > 1e-12
            if not np.any(keep):
                break
            Q = Q[:, keep]

            # Expand subspace
            V = np.hstack([V, Q])

            # Collapse subspace if too large
            if V.shape[1] > self.max_subspace_size:
                # Keep best Ritz vectors
                V = ritz_vectors.copy()

        info = {
            'converged': converged,
            'iterations': iteration,
            'final_residuals': residual_norms,
            'history': history,
        }

        return ritz_values, ritz_vectors, info


def davidson_eigensolver(
    H: np.ndarray,
    n_eigenvalues: int = 1,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-8,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for Davidson eigensolver.

    Args:
        H: Hamiltonian matrix
        n_eigenvalues: Number of eigenvalues to compute
        max_iterations: Maximum iterations
        convergence_threshold: Convergence threshold
        verbose: Print convergence info

    Returns:
        eigenvalues: Lowest eigenvalues
        eigenvectors: Corresponding eigenvectors
    """
    solver = DavidsonSolver(
        n_eigenvalues=n_eigenvalues,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        verbose=verbose,
    )
    eigenvalues, eigenvectors, _ = solver.solve(H)
    return eigenvalues, eigenvectors


def adaptive_eigensolver(
    H: np.ndarray,
    n_eigenvalues: int = 1,
    use_gpu: bool = True,
    davidson_threshold: int = 500,
    sparse_threshold: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically select best eigensolver based on matrix size.

    Args:
        H: Hamiltonian matrix
        n_eigenvalues: Number of eigenvalues to compute
        use_gpu: Whether to use GPU if available
        davidson_threshold: Use Davidson for matrices larger than this
        sparse_threshold: Use sparse eigsh for matrices larger than this

    Returns:
        eigenvalues: Lowest eigenvalues
        eigenvectors: Corresponding eigenvectors
    """
    n_dim = H.shape[0]

    # Handle sparse input
    if hasattr(H, 'toarray'):
        H_dense = None  # Keep sparse
        is_sparse = True
    else:
        H_dense = np.asarray(H)
        is_sparse = False

    # Small matrices: use full diagonalization
    if n_dim < davidson_threshold:
        if is_sparse:
            H_dense = H.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
        return eigenvalues[:n_eigenvalues], eigenvectors[:, :n_eigenvalues]

    # Medium matrices: use Davidson
    elif n_dim < sparse_threshold:
        if is_sparse:
            H_dense = H.toarray()
        return davidson_eigensolver(
            H_dense, n_eigenvalues=n_eigenvalues
        )

    # Large matrices: use sparse eigsh
    else:
        if not is_sparse:
            H = csr_matrix_cpu(H)
        eigenvalues, eigenvectors = solve_generalized_eigenvalue(
            H, k=n_eigenvalues, which="SA", use_gpu=use_gpu
        )
        return eigenvalues, eigenvectors
