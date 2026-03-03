"""
GPU-accelerated linear algebra utilities.

Provides unified interface for eigensolvers and matrix exponential
that stay entirely on GPU, avoiding CPU transfers.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union

# Check for CuPy availability
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def gpu_eigh(
    H: torch.Tensor,
    use_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated dense eigendecomposition.

    Uses torch.linalg.eigh which runs on GPU when input is CUDA tensor.
    Automatically handles symmetrization for numerical stability.

    Args:
        H: Hermitian matrix (n, n) - can be on CPU or GPU
        use_gpu: If True, keep computation on GPU

    Returns:
        eigenvalues: (n,) tensor of eigenvalues in ascending order
        eigenvectors: (n, n) tensor where column i is eigenvector for eigenvalue i
    """
    device = H.device
    dtype = H.dtype

    # Ensure matrix is on correct device
    if use_gpu and torch.cuda.is_available() and not H.is_cuda:
        H = H.cuda()
    elif not use_gpu and H.is_cuda:
        H = H.cpu()

    # Use float64 for numerical precision
    if dtype not in (torch.float64, torch.complex128):
        H = H.double()

    # Ensure Hermitian (symmetrize for numerical stability)
    if H.is_complex():
        H = 0.5 * (H + H.conj().T)
    else:
        H = 0.5 * (H + H.T)

    # PyTorch's eigh runs on GPU when input is CUDA tensor
    eigenvalues, eigenvectors = torch.linalg.eigh(H)

    return eigenvalues, eigenvectors


def gpu_eigsh(
    H: torch.Tensor,
    k: int = 1,
    which: str = 'SA',
    use_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated sparse/large matrix eigendecomposition.

    For matrices that fit in GPU memory, uses dense torch.linalg.eigh.
    Falls back to CuPy sparse eigsh if available for very large matrices.

    Args:
        H: Hermitian matrix (n, n)
        k: Number of eigenvalues/vectors to compute
        which: 'SA' for smallest algebraic, 'LA' for largest
        use_gpu: If True, keep computation on GPU

    Returns:
        eigenvalues: (k,) tensor of eigenvalues
        eigenvectors: (n, k) tensor of eigenvectors
    """
    n = H.shape[0]

    # For moderate sizes, dense GPU eigensolver is fastest
    # Threshold chosen based on typical GPU memory (16-48GB)
    dense_threshold = 10000

    if n <= dense_threshold:
        # Use dense eigensolver
        eigenvalues, eigenvectors = gpu_eigh(H, use_gpu=use_gpu)

        # Select requested eigenvalues
        if which == 'SA':
            # Smallest algebraic (already sorted ascending)
            return eigenvalues[:k], eigenvectors[:, :k]
        elif which == 'LA':
            # Largest algebraic
            return eigenvalues[-k:].flip(0), eigenvectors[:, -k:].flip(1)
        else:
            raise ValueError(f"Unknown which={which}, expected 'SA' or 'LA'")

    # For very large matrices, try CuPy sparse
    if use_gpu and CUPY_AVAILABLE and torch.cuda.is_available():
        try:
            # Zero-copy GPU→CuPy via DLPack when tensor is already on CUDA
            if H.is_cuda:
                H_gpu = cp.from_dlpack(H.detach().contiguous())
            else:
                H_gpu = cp.asarray(H.numpy())
            H_sparse = cupy_csr(H_gpu)

            eigenvalues_cp, eigenvectors_cp = cupy_eigsh(
                H_sparse, k=k, which=which
            )

            eigenvalues = torch.from_numpy(cp.asnumpy(eigenvalues_cp)).cuda()
            eigenvectors = torch.from_numpy(cp.asnumpy(eigenvectors_cp)).cuda()

            return eigenvalues, eigenvectors
        except Exception as e:
            print(f"CuPy sparse eigsh failed: {e}, falling back to dense")

    # Fallback to dense
    eigenvalues, eigenvectors = gpu_eigh(H, use_gpu=use_gpu)

    if which == 'SA':
        return eigenvalues[:k], eigenvectors[:, :k]
    elif which == 'LA':
        return eigenvalues[-k:].flip(0), eigenvectors[:, -k:].flip(1)
    else:
        raise ValueError(f"Unknown which={which}")


def gpu_expm_multiply(
    A: torch.Tensor,
    v: torch.Tensor,
    t: complex = 1.0,
    krylov_dim: int = 30,
    tol: float = 1e-12,
) -> torch.Tensor:
    """
    GPU-accelerated computation of exp(t*A) @ v.

    Uses direct matrix exponential for matrices that fit in GPU memory,
    which is highly optimized in torch.linalg.matrix_exp.

    For very large matrices (> 10000), falls back to Lanczos approximation
    to avoid memory issues.

    Args:
        A: Hermitian matrix (n, n) - typically -i*H for time evolution
        v: Vector (n,) to multiply
        t: Scalar multiplier (typically -i*dt for time evolution)
        krylov_dim: Maximum Krylov subspace dimension
        tol: Convergence tolerance

    Returns:
        result: exp(t*A) @ v as a tensor on same device as v
    """
    n = v.shape[0]
    device = v.device
    dtype = v.dtype

    # For matrices that fit in GPU memory, direct matrix exponential is fastest
    # GPU memory is typically 16-48GB, so 10k x 10k complex128 = 1.6GB is safe
    if n < 10000:
        return _expm_multiply_dense(A, v, t)

    # For very large systems, use Krylov approximation to save memory
    return _expm_multiply_lanczos(A, v, t, krylov_dim, tol)


def _expm_multiply_dense(
    A: torch.Tensor,
    v: torch.Tensor,
    t: complex = 1.0,
) -> torch.Tensor:
    """Direct matrix exponential for small systems."""
    # Ensure complex dtype for time evolution
    if not A.is_complex():
        A = A.to(torch.complex128)
    if not v.is_complex():
        v = v.to(torch.complex128)

    expA = torch.linalg.matrix_exp(t * A)
    return expA @ v


def _expm_multiply_lanczos(
    A: torch.Tensor,
    v: torch.Tensor,
    t: complex = 1.0,
    krylov_dim: int = 30,
    tol: float = 1e-12,
) -> torch.Tensor:
    """
    Lanczos-based approximation of exp(t*A) @ v.

    This is the GPU equivalent of scipy.sparse.linalg.expm_multiply.
    All operations stay on GPU.
    """
    n = v.shape[0]
    device = v.device

    # Ensure complex dtype
    if not A.is_complex():
        A = A.to(torch.complex128)
    if not v.is_complex():
        v = v.to(torch.complex128)

    # Normalize input vector
    beta = torch.linalg.norm(v)
    if beta < tol:
        return torch.zeros_like(v)

    # Allocate Lanczos vectors and tridiagonal matrix
    V = torch.zeros(n, krylov_dim + 1, dtype=v.dtype, device=device)
    alpha = torch.zeros(krylov_dim, dtype=torch.float64, device=device)
    beta_vec = torch.zeros(krylov_dim + 1, dtype=torch.float64, device=device)

    V[:, 0] = v / beta
    beta_vec[0] = beta

    # Lanczos iteration
    actual_dim = krylov_dim
    for j in range(krylov_dim):
        # Matrix-vector product (main GPU operation)
        w = A @ V[:, j]

        # Compute diagonal element
        alpha[j] = torch.vdot(V[:, j], w).real

        # Orthogonalize against previous vectors
        if j > 0:
            w = w - beta_vec[j] * V[:, j - 1]
        w = w - alpha[j] * V[:, j]

        # Compute off-diagonal element
        beta_new = torch.linalg.norm(w).real
        beta_vec[j + 1] = beta_new

        # Check for breakdown (invariant subspace found)
        if beta_new < tol:
            actual_dim = j + 1
            break

        V[:, j + 1] = w / beta_new

    # Build tridiagonal matrix T
    T = torch.zeros(actual_dim, actual_dim, dtype=v.dtype, device=device)
    for j in range(actual_dim):
        T[j, j] = alpha[j]
        if j > 0:
            T[j, j - 1] = beta_vec[j]
            T[j - 1, j] = beta_vec[j]

    # Compute exp(t*T) on small matrix
    expT = torch.linalg.matrix_exp(t * T)

    # Project back: result = beta * V @ expT @ e_1
    # e_1 = [1, 0, 0, ...]
    result = beta_vec[0] * (V[:, :actual_dim] @ expT[:, 0])

    return result


def gpu_sparse_mv(
    H_sparse: 'torch.sparse.Tensor',
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse matrix-vector multiply on GPU.

    Args:
        H_sparse: Sparse matrix in COO or CSR format
        v: Dense vector

    Returns:
        H_sparse @ v
    """
    return torch.sparse.mm(H_sparse.unsqueeze(0), v.unsqueeze(1)).squeeze()


def build_sparse_hamiltonian_gpu(
    hamiltonian,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Build sparse Hamiltonian matrix on GPU.

    Uses the Hamiltonian's matrix_elements method but keeps result on GPU.

    Args:
        hamiltonian: Hamiltonian object with matrix_elements method
        basis: Basis configurations (n_basis, n_sites)

    Returns:
        H: Dense Hamiltonian matrix on GPU (n_basis, n_basis)
    """
    device = basis.device

    # Build matrix using existing method
    H = hamiltonian.matrix_elements(basis, basis)

    # Ensure on GPU
    if not H.is_cuda and torch.cuda.is_available():
        H = H.cuda()

    return H


# Convenience function for ground state computation
def compute_ground_state_gpu(
    H: torch.Tensor,
    return_eigenvector: bool = False,
) -> Union[float, Tuple[float, torch.Tensor]]:
    """
    Compute ground state energy (and optionally eigenvector) on GPU.

    Args:
        H: Hamiltonian matrix on GPU
        return_eigenvector: If True, also return ground state eigenvector

    Returns:
        energy: Ground state energy
        eigenvector: (optional) Ground state eigenvector
    """
    eigenvalues, eigenvectors = gpu_eigh(H, use_gpu=True)

    E0 = float(eigenvalues[0].cpu())

    if return_eigenvector:
        return E0, eigenvectors[:, 0]
    return E0
