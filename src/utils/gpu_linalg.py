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
        H = H.to(torch.complex128) if H.is_complex() else H.double()

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


def mixed_precision_eigh(
    H: torch.Tensor,
    use_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixed-precision eigendecomposition: FP32 solve + FP64 Rayleigh quotient refinement.

    DGX Spark GB10 has FP64=0.48 TFLOPS but TF32=53 TFLOPS (110x ratio).
    Strategy:
      1. Solve eigh in FP32 (uses TF32 matmul on GPU, ~100x faster)
      2. Refine eigenvalues via FP64 Rayleigh quotient: E_i = v_i^T H_64 v_i / (v_i^T v_i)
      3. Return FP64-refined eigenvalues with FP64-upcast eigenvectors

    For small matrices or already-FP64 input, falls back to standard FP64 eigh
    since FP32 rounding offers no speedup benefit.

    Args:
        H: Hermitian matrix (n, n) — real or complex, any dtype
        use_gpu: If True, keep computation on GPU

    Returns:
        eigenvalues: (n,) tensor of FP64-refined eigenvalues in ascending order
        eigenvectors: (n, n) tensor of eigenvectors (FP64)
    """
    device = H.device

    # Move to correct device
    if use_gpu and torch.cuda.is_available() and not H.is_cuda:
        H = H.cuda()
    elif not use_gpu and H.is_cuda:
        H = H.cpu()

    n = H.shape[0]

    # For very small matrices (n <= 64), FP32 rounding noise is negligible
    # and the overhead of two solves isn't worth it. Use FP64 directly.
    if n <= 64:
        return gpu_eigh(H, use_gpu=use_gpu)

    # For already-FP64/complex128 input, the whole point of mixed-precision is
    # to avoid full FP64 eigh. But if the caller explicitly provides FP64, honor it.
    # Only apply mixed-precision when input is FP32 or lower precision.
    if H.dtype in (torch.float64, torch.complex128):
        return gpu_eigh(H, use_gpu=use_gpu)

    # Determine FP32/FP64 dtype pairs based on real vs complex input
    is_complex = H.is_complex()
    if is_complex:
        low_dtype = torch.complex64
        high_dtype = torch.complex128
    else:
        low_dtype = torch.float32
        high_dtype = torch.float64

    # Keep FP64 copy for Rayleigh refinement
    H_fp64 = H.to(high_dtype)

    # Symmetrize for numerical stability
    if is_complex:
        H_fp64 = 0.5 * (H_fp64 + H_fp64.conj().T)
    else:
        H_fp64 = 0.5 * (H_fp64 + H_fp64.T)

    # Step 1: Solve in FP32 (leverages TF32 matmul on Ampere+ GPUs)
    H_fp32 = H_fp64.to(low_dtype)
    try:
        evals_fp32, evecs_fp32 = torch.linalg.eigh(H_fp32)
    except Exception:
        # FP32 eigh failed (e.g., severe ill-conditioning) — fall back to FP64
        evals_fp64, evecs_fp64 = torch.linalg.eigh(H_fp64)
        return evals_fp64, evecs_fp64

    # Step 2: Upcast eigenvectors to FP64
    evecs_fp64 = evecs_fp32.to(high_dtype)

    # Step 3: FP64 Rayleigh quotient refinement
    # E_i = v_i^T H_64 v_i / (v_i^T v_i)
    # This recovers FP64 accuracy from FP32 eigenvectors because the
    # Rayleigh quotient is stationary at exact eigenvectors, so first-order
    # errors in v cancel out, giving second-order accurate energies.
    Hv = H_fp64 @ evecs_fp64  # (n, n) @ (n, n) = (n, n)

    # Compute Rayleigh quotient for each eigenvector
    if is_complex:
        # For complex: E_i = Re(v_i^H @ H @ v_i) / (v_i^H @ v_i)
        numerator = (evecs_fp64.conj() * Hv).sum(dim=0).real
        denominator = (evecs_fp64.conj() * evecs_fp64).sum(dim=0).real
    else:
        numerator = (evecs_fp64 * Hv).sum(dim=0)
        denominator = (evecs_fp64 * evecs_fp64).sum(dim=0)

    evals_refined = numerator / denominator

    # Sort by eigenvalue (FP32 ordering should be correct, but refine may shift)
    sort_idx = evals_refined.argsort()
    evals_refined = evals_refined[sort_idx]
    evecs_fp64 = evecs_fp64[:, sort_idx]

    return evals_refined, evecs_fp64


def sparse_hamiltonian_eigsh(
    hamiltonian,
    basis: torch.Tensor,
    k: int = 2,
    which: str = 'SA',
    shift_invert: bool = False,
    tol: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build projected Hamiltonian in sparse CSR format and solve with eigsh.

    Never materializes the full dense matrix. Memory: O(n * avg_nnz_per_row)
    instead of O(n^2).

    For molecular Hamiltonians with Slater-Condon rules, nnz/row varies:
    HF row ~7.8K, singles ~250, doubles ~36-50. Weighted avg ~54-128 for
    essential-only basis. Matrix is 0.1-6% dense — highly efficient for eigsh.

    GPU acceleration: When CuPy is available and basis is on CUDA, uses
    CuPy's GPU-accelerated eigsh (CUDA ARPACK). On DGX Spark UMA, the
    scipy→CuPy CSR conversion is near-zero-copy. Falls back to SciPy CPU
    ARPACK on failure or when CuPy is unavailable.

    Args:
        hamiltonian: MolecularHamiltonian with get_sparse_matrix_elements()
                     and diagonal_elements_batch() methods
        basis: (n_basis, n_sites) configurations
        k: Number of eigenvalues to compute
        which: 'SA' for smallest algebraic
        shift_invert: Use shift-invert mode with sigma=E_HF for faster convergence
        tol: Convergence tolerance for eigsh (0 = machine precision)

    Returns:
        eigenvalues: (k,) tensor
        eigenvectors: (n_basis, k) tensor
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh

    n = len(basis)
    device = basis.device

    # Off-diagonal elements via vectorized batch method
    rows, cols, vals = hamiltonian.get_sparse_matrix_elements(basis)
    rows_np = rows.cpu().numpy()
    cols_np = cols.cpu().numpy()
    vals_np = vals.cpu().numpy().astype(np.float64)

    H_coo = coo_matrix((vals_np, (rows_np, cols_np)), shape=(n, n))

    # Diagonal elements (vectorized, no Python loop)
    diag_np = hamiltonian.diagonal_elements_batch(basis).cpu().numpy().astype(np.float64)

    # Symmetrize off-diagonal + add diagonal
    H_csr = H_coo.tocsr()
    H_csr = 0.5 * (H_csr + H_csr.T)
    H_csr = H_csr + diags(diag_np, 0, shape=(n, n), format='csr')

    k_eig = min(k, n - 1)

    # Single-config basis: eigsh requires k >= 1 but n-1 = 0.
    # The answer is trivially the diagonal element.
    if k_eig < 1:
        eigenvalues = torch.tensor([diag_np[0]], dtype=torch.float64)
        eigenvectors = torch.eye(n, 1, dtype=torch.float64)
        return eigenvalues.to(device), eigenvectors.to(device)

    # ── CuPy GPU eigsh path ────────────────────────────────────────────
    # For basis on CUDA with CuPy available, use GPU-accelerated ARPACK.
    # On DGX Spark UMA, scipy CSR → CuPy CSR is a near-zero-copy transfer.
    # Shift-invert is not supported by CuPy eigsh, so we fall through to CPU.
    if (
        CUPY_AVAILABLE
        and device.type == "cuda"
        and not shift_invert
        and n >= 3000  # GPU overhead not worth it below ~3K (benchmark: 60x at 15K)
    ):
        try:
            H_cupy = cupy_csr(
                (
                    cp.asarray(H_csr.data),
                    cp.asarray(H_csr.indices),
                    cp.asarray(H_csr.indptr),
                ),
                shape=H_csr.shape,
            )
            eigenvalues_cp, eigenvectors_cp = cupy_eigsh(
                H_cupy, k=k_eig, which=which
            )
            evals_t = torch.as_tensor(
                eigenvalues_cp.get(), dtype=torch.float64, device=device
            )
            evecs_t = torch.as_tensor(
                eigenvectors_cp.get(), dtype=torch.float64, device=device
            )
            return evals_t, evecs_t
        except Exception:
            pass  # fall through to CPU eigsh

    # ── CPU SciPy eigsh fallback ───────────────────────────────────────
    if shift_invert:
        sigma = float(hamiltonian.diagonal_element(
            hamiltonian.get_hf_state()
        ).item())
        eigenvalues, eigenvectors = eigsh(
            H_csr, k=k_eig, sigma=sigma, which='SA', tol=tol
        )
    else:
        eigenvalues, eigenvectors = eigsh(
            H_csr, k=k_eig, which=which, tol=tol
        )

    evals_t = torch.from_numpy(eigenvalues).to(device)
    evecs_t = torch.from_numpy(eigenvectors).to(device)
    return evals_t, evecs_t


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

    # Sparse tensors must use Lanczos (matrix_exp doesn't support sparse)
    # For small dense matrices, direct matrix exponential is fastest
    if n < 10000 and not A.is_sparse:
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

    # Clamp krylov_dim to matrix size (Krylov subspace is at most n-dimensional)
    krylov_dim = min(krylov_dim, n)

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

        # Orthogonalize against previous vectors (3-term recurrence)
        if j > 0:
            w = w - beta_vec[j] * V[:, j - 1]
        w = w - alpha[j] * V[:, j]

        # Full reorthogonalization against ALL previous Lanczos vectors.
        # In finite-precision arithmetic, the 3-term recurrence loses
        # orthogonality (Paige 1980). This is O(j*n) per step but critical
        # for numerical correctness when krylov_dim > ~10.
        for i in range(j + 1):
            proj = torch.vdot(V[:, i], w)
            w = w - proj * V[:, i]

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
