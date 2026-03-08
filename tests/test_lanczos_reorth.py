"""Tests for Lanczos reorthogonalization fix (PR-A1, ADR-002).

Bug: _expm_multiply_lanczos() only does 3-term recurrence without full
reorthogonalization, causing loss of orthogonality for krylov_dim > 10.

The Lanczos algorithm operates on Hermitian matrices. In the pipeline,
gpu_expm_multiply(H, v, t=-i*dt) computes exp(-i*dt*H)@v where H is the
real-symmetric Hamiltonian. The Lanczos iteration builds a tridiagonal
decomposition of H (Hermitian), then applies exp(t*T) on the small
tridiagonal matrix T.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLanczosOrthogonality:
    """Test that Lanczos vectors maintain orthogonality."""

    def test_lanczos_orthogonality_small(self):
        """krylov_dim=10: orthogonality should be maintained."""
        from utils.gpu_linalg import _expm_multiply_lanczos

        n = 50
        torch.manual_seed(42)
        # Create a random Hermitian (real symmetric) matrix
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2

        v = torch.randn(n, dtype=torch.float64)
        v = v / torch.linalg.norm(v)

        # exp(-i*H*t) preserves norm for Hermitian H
        result = _expm_multiply_lanczos(A, v, t=-1j, krylov_dim=10)

        # Result should be normalized (unitary evolution preserves norm)
        assert abs(torch.linalg.norm(result).item() - 1.0) < 1e-8, (
            f"Norm should be ~1.0, got {torch.linalg.norm(result).item()}"
        )

    def test_lanczos_orthogonality_krylov30(self):
        """krylov_dim=30: without reorth, orthogonality is lost."""
        from utils.gpu_linalg import _expm_multiply_lanczos

        n = 200
        torch.manual_seed(42)
        # Create a Hermitian matrix with spread eigenvalues
        eigvals = torch.linspace(-10, 10, n, dtype=torch.float64)
        Q = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]
        A = Q @ torch.diag(eigvals) @ Q.T

        v = torch.randn(n, dtype=torch.float64)
        v = v / torch.linalg.norm(v)

        # exp(-i*H*t) should preserve norm
        result = _expm_multiply_lanczos(A, v, t=-1j, krylov_dim=30)

        # Unitary evolution preserves norm
        norm = torch.linalg.norm(result).item()
        assert abs(norm - 1.0) < 1e-6, (
            f"Norm preservation failed: got {norm} (should be ~1.0)"
        )


class TestLanczosAccuracy:
    """Test Lanczos exp(tA)v accuracy against exact computation."""

    def test_lanczos_vs_exact_small(self):
        """Small matrix: Lanczos should match exact matrix exponential."""
        from utils.gpu_linalg import _expm_multiply_lanczos

        n = 30
        torch.manual_seed(42)
        # Build Hermitian H, compute exp(-iH)v via Lanczos vs exact
        H = torch.randn(n, n, dtype=torch.float64)
        H = (H + H.T) / 2

        v = torch.randn(n, dtype=torch.complex128)
        v = v / torch.linalg.norm(v)

        # Exact result: exp(-iH) @ v
        exact = torch.linalg.matrix_exp(-1j * H.to(torch.complex128)) @ v

        # Lanczos result: _expm_multiply_lanczos(H, v, t=-1j)
        lanczos = _expm_multiply_lanczos(H.to(torch.complex128), v, t=-1j, krylov_dim=30)

        error = torch.linalg.norm(lanczos - exact).item()
        assert error < 1e-10, f"Lanczos error too large: {error}"

    def test_lanczos_vs_exact_spread_spectrum(self):
        """Matrix with wide eigenvalue spread -- harder for Lanczos.

        Wide spectrum [-50, 50] requires higher krylov_dim for convergence.
        Without reorthogonalization, even krylov_dim=60 would give garbage
        (10^20+ errors). With reorth, it converges cleanly.
        """
        from utils.gpu_linalg import _expm_multiply_lanczos

        n = 100
        torch.manual_seed(7)
        eigvals = torch.linspace(-50, 50, n, dtype=torch.float64)
        Q = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]
        H = Q @ torch.diag(eigvals) @ Q.T

        v = torch.randn(n, dtype=torch.complex128)
        v = v / torch.linalg.norm(v)

        # Exact: exp(-iH) @ v
        exact = torch.linalg.matrix_exp(-1j * H.to(torch.complex128)) @ v

        # Lanczos with krylov_dim=80 for wider spectrum convergence.
        # Without reorthogonalization, even krylov_dim=80 gives 10^20+ error.
        lanczos = _expm_multiply_lanczos(H.to(torch.complex128), v, t=-1j, krylov_dim=80)

        error = torch.linalg.norm(lanczos - exact).item()
        assert error < 1e-8, f"Wide-spectrum Lanczos error: {error}"


class TestLanczosUnitarity:
    """Test that e^{-iHdt} preserves state normalization."""

    def test_unitarity_single_step(self):
        """Single time evolution step preserves norm."""
        from utils.gpu_linalg import _expm_multiply_lanczos

        n = 50
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        H = (A + A.T) / 2  # Hermitian

        v = torch.randn(n, dtype=torch.complex128)
        v = v / torch.linalg.norm(v)

        # e^{-iH*dt} should preserve norm; pass H (Hermitian) with t=-i*dt
        result = _expm_multiply_lanczos(H.to(torch.complex128), v, t=-1j, krylov_dim=20)

        norm_before = torch.linalg.norm(v).item()
        norm_after = torch.linalg.norm(result).item()
        assert abs(norm_after - norm_before) < 1e-8, (
            f"Norm changed: {norm_before} -> {norm_after}"
        )

    def test_unitarity_multiple_steps(self):
        """Multiple time evolution steps accumulate preserve norm."""
        from utils.gpu_linalg import _expm_multiply_lanczos

        n = 50
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        H = (A + A.T) / 2

        v = torch.randn(n, dtype=torch.complex128)
        v = v / torch.linalg.norm(v)

        # Apply 10 time steps: exp(-i*H*0.1) each
        state = v.clone()
        H_c = H.to(torch.complex128)
        for _ in range(10):
            state = _expm_multiply_lanczos(H_c, state, t=-0.1j, krylov_dim=20)

        norm = torch.linalg.norm(state).item()
        assert abs(norm - 1.0) < 1e-6, (
            f"Norm after 10 steps: {norm} (should be ~1.0)"
        )


class TestLanczosNoRegression:
    """Verify fix doesn't break existing SKQD results."""

    def test_lih_energy_unchanged(self):
        """LiH SKQD pipeline smoke test (uses dense path, not Lanczos).

        Note: LiH has only 225 configs (< 10000 threshold), so it uses the
        dense matrix exponential path, not the Lanczos path. This test verifies
        that the overall pipeline still achieves chemical accuracy.
        """
        try:
            from hamiltonians.molecular import create_lih_hamiltonian
            from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        except ImportError:
            pytest.skip("PySCF not available")

        H = create_lih_hamiltonian(bond_length=1.6)
        config = PipelineConfig(subspace_mode="skqd", skip_nf_training=True)
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run()

        fci_energy = H.fci_energy()
        skqd_energy = results.get(
            "combined_energy",
            results.get("skqd_energy", results.get("nf_nqs_energy")),
        )
        assert skqd_energy is not None, f"No energy key found in results: {list(results.keys())}"
        error_mha = abs(skqd_energy - fci_energy) * 1000

        # Should be within chemical accuracy (1.6 mHa)
        assert error_mha < 1.6, f"LiH error {error_mha:.3f} mHa exceeds chemical accuracy"
