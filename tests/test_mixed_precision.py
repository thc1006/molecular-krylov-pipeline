"""Tests for PR 1.8: Mixed-precision eigensolver.

DGX Spark GB10 has FP64=0.48 TFLOPS but TF32=53 TFLOPS (110x ratio).
Building and solving in FP32/TF32 then refining in FP64 can give ~100x speedup
while maintaining chemical accuracy.

Strategy:
1. Build Hamiltonian in FP32 (uses TF32 matmul on GPU)
2. Solve eigenproblem in FP32 (fast)
3. Rayleigh quotient refinement in FP64 (accurate)
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMixedPrecisionEigh:
    """Test FP32 eigensolver with FP64 refinement."""

    @pytest.mark.molecular
    def test_mixed_precision_function_exists(self):
        """mixed_precision_eigh should be importable."""
        from utils.gpu_linalg import mixed_precision_eigh
        assert callable(mixed_precision_eigh)

    @pytest.mark.molecular
    def test_mixed_precision_matches_fp64_h2(self, h2_hamiltonian):
        """Mixed-precision energy must match FP64 within 0.1 mHa for H2."""
        from utils.gpu_linalg import mixed_precision_eigh, gpu_eigh

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        H_mat = H.matrix_elements(configs, configs).detach()
        H_sym = 0.5 * (H_mat + H_mat.T)

        # FP64 reference
        evals_64, evecs_64 = gpu_eigh(H_sym.double(), use_gpu=False)
        E_fp64 = float(evals_64[0])

        # Mixed precision
        evals_mp, evecs_mp = mixed_precision_eigh(H_sym)
        E_mp = float(evals_mp[0])

        assert abs(E_fp64 - E_mp) < 1e-4, (
            f"Mixed-precision {E_mp:.10f} != FP64 {E_fp64:.10f}, "
            f"diff = {abs(E_fp64 - E_mp):.2e}"
        )

    @pytest.mark.molecular
    def test_mixed_precision_matches_fp64_lih(self, lih_hamiltonian):
        """Mixed-precision must match FP64 within 0.1 mHa for LiH (225 configs)."""
        from utils.gpu_linalg import mixed_precision_eigh, gpu_eigh
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        H_mat = lih_hamiltonian.matrix_elements(basis, basis).detach()
        H_sym = 0.5 * (H_mat + H_mat.T)

        # FP64 reference
        evals_64, _ = gpu_eigh(H_sym.double(), use_gpu=False)
        E_fp64 = float(evals_64[0])

        # Mixed precision
        evals_mp, _ = mixed_precision_eigh(H_sym)
        E_mp = float(evals_mp[0])

        error_mha = abs(E_fp64 - E_mp) * 1000
        assert error_mha < 0.1, (
            f"Mixed-precision error {error_mha:.4f} mHa exceeds 0.1 mHa tolerance"
        )

    @pytest.mark.molecular
    def test_mixed_precision_matches_fp64_beh2(self, beh2_hamiltonian):
        """Mixed-precision must match FP64 within 0.1 mHa for BeH2 (1225 configs)."""
        from utils.gpu_linalg import mixed_precision_eigh, gpu_eigh
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(beh2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        H_mat = beh2_hamiltonian.matrix_elements(basis, basis).detach()
        H_sym = 0.5 * (H_mat + H_mat.T)

        evals_64, _ = gpu_eigh(H_sym.double(), use_gpu=False)
        E_fp64 = float(evals_64[0])

        evals_mp, _ = mixed_precision_eigh(H_sym)
        E_mp = float(evals_mp[0])

        error_mha = abs(E_fp64 - E_mp) * 1000
        assert error_mha < 0.1, (
            f"Mixed-precision error {error_mha:.4f} mHa on BeH2"
        )

    @pytest.mark.molecular
    def test_mixed_precision_eigenvectors_orthogonal(self, lih_hamiltonian):
        """Mixed-precision eigenvectors must be orthonormal."""
        from utils.gpu_linalg import mixed_precision_eigh
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        H_mat = lih_hamiltonian.matrix_elements(basis, basis).detach()
        H_sym = 0.5 * (H_mat + H_mat.T)

        _, evecs = mixed_precision_eigh(H_sym)

        # Check first 5 eigenvectors are orthonormal
        k = min(5, evecs.shape[1])
        V = evecs[:, :k].double()
        overlap = V.T @ V
        identity = torch.eye(k, dtype=torch.float64, device=V.device)
        max_err = (overlap - identity).abs().max().item()
        # FP32 eigenvectors upcast to FP64: orthogonality limited by FP32 precision
        assert max_err < 1e-5, f"Eigenvectors not orthonormal, max overlap error = {max_err:.2e}"


class TestMixedPrecisionIntegration:
    """Test mixed-precision integrated into pipeline eigensolver."""

    @pytest.mark.molecular
    def test_pipeline_mixed_precision_h2(self, h2_hamiltonian):
        """Pipeline with mixed_precision=True must still be accurate for H2."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = h2_hamiltonian.fci_energy()

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(h2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(h2_hamiltonian, config=config)

        E, _ = skqd.compute_ground_state_energy(basis, mixed_precision=True)
        error_mha = abs(E - e_fci) * 1000
        assert error_mha < 1.0, f"Mixed-precision pipeline error {error_mha:.4f} mHa"

    @pytest.mark.molecular
    def test_pipeline_mixed_precision_lih(self, lih_hamiltonian):
        """Pipeline with mixed_precision=True must achieve chemical accuracy for LiH."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = lih_hamiltonian.fci_energy()

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)

        E, _ = skqd.compute_ground_state_energy(basis, mixed_precision=True)
        error_mha = abs(E - e_fci) * 1000
        assert error_mha < 1.0, f"Mixed-precision pipeline error {error_mha:.4f} mHa on LiH"

    @pytest.mark.molecular
    def test_mixed_precision_default_off(self, h2_hamiltonian):
        """mixed_precision should default to False for backward compatibility."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(h2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(h2_hamiltonian, config=config)

        # Should work without mixed_precision argument (backward compatible)
        E, _ = skqd.compute_ground_state_energy(basis)
        assert isinstance(E, float)


class TestRayleighRefinement:
    """Test Rayleigh quotient refinement specifically."""

    @pytest.mark.molecular
    def test_rayleigh_improves_fp32_energy(self, lih_hamiltonian):
        """FP64 Rayleigh quotient must be closer to FP64 eigh than raw FP32."""
        from utils.gpu_linalg import mixed_precision_eigh
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        H_mat = lih_hamiltonian.matrix_elements(basis, basis).detach()
        H_sym = 0.5 * (H_mat + H_mat.T)

        # FP64 ground truth
        evals_64, _ = torch.linalg.eigh(H_sym.double())
        E_fp64 = float(evals_64[0])

        # Raw FP32 (no refinement)
        evals_32, evecs_32 = torch.linalg.eigh(H_sym.float())
        E_fp32_raw = float(evals_32[0])

        # FP64 Rayleigh quotient on FP32 eigenvector
        v = evecs_32[:, 0].double()
        H64 = H_sym.double()
        E_rayleigh = float((v @ H64 @ v) / (v @ v))

        err_raw = abs(E_fp32_raw - E_fp64)
        err_rayleigh = abs(E_rayleigh - E_fp64)

        print(f"  FP64:     {E_fp64:.12f}")
        print(f"  FP32 raw: {E_fp32_raw:.12f} (err={err_raw:.2e})")
        print(f"  Rayleigh: {E_rayleigh:.12f} (err={err_rayleigh:.2e})")

        assert err_rayleigh <= err_raw, (
            f"Rayleigh refinement made it worse: {err_rayleigh:.2e} > {err_raw:.2e}"
        )


class TestEdgeCases:
    """Edge cases for mixed-precision eigensolver."""

    def test_mixed_precision_1x1(self):
        """1x1 matrix should work."""
        from utils.gpu_linalg import mixed_precision_eigh

        H = torch.tensor([[3.14]], dtype=torch.float32)
        evals, evecs = mixed_precision_eigh(H)
        assert abs(evals[0].item() - 3.14) < 1e-5
        assert evecs.shape == (1, 1)

    def test_mixed_precision_2x2(self):
        """2x2 known eigenvalue problem."""
        from utils.gpu_linalg import mixed_precision_eigh

        H = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        evals, evecs = mixed_precision_eigh(H)
        # Eigenvalues should be 1.0 and 3.0
        assert abs(evals[0].item() - 1.0) < 1e-5
        assert abs(evals[1].item() - 3.0) < 1e-5

    def test_mixed_precision_already_fp64(self):
        """FP64 input should still work (no double conversion)."""
        from utils.gpu_linalg import mixed_precision_eigh

        H = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        evals, evecs = mixed_precision_eigh(H)
        assert abs(evals[0].item() - 1.0) < 1e-10

    def test_mixed_precision_complex_input(self):
        """Complex Hermitian input should work."""
        from utils.gpu_linalg import mixed_precision_eigh

        H = torch.tensor([[2.0+0j, 1.0+0j], [1.0+0j, 2.0+0j]], dtype=torch.complex64)
        evals, evecs = mixed_precision_eigh(H)
        assert abs(evals[0].item() - 1.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
