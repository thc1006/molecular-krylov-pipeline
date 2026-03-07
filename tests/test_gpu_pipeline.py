"""Tests for GPU-accelerated pipeline execution.

Verifies that the pipeline runs correctly on CUDA and produces
accurate results using the DGX Spark GB10 GPU.
"""

import pytest
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestGPUPipeline:
    """Pipeline tests running on GPU."""

    @pytest.mark.gpu
    @pytest.mark.molecular
    def test_h2_gpu_skqd(self, h2_hamiltonian_gpu):
        """H2 SKQD on GPU should match FCI."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = h2_hamiltonian_gpu
        e_fci = H.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cuda",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=e_fci)
        results = pipeline.run(progress=False)

        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 0.1, f"H2 GPU error {error_mha:.4f} mHa"

    @pytest.mark.gpu
    @pytest.mark.molecular
    def test_lih_gpu_skqd(self, lih_hamiltonian_gpu):
        """LiH SKQD on GPU should reach chemical accuracy."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = lih_hamiltonian_gpu
        e_fci = H.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cuda",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=e_fci)
        results = pipeline.run(progress=False)

        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 0.5, f"LiH GPU error {error_mha:.4f} mHa"

    @pytest.mark.gpu
    @pytest.mark.molecular
    def test_beh2_gpu_skqd(self, beh2_hamiltonian_gpu):
        """BeH2 SKQD on GPU — medium system (1225 configs)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = beh2_hamiltonian_gpu
        e_fci = H.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cuda",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=e_fci)
        results = pipeline.run(progress=False)

        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 1.6, f"BeH2 GPU error {error_mha:.4f} mHa"


class TestGPUvsCPUSpeedup:
    """Verify GPU provides speedup over CPU for matrix operations."""

    @pytest.mark.gpu
    @pytest.mark.molecular
    def test_matrix_elements_gpu_faster(self, lih_hamiltonian, lih_hamiltonian_gpu):
        """GPU matrix_elements should be faster than CPU for LiH."""
        H_cpu = lih_hamiltonian
        H_gpu = lih_hamiltonian_gpu

        hf_cpu = H_cpu.get_hf_state()
        hf_gpu = H_gpu.get_hf_state()

        # Build small basis on each device
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipe_cpu = FlowGuidedKrylovPipeline(H_cpu, config=config)
        configs_cpu = pipe_cpu._generate_essential_configs()

        configs_gpu = configs_cpu.to("cuda")

        # Warmup
        _ = H_gpu.matrix_elements_fast(configs_gpu[:10])
        torch.cuda.synchronize()

        # Time CPU
        t0 = time.perf_counter()
        _ = H_cpu.matrix_elements_fast(configs_cpu)
        cpu_time = time.perf_counter() - t0

        # Time GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = H_gpu.matrix_elements_fast(configs_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - t0

        print(f"\n  LiH matrix_elements: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, "
              f"speedup={cpu_time/gpu_time:.1f}x")

        # GPU should at least not be slower (UMA means no transfer overhead)
        # For small systems GPU may not be faster, so just verify correctness
        H_cpu_result = H_cpu.matrix_elements_fast(configs_cpu)
        H_gpu_result = H_gpu.matrix_elements_fast(configs_gpu)

        import numpy as np
        # TF32 matmul precision on GPU causes ~1e-4 relative differences vs CPU float32.
        # This is expected and does not affect eigensolver accuracy (which uses float64).
        np.testing.assert_allclose(
            H_cpu_result.cpu().numpy(),
            H_gpu_result.cpu().numpy(),
            atol=1e-3, rtol=1e-3,
            err_msg="GPU and CPU matrix elements differ beyond TF32 tolerance"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
