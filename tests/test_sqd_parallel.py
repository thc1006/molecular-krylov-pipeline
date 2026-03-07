"""Tests for PR 1.5: SQD batch parallelization.

The SQD algorithm runs K independent batch diagonalizations per self-consistent
iteration. These are embarrassingly parallel — each batch builds its own projected
Hamiltonian and solves an independent eigenvalue problem.

On DGX Spark with 20 ARM cores, parallelizing 5-10 batches should give 5-8x speedup.
"""

import pytest
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSQDParallelBatches:
    """Test that parallel batch diag produces correct results."""

    @pytest.mark.molecular
    def test_parallel_matches_sequential(self, lih_hamiltonian):
        """Parallel batch results must match sequential within floating-point tolerance."""
        from krylov.sqd import SQDSolver, SQDConfig

        config = SQDConfig(
            num_batches=5,
            self_consistent_iters=1,
            noise_rate=0.0,
            max_workers=1,  # sequential
        )
        solver_seq = SQDSolver(lih_hamiltonian, config=config)

        config_par = SQDConfig(
            num_batches=5,
            self_consistent_iters=1,
            noise_rate=0.0,
            max_workers=4,  # parallel
        )
        solver_par = SQDSolver(lih_hamiltonian, config=config_par)

        # Generate a basis
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        results_seq = solver_seq.run(basis, progress=False)
        results_par = solver_par.run(basis, progress=False)

        # Energies should match (same batches, same seeds)
        e_seq = results_seq["energy"]
        e_par = results_par["energy"]
        assert abs(e_seq - e_par) < 1e-8, (
            f"Parallel energy {e_par:.10f} != sequential {e_seq:.10f}"
        )

    @pytest.mark.molecular
    def test_parallel_config_exists(self, h2_hamiltonian):
        """SQDConfig should accept max_workers parameter."""
        from krylov.sqd import SQDConfig

        config = SQDConfig(max_workers=4)
        assert config.max_workers == 4

    @pytest.mark.molecular
    def test_parallel_default_is_one(self):
        """Default max_workers should be 1 (sequential, backward compatible)."""
        from krylov.sqd import SQDConfig

        config = SQDConfig()
        assert config.max_workers == 1

    @pytest.mark.molecular
    def test_parallel_energy_accuracy(self, lih_hamiltonian):
        """Parallel SQD must still achieve chemical accuracy on LiH."""
        from krylov.sqd import SQDSolver, SQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = lih_hamiltonian.fci_energy()

        config = SQDConfig(
            num_batches=5,
            self_consistent_iters=2,
            noise_rate=0.0,
            max_workers=4,
        )
        solver = SQDSolver(lih_hamiltonian, config=config)

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        results = solver.run(basis, progress=False)
        error_mha = abs(results["energy"] - e_fci) * 1000
        assert error_mha < 5.0, f"Parallel SQD error {error_mha:.4f} mHa too large"


class TestSQDParallelSpeedup:
    """Verify parallel batches provide actual speedup."""

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_parallel_faster_than_sequential(self, beh2_hamiltonian):
        """With 4 workers on a 1225-config system, parallel should be faster."""
        from krylov.sqd import SQDSolver, SQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(beh2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        # Sequential
        config_seq = SQDConfig(num_batches=8, self_consistent_iters=1, max_workers=1)
        solver_seq = SQDSolver(beh2_hamiltonian, config=config_seq)
        t0 = time.perf_counter()
        solver_seq.run(basis, progress=False)
        t_seq = time.perf_counter() - t0

        # Parallel
        config_par = SQDConfig(num_batches=8, self_consistent_iters=1, max_workers=8)
        solver_par = SQDSolver(beh2_hamiltonian, config=config_par)
        t0 = time.perf_counter()
        solver_par.run(basis, progress=False)
        t_par = time.perf_counter() - t0

        speedup = t_seq / t_par
        print(f"\n  BeH2 SQD 8 batches: seq={t_seq:.2f}s, par={t_par:.2f}s, speedup={speedup:.1f}x")

        # Even modest speedup proves parallelization works
        # On 20-core DGX Spark with 8 batches, expect 3-6x
        assert speedup > 1.5, f"Parallel speedup only {speedup:.1f}x, expected > 1.5x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
