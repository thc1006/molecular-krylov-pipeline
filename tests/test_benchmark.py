"""Tests for benchmark infrastructure (wall-time + memory tracking)."""

import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.benchmark import BenchmarkTimer, MemoryTracker, BenchmarkSuite, BenchmarkResult


class TestBenchmarkResult:
    """Test the BenchmarkResult dataclass."""

    def test_result_stores_label(self):
        """BenchmarkResult should store a label."""
        r = BenchmarkResult(label="test_op", elapsed_seconds=1.0)
        assert r.label == "test_op"

    def test_result_stores_elapsed_seconds(self):
        """BenchmarkResult should store elapsed_seconds."""
        r = BenchmarkResult(label="op", elapsed_seconds=2.5)
        assert r.elapsed_seconds == 2.5

    def test_result_default_peak_memory(self):
        """Peak memory should default to 0."""
        r = BenchmarkResult(label="op", elapsed_seconds=1.0)
        assert r.peak_memory_bytes == 0

    def test_result_stores_peak_memory(self):
        """BenchmarkResult should store peak_memory_bytes."""
        r = BenchmarkResult(label="op", elapsed_seconds=1.0, peak_memory_bytes=1024)
        assert r.peak_memory_bytes == 1024

    def test_result_default_metadata(self):
        """Metadata should default to empty dict."""
        r = BenchmarkResult(label="op", elapsed_seconds=1.0)
        assert r.metadata == {}

    def test_result_stores_metadata(self):
        """BenchmarkResult should store arbitrary metadata."""
        r = BenchmarkResult(label="op", elapsed_seconds=1.0, metadata={"n_configs": 14400})
        assert r.metadata["n_configs"] == 14400


class TestBenchmarkTimer:
    """Test wall-time measurement."""

    def test_timer_measures_elapsed_time(self):
        """BenchmarkTimer should measure elapsed wall-clock time."""
        timer = BenchmarkTimer(label="sleep_test")
        with timer:
            time.sleep(0.05)
        assert timer.elapsed_seconds >= 0.04  # Allow small scheduling variance

    def test_timer_context_manager(self):
        """Should work as context manager: with BenchmarkTimer() as t: ..."""
        with BenchmarkTimer(label="ctx") as t:
            x = sum(range(1000))
        assert t.elapsed_seconds > 0

    def test_timer_records_label(self):
        """Timer should store a label for the operation being timed."""
        timer = BenchmarkTimer(label="my_operation")
        assert timer.label == "my_operation"

    def test_timer_result_has_seconds(self):
        """Result should have elapsed_seconds attribute."""
        with BenchmarkTimer(label="res") as t:
            time.sleep(0.01)
        result = t.result()
        assert isinstance(result, BenchmarkResult)
        assert result.elapsed_seconds >= 0.005
        assert result.label == "res"

    def test_timer_default_label(self):
        """Timer with no label should default to empty string."""
        timer = BenchmarkTimer()
        assert timer.label == ""

    def test_timer_initial_elapsed_is_zero(self):
        """Before entering context, elapsed_seconds should be 0."""
        timer = BenchmarkTimer(label="test")
        assert timer.elapsed_seconds == 0.0

    def test_timer_measures_compute(self):
        """Timer should measure actual compute, not just sleep."""
        with BenchmarkTimer(label="compute") as t:
            total = 0
            for i in range(100000):
                total += i
        assert t.elapsed_seconds > 0


class TestMemoryTracker:
    """Test peak memory measurement."""

    def test_memory_tracker_measures_peak(self):
        """MemoryTracker should measure peak memory usage in bytes."""
        with MemoryTracker(label="mem_test") as m:
            data = [0] * 10000
        assert m.peak_memory_bytes > 0

    def test_memory_tracker_context_manager(self):
        """Should work as context manager."""
        with MemoryTracker(label="ctx") as m:
            pass
        assert isinstance(m.peak_memory_bytes, int)

    def test_memory_allocation_detected(self):
        """Allocating memory inside tracker should increase peak measurement."""
        with MemoryTracker(label="small") as m_small:
            small = bytearray(1000)

        with MemoryTracker(label="large") as m_large:
            large = bytearray(1_000_000)

        # Larger allocation should result in larger peak
        assert m_large.peak_memory_bytes > m_small.peak_memory_bytes

    def test_memory_tracker_result(self):
        """Result should have peak_memory_bytes."""
        with MemoryTracker(label="mem") as m:
            data = bytearray(5000)
        result = m.result()
        assert isinstance(result, BenchmarkResult)
        assert result.peak_memory_bytes > 0
        assert result.label == "mem"

    def test_memory_tracker_default_label(self):
        """Tracker with no label should default to empty string."""
        tracker = MemoryTracker()
        assert tracker.label == ""


class TestBenchmarkSuite:
    """Test suite for collecting and comparing benchmarks."""

    def test_suite_collects_results(self):
        """BenchmarkSuite should collect multiple benchmark results."""
        suite = BenchmarkSuite()
        suite.add(BenchmarkResult(label="op1", elapsed_seconds=1.0))
        suite.add(BenchmarkResult(label="op2", elapsed_seconds=2.0))
        assert len(suite.results) == 2

    def test_suite_summary_format(self):
        """Summary should include label, time, memory for each benchmark."""
        suite = BenchmarkSuite()
        suite.add(BenchmarkResult(label="op1", elapsed_seconds=1.234, peak_memory_bytes=1048576))
        suite.add(BenchmarkResult(label="op2", elapsed_seconds=0.567, peak_memory_bytes=2097152))

        summary = suite.summary()

        assert "op1" in summary
        assert "op2" in summary
        assert "1.234" in summary
        assert "1.0" in summary  # 1 MB
        assert "2.0" in summary  # 2 MB

    def test_suite_comparison_detects_regression(self):
        """Suite should detect regressions vs baseline."""
        baseline = BenchmarkSuite()
        baseline.add(BenchmarkResult(label="op1", elapsed_seconds=1.0))

        current = BenchmarkSuite()
        current.add(BenchmarkResult(label="op1", elapsed_seconds=3.0))

        # 3x slower should be flagged with default threshold=2.0
        regressions = current.check_regression(baseline, threshold=2.0)
        assert len(regressions) == 1
        assert "op1" in regressions[0]

    def test_suite_comparison_no_regression(self):
        """Results within threshold should not be flagged."""
        baseline = BenchmarkSuite()
        baseline.add(BenchmarkResult(label="op1", elapsed_seconds=1.0))

        current = BenchmarkSuite()
        current.add(BenchmarkResult(label="op1", elapsed_seconds=1.5))

        # 1.5x slower is within default threshold=2.0
        regressions = current.check_regression(baseline, threshold=2.0)
        assert len(regressions) == 0

    def test_suite_comparison_missing_baseline(self):
        """New operations without baseline should not be flagged."""
        baseline = BenchmarkSuite()
        baseline.add(BenchmarkResult(label="op1", elapsed_seconds=1.0))

        current = BenchmarkSuite()
        current.add(BenchmarkResult(label="op2", elapsed_seconds=5.0))

        regressions = current.check_regression(baseline, threshold=2.0)
        assert len(regressions) == 0

    def test_suite_comparison_zero_baseline(self):
        """Zero-time baseline should not cause division error."""
        baseline = BenchmarkSuite()
        baseline.add(BenchmarkResult(label="op1", elapsed_seconds=0.0))

        current = BenchmarkSuite()
        current.add(BenchmarkResult(label="op1", elapsed_seconds=1.0))

        # Should not raise
        regressions = current.check_regression(baseline, threshold=2.0)
        assert isinstance(regressions, list)

    def test_suite_empty(self):
        """Empty suite should produce valid summary."""
        suite = BenchmarkSuite()
        summary = suite.summary()
        assert "Benchmark Results" in summary

    def test_suite_add_from_timer(self):
        """Suite should accept results from BenchmarkTimer."""
        suite = BenchmarkSuite()
        with BenchmarkTimer(label="timed_op") as t:
            time.sleep(0.01)
        suite.add(t.result())
        assert len(suite.results) == 1
        assert suite.results[0].label == "timed_op"

    def test_suite_add_from_memory_tracker(self):
        """Suite should accept results from MemoryTracker."""
        suite = BenchmarkSuite()
        with MemoryTracker(label="mem_op") as m:
            data = bytearray(1000)
        suite.add(m.result())
        assert len(suite.results) == 1
        assert suite.results[0].peak_memory_bytes > 0


class TestPipelineBenchmarks:
    """Benchmark tests for molecular pipeline operations."""

    @pytest.mark.molecular
    def test_benchmark_h2_hamiltonian_creation(self, h2_hamiltonian):
        """H2 Hamiltonian creation should be fast (4 qubits, 4 configs)."""
        with BenchmarkTimer(label="h2_creation") as t:
            from hamiltonians.molecular import create_h2_hamiltonian

            H = create_h2_hamiltonian(bond_length=0.74, device="cpu")
        result = t.result()
        # H2 creation should complete in < 30s even cold
        assert result.elapsed_seconds < 30.0
        assert result.label == "h2_creation"

    @pytest.mark.molecular
    def test_benchmark_h2_pipeline(self, h2_hamiltonian):
        """Full H2 pipeline should complete in reasonable time."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )

        suite = BenchmarkSuite()

        with BenchmarkTimer(label="h2_pipeline") as t:
            pipeline = FlowGuidedKrylovPipeline(
                h2_hamiltonian, config=config, auto_adapt=False
            )
            results = pipeline.run(progress=False)

        suite.add(t.result())

        # H2 pipeline (4 configs) should complete quickly
        assert t.elapsed_seconds < 60.0
        # Should produce a valid energy
        assert "skqd_energy" in results or "best_energy" in results

    @pytest.mark.molecular
    def test_benchmark_lih_eigensolver(self, lih_hamiltonian):
        """LiH eigensolver benchmark (225 configs)."""
        import numpy as np

        suite = BenchmarkSuite()

        with BenchmarkTimer(label="lih_fci") as t:
            E_fci = lih_hamiltonian.fci_energy()

        suite.add(t.result())
        assert t.elapsed_seconds < 60.0
        assert E_fci < 0  # LiH ground state is negative

    @pytest.mark.molecular
    def test_benchmark_h2o_pipeline(self, h2o_hamiltonian):
        """H2O pipeline benchmark (441 configs)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )

        with BenchmarkTimer(label="h2o_pipeline") as t:
            pipeline = FlowGuidedKrylovPipeline(
                h2o_hamiltonian, config=config, auto_adapt=False
            )
            results = pipeline.run(progress=False)

        assert t.elapsed_seconds < 120.0

    @pytest.mark.molecular
    def test_benchmark_memory_lih_pipeline(self, lih_hamiltonian):
        """LiH pipeline memory usage should be reasonable."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )

        with MemoryTracker(label="lih_pipeline_mem") as m:
            pipeline = FlowGuidedKrylovPipeline(
                lih_hamiltonian, config=config, auto_adapt=False
            )
            results = pipeline.run(progress=False)

        result = m.result()
        # LiH (225 configs) should use < 512 MB
        assert result.peak_memory_bytes < 512 * 1024 * 1024

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_benchmark_n2_sparse_eigsh(self, n2_hamiltonian):
        """N2 sparse eigensolver benchmark (14400 configs)."""
        suite = BenchmarkSuite()

        with BenchmarkTimer(label="n2_fci") as t:
            E_fci = n2_hamiltonian.fci_energy()

        suite.add(t.result())
        assert E_fci < 0  # N2 ground state is negative

        # N2 pipeline
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )

        with BenchmarkTimer(label="n2_pipeline") as t2:
            pipeline = FlowGuidedKrylovPipeline(
                n2_hamiltonian, config=config, auto_adapt=False
            )
            results = pipeline.run(progress=False)

        suite.add(t2.result())

        # Print summary for manual inspection
        print(suite.summary())

        # N2 pipeline (14400 configs) should complete in < 10 minutes
        assert t2.elapsed_seconds < 600.0

    @pytest.mark.molecular
    def test_benchmark_suite_integration(self, h2_hamiltonian, lih_hamiltonian):
        """Integration test: collect benchmarks from multiple molecules."""
        suite = BenchmarkSuite()

        with BenchmarkTimer(label="h2_fci") as t1:
            E_h2 = h2_hamiltonian.fci_energy()
        suite.add(t1.result())

        with BenchmarkTimer(label="lih_fci") as t2:
            E_lih = lih_hamiltonian.fci_energy()
        suite.add(t2.result())

        # Suite should have both results
        assert len(suite.results) == 2
        labels = [r.label for r in suite.results]
        assert "h2_fci" in labels
        assert "lih_fci" in labels

        # Summary should be formatted
        summary = suite.summary()
        assert "h2_fci" in summary
        assert "lih_fci" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
