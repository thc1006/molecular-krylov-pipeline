"""Benchmark utilities for wall-time and memory tracking.

Provides BenchmarkTimer, MemoryTracker, and BenchmarkSuite for measuring
pipeline performance and detecting regressions as we scale to 40Q.

Usage:
    from utils.benchmark import BenchmarkTimer, MemoryTracker, BenchmarkSuite

    suite = BenchmarkSuite()

    with BenchmarkTimer(label="eigensolver") as t:
        E = hamiltonian.fci_energy()
    suite.add(t.result())

    with MemoryTracker(label="pipeline") as m:
        pipeline.run()
    suite.add(m.result())

    print(suite.summary())
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement.

    Attributes:
        label: Human-readable name of the benchmarked operation.
        elapsed_seconds: Wall-clock time in seconds.
        peak_memory_bytes: Peak memory usage in bytes (0 if not measured).
        metadata: Arbitrary extra data (e.g., n_configs, n_qubits).
    """

    label: str
    elapsed_seconds: float
    peak_memory_bytes: int = 0
    metadata: dict = field(default_factory=dict)


class BenchmarkTimer:
    """Context manager for wall-time measurement using perf_counter.

    Example:
        with BenchmarkTimer(label="my_op") as t:
            do_work()
        print(f"Took {t.elapsed_seconds:.3f}s")
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_seconds = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_seconds = time.perf_counter() - self._start

    def result(self) -> BenchmarkResult:
        """Return a BenchmarkResult with the measured time."""
        return BenchmarkResult(label=self.label, elapsed_seconds=self.elapsed_seconds)


class MemoryTracker:
    """Context manager for peak memory measurement using tracemalloc.

    Measures the peak memory allocated by Python during the tracked block.
    Note: this tracks Python allocations only, not GPU memory or mmap'd files.

    Example:
        with MemoryTracker(label="pipeline") as m:
            run_pipeline()
        print(f"Peak: {m.peak_memory_bytes / 1e6:.1f} MB")
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.peak_memory_bytes = 0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *args):
        _, self.peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def result(self) -> BenchmarkResult:
        """Return a BenchmarkResult with the measured peak memory."""
        return BenchmarkResult(
            label=self.label,
            elapsed_seconds=0.0,
            peak_memory_bytes=self.peak_memory_bytes,
        )


class BenchmarkSuite:
    """Collect and compare benchmark results.

    Example:
        suite = BenchmarkSuite()
        suite.add(timer.result())
        suite.add(tracker.result())
        print(suite.summary())

        # Compare against a baseline
        regressions = suite.check_regression(baseline_suite, threshold=2.0)
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def summary(self) -> str:
        """Format a human-readable summary of all results.

        Returns:
            Multi-line string with label, time, and memory for each result.
        """
        lines = ["Benchmark Results:", "=" * 60]
        for r in self.results:
            mem_mb = r.peak_memory_bytes / (1024 * 1024)
            lines.append(f"  {r.label}: {r.elapsed_seconds:.3f}s, {mem_mb:.1f} MB peak")
        return "\n".join(lines)

    def check_regression(
        self, baseline: "BenchmarkSuite", threshold: float = 2.0
    ) -> List[str]:
        """Flag results that are >threshold times slower than baseline.

        Args:
            baseline: Previous benchmark suite to compare against.
            threshold: Slowdown factor that triggers a regression flag.

        Returns:
            List of human-readable regression descriptions. Empty if no regressions.
        """
        regressions = []
        baseline_map = {r.label: r for r in baseline.results}
        for r in self.results:
            if r.label in baseline_map:
                base = baseline_map[r.label]
                if (
                    base.elapsed_seconds > 0
                    and r.elapsed_seconds > threshold * base.elapsed_seconds
                ):
                    regressions.append(
                        f"{r.label}: {r.elapsed_seconds:.3f}s vs baseline "
                        f"{base.elapsed_seconds:.3f}s "
                        f"({r.elapsed_seconds / base.elapsed_seconds:.1f}x slower)"
                    )
        return regressions
