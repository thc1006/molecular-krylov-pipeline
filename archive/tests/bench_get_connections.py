"""Benchmark: get_connections() vs get_connections_vectorized_batch()

PR 2.3 prerequisite: measure current performance before deciding on Numba.
Tests on LiH (6 orb), BeH2 (7 orb), N2 (10 orb) to understand scaling.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hamiltonians.molecular import (
    create_lih_hamiltonian,
    create_beh2_hamiltonian,
    create_n2_hamiltonian,
)


def benchmark_sequential(hamiltonian, configs, label):
    """Benchmark sequential get_connections() one config at a time."""
    n = len(configs)
    total_connections = 0

    t0 = time.perf_counter()
    for i in range(n):
        connected, elements = hamiltonian.get_connections(configs[i])
        total_connections += len(connected)
    elapsed = time.perf_counter() - t0

    per_config_us = elapsed / n * 1e6
    print(f"  [{label}] Sequential: {n} configs, {elapsed:.3f}s, "
          f"{per_config_us:.0f} us/config, {total_connections} total connections")
    return elapsed, total_connections


def benchmark_vectorized_batch(hamiltonian, configs, label):
    """Benchmark get_connections_vectorized_batch() (GPU tensor ops)."""
    n = len(configs)

    # Warm up
    if n > 10:
        _ = hamiltonian.get_connections_vectorized_batch(configs[:10])

    t0 = time.perf_counter()
    connected, elements, indices = hamiltonian.get_connections_vectorized_batch(configs)
    elapsed = time.perf_counter() - t0

    total_connections = len(connected)
    per_config_us = elapsed / n * 1e6
    print(f"  [{label}] Vectorized: {n} configs, {elapsed:.3f}s, "
          f"{per_config_us:.0f} us/config, {total_connections} total connections")
    return elapsed, total_connections


def benchmark_molecule(name, hamiltonian, n_configs_list):
    """Run benchmarks for one molecule at various batch sizes."""
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    print(f"\n{'='*70}")
    print(f"{name}: {n_orb} orbitals, {n_alpha}a+{n_beta}b, "
          f"num_sites={hamiltonian.num_sites}")
    print(f"{'='*70}")

    # Generate essential configs (HF + singles + doubles)
    from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
    pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
    pipeline = FlowGuidedKrylovPipeline(hamiltonian, config=pipe_config)
    all_configs = pipeline._generate_essential_configs()
    print(f"  Essential configs: {len(all_configs)}")

    for n_configs in n_configs_list:
        n = min(n_configs, len(all_configs))
        configs = all_configs[:n]
        label = f"n={n}"

        t_seq, conn_seq = benchmark_sequential(hamiltonian, configs, label)
        t_vec, conn_vec = benchmark_vectorized_batch(hamiltonian, configs, label)

        # Verify correctness
        if conn_seq != conn_vec:
            print(f"  WARNING: connection count mismatch! seq={conn_seq}, vec={conn_vec}")

        speedup = t_seq / t_vec if t_vec > 0 else float('inf')
        print(f"  Speedup: {speedup:.1f}x")
        print()


def benchmark_jw_sign(hamiltonian, n_calls=10000):
    """Benchmark JW sign computation specifically."""
    n_orb = hamiltonian.n_orbitals
    config_np = np.zeros(hamiltonian.num_sites, dtype=np.int64)
    config_np[:hamiltonian.n_alpha] = 1
    config_np[n_orb:n_orb + hamiltonian.n_beta] = 1

    # Single JW sign
    t0 = time.perf_counter()
    for _ in range(n_calls):
        hamiltonian._jw_sign_np(config_np, n_orb - 1, 0)
    t_single = time.perf_counter() - t0

    # Double JW sign
    t0 = time.perf_counter()
    for _ in range(n_calls):
        hamiltonian._jw_sign_double_np(
            config_np, 0, n_orb + n_orb - 1, n_orb - 1, n_orb
        )
    t_double = time.perf_counter() - t0

    print(f"\n  JW sign ({n_calls} calls): single={t_single*1e3:.1f}ms, "
          f"double={t_double*1e3:.1f}ms")
    print(f"  Per call: single={t_single/n_calls*1e6:.1f}us, "
          f"double={t_double/n_calls*1e6:.1f}us")


if __name__ == "__main__":
    print("PR 2.3 Prerequisite Benchmark: get_connections performance")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")

    # LiH: 6 orbitals, small system
    lih = create_lih_hamiltonian()
    benchmark_molecule("LiH", lih, [50, 225])
    benchmark_jw_sign(lih)

    # BeH2: 7 orbitals, medium system
    beh2 = create_beh2_hamiltonian()
    benchmark_molecule("BeH2", beh2, [50, 200])
    benchmark_jw_sign(beh2)

    # N2: 10 orbitals, large system (40Q bridge target)
    n2 = create_n2_hamiltonian()
    benchmark_molecule("N2", n2, [50, 200, 500])
    benchmark_jw_sign(n2)

    print("\n" + "="*70)
    print("ANALYSIS:")
    print("If vectorized_batch provides >20x speedup on N2,")
    print("Numba may not be needed for get_connections().")
    print("Focus Numba effort on JW sign and _precompute_sparse_h2e instead.")
    print("="*70)
