"""Benchmark: _precompute_sparse_h2e initialization time.

This is the REAL bottleneck for 40Q — O(n_orb^4) Python loops run once
at Hamiltonian creation, but at 20 orbitals that's 160K iterations.
"""
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hamiltonians.molecular import create_n2_hamiltonian


def measure_precompute_loops(n_orb, h2e_np):
    """Simulate _precompute_sparse_h2e loop timing."""
    tol = 1e-12

    # Same-spin loop: q < s, p < r, no overlap
    t0 = time.perf_counter()
    count_same = 0
    total_same = 0
    for q in range(n_orb):
        for s in range(q + 1, n_orb):
            for p in range(n_orb):
                for r in range(p + 1, n_orb):
                    if p == q or p == s or r == q or r == s:
                        continue
                    total_same += 1
                    val = h2e_np[p, q, r, s] - h2e_np[p, s, r, q]
                    if abs(val) > tol:
                        count_same += 1
    t_same = time.perf_counter() - t0

    # Alpha-beta loop: no ordering constraints
    t0 = time.perf_counter()
    count_ab = 0
    total_ab = 0
    for q in range(n_orb):
        for s in range(n_orb):
            for p in range(n_orb):
                if p == q:
                    continue
                for r in range(n_orb):
                    if r == s:
                        continue
                    total_ab += 1
                    val = h2e_np[p, q, r, s]
                    if abs(val) > tol:
                        count_ab += 1
    t_ab = time.perf_counter() - t0

    return t_same, t_ab, total_same, total_ab, count_same, count_ab


def numpy_vectorized_precompute(n_orb, h2e_np):
    """Alternative: fully vectorized precompute using NumPy broadcasting."""
    tol = 1e-12

    # Same-spin: generate all valid (q,s,p,r) with q<s, p<r, no overlap
    t0 = time.perf_counter()
    qs_pairs = np.array([(q, s) for q in range(n_orb) for s in range(q + 1, n_orb)])
    pr_pairs = np.array([(p, r) for p in range(n_orb) for r in range(p + 1, n_orb)])

    if len(qs_pairs) > 0 and len(pr_pairs) > 0:
        # Broadcast: (n_qs, 1, 2) vs (1, n_pr, 2)
        q_all = np.repeat(qs_pairs[:, 0], len(pr_pairs))
        s_all = np.repeat(qs_pairs[:, 1], len(pr_pairs))
        p_all = np.tile(pr_pairs[:, 0], len(qs_pairs))
        r_all = np.tile(pr_pairs[:, 1], len(qs_pairs))

        # Filter overlaps
        valid = (p_all != q_all) & (p_all != s_all) & (r_all != q_all) & (r_all != s_all)
        q_v, s_v, p_v, r_v = q_all[valid], s_all[valid], p_all[valid], r_all[valid]

        # Vectorized h2e lookup
        vals = h2e_np[p_v, q_v, r_v, s_v] - h2e_np[p_v, s_v, r_v, q_v]
        nonzero = np.abs(vals) > tol
        count_same = nonzero.sum()
    else:
        count_same = 0
    t_same = time.perf_counter() - t0

    # Alpha-beta: all (q,s,p,r) with p!=q, r!=s
    t0 = time.perf_counter()
    idx = np.arange(n_orb)
    q_all, s_all, p_all, r_all = np.meshgrid(idx, idx, idx, idx, indexing='ij')
    q_all, s_all, p_all, r_all = q_all.ravel(), s_all.ravel(), p_all.ravel(), r_all.ravel()

    valid = (p_all != q_all) & (r_all != s_all)
    q_v, s_v, p_v, r_v = q_all[valid], s_all[valid], p_all[valid], r_all[valid]
    vals = h2e_np[p_v, q_v, r_v, s_v]
    nonzero = np.abs(vals) > tol
    count_ab = nonzero.sum()
    t_ab = time.perf_counter() - t0

    return t_same, t_ab, count_same, count_ab


if __name__ == "__main__":
    print("PR 2.3 Precompute Loop Benchmark")
    print("=" * 60)

    n2 = create_n2_hamiltonian()
    n_orb = n2.n_orbitals
    h2e_np = n2._h2e_np
    print(f"N2: {n_orb} orbitals")

    # Python loops
    print("\n--- Python 4-nested loops ---")
    t_same, t_ab, total_same, total_ab, cnt_same, cnt_ab = measure_precompute_loops(n_orb, h2e_np)
    print(f"  Same-spin: {t_same:.3f}s, {total_same} iters, {cnt_same} nonzero")
    print(f"  Alpha-beta: {t_ab:.3f}s, {total_ab} iters, {cnt_ab} nonzero")
    print(f"  Total: {t_same + t_ab:.3f}s")

    # NumPy vectorized
    print("\n--- NumPy vectorized ---")
    t_same_v, t_ab_v, cnt_same_v, cnt_ab_v = numpy_vectorized_precompute(n_orb, h2e_np)
    print(f"  Same-spin: {t_same_v:.4f}s, {cnt_same_v} nonzero")
    print(f"  Alpha-beta: {t_ab_v:.4f}s, {cnt_ab_v} nonzero")
    print(f"  Total: {t_same_v + t_ab_v:.4f}s")
    print(f"  Speedup vs Python: {(t_same + t_ab) / (t_same_v + t_ab_v):.1f}x")

    # Verify counts match
    if cnt_same != cnt_same_v or cnt_ab != cnt_ab_v:
        print(f"  WARNING: count mismatch! same={cnt_same} vs {cnt_same_v}, ab={cnt_ab} vs {cnt_ab_v}")
    else:
        print(f"  Counts match!")

    # Extrapolation
    print("\n--- Extrapolation to 40Q (20 orbitals) ---")
    scale = (20 / 10) ** 4  # O(n^4) scaling
    print(f"  Python loops: ~{(t_same + t_ab) * scale:.1f}s ({(t_same + t_ab) * scale / 60:.1f} min)")
    print(f"  NumPy vectorized: ~{(t_same_v + t_ab_v) * scale:.2f}s")
    print(f"  NumPy memory for meshgrid: ~{20**4 * 8 * 4 / 1e6:.0f} MB (4 arrays of 160K int64)")
