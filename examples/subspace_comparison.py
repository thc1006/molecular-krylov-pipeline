"""
Subspace Method Comparison: SKQD vs SQD-Clean vs SQD-Recovery.

Runs all three subspace construction modes on the same molecular systems
and prints a side-by-side comparison table.

Approach 1 (SKQD - Krylov-based):
    NF-NQS generates initial configs, Krylov time evolution
    U^k = e^{-iHdt} expands subspace iteratively, union and diagonalize.

Approach 2 (SQD-Clean - Sampling-based, no noise):
    Particle-conserving NF-NQS samples → batch diagonalization +
    energy-variance extrapolation. No configuration recovery.

Approach 3 (SQD-Recovery - Sampling-based, with noise injection):
    Particle-conserving NF-NQS samples + depolarizing noise injection →
    S-CORE self-consistent configuration recovery → batch diagonalization.
    Emulates the full IBM "Chemistry Beyond Exact Diagonalization" workflow.

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py
    docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py --systems h2o beh2
    docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py --noise-rate 0.15
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from hamiltonians.molecular import (
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_h2o_hamiltonian,
    create_beh2_hamiltonian,
    create_nh3_hamiltonian,
    create_ch4_hamiltonian,
)
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


@dataclass
class ComparisonResult:
    """Result from comparing three subspace modes on one system."""
    system: str
    n_qubits: int
    n_configs: int
    fci_energy: float
    skqd_energy: float
    sqd_clean_energy: float
    sqd_recovery_energy: float
    skqd_error_mha: float
    sqd_clean_error_mha: float
    sqd_recovery_error_mha: float
    skqd_time: float
    sqd_clean_time: float
    sqd_recovery_time: float


SYSTEMS = {
    "h2": ("H2 (STO-3G)", create_h2_hamiltonian, 0.74),
    "lih": ("LiH (STO-3G)", create_lih_hamiltonian, 1.6),
    "h2o": ("H2O (STO-3G)", create_h2o_hamiltonian, None),
    "beh2": ("BeH2 (STO-3G)", create_beh2_hamiltonian, None),
    "nh3": ("NH3 (STO-3G)", create_nh3_hamiltonian, None),
    "ch4": ("CH4 (STO-3G)", create_ch4_hamiltonian, None),
}


def _get_energy(results):
    """Extract final energy from pipeline results."""
    return results.get(
        'combined_energy',
        results.get('skqd_energy',
        results.get('sqd_energy', float('inf')))
    )


def run_comparison(
    system_key: str,
    noise_rate: float = 0.1,
    verbose: bool = True,
) -> ComparisonResult:
    """Run SKQD, SQD-Clean, and SQD-Recovery on a single system."""
    name, create_fn, bond_length = SYSTEMS[system_key]

    print(f"\n{'='*70}")
    print(f"COMPARING: {name}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Hamiltonian
    if bond_length is not None:
        H = create_fn(bond_length=bond_length, device=device)
    else:
        H = create_fn(device=device)

    n_qubits = H.num_sites
    from math import comb
    n_configs = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    # FCI reference
    E_fci = H.fci_energy()
    print(f"  Qubits: {n_qubits}, Configs: {n_configs}, FCI: {E_fci:.8f} Ha")

    # --- SKQD Mode ---
    print(f"\n  [1/3] Running SKQD (Krylov time evolution)...")
    t0 = time.time()

    config_skqd = PipelineConfig(
        subspace_mode="skqd",
        skip_nf_training=True,
        device=device,
    )
    config_skqd.adapt_to_system_size(n_configs)

    pipeline = FlowGuidedKrylovPipeline(H, config=config_skqd, exact_energy=E_fci)
    results_skqd = pipeline.run(progress=verbose)
    skqd_energy = _get_energy(results_skqd)
    skqd_time = time.time() - t0
    skqd_error = abs(skqd_energy - E_fci) * 1000
    print(f"  SKQD: {skqd_energy:.8f} Ha (error: {skqd_error:.4f} mHa, time: {skqd_time:.1f}s)")

    # --- SQD-Clean Mode ---
    print(f"\n  [2/3] Running SQD-Clean (batch diag, no noise)...")
    t0 = time.time()

    config_sqd_clean = PipelineConfig(
        subspace_mode="sqd",
        skip_nf_training=True,
        sqd_num_batches=5,
        sqd_self_consistent_iters=3,
        sqd_noise_rate=0.0,
        device=device,
    )
    config_sqd_clean.adapt_to_system_size(n_configs)

    pipeline = FlowGuidedKrylovPipeline(H, config=config_sqd_clean, exact_energy=E_fci)
    results_sqd_clean = pipeline.run(progress=verbose)
    sqd_clean_energy = _get_energy(results_sqd_clean)
    sqd_clean_time = time.time() - t0
    sqd_clean_error = abs(sqd_clean_energy - E_fci) * 1000
    print(f"  SQD-Clean: {sqd_clean_energy:.8f} Ha (error: {sqd_clean_error:.4f} mHa, time: {sqd_clean_time:.1f}s)")

    # --- SQD-Recovery Mode ---
    print(f"\n  [3/3] Running SQD-Recovery (noise={noise_rate}, S-CORE recovery)...")
    t0 = time.time()

    config_sqd_recovery = PipelineConfig(
        subspace_mode="sqd",
        skip_nf_training=True,
        sqd_num_batches=5,
        sqd_self_consistent_iters=5,
        sqd_noise_rate=noise_rate,
        device=device,
    )
    config_sqd_recovery.adapt_to_system_size(n_configs)

    pipeline = FlowGuidedKrylovPipeline(H, config=config_sqd_recovery, exact_energy=E_fci)
    results_sqd_recovery = pipeline.run(progress=verbose)
    sqd_recovery_energy = _get_energy(results_sqd_recovery)
    sqd_recovery_time = time.time() - t0
    sqd_recovery_error = abs(sqd_recovery_energy - E_fci) * 1000
    print(f"  SQD-Recovery: {sqd_recovery_energy:.8f} Ha (error: {sqd_recovery_error:.4f} mHa, time: {sqd_recovery_time:.1f}s)")

    return ComparisonResult(
        system=name,
        n_qubits=n_qubits,
        n_configs=n_configs,
        fci_energy=E_fci,
        skqd_energy=skqd_energy,
        sqd_clean_energy=sqd_clean_energy,
        sqd_recovery_energy=sqd_recovery_energy,
        skqd_error_mha=skqd_error,
        sqd_clean_error_mha=sqd_clean_error,
        sqd_recovery_error_mha=sqd_recovery_error,
        skqd_time=skqd_time,
        sqd_clean_time=sqd_clean_time,
        sqd_recovery_time=sqd_recovery_time,
    )


def main():
    parser = argparse.ArgumentParser(description="SKQD vs SQD-Clean vs SQD-Recovery Comparison")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["h2", "lih", "h2o", "beh2"],
        choices=list(SYSTEMS.keys()),
        help="Systems to compare (default: h2 lih h2o beh2)",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.1,
        help="Depolarizing noise rate for SQD-Recovery mode (default: 0.1)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    args = parser.parse_args()

    print("=" * 130)
    print("SUBSPACE METHOD COMPARISON: SKQD vs SQD-Clean vs SQD-Recovery")
    print("=" * 130)
    print("SKQD:         Krylov time evolution (U^k = e^{-iHdt})")
    print("SQD-Clean:    Batch diagonalization + energy-variance extrapolation (no noise)")
    print(f"SQD-Recovery: Noise injection (rate={args.noise_rate}) + S-CORE recovery + batch diag")
    print("=" * 130)

    results = []
    for key in args.systems:
        try:
            r = run_comparison(key, noise_rate=args.noise_rate, verbose=not args.quiet)
            results.append(r)
        except Exception as e:
            print(f"\nERROR on {key}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 140)
    print("COMPARISON TABLE")
    print("=" * 140)
    print(f"{'System':<18} {'Qubits':>6} {'Configs':>8} {'FCI Energy':>14} "
          f"{'SKQD err':>10} {'Clean err':>10} {'Recov err':>10} "
          f"{'SKQD t':>8} {'Clean t':>8} {'Recov t':>8} {'Best':>10}")
    print("-" * 140)

    for r in results:
        errors = {
            "SKQD": r.skqd_error_mha,
            "Clean": r.sqd_clean_error_mha,
            "Recovery": r.sqd_recovery_error_mha,
        }
        best = min(errors, key=errors.get)
        print(f"{r.system:<18} {r.n_qubits:>6} {r.n_configs:>8} {r.fci_energy:>14.8f} "
              f"{r.skqd_error_mha:>10.4f} {r.sqd_clean_error_mha:>10.4f} {r.sqd_recovery_error_mha:>10.4f} "
              f"{r.skqd_time:>7.1f}s {r.sqd_clean_time:>7.1f}s {r.sqd_recovery_time:>7.1f}s {best:>10}")

    print("-" * 140)

    # Chemical accuracy check
    chem_acc_threshold = 1.6  # mHa
    total = len(results)
    skqd_pass = sum(1 for r in results if r.skqd_error_mha < chem_acc_threshold)
    clean_pass = sum(1 for r in results if r.sqd_clean_error_mha < chem_acc_threshold)
    recov_pass = sum(1 for r in results if r.sqd_recovery_error_mha < chem_acc_threshold)

    print(f"\nChemical accuracy (<{chem_acc_threshold} mHa):")
    print(f"  SKQD:         {skqd_pass}/{total} systems")
    print(f"  SQD-Clean:    {clean_pass}/{total} systems")
    print(f"  SQD-Recovery: {recov_pass}/{total} systems")

    # Average errors
    if results:
        avg_skqd = np.mean([r.skqd_error_mha for r in results])
        avg_clean = np.mean([r.sqd_clean_error_mha for r in results])
        avg_recov = np.mean([r.sqd_recovery_error_mha for r in results])
        print(f"\nAverage error:")
        print(f"  SKQD:         {avg_skqd:.4f} mHa")
        print(f"  SQD-Clean:    {avg_clean:.4f} mHa")
        print(f"  SQD-Recovery: {avg_recov:.4f} mHa")

        avg_skqd_t = np.mean([r.skqd_time for r in results])
        avg_clean_t = np.mean([r.sqd_clean_time for r in results])
        avg_recov_t = np.mean([r.sqd_recovery_time for r in results])
        print(f"\nAverage time:")
        print(f"  SKQD:         {avg_skqd_t:.1f}s")
        print(f"  SQD-Clean:    {avg_clean_t:.1f}s")
        print(f"  SQD-Recovery: {avg_recov_t:.1f}s")

    print("=" * 140)


if __name__ == "__main__":
    main()
