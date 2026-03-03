"""
Small System Validation for Flow-Guided Krylov Pipeline.

Tests the pipeline end-to-end on small molecular systems (< 18 qubits)
where FCI is tractable, to verify correctness before scaling up.

Systems tested:
- H2:   4 qubits,  2 electrons,    4 configs
- LiH: 12 qubits,  4 electrons,  100 configs
- H2O: 14 qubits, 10 electrons,  441 configs
- BeH2: 14 qubits,  6 electrons, 1225 configs

Success criteria: All systems achieve chemical accuracy (< 1.6 mHa error).

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/validate_small_systems.py
    docker-compose run --rm flow-krylov-gpu python examples/validate_small_systems.py --system h2
"""

import sys
import time
import argparse
from pathlib import Path

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


CHEMICAL_ACCURACY_MHA = 1.6  # 1 kcal/mol ≈ 1.6 mHa


def validate_system(name, create_fn, bond_length=None, verbose=True):
    """Run full pipeline on a small system and check chemical accuracy."""
    print("\n" + "=" * 70)
    print(f"VALIDATING: {name}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create Hamiltonian
    if bond_length is not None:
        H = create_fn(bond_length=bond_length, device=device)
    else:
        H = create_fn(device=device)

    n_qubits = H.num_sites
    n_electrons = H.n_electrons
    from math import comb
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons} ({H.n_alpha}α + {H.n_beta}β)")
    print(f"  Valid configs: {n_valid}")

    # Compute FCI (exact) energy
    print("  Computing FCI energy...")
    E_fci = H.fci_energy()
    print(f"  FCI Energy: {E_fci:.8f} Ha")

    # Configure pipeline for small systems
    # Direct-CI mode: skip NF training, use essential configs + subspace diag
    config = PipelineConfig(
        use_particle_conserving_flow=True,
        use_diversity_selection=True,
        skip_nf_training=True,
        max_epochs=200,
        min_epochs=30,
        convergence_threshold=0.15,
        device=device,
    )

    # Run the full pipeline
    t0 = time.time()
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_fci)
    results = pipeline.run(progress=verbose)
    elapsed = time.time() - t0

    # Extract final energy
    final_energy = results.get("combined_energy",
                     results.get("skqd_energy",
                     results.get("sqd_energy",
                     results.get("nf_nqs_energy", float('inf')))))

    error_mha = abs(final_energy - E_fci) * 1000
    error_kcal = error_mha / 1.6
    passed = error_mha < CHEMICAL_ACCURACY_MHA

    print(f"\n{'='*70}")
    print(f"RESULT: {name}")
    print(f"{'='*70}")
    print(f"  FCI Energy:   {E_fci:.8f} Ha")
    print(f"  Final Energy: {final_energy:.8f} Ha")
    print(f"  Error:        {error_mha:.4f} mHa ({error_kcal:.2f} kcal/mol)")
    print(f"  Time:         {elapsed:.1f}s")
    print(f"  Status:       {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")

    return {
        "name": name,
        "n_qubits": n_qubits,
        "n_valid": n_valid,
        "fci_energy": E_fci,
        "final_energy": final_energy,
        "error_mha": error_mha,
        "elapsed_s": elapsed,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Small System Validation")
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["h2", "lih", "h2o", "beh2", "nh3", "ch4", "all", "medium"],
        help="System to validate (default: all, medium=nh3+ch4)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    systems = {
        "h2": ("H2 (STO-3G)", create_h2_hamiltonian, 0.74),
        "lih": ("LiH (STO-3G)", create_lih_hamiltonian, 1.6),
        "h2o": ("H2O (STO-3G)", create_h2o_hamiltonian, None),
        "beh2": ("BeH2 (STO-3G)", create_beh2_hamiltonian, None),
        "nh3": ("NH3 (STO-3G)", create_nh3_hamiltonian, None),
        "ch4": ("CH4 (STO-3G)", create_ch4_hamiltonian, None),
    }

    if args.system == "all":
        run_systems = ["h2", "lih", "h2o", "beh2", "nh3", "ch4"]
    elif args.system == "medium":
        run_systems = ["nh3", "ch4"]
    else:
        run_systems = [args.system]

    results = []
    for key in run_systems:
        name, create_fn, bond_length = systems[key]
        try:
            r = validate_system(name, create_fn, bond_length, verbose=not args.quiet)
            results.append(r)
        except Exception as e:
            print(f"\nERROR running {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": name, "passed": False, "error_mha": float('inf'),
                "n_qubits": 0, "elapsed_s": 0,
            })

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'System':<20} {'Qubits':>8} {'Configs':>10} {'Error (mHa)':>12} {'Time':>8} {'Status':>8}")
    print("-" * 80)

    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        qubits = r.get("n_qubits", "?")
        configs = r.get("n_valid", "?")
        error = r.get("error_mha", float('inf'))
        elapsed = r.get("elapsed_s", 0)
        print(f"{r['name']:<20} {qubits:>8} {configs:>10} {error:>12.4f} {elapsed:>7.1f}s {status:>8}")

    print("-" * 80)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"Chemical accuracy threshold: {CHEMICAL_ACCURACY_MHA} mHa (1 kcal/mol)")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
