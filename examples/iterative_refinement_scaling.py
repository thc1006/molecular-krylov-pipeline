"""
Iterative Refinement Scaling Benchmark
========================================
Ablation: Direct-CI vs iterative refinement across scales.

Systems:
  - CAS(10,12) 24Q  (N2/cc-pVDZ, FCI available)
  - CAS(10,15) 30Q  (N2/cc-pVDZ, FCI via CASCI)
  - CAS(10,20) 40Q  (N2/cc-pVDZ, no FCI — compare to HF)

Usage:
  uv run python examples/iterative_refinement_scaling.py
  uv run python examples/iterative_refinement_scaling.py --system 40q_extended
"""

import sys
import os
import time
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hamiltonians.molecular import create_n2_cas_hamiltonian
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Shared refinement config ──────────────────────────────────────────────────
REFINEMENT_CONFIG = dict(
    subspace_mode="skqd",
    skip_nf_training=True,     # Iteration 1 = Direct-CI
    device=DEVICE,
    n_refinement_iterations=3,
    h_coupling_weight=0.2,
    h_coupling_n_ref=100,
    refinement_epochs=50,
    max_epochs=50,
    min_epochs=10,
    samples_per_batch=500,
    entropy_weight=0.05,
    physics_weight=0.1,
    max_diag_basis_size=5000,
)

DIRECT_CI_CONFIG = dict(
    subspace_mode="skqd",
    skip_nf_training=True,
    device=DEVICE,
    n_refinement_iterations=1,
    max_diag_basis_size=5000,
)


def run_experiment(name, H, fci_energy, config_dict, seed=42):
    torch.manual_seed(seed)
    cfg = PipelineConfig(**config_dict)
    pipeline = FlowGuidedKrylovPipeline(H, config=cfg, exact_energy=fci_energy)

    t0 = time.time()
    result = pipeline.run()
    wall = time.time() - t0

    energy = result["combined_energy"]
    refinement = result.get("refinement_energies", [energy])

    print(f"\n{'=' * 60}")
    print(f"  {name}  [{wall:.0f}s]  device={DEVICE}")
    print(f"{'=' * 60}")
    for i, e in enumerate(refinement):
        if fci_energy is not None:
            err = abs(e - fci_energy) * 1000
            mark = " ← chemical accuracy" if err < 1.6 else ""
            print(f"  Iter {i+1}: {e:.8f} Ha  ({err:.3f} mHa){mark}")
        else:
            hf_e = H.diagonal_element(H.get_hf_state()).item()
            print(f"  Iter {i+1}: {e:.8f} Ha  ({(hf_e - e)*1000:.2f} mHa below HF)")

    if fci_energy is not None and len(refinement) > 1:
        e1 = abs(refinement[0] - fci_energy) * 1000
        ef = abs(refinement[-1] - fci_energy) * 1000
        print(f"\n  Improvement: {e1:.3f} → {ef:.3f} mHa  ({e1/ef:.1f}x)")

    return result, wall


def run_24q(n_iters=3):
    print("\n" + "=" * 60)
    print("  CAS(10,12) 24Q — N2/cc-pVDZ")
    print("=" * 60)
    H = create_n2_cas_hamiltonian(basis="cc-pvdz", cas=(10, 12), device=DEVICE)
    fci = H.fci_energy()
    print(f"  FCI = {fci:.8f} Ha")

    cfg = {**REFINEMENT_CONFIG, "n_refinement_iterations": n_iters,
           "refinement_epochs": 30, "max_epochs": 30, "samples_per_batch": 300}
    run_experiment("24Q Direct-CI baseline", H, fci, DIRECT_CI_CONFIG)
    run_experiment(f"24Q Iterative Refinement ({n_iters} iter)", H, fci, cfg)


def run_30q(n_iters=3):
    print("\n" + "=" * 60)
    print("  CAS(10,15) 30Q — N2/cc-pVDZ")
    print("=" * 60)
    H = create_n2_cas_hamiltonian(basis="cc-pvdz", cas=(10, 15), device=DEVICE)
    try:
        fci = H.fci_energy()
        if fci is not None:
            print(f"  FCI = {fci:.8f} Ha")
        else:
            print("  FCI: not computable (too large)")
    except Exception:
        fci = None
        print("  FCI: not computable (too large)")
        # Critical on UMA: release CUDA cached memory after failed FCI attempt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cfg = {**REFINEMENT_CONFIG, "n_refinement_iterations": n_iters,
           "refinement_epochs": 40, "max_epochs": 40, "samples_per_batch": 400}
    run_experiment("30Q Direct-CI baseline", H, fci, DIRECT_CI_CONFIG)
    run_experiment(f"30Q Iterative Refinement ({n_iters} iter)", H, fci, cfg)


def run_40q_extended(n_iters=3, epochs=100):
    print("\n" + "=" * 60)
    print("  CAS(10,20) 40Q — N2/cc-pVDZ (extended)")
    print("=" * 60)
    H = create_n2_cas_hamiltonian(basis="cc-pvdz", cas=(10, 20), device=DEVICE)
    hf = H.diagonal_element(H.get_hf_state()).item()
    print(f"  HF = {hf:.8f} Ha  (FCI not computable — compare to HF)")

    cfg = {**REFINEMENT_CONFIG, "n_refinement_iterations": n_iters,
           "refinement_epochs": epochs, "max_epochs": epochs,
           "samples_per_batch": 500}
    run_experiment("40Q Direct-CI baseline", H, None, DIRECT_CI_CONFIG)
    run_experiment(f"40Q Iterative Refinement ({n_iters} iter, {epochs} ep)", H, None, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["24q", "30q", "40q_extended", "all"],
                        default="all", help="Which system to run")
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--epochs_40q", type=int, default=100)
    args = parser.parse_args()

    print(f"\nIterative Refinement Scaling Benchmark")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    t_total = time.time()

    if args.system in ("24q", "all"):
        run_24q(n_iters=args.n_iters)
    if args.system in ("30q", "all"):
        run_30q(n_iters=args.n_iters)
    if args.system in ("40q_extended", "all"):
        run_40q_extended(n_iters=args.n_iters, epochs=args.epochs_40q)

    print(f"\nTotal wall time: {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
