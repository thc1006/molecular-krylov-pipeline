"""
Moderate System Benchmark: SKQD vs SQD-Clean vs SQD-Recovery Comparison.

Tests molecular systems in the 20-30 qubit range to compare three
subspace construction strategies:

- SKQD (Krylov-based): NF-NQS generates initial configs, Krylov time
  evolution U^k = e^{-iHdt} expands subspace iteratively
- SQD-Clean (Sampling-based, no noise): Particle-conserving NF-NQS
  samples -> batch diagonalization + energy-variance extrapolation
- SQD-Recovery (Sampling-based, with noise): Particle-conserving NF-NQS
  samples + depolarizing noise -> S-CORE configuration recovery + batch diag

Systems tested (ordered by qubit count):
- CO (carbon monoxide): 20 qubits, 14 electrons
- HCN (hydrogen cyanide): 22 qubits, 14 electrons
- C2H2 (acetylene): 24 qubits, 14 electrons
- H2O 6-31G: 26 qubits, 10 electrons
- H2S (hydrogen sulfide): 26 qubits, 18 electrons
- C2H4 (ethylene): 28 qubits, 16 electrons
- NH3 6-31G: 30 qubits, 10 electrons

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/moderate_system_benchmark.py --system all
    docker-compose run --rm flow-krylov-gpu python examples/moderate_system_benchmark.py --system c2h4
    docker-compose run --rm flow-krylov-gpu python examples/moderate_system_benchmark.py --system co --mode sqd-recovery
    docker-compose run --rm flow-krylov-gpu python examples/moderate_system_benchmark.py --noise-rate 0.15
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb
from typing import Dict, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from pyscf import gto, scf, ao2mo, cc, fci
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required")
    sys.exit(1)

from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


@dataclass
class MoleculeData:
    """Container for molecule data including Hamiltonian and reference energies."""
    hamiltonian: MolecularHamiltonian
    hf_energy: float
    ccsd_energy: Optional[float] = None
    ccsd_t_energy: Optional[float] = None
    fci_energy: Optional[float] = None
    geometry: list = None
    basis: str = "sto-3g"


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    system: str
    n_qubits: int
    n_electrons: int
    n_valid_configs: int
    exact_energy: float
    energy_type: str = "FCI"  # FCI, CCSD(T), CCSD, or HF

    # Configuration counts
    nf_configs: int = 0

    # Energies from each mode
    nf_energy: float = 0.0
    skqd_energy: float = 0.0
    sqd_clean_energy: float = 0.0
    sqd_recovery_energy: float = 0.0

    # Timing
    time_nf: float = 0.0
    time_skqd: float = 0.0
    time_sqd_clean: float = 0.0
    time_sqd_recovery: float = 0.0

    # Errors (mHa)
    skqd_error_mha: float = 0.0
    sqd_clean_error_mha: float = 0.0
    sqd_recovery_error_mha: float = 0.0


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def format_energy_comparison(energy: float, reference: float, ref_type: str) -> str:
    """Format energy comparison, correctly handling sub-reference energies."""
    diff_mha = (energy - reference) * 1000

    if ref_type == "FCI":
        error_mha = abs(diff_mha)
        return f"error vs FCI: {error_mha:.2f} mHa"
    else:
        if diff_mha < -0.01:
            return f"{abs(diff_mha):.2f} mHa below {ref_type} (expected: FCI < {ref_type})"
        else:
            return f"{abs(diff_mha):.2f} mHa above {ref_type}"


def compute_pyscf_fci(geometry, basis, charge=0, spin=0, max_memory=8000):
    """Compute FCI energy using PySCF's iterative Davidson solver."""
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()

    cisolver = fci.FCI(mf)
    cisolver.max_memory = max_memory
    cisolver.max_cycle = 300
    cisolver.conv_tol = 1e-10

    e_fci, ci = cisolver.kernel()
    return float(e_fci)


# =============================================================================
# Helper: Create Hamiltonian with Reference Energies
# =============================================================================

def create_molecule_data(
    geometry: list,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    compute_ccsd: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create MoleculeData with Hamiltonian and reference energies."""
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()
    hf_energy = float(mf.e_tot)

    ccsd_energy = None
    ccsd_t_energy = None
    if compute_ccsd:
        try:
            mycc = cc.CCSD(mf)
            mycc.kernel()
            ccsd_energy = float(mycc.e_tot)
            try:
                et = mycc.ccsd_t()
                ccsd_t_energy = ccsd_energy + et
            except Exception as e:
                print(f"  CCSD(T) failed: {e}")
        except Exception as e:
            print(f"  CCSD failed: {e}")

    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)

    n_electrons = mol.nelectron
    n_orbitals = mol.nao
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2

    integrals = MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=mol.energy_nuc(),
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    hamiltonian = MolecularHamiltonian(integrals, device=device)

    return MoleculeData(
        hamiltonian=hamiltonian,
        hf_energy=hf_energy,
        ccsd_energy=ccsd_energy,
        ccsd_t_energy=ccsd_t_energy,
        geometry=geometry,
        basis=basis,
    )


# =============================================================================
# Hamiltonian Factories for Moderate Systems
# =============================================================================

def create_co_molecule(
    bond_length: float = 1.128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create CO (carbon monoxide). 14e, 10 orb in STO-3G = 20 qubits."""
    geometry = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, bond_length)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_hcn_molecule(
    ch_length: float = 1.066,
    cn_length: float = 1.156,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create HCN (hydrogen cyanide). 14e, 11 orb in STO-3G = 22 qubits."""
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, ch_length)),
        ("N", (0.0, 0.0, ch_length + cn_length)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_c2h2_molecule(
    cc_length: float = 1.203,
    ch_length: float = 1.063,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create C2H2 (acetylene). 14e, 12 orb in STO-3G = 24 qubits."""
    geometry = [
        ("H", (0.0, 0.0, -ch_length - cc_length/2)),
        ("C", (0.0, 0.0, -cc_length/2)),
        ("C", (0.0, 0.0, cc_length/2)),
        ("H", (0.0, 0.0, ch_length + cc_length/2)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_h2o_631g_molecule(
    oh_length: float = 0.96,
    angle: float = 104.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create H2O with 6-31G basis. 10e, 13 orb = 26 qubits."""
    angle_rad = np.radians(angle)
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_length, 0.0, 0.0)),
        ("H", (oh_length * np.cos(angle_rad), oh_length * np.sin(angle_rad), 0.0)),
    ]
    return create_molecule_data(geometry, basis="6-31g", device=device)


def create_h2s_molecule(
    sh_length: float = 1.336,
    angle: float = 92.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create H2S (hydrogen sulfide). 18e, 13 orb in STO-3G = 26 qubits."""
    angle_rad = np.radians(angle)
    geometry = [
        ("S", (0.0, 0.0, 0.0)),
        ("H", (sh_length, 0.0, 0.0)),
        ("H", (sh_length * np.cos(angle_rad), sh_length * np.sin(angle_rad), 0.0)),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_c2h4_molecule(
    cc_length: float = 1.339,
    ch_length: float = 1.087,
    hcc_angle: float = 121.3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create C2H4 (ethylene). 16e, 14 orb in STO-3G = 28 qubits."""
    angle_rad = np.radians(hcc_angle)
    geometry = [
        ("C", (0.0, 0.0, -cc_length/2)),
        ("C", (0.0, 0.0, cc_length/2)),
        ("H", (ch_length * np.sin(angle_rad), 0.0, -cc_length/2 - ch_length * np.cos(angle_rad))),
        ("H", (-ch_length * np.sin(angle_rad), 0.0, -cc_length/2 - ch_length * np.cos(angle_rad))),
        ("H", (ch_length * np.sin(angle_rad), 0.0, cc_length/2 + ch_length * np.cos(angle_rad))),
        ("H", (-ch_length * np.sin(angle_rad), 0.0, cc_length/2 + ch_length * np.cos(angle_rad))),
    ]
    return create_molecule_data(geometry, basis="sto-3g", device=device)


def create_nh3_631g_molecule(
    nh_length: float = 1.012,
    hnh_angle: float = 106.7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MoleculeData:
    """Create NH3 with 6-31G basis. 10e, 15 orb = 30 qubits."""
    angle_rad = np.radians(hnh_angle)
    h = nh_length * np.cos(np.arcsin(np.sin(angle_rad/2) / np.sin(np.radians(60))))
    r = np.sqrt(nh_length**2 - h**2)
    geometry = [
        ("N", (0.0, 0.0, h)),
        ("H", (r, 0.0, 0.0)),
        ("H", (r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0.0)),
        ("H", (r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0.0)),
    ]
    return create_molecule_data(geometry, basis="6-31g", device=device)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    molecule_key: str,
    mode: str = "all",
    noise_rate: float = 0.1,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run subspace method benchmark for a single molecule.

    Args:
        molecule_key: Key for molecule factory
        mode: "skqd", "sqd-clean", "sqd-recovery", or "all"
        noise_rate: Depolarizing noise rate for SQD-Recovery mode
        verbose: Print detailed progress
    """
    factories = {
        'co': (create_co_molecule, "CO (STO-3G)"),
        'hcn': (create_hcn_molecule, "HCN (STO-3G)"),
        'c2h2': (create_c2h2_molecule, "C2H2 (STO-3G)"),
        'h2o_631g': (create_h2o_631g_molecule, "H2O (6-31G)"),
        'h2s': (create_h2s_molecule, "H2S (STO-3G)"),
        'c2h4': (create_c2h4_molecule, "C2H4 (STO-3G)"),
        'nh3_631g': (create_nh3_631g_molecule, "NH3 (6-31G)"),
    }

    if molecule_key not in factories:
        raise ValueError(f"Unknown molecule: {molecule_key}")

    factory_fn, system_name = factories[molecule_key]

    print_banner(f"BENCHMARK: {system_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create molecule data with Hamiltonian and reference energies
    print("Creating Hamiltonian and computing reference energies...")
    mol_data = factory_fn(device=device)
    H = mol_data.hamiltonian

    n_qubits = H.num_sites
    n_electrons = H.n_electrons
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print(f"  System: {system_name}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons}")
    print(f"  Valid configs: {n_valid:,}")
    print(f"  HF Energy: {mol_data.hf_energy:.8f} Ha")
    if mol_data.ccsd_energy:
        print(f"  CCSD Energy: {mol_data.ccsd_energy:.8f} Ha")
    if mol_data.ccsd_t_energy:
        print(f"  CCSD(T) Energy: {mol_data.ccsd_t_energy:.8f} Ha")

    # Determine best available reference energy
    E_exact = mol_data.hf_energy
    energy_type = "HF"

    if n_valid <= 100000:
        print("Computing FCI energy (matrix-based)...")
        try:
            E_exact = H.fci_energy()
            energy_type = "FCI"
            print(f"  FCI Energy: {E_exact:.8f} Ha")
        except Exception as e:
            print(f"  Matrix-based FCI failed: {e}")

    if energy_type != "FCI" and n_valid <= 15_000_000:
        print(f"Computing FCI energy (PySCF iterative Davidson, {n_valid:,} determinants)...")
        try:
            t0 = time.time()
            E_fci = compute_pyscf_fci(mol_data.geometry, mol_data.basis)
            elapsed = time.time() - t0
            E_exact = E_fci
            energy_type = "FCI"
            mol_data.fci_energy = E_fci
            print(f"  PySCF FCI Energy: {E_exact:.8f} Ha (computed in {elapsed:.1f}s)")
        except Exception as e:
            print(f"  PySCF FCI failed: {e}")

    if energy_type != "FCI":
        if mol_data.ccsd_t_energy:
            E_exact = mol_data.ccsd_t_energy
            energy_type = "CCSD(T)"
        elif mol_data.ccsd_energy:
            E_exact = mol_data.ccsd_energy
            energy_type = "CCSD"

    result = BenchmarkResult(
        system=system_name,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        n_valid_configs=n_valid,
        exact_energy=E_exact,
        energy_type=energy_type,
    )

    def _get_energy(results):
        return results.get(
            'combined_energy',
            results.get('skqd_energy',
            results.get('sqd_energy', float('inf')))
        )

    # =======================================================================
    # Run SKQD mode
    # =======================================================================
    if mode in ("skqd", "all"):
        print("\n--- SKQD Mode (Krylov Time Evolution) ---")
        t0 = time.time()

        config_skqd = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            max_epochs=400,
            device=device,
        )
        config_skqd.adapt_to_system_size(n_valid)

        pipeline_skqd = FlowGuidedKrylovPipeline(H, config=config_skqd, exact_energy=E_exact)
        results_skqd = pipeline_skqd.run(progress=verbose)

        result.skqd_energy = _get_energy(results_skqd)
        result.time_skqd = time.time() - t0
        result.nf_configs = results_skqd.get('nf_basis_size', 0)
        result.nf_energy = results_skqd.get('nf_nqs_energy', 0.0)

        skqd_comparison = format_energy_comparison(result.skqd_energy, E_exact, energy_type)
        print(f"  SKQD energy: {result.skqd_energy:.8f} Ha ({skqd_comparison})")
        print(f"  SKQD time: {result.time_skqd:.1f}s")

    # =======================================================================
    # Run SQD-Clean mode
    # =======================================================================
    if mode in ("sqd-clean", "all"):
        print("\n--- SQD-Clean Mode (Batch Diag, No Noise) ---")
        t0 = time.time()

        config_sqd_clean = PipelineConfig(
            subspace_mode="sqd",
            skip_nf_training=True,
            max_epochs=400,
            sqd_num_batches=5,
            sqd_self_consistent_iters=3,
            sqd_noise_rate=0.0,
            device=device,
        )
        config_sqd_clean.adapt_to_system_size(n_valid)

        pipeline_sqd = FlowGuidedKrylovPipeline(H, config=config_sqd_clean, exact_energy=E_exact)
        results_sqd = pipeline_sqd.run(progress=verbose)

        result.sqd_clean_energy = _get_energy(results_sqd)
        result.time_sqd_clean = time.time() - t0

        sqd_comparison = format_energy_comparison(result.sqd_clean_energy, E_exact, energy_type)
        print(f"  SQD-Clean energy: {result.sqd_clean_energy:.8f} Ha ({sqd_comparison})")
        print(f"  SQD-Clean time: {result.time_sqd_clean:.1f}s")

    # =======================================================================
    # Run SQD-Recovery mode
    # =======================================================================
    if mode in ("sqd-recovery", "all"):
        print(f"\n--- SQD-Recovery Mode (Noise={noise_rate}, S-CORE Recovery) ---")
        t0 = time.time()

        config_sqd_recovery = PipelineConfig(
            subspace_mode="sqd",
            skip_nf_training=True,
            max_epochs=400,
            sqd_num_batches=5,
            sqd_self_consistent_iters=5,
            sqd_noise_rate=noise_rate,
            device=device,
        )
        config_sqd_recovery.adapt_to_system_size(n_valid)

        pipeline_sqd = FlowGuidedKrylovPipeline(H, config=config_sqd_recovery, exact_energy=E_exact)
        results_sqd = pipeline_sqd.run(progress=verbose)

        result.sqd_recovery_energy = _get_energy(results_sqd)
        result.time_sqd_recovery = time.time() - t0

        sqd_comparison = format_energy_comparison(result.sqd_recovery_energy, E_exact, energy_type)
        print(f"  SQD-Recovery energy: {result.sqd_recovery_energy:.8f} Ha ({sqd_comparison})")
        print(f"  SQD-Recovery time: {result.time_sqd_recovery:.1f}s")

    # =======================================================================
    # Compute errors
    # =======================================================================
    if result.skqd_energy != 0.0:
        result.skqd_error_mha = abs(result.skqd_energy - E_exact) * 1000
    if result.sqd_clean_energy != 0.0:
        result.sqd_clean_error_mha = abs(result.sqd_clean_energy - E_exact) * 1000
    if result.sqd_recovery_energy != 0.0:
        result.sqd_recovery_error_mha = abs(result.sqd_recovery_energy - E_exact) * 1000

    # =======================================================================
    # Summary
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY: {system_name}")
    print(f"{'='*70}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Valid Configs: {n_valid:,}")
    print(f"  Reference: {energy_type} = {E_exact:.8f} Ha")
    print(f"  NF Configs: {result.nf_configs}")

    if mode in ("skqd", "all"):
        print(f"  SKQD Energy:         {result.skqd_energy:.8f} Ha (error: {result.skqd_error_mha:.4f} mHa)")
    if mode in ("sqd-clean", "all"):
        print(f"  SQD-Clean Energy:    {result.sqd_clean_energy:.8f} Ha (error: {result.sqd_clean_error_mha:.4f} mHa)")
    if mode in ("sqd-recovery", "all"):
        print(f"  SQD-Recovery Energy: {result.sqd_recovery_energy:.8f} Ha (error: {result.sqd_recovery_error_mha:.4f} mHa)")

    if mode == "all":
        errors = {}
        if result.skqd_energy != 0.0:
            errors["SKQD"] = result.skqd_error_mha
        if result.sqd_clean_energy != 0.0:
            errors["SQD-Clean"] = result.sqd_clean_error_mha
        if result.sqd_recovery_energy != 0.0:
            errors["SQD-Recovery"] = result.sqd_recovery_error_mha
        if errors:
            best = min(errors, key=errors.get)
            print(f"  Best method: {best} ({errors[best]:.4f} mHa)")

    if energy_type == "FCI":
        best_error = min(
            result.skqd_error_mha if result.skqd_energy != 0.0 else float('inf'),
            result.sqd_clean_error_mha if result.sqd_clean_energy != 0.0 else float('inf'),
            result.sqd_recovery_error_mha if result.sqd_recovery_energy != 0.0 else float('inf'),
        )
        chem_acc = "PASS" if best_error < 1.6 else "FAIL"
        print(f"  Chemical accuracy: {chem_acc} (best error: {best_error:.4f} mHa)")

    print(f"{'='*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Moderate System Subspace Benchmark")
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["co", "hcn", "c2h2", "h2o_631g", "h2s", "c2h4", "nh3_631g", "all"],
        help="System to benchmark",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["skqd", "sqd-clean", "sqd-recovery", "all"],
        help="Subspace mode to run (default: all)",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.1,
        help="Depolarizing noise rate for SQD-Recovery mode (default: 0.1)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    system_order = ["co", "hcn", "c2h2", "h2o_631g", "h2s", "c2h4", "nh3_631g"]

    if args.system == "all":
        systems_to_run = system_order
    else:
        systems_to_run = [args.system]

    all_results = []

    for system_key in systems_to_run:
        try:
            result = run_benchmark(system_key, mode=args.mode, noise_rate=args.noise_rate, verbose=not args.quiet)
            all_results.append(result)
        except Exception as e:
            print(f"\nError running {system_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if len(all_results) > 1:
        print("\n" + "="*90)
        print("OVERALL RESULTS SUMMARY")
        print("="*90)

        if args.mode == "all":
            print(f"\n{'System':<20} {'Qubits':>8} {'Valid':>12} {'Ref':>8} "
                  f"{'SKQD err':>10} {'Clean err':>10} {'Recov err':>10} {'Best':<10}")
            print("-"*110)

            for r in all_results:
                errors = {}
                if r.skqd_energy != 0.0:
                    errors["SKQD"] = r.skqd_error_mha
                if r.sqd_clean_energy != 0.0:
                    errors["Clean"] = r.sqd_clean_error_mha
                if r.sqd_recovery_energy != 0.0:
                    errors["Recovery"] = r.sqd_recovery_error_mha
                best = min(errors, key=errors.get) if errors else ""
                print(f"{r.system:<20} {r.n_qubits:>8} {r.n_valid_configs:>12,} "
                      f"{r.energy_type:>8} {r.skqd_error_mha:>10.4f} "
                      f"{r.sqd_clean_error_mha:>10.4f} {r.sqd_recovery_error_mha:>10.4f} "
                      f"{best:<10}")
        else:
            print(f"\n{'System':<20} {'Qubits':>8} {'Valid':>12} {'Ref':>8} "
                  f"{'Error (mHa)':>12} {'Time (s)':>10}")
            print("-"*80)

            for r in all_results:
                if args.mode == "skqd":
                    error, t = r.skqd_error_mha, r.time_skqd
                elif args.mode == "sqd-clean":
                    error, t = r.sqd_clean_error_mha, r.time_sqd_clean
                else:
                    error, t = r.sqd_recovery_error_mha, r.time_sqd_recovery
                print(f"{r.system:<20} {r.n_qubits:>8} {r.n_valid_configs:>12,} "
                      f"{r.energy_type:>8} {error:>12.4f} {t:>10.1f}")

        print("-"*110)

        # Chemical accuracy count
        chem_acc_count = sum(
            1 for r in all_results
            if r.energy_type == "FCI" and min(
                r.skqd_error_mha if r.skqd_energy != 0.0 else float('inf'),
                r.sqd_clean_error_mha if r.sqd_clean_energy != 0.0 else float('inf'),
                r.sqd_recovery_error_mha if r.sqd_recovery_energy != 0.0 else float('inf'),
            ) < 1.6
        )
        fci_count = sum(1 for r in all_results if r.energy_type == "FCI")
        if fci_count > 0:
            print(f"\nChemical accuracy (<1.6 mHa): {chem_acc_count}/{fci_count} systems with FCI reference")


if __name__ == "__main__":
    main()
