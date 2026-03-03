"""
Large System Benchmark: SKQD vs SQD on Challenging Molecules

This benchmark tests both subspace construction strategies on larger
molecular systems with active space selection:

1. Benzene - 6 pi electrons in 6 orbitals
2. Butadiene - Extended pi system (8e, 8o)
3. Fe(II)-porphyrin model - Transition metal with d-orbital correlations
4. N2 with larger basis (cc-pVDZ)
5. Ozone - Multi-reference system
6. Cr2 (chromium dimer) - Notorious multi-reference system

Usage:
    docker-compose run --rm flow-krylov-gpu python examples/large_system_benchmark.py --system all
    docker-compose run --rm flow-krylov-gpu python examples/large_system_benchmark.py --system cr2
    docker-compose run --rm flow-krylov-gpu python examples/large_system_benchmark.py --system benzene --mode sqd
"""

import sys
from pathlib import Path
import argparse
import time
from math import comb
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

try:
    from pyscf import gto, scf, mcscf, ao2mo, fci
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("ERROR: PySCF required. Install with: pip install pyscf")
    sys.exit(1)

from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


@dataclass
class LargeSystemResult:
    """Container for large system benchmark results."""
    name: str
    n_orbitals: int
    n_electrons: int
    n_alpha: int
    n_beta: int
    n_qubits: int
    n_valid_configs: int

    # Reference energies
    hf_energy: float = 0.0
    casci_energy: Optional[float] = None

    # NF baseline
    nf_energy: float = 0.0
    nf_configs: int = 0

    # Mode energies
    skqd_energy: float = 0.0
    sqd_energy: float = 0.0

    # Timing
    time_nf: float = 0.0
    time_skqd: float = 0.0
    time_sqd: float = 0.0

    # Errors (mHa, relative to best available reference)
    skqd_error_mha: float = 0.0
    sqd_error_mha: float = 0.0

    notes: str = ""


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compute_active_space_integrals(
    mol,
    mf,
    n_active_orbitals: int,
    n_active_electrons: int,
    active_orbitals: Optional[List[int]] = None,
) -> Tuple[MolecularIntegrals, float]:
    """Compute molecular integrals in an active space."""
    n_orbitals = mol.nao
    n_electrons = mol.nelectron

    if active_orbitals is None:
        n_occ = n_electrons // 2
        n_active_occ = n_active_electrons // 2
        n_active_virt = n_active_orbitals - n_active_occ

        start_occ = max(0, n_occ - n_active_occ)
        end_virt = min(n_orbitals, n_occ + n_active_virt)
        active_orbitals = list(range(start_occ, end_virt))

    mo_coeff = mf.mo_coeff
    mo_active = mo_coeff[:, active_orbitals]

    core_orbitals = [i for i in range(min(active_orbitals))]
    n_core = len(core_orbitals)

    if n_core > 0:
        mo_core = mo_coeff[:, core_orbitals]
        dm_core = 2.0 * mo_core @ mo_core.T
        h1_ao = mf.get_hcore()
        j_core, k_core = mf.get_jk(dm=dm_core)
        core_energy = mol.energy_nuc() + 0.5 * np.einsum('ij,ji->', h1_ao + 0.5 * j_core - 0.25 * k_core, dm_core)
        h1e_eff = mo_active.T @ (h1_ao + j_core - 0.5 * k_core) @ mo_active
    else:
        core_energy = mol.energy_nuc()
        h1e_eff = mo_active.T @ mf.get_hcore() @ mo_active

    h2e_active = ao2mo.kernel(mol, mo_active)
    h2e_active = ao2mo.restore(1, h2e_active, n_active_orbitals)

    n_alpha = (n_active_electrons + mol.spin) // 2
    n_beta = (n_active_electrons - mol.spin) // 2

    integrals = MolecularIntegrals(
        h1e=h1e_eff,
        h2e=h2e_active,
        nuclear_repulsion=core_energy,
        n_electrons=n_active_electrons,
        n_orbitals=n_active_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    return integrals, core_energy


# =============================================================================
# Large Molecule Definitions
# =============================================================================

def create_cr2_hamiltonian(
    bond_length: float = 1.68,
    n_active_orbitals: int = 12,
    n_active_electrons: int = 12,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """Create Cr2 (chromium dimer) Hamiltonian with active space."""
    print(f"Creating Cr2 with bond length {bond_length} A")
    print(f"Active space: ({n_active_electrons}e, {n_active_orbitals}o)")

    mol = gto.Mole()
    mol.atom = [
        ('Cr', (0.0, 0.0, 0.0)),
        ('Cr', (0.0, 0.0, bond_length)),
    ]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.symmetry = False
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )
    H = MolecularHamiltonian(integrals, device=device)
    return H, {'hf_energy': hf_energy, 'core_energy': core_energy, 'mol': mol, 'mf': mf}


def create_benzene_hamiltonian(
    n_active_orbitals: int = 6,
    n_active_electrons: int = 6,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """Create benzene Hamiltonian with pi-electron active space."""
    print(f"Creating benzene with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    cc_bond = 1.40
    ch_bond = 1.09
    angles = [i * 60 for i in range(6)]
    carbon_coords = []
    hydrogen_coords = []

    for angle in angles:
        rad = np.radians(angle)
        carbon_coords.append(('C', (cc_bond * np.cos(rad), cc_bond * np.sin(rad), 0.0)))
        hydrogen_coords.append(('H', (
            (cc_bond + ch_bond) * np.cos(rad),
            (cc_bond + ch_bond) * np.sin(rad), 0.0)))

    mol = gto.Mole()
    mol.atom = carbon_coords + hydrogen_coords
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )
    H = MolecularHamiltonian(integrals, device=device)
    return H, {'hf_energy': hf_energy, 'core_energy': core_energy, 'mol': mol, 'mf': mf}


def create_fe_porphyrin_model_hamiltonian(
    n_active_orbitals: int = 10,
    n_active_electrons: int = 8,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """Create a simplified Fe-porphyrin model Hamiltonian."""
    print(f"Creating Fe-porphyrin model with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    fe_n_dist = 2.0
    mol = gto.Mole()
    mol.atom = [
        ('Fe', (0.0, 0.0, 0.0)),
        ('N', (fe_n_dist, 0.0, 0.0)),
        ('N', (-fe_n_dist, 0.0, 0.0)),
        ('N', (0.0, fe_n_dist, 0.0)),
        ('N', (0.0, -fe_n_dist, 0.0)),
    ]
    mol.basis = 'sto-3g'
    mol.charge = 2
    mol.spin = 4
    mol.verbose = 3
    mol.build()

    mf = scf.ROHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )
    H = MolecularHamiltonian(integrals, device=device)
    return H, {'hf_energy': hf_energy, 'core_energy': core_energy, 'mol': mol, 'mf': mf}


def create_n2_large_basis_hamiltonian(
    bond_length: float = 1.10,
    basis: str = "cc-pvdz",
    n_active_orbitals: int = 14,
    n_active_electrons: int = 10,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """Create N2 with larger basis set and active space."""
    print(f"Creating N2 ({basis}) with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    mol = gto.Mole()
    mol.atom = [
        ('N', (0.0, 0.0, 0.0)),
        ('N', (0.0, 0.0, bond_length)),
    ]
    mol.basis = basis
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )
    H = MolecularHamiltonian(integrals, device=device)
    return H, {'hf_energy': hf_energy, 'core_energy': core_energy, 'mol': mol, 'mf': mf}


def create_butadiene_hamiltonian(
    n_active_orbitals: int = 8,
    n_active_electrons: int = 8,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """Create butadiene (C4H6) Hamiltonian."""
    print(f"Creating butadiene with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    cc_single = 1.46
    cc_double = 1.34
    ch_bond = 1.09

    mol = gto.Mole()
    mol.atom = [
        ('C', (0.0, 0.0, 0.0)),
        ('C', (cc_double, 0.0, 0.0)),
        ('C', (cc_double + cc_single, 0.0, 0.0)),
        ('C', (2*cc_double + cc_single, 0.0, 0.0)),
        ('H', (-ch_bond * 0.866, ch_bond * 0.5, 0.0)),
        ('H', (-ch_bond * 0.866, -ch_bond * 0.5, 0.0)),
        ('H', (2*cc_double + cc_single + ch_bond * 0.866, ch_bond * 0.5, 0.0)),
        ('H', (2*cc_double + cc_single + ch_bond * 0.866, -ch_bond * 0.5, 0.0)),
        ('H', (cc_double + cc_single/2, ch_bond, 0.0)),
        ('H', (cc_double + cc_single/2, -ch_bond, 0.0)),
    ]
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )
    H = MolecularHamiltonian(integrals, device=device)
    return H, {'hf_energy': hf_energy, 'core_energy': core_energy, 'mol': mol, 'mf': mf}


def create_ozone_hamiltonian(
    n_active_orbitals: int = 12,
    n_active_electrons: int = 12,
    device: str = "cuda",
) -> Tuple[MolecularHamiltonian, dict]:
    """Create ozone (O3) Hamiltonian."""
    print(f"Creating ozone with ({n_active_electrons}e, {n_active_orbitals}o) active space")

    oo_bond = 1.278
    angle = 116.8
    angle_rad = np.radians(angle / 2)

    mol = gto.Mole()
    mol.atom = [
        ('O', (0.0, 0.0, 0.0)),
        ('O', (oo_bond * np.cos(angle_rad), oo_bond * np.sin(angle_rad), 0.0)),
        ('O', (oo_bond * np.cos(angle_rad), -oo_bond * np.sin(angle_rad), 0.0)),
    ]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    print(f"HF energy: {hf_energy:.8f} Ha")

    integrals, core_energy = compute_active_space_integrals(
        mol, mf, n_active_orbitals, n_active_electrons
    )
    H = MolecularHamiltonian(integrals, device=device)
    return H, {'hf_energy': hf_energy, 'core_energy': core_energy, 'mol': mol, 'mf': mf}


# =============================================================================
# Benchmark Function
# =============================================================================

def run_large_system_benchmark(
    system_name: str,
    H: MolecularHamiltonian,
    info: dict,
    mode: str = "both",
    compute_casci: bool = False,
) -> LargeSystemResult:
    """Run benchmark on a large molecular system using SKQD and/or SQD."""
    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)

    print_banner(f"BENCHMARK: {system_name}")
    print(f"Active space: ({H.n_alpha + H.n_beta}e, {H.n_orbitals}o)")
    print(f"Qubits: {H.num_sites}")
    print(f"Valid configurations: {n_valid:,}")
    print(f"Full Hilbert space: {2**H.num_sites:,}")

    result = LargeSystemResult(
        name=system_name,
        n_orbitals=H.n_orbitals,
        n_electrons=H.n_alpha + H.n_beta,
        n_alpha=H.n_alpha,
        n_beta=H.n_beta,
        n_qubits=H.num_sites,
        n_valid_configs=n_valid,
        hf_energy=info.get('hf_energy', 0.0),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optionally compute CASCI reference
    if compute_casci and n_valid <= 50000:
        try:
            print("\nComputing CASCI reference...")
            start = time.time()
            casci_energy = H.fci_energy()
            result.casci_energy = casci_energy
            print(f"CASCI energy: {casci_energy:.8f} Ha ({time.time()-start:.1f}s)")
        except Exception as e:
            print(f"CASCI failed: {e}")

    # Reference energy for error computation
    ref_energy = result.casci_energy if result.casci_energy is not None else result.hf_energy

    # =======================================================================
    # SKQD Mode
    # =======================================================================
    if mode in ("skqd", "both"):
        print("\n--- SKQD Mode (Krylov Time Evolution) ---")
        start_skqd = time.time()

        config_skqd = PipelineConfig(
            subspace_mode="skqd",
            max_epochs=800,
            samples_per_batch=4000,
            device=device,
        )
        config_skqd.adapt_to_system_size(n_valid)

        pipeline_skqd = FlowGuidedKrylovPipeline(H, config=config_skqd, exact_energy=result.casci_energy)
        results_skqd = pipeline_skqd.run(progress=True)

        result.skqd_energy = results_skqd.get(
            'combined_energy',
            results_skqd.get('skqd_energy',
            results_skqd.get('sqd_energy', float('inf')))
        )
        result.nf_configs = results_skqd.get('nf_basis_size', 0)
        result.nf_energy = results_skqd.get('nf_nqs_energy', 0.0)
        result.time_skqd = time.time() - start_skqd

        if result.casci_energy is not None:
            result.skqd_error_mha = abs(result.skqd_energy - result.casci_energy) * 1000

        print(f"SKQD energy: {result.skqd_energy:.8f} Ha")
        if result.casci_energy is not None:
            print(f"SKQD error: {result.skqd_error_mha:.4f} mHa")
        print(f"SKQD time: {result.time_skqd:.1f}s")

    # =======================================================================
    # SQD Mode
    # =======================================================================
    if mode in ("sqd", "both"):
        print("\n--- SQD Mode (Sampling-Based Batch Diagonalization) ---")
        start_sqd = time.time()

        config_sqd = PipelineConfig(
            subspace_mode="sqd",
            max_epochs=800,
            samples_per_batch=4000,
            sqd_num_batches=5,
            sqd_self_consistent_iters=3,
            device=device,
        )
        config_sqd.adapt_to_system_size(n_valid)

        pipeline_sqd = FlowGuidedKrylovPipeline(H, config=config_sqd, exact_energy=result.casci_energy)
        results_sqd = pipeline_sqd.run(progress=True)

        result.sqd_energy = results_sqd.get(
            'combined_energy',
            results_sqd.get('sqd_energy',
            results_sqd.get('skqd_energy', float('inf')))
        )
        result.time_sqd = time.time() - start_sqd

        if result.casci_energy is not None:
            result.sqd_error_mha = abs(result.sqd_energy - result.casci_energy) * 1000

        print(f"SQD energy: {result.sqd_energy:.8f} Ha")
        if result.casci_energy is not None:
            print(f"SQD error: {result.sqd_error_mha:.4f} mHa")
        print(f"SQD time: {result.time_sqd:.1f}s")

    # =======================================================================
    # Results Summary
    # =======================================================================
    print("\n" + "=" * 80)
    print(f"RESULTS: {system_name}")
    print("=" * 80)
    print(f"{'Metric':<35} {'Value':<20}")
    print("-" * 55)
    print(f"{'Valid configurations':<35} {n_valid:<20,}")
    print(f"{'NF configs':<35} {result.nf_configs:<20,}")
    print("-" * 55)
    print(f"{'HF energy (Ha)':<35} {result.hf_energy:<20.8f}")
    print(f"{'NF energy (Ha)':<35} {result.nf_energy:<20.8f}")
    if mode in ("skqd", "both"):
        print(f"{'SKQD energy (Ha)':<35} {result.skqd_energy:<20.8f}")
    if mode in ("sqd", "both"):
        print(f"{'SQD energy (Ha)':<35} {result.sqd_energy:<20.8f}")
    if result.casci_energy is not None:
        print(f"{'CASCI energy (Ha)':<35} {result.casci_energy:<20.8f}")
    print("-" * 55)
    if mode in ("skqd", "both"):
        print(f"{'Time SKQD (s)':<35} {result.time_skqd:<20.1f}")
    if mode in ("sqd", "both"):
        print(f"{'Time SQD (s)':<35} {result.time_sqd:<20.1f}")

    if mode == "both" and result.skqd_energy != 0.0 and result.sqd_energy != 0.0:
        better = "SKQD" if result.skqd_energy < result.sqd_energy else "SQD"
        diff = abs(result.skqd_energy - result.sqd_energy) * 1000
        print(f"\n>>> Better method: {better} (by {diff:.2f} mHa) <<<")

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Large System Subspace Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Systems available:
    cr2           - Chromium dimer (12e, 12o) - ~43,000 configs
    benzene       - Benzene pi system (6e, 6o) - 400 configs
    benzene_large - Benzene extended (6e, 12o) - ~40,000 configs
    fe_porphyrin  - Fe-porphyrin model (8e, 10o) - ~6,300 configs
    n2_large      - N2 cc-pVDZ (10e, 14o) - ~26,000 configs
    butadiene     - Butadiene (8e, 8o) - ~4,900 configs
    ozone         - Ozone (12e, 12o) - ~62,000 configs
    all           - Run all benchmarks
        """
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default="all",
        help="System to benchmark"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["skqd", "sqd", "both"],
        help="Subspace mode to run (default: both)",
    )
    parser.add_argument(
        "--compute-casci",
        action="store_true",
        help="Compute CASCI reference (slow for large systems)"
    )

    args = parser.parse_args()

    print_banner("LARGE SYSTEM SUBSPACE BENCHMARK")
    print("Comparing SKQD (Krylov) vs SQD (Sampling-based) subspace construction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []

    systems = {
        'benzene': lambda: create_benzene_hamiltonian(6, 6, device),
        'butadiene': lambda: create_butadiene_hamiltonian(8, 8, device),
        'fe_porphyrin': lambda: create_fe_porphyrin_model_hamiltonian(10, 8, device),
        'n2_large': lambda: create_n2_large_basis_hamiltonian(1.10, 'cc-pvdz', 12, 10, device),
        'ozone': lambda: create_ozone_hamiltonian(9, 12, device),
        'benzene_large': lambda: create_benzene_hamiltonian(12, 6, device),
        'cr2': lambda: create_cr2_hamiltonian(1.68, 12, 12, device),
    }

    if args.system == "all":
        test_order = ['benzene', 'butadiene', 'fe_porphyrin', 'n2_large', 'ozone']
    else:
        test_order = [args.system]

    for system_name in test_order:
        if system_name not in systems:
            print(f"Unknown system: {system_name}")
            continue

        try:
            H, info = systems[system_name]()
            result = run_large_system_benchmark(
                system_name,
                H,
                info,
                mode=args.mode,
                compute_casci=args.compute_casci,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR running {system_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    if results:
        print("\n" + "=" * 100)
        print("LARGE SYSTEM BENCHMARK SUMMARY")
        print("=" * 100)

        if args.mode == "both":
            print(f"{'System':<15} {'Configs':<12} {'NF':<8} "
                  f"{'SKQD E':<16} {'SQD E':<16} {'Better':<10}")
            print("-" * 100)
            for r in results:
                better = ""
                if r.skqd_energy != 0.0 and r.sqd_energy != 0.0:
                    better = "SKQD" if r.skqd_energy < r.sqd_energy else "SQD"
                print(f"{r.name:<15} {r.n_valid_configs:<12,} {r.nf_configs:<8,} "
                      f"{r.skqd_energy:<16.8f} {r.sqd_energy:<16.8f} {better:<10}")
        else:
            print(f"{'System':<15} {'Configs':<12} {'NF':<8} {'Energy':<16} {'Time (s)':<10}")
            print("-" * 70)
            for r in results:
                energy = r.skqd_energy if args.mode == "skqd" else r.sqd_energy
                t = r.time_skqd if args.mode == "skqd" else r.time_sqd
                print(f"{r.name:<15} {r.n_valid_configs:<12,} {r.nf_configs:<8,} "
                      f"{energy:<16.8f} {t:<10.1f}")

        print("-" * 100)

        if args.mode == "both":
            print("\nScaling trend (sorted by system size):")
            sorted_results = sorted(results, key=lambda r: r.n_valid_configs)
            for r in sorted_results:
                diff = abs(r.skqd_energy - r.sqd_energy) * 1000 if r.skqd_energy != 0.0 and r.sqd_energy != 0.0 else 0
                better = "SKQD" if r.skqd_energy < r.sqd_energy else "SQD"
                print(f"  {r.n_valid_configs:>10,} configs: {better} better by {diff:.2f} mHa")


if __name__ == "__main__":
    main()
