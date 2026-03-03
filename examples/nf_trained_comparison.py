"""
Direct-CI Ablation: 7-Experiment Comparison (Basis Strategy x Solver).

Compares 7 subspace diagonalization configurations testing the impact of
Direct-CI (HF + singles + doubles) injection:

                        SKQD solver              SQD solver
  +-----------------+-----------------------+-----------------------+
  | HF only         | CudaQ SKQD            |                       |
  | (CudaQ-style)   | HF ref -> Krylov      |                       |
  +-----------------+-----------------------+-----------------------+
  | Direct-CI       | Pure SKQD             | Pure SQD              |
  | (HF+S+D)       | HF+S+D -> Krylov      | HF+S+D -> noise+SCORE |
  +-----------------+-----------------------+-----------------------+
  | NF + Direct-CI  | NF-Trained SKQD       | NF-Trained SQD        |
  | (current)       | NF+HF+S+D -> Krylov   | NF+HF+S+D -> noise   |
  +-----------------+-----------------------+-----------------------+
  | NF only         | NF-only SKQD          | NF-only SQD           |
  | (no Direct-CI)  | NF basis -> Krylov    | NF basis -> noise     |
  +-----------------+-----------------------+-----------------------+

Two ablation axes:
  1. HF-only vs Direct-CI: Does pre-injecting singles+doubles help vs
     letting Krylov discover them? (CudaQ SKQD vs Pure SKQD)
  2. NF+Direct-CI vs NF-only: Does essential config injection help when
     the NF provides a learned basis? (NF+CI vs NF-only)

References:
  [1] Yu, Robledo-Moreno et al., "Sample-based Krylov Quantum Diagonalization"
  [2] Robledo-Moreno, Motta et al., "Chemistry Beyond Exact Diagonalization", Science 2024
  [3] "Improved Ground State Estimation via Normalising Flow-Assisted NQS"
  [4] NVIDIA CUDA-Q SKQD Tutorial (Heisenberg model, Trotterized evolution)

Usage:
    # In Docker (GPU):
    docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py
    docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py --systems h2 lih
    docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py --systems nh3 ch4 n2

    # Local (CPU):
    python examples/nf_trained_comparison.py --systems h2
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from math import comb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from hamiltonians.molecular import (
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_h2o_hamiltonian,
    create_beh2_hamiltonian,
    create_nh3_hamiltonian,
    create_n2_hamiltonian,
    create_ch4_hamiltonian,
)
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Molecular system registry
# ---------------------------------------------------------------------------

SYSTEMS = {
    "h2": {
        "name": "H2 (STO-3G)",
        "factory": create_h2_hamiltonian,
        "kwargs": {"bond_length": 0.74},
    },
    "lih": {
        "name": "LiH (STO-3G)",
        "factory": create_lih_hamiltonian,
        "kwargs": {"bond_length": 1.6},
    },
    "h2o": {
        "name": "H2O (STO-3G)",
        "factory": create_h2o_hamiltonian,
        "kwargs": {},
    },
    "beh2": {
        "name": "BeH2 (STO-3G)",
        "factory": create_beh2_hamiltonian,
        "kwargs": {},
    },
    "nh3": {
        "name": "NH3 (STO-3G)",
        "factory": create_nh3_hamiltonian,
        "kwargs": {},
    },
    "ch4": {
        "name": "CH4 (STO-3G)",
        "factory": create_ch4_hamiltonian,
        "kwargs": {},
    },
    "n2": {
        "name": "N2 (STO-3G)",
        "factory": create_n2_hamiltonian,
        "kwargs": {},
    },
}


@dataclass
class ComparisonResult:
    """Stores results for one molecular system."""

    system: str
    name: str
    n_qubits: int
    n_configs: int
    fci_energy: float

    # Shared NF training
    nf_training_time_s: Optional[float] = None
    nf_basis_size: Optional[int] = None
    nf_energy: Optional[float] = None

    # Pure SKQD results (Direct-CI only, no NF training)
    pure_skqd_energy: Optional[float] = None
    pure_skqd_error_mha: Optional[float] = None
    pure_skqd_time_s: Optional[float] = None
    pure_skqd_basis_size: Optional[int] = None

    # NF-Trained SKQD results (NF + Direct-CI)
    skqd_energy: Optional[float] = None
    skqd_error_mha: Optional[float] = None
    skqd_diag_time_s: Optional[float] = None
    skqd_basis_size: Optional[int] = None

    # NF-Trained SQD results (NF + Direct-CI)
    sqd_energy: Optional[float] = None
    sqd_error_mha: Optional[float] = None
    sqd_diag_time_s: Optional[float] = None
    sqd_basis_size: Optional[int] = None
    sqd_noise_rate: Optional[float] = None
    sqd_valid_after_noise: Optional[int] = None
    sqd_shots_multiplier: Optional[int] = None

    # Pure SQD results (Direct-CI only, no NF training)
    pure_sqd_energy: Optional[float] = None
    pure_sqd_error_mha: Optional[float] = None
    pure_sqd_time_s: Optional[float] = None
    pure_sqd_basis_size: Optional[int] = None
    pure_sqd_valid_after_noise: Optional[int] = None

    # NF-only SKQD results (NF basis only, NO Direct-CI injection)
    nf_only_skqd_energy: Optional[float] = None
    nf_only_skqd_error_mha: Optional[float] = None
    nf_only_skqd_time_s: Optional[float] = None
    nf_only_skqd_basis_size: Optional[int] = None

    # NF-only SQD results (NF basis only, NO Direct-CI injection)
    nf_only_sqd_energy: Optional[float] = None
    nf_only_sqd_error_mha: Optional[float] = None
    nf_only_sqd_time_s: Optional[float] = None
    nf_only_sqd_basis_size: Optional[int] = None
    nf_only_sqd_valid_after_noise: Optional[int] = None

    # CudaQ-style SKQD results (HF-only start, Krylov discovers everything)
    cudaq_skqd_energy: Optional[float] = None
    cudaq_skqd_error_mha: Optional[float] = None
    cudaq_skqd_time_s: Optional[float] = None
    cudaq_skqd_basis_size: Optional[int] = None



# ---------------------------------------------------------------------------
# NF training parameters (tuned per system size)
# ---------------------------------------------------------------------------


def get_training_params(n_configs: int) -> dict:
    """
    Return NF training parameters scaled to system size.

    Small systems converge quickly; larger ones need more epochs and samples.
    Architecture dims also scale to capture the larger ground-state support.
    """
    if n_configs <= 10:
        return dict(
            max_epochs=100, min_epochs=30, samples_per_batch=500,
            nf_hidden_dims=[128, 128], nqs_hidden_dims=[128, 128, 128],
        )
    elif n_configs <= 300:
        return dict(
            max_epochs=150, min_epochs=50, samples_per_batch=1000,
            nf_hidden_dims=[128, 128], nqs_hidden_dims=[128, 128, 128],
        )
    elif n_configs <= 2000:
        return dict(
            max_epochs=200, min_epochs=80, samples_per_batch=1500,
            nf_hidden_dims=[256, 256], nqs_hidden_dims=[256, 256, 256],
        )
    elif n_configs <= 5000:
        return dict(
            max_epochs=300, min_epochs=100, samples_per_batch=2000,
            nf_hidden_dims=[256, 256], nqs_hidden_dims=[256, 256, 256],
        )
    else:
        return dict(
            max_epochs=400, min_epochs=150, samples_per_batch=3000,
            nf_hidden_dims=[512, 512], nqs_hidden_dims=[512, 512, 512],
        )


def get_noise_rate(n_qubits: int) -> float:
    """
    Return depolarizing noise rate scaled to system size.

    Calibrated so that with get_shots_multiplier() replication, >99% of
    essential configs survive noise in at least one copy. The per-config
    survival probability is (1 - noise_rate)^n_qubits; with M replications,
    P(at least one survives) = 1 - (1 - p_survive)^M.

    For the IBM SQD paper [2], the quantum circuit produces millions of shots,
    so even high noise rates preserve all important configs statistically.
    With our limited replication (~15-200x), we use moderate noise rates
    that balance S-CORE diversity against config loss.
    """
    if n_qubits <= 4:
        return 0.03  # H2: 4 qubits, P(survive)=0.89
    elif n_qubits <= 12:
        return 0.05  # LiH: 12 qubits, P(survive)=0.54
    elif n_qubits <= 14:
        return 0.05  # H2O, BeH2: 14 qubits, P(survive)=0.49
    elif n_qubits <= 16:
        return 0.05  # NH3: 16 qubits, P(survive)=0.44
    else:
        return 0.05  # CH4 (18), N2 (20): P(survive)=0.40/0.36


def get_shots_multiplier(n_unique: int, n_qubits: int) -> int:
    """
    Return number of replications per unique config to simulate circuit shots.

    In the IBM SQD paper [2], the quantum circuit produces millions of shots
    (many copies of each configuration). Hardware noise corrupts each shot
    independently, so important configs survive in some copies while corrupted
    copies provide diverse inputs for S-CORE recovery.

    With a unique NF basis (no duplicates), each config has only ONE chance —
    if noise corrupts it, it is lost. Replicating simulates the multi-shot
    nature of quantum circuit sampling.

    Scaled to produce ~10K-50K total shots (balancing S-CORE diversity vs memory).
    """
    target_shots = 20_000
    # Ensure at least 10 copies, at most 200
    multiplier = max(10, min(200, target_shots // max(n_unique, 1)))
    return multiplier


def get_skqd_params(n_configs: int) -> dict:
    """
    Return SHARED Krylov parameters for all SKQD variants.

    These are identical across CudaQ SKQD, Pure SKQD, NF+CI SKQD, and
    NF-only SKQD to ensure fair comparison where the ONLY variable is
    the initial basis.
    """
    if n_configs <= 300:
        return dict(max_krylov_dim=8, shots_per_krylov=100_000)
    elif n_configs <= 5000:
        return dict(max_krylov_dim=10, shots_per_krylov=200_000)
    else:
        return dict(max_krylov_dim=12, shots_per_krylov=200_000)


def get_sqd_params(n_configs: int) -> dict:
    """
    Return SHARED SQD parameters for all SQD variants.

    These are identical across Pure SQD, NF+CI SQD, and NF-only SQD
    to ensure fair comparison where the ONLY variable is the initial basis.
    """
    if n_configs <= 2000:
        sqd_num_batches = 5
    elif n_configs <= 5000:
        sqd_num_batches = 8
    else:
        sqd_num_batches = 10
    return dict(
        sqd_num_batches=sqd_num_batches,
        sqd_sc_iters=5,
        use_spin_sym=n_configs <= 5000,
    )


# ---------------------------------------------------------------------------
# Shared NF training (once per system)
# ---------------------------------------------------------------------------


def train_shared_nf(
    H, E_fci: float, n_configs: int, device: str, verbose: bool = True
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Train NF once with particle-conserving flow, return the accumulated basis.

    Both SKQD and SQD paths will reuse this basis, isolating the subspace
    method as the only variable in the comparison.

    Returns:
        nf_basis: Tensor of particle-conserving configurations
        nf_energy: NF-NQS variational energy estimate
        full_results: Pipeline results dict from training stages
    """
    train_params = get_training_params(n_configs)

    config = PipelineConfig(
        # Particle-conserving flow (GumbelTopK enforces exact N_alpha + N_beta)
        use_particle_conserving_flow=True,
        # SKQD mode for stage 3 (will not actually run stage 3)
        subspace_mode="skqd",
        # NF training enabled
        skip_nf_training=False,
        # Training hyperparameters
        max_epochs=train_params["max_epochs"],
        min_epochs=train_params["min_epochs"],
        samples_per_batch=train_params["samples_per_batch"],
        nf_lr=5e-4,
        nqs_lr=1e-3,
        convergence_threshold=0.20,
        # Architecture (scaled with system size)
        nf_hidden_dims=train_params["nf_hidden_dims"],
        nqs_hidden_dims=train_params["nqs_hidden_dims"],
        # Basis management
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        # Device
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Run only stages 1 + 2 (training + basis extraction)
    pipeline.train_flow_nqs(progress=verbose)
    nf_basis = pipeline.extract_and_select_basis()

    nf_energy = pipeline.results.get("nf_nqs_energy", None)

    return nf_basis, nf_energy, pipeline.results


# ---------------------------------------------------------------------------
# Pure SKQD path: Direct-CI (HF + singles + doubles) -> Krylov expansion
# ---------------------------------------------------------------------------


def run_pure_skqd_path(
    H, E_fci: float, n_configs: int,
    device: str, verbose: bool = True,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Pure SKQD: HF + singles + doubles -> Krylov time evolution -> eigensolver.

    This is the classical analog of the NVIDIA CUDA-Q SKQD tutorial [1].
    No NF training — the initial basis is purely deterministic (Direct-CI):
      - Hartree-Fock state (reference)
      - All single excitations
      - All double excitations

    Krylov time evolution e^{-iHdt} then expands this basis by discovering
    connected configurations through Hamiltonian connectivity, exactly as
    in the quantum circuit version where U^k|psi_0> explores Hilbert space.

    This establishes the SKQD baseline: how well does Krylov expansion alone
    perform, without NF-guided importance sampling?

    References:
      [1] NVIDIA CUDA-Q SKQD Tutorial (Heisenberg model, Trotterized evolution)
      [3] Yu, Robledo-Moreno et al., "Sample-based Krylov Quantum Diagonalization"
    """
    skqd_p = get_skqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="skqd",
        skip_nf_training=True,  # Direct-CI: HF + singles + doubles only
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        max_krylov_dim=skqd_p["max_krylov_dim"],
        time_step=0.1,
        shots_per_krylov=skqd_p["shots_per_krylov"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (HF + singles + doubles)
    # NO NF basis injection — this is the key difference from NF-trained SKQD
    pipeline.train_flow_nqs(progress=verbose)

    # Stage 2: Basis extraction (essential configs only)
    pipeline.extract_and_select_basis()

    # Stage 3: SKQD Krylov expansion + diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("skqd_energy", float("inf")),
    )
    basis_size = pipeline.results.get("nf_basis_size", 0)

    return energy, basis_size, pipeline.results


# ---------------------------------------------------------------------------
# Pure SQD path: Direct-CI (HF + singles + doubles) -> SQD batch diag
# ---------------------------------------------------------------------------


def run_pure_sqd_path(
    H, E_fci: float, n_configs: int, n_qubits: int,
    noise_rate: float, device: str, verbose: bool = True,
) -> Tuple[float, int, int, Dict[str, Any]]:
    """
    Pure SQD: HF + singles + doubles -> shot replication -> noise -> S-CORE -> batch diag.

    Same as NF-Trained SQD but uses only Direct-CI (essential configs) as the
    initial basis, with NO NF training. Tests whether SQD's S-CORE recovery
    can compensate for a small, deterministic initial basis.
    """
    sqd_p = get_sqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="sqd",
        skip_nf_training=True,  # Direct-CI: HF + singles + doubles only
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        sqd_num_batches=sqd_p["sqd_num_batches"],
        sqd_batch_size=0,  # auto
        sqd_self_consistent_iters=sqd_p["sqd_sc_iters"],
        sqd_noise_rate=noise_rate,
        sqd_use_spin_symmetry=sqd_p["use_spin_sym"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (HF + singles + doubles)
    pipeline.train_flow_nqs(progress=verbose)

    # Stage 2: Basis extraction (essential configs only)
    pipeline.extract_and_select_basis()

    # Replicate basis to simulate quantum circuit multi-shot sampling
    n_unique = len(pipeline.nf_basis)
    shots_mult = get_shots_multiplier(n_unique, n_qubits)
    replicated = pipeline.nf_basis.repeat(shots_mult, 1)
    pipeline.nf_basis = replicated
    if verbose:
        print(f"  Shot replication: {n_unique} unique x {shots_mult} = "
              f"{len(replicated)} shots (simulating quantum circuit sampling)")

    # Stage 3: SQD with noise injection + S-CORE recovery + batch diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("sqd_energy", float("inf")),
    )
    basis_size = n_unique

    sqd_results = pipeline.results.get("sqd_results", {})
    valid_configs = sqd_results.get("num_valid_configs", 0)

    return energy, basis_size, valid_configs, pipeline.results


# ---------------------------------------------------------------------------
# CudaQ-style SKQD: HF-only start -> Krylov discovers everything
# ---------------------------------------------------------------------------


def run_cudaq_skqd_path(
    H, E_fci: float, n_configs: int,
    device: str, verbose: bool = True,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    CudaQ-style SKQD: HF reference only -> Krylov time evolution -> eigensolver.

    Faithful to the NVIDIA CUDA-Q SKQD tutorial [4]: starts from ONLY the
    Hartree-Fock reference state (no singles/doubles pre-injection). Krylov
    time evolution e^{-iHdt} must discover all connected configurations
    through Hamiltonian connectivity alone.

    This tests how much the Direct-CI (singles+doubles) pre-injection in
    Pure SKQD actually contributes vs letting Krylov do all the work.

    References:
      [4] NVIDIA CUDA-Q SKQD Tutorial (Trotterized evolution from reference)
    """
    skqd_p = get_skqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="skqd",
        skip_nf_training=True,
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        max_krylov_dim=skqd_p["max_krylov_dim"],
        time_step=0.1,
        shots_per_krylov=skqd_p["shots_per_krylov"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (we'll replace with HF only)
    pipeline.train_flow_nqs(progress=verbose)

    # REPLACE essential configs with HF-only (CudaQ style: single reference state)
    n_essentials = len(pipeline._essential_configs) if hasattr(pipeline, '_essential_configs') else 0
    hf_state = H.get_hf_state().to(device)
    pipeline._essential_configs = hf_state.unsqueeze(0)  # shape: (1, n_sites)
    if verbose:
        print(f"  CudaQ-style: using HF reference only (1 config, "
              f"discarded {n_essentials - 1} Direct-CI configs)")

    # Stage 2: Basis extraction (HF-only)
    pipeline.extract_and_select_basis()

    # Stage 3: SKQD Krylov expansion + diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("skqd_energy", float("inf")),
    )
    basis_size = pipeline.results.get("nf_basis_size", 0)

    return energy, basis_size, pipeline.results


# ---------------------------------------------------------------------------
# NF-Trained SKQD path: shared NF basis -> Krylov time evolution
# ---------------------------------------------------------------------------


def run_skqd_path(
    H, E_fci: float, n_configs: int, nf_basis: torch.Tensor,
    device: str, verbose: bool = True,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    SKQD: Pre-trained NF basis -> Krylov time evolution -> eigensolver.

    Following [1] (SKQD paper):
    - Krylov states e^{-ikHdt}|psi> discover connected configurations
    - Union of Krylov-step bases is diagonalized via sparse eigsh
    - Subspace grows at each step via Hamiltonian connectivity

    Uses skip_nf_training=True with pre-trained basis injected, so only
    Stage 3 (Krylov expansion + diag) runs.
    """
    skqd_p = get_skqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="skqd",
        skip_nf_training=True,  # use pre-trained basis
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        max_krylov_dim=skqd_p["max_krylov_dim"],
        time_step=0.1,
        shots_per_krylov=skqd_p["shots_per_krylov"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (HF + singles + doubles)
    pipeline.train_flow_nqs(progress=verbose)

    # Inject pre-trained NF basis (merge with essential configs)
    essential = pipeline._essential_configs
    combined = torch.cat([essential, nf_basis.to(essential.device)], dim=0)
    pipeline._essential_configs = torch.unique(combined, dim=0)

    # Stage 2: Basis extraction (uses merged essential + NF basis)
    pipeline.extract_and_select_basis()

    # Stage 3: SKQD Krylov expansion + diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("skqd_energy", float("inf")),
    )
    basis_size = pipeline.results.get("nf_basis_size", 0)

    return energy, basis_size, pipeline.results


# ---------------------------------------------------------------------------
# NF-Trained SQD path: shared NF basis -> noise injection -> S-CORE -> batch diag
# ---------------------------------------------------------------------------


def run_sqd_path(
    H, E_fci: float, n_configs: int, n_qubits: int,
    nf_basis: torch.Tensor, noise_rate: float,
    device: str, verbose: bool = True,
) -> Tuple[float, int, int, Dict[str, Any]]:
    """
    SQD: Pre-trained NF basis -> shot replication -> noise -> S-CORE -> batch diag.

    Following [2] (IBM SQD paper):
    - Particle-conserving sampler (NF replaces LUCJ circuit) produces configs
    - Configs are replicated to simulate multi-shot quantum circuit sampling
    - Depolarizing noise (emulates quantum hardware noise) corrupts some shots
    - S-CORE (Self-Consistent Configuration Recovery) restores valid configs
      using self-consistent orbital occupancies
    - K independent batch diagonalizations
    - Energy-variance extrapolation across batches

    Shot replication is necessary because the IBM paper's quantum circuit
    produces millions of shots (many copies of each config), so important
    configs survive noise in some copies. With unique NF configs, each has
    only one chance — replication restores this statistical redundancy.
    """
    sqd_p = get_sqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="sqd",
        skip_nf_training=True,  # use pre-trained basis
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        sqd_num_batches=sqd_p["sqd_num_batches"],
        sqd_batch_size=0,  # auto
        sqd_self_consistent_iters=sqd_p["sqd_sc_iters"],
        sqd_noise_rate=noise_rate,
        sqd_use_spin_symmetry=sqd_p["use_spin_sym"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (HF + singles + doubles)
    pipeline.train_flow_nqs(progress=verbose)

    # Inject pre-trained NF basis (merge with essential configs)
    essential = pipeline._essential_configs
    combined = torch.cat([essential, nf_basis.to(essential.device)], dim=0)
    unique_basis = torch.unique(combined, dim=0)
    pipeline._essential_configs = unique_basis

    # Stage 2: Basis extraction (uses merged essential + NF basis)
    pipeline.extract_and_select_basis()

    # Replicate basis to simulate quantum circuit multi-shot sampling [2].
    # The IBM paper's circuit produces millions of shots — many copies of each
    # configuration. Hardware noise corrupts each shot independently, so:
    #   - Important configs survive in some copies (statistical redundancy)
    #   - Corrupted copies provide diverse inputs for S-CORE recovery
    # Without replication, each unique config has only ONE chance; if noise
    # corrupts it, that configuration is permanently lost.
    n_unique = len(pipeline.nf_basis)
    shots_mult = get_shots_multiplier(n_unique, n_qubits)
    replicated = pipeline.nf_basis.repeat(shots_mult, 1)
    pipeline.nf_basis = replicated
    if verbose:
        print(f"  Shot replication: {n_unique} unique x {shots_mult} = "
              f"{len(replicated)} shots (simulating quantum circuit sampling)")

    # Stage 3: SQD with noise injection + S-CORE recovery + batch diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("sqd_energy", float("inf")),
    )
    # Report original unique basis size (not replicated count)
    basis_size = n_unique

    # Extract SQD-specific diagnostics
    sqd_results = pipeline.results.get("sqd_results", {})
    valid_configs = sqd_results.get("num_valid_configs", 0)

    return energy, basis_size, valid_configs, pipeline.results


# ---------------------------------------------------------------------------
# NF-only SKQD path: NF basis only (no Direct-CI injection) -> Krylov
# ---------------------------------------------------------------------------


def run_nf_only_skqd_path(
    H, E_fci: float, n_configs: int, nf_basis: torch.Tensor,
    device: str, verbose: bool = True,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    NF-only SKQD: NF basis ONLY (no essential config injection) -> Krylov.

    Same as NF-Trained SKQD but does NOT merge HF + singles + doubles.
    Tests how much the NF-discovered basis alone suffices for Krylov expansion.
    """
    skqd_p = get_skqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="skqd",
        skip_nf_training=True,
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        max_krylov_dim=skqd_p["max_krylov_dim"],
        time_step=0.1,
        shots_per_krylov=skqd_p["shots_per_krylov"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (we'll discard them)
    pipeline.train_flow_nqs(progress=verbose)

    # REPLACE essential configs with NF-only basis (NO merge with Direct-CI)
    pipeline._essential_configs = nf_basis.to(device)

    # Stage 2: Basis extraction (NF-only basis)
    pipeline.extract_and_select_basis()

    # Stage 3: SKQD Krylov expansion + diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("skqd_energy", float("inf")),
    )
    basis_size = pipeline.results.get("nf_basis_size", 0)

    return energy, basis_size, pipeline.results


# ---------------------------------------------------------------------------
# NF-only SQD path: NF basis only (no Direct-CI injection) -> SQD
# ---------------------------------------------------------------------------


def run_nf_only_sqd_path(
    H, E_fci: float, n_configs: int, n_qubits: int,
    nf_basis: torch.Tensor, noise_rate: float,
    device: str, verbose: bool = True,
) -> Tuple[float, int, int, Dict[str, Any]]:
    """
    NF-only SQD: NF basis ONLY (no essential config injection) -> SQD.

    Same as NF-Trained SQD but does NOT merge HF + singles + doubles.
    Tests how SQD's S-CORE recovery performs when the initial basis
    comes purely from the normalizing flow, without deterministic
    essential configs as anchors.
    """
    sqd_p = get_sqd_params(n_configs)

    config = PipelineConfig(
        use_particle_conserving_flow=True,
        subspace_mode="sqd",
        skip_nf_training=True,
        max_accumulated_basis=min(n_configs, 8192),
        use_diversity_selection=True,
        max_diverse_configs=min(n_configs, 4096),
        sqd_num_batches=sqd_p["sqd_num_batches"],
        sqd_batch_size=0,
        sqd_self_consistent_iters=sqd_p["sqd_sc_iters"],
        sqd_noise_rate=noise_rate,
        sqd_use_spin_symmetry=sqd_p["use_spin_sym"],
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=E_fci, auto_adapt=False
    )

    # Stage 1: Direct-CI generates essential configs (we'll discard them)
    pipeline.train_flow_nqs(progress=verbose)

    # REPLACE essential configs with NF-only basis (NO merge with Direct-CI)
    pipeline._essential_configs = nf_basis.to(device)

    # Stage 2: Basis extraction (NF-only basis)
    pipeline.extract_and_select_basis()

    # Replicate basis to simulate quantum circuit multi-shot sampling
    n_unique = len(pipeline.nf_basis)
    shots_mult = get_shots_multiplier(n_unique, n_qubits)
    replicated = pipeline.nf_basis.repeat(shots_mult, 1)
    pipeline.nf_basis = replicated
    if verbose:
        print(f"  Shot replication: {n_unique} unique x {shots_mult} = "
              f"{len(replicated)} shots (simulating quantum circuit sampling)")

    # Stage 3: SQD with noise injection + S-CORE recovery + batch diag
    pipeline.run_subspace_diag(progress=verbose)
    pipeline._print_summary()

    energy = pipeline.results.get(
        "combined_energy",
        pipeline.results.get("sqd_energy", float("inf")),
    )
    basis_size = n_unique

    sqd_results = pipeline.results.get("sqd_results", {})
    valid_configs = sqd_results.get("num_valid_configs", 0)

    return energy, basis_size, valid_configs, pipeline.results


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------


def run_comparison(system_keys, device, noise_rate_override=None, verbose=True):
    """Run SKQD vs SQD comparison on specified systems."""
    results = []

    for key in system_keys:
        if key not in SYSTEMS:
            print(f"Unknown system '{key}', skipping.")
            continue

        info = SYSTEMS[key]
        print("\n" + "#" * 70)
        print(f"# System: {info['name']}")
        print("#" * 70)

        # Create Hamiltonian
        H = info["factory"](**info["kwargs"], device=device)
        E_fci = H.fci_energy()
        n_qubits = H.num_sites
        n_configs = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
        noise_rate = noise_rate_override if noise_rate_override is not None else get_noise_rate(n_qubits)

        print(f"  Qubits: {n_qubits}, Configs: {n_configs:,}, FCI: {E_fci:.8f} Ha")
        print(f"  SQD noise rate: {noise_rate:.2f}")

        result = ComparisonResult(
            system=key,
            name=info["name"],
            n_qubits=n_qubits,
            n_configs=n_configs,
            fci_energy=E_fci,
            sqd_noise_rate=noise_rate,
        )

        # --- Pure SKQD (Direct-CI only, no NF training) ---
        print(f"\n{'='*60}")
        print(f"  Running Pure SKQD (HF+singles+doubles -> Krylov) [3]...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            pure_energy, pure_basis, _ = run_pure_skqd_path(
                H, E_fci, n_configs, device, verbose=verbose
            )
            pure_time = time.time() - t0
            result.pure_skqd_energy = pure_energy
            result.pure_skqd_error_mha = abs(pure_energy - E_fci) * 1000
            result.pure_skqd_time_s = pure_time
            result.pure_skqd_basis_size = pure_basis
        except Exception as e:
            print(f"  PURE SKQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        # --- CudaQ-style SKQD (HF-only -> Krylov) ---
        print(f"\n{'='*60}")
        print(f"  Running CudaQ SKQD (HF-only -> Krylov) [4]...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            cq_skqd_energy, cq_skqd_basis, _ = run_cudaq_skqd_path(
                H, E_fci, n_configs, device, verbose=verbose
            )
            cq_skqd_time = time.time() - t0
            result.cudaq_skqd_energy = cq_skqd_energy
            result.cudaq_skqd_error_mha = abs(cq_skqd_energy - E_fci) * 1000
            result.cudaq_skqd_time_s = cq_skqd_time
            result.cudaq_skqd_basis_size = cq_skqd_basis
        except Exception as e:
            print(f"  CUDAQ SKQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        # --- Shared NF training ---
        print(f"\n{'='*60}")
        print(f"  Training shared NF (particle-conserving flow)...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            nf_basis, nf_energy, _ = train_shared_nf(
                H, E_fci, n_configs, device, verbose=verbose
            )
            nf_time = time.time() - t0
            result.nf_training_time_s = nf_time
            result.nf_basis_size = len(nf_basis)
            result.nf_energy = nf_energy
            print(f"\n  NF training complete: {len(nf_basis)} basis configs in {nf_time:.1f}s")
            if nf_energy is not None:
                print(f"  NF-NQS energy: {nf_energy:.8f} Ha")
        except Exception as e:
            print(f"  NF TRAINING FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(result)
            continue

        # --- NF-Trained SKQD path ---
        print(f"\n{'='*60}")
        print(f"  Running NF-Trained SKQD (NF basis + Krylov expansion) [1]...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            skqd_energy, skqd_basis, _ = run_skqd_path(
                H, E_fci, n_configs, nf_basis, device, verbose=verbose
            )
            skqd_time = time.time() - t0
            result.skqd_energy = skqd_energy
            result.skqd_error_mha = abs(skqd_energy - E_fci) * 1000
            result.skqd_diag_time_s = skqd_time
            result.skqd_basis_size = skqd_basis
        except Exception as e:
            print(f"  SKQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        # --- NF-Trained SQD path (NF + Direct-CI -> SQD) ---
        print(f"\n{'='*60}")
        print(f"  Running NF-Trained SQD (NF+Direct-CI + noise={noise_rate:.2f} + S-CORE) [2]...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            sqd_energy, sqd_basis, sqd_valid, sqd_results = run_sqd_path(
                H, E_fci, n_configs, n_qubits, nf_basis, noise_rate,
                device, verbose=verbose,
            )
            sqd_time = time.time() - t0
            result.sqd_energy = sqd_energy
            result.sqd_error_mha = abs(sqd_energy - E_fci) * 1000
            result.sqd_diag_time_s = sqd_time
            result.sqd_basis_size = sqd_basis
            result.sqd_valid_after_noise = sqd_valid
        except Exception as e:
            print(f"  SQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        # --- Pure SQD (Direct-CI only -> SQD) ---
        print(f"\n{'='*60}")
        print(f"  Running Pure SQD (HF+singles+doubles + noise={noise_rate:.2f} + S-CORE)...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            pure_sqd_energy, pure_sqd_basis, pure_sqd_valid, _ = run_pure_sqd_path(
                H, E_fci, n_configs, n_qubits, noise_rate,
                device, verbose=verbose,
            )
            pure_sqd_time = time.time() - t0
            result.pure_sqd_energy = pure_sqd_energy
            result.pure_sqd_error_mha = abs(pure_sqd_energy - E_fci) * 1000
            result.pure_sqd_time_s = pure_sqd_time
            result.pure_sqd_basis_size = pure_sqd_basis
            result.pure_sqd_valid_after_noise = pure_sqd_valid
        except Exception as e:
            print(f"  PURE SQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        # --- NF-only SKQD (NF basis only, no Direct-CI -> SKQD) ---
        print(f"\n{'='*60}")
        print(f"  Running NF-only SKQD (NF basis only, NO Direct-CI -> Krylov)...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            nfo_skqd_energy, nfo_skqd_basis, _ = run_nf_only_skqd_path(
                H, E_fci, n_configs, nf_basis, device, verbose=verbose
            )
            nfo_skqd_time = time.time() - t0
            result.nf_only_skqd_energy = nfo_skqd_energy
            result.nf_only_skqd_error_mha = abs(nfo_skqd_energy - E_fci) * 1000
            result.nf_only_skqd_time_s = nfo_skqd_time
            result.nf_only_skqd_basis_size = nfo_skqd_basis
        except Exception as e:
            print(f"  NF-ONLY SKQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        # --- NF-only SQD (NF basis only, no Direct-CI -> SQD) ---
        print(f"\n{'='*60}")
        print(f"  Running NF-only SQD (NF basis only, NO Direct-CI + noise={noise_rate:.2f})...")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            nfo_sqd_energy, nfo_sqd_basis, nfo_sqd_valid, _ = run_nf_only_sqd_path(
                H, E_fci, n_configs, n_qubits, nf_basis, noise_rate,
                device, verbose=verbose,
            )
            nfo_sqd_time = time.time() - t0
            result.nf_only_sqd_energy = nfo_sqd_energy
            result.nf_only_sqd_error_mha = abs(nfo_sqd_energy - E_fci) * 1000
            result.nf_only_sqd_time_s = nfo_sqd_time
            result.nf_only_sqd_basis_size = nfo_sqd_basis
            result.nf_only_sqd_valid_after_noise = nfo_sqd_valid
        except Exception as e:
            print(f"  NF-ONLY SQD FAILED: {e}")
            import traceback
            traceback.print_exc()

        results.append(result)

    return results


def _fmt_err(v):
    """Format error value in mHa."""
    return f"{v:.4f}" if v is not None else "FAIL"


def _fmt_t(v):
    """Format time value in seconds."""
    return f"{v:.1f}s" if v is not None else "-"


def _fmt_e(v):
    """Format energy value in Ha."""
    return f"{v:.8f}" if v is not None else "FAIL"


def print_summary_table(results):
    """Print formatted 7-experiment ablation comparison table."""
    W = 150
    print("\n")
    print("=" * W)
    print("DIRECT-CI ABLATION: 7-EXPERIMENT COMPARISON")
    print("=" * W)

    # --- Error summary (mHa) ---
    print(f"\nError (mHa) — lower is better:")
    print(
        f"{'System':<18} {'Qubits':>6} {'Configs':>8} "
        f"{'|':>2} {'CudaQ SKQD':>11} "
        f"{'|':>2} {'PureSKQD':>10} {'PureSQD':>10} "
        f"{'|':>2} {'NF+CI SKQD':>11} {'NF+CI SQD':>11} "
        f"{'|':>2} {'NF-only SKQD':>13} {'NF-only SQD':>13}"
    )
    print(
        f"{'':18} {'':>6} {'':>8} "
        f"{'|':>2} {'(HF only)':>11} "
        f"{'|':>2} {'(HF+S+D)':>10} {'(HF+S+D)':>10} "
        f"{'|':>2} {'(NF+HF+S+D)':>11} {'(NF+HF+S+D)':>11} "
        f"{'|':>2} {'(NF only)':>13} {'(NF only)':>13}"
    )
    print("-" * W)

    for r in results:
        print(
            f"{r.name:<18} {r.n_qubits:>6} {r.n_configs:>8,} "
            f"{'|':>2} {_fmt_err(r.cudaq_skqd_error_mha):>11} "
            f"{'|':>2} {_fmt_err(r.pure_skqd_error_mha):>10} "
            f"{_fmt_err(r.pure_sqd_error_mha):>10} "
            f"{'|':>2} {_fmt_err(r.skqd_error_mha):>11} "
            f"{_fmt_err(r.sqd_error_mha):>11} "
            f"{'|':>2} {_fmt_err(r.nf_only_skqd_error_mha):>13} "
            f"{_fmt_err(r.nf_only_sqd_error_mha):>13}"
        )

    print("-" * W)

    # --- Time summary ---
    print(f"\nDiag Time (seconds):")
    print(
        f"{'System':<18} "
        f"{'|':>2} {'CudaQ SKQD':>11} "
        f"{'|':>2} {'PureSKQD':>10} {'PureSQD':>10} "
        f"{'|':>2} {'NF+CI SKQD':>11} {'NF+CI SQD':>11} "
        f"{'|':>2} {'NF-only SKQD':>13} {'NF-only SQD':>13} "
        f"{'|':>2} {'NF train':>10}"
    )
    print("-" * W)

    for r in results:
        print(
            f"{r.name:<18} "
            f"{'|':>2} {_fmt_t(r.cudaq_skqd_time_s):>11} "
            f"{'|':>2} {_fmt_t(r.pure_skqd_time_s):>10} "
            f"{_fmt_t(r.pure_sqd_time_s):>10} "
            f"{'|':>2} {_fmt_t(r.skqd_diag_time_s):>11} "
            f"{_fmt_t(r.sqd_diag_time_s):>11} "
            f"{'|':>2} {_fmt_t(r.nf_only_skqd_time_s):>13} "
            f"{_fmt_t(r.nf_only_sqd_time_s):>13} "
            f"{'|':>2} {_fmt_t(r.nf_training_time_s):>10}"
        )

    print("-" * W)

    # --- Detailed energies ---
    print(f"\nDetailed Energies (Ha):")
    print(
        f"{'System':<18} {'FCI':>14} "
        f"{'CudaQ SKQD':>14} "
        f"{'PureSKQD':>14} {'PureSQD':>14} "
        f"{'NF+CI SKQD':>14} {'NF+CI SQD':>14} "
        f"{'NFonly SKQD':>14} {'NFonly SQD':>14}"
    )
    print("-" * 130)
    for r in results:
        print(
            f"{r.name:<18} {r.fci_energy:>14.8f} "
            f"{_fmt_e(r.cudaq_skqd_energy):>14} "
            f"{_fmt_e(r.pure_skqd_energy):>14} {_fmt_e(r.pure_sqd_energy):>14} "
            f"{_fmt_e(r.skqd_energy):>14} {_fmt_e(r.sqd_energy):>14} "
            f"{_fmt_e(r.nf_only_skqd_energy):>14} {_fmt_e(r.nf_only_sqd_energy):>14}"
        )

    # --- Chemical accuracy ---
    CHEM_ACC = 1.6  # mHa (= 1 kcal/mol)
    print(f"\nChemical Accuracy Threshold: {CHEM_ACC} mHa (1 kcal/mol)")
    labels = [
        ("CudaQ SKQD", "cudaq_skqd_error_mha"),
        ("PureSKQD", "pure_skqd_error_mha"),
        ("PureSQD", "pure_sqd_error_mha"),
        ("NF+CI SKQD", "skqd_error_mha"),
        ("NF+CI SQD", "sqd_error_mha"),
        ("NF-only SKQD", "nf_only_skqd_error_mha"),
        ("NF-only SQD", "nf_only_sqd_error_mha"),
    ]
    for r in results:
        parts = []
        for label, attr in labels:
            val = getattr(r, attr)
            if val is not None:
                tag = "PASS" if val < CHEM_ACC else "FAIL"
            else:
                tag = "ERR"
            parts.append(f"{label}: {tag}")
        print(f"  {r.name:<18}  {' | '.join(parts)}")

    # --- Ablation 1: HF-only vs Direct-CI (SKQD only) ---
    print(f"\nAblation 1: HF-only (CudaQ) vs Direct-CI (Pure) — SKQD only, no NF:")
    print(
        f"{'System':<18} "
        f"{'CudaQ SKQD':>12} {'Pure SKQD':>12} {'delta':>10} "
        f"{'Interpretation':>20}"
    )
    print("-" * 75)
    for r in results:
        cq = r.cudaq_skqd_error_mha
        pure = r.pure_skqd_error_mha
        d = (cq - pure) if (cq is not None and pure is not None) else None
        interp = ""
        if d is not None:
            if d > 0.01:
                interp = "S+D HELPS"
            elif d < -0.01:
                interp = "HF-only better"
            else:
                interp = "Negligible"
        print(
            f"{r.name:<18} "
            f"{_fmt_err(cq):>12} {_fmt_err(pure):>12} "
            f"{(f'{d:+.4f}' if d is not None else '-'):>10} "
            f"{interp:>20}"
        )

    # --- Ablation 2: NF+CI vs NF-only ---
    print(f"\nAblation 2: NF+Direct-CI vs NF-only — impact of essential config injection:")
    print(
        f"{'System':<18} "
        f"{'SKQD+CI':>10} {'SKQD-only':>10} {'delta':>10} "
        f"{'SQD+CI':>10} {'SQD-only':>10} {'delta':>10} "
        f"{'Interpretation':>20}"
    )
    print("-" * 100)
    for r in results:
        s_ci = r.skqd_error_mha
        s_no = r.nf_only_skqd_error_mha
        q_ci = r.sqd_error_mha
        q_no = r.nf_only_sqd_error_mha
        d_s = (s_ci - s_no) if (s_ci is not None and s_no is not None) else None
        d_q = (q_ci - q_no) if (q_ci is not None and q_no is not None) else None

        interp = ""
        if d_s is not None and d_q is not None:
            if d_s < -0.01 or d_q < -0.01:
                interp = "CI HELPS"
            elif d_s > 0.01 or d_q > 0.01:
                interp = "CI HURTS"
            else:
                interp = "Negligible"

        print(
            f"{r.name:<18} "
            f"{_fmt_err(s_ci):>10} {_fmt_err(s_no):>10} "
            f"{(f'{d_s:+.4f}' if d_s is not None else '-'):>10} "
            f"{_fmt_err(q_ci):>10} {_fmt_err(q_no):>10} "
            f"{(f'{d_q:+.4f}' if d_q is not None else '-'):>10} "
            f"{interp:>20}"
        )

    # --- Configuration space analysis ---
    print(f"\nConfiguration Space Analysis:")
    for r in results:
        parts = []
        if r.cudaq_skqd_basis_size is not None:
            parts.append(f"CudaQ={r.cudaq_skqd_basis_size}")
        if r.pure_skqd_basis_size is not None:
            parts.append(f"Direct-CI={r.pure_skqd_basis_size}")
        if r.nf_basis_size is not None:
            parts.append(f"NF={r.nf_basis_size}")
        if r.skqd_basis_size is not None:
            parts.append(f"NF+CI SKQD={r.skqd_basis_size}")
        if r.nf_only_skqd_basis_size is not None:
            parts.append(f"NF-only SKQD={r.nf_only_skqd_basis_size}")
        if parts:
            print(f"  {r.name}: {', '.join(parts)}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Direct-CI Ablation: 7-Experiment Comparison (Basis Strategy x Solver)"
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["h2", "lih", "h2o", "beh2"],
        choices=list(SYSTEMS.keys()),
        help="Molecular systems to compare (default: h2 lih h2o beh2)",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=None,
        help="Override SQD depolarizing noise rate (default: auto-scale by system size)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step training output",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Systems: {args.systems}")
    if args.noise_rate is not None:
        print(f"SQD noise rate override: {args.noise_rate}")
    else:
        print("SQD noise rate: auto-scaled by system size")

    results = run_comparison(
        args.systems, device,
        noise_rate_override=args.noise_rate,
        verbose=not args.quiet,
    )
    print_summary_table(results)


if __name__ == "__main__":
    main()
