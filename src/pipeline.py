"""
Flow-Guided Krylov Pipeline for Molecular Ground State Energy Calculation.

This module provides an end-to-end pipeline that combines:
1. Normalizing Flow-Assisted Neural Quantum States (NF-NQS) for basis discovery
2. Subspace diagonalization via SKQD (Krylov) or SQD (sampling-based)

Key Features:
- Particle Conservation: Samples valid molecular configurations only
- Physics-Guided Training: Mixed objective with energy importance
- Diversity Selection: Excitation-rank stratified basis selection
- Dual subspace mode: SKQD (Krylov time evolution) or SQD (batch diagonalization)
- Adaptive Scaling: Automatic parameter adjustment for system size

Usage:
    from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
    from src.hamiltonians.molecular import create_lih_hamiltonian

    H = create_lih_hamiltonian(bond_length=1.6)
    pipeline = FlowGuidedKrylovPipeline(H)
    results = pipeline.run()
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Flow components
try:
    from .flows.particle_conserving_flow import (
        ParticleConservingFlowSampler,
        verify_particle_conservation,
    )
    from .flows.autoregressive_flow import AutoregressiveFlowSampler
    from .flows.physics_guided_training import (
        PhysicsGuidedFlowTrainer,
        PhysicsGuidedConfig,
    )
    from .flows.vmc_training import VMCTrainer, VMCConfig
    from .flows.sign_network import SignNetwork
except ImportError:
    from flows.particle_conserving_flow import (
        ParticleConservingFlowSampler,
        verify_particle_conservation,
    )
    from flows.autoregressive_flow import AutoregressiveFlowSampler
    from flows.physics_guided_training import (
        PhysicsGuidedFlowTrainer,
        PhysicsGuidedConfig,
    )
    from flows.vmc_training import VMCTrainer, VMCConfig
    from flows.sign_network import SignNetwork

# NQS components
try:
    from .nqs.dense import DenseNQS
except ImportError:
    from nqs.dense import DenseNQS

# Hamiltonian components
try:
    from .hamiltonians.base import Hamiltonian
    from .hamiltonians.molecular import MolecularHamiltonian
except ImportError:
    from hamiltonians.base import Hamiltonian
    from hamiltonians.molecular import MolecularHamiltonian

# Postprocessing components
try:
    from .postprocessing.diversity_selection import (
        DiversitySelector,
        DiversityConfig,
    )
except ImportError:
    from postprocessing.diversity_selection import (
        DiversitySelector,
        DiversityConfig,
    )

# Krylov / subspace diagonalization components
try:
    from .krylov.skqd import (
        SampleBasedKrylovDiagonalization,
        FlowGuidedSKQD,
        SKQDConfig,
    )
    from .krylov.sqd import SQDSolver, SQDConfig
except ImportError:
    from krylov.skqd import (
        SampleBasedKrylovDiagonalization,
        FlowGuidedSKQD,
        SKQDConfig,
    )
    from krylov.sqd import SQDSolver, SQDConfig


@dataclass
class PipelineConfig:
    """
    Configuration for the Flow-Guided Krylov pipeline.

    This configuration supports molecular systems with particle conservation.
    """

    # Flow type
    use_particle_conserving_flow: bool = True  # Use particle-conserving flow for molecules
    # Autoregressive transformer flow (Phase 4a): captures inter-orbital correlations.
    # When True, uses AutoregressiveFlowSampler instead of ParticleConservingFlowSampler.
    # None = auto (adapt_to_system_size decides: >20K configs -> autoregressive)
    use_autoregressive_flow: Optional[bool] = None

    # NF-NQS architecture
    nf_hidden_dims: list = field(default_factory=lambda: [256, 256])
    nqs_hidden_dims: list = field(default_factory=lambda: [256, 256, 256, 256])

    # Training parameters
    samples_per_batch: int = 2000
    num_batches: int = 1
    max_epochs: int = 400
    min_epochs: int = 100
    convergence_threshold: float = 0.20

    # Physics-guided training weights - following paper's approach
    # Paper uses only cross-entropy for NF, weighted by |E|
    teacher_weight: float = 1.0  # Cross-entropy (paper's only term)
    physics_weight: float = 0.0  # Paper doesn't use this
    entropy_weight: float = 0.0  # Paper doesn't use this

    # Learning rates
    nf_lr: float = 5e-4
    nqs_lr: float = 1e-3

    # Basis management - ADAPTIVE: will be scaled by system size
    max_accumulated_basis: int = 4096  # Base value, scaled automatically

    # Diversity selection - ADAPTIVE: will be scaled by system size
    use_diversity_selection: bool = True
    max_diverse_configs: int = 2048  # Base value, scaled automatically
    rank_2_fraction: float = 0.50  # Emphasize double excitations

    # Subspace diagonalization mode
    subspace_mode: str = "skqd"  # "skqd" (Krylov time evolution) or "sqd" (SQD sampling-based)

    # SQD-specific parameters (used when subspace_mode="sqd")
    sqd_num_batches: int = 5  # K batches for independent diagonalization
    sqd_batch_size: int = 0  # d configs per batch (0 = auto from NF samples)
    sqd_self_consistent_iters: int = 3  # Self-consistent config recovery iterations
    sqd_spin_penalty: float = 0.0  # Lambda for S^2 penalty (0 = disabled)
    sqd_noise_rate: float = 0.0  # Depolarizing noise rate for SQD recovery mode (0 = clean SQD)
    sqd_use_spin_symmetry: bool = True  # Spin-up/down recombination in SQD batches

    # SKQD parameters (used when subspace_mode="skqd")
    max_krylov_dim: int = 8
    time_step: float = 0.1
    shots_per_krylov: int = 50000
    skqd_regularization: float = 1e-8  # Regularization for numerical stability
    max_diag_basis_size: int = 15000  # Max basis for diag (matches SKQDConfig default)
    skip_skqd: bool = False  # Skip Krylov refinement (for NF-only mode comparison)
    skqd_convergence_threshold: float = 1e-5  # Energy convergence threshold (Ha)
    skqd_adaptive_dt: bool = False  # Adaptive dt based on spectral range

    # NNCI parameters (Neural Network Configuration Interaction)
    # NNCI uses a feedforward NN to classify important higher-excitation configs
    # (triples, quadruples) beyond CISD, expanding the basis via active learning.
    # None = auto (adapt_to_system_size decides based on system size)
    use_nnci: Optional[bool] = None
    nnci_iterations: int = 5  # Active learning iterations
    nnci_candidates_per_iter: int = 5000  # Max candidates to generate per iteration

    # Training mode
    use_local_energy: bool = True  # Use VMC local energy (proper variational estimator)

    # VMC (Variational Monte Carlo) training mode
    # When True, after NF training (or instead of it), runs VMC to directly minimize
    # the variational energy <psi|H|psi> using REINFORCE on the autoregressive flow.
    # Requires use_autoregressive_flow=True.
    use_vmc_training: bool = False
    vmc_n_steps: int = 500  # VMC optimization steps
    vmc_lr: float = 1e-3  # VMC learning rate
    vmc_n_samples: int = 2000  # Samples per VMC step

    # Sign network for wavefunction sign structure
    # When True, adds a small feedforward network that learns sign(ψ(x)),
    # enabling the VMC ansatz ψ(x) = √p(x) × s(x) to represent wavefunctions
    # with negative CI coefficients.  Requires use_vmc_training=True.
    use_sign_network: bool = False

    # Direct-CI mode: skip NF-NQS training entirely
    # When True, pipeline uses essential configs (HF + singles + doubles) → subspace diagonalization
    # None = auto (let adapt_to_system_size decide based on system size)
    skip_nf_training: Optional[bool] = None

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Track if user explicitly set skip_nf_training (True or False)
        # so adapt_to_system_size() won't override their choice.
        if self.skip_nf_training is not None:
            self._user_set_skip_nf = True
        else:
            self.skip_nf_training = False  # Resolve None to default False

        # Track if user explicitly set use_nnci (True or False)
        # so adapt_to_system_size() won't override their choice.
        if self.use_nnci is not None:
            self._user_set_nnci = True
        else:
            self.use_nnci = False  # Resolve None to default False

        # Track if user explicitly set use_autoregressive_flow (True or False)
        # so adapt_to_system_size() won't override their choice.
        if self.use_autoregressive_flow is not None:
            self._user_set_autoregressive_flow = True
        else:
            self.use_autoregressive_flow = False  # Resolve None to default False

    # === PERFORMANCE OPTIMIZATIONS FOR LARGE SYSTEMS ===
    # These dramatically reduce training time for large molecules (>20 qubits)

    # Truncate connections to top-k by matrix element magnitude (0 = no truncation)
    max_connections_per_config: int = 0

    # Skip off-diagonal computation for first N epochs (diagonal-only warmup)
    diagonal_only_warmup_epochs: int = 0

    # Sample a fraction of connections for stochastic local energy (1.0 = all)
    stochastic_connections_fraction: float = 1.0

    def adapt_to_system_size(self, n_valid_configs: int, verbose: bool = True) -> "PipelineConfig":
        """
        Adapt configuration parameters based on the valid configuration space size.

        This is CRITICAL for larger molecules where the default parameters
        are insufficient for adequate basis coverage.

        Args:
            n_valid_configs: Number of particle-conserving configurations
            verbose: Whether to print adaptation info (default True, set False to suppress)

        Returns:
            Updated config (modifies self in-place and returns)
        """
        # Skip if already adapted to same size
        if hasattr(self, "_adapted_size") and self._adapted_size == n_valid_configs:
            return self

        # Determine system complexity tier
        if n_valid_configs <= 1000:
            tier = "small"
        elif n_valid_configs <= 5000:
            tier = "medium"
        elif n_valid_configs <= 20000:
            tier = "large"
        else:
            tier = "very_large"

        # Conditional NF training based on system size (PR 2.1 / ADR-001)
        # Systems <= 20K configs: Direct-CI sufficient (HF + singles + doubles)
        # Systems > 20K configs: NF training beneficial for finding important higher excitations
        # User explicit override (_user_set_skip_nf) is always preserved
        if not hasattr(self, "_user_set_skip_nf"):
            if n_valid_configs <= 20000:
                self.skip_nf_training = True
            else:
                self.skip_nf_training = False

        # Conditional NNCI based on system size (PR-B1)
        # Systems > 20K configs: NNCI discovers important triples/quadruples via NN classifier
        # User explicit override (_user_set_nnci) is always preserved
        if not hasattr(self, "_user_set_nnci"):
            if n_valid_configs > 20000:
                self.use_nnci = True
            else:
                self.use_nnci = False

        # Conditional autoregressive flow based on system size (Phase 4a)
        # Systems > 20K configs: autoregressive transformer captures inter-orbital
        # correlations that the non-autoregressive product-of-marginals model cannot.
        # User explicit override (_user_set_autoregressive_flow) is always preserved.
        if not hasattr(self, "_user_set_autoregressive_flow"):
            if n_valid_configs > 20000:
                self.use_autoregressive_flow = True
            else:
                self.use_autoregressive_flow = False

        if verbose:
            print(f"System size: {n_valid_configs:,} valid configs -> {tier} tier")
            if self.skip_nf_training:
                if hasattr(self, "_user_set_skip_nf"):
                    print("Direct-CI mode: user override (skip_nf_training=True)")
                else:
                    print(f"Direct-CI mode: {n_valid_configs:,} configs <= 20K threshold")
            else:
                if hasattr(self, "_user_set_skip_nf"):
                    print("NF training enabled: user override (skip_nf_training=False)")
                else:
                    print(
                        f"NF training enabled: {n_valid_configs:,} configs exceeds "
                        f"Direct-CI threshold (20K)"
                    )

        if tier == "small":
            # Small systems: default parameters are fine
            self.max_accumulated_basis = max(self.max_accumulated_basis, n_valid_configs)
            self.max_diverse_configs = min(n_valid_configs, self.max_diverse_configs)

        elif tier == "medium":
            # Medium systems: need more basis coverage
            self.max_accumulated_basis = min(n_valid_configs, 8192)
            self.max_diverse_configs = min(n_valid_configs, 4096)
            # Larger networks for more complex systems
            if len(self.nqs_hidden_dims) < 5:
                self.nqs_hidden_dims = [384, 384, 384, 384, 384]
            # SQD: more batches for better statistics
            if self.subspace_mode == "sqd":
                self.sqd_num_batches = max(self.sqd_num_batches, 8)
                if self.sqd_noise_rate > 0:
                    self.sqd_noise_rate = max(self.sqd_noise_rate, 0.05)

        elif tier == "large":
            # Large systems: aggressive basis collection
            self.max_accumulated_basis = min(n_valid_configs, 12288)
            self.max_diverse_configs = min(n_valid_configs, 8192)
            self.max_diag_basis_size = 25000
            # Larger networks
            self.nqs_hidden_dims = [512, 512, 512, 512, 512]
            # More training
            self.max_epochs = max(self.max_epochs, 600)
            self.samples_per_batch = 4000
            # Enable adaptive dt for large systems (prevents aliasing)
            self.skqd_adaptive_dt = True
            # SQD: more batches
            if self.subspace_mode == "sqd":
                self.sqd_num_batches = max(self.sqd_num_batches, 10)
                if self.sqd_noise_rate > 0:
                    self.sqd_noise_rate = max(self.sqd_noise_rate, 0.10)

        else:  # very_large
            # Very large systems (>20K valid configs, e.g. C2H4 with 9M)
            self.max_accumulated_basis = 16384
            self.max_diverse_configs = min(n_valid_configs, 12288)
            # Keep aligned with MAX_FULL_SUBSPACE_SIZE (15000) to ensure
            # even dense fallback paths stay within memory budget.
            # 15000² × 16 (complex128) = 3.6 GB — safe on DGX Spark 128GB UMA.
            self.max_diag_basis_size = 15000

            # Network capacity
            self.nqs_hidden_dims = [512, 512, 512, 512]
            self.nf_hidden_dims = [256, 256]

            # SHORT training: NF serves as warm-start, not primary basis builder
            self.max_epochs = max(self.max_epochs, 200)
            self.min_epochs = max(self.min_epochs, 50)
            self.samples_per_batch = 2000

            # IMPORTANT: Do NOT use performance hacks that cripple the energy signal
            self.max_connections_per_config = 0
            self.diagonal_only_warmup_epochs = 0
            self.stochastic_connections_fraction = 1.0

            # Reduce Krylov dimension for large systems
            self.max_krylov_dim = 4
            # Enable adaptive dt for very large systems (prevents aliasing)
            self.skqd_adaptive_dt = True

            # SQD: more batches for large systems
            if self.subspace_mode == "sqd":
                self.sqd_num_batches = max(self.sqd_num_batches, 10)
                if self.sqd_noise_rate > 0:
                    self.sqd_noise_rate = max(self.sqd_noise_rate, 0.15)

        # Compute coverage statistics
        coverage_accumulated = min(1.0, self.max_accumulated_basis / n_valid_configs)
        coverage_diverse = min(1.0, self.max_diverse_configs / n_valid_configs)

        if verbose:
            print(f"Adapted parameters:")
            print(f"  subspace_mode: {self.subspace_mode}")
            print(
                f"  max_accumulated_basis: {self.max_accumulated_basis:,} ({coverage_accumulated*100:.1f}% of valid)"
            )
            print(
                f"  max_diverse_configs: {self.max_diverse_configs:,} ({coverage_diverse*100:.1f}% of valid)"
            )
            print(f"  NQS hidden dims: {self.nqs_hidden_dims}")

        # Mark as adapted to prevent duplicate adaptation
        self._adapted_size = n_valid_configs

        return self


class FlowGuidedKrylovPipeline:
    """
    Flow-Guided Pipeline for ground state energy computation.

    This pipeline combines:
    1. Particle-conserving normalizing flow for valid molecular configurations
    2. Physics-guided NF-NQS co-training with mixed objective
    3. Diversity-aware basis selection by excitation rank
    4. Subspace diagonalization via SKQD (Krylov) or SQD (sampling-based)

    The workflow is:
    - Stage 1: Physics-guided NF-NQS training (discovers ground state support)
    - Stage 2: Diversity-aware basis extraction (stratified by excitation rank)
    - Stage 3: Subspace diagonalization (SKQD or SQD)

    Example usage:
    ```python
    from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
    from src.hamiltonians.molecular import create_lih_hamiltonian

    H = create_lih_hamiltonian(bond_length=1.6)
    E_exact = H.fci_energy()

    pipeline = FlowGuidedKrylovPipeline(H, exact_energy=E_exact)
    results = pipeline.run()

    print(f"Final energy: {results['combined_energy']:.6f} Ha")
    print(f"Error: {abs(results['combined_energy'] - E_exact) * 1000:.2f} mHa")
    ```

    Args:
        hamiltonian: System Hamiltonian
        config: Pipeline configuration
        exact_energy: Known exact energy for validation (optional)
        auto_adapt: Automatically adapt config to system size
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: Optional[PipelineConfig] = None,
        exact_energy: Optional[float] = None,
        auto_adapt: bool = True,
    ):
        from math import comb

        self.hamiltonian = hamiltonian
        self.config = config or PipelineConfig()
        self.exact_energy = exact_energy
        self.num_sites = hamiltonian.num_sites
        self.device = self.config.device

        # Verify molecular Hamiltonian (particle conservation required)
        if not isinstance(hamiltonian, MolecularHamiltonian):
            raise TypeError(
                "FlowGuidedKrylovPipeline requires a MolecularHamiltonian. "
                "Use create_*_hamiltonian() factory functions from src.hamiltonians.molecular."
            )
        self.is_molecular = True

        # Compute valid configuration space size
        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta
        self.n_valid_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

        # Automatically adapt configuration to system size
        if auto_adapt:
            self.config.adapt_to_system_size(self.n_valid_configs)

        # Initialize components
        self._init_components()

        # Results storage
        self.results: Dict[str, Any] = {}

    def _init_components(self):
        """Initialize flow, NQS, and auxiliary components."""
        cfg = self.config

        # Initialize flow sampler for molecules — skip if Direct-CI mode
        # (avoid allocating GPU memory for a model that won't be used)
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        if cfg.skip_nf_training and not cfg.use_vmc_training:
            self.flow = None
            print("Direct-CI mode: flow sampler not initialized (saves GPU memory)")
        elif cfg.use_autoregressive_flow:
            self.flow = AutoregressiveFlowSampler(
                num_sites=self.num_sites,
                n_alpha=n_alpha,
                n_beta=n_beta,
            ).to(self.device)
            print(
                f"Using autoregressive transformer flow: "
                f"{n_alpha}\u03b1 + {n_beta}\u03b2 electrons"
            )
        else:
            self.flow = ParticleConservingFlowSampler(
                num_sites=self.num_sites,
                n_alpha=n_alpha,
                n_beta=n_beta,
                hidden_dims=cfg.nf_hidden_dims,
            ).to(self.device)
            print(f"Using particle-conserving flow: " f"{n_alpha}\u03b1 + {n_beta}\u03b2 electrons")

        # Sign network for wavefunction sign structure
        if cfg.use_sign_network and cfg.use_vmc_training:
            self.sign_network = SignNetwork(num_sites=self.num_sites).to(self.device)
            print(f"Sign network initialized ({sum(p.numel() for p in self.sign_network.parameters())} params)")
        else:
            if cfg.use_sign_network and not cfg.use_vmc_training:
                print("WARNING: use_sign_network=True requires use_vmc_training=True. Sign network not created.")
            self.sign_network = None

        # Neural Quantum State
        self.nqs = DenseNQS(
            num_sites=self.num_sites,
            hidden_dims=cfg.nqs_hidden_dims,
        ).to(self.device)

        # Get reference state (HF state)
        self.reference_state = self.hamiltonian.get_hf_state()

    def _generate_essential_configs(self) -> torch.Tensor:
        """
        Generate essential configurations (HF + singles + doubles) without NF training.

        Reuses the same logic as PhysicsGuidedFlowTrainer._generate_essential_configs()
        but can be called standalone for Direct-CI mode.

        Returns:
            Tensor of essential configurations
        """
        from itertools import combinations

        n_orb = self.hamiltonian.n_orbitals
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        hf_state = self.hamiltonian.get_hf_state()
        essential = [hf_state.clone()]

        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Single excitations
        for i in occ_alpha:
            for a in virt_alpha:
                new_config = hf_state.clone()
                new_config[i] = 0
                new_config[a] = 1
                essential.append(new_config)

        for i in occ_beta:
            for a in virt_beta:
                new_config = hf_state.clone()
                new_config[i + n_orb] = 0
                new_config[a + n_orb] = 1
                essential.append(new_config)

        # Double excitations with proportional allocation per type.
        # αβ doubles dominate correlation energy, so each type gets a fair
        # share proportional to its total count, with αβ guaranteed >= 50%.
        max_doubles = 5000
        from math import comb as _comb

        n_aa_total = _comb(n_alpha, 2) * _comb(len(virt_alpha), 2)
        n_bb_total = _comb(n_beta, 2) * _comb(len(virt_beta), 2)
        n_ab_total = n_alpha * n_beta * len(virt_alpha) * len(virt_beta)
        total_possible = n_aa_total + n_bb_total + n_ab_total

        if total_possible <= max_doubles:
            max_aa = n_aa_total
            max_bb = n_bb_total
            max_ab = n_ab_total
        else:
            # Proportional allocation with αβ floor of 50%
            ab_frac = max(0.5, n_ab_total / total_possible if total_possible > 0 else 0.5)
            remaining_frac = 1.0 - ab_frac
            aa_frac = (
                remaining_frac * (n_aa_total / (n_aa_total + n_bb_total))
                if (n_aa_total + n_bb_total) > 0
                else 0
            )
            bb_frac = remaining_frac - aa_frac
            max_ab = int(ab_frac * max_doubles)
            max_aa = int(aa_frac * max_doubles)
            max_bb = max_doubles - max_ab - max_aa

        # Alpha-alpha doubles
        aa_count = 0
        for i, j in combinations(occ_alpha, 2):
            for a, b in combinations(virt_alpha, 2):
                if aa_count >= max_aa:
                    break
                new_config = hf_state.clone()
                new_config[i] = 0
                new_config[j] = 0
                new_config[a] = 1
                new_config[b] = 1
                essential.append(new_config)
                aa_count += 1
            if aa_count >= max_aa:
                break

        # Beta-beta doubles
        bb_count = 0
        for i, j in combinations(occ_beta, 2):
            for a, b in combinations(virt_beta, 2):
                if bb_count >= max_bb:
                    break
                new_config = hf_state.clone()
                new_config[i + n_orb] = 0
                new_config[j + n_orb] = 0
                new_config[a + n_orb] = 1
                new_config[b + n_orb] = 1
                essential.append(new_config)
                bb_count += 1
            if bb_count >= max_bb:
                break

        # Alpha-beta doubles (most important for correlation)
        ab_count = 0
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b in virt_beta:
                        if ab_count >= max_ab:
                            break
                        new_config = hf_state.clone()
                        new_config[i] = 0
                        new_config[j + n_orb] = 0
                        new_config[a] = 1
                        new_config[b + n_orb] = 1
                        essential.append(new_config)
                        ab_count += 1
                    if ab_count >= max_ab:
                        break
                if ab_count >= max_ab:
                    break
            if ab_count >= max_ab:
                break

        essential_tensor = torch.stack(essential).to(self.device)
        essential_tensor = torch.unique(essential_tensor, dim=0)

        hf_dev = hf_state.to(essential_tensor.device)
        n_singles = sum(1 for c in essential_tensor if torch.sum(torch.abs(c - hf_dev)) == 2)
        n_doubles = len(essential_tensor) - n_singles - 1

        print(
            f"Generated {len(essential_tensor)} essential configs: "
            f"1 HF + {n_singles} singles + {n_doubles} doubles"
        )

        return essential_tensor

    def train_flow_nqs(self, progress: bool = True) -> Dict[str, list]:
        """
        Stage 1: Train NF-NQS with physics-guided objective.

        If skip_nf_training is True (Direct-CI mode), generates essential
        configs directly without any NF training.
        """
        print("=" * 60)
        print("Stage 1: Physics-Guided NF-NQS Training")
        print("=" * 60)

        cfg = self.config

        # Direct-CI mode: skip NF training, use essential configs directly
        if cfg.skip_nf_training and self.is_molecular:
            print("Direct-CI mode: skipping NF-NQS training")
            print("Generating essential configurations (HF + singles + doubles)...")

            self._essential_configs = self._generate_essential_configs()
            self.results["training_history"] = {"energies": [], "skipped": True}
            self.results["nf_nqs_energy"] = None
            self.results["skip_nf_training"] = True

            return {"energies": [], "skipped": True}

        # Create physics-guided trainer
        train_config = PhysicsGuidedConfig(
            samples_per_batch=cfg.samples_per_batch,
            num_batches=cfg.num_batches,
            flow_lr=cfg.nf_lr,
            nqs_lr=cfg.nqs_lr,
            num_epochs=cfg.max_epochs,
            min_epochs=cfg.min_epochs,
            convergence_threshold=cfg.convergence_threshold,
            teacher_weight=cfg.teacher_weight,
            physics_weight=cfg.physics_weight,
            entropy_weight=cfg.entropy_weight,
            max_accumulated_basis=cfg.max_accumulated_basis,
            # Performance optimizations for large systems
            max_connections_per_config=cfg.max_connections_per_config,
            diagonal_only_warmup_epochs=cfg.diagonal_only_warmup_epochs,
            stochastic_connections_fraction=cfg.stochastic_connections_fraction,
        )

        self.trainer = PhysicsGuidedFlowTrainer(
            flow=self.flow,
            nqs=self.nqs,
            hamiltonian=self.hamiltonian,
            config=train_config,
            device=self.device,
        )

        history = self.trainer.train()

        self.results["training_history"] = history
        self.results["nf_nqs_energy"] = history["energies"][-1] if history.get("energies") else None

        return history

    def _run_vmc_training(self):
        """Stage 1b: VMC training to directly minimize variational energy.

        Uses REINFORCE on the autoregressive flow to minimize <psi|H|psi>.
        Only runs when ``use_vmc_training=True`` and the flow is autoregressive.
        """
        cfg = self.config

        if not isinstance(self.flow, AutoregressiveFlowSampler):
            print(
                "WARNING: VMC training requires AutoregressiveFlowSampler. "
                "Set use_autoregressive_flow=True. Skipping VMC."
            )
            self.results["vmc_energy"] = None
            return

        print("\n" + "=" * 60)
        print("Stage 1b: VMC Training (Variational Monte Carlo)")
        print("=" * 60)

        vmc_config = VMCConfig(
            n_samples=cfg.vmc_n_samples,
            n_steps=cfg.vmc_n_steps,
            lr=cfg.vmc_lr,
        )

        vmc_trainer = VMCTrainer(
            flow=self.flow,
            hamiltonian=self.hamiltonian,
            config=vmc_config,
            device=self.device,
            sign_network=self.sign_network,
        )

        vmc_results = vmc_trainer.train(verbose=True)

        self.results["vmc_energy"] = vmc_results["best_energy"]
        self.results["vmc_history"] = vmc_results
        print(f"VMC best energy: {vmc_results['best_energy']:.6f} Ha")
        print(f"VMC steps: {vmc_results['n_steps']}, converged: {vmc_results['converged']}")

    def extract_and_select_basis(self) -> torch.Tensor:
        """
        Stage 2: Extract basis with diversity-aware selection.

        In Direct-CI mode, uses essential configs directly without
        diversity selection (they are already a curated, physically-motivated set).
        """
        print("=" * 60)
        print("Stage 2: Diversity-Aware Basis Extraction")
        print("=" * 60)

        cfg = self.config

        # Direct-CI mode: use essential configs directly
        if cfg.skip_nf_training and hasattr(self, "_essential_configs"):
            print("Direct-CI mode: using essential configs as basis")
            selected_basis = self._essential_configs
            print(f"Essential configs basis: {len(selected_basis)} configs")
            self.nf_basis = selected_basis
            self.results["nf_basis_size"] = len(selected_basis)
            self.results["diversity_stats"] = {"skipped": True, "reason": "Direct-CI mode"}
            return selected_basis

        # Get accumulated basis from training
        if hasattr(self, "trainer") and self.trainer.accumulated_basis is not None:
            raw_basis = self.trainer.accumulated_basis
            print(f"Raw accumulated basis: {len(raw_basis)} configs")
        else:
            # Fallback: sample from flow
            with torch.no_grad():
                _, raw_basis = self.flow.sample(cfg.samples_per_batch * 5)
            print(f"Sampled basis: {len(raw_basis)} configs")

        # Verify particle conservation for molecular systems
        if self.is_molecular and cfg.use_particle_conserving_flow:
            n_orbitals = self.hamiltonian.n_orbitals
            valid, stats = verify_particle_conservation(
                raw_basis, n_orbitals, self.hamiltonian.n_alpha, self.hamiltonian.n_beta
            )
            if not valid:
                print(
                    f"WARNING: {stats['alpha_violations'] + stats['beta_violations']} "
                    f"particle number violations detected!"
                )
            else:
                print("All configurations satisfy particle conservation")

        # Apply diversity selection
        if cfg.use_diversity_selection:
            diversity_config = DiversityConfig(
                max_configs=cfg.max_diverse_configs,
                rank_2_fraction=cfg.rank_2_fraction,
            )

            selector = DiversitySelector(
                config=diversity_config,
                reference=self.reference_state,
                n_orbitals=(
                    self.hamiltonian.n_orbitals if self.is_molecular else self.num_sites // 2
                ),
            )

            selected_basis, select_stats = selector.select(raw_basis)
            print(f"Selected {len(selected_basis)} diverse configs from {len(raw_basis)}")
            print(f"Bucket distribution: {select_stats.get('bucket_stats', {})}")

            self.results["diversity_stats"] = select_stats
        else:
            selected_basis = raw_basis

        # CRITICAL: Always include essential configs (HF + singles + doubles)
        # even if diversity selection filtered them out. For large systems,
        # NF may never generate these, but they dominate the ground state.
        if (
            self.is_molecular
            and hasattr(self, "trainer")
            and hasattr(self.trainer, "_essential_configs")
            and self.trainer._essential_configs is not None
        ):
            essential = self.trainer._essential_configs
            combined = torch.cat([essential.to(selected_basis.device), selected_basis], dim=0)
            selected_basis = torch.unique(combined, dim=0)
            print(f"After merging essential configs: {len(selected_basis)} total")

        self.nf_basis = selected_basis
        self.results["nf_basis_size"] = len(selected_basis)

        return selected_basis

    def run_subspace_diag(self, progress: bool = True) -> Dict[str, Any]:
        """
        Stage 3: Subspace diagonalization via SKQD or SQD.

        Mode "skqd": Krylov time evolution expands the NF basis, then diagonalizes.
        Mode "sqd": SQD sampling-based batch diagonalization with self-consistent
                    configuration recovery (following the IBM quantum-centric paper).
        """
        cfg = self.config
        nf_basis = self.nf_basis

        if cfg.subspace_mode == "sqd":
            return self._run_sqd(nf_basis, progress)
        else:
            return self._run_skqd(nf_basis, progress)

    def _run_skqd(self, nf_basis: torch.Tensor, progress: bool = True) -> Dict[str, Any]:
        """Run SKQD (Krylov time evolution) subspace diagonalization."""
        print("=" * 60)
        print("Stage 3: Sample-Based Krylov Quantum Diagonalization (SKQD)")
        print("=" * 60)

        cfg = self.config

        # Check if SKQD is disabled
        if cfg.skip_skqd or cfg.max_krylov_dim <= 0:
            print("SKQD disabled, computing direct diagonalization...")
            return self._direct_diagonalize(nf_basis)

        # NNCI expansion: expand basis with NN-guided active learning (PR-B1)
        # before Krylov refinement. NNCI discovers important triples/quadruples
        # beyond the CISD basis, improving the starting point for SKQD.
        if cfg.use_nnci:
            print("Running NNCI basis expansion...")
            try:
                try:
                    from .krylov.nnci import NNCIConfig, NNCIActiveLearning
                except ImportError:
                    from krylov.nnci import NNCIConfig, NNCIActiveLearning

                nnci_config = NNCIConfig(
                    max_iterations=cfg.nnci_iterations,
                    top_k=min(50, cfg.nnci_candidates_per_iter),
                    max_candidates=cfg.nnci_candidates_per_iter,
                    max_excitation_rank=4,
                    training_epochs=100,
                    convergence_threshold=1e-6,
                    max_basis_size=min(cfg.max_diag_basis_size, 15000),
                )

                nnci = NNCIActiveLearning(
                    hamiltonian=self.hamiltonian,
                    initial_basis=nf_basis,
                    config=nnci_config,
                )
                nnci_results = nnci.run()

                expanded_basis = nnci_results["final_basis"]
                n_added = len(expanded_basis) - len(nf_basis)
                print(
                    f"NNCI added {n_added} configs "
                    f"(basis: {len(nf_basis)} -> {len(expanded_basis)})"
                )
                nf_basis = expanded_basis
                self.results["nnci_configs_added"] = n_added
                self.results["nnci_energy"] = nnci_results["energy"]
            except (RuntimeError, MemoryError, ValueError, ImportError) as e:
                import traceback

                print(f"NNCI expansion failed: {e}")
                traceback.print_exc()
                self.results["nnci_configs_added"] = 0
        else:
            self.results["nnci_configs_added"] = 0

        # Configure SKQD
        skqd_config = SKQDConfig(
            max_krylov_dim=cfg.max_krylov_dim,
            time_step=cfg.time_step,
            shots_per_krylov=cfg.shots_per_krylov,
            use_gpu=(self.device == "cuda"),
            regularization=getattr(cfg, "skqd_regularization", 1e-8),
            max_diag_basis_size=cfg.max_diag_basis_size,
            convergence_threshold=getattr(cfg, "skqd_convergence_threshold", 1e-5),
            adaptive_dt=getattr(cfg, "skqd_adaptive_dt", False),
        )

        skqd = FlowGuidedSKQD(
            hamiltonian=self.hamiltonian,
            nf_basis=nf_basis,
            config=skqd_config,
        )

        results = skqd.run_with_nf(progress=progress)

        # Use best stable energy from SKQD
        skqd_energy = results.get("best_stable_energy", results["energies_combined"][-1])

        self.results["skqd_results"] = results
        self.results["skqd_energy"] = skqd_energy

        # Variational consistency check
        if self.exact_energy is not None and skqd_energy < self.exact_energy - 0.001:
            print(
                f"WARNING: SKQD energy ({skqd_energy:.6f}) below exact ({self.exact_energy:.6f})!"
            )
            print("Numerical instability detected. Falling back to direct diagonalization.")
            return self._direct_diagonalize(nf_basis)

        self.results["combined_energy"] = skqd_energy
        return results

    def _run_sqd(self, nf_basis: torch.Tensor, progress: bool = True) -> Dict[str, Any]:
        """Run SQD (sampling-based) subspace diagonalization."""
        cfg = self.config

        # Enable config recovery when noise injection is active
        # (emulating quantum hardware depolarizing noise for SQD-Recovery mode)
        enable_recovery = cfg.sqd_noise_rate > 0

        print("=" * 60)
        sqd_mode = "Recovery" if enable_recovery else "Clean"
        print(f"Stage 3: Sample-Based Quantum Diagonalization (SQD-{sqd_mode})")
        print("=" * 60)

        sqd_config = SQDConfig(
            num_batches=cfg.sqd_num_batches,
            batch_size=cfg.sqd_batch_size,
            self_consistent_iters=cfg.sqd_self_consistent_iters,
            spin_penalty=cfg.sqd_spin_penalty,
            noise_rate=cfg.sqd_noise_rate,
            enable_config_recovery=enable_recovery,
            use_spin_symmetry_enhancement=cfg.sqd_use_spin_symmetry,
        )

        solver = SQDSolver(
            hamiltonian=self.hamiltonian,
            config=sqd_config,
        )

        results = solver.run(nf_basis)

        sqd_energy = results["energy"]

        self.results["sqd_results"] = results
        self.results["sqd_energy"] = sqd_energy

        # Variational consistency check
        if self.exact_energy is not None and sqd_energy < self.exact_energy - 0.001:
            print(f"WARNING: SQD energy ({sqd_energy:.6f}) below exact ({self.exact_energy:.6f})!")
            print("Numerical instability detected. Falling back to direct diagonalization.")
            return self._direct_diagonalize(nf_basis)

        self.results["combined_energy"] = sqd_energy
        return results

    def _direct_diagonalize(self, basis: torch.Tensor) -> Dict[str, Any]:
        """Compute energy by direct diagonalization of basis."""
        try:
            from utils.memory_logger import log_allocation
        except ImportError:
            try:
                from src.utils.memory_logger import log_allocation
            except ImportError:
                log_allocation = None

        print("Computing energy via direct diagonalization...")
        if log_allocation:
            log_allocation("_direct_diagonalize", len(basis), dtype="float64", layout="dense")
        H_matrix = self.hamiltonian.matrix_elements(basis, basis)
        H_np = H_matrix.detach().cpu().numpy()
        H_np = 0.5 * (H_np + H_np.T)
        eigenvalues, _ = np.linalg.eigh(H_np)
        energy = float(eigenvalues[0])
        print(f"Direct diag energy: {energy:.8f} Ha ({len(basis)} configs)")
        self.results["combined_energy"] = energy
        return {"energies_combined": [energy], "direct_diag": True}

    def run(self, progress: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline.

        Args:
            progress: Show progress bars

        Returns:
            Complete results dictionary with energies and statistics
        """
        print("\n" + "=" * 60)
        print("Flow-Guided Krylov Pipeline")
        print("=" * 60)
        print(f"System: {self.num_sites} sites")
        print(f"Device: {self.device}")
        if self.is_molecular:
            print(f"Electrons: {self.hamiltonian.n_alpha}α + {self.hamiltonian.n_beta}β")
        if self.exact_energy is not None:
            print(f"Exact energy: {self.exact_energy:.8f}")
        print("=" * 60 + "\n")

        # Stage 1: Training
        self.train_flow_nqs(progress=progress)

        # Stage 1b (optional): VMC training
        if self.config.use_vmc_training:
            self._run_vmc_training()

        # Stage 2: Basis extraction
        self.extract_and_select_basis()

        # Stage 3: Subspace diagonalization (SKQD or SQD)
        self.run_subspace_diag(progress=progress)

        # Summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print results summary."""
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        if self.results.get("skip_nf_training"):
            print(f"Mode:              Direct-CI (NF training skipped)")

        mode_label = self.config.subspace_mode.upper()
        if self.config.subspace_mode == "sqd":
            if self.config.sqd_noise_rate > 0:
                mode_label += f" (Recovery, noise={self.config.sqd_noise_rate:.2f})"
            else:
                mode_label += " (Clean)"
        print(f"Subspace mode:     {mode_label}")

        if "nf_nqs_energy" in self.results and self.results["nf_nqs_energy"] is not None:
            print(f"NF-NQS Energy:     {self.results['nf_nqs_energy']:.8f}")

        if "nf_basis_size" in self.results:
            print(f"NF Basis Size:     {self.results['nf_basis_size']}")

        if "combined_energy" in self.results:
            print(f"Final Energy:      {self.results['combined_energy']:.8f}")

        if self.exact_energy is not None:
            best_energy = self.results.get(
                "combined_energy",
                self.results.get(
                    "skqd_energy", self.results.get("sqd_energy", self.results.get("nf_nqs_energy"))
                ),
            )
            if best_energy is None:
                print("\nError: N/A (energy not computed)")
            else:
                error_ha = abs(best_energy - self.exact_energy)
                error_mha = error_ha * 1000
                error_kcal = error_ha * 627.5
                print(f"\nError: {error_mha:.4f} mHa ({error_kcal:.4f} kcal/mol)")

                if error_kcal < 1.0:
                    print("Chemical accuracy: PASS")
                else:
                    print("Chemical accuracy: FAIL")

        print("=" * 60)


def run_molecular_benchmark(
    molecule: str = "lih",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run pipeline on a molecular system.

    Args:
        molecule: "h2", "lih", "h2o", "beh2", "nh3", "n2", or "ch4"
        verbose: Print progress

    Returns:
        Results dictionary
    """
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
        create_beh2_hamiltonian,
        create_nh3_hamiltonian,
        create_n2_hamiltonian,
        create_ch4_hamiltonian,
    )

    # Create Hamiltonian
    molecule_lower = molecule.lower()
    if molecule_lower == "h2":
        H = create_h2_hamiltonian(bond_length=0.74)
    elif molecule_lower == "lih":
        H = create_lih_hamiltonian(bond_length=1.6)
    elif molecule_lower == "h2o":
        H = create_h2o_hamiltonian()
    elif molecule_lower == "beh2":
        H = create_beh2_hamiltonian()
    elif molecule_lower == "nh3":
        H = create_nh3_hamiltonian()
    elif molecule_lower == "n2":
        H = create_n2_hamiltonian()
    elif molecule_lower == "ch4":
        H = create_ch4_hamiltonian()
    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    # Get exact energy
    E_exact = H.fci_energy()

    # Configure pipeline (Direct-CI mode by default for molecular systems)
    config = PipelineConfig(
        use_particle_conserving_flow=True,
        use_diversity_selection=True,
        skip_nf_training=True,
    )

    # Run pipeline
    pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)
    results = pipeline.run(progress=verbose)

    return results
