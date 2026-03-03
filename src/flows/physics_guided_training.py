"""
Physics-Guided Training for Normalizing Flows.

This module implements mixed-objective training that combines:
1. Teacher signal: Match NQS probability distribution (standard approach)
2. Physics signal: Energy-weighted importance (not just probability)
3. Exploration bonus: Entropy regularization to avoid collapse

The key insight is that NF should learn to sample configurations that:
- Have high NQS probability (teacher)
- Have LOW local energy (physics - ground state has lowest energy)
- Maintain diversity (exploration)

References:
- NF-NQS paper: "Improved Ground State Estimation via NF-Assisted NQS"
- Importance sampling: configurations with low local energy matter more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm

# Enable TensorFloat32 for better performance on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Connection cache for avoiding recomputation
try:
    from ..utils.connection_cache import ConnectionCache
except ImportError:
    from utils.connection_cache import ConnectionCache


@dataclass
class PhysicsGuidedConfig:
    """Configuration for physics-guided flow training."""

    # Batch sizes
    samples_per_batch: int = 2000
    num_batches: int = 1
    nqs_chunk_size: int = 16384  # Increased chunk size for better GPU saturation

    # Learning rates
    flow_lr: float = 5e-4
    nqs_lr: float = 1e-3

    # Training epochs
    num_epochs: int = 500
    min_epochs: int = 150
    convergence_threshold: float = 0.20

    # Loss weights - following paper's approach
    # Paper uses only cross-entropy for NF, weighted by |E|
    # Set physics_weight=0 and entropy_weight=0 to match paper exactly
    teacher_weight: float = 1.0  # Cross-entropy (paper's only term)
    physics_weight: float = 0.0  # Paper doesn't use this
    entropy_weight: float = 0.0  # Paper doesn't use this

    # Energy baseline for physics signal
    use_energy_baseline: bool = True  # Subtract baseline for variance reduction

    # Accumulated basis for energy computation
    use_accumulated_energy: bool = True
    max_accumulated_basis: int = 2048
    accumulated_energy_interval: int = 50  # Increased to reduce overhead
    prune_basis_threshold: float = 1e-6

    # EMA for stable tracking
    ema_decay: float = 0.95

    # Temperature annealing for particle-conserving flow
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_decay_epochs: int = 200

    # Connection caching for avoiding recomputation
    use_connection_cache: bool = True
    max_cache_size: int = 100000  # Max cached configurations
    cache_warmup: bool = True  # Pre-populate cache with HF neighborhood
    cache_warmup_excitation_level: int = 2  # Include singles (1) and doubles (2)

    # torch.compile() for faster NQS evaluation
    use_torch_compile: bool = True

    # Parallel connection computation
    use_parallel_connections: bool = True
    parallel_workers: int = 8  # Number of parallel workers for connection computation

    # Early stopping on energy plateau
    early_stopping_patience: int = 100  # Stop if energy doesn't improve for N epochs
    early_stopping_threshold: float = 0.001  # Minimum improvement (Ha) to reset patience

    # === PERFORMANCE OPTIMIZATIONS FOR LARGE SYSTEMS ===
    # These options dramatically reduce training time for large molecules (>20 qubits)

    # Truncated connections: Only use top-k connections by matrix element magnitude
    # For C2H4 (28 qubits): reduces from ~3000 to ~200 connections per config
    # Set to 0 to disable truncation (use all connections)
    max_connections_per_config: int = 0

    # Diagonal-only warmup: Skip expensive off-diagonal computation for first N epochs
    # During warmup, only diagonal energies are used (much faster)
    # Recommended: 50-100 epochs for large systems
    diagonal_only_warmup_epochs: int = 0

    # Stochastic local energy: Sample a fraction of connections instead of using all
    # Provides unbiased energy estimate with reduced variance
    # Set to 1.0 to use all connections, 0.1 to use 10%
    stochastic_connections_fraction: float = 1.0

    # === PAPER-ALIGNED ENERGY COMPUTATION ===
    # Use subspace diagonalization for energy (paper's method) instead of local energy
    # Paper Section 3.2-3.3: "Energy is computed by diagonalizing H restricted to S"
    # This fixes the training-vs-diagonalization energy gap
    use_subspace_energy: bool = True

    # Maximum basis size for subspace diagonalization (for memory efficiency)
    # If basis exceeds this, use top-N configs by NQS probability
    max_subspace_diag_size: int = 2048

    # Compute subspace energy every N epochs (for performance on large systems)
    # Set to 1 to compute every epoch (most accurate but slower)
    # Set to 5-10 for faster training with periodic energy updates
    subspace_energy_interval: int = 1

    # === ESSENTIAL CONFIGURATION INJECTION ===
    # For molecular systems, always inject HF + low-excitation determinants
    # into the sampled basis to ensure proper ground state discovery.
    # Without this, NF may explore wrong region of Hilbert space.
    inject_essential_configs: bool = True

    # Include HF state in every subspace energy computation
    always_include_hf: bool = True

    # Include single excitations (important for orbital relaxation)
    include_singles_in_basis: bool = True

    # Include double excitations (most important for electron correlation)
    include_doubles_in_basis: bool = True


class PhysicsGuidedFlowTrainer:
    """
    Trainer with mixed-objective loss for normalizing flows.

    The training objective combines three signals:
    1. Teacher (KL divergence): Flow matches NQS probability
    2. Physics (energy importance): Flow favors low-energy configurations
    3. Exploration (entropy): Flow maintains sampling diversity

    Loss = w_teacher * L_teacher + w_physics * L_physics - w_entropy * H(flow)
    """

    def __init__(
        self,
        flow: nn.Module,
        nqs: nn.Module,
        hamiltonian: Any,
        config: PhysicsGuidedConfig,
        device: str = "cuda",
    ):
        self.flow = flow
        self.nqs = nqs
        self.hamiltonian = hamiltonian
        self.config = config
        self.device = device

        # Optimizers
        self.flow_optimizer = torch.optim.AdamW(
            flow.parameters(), lr=config.flow_lr, weight_decay=1e-5
        )
        self.nqs_optimizer = torch.optim.AdamW(
            nqs.parameters(), lr=config.nqs_lr, weight_decay=1e-5
        )

        # Schedulers
        self.flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.flow_optimizer, T_max=config.num_epochs, eta_min=1e-6
        )
        self.nqs_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.nqs_optimizer, T_max=config.num_epochs, eta_min=1e-6
        )

        # Accumulated basis
        self.accumulated_basis = None

        # Connection cache for avoiding recomputation of Hamiltonian connections
        self.connection_cache = None
        if config.use_connection_cache:
            num_sites = hamiltonian.num_sites
            self.connection_cache = ConnectionCache(
                num_sites=num_sites,
                max_cache_size=config.max_cache_size,
                device=device
            )

        # torch.compile is disabled for NQS forward passes due to incompatibility
        # with the encode_configuration method's dynamic control flow and the
        # varying batch sizes used during local energy computation.
        # The overhead of recompilation negates any performance benefits.
        self._nqs_compiled = None

        # Tracking
        self.energy_ema = None
        self.history = {
            'energies': [],
            'accumulated_energies': [],
            'teacher_losses': [],
            'physics_losses': [],
            'entropy_values': [],
            'unique_ratios': [],
            'basis_sizes': [],
            'cache_hit_rates': [],
        }

        # Early stopping tracking
        self._best_energy = float('inf')
        self._patience_counter = 0

        # Essential configurations (HF + singles + doubles) for molecular systems
        self._essential_configs = None
        if config.inject_essential_configs and hasattr(hamiltonian, 'n_alpha'):
            self._essential_configs = self._generate_essential_configs()

    def _generate_essential_configs(self) -> torch.Tensor:
        """
        Generate essential configurations: HF + singles + doubles.

        These configurations are CRITICAL for accurate ground state energy:
        - HF: Has largest coefficient (~0.9) in ground state
        - Singles: Important for orbital relaxation
        - Doubles: Capture electron correlation (most important for chemical accuracy)

        Without these, NF may explore wrong region of Hilbert space and
        the subspace energy signal won't guide NF toward ground state.

        Returns:
            Tensor of essential configurations
        """
        config = self.config
        n_orb = self.hamiltonian.n_orbitals
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        # Get HF state
        hf_state = self.hamiltonian.get_hf_state()
        essential = [hf_state.clone()]

        # Get occupied and virtual orbitals
        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Add single excitations
        if config.include_singles_in_basis:
            # Alpha singles
            for i in occ_alpha:
                for a in virt_alpha:
                    new_config = hf_state.clone()
                    new_config[i] = 0
                    new_config[a] = 1
                    essential.append(new_config)

            # Beta singles
            for i in occ_beta:
                for a in virt_beta:
                    new_config = hf_state.clone()
                    new_config[i + n_orb] = 0
                    new_config[a + n_orb] = 1
                    essential.append(new_config)

        # Add double excitations
        if config.include_doubles_in_basis:
            from itertools import combinations

            # Limit doubles for very large systems (avoid memory issues)
            max_doubles = 5000

            doubles_count = 0

            # Alpha-alpha doubles
            for i, j in combinations(occ_alpha, 2):
                for a, b in combinations(virt_alpha, 2):
                    if doubles_count >= max_doubles:
                        break
                    new_config = hf_state.clone()
                    new_config[i] = 0
                    new_config[j] = 0
                    new_config[a] = 1
                    new_config[b] = 1
                    essential.append(new_config)
                    doubles_count += 1
                if doubles_count >= max_doubles:
                    break

            # Beta-beta doubles
            for i, j in combinations(occ_beta, 2):
                for a, b in combinations(virt_beta, 2):
                    if doubles_count >= max_doubles:
                        break
                    new_config = hf_state.clone()
                    new_config[i + n_orb] = 0
                    new_config[j + n_orb] = 0
                    new_config[a + n_orb] = 1
                    new_config[b + n_orb] = 1
                    essential.append(new_config)
                    doubles_count += 1
                if doubles_count >= max_doubles:
                    break

            # Alpha-beta doubles (most important for correlation)
            for i in occ_alpha:
                for j in occ_beta:
                    for a in virt_alpha:
                        for b in virt_beta:
                            if doubles_count >= max_doubles:
                                break
                            new_config = hf_state.clone()
                            new_config[i] = 0
                            new_config[j + n_orb] = 0
                            new_config[a] = 1
                            new_config[b + n_orb] = 1
                            essential.append(new_config)
                            doubles_count += 1
                        if doubles_count >= max_doubles:
                            break
                    if doubles_count >= max_doubles:
                        break
                if doubles_count >= max_doubles:
                    break

        # Stack and remove duplicates
        essential_tensor = torch.stack(essential).to(self.device)
        essential_tensor = torch.unique(essential_tensor, dim=0)

        n_singles = len([c for c in essential if torch.sum(torch.abs(c - hf_state)) == 2])
        n_doubles = len(essential_tensor) - n_singles - 1

        print(f"Generated {len(essential_tensor)} essential configs: "
              f"1 HF + {n_singles} singles + {n_doubles} doubles")

        return essential_tensor

    def _warmup_cache_with_hf_neighborhood(self):
        """
        Pre-populate the connection cache with HF neighborhood configurations.

        This dramatically improves cache hit rate because:
        1. HF and nearby excitations are the most physically important configs
        2. During early training, the flow explores configs near HF
        3. Pre-computing these avoids expensive on-the-fly computation

        For C2H4 (14 orbitals, 8 alpha, 8 beta):
        - Singles: 2 * n_occ * n_virt = 2 * 8 * 6 = 96 configs
        - Doubles: C(8,2)*C(6,2)*2 + 8*8*6*6 = 840 + 2304 = 3144 configs
        - Total: ~3240 important configs pre-cached
        """
        if self.connection_cache is None:
            return

        if not hasattr(self.hamiltonian, 'n_alpha'):
            # Not a molecular Hamiltonian
            return

        config = self.config
        n_orb = self.hamiltonian.n_orbitals
        n_alpha = self.hamiltonian.n_alpha
        n_beta = self.hamiltonian.n_beta

        print("Warming up connection cache with HF neighborhood...")

        # Get HF state
        hf_state = self.hamiltonian.get_hf_state()

        # Collect configurations to pre-cache
        configs_to_cache = [hf_state]

        # Get occupied and virtual orbitals
        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Add single excitations
        if config.cache_warmup_excitation_level >= 1:
            # Alpha singles
            for i in occ_alpha:
                for a in virt_alpha:
                    new_config = hf_state.clone()
                    new_config[i] = 0
                    new_config[a] = 1
                    configs_to_cache.append(new_config)

            # Beta singles
            for i in occ_beta:
                for a in virt_beta:
                    new_config = hf_state.clone()
                    new_config[i + n_orb] = 0
                    new_config[a + n_orb] = 1
                    configs_to_cache.append(new_config)

        # Add double excitations
        if config.cache_warmup_excitation_level >= 2:
            from itertools import combinations

            # Alpha-alpha doubles
            for i, j in combinations(occ_alpha, 2):
                for a, b in combinations(virt_alpha, 2):
                    new_config = hf_state.clone()
                    new_config[i] = 0
                    new_config[j] = 0
                    new_config[a] = 1
                    new_config[b] = 1
                    configs_to_cache.append(new_config)

            # Beta-beta doubles
            for i, j in combinations(occ_beta, 2):
                for a, b in combinations(virt_beta, 2):
                    new_config = hf_state.clone()
                    new_config[i + n_orb] = 0
                    new_config[j + n_orb] = 0
                    new_config[a + n_orb] = 1
                    new_config[b + n_orb] = 1
                    configs_to_cache.append(new_config)

            # Alpha-beta doubles (most numerous)
            for i in occ_alpha:
                for j in occ_beta:
                    for a in virt_alpha:
                        for b in virt_beta:
                            new_config = hf_state.clone()
                            new_config[i] = 0
                            new_config[j + n_orb] = 0
                            new_config[a] = 1
                            new_config[b + n_orb] = 1
                            configs_to_cache.append(new_config)

        # Stack and remove duplicates
        configs_tensor = torch.stack(configs_to_cache).to(self.device)
        configs_tensor = torch.unique(configs_tensor, dim=0)

        n_warmup = len(configs_tensor)
        print(f"  Pre-caching {n_warmup} HF neighborhood configurations...")

        # Cache connections for all configs (in batches to avoid memory issues)
        batch_size = 500
        cached = 0

        for start in range(0, n_warmup, batch_size):
            end = min(start + batch_size, n_warmup)
            batch = configs_tensor[start:end]

            for cfg in batch:
                connected, elements = self.hamiltonian.get_connections(cfg)
                self.connection_cache.put(cfg, connected, elements)
                cached += 1

        print(f"  Cached {cached} configurations ({self.connection_cache.stats()['size']} entries)")

    def train(self) -> Dict[str, list]:
        """Run physics-guided training loop."""
        config = self.config

        print(f"Starting physics-guided NF-NQS training")
        print(f"  Teacher weight: {config.teacher_weight}")
        print(f"  Physics weight: {config.physics_weight}")
        print(f"  Entropy weight: {config.entropy_weight}")
        if config.use_connection_cache:
            print(f"  Connection cache: enabled (max {config.max_cache_size} entries)")

        # Warm up cache with HF neighborhood before training
        if config.use_connection_cache and config.cache_warmup:
            self._warmup_cache_with_hf_neighborhood()

        pbar = tqdm(range(config.num_epochs), desc="Training")

        for epoch in pbar:
            # Temperature annealing for particle-conserving flow
            if hasattr(self.flow, 'set_temperature'):
                progress = min(1.0, epoch / config.temperature_decay_epochs)
                temperature = config.initial_temperature + progress * (
                    config.final_temperature - config.initial_temperature
                )
                self.flow.set_temperature(temperature)

            # Training step
            metrics = self._train_epoch(epoch)

            # Update history
            self.history['energies'].append(metrics['energy'])
            self.history['teacher_losses'].append(metrics['teacher_loss'])
            self.history['physics_losses'].append(metrics['physics_loss'])
            self.history['entropy_values'].append(metrics['entropy'])
            self.history['unique_ratios'].append(metrics['unique_ratio'])

            if 'accumulated_energy' in metrics:
                self.history['accumulated_energies'].append(metrics['accumulated_energy'])
            if self.accumulated_basis is not None:
                self.history['basis_sizes'].append(len(self.accumulated_basis))

            # Track cache hit rate
            cache_hit_rate = 0.0
            if self.connection_cache is not None:
                cache_hit_rate = self.connection_cache.hit_rate
                self.history['cache_hit_rates'].append(cache_hit_rate)

            # Update schedulers
            self.flow_scheduler.step()
            self.nqs_scheduler.step()

            # Progress bar update with cache hit rate
            postfix = {
                'E': f"{metrics['energy']:.4f}",
                'unique': f"{metrics['unique_ratio']:.2f}",
                'T_loss': f"{metrics['teacher_loss']:.4f}",
            }
            if self.connection_cache is not None:
                postfix['cache'] = f"{cache_hit_rate:.0%}"
            pbar.set_postfix(postfix)

            # Check convergence
            if epoch >= config.min_epochs:
                if metrics['unique_ratio'] < config.convergence_threshold:
                    print(f"\nConverged at epoch {epoch}: unique_ratio={metrics['unique_ratio']:.3f}")
                    if self.connection_cache is not None:
                        stats = self.connection_cache.stats()
                        print(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
                              f"{stats['hit_rate']:.1%} hit rate, {stats['size']} entries")
                    break

            # Early stopping on energy plateau
            current_energy = metrics['energy']
            if current_energy < self._best_energy - config.early_stopping_threshold:
                self._best_energy = current_energy
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            if self._patience_counter >= config.early_stopping_patience and epoch >= config.min_epochs:
                print(f"\nEarly stopping at epoch {epoch}: no energy improvement for "
                      f"{config.early_stopping_patience} epochs (best: {self._best_energy:.6f} Ha)")
                if self.connection_cache is not None:
                    stats = self.connection_cache.stats()
                    print(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
                          f"{stats['hit_rate']:.1%} hit rate, {stats['size']} entries")
                break

        # Print final cache stats
        if self.connection_cache is not None:
            stats = self.connection_cache.stats()
            print(f"\nFinal cache stats: {stats['hits']} hits, {stats['misses']} misses, "
                  f"{stats['hit_rate']:.1%} hit rate, {stats['size']} entries")

        return self.history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        config = self.config
        self.flow.train()
        self.nqs.train()

        total_metrics = {
            'energy': 0.0,
            'teacher_loss': 0.0,
            'physics_loss': 0.0,
            'entropy': 0.0,
            'unique_ratio': 0.0,
        }

        for batch_idx in range(config.num_batches):
            # Sample from flow
            if hasattr(self.flow, 'sample_with_probs'):
                configs, log_probs, unique_configs = self.flow.sample_with_probs(
                    config.samples_per_batch
                )
            else:
                log_probs, unique_configs = self.flow.sample(config.samples_per_batch)
                configs = unique_configs

            n_unique = len(unique_configs)
            unique_ratio = n_unique / config.samples_per_batch

            # Compute NQS probabilities
            with torch.no_grad():
                nqs_log_amp = self.nqs.log_amplitude(unique_configs.float())
                nqs_probs = torch.exp(2 * nqs_log_amp)  # |psi|^2 = exp(2*log|psi|)
                nqs_probs = nqs_probs / nqs_probs.sum()

            # Compute energy using paper's subspace diagonalization method
            # This is the CORRECT approach per paper Section 3.2-3.3
            if config.use_subspace_energy:
                # OPTIMIZATION: Compute subspace energy at specified interval
                # (reduces O(n³) diagonalization overhead for large systems)
                compute_subspace_this_epoch = (
                    epoch % config.subspace_energy_interval == 0 or
                    epoch == 0 or  # Always compute first epoch
                    not hasattr(self, '_cached_subspace_energy')
                )

                if compute_subspace_this_epoch:
                    energy = self._compute_subspace_energy(unique_configs, nqs_probs)
                    self._cached_subspace_energy = energy
                else:
                    energy = self._cached_subspace_energy

                # Still compute local energies for REINFORCE gradient (NQS training)
                use_diagonal_only = (epoch < config.diagonal_only_warmup_epochs)
                local_energies = self._compute_local_energies(
                    unique_configs,
                    nqs_chunk_size=config.nqs_chunk_size,
                    diagonal_only=use_diagonal_only,
                )
            else:
                # Legacy mode: use local energy (not paper's approach)
                use_diagonal_only = (epoch < config.diagonal_only_warmup_epochs)
                local_energies = self._compute_local_energies(
                    unique_configs,
                    nqs_chunk_size=config.nqs_chunk_size,
                    diagonal_only=use_diagonal_only,
                )
                energy = (local_energies * nqs_probs).sum()

            # Update accumulated basis
            self._update_accumulated_basis(unique_configs)

            # Compute flow loss with mixed objectives
            flow_loss, loss_components = self._compute_flow_loss(
                configs, unique_configs, nqs_probs, local_energies, energy
            )

            # NQS loss (minimize energy)
            nqs_loss = self._compute_nqs_loss(unique_configs, nqs_probs, local_energies)

            # Backward pass
            self.flow_optimizer.zero_grad()
            self.nqs_optimizer.zero_grad()

            flow_loss.backward(retain_graph=True)
            nqs_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.nqs.parameters(), max_norm=1.0)

            self.flow_optimizer.step()
            self.nqs_optimizer.step()

            # Accumulate metrics
            total_metrics['energy'] += energy.item()
            total_metrics['teacher_loss'] += loss_components['teacher'].item()
            total_metrics['physics_loss'] += loss_components['physics'].item()
            total_metrics['entropy'] += loss_components['entropy'].item()
            total_metrics['unique_ratio'] += unique_ratio

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= config.num_batches

        # Update EMA
        if self.energy_ema is None:
            self.energy_ema = total_metrics['energy']
        else:
            self.energy_ema = (config.ema_decay * self.energy_ema +
                             (1 - config.ema_decay) * total_metrics['energy'])
        total_metrics['energy_ema'] = self.energy_ema

        # Compute accumulated energy periodically
        if (config.use_accumulated_energy and
            epoch % config.accumulated_energy_interval == 0 and
            self.accumulated_basis is not None):
            acc_energy = self._compute_accumulated_energy()
            total_metrics['accumulated_energy'] = acc_energy

        return total_metrics

    def _compute_subspace_energy(
        self,
        configs: torch.Tensor,
        nqs_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy by diagonalizing H in sampled subspace (paper's method).

        From paper Section 3.2-3.3:
        "Energy is computed by diagonalizing H restricted to the sampled subspace S"

        CRITICAL FIX: Always include essential configurations (HF + singles + doubles)
        in the subspace. Without this, the NF may explore a region of Hilbert space
        that doesn't include HF, leading to a poor energy signal that doesn't guide
        the NF toward the ground state.

        Args:
            configs: (n_configs, num_sites) sampled configurations
            nqs_probs: (n_configs,) NQS probabilities for each config

        Returns:
            Ground state energy in the subspace (scalar tensor for gradient flow)
        """
        config = self.config
        n_configs = len(configs)

        # CRITICAL: Merge with essential configs (HF + singles + doubles)
        # This ensures the subspace always contains the ground state region
        if self._essential_configs is not None:
            configs = torch.cat([self._essential_configs, configs], dim=0)
            configs = torch.unique(configs, dim=0)
            n_configs = len(configs)

        # For large bases, select top-N by NQS probability to keep computation tractable
        # But ALWAYS keep essential configs (HF + singles + doubles)
        if n_configs > config.max_subspace_diag_size:
            if self._essential_configs is not None:
                n_essential = len(self._essential_configs)

                if n_essential >= config.max_subspace_diag_size:
                    # Essential configs alone exceed limit - select most important ones
                    # Always keep HF (index 0), then select by diagonal energy
                    with torch.no_grad():
                        diag_energies = self.hamiltonian.diagonal_elements_batch(
                            self._essential_configs
                        )
                    # HF is always index 0 - keep it. Select top by lowest diagonal energy.
                    n_select = config.max_subspace_diag_size - 1
                    _, top_idx = torch.topk(diag_energies[1:], min(n_select, len(diag_energies) - 1), largest=False)
                    selected_essential = torch.cat([
                        self._essential_configs[:1],  # HF
                        self._essential_configs[1:][top_idx],
                    ], dim=0)
                    configs = selected_essential
                else:
                    # Essential configs fit within limit - add best NF configs for the remainder
                    max_sampled = config.max_subspace_diag_size - n_essential

                    # Compute NQS probs for all configs to select best sampled ones
                    with torch.no_grad():
                        all_log_amp = self.nqs.log_amplitude(configs.float())
                        all_probs = torch.exp(2 * all_log_amp)

                    # Identify which configs are NOT in essential set using integer encoding
                    n_sites = configs.shape[1]
                    powers = (2 ** torch.arange(n_sites, device=configs.device, dtype=torch.long)).flip(0)
                    all_ints = (configs.long() * powers).sum(dim=1)
                    ess_ints = (self._essential_configs.long() * powers).sum(dim=1)
                    ess_set = set(ess_ints.cpu().tolist())

                    # Mask for non-essential configs
                    non_ess_mask = torch.tensor(
                        [int(x) not in ess_set for x in all_ints.cpu().tolist()],
                        device=configs.device, dtype=torch.bool,
                    )

                    if non_ess_mask.any() and max_sampled > 0:
                        non_ess_probs = all_probs[non_ess_mask]
                        non_ess_configs = configs[non_ess_mask]
                        k = min(max_sampled, len(non_ess_probs))
                        _, top_indices = torch.topk(non_ess_probs, k)
                        sampled_configs = non_ess_configs[top_indices]
                        configs = torch.cat([self._essential_configs, sampled_configs], dim=0)
                        configs = torch.unique(configs, dim=0)
                    else:
                        configs = self._essential_configs
            else:
                _, top_indices = torch.topk(nqs_probs, config.max_subspace_diag_size)
                configs = configs[top_indices]

            n_configs = len(configs)

        with torch.no_grad():
            # Build subspace Hamiltonian H_ij = <x_i|H|x_j>
            H_subspace = self.hamiltonian.matrix_elements(configs, configs)

            # Ensure Hermitian symmetry
            H_subspace = 0.5 * (H_subspace + H_subspace.T)

            # Use float64 for numerical stability
            H_sub = H_subspace.double()

            # Diagonalize to get ground state energy
            eigenvalues = torch.linalg.eigvalsh(H_sub)
            E_ground = eigenvalues[0]

        # Return as tensor that can flow through gradient computation
        # (even though we detach it, keeping consistent tensor type helps)
        return E_ground.float()

    def _compute_local_energies(
        self,
        configs: torch.Tensor,
        nqs_chunk_size: int = 16384,
        diagonal_only: bool = False,
    ) -> torch.Tensor:
        """
        Compute local energies E_loc(x) = <x|H|psi>/<x|psi>.

        Optimized version with multiple acceleration options:
        1. diagonal_only: Skip expensive off-diagonal computation (for warmup)
        2. max_connections_per_config: Truncate to top-k connections by magnitude
        3. stochastic_connections_fraction: Sample a fraction of connections

        Args:
            configs: (n_configs, num_sites) basis configurations
            nqs_chunk_size: Maximum batch size for NQS evaluation (default 16384)
            diagonal_only: If True, only compute diagonal elements (fast warmup mode)

        Returns:
            (n_configs,) local energies
        """
        n_configs = len(configs)
        config = self.config

        with torch.no_grad():
            # Step 1: Get diagonal elements (already vectorized and efficient)
            diag = self.hamiltonian.diagonal_elements_batch(configs)

            # Fast path: diagonal-only mode for warmup
            if diagonal_only:
                return diag

            # Step 2: Get ALL connections
            all_connected, all_elements, all_orig_indices = self._get_connections_batch(configs)

            # If no off-diagonal connections, return diagonal energies
            if len(all_connected) == 0:
                return diag

            # Step 3: Apply truncation if configured
            if config.max_connections_per_config > 0:
                all_connected, all_elements, all_orig_indices = self._truncate_connections(
                    all_connected, all_elements, all_orig_indices,
                    n_configs, config.max_connections_per_config
                )

            # Step 4: Apply stochastic sampling if configured
            if config.stochastic_connections_fraction < 1.0:
                all_connected, all_elements, all_orig_indices = self._sample_connections(
                    all_connected, all_elements, all_orig_indices,
                    config.stochastic_connections_fraction
                )

            # If all connections were filtered out, return diagonal
            if len(all_connected) == 0:
                return diag

            total_connections = len(all_connected)

            # Use compiled NQS if available for faster forward passes
            nqs_forward = self._nqs_compiled if self._nqs_compiled is not None else self.nqs.log_amplitude

            # Step 5: Evaluate NQS on original configs (single batch)
            log_psi_orig = nqs_forward(configs.float()).clone()

            # Step 6: Evaluate NQS on ALL connected configs in large chunks
            log_psi_connected = torch.empty(total_connections, device=self.device)

            for start in range(0, total_connections, nqs_chunk_size):
                end = min(start + nqs_chunk_size, total_connections)
                log_psi_connected[start:end] = nqs_forward(
                    all_connected[start:end].float()
                ).clone()

            # Step 7: Compute amplitude ratios psi(connected)/psi(original)
            log_psi_orig_expanded = log_psi_orig[all_orig_indices]
            ratios = torch.exp(log_psi_connected - log_psi_orig_expanded)

            # Step 8: Compute weighted contributions
            weighted = all_elements * ratios

            # Step 9: Accumulate off-diagonal contributions using scatter_add
            off_diag = torch.zeros(n_configs, device=self.device)
            off_diag.scatter_add_(0, all_orig_indices, weighted)

            # Step 10: Total local energy = diagonal + off-diagonal
            local_energies = diag + off_diag

            # Handle complex values if present
            if torch.is_complex(local_energies):
                local_energies = local_energies.real

        return local_energies

    def _get_connections_batch(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all connections for a batch of configurations."""
        config = self.config

        # Determine if we should use the cache
        use_cache = False
        if self.connection_cache is not None:
            # Use cache if: hit rate is reasonable (>0.1), OR cache was just warmed up
            # (no queries yet), OR cache is still small. The previous condition
            # (hit_rate > 0.3 or size < 1000) was buggy: after warmup with 3241 entries
            # and 0 initial queries, hit_rate=0 caused the cache to never be used.
            total_queries = self.connection_cache.hits + self.connection_cache.misses
            use_cache = (
                total_queries < 100 or  # Not enough data to judge hit rate yet
                self.connection_cache.hit_rate > 0.1 or  # Hit rate is decent
                len(self.connection_cache) < 1000  # Cache is small, overhead is low
            )

        if use_cache and self.connection_cache is not None:
            return self.connection_cache.get_batch(configs, self.hamiltonian)
        elif (config.use_parallel_connections and
              hasattr(self.hamiltonian, 'get_connections_parallel')):
            return self.hamiltonian.get_connections_parallel(
                configs, max_workers=config.parallel_workers
            )
        else:
            # Fallback: collect connections serially
            all_connected = []
            all_elements = []
            all_orig_indices = []

            for i in range(len(configs)):
                connected, elements = self.hamiltonian.get_connections(configs[i])
                n_conn = len(connected)
                if n_conn > 0:
                    all_connected.append(connected)
                    all_elements.append(elements)
                    all_orig_indices.append(
                        torch.full((n_conn,), i, dtype=torch.long, device=self.device)
                    )

            if all_connected:
                return (
                    torch.cat(all_connected, dim=0),
                    torch.cat(all_elements, dim=0),
                    torch.cat(all_orig_indices, dim=0)
                )
            return (
                torch.empty(0, configs.shape[1], device=self.device),
                torch.empty(0, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device)
            )

    def _truncate_connections(
        self,
        all_connected: torch.Tensor,
        all_elements: torch.Tensor,
        all_orig_indices: torch.Tensor,
        n_configs: int,
        max_per_config: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Truncate connections to keep only the top-k by matrix element magnitude.

        This provides a controlled approximation that keeps the most important
        connections (typically single excitations and strong doubles).

        For C2H4: reduces from ~3000 to ~200 connections per config (15x speedup)
        """
        if len(all_connected) == 0:
            return all_connected, all_elements, all_orig_indices

        # Get absolute values for sorting
        abs_elements = torch.abs(all_elements)

        # Process each config's connections
        keep_mask = torch.zeros(len(all_connected), dtype=torch.bool, device=self.device)

        for i in range(n_configs):
            config_mask = all_orig_indices == i
            if config_mask.sum() <= max_per_config:
                # Keep all connections for this config
                keep_mask |= config_mask
            else:
                # Keep only top-k by magnitude
                config_indices = torch.where(config_mask)[0]
                config_abs = abs_elements[config_indices]
                _, topk_local = torch.topk(config_abs, max_per_config)
                keep_indices = config_indices[topk_local]
                keep_mask[keep_indices] = True

        return (
            all_connected[keep_mask],
            all_elements[keep_mask],
            all_orig_indices[keep_mask]
        )

    def _sample_connections(
        self,
        all_connected: torch.Tensor,
        all_elements: torch.Tensor,
        all_orig_indices: torch.Tensor,
        fraction: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stochastically sample a fraction of connections.

        Uses importance sampling weighted by matrix element magnitude.
        The returned elements are reweighted to provide an unbiased estimate.
        """
        if len(all_connected) == 0 or fraction >= 1.0:
            return all_connected, all_elements, all_orig_indices

        n_total = len(all_connected)
        n_sample = max(1, int(n_total * fraction))

        # Importance sampling: probability proportional to |element|
        abs_elements = torch.abs(all_elements)
        probs = abs_elements / (abs_elements.sum() + 1e-10)

        # Sample indices
        indices = torch.multinomial(probs, n_sample, replacement=False)

        # Reweight elements to account for importance sampling
        # For importance sampling with prob_i ∝ |element_i|:
        #   E[sum over sampled (element_i / (n_sample * prob_i))] = sum(elements)
        # This provides an unbiased estimate of the full sum.
        # NOTE: Do NOT multiply by n_total - that was a bug causing ~300,000x energy inflation!
        sampled_probs = probs[indices]
        reweight_factor = 1.0 / (n_sample * sampled_probs + 1e-10)
        reweighted_elements = all_elements[indices] * reweight_factor

        return (
            all_connected[indices],
            reweighted_elements,
            all_orig_indices[indices]
        )

    def _compute_flow_loss(
        self,
        all_configs: torch.Tensor,
        unique_configs: torch.Tensor,
        nqs_probs: torch.Tensor,
        local_energies: torch.Tensor,
        energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute mixed-objective flow loss.

        L = w_t * L_teacher + w_p * L_physics - w_e * H(flow)

        Teacher loss: Cross-entropy between flow and NQS distributions
        Physics loss: Encourage sampling low-energy configurations
        Entropy: Maintain exploration diversity
        """
        config = self.config

        # Get flow probabilities
        flow_probs = self.flow.estimate_discrete_prob(unique_configs)
        flow_probs = flow_probs / (flow_probs.sum() + 1e-10)
        log_flow_probs = torch.log(flow_probs + 1e-10)

        # === Teacher Loss ===
        # KL(NQS || Flow) = sum p_nqs * (log p_nqs - log p_flow)
        teacher_loss = -torch.sum(nqs_probs.detach() * log_flow_probs)

        # === Physics Loss ===
        # Encourage flow to sample low-energy configurations
        # Use energy importance: w_i propto exp(-beta * E_loc_i)
        if config.use_energy_baseline:
            # Subtract baseline for variance reduction
            energy_deviation = local_energies - energy.detach()
        else:
            energy_deviation = local_energies

        # Soft importance weighting (lower energy = higher importance)
        # We want flow to assign higher probability to lower energy configs
        # Physics loss = E_flow[E_loc] (minimize expected energy under flow)
        physics_loss = (flow_probs * energy_deviation.detach()).sum()

        # === Entropy Bonus ===
        # H(flow) = -sum p_flow * log p_flow
        entropy = -torch.sum(flow_probs * log_flow_probs)

        # Combined loss
        total_loss = (
            config.teacher_weight * teacher_loss +
            config.physics_weight * physics_loss -
            config.entropy_weight * entropy
        )

        # Scale by energy magnitude - MULTIPLY not divide (paper Eq. 16)
        # Paper: L_φ = -|E[ψ_θ]|/|S| × Σ p_θ(x) × log(p̂_φ(x))
        # The |E| factor prioritizes learning when energy is poor (high magnitude)
        # For molecular systems with E ~ -60 to -80 Ha, this amplifies gradients
        # We divide by |S| (number of unique configs) for normalization
        n_unique = len(unique_configs)
        total_loss = total_loss * torch.abs(energy.detach()) / n_unique

        components = {
            'teacher': teacher_loss,
            'physics': physics_loss,
            'entropy': entropy,
        }

        return total_loss, components

    def _compute_nqs_loss(
        self,
        configs: torch.Tensor,
        probs: torch.Tensor,
        local_energies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NQS loss (energy minimization).

        Uses variance-reduced estimator:
        L = <E_loc> + lambda * Var(E_loc)
        """
        # Recompute with gradients
        log_amp = self.nqs.log_amplitude(configs.float())
        log_probs = 2 * log_amp  # log(|psi|^2) = 2 * log|psi|

        # REINFORCE-style gradient
        # d<E>/d_theta = 2 * Re[<(E_loc - <E>) * d log psi/d_theta>]
        energy = (local_energies.detach() * probs.detach()).sum()
        centered_energies = local_energies.detach() - energy

        # Policy gradient loss
        loss = (centered_energies * log_probs * probs.detach()).sum()

        return loss

    def _update_accumulated_basis(self, new_configs: torch.Tensor):
        """
        Add new configurations to accumulated basis.

        Uses hash-based deduplication for O(n) complexity instead of
        torch.unique which is O(n log n).

        CRITICAL: Always includes essential configs (HF + singles + doubles)
        to ensure the accumulated basis contains the ground state region.
        """
        device = new_configs.device
        num_sites = new_configs.shape[1]
        max_size = self.config.max_accumulated_basis

        # CRITICAL: Always include essential configs first
        if self._essential_configs is not None and self.accumulated_basis is None:
            # First call: initialize with essential configs
            new_configs = torch.cat([self._essential_configs, new_configs], dim=0)

        # Use integer hash for fast deduplication
        # Note: CUDA doesn't support matmul for Long tensors, so we use float64
        # which has enough precision for exact integers up to 2^53
        if num_sites <= 52:
            powers = (2.0 ** torch.arange(num_sites - 1, -1, -1, device=device, dtype=torch.float64))

            if self.accumulated_basis is None:
                # First batch: just deduplicate new_configs
                keys = (new_configs.double() @ powers).long().tolist()
                seen = {}
                unique_indices = []
                for i, k in enumerate(keys):
                    if k not in seen:
                        seen[k] = True
                        unique_indices.append(i)
                self.accumulated_basis = new_configs[unique_indices]
            else:
                # Compute keys for existing basis
                existing_keys = (self.accumulated_basis.double() @ powers).long().tolist()
                existing_set = set(existing_keys)

                # Find new unique configs
                new_keys = (new_configs.double() @ powers).long().tolist()
                new_unique_indices = []
                for i, k in enumerate(new_keys):
                    if k not in existing_set:
                        existing_set.add(k)
                        new_unique_indices.append(i)

                # Append new unique configs
                if new_unique_indices:
                    self.accumulated_basis = torch.cat([
                        self.accumulated_basis,
                        new_configs[new_unique_indices]
                    ], dim=0)
        else:
            # For very large systems, fall back to torch.unique
            if self.accumulated_basis is None:
                self.accumulated_basis = torch.unique(new_configs, dim=0)
            else:
                combined = torch.cat([self.accumulated_basis, new_configs], dim=0)
                self.accumulated_basis = torch.unique(combined, dim=0)

        # Prune if too large - keep essential configs + random subset of the rest
        if len(self.accumulated_basis) > max_size:
            if self._essential_configs is not None:
                # CRITICAL: Always preserve essential configs (HF + singles + doubles)
                # Previous bug: random pruning could lose these, causing NF to explore
                # wrong region of Hilbert space for large systems like C2H4
                n_essential = len(self._essential_configs)
                n_sites = self.accumulated_basis.shape[1]
                powers = (2.0 ** torch.arange(n_sites - 1, -1, -1, device=device, dtype=torch.float64))

                ess_ints = set((self._essential_configs.double() @ powers).long().tolist())
                acc_ints = (self.accumulated_basis.double() @ powers).long().tolist()

                essential_mask = torch.tensor(
                    [k in ess_ints for k in acc_ints], dtype=torch.bool, device=device
                )
                non_essential_mask = ~essential_mask

                essential_part = self.accumulated_basis[essential_mask]
                non_essential_part = self.accumulated_basis[non_essential_mask]

                # Fill remaining budget with random non-essential configs
                remaining_budget = max_size - len(essential_part)
                if remaining_budget > 0 and len(non_essential_part) > 0:
                    n_keep = min(remaining_budget, len(non_essential_part))
                    rand_idx = torch.randperm(len(non_essential_part), device=device)[:n_keep]
                    self.accumulated_basis = torch.cat([essential_part, non_essential_part[rand_idx]], dim=0)
                else:
                    self.accumulated_basis = essential_part[:max_size]
            else:
                indices = torch.randperm(len(self.accumulated_basis), device=device)[:max_size]
                self.accumulated_basis = self.accumulated_basis[indices]

    def _compute_accumulated_energy(self) -> float:
        """
        Compute energy in accumulated basis via diagonalization.

        Uses sparse eigensolver for large bases (>500 configs) for efficiency.
        """
        if self.accumulated_basis is None or len(self.accumulated_basis) == 0:
            return float('inf')

        n_basis = len(self.accumulated_basis)

        with torch.no_grad():
            H_matrix = self.hamiltonian.matrix_elements(
                self.accumulated_basis, self.accumulated_basis
            )
            H_np = H_matrix.cpu().numpy().astype(np.float64)

            # Ensure Hermitian
            H_np = 0.5 * (H_np + H_np.T)

            # Use sparse eigensolver for large matrices
            if n_basis > 500:
                try:
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.linalg import eigsh
                    H_sparse = csr_matrix(H_np)
                    eigenvalues, _ = eigsh(H_sparse, k=1, which='SA', tol=1e-6)
                    return float(eigenvalues[0])
                except Exception:
                    pass  # Fall back to dense

            # Dense diagonalization for small matrices
            eigenvalues, _ = np.linalg.eigh(H_np)
            return float(eigenvalues[0])


def create_physics_guided_trainer(
    flow: nn.Module,
    nqs: nn.Module,
    hamiltonian: Any,
    device: str = "cuda",
    teacher_weight: float = 0.5,
    physics_weight: float = 0.4,
    entropy_weight: float = 0.1,
    **kwargs,
) -> PhysicsGuidedFlowTrainer:
    """
    Factory function to create physics-guided trainer.

    Args:
        flow: Normalizing flow model
        nqs: Neural quantum state model
        hamiltonian: System Hamiltonian
        device: Compute device
        teacher_weight: Weight for teacher signal (match NQS)
        physics_weight: Weight for physics signal (favor low energy)
        entropy_weight: Weight for entropy bonus (exploration)
        **kwargs: Additional config parameters

    Returns:
        Configured PhysicsGuidedFlowTrainer
    """
    config = PhysicsGuidedConfig(
        teacher_weight=teacher_weight,
        physics_weight=physics_weight,
        entropy_weight=entropy_weight,
        **kwargs,
    )

    return PhysicsGuidedFlowTrainer(
        flow=flow,
        nqs=nqs,
        hamiltonian=hamiltonian,
        config=config,
        device=device,
    )
