"""
Sample-based Quantum Diagonalization (SQD).

Implements the SQD algorithm from:
    "Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer"

In this pipeline, the NF-NQS replaces the quantum circuit as the sampler.
The algorithm:
1. Takes NF-NQS sampled configurations as input
2. Optionally injects depolarizing noise to emulate quantum hardware noise
3. Filters for correct particle number; recovers wrong-N configs via S-CORE
4. Creates K batches with spin symmetry enhancement
5. Diagonalizes projected Hamiltonian in each batch independently
6. Self-consistently updates orbital occupancies and re-batches
7. Performs energy-variance extrapolation across batches

Two SQD sub-modes:
- SQD-Clean (noise_rate=0): particle-conserving samples → batch diag + extrapolation
- SQD-Recovery (noise_rate>0): noise injection → S-CORE recovery loop → batch diag
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Support both package imports and direct script execution
try:
    from ..utils.gpu_linalg import gpu_eigh, gpu_eigsh
except ImportError:
    try:
        from utils.gpu_linalg import gpu_eigh, gpu_eigsh
    except ImportError:
        gpu_eigh = None
        gpu_eigsh = None


def inject_depolarizing_noise(
    configs: torch.Tensor,
    noise_rate: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """
    Inject depolarizing noise into particle-conserving configurations.

    Emulates quantum hardware noise by randomly flipping bits at rate `noise_rate`.
    This produces configs with wrong particle numbers, enabling SQD's S-CORE
    (Self-Consistent Configuration Recovery) mechanism to function as designed.

    The IBM SQD paper uses LUCJ circuits that conserve particle number in the
    noiseless setting; it is hardware noise that breaks this symmetry. This
    function emulates that noise channel classically.

    Args:
        configs: (n_configs, num_sites) binary tensor of valid configurations
        noise_rate: probability of flipping each bit (0.0 = no noise, 1.0 = full scramble)
        seed: random seed for reproducibility

    Returns:
        Noisy configs tensor (same shape, some with wrong particle number)
    """
    if noise_rate <= 0.0:
        return configs.clone()

    gen = torch.Generator(device=configs.device)
    gen.manual_seed(seed)

    # Each bit is flipped independently with probability noise_rate
    flip_mask = torch.bernoulli(
        torch.full_like(configs, noise_rate, dtype=torch.float32),
        generator=gen,
    ).to(configs.dtype)

    # XOR: flip where mask is 1
    noisy = (configs + flip_mask) % 2

    return noisy


@dataclass
class SQDConfig:
    """Configuration for SQD solver."""

    # Batch parameters
    num_batches: int = 5              # K: number of independent batches
    batch_size: int = 0               # d: configs per batch (0 = auto)

    # Self-consistent configuration recovery
    self_consistent_iters: int = 3    # Max iterations for self-consistency
    occupancy_convergence: float = 0.01  # Convergence threshold for orbital occupancies

    # Spin symmetry
    spin_penalty: float = 0.0         # Lambda for S^2 penalty (0 = disabled)
    use_spin_symmetry_enhancement: bool = True  # Spin-up/down recombination

    # Noise injection (emulates quantum hardware depolarizing noise)
    noise_rate: float = 0.0           # Bit flip probability (0 = clean SQD, >0 = recovery mode)

    # Configuration recovery (auto-enabled when noise_rate > 0)
    enable_config_recovery: bool = False  # Disabled by default (NF conserves particles)
    recovery_delta: float = 0.01      # Modified ReLU parameter delta
    recovery_h: float = 0.0           # Modified ReLU corner (0 = auto from filling)

    # Parallelization
    max_workers: int = 1              # Number of parallel workers for batch diag (1 = sequential)


class SQDSolver:
    """
    Sample-based Quantum Diagonalization (SQD).

    Replaces quantum circuit sampling with NF-NQS sampling.
    Implements batch diagonalization with self-consistent orbital occupancies
    from the IBM quantum-centric supercomputer paper.
    """

    def __init__(self, hamiltonian, config: Optional[SQDConfig] = None):
        self.hamiltonian = hamiltonian
        self.config = config or SQDConfig()
        self.num_sites = hamiltonian.num_sites

        # Hamiltonian cache: avoids rebuilding H matrix for identical batches
        # across self-consistent iterations (B1 optimization).
        # Note: accessed by ThreadPoolExecutor workers, but safe under CPython GIL
        # because each batch has a unique hash (no concurrent writes to same key).
        self._h_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Molecular properties
        self._is_molecular = hasattr(hamiltonian, 'n_alpha')
        if self._is_molecular:
            self.n_alpha = hamiltonian.n_alpha
            self.n_beta = hamiltonian.n_beta
            self.n_orbitals = hamiltonian.n_orbitals
            self.n_electrons = self.n_alpha + self.n_beta
        else:
            self.n_alpha = None
            self.n_beta = None
            self.n_orbitals = self.num_sites // 2
            self.n_electrons = None

    def run(self, nf_basis: torch.Tensor, progress: bool = True) -> Dict[str, Any]:
        """
        Run SQD algorithm on NF-NQS sampled configurations.

        Args:
            nf_basis: Tensor of configurations (n_configs, num_sites)
            progress: Whether to print progress

        Returns:
            Dictionary with energy, batch results, and diagnostics
        """
        cfg = self.config
        device = nf_basis.device
        n_configs = len(nf_basis)

        # Clear H-matrix cache for fresh run
        self._h_cache.clear()

        # Determine recovery mode (local variable — don't mutate shared config)
        enable_recovery = cfg.enable_config_recovery or (cfg.noise_rate > 0)

        mode_str = "Recovery" if enable_recovery else "Clean"
        print(f"SQD ({mode_str}): {n_configs} input configurations, {cfg.num_batches} batches")

        # Step 0: Inject depolarizing noise (emulates quantum hardware noise)
        if cfg.noise_rate > 0:
            nf_basis = inject_depolarizing_noise(nf_basis, cfg.noise_rate)
            print(f"  Noise injection: rate={cfg.noise_rate:.2f}")

        # Step 1: Filter for correct particle number
        valid_configs, invalid_configs = self._filter_particle_number(nf_basis)
        print(f"  Valid configs (correct N): {len(valid_configs)}")
        if len(invalid_configs) > 0:
            print(f"  Invalid configs (wrong N): {len(invalid_configs)}")

        # Step 1b: Inject clean essential configs (HF + singles + doubles).
        # Noise can corrupt all copies of important configs, permanently losing
        # them. The IBM paper's LUCJ circuit heavily weights these states, so
        # they survive noise statistically. We guarantee them explicitly by
        # injecting clean copies after noise — this is the NF-sampler equivalent
        # of the circuit's bias toward low-excitation determinants.
        if cfg.noise_rate > 0 and self._is_molecular:
            essential_clean = self._generate_essential_configs(device)
            if len(essential_clean) > 0:
                n_before = len(valid_configs)
                valid_configs = torch.cat([valid_configs, essential_clean], dim=0)
                valid_configs = torch.unique(valid_configs, dim=0)
                n_injected = len(valid_configs) - n_before
                if n_injected > 0 and progress:
                    print(f"  Essential config injection: +{n_injected} clean configs "
                          f"({len(essential_clean)} generated, "
                          f"{len(essential_clean) - n_injected} already present)")

        # Determine base batch size (may be updated after recovery)
        user_batch_size = cfg.batch_size

        # Step 2: Initialize orbital occupancies from HF state
        if self._is_molecular:
            hf_state = self.hamiltonian.get_hf_state().float()
            orbital_occ = hf_state.cpu().numpy()  # small 1D array, CPU is fine
        else:
            orbital_occ = np.full(self.num_sites, 0.5)

        # Step 3: Self-consistent loop
        all_configs = valid_configs
        best_energy = float('inf')
        best_results = None

        for sc_iter in range(max(1, cfg.self_consistent_iters)):
            if progress:
                print(f"\n  Self-consistent iteration {sc_iter + 1}/{cfg.self_consistent_iters}")

            # Configuration recovery (if enabled and there are invalid configs)
            if enable_recovery and len(invalid_configs) > 0:
                recovered = self._recover_configurations(
                    invalid_configs, orbital_occ, device
                )
                if len(recovered) > 0:
                    all_configs = torch.cat([valid_configs, recovered], dim=0)
                    all_configs = torch.unique(all_configs, dim=0)
                    if progress:
                        print(f"    Recovered {len(recovered)} configs -> {len(all_configs)} total")
            else:
                all_configs = valid_configs

            # Compute batch size based on available configs (after recovery).
            # For small systems (n <= 5000): d = n (full diag is fast, O(n^3) cheap).
            # For large systems (n > 5000): d = 2n/3 — random overlapping subsets
            # enable energy-variance extrapolation across batches [2],
            # and reduce diag cost by ~3.4x per batch.
            n_available = len(all_configs)
            if user_batch_size > 0:
                batch_size = min(user_batch_size, n_available)
            elif n_available <= 5000:
                batch_size = n_available
            else:
                batch_size = n_available * 2 // 3
            if sc_iter == 0:
                print(f"  Batch size (d): {batch_size}")

            # Create K batches
            batches = self._create_batches(all_configs, batch_size, cfg.num_batches)

            # Diagonalize each batch (parallel when max_workers > 1)
            if cfg.max_workers > 1 and len(batches) > 1:
                from concurrent.futures import ThreadPoolExecutor
                import os
                # Prevent BLAS thread oversubscription: each worker uses 1 BLAS thread
                old_omp = os.environ.get('OMP_NUM_THREADS')
                old_openblas = os.environ.get('OPENBLAS_NUM_THREADS')
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['OPENBLAS_NUM_THREADS'] = '1'
                try:
                    n_workers = min(cfg.max_workers, len(batches))
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futures = {
                            pool.submit(self._diagonalize_batch, batch, k): k
                            for k, batch in enumerate(batches)
                        }
                        batch_results = [None] * len(batches)
                        for future in futures:
                            k = futures[future]
                            batch_results[k] = future.result()
                finally:
                    # Restore BLAS thread settings
                    if old_omp is not None:
                        os.environ['OMP_NUM_THREADS'] = old_omp
                    else:
                        os.environ.pop('OMP_NUM_THREADS', None)
                    if old_openblas is not None:
                        os.environ['OPENBLAS_NUM_THREADS'] = old_openblas
                    else:
                        os.environ.pop('OPENBLAS_NUM_THREADS', None)
                if progress:
                    for k, result in enumerate(batch_results):
                        print(f"    Batch {k+1}: E = {result['energy']:.8f} Ha "
                              f"({result['batch_size']} configs, var = {result['variance']:.2e})")
            else:
                batch_results = []
                for k, batch in enumerate(batches):
                    result = self._diagonalize_batch(batch, k)
                    batch_results.append(result)
                    if progress:
                        print(f"    Batch {k+1}: E = {result['energy']:.8f} Ha "
                              f"({len(batch)} configs, var = {result['variance']:.2e})")

            # Compute orbital occupancies from eigenstates
            new_occ = self._compute_orbital_occupancies(batch_results)

            # Check convergence
            occ_change = np.max(np.abs(new_occ - orbital_occ))
            orbital_occ = new_occ
            if progress:
                print(f"    Max occupancy change: {occ_change:.6f}")

            # Track best energy
            energies = [r['energy'] for r in batch_results]
            mean_energy = np.mean(energies)
            if mean_energy < best_energy:
                best_energy = mean_energy
                best_results = batch_results

            if occ_change < cfg.occupancy_convergence and sc_iter > 0:
                if progress:
                    print(f"    Converged after {sc_iter + 1} iterations")
                break

        # Step 4: Energy-variance extrapolation
        extrapolated_energy, ev_results = self._energy_variance_extrapolation(best_results)

        # Pick best energy: minimum of batch energies and extrapolated
        batch_energies = [r['energy'] for r in best_results]
        min_batch_energy = min(batch_energies)

        # Use minimum batch energy (variational upper bound)
        final_energy = min_batch_energy
        if extrapolated_energy is not None and extrapolated_energy < final_energy:
            # Extrapolation can go below variational bound
            final_energy = extrapolated_energy

        print(f"\n  SQD Results:")
        print(f"    Min batch energy:    {min_batch_energy:.8f} Ha")
        print(f"    Mean batch energy:   {np.mean(batch_energies):.8f} Ha")
        print(f"    Std batch energy:    {np.std(batch_energies):.8f} Ha")
        if extrapolated_energy is not None:
            print(f"    Extrapolated energy: {extrapolated_energy:.8f} Ha")
        print(f"    Final energy:        {final_energy:.8f} Ha")

        return {
            "energy": final_energy,
            "min_batch_energy": min_batch_energy,
            "mean_batch_energy": float(np.mean(batch_energies)),
            "std_batch_energy": float(np.std(batch_energies)),
            "extrapolated_energy": extrapolated_energy,
            "batch_energies": batch_energies,
            "batch_variances": [r['variance'] for r in best_results],
            "batch_sizes": [r['batch_size'] for r in best_results],
            "energy_variance_fit": ev_results,
            "self_consistent_iters": sc_iter + 1,
            "orbital_occupancies": orbital_occ.tolist(),
            "num_input_configs": n_configs,
            "num_valid_configs": len(valid_configs),
        }

    def _filter_particle_number(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split configs into correct and incorrect particle number sets."""
        if not self._is_molecular:
            return configs, torch.empty(0, self.num_sites, device=configs.device)

        n_orb = self.n_orbitals
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        valid_mask = (alpha_counts == self.n_alpha) & (beta_counts == self.n_beta)

        return configs[valid_mask], configs[~valid_mask]

    def _recover_configurations(
        self,
        wrong_configs: torch.Tensor,
        orbital_occ: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Recover configurations with wrong particle number by flipping orbitals.

        Uses the modified ReLU weighting from the paper:
        w(y) = delta * y / h          if y <= h
        w(y) = delta + (1-delta)(y-h)/(1-h)  if y > h

        where y = |x_ps - n_ps|, h = filling factor, delta = 0.01

        B6 optimization: Vectorized — groups configs by (alpha_diff, beta_diff)
        to batch-process identical correction patterns together, avoiding the
        per-config Python loop for the common case.
        """
        if len(wrong_configs) == 0:
            return torch.empty(0, self.num_sites, device=device)

        cfg = self.config
        n_orb = self.n_orbitals
        delta = cfg.recovery_delta
        h = cfg.recovery_h if cfg.recovery_h > 0 else self.n_electrons / self.num_sites

        occ_tensor = torch.tensor(orbital_occ, dtype=torch.float32, device=device)

        # Vectorized particle counts for all configs at once
        alpha_counts = wrong_configs[:, :n_orb].sum(dim=1).long()
        beta_counts = wrong_configs[:, n_orb:].sum(dim=1).long()

        recovered = wrong_configs.clone()

        # Fix alpha sector for all configs that need it
        alpha_needs_fix = alpha_counts != self.n_alpha
        if alpha_needs_fix.any():
            fix_indices = alpha_needs_fix.nonzero(as_tuple=True)[0]
            for idx in fix_indices:
                recovered[idx, :n_orb] = self._fix_spin_sector(
                    recovered[idx, :n_orb], occ_tensor[:n_orb],
                    self.n_alpha, int(alpha_counts[idx].item()), delta, h
                )

        # Fix beta sector for all configs that need it
        beta_needs_fix = beta_counts != self.n_beta
        if beta_needs_fix.any():
            fix_indices = beta_needs_fix.nonzero(as_tuple=True)[0]
            for idx in fix_indices:
                recovered[idx, n_orb:] = self._fix_spin_sector(
                    recovered[idx, n_orb:], occ_tensor[n_orb:],
                    self.n_beta, int(beta_counts[idx].item()), delta, h
                )

        return torch.unique(recovered, dim=0)

    def _fix_spin_sector(
        self,
        sector: torch.Tensor,
        occ: torch.Tensor,
        target_n: int,
        current_n: int,
        delta: float,
        h: float,
    ) -> torch.Tensor:
        """Fix particle number in a spin sector by probabilistic bit flipping."""
        result = sector.clone()
        diff = current_n - target_n

        if diff > 0:
            # Too many particles: flip some occupied -> unoccupied
            occupied = (result == 1).nonzero(as_tuple=True)[0]
            if len(occupied) == 0:
                return result
            # Weight: high weight for orbitals that should be empty
            distances = torch.abs(1.0 - occ[occupied])
            weights = self._modified_relu(distances, delta, h)
            weights = weights / (weights.sum() + 1e-12)
            n_flip = min(abs(diff), len(occupied))
            indices = torch.multinomial(weights, n_flip, replacement=False)
            result[occupied[indices]] = 0

        elif diff < 0:
            # Too few particles: flip some unoccupied -> occupied
            empty = (result == 0).nonzero(as_tuple=True)[0]
            if len(empty) == 0:
                return result
            # Weight: high weight for orbitals that should be occupied
            distances = torch.abs(0.0 - occ[empty])
            weights = self._modified_relu(distances, delta, h)
            weights = weights / (weights.sum() + 1e-12)
            n_flip = min(abs(diff), len(empty))
            indices = torch.multinomial(weights, n_flip, replacement=False)
            result[empty[indices]] = 1

        return result

    @staticmethod
    def _modified_relu(y: torch.Tensor, delta: float, h: float) -> torch.Tensor:
        """Modified ReLU function from the paper (Eq. in supplement)."""
        result = torch.zeros_like(y)
        low_mask = y <= h
        high_mask = ~low_mask

        if h > 0:
            result[low_mask] = delta * y[low_mask] / h
        result[high_mask] = delta + (1.0 - delta) * (y[high_mask] - h) / max(1.0 - h, 1e-12)

        return result.clamp(min=1e-12)

    def _create_batches(
        self,
        configs: torch.Tensor,
        batch_size: int,
        num_batches: int,
    ) -> List[torch.Tensor]:
        """
        Create K batches of configurations.

        If spin symmetry enhancement is enabled, samples sqrt(d/2) configs,
        extracts unique spin-up/down parts, and forms all combinations.
        This facilitates singlet state construction.

        When random subsampling is used, essential CI configs (HF + singles
        + doubles) are always included in every batch to ensure the core
        wavefunction components are present. In the IBM paper [2], the quantum
        circuit heavily weights these configs, so they are always well-represented
        in every batch. Our random subsampling needs explicit inclusion.
        """
        cfg = self.config
        n_configs = len(configs)

        if n_configs == 0:
            return [configs for _ in range(num_batches)]

        # Identify essential config indices (for guaranteed inclusion in batches)
        essential_mask = self._identify_essential_configs(configs)

        batches = []

        for k in range(num_batches):
            if n_configs <= batch_size:
                # Small system: use all configs in every batch
                batch = configs.clone()
            elif cfg.use_spin_symmetry_enhancement and self._is_molecular:
                batch = self._create_spin_enhanced_batch(configs, batch_size, k)
            else:
                # Random subsampling with essential config guarantee
                batch = self._create_batch_with_essentials(
                    configs, batch_size, k, essential_mask
                )

            batches.append(batch)

        return batches

    def _identify_essential_configs(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Identify which configs in the pool are essential (HF + singles + doubles).

        Returns a boolean mask over configs indicating essential membership.
        """
        if not self._is_molecular or not hasattr(self.hamiltonian, 'get_hf_state'):
            return torch.zeros(len(configs), dtype=torch.bool, device=configs.device)

        hf = self.hamiltonian.get_hf_state().to(configs.device)

        # Compute Hamming distance from HF for each config
        hamming = (configs != hf.unsqueeze(0)).sum(dim=1)

        # Essential = HF (dist 0), singles (dist 2: one occ→virt), doubles (dist 4)
        # In second quantization, a single excitation flips 2 bits (remove one, add one)
        essential_mask = hamming <= 4

        return essential_mask

    def _generate_essential_configs(self, device: torch.device) -> torch.Tensor:
        """
        Generate clean HF + singles + doubles configs from the Hamiltonian.

        These are injected after noise to guarantee the core wavefunction
        components survive regardless of noise rate. Mirrors the pipeline's
        _generate_essential_configs() logic.
        """
        from itertools import combinations

        if not self._is_molecular or not hasattr(self.hamiltonian, 'get_hf_state'):
            return torch.empty(0, self.num_sites, device=device)

        n_orb = self.n_orbitals
        n_alpha = self.n_alpha
        n_beta = self.n_beta

        hf_state = self.hamiltonian.get_hf_state()
        essential = [hf_state.clone()]

        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Single excitations
        for i in occ_alpha:
            for a in virt_alpha:
                cfg = hf_state.clone()
                cfg[i] = 0
                cfg[a] = 1
                essential.append(cfg)
        for i in occ_beta:
            for a in virt_beta:
                cfg = hf_state.clone()
                cfg[i + n_orb] = 0
                cfg[a + n_orb] = 1
                essential.append(cfg)

        # Double excitations with proportional allocation per type.
        # αβ doubles dominate correlation energy — each type gets a fair
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
            ab_frac = max(0.5, n_ab_total / total_possible if total_possible > 0 else 0.5)
            remaining_frac = 1.0 - ab_frac
            aa_frac = remaining_frac * (n_aa_total / (n_aa_total + n_bb_total)) if (n_aa_total + n_bb_total) > 0 else 0
            max_ab = int(ab_frac * max_doubles)
            max_aa = int(aa_frac * max_doubles)
            max_bb = max_doubles - max_ab - max_aa

        aa_count = 0
        for i, j in combinations(occ_alpha, 2):
            for a, b in combinations(virt_alpha, 2):
                if aa_count >= max_aa:
                    break
                cfg = hf_state.clone()
                cfg[i] = 0
                cfg[j] = 0
                cfg[a] = 1
                cfg[b] = 1
                essential.append(cfg)
                aa_count += 1
            if aa_count >= max_aa:
                break

        bb_count = 0
        for i, j in combinations(occ_beta, 2):
            for a, b in combinations(virt_beta, 2):
                if bb_count >= max_bb:
                    break
                cfg = hf_state.clone()
                cfg[i + n_orb] = 0
                cfg[j + n_orb] = 0
                cfg[a + n_orb] = 1
                cfg[b + n_orb] = 1
                essential.append(cfg)
                bb_count += 1
            if bb_count >= max_bb:
                break

        ab_count = 0
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b in virt_beta:
                        if ab_count >= max_ab:
                            break
                        cfg = hf_state.clone()
                        cfg[i] = 0
                        cfg[j + n_orb] = 0
                        cfg[a] = 1
                        cfg[b + n_orb] = 1
                        essential.append(cfg)
                        ab_count += 1
                    if ab_count >= max_ab:
                        break
                if ab_count >= max_ab:
                    break
            if ab_count >= max_ab:
                break

        return torch.unique(torch.stack(essential).to(device), dim=0)

    def _create_batch_with_essentials(
        self,
        configs: torch.Tensor,
        batch_size: int,
        batch_index: int,
        essential_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a batch that always includes essential configs, plus random fill.
        """
        n_configs = len(configs)
        essential_idx = torch.where(essential_mask)[0]
        non_essential_idx = torch.where(~essential_mask)[0]

        n_essential = len(essential_idx)

        if n_essential >= batch_size:
            # More essentials than batch size — subsample essentials
            gen = torch.Generator(device='cpu')
            gen.manual_seed(42 + batch_index)
            perm = torch.randperm(n_essential, generator=gen)[:batch_size]
            return configs[essential_idx[perm]]

        # Include all essentials + random fill from non-essentials
        n_fill = batch_size - n_essential
        n_fill = min(n_fill, len(non_essential_idx))

        gen = torch.Generator(device='cpu')
        gen.manual_seed(42 + batch_index)
        fill_perm = torch.randperm(len(non_essential_idx), generator=gen)[:n_fill]
        fill_idx = non_essential_idx[fill_perm]

        batch_idx = torch.cat([essential_idx, fill_idx])
        return configs[batch_idx]

    def _create_spin_enhanced_batch(
        self,
        configs: torch.Tensor,
        batch_size: int,
        batch_index: int,
    ) -> torch.Tensor:
        """
        Create a batch with spin symmetry enhancement.

        Sample sqrt(d/2) configs, extract unique spin-up and spin-down parts,
        form all alpha-beta combinations to facilitate singlet state construction.
        """
        n_orb = self.n_orbitals
        n_configs = len(configs)

        # Sample sqrt(d/2) configurations
        n_sample = max(1, int(np.sqrt(batch_size / 2)))
        n_sample = min(n_sample, n_configs)

        gen = torch.Generator(device='cpu')
        gen.manual_seed(42 + batch_index)
        perm = torch.randperm(n_configs, generator=gen)[:n_sample]
        sampled = configs[perm]

        # Extract unique alpha and beta parts
        alpha_parts = sampled[:, :n_orb]
        beta_parts = sampled[:, n_orb:]

        unique_alpha = torch.unique(alpha_parts, dim=0)
        unique_beta = torch.unique(beta_parts, dim=0)

        # Form all alpha-beta combinations
        n_alpha_unique = len(unique_alpha)
        n_beta_unique = len(unique_beta)

        # If too many combinations, subsample
        max_combos = batch_size
        total_combos = n_alpha_unique * n_beta_unique

        if total_combos <= max_combos:
            # All combinations
            alpha_expanded = unique_alpha.repeat_interleave(n_beta_unique, dim=0)
            beta_expanded = unique_beta.repeat(n_alpha_unique, 1)
            batch = torch.cat([alpha_expanded, beta_expanded], dim=1)
        else:
            # Random subset of combinations
            combos = []
            gen2 = torch.Generator(device='cpu')
            gen2.manual_seed(137 + batch_index)
            for _ in range(max_combos):
                ai = torch.randint(n_alpha_unique, (1,), generator=gen2).item()
                bi = torch.randint(n_beta_unique, (1,), generator=gen2).item()
                combo = torch.cat([unique_alpha[ai], unique_beta[bi]])
                combos.append(combo)
            batch = torch.stack(combos)

        # Filter for correct particle number
        if self._is_molecular:
            alpha_counts = batch[:, :n_orb].sum(dim=1)
            beta_counts = batch[:, n_orb:].sum(dim=1)
            valid = (alpha_counts == self.n_alpha) & (beta_counts == self.n_beta)
            batch = batch[valid]

        batch = torch.unique(batch, dim=0)
        return batch

    @staticmethod
    def _batch_hash(batch: torch.Tensor) -> int:
        """Compute a stable hash for a batch of configurations (for caching)."""
        # Sort rows for order-invariance, then hash the bytes
        sorted_batch, _ = torch.sort(batch, dim=0)
        return hash(sorted_batch.cpu().numpy().tobytes())

    def _diagonalize_batch(
        self, batch: torch.Tensor, batch_index: int
    ) -> Dict[str, Any]:
        """
        Project Hamiltonian into batch subspace and diagonalize.

        GPU-accelerated: builds H on GPU (via hamiltonian.matrix_elements),
        diagonalizes on GPU (via gpu_eigsh/gpu_eigh), and keeps eigenvectors
        on GPU. No CPU transfers unless fallback is needed.

        Caches H tensor per batch hash to avoid redundant builds across SC iters.
        """
        n = len(batch)
        if n == 0:
            return {
                'energy': float('inf'),
                'variance': float('inf'),
                'eigenvector': None,
                'batch_size': 0,
                'batch_index': batch_index,
            }

        device = batch.device

        # B1: Check H-tensor cache before building
        b_hash = self._batch_hash(batch)
        if b_hash in self._h_cache:
            H_gpu, cached_batch = self._h_cache[b_hash]
        else:
            # Build projected Hamiltonian (stays on GPU via hamiltonian.matrix_elements)
            H_matrix = self.hamiltonian.matrix_elements(batch, batch)
            H_gpu = H_matrix.detach().double()
            # Take real part (molecular Hamiltonians are real)
            if H_gpu.is_complex():
                H_gpu = H_gpu.real
            # Symmetrize on GPU
            H_gpu = 0.5 * (H_gpu + H_gpu.T)
            # Cache for reuse in subsequent SC iterations
            self._h_cache[b_hash] = (H_gpu, batch)

        # Add S^2 penalty if configured (applied fresh each time, not cached)
        if self.config.spin_penalty > 0 and self._is_molecular:
            S2_gpu = self._compute_s2_matrix(batch)
            H_work = H_gpu + self.config.spin_penalty * (S2_gpu @ S2_gpu)
        else:
            H_work = H_gpu

        # GPU-accelerated diagonalization
        if n == 1:
            # Trivial case: 1x1 matrix
            E0 = float(H_work[0, 0].cpu())
            ground_state = torch.ones(1, dtype=H_work.dtype, device=device)
        else:
            try:
                use_gpu = device.type == 'cuda'
                if gpu_eigsh is not None:
                    k_eig = min(2, n - 1)
                    eigenvalues, eigenvectors = gpu_eigsh(H_work, k=k_eig, which='SA', use_gpu=use_gpu)
                elif gpu_eigh is not None:
                    eigenvalues, eigenvectors = gpu_eigh(H_work, use_gpu=use_gpu)
                else:
                    raise RuntimeError("No GPU eigensolver available")

                E0 = float(eigenvalues[0].cpu())
                ground_state = eigenvectors[:, 0].double()  # stays on GPU, enforce float64
            except Exception as e:
                print(f"WARNING: GPU eigensolver failed ({type(e).__name__}: {e}), falling back to CPU")
                H_np = H_work.cpu().numpy()
                eigenvalues_np, eigenvectors_np = np.linalg.eigh(H_np)
                E0 = float(eigenvalues_np[0])
                ground_state = torch.from_numpy(eigenvectors_np[:, 0]).double().to(device)

        # Compute variance on GPU: <H^2> - <H>^2
        Hv = H_work @ ground_state
        H2_expectation = float(torch.dot(ground_state, H_work @ Hv).cpu())
        variance = H2_expectation - E0 ** 2
        variance = max(0.0, variance)

        return {
            'energy': E0,
            'variance': variance,
            'eigenvector': ground_state,  # torch tensor on device
            'batch': batch,
            'batch_size': n,
            'batch_index': batch_index,
        }

    def _compute_s2_matrix(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute S^2 matrix elements in the configuration basis.

        S^2 = S_z^2 + S_z + S_- S_+

        GPU-accelerated: diagonal computed via vectorized torch ops,
        off-diagonal via batched difference checks. Returns torch tensor on device.
        """
        n = len(configs)
        n_orb = self.n_orbitals
        device = configs.device
        S2 = torch.zeros((n, n), dtype=torch.float64, device=device)

        alpha = configs[:, :n_orb].double()  # (n, n_orb)
        beta = configs[:, n_orb:].double()   # (n, n_orb)

        # Vectorized diagonal: S_z^2 + S_z + sum_p n_{p,alpha}(1 - n_{p,beta})
        ms = 0.5 * (alpha.sum(dim=1) - beta.sum(dim=1))  # (n,)
        diag = ms * (ms + 1.0) + (alpha * (1.0 - beta)).sum(dim=1)
        S2.diagonal().copy_(diag)

        # Off-diagonal: find pairs differing by exactly one alpha-beta swap
        # Vectorized: compute all pairwise alpha/beta diffs at once
        if n <= 2000:
            # Batched pairwise computation (memory: O(n^2 * n_orb))
            alpha_diff = alpha.unsqueeze(0) - alpha.unsqueeze(1)  # (n, n, n_orb)
            beta_diff = beta.unsqueeze(0) - beta.unsqueeze(1)     # (n, n, n_orb)

            # Count +1 and -1 entries per pair
            a_plus = (alpha_diff == 1).sum(dim=2)    # (n, n)
            a_minus = (alpha_diff == -1).sum(dim=2)  # (n, n)
            b_plus = (beta_diff == 1).sum(dim=2)
            b_minus = (beta_diff == -1).sum(dim=2)

            # Exactly one +1 and one -1 in each
            count_match = (a_plus == 1) & (a_minus == 1) & (b_plus == 1) & (b_minus == 1)

            # Check that the +1 alpha site matches -1 beta site and vice versa
            # argmax gives position of the +1/-1 entries
            a_plus_pos = (alpha_diff == 1).float().argmax(dim=2)   # (n, n)
            a_minus_pos = (alpha_diff == -1).float().argmax(dim=2)
            b_plus_pos = (beta_diff == 1).float().argmax(dim=2)
            b_minus_pos = (beta_diff == -1).float().argmax(dim=2)

            site_match = (a_plus_pos == b_minus_pos) & (a_minus_pos == b_plus_pos)
            # Upper triangle only, then mirror
            upper = torch.triu(count_match & site_match, diagonal=1)
            S2 = S2 - upper.double() - upper.double().T
        else:
            # Fallback for very large n: loop on CPU to avoid CUDA scalar writes
            S2_cpu = S2.cpu()
            configs_np = configs.cpu().numpy()
            for i in range(n):
                for j in range(i + 1, n):
                    ad = configs_np[i, :n_orb] - configs_np[j, :n_orb]
                    bd = configs_np[i, n_orb:] - configs_np[j, n_orb:]
                    ap = np.where(ad == 1)[0]
                    am = np.where(ad == -1)[0]
                    bp = np.where(bd == 1)[0]
                    bm = np.where(bd == -1)[0]
                    if (len(ap) == 1 and len(am) == 1 and len(bp) == 1 and len(bm) == 1
                            and ap[0] == bm[0] and am[0] == bp[0]):
                        S2_cpu[i, j] = -1.0
                        S2_cpu[j, i] = -1.0
            S2 = S2_cpu.to(device)

        return S2

    def _compute_orbital_occupancies(
        self, batch_results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute average orbital occupancies from batch eigenstates.

        n_{p,sigma} = (1/K) sum_k <psi^(k)| n_hat_{p,sigma} |psi^(k)>

        GPU-accelerated: eigenvectors and configs stay on GPU for the matmul,
        only the final (num_sites,) result is transferred to CPU.
        """
        occupancies = None
        n_valid = 0

        for result in batch_results:
            if result['eigenvector'] is None or result['batch'] is None:
                continue

            coeffs = result['eigenvector']  # torch tensor on device or np array
            configs = result['batch']  # torch tensor on device

            # Ensure torch tensors
            if isinstance(coeffs, np.ndarray):
                coeffs = torch.from_numpy(coeffs).to(configs.device)

            # <psi|n_hat_p|psi> = sum_i |c_i|^2 * x_i_p  (GPU matmul)
            probs = (coeffs * coeffs).double()  # (d,)
            occ_k = probs @ configs.double()  # (num_sites,) — GPU matmul

            if occupancies is None:
                occupancies = occ_k
            else:
                occupancies = occupancies + occ_k
            n_valid += 1

        if occupancies is None:
            return np.zeros(self.num_sites, dtype=np.float64)

        if n_valid > 0:
            occupancies = occupancies / n_valid

        # Single transfer to CPU at the end
        return occupancies.cpu().numpy()

    def _energy_variance_extrapolation(
        self, batch_results: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Energy-variance extrapolation across batches.

        Linear fit: delta_E ~ a * (Delta_H / E^2)
        Extrapolate to Delta_H = 0 to estimate true ground state energy.
        """
        energies = []
        variances = []

        for r in batch_results:
            if r['energy'] != float('inf') and r['variance'] != float('inf'):
                energies.append(r['energy'])
                variances.append(r['variance'])

        if len(energies) < 3:
            return None, {"fit_quality": "insufficient_data", "n_points": len(energies)}

        energies = np.array(energies)
        variances = np.array(variances)

        # Compute Delta_H / E^2
        x = variances / (energies ** 2)
        y = energies

        # Linear fit: E = E_T + a * (Delta_H / E^2)
        # Use least squares
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            result = np.linalg.lstsq(A, y, rcond=None)
            slope, intercept = result[0]
            residuals = result[1] if len(result[1]) > 0 else None

            # R^2 quality metric
            ss_res = np.sum((y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

            extrapolated = float(intercept)

            return extrapolated, {
                "extrapolated_energy": extrapolated,
                "slope": float(slope),
                "r_squared": float(r_squared),
                "n_points": len(energies),
                "fit_quality": "good" if r_squared > 0.8 else "poor",
            }

        except np.linalg.LinAlgError:
            return None, {"fit_quality": "fit_failed", "n_points": len(energies)}
