"""
Diversity-Aware Basis Selection for Quantum Diagonalization.

This module implements intelligent basis selection strategies that:
1. Remove redundant/duplicate configurations
2. Bucket by excitation rank (singles, doubles, etc.)
3. Use diversity metrics (Hamming distance) for selection
4. Implement DPP-inspired selection for maximal coverage

The goal is to build a compact but representative basis that captures
the essential physics of the ground state wavefunction.

References:
- Selected-CI: Configurations selected based on importance
- DPP: Determinantal Point Processes for diverse subset selection
- Excitation-ranked CI: Systematic inclusion by excitation level
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DiversityConfig:
    """Configuration for diversity-aware selection."""

    # Maximum configurations to select
    max_configs: int = 2048

    # Excitation rank budgets (as fractions)
    rank_0_fraction: float = 0.05   # HF and near-HF
    rank_1_fraction: float = 0.25   # Single excitations
    rank_2_fraction: float = 0.50   # Double excitations (most important)
    rank_3_fraction: float = 0.15   # Triple excitations
    rank_4_plus_fraction: float = 0.05  # Higher excitations

    # Diversity parameters
    min_hamming_distance: int = 2  # Minimum distance between selected configs
    use_dpp_selection: bool = True  # Use DPP-inspired selection
    dpp_kernel_scale: float = 0.5   # Scaling for DPP kernel

    # Importance weighting
    use_nqs_importance: bool = True  # Weight by NQS probability
    use_energy_importance: bool = True  # Weight by local energy


def compute_excitation_rank(
    config: torch.Tensor,
    reference: torch.Tensor,
) -> int:
    """
    Compute excitation rank relative to reference (usually HF state).

    Excitation rank = number of orbitals that differ from reference.
    For molecular systems, we count alpha and beta separately and sum.
    """
    diff = (config != reference).sum().item()
    # Divide by 2 because each excitation involves one hole and one particle
    return diff // 2


def compute_hamming_distance(
    config1: torch.Tensor,
    config2: torch.Tensor,
) -> int:
    """Compute Hamming distance between two configurations."""
    return (config1 != config2).sum().item()


def bitpack_configs(configs: torch.Tensor) -> torch.Tensor:
    """Pack binary configs into int64 for O(1) Hamming via XOR + popcount.

    Each config (up to 64 sites) is packed into a single int64.
    Hamming distance = popcount(a XOR b).

    Args:
        configs: (n, sites) binary tensor, sites <= 64

    Returns:
        (n,) int64 tensor of packed configs
    """
    n, sites = configs.shape
    assert sites <= 64, f"Bitpacking supports up to 64 sites, got {sites}"
    powers = (1 << torch.arange(sites, dtype=torch.int64)).flip(0)
    return (configs.long() * powers).sum(dim=1)


def hamming_bitpacked(a, b) -> int:
    """Hamming distance between two bitpacked configs. O(1)."""
    # Convert torch tensors to Python int via raw bytes to avoid int64 overflow
    # in older PyTorch versions (2.5 etc.) where .item() raises on large uint values
    if hasattr(a, 'numpy'):
        a = int.from_bytes(a.cpu().numpy().tobytes(), byteorder='little', signed=True)
    if hasattr(b, 'numpy'):
        b = int.from_bytes(b.cpu().numpy().tobytes(), byteorder='little', signed=True)
    # Mask to 64 bits to handle signed int64 correctly (Python bin(-x) gives '-0b...')
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count('1')


def hamming_bitpacked_batch(packed: torch.Tensor, idx: int, indices: torch.Tensor) -> torch.Tensor:
    """Hamming distance from packed[idx] to packed[indices]. Vectorized O(k)."""
    xor = packed[indices] ^ packed[idx]
    return _popcount_int64(xor)


# Byte-level popcount lookup table (0-255)
_POPCOUNT_TABLE = torch.zeros(256, dtype=torch.int32)
for _i in range(256):
    _POPCOUNT_TABLE[_i] = bin(_i).count('1')
_POPCOUNT_TABLE_CACHE: dict = {}  # device -> table tensor


def _popcount_int64(x: torch.Tensor) -> torch.Tensor:
    """Vectorized popcount for int64 tensor using byte lookup table."""
    # Cache table per device to avoid repeated .to() calls in hot loops
    dev = x.device
    if dev not in _POPCOUNT_TABLE_CACHE:
        _POPCOUNT_TABLE_CACHE[dev] = _POPCOUNT_TABLE.to(dev)
    table = _POPCOUNT_TABLE_CACHE[dev]

    # view(uint8) requires at least 1-dim; unsqueeze scalar tensors
    was_scalar = x.dim() == 0
    if was_scalar:
        x = x.unsqueeze(0)

    # View as uint8 bytes (8 bytes per int64), sum popcount per byte
    shape = x.shape
    x_bytes = x.contiguous().view(torch.uint8)  # flatten to bytes
    counts = table[x_bytes.long()]  # lookup each byte
    # Reshape to (original_shape..., 8) and sum over last dim
    counts = counts.reshape(*shape, 8).sum(dim=-1)
    result = counts.int()
    if was_scalar:
        result = result.squeeze(0)
    return result


def stochastic_greedy_select(
    configs: torch.Tensor,
    weights: torch.Tensor,
    n_select: int,
    epsilon: float = 0.01,
    min_hamming: int = 0,
    seed: int = 42,
) -> torch.Tensor:
    """Stochastic greedy diversity selection (Mirzasoleiman et al., AAAI 2015).

    Instead of evaluating ALL n candidates at each step (O(n*k)),
    samples a random subset of size (n/k)*ln(1/epsilon) and picks the best.
    Total evaluations: O(n * ln(1/epsilon)) instead of O(n*k).

    For n=50K, k=5K, epsilon=0.01: ~230K evals instead of 250M (~1000x faster).

    Args:
        configs: (n, sites) binary configs
        weights: (n,) importance weights
        n_select: number of configs to select
        epsilon: approximation parameter (smaller = better quality, more evals)
        min_hamming: minimum Hamming distance between selected configs
        seed: random seed

    Returns:
        (n_select,) tensor of selected indices
    """
    import math

    n = len(configs)
    n_select = min(n_select, n)

    if n_select <= 0:
        return torch.tensor([], dtype=torch.long)

    # Bitpack for fast Hamming
    packed = bitpack_configs(configs)

    # Sample size per step: (n/k) * ln(1/eps)
    sample_size = max(1, int((n / max(n_select, 1)) * math.log(1.0 / epsilon)))
    sample_size = min(sample_size, n)

    gen = torch.Generator()
    gen.manual_seed(seed)

    selected = []
    selected_mask = torch.zeros(n, dtype=torch.bool)
    # Track min distance from each candidate to selected set
    min_dists = torch.full((n,), fill_value=999999, dtype=torch.int32)

    # Start with highest-weight config
    first = weights.argmax().item()
    selected.append(first)
    selected_mask[first] = True

    # Update min_dists from first selected — vectorized
    xor = packed ^ packed[first]
    dists = _popcount_int64(xor)
    min_dists = dists
    min_dists[first] = 0  # self-distance irrelevant, masked by selected_mask

    while len(selected) < n_select:
        # Get remaining indices
        remaining_indices = torch.where(~selected_mask)[0]
        if len(remaining_indices) == 0:
            break

        # Sample subset
        actual_sample_size = min(sample_size, len(remaining_indices))
        perm = torch.randperm(len(remaining_indices), generator=gen)[:actual_sample_size]
        candidates = remaining_indices[perm]

        # Score candidates vectorized: weight * max(min_dist, 1)
        cand_dists = min_dists[candidates].float()
        cand_weights = weights[candidates]
        scores = cand_weights * torch.clamp(cand_dists, min=1.0)

        # Penalize candidates below min_hamming (only if any valid alternatives exist)
        if min_hamming > 0:
            valid = cand_dists >= min_hamming
            if valid.any():
                scores[~valid] = -float('inf')

        best_local = scores.argmax().item()
        best_idx = candidates[best_local].item()

        selected.append(best_idx)
        selected_mask[best_idx] = True

        # Update min_dists incrementally — vectorized XOR + popcount
        xor = packed ^ packed[best_idx]
        new_dists = _popcount_int64(xor)
        min_dists = torch.minimum(min_dists, new_dists)

    return torch.tensor(selected, dtype=torch.long)


def compute_hamming_distance_matrix(
    configs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise Hamming distance matrix.

    Args:
        configs: (n_configs, n_sites) configurations

    Returns:
        (n_configs, n_configs) distance matrix
    """
    n = len(configs)
    # Expand for broadcasting
    c1 = configs.unsqueeze(1)  # (n, 1, sites)
    c2 = configs.unsqueeze(0)  # (1, n, sites)

    # Hamming distance = number of differing positions
    distances = (c1 != c2).sum(dim=-1)  # (n, n)

    return distances


class ExcitationBucketer:
    """
    Organizes configurations by excitation rank.

    Groups configurations into buckets based on their excitation level
    relative to a reference state (typically Hartree-Fock).
    """

    def __init__(self, reference: torch.Tensor, n_orbitals: int):
        """
        Args:
            reference: Reference configuration (e.g., HF state)
            n_orbitals: Number of spatial orbitals
        """
        self.reference = reference
        self.n_orbitals = n_orbitals
        self.buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def add_configs(self, configs: torch.Tensor):
        """Add configurations to appropriate buckets."""
        for i in range(len(configs)):
            rank = compute_excitation_rank(configs[i], self.reference)
            self.buckets[rank].append(configs[i])

    def get_bucket(self, rank: int) -> List[torch.Tensor]:
        """Get configurations at a specific excitation rank."""
        return self.buckets.get(rank, [])

    def get_bucket_sizes(self) -> Dict[int, int]:
        """Get size of each bucket."""
        return {rank: len(configs) for rank, configs in self.buckets.items()}

    def get_all_configs(self) -> torch.Tensor:
        """Get all configurations as a single tensor."""
        all_configs = []
        for rank in sorted(self.buckets.keys()):
            all_configs.extend(self.buckets[rank])
        if not all_configs:
            return torch.empty(0, len(self.reference))
        return torch.stack(all_configs)


class DiversitySelector:
    """
    Selects diverse subset of configurations.

    Uses a combination of:
    1. Excitation rank stratification
    2. Hamming distance diversity
    3. Importance weighting (NQS probability, local energy)
    """

    def __init__(
        self,
        config: DiversityConfig,
        reference: torch.Tensor,
        n_orbitals: int,
    ):
        self.config = config
        self.reference = reference
        self.n_orbitals = n_orbitals
        self.bucketer = ExcitationBucketer(reference, n_orbitals)

    def select(
        self,
        configs: torch.Tensor,
        nqs_probs: Optional[torch.Tensor] = None,
        local_energies: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Select diverse subset of configurations.

        Args:
            configs: (n_configs, n_sites) candidate configurations
            nqs_probs: (n_configs,) optional NQS probabilities
            local_energies: (n_configs,) optional local energies

        Returns:
            selected: (n_selected, n_sites) selected configurations
            stats: dictionary with selection statistics
        """
        device = configs.device
        n_configs = len(configs)

        # Remove duplicates first
        unique_configs, inverse = torch.unique(configs, dim=0, return_inverse=True)
        n_unique = len(unique_configs)

        # Bucket by excitation rank
        self.bucketer = ExcitationBucketer(self.reference.to(device), self.n_orbitals)
        self.bucketer.add_configs(unique_configs)

        # Compute importance weights
        if nqs_probs is not None or local_energies is not None:
            weights = self._compute_importance_weights(
                unique_configs, nqs_probs, local_energies, inverse
            )
        else:
            weights = torch.ones(n_unique, device=device)

        # Select from each bucket according to budget
        selected_indices = []
        bucket_stats = {}

        budget = self._compute_bucket_budgets()

        for rank in sorted(self.bucketer.buckets.keys()):
            bucket_configs = self.bucketer.get_bucket(rank)
            if not bucket_configs:
                continue

            n_bucket = len(bucket_configs)
            n_select = budget.get(rank, 0)

            if n_select == 0:
                continue

            # Find indices in unique_configs for this bucket
            bucket_tensor = torch.stack(bucket_configs)
            bucket_indices = self._find_indices(unique_configs, bucket_tensor)

            # Get weights for this bucket
            bucket_weights = weights[bucket_indices]

            # Select diverse subset from bucket
            if self.config.use_dpp_selection and n_bucket > n_select:
                selected = self._dpp_select(
                    bucket_tensor, bucket_weights, n_select
                )
            else:
                # Simple weighted selection
                selected = self._weighted_select(
                    bucket_indices, bucket_weights, min(n_select, n_bucket)
                )

            selected_indices.extend(selected.tolist())
            bucket_stats[f'rank_{rank}'] = {
                'available': n_bucket,
                'selected': len(selected),
            }

        # Get final selected configurations
        if not selected_indices:
            # Fallback: select top configs by weight
            n_select = min(self.config.max_configs, n_unique)
            _, top_indices = torch.topk(weights, n_select)
            selected_indices = top_indices.tolist()

        selected = unique_configs[selected_indices]

        stats = {
            'n_input': n_configs,
            'n_unique': n_unique,
            'n_selected': len(selected),
            'bucket_stats': bucket_stats,
        }

        return selected, stats

    def _compute_bucket_budgets(self) -> Dict[int, int]:
        """Compute number of configs to select from each excitation rank."""
        cfg = self.config
        max_configs = cfg.max_configs

        budgets = {
            0: int(max_configs * cfg.rank_0_fraction),
            1: int(max_configs * cfg.rank_1_fraction),
            2: int(max_configs * cfg.rank_2_fraction),
            3: int(max_configs * cfg.rank_3_fraction),
        }

        # Remaining goes to rank 4+
        used = sum(budgets.values())
        budgets[4] = max_configs - used

        return budgets

    def _compute_importance_weights(
        self,
        unique_configs: torch.Tensor,
        nqs_probs: Optional[torch.Tensor],
        local_energies: Optional[torch.Tensor],
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance weights for unique configurations."""
        n_unique = len(unique_configs)
        device = unique_configs.device
        weights = torch.ones(n_unique, device=device)

        if self.config.use_nqs_importance and nqs_probs is not None:
            # Aggregate probabilities for unique configs
            unique_probs = torch.zeros(n_unique, device=device)
            unique_probs.scatter_add_(0, inverse, nqs_probs)
            weights = weights * (unique_probs + 1e-10)

        if self.config.use_energy_importance and local_energies is not None:
            # Lower energy = higher importance
            # Use Boltzmann-like weighting
            unique_energies = torch.zeros(n_unique, device=device)
            counts = torch.zeros(n_unique, device=device)
            unique_energies.scatter_add_(0, inverse, local_energies)
            counts.scatter_add_(0, inverse, torch.ones_like(local_energies))
            unique_energies = unique_energies / (counts + 1e-10)

            # Shift to positive range
            e_min = unique_energies.min()
            e_shifted = unique_energies - e_min + 1.0

            # Inverse energy weighting (lower is better)
            energy_weights = 1.0 / e_shifted
            weights = weights * energy_weights

        return weights

    def _find_indices(
        self,
        all_configs: torch.Tensor,
        subset_configs: torch.Tensor,
    ) -> torch.Tensor:
        """Find indices of subset_configs in all_configs."""
        indices = []
        for i in range(len(subset_configs)):
            matches = (all_configs == subset_configs[i]).all(dim=1)
            idx = torch.where(matches)[0]
            if len(idx) > 0:
                indices.append(idx[0].item())
        return torch.tensor(indices, device=all_configs.device)

    def _weighted_select(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        n_select: int,
    ) -> torch.Tensor:
        """Select top-n by weight."""
        if len(indices) <= n_select:
            return indices

        _, top_local = torch.topk(weights, n_select)
        return indices[top_local]

    def _dpp_select(
        self,
        configs: torch.Tensor,
        weights: torch.Tensor,
        n_select: int,
    ) -> torch.Tensor:
        """
        DPP-inspired selection for diversity.

        Uses greedy approximation to DPP:
        1. Start with highest-weight config
        2. Iteratively add config that maximizes weight * min_distance
        """
        n = len(configs)
        device = configs.device

        if n <= n_select:
            return torch.arange(n, device=device)

        # Compute distance matrix
        distances = compute_hamming_distance_matrix(configs).float()

        # Greedy selection
        selected = []
        remaining = set(range(n))

        # Start with highest weight
        first = weights.argmax().item()
        selected.append(first)
        remaining.remove(first)

        while len(selected) < n_select and remaining:
            best_score = -float('inf')
            best_idx = None

            for idx in remaining:
                # Minimum distance to already selected
                min_dist = distances[idx, selected].min().item()

                # Skip if too close
                if min_dist < self.config.min_hamming_distance:
                    continue

                # Score = weight * distance^scale
                score = weights[idx].item() * (min_dist ** self.config.dpp_kernel_scale)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                # All remaining are too close, pick by weight
                remaining_list = list(remaining)
                remaining_weights = weights[remaining_list]
                best_local = remaining_weights.argmax().item()
                best_idx = remaining_list[best_local]

            selected.append(best_idx)
            remaining.remove(best_idx)

        return torch.tensor(selected, device=device)


def select_diverse_basis(
    configs: torch.Tensor,
    reference: torch.Tensor,
    n_orbitals: int,
    max_configs: int = 2048,
    nqs_probs: Optional[torch.Tensor] = None,
    local_energies: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function for diverse basis selection.

    Args:
        configs: Candidate configurations
        reference: Reference state (e.g., HF)
        n_orbitals: Number of spatial orbitals
        max_configs: Maximum configs to select
        nqs_probs: Optional NQS probabilities
        local_energies: Optional local energies
        **kwargs: Additional DiversityConfig parameters

    Returns:
        selected: Selected configurations
        stats: Selection statistics
    """
    config = DiversityConfig(max_configs=max_configs, **kwargs)
    selector = DiversitySelector(config, reference, n_orbitals)
    return selector.select(configs, nqs_probs, local_energies)


def analyze_basis_diversity(
    configs: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, any]:
    """
    Analyze diversity of a configuration basis.

    Returns statistics about excitation ranks and distances.
    """
    n = len(configs)
    device = configs.device

    # Excitation rank distribution
    ranks = []
    for i in range(n):
        rank = compute_excitation_rank(configs[i], reference.to(device))
        ranks.append(rank)

    rank_counts = {}
    for r in set(ranks):
        rank_counts[f'rank_{r}'] = ranks.count(r)

    # Distance statistics
    if n > 1:
        distances = compute_hamming_distance_matrix(configs)
        # Get upper triangle (excluding diagonal)
        triu_indices = torch.triu_indices(n, n, offset=1, device=device)
        pairwise_dists = distances[triu_indices[0], triu_indices[1]]

        dist_stats = {
            'mean_distance': pairwise_dists.float().mean().item(),
            'min_distance': pairwise_dists.min().item(),
            'max_distance': pairwise_dists.max().item(),
        }
    else:
        dist_stats = {}

    return {
        'n_configs': n,
        'rank_distribution': rank_counts,
        **dist_stats,
    }
