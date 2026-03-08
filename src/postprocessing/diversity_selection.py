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

    For sites <= 64: each config is packed into a single int64, returns (n,).
    At 64 sites, the sign bit (bit 63) is used, producing negative values --
    this is correct for XOR+popcount since those operate on bit patterns.

    For sites > 64: configs are split into ceil(sites/64) words, returns (n, num_words).
    Hamming distance = sum of popcount(a_word XOR b_word) for each word.
    Use ``bitpacked_hamming`` for computing Hamming distances from packed tensors.

    Args:
        configs: (n, sites) binary tensor

    Returns:
        (n,) int64 tensor if sites <= 64, or (n, num_words) int64 tensor if sites > 64
    """
    n, sites = configs.shape
    device = configs.device
    if sites <= 64:
        powers = (1 << torch.arange(sites, dtype=torch.int64, device=device)).flip(0)
        return (configs.long() * powers).sum(dim=1)
    else:
        # Split into 64-bit words. XOR+popcount works on bit patterns,
        # so the sign bit is fine.
        word_size = 64
        num_words = (sites + word_size - 1) // word_size
        words = []
        for w in range(num_words):
            start = w * word_size
            end = min(start + word_size, sites)
            chunk_size = end - start
            powers = (1 << torch.arange(chunk_size, dtype=torch.int64, device=device)).flip(0)
            words.append((configs[:, start:end].long() * powers).sum(dim=1))
        return torch.stack(words, dim=1)


def bitpacked_hamming(
    packed: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    """Compute Hamming distances between bitpacked config pairs.

    Works for both single-word (n,) and multi-word (n, num_words) packed tensors.

    Args:
        packed: packed configs from ``bitpack_configs``
        idx_a: (k,) indices for first set of configs
        idx_b: (k,) indices for second set of configs

    Returns:
        (k,) int32 tensor of Hamming distances
    """
    if packed.dim() == 1:
        xor = packed[idx_a] ^ packed[idx_b]
        return _popcount_int64(xor)
    else:
        xor = packed[idx_a] ^ packed[idx_b]  # (k, num_words)
        return sum(_popcount_int64(xor[:, w]) for w in range(xor.shape[1]))


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
    """Hamming distance from packed[idx] to packed[indices]. Vectorized O(k).

    Works for both single-word (n,) and multi-word (n, num_words) packed tensors.
    """
    xor = packed[indices] ^ packed[idx]
    if packed.dim() == 1:
        return _popcount_int64(xor)
    else:
        return sum(_popcount_int64(xor[:, w]) for w in range(xor.shape[1]))


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

    # Ensure at least 1-dim; unsqueeze scalar tensors
    was_scalar = x.dim() == 0
    if was_scalar:
        x = x.unsqueeze(0)

    # Extract 8 bytes per int64 via bit-shifts (avoids view(dtype) which is unsupported)
    shifts = torch.tensor(
        [0, 8, 16, 24, 32, 40, 48, 56], device=x.device, dtype=torch.long,
    )
    x_bytes = ((x.unsqueeze(-1) >> shifts) & 0xFF).long()
    counts = table[x_bytes].sum(dim=-1)
    result = counts.int()
    if was_scalar:
        result = result.squeeze(0)
    return result


def _hamming_from_one(packed: torch.Tensor, idx: int) -> torch.Tensor:
    """Compute Hamming distances from packed[idx] to all rows.

    Works for both single-word (n,) and multi-word (n, num_words) packed tensors.

    Returns:
        (n,) int32 tensor of distances
    """
    xor = packed ^ packed[idx]
    if packed.dim() == 1:
        return _popcount_int64(xor)
    else:
        return sum(_popcount_int64(xor[:, w]) for w in range(xor.shape[1]))


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
    dists = _hamming_from_one(packed, first)
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
        new_dists = _hamming_from_one(packed, best_idx)
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
                # _dpp_select returns LOCAL indices into bucket_tensor (0..n_bucket-1)
                local_selected = self._dpp_select(
                    bucket_tensor, bucket_weights, n_select
                )
                # Map local → global indices into unique_configs
                selected = bucket_indices[local_selected]
            else:
                # _weighted_select already returns global indices
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

        # Remaining goes to rank 4+ (shared across all ranks >= 4)
        used = sum(budgets.values())
        rank4_plus_budget = max_configs - used

        # Distribute rank 4+ budget across all ranks >= 4 present in buckets
        high_ranks = sorted(r for r in self.bucketer.buckets.keys() if r >= 4)
        if high_ranks:
            high_rank_counts = {r: len(self.bucketer.get_bucket(r)) for r in high_ranks}
            total_high = sum(high_rank_counts.values())
            if total_high > 0:
                for r in high_ranks:
                    budgets[r] = max(1, int(rank4_plus_budget * high_rank_counts[r] / total_high))
            else:
                budgets[4] = rank4_plus_budget
        else:
            budgets[4] = rank4_plus_budget

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
        """Find indices of subset_configs in all_configs.

        Uses hash-based O(n + m) lookup instead of O(n * m) pairwise comparison.
        Overflow-safe: handles n_sites >= 64 via config_integer_hash.
        """
        n_sub = len(subset_configs)
        if n_sub == 0:
            return torch.tensor([], dtype=torch.long, device=all_configs.device)

        try:
            from utils.config_hash import config_integer_hash
        except ImportError:
            from ..utils.config_hash import config_integer_hash

        all_hashes = config_integer_hash(all_configs)
        sub_hashes = config_integer_hash(subset_configs)

        hash_to_idx: dict = {}
        for i, h in enumerate(all_hashes):
            if h not in hash_to_idx:
                hash_to_idx[h] = i

        indices = [hash_to_idx[h] for h in sub_hashes if h in hash_to_idx]

        return torch.tensor(indices, dtype=torch.long, device=all_configs.device)

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
        DPP-inspired diversity selection using streaming greedy.

        Delegates to stochastic_greedy_select() which uses bitpacked
        Hamming + incremental min-distance tracking. Memory O(n) instead
        of O(n²). Time O(n·log(1/ε)) instead of O(n·k).
        """
        n = len(configs)
        device = configs.device

        if n <= n_select:
            return torch.arange(n, device=device)

        return stochastic_greedy_select(
            configs,
            weights,
            n_select,
            min_hamming=self.config.min_hamming_distance,
        )


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
    # For large bases, use sampled pairs to avoid O(n²) memory
    SAMPLED_DISTANCE_THRESHOLD = 5000
    if n > 1:
        if n <= SAMPLED_DISTANCE_THRESHOLD:
            distances = compute_hamming_distance_matrix(configs)
            triu_indices = torch.triu_indices(n, n, offset=1, device=device)
            pairwise_dists = distances[triu_indices[0], triu_indices[1]]
        else:
            # Sample random pairs for distance statistics
            n_samples = min(50000, n * (n - 1) // 2)
            packed = bitpack_configs(configs)
            gen = torch.Generator(device='cpu')
            gen.manual_seed(42)
            idx_a = torch.randint(0, n, (n_samples,), generator=gen)
            idx_b = torch.randint(0, n, (n_samples,), generator=gen)
            # Re-draw pairs where idx_a == idx_b
            same = idx_a == idx_b
            idx_b[same] = (idx_b[same] + 1) % n
            pairwise_dists = bitpacked_hamming(packed, idx_a, idx_b)

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
