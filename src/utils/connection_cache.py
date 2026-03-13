"""
GPU-Accelerated Connection Caching for Hamiltonian Operations.

This module provides caching mechanisms to avoid recomputing Hamiltonian
connections for configurations that have been seen before.

Key optimizations:
- GPU-based integer encoding using tensor operations (no CPU transfer)
- LRU-style eviction for memory management
- Batch operations with parallel processing
- Adaptive bypass mode when cache hit rate is low
"""

import torch
from typing import Dict, Tuple, Optional, List


def compute_max_cache_size(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    num_sites: int = 0,
    memory_budget_mb: float = 8192.0,
) -> int:
    """
    Compute max ConnectionCache entries that fit within a memory budget.

    At 52Q (n_orb=26, n_alpha=5, n_beta=5): each entry stores ~15,435
    connections × (52 sites × 8 + 8) bytes = ~6.6 MB. The default 100K
    entries would need 666 GB — far exceeding 128 GB UMA.

    This function computes the combinatorial upper bound on connections
    per config (singles + same-spin doubles + alpha-beta doubles) and
    sizes the cache accordingly.

    Args:
        n_orbitals: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
        num_sites: Number of spin-orbital sites (default: 2 * n_orbitals)
        memory_budget_mb: Target memory budget in MB (default 8192 = 8 GB)

    Returns:
        Maximum number of cache entries that fit within the budget.
    """
    from math import comb

    if num_sites <= 0:
        num_sites = 2 * n_orbitals

    # Upper-bound connections per config (Slater-Condon rules)
    singles = n_alpha * (n_orbitals - n_alpha) + n_beta * (n_orbitals - n_beta)
    same_doubles = (
        comb(n_alpha, 2) * comb(n_orbitals - n_alpha, 2)
        + comb(n_beta, 2) * comb(n_orbitals - n_beta, 2)
    )
    ab_doubles = (
        n_alpha * (n_orbitals - n_alpha) * n_beta * (n_orbitals - n_beta)
    )
    avg_conn = singles + same_doubles + ab_doubles

    # Memory per entry: connected configs tensor + elements tensor + Python overhead
    # connected: (n_conn, num_sites) int64 → n_conn * num_sites * 8
    # elements:  (n_conn,) float64         → n_conn * 8
    # Python dict/LRU overhead             → ~200 bytes per entry
    bytes_per_entry = avg_conn * (num_sites * 8 + 8) + 200

    budget_bytes = memory_budget_mb * 1024 * 1024
    max_entries = int(budget_bytes / max(bytes_per_entry, 1))

    # Floor at 100 entries (cache is still useful for HF + singles)
    return max(100, max_entries)


class ConnectionCache:
    """
    Cache for Hamiltonian connections using GPU-accelerated integer encoding.

    Key insight: For a fixed Hamiltonian, connections for a configuration
    never change. Caching them avoids expensive recomputation.

    Uses GPU tensor operations for fast batch encoding:
    configs @ powers -> integer keys (single matmul, no Python loops)

    Args:
        num_sites: Number of sites/qubits
        max_cache_size: Maximum number of cached entries
        device: Torch device
        bypass_threshold: Hit rate below which cache is bypassed (default 0.3)
    """

    def __init__(
        self,
        num_sites: int,
        max_cache_size: int = 100000,
        device: str = 'cuda',
        bypass_threshold: float = 0.3,
    ):
        self.num_sites = num_sites
        self.max_cache_size = max_cache_size
        self.device = device
        self.bypass_threshold = bypass_threshold

        # Powers of 2 for integer encoding - precomputed on GPU.
        # For num_sites <= 52, float64 matmul is exact (53-bit mantissa).
        # For num_sites >= 53, use split encoding via config_integer_hash
        # to avoid hash collisions from float64 precision loss.
        self._use_split_hash = num_sites >= 53
        if not self._use_split_hash:
            # Float64 powers for CUDA matmul compatibility (CUDA doesn't
            # support matmul for int64). Safe for num_sites <= 52 (53-bit mantissa).
            # For 53-63 sites: float64 loses low bits but config_integer_hash
            # is used as primary path in _encode_batch.
            self.powers_gpu = (2.0 ** torch.arange(
                num_sites - 1, -1, -1, device=device, dtype=torch.float64
            ))
            self.powers_cpu = self.powers_gpu.cpu()
        else:
            self.powers_gpu = None
            self.powers_cpu = None

        # Cache storage: key -> (connected_configs, matrix_elements)
        # Keys are int (n_sites < 64) or tuple[int,int] (n_sites >= 64)
        self._cache: Dict = {}

        # Access counter for LRU eviction
        self._access_count: Dict[int, int] = {}
        self._total_accesses = 0

        # Statistics
        self.hits = 0
        self.misses = 0

        # Bypass mode tracking
        self._recent_hits = 0
        self._recent_total = 0
        self._bypass_check_interval = 100

    def _encode_config(self, config: torch.Tensor):
        """Encode a single configuration as a hashable key (GPU-accelerated).

        Returns int for num_sites < 64, or tuple[int,int] for num_sites >= 64.
        """
        if self._use_split_hash:
            from .config_hash import config_integer_hash
            return config_integer_hash(config.unsqueeze(0))[0]
        if config.device != self.powers_gpu.device:
            # Use CPU version if config is on CPU
            return int((config.double().cpu() * self.powers_cpu).sum().item())
        # Use double precision (powers_gpu is already float64)
        return int((config.double() @ self.powers_gpu).item())

    def _encode_batch_gpu(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of configurations as integers using GPU matmul.

        This is 10-50x faster than the CPU Python loop version.

        Note: For num_sites >= 64, this falls through to _encode_batch
        which uses config_integer_hash (returns list, not tensor).

        Args:
            configs: (n_configs, num_sites) tensor on GPU

        Returns:
            (n_configs,) tensor of integer keys (only for num_sites < 64)
        """
        if self._use_split_hash:
            raise RuntimeError(
                "Cannot use _encode_batch_gpu for num_sites >= 64; "
                "use _encode_batch instead."
            )
        configs_gpu = configs.to(self.device, dtype=torch.float64)
        return (configs_gpu @ self.powers_gpu).long()

    def _encode_batch(self, configs: torch.Tensor) -> List:
        """Encode batch of configurations as hashable keys (returns Python list).

        Returns list of int for num_sites < 64, or list of tuple[int,int]
        for num_sites >= 64.
        """
        if self._use_split_hash:
            from .config_hash import config_integer_hash
            return config_integer_hash(configs)
        keys_tensor = self._encode_batch_gpu(configs)
        return keys_tensor.tolist()

    def get(
        self, config: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached connections for a configuration.

        Args:
            config: (num_sites,) configuration tensor

        Returns:
            (connected, elements) if cached, None otherwise
        """
        key = self._encode_config(config)

        if key in self._cache:
            self.hits += 1
            self._recent_hits += 1
            self._recent_total += 1
            self._total_accesses += 1
            self._access_count[key] = self._total_accesses
            return self._cache[key]

        self.misses += 1
        self._recent_total += 1
        return None

    def put(
        self,
        config: torch.Tensor,
        connected: torch.Tensor,
        elements: torch.Tensor
    ):
        """
        Cache connections for a configuration.

        Args:
            config: (num_sites,) configuration tensor
            connected: (n_connections, num_sites) connected configurations
            elements: (n_connections,) matrix elements
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_cache_size:
            self._evict()

        key = self._encode_config(config)
        self._total_accesses += 1
        self._access_count[key] = self._total_accesses

        # Store on GPU for fast retrieval
        self._cache[key] = (
            connected.to(self.device) if len(connected) > 0 else connected,
            elements.to(self.device) if len(elements) > 0 else elements
        )

    def get_or_compute(
        self,
        config: torch.Tensor,
        hamiltonian,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get connections from cache or compute and cache them.

        Args:
            config: (num_sites,) configuration tensor
            hamiltonian: Hamiltonian object with get_connections method

        Returns:
            (connected, elements) tuple
        """
        cached = self.get(config)
        if cached is not None:
            return cached

        # Compute and cache
        connected, elements = hamiltonian.get_connections(config)
        self.put(config, connected, elements)
        return connected, elements

    def should_bypass(self) -> bool:
        """Check if cache should be bypassed due to low hit rate.

        NOTE: Do NOT increment _recent_total here — it is already incremented
        in get() on cache miss (line 152) and _recent_hits on cache hit (line 146).
        Double-counting would inflate the denominator and trigger premature bypass.
        """
        if self._recent_total >= self._bypass_check_interval:
            recent_hit_rate = self._recent_hits / self._recent_total
            # Reset counters
            self._recent_hits = 0
            self._recent_total = 0
            return recent_hit_rate < self.bypass_threshold
        return False

    def get_batch(
        self,
        configs: torch.Tensor,
        hamiltonian,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get connections for a batch of configurations, using cache where available.

        Optimized version with GPU-accelerated encoding and adaptive bypass.

        Args:
            configs: (n_configs, num_sites) configurations
            hamiltonian: Hamiltonian object

        Returns:
            all_connected: (total_connections, num_sites) all connected configs
            all_elements: (total_connections,) all matrix elements
            config_indices: (total_connections,) which original config each belongs to
        """
        n_configs = len(configs)
        device = self.device

        # Batch encoding (overflow-safe for num_sites >= 64)
        keys = self._encode_batch(configs)

        all_connected = []
        all_elements = []
        all_indices = []

        # Track which configs need computation
        configs_to_compute = []
        configs_to_compute_indices = []

        # First pass: check cache for all configs
        for i, key in enumerate(keys):
            if key in self._cache:
                self.hits += 1
                self._total_accesses += 1
                self._access_count[key] = self._total_accesses
                connected, elements = self._cache[key]

                n_conn = len(connected)
                if n_conn > 0:
                    all_connected.append(connected)
                    all_elements.append(elements)
                    all_indices.append(
                        torch.full((n_conn,), i, dtype=torch.long, device=device)
                    )
            else:
                self.misses += 1
                configs_to_compute.append(i)
                configs_to_compute_indices.append((i, key))

        # Second pass: compute missing configs
        # Check if hamiltonian has batched method
        if len(configs_to_compute) > 0:
            if hasattr(hamiltonian, 'get_connections_batch') and len(configs_to_compute) > 10:
                # Use batched computation if available and worthwhile
                compute_configs = configs[configs_to_compute]
                batch_connected, batch_elements, batch_indices = \
                    hamiltonian.get_connections_batch(compute_configs)

                # Ensure batch_indices is long for indexing
                batch_indices_long = batch_indices.long() if len(batch_indices) > 0 else batch_indices

                # Remap indices to original positions
                if len(batch_connected) > 0:
                    original_indices = torch.tensor(
                        configs_to_compute, device=device, dtype=torch.long
                    )
                    remapped_indices = original_indices[batch_indices_long]
                    all_connected.append(batch_connected)
                    all_elements.append(batch_elements)
                    all_indices.append(remapped_indices)

                # Cache individual results (O(n) lookup via reverse map)
                idx_to_local = {idx: local for local, idx in enumerate(configs_to_compute)}
                for idx, key in configs_to_compute_indices:
                    mask = batch_indices_long == idx_to_local[idx]
                    if mask.sum() > 0:
                        self._cache[key] = (
                            batch_connected[mask],
                            batch_elements[mask]
                        )
                        self._total_accesses += 1
                        self._access_count[key] = self._total_accesses
            else:
                # Fall back to serial computation
                for idx, key in configs_to_compute_indices:
                    connected, elements = hamiltonian.get_connections(configs[idx])

                    # Cache result
                    if len(self._cache) < self.max_cache_size:
                        self._cache[key] = (
                            connected.to(device) if len(connected) > 0 else connected,
                            elements.to(device) if len(elements) > 0 else elements
                        )
                        self._total_accesses += 1
                        self._access_count[key] = self._total_accesses

                    n_conn = len(connected)
                    if n_conn > 0:
                        all_connected.append(connected.to(device))
                        all_elements.append(elements.to(device))
                        all_indices.append(
                            torch.full((n_conn,), idx, dtype=torch.long, device=device)
                        )

        if not all_connected:
            return (
                torch.empty(0, self.num_sites, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        return (
            torch.cat(all_connected, dim=0),
            torch.cat(all_elements, dim=0),
            torch.cat(all_indices, dim=0)
        )

    def _evict(self):
        """Evict least recently used entries."""
        # Remove bottom 20% by access time
        n_evict = max(1, self.max_cache_size // 5)

        # Sort by access count (oldest first)
        sorted_keys = sorted(self._access_count.keys(),
                           key=lambda k: self._access_count[k])

        for key in sorted_keys[:n_evict]:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_count.clear()
        self.hits = 0
        self.misses = 0
        self._total_accesses = 0
        self._recent_hits = 0
        self._recent_total = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
        }
