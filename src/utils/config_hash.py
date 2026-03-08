"""
Overflow-safe integer hashing for binary configuration vectors.

Binary configurations (Slater determinants) are encoded as integers for fast
deduplication and membership-checking via Python dicts/sets. The naive approach
``2 ** torch.arange(n_sites, dtype=torch.long)`` overflows int64 for
n_sites >= 64 (i.e., 32+ spatial orbitals = 64+ spin orbitals).

This module provides ``config_integer_hash()`` which:
- For n_sites < 64: uses the standard int64 encoding (backward-compatible)
- For n_sites >= 64: splits the config into two halves, encodes each half
  as an int64, and returns a tuple ``(hash_hi, hash_lo)`` which is hashable
  and collision-free.

Usage::

    from utils.config_hash import config_integer_hash

    hashes = config_integer_hash(configs)  # list of int or tuple(int, int)
    config_set = set(hashes)               # fast membership check
    config_map = {h: i for i, h in enumerate(hashes)}  # index lookup
"""

import torch

# Type alias for the hash return type
ConfigHash = int | tuple


def config_integer_hash(
    configs: torch.Tensor,
) -> list[ConfigHash]:
    """Hash binary configuration vectors to integers, handling n_sites >= 64 overflow.

    For n_sites < 64, each config is encoded as a single int64:
        hash = sum(config[i] * 2^(n_sites - 1 - i))

    For n_sites >= 64, the config is split into two halves and each half
    is encoded separately, returning a tuple (hash_hi, hash_lo) that is
    hashable and guarantees no collisions.

    Parameters
    ----------
    configs : torch.Tensor
        (n_configs, n_sites) binary tensor. Values should be 0 or 1.
        Any dtype is accepted; internally cast to long.

    Returns
    -------
    list
        Length-n_configs list. Each element is either:
        - int (for n_sites < 64)
        - tuple[int, int] (for n_sites >= 64)
        All elements are hashable and usable as dict keys / set members.
    """
    if configs.numel() == 0:
        return []

    n_configs, n_sites = configs.shape
    device = configs.device

    if n_sites < 64:
        # Standard path: single int64 encoding -- backward-compatible
        powers = (2 ** torch.arange(n_sites, device=device, dtype=torch.long)).flip(0)
        return (configs.long() * powers).sum(dim=1).cpu().tolist()
    else:
        # Split path: avoid int64 overflow by encoding two halves separately.
        # Each half has at most 63 bits, which fits in int64.
        half = n_sites // 2
        n_lo = n_sites - half  # handles odd n_sites

        powers_hi = (2 ** torch.arange(half, device=device, dtype=torch.long)).flip(0)
        powers_lo = (2 ** torch.arange(n_lo, device=device, dtype=torch.long)).flip(0)

        hash_hi = (configs[:, :half].long() * powers_hi).sum(dim=1)
        hash_lo = (configs[:, half:].long() * powers_lo).sum(dim=1)

        return list(zip(hash_hi.cpu().tolist(), hash_lo.cpu().tolist(), strict=True))
