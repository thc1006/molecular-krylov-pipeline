"""Tests for config_integer_hash utility — overflow-safe integer hashing.

Bug M4: `2 ** torch.arange(n_sites, dtype=torch.long)` overflows int64 for
n_sites >= 64 (i.e., 32+ orbitals = 64+ qubits). This module tests the
replacement utility that splits into two halves for large n_sites.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfigIntegerHash:
    """Tests for the config_integer_hash utility function."""

    def test_import(self):
        """Utility function should be importable."""
        from utils.config_hash import config_integer_hash

    # ---- Small systems (n_sites < 64): normal int64 path ----

    def test_small_system_basic(self):
        """n_sites=4 (H2): hashing produces correct integer encoding."""
        from utils.config_hash import config_integer_hash

        # HF state for H2: [1, 0, 1, 0] = 1*8 + 0*4 + 1*2 + 0*1 = 10
        configs = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        hashes = config_integer_hash(configs)
        assert hashes == [10]

    def test_small_system_multiple(self):
        """Multiple configs with n_sites=4 produce distinct hashes."""
        from utils.config_hash import config_integer_hash

        configs = torch.tensor(
            [
                [1, 0, 1, 0],  # 10
                [0, 1, 0, 1],  # 5
                [1, 1, 0, 0],  # 12
                [0, 0, 1, 1],  # 3
            ],
            dtype=torch.float32,
        )
        hashes = config_integer_hash(configs)
        assert hashes == [10, 5, 12, 3]

    def test_small_system_uniqueness(self):
        """All distinct configs produce distinct hashes."""
        from utils.config_hash import config_integer_hash

        # All 2-bit configs
        configs = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=torch.float32,
        )
        hashes = config_integer_hash(configs)
        assert len(set(hashes)) == 4, "4 distinct configs should give 4 distinct hashes"

    def test_small_system_20_qubits(self):
        """N2-scale system (20 qubits) — well below 64 boundary."""
        from utils.config_hash import config_integer_hash

        n_sites = 20
        torch.manual_seed(42)
        configs = (torch.rand(100, n_sites) > 0.5).float()
        hashes = config_integer_hash(configs)
        assert len(hashes) == 100
        # Unique configs should give unique hashes
        unique_configs = torch.unique(configs, dim=0)
        unique_hashes = config_integer_hash(unique_configs)
        assert len(set(unique_hashes)) == len(unique_configs)

    def test_small_system_63_qubits(self):
        """n_sites=63 is the largest that fits in int64 without overflow."""
        from utils.config_hash import config_integer_hash

        n_sites = 63
        # All zeros -> hash 0
        configs = torch.zeros(1, n_sites, dtype=torch.float32)
        hashes = config_integer_hash(configs)
        assert hashes == [0]

        # Single 1 at position 0 (MSB) -> 2^62
        configs = torch.zeros(1, n_sites, dtype=torch.float32)
        configs[0, 0] = 1.0
        hashes = config_integer_hash(configs)
        assert hashes == [2**62]

    # ---- Boundary: n_sites = 64 ----

    def test_boundary_64_qubits(self):
        """n_sites=64 triggers the split-hash path (int64 would overflow at 2^63)."""
        from utils.config_hash import config_integer_hash

        n_sites = 64
        configs = torch.zeros(2, n_sites, dtype=torch.float32)
        configs[0, 0] = 1.0  # Only MSB set
        configs[1, -1] = 1.0  # Only LSB set

        hashes = config_integer_hash(configs)
        assert len(hashes) == 2
        assert hashes[0] != hashes[1], "MSB-only and LSB-only must differ"

    def test_boundary_64_uniqueness(self):
        """Distinct configs at n_sites=64 produce distinct hashes."""
        from utils.config_hash import config_integer_hash

        n_sites = 64
        torch.manual_seed(123)
        configs = (torch.rand(200, n_sites) > 0.5).float()
        unique_configs = torch.unique(configs, dim=0)
        hashes = config_integer_hash(unique_configs)
        assert len(set(hashes)) == len(
            unique_configs
        ), f"Expected {len(unique_configs)} unique hashes, got {len(set(hashes))}"

    def test_boundary_64_all_zeros_and_ones(self):
        """All-zeros and all-ones at n_sites=64 produce different hashes."""
        from utils.config_hash import config_integer_hash

        n_sites = 64
        configs = torch.tensor(
            [
                [0] * n_sites,
                [1] * n_sites,
            ],
            dtype=torch.float32,
        )
        hashes = config_integer_hash(configs)
        assert hashes[0] != hashes[1]

    # ---- Large systems: n_sites = 80 (40 orbitals) ----

    def test_large_system_80_qubits(self):
        """n_sites=80 (40 orbitals, the project's scale-up target)."""
        from utils.config_hash import config_integer_hash

        n_sites = 80
        torch.manual_seed(42)
        configs = (torch.rand(500, n_sites) > 0.5).float()
        unique_configs = torch.unique(configs, dim=0)
        hashes = config_integer_hash(unique_configs)
        assert len(set(hashes)) == len(
            unique_configs
        ), f"Expected {len(unique_configs)} unique hashes, got {len(set(hashes))}"

    def test_large_system_no_int64_overflow(self):
        """Confirm no int64 overflow occurs for n_sites=80."""
        from utils.config_hash import config_integer_hash

        n_sites = 80
        # All ones — would be 2^80 - 1 in naive encoding (overflows int64)
        configs = torch.ones(1, n_sites, dtype=torch.float32)
        # This must NOT raise an overflow error
        hashes = config_integer_hash(configs)
        assert len(hashes) == 1
        # The hash should be a tuple (split hash), not a plain int
        assert isinstance(
            hashes[0], tuple
        ), f"For n_sites >= 64, hash should be a tuple, got {type(hashes[0])}"

    def test_large_system_100_qubits(self):
        """n_sites=100 (50 orbitals) — stress test for very large systems."""
        from utils.config_hash import config_integer_hash

        n_sites = 100
        torch.manual_seed(7)
        configs = (torch.rand(300, n_sites) > 0.5).float()
        unique_configs = torch.unique(configs, dim=0)
        hashes = config_integer_hash(unique_configs)
        assert len(set(hashes)) == len(unique_configs)

    # ---- Return type consistency ----

    def test_return_type_small(self):
        """For n_sites < 64, hashes are plain Python ints."""
        from utils.config_hash import config_integer_hash

        configs = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        hashes = config_integer_hash(configs)
        assert isinstance(hashes[0], int)

    def test_return_type_large(self):
        """For n_sites >= 64, hashes are tuples of two ints."""
        from utils.config_hash import config_integer_hash

        configs = torch.tensor([[1] * 64], dtype=torch.float32)
        hashes = config_integer_hash(configs)
        assert isinstance(hashes[0], tuple)
        assert len(hashes[0]) == 2
        assert isinstance(hashes[0][0], int)
        assert isinstance(hashes[0][1], int)

    # ---- Edge cases ----

    def test_empty_configs(self):
        """Empty config tensor returns empty list."""
        from utils.config_hash import config_integer_hash

        configs = torch.empty(0, 10, dtype=torch.float32)
        hashes = config_integer_hash(configs)
        assert hashes == []

    def test_single_site(self):
        """n_sites=1 is a degenerate case that should still work."""
        from utils.config_hash import config_integer_hash

        configs = torch.tensor([[0], [1]], dtype=torch.float32)
        hashes = config_integer_hash(configs)
        assert hashes == [0, 1]

    # ---- GPU compatibility ----

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_small(self):
        """config_integer_hash works on CUDA tensors (small)."""
        from utils.config_hash import config_integer_hash

        configs = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32, device="cuda")
        hashes = config_integer_hash(configs)
        assert hashes == [10]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_large(self):
        """config_integer_hash works on CUDA tensors (large, n_sites=80)."""
        from utils.config_hash import config_integer_hash

        n_sites = 80
        torch.manual_seed(99)
        configs = (torch.rand(100, n_sites, device="cuda") > 0.5).float()
        unique_configs = torch.unique(configs, dim=0)
        hashes = config_integer_hash(unique_configs)
        assert len(set(hashes)) == len(unique_configs)

    # ---- Hashability: hashes should work as dict keys and set members ----

    def test_hashable_small(self):
        """Small-system int hashes are dict-key compatible."""
        from utils.config_hash import config_integer_hash

        configs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        hashes = config_integer_hash(configs)
        d = {h: i for i, h in enumerate(hashes)}
        assert len(d) == 2

    def test_hashable_large(self):
        """Large-system tuple hashes are dict-key compatible."""
        from utils.config_hash import config_integer_hash

        n_sites = 80
        torch.manual_seed(42)
        configs = (torch.rand(50, n_sites) > 0.5).float()
        unique_configs = torch.unique(configs, dim=0)
        hashes = config_integer_hash(unique_configs)
        d = {h: i for i, h in enumerate(hashes)}
        assert len(d) == len(unique_configs)


class TestConfigIntegerHashConsistencyWithLegacy:
    """Verify the new utility matches legacy behavior for n_sites < 64."""

    def test_matches_legacy_powers_encoding(self):
        """New utility produces identical hashes to the legacy 2**arange pattern."""
        from utils.config_hash import config_integer_hash

        n_sites = 20
        torch.manual_seed(42)
        configs = (torch.rand(50, n_sites) > 0.5).float()

        # Legacy encoding
        powers = (2 ** torch.arange(n_sites, dtype=torch.long)).flip(0)
        legacy_hashes = (configs.long() * powers).sum(dim=1).tolist()

        # New utility
        new_hashes = config_integer_hash(configs)

        assert legacy_hashes == new_hashes

    def test_matches_legacy_float64_encoding(self):
        """New utility matches the float64 matmul encoding for n_sites <= 52."""
        from utils.config_hash import config_integer_hash

        n_sites = 40
        torch.manual_seed(7)
        configs = (torch.rand(50, n_sites) > 0.5).float()

        # Float64 matmul encoding (used in connection_cache.py)
        powers_f64 = 2.0 ** torch.arange(n_sites - 1, -1, -1, dtype=torch.float64)
        legacy_hashes = (configs.double() @ powers_f64).long().tolist()

        new_hashes = config_integer_hash(configs)

        assert legacy_hashes == new_hashes
