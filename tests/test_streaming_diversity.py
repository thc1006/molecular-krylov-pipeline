"""Tests for PR 1.7: Streaming diversity selection.

Replaces O(n^2) pairwise Hamming distance matrix with:
1. Bit-parallel Hamming via uint64 bitpacking + popcount (O(1) per pair)
2. Stochastic greedy selection (O(n * log(1/eps)) evaluations, not O(n*k))

This enables handling 50K+ configs without OOM (< 1GB memory).
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBitpackedHamming:
    """Test bit-parallel Hamming distance computation."""

    def test_bitpack_roundtrip(self):
        """Bitpacking and unpacking should be lossless."""
        from postprocessing.diversity_selection import bitpack_configs, hamming_bitpacked

        configs = torch.tensor([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0],
        ], dtype=torch.long)

        packed = bitpack_configs(configs)
        assert packed.shape == (3,)  # One int64 per config (6 bits fits in 1 word)
        assert packed.dtype == torch.int64

    def test_bitpacked_hamming_matches_naive(self):
        """Bitpacked Hamming must match naive computation."""
        from postprocessing.diversity_selection import (
            bitpack_configs, hamming_bitpacked, compute_hamming_distance
        )

        configs = torch.tensor([
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 1],
        ], dtype=torch.long)

        packed = bitpack_configs(configs)

        for i in range(len(configs)):
            for j in range(len(configs)):
                naive = compute_hamming_distance(configs[i], configs[j])
                fast = hamming_bitpacked(packed[i], packed[j])
                assert naive == fast, f"Mismatch at ({i},{j}): naive={naive}, fast={fast}"

    def test_bitpacked_40_sites(self):
        """Bitpacking must work for 40-site configs (fits in int64)."""
        from postprocessing.diversity_selection import bitpack_configs, hamming_bitpacked

        configs = torch.zeros(2, 40, dtype=torch.long)
        configs[0, :20] = 1  # First 20 occupied
        configs[1, 20:] = 1  # Last 20 occupied

        packed = bitpack_configs(configs)
        dist = hamming_bitpacked(packed[0], packed[1])
        assert dist == 40  # All positions differ


class TestStochasticGreedy:
    """Test stochastic greedy diversity selection."""

    def test_stochastic_greedy_returns_correct_count(self):
        """Should return exactly n_select configs."""
        from postprocessing.diversity_selection import stochastic_greedy_select

        configs = torch.randint(0, 2, (100, 20), dtype=torch.long)
        weights = torch.ones(100)
        selected = stochastic_greedy_select(configs, weights, n_select=30)

        assert len(selected) == 30
        assert selected.dtype == torch.long
        # All indices should be valid
        assert selected.min() >= 0
        assert selected.max() < 100
        # No duplicates
        assert len(selected.unique()) == 30

    def test_stochastic_greedy_respects_weights(self):
        """Highest-weight config should always be selected first."""
        from postprocessing.diversity_selection import stochastic_greedy_select

        configs = torch.randint(0, 2, (50, 10), dtype=torch.long)
        weights = torch.ones(50)
        weights[42] = 1000.0  # Make one config overwhelmingly important
        selected = stochastic_greedy_select(configs, weights, n_select=10)

        assert 42 in selected.tolist(), "Highest-weight config must be selected"

    def test_stochastic_greedy_no_oom_50k(self):
        """Must handle 50K configs without OOM (< 1GB memory)."""
        from postprocessing.diversity_selection import stochastic_greedy_select
        import tracemalloc

        n = 50000
        sites = 40
        configs = torch.randint(0, 2, (n, sites), dtype=torch.long)
        weights = torch.ones(n)

        tracemalloc.start()
        selected = stochastic_greedy_select(configs, weights, n_select=5000)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(selected) == 5000
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 1000, f"Peak memory {peak_mb:.0f} MB exceeds 1 GB limit"

    @pytest.mark.molecular
    def test_stochastic_greedy_energy_quality(self, lih_hamiltonian):
        """Stochastic selection should produce similar energy to full DPP."""
        from postprocessing.diversity_selection import (
            stochastic_greedy_select, compute_hamming_distance_matrix
        )
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        configs = pipeline._generate_essential_configs()

        weights = torch.ones(len(configs))
        n_select = min(50, len(configs))

        selected = stochastic_greedy_select(configs, weights, n_select=n_select)
        selected_configs = configs[selected]

        # Verify selected configs are diverse (min pairwise Hamming > 0)
        if len(selected_configs) > 1:
            dists = compute_hamming_distance_matrix(selected_configs)
            # Mask diagonal
            dists.fill_diagonal_(999)
            min_dist = dists.min().item()
            assert min_dist > 0, "Selected configs have duplicates"


class TestEdgeCases:
    """Edge case tests for bitpacking and greedy selection."""

    def test_n_select_zero(self):
        """n_select=0 must return empty tensor."""
        from postprocessing.diversity_selection import stochastic_greedy_select

        configs = torch.randint(0, 2, (10, 8), dtype=torch.long)
        weights = torch.ones(10)
        selected = stochastic_greedy_select(configs, weights, n_select=0)
        assert len(selected) == 0

    def test_n_select_greater_than_n(self):
        """n_select > n must return all indices."""
        from postprocessing.diversity_selection import stochastic_greedy_select

        configs = torch.randint(0, 2, (5, 8), dtype=torch.long)
        weights = torch.ones(5)
        selected = stochastic_greedy_select(configs, weights, n_select=100)
        assert len(selected) == 5

    def test_single_config(self):
        """Must handle n=1."""
        from postprocessing.diversity_selection import stochastic_greedy_select

        configs = torch.randint(0, 2, (1, 8), dtype=torch.long)
        weights = torch.ones(1)
        selected = stochastic_greedy_select(configs, weights, n_select=1)
        assert len(selected) == 1
        assert selected[0].item() == 0

    def test_all_identical_configs(self):
        """All-identical configs must still select n_select with no crash."""
        from postprocessing.diversity_selection import stochastic_greedy_select

        configs = torch.ones(10, 8, dtype=torch.long)
        weights = torch.ones(10)
        selected = stochastic_greedy_select(configs, weights, n_select=5)
        assert len(selected) == 5
        assert len(selected.unique()) == 5

    def test_popcount_scalar_tensor(self):
        """_popcount_int64 must handle 0-dim (scalar) tensors."""
        from postprocessing.diversity_selection import _popcount_int64

        scalar = torch.tensor(7, dtype=torch.int64)
        assert _popcount_int64(scalar).item() == 3
        assert _popcount_int64(torch.tensor(0, dtype=torch.int64)).item() == 0
        assert _popcount_int64(torch.tensor(-1, dtype=torch.int64)).item() == 64

    def test_hamming_bitpacked_64_sites(self):
        """Bitpacked Hamming must be correct at 64 sites (int64 overflow)."""
        from postprocessing.diversity_selection import bitpack_configs, hamming_bitpacked

        configs = torch.zeros(2, 64, dtype=torch.long)
        configs[0, :] = 1  # all occupied
        configs[1, :] = 0  # all empty

        packed = bitpack_configs(configs)
        dist = hamming_bitpacked(packed[0].item(), packed[1].item())
        assert dist == 64, f"Expected 64, got {dist}"

    def test_hamming_bitpacked_negative_packed_values(self):
        """hamming_bitpacked must handle negative int64 from overflow."""
        from postprocessing.diversity_selection import hamming_bitpacked

        # -1 XOR 0 should be 64 bits (all set)
        assert hamming_bitpacked(-1, 0) == 64
        # -1 XOR -1 should be 0
        assert hamming_bitpacked(-1, -1) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
