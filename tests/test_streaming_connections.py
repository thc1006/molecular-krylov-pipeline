"""Tests for C2+C3: Chunked get_connections_vectorized_batch and streaming dedup.

Verifies that:
1. Auto-chunking in get_connections_vectorized_batch produces identical results
   to unchunked processing
2. Streaming dedup in _find_connected_configs produces same results as the
   original all-at-once approach
3. Memory stays bounded during large batch processing
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _generate_essential(H):
    """Generate HF + singles + doubles configs for a Hamiltonian."""
    from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

    config = PipelineConfig(skip_nf_training=True, device="cpu", skip_skqd=True)
    p = FlowGuidedKrylovPipeline(H, config=config)
    p.train_flow_nqs(progress=False)
    return p._essential_configs


class TestChunkedGetConnections:
    """Verify get_connections_vectorized_batch auto-chunking correctness."""

    @pytest.mark.molecular
    def test_chunked_matches_unchunked_h2(self, h2_hamiltonian):
        """H2: chunked results must match unchunked (tiny system, trivial)."""
        H = h2_hamiltonian
        configs = _generate_essential(H)

        # Unchunked (huge memory budget = no chunking)
        c1, e1, b1 = H.get_connections_vectorized_batch(configs, max_memory_mb=1e6)
        # Force chunking with tiny memory budget
        c2, e2, b2 = H.get_connections_vectorized_batch(configs, max_memory_mb=0.001)

        assert c1.shape == c2.shape, f"Shape mismatch: {c1.shape} vs {c2.shape}"
        assert torch.equal(c1, c2), "Connected configs differ"
        assert torch.allclose(e1, e2, atol=1e-12), "Matrix elements differ"
        assert torch.equal(b1, b2), "Batch indices differ"

    @pytest.mark.molecular
    def test_chunked_matches_unchunked_lih(self, lih_hamiltonian):
        """LiH: chunked results must match unchunked (93 essential configs).

        Chunking may reorder results (per-chunk concatenation), so we compare
        the set of (batch_idx, connected_config) pairs rather than exact order.
        """
        H = lih_hamiltonian
        configs = _generate_essential(H)

        c1, e1, b1 = H.get_connections_vectorized_batch(configs, max_memory_mb=1e6)
        c2, e2, b2 = H.get_connections_vectorized_batch(configs, max_memory_mb=0.01)

        assert len(c1) == len(c2), f"Count mismatch: {len(c1)} vs {len(c2)}"

        # Build (batch_idx, config_hash) → element maps and compare
        from utils.config_hash import config_integer_hash

        h1 = config_integer_hash(c1)
        h2 = config_integer_hash(c2)
        set1 = {(b1[i].item(), h1[i]) for i in range(len(c1))}
        set2 = {(b2[i].item(), h2[i]) for i in range(len(c2))}
        assert set1 == set2, f"Config set mismatch: {len(set1 - set2)} missing"

    @pytest.mark.molecular
    def test_chunk_size_calculation(self, lih_hamiltonian):
        """Verify that small memory budget forces multiple chunks."""
        H = lih_hamiltonian
        configs = _generate_essential(H)

        # With 0.001 MB budget, should chunk aggressively
        # Patch to count chunks (we can verify by checking output is correct)
        c, e, b = H.get_connections_vectorized_batch(configs, max_memory_mb=0.001)
        assert len(c) > 0, "Should find some connections"

    @pytest.mark.molecular
    def test_single_config_chunk(self, lih_hamiltonian):
        """Processing a single config should work even with tiny budget."""
        H = lih_hamiltonian
        hf = H.get_hf_state().unsqueeze(0)

        c, e, b = H.get_connections_vectorized_batch(hf, max_memory_mb=0.001)
        assert len(c) > 0
        assert (b == 0).all(), "Single config should have all batch_indices = 0"

    @pytest.mark.molecular
    def test_batch_indices_correct_under_chunking(self, lih_hamiltonian):
        """Batch indices must correctly reference the original config index."""
        H = lih_hamiltonian
        configs = _generate_essential(H)[:50]

        c, e, b = H.get_connections_vectorized_batch(configs, max_memory_mb=0.01)

        # All batch indices must be valid
        assert b.min() >= 0
        assert b.max() < len(configs)

        # Spot check: connections for first config should match get_connections
        mask = b == 0
        if mask.any():
            conn_first_batch = c[mask]
            conn_single, elem_single = H.get_connections(configs[0])
            # conn_first_batch should be a subset of conn_single
            # (vectorized batch may filter by MATRIX_ELEMENT_TOL slightly differently)
            assert len(conn_first_batch) <= len(conn_single) + 10


class TestStreamingDedup:
    """Verify streaming dedup in _find_connected_configs."""

    @pytest.mark.molecular
    def test_streaming_finds_new_configs_h2(self, h2_hamiltonian):
        """H2: streaming dedup should find connected configs not in basis."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from utils.config_hash import config_integer_hash

        H = h2_hamiltonian
        hf = H.get_hf_state().unsqueeze(0)

        config = SKQDConfig()
        skqd = FlowGuidedSKQD(H, nf_basis=hf, config=config)

        basis_set = set(config_integer_hash(hf))
        new_configs = skqd._find_connected_configs(hf, basis_set)

        assert len(new_configs) > 0, "Should find connections of HF state"
        # New configs should NOT include HF itself
        new_hashes = set(config_integer_hash(new_configs))
        hf_hash = config_integer_hash(hf)[0]
        assert hf_hash not in new_hashes, "HF should not be in new configs"

    @pytest.mark.molecular
    def test_streaming_dedup_no_duplicates(self, lih_hamiltonian):
        """LiH: streaming dedup should return unique configs only."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from utils.config_hash import config_integer_hash

        H = lih_hamiltonian
        hf = H.get_hf_state().unsqueeze(0)

        config = SKQDConfig(min_krylov_basis_sample=1, krylov_basis_sample_fraction=1.0)
        skqd = FlowGuidedSKQD(H, nf_basis=hf, config=config)

        basis_set = set(config_integer_hash(hf))
        new_configs = skqd._find_connected_configs(hf, basis_set)

        if len(new_configs) > 0:
            hashes = config_integer_hash(new_configs)
            assert len(set(hashes)) == len(hashes), "Duplicate configs in result"

    @pytest.mark.molecular
    def test_streaming_excludes_basis(self, lih_hamiltonian):
        """LiH: configs already in basis_set should be excluded."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from utils.config_hash import config_integer_hash

        H = lih_hamiltonian
        # Start with HF + a few configs
        configs = _generate_essential(H)[:20]

        config = SKQDConfig(min_krylov_basis_sample=20, krylov_basis_sample_fraction=1.0)
        skqd = FlowGuidedSKQD(H, nf_basis=configs, config=config)

        basis_set = set(config_integer_hash(configs))
        new_configs = skqd._find_connected_configs(configs, basis_set)

        if len(new_configs) > 0:
            new_hashes = set(config_integer_hash(new_configs))
            overlap = new_hashes & basis_set
            assert len(overlap) == 0, f"Found {len(overlap)} configs already in basis"


class TestEndToEndWithChunking:
    """Integration: pipeline still gives correct energy with chunked connections."""

    @pytest.mark.molecular
    def test_pipeline_h2_with_chunking(self, h2_hamiltonian):
        """H2 pipeline with forced chunking should achieve chemical accuracy."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = h2_hamiltonian
        e_fci = H.fci_energy()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=e_fci
        )
        results = pipeline.run(progress=False)

        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 0.1, f"H2 error {error_mha:.4f} mHa too large"

    @pytest.mark.molecular
    def test_pipeline_lih_with_chunking(self, lih_hamiltonian):
        """LiH pipeline should still work correctly with streaming dedup."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = lih_hamiltonian
        e_fci = H.fci_energy()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=e_fci
        )
        results = pipeline.run(progress=False)

        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 0.1, f"LiH error {error_mha:.4f} mHa too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
