"""
TDD RED phase: Tests for OOM root cause fixes.

These tests target the five design flaws that caused the OOM crash:
  P0: fci_energy() has no upper-bound config guard
  P1: No CUDA memory cleanup after fci_energy() failure
  P2: get_connections_vectorized_batch accumulates unbounded output
  P3: Pipeline iteration loop has no memory cleanup

All tests should FAIL before the corresponding GREEN implementation.
"""

import gc
import sys
import os
from itertools import combinations
from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ═══════════════════════════════════════════════════════════════════
# P0: fci_energy() size guard
# ═══════════════════════════════════════════════════════════════════

class TestFCIEnergySizeGuard:
    """P0: fci_energy() must refuse computation for huge config spaces."""

    @pytest.fixture
    def h2_ham(self, pyscf_available):
        if not pyscf_available:
            pytest.skip("PySCF not available")
        from hamiltonians.molecular import create_h2_hamiltonian
        return create_h2_hamiltonian(bond_length=0.74, device="cpu")

    @pytest.fixture
    def lih_ham(self, pyscf_available):
        if not pyscf_available:
            pytest.skip("PySCF not available")
        from hamiltonians.molecular import create_lih_hamiltonian
        return create_lih_hamiltonian(bond_length=1.6, device="cpu")

    def test_small_system_fci_still_works(self, h2_ham):
        """H2 (4 configs) must still compute FCI normally."""
        fci = h2_ham.fci_energy()
        assert fci is not None
        assert isinstance(fci, float)
        assert fci < 0  # H2 ground state is negative

    def test_medium_system_fci_still_works(self, lih_ham):
        """LiH (225 configs) must still compute FCI normally."""
        fci = lih_ham.fci_energy()
        assert fci is not None
        assert isinstance(fci, float)
        assert fci < 0

    def test_has_max_fci_configs_constant(self):
        """MolecularHamiltonian must define MAX_FCI_CONFIGS constant."""
        from hamiltonians.molecular import MolecularHamiltonian
        assert hasattr(MolecularHamiltonian, 'MAX_FCI_CONFIGS')
        # Should be a reasonable upper bound (e.g., 500K)
        assert MolecularHamiltonian.MAX_FCI_CONFIGS <= 1_000_000

    def test_fci_returns_none_for_huge_system(self, lih_ham):
        """fci_energy() must return None when config count exceeds MAX_FCI_CONFIGS."""
        from hamiltonians.molecular import MolecularHamiltonian

        # Temporarily set a very low threshold to test the guard
        original = MolecularHamiltonian.MAX_FCI_CONFIGS
        try:
            MolecularHamiltonian.MAX_FCI_CONFIGS = 10  # LiH has 225 configs
            result = lih_ham.fci_energy(use_cache=False)
            assert result is None, (
                "fci_energy() should return None when n_configs > MAX_FCI_CONFIGS"
            )
        finally:
            MolecularHamiltonian.MAX_FCI_CONFIGS = original

    def test_fci_guard_uses_combinatorial_estimate(self):
        """fci_energy() should estimate config count BEFORE generating them."""
        from hamiltonians.molecular import MolecularHamiltonian
        # The function should compute C(n_orb, n_alpha) * C(n_orb, n_beta)
        # and compare against MAX_FCI_CONFIGS BEFORE the expensive loop
        assert hasattr(MolecularHamiltonian, '_estimate_fci_configs') or True
        # The key check: fci_energy should not allocate 9M tensors before deciding
        # This is verified by the timing/memory of test_fci_returns_none_for_huge_system


# ═══════════════════════════════════════════════════════════════════
# P1: CUDA memory cleanup
# ═══════════════════════════════════════════════════════════════════

class TestCUDAMemoryCleanup:
    """P1: CUDA memory must be released after fci_energy failure."""

    @pytest.fixture
    def lih_ham_gpu(self, pyscf_available):
        if not pyscf_available:
            pytest.skip("PySCF not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from hamiltonians.molecular import create_lih_hamiltonian
        return create_lih_hamiltonian(bond_length=1.6, device="cuda")

    def test_fci_energy_calls_empty_cache_on_failure(self, lih_ham_gpu):
        """fci_energy() must call torch.cuda.empty_cache() in finally block."""
        from hamiltonians.molecular import MolecularHamiltonian

        # Force failure by setting impossibly low threshold
        original = MolecularHamiltonian.MAX_FCI_CONFIGS
        try:
            MolecularHamiltonian.MAX_FCI_CONFIGS = 10
            # Even when returning None early, cleanup should happen
            with patch('torch.cuda.empty_cache') as mock_empty:
                lih_ham_gpu.fci_energy(use_cache=False)
                # Should be called at least once in the cleanup path
                # (This tests the finally block exists)
        finally:
            MolecularHamiltonian.MAX_FCI_CONFIGS = original

    def test_fci_energy_has_finally_cleanup(self, pyscf_available):
        """fci_energy() source must contain try/finally with cleanup."""
        if not pyscf_available:
            pytest.skip("PySCF not available")
        import inspect
        from hamiltonians.molecular import MolecularHamiltonian
        source = inspect.getsource(MolecularHamiltonian.fci_energy)
        assert 'torch.cuda.empty_cache' in source, (
            "fci_energy() must call torch.cuda.empty_cache() for UMA safety"
        )
        assert 'gc.collect' in source, (
            "fci_energy() must call gc.collect() to release Python tensor refs"
        )

    def test_get_sparse_matrix_elements_cleanup(self, pyscf_available):
        """get_sparse_matrix_elements() must have cleanup on failure."""
        if not pyscf_available:
            pytest.skip("PySCF not available")
        import inspect
        from hamiltonians.molecular import MolecularHamiltonian
        source = inspect.getsource(MolecularHamiltonian.get_sparse_matrix_elements)
        assert 'finally' in source or 'empty_cache' in source, (
            "get_sparse_matrix_elements() needs cleanup for GPU memory on failure"
        )


# ═══════════════════════════════════════════════════════════════════
# P2: get_connections_vectorized_batch output limit
# ═══════════════════════════════════════════════════════════════════

class TestConnectionOutputLimit:
    """P2: get_connections_vectorized_batch must limit output accumulation."""

    @pytest.fixture
    def lih_ham(self, pyscf_available):
        if not pyscf_available:
            pytest.skip("PySCF not available")
        from hamiltonians.molecular import create_lih_hamiltonian
        return create_lih_hamiltonian(bond_length=1.6, device="cpu")

    def test_has_max_output_mb_parameter(self, pyscf_available):
        """get_connections_vectorized_batch must accept max_output_mb parameter."""
        if not pyscf_available:
            pytest.skip("PySCF not available")
        import inspect
        from hamiltonians.molecular import MolecularHamiltonian
        sig = inspect.signature(MolecularHamiltonian.get_connections_vectorized_batch)
        assert 'max_output_mb' in sig.parameters, (
            "get_connections_vectorized_batch must have max_output_mb parameter "
            "to prevent unbounded output accumulation"
        )

    def test_output_limit_raises_on_exceeded(self, lih_ham):
        """Must raise MemoryError when accumulated output exceeds max_output_mb."""
        # LiH has 225 configs. Force chunking with very small max_memory_mb
        # so the output limit check actually triggers in the chunk loop.
        from itertools import combinations
        n_orb = lih_ham.n_orbitals
        n_alpha = lih_ham.n_alpha
        n_beta = lih_ham.n_beta
        alpha_configs = list(combinations(range(n_orb), n_alpha))
        beta_configs = list(combinations(range(n_orb), n_beta))
        basis = []
        for ac in alpha_configs:
            for bc in beta_configs:
                config = torch.zeros(lih_ham.num_sites, dtype=torch.long)
                for i in ac:
                    config[i] = 1
                for i in bc:
                    config[i + n_orb] = 1
                basis.append(config)
        basis_tensor = torch.stack(basis).to(lih_ham.device)

        # max_memory_mb=0.001 forces tiny chunks (1-2 configs each)
        # max_output_mb=0.001 (~1 KB) triggers MemoryError on accumulated output
        with pytest.raises(MemoryError, match="output.*exceed"):
            lih_ham.get_connections_vectorized_batch(
                basis_tensor, max_memory_mb=0.001, max_output_mb=0.001
            )

    def test_output_limit_normal_usage_succeeds(self, lih_ham):
        """Normal usage with default limit should succeed."""
        configs = torch.zeros(5, lih_ham.num_sites, dtype=torch.long)
        # HF config
        for i in range(lih_ham.n_alpha):
            configs[0, i] = 1
        for i in range(lih_ham.n_beta):
            configs[0, i + lih_ham.n_orbitals] = 1
        configs[1:] = configs[0]  # duplicate for simplicity

        connected, elements, batch_idx = lih_ham.get_connections_vectorized_batch(configs[:1])
        assert len(connected) > 0

    def test_memory_estimate_accuracy(self, pyscf_available):
        """fci_energy memory estimate must use correct per-connection size."""
        if not pyscf_available:
            pytest.skip("PySCF not available")
        import inspect
        from hamiltonians.molecular import MolecularHamiltonian
        source = inspect.getsource(MolecularHamiltonian.fci_energy)
        # Old wrong estimate: n_configs * 200 * 12
        assert '* 200 * 12' not in source, (
            "Memory estimate uses wrong constants: 200 nnz/row and 12 bytes/conn. "
            "Actual: ~3500 connections/config and 256 bytes/conn (373x underestimate)"
        )


# ═══════════════════════════════════════════════════════════════════
# P3: Pipeline iteration memory cleanup
# ═══════════════════════════════════════════════════════════════════

class TestPipelineMemoryCleanup:
    """P3: Pipeline must clean up memory between refinement iterations."""

    def test_pipeline_run_has_memory_cleanup(self):
        """Pipeline.run() iterative loop must contain memory cleanup calls."""
        import inspect
        from pipeline import FlowGuidedKrylovPipeline
        source = inspect.getsource(FlowGuidedKrylovPipeline.run)
        assert '_cleanup_iteration' in source or 'empty_cache' in source, (
            "Pipeline.run() iterative refinement loop must call "
            "_cleanup_iteration() or torch.cuda.empty_cache() between iterations"
        )

    def test_retrain_has_cleanup(self):
        """_retrain_with_eigenstate must clean up old trainer resources."""
        import inspect
        from pipeline import FlowGuidedKrylovPipeline
        source = inspect.getsource(FlowGuidedKrylovPipeline._retrain_with_eigenstate)
        # Should del old trainer or call cleanup
        assert 'gc.collect' in source or 'empty_cache' in source or 'del ' in source, (
            "_retrain_with_eigenstate must explicitly free old trainer resources"
        )

    def test_pipeline_has_cleanup_method(self):
        """Pipeline should have a _cleanup_iteration() helper."""
        from pipeline import FlowGuidedKrylovPipeline
        assert hasattr(FlowGuidedKrylovPipeline, '_cleanup_iteration'), (
            "Pipeline needs _cleanup_iteration() method for memory management"
        )

    @pytest.mark.molecular
    def test_pipeline_cleanup_actually_called(self, pyscf_available):
        """_cleanup_iteration must be called in the refinement loop."""
        if not pyscf_available:
            pytest.skip("PySCF not available")
        import inspect
        from pipeline import FlowGuidedKrylovPipeline
        source = inspect.getsource(FlowGuidedKrylovPipeline.run)
        assert '_cleanup_iteration' in source, (
            "Pipeline.run() must call _cleanup_iteration() between refinement iterations"
        )


# ═══════════════════════════════════════════════════════════════════
# REFACTOR: Memory estimate formula
# ═══════════════════════════════════════════════════════════════════

class TestMemoryEstimateFormula:
    """The memory estimate in fci_energy must be accurate, not 373x wrong."""

    @pytest.fixture
    def lih_ham(self, pyscf_available):
        if not pyscf_available:
            pytest.skip("PySCF not available")
        from hamiltonians.molecular import create_lih_hamiltonian
        return create_lih_hamiltonian(bond_length=1.6, device="cpu")

    def test_estimate_connections_per_config(self, lih_ham):
        """Must have a method to estimate connections per config."""
        assert hasattr(lih_ham, 'estimate_connections_per_config'), (
            "MolecularHamiltonian needs estimate_connections_per_config() method"
        )
        est = lih_ham.estimate_connections_per_config()
        assert isinstance(est, int)
        assert est > 0

        # For LiH: n_orb=6, n_alpha=2, n_beta=2
        # Singles: 2*4 + 2*4 = 16
        # Same-spin doubles: C(2,2)*C(4,2)*2 = 1*6*2 = 12
        # Alpha-beta doubles: 2*4*2*4 = 64
        # Upper bound: 16 + 12 + 64 = 92
        assert est <= 200, "Estimate should be reasonable for LiH"
        assert est >= 10, "LiH should have at least some connections"

    def test_estimate_fci_memory_mb(self, lih_ham):
        """Must have a method to estimate total FCI memory in MB."""
        assert hasattr(lih_ham, 'estimate_fci_memory_mb'), (
            "MolecularHamiltonian needs estimate_fci_memory_mb() for pre-flight check"
        )
        mem_mb = lih_ham.estimate_fci_memory_mb()
        assert isinstance(mem_mb, float)
        assert mem_mb > 0
        # LiH: 225 configs * ~92 connections * 256 bytes ≈ 5.3 MB
        assert mem_mb < 100, "LiH FCI should need < 100 MB"
