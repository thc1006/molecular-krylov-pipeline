"""Tests for Sample-Based Krylov Quantum Diagonalization."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig


class TestSKQDConfig:
    """Test SKQD configuration."""

    def test_defaults(self):
        config = SKQDConfig()
        assert config.max_krylov_dim == 12
        assert config.max_diag_basis_size == 15000
        assert config.max_new_configs_per_krylov_step == 1000

    def test_custom_config(self):
        config = SKQDConfig(max_krylov_dim=5, max_diag_basis_size=500)
        assert config.max_krylov_dim == 5
        assert config.max_diag_basis_size == 500


class TestSKQDWithMolecular:
    """Test SKQD with molecular Hamiltonians."""

    @pytest.mark.molecular
    def test_construction(self, h2_hamiltonian):
        config = SKQDConfig(max_krylov_dim=3)
        skqd = SampleBasedKrylovDiagonalization(h2_hamiltonian, config=config)
        assert skqd.num_sites == 4

    @pytest.mark.molecular
    def test_run(self, h2_hamiltonian):
        """SKQD should produce results with energies."""
        config = SKQDConfig(max_krylov_dim=3, max_diag_basis_size=100)
        skqd = SampleBasedKrylovDiagonalization(h2_hamiltonian, config=config)
        results = skqd.run(max_krylov_dim=3, progress=False)
        assert "energies" in results
        assert len(results["energies"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
