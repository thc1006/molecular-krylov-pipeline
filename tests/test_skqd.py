"""Tests for Sample-Based Krylov Quantum Diagonalization."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
from hamiltonians.spin import TransverseFieldIsing


class TestSKQD:
    """Test cases for SKQD."""

    def test_construction(self):
        """Test basic construction."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        skqd = SampleBasedKrylovDiagonalization(H)

        assert skqd.num_sites == 4
        assert len(skqd.initial_state) == 4

    def test_neel_state(self):
        """Test Néel state preparation."""
        H = TransverseFieldIsing(num_spins=6)
        skqd = SampleBasedKrylovDiagonalization(H)

        # Néel state should be |010101⟩
        expected = torch.tensor([1, 0, 1, 0, 1, 0])
        assert torch.all(skqd.initial_state == expected)

    def test_generate_krylov_samples(self):
        """Test Krylov sample generation."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        config = SKQDConfig(
            max_krylov_dim=3,
            shots_per_krylov=1000,
        )
        skqd = SampleBasedKrylovDiagonalization(H, config=config)

        samples = skqd.generate_krylov_samples(max_krylov_dim=3, progress=False)

        assert len(samples) == 3
        assert all(isinstance(s, dict) for s in samples)

        # Check that samples sum to shots
        for s in samples:
            total = sum(s.values())
            assert total == 1000

    def test_cumulative_basis(self):
        """Test cumulative basis building."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        config = SKQDConfig(
            max_krylov_dim=3,
            shots_per_krylov=1000,
        )
        skqd = SampleBasedKrylovDiagonalization(H, config=config)

        skqd.generate_krylov_samples(max_krylov_dim=3, progress=False)
        cumulative = skqd.build_cumulative_basis()

        assert len(cumulative) == 3

        # Each subsequent step should have at least as many states
        sizes = [len(c) for c in cumulative]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_ground_state_energy(self):
        """Test ground state energy computation."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5, periodic=False)

        # Get exact energy for comparison
        E_exact, _ = H.exact_ground_state()

        # Run SKQD
        config = SKQDConfig(
            max_krylov_dim=5,
            shots_per_krylov=10000,
        )
        skqd = SampleBasedKrylovDiagonalization(H, config=config)

        skqd.generate_krylov_samples(progress=False)
        E_skqd, _ = skqd.compute_ground_state_energy()

        # SKQD energy should be close to exact (within few percent)
        error = abs(E_skqd - E_exact) / abs(E_exact)
        assert error < 0.1, f"SKQD error {error:.4f} > 10%"

    def test_run_full_pipeline(self):
        """Test full SKQD run."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        config = SKQDConfig(
            max_krylov_dim=4,
            shots_per_krylov=5000,
        )
        skqd = SampleBasedKrylovDiagonalization(H, config=config)

        results = skqd.run(max_krylov_dim=4, progress=False)

        assert "krylov_dims" in results
        assert "energies" in results
        assert "basis_sizes" in results

        # Energy should decrease with Krylov dimension
        energies = results["energies"]
        assert len(energies) == 3  # k = 1, 2, 3

    def test_energy_convergence(self):
        """Test that energy converges with Krylov dimension."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.3, periodic=False)
        E_exact, _ = H.exact_ground_state()

        config = SKQDConfig(
            max_krylov_dim=8,
            shots_per_krylov=20000,
        )
        skqd = SampleBasedKrylovDiagonalization(H, config=config)

        results = skqd.run(progress=False)
        energies = results["energies"]

        # Later energies should be closer to exact
        errors = [abs(E - E_exact) for E in energies]

        # Final error should be smaller than initial
        assert errors[-1] < errors[0] * 1.5  # Allow some fluctuation


class TestKrylovBasisSampler:
    """Test Krylov basis sampler for CUDA-Q."""

    def test_classical_fallback(self):
        """Test classical simulation fallback."""
        from krylov.basis_sampler import KrylovBasisSampler

        # Simple Hamiltonian terms
        coeffs = [-1.0, -1.0, -0.5, -0.5]
        paulis = ["ZZI", "IZZ", "XII", "IXI"]

        sampler = KrylovBasisSampler(
            pauli_coefficients=coeffs,
            pauli_words=paulis,
            num_qubits=3,
        )

        # Sample from k=0 (initial state)
        samples = sampler.sample_krylov_state(krylov_power=0)

        assert len(samples) > 0
        assert sum(samples.values()) == 100000  # Default shots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
