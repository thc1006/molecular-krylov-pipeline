"""Tests for PR 1.6: Shift-invert eigsh.

Shift-invert mode (sigma=E_hf) makes ARPACK converge to eigenvalues near
the Hartree-Fock energy, which is close to the ground state. This reduces
the number of ARPACK iterations needed by 2-5x.

Key idea: instead of finding smallest eigenvalue of H (which='SA'),
find largest eigenvalue of (H - sigma*I)^{-1} (which='LM').
The eigenvalues nearest sigma map to the largest eigenvalues of the inverse.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestShiftInvertAccuracy:
    """Shift-invert must produce identical ground state energy."""

    @pytest.mark.molecular
    def test_shift_invert_matches_standard_h2(self, h2_hamiltonian):
        """H2: shift-invert energy must match standard eigsh."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(h2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(h2_hamiltonian, config=config)

        E_standard, _ = skqd.compute_ground_state_energy(basis)
        E_shift_inv, _ = skqd.compute_ground_state_energy(
            basis, shift_invert=True
        )

        assert abs(E_standard - E_shift_inv) < 1e-8, (
            f"Shift-invert {E_shift_inv:.10f} != standard {E_standard:.10f}"
        )

    @pytest.mark.molecular
    def test_shift_invert_matches_standard_lih(self, lih_hamiltonian):
        """LiH: shift-invert energy must match standard eigsh."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)

        E_standard, _ = skqd.compute_ground_state_energy(basis)
        E_shift_inv, _ = skqd.compute_ground_state_energy(
            basis, shift_invert=True
        )

        assert abs(E_standard - E_shift_inv) < 1e-8, (
            f"Shift-invert {E_shift_inv:.10f} != standard {E_standard:.10f}"
        )

    @pytest.mark.molecular
    def test_shift_invert_matches_standard_beh2(self, beh2_hamiltonian):
        """BeH2 (1225 configs): shift-invert on larger system."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(beh2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(beh2_hamiltonian, config=config)

        E_standard, _ = skqd.compute_ground_state_energy(basis)
        E_shift_inv, _ = skqd.compute_ground_state_energy(
            basis, shift_invert=True
        )

        assert abs(E_standard - E_shift_inv) < 1e-6, (
            f"Shift-invert {E_shift_inv:.10f} != standard {E_standard:.10f}"
        )


class TestShiftInvertSparse:
    """Shift-invert should also work with sparse eigsh path."""

    @pytest.mark.molecular
    def test_sparse_shift_invert_matches(self, beh2_hamiltonian):
        """Sparse shift-invert must match sparse standard on BeH2."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(beh2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(beh2_hamiltonian, config=config)

        # Standard sparse
        E_standard, _ = skqd._sparse_ground_state(basis)

        # Shift-invert sparse
        E_shift_inv, _ = skqd._sparse_ground_state(basis, shift_invert=True)

        assert abs(E_standard - E_shift_inv) < 1e-6, (
            f"Sparse shift-invert {E_shift_inv:.10f} != standard {E_standard:.10f}"
        )

    @pytest.mark.molecular
    def test_sparse_shift_invert_chemical_accuracy(self, lih_hamiltonian):
        """Sparse shift-invert must achieve chemical accuracy on LiH."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = lih_hamiltonian.fci_energy()

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)

        E_si, _ = skqd._sparse_ground_state(basis, shift_invert=True)
        error_mha = abs(E_si - e_fci) * 1000

        assert error_mha < 5.0, (
            f"Shift-invert error {error_mha:.4f} mHa exceeds chemical accuracy"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
