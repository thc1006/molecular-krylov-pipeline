"""Tests for the Flow-Guided Krylov pipeline.

NOTE: Original tests used TransverseFieldIsing (spin model) which no longer exists.
      Pipeline now requires MolecularHamiltonian. Tests updated accordingly.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


class TestPipelineConstruction:
    """Test pipeline construction."""

    @pytest.mark.molecular
    def test_basic_construction(self, h2_hamiltonian):
        pipeline = FlowGuidedKrylovPipeline(
            h2_hamiltonian,
            config=PipelineConfig(device="cpu", skip_nf_training=True),
        )
        assert pipeline.num_sites == 4
        assert pipeline.hamiltonian is h2_hamiltonian

    @pytest.mark.molecular
    def test_with_config(self, h2_hamiltonian):
        config = PipelineConfig(
            nf_hidden_dims=[64, 64],
            nqs_hidden_dims=[64, 64],
            max_epochs=10,
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(h2_hamiltonian, config=config)
        assert pipeline.config.max_epochs == 10

    @pytest.mark.molecular
    def test_with_exact_energy(self, h2_hamiltonian):
        e_fci = h2_hamiltonian.fci_energy()
        pipeline = FlowGuidedKrylovPipeline(
            h2_hamiltonian,
            config=PipelineConfig(device="cpu", skip_nf_training=True),
            exact_energy=e_fci,
        )
        assert pipeline.exact_energy == e_fci

    @pytest.mark.molecular
    def test_rejects_non_molecular(self):
        """Pipeline should reject non-MolecularHamiltonian."""
        from hamiltonians.base import Hamiltonian

        class FakeH(Hamiltonian):
            def __init__(self):
                self.num_sites = 4
            def diagonal_element(self, config):
                return torch.tensor(0.0)
            def get_connections(self, config):
                return [], []

        with pytest.raises(TypeError, match="MolecularHamiltonian"):
            FlowGuidedKrylovPipeline(FakeH())


class TestPipelineDirectCI:
    """Test Direct-CI mode (skip_nf_training=True)."""

    @pytest.mark.molecular
    def test_direct_ci_run(self, h2_hamiltonian):
        """Test full pipeline run in Direct-CI mode."""
        e_fci = h2_hamiltonian.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            h2_hamiltonian, config=config, exact_energy=e_fci
        )
        results = pipeline.run(progress=False)

        assert "combined_energy" in results or "skqd_energy" in results
        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 0.1, f"H2 Direct-CI error {error_mha:.4f} mHa too large"

    @pytest.mark.molecular
    def test_direct_ci_sqd_mode(self, lih_hamiltonian):
        """Test Direct-CI + SQD mode."""
        e_fci = lih_hamiltonian.fci_energy()
        config = PipelineConfig(
            subspace_mode="sqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            lih_hamiltonian, config=config, exact_energy=e_fci
        )
        results = pipeline.run(progress=False)

        best = results.get("combined_energy", results.get("sqd_energy"))
        assert best is not None, "No energy found in results"
        error_mha = abs(best - e_fci) * 1000
        # SQD with noise may have larger error
        assert error_mha < 5.0, f"LiH SQD error {error_mha:.4f} mHa too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
