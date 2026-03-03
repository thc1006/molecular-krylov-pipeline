"""Tests for the full Flow-Guided Krylov pipeline."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from hamiltonians.spin import TransverseFieldIsing


class TestPipelineConstruction:
    """Test pipeline construction."""

    def test_basic_construction(self):
        """Test basic pipeline construction."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)
        pipeline = FlowGuidedKrylovPipeline(H)

        assert pipeline.num_sites == 4
        assert pipeline.hamiltonian is H

    def test_with_config(self):
        """Test construction with custom config."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0)
        config = PipelineConfig(
            nf_coupling_layers=2,
            nqs_hidden_dims=[64, 64],
            max_epochs=10,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)

        assert pipeline.config.nf_coupling_layers == 2
        assert pipeline.config.max_epochs == 10

    def test_with_exact_energy(self):
        """Test construction with known exact energy."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        E_exact, _ = H.exact_ground_state()

        pipeline = FlowGuidedKrylovPipeline(H, exact_energy=E_exact)

        assert pipeline.exact_energy == E_exact


class TestPipelineStages:
    """Test individual pipeline stages."""

    @pytest.fixture
    def small_pipeline(self):
        """Create a small pipeline for testing."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        config = PipelineConfig(
            nf_coupling_layers=2,
            nqs_hidden_dims=[32, 32],
            samples_per_batch=100,
            num_batches=2,
            max_epochs=5,
            inference_samples=100,
            inference_iterations=10,
            max_krylov_dim=3,
            shots_per_krylov=1000,
            device="cpu",
        )
        return FlowGuidedKrylovPipeline(H, config=config)

    def test_train_nf_nqs(self, small_pipeline):
        """Test NF-NQS training stage."""
        history = small_pipeline.train_nf_nqs(progress=False)

        assert "energies" in history
        assert "flow_loss" in history
        assert len(history["energies"]) > 0

    def test_extract_basis(self, small_pipeline):
        """Test basis extraction stage."""
        # First train
        small_pipeline.train_nf_nqs(progress=False)

        # Then extract basis
        basis = small_pipeline.extract_basis(n_samples=100)

        assert basis.shape[1] == 4  # num_sites
        assert len(basis) > 0
        assert len(basis) <= 100  # At most n_samples unique

    def test_run_skqd(self, small_pipeline):
        """Test SKQD stage."""
        # Train and extract basis first
        small_pipeline.train_nf_nqs(progress=False)
        small_pipeline.extract_basis(n_samples=100)

        # Run SKQD
        results = small_pipeline.run_skqd(use_nf_basis=True, progress=False)

        assert "energies_combined" in results or "energies" in results


class TestPipelineIntegration:
    """Test full pipeline integration."""

    @pytest.mark.slow
    def test_full_run(self):
        """Test full pipeline run on small system."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        E_exact, _ = H.exact_ground_state()

        config = PipelineConfig(
            nf_coupling_layers=2,
            nqs_hidden_dims=[32, 32],
            samples_per_batch=100,
            num_batches=2,
            max_epochs=10,
            inference_samples=100,
            inference_iterations=20,
            max_krylov_dim=4,
            shots_per_krylov=5000,
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=E_exact)

        results = pipeline.run(progress=False)

        # Should have results from all stages
        assert "nf_nqs_energy" in results
        assert "inference_energy" in results
        assert "skqd_results" in results

        # Final energy should be reasonable
        final_energy = results.get(
            "combined_energy", results.get("skqd_energy")
        )
        error = abs(final_energy - E_exact) / abs(E_exact)

        # Allow generous error for fast test
        assert error < 0.5, f"Final error {error:.2%} > 50%"


class TestPipelineCheckpoints:
    """Test checkpoint functionality."""

    def test_save_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
        config = PipelineConfig(
            nf_coupling_layers=2,
            nqs_hidden_dims=[32, 32],
            samples_per_batch=50,
            num_batches=1,
            max_epochs=2,
            device="cpu",
        )

        # Create and train pipeline
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        pipeline.train_nf_nqs(progress=False)
        pipeline.extract_basis(n_samples=50)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        pipeline.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load into new pipeline
        new_pipeline = FlowGuidedKrylovPipeline(H, config=config)
        new_pipeline.load_checkpoint(str(checkpoint_path))

        # Should have the same results
        assert "nf_nqs_energy" in new_pipeline.results
        assert hasattr(new_pipeline, "nf_basis")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
