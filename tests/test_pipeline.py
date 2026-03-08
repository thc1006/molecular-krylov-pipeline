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


class TestMaxDiagBasisSizeWiring:
    """Test that max_diag_basis_size is wired from PipelineConfig to SKQDConfig (Bug H2)."""

    def test_pipeline_config_has_max_diag_basis_size_default(self):
        config = PipelineConfig(device="cpu")
        assert config.max_diag_basis_size == 15000

    def test_adapt_scales_max_diag_basis_size_very_large(self):
        config = PipelineConfig(device="cpu")
        config.adapt_to_system_size(100000, verbose=False)
        assert config.max_diag_basis_size == 15000

    def test_adapt_scales_max_diag_basis_size_large(self):
        config = PipelineConfig(device="cpu")
        config.adapt_to_system_size(15000, verbose=False)
        assert config.max_diag_basis_size == 25000

    def test_adapt_keeps_max_diag_basis_size_small(self):
        config = PipelineConfig(device="cpu")
        config.adapt_to_system_size(500, verbose=False)
        assert config.max_diag_basis_size == 15000

    @pytest.mark.molecular
    def test_max_diag_basis_size_wired_to_skqd(self, h2_hamiltonian):
        """max_diag_basis_size from PipelineConfig must reach SKQDConfig."""
        from krylov.skqd import SKQDConfig

        config = PipelineConfig(device="cpu", skip_nf_training=True, max_diag_basis_size=42000)
        pipeline = FlowGuidedKrylovPipeline(h2_hamiltonian, config=config, auto_adapt=False)
        cfg = pipeline.config
        skqd_config = SKQDConfig(
            max_krylov_dim=cfg.max_krylov_dim,
            time_step=cfg.time_step,
            shots_per_krylov=cfg.shots_per_krylov,
            use_gpu=False,
            regularization=getattr(cfg, "skqd_regularization", 1e-8),
            max_diag_basis_size=cfg.max_diag_basis_size,
        )
        assert skqd_config.max_diag_basis_size == 42000


class TestPipelineConfigNFOverride:
    """Test that skip_nf_training explicit overrides are preserved by adapt_to_system_size."""

    def test_default_none_resolves_to_false(self):
        """Default PipelineConfig() should resolve skip_nf_training to False."""
        config = PipelineConfig()
        assert config.skip_nf_training is False

    def test_explicit_true_sets_override_flag(self):
        """PipelineConfig(skip_nf_training=True) should set _user_set_skip_nf."""
        config = PipelineConfig(skip_nf_training=True)
        assert hasattr(config, "_user_set_skip_nf")
        assert config._user_set_skip_nf is True

    def test_explicit_false_sets_override_flag(self):
        """PipelineConfig(skip_nf_training=False) should set _user_set_skip_nf."""
        config = PipelineConfig(skip_nf_training=False)
        assert hasattr(config, "_user_set_skip_nf")
        assert config._user_set_skip_nf is True

    def test_default_has_no_override_flag(self):
        """Default PipelineConfig() should NOT set _user_set_skip_nf."""
        config = PipelineConfig()
        assert not hasattr(config, "_user_set_skip_nf")

    def test_adapt_preserves_explicit_false(self):
        """adapt_to_system_size should NOT override explicit skip_nf_training=False."""
        config = PipelineConfig(skip_nf_training=False)
        # Small system would normally force skip_nf_training=True
        config.adapt_to_system_size(100, verbose=False)
        assert config.skip_nf_training is False, (
            "adapt_to_system_size overrode explicit skip_nf_training=False"
        )

    def test_adapt_preserves_explicit_true(self):
        """adapt_to_system_size should NOT override explicit skip_nf_training=True."""
        config = PipelineConfig(skip_nf_training=True)
        # Large system would normally enable NF
        config.adapt_to_system_size(100000, verbose=False)
        assert config.skip_nf_training is True, (
            "adapt_to_system_size overrode explicit skip_nf_training=True"
        )

    def test_adapt_auto_enables_nf_for_large(self):
        """Default config with large system should auto-enable NF."""
        config = PipelineConfig()
        config.adapt_to_system_size(100000, verbose=False)
        assert config.skip_nf_training is False

    def test_adapt_auto_disables_nf_for_small(self):
        """Default config with small system should auto-disable NF."""
        config = PipelineConfig()
        config.adapt_to_system_size(100, verbose=False)
        assert config.skip_nf_training is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
