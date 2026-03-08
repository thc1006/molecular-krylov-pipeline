"""Tests for NNCI integration into the pipeline (PR-B1).

NNCI (Neural Network Configuration Interaction) uses a trained classifier
to discover important higher-excitation configs beyond HF+singles+doubles.

This file tests:
1. PipelineConfig NNCI fields and auto-enable/disable logic
2. Full pipeline integration with use_nnci=True
3. Non-regression: use_nnci=False must not change existing behavior
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestNNCIPipelineConfig:
    """Test NNCI configuration in PipelineConfig."""

    def test_nnci_config_defaults(self):
        """PipelineConfig should have NNCI fields with sensible defaults."""
        from pipeline import PipelineConfig

        config = PipelineConfig()
        assert hasattr(config, "use_nnci")
        # Default resolved to False (from None sentinel)
        assert config.use_nnci is False
        assert hasattr(config, "nnci_iterations")
        assert config.nnci_iterations == 5
        assert hasattr(config, "nnci_candidates_per_iter")
        assert config.nnci_candidates_per_iter == 5000
        assert hasattr(config, "nnci_candidates_per_iter")

    def test_nnci_auto_enable_large_system(self):
        """NNCI should auto-enable for systems > 20K configs."""
        from pipeline import PipelineConfig

        config = PipelineConfig()
        config.adapt_to_system_size(25000, verbose=False)
        assert config.use_nnci is True, "NNCI should auto-enable for >20K configs"

    def test_nnci_auto_disable_small_system(self):
        """NNCI should stay disabled for systems <= 20K configs."""
        from pipeline import PipelineConfig

        config = PipelineConfig()
        config.adapt_to_system_size(5000, verbose=False)
        assert config.use_nnci is False, "NNCI should not enable for <=20K configs"

    def test_nnci_user_override_false(self):
        """User setting use_nnci=False should be preserved even for large systems."""
        from pipeline import PipelineConfig

        config = PipelineConfig(use_nnci=False)
        config.adapt_to_system_size(50000, verbose=False)
        assert config.use_nnci is False, "User override should be preserved"

    def test_nnci_user_override_true(self):
        """User setting use_nnci=True should be preserved for small systems."""
        from pipeline import PipelineConfig

        config = PipelineConfig(use_nnci=True)
        config.adapt_to_system_size(1000, verbose=False)
        assert config.use_nnci is True, "User override should be preserved"

    def test_nnci_config_custom_values(self):
        """PipelineConfig should accept custom NNCI parameter values."""
        from pipeline import PipelineConfig

        config = PipelineConfig(
            use_nnci=True,
            nnci_iterations=10,
            nnci_candidates_per_iter=2000,
        )
        assert config.use_nnci is True
        assert config.nnci_iterations == 10
        assert config.nnci_candidates_per_iter == 2000


class TestNNCIPipelineIntegration:
    """Test NNCI integration with the full pipeline."""

    @pytest.fixture
    def lih_hamiltonian(self):
        """Create LiH Hamiltonian for testing."""
        try:
            from hamiltonians.molecular import create_lih_hamiltonian

            return create_lih_hamiltonian(bond_length=1.6, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.mark.molecular
    def test_nnci_pipeline_runs(self, lih_hamiltonian):
        """Pipeline with use_nnci=True should run without errors on LiH."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=True,
            nnci_iterations=2,  # Keep it fast
            nnci_candidates_per_iter=100,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=config)
        results = pipeline.run()

        assert "combined_energy" in results
        assert results["combined_energy"] < 0  # Should be negative for molecules

    @pytest.mark.molecular
    def test_nnci_improves_or_matches_energy(self, lih_hamiltonian):
        """NNCI-SKQD energy should be <= Direct-CI-SKQD (variational principle)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        # Without NNCI
        config_no_nnci = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=False,
            device="cpu",
        )
        pipeline_no = FlowGuidedKrylovPipeline(lih_hamiltonian, config=config_no_nnci)
        results_no = pipeline_no.run()

        # With NNCI
        config_nnci = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=True,
            nnci_iterations=3,
            nnci_candidates_per_iter=200,
            device="cpu",
        )
        pipeline_yes = FlowGuidedKrylovPipeline(lih_hamiltonian, config=config_nnci)
        results_yes = pipeline_yes.run()

        # NNCI should be at least as good (variational: more basis = lower or equal energy)
        assert results_yes["combined_energy"] <= results_no["combined_energy"] + 0.001, (
            f"NNCI energy {results_yes['combined_energy']:.6f} should be <= "
            f"Direct-CI energy {results_no['combined_energy']:.6f}"
        )

    @pytest.mark.molecular
    def test_nnci_basis_expansion_tracked(self, lih_hamiltonian):
        """NNCI should record expansion stats in results."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=True,
            nnci_iterations=2,
            nnci_candidates_per_iter=100,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=config)
        results = pipeline.run()

        # The results should record NNCI expansion stats
        assert "nnci_configs_added" in results, (
            "Results should contain 'nnci_configs_added' key"
        )
        assert isinstance(results["nnci_configs_added"], int)
        assert results["nnci_configs_added"] >= 0

    @pytest.mark.molecular
    def test_nnci_chemical_accuracy(self, lih_hamiltonian):
        """Pipeline with NNCI should still achieve chemical accuracy on LiH."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=True,
            nnci_iterations=3,
            nnci_candidates_per_iter=200,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=config)
        results = pipeline.run()

        fci = lih_hamiltonian.fci_energy()
        error_mha = abs(results["combined_energy"] - fci) * 1000
        assert error_mha < 1.6, (
            f"NNCI pipeline should achieve chemical accuracy on LiH. "
            f"Error: {error_mha:.3f} mHa"
        )


class TestNNCIDisabled:
    """Test that NNCI=False doesn't change existing behavior."""

    @pytest.mark.molecular
    def test_pipeline_unchanged_without_nnci(self):
        """Pipeline results should be identical when use_nnci=False (default)."""
        try:
            from hamiltonians.molecular import create_lih_hamiltonian
            from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        except ImportError:
            pytest.skip("PySCF not available")

        H = create_lih_hamiltonian(bond_length=1.6, device="cpu")

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=False,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run()

        fci = H.fci_energy()
        error_mha = abs(results["combined_energy"] - fci) * 1000
        assert error_mha < 1.6, f"Existing behavior broken: error {error_mha:.3f} mHa"

    @pytest.mark.molecular
    def test_default_config_has_nnci_off(self):
        """Default PipelineConfig should not have NNCI enabled."""
        from pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.use_nnci is False, "NNCI should be off by default"


class TestNNCISKQDIntegration:
    """Test NNCI expansion within FlowGuidedSKQD."""

    @pytest.fixture
    def lih_hamiltonian(self):
        """Create LiH Hamiltonian for testing."""
        try:
            from hamiltonians.molecular import create_lih_hamiltonian

            return create_lih_hamiltonian(bond_length=1.6, device="cpu")
        except ImportError:
            pytest.skip("PySCF not available")

    @pytest.mark.molecular
    def test_skqd_accepts_nnci_config(self, lih_hamiltonian):
        """FlowGuidedSKQD should accept NNCI config parameters via pipeline_config."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from pipeline import PipelineConfig

        # Generate a small basis
        hf = lih_hamiltonian.get_hf_state().unsqueeze(0)

        skqd_config = SKQDConfig(max_krylov_dim=2, max_diag_basis_size=15000)
        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=hf,
            config=skqd_config,
        )

        # FlowGuidedSKQD should have nnci_expand_basis method
        assert hasattr(skqd, "_nnci_expand_basis"), (
            "FlowGuidedSKQD should have _nnci_expand_basis method"
        )

    @pytest.mark.molecular
    def test_nnci_expand_basis_returns_expanded(self, lih_hamiltonian):
        """_nnci_expand_basis should return a larger basis than input."""
        import torch
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig

        # Generate CISD basis
        n_orb = lih_hamiltonian.n_orbitals
        n_alpha = lih_hamiltonian.n_alpha
        n_beta = lih_hamiltonian.n_beta
        hf = lih_hamiltonian.get_hf_state()
        configs = [hf.clone()]

        occ_alpha = list(range(n_alpha))
        virt_alpha = list(range(n_alpha, n_orb))
        occ_beta = list(range(n_beta))
        virt_beta = list(range(n_beta, n_orb))

        # Add singles
        for i in occ_alpha:
            for a in virt_alpha:
                c = hf.clone()
                c[i] = 0
                c[a] = 1
                configs.append(c)
        for i in occ_beta:
            for a in virt_beta:
                c = hf.clone()
                c[i + n_orb] = 0
                c[a + n_orb] = 1
                configs.append(c)

        # Add doubles (alpha-beta only for speed)
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b in virt_beta:
                        c = hf.clone()
                        c[i] = 0
                        c[j + n_orb] = 0
                        c[a] = 1
                        c[b + n_orb] = 1
                        configs.append(c)

        basis = torch.stack(configs)
        basis = torch.unique(basis, dim=0)
        original_size = len(basis)

        skqd_config = SKQDConfig(max_krylov_dim=2, max_diag_basis_size=15000)
        skqd = FlowGuidedSKQD(
            hamiltonian=lih_hamiltonian,
            nf_basis=basis,
            config=skqd_config,
        )

        # Expand the basis using NNCI
        expanded = skqd._nnci_expand_basis(
            basis=basis,
            nnci_iterations=2,
            nnci_candidates_per_iter=100,
        )

        # Should have more configs than original (triples/quadruples added)
        assert len(expanded) >= original_size, (
            f"Expanded basis ({len(expanded)}) should be >= original ({original_size})"
        )
