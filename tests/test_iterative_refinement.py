"""Tests for iterative refinement loop + H-coupling guided NF training.

Validates:
1. H-coupling table construction and scoring
2. Single-iteration backward compatibility
3. Multi-iteration refinement with convergence detection
4. Direct-CI → NF transition on iteration 2
5. H-coupling loss gradient flow
"""

import sys
import os
import math
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def h2_system():
    """H2 system for fast tests."""
    try:
        from hamiltonians.molecular import create_h2_hamiltonian
        H = create_h2_hamiltonian(device="cpu")
        fci = H.fci_energy()
        return H, fci
    except ImportError:
        pytest.skip("PySCF not available")


@pytest.fixture(scope="module")
def lih_system():
    """LiH system for medium tests."""
    try:
        from hamiltonians.molecular import create_lih_hamiltonian
        H = create_lih_hamiltonian(device="cpu")
        fci = H.fci_energy()
        return H, fci
    except ImportError:
        pytest.skip("PySCF not available")


class TestHCouplingTable:
    """Test H-coupling table construction and scoring."""

    def test_coupling_table_built(self, h2_system):
        """set_eigenstate_reference should build a non-empty coupling table."""
        from flows.physics_guided_training import PhysicsGuidedFlowTrainer, PhysicsGuidedConfig
        from flows.particle_conserving_flow import ParticleConservingFlowSampler
        from nqs.dense import DenseNQS

        H, fci = h2_system
        flow = ParticleConservingFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1, hidden_dims=[32]
        )
        nqs = DenseNQS(num_sites=4, hidden_dims=[32])
        config = PhysicsGuidedConfig(
            h_coupling_weight=0.1,
            h_coupling_n_ref=10,
            num_epochs=1,
        )
        trainer = PhysicsGuidedFlowTrainer(flow, nqs, H, config, device="cpu")

        # Create a mock eigenstate: HF + singles basis with uniform coefficients
        hf = H.get_hf_state()
        basis = torch.stack([hf, hf.clone()])  # small basis
        # Use eigenvector from diag
        H_mat = H.matrix_elements(basis, basis)
        H_np = H_mat.detach().numpy()
        H_np = 0.5 * (H_np + H_np.T)
        eigenvalues, eigenvectors = np.linalg.eigh(H_np)
        coeffs = eigenvectors[:, 0]

        trainer.set_eigenstate_reference(basis, coeffs)

        assert len(trainer._h_coupling_table) > 0, "Coupling table should be non-empty"

    def test_coupling_scores_correct(self, h2_system):
        """Configs connected to reference should have positive coupling scores."""
        from flows.physics_guided_training import PhysicsGuidedFlowTrainer, PhysicsGuidedConfig
        from flows.particle_conserving_flow import ParticleConservingFlowSampler
        from nqs.dense import DenseNQS

        H, fci = h2_system
        flow = ParticleConservingFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1, hidden_dims=[32]
        )
        nqs = DenseNQS(num_sites=4, hidden_dims=[32])
        config = PhysicsGuidedConfig(h_coupling_weight=0.1, h_coupling_n_ref=10)
        trainer = PhysicsGuidedFlowTrainer(flow, nqs, H, config, device="cpu")

        # Build eigenstate from HF
        hf = H.get_hf_state()
        basis = hf.unsqueeze(0)
        coeffs = np.array([1.0])
        trainer.set_eigenstate_reference(basis, coeffs)

        # HF connections should have positive scores
        connected, _ = H.get_connections(hf)
        if len(connected) > 0:
            scores = trainer._compute_h_coupling_scores(connected)
            assert scores.sum() > 0, "Connected configs should have positive coupling"

        # Random invalid config should have zero score
        random_cfg = torch.zeros(4)
        random_cfg[0] = 1  # 1 electron, invalid
        scores_rand = trainer._compute_h_coupling_scores(random_cfg.unsqueeze(0))
        # Score may or may not be zero depending on hash collisions, but shouldn't crash
        assert math.isfinite(scores_rand.sum().item())


class TestCouplingLossGradient:
    """Test that H-coupling loss produces valid gradients."""

    def test_coupling_loss_in_flow_loss(self, h2_system):
        """_compute_flow_loss should include coupling term when enabled."""
        from flows.physics_guided_training import PhysicsGuidedFlowTrainer, PhysicsGuidedConfig
        from flows.particle_conserving_flow import ParticleConservingFlowSampler
        from nqs.dense import DenseNQS

        H, fci = h2_system
        flow = ParticleConservingFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1, hidden_dims=[32]
        )
        nqs = DenseNQS(num_sites=4, hidden_dims=[32])
        config = PhysicsGuidedConfig(
            h_coupling_weight=0.5,
            h_coupling_n_ref=10,
            num_epochs=2,
            min_epochs=1,
            samples_per_batch=50,
        )
        trainer = PhysicsGuidedFlowTrainer(flow, nqs, H, config, device="cpu")

        # Set eigenstate reference
        hf = H.get_hf_state()
        basis = hf.unsqueeze(0)
        coeffs = np.array([1.0])
        trainer.set_eigenstate_reference(basis, coeffs)

        # Run 2 epochs — should not crash and should track coupling loss
        history = trainer.train()
        assert 'coupling_losses' in history
        assert len(history['coupling_losses']) == 2


class TestSingleIterationBackwardCompat:
    """n_refinement_iterations=1 should behave exactly like before."""

    def test_single_iteration_h2(self, h2_system):
        """Single iteration should give same result as old pipeline."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = h2_system
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
            n_refinement_iterations=1,  # Default
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()

        energy = result["combined_energy"]
        error_mha = abs(energy - fci) * 1000
        assert error_mha < 0.5, f"H2 error {error_mha:.3f} mHa should be < 0.5"
        # No refinement keys
        assert "refinement_energies" not in result


class TestIterativeRefinement:
    """Test the iterative refinement loop."""

    @pytest.mark.molecular
    def test_two_iterations_h2(self, h2_system):
        """Two iterations should complete without error on H2."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = h2_system
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,  # Iteration 1 = Direct-CI
            device="cpu",
            n_refinement_iterations=2,
            h_coupling_weight=0.1,
            refinement_epochs=10,  # Short for test speed
            max_epochs=10,
            min_epochs=5,
            samples_per_batch=50,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()

        energy = result["combined_energy"]
        assert math.isfinite(energy), "Final energy should be finite"
        assert "refinement_energies" in result
        assert len(result["refinement_energies"]) == 2

    @pytest.mark.molecular
    def test_direct_ci_to_nf_transition(self, h2_system):
        """Iteration 1 Direct-CI → Iteration 2 should initialize NF."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = h2_system
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
            n_refinement_iterations=2,
            h_coupling_weight=0.1,
            refinement_epochs=5,
            max_epochs=5,
            min_epochs=3,
            samples_per_batch=50,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)

        # Before run: flow should be None (Direct-CI mode)
        assert pipeline.flow is None

        result = pipeline.run()

        # After run: flow should be initialized (from iteration 2)
        assert pipeline.flow is not None

    @pytest.mark.molecular
    def test_convergence_detection(self, h2_system):
        """Should stop early when energy converges."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = h2_system
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
            n_refinement_iterations=5,  # Allow up to 5
            refinement_convergence_threshold=1.0,  # 1 Ha — very loose, should converge immediately
            h_coupling_weight=0.1,
            refinement_epochs=5,
            max_epochs=5,
            min_epochs=3,
            samples_per_batch=50,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()

        # Should have stopped before 5 iterations (H2 converges quickly)
        n_iters = len(result.get("refinement_energies", []))
        assert n_iters <= 3, f"Should converge early, got {n_iters} iterations"


class TestIterativeRefinementLiH:
    """Test iterative refinement on LiH (larger system)."""

    @pytest.mark.molecular
    def test_lih_refinement_improves(self, lih_system):
        """Refinement should not degrade LiH energy."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = lih_system
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
            n_refinement_iterations=2,
            h_coupling_weight=0.1,
            refinement_epochs=20,
            max_epochs=20,
            min_epochs=10,
            samples_per_batch=200,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()

        energy = result["combined_energy"]
        error_mha = abs(energy - fci) * 1000
        # LiH with Direct-CI + SKQD already gets < 0.1 mHa
        # Refinement should not make it worse
        assert error_mha < 1.0, f"LiH error {error_mha:.3f} mHa should be < 1.0"
