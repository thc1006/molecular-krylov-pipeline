"""Tests for NNCI (Neural Network Configuration Importance) active learning module.

Validates:
1. ConfigImportanceClassifier: NN architecture, forward pass, training
2. CandidateGenerator: excitation-based candidate generation with dedup
3. NNCIActiveLearning: active learning loop convergence and integration
4. Pipeline integration: NNCI vs Direct-CI on molecular systems

TDD Phase 1: RED -- all tests should fail until src/krylov/nnci.py is implemented.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _generate_cisd_configs(hamiltonian):
    """Generate full CISD (HF + singles + doubles) config set for testing."""
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    hf = hamiltonian.get_hf_state()
    configs = [hf.clone()]

    occ_alpha = list(range(n_alpha))
    occ_beta = list(range(n_beta))
    virt_alpha = list(range(n_alpha, n_orb))
    virt_beta = list(range(n_beta, n_orb))

    # Singles: alpha
    for i in occ_alpha:
        for a in virt_alpha:
            c = hf.clone()
            c[i] = 0
            c[a] = 1
            configs.append(c)

    # Singles: beta
    for i in occ_beta:
        for a in virt_beta:
            c = hf.clone()
            c[i + n_orb] = 0
            c[a + n_orb] = 1
            configs.append(c)

    # Doubles: alpha-alpha
    for i, j in combinations(occ_alpha, 2):
        for a, b in combinations(virt_alpha, 2):
            c = hf.clone()
            c[i] = 0
            c[j] = 0
            c[a] = 1
            c[b] = 1
            configs.append(c)

    # Doubles: beta-beta
    for i, j in combinations(occ_beta, 2):
        for a, b in combinations(virt_beta, 2):
            c = hf.clone()
            c[i + n_orb] = 0
            c[j + n_orb] = 0
            c[a + n_orb] = 1
            c[b + n_orb] = 1
            configs.append(c)

    # Doubles: alpha-beta
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

    tensor = torch.stack(configs)
    tensor = torch.unique(tensor, dim=0)
    return tensor


def _diagonalize_subspace(hamiltonian, configs):
    """Diagonalize Hamiltonian in a config subspace. Returns (energy, eigenvector)."""
    H = hamiltonian.matrix_elements(configs, configs)
    H_np = H.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    return float(eigenvalues[0]), eigenvectors[:, 0]


# =============================================================================
# Tests: ConfigImportanceClassifier
# =============================================================================


class TestConfigImportanceClassifier:
    """Test the NN importance classifier architecture and training."""

    def test_classifier_creation(self):
        """Classifier should be created with correct architecture."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 12
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[64, 32])
        assert isinstance(clf, torch.nn.Module)
        # Should have learnable parameters
        n_params = sum(p.numel() for p in clf.parameters())
        assert n_params > 0, "Classifier should have learnable parameters"

    def test_classifier_default_hidden_dims(self):
        """Classifier should use sensible default hidden dims when not specified."""
        from krylov.nnci import ConfigImportanceClassifier

        clf = ConfigImportanceClassifier(n_sites=12)
        n_params = sum(p.numel() for p in clf.parameters())
        assert n_params > 0

    def test_forward_pass_shape(self):
        """Forward pass should output (batch, 1) importance scores."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 12
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[64, 32])
        batch = torch.zeros(10, n_sites, dtype=torch.float32)
        output = clf(batch)
        assert output.shape == (10, 1), f"Expected (10, 1), got {output.shape}"

    def test_forward_pass_single_sample(self):
        """Forward pass should handle single sample input."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 12
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[64, 32])
        single = torch.zeros(1, n_sites, dtype=torch.float32)
        output = clf(single)
        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"

    def test_output_is_non_negative(self):
        """Importance scores should be non-negative (using ReLU/Softplus output)."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 12
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[64, 32])
        batch = torch.randn(50, n_sites)
        output = clf(batch)
        assert (output >= 0).all(), "Importance scores should be non-negative"

    def test_training_reduces_loss(self):
        """Training on labeled configs should reduce MSE loss over epochs."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 12
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[64, 32])

        # Create synthetic training data: random configs with random importance labels
        torch.manual_seed(42)
        configs = torch.randint(0, 2, (100, n_sites), dtype=torch.float32)
        labels = torch.rand(100, 1)  # importance scores in [0, 1]

        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # Record initial loss
        initial_loss = criterion(clf(configs), labels).item()

        # Train for a few epochs
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(clf(configs), labels)
            loss.backward()
            optimizer.step()

        final_loss = criterion(clf(configs), labels).item()
        assert final_loss < initial_loss * 0.5, (
            f"Training should reduce loss significantly. "
            f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
        )

    @pytest.mark.molecular
    def test_importance_ordering(self, lih_hamiltonian):
        """After training, HF score should exceed mean score of low-importance configs."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = lih_hamiltonian.num_sites

        # Generate CISD basis and diagonalize
        cisd = _generate_cisd_configs(lih_hamiltonian)
        _, eigvec = _diagonalize_subspace(lih_hamiltonian, cisd)
        ci_coeffs = np.abs(eigvec)  # importance = |c_i|

        # Train classifier on CI coefficients
        torch.manual_seed(42)
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[128, 64])
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        configs_float = cisd.float()
        labels = torch.from_numpy(ci_coeffs).float().unsqueeze(1)

        for _ in range(300):
            optimizer.zero_grad()
            loss = criterion(clf(configs_float), labels)
            loss.backward()
            optimizer.step()

        # Score all configs in the training set
        with torch.no_grad():
            all_scores = clf(configs_float).squeeze(1).numpy()

        # HF is the config with highest CI coefficient
        hf_idx = np.argmax(ci_coeffs)
        hf_score = all_scores[hf_idx]

        # Low-importance configs: bottom 50% by true CI coefficient
        sorted_indices = np.argsort(ci_coeffs)
        bottom_half = sorted_indices[: len(sorted_indices) // 2]
        mean_low_score = all_scores[bottom_half].mean()

        assert hf_score > mean_low_score, (
            f"HF score ({hf_score:.4f}) should exceed mean of bottom-50% "
            f"({mean_low_score:.4f}) after training"
        )

    def test_batch_prediction(self):
        """Should handle batch prediction efficiently."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 20
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[128, 64])
        batch = torch.randint(0, 2, (1000, n_sites), dtype=torch.float32)
        output = clf(batch)
        assert output.shape == (1000, 1)
        assert not torch.isnan(output).any(), "No NaN in batch predictions"
        assert not torch.isinf(output).any(), "No Inf in batch predictions"


# =============================================================================
# Tests: CandidateGenerator
# =============================================================================


class TestCandidateGenerator:
    """Test generation of candidate configurations beyond CISD."""

    @pytest.mark.molecular
    def test_generate_triples_from_hf(self, lih_hamiltonian):
        """Should generate triple excitations from HF reference."""
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        gen = CandidateGenerator(
            n_orbitals=lih_hamiltonian.n_orbitals,
            n_alpha=lih_hamiltonian.n_alpha,
            n_beta=lih_hamiltonian.n_beta,
        )
        triples = gen.generate_excitations(hf, max_rank=3, min_rank=3)
        assert len(triples) > 0, "Should generate at least one triple excitation"

    @pytest.mark.molecular
    def test_generate_quadruples_from_hf(self, lih_hamiltonian):
        """Should generate quadruple excitations from HF reference."""
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        gen = CandidateGenerator(
            n_orbitals=lih_hamiltonian.n_orbitals,
            n_alpha=lih_hamiltonian.n_alpha,
            n_beta=lih_hamiltonian.n_beta,
        )
        quads = gen.generate_excitations(hf, max_rank=4, min_rank=4)
        # LiH has 2 alpha, 2 beta, 6 orbitals -> quadruples exist
        assert len(quads) > 0, "Should generate quadruple excitations for LiH"

    @pytest.mark.molecular
    def test_particle_conservation(self, lih_hamiltonian):
        """All generated candidates must conserve particle number."""
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        n_orb = lih_hamiltonian.n_orbitals
        n_alpha = lih_hamiltonian.n_alpha
        n_beta = lih_hamiltonian.n_beta

        gen = CandidateGenerator(
            n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta
        )
        candidates = gen.generate_excitations(hf, max_rank=4, min_rank=1)

        for i in range(len(candidates)):
            c = candidates[i]
            alpha_count = c[:n_orb].sum().item()
            beta_count = c[n_orb:].sum().item()
            assert alpha_count == n_alpha, (
                f"Config {i}: alpha electron count {alpha_count} != {n_alpha}"
            )
            assert beta_count == n_beta, (
                f"Config {i}: beta electron count {beta_count} != {n_beta}"
            )

    @pytest.mark.molecular
    def test_no_duplicates(self, lih_hamiltonian):
        """Generated candidates should be unique."""
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        gen = CandidateGenerator(
            n_orbitals=lih_hamiltonian.n_orbitals,
            n_alpha=lih_hamiltonian.n_alpha,
            n_beta=lih_hamiltonian.n_beta,
        )
        candidates = gen.generate_excitations(hf, max_rank=4, min_rank=1)

        unique = torch.unique(candidates, dim=0)
        assert len(unique) == len(candidates), (
            f"Found {len(candidates) - len(unique)} duplicate candidates"
        )

    @pytest.mark.molecular
    def test_excludes_existing_basis(self, lih_hamiltonian):
        """Should not generate configs already in the existing basis."""
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        existing = _generate_cisd_configs(lih_hamiltonian)

        gen = CandidateGenerator(
            n_orbitals=lih_hamiltonian.n_orbitals,
            n_alpha=lih_hamiltonian.n_alpha,
            n_beta=lih_hamiltonian.n_beta,
        )
        candidates = gen.generate_excitations(
            hf, max_rank=4, min_rank=3, exclude=existing
        )

        # Check that no candidate is in existing
        if len(candidates) > 0:
            for i in range(len(candidates)):
                matches = (existing == candidates[i]).all(dim=1)
                assert not matches.any(), (
                    f"Candidate {i} is already in existing basis"
                )

    @pytest.mark.molecular
    def test_max_candidates_limit(self, lih_hamiltonian):
        """Should respect max_candidates limit."""
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        gen = CandidateGenerator(
            n_orbitals=lih_hamiltonian.n_orbitals,
            n_alpha=lih_hamiltonian.n_alpha,
            n_beta=lih_hamiltonian.n_beta,
        )
        candidates = gen.generate_excitations(
            hf, max_rank=4, min_rank=1, max_candidates=10
        )
        assert len(candidates) <= 10, f"Expected <= 10, got {len(candidates)}"

    @pytest.mark.molecular
    def test_excitation_rank_correct(self, lih_hamiltonian):
        """Generated configs should have the requested excitation rank."""
        from postprocessing.diversity_selection import compute_excitation_rank
        from krylov.nnci import CandidateGenerator

        hf = lih_hamiltonian.get_hf_state()
        gen = CandidateGenerator(
            n_orbitals=lih_hamiltonian.n_orbitals,
            n_alpha=lih_hamiltonian.n_alpha,
            n_beta=lih_hamiltonian.n_beta,
        )

        # Generate only triples
        triples = gen.generate_excitations(hf, max_rank=3, min_rank=3)
        for i in range(len(triples)):
            rank = compute_excitation_rank(triples[i], hf)
            assert rank == 3, f"Expected rank 3, got {rank}"


# =============================================================================
# Tests: NNCIConfig
# =============================================================================


class TestNNCIConfig:
    """Test NNCI configuration dataclass."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        from krylov.nnci import NNCIConfig

        config = NNCIConfig()
        assert config.max_iterations > 0
        assert config.top_k > 0
        assert config.hidden_dims is not None
        assert len(config.hidden_dims) >= 2
        assert config.training_epochs > 0
        assert config.learning_rate > 0

    def test_config_customization(self):
        """Config should accept custom values."""
        from krylov.nnci import NNCIConfig

        config = NNCIConfig(
            max_iterations=10,
            top_k=50,
            hidden_dims=[128, 64],
            training_epochs=200,
            learning_rate=1e-4,
        )
        assert config.max_iterations == 10
        assert config.top_k == 50
        assert config.hidden_dims == [128, 64]

    def test_config_max_excitation_rank(self):
        """Config should control maximum excitation rank for candidates."""
        from krylov.nnci import NNCIConfig

        config = NNCIConfig(max_excitation_rank=3)
        assert config.max_excitation_rank == 3


# =============================================================================
# Tests: NNCIActiveLearning
# =============================================================================


class TestNNCIActiveLearning:
    """Test the active learning iteration loop."""

    @pytest.mark.molecular
    def test_creation(self, lih_hamiltonian):
        """NNCIActiveLearning should be creatable from Hamiltonian and basis."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=1, top_k=5)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        assert nnci is not None

    @pytest.mark.molecular
    def test_single_iteration(self, lih_hamiltonian):
        """One iteration should produce results with energy and basis size."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=1, top_k=5, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        assert "energy" in results, "Results should contain 'energy'"
        assert "basis_size" in results, "Results should contain 'basis_size'"
        assert "iterations" in results, "Results should contain 'iterations'"
        assert isinstance(results["energy"], float)
        assert results["basis_size"] >= len(basis), (
            f"Basis should not shrink: {results['basis_size']} < {len(basis)}"
        )

    @pytest.mark.molecular
    def test_energy_is_variational(self, lih_hamiltonian):
        """NNCI energy should be >= FCI energy (variational principle)."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=2, top_k=10, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        fci_energy = lih_hamiltonian.fci_energy()
        # Allow small numerical tolerance below FCI
        assert results["energy"] >= fci_energy - 1e-6, (
            f"NNCI energy {results['energy']:.8f} below FCI {fci_energy:.8f} "
            f"(violates variational principle)"
        )

    @pytest.mark.molecular
    def test_convergence_multiple_iterations(self, lih_hamiltonian):
        """Energy should improve (or stay same) across iterations."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=3, top_k=10, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        energies = results["energy_history"]
        assert len(energies) >= 1, "Should have at least one energy entry"
        # Each iteration's energy should be <= previous (or very close)
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-10, (
                f"Energy increased at iteration {i}: "
                f"{energies[i]:.8f} > {energies[i-1]:.8f}"
            )

    @pytest.mark.molecular
    def test_basis_growth_bounded(self, lih_hamiltonian):
        """Basis size should respect max_basis_size limit."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        max_basis = len(basis) + 20
        config = NNCIConfig(
            max_iterations=5,
            top_k=50,
            max_basis_size=max_basis,
            training_epochs=50,
        )
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        assert results["basis_size"] <= max_basis, (
            f"Basis {results['basis_size']} exceeds max {max_basis}"
        )

    @pytest.mark.molecular
    def test_early_stopping_on_convergence(self, lih_hamiltonian):
        """Should stop early if energy converges within threshold."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(
            max_iterations=20,  # High max, but should converge early
            top_k=5,
            training_epochs=50,
            convergence_threshold=1e-4,  # mHa convergence
        )
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        # Should have stopped before max_iterations
        assert results["iterations"] <= 20, "Should have completed"
        assert results["converged"] is True or results["iterations"] < 20, (
            "Should either converge or complete iterations"
        )

    @pytest.mark.molecular
    def test_results_contain_final_basis(self, lih_hamiltonian):
        """Results should include the final basis tensor."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=1, top_k=5, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        assert "final_basis" in results
        assert isinstance(results["final_basis"], torch.Tensor)
        assert results["final_basis"].shape[1] == lih_hamiltonian.num_sites


# =============================================================================
# Tests: Pipeline Integration
# =============================================================================


class TestNNCIPipelineIntegration:
    """Integration with the existing pipeline / Hamiltonian systems."""

    @pytest.mark.molecular
    def test_nnci_h2_exact(self, h2_hamiltonian):
        """On H2 (4 configs total), NNCI should find exact FCI energy.

        H2 has only 4 particle-conserving configs, so CISD already spans
        the entire space. NNCI should reach FCI exactly.
        """
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(h2_hamiltonian)
        config = NNCIConfig(max_iterations=1, top_k=5, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=h2_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        fci_energy = h2_hamiltonian.fci_energy()
        error_mha = abs(results["energy"] - fci_energy) * 1000
        assert error_mha < 0.01, (
            f"H2 NNCI should match FCI exactly. Error: {error_mha:.4f} mHa"
        )

    @pytest.mark.molecular
    def test_nnci_lih_chemical_accuracy(self, lih_hamiltonian):
        """NNCI should achieve chemical accuracy on LiH."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=3, top_k=20, training_epochs=100)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        fci_energy = lih_hamiltonian.fci_energy()
        error_mha = abs(results["energy"] - fci_energy) * 1000
        # LiH/STO-3G: CISD already achieves chemical accuracy,
        # NNCI should at least match it
        assert error_mha < 1.6, (
            f"LiH NNCI error {error_mha:.4f} mHa exceeds chemical accuracy"
        )

    @pytest.mark.molecular
    def test_nnci_energy_at_least_as_good_as_direct_ci(self, lih_hamiltonian):
        """NNCI should find energy at least as good as Direct-CI basis."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)

        # Direct-CI energy (just diagonalize CISD basis)
        direct_ci_energy, _ = _diagonalize_subspace(lih_hamiltonian, basis)

        # NNCI energy (should be <= since basis can only grow)
        config = NNCIConfig(max_iterations=3, top_k=10, training_epochs=100)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        assert results["energy"] <= direct_ci_energy + 1e-10, (
            f"NNCI energy {results['energy']:.8f} should be <= Direct-CI "
            f"{direct_ci_energy:.8f} (basis only grows)"
        )

    @pytest.mark.molecular
    def test_nnci_preserves_essential_configs(self, lih_hamiltonian):
        """NNCI should preserve all essential configs (HF + singles + doubles)."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig
        from postprocessing.diversity_selection import compute_excitation_rank

        basis = _generate_cisd_configs(lih_hamiltonian)
        hf = lih_hamiltonian.get_hf_state()

        config = NNCIConfig(max_iterations=2, top_k=5, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        final_basis = results["final_basis"]

        # Count essential configs in initial basis
        initial_essential = set()
        for i in range(len(basis)):
            rank = compute_excitation_rank(basis[i], hf)
            if rank <= 2:
                initial_essential.add(tuple(basis[i].cpu().tolist()))

        # Count essential configs in final basis
        final_essential = set()
        for i in range(len(final_basis)):
            rank = compute_excitation_rank(final_basis[i], hf)
            if rank <= 2:
                final_essential.add(tuple(final_basis[i].cpu().tolist()))

        assert initial_essential.issubset(final_essential), (
            f"Lost {len(initial_essential - final_essential)} essential configs"
        )

    @pytest.mark.molecular
    def test_nnci_with_h2o(self, h2o_hamiltonian):
        """NNCI should work on H2O (14 qubits, 441 configs)."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(h2o_hamiltonian)
        config = NNCIConfig(max_iterations=2, top_k=10, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=h2o_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        fci_energy = h2o_hamiltonian.fci_energy()
        error_mha = abs(results["energy"] - fci_energy) * 1000
        # H2O/STO-3G has 441 configs, CISD should be close
        assert error_mha < 5.0, (
            f"H2O NNCI error {error_mha:.4f} mHa too large"
        )

    @pytest.mark.molecular
    def test_added_configs_are_particle_conserving(self, lih_hamiltonian):
        """Configs added by NNCI must conserve particle number."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        n_initial = len(basis)
        n_orb = lih_hamiltonian.n_orbitals
        n_alpha = lih_hamiltonian.n_alpha
        n_beta = lih_hamiltonian.n_beta

        config = NNCIConfig(max_iterations=2, top_k=10, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        final_basis = results["final_basis"]
        for i in range(len(final_basis)):
            c = final_basis[i]
            alpha_count = c[:n_orb].sum().item()
            beta_count = c[n_orb:].sum().item()
            assert alpha_count == n_alpha, (
                f"Config {i}: alpha count {alpha_count} != {n_alpha}"
            )
            assert beta_count == n_beta, (
                f"Config {i}: beta count {beta_count} != {n_beta}"
            )


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestNNCIEdgeCases:
    """Test edge cases and robustness."""

    @pytest.mark.molecular
    def test_no_candidates_available(self, h2_hamiltonian):
        """When CISD spans the full space, NNCI should gracefully handle no new candidates."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        # H2 has only 4 configs; CISD already covers all of them
        basis = _generate_cisd_configs(h2_hamiltonian)
        config = NNCIConfig(max_iterations=3, top_k=10, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=h2_hamiltonian,
            initial_basis=basis,
            config=config,
        )
        results = nnci.run()

        # Should not crash; energy should match FCI
        fci_energy = h2_hamiltonian.fci_energy()
        error_mha = abs(results["energy"] - fci_energy) * 1000
        assert error_mha < 0.01

    @pytest.mark.molecular
    def test_single_config_initial_basis(self, lih_hamiltonian):
        """Should handle starting with only HF state."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        hf = lih_hamiltonian.get_hf_state().unsqueeze(0)
        config = NNCIConfig(max_iterations=2, top_k=20, training_epochs=50)
        nnci = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=hf,
            config=config,
        )
        results = nnci.run()

        # Should expand beyond just HF
        assert results["basis_size"] > 1, "Basis should grow from single HF state"
        assert isinstance(results["energy"], float)

    def test_classifier_gradient_flow(self):
        """Verify gradients flow through the classifier."""
        from krylov.nnci import ConfigImportanceClassifier

        n_sites = 12
        clf = ConfigImportanceClassifier(n_sites=n_sites, hidden_dims=[32, 16])
        x = torch.randn(5, n_sites, requires_grad=True)
        out = clf(x)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in clf.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    @pytest.mark.molecular
    def test_reproducibility_with_seed(self, lih_hamiltonian):
        """Same seed should produce same results."""
        from krylov.nnci import NNCIActiveLearning, NNCIConfig

        basis = _generate_cisd_configs(lih_hamiltonian)
        config = NNCIConfig(max_iterations=2, top_k=5, training_epochs=50, seed=42)

        nnci1 = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis.clone(),
            config=config,
        )
        results1 = nnci1.run()

        nnci2 = NNCIActiveLearning(
            hamiltonian=lih_hamiltonian,
            initial_basis=basis.clone(),
            config=config,
        )
        results2 = nnci2.run()

        assert abs(results1["energy"] - results2["energy"]) < 1e-10, (
            f"Results should be reproducible with same seed: "
            f"{results1['energy']:.8f} vs {results2['energy']:.8f}"
        )
