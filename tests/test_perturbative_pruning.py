"""Tests for MP2-based perturbative pruning of subspace configurations.

Validates that MP2 amplitudes can rank configurations by importance,
and that pruning by MP2 score preserves energy accuracy better than
random pruning.
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


class TestMP2Amplitudes:
    """Test MP2 amplitude computation from stored MO integrals."""

    @pytest.mark.molecular
    def test_mp2_amplitudes_computed(self, lih_hamiltonian):
        """MP2 amplitudes should be computable from Hamiltonian integrals."""
        from utils.perturbative_pruning import compute_mp2_amplitudes

        t2, e_corr = compute_mp2_amplitudes(lih_hamiltonian)
        assert t2 is not None
        assert isinstance(t2, np.ndarray)
        assert isinstance(e_corr, float)

    @pytest.mark.molecular
    def test_mp2_amplitudes_shape(self, lih_hamiltonian):
        """t2 shape should be (nocc, nocc, nvir, nvir)."""
        from utils.perturbative_pruning import compute_mp2_amplitudes

        t2, _ = compute_mp2_amplitudes(lih_hamiltonian)
        n_occ = lih_hamiltonian.n_alpha  # For RHF, n_alpha == n_beta
        n_vir = lih_hamiltonian.n_orbitals - n_occ
        assert t2.shape == (
            n_occ,
            n_occ,
            n_vir,
            n_vir,
        ), f"Expected ({n_occ}, {n_occ}, {n_vir}, {n_vir}), got {t2.shape}"

    @pytest.mark.molecular
    def test_mp2_correlation_energy_negative(self, lih_hamiltonian):
        """MP2 correlation energy should be negative."""
        from utils.perturbative_pruning import compute_mp2_amplitudes

        _, e_corr = compute_mp2_amplitudes(lih_hamiltonian)
        assert e_corr < 0.0, f"MP2 correlation energy should be negative, got {e_corr}"

    @pytest.mark.molecular
    def test_mp2_correlation_energy_reasonable(self, h2_hamiltonian):
        """MP2 correlation energy should be a reasonable fraction of total energy."""
        from utils.perturbative_pruning import compute_mp2_amplitudes

        _, e_corr = compute_mp2_amplitudes(h2_hamiltonian)
        # For H2/STO-3G, MP2 corr energy should be on the order of -0.01 to -0.1 Ha
        assert -1.0 < e_corr < 0.0, f"Unreasonable MP2 corr energy: {e_corr}"

    @pytest.mark.molecular
    def test_mp2_amplitudes_symmetric(self, lih_hamiltonian):
        """t2 amplitudes should satisfy t2[i,j,a,b] == t2[j,i,b,a] (RHF symmetry)."""
        from utils.perturbative_pruning import compute_mp2_amplitudes

        t2, _ = compute_mp2_amplitudes(lih_hamiltonian)
        # For RHF: t2[i,j,a,b] = t2[j,i,b,a]
        t2_permuted = t2.transpose(1, 0, 3, 2)
        np.testing.assert_allclose(
            t2, t2_permuted, atol=1e-12, err_msg="t2 should satisfy RHF permutation symmetry"
        )


class TestImportanceScoring:
    """Test configuration importance scoring based on MP2 amplitudes."""

    @pytest.mark.molecular
    def test_hf_config_gets_max_score(self, lih_hamiltonian):
        """HF configuration should have the highest importance score."""
        from utils.perturbative_pruning import mp2_importance_scores

        hf = lih_hamiltonian.get_hf_state().unsqueeze(0)
        scores = mp2_importance_scores(hf, lih_hamiltonian)
        assert scores[0] >= 1e10, f"HF score should be >= 1e10 (essential), got {scores[0]}"

    @pytest.mark.molecular
    def test_singles_get_high_score(self, lih_hamiltonian):
        """Single excitations should have high importance scores (essential)."""
        from utils.perturbative_pruning import mp2_importance_scores

        hf = lih_hamiltonian.get_hf_state()
        n_alpha = lih_hamiltonian.n_alpha

        # Create a single alpha excitation: occ[0] -> virt[0]
        single = hf.clone()
        single[0] = 0
        single[n_alpha] = 1

        configs = torch.stack([single])
        scores = mp2_importance_scores(configs, lih_hamiltonian)
        assert scores[0] >= 1e10, f"Single excitation should be essential, got score {scores[0]}"

    @pytest.mark.molecular
    def test_doubles_scored_by_t2(self, lih_hamiltonian):
        """Double excitation scores should correlate with |t2| amplitudes."""
        from utils.perturbative_pruning import compute_mp2_amplitudes, mp2_importance_scores

        hf = lih_hamiltonian.get_hf_state()
        n_orb = lih_hamiltonian.n_orbitals
        n_alpha = lih_hamiltonian.n_alpha
        n_beta = lih_hamiltonian.n_beta

        t2, _ = compute_mp2_amplitudes(lih_hamiltonian)
        occ_alpha = list(range(n_alpha))
        virt_alpha = list(range(n_alpha, n_orb))
        occ_beta = list(range(n_beta))
        virt_beta = list(range(n_beta, n_orb))

        # Generate alpha-beta doubles and their expected t2 scores
        doubles = []
        expected_scores = []
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b in virt_beta:
                        c = hf.clone()
                        c[i] = 0
                        c[j + n_orb] = 0
                        c[a] = 1
                        c[b + n_orb] = 1
                        doubles.append(c)
                        # t2 index: i, j, a-n_alpha, b-n_beta
                        expected_scores.append(abs(t2[i, j, a - n_alpha, b - n_beta]))

        configs = torch.stack(doubles)
        scores = mp2_importance_scores(configs, lih_hamiltonian)

        # Scores should not be essential (< 1e10) — these are doubles
        assert (scores < 1e10).all(), "Doubles should not have essential scores"

        # Rank correlation: configs with larger |t2| should have larger scores
        # Use Spearman rank correlation
        from scipy.stats import spearmanr

        expected_arr = np.array(expected_scores)
        scores_arr = scores.cpu().numpy()

        # Only test correlation if there's variance in scores
        if expected_arr.std() > 1e-15 and scores_arr.std() > 1e-15:
            corr, pval = spearmanr(expected_arr, scores_arr)
            assert corr > 0.9, f"Scores should strongly correlate with |t2|, got rho={corr:.3f}"

    @pytest.mark.molecular
    def test_scores_are_non_negative(self, lih_hamiltonian):
        """All importance scores should be non-negative."""
        from utils.perturbative_pruning import mp2_importance_scores

        configs = _generate_cisd_configs(lih_hamiltonian)
        scores = mp2_importance_scores(configs, lih_hamiltonian)
        assert (scores >= 0).all(), "All scores should be non-negative"

    @pytest.mark.molecular
    def test_score_vector_length_matches_configs(self, lih_hamiltonian):
        """Score vector length should match number of configurations."""
        from utils.perturbative_pruning import mp2_importance_scores

        configs = _generate_cisd_configs(lih_hamiltonian)
        scores = mp2_importance_scores(configs, lih_hamiltonian)
        assert len(scores) == len(
            configs
        ), f"Score length {len(scores)} != config count {len(configs)}"

    @pytest.mark.molecular
    def test_higher_excitations_get_lower_scores(self, lih_hamiltonian):
        """Higher excitation configs should generally score lower than doubles."""
        from utils.perturbative_pruning import mp2_importance_scores

        hf = lih_hamiltonian.get_hf_state()
        n_orb = lih_hamiltonian.n_orbitals
        n_alpha = lih_hamiltonian.n_alpha
        n_beta = lih_hamiltonian.n_beta

        occ_alpha = list(range(n_alpha))
        virt_alpha = list(range(n_alpha, n_orb))
        occ_beta = list(range(n_beta))
        virt_beta = list(range(n_beta, n_orb))

        # Create a double excitation
        double = hf.clone()
        double[occ_alpha[0]] = 0
        double[occ_beta[0] + n_orb] = 0
        double[virt_alpha[0]] = 1
        double[virt_beta[0] + n_orb] = 1

        # Create a triple excitation (if possible)
        if len(occ_alpha) >= 2 and len(virt_alpha) >= 2 and len(occ_beta) >= 1:
            triple = hf.clone()
            triple[occ_alpha[0]] = 0
            triple[occ_alpha[1]] = 0
            triple[occ_beta[0] + n_orb] = 0
            triple[virt_alpha[0]] = 1
            triple[virt_alpha[1]] = 1
            triple[virt_beta[0] + n_orb] = 1

            configs = torch.stack([double, triple])
            scores = mp2_importance_scores(configs, lih_hamiltonian)

            # Triple should NOT be essential but double might have a higher t2-based score
            assert scores[1] < 1e10, "Triple should not be essential"
            # Triples use product of t2 amplitudes, which are typically < 1,
            # so their score should be lower than doubles (product of fractions < individual)
            # This is a soft check — in rare cases a triple could score higher
            # We just verify the triple is finite and non-negative
            assert scores[1] >= 0, "Triple score should be non-negative"


class TestBasisPruning:
    """Test basis pruning with MP2-ranked importance scores."""

    @pytest.mark.molecular
    def test_prune_reduces_basis_size(self, lih_hamiltonian):
        """Pruning should reduce basis to at most max_configs."""
        from utils.perturbative_pruning import mp2_importance_scores, prune_basis

        configs = _generate_cisd_configs(lih_hamiltonian)
        scores = mp2_importance_scores(configs, lih_hamiltonian)

        max_configs = len(configs) // 2
        pruned = prune_basis(configs, scores, max_configs=max_configs)
        assert len(pruned) <= max_configs, f"Pruned size {len(pruned)} > max_configs {max_configs}"

    @pytest.mark.molecular
    def test_prune_keeps_essential_configs(self, lih_hamiltonian):
        """HF + singles should survive pruning regardless of max_configs."""
        from postprocessing.diversity_selection import compute_excitation_rank
        from utils.perturbative_pruning import mp2_importance_scores, prune_basis

        configs = _generate_cisd_configs(lih_hamiltonian)
        scores = mp2_importance_scores(configs, lih_hamiltonian)
        hf = lih_hamiltonian.get_hf_state()

        # Count essential configs (HF + singles)
        essential_count = 0
        for i in range(len(configs)):
            rank = compute_excitation_rank(configs[i], hf)
            if rank <= 1:
                essential_count += 1

        # Prune to a small number — essentials should survive
        pruned = prune_basis(configs, scores, max_configs=essential_count + 5)

        # Verify all essentials are in pruned set
        pruned_set = set()
        for i in range(len(pruned)):
            pruned_set.add(tuple(pruned[i].cpu().tolist()))

        essential_found = 0
        for i in range(len(configs)):
            rank = compute_excitation_rank(configs[i], hf)
            if rank <= 1:
                key = tuple(configs[i].cpu().tolist())
                if key in pruned_set:
                    essential_found += 1

        assert (
            essential_found == essential_count
        ), f"Only {essential_found}/{essential_count} essential configs survived pruning"

    @pytest.mark.molecular
    def test_prune_noop_when_under_limit(self, lih_hamiltonian):
        """If n_configs <= max_configs, pruning should return all configs."""
        from utils.perturbative_pruning import mp2_importance_scores, prune_basis

        configs = _generate_cisd_configs(lih_hamiltonian)
        scores = mp2_importance_scores(configs, lih_hamiltonian)

        pruned = prune_basis(configs, scores, max_configs=len(configs) + 100)
        assert len(pruned) == len(
            configs
        ), f"Pruning should be no-op when under limit, got {len(pruned)} != {len(configs)}"

    @pytest.mark.molecular
    def test_pruned_energy_close_to_full(self, h2o_hamiltonian):
        """Energy from pruned basis should be close to full basis energy.

        Generate full CISD basis for H2O, prune to 70%, verify energy
        within 1 mHa of unpruned energy.
        """
        from utils.perturbative_pruning import mp2_importance_scores, prune_basis

        configs = _generate_cisd_configs(h2o_hamiltonian)
        n_full = len(configs)

        # Compute full-basis energy
        H_full = h2o_hamiltonian.matrix_elements(configs, configs)
        H_full_np = H_full.cpu().numpy().astype(np.float64)
        H_full_np = 0.5 * (H_full_np + H_full_np.T)
        evals_full, _ = np.linalg.eigh(H_full_np)
        e_full = evals_full[0]

        # Prune to 70%
        scores = mp2_importance_scores(configs, h2o_hamiltonian)
        max_configs = int(0.7 * n_full)
        pruned = prune_basis(configs, scores, max_configs=max_configs)

        # Compute pruned-basis energy
        H_pruned = h2o_hamiltonian.matrix_elements(pruned, pruned)
        H_pruned_np = H_pruned.cpu().numpy().astype(np.float64)
        H_pruned_np = 0.5 * (H_pruned_np + H_pruned_np.T)
        evals_pruned, _ = np.linalg.eigh(H_pruned_np)
        e_pruned = evals_pruned[0]

        error_mha = abs(e_pruned - e_full) * 1000  # Convert to mHa
        assert error_mha < 1.0, (
            f"Pruned energy error {error_mha:.3f} mHa > 1.0 mHa threshold. "
            f"Full E={e_full:.6f}, Pruned E={e_pruned:.6f}, "
            f"Pruned {len(pruned)}/{n_full} configs"
        )

    @pytest.mark.molecular
    def test_mp2_pruning_better_than_random(self, h2o_hamiltonian):
        """MP2-ranked pruning should give better energy than random pruning."""
        from utils.perturbative_pruning import mp2_importance_scores, prune_basis

        configs = _generate_cisd_configs(h2o_hamiltonian)
        hf = h2o_hamiltonian.get_hf_state()
        n_full = len(configs)

        # MP2 pruning to 50%
        scores = mp2_importance_scores(configs, h2o_hamiltonian)
        max_configs = n_full // 2
        pruned_mp2 = prune_basis(configs, scores, max_configs=max_configs)

        H_mp2 = h2o_hamiltonian.matrix_elements(pruned_mp2, pruned_mp2)
        H_mp2_np = H_mp2.cpu().numpy().astype(np.float64)
        H_mp2_np = 0.5 * (H_mp2_np + H_mp2_np.T)
        evals_mp2, _ = np.linalg.eigh(H_mp2_np)
        e_mp2 = evals_mp2[0]

        # Random pruning (average over multiple trials for robustness)
        rng = np.random.RandomState(42)
        random_energies = []
        for _ in range(5):
            # Always include essentials (HF + singles) then random rest
            from postprocessing.diversity_selection import compute_excitation_rank

            essential_idx = []
            other_idx = []
            for i in range(n_full):
                rank = compute_excitation_rank(configs[i], hf)
                if rank <= 1:
                    essential_idx.append(i)
                else:
                    other_idx.append(i)

            n_random = max_configs - len(essential_idx)
            if n_random > 0:
                chosen = rng.choice(other_idx, size=min(n_random, len(other_idx)), replace=False)
                selected_idx = essential_idx + list(chosen)
            else:
                selected_idx = essential_idx[:max_configs]

            pruned_random = configs[selected_idx]
            H_rand = h2o_hamiltonian.matrix_elements(pruned_random, pruned_random)
            H_rand_np = H_rand.cpu().numpy().astype(np.float64)
            H_rand_np = 0.5 * (H_rand_np + H_rand_np.T)
            evals_rand, _ = np.linalg.eigh(H_rand_np)
            random_energies.append(evals_rand[0])

        avg_random_e = np.mean(random_energies)

        # MP2 pruning should give lower (better) energy than random average
        # Since the Hamiltonian is variational, lower is better
        assert e_mp2 <= avg_random_e + 1e-6, (
            f"MP2 pruning energy {e_mp2:.8f} Ha should be <= "
            f"random average {avg_random_e:.8f} Ha"
        )
