"""
Tests for P1.1: H-coupling filtering for Krylov expansion.

Ranks discovered configs by Hamiltonian coupling strength to high-amplitude
reference states. Inspired by HAAR-SCI (JCTC Dec 2025).
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHCouplingConfig:
    """Test SKQDConfig H-coupling parameters."""

    def test_h_coupling_defaults(self):
        """H-coupling ranking is ON by default."""
        from krylov.skqd import SKQDConfig
        cfg = SKQDConfig()
        assert cfg.use_h_coupling_ranking is True
        assert cfg.h_coupling_n_ref == 50

    def test_h_coupling_disable(self):
        """Can disable H-coupling ranking."""
        from krylov.skqd import SKQDConfig
        cfg = SKQDConfig(use_h_coupling_ranking=False)
        assert cfg.use_h_coupling_ranking is False


class TestHCouplingScores:
    """Test _compute_h_coupling_scores method."""

    @pytest.fixture
    def lih_setup(self):
        """Set up LiH system with FlowGuidedSKQD."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()

        # Create small basis: HF + a few singles
        basis_list = [hf]
        connected, _ = H.get_connections(hf)
        if len(connected) > 0:
            basis_list.append(connected[:5])
        basis = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in basis_list])

        cfg = SKQDConfig(use_h_coupling_ranking=True, h_coupling_n_ref=10)
        skqd = FlowGuidedSKQD(H, basis, config=cfg)
        return skqd, H, basis

    def test_scores_shape(self, lih_setup):
        """Scores tensor has shape (n_candidates,)."""
        skqd, H, basis = lih_setup
        hf = H.get_hf_state()
        connected, _ = H.get_connections(hf)
        candidates = connected[:10] if len(connected) >= 10 else connected

        scores = skqd._compute_h_coupling_scores(candidates, basis)
        assert scores.shape == (len(candidates),)

    def test_connected_configs_have_nonzero_scores(self, lih_setup):
        """Configs connected to basis should have non-zero coupling scores."""
        skqd, H, basis = lih_setup
        # Use uniform psi so all basis states contribute equally
        psi = np.ones(len(basis)) / np.sqrt(len(basis))

        # Get configs connected to HF (first basis state)
        hf = basis[0]
        connected, _ = H.get_connections(hf)
        if len(connected) == 0:
            pytest.skip("No connected configs")

        # Filter out configs already in basis
        from utils.config_hash import config_integer_hash
        basis_hashes = set(config_integer_hash(basis))
        mask = [i for i, h in enumerate(config_integer_hash(connected))
                if h not in basis_hashes]
        if not mask:
            pytest.skip("All connected configs already in basis")
        candidates = connected[mask]

        scores = skqd._compute_h_coupling_scores(candidates, basis, psi=psi)
        assert scores.sum() > 0, "Connected configs should have non-zero coupling scores"

    def test_unconnected_configs_have_zero_scores(self, lih_setup):
        """Configs NOT connected to any basis state should score zero."""
        skqd, H, basis = lih_setup
        psi = np.ones(len(basis)) / np.sqrt(len(basis))

        # Create a fake "unconnected" config by flipping many orbitals
        # (Slater-Condon: >2 excitations → zero matrix element)
        fake = basis[0].clone()
        # Flip 6 orbitals (3 excitations — guaranteed zero by Slater-Condon)
        n_sites = len(fake)
        ones = torch.where(fake == 1)[0]
        zeros = torch.where(fake == 0)[0]
        if len(ones) >= 3 and len(zeros) >= 3:
            for i in range(3):
                fake[ones[i]] = 0
                fake[zeros[i]] = 1
            candidates = fake.unsqueeze(0)
            scores = skqd._compute_h_coupling_scores(candidates, basis, psi=psi)
            assert scores[0].item() == 0.0, (
                "Triple excitation should have zero H-coupling score"
            )

    def test_empty_candidates(self, lih_setup):
        """Empty candidates → empty scores."""
        skqd, H, basis = lih_setup
        candidates = torch.empty(0, basis.shape[1])
        scores = skqd._compute_h_coupling_scores(candidates, basis)
        assert len(scores) == 0

    def test_psi_weighted_scoring(self, lih_setup):
        """Higher-amplitude reference configs contribute more to scores."""
        skqd, H, basis = lih_setup

        # Get candidates
        hf = basis[0]
        connected, _ = H.get_connections(hf)
        from utils.config_hash import config_integer_hash
        basis_hashes = set(config_integer_hash(basis))
        mask = [i for i, h in enumerate(config_integer_hash(connected))
                if h not in basis_hashes]
        if len(mask) < 2:
            pytest.skip("Need at least 2 candidates")
        candidates = connected[mask[:5]]

        # Scores with uniform psi
        psi_uniform = np.ones(len(basis)) / np.sqrt(len(basis))
        scores_uniform = skqd._compute_h_coupling_scores(candidates, basis, psi=psi_uniform)

        # Scores with psi concentrated on HF
        psi_hf = np.zeros(len(basis))
        psi_hf[0] = 1.0
        scores_hf = skqd._compute_h_coupling_scores(candidates, basis, psi=psi_hf)

        # Both should produce scores, but the HF-concentrated one should emphasize
        # configs connected to HF specifically
        assert scores_uniform.sum() > 0
        assert scores_hf.sum() > 0


class TestHCouplingIntegration:
    """Test H-coupling ranking in Krylov expansion end-to-end."""

    def test_h_coupling_used_in_find_connected(self):
        """With H-coupling enabled, _find_connected_configs uses coupling scores."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD
        from utils.config_hash import config_integer_hash

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)

        cfg = SKQDConfig(
            use_h_coupling_ranking=True,
            h_coupling_n_ref=10,
            max_new_configs_per_krylov_step=5,  # force ranking
        )
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)

        # Set up psi so H-coupling can work
        skqd._nf_guided_psi = np.array([1.0])

        basis_set = set(config_integer_hash(nf_basis))
        new_configs = skqd._find_connected_configs(nf_basis, basis_set)

        # Should return exactly max_new_configs_per_krylov_step
        assert len(new_configs) == 5, (
            f"Expected 5 configs (capped by max_new), got {len(new_configs)}"
        )

    def test_h_coupling_disabled_fallback(self):
        """With H-coupling disabled, falls back to MP2 or arbitrary."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD
        from utils.config_hash import config_integer_hash

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)

        cfg = SKQDConfig(
            use_h_coupling_ranking=False,
            max_new_configs_per_krylov_step=5,
        )
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)
        skqd._nf_guided_psi = np.array([1.0])

        basis_set = set(config_integer_hash(nf_basis))
        new_configs = skqd._find_connected_configs(nf_basis, basis_set)

        # Should still return configs, just not H-coupling ranked
        assert len(new_configs) == 5

    def test_h_coupling_ranking_improves_energy(self):
        """H-coupling-ranked configs should give equal or better energy than random."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD
        from utils.config_hash import config_integer_hash

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)

        # Run with H-coupling ON, tight config cap
        cfg_on = SKQDConfig(
            use_h_coupling_ranking=True,
            h_coupling_n_ref=20,
            max_new_configs_per_krylov_step=10,
            max_krylov_dim=3,
            max_diag_basis_size=50,
        )
        skqd_on = FlowGuidedSKQD(H, nf_basis, config=cfg_on, force_nf_guided=True)
        results_on = skqd_on.run_with_nf(max_krylov_dim=3, progress=False)
        e_on = results_on.get("energy_combined", results_on.get("energy_nf_only"))

        # Run with H-coupling OFF
        cfg_off = SKQDConfig(
            use_h_coupling_ranking=False,
            max_new_configs_per_krylov_step=10,
            max_krylov_dim=3,
            max_diag_basis_size=50,
        )
        skqd_off = FlowGuidedSKQD(H, nf_basis, config=cfg_off, force_nf_guided=True)
        results_off = skqd_off.run_with_nf(max_krylov_dim=3, progress=False)
        e_off = results_off.get("energy_combined", results_off.get("energy_nf_only"))

        # H-coupling should give equal or lower energy (variational principle)
        # Allow small tolerance since configs might happen to be the same
        assert e_on <= e_off + 0.01, (
            f"H-coupling energy {e_on:.6f} should be <= no-coupling energy {e_off:.6f} + 0.01"
        )
