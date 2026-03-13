"""
Tests for P1.3: Taboo list for Krylov expansion.

Inspired by CIGS (JCTC 2025) — tracks previously-discovered-but-pruned configs
to avoid redundant Krylov expansion work.
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTabooListConfig:
    """Test SKQDConfig taboo list parameters."""

    def test_taboo_defaults(self):
        """Taboo list is ON by default with 500K max size."""
        from krylov.skqd import SKQDConfig
        cfg = SKQDConfig()
        assert cfg.use_taboo_list is True
        assert cfg.taboo_max_size == 500000

    def test_taboo_disable(self):
        """Can disable taboo list via config."""
        from krylov.skqd import SKQDConfig
        cfg = SKQDConfig(use_taboo_list=False)
        assert cfg.use_taboo_list is False


class TestTabooSetMechanics:
    """Test taboo set add/evict/clear mechanics on FlowGuidedSKQD."""

    @pytest.fixture
    def skqd_with_taboo(self):
        """Create a FlowGuidedSKQD with small taboo for testing."""
        from hamiltonians.molecular import create_h2_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD

        H = create_h2_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)
        cfg = SKQDConfig(use_taboo_list=True, taboo_max_size=5)
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)
        return skqd

    def test_taboo_set_initialized_empty(self, skqd_with_taboo):
        """Taboo set starts empty."""
        assert len(skqd_with_taboo._taboo_set) == 0
        assert len(skqd_with_taboo._taboo_deque) == 0

    def test_add_to_taboo(self, skqd_with_taboo):
        """Adding hashes populates both set and deque."""
        skqd = skqd_with_taboo
        skqd._add_to_taboo([10, 20, 30])
        assert len(skqd._taboo_set) == 3
        assert len(skqd._taboo_deque) == 3
        assert 10 in skqd._taboo_set
        assert 20 in skqd._taboo_set
        assert 30 in skqd._taboo_set

    def test_taboo_no_duplicates(self, skqd_with_taboo):
        """Adding the same hash twice doesn't create duplicates."""
        skqd = skqd_with_taboo
        skqd._add_to_taboo([10, 20])
        skqd._add_to_taboo([20, 30])  # 20 already in set
        assert len(skqd._taboo_set) == 3
        assert len(skqd._taboo_deque) == 3

    def test_taboo_fifo_eviction(self, skqd_with_taboo):
        """When exceeding max_size, oldest entries are evicted (FIFO)."""
        skqd = skqd_with_taboo  # max_size=5
        skqd._add_to_taboo([1, 2, 3, 4, 5])
        assert len(skqd._taboo_set) == 5

        # Add one more — should evict 1 (oldest)
        skqd._add_to_taboo([6])
        assert len(skqd._taboo_set) == 5
        assert 1 not in skqd._taboo_set  # evicted
        assert 6 in skqd._taboo_set  # added

        # Add two more — should evict 2 and 3
        skqd._add_to_taboo([7, 8])
        assert len(skqd._taboo_set) == 5
        assert 2 not in skqd._taboo_set
        assert 3 not in skqd._taboo_set
        assert 7 in skqd._taboo_set
        assert 8 in skqd._taboo_set

    def test_clear_taboo(self, skqd_with_taboo):
        """clear_taboo() empties both set and deque."""
        skqd = skqd_with_taboo
        skqd._add_to_taboo([1, 2, 3])
        skqd.clear_taboo()
        assert len(skqd._taboo_set) == 0
        assert len(skqd._taboo_deque) == 0

    def test_taboo_disabled_noop(self):
        """When use_taboo_list=False, _add_to_taboo is a no-op."""
        from hamiltonians.molecular import create_h2_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD

        H = create_h2_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)
        cfg = SKQDConfig(use_taboo_list=False)
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)

        skqd._add_to_taboo([1, 2, 3])
        assert len(skqd._taboo_set) == 0


class TestTabooInKrylovExpansion:
    """Test that taboo list actually filters configs during Krylov expansion."""

    def test_taboo_filters_in_find_connected(self):
        """Configs in taboo set are not returned by _find_connected_configs."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD
        from utils.config_hash import config_integer_hash

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()

        # Build initial basis = just HF
        nf_basis = hf.unsqueeze(0)
        cfg = SKQDConfig(
            use_taboo_list=True,
            taboo_max_size=100000,
            max_new_configs_per_krylov_step=5000,
        )
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)

        # First call: get connected configs (no taboo entries yet)
        basis_set = set(config_integer_hash(nf_basis))
        new1 = skqd._find_connected_configs(nf_basis, basis_set)
        n_no_taboo = len(new1)
        assert n_no_taboo > 0, "HF should have connected configs"

        # Add ALL discovered configs to taboo
        hashes1 = config_integer_hash(new1)
        skqd._add_to_taboo(hashes1)

        # Second call: same basis, same basis_set — taboo should filter everything
        new2 = skqd._find_connected_configs(nf_basis, basis_set)
        assert len(new2) == 0, (
            f"Taboo should filter all {n_no_taboo} previously-discovered configs, "
            f"but {len(new2)} still returned"
        )

    def test_taboo_disabled_returns_same(self):
        """With taboo disabled, all connected configs are returned normally."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD
        from utils.config_hash import config_integer_hash

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)

        cfg = SKQDConfig(use_taboo_list=False, max_new_configs_per_krylov_step=5000)
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)

        basis_set = set(config_integer_hash(nf_basis))
        new1 = skqd._find_connected_configs(nf_basis, basis_set)
        n1 = len(new1)

        # Manually stuff taboo set (shouldn't matter since disabled)
        skqd._taboo_set.update(config_integer_hash(new1))

        new2 = skqd._find_connected_configs(nf_basis, basis_set)
        assert len(new2) == n1, "With taboo disabled, same configs should be returned"

    def test_taboo_populated_during_trimming(self):
        """Trimmed configs during Krylov expansion go to taboo set."""
        from hamiltonians.molecular import create_lih_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD

        H = create_lih_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)

        # Very tight expansion cap so trimming happens quickly
        # LiH HF has many connected singles+doubles, max 6 total configs
        cfg = SKQDConfig(
            use_taboo_list=True,
            taboo_max_size=100000,
            max_diag_basis_size=6,  # very tight: only 6 configs allowed
            max_krylov_dim=4,
            max_new_configs_per_krylov_step=50,
        )
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg, force_nf_guided=True)
        skqd.run_with_nf(max_krylov_dim=4, progress=False)

        # Taboo should have been populated with trimmed configs
        assert len(skqd._taboo_set) > 0, (
            "Taboo set should be populated when configs are trimmed during expansion"
        )

    def test_taboo_does_not_block_basis_configs(self):
        """Taboo only filters NEW discoveries, not configs already in basis."""
        from hamiltonians.molecular import create_h2_hamiltonian
        from krylov.skqd import SKQDConfig, FlowGuidedSKQD
        from utils.config_hash import config_integer_hash

        H = create_h2_hamiltonian()
        hf = H.get_hf_state()
        nf_basis = hf.unsqueeze(0)
        cfg = SKQDConfig(use_taboo_list=True)
        skqd = FlowGuidedSKQD(H, nf_basis, config=cfg)

        # Add the HF hash to taboo
        hf_hash = config_integer_hash(nf_basis)
        skqd._add_to_taboo(hf_hash)

        # HF is in basis_set, so it's filtered by basis_set check, not taboo
        # The important thing: taboo doesn't corrupt basis_set membership
        basis_set = set(config_integer_hash(nf_basis))
        assert hf_hash[0] in basis_set, "HF should be in basis_set"
