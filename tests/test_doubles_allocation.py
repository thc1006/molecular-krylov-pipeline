"""Tests for PR 1.3: Adaptive doubles allocation.

The bug (B3): αα→ββ→αβ doubles share a single counter with max_doubles=5000.
For systems with many orbitals, αα fills the budget first and αβ doubles
(most important for electron correlation) get starved.

Fix: Proportional allocation — each type gets its fair share of the budget,
with αβ getting at least 50% since it dominates correlation energy.
"""

import pytest
import torch
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDoublesAllocation:
    """Test that doubles allocation is proportional, not sequential."""

    @pytest.mark.molecular
    def test_ab_doubles_not_starved(self, n2_hamiltonian):
        """For N2 (7 occ, 3 virt per spin), αβ doubles must not be starved.

        N2: n_alpha=7, n_beta=7, n_orb=10
        - αα doubles: C(7,2)*C(3,2) = 21*3 = 63
        - ββ doubles: C(7,2)*C(3,2) = 21*3 = 63
        - αβ doubles: 7*7*3*3 = 441
        Total: 567 (well under 5000, so no starvation here)

        But for larger systems (40Q), αα could be 10K+ and αβ gets 0.
        We test the proportional allocation logic even for N2.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(n2_hamiltonian, config=config)
        essential = pipeline._generate_essential_configs()

        n_orb = n2_hamiltonian.n_orbitals
        n_alpha = n2_hamiltonian.n_alpha
        n_beta = n2_hamiltonian.n_beta
        hf = n2_hamiltonian.get_hf_state()

        # Count doubles by type
        aa_count = 0
        bb_count = 0
        ab_count = 0

        for cfg in essential:
            alpha_diff = (cfg[:n_orb] != hf[:n_orb]).sum().item()
            beta_diff = (cfg[n_orb:] != hf[n_orb:]).sum().item()

            if alpha_diff == 4 and beta_diff == 0:
                aa_count += 1
            elif alpha_diff == 0 and beta_diff == 4:
                bb_count += 1
            elif alpha_diff == 2 and beta_diff == 2:
                ab_count += 1

        # αβ doubles should not be zero or drastically less than αα
        assert ab_count > 0, "αβ doubles completely starved!"
        total_doubles = aa_count + bb_count + ab_count
        if total_doubles > 0:
            ab_fraction = ab_count / total_doubles
            # αβ should be at least 30% of doubles (it's 78% of total possible for N2)
            assert ab_fraction >= 0.3, (
                f"αβ doubles only {ab_fraction:.1%} of total "
                f"(αα={aa_count}, ββ={bb_count}, αβ={ab_count})"
            )

    @pytest.mark.molecular
    def test_all_double_types_present(self, lih_hamiltonian):
        """LiH should have all 3 types of doubles."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=config)
        essential = pipeline._generate_essential_configs()

        n_orb = lih_hamiltonian.n_orbitals
        hf = lih_hamiltonian.get_hf_state()

        types_found = set()
        for cfg in essential:
            alpha_diff = (cfg[:n_orb] != hf[:n_orb]).sum().item()
            beta_diff = (cfg[n_orb:] != hf[n_orb:]).sum().item()

            if alpha_diff == 4 and beta_diff == 0:
                types_found.add("aa")
            elif alpha_diff == 0 and beta_diff == 4:
                types_found.add("bb")
            elif alpha_diff == 2 and beta_diff == 2:
                types_found.add("ab")

        assert "aa" in types_found, "Missing αα doubles"
        assert "bb" in types_found, "Missing ββ doubles"
        assert "ab" in types_found, "Missing αβ doubles"

    @pytest.mark.molecular
    def test_essential_configs_include_hf_and_singles(self, h2_hamiltonian):
        """Essential configs must include HF state and all single excitations."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(h2_hamiltonian, config=config)
        essential = pipeline._generate_essential_configs()

        hf = h2_hamiltonian.get_hf_state()
        n_orb = h2_hamiltonian.n_orbitals

        # HF must be present
        hf_found = any(torch.all(cfg == hf) for cfg in essential)
        assert hf_found, "HF state not in essential configs"

        # Count singles
        singles_count = 0
        for cfg in essential:
            alpha_diff = (cfg[:n_orb] != hf[:n_orb]).sum().item()
            beta_diff = (cfg[n_orb:] != hf[n_orb:]).sum().item()
            # Single excitation: 2 sites differ in one spin channel
            if (alpha_diff == 2 and beta_diff == 0) or (alpha_diff == 0 and beta_diff == 2):
                singles_count += 1

        # H2: 1 occ, 1 virt per spin → 1 alpha single + 1 beta single = 2
        assert singles_count == 2, f"Expected 2 singles for H2, got {singles_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
