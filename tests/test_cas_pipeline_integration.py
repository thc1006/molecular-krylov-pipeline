"""PR 3.3b: CAS active space pipeline integration tests.

End-to-end tests verifying that FlowGuidedKrylovPipeline works correctly
on CAS (Complete Active Space) systems at increasing scale:

- CAS(10,8): 8 active orbitals, 3136 configs — "medium" tier, FCI reference
- CAS(10,10): 10 active orbitals, 63504 configs — "very_large" tier, 40Q-scale path

These tests validate:
1. Pipeline runs to completion without crash/OOM on CAS systems
2. Energy accuracy vs FCI-in-CAS reference (CAS(10,8))
3. adapt_to_system_size() tier classification is correct
4. Essential config generation preserves particle conservation
5. SKQD handles CAS Hamiltonians (Krylov expansion, sparse eigensolver)

Usage:
    uv run pytest tests/test_cas_pipeline_integration.py -v --no-header -m slow
"""

import math
import os
import sys
from math import comb

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for CAS tests")


# ---------------------------------------------------------------------------
# Module-scoped fixtures (CAS Hamiltonians are expensive to create)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n2_cas10_8():
    """N2/cc-pVDZ CAS(10,8) Hamiltonian: 8 orbitals, 3136 configs."""
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 8), device="cpu"
    )
    return H


@pytest.fixture(scope="module")
def n2_cas10_10():
    """N2/cc-pVDZ CAS(10,10) Hamiltonian: 10 orbitals, 63504 configs."""
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 10), device="cpu"
    )
    return H


# ---------------------------------------------------------------------------
# CAS(10,8) — "medium" tier, exact FCI reference available
# ---------------------------------------------------------------------------

CHEMICAL_ACCURACY_HA = 1.594e-3  # 1.0 kcal/mol


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS10_8PipelineIntegration:
    """Full pipeline on N2/cc-pVDZ CAS(10,8): 3136 configs, medium tier."""

    def test_hamiltonian_shape(self, n2_cas10_8):
        """CAS(10,8) Hamiltonian has correct dimensions."""
        H = n2_cas10_8
        assert H.n_orbitals == 8
        assert H.n_alpha == 5
        assert H.n_beta == 5
        assert H.num_sites == 16  # 2 * 8 spin orbitals
        n_configs = comb(8, 5) ** 2
        assert n_configs == 3136

    def test_tier_classification(self, n2_cas10_8):
        """3136 configs should be classified as 'medium' tier."""
        from pipeline import PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device="cpu"
        )
        n_valid = comb(8, 5) ** 2  # 3136
        config.adapt_to_system_size(n_valid, verbose=False)
        # 1000 < 3136 <= 5000 → medium
        assert config.skip_nf_training is True  # Within 20K threshold
        assert config.max_accumulated_basis >= n_valid

    def test_fci_energy_computable(self, n2_cas10_8):
        """FCI energy within CAS(10,8) space is computable and reasonable."""
        H = n2_cas10_8
        fci_e = H.fci_energy()
        assert math.isfinite(fci_e), f"FCI energy not finite: {fci_e}"
        # N2 total energy should be around -109 Ha.
        # CAS energy includes frozen core (e_core), so it should be similar.
        assert fci_e < 0, f"FCI energy is positive (unphysical): {fci_e}"
        assert fci_e > -200, f"FCI energy unreasonably low: {fci_e}"

    def test_pipeline_completes(self, n2_cas10_8):
        """Pipeline runs to completion on CAS(10,8) without error."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        fci_e = H.fci_energy()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=fci_e
        )
        results = pipeline.run(progress=False)

        assert "combined_energy" in results, "Pipeline did not produce combined_energy"
        energy = results["combined_energy"]
        assert math.isfinite(energy), f"Pipeline energy not finite: {energy}"
        assert energy < 0, f"Pipeline energy is positive (unphysical): {energy}"

    def test_energy_within_chemical_accuracy(self, n2_cas10_8):
        """Pipeline energy should be within chemical accuracy of CAS-FCI."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        fci_e = H.fci_energy()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=fci_e
        )
        results = pipeline.run(progress=False)

        energy = results["combined_energy"]
        error_ha = abs(energy - fci_e)
        error_mha = error_ha * 1000

        print(f"CAS(10,8) FCI:      {fci_e:.8f} Ha")
        print(f"CAS(10,8) Pipeline: {energy:.8f} Ha")
        print(f"Error: {error_mha:.4f} mHa")

        assert error_ha < CHEMICAL_ACCURACY_HA, (
            f"Energy error {error_mha:.4f} mHa exceeds chemical accuracy "
            f"({CHEMICAL_ACCURACY_HA * 1000:.1f} mHa)"
        )

    def test_essential_configs_particle_conservation(self, n2_cas10_8):
        """All essential configs must conserve particle number."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            skip_skqd=True,  # Only test config generation, skip diag
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)

        # Generate essential configs (Stage 1 in Direct-CI mode)
        pipeline.train_flow_nqs(progress=False)
        essential = pipeline._essential_configs

        assert essential is not None, "Essential configs not generated"
        assert len(essential) > 0, "Empty essential configs"

        n_orb = H.n_orbitals
        alpha_counts = essential[:, :n_orb].sum(dim=1)
        beta_counts = essential[:, n_orb:].sum(dim=1)

        assert (alpha_counts == H.n_alpha).all(), (
            f"Alpha count violations: expected {H.n_alpha}, "
            f"got {torch.unique(alpha_counts).tolist()}"
        )
        assert (beta_counts == H.n_beta).all(), (
            f"Beta count violations: expected {H.n_beta}, "
            f"got {torch.unique(beta_counts).tolist()}"
        )

    def test_essential_configs_include_hf(self, n2_cas10_8):
        """HF state must be present in essential configs."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            skip_skqd=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        pipeline.train_flow_nqs(progress=False)
        essential = pipeline._essential_configs

        hf_state = H.get_hf_state()
        # Check HF state is in essential configs
        matches = (essential == hf_state.unsqueeze(0)).all(dim=1)
        assert matches.any(), "HF state not found in essential configs"

    def test_essential_configs_contain_doubles(self, n2_cas10_8):
        """Essential configs should contain double excitations."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        n_orb = H.n_orbitals
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            skip_skqd=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        pipeline.train_flow_nqs(progress=False)
        essential = pipeline._essential_configs

        hf_state = H.get_hf_state()

        # Count excitation ranks
        ranks = set()
        for cfg in essential:
            alpha_diff = (cfg[:n_orb] != hf_state[:n_orb]).sum().item()
            beta_diff = (cfg[n_orb:] != hf_state[n_orb:]).sum().item()
            rank = (alpha_diff + beta_diff) // 2
            ranks.add(rank)

        assert 0 in ranks, "HF (rank 0) not found"
        assert 1 in ranks, "Singles (rank 1) not found"
        assert 2 in ranks, "Doubles (rank 2) not found"

        print(f"Excitation ranks in essential configs: {sorted(ranks)}")
        print(f"Essential config count: {len(essential)}")


# ---------------------------------------------------------------------------
# CAS(10,10) — "very_large" tier, 40Q-scale pipeline path
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS10_10PipelineIntegration:
    """Pipeline on N2/cc-pVDZ CAS(10,10): 63504 configs, very_large tier.

    This tests the pipeline's behavior when the total configuration space
    is very large (>20K) but the actual basis used (Direct-CI essential
    configs) is much smaller. Validates the 40Q-scale code path.
    """

    def test_hamiltonian_shape(self, n2_cas10_10):
        """CAS(10,10) Hamiltonian has correct dimensions."""
        H = n2_cas10_10
        assert H.n_orbitals == 10
        assert H.n_alpha == 5
        assert H.n_beta == 5
        assert H.num_sites == 20  # 2 * 10 spin orbitals
        n_configs = comb(10, 5) ** 2
        assert n_configs == 63504

    def test_tier_classification(self, n2_cas10_10):
        """63504 configs should be classified as 'very_large' tier."""
        from pipeline import PipelineConfig

        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device="cpu"
        )
        n_valid = comb(10, 5) ** 2  # 63504
        config.adapt_to_system_size(n_valid, verbose=False)
        # 63504 > 20000 → very_large
        # User set skip_nf_training=True so it should be preserved
        assert config.skip_nf_training is True
        assert config.max_accumulated_basis == 16384

    def test_pipeline_completes_without_oom(self, n2_cas10_10):
        """Pipeline on CAS(10,10) completes without crash or OOM.

        Even though config space is 63504, Direct-CI essential configs
        are only ~876 (HF + singles + doubles with 5 occ, 5 vir).
        SKQD should handle this without issues.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run(progress=False)

        assert "combined_energy" in results
        energy = results["combined_energy"]
        assert math.isfinite(energy), f"Energy not finite: {energy}"
        assert energy < 0, f"Energy is positive (unphysical): {energy}"

    def test_energy_is_reasonable(self, n2_cas10_10):
        """Pipeline energy should be in physically reasonable range.

        N2 total energy ~ -109 Ha. CAS energy includes frozen core,
        so should be in similar range. We check it's between -200 and -50 Ha.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run(progress=False)

        energy = results["combined_energy"]
        assert -200 < energy < -50, (
            f"Energy {energy:.4f} Ha outside reasonable range for N2"
        )
        print(f"CAS(10,10) pipeline energy: {energy:.8f} Ha")

    def test_essential_configs_much_smaller_than_config_space(self, n2_cas10_10):
        """Essential configs should be << total config space for CAS(10,10).

        CAS(10,10): 63504 total configs but only ~876 essential (HF+S+D).
        This validates that Direct-CI doesn't try to enumerate 63504 configs.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            skip_skqd=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        pipeline.train_flow_nqs(progress=False)
        essential = pipeline._essential_configs

        n_total = comb(10, 5) ** 2  # 63504
        n_essential = len(essential)

        # Essential configs should be a small fraction of total
        ratio = n_essential / n_total
        assert ratio < 0.1, (
            f"Essential configs ({n_essential}) are >{10}% of total "
            f"config space ({n_total})"
        )
        # Should be around 876 (1 HF + 50 singles + 825 doubles)
        assert 100 < n_essential < 5000, (
            f"Unexpected essential config count: {n_essential}"
        )

        print(f"CAS(10,10) essential configs: {n_essential} / {n_total} "
              f"({ratio * 100:.2f}%)")

    def test_particle_conservation(self, n2_cas10_10):
        """All essential configs conserve particle number for CAS(10,10)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            skip_skqd=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        pipeline.train_flow_nqs(progress=False)
        essential = pipeline._essential_configs

        n_orb = H.n_orbitals
        alpha_counts = essential[:, :n_orb].sum(dim=1)
        beta_counts = essential[:, n_orb:].sum(dim=1)

        assert (alpha_counts == H.n_alpha).all()
        assert (beta_counts == H.n_beta).all()


# ---------------------------------------------------------------------------
# Cross-CAS comparison tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCASScaleComparison:
    """Compare pipeline behavior across CAS sizes."""

    def test_larger_cas_has_lower_energy(self, n2_cas10_8, n2_cas10_10):
        """CAS(10,10) should give equal or lower energy than CAS(10,8).

        A larger active space captures more correlation energy, so the
        pipeline energy should improve (decrease) with larger CAS.
        Note: Both use Direct-CI (HF+S+D only), so the comparison
        reflects how many important doubles each CAS can capture.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        results = {}
        for label, H in [("cas10_8", n2_cas10_8), ("cas10_10", n2_cas10_10)]:
            config = PipelineConfig(
                subspace_mode="skqd",
                skip_nf_training=True,
                device="cpu",
            )
            pipeline = FlowGuidedKrylovPipeline(H, config=config)
            r = pipeline.run(progress=False)
            results[label] = r["combined_energy"]

        e_8 = results["cas10_8"]
        e_10 = results["cas10_10"]

        print(f"CAS(10,8)  pipeline energy: {e_8:.8f} Ha")
        print(f"CAS(10,10) pipeline energy: {e_10:.8f} Ha")
        print(f"Difference: {(e_10 - e_8) * 1000:.4f} mHa")

        # Larger CAS should give equal or lower energy
        # Allow 5 mHa tolerance for numerical noise from different
        # CASSCF orbital optimizations and truncated bases
        assert e_10 <= e_8 + 0.005, (
            f"CAS(10,10) energy ({e_10:.6f}) is higher than "
            f"CAS(10,8) energy ({e_8:.6f}) by > 5 mHa"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
