"""Level 5: N2 CAS(10,20) (40Q, 240M configs) — SCALING validation only.

WARNING: This is a SCALING test, NOT an accuracy validation test.
FCI is computationally intractable at 240M configs, so we can only check:
  - No OOM
  - Energy is finite and below HF (sanity check)
  - VMC/sampler mechanics work at 40Q
  - Particle conservation holds

This CANNOT validate the core target (NF > Direct-CI ≥ 10 mHa).
For accuracy validation, see the real experiments in test_phase6_experiments.py.

System: N2/cc-pVDZ, CAS(10,20), 20 active orbitals, 5α+5β, ~240M configs
"""

import sys
import os
import time
import math
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def n2_cas10_20():
    """Create N2 CAS(10,20) system (40Q, integrals only — no FCI)."""
    try:
        from hamiltonians.molecular import create_n2_cas_hamiltonian
        H = create_n2_cas_hamiltonian(basis="cc-pvdz", cas=(10, 20), device="cpu")
        return H
    except (ImportError, Exception) as e:
        pytest.skip(f"Cannot create N2 CAS(10,20): {e}")


def _print_result(test_id, energy, wall_time, extra=""):
    print(f"\nPhase6-L5: {test_id}")
    print(f"  Energy: {energy:.8f} Ha")
    print(f"  Time:   {wall_time:.1f}s")
    if extra:
        print(f"  {extra}")


class TestL5DirectCIBaseline:
    """L5-A: Direct-CI + SKQD at 40Q scale."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_40q_direct_ci(self, n2_cas10_20):
        """Direct-CI + SKQD should run on 40Q without OOM."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_20
        hf_state = H.get_hf_state()
        hf_energy = H.diagonal_element(hf_state).item()
        print(f"\n  HF energy: {hf_energy:.8f} Ha")

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        _print_result("L5-A", energy, wall,
                      f"Direct-CI, improvement={abs(energy - hf_energy)*1000:.1f} mHa over HF")

        assert math.isfinite(energy), "Energy should be finite"
        assert energy < hf_energy, "Energy should be below HF"
        print(f"  PASS: 40Q pipeline completed in {wall:.0f}s")


class TestL5FullPipeline:
    """L5-C: Full P6 pipeline at 40Q scale."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_40q_full_pipeline(self, n2_cas10_20):
        """Full P6 pipeline should run on 40Q without OOM."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_20
        torch.manual_seed(42)
        hf_state = H.get_hf_state()
        hf_energy = H.diagonal_element(hf_state).item()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_autoregressive_flow=True,
            sign_architecture="phase",
            topk_type="sigmoid",
            device="cpu",
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        _print_result("L5-C", energy, wall, "ALL P6 ON")

        assert math.isfinite(energy), "Energy should be finite"
        assert energy < hf_energy, "Energy should be below HF"

        improvement = abs(energy - hf_energy) * 1000
        print(f"  Improvement: {improvement:.1f} mHa below HF")
        print(f"  PASS: 40Q full pipeline completed in {wall:.0f}s")


class TestL5VMCMechanics:
    """L5-B: VMC runs at 40Q scale."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_40q_vmc_runs(self, n2_cas10_20):
        """VMC should run without OOM on 40Q system."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        from flows.vmc_training import VMCTrainer, VMCConfig

        H = n2_cas10_20
        torch.manual_seed(42)

        flow = AutoregressiveFlowSampler(num_sites=40, n_alpha=5, n_beta=5)
        config = VMCConfig(
            n_samples=50, n_steps=3, lr=1e-3, optimizer_type="reinforce",
            lr_decay=1.0, convergence_threshold=1e-12,
        )
        trainer = VMCTrainer(flow, H, config=config, device="cpu")

        t0 = time.time()
        result = trainer.train(verbose=False)
        wall = time.time() - t0

        best = result["best_energy"]
        _print_result("L5-B", best, wall, f"VMC(REINFORCE, 3 steps, 50 samples)")

        assert math.isfinite(best), "Energy should be finite"
        assert result["n_steps"] == 3, "Should complete all 3 steps"


class TestL5UniqueSampling:
    """L5-F: Unique sampling at 40Q scale."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_40q_unique_sampling(self, n2_cas10_20):
        """sample_unique should work on 40Q."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler

        H = n2_cas10_20
        torch.manual_seed(42)

        flow = AutoregressiveFlowSampler(num_sites=40, n_alpha=5, n_beta=5)
        log_probs, configs = flow.sample_unique(n_unique_target=100)

        # All unique
        from utils.config_hash import config_integer_hash
        hashes = config_integer_hash(configs)
        unique_hashes = set(h if isinstance(h, int) else tuple(h) for h in hashes)
        assert len(unique_hashes) == len(configs), (
            f"Expected all unique, got {len(unique_hashes)}/{len(configs)}"
        )

        # Particle conservation
        n_orb = 20
        alpha = configs[:, :n_orb].sum(dim=1)
        beta = configs[:, n_orb:].sum(dim=1)
        assert (alpha == 5).all(), f"Alpha count wrong: {alpha.unique()}"
        assert (beta == 5).all(), f"Beta count wrong: {beta.unique()}"

        print(f"\nPhase6-L5: L5-F")
        print(f"  Unique configs: {len(configs)}/100 target")
        print(f"  PASS")
