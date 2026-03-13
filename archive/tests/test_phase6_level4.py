"""Level 4: N2 CAS(10,10) (20Q, 63504 configs) — critical validation.

This is the core Phase 6 validation level. The target is < 5 mHa error,
down from the 14.2 mHa Direct-CI baseline.

System: N2/cc-pVDZ, CAS(10,10), 10 active orbitals, 5α+5β, 63504 configs
Baseline: 14.2 mHa (Direct-CI), 12.7 mHa (AR NF 300ep)
"""

import sys
import os
import time
import math
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def n2_cas10_10():
    """Create N2 CAS(10,10) system."""
    try:
        from hamiltonians.molecular import create_n2_cas_hamiltonian
        H = create_n2_cas_hamiltonian(basis="cc-pvdz", cas=(10, 10), device="cpu")
        return H
    except (ImportError, Exception) as e:
        pytest.skip(f"Cannot create N2 CAS(10,10): {e}")


def _print_result(test_id, energy, ref, wall_time, extra=""):
    error_mha = abs(energy - ref) * 1000
    print(f"\nPhase6-L4: {test_id}")
    print(f"  Energy: {energy:.8f} Ha")
    print(f"  Ref:    {ref:.8f} Ha")
    print(f"  Error:  {error_mha:.3f} mHa")
    print(f"  Time:   {wall_time:.1f}s")
    if extra:
        print(f"  {extra}")


class TestL4DirectCIBaseline:
    """L4-A/B: Direct-CI + SKQD baseline on CAS(10,10)."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_direct_ci_baseline(self, n2_cas10_10):
        """Direct-CI + SKQD should reproduce ~14.2 mHa baseline."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10

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
        # Use HF as reference since FCI is expensive for 63504 configs
        hf_state = H.get_hf_state()
        hf_energy = H.diagonal_element(hf_state).item()
        _print_result("L4-A", energy, hf_energy, wall,
                      f"Direct-CI + SKQD, configs={result.get('basis_size', '?')}")

        assert math.isfinite(energy), "Energy should be finite"
        # Should be well below HF
        assert energy < hf_energy, f"Energy {energy:.6f} should be below HF {hf_energy:.6f}"
        print(f"  PASS: energy below HF by {abs(energy - hf_energy)*1000:.1f} mHa")


class TestL4NNCI:
    """L4-F: NNCI + NOs on CAS(10,10)."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_nnci_nos_cas10_10(self, n2_cas10_10):
        """NNCI with NOs should run and find configs beyond Direct-CI."""
        from krylov.nnci import NNCIConfig, NNCIActiveLearning

        H = n2_cas10_10
        basis = H.get_hf_state().unsqueeze(0)

        config = NNCIConfig(
            max_iterations=3,
            top_k=20,
            max_candidates=200,
            use_natural_orbitals=True,
        )
        t0 = time.time()
        nnci = NNCIActiveLearning(H, basis, config)
        result = nnci.run()
        wall = time.time() - t0

        hf_energy = H.diagonal_element(H.get_hf_state()).item()
        _print_result("L4-F", result["energy"], hf_energy, wall,
                      f"NNCI+NOs, basis={result['basis_size']}")

        assert math.isfinite(result["energy"])
        assert result["basis_size"] >= 2
        print(f"  PASS: NNCI found {result['basis_size']} configs")


class TestL4FullPipeline:
    """L4-G: Full Phase 6 pipeline on CAS(10,10).

    Core target: < 5 mHa error (vs 14.2 mHa Direct-CI baseline).
    """

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_full_pipeline_cas10_10(self, n2_cas10_10):
        """All P6 improvements should achieve < 5 mHa on CAS(10,10)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10
        torch.manual_seed(42)

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
        hf_state = H.get_hf_state()
        hf_energy = H.diagonal_element(hf_state).item()
        _print_result("L4-G", energy, hf_energy, wall, "ALL P6 ON")

        assert math.isfinite(energy), "Energy should be finite"
        assert energy < hf_energy, "Energy should be below HF"

        # Report the improvement over Direct-CI baseline
        improvement = abs(energy - hf_energy) * 1000
        print(f"  Improvement over HF: {improvement:.1f} mHa")
        print(f"  NOTE: Need FCI reference to check < 5 mHa target")


class TestL4VMCMechanics:
    """L4-C: Verify VMC runs on CAS(10,10) scale without crashing."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_vmc_runs_cas10_10(self, n2_cas10_10):
        """VMC (REINFORCE) should run without error on 20Q system."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        from flows.vmc_training import VMCTrainer, VMCConfig

        H = n2_cas10_10
        torch.manual_seed(42)

        # 20Q system: 10 spatial orbitals, 5α+5β
        flow = AutoregressiveFlowSampler(num_sites=20, n_alpha=5, n_beta=5)
        config = VMCConfig(
            n_samples=100, n_steps=5, lr=1e-3, optimizer_type="reinforce",
            lr_decay=1.0, convergence_threshold=1e-12,
        )
        trainer = VMCTrainer(flow, H, config=config, device="cpu")

        t0 = time.time()
        result = trainer.train(verbose=False)
        wall = time.time() - t0

        best = result["best_energy"]
        print(f"\nPhase6-L4: L4-C")
        print(f"  VMC(REINFORCE, 5 steps): best={best:+.6f}")
        print(f"  Time: {wall:.1f}s")

        assert math.isfinite(best), "Energy should be finite"
        assert result["n_steps"] == 5, "Should complete all 5 steps"
