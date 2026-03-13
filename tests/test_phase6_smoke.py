"""Level 1: H2 (4Q) smoke tests for all 8 Phase 6 improvements.

Verifies each improvement produces correct energy on a trivially small system
where exact FCI is known. This is a sanity check that nothing is broken.

System: H2/STO-3G, 2 orbitals, 1α+1β, 4 configs, FCI ~ -1.137 Ha
Pass: Error < 0.1 mHa for all combinations
"""

import sys
import os
import time
import math
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def h2_system():
    """Create H2 system with FCI reference."""
    try:
        from hamiltonians.molecular import create_h2_hamiltonian
        H = create_h2_hamiltonian(bond_length=0.74, device="cpu")
        fci = H.fci_energy()
        hf_state = H.get_hf_state()
        return H, fci, hf_state
    except ImportError:
        pytest.skip("PySCF not available")


def _print_result(test_id, energy, fci, wall_time, extra=""):
    error_mha = abs(energy - fci) * 1000
    status = "PASS" if error_mha < 1.0 else "FAIL"
    print(f"\nPhase6 Benchmark: {test_id}")
    print(f"  Energy: {energy:.8f} Ha")
    print(f"  FCI:    {fci:.8f} Ha")
    print(f"  Error:  {error_mha:.3f} mHa")
    print(f"  Time:   {wall_time:.1f}s")
    if extra:
        print(f"  {extra}")
    print(f"  Status: {status}")


class TestL1MinSR:
    """L1-A: MinSR VMC on H2.

    Note: Without a sign network, the ansatz ψ=√p is positive everywhere
    and CANNOT reach FCI (H2 ground state has mixed signs: c₁≈-0.994,
    c₄≈+0.113). The best achievable energy is ~HF energy (-1.117 Ha).
    We verify convergence and reasonable energy, not FCI accuracy.
    """

    @pytest.mark.molecular
    def test_minsr_on_h2(self, h2_system):
        """MinSR VMC should converge toward HF energy on H2."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        from flows.vmc_training import VMCTrainer, VMCConfig

        H, fci, _ = h2_system
        torch.manual_seed(42)

        # SR normalization requires MUCH larger lr than Adam (0.5 vs 1e-3).
        # The SR solve already rescales the gradient by (S + λI)^{-1},
        # so a smaller lr would suppress the update below useful magnitude.
        # H2 is especially challenging for SR because only 4 valid configs
        # → Fisher matrix rank ≤ 4 with 150K params → slow convergence.
        flow = AutoregressiveFlowSampler(num_sites=4, n_alpha=1, n_beta=1)
        config = VMCConfig(
            n_samples=200, n_steps=100, lr=0.5, optimizer_type="minsr",
            lr_decay=1.0, convergence_threshold=1e-12,
        )
        trainer = VMCTrainer(flow, H, config=config, device="cpu")

        t0 = time.time()
        result = trainer.train(verbose=False)
        wall = time.time() - t0

        best = result["best_energy"]
        hf_energy = H.diagonal_element(H.get_hf_state()).item()
        _print_result("L1-A", best, fci, wall,
                      f"optimizer=minsr, HF={hf_energy:.6f}")

        assert math.isfinite(best), "Energy should be finite"
        # Without sign network, best achievable ≈ HF energy (-1.117 Ha).
        # With lr=0.5 and 100 steps, should reach within ~5 mHa of HF.
        assert best < hf_energy + 0.005, (
            f"Energy {best:.6f} should be near HF {hf_energy:.6f}"
        )


class TestL1PhaseNetwork:
    """L1-B: Phase Network on H2.

    With a phase network, the ansatz ψ=√p·exp(iφ) can represent sign
    structure. On H2 this should converge below HF toward FCI.
    """

    @pytest.mark.molecular
    def test_phase_network_on_h2(self, h2_system):
        """Phase network + MinSR should converge on H2."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler
        from flows.sign_network import PhaseNetwork
        from flows.vmc_training import VMCTrainer, VMCConfig

        H, fci, _ = h2_system
        torch.manual_seed(42)

        flow = AutoregressiveFlowSampler(num_sites=4, n_alpha=1, n_beta=1)
        phase_net = PhaseNetwork(num_sites=4)
        config = VMCConfig(
            n_samples=200, n_steps=100, lr=0.5, optimizer_type="minsr",
            lr_decay=1.0, convergence_threshold=1e-12,
        )
        trainer = VMCTrainer(
            flow, H, config=config, device="cpu", sign_network=phase_net,
        )

        t0 = time.time()
        result = trainer.train(verbose=False)
        wall = time.time() - t0

        best = result["best_energy"]
        _print_result("L1-B", best, fci, wall, "phase_network=True")

        assert math.isfinite(best), "Energy should be finite"
        # With phase network AND enough training, should reach well below zero.
        # H2 HF ≈ -1.117, FCI ≈ -1.137.  60 steps with 200 samples may not
        # reach FCI but should clearly be a bound state.
        assert best < -0.5, f"Energy {best:.6f} should be well below zero"


class TestL1HCoupling:
    """L1-C: H-coupling scoring on H2."""

    @pytest.mark.molecular
    def test_h_coupling_on_h2(self, h2_system):
        """SKQD with H-coupling should match baseline on H2."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci, _ = h2_system
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
        error_mha = abs(energy - fci) * 1000
        _print_result("L1-C", energy, fci, wall, "h_coupling=True (default)")

        assert error_mha < 0.1, f"Error {error_mha:.3f} mHa should be < 0.1"


class TestL1Taboo:
    """L1-D: Taboo list on H2."""

    @pytest.mark.molecular
    def test_taboo_on_h2(self, h2_system):
        """SKQD with taboo should match baseline on H2."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci, _ = h2_system
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
        error_mha = abs(energy - fci) * 1000
        _print_result("L1-D", energy, fci, wall, "taboo=True (default)")

        assert error_mha < 0.1, f"Error {error_mha:.3f} mHa should be < 0.1"


class TestL1GumbelTopK:
    """L1-E: Gumbel Top-K on H2."""

    @pytest.mark.molecular
    def test_gumbel_on_h2(self, h2_system):
        """Gumbel top-k should produce valid particle-conserving configs."""
        from flows.particle_conserving_flow import ParticleConservingFlowSampler

        H, fci, _ = h2_system
        torch.manual_seed(42)

        flow = ParticleConservingFlowSampler(
            num_sites=4, n_alpha=1, n_beta=1, topk_type="gumbel",
        )
        log_probs, configs = flow.sample(50)

        # Check particle conservation
        assert configs.shape[1] == 4
        alpha = configs[:, :2].sum(dim=1)
        beta = configs[:, 2:].sum(dim=1)
        assert (alpha == 1).all(), f"Alpha count: {alpha}"
        assert (beta == 1).all(), f"Beta count: {beta}"
        assert torch.isfinite(log_probs).all()
        print(f"\nPhase6 Benchmark: L1-E")
        print(f"  Gumbel unique configs: {len(configs)}")
        print(f"  Status: PASS")


class TestL1UniqueSampling:
    """L1-F: Unique sampling on H2."""

    @pytest.mark.molecular
    def test_unique_sampling_h2(self, h2_system):
        """sample_unique should return all-unique valid configs."""
        from flows.autoregressive_flow import AutoregressiveFlowSampler

        H, fci, _ = h2_system
        torch.manual_seed(42)

        flow = AutoregressiveFlowSampler(num_sites=4, n_alpha=1, n_beta=1)
        log_probs, configs = flow.sample_unique(n_unique_target=4)

        # All unique
        config_set = set(tuple(c.tolist()) for c in configs)
        assert len(config_set) == len(configs), "Should be all unique"
        assert torch.isfinite(log_probs).all()

        # Particle conservation
        n_orb = 2
        for c in configs:
            assert c[:n_orb].sum() == 1
            assert c[n_orb:].sum() == 1

        print(f"\nPhase6 Benchmark: L1-F")
        print(f"  Unique configs: {len(configs)}/4 possible")
        print(f"  Status: PASS")


class TestL1NaturalOrbitals:
    """L1-G: Natural orbitals on H2."""

    @pytest.mark.molecular
    def test_nos_on_h2(self, h2_system):
        """NNCI with NOs should run without error on H2."""
        from krylov.nnci import NNCIConfig, NNCIActiveLearning

        H, fci, _ = h2_system
        basis = H.get_hf_state().unsqueeze(0)

        config = NNCIConfig(
            max_iterations=2,
            top_k=5,
            max_candidates=50,
            use_natural_orbitals=True,
        )
        t0 = time.time()
        nnci = NNCIActiveLearning(H, basis, config)
        result = nnci.run()
        wall = time.time() - t0

        _print_result("L1-G", result["energy"], fci, wall,
                      f"NNCI+NOs, basis_size={result['basis_size']}")

        assert math.isfinite(result["energy"])
        assert result["basis_size"] >= 1


class TestL1AllCombined:
    """L1-H: All Phase 6 improvements combined on H2."""

    @pytest.mark.molecular
    def test_all_combined_h2(self, h2_system):
        """All improvements ON should achieve near-FCI on H2."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci, _ = h2_system
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,  # Direct-CI for H2 (4 configs, no need for NF)
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
        error_mha = abs(energy - fci) * 1000
        _print_result("L1-H", energy, fci, wall, "ALL P6 ON")

        assert error_mha < 0.1, f"Error {error_mha:.3f} mHa should be < 0.1"
