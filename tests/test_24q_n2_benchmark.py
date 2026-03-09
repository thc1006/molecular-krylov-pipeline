"""N2/cc-pVDZ CAS(10,12) = 24 Qubit Benchmark Tests.

First step in the 24Q -> 40Q scaling ladder.  CAS(10,12) has 12 active
orbitals, 10 electrons (5 alpha + 5 beta), 24 spin-orbitals, and a
configuration space of C(12,5)^2 = 627,264 determinants.

This is too large for explicit FCI enumeration via dense matrix construction,
but our SKQD pipeline handles it via NF-guided truncation to ~15K configs.
The CISD space covers only 1716 configs (0.27%), motivating the need for
autoregressive NF to discover important higher excitations.

These tests validate:
1. Hamiltonian dimensions and construction correctness
2. Autoregressive flow sampling with particle conservation at 24Q
3. AR flow log_prob consistency (sampling vs teacher forcing)
4. VMC training convergence with sign network at 24Q scale
5. AR flow basis diversity beyond Direct-CI (CISD) set
6. Full AR+VMC+Sign+SKQD pipeline end-to-end
7. CISD coverage analysis documenting the NF motivation
8. Variational improvement over HF energy

Usage:
    uv run pytest tests/test_24q_n2_benchmark.py -x -v --tb=short -m slow
"""

import math
import sys
import time

import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for CAS(10,12) tests")

from flows.autoregressive_flow import (
    AutoregressiveConfig,
    AutoregressiveFlowSampler,
    states_to_configs,
)
from flows.sign_network import SignNetwork
from flows.vmc_training import VMCConfig, VMCTrainer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEMICAL_ACCURACY_HA = 1.594e-3  # 1.0 kcal/mol

# CAS(10,12) combinatorics: C(12,5)^2 = 792^2 = 627,264
N_ORBITALS = 12
N_ALPHA = 5
N_BETA = 5
NUM_SITES = 24
TOTAL_CONFIGS = 627264

# Device strategy: Hamiltonian on CPU (FP64 Numba JIT), NN on GPU (TF32).
NN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cas12_sampler(n_orbitals, n_alpha, n_beta, device=NN_DEVICE):
    """Build a compact AR flow for CAS(10,12) tests, on GPU if available."""
    num_sites = 2 * n_orbitals
    config = AutoregressiveConfig(
        n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0,
    )
    flow = AutoregressiveFlowSampler(
        num_sites=num_sites,
        n_alpha=n_alpha,
        n_beta=n_beta,
        transformer_config=config,
    )
    return flow.to(device)


def _count_excitation_rank(config, hf_state):
    """Count excitation rank (number of orbital differences from HF).

    Each excitation flips one occupied -> unoccupied and one unoccupied -> occupied,
    so excitation rank = total_differences / 2.
    """
    diff = (config != hf_state).sum().item()
    return diff // 2


def _compute_cisd_count(n_orb, n_alpha, n_beta):
    """Compute the number of HF + singles + doubles configurations.

    Returns (hf, singles, doubles, total_cisd).
    """
    from math import comb

    hf = 1

    # Singles: one alpha OR one beta excitation
    alpha_singles = n_alpha * (n_orb - n_alpha)
    beta_singles = n_beta * (n_orb - n_beta)
    singles = alpha_singles + beta_singles

    # Doubles: alpha-alpha, beta-beta, alpha-beta
    aa = comb(n_alpha, 2) * comb(n_orb - n_alpha, 2)
    bb = comb(n_beta, 2) * comb(n_orb - n_beta, 2)
    ab = alpha_singles * beta_singles
    doubles = aa + bb + ab

    return hf, singles, doubles, hf + singles + doubles


# ---------------------------------------------------------------------------
# Module-scoped fixture (CASSCF takes ~30-60s)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n2_cas12():
    """N2/cc-pVDZ CAS(10,12) -- 24Q, 627K configs.

    Module-scoped because CASSCF orbital optimization is expensive.
    Reused across all tests in this module.
    """
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 12), device="cpu"
    )
    return H


@pytest.fixture(scope="module")
def cas12_hf_state(n2_cas12):
    """Cached HF state for CAS(10,12)."""
    return n2_cas12.get_hf_state()


@pytest.fixture(scope="module")
def cas12_hf_energy(n2_cas12, cas12_hf_state):
    """HF energy (diagonal element) for CAS(10,12).

    This is cheap to compute and serves as the upper bound for
    variational energy tests.
    """
    e_hf = n2_cas12.diagonal_element(cas12_hf_state)
    return float(e_hf)


@pytest.fixture(scope="module")
def cas12_fci_energy(n2_cas12):
    """FCI energy for CAS(10,12), if computable.

    CAS(10,12) has 627,264 configs. The sparse eigensolver path in
    fci_energy() should handle this, but it may take several minutes.
    Returns None if FCI computation fails or times out.
    """
    try:
        fci_e = n2_cas12.fci_energy()
        return fci_e
    except Exception as e:
        print(f"\n[WARNING] FCI computation failed for CAS(10,12): {e}")
        return None


# =========================================================================
# Test 1: Hamiltonian dimensions
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12HamiltonianDimensions:
    """Verify CAS(10,12) Hamiltonian has correct dimensions."""

    def test_hamiltonian_dimensions(self, n2_cas12):
        """CAS(10,12) should have 12 orbitals, 5+5 electrons, 24 spin-orbitals."""
        H = n2_cas12
        assert H.n_orbitals == N_ORBITALS, (
            f"Expected {N_ORBITALS} orbitals, got {H.n_orbitals}"
        )
        assert H.n_alpha == N_ALPHA, (
            f"Expected {N_ALPHA} alpha electrons, got {H.n_alpha}"
        )
        assert H.n_beta == N_BETA, (
            f"Expected {N_BETA} beta electrons, got {H.n_beta}"
        )
        assert H.num_sites == NUM_SITES, (
            f"Expected {NUM_SITES} spin-orbitals, got {H.num_sites}"
        )


# =========================================================================
# Test 2: AR flow samples valid
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12ARFlowSamples:
    """AR flow must produce particle-conserving binary configs at 24Q."""

    def test_ar_flow_samples_valid(self, n2_cas12):
        """Sample 500 configs from AR flow and verify particle conservation.

        Each configuration must have exactly 5 alpha electrons (sites 0-11)
        and 5 beta electrons (sites 12-23), all values binary (0 or 1).
        """
        H = n2_cas12
        torch.manual_seed(42)

        flow = _make_cas12_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(500)
            configs = states_to_configs(states, flow.n_orbitals)

        n_orb = H.n_orbitals

        # Check particle conservation
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == H.n_alpha).all(), (
            f"Alpha electron violation: expected {H.n_alpha}, "
            f"got counts {alpha_counts.unique().tolist()}"
        )
        assert (beta_counts == H.n_beta).all(), (
            f"Beta electron violation: expected {H.n_beta}, "
            f"got counts {beta_counts.unique().tolist()}"
        )

        # All values must be binary
        assert ((configs == 0) | (configs == 1)).all(), (
            "Non-binary values in sampled configs"
        )

        # Log probs should be finite and non-positive
        assert torch.isfinite(log_probs).all(), "Non-finite log probs from AR flow"
        assert (log_probs <= 0).all(), "log_prob > 0 detected (impossible)"

        print(f"\n24Q AR flow: 500/500 samples valid ({H.n_alpha}a + {H.n_beta}b)")


# =========================================================================
# Test 3: AR flow log_prob finite and consistent
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12ARFlowLogProb:
    """Log probabilities from AR flow must be finite and consistent."""

    def test_ar_flow_log_prob_finite(self, n2_cas12):
        """Sample 200 configs, compute log_prob via teacher forcing.

        All log_probs must be finite and negative.  Teacher-forced values
        must agree with the sampling-time log_probs.
        """
        H = n2_cas12
        torch.manual_seed(42)

        flow = _make_cas12_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )

        with torch.no_grad():
            states, sample_log_probs = flow._sample_autoregressive(200)
            configs = states_to_configs(states, flow.n_orbitals)
            teacher_log_probs = flow.log_prob(configs.float())

        # All must be finite and negative
        assert torch.isfinite(sample_log_probs).all(), (
            "Non-finite sampling log probs"
        )
        assert torch.isfinite(teacher_log_probs).all(), (
            "Non-finite teacher-forced log probs"
        )
        assert (sample_log_probs <= 0).all(), "Sampling log_prob > 0"
        assert (teacher_log_probs <= 0).all(), "Teacher log_prob > 0"

        # Sampling and teacher-forced should agree
        assert torch.allclose(sample_log_probs, teacher_log_probs, atol=1e-4), (
            f"Log prob mismatch: max diff = "
            f"{(sample_log_probs - teacher_log_probs).abs().max().item():.6f}"
        )

        print(f"\n24Q log_prob: 200/200 finite, max sample-teacher diff = "
              f"{(sample_log_probs - teacher_log_probs).abs().max().item():.2e}")


# =========================================================================
# Test 4: VMC training converges
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12VMCConvergence:
    """VMC training on CAS(10,12) with AR flow + sign network.

    Tests that VMC energy shows a decreasing trend over 30 training steps,
    energies are all finite, and sign magnitudes are non-trivial.
    """

    def test_vmc_training_converges(self, n2_cas12):
        """VMC energy should decrease from first to last 5-step window.

        Uses 200 samples per step and 30 steps to keep runtime under ~5 min.
        """
        H = n2_cas12
        torch.manual_seed(42)

        flow = _make_cas12_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites).to(NN_DEVICE)

        vmc_cfg = VMCConfig(
            n_samples=200,
            n_steps=30,
            lr=2e-3,
            min_steps=30,
        )
        trainer = VMCTrainer(
            flow=flow,
            hamiltonian=H,
            config=vmc_cfg,
            device=NN_DEVICE,
            sign_network=sign_net,
        )

        t_start = time.time()
        result = trainer.train(verbose=False)
        t_elapsed = time.time() - t_start

        energies = result["energies"]
        assert len(energies) == 30, f"Expected 30 steps, got {len(energies)}"

        # All energies must be finite
        for i, e in enumerate(energies):
            assert math.isfinite(e), f"Energy at step {i} is not finite: {e}"

        # Energy should decrease: early 5-step average > late 5-step average
        n_window = 5
        early_avg = sum(energies[:n_window]) / n_window
        late_avg = sum(energies[-n_window:]) / n_window

        print(f"\n24Q VMC convergence ({t_elapsed:.1f}s):")
        print(f"  Early avg (steps 0-{n_window}): {early_avg:.6f} Ha")
        print(f"  Late avg (steps {30-n_window}-30): {late_avg:.6f} Ha")
        print(f"  Improvement: {(early_avg - late_avg) * 1000:.4f} mHa")

        assert late_avg < early_avg, (
            f"VMC energy did not decrease: early={early_avg:.6f}, late={late_avg:.6f}. "
            f"The optimizer should lower the variational energy over 30 steps."
        )

        # Sign magnitudes should be non-trivial after training
        with torch.no_grad():
            states, _ = flow._sample_autoregressive(200)
            configs = states_to_configs(states, flow.n_orbitals)
            signs = sign_net(configs.float())

        mean_magnitude = signs.abs().mean().item()
        print(f"  Sign mean |s|: {mean_magnitude:.4f}")

        assert mean_magnitude > 0.02, (
            f"Sign magnitudes too small after VMC training: mean |s| = {mean_magnitude:.4f}. "
            f"The sign network should develop non-trivial sign structure on CAS(10,12)."
        )


# =========================================================================
# Test 5: Direct-CI vs AR NF basis diversity
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12BasisDiversity:
    """AR flow should produce configs beyond the CISD (Direct-CI) set."""

    def test_directci_vs_ar_nf_basis(self, n2_cas12, cas12_hf_state):
        """Direct-CI covers ~1716 configs; AR flow should explore beyond.

        An untrained (random) AR flow has high entropy and naturally
        produces configs at multiple excitation ranks, including
        triples and quadruples not in the CISD set.
        """
        H = n2_cas12
        hf_state = cas12_hf_state
        torch.manual_seed(42)

        # Build CISD set for comparison
        _, _, _, cisd_total = _compute_cisd_count(H.n_orbitals, H.n_alpha, H.n_beta)
        assert cisd_total == 1716, (
            f"CISD count mismatch: expected 1716, got {cisd_total}"
        )

        # Use an untrained flow (maximum entropy) to test diversity
        flow = _make_cas12_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )

        with torch.no_grad():
            states, _ = flow._sample_autoregressive(1000)
            ar_configs = states_to_configs(states, flow.n_orbitals).cpu()

        # Classify excitation ranks
        hf_long = hf_state.long()
        ar_ranks = [
            _count_excitation_rank(c, hf_long) for c in ar_configs.long()
        ]

        rank_counts = {}
        for r in ar_ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        # Count unique configs
        ar_set = set()
        for c in ar_configs:
            ar_set.add(tuple(c.tolist()))
        n_unique = len(ar_set)

        # Count configs beyond CISD (excitation rank >= 3)
        beyond_cisd = sum(1 for r in ar_ranks if r >= 3)

        print(f"\n24Q basis diversity (untrained AR flow):")
        print(f"  Sampled: {len(ar_configs)}, Unique: {n_unique}")
        print(f"  Excitation rank distribution:")
        for rank in sorted(rank_counts.keys()):
            pct = rank_counts[rank] / len(ar_configs) * 100
            print(f"    Rank {rank}: {rank_counts[rank]} ({pct:.1f}%)")
        print(f"  Beyond CISD (rank >= 3): {beyond_cisd} "
              f"({beyond_cisd / len(ar_configs) * 100:.1f}%)")

        # AR flow should produce at least some unique configs (not degenerate)
        assert n_unique >= 2, (
            f"AR flow produced only {n_unique} unique config from 1000 samples. "
            f"Flow sampling is completely degenerate."
        )

        # AR flow should produce configs at multiple excitation ranks
        n_ranks = len(rank_counts)
        assert n_ranks >= 2, (
            f"AR flow produced configs at only {n_ranks} excitation rank(s). "
            f"Expected diversity across multiple ranks."
        )

        # An untrained flow on a 24-qubit system should produce SOME beyond-CISD
        # configs (triples/quadruples).  With 5 electrons in 12 orbitals, the
        # random walk through the autoregressive decisions naturally explores
        # higher excitations.
        assert beyond_cisd > 0, (
            f"AR flow produced 0 configs beyond CISD (rank >= 3) from 1000 samples. "
            f"Expected at least some triples/quadruples on a 24-qubit system."
        )


# =========================================================================
# Test 6: Full pipeline AR+VMC+Sign+SKQD
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12FullPipeline:
    """End-to-end AR+VMC+Sign+SKQD pipeline on CAS(10,12).

    The full pipeline must complete without OOM, produce a finite negative
    energy, and the pipeline results must contain the expected keys.
    """

    def test_full_pipeline_ar_vmc_sign_skqd(self, n2_cas12, cas12_fci_energy):
        """Full AR+VMC+Sign+SKQD pipeline on CAS(10,12).

        Uses skip_nf_training=True to bypass the old non-autoregressive NF
        and rely on Direct-CI + AR VMC expansion + SKQD.  VMC steps and
        samples are kept low for test speed.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas12
        fci_e = cas12_fci_energy  # May be None if FCI failed

        t_start = time.time()

        config = PipelineConfig(
            subspace_mode="skqd",
            use_autoregressive_flow=True,
            use_vmc_training=True,
            use_sign_network=True,
            vmc_n_steps=30,
            vmc_n_samples=200,
            skip_nf_training=True,
            device=NN_DEVICE,
        )
        pipeline = FlowGuidedKrylovPipeline(
            H,
            config=config,
            exact_energy=fci_e,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)

        t_elapsed = time.time() - t_start

        # Extract best energy
        best_e = results.get("combined_energy") or results.get("skqd_energy")
        if best_e is None:
            best_e = results.get("vmc_energy")
        assert best_e is not None, (
            f"Pipeline produced no energy result. Keys: {list(results.keys())}"
        )

        # Energy must be finite and negative (physical)
        assert math.isfinite(best_e), f"Pipeline energy not finite: {best_e}"
        assert best_e < 0, f"Pipeline energy is positive (unphysical): {best_e}"

        print(f"\n24Q AR+VMC+Sign+SKQD pipeline results ({t_elapsed:.1f}s):")
        print(f"  Pipeline energy: {best_e:.8f} Ha")
        if fci_e is not None:
            error_mha = abs(best_e - fci_e) * 1000
            print(f"  FCI reference: {fci_e:.8f} Ha")
            print(f"  Error: {error_mha:.4f} mHa")

        # combined_energy should exist (SKQD produces this)
        assert "skqd_energy" in results or "combined_energy" in results, (
            f"Neither 'skqd_energy' nor 'combined_energy' in results. "
            f"Keys: {list(results.keys())}"
        )


# =========================================================================
# Test 7: CISD coverage analysis
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12CISDCoverage:
    """Document the CISD insufficiency at 24Q, motivating NF."""

    def test_cisd_coverage_analysis(self):
        """Compute and verify CISD coverage for CAS(10,12).

        CISD generates HF + singles + doubles.  For CAS(10,12) with
        12 orbitals and 5 alpha + 5 beta electrons:
        - Singles: 5*7 * 2 = 70
        - Doubles: C(5,2)*C(7,2) * 2 + 35*35 = 210*2 + 1225 = 1645
        - CISD total: 1 + 70 + 1645 = 1716
        - Total configs: 627,264
        - Coverage: ~0.27%

        This documents why NF is essential at 24Q: Direct-CI (CISD)
        covers less than 0.3% of the configuration space.
        """
        hf, singles, doubles, cisd_total = _compute_cisd_count(
            N_ORBITALS, N_ALPHA, N_BETA
        )

        coverage = cisd_total / TOTAL_CONFIGS * 100

        print(f"\n24Q CISD coverage analysis:")
        print(f"  Active space: CAS(10,12) = {N_ORBITALS} orbitals, "
              f"{N_ALPHA}a + {N_BETA}b electrons")
        print(f"  Total configs: {TOTAL_CONFIGS:,}")
        print(f"  HF: {hf}")
        print(f"  Singles: {singles}")
        print(f"  Doubles: {doubles}")
        print(f"  CISD total: {cisd_total:,}")
        print(f"  CISD coverage: {coverage:.3f}%")

        # Verify combinatorics
        assert hf == 1
        assert singles == 70
        assert doubles == 1645
        assert cisd_total == 1716

        # Total config space
        from math import comb
        assert comb(N_ORBITALS, N_ALPHA) ** 2 == TOTAL_CONFIGS

        # Coverage should be ~0.27%
        assert coverage < 0.5, (
            f"CISD coverage {coverage:.3f}% higher than expected for 24Q system"
        )
        assert coverage > 0.1, (
            f"CISD coverage {coverage:.3f}% lower than expected"
        )


# =========================================================================
# Test 8: Pipeline energy below HF (variational improvement)
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS12VariationalImprovement:
    """Pipeline energy must be below HF energy (variational principle)."""

    def test_pipeline_energy_below_threshold(
        self, n2_cas12, cas12_hf_energy, cas12_fci_energy
    ):
        """Direct-CI + SKQD should produce energy below HF.

        Even without VMC/sign, the pipeline (Direct-CI generating
        HF+singles+doubles followed by SKQD Krylov expansion) should
        produce a variational improvement over the bare HF energy.
        This holds regardless of whether exact FCI is available.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas12
        e_hf = cas12_hf_energy
        fci_e = cas12_fci_energy

        t_start = time.time()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device=NN_DEVICE,
        )
        pipeline = FlowGuidedKrylovPipeline(
            H,
            config=config,
            exact_energy=fci_e,
        )
        results = pipeline.run(progress=False)

        t_elapsed = time.time() - t_start

        # Extract best energy
        best_e = results.get("combined_energy") or results.get("skqd_energy")
        if best_e is None:
            best_e = results.get("vmc_energy")
        assert best_e is not None, (
            f"Pipeline produced no energy result. Keys: {list(results.keys())}"
        )

        assert math.isfinite(best_e), f"Pipeline energy not finite: {best_e}"

        # Variational improvement: pipeline energy must be below HF
        # Allow a small tolerance (0.1 mHa) for numerical noise in the
        # diagonal element computation.
        assert best_e < e_hf + 1e-4, (
            f"Pipeline energy {best_e:.8f} Ha is not below HF energy "
            f"{e_hf:.8f} Ha. No variational improvement detected."
        )

        # The improvement should be substantial: correlation energy is typically
        # tens of mHa for N2/cc-pVDZ.  SKQD with CISD basis should capture
        # at least some correlation energy.
        improvement_mha = (e_hf - best_e) * 1000
        assert improvement_mha > 1.0, (
            f"Pipeline improvement over HF is only {improvement_mha:.4f} mHa. "
            f"Expected significant correlation energy recovery on CAS(10,12)."
        )

        print(f"\n24Q Direct-CI+SKQD variational improvement ({t_elapsed:.1f}s):")
        print(f"  HF energy: {e_hf:.8f} Ha")
        print(f"  Pipeline energy: {best_e:.8f} Ha")
        print(f"  Improvement: {improvement_mha:.4f} mHa")

        if fci_e is not None:
            error_mha = abs(best_e - fci_e) * 1000
            correlation_e = abs(fci_e - e_hf) * 1000
            recovery_pct = improvement_mha / correlation_e * 100 if correlation_e > 0 else 0
            print(f"  FCI reference: {fci_e:.8f} Ha")
            print(f"  Error vs FCI: {error_mha:.4f} mHa")
            print(f"  Correlation energy: {correlation_e:.4f} mHa")
            print(f"  Recovery: {recovery_pct:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
