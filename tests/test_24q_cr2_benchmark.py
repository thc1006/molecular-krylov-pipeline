"""Cr2/STO-3G CAS(12,12) = 24 Qubit Multi-Reference Benchmark Tests.

Cr2 (chromium dimer) is one of the most challenging multi-reference systems
in quantum chemistry, with a formal sextuple bond and strong static correlation
from the near-degenerate 3d orbitals. CAS(12,12) captures the 3d+4s active
space with 12 active orbitals, 12 active electrons (6 alpha + 6 beta),
24 spin-orbitals, and C(12,6)^2 = 853,776 configurations.

Key physics:
- CASSCF without spin constraint converges to the WRONG state (septet S=3)
- ``fix_spin_(ss=0)`` is required to target the singlet ground state
- The singlet has LOWER energy than the septet (-2064.399 vs -2064.382 Ha)
- This is a strong test of the pipeline's multi-reference capability

These tests validate:
1. Factory function with fix_spin_ singlet constraint
2. Hamiltonian dimensions and basic properties
3. CISD coverage analysis (0.20% at CAS(12,12))
4. AR flow sampling with particle conservation at 24Q
5. Full pipeline end-to-end (Direct-CI + SKQD)

Usage:
    uv run pytest tests/test_24q_cr2_benchmark.py -x -v --tb=short -m slow
"""

import math
import sys
import time

import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for Cr2 CAS(12,12) tests")

from flows.autoregressive_flow import (
    AutoregressiveConfig,
    AutoregressiveFlowSampler,
    states_to_configs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ORBITALS = 12
N_ALPHA = 6
N_BETA = 6
NUM_SITES = 24
TOTAL_CONFIGS = 853776  # C(12,6)^2 = 924^2

# Device strategy: Hamiltonian on CPU (FP64 Numba JIT), NN on GPU (TF32).
NN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cr2_sampler(n_orbitals, n_alpha, n_beta, device=NN_DEVICE):
    """Build a compact AR flow for Cr2 CAS(12,12) tests, on GPU if available."""
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


def _compute_cisd_count(n_orb, n_alpha, n_beta):
    """Compute HF + singles + doubles count."""
    from math import comb

    hf = 1
    alpha_singles = n_alpha * (n_orb - n_alpha)
    beta_singles = n_beta * (n_orb - n_beta)
    singles = alpha_singles + beta_singles

    aa = comb(n_alpha, 2) * comb(n_orb - n_alpha, 2)
    bb = comb(n_beta, 2) * comb(n_orb - n_beta, 2)
    ab = alpha_singles * beta_singles
    doubles = aa + bb + ab

    return hf, singles, doubles, hf + singles + doubles


def _count_excitation_rank(config, hf_state):
    """Count excitation rank (number of orbital differences / 2)."""
    diff = (config != hf_state).sum().item()
    return diff // 2


# ---------------------------------------------------------------------------
# Module-scoped fixture (Cr2 CASSCF with fix_spin_ is expensive: ~150s)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cr2_cas12():
    """Cr2/STO-3G CAS(12,12) -- 24Q, 854K configs.

    Uses fix_spin_(ss=0) to target singlet ground state.
    Module-scoped because CASSCF is very expensive (~150s).
    """
    from hamiltonians.molecular import create_cr2_hamiltonian

    H = create_cr2_hamiltonian(
        bond_length=1.68, basis="sto-3g", cas=(12, 12), device="cpu"
    )
    return H


@pytest.fixture(scope="module")
def cr2_hf_state(cr2_cas12):
    """Cached HF state for Cr2 CAS(12,12)."""
    return cr2_cas12.get_hf_state()


@pytest.fixture(scope="module")
def cr2_hf_energy(cr2_cas12, cr2_hf_state):
    """HF energy (diagonal element) for Cr2 CAS(12,12)."""
    return float(cr2_cas12.diagonal_element(cr2_hf_state))


# =========================================================================
# Test 1: Factory function and Hamiltonian dimensions
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCr2HamiltonianDimensions:
    """Verify Cr2 CAS(12,12) Hamiltonian has correct dimensions."""

    def test_hamiltonian_dimensions(self, cr2_cas12):
        """CAS(12,12) should have 12 orbitals, 6+6 electrons, 24 spin-orbitals."""
        H = cr2_cas12
        assert H.n_orbitals == N_ORBITALS
        assert H.n_alpha == N_ALPHA
        assert H.n_beta == N_BETA
        assert H.num_sites == NUM_SITES

    def test_hf_state_valid(self, cr2_cas12, cr2_hf_state):
        """HF state must be binary with correct electron count."""
        hf = cr2_hf_state
        assert hf.shape == (NUM_SITES,)
        assert ((hf == 0) | (hf == 1)).all()
        assert hf[:N_ORBITALS].sum().item() == N_ALPHA
        assert hf[N_ORBITALS:].sum().item() == N_BETA

    def test_hf_energy_negative(self, cr2_hf_energy):
        """HF energy must be negative (bound system)."""
        assert cr2_hf_energy < 0
        assert math.isfinite(cr2_hf_energy)

    def test_get_connections_from_hf(self, cr2_cas12, cr2_hf_state):
        """HF state must have non-trivial Hamiltonian connections."""
        conns, elems = cr2_cas12.get_connections(cr2_hf_state)
        assert len(conns) > 0, "HF state has no connections"
        assert len(conns) == len(elems)
        assert all(math.isfinite(e) for e in elems.tolist())


# =========================================================================
# Test 2: CISD coverage analysis
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCr2CISDCoverage:
    """Document CISD insufficiency at CAS(12,12), motivating NF."""

    def test_cisd_coverage_analysis(self):
        """Compute and verify CISD coverage for Cr2 CAS(12,12).

        For CAS(12,12) with 12 orbitals and 6 alpha + 6 beta electrons:
        - Singles: 6*6 * 2 = 72
        - Doubles: C(6,2)*C(6,2) * 2 + 36*36 = 225*2 + 1296 = 1746
        - CISD total: 1 + 72 + 1746 = 1819
        - Total configs: C(12,6)^2 = 853,776
        - Coverage: ~0.21%
        """
        from math import comb

        hf, singles, doubles, cisd_total = _compute_cisd_count(
            N_ORBITALS, N_ALPHA, N_BETA
        )

        coverage = cisd_total / TOTAL_CONFIGS * 100

        print(f"\nCr2 24Q CISD coverage analysis:")
        print(f"  Active space: CAS(12,12) = {N_ORBITALS} orbitals, "
              f"{N_ALPHA}a + {N_BETA}b electrons")
        print(f"  Total configs: {TOTAL_CONFIGS:,}")
        print(f"  HF: {hf}")
        print(f"  Singles: {singles}")
        print(f"  Doubles: {doubles}")
        print(f"  CISD total: {cisd_total:,}")
        print(f"  CISD coverage: {coverage:.3f}%")

        # Verify combinatorics
        assert hf == 1
        assert singles == 72
        assert doubles == 1746
        assert cisd_total == 1819

        # Total config space
        assert comb(N_ORBITALS, N_ALPHA) ** 2 == TOTAL_CONFIGS

        # Coverage should be ~0.21%
        assert coverage < 0.5
        assert coverage > 0.1


# =========================================================================
# Test 3: AR flow sampling at 24Q
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCr2ARFlowSamples:
    """AR flow must produce particle-conserving configs at Cr2 24Q."""

    def test_ar_flow_samples_valid(self, cr2_cas12):
        """Sample 500 configs and verify particle conservation (6a + 6b)."""
        H = cr2_cas12
        torch.manual_seed(42)

        flow = _make_cr2_sampler(H.n_orbitals, H.n_alpha, H.n_beta)

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(500)
            configs = states_to_configs(states, flow.n_orbitals)

        n_orb = H.n_orbitals
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == H.n_alpha).all(), (
            f"Alpha electron violation: expected {H.n_alpha}, "
            f"got {alpha_counts.unique().tolist()}"
        )
        assert (beta_counts == H.n_beta).all(), (
            f"Beta electron violation: expected {H.n_beta}, "
            f"got {beta_counts.unique().tolist()}"
        )
        assert ((configs == 0) | (configs == 1)).all()
        assert torch.isfinite(log_probs).all()
        assert (log_probs <= 0).all()

        print(f"\nCr2 24Q AR flow: 500/500 samples valid ({H.n_alpha}a + {H.n_beta}b)")

    def test_ar_flow_basis_diversity(self, cr2_cas12, cr2_hf_state):
        """Untrained AR flow should produce configs at multiple excitation ranks."""
        H = cr2_cas12
        hf_state = cr2_hf_state
        torch.manual_seed(42)

        flow = _make_cr2_sampler(H.n_orbitals, H.n_alpha, H.n_beta)

        with torch.no_grad():
            states, _ = flow._sample_autoregressive(1000)
            configs = states_to_configs(states, flow.n_orbitals).cpu()

        hf_long = hf_state.long()
        ranks = [_count_excitation_rank(c, hf_long) for c in configs.long()]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        n_unique = len(set(tuple(c.tolist()) for c in configs))
        beyond_cisd = sum(1 for r in ranks if r >= 3)

        print(f"\nCr2 24Q basis diversity:")
        print(f"  Unique: {n_unique}/1000")
        for rank in sorted(rank_counts.keys()):
            pct = rank_counts[rank] / len(configs) * 100
            print(f"    Rank {rank}: {rank_counts[rank]} ({pct:.1f}%)")
        print(f"  Beyond CISD (rank >= 3): {beyond_cisd}")

        assert n_unique >= 2, "AR flow completely degenerate"
        assert len(rank_counts) >= 2, "Only one excitation rank produced"


# =========================================================================
# Test 4: Full pipeline (Direct-CI + SKQD)
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCr2FullPipeline:
    """Direct-CI + SKQD on Cr2 CAS(12,12)."""

    def test_pipeline_energy_below_hf(self, cr2_cas12, cr2_hf_energy):
        """Pipeline must produce energy below HF (variational principle)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = cr2_cas12
        e_hf = cr2_hf_energy

        t_start = time.time()

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device=NN_DEVICE,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run(progress=False)

        t_elapsed = time.time() - t_start

        best_e = results.get("combined_energy") or results.get("skqd_energy")
        if best_e is None:
            best_e = results.get("vmc_energy")
        assert best_e is not None, (
            f"Pipeline produced no energy. Keys: {list(results.keys())}"
        )
        assert math.isfinite(best_e), f"Pipeline energy not finite: {best_e}"
        assert best_e < 0, f"Pipeline energy positive: {best_e}"

        # Variational improvement over HF
        assert best_e < e_hf + 1e-4, (
            f"Pipeline {best_e:.8f} not below HF {e_hf:.8f}"
        )

        improvement_mha = (e_hf - best_e) * 1000
        assert improvement_mha > 1.0, (
            f"Improvement only {improvement_mha:.4f} mHa, expected significant "
            f"correlation energy recovery on CAS(12,12)"
        )

        print(f"\nCr2 24Q Direct-CI+SKQD ({t_elapsed:.1f}s):")
        print(f"  HF energy: {e_hf:.8f} Ha")
        print(f"  Pipeline energy: {best_e:.8f} Ha")
        print(f"  Improvement: {improvement_mha:.4f} mHa")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
