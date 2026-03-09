"""40 Qubit Benchmark Tests: N2/cc-pVDZ CAS(10,20).

Phase C of the 24Q -> 40Q scaling ladder.  At 40Q, CISD coverage drops to
~0.003%, demonstrating that NF is essential for finding important configs
beyond singles+doubles.

System:
- N2/cc-pVDZ CAS(10,20): 20 orbitals, 5α+5β, 240,374,016 configs
  Uses integrals-only CASCI (FCI infeasible for 240M configs).
  PySCF extracts h1e/h2e without solving the eigenvalue problem.
  No exact FCI reference available — tests use HF energy and variational
  principle as validation criteria.

Key physics at 40Q:
- CISD covers only 0.003% of configuration space
- Ground state has significant contributions from triples/quadruples
- NF must discover high-importance configs in 240M-dimensional space
- Pipeline truncates to ~15K configs via NF + importance ranking

Usage:
    uv run pytest tests/test_40q_benchmark.py -x -v --tb=short -m slow --override-ini="addopts="
"""

import math
import sys
import time

import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for 40Q tests")

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

N_ORBITALS = 20
N_ALPHA = 5
N_BETA = 5
NUM_SITES = 40
TOTAL_CONFIGS = 240374016  # C(20,5)^2 = 15504^2

# Device strategy: Hamiltonian on CPU (FP64 Numba JIT), NN on GPU (TF32).
NN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_40q_sampler(n_orbitals, n_alpha, n_beta, device=NN_DEVICE):
    """Build a compact AR flow for 40Q tests, on GPU if available."""
    config = AutoregressiveConfig(
        n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0,
    )
    flow = AutoregressiveFlowSampler(
        num_sites=2 * n_orbitals,
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
    """Count excitation rank (differences / 2)."""
    return (config != hf_state).sum().item() // 2


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n2_cas20():
    """N2/cc-pVDZ CAS(10,20) -- 40Q, 240M configs.

    Uses integrals-only CASCI (FCI skipped for 240M configs).
    ~10s for integral extraction.
    """
    from hamiltonians.molecular import create_n2_cas_hamiltonian
    return create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 20), device="cpu"
    )


@pytest.fixture(scope="module")
def n2_cas20_hf(n2_cas20):
    return n2_cas20.get_hf_state()


@pytest.fixture(scope="module")
def n2_cas20_hf_energy(n2_cas20, n2_cas20_hf):
    return float(n2_cas20.diagonal_element(n2_cas20_hf))


# =========================================================================
# Test 1: Hamiltonian dimensions and basic properties
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS20Dimensions:
    """Verify N2 CAS(10,20) Hamiltonian dimensions at 40Q."""

    def test_dimensions(self, n2_cas20):
        H = n2_cas20
        assert H.n_orbitals == N_ORBITALS
        assert H.n_alpha == N_ALPHA
        assert H.n_beta == N_BETA
        assert H.num_sites == NUM_SITES

    def test_hf_valid(self, n2_cas20, n2_cas20_hf):
        hf = n2_cas20_hf
        assert hf.shape == (NUM_SITES,)
        assert ((hf == 0) | (hf == 1)).all()
        assert hf[:N_ORBITALS].sum().item() == N_ALPHA
        assert hf[N_ORBITALS:].sum().item() == N_BETA

    def test_hf_energy_negative(self, n2_cas20_hf_energy):
        """HF energy must be negative (bound system)."""
        assert n2_cas20_hf_energy < 0
        assert math.isfinite(n2_cas20_hf_energy)
        print(f"\nN2 40Q HF energy: {n2_cas20_hf_energy:.8f} Ha")

    def test_connections(self, n2_cas20, n2_cas20_hf):
        """HF state must have non-trivial Hamiltonian connections."""
        conns, elems = n2_cas20.get_connections(n2_cas20_hf)
        assert len(conns) > 0
        assert len(conns) == len(elems)
        assert all(math.isfinite(e) for e in elems.tolist())
        print(f"\nN2 40Q HF connections: {len(conns)}")

    def test_integrals_shape(self, n2_cas20):
        """Verify integral dimensions match CAS(10,20)."""
        H = n2_cas20
        h1e = H.h1e.cpu().numpy()
        h2e = H.h2e.cpu().numpy()
        assert h1e.shape == (N_ORBITALS, N_ORBITALS)
        assert h2e.shape == (N_ORBITALS, N_ORBITALS, N_ORBITALS, N_ORBITALS)


# =========================================================================
# Test 2: CISD coverage analysis
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS20CISDCoverage:
    """CISD coverage at CAS(10,20): ~0.003%."""

    def test_cisd_coverage(self):
        from math import comb
        hf, singles, doubles, cisd = _compute_cisd_count(
            N_ORBITALS, N_ALPHA, N_BETA
        )
        coverage = cisd / TOTAL_CONFIGS * 100

        print(f"\nN2 40Q CISD coverage:")
        print(f"  HF: {hf}")
        print(f"  Singles: {singles}")
        print(f"  Doubles: {doubles}")
        print(f"  CISD total: {cisd:,}")
        print(f"  Total configs: {TOTAL_CONFIGS:,}")
        print(f"  Coverage: {coverage:.5f}%")

        # Verify combinatorics
        assert hf == 1
        # Singles: 5*15 * 2 = 150
        assert singles == 150
        # Doubles: C(5,2)*C(15,2) * 2 + (5*15)^2 = 1050*2 + 5625 = 7725
        assert doubles == 7725
        assert cisd == 7876
        assert comb(N_ORBITALS, N_ALPHA) ** 2 == TOTAL_CONFIGS

        # Coverage < 0.01% — CISD is essentially useless at 40Q
        assert coverage < 0.01

    def test_cisd_coverage_decay(self):
        """Verify exponential CISD coverage decay from 20Q to 40Q."""
        from math import comb
        data = [
            (10, 5, 5, "20Q STO-3G"),     # CAS(10,10)
            (12, 5, 5, "24Q"),             # CAS(10,12)
            (15, 5, 5, "30Q"),             # CAS(10,15)
            (20, 5, 5, "40Q"),             # CAS(10,20)
        ]
        coverages = []
        print("\nCISD coverage scaling:")
        for n_orb, na, nb, label in data:
            _, _, _, cisd = _compute_cisd_count(n_orb, na, nb)
            total = comb(n_orb, na) * comb(n_orb, nb)
            cov = cisd / total * 100
            coverages.append(cov)
            print(f"  {label}: {cisd:>8,} / {total:>14,} = {cov:.5f}%")

        # Each step should have lower coverage (monotone decay)
        for i in range(1, len(coverages)):
            assert coverages[i] < coverages[i - 1]


# =========================================================================
# Test 3: AR flow sampling at 40Q
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS20ARFlow:
    """AR flow must produce particle-conserving configs at 40Q."""

    def test_particle_conservation(self, n2_cas20):
        H = n2_cas20
        torch.manual_seed(42)
        flow = _make_40q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
        with torch.no_grad():
            states, lp = flow._sample_autoregressive(500)
            configs = states_to_configs(states, flow.n_orbitals)
        alpha_c = configs[:, :H.n_orbitals].sum(dim=1)
        beta_c = configs[:, H.n_orbitals:].sum(dim=1)
        assert (alpha_c == H.n_alpha).all()
        assert (beta_c == H.n_beta).all()
        assert torch.isfinite(lp).all()
        print(f"\nN2 40Q: 500/500 valid ({H.n_alpha}a + {H.n_beta}b)")

    def test_basis_diversity(self, n2_cas20, n2_cas20_hf):
        """Untrained AR flow should produce configs at multiple excitation ranks."""
        H = n2_cas20
        torch.manual_seed(42)
        flow = _make_40q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
        with torch.no_grad():
            states, _ = flow._sample_autoregressive(1000)
            configs = states_to_configs(states, flow.n_orbitals).cpu()
        hf_long = n2_cas20_hf.long()
        ranks = [_count_excitation_rank(c, hf_long) for c in configs.long()]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        beyond_cisd = sum(1 for r in ranks if r >= 3)

        print(f"\nN2 40Q diversity:")
        for rank in sorted(rank_counts):
            print(f"  Rank {rank}: {rank_counts[rank]}")
        print(f"  Beyond CISD: {beyond_cisd}")

        # At 40Q, untrained flow should naturally produce high-rank excitations
        assert len(rank_counts) >= 2
        assert beyond_cisd > 0, "AR flow must produce beyond-CISD configs at 40Q"


# =========================================================================
# Test 4: VMC training convergence at 40Q
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS20VMC:
    """VMC training at 40Q must converge."""

    def test_vmc_converges(self, n2_cas20):
        H = n2_cas20
        torch.manual_seed(42)
        flow = _make_40q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
        sign_net = SignNetwork(num_sites=H.num_sites).to(NN_DEVICE)

        vmc_cfg = VMCConfig(n_samples=200, n_steps=30, lr=2e-3, min_steps=30)
        trainer = VMCTrainer(
            flow=flow, hamiltonian=H, config=vmc_cfg,
            device=NN_DEVICE, sign_network=sign_net,
        )
        t0 = time.time()
        result = trainer.train(verbose=False)
        dt = time.time() - t0

        energies = result["energies"]
        assert len(energies) == 30
        assert all(math.isfinite(e) for e in energies)

        early = sum(energies[:5]) / 5
        late = sum(energies[-5:]) / 5
        print(f"\nN2 40Q VMC ({dt:.1f}s): early={early:.4f}, late={late:.4f}, "
              f"improve={(early-late)*1000:.2f} mHa")
        assert late < early


# =========================================================================
# Test 5: Full pipeline (Direct-CI + SKQD)
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS20Pipeline:
    """Full pipeline on N2 CAS(10,20) at 40Q."""

    def test_pipeline_below_hf(self, n2_cas20, n2_cas20_hf_energy):
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas20
        e_hf = n2_cas20_hf_energy
        t0 = time.time()

        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device=NN_DEVICE,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run(progress=False)
        dt = time.time() - t0

        best_e = results.get("combined_energy") or results.get("skqd_energy")
        assert best_e is not None, (
            f"Pipeline produced no energy. Keys: {list(results.keys())}"
        )
        assert math.isfinite(best_e), f"Pipeline energy not finite: {best_e}"
        assert best_e < 0, f"Pipeline energy positive: {best_e}"

        # Variational improvement over HF
        assert best_e < e_hf + 1e-4, (
            f"Pipeline {best_e:.8f} not below HF {e_hf:.8f}"
        )

        improvement = (e_hf - best_e) * 1000
        assert improvement > 0.5, (
            f"Improvement only {improvement:.4f} mHa on 40Q — expected "
            f"significant correlation recovery"
        )

        print(f"\nN2 40Q pipeline ({dt:.1f}s): HF={e_hf:.6f}, "
              f"best={best_e:.6f}, improve={improvement:.2f} mHa")


# =========================================================================
# Test 6: Scaling comparison across the ladder
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestScalingLadder:
    """Cross-scale comparison of pipeline behavior."""

    def test_connection_count_at_40q(self, n2_cas20, n2_cas20_hf):
        """HF connections should grow with active space."""
        conns, _ = n2_cas20.get_connections(n2_cas20_hf)
        n_conns = len(conns)

        # For N2 with 5a+5b, singles = 5*15*2 = 150, doubles depend on
        # pair excitations — total connections should be in hundreds
        print(f"\nN2 40Q HF connections: {n_conns}")
        assert n_conns >= 100, (
            f"Expected significant connections at 40Q, got {n_conns}"
        )

    def test_40q_config_space_intractable(self):
        """Verify 40Q config space is too large for exact methods."""
        from math import comb
        n_configs = comb(N_ORBITALS, N_ALPHA) * comb(N_ORBITALS, N_BETA)
        assert n_configs == TOTAL_CONFIGS

        # Dense H would be 240M × 240M × 8 bytes ≈ 461 PB
        dense_bytes = n_configs * n_configs * 8
        dense_pb = dense_bytes / (1024**5)
        print(f"\nN2 40Q intractability:")
        print(f"  Configs: {n_configs:,}")
        print(f"  Dense H: {dense_pb:.0f} PB")
        assert dense_pb > 100, "40Q should be intractable for dense methods"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
