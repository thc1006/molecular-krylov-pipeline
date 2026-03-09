"""30 Qubit Benchmark Tests: N2/cc-pVDZ CAS(10,15) + Benzene/STO-3G CAS(6,15).

Phase B of the 24Q -> 40Q scaling ladder.  At 30Q, CISD coverage drops to
~0.039% (N2) and ~0.85% (benzene), demonstrating NF necessity.

Systems:
- N2/cc-pVDZ CAS(10,15): 15 orbitals, 5α+5β, 9,018,009 configs
  Uses CASCI (no orbital optimization) since CASSCF's internal FCI solver
  cannot handle 9M configs.
- Benzene/STO-3G CAS(6,15): 15 orbitals, 3α+3β, 207,025 configs
  First non-diatomic benchmark; pi-electron correlation.

Usage:
    uv run pytest tests/test_30q_benchmark.py -x -v --tb=short -m slow --override-ini="addopts="
"""

import math
import sys
import time

import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for 30Q tests")

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

# N2 CAS(10,15)
N2_N_ORBITALS = 15
N2_N_ALPHA = 5
N2_N_BETA = 5
N2_NUM_SITES = 30
N2_TOTAL_CONFIGS = 9018009  # C(15,5)^2 = 3003^2

# Benzene CAS(6,15)
BZ_N_ORBITALS = 15
BZ_N_ALPHA = 3
BZ_N_BETA = 3
BZ_NUM_SITES = 30
BZ_TOTAL_CONFIGS = 207025  # C(15,3)^2 = 455^2

# Device strategy: Hamiltonian on CPU (FP64 Numba JIT), NN on GPU (TF32).
NN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_30q_sampler(n_orbitals, n_alpha, n_beta, device=NN_DEVICE):
    """Build a compact AR flow for 30Q tests, on GPU if available."""
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
def n2_cas15():
    """N2/cc-pVDZ CAS(10,15) -- 30Q, 9M configs.
    Uses CASCI (auto-fallback for ncas >= 15). ~45s.
    """
    from hamiltonians.molecular import create_n2_cas_hamiltonian
    return create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 15), device="cpu"
    )


@pytest.fixture(scope="module")
def n2_cas15_hf(n2_cas15):
    return n2_cas15.get_hf_state()


@pytest.fixture(scope="module")
def n2_cas15_hf_energy(n2_cas15, n2_cas15_hf):
    return float(n2_cas15.diagonal_element(n2_cas15_hf))


@pytest.fixture(scope="module")
def benzene_cas15():
    """Benzene/STO-3G CAS(6,15) -- 30Q, 207K configs. ~6s."""
    from hamiltonians.molecular import create_benzene_hamiltonian
    return create_benzene_hamiltonian(basis="sto-3g", cas=(6, 15), device="cpu")


@pytest.fixture(scope="module")
def benzene_hf(benzene_cas15):
    return benzene_cas15.get_hf_state()


@pytest.fixture(scope="module")
def benzene_hf_energy(benzene_cas15, benzene_hf):
    return float(benzene_cas15.diagonal_element(benzene_hf))


# =========================================================================
# N2 CAS(10,15) Tests
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS15Dimensions:
    """Verify N2 CAS(10,15) Hamiltonian dimensions."""

    def test_dimensions(self, n2_cas15):
        H = n2_cas15
        assert H.n_orbitals == N2_N_ORBITALS
        assert H.n_alpha == N2_N_ALPHA
        assert H.n_beta == N2_N_BETA
        assert H.num_sites == N2_NUM_SITES

    def test_hf_valid(self, n2_cas15, n2_cas15_hf):
        hf = n2_cas15_hf
        assert hf.shape == (N2_NUM_SITES,)
        assert ((hf == 0) | (hf == 1)).all()
        assert hf[:N2_N_ORBITALS].sum().item() == N2_N_ALPHA
        assert hf[N2_N_ORBITALS:].sum().item() == N2_N_BETA

    def test_connections(self, n2_cas15, n2_cas15_hf):
        conns, elems = n2_cas15.get_connections(n2_cas15_hf)
        assert len(conns) > 0
        assert len(conns) == len(elems)


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS15CISDCoverage:
    """CISD coverage at CAS(10,15): ~0.039%."""

    def test_cisd_coverage(self):
        from math import comb
        hf, singles, doubles, cisd = _compute_cisd_count(
            N2_N_ORBITALS, N2_N_ALPHA, N2_N_BETA
        )
        coverage = cisd / N2_TOTAL_CONFIGS * 100

        print(f"\nN2 30Q CISD: {cisd:,}/{N2_TOTAL_CONFIGS:,} = {coverage:.3f}%")
        assert hf == 1
        assert singles == 100  # 5*10 * 2
        assert cisd == 3501    # HF + 100 + 3400
        assert comb(N2_N_ORBITALS, N2_N_ALPHA) ** 2 == N2_TOTAL_CONFIGS
        assert coverage < 0.05


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS15ARFlow:
    """AR flow sampling at 30Q for N2."""

    def test_particle_conservation(self, n2_cas15):
        H = n2_cas15
        torch.manual_seed(42)
        flow = _make_30q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
        with torch.no_grad():
            states, lp = flow._sample_autoregressive(500)
            configs = states_to_configs(states, flow.n_orbitals)
        alpha_c = configs[:, :H.n_orbitals].sum(dim=1)
        beta_c = configs[:, H.n_orbitals:].sum(dim=1)
        assert (alpha_c == H.n_alpha).all()
        assert (beta_c == H.n_beta).all()
        assert torch.isfinite(lp).all()
        print(f"\nN2 30Q: 500/500 valid ({H.n_alpha}a + {H.n_beta}b)")

    def test_basis_diversity(self, n2_cas15, n2_cas15_hf):
        H = n2_cas15
        torch.manual_seed(42)
        flow = _make_30q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
        with torch.no_grad():
            states, _ = flow._sample_autoregressive(1000)
            configs = states_to_configs(states, flow.n_orbitals).cpu()
        hf_long = n2_cas15_hf.long()
        ranks = [_count_excitation_rank(c, hf_long) for c in configs.long()]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        beyond_cisd = sum(1 for r in ranks if r >= 3)

        print(f"\nN2 30Q diversity:")
        for rank in sorted(rank_counts):
            print(f"  Rank {rank}: {rank_counts[rank]}")
        print(f"  Beyond CISD: {beyond_cisd}")

        assert len(rank_counts) >= 2
        assert beyond_cisd > 0


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS15VMC:
    """VMC training convergence at 30Q."""

    def test_vmc_converges(self, n2_cas15):
        H = n2_cas15
        torch.manual_seed(42)
        flow = _make_30q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
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
        print(f"\nN2 30Q VMC ({dt:.1f}s): early={early:.4f}, late={late:.4f}, "
              f"improve={(early-late)*1000:.2f} mHa")
        assert late < early


@pytest.mark.slow
@pytest.mark.molecular
class TestN2CAS15Pipeline:
    """Full pipeline on N2 CAS(10,15)."""

    def test_pipeline_below_hf(self, n2_cas15, n2_cas15_hf_energy):
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas15
        e_hf = n2_cas15_hf_energy
        t0 = time.time()

        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device=NN_DEVICE,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run(progress=False)
        dt = time.time() - t0

        best_e = results.get("combined_energy") or results.get("skqd_energy")
        assert best_e is not None
        assert math.isfinite(best_e)
        assert best_e < 0

        improvement = (e_hf - best_e) * 1000
        assert best_e < e_hf + 1e-4
        assert improvement > 1.0

        print(f"\nN2 30Q pipeline ({dt:.1f}s): HF={e_hf:.6f}, "
              f"best={best_e:.6f}, improve={improvement:.2f} mHa")


# =========================================================================
# Benzene CAS(6,15) Tests
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestBenzeneDimensions:
    """Verify benzene CAS(6,15) Hamiltonian dimensions."""

    def test_dimensions(self, benzene_cas15):
        H = benzene_cas15
        assert H.n_orbitals == BZ_N_ORBITALS
        assert H.n_alpha == BZ_N_ALPHA
        assert H.n_beta == BZ_N_BETA
        assert H.num_sites == BZ_NUM_SITES

    def test_hf_valid(self, benzene_cas15, benzene_hf):
        hf = benzene_hf
        assert hf.shape == (BZ_NUM_SITES,)
        assert ((hf == 0) | (hf == 1)).all()
        assert hf[:BZ_N_ORBITALS].sum().item() == BZ_N_ALPHA
        assert hf[BZ_N_ORBITALS:].sum().item() == BZ_N_BETA


@pytest.mark.slow
@pytest.mark.molecular
class TestBenzeneCISDCoverage:
    """CISD coverage for benzene CAS(6,15): ~0.85%."""

    def test_cisd_coverage(self):
        from math import comb
        hf, singles, doubles, cisd = _compute_cisd_count(
            BZ_N_ORBITALS, BZ_N_ALPHA, BZ_N_BETA
        )
        coverage = cisd / BZ_TOTAL_CONFIGS * 100
        print(f"\nBenzene 30Q CISD: {cisd:,}/{BZ_TOTAL_CONFIGS:,} = {coverage:.3f}%")
        assert comb(BZ_N_ORBITALS, BZ_N_ALPHA) ** 2 == BZ_TOTAL_CONFIGS
        assert coverage < 1.0


@pytest.mark.slow
@pytest.mark.molecular
class TestBenzeneARFlow:
    """AR flow sampling at 30Q for benzene."""

    def test_particle_conservation(self, benzene_cas15):
        H = benzene_cas15
        torch.manual_seed(42)
        flow = _make_30q_sampler(H.n_orbitals, H.n_alpha, H.n_beta)
        with torch.no_grad():
            states, lp = flow._sample_autoregressive(500)
            configs = states_to_configs(states, flow.n_orbitals)
        alpha_c = configs[:, :H.n_orbitals].sum(dim=1)
        beta_c = configs[:, H.n_orbitals:].sum(dim=1)
        assert (alpha_c == H.n_alpha).all()
        assert (beta_c == H.n_beta).all()
        assert torch.isfinite(lp).all()
        print(f"\nBenzene 30Q: 500/500 valid ({H.n_alpha}a + {H.n_beta}b)")


@pytest.mark.slow
@pytest.mark.molecular
class TestBenzenePipeline:
    """Full pipeline on benzene CAS(6,15)."""

    def test_pipeline_below_hf(self, benzene_cas15, benzene_hf_energy):
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = benzene_cas15
        e_hf = benzene_hf_energy
        t0 = time.time()

        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device=NN_DEVICE,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        results = pipeline.run(progress=False)
        dt = time.time() - t0

        best_e = results.get("combined_energy") or results.get("skqd_energy")
        assert best_e is not None
        assert math.isfinite(best_e)
        assert best_e < 0
        assert best_e < e_hf + 1e-4

        improvement = (e_hf - best_e) * 1000
        assert improvement > 0.5

        print(f"\nBenzene 30Q pipeline ({dt:.1f}s): HF={e_hf:.6f}, "
              f"best={best_e:.6f}, improve={improvement:.2f} mHa")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
