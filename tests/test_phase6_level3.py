"""Level 3: BeH2/NH3/CH4 (14-18Q) regression tests for Phase 6.

Verifies all Phase 6 improvements on medium-sized systems where
SKQD must handle larger Krylov subspaces.

Systems:
  BeH2: 14Q, 1225 configs, FCI ~ -15.835 Ha
  NH3:  16Q, 3136 configs, FCI ~ -56.195 Ha
  CH4:  18Q, 15876 configs, FCI ~ -40.199 Ha
"""

import sys
import os
import time
import math
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def beh2_system():
    try:
        from hamiltonians.molecular import create_beh2_hamiltonian
        H = create_beh2_hamiltonian(device="cpu")
        fci = H.fci_energy()
        return H, fci
    except ImportError:
        pytest.skip("PySCF not available")


@pytest.fixture(scope="module")
def nh3_system():
    try:
        from hamiltonians.molecular import create_nh3_hamiltonian
        H = create_nh3_hamiltonian(device="cpu")
        fci = H.fci_energy()
        return H, fci
    except ImportError:
        pytest.skip("PySCF not available")


@pytest.fixture(scope="module")
def ch4_system():
    try:
        from hamiltonians.molecular import create_ch4_hamiltonian
        H = create_ch4_hamiltonian(device="cpu")
        fci = H.fci_energy()
        return H, fci
    except ImportError:
        pytest.skip("PySCF not available")


def _print_result(test_id, energy, fci, wall_time, extra=""):
    error_mha = abs(energy - fci) * 1000
    status = "PASS" if error_mha < 1.6 else "WARN" if error_mha < 5 else "FAIL"
    print(f"\nPhase6-L3: {test_id}")
    print(f"  Energy: {energy:.8f} Ha")
    print(f"  FCI:    {fci:.8f} Ha")
    print(f"  Error:  {error_mha:.3f} mHa")
    print(f"  Time:   {wall_time:.1f}s")
    if extra:
        print(f"  {extra}")
    print(f"  Status: {status}")


# --- BeH2 (14Q, 1225 configs) ---

class TestL3BeH2Regression:
    """L3-BeH2-A: SKQD regression on BeH2."""

    @pytest.mark.molecular
    def test_beh2_chemical_accuracy(self, beh2_system):
        """Direct-CI + SKQD should achieve chemical accuracy on BeH2."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = beh2_system
        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device="cpu",
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        error_mha = abs(energy - fci) * 1000
        _print_result("L3-BeH2-A", energy, fci, wall, "Direct-CI + SKQD")

        assert error_mha < 1.6, f"Error {error_mha:.3f} mHa should be < 1.6"


class TestL3BeH2NNCI:
    """L3-BeH2-C: NNCI + NOs on BeH2."""

    @pytest.mark.molecular
    def test_beh2_nnci_nos(self, beh2_system):
        """NNCI with natural orbitals should run and produce valid energy on BeH2."""
        from krylov.nnci import NNCIConfig, NNCIActiveLearning

        H, fci = beh2_system
        basis = H.get_hf_state().unsqueeze(0)

        config = NNCIConfig(
            max_iterations=2, top_k=10, max_candidates=100,
            use_natural_orbitals=True,
        )
        t0 = time.time()
        nnci = NNCIActiveLearning(H, basis, config)
        result = nnci.run()
        wall = time.time() - t0

        _print_result("L3-BeH2-C", result["energy"], fci, wall,
                      f"NNCI+NOs, basis={result['basis_size']}")

        assert math.isfinite(result["energy"])
        assert result["basis_size"] >= 2


# --- NH3 (16Q, 3136 configs) ---

class TestL3NH3Regression:
    """L3-NH3-A: SKQD regression on NH3."""

    @pytest.mark.molecular
    def test_nh3_chemical_accuracy(self, nh3_system):
        """Direct-CI + SKQD should achieve chemical accuracy on NH3."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = nh3_system
        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device="cpu",
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        error_mha = abs(energy - fci) * 1000
        _print_result("L3-NH3-A", energy, fci, wall, "Direct-CI + SKQD")

        assert error_mha < 1.6, f"Error {error_mha:.3f} mHa should be < 1.6"


# --- CH4 (18Q, 15876 configs) ---

class TestL3CH4Regression:
    """L3-CH4-A: SKQD regression on CH4."""

    @pytest.mark.molecular
    def test_ch4_chemical_accuracy(self, ch4_system):
        """Direct-CI + SKQD should achieve chemical accuracy on CH4."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = ch4_system
        config = PipelineConfig(
            subspace_mode="skqd", skip_nf_training=True, device="cpu",
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        error_mha = abs(energy - fci) * 1000
        _print_result("L3-CH4-A", energy, fci, wall, "Direct-CI + SKQD")

        assert error_mha < 1.6, f"Error {error_mha:.3f} mHa should be < 1.6"


class TestL3CH4FullPipeline:
    """L3-CH4-B: Full P6 pipeline on CH4."""

    @pytest.mark.molecular
    def test_ch4_full_p6(self, ch4_system):
        """All P6 improvements should not degrade CH4 accuracy."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, fci = ch4_system
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
        error_mha = abs(energy - fci) * 1000
        _print_result("L3-CH4-B", energy, fci, wall, "ALL P6 ON")

        assert error_mha < 1.6, f"Error {error_mha:.3f} mHa should be < 1.6"
