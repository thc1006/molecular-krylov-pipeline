"""40Q iterative refinement validation — N2/cc-pVDZ CAS(10,20).

Validates that the iterative refinement loop (Phase 7) scales to 40 qubits
without OOM or numerical errors. FCI is intractable at 240M configs, so
accuracy is assessed by:
  1. Energy below HF (variational)
  2. Energy monotonically non-increasing across iterations (variational)
  3. No crash / OOM
  4. refinement_energies populated correctly

Use GPU when available for reasonable wall time (~1-3 min).
CPU is supported for CI but much slower.
"""

import sys
import os
import time
import math
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def n2_40q():
    """N2 CAS(10,20) — 40Q, 240M configs. Integrals only (no FCI)."""
    try:
        from hamiltonians.molecular import create_n2_cas_hamiltonian
        device = "cuda" if torch.cuda.is_available() else "cpu"
        H = create_n2_cas_hamiltonian(basis="cc-pvdz", cas=(10, 20), device=device)
        hf_state = H.get_hf_state()
        hf_energy = H.diagonal_element(hf_state).item()
        print(f"\n[40Q fixture] HF energy: {hf_energy:.8f} Ha  device={device}")
        return H, hf_energy
    except (ImportError, Exception) as e:
        pytest.skip(f"Cannot create N2 CAS(10,20): {e}")


class TestIterativeRefinement40Q:
    """Iterative refinement at 40Q scale — functional + scaling tests."""

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_single_iter_no_oom(self, n2_40q):
        """Single-iteration Direct-CI should complete at 40Q without OOM."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, hf_energy = n2_40q
        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device=device,
            n_refinement_iterations=1,
            max_diag_basis_size=5000,
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        print(f"\n[40Q single-iter] energy={energy:.8f} Ha  wall={wall:.1f}s")
        print(f"  Improvement: {(hf_energy - energy) * 1000:.1f} mHa below HF")

        assert math.isfinite(energy), "Energy must be finite"
        assert energy < hf_energy, f"Energy {energy:.8f} must be below HF {hf_energy:.8f}"
        assert "refinement_energies" not in result, "No refinement with n_iters=1"

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_two_iter_monotonic(self, n2_40q):
        """Two-iteration refinement: iter 2 energy ≤ iter 1 energy (variational)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, hf_energy = n2_40q
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,   # Iter 1 = Direct-CI
            device=device,
            n_refinement_iterations=2,
            h_coupling_weight=0.2,
            h_coupling_n_ref=50,
            refinement_epochs=15,    # Short for test speed
            max_epochs=15,
            min_epochs=5,
            samples_per_batch=200,
            entropy_weight=0.05,
            physics_weight=0.1,
            max_diag_basis_size=5000,
        )
        t0 = time.time()
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()
        wall = time.time() - t0

        energy = result["combined_energy"]
        refinement_energies = result.get("refinement_energies", [])
        print(f"\n[40Q 2-iter] final={energy:.8f} Ha  wall={wall:.1f}s")
        print(f"  Refinement trajectory: {[f'{e:.6f}' for e in refinement_energies]}")
        print(f"  Iter 1: {(hf_energy - refinement_energies[0])*1000:.1f} mHa below HF")
        if len(refinement_energies) >= 2:
            delta = (refinement_energies[0] - refinement_energies[1]) * 1000
            print(f"  Iter 2 improvement: {delta:.3f} mHa")

        assert math.isfinite(energy), "Final energy must be finite"
        assert energy < hf_energy, "Final energy must be below HF"
        assert len(refinement_energies) == 2, (
            f"Expected 2 refinement energies, got {len(refinement_energies)}"
        )

        # Variational principle: iter 2 ≤ iter 1 (basis is a superset)
        # Allow 0.5 mHa tolerance for numerical noise
        E1, E2 = refinement_energies[0], refinement_energies[1]
        assert E2 <= E1 + 5e-4, (
            f"Variational violation: iter2={E2:.6f} > iter1={E1:.6f} "
            f"(+{(E2 - E1)*1000:.3f} mHa, tolerance=0.5 mHa)"
        )
        print(f"  PASS: Monotonic refinement at 40Q scale in {wall:.0f}s")

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_nf_initialized_after_refinement(self, n2_40q):
        """NF should be initialized after 2 iterations (Direct-CI → NF transition)."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, hf_energy = n2_40q
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,   # Direct-CI for iter 1
            device=device,
            n_refinement_iterations=2,
            h_coupling_weight=0.1,
            h_coupling_n_ref=30,
            refinement_epochs=10,
            max_epochs=10,
            min_epochs=5,
            samples_per_batch=100,
            max_diag_basis_size=3000,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)

        # Before run: no flow (Direct-CI mode)
        assert pipeline.flow is None, "Flow should not exist before refinement"

        result = pipeline.run()

        # After run: flow should exist (created during iter 2 transition)
        assert pipeline.flow is not None, "Flow should be created during refinement"
        energy = result["combined_energy"]
        assert math.isfinite(energy), "Energy must be finite after NF initialization"
        print(f"\n[40Q NF init] PASS: flow={type(pipeline.flow).__name__}  "
              f"energy={energy:.8f}")

    @pytest.mark.slow
    @pytest.mark.molecular
    def test_convergence_stops_early_40q(self, n2_40q):
        """Loose convergence threshold should stop before max iterations at 40Q."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H, hf_energy = n2_40q
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device=device,
            n_refinement_iterations=4,           # Allow up to 4
            refinement_convergence_threshold=1.0,  # 1 Ha — extremely loose
            h_coupling_weight=0.1,
            h_coupling_n_ref=30,
            refinement_epochs=10,
            max_epochs=10,
            min_epochs=5,
            samples_per_batch=100,
            max_diag_basis_size=3000,
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        result = pipeline.run()

        n_iters = len(result.get("refinement_energies", []))
        print(f"\n[40Q convergence] completed {n_iters} iterations (max=4)")
        assert n_iters <= 3, (
            f"With 1 Ha threshold, should converge in ≤3 iterations, got {n_iters}"
        )
        assert math.isfinite(result["combined_energy"]), "Energy must be finite"
        print(f"  PASS: Early convergence at 40Q in {n_iters} iterations")
