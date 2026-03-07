# Implementation TODO (TDD Approach)

> Generated 2026-03-06. Updated with optimization research findings.
> Hardware: DGX Spark GB10 (20 ARM cores, 128GB UMA, CUDA 13.0, TF32=53 TFLOPS, FP64=0.48 TFLOPS)
> Principle: RED-GREEN-REFACTOR for every PR.
> Research: See `docs/OPTIMIZATION-RESEARCH.md` for full findings.

---

## Phase 0: Foundation ✅

### PR 0.1: Regression Test Suite + conftest.py ✅
- [x] Create `tests/conftest.py` with shared fixtures (H2, LiH, H2O, BeH2, NH3, CH4, N2 + GPU)
- [x] Create `tests/test_regression.py` — regression gates for all 7 molecules (22 tests)
- [x] Fix broken existing tests (test_flows, test_nqs, test_hamiltonians, test_pipeline, test_skqd)
- [x] Add GPU pipeline tests (`test_gpu_pipeline.py`, 4 tests)
- [x] Fix cu128→cu130 for NVRTC JIT on sm_121
- [x] Fix SQD/SKQD `use_gpu` hardcoding (device-aware)
- [x] Fix `_bitstring_to_tensor` device propagation
- [x] Verify all 67 tests GREEN

---

## Phase 1: Sparse & Scalability

### PR 1.1: Sparse Eigensolver Path ✅
- [x] TEST: `test_sparse_matrix_build` — COO from `get_sparse_matrix_elements()` matches dense
- [x] TEST: `test_sparse_eigsh_matches_dense` — sparse eigsh matches eigh for H2/LiH
- [x] TEST: `test_max_full_subspace_guard` — standard SKQD has guard
- [x] IMPL: Wire `get_sparse_matrix_elements()` into eigensolver (SPARSE_THRESHOLD=3000)
- [x] IMPL: Upgrade `get_sparse_matrix_elements` to use vectorized batch path

### PR 1.2: Importance-Ranked Basis Truncation ✅
- [x] TEST: `test_importance_truncation.py` — 7 tests (ranking, essential preservation, energy improvement)
- [x] IMPL: `_rank_and_truncate_basis()` ranks by diagonal energy, preserves essentials (excitation rank ≤ 2)
- [x] IMPL: Wired into `compute_ground_state_energy()` replacing blind `basis[:max_diag]`

### PR 1.3: Adaptive Doubles Allocation ✅
- [x] IMPL: Proportional allocation with αβ >= 50% in both pipeline.py and physics_guided_training.py

### ~~PR 1.6: Shift-Invert eigsh~~ ❌ REJECTED
> Tested on N2 (14,400 configs): 259s vs 1.45s standard mode. LU factorization cost
> dominates at moderate sizes. Only beneficial for very large (>100K) sparse matrices.
> Shift-invert is already available as `shift_invert=True` flag but NOT the default.

### PR 1.4: Streaming Diversity Selection ✅ (ADR-001 PR 1.4)
- [x] TEST: `test_streaming_diversity.py` — 14 tests (bitpack, stochastic greedy, edge cases)
- [x] TEST: `test_dpp_streaming.py` — 9 tests (DPP delegation, sampled analyze, N2 integration)
- [x] IMPL: `_dpp_select()` delegates to `stochastic_greedy_select()` (bitpacked XOR + min-dist)
- [x] IMPL: `analyze_basis_diversity()` uses sampled pairs for n > 5000
- [x] IMPL: Memory O(n) instead of O(n²). 10K configs: ~160KB vs 4GB.

### PR 1.8: Mixed-Precision Eigensolver (Tier 2, ~100x for diag on DGX Spark)
- [ ] TEST: `test_tf32_eigh_accuracy` — TF32 eigensolver within 0.1 mHa of FP64
- [ ] TEST: `test_mixed_precision_refinement` — FP32 solve + FP64 refinement matches FP64
- [ ] IMPL: TF32/FP32 Hamiltonian construction path
- [ ] IMPL: Iterative refinement: FP32 eigsh → FP64 Rayleigh quotient correction

---

## Phase 2: Acceleration & NF Unlock

### PR 2.0: Numba JIT for get_connections ✅
- [x] IMPL: `@numba.njit` for single/double excitation enumeration (18.7x speedup)
- [x] IMPL: Fallback to Python when numba unavailable

### PR 2.1: NVPL BLAS/LAPACK (Tier 2, 2-5x CPU linear algebra)
- [ ] TEST: `test_nvpl_eigh_matches_openblas` — identical eigenvalues
- [ ] IMPL: Build NumPy/SciPy with NVPL backend (or conda `blas=*=nvpl`)

### PR 2.2: NF Architecture Fixes ✅
- [x] IMPL: Remove dead `beta_scorer` (~160K params)
- [x] IMPL: SigmoidTopK replaces GumbelTopK (gradient-exact, deterministic)
- [x] IMPL: Auto-scale hidden_dims by n_orbitals
- [x] IMPL: Alpha-beta input fix (no zero-padding waste)
- [x] IMPL: Learnable temperature (nn.Parameter + softplus + min_temperature=0.1)
- [x] IMPL: Non-autoregressive limitation documented in class docstring
- [x] TEST: `test_learnable_temperature.py` — 12 tests

### PR 2.3: Sigmoid top-k ✅ (merged with PR 2.2)
- [x] IMPL: SigmoidTopK with implicit differentiation (exact Jacobian)
- [x] IMPL: Plackett-Luce probability model (exact k≤5, approximate k>5)

### PR 2.4: NF Probability Model Fix ✅
- [x] IMPL: Sequential conditional factorization (`_sequential_conditional`)
- [x] IMPL: Loss scaling `|E|/batch_size` (not `|E|/|S|`)
- [x] IMPL: Entropy separated from energy scaling
- [x] DOC: REINFORCE 2x factor documented
- [x] DOC: `_compute_subspace_energy` no-grad documented
- [x] TEST: `test_probability_model.py`, `test_physics_training_config.py`

### PR 2.5: Enable NF Training for Molecular Systems ✅
- [x] IMPL: Conditional NF enable (>20K configs) with `_user_set_skip_nf` override
- [x] IMPL: `physics_weight=0.1`, `convergence_threshold=0.35`
- [x] IMPL: Factory function defaults reconciled with PhysicsGuidedConfig
- [x] TEST: `test_conditional_nf.py` — 14 tests

### PR 2.6: Validate NF on Molecular Systems ✅
- [x] TEST: `test_nf_lih_basic` — NF trains on LiH without error
- [x] TEST: `test_nf_particle_conservation` — sampled configs preserve electron count
- [x] TEST: `test_nf_h2o_sample_diversity` — NF produces diverse excitation ranks
- [x] TEST: `test_nf_beh2_convergence` — energy decreases over training
- [x] TEST: `test_nf_produces_valid_configs` — binary configs, correct length
- [x] TEST: `test_nf_training_with_physics_weight` — no NaN/Inf with physics_weight>0
- [x] FIX: `_compute_flow_loss` NameError (`configs` → `all_configs`) at line 1113

---

## Phase 3: 40Q Validation

### PR 3.3a: cc-pVDZ + CAS Active Space Support ✅
- [x] IMPL: `cas` parameter in `compute_molecular_integrals()` — CASSCF → active-space integrals
- [x] IMPL: `basis` parameter on all 7 factory functions (backward-compatible defaults)
- [x] IMPL: `create_n2_cas_hamiltonian()` factory (CAS(10,8) on cc-pVDZ, 3136 configs)
- [x] IMPL: Cache bypass for CAS computations
- [x] TEST: `test_cas_support.py` — 21 tests (basis param, CAS shapes, FCI vs CASSCF golden test)

### PR 3.2: Perturbative Pruning ✅
- [x] IMPL: `compute_mp2_amplitudes()` — t2 from stored MO integrals (no PySCF re-run)
- [x] IMPL: `mp2_importance_scores()` — HF/singles essential, doubles by |t2|, triples by product
- [x] IMPL: `prune_basis()` — top-K by score, preserves essential configs
- [x] TEST: `test_perturbative_pruning.py` — 16 tests (MP2 amplitudes, scoring, pruning vs random)

### PR 3.4: Benchmark Suite ✅
- [x] IMPL: `BenchmarkTimer` (wall-clock), `MemoryTracker` (peak memory), `BenchmarkSuite` (regression detection)
- [x] TEST: `test_benchmark.py` — 34 tests (timer, memory, suite, pipeline benchmarks)

### PR 3.1: NNCI Active Learning (research finding, 10⁵x subspace reduction)
- [ ] TEST: `test_nnci_classifier_trains` — NN learns to predict large CI coefficients
- [ ] TEST: `test_nnci_reduces_subspace` — fewer configs with same energy accuracy
- [ ] IMPL: NN classifier for determinant importance scoring
- [ ] IMPL: Active learning loop: classify → diag → retrain

### PR 3.3b: N2/cc-pVDZ CAS(10,20) Integration Test
- [ ] TEST: Full pipeline on CAS(10,8) or CAS(10,20) system completes without OOM/crash
- [ ] TEST: Energy within expected range

---

## Phase 4: Advanced Methods

### PR 4.1: PySCF selected_ci Backend (IBM approach)
- [ ] TEST: `test_pyscf_selected_ci_matches_our_diag` — same energy
- [ ] IMPL: Delegate to `pyscf.fci.selected_ci.kernel_fixed_space` (avoids building H)

### PR 4.2: SQD S² Vectorization
- [ ] TEST: `test_s_squared_vectorized` — matches Python loop result
- [ ] IMPL: Vectorize S² computation in sqd.py

### PR 4.3: CuPy GPU Sparse Eigensolver (for 40Q+)
- [ ] TEST: `test_cupy_eigsh_matches_scipy` — same ground state energy
- [ ] IMPL: CuPy sparse eigsh path for basis > 15K
- [ ] DEP: Add cupy-cuda13x to pyproject.toml

---

## Environment

- [x] Install CUDA-enabled PyTorch (torch 2.10.0+cu130, sm_121 binary compatible)
- [x] Fix NVRTC JIT for complex128 on GB10
- [ ] Add numba to dependencies (PR 2.0)
- [ ] Build NVPL-linked NumPy/SciPy (PR 2.1)
- [ ] Add cupy-cuda13x to dependencies (PR 4.3)
- [ ] Verify `uv sync --extra dev` installs all test dependencies
