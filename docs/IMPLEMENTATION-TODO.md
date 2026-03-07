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

### PR 1.1: Sparse Eigensolver Path (partially done)
- [x] TEST: `test_sparse_matrix_build` — COO from `get_sparse_matrix_elements()` matches dense
- [x] TEST: `test_sparse_eigsh_matches_dense` — sparse eigsh matches eigh for H2/LiH
- [x] TEST: `test_max_full_subspace_guard` — standard SKQD has guard
- [x] IMPL: Wire `get_sparse_matrix_elements()` into eigensolver (SPARSE_THRESHOLD=3000)
- [x] IMPL: Upgrade `get_sparse_matrix_elements` to use vectorized batch path

### PR 1.3: Adaptive Doubles Allocation ✅
- [x] IMPL: Proportional allocation with αβ >= 50% in both pipeline.py and physics_guided_training.py

### PR 1.5: SQD Batch Parallelization ⬅️ NEXT (Tier 1, 8-10x speedup)
- [ ] TEST: `test_sqd_parallel_batches` — parallel results match sequential
- [ ] TEST: `test_sqd_parallel_speedup` — wall time < sequential / n_workers
- [ ] TEST: `test_sqd_parallel_energy_accuracy` — energy matches non-parallel within tolerance
- [ ] IMPL: `ProcessPoolExecutor` with `OPENBLAS_NUM_THREADS=1` in `sqd.py`
- [ ] IMPL: Serializable batch diag function (no lambda/closure)

### PR 1.6: Shift-Invert eigsh (Tier 1, 2-5x speedup)
- [ ] TEST: `test_shift_invert_matches_standard` — same ground state energy
- [ ] TEST: `test_shift_invert_fewer_iterations` — fewer ARPACK iterations
- [ ] IMPL: Add `sigma=E_hf` shift-invert mode to `_sparse_ground_state()`

### PR 1.7: Streaming Diversity Selection (Tier 1, ~40,000x for n=50K)
- [ ] TEST: `test_bitpacked_hamming` — uint64 popcount matches naive Hamming
- [ ] TEST: `test_stochastic_greedy_coverage` — selected set covers all excitation ranks
- [ ] TEST: `test_stochastic_greedy_no_oom` — handles 50K+ configs, < 1GB memory
- [ ] TEST: `test_stochastic_greedy_quality` — energy within 1 mHa of full DPP result
- [ ] IMPL: Bit-parallel Hamming via `torch.bitwise_count` (O(1) per pair)
- [ ] IMPL: Stochastic Greedy (Mirzasoleiman 2015) replacing O(n²) broadcast

### PR 1.8: Mixed-Precision Eigensolver (Tier 2, ~100x for diag on DGX Spark)
- [ ] TEST: `test_tf32_eigh_accuracy` — TF32 eigensolver within 0.1 mHa of FP64
- [ ] TEST: `test_mixed_precision_refinement` — FP32 solve + FP64 refinement matches FP64
- [ ] IMPL: TF32/FP32 Hamiltonian construction path
- [ ] IMPL: Iterative refinement: FP32 eigsh → FP64 Rayleigh quotient correction

---

## Phase 2: Acceleration & NF Unlock

### PR 2.0: Numba JIT for get_connections (Tier 2, 50-200x)
- [ ] TEST: `test_numba_connections_matches_python` — identical results for H2/LiH/N2
- [ ] TEST: `test_numba_connections_speedup` — > 10x faster than Python loops
- [ ] IMPL: `@numba.njit` XOR+popcount excitation-degree detection (Scemama algorithm)
- [ ] IMPL: Numba-compiled Slater-Condon single/double matrix element evaluation
- [ ] DEP: Add numba to pyproject.toml

### PR 2.1: NVPL BLAS/LAPACK (Tier 2, 2-5x CPU linear algebra)
- [ ] TEST: `test_nvpl_eigh_matches_openblas` — identical eigenvalues
- [ ] IMPL: Build NumPy/SciPy with NVPL backend (or conda `blas=*=nvpl`)

### PR 2.2: NF Architecture Fixes
- [ ] TEST: `test_no_dead_beta_scorer` — no unused beta_scorer params
- [ ] TEST: `test_gumbel_gradient_flow` — gradients flow to all positions
- [ ] TEST: `test_entropy_weight_nonzero` — default entropy_weight > 0
- [ ] IMPL: Remove dead `beta_scorer` (~160K params)
- [ ] IMPL: Fix GumbelTopK gradient masking
- [ ] IMPL: Set default entropy_weight > 0

### PR 2.3: Sigmoid top-k (research finding, O(n) vs O(nk))
- [ ] TEST: `test_sigmoid_topk_particle_conservation` — exact electron count
- [ ] TEST: `test_sigmoid_topk_gradient_flow` — gradients to all positions
- [ ] IMPL: Sigmoid top-k as drop-in replacement for GumbelTopK

### PR 2.4: NF Probability Model Fix
- [ ] TEST: `test_sequential_log_prob` — conditional factorization matches exact
- [ ] TEST: `test_loss_scaling_correct` — |E|/batch_size not |E|/|S|
- [ ] IMPL: Replace product-of-marginals with sequential conditional
- [ ] IMPL: Fix loss scaling denominator

### PR 2.5: Enable NF Training for Molecular Systems
- [ ] TEST: `test_nf_training_enabled` — molecular systems can train NF
- [ ] TEST: `test_nf_produces_valid_configs` — particle conservation
- [ ] TEST: `test_nf_improves_energy` — NF configs improve over Direct-CI
- [ ] IMPL: Remove `skip_nf_training = True` force at pipeline.py:217

---

## Phase 3: 40Q Validation

### PR 3.1: NNCI Active Learning (research finding, 10⁵x subspace reduction)
- [ ] TEST: `test_nnci_classifier_trains` — NN learns to predict large CI coefficients
- [ ] TEST: `test_nnci_reduces_subspace` — fewer configs with same energy accuracy
- [ ] IMPL: NN classifier for determinant importance scoring
- [ ] IMPL: Active learning loop: classify → diag → retrain

### PR 3.2: Perturbative Pruning (PIGen-SQD finding, 70% subspace reduction)
- [ ] TEST: `test_mp2_importance_scoring` — MP2 amplitudes rank configs correctly
- [ ] IMPL: MP2/CISD-level importance scoring before diagonalization

### PR 3.3: N2/cc-pVDZ CAS(10,20) Integration Test
- [ ] TEST: Full pipeline on 40Q system completes without OOM/crash
- [ ] TEST: Energy within expected range

### PR 3.4: Benchmark Suite
- [ ] TEST: Wall-time regression tracking
- [ ] TEST: Memory peak tracking

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
