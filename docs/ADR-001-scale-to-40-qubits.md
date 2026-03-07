# ADR-001: Scale Flow-Guided Krylov Pipeline to 40 Qubits

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-03-06 |
| **Authors** | Project Team |
| **Deciders** | Project Team |
| **Supersedes** | N/A |
| **Related** | SPEC.md, RESULTS.md |

---

## 1. Context & Motivation

### 1.1 Current State

The Flow-Guided Krylov Pipeline currently supports molecular systems up to **20 qubits** (N2/STO-3G, 10 spatial orbitals, 14,400 configs). All validated results (RESULTS.md) are within this range, achieving chemical accuracy (<1.6 mHa) on 7 molecules from H2 (4Q) to N2 (20Q).

### 1.2 Target

Scale the pipeline to reliably handle **40-qubit** molecular systems (~20 spatial orbitals). At this scale:

- Configuration space varies dramatically by electron filling:
  - Quarter-filled (10e/20o, e.g. N2): **C(20,5)^2 ≈ 240M** determinants
  - Three-quarter-filled (30e/20o, e.g. [2Fe-2S]): **C(20,15)^2 ≈ 240M** determinants
  - True half-filled (20e/20o): **C(20,10)^2 ≈ 34.1B** determinants (hardest case)
- Essential subspace (HF + singles + doubles): **~7.9K-14K** configs, covering **<0.01%** of total space
- Practical diagonalization subspace: **10K-100K** selected configs
- Two-body integrals: **20^4 = 160K** elements (manageable)
- Full enumeration is impossible; **sampling-based methods become essential**
- NF training is **critical** (not optional) for discovering important higher excitations beyond singles+doubles

### 1.3 Why 40 Qubits Matters

40 qubits sits at the frontier where quantum-classical hybrid methods provide genuine scientific value:

- **IBM demonstrated** SQD on [2Fe-2S] clusters at 45 qubits and [4Fe-4S] at 77 qubits on Heron processor (Science Advances, 2025). Note: these are *circuit* qubit counts including routing overhead on Heron's heavy-hex topology; the active spaces are CAS(30,20) = 40 spin-orbitals and CAS(54,36) = 72 spin-orbitals respectively. The extra qubits (5 and 5) are for non-local gate routing in the LUCJ ansatz, not quantum error correction.
- **UTokyo/IBM KQD** showed exponential convergence on 56-qubit lattice systems (Nature Communications, 2025)
- **DMET-SQD** decomposed 41- and 89-qubit cyclohexane conformers into 27-32 qubit fragments achieving <1 kcal/mol accuracy (JCTC, 2025)
- The **25-100 logical qubit** regime maps to 20-50 orbital active spaces — the sweet spot for near-term quantum advantage in chemistry (arXiv:2506.19337)
- Molecules at this scale (iron-sulfur clusters, chromium dimer, porphyrins) are **strongly correlated** — precisely where normalizing flows should outperform Direct-CI

### 1.4 Known Criticisms

The "Critical Limitations in Quantum-Selected Configuration Interaction Methods" paper (arXiv:2501.07231, Jan 2025; colloquially known as the "Fatal Flaw" paper) identified that QSCI/SQD methods suffer from sampling inefficiency as system size grows — repeatedly sampling already-seen configurations. IBM responded with **SqDRIFT** (arXiv:2508.02578, randomized Krylov with provable convergence), **higher expansion order** (r=2→5, arXiv:2512.04962), and **ph-AFQMC-SQD** (using SQD wavefunctions as AFQMC trial states). Our SKQD solver, which already dramatically outperforms SQD (193x better on N2 per RESULTS.md), naturally mitigates this via Krylov time-evolution exploration. Nevertheless, we must design for these scaling challenges.

**Important caveat**: NF-NQS has been demonstrated on spin systems up to 50 sites (arXiv:2506.12128), but **no published work demonstrates NF-NQS specifically on molecular electronic structure beyond 20 qubits**. Our Phase 2 NF unlock is a reasonable extrapolation but carries execution risk. Separately, IBM's **PIGen-SQD** (arXiv:2512.06858, submitted Dec 2025, published Jan 2026) uses physics-informed generative ML to accelerate SQD up to 52 qubits on Heron R2/R3 — this validates the generative-model-guided approach but also represents a direct competitor. Recent theoretical work on normalizing flows for the electronic Schrödinger equation with DPP base distributions (arXiv:2406.00047) provides additional foundation for molecular NF approaches.

---

## 2. Decision

We will execute a **5-phase incremental scale-up** from 20Q to 40Q (Phase 0 prerequisites + 4 implementation phases), with each phase independently testable and deployable. The phases are ordered by dependency: test infrastructure first (Phase 0), then sparse infrastructure (Phase 1), then NF unlock (Phase 2), then new molecules (Phase 3), then advanced methods (Phase 4).

**Timeline summary**: Phase 0 (Week 0) + Phase 1 (Weeks 1-3) + Phase 2 (Weeks 4-8) + Phase 3 (Weeks 9-14) + Phase 4 (Weeks 15-22) = **22 weeks** + 4 weeks contingency buffer = **26 weeks realistic**. Phase 2 (molecular NF first-ever validation) and Phase 3 (CASSCF convergence) carry highest schedule risk.

**Key architectural principles:**
1. **Sparse-first**: Replace dense matrix operations with sparse equivalents as the default path
2. **Importance-ranked truncation**: Never discard configs blindly; always rank by energy or probability
3. **Adaptive scaling**: All limits scale with system size, no hardcoded caps
4. **NF unlocked**: Normalizing flow training must be available for large systems where Direct-CI is insufficient
5. **Incremental validation**: Each change validated against existing RESULTS.md baselines before proceeding

---

## 3. Current Bottleneck Analysis

### 3.1 Bottleneck Inventory

We audited every source file. The table below lists all scaling blockers with exact locations.

| # | Bottleneck | File:Line | Current Limit | Impact at 40Q | Severity |
|---|-----------|-----------|--------------|---------------|----------|
| B1 | Dense eigensolve | `src/krylov/skqd.py:803-820` | `torch.linalg.eigh` for n<=10K, CuPy `eigsh` for larger | 15K x 15K dense = 1.8GB, O(n^3) | **CRITICAL** |
| B2 | Basis truncation (no ranking) | `src/krylov/skqd.py:759-764` | `basis[:max_diag]` — keeps first N by index | Loses important configs arbitrarily | **CRITICAL** |
| B3 | Essential doubles hardcoded cap + ordering bias | `src/pipeline.py:444` | `max_doubles = 5000`, enumerated αα→ββ→αβ | 40Q: 7,725 doubles, cap discards 35%; **worse**: αβ doubles (most important for correlation) enumerated last → 48% of αβ discarded while all αα/ββ kept | **CRITICAL** |
| B4 | NF training force-disabled | `src/pipeline.py:217` | `self.skip_nf_training = True` (always) | NF never trains; Direct-CI basis insufficient at 40Q | **CRITICAL** |
| B5 | `get_connections()` nested loops | `src/hamiltonians/molecular.py:504-569` | 4-nested Python for-loops for doubles | ~8.1K iterations/config (7.7K doubles + 0.4K singles) × 50K configs = 405M loops | **HIGH** |
| B6 | Hamming distance matrix O(n^2) | `src/postprocessing/diversity_selection.py:73-93` | Full (n,n,sites) tensor via broadcasting | DPP path: largest bucket ~25K → ~25GB peak; diagnostic path: 50K → ~100GB | **HIGH** |
| B7 | `max_diag_basis_size` cap | `src/krylov/skqd.py:87` | `15000` | Adequate for 40Q if sparse solver works; needs tuning | **MEDIUM** |
| B8 | NF network capacity | `src/flows/particle_conserving_flow.py` | `hidden_dims=[256, 256]` default | May underfit 20-orbital correlations | **MEDIUM** |
| B9 | SQD S^2 matrix fallback | `src/krylov/sqd.py:775-842` | GPU path for n<=2000, Python loop otherwise | 10K batch: O(n^2) Python loops | **MEDIUM** |
| B10 | `adapt_to_system_size()` tiers | `src/pipeline.py:206-213` | 4 tiers, max="very_large" (>20K) | No 40Q-specific tuning | **LOW** |
| B11 | SQD S² O(n²) Python loop at n>2000 | `src/krylov/sqd.py:826-840` | Vectorized for n<=2000, serial Python double loop otherwise | 40Q batch=33K: 544M iterations, **hours** of compute. Blocks SQD at scale. | **CRITICAL** |
| B12 | NF non-autoregressive architecture | `src/flows/particle_conserving_flow.py:232-243` | Neither alpha nor beta channel uses autoregressive factoring | Cannot capture intra-channel orbital correlations (e.g., "orbital 3 occupied → orbital 5 also occupied"). Fatal for strongly correlated systems (Cr2, [2Fe-2S]). | **HIGH** |
| B13 | Standard SKQD crashes at 40Q | `src/krylov/skqd.py:141` | `_setup_particle_conserving_subspace()` enumerates all configs; no `MAX_FULL_SUBSPACE_SIZE` guard (only `FlowGuidedSKQD` has it at line 942) | At 40Q: attempts to enumerate 240M configs → OOM crash | **CRITICAL** |
| B14 | `_topk_log_prob` incorrect probability model | `src/flows/particle_conserving_flow.py:274-295` | Uses product-of-marginals with k! correction. Treats orbital selections as independent, but they are dependent (without replacement). | Probability bias 10-30% per config; **ranking of configs can be inverted**. Training optimizes wrong target distribution. | **CRITICAL** |
| B15 | `\|E\|/\|S\|` loss scaling punishes diversity | `src/flows/physics_guided_training.py:1087` | Loss multiplied by `\|E\|/\|S\|` where \|S\| = unique sample count | Mode-collapsed flow (\|S\|=1) gets 1000x more gradient than diverse flow (\|S\|=1000). Directly incentivizes collapse. | **CRITICAL** |
| B16 | `beta_scorer` dead code | `src/flows/particle_conserving_flow.py:186-188` | `OrbitalScoringNetwork` instantiated but never called in `sample()` or `log_prob()` | ~160K wasted parameters (half the model). `beta_conditioned_scorer` is used instead. | **MEDIUM** |
| B17 | Krylov time evolution builds dense H | `src/krylov/skqd.py:1117` | `_generate_krylov_samples_nf_guided()` builds dense H at each Krylov step | If basis grows to 15K per step, each step needs 1.8 GB dense matrix | **HIGH** |

### 3.2 Memory Budget Analysis (128GB UMA, DGX Spark)

| Component | 20Q (current) | 40Q (target) | Notes |
|-----------|--------------|-------------|-------|
| H_proj dense matrix (15K) | 1.8 GB | 1.8 GB | Capped by max_diag_basis_size |
| H_proj dense matrix (50K) | N/A | **20 GB** | If cap raised |
| H_proj sparse CSR (50K, ~300-1K avg nnz/row) | N/A | **0.2-0.6 GB** | Sparse alternative (nnz/row varies 36-7.8K; essential-only avg ~100, with NF triples avg ~300-1K) |
| Hamming distance peak | N/A | **25-100 GB** | DPP per-bucket: ~25GB (25K doubles bucket); diagnostic: ~100GB (50K full). Must eliminate. |
| Two-body integrals h2e | 0.08 MB | 1.3 MB | Not a concern |
| NF model parameters | ~1.2 MB | ~5.9 MB | Not a concern |
| NQS model parameters | ~5 MB | ~10 MB | Not a concern |
| ConnectionCache (10K entries) | ~200 MB | ~500 MB | LRU keeps bounded |
| PySCF integral computation | ~200 MB | ~1 GB | One-time |
| **Total estimated** | **~2.3 GB** | **~5-25 GB** | Depends on sparse vs dense |

**Conclusion**: 40Q is feasible on 128GB UMA **if** we: (1) use sparse eigensolvers, (2) eliminate O(n²) distance matrices, and (3) never materialize the full config space (240M configs). The diagonalization subspace (10K-50K selected configs) fits comfortably; the challenge is **finding** the right configs from the vast space.

**Note on nnz/row**: In the projected Hamiltonian, nnz per row is the number of connections that fall *within* the selected basis (not the total connections per config). Slater-Condon analysis shows **highly asymmetric** row densities:
- **HF row**: connects to all ~7,876 essential configs (all S+D excitations). nnz ≈ 7,876.
- **Single excitation rows** (150 configs): a single excitation from a single → produces singles or doubles from HF, but from most doubles → produces triples (NOT in essentials). Typical nnz ≈ **200-300**.
- **Double excitation rows** (7,725 configs): a single/double excitation from a double mostly produces triples/quadruples from HF. Within the essential-only basis, typical nnz ≈ **36-50** (only connections that undo/rearrange the original excitations land back in essentials).

**Weighted average nnz/row ≈ 54-128** for the essential-only basis (dominated by the sparse double rows), not 7,700. For a 50K basis including NF-discovered triples/quadruples, average nnz/row may reach **300-3,000** depending on basis composition. The sparse matrix is **0.1-6% dense** — genuinely sparse, making sparse eigensolvers highly effective. CSR memory estimate: 50K × 1K avg nnz × 12 bytes ≈ **0.6 GB** (much less than the 20 GB dense alternative).

---

## 4. Phased Implementation Plan

### Phase 0: Prerequisites (Week 0)

**Goal**: Establish test infrastructure before any code changes.

#### PR 0.1: Regression Gate as Standalone Test Suite

**File**: `tests/test_regression.py` (**New**)

**Rationale**: The regression gate is referenced by every subsequent PR's acceptance criteria, but currently exists only as a concept. It must be a working, runnable test suite before any Phase 1 code changes begin. Additionally, `tests/` is currently in `.gitignore` — this must be fixed so tests are tracked in version control.

**Change**:
1. Remove `tests/` from `.gitignore` (or create tests in a tracked location like `src/tests/`)
2. Create `tests/test_regression.py` that runs all 7 RESULTS.md molecules and asserts energies within 0.01 mHa
3. Create `tests/conftest.py` with shared fixtures (Hamiltonian creation, pipeline config)

**Acceptance criteria**:
- [ ] `tests/` tracked in version control
- [ ] `pytest tests/test_regression.py` passes on current codebase
- [ ] All 7 molecules (H2, LiH, H2O, BeH2, NH3, CH4, N2) tested
- [ ] Shared fixtures in `conftest.py` to avoid duplication across test files

---

#### PR 0.2: Minimal CI/CD

**File**: `.github/workflows/regression.yml` (**New**)

**Rationale**: With 16+ PRs modifying core code over 18+ weeks, manual regression testing is error-prone. A minimal GitHub Actions workflow running `pytest tests/test_regression.py` catches regressions automatically on every push.

**Change**: Create GitHub Actions workflow that:
1. Sets up Python 3.10 + uv
2. Installs dependencies (`uv sync --extra dev`)
3. Runs `uv run pytest tests/test_regression.py -v`

**Acceptance criteria**:
- [ ] CI runs on every push to main and on PRs
- [ ] Regression test results visible in PR checks
- [ ] PySCF tests skipped gracefully if PySCF unavailable in CI

---

### Phase 1: Sparse Infrastructure (Weeks 1-3)

**Goal**: Make the pipeline capable of handling >15K config bases without OOM or hangs.

#### PR 1.1: Sparse Eigensolver as Default Path

**File**: `src/utils/gpu_linalg.py` (lines 66-134), `src/krylov/skqd.py` (lines 800-820)

**Current**: `gpu_eigsh()` uses dense `torch.linalg.eigh` for n<=10,000, CuPy sparse for larger. But the input `H` is always a dense tensor — even the "sparse" path converts a dense matrix to CuPy CSR.

**Change**:
```python
# New function: build and solve directly in sparse format
def sparse_hamiltonian_eigsh(
    hamiltonian,           # MolecularHamiltonian
    basis: torch.Tensor,   # (n_basis, n_sites)
    k: int = 2,
    which: str = 'SA',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build H_ij in sparse CSR format and solve with iterative eigensolver.

    Never materializes full dense matrix. Memory: O(n * nnz_per_row)
    instead of O(n^2).

    For molecular Hamiltonians with Slater-Condon rules:
    nnz_per_row varies by config type (HF: ~7.8K, singles: ~250, doubles: ~36-50).
    Weighted average: ~54-128 for essential-only basis, ~300-3K with NF configs.
    For 40Q with 50K basis: matrix is 0.6-6% dense, ~0.6 GB CSR vs 20 GB dense.
    """
```

**Existing code to build on**: `get_sparse_matrix_elements()` already exists at `src/hamiltonians/molecular.py:1220` and returns COO format. However, it uses the slow per-config `get_connections()`. Additionally, `matrix_elements_fast()` at line 1085 already uses the faster `get_connections_vectorized_batch()` but outputs a dense matrix.

**Algorithm**:
1. Build a hash set of all basis config integer encodings for O(1) membership testing
2. Process configs in **mini-batches** (e.g., 1000 at a time) via `get_connections_vectorized_batch()`
3. For each mini-batch, immediately filter connections against the hash set and append to COO lists
4. Convert accumulated COO triplets to CSR
5. Call `scipy.sparse.linalg.eigsh` (CPU) or `cupyx.scipy.sparse.linalg.eigsh` (GPU)

**Memory analysis**: The naive approach (generate all connections then filter) produces 50K × 7.7K = 385M intermediate connections ≈ **21.6 GB** — almost as large as the dense matrix (20 GB), defeating the purpose. The mini-batch approach (step 2-3) keeps intermediate memory bounded by `batch_size × max_connections_per_config × n_sites × sizeof(long)`. With batch_size=1000: `get_connections_vectorized_batch` returns full config tensors (not scalars), so peak = 1K × 7.7K connections × 40 sites × 4 bytes ≈ **1.3 GB** (not 430 MB — the connected configs are full tensors, not COO triplets). With batch_size=500: ~650 MB peak. Use batch_size=500 for 40Q systems.

**Critical integration point**: The call site at `skqd.py:768` currently calls `self.hamiltonian.matrix_elements(basis, basis)`, which dispatches to `matrix_elements_fast()` and **always builds a dense matrix**. This call site must be modified to use the new sparse path when `n_basis > dense_threshold`.

**Eigensolver selection logic** (new):
```
n <= 5,000  → dense torch.linalg.eigh (fastest for small)
5,000 < n <= 50,000  → sparse eigsh (Lanczos, optimal for 1-2 eigenvalues)
n > 50,000  → LOBPCG (memory-efficient for very large sparse matrices)
```

**Rationale**: For computing 1-2 lowest eigenvalues (our use case), **Lanczos-based `eigsh`** is the standard choice — it is the implicit restart Lanczos algorithm optimized for finding extremal eigenvalues of sparse symmetric matrices. Davidson is preferred when computing **many** roots (e.g., 10+ excited states) in diagonally-dominant systems, which is not our case. LOBPCG uses significantly less memory for very large matrices (Theor. Chem. Acc. 142, 2023; Nottoli et al.) and serves as a fallback for n > 50K where even sparse Lanczos memory becomes a concern.

**Validation**: Run existing RESULTS.md benchmarks (H2 through N2). All energies must match within 0.01 mHa. No regression.

**Additional scope** (from Round 4 audit):
- **C8**: Add `MAX_FULL_SUBSPACE_SIZE` guard to `SampleBasedKrylovDiagonalization._setup_particle_conserving_subspace()` at `skqd.py:141`. Currently only `FlowGuidedSKQD` (line 942) has this protection — standard SKQD will crash at 40Q by attempting to enumerate 240M configs.
- **H3**: The Krylov time evolution in `FlowGuidedSKQD._generate_krylov_samples_nf_guided()` at line 1117 also builds dense H via `_build_hamiltonian_in_basis_gpu()`. The sparse path must be used here too, not just at the final eigensolve.
- **H6**: `_compute_accumulated_energy()` in `physics_guided_training.py:1230` builds a dense matrix then converts to CSR for eigsh. Should build sparse directly using the new `sparse_hamiltonian_eigsh()`.
- **H12**: `max_diag_basis_size=15000` should be tuned per system size. At 40Q, 50K may be needed. Add to `adapt_to_system_size()` scaling: e.g., 15K for <=20Q, 30K for 24-32Q, 50K for 40Q+.

**Acceptance criteria**:
- [ ] `sparse_hamiltonian_eigsh()` implemented and tested
- [ ] `gpu_eigsh()` updated with eigsh/LOBPCG tiers
- [ ] All 7 RESULTS.md molecules produce identical results
- [ ] Memory usage at 15K basis reduced by >50%
- [ ] `MAX_FULL_SUBSPACE_SIZE` guard added to standard SKQD (not just FlowGuidedSKQD)
- [ ] Krylov time evolution steps use sparse H construction
- [ ] `max_diag_basis_size` scales with system size in `adapt_to_system_size()`

---

#### PR 1.2: Importance-Ranked Basis Truncation

**File**: `src/krylov/skqd.py` (lines 759-764)

**Current**:
```python
# Line 759-764: BLIND TRUNCATION
max_diag = self.config.max_diag_basis_size
if max_diag > 0 and len(basis) > max_diag:
    basis = basis[:max_diag]  # <-- Just keeps first N by index order!
```

**Change**: Before truncation, rank configs by diagonal energy `<x_i|H|x_i>` and keep the lowest-energy configs. Essential configs (HF + singles + doubles) are always preserved.

```python
def _rank_and_truncate_basis(self, basis, max_size, essential_mask=None):
    """Rank basis by diagonal energy, keep lowest + all essential."""
    # Use batch method (not per-config loop) for performance at 50K configs
    diag_energies = self.hamiltonian.diagonal_elements_batch(basis)

    # Essential configs always survive
    if essential_mask is not None:
        diag_energies[essential_mask] = float('-inf')

    # Keep top-max_size by lowest energy
    _, indices = torch.sort(diag_energies)
    return basis[indices[:max_size]]
```

**Validation**: On N2 (14,400 configs), artificially cap to 5K and compare energy with/without ranking. Ranked truncation must yield lower energy.

**Acceptance criteria**:
- [ ] Truncation uses diagonal energy ranking
- [ ] Essential configs protected from truncation
- [ ] Energy improvement demonstrated on N2 with artificial cap

---

#### PR 1.3: Adaptive Essential Doubles Cap

**File**: `src/pipeline.py` (lines 443-496)

**Current**: Two bugs:

1. **Hardcoded cap**: `max_doubles = 5000` discards 35% of doubles at 40Q.
2. **Ordering bias** (worse): Doubles are enumerated αα→ββ→αβ in sequence with a shared counter. Alpha-beta doubles — the most important for correlation energy — are generated **last** and thus preferentially discarded. At 40Q with a 5000 cap: αα gets all 1,050, ββ gets all 1,050, leaving only 2,900/5,625 (48%) of αβ doubles.

Examples at 40Q:

For N2 CAS(10,20) — 5α, 5β in 20 orbitals (5 occ, 15 virt per spin):
- Alpha-alpha: C(5,2) × C(15,2) = 10 × 105 = 1,050
- Beta-beta: 1,050
- Alpha-beta: 5 × 15 × 5 × 15 = 5,625
- **Total: 7,725** doubles → exceeds 5,000 cap (35% discarded, **48% of αβ lost**)

For [2Fe-2S] CAS(30,20) — 15α, 15β in 20 orbitals (15 occ, 5 virt per spin):
- Alpha-alpha: C(15,2) × C(5,2) = 105 × 10 = 1,050
- Beta-beta: 1,050
- Alpha-beta: 15 × 5 × 15 × 5 = 5,625
- **Total: 7,725** doubles → exceeds 5,000 cap (35% discarded, **48% of αβ lost**)

Note: CAS(22,20) variant (11 occ, 9 virt) would have 13,761 doubles — even worse.

**Change**:
```python
def _generate_essential_configs(self):
    n_orb = self.hamiltonian.n_orbitals
    # BUG FIX: compute per-spin independently for open-shell
    n_occ_a, n_virt_a = self.hamiltonian.n_alpha, n_orb - self.hamiltonian.n_alpha
    n_occ_b, n_virt_b = self.hamiltonian.n_beta, n_orb - self.hamiltonian.n_beta

    # Estimate total doubles
    n_aa = comb(n_occ_a, 2) * comb(n_virt_a, 2)
    n_bb = comb(n_occ_b, 2) * comb(n_virt_b, 2)
    n_ab = n_occ_a * n_virt_a * n_occ_b * n_virt_b
    total_doubles = n_aa + n_bb + n_ab

    # Adaptive cap: allow all doubles up to 20K,
    # then importance-sample by |h2e| magnitude
    if total_doubles <= 20000:
        max_doubles = total_doubles  # Keep all
    else:
        max_doubles = 20000
        # FIX: allocate cap proportionally across spin channels
        # (not sequential enumeration that starves αβ)
        frac_aa = n_aa / total_doubles
        frac_bb = n_bb / total_doubles
        frac_ab = n_ab / total_doubles
        cap_aa = int(max_doubles * frac_aa)
        cap_bb = int(max_doubles * frac_bb)
        cap_ab = max_doubles - cap_aa - cap_bb
        # Within each channel, rank by |h2e[p,q,r,s]| and keep top-K
        use_importance_sampling = True
```

When importance sampling is needed (>20K potential doubles), rank by `|h2e[p,q,r,s]|` and keep the top-K most important ones **within each spin channel proportionally**. This fixes the ordering bias where αβ doubles were starved.

**Validation**: On N2/STO-3G (567 doubles, well under any cap), verify energy unchanged. On a 40Q mock system, verify all doubles are generated when under the 20K cap, and importance sampling activates when above.

**Acceptance criteria**:
- [ ] `max_doubles` scales with system size
- [ ] **Ordering bias fixed**: cap allocated proportionally across αα/ββ/αβ channels, not sequentially
- [ ] Importance sampling by `|h2e|` for systems exceeding 20K doubles
- [ ] Per-spin computation (not `max(n_alpha, n_beta)`) to support open-shell systems
- [ ] N2 results unchanged (was already under new cap)
- [ ] Test with 20-orbital mock system to verify importance sampling and proportional allocation
- [ ] **H5**: Same ordering bias exists in `PhysicsGuidedFlowTrainer._generate_essential_configs()` at `physics_guided_training.py:286-337` — fix both locations, not just `pipeline.py`

---

#### PR 1.4: Eliminate O(n^2) Hamming Distance Matrix

**File**: `src/postprocessing/diversity_selection.py` (lines 73-93, 357, 454)

**Current**: `compute_hamming_distance_matrix()` allocates full `(n, n, sites)` boolean tensor. In the main DPP selection path (`_dpp_select` at line 357), this is called per excitation-rank bucket — the largest bucket (doubles, 50% budget) could have ~25K configs, producing a (25K, 25K, 40) = **~25 GB** peak. The `analyze_basis_diversity()` diagnostic (line 454) calls it on the full basis and would hit **~100 GB** at 50K configs. Both are unacceptable.

**Change**: Replace DPP-greedy with **streaming greedy selection** that never materializes the full distance matrix. Also fix `analyze_basis_diversity()` (line 454) which calls the same O(n²) function on the full basis — either skip the diagnostic for large bases or use a sampled approximation:

```python
def select_diverse_streaming(configs, weights, max_configs, min_distance):
    """
    Greedy diversity selection without O(n^2) distance matrix.

    Algorithm:
    1. Sort configs by weight (descending)
    2. Initialize selected = [highest-weight config]
    3. For each remaining config in weight order:
       a. Compute Hamming distance to all selected configs (O(|selected|))
       b. If min distance >= threshold, add to selected
    4. Stop when max_configs reached

    Complexity: O(n * max_configs * sites) instead of O(n^2 * sites)
    For n=50K, max_configs=15K, sites=40: ~30B ops vs 100B ops
    But critically: memory O(max_configs * sites) vs O(n^2)
    """
```

**Validation**: On CH4 (15,876 configs), compare selected basis quality (overlap with FCI ground state) between old and new method. Must be within 5%.

**Acceptance criteria**:
- [ ] O(n^2) distance matrix eliminated
- [ ] Memory usage bounded by O(max_configs * n_sites)
- [ ] Basis quality maintained on existing benchmarks
- [ ] Works for 50K+ configs without OOM

---

### Phase 2: NF Unlock for Large Systems (Weeks 4-8)

**Goal**: Enable normalizing flow training on 40Q systems where Direct-CI alone is insufficient.

**Note**: Phase 2 expanded from 3 to 5 weeks due to addition of PRs 2.2b, 2.2c, and 2.2d. These are prerequisites for reliable 40Q operation, not optional. Recommended order: **2.1** (unlock NF) → **2.2** (scale architecture + fix dead code) → **2.2b** (entropy regularization + loss fixes) → **2.2d** (fix probability model) → **2.2c** (validate on molecules) → **2.3** (Numba). Rationale: PR 2.2c requires NF to be enabled (2.1), architecturally scaled (2.2), entropy-regularized (2.2b), and probability-corrected (2.2d) to produce meaningful validation results.

#### PR 2.1: Conditional NF Training

**File**: `src/pipeline.py` (line 217)

**Current**: `self.skip_nf_training = True` — forced for ALL molecular systems, regardless of size.

**Change**: NF training is unlocked for large systems where the configuration space is too large for Direct-CI to adequately cover:

```python
def adapt_to_system_size(self, n_valid_configs, verbose=True):
    if n_valid_configs <= 20000:
        # Small/medium: Direct-CI is sufficient
        self.skip_nf_training = True
    else:
        # Large: NF training is beneficial
        # User can still override with skip_nf_training=True
        if not hasattr(self, '_user_set_skip_nf'):
            self.skip_nf_training = False
            if verbose:
                print(f"NF training enabled: {n_valid_configs:,} configs "
                      f"exceeds Direct-CI threshold (20K)")
```

**Rationale**: Our RESULTS.md shows NF contribution is "modest for STO-3G systems" — because those systems have <=15,876 configs where Direct-CI already provides complete coverage. At 40Q:
- N2 CAS(10,20): 240M configs, essential configs (HF+S+D ≈ 7,876) cover **0.003%**
- [2Fe-2S] CAS(30,20): 240M configs, essential configs (HF+S+D ≈ 7,876) cover **0.003%**

The impact of Direct-CI's limited coverage depends on **correlation strength**:
- **N2 at equilibrium** (weakly correlated): |c_HF|² ≈ 0.92, CISD captures ~99% of wavefunction norm. Direct-CI may suffice for 2 mHa accuracy even at 40Q — NF provides marginal improvement.
- **N2 at stretched geometries** (strongly correlated): |c_HF|² ≈ 0.4, CISD captures ~80%. NF essential for discovering critical triples/quadruples.
- **Cr2, [2Fe-2S]** (strongly multireference): |c_HF|² ≈ 0.1-0.3. Direct-CI is catastrophically insufficient. NF is essential.

This means the NF's value-add is **molecule-dependent**, not a universal requirement at 40Q. The pipeline should attempt Direct-CI first and only invoke NF when Direct-CI energy does not converge. This can be detected by comparing energies with increasing basis size — if the energy plateaus with only S+D configs, NF is needed.

**Threshold choice**: The 20K threshold means systems like N2 CAS(10,10) (63K configs) will trigger NF training even though exact FCI is feasible via sparse eigsh. This is intentional: the NF training at 63K provides a validation stepping stone before jumping to 240M. At 63K, we can compare NF+SKQD vs exact FCI to measure the NF's actual value-add.

**Validation**:
- Systems <=20K configs: behavior unchanged (Direct-CI)
- Systems >20K configs: NF trains, produces configs beyond singles+doubles

---

#### PR 2.2: Scale NF Architecture

**File**: `src/flows/particle_conserving_flow.py`

**Current**: `OrbitalScoringNetwork` uses fixed `hidden_dims=[256, 256]`. `GumbelTopK` temperature defaults to 1.0 (has constructor parameter and `set_temperature()` setter, but is not learnable during training). At ~306K total parameters (~160K of which are the unused `beta_scorer` — see B16), the NF is too small for 240M-config spaces.

**Change**:
```python
class ParticleConservingFlowSampler:
    def __init__(self, num_sites, n_alpha, n_beta,
                 hidden_dims=None, **kwargs):
        n_orbitals = num_sites // 2  # num_sites = 2 * n_orbitals (alpha + beta spin-orbitals)
        if hidden_dims is None:
            # Scale with system size
            if n_orbitals <= 10:
                hidden_dims = [256, 256]
            elif n_orbitals <= 15:
                hidden_dims = [384, 384, 256]
            else:  # 16+ orbitals (32+ qubits)
                hidden_dims = [512, 512, 384, 256]
```

Additionally, add **learnable temperature** for GumbelTopK:
```python
class GumbelTopK(nn.Module):
    def __init__(self, initial_temperature=1.0, min_temperature=0.3):
        # Note: k is NOT a constructor param — it is passed per-call in forward(),
        # because the same GumbelTopK instance is used for both alpha (k=n_alpha)
        # and beta (k=n_beta) selections, which may differ for open-shell systems.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.min_temperature = min_temperature
```

**Rationale**: FermiNet/Psiformer research shows that NQS architectures need to scale with system complexity. The NNQS-Transformer paper (arXiv:2306.16705) demonstrates scaling to 120 spin orbitals with attention-based architectures. We start with wider MLPs; attention can be added in a future phase.

**Additional scope** (from Round 4 audit):

1. **B16 — Remove dead `beta_scorer`**: `self.beta_scorer = OrbitalScoringNetwork(...)` at line 186-188 is instantiated but **never called** in `sample()` or `log_prob()`. The actual beta scoring uses `self.beta_conditioned_scorer`. Remove the dead module to free ~160K wasted parameters (half the current model).

2. **B12 — Address non-autoregressive limitation**: Neither alpha nor beta channel uses autoregressive factoring — both predict all orbital logits simultaneously via a single MLP pass, then select top-k. This means the model **cannot capture intra-channel correlations** (e.g., "if orbital 3 is occupied, orbital 5 should also be occupied"). For weakly correlated systems (N2 at equilibrium) this may suffice, but for strongly correlated systems (Cr2, [2Fe-2S]) it is a fundamental limitation. Options:
   - *Minimum*: Document the limitation and plan autoregressive extension for Phase 4
   - *Better*: Add optional autoregressive mode where orbitals are selected one-by-one with updated context
   - *Best*: Use attention mechanism (transformer) over orbital embeddings for context-aware scoring

3. **H2 — GumbelTopK gradient masking**: The straight-through estimator at lines 70-75 computes `soft_topk = soft * one_hot`, zeroing gradients for non-selected positions. At n=20, k=5, only 25% of positions receive gradients. The 15 unselected orbitals get zero gradient despite potentially being near the selection boundary. Consider using the full softmax gradient without masking, or using relaxed top-k (entmax).

4. **M2 — Fix alpha-beta correlation waste**: `beta_conditioned_scorer` receives `[zeros, alpha_context]` where the first `n_orbitals` dimensions are always zero — wasting half the input bandwidth. Feed `alpha_context` directly with appropriate input dimension.

**Validation**: Train NF on N2/STO-3G with scaled architecture. Verify particle conservation holds. Verify sample diversity >= baseline.

**Acceptance criteria**:
- [ ] Architecture scales with n_orbitals
- [ ] Learnable temperature for GumbelTopK
- [ ] Particle conservation verified at 20 orbitals
- [ ] No regression on existing systems
- [ ] Dead `beta_scorer` module removed
- [ ] Non-autoregressive limitation documented with plan for autoregressive extension
- [ ] Alpha-beta correlation input fixed (no zero-padding waste)

---

#### PR 2.2b: Enable Entropy Regularization and Tune Loss Weights

**File**: `src/flows/physics_guided_training.py` (lines 62-63, 1028-1095)

**Current**: The NF training loss uses three components, but two are disabled by default:
- `physics_weight=0.0` — energy signal completely OFF
- `entropy_weight=0.0` — anti-collapse mechanism completely OFF
- Only the teacher loss (KL divergence from NQS to flow) is active, scaled by `|E|/|S|` (see PR 2.2d for why `|S|` is problematic)

Additionally, the temperature annealing schedule (lines 486-491) decays from 1.0→0.1 over 200 epochs, which **actively promotes mode collapse** by making sampling greedier. The `unique_ratio < 0.20` convergence criterion (line 530) treats mode collapse as "convergence" rather than a failure.

**Why this is critical**: At 240M configs, the NF must concentrate probability on 0.02% of the space. Without entropy regularization, there is **no mechanism preventing the NF from collapsing** onto a handful of high-probability configs. The existing essential config injection (line 709) partially compensates but doesn't fix the NF itself. Furthermore, the teacher loss creates a chicken-and-egg problem: the NF learns to match the NQS distribution, but the NQS is co-trained using NF samples — if either starts poorly, they reinforce each other's errors.

**Change**:
```python
@dataclass
class PhysicsGuidedConfig:
    # Enable physics loss for direct energy signal
    physics_weight: float = 0.1      # was 0.0
    # Enable entropy regularization for anti-collapse
    entropy_weight: float = 0.05     # was 0.0
    # Minimum unique_ratio before declaring convergence (raise threshold)
    min_unique_ratio: float = 0.35   # was 0.20
    # Temperature annealing: slower decay, higher floor
    initial_temperature: float = 1.0
    min_temperature: float = 0.3     # was 0.1
    temperature_decay_epochs: int = 400  # was 200
```

**Backward compatibility**: Changing `PhysicsGuidedConfig` defaults affects all users, including spin system training. Since `skip_nf_training=True` for all molecular systems today, the current defaults only affect spin systems. New defaults should be gated by system type: `entropy_weight=0.05` for molecular, keep `0.0` for spin systems (where the existing training already works). Alternatively, use a separate `MolecularTrainingConfig` subclass.

**Important clarification** (from Round 4 audit): The `_compute_subspace_energy()` at lines 775-791 is wrapped in `torch.no_grad()` — it provides **zero gradient** to the flow or NQS. It is purely a monitoring metric. The actual training gradient comes from local energies via REINFORCE in `_compute_nqs_loss()`. The `|E|` value used in loss scaling is this detached scalar — it does not backpropagate through the eigensolve. Enabling `physics_weight > 0` adds a direct energy signal via local energies, NOT via subspace eigensolve.

**Additional fix**: The factory function `create_physics_guided_trainer()` at line 1254 defaults to `teacher_weight=0.5, physics_weight=0.4, entropy_weight=0.1` — but `PhysicsGuidedConfig` defaults to `1.0, 0.0, 0.0`. The pipeline uses `PhysicsGuidedConfig` directly. Reconcile these by either updating `PhysicsGuidedConfig` defaults or removing the divergent factory function.

**Validation**: On N2/STO-3G (14,400 configs), compare NF sample diversity and energy convergence with entropy ON vs OFF. The NF should produce >50% unique configs per batch after 500 epochs, and energy should match Direct-CI baseline. Then test on a mock 20-orbital system.

**Acceptance criteria**:
- [ ] `entropy_weight > 0` by default for molecular systems with >20K configs
- [ ] `physics_weight > 0` provides direct energy gradient to the NF
- [ ] Temperature annealing uses higher floor (0.3) and slower decay
- [ ] `unique_ratio` convergence threshold raised to 0.35
- [ ] NF sample diversity measured and logged (distinct excitation ranks, Hamming diversity)
- [ ] No energy regression on existing benchmarks

---

#### PR 2.2c: Validate NF Training on Molecular Systems

**File**: `src/flows/physics_guided_training.py`, `tests/test_nf_molecular.py` (**New**)

**Current**: NF training for molecular systems has **never been tested**. `pipeline.py:217` forces `skip_nf_training=True` for ALL molecular systems. While the training code appears correct at the API level (handles molecular Hamiltonians via polymorphic `get_connections()`), the actual NF training loop has only been validated on spin systems.

**Change**: Before enabling NF at 40Q (PR 2.1), create a validation suite that proves NF training works on molecular systems at manageable scale:

```python
class TestNFMolecularTraining:
    """Validate NF training on molecular systems before scaling to 40Q."""

    def test_nf_lih_basic(self):
        """LiH (12Q, 225 configs): NF should match Direct-CI energy."""

    def test_nf_h2o_diversity(self):
        """H2O (14Q, 441 configs): NF should produce configs beyond S+D."""

    def test_nf_beh2_convergence(self):
        """BeH2 (14Q, 1225 configs): NF energy should converge within 200 epochs."""

    def test_nf_n2_sto3g_scaling(self):
        """N2/STO-3G (20Q, 14400 configs): Largest current system.
        NF must produce triples/quadruples that improve energy vs Direct-CI."""

    def test_nf_sample_quality_metrics(self):
        """Verify NF produces diverse excitation ranks, not just re-discovering S+D."""
```

**Acceptance criteria**:
- [ ] NF trains to convergence on LiH, H2O, BeH2, N2/STO-3G
- [ ] NF discovers higher excitations (triples+) that improve energy on N2
- [ ] Sample diversity metrics logged: excitation rank distribution, unique ratio
- [ ] No mode collapse (unique_ratio > 0.35 after 500 epochs)

---

#### PR 2.2d: Fix Probability Model and Loss Function

**Files**: `src/flows/particle_conserving_flow.py` (lines 274-295), `src/flows/physics_guided_training.py` (lines 1087, 775-791)

**Current**: Three compounding bugs make NF training structurally broken:

1. **`_topk_log_prob` uses product-of-marginals** (B14): Treats orbital selections as independent (`sum of log_softmax for selected items`), but top-k selection is *without replacement* — selecting orbital 3 removes it from the pool. The correct probability for an unordered subset requires computing the **permanent** of a softmax submatrix (#P-hard). The `log(k!)` correction only works when all softmax values are equal; for non-uniform values, per-config bias is 10-30% and **probability rankings can be inverted**, meaning the training loss optimizes toward the wrong distribution.

2. **`|E|/|S|` loss scaling punishes diversity** (B15): The loss is multiplied by `|E|/|S|` where `|S|` is the number of unique samples. A mode-collapsed flow producing 1 unique config gets 1000x more gradient than a diverse flow producing 1000 unique configs. This directly incentivizes collapse, working against entropy regularization (PR 2.2b).

3. **`_compute_subspace_energy` provides no gradient** (mischaracterized in PR 2.2b): The subspace energy computation at lines 775-791 is wrapped in `torch.no_grad()`. The returned energy is **detached from the computation graph** — it provides zero gradient to the flow or NQS. The actual training signal comes from local energies via REINFORCE (line 615). The subspace energy is purely a monitoring metric. The `|E|` in the loss scaling (item 2) uses this detached value as a scalar multiplier only.

**Change**:

```python
# Fix 1: Replace product-of-marginals with sequential conditional probability
def _topk_log_prob(self, logits, selection, k):
    """
    Compute log P(selection) using sequential conditional factorization.

    P({s1,...,sk}) = P(s1) * P(s2|s1) * ... * P(sk|s1,...,sk-1)

    At each step, compute softmax over REMAINING (unselected) orbitals,
    then accumulate log probability of the chosen orbital.
    """
    batch_size, n = logits.shape
    log_prob = torch.zeros(batch_size, device=logits.device)
    remaining_mask = torch.ones_like(logits, dtype=torch.bool)

    # Get indices of selected orbitals (sorted by logit magnitude for stability)
    selected_indices = selection.nonzero(as_tuple=False)  # (batch*k, 2)

    for step in range(k):
        # Mask already-selected orbitals with -inf
        masked_logits = logits.masked_fill(~remaining_mask, float('-inf'))
        log_probs_step = F.log_softmax(masked_logits, dim=-1)

        # Get the orbital selected at this step
        step_selection = ...  # k-th selected orbital per batch
        log_prob += log_probs_step.gather(1, step_selection.unsqueeze(1)).squeeze(1)

        # Remove selected orbital from remaining
        remaining_mask.scatter_(1, step_selection.unsqueeze(1), False)

    return log_prob

# Fix 2: Replace |E|/|S| with |E|/batch_size (fixed denominator)
# In _compute_flow_loss():
scale_factor = abs(energy) / batch_size  # was: abs(energy) / len(unique_configs)

# Fix 3: Document that subspace energy is diagnostic-only
# Add comment at _compute_subspace_energy():
# NOTE: This is wrapped in torch.no_grad() — it provides NO gradient
# to the flow or NQS. It is used only for monitoring and as a scalar
# in the |E| loss multiplier. The actual training gradient comes from
# local energies via REINFORCE in _compute_nqs_loss().
```

**Additional fix**: The NQS REINFORCE loss at line 1109-1121 computes `log_probs = 2 * log_amp`, which is `log(|ψ|²)`. Its derivative is `2 * d(log|ψ|)/dθ`, effectively **doubling the NQS learning rate** vs standard REINFORCE. For real-valued wavefunctions, this is equivalent to `nqs_lr *= 2`. Document this and consider halving `nqs_lr` for consistency.

**Also fix**: The `hard=False` path in `GumbelTopK.forward()` (line 78) applies `F.softmax(perturbed_logits / self.temperature)`, but `perturbed_logits` was already divided by temperature at line 61. Effective temperature = `temperature²`. Fix by removing the redundant division.

**Validation**: Compare probability estimates from old (product-of-marginals) vs new (sequential conditional) on N2/STO-3G configs. Verify sequential probabilities sum closer to 1.0 across sampled configs. Verify NF training convergence improves with fixed loss scaling.

**Acceptance criteria**:
- [ ] `_topk_log_prob` uses sequential conditional factorization (not product-of-marginals)
- [ ] Loss scaling uses `|E|/batch_size` (not `|E|/|S|`)
- [ ] `_compute_subspace_energy` clearly documented as gradient-free diagnostic
- [ ] NQS REINFORCE 2x factor documented (or `nqs_lr` halved)
- [ ] `hard=False` double-temperature bug fixed
- [ ] NF training converges on LiH/H2O without mode collapse

---

#### PR 2.3: Accelerate `get_connections()` with Numba

**File**: `src/hamiltonians/molecular.py` (lines 504-569)

**Current**: 4-nested Python for-loops for double excitation enumeration. Worst case at 40Q:

For N2/[2Fe-2S] at 40Q — worst case 15 occ, 5 virt per spin:
- Alpha-alpha: C(15,2) × C(5,2) = 1,050 iterations
- Beta-beta: 1,050 iterations
- Alpha-beta: 15 × 5 × 15 × 5 = 5,625 iterations
- **Per config: ~8,105 iterations** in pure Python (7,725 doubles + ~380 singles)

For 50K training samples: 50K × 8.1K = **405M Python loop iterations per epoch**. Each iteration involves numpy indexing, h2e lookups, config copying, and list appends — realistic cost is **500ns-2μs per iteration** (not just bare loop overhead). Estimated time: **200-400 seconds per epoch**. For 500+ epoch training, this is **28-56 hours** of pure connection computation. Unacceptable.

**Change**: Rewrite inner loops with `@numba.njit`:

```python
import numba

@numba.njit(cache=True)
def _get_double_connections_numba(
    config_np,          # (n_sites,) int8
    occ_alpha,          # (n_occ_a,) int
    occ_beta,           # (n_occ_b,) int
    virt_alpha,         # (n_virt_a,) int
    virt_beta,          # (n_virt_b,) int
    h2e_aa, h2e_bb, h2e_ab,  # integral slices
    n_orb,
):
    """
    Numba-accelerated double excitation enumeration.
    Returns arrays of (connected_config, matrix_element) pairs.

    Expected speedup: 50-100x over pure Python.
    """
```

Numba `@njit` compiles Python loops to LLVM machine code, achieving C-level performance. Literature reports **50-120x speedups** on nested numerical loops (numba.pydata.org benchmarks).

**Existing partial solution**: `src/hamiltonians/molecular.py:769-996` already contains `get_connections_vectorized_batch()`, a NumPy-vectorized implementation that is 10-50x faster than the Python loops. Before implementing Numba, benchmark this existing code at 40Q to determine if it already meets performance targets. Numba may only be needed if vectorized batch is insufficient.

**Alternative considered**: Cython or Rust (via PyO3). Rejected because:
- Cython adds build complexity (setup.py, .pyx files)
- Rust/PyO3 adds a language barrier and build toolchain (though IBM's qiskit-addon-sqd uses this approach for production)
- Numba requires zero build changes — just `@njit` decorator

**Validation**: Same outputs as Python loops on all 7 benchmark molecules. Timing comparison.

**Acceptance criteria**:
- [ ] `get_connections()` uses Numba-compiled inner loops
- [ ] Identical matrix elements vs Python reference (bitwise)
- [ ] >20x speedup demonstrated on N2
- [ ] Fallback to Python if Numba unavailable
- [ ] `numba` added as optional dependency in pyproject.toml

---

### Phase 3: New Molecular Systems (Weeks 9-14)

**Goal**: Add 3 target molecules at 24-40 qubits, validating the infrastructure from Phases 1-2.

**Note**: Phase 3 expanded from 4 to 6 weeks. CASSCF/DMRG-SCF orbital optimization for Cr2 and [2Fe-2S] is research-grade work with convergence challenges (see orbital optimization caveats below). N2/cc-pVDZ (PR 3.1) should be attempted first as the bridge system. If Cr2 or [2Fe-2S] orbital optimization proves intractable, use published integrals from the literature.

#### PR 3.1: N2/cc-pVDZ — The Bridge System (28-40Q)

**Rationale**: N2 is already validated at 20Q (STO-3G). Upgrading to cc-pVDZ basis increases orbitals from 10 to 28 (56 qubits full) or 10-20 in active space (20-40 qubits). This provides a smooth scaling path with known reference energies.

```python
def create_n2_ccpvdz_hamiltonian(
    bond_length: float = 1.10,
    active_space: Tuple[int, int] = (10, 14),  # (electrons, orbitals) = 28Q
) -> MolecularHamiltonian:
    """
    N2 with cc-pVDZ basis and CASSCF-optimized active space.

    Active space options:
    - (6,6) = 12Q: Minimal valence (pi/pi* + sigma/sigma*)
    - (10,10) = 20Q: 10 active electrons in 10 orbitals (63,504 configs; NOT the same as N2/STO-3G which uses all 14 electrons → 14,400 configs)
    - (10,14) = 28Q: Extended with extra correlating orbitals
    - (10,20) = 40Q: Large active space target

    Reference: DMRG with moderate bond dimensions achieves sub-mHa
    accuracy for N2/cc-pVDZ (Chan et al., ARPC 2011). For Cr2 DMRG/SHCI
    benchmarks, see Larsson et al., JACS 2022.
    """
    mol = gto.Mole()
    mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, bond_length))]
    mol.basis = "cc-pvdz"
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    # CASSCF orbital optimization
    n_elec, n_orb = active_space
    mc = mcscf.CASSCF(mf, n_orb, n_elec)
    mc.kernel()

    # Extract active space integrals
    h1e, e_core = mc.get_h1cas()
    h2e = mc.get_h2cas()
    ...
```

**Config space sizes** (determinants = C(n_orb, n_alpha) × C(n_orb, n_beta)):
| Active Space | Qubits | n_alpha, n_beta | Configs | Tractability |
|-------------|--------|-----------------|---------|-------------|
| (6,6) | 12 | 3α, 3β | C(6,3)² = 400 | Exact FCI |
| (10,10) | 20 | 5α, 5β | C(10,5)² = 63,504 | Exact FCI (feasible) |
| (10,14) | 28 | 5α, 5β | C(14,5)² = 4,008,004 | Sampling required |
| (10,20) | 40 | 5α, 5β | C(20,5)² = 240,374,016 | Heavy sampling + NF essential |

**Note**: N2/STO-3G (our validated system) uses all 14 electrons in 10 orbitals (7α, 7β), giving C(10,7)² = 14,400 configs. The cc-pVDZ active spaces above freeze 4 core electrons, leaving 10 active.

**Validation**: Compare ground-state energies with PySCF CASCI/DMRG at each active space size. Pipeline energy must be within 1 mHa of CASCI reference.

---

#### PR 3.2: Cr2 (Chromium Dimer) — The Classic Hard Problem (24-48Q)

**Rationale**: Cr2 is one of the "Top 20 molecules for quantum computing" (PennyLane/Xanadu), called a "famous problem" that "closed a chapter of quantum chemistry" (JACS, 2022). The minimal (12e, 12o) active space requires 24 qubits. Expanding to (12e, 24o) reaches 48 qubits.

Reference data: DMRG with bond dimension 28,000 + SHCI provide near-exact benchmarks. Spectroscopic constants: Re=1.68 A, D0=1.53 eV.

```python
def create_cr2_hamiltonian(
    bond_length: float = 1.68,
    active_space: Tuple[int, int] = (12, 12),  # CAS(12,12) = 24Q
) -> MolecularHamiltonian:
    """
    Chromium dimer with CASSCF active space.

    The Cr2 ground state is notoriously difficult due to strong
    static correlation from 12 active electrons in 3d+4s orbitals.

    Active space options (6α, 6β):
    - (12,12) = 24Q: Minimal 3d+4s valence, C(12,6)² = 853,776 configs
    - (12,18) = 36Q: Extended with 4p shell, C(18,6)² = 344,622,096 configs
    - (12,24) = 48Q: Including 4d double-shell, C(24,6)² = 18,116,083,216 configs

    Reference: Larsson et al., JACS 2022 (DMRG/SHCI near-exact)
    """
```

**Orbital optimization caveat**: Cr2 CASSCF is notoriously difficult to converge. The 3d orbital near-degeneracy causes multiple local minima. Recommended approach:
- Use cc-pVDZ-DK basis with scalar relativistic effects (Douglas-Kroll-Hess)
- Start from high-spin (S=6) UHF reference, then rotate orbitals to singlet
- Consider AVAS (Atomic Valence Active Space) for automated orbital selection
- If CASSCF fails to converge, use DMRG-SCF (`pyscf.dmrgscf`) as orbital optimizer

---

#### PR 3.3: [2Fe-2S] Iron-Sulfur Cluster — The Biological Target (40-45Q)

**Rationale**: IBM demonstrated SQD on methyl-capped [2Fe-2S] with (30e, 20o) active space at 45 qubits on a Heron processor (Science Advances, 2025). This is the ideal molecule for our 40Q target:

- Biologically critical (electron transfer chains, nitrogenase)
- Strongly correlated (antiferromagnetic coupling between Fe centers)
- IBM provides direct quantum hardware comparison data
- Well-studied with DMRG/CASSCF in PySCF (JCTC, 2021)

```python
def create_fe2s2_hamiltonian(
    active_space: Tuple[int, int] = (30, 20),  # 40Q — matches IBM
    geometry: str = "model",  # "model" or "biological"
) -> MolecularHamiltonian:
    """
    [2Fe-2S] iron-sulfur cluster.

    Active space options:
    - (12,10) = 20Q: Fe 3d orbitals only, C(10,6)² = 44,100 configs
    - (20,20) = 40Q: Fe 3d magnetic orbitals (GUGA-FCIQMC, JCTC 2021),
                      C(20,10)² = 184,756² = 34.1B configs
    - (30,20) = 40Q: IBM's choice (Fe 3d + S 3p + bonding),
                      C(20,15)² = 15,504² = 240,374,016 configs ← DEFAULT

    We use CAS(30,20) as the default because:
    1. Direct comparison with IBM's Science Advances 2025 results
    2. Config space (240M) is large but identical to N2 CAS(10,20)
    3. More electrons → more doubly-occupied orbitals → smaller effective space
    4. CAS(20,20) has 34B configs, making it much harder without benefit

    Reference: IBM Science Advances (2025), 45Q SQD experiment
    """
```

**Orbital optimization caveat**: [2Fe-2S] has an antiferromagnetic singlet ground state that cannot be described by a single RHF determinant. Recommended approach:
- Use broken-symmetry UHF (BS-UHF) as initial guess — localize alpha spin on Fe1, beta on Fe2
- Apply AVAS with Fe 3d + S 3p atomic orbitals for active space selection
- For CAS(30,20), DMRG-SCF orbital optimization is strongly recommended over conventional CASSCF
- IBM's published integrals (if available) can bypass orbital optimization entirely

**Validation**: Compare with IBM's published SQD energies for CAS(30,20). For the (12,10) active space, exact CASCI should be achievable.

---

### Phase 4: Advanced Methods (Weeks 15-22)

**Goal**: Add capabilities that unlock 50+ qubit scaling for future work.

#### PR 4.1: DMET Fragment Decomposition

**Rationale**: DMET (Density Matrix Embedding Theory) is how IBM scaled SQD from 41 to 89 qubits for cyclohexane (JCTC, 2025). It decomposes a large molecule into fragments, each solvable independently at ~30Q.

**Implementation**: Integrate with existing PySCF DMET via `libdmet` or `pDMET`:

```python
class DMETFragmentSolver:
    """
    DMET wrapper that uses FlowGuidedKrylovPipeline as the impurity solver.

    Algorithm:
    1. Run mean-field (RHF/UHF) on full molecule
    2. Localize orbitals (Boys/PM)
    3. Define fragments (atom-centered)
    4. For each fragment:
       a. Schmidt decomposition → impurity + bath
       b. Build fragment Hamiltonian
       c. Solve with FlowGuidedKrylovPipeline
       d. Compute 1-RDM
    5. Update mean-field potential
    6. Repeat until convergence
    """
```

**Dependencies**: `libdmet` (requires manual installation with Fortran compilation; ARM64/DGX Spark compatibility unverified), PySCF DMET module

**Validation**: H-ring (18 atoms) as in IBM paper. Compare total energy with full-system calculation.

---

#### PR 4.2: SQD S-CORE Acceleration

**File**: `src/krylov/sqd.py`

Upgrade the S-CORE configuration recovery and S² computation to handle 40Q:
- Vectorized spin-sector fixing (currently Python loops for n>2000)
- **B11 (CRITICAL)**: GPU-batched S² matrix computation. Currently `_compute_s2_matrix()` at `sqd.py:826-840` falls back to an O(n²) **serial Python double loop** for n>2000. At 40Q batch=33K, this is 544M iterations — estimated **hours** of compute. Must be vectorized for all batch sizes, not just n<=2000.
- Consider interfacing with IBM's `qiskit-addon-sqd` (v0.12.0, Rust backend) or `qiskit-addon-sqd-hpc` (C++17/20) for the inner loop

**Acceptance criteria**:
- [ ] S² matrix computation vectorized for n>2000 (eliminate Python double loop)
- [ ] S-CORE recovery handles 40Q batch sizes within 60 seconds
- [ ] Energy within 1 mHa of noiseless on N2 with `noise_rate > 0`

---

#### PR 4.3: SqDRIFT (Randomized Krylov)

**Rationale**: IBM's SqDRIFT (arXiv:2508.02578) combines SKQD with qDRIFT randomized Hamiltonian compilation, providing provable convergence guarantees. Demonstrated on polycyclic aromatic hydrocarbons (naphthalene, coronene) up to 48 qubits on IBM Heron processors. This addresses the sampling inefficiency critique from arXiv:2501.07231.

This is a natural extension of our existing SKQD solver:

```python
class SqDRIFTSolver(SampleBasedKrylovDiagonalization):
    """
    Randomized SKQD with stochastic time evolution.

    Instead of exact e^{-iHt}, uses random Pauli term sampling:
    At each step, randomly select a Pauli term P_j with probability
    proportional to |h_j|, and apply e^{-i * sign(h_j) * dt * P_j}.

    Advantages:
    - Lower per-step cost (single Pauli rotation vs full evolution)
    - Provable convergence to ground state
    - Naturally explores diverse configurations
    """
```

**Acceptance criteria**:
- [ ] SqDRIFT solver converges on N2/STO-3G within 2x wall time of standard SKQD
- [ ] Ground-state energy matches SKQD within 0.1 mHa
- [ ] Demonstrated advantage on 40Q system (better config diversity per unit time)

---

## 5. Target Molecule Selection Rationale

### 5.1 Selection Criteria

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Existing validation path | High | Must have smaller active space where exact FCI is possible |
| Reference data available | High | Need DMRG/CASCI/FCI benchmarks to validate against |
| Chemical importance | Medium | Motivates the work scientifically |
| IBM/literature comparison | Medium | Allows cross-validation with quantum hardware results |
| PySCF support | High | Must be implementable with current tool stack |
| Strong correlation | High | Tests NF's value (weak correlation → Direct-CI sufficient) |

### 5.2 Selected Systems

| Priority | Molecule | Target Active Space | Qubits | Config Space | Key Reference |
|----------|----------|-------------------|--------|-------------|--------------|
| **P0 (bridge)** | N2/cc-pVDZ | (10,14) → (10,20) | 28 → 40 | 4M → 240M | DMRG sub-mHa accuracy (Chan ARPC 2011; Larsson JACS 2022) |
| **P1 (classic)** | Cr2 | (12,12) → (12,24) | 24 → 48 | 854K → 18.1B | DMRG/SHCI near-exact (JACS 2022) |
| **P2 (bio)** | [2Fe-2S] | (30,20) | 40 | 240M | IBM SQD 45Q experiment (Science Advances 2025) |

### 5.3 Why NOT Other Molecules

| Candidate | Qubits | Reason for Exclusion |
|-----------|--------|---------------------|
| FeMoco | 108-152 | Far beyond 40Q; needs DMET (Phase 4) |
| U2 (uranium dimer) | ~52 | Requires relativistic treatment (spin-orbit); PySCF support limited |
| Pentacene | 44 | Weakly correlated; Direct-CI would suffice |
| Benzene | 30 | Already solved classically; not challenging enough |
| C2H4/ethylene | 28 | Already in examples/ (moderate_system_benchmark.py) |

---

## 6. Testing & Validation Strategy

### 6.1 Regression Gate

Every PR must pass the **Regression Gate**: all 7 molecules from RESULTS.md (H2, LiH, H2O, BeH2, NH3, CH4, N2) must produce energies within **0.01 mHa** of the recorded values for experiments 2-4 (SKQD variants).

```bash
# Regression test command (to be automated)
uv run pytest tests/test_regression.py -v --tb=short
```

### 6.2 Scaling Validation Ladder

| Step | System | Qubits | Configs | Method | Expected Outcome |
|------|--------|--------|---------|--------|-----------------|
| S1 | N2/STO-3G (14e,10o) | 20 | 14,400 | Exact FCI | Match RESULTS.md |
| S2 | N2/cc-pVDZ CAS(6,6) | 12 | 400 | Exact FCI | Match PySCF CASCI |
| S3 | N2/cc-pVDZ CAS(10,10) | 20 | 63,504 | Sparse eigsh (PR 1.1 required; exceeds max_diag_basis_size=15K) | Match PySCF CASCI |
| S4 | Cr2 CAS(12,12) | 24 | 853,776 | NF or importance-sampled Direct-CI → 15-50K subset → sparse eigsh | Within 1 mHa of DMRG |
| S5 | N2/cc-pVDZ CAS(10,14) | 28 | 4,008,004 | NF + SKQD | Within 1 mHa of DMRG |
| S6 | N2/cc-pVDZ CAS(10,20) | 40 | 240,374,016 | NF + SKQD | Within 2 mHa of DMRG |
| S7 | [2Fe-2S] CAS(30,20) | 40 | 240,374,016 | NF + SKQD | Within 5 mHa of IBM SQD |

**Key transitions**: S3→S4 jumps from exact-FCI-feasible (~64K) to sampling-required (~854K). S5→S6 jumps to 240M configs where NF becomes absolutely essential. S7 has the same config-space size as S6 (both 240M) but is chemically harder due to strong antiferromagnetic coupling between Fe centers.

**Integration test** (M13): After S6 or S7, run a full end-to-end pipeline test on a 40Q system: `PipelineConfig(subspace_mode="skqd", skip_nf_training=False)` → NF training → basis extraction → sparse diag → energy. This validates all phases working together, not just individual PRs.

### 6.3 Performance Benchmarks

| Metric | 20Q Baseline | 40Q Target | Notes |
|--------|-------------|------------|-------|
| Time per `get_connections()` call | ~0.5 ms | <5 ms (with Numba) | 20 orbitals: ~8.1K iterations/config (7.7K doubles + 0.4K singles) |
| Eigensolver for 15K basis | ~30s (dense) | <10s (sparse) | Sparse CSR (avg ~100-1K nnz/row, 0.1-6% dense) + iterative eigsh |
| Diversity selection for 50K | OOM | <30s, <2GB RAM | Streaming greedy, no O(n²) matrix |
| NF training (500 epochs) | N/A | <90 min | **Requires PR 2.3 (Numba)**: without it, connections alone take 28-56 hours. With Numba (50x): ~60 min for connections + ~30 min training overhead. Original 60-min target was optimistic. |
| Full pipeline (SKQD) | ~2 min | <120 min | Including NF training |
| Peak memory | ~2 GB | <25 GB | Dominated by H_proj if dense |

**Note**: The 40Q wall-time target is increased from the original 30min to 120min due to the corrected config space sizes (~240M for both N2 CAS(10,20) and [2Fe-2S] CAS(30,20)) requiring NF training and more extensive sampling.

---

## 7. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sparse eigensolver convergence issues for ill-conditioned H | Medium | High | Shift-invert mode for eigsh; LOBPCG as fallback; eigendecomposition-based regularization with mode filtering (exists as `_svd_ground_state` at skqd.py:827, misnomer — actually uses `eigh`) |
| NF fails to learn at 20 orbitals (mode collapse) | **High** | High | No published NF-NQS results on molecules >20Q. Currently **no anti-collapse mechanism**: `entropy_weight=0.0`, temperature annealing promotes collapse. PR 2.2b enables entropy regularization (critical). Also: learnable temperature, NNQS-SCI fallback, Direct-CI + importance sampling |
| NF-NQS co-training chicken-and-egg | **High** | Medium | NF learns to match NQS, NQS trains on NF samples — if either starts poorly, they reinforce errors. Mitigate: pre-train NQS on Direct-CI configs before NF training; enable `physics_weight > 0` for direct energy signal |
| [2Fe-2S] CASSCF orbital optimization fails | **Medium** | High | Use IBM's published orbitals; BS-UHF initial guess; DMRG-SCF orbital optimizer; AVAS for active space selection |
| Cr2 CASSCF converges to wrong orbital solution | Medium | Medium | High-spin UHF initial guess; AVAS orbital selection; DMRG-SCF; DK relativistic corrections |
| GumbelTopK probability approximation bias at n=20 | Medium | Medium | `estimate_discrete_prob()` uses product-of-marginals, ignoring selection correlations. At (20,5), bias ~10-30% per config. Affects teacher loss convergence. Mitigate: verify training still converges; consider permanent-based estimator if needed |
| GPU memory fragmentation on UMA (torch/scipy/cupy interop) | Low | Medium | Three memory allocators on 128GB shared DRAM. Mitigate: minimize framework crossings; prefer torch-native sparse when available |
| Numba compilation fails on ARM64 (DGX Spark) | Low | Low | Fallback to `get_connections_vectorized_batch()` (10-50x faster, already exists); consider Cython |
| RESULTS.md regressions during refactoring | Medium | High | Automated regression gate; CI/CD (future) |
| SQD "fatal flaw" affects our SQD solver at 40Q | Medium | Low | SKQD is our primary solver (193x better); SQD is secondary |
| NF trains successfully but produces useless configs | **High** | **High** | NF converges (no mode collapse) but discovered triples/quadruples don't improve energy beyond Direct-CI. Would invalidate Phase 2 premise. Mitigate: compare NF+SKQD vs Direct-CI+SKQD on N2 CAS(10,10) first; fallback to importance-sampled CISDT as classical alternative |
| NQS scaling laws may not apply to quantum chemistry | Medium | Medium | arXiv:2509.26397 argues NQS scaling does NOT follow favorable LLM-like laws. If NF-NQS accuracy plateaus with system size, the 40Q target may require fundamentally different approaches. Mitigate: empirical validation at each scaling step (S1-S7) |
| NF non-autoregressive architecture insufficient for strong correlation | **High** | **High** | Neither spin channel uses autoregressive factoring (B12). Cannot capture intra-channel orbital correlations critical for Cr2/[2Fe-2S]. Mitigate: PR 2.2 documents limitation; plan autoregressive extension; Direct-CI + importance sampling as fallback |
| S6 wall time (2h) has hard dependency on PR 2.3 (Numba) | Medium | Medium | Without Numba, connection computation alone takes 28-56 hours. If Numba fails on ARM64, must fall back to `get_connections_vectorized_batch()` (10-50x faster but not 50-100x) |

---

## 8. Dependencies & Prerequisites

### 8.1 New Dependencies

| Package | Version | Purpose | Optional? |
|---------|---------|---------|----------|
| `numba` | >=0.59 | JIT compilation for `get_connections()` | Yes (fallback to Python) |
| `block2` | latest | DMRG-SCF orbital optimization for Cr2/[2Fe-2S] (Phase 3) | Yes (Phase 3 only; requires Fortran/MPI; ARM64 compatibility unverified) |
| `libdmet` | latest | DMET fragment decomposition (Phase 4) | Yes (Phase 4 only; requires Fortran compilation; ARM64 compatibility unverified) |

### 8.2 Existing Dependencies (No Changes)

- `torch>=2.0`: Core framework (already supports sparse tensors)
- `pyscf>=2.3`: Molecular integrals + CASSCF (already supports active spaces)
- `scipy`: Sparse eigensolvers (`eigsh`, `lobpcg` already included)
- `cupy` (optional): GPU sparse eigsh (already integrated)

---

## 9. File Change Summary

| File | Phase | Changes |
|------|-------|---------|
| `.gitignore` | 0.1 | Remove `tests/` from gitignore |
| `.github/workflows/regression.yml` | 0.2 | **New**: Minimal CI/CD for regression tests |
| `tests/conftest.py` | 0.1 | **New**: Shared test fixtures |
| `tests/test_regression.py` | 0.1 | **New**: Automated regression gate |
| `tests/test_nf_molecular.py` | 2.2c | **New**: NF molecular training validation |
| `tests/test_scaling.py` | 3.x | **New**: Scaling ladder validation |
| `src/utils/gpu_linalg.py` | 1.1 | Add `sparse_hamiltonian_eigsh()`, eigsh/LOBPCG tier logic |
| `src/krylov/skqd.py` | 1.1, 1.2 | Use sparse solver; importance-ranked truncation; add MAX_FULL_SUBSPACE guard to standard SKQD; sparse H for Krylov steps; scale max_diag_basis_size |
| `src/pipeline.py` | 1.3, 2.1, 2.2 | Adaptive doubles cap + ordering fix (both locations); conditional NF; new adapt tiers |
| `src/flows/physics_guided_training.py` | 1.3, 2.2b, 2.2d | Fix doubles bias (second copy); enable entropy/physics loss weights; fix temperature annealing; fix `\|E\|/\|S\|` scaling; document subspace energy no-gradient; reconcile factory defaults; fix accumulated energy dense→sparse |
| `src/postprocessing/diversity_selection.py` | 1.4 | Streaming greedy selection, eliminate O(n^2) matrix |
| `src/hamiltonians/molecular.py` | 2.3, 3.x | Numba-accelerated loops; new factory functions |
| `src/flows/particle_conserving_flow.py` | 2.2, 2.2d | Scaled architecture; learnable temperature; remove dead beta_scorer; fix alpha-beta input; fix _topk_log_prob; fix GumbelTopK double-temp bug; document non-autoregressive limitation |
| `src/krylov/sqd.py` | 4.2 | GPU-batched S^2 for all n (eliminate Python loop); S-CORE acceleration |
| `pyproject.toml` | 2.3, 3.x | Add `numba`, `block2` to optional dependencies |
| `docs/ADR-001-scale-to-40-qubits.md` | N/A | This document |

---

## 10. Success Criteria

This ADR is **accepted** when:

1. **N2/cc-pVDZ at CAS(10,20) = 40 qubits** converges to within 2 mHa of DMRG reference
2. **Cr2 at CAS(12,12) = 24 qubits** converges to within 1 mHa of published DMRG/SHCI values
3. **[2Fe-2S] at CAS(30,20) = 40 qubits** produces energy comparable to IBM's published SQD result (within 5 mHa). Note: IBM's energy was obtained on quantum hardware with error mitigation; meaningful comparison requires running the same integrals through classical CASCI/DMRG. If IBM's integrals are not publicly available, compare against our own PySCF CASCI/DMRG reference.
4. **No regressions** on existing 7-molecule benchmark suite
5. **Peak memory** stays within 25 GB for all 40Q calculations (fits DGX Spark 128GB UMA)
6. **Wall time** for full pipeline on 40Q system is under 2 hours on DGX Spark (including NF training)
7. **NF sample quality** (new): At 40Q, the NF produces configs spanning multiple excitation ranks (not just rediscovering singles+doubles). Specifically: >10% of NF-discovered configs should be triples or higher, and these should improve the ground-state energy vs Direct-CI alone by **at least 0.5 mHa**. Without this criterion, a mode-collapsed NF that produces only S+D configs would pass criteria 1-6 while the NF is effectively useless.
8. **NF molecular validation** (new): NF training converges on N2/STO-3G (20Q) with `skip_nf_training=False`, producing energy within 0.5 mHa of Direct-CI baseline (relaxed from 0.1 mHa — NF is a stochastic sampler; achieving 0.1 mHa requires essentially matching FCI, which is unrealistic for the NF alone). This must be demonstrated before attempting 40Q.
9. **NF probability model correctness** (new): Sequential conditional `_topk_log_prob` (PR 2.2d) produces probability estimates that are within 5% of exact enumeration on LiH (225 configs, small enough for exact verification). The old product-of-marginals model must be replaced.

---

## 11. References

### Core Algorithm Papers
- [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization (arXiv:2501.09702)](https://arxiv.org/abs/2501.09702)
- [Chemistry Beyond Exact Diagonalization (Science Advances, 2025)](https://www.science.org/doi/10.1126/sciadv.adu9991)
- [KQD on 56 Qubits (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-59716-z)
- [SqDRIFT: Provable Convergence via Randomized SKQD (arXiv:2508.02578)](https://arxiv.org/abs/2508.02578)
- [NF-Assisted NQS (arXiv:2506.12128)](https://arxiv.org/abs/2506.12128)

### Molecule References
- [Cr2: Closing a Chapter of Quantum Chemistry (JACS, 2022)](https://pubs.acs.org/doi/10.1021/jacs.2c06357)
- [Iron-Sulfur Clusters via GUGA-FCIQMC (JCTC, 2021)](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00589)
- [Top 20 Molecules for Quantum Computing (PennyLane, 2024)](https://pennylane.ai/blog/2024/01/top-20-molecules-for-quantum-computing)
- [Chemically Decisive Benchmarks (arXiv:2601.10813)](https://arxiv.org/abs/2601.10813)
- [25-100 Logical Qubits Perspective (arXiv:2506.19337)](https://arxiv.org/abs/2506.19337)

### Scaling Methods
- [DMET-SQD for Extended Molecules (JCTC, 2025)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00114)
- [Quantum Bootstrap Embedding SQD (Digital Discovery, 2026)](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00416k)
- [NNQS-Transformer: Scalable NQS (arXiv:2306.16705)](https://arxiv.org/abs/2306.16705)
- [LOBPCG vs Davidson (Theor. Chem. Acc. 142, 2023)](https://link.springer.com/article/10.1007/s00214-023-03010-y)

### Recent Developments (2025-2026)
- [Implicit Solvent SQD, 27-52 Qubits (J. Phys. Chem. B, 2025)](https://pubs.acs.org/doi/10.1021/acs.jpcb.5c01030)
- [ph-AFQMC + SQD Trial Wavefunctions (JCTC, 2025)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01407)
- [Lockheed Martin: 52-Qubit Open-Shell SQD (JCTC, 2025)](https://www.ibm.com/quantum/blog/lockheed-martin-sqd)
- [NNQS-SCI: Trillion-Dimensional Hilbert Spaces (SC'25)](https://dl.acm.org/doi/10.1145/3712285.3759800)
- [Deterministic NQS for Cr2 Dissociation (arXiv:2601.21310)](https://arxiv.org/abs/2601.21310)
- [LLM Scaling Laws for NQS in Quantum Chemistry (arXiv:2509.12679)](https://arxiv.org/abs/2509.12679)
- [PIGen-SQD: Physics-Informed Generative ML for SQD (arXiv:2512.06858)](https://arxiv.org/abs/2512.06858)
- [DMET-SQD for Ligand-like Molecules (arXiv:2511.22158)](https://arxiv.org/abs/2511.22158)
- [NF for Electronic Schrödinger Equation with DPP (arXiv:2406.00047)](https://arxiv.org/abs/2406.00047)
- [Are Neural Scaling Laws Leading Quantum Chemistry Astray? (arXiv:2509.26397)](https://arxiv.org/abs/2509.26397)
- [Gumbel-Top-k: Stochastic Beams and Where to Find Them (ICML 2019)](https://proceedings.mlr.press/v97/kool19a.html)

### Criticisms & Responses
- [Critical Limitations in QSCI Methods (arXiv:2501.07231)](https://arxiv.org/abs/2501.07231)
- [SQD Convergence on Cuprate Chains (arXiv:2512.04962)](https://arxiv.org/abs/2512.04962)

### Tools & Libraries
- [qiskit-addon-sqd (IBM, Rust-accelerated)](https://qiskit.github.io/qiskit-addon-sqd/)
- [PySCF CASSCF Documentation](https://pyscf.org/quickstart.html)
- [CuPy Sparse Eigensolver](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html)
- [NVIDIA CUDA-Q SKQD Tutorial](https://nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html)
- [libdmet (PySCF DMET)](https://github.com/gkclab/libdmet_preview)
- [QC-DMET (Python DMET)](https://github.com/sebwouters/qc-dmet)
