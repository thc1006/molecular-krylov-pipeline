# Flow-Guided Krylov Pipeline

## Project Overview

Quantum chemistry pipeline for computing molecular ground-state energies. Combines **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** with **Krylov subspace diagonalization** to systematically converge toward FCI-level accuracy.

The core idea: use a normalizing flow to discover high-probability molecular configurations (determinants), then diagonalize the Hamiltonian projected onto that basis. The primary subspace solver is **SKQD** (Krylov time evolution). **SQD** (IBM's sampling-based batch diagonalization) is also available but performs poorly on systems beyond 16 qubits.

### Scale-up Goal

The pipeline's target is scaling to **40+ qubits**. Two parallel challenges:
1. **SKQD (core solver)**: Must handle 40Q-scale subspaces — sparse eigensolve, memory management, Krylov expansion efficiency
2. **NF (auxiliary sampler)**: Autoregressive Transformer implemented (Phase 4a), but deep research (60+ papers) found NF provides **NO benefit** over Direct-CI + Krylov at 40Q after SKQD bug fix. Three root causes identified: (1) REINFORCE is wrong optimizer, (2) wrong sign factorization, (3) no H-coupling guidance. See `docs/TODO-PHASE-6.md`.

### Current Status (as of 2026-03-09)

- **SKQD**: 7/7 STO-3G PASS + CAS(10,12) 24Q + CAS(10,15) 30Q + CAS(10,20) 40Q tested. Sparse eigensolver, OOM guards, Numba JIT, adaptive dt, Lanczos full reorthogonalization.
- **NF**: Autoregressive Transformer implemented (Phase 4a). KV-cached decoding. Non-AR flow still available. But deep research (60+ papers) found NF provides NO benefit over Direct-CI + Krylov at 40Q after SKQD bug fix. Root causes: (1) REINFORCE wrong optimizer, (2) wrong sign factorization (psi=sqrt(p)*s(x)), (3) no H-coupling guidance.
- **VMC**: Implemented (Phase 4b) but doesn't converge at 24Q+. Uses REINFORCE (wrong -- field uses SR/MinSR/SPRING). VMC default OFF. NQS-SC > NQS-VMC validated by ETH Zurich (arXiv:2602.12993).
- **Sign Network**: Phase 4c implemented but architecture is wrong (feedforward->tanh). SOTA uses determinant-based or continuous phase (e^{i*phi}).
- **SQD**: Still fails >16Q. Not the focus.
- **Basis sets**: STO-3G + cc-pVDZ CAS support. CAS(10,20) 40Q tested.
- **Tests**: 587 tests all pass.
- **Phase 6 TODO**: `docs/TODO-PHASE-6.md` -- MinSR, H-coupling filtering, iterative refinement.

### Key References (PDFs in `papers/`)

- "Improved Ground State Estimation via Normalising Flow-Assisted Neural Quantum States"
- "Sample-based Krylov Quantum Diagonalization"
- "Chemistry Beyond the Scale of Exact Diagonalization" (IBM SQD paper)

### Repository Notes

- **`CLAUDE.md` is gitignored** -- this file is local-only, not pushed to remote
- **`tests/` is gitignored** -- the test suite lives locally, not in the remote repo
- **`papers/` is gitignored** -- reference PDFs are local-only
- **No CI/CD** -- no `.github/workflows/` exist; all testing is manual
- **README.md is stale** -- still describes a 4-stage pipeline with PT2/residual expansion that has been removed

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Package manager | uv (hatchling build) |
| Neural networks | PyTorch >= 2.0 (2.10.0+cu130 on DGX Spark) |
| Molecular integrals | PySCF >= 2.3 |
| Eigensolvers | SciPy (CPU) / CuPy (GPU) |
| Quantum circuits | CUDA-Q (optional, graceful fallback) |
| Formatting | black (line-length 100) |
| Linting | ruff (line-length 100) |
| Type checking | mypy |
| Testing | pytest (587 tests, all passing) |
| Containerization | Docker (pytorch/pytorch:2.2.0-cuda12.1) |
| Hardware | NVIDIA DGX Spark (GB10, ARM64, 128GB UMA, SM121) |

---

## Architecture

### 3-Stage Pipeline (`src/pipeline.py`)

```
Stage 1: NF-NQS Training (or Direct-CI)
  ├── ParticleConservingFlowSampler learns ground-state distribution
  ├── AutoregressiveFlowSampler (Phase 4a, decoder-only transformer)
  ├── DenseNQS learns wavefunction amplitudes
  └── OR skip training, use HF + singles + doubles (Direct-CI mode)
        ↓
Stage 2: Diversity-Aware Basis Extraction
  ├── Bucket configs by excitation rank (0=HF, 1=singles, 2=doubles, ...)
  ├── DPP-greedy selection for diversity within each bucket
  └── Essential configs (HF + singles + doubles) always preserved
        ↓
Stage 3: Subspace Diagonalization
  ├── SKQD: Krylov time evolution e^{-iHdt}, expanding subspace, sparse eigsh
  └── SQD: S-CORE config recovery, batch diag, energy-variance extrapolation
```

**Note on NF conditional enable**: `adapt_to_system_size()` (called by default via `auto_adapt=True`) now conditionally sets `skip_nf_training`: >20K configs enables NF, <=20K skips. `_user_set_skip_nf` preserves explicit user override.

### Module Map

```
src/
├── pipeline.py                    # PipelineConfig + FlowGuidedKrylovPipeline orchestrator
├── flows/
│   ├── particle_conserving_flow.py  # NF that enforces exact electron count (SigmoidTopK)
│   ├── autoregressive_flow.py       # AR Transformer (decoder-only, KV cache, quaternary states)
│   ├── discrete_flow.py             # NF for spin systems (no particle constraint)
│   ├── physics_guided_training.py   # Co-trains NF + NQS (molecular systems)
│   ├── vmc_training.py              # VMCTrainer (REINFORCE -- needs MinSR upgrade)
│   ├── sign_network.py              # SignNetwork (FC->tanh -- needs phase/det upgrade)
│   └── training.py                  # Legacy trainer for spin systems
├── nqs/
│   ├── base.py                      # NeuralQuantumState ABC
│   ├── dense.py                     # DenseNQS, SignedDenseNQS
│   └── complex_nqs.py               # Complex-valued NQS
├── hamiltonians/
│   ├── base.py                      # Hamiltonian ABC (diagonal_element, get_connections)
│   ├── molecular.py                 # MolecularHamiltonian (PySCF integrals, Slater-Condon)
│   └── spin.py                      # TransverseFieldIsing, HeisenbergHamiltonian
├── krylov/
│   ├── skqd.py                      # SKQD solver (Krylov time evolution)
│   ├── sqd.py                       # SQD solver (IBM paper algorithm)
│   ├── nnci.py                      # NNCI active learning (NN classifier + iterative)
│   ├── basis_sampler.py             # CUDA-Q / classical Krylov sampling
│   └── residual_expansion.py        # Legacy PT2 code (NOT imported by pipeline)
├── postprocessing/
│   ├── diversity_selection.py       # DPP-inspired diversity-aware basis selection
│   ├── projected_hamiltonian.py     # H_ij = <x_i|H|x_j> construction
│   ├── eigensolver.py               # Davidson / sparse eigsh / adaptive selection
│   └── utils.py
└── utils/
    ├── connection_cache.py          # GPU-accelerated Hamiltonian connection cache
    ├── gpu_linalg.py                # gpu_eigh, gpu_eigsh, gpu_expm_multiply
    ├── config_hash.py               # Overflow-safe config hashing (n_sites >= 64)
    ├── hamiltonian_cache.py         # Disk cache for integrals + FCI energy
    ├── memory_logger.py             # Pre-allocation memory logging
    ├── benchmark.py                 # BenchmarkTimer, MemoryTracker
    └── perturbative_pruning.py      # MP2 importance scoring
```

Note: `src/utils/system_scaler.py` does NOT exist despite being listed in older docs. Stale reference.

### Key Classes

- **`PipelineConfig`** -- Master dataclass. `subspace_mode`: `"skqd"` or `"sqd"`. `skip_nf_training`: enables Direct-CI mode. `adapt_to_system_size()` auto-scales parameters; conditionally enables NF for >20K configs, skips for <=20K. `_user_set_skip_nf` preserves explicit override.
- **`FlowGuidedKrylovPipeline`** -- Orchestrator. `.run()` executes all 3 stages. `auto_adapt=True` by default (calls `adapt_to_system_size()`).
- **`MolecularHamiltonian`** -- Second-quantized electronic Hamiltonian from PySCF integrals (FP64). Implements Slater-Condon rules with Numba JIT. Factory functions: `create_h2_hamiltonian()`, `create_lih_hamiltonian()`, `create_n2_cas_hamiltonian()`, `create_cr2_hamiltonian()`, `create_benzene_hamiltonian()`, etc.
- **`ParticleConservingFlowSampler`** -- Non-autoregressive NF that always produces configs with exactly n_alpha + n_beta electrons. Uses `SigmoidTopK` for differentiable top-k orbital selection with `Plackett-Luce` probability model. Learnable temperature with softplus reparameterization.
- **`AutoregressiveFlowSampler`** -- Decoder-only transformer (Phase 4a). Quaternary states (empty/beta/alpha/alpha-beta). BOS token, pre-LN, causal mask. KV-cached O(n^2) decoding. Drop-in replacement for ParticleConservingFlowSampler.
- **`VMCTrainer`** -- REINFORCE-based VMC energy minimization (Phase 4b). Default OFF. Does NOT converge at 24Q+. Needs MinSR upgrade (see `docs/TODO-PHASE-6.md`).
- **`SignNetwork`** -- FC->GELU->tanh sign network (Phase 4c). Enables psi(x)=sqrt(p(x))*s(x) ansatz. Wrong approach -- SOTA uses determinant-based or continuous phase (e^{i*phi}).
- **`NNCIActiveLearning`** -- NN classifier + iterative train->score->expand->diag cycle. Auto-enabled for >20K configs via pipeline.
- **`SampleBasedKrylovDiagonalization`** / **`FlowGuidedSKQD`** -- Krylov subspace construction via time evolution. Key knobs: `max_diag_basis_size=15000`, `MAX_FULL_SUBSPACE_SIZE=100000`. Has sparse eigensolver path (`_sparse_ground_state`, threshold=3000). Adaptive dt, energy convergence monitoring, Lanczos full reorthogonalization.
- **`SQDSolver`** -- IBM's SQD with two sub-modes: **SQD-Clean** (`noise_rate=0`, default) and **SQD-Recovery** (`noise_rate>0`, injects depolarizing noise then runs S-CORE config recovery). Fails on systems >16Q.
- **`PhysicsGuidedFlowTrainer`** -- Co-trains NF + NQS. Uses cross-entropy loss with `entropy_weight=0.05` and exponential temperature annealing. `ConnectionCache` with warmup. Handles both AR and non-AR flows (AR skips batch logsumexp re-normalization).

---

## Common Commands

### Environment

```bash
uv sync                          # Install all dependencies
uv sync --extra dev              # Include dev tools (pytest, black, ruff, mypy)
uv sync --extra cuda             # Include CuPy for GPU acceleration
```

### Testing

```bash
uv run pytest                                  # Run all tests (587 tests, excludes @slow)
uv run pytest -m slow                          # Run only slow integration tests
uv run pytest -m molecular                     # Molecular-only tests (need PySCF)
uv run pytest tests/test_pipeline.py           # Single file
uv run pytest tests/test_pipeline.py -k "test_basic"  # Single test
uv run pytest --cov=src --cov-report=term      # With coverage
```

Note: `tests/` is gitignored. `tests/conftest.py` has session-scoped molecular and GPU fixtures. Molecular tests auto-skip if PySCF unavailable. CUDA tests auto-skip on CPU.

### Formatting & Linting

```bash
uv run black src/ tests/ examples/    # Format (line-length 100)
uv run ruff check src/ tests/         # Lint
uv run ruff check --fix src/ tests/   # Lint + auto-fix
uv run mypy src/                      # Type check
```

### Running Examples

```bash
# Local (CPU)
uv run python examples/validate_small_systems.py     # H2/LiH/H2O/BeH2/NH3/CH4
uv run python examples/subspace_comparison.py         # SKQD vs SQD side-by-side
uv run python examples/moderate_system_benchmark.py   # 20-30 qubit systems

# Docker (GPU)
docker-compose run --rm flow-krylov-gpu python examples/validate_small_systems.py
docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py
```

### Pipeline Usage

```python
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.molecular import create_lih_hamiltonian

# Direct-CI + SKQD (recommended default)
H = create_lih_hamiltonian(bond_length=1.6)
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=True)
pipeline = FlowGuidedKrylovPipeline(H, config=config)
results = pipeline.run()

# AR Transformer + SKQD (Phase 4a)
config = PipelineConfig(
    subspace_mode="skqd",
    skip_nf_training=False,
    use_autoregressive_flow=True,  # Phase 4a AR transformer
    # use_vmc_training=False,  # Default OFF (needs MinSR)
    # use_sign_network=False,  # Default OFF (needs phase/det upgrade)
)
pipeline = FlowGuidedKrylovPipeline(H, config=config)
results = pipeline.run()
```

---

## Code Style & Conventions

- **Line length**: 100 (black + ruff)
- **Target Python**: 3.10 (walrus operator, match statements OK; no 3.12+ features)
- **Imports**: Dual try/except pattern for relative vs absolute imports (supports both `python -m` and direct execution)
- **Config**: Dataclasses with defaults, not dicts. `PipelineConfig` is the single source of truth.
- **Immutability preferred**: Return new objects rather than mutating in place
- **Type hints**: Used throughout, but `mypy` configured with `ignore_missing_imports = true`
- **Docstrings**: Module-level and class-level docstrings present. NumPy-style parameter docs.
- **Tests**: pytest with class-based grouping (`class TestXxx`), `@pytest.fixture` for setup. Tests import via `sys.path.insert` from `src/`. `conftest.py` provides session-scoped molecular and GPU fixtures.
- **Commit style**: Conventional commits -- `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`, `chore:`

---

## System Size Tiers

| Tier | Qubits | Configs | Examples | Notes |
|------|--------|---------|----------|-------|
| small | 4-14 | < 2K | H2, LiH, H2O, BeH2 | Direct diag works, exact FCI achievable |
| medium | 16-18 | 2K-20K | NH3, CH4 | Direct-CI still hits FCI |
| large | 20-24 | 14K-100K | N2, CO, HCN, CAS(10,12) | Direct-CI sufficient on STO-3G; NF needed with larger basis sets |
| very_large | 26-40 | 100K-10M+ | CAS(10,15), CAS(10,20) | NF provides no benefit over Direct-CI + Krylov at 40Q |
| target | 40+ | 10M+ | [2Fe-2S] CAS(30,20) | Requires MinSR + H-coupling guidance (see TODO-PHASE-6) |

`PipelineConfig.adapt_to_system_size(n_valid_configs)` auto-scales training parameters, basis limits, and Krylov settings per tier. Conditionally enables NF for >20K configs, skips for <=20K. `_user_set_skip_nf` preserves explicit user override.

### Available Molecular Systems

Factory functions in `src/hamiltonians/molecular.py`. STO-3G systems use FP64 integrals. CAS systems support cc-pVDZ. Reference energies come from `MolecularHamiltonian.fci_energy()` at runtime.

| Factory function | Molecule | Basis | Electrons | Orbitals | Configs |
|-----------------|----------|-------|-----------|----------|---------|
| `create_h2_hamiltonian(bond_length=0.74)` | H2 | STO-3G | 2 | 2 | 4 |
| `create_lih_hamiltonian(bond_length=1.6)` | LiH | STO-3G | 4 | 6 | 225 |
| `create_h2o_hamiltonian()` | H2O | STO-3G | 10 | 7 | 441 |
| `create_beh2_hamiltonian()` | BeH2 | STO-3G | 6 | 7 | 1,225 |
| `create_nh3_hamiltonian()` | NH3 | STO-3G | 10 | 8 | 3,136 |
| `create_ch4_hamiltonian()` | CH4 | STO-3G | 10 | 9 | 15,876 |
| `create_n2_hamiltonian(bond_length=1.10)` | N2 | STO-3G | 14 | 10 | 14,400 |
| `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(10,12))` | N2 CAS | cc-pVDZ | 10 | 12 | 427,350 |
| `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(10,20))` | N2 CAS | cc-pVDZ | 10 | 20 | 240M |
| `create_cr2_hamiltonian()` | Cr2 | STO-3G | 24 | 12 | -- |
| `create_benzene_hamiltonian()` | Benzene | STO-3G | 6 | 15 | -- |

**Basis set limitation**: STO-3G is only acceptable for method development. 2026 publication standard requires cc-pVDZ minimum. IBM uses N2/cc-pVDZ CAS(10,26o) as de facto SQD benchmark.

---

## Ablation Study Results (RESULTS.md)

### Key Findings from 7-experiment study on STO-3G

1. **SKQD >> SQD**: SKQD achieves chemical accuracy on all 7 systems (7/7 PASS). SQD fails on CH4 and N2 (errors 2.4-12.1 mHa). Gap is 8-193x. **SKQD is the correct solver.**
2. **Direct-CI is sufficient on STO-3G**: HF + singles + doubles capture ~99.99% correlation energy. NF adds marginal improvement (0.7-21%).
3. **NF's value does NOT grow at 40Q**: After SKQD bug fix (get_combined_basis), Direct-CI + Krylov expansion discovers the same configs as NF. NF provides no benefit at CAS(10,20) 40Q.
4. **NF training is expensive**: CH4 takes 76 min NF training vs 7.5 min SKQD. Cost-benefit only makes sense when NF finds configs that Direct-CI cannot -- and it currently does not.

---

## Important Notes & Gotchas

### Physics Constraints

- **Particle conservation is mandatory** for molecular systems. The flow must produce configs with exactly `n_alpha` alpha + `n_beta` beta electrons. `verify_particle_conservation()` checks this.
- **CCSD is NOT a variational lower bound** -- only FCI is. Never use CCSD energy as a lower-bound reference.
- **Essential configs (HF + singles + doubles) must survive all pipeline stages.** The ground state wavefunction is dominated by these. If they're filtered out, accuracy collapses.
- Hamiltonian matrix elements follow **Slater-Condon rules**: only single and double excitations have non-zero off-diagonal elements.

### NF/VMC Architecture Limitations (Phase 4 Findings)

- **Non-autoregressive limitation is RESOLVED** (Phase 4a): `AutoregressiveFlowSampler` is a decoder-only transformer with quaternary states, KV-cached decoding, and exact log_prob via teacher forcing.
- **NEW limitations identified by deep research (60+ papers):**
  1. **REINFORCE is wrong optimizer** -- field uses SR (Stochastic Reconfiguration), MinSR, or SPRING. REINFORCE has O(1/sqrt(N)) convergence with high variance. See `docs/TODO-PHASE-6.md` P0.1.
  2. **No H-coupling guidance** -- HAAR-SCI (JCTC Dec 2025) uses Hamiltonian coupling to filter configs, achieving 0.51 mHa on 116 spin orbitals. Our NF generates configs blindly.
  3. **Wrong sign factorization** -- psi=sqrt(p)*s(x) with FC->tanh is wrong. SOTA uses determinant-based (Transformer Backflow) or continuous phase (e^{i*phi}). See `docs/TODO-PHASE-6.md` P2.1.
- **NQS-SC > NQS-VMC**: ETH Zurich (arXiv:2602.12993) validated that NQS as Selected Configurations outperforms NQS with VMC optimization. This supports our Direct-CI + SKQD approach.
- **Reference**: `docs/TODO-PHASE-6.md` for the fix plan (MinSR, H-coupling, iterative refinement).

### Performance

- **FP64 integrals**: h1e/h2e stored as FP64 (torch.float64). All Hamiltonian matrix elements computed in FP64. MATRIX_ELEMENT_TOL = 1e-12.
- **Numba JIT**: `get_connections` uses Numba-compiled functions for 18.7x speedup.
- **Sparse eigensolver**: SKQD uses sparse eigsh for subspaces > 3000 configs.
- `SKQDConfig.max_diag_basis_size=15000` caps the subspace for dense diag. `MAX_FULL_SUBSPACE_SIZE=100000` is the hard OOM guard.
- `ConnectionCache` (LRU, GPU-encoded keys) prevents redundant Hamiltonian connection recomputation.
- **TF32 global side-effect**: Importing `src/flows/physics_guided_training.py` sets `torch.set_float32_matmul_precision('high')` and enables TF32 for CUDA matmul and cuDNN.
- **Adaptive dt**: SKQD auto-selects dt = pi / spectral_range for large/very_large tiers. Energy convergence monitoring (3-step window).
- **Lanczos full reorthogonalization**: `_expm_multiply_lanczos()` uses full reorthogonalization (not just 3-term recurrence).
- **scipy expm_multiply dispatch**: CSR matrices use `scipy.sparse.linalg.expm_multiply` (Al-Mohy & Higham), dense use GPU. Eliminates slow PyTorch sparse COO matmul on ARM64/SM121.

### DGX Spark Specifics

- **FP64**: 0.48 TFLOPS (1:64 vs FP32). NVIDIA says FP64 is NOT a target use case. Eigensolves should use CPU SciPy for small systems, GPU sparse eigsh only for large.
- **TF32**: 53.3 TFLOPS. Use for NF training.
- **UMA**: 128GB shared CPU/GPU memory. Zero-copy advantage for large matrices.
- **Pageable memory**: 50x slowdown for small H2D copies on ARM64/UMA. Use pinned memory or allocate directly on GPU.
- **torch.compile**: Broken on SM121 (Triton treats as SM80). Use eager mode.
- **PyTorch**: 2.10.0+cu130 recommended.

### Code Patterns

- `src/krylov/residual_expansion.py` exists on disk but is **not imported** by the pipeline. It's legacy PT2 code kept for reference.
- CUDA-Q is optional. All solvers fall back to classical (NumPy/SciPy) when `cudaq` is unavailable.
- CuPy is optional. GPU eigensolvers fall back to SciPy when `cupy` is unavailable.
- Docker mounts the workspace at `/app` with `PYTHONPATH=/app/src`.

### Known Bugs

- **VMCTrainer uses REINFORCE** (wrong optimizer -- needs MinSR/SR/SPRING). VMC does not converge at 24Q+. See `docs/TODO-PHASE-6.md` P0.1.
- **SignNetwork uses psi=sqrt(p)*s(x)** (wrong factorization -- SOTA uses determinant-based or continuous phase e^{i*phi}). See `docs/TODO-PHASE-6.md` P2.1.

### Diversity Selection Defaults

The `DiversityConfig` excitation-rank budget is a strong physics-informed prior:

| Rank | Fraction | Meaning |
|------|----------|---------|
| 0 | 5% | HF and near-HF |
| 1 | 25% | Single excitations |
| 2 | 50% | Double excitations (dominant in ground state) |
| 3 | 15% | Triple excitations |
| 4+ | 5% | Higher excitations |

`min_hamming_distance=2` enforces diversity. DPP-greedy selection maximizes `weight * hamming_distance` within each bucket.

---

## Competitive Landscape (March 2026)

### Direct Competitors (Generative Model + CI/Diag)

| Method | Architecture | Scale | Key Advantage |
|--------|-------------|-------|---------------|
| HAAR-SCI | Gated Transformer + Gumbel Top-K | 116 spin orbs | H-coupling guided, 0.51 mHa |
| GTNN-SCI | Generative Transformer SCI | [2Fe-2S] | 10x speedup, chemical accuracy |
| QiankunNet | Autoregressive Transformer + MCTS | 30 spin orbs, 99.9% FCI | Proven architecture |
| Transformer Backflow | Transformer + determinant | CAS(30,20) | [2Fe-2S] chemical accuracy |
| NQS-SC (ETH Zurich) | NQS + selected configs | molecular | NQS-SC > NQS-VMC |
| NNQS-SCI | Transformer + Selected CI | 152 spin orbs | Best scaling |
| NNCI | NN classifier + active learning | N2, 4x10^5 dets | Simplest approach |
| PIGen-SQD | RBM + tensor decomposition | 52 qubits | IBM ecosystem |
| AB-SND | Autoregressive NN + basis opt | molecular systems | Classical, no quantum HW |

### Classical Baselines

| Method | Best Result (2025-2026) |
|--------|----------------------|
| GPU-DMRG (SandboxAQ) | CAS(82,82) orbital-optimized, CAS(113,76) single-point |
| SqDRIFT (IBM) | 72-100 qubits, Science Mar 2026 |
| SHCI | Cr2 28e/76o, 2 billion determinants |
| FCIQMC-GAS | 96e/159o |

**CAS(30,20) for [2Fe-2S] is "trivially classical" for DMRG.** The quantum/NF advantage threshold is currently beyond CAS(100,100).

### Critical Papers

- Reinholdt et al. (JCTC 2025): "Critical Limitations in QSCI" -- QSCI less compact than classical SCI
- MIT review (arXiv:2508.20972): Classical methods superior for ~20 years
- IBM SqDRIFT (Science, March 2026): 72 qubits, half-Mobius molecule
- NQS-SC > NQS-VMC (arXiv:2602.12993, ETH Zurich Feb 2026): Selected configurations outperform VMC
- HAAR-SCI (JCTC Dec 2025): H-coupling guided generation, 116 spin orbitals, 0.51 mHa
- GTNN-SCI (JCTC Dec 2025): Generative Transformer for [2Fe-2S], 10x speedup
- Transformer Backflow CAS(30,20) (arXiv:2509.25720): Determinant-based sign, [2Fe-2S] chemical accuracy

---

### Deep Research Findings (March 2026)

See `docs/TODO-PHASE-6.md` for prioritized action plan. Four survey documents in `docs/`:
- `NQS-VMC-SURVEY-2026-Q1.md` -- NQS architectures, VMC convergence, SR/MinSR/SPRING
- `LITERATURE-SURVEY-2026-Q1.md` -- SQD/SKQD/Krylov, SCI+ML, GPU-DMRG
- `GENERATIVE-MODEL-SURVEY-2026-03.md` -- HAAR-SCI, GTNN-SCI, NNCI, CIGS, NQS-SC
- `VMC-SIGN-PROBLEM-RESEARCH.md` -- Sign architectures, NNBF, Transformer Backflow

**Three root causes of NF/VMC failure at 40Q:**
1. REINFORCE is wrong -- field uses SR/MinSR/SPRING
2. psi=sqrt(p)*s(x) is wrong -- SOTA uses determinant-based or continuous phase
3. NF lacks H-coupling guidance -- HAAR-SCI uses Hamiltonian coupling to filter configs

**Key validation:** NQS-SC > NQS-VMC (arXiv:2602.12993). Direct-CI + SKQD is the correct approach.
