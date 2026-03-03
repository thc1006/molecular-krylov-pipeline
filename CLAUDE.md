# Flow-Guided Krylov Pipeline

## Project Overview

Quantum chemistry pipeline for computing molecular ground-state energies. Combines **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** with **Krylov subspace diagonalization** to systematically converge toward FCI-level accuracy.

The core idea: use a normalizing flow to discover high-probability molecular configurations (determinants), then diagonalize the Hamiltonian projected onto that basis. Two subspace solvers are available: **SKQD** (Krylov time evolution) and **SQD** (IBM's sampling-based batch diagonalization).

### Key References (PDFs in `papers/`)

- "Improved Ground State Estimation via Normalising Flow-Assisted Neural Quantum States"
- "Sample-based Krylov Quantum Diagonalization"
- "Chemistry Beyond the Scale of Exact Diagonalization" (IBM SQD paper)

### Repository Notes

- **`CLAUDE.md` is gitignored** — this file is local-only, not pushed to remote
- **`tests/` is gitignored** — the test suite lives locally, not in the remote repo
- **`papers/` is gitignored** — reference PDFs are local-only
- **No CI/CD** — no `.github/workflows/` exist; all testing is manual
- **README.md is stale** — still describes a 4-stage pipeline with PT2/residual expansion that has been removed

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Package manager | uv (hatchling build) |
| Neural networks | PyTorch >= 2.0 |
| Molecular integrals | PySCF >= 2.3 |
| Eigensolvers | SciPy (CPU) / CuPy (GPU) |
| Quantum circuits | CUDA-Q (optional, graceful fallback) |
| Formatting | black (line-length 100) |
| Linting | ruff (line-length 100) |
| Type checking | mypy |
| Testing | pytest |
| Containerization | Docker (pytorch/pytorch:2.2.0-cuda12.1) |

---

## Architecture

### 3-Stage Pipeline (`src/pipeline.py`)

```
Stage 1: NF-NQS Training (or Direct-CI)
  ├── ParticleConservingFlowSampler learns ground-state distribution
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

### Module Map

```
src/
├── pipeline.py                    # PipelineConfig + FlowGuidedKrylovPipeline orchestrator
├── flows/
│   ├── particle_conserving_flow.py  # NF that enforces exact electron count
│   ├── discrete_flow.py             # NF for spin systems (no particle constraint)
│   ├── physics_guided_training.py   # Co-trains NF + NQS (molecular systems)
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
    └── system_scaler.py             # Auto-scale params by system size tier
```

### Key Classes

- **`PipelineConfig`** — Master dataclass. `subspace_mode`: `"skqd"` or `"sqd"`. `skip_nf_training`: enables Direct-CI mode. `adapt_to_system_size()` auto-scales parameters.
- **`FlowGuidedKrylovPipeline`** — Orchestrator. `.run()` executes all 3 stages.
- **`MolecularHamiltonian`** — Second-quantized electronic Hamiltonian from PySCF integrals. Implements Slater-Condon rules. Factory functions: `create_h2_hamiltonian()`, `create_lih_hamiltonian()`, etc.
- **`ParticleConservingFlowSampler`** — Normalizing flow that always produces configs with exactly n_alpha + n_beta electrons. Uses `GumbelTopK` (Gumbel-Softmax straight-through estimator) for differentiable top-k orbital selection.
- **`SampleBasedKrylovDiagonalization`** / **`FlowGuidedSKQD`** — Krylov subspace construction via time evolution. Key knob: `SKQDConfig.max_diag_basis_size=15000` caps dense diag to prevent OOM/hangs.
- **`SQDSolver`** — IBM's SQD with two sub-modes: **SQD-Clean** (`noise_rate=0`, default) and **SQD-Recovery** (`noise_rate>0`, injects depolarizing noise then runs S-CORE config recovery).
- **`PhysicsGuidedFlowTrainer`** — Co-trains NF + NQS. Default loss is **pure cross-entropy** weighted by `|E|` (`physics_weight=0.0`, `entropy_weight=0.0`). Uses `ConnectionCache` with warmup (HF + singles + doubles pre-cached).

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
uv run pytest                                  # Run all tests (excludes @slow)
uv run pytest -m slow                          # Run only slow integration tests
uv run pytest tests/test_pipeline.py           # Single file
uv run pytest tests/test_pipeline.py -k "test_basic"  # Single test
uv run pytest --cov=src --cov-report=term      # With coverage
```

Note: `tests/` is gitignored. Tests with `@pytest.mark.slow` exist in `test_pipeline.py`. CUDA-only tests auto-skip on CPU. Molecular tests auto-skip if PySCF is unavailable.

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

H = create_lih_hamiltonian(bond_length=1.6)
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=True)
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
- **Tests**: pytest with class-based grouping (`class TestXxx`), `@pytest.fixture` for setup. Tests import via `sys.path.insert` from `src/`.
- **Commit style**: Conventional commits — `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`, `chore:`

---

## System Size Tiers

| Tier | Qubits | Configs | Examples | Notes |
|------|--------|---------|----------|-------|
| small | 4-14 | < 2K | H2, LiH, H2O, BeH2 | Direct diag works, exact FCI achievable |
| medium | 16-18 | 2K-20K | NH3, CH4 | Direct-CI still hits FCI |
| large | 20-24 | 14K-100K | N2, CO, HCN | NF training or large Direct-CI needed |
| very_large | 26+ | 1M+ | C2H4 (9M) | NF essential, memory-sensitive |

`PipelineConfig.adapt_to_system_size(n_valid_configs)` auto-scales training parameters, basis limits, and Krylov settings per tier. It always forces `skip_nf_training=True` (Direct-CI mode) for molecular systems.

Note: `src/utils/system_scaler.py` contains a separate, more advanced `SystemScaler` class (6 tiers, logarithmic scaling, quality presets). It is a standalone utility and is **not** used by the pipeline's `adapt_to_system_size()`.

### Available Molecular Systems

All factory functions in `src/hamiltonians/molecular.py` use **STO-3G** basis. Reference energies come from `MolecularHamiltonian.fci_energy()` at runtime.

| Factory function | Molecule | Default geometry | Electrons | Orbitals | Configs |
|-----------------|----------|-----------------|-----------|----------|---------|
| `create_h2_hamiltonian(bond_length=0.74)` | H2 | 0.74 A | 2 | 2 | 4 |
| `create_lih_hamiltonian(bond_length=1.6)` | LiH | 1.6 A | 4 | 6 | 225 |
| `create_h2o_hamiltonian()` | H2O | OH=0.96 A, 104.5 deg | 10 | 7 | 441 |
| `create_beh2_hamiltonian()` | BeH2 | Be-H=1.33 A, linear | 6 | 7 | 1,225 |
| `create_nh3_hamiltonian()` | NH3 | N-H=1.01 A, 107.8 deg | 10 | 8 | 3,136 |
| `create_ch4_hamiltonian()` | CH4 | C-H=1.09 A, tetrahedral | 10 | 9 | 15,876 |
| `create_n2_hamiltonian(bond_length=1.10)` | N2 | 1.10 A | 14 | 10 | 14,400 |

Note: `create_c2h4_hamiltonian` does **not** exist despite C2H4 being referenced in benchmarks. C2H4 benchmarks construct the Hamiltonian manually.

---

## Important Notes & Gotchas

### Physics Constraints

- **Particle conservation is mandatory** for molecular systems. The flow must produce configs with exactly `n_alpha` alpha + `n_beta` beta electrons. `verify_particle_conservation()` checks this.
- **CCSD is NOT a variational lower bound** — only FCI is. Never use CCSD energy as a lower-bound reference.
- **Essential configs (HF + singles + doubles) must survive all pipeline stages.** The ground state wavefunction is dominated by these. If they're filtered out, accuracy collapses.
- Hamiltonian matrix elements follow **Slater-Condon rules**: only single and double excitations have non-zero off-diagonal elements.

### Performance

- For large systems (>20K configs), skip per-dimension Krylov diagnostics — they cause hangs.
- `SKQDConfig.max_diag_basis_size=15000` caps the subspace for dense diag. This is the primary OOM/hang prevention knob.
- `SKQDConfig.max_new_configs_per_krylov_step=1000` caps the expansion rate per Krylov step.
- `ConnectionCache` (LRU, GPU-encoded keys) prevents redundant Hamiltonian connection recomputation.
- Time evolution uses subspace method for molecules (10-100x smaller than full Hilbert space) and Lanczos (`gpu_expm_multiply`) for large spin systems.
- `max_connections_per_config`, `diagonal_only_warmup_epochs`, `stochastic_connections_fraction` are knobs for large-system training speed.
- **TF32 global side-effect**: Importing `src/flows/physics_guided_training.py` sets `torch.set_float32_matmul_precision('high')` and enables TF32 for CUDA matmul and cuDNN. This affects all subsequent torch operations in the process.

### Code Patterns

- `src/krylov/residual_expansion.py` exists on disk but is **not imported** by the pipeline. It's legacy PT2 code kept for reference.
- CUDA-Q is optional. All solvers fall back to classical (NumPy/SciPy) when `cudaq` is unavailable.
- CuPy is optional. GPU eigensolvers fall back to SciPy when `cupy` is unavailable.
- Docker mounts the workspace at `/app` with `PYTHONPATH=/app/src`.

### Testing Caveats

- Some tests reference older API signatures (e.g., `nf_coupling_layers`, `extract_basis()`). The test suite may need updates to match the current `remove-pt2-add-sqd` branch API.
- Molecular tests require PySCF, which needs ~1GB RAM for integral computation on larger systems.
- Tests use small spin systems (4-site Ising) for speed. Molecular integration tests are in `examples/`.
- No `conftest.py` exists — fixtures are defined inside test classes, not shared.

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
