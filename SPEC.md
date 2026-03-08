# SPEC: Flow-Guided Krylov Pipeline

This specification is written for an AI coding assistant (Claude Code) or a new
teammate picking up the project. It describes every module, class, function
signature, data flow, physics constraint, and known limitation so that you can
confidently implement new features without re-reading every source file.

---

## 1. Project Purpose

Compute **molecular electronic ground-state energies** approaching FCI (Full
Configuration Interaction) accuracy by:

1. Generating a compact basis of Slater determinants via a **Normalizing Flow**
   (NF) that respects particle conservation.
2. Selecting a diverse, physics-informed subset of those determinants.
3. Projecting the molecular Hamiltonian onto that subspace and diagonalizing it
   with either **SKQD** (Krylov time evolution) or **SQD** (IBM's batch
   diagonalization with config recovery).

All molecular integrals come from **PySCF** (STO-3G basis). Reference energies
are FCI, computed at runtime via `MolecularHamiltonian.fci_energy()`.

---

## 2. Directory Layout

```
src/
├── pipeline.py                        # Orchestrator: PipelineConfig + FlowGuidedKrylovPipeline
├── flows/
│   ├── particle_conserving_flow.py    # ParticleConservingFlowSampler (SigmoidTopK + Plackett-Luce)
│   └── physics_guided_training.py     # PhysicsGuidedFlowTrainer co-trains NF + NQS
├── nqs/
│   ├── base.py                        # NeuralQuantumState ABC
│   └── dense.py                       # DenseNQS (real-valued log-amplitude network)
├── hamiltonians/
│   ├── base.py                        # Hamiltonian ABC (diagonal_element, get_connections)
│   └── molecular.py                   # MolecularHamiltonian (PySCF, Slater-Condon, Numba JIT)
├── krylov/
│   ├── skqd.py                        # SampleBasedKrylovDiagonalization + FlowGuidedSKQD
│   ├── sqd.py                         # SQDSolver (IBM paper algorithm)
│   ├── nnci.py                        # NNCIActiveLearning (NN classifier + active learning)
│   └── basis_sampler.py               # CUDA-Q classical Krylov sampling (optional)
├── postprocessing/
│   ├── diversity_selection.py         # DiversitySelector (DPP-greedy, excitation-rank buckets)
│   ├── projected_hamiltonian.py       # Build H_ij = <x_i|H|x_j> matrix
│   ├── eigensolver.py                 # Davidson / sparse eigsh / adaptive eigensolver
│   └── utils.py                       # Hamming distance, excitation rank utilities
└── utils/
    ├── gpu_linalg.py                  # gpu_eigh, gpu_eigsh, gpu_expm_multiply
    ├── connection_cache.py            # LRU cache for Hamiltonian connections (GPU keys)
    ├── config_hash.py                 # Overflow-safe config integer hashing (n_sites >= 64)
    ├── hamiltonian_cache.py           # Disk cache for integrals (h1e/h2e) and FCI energy
    ├── memory_logger.py               # Pre-allocation memory logging (/proc/meminfo)
    ├── benchmark.py                   # BenchmarkTimer, MemoryTracker, regression detection
    └── perturbative_pruning.py        # MP2 importance scoring and basis pruning
```

---

## 3. Pipeline Stages (src/pipeline.py)

### 3.1 Entry Point

```python
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.molecular import create_lih_hamiltonian

H = create_lih_hamiltonian(bond_length=1.6)
config = PipelineConfig(subspace_mode="skqd")
pipeline = FlowGuidedKrylovPipeline(H, config=config, exact_energy=H.fci_energy())
results = pipeline.run()
# results["combined_energy"] -> final ground-state energy in Hartree
```

`FlowGuidedKrylovPipeline.__init__` requires a `MolecularHamiltonian` (raises
`TypeError` otherwise). It calls `config.adapt_to_system_size(n_valid_configs)`
to auto-tune parameters by tier (small/medium/large/very_large).

### 3.2 Stage 1 — Basis Generation (`train_flow_nqs`)

Two modes controlled by `PipelineConfig.skip_nf_training`:

| Mode | skip_nf_training | What happens |
|------|-----------------|--------------|
| **Direct-CI** | `True` (default for molecular via `adapt_to_system_size`) | `_generate_essential_configs()` produces HF + all singles + up to 5000 doubles |
| **NF-NQS** | `False` | `PhysicsGuidedFlowTrainer.train()` co-trains the normalizing flow + NQS for `max_epochs` epochs |

When NF training runs, the trainer accumulates discovered configs in
`trainer.accumulated_basis` (a torch.Tensor on device).

### 3.3 Stage 2 — Diversity-Aware Selection (`extract_and_select_basis`)

- In Direct-CI mode: uses essential configs as-is (no diversity filtering).
- In NF mode: `DiversitySelector` stratifies configs by excitation rank
  relative to HF, then DPP-greedy selects within each bucket.

**Excitation-rank budget** (default `DiversityConfig`):

| Rank | Fraction | Meaning |
|------|----------|---------|
| 0 | 5% | HF and near-HF |
| 1 | 25% | Single excitations |
| 2 | 50% | Double excitations (dominant) |
| 3 | 15% | Triple excitations |
| 4+ | 5% | Higher excitations |

After diversity selection, essential configs (from trainer) are always merged
back to guarantee HF + singles + doubles survive.

### 3.4 Stage 3 — Subspace Diagonalization (`run_subspace_diag`)

Dispatches to `_run_skqd` or `_run_sqd` based on `config.subspace_mode`.

**SKQD path** (`_run_skqd`):
1. Creates `FlowGuidedSKQD(hamiltonian, nf_basis, config=SKQDConfig(...))`
2. Calls `skqd.run_with_nf()` which:
   - For small/medium systems: builds full particle-conserving subspace
     Hamiltonian, performs Krylov time evolution via `gpu_expm_multiply`
   - For very large systems (>100K configs): uses NF-guided Krylov that
     expands the basis by finding connected configs via Slater-Condon rules
3. Returns `best_stable_energy` from the Krylov sequence

**SQD path** (`_run_sqd`):
1. Creates `SQDSolver(hamiltonian, config=SQDConfig(...))`
2. Calls `solver.run(nf_basis)` which:
   - Optionally injects depolarizing noise (`noise_rate > 0`)
   - Filters/recovers configs via S-CORE (self-consistent config recovery)
   - Creates K batches with spin-symmetry enhancement
   - Diagonalizes each batch independently
   - Runs self-consistent orbital occupancy loop
   - Performs energy-variance extrapolation (linear fit E vs dH/E^2)
3. Returns extrapolated energy

Both paths include a variational consistency check: if computed energy is
below `exact_energy - 0.001 Ha`, it flags numerical instability and falls
back to direct diagonalization.

---

## 4. Key Classes — Full Interface Reference

### 4.1 PipelineConfig (dataclass)

All fields with defaults — grouped by function:

```
# Architecture
use_particle_conserving_flow: bool = True
nf_hidden_dims: list = [256, 256]          # NF network layers
nqs_hidden_dims: list = [256, 256, 256, 256]  # NQS network layers

# Training
samples_per_batch: int = 2000
num_batches: int = 1
max_epochs: int = 400
min_epochs: int = 100
convergence_threshold: float = 0.20
teacher_weight: float = 1.0               # Cross-entropy weight (paper's only loss)
physics_weight: float = 0.0               # VMC energy weight (disabled by default)
entropy_weight: float = 0.0               # Entropy regularization (disabled)
nf_lr: float = 5e-4
nqs_lr: float = 1e-3

# Basis management (auto-scaled by adapt_to_system_size)
max_accumulated_basis: int = 4096
use_diversity_selection: bool = True
max_diverse_configs: int = 2048
rank_2_fraction: float = 0.50

# Subspace diag mode
subspace_mode: str = "skqd"               # "skqd" or "sqd"

# SQD parameters
sqd_num_batches: int = 5
sqd_batch_size: int = 0                   # 0 = auto from NF samples
sqd_self_consistent_iters: int = 3
sqd_spin_penalty: float = 0.0
sqd_noise_rate: float = 0.0              # 0 = SQD-Clean, >0 = SQD-Recovery
sqd_use_spin_symmetry: bool = True

# SKQD parameters
max_krylov_dim: int = 8
time_step: float = 0.1
shots_per_krylov: int = 50000
skqd_regularization: float = 1e-8
skip_skqd: bool = False                  # True = NF-only mode (no Krylov)

# Training mode
use_local_energy: bool = True
skip_nf_training: Optional[bool] = None   # None=auto, True=Direct-CI, False=NF

# NNCI parameters
use_nnci: Optional[bool] = None           # None=auto, True=enable, False=disable
nnci_iterations: int = 5
nnci_candidates_per_iter: int = 5000

# Hardware
device: str = "cuda" if available else "cpu"

# Large-system performance
max_connections_per_config: int = 0       # 0 = no truncation
diagonal_only_warmup_epochs: int = 0
stochastic_connections_fraction: float = 1.0
```

**`adapt_to_system_size(n_valid_configs)`** mutates self in-place. Tiers:
- small (<=1K): defaults fine
- medium (1K-5K): NQS [384]*5, basis 8K
- large (5K-20K): NQS [512]*5, basis 12K, 600 epochs
- very_large (>20K): NQS [512]*4, basis 16K, Krylov dim 4

Conditional NF: >20K configs enables NF (`skip_nf_training=False`), ≤20K skips.
`_user_set_skip_nf` preserves explicit user overrides. NNCI auto-enables for >20K.

### 4.2 MolecularHamiltonian (src/hamiltonians/molecular.py)

Created via factory functions:

```python
create_h2_hamiltonian(bond_length=0.74)   # 4 qubits, 4 configs
create_lih_hamiltonian(bond_length=1.6)   # 12 qubits, 225 configs
create_h2o_hamiltonian()                  # 14 qubits, 441 configs
create_beh2_hamiltonian()                 # 14 qubits, 1225 configs
create_nh3_hamiltonian()                  # 16 qubits, 3136 configs
create_ch4_hamiltonian()                  # 18 qubits, 15876 configs
create_n2_hamiltonian(bond_length=1.10)   # 20 qubits, 14400 configs
```

Key attributes:
- `num_sites`: 2 * n_orbitals (alpha spin orbitals + beta spin orbitals)
- `n_orbitals`, `n_alpha`, `n_beta`: orbital/electron counts
- `hilbert_dim`: 2^num_sites (full, NOT particle-conserving)

Key methods:
- `diagonal_element(config) -> float`: <x|H|x>
- `get_connections(config) -> (connected_configs, matrix_elements)`: Slater-Condon
- `get_connections_vectorized_batch(configs) -> (all_connected, all_elements, counts)`: GPU batch
- `matrix_elements(bra_configs, ket_configs) -> Tensor`: full H_ij matrix
- `fci_energy() -> float`: exact FCI from PySCF (runtime, not cached)
- `get_hf_state() -> Tensor`: Hartree-Fock reference state
- `to_dense(device) -> Tensor`: full Hamiltonian matrix (only for small systems)

Config encoding: binary tensor of length `num_sites = 2 * n_orbitals`.
First `n_orbitals` bits = alpha spin orbitals, last `n_orbitals` = beta.

### 4.3 ParticleConservingFlowSampler (src/flows/particle_conserving_flow.py)

Normalizing flow that ALWAYS produces configs with exactly `n_alpha` + `n_beta`
electrons.

```python
flow = ParticleConservingFlowSampler(
    num_sites=14, n_alpha=5, n_beta=5, hidden_dims=[256, 256]
)
log_probs, configs = flow.sample(batch_size=1000)
# configs: (1000, 14) binary tensor, guaranteed particle-conserving
# log_probs: (1000,) log-probabilities
```

Uses **SigmoidTopK** with **Plackett-Luce** probability model for
differentiable top-k orbital selection. Learnable temperature (nn.Parameter
with softplus reparameterization, min_temperature=0.1). The flow learns
per-orbital logits, then selects exactly k occupied orbitals.

### 4.4 PhysicsGuidedFlowTrainer (src/flows/physics_guided_training.py)

Co-trains NF + NQS with a mixed objective:

```
L = teacher_weight * L_CE + physics_weight * L_energy + entropy_weight * L_entropy
```

Default (paper): only cross-entropy (`teacher_weight=1.0`, others 0.0),
weighted by `|E_local|` so low-energy configs get higher gradient signal.

Uses `ConnectionCache` to avoid redundant Hamiltonian connection lookups.
Warms up the cache with HF + singles + doubles ("essential configs") before
training starts. Accumulates all discovered configs in `self.accumulated_basis`.

### 4.5 SampleBasedKrylovDiagonalization (src/krylov/skqd.py)

Core SKQD algorithm. For molecular systems:
1. Builds particle-conserving subspace (all valid alpha x beta configs)
2. Constructs subspace Hamiltonian H_sub (sparse or dense)
3. Time-evolves via `gpu_expm_multiply`: |psi_k> = e^{-iH*dt} |psi_{k-1}>
4. Samples from evolved state to expand basis
5. Diagonalizes combined basis with `gpu_eigsh`

```python
SKQDConfig(
    max_krylov_dim=8,        # Number of Krylov steps
    time_step=0.1,           # dt for time evolution
    shots_per_krylov=100000, # Sampling shots per step
    max_new_configs_per_krylov_step=1000,  # Cap expansion rate
    max_diag_basis_size=15000,  # Cap for dense diag (OOM prevention)
    regularization=1e-8,
)
```

### 4.6 FlowGuidedSKQD (src/krylov/skqd.py)

Extends `SampleBasedKrylovDiagonalization`. Merges NF-discovered basis with
Krylov-expanded basis at each step.

Two modes (auto-selected by config space size):
- **Full subspace** (<=100K configs): builds complete particle-conserving
  Hamiltonian, evolves in that subspace
- **NF-guided** (>100K configs): skips full subspace construction; instead
  expands the NF basis by finding Hamiltonian-connected configs via
  Slater-Condon rules (single + double excitations of existing basis)

```python
skqd = FlowGuidedSKQD(hamiltonian, nf_basis, config=SKQDConfig(...))
results = skqd.run_with_nf(progress=True)
# results["best_stable_energy"] -> best energy from Krylov sequence
# results["energies_combined"] -> energy at each Krylov step
```

### 4.7 SQDSolver (src/krylov/sqd.py)

IBM's SQD algorithm adapted for NF-NQS sampling.

```python
SQDConfig(
    num_batches=5,              # K independent batches
    batch_size=0,               # 0 = auto
    self_consistent_iters=3,    # S-CORE iterations
    spin_penalty=0.0,           # Lambda for S^2 penalty
    noise_rate=0.0,             # 0 = SQD-Clean, >0 = SQD-Recovery
    enable_config_recovery=False,  # S-CORE on/off
    use_spin_symmetry_enhancement=True,
)
```

Key internal methods:
- `_inject_noise(configs)`: depolarizing noise (bit flips)
- `_filter_and_recover_configs(configs)`: particle-number filtering + S-CORE
- `_create_batches(configs)`: K batches with spin recombination
- `_diagonalize_batch(batch)`: build H on GPU, diag with gpu_eigsh
- `_energy_variance_extrapolation(energies, variances)`: linear fit E vs dH/E^2

### 4.8 DiversitySelector (src/postprocessing/diversity_selection.py)

```python
DiversityConfig(
    max_configs=2048,
    rank_2_fraction=0.50,
    min_hamming_distance=2,
    excitation_budgets={0: 0.05, 1: 0.25, 2: 0.50, 3: 0.15, 4: 0.05},
)
selector = DiversitySelector(config, reference=hf_state, n_orbitals=7)
selected, stats = selector.select(all_configs)
```

### 4.9 GPU Utilities (src/utils/gpu_linalg.py)

- `gpu_eigh(H)`: dense symmetric eigendecomposition (torch on GPU, scipy on CPU)
- `gpu_eigsh(H, k)`: sparse top-k eigenvalues. Dense torch for n<=10K, CuPy
  sparse eigsh for larger. Falls back to scipy if CuPy unavailable.
- `gpu_expm_multiply(H, v, t)`: Lanczos-based matrix exponential e^{-iHt}|v>.
  Does NOT form the full matrix exponential — uses Krylov approximation.

CuPy integration uses DLPack zero-copy: `cp.from_dlpack(tensor.detach())`

### 4.10 ConnectionCache (src/utils/connection_cache.py)

LRU cache for `hamiltonian.get_connections(config)`. Keys are GPU-encoded
integers (config tensor → single int via bit packing). Prevents redundant
Slater-Condon rule evaluations during training.

---

## 5. The 7 Ablation Experiments

Defined in `examples/nf_trained_comparison.py`. Each is a function that
constructs a `PipelineConfig` and runs the pipeline:

| # | Name | Basis | Solver | Key Config |
|---|------|-------|--------|------------|
| 1 | CudaQ SKQD | HF only | SKQD | `skip_nf_training=True`, only HF in basis, `max_krylov_dim=12` |
| 2 | Pure SKQD | Direct-CI (HF+S+D) | SKQD | `skip_nf_training=True`, `max_krylov_dim=8` |
| 3 | Pure SQD | Direct-CI (HF+S+D) | SQD | `skip_nf_training=True`, `subspace_mode="sqd"`, `noise_rate=0.05` |
| 4 | NF+CI SKQD | NF + Direct-CI | SKQD | `skip_nf_training=False`, essential configs merged after NF |
| 5 | NF+CI SQD | NF + Direct-CI | SQD | `skip_nf_training=False`, `subspace_mode="sqd"`, `noise_rate=0.05` |
| 6 | NF-only SKQD | NF only | SKQD | `skip_nf_training=False`, essential config merging disabled |
| 7 | NF-only SQD | NF only | SQD | `skip_nf_training=False`, `subspace_mode="sqd"`, no essentials |

Ablation axes:
- **Axis 1** (Basis completeness): HF-only (#1) vs Direct-CI (#2) vs NF+CI (#4)
- **Axis 2** (Essential injection): NF+CI (#4/#5) vs NF-only (#6/#7)
- **Axis 3** (Solver): SKQD (#2/#4/#6) vs SQD (#3/#5/#7)

---

## 6. Available Molecular Systems

| Key | Molecule | Factory | Qubits | Valid Configs | FCI Tractable? |
|-----|----------|---------|--------|---------------|----------------|
| h2 | H2 | `create_h2_hamiltonian(bond_length=0.74)` | 4 | 4 | Trivial |
| lih | LiH | `create_lih_hamiltonian(bond_length=1.6)` | 12 | 225 | Yes |
| h2o | H2O | `create_h2o_hamiltonian()` | 14 | 441 | Yes |
| beh2 | BeH2 | `create_beh2_hamiltonian()` | 14 | 1,225 | Yes |
| nh3 | NH3 | `create_nh3_hamiltonian()` | 16 | 3,136 | Yes |
| ch4 | CH4 | `create_ch4_hamiltonian()` | 18 | 15,876 | Yes |
| n2 | N2 | `create_n2_hamiltonian(bond_length=1.10)` | 20 | 14,400 | Yes |

All use STO-3G basis. `create_c2h4_hamiltonian` does **not** exist as a
factory function — C2H4 (28 qubits, 9M configs) must be constructed manually
via PySCF integrals.

---

## 7. Physics Constraints (Must-Follow Rules)

1. **Particle conservation is mandatory.** Every configuration tensor must have
   exactly `n_alpha` ones in the first `n_orbitals` positions and `n_beta` ones
   in the last `n_orbitals` positions. Use `verify_particle_conservation()` to
   check. The `ParticleConservingFlowSampler` enforces this by construction.

2. **Essential configs (HF + singles + doubles) must survive all stages.**
   The ground-state wavefunction is dominated by these. If they get filtered
   out by diversity selection or basis truncation, accuracy collapses.

3. **CCSD is NOT a variational lower bound** — only FCI is. Never use CCSD
   energy as a reference floor. Always use `H.fci_energy()` for validation.

4. **Slater-Condon rules**: only single and double excitations produce
   non-zero off-diagonal Hamiltonian matrix elements. This means the
   Hamiltonian matrix is sparse in the determinant basis.

5. **Variational principle**: any computed energy MUST be >= FCI energy.
   If it goes below, there is a numerical instability (overlap matrix
   conditioning, basis linear dependence, etc.).

---

## 8. System Size Tiers and Scaling

| Tier | Configs | Behavior |
|------|---------|----------|
| small (<=1K) | H2, LiH, H2O | Direct diag works. Full FCI reachable trivially. |
| medium (1K-5K) | BeH2, NH3 | Direct-CI hits FCI. NQS [384]*5. |
| large (5K-20K) | CH4, N2 | NF training or large Direct-CI needed. NQS [512]*5. |
| very_large (>20K) | C2H4 | NF essential. NF-guided Krylov mode. Memory-sensitive. |

For very_large systems, `FlowGuidedSKQD` auto-switches to NF-guided mode
(skips full subspace construction). The NF-guided Krylov expansion finds
connected configs by applying Slater-Condon rules to the existing NF basis.

---

## 9. GPU Acceleration Details

| Component | GPU Path | CPU Fallback |
|-----------|----------|--------------|
| NF training | PyTorch autograd on CUDA | PyTorch on CPU |
| Hamiltonian construction | `matrix_elements` with torch tensors on GPU | Same, on CPU |
| Time evolution | `gpu_expm_multiply` (Lanczos on GPU) | `scipy.sparse.linalg.expm_multiply` |
| Eigensolvers | `gpu_eigsh` → CuPy sparse or torch dense | `scipy.sparse.linalg.eigsh` |
| Connection cache | GPU integer keys (bit-packed) | CPU dict |
| S^2 matrix (SQD) | Vectorized pairwise GPU for n<=2000 | CPU for larger |

CuPy zero-copy: `cp.from_dlpack(tensor.detach().contiguous())` avoids
GPU→CPU→GPU round-trips.

**TF32 side-effect**: Importing `src/flows/physics_guided_training.py` calls
`torch.set_float32_matmul_precision('high')` globally.

---

## 10. Critical Performance Knobs

| Knob | Location | Default | Effect |
|------|----------|---------|--------|
| `max_diag_basis_size` | `SKQDConfig` | 15000 | Caps subspace for dense diag. PRIMARY OOM prevention. |
| `max_new_configs_per_krylov_step` | `SKQDConfig` | 1000 | Caps expansion rate per Krylov step. |
| `max_connections_per_config` | `PipelineConfig` | 0 (off) | Truncates connections to top-k by magnitude. |
| `diagonal_only_warmup_epochs` | `PipelineConfig` | 0 (off) | Skips off-diagonal H for first N epochs. |
| `stochastic_connections_fraction` | `PipelineConfig` | 1.0 (all) | Samples fraction of connections. |
| `krylov_basis_sample_fraction` | `SKQDConfig` | 0.8 | Fraction of basis sampled for connected configs. |
| `MAX_FULL_SUBSPACE_SIZE` | `FlowGuidedSKQD` | 100000 | Threshold for NF-guided vs full-subspace mode. |

---

## 11. Adding a New Molecular System

To add a new molecule (e.g., F2):

1. **Add factory function** in `src/hamiltonians/molecular.py`:
   ```python
   def create_f2_hamiltonian(bond_length: float = 1.42) -> MolecularHamiltonian:
       """Create F2 Hamiltonian with STO-3G basis."""
       mol = gto.M(atom=f"F 0 0 0; F 0 0 {bond_length}", basis="sto-3g", symmetry=False)
       mf = scf.RHF(mol).run(verbose=0)
       return MolecularHamiltonian.from_pyscf(mf)
   ```

2. **Register in `__init__.py`**: Add to `src/hamiltonians/__init__.py` imports
   and `__all__`.

3. **Add to example scripts**: Add entry in `SYSTEMS` dict in
   `examples/nf_trained_comparison.py` and `examples/validate_small_systems.py`.

4. **Check tier**: Compute `C(n_orb, n_alpha) * C(n_orb, n_beta)` for the
   config count. If >20K, NF training is needed (`skip_nf_training=False`).

5. **Memory check**: For >50K configs, verify the subspace Hamiltonian fits in
   GPU memory. A 50K x 50K complex128 matrix = ~40 GB. Use NF-guided Krylov
   mode for >100K.

---

## 12. Adding a New Solver

To add a third subspace diagonalization method:

1. Create `src/krylov/new_solver.py` with a class that takes
   `(hamiltonian, config)` and has a `run(nf_basis) -> Dict` method.

2. Add the mode string to `PipelineConfig.subspace_mode` options.

3. Add a `_run_new_solver` method in `FlowGuidedKrylovPipeline`.

4. Update `run_subspace_diag` dispatch.

5. Add corresponding config parameters to `PipelineConfig`.

6. Add an experiment function in `examples/nf_trained_comparison.py`.

---

## 13. Known Limitations and Open Issues

1. **`adapt_to_system_size` conditionally enables NF for >20K configs.**
   For ≤20K configs, NF is skipped (Direct-CI). Explicit user overrides
   via `skip_nf_training=True/False` are preserved (`_user_set_skip_nf`).
   NNCI is also auto-enabled for >20K configs unless user overrides.

2. **No `create_c2h4_hamiltonian` factory.** C2H4 (28 qubits, 9M configs)
   requires manual PySCF construction. The very_large tier code paths exist
   but have not been validated end-to-end on C2H4.

3. **SQD energy-variance extrapolation** assumes linear E vs dH/E^2. This
   may not hold for poorly conditioned batches. The extrapolated energy can
   occasionally be below FCI (violating the variational principle).

4. **Tests reference stale API.** Some test files in `tests/` may reference
   old class names or removed modules. Run `pytest` and fix failures before
   adding new tests.

5. **No larger basis sets.** All factory functions use STO-3G. Supporting
   cc-pVDZ or larger requires only changing the `basis` parameter in PySCF,
   but the resulting Hilbert space grows dramatically.

6. **RESULTS.md has placeholder data.** The error/timing tables need to be
   filled after running `examples/nf_trained_comparison.py` on GPU.

---

## 14. Running the Project

```bash
# Install
uv sync
uv sync --extra cuda    # GPU support

# Validate (CPU OK for small systems)
python examples/validate_small_systems.py

# Full 7-experiment ablation (GPU recommended)
docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py

# Specific systems
python examples/nf_trained_comparison.py --systems h2 lih h2o beh2 nh3 ch4 n2

# SKQD vs SQD side-by-side
python examples/subspace_comparison.py

# Formatting and linting
uv run black src/ tests/ examples/
uv run ruff check src/ tests/
uv run mypy src/
```

---

## 15. Import Patterns

All `src/` modules use a dual try/except pattern for imports:

```python
try:
    from .module import Class       # Works with `python -m src.pipeline`
except ImportError:
    from module import Class        # Works with `python src/pipeline.py` or PYTHONPATH=src
```

This is intentional. Do NOT refactor to a single import style — both paths
are needed for Docker (`PYTHONPATH=/app/src`) and local development.

---

## 16. Commit and Code Style

- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`, `chore:`
- **Line length**: 100 (black + ruff)
- **Python**: 3.10+ (walrus operator OK, no 3.12+ features)
- **Immutability preferred**: return new objects, don't mutate shared state
- **Config via dataclasses**: `PipelineConfig` is the single source of truth
- **No hardcoded secrets**: all config is in code or CLI args
