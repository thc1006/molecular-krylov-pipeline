# Flow-Guided Krylov Quantum Diagonalization

A quantum chemistry pipeline for computing molecular ground-state energies by combining **Normalizing Flow-Assisted Neural Quantum States (NF-NQS)** with **Krylov subspace diagonalization**. The core idea: use a normalizing flow to discover high-probability Slater determinants, then diagonalize the Hamiltonian projected onto that basis to systematically converge toward FCI-level accuracy.

Two subspace solvers are available: **SKQD** (Krylov time evolution) and **SQD** (IBM's sampling-based batch diagonalization). Together with three basis generation strategies, the pipeline supports 7 distinct ablation configurations for systematic comparison.

---

## Table of Contents

- [Methodology](#methodology)
- [The 7 Ablation Pipelines](#the-7-ablation-pipelines)
- [Available Molecular Systems](#available-molecular-systems)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running Examples](#running-examples)
- [GPU Acceleration](#gpu-acceleration)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [References](#references)
- [License](#license)

---

## Methodology

### 3-Stage Pipeline

```
Stage 1: Basis Generation
  ├── Direct-CI: deterministic HF + singles + doubles (Slater-Condon rules)
  ├── NF-NQS: particle-conserving flow (GumbelTopK) co-trained with NQS
  └── Hybrid: NF-learned basis merged with Direct-CI essentials
        |
        v
Stage 2: Diversity-Aware Selection
  ├── Bucket configs by excitation rank (0=HF, 1=singles, 2=doubles, ...)
  ├── DPP-greedy selection for diversity within each bucket
  └── Essential configs (HF + singles + doubles) always preserved
        |
        v
Stage 3: Subspace Diagonalization
  ├── SKQD: Krylov time evolution e^{-iHdt}, expanding subspace, sparse eigsh
  └── SQD: noise injection -> S-CORE config recovery -> batch diag -> energy-variance extrapolation
```

**Stage 1 -- Basis Generation** supports three strategies. *Direct-CI* deterministically enumerates Hartree-Fock plus all single and double excitations using Slater-Condon rules. *NF-NQS* trains a particle-conserving normalizing flow (using the Gumbel-Top-K straight-through estimator to enforce exact electron count) co-trained with a neural quantum state to learn the ground-state distribution. *Hybrid* merges NF-discovered configurations with Direct-CI essentials for robustness.

**Stage 2 -- Diversity-Aware Selection** applies excitation-rank stratification with a physics-informed budget:

| Excitation Rank | Budget | Description |
|-----------------|--------|-------------|
| 0 | 5% | HF and near-HF configurations |
| 1 | 25% | Single excitations |
| 2 | 50% | Double excitations (dominant in ground state) |
| 3 | 15% | Triple excitations |
| 4+ | 5% | Higher excitations |

Within each bucket, DPP-greedy selection maximizes `weight * hamming_distance` to ensure diversity (`min_hamming_distance=2`).

**Stage 3 -- Subspace Diagonalization** offers two solvers. *SKQD* constructs a Krylov subspace via time evolution `|psi_k> = e^{-ikH*dt}|psi_0>`, expanding the basis at each step, and solves via sparse eigendecomposition. *SQD* (from IBM's "Chemistry Beyond Exact Diagonalization") injects depolarizing noise, applies S-CORE configuration recovery with probabilistic bit-flipping, performs batch diagonalization with self-consistent orbital occupancy iteration, and extrapolates to zero variance via linear `E vs dH/E^2` fitting.

---

## The 7 Ablation Pipelines

The pipeline supports 7 experimental configurations spanning two ablation axes:

| # | Pipeline | Basis Strategy | Solver | Workflow |
|---|----------|---------------|--------|----------|
| 1 | CudaQ SKQD | HF only | SKQD | HF reference state -> Krylov expansion |
| 2 | Pure SKQD | Direct-CI (HF+S+D) | SKQD | HF + singles + doubles -> Krylov expansion |
| 3 | Pure SQD | Direct-CI (HF+S+D) | SQD | HF + singles + doubles -> noise -> S-CORE -> batch diag |
| 4 | NF-Trained SKQD | NF + Direct-CI | SKQD | NF training + HF+S+D -> Krylov |
| 5 | NF-Trained SQD | NF + Direct-CI | SQD | NF training + HF+S+D -> noise -> S-CORE |
| 6 | NF-only SKQD | NF only | SKQD | NF basis -> Krylov (no essential config injection) |
| 7 | NF-only SQD | NF only | SQD | NF basis -> noise -> S-CORE (no essential config injection) |

**Ablation Axis 1 -- HF-only vs Direct-CI:** Does pre-injecting singles and doubles help, or can Krylov discover them on its own? Compare CudaQ SKQD (#1) against Pure SKQD (#2).

**Ablation Axis 2 -- NF+Direct-CI vs NF-only:** Does essential config injection help when the NF provides a learned basis? Compare pipelines #4/#5 against #6/#7.

---

## Available Molecular Systems

All factory functions use the **STO-3G** basis set. Reference energies are computed from FCI at runtime via PySCF.

| Factory Function | Molecule | Electrons | Orbitals | Qubits | Configs |
|-----------------|----------|-----------|----------|--------|---------|
| `create_h2_hamiltonian(bond_length=0.74)` | H2 | 2 | 2 | 4 | 4 |
| `create_lih_hamiltonian(bond_length=1.6)` | LiH | 4 | 6 | 12 | 225 |
| `create_h2o_hamiltonian()` | H2O | 10 | 7 | 14 | 441 |
| `create_beh2_hamiltonian()` | BeH2 | 6 | 7 | 14 | 1,225 |
| `create_nh3_hamiltonian()` | NH3 | 10 | 8 | 16 | 3,136 |
| `create_ch4_hamiltonian()` | CH4 | 10 | 9 | 18 | 15,876 |
| `create_n2_hamiltonian(bond_length=1.10)` | N2 | 14 | 10 | 20 | 14,400 |

All small systems (up to 18 qubits) achieve exact FCI energy (0.0000 mHa error) in Direct-CI mode.

---

## Quick Start

```python
from src.pipeline import FlowGuidedKrylovPipeline, PipelineConfig
from src.hamiltonians.molecular import create_lih_hamiltonian

# Create molecular Hamiltonian
H = create_lih_hamiltonian(bond_length=1.6)

# Run pipeline with SKQD solver in Direct-CI mode
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=True)
pipeline = FlowGuidedKrylovPipeline(H, config=config)
results = pipeline.run()

# Check results against FCI
E_exact = H.fci_energy()
print(f"Energy: {results['combined_energy']:.6f} Ha")
print(f"Error:  {abs(results['combined_energy'] - E_exact) * 1000:.4f} mHa")
```

To use the SQD solver instead:

```python
config = PipelineConfig(subspace_mode="sqd", skip_nf_training=True)
```

To enable NF-NQS training (recommended for systems with >20 qubits):

```python
config = PipelineConfig(subspace_mode="skqd", skip_nf_training=False)
```

---

## Installation

### With uv (recommended)

```bash
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov

# Install core dependencies
uv sync

# Include GPU support (CuPy)
uv sync --extra cuda

# Include dev tools (pytest, black, ruff, mypy)
uv sync --extra dev
```

### With pip

```bash
git clone https://github.com/George930502/Flow-Guided-Krylov.git
cd Flow-Guided-Krylov

pip install -e .

# For GPU support
pip install cupy-cuda12x>=12.0.0
```

### With Docker (GPU)

```bash
docker-compose build
docker-compose run --rm flow-krylov-gpu
```

---

## Running Examples

```bash
# Validate all small systems (H2, LiH, H2O, BeH2, NH3, CH4)
python examples/validate_small_systems.py

# SKQD vs SQD side-by-side comparison
python examples/subspace_comparison.py

# Full 7-experiment ablation study
python examples/nf_trained_comparison.py
python examples/nf_trained_comparison.py --systems h2 lih h2o

# Moderate system benchmarks (20-30 qubits)
python examples/moderate_system_benchmark.py

# Docker (GPU)
docker-compose run --rm flow-krylov-gpu python examples/validate_small_systems.py
docker-compose run --rm flow-krylov-gpu python examples/subspace_comparison.py
docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py
```

---

## GPU Acceleration

The pipeline is designed for end-to-end GPU execution, from NF training through final diagonalization.

- **Particle-conserving subspace**: operates in the electron-number subspace (10-100x smaller than the full Hilbert space)
- **GPU Lanczos matrix exponential** (`gpu_expm_multiply`): Krylov time evolution without materializing the full matrix exponential
- **GPU eigensolvers** (`gpu_eigsh`): dense `torch.linalg.eigh` for n <= 10K, CuPy sparse `eigsh` for larger subspaces
- **DLPack zero-copy**: CuPy interop via `cp.from_dlpack()` avoids GPU-to-CPU-to-GPU round-trips
- **ConnectionCache**: GPU integer-encoded LRU cache prevents redundant Hamiltonian connection recomputation
- **Vectorized Slater-Condon rules**: batch matrix element evaluation on GPU
- **TF32 matmul acceleration**: automatically enabled for CUDA matmul and cuDNN operations

All GPU features degrade gracefully: CuPy falls back to SciPy, CUDA-Q falls back to classical NumPy sampling.

---

## Architecture

```
src/
├── pipeline.py                        # PipelineConfig + FlowGuidedKrylovPipeline orchestrator
├── flows/
│   ├── particle_conserving_flow.py    # NF with exact electron count (GumbelTopK)
│   ├── discrete_flow.py              # NF for spin systems (no particle constraint)
│   ├── physics_guided_training.py    # Co-trains NF + NQS
│   └── training.py                   # Legacy trainer for spin systems
├── nqs/
│   ├── base.py                       # NeuralQuantumState ABC
│   ├── dense.py                      # DenseNQS, SignedDenseNQS
│   └── complex_nqs.py               # Complex-valued NQS
├── hamiltonians/
│   ├── base.py                       # Hamiltonian ABC (diagonal_element, get_connections)
│   └── molecular.py                  # MolecularHamiltonian (PySCF integrals, Slater-Condon)
├── krylov/
│   ├── skqd.py                       # SKQD solver (Krylov time evolution)
│   ├── sqd.py                        # SQD solver (IBM paper algorithm)
│   └── basis_sampler.py             # CUDA-Q / classical Krylov sampling
├── postprocessing/
│   ├── diversity_selection.py        # DPP-greedy diversity selection
│   ├── projected_hamiltonian.py      # H_ij = <x_i|H|x_j> construction
│   ├── eigensolver.py               # Davidson / sparse eigsh / adaptive selection
│   └── utils.py
└── utils/
    ├── gpu_linalg.py                 # gpu_eigh, gpu_eigsh, gpu_expm_multiply
    ├── connection_cache.py           # GPU-accelerated Hamiltonian connection cache
    └── system_scaler.py             # Auto-scale params by system size tier
```

### Key Classes

- **`PipelineConfig`** -- Master dataclass controlling the entire pipeline. Key fields: `subspace_mode` (`"skqd"` or `"sqd"`), `skip_nf_training` (enables Direct-CI mode). `adapt_to_system_size()` auto-scales parameters based on system size.
- **`FlowGuidedKrylovPipeline`** -- Orchestrator that executes all 3 stages via `.run()`.
- **`MolecularHamiltonian`** -- Second-quantized electronic Hamiltonian from PySCF one- and two-electron integrals. Implements Slater-Condon rules for matrix element evaluation.
- **`ParticleConservingFlowSampler`** -- Normalizing flow that always produces configurations with exactly `n_alpha + n_beta` electrons via Gumbel-Top-K differentiable sampling.
- **`SampleBasedKrylovDiagonalization`** -- SKQD solver. Constructs Krylov subspace via time evolution, expanding the basis at each step. `max_diag_basis_size=15000` caps dense diagonalization to prevent OOM.
- **`SQDSolver`** -- IBM SQD implementation. Two sub-modes: SQD-Clean (`noise_rate=0`, default) and SQD-Recovery (`noise_rate>0`, injects depolarizing noise then runs S-CORE configuration recovery).
- **`PhysicsGuidedFlowTrainer`** -- Co-trains NF + NQS with cross-entropy loss weighted by `|E|`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Package Manager | uv (hatchling build) |
| Neural Networks | PyTorch >= 2.0 |
| Molecular Integrals | PySCF >= 2.3 |
| Eigensolvers | SciPy (CPU) / CuPy (GPU) |
| Quantum Circuits | CUDA-Q (optional, graceful fallback) |
| Formatting | black (line-length 100) |
| Linting | ruff (line-length 100) |
| Type Checking | mypy |
| Testing | pytest |
| Containerization | Docker (pytorch/pytorch:2.2.0-cuda12.1) |

---

## References

1. Yu, Robledo-Moreno et al., "Sample-based Krylov Quantum Diagonalization" ([arXiv:2501.09702](https://arxiv.org/abs/2501.09702))
2. Robledo-Moreno, Motta et al., "Chemistry Beyond the Scale of Exact Diagonalization", *Science* 2024
3. "Improved Ground State Estimation via Normalising Flow-Assisted Neural Quantum States" ([arXiv:2506.12128](https://arxiv.org/abs/2506.12128))
4. NVIDIA CUDA-Q SKQD Tutorial (Heisenberg model, Trotterized evolution)

---

## License

MIT License
