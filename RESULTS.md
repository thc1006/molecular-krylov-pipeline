# Flow-Guided Krylov Pipeline: Ablation Study Results

Results from the 7-experiment ablation study defined in `examples/nf_trained_comparison.py`.
All systems use the STO-3G basis set. FCI reference energies are computed at runtime via PySCF.

---

## Experimental Setup

### Molecular Systems

| System | Formula | Qubits | Electrons | Orbitals | Configs | Geometry |
|--------|---------|--------|-----------|----------|---------|----------|
| H2     | H2      | 4      | 2         | 2        | 4       | 0.74 A   |
| LiH    | LiH     | 12     | 4         | 6        | 225     | 1.6 A    |
| H2O    | H2O     | 14     | 10        | 7        | 441     | OH=0.96 A, 104.5 deg |
| BeH2   | BeH2    | 14     | 6         | 7        | 1,225   | Be-H=1.33 A, linear  |
| NH3    | NH3     | 16     | 10        | 8        | 3,136   | N-H=1.01 A, 107.8 deg |
| CH4    | CH4     | 18     | 10        | 9        | 15,876  | C-H=1.09 A, tetrahedral |
| N2     | N2      | 20     | 14        | 10       | 14,400  | 1.10 A   |

### 7 Ablation Experiments

The study tests two axes: **initial basis strategy** (rows) x **subspace solver** (columns).

|                     | SKQD Solver                  | SQD Solver                  |
|---------------------|------------------------------|-----------------------------|
| **HF only**         | 1. CudaQ SKQD               | --                          |
| **Direct-CI (HF+S+D)** | 2. Pure SKQD            | 3. Pure SQD                 |
| **NF + Direct-CI**  | 4. NF+CI SKQD               | 5. NF+CI SQD               |
| **NF only**         | 6. NF-only SKQD             | 7. NF-only SQD             |

1. **CudaQ SKQD** -- HF reference state only; Krylov time evolution discovers all connected configs. Faithful to the NVIDIA CUDA-Q tutorial.
2. **Pure SKQD** -- Direct-CI basis (HF + singles + doubles) fed into Krylov expansion. No NF training.
3. **Pure SQD** -- Direct-CI basis replicated to simulate circuit shots, depolarizing noise injected, S-CORE recovery, batch diagonalization. No NF training.
4. **NF+CI SKQD** -- Pre-trained NF basis merged with Direct-CI essential configs, then Krylov expansion.
5. **NF+CI SQD** -- Pre-trained NF basis merged with Direct-CI, replicated, noise injected, S-CORE recovery, batch diag.
6. **NF-only SKQD** -- NF basis only (no essential config injection) fed into Krylov expansion.
7. **NF-only SQD** -- NF basis only (no essential config injection), replicated, noise injected, S-CORE, batch diag.

For experiments 4-7, a single NF is trained once per system and shared across all four runs, isolating the subspace method as the only variable.

### Key Parameters

- **Chemical accuracy threshold**: 1.0 kcal/mol = 1.594 mHa
- **SKQD Krylov dim**: 8 (small), 10 (medium), 12 (large)
- **SQD batches**: 5 (small), 8 (medium), 10 (large); 5 self-consistent iters
- **SQD noise rate**: 0.03 (H2), 0.05 (all others)
- **Shot replication**: ~20K total shots per SQD run (10-200x multiplier)

---

## Error Table (mHa from FCI)

Lower is better. Chemical accuracy = 1.594 mHa.

| System | Qubits | CudaQ SKQD | Pure SKQD | Pure SQD | NF+CI SKQD | NF+CI SQD | NF-only SKQD | NF-only SQD |
|--------|--------|-----------|-----------|----------|------------|-----------|-------------|------------|
| H2     | 4      | --        | --        | --       | --         | --        | --          | --         |
| LiH    | 12     | --        | --        | --       | --         | --        | --          | --         |
| H2O    | 14     | --        | --        | --       | --         | --        | --          | --         |
| BeH2   | 14     | --        | --        | --       | --         | --        | --          | --         |
| NH3    | 16     | --        | --        | --       | --         | --        | --          | --         |
| CH4    | 18     | --        | --        | --       | --         | --        | --          | --         |
| N2     | 20     | --        | --        | --       | --         | --        | --          | --         |

---

## Chemical Accuracy (PASS/FAIL)

PASS = error < 1.594 mHa (1.0 kcal/mol). FAIL = error >= 1.594 mHa.

| System | Qubits | CudaQ SKQD | Pure SKQD | Pure SQD | NF+CI SKQD | NF+CI SQD | NF-only SKQD | NF-only SQD |
|--------|--------|-----------|-----------|----------|------------|-----------|-------------|------------|
| H2     | 4      | --        | --        | --       | --         | --        | --          | --         |
| LiH    | 12     | --        | --        | --       | --         | --        | --          | --         |
| H2O    | 14     | --        | --        | --       | --         | --        | --          | --         |
| BeH2   | 14     | --        | --        | --       | --         | --        | --          | --         |
| NH3    | 16     | --        | --        | --       | --         | --        | --          | --         |
| CH4    | 18     | --        | --        | --       | --         | --        | --          | --         |
| N2     | 20     | --        | --        | --       | --         | --        | --          | --         |

---

## Ablation Analysis

### Axis 1: HF-only vs Direct-CI (CudaQ SKQD vs Pure SKQD)

**Question**: Does pre-injecting singles and doubles into the initial basis help, or can Krylov time evolution discover them on its own?

- **CudaQ SKQD** starts from a single HF reference state. Krylov expansion via e^{-iHdt} must discover all connected configurations through Hamiltonian connectivity.
- **Pure SKQD** pre-injects all single and double excitations (Direct-CI), giving Krylov a head start with the configurations that dominate the ground-state wavefunction.

| System | CudaQ SKQD (mHa) | Pure SKQD (mHa) | Improvement |
|--------|-------------------|------------------|-------------|
| H2     | --                | --               | --          |
| LiH    | --                | --               | --          |
| H2O    | --                | --               | --          |
| BeH2   | --                | --               | --          |
| NH3    | --                | --               | --          |
| CH4    | --                | --               | --          |
| N2     | --                | --               | --          |

**Expected finding**: Direct-CI pre-injection should help for larger systems where Krylov alone cannot reach all important configurations within the limited Krylov dimension budget. For small systems (H2, LiH), the difference should be negligible since the Hilbert space is small enough for Krylov to explore fully.

### Axis 2: NF+CI vs NF-only (Essential config injection ablation)

**Question**: When the NF provides a learned basis, does injecting essential configs (HF + singles + doubles) still help?

| System | NF+CI SKQD (mHa) | NF-only SKQD (mHa) | NF+CI SQD (mHa) | NF-only SQD (mHa) |
|--------|-------------------|---------------------|------------------|---------------------|
| H2     | --                | --                  | --               | --                  |
| LiH    | --                | --                  | --               | --                  |
| H2O    | --                | --                  | --               | --                  |
| BeH2   | --                | --                  | --               | --                  |
| NH3    | --                | --                  | --               | --                  |
| CH4    | --                | --                  | --               | --                  |
| N2     | --                | --                  | --               | --                  |

**Expected finding**: Essential config injection should serve as a safety net. If the NF successfully discovers the ground-state region, NF-only and NF+CI should perform similarly. If the NF misses key configurations, the CI injection should rescue accuracy. The effect should be most visible for larger systems where NF training is harder.

### Axis 3: SKQD vs SQD (Solver comparison)

**Question**: Given the same initial basis, which subspace solver produces better energies?

- **SKQD** uses Krylov time evolution to expand the basis, then diagonalizes the full union.
- **SQD** uses depolarizing noise + S-CORE config recovery + batch diagonalization with energy-variance extrapolation.

| System | Pure SKQD (mHa) | Pure SQD (mHa) | NF+CI SKQD (mHa) | NF+CI SQD (mHa) |
|--------|------------------|-----------------|-------------------|-------------------|
| H2     | --               | --              | --                | --                |
| LiH    | --               | --              | --                | --                |
| H2O    | --               | --              | --                | --                |
| BeH2   | --               | --              | --                | --                |
| NH3    | --               | --              | --                | --                |
| CH4    | --               | --              | --                | --                |
| N2     | --               | --              | --                | --                |

**Expected finding**: SKQD should excel for small-to-medium systems where Krylov expansion can fully explore the relevant subspace. SQD may show advantages for larger systems where its batch diagonalization and energy-variance extrapolation can compensate for incomplete basis coverage.

---

## Timing Results (seconds)

Wall-clock time per experiment. NF training time is shared across NF+CI and NF-only variants.

| System | NF Train | CudaQ SKQD | Pure SKQD | Pure SQD | NF+CI SKQD | NF+CI SQD | NF-only SKQD | NF-only SQD |
|--------|----------|-----------|-----------|----------|------------|-----------|-------------|------------|
| H2     | --       | --        | --        | --       | --         | --        | --          | --         |
| LiH    | --       | --        | --        | --       | --         | --        | --          | --         |
| H2O    | --       | --        | --        | --       | --         | --        | --          | --         |
| BeH2   | --       | --        | --        | --       | --         | --        | --          | --         |
| NH3    | --       | --        | --        | --       | --         | --        | --          | --         |
| CH4    | --       | --        | --        | --       | --         | --        | --          | --         |
| N2     | --       | --        | --        | --       | --         | --        | --          | --         |

---

## Basis Size (unique configurations)

Number of unique configurations in the subspace at diagonalization time.

| System | CudaQ SKQD | Pure SKQD | Pure SQD | NF+CI SKQD | NF+CI SQD | NF-only SKQD | NF-only SQD |
|--------|-----------|-----------|----------|------------|-----------|-------------|------------|
| H2     | --        | --        | --       | --         | --        | --          | --         |
| LiH    | --        | --        | --       | --         | --        | --          | --         |
| H2O    | --        | --        | --       | --         | --        | --          | --         |
| BeH2   | --        | --        | --       | --         | --        | --          | --         |
| NH3    | --        | --        | --       | --         | --        | --          | --         |
| CH4    | --        | --        | --       | --         | --        | --          | --         |
| N2     | --        | --        | --       | --         | --        | --          | --         |

---

## Key Findings

*To be filled after running experiments.*

1. **Direct-CI value**: Does pre-injecting HF + singles + doubles significantly improve over HF-only Krylov?
2. **NF contribution**: Does NF training improve over pure Direct-CI baselines?
3. **Essential config safety net**: Does CI injection help when combined with NF?
4. **SKQD vs SQD**: Which solver is more accurate and when?
5. **Scaling behavior**: How do the methods compare as system size grows from 4 to 20 qubits?

---

## How to Reproduce

```bash
# Full ablation study (all 7 systems, all 7 experiments)
uv run python examples/nf_trained_comparison.py

# Specific systems (small, fast)
uv run python examples/nf_trained_comparison.py --systems h2 lih h2o beh2

# Larger systems (GPU recommended)
uv run python examples/nf_trained_comparison.py --systems nh3 ch4 n2

# Docker (GPU, recommended for NH3/CH4/N2)
docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py

# Single system for quick testing
docker-compose run --rm flow-krylov-gpu python examples/nf_trained_comparison.py --systems h2
```

### Hardware Requirements

- **H2, LiH, H2O, BeH2**: CPU sufficient, <1 min each
- **NH3, CH4**: GPU recommended, 5-30 min each depending on hardware
- **N2**: GPU strongly recommended, 20-60 min

### References

1. Yu, Robledo-Moreno et al., "Sample-based Krylov Quantum Diagonalization"
2. Robledo-Moreno, Motta et al., "Chemistry Beyond the Scale of Exact Diagonalization", Science 2024
3. "Improved Ground State Estimation via Normalising Flow-Assisted Neural Quantum States"
4. NVIDIA CUDA-Q SKQD Tutorial (Trotterized evolution from reference state)
