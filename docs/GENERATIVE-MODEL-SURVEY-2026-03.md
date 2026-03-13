# Generative Models for Quantum Chemistry Configuration Sampling
## Comprehensive Survey (2025 - March 2026)

> **Date**: 2026-03-09
> **Context**: Our NF-guided SKQD pipeline's autoregressive transformer provides NO benefit over Direct-CI + Krylov expansion at 40Q scale. This survey evaluates all alternatives.

---

## Executive Summary

The field has moved decisively away from pure normalizing flows toward **Hamiltonian-guided autoregressive selected-CI** and **iterative train-predict-expand loops (NNCI)**. The two most important papers for our project are:

1. **HAAR-SCI** (JCTC, Dec 2025): Gated Transformer + Gumbel Top-K + GPU min-heap. 116 spin orbitals, 0.51 mHa MAE, 72% determinant reduction on [2Fe-2S]. **This is what our pipeline should become.**
2. **GTNN-SCI** (JCTC, Dec 2025): Generative Transformer SCI achieving 10x speedup over prior NN methods, chemical accuracy on [2Fe-2S] where conventional SCI fails.

Key insight: **the winning approaches don't just sample -- they use the Hamiltonian to guide which configurations to keep**. Our pipeline samples from p(x) then feeds to SKQD, but never uses H-coupling information to filter/rank during generation. This is the root cause of NF's irrelevance.

---

## Table of Contents

1. [Autoregressive Transformer + Selected CI (HAAR-SCI, GTNN-SCI)](#1-autoregressive-transformer--selected-ci)
2. [MCTS-Guided Sampling (QiankunNet)](#2-mcts-guided-sampling-qiankunnet)
3. [Neural Network Configuration Interaction (NNCI)](#3-neural-network-configuration-interaction-nnci)
4. [RBM-Based Approaches (CIGS, PIGen-SQD)](#4-rbm-based-approaches)
5. [Deterministic NQS Frameworks](#5-deterministic-nqs-frameworks)
6. [NQS-SC: Selected Configurations from NQS](#6-nqs-sc-selected-configurations-from-nqs)
7. [Physics-Informed Transformers](#7-physics-informed-transformers)
8. [Reinforcement Learning CI (RLCI)](#8-reinforcement-learning-ci)
9. [Machine Learning Assisted SCI (ML-ASCI)](#9-machine-learning-assisted-sci)
10. [AB-SND: Adaptive-Basis Neural Diagonalization](#10-ab-snd-adaptive-basis-neural-diagonalization)
11. [Diffusion Models](#11-diffusion-models)
12. [Foundation NQS Models](#12-foundation-nqs-models)
13. [Hybrid Tensor Network + NQS](#13-hybrid-tensor-network--nqs)
14. [Critical Assessments of SQD/QSCI](#14-critical-assessments-of-sqdqsci)
15. [Large-Scale HPC Results](#15-large-scale-hpc-results)
16. [Comparative Analysis & Recommendations](#16-comparative-analysis--recommendations)

---

## 1. Autoregressive Transformer + Selected CI

### HAAR-SCI (Hamiltonian-Guided Autoregressive Selected-CI)

**Paper**: "Hamiltonian-Guided Autoregressive Selected-Configuration Interaction Achieves Chemical Accuracy in Strongly Correlated Systems"
**Journal**: J. Chem. Theory Comput. 2025, 21, 24, 12622-12633
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415

**Architecture**: Learn-sample-compress workflow on single GPU:
1. **Gated Transformer** samples determinants autoregressively
2. **Gumbel Top-K noise** encourages exploration beyond greedy sampling
3. **GPU min-heap kernels** retain only configs with largest Hamiltonian couplings

**Key Results**:
- 18-molecule benchmark, up to **116 spin orbitals**
- Mean absolute error: **0.51 mHa**
- N2 dissociation curve: within **0.67 mHa** across entire curve
- **[2Fe-2S] cluster** (40 spin orbitals): matches HCI accuracy with **72% fewer determinants**
- Probability-mass pruning: further **10-50x compression**, retaining <0.01% of Hilbert space while capturing >99.9% correlation energy

**Why this matters for us**: HAAR-SCI's key innovation is **Hamiltonian coupling as a filter**. Instead of sampling p(x) and hoping important configs appear, it samples autoregressively then keeps only configs with large H-matrix couplings to the existing set. This is exactly what our pipeline lacks -- our NF samples, then SKQD does Krylov expansion, but there's no H-guided filtering during the sampling phase.

### GTNN-SCI (Generative Transformer Neural Network SCI)

**Paper**: "Accelerating Many-Body Quantum Chemistry via Generative Transformer-Enhanced Configuration Interaction"
**Journal**: J. Chem. Theory Comput. 2025, 21, 23, 11989-12000
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01429

**Architecture**: Transformer with self-attention for long-range electron correlations, generatively sampling important configurations.

**Key Results**:
- N2, H2O, C2 with **cc-pVDZ** and plane-wave basis sets
- **10x speedup** over prior neural network SCI methods
- **[2Fe-2S] cluster**: achieves chemical accuracy vs DMRG benchmarks
- Finds **higher-order excitations** missed by conventional coupling schemes (heat-bath CI)
- Lower variational energies than HCI on strongly correlated systems

**Why this matters**: GTNN-SCI proves that transformer-based generative sampling can beat established classical SCI methods (HCI/CIPSI) specifically because it finds important triples/quadruples that coupling-based methods miss. This is the exact gap our pipeline faces at 40Q.

---

## 2. MCTS-Guided Sampling (QiankunNet)

**Paper**: "Solving the many-electron Schrodinger equation with a transformer-based framework"
**Journal**: Nature Communications (2025)
**URL**: https://www.nature.com/articles/s41467-025-63219-2

**Architecture**:
- Decoder-only Transformer as wavefunction ansatz
- **Layer-wise MCTS** for autoregressive sampling (naturally enforces electron conservation)
- Physics-informed initialization from truncated CI solutions

**Key Results**:
- 16-molecule benchmark, up to **30 spin orbitals**
- Correlation energies reaching **99.9% of FCI**
- Independent samples (no MCMC autocorrelation)

**Comparison to our approach**: QiankunNet's MCTS differs from pure ancestral sampling. MCTS explores the decision tree of orbital occupations with lookahead, so it can avoid dead-end branches. Our autoregressive transformer does ancestral sampling only -- it commits to each orbital sequentially without backtracking.

**Gap**: QiankunNet solves the wavefunction directly (VMC-style energy minimization), not subspace diagonalization. Adapting MCTS to our sample-then-diagonalize pipeline would require using MCTS to find high-coupling configs rather than high-amplitude configs.

---

## 3. Neural Network Configuration Interaction (NNCI)

### FAU Erlangen NNCI

**Papers**:
- "Neural-Network-Based Selective Configuration Interaction" -- JCTC 2025
- "Natural-Orbital-Based NNCI" -- arXiv:2510.27665 (Oct 2025)
- "Orbital Optimization and NNCI for Rydberg States" -- arXiv:2510.26751 (Oct 2025)

**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.4c01479

**Architecture**: CNN classifier trained on-the-fly in active learning loop:
1. Start with small exact diagonalization
2. CNN predicts importance of new determinants from occupation-number representation
3. Add predicted-important determinants to CI basis
4. Re-diagonalize, retrain CNN, iterate

**Key Results**:
- N2: **4 x 10^5 determinants** vs 10^10 in FCI (5 orders of magnitude compression)
- NH3 and H2O Rydberg states: ~10^5 determinants (also 5 OOM fewer than FCI)
- Natural orbital basis reduces required determinants further vs canonical HF orbitals

**Comparison to our NNCI module**: Our `src/krylov/nnci.py` implements a similar loop but uses a feedforward classifier. The FAU version uses CNN on occupation vectors. Natural orbitals would help our pipeline too -- currently we use canonical HF orbitals only.

**Key insight**: NNCI's strength is **iterative refinement** -- each cycle discovers determinants the previous cycle missed. Our pipeline is single-shot: NF generates, SKQD expands, done. Adding NNCI-style iterative loops would likely help more than improving the NF architecture.

---

## 4. RBM-Based Approaches

### CIGS (Configuration Interaction Guided Sampling)

**Paper**: "Configuration Interaction Guided Sampling with Interpretable Restricted Boltzmann Machine"
**Published**: arXiv:2409.06146, J. Chem. Theory Comput. 2025
**URL**: https://arxiv.org/abs/2409.06146

**Architecture**: RBM with iterative guided training + taboo list:
1. RBM learns to sample important determinants
2. Diagonalize in sampled subspace
3. Prune low-importance determinants (add to taboo list)
4. Retrain RBM on retained set
5. Iterate until energy convergence

**Key Results**:
- **99.99% correlation energy** with 4 OOM fewer determinants than FCI
- **30-50% fewer** than CIPSI
- RBM learns interpretable patterns resembling radial distribution functions
- Taboo list prevents wasted resampling of already-rejected configs

**Why this matters**: The taboo list is a simple but powerful idea our pipeline lacks. During Krylov expansion, we re-discover many configs already in the basis. A hash set of rejected configs would prevent wasted H-matrix evaluations.

### PIGen-SQD (IBM)

**Paper**: "Physics-Informed Generative Machine Learning for Accelerated Quantum-Centric Supercomputing"
**Published**: arXiv:2512.06858 (Dec 2025)
**URL**: https://arxiv.org/html/2512.06858

**Architecture**:
- RBM + perturbatively predicted dominant states
- Self-consistent configuration recovery
- Integrated with IBM SQD pipeline

**Key Results**:
- Up to **52 qubits** on IBM Heron R2/R3 processors
- **70% reduction** in diagonalization subspace dimension
- ~1-1.5M determinants (order of magnitude less than standard SQD)
- RBM learns underlying structure of dominant Hilbert space sector

**Relevance**: PIGen-SQD shows that even a simple RBM, when combined with perturbative guidance (PT2-predicted important states), dramatically compresses the subspace. This is essentially what our MP2 pruning does, but PIGen-SQD integrates it into the generative model's training objective.

---

## 5. Deterministic NQS Frameworks

**Paper**: "A Deterministic Framework for Neural Network Quantum States in Quantum Chemistry"
**Published**: arXiv:2601.21310 (Jan 2026)
**URL**: https://arxiv.org/abs/2601.21310

**Architecture**:
- Neural backflow ansatz optimized within dynamically adaptive configuration subspaces
- **Second-order perturbation theory** corrections
- **No Monte Carlo noise** -- fully deterministic
- Hybrid CPU-GPU implementation

**Key Results**:
- Sub-linear scaling with subspace size
- Validated on H2O, N2 bond dissociation, Cr2 (strongly correlated)

**Why this matters**: Eliminates stochastic noise entirely. Our VMC training suffers from high variance at 24Q+ (VMC doesn't converge). A deterministic framework could fix this.

---

## 6. NQS-SC: Selected Configurations from NQS

**Paper**: "Neural Quantum States Based on Selected Configurations"
**Published**: arXiv:2602.12993 (Feb 2026)
**Authors**: Solanki, Ding, Reiher (ETH Zurich)
**URL**: https://arxiv.org/abs/2602.12993

**Architecture**:
- NQS predicts probability amplitudes for configurations
- Configurations selected dynamically based on predicted amplitudes
- Non-stochastic local energy evaluation
- Systematic improvability (unlike VMC)

**Key Results**:
- NQS-SC significantly outperforms NQS-VMC in energy accuracy
- **Especially strong advantage for molecules with static correlation**
- Robust systematic improvability (VMC does not have this)
- Authors argue NQS-SC should be the **new default** over NQS-VMC

**Critical relevance**: This is exactly our problem. We tried VMC (Phase 4b) and it didn't converge at 24Q+. ETH Zurich's answer: **don't do VMC. Use the NQS to predict important configs, then diagonalize**. This validates our sample-then-diagonalize architecture but suggests the NQS should be used for config selection, not energy minimization.

---

## 7. Physics-Informed Transformers

**Paper**: "Physics-informed Transformers for Electronic Quantum States"
**Journal**: Nature Communications 16, 10811 (2025)
**URL**: https://www.nature.com/articles/s41467-025-66844-z

**Architecture**:
- Modified VMC framework with physics-informed basis construction
- Reference state (HF or strong-coupling limit) as starting point
- Transformer autoregressively samples **corrections** to reference state
- Interpretable hidden representations that capture energetic ordering

**Key Innovation**: Instead of learning the full wavefunction from scratch, the Transformer learns **deviations from a known reference**. This is inherently more efficient for molecular systems where HF is already a reasonable approximation.

**Relevance to our pipeline**: We could restructure our AR transformer to learn corrections to the Direct-CI basis rather than sampling configs from scratch. The transformer would learn "given the HF+S+D basis, which triples/quadruples should I add?"

---

## 8. Reinforcement Learning CI (RLCI)

**Paper**: "Reinforcement Learning Configuration Interaction"
**Journal**: J. Chem. Theory Comput. 2021, 17, 9, 5258-5270
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.1c00010
**Code**: https://github.com/jjgoings/rlci

**Architecture**:
- CI problem mapped to sequential decision-making (MDP)
- Q-learning agent decides which determinants to include/exclude
- Off-policy learning allows physics-inspired search policies
- Learns optimal state-action value function

**Key Results**:
- Near-FCI accuracy with compressed wavefunctions
- Agent learns on-the-fly which determinants matter

**Assessment**: Interesting conceptually but limited scalability demonstrated. No follow-up papers with large-scale results (>20 orbitals). Q-learning becomes intractable when action space is exponential. More of a proof-of-concept than a practical tool at 40Q scale.

---

## 9. Machine Learning Assisted SCI (ML-ASCI)

**Paper**: "Machine Learning Assisted Selective Configuration Interaction for Accurate Ground and Excited State Calculations"
**Journal**: J. Chem. Theory Comput. 2026, 22, 4, 1664-1673
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01652

**Architecture**:
- Perturbative SCI guided by binary ML classifier
- Lightweight feedforward neural network (fast inference)
- Designed for both ground and excited states

**Relevance**: Very similar to our NNCI module. The emphasis on "lightweight FNN for fast inference" aligns with our design. The extension to excited states is notable -- our pipeline only targets ground states.

---

## 10. AB-SND: Adaptive-Basis Neural Diagonalization

**Paper**: "Adaptive-basis sample-based neural diagonalization for quantum many-body systems"
**Published**: arXiv:2508.12724 (Aug 2025)
**URL**: https://arxiv.org/abs/2508.12724

**Architecture**:
- Autoregressive NN samples basis configurations
- **Additionally optimizes a parameterized basis transformation**
- The basis transformation concentrates the ground state, making sampling easier
- Classes: single-spin rotations, two-spin rotations, global unitaries

**Key Innovation**: Instead of sampling better in a fixed basis, AB-SND changes the basis to make sampling easier. This is orthogonal to all other approaches.

**Relevance**: We use canonical HF orbitals. Natural orbitals or optimized orbitals could make our NF more effective. NNCI (FAU) showed natural orbitals help significantly. AB-SND goes further with learned basis transformations.

---

## 11. Diffusion Models

### For Discrete Configuration Sampling

**Paper**: "Discrete generative diffusion models without stochastic differential equations: A tensor network approach"
**Journal**: Phys. Rev. E 111, 025302 (2025)
**URL**: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.025302

**Status**: Early-stage research. Discrete diffusion models for quantum lattice systems exist but have NOT been applied to molecular configuration sampling (Slater determinants in second quantization).

**Key Problem**: Standard diffusion models operate in continuous space. Molecular configurations are discrete (binary occupation vectors). Adapting diffusion to discrete spaces requires either:
- Continuous relaxation (losing physical meaning)
- Discrete noise processes (mask/absorb transitions)

**Assessment**: **No evidence of diffusion models being used for Slater determinant generation as of March 2026.** The field has moved toward autoregressive transformers instead. Diffusion models are being used for 3D molecular geometry generation (DiffMC-Gen, etc.) but not for electronic configuration sampling.

**Why diffusion models are unlikely to help**:
1. Particle conservation is hard to enforce in diffusion (noising destroys electron count)
2. Autoregressive models naturally enforce constraints via masked sampling
3. Diffusion requires many denoising steps; autoregressive generates in one pass
4. No interpretability of intermediate noisy configs

---

## 12. Foundation NQS Models

**Paper**: "Foundation Neural-Networks Quantum States as a Unified Ansatz for Multiple Hamiltonians"
**Journal**: Nature Communications (Aug 2025)
**URL**: https://www.nature.com/articles/s41467-025-62098-x

**Architecture**:
- Single versatile architecture processes multimodal inputs (spin configs + Hamiltonian couplings)
- Generalizes to Hamiltonians **not seen during training**
- Enables fidelity susceptibility estimation for phase transition detection

**Relevance**: Foundation models for NQS are an emerging trend. A pre-trained foundation NQS could provide better initial configs for our pipeline across different molecules. Currently our NF trains from scratch for each molecule.

---

## 13. Hybrid Tensor Network + NQS

**Paper**: "Hybrid Tensor Network and Neural Network Quantum States for Quantum Chemistry"
**Journal**: J. Chem. Theory Comput. 2025, 21, 20, 10252-10262
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01228

**Architecture**: Combines MPS/DMRG tensor network states with neural network corrections.

**Key Results**:
- Accuracy surpassing both DMRG and CCSD
- Particularly effective in strongly correlated regimes

**Relevance**: If DMRG can provide a good initial state for strongly correlated systems, the NQS only needs to learn corrections. This echoes the physics-informed transformer approach (Section 7).

---

## 14. Critical Assessments of SQD/QSCI

### "Critical Limitations in QSCI Methods" (Reinholdt et al.)

**Journal**: J. Chem. Theory Comput. 2025
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.5c00375

**Key Findings**:
1. **Sampling inefficiency**: Repeatedly selects already-seen configurations
2. **Less compact expansions** than classical heuristic SCI (HCI/CIPSI)
3. **Inescapable trade-off**: finding enough determinants vs keeping the expansion compact
4. QSCI is overall **more expensive** than classical SCI

**Implication for us**: This critique applies equally to our NF-guided approach. Sampling from |psi|^2 (or approximations thereof) is inherently less efficient than Hamiltonian-guided deterministic selection. The winning approaches (HAAR-SCI, NNCI, CIGS) all use H-coupling information, not just wavefunction amplitudes.

### IBM SqDRIFT (Science, March 2026)

**URL**: https://www.science.org/doi/10.1126/sciadv.adu9991

IBM used 72 qubits to characterize a half-Mobius molecule. The quantum computer provides samples; classical post-processing (SQD) does the heavy lifting. This is quantum hardware-dependent and not directly comparable to our classical pipeline, but validates the sample-then-diagonalize paradigm at scale.

### ph-AFQMC + SQD

**Journal**: J. Chem. Theory Comput. 2025
**URL**: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01407

Phaseless auxiliary-field QMC using truncated SQD trial wavefunctions recovers O(100) mHa of correlation energy for N2 and [2Fe-2S]. Shows that even approximate subspace wavefunctions are useful as trial states for more sophisticated methods.

---

## 15. Large-Scale HPC Results

### Sunway Supercomputer (IEEE 2025)

**URL**: https://ieeexplore.ieee.org/abstract/document/11204692

- NNQS-Transformer on **120 spin orbitals** (largest AI-driven quantum chemistry calculation)
- 37 million processor cores, 92% strong scaling
- Validates that transformer-based NQS scales to exascale hardware

### NNQS-Transformer

**URL**: https://arxiv.org/abs/2306.16705

- Data-parallel implementation
- Up to 120 spin orbitals validated
- Self-attention captures long-range correlations

---

## 16. Comparative Analysis & Recommendations

### Method Comparison Table

| Method | Architecture | Scale Demonstrated | H-Guided? | Iterative? | Our Relevance |
|--------|-------------|-------------------|-----------|-----------|---------------|
| **HAAR-SCI** | Gated Transformer + Gumbel Top-K | 116 spin orbs, [2Fe-2S] | **YES** (H-coupling filter) | No (single pass) | **HIGHEST** |
| **GTNN-SCI** | Transformer self-attention | cc-pVDZ, [2Fe-2S] | **YES** (SCI framework) | No | **HIGHEST** |
| **QiankunNet** | Decoder Transformer + MCTS | 30 spin orbs | No (VMC) | No | MEDIUM |
| **NNCI (FAU)** | CNN classifier | N2 (10^5 dets) | Indirect (diag feedback) | **YES** | **HIGH** |
| **CIGS** | RBM + taboo list | FCI-level molecules | Indirect | **YES** | HIGH |
| **PIGen-SQD** | RBM + PT2 guidance | 52 qubits (hardware) | **YES** (PT2) | **YES** | MEDIUM |
| **NQS-SC (ETH)** | NQS amplitude prediction | Molecular systems | Indirect | **YES** | HIGH |
| **Physics-Informed Transformer** | Transformer corrections | Model Hamiltonians | **YES** (reference state) | No | MEDIUM |
| **ML-ASCI** | FNN classifier | Ground + excited | Indirect (perturbative) | **YES** | MEDIUM |
| **AB-SND** | AR NN + basis optimization | Many-body | No | No | LOW |
| **RLCI** | Q-learning | Small molecules | Indirect | **YES** | LOW |
| **Diffusion Models** | Discrete diffusion | N/A for this problem | No | No | **NONE** |
| **Our Pipeline** | AR Transformer + SKQD | 40Q (CAS(10,20)) | No | No | -- |

### Root Cause Analysis: Why Our NF Fails at 40Q

From this survey, the diagnosis is clear:

1. **No Hamiltonian guidance during generation**: HAAR-SCI and GTNN-SCI succeed because they filter generated configs by H-coupling strength. Our NF generates configs based on learned p(x), then hands them to SKQD. The NF has no access to H during sampling.

2. **Single-shot pipeline**: NNCI, CIGS, and NQS-SC all use iterative loops: generate -> diagonalize -> learn from results -> generate better. Our pipeline is one-pass.

3. **Krylov expansion makes NF redundant**: SKQD's `get_connections` already discovers all single/double excitations from existing configs. At STO-3G and even cc-pVDZ, this is sufficient. NF would only help if it found important **triples/quadruples** that Krylov expansion misses -- but our NF doesn't know which triples/quadruples are important because it has no H-coupling information.

4. **VMC training doesn't converge at scale**: Our Phase 4b VMC trainer fails at 24Q+. ETH Zurich (NQS-SC) confirms this is a general VMC problem and recommends selected-configuration approaches instead.

### Recommended Architecture Changes (Priority Order)

#### P0: Hamiltonian-Guided Filtering (HAAR-SCI style)
**Impact**: Transformative. **Effort**: Medium.

Add H-coupling filtering after NF sampling:
```
NF generates N configs -> compute H_ij between new configs and existing basis
-> keep only configs with |H_ij| > threshold -> feed to SKQD
```
This is essentially what HAAR-SCI does with GPU min-heap kernels. We already have `get_connections()` infrastructure.

#### P1: Iterative Refinement Loop (NNCI-style)
**Impact**: High. **Effort**: Medium.

Convert pipeline from single-pass to iterative:
```
for iteration in range(max_iter):
    1. Generate candidate configs (NF or combinatorial)
    2. Score candidates (NNCI classifier or H-coupling)
    3. Add top-k to basis
    4. Diagonalize (SKQD)
    5. Update NF/classifier with diag results
    6. Check energy convergence
```
We already have `src/krylov/nnci.py` (PR-B1). The gap is integrating it into the main pipeline loop rather than running it as a pre-processing step.

#### P2: Physics-Informed Reference Corrections
**Impact**: Medium. **Effort**: Low.

Instead of training NF to learn full p(x), train it to learn corrections to Direct-CI:
```
P(config) = P_DirectCI(config) * correction(config; theta)
```
The transformer learns "which triples/quadruples to add to the CISD basis" rather than "which configs are important from scratch."

#### P3: Natural Orbital Basis
**Impact**: Medium. **Effort**: Low.

NNCI (FAU) showed natural orbitals consistently reduce required determinants. Replace canonical HF orbitals with approximate natural orbitals from MP2 or CISD density matrix. We already compute MP2 amplitudes (PR 3.2) -- computing the MP2 natural orbitals is trivial.

#### P4: Taboo List for Krylov Expansion
**Impact**: Low-Medium. **Effort**: Very Low.

CIGS's taboo list prevents re-evaluating rejected configs. In our Krylov expansion, `get_connections()` rediscovers the same configs repeatedly. A hash set of previously-evaluated-and-rejected configs would save significant computation.

### What NOT to Pursue

1. **Diffusion models**: No evidence of applicability to discrete configuration sampling. Particle conservation is fundamentally hard to enforce.
2. **Pure VMC at scale**: ETH Zurich, our own results, and multiple 2025 papers confirm VMC doesn't converge reliably for strongly correlated molecules at >20 orbitals.
3. **Bigger NF without H-guidance**: Adding more parameters/layers to our AR transformer won't help. The problem isn't model capacity -- it's that the NF doesn't know which configs are important from H's perspective.
4. **RLCI**: No evidence of scaling beyond small molecules. Q-learning in exponential action spaces is fundamentally limited.

---

## Key Papers Quick Reference

| # | Paper | Venue | Date | arXiv/DOI |
|---|-------|-------|------|-----------|
| 1 | HAAR-SCI | JCTC 21(24) | Dec 2025 | 10.1021/acs.jctc.5c01415 |
| 2 | GTNN-SCI | JCTC 21(23) | Dec 2025 | 10.1021/acs.jctc.5c01429 |
| 3 | QiankunNet | Nature Comms | 2025 | 10.1038/s41467-025-63219-2 |
| 4 | NNCI (FAU) | JCTC | 2025 | 10.1021/acs.jctc.4c01479 |
| 5 | NO-NNCI | arXiv | Oct 2025 | 2510.27665 |
| 6 | CIGS (RBM) | JCTC | 2025 | arXiv:2409.06146 |
| 7 | PIGen-SQD | arXiv | Dec 2025 | 2512.06858 |
| 8 | Deterministic NQS | arXiv | Jan 2026 | 2601.21310 |
| 9 | NQS-SC (ETH) | arXiv | Feb 2026 | 2602.12993 |
| 10 | Physics-Informed Transformer | Nature Comms | 2025 | 10.1038/s41467-025-66844-z |
| 11 | RLCI | JCTC 17(9) | 2021 | 10.1021/acs.jctc.1c00010 |
| 12 | ML-ASCI | JCTC 22(4) | 2026 | 10.1021/acs.jctc.5c01652 |
| 13 | AB-SND | arXiv | Aug 2025 | 2508.12724 |
| 14 | QSCI Critique | JCTC | 2025 | 10.1021/acs.jctc.5c00375 |
| 15 | SqDRIFT (IBM) | Science Advances | Mar 2026 | 10.1126/sciadv.adu9991 |
| 16 | ph-AFQMC + SQD | JCTC | 2025 | 10.1021/acs.jctc.5c01407 |
| 17 | Foundation NQS | Nature Comms | Aug 2025 | 10.1038/s41467-025-62098-x |
| 18 | Hybrid TN+NQS | JCTC 21(20) | 2025 | 10.1021/acs.jctc.5c01228 |
| 19 | Sunway NNQS | IEEE TPDS | 2025 | 10.1109/TPDS.2025.3554372 |
| 20 | HSB-QSCI | PCCP | 2025 | 10.1039/D5CP02202A |

---

## Sources

- [HAAR-SCI - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415)
- [GTNN-SCI - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01429)
- [QiankunNet - Nature Communications](https://www.nature.com/articles/s41467-025-63219-2)
- [NNCI - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01479)
- [Natural-Orbital NNCI - arXiv](https://arxiv.org/abs/2510.27665)
- [NNCI Rydberg States - arXiv](https://arxiv.org/abs/2510.26751)
- [CIGS RBM - arXiv](https://arxiv.org/abs/2409.06146)
- [PIGen-SQD - arXiv](https://arxiv.org/html/2512.06858)
- [Deterministic NQS Framework - arXiv](https://arxiv.org/abs/2601.21310)
- [NQS-SC ETH Zurich - arXiv](https://arxiv.org/abs/2602.12993)
- [Physics-Informed Transformers - Nature Communications](https://www.nature.com/articles/s41467-025-66844-z)
- [RLCI - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00010)
- [ML-ASCI - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01652)
- [AB-SND - arXiv](https://arxiv.org/abs/2508.12724)
- [Critical Limitations QSCI - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00375)
- [SqDRIFT IBM - Science Advances](https://www.science.org/doi/10.1126/sciadv.adu9991)
- [ph-AFQMC + SQD - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01407)
- [Foundation NQS - Nature Communications](https://www.nature.com/articles/s41467-025-62098-x)
- [Hybrid TN+NQS - JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01228)
- [Sunway NNQS 120 Orbitals - IEEE](https://ieeexplore.ieee.org/abstract/document/11204692)
- [HSB-QSCI - PCCP](https://pubs.rsc.org/en/content/articlehtml/2025/cp/d5cp02202a)
- [Discrete Diffusion Tensor Network - Phys Rev E](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.025302)
- [IBM Half-Mobius Molecule](https://research.ibm.com/blog/half-mobius-molecule)
- [Normalizing Flow Framework - arXiv](https://arxiv.org/abs/2406.00047)
- [Autoregressive NQS - Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00461-z)
