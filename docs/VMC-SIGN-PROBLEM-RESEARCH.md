# VMC Sign Problem & Wavefunction Sign Structure: Research Report

> **Date**: 2026-03-09
> **Focus**: Why our VMC + sign network doesn't converge at 24-40Q, and what the field does instead
> **Our architecture**: psi(x) = sqrt(p(x)) * s(x), autoregressive flow for p(x), feedforward NN -> tanh for s(x), REINFORCE gradient

---

## Executive Summary

Our architecture has **three fundamental problems**:

1. **REINFORCE is the wrong optimizer** -- The entire field uses Stochastic Reconfiguration (SR) / natural gradient, not REINFORCE. SR respects the quantum geometric tensor (Fisher information matrix) and is essential for VMC convergence. REINFORCE has high variance and ignores the parameter-space geometry.

2. **Separate amplitude x sign is the wrong factorization** -- State-of-the-art methods (FermiNet, PsiFormer, NNBF, QiankunNet) do NOT use psi = sqrt(p) * s. They either (a) use determinant-based architectures where the sign emerges naturally from antisymmetry, or (b) use amplitude + phase (not sign) with psi = |psi| * e^{i*phi} and a dedicated phase MLP.

3. **The sign network architecture is too weak** -- A feedforward NN -> tanh cannot capture the sign structure of strongly correlated wavefunctions. The sign structure is fundamentally tied to fermionic antisymmetry (Slater-Condon rules, nodal surfaces). Successful approaches embed this physics into the architecture via determinants or backflow.

---

## 1. How State-of-the-Art Methods Handle Sign/Phase

### 1.1 FermiNet / PsiFormer (Google DeepMind) -- First Quantization

**Architecture**: Permutation-equivariant NN -> Slater determinant layer

- The sign is NOT learned by a separate network
- Instead, FermiNet outputs configuration-dependent orbitals via a deep NN (the "generalized backflow")
- These orbitals are assembled into Slater determinants: det(phi_i(r_j))
- The determinant AUTOMATICALLY provides the correct sign under electron exchange
- Multiple determinants are summed: psi = sum_k w_k * det(Phi_k)
- Implementation detail: sign and log|det| computed via slogdet for numerical stability
- PsiFormer replaces the equivariant NN layers with self-attention (Transformer), achieving higher accuracy

**Key insight**: The sign emerges from the mathematical structure of determinants. No separate sign network needed.

**Scale**: Benzene (42 electrons), excited states on butadiene and tetrazine. NES-VMC (2024, Science) achieves chemical accuracy for excited states.

**Ref**: [FermiNet GitHub](https://github.com/google-deepmind/ferminet), [NES-VMC (Science 2024)](https://www.science.org/doi/10.1126/science.adn0137)

### 1.2 QiankunNet (Nature Communications 2025) -- Second Quantization

**Architecture**: Decoder-only Transformer (autoregressive) + separate Phase MLP

- Wavefunction: psi(x) = |psi(x)| * e^{i*phi(x)}
- The Transformer generates |psi(x)| autoregressively (conditional probabilities for occupancy)
- A separate MLP generates the PHASE phi(x), NOT binary sign
- Phase is continuous: phi in [0, 2*pi), so sign = cos(phi) is a special case
- Quaternary states: {empty, spin-down, spin-up, doubly-occupied} per orbital
- BOS token starts the autoregressive chain
- Masked attention ensures causal ordering

**Key insight**: Phase (continuous) is strictly more expressive than sign (binary). The phase MLP sees the full configuration and outputs a continuous angle.

**Scale**: 30 spin orbitals, 99.9% FCI energy recovery. Extended to solids (QiankunNet-Solid, 2025).

**Ref**: [QiankunNet (Nature Comms 2025)](https://www.nature.com/articles/s41467-025-63219-2)

### 1.3 Neural Network Backflow (NNBF) -- Second Quantization

**Architecture**: MLP generates configuration-dependent orbitals -> Slater determinant

- In occupancy basis: x = (x_1, ..., x_N) where x_i in {0,1}
- An MLP takes the FULL occupation vector and outputs modified orbital coefficients
- These are assembled into a Slater determinant
- The sign comes from the determinant, not a separate network
- Universal approximation: NNBF can approximate ANY wavefunction on finite config space (proven)
- Recent work (arXiv:2403.03286, PRB 2024): NNBF surpasses CCSD and CCSD(T) for molecules

**Key insight for our project**: NNBF is the closest analog to what we need. It works in second quantization (occupancy basis, like our pipeline) and handles the sign via determinantal structure. It is proven to be a universal approximator.

**Ref**: [NNBF for quantum chemistry (PRB 2024)](https://link.aps.org/doi/10.1103/PhysRevB.110.115137), [Efficient NNBF optimization (arXiv:2502.18843)](https://arxiv.org/abs/2502.18843)

### 1.4 Transformer Backflow (arXiv:2509.25720) -- Second Quantization

**Architecture**: Transformer backbone + backflow orbitals -> determinant

- Electronic configurations processed as token sequences
- Self-attention layers learn non-local orbital correlations
- Token-specific NNs map contextual representations to backflowed orbitals
- Determinant provides the sign structure
- **Achieved chemical accuracy on [2Fe-2S] CAS(30,20)** -- matching DMRG
- This is our target system!

**Key insight**: The most successful method for CAS(30,20) uses Transformer + backflow + determinant. NOT amplitude x sign.

**Ref**: [Transformer Backflow (arXiv:2509.25720)](https://arxiv.org/abs/2509.25720)

### 1.5 NAQS (Barrett et al., Nature Machine Intelligence 2022) -- Second Quantization

**Architecture**: Autoregressive NN with embedded physical priors

- Direct autoregressive sampling: P(config) = prod P(x_i | x_{<i})
- Physical priors: particle number conservation, spin symmetry
- Complex-valued output: log psi = log|psi| + i*phi
- Demonstrated on molecules up to 30 spin orbitals
- Orders of magnitude more determinants than previous NQS methods

**Key insight**: Autoregressive structure + physics priors + complex output. No separate sign network.

**Ref**: [NAQS (Nature Machine Intelligence 2022)](https://www.nature.com/articles/s42256-022-00461-z)

---

## 2. Why REINFORCE Fails and SR is Essential

### 2.1 The Problem with REINFORCE

REINFORCE computes the energy gradient as:

    grad_theta E = E_p[ (E_loc(x) - E) * grad_theta log p(x) ]

This is a score-function estimator with:
- **High variance**: proportional to |E_loc - E|^2, which can be enormous for molecular systems
- **No curvature information**: treats all parameter directions equally
- **Cartesian metric**: uses Euclidean distance in parameter space, which is physically meaningless

For molecular wavefunctions, E_loc(x) = <x|H|psi> / <x|psi> can have wild fluctuations, making REINFORCE practically useless for convergence.

### 2.2 Stochastic Reconfiguration (SR) = Natural Gradient

SR computes the parameter update as:

    delta_theta = S^{-1} * f

where S is the quantum geometric tensor (covariance of log-derivatives) and f is the energy gradient force. This is equivalent to natural gradient descent on the quantum state manifold.

**Why SR works**:
- Respects the geometry of the wavefunction manifold
- Small steps in Hilbert space, not parameter space
- Originally developed to address the sign problem (Sorella 2001)
- The Monte Carlo average sign remains finite and stable with SR
- Asymptotically converges to the correct ground state

### 2.3 SR Scalability (2024-2026 Advances)

The original SR requires inverting an N_params x N_params matrix -- O(N^3) and infeasible for large networks.

**MinSR** (Nature Physics 2024): Reduces cost to O(N_params) linear. Reformulates SR update as minimal-norm solution of an overdetermined least-squares problem. Enables training networks with 10^6 parameters and 64 layers.

**SPRING** (JCP 2024): Combines MinSR with Kaczmarz method. Outperforms both MinSR and KFAC across atoms and molecules.

**IRL** (arXiv:2601.01437, January 2026): Implicitly Restarted Lanczos for NQS training. Recasts parameter update as Hermitian eigenvalue problem. Enables chemically-accurate shallow NQS. Addresses hyperparameter sensitivity that plagues Adam/REINFORCE.

**Ref**: [MinSR (Nature Physics 2024)](https://www.nature.com/articles/s41567-024-02566-1), [SPRING (arXiv:2401.10190)](https://arxiv.org/abs/2401.10190), [IRL (arXiv:2601.01437)](https://arxiv.org/abs/2601.01437)

### 2.4 Implication for Our Pipeline

Our VMC trainer uses REINFORCE. This is fundamentally inadequate:
- High-variance gradient estimates prevent convergence
- No curvature preconditioning means the optimizer wanders in irrelevant directions
- The sign network receives essentially noise as gradient signal

**Minimum fix**: Replace REINFORCE with SR (or MinSR for scalability).
**Better fix**: Adopt the full NNBF or QiankunNet-style architecture with determinant-based sign handling.

---

## 3. The Sign Problem in Second Quantization

### 3.1 Why Sign Structure is Hard

The wavefunction sign structure in second quantization is determined by:
1. **Fermionic antisymmetry**: exchanging two electrons flips sign
2. **Jordan-Wigner signs**: from the JW transformation to qubits
3. **Correlation-induced sign changes**: beyond mean-field, the ground state has nontrivial nodal structure

For strongly correlated systems (our target), the sign structure is complex and non-local. A simple feedforward NN cannot capture it because:
- The sign depends on the GLOBAL configuration (all orbitals)
- It follows from the antisymmetry of the many-body wavefunction
- The number of sign changes grows combinatorially with system size

### 3.2 Key Paper: "Neural network wave functions and the sign problem" (PRR 2020)

Szabo & Castelnovo showed that:
- CNNs fail to converge on frustrated systems with nontrivial sign structure
- An explicit, interpretable phase ansatz can fix this
- BUT: even with good phase ansatz, the optimizer can get stuck in Marshall-sign-rule states
- The sign problem is NOT just an architecture problem -- it's also an optimization problem

**Ref**: [Szabo & Castelnovo (PRR 2020)](https://arxiv.org/abs/2002.04613)

### 3.3 Key Paper: "Improving neural network performance for solving quantum sign structure" (PRB 2025)

Ou, Huang, and Ozolins (October 2025) propose:
- Modified SR with different imaginary time steps for amplitude vs. phase
- Larger time step for phase optimization
- Enables simultaneous training of phase and amplitude networks
- Demonstrated on Heisenberg J1-J2 model

**Key insight**: Phase/sign needs MORE aggressive optimization than amplitude. Our current approach treats them equally.

**Ref**: [Ou et al. (arXiv:2510.02051)](https://arxiv.org/abs/2510.02051)

### 3.4 "From Wavefunction Sign Structure to Static Correlation" (arXiv:2511.01569)

Dubecky (November 2025) reframes static correlation as a quantitative gauge of the fermion-sign problem. The variational energy gap between mean-field and exact nodal surfaces defines a "nodal penalty" that measures the intrinsic complexity of fermionic correlations. This is method-agnostic.

**Implication**: For our target systems (CAS(10,10)+), the sign structure is inherently complex. A simple NN cannot avoid this -- the architecture must encode the physics.

---

## 4. Alternative Architectures for Sign/Phase

### 4.1 Determinant-Based (Recommended for Our Case)

Replace psi = sqrt(p) * s with psi = sum_k w_k * det(Phi_k(x)):

- MLP takes configuration x and outputs orbital matrix Phi_k
- Determinant provides correct sign structure
- Weights w_k are learnable
- NNBF (Liu & Clark, PRB 2024) proves universal approximation in 2nd quantization
- Efficient optimization via compact subspace + truncated local energy (arXiv:2502.18843)

**Pros**: Physically motivated, proven universal, handles sign automatically
**Cons**: det() is O(n_orb^3), multiple determinants increase cost

### 4.2 Complex-Valued Output (Phase Network)

Replace s(x) in {-1, +1} with e^{i*phi(x)}:

- psi(x) = |psi(x)| * e^{i*phi(x)}
- Phase MLP outputs continuous phi in [0, 2*pi)
- More expressive than binary sign
- Used by QiankunNet (Nature Comms 2025)
- Gradient is well-defined (no discrete jump)

**Pros**: Continuous gradient, more expressive than binary sign
**Cons**: Still needs good optimization (SR, not REINFORCE)

### 4.3 Transformer + Backflow (State of the Art)

Combine Transformer attention with backflow orbital reparameterization:

- arXiv:2509.25720: Chemical accuracy on [2Fe-2S] CAS(30,20)
- Configuration tokens -> self-attention -> backflow orbitals -> determinant
- The Transformer captures non-local orbital correlations
- The determinant handles antisymmetry and sign

**Pros**: Best results on strongly correlated systems, scales to 30e/20o
**Cons**: Complex implementation, requires distributed VMC

### 4.4 Physics-Informed Reference + Corrections (December 2024)

arXiv:2412.12248: Use HF or strong-coupling state as reference, Transformer learns corrections:

- psi = psi_ref + delta_psi (Transformer)
- The reference provides baseline sign structure
- Corrections are smaller and easier to learn
- More interpretable than learning from scratch

**Pros**: Leverages existing quantum chemistry, interpretable
**Cons**: Quality depends on reference state

### 4.5 Pfaffian-Based (Alternative to Determinant)

Neural Network-Augmented Pfaffian (arXiv:2507.10705, July 2025):

- Pfaffian generalizes determinant to electron pairs
- More flexible than Slater determinant (superset)
- Still provides antisymmetry
- O(N^3) but with larger basis

**Pros**: More expressive than single determinant
**Cons**: Higher computational cost

### 4.6 NQS-SC: Selected Configurations (February 2026)

arXiv:2602.12993: Instead of VMC sampling, select configurations directly:

- NQS predicts amplitudes on full Hilbert space
- Select high-probability configs (like our pipeline does!)
- Diagonalize in selected subspace
- More robust than VMC for static correlation
- "NQS-SC should be the default approach over NQS-VMC"

**Key insight**: This is essentially what our pipeline already does (NF sampling -> SKQD diag). The problem is not VMC per se -- it's the sign/amplitude factorization.

**Ref**: [NQS-SC (arXiv:2602.12993)](https://arxiv.org/abs/2602.12993)

---

## 5. Optimization Methods Beyond REINFORCE

### 5.1 Comparison Table

| Method | Cost per step | Convergence | Sign handling | Used by |
|--------|--------------|-------------|---------------|---------|
| REINFORCE | O(N_params) | Poor | Very poor | Our pipeline |
| Adam | O(N_params) | Moderate | Poor | Baselines |
| SR (full) | O(N_params^3) | Excellent | Good | NetKet, classic NQS |
| MinSR | O(N_params) | Excellent | Good | Deep NQS (10^6 params) |
| SPRING | O(N_params) | Best | Good | State-of-the-art |
| KFAC | O(N_params) | Good | Moderate | FermiNet/PsiFormer |
| IRL | O(N_params * k) | Excellent | Good | Shallow NQS (2026) |

### 5.2 Implementation in NetKet

NetKet provides production-ready SR implementation:
- `nk.driver.VMC` with `nk.optimizer.SR(diag_shift=0.01)`
- Supports second-quantized Hamiltonians
- Fermionic operators: `nk.operator.fermion.{c, cdag, nc}`
- Built-in backflow models
- JAX-based, GPU-accelerated
- Latest version: NetKet 3.x (2024-2025)

**Tutorial**: [NetKet Lattice Fermions](https://netket.readthedocs.io/en/latest/tutorials/lattice-fermions.html)

### 5.3 RetNet Alternative (SandboxAQ, November 2024)

arXiv:2411.03900: Retentive Network (RetNet) as alternative to Transformer:
- O(n) inference (recurrent) vs O(n^2) for Transformer
- Variational Neural Annealing as training strategy
- Overcomes expressiveness gap vs Transformer
- Published in ML: Science and Technology (2025)

---

## 6. Latest Scale Records (2025-March 2026)

| Method | System | Scale | Accuracy | Date |
|--------|--------|-------|----------|------|
| Transformer Backflow | [2Fe-2S] CAS(30,20) | 40 spin orbs | Chemical accuracy vs DMRG | Sep 2025 |
| QiankunNet | 30 spin orbitals | 30 spin orbs | 99.9% FCI | 2025 |
| NNQS-SCI (Sunway) | 120 spin orbitals | 120 spin orbs | Competitive | 2024-2025 |
| NNBF (efficient) | Molecules | ~50 orbitals | Beats CCSD(T) | Feb 2025 |
| NES-VMC/PsiFormer | Benzene excitations | 42 electrons | Chemical accuracy | 2024 |
| RetNet NQS | Molecular | Competitive | Faster than Transformer | Nov 2024 |
| NQS-SC | Molecules | Various | Better than NQS-VMC | Feb 2026 |
| IRL NQS | Atoms/molecules | Shallow nets | Chemical accuracy | Jan 2026 |

---

## 7. Recommendations for Our Pipeline

### Priority 1: Replace REINFORCE with SR/MinSR (CRITICAL)

Our VMC trainer MUST use stochastic reconfiguration, not REINFORCE. Without SR, the sign network receives garbage gradients and cannot converge. This is the single biggest reason our VMC doesn't work at 24-40Q.

Implementation path:
1. Compute log-derivative matrix O_ij = d log psi / d theta_j for sample x_i
2. Compute S = <O^dagger O> - <O^dagger><O> (covariance matrix)
3. Compute f = <O^dagger E_loc> - <O^dagger><E_loc> (force vector)
4. Solve S * delta_theta = f (with regularization: S + epsilon * I)
5. For large networks, use MinSR (linear cost)

### Priority 2: Replace sign network with determinant-based approach

Options in order of implementation difficulty:

**Option A** (Easiest): Replace s(x) with e^{i*phi(x)} using phase MLP
- Change tanh output to continuous phase angle
- psi(x) = sqrt(p(x)) * e^{i*phi(x)}
- Still need SR for optimization
- Quick win but not fundamental fix

**Option B** (Moderate): NNBF in second quantization
- MLP takes occupation vector -> outputs orbital coefficients
- Assemble into Slater determinant(s)
- Sign comes from det() automatically
- Proven universal approximator
- Most aligned with our occupancy-basis representation

**Option C** (Best but hardest): Transformer Backflow
- Replace our autoregressive flow with Transformer backflow
- Self-attention on orbital tokens -> backflow orbitals -> determinant
- State-of-the-art for CAS(30,20)
- Significant rewrite required

### Priority 3: Consider NQS-SC approach

Our pipeline already does NF sampling -> SKQD diagonalization. This is conceptually similar to NQS-SC (arXiv:2602.12993), which was shown to be superior to NQS-VMC for static correlation. Rather than trying to make VMC converge, we could:
1. Train the autoregressive flow to approximate |psi|^2 (which it already does)
2. Use the flow to generate candidate configurations
3. Skip VMC entirely -- go directly to SKQD diagonalization
4. The sign structure is handled by the Hamiltonian matrix elements in SKQD

This validates our existing architecture choice: Direct-CI + SKQD may be the right approach, and VMC with sign learning may be unnecessary overhead.

### Priority 4: Differential time steps for amplitude vs. phase

If we keep a separate phase/sign network, use the Ou et al. (2025) insight: larger optimization steps for the phase network than for the amplitude network. The phase landscape is harder and needs more aggressive optimization.

---

## 8. Key Papers Reference List

### Architecture & Sign Problem
1. Szabo & Castelnovo, "Neural network wave functions and the sign problem", PRR 2020 ([arXiv:2002.04613](https://arxiv.org/abs/2002.04613))
2. Li et al., "Representational power of selected NQS in second quantization", JCP 2025 ([arXiv:2511.04932](https://arxiv.org/abs/2511.04932))
3. Gao & Gunnemann, "On Representing Electronic Wave Functions with Sign Equivariant NNs", 2024 ([arXiv:2403.05249](https://arxiv.org/abs/2403.05249))
4. Dubecky, "From Wavefunction Sign Structure to Static Correlation", 2025 ([arXiv:2511.01569](https://arxiv.org/abs/2511.01569))
5. Ou et al., "Improving neural network performance for solving quantum sign structure", PRB 2025 ([arXiv:2510.02051](https://arxiv.org/abs/2510.02051))

### Neural Network Backflow
6. Liu & Clark, "Neural network backflow for ab-initio quantum chemistry", PRB 2024 ([arXiv:2403.03286](https://arxiv.org/abs/2403.03286))
7. Liu & Clark, "Efficient optimization of NNBF for ab-initio quantum chemistry", PRB 2025 ([arXiv:2502.18843](https://arxiv.org/abs/2502.18843))
8. arXiv:2509.25720, "Transformer-Based Neural Networks Backflow for Strongly Correlated Electronic Structure", 2025

### Autoregressive Methods
9. Barrett et al., "Autoregressive neural-network wavefunctions for ab initio quantum chemistry", Nature Machine Intelligence 2022
10. QiankunNet, "Solving the many-electron Schrodinger equation with a transformer-based framework", Nature Communications 2025
11. Knitter et al., "Retentive Neural Quantum States", ML: Sci. Technol. 2025 ([arXiv:2411.03900](https://arxiv.org/abs/2411.03900))

### Optimization Methods
12. Rende et al., "Empowering deep NQS through efficient optimization (MinSR)", Nature Physics 2024 ([arXiv:2302.01941](https://arxiv.org/abs/2302.01941))
13. Nys & Lovato, "A Kaczmarz-inspired approach (SPRING)", JCP 2024 ([arXiv:2401.10190](https://arxiv.org/abs/2401.10190))
14. Liu & Dou, "Implicitly Restarted Lanczos for Chemically-Accurate Shallow NQS", 2026 ([arXiv:2601.01437](https://arxiv.org/abs/2601.01437))

### NQS-SC and Selected CI
15. Dash et al., "Neural Quantum States Based on Selected Configurations", 2026 ([arXiv:2602.12993](https://arxiv.org/abs/2602.12993))
16. Sobral et al., "Physics-informed Transformers for Electronic Quantum States", 2024 ([arXiv:2412.12248](https://arxiv.org/abs/2412.12248))

### FermiNet / PsiFormer / DeepQMC
17. Spencer et al., "Better, Faster Fermionic Neural Networks", NeurIPS 2020
18. von Glehn et al., "A Self-Attention Ansatz for Ab-initio Quantum Chemistry (PsiFormer)", ICLR 2023
19. NES-VMC, "Accurate computation of quantum excited states with neural networks", Science 2024
20. [FermiNet GitHub](https://github.com/google-deepmind/ferminet)
21. [DeepQMC GitHub](https://github.com/deepqmc/deepqmc)

### Frameworks
22. [NetKet](https://www.netket.org/) -- JAX-based NQS toolkit with SR, backflow, fermion operators
23. [NetKet Lattice Fermions Tutorial](https://netket.readthedocs.io/en/latest/tutorials/lattice-fermions.html)

---

## 9. Bottom Line

**Why our VMC doesn't converge**: We use REINFORCE (high variance, no curvature info) to train a feedforward sign network (cannot represent fermionic antisymmetry). Both the optimizer AND the architecture are wrong.

**What the field does instead**: Determinant-based architectures (sign from det() automatically) + SR/MinSR optimization (natural gradient in Hilbert space). This combination is essential -- neither alone suffices.

**Silver lining**: Our pipeline's core strength is SKQD diagonalization, not VMC. The NQS-SC paper (Feb 2026) validates this: selected-configuration + diagonalization outperforms NQS-VMC for static correlation. Our flow can serve as the configuration generator without needing correct signs -- SKQD handles the sign structure through exact Hamiltonian matrix elements.

**Recommended strategy**:
1. Short-term: Keep Direct-CI + SKQD as the primary path (it works)
2. Medium-term: If VMC is needed, implement SR (MinSR) as optimizer
3. Long-term: Replace sign network with NNBF (determinant-based) or adopt Transformer backflow
