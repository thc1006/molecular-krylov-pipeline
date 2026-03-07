# ADR-001 修復清單 (第四輪審計 2026-03-06)

> 6 agents parallel audit. 41 new issues found. Prioritized by severity.
> ALL ITEMS FIXED in ADR-001-scale-to-40-qubits.md on 2026-03-06.
>
> Fixes applied: Phase 0 (PR 0.1, 0.2) added, PR 2.2d added, B11-B17 added,
> 4 new risks added, S7/S8 relaxed, S9 added, Phase 2 ordering fixed,
> PR scopes expanded (1.1, 1.3, 2.2, 2.2b, 4.2, 4.3), numerical corrections
> (param counts, nnz/row, memory, line numbers), references added,
> timeline extended to 26 weeks, file change summary updated.

---

## CRITICAL (must fix before ADR can be approved)

### C1: _topk_log_prob probability model fundamentally wrong
- [x]**ADR fix**: Add new PR 2.2d — "Fix probability model and loss function"
- [x]**ADR fix**: Document that `_topk_log_prob` uses product-of-marginals (incorrect for dependent selections)
- [x]**ADR fix**: Propose replacement: either (a) sequential autoregressive sampling with proper conditional probabilities, or (b) importance-weighted correction using permanent approximation
- [x]**ADR fix**: Add to Risk table with CRITICAL severity
- **File**: `particle_conserving_flow.py:274-295`
- **Impact**: Entire NF training optimizes wrong probability target

### C2: |E|/|S| loss scaling punishes diversity
- [x]**ADR fix**: Add to PR 2.2d scope — replace `|E|/|S|` with `|E|/batch_size` or remove `|S|` denominator entirely
- [x]**ADR fix**: Document the perverse incentive: collapsed flow (|S|=1) gets 1000x more gradient than diverse flow (|S|=1000)
- **File**: `physics_guided_training.py:1087`

### C3: _compute_subspace_energy has no gradient flow
- [x]**ADR fix**: Correct PR 2.2b description — clarify that subspace energy is diagnostic only, not a training signal
- [x]**ADR fix**: If subspace energy IS meant to guide training, PR 2.2b must remove `torch.no_grad()` and implement differentiable eigensolve (or REINFORCE-through-eigensolve)
- **File**: `physics_guided_training.py:775-791`

### C4: SQD S-squared O(n^2) Python loop at n>2000
- [x]**ADR fix**: Add to bottleneck inventory as B11 (CRITICAL for SQD at 40Q)
- [x]**ADR fix**: Expand PR 4.2 scope to include vectorized S^2 computation, or add new PR
- [x]**ADR fix**: Note that at 40Q batch=33K, S^2 computation takes hours
- **File**: `sqd.py:826-840`

### C5: Non-autoregressive NF architecture cannot capture intra-channel correlations
- [x]**ADR fix**: Add to bottleneck inventory as B12 (HIGH for strongly correlated systems)
- [x]**ADR fix**: PR 2.2 must address architecture, not just hidden_dims — propose autoregressive factoring or attention mechanism for orbital scoring
- [x]**ADR fix**: Add to Risk table — "NF architecture fundamentally limited for Cr2/[2Fe-2S]"
- **File**: `particle_conserving_flow.py:232-243`

### C6: Mini-batch memory estimate wrong (430MB -> 1.3GB)
- [x]**ADR fix**: Update PR 1.1 memory analysis — connected configs tensor is (batch × connections × n_sites × 4 bytes), not (batch × connections × 3_scalars)
- [x]**ADR fix**: Recommend batch_size=500 for 40Q (~650 MB peak) instead of 1000

### C7: Phase 2 recommended ordering self-contradictory
- [x]**ADR fix**: Change recommended order from `2.2c -> 2.2b -> 2.1 -> 2.2 -> 2.3` to `2.1 -> 2.2 -> 2.2b -> 2.2c -> 2.3`
- [x]**ADR fix**: Explain why: 2.2c requires skip_nf_training=False (PR 2.1) + entropy reg (PR 2.2b)

### C8: Standard SKQD crashes at 40Q (no MAX_FULL_SUBSPACE guard)
- [x]**ADR fix**: Add to bottleneck inventory as B13
- [x]**ADR fix**: Add to PR 1.1 scope — add `MAX_FULL_SUBSPACE_SIZE` guard to `SampleBasedKrylovDiagonalization._setup_particle_conserving_subspace()` (not just FlowGuidedSKQD)
- **File**: `skqd.py:141` (no guard) vs `skqd.py:942` (has guard)

---

## HIGH (significant impact on project success)

### H1: beta_scorer is dead code (~160K wasted parameters)
- [x]**ADR fix**: Add to PR 2.2 scope — remove `self.beta_scorer` from `ParticleConservingFlow.__init__()`, since `beta_conditioned_scorer` is used instead
- [x]**ADR fix**: Note this frees ~160K parameters (half the model)
- **File**: `particle_conserving_flow.py:186-188`

### H2: GumbelTopK gradient masking (only k/n positions get gradients)
- [x]**ADR fix**: Add to PR 2.2 discussion — `soft_topk = soft * one_hot` zeros out non-selected positions' gradients
- [x]**ADR fix**: Propose fix: use full softmax gradient without masking, or use relaxed top-k (e.g., entmax)
- **File**: `particle_conserving_flow.py:70-75`

### H3: Krylov time evolution also builds dense matrices
- [x]**ADR fix**: Expand PR 1.1 scope to cover `FlowGuidedSKQD._generate_krylov_samples_nf_guided()` line 1117
- [x]**ADR fix**: Note that sparse eigensolver alone is insufficient — the Krylov expansion steps also need sparse H

### H4: NQS REINFORCE loss has implicit 2x learning rate factor
- [x]**ADR fix**: Add as known issue in PR 2.2b — `log_probs = 2 * log_amp` doubles effective NQS learning rate vs standard REINFORCE
- **File**: `physics_guided_training.py:1109-1121`

### H5: Doubles ordering bias duplicated in trainer
- [x]**ADR fix**: Expand PR 1.3 scope to also fix `PhysicsGuidedFlowTrainer._generate_essential_configs()`, not just `pipeline.py`
- **File**: `physics_guided_training.py:286-337` (same bug as `pipeline.py:444`)

### H6: _compute_accumulated_energy builds dense then converts to sparse
- [x]**ADR fix**: Add to PR 1.1 scope — `_compute_accumulated_energy()` should build sparse directly
- **File**: `physics_guided_training.py:1230-1231`

### H7: NF parameter counts wrong in ADR
- [x]**ADR fix**: Correct [256,256] from "319K" to "306K"
- [x]**ADR fix**: Correct [512,512,384,256] from "~1.1M" to "~1.47M"

### H8: matrix_elements_fast line number wrong
- [x]**ADR fix**: Change "line 1120" to "line 1085" throughout ADR

### H9: Regression test should be standalone PR 0.1
- [x]**ADR fix**: Extract regression gate from PR 1.1 acceptance criteria into PR 0.1
- [x]**ADR fix**: Make PR 0.1 a Phase 0 prerequisite (Week 0) before any code changes

### H10: Test files are gitignored
- [x]**ADR fix**: Add note that tests/ should be un-gitignored before Phase 1 begins
- [x]**ADR fix**: Or create tests in a tracked location (e.g., src/tests/)

### H11: Risk of NF producing useless configs not addressed
- [x]**ADR fix**: Add to Risk table — "NF trains successfully but configs don't improve energy beyond Direct-CI"
- [x]**ADR fix**: Mitigation: fallback to importance-sampled CISDT as classical alternative

### H12: max_diag_basis_size=15000 never tuned for 40Q
- [x]**ADR fix**: Add to PR 1.1 or PR 3.1 — tune max_diag_basis_size based on system size (e.g., 50K for 40Q with sparse solver)
- [x]**ADR fix**: Add to adapt_to_system_size() scaling logic

---

## MEDIUM (should fix for completeness)

### M1: Double temperature bug in soft path
- [x]**ADR fix**: Add to PR 2.2 — fix `hard=False` path: remove redundant `/self.temperature` at line 78
- **File**: `particle_conserving_flow.py:78`

### M2: Alpha-beta correlation wastes half input bandwidth
- [x]**ADR fix**: Add to PR 2.2 — `beta_conditioned_scorer` receives `[zeros, alpha_context]`; the zeros are wasted. Feed `alpha_context` directly.

### M3: Factory function defaults diverge from config defaults
- [x]**ADR fix**: Add to PR 2.2b — reconcile `create_physics_guided_trainer()` defaults with `PhysicsGuidedConfig` defaults, or remove factory function

### M4: No CI/CD PR
- [x]**ADR fix**: Add PR 0.2 — minimal GitHub Actions running `pytest tests/test_regression.py`

### M5: Phase 4 timeline unrealistic (3 research features in 4 weeks)
- [x]**ADR fix**: Extend Phase 4 to 8 weeks, or reduce scope to 1 PR (most impactful: SqDRIFT or DMET)
- [x]**ADR fix**: Add contingency buffer of 4 weeks to total timeline (18 -> 26 weeks realistic)

### M6: Missing block2/dmrgscf dependencies
- [x]**ADR fix**: Add `block2` and `pyscf[dmrgscf]` to Section 8.1 (New Dependencies)
- [x]**ADR fix**: Note ARM64/DGX Spark compatibility must be verified

### M7: S6 success criterion (2h wall time) has hard dependency on PR 2.3
- [x]**ADR fix**: Call out explicitly: "S6 requires PR 2.3 (Numba). Without Numba: 28-56h."

### M8: Double excitation row nnz overestimated (100-150 -> 36-50)
- [x]**ADR fix**: Update nnz/row analysis in Section 3.2 — doubles rows in essential-only basis: ~36-50, not 100-150
- [x]**ADR fix**: Update weighted average from "150-500" to "54-128 (essential-only), 300-3000 (with NF triples)"
- [x]**ADR fix**: Update 0.6-6% dense range accordingly

### M9: Larsson 2022 cited for N2 but is Cr2 paper
- [x]**ADR fix**: In PR 3.1 (line 577), separate citations: "Chan et al., ARPC 2011" for N2, "Larsson et al., JACS 2022" for Cr2 only

### M10: Missing Kool et al. 2019 citation
- [x]**ADR fix**: Add "Kool, van Hoof, Welling. Stochastic Beams and Where to Find Them. ICML 2019" to references, cite in PR 2.2

### M11: Missing arXiv:2509.26397 rebuttal paper
- [x]**ADR fix**: Add to Risk table — "Neural scaling laws may not apply to quantum chemistry (arXiv:2509.26397)"
- [x]**ADR fix**: Add to references section

### M12: system_scaler.py does not exist
- [x]**ADR fix**: Remove references to SystemScaler, or note it as planned future work
- [x]**Memory fix**: Update CLAUDE.md and MEMORY.md to remove stale reference

### M13: No end-to-end 40Q integration test
- [x]**ADR fix**: Add to Phase 3 testing strategy — full pipeline run on N2/cc-pVDZ CAS(10,20) as integration test

### M14: PRs 4.2 and 4.3 have no acceptance criteria
- [x]**ADR fix**: Add measurable acceptance criteria for PR 4.2 (e.g., "S-CORE recovery within 1 mHa of noiseless on N2")
- [x]**ADR fix**: Add measurable acceptance criteria for PR 4.3 (e.g., "SqDRIFT converges within 2x wall time of SKQD")

---

## LOW (nice to have)

### L1: SystemScaler vs adapt_to_system_size not reconciled
- [x]Remove stale SystemScaler reference or plan reconciliation

### L2: arXiv:2501.09702 title uses v1
- [x]Update to current title "Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization"

### L3: PIGen-SQD date Dec 2025 vs Jan 2026
- [x]Clarify: "arXiv submitted Dec 2025, published Jan 2026"

### L4: Essential subspace "8K-14K" -> "~7.9K-14K"
- [x]Minor precision fix

### L5: Iteration count ~7.7K misses singles (~8.1K total)
- [x]Update to "~8.1K iterations/config (7.7K doubles + 0.4K singles)"

### L6: NF model memory ~2MB overestimated (~1.2MB)
- [x]Correct memory table

### L7: Eigensolve line reference 804-818 off by 1
- [x]Update to 803-820

---

## Execution Order

Recommended fix order (group by ADR section, minimize context switches):

### Pass 1: Structural additions (new PRs, bottlenecks, risks)
1. Add PR 0.1 (regression gate) and PR 0.2 (CI/CD) — H9, M4
2. Add PR 2.2d (fix probability model + loss function) — C1, C2
3. Add bottlenecks B11 (S^2), B12 (non-autoregressive), B13 (standard SKQD crash) — C4, C5, C8
4. Add missing risks (NF useless configs, scaling laws rebuttal, architecture limit) — H11, M11, C5
5. Fix Phase 2 ordering — C7
6. Extend Phase 4 timeline + add buffer — M5

### Pass 2: Existing PR scope corrections
7. Expand PR 1.1 scope (standard SKQD guard, Krylov steps, accumulated energy, memory fix) — C6, C8, H3, H6
8. Expand PR 1.3 scope (trainer doubles bias) — H5
9. Rewrite PR 2.2 scope (dead beta_scorer, autoregressive, gradient masking, double temp, alpha-beta waste) — H1, H2, C5, M1, M2
10. Correct PR 2.2b description (subspace energy is diagnostic, REINFORCE 2x, factory defaults) — C3, H4, M3
11. Expand PR 4.2 scope (S^2 vectorization) — C4
12. Add PR 4.2/4.3 acceptance criteria — M14

### Pass 3: Numerical corrections
13. Fix parameter counts (306K, 1.47M) — H7
14. Fix line numbers (1085, 803-820) — H8, L7
15. Fix nnz/row estimates (36-50, 54-128) — M8
16. Fix mini-batch memory (1.3GB) — C6
17. Fix minor numerical values — L4, L5, L6

### Pass 4: References and metadata
18. Fix/add citations (Kool 2019, arXiv:2509.26397, Larsson misattribution) — M9, M10, M11
19. Fix title/date discrepancies — L2, L3
20. Remove stale system_scaler.py references — M12, L1

### Pass 5: Dependencies and infrastructure
21. Add block2/dmrgscf dependencies — M6
22. Note tests/ gitignore issue — H10
23. Add S6 Numba dependency callout — M7
24. Add max_diag_basis_size tuning — H12
25. Add 40Q integration test — M13
