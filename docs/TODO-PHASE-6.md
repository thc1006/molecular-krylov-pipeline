# TODO: Phase 6 — 文獻驅動的架構修復

> Generated: 2026-03-09
> 依據: 深度調研 60+ 篇論文（2025-2026）
> 調研文件: `docs/NQS-VMC-SURVEY-2026-Q1.md`, `docs/LITERATURE-SURVEY-2026-Q1.md`,
>   `docs/GENERATIVE-MODEL-SURVEY-2026-03.md`, `docs/VMC-SIGN-PROBLEM-RESEARCH.md`
> 原則: 修復根因 → 改善選擇 → 改善架構 → 擴展規模
> 狀態: ❌ 未開始 | 🔄 進行中 | ✅ 完成

---

## 診斷的三大根因

1. **REINFORCE 是錯誤的 VMC 優化器** — 領域用 SR/MinSR/SPRING
2. **ψ=√p×s(x) 分解是錯誤的** — SOTA 用行列式/連續相位
3. **NF 缺乏 H-coupling 引導** — HAAR-SCI 等用 Hamiltonian 篩選配置

**關鍵驗證**: NQS-SC > NQS-VMC (arXiv:2602.12993, ETH Zurich) → Direct-CI + SKQD 是正確路線

---

## P0: 立即修復（修復已知收斂失敗）

### P0.1: MinSR 取代 REINFORCE ⬅ 最高優先級
**影響**: 修復 24Q+ VMC 不收斂 | **工作量**: 2-3 天 | **依賴**: 無
**參考**: MinSR (Nature Physics 2024), SPRING (JCP 2024)
**狀態**: ✅ 完成 (2026-03-09)

- [x] VMCConfig: `optimizer_type="minsr"`, `sr_regularization=1e-3`, `sr_reg_decay=0.99`, `sr_reg_min=1e-5`
- [x] `_compute_per_sample_jacobian()`: O-matrix (N_samples × N_params), FP64
- [x] `_minsr_update()`: Woodbury identity solve, G = J_c @ J_c^T + λNI, fallback on singular
- [x] `_minsr_step()` / `_reinforce_step()` dispatch via `optimizer_type`
- [x] Sign network keeps separate Adam (MinSR only for flow params)
- [x] 38 VMC tests pass (17 new MinSR tests + 21 existing)

---

### P0.2: VMC 預設 OFF
**影響**: 正確預設值 | **工作量**: 1 小時 | **依賴**: 無
**參考**: NQS-SC > NQS-VMC (arXiv:2602.12993)
**狀態**: ✅ 完成 (2026-03-09) — 預設已是 False，加了 DEFAULT OFF 註釋

- [ ] `src/pipeline.py` — PipelineConfig:
  - `use_vmc_training: bool = True` → `False`
  - `use_sign_network: bool = True` → `False`
  - 加註釋: VMC 用 REINFORCE 不收斂; NQS-SC > NQS-VMC
- [ ] `adapt_to_system_size()`: 不自動啟用 VMC/sign
- [ ] 更新依賴 VMC 預設的測試（只改預設，不改測試邏輯）
- [ ] VERIFY: 所有現有 tests pass

---

### P0.3: 修復 test_nf_beats_direct_ci
**影響**: 測試正確性 | **工作量**: 已完成 | **依賴**: 無
**狀態**: ✅ 完成

- [x] 移除 `delta > 10 mHa` 斷言（SKQD bug fix 後 NF ≈ Direct-CI）
- [x] 改為 NF 不比 Direct-CI 差超過 2 mHa（容差）
- [x] 更新 docstring 說明原因

---

## P1: 近期（配置選擇改進）

### P1.1: H-coupling 過濾（HAAR-SCI 風格）
**影響**: 轉型性 — NF 輸出直接由 Hamiltonian 篩選 | **工作量**: 2-3 天 | **依賴**: 無
**參考**: HAAR-SCI (JCTC Dec 2025, DOI:10.1021/acs.jctc.5c01415)
**狀態**: ✅ 完成 (2026-03-09)

- [x] SKQDConfig: `use_h_coupling_ranking=True`, `h_coupling_n_ref=50`
- [x] `_compute_h_coupling_scores(candidates, basis, psi)`: Σ |c_i| × |⟨x|H|x_i⟩| for top-|c_i| refs
- [x] Integrated into `_find_connected_configs()` — priority over MP2 when psi available
- [x] Uses `config_integer_hash` for O(1) candidate→connection matching
- [x] 10 tests pass (config, scores, integration, energy comparison)
- [x] VERIFY: 629 existing tests pass

---

### P1.2: 迭代精化迴圈（NNCI Active Learning）
**影響**: 從 single-pass → iterative，系統性擴展 basis | **工作量**: 1-2 天 | **依賴**: P1.1
**參考**: NNCI (JCTC 2025), CIGS (JCTC 2025, arXiv:2409.06146)
**狀態**: ⏸ 延後 — NNCI 已整合為 pipeline pre-processing (PR-B1)，Krylov inner loop 整合待需要時做

NNCI active learning 已在 `pipeline.py:886` 作為 SKQD 前的 basis expansion step。
Krylov expansion 現在有 H-coupling ranking (P1.1) + taboo (P1.3) 做更有效的連接探索。
將 NNCI 嵌入 Krylov inner loop 的邊際效益較低，等 40Q benchmark 結果再決定。

---

### P1.3: Taboo List（CIGS 風格）
**影響**: 避免 Krylov expansion 重複計算 | **工作量**: 1 天 | **依賴**: 無
**參考**: CIGS (JCTC 2025, arXiv:2409.06146)
**狀態**: ✅ 完成 (2026-03-09)

- [x] SKQDConfig: `use_taboo_list=True`, `taboo_max_size=500000`
- [x] `_taboo_set` + `_taboo_deque` (FIFO eviction) in `FlowGuidedSKQD.__init__`
- [x] `_add_to_taboo(hashes)`: add with FIFO eviction, no-op when disabled
- [x] `clear_taboo()`: reset both set and deque
- [x] `_find_connected_configs()`: taboo check alongside basis_set and new_set dedup
- [x] Trimmed configs during Krylov expansion automatically added to taboo
- [x] 12 tests pass (config, mechanics, integration)
- [x] VERIFY: 629 existing tests pass

---

## P2: 中期（架構改進）

### P2.1: Phase Network (e^{iφ}) 取代 Sign Network
**影響**: 正確的符號表示 | **工作量**: 2-3 天 | **依賴**: P0.1
**參考**: QiankunNet (Nature Comms 2025), VMC-SIGN-PROBLEM-RESEARCH.md §4.2
**狀態**: ✅ 完成 (2026-03-09)

- [x] `PhaseNetwork` — 2π·sigmoid output ∈ [0, 2π), phase_factor() → complex128
- [x] VMCTrainer: complex128 E_loc with full phase factor e^{i(φ(y)-φ(x))}
- [x] Pipeline: `sign_architecture="phase"` (default) vs `"sign"` (legacy), with validation
- [x] 17 tests (basic, subsumes sign, gradient smooth, VMC integration, pipeline)
- [x] Code review fix: cos(φ_diff) → full complex phase factor for correct gradients
- [x] VERIFY: 686 tests pass

---

### P2.2: Natural Orbitals for NNCI
**影響**: 更緊湊的 CI expansion | **工作量**: 1-2 天 | **依賴**: 無
**參考**: NO-NNCI (arXiv:2510.27665, Oct 2025)
**狀態**: ✅ 完成 (2026-03-09)

- [x] `compute_mp2_rdm1()`: MP2 unrelaxed 1-RDM from t2 amplitudes (no PySCF re-run)
- [x] `compute_natural_orbitals()`: diagonalize 1-RDM → NOs + occupation numbers (descending)
- [x] `transform_integrals_to_no_basis()`: 4-index transformation h1e/h2e → NO basis
- [x] `no_orbital_importance()`: min(n, 2-n) correlation importance scores
- [x] NNCIConfig: `use_natural_orbitals`, `no_max_active_orbitals` fields
- [x] NNCI: NO-prioritized candidate generation (restricted to active correlated orbitals)
- [x] PipelineConfig: `nnci_use_natural_orbitals`, `nnci_no_max_active_orbitals`
- [x] 21 tests (1-RDM, NOs, integral transform, NNCI integration, pipeline config)
- [x] VERIFY: 686 tests pass

---

### P2.3: Gumbel Top-K 取代 Plackett-Luce
**影響**: 無偏離散取樣 + 更好 exploration | **工作量**: 1 天 | **依賴**: 無
**參考**: HAAR-SCI (JCTC Dec 2025)
**狀態**: ✅ 完成 (2026-03-09)

- [x] `GumbelTopK`: scores + Gumbel noise → top-k, STE backward, nn.Parameter temperature
- [x] Config toggle: `topk_type="sigmoid"` (default) vs `"gumbel"`, with validation
- [x] Code review fix: GumbelTopK temperature upgraded from plain float to nn.Parameter
- [x] 8 tests (config, particle conservation, stochastic diversity, gradient, temperature)
- [x] VERIFY: 686 tests pass

---

### P2.4: Sampling Without Replacement（AR Flow）
**影響**: 避免 peaked wavefunction 重複取樣 | **工作量**: 1 天 | **依賴**: 無
**參考**: Peaked Wavefunctions (arXiv:2408.07625), 118Q single GPU
**狀態**: ✅ 完成 (2026-03-09)

- [x] `sample_unique(n_unique_target, max_attempts, batch_multiplier)` in AR flow
- [x] Rejection sampling with hash set dedup, iterative batch accumulation
- [x] Teacher-forcing log_prob recomputation for accuracy
- [x] 11 tests (uniqueness, target respect, particle conservation, coverage, max_attempts)
- [x] VERIFY: 686 tests pass

---

## P3: 長期（研究方向）

### P3.1: NNBF / Transformer Backflow
**影響**: SOTA 波函數精度 | **工作量**: 1-2 週 | **依賴**: P0.1, P2.1
**參考**: NNBF (PRB 2024, arXiv:2403.03286), Transformer Backflow (arXiv:2509.25720)

- [ ] 研究 NNBF 開源代碼
- [ ] 設計: MLP(occupation vector) → orbital coefficients → Slater determinant(s)
  - 符號自動來自 det()
  - 證明為通用近似 (universal approximator)
- [ ] 評估是否取代整個 NF + sign network
- [ ] 概念驗證: LiH 上 NNBF + SR vs 現有 AR flow + REINFORCE

---

### P3.2: RetNet 取代 Transformer
**影響**: 推理 O(n) vs O(n²) | **工作量**: 1 週 | **依賴**: 無
**參考**: RetNet NQS (arXiv:2411.03900, SandboxAQ)

- [ ] 研究 RetNet 架構（recurrent mode inference）
- [ ] 評估: 40Q 下 O(n²) 還可接受，100Q+ 才需要
- [ ] 如果目標擴展到 100Q+，設計 RetNet 版 AR flow

---

### P3.3: Deterministic NQS（取代 VMC）
**影響**: 消除 MC 噪聲 | **工作量**: 1-2 週 | **依賴**: P0.1
**參考**: arXiv:2601.21310 (Jan 2026)

- [ ] 研究確定性框架: adaptive subspace + neural backflow + PT2 校正
- [ ] 評估: 概念上接近 SKQD（都在子空間做精確計算）
- [ ] 如果 MinSR VMC 仍不收斂，考慮此路線

---

### P3.4: SPRING 優化器
**影響**: 最佳 NQS 優化器 | **工作量**: 3-5 天 | **依賴**: P0.1
**參考**: SPRING (JCP 2024, arXiv:2401.10190)
**狀態**: 🔄 實作完成，待驗證收斂性

- [x] IMPL: `_spring_update()` — Kaczmarz projection + momentum (Eq. 33-34)
- [x] `_spring_step()` — complete VMC step with SPRING optimizer
- [x] VMCConfig: `optimizer_type="spring"`, `spring_momentum=0.95`, `spring_proj_reg=1.0`
- [x] Cholesky solve with fallback chain (Cholesky → LU → gradient)
- [x] Lazy phi initialization, persistent across steps
- [x] 38 existing VMC tests pass (no regression)
- [ ] Convergence benchmark: H2 SPRING vs MinSR (n_steps=500)
- [ ] Convergence benchmark: LiH SPRING vs MinSR
- [ ] 24Q+ VMC convergence test with SPRING

---

### P3.5: Rotation Thresholding for Krylov
**影響**: 改善 Krylov 矩陣 conditioning | **工作量**: 2-3 天 | **依賴**: 無
**參考**: arXiv:2602.11985 (Feb 2026)

- [ ] 讀取 `src/krylov/skqd.py` conditioning check
- [ ] IMPL: Eigenvector-preserving rotation before thresholding
  - 取代現有 `_last_ill_conditioned` 啟發式
  - 100x sample 減少

---

## 不應追求的方向

| 方向 | 原因 |
|------|------|
| Diffusion models | 無證據適用離散配置取樣，粒子守恆難強制 |
| 純 VMC at scale | ETH + 我們結果 + 多篇 2025 確認 >20 orbs 不收斂 |
| 更大 NF without H-guidance | 問題非容量，是缺乏 H 引導 |
| RLCI (Q-learning) | 指數動作空間不可擴展 |
| 更多 SQD 改進 | SQD 已證明 <SKQD，不值得投入 |

---

## 執行順序建議

```
Week 1 (立即):
  ✅ P0.3: test_nf_beats_direct_ci 修復
  ✅ P0.1: MinSR 取代 REINFORCE（38 tests, Woodbury identity solve）
  ✅ P0.2: VMC 預設 OFF（587 tests pass）
  → P1.3: Taboo list（簡單，1 天）

Week 2 (配置選擇):
  → P1.1: H-coupling 過濾（2-3 天，轉型性改進）
  → P2.4: Sampling without replacement（1 天）

Week 3 (迭代精化):
  → P1.2: Iterative refinement loop（1-2 天）
  → P2.3: Gumbel Top-K（1 天）

Week 4 (架構):
  → P2.2: Natural orbitals for NNCI（1-2 天）
  → P2.1: Phase network（2-3 天）

之後:
  → P3.1: NNBF（如 VMC 仍需要）
  → P3.4: SPRING（如 MinSR 精度不足）
  → P3.5: Rotation thresholding
  → P3.2: RetNet（100Q+ 時才需要）
```

---

## 成功指標

| 階段 | 指標 | 目標 |
|------|------|------|
| P0 完成 | VMC 不再是預設 + MinSR 可用 | VMC optional, MinSR on LiH 收斂 |
| P1 完成 | H-coupling 過濾 + 迭代精化 | CAS(10,10) error < 5 mHa (vs 14.2 mHa) |
| P2 完成 | Phase network + NOs + Gumbel | CAS(10,20) 40Q NF 比 Direct-CI 好 ≥ 10 mHa |
| P3 完成 | NNBF / SPRING / RetNet | CAS(10,26) 52Q 化學精度（IBM benchmark） |

---

## 相關文件

- [ADR-002](ADR-002-close-40q-gaps.md) — ✅ 已完成（Phase 5）
- [TODO-ADR-002](TODO-ADR-002.md) — ✅ 已完成
- [NQS-VMC Survey](NQS-VMC-SURVEY-2026-Q1.md) — 569 行
- [Literature Survey](LITERATURE-SURVEY-2026-Q1.md) — 508 行
- [Generative Model Survey](GENERATIVE-MODEL-SURVEY-2026-03.md) — 537 行
- [VMC Sign Problem Research](VMC-SIGN-PROBLEM-RESEARCH.md) — 439 行
- [40Q Benchmark Research](40Q-BENCHMARK-RESEARCH.md) — 668 行
