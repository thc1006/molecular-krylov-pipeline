# ADR-002: Close 40Q Precision Gaps — From Infrastructure to Accuracy

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-03-08 |
| **Authors** | Project Team |
| **Supersedes** | ADR-001 (partial — updates Phase 3-4 roadmap) |
| **Related** | ADR-001, RESULTS.md, IMPLEMENTATION-TODO.md |

---

## 1. Context: Where We Actually Are

### 1.1 ADR-001 執行狀況

ADR-001 提出 5 個 Phase（0-4），21 個 PR。截至 2026-03-08：

| Phase | 完成 | 總計 | 狀態 |
|-------|------|------|------|
| Phase 0 (基礎) | 1/2 | PR 0.1 ✅, PR 0.2 CI/CD ❌ | 測試基礎完備，無 CI/CD |
| Phase 1 (稀疏) | 4/5 | PR 1.1-1.4 ✅, PR 1.8 混合精度 ❌ | 核心完成 |
| Phase 2 (NF 解鎖) | 6/6 | PR 2.0-2.6 全 ✅ | 完成但 NF 架構有根本限制 |
| Phase 3 (40Q 驗證) | 4/5 | PR 3.2-3.4 ✅, 3.3a/b CAS ✅, PR 3.1 NNCI 半完成 | NNCI 有代碼無整合 |
| Phase 4 (進階) | 1/4 | PR 4.1 expm_multiply ✅, 4.2-4.4 ❌ | 核心完成 |
| **總計** | **16/21** (76%) | | |

**額外完成（ADR-001 未規劃）**：
- HamiltonianCache (P0)
- config_integer_hash (M4 int64 溢出)
- bitpack_configs >64 sites
- Memory logger (6 點位)
- OOM 全域防護 (6 層)
- C2+C3 streaming connections
- 27 個 bug fix（Track D HIGH/MEDIUM + code review）
- 430 tests (401 default + 29 slow)

### 1.2 精確缺口分析（經網路二次驗證）

| # | 缺口 | 嚴重度 | 根因 | 文獻佐證 |
|---|------|--------|------|----------|
| G1 | NF 非自回歸架構 | **CRITICAL** | product-of-marginals P(o₁)×P(o₂)×...×P(oₙ) 無法捕捉 inter-orbital correlations | QiankunNet (Nature Comms 2025): autoregressive 在 30 spin orb 達 99.9% FCI；MADE (非自回歸) 精度差 2 個數量級 |
| G2 | NNCI 未整合進 pipeline | **HIGH** | nnci.py 有 556 行代碼 + 35 tests，但 pipeline.py 從未 import | Schmerwitz NNCI (JCTC 2025): N2 10¹⁰→7.16×10⁵ dets (5 OoM 壓縮) |
| G3 | 無 cc-pVDZ 端到端驗證 | **HIGH** | 所有 RESULTS.md 數據都是 STO-3G | IBM de facto benchmark = N2/cc-pVDZ CAS(10,26)；2026 發表標準 ≥ cc-pVDZ |
| G4 | Lanczos 缺再正交化 | **HIGH** | `_expm_multiply_lanczos()` 只做 3-term recurrence，30 步累積正交性損失 | ARPACK 的 IRL 有完整再正交化；自訂 Lanczos 缺失是已知 bug |
| G5 | 無自適應 Krylov 收斂 | **MEDIUM** | 固定 `max_krylov_dim=12`，無 energy convergence 監控 | SqDRIFT (Science 2026): 數學證明 SKQD polynomial 收斂需 adaptive stopping |
| G6 | retain_graph=True 記憶體洩漏 | **MEDIUM** | physics_guided_training.py 中的 backward 可能持有不必要的圖 | PyTorch best practice: 僅在多次 backward 時需要 |
| G7 | Dead imports / stale docs | **LOW** | pipeline.py 4 個死 import；SPEC.md 列 3 個不存在的檔案 | 代碼衛生 |

### 1.3 文獻驗證的關鍵事實

以下事實經 5 個並行研究代理上網驗證（2026-03-08）：

**自回歸 NF 是必須的（G1 確認）**：
- **QiankunNet** (Nature Comms 16, 8464, 2025): Decoder-only Transformer + MCTS，CAS(46e,26o) = 92 spin orbs。**代碼開源**: `github.com/xzzeng001/QiankunNet-VQE`
- **NNQS-SCI** (SC25): Transformer decoder + adaptive SCI，152 spin orbitals (10¹⁴ Hilbert)，Cr2 ground state -2086.400429 Ha
- **NQS-SC** (arXiv:2602.12993, ETH Zurich, Feb 2026): 明確證明 "sample configs then diagonalize" 優於純 VMC — **直接支持我們 pipeline 的架構**
- **AB-SND** (arXiv:2508.12724): Autoregressive NN + subspace diag，但僅在 spin models 上測試
- QiankunNet 明確比較：MADE (非自回歸，同我們架構類別) **無法達到 chemical accuracy**，autoregressive 精度高 2 個數量級

**NNCI 有文獻支持（G2 確認）**：
- **Schmerwitz NNCI** (JCTC 2025, arXiv:2406.08154): CNN classifier + active learning，N2 10e/52MO 只需 7.16×10⁵ dets（vs FCI 9.68×10¹⁰）
- **NO-NNCI** (arXiv:2510.27665): Natural orbital 旋轉可進一步減少 determinant 數量
- **CIGen-RBM** (arXiv:2409.06146): RBM + taboo list，N2/cc-pVDZ 99.99%，比 CIPSI 少 40% dets
- **PIGen-SQD** (arXiv:2512.06858): RBM + perturbative tensor，52 qubits，70% subspace reduction

**cc-pVDZ 是必要的（G3 確認）**：
- IBM SQD N2/cc-pVDZ CAS(10,26): 58 qubits，energy error ~10-50 mHa vs DMRG（**未達 chemical accuracy**）
- Reinholdt (JCTC 2025, arXiv:2501.07231): QSCI 在 N2/cc-pVDZ 需 10⁸-10⁹ samples = 10 天 QPU vs HCI 15 分鐘 1 CPU
- SandboxAQ GPU-DMRG: CAS(82,82) orbital-optimized on 單台 DGX-H100。[2Fe-2S] CAS(30,20) **trivially classical**

**eigsh 夠用但 Lanczos 要修（G4 確認）**：
- scipy ARPACK `eigsh`: Implicitly Restarted Lanczos 有完整再正交化，k=1 ground state 是最佳場景
- 自訂 `_expm_multiply_lanczos()`: **缺少再正交化**，krylov_dim=30 會累積正交性損失
- `scipy.sparse.linalg.expm_multiply` (Al-Mohy & Higham): 對 skew-Hermitian structured condition number = 0，inherently stable

**40Q 量子優勢不存在（補充）**：
- MIT PNAS Nexus (arXiv:2508.20972, Aug 2025): 量子化學 quantum advantage ≥ 2031 (FCI), ≥ 2036 (CCSD(T)), >2050 (DFT)
- SandboxAQ CAS(82,82) + STP-DAS 10¹⁵ dets = 經典已經 trivially 解決 40Q
- 我們的 pipeline 定位：**純經典 NF+Krylov，目標是 match DMRG 精度，不是 quantum advantage**

---

## 2. Decision: 按嚴重度依序關閉缺口

### 2.1 原則

1. **精度優先於功能**：不加新分子，先讓現有功能達到正確精度
2. **最小侵入修復**：優先修改現有代碼，不引入新架構
3. **cc-pVDZ 作為驗證標準**：所有修復必須在 cc-pVDZ 上驗證
4. **NNCI 作為自回歸 NF 的橋梁**：在自回歸 NF 完成前，NNCI + Direct-CI 是 40Q 的可行路徑

### 2.2 範圍限制

**本 ADR 不包含**：
- 自回歸 NF 重寫（Phase 4 長期目標，需 1-2 月）
- [2Fe-2S] DMRG-CASSCF 軌道優化（需 block2 安裝 + 研究級 convergence 調試）
- CI/CD pipeline（PR 0.2）
- 混合精度 eigensolver（PR 1.8）
- SQD 加速（PR 4.2）
- DMET fragment decomposition（PR 4.1 ADR-001）

---

## 3. Implementation Plan

### Track A: Numerical Correctness（2 個 PR）

#### PR-A1: Lanczos 再正交化修復

**檔案**: `src/utils/gpu_linalg.py:419-422`

**問題**: `_expm_multiply_lanczos()` 只做 3-term Lanczos recurrence（只對前 2 個向量正交化）。krylov_dim=30 時，正交性損失累積導致 time evolution 偏離 unitarity。

**修復**:
```python
# 在 line 422 後加入完整再正交化
# w = w - V[:, :j+1] @ (V[:, :j+1].conj().T @ w)
```

**文獻依據**: ARPACK 的 IRL 對所有 ncv 向量做完整再正交化。我們的自訂 Lanczos 沒有這個保護。

**驗證**:
- N2/STO-3G SKQD energy 不可退化
- krylov_dim=30 的 V 矩陣正交性 ||V^T V - I||_F < 1e-10
- 430 tests 全過

**優先度**: HIGH — 影響所有走 GPU Lanczos 路徑的 Krylov time evolution

---

#### PR-A2: Krylov 自適應收斂 + dt 縮放

**檔案**: `src/krylov/skqd.py`, `src/pipeline.py`

**問題**: 固定 `max_krylov_dim=12` 和 `dt=0.1`，無 energy convergence 監控。cc-pVDZ Hamiltonian 的 spectral norm 可能比 STO-3G 大 5-10x，需要 adaptive dt。

**修復**:
1. 每個 Krylov step 後計算 ground state energy；|E_k - E_{k-1}| < 0.01 mHa 時 early stop
2. `dt = min(0.5, pi / spectral_norm_estimate)` — spectral norm 從 diagonal energy range 估計
3. S matrix conditioning 監控：cond(S) > 1e12 時停止擴展

**文獻依據**: SqDRIFT (arXiv:2508.02578) 數學證明 SKQD polynomial 收斂需要 adaptive stopping。

**驗證**:
- N2/STO-3G 結果不變（dt=0.1 應該在 safe range）
- N2/cc-pVDZ CAS(10,8) 收斂
- 430 tests 全過

**優先度**: MEDIUM — cc-pVDZ 必需

---

### Track B: NNCI Pipeline 整合（2 個 PR）

#### PR-B1: NNCI 模組整合進 FlowGuidedSKQD

**檔案**: `src/krylov/skqd.py`, `src/krylov/nnci.py`, `src/pipeline.py`

**問題**: nnci.py 有 556 行完整實現（NNCIConfig, ConfigImportanceClassifier, CandidateGenerator, NNCIActiveLearning）+ 35 tests，但 pipeline.py 從未 import。

**整合方案**:
```python
class PipelineConfig:
    # 新增
    use_nnci: bool = False          # 啟用 NNCI active learning
    nnci_iterations: int = 5        # NNCI 迭代次數
    nnci_candidates_per_iter: int = 5000  # 每輪候選數

# 在 FlowGuidedSKQD.compute_ground_state_energy() 中:
if self.pipeline_config.use_nnci:
    basis = self._nnci_expand_basis(basis)
```

**NNCI expansion 流程**:
1. 用當前 basis diag 結果的 CI coefficients 訓練 ConfigImportanceClassifier
2. CandidateGenerator 產生 single/double/triple 候選 configs
3. Classifier 篩選 top-K important candidates
4. 合併進 basis，重新 diag
5. 迭代直到 energy 收斂

**文獻依據**:
- Schmerwitz NNCI: N2 10¹⁰ → 7.16×10⁵ (5 OoM 壓縮)
- CIGen-RBM taboo list: 防止重複評估已拒絕的 candidates

**改進（來自文獻）**:
1. 加 **taboo set**: 被 classifier 拒絕的 configs 不會在下一輪重複出現（CIGen 的做法）
2. 加 **dynamic coefficient cutoff**: 根據 CI coefficient 分佈動態調整 importance threshold（Schmerwitz 的做法）

**驗證**:
- N2/STO-3G: NNCI-SKQD energy ≤ Direct-CI-SKQD energy
- N2/cc-pVDZ CAS(10,8): NNCI 發現 triples 改善 energy
- NNCI 不破壞現有 430 tests

**優先度**: HIGH — 這是自回歸 NF 完成前的 40Q 橋梁方案

---

#### PR-B2: MP2 Pruning 接入 Krylov Expansion

**檔案**: `src/krylov/skqd.py`

**問題**: `perturbative_pruning.py` (PR 3.2) 已實現 `compute_mp2_amplitudes()` + `mp2_importance_scores()` + `prune_basis()`，但只在 `FlowGuidedSKQD.__init__` 做一次性 cache。未在 Krylov expansion 中用於引導 `_find_connected_configs()`。

**修復**: 在 `_find_connected_configs()` 中，用 MP2 importance scores 對 connected configs 排序，優先保留 high-importance 的 connections。

**文獻依據**: PIGen-SQD (arXiv:2512.06858) 用 perturbative measures 做 physics-informed pruning，70% subspace reduction。

**驗證**:
- N2/STO-3G 結果不退化
- N2/cc-pVDZ CAS(10,8): pruned Krylov 保留精度
- MP2 scores 與 actual CI coefficients 有正相關

**優先度**: MEDIUM — 降低 Krylov expansion 的無效 discovery

---

### Track C: cc-pVDZ 端到端驗證（2 個 PR）

#### PR-C1: N2/cc-pVDZ 階梯式驗證

**檔案**: `tests/test_ccpvdz_validation.py` (新), `src/hamiltonians/molecular.py`

**已有基礎**: PR 3.3a 已加入 `cas=(nelecas, ncas)` 支援 + `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(10,8))`。

**驗證階梯**:

| Step | Active Space | Qubits | Configs | 方法 | 期望結果 |
|------|-------------|--------|---------|------|----------|
| C1.1 | N2/cc-pVDZ CAS(6,6) | 12 | 400 | Exact FCI | 匹配 PySCF CASCI |
| C1.2 | N2/cc-pVDZ CAS(10,8) | 16 | 3,136 | Direct-CI SKQD | < 1 mHa vs CASCI |
| C1.3 | N2/cc-pVDZ CAS(10,10) | 20 | 63,504 | Sparse SKQD | < 1 mHa vs CASCI |
| C1.4 | N2/cc-pVDZ CAS(10,14) | 28 | 4,008,004 | NNCI + SKQD | < 2 mHa vs DMRG |

**C1.1-C1.3 是驗證性測試**：確認 infrastructure 在 cc-pVDZ 上正常工作。
**C1.4 是目標測試**：第一次需要 sampling-based discovery（400 萬 configs）。

**驗證**:
- 每個 step 的 energy 與 PySCF reference 比較
- C1.3 的 OOM 安全性（63K configs in sparse mode）
- C1.4 的 NNCI 發現 triples/quadruples

**優先度**: HIGH — 這是確認 40Q readiness 的核心 checkpoint

---

#### PR-C2: Chemical Accuracy Report on cc-pVDZ

**檔案**: `examples/ccpvdz_benchmark.py` (新), 更新 `RESULTS.md`

**內容**: 在 C1.1-C1.4 全通過後，產生正式的 cc-pVDZ benchmark results：
1. N2/cc-pVDZ 各 active space 的 energy + wall time + peak memory
2. 與 IBM SQD N2/cc-pVDZ 數據比較（IBM 用 CAS(10,26) = 58Q，我們用 CAS(10,14) = 28Q 作為中間點）
3. NNCI 的 config discovery 分析：多少 triples/quadruples 被發現，energy 改善量

**優先度**: MEDIUM — 在 Track A+B+C1 完成後

---

### Track D: 代碼清理（1 個 PR）

#### PR-D1: Dead Code + Stale Docs 清理

**檔案**: `src/pipeline.py`, `SPEC.md`

**修復項目**:
1. 移除 pipeline.py 的 4 個死 import: `select_diverse_basis`, `davidson_eigensolver`, `adaptive_eigensolver`, `SampleBasedKrylovDiagonalization`
2. 移除 pipeline.py 的 2 個 vestigial config fields: `use_davidson`, `use_ci_seeding`
3. 更新 SPEC.md: 移除 3 個不存在的檔案引用 (`complex_nqs.py`, `discrete_flow.py`, `training.py`)
4. `retain_graph=True` 審計：確認 physics_guided_training.py 中是否真的需要

**優先度**: LOW — 不影響功能

---

## 4. 依賴順序

```
PR-A1 (Lanczos fix)  ──────────────────────────┐
                                                 │
PR-A2 (Krylov adaptive) ─────┐                  ├──→ PR-C1 (cc-pVDZ validation)
                              │                  │          │
PR-B1 (NNCI integration) ────┼──────────────────┘          │
                              │                             ↓
PR-B2 (MP2 in Krylov) ───────┘                  PR-C2 (cc-pVDZ benchmark report)

PR-D1 (cleanup) ── 獨立，任何時候可做
```

**關鍵路徑**: A1 → C1（Lanczos 修復後才能信任 cc-pVDZ 結果）
**平行路徑**: B1 可與 A1/A2 同時進行

---

## 5. 成功標準

| # | 標準 | 量化指標 |
|---|------|----------|
| S1 | Lanczos 正交性 | krylov_dim=30 時 ‖V^T V - I‖_F < 1e-10 |
| S2 | N2/cc-pVDZ CAS(10,8) chemical accuracy | Energy error < 1.594 mHa vs PySCF CASCI |
| S3 | N2/cc-pVDZ CAS(10,10) sparse mode 正常 | 63K configs 不 OOM，energy error < 1 mHa vs CASCI |
| S4 | NNCI 整合可用 | `PipelineConfig(use_nnci=True)` 在 N2/STO-3G 上改善 energy |
| S5 | NNCI 發現 higher excitations | N2/cc-pVDZ CAS(10,8) 上 NNCI 發現 >5% triples |
| S6 | 無回歸 | 全部 430 tests 通過 |
| S7 | N2/cc-pVDZ CAS(10,14) 可執行 | 28Q 系統完成（不 crash / OOM），energy 在合理範圍 |

---

## 6. 風險分析

| 風險 | 可能性 | 影響 | 緩解 |
|------|--------|------|------|
| cc-pVDZ CASSCF 收斂不穩定 | Medium | High | 使用 `mol.symmetry=True`；CAS(10,8) 先驗證再做 CAS(10,14) |
| NNCI 在 cc-pVDZ 上效果不佳 | Medium | Medium | 回退到 Direct-CI + SKQD（STO-3G 上已證明有效） |
| Lanczos fix 導致 energy 變化 | Low | Medium | 變化應向更精確方向；若退化則檢查 Krylov dim 設定 |
| CAS(10,14) 28Q 超出 DGX Spark 能力 | Low | High | 400 萬 configs 不需全 enumerate；50K selected basis fits easily |
| N2 MO non-determinism 影響 cc-pVDZ | Medium | Medium | HamiltonianCache 固定 MO coefficients |

---

## 7. 40Q 長期路線圖（本 ADR 之後）

| 階段 | 目標 | 預估時間 | 依賴 |
|------|------|----------|------|
| **本 ADR** | 關閉 G1-G7 精確缺口，cc-pVDZ 28Q 驗證 | 2-3 週 | 無 |
| **Phase 4a** | Autoregressive NF (QiankunNet-style Transformer + MCTS) | 4-6 週 | 本 ADR 完成 |
| **Phase 4b** | N2/cc-pVDZ CAS(10,20) = 40Q 端到端 | 1-2 週 | Phase 4a |
| **Phase 4c** | [2Fe-2S] CAS(30,20) = 40Q benchmark vs IBM/DMRG | 2-4 週 | Phase 4b + block2 |

**Phase 4a 架構方向（基於文獻調研）**：
- **目標架構**: Decoder-only Transformer (amplitude) + MLP (phase)
- **取樣策略**: Quaternary MCTS with particle-number pruning（QiankunNet 的做法）
- **開源參考**: `github.com/xzzeng001/QiankunNet-VQE`
- **替代方案**: RetNet (arXiv:2411.03900, SandboxAQ) — O(n) inference cost for long sequences
- **風格**: 非 VMC — sample configs then diagonalize (NQS-SC, arXiv:2602.12993 支持)

---

## 8. 參考文獻（全部經網路驗證 2026-03-08）

### 自回歸 NF/NQS
- [QiankunNet — Nature Communications 16, 8464 (2025)](https://www.nature.com/articles/s41467-025-63219-2) | [GitHub](https://github.com/xzzeng001/QiankunNet-VQE)
- [NNQS-SCI — SC25 (152 spin orbs)](https://dl.acm.org/doi/10.1145/3712285.3759800)
- [NQS-SC vs VMC — arXiv:2602.12993 (Feb 2026)](https://arxiv.org/abs/2602.12993)
- [Autoregressive NQS with symmetries — arXiv:2310.04166](https://arxiv.org/abs/2310.04166)
- [RetNet NQS — arXiv:2411.03900 (SandboxAQ)](https://arxiv.org/abs/2411.03900)
- [AB-SND — arXiv:2508.12724](https://arxiv.org/abs/2508.12724)
- [Physics-Informed Transformer — Nature Comms 16, 10811 (2025)](https://www.nature.com/articles/s41467-025-66844-z)
- [Deterministic NQS + PT2 — arXiv:2601.21310 (Jan 2026)](https://arxiv.org/abs/2601.21310)

### NNCI / ML-guided CI
- [Schmerwitz NNCI — JCTC 2025, arXiv:2406.08154](https://arxiv.org/abs/2406.08154)
- [NO-NNCI — arXiv:2510.27665 (Nov 2025)](https://arxiv.org/abs/2510.27665)
- [CIGen RBM — arXiv:2409.06146](https://arxiv.org/abs/2409.06146)
- [PIGen-SQD — arXiv:2512.06858 (IBM)](https://arxiv.org/abs/2512.06858)
- [RLCI — JCTC 2021](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00010)

### SQD/SKQD/QSCI Benchmarks
- [IBM SQD Science Advances 2025](https://www.science.org/doi/10.1126/sciadv.adu9991) — N2/cc-pVDZ CAS(10,26), [2Fe-2S], [4Fe-4S]
- [SqDRIFT — arXiv:2508.02578 (Science Mar 2026)](https://arxiv.org/abs/2508.02578) — 72Q half-Möbius
- [Reinholdt "Fatal Flaw" — JCTC 2025, arXiv:2501.07231](https://arxiv.org/abs/2501.07231)
- [QBE-SQD — Digital Discovery 2026](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00416k)

### Classical Baselines
- [SandboxAQ GPU-DMRG CAS(82,82) — JCTC 2025, arXiv:2503.20700](https://arxiv.org/abs/2503.20700)
- [STP-DAS 10¹⁵ dets — Nature Comms 2025](https://www.nature.com/articles/s41467-025-65967-7)
- [SHCI Cr2 28e/76o — JACS 2022](https://pubs.acs.org/doi/10.1021/jacs.2c06357)
- [FCIQMC-GAS 96e/159o — JCTC 2022](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00936)
- [MIT Quantum Advantage Review — arXiv:2508.20972](https://arxiv.org/abs/2508.20972)

### Krylov / Eigensolver
- [SciPy ARPACK eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- [SciPy expm_multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html)
- [SciPy Issue #22974: Krylov matrix functions](https://github.com/scipy/scipy/issues/22974)
- [CPU vs GPU eigensolver benchmark — Inductiva.AI 2025](https://inductiva.ai/blog/article/cpu-vs-gpu-eigensolver-benchmark)
- [GenKSR — arXiv:2512.19420](https://arxiv.org/abs/2512.19420)
