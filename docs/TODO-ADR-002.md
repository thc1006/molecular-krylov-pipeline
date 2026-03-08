# TODO: ADR-002 — Close 40Q Precision Gaps

> Generated: 2026-03-08
> 依賴: ADR-002-close-40q-gaps.md
> 原則: 精度優先於功能。不做新架構，先讓現有功能達到正確精度。
> 狀態追蹤: ❌ = 未開始, 🔄 = 進行中, ✅ = 完成

---

## Track A: Numerical Correctness

### PR-A1: Lanczos 再正交化修復
**優先度**: HIGH | **預估**: 2-4h | **依賴**: 無

- [ ] 讀取 `src/utils/gpu_linalg.py` lines 375-450 (`_expm_multiply_lanczos`)
- [ ] 在 3-term recurrence 後 (line ~422) 加入完整再正交化:
  ```python
  # Full reorthogonalization against all previous Lanczos vectors
  for i in range(j + 1):
      w = w - torch.dot(V[:, i].conj(), w) * V[:, i]
  ```
- [ ] TEST: `test_lanczos_orthogonality` — krylov_dim=30 時 ‖V^T V - I‖_F < 1e-10
- [ ] TEST: `test_lanczos_fix_no_regression` — N2/STO-3G SKQD energy 不變（< 0.01 mHa 差異）
- [ ] TEST: `test_lanczos_unitarity` — e^{-iHdt} 保持 state normalization
- [ ] VERIFY: 430 existing tests pass

---

### PR-A2: Krylov 自適應收斂 + dt 縮放
**優先度**: MEDIUM | **預估**: 4-6h | **依賴**: PR-A1

- [ ] 讀取 `src/krylov/skqd.py` Krylov expansion loop
- [ ] IMPL: Energy convergence monitoring
  - 每個 Krylov step 後計算 ground state energy
  - |E_k - E_{k-1}| < `convergence_threshold` (default 0.01 mHa) 時 early stop
  - 日誌記錄每步 energy 用於調試
- [ ] IMPL: Adaptive dt scaling
  - 在 `adapt_to_system_size()` 中加入 dt 縮放
  - `dt = min(0.5, pi / spectral_norm_estimate)`
  - Spectral norm 從 `max(diag_energies) - min(diag_energies)` 估計
- [ ] IMPL: S matrix conditioning guard
  - `cond(S) > 1e12` 時停止 Krylov expansion
  - 防止 ill-conditioned generalized eigenvalue problem
- [ ] TEST: `test_adaptive_dt_sto3g` — dt=0.1 在 safe range，結果不變
- [ ] TEST: `test_adaptive_dt_ccpvdz` — cc-pVDZ CAS(10,8) 收斂
- [ ] TEST: `test_early_stopping` — energy 收斂後不浪費 Krylov steps
- [ ] TEST: `test_s_matrix_conditioning` — ill-conditioned S 不 crash
- [ ] VERIFY: 430 existing tests pass

---

## Track B: NNCI Pipeline 整合

### PR-B1: NNCI 模組整合進 pipeline
**優先度**: HIGH | **預估**: 1-2d | **依賴**: 無（可與 Track A 平行）

- [ ] 讀取 `src/krylov/nnci.py` 完整代碼 (556 行)
- [ ] 讀取 `src/pipeline.py` 的 Stage 3 subspace diag 入口
- [ ] IMPL: PipelineConfig 新增 NNCI 相關配置
  ```python
  use_nnci: bool = False
  nnci_iterations: int = 5
  nnci_candidates_per_iter: int = 5000
  nnci_importance_threshold: float = 0.5
  ```
- [ ] IMPL: `adapt_to_system_size()` 中 auto-enable NNCI for >20K configs
- [ ] IMPL: FlowGuidedSKQD 新增 `_nnci_expand_basis()` 方法
  - 從當前 eigenvector CI coefficients 提取 training labels
  - 訓練 ConfigImportanceClassifier (nnci.py 已有)
  - CandidateGenerator 產生 singles/doubles/triples
  - Classifier 篩選 top-K
  - 合併進 basis
  - 重複 `nnci_iterations` 次
- [ ] IMPL: Taboo set（來自 CIGen 文獻）
  - Track 被 classifier 拒絕的 configs
  - 下一輪不重複評估
  - 使用 config_integer_hash 做快速 membership check
- [ ] IMPL: Dynamic coefficient cutoff（來自 Schmerwitz NNCI 文獻）
  - 根據 CI coefficient 分佈自動調整 importance threshold
  - 而非固定 top-K
- [ ] TEST: `test_nnci_pipeline_basic` — PipelineConfig(use_nnci=True) 在 LiH 上不 crash
- [ ] TEST: `test_nnci_improves_energy` — N2/STO-3G NNCI-SKQD ≤ Direct-CI-SKQD
- [ ] TEST: `test_nnci_discovers_triples` — NNCI 的 expanded basis 含 excitation rank ≥ 3
- [ ] TEST: `test_nnci_taboo_works` — 被拒絕的 configs 不會重複出現
- [ ] TEST: `test_nnci_auto_enable` — >20K configs 自動啟用
- [ ] TEST: `test_nnci_user_override` — use_nnci=False 時不啟用，即使 >20K
- [ ] VERIFY: 430 existing tests pass + 35 existing nnci tests pass

---

### PR-B2: MP2 Pruning 接入 Krylov Expansion
**優先度**: MEDIUM | **預估**: 4-6h | **依賴**: 無

- [ ] 讀取 `src/krylov/skqd.py` `_find_connected_configs()` 方法
- [ ] 讀取 `src/hamiltonians/molecular.py` `perturbative_pruning` 相關代碼
- [ ] IMPL: 在 `_find_connected_configs()` 中用 MP2 scores 排序
  - 對 connected configs 計算 MP2 importance score
  - 按 score 降序排列
  - 保留 top fraction（configurable, default 80%）
  - 必須保留所有 essential configs (rank ≤ 2)
- [ ] TEST: `test_mp2_guided_krylov` — MP2-guided 的 basis 不比 unguided 差
- [ ] TEST: `test_mp2_preserves_essentials` — singles/doubles 永遠保留
- [ ] TEST: `test_mp2_reduces_basis` — connected configs 數量減少
- [ ] VERIFY: 430 existing tests pass

---

## Track C: cc-pVDZ 端到端驗證

### PR-C1: N2/cc-pVDZ 階梯式驗證
**優先度**: HIGH | **預估**: 1-2d | **依賴**: PR-A1, PR-A2, PR-B1

- [ ] 讀取 `src/hamiltonians/molecular.py` CAS 支援代碼
- [ ] IMPL: 確認/補充 factory functions
  - `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(6,6))` — 12Q, 400 configs
  - `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(10,8))` — 16Q, 3136 configs
  - `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(10,10))` — 20Q, 63504 configs
  - `create_n2_cas_hamiltonian(basis='cc-pvdz', cas=(10,14))` — 28Q, ~4M configs
- [ ] TEST (slow): `test_ccpvdz_cas66_exact_fci`
  - CAS(6,6) 400 configs → exact diag → 匹配 PySCF CASCI within 0.01 mHa
- [ ] TEST (slow): `test_ccpvdz_cas10_8_skqd`
  - CAS(10,8) 3136 configs → Direct-CI SKQD → < 1 mHa vs CASCI
- [ ] TEST (slow): `test_ccpvdz_cas10_10_sparse`
  - CAS(10,10) 63504 configs → Sparse SKQD → < 1 mHa vs CASCI
  - 確認不 OOM（已有 CAS(10,10) OOM 保護）
- [ ] TEST (slow): `test_ccpvdz_cas10_14_nnci`
  - CAS(10,14) 28Q → NNCI + SKQD → < 2 mHa vs DMRG/CASCI reference
  - 這是第一個真正需要 sampling-based discovery 的 cc-pVDZ 測試
- [ ] VERIFY: 所有 test 在 DGX Spark 上跑得過（memory + time）

---

### PR-C2: cc-pVDZ Benchmark Report
**優先度**: MEDIUM | **預估**: 4-6h | **依賴**: PR-C1

- [ ] IMPL: `examples/ccpvdz_benchmark.py`
  - 跑 C1.1-C1.4 所有 active spaces
  - 記錄 energy, wall time, peak memory, basis size
  - 輸出 markdown table
- [ ] 更新 `RESULTS.md` 加入 cc-pVDZ section
- [ ] 比較分析:
  - vs PySCF CASCI reference
  - vs IBM SQD N2/cc-pVDZ 數據 (literature values)
  - NNCI 的 config discovery statistics (excitation rank distribution)

---

## Track D: 代碼清理

### PR-D1: Dead Code + Stale Docs
**優先度**: LOW | **預估**: 1-2h | **依賴**: 無（任何時候可做）

- [ ] 移除 `src/pipeline.py` 死 import:
  - `select_diverse_basis`
  - `davidson_eigensolver`
  - `adaptive_eigensolver`
  - `SampleBasedKrylovDiagonalization`
- [ ] 移除 `src/pipeline.py` vestigial config fields:
  - `use_davidson`
  - `use_ci_seeding`
- [ ] 更新 `SPEC.md`:
  - 移除不存在的 `complex_nqs.py` 引用
  - 移除不存在的 `discrete_flow.py` 引用
  - 移除不存在的 `training.py` 引用
  - 加入 `config_hash.py`, `hamiltonian_cache.py`, `memory_logger.py`, `connection_cache.py`
  - 加入 `nnci.py`, `perturbative_pruning.py`（如果 SPEC.md 還維護的話）
- [ ] 審計 `retain_graph=True`:
  - `physics_guided_training.py` 中的 backward calls
  - 確認是否真的需要（如果 NF 和 NQS 共享 computation graph）
  - 如不需要，移除以節省記憶體
- [ ] VERIFY: 430 existing tests pass

---

## 執行順序建議

```
Week 1:
  Day 1-2: PR-A1 (Lanczos fix) + PR-D1 (cleanup) — 平行
  Day 3-5: PR-B1 (NNCI integration) — 開始

Week 2:
  Day 1-2: PR-B1 (NNCI integration) — 完成
  Day 3-4: PR-A2 (Krylov adaptive) + PR-B2 (MP2 in Krylov) — 平行
  Day 5:   PR-C1 開始 (cc-pVDZ validation)

Week 3:
  Day 1-3: PR-C1 (cc-pVDZ validation) — 完成
  Day 4-5: PR-C2 (benchmark report)
```

---

## 之後的 Phase 4 長期 TODO（不在本 ADR 範圍）

> 僅作記錄，待本 ADR 完成後另開 ADR-003。

### Phase 4a: Autoregressive NF (QiankunNet-style)
- [ ] 研究 QiankunNet 開源代碼 (`github.com/xzzeng001/QiankunNet-VQE`)
- [ ] 設計 Decoder-only Transformer amplitude network
- [ ] 實現 Quaternary MCTS sampling with particle-number pruning
- [ ] 替換 ParticleConservingFlowSampler
- [ ] 在 N2/STO-3G 上驗證 ≥ 99% FCI correlation energy
- [ ] 在 N2/cc-pVDZ CAS(10,14) 上驗證

### Phase 4b: 40Q 端到端
- [ ] N2/cc-pVDZ CAS(10,20) = 40Q
- [ ] Autoregressive NF → SKQD → energy < 2 mHa vs DMRG

### Phase 4c: [2Fe-2S] Benchmark
- [ ] 安裝 block2 (DMRG-CASSCF)
- [ ] [2Fe-2S] CAS(30,20) 軌道優化 (DMRG-SCF)
- [ ] Pipeline 跑 NF+NNCI+SKQD → energy vs DMRG reference
- [ ] vs IBM SQD 數據比較

### 其他 NNCI 改進 (文獻啟發)
- [ ] CNN classifier option (來自 Schmerwitz NNCI)
- [ ] Natural orbital rotation (來自 NO-NNCI, arXiv:2510.27665)
- [ ] MPPT tensor-based candidate generation (來自 PIGen-SQD)
- [ ] Self-consistent refinement loop (來自 PIGen-SQD)

### Eigensolver 改進 (低優先度)
- [ ] PRIMME drop-in eigsh (`pip install primme`) for CAS systems
- [ ] CuPy eigsh NaN guard
- [ ] 降低 dense matrix_exp threshold 10000 → 2000
- [ ] SciPy v1.17.0 Krylov-based expm (when available)
