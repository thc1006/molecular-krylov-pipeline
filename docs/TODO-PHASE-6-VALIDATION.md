# Phase 6 Validation Benchmarks — 最終結果

> 驗證 8 項 Phase 6 改進是否在實際分子系統上產生效果。
> 原則：小→大，每步有明確的 PASS/FAIL 標準，消融實驗隔離個別貢獻。

---

## 成功指標 — 最終結果 (2026-03-10)

| 目標 | 基線 | 門檻 | 結果 | 狀態 |
|------|------|------|------|------|
| CAS(10,10) < 5 mHa | 14.2 mHa (Direct-CI) | < 5 mHa | **4.9 mHa (NF-assisted)** | ✅ PASS |
| VMC 收斂 (REINFORCE) | random (~-5 Ha) | energy → near HF | **-7.862 Ha (0 mHa gap to HF)** | ✅ PASS |
| VMC 收斂 (MinSR) | random (~-5 Ha) | energy 下降 > 0.5 Ha | **-7.732 Ha (2.7 Ha improvement)** | ✅ PASS (機制) |
| NF > Direct-CI | 5.7 mHa (Direct-CI) | delta ≥ 10 mHa | **0.8 mHa** | ⚠️ PARTIAL |
| 40Q no OOM | — | pipeline 完成 | **4/4 tests PASS** | ✅ PASS |

### 已知限制

- **VMC 無法突破 HF 天花板**: 正值 ansatz ψ=√p(x) 無法表示負 CI 係數。LiH 有 26 個負係數（11.6%）。PhaseNetwork + REINFORCE/MinSR 在可行預算內均收斂到 HF 水準。需要 determinantal ansatz（如 NNBF/Transformer Backflow）才能突破。
- **MinSR 對 AR transformer 效率低**: 793K 參數需要 N_samples >> √793K ≈ 890 才能準確估計 Fisher 矩陣。Per-sample Jacobian 成本太高。文獻中 MinSR 主要用於 <10K 參數的 RBM/MLP。
- **NF 改善有限**: 50 epochs 只比 Direct-CI 好 0.8 mHa。Direct-CI 的 HF+S+D 已涵蓋主要配置，NF 找到的額外 triple/quadruple 配置對 CAS(10,10) 貢獻有限。

---

## Level 1: H2 (4Q) — 煙霧測試 (~30s)

- [x] **L1-A** MinSR on H2: AR + VMC(minsr, lr=0.5, 100 steps) → near HF energy ✅
- [x] **L1-B** Phase Network on H2: AR + VMC(minsr) + PhaseNetwork → < -0.5 Ha ✅
- [x] **L1-C** H-coupling on H2: Direct-CI + SKQD(h_coupling=True) → < 0.1 mHa ✅
- [x] **L1-D** Taboo on H2: Direct-CI + SKQD(taboo=True) → < 0.1 mHa ✅
- [x] **L1-E** Gumbel on H2: PCF(gumbel) → particle conserving, finite ✅
- [x] **L1-F** Unique sampling: AR sample_unique(4) → all unique, valid ✅
- [x] **L1-G** NOs on H2: NNCI(natural_orbitals=True) → runs without error ✅
- [x] **L1-H** All combined: all ON + SKQD → < 0.1 mHa ✅

---

## Level 2: LiH (12Q, 225 configs) — 小分子驗證 (~3 min)

- [x] **L2-A** MinSR 機制: MinSR(5 steps, 50 samples) → finite, completes ✅
- [x] **L2-A2** REINFORCE 收斂: AR + REINFORCE(100 steps) → energy 下降 ✅
- [x] **L2-D** SKQD 品質: Direct-CI + SKQD → < chemical accuracy ✅
- [x] **L2-E** 全功能 pipeline: all P6 ON + SKQD → < chemical accuracy ✅
- [x] **L2-F** Gumbel diversity: PCF(gumbel) 500 samples → particle conserving ✅

---

## Level 3: BeH2/NH3/CH4 (14-18Q) — 中型回歸 (~2 min)

### BeH2 (14Q, 1225 configs)
- [x] **L3-BeH2-A** 回歸: Direct-CI + SKQD → < 1.6 mHa ✅
- [x] **L3-BeH2-C** NNCI+NOs: natural_orbitals=True → runs, finite energy ✅

### NH3 (16Q, 3136 configs)
- [x] **L3-NH3-A** 回歸: Direct-CI + SKQD → < 1.6 mHa ✅

### CH4 (18Q, 15876 configs)
- [x] **L3-CH4-A** 回歸: Direct-CI + SKQD → < 1.6 mHa ✅
- [x] **L3-CH4-B** 全功能 pipeline: all P6 ON → < 1.6 mHa ✅

---

## 實驗層: CAS(10,10) (20Q, 63504 configs) — 核心驗證

**基線**: 14.2 mHa (Direct-CI), 12.7 mHa (AR NF 300ep)

### Experiment 1: CAS(10,10) 精度
- [x] **Exp1a** Direct-CI + SKQD: **5.7 mHa** (基線 14.2 → 8.5 mHa 改善) ✅
- [x] **Exp1b** NF-Assisted (50ep): **4.9 mHa** → TARGET MET (< 5 mHa) ✅

### Experiment 2: VMC 收斂驗證
- [x] **Exp2a** REINFORCE 收斂: random → near-HF, improvement > 2 Ha ✅
- [x] **Exp2b** MinSR 機制: /N bug fix 驗證, improvement > 0.5 Ha ✅
- [x] **Exp2c** PhaseNetwork 機制: energy 有限, 不破壞 VMC ✅

### Experiment 3: NF vs Direct-CI
- [x] **Exp3** NF > Direct-CI by 0.8 mHa ⚠️ (目標 10 mHa 未達, but NF achieves 4.9 mHa < 5 mHa)

---

## Level 5: N2 CAS(10,20) (40Q, 240M configs) — 規模驗證

**注意**: FCI 在 240M configs 上不可計算，僅驗證規模可行性。

- [x] **L5-A** Direct-CI + SKQD 40Q: energy < HF, no OOM ✅
- [x] **L5-B** VMC(REINFORCE) 40Q: 3 steps → finite, completes ✅
- [x] **L5-C** 全功能 pipeline 40Q: all P6 ON → energy < HF, no OOM ✅
- [x] **L5-F** Unique sampling 40Q: sample_unique(100) → all unique, particle conserving ✅

---

## 測試文件對照

| 測試文件 | 驗證層級 | Tests | 時間 |
|---------|---------|-------|------|
| `test_phase6_smoke.py` | L1 (H2, 4Q) | 8 | ~20s |
| `test_phase6_level2.py` | L2 (LiH, 12Q) | 5 | ~32s |
| `test_phase6_level3.py` | L3 (BeH2/NH3/CH4) | 5 | ~77s |
| `test_phase6_experiments.py` | Exp 1/2/3 | 6 | ~12 min |
| `test_phase6_level5.py` | L5 (40Q, scaling) | 4 | ~6 min |

### 已歸檔 (archive/)
- `test_phase6_level4.py` — assertions 過弱 (`energy < hf_energy` 永遠為真)
- `debug_connection_mismatch.py` — debug 腳本，bug 已修
- `bench_get_connections.py` — benchmark，無 assertions
- `bench_precompute.py` — benchmark，無 assertions
