# NQS & VMC Optimization Survey: January-March 2026

> **Date**: 2026-03-09
> **Scope**: Neural Quantum States architectures, VMC convergence techniques, natural gradient / SR advances, sign problem handling, training efficiency for large active spaces (CAS(10,20)+)
> **Purpose**: 識別可直接應用於 pipeline VMC/NF 組件的最新技術，指導 Phase 6 開發方向
> **Complements**: `LITERATURE-SURVEY-2026-Q1.md` (SQD/Krylov/SCI 競爭格局), `VMC-SIGN-PROBLEM-RESEARCH.md` (sign 架構分析), `OPTIMIZATION-RESEARCH.md` (DGX Spark 硬體)

---

## Executive Summary

2026 Q1 NQS/VMC 領域的 7 個關鍵趨勢：

1. **NQS-SC > NQS-VMC 正式確立**：Selected Configuration 方法在能量精度和波函數係數上全面優於 VMC 優化，特別對靜態關聯系統。這**直接驗證我們的 Direct-CI + SKQD 路線**。
2. **Deterministic NQS 取代 stochastic VMC**：2026 年 1 月論文用自適應子空間 + PT2 校正取代 Monte Carlo 採樣，消除取樣噪聲。概念上接近我們的 SKQD。
3. **SPRING 優化器打敗 MinSR 和 KFAC**：在 40K 迭代內達到化學精度，MinSR/KFAC 在 100K 迭代後仍失敗。REINFORCE 在 NQS 文獻中幾乎不被使用。
4. **118 量子比特單 GPU**：利用 peaked wavefunctions 的稀疏性 + autoregressive sampling without replacement，在 10^15 Slater 行列式空間中超越 CCSD(T)。
5. **RetNet 替代 Transformer**：推理時間從 O(n²) 降到 O(n)，訓練保持平行化。SandboxAQ 發表。
6. **Loss landscape 比預期良性**：獨立訓練的 NQS 之間幾乎沒有能量壁壘（mode connectivity），暗示多次重啟不如一次好的優化。
7. **GPU-DMRG 基線持續進步**：CAS(82,82) 在單 DGX-H100 上數天完成。NQS 的價值在 CAS(100,100)+ 才開始體現。

**對 pipeline 的核心建議**：
- VMC 優化器從 REINFORCE 改為 MinSR/SPRING（P0 優先級）
- 考慮放棄純 VMC，轉向 NQS-SC 範式（neural corrector + subspace diag）
- AR 取樣加入 sampling without replacement
- 評估 RetNet 替代 Transformer

---

## 1. VMC Convergence Techniques for 20-40+ Qubit Systems

### 1.1 The Consensus: Stochastic VMC is the Bottleneck

2026 年初的共識是 **VMC 的隨機優化（非 ansatz 表達力）是分子系統 NQS 的主要障礙**。

#### A. Deterministic Framework — 完全消除 MC 噪聲

**[arXiv:2601.21310]** Zheng Che, Jan 2026

- 在動態自適應 configuration subspace 中優化 neural backflow ansatz
- PT2 校正確保子空間外的配置貢獻被計入
- Hybrid CPU-GPU 實現，計算成本相對子空間大小為 **sub-linear**
- 在 H₂O、N₂ 鍵斷裂、Cr₂ 強關聯系統驗證

**與我們 pipeline 的關係**：概念上接近 SKQD — 都是在子空間中做精確對角化，但加了 neural network corrector。差異是他用 backflow ansatz 參數化波函數而非直接 diag。

#### B. NQS-SC — Selected Configuration 優於 VMC

**[arXiv:2602.12993]** Feb 2026

系統性比較 NQS-VMC vs NQS-SC（selected configurations + NN fitting）：

| 性質 | NQS-VMC | NQS-SC |
|------|---------|--------|
| 能量精度 | 差，不穩定 | 好，systematically improvable |
| 波函數係數 | 不準確 | 準確 |
| 靜態關聯 | 困難 | 有效 |
| 動態關聯 | 無法有效捕捉 | 無法有效有效捕捉 |
| 系統性改進 | 不存在 | 存在 |

**結論**：NQS-SC 被定位為**新默認方法**。兩者都無法有效捕捉動態關聯 → 需要 multiconfigurational PT 等混合方法。

**對我們的意義**：直接驗證 Direct-CI + SKQD 路線。我們的 pipeline 本質上就是 NQS-SC 的變體（NF 選配置 + Krylov 對角化）。

#### C. Peaked Wavefunctions — 118Q on Single GPU

**[arXiv:2408.07625]** Reh et al., updated 2025

分子波函數的「尖銳性」（大部分振幅集中在少數配置）被轉化為優勢：

1. **Autoregressive sampling without replacement**：避免重複取樣，每個配置只取一次
2. **Local energy surrogate**：計算更便宜的局域能量代理
3. **Custom SR modification**：針對 peaked wavefunctions 的隨機重構改進
4. **GPU 優化**：高度優化的 GPU 實現

結果：
- 單 GPU 處理 118 量子比特分子
- 在 ~10¹⁵ Slater 行列式空間超越 CCSD(T)
- 比之前 NQS-VMC 工作加速 **10 倍以上**

**關鍵技術**：sampling without replacement 對 peaked wavefunctions 特別有效，因為少數高權重配置不需要重複取樣。

### 1.2 Implications for Our Pipeline

我們的觀察「VMC training does NOT converge at 24Q/40Q」完全符合領域共識：

1. **REINFORCE 是錯誤的優化器** — 領域使用 SR/MinSR/SPRING
2. **純 VMC 在分子系統上本質困難** — NQS-SC 方法更可靠
3. **Direct-CI + SKQD 是正確方向** — 在配置空間中做精確計算，而非依賴隨機優化

---

## 2. Architectures Beyond Autoregressive Transformers

### 2.1 RetNet — O(n) Inference for NQS

**[arXiv:2411.03900]** SandboxAQ, Nov 2024

Retentive Network (RetNet) 取代 Transformer：

| 特性 | Transformer | RetNet |
|------|-------------|--------|
| 訓練 | 平行 O(n²) | 平行 O(n²) |
| 推理 | O(n²) 注意力 | O(n) 遞迴 |
| KV Cache 記憶體 | O(n × d) 每層 | O(d²) 每層（固定） |
| 自回歸取樣 | 需 KV cache | 自然遞迴 |

**對我們的意義**：
- 當前 AR flow 用 KV-cached transformer，推理 O(n²)
- 40Q = 40 tokens，O(n²) 還可以接受
- 但如果要 scale 到 100Q+，RetNet 的 O(n) 推理成為必要
- **建議**：Phase 6 考慮 RetNet 架構，但 40Q 暫不急迫

### 2.2 Physics-Inspired Transformer Quantum States (PITQS)

**[arXiv:2602.03031]** Yamazaki et al., Feb 2026

將 Transformer NQS 解釋為**潛在虛時間演化的神經近似**：

- 標準 Transformer 對應含時有效 Hamiltonian → 物理上無動機
- PITQS 透過 **跨層權重共享** 強制靜態有效 Hamiltonian
- 用 **Trotter-Suzuki 分解**提升傳播精度，不增加參數
- 在 J₁-J₂ Heisenberg 模型達到 state-of-the-art，參數量更少

**物理洞察**：Transformer 的多層結構 ≈ 虛時間演化 e^{-βH} 的離散化，權重共享 ≈ H 不隨 β 變化。

**對我們的意義**：我們的 SKQD 用 e^{-iHΔt} 做實時間演化展開 Krylov 子空間。PITQS 用 e^{-βH} 做虛時間演化。兩者互補 — SKQD 在子空間中精確，PITQS 在表示上精確。

### 2.3 Neuralized Fermionic Tensor Networks (NN-fTNS)

**[arXiv:2506.08329]** Garnet Chan group (Caltech), Phys. Rev. B 2026

- 在 fermionic tensor network 的局部張量上加 **配置依賴的 NN 變換**
- 與標準 DMRG/TN 演算法相容
- 同 bond dimension 下，能量比純 fTN **好一個數量級**
- 可同時透過 bond dimension 和 NN 參數化系統性改進

**新的 sign 結構來源**：不依賴 Slater 行列式或 Pfaffian，而是從 tensor network 的反對稱結構自然產生。

### 2.4 Hybrid TN + NN / PyNQS

**[arXiv:2507.19276]** JCTC Oct 2025

- Bounded-Degree Graph Recurrent Neural Network (BDG-RNN) ansatz
- RBM correlators 增強表達力
- **Semi-stochastic local energy** 顯著降低計算成本
- 在 H₅₀ 鏈和 **[Fe₂S₂(SCH₃)₄]²⁻ 鐵硫簇** 達到化學精度
- 開源：**PyNQS** 套件

**直接相關**：[Fe₂S₂] 是我們的 40Q+ 目標系統之一。PyNQS 在這個系統上達到化學精度。

### 2.5 Transformer NQS for Composite Hilbert Spaces

**[arXiv:2603.02316]** March 2026

- 將 NLP tokenization 應用於**混合自旋+費米子**局部 Hilbert 空間
- DMRG 級別精度
- 延伸性：可處理非均勻局部結構

### 2.6 Architecture Summary for Our Pipeline

| 架構 | 適用場景 | 與我們的相關性 |
|------|---------|-------------|
| Autoregressive Transformer | 通用，30-50Q | 已實現，Phase 4a |
| RetNet | >50Q，推理速度關鍵 | Phase 6 候選 |
| PITQS (physics-inspired) | 需要虛時間演化解釋 | 理論參考 |
| NN-fTNS | 需要 TN 精度 + NN 靈活性 | 替代路線 |
| BDG-RNN (PyNQS) | 圖結構分子 | [Fe₂S₂] benchmark |
| NNBF (backflow) | 精度最優先 | 架構升級候選 |

---

## 3. Natural Gradient / Stochastic Reconfiguration Advances

### 3.1 Optimizer Hierarchy (2026 State-of-the-Art)

```
精度排名（分子系統）：
SPRING > MinSR ≈ KFAC（改進版）>> Adam >> REINFORCE

計算成本排名：
REINFORCE < Adam < MinSR < SPRING < 完整 SR

推薦（20-40Q 分子）：
SPRING（最佳平衡）或 MinSR（如果計算受限）
```

### 3.2 MinSR — Minimum-Step Stochastic Reconfiguration

**[Nature Physics, 2024]** Chen & Heyl

- 優化成本 **線性於參數數量**（vs 完整 SR 的二次）
- 使 64 層、10⁶ 參數的深層 NQS 可訓練
- 透過最小範數解替代 S⁻¹ 求解

```python
# MinSR 概念（簡化版）
# 標準 SR: delta_theta = S^{-1} @ grad_E  (O(p^2) for S construction)
# MinSR:    delta_theta = O^T @ (O @ O^T)^{-1} @ E_loc  (O(n_samples * p))
# 其中 O = Jacobian matrix (n_samples x n_params)
```

### 3.3 SPRING — 當前最佳 NQS 優化器

**Goldshlager et al., 2024-2025**

Subsampled Projected-Increment Natural Gradient Descent：

- 結合 MinSR + randomized Kaczmarz + momentum
- 在 **O 原子上 40K 迭代達到化學精度**（MinSR 和 KFAC 100K 後仍失敗）
- 首個收斂性證明
- 關鍵：momentum 項針對 natural gradient 特別設計

**收斂性比較（O 原子）**：

| 優化器 | 40K iterations | 100K iterations | Chemical accuracy? |
|--------|---------------|----------------|-------------------|
| SPRING | 達到 | — | YES |
| MinSR | 未達到 | 未達到 | NO |
| KFAC | 未達到 | 未達到 | NO |

### 3.4 Warm-Started SVD for SR

**[arXiv:2512.05749]** Dec 2025

- WSSR 演算法：iterative warm-started SVD 精化 SR preconditioner 的低秩近似
- 利用 SR 矩陣的低秩結構
- 可擴展到大參數空間（MinSR 的替代方案）

### 3.5 Implicitly Restarted Lanczos (IRL) for NQS

**[arXiv:2601.01437]** Liu & Dou, Jan 2026

**極其相關**：將 NQS 參數更新問題轉化為**小規模 Hermitian 特徵值問題**：

- 第二階優化框架
- 比標準 Adam/SR 更穩定
- 使 **淺層 NQS** 就能達到化學精度（不需要深層網路）
- 結合了 Lanczos（我們 SKQD 的核心）和 NQS 優化

**對我們的意義**：這是 Lanczos + NQS 的直接結合。概念上與 SKQD 互補：
- SKQD 用 Lanczos 在配置空間中展開子空間
- IRL 用 Lanczos 在**參數空間**中優化 NQS

### 3.6 Generalized NQS Lanczos

**[arXiv:2502.01264]** Wang et al., Feb 2025, Phys. Rev. B

- 監督學習表示 Lanczos 態 → NQS
- VMC 進一步優化
- 計算成本**線性增長**（vs RBM-Lanczos 的指數增長）
- 在 frustrated J₁-J₂ 模型中系統性改進

### 3.7 Functional Neural Wavefunction Optimization

**[arXiv:2507.10835]** Armegioiu et al., Jul 2025

統一框架，將無限維函數空間優化通過 Galerkin 投影轉化為參數空間：

- 統一 SR 和 Rayleigh-Gauss-Newton 方法
- 提供**幾何原則化的超參數選擇**
- 新演算法的理論推導

### 3.8 Practical Recommendations for Our VMC Trainer

```
當前實現：REINFORCE (src/flows/vmc_training.py)
問題：高方差，忽略參數空間幾何，無法收斂 24Q+

建議升級路徑：
1. [立即] MinSR：線性成本，已有開源實現 (github.com/ChenAo-Phys/MinSR)
2. [中期] SPRING：最佳精度，需要自己實現 momentum 項
3. [替代] IRL：如果要保持 Lanczos 為核心

具體改動：
- VMCTrainer._compute_gradient() 中替換 REINFORCE
- 需要計算 Jacobian matrix O_{ij} = ∂log ψ(x_i)/∂θ_j
- MinSR 只需 O 矩陣，不需要顯式構造 S = O^T O
- 記憶體：O(n_samples × n_params)，40Q AR transformer ~50K params → ~800MB @ 4K samples
```

---

## 4. Learning Rate Schedules & Optimization Tricks

### 4.1 What Works for VMC (Field Consensus)

1. **Warm-up 必要**：~2000 步線性 warmup（Adam β₂=0.999 的理論穩態）
2. **Scale-invariant pretraining**：預訓練用 scale-invariant loss，VMC 能量最小化本身已是 scale-invariant
3. **WSD schedule**：Warmup → Stable → Decay（借鑑 LLM 訓練）
4. **Norm-constrained updates**：穩定優化，允許更大學習率，改善跨系統可遷移性

### 4.2 QiankunNet Recipe (最成功的分子 NQS)

```
1. Physics-informed initialization：從 HF/CASSCF 軌道初始化
2. Adam optimizer (initial phase)
3. Switch to SR refinement (later phase)
4. Temperature annealing for sampling
5. Autoregressive sampling for exact log_prob
```

### 4.3 Peaked Wavefunctions Recipe

```
1. Autoregressive sampling without replacement
2. Custom SR modification（降低高權重配置的影響）
3. Local energy surrogate（避免完整 H matrix element 計算）
4. GPU-optimized implementation
```

### 4.4 Scale-Invariant Pretraining

**[JCP 2024]** Abrahamsen et al.

- VMC 能量最小化是 scale-invariant（ψ → cψ 不影響 E[H]）
- 但預訓練（例如 HF 波函數擬合）不是 scale-invariant
- 使用 scale-invariant pretraining loss 加速收斂
- 數學收斂保證

### 4.5 Basis Rotation Effects

**[arXiv:2512.17893]** Dec 2025

- NQS 性能**強烈依賴單粒子基底選擇**
- RBM 等淺層架構對基底旋轉特別敏感
- 好的初始基底（CASSCF natural orbitals）大幅改善收斂
- 壞的基底導致優化被困在 saddle points

**實踐建議**：在 VMC 之前做 orbital optimization（CASSCF → natural orbitals）。

---

## 5. Sign Problem Handling in Molecular VMC

### 5.1 State-of-the-Art Approaches (Summary)

| 方法 | 代表作 | Sign 來源 | 優劣 |
|------|--------|----------|------|
| Slater det + backflow | FermiNet, PsiFormer | det(φᵢ(rⱼ)) | 最準確，自動反對稱 |
| Amplitude + Phase MLP | QiankunNet | e^{iφ(x)} | 彈性，但 phase 難學 |
| Neural backflow (2Q) | NNBF 2502.18843 | Backflow 變換 | 二次量子化最佳 |
| NN nodal optimization | PRB 2025 | NN 學習 nodal surface | 超越 fixed-node DMC |
| NN-fTNS | Chan group 2025 | 張量網路反對稱 | 新的 sign 結構來源 |
| Sign network (ours) | Phase 4c | tanh(NN(x)) | 太弱，缺乏物理約束 |

### 5.2 Neural Network Nodal Structure Optimization

**[arXiv:2411.02257]** Phys. Rev. B, Oct 2025

- NN 在能量最小化過程中**自適應學習和優化 nodal structures**
- 量子點，最多 30 電子
- 基態能量超越 fixed-node DMC
- 關鍵洞察：NN 可以發現比傳統 trial functions 更好的 nodal surfaces

### 5.3 Mode Connectivity — No Energy Barriers

**[arXiv:2601.06939]** Jan 2026

- 獨立訓練的 NQS 之間**幾乎沒有能量壁壘**
- GeoNEB 路徑優化器（SR + nudged elastic band）
- 1.6M 參數 Psiformer，6 電子量子點
- 能量壁壘 ~10⁻⁵
- **物理 symmetry 沿路徑保持**

**含義**：
- Loss landscape 比預期更良性
- 不同隨機種子學到的 sign structure 本質等價
- 多次重啟不如一次好的優化
- 優化器的選擇（SR vs Adam）比初始化更重要

### 5.4 Implications for Our Sign Network

我們的 `sign_network.py` 使用 FC → GELU → tanh 架構。根據文獻：

1. **架構問題**：feedforward NN → tanh 無法捕捉強關聯波函數的 sign structure
2. **梯度問題**：direct backprop through E_loc，應改用 SR-based updates
3. **概念問題**：√p(x) × s(x) 不是標準分解 — 應考慮 |ψ| × e^{iφ} 或 backflow

**但是**：既然 NQS-SC > NQS-VMC，也許 sign network 的改進不是最高優先級。如果 Direct-CI + SKQD 已經足夠好，那麼 VMC/sign 只是可選的增強。

---

## 6. Training Efficiency for Large Active Spaces

### 6.1 QChem-Trainer on Fugaku

**[arXiv:2506.23809]** Jun 2025

NQS 訓練的高效能框架：
- **多層級並行**：sampling parallelism + local energy parallelism
- **Hybrid sampling**：混合取樣策略打破可擴展性障礙
- **KV cache 管理**：Transformer 的 key/value cache 記憶體控制
- **8.41x 加速**，95.8% 並行效率（1,536 節點）

**對我們的意義**：我們的 KV cache 實現 (Phase 4b) 方向正確。QChem-Trainer 的 cache 管理策略可參考。

### 6.2 Neural Importance Resampling (NIR)

**[arXiv:2507.20510]** Jul 2025

- 獨立的 **Sampling Neural Network (SNN)** 學習 NQS 分布
- 消除 MCMC mixing 問題
- 不限制 NQS 架構（取樣和波函數用不同 NN）
- 比 MCMC 更穩健，比 autoregressive 更靈活

**對我們的意義**：可以用作 NF sampling 的替代方案 — 訓練一個單獨的 SNN 來近似 NQS 分布，而不是用 NF 直接做波函數。

### 6.3 Semi-Stochastic Local Energy (PyNQS)

**[JCTC Oct 2025]**

- 顯著降低 local energy 計算成本
- 對二次量子化 Hamiltonian（O(n⁴) scaling）特別重要
- Truncated evaluations 保持精度

### 6.4 Compact Subspace Construction (NNBF)

**[arXiv:2502.18843]** Feb 2026

- 週期性 compact subspace 構建
- Truncated local energy evaluations
- Physics-informed modifications
- 對 >40 orbitals 至關重要

### 6.5 Efficiency Comparison Table

| 方法 | 最大規模 | 硬體 | 時間 | 精度 |
|------|---------|------|------|------|
| Peaked NQS (Reh) | 118Q, ~10¹⁵ dets | 1 GPU | hours | > CCSD(T) |
| QChem-Trainer | >100Q | 1536 nodes | — | ~FCI |
| QiankunNet | CAS(46,26) | multi-GPU | — | 99.9% FCI |
| GPU-DMRG (SandboxAQ) | CAS(82,82) | 1 DGX-H100 | days | near-FCI |
| NNBF enhanced | — | 1 GPU | — | > CCSD(T) |
| Our pipeline | CAS(10,20) 40Q | 1 DGX Spark | 23s SKQD | 14.2 mHa |

### 6.6 Key Bottleneck: O(n⁴) Local Energy

二次量子化 Hamiltonian 的 local energy 計算 scales as O(M⁴)（M = orbitals）。
- 20 orbitals: 160K terms
- 40 orbitals: 2.56M terms
- 80 orbitals: 40.96M terms

**解法**（文獻中使用的）：
1. **Truncated evaluations**：只計算最大的 matrix elements
2. **Surrogate local energy**：用近似值取代精確計算
3. **Semi-stochastic**：部分精確 + 部分隨機
4. **我們的解法**：get_connections + Slater-Condon 規則只計算非零項（已有 Numba JIT）

---

## 7. Competitive Landscape Update (NQS-Specific)

### 7.1 Key Players and Their 2026 Position

| Group | Method | Latest Scale | Status |
|-------|--------|-------------|--------|
| Google DeepMind | FermiNet/PsiFormer | ~30e, first Q | No 2026 updates found |
| Caltech (Chan) | NN-fTNS | Hubbard model | PRB 2026, promising |
| SandboxAQ | RetNet NQS + GPU-DMRG | CAS(82,82) | Dual approach |
| QiankunNet team | AR Transformer | CAS(46,26) | Nature Comms 2025 |
| IBM | PIGen-SQD | 72 qubits (quantum) | Science 2026 (SqDRIFT) |
| Reh et al. | Peaked NQS | 118Q single GPU | arXiv 2024-2025 |
| NetKet community | Various (JAX) | Lattice models | Active development |
| NNCI (FAU group) | NN classifier + CI | N₂, 4×10⁵ dets | JCTC 2025 |

### 7.2 Open-Source Frameworks (2026)

| Framework | Language/Backend | Strength | URL |
|-----------|-----------------|----------|-----|
| NetKet | JAX | Lattice NQS, SR/MinSR | netket.org |
| DeepQMC | PyTorch | First-Q VMC | github.com/deepqmc/deepqmc |
| FermiNet | JAX | First-Q, PsiFormer | github.com/google-deepmind/ferminet |
| PyNQS | — | Hybrid TN+NN, molecular | New (2025) |
| QChem-Trainer | — | HPC-scale NQS | Fugaku-specific |

---

## 8. Actionable Recommendations for Our Pipeline

### Priority 1 (Immediate — Fixes Known Convergence Failure)

**P1.1: Replace REINFORCE with MinSR in VMCTrainer**
- 影響：修復 24Q+ VMC 不收斂問題
- 實現：計算 Jacobian O, 用 O^T(OO^T)^{-1} E_loc 取代 REINFORCE gradient
- 參考：github.com/ChenAo-Phys/MinSR
- 估計工作量：2-3 天

**P1.2: Add sampling without replacement to AR flow**
- 影響：減少重複取樣，提升取樣效率
- 實現：在 AutoregressiveFlowSampler.sample() 中追蹤已取樣配置
- 參考：arXiv:2408.07625
- 估計工作量：1 天

### Priority 2 (Near-term — Architecture Improvements)

**P2.1: Natural orbital basis for NQS initialization**
- 使用 CASSCF natural orbitals 而非 canonical HF orbitals
- 改善 NQS 收斂性（arXiv:2512.17893）
- 已有 PySCF CASSCF infrastructure

**P2.2: Evaluate NQS-SC paradigm**
- 不做純 VMC，而是：NF 選配置 → NN 修正振幅 → SKQD 對角化
- 這本質上是我們已有架構的理論驗證
- 可加入 PT2 校正（arXiv:2601.21310 的思路）

### Priority 3 (Medium-term — Scaling)

**P3.1: RetNet architecture evaluation**
- 對 40Q 不急迫，對 100Q+ 必要
- 推理時間 O(n) vs O(n²)

**P3.2: Semi-stochastic local energy**
- 降低 E_loc 計算成本
- 對 >40 orbitals 重要

### Priority 4 (Long-term — Research)

**P4.1: Physics-inspired transformer (PITQS)**
- 跨層權重共享 → 減少參數
- 理論上更好（靜態有效 Hamiltonian）

**P4.2: Neural backflow in second quantization**
- NNBF 是當前二次量子化最佳方法
- 架構改動大，但精度最高

---

## Appendix: Paper Reference List

### 2026 Papers (January-March)

1. **[2602.12993]** Neural Quantum States Based on Selected Configurations — NQS-SC > NQS-VMC
2. **[2601.21310]** A Deterministic Framework for Neural Network Quantum States — 無 MC 噪聲
3. **[2602.03031]** Physics-inspired transformer quantum states via latent ITE — PITQS
4. **[2601.01437]** Implicitly Restarted Lanczos for Chemically-Accurate Shallow NQS
5. **[2601.06939]** No Energy Barrier Between Independent Fermionic NQS Minima
6. **[2502.01264]** Generalized Lanczos method for NQS optimization — Phys. Rev. B
7. **[2502.18843]** Efficient optimization of neural network backflow — surpasses CCSD(T)
8. **[2603.02316]** Transformer NQS for lattice spins and fermions — March 2026
9. **[2601.15169]** Assessing Orbital Optimization in VMC and DMC

### 2025 Papers (Key)

10. **QiankunNet** — Nature Communications: Transformer NQS, CAS(46,26), 99.9% FCI
11. **Physics-informed transformers** — Nature Communications: Interpretable NQS
12. **[2408.07625]** Peaked molecular wavefunctions: 118Q single GPU
13. **[2512.05749]** Warm-Started SVD for SR
14. **[2512.17893]** Basis rotation effects on NQS
15. **[2507.19276]** Hybrid TN+NN / PyNQS — JCTC, [Fe₂S₂]
16. **[2506.08329]** Neuralized fermionic tensor networks — PRB 2026
17. **[2506.23809]** QChem-Trainer on Fugaku — HPC NQS
18. **[2507.20510]** Neural Importance Resampling — decoupled sampling
19. **[2411.03900]** RetNet NQS — SandboxAQ
20. **[2512.06858]** PIGen-SQD — IBM RBM + tensor decomposition
21. **MinSR** — Nature Physics 2024: Linear-cost SR
22. **SPRING** — Goldshlager et al.: Best NQS optimizer
23. **[2411.02257]** Neural nodal structure optimization — PRB
24. **SandboxAQ GPU-DMRG** — JCTC: CAS(82,82)
25. **[2507.10835]** Functional Neural Wavefunction Optimization — unified framework

### Frameworks

26. **NetKet** — netket.org — JAX NQS toolkit
27. **DeepQMC** — deepqmc.org — PyTorch first-Q VMC
28. **FermiNet** — google-deepmind/ferminet — JAX
29. **PyNQS** — New 2025, hybrid TN+NN
30. **QChem-Trainer** — Fugaku HPC NQS
