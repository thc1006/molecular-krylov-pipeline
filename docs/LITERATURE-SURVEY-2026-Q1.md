# Literature Survey: Quantum Chemistry Advances (January--March 2026)

> **Date**: 2026-03-09
> **Scope**: SQD/SqDRIFT, Krylov subspace methods, Selected CI + ML, hybrid quantum-classical at scale, GPU-DMRG
> **Purpose**: 評估競爭格局，識別我們 pipeline 可採用的改進策略

---

## 1. Executive Summary

2026 年 Q1 的核心趨勢：

1. **SqDRIFT 登上 Science**：IBM 用 72-100 量子比特驗證半 Mobius 分子 (C13Cl2) 的電子拓撲，active space 32 electrons。這是 sample-based quantum diag 首次用於實際科學發現。
2. **QSCI/SQD 的 "fatal flaw" 被正式發表**：Reinholdt et al. (JCTC 2025) 證明 ground-state sampling 有不可避免的效率-精度 trade-off。**驗證我們的發現**：Direct-CI + Krylov 與 NF-guided 表現相當。
3. **Hamiltonian-guided autoregressive SCI (HAAR-SCI)** 達到 chemical accuracy：Gated Transformer + Gumbel Top-K，比 Krylov 更直接。
4. **NQS-SC > NQS-VMC**：NQS-based selected configuration 優於 VMC，驗證我們 NF+diag 路線優於 VMC training。
5. **GPU-DMRG 達到 CAS(82,82)**：SandboxAQ 在 DGX-H100 上一天完成 CASSCF 軌道優化。CAS(30,20) 對 DMRG 是 trivial。
6. **GPU 加速 SQD 後處理**：兩篇 Jan 2026 論文實現 40-95x speedup。

**對我們的影響**：
- Krylov time evolution 仍然是有效的 subspace expansion 策略
- HAAR-SCI 的 Hamiltonian-guided selection 是值得考慮的替代方案
- NQS-SC 論文驗證了 NF+diag > VMC 的路線
- 自然軌道 (NO) 基底可能提升 NNCI 效率
- Rotation thresholding 可改善 Krylov 矩陣的數值穩定性

---

## 2. Sample-Based Quantum Diagonalization (SQD) — IBM 生態系

### 2.1 SqDRIFT (Science, March 2026)

**論文**: "Quantum chemistry with provable convergence via randomized sample-based Krylov quantum diagonalization"
**arXiv**: [2508.02578](https://arxiv.org/abs/2508.02578) (revised Jan 2026)
**應用**: Science Mar 2026 — 半 Mobius 分子 C13Cl2

**核心思想**: 用 qDRIFT stochastic compilation 取代 coherent Trotter time evolution。每個電路隨機選擇 Hamiltonian terms，機率正比於其係數。

**優勢**:
- 保留 SKQD 的收斂保證
- 大幅降低 circuit depth
- 有效抑制硬體噪聲

**應用結果**: IBM 用 72-100 qubits 驗證了一個從未被合成過的分子 (C13Cl2) 的電子拓撲結構。Active space = 32 electrons，遠超精確對角化 (~18e) 的極限。

**與我們的關係**: SqDRIFT 是量子硬體上的演算法。我們的古典模擬中，Krylov time evolution via scipy expm_multiply 已經是高效的。SqDRIFT 的收斂保證分析對我們仍有參考價值。

### 2.2 GPU 加速 SQD 後處理 (Jan 2026)

兩篇同時出現的論文解決了 SQD 的古典瓶頸：

**Paper 1: OpenMP Offload** — [arXiv 2601.16169](https://arxiv.org/abs/2601.16169)
- Frontier 超算上達到 **95x speedup** (AMD MI250X)
- MI300X 額外 **3x** 提升
- 作者: Walkup, Järkkä, Pasichnyk, Streeter et al.

**Paper 2: Thrust-based SBD** — [arXiv 2601.16637](https://arxiv.org/abs/2601.16637)
- NVIDIA GPU 上 **~40x speedup**
- 重構 configuration processing、excitation generation、matrix-vector ops
- 作者: Doi, Shirakawa, Kawashima, Yunoki, Horii

**與我們的關係**: 我們的 `_build_hamiltonian_in_basis_gpu()` 已用 sparse CSR。但 Thrust-based excitation generation 的思路可以加速 `get_connections_vectorized_batch()`。

### 2.3 PIGen-SQD (Dec 2025)

**arXiv**: [2512.06858](https://arxiv.org/abs/2512.06858)
**核心**: RBM + implicit low-rank tensor decompositions + perturbative configuration screening

**關鍵結果**:
- 比標準 SQD **減少 70% diagonalization subspace**
- 同時能量精度提高數個數量級
- 在 IBM Heron R2 上驗證

**與我們的關係**: PIGen-SQD 的 tensor decomposition 用於識別重要配置的思路，可與我們的 MP2 pruning (`compute_mp2_amplitudes()`) 互補。

### 2.4 SQD 擴展應用

| 論文 | 系統 | Qubits | 方法 | 新穎點 |
|------|------|--------|------|--------|
| Implicit Solvent SQD ([2502.10189](https://arxiv.org/abs/2502.10189)) | MeOH, EtOH, H2O in water | 27-52 | SQD + IEF-PCM | 首次 SQD 溶劑效應 |
| Band Gap SQD ([2503.10901](https://arxiv.org/abs/2503.10901)) | HfO2, ZrO2 | periodic | SQD + LUCJ | 週期性材料能帶 |
| Symmetry-Adapted SQD ([2505.00914](https://arxiv.org/abs/2505.00914)) | Hubbard ladder | 156Q IBM | SQD + point group | 對稱性後處理 |
| QBE-SQD (ChemRxiv Feb 2026) | H8/cc-pVDZ | 43-67 | QBE + SQD | fragment (8e,19o) > full (8e,30o) |
| ph-AFQMC + SQD ([2503.05967](https://arxiv.org/abs/2503.05967)) | N2, [2Fe-2S] | 26-40o | AFQMC + SQD trial | 恢復 O(100) mHa correlation |
| Lockheed+IBM (JCTC 2025) | CH2 open-shell | 52 | SQD | 首次 open-shell SQD |
| Cuprate chain ([2512.04962](https://arxiv.org/abs/2512.04962)) | 2-6 Cu-O plaquettes | varies | SQD | 噪聲反而改善收斂 |

### 2.5 QSCI/SQD 的根本缺陷

**Paper**: Reinholdt et al., "Critical Limitations in Quantum-Selected Configuration Interaction Methods"
**Published**: JCTC 2025, [arXiv 2501.07231](https://arxiv.org/abs/2501.07231)

**核心發現**:
1. QSCI 的 CI expansions **不如古典 SCI heuristics 緊湊**
2. Sampling 反覆選擇已見過的配置 — **邊際效益遞減**
3. 存在 **不可避免的 trade-off**：找到足夠多 determinants vs 生成緊湊的 CI expansion
4. 在 N2 和 [2Fe-2S] 上測試 — QSCI 落後於古典方法

**對我們的直接影響**:
- 驗證我們的發現：NF sampling 在 40Q 上與 Direct-CI + Krylov 表現相當
- Ground-state sampling 本質上是自我挫敗的：最好的配置已經是最 probable 的
- **Krylov time evolution 是根本不同的 expansion 策略**，不受此 fatal flaw 影響
- NNCI active learning 也是不同策略（classifier-guided，非 probability-weighted sampling）

---

## 3. Krylov Subspace Methods

### 3.1 SKQD 實際硬體 85 Qubits (Jan 2025)

**arXiv**: [2501.09702](https://arxiv.org/abs/2501.09702)
**作者**: Jeffery Yu et al. (IBM + ORNL)

**關鍵結果**:
- **85 qubits, 6000 two-qubit gates** on IBM processor
- Single-impurity Anderson model: 42 electrons in 42 orbitals
- 最大的 SKQD 硬體實驗

**演算法特點**:
- 結合 Krylov 子空間構建（收斂保證）+ sample-based 技術（噪聲 resilience）
- 多項式時間收斂（假設 Krylov 方法的標準條件 + ground state 稀疏性）
- 在 shot noise 下 SKQD > KQD

### 3.2 Rotation Thresholding (Feb 2026)

**Paper**: "A New Angle on Quantum Subspace Diagonalization for Quantum Chemistry"
**arXiv**: [2602.11985](https://arxiv.org/abs/2602.11985)
**作者**: De Vriendt, Bringewatt, Gjonbalaj, Ostermann, Vodola, Borregaard, Kühn, Yelin

**核心創新**: Eigenvector-preserving rotations of the generalized eigenvalue problem **before** thresholding.

**關鍵結果**:
- 在工業相關的 Fe(III)-NTA chelate complex 上測試
- 某些系統和噪聲條件下 **減少所需 samples 100 倍**
- 用啟發式方法從 noisy data 選擇旋轉角度

**對我們的影響**: 我們的 SKQD 中 Krylov overlap/Hamiltonian matrices 的 conditioning 問題 (`_last_ill_conditioned`) 可以用這個方法改善。目前我們用 `convergence_threshold=1e-5` 和 SVD-based condition check，rotation thresholding 是更好的替代。

### 3.3 Generative Krylov Subspace (GenKSR) (Dec 2025)

**arXiv**: [2512.19420](https://arxiv.org/abs/2512.19420)

**核心思想**: 訓練 generative model（Transformer 或 Mamba SSM）學習整個 Krylov diagonalization 過程。

**能力**:
- 為未見過的 Hamiltonian 生成 Krylov subspace samples
- 生成比訓練資料更大的 subspace dimensions
- 15-qubit 1D, 16-qubit 2D Heisenberg, 20-qubit XXZ (IBM hardware)

**與我們的關係**: 概念上類似我們的 NF 方法，但在 Krylov state space 而非 configuration space 操作。如果 generative model 足夠準確，可以同時取代 NF sampling 和顯式 time evolution。但目前僅在 spin 模型上驗證，分子系統未測試。

### 3.4 Quantum Block Krylov Subspace Projector (QBKSP)

**核心改進**:
- **線性 scaling** with Krylov iterations（vs 標準方法的二次方）
- 對 real Hamiltonians，**reference 數量的 scaling 減半**
- 能計算 degenerate low-lying eigenenergies

**與我們的關係**: 我們的 SKQD 是 single-reference。多 reference 方法對 multi-reference 系統（如 Cr2、[2Fe-2S]）可能更有效。

---

## 4. Selected CI + Neural Network Methods

### 4.1 GTNN-SCI (JCTC 2025)

**Paper**: "Accelerating Many-Body Quantum Chemistry via Generative Transformer-Enhanced Configuration Interaction"
**Published**: [JCTC 21:11989-12000](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01429)

**核心**: Transformer self-attention 捕獲長程電子關聯，generatively sample important configurations。

**結果**:
- **10x speedup** over 先前的 NN-SCI 方法
- N2, H2O, C2 with cc-pVDZ
- 比先前 NN-SCI 更快收斂、更低能量

**與我們的關係**: 最接近的架構競爭者。他們用 self-attention 捕獲關聯，我們用 causal mask 的 decoder-only transformer。差異：他們直接 sample + diag（SCI 風格），我們用 Krylov expansion。

### 4.2 HAAR-SCI (JCTC 2025) ⭐ 重要競爭者

**Paper**: "Hamiltonian-Guided Autoregressive Selected-Configuration Interaction Achieves Chemical Accuracy in Strongly Correlated Systems"
**Published**: [JCTC 21:12622-12633](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415)

**工作流程**: learn → sample → compress
1. Gated Transformer autoregressively sample determinants
2. Gumbel Top-K noise 鼓勵 exploration
3. GPU min-heap kernels 只保留最大 Hamiltonian coupling 的配置
4. 網路 retrain，迭代直到連續能量差 ≤ 1 mHa

**關鍵優勢**:
- 在單個 GPU 上執行
- 用 **Hamiltonian coupling** 直接引導選擇（不是 probability-weighted sampling）
- 達到 chemical accuracy in strongly correlated systems

**與我們的 pipeline 比較**:

| 面向 | HAAR-SCI | Our Pipeline |
|------|----------|-------------|
| 配置生成 | Gated Transformer + Gumbel | AR Transformer + Plackett-Luce |
| Exploration 機制 | Gumbel Top-K noise | Entropy regularization |
| 配置選擇 | H coupling (min-heap) | Krylov expansion + importance ranking |
| 迭代 | retrain until ΔE ≤ 1 mHa | Single-pass Krylov |
| Diagonalization | Selected CI | SKQD (Krylov subspace) |

**可借鑒的改進**:
1. **Gumbel Top-K** 替代 Plackett-Luce 的 exploration
2. **Hamiltonian coupling scoring** 可以加入 `_rank_and_truncate_basis()` 的 ranking
3. **迭代 retrain** 可以加入 pipeline（目前是 single-pass）

### 4.3 ML-Assisted SCI (JCTC 2026)

**Published**: [JCTC 22(4)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01652)
**方法**: Binary ML classifier 引導 perturbative SCI

比 Transformer 方法更簡單但有效。類似我們的 NNCI module (`src/krylov/nnci.py`)。

### 4.4 NQS-SC (Feb 2026) ⭐ 驗證我們的路線

**arXiv**: [2602.12993](https://arxiv.org/abs/2602.12993)
**Paper**: "Neural Quantum States Based on Selected Configurations"

**核心發現**: NQS-based selected configuration (NQS-SC) **優於 NQS-VMC**：
- 能量精度更高
- 波函數係數更準確
- 特別在 **statically correlated molecules** 上優勢明顯
- 具有 robust systematic improvability（VMC 沒有）

**結論**: "NQS-SC 是電子結構計算的新預設方法，取代 NQS-VMC"

**對我們的直接影響**:
- **驗證我們的 NF + diag > VMC 路線**
- 我們 VMC training 在 24Q/40Q 不收斂（best = HF）是已知的普遍問題
- 建議：將 `use_vmc_training` 保留為可選，預設關閉
- NF 的真正價值在於 configuration selection，不在 energy optimization

### 4.5 NO-NNCI (Oct 2025)

**arXiv**: [2510.27665](https://arxiv.org/abs/2510.27665)
**Paper**: "Natural-Orbital-Based Neural Network Configuration Interaction"

**核心**: 用 approximate natural orbitals（1-RDM 的 eigenfunctions）取代 canonical HF orbitals。

**結果**: H2O, NH3, CO, C3H8 上一致減少所需 determinants 數量。

**可行改進**: 在我們的 NNCI module 前加入 natural orbital rotation。PySCF 的 `mcscf.CASSCF` 已經做了軌道優化，所以 CAS 計算中這可能已經部分實現。但對 Direct-CI（HF 軌道）的改進空間更大。

### 4.6 Deterministic NQS (Jan 2026)

**arXiv**: [2601.21310](https://arxiv.org/abs/2601.21310)

**核心**: 消除 MC noise — 在 dynamically adaptive configuration subspaces 中確定性優化 neural backflow ansatz + PT2 correction。

**結果**: H2O, N2, Cr2 in large Hilbert spaces. Sub-linear scaling via hybrid CPU-GPU.

**與我們的關係**: 替代 VMC training 的候選方案。確定性方法避免了 VMC 的 sampling variance 問題。

### 4.7 IRL for NQS Training (Jan 2026)

**arXiv**: [2601.01437](https://arxiv.org/abs/2601.01437)
**Paper**: "Implicitly Restarted Lanczos Enables Chemically-Accurate Shallow Neural Quantum States"

**核心**: 用 Implicitly Restarted Lanczos (IRL) 取代 Adam/SR 作為 NQS training engine。

**優勢**: 將 ill-conditioned parameter update 轉化為小規模 Hermitian eigenvalue problem。

**與我們的關係**: 我們的 AR flow 用 Adam optimizer + REINFORCE。IRL 可能更穩定，但需要 energy landscape 的 Hessian 資訊。

### 4.8 Efficient NQS Backflow (Feb 2025)

**arXiv**: [2502.18843](https://arxiv.org/abs/2502.18843)

**核心**: 一系列 algorithmic enhancements — efficient periodic compact subspace construction, truncated local energy evaluations, improved stochastic sampling, physics-informed modifications.

**結果**: 超越 CCSD, CCSD(T), 其他 NQS 方法。與 HCI, ASCI, FCIQMC, DMRG 競爭。

---

## 5. GPU-DMRG 進展

### 5.1 當前 CAS 規模紀錄

| 方法 | CAS 規模 | 系統 | 硬體 | 時間/性能 |
|------|---------|------|------|----------|
| GPU DMRG-SCF | **CAS(82,82)** | PAH, Fe-S complexes | DGX-H100 | ~1 天 |
| GPU DMRG single-point | CAS(113,76) | Cytochrome P450 | DGX-H100 | 246 TFLOPS |
| Multi-GPU DMRG | CAS(114,73) D=14000 | Nitrogenase P-cluster | 48× A100 | 80× vs CPU |
| SHCI | 28e/76o (2B dets) | Cr2 | CPU cluster | — |
| FCIQMC-GAS | 96e/159o | — | CPU cluster | — |

### 5.2 SandboxAQ GPU-DMRG 詳情

**Paper**: [JCTC 2024](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00903)

- **Quarter petaFLOPS** (246 TFLOPS sustained) on single DGX-H100 node
- **80× acceleration** vs 128-core CPU
- **2.5× improvement** over DGX-A100
- 與 ORCA 程式整合，spin-adapted DMRG

**CAS(82,82) DMRG-SCF** (March 2025, [arXiv 2503.20700](https://arxiv.org/abs/2503.20700)):
- 完全軌道優化的 CASSCF 在 ~1 天內完成
- 遠超任何其他 state-of-the-art 方法
- 消除了 active space 選擇問題（可以直接算大 CAS）

### 5.3 Multi-GPU DMRG: Nitrogenase P-cluster

**Paper**: [JCTC 2024](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01228)

- **D = 14,000** on 48× A100 (80 GB SXM)
- CAS(114,73) — 比之前的計算大 ~3×
- 解決多核過渡金屬（含退化 d/f 軌道）的挑戰

### 5.4 "Trivially Classical" 的界限 (2026 年 3 月)

| CAS 規模 | DMRG 可行性 | 備註 |
|----------|-----------|------|
| CAS(30,20) | ✅ Trivial | 單節點 GPU，分鐘級 |
| CAS(50,50) | ✅ 可行 | 單節點 DGX-H100，小時級 |
| CAS(82,82) | ✅ 可行 | 單節點 DGX-H100，~1 天 |
| CAS(100,100) | ⚠️ 邊界 | 需多節點 GPU |
| CAS(114,73) | ✅ 已示範 | 48× A100 multi-GPU |
| CAS(150+,150+) | ❌ 未達 | 當前 quantum/NF advantage threshold |

**結論**: CAS(30,20) 的 [2Fe-2S] 對 DMRG 完全是 trivial。量子/NF 的優勢門檻在 CAS(100+,100+)。但對 **SCI 類方法**（SHCI, FCIQMC），門檻較低，因為 SCI 不像 DMRG 那樣受益於 1D 纏結結構。

---

## 6. Hybrid Classical-Quantum: 40-100 Qubit Scale 的方法

### 6.1 當前規模成就

| 方法 | Qubits | 系統 | 參考 |
|------|--------|------|------|
| SqDRIFT | 72-100 | half-Mobius C13Cl2 (32e) | Science Mar 2026 |
| SKQD | 85 | Anderson model (42e, 42o) | arXiv 2501.09702 |
| SQD | 52-67 | CH2 open-shell, H8 | Lockheed/IBM, QBE-SQD |
| SQD+DMET | 41 | Ligand-like molecules | arXiv 2511.22158 |
| ADAPT-GCIM | varies | N2, FeS, [2Fe-2S], U2 | arXiv 2601.10813 |

### 6.2 40-100Q Scaling 策略

1. **Embedding Methods** (DMET, QBE): 分割分子，每個 fragment 執行 SQD/SKQD。QBE-SQD 用更小的 fragment 達到更好的精度。
2. **qDRIFT Compilation**: 隨機電路編譯降低深度，以更多 shots 為代價。
3. **GPU Post-processing**: 40-95× speedup 使更大 subspace 可行。
4. **Generative Pre-filtering**: ML（RBM, Transformer）預選配置，減少 subspace 50-70%。
5. **ph-AFQMC Post-processing**: 用 SQD trial wavefunction 做 AFQMC 恢復額外 correlation。
6. **ADAPT-GCIM**: Generator-coordinate-inspired subspace expansion with automated active space selection.

### 6.3 Chemically Decisive Benchmarks (Jan 2026)

**arXiv**: [2601.10813](https://arxiv.org/abs/2601.10813)

定義了從 proof-of-principle 到 quantum utility 的基準分子階梯：

| 系統 | 化學意義 | Active Space |
|------|---------|-------------|
| N2 | Multireference bond breaking | CAS(10,26) cc-pVDZ |
| FeS | High-spin transition-metal | CAS(6,6) ANO-RCC-MB |
| [2Fe-2S] | Bioinorganic iron-sulfur | CAS(30,20) |
| U2 | Actinide-actinide bonding | Large CAS |

Hamiltonians 公開可用。

### 6.4 Quantum Advantage Assessment (Aug 2025)

**arXiv**: [2508.20972](https://arxiv.org/abs/2508.20972) — MIT review "Quantum Advantage in Computational Chemistry?"

**結論**: Classical methods 在未來 ~20 年可能仍然優於量子演算法。但 QPE 可能在 10 年內超越 FCI for 數十/數百原子系統。

---

## 7. 競爭者完整表格 (March 2026 更新)

### 7.1 Generative Model + CI/Diag

| 方法 | 架構 | 規模 | 關鍵結果 | 與我們的差異 |
|------|-----|------|---------|------------|
| **QiankunNet** | AR Transformer + MCTS | CAS(46,26), 30 spin orbs 99.9% FCI | Fenton reaction Fe(II)/Fe(III) | 我們 Phase 4 的靈感來源 |
| **GTNN-SCI** | Generative Transformer | N2/H2O/C2 cc-pVDZ | 10× speedup over prior NN-SCI | Self-attention vs causal mask |
| **HAAR-SCI** | AR Transformer + Gumbel | Strongly correlated | Chemical accuracy, single GPU | H-guided selection vs Krylov |
| **NQS-SC** | NQS + selected configs | Molecular systems | Better than NQS-VMC | 驗證 NF+diag > VMC |
| **PIGen-SQD** | RBM + tensor decomp | IBM hardware scale | 70% subspace reduction | IBM 生態 |
| **AB-SND** | AR NN + basis optimization | Ising models | Non-concentrated ground states | Basis transformation idea |
| **GenKSR** | Transformer/Mamba | 15-20 qubits | Learns Krylov process | 學習 Krylov 而非顯式 time evolve |
| **Deterministic NQS** | Backflow + adaptive subspace + PT2 | H2O, N2, Cr2 | No MC noise, sub-linear scaling | 確定性替代 VMC |
| **ML-SCI** | Binary ML classifier | Molecular systems | Simple but effective | 類似我們的 NNCI |
| **NO-NNCI** | NNCI + natural orbitals | H2O, NH3, CO, C3H8 | More compact expansions | 可直接整合到我們的 NNCI |

### 7.2 Classical Baselines (2026 更新)

| 方法 | 最佳結果 | 時間/資源 |
|------|---------|----------|
| GPU-DMRG (SandboxAQ) | CAS(82,82) orbital-optimized | ~1 day / DGX-H100 |
| GPU-DMRG single-point | CAS(113,76) | DGX-H100 |
| Multi-GPU DMRG | CAS(114,73) D=14000 | 48× A100 |
| SHCI | Cr2 28e/76o, 2B determinants | CPU cluster |
| FCIQMC-GAS | 96e/159o (32,34) active ref | CPU cluster |
| NQS backflow | Competes with HCI/ASCI/FCIQMC/DMRG | Single GPU |

---

## 8. 對我們 Pipeline 的可行改進

### Priority 1: 高價值、可實施

| 改進 | 來源 | 影響 | 工作量 |
|------|------|------|--------|
| **Natural orbital rotation for NNCI** | NO-NNCI paper | 減少所需 determinants | 中 — PySCF 1-RDM + orbital rotation |
| **Hamiltonian coupling scoring** in `_rank_and_truncate_basis()` | HAAR-SCI | 更好的配置排序 | 低 — 已有 diagonal elements，加 off-diagonal coupling |
| **Rotation thresholding** for Krylov matrices | arXiv 2602.11985 | 改善 conditioning，減少 ill-conditioned fallbacks | 中 |
| **VMC default OFF** | NQS-SC paper | VMC 在分子系統不收斂是已知問題 | 低 — 改預設值 |

### Priority 2: 中期改進

| 改進 | 來源 | 影響 | 工作量 |
|------|------|------|--------|
| **Gumbel Top-K exploration** 替代 Plackett-Luce | HAAR-SCI | 更好的 exploration-exploitation balance | 中 |
| **Iterative refine-and-diag** (HAAR-SCI style) | HAAR-SCI | 比 single-pass 更好的收斂 | 高 |
| **IRL training** for AR flow | arXiv 2601.01437 | 更穩定的 NQS training | 高 |
| **GPU excitation generation** (Thrust-style) | arXiv 2601.16637 | 加速 `get_connections` | 高 |

### Priority 3: 研究方向

| 方向 | 來源 | 風險 |
|------|------|------|
| **GenKSR**: 學習 Krylov 過程的 generative model | arXiv 2512.19420 | 高 — 未在分子系統驗證 |
| **Embedding (DMET/QBE)** + SKQD | QBE-SQD, DMET-SQD papers | 中 — 需要 fragment 分割邏輯 |
| **Deterministic NQS** 替代 VMC | arXiv 2601.21310 | 中 — 需要重寫 training loop |
| **Block Krylov** (multi-reference) | QBKSP | 高 — 根本架構改變 |

---

## 9. 核心問題回答

### Q1: Is there a better subspace expansion strategy than Krylov time evolution?

**答案：有條件的 Yes。**

- **For classical simulation (our case)**: Krylov time evolution 仍然是 competitive 的。但 **HAAR-SCI 的 Hamiltonian-guided selection** 是更直接的替代方案 — 它用 H matrix elements 直接識別重要 determinants，避免 time evolution 的計算開銷。
- **Krylov 的獨特優勢**: Krylov expansion 不受 Reinholdt 的 "fatal flaw" 影響。Ground-state sampling（包括 NF）有邊際效益遞減，但 Krylov 是幾何展開，每步產生真正新的配置。
- **建議**: 保持 Krylov 作為核心策略，但加入 **Hamiltonian coupling scoring** 作為配置排序的輔助指標。考慮 **iterative expansion** (HAAR-SCI style) 作為未來改進。

### Q2: How do competitors handle 40-100 qubit molecular systems?

**答案：四種主要策略。**

1. **Embedding** (DMET, QBE): 最有效。QBE-SQD 用更小的 fragments 達到更好精度。
2. **GPU acceleration**: 40-95× 加速使更大 subspace 可行。
3. **Generative pre-filtering**: ML 模型（RBM, Transformer）預選配置，減少 70% subspace。
4. **在我們的規模 (40Q classical)**: Direct-CI + Krylov 足夠，因為瓶頸是 diagonalization cost 而非 sampling。

### Q3: What is the current state-of-the-art for N2/cc-pVDZ benchmarks?

**答案：**

- **IBM standard**: CAS(10,26) 52Q via SqDRIFT on quantum hardware
- **Classical SCI**: GTNN-SCI 比先前 NN-SCI 方法更快收斂、更低能量
- **Classical DMRG**: CAS(10,26) 對 GPU-DMRG 是 trivial
- **HAAR-SCI**: 達到 chemical accuracy
- **Our pipeline**: CAS(10,20) 40Q tested, 14.2 mHa gap at CAS(10,10)。需要 CAS(10,26) 測試來直接比較。

---

## 10. References

### SQD / SqDRIFT
- [SqDRIFT paper](https://arxiv.org/abs/2508.02578)
- [Half-Mobius — IBM Research Blog](https://research.ibm.com/blog/half-mobius-molecule)
- [IBM Press Release](https://newsroom.ibm.com/2026-03-05-ibm-and-university-researchers-create-a-never-before-seen-molecule-and-prove-its-exotic-nature-with-quantum-computing)
- [GPU SQD — OpenMP Offload](https://arxiv.org/abs/2601.16169)
- [GPU SQD — Thrust SBD](https://arxiv.org/abs/2601.16637)
- [PIGen-SQD](https://arxiv.org/abs/2512.06858)
- [Implicit Solvent SQD](https://arxiv.org/abs/2502.10189)
- [Band Gap SQD](https://arxiv.org/abs/2503.10901)
- [ph-AFQMC + SQD](https://arxiv.org/abs/2503.05967)
- [Symmetry-Adapted SQD](https://arxiv.org/abs/2505.00914)
- [QBE-SQD](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-lwj7d/v2)
- [Cuprate Chain SQD](https://arxiv.org/abs/2512.04962)
- [Lockheed+IBM SQD](https://www.ibm.com/quantum/blog/lockheed-martin-sqd)
- [Critical Limitations QSCI](https://arxiv.org/abs/2501.07231)

### Krylov Methods
- [SKQD 85 qubits](https://arxiv.org/abs/2501.09702)
- [Rotation Thresholding](https://arxiv.org/abs/2602.11985)
- [GenKSR](https://arxiv.org/abs/2512.19420)
- [QBKSP](https://journals.aps.org/pra/abstract/10.1103/p8wj-gplb)

### Selected CI + Neural Network
- [GTNN-SCI](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01429)
- [HAAR-SCI](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415)
- [ML-Assisted SCI](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01652)
- [NQS-SC](https://arxiv.org/abs/2602.12993)
- [NO-NNCI](https://arxiv.org/abs/2510.27665)
- [Deterministic NQS](https://arxiv.org/abs/2601.21310)
- [IRL for NQS](https://arxiv.org/abs/2601.01437)
- [Efficient NQS Backflow](https://arxiv.org/abs/2502.18843)
- [QiankunNet](https://www.nature.com/articles/s41467-025-63219-2)
- [AB-SND](https://arxiv.org/abs/2508.12724)

### GPU-DMRG
- [SandboxAQ Quarter petaFLOPS](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00903)
- [DMRG-SCF CAS(82,82)](https://arxiv.org/abs/2503.20700)
- [Multi-GPU DMRG Nitrogenase](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01228)

### Benchmarks & Reviews
- [Chemically Decisive Benchmarks](https://arxiv.org/abs/2601.10813)
- [Quantum Advantage in Computational Chemistry?](https://arxiv.org/abs/2508.20972)
- [Hybrid Tensor Network + NQS](https://arxiv.org/abs/2507.19276)
