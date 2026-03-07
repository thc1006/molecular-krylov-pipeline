# 深度優化調研報告

> 調研日期：2026-03-06
> 硬體：DGX Spark GB10 (20 ARM cores, 128GB UMA, CUDA 13.0, sm_121)
> 調研範圍：5 個平行研究代理，涵蓋 30+ 項技術

---

## 關鍵發現摘要

### DGX Spark 硬體限制（影響全局策略）

| 精度 | TFLOPS | 備註 |
|------|--------|------|
| FP64 | **0.48** | FP32:FP64 = 64:1，NVIDIA 官方表示「FP64 不是 Spark 的目標用途」 |
| FP32 | ~30 | CUDA cores |
| TF32 | ~53 | Tensor cores，已在 codebase 啟用 |
| FP16/BF16 | ~213 | Tensor cores |

**結論：FP64 在 DGX Spark 上基本不可用（0.48 TFLOPS）。整個 pipeline 策略必須轉向 FP32/TF32 為主、FP64 僅做最終 refinement。**

### UMA 零拷貝

- DGX Spark CPU/GPU 共享 128GB LPDDR5x，無 PCIe 瓶頸
- PyTorch 預設 `cudaMalloc` 不是 managed memory，需自定義 allocator
- **已知問題**：pageable memory 的小批量 H2D copy 有 50x 減速（NVIDIA 論壇已確認）
- 對策：使用 `pin_memory=True`、批次傳輸、避免大量小 tensor copy

---

## 按瓶頸分類的優化方案

### 瓶頸 1：Hamiltonian 矩陣元素計算（`get_connections` 4 層 Python 迴圈）

| 優先度 | 方案 | 預期加速 | 難度 | 來源 |
|--------|------|---------|------|------|
| **P0** | Numba JIT `@njit` 編譯 `get_connections` 內迴圈 | 50-200x | 中 | Scemama 2013, PyFock |
| **P0** | 確保 pipeline 永遠使用 `get_connections_vectorized_batch()` 而非逐一 `get_connections()` | 10-100x | 低 | 已有程式碼 |
| P1 | PySCF `selected_ci.kernel_fixed_space` 作為替代 diag 後端（IBM SQD addon 的做法） | 免除 H 建構 | 低 | qiskit-addon-sqd |
| P1 | Scemama 演算法：XOR + popcount 判斷激發度（< 10 CPU cycles/pair） | 6x（常數因子） | 中 | arXiv:1311.6244 |
| P2 | Cython/C 擴展模組（Quantum Package 的做法） | 100-500x | 高 | Quantum Package 2.0 |

**Numba JIT 核心概念：**
```
# 虛擬碼 — 用 XOR + popcount 取代 4 層 Python 迴圈
excitation_degree = popcount(det_i XOR det_j) // 2
if excitation_degree == 1:  # 單激發
    apply_slater_condon_single(...)
elif excitation_degree == 2:  # 雙激發
    apply_slater_condon_double(...)
# 跳過 degree > 2（Slater-Condon 規則保證 H_ij = 0）
```

**Numba ARM64 狀態：** CPU JIT 可正常運作。CUDA JIT 對 sm_121 不確定，需先測試。

---

### 瓶頸 2：Eigensolver（dense eigh / sparse eigsh）

| 優先度 | 方案 | 預期加速 | 難度 | 來源 |
|--------|------|---------|------|------|
| **P0** | cuSOLVER Blackwell math mode（TF32 精度做 eigh，53 vs 0.48 TFLOPS） | ~100x | 中 | CUDA 13.0 docs |
| **P0** | SQD 批次平行化：`ProcessPoolExecutor` + `OPENBLAS_NUM_THREADS=1` | 8-10x | 低 | IBM SQD 設計 |
| P1 | Shift-invert eigsh（`sigma=E_hf`）加速收斂 | 2-5x | 低 | SciPy ARPACK |
| P1 | PyTorch 批次 `torch.linalg.eigh`：堆疊多個小矩陣一次 GPU 呼叫 | 5-15x | 中 | PyTorch docs |
| P2 | CuPy GPU sparse eigsh（>15K 矩陣才有優勢） | 10-20x@50K+ | 中高 | CuPy v14 |
| P2 | NVPL BLAS/LAPACK 取代 OpenBLAS（ARM64 優化） | 2-5x CPU LA | 低 | NVIDIA NVPL docs |

**SQD 批次平行化核心概念：**
```python
# 目前：序列執行 10 個 batch
for batch in batches:
    result = diag_batch(batch)  # 單核

# 優化後：20 核心平行
with ProcessPoolExecutor(max_workers=10) as pool:
    results = list(pool.map(diag_batch, batches))  # 10 核心同時
```

**混合精度策略（針對 FP64 0.48 TFLOPS 限制）：**
1. Hamiltonian 建構：FP32/TF32（53 TFLOPS）
2. Eigensolver：FP32 + cuSOLVER Blackwell math mode
3. 最終能量：FP64 refinement（少量計算）

---

### 瓶頸 3：O(n²) Hamming 距離多樣性選擇

| 優先度 | 方案 | 複雜度改善 | 難度 | 來源 |
|--------|------|-----------|------|------|
| **P0** | Bit-parallel Hamming：uint64 pack + `torch.bitwise_count` | O(n²·sites) → O(n²) | 低 | SimSIMD, FAISS |
| **P0** | Stochastic Greedy（Mirzasoleiman 2015） | O(nk) → O(n·log(1/ε)) | 低 | AAAI 2015, apricot lib |
| P1 | Fast Greedy DPP（Cholesky 增量更新） | 免除完整距離矩陣 | 中 | NeurIPS 2018 |
| P1 | Streaming Diversity（Ceccarello） | O(nk) 單次掃描 | 中 | VLDB 2017 |
| P2 | LSH for Hamming（bit sampling） | O(n^ρ), ρ<1 | 低中 | FAISS IndexBinaryHash |

**組合效果（n=50K, sites=40, k=5K）：**
- 目前：O(n²·sites) = 50K² × 40 = 100B ops，需 100GB+ 記憶體 → OOM
- Bit-parallel + Stochastic Greedy：O(n·log(1/ε)) × O(1)/pair = ~230K ops → **~40,000x 加速**

---

### 瓶頸 4：NF-NQS 訓練（目前強制停用）

| 優先度 | 方案 | 關鍵創新 | 難度 | 來源 |
|--------|------|---------|------|------|
| **P0** | NNCI（Neural Network Configuration Interaction）主動學習 | N₂: 10¹⁰ → 4×10⁵ 行列式（10⁵x 縮減） | 中 | arXiv:2406.08154 |
| P1 | PIGen-SQD 攝動剪枝：MP2/CISD 重要性評分 | 子空間縮減 70% | 中 | arXiv:2512.06858 |
| P1 | Sigmoid top-k 取代 GumbelTopK | O(nk) → O(n)，連續 k | 低 | EurIPS 2025 |
| P2 | SFESS 無偏 k-subset 梯度 | 消除 straight-through 偏差 | 中 | ICLR 2025 |
| P2 | Autoregressive model（Transformer）取代 NF | 精確取樣、無 mode collapse | 高 | Nature MI 2022 |

**NNCI 主動學習迴圈：**
1. 從 HF + singles + doubles 開始
2. 訓練 NN 分類器預測「哪些行列式有大 CI 係數」
3. 用分類器篩選 NF 或系統枚舉的候選配置
4. 對角化、提取 CI 係數、重新訓練分類器
5. 迭代直到收斂

---

## 綜合優先排序（全 pipeline）

### Tier 1：立即可做（1-2 天，大幅加速）

| # | 項目 | 目標 | 預期效果 |
|---|------|------|---------|
| 1 | SQD 批次平行化（ProcessPoolExecutor） | `sqd.py` batch diag | 8-10x |
| 2 | SKQD shift-invert eigsh | `skqd.py` eigensolver | 2-5x |
| 3 | 確保 `get_sparse_matrix_elements` 用向量化路徑 | `molecular.py` | 已完成 ✅ |
| 4 | Bit-parallel Hamming（bitwise_count） | `diversity_selection.py` | 40x 常數因子 |
| 5 | Stochastic Greedy 取代 O(n²) 廣播 | `diversity_selection.py` | 1000x |

### Tier 2：中期改善（1-2 週）

| # | 項目 | 目標 | 預期效果 |
|---|------|------|---------|
| 6 | Numba JIT `get_connections` | `molecular.py:504-569` | 50-200x |
| 7 | NVPL BLAS/LAPACK 取代 OpenBLAS | 全域 CPU LA | 2-5x |
| 8 | 混合精度策略（TF32 建構 + FP64 refinement） | eigensolver | ~100x diag |
| 9 | PyTorch 批次 eigh（SQD 小矩陣） | `sqd.py` | 5-15x |

### Tier 3：長期架構升級（1-2 月）

| # | 項目 | 目標 | 預期效果 |
|---|------|------|---------|
| 10 | NNCI 主動學習（NN 分類器篩選配置） | 配置選擇 | 10⁵x 子空間縮減 |
| 11 | PIGen-SQD 式攝動剪枝 | 配置選擇 | 70% 子空間縮減 |
| 12 | PySCF selected_ci 後端（IBM 做法） | 免除自建 H | 架構簡化 |
| 13 | Sigmoid top-k / SFESS 取代 GumbelTopK | NF 訓練 | 解決 mode collapse |
| 14 | CuPy GPU sparse eigsh（40Q+ 系統） | eigensolver | 10-20x@50K+ |

---

## 參考文獻

### Hamiltonian 建構
- [Scemama & Giner 2013 — Slater-Condon with popcount](https://arxiv.org/abs/1311.6244)
- [Quantum Package 2.0](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00176)
- [Dice SHCI](https://sanshar.github.io/Dice/)
- [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd)
- [PyFock — Numba for quantum chemistry](https://pyfock.bragitoff.com/)

### Eigensolver / 平行化
- [IBM SQD 概述](https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/sqd-overview)
- [qiskit-addon-sqd-hpc (C++/MPI)](https://github.com/Qiskit/qiskit-addon-sqd-hpc)
- [SciPy eigsh shift-invert](https://docs.scipy.org/doc/scipy/tutorial/arpack.html)
- [CuPy sparse eigsh](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html)
- [PyTorch batched eigh](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html)

### 多樣性選擇
- [Lazier Than Lazy Greedy (AAAI 2015)](https://arxiv.org/abs/1409.7938)
- [Fast Greedy DPP (NeurIPS 2018)](https://arxiv.org/abs/1709.05135)
- [FAISS Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)
- [SimSIMD — ARM NEON Hamming](https://github.com/ashvardanian/SimSIMD)
- [apricot — submodular selection](https://apricot-select.readthedocs.io/)

### NF-NQS / 生成模型
- [PIGen-SQD (arXiv:2512.06858)](https://arxiv.org/abs/2512.06858)
- [NNCI (arXiv:2406.08154)](https://arxiv.org/abs/2406.08154)
- [Sigmoid top-k (EurIPS 2025)](https://openreview.net/forum?id=VMlajIH1oF)
- [SFESS (ICLR 2025)](https://openreview.net/forum?id=q87GUkdQBm)
- [MinSR (Nature Physics 2024)](https://www.nature.com/articles/s41567-024-02566-1)

### DGX Spark 硬體
- [DGX Spark FP64 論壇](https://forums.developer.nvidia.com/t/dgx-spark-fp64-performance/346607)
- [DGX Spark 效能指標](https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993)
- [NVPL BLAS/LAPACK](https://docs.nvidia.com/nvpl/latest/index.html)
- [cuSOLVER Blackwell math mode](https://docs.nvidia.com/cuda/cusolver/index.html)
- [Pageable memory 50x 減速](https://forums.developer.nvidia.com/t/dgx-spark-arm64-cuda-13-pathological-slowdown-for-many-small-h2d-copies-from-pageable-cpu-memory-50x-vs-pinned-impacts-pytorch-model-load-patt/354506)
