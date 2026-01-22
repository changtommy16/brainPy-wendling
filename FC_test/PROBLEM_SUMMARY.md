# Wendling Model FC Correlation 問題總結

## 問題描述
為什麼我們的 brainPy Wendling model (`demo_hcp_optimize.py`) 只能達到 FC correlation ~0.14，而 Neurolib 的 `optimize_fc_correlation.py` 可以達到 ~0.6？

## Update (repo changes)
- `src/wendling_sim/model/wendling_network.py` now injects `u_net` into `dy5` (y0 path) to match Neurolib.
- `FC_test/brainpy_fc_sweep.py` now has a `neurolib_like_psp_no_norm` case and default config uses it.
- Added a step-by-step notebook at `FC_test/FC_debug_steps.ipynb`.
- `src/wendling_sim/sim/runner.py` uses BrainPy's explicit Euler integrator when `integrator='euler'`.

## 關鍵發現

### 1. 連接注入位置差異 (主要問題)
- **Neurolib (正規)**: 注入到 `y5` (y0 的導數，興奮性內部神經元 PSP)
  ```python
  dy5 = A * a * (sigmoid(y1 - y2 - y3) + coupling_input) - 2*a*y5 - a^2*y0
  ```
- **我們的 brainPy (非正規)**: 注入到 `y6` (興奮性→錐體 PSP 的導數)
  ```python
  dy6 = A * a * (C2*sigmoid(C1*y0) + p_t + u_net) - 2*a*y6 - a^2*y1
  ```

### 2. 連接矩陣處理差異
- **Neurolib**: 使用原始結構連接矩陣 `Cmat`
- **我們的 brainPy**: 使用歸一化矩陣 `W` (row_sum normalization)

### 3. 參數尺度差異
- **Neurolib**: `K_gl` ~ 0.05 就能達到高 correlation
- **我們的 brainPy**: 需要 `G_net` ~ 2.5-3.0 才有類似效果

### 4. 其他差異
- **數據**: Neurolib 使用 HCP 多受試者平均，我們使用單一受試者
- **模擬長度**: Neurolib 10 秒，我們 2 秒
 - **節點數**: Neurolib HCP dataset 為 80 節點；我們的 HCP 102816 為 360 節點
 - **delay**: Neurolib 有 distance-based delay (`Dmat`)，BrainPy 目前沒有

## 文獻驗證
根據 David & Friston (2003) 和神經質量模型文獻，**Neurolib 的做法是正規的**：
- 網絡耦合應該注入到興奮性內部神經元路徑 (y0/y5)
- 這符合生理學和解剖學的連接方式

## 嘗試的解決方案

### 1. 修改連接注入位置 ✅
```python
# 從 y6 改為 y5
dy5 = self.A * self.a * (self.sigmoid(v_pyr) + u_net) - 2.0 * self.a * y5 - self.a**2 * y0
dy6 = self.A * self.a * (self.C2 * self.sigmoid(self.C1 * y0) + p_t + u_stim) - 2.0 * self.a * y6 - self.a**2 * y1
```

### 2. 修改連接矩陣處理 ✅
```python
# 使用原始矩陣，不歸一化
W = _normalize(W.astype(np.float32), cfg.get('normalize', 'none'))
```

### 3. 調整參數範圍 ✅
```python
# 改為更小的 G_net 值，配合 Neurolib 的尺度
"G_net_values": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
```

## 測試結果
修改後測試 (`brainpy_fc_sweep.py`):
- **baseline_psp**: fc_corr=0.2086 (G_net=0.100, het=0.20)
- **neurolib_like_psp**: fc_corr=0.1102 (G_net=3.000, het=0.00)

使用 Neurolib dataset (80 nodes, mean FC) 與 BrainPy：
- **neurolib_like_psp**: fc_corr=0.5723 (G_net=0.100, het=0.00)
- **neurolib_like_bold**: fc_corr=0.5816 (G_net=0.100, het=0.00)
- **fine sweep (BOLD)**: fc_corr=0.6035 (G_net=0.100, seed=3)

**可達到 ~0.6，但前提是使用 Neurolib dataset 與設定**

## 剩餘問題
即使修改了連接注入位置和矩陣處理，我們的 brainPy 實作仍然無法達到 Neurolib 的 0.6 correlation。可能的原因：

1. **數值積分器差異**: Neurolib 使用不同的積分方法
2. **參數初始化差異**: 兩者的標準參數可能有細微差別
3. **sigmoid 函數實作差異**: 可能有不同的數值穩定性
4. **時間步長差異**: dt 的選擇影響數值精度
5. **邊界條件處理**: 初始條件和 burn-in 處理方式不同

## 給下一個人的建議

### 立即可嘗試的方向：
1. **詳細比較數值積分器**: 檢查 Neurolib 的 `timeIntegration.py` 與我們的 Euler 積分器
2. **參數逐一對比**: 確保所有參數 (A, B, G, a, b, g, e0, v0, r) 完全一致
3. **sigmoid 函數驗證**: 確保兩者的 sigmoid 實作完全相同
4. **時間步長測試**: 嘗試更小的 dt (0.01ms) 提高數值精度
5. **模擬長度測試**: 延長到 10 秒匹配 Neurolib
6. **資料集差異**: 360 nodes 的單一受試者 vs 80 nodes 的多受試者平均，難度完全不同

### 深入分析方向：
1. **單節點模型驗證**: 先確保單節點 Wendling model 的動態完全一致
2. **網絡耦合驗證**: 簡化為 2 節點網絡，逐步驗證耦合實作
3. **頻域分析**: 比較 PSD 和頻譜特徵，不僅看 FC correlation
4. **數值穩定性分析**: 檢查是否有數值爆炸或不穩定問題

## 關鍵文件位置
- **Neurolib 實作**: `Neurolib_package/neurolib/models/wendling/timeIntegration.py`
- **我們的實作**: `src/wendling_sim/model/wendling_network.py`
- **測試腳本**: `FC_test/brainpy_fc_sweep.py`
- **目標腳本**: `optimize_fc_correlation.py`

## 成功標準
當修改後的 brainPy 實作能達到：
- FC correlation ≥ 0.5
- 使用合理的 G_net 參數範圍 (0.01-1.0)
- 與 Neurolib 的結果可重複

這個問題需要深入的數值分析和逐步驗證，建議從單節點模型開始，確保每個組件都正確後再處理網絡耦合。
