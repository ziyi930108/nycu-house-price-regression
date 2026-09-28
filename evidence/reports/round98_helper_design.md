# R98：三分支 L1 權重求解器的數學與數值契約

日期：2026-09-29。依[固定研究計畫](round98_plan.md)，本文件只記錄求解器實作與人工資料驗證；沒有讀取真實預測或價格、沒有真實 meta-fit，也沒有 GPU 操作。程式是 [round98_l1_stacker.py](../src/round98_l1_stacker.py)。

## 只改最後的全域組合

P 的三欄固定依序為已完成 reference／spatial 修正的 N、H、U，單位 TWD；目標為每個 meta-fit 交易列等權的 MAE。解 `min_w Σ|P_i w−y_i|`，限制 `w>=0`、`sum(w)=1`，無截距、正則化、分群、時間權重或比例搜尋。將 P 與 y 同除以 1e6 只是求解尺度，並不改變原 MAE 最佳權重。

原 absolute-residual epigraph 需要 n+3 個變數、2n 個不等式。這裡改解其等價對偶：

```text
minimize    −y_scaledᵀ u − v
subject to  P_scaledᵀ u + v·1 <= 0       (exactly three inequalities)
            −1 <= u_i <= 1
            v is free
```

這個 LP 只有 n+1 個變數、3 個不等式。三個不等式的非負 Lagrange 乘數就是原問題權重；SciPy 回傳的 `ineqlin.marginals` 使用相反號，因此 **w = −marginals**。自由變數 v 的 stationarity 要求三權重和為 1。由 `|u_i|<=1` 與三個不等式，可以推出 `y_scaledᵀu+v <= Σ|P_scaled w−y_scaled|`；完整 KKT 與兩端差距提供數值最優證據。

SciPy 官方說明確認 HiGHS dual simplex 的回傳狀態、marginals 符號與可設定的時間／可行性容差；本機實際使用 **SciPy 1.18.1、HiGHS 1.12.0**，版本不同直接拒絕 fit。[官方 linprog highs-ds 文件](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html)；實際本機 `_linprog.py`、`_linprog_highs.py`、`_highs_wrapper.py` 與 compiled core SHA 另存於來源映射。

## 事前固定數值規約

`helper.parameters()` 是 root launch request 的精確工程契約。固定 `highs-ds`、presolve=True、steepest-devex、threads=1、每次 solver 300 秒上限、maxiter=1,000,000，primal／dual feasibility tolerance 都是 1e-8。沒有第二目標、重試、求解器輪替或根據 query 表現挑一組等價權重；多解時接受固定 solver 版本及原輸入列序回傳的那一組。

SciPy 的公開 wrapper 會發出將 `threads=1` 原樣轉交 HiGHS 的 OptimizeWarning；本機 core 的確提供該選項，wrapper 會驗證並傳入。每次 warning 的全文與實際版本保留於 metadata，沒有隱藏警告或聲稱量測到實際 OS 執行緒數。

直接保留原 solver 權重，**不 clip、不 normalize**。只有 status=0 且全部證書檢查通過才可產生模型：

|檢查|固定容許誤差|
|---|---:|
|最小權重|嚴格 >=0|
|權重和距離 1|<=1e-10|
|dual u 的 box violation|<=1e-8|
|dual constraint violation，除 n 並還原平均 TWD|<=0.01 元|
|u stationarity 最大值，還原 TWD|<=0.10 元|
|v stationarity|<=1e-10|
|solver objective 與重算 dual objective，每列 TWD|<=0.01 元|
|直接原 TWD 預測 MAE − dual bound，每列 TWD 絕對值|<=0.10 元|
|不等式與 bounds 的互補殘差，各自每列 TWD|<=0.10 元|
|相對三個可行 vertex 中最低 MAE 的退步|<=0.10 元|

另外檢查 marginal 符號與自由變數 bound multipliers 為零。這是預先固定容差內的數值最優證據，不是對泛化能力或 Private 分數的保證。證書保存實際 weights、primal／dual objective、gap、所有殘差及各檢查結果，不只存 solver 的 success 布林值。

## 記憶體與成本

以目前最大 meta-fit 的 186,745 列為例：186,746 個變數、3 列不等式、560,238 個非零。直接建 CSC，float64 data 加 int32 indices／indptr 的 payload 精確為 **7,469,844 bytes（約 7.12 MiB）**；原 P+y 四個 float64 通道為 **5,975,840 bytes（約 5.70 MiB）**。不建立 n×n dense identity。

以上是陣列大小計算，**不是實測峰值記憶體**。HiGHS 工作空間、複本、證書與 Python 開銷另計；沒有用小型人工 fit 外推真實資料秒數。每次 300 秒是 solver 限額，不包含資料建構／保存與啟動耗時。若達上限或數值證書不符，保留失敗，不退回任意權重或放寬容差。

## API 與保存

```python
state, solver_arrays, metadata = fit_stacker(
    P, y, prediction_row_ids=ids, target_row_ids=ids.copy())
pred = predict(state, Q)
save_artifacts(new_model_directory, state, solver_arrays, metadata)
restored = load_state(new_model_directory / "state.json")
restored_pred = predict(restored, Q)
```

P 為 float64、n×3，y 為 float64、n；皆有限且非負。禁止暗中轉 float32、截尾或去除資料列。兩個選填 row-ID 向量必須同時提供、int64、唯一且逐位一致；可接受按合法 forward blocks 串接而未全域排序的索引。未提供 row IDs 时，metadata 明說列序需由 caller 的 manifest 保證，不能由形狀相同推論正確對齊。真正 cutoff、meta/query 排除與早期預測來源由 driver 的封存契約另外驗證，純數值 helper 不猜日期。

predict 不接受價格或更新 state，嚴格按 float64 `(P[:,0]*w[0] + P[:,1]*w[1]) + P[:,2]*w[2]` 計算，沒有 BLAS matrix multiplication、截距、rounding、clip。driver 對自己的完整 query 驗證保存／重載逐位一致。

新 model 目錄含 state.json、fit_metadata.json、solver_arrays.npz、receipt.json。保存 solver x、三個 inequality 的 residual／marginals，以及所有 lower／upper residual／marginals；自由變數 bound residual 允許相應 infinity，NPZ 不使用 pickle。state 的 JSON 權重完整往返，來源、參數、輸入數值 SHA 與 solver evidence 各自可追溯。

數值失敗會丟出 `StackerFitError`，攜帶 `metadata` 與有取得的 `solver_arrays`，由 root driver 保存失敗現場並非零結束。一般格式錯誤直接拒絕。helper 沒有真資料 loader，沒有模型主入口，CLI 只有合成 self-check。

## 人工驗證與反證

[attempt2 收據](round98_helper_selfcheck_attempt2.json)：**42 項通過、6 個人工 LP、總計 0.172 秒**。包含唯一內點權重 [0.2,0.3,0.5]、已知 vertex、極端離群值下 L1 與平方誤差答案不同、重複分支多解、共同加上 10 億元價格位移，以及完整人工 query 保存重載 bitwise 一致。另用 mock solver 驗證 timeout 不回傳模型也不重試、status=0／success=True 但壞 marginals 仍被數值證書拒絕且不 normalize。mock 不計入六次真正合成 LP。

刻意保留重要反例：三筆真值都是 0、分支矩陣為 `[[0,0,3],[0,3,0],[3,0,0]]×1e6`。逐筆 median 的 MAE 為 0；所有非負且和為 1 的全域權重，MAE 都是 100 萬元。**MAE 最優固定組合不保證勝過原本的逐筆 median，甚至在 meta-fit 上也不保證。** 這正是後續必須完整測三窗、保留負面結果的原因。

[attempt1](round98_helper_selfcheck_attempt1.json)原有 38 項亦通過，原 source bytes 保存於 [prefreeze source](../outputs/round98_helper_prefreeze/source_attempt1.py)與身分紀錄。第二版僅增加 timeout 回傳 iteration 欄位 None 的安全處理與兩個失敗分支測例；未改科學模型、數值容差或 solver。沒有真實 fit 或失敗被藏起來。獨立審查無阻擋後，[final canonical 收據](round98_helper_selfcheck.json)已以 attempt2 相同 bytes 保存，供 root 凍結。helper source SHA256 `a1a501bc2c7874fc0d21983ad5a2be63e8d1c085f21d24a096220ff32dbfc864`；canonical 收據 SHA256 `7261604d4d671960ecba5fc6ed77a5b3d06419ee5f9b43e8f5f8a02a2556549e`。
