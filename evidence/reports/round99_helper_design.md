# R99 四輸入 L1 組合：數值 helper 與人工驗證

日期：2026-09-29（Asia/Taipei）。本文件只記錄 helper 實作與合成檢查；尚未執行任何真實 meta-fit 或 query 評分。固定研究問題、時間隔離與判讀門檻見 [R99 計畫](round99_plan.md)。本 helper 作者參與方法實作，不能把本文件稱為完全獨立的第三方驗證。

## 唯一模型改動與數學範圍

[新 helper](../src/round99_l1_stacker.py) 複用 [R98 helper](../src/round98_l1_stacker.py) 的同一等列權、非負且權重和為 1、無截距 L1 問題，輸入從 `[N,H,U]` 擴為 `[N,H,U,M]`，其中 `M=np.median(P[:,:3],axis=1)`。四欄是已完成原 reference/spatial 的未取整總價，不重新訓練或改動原生模型。

對四欄價格矩陣 P 與標籤 y，價格共同除以 1e6 後解等價 dual：最小化 `-y_scaled @ u - v`，限制 `P_scaled.T @ u + v <= 0`、`-1 <= u <= 1`，v 不設上下限。只有四個不等式；四權重直接取負的原始 inequality marginals。v 的 stationarity 對應權重和為 1；不另作 clip、normalize、第二目標、正則或失敗後重試。

M 權重為 1 是可行解，原三欄權重加第四欄 0 也是可行解。因此同一 meta 樣本、同一等列權 MAE 的精確最優值不應差於原 median 或原三欄可行解。這個保證不適用於 query、固定群加權 MAE、Private，亦不代表新方法必然有不同預測。helper 檢查四個 vertex 與 M；與原 R98 同 meta 保存 primal 的比較由 driver 執行，helper 不讀舊真實模型或標籤。

## 固定求解與數值證書

沿原 SciPy 1.18.1／HiGHS 1.12.0、`highs-ds`、presolve、steepest-devex、單執行緒、300 秒及 1,000,000 次迭代上限、primal/dual feasibility `1e-8`。不因解出多個最優權重而挑另一組；接受固定 runtime 與列序所得解。SciPy 對 `threads` 的轉交警告保存在每次求解 metadata，不隱藏為不存在。

固定證書沿 R98：原權重嚴格非負、和偏差最多 `1e-10`；dual box `1e-8`；dual constraint 平均 0.01 元；price stationarity 最大 0.10 元及 free variable `1e-10`；objective identity 平均 0.01 元；primal-dual gap、互補性與 vertex optimality 平均各 0.10 元。任一超標即失敗，不修補權重。

原 `certificate.native_vertex_MAE_TWD` 現含 N/H/U/M 四鍵，另有 `median_vertex_MAE_TWD`、`primal_minus_median_vertex_MAE_TWD` 與 `checks.median_vertex_optimality`。保留 primal、dual、gap、約束 slack、stationarity、互補性與原始 solver vectors。證書是浮點數值驗證，不是對未來交易的保證。

## API 與資料契約

```python
state, solver_arrays, metadata = fit_stacker(
    P4, y,
    prediction_row_ids=ids,
    target_row_ids=ids.copy(),
)
prediction = predict(state, Q4)
save_artifacts(new_output_directory, state, solver_arrays, metadata)
restored_state = load_state(new_output_directory / "state.json")
```

P/Q 必須是非空、有限、非負 `float64 n×4`，y 是同列數有限非負 float64。第四欄須逐 bytes 等於前三欄的 NumPy median，錯誤一 ULP 也拒絕。若提供列身分，兩份都必須是唯一、非負、同列序的 int64；未提供時由 caller 的來源 manifest 負責對齐。driver 會提供身分，不憑 helper 假設對齊。

純數值無法辨認 N/H/U 三欄的語義互換，因為那不會改變 median；這部分必須由 driver 對原 branch 檔案與欄位次序的 SHA 綁定。helper 可檢查 M 的位置與數值、模型 `branch_order`、shape、dtype 和列身分，不誇大成能偵測所有欄位錯置。

state `format` 固定為 `R99_convex_L1_stacker_v1`、`branch_order=['N','H','U','M']`。保留原 `parameters`、`weights`、`input_arrays`、`runtime`、`source_sha256`、`fit_rows`、`solver_status`、`certificate_passed` 與 `query_dependent_state=false` schema。`input_arrays.P` 的雜湊是實際四欄矩陣；不是原三欄的雜湊。`parameters()` 與 `source_hashes()` 可供 launcher/scorer 固定身分，後者包含自身及四份 SciPy/HiGHS wrapper/binary 檔。

完整推論用固定 float64 順序 `((P[:,0]*w[0]+P[:,1]*w[1])+P[:,2]*w[2])+P[:,3]*w[3]`，不使用 BLAS 矩陣乘法改變求和順序。query API 沒有 y，不重新 fit 或改 fitted state。保存目錄必須不存在，輸出 `state.json`、`fit_metadata.json`、`solver_arrays.npz`、`receipt.json`；真實完整 query 重載相同由 driver 另驗。`StackerFitError` 保留 metadata 與可用 solver arrays，caller 必須另存失敗證據而不當作成功模型。

## 人工檢查與成本界線

[canonical 收據](round99_helper_selfcheck.json) 與 [attempt1 收據](round99_helper_selfcheck_attempt1.json) bytes 完全相同；首次執行 exit 0，53 項檢查、6 次小型人工 LP，共量測 0.172 秒。另有兩個 mocked solver failure，不計為真 LP。沒有真實陣列或價格、真 meta-fit、query 評分、GPU 操作。首次來源已另存 [原碼快照](../outputs/round99_helper_prefreeze/source_attempt1.py) 與 [身分紀錄](../outputs/round99_helper_prefreeze/source_attempt1.json)；沒有失敗版本或求解重試。

六個人工問題包括唯一四權重內點、單支完美的邊界解、離群值下 L1 與平方損失的區別、同欄多解、原 R98 median 反例、共同增加十億元後仍可還原的解。反例原前三列為 `(0,0,3)`、`(0,3,0)`、`(3,0,0)` 百萬元，y 全為 0；任一三欄 simplex 的平均誤差都是一百萬元，新增 M=0 後確實得到 M 權重 1、零誤差。這不是對真實 query 的實驗結果。

其他檢查涵蓋 caller 輸入不變、錯列身分／重複身分／錯 dtype／非有限／負價格／舊三欄／額外第五欄拒絕、M 損壞與錯位拒絕、權重不合法不修補、全人工 query JSON 重載逐位一致、NPZ solver evidence 重載及拒覆寫。timeout 與假成功但權重和不合法的 mock 都保留失敗 metadata，確認只有一次求解呼叫。

最大規劃 meta 186,745 列時，dual 為 186,746 變數、4 個不等式、746,984 個儲存係數。CSC data/indices/indptr payload 為 9,710,796 bytes，P+y 為 7,469,800 bytes；這只是可計算的陣列大小，不含求解器 workspace、複本、Python 與證書記憶體，沒有量測真資料峰值 RAM 或估算其求解時間。人工問題的單次 solver timer 有 0 秒值，反映短於時鐘解析度，不能解釋為無計算成本。

## 凍結身分與後續界線

- plan SHA256：`8918ba8a4a0b50378aca56649131678ce069350902c044a6678384b4e95d75c9`。
- 新 helper SHA256：`f07d818aa9b12cf03726267188e99d111ce2529dd6d6cd5f9d921fee259fc09e`。
- canonical selfcheck SHA256：`59e4d31ed617badc152d597b15ef2e5941d16912799d470d27a0fd2b6b0d9644`。
- 原 R98 helper SHA256 保持：`a1a501bc2c7874fc0d21983ad5a2be63e8d1c085f21d24a096220ff32dbfc864`。

真 meta-fit 僅由主執行緒在完整 driver/scorer 前置檢查與來源凍結後啟動；此文件不代表已通過真資料門檻。R99 是觀看 R98 結果後提出的適應性研究，沿用窗口也不是新盲測。沒有更改 v22 唯一提交、CSV、正式報告或簡報。
