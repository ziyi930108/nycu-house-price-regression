# 錯開時間切點：R97 前置快照，研究進行中

2026-09-29（Asia/Taipei）。**目前只完成資料與標籤前置、固定來源與啟動檢查；主線已啟動模型重建，七個模型及完整評分尚未宣稱完成。** 本頁沒有本窗口 MAE 或採用結論，後續需依完整預測封存及獨立評分收據更新。

使用者唯一提交仍是 v22，既有 Public MAE 1,998,568.5706 元，Private 未知。這輪只檢查既有完整 v22 與 v24 配方的時間穩定性，不改選定版本、不訓練正式 test 全量模型、不新增 CSV，也不代為上傳或勾選 Kaggle。

## 為什麼換切點

之前主要比較每年 8 月起的年度窗口。這次固定把切點移到 2023-02-11，檢查原有比較是否依賴切分日期。新的切點必須重建對應的模型和價格狀態；不能把已看過該期交易的後期模型切片回測。

|用途|固定範圍|列數|
|---|---|---:|
|fit|交易日期早於 2023-02-11|272,117|
|query|2023-02-11 至 2024-02-10，兩端含|107,169|

使用 canonical train，不混 valid、不讀 test。既有 pure-X 準備保留來源 2023 地理 snapshot 早於切點的證據；原 R6 地理供 tree／NN，R39 補位地理只供 spatial。[純X來源 manifest](../evidence/outputs/round90/preparation/manifest.json)早已保存，這次未重新準備或更改它。

這些歷史資料与既有選型窗口重疊，因此是錯開日期的穩定性檢查，不能称為新盲測，也不能據此聲稱 Private 一定改善。

## 固定重建什麼

共 5 個樹模型＋2 個 NN，全部依原配方串行訓練：直接總價 CatBoost、單價 LightGBM、log CatBoost、單價 CatBoost seed42／137、area NN 與 prior NN。四個新 base 各套同切點新 fit 的 reference／spatial，再得到兩個完整 median。

~~~text
N   = .75 × (.5 × N_area + .5 × N_prior) + .25 × L_unit
H   = .75 × (.5 × C_raw  + .5 × C_log)   + .25 × L_unit
U22 = .75 × C_unit42 + .25 × L_unit
U24 = .5 × C_unit42 + .5 × C_unit137
v22 = median(RS(N), RS(H), RS(U22))
v24 = median(RS(N), RS(H), RS(U24))
~~~

U24 保留原配方，不再混入 LightGBM。這次不加入 Attention 或信任折半對照，不搜尋比例、seed、輪數或 epoch。新 fit 重建 LGB category maps、NN encoder／ramps、分組五折 OOF 與 full prior、日期權重及 reference／spatial；模型保存重載要涵蓋全部 query。

## 先封存預測，再讀評分標籤

隔離擷取程序先核 raw train pickle 與兩種 fit 索引，只輸出 fit_y.npy。pickle 會反序列化整張來源價格表，不能稱擷取程序從未接觸 query 價格；但沒有計算 query 統計、沒有輸出 query 標籤。模型 driver 只讀 fit-only 檔。

raw cache 的 fit_source_cache_indices 專供價格對齊；NN 的 OOF source_indices 用 canonical fit_indices，不能混為同一軸。七模型、四個修正分支與兩個完整預測全部重載核對後，driver 才寫 sealed_manifest。另一支獨立評分工具須在封存成功後才取得 query 標籤；其來源另凍結。這份前置快照不包含尚未定稿的 scorer 或活動模型產物。

事前評估契約已固定：TWD MAE、原群組加權、新舊建案、季度／月份／城市／價帶／尾部，以及 project 和三個月循環 bootstrap，各 2,000 次。區間不是 Private 勝率；真價分帶只作事後診斷，不能用於未知 test 分流。無論本窗口結果如何，都不自動切換提交。

## 已完成的前置與尚待完成的工作

|項目|本快照保存的事實|
|---|---|
|fit 標籤擷取|0.813 秒，272,117 個 float64 完整重載；0 native fit|
|driver 合成|18 項通過；最終收據 1.565 秒，無真資料／模型 fit|
|主線啟動檢查|3.328 秒；核 33 份凍結來源，pure-X／fit 標籤契約及日期規則通過|
|工程修正|訓練前獨立審查發現 seed42 為直接 R19 state、seed137 才包 wrapper；已修正並補三項測例，原首版收據保留|
|七 native fit、修正狀態、封存預測|主線執行中；本頁不填完成時間或結果|
|完整 query 評分及完成後獨立審查|待預測封存後執行與補充|

已過的合成檢查不等於完成真實七模型重建；來源可追溯也不等於乾淨環境端到端重訓驗證。固定 GPU seed 不保證重訓逐位一致，保存模型重載與重訓是不同要求。

## 程式與原始收據

- [R97 driver](../research/src/round97_offset_experiment.py)：七模型、四個 base／R／S、兩個 median 與預測封存。
- [隔離 fit 標籤擷取](../research/src/round97_prepare_fit_labels.py)：按 fit raw cache 索引擷取，不計 query 統計。
- 已保存的 R90 helpers：[純X資料](../research/src/round90_offset_data.py)、[四樹](../research/src/round90_offset_trees.py)、[兩NN](../research/src/round90_offset_nn.py)。
- [固定計畫](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/reports/round97_plan.md)、[18項合成收據](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/reports/round97_driver_selfcheck_v2.json)、[原首版收據](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/reports/round97_driver_selfcheck.json)。
- [fit-only manifest](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/outputs/round97_inputs/manifest.json)只保存身分與小型收據，價格 NPY 不隨包提供。
- [啟動檢查](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/outputs/round97_launcher_20260929/check_launch.json)、[凍結請求](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/outputs/round97_launcher_20260929/research_request.json)、[啟動身分](../evidence/outputs/research_snapshot_r97_20260929/preparation/original/outputs/round97_launcher_20260929/pipeline_started.json)。
- [本次原bytes複製清單](../evidence/outputs/research_snapshot_r97_20260929/preparation/copy_manifest.json)。

這是本機前置快照，尚未為 R97 commit、push 或打 ZIP。既有 R95／R96 完成批次與所有封存收據、379 支歷史原碼、正式 REPORT／PDF／PPTX 不動。本頁的「進行中」是快照時狀態，不是聊天結束後自動持續工作的承諾；實際執行以主線程序與後續收據為準。
