# 錯開時間切點：R97 完整重建與評估已完成

2026-09-29（Asia/Taipei）。**七個原生模型、全部新 fit 狀態、完整 query 重載、獨立評分及完成後審查均已完成。** 這輪比較既有 v22／v24 配方的時間穩定性，不是新提交版本。v22 仍是使用者唯一選定提交；已回報 Public MAE 1,998,568.5706 元，Private 未知，未新增 CSV 或操作 Kaggle。

## 結果與反證一起看

MAE 單位為新臺幣元，越低越好；下表差額為 v24−v22，正數表示 v24 較差。

|窗口|v22 整體 MAE|v24 整體 MAE|差額|固定加權差額|
|---|---:|---:|---:|---:|
|2021-08-11～2022-08-10|2,309,352.36|2,275,409.14|−33,943.22|−30,594.69|
|2022-08-11～2023-08-10|1,407,387.07|1,407,340.72|−46.35|−21.82|
|2023-08-11～2024-08-10|1,383,228.41|1,373,195.92|−10,032.49|−11,064.53|
|**2023-02-11～2024-02-10（本次）**|**1,336,410.67**|**1,340,328.14**|**+3,917.46**|**+4,561.93**|

新窗固定加權 MAE 分別為 **1,392,890.34／1,397,452.28 元**。只有最後一列是本輪重新訓練；前三列讀既有保存結果。資料與 fit 範圍重疊，不能當成四個獨立樣本或相加計算勝率。

![四窗口 MAE 差額](../evidence/outputs/round97_summary_v2/four_window_mae_delta.png)

本次 v24 整體高約 0.293%，四季點估計均較高，新案退 7,511.52 元；但 seen 建案改善 543.59 元，2,000 萬至 1 億元的兩個價帶亦改善。低於 2,000 萬及億元群退步；億元群只有 270 列，仍完整揭露，不能因未達 500 列通報條件而略去。原三個 8 月窗口中，v24 的整體與加權 MAE 全部較低，這項相反證據保持。

|固定配對抽樣|整體差額的 95% 區間|加權差額的 95% 區間|
|---|---:|---:|
|建案群組|[−6,109.91, +15,316.77]|[−6,746.06, +16,849.02]|
|三個月循環區塊|[+488.17, +7,176.21]|[+903.73, +9,072.55]|

兩個建案區間跨零，兩個月份區間偏退；不是全面一致的顯著差異，更不是 Private 勝率。每種抽法各 2,000 次，固定 seed 與 stage index 已事前寫入契約。這些區間不涵蓋累積選型、重訓隨機性或未來分布改變。完整群組／城市／月份／價帶／尾部見[評分報告](../evidence/outputs/round97/evaluation/report.md)、[原始聚合結果](../evidence/outputs/round97/evaluation/results.json)及[獨立審查](../evidence/reports/round97_independent_review.md)。

## 為什麼重建，實際重建什麼

原回測主要從每年 8 月起算；本輪事前固定 2023-02-11 切點，fit 為該日以前 **272,117 列**，query 為 2023-02-11 至 2024-02-10 共 **107,169 列**。只用 canonical train，不混 valid、不讀 test。歷史資料與舊選型窗口重疊，應稱錯開起點的穩定性檢查，不能稱新盲測。

新切點不能沿用已見過未來交易的模型。R90 pure-X 準備與 helpers 在 R97 正式使用：核既有 2023 地理來源早於新切點，重建 5 個樹模型＋2 個 NN，LGB 類別映射、NN encoder／ramps、group 五折 OOF／full prior、日期權重、reference／spatial 也都來自新 fit。group OOF 不是時間 OOF。

~~~text
N   = .75 × (.5 × N_area + .5 × N_prior) + .25 × L_unit
H   = .75 × (.5 × C_raw  + .5 × C_log)   + .25 × L_unit
U22 = .75 × C_unit42 + .25 × L_unit
U24 = .5 × C_unit42 + .5 × C_unit137
v22 = median(RS(N), RS(H), RS(U22))
v24 = median(RS(N), RS(H), RS(U24))
~~~

四支 base 各使用同切點的新 reference／spatial。U24 沒有 LGB；native seed42／137 的差與 U22／U24 base 的差不是同一個比較，不把分層 MAE 差相加當成因果貢獻。本輪未加入 Attention、confidence 折半或新比例搜尋。

## 隔離、執行與工程紀錄

隔離擷取器只輸出 fit 標籤；底層 pickle 必須反序列化整張原價格表，此事明確保留，沒有宣稱 extractor 從未接觸 query 價格。driver 僅取得 fit-only 檔，OOF 用 canonical fit index，價格擷取用 raw cache index。所有 native／四個修正分支與兩個最終向量完成保存重載、封存後，另一支固定 scorer 才讀 query y。

|實測項目|結果|
|---|---|
|fit 標籤擷取|0.813 秒；fit 標籤完整重載，0 native fit|
|driver 最終合成／啟動檢查|18 checks、1.565 秒；主線真 API 啟動檢查 3.328 秒|
|5 樹＋2 NN fit 呼叫段|1,044.938＋55.453＝**1,100.391 秒**|
|main|**1,410.656 秒，exit 0**；fit 秒數包含於此，不相加|
|獨立評分程序|**29.578 秒，exit 0**|
|完成後獨立審查|1.500 秒，exit 0；220 個唯一 SHA、固定公式及 4 個保存區間核對|

部分 fit 段含 helper 檢查，不能全稱 GPU 運算秒數；也沒有量測整段聊天工時。main 原生模型只跑一次，沒有自動重試。保存重載通過不代表跨硬體重訓位元一致，也不等於全新環境重訓正式 test 模型。

訓練前已修正 seed42 直接 state／seed137 wrapper 的工程契約；最初 driver 合成收據保留。scorer 的合成斷言、import／syntax 失敗與後續 43 checks 通過收據也全保留，均不算真候選 fit 失敗或重新訓練。[語法失敗原稿](../evidence/outputs/round97_evaluator_prefreeze/attempt2_syntax_failure.py)是**不可執行的歷史證據**，不是入口。

四窗口摘要初版有圖表字型警告，保留[原稿](../evidence/reports/round97_stability_summary.md)、[原圖](../evidence/outputs/round97_summary/four_window_mae_delta.png)與原 source。第二版改 ASCII 減號並完整列出分群反證，圖已目視確認；沒有重新訓練或評分。[目前摘要](../evidence/reports/round97_stability_summary_v2.md)及[其收據](../evidence/outputs/round97_summary_v2/receipt.json)保存實際來源。

## 程式、收據與交接範圍

- [固定計畫](../evidence/reports/round97_plan.md)、[main 前置審查](../evidence/reports/round97_preflight_review.md)、[scorer 前置審查](../evidence/reports/round97_evaluator_preflight_review.md)。
- [fit 標籤擷取](../research/src/round97_prepare_fit_labels.py)、[七模型 driver](../research/src/round97_offset_experiment.py)、[獨立 scorer](../research/src/round97_offset_evaluate.py)、[獨立完成審查程式](../research/src/round97_independent_review.py)、[目前摘要程式](../research/src/summarize_round97_stability_v2.py)。
- R90 immutable helpers：[純X](../research/src/round90_offset_data.py)、[四樹](../research/src/round90_offset_trees.py)、[兩NN](../research/src/round90_offset_nn.py)。
- [預測封存 manifest](../evidence/outputs/round97/sealed_manifest.json)、[main execution](../evidence/outputs/round97/execution.json)、[評分 execution](../evidence/outputs/round97/evaluation/execution.json)、[獨立核驗 JSON](../evidence/outputs/round97/independent_review.json)。
- [前置複製清單](../evidence/outputs/research_snapshot_r97_20260929/preparation/copy_manifest.json)不可變；[本次完成複製清單](../evidence/outputs/research_snapshot_r97_20260929/final/copy_manifest.json)另存來源 SHA、排除項及導覽修改前 bytes。

[時間外 L1 組合的有限盤點](../evidence/reports/regression_direction_after_r97_inventory.md)是在 R97 結果未知時完成，該句保留其歷史時序。尚未立項、fit 或挑係數；不能把旧同窗 GroupKFold 選比例當成真正時間前置 meta-fit。日後是否研究由新的固定計畫決定。

本機快照以已發布 40b2f36c…6faf 為基礎，**尚未為 R97 package、commit、push 或打 ZIP**。文件／程式保持原 bytes；交易表、labels、模型、逐筆向量、encoder／prior／ramp／category state 不隨包提供。聚合 metadata 中的 artifact SHA 只記錄身分，不代表私有依賴已包含。379 支歷史 source manifest、正式 REPORT／PDF／PPTX、v22 推論設定及選定 CSV 均不改；這不是完整可重訓備份。
