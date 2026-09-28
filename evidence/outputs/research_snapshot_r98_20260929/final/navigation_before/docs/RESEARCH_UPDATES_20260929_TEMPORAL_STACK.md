# 較早預測學組合權重：R98 三窗研究已完成

2026-09-29（Asia/Taipei）。R98 完成三個 CPU 線性組合、全部預測封存、獨立評分與保存結果稽核；**沒有足夠穩定證據取代 v22**。本輪只改最後的組合方法，沒有重新訓練樹或 NN、產生新 CSV、操作 Kaggle 或改變唯一提交。v22 已回報 Public MAE 1,998,568.5706 元，Private 仍未知。

## 研究問題與結果

原方法對完成成交參考及空間修正的 N／H／U 三個總價分支逐筆取中位數。本輪問的是：能否只用較早時間的真正 forward 預測與成交標籤，學一組非負且和為 1 的全域權重，使更晚窗口的總價 MAE 降低？它不是在同一個 query 上調比例，也没有用 Public 搜權重。

下表為 R98−v22；正數表示 R98 誤差較高，單位為新臺幣元。

|query 窗口|v22 MAE|R98 MAE|整體差|固定加權差|
|---|---:|---:|---:|---:|
|2022-08-11～2023-08-10|1,407,387.07|1,416,185.71|+8,798.64|+8,728.33|
|2023-02-11～2024-02-10|1,336,410.67|1,332,155.65|−4,255.03|−5,807.86|
|2023-08-11～2024-08-10|1,383,228.41|1,388,419.07|+5,190.67|+5,002.34|

兩個 8 月窗口的整體及固定加權 MAE 都較高；錯開起點的 offset 窗口則較低。三期均有事前定義的大群退步超過 5%，因此連 offset 也未通過完整研究門檻。建案與三個月循環區塊、整體與加權共 **12 個配對區間全部跨零**；不能把門檻未過說成全面顯著退步，更不能稱為 Private 勝率。

有利與不利的分群都保留：offset 新案改善約 12,915 元、億元群 270 列改善約 238 萬元，但 seen 退約 6,494 元，嘉義縣 1,532 列退 10.769%；confirmation 的 2,000–5,000 萬元帶改善约 20,060 元，但億元群 319 列退約 472 萬元。development 基隆市及部分月份、confirmation 屏東縣也超過固定 5% 通報門檻。真價帶只作評估診斷，不用來依未知真價分流。完整月份／城市／季度／價帶、五支向量及全部區間見[評分報告](../evidence/outputs/round98/evaluation/report.md)與[聚合 JSON](../evidence/outputs/round98/evaluation/results.json)。

## 如何避免同窗學習與評分混在一起

三分支原生模型、reference／spatial 均使用其原有合法時間切點狀態；R98 重用已核對 SHA、列序及原 v22 median 的保存 forward 預測。N 為 NN 組合分支、H 為總價／log 樹組合分支、U 為近期單價樹組合分支，全部已還原為總價並完成 R/S。

|本次 query|學組合權重用的歷史 forward block|meta 列數|query 列數|
|---|---|---:|---:|
|development|2021-08-11～2022-08-10 全期|105,511|81,234|
|offset|上述全期＋development 至 2023-02-10 的 31,801 列|137,312|107,169|
|confirmation|上述全期＋development 全期|186,745|133,697|

每一期 meta 日期均嚴格早於該期 query cutoff，canonical 索引不重疊，不重排預測與標籤。較早期 query 標籤合法成為較晚期的 meta 標籤，不代表同一期 query 參與其自身 fit。[來源盤點](../evidence/reports/round98_artifact_inventory.md)及[資料準備 manifest](../evidence/outputs/round98_inputs/manifest.json)記錄實際身分。

隔離擷取器必須反序列化原始整張價格 pickle，但只輸出各期合法 meta_y；這個界線如實保留。三個 driver 依序在獨立程序中執行，只准讀自身 11 個精確名稱及路徑的陣列，沒有 query_y。所有 query 預測保存重載一致並共同封存後，另支 scorer 才讀 query 真價。評分器只對 meta_y 核檔案 SHA，不反序列化它。

## 固定模型與求解方式

~~~text
R98 = (N × wN + H × wH) + U × wU
wN, wH, wU ≥ 0；wN + wH + wU = 1
用較早 meta 資料最小化每列等權的總價 MAE
~~~

沒有截距、日期衰減、分群權重、額外正則化或權重網格。共同除以 100 萬只是數值尺度；輸出仍為未取整的 float64 新臺幣總價。固定 SciPy 1.18.1／HiGHS 1.12.0 的 highs-ds 與單執行緒選項，300 秒上限，無重試、fallback、事後 clip 或 normalize；多解時接受固定 runtime 與列序的求解結果，不看 query 再挑。

|窗口|N 權重|H 權重|U 權重|
|---|---:|---:|---:|
|development|0.228046|0.269965|0.501988|
|offset|0.259077|0.270462|0.470461|
|confirmation|0.331824|0.271627|0.396549|

求解器使用等價的 dual LP，從原始 inequality marginals 取三個權重，核 primal／dual gap、可行性與互補性。這些證書驗的是數值求解，不是未來 MAE。固定全域三權重的可行集合不一定包含逐筆 median，因此即使 meta 目標達最優，也不保證勝過 v22 median，更不保證較晚 query。數學與合成反例見[helper 設計](../evidence/reports/round98_helper_design.md)。

## 執行、審查與失敗證據

|實測項目|範圍及時間|
|---|---|
|資料準備|3.687 秒；各期輸入保存重載一致，無新模型|
|三次 CPU meta-fit 呼叫|合計 2.828 秒；0 原生模型 fit／載入，0 GPU|
|main|6.891 秒，exit 0；fit 呼叫已包含其中|
|獨立評分|26.609 秒，exit 0；三期共同封存後執行|
|保存結果稽核|2.609 秒，exit 0；233 個唯一 SHA、9 份保護檔、12 個保存區間|

合成自查 helper 42 項、scorer 42 項均通過。前置曾因 artifact allowlist 過窄擋住合法固定 JSON；helper 也保存首輪合成失敗與修正，沒有把這些算成真實候選 fit 失敗。driver 在真 fit 前補上 solver 失敗 metadata／arrays 的保存，以及精確 stage readset。原稿、attempt 收據、最後來源與工程時序均保留。

[保存結果稽核](../evidence/reports/round98_independent_review.md)由 helper 作者執行，明確不是對自己求解器的完全獨立重作；scorer 由另一代理編寫，前置另有代理審查。稽核没有再讀真價、重新求 LP、計算逐筆 MAE或重抽 bootstrap。最後 request 的獨立複核與[書面 preflight](../evidence/reports/round98_driver_preflight_review.md)晚於啟動通知，沒有美化成書面先於 launch。

## 程式、交接與研究界線

- [固定計畫](../evidence/reports/round98_plan.md)、[artifact audit](../evidence/outputs/round98_artifact_inventory/audit.py)、[fit/meta 準備](../research/src/round98_prepare_inputs.py)。
- [共用契約](../research/src/round98_contract.py)、[L1 helper](../research/src/round98_l1_stacker.py)、[每期 driver](../research/src/round98_temporal_stack.py)、[scorer](../research/src/round98_temporal_evaluate.py)、[保存結果稽核程式](../research/src/round98_independent_review.py)。
- [封存 manifest](../evidence/outputs/round98/sealed_manifest.json)、[main execution](../evidence/outputs/round98/execution.json)、[scorer execution](../evidence/outputs/round98/evaluation/execution.json)、[稽核 JSON](../evidence/outputs/round98/independent_review.json)。
- [前置複製收據](../evidence/outputs/research_snapshot_r98_20260929/preparation/copy_manifest.json)保持不變；完成複製及導覽核驗另存，模型狀態、labels、逐筆價格／預測與 solver／bootstrap 陣列均不隨包提供。

這是歷史窗口上的時間前推研究，不是新盲測；offset 與其他窗口重疊，資料也已反覆參與選型。缺少 auxiliary 以前的 forward block，所以本次不能冒稱通過原三個 8 月窗口的完整採用門檻。下一個「讓 median 成為第四欄可選基線」只屬有限提案，尚未立項或 fit；meta 最優可不差於 median，也不代表 query 改善，不能直接替換 v22。

本頁內容於發布前核驗，基底為 2eec4b006db5ecf5b0624f791b90faf02ae1799f；**當前發布身分以此儲存庫 Git main 歷史與 SOURCE_MANIFEST 為準**，根工作區另存發布收據。379 支歷史 source manifest、正式 REPORT／PDF／PPTX、v22 推論設定及選定 CSV 均保持。此為文件與程式快照，不是包含私有資料與狀態的完整可重訓備份。
