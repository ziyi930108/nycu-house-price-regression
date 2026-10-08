# 2026-10-08：Ordered boosting 既有方向核對

本輪不追加 Ordered 訓練。R17 已有一次真正的 GPU／MAE／2,999 樹房價 fit，而且出現明顯反證；不能再把這個方法說成從未試過。R79 的單價加日期權重分支尚未找到 Ordered 真 fit，屬於未執行的交互假說，並非已證明有益。此次不建立 R111 或額外提交檔。

|證據類別|實際內容|來源|
|---|---|---|
|正式真 fit|R17 raw、均一權重、95欄／23類別，只把 Plain 改 Ordered；development 240,316／81,234；原生 Ordered、depth8、2,999樹、保存重載全 query 一致|[helper](../../src/round17_ordered_geo.py#L88)、[原生 metadata](../round17/models/development/metadata.json)、[正式工作紀錄](../../reports/round17_work_log.md#L45)|
|最強反證|融合與一次 reference 後，整體 MAE 1,502,390.31→1,593,812.86（+6.0851%）；加權 +6.0414%；未見具名建案 +7.4351%；Q2 +9.2054%；四項 gate 全失敗|[結果表及分組](../../reports/round17_report.md#L23)、[拒絕判定](../round17/accepted_candidate.json)|
|獨立重播|保存的 Ordered CBM／全部 81,234 query 原始輸出、融合及指標核對；這是重播驗證，並非新增 fit|[development replay](../round17/audit/development_replay.json)|
|人工相容性|R17 另有 512 列人工、2 樹 GPU／MAE／Ordered 成功；它與正式房價 fit 分開，不能用人工速度推估正式成本|[方法紀錄](../../reports/round17_categorical_method_review.md#L34)|
|Fake／拒絕測試|R19 helper 的合成測試故意把 native mode 改 Ordered 並要求拒絕；後輪 verifier／archive 的相同測試不是 Ordered 真 fit|[R19 synthetic check](../../src/round19_unit_geo.py#L522)|
|當前單價分支|R79 的三個窗口及 final_model metadata 全為 native Plain；它沿用 R19 的單價、面積權重並乘保存 R63 日期權重|[R79 helper](../../src/round79_recency_unit.py#L18)、[development metadata](../round79/development/model/metadata.json)|

R17 開發期被拒絕後，confirmation、auxiliary 與 final 均未執行；不能填零或稱為三窗證偽。上述 MAE 是歷史已保存報告，本次沒有重算價格誤差、載入模型或預測陣列。

原始 train_catboost 明列 Plain，不是依小資料 GPU 預設而隱性使用 Ordered。[官方參數](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type)確認 Ordered 可指定於 CPU／GPU，GPU 大於50,000列預設 Plain；[官方 objective 表](https://catboost.ai/docs/en/concepts/loss-functions-regression#used-for-optimization)列 MAE 支援 GPU。此次 CatBoost 1.2.10 native parser 對完整 R79 參數及只改 boosting_type 的版本均接受，沒有 fit 或 GPU 呼叫；parser 通過不證明全量記憶體、耗時或泛化。R17 正式 metadata 的 runner_elapsed_seconds 約238.922，僅代表那次240,316列 raw訓練。

[原始 CatBoost 論文](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)把 permutation-driven boosting 與類別統計處理分成兩個方法，用來處理訓練時 target 重用造成的 prediction shift。Plain 本身仍有原生類別處理；Ordered 不等於日期排序、has_time=True、建案隔離或未知 Private 改善。

將來若重新提出 R79 unit/date weights 的 Plain→Ordered，只能稱為舊方法在不同 target／weight 參數化上的新交互假說。單價與面積／日期權重會改變 boosting 梯度及 CTR 所見的 target 分布，因此不能把 raw 的失敗直接當成所有 unit 版本的定理；同樣也沒有證據預期它能克服已保存的強反證。主線本次已決定不追加。未提出其他有充分新證據的替代訓練，也不安排 seed、depth、CTR 或 fold 超參數掃描。

未來若獲選，須另立固定規約，保留95／23、seed42、depth8、2,999樹、原面積和保存日期權重、全部query／fit狀態分離及原組合公式；只改 boosting_type，檢查可能連動的 native defaults。資源不足應保存技術中止，不在見到房價誤差後改配方。重用歷史窗口的選型偏差仍在；新候選 Public／Private未知，v22保留。

本核對只讀 src／reports、指定歷史 aggregate metadata／decision／replay，並查演算法官方文件；沒有查內政部成交答案、沒有新模型 fit、沒有 Kaggle 或 GitHub 操作。[機器證據與來源雜湊](direction_audit.json)。
