# Attention 研究前置快照：進行中

2026-09-29（Asia/Taipei）。這是本機文件準備批次，**三期回歸研究尚未完成，沒有本輪MAE、採用結論或新提交檔**。主線回報R95已啟動，第一期prepared／舊控制重播通過並進入trainer；本頁不把程序啟動當作訓練完成，也不承諾聊天結束後背景持續運作。後續結果須由新的完成證據補充。

使用者唯一提交仍是v22，既有Public 1,998,568.5706元、Private未知。本批不改選定CSV、正式REPORT／PDF或22頁簡報；相關完整模型說明仍見[專案README](../README.md)。前一批[R93](RESEARCH_UPDATES_20260929_R93.md)已完成且未採用。

## 這次研究什麼，為什麼值得試

這輪測試小型欄位 self-attention：將房屋的數值與類別欄位視為各自的token，讓模型依其他欄位調整交互作用。它可能有助於結合地段、面積、屋況與建案資訊，但也可能增加建案身分記憶與高價尾端誤差；這只是待驗假設，不能把架構更新當成精度保證。

新模型只替換原集成中的prior神經網路。其餘area NN、unit LightGBM、時間／相似物件修正、空間修正、另兩個模型分支及融合比例皆沿同窗保存版本。仍預測總價，屬監督式回歸；訓練目標是日期加權的總價絕對誤差，最終評估以新臺幣MAE。

固定模型為144數值＋23類別＋CLS共168tokens、維度64、4heads、2blocks。第一block完整attention；第二block僅CLS作query並裁切不使用的residual／FFN輸出。最後Softplus輸出正ratio，乘同窗價格prior與面積因子还原總價。epoch30、seed42、AdamW固定，微批256累積成2048列一個optimizer step；沒有query early stopping或超參數掃描。

[固定科學計畫](../evidence/reports/round94_plan.md)明列所有設定與採用門檻。[資料整合盤點](../evidence/reports/round94_integration_inventory.md)說明原fit-only encoder、建案五折OOF prior與日期權重；prior含fit標籤，不能稱全部前處理純X。query標籤不傳trainer，driver僅供評分。三個年度窗口已反覆使用，不是新盲測；bootstrap不是Private勝率。

## R94與R95是一次假設的工程恢復

|證據|已完成事實|
|---|---|
|原trainer CPU|45項通過；最早double合成累加器dtype錯誤保留attempt1，未真fit|
|trainer v2 CPU|47項、1.250秒；先驗NPZ九鍵再讀陣列，CONFIG不變|
|v2 GPU資源smoke|2暖身＋3量測groups通過；每2048列均值0.182333秒|
|三期成本估算|1,870.74秒，約31.18分鐘；是合成推估，非已完成fit時間|
|v2 smoke峰值|allocated 565,207,040 bytes；reserved 746,586,112 bytes|
|R94第一次main|7.813秒exit1，在真fit前因R48 encoder JSON無獨立sidecar停止|
|R95恢復|保持同模型與科學計畫；修正JSON reader，以原R48 artifact SHA驗證|

smoke未包含全部真prepared陣列常駐，也未含準備／重載／診斷，不可拿估算當實測完成時間。相同batch保存重載要求逐位一致；不同batch語義合成驗證只聲稱容差內一致。

[R94原審查](../evidence/reports/r94_validation_audit.md)漏抓read_locked還需個別sidecar的契約，原source／失敗／審查均保留；未把它改寫成模型表現失敗。[R95恢復審查](../evidence/reports/r95_validation_audit.md)實核27組JSON與sidecar及3份R48 encoder的原authority SHA。兩編號是同一科學候選，不能算成兩套調參成功案例。

## 程式、授權與待補項目

- 實際訓練器：[trainer v2](../research/src/round94_attention_prior_train_v2.py)；保留[原trainer](../research/src/round94_attention_prior_train.py)與[修訂紀錄](../evidence/reports/round94_trainer_revision.md)。
- 執行與完成後審查程式：[R95 driver](../research/src/round95_experiment.py)、[R95 review](../research/src/round95_review.py)。review程式已保存，不代表已經產生審查結果。
- 失敗證據：[R94 execution](../evidence/outputs/round94/execution.json)、[R94 failed driver](../research/src/round94_experiment.py)、[工程恢復計畫](../evidence/reports/round95_plan.md)與[差異](../evidence/outputs/round95_launcher_20260929/driver_engineering_diff.patch)。
- 合成：[v2 CPU](../evidence/reports/round94_trainer_selfcheck_v2.json)、[v2 GPU smoke](../evidence/outputs/round94_smoke_v2_20260929/smoke.json)；v1與首輪失敗一併保留。
- 上游：[來源與改編說明](../research/upstream/README.md)、[完整Apache-2.0授權](third_party/rtdl_LICENSE.txt)、[固定來源manifest](../evidence/outputs/round94_upstream/manifest.json)。
- 本批：[原bytes複製清單](../evidence/outputs/research_snapshot_attention_20260929_preparation/copy_manifest.json)。

本批沒有複製活動中的R95 fit artifacts，也沒有資料、模型、NPY／NPZ／CSV／pickle或逐筆價格。待補三期正式結果、重載證據、分群診斷、24個配對區間、採用審查及獨立完成後審核；目前不填任何假分數。若通過原門檻，仍需新切點R90重建模型與相關state，不能用後期模型冒充新盲測。

379支歷史source_manifest保持；本批只增加研究來源與證據，未commit、push或打ZIP。這是可追溯的程式／文件快照，不是全新環境端到端重訓備份。
