# R100 driver 實作與前置核驗

2026-09-29。此收據記錄真實 prior／NN 訓練前的 source、metadata 與人工陣列核驗；沒有讀實際 fit/query 價格或逐筆資料，沒有建立真 prior、模型、預測或評分。正式計畫與主線執行收據另行保存。

## 介面及隔離

- [driver](../src/round100_temporal_prior_experiment.py) SHA `8d91db83751c6b55e72d0e07a3d0674458af0c2235896e4d04f6c826549b26bf`；[共用契約](../src/round100_contract.py) SHA `f6483f2e1e92f8286ea7aed081080c287888b1b6c9f4a3a519964f63ffdd77f5`。目前純 AST 來源閉包 33 檔；正式 request 仍須在執行前固定完整 SHA。
- `--run --request PATH` 父程序只讀 metadata，依 development、confirmation、auxiliary 各開一個 `--stage` 子程序。每期 C、T 依序串行；所有六次 fit 成功封存後才允許外部 scorer 評分。driver 不呼叫 scorer。
- 每期只載自己的 root 隔離 fit-y 及 R63 NPZ；canonical 索引作 group fold，raw-cache 映射只作價格對齐。所有 prepared 恰有原 R63 九鍵；query 標籤不出現在 trainer 契約。來源 raw price pickle 不由 driver 打開。
- helper 實際五參數及 `state_sink` 介面相容。逐季／fold 完整 state 保存到 `prefix_states/q{day}_fold{f}.json`，附 SHA；evidence、所有新 diagnostics、NPZ、模型及向量均納 stage artifact map。既有檔案拒覆寫，失敗收據保留，不自動重試。

## 實際已核

只讀三期既有 JSON，核 R48／54／60／63／71／79／83 動態 artifact keys、R/S／原 encoder／兩種 geography 的綁定、root fit-label nested `stages[s].arrays`、pure-X 五種 array record schema。沒有為這项核驗載這些陣列。

Pure-X 保存證據每期只排除 56 筆暖身，invalid-date 0；eligible 為 240,260／321,494／134,749。driver 會在真執行再核 mask、原 folds、canonical 及 full-fit float32 日期權重 SHA。兩臂共用同一 mask，C 取原 OOF factor 切片；T 只換訓練 factor，先 float64 `A*prior/1e6` 再轉 float32。不重正規化 loss weights，query factor／encoder／R/S 保持原完整 outer-fit 範圍。

R63 trainer CLI、model.pt、training.json、predictions_twd.npy 與全 query 重載欄位已對照原 source；沒有套用 Attention 的 ratio/execution 輸出契約。各臂初始化 bias 使用各自 y/factor 的原公式，不錯比原 baseline 數值。完整 N 使用 R71/v22 公式 `.75*(.5*area+.5*prior)+.25*LGB`；不沿用更早 R63 含 oldraw 的歷史 blend。執行時會核 R71 N／diagnostics、R79 selected N 與完整 v22 median。

[人工自測](round100_driver_selfcheck.json) SHA `6302d7b3efe04e77cb714f0ebd027cd3408bd3e936a133edfaf4bb95093601db`：首次 14 項通過，內部計時 0.000 秒是解析度限制，命令程序 1.131 秒；8 個共用陣列、C 原 factor 切片、T factor 公式、輸入不變及空 mask／錯 dtype／非正值拒絕。這不是實際 trainer 或端到端測試。

與 scorer 作者交叉核驗時發現 pure-X audit 執行順序是 auxiliary／development／confirmation，不能等同模型固定順序。scorer 在真執行前改為驗集合完整，保留 audit 原順序；其 38 項人工檢查另有獨立收據。沒有將此 source 修正稱為真 fit／score 失敗。

## 成本與限制

同架構歷史 R63 三次 NN fit 合 56.436 秒；六次約 113 秒僅粗估，不是 R100 實測。原 R60 每期五折加 full-state 準備為 8.453／11.453／5.063 秒，新的多季 prefix 數量較多，CPU prior 準備及完整 state 落盤須另計，不能混作 GPU 訓練時間。

此時 source/API 沒有剩餘阻擋，仍須主線固定 request 後執行，並讀真實 exit／重載／評分證據才能下結論。保留重用窗口、單 seed 變異及只有 prior 是逐列時間前置的限制。v22 唯一提交、Private 未知；本文件不採用候選或建立 CSV。
