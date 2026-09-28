# R98：歷史 forward 分支 artifact 與時間隔離盤點

2026-09-29。本次已執行只讀 audit，**沒有讀取任何 fit／query 真價、沒有 fit、載模型、模型推論、MAE 或候選評分**。只開啟 pure-X、索引、已保存預測及來源 metadata；結果不代表 stacker 已實作或有效。原模型、R97、GitHub 快照與根交接未改。

- [可重跑 audit.py](../outputs/round98_artifact_inventory/audit.py)：來源 SHA 03be8b88aaffe722a25784cc05bd3b0944d95a4678f4b52b713461f5211f30ee。
- [完整 results.json](../outputs/round98_artifact_inventory/results.json)：SHA 11eccec76cee77b5d660bf60eb1f9679b0cb71c9cb1ba4a5d044dd79df54b359。完整 file/value SHA、canonical indices、分支 authority 與日期契約均在此；下表只縮寫 SHA 方便閱讀。
- 實際 audit 2.578 秒，外層程序 3.471 秒、exit 0；核 **58 個 source/config、108 個已開啟檔案 SHA**。所有陣列皆 allow_pickle=False；唯一反序列化的 pickle 是無目標 X 特徵，沒有 model／reference／spatial state pickle。

## 四個可重用 block 與 authority

|block|native fit 嚴格早於|query 末日（含）|fit 列數|query 列數|
|---|---|---|---:|---:|
|auxiliary|2021-08-11|2022-08-10|134,805|105,511|
|development|2022-08-11|2023-08-10|240,316|81,234|
|confirmation|2023-08-11|2024-08-10|321,550|133,697|
|offset|2023-02-11|2024-02-10|272,117|107,169|

舊三期以 [R79 plan](../outputs/round79/plan.json)固定切點與 split_contract，fit／query 索引逐值對照 canonical 日期遮罩；R46 model、R71、R79 保存索引的 dtype、shape、bytes 全相同。三份 results 的日期／列數／query-label 邊界亦核對。N 的來源是 [R71 results](../outputs/round71/results.json) 的 seed42 完整分支；H 是 [R46 results](../outputs/round46/results.json) 的 recency log 完整分支。兩者 artifact SHA 另與 R79 plan.input_sha256 精確一致。U／control 由 [R79 results](../outputs/round79/results.json)授權，control 另外核 prediction_sha256。

offset 使用 [R97 sealed_manifest](../outputs/round97/sealed_manifest.json)（429e5405…893cb）的 branches.N/H/U22.spatial 與 predictions.v22。U22 是既有 v22 單價分支，不能誤拿 U24、native 或 base。其 canonical／raw 兩種索引對應 [R90 pure-X manifest](../outputs/round90/preparation/manifest.json)；保存 Xquery 與 canonical X 的 query 選列完整一致。只核 R97 已保存的七模型重載與來源收據，不再開模型。

全部 16 份向量為非負、有限 float64；每份行數等於該 block.query_indices。四個 control 都通過 median(column_stack([N,H,U]),axis=1) 的 dtype／shape／bytes 全等，沒有改價、取整或重排。

|block／角色|實際相對路徑|file SHA256 前綴|value SHA256 前綴|
|---|---|---|---|
|auxiliary / N|outputs/round71/auxiliary/seed42/nn_spatial.npy|63dd3be20c77…|c1566c05e4ed…|
|auxiliary / H|outputs/round46/auxiliary/recency_log_spatial.npy|41424bbabb65…|9241285b9a6c…|
|auxiliary / U|outputs/round79/auxiliary/unit_spatial.npy|5f7e2089b050…|56d98d8bf5d3…|
|auxiliary / control_v22|outputs/round79/auxiliary/predictions.npy|f4ac70c6c738…|35610da01aaf…|
|development / N|outputs/round71/development/seed42/nn_spatial.npy|605ab0cdffb8…|ff516f5893c8…|
|development / H|outputs/round46/development/recency_log_spatial.npy|0c5a0bcb82f8…|c5271c426677…|
|development / U|outputs/round79/development/unit_spatial.npy|8eb22194d22c…|6960e90f311f…|
|development / control_v22|outputs/round79/development/predictions.npy|82dada9c5158…|6307476eb4fc…|
|confirmation / N|outputs/round71/confirmation/seed42/nn_spatial.npy|948ba6c8887f…|cbe1854beb05…|
|confirmation / H|outputs/round46/confirmation/recency_log_spatial.npy|bb96799aa97f…|c3c0f22d294e…|
|confirmation / U|outputs/round79/confirmation/unit_spatial.npy|01cce0d58655…|f7623511dc00…|
|confirmation / control_v22|outputs/round79/confirmation/predictions.npy|f7fb850c4f22…|80ee6c39452e…|
|offset / N|outputs/round97/reference_spatial/N/spatial.npy|4b4ed462ecbb…|785972d72c28…|
|offset / H|outputs/round97/reference_spatial/H/spatial.npy|bd6d4b91bf97…|41e24b306249…|
|offset / U|outputs/round97/reference_spatial/U22/spatial.npy|4204e008d509…|e53e082ebb1f…|
|offset / control_v22|outputs/round97/v22_predictions.npy|b957793d3aaf…|d63de133cd3d…|

語意 value SHA 統一使用 contiguous little-endian float64 bytes；索引使用 little-endian int64。讀向量只為身分與原 control 公式核對，沒有真價輸入，因此未計算誤差。

## Meta-fit 的實際前綴與順序

|目標 block|禁止跨越的 cutoff|來源順序與列數|meta 列數|實際日期最小～最大|
|---|---|---|---:|---|
|development|2022-08-11|auxiliary 105,511|105,511|2021-08-11～2022-08-10|
|offset|2023-02-11|auxiliary 105,511＋development 31,801|137,312|2021-08-11～2023-02-10|
|confirmation|2023-08-11|auxiliary 105,511＋development 81,234|186,745|2021-08-11～2023-08-10|

offset 的 development 前綴實測 **31,801 列**，日期 2022-08-11～2023-02-10。先取 source 原 query 順序的日期遮罩，再依 auxiliary、development 的順序串接；**不是排序日期或重排預測**。meta 索引／raw source 索引／三欄矩陣 bytes SHA 都已保存於 meta_fit，各 piece 另保存 selected_source_positions_values_sha256。

三個目標皆通過：meta 日期嚴格早於各自 cutoff、meta 列全部屬於目標 native fit 集、零重複、與目標 query 零重疊。auxiliary／development／confirmation 三個原 forward query block 互不重疊；offset query 雖與舊 block 有重疊，但其 meta-fit 已按自身切點截斷，不能使用整個 development 答案。

## 主線可直接沿用的介面

results.schema_version=1；blocks 包含 auxiliary／development／confirmation／offset，每個 block 有：

- cutoff、query_end、fit_indices／query_indices、branches.N/H/U、control_v22。每份索引或向量 record 含 path、file_sha256、dtype、shape、values_sha256、authority。
- feature_path、feature_row_axis、canonical_mapping、raw_train_source、authority_sha256、model_origin。前三期 feature_path 是 raw 行軸的 train_features.pkl，必須先按 canonical_mapping 取列，再套 query_indices；offset 的 feature_path 已是 query-local Xquery.pkl。
- canonical mapping 為 outputs/round6/geography/development/train_source_cache_indices.npy，共 455,255 列，對應 raw feature cache 455,784 列；不能把 canonical query index 直接當 raw label index。
- raw_train_source 只宣告原來源 SHA，**本 audit 連原價格 pickle 位元組都沒有重新讀取或 hash**。真正 labels 擷取應由主線的隔離 preparer 另核原價格 SHA，按保存 mapping 只輸出該 stage 已授權 meta 列。

meta_fit[development/offset/confirmation] 含 pieces、source_order、各 piece 日期遮罩、selected_rows、meta canonical/raw 索引 SHA、prediction_matrix_shape／values SHA。matrix 欄序固定 N,H,U。建議 preparer／driver各 stage 單獨處理：重建相同 selector 並核這些 SHA，先擬合與封存係數及該 stage query 預測，再由獨立評估端取得目標 query y；不可把另一個較晚 stage 的 meta 標籤先放進目前 stage 的記憶體。這裡不產出價格檔、係數或新預測。

## 失敗紀錄與限制

首次 audit 在載入任何陣列之前，因 source allowlist 只允許 src/*.py，錯拒 R97 正式 source map 內的 runs/v1/models/final_config.json。原碼與退出紀錄保留於 [attempt1_source.py](../outputs/round98_artifact_inventory/attempt1_source.py)／[attempt1.json](../outputs/round98_artifact_inventory/attempt1.json)。只放行這個已核實的固定設定檔後，第二次 audit 通過；這是盤點工具的 schema 修正，沒有模型或實驗配方改動。

既有 results metadata 仍含當時的聚合 score 欄位，但本 audit 程式從未存取那些欄位，不用舊或新 MAE 決定 prefix。Native 係數、encoder 與價格狀態未在本次反序列化，模型 origin 依保存的來源、切點、索引及重載收據核驗。配方曾以多輪重用窗口選型；合法時間隔離不等於新盲測，也不能保證 Private。最早 auxiliary 仍沒有更早完整 meta-fit block，本方案沒有替它虛構一次測試。

重跑請指定新的輸出目錄，例如：.venv/Scripts/python.exe -B -X utf8 outputs/round98_artifact_inventory/audit.py --out outputs/round98_artifact_inventory_recheck。既有 results.json 拒絕覆寫。沒有執行此重跑範例。
