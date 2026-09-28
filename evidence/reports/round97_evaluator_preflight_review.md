# R97 獨立評分程式前置審查

2026-09-29（Asia/Taipei）。**來源／API 審查無剩餘 blocker。** 評分仍須由主線另凍結 evaluator 後、七模型與兩完整預測封存且 main exit 0 才能執行。本文件不宣稱新窗口分數已產生；沒有呼叫 scorer、模型、真資料 preparation 或訓練。只讀 source、小型 metadata／合成收據及雜湊，未讀 query y／test。

## 與正在執行的主流程一致

最終 driver `00d927b8…14f1` 已有 evaluator 所需 `schema_version=1`、`cutoff`、`query_end` 與 seal 狀態；七 native、四分支三層、兩完整 median、arrays／frames／input/source/artifact 的 record schema 均逐項對上。R90 helper 自存 JSON 沒有強加 sidecar；只有 driver 產生的 seal、另建 evaluation request 使用其真正存在的 sidecar 契約。

preflight 的順序是：先檢 sealed header、七 fits 完成、完整 query 保存重載、input 不變、未先算 query metrics、v22 選定與原保護檔案相等；再核 evaluation request 的固定 source closure／parameters／plan；核 main execution exit 0 與封存完成，檢全部 SHA、純X split、vectors與組合。只有這些都過，才落 `prelabel_gate.json` 並反序列化 raw train 價格。

主流程的 source freeze 不必預先包含尚在寫的 evaluator；使用獨立 `evaluation_frozen_before_query_score` request，SHA 與 parameters 完整相等才接受。不能直接復用 main research_request schema。

- canonical `fit_indices/query_indices` 與 raw-cache mappings 都由 R90 preparation 核身分、重新比對 `mapping[canonical] == raw`。實際 y 使用 `raw.iloc[query_source_cache_indices]`；canonical query indices 只供 cluster singleton 身分，不能拿來抽 raw y。
- raw pickle 整表反序列化發生在完成封存後的 scorer，並非聲稱只從磁碟讀 query 列；未輸出 query 價格向量，只留其內容 SHA。沒有 valid/test 讀取、模型 load/predict/fit、GPU 或改寫預測。
- 七 native 完整向量、N/H/U22/U24 各 base/reference/spatial、兩個 median 的 TWD dtype／shape／file與value SHA均檢查。四 base 重算與兩 median 必須逐位吻合；U24 明確不混 LightGBM。R/S 不在 scorer 重新推論。
- 每個 native／分支層／完整 median 共21份指標獨立保存；各 native 與 median 的表現不可混稱。評分程式不產生CSV、不自動採用v24。

## 固定評估與抽樣

原 `round3_evaluate.score` 提供整體、固定加權、新舊建案、相對季度、calendar月份及城市；本次新增固定五價帶、relative月份及尾部描述。相對季度仍沿原 day offsets：Q1 0–91、Q2 92–183、Q3 184–275、Q4 276起，不是重新分四個等筆數群或曆季。

權重固定 seen=0.36617593949822613、new_nonmissing=0.6338240605017739、missing=0；missing 的實際列仍計入整體MAE。relative月份沿每月11日起算，2023-02-11為0、2024-02-10為11，翌日拒絕；bootstrap uses canonical query singleton IDs及fit/query純X分類。

paired predictions 固定欄序[v22,v24]，差額為v24−v22。project 分層完整建案群抽樣；month四段連續三月循環區塊，各2000次，random generator 確切為 `SeedSequence([20260925,0,family_index])`，project=0／month=1。正權重層缺席的weighted replicate保留無效，不重抽；保存(2000,4)陣列及有效／無效數、兩抽法兩指標共4區間。paired point另與完整 MAE在1e-6內核對。

預先記錄所有季度、或至少500列群，若v24比v22退步超過5%就列出；沒有因此調參或改分流。真價帶、各模型自身最大誤差尾部僅供事後診斷。區間互相依賴，也未校正多年研究選型，不能稱Private勝率、新盲測或保證改善。原三年度的上下文應由主線完成後綜合報告並列，這個單窗口scorer不假裝產生三期新分數。

## 合成、工程紀錄與界線

獨立讀 attempt3：43項passed、0.110秒、0模型操作／GPU，`synthetic_only=true`、`real_price_or_prediction_files_loaded=false`；source自身及其10項import closure均實核SHA一致。本審查未重跑整套合成。

合成包括未封存及未完成reload拒絕、先封存才能讀價、raw positional mapping、origin月份邊界、手算MAE與固定權重、五價帶分割、零誤差share、缺正權重層、兩bootstrap常數差／同向量、U24不混LGB、四base兩median、檔案/value SHA與conflict拒絕。初稿新增拒絕測例時，attempt2有括號SyntaxError，未成功import或真評分；失敗收據保留。最新body確實驗證reload/input/未score/protected等欄位，不只是測項名稱。

任何真評分都仍須驗主線完成的 seal 與 execution；來源審查及人工小陣列通過不等於完整真資料流程已跑通。禁止重跑已失敗输出目錄，若遇工程問題保留證據再另作恢復決定，不據新query結果改科學設定。

|證據|SHA-256|
|---|---|
|`src/round97_offset_evaluate.py`|`fbe0b0f841fce7f13afa6e499b6f88498ae061c816ab381fc89a5357eb87d0d8`|
|`src/round97_offset_experiment.py`|`00d927b81170b9b8acfab984bb72635f413513f62da42733b85e5873528914f1`|
|`reports/round97_plan.md`|`8372960e832bd66203374c942e0f7988e66522aedf358c894d892e82669a3a08`|
|`reports/round97_evaluator_selfcheck_attempt3.json`|`08c24fd18f12dca3652d71b07b67c1b640a9c98f13200628a417fda0f7e3ce47`|
|`reports/round97_evaluator_selfcheck_attempt2_import_failure.json`|`bd54084331311b82e6fee9650a7f25dbc573a925e95222c4c6098cdd4ec62fcf`|

[評分程式](../src/round97_offset_evaluate.py) · [attempt3合成](round97_evaluator_selfcheck_attempt3.json) · [保留syntax失敗](round97_evaluator_selfcheck_attempt2_import_failure.json) · [main前置](round97_preflight_review.md) · [固定計畫](round97_plan.md)
