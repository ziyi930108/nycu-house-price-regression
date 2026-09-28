# R100：時間前置且排除同建案折的價格先驗

2026-09-29。六次 NN 訓練、三期完整預測、獨立評分與保存結果稽核均已完成。**T 未通過原固定研究門檻，不採用；v22 仍是唯一提交，Private 未知。** 使用者回報的 v22 Public MAE 為 1,998,568.5706 元，以下是本機年度回測，不能混用。

本頁於發布前核驗，基底為 c34842980f8abc4bcc15dc9d5940fe2d50ac118f；當前發布身分以 Git main 歷史與 SOURCE_MANIFEST 為準，根工作區另存發布收據。[上一輪 R99](RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)及其失敗反證保持。

## 為什麼做這個比較

原先的價格先驗採建案分組五折 OOF：每列的先驗不使用同一 held fold，但可使用外層 fit 內較晚日期。這**不是外層 query 洩漏**。本輪的單一假設是：若 fit 列的先驗也只用其日曆季以前交易，能否改善較晚房價的回歸？[固定計畫](../evidence/reports/round100_plan.md)在訓練前記錄兩組、日期邊界、固定配置與採用限制。

對每一日曆季，參考資料須同時滿足有效日期、日期嚴格早於季首、排除該列原建案 held fold。五個 held fold 各至少有 30 筆參考時，整季才納入 loss。無效日期不納入；本次三期實際均為零筆。沿用原完整 fit 的五折群身分，不在裁切後重分折。

|訓練組|價格先驗|其他條件|
|---|---|---|
|C：機制對照|原 R60 group-OOF factor，僅取相同 eligible 列|與 T 相同 num／cat／y／原 float32 日期權重，不重新正規化|
|T：研究候選|同一 loss 列改為日曆季前置且排除同折的 factor|原 R63 trainer、seed 42、30 epochs、batch 2048 及架構保持|

development／confirmation／auxiliary 的 loss 列分別為 240,260／321,494／134,749，每期比原 fit 少 56 筆 warmup。採相同樣本對照，避免把刪除早期樣本與先驗時間方向混在一起。兩組仍使用原完整 outer-fit encoder、完整 fit 的 query prior 與同窗 reference／spatial；**並非每一個輸入都改成逐列當時可知**。沒有重新搜尋季長、比例或 seed。

每組只替換先驗倍率 NN；保留原面積 NN、LightGBM 與另外兩個完整分支。新 NN 分支 base 為 .75×(.5×areaNN＋.5×priorNN)＋.25×LGB，再套原 reference／spatial，最後與 H／U 取 median。三期共六個 NN，0 個新樹模型。全部 query 預測及模型重載封存後，獨立 scorer 才讀 query 真價。[來源與實測摘要](../evidence/reports/round100_summary.md)保留完整細節。

## 完整模型結果

單位為元；差值越負越好。C 是機制對照，不能在看完結果後直接升格為另一個提交候選。

|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T−v22 加權差|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,405,993.40|1,402,289.26|−5,097.81|−3,704.14|−4,948.79|
|confirmation|1,383,228.41|1,382,132.85|1,393,291.76|+10,063.35|+11,158.91|+11,056.78|
|auxiliary|2,309,352.36|2,302,648.68|2,297,675.88|−11,676.48|−4,972.80|−10,303.47|

最近 confirmation 整體退步 0.7275%，未達預先要求至少改善 0.5%。三期 overall／weighted 同時改善的條件亦未過；auxiliary 的嘉義市 731 筆退步 129,464.62 元（6.1285%），觸及固定大群 >5% 限制。T 相對 v24 的三期差為 −5,051.46／+20,095.84／+22,266.74 元，完整反證不刪除。[正式評分報告](../evidence/outputs/round100/evaluation/report.md)與[聚合結果 JSON](../evidence/outputs/round100/evaluation/results.json)為數值依據。

兩個主要比較 T−v22、T−C，各做建案與連續三個月 block 的 2,000 次配對重抽樣，固定 seed。共 24 個區間：4 個全負、14 個跨零、6 個全正。T−v22 在 auxiliary 的四個區間全負；confirmation 的兩個月份區間全正。T−C 在 confirmation 四個區間全正；其餘八個跨零。這支持收益隨年份改變，不能轉述成 Private 勝率，也不能說所有群都無效。

## 有利結果與反證一起看

- T 的原生 prior NN 三期都比原 NN 差，MAE 增加約 157,352／340,775／108,849 元；完整 median 卻在兩期改善。層間差不能直接相加成因果貢獻。
- 新建案 T−v22 三期為 −10,743.61／+16,682.97／−12,556.13 元；最近窗仍退。已見建案則為 +5,081.61／+1,318.26／−6,404.29 元。
- confirmation 的 2,000–5,000 萬價帶對總差貢獻 +8,769.34 元；1 億以上只貢獻 +674.95 元，退步並非少數豪宅造成。
- auxiliary 的 1 億以上貢獻 −2,037.88 元，其餘價格合計 −9,638.60 元；改善也不限於極端高價。
- C 比 v22 的三期點估計都略低，仍只是同樣本機制對照。不能由此斷言少 56 筆本身有效，也不能事後把 C 當作已通過完整採用審查的新版本。

以上為完成後描述，不依 query 真價挑選補價群。年度窗口反覆使用有選型偏差，沒有新的 test 真價或新的盲測。

## 實測成本與評分前工程修正

|步驟|實測秒數|計時範圍|
|---|---:|---|
|六次 NN fit|118.203|已包含於 main|
|三期 temporal prior 生成|160.876|已包含於 main|
|main|391.938|完整前置、六 NN、重載與封存；不能再加上前兩列|
|独立 scorer v2|68.985|main 完成後的來源核對、評分及配對區間|
|保存結果稽核|77.078|內部核驗；外層程序另計 80.109 秒|
|聚合摘要|0.172|只整理已保存 JSON，沒有新評分或圖表|

原 scorer 的前置草稿曾把純 X 稽核順序要求成模型順序，真評分前修正並保留 36→38 項成功合成檢查與來源。模型真訓練期間又發現純 X 稽核計畫 SHA 與科學計畫 SHA 應分別驗證：原碼保持，另建 scorer v2。新契約沿純 X result→request→audit plan 與 scientific plan 核對；31 個其他函數 AST 保持，MAE、bootstrap、gate、公式均未改。

[v2 契約](../evidence/reports/round100_evaluation_contract_v2.md)、[窄差異](../evidence/outputs/round100_evaluator_v2_revision/engineering_diff.patch)、[真實 metadata 檢查](../evidence/outputs/round100_evaluator_v2_revision/actual_metadata_check.json)與[修訂收據](../evidence/outputs/round100_evaluator_v2_revision/final_receipt.json)完整保留。v2 42 項人工檢查成功，真 metadata 前置核對通過後才評分。這是評分前工程契約修正，沒有真實模型失敗／重試，也沒有利用 query 結果改科學計畫。

[完成稽核](../evidence/reports/round100_independent_review.md)核對 1,827 個唯一 SHA、9 個保護檔、705 個完整 prefix state 與 24 個既存區間；不重新載標籤、求 prior、評分或 bootstrap。稽核者亦參與純 X 政策／稽核且審閱 helper；helper、driver、scorer 由其他作者負責。這是跨作者保存證據核驗，不稱完全獨立重作。

## 程式與交接界線

|入口|內容|
|---|---|
|[契約](../research/src/round100_contract.py)／[計畫](../evidence/reports/round100_plan.md)|固定兩組、資料讀取與評分參數|
|[fit 標籤擷取](../research/src/round100_prepare_fit_labels.py)／[純 X 稽核](../research/src/round100_temporal_prior_x_audit.py)|canonical／raw mapping、日期支持、只輸出各期 fit 標籤|
|[prior helper](../research/src/round100_temporal_prior.py)／[主 driver](../research/src/round100_temporal_prior_experiment.py)|季首與 held-fold 排除、C／T 串行、全向量封存|
|[原 scorer](../research/src/round100_temporal_prior_evaluate.py)／[scorer v2](../research/src/round100_temporal_prior_evaluate_v2.py)|原稿與修正版分開保留，實際評分使用 v2|
|[保存結果稽核](../research/src/round100_independent_review.py)／[摘要](../research/src/summarize_round100.py)|既存證據核對與聚合整理|
|[前置複製清單](../evidence/outputs/research_snapshot_r100_20260929/preparation/copy_manifest.json)／[完成複製清單](../evidence/outputs/research_snapshot_r100_20260929/final/copy_manifest.json)|來源原 bytes 與 SHA，不回寫歷史收據|

本包不含交易表、fit 標籤、逐筆向量、NN 模型、NPZ、完整 prefix state 或含各 prefix 價格的 temporal_prior_evidence。705 個狀態及價格證據僅在原工作區，身分由 stage seal 的 artifact SHA 保留。因此原來源報告部分相對連結仍依賴原工作區，文件快照不等於完整可重訓備份或乾淨環境重現。

[有限下一方向](../evidence/reports/round100_next_direction_inventory.md)只提出先另固定 fit 側的 old/new prior 比值、支持層級、ratio／bias 分布診斷；尚未執行、沒有新模型計畫，也不再把 temporal prior 當成未做過的方法。不因本頁自行掃季長、seed 或融合比例。正式 v22 REPORT／PDF／PPTX、379 支歷史來源與既有提交保持。
