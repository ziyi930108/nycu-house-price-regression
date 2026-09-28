# R101／R102：把時間價格先驗明確加入回歸輸入

2026-09-29。R101 診斷與 R102 六次 NN／三期隔離評分均已完成。T 相對 v22 的 development／confirmation／auxiliary MAE 差為 **+934.33／+7,219.36／-9,619.90 元**。Primary gate 未通過，mechanism gate 未通過；**研究門檻未全通過，不採用；唯一提交仍是 v22，Private 未知。**

本頁於發布前核驗，基底 4217cc53e54c917a27a57a48f994eb16ea05b066；當前發布身分以 Git main 歷史及 SOURCE_MANIFEST 為準，根工作區另存发布收據。v22 Public MAE 1,998,568.5706 元是使用者回報；下表是本機回測。

## 診斷與固定假設

[R101 診斷](../evidence/reports/round101_report.md)只讀保存的 fit／無真價 query 分布，檢查 prior、ratio、日期權重及有效 ratio loss 權重。沒有新 prior、模型、MAE 或 bootstrap；不能把分布差異當成因果。R100 結束時提出的這項診斷現在已完成。

[R102 固定計畫](../evidence/reports/round102_plan.md)檢查：當時間先驗已用於還原總價時，再讓 NN 看見先驗大小，是否改善回歸？兩組均重用同一期 R100T 的 eligible loss 列、factor、y、日期權重與原 144 欄；七個非 numeric array 位元保持，不再建立 prior。

|組別|第 145 欄|作用|
|---|---|---|
|C：同寬度對照|float32 正零|控制新增一欄及相同架構，不是 R100 舊 C|
|T：研究候選|log(單價 prior)，以 eligible fit 的 float64 平均／標準差標準化，再轉 float32|query 用原 full-fit prior 與同一組 fit 參數，無 clipping|

兩組共用原 R63 trainer、seed 42、30 epochs、batch 2048 及相同初始化。只有新欄不同；完整模型仍保留原 area NN／LGB、R/S 與 H/U 分支，再取三支 median。六個 NN 與所有 query 預測重載封存後，隔離 scorer 才讀 query 真價。

## 完整模型與原生 NN 結果

單位為元，負差較好。C 僅作機制對照，不依結果另選提交。

|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T−v22 加權差|native T−C|
|---|---:|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,404,025.68|1,408,321.41|+934.33|+4,295.72|+944.33|-304,378.87|
|confirmation|1,383,228.41|1,391,350.42|1,390,447.76|+7,219.36|-902.66|+8,066.03|-386,280.18|
|auxiliary|2,309,352.36|2,292,749.46|2,299,732.46|-9,619.90|+6,982.99|-8,254.12|-298,184.14|

|門檻|固定條件|結果|
|---|---|---|
|Primary|三期完整 overall 均改善|未通過|
|Primary|三期完整 weighted 均改善|未通過|
|Primary|confirmation 至少改善 0.5%|未通過|
|Primary|無季度或至少 500 列固定群退步超過 5%|未通過|
|Mechanism|三期完整 T overall 均優於 C|未通過|
|Mechanism|三期完整 T weighted 均優於 C|未通過|
|Mechanism|三期原生 T 均優於 C|通過|

觸及固定群／季度 >5% 限制的列共 2 條；全部群別、尾部、T−v24 及 C−v22 點估計保留於[完整評分](../evidence/outputs/round102/evaluation/report.md)與[聚合摘要](../evidence/reports/round102_report.md)。兩項 gate 分開判斷，不能用後處理改善代替原生 NN 改善，也不能把層間差加總成因果貢獻。

## 配對區間與限制

固定 seed 20260925、每種 2,000 次 project／連續三個月 block 配對，共 24 CI：3 全負、17 跨零、4 全正。

|窗口|比較|全負|跨零|全正|
|---|---|---:|---:|---:|
|development|T−v22|0|4|0|
|development|T−C|0|4|0|
|confirmation|T−v22|0|2|2|
|confirmation|T−C|0|4|0|
|auxiliary|T−v22|3|1|0|
|auxiliary|T−C|0|2|2|

年度窗口反覆使用且只有一個 seed；這是條件於保存預測的區間，不是新的盲測或 Private 勝率。C−v22、T−v24 沒有另外新增 bootstrap。原 R100T 與 R100C 的 native 結果僅是歷史背景，不能混成本輪 Czero。

## 實測耗時與來源

|步驟|秒數|範圍|
|---|---:|---|
|R101 診斷 main|64.359|0 新模型／MAE|
|R102 六次 NN fit|107.155|已包含於 main，不相加|
|R102 main|189.891|輸入、六 NN、重載與封存|
|R102 scorer|65.547|來源核驗、指標與 24 CI|
|R102 聚合摘要|0.141|只整理 JSON，沒有新評分|

[R101 原碼](../research/src/round101_prior_diagnostics.py)／[統計 helper](../research/src/round101_statistics.py)；[R102 driver](../research/src/round102_prior_input_experiment.py)／[隔離 scorer](../research/src/round102_prior_input_evaluate.py)／[scorer 契約](../evidence/reports/round102_evaluation_contract.md)／[實際 metadata 前置核對](../evidence/reports/round102_evaluator_metadata_preflight.json)／[完整聚合結果](../evidence/outputs/round102/evaluation/results.json)。

[保存結果有限核驗](../evidence/reports/round102_results_review.md)另列實際範圍與作者角色，不將保存證據核對冒稱完全獨立重訓。

## 交接與下一步界線

[本批精確複製清單](../evidence/outputs/research_snapshot_r102_20260929/copy_manifest.json)保留來源 bytes／SHA。本包不含原交易表、標籤、NPY／NPZ、模型、完整 prefix／price-prior state 或逐筆向量；原 source/report 的部分連結仍依賴本機工作區。這是文件與程式快照，並非從零可重訓備份或乾淨環境重現。

[有限下一方向](../evidence/reports/round102_next_direction.md)只提出先另固定誤差互補與 median 路由診斷，尚未執行。原生 NN 三窗改善，但 confirmation 在固定 base 融合已翻負、development 到 median 才翻負；不能直接歸咎 R/S，也不能用這些 query 搜最佳權重。尚無新模型計畫、係數、篩選規則或 CSV。

正式 v22 REPORT／PDF／PPTX、379 支歷史來源、既有 CSV 與未知 Private 維持。本頁不新增下一個候選、調參或訓練承諾；後續研究須另有固定假設与來源證據。[R100 歷史頁](RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)保留當時結果及反證。
