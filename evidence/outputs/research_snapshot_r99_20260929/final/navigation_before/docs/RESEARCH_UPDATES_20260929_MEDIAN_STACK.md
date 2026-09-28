# 把中位數納入可選基線：R99 時間前推研究

2026-09-29（Asia/Taipei）。R99 三次 CPU meta-fit、全部 query 保存重載與獨立評分已完成，未通過事前研究門檻。完成後稽核的最終收據尚待加入本頁。**唯一提交保持 v22**，Public MAE 1,998,568.5706 元，Private 未知；沒有新 CSV 或 Kaggle 操作，正式報告與簡報保持。

## 為何多加一欄 median

前一輪只用 N／H／U 的三個固定全域權重，其可行集合未必包含「每筆都取中位數」的原 v22。這次沿用所有原生模型與三支完成成交參考／空間修正的總價，增加確定性的第四欄 M=median(N,H,U)，再只用較早 forward 預測及其成交標籤學權重。

~~~text
R99 = (((N×wN + H×wH) + U×wU) + M×wM)
M = median(N,H,U)
四個權重非負、和為 1；無截距
以較早 meta 資料最小化每列等權的新臺幣總價 MAE
~~~

M 權重 1 與舊 R98 三權重加上 M 權重 0，現在都是合法解。因此精確求解時，同一批 meta 資料的最優目標不應較這兩個基線差；工程檢查容差為 0.10 元。**這只保證同一 meta 訓練目標，並不保證更晚 query、固定群加權 MAE 或未知 Private 較好。** 本輪是看到 R98 結果後提出的適應性研究，不是新盲測。

## 完整結果與兩個對照

MAE 單位為元，越低越好；差值以 R99 減去對照。

|query 窗口|v22 MAE|R99 MAE|R99−v22|加權差|
|---|---:|---:|---:|---:|
|2022-08-11～2023-08-10|1,407,387.07|1,417,836.89|+10,449.82|+10,336.51|
|2023-02-11～2024-02-10|1,336,410.67|1,330,129.14|−6,281.53|−7,487.25|
|2023-08-11～2024-08-10|1,383,228.41|1,386,049.92|+2,821.52|+2,732.32|

development／confirmation 仍高於 v22，offset 較低。development 的基隆市與部分月份超過固定 5% 退步門檻；offset 的嘉義縣 1,532 列退 6.080%。confirmation 本次沒有固定大群超過 5% 退步，但整體及加權 MAE 仍未低於 v22。因此三個窗口均未通過完整探索門檻，不升級為提交候選。

|窗口|R98 MAE|R99−R98 整體差|R99−R98 加權差|
|---|---:|---:|---:|
|development|1,416,185.71|+1,651.18|+1,608.18|
|offset|1,332,155.65|−2,026.51|−1,679.39|
|confirmation|1,388,419.07|−2,369.15|−2,270.02|

相對舊 R98，兩個窗口確實改善，development 仍退；沒有相對 R98 的季度或至少 500 列群超過 5% 退步。這些是事前列出的附加點比較，沒有另抽 R98 bootstrap，也不是取代主要 v22 門檻。

六個保存向量的整體 MAE 均有記錄，避免把單支與最終組合混在一起：

|窗口|N|H|U|v22 median|R98|R99|
|---|---:|---:|---:|---:|---:|---:|
|development|1,469,402.11|1,485,512.75|1,509,405.16|1,407,387.07|1,416,185.71|1,417,836.89|
|offset|1,455,654.40|1,372,629.24|1,394,719.31|1,336,410.67|1,332,155.65|1,330,129.14|
|confirmation|1,403,580.19|1,486,906.73|1,462,277.22|1,383,228.41|1,388,419.07|1,386,049.92|

對 v22 的建案群組／三個月循環區塊、整體／加權共 **12 個 95% 配對區間全部跨零**。全部結果、兩個對照的點差、分群與六個向量見[評分報告](../evidence/outputs/round99/evaluation/report.md)及[聚合 JSON](../evidence/outputs/round99/evaluation/results.json)。門檻未過不等於全面顯著更差，bootstrap 不是 Private 勝率。

## 保留改善與退步的證據

offset 相對 v22 的新案改善約 13,006 元、億元群 270 列改善約 228 萬元；但 seen 退約 2,065 元，低於千萬元價帶也退。confirmation 的 1,000–5,000 萬元兩帶改善，但億元群 319 列退約 403 萬元。development 除第一季外，其他季度與全部五個價帶均退。小群即使未達 500 列通報條件仍完整呈現，不靠省略尾部讓結果較好看。

相對 R98，confirmation 的 seen／新案都較低，億元群改善約 69 萬元；offset 的 seen／新案也較低，但其億元群反而較 R98 高約 9.8 萬元。兩個對照不能混成同一個改善結論。真價帶只用於事後診斷，從未用未知真價決定推論路徑。

## 資料、權重與防洩漏

重用 R98 已封存的原始 11-key stage 資料，仍核它的原 R98 plan 與 manifest 身分，不把舊準備檔改綁新計畫。原三欄、canonical／raw 索引、列序、日期及 meta 標籤皆不改；只在記憶體追加第四欄，保存四欄的 dtype／shape／值 SHA。

|query stage|較早 meta 來源|meta／query 列數|N／H／U／M 權重|
|---|---|---:|---|
|development|2021-08-11～2022-08-10 全期|105,511／81,234|0.181672／0.269223／0.473599／0.075507|
|offset|上述全期＋development 至 2023-02-10 的 31,801 列|137,312／107,169|0.200130／0.269518／0.406423／0.123929|
|confirmation|上述全期＋development 全期|186,745／133,697|0.278766／0.251445／0.319509／0.150280|

三期獨立程序只讀自身合法 meta_y；meta 日期嚴格早於自己的 query cutoff，與自己的 query 索引不重疊。較早 query 成為後期 meta 標籤是合法角色轉換，不能稱整批從未見過這些交易價格。原價格整表 pickle 只在前批隔離擷取器與本輪封存後 scorer 的既定步驟中反序列化；本輪沒有重新抽 meta 資料，也沒有讓 driver 取得 query_y。

三期全 query 預測、模型 JSON 與原 median 均保存重載一致並共同封存後，另一 scorer 才評分。舊 R98 的 P3／y／row IDs 身分相同，原權重合法，同 meta primal 比較亦通過：

|stage|原 median 的 meta MAE|R98 的 meta MAE|R99 的 meta MAE|
|---|---:|---:|---:|
|development|2,309,352.36|2,283,828.80|2,283,439.97|
|offset|2,069,966.02|2,052,521.26|2,051,932.05|
|confirmation|1,916,997.82|1,902,764.29|1,902,161.60|

meta 三期較好，query 卻有得有失，正好說明訓練目標改善不能直接當成泛化改善。這些 meta MAE 不是新的 query 分數。

## 固定求解與工程時序

沿用 SciPy 1.18.1／HiGHS 1.12.0 的等價 dual LP，只把價格約束由三個擴成四個。無截距、正則化、日期衰減、比例網格、median 權重下限或第二目標；權重不 clip／normalize，沒有 fallback／自動重試。固定分項相加、未取整 float64 輸出。helper 證書與舊 R98 meta 可行解檢查均保留，但不作未知 query 保證。

本輪新增 **3 次 CPU meta-fit、0 次原生樹／NN fit、0 GPU**。fit 呼叫合計 **2.781 秒**，包含於 main **6.782 秒**；獨立 scorer **30.156 秒**。時間不能重複相加，也不是整段聊天工時。沒有全量 test 模型或新 CSV。

前置時修正原 R98 stage_manifest 真鍵名；交叉讀碼又發現移動保存呼叫時因 CRLF 處理漏刪舊呼叫，會對同一新 model 目錄保存兩次。問題在任何真 fit 前修正，只保留 fit 後、額外可行性檢查前的一次保存，以保留工程拒絕時的數值證據。舊 28 項 driver 自查未覆蓋這一點，原碼／收據保持；最終 29 項另存，明確加入單次保存檢查。修補命令本身的斷言與不存在檢查腳本的錯誤也保存，不算模型實驗失敗。

helper 的 53 項、scorer 的 47 項人工檢查與 source／parameters 均匹配。driver 前置文件由 driver 作者撰寫，明確是作者自查；helper 作者另交叉讀碼，scorer 由另一代理編寫。[driver 前置核驗](../evidence/reports/round99_driver_preflight_review.md)及[修正紀錄](../evidence/outputs/round99_driver_prefreeze/duplicate_save_revision.json)保留實際時序。

## 程式與交接範圍

- [固定計畫](../evidence/reports/round99_plan.md)、[contract](../research/src/round99_contract.py)、[四欄 helper](../research/src/round99_l1_stacker.py)、[stage driver](../research/src/round99_temporal_stack.py)、[scorer](../research/src/round99_temporal_evaluate.py)。
- [helper 自查](../evidence/reports/round99_helper_selfcheck.json)、[driver 最終自查](../evidence/reports/round99_driver_selfcheck_v2.json)、[scorer 自查](../evidence/reports/round99_evaluator_selfcheck.json)。
- [封存 manifest](../evidence/outputs/round99/sealed_manifest.json)、[main execution](../evidence/outputs/round99/execution.json)、[scorer execution](../evidence/outputs/round99/evaluation/execution.json)。
- [前置複製清單](../evidence/outputs/research_snapshot_r99_20260929/preparation/copy_manifest.json)不可變；完成清單與導覽修改前 bytes 另存。

offset 與另外兩窗重疊，歷史資料已多輪參與選型；仍缺 auxiliary 以前的 forward block，不能稱原三個 8 月窗口的完整採用驗證。這次結果不足以替換 v22；沒有新增未知 Private 成績，也沒有背景自動訓練承諾。

本頁內容於發布前核驗，基底為 77ec5b46a6b28ecef4cc2f64356083fa31c4ec30；**當前發布身分以此儲存庫 Git main 歷史與 SOURCE_MANIFEST 為準**，根工作區另存發布收據。原 379 支研究來源清單、正式 REPORT／PDF／PPTX、v22 設定均保持。快照不含原交易表、標籤、模型 state、solver／bootstrap 陣列或逐筆向量，不是完整可重訓備份。
