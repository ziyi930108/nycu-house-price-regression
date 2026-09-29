# R103／R104：從誤差互補診斷到固定 base 目標

2026-09-29。R103 誤差互補診斷與 R104 兩教師／六學生、三期隔離評分均已完成。完整 T−v22 MAE 為 **+39,238.32／+11,007.42／-43,442.06 元**（development／confirmation／auxiliary）；兩項固定門檻未通過，**不採用、無新 CSV；唯一提交仍為 v22，Private 未知。**

本頁於發布前核驗，基底 cca69847befe8f123f19ff7351be0e23cc1ec7f2；實際發布身分以 Git main／SOURCE_MANIFEST 為準。Public 1,998,568.5706 元是使用者回報；本頁其餘分數均為離線回測。

## 先診斷，再固定一個假設

[R103 診斷報告](../evidence/reports/round103_report.md)以已保存預測做代數分解，沒有新模型或 bootstrap。最近窗 native 改善後，固定 base 的誤差抵銷減少；development 有 49,017 筆落在相同側外，spatial 的淨改善被 median 截住。median 也會阻擋退步，不能说它一律有害，更不能將事後分解當作因果。

[R104 計畫](../evidence/reports/round104_plan.md)只測一個假設：訓練 prior NN 時，把固定 area／LGB 分支納入損失，是否能改善完整回歸？兩組同合法 forward teacher 覆蓋列、145 欄、factor、原日期權重、初始化 bias、seed 42、30 epochs；不按 y 或誤差選列，不搜融合比例。

|組別|每批損失（價格先還原成百萬元）|
|---|---|
|C：同列、梯度尺度對照|0.375 × mean(w × abs(p−y))|
|T：固定 base 目標|mean(w × abs(0.375p + 0.375A + 0.25L − y))|

A／L 為較早 origin 的完整 forward 預測。新增 2020-08-11 origin 的 area NN／unit LGB；auxiliary、development、confirmation 學生依序使用 1／2／3 個老師期間，與原 eligible fit 列按 canonical 索引交集。學生不重建 prior／encoder／R/S，query 沿原同窗 A/L、R/S、H/U 與三支 median。所有八個 native fit、全 query 重載封存後才隔離評分。

## 完整模型與機制結果

單位：元；差值負數較好。本輪 C 不是 R100 或 R102 的舊對照。

|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T−v22 weighted|base T−C|
|---|---:|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,405,023.09|1,446,625.39|+39,238.32|+41,602.30|+39,110.20|+793,761.94|
|confirmation|1,383,228.41|1,393,681.62|1,394,235.82|+11,007.42|+554.21|+12,908.42|+121,238.08|
|auxiliary|2,309,352.36|2,296,532.20|2,265,910.30|-43,442.06|-30,621.90|-37,157.08|+22,466.61|

Primary 要求三窗完整 overall／weighted 改善、最近窗至少改善 0.5%，且無季度或至少 500 列固定群退步超過 5%；mechanism 要求 T 相對 C 的 base 與 median、兩種 MAE 三窗均改善。原生 NN 單獨改善不是這次 mechanism 的必要條件。

|門檻|條件鍵（對應完整 JSON）|通過|
|---|---|---|
|primary|all_three_overall_improve|否|
|primary|all_three_weighted_improve|否|
|primary|confirmation_at_least_half_percent|否|
|primary|no_quarter_or_large_group_deterioration_above_5pct|否|
|mechanism|base_all_three_overall_mae_improve|否|
|mechanism|base_all_three_weighted_mae_improve|否|
|mechanism|median_all_three_overall_mae_improve|否|
|mechanism|median_all_three_weighted_mae_improve|否|

T−C 的 base 三窗均退步，不能把只有最舊窗口完整改善當作研究成功。各層、分群／價帶、尾部貢獻與最強反證見[R104 摘要](../evidence/reports/round104_report.md)及[全部聚合 JSON](../evidence/outputs/round104/evaluation/results.json)。原 R102T 僅為固定背景，不額外搜模型。

## 配對區間、耗時與限制

保存全部 24 個 CI：8 全負、6 跨零、10 全正。T−v22／T−C 各三窗，project 與連續三個月 block，overall／weighted，固定 seed 20260925、每種 2,000 次。窗口反覆使用且只有單一 seed，區間不是 Private 勝率。

|步驟|秒數|包含關係|
|---|---:|---|
|R103 診斷 main|6.890|0 新 fit／bootstrap|
|兩教師 fit|114.235|已含於教師 main|
|教師 main|131.000|已含於教師五步 launcher|
|教師五步 launcher|151.375|凍結、準備、隔離 fit 標籤、2 fit|
|六學生 fit|58.721|已含於學生 main|
|學生 main|153.718|準備、6 fit、重載、封存|
|獨立 scorer|58.594|指標與 24 CI|
|聚合摘要|0.156|僅 JSON，不新評分|

以上是程序實測，不是聊天／開發工時；內含時間不能再相加。2023 靜態非價格地理晚於早期 origin，因此價格標籤與 learned state 雖按時間隔離，仍不屬完整 historical-as-of 驗證；教師與 query 模型分布也不同。沒有因果或 Private 最佳保證。

## 程式、證據與交接界線

[R103 helper](../research/src/round103_complementarity_math.py)／[driver](../research/src/round103_complementarity_diagnostics.py)；[教師準備](../research/src/round104_teacher_preparation.py)／[教師訓練](../research/src/round104_teacher_fit.py)；[學生準備](../research/src/round104_student_preparation.py)／[loss trainer](../research/src/round104_base_aware_train.py)／[main](../research/src/round104_experiment.py)／[組合](../research/src/round104_composition.py)／[隔離 scorer](../research/src/round104_evaluate.py)。

[有限結果審查](../evidence/reports/round104_results_review.md)及[下一方向盤點](../evidence/reports/round104_next_direction_inventory.md)保留反證；後者只有 source／report 盤點，尚無新計畫或 fit。

[本批精確複製清單](../evidence/outputs/research_snapshot_r104_20260929/copy_manifest.json)保留原碼、人工測例的各次來源／收據、計畫、模型 metadata、聚合結果與來源 SHA。不包原交易表、逐筆標籤／向量、模型、完整 encoder／ramp／category／prior states。部分原碼及報告連結仍依賴本機工作區；這是文件與程式快照，不是完整重訓備份或乾淨環境重現。

[R101／R102 歷史頁](RESEARCH_UPDATES_20260929_PRIOR_INPUT.md)中提出的診斷已完成，歷史頁原 bytes 保留。正式 v22 報告／PDF／簡報與歷史 source 不變；不新增提交，也不承諾下一個改良。
