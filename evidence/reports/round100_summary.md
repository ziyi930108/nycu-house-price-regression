# R100 時間前置先驗：完成摘要

2026-09-29。**固定探索門檻未通過；v22 唯一提交不變，Private 未知，沒有新 CSV 或自動採用。**

T 為時間前置且排除 held fold 的 prior；C 只使用相同 eligible 訓練列，prior 仍為原 R60 分組 OOF。兩臂各三次 NN fit，其餘分支、query prior、encoder 與 R/S 保留原同窗完整 fit 狀態。此摘要只读已保存聚合 JSON，不載逐筆資料或重算分數。

所有差值是前者減後者，負數較好；表內「整體／加權」均為新臺幣元，後者是固定 seen/new 組成評分，不是日期訓練 loss。

|窗口|v22 整體／加權|C 整體／加權|T 整體／加權|eligible／原 fit|
|---|---:|---:|---:|---:|
|development|1,407,387.07 / 1,399,717.74|1,405,993.40 / 1,398,339.06|1,402,289.26 / 1,394,768.95|240,260 / 240,316|
|confirmation|1,383,228.41 / 1,447,153.42|1,382,132.85 / 1,446,039.16|1,393,291.76 / 1,458,210.20|321,494 / 321,550|
|auxiliary|2,309,352.36 / 2,023,165.66|2,302,648.68 / 2,018,114.26|2,297,675.88 / 2,012,862.19|134,749 / 134,805|

|窗口|T−v22|T−C|C−v22|T−v24|
|---|---:|---:|---:|---:|
|development|-5,097.81 / -4,948.79|-3,704.14 / -3,570.11|-1,393.67 / -1,378.69|-5,051.46 / -4,926.97|
|confirmation|10,063.35 / 11,056.78|11,158.91 / 12,171.04|-1,095.56 / -1,114.26|20,095.84 / 22,121.32|
|auxiliary|-11,676.48 / -10,303.47|-4,972.80 / -5,252.08|-6,703.68 / -5,051.40|22,266.74 / 20,291.21|

T−C 才是相同 loss rows 下的先驗對照；T−v22 同時含暖身資料移除。C 不是可依結果另挑的候選，較低訓練 loss 或單一窗口改善也不代表 Private 改善。

## 原生 NN 到完整模型

每格為整體／加權差。native 是 prior NN，spatial 是完整 N 分支，median 才是與固定 H/U 合併的最終總價；各層改善不可互相代稱。

|窗口|層級|T−原模型|T−C|C−原模型|
|---|---|---:|---:|---:|
|development|native|157,352.17 / 159,759.09|142,803.30 / 145,176.04|14,548.87 / 14,583.05|
|development|base|9,276.15 / 10,234.27|12,799.31 / 13,646.93|-3,523.15 / -3,412.66|
|development|reference|-7,728.97 / -7,025.43|-6,735.91 / -6,052.39|-993.06 / -973.03|
|development|spatial|-7,347.23 / -6,645.53|-5,933.41 / -5,262.36|-1,413.82 / -1,383.17|
|development|median|-5,097.81 / -4,948.79|-3,704.14 / -3,570.11|-1,393.67 / -1,378.69|
|confirmation|native|340,774.56 / 346,986.26|335,801.82 / 340,778.96|4,972.74 / 6,207.30|
|confirmation|base|171,664.11 / 178,527.81|180,161.17 / 187,673.67|-8,497.06 / -9,145.86|
|confirmation|reference|74,567.83 / 79,380.78|81,191.54 / 86,335.75|-6,623.71 / -6,954.97|
|confirmation|spatial|75,242.92 / 80,228.75|81,390.36 / 86,673.17|-6,147.44 / -6,444.41|
|confirmation|median|10,063.35 / 11,056.78|11,158.91 / 12,171.04|-1,095.56 / -1,114.26|
|auxiliary|native|108,848.70 / 144,795.16|173,924.78 / 203,206.73|-65,076.08 / -58,411.57|
|auxiliary|base|-20,539.47 / -7,125.84|2,764.94 / 12,270.20|-23,304.40 / -19,396.04|
|auxiliary|reference|-15,336.21 / -6,242.60|7,055.18 / 11,671.88|-22,391.38 / -17,914.49|
|auxiliary|spatial|-14,490.04 / -5,816.90|7,925.28 / 12,278.66|-22,415.33 / -18,095.56|
|auxiliary|median|-11,676.48 / -10,303.47|-4,972.80 / -5,252.08|-6,703.68 / -5,051.40|

## 固定門檻與分群反證

|門檻|結果|
|---|---|
|all_three_overall_improve|False|
|all_three_weighted_improve|False|
|confirmation_at_least_half_percent|False|
|no_quarter_or_large_group_deterioration_above_5pct|False|

confirmation 整體改善比例：-0.7275%（固定需求至少 0.5%）。

|窗口|群|列數|T−v22|T−C|C−v22|
|---|---|---:|---:|---:|---:|
|development|new_nonmissing|52,253|-10,743.61|-8,782.44|-1,961.17|
|development|seen|28,981|5,081.61|5,452.06|-370.45|
|confirmation|new_nonmissing|76,096|16,682.97|17,903.15|-1,220.18|
|confirmation|seen|57,601|1,318.26|2,249.18|-930.92|
|auxiliary|new_nonmissing|90,424|-12,556.13|-4,793.87|-7,762.26|
|auxiliary|seen|15,087|-6,404.29|-6,045.20|-359.09|

以下列每窗、每個主要對照退步超過 5% 的固定季度或至少 500 列群數，並列最大相對退步群。完整清單不省略地保存在本摘要 JSON 的 comparisons，群可重疊，不能相加當獨立列數。

|窗口|比較|超限群數|最大相對退步群／列數|MAE 差／增幅|
|---|---|---:|---|---:|
|development|T-v22|0|無|—|
|development|T-C|0|無|—|
|development|C-v22|0|無|—|
|development|T-v24|1|cities/新竹市 / 1,250|536,396.86 / 13.449%|
|confirmation|T-v22|0|無|—|
|confirmation|T-C|0|無|—|
|confirmation|C-v22|0|無|—|
|confirmation|T-v24|2|cities/新竹市 / 2,131|148,381.26 / 6.145%|
|auxiliary|T-v22|1|cities/嘉義市 / 731|129,464.62 / 6.128%|
|auxiliary|T-C|1|cities/嘉義市 / 731|110,535.99 / 5.186%|
|auxiliary|C-v22|0|無|—|
|auxiliary|T-v24|3|cities/基隆市 / 1,577|731,837.40 / 27.620%|

## 價格帶正反證與億元群貢獻

|窗口|真價帶|列數|T−v22 帶內 MAE|T−C 帶內 MAE|對全窗貢獻 T−v22|對全窗貢獻 T−C|
|---|---|---:|---:|---:|---:|---:|
|development|<10m|22,139|1,409.03|3,771.89|384.01|1,027.97|
|development|10-20m|43,914|-5,429.90|-4,691.64|-2,935.33|-2,536.24|
|development|20-50m|14,197|-16,761.33|-16,284.59|-2,929.32|-2,846.00|
|development|50-100m|802|-10,246.92|1,351.37|-101.16|13.34|
|development|>=100m|182|216,029.49|284,224.87|484.00|636.79|
|confirmation|<10m|25,585|-7,901.36|-8,758.66|-1,512.05|-1,676.11|
|confirmation|10-20m|74,109|1,068.68|3,249.66|592.38|1,801.30|
|confirmation|20-50m|31,959|36,685.57|38,788.41|8,769.34|9,272.00|
|confirmation|50-100m|1,725|119,260.40|103,136.36|1,538.73|1,330.70|
|confirmation|>=100m|319|282,882.05|180,643.26|674.95|431.01|
|auxiliary|<10m|41,132|4,646.17|5,184.65|1,811.24|2,021.16|
|auxiliary|10-20m|47,793|-5,524.98|-3,632.36|-2,502.63|-1,645.34|
|auxiliary|20-50m|15,033|-53,517.77|-32,950.94|-7,625.11|-4,694.79|
|auxiliary|50-100m|1,382|-100,938.04|-76,287.40|-1,322.10|-999.22|
|auxiliary|>=100m|171|-1,257,420.05|213,115.35|-2,037.88|345.39|

各帶對全窗貢獻 = 保存群列數 × 帶內 MAE 差 ÷ 全窗列數，五帶加總核對 overall 差至 0.000001 元。這是事後聚合診斷，不是新門檻、因果證明或依未知真價推論分流。

development：億元群 182 筆，對全窗 T−v22 貢獻 484.00 元，其餘價帶合計 -5,581.81 元；T−C 的億元貢獻為 636.79 元。
confirmation：億元群 319 筆，對全窗 T−v22 貢獻 674.95 元，其餘價帶合計 9,388.40 元；T−C 的億元貢獻為 431.01 元。
auxiliary：億元群 171 筆，對全窗 T−v22 貢獻 -2,037.88 元，其餘價帶合計 -9,638.60 元；T−C 的億元貢獻為 345.39 元。

## 全部 24 個保存配對區間

每族固定 2,000 次；只轉錄保存端點與有效次數，沒有重新 bootstrap。負向表示整段小於零，正向表示整段大於零，跨零保留不確定性。

|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|
|---|---|---|---:|---|---:|---|
|development|T−v22|project/overall|-5,097.81|-11,859.81 ～ 1,407.97|2000 / 0|crosses_zero|
|development|T−v22|project/weighted|-4,948.79|-11,719.15 ～ 1,436.60|2000 / 0|crosses_zero|
|development|T−v22|month/overall|-5,097.81|-13,388.80 ～ 1,067.43|2000 / 0|crosses_zero|
|development|T−v22|month/weighted|-4,948.79|-17,763.49 ～ 1,321.39|2000 / 0|crosses_zero|
|development|T−C|project/overall|-3,704.14|-10,845.49 ～ 3,002.64|2000 / 0|crosses_zero|
|development|T−C|project/weighted|-3,570.11|-10,710.65 ～ 3,039.17|2000 / 0|crosses_zero|
|development|T−C|month/overall|-3,704.14|-12,204.75 ～ 2,336.95|2000 / 0|crosses_zero|
|development|T−C|month/weighted|-3,570.11|-16,429.55 ～ 2,074.68|2000 / 0|crosses_zero|
|confirmation|T−v22|project/overall|10,063.35|-962.01 ～ 21,018.28|2000 / 0|crosses_zero|
|confirmation|T−v22|project/weighted|11,056.78|-1,204.40 ～ 23,241.97|2000 / 0|crosses_zero|
|confirmation|T−v22|month/overall|10,063.35|2,720.94 ～ 17,887.70|2000 / 0|positive|
|confirmation|T−v22|month/weighted|11,056.78|2,572.46 ～ 18,036.46|2000 / 0|positive|
|confirmation|T−C|project/overall|11,158.91|539.32 ～ 21,378.45|2000 / 0|positive|
|confirmation|T−C|project/weighted|12,171.04|507.17 ～ 23,457.43|2000 / 0|positive|
|confirmation|T−C|month/overall|11,158.91|5,358.15 ～ 17,789.98|2000 / 0|positive|
|confirmation|T−C|month/weighted|12,171.04|5,764.28 ～ 19,172.68|2000 / 0|positive|
|auxiliary|T−v22|project/overall|-11,676.48|-18,308.52 ～ -5,571.88|2000 / 0|negative|
|auxiliary|T−v22|project/weighted|-10,303.47|-16,160.55 ～ -4,835.10|2000 / 0|negative|
|auxiliary|T−v22|month/overall|-11,676.48|-18,130.88 ～ -4,436.92|2000 / 0|negative|
|auxiliary|T−v22|month/weighted|-10,303.47|-18,208.80 ～ -3,486.55|2000 / 0|negative|
|auxiliary|T−C|project/overall|-4,972.80|-12,700.51 ～ 3,116.05|2000 / 0|crosses_zero|
|auxiliary|T−C|project/weighted|-5,252.08|-12,106.14 ～ 1,811.26|2000 / 0|crosses_zero|
|auxiliary|T−C|month/overall|-4,972.80|-9,820.75 ～ 335.19|2000 / 0|crosses_zero|
|auxiliary|T−C|month/weighted|-5,252.08|-11,460.09 ～ 187.56|2000 / 0|crosses_zero|

全部方向計數：{"negative": 4, "crosses_zero": 14, "positive": 6, "unavailable": 0}。兩個對照的所有區間皆保留，不擇優呈現；它們不是 Private 勝率。

## 工程修正時間線

真訓練前已固定 45 份來源／計畫與 11 份歷史及前置 JSON。訓練進行中、真評分尚未開始時，來源審查發現 scorer 把 pure-X 專用計畫 d536… 的身分誤要求為科學計畫 bbb…；後者實際另綁於 pure-X request。保留原 a86… scorer、request 與來源副本，只新增 _v2 窄工程修正及 42 項人工檢查，兩版評估參數完全相同。原錯誤版未執行真評分，這不是模型 fit 失敗、模型改動或看到 query 分數後改設計。
[修正收據](../outputs/round100_evaluator_v2_revision/revision.json)與[實際 metadata 核驗](../outputs/round100_evaluator_v2_revision/actual_metadata_check.json)保留完整細節；本次結果由另行固定的 v2 request 產生。

## 實測時間與限制

|窗口|C NN fit 秒|T NN fit 秒|prior 準備秒|stage 全流程秒|prefix states|
|---|---:|---:|---:|---:|---:|
|development|21.156|19.125|56.719|138.875|235|
|confirmation|30.094|26.953|70.282|169.453|255|
|auxiliary|10.250|10.625|33.875|79.719|215|

六次 NN fit 共 **118.203 秒**；prior 準備共 **160.876 秒**，兩者均已包含主流程 **391.938 秒**，不可再相加。獨立評分與固定抽樣另 **68.985 秒**。沒有新樹模型 fit；開發、聊天與等待時間未量成訓練工時。
三個年度窗口已多輪重用，單 seed 控制不能消除隨機訓練變異。只有 fit prior 逐列時間前置；encoder、query prior、R/S 仍取原完整 outer fit，不能宣稱全部輸入皆為逐列 as-of。bootstrap 不涵蓋重訓、累積選型或未來市場移轉。完整正反證據保留，未由此改 v22 唯一提交或產生新 CSV，Private 仍未知。

[固定計畫](round100_plan.md) · [原評估結果](../outputs/round100/evaluation/results.json) · [聚合摘要](../outputs/round100_summary/summary.json) · [摘要程式](../src/summarize_round100.py)
