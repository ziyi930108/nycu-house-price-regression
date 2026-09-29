# R105：兩個 prior NN 的固定原生等權平均

Primary **False**；相對 R102T 端點補充 **False**。唯一提交 v22 不變，Private 未知，無新 CSV。

候選 T=.5P+.5Q；P 為 v22 原 prior NN，Q 為 R102T。**本輪 C 是完整 R102T 端點**，不是 R102zeroC 或 R104 loss control。平均只在原生 TWD 層，再沿原 base／reference／spatial／median。無新 fit，兩端五層均重播後才評分。
只讀保存 aggregate JSON，未載逐筆陣列／真價／模型或重算 MAE／bootstrap。金額為元，每格為整體／固定加權；差值負數較好。

|窗口|v22 整體／加權|C=R102T|T|T−v22|T−C|
|---|---:|---:|---:|---:|---:|
|development|1,407,387.07 / 1,399,717.74|1,408,321.41 / 1,400,662.07|1,407,069.26 / 1,399,419.69|-317.81 / -298.05|-1,252.15 / -1,242.38|
|confirmation|1,383,228.41 / 1,447,153.42|1,390,447.76 / 1,455,219.45|1,385,239.10 / 1,449,413.21|2,010.70 / 2,259.80|-5,208.66 / -5,806.24|
|auxiliary|2,309,352.36 / 2,023,165.66|2,299,732.46 / 2,014,911.54|2,303,291.69 / 2,018,079.02|-6,060.66 / -5,086.64|3,559.24 / 3,167.48|

## 固定門檻

Primary 與相對 C 的完整三窗補充分開呈現，不要求 native 勝兩端；補充不是因果機制證明。保存的完整 gate 結果如下：

```json
{
  "Private_score": null,
  "automatic_adoption": false,
  "both_research_gates_passed": false,
  "new_CSV": false,
  "primary": {
    "Private_score": null,
    "adoption_research_gate_passed": false,
    "automatic_adoption": false,
    "components": {
      "all_three_overall_improve": false,
      "all_three_weighted_improve": false,
      "confirmation_at_least_half_percent": false,
      "no_quarter_or_large_group_deterioration_above_5pct": true
    },
    "confirmation_fractional_improvement": -0.001453625515603567,
    "new_CSV": false,
    "selected_submission": "v22"
  },
  "selected_submission": "v22",
  "supplementary_endpoint": {
    "causal_identification_claimed": false,
    "components": {
      "all_three_final_overall_mae_improve": false,
      "all_three_final_weighted_mae_improve": false
    },
    "control": "saved complete R102T",
    "native_improvement_required": false,
    "supplementary_endpoint_gate_passed": false
  }
}
```

兩項全過也不自動替換使用者選定的 v22；未通過則保留反證，不依結果追加 alpha 或子群候選。

## 五層結果

native 為 prior NN；spatial 是 N 分支，median 才是完整模型。層間 MAE 差不能相加為因果貢獻。

|窗口|層級|v22 整體／加權|C 整體／加權|T 整體／加權|T−v22|T−C|
|---|---|---:|---:|---:|---:|---:|
|development|native|1,834,776.38 / 1,826,042.86|1,679,443.61 / 1,671,066.06|1,673,062.39 / 1,664,796.89|-161,713.98 / -161,245.97|-6,381.22 / -6,269.17|
|development|base|1,479,752.85 / 1,470,202.73|1,459,419.74 / 1,449,487.86|1,453,284.26 / 1,443,568.96|-26,468.59 / -26,633.77|-6,135.48 / -5,918.90|
|development|reference|1,478,842.86 / 1,470,966.77|1,437,981.76 / 1,430,529.11|1,446,683.20 / 1,439,118.48|-32,159.66 / -31,848.29|8,701.44 / 8,589.37|
|development|spatial|1,469,402.11 / 1,461,655.61|1,430,608.02 / 1,423,267.16|1,439,041.92 / 1,431,588.25|-30,360.19 / -30,067.35|8,433.90 / 8,321.10|
|development|median|1,407,387.07 / 1,399,717.74|1,408,321.41 / 1,400,662.07|1,407,069.26 / 1,399,419.69|-317.81 / -298.05|-1,252.15 / -1,242.38|
|confirmation|native|1,776,517.58 / 1,834,121.89|1,712,949.80 / 1,778,626.67|1,648,779.14 / 1,708,269.43|-127,738.44 / -125,852.45|-64,170.66 / -70,357.23|
|confirmation|base|1,623,355.77 / 1,696,866.14|1,680,416.41 / 1,762,104.05|1,635,657.91 / 1,713,007.55|12,302.14 / 16,141.41|-44,758.50 / -49,096.50|
|confirmation|reference|1,407,726.19 / 1,465,885.99|1,426,262.29 / 1,487,292.99|1,405,020.87 / 1,463,805.96|-2,705.32 / -2,080.03|-21,241.42 / -23,487.03|
|confirmation|spatial|1,403,580.19 / 1,460,855.82|1,423,811.74 / 1,484,150.07|1,402,421.06 / 1,460,485.45|-1,159.14 / -370.36|-21,390.68 / -23,664.61|
|confirmation|median|1,383,228.41 / 1,447,153.42|1,390,447.76 / 1,455,219.45|1,385,239.10 / 1,449,413.21|2,010.70 / 2,259.80|-5,208.66 / -5,806.24|
|auxiliary|native|2,825,445.45 / 2,540,990.71|2,552,188.98 / 2,297,542.88|2,604,722.64 / 2,341,309.02|-220,722.81 / -199,681.69|52,533.66 / 43,766.13|
|auxiliary|base|2,549,511.97 / 2,247,696.15|2,508,974.91 / 2,208,693.70|2,514,096.80 / 2,212,352.48|-35,415.17 / -35,343.66|5,121.88 / 3,658.79|
|auxiliary|reference|2,450,073.42 / 2,135,883.50|2,412,828.44 / 2,103,199.22|2,418,525.28 / 2,107,049.60|-31,548.13 / -28,833.89|5,696.85 / 3,850.38|
|auxiliary|spatial|2,451,856.28 / 2,137,962.02|2,414,079.78 / 2,105,167.95|2,420,119.70 / 2,109,122.33|-31,736.58 / -28,839.69|6,039.92 / 3,954.38|
|auxiliary|median|2,309,352.36 / 2,023,165.66|2,299,732.46 / 2,014,911.54|2,303,291.69 / 2,018,079.02|-6,060.66 / -5,086.64|3,559.24 / 3,167.48|

## 固定分群的正反證

|窗口|比較|seen 差|new 差|>5%違例數|最大相對退步|
|---|---|---:|---:|---:|---|
|development|T-v22|1,032.01|-1,066.46|0|無|
|development|T-C|-584.94|-1,622.20|0|無|
|confirmation|T-v22|-182.12|3,670.55|0|無|
|confirmation|T-C|51.79|-9,190.57|0|無|
|auxiliary|T-v22|-2,320.51|-6,684.70|0|無|
|auxiliary|T-C|2,054.95|3,810.22|0|無|

|窗口|比較|至少500列城市群的最低／最高 MAE 差|
|---|---|---|
|development|T-v22|屏東縣 -6,413.65／臺北市 25,575.33|
|development|T-C|臺北市 -29,423.39／嘉義市 12,319.81|
|confirmation|T-v22|基隆市 -18,625.90／臺北市 37,733.45|
|confirmation|T-C|臺北市 -77,527.34／基隆市 34,000.81|
|auxiliary|T-v22|臺北市 -58,506.24／嘉義市 81,156.14|
|auxiliary|T-C|嘉義市 -56,321.97／嘉義縣 62,829.54|

最低也可能退步，最高也可能改善。城市、新舊案、季度是重疊分群，不能相加；所有月份、固定群及空群保留於摘要 saved_stage_results。

|窗口|真價帶|列數|T−v22 群MAE差|T−C 群MAE差|全窗貢獻 T−v22|全窗貢獻 T−C|
|---|---|---:|---:|---:|---:|---:|
|development|<10m|22139|-398.94|917.95|-108.73|250.17|
|development|10-20m|43914|-977.34|-39.37|-528.34|-21.28|
|development|20-50m|14197|-2,298.11|-3,480.69|-401.63|-608.31|
|development|50-100m|802|10,314.16|-18,745.89|101.83|-185.07|
|development|>=100m|182|276,307.92|-306,927.74|619.05|-687.65|
|confirmation|<10m|25585|-2,532.00|2,134.25|-484.54|408.42|
|confirmation|10-20m|74109|-2,692.43|1,279.28|-1,492.43|709.11|
|confirmation|20-50m|31959|13,181.92|-15,726.40|3,151.01|-3,759.25|
|confirmation|50-100m|1725|45,972.42|-70,518.08|593.15|-909.85|
|confirmation|>=100m|319|102,054.39|-694,513.27|243.50|-1,657.10|
|auxiliary|<10m|41132|1,606.56|-1,179.72|626.29|-459.90|
|auxiliary|10-20m|47793|-6,607.23|3,047.61|-2,992.86|1,380.47|
|auxiliary|20-50m|15033|-12,047.90|5,863.43|-1,716.56|835.41|
|auxiliary|50-100m|1382|-6,898.62|-9,220.69|-90.36|-120.77|
|auxiliary|>=100m|171|-1,164,435.13|1,187,172.64|-1,887.18|1,924.03|

全窗貢獻＝保存群列數×群MAE差÷全窗列數，五帶加總核至1e−6元；這是事後聚合診斷，不是新增門檻或依未知真價分流。

## 全部24個保存配對區間

|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|
|---|---|---|---:|---|---:|---|
|development|T−v22|project/overall|-317.81|-3,697.82～3,594.58|2000/0|crosses_zero|
|development|T−v22|project/weighted|-298.05|-3,616.03～3,600.60|2000/0|crosses_zero|
|development|T−v22|month/overall|-317.81|-2,409.48～1,390.81|2000/0|crosses_zero|
|development|T−v22|month/weighted|-298.05|-2,734.45～1,335.82|2000/0|crosses_zero|
|development|T−C|project/overall|-1,252.15|-5,249.65～2,217.64|2000/0|crosses_zero|
|development|T−C|project/weighted|-1,242.38|-5,209.44～2,173.81|2000/0|crosses_zero|
|development|T−C|month/overall|-1,252.15|-3,500.18～1,818.51|2000/0|crosses_zero|
|development|T−C|month/weighted|-1,242.38|-3,153.61～2,171.68|2000/0|crosses_zero|
|confirmation|T−v22|project/overall|2,010.70|-4,226.42～8,235.47|2000/0|crosses_zero|
|confirmation|T−v22|project/weighted|2,259.80|-4,678.28～9,170.88|2000/0|crosses_zero|
|confirmation|T−v22|month/overall|2,010.70|-78.70～4,182.30|2000/0|crosses_zero|
|confirmation|T−v22|month/weighted|2,259.80|-66.01～4,305.56|2000/0|crosses_zero|
|confirmation|T−C|project/overall|-5,208.66|-11,707.59～693.05|2000/0|crosses_zero|
|confirmation|T−C|project/weighted|-5,806.24|-13,005.06～746.45|2000/0|crosses_zero|
|confirmation|T−C|month/overall|-5,208.66|-9,013.29～-1,448.58|2000/0|negative|
|confirmation|T−C|month/weighted|-5,806.24|-9,210.88～-1,764.43|2000/0|negative|
|auxiliary|T−v22|project/overall|-6,060.66|-12,238.05～-861.40|2000/0|negative|
|auxiliary|T−v22|project/weighted|-5,086.64|-10,064.72～-932.35|2000/0|negative|
|auxiliary|T−v22|month/overall|-6,060.66|-10,604.74～-1,808.93|2000/0|negative|
|auxiliary|T−v22|month/weighted|-5,086.64|-7,382.01～-2,259.46|2000/0|negative|
|auxiliary|T−C|project/overall|3,559.24|-1,606.56～10,187.33|2000/0|crosses_zero|
|auxiliary|T−C|project/weighted|3,167.48|-1,028.23～8,472.55|2000/0|crosses_zero|
|auxiliary|T−C|month/overall|3,559.24|-1,196.57～8,290.89|2000/0|crosses_zero|
|auxiliary|T−C|month/weighted|3,167.48|-472.00～6,463.23|2000/0|crosses_zero|

方向計數：{"negative": 6, "crosses_zero": 18, "positive": 0, "unavailable": 0}。每組固定2000次、seed20260925；本摘要沒有重抽。CI不是Private勝率。

## 實測時間與解讀界線

|步驟|秒數|
|---|---:|
|development driver stage|15.766|
|confirmation driver stage|21.656|
|auxiliary driver stage|11.953|
|driver main|54.328|
|scorer 內層／execution|66.328／66.375|

三期 stage 已含於 main，scorer 內層已含於其 execution，不能重複相加。0 新 fit；driver 會載同窗已學 R/S 狀態並套用預測，不是全程無模型狀態讀取。以上非聊天／開發工時。
MAE 凸性只保證原生等權損失不高於兩端損失的平均，不保證勝過任一端，更不保證 R/S／median 後改善。沒有多seed重訓或方差估計，不能宣稱已降低統計方差。固定 R/S 未針對新平均重新最佳化，保留這個限制。
本候選受 R102–R104 結果啟發，仍是重用窗口的適應性研究；2023 靜態非價格地理晚於早期 origin，不是完整 historical-as-of 或新盲測。CI不涵蓋累積選型、重訓或未來分布變化。
使用者唯一選定仍為 v22，已回報 Public MAE 1,998,568.5706 元，Private 未知。無新 CSV、上傳或最終勾選；來源與保存核對不等於乾淨環境端到端重訓。

[固定計畫](round105_plan.md) · [全部聚合結果](../outputs/round105/evaluation/results.json) · [摘要JSON](../outputs/round105_summary/summary.json) · [來源](../src/summarize_round105.py)
