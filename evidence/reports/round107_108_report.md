# 2026-10-08：R107／R108房價模型研究結果

使用者重新授權模型研究與新試交。v22及上台材料保留，原Public為1,998,568.5706元；Private未知。以下都是重用歷史時間窗口的離線MAE，不能當作新Kaggle成績。

R107在原日期加權prior NN上使用固定10萬元SmoothL1；R108只將單價CatBoost的max_ctr_complexity由2改1。兩候選沒有合併或掃參數，全部六模型／三窗向量封存後才評分。

|窗口|v22 MAE|v24 MAE|R107 MAE|R107−v22|R108 MAE|R108−v22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,407,340.72|1,405,971.19|-1,415.88|1,419,128.48|+11,741.41|
|confirmation|1,383,228.41|1,373,195.92|1,381,702.87|-1,525.54|1,391,976.13|+8,747.72|
|auxiliary|2,309,352.36|2,275,409.14|2,310,236.02|+883.66|2,308,642.06|-710.30|

|窗口|v22加權|v24加權|R107加權|差vs22|R108加權|差vs22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,399,717.74|1,399,695.92|1,398,319.72|-1,398.03|1,411,218.42|+11,500.67|
|confirmation|1,447,153.42|1,436,088.88|1,445,430.37|-1,723.04|1,456,285.16|+9,131.74|
|auxiliary|2,023,165.66|1,992,570.98|2,023,921.62|+755.96|2,022,415.67|-749.99|

## R107判定

強推薦條件通過：**False**；探索提交條件通過：**True**。兩套條件已事前分開，不因結果放寬。

```json
{
  "automatic_selection": false,
  "exploratory_submission": {
    "components": {
      "at_least_two_windows_overall_improve": true,
      "each_metric_each_window_within_quarter_percent": true,
      "latest_both_no_worse": true,
      "no_large_group_or_quarter_above_five_percent": true
    },
    "passed": true
  },
  "private_score": null,
  "promotion": {
    "Private_score": null,
    "adoption_research_gate_passed": false,
    "automatic_adoption": false,
    "components": {
      "all_three_overall_improve": false,
      "all_three_weighted_improve": false,
      "confirmation_at_least_half_percent": false,
      "no_quarter_or_large_group_deterioration_above_5pct": true
    },
    "confirmation_fractional_improvement": 0.001102881788732388,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|
|---|---:|---:|---:|
|auxiliary|+393.29|+965.48|0|
|confirmation|+213.10|-2,841.61|0|
|development|-196.46|-2,092.20|0|

### 全部退步門檻反證

- auxiliary：沒有超5%的季度或大型群退步。
- confirmation：沒有超5%的季度或大型群退步。
- development：沒有超5%的季度或大型群退步。

### 五層診斷

|窗口|層|MAE|
|---|---|---:|
|auxiliary|base|2,551,143.40|
|auxiliary|native|2,826,164.70|
|auxiliary|reference|2,451,690.05|
|auxiliary|spatial|2,452,809.11|
|auxiliary|median|2,310,236.02|
|confirmation|base|1,621,850.58|
|confirmation|native|1,777,133.98|
|confirmation|reference|1,405,482.31|
|confirmation|spatial|1,401,566.48|
|confirmation|median|1,381,702.87|
|development|base|1,488,689.99|
|development|native|1,860,810.11|
|development|reference|1,484,640.45|
|development|spatial|1,474,396.27|
|development|median|1,405,971.19|

### 配對區間（全部12個）

|窗口|抽法|指標|點差|95%條件區間|
|---|---|---|---:|---|
|auxiliary|month|overall|+883.66|`{"bootstrap_mean_delta": 937.1249898019585, "invalid_replicates": 0, "negative_delta_fraction": 0.1425, "percentile_2_5": -749.4503242131323, "percentile_97_5": 2695.594794404693, "valid_replicates": 2000}`|
|auxiliary|month|weighted|+755.96|`{"bootstrap_mean_delta": 750.9368700986181, "invalid_replicates": 0, "negative_delta_fraction": 0.178, "percentile_2_5": -1012.075269029953, "percentile_97_5": 2462.3404682153023, "valid_replicates": 2000}`|
|auxiliary|project|overall|+883.66|`{"bootstrap_mean_delta": 893.2857999991807, "invalid_replicates": 0, "negative_delta_fraction": 0.2875, "percentile_2_5": -1921.029597792204, "percentile_97_5": 4066.0421942271496, "valid_replicates": 2000}`|
|auxiliary|project|weighted|+755.96|`{"bootstrap_mean_delta": 760.4065921338935, "invalid_replicates": 0, "negative_delta_fraction": 0.2735, "percentile_2_5": -1548.5338621219912, "percentile_97_5": 3286.5106682933033, "valid_replicates": 2000}`|
|confirmation|month|overall|-1,525.54|`{"bootstrap_mean_delta": -1544.8838370997473, "invalid_replicates": 0, "negative_delta_fraction": 1.0, "percentile_2_5": -2931.0293803107925, "percentile_97_5": -554.5603396070655, "valid_replicates": 2000}`|
|confirmation|month|weighted|-1,723.04|`{"bootstrap_mean_delta": -1824.843690580636, "invalid_replicates": 0, "negative_delta_fraction": 1.0, "percentile_2_5": -3987.223479075229, "percentile_97_5": -478.9001393448445, "valid_replicates": 2000}`|
|confirmation|project|overall|-1,525.54|`{"bootstrap_mean_delta": -1517.0169215039634, "invalid_replicates": 0, "negative_delta_fraction": 0.9325, "percentile_2_5": -3594.910185398179, "percentile_97_5": 523.352777298243, "valid_replicates": 2000}`|
|confirmation|project|weighted|-1,723.04|`{"bootstrap_mean_delta": -1712.3277805848907, "invalid_replicates": 0, "negative_delta_fraction": 0.94, "percentile_2_5": -3972.718410707993, "percentile_97_5": 508.6508379055986, "valid_replicates": 2000}`|
|development|month|overall|-1,415.88|`{"bootstrap_mean_delta": -1380.7383857145346, "invalid_replicates": 0, "negative_delta_fraction": 1.0, "percentile_2_5": -1941.2464643393178, "percentile_97_5": -625.9457929048222, "valid_replicates": 2000}`|
|development|month|weighted|-1,398.03|`{"bootstrap_mean_delta": -1364.980738420072, "invalid_replicates": 0, "negative_delta_fraction": 1.0, "percentile_2_5": -1910.388528324198, "percentile_97_5": -626.299257342687, "valid_replicates": 2000}`|
|development|project|overall|-1,415.88|`{"bootstrap_mean_delta": -1420.9042702007484, "invalid_replicates": 0, "negative_delta_fraction": 0.9235, "percentile_2_5": -3371.5796518557413, "percentile_97_5": 550.8371527053694, "valid_replicates": 2000}`|
|development|project|weighted|-1,398.03|`{"bootstrap_mean_delta": -1399.047371012624, "invalid_replicates": 0, "negative_delta_fraction": 0.9245, "percentile_2_5": -3300.910870225064, "percentile_97_5": 549.1564250281754, "valid_replicates": 2000}`|

R107三次原生fit秒數：16.078, 33.390, 13.156；合計62.624秒。
[完整分群與五價帶結果](../outputs/round107/results.json)。分群互相重疊，不能加總；真價帶只診斷，不作test路由。

## R108判定

強推薦條件通過：**False**；探索提交條件通過：**False**。兩套條件已事前分開，不因結果放寬。

```json
{
  "automatic_selection": false,
  "exploratory_submission": {
    "components": {
      "at_least_two_windows_overall_improve": false,
      "each_metric_each_window_within_quarter_percent": false,
      "latest_both_no_worse": false,
      "no_large_group_or_quarter_above_five_percent": false
    },
    "passed": false
  },
  "private_score": null,
  "promotion": {
    "Private_score": null,
    "adoption_research_gate_passed": false,
    "automatic_adoption": false,
    "components": {
      "all_three_overall_improve": false,
      "all_three_weighted_improve": false,
      "confirmation_at_least_half_percent": false,
      "no_quarter_or_large_group_deterioration_above_5pct": false
    },
    "confirmation_fractional_improvement": -0.006324135463991502,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|
|---|---:|---:|---:|
|auxiliary|-862.71|-684.87|0|
|confirmation|+5,367.22|+11,306.61|0|
|development|-4,703.38|+20,862.16|1|

### 全部退步門檻反證

- auxiliary：沒有超5%的季度或大型群退步。
- confirmation：沒有超5%的季度或大型群退步。
- development：[{"candidate_mae": 4676319.933756117, "control_mae": 4386933.1422169255, "delta": 289386.79153919127, "group": "臺北市", "relative_change_pct": 6.596562613510685, "rows": 2698, "section": "cities"}]

### 五層診斷

|窗口|層|MAE|
|---|---|---:|
|auxiliary|base|2,438,155.40|
|auxiliary|native|2,451,309.74|
|auxiliary|reference|2,345,013.62|
|auxiliary|spatial|2,348,590.82|
|auxiliary|median|2,308,642.06|
|confirmation|base|1,803,955.60|
|confirmation|native|1,807,831.30|
|confirmation|reference|1,502,221.73|
|confirmation|spatial|1,494,307.10|
|confirmation|median|1,391,976.13|
|development|base|1,662,717.19|
|development|native|1,742,084.25|
|development|reference|1,530,623.76|
|development|spatial|1,516,279.77|
|development|median|1,419,128.48|

### 配對區間（全部12個）

|窗口|抽法|指標|點差|95%條件區間|
|---|---|---|---:|---|
|auxiliary|month|overall|-710.30|`{"bootstrap_mean_delta": -687.3391264217632, "invalid_replicates": 0, "negative_delta_fraction": 0.639, "percentile_2_5": -4144.40405624127, "percentile_97_5": 2823.4517431780696, "valid_replicates": 2000}`|
|auxiliary|month|weighted|-749.99|`{"bootstrap_mean_delta": -792.3137534113926, "invalid_replicates": 0, "negative_delta_fraction": 0.632, "percentile_2_5": -5107.793400373531, "percentile_97_5": 3115.6483113977592, "valid_replicates": 2000}`|
|auxiliary|project|overall|-710.30|`{"bootstrap_mean_delta": -1157.4080130959792, "invalid_replicates": 0, "negative_delta_fraction": 0.5475, "percentile_2_5": -17808.3270184904, "percentile_97_5": 15100.275824142072, "valid_replicates": 2000}`|
|auxiliary|project|weighted|-749.99|`{"bootstrap_mean_delta": -1192.7354967770013, "invalid_replicates": 0, "negative_delta_fraction": 0.564, "percentile_2_5": -14707.143170958176, "percentile_97_5": 12535.96345406336, "valid_replicates": 2000}`|
|confirmation|month|overall|+8,747.72|`{"bootstrap_mean_delta": 8842.709721124143, "invalid_replicates": 0, "negative_delta_fraction": 0.0005, "percentile_2_5": 2610.6990342894396, "percentile_97_5": 15542.88569728448, "valid_replicates": 2000}`|
|confirmation|month|weighted|+9,131.74|`{"bootstrap_mean_delta": 9072.356567902818, "invalid_replicates": 0, "negative_delta_fraction": 0.001, "percentile_2_5": 2367.6527563435025, "percentile_97_5": 16705.737678183017, "valid_replicates": 2000}`|
|confirmation|project|overall|+8,747.72|`{"bootstrap_mean_delta": 8926.632989760245, "invalid_replicates": 0, "negative_delta_fraction": 0.0675, "percentile_2_5": -3028.4777112588986, "percentile_97_5": 21185.67711103574, "valid_replicates": 2000}`|
|confirmation|project|weighted|+9,131.74|`{"bootstrap_mean_delta": 9320.22025675504, "invalid_replicates": 0, "negative_delta_fraction": 0.073, "percentile_2_5": -3415.949119604286, "percentile_97_5": 22569.518972530655, "valid_replicates": 2000}`|
|development|month|overall|+11,741.41|`{"bootstrap_mean_delta": 11925.567985779711, "invalid_replicates": 0, "negative_delta_fraction": 0.006, "percentile_2_5": 2183.998611571733, "percentile_97_5": 21539.301781403377, "valid_replicates": 2000}`|
|development|month|weighted|+11,500.67|`{"bootstrap_mean_delta": 11849.55810559054, "invalid_replicates": 0, "negative_delta_fraction": 0.0045, "percentile_2_5": 2040.5478683541996, "percentile_97_5": 24540.48471415392, "valid_replicates": 2000}`|
|development|project|overall|+11,741.41|`{"bootstrap_mean_delta": 11869.750355794165, "invalid_replicates": 0, "negative_delta_fraction": 0.049, "percentile_2_5": -1928.8451328030671, "percentile_97_5": 27280.506215323643, "valid_replicates": 2000}`|
|development|project|weighted|+11,500.67|`{"bootstrap_mean_delta": 11660.443846706608, "invalid_replicates": 0, "negative_delta_fraction": 0.0485, "percentile_2_5": -1886.6195447059113, "percentile_97_5": 26687.32141227649, "valid_replicates": 2000}`|

R108三次原生fit秒數：30.172, 40.235, 30.438；合計100.845秒。
[完整分群與五價帶結果](../outputs/round108/results.json)。分群互相重疊，不能加總；真價帶只診斷，不作test路由。

## 時間與限制

序列訓練主程序270.954秒（已含六次fit與重播／保存），獨立評分82.844秒；不是整段聊天／開發工時，fit不重複加總。

原v22完整鏈與原v24指標已精確重播；新模型全部query保存重載逐bytes相同，原CSV／selection／登錄及保護檔保持。R108在訓練前審查修正了原生參數序列化尾數的誤拒絕：有限float用rtol1e-7，其餘型別／值精確，兩組診斷與微差完整保存；價格向量仍要求bytes精確。

窗口多輪參與選型；單seed且保存v22不是同期matched重訓，小改善不能證明獨立因果效果。區塊bootstrap只條件於本次預測，沒有校正累積選型／未來分布／重訓變化，不能稱Private勝率。原fit與query狀態、標籤隔離不代表全新未見驗證。v24離線較強的反證完整保留。沒有對test ID／未知真價手改價格，也沒有查成交來源。

[事前計畫](round107_108_plan.md) · [要求與前置證據](../outputs/research_resume_20261008/user_request.json) · [凍結request](../outputs/research_resume_20261008/request.json) · [全模型預測封存](../outputs/research_resume_20261008/seal.json)
