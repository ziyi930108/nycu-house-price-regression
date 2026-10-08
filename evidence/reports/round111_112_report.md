# R111／R112：固定架構與單價Ordered研究結果

2026-10-08。這是課程標籤的時間外離線MAE，單位新臺幣元，越低越好。不是新增Public或Private分數；v22的已回報Public為1,998,568.5706，Private未知。使用者最新要求持續優化至Codex剩餘20%，不用重置或額外點數。

R111把prior NN換成固定兩個殘差block，仍以日期加權總價MAE訓練正倍率；R112只把原R79 unit/date×面積CatBoost的Plain改Ordered。沒有合併兩候選、搜尋容量／比例／epoch，亦未帶入R107 SmoothL1或R110八邊界。六模型、metadata與五層向量直接封存並核對後，才讀query價格評分。

|窗口|v22 MAE|v24 MAE|R111 MAE|R111−v22|R112 MAE|R112−v22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,407,340.72|1,409,404.44|+2,017.37|1,411,675.28|+4,288.21|
|confirmation|1,383,228.41|1,373,195.92|1,388,961.64|+5,733.24|1,399,546.14|+16,317.73|
|auxiliary|2,309,352.36|2,275,409.14|2,303,159.38|-6,192.97|2,318,041.62|+8,689.27|

|窗口|v22固定加權|v24固定加權|R111加權|差vs22|R112加權|差vs22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,399,717.74|1,399,695.92|1,401,690.89|+1,973.14|1,403,796.67|+4,078.92|
|confirmation|1,447,153.42|1,436,088.88|1,453,709.22|+6,555.80|1,465,122.58|+17,969.16|
|auxiliary|2,023,165.66|1,992,570.98|2,017,848.34|-5,317.32|2,030,596.19|+7,430.53|

## R111完整判定

強推薦通過：**False**；探索試交通過：**False**。兩套門檻事前固定，未因新分數放寬。

```json
{
  "automatic_selection": false,
  "exploratory_submission": {
    "components": {
      "at_least_two_windows_overall_improve": false,
      "each_metric_each_window_within_quarter_percent": false,
      "latest_both_no_worse": false,
      "no_large_group_or_quarter_above_five_percent": true
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
      "no_quarter_or_large_group_deterioration_above_5pct": true
    },
    "confirmation_fractional_improvement": -0.004144824868320932,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|
|---|---:|---:|---:|
|development|-1,003.68|+3,692.93|0|
|confirmation|-1,507.72|+11,214.30|0|
|auxiliary|-2,830.56|-6,753.98|0|

所有超門檻反證（群可能重疊，不可加總）：

- development：沒有超門檻季度或大型群。
- confirmation：沒有超門檻季度或大型群。
- auxiliary：沒有超門檻季度或大型群。

|窗口|層|MAE|
|---|---|---:|
|development|base|1,499,792.66|
|development|native|1,874,779.55|
|development|reference|1,482,444.91|
|development|spatial|1,474,662.44|
|development|median|1,409,404.44|
|confirmation|base|1,661,657.80|
|confirmation|native|1,831,171.36|
|confirmation|reference|1,422,799.48|
|confirmation|spatial|1,417,098.24|
|confirmation|median|1,388,961.64|
|auxiliary|base|2,539,228.32|
|auxiliary|native|2,786,112.20|
|auxiliary|reference|2,435,812.52|
|auxiliary|spatial|2,437,232.55|
|auxiliary|median|2,303,159.38|

全部配對條件區間：

|窗口|抽法|指標|點差|條件摘要|
|---|---|---|---:|---|
|development|month|overall|+2,017.37|`{"bootstrap_mean_delta": 2120.077712141266, "invalid_replicates": 0, "negative_delta_fraction": 0.0185, "percentile_2_5": 56.07035247353488, "percentile_97_5": 4735.253876430412, "valid_replicates": 2000}`|
|development|month|weighted|+1,973.14|`{"bootstrap_mean_delta": 2227.158364848459, "invalid_replicates": 0, "negative_delta_fraction": 0.0115, "percentile_2_5": 186.50119900400168, "percentile_97_5": 5197.604983412461, "valid_replicates": 2000}`|
|development|project|overall|+2,017.37|`{"bootstrap_mean_delta": 2065.55446192916, "invalid_replicates": 0, "negative_delta_fraction": 0.1845, "percentile_2_5": -2369.556350003654, "percentile_97_5": 6412.172952483605, "valid_replicates": 2000}`|
|development|project|weighted|+1,973.14|`{"bootstrap_mean_delta": 2012.8647690063235, "invalid_replicates": 0, "negative_delta_fraction": 0.188, "percentile_2_5": -2335.0062219488727, "percentile_97_5": 6208.958543918584, "valid_replicates": 2000}`|
|confirmation|month|overall|+5,733.24|`{"bootstrap_mean_delta": 5462.454886024841, "invalid_replicates": 0, "negative_delta_fraction": 0.0715, "percentile_2_5": -1440.8995014195445, "percentile_97_5": 12763.385417844924, "valid_replicates": 2000}`|
|confirmation|month|weighted|+6,555.80|`{"bootstrap_mean_delta": 6082.409834358505, "invalid_replicates": 0, "negative_delta_fraction": 0.053, "percentile_2_5": -1280.341751599207, "percentile_97_5": 12181.524997360311, "valid_replicates": 2000}`|
|confirmation|project|overall|+5,733.24|`{"bootstrap_mean_delta": 5839.122231291847, "invalid_replicates": 0, "negative_delta_fraction": 0.0475, "percentile_2_5": -873.4242079711461, "percentile_97_5": 14481.91790072057, "valid_replicates": 2000}`|
|confirmation|project|weighted|+6,555.80|`{"bootstrap_mean_delta": 6677.0831580393315, "invalid_replicates": 0, "negative_delta_fraction": 0.0435, "percentile_2_5": -792.1045528834802, "percentile_97_5": 16380.178360291458, "valid_replicates": 2000}`|
|auxiliary|month|overall|-6,192.97|`{"bootstrap_mean_delta": -6659.622772968741, "invalid_replicates": 0, "negative_delta_fraction": 0.7465, "percentile_2_5": -24562.14079683288, "percentile_97_5": 4155.790840904467, "valid_replicates": 2000}`|
|auxiliary|month|weighted|-5,317.32|`{"bootstrap_mean_delta": -5507.987082672343, "invalid_replicates": 0, "negative_delta_fraction": 0.7965, "percentile_2_5": -18165.350887400564, "percentile_97_5": 2660.3639835037757, "valid_replicates": 2000}`|
|auxiliary|project|overall|-6,192.97|`{"bootstrap_mean_delta": -6229.576602639901, "invalid_replicates": 0, "negative_delta_fraction": 0.733, "percentile_2_5": -25769.69406991524, "percentile_97_5": 7395.536340185009, "valid_replicates": 2000}`|
|auxiliary|project|weighted|-5,317.32|`{"bootstrap_mean_delta": -5342.652363317011, "invalid_replicates": 0, "negative_delta_fraction": 0.776, "percentile_2_5": -19732.352462211195, "percentile_97_5": 4978.085775588895, "valid_replicates": 2000}`|

R111三次原生fit：32.781, 43.172, 18.234秒，合計94.187秒。
[完整各城市／月份／季度／五價帶與v24反證](../outputs/round111/results.json)。

## R112完整判定

強推薦通過：**False**；探索試交通過：**False**。兩套門檻事前固定，未因新分數放寬。

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
    "confirmation_fractional_improvement": -0.011796845246239708,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|
|---|---:|---:|---:|
|development|-10,008.54|+12,217.60|3|
|confirmation|+1,780.25|+27,321.91|1|
|auxiliary|+3,855.84|+9,495.71|0|

所有超門檻反證（群可能重疊，不可加總）：

- development：[{"candidate_mae": 1552224.9106275486, "control_mae": 1460686.3735460863, "delta": 91538.53708146233, "group": "2022-10", "relative_change_pct": 6.266816665047378, "rows": 5654, "section": "months"}, {"candidate_mae": 4658525.443054098, "control_mae": 4386933.1422169255, "delta": 271592.3008371722, "group": "臺北市", "relative_change_pct": 6.190937769795227, "rows": 2698, "section": "cities"}, {"candidate_mae": 1626017.3905121384, "control_mae": 1516159.6860609034, "delta": 109857.70445123501, "group": "2", "relative_change_pct": 7.245787199147444, "rows": 5228, "section": "relative_forecast_months"}]
- confirmation：[{"candidate_mae": 1384149.7581684843, "control_mae": 1317138.3839079912, "delta": 67011.37426049309, "group": "新北市", "relative_change_pct": 5.087648729943488, "rows": 21822, "section": "cities"}]
- auxiliary：沒有超門檻季度或大型群。

|窗口|層|MAE|
|---|---|---:|
|development|base|1,708,227.02|
|development|native|1,733,015.01|
|development|reference|1,494,558.09|
|development|spatial|1,485,552.63|
|development|median|1,411,675.28|
|confirmation|base|1,833,002.91|
|confirmation|native|1,786,384.33|
|confirmation|reference|1,489,967.32|
|confirmation|spatial|1,482,300.50|
|confirmation|median|1,399,546.14|
|auxiliary|base|2,404,581.03|
|auxiliary|native|2,343,600.65|
|auxiliary|reference|2,308,112.89|
|auxiliary|spatial|2,310,794.17|
|auxiliary|median|2,318,041.62|

全部配對條件區間：

|窗口|抽法|指標|點差|條件摘要|
|---|---|---|---:|---|
|development|month|overall|+4,288.21|`{"bootstrap_mean_delta": 5349.1192996240225, "invalid_replicates": 0, "negative_delta_fraction": 0.349, "percentile_2_5": -9397.597047622548, "percentile_97_5": 28804.765076507538, "valid_replicates": 2000}`|
|development|month|weighted|+4,078.92|`{"bootstrap_mean_delta": 5858.740109903571, "invalid_replicates": 0, "negative_delta_fraction": 0.3685, "percentile_2_5": -10251.13251364362, "percentile_97_5": 34534.90058171516, "valid_replicates": 2000}`|
|development|project|overall|+4,288.21|`{"bootstrap_mean_delta": 4281.071811257142, "invalid_replicates": 0, "negative_delta_fraction": 0.3655, "percentile_2_5": -16072.70966208647, "percentile_97_5": 27865.743410050207, "valid_replicates": 2000}`|
|development|project|weighted|+4,078.92|`{"bootstrap_mean_delta": 4160.245023328272, "invalid_replicates": 0, "negative_delta_fraction": 0.3705, "percentile_2_5": -15955.61877128515, "percentile_97_5": 27745.163892169167, "valid_replicates": 2000}`|
|confirmation|month|overall|+16,317.73|`{"bootstrap_mean_delta": 15687.793501806013, "invalid_replicates": 0, "negative_delta_fraction": 0.005, "percentile_2_5": 2429.688190644054, "percentile_97_5": 29789.541481541382, "valid_replicates": 2000}`|
|confirmation|month|weighted|+17,969.16|`{"bootstrap_mean_delta": 16644.196028036444, "invalid_replicates": 0, "negative_delta_fraction": 0.015, "percentile_2_5": 1965.4895906173565, "percentile_97_5": 28820.92859488354, "valid_replicates": 2000}`|
|confirmation|project|overall|+16,317.73|`{"bootstrap_mean_delta": 16076.77069390957, "invalid_replicates": 0, "negative_delta_fraction": 0.009, "percentile_2_5": 2358.1715215844624, "percentile_97_5": 31320.270991389905, "valid_replicates": 2000}`|
|confirmation|project|weighted|+17,969.16|`{"bootstrap_mean_delta": 17723.625771318, "invalid_replicates": 0, "negative_delta_fraction": 0.009, "percentile_2_5": 2710.219234396587, "percentile_97_5": 34427.83074696871, "valid_replicates": 2000}`|
|auxiliary|month|overall|+8,689.27|`{"bootstrap_mean_delta": 9618.86759935949, "invalid_replicates": 0, "negative_delta_fraction": 0.1105, "percentile_2_5": -4438.269358129846, "percentile_97_5": 27203.89358606352, "valid_replicates": 2000}`|
|auxiliary|month|weighted|+7,430.53|`{"bootstrap_mean_delta": 7881.58120517409, "invalid_replicates": 0, "negative_delta_fraction": 0.0755, "percentile_2_5": -2587.167645340145, "percentile_97_5": 18851.60851945217, "valid_replicates": 2000}`|
|auxiliary|project|overall|+8,689.27|`{"bootstrap_mean_delta": 8404.56896260106, "invalid_replicates": 0, "negative_delta_fraction": 0.2535, "percentile_2_5": -19509.1104847335, "percentile_97_5": 32685.784124263468, "valid_replicates": 2000}`|
|auxiliary|project|weighted|+7,430.53|`{"bootstrap_mean_delta": 7217.461470982959, "invalid_replicates": 0, "negative_delta_fraction": 0.2255, "percentile_2_5": -13587.800939890247, "percentile_97_5": 25468.839648769885, "valid_replicates": 2000}`|

R112三次原生fit：286.297, 302.187, 284.328秒，合計872.812秒。
[完整各城市／月份／季度／五價帶與v24反證](../outputs/round112/results.json)。

## 執行與限制

序列主程序1111.563秒，已含六次fit／重載／組合；評分105.282秒，不能把fit再加到主程序。整段聊天／開發工時未量測。

R111為本題LayerNorm／正倍率／加權總價loss的ResNet適配，不是官方完整復刻。模型容量、初始化及亂數消耗同時改變，不能純歸因skip。R112保留R17 raw uniform Ordered的development整體+6.085%、新案+7.435%、Q2+9.205%反證；本輪只測unit/date×面積交互，不是Ordered首次試驗。原native CBM無法恢復的三項runtime控制在metadata另明示，不偽稱已由模型驗出。

原v22和v24重播是保存模型對照，不是同期matched重訓。固定seed不保證GPU重訓逐bytes相同；保存模型全query重載exact是不同檢查。群組OOF prior非時間OOF，三個年度已多輪重用，不能當新盲測。Bootstrap條件於保存預測，未校正累積選型、重訓或未來移轉，也不是Private勝率。

不取得成交來源答案、不按test ID或未知價格手改、不以Public搜尋。新CSV仍須門檻、另立全量規約及獨立發行核對；本報告本身不代表已完成fullfit、上傳或最終選定。保存可追溯不等於新環境從原始資料端到端重訓。

[事前規約](round111_112_plan.md) · [最新要求](../outputs/research_continue_20261008/user_request.json) · [凍結request](../outputs/research_round111_112_20261008/request.json) · [全模型封存](../outputs/research_round111_112_20261008/seal.json)
