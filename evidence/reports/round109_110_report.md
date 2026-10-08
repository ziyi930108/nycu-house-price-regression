# 2026-10-08：R109／R110房價模型研究結果

使用者重新授權模型研究與新試交。v22及上台材料保留，原Public為1,998,568.5706元；Private未知。以下都是重用歷史時間窗口的離線MAE，不能當作新Kaggle成績。

R109只對原日期加權MAE prior NN末10epoch作參數平均；R110只在原max_ctr_complexity=2的單價CatBoost新增全域ctr_target_border_count=8。兩候選沒有合併或掃參數，全部六模型／三窗向量封存後才評分。

|窗口|v22 MAE|v24 MAE|R109 MAE|R109−v22|R110 MAE|R110−v22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,407,340.72|1,407,140.93|-246.14|1,431,202.66|+23,815.59|
|confirmation|1,383,228.41|1,373,195.92|1,383,418.76|+190.36|1,394,301.73|+11,073.32|
|auxiliary|2,309,352.36|2,275,409.14|2,311,013.90|+1,661.54|2,355,890.58|+46,538.22|

|窗口|v22加權|v24加權|R109加權|差vs22|R110加權|差vs22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,399,717.74|1,399,695.92|1,399,468.57|-249.17|1,423,055.24|+23,337.50|
|confirmation|1,447,153.42|1,436,088.88|1,447,369.30|+215.88|1,460,657.94|+13,504.53|
|auxiliary|2,023,165.66|1,992,570.98|2,024,485.34|+1,319.68|2,050,971.13|+27,805.47|

## R109判定

強推薦條件通過：**False**；探索提交條件通過：**False**。兩套條件已事前分開，不因結果放寬。

```json
{
  "automatic_selection": false,
  "exploratory_submission": {
    "components": {
      "at_least_two_windows_overall_improve": false,
      "each_metric_each_window_within_quarter_percent": true,
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
    "confirmation_fractional_improvement": -0.00013761680094171892,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|
|---|---:|---:|---:|
|auxiliary|+348.83|+1,880.56|0|
|confirmation|-34.37|+360.46|0|
|development|-453.31|-131.24|0|

### 全部退步門檻反證

- auxiliary：沒有超5%的季度或大型群退步。
- confirmation：沒有超5%的季度或大型群退步。
- development：沒有超5%的季度或大型群退步。

### 同軌跡epoch30對照

|窗口|epoch30 MAE|平均參數MAE|平均−epoch30|epoch30與保存v22最大預測差|
|---|---:|---:|---:|---:|
|auxiliary|2,309,352.36|2,311,013.90|+1,661.54|0.00|
|confirmation|1,383,228.41|1,383,418.76|+190.36|0.00|
|development|1,407,387.07|1,407,140.93|-246.14|0.00|

平均模型與epoch30來自同一次訓練；這個配對機制診斷不是另選一個epoch或融合比例。對保存v22的差仍可能含GPU重訓變化，不能保證固定seed逐bytes相同。

### 五層診斷

|窗口|層|MAE|
|---|---|---:|
|auxiliary|base|2,556,702.85|
|auxiliary|native|2,840,869.82|
|auxiliary|reference|2,455,970.58|
|auxiliary|spatial|2,457,574.38|
|auxiliary|median|2,311,013.90|
|confirmation|base|1,624,548.94|
|confirmation|native|1,784,581.38|
|confirmation|reference|1,408,814.25|
|confirmation|spatial|1,404,732.09|
|confirmation|median|1,383,418.76|
|development|base|1,485,935.45|
|development|native|1,841,476.21|
|development|reference|1,478,480.70|
|development|spatial|1,469,414.82|
|development|median|1,407,140.93|

### 配對區間（全部12個）

|窗口|抽法|指標|點差|95%條件區間|
|---|---|---|---:|---|
|auxiliary|month|overall|+1,661.54|`{"bootstrap_mean_delta": 1743.0989452098484, "invalid_replicates": 0, "negative_delta_fraction": 0.0, "percentile_2_5": 694.6708454155072, "percentile_97_5": 3035.6082550427063, "valid_replicates": 2000}`|
|auxiliary|month|weighted|+1,319.68|`{"bootstrap_mean_delta": 1342.518808699288, "invalid_replicates": 0, "negative_delta_fraction": 0.0, "percentile_2_5": 633.4320176930341, "percentile_97_5": 2166.7524498183043, "valid_replicates": 2000}`|
|auxiliary|project|overall|+1,661.54|`{"bootstrap_mean_delta": 1676.821795894731, "invalid_replicates": 0, "negative_delta_fraction": 0.0, "percentile_2_5": 893.9562336043455, "percentile_97_5": 2557.406720676599, "valid_replicates": 2000}`|
|auxiliary|project|weighted|+1,319.68|`{"bootstrap_mean_delta": 1329.3532050710608, "invalid_replicates": 0, "negative_delta_fraction": 0.0015, "percentile_2_5": 452.34121398298305, "percentile_97_5": 2181.402186222741, "valid_replicates": 2000}`|
|confirmation|month|overall|+190.36|`{"bootstrap_mean_delta": 171.76816143465228, "invalid_replicates": 0, "negative_delta_fraction": 0.242, "percentile_2_5": -244.31836836307775, "percentile_97_5": 649.6416560483048, "valid_replicates": 2000}`|
|confirmation|month|weighted|+215.88|`{"bootstrap_mean_delta": 194.0208977565946, "invalid_replicates": 0, "negative_delta_fraction": 0.207, "percentile_2_5": -237.295582360588, "percentile_97_5": 631.7164371947348, "valid_replicates": 2000}`|
|confirmation|project|overall|+190.36|`{"bootstrap_mean_delta": 198.27727998604743, "invalid_replicates": 0, "negative_delta_fraction": 0.2355, "percentile_2_5": -347.037390365178, "percentile_97_5": 748.9020613827217, "valid_replicates": 2000}`|
|confirmation|project|weighted|+215.88|`{"bootstrap_mean_delta": 223.831320758407, "invalid_replicates": 0, "negative_delta_fraction": 0.225, "percentile_2_5": -359.7317749368376, "percentile_97_5": 824.237509977672, "valid_replicates": 2000}`|
|development|month|overall|-246.14|`{"bootstrap_mean_delta": -204.15202926241048, "invalid_replicates": 0, "negative_delta_fraction": 0.7645, "percentile_2_5": -702.8790252612904, "percentile_97_5": 443.9042226579098, "valid_replicates": 2000}`|
|development|month|weighted|-249.17|`{"bootstrap_mean_delta": -202.68592506187153, "invalid_replicates": 0, "negative_delta_fraction": 0.761, "percentile_2_5": -702.5692094611819, "percentile_97_5": 540.2592645806726, "valid_replicates": 2000}`|
|development|project|overall|-246.14|`{"bootstrap_mean_delta": -252.0886982567471, "invalid_replicates": 0, "negative_delta_fraction": 0.798, "percentile_2_5": -821.6538300974586, "percentile_97_5": 298.1561186288657, "valid_replicates": 2000}`|
|development|project|weighted|-249.17|`{"bootstrap_mean_delta": -254.89480038519866, "invalid_replicates": 0, "negative_delta_fraction": 0.8035, "percentile_2_5": -818.7327779350453, "percentile_97_5": 280.62283637849026, "valid_replicates": 2000}`|

R109三次原生fit秒數：30.844, 36.328, 14.969；合計82.141秒。
[完整分群與五價帶結果](../outputs/round109/results.json)。分群互相重疊，不能加總；真價帶只診斷，不作test路由。

## R110判定

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
    "confirmation_fractional_improvement": -0.00800541931044541,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|
|---|---:|---:|---:|
|auxiliary|-25,393.50|+58,539.83|1|
|confirmation|-10,328.46|+27,273.44|1|
|development|-8,843.29|+41,929.14|5|

### 全部退步門檻反證

- auxiliary：[{"candidate_mae": 3997039.504757792, "control_mae": 3376918.2425181074, "delta": 620121.2622396848, "group": "基隆市", "relative_change_pct": 18.363526082208953, "rows": 1577, "section": "cities"}]
- confirmation：[{"candidate_mae": 2045300.0704692707, "control_mae": 1823161.7663045553, "delta": 222138.30416471534, "group": "基隆市", "relative_change_pct": 12.184234458524049, "rows": 1100, "section": "cities"}]
- development：[{"candidate_mae": 1566732.1187290675, "control_mae": 1460686.3735460863, "delta": 106045.74518298125, "group": "2022-10", "relative_change_pct": 7.259994144091014, "rows": 5654, "section": "months"}, {"candidate_mae": 4632551.344272973, "control_mae": 4386933.1422169255, "delta": 245618.2020560475, "group": "臺北市", "relative_change_pct": 5.598859022773366, "rows": 2698, "section": "cities"}, {"candidate_mae": 1635459.0070551352, "control_mae": 1516159.6860609034, "delta": 119299.32099423185, "group": "2", "relative_change_pct": 7.868519529376252, "rows": 5228, "section": "relative_forecast_months"}, {"candidate_mae": 1563232.4116487913, "control_mae": 1473624.7721666559, "delta": 89607.63948213542, "group": "9", "relative_change_pct": 6.08076364992061, "rows": 9401, "section": "relative_forecast_months"}, {"band": "50-100m", "candidate_mae": 6955285.398620477, "control_mae": 6570031.86672673, "delta": 385253.5318937469, "relative_change_pct": 5.863800050115814, "rows": 802, "section": "price_bands"}]

### 五層診斷

|窗口|層|MAE|
|---|---|---:|
|auxiliary|base|2,648,625.60|
|auxiliary|native|2,677,401.60|
|auxiliary|reference|2,546,417.44|
|auxiliary|spatial|2,546,454.14|
|auxiliary|median|2,355,890.58|
|confirmation|base|1,862,853.95|
|confirmation|native|1,852,191.49|
|confirmation|reference|1,560,151.74|
|confirmation|spatial|1,545,147.68|
|confirmation|median|1,394,301.73|
|development|base|1,780,591.71|
|development|native|1,847,084.09|
|development|reference|1,617,327.84|
|development|spatial|1,599,936.99|
|development|median|1,431,202.66|

### 配對區間（全部12個）

|窗口|抽法|指標|點差|95%條件區間|
|---|---|---|---:|---|
|auxiliary|month|overall|+46,538.22|`{"bootstrap_mean_delta": 46637.850415966925, "invalid_replicates": 0, "negative_delta_fraction": 0.0, "percentile_2_5": 34579.887225811835, "percentile_97_5": 60520.02172063851, "valid_replicates": 2000}`|
|auxiliary|month|weighted|+27,805.47|`{"bootstrap_mean_delta": 26959.92620753645, "invalid_replicates": 0, "negative_delta_fraction": 0.001, "percentile_2_5": 14427.058404006064, "percentile_97_5": 37559.90004643705, "valid_replicates": 2000}`|
|auxiliary|project|overall|+46,538.22|`{"bootstrap_mean_delta": 46252.918194197424, "invalid_replicates": 0, "negative_delta_fraction": 0.0, "percentile_2_5": 21509.129233868887, "percentile_97_5": 73423.11330425536, "valid_replicates": 2000}`|
|auxiliary|project|weighted|+27,805.47|`{"bootstrap_mean_delta": 27542.85783935583, "invalid_replicates": 0, "negative_delta_fraction": 0.003, "percentile_2_5": 6880.019675397011, "percentile_97_5": 50228.201569169774, "valid_replicates": 2000}`|
|confirmation|month|overall|+11,073.32|`{"bootstrap_mean_delta": 10634.072216324648, "invalid_replicates": 0, "negative_delta_fraction": 0.01, "percentile_2_5": 1096.2962922072038, "percentile_97_5": 20998.445780644433, "valid_replicates": 2000}`|
|confirmation|month|weighted|+13,504.53|`{"bootstrap_mean_delta": 12388.201782100778, "invalid_replicates": 0, "negative_delta_fraction": 0.005, "percentile_2_5": 2968.3037344019276, "percentile_97_5": 19651.196725832116, "valid_replicates": 2000}`|
|confirmation|project|overall|+11,073.32|`{"bootstrap_mean_delta": 10992.132638472378, "invalid_replicates": 0, "negative_delta_fraction": 0.072, "percentile_2_5": -3601.2458948137355, "percentile_97_5": 26252.522573171642, "valid_replicates": 2000}`|
|confirmation|project|weighted|+13,504.53|`{"bootstrap_mean_delta": 13416.615582607918, "invalid_replicates": 0, "negative_delta_fraction": 0.0525, "percentile_2_5": -2556.1353290928587, "percentile_97_5": 30093.44077719689, "valid_replicates": 2000}`|
|development|month|overall|+23,815.59|`{"bootstrap_mean_delta": 22989.338000786425, "invalid_replicates": 0, "negative_delta_fraction": 0.003, "percentile_2_5": 5902.822973163706, "percentile_97_5": 38069.64720121539, "valid_replicates": 2000}`|
|development|month|weighted|+23,337.50|`{"bootstrap_mean_delta": 23043.647607554434, "invalid_replicates": 0, "negative_delta_fraction": 0.005, "percentile_2_5": 5464.673444585683, "percentile_97_5": 40533.74698642749, "valid_replicates": 2000}`|
|development|project|overall|+23,815.59|`{"bootstrap_mean_delta": 24008.67600809332, "invalid_replicates": 0, "negative_delta_fraction": 0.031, "percentile_2_5": -1005.1862594624049, "percentile_97_5": 52951.15817959948, "valid_replicates": 2000}`|
|development|project|weighted|+23,337.50|`{"bootstrap_mean_delta": 23615.355275326518, "invalid_replicates": 0, "negative_delta_fraction": 0.0345, "percentile_2_5": -1197.5940440103818, "percentile_97_5": 52349.44879030746, "valid_replicates": 2000}`|

R110三次原生fit秒數：526.406, 518.110, 377.500；合計1422.016秒。
[完整分群與五價帶結果](../outputs/round110/results.json)。分群互相重疊，不能加總；真價帶只診斷，不作test路由。

## 時間與限制

序列訓練主程序1654.312秒（已含六次fit與重播／保存），獨立評分100.625秒；不是整段聊天／開發工時，fit不重複加總。

原v22完整鏈與原v24指標已精確重播；新模型全部query保存重載逐bytes相同，原CSV／selection／登錄及保護檔保持。R110沿用前輪審過的原生參數序列化比較：有限float用rtol1e-7，其餘型別／值精確，兩組診斷與微差完整保存；價格向量仍要求bytes精確。實際CTR邊界使用摘要是結構證據，不是泛化證據，未要求八個邊界全部入樹。

窗口多輪參與選型；單seed且保存v22不是同期matched重訓，小改善不能證明獨立因果效果。區塊bootstrap只條件於本次預測，沒有校正累積選型／未來分布／重訓變化，不能稱Private勝率。原fit與query狀態、標籤隔離不代表全新未見驗證。v24離線較強的反證完整保留。沒有對test ID／未知真價手改價格，也沒有查成交來源。

[事前計畫](round109_110_plan.md) · [恢復要求](../outputs/research_resume_20261008/user_request.json) · [凍結request](../outputs/research_round109_110_20261008/request.json) · [全模型預測封存](../outputs/research_round109_110_20261008/seal.json)
