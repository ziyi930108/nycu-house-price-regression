# R114：參考總價加金額價差，未通過完整門檻

2026-10-08。四個起點全部完成固定30 epochs訓練、封存及分開評分，獨立postfit已通過。原三窗探索／強門檻均失敗，額外2月起點在允許的微幅退步範圍內；combined門檻失敗。**不全量fit，不產v30 CSV，不改v22選定或上台材料。** Public／Private沒有新增分數。

動機是檢查原「參考總價×NN正倍率」的參數化是否限制金額誤差的學習。改為「參考總價＋NN有正負值的價差」，保持原MLP、144數值／23類別、九array、日期加權總價MAE、optimizer、30epochs、seed42和固定組合。head activation、單位、bias、正值約束及梯度尺度一同改變，不能將差異只歸因於某一條梯度。

|起點|fit／query列|v22整體MAE|R114整體MAE|差vs v22|固定加權差|
|---|---:|---:|---:|---:|---:|
|development|240,316／81,234|1,407,387.07|1,410,687.83|+3,300.76|+3,265.18|
|confirmation|321,550／133,697|1,383,228.41|1,388,411.32|+5,182.91|+5,340.29|
|auxiliary|134,805／105,511|2,309,352.36|2,303,762.54|-5,589.82|-3,188.83|
|offset_2023_02_11|272,117／107,169|1,336,410.67|1,336,671.42|+260.75|+454.19|

單位為新臺幣元，負值較好。最新confirmation退步約0.3747%，兩指標都變差，且只有auxiliary整體改善，未達探索條件；auxiliary嘉義市731列退步5.6063%，也觸及大型分群限制。額外起點微幅退步但通過其上限，並不代表該起點改善。

## 原定判定與負值政策

```json
{
  "august": {
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
      "confirmation_fractional_improvement": -0.003746969678989488,
      "selected_submission": "v22"
    },
    "selected_submission": "v22"
  },
  "automatic_selection": false,
  "combined_exploratory_submission": {
    "components": {
      "fixed_extra_origin": true,
      "original_three_windows": false
    },
    "passed": false
  },
  "combined_promotion": {
    "components": {
      "fixed_extra_origin": true,
      "original_three_windows": false
    },
    "passed": false
  },
  "extra_origin": {
    "components": {
      "no_quarter_or_large_group_above_5pct": true,
      "overall_regression_at_most_0_25pct": true,
      "weighted_regression_at_most_0_25pct": true
    },
    "passed": true
  },
  "full_data_fit": false,
  "new_CSV": false,
  "private_score": null,
  "reserved_exploratory_version": 30,
  "selected_submission": "v22"
}
```

訓練以未裁剪的F+h計算總價MAE，fit不clip；native推論才作一次max(total,0)，之後沿原reference／spatial及median組合。對非負真價，這個投影本身不會增加單筆絕對誤差；這個數學性質不代表R114勝過v22。保存的raw向量是未投影的總價，不是NN價差h。

**評分欄位語意：** 原JSON的query_y_used_for_diagnostics=false只指先封存的負值／投影統計。同一區塊後加入的overall_mae是在封存後讀query真價計算，確實使用query y。原結果不回寫；[語意補充](../outputs/research_round114_20261008/evaluation_semantics_clarification.json)明列此界線，不能把MAE說成不需要答案的指標。

## development完整診斷

fit實測29.094秒。

|層|總價MAE|
|---|---:|
|base|1,471,567.44|
|native|1,764,314.82|
|reference|1,469,957.16|
|spatial|1,460,982.84|
|median|1,410,687.83|

未投影總價與固定非負投影：

```json
{
  "negative_residual_not_equivalent_to_negative_total": true,
  "overall_mae": 1766430.1062908375,
  "projected_max_TWD": 6688758640.289307,
  "projected_min_TWD": 0.0,
  "projected_zero_fraction": 0.0005047147746017678,
  "projected_zero_rows": 41,
  "projection_changes_count": 41,
  "projection_max_correction_TWD": 17046676.635742188,
  "projection_mean_correction_TWD": 2115.286370689309,
  "query_y_used_for_diagnostics": false,
  "raw_max_TWD": 6688758640.289307,
  "raw_min_TWD": -17046676.635742188,
  "raw_negative_total_fraction": 0.0005047147746017678,
  "raw_negative_total_rows": 41,
  "raw_zero_total_rows": 0
}
```

所有季度／至少500列群退步超5%（群可重疊，不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+3,300.76|[+1,844.44, +4,888.07]|
|month|weighted|+3,265.18|[+1,797.21, +5,116.43]|
|project|overall|+3,300.76|[-1,671.12, +8,732.56]|
|project|weighted|+3,265.18|[-1,685.75, +8,721.40]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R114",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R91 additive CatBoost and R104 offset NN supplied strong contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R114 minus v22",
  "fallback": {
    "address": 0,
    "road": 0,
    "singleton": 0
  },
  "methods": {
    "month": {
      "family_index": 1,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": 3315.5807149606744,
          "candidate_mae": 1410687.8285144023,
          "control_mae": 1407387.0708058395,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0,
          "percentile_2_5": 1844.4354095535236,
          "percentile_97_5": 4888.071929120167,
          "point_delta": 3300.757708562771,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 3250.6883894041366,
          "candidate_mae": 1402982.9283901912,
          "control_mae": 1399717.744053713,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0,
          "percentile_2_5": 1797.2083320613542,
          "percentile_97_5": 5116.425487368423,
          "point_delta": 3265.1843364781234,
          "valid_replicates": 2000
        }
      },
      "support": {
        "block_length_months": 3,
        "blocks_per_replicate": 4,
        "circular_boundary_assumption": true,
        "months_with_rows": 12,
        "rows_by_relative_month": [
          6145,
          6235,
          5228,
          5046,
          5881,
          3266,
          5055,
          7516,
          9057,
          9401,
          11490,
          6914
        ],
        "stratum_rows_by_relative_month": {
          "missing": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
          ],
          "new_nonmissing": [
            1873,
            2516,
            2453,
            2908,
            3879,
            2127,
            3277,
            5270,
            6645,
            7274,
            8899,
            5132
          ],
          "seen": [
            4272,
            3719,
            2775,
            2138,
            2002,
            1139,
            1778,
            2246,
            2412,
            2127,
            2591,
            1782
          ]
        }
      }
    },
    "project": {
      "family_index": 0,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": 3290.6523626756994,
          "candidate_mae": 1410687.8285144023,
          "control_mae": 1407387.0708058395,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.1195,
          "percentile_2_5": -1671.1237893511075,
          "percentile_97_5": 8732.558205543779,
          "point_delta": 3300.757708562771,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 3264.7811450988047,
          "candidate_mae": 1402982.9283901912,
          "control_mae": 1399717.744053713,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.1205,
          "percentile_2_5": -1685.7483117303111,
          "percentile_97_5": 8721.397518723783,
          "point_delta": 3265.1843364781234,
          "valid_replicates": 2000
        }
      },
      "support": {
        "missing": {
          "clusters": 0,
          "largest_cluster_row_share": null,
          "largest_cluster_rows": 0,
          "row_weight_effective_cluster_count": null,
          "rows": 0
        },
        "new_nonmissing": {
          "clusters": 1944,
          "largest_cluster_row_share": 0.03303159627198438,
          "largest_cluster_rows": 1726,
          "row_weight_effective_cluster_count": 326.58441409315515,
          "rows": 52253
        },
        "seen": {
          "clusters": 1861,
          "largest_cluster_row_share": 0.022877057382422968,
          "largest_cluster_rows": 663,
          "row_weight_effective_cluster_count": 305.65955703842644,
          "rows": 28981
        }
      }
    }
  },
  "replicates": 2000,
  "seed": 20260925,
  "stage_index": 0
}
```

v24完整次對照（不以較弱對照取代v22）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R114",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 279735.0033029721,
      "control_mae": 240685.11008695266,
      "delta": 39049.89321601947,
      "relative_change_pct": 16.224474044909492,
      "rows": 3
    },
    "南投縣": {
      "candidate_mae": 808414.8604907803,
      "control_mae": 815392.6297455619,
      "delta": -6977.769254781539,
      "relative_change_pct": -0.8557557427222402,
      "rows": 659
    },
    "嘉義市": {
      "candidate_mae": 1450461.515995185,
      "control_mae": 1432770.9123328016,
      "delta": 17690.603662383277,
      "relative_change_pct": 1.2347126473680206,
      "rows": 569
    },
    "嘉義縣": {
      "candidate_mae": 797200.1974840396,
      "control_mae": 798121.9796995061,
      "delta": -921.78221546649,
      "relative_change_pct": -0.11549390179851232,
      "rows": 798
    },
    "基隆市": {
      "candidate_mae": 1577200.2137326049,
      "control_mae": 1573023.596581248,
      "delta": 4176.617151356768,
      "relative_change_pct": 0.26551522560971585,
      "rows": 737
    },
    "宜蘭縣": {
      "candidate_mae": 838366.2180950436,
      "control_mae": 833869.7553878439,
      "delta": 4496.462707199622,
      "relative_change_pct": 0.5392284200436382,
      "rows": 2242
    },
    "屏東縣": {
      "candidate_mae": 804453.3665645907,
      "control_mae": 806099.9039080202,
      "delta": -1646.537343429518,
      "relative_change_pct": -0.20425971215813413,
      "rows": 615
    },
    "彰化縣": {
      "candidate_mae": 1086637.69044091,
      "control_mae": 1099316.8196964574,
      "delta": -12679.129255547421,
      "relative_change_pct": -1.1533644376557772,
      "rows": 2246
    },
    "新北市": {
      "candidate_mae": 1146669.0637768428,
      "control_mae": 1175055.059966161,
      "delta": -28385.99618931813,
      "relative_change_pct": -2.415716263554117,
      "rows": 15979
    },
    "新竹市": {
      "candidate_mae": 4517013.17979996,
      "control_mae": 3988384.5966167687,
      "delta": 528628.5831831913,
      "relative_change_pct": 13.254202807613177,
      "rows": 1250
    },
    "新竹縣": {
      "candidate_mae": 2976782.781626223,
      "control_mae": 2973954.3281137035,
      "delta": 2828.453512519598,
      "relative_change_pct": 0.09510749663440889,
      "rows": 3639
    },
    "桃園市": {
      "candidate_mae": 977523.1854765037,
      "control_mae": 968876.870646954,
      "delta": 8646.314829549636,
      "relative_change_pct": 0.8924059487328023,
      "rows": 15846
    },
    "澎湖縣": {
      "candidate_mae": 911164.7712141208,
      "control_mae": 875582.3630456572,
      "delta": 35582.40816846362,
      "relative_change_pct": 4.06385620248134,
      "rows": 28
    },
    "臺中市": {
      "candidate_mae": 1583170.8214015488,
      "control_mae": 1572125.9434656193,
      "delta": 11044.877935929457,
      "relative_change_pct": 0.7025440920834849,
      "rows": 14719
    },
    "臺北市": {
      "candidate_mae": 4393228.320294621,
      "control_mae": 4511801.4476898415,
      "delta": -118573.1273952201,
      "relative_change_pct": -2.628066167582188,
      "rows": 2698
    },
    "臺南市": {
      "candidate_mae": 1232486.9943061038,
      "control_mae": 1239277.1706731338,
      "delta": -6790.176367030013,
      "relative_change_pct": -0.5479142622583628,
      "rows": 7530
    },
    "臺東縣": {
      "candidate_mae": 708264.886373608,
      "control_mae": 642748.5280153912,
      "delta": 65516.35835821682,
      "relative_change_pct": 10.193155721493612,
      "rows": 265
    },
    "花蓮縣": {
      "candidate_mae": 562155.679154983,
      "control_mae": 558290.847066315,
      "delta": 3864.832088668016,
      "relative_change_pct": 0.6922614097968443,
      "rows": 535
    },
    "苗栗縣": {
      "candidate_mae": 1301468.9938131392,
      "control_mae": 1309189.6028082767,
      "delta": -7720.608995137503,
      "relative_change_pct": -0.5897242827606035,
      "rows": 2402
    },
    "金門縣": {
      "candidate_mae": 669812.9683022129,
      "control_mae": 664978.0807066441,
      "delta": 4834.887595568784,
      "relative_change_pct": 0.7270747316108455,
      "rows": 29
    },
    "雲林縣": {
      "candidate_mae": 919043.8224982439,
      "control_mae": 900864.7686889353,
      "delta": 18179.053809308563,
      "relative_change_pct": 2.0179559064969617,
      "rows": 1350
    },
    "高雄市": {
      "candidate_mae": 986005.3947094999,
      "control_mae": 970391.5011320283,
      "delta": 15613.893577471608,
      "relative_change_pct": 1.609030330465274,
      "rows": 7095
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1244977.9607451647,
      "control_mae": 1257065.2649268839,
      "delta": -12087.304181719199,
      "relative_change_pct": -0.961549453235608,
      "rows": 17608
    },
    "Q2": {
      "candidate_mae": 1318900.654326374,
      "control_mae": 1304412.7913579252,
      "delta": 14487.862968448782,
      "relative_change_pct": 1.1106808415583358,
      "rows": 14193
    },
    "Q3": {
      "candidate_mae": 1454653.7124260683,
      "control_mae": 1449548.9071540441,
      "delta": 5104.805272024125,
      "relative_change_pct": 0.3521650940392614,
      "rows": 22487
    },
    "Q4": {
      "candidate_mae": 1530627.4812795792,
      "control_mae": 1524529.5621492898,
      "delta": 6097.9191302894615,
      "relative_change_pct": 0.399986939032693,
      "rows": 26946
    }
  },
  "months": {
    "2022-08": {
      "candidate_mae": 945361.8602149428,
      "control_mae": 950683.2454188134,
      "delta": -5321.385203870595,
      "relative_change_pct": -0.559743240402466,
      "rows": 3937
    },
    "2022-09": {
      "candidate_mae": 1184877.3618254536,
      "control_mae": 1183834.4147522436,
      "delta": 1042.9470732100308,
      "relative_change_pct": 0.08809906691454834,
      "rows": 6480
    },
    "2022-10": {
      "candidate_mae": 1468721.7341369472,
      "control_mae": 1501876.047428305,
      "delta": -33154.31329135783,
      "relative_change_pct": -2.2075266030195158,
      "rows": 5654
    },
    "2022-11": {
      "candidate_mae": 1542729.0784832514,
      "control_mae": 1518956.0042070919,
      "delta": 23773.074276159517,
      "relative_change_pct": 1.5650930119315252,
      "rows": 4732
    },
    "2022-12": {
      "candidate_mae": 1228709.334426861,
      "control_mae": 1228901.1954487665,
      "delta": -191.86102190543897,
      "relative_change_pct": -0.015612404204340914,
      "rows": 6122
    },
    "2023-01": {
      "candidate_mae": 1264051.6697068105,
      "control_mae": 1241965.4282693113,
      "delta": 22086.24143749918,
      "relative_change_pct": 1.7783298097336357,
      "rows": 3545
    },
    "2023-02": {
      "candidate_mae": 1178074.799862833,
      "control_mae": 1181921.8231404028,
      "delta": -3847.023277569795,
      "relative_change_pct": -0.3254888100253649,
      "rows": 4227
    },
    "2023-03": {
      "candidate_mae": 1421645.3866017726,
      "control_mae": 1425446.287641478,
      "delta": -3800.901039705379,
      "relative_change_pct": -0.26664638805817736,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1521738.5301028534,
      "control_mae": 1508305.4461258438,
      "delta": 13433.083977009635,
      "relative_change_pct": 0.8906076691238614,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1543350.0934970377,
      "control_mae": 1532629.9115433628,
      "delta": 10720.18195367488,
      "relative_change_pct": 0.699463182398654,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1553994.908622252,
      "control_mae": 1543362.271613415,
      "delta": 10632.637008836959,
      "relative_change_pct": 0.688926845265027,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1535312.1601923488,
      "control_mae": 1539075.911771725,
      "delta": -3763.7515793761704,
      "relative_change_pct": -0.24454619493352245,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1454847.6112415385,
      "control_mae": 1454058.0801178366,
      "delta": 789.5311237019487,
      "relative_change_pct": 0.0542984585346113,
      "rows": 2324
    }
  },
  "overall_mae": {
    "candidate_mae": 1410687.8285144106,
    "control_mae": 1407340.7168109824,
    "delta": 3347.111703428207,
    "relative_change_pct": 0.23783236450464695
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 735254.7059752645,
      "control_mae": 726223.9901538457,
      "delta": 9030.715821418795,
      "relative_change_pct": 1.2435165932077923,
      "rows": 22139
    },
    {
      "band": "10-20m",
      "candidate_mae": 1120037.9464687384,
      "control_mae": 1119987.3894261469,
      "delta": 50.557042591506615,
      "relative_change_pct": 0.004514072485888503,
      "rows": 43914
    },
    {
      "band": "20-50m",
      "candidate_mae": 2677414.468703014,
      "control_mae": 2676120.810418359,
      "delta": 1293.6582846548408,
      "relative_change_pct": 0.048340802837395176,
      "rows": 14197
    },
    {
      "band": "50-100m",
      "candidate_mae": 6559736.257160487,
      "control_mae": 6578422.082288335,
      "delta": -18685.825127848424,
      "relative_change_pct": -0.28404722126538395,
      "rows": 802
    },
    {
      "band": ">=100m",
      "candidate_mae": 32200566.232448697,
      "control_mae": 31835906.574960757,
      "delta": 364659.65748794004,
      "relative_change_pct": 1.1454351288200737,
      "rows": 182
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 4517013.17979996,
      "control_mae": 3988384.5966167687,
      "delta": 528628.5831831913,
      "group": "新竹市",
      "relative_change_pct": 13.254202807613177,
      "rows": 1250,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 1051881.0729116793,
      "control_mae": 1056814.584844697,
      "delta": -4933.511933017755,
      "relative_change_pct": -0.46682852448925594,
      "rows": 6145
    },
    "1": {
      "candidate_mae": 1196331.5165868,
      "control_mae": 1196559.960172717,
      "delta": -228.44358591688797,
      "relative_change_pct": -0.019091695654258174,
      "rows": 6235
    },
    "10": {
      "candidate_mae": 1564979.41939387,
      "control_mae": 1557748.2179694306,
      "delta": 7231.201424439438,
      "relative_change_pct": 0.46420861478278663,
      "rows": 11490
    },
    "11": {
      "candidate_mae": 1535090.5591197854,
      "control_mae": 1536870.8291656736,
      "delta": -1780.2700458881445,
      "relative_change_pct": -0.11583732426326329,
      "rows": 6914
    },
    "2": {
      "candidate_mae": 1529960.9284315023,
      "control_mae": 1564599.8870093762,
      "delta": -34638.958577873884,
      "relative_change_pct": -2.2139180032847787,
      "rows": 5228
    },
    "3": {
      "candidate_mae": 1450220.1090858441,
      "control_mae": 1423037.1772045153,
      "delta": 27182.931881328812,
      "relative_change_pct": 1.9102053211798942,
      "rows": 5046
    },
    "4": {
      "candidate_mae": 1220647.0472940651,
      "control_mae": 1216912.2856501525,
      "delta": 3734.761643912643,
      "relative_change_pct": 0.3069047529516307,
      "rows": 5881
    },
    "5": {
      "candidate_mae": 1292933.5674435585,
      "control_mae": 1278696.876809706,
      "delta": 14236.6906338525,
      "relative_change_pct": 1.1133749438234677,
      "rows": 3266
    },
    "6": {
      "candidate_mae": 1188681.448570989,
      "control_mae": 1193017.763002033,
      "delta": -4336.31443104404,
      "relative_change_pct": -0.36347442305740846,
      "rows": 5055
    },
    "7": {
      "candidate_mae": 1485781.583514759,
      "control_mae": 1487234.18211577,
      "delta": -1452.5986010108609,
      "relative_change_pct": -0.09767114140318939,
      "rows": 7516
    },
    "8": {
      "candidate_mae": 1580459.4595123406,
      "control_mae": 1560492.9821896846,
      "delta": 19966.477322655963,
      "relative_change_pct": 1.279498052893451,
      "rows": 9057
    },
    "9": {
      "candidate_mae": 1475344.756756539,
      "control_mae": 1468927.3561406387,
      "delta": 6417.400615900289,
      "relative_change_pct": 0.43687664941858917,
      "rows": 9401
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1702601.9650131522,
      "control_mae": 1696977.6628507746,
      "delta": 5624.302162377629,
      "relative_change_pct": 0.3314305359169721,
      "rows": 52253
    },
    "seen": {
      "candidate_mae": 884364.0517479517,
      "control_mae": 885122.7346358591,
      "delta": -758.6828879074892,
      "relative_change_pct": -0.08571499275968915,
      "rows": 28981
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 1402982.9283901993,
    "control_mae": 1399695.9217754127,
    "delta": 3287.0066147865728,
    "relative_change_pct": 0.23483719311100396
  }
}
```

完整城市／月份／新舊案／真價帶／季度／誤差尾部，僅事後診斷，不作未知test路由：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 279735.0033029721,
      "rows": 3
    },
    "南投縣": {
      "mae": 808414.8604907803,
      "rows": 659
    },
    "嘉義市": {
      "mae": 1450461.515995185,
      "rows": 569
    },
    "嘉義縣": {
      "mae": 797200.1974840396,
      "rows": 798
    },
    "基隆市": {
      "mae": 1577200.2137326049,
      "rows": 737
    },
    "宜蘭縣": {
      "mae": 838366.2180950436,
      "rows": 2242
    },
    "屏東縣": {
      "mae": 804453.3665645907,
      "rows": 615
    },
    "彰化縣": {
      "mae": 1086637.69044091,
      "rows": 2246
    },
    "新北市": {
      "mae": 1146669.0637768428,
      "rows": 15979
    },
    "新竹市": {
      "mae": 4517013.17979996,
      "rows": 1250
    },
    "新竹縣": {
      "mae": 2976782.781626223,
      "rows": 3639
    },
    "桃園市": {
      "mae": 977523.1854765037,
      "rows": 15846
    },
    "澎湖縣": {
      "mae": 911164.7712141208,
      "rows": 28
    },
    "臺中市": {
      "mae": 1583170.8214015488,
      "rows": 14719
    },
    "臺北市": {
      "mae": 4393228.320294621,
      "rows": 2698
    },
    "臺南市": {
      "mae": 1232486.9943061038,
      "rows": 7530
    },
    "臺東縣": {
      "mae": 708264.886373608,
      "rows": 265
    },
    "花蓮縣": {
      "mae": 562155.679154983,
      "rows": 535
    },
    "苗栗縣": {
      "mae": 1301468.9938131392,
      "rows": 2402
    },
    "金門縣": {
      "mae": 669812.9683022129,
      "rows": 29
    },
    "雲林縣": {
      "mae": 919043.8224982439,
      "rows": 1350
    },
    "高雄市": {
      "mae": 986005.3947094999,
      "rows": 7095
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1244977.9607451647,
      "rows": 17608
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1318900.654326374,
      "rows": 14193
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1454653.7124260683,
      "rows": 22487
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1530627.4812795792,
      "rows": 26946
    }
  },
  "months": {
    "2022-08": {
      "mae": 945361.8602149428,
      "rows": 3937
    },
    "2022-09": {
      "mae": 1184877.3618254536,
      "rows": 6480
    },
    "2022-10": {
      "mae": 1468721.7341369472,
      "rows": 5654
    },
    "2022-11": {
      "mae": 1542729.0784832514,
      "rows": 4732
    },
    "2022-12": {
      "mae": 1228709.334426861,
      "rows": 6122
    },
    "2023-01": {
      "mae": 1264051.6697068105,
      "rows": 3545
    },
    "2023-02": {
      "mae": 1178074.799862833,
      "rows": 4227
    },
    "2023-03": {
      "mae": 1421645.3866017726,
      "rows": 7500
    },
    "2023-04": {
      "mae": 1521738.5301028534,
      "rows": 8639
    },
    "2023-05": {
      "mae": 1543350.0934970377,
      "rows": 8750
    },
    "2023-06": {
      "mae": 1553994.908622252,
      "rows": 13180
    },
    "2023-07": {
      "mae": 1535312.1601923488,
      "rows": 6144
    },
    "2023-08": {
      "mae": 1454847.6112415385,
      "rows": 2324
    }
  },
  "overall_mae": 1410687.8285144106,
  "price_bands": [
    {
      "absolute_error_share": 0.1420453611403258,
      "absolute_error_sum": 16277803935.586382,
      "band": "<10m",
      "mae": 735254.7059752645,
      "mean_signed_error_prediction_minus_truth": 391641.92053202953,
      "rows": 22139,
      "underprediction_fraction": 0.3308640860020778
    },
    {
      "absolute_error_share": 0.42920717789576285,
      "absolute_error_sum": 49185346381.22818,
      "band": "10-20m",
      "mae": 1120037.9464687384,
      "mean_signed_error_prediction_minus_truth": -34564.23974608424,
      "rows": 43914,
      "underprediction_fraction": 0.4788450152570934
    },
    {
      "absolute_error_share": 0.3316984410972084,
      "absolute_error_sum": 38011253212.17669,
      "band": "20-50m",
      "mae": 2677414.468703014,
      "mean_signed_error_prediction_minus_truth": -849816.3541955232,
      "rows": 14197,
      "underprediction_fraction": 0.5381418609565402
    },
    {
      "absolute_error_share": 0.04590838221637959,
      "absolute_error_sum": 5260908478.24271,
      "band": "50-100m",
      "mae": 6559736.257160487,
      "mean_signed_error_prediction_minus_truth": -1584947.3749486562,
      "rows": 802,
      "underprediction_fraction": 0.4975062344139651
    },
    {
      "absolute_error_share": 0.05114063765032333,
      "absolute_error_sum": 5860503054.305663,
      "band": ">=100m",
      "mae": 32200566.232448697,
      "mean_signed_error_prediction_minus_truth": 1736605.3693900278,
      "rows": 182,
      "underprediction_fraction": 0.3791208791208791
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 1051881.0729116793,
      "rows": 6145
    },
    "1": {
      "mae": 1196331.5165868,
      "rows": 6235
    },
    "10": {
      "mae": 1564979.41939387,
      "rows": 11490
    },
    "11": {
      "mae": 1535090.5591197854,
      "rows": 6914
    },
    "2": {
      "mae": 1529960.9284315023,
      "rows": 5228
    },
    "3": {
      "mae": 1450220.1090858441,
      "rows": 5046
    },
    "4": {
      "mae": 1220647.0472940651,
      "rows": 5881
    },
    "5": {
      "mae": 1292933.5674435585,
      "rows": 3266
    },
    "6": {
      "mae": 1188681.448570989,
      "rows": 5055
    },
    "7": {
      "mae": 1485781.583514759,
      "rows": 7516
    },
    "8": {
      "mae": 1580459.4595123406,
      "rows": 9057
    },
    "9": {
      "mae": 1475344.756756539,
      "rows": 9401
    }
  },
  "rows": 81234,
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 1702601.9650131522,
      "rows": 52253
    },
    "seen": {
      "mae": 884364.0517479517,
      "rows": 28981
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 792276.8470990416,
      "0.9": 3151927.7362754783,
      "0.95": 4804067.284696166,
      "0.99": 8989868.69162768
    },
    "absolute_error_sum": 114595815061.53963,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 527435029.011781,
    "mean_signed_error_prediction_minus_truth": -72225.93918241897,
    "top10_absolute_error_sum": 2235538430.938229,
    "top10_error_rows": 10,
    "top10_error_share": 0.019508028541336455,
    "top1pct_absolute_error_sum": 14027425273.445702,
    "top1pct_error_rows": 813,
    "top1pct_error_share": 0.12240783196064157,
    "underprediction_fraction": 0.44883915601841595
  },
  "weighted_mae": 1402982.9283901993
}
```

## confirmation完整診斷

fit實測29.593秒。

|層|總價MAE|
|---|---:|
|base|1,629,629.94|
|native|1,676,061.68|
|reference|1,395,835.16|
|spatial|1,393,281.93|
|median|1,388,411.32|

未投影總價與固定非負投影：

```json
{
  "negative_residual_not_equivalent_to_negative_total": true,
  "overall_mae": 1676337.3011527935,
  "projected_max_TWD": 2178426101.6845703,
  "projected_min_TWD": 0.0,
  "projected_zero_fraction": 5.2357195748595705e-05,
  "projected_zero_rows": 7,
  "projection_changes_count": 7,
  "projection_max_correction_TWD": 7637391.567230225,
  "projection_mean_correction_TWD": 275.62255480321966,
  "query_y_used_for_diagnostics": false,
  "raw_max_TWD": 2178426101.6845703,
  "raw_min_TWD": -7637391.567230225,
  "raw_negative_total_fraction": 5.2357195748595705e-05,
  "raw_negative_total_rows": 7,
  "raw_zero_total_rows": 0
}
```

所有季度／至少500列群退步超5%（群可重疊，不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+5,182.91|[-594.30, +10,328.20]|
|month|weighted|+5,340.29|[-2,550.49, +9,984.02]|
|project|overall|+5,182.91|[-2,850.84, +13,998.71]|
|project|weighted|+5,340.29|[-3,124.01, +14,661.17]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R114",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R91 additive CatBoost and R104 offset NN supplied strong contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R114 minus v22",
  "fallback": {
    "address": 0,
    "road": 0,
    "singleton": 0
  },
  "methods": {
    "month": {
      "family_index": 1,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": 4997.692775252048,
          "candidate_mae": 1388411.320344285,
          "control_mae": 1383228.4054499408,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0445,
          "percentile_2_5": -594.29537553853,
          "percentile_97_5": 10328.199958733981,
          "point_delta": 5182.914894344285,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 4876.650449859588,
          "candidate_mae": 1452493.7091693885,
          "control_mae": 1447153.4154201136,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0875,
          "percentile_2_5": -2550.491744492983,
          "percentile_97_5": 9984.021034556934,
          "point_delta": 5340.293749274919,
          "valid_replicates": 2000
        }
      },
      "support": {
        "block_length_months": 3,
        "blocks_per_replicate": 4,
        "circular_boundary_assumption": true,
        "months_with_rows": 12,
        "rows_by_relative_month": [
          7252,
          9394,
          9933,
          10809,
          9935,
          10413,
          8995,
          12434,
          14629,
          14519,
          14072,
          11312
        ],
        "stratum_rows_by_relative_month": {
          "missing": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
          ],
          "new_nonmissing": [
            787,
            2706,
            3778,
            5451,
            5004,
            5959,
            5017,
            7708,
            9840,
            10260,
            10630,
            8956
          ],
          "seen": [
            6465,
            6688,
            6155,
            5358,
            4931,
            4454,
            3978,
            4726,
            4789,
            4259,
            3442,
            2356
          ]
        }
      }
    },
    "project": {
      "family_index": 0,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": 5182.860325308284,
          "candidate_mae": 1388411.320344285,
          "control_mae": 1383228.4054499408,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.109,
          "percentile_2_5": -2850.8377443610398,
          "percentile_97_5": 13998.71064422402,
          "point_delta": 5182.914894344285,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 5361.557854384171,
          "candidate_mae": 1452493.7091693885,
          "control_mae": 1447153.4154201136,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.116,
          "percentile_2_5": -3124.0101433581613,
          "percentile_97_5": 14661.1691584163,
          "point_delta": 5340.293749274919,
          "valid_replicates": 2000
        }
      },
      "support": {
        "missing": {
          "clusters": 0,
          "largest_cluster_row_share": null,
          "largest_cluster_rows": 0,
          "row_weight_effective_cluster_count": null,
          "rows": 0
        },
        "new_nonmissing": {
          "clusters": 2075,
          "largest_cluster_row_share": 0.01755677039529016,
          "largest_cluster_rows": 1336,
          "row_weight_effective_cluster_count": 383.30050459350207,
          "rows": 76096
        },
        "seen": {
          "clusters": 2403,
          "largest_cluster_row_share": 0.00946164129094981,
          "largest_cluster_rows": 545,
          "row_weight_effective_cluster_count": 533.2518158646615,
          "rows": 57601
        }
      }
    }
  },
  "replicates": 2000,
  "seed": 20260925,
  "stage_index": 1
}
```

v24完整次對照（不以較弱對照取代v22）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R114",
  "cities": {
    "南投縣": {
      "candidate_mae": 957218.693742541,
      "control_mae": 965273.7684037857,
      "delta": -8055.074661244638,
      "relative_change_pct": -0.8344860209518409,
      "rows": 1015
    },
    "嘉義市": {
      "candidate_mae": 1172548.5514047272,
      "control_mae": 1192323.4691625503,
      "delta": -19774.917757823132,
      "relative_change_pct": -1.6585195435021,
      "rows": 689
    },
    "嘉義縣": {
      "candidate_mae": 1417547.8800514913,
      "control_mae": 1401441.997463974,
      "delta": 16105.882587517379,
      "relative_change_pct": 1.149236473336914,
      "rows": 2251
    },
    "基隆市": {
      "candidate_mae": 1809338.403956224,
      "control_mae": 1811344.9669182824,
      "delta": -2006.562962058466,
      "relative_change_pct": -0.11077751608365999,
      "rows": 1100
    },
    "宜蘭縣": {
      "candidate_mae": 940228.0237888908,
      "control_mae": 940693.1296201872,
      "delta": -465.105831296416,
      "relative_change_pct": -0.04944288595837905,
      "rows": 2907
    },
    "屏東縣": {
      "candidate_mae": 754927.0465469316,
      "control_mae": 757610.0933880565,
      "delta": -2683.0468411248876,
      "relative_change_pct": -0.3541461319669352,
      "rows": 1013
    },
    "彰化縣": {
      "candidate_mae": 1062807.8011626843,
      "control_mae": 1068085.3267788775,
      "delta": -5277.52561619319,
      "relative_change_pct": -0.49411086210772187,
      "rows": 3295
    },
    "新北市": {
      "candidate_mae": 1329141.0744820882,
      "control_mae": 1308034.3659481003,
      "delta": 21106.70853398787,
      "relative_change_pct": 1.6136203362431636,
      "rows": 21822
    },
    "新竹市": {
      "candidate_mae": 2555220.7873846884,
      "control_mae": 2414804.346618525,
      "delta": 140416.44076616317,
      "relative_change_pct": 5.814816465888415,
      "rows": 2131
    },
    "新竹縣": {
      "candidate_mae": 2013143.4007731555,
      "control_mae": 2023872.173760328,
      "delta": -10728.772987172473,
      "relative_change_pct": -0.5301111960662295,
      "rows": 5589
    },
    "桃園市": {
      "candidate_mae": 1218797.586575854,
      "control_mae": 1236542.7211960873,
      "delta": -17745.134620233206,
      "relative_change_pct": -1.4350603756794291,
      "rows": 26327
    },
    "澎湖縣": {
      "candidate_mae": 2723463.4595907903,
      "control_mae": 2558605.6252502515,
      "delta": 164857.83434053883,
      "relative_change_pct": 6.4432686582721965,
      "rows": 30
    },
    "臺中市": {
      "candidate_mae": 1422792.1808409446,
      "control_mae": 1396562.2952675254,
      "delta": 26229.885573419277,
      "relative_change_pct": 1.8781751206017403,
      "rows": 27157
    },
    "臺北市": {
      "candidate_mae": 4214247.267790878,
      "control_mae": 4016190.85969717,
      "delta": 198056.4080937081,
      "relative_change_pct": 4.931449102213285,
      "rows": 4957
    },
    "臺南市": {
      "candidate_mae": 911051.3625864057,
      "control_mae": 924730.213914743,
      "delta": -13678.851328337332,
      "relative_change_pct": -1.479226170239364,
      "rows": 9822
    },
    "臺東縣": {
      "candidate_mae": 534336.0387534925,
      "control_mae": 511517.47384219157,
      "delta": 22818.564911300957,
      "relative_change_pct": 4.460955114573608,
      "rows": 115
    },
    "花蓮縣": {
      "candidate_mae": 619437.8218741323,
      "control_mae": 617584.5960098031,
      "delta": 1853.2258643291425,
      "relative_change_pct": 0.3000764391312191,
      "rows": 583
    },
    "苗栗縣": {
      "candidate_mae": 1489787.3882472499,
      "control_mae": 1487508.760371721,
      "delta": 2278.627875528997,
      "relative_change_pct": 0.1531841651110397,
      "rows": 4001
    },
    "金門縣": {
      "candidate_mae": 783762.8945449226,
      "control_mae": 790515.819334728,
      "delta": -6752.9247898053145,
      "relative_change_pct": -0.8542428405149884,
      "rows": 86
    },
    "雲林縣": {
      "candidate_mae": 901573.5381207411,
      "control_mae": 920111.576360945,
      "delta": -18538.038240203867,
      "relative_change_pct": -2.0147598091877166,
      "rows": 2364
    },
    "高雄市": {
      "candidate_mae": 1017857.9084637222,
      "control_mae": 1000969.9302470884,
      "delta": 16887.978216633783,
      "relative_change_pct": 1.6871613927967848,
      "rows": 16443
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 914108.7700330804,
      "control_mae": 912993.6515105661,
      "delta": 1115.1185225142399,
      "relative_change_pct": 0.12213869402807502,
      "rows": 26579
    },
    "Q2": {
      "candidate_mae": 1174288.399580439,
      "control_mae": 1165407.6059651442,
      "delta": 8880.793615294853,
      "relative_change_pct": 0.762033263712925,
      "rows": 31157
    },
    "Q3": {
      "candidate_mae": 1496052.3219708407,
      "control_mae": 1475619.1470166727,
      "delta": 20433.174954168033,
      "relative_change_pct": 1.3847187463973156,
      "rows": 37236
    },
    "Q4": {
      "candidate_mae": 1782724.9417877123,
      "control_mae": 1757752.2708320958,
      "delta": 24972.670955616515,
      "relative_change_pct": 1.4207161822521666,
      "rows": 38725
    }
  },
  "months": {
    "2023-08": {
      "candidate_mae": 668406.259364738,
      "control_mae": 661692.706617775,
      "delta": 6713.552746962989,
      "relative_change_pct": 1.0146028027540364,
      "rows": 5076
    },
    "2023-09": {
      "candidate_mae": 815375.3023356238,
      "control_mae": 819211.9589262545,
      "delta": -3836.656590630766,
      "relative_change_pct": -0.4683350320788154,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1052160.4694916357,
      "control_mae": 1031792.2770912236,
      "delta": 20368.192400412052,
      "relative_change_pct": 1.974059396706576,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1173620.1902615398,
      "control_mae": 1207667.0396223704,
      "delta": -34046.849360830616,
      "relative_change_pct": -2.819224856172016,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1125536.2497248542,
      "control_mae": 1122539.8496172,
      "delta": 2996.4001076542772,
      "relative_change_pct": 0.26693039972488164,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1206778.9992792052,
      "control_mae": 1177153.3779831533,
      "delta": 29625.62129605189,
      "relative_change_pct": 2.5167171797790884,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1187702.8822450952,
      "control_mae": 1170552.7720134521,
      "delta": 17150.110231643077,
      "relative_change_pct": 1.4651291801345618,
      "rows": 7388
    },
    "2024-03": {
      "candidate_mae": 1412657.8338993972,
      "control_mae": 1396897.8203355782,
      "delta": 15760.013563818997,
      "relative_change_pct": 1.1282152018844838,
      "rows": 12933
    },
    "2024-04": {
      "candidate_mae": 1613496.5243537356,
      "control_mae": 1587211.1045129716,
      "delta": 26285.41984076402,
      "relative_change_pct": 1.6560758531757866,
      "rows": 13856
    },
    "2024-05": {
      "candidate_mae": 1707481.3204483623,
      "control_mae": 1671790.446103836,
      "delta": 35690.874344526324,
      "relative_change_pct": 2.134889239719315,
      "rows": 14480
    },
    "2024-06": {
      "candidate_mae": 1799750.2799978163,
      "control_mae": 1771497.5397129185,
      "delta": 28252.740284897853,
      "relative_change_pct": 1.594850664566905,
      "rows": 14671
    },
    "2024-07": {
      "candidate_mae": 1772721.8213127358,
      "control_mae": 1756731.8339828255,
      "delta": 15989.987329910276,
      "relative_change_pct": 0.910212191786729,
      "rows": 12090
    },
    "2024-08": {
      "candidate_mae": 1897347.0171341589,
      "control_mae": 1893707.7653277898,
      "delta": 3639.2518063690513,
      "relative_change_pct": 0.19217599848301398,
      "rows": 3109
    }
  },
  "overall_mae": {
    "candidate_mae": 1388411.3203443035,
    "control_mae": 1373195.9153073074,
    "delta": 15215.405036996119,
    "relative_change_pct": 1.108028713702594
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 647651.0199898747,
      "control_mae": 652232.5472355905,
      "delta": -4581.527245715843,
      "relative_change_pct": -0.7024376911477505,
      "rows": 25585
    },
    {
      "band": "10-20m",
      "candidate_mae": 1042930.237904171,
      "control_mae": 1040802.0332451425,
      "delta": 2128.2046590284444,
      "relative_change_pct": 0.20447737331881088,
      "rows": 74109
    },
    {
      "band": "20-50m",
      "candidate_mae": 2322569.29112561,
      "control_mae": 2283352.845939235,
      "delta": 39216.445186375175,
      "relative_change_pct": 1.7174938711780166,
      "rows": 31959
    },
    {
      "band": "50-100m",
      "candidate_mae": 6462212.330912766,
      "control_mae": 6217166.085579387,
      "delta": 245046.24533337913,
      "relative_change_pct": 3.9414460215524856,
      "rows": 1725
    },
    {
      "band": ">=100m",
      "candidate_mae": 20035898.755120076,
      "control_mae": 19039879.576983564,
      "delta": 996019.1781365126,
      "relative_change_pct": 5.231226248618476,
      "rows": 319
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 2555220.7873846884,
      "control_mae": 2414804.346618525,
      "delta": 140416.44076616317,
      "group": "新竹市",
      "relative_change_pct": 5.814816465888415,
      "rows": 2131,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 686947.1909904138,
      "control_mae": 678922.6899683854,
      "delta": 8024.5010220283875,
      "relative_change_pct": 1.1819462128157854,
      "rows": 7252
    },
    "1": {
      "candidate_mae": 884955.3903250569,
      "control_mae": 889826.2099094184,
      "delta": -4870.81958436151,
      "relative_change_pct": -0.5473899880806327,
      "rows": 9394
    },
    "10": {
      "candidate_mae": 1773393.8100696837,
      "control_mae": 1753961.107622659,
      "delta": 19432.70244702464,
      "relative_change_pct": 1.1079323459665515,
      "rows": 14072
    },
    "11": {
      "candidate_mae": 1818541.3827853964,
      "control_mae": 1798487.2970159906,
      "delta": 20054.08576940582,
      "relative_change_pct": 1.1150529560413969,
      "rows": 11312
    },
    "2": {
      "candidate_mae": 1107528.947239825,
      "control_mae": 1105797.1911768373,
      "delta": 1731.7560629877262,
      "relative_change_pct": 0.15660702313276056,
      "rows": 9933
    },
    "3": {
      "candidate_mae": 1163051.5073393565,
      "control_mae": 1182811.8951608276,
      "delta": -19760.387821471086,
      "relative_change_pct": -1.6706280941471472,
      "rows": 10809
    },
    "4": {
      "candidate_mae": 1126746.157597324,
      "control_mae": 1113684.87976537,
      "delta": 13061.277831954183,
      "relative_change_pct": 1.1727983444209031,
      "rows": 9935
    },
    "5": {
      "candidate_mae": 1231312.4793207746,
      "control_mae": 1196689.8803220654,
      "delta": 34622.598998709116,
      "relative_change_pct": 2.893197274250462,
      "rows": 10413
    },
    "6": {
      "candidate_mae": 1267278.551982619,
      "control_mae": 1253733.2273133975,
      "delta": 13545.324669221649,
      "relative_change_pct": 1.0803992726784217,
      "rows": 8995
    },
    "7": {
      "candidate_mae": 1438534.3948636202,
      "control_mae": 1422678.507141905,
      "delta": 15855.887721715262,
      "relative_change_pct": 1.1145095425367046,
      "rows": 12434
    },
    "8": {
      "candidate_mae": 1674784.8170043111,
      "control_mae": 1646182.6876466258,
      "delta": 28602.129357685335,
      "relative_change_pct": 1.7374820894620626,
      "rows": 14629
    },
    "9": {
      "candidate_mae": 1751508.7461596811,
      "control_mae": 1717746.4536748976,
      "delta": 33762.29248478357,
      "relative_change_pct": 1.9654991813578475,
      "rows": 14519
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1815418.4204172594,
      "control_mae": 1792277.4115217023,
      "delta": 23141.00889555714,
      "relative_change_pct": 1.2911510654988205,
      "rows": 76096
    },
    "seen": {
      "candidate_mae": 824297.289560956,
      "control_mae": 819552.3060656169,
      "delta": 4744.983495339053,
      "relative_change_pct": 0.5789726244707984,
      "rows": 57601
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 1452493.7091694083,
    "control_mae": 1436088.882157809,
    "delta": 16404.827011599205,
    "relative_change_pct": 1.1423267191477713
  }
}
```

完整城市／月份／新舊案／真價帶／季度／誤差尾部，僅事後診斷，不作未知test路由：

```json
{
  "cities": {
    "南投縣": {
      "mae": 957218.693742541,
      "rows": 1015
    },
    "嘉義市": {
      "mae": 1172548.5514047272,
      "rows": 689
    },
    "嘉義縣": {
      "mae": 1417547.8800514913,
      "rows": 2251
    },
    "基隆市": {
      "mae": 1809338.403956224,
      "rows": 1100
    },
    "宜蘭縣": {
      "mae": 940228.0237888908,
      "rows": 2907
    },
    "屏東縣": {
      "mae": 754927.0465469316,
      "rows": 1013
    },
    "彰化縣": {
      "mae": 1062807.8011626843,
      "rows": 3295
    },
    "新北市": {
      "mae": 1329141.0744820882,
      "rows": 21822
    },
    "新竹市": {
      "mae": 2555220.7873846884,
      "rows": 2131
    },
    "新竹縣": {
      "mae": 2013143.4007731555,
      "rows": 5589
    },
    "桃園市": {
      "mae": 1218797.586575854,
      "rows": 26327
    },
    "澎湖縣": {
      "mae": 2723463.4595907903,
      "rows": 30
    },
    "臺中市": {
      "mae": 1422792.1808409446,
      "rows": 27157
    },
    "臺北市": {
      "mae": 4214247.267790878,
      "rows": 4957
    },
    "臺南市": {
      "mae": 911051.3625864057,
      "rows": 9822
    },
    "臺東縣": {
      "mae": 534336.0387534925,
      "rows": 115
    },
    "花蓮縣": {
      "mae": 619437.8218741323,
      "rows": 583
    },
    "苗栗縣": {
      "mae": 1489787.3882472499,
      "rows": 4001
    },
    "金門縣": {
      "mae": 783762.8945449226,
      "rows": 86
    },
    "雲林縣": {
      "mae": 901573.5381207411,
      "rows": 2364
    },
    "高雄市": {
      "mae": 1017857.9084637222,
      "rows": 16443
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 914108.7700330804,
      "rows": 26579
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1174288.399580439,
      "rows": 31157
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1496052.3219708407,
      "rows": 37236
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1782724.9417877123,
      "rows": 38725
    }
  },
  "months": {
    "2023-08": {
      "mae": 668406.259364738,
      "rows": 5076
    },
    "2023-09": {
      "mae": 815375.3023356238,
      "rows": 8602
    },
    "2023-10": {
      "mae": 1052160.4694916357,
      "rows": 9855
    },
    "2023-11": {
      "mae": 1173620.1902615398,
      "rows": 10336
    },
    "2023-12": {
      "mae": 1125536.2497248542,
      "rows": 10416
    },
    "2024-01": {
      "mae": 1206778.9992792052,
      "rows": 10885
    },
    "2024-02": {
      "mae": 1187702.8822450952,
      "rows": 7388
    },
    "2024-03": {
      "mae": 1412657.8338993972,
      "rows": 12933
    },
    "2024-04": {
      "mae": 1613496.5243537356,
      "rows": 13856
    },
    "2024-05": {
      "mae": 1707481.3204483623,
      "rows": 14480
    },
    "2024-06": {
      "mae": 1799750.2799978163,
      "rows": 14671
    },
    "2024-07": {
      "mae": 1772721.8213127358,
      "rows": 12090
    },
    "2024-08": {
      "mae": 1897347.0171341589,
      "rows": 3109
    }
  },
  "overall_mae": 1388411.3203443035,
  "price_bands": [
    {
      "absolute_error_share": 0.08926612174001275,
      "absolute_error_sum": 16570151346.440943,
      "band": "<10m",
      "mae": 647651.0199898747,
      "mean_signed_error_prediction_minus_truth": 208288.93272924668,
      "rows": 25585,
      "underprediction_fraction": 0.38522571819425444
    },
    {
      "absolute_error_share": 0.4163766857462914,
      "absolute_error_sum": 77290517000.84021,
      "band": "10-20m",
      "mae": 1042930.237904171,
      "mean_signed_error_prediction_minus_truth": -200510.5425686011,
      "rows": 74109,
      "underprediction_fraction": 0.5275472614662187
    },
    {
      "absolute_error_share": 0.399872974211905,
      "absolute_error_sum": 74226991975.08337,
      "band": "20-50m",
      "mae": 2322569.29112561,
      "mean_signed_error_prediction_minus_truth": -1105437.2353923484,
      "rows": 31959,
      "underprediction_fraction": 0.6256453581150849
    },
    {
      "absolute_error_share": 0.06005242019226196,
      "absolute_error_sum": 11147316270.824522,
      "band": "50-100m",
      "mae": 6462212.330912766,
      "mean_signed_error_prediction_minus_truth": -1544384.0488884293,
      "rows": 1725,
      "underprediction_fraction": 0.5350724637681159
    },
    {
      "absolute_error_share": 0.03443179810952889,
      "absolute_error_sum": 6391451702.883305,
      "band": ">=100m",
      "mae": 20035898.755120076,
      "mean_signed_error_prediction_minus_truth": -6738452.649180267,
      "rows": 319,
      "underprediction_fraction": 0.5109717868338558
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 686947.1909904138,
      "rows": 7252
    },
    "1": {
      "mae": 884955.3903250569,
      "rows": 9394
    },
    "10": {
      "mae": 1773393.8100696837,
      "rows": 14072
    },
    "11": {
      "mae": 1818541.3827853964,
      "rows": 11312
    },
    "2": {
      "mae": 1107528.947239825,
      "rows": 9933
    },
    "3": {
      "mae": 1163051.5073393565,
      "rows": 10809
    },
    "4": {
      "mae": 1126746.157597324,
      "rows": 9935
    },
    "5": {
      "mae": 1231312.4793207746,
      "rows": 10413
    },
    "6": {
      "mae": 1267278.551982619,
      "rows": 8995
    },
    "7": {
      "mae": 1438534.3948636202,
      "rows": 12434
    },
    "8": {
      "mae": 1674784.8170043111,
      "rows": 14629
    },
    "9": {
      "mae": 1751508.7461596811,
      "rows": 14519
    }
  },
  "rows": 133697,
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 1815418.4204172594,
      "rows": 76096
    },
    "seen": {
      "mae": 824297.289560956,
      "rows": 57601
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 774342.7563576177,
      "0.9": 3183987.833501587,
      "0.95": 4554877.529013227,
      "0.99": 8973795.107679972
    },
    "absolute_error_sum": 185626428296.07236,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 318250341.3049927,
    "mean_signed_error_prediction_minus_truth": -371533.1005233011,
    "top10_absolute_error_sum": 1283834362.7448292,
    "top10_error_rows": 10,
    "top10_error_share": 0.0069162261781879775,
    "top1pct_absolute_error_sum": 20137274775.65692,
    "top1pct_error_rows": 1337,
    "top1pct_error_share": 0.10848280043151055,
    "underprediction_fraction": 0.5238187842659148
  },
  "weighted_mae": 1452493.7091694083
}
```

## auxiliary完整診斷

fit實測12.406秒。

|層|總價MAE|
|---|---:|
|base|2,547,366.20|
|native|2,772,987.78|
|reference|2,447,743.62|
|spatial|2,448,737.71|
|median|2,303,762.54|

未投影總價與固定非負投影：

```json
{
  "negative_residual_not_equivalent_to_negative_total": true,
  "overall_mae": 2774108.83879135,
  "projected_max_TWD": 2677156421.661377,
  "projected_min_TWD": 0.0,
  "projected_zero_fraction": 0.00038858507643752785,
  "projected_zero_rows": 41,
  "projection_changes_count": 41,
  "projection_max_correction_TWD": 28410600.662231445,
  "projection_mean_correction_TWD": 1121.0629897799986,
  "query_y_used_for_diagnostics": false,
  "raw_max_TWD": 2677156421.661377,
  "raw_min_TWD": -28410600.662231445,
  "raw_negative_total_fraction": 0.00038858507643752785,
  "raw_negative_total_rows": 41,
  "raw_zero_total_rows": 0
}
```

所有季度／至少500列群退步超5%（群可重疊，不能相加）：

```json
[
  {
    "candidate_mae": 2230942.9743844937,
    "control_mae": 2112509.041619518,
    "delta": 118433.93276497582,
    "group": "嘉義市",
    "relative_change_pct": 5.606316017193495,
    "rows": 731,
    "section": "cities"
  }
]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-5,589.82|[-23,609.52, +3,538.13]|
|month|weighted|-3,188.83|[-14,647.46, +3,454.98]|
|project|overall|-5,589.82|[-26,053.29, +6,732.45]|
|project|weighted|-3,188.83|[-18,493.44, +6,869.72]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R114",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R91 additive CatBoost and R104 offset NN supplied strong contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R114 minus v22",
  "fallback": {
    "address": 0,
    "road": 0,
    "singleton": 0
  },
  "methods": {
    "month": {
      "family_index": 1,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": -6095.6526263412425,
          "candidate_mae": 2303762.5409781705,
          "control_mae": 2309352.3576965467,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.712,
          "percentile_2_5": -23609.519342197116,
          "percentile_97_5": 3538.1296443776228,
          "point_delta": -5589.816718376242,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -3171.352968787526,
          "candidate_mae": 2019976.8311982658,
          "control_mae": 2023165.66227504,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.696,
          "percentile_2_5": -14647.464548685128,
          "percentile_97_5": 3454.976300386479,
          "point_delta": -3188.8310767742805,
          "valid_replicates": 2000
        }
      },
      "support": {
        "block_length_months": 3,
        "blocks_per_replicate": 4,
        "circular_boundary_assumption": true,
        "months_with_rows": 12,
        "rows_by_relative_month": [
          9387,
          12463,
          11844,
          12872,
          9987,
          6310,
          8284,
          9188,
          7187,
          5780,
          5992,
          6217
        ],
        "stratum_rows_by_relative_month": {
          "missing": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
          ],
          "new_nonmissing": [
            6221,
            8890,
            9717,
            11020,
            8980,
            5642,
            7528,
            8530,
            6699,
            5484,
            5703,
            6010
          ],
          "seen": [
            3166,
            3573,
            2127,
            1852,
            1007,
            668,
            756,
            658,
            488,
            296,
            289,
            207
          ]
        }
      }
    },
    "project": {
      "family_index": 0,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": -5644.7050752608075,
          "candidate_mae": 2303762.5409781705,
          "control_mae": 2309352.3576965467,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.6955,
          "percentile_2_5": -26053.288017757353,
          "percentile_97_5": 6732.451480454213,
          "point_delta": -5589.816718376242,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -3214.1653655837163,
          "candidate_mae": 2019976.8311982658,
          "control_mae": 2023165.66227504,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.6395,
          "percentile_2_5": -18493.441693606157,
          "percentile_97_5": 6869.720228859749,
          "point_delta": -3188.8310767742805,
          "valid_replicates": 2000
        }
      },
      "support": {
        "missing": {
          "clusters": 0,
          "largest_cluster_row_share": null,
          "largest_cluster_rows": 0,
          "row_weight_effective_cluster_count": null,
          "rows": 0
        },
        "new_nonmissing": {
          "clusters": 3539,
          "largest_cluster_row_share": 0.020171635848889674,
          "largest_cluster_rows": 1824,
          "row_weight_effective_cluster_count": 555.4469244346247,
          "rows": 90424
        },
        "seen": {
          "clusters": 546,
          "largest_cluster_row_share": 0.027308278650493803,
          "largest_cluster_rows": 412,
          "row_weight_effective_cluster_count": 161.37173036553264,
          "rows": 15087
        }
      }
    }
  },
  "replicates": 2000,
  "seed": 20260925,
  "stage_index": 2
}
```

v24完整次對照（不以較弱對照取代v22）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R114",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 466689.4190578535,
      "control_mae": 466689.4190578535,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1120722.162404482,
      "control_mae": 1118876.9223743458,
      "delta": 1845.2400301361922,
      "relative_change_pct": 0.1649189462430279,
      "rows": 869
    },
    "嘉義市": {
      "candidate_mae": 2230942.9743844937,
      "control_mae": 2076628.4947043534,
      "delta": 154314.47968014027,
      "relative_change_pct": 7.431010412967959,
      "rows": 731
    },
    "嘉義縣": {
      "candidate_mae": 1340876.400826514,
      "control_mae": 1377314.8007233748,
      "delta": -36438.39989686082,
      "relative_change_pct": -2.6456115826042916,
      "rows": 1560
    },
    "基隆市": {
      "candidate_mae": 3395247.9431549087,
      "control_mae": 2649701.530279867,
      "delta": 745546.4128750418,
      "relative_change_pct": 28.136995973138745,
      "rows": 1577
    },
    "宜蘭縣": {
      "candidate_mae": 1314744.6114818668,
      "control_mae": 1294116.4134795247,
      "delta": 20628.198002342135,
      "relative_change_pct": 1.5939986377947684,
      "rows": 2330
    },
    "屏東縣": {
      "candidate_mae": 1422835.8826050903,
      "control_mae": 1418492.199859719,
      "delta": 4343.682745371247,
      "relative_change_pct": 0.30621830319552074,
      "rows": 1240
    },
    "彰化縣": {
      "candidate_mae": 1449978.5904538704,
      "control_mae": 1437729.9372029828,
      "delta": 12248.653250887524,
      "relative_change_pct": 0.8519439523333946,
      "rows": 2607
    },
    "新北市": {
      "candidate_mae": 1957915.5912542006,
      "control_mae": 1931141.0431383061,
      "delta": 26774.54811589443,
      "relative_change_pct": 1.3864625896191918,
      "rows": 17191
    },
    "新竹市": {
      "candidate_mae": 4244867.420642141,
      "control_mae": 4206008.947003277,
      "delta": 38858.473638864234,
      "relative_change_pct": 0.9238799567117031,
      "rows": 1586
    },
    "新竹縣": {
      "candidate_mae": 2786357.9218253437,
      "control_mae": 2839386.2871954134,
      "delta": -53028.36537006963,
      "relative_change_pct": -1.8675995446342766,
      "rows": 5042
    },
    "桃園市": {
      "candidate_mae": 1823176.7387523102,
      "control_mae": 1801809.0342872667,
      "delta": 21367.7044650435,
      "relative_change_pct": 1.1859028375610194,
      "rows": 20723
    },
    "澎湖縣": {
      "candidate_mae": 960793.3000445202,
      "control_mae": 905431.1381984788,
      "delta": 55362.16184604133,
      "relative_change_pct": 6.114453050089981,
      "rows": 40
    },
    "臺中市": {
      "candidate_mae": 2877951.66481605,
      "control_mae": 2813849.3975944812,
      "delta": 64102.26722156862,
      "relative_change_pct": 2.2780987239888786,
      "rows": 20837
    },
    "臺北市": {
      "candidate_mae": 6496658.390980179,
      "control_mae": 6619596.082977112,
      "delta": -122937.69199693296,
      "relative_change_pct": -1.8571781488764596,
      "rows": 4712
    },
    "臺南市": {
      "candidate_mae": 1773449.5907854536,
      "control_mae": 1770047.7116755582,
      "delta": 3401.8791098953225,
      "relative_change_pct": 0.19219137921853213,
      "rows": 8886
    },
    "臺東縣": {
      "candidate_mae": 1956267.910025592,
      "control_mae": 1883004.2703493305,
      "delta": 73263.63967626146,
      "relative_change_pct": 3.890784573880428,
      "rows": 235
    },
    "花蓮縣": {
      "candidate_mae": 905821.7502370309,
      "control_mae": 829143.651670074,
      "delta": 76678.09856695693,
      "relative_change_pct": 9.247866568417995,
      "rows": 691
    },
    "苗栗縣": {
      "candidate_mae": 1680838.2682701568,
      "control_mae": 1653807.183102974,
      "delta": 27031.085167182842,
      "relative_change_pct": 1.6344762220989675,
      "rows": 2607
    },
    "金門縣": {
      "candidate_mae": 1553732.3032964822,
      "control_mae": 1495237.6344433622,
      "delta": 58494.66885311995,
      "relative_change_pct": 3.91206504609523,
      "rows": 35
    },
    "雲林縣": {
      "candidate_mae": 1071112.753533618,
      "control_mae": 1077147.497864338,
      "delta": -6034.744330720045,
      "relative_change_pct": -0.5602523649439971,
      "rows": 1873
    },
    "高雄市": {
      "candidate_mae": 1803734.2946698545,
      "control_mae": 1798279.317019422,
      "delta": 5454.977650432382,
      "relative_change_pct": 0.3033442913347741,
      "rows": 10138
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1789656.5618261937,
      "control_mae": 1762875.3535570733,
      "delta": 26781.208269120427,
      "relative_change_pct": 1.5191776443571103,
      "rows": 33694
    },
    "Q2": {
      "candidate_mae": 2396207.270683969,
      "control_mae": 2362943.1568230344,
      "delta": 33264.11386093451,
      "relative_change_pct": 1.4077407560517854,
      "rows": 29169
    },
    "Q3": {
      "candidate_mae": 2563209.9513362497,
      "control_mae": 2551611.570300658,
      "delta": 11598.381035591941,
      "relative_change_pct": 0.4545512009190842,
      "rows": 25213
    },
    "Q4": {
      "candidate_mae": 2767445.8781392057,
      "control_mae": 2720040.168653072,
      "delta": 47405.709486133885,
      "relative_change_pct": 1.7428312284670622,
      "rows": 17435
    }
  },
  "months": {
    "2021-08": {
      "candidate_mae": 1589326.987509469,
      "control_mae": 1563207.6855669126,
      "delta": 26119.301942556398,
      "relative_change_pct": 1.6708785520769736,
      "rows": 5654
    },
    "2021-09": {
      "candidate_mae": 1639704.6069366666,
      "control_mae": 1621468.4185501335,
      "delta": 18236.188386533177,
      "relative_change_pct": 1.124671204070654,
      "rows": 12352
    },
    "2021-10": {
      "candidate_mae": 1974918.5884949032,
      "control_mae": 1937066.7553911388,
      "delta": 37851.83310376434,
      "relative_change_pct": 1.9540799509576618,
      "rows": 11842
    },
    "2021-11": {
      "candidate_mae": 2065552.6642276784,
      "control_mae": 2042288.2736402748,
      "delta": 23264.39058740367,
      "relative_change_pct": 1.1391335340694133,
      "rows": 12524
    },
    "2021-12": {
      "candidate_mae": 2338209.126863814,
      "control_mae": 2308971.313995403,
      "delta": 29237.812868411187,
      "relative_change_pct": 1.2662700784194065,
      "rows": 11482
    },
    "2022-01": {
      "candidate_mae": 2822069.146588822,
      "control_mae": 2773309.212675835,
      "delta": 48759.93391298689,
      "relative_change_pct": 1.758185985541105,
      "rows": 8017
    },
    "2022-02": {
      "candidate_mae": 2486851.3507870864,
      "control_mae": 2452771.3776768353,
      "delta": 34079.97311025113,
      "relative_change_pct": 1.3894476028389684,
      "rows": 6204
    },
    "2022-03": {
      "candidate_mae": 2497749.1206258824,
      "control_mae": 2450727.282373386,
      "delta": 47021.83825249644,
      "relative_change_pct": 1.9186891414110565,
      "rows": 9626
    },
    "2022-04": {
      "candidate_mae": 2617839.7870368916,
      "control_mae": 2671623.6542875506,
      "delta": -53783.86725065904,
      "relative_change_pct": -2.013152831774943,
      "rows": 7535
    },
    "2022-05": {
      "candidate_mae": 2664814.4816410094,
      "control_mae": 2633628.192055837,
      "delta": 31186.289585172664,
      "relative_change_pct": 1.184156885897714,
      "rows": 6424
    },
    "2022-06": {
      "candidate_mae": 2734786.4730445514,
      "control_mae": 2695679.6338523687,
      "delta": 39106.83919218276,
      "relative_change_pct": 1.4507228047828356,
      "rows": 5629
    },
    "2022-07": {
      "candidate_mae": 2856846.140751275,
      "control_mae": 2793109.3935268214,
      "delta": 63736.74722445337,
      "relative_change_pct": 2.281928068129614,
      "rows": 6566
    },
    "2022-08": {
      "candidate_mae": 2799924.925446754,
      "control_mae": 2766911.3766995794,
      "delta": 33013.54874717444,
      "relative_change_pct": 1.193155264212097,
      "rows": 1656
    }
  },
  "overall_mae": {
    "candidate_mae": 2303762.540978195,
    "control_mae": 2275409.1418580045,
    "delta": 28353.399120190646,
    "relative_change_pct": 1.2460791599456456
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 1113339.5151574803,
      "control_mae": 1107505.6060768394,
      "delta": 5833.9090806408785,
      "relative_change_pct": 0.5267611331834756,
      "rows": 41132
    },
    {
      "band": "10-20m",
      "candidate_mae": 2064956.3963315822,
      "control_mae": 2024569.5907328655,
      "delta": 40386.80559871672,
      "relative_change_pct": 1.9948341506056735,
      "rows": 47793
    },
    {
      "band": "20-50m",
      "candidate_mae": 4692687.172438978,
      "control_mae": 4605603.27388477,
      "delta": 87083.89855420776,
      "relative_change_pct": 1.890825009787557,
      "rows": 15033
    },
    {
      "band": "50-100m",
      "candidate_mae": 13854446.31702492,
      "control_mae": 13891916.012976805,
      "delta": -37469.69595188461,
      "relative_change_pct": -0.2697230239290475,
      "rows": 1382
    },
    {
      "band": ">=100m",
      "candidate_mae": 52023019.88546188,
      "control_mae": 54572263.048466735,
      "delta": -2549243.163004853,
      "relative_change_pct": -4.671316563765769,
      "rows": 171
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 2230942.9743844937,
      "control_mae": 2076628.4947043534,
      "delta": 154314.47968014027,
      "group": "嘉義市",
      "relative_change_pct": 7.431010412967959,
      "rows": 731,
      "section": "cities"
    },
    {
      "candidate_mae": 3395247.9431549087,
      "control_mae": 2649701.530279867,
      "delta": 745546.4128750418,
      "group": "基隆市",
      "relative_change_pct": 28.136995973138745,
      "rows": 1577,
      "section": "cities"
    },
    {
      "candidate_mae": 905821.7502370309,
      "control_mae": 829143.651670074,
      "delta": 76678.09856695693,
      "group": "花蓮縣",
      "relative_change_pct": 9.247866568417995,
      "rows": 691,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 1592279.6863470569,
      "control_mae": 1567209.9156508578,
      "delta": 25069.770696199033,
      "relative_change_pct": 1.5996434457082687,
      "rows": 9387
    },
    "1": {
      "candidate_mae": 1736230.4549414301,
      "control_mae": 1715851.8997759607,
      "delta": 20378.555165469414,
      "relative_change_pct": 1.1876639917541978,
      "rows": 12463
    },
    "10": {
      "candidate_mae": 2755578.908782729,
      "control_mae": 2710524.938303519,
      "delta": 45053.97047920991,
      "relative_change_pct": 1.6621861633713129,
      "rows": 5992
    },
    "11": {
      "candidate_mae": 2877092.4320584545,
      "control_mae": 2817934.7094697705,
      "delta": 59157.722588683944,
      "relative_change_pct": 2.0993290720995876,
      "rows": 6217
    },
    "2": {
      "candidate_mae": 2002306.536516118,
      "control_mae": 1967431.6495803462,
      "delta": 34874.88693577191,
      "relative_change_pct": 1.772609835935634,
      "rows": 11844
    },
    "3": {
      "candidate_mae": 2107820.109171708,
      "control_mae": 2077612.941580479,
      "delta": 30207.16759122908,
      "relative_change_pct": 1.4539362451337987,
      "rows": 12872
    },
    "4": {
      "candidate_mae": 2528752.6379371607,
      "control_mae": 2505314.56244677,
      "delta": 23438.075490390882,
      "relative_change_pct": 0.9355342375649832,
      "rows": 9987
    },
    "5": {
      "candidate_mae": 2774715.822225678,
      "control_mae": 2719663.807003374,
      "delta": 55052.01522230357,
      "relative_change_pct": 2.0242213423784134,
      "rows": 6310
    },
    "6": {
      "candidate_mae": 2543486.1980046746,
      "control_mae": 2504077.440253743,
      "delta": 39408.75775093166,
      "relative_change_pct": 1.5737835067488286,
      "rows": 8284
    },
    "7": {
      "candidate_mae": 2460283.7712245747,
      "control_mae": 2500338.7490206924,
      "delta": -40054.977796117775,
      "relative_change_pct": -1.6019820439052952,
      "rows": 9188
    },
    "8": {
      "candidate_mae": 2700017.1454143743,
      "control_mae": 2655677.968591626,
      "delta": 44339.1768227485,
      "relative_change_pct": 1.6695991512202326,
      "rows": 7187
    },
    "9": {
      "candidate_mae": 2664008.432264562,
      "control_mae": 2628697.2045045653,
      "delta": 35311.227759996895,
      "relative_change_pct": 1.3432976494777402,
      "rows": 5780
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 2485577.0396309886,
      "control_mae": 2456616.5723295277,
      "delta": 28960.467301460914,
      "relative_change_pct": 1.1788761676389192,
      "rows": 90424
    },
    "seen": {
      "candidate_mae": 1214056.5539575692,
      "control_mae": 1189341.620617399,
      "delta": 24714.933340170188,
      "relative_change_pct": 2.0780348481658635,
      "rows": 15087
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 2019976.8311982835,
    "control_mae": 1992570.9762837698,
    "delta": 27405.85491451365,
    "relative_change_pct": 1.3754016916188723
  }
}
```

完整城市／月份／新舊案／真價帶／季度／誤差尾部，僅事後診斷，不作未知test路由：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 466689.4190578535,
      "rows": 1
    },
    "南投縣": {
      "mae": 1120722.162404482,
      "rows": 869
    },
    "嘉義市": {
      "mae": 2230942.9743844937,
      "rows": 731
    },
    "嘉義縣": {
      "mae": 1340876.400826514,
      "rows": 1560
    },
    "基隆市": {
      "mae": 3395247.9431549087,
      "rows": 1577
    },
    "宜蘭縣": {
      "mae": 1314744.6114818668,
      "rows": 2330
    },
    "屏東縣": {
      "mae": 1422835.8826050903,
      "rows": 1240
    },
    "彰化縣": {
      "mae": 1449978.5904538704,
      "rows": 2607
    },
    "新北市": {
      "mae": 1957915.5912542006,
      "rows": 17191
    },
    "新竹市": {
      "mae": 4244867.420642141,
      "rows": 1586
    },
    "新竹縣": {
      "mae": 2786357.9218253437,
      "rows": 5042
    },
    "桃園市": {
      "mae": 1823176.7387523102,
      "rows": 20723
    },
    "澎湖縣": {
      "mae": 960793.3000445202,
      "rows": 40
    },
    "臺中市": {
      "mae": 2877951.66481605,
      "rows": 20837
    },
    "臺北市": {
      "mae": 6496658.390980179,
      "rows": 4712
    },
    "臺南市": {
      "mae": 1773449.5907854536,
      "rows": 8886
    },
    "臺東縣": {
      "mae": 1956267.910025592,
      "rows": 235
    },
    "花蓮縣": {
      "mae": 905821.7502370309,
      "rows": 691
    },
    "苗栗縣": {
      "mae": 1680838.2682701568,
      "rows": 2607
    },
    "金門縣": {
      "mae": 1553732.3032964822,
      "rows": 35
    },
    "雲林縣": {
      "mae": 1071112.753533618,
      "rows": 1873
    },
    "高雄市": {
      "mae": 1803734.2946698545,
      "rows": 10138
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1789656.5618261937,
      "rows": 33694
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 2396207.270683969,
      "rows": 29169
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 2563209.9513362497,
      "rows": 25213
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 2767445.8781392057,
      "rows": 17435
    }
  },
  "months": {
    "2021-08": {
      "mae": 1589326.987509469,
      "rows": 5654
    },
    "2021-09": {
      "mae": 1639704.6069366666,
      "rows": 12352
    },
    "2021-10": {
      "mae": 1974918.5884949032,
      "rows": 11842
    },
    "2021-11": {
      "mae": 2065552.6642276784,
      "rows": 12524
    },
    "2021-12": {
      "mae": 2338209.126863814,
      "rows": 11482
    },
    "2022-01": {
      "mae": 2822069.146588822,
      "rows": 8017
    },
    "2022-02": {
      "mae": 2486851.3507870864,
      "rows": 6204
    },
    "2022-03": {
      "mae": 2497749.1206258824,
      "rows": 9626
    },
    "2022-04": {
      "mae": 2617839.7870368916,
      "rows": 7535
    },
    "2022-05": {
      "mae": 2664814.4816410094,
      "rows": 6424
    },
    "2022-06": {
      "mae": 2734786.4730445514,
      "rows": 5629
    },
    "2022-07": {
      "mae": 2856846.140751275,
      "rows": 6566
    },
    "2022-08": {
      "mae": 2799924.925446754,
      "rows": 1656
    }
  },
  "overall_mae": 2303762.540978195,
  "price_bands": [
    {
      "absolute_error_share": 0.18839613943232553,
      "absolute_error_sum": 45793880937.45747,
      "band": "<10m",
      "mae": 1113339.5151574803,
      "mean_signed_error_prediction_minus_truth": 51843.4407405099,
      "rows": 41132,
      "underprediction_fraction": 0.5400661285617038
    },
    {
      "absolute_error_share": 0.4060128008365542,
      "absolute_error_sum": 98690461049.8753,
      "band": "10-20m",
      "mae": 2064956.3963315822,
      "mean_signed_error_prediction_minus_truth": -1366732.5559841257,
      "rows": 47793,
      "underprediction_fraction": 0.7911828091979997
    },
    {
      "absolute_error_share": 0.290222988476645,
      "absolute_error_sum": 70545166263.27516,
      "band": "20-50m",
      "mae": 4692687.172438978,
      "mean_signed_error_prediction_minus_truth": -3618328.1657690913,
      "rows": 15033,
      "underprediction_fraction": 0.8203951307124326
    },
    {
      "absolute_error_share": 0.07877016690209204,
      "absolute_error_sum": 19146844810.12844,
      "band": "50-100m",
      "mae": 13854446.31702492,
      "mean_signed_error_prediction_minus_truth": -9244771.11186754,
      "rows": 1382,
      "underprediction_fraction": 0.7771345875542692
    },
    {
      "absolute_error_share": 0.03659790435238319,
      "absolute_error_sum": 8895936400.413982,
      "band": ">=100m",
      "mae": 52023019.88546188,
      "mean_signed_error_prediction_minus_truth": -41462702.096991755,
      "rows": 171,
      "underprediction_fraction": 0.9005847953216374
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 1592279.6863470569,
      "rows": 9387
    },
    "1": {
      "mae": 1736230.4549414301,
      "rows": 12463
    },
    "10": {
      "mae": 2755578.908782729,
      "rows": 5992
    },
    "11": {
      "mae": 2877092.4320584545,
      "rows": 6217
    },
    "2": {
      "mae": 2002306.536516118,
      "rows": 11844
    },
    "3": {
      "mae": 2107820.109171708,
      "rows": 12872
    },
    "4": {
      "mae": 2528752.6379371607,
      "rows": 9987
    },
    "5": {
      "mae": 2774715.822225678,
      "rows": 6310
    },
    "6": {
      "mae": 2543486.1980046746,
      "rows": 8284
    },
    "7": {
      "mae": 2460283.7712245747,
      "rows": 9188
    },
    "8": {
      "mae": 2700017.1454143743,
      "rows": 7187
    },
    "9": {
      "mae": 2664008.432264562,
      "rows": 5780
    }
  },
  "rows": 105511,
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 2485577.0396309886,
      "rows": 90424
    },
    "seen": {
      "mae": 1214056.5539575692,
      "rows": 15087
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 1460701.2672251128,
      "0.9": 4822505.884835187,
      "0.95": 7133159.292870766,
      "0.99": 14644524.480714034
    },
    "absolute_error_sum": 243072289461.15036,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 397542483.58458996,
    "mean_signed_error_prediction_minus_truth": -1302694.0100539655,
    "top10_absolute_error_sum": 1628852516.9468846,
    "top10_error_rows": 10,
    "top10_error_share": 0.006701103283133473,
    "top1pct_absolute_error_sum": 27080588883.505657,
    "top1pct_error_rows": 1056,
    "top1pct_error_share": 0.11140960964138975,
    "underprediction_fraction": 0.6974438684118244
  },
  "weighted_mae": 2019976.8311982835
}
```

## offset_2023_02_11完整診斷

fit實測31.312秒。

|層|總價MAE|
|---|---:|
|base|1,344,608.40|
|native|1,659,237.74|
|reference|1,462,286.25|
|spatial|1,443,672.32|
|median|1,336,671.42|

未投影總價與固定非負投影：

```json
{
  "negative_residual_not_equivalent_to_negative_total": true,
  "overall_mae": 1659299.1300766072,
  "projected_max_TWD": 1468543460.8459473,
  "projected_min_TWD": 0.0,
  "projected_zero_fraction": 4.6655282777668915e-05,
  "projected_zero_rows": 5,
  "projection_changes_count": 5,
  "projection_max_correction_TWD": 2548185.348510742,
  "projection_mean_correction_TWD": 61.39130469498409,
  "query_y_used_for_diagnostics": false,
  "raw_max_TWD": 1468543460.8459473,
  "raw_min_TWD": -2548185.348510742,
  "raw_negative_total_fraction": 4.6655282777668915e-05,
  "raw_negative_total_rows": 5,
  "raw_zero_total_rows": 0
}
```

所有季度／至少500列群退步超5%（群可重疊，不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+260.75|[-4,423.34, +4,979.52]|
|month|weighted|+454.19|[-4,806.16, +5,520.43]|
|project|overall|+260.75|[-7,205.31, +8,900.98]|
|project|weighted|+454.19|[-8,023.12, +10,368.02]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R114",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R91 additive CatBoost and R104 offset NN supplied strong contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R114 minus v22",
  "fallback": {
    "address": 0,
    "road": 0,
    "singleton": 0
  },
  "methods": {
    "month": {
      "family_index": 1,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": 256.8770664145632,
          "candidate_mae": 1336671.4247535267,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.4645,
          "percentile_2_5": -4423.343107543367,
          "percentile_97_5": 4979.519761004695,
          "point_delta": 260.7500892127864,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 419.81659238538447,
          "candidate_mae": 1393344.5318574405,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.435,
          "percentile_2_5": -4806.159754872904,
          "percentile_97_5": 5520.430863209534,
          "point_delta": 454.18727069790475,
          "valid_replicates": 2000
        }
      },
      "support": {
        "block_length_months": 3,
        "blocks_per_replicate": 4,
        "circular_boundary_assumption": true,
        "months_with_rows": 12,
        "rows_by_relative_month": [
          5055,
          7516,
          9057,
          9401,
          11490,
          6914,
          7252,
          9394,
          9933,
          10809,
          9935,
          10413
        ],
        "stratum_rows_by_relative_month": {
          "missing": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
          ],
          "new_nonmissing": [
            641,
            2307,
            3701,
            4812,
            6206,
            3317,
            3823,
            5538,
            6336,
            7768,
            7074,
            7829
          ],
          "seen": [
            4414,
            5209,
            5356,
            4589,
            5284,
            3597,
            3429,
            3856,
            3597,
            3041,
            2861,
            2584
          ]
        }
      }
    },
    "project": {
      "family_index": 0,
      "summary": {
        "overall": {
          "bootstrap_mean_delta": 177.72259313111834,
          "candidate_mae": 1336671.4247535267,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.501,
          "percentile_2_5": -7205.312898050504,
          "percentile_97_5": 8900.97876585585,
          "point_delta": 260.7500892127864,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 372.64756167551593,
          "candidate_mae": 1393344.5318574405,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.49,
          "percentile_2_5": -8023.116588961583,
          "percentile_97_5": 10368.021849423316,
          "point_delta": 454.18727069790475,
          "valid_replicates": 2000
        }
      },
      "support": {
        "missing": {
          "clusters": 0,
          "largest_cluster_row_share": null,
          "largest_cluster_rows": 0,
          "row_weight_effective_cluster_count": null,
          "rows": 0
        },
        "new_nonmissing": {
          "clusters": 1925,
          "largest_cluster_row_share": 0.015467044076021027,
          "largest_cluster_rows": 918,
          "row_weight_effective_cluster_count": 343.1287463867188,
          "rows": 59352
        },
        "seen": {
          "clusters": 2176,
          "largest_cluster_row_share": 0.02643411339063513,
          "largest_cluster_rows": 1264,
          "row_weight_effective_cluster_count": 362.4854031036448,
          "rows": 47817
        }
      }
    }
  },
  "replicates": 2000,
  "seed": 20260925,
  "stage_index": 0
}
```

v24完整次對照（不以較弱對照取代v22）：

```json
{
  "automatic_adoption": false,
  "both_point_metrics_improve": true,
  "candidate": "R114",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 748690.2509114165,
      "control_mae": 748690.2509114165,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1019841.129495744,
      "control_mae": 1015289.1847550352,
      "delta": 4551.944740708801,
      "relative_change_pct": 0.44833972517958764,
      "rows": 899
    },
    "嘉義市": {
      "candidate_mae": 1142178.530411801,
      "control_mae": 1169659.4148716796,
      "delta": -27480.88445987855,
      "relative_change_pct": -2.3494774727131493,
      "rows": 604
    },
    "嘉義縣": {
      "candidate_mae": 706052.6417529159,
      "control_mae": 703921.1658472866,
      "delta": 2131.4759056293406,
      "relative_change_pct": 0.302800371553504,
      "rows": 1532
    },
    "基隆市": {
      "candidate_mae": 822268.3823382322,
      "control_mae": 824405.9509523374,
      "delta": -2137.568614105112,
      "relative_change_pct": -0.25928592723473615,
      "rows": 842
    },
    "宜蘭縣": {
      "candidate_mae": 781900.1163969607,
      "control_mae": 785285.7426309958,
      "delta": -3385.626234035124,
      "relative_change_pct": -0.43113303225040506,
      "rows": 2576
    },
    "屏東縣": {
      "candidate_mae": 902256.6697624925,
      "control_mae": 920486.7394754328,
      "delta": -18230.069712940254,
      "relative_change_pct": -1.9804815138704996,
      "rows": 691
    },
    "彰化縣": {
      "candidate_mae": 1017822.5380812474,
      "control_mae": 1011819.6970080789,
      "delta": 6002.841073168442,
      "relative_change_pct": 0.5932718142292215,
      "rows": 2629
    },
    "新北市": {
      "candidate_mae": 1133382.694833419,
      "control_mae": 1175179.3847952979,
      "delta": -41796.689961878816,
      "relative_change_pct": -3.556622121069567,
      "rows": 20773
    },
    "新竹市": {
      "candidate_mae": 2376173.0540875327,
      "control_mae": 2294808.7567983773,
      "delta": 81364.29728915542,
      "relative_change_pct": 3.545580739489226,
      "rows": 1852
    },
    "新竹縣": {
      "candidate_mae": 2279992.95259163,
      "control_mae": 2237487.2731571877,
      "delta": 42505.67943444243,
      "relative_change_pct": 1.899705975733446,
      "rows": 4704
    },
    "桃園市": {
      "candidate_mae": 1062065.069099189,
      "control_mae": 1066753.324847084,
      "delta": -4688.255747894989,
      "relative_change_pct": -0.4394882714396074,
      "rows": 21410
    },
    "澎湖縣": {
      "candidate_mae": 3803804.992448648,
      "control_mae": 3812375.149056576,
      "delta": -8570.15660792822,
      "relative_change_pct": -0.22479835464379788,
      "rows": 20
    },
    "臺中市": {
      "candidate_mae": 1612638.8908247126,
      "control_mae": 1612776.5130701174,
      "delta": -137.62224540486932,
      "relative_change_pct": -0.008533249603374279,
      "rows": 20055
    },
    "臺北市": {
      "candidate_mae": 4159018.4894177583,
      "control_mae": 4030540.187754385,
      "delta": 128478.30166337313,
      "relative_change_pct": 3.1876199139191512,
      "rows": 3514
    },
    "臺南市": {
      "candidate_mae": 1267398.2856164083,
      "control_mae": 1290312.176866015,
      "delta": -22913.89124960662,
      "relative_change_pct": -1.7758408903232399,
      "rows": 8483
    },
    "臺東縣": {
      "candidate_mae": 692639.5222067528,
      "control_mae": 694696.0897249224,
      "delta": -2056.5675181696424,
      "relative_change_pct": -0.29603844739992385,
      "rows": 213
    },
    "花蓮縣": {
      "candidate_mae": 653646.734598808,
      "control_mae": 651104.8750916787,
      "delta": 2541.859507129295,
      "relative_change_pct": 0.3903917178889789,
      "rows": 578
    },
    "苗栗縣": {
      "candidate_mae": 1056040.074082078,
      "control_mae": 1025581.2731701825,
      "delta": 30458.800911895465,
      "relative_change_pct": 2.969906111657443,
      "rows": 2973
    },
    "金門縣": {
      "candidate_mae": 912040.8483768943,
      "control_mae": 893641.4111528273,
      "delta": 18399.437224066933,
      "relative_change_pct": 2.0589284465153694,
      "rows": 65
    },
    "雲林縣": {
      "candidate_mae": 998411.1532723767,
      "control_mae": 996471.2687277427,
      "delta": 1939.8845446340274,
      "relative_change_pct": 0.19467541167652525,
      "rows": 1719
    },
    "高雄市": {
      "candidate_mae": 899080.881473293,
      "control_mae": 908831.0314082025,
      "delta": -9750.14993490954,
      "relative_change_pct": -1.072823175921053,
      "rows": 11036
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 992681.6171243774,
      "control_mae": 993826.2791985838,
      "delta": -1144.662074206397,
      "relative_change_pct": -0.11517727978872186,
      "rows": 22487
    },
    "Q2": {
      "candidate_mae": 1225777.906126247,
      "control_mae": 1236271.5483622926,
      "delta": -10493.642236045562,
      "relative_change_pct": -0.848813697115868,
      "rows": 27820
    },
    "Q3": {
      "candidate_mae": 1528569.1420583085,
      "control_mae": 1519816.2249750085,
      "delta": 8752.917083299952,
      "relative_change_pct": 0.5759194394337962,
      "rows": 27006
    },
    "Q4": {
      "candidate_mae": 1525510.0749020292,
      "control_mae": 1535913.1617296129,
      "delta": -10403.086827583611,
      "relative_change_pct": -0.6773225913285718,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "candidate_mae": 594268.0740787686,
      "control_mae": 596296.3427625896,
      "delta": -2028.268683820963,
      "relative_change_pct": -0.34014441115371746,
      "rows": 2896
    },
    "2023-03": {
      "candidate_mae": 920623.8226069277,
      "control_mae": 925522.7748700764,
      "delta": -4898.952263148734,
      "relative_change_pct": -0.5293173108394271,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1093648.3829126407,
      "control_mae": 1094744.1862448624,
      "delta": -1095.8033322216943,
      "relative_change_pct": -0.10009674826230089,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1217652.8369876246,
      "control_mae": 1213455.5062566611,
      "delta": 4197.330730963498,
      "relative_change_pct": 0.34589902219914687,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1258408.1034825586,
      "control_mae": 1276086.4061312978,
      "delta": -17678.30264873919,
      "relative_change_pct": -1.3853531049150798,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1174868.037063123,
      "control_mae": 1183156.183768008,
      "delta": -8288.146704884944,
      "relative_change_pct": -0.700511633087156,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1300749.5726528019,
      "control_mae": 1306980.5584263823,
      "delta": -6230.985773580382,
      "relative_change_pct": -0.4767466304994277,
      "rows": 7400
    },
    "2023-09": {
      "candidate_mae": 1366938.372726994,
      "control_mae": 1365893.9517492831,
      "delta": 1044.420977710979,
      "relative_change_pct": 0.07646428014220301,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1610293.6814103387,
      "control_mae": 1596675.0554627297,
      "delta": 13618.625947609078,
      "relative_change_pct": 0.8529366010332194,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1710197.741073452,
      "control_mae": 1708449.945458693,
      "delta": 1747.795614758972,
      "relative_change_pct": 0.10230300392498273,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1489839.4269801285,
      "control_mae": 1506056.8881887118,
      "delta": -16217.461208583321,
      "relative_change_pct": -1.0768159779201676,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1520647.3031196562,
      "control_mae": 1526897.035515871,
      "delta": -6249.732396214735,
      "relative_change_pct": -0.409309354255392,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1448538.5862590764,
      "control_mae": 1437162.9778553827,
      "delta": 11375.608403693652,
      "relative_change_pct": 0.7915322464449362,
      "rows": 2566
    }
  },
  "overall_mae": {
    "candidate_mae": 1336671.4247535176,
    "control_mae": 1340328.1391452,
    "delta": -3656.714391682297,
    "relative_change_pct": -0.2728223249878483
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 707396.7911573211,
      "control_mae": 710632.9248020112,
      "delta": -3236.133644690155,
      "relative_change_pct": -0.45538751889265067,
      "rows": 25432
    },
    {
      "band": "10-20m",
      "candidate_mae": 1060304.857786816,
      "control_mae": 1067452.758311166,
      "delta": -7147.900524349883,
      "relative_change_pct": -0.6696221887757067,
      "rows": 59481
    },
    {
      "band": "20-50m",
      "candidate_mae": 2230304.680510607,
      "control_mae": 2241526.0750715095,
      "delta": -11221.39456090238,
      "relative_change_pct": -0.5006140542239462,
      "rows": 20803
    },
    {
      "band": "50-100m",
      "candidate_mae": 6771695.1197890965,
      "control_mae": 6794509.1460085185,
      "delta": -22814.026219421998,
      "relative_change_pct": -0.3357715138675508,
      "rows": 1183
    },
    {
      "band": ">=100m",
      "candidate_mae": 28826992.16780181,
      "control_mae": 27434374.16274089,
      "delta": 1392618.0050609186,
      "relative_change_pct": 5.076179237039997,
      "rows": 270
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 649810.0371942542,
      "control_mae": 645679.7496816863,
      "delta": 4130.287512567942,
      "relative_change_pct": 0.6396805095101918,
      "rows": 5055
    },
    "1": {
      "candidate_mae": 1010876.8825470086,
      "control_mae": 1023592.2057761417,
      "delta": -12715.323229133151,
      "relative_change_pct": -1.242225483681924,
      "rows": 7516
    },
    "10": {
      "candidate_mae": 1486401.9934001125,
      "control_mae": 1497637.3318924468,
      "delta": -11235.338492334355,
      "relative_change_pct": -0.7502042218817515,
      "rows": 9935
    },
    "11": {
      "candidate_mae": 1511829.6504635974,
      "control_mae": 1514286.76981743,
      "delta": -2457.119353832677,
      "relative_change_pct": -0.1622624857330636,
      "rows": 10413
    },
    "2": {
      "candidate_mae": 1155715.7614205652,
      "control_mae": 1151164.1928843837,
      "delta": 4551.5685361814685,
      "relative_change_pct": 0.3953883003237751,
      "rows": 9057
    },
    "3": {
      "candidate_mae": 1182282.8928178626,
      "control_mae": 1198704.169185322,
      "delta": -16421.276367459446,
      "relative_change_pct": -1.3699190166844815,
      "rows": 9401
    },
    "4": {
      "candidate_mae": 1264580.7779082428,
      "control_mae": 1271555.585854942,
      "delta": -6974.807946699206,
      "relative_change_pct": -0.5485256031500684,
      "rows": 11490
    },
    "5": {
      "candidate_mae": 1199161.0805049413,
      "control_mae": 1206414.4495510287,
      "delta": -7253.369046087377,
      "relative_change_pct": -0.6012336016687088,
      "rows": 6914
    },
    "6": {
      "candidate_mae": 1340608.3222716923,
      "control_mae": 1342783.9583831823,
      "delta": -2175.636111489963,
      "relative_change_pct": -0.16202428528484955,
      "rows": 7252
    },
    "7": {
      "candidate_mae": 1409079.859898145,
      "control_mae": 1412213.0920092391,
      "delta": -3133.232111094054,
      "relative_change_pct": -0.22186680812002804,
      "rows": 9394
    },
    "8": {
      "candidate_mae": 1737694.8987225576,
      "control_mae": 1708489.12857141,
      "delta": 29205.77015114762,
      "relative_change_pct": 1.7094501605385486,
      "rows": 9933
    },
    "9": {
      "candidate_mae": 1594468.53697008,
      "control_mae": 1611253.9941320717,
      "delta": -16785.457161991624,
      "relative_change_pct": -1.0417635719210976,
      "rows": 10809
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1652725.3170915348,
      "control_mae": 1658897.3265622966,
      "delta": -6172.009470761754,
      "relative_change_pct": -0.3720549410705182,
      "rows": 59352
    },
    "seen": {
      "candidate_mae": 944375.1573581145,
      "control_mae": 944909.8065107913,
      "delta": -534.6491526768077,
      "relative_change_pct": -0.056582030262874804,
      "rows": 47817
    }
  },
  "used_for_release_gate": false,
  "weighted_mae": {
    "candidate_mae": 1393344.531857431,
    "control_mae": 1397452.2756174281,
    "delta": -4107.743759997189,
    "relative_change_pct": -0.2939451909498869
  }
}
```

完整城市／月份／新舊案／真價帶／季度／誤差尾部，僅事後診斷，不作未知test路由：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 748690.2509114165,
      "rows": 1
    },
    "南投縣": {
      "mae": 1019841.129495744,
      "rows": 899
    },
    "嘉義市": {
      "mae": 1142178.530411801,
      "rows": 604
    },
    "嘉義縣": {
      "mae": 706052.6417529159,
      "rows": 1532
    },
    "基隆市": {
      "mae": 822268.3823382322,
      "rows": 842
    },
    "宜蘭縣": {
      "mae": 781900.1163969607,
      "rows": 2576
    },
    "屏東縣": {
      "mae": 902256.6697624925,
      "rows": 691
    },
    "彰化縣": {
      "mae": 1017822.5380812474,
      "rows": 2629
    },
    "新北市": {
      "mae": 1133382.694833419,
      "rows": 20773
    },
    "新竹市": {
      "mae": 2376173.0540875327,
      "rows": 1852
    },
    "新竹縣": {
      "mae": 2279992.95259163,
      "rows": 4704
    },
    "桃園市": {
      "mae": 1062065.069099189,
      "rows": 21410
    },
    "澎湖縣": {
      "mae": 3803804.992448648,
      "rows": 20
    },
    "臺中市": {
      "mae": 1612638.8908247126,
      "rows": 20055
    },
    "臺北市": {
      "mae": 4159018.4894177583,
      "rows": 3514
    },
    "臺南市": {
      "mae": 1267398.2856164083,
      "rows": 8483
    },
    "臺東縣": {
      "mae": 692639.5222067528,
      "rows": 213
    },
    "花蓮縣": {
      "mae": 653646.734598808,
      "rows": 578
    },
    "苗栗縣": {
      "mae": 1056040.074082078,
      "rows": 2973
    },
    "金門縣": {
      "mae": 912040.8483768943,
      "rows": 65
    },
    "雲林縣": {
      "mae": 998411.1532723767,
      "rows": 1719
    },
    "高雄市": {
      "mae": 899080.881473293,
      "rows": 11036
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 992681.6171243774,
      "rows": 22487
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1225777.906126247,
      "rows": 27820
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1528569.1420583085,
      "rows": 27006
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1525510.0749020292,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "mae": 594268.0740787686,
      "rows": 2896
    },
    "2023-03": {
      "mae": 920623.8226069277,
      "rows": 7500
    },
    "2023-04": {
      "mae": 1093648.3829126407,
      "rows": 8639
    },
    "2023-05": {
      "mae": 1217652.8369876246,
      "rows": 8750
    },
    "2023-06": {
      "mae": 1258408.1034825586,
      "rows": 13180
    },
    "2023-07": {
      "mae": 1174868.037063123,
      "rows": 6144
    },
    "2023-08": {
      "mae": 1300749.5726528019,
      "rows": 7400
    },
    "2023-09": {
      "mae": 1366938.372726994,
      "rows": 8602
    },
    "2023-10": {
      "mae": 1610293.6814103387,
      "rows": 9855
    },
    "2023-11": {
      "mae": 1710197.741073452,
      "rows": 10336
    },
    "2023-12": {
      "mae": 1489839.4269801285,
      "rows": 10416
    },
    "2024-01": {
      "mae": 1520647.3031196562,
      "rows": 10885
    },
    "2024-02": {
      "mae": 1448538.5862590764,
      "rows": 2566
    }
  },
  "overall_mae": 1336671.4247535176,
  "price_bands": [
    {
      "absolute_error_share": 0.12558846670740342,
      "absolute_error_sum": 17990515192.71299,
      "band": "<10m",
      "mae": 707396.7911573211,
      "mean_signed_error_prediction_minus_truth": 421500.4489336005,
      "rows": 25432,
      "underprediction_fraction": 0.2833831393519975
    },
    {
      "absolute_error_share": 0.4402660226922489,
      "absolute_error_sum": 63067993246.0176,
      "band": "10-20m",
      "mae": 1060304.857786816,
      "mean_signed_error_prediction_minus_truth": 318209.4345231306,
      "rows": 59481,
      "underprediction_fraction": 0.35766042938081066
    },
    {
      "absolute_error_share": 0.32388909253702286,
      "absolute_error_sum": 46397028268.662155,
      "band": "20-50m",
      "mae": 2230304.680510607,
      "mean_signed_error_prediction_minus_truth": -148453.4616618091,
      "rows": 20803,
      "underprediction_fraction": 0.45935682353506707
    },
    {
      "absolute_error_share": 0.055922721613437684,
      "absolute_error_sum": 8010915326.710502,
      "band": "50-100m",
      "mae": 6771695.1197890965,
      "mean_signed_error_prediction_minus_truth": -976452.3886192502,
      "rows": 1183,
      "underprediction_fraction": 0.44716821639898563
    },
    {
      "absolute_error_share": 0.054333696449887135,
      "absolute_error_sum": 7783287885.306488,
      "band": ">=100m",
      "mae": 28826992.16780181,
      "mean_signed_error_prediction_minus_truth": -9663168.073738616,
      "rows": 270,
      "underprediction_fraction": 0.7111111111111111
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 649810.0371942542,
      "rows": 5055
    },
    "1": {
      "mae": 1010876.8825470086,
      "rows": 7516
    },
    "10": {
      "mae": 1486401.9934001125,
      "rows": 9935
    },
    "11": {
      "mae": 1511829.6504635974,
      "rows": 10413
    },
    "2": {
      "mae": 1155715.7614205652,
      "rows": 9057
    },
    "3": {
      "mae": 1182282.8928178626,
      "rows": 9401
    },
    "4": {
      "mae": 1264580.7779082428,
      "rows": 11490
    },
    "5": {
      "mae": 1199161.0805049413,
      "rows": 6914
    },
    "6": {
      "mae": 1340608.3222716923,
      "rows": 7252
    },
    "7": {
      "mae": 1409079.859898145,
      "rows": 9394
    },
    "8": {
      "mae": 1737694.8987225576,
      "rows": 9933
    },
    "9": {
      "mae": 1594468.53697008,
      "rows": 10809
    }
  },
  "rows": 107169,
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 1652725.3170915348,
      "rows": 59352
    },
    "seen": {
      "mae": 944375.1573581145,
      "rows": 47817
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 788470.0043932088,
      "0.9": 2915211.0736038336,
      "0.95": 4190283.609496908,
      "0.99": 8598554.48493627
    },
    "absolute_error_sum": 143249739919.40973,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 749477061.2782035,
    "mean_signed_error_prediction_minus_truth": 212697.13138643678,
    "top10_absolute_error_sum": 2639648919.7105155,
    "top10_error_rows": 10,
    "top10_error_share": 0.018426902004817215,
    "top1pct_absolute_error_sum": 18204349223.34838,
    "top1pct_error_rows": 1072,
    "top1pct_error_share": 0.12708120261572475,
    "underprediction_fraction": 0.36165308997937834
  },
  "weighted_mae": 1393344.531857431
}
```

## 執行、核驗與限制

四fit流程命令實測286.812秒；分開evaluate命令170.828秒。這是程序實測，不是整個專案總工時。四起點各自encoder／group OOF fit prior／full-fit query prior、日期權重、reference／spatial均依相應外層切點，完整原v22/v24重播一致；不使用後期模型回測。

[獨立核驗](../outputs/research_round114_20261008/postfit_review.json)核對全部來源／input／protected、模型metadata、signed native投影、五層、評分及人工gates。其完整實作範圍和限制如下：

```json
[
  "All four origins are reused and overlapping, not fresh blind validation or four independent votes.",
  "CPU checkpoint constructors/tensors/config were inspected without independent full-query NN inference; exact saved reload is frozen training evidence.",
  "Saved factors, signed total/projection and all complete-chain layers were verified. No separately saved residual output exists, so raw total inverse relies on frozen trainer code/reload.",
  "Composition and descriptive definitions reuse frozen helpers; independent MAE/group/gate arithmetic supplements replay.",
  "Bootstrap summaries use saved replicates without resampling, selection/refit/future-shift correction; not Private probability.",
  "No fullfit/release/CSV/Kaggle upload or v22 replacement is authorized by audit pass."
]
```

Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R91 additive CatBoost and R104 offset NN supplied strong contrary evidence; native improvement alone cannot establish complete-composition improvement.

訓練前已固定R91加法CatBoost跨窗與高價群惡化、R104教師offset NN失敗、R102 native改善未保留到完整組合等反證。本結果繼續反對「改加法就會改善」；沒有事後再搜尋bias、learning rate、clip或融合比例。四窗口重疊且已被反覆使用，bootstrap不是Private勝率，保存重載也不是乾淨環境端到端重訓。禁止內政部／外部成交答案，沒有test ID人工改價、Public搜參數或Kaggle操作。

- [事前固定規約](round114_plan.md)
- [作者方法與梯度說明（訓練前狀態）](round114_method_review.md)
- [訓練程式](../src/round114_additive_prior_train.py)
- [四起點研究流程](../src/round114_research.py)
- [完整機器結果](../outputs/round114/results.json)
