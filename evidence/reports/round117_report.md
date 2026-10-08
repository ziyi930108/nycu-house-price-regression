# R117：fit先驗固定交易五折比較

2026-10-08。combined探索門檻失敗，不全量fit、不產v32 CSV。
combined強推薦門檻為False。v22選定與上台材料保持，Public／Private沒有新回報。

原本先驗為每筆房屋估計參考單價，訓練時用分組五折，避免同案價格互相暴露。R115純X盤點顯示交易五折可能增加細層級支持，因此本輪只把fit的OOF先驗改為canonical列順序上seed42固定交易五折；每一fold用自己的補集建立價格狀態、日期anchor及日期權重。保留四cell、至少30列支持、原weighted median與fallback；query仍使用完整fit狀態。同案交易可能高度相似，交易fold會暴露同案peers，而真query多為新案，這是可能退步的主要原因。

九個NN陣列僅train_price_factor_million改變，其他八項逐bytes保持。原Softplus倍率NN、512/256/128架構、seed42、30epochs、2048batch、AdamW與cosine排程保持。價格因子改變會依原公式連動初始ratio/bias；不是固定相同數值的初始化。query prior及query factor逐bytes等於各起點原版；不混signed head、Depthwise或其他新設定。

|起點|fit／query列|v22整體MAE|R117整體MAE|差vs v22|固定加權差|
|---|---:|---:|---:|---:|---:|
|development|240,316／81,234|1,407,387.07|1,405,925.19|-1,461.88|-1,528.45|
|confirmation|321,550／133,697|1,383,228.41|1,385,049.19|+1,820.78|+2,275.55|
|auxiliary|134,805／105,511|2,309,352.36|2,304,247.27|-5,105.09|-2,048.06|
|offset_2023_02_11|272,117／107,169|1,336,410.67|1,332,912.05|-3,498.62|-3,435.06|

單位新臺幣元，負值較好。離線MAE差不是Kaggle分數或預測向量差。

## 事前門檻與完整對照

```json
{
  "august": {
    "automatic_selection": false,
    "exploratory_submission": {
      "components": {
        "at_least_two_windows_overall_improve": true,
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
      "confirmation_fractional_improvement": -0.0013163269464747278,
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
  "reserved_exploratory_version": 32,
  "selected_submission": "v22"
}
```

只替換原prior NN槽；與原area NN各半後乘.75，再加.25原LGB，沿原reference、spatial各一次，最後與原H/U逐筆median。舊分支與控制是各起點自己的保存狀態，不是同seed重新訓練的matched control。四個新prior、四個NN和全部五層封存並成功退出後，另命令讀query價格評分。

## development完整診斷

prior／prepare實測14.813秒，native NN fit 29.953秒。

交易fold、初始化與fit價格因子變動：

```json
{
  "fold_policy": "transaction-permutation-5-fold",
  "fold_seed": 42,
  "folds": 5,
  "fold_rows": [
    48064,
    48063,
    48063,
    48063,
    48063
  ],
  "chronological_oof": false,
  "fit_factor_change_summary": {
    "changed_fraction": 0.9806130261821935,
    "changed_rows": 235657,
    "mean_absolute_change_million": 0.7517899033457882,
    "this_is_not_query_MAE": true
  },
  "original_initialization": {
    "initial_head_bias_float32": 0.3675074279308319,
    "initial_head_bias_formula_float64": 0.36750743408829034,
    "initial_training_median_ratio": 0.8936894499827281
  },
  "transaction_initialization": {
    "initial_head_bias_float32": 0.40230312943458557,
    "initial_head_bias_formula_float64": 0.40230312699625925,
    "initial_training_median_ratio": 0.9143947432344293
  },
  "initial_bias_changes_linked_to_fit_factor": true,
  "numeric_input_width": 144,
  "categorical_input_width": 23,
  "encoder_refit": false,
  "full_fit_state_and_query_prior_and_factor_exact_original": true,
  "query_prior_level_counts": {
    "address_city|address_district|transaction_type|building_type": 73327,
    "address_city|transaction_type|building_type": 6934,
    "global": 0,
    "transaction_type": 126,
    "transaction_type|building_type": 847
  },
  "legacy_state_group_rule_string_is_not_transaction_fold_evidence": true
}
```

|層|總價MAE|
|---|---:|
|base|1,466,441.30|
|native|1,677,878.51|
|reference|1,443,199.02|
|spatial|1,436,596.77|
|median|1,405,925.19|

季度／至少500列群退步超5%（重疊群不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-1,461.88|[-3,218.22, -40.35]|
|month|weighted|-1,528.45|[-3,759.34, +464.35]|
|project|overall|-1,461.88|[-5,414.31, +2,798.59]|
|project|weighted|-1,528.45|[-5,412.28, +2,663.47]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R117",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Transaction OOF excludes each whole transaction fold, not all same-project peers; it is neither chronological OOF nor per-row as-of prediction. Prior factors and the linked initial ratio/head bias change together. Original group OOF may generalize better to new projects. Conditional single-seed bootstrap omits cumulative selection, retraining and future shift; not Private probability. R67/R100/R65/R102 contrary evidence and v24 comparisons remain; native improvement alone is insufficient.",
  "control": "v22",
  "difference": "R117 minus v22",
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
          "bootstrap_mean_delta": -1476.6052270165953,
          "candidate_mae": 1405925.19171015,
          "control_mae": 1407387.0708058395,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.9765,
          "percentile_2_5": -3218.220595337596,
          "percentile_97_5": -40.34703912331056,
          "point_delta": -1461.8790956893936,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -1609.3527500177188,
          "candidate_mae": 1398189.290701774,
          "control_mae": 1399717.744053713,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.9425,
          "percentile_2_5": -3759.341477830324,
          "percentile_97_5": 464.3509662081191,
          "point_delta": -1528.453351939097,
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
          "bootstrap_mean_delta": -1505.829295619802,
          "candidate_mae": 1405925.19171015,
          "control_mae": 1407387.0708058395,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.756,
          "percentile_2_5": -5414.313081887475,
          "percentile_97_5": 2798.5933774632645,
          "point_delta": -1461.8790956893936,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -1561.6384235444782,
          "candidate_mae": 1398189.290701774,
          "control_mae": 1399717.744053713,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.7685,
          "percentile_2_5": -5412.281103762967,
          "percentile_97_5": 2663.4685871764127,
          "point_delta": -1528.453351939097,
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

v24次對照與反證（不代替v22門檻）：

```json
{
  "both_point_metrics_improve": true,
  "candidate": "R117",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 279735.0033029721,
      "control_mae": 240685.11008695266,
      "delta": 39049.89321601947,
      "relative_change_pct": 16.224474044909492,
      "rows": 3
    },
    "南投縣": {
      "candidate_mae": 816179.838555595,
      "control_mae": 815392.6297455619,
      "delta": 787.2088100331603,
      "relative_change_pct": 0.09654352778229107,
      "rows": 659
    },
    "嘉義市": {
      "candidate_mae": 1454128.7915943349,
      "control_mae": 1432770.9123328016,
      "delta": 21357.879261533264,
      "relative_change_pct": 1.49066951860147,
      "rows": 569
    },
    "嘉義縣": {
      "candidate_mae": 790360.7700056945,
      "control_mae": 798121.9796995061,
      "delta": -7761.209693811601,
      "relative_change_pct": -0.9724340252769015,
      "rows": 798
    },
    "基隆市": {
      "candidate_mae": 1574122.0450643224,
      "control_mae": 1573023.596581248,
      "delta": 1098.4484830743168,
      "relative_change_pct": 0.06983038814304149,
      "rows": 737
    },
    "宜蘭縣": {
      "candidate_mae": 840147.5904892137,
      "control_mae": 833869.7553878439,
      "delta": 6277.835101369768,
      "relative_change_pct": 0.7528555941508951,
      "rows": 2242
    },
    "屏東縣": {
      "candidate_mae": 811142.9244038523,
      "control_mae": 806099.9039080202,
      "delta": 5043.020495832083,
      "relative_change_pct": 0.6256073808448829,
      "rows": 615
    },
    "彰化縣": {
      "candidate_mae": 1085367.9242787932,
      "control_mae": 1099316.8196964574,
      "delta": -13948.8954176642,
      "relative_change_pct": -1.2688694621734034,
      "rows": 2246
    },
    "新北市": {
      "candidate_mae": 1149104.34756516,
      "control_mae": 1175055.059966161,
      "delta": -25950.712401001016,
      "relative_change_pct": -2.2084677803734865,
      "rows": 15979
    },
    "新竹市": {
      "candidate_mae": 4529206.6375304945,
      "control_mae": 3988384.5966167687,
      "delta": 540822.0409137257,
      "relative_change_pct": 13.559927028413695,
      "rows": 1250
    },
    "新竹縣": {
      "candidate_mae": 2988892.0653217793,
      "control_mae": 2973954.3281137035,
      "delta": 14937.737208075821,
      "relative_change_pct": 0.5022853601639005,
      "rows": 3639
    },
    "桃園市": {
      "candidate_mae": 965148.1025658292,
      "control_mae": 968876.870646954,
      "delta": -3728.768081124872,
      "relative_change_pct": -0.38485469042470166,
      "rows": 15846
    },
    "澎湖縣": {
      "candidate_mae": 905655.6976841474,
      "control_mae": 875582.3630456572,
      "delta": 30073.334638490225,
      "relative_change_pct": 3.4346665611082043,
      "rows": 28
    },
    "臺中市": {
      "candidate_mae": 1575819.2009152335,
      "control_mae": 1572125.9434656193,
      "delta": 3693.257449614117,
      "relative_change_pct": 0.2349212202091546,
      "rows": 14719
    },
    "臺北市": {
      "candidate_mae": 4369038.759072104,
      "control_mae": 4511801.4476898415,
      "delta": -142762.68861773796,
      "relative_change_pct": -3.164205922466649,
      "rows": 2698
    },
    "臺南市": {
      "candidate_mae": 1224847.0917218719,
      "control_mae": 1239277.1706731338,
      "delta": -14430.078951261938,
      "relative_change_pct": -1.164394801481254,
      "rows": 7530
    },
    "臺東縣": {
      "candidate_mae": 621644.2155407564,
      "control_mae": 642748.5280153912,
      "delta": -21104.312474634848,
      "relative_change_pct": -3.2834478111989522,
      "rows": 265
    },
    "花蓮縣": {
      "candidate_mae": 557662.6688718565,
      "control_mae": 558290.847066315,
      "delta": -628.17819445848,
      "relative_change_pct": -0.11251808940795041,
      "rows": 535
    },
    "苗栗縣": {
      "candidate_mae": 1319105.4345234672,
      "control_mae": 1309189.6028082767,
      "delta": 9915.83171519055,
      "relative_change_pct": 0.7574022657925635,
      "rows": 2402
    },
    "金門縣": {
      "candidate_mae": 676323.0899025114,
      "control_mae": 664978.0807066441,
      "delta": 11345.009195867344,
      "relative_change_pct": 1.7060726548778091,
      "rows": 29
    },
    "雲林縣": {
      "candidate_mae": 916711.7963735203,
      "control_mae": 900864.7686889353,
      "delta": 15847.027684584958,
      "relative_change_pct": 1.759090624406122,
      "rows": 1350
    },
    "高雄市": {
      "candidate_mae": 975203.82760884,
      "control_mae": 970391.5011320283,
      "delta": 4812.326476811664,
      "relative_change_pct": 0.4959159752736658,
      "rows": 7095
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1242740.470081904,
      "control_mae": 1257065.2649268839,
      "delta": -14324.794844979886,
      "relative_change_pct": -1.1395426510184479,
      "rows": 17608
    },
    "Q2": {
      "candidate_mae": 1315877.2379881144,
      "control_mae": 1304412.7913579252,
      "delta": 11464.446630189195,
      "relative_change_pct": 0.8788971333418488,
      "rows": 14193
    },
    "Q3": {
      "candidate_mae": 1445151.1414356926,
      "control_mae": 1449548.9071540441,
      "delta": -4397.765718351584,
      "relative_change_pct": -0.3033885712063271,
      "rows": 22487
    },
    "Q4": {
      "candidate_mae": 1527254.266679711,
      "control_mae": 1524529.5621492898,
      "delta": 2724.704530421179,
      "relative_change_pct": 0.1787242830883434,
      "rows": 26946
    }
  },
  "months": {
    "2022-08": {
      "candidate_mae": 949766.6220370523,
      "control_mae": 950683.2454188134,
      "delta": -916.6233817611355,
      "relative_change_pct": -0.09641732787215859,
      "rows": 3937
    },
    "2022-09": {
      "candidate_mae": 1182396.97714142,
      "control_mae": 1183834.4147522436,
      "delta": -1437.437610823661,
      "relative_change_pct": -0.12142218480145235,
      "rows": 6480
    },
    "2022-10": {
      "candidate_mae": 1464501.3984518268,
      "control_mae": 1501876.047428305,
      "delta": -37374.64897647826,
      "relative_change_pct": -2.488530863813674,
      "rows": 5654
    },
    "2022-11": {
      "candidate_mae": 1539087.797541993,
      "control_mae": 1518956.0042070919,
      "delta": 20131.7933349011,
      "relative_change_pct": 1.3253704043528285,
      "rows": 4732
    },
    "2022-12": {
      "candidate_mae": 1224004.6993001553,
      "control_mae": 1228901.1954487665,
      "delta": -4896.496148611186,
      "relative_change_pct": -0.3984450635043201,
      "rows": 6122
    },
    "2023-01": {
      "candidate_mae": 1262374.859039624,
      "control_mae": 1241965.4282693113,
      "delta": 20409.43077031267,
      "relative_change_pct": 1.6433171411826957,
      "rows": 3545
    },
    "2023-02": {
      "candidate_mae": 1168296.898194649,
      "control_mae": 1181921.8231404028,
      "delta": -13624.924945753766,
      "relative_change_pct": -1.1527771701136644,
      "rows": 4227
    },
    "2023-03": {
      "candidate_mae": 1415885.241396961,
      "control_mae": 1425446.287641478,
      "delta": -9561.046244516969,
      "relative_change_pct": -0.6707405482346537,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1511533.575738425,
      "control_mae": 1508305.4461258438,
      "delta": 3228.129612581106,
      "relative_change_pct": 0.21402359985324687,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1539542.351538595,
      "control_mae": 1532629.9115433628,
      "delta": 6912.439995232271,
      "relative_change_pct": 0.451018210147773,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1552736.4276436698,
      "control_mae": 1543362.271613415,
      "delta": 9374.156030254671,
      "relative_change_pct": 0.6073853302410344,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1523218.2717995953,
      "control_mae": 1539075.911771725,
      "delta": -15857.639972129604,
      "relative_change_pct": -1.0303351414209907,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1448214.1984197604,
      "control_mae": 1454058.0801178366,
      "delta": -5843.881698076148,
      "relative_change_pct": -0.40190153185645516,
      "rows": 2324
    }
  },
  "overall_mae": {
    "candidate_mae": 1405925.1917101631,
    "control_mae": 1407340.7168109824,
    "delta": -1415.5251008193009,
    "relative_change_pct": -0.10058154957861691
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 720842.5837444721,
      "control_mae": 726223.9901538457,
      "delta": -5381.406409373623,
      "relative_change_pct": -0.7410119305248519,
      "rows": 22139
    },
    {
      "band": "10-20m",
      "candidate_mae": 1120757.0874671072,
      "control_mae": 1119987.3894261469,
      "delta": 769.6980409603566,
      "relative_change_pct": 0.06872381316317591,
      "rows": 43914
    },
    {
      "band": "20-50m",
      "candidate_mae": 2679530.9380360935,
      "control_mae": 2676120.810418359,
      "delta": 3410.1276177344844,
      "relative_change_pct": 0.12742801462693973,
      "rows": 14197
    },
    {
      "band": "50-100m",
      "candidate_mae": 6567958.302613317,
      "control_mae": 6578422.082288335,
      "delta": -10463.779675018042,
      "relative_change_pct": -0.15906215113789365,
      "rows": 802
    },
    {
      "band": ">=100m",
      "candidate_mae": 31453093.609009217,
      "control_mae": 31835906.574960757,
      "delta": -382812.9659515396,
      "relative_change_pct": -1.2024566193840562,
      "rows": 182
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 4529206.6375304945,
      "control_mae": 3988384.5966167687,
      "delta": 540822.0409137257,
      "group": "新竹市",
      "relative_change_pct": 13.559927028413695,
      "rows": 1250,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 1057271.3480485177,
      "control_mae": 1056814.584844697,
      "delta": 456.76320382067934,
      "relative_change_pct": 0.04322075133811694,
      "rows": 6145
    },
    "1": {
      "candidate_mae": 1194892.9471022948,
      "control_mae": 1196559.960172717,
      "delta": -1667.0130704222247,
      "relative_change_pct": -0.13931713628304931,
      "rows": 6235
    },
    "10": {
      "candidate_mae": 1561156.649853505,
      "control_mae": 1557748.2179694306,
      "delta": 3408.431884074351,
      "relative_change_pct": 0.21880505750263926,
      "rows": 11490
    },
    "11": {
      "candidate_mae": 1524754.0520471917,
      "control_mae": 1536870.8291656736,
      "delta": -12116.777118481928,
      "relative_change_pct": -0.7884056934739145,
      "rows": 6914
    },
    "2": {
      "candidate_mae": 1517804.9422840883,
      "control_mae": 1564599.8870093762,
      "delta": -46794.944725287845,
      "relative_change_pct": -2.990856966935689,
      "rows": 5228
    },
    "3": {
      "candidate_mae": 1448062.1779834642,
      "control_mae": 1423037.1772045153,
      "delta": 25025.00077894889,
      "relative_change_pct": 1.758562684083155,
      "rows": 5046
    },
    "4": {
      "candidate_mae": 1219157.4714918674,
      "control_mae": 1216912.2856501525,
      "delta": 2245.1858417149633,
      "relative_change_pct": 0.18449857628937005,
      "rows": 5881
    },
    "5": {
      "candidate_mae": 1285811.0222954915,
      "control_mae": 1278696.876809706,
      "delta": 7114.145485785557,
      "relative_change_pct": 0.5563590257243018,
      "rows": 3266
    },
    "6": {
      "candidate_mae": 1180076.4358754505,
      "control_mae": 1193017.763002033,
      "delta": -12941.327126582619,
      "relative_change_pct": -1.084755611183684,
      "rows": 5055
    },
    "7": {
      "candidate_mae": 1476456.637952257,
      "control_mae": 1487234.18211577,
      "delta": -10777.544163512997,
      "relative_change_pct": -0.7246702834775248,
      "rows": 7516
    },
    "8": {
      "candidate_mae": 1570284.612948493,
      "control_mae": 1560492.9821896846,
      "delta": 9791.63075880846,
      "relative_change_pct": 0.6274703488296909,
      "rows": 9057
    },
    "9": {
      "candidate_mae": 1477105.153820954,
      "control_mae": 1468927.3561406387,
      "delta": 8177.797680315329,
      "relative_change_pct": 0.5567189994882474,
      "rows": 9401
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1699013.853015954,
      "control_mae": 1696977.6628507746,
      "delta": 2036.190165179316,
      "relative_change_pct": 0.11998921433996332,
      "rows": 52253
    },
    "seen": {
      "candidate_mae": 877483.7363010505,
      "control_mae": 885122.7346358591,
      "delta": -7638.998334808624,
      "relative_change_pct": -0.8630439639482675,
      "rows": 28981
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 1398189.2907017868,
    "control_mae": 1399695.9217754127,
    "delta": -1506.6310736259911,
    "relative_change_pct": -0.1076398845054102
  }
}
```

完整城市／月份／新舊案／季度／價格帶／尾部：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 279735.0033029721,
      "rows": 3
    },
    "南投縣": {
      "mae": 816179.838555595,
      "rows": 659
    },
    "嘉義市": {
      "mae": 1454128.7915943349,
      "rows": 569
    },
    "嘉義縣": {
      "mae": 790360.7700056945,
      "rows": 798
    },
    "基隆市": {
      "mae": 1574122.0450643224,
      "rows": 737
    },
    "宜蘭縣": {
      "mae": 840147.5904892137,
      "rows": 2242
    },
    "屏東縣": {
      "mae": 811142.9244038523,
      "rows": 615
    },
    "彰化縣": {
      "mae": 1085367.9242787932,
      "rows": 2246
    },
    "新北市": {
      "mae": 1149104.34756516,
      "rows": 15979
    },
    "新竹市": {
      "mae": 4529206.6375304945,
      "rows": 1250
    },
    "新竹縣": {
      "mae": 2988892.0653217793,
      "rows": 3639
    },
    "桃園市": {
      "mae": 965148.1025658292,
      "rows": 15846
    },
    "澎湖縣": {
      "mae": 905655.6976841474,
      "rows": 28
    },
    "臺中市": {
      "mae": 1575819.2009152335,
      "rows": 14719
    },
    "臺北市": {
      "mae": 4369038.759072104,
      "rows": 2698
    },
    "臺南市": {
      "mae": 1224847.0917218719,
      "rows": 7530
    },
    "臺東縣": {
      "mae": 621644.2155407564,
      "rows": 265
    },
    "花蓮縣": {
      "mae": 557662.6688718565,
      "rows": 535
    },
    "苗栗縣": {
      "mae": 1319105.4345234672,
      "rows": 2402
    },
    "金門縣": {
      "mae": 676323.0899025114,
      "rows": 29
    },
    "雲林縣": {
      "mae": 916711.7963735203,
      "rows": 1350
    },
    "高雄市": {
      "mae": 975203.82760884,
      "rows": 7095
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1242740.470081904,
      "rows": 17608
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1315877.2379881144,
      "rows": 14193
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1445151.1414356926,
      "rows": 22487
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1527254.266679711,
      "rows": 26946
    }
  },
  "months": {
    "2022-08": {
      "mae": 949766.6220370523,
      "rows": 3937
    },
    "2022-09": {
      "mae": 1182396.97714142,
      "rows": 6480
    },
    "2022-10": {
      "mae": 1464501.3984518268,
      "rows": 5654
    },
    "2022-11": {
      "mae": 1539087.797541993,
      "rows": 4732
    },
    "2022-12": {
      "mae": 1224004.6993001553,
      "rows": 6122
    },
    "2023-01": {
      "mae": 1262374.859039624,
      "rows": 3545
    },
    "2023-02": {
      "mae": 1168296.898194649,
      "rows": 4227
    },
    "2023-03": {
      "mae": 1415885.241396961,
      "rows": 7500
    },
    "2023-04": {
      "mae": 1511533.575738425,
      "rows": 8639
    },
    "2023-05": {
      "mae": 1539542.351538595,
      "rows": 8750
    },
    "2023-06": {
      "mae": 1552736.4276436698,
      "rows": 13180
    },
    "2023-07": {
      "mae": 1523218.2717995953,
      "rows": 6144
    },
    "2023-08": {
      "mae": 1448214.1984197604,
      "rows": 2324
    }
  },
  "overall_mae": 1405925.1917101631,
  "price_bands": [
    {
      "absolute_error_share": 0.13973280703575336,
      "absolute_error_sum": 15958733961.51887,
      "band": "<10m",
      "mae": 720842.5837444721,
      "mean_signed_error_prediction_minus_truth": 363522.3321090219,
      "rows": 22139,
      "underprediction_fraction": 0.34084647003026336
    },
    {
      "absolute_error_share": 0.43093765103802933,
      "absolute_error_sum": 49216926739.03055,
      "band": "10-20m",
      "mae": 1120757.0874671072,
      "mean_signed_error_prediction_minus_truth": -65585.49650648716,
      "rows": 43914,
      "underprediction_fraction": 0.487361661429157
    },
    {
      "absolute_error_share": 0.3330851774792504,
      "absolute_error_sum": 38041300727.298416,
      "band": "20-50m",
      "mae": 2679530.9380360935,
      "mean_signed_error_prediction_minus_truth": -863079.9416649276,
      "rows": 14197,
      "underprediction_fraction": 0.540114108614496
    },
    {
      "absolute_error_share": 0.046121635987503844,
      "absolute_error_sum": 5267502558.69588,
      "band": "50-100m",
      "mae": 6567958.302613317,
      "mean_signed_error_prediction_minus_truth": -1542124.237803236,
      "rows": 802,
      "underprediction_fraction": 0.4937655860349127
    },
    {
      "absolute_error_share": 0.050122728459463056,
      "absolute_error_sum": 5724463036.839678,
      "band": ">=100m",
      "mae": 31453093.609009217,
      "mean_signed_error_prediction_minus_truth": 2260533.233384117,
      "rows": 182,
      "underprediction_fraction": 0.3626373626373626
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 1057271.3480485177,
      "rows": 6145
    },
    "1": {
      "mae": 1194892.9471022948,
      "rows": 6235
    },
    "10": {
      "mae": 1561156.649853505,
      "rows": 11490
    },
    "11": {
      "mae": 1524754.0520471917,
      "rows": 6914
    },
    "2": {
      "mae": 1517804.9422840883,
      "rows": 5228
    },
    "3": {
      "mae": 1448062.1779834642,
      "rows": 5046
    },
    "4": {
      "mae": 1219157.4714918674,
      "rows": 5881
    },
    "5": {
      "mae": 1285811.0222954915,
      "rows": 3266
    },
    "6": {
      "mae": 1180076.4358754505,
      "rows": 5055
    },
    "7": {
      "mae": 1476456.637952257,
      "rows": 7516
    },
    "8": {
      "mae": 1570284.612948493,
      "rows": 9057
    },
    "9": {
      "mae": 1477105.153820954,
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
      "mae": 1699013.853015954,
      "rows": 52253
    },
    "seen": {
      "mae": 877483.7363010505,
      "rows": 28981
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 788278.2986260345,
      "0.9": 3148570.064078653,
      "0.95": 4802366.615242496,
      "0.99": 8985802.96658688
    },
    "absolute_error_sum": 114208927023.38339,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 527435029.011781,
    "mean_signed_error_prediction_minus_truth": -97380.5685437564,
    "top10_absolute_error_sum": 2116839504.6713252,
    "top10_error_rows": 10,
    "top10_error_share": 0.018534798985003324,
    "top1pct_absolute_error_sum": 13889286791.692059,
    "top1pct_error_rows": 813,
    "top1pct_error_share": 0.12161296978867803,
    "underprediction_fraction": 0.45643449787034984
  },
  "weighted_mae": 1398189.2907017868
}
```

## confirmation完整診斷

prior／prepare實測19.906秒，native NN fit 35.468秒。

交易fold、初始化與fit價格因子變動：

```json
{
  "fold_policy": "transaction-permutation-5-fold",
  "fold_seed": 42,
  "folds": 5,
  "fold_rows": [
    64310,
    64310,
    64310,
    64310,
    64310
  ],
  "chronological_oof": false,
  "fit_factor_change_summary": {
    "changed_fraction": 0.9762618566319391,
    "changed_rows": 313917,
    "mean_absolute_change_million": 0.6731677610480897,
    "this_is_not_query_MAE": true
  },
  "original_initialization": {
    "initial_head_bias_float32": 0.34617486596107483,
    "initial_head_bias_formula_float64": 0.3461748681408253,
    "initial_training_median_ratio": 0.8811400402859173
  },
  "transaction_initialization": {
    "initial_head_bias_float32": 0.37451624870300293,
    "initial_head_bias_formula_float64": 0.37451623527471534,
    "initial_training_median_ratio": 0.8978365811944984
  },
  "initial_bias_changes_linked_to_fit_factor": true,
  "numeric_input_width": 144,
  "categorical_input_width": 23,
  "encoder_refit": false,
  "full_fit_state_and_query_prior_and_factor_exact_original": true,
  "query_prior_level_counts": {
    "address_city|address_district|transaction_type|building_type": 126458,
    "address_city|transaction_type|building_type": 6024,
    "global": 3,
    "transaction_type": 23,
    "transaction_type|building_type": 1189
  },
  "legacy_state_group_rule_string_is_not_transaction_fold_evidence": true
}
```

|層|總價MAE|
|---|---:|
|base|1,632,391.98|
|native|1,596,349.94|
|reference|1,397,651.94|
|spatial|1,394,570.21|
|median|1,385,049.19|

季度／至少500列群退步超5%（重疊群不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+1,820.78|[-3,402.75, +6,996.41]|
|month|weighted|+2,275.55|[-3,651.37, +6,744.48]|
|project|overall|+1,820.78|[-5,641.82, +8,332.83]|
|project|weighted|+2,275.55|[-5,981.71, +9,552.96]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R117",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Transaction OOF excludes each whole transaction fold, not all same-project peers; it is neither chronological OOF nor per-row as-of prediction. Prior factors and the linked initial ratio/head bias change together. Original group OOF may generalize better to new projects. Conditional single-seed bootstrap omits cumulative selection, retraining and future shift; not Private probability. R67/R100/R65/R102 contrary evidence and v24 comparisons remain; native improvement alone is insufficient.",
  "control": "v22",
  "difference": "R117 minus v22",
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
          "bootstrap_mean_delta": 1611.59853700215,
          "candidate_mae": 1385049.1862731676,
          "control_mae": 1383228.4054499408,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.2945,
          "percentile_2_5": -3402.7497194068505,
          "percentile_97_5": 6996.4120677069295,
          "point_delta": 1820.7808232267853,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 1877.562605990407,
          "candidate_mae": 1449428.9633735057,
          "control_mae": 1447153.4154201136,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.2475,
          "percentile_2_5": -3651.3708110960665,
          "percentile_97_5": 6744.476000428201,
          "point_delta": 2275.547953392146,
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
          "bootstrap_mean_delta": 1776.390327368235,
          "candidate_mae": 1385049.1862731676,
          "control_mae": 1383228.4054499408,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.2965,
          "percentile_2_5": -5641.821386147273,
          "percentile_97_5": 8332.833955099995,
          "point_delta": 1820.7808232267853,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 2225.4681789375886,
          "candidate_mae": 1449428.9633735057,
          "control_mae": 1447153.4154201136,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.27,
          "percentile_2_5": -5981.709984406101,
          "percentile_97_5": 9552.960423308516,
          "point_delta": 2275.547953392146,
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

v24次對照與反證（不代替v22門檻）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R117",
  "cities": {
    "南投縣": {
      "candidate_mae": 957599.4097196419,
      "control_mae": 965273.7684037857,
      "delta": -7674.35868414375,
      "relative_change_pct": -0.7950447774867403,
      "rows": 1015
    },
    "嘉義市": {
      "candidate_mae": 1188793.7801414107,
      "control_mae": 1192323.4691625503,
      "delta": -3529.6890211396385,
      "relative_change_pct": -0.29603451684288146,
      "rows": 689
    },
    "嘉義縣": {
      "candidate_mae": 1454205.8612920668,
      "control_mae": 1401441.997463974,
      "delta": 52763.86382809281,
      "relative_change_pct": 3.7649695045227287,
      "rows": 2251
    },
    "基隆市": {
      "candidate_mae": 1821642.7963234452,
      "control_mae": 1811344.9669182824,
      "delta": 10297.829405162716,
      "relative_change_pct": 0.5685183989377157,
      "rows": 1100
    },
    "宜蘭縣": {
      "candidate_mae": 941079.1268643657,
      "control_mae": 940693.1296201872,
      "delta": 385.9972441784339,
      "relative_change_pct": 0.041033279825726326,
      "rows": 2907
    },
    "屏東縣": {
      "candidate_mae": 759494.9418293188,
      "control_mae": 757610.0933880565,
      "delta": 1884.8484412622638,
      "relative_change_pct": 0.24878871832780916,
      "rows": 1013
    },
    "彰化縣": {
      "candidate_mae": 1079872.4289088557,
      "control_mae": 1068085.3267788775,
      "delta": 11787.102129978128,
      "relative_change_pct": 1.103573079271257,
      "rows": 3295
    },
    "新北市": {
      "candidate_mae": 1318087.1221433086,
      "control_mae": 1308034.3659481003,
      "delta": 10052.75619520829,
      "relative_change_pct": 0.768539149804506,
      "rows": 21822
    },
    "新竹市": {
      "candidate_mae": 2582720.248336649,
      "control_mae": 2414804.346618525,
      "delta": 167915.90171812382,
      "relative_change_pct": 6.953602760954864,
      "rows": 2131
    },
    "新竹縣": {
      "candidate_mae": 2027346.5540100844,
      "control_mae": 2023872.173760328,
      "delta": 3474.3802497563884,
      "relative_change_pct": 0.17166994510829384,
      "rows": 5589
    },
    "桃園市": {
      "candidate_mae": 1218734.7888823058,
      "control_mae": 1236542.7211960873,
      "delta": -17807.932313781464,
      "relative_change_pct": -1.4401388652836957,
      "rows": 26327
    },
    "澎湖縣": {
      "candidate_mae": 2630675.193336504,
      "control_mae": 2558605.6252502515,
      "delta": 72069.56808625255,
      "relative_change_pct": 2.8167517250417826,
      "rows": 30
    },
    "臺中市": {
      "candidate_mae": 1419643.5053029011,
      "control_mae": 1396562.2952675254,
      "delta": 23081.210035375785,
      "relative_change_pct": 1.652716109663719,
      "rows": 27157
    },
    "臺北市": {
      "candidate_mae": 4116620.16977752,
      "control_mae": 4016190.85969717,
      "delta": 100429.31008034991,
      "relative_change_pct": 2.5006109915782866,
      "rows": 4957
    },
    "臺南市": {
      "candidate_mae": 903570.1074147354,
      "control_mae": 924730.213914743,
      "delta": -21160.106500007678,
      "relative_change_pct": -2.288246472495876,
      "rows": 9822
    },
    "臺東縣": {
      "candidate_mae": 542793.5782790222,
      "control_mae": 511517.47384219157,
      "delta": 31276.104436830676,
      "relative_change_pct": 6.114376543562552,
      "rows": 115
    },
    "花蓮縣": {
      "candidate_mae": 618745.6998222512,
      "control_mae": 617584.5960098031,
      "delta": 1161.1038124480983,
      "relative_change_pct": 0.1880072495249975,
      "rows": 583
    },
    "苗栗縣": {
      "candidate_mae": 1493117.4610945142,
      "control_mae": 1487508.760371721,
      "delta": 5608.70072279335,
      "relative_change_pct": 0.3770532901864567,
      "rows": 4001
    },
    "金門縣": {
      "candidate_mae": 768414.159166023,
      "control_mae": 790515.819334728,
      "delta": -22101.660168704926,
      "relative_change_pct": -2.795852989672611,
      "rows": 86
    },
    "雲林縣": {
      "candidate_mae": 898514.2559596185,
      "control_mae": 920111.576360945,
      "delta": -21597.32040132652,
      "relative_change_pct": -2.347250154893632,
      "rows": 2364
    },
    "高雄市": {
      "candidate_mae": 1025447.2738559459,
      "control_mae": 1000969.9302470884,
      "delta": 24477.34360885748,
      "relative_change_pct": 2.4453625298029955,
      "rows": 16443
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 908250.1534932358,
      "control_mae": 912993.6515105661,
      "delta": -4743.498017330305,
      "relative_change_pct": -0.5195543265259395,
      "rows": 26579
    },
    "Q2": {
      "candidate_mae": 1174382.5629296068,
      "control_mae": 1165407.6059651442,
      "delta": 8974.95696446253,
      "relative_change_pct": 0.7701131276751731,
      "rows": 31157
    },
    "Q3": {
      "candidate_mae": 1489799.1719515522,
      "control_mae": 1475619.1470166727,
      "delta": 14180.024934879504,
      "relative_change_pct": 0.9609542518845676,
      "rows": 37236
    },
    "Q4": {
      "candidate_mae": 1781075.2936729263,
      "control_mae": 1757752.2708320958,
      "delta": 23323.022840830497,
      "relative_change_pct": 1.3268663183003426,
      "rows": 38725
    }
  },
  "months": {
    "2023-08": {
      "candidate_mae": 664459.1123054287,
      "control_mae": 661692.706617775,
      "delta": 2766.4056876536924,
      "relative_change_pct": 0.41808012389831267,
      "rows": 5076
    },
    "2023-09": {
      "candidate_mae": 811502.6084057463,
      "control_mae": 819211.9589262545,
      "delta": -7709.350520508247,
      "relative_change_pct": -0.9410690892028645,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1043195.3367916133,
      "control_mae": 1031792.2770912236,
      "delta": 11403.059700389626,
      "relative_change_pct": 1.1051700961104838,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1173549.3462480968,
      "control_mae": 1207667.0396223704,
      "delta": -34117.6933742736,
      "relative_change_pct": -2.8250910437153256,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1126326.259124352,
      "control_mae": 1122539.8496172,
      "delta": 3786.409507151926,
      "relative_change_pct": 0.337307357813902,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1204058.4221089447,
      "control_mae": 1177153.3779831533,
      "delta": 26905.044125791406,
      "relative_change_pct": 2.285602252774273,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1189535.2204327958,
      "control_mae": 1170552.7720134521,
      "delta": 18982.44841934368,
      "relative_change_pct": 1.6216653254079458,
      "rows": 7388
    },
    "2024-03": {
      "candidate_mae": 1402430.0862294587,
      "control_mae": 1396897.8203355782,
      "delta": 5532.265893880511,
      "relative_change_pct": 0.39603941056701547,
      "rows": 12933
    },
    "2024-04": {
      "candidate_mae": 1608630.9032990239,
      "control_mae": 1587211.1045129716,
      "delta": 21419.79878605227,
      "relative_change_pct": 1.3495242520133979,
      "rows": 13856
    },
    "2024-05": {
      "candidate_mae": 1699669.0041339754,
      "control_mae": 1671790.446103836,
      "delta": 27878.558030139422,
      "relative_change_pct": 1.6675868734093644,
      "rows": 14480
    },
    "2024-06": {
      "candidate_mae": 1803033.682357434,
      "control_mae": 1771497.5397129185,
      "delta": 31536.142644515494,
      "relative_change_pct": 1.7801968073648078,
      "rows": 14671
    },
    "2024-07": {
      "candidate_mae": 1765852.1198413707,
      "control_mae": 1756731.8339828255,
      "delta": 9120.28585854522,
      "relative_change_pct": 0.519162098740358,
      "rows": 12090
    },
    "2024-08": {
      "candidate_mae": 1912937.9435570117,
      "control_mae": 1893707.7653277898,
      "delta": 19230.17822922184,
      "relative_change_pct": 1.0154776033192856,
      "rows": 3109
    }
  },
  "overall_mae": {
    "candidate_mae": 1385049.1862731886,
    "control_mae": 1373195.9153073074,
    "delta": 11853.27096588118,
    "relative_change_pct": 0.8631886268922186
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 648272.8281098258,
      "control_mae": 652232.5472355905,
      "delta": -3959.7191257646773,
      "relative_change_pct": -0.6071023506182683,
      "rows": 25585
    },
    {
      "band": "10-20m",
      "candidate_mae": 1043817.5163552392,
      "control_mae": 1040802.0332451425,
      "delta": 3015.483110096655,
      "relative_change_pct": 0.28972686579931106,
      "rows": 74109
    },
    {
      "band": "20-50m",
      "candidate_mae": 2328982.841285219,
      "control_mae": 2283352.845939235,
      "delta": 45629.99534598412,
      "relative_change_pct": 1.998376879295441,
      "rows": 31959
    },
    {
      "band": "50-100m",
      "candidate_mae": 6413239.827664719,
      "control_mae": 6217166.085579387,
      "delta": 196073.74208533205,
      "relative_change_pct": 3.1537478553150096,
      "rows": 1725
    },
    {
      "band": ">=100m",
      "candidate_mae": 17993063.018966377,
      "control_mae": 19039879.576983564,
      "delta": -1046816.5580171868,
      "relative_change_pct": -5.498020897582962,
      "rows": 319
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 2582720.248336649,
      "control_mae": 2414804.346618525,
      "delta": 167915.90171812382,
      "group": "新竹市",
      "relative_change_pct": 6.953602760954864,
      "rows": 2131,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 681409.0820641867,
      "control_mae": 678922.6899683854,
      "delta": 2486.392095801304,
      "relative_change_pct": 0.36622610093014935,
      "rows": 7252
    },
    "1": {
      "candidate_mae": 879886.3591392922,
      "control_mae": 889826.2099094184,
      "delta": -9939.850770126213,
      "relative_change_pct": -1.1170552923068044,
      "rows": 9394
    },
    "10": {
      "candidate_mae": 1774335.421327419,
      "control_mae": 1753961.107622659,
      "delta": 20374.313704760047,
      "relative_change_pct": 1.1616171884435709,
      "rows": 14072
    },
    "11": {
      "candidate_mae": 1814034.7502345564,
      "control_mae": 1798487.2970159906,
      "delta": 15547.453218565788,
      "relative_change_pct": 0.8644738967220825,
      "rows": 11312
    },
    "2": {
      "candidate_mae": 1100689.591141923,
      "control_mae": 1105797.1911768373,
      "delta": -5107.600034914212,
      "relative_change_pct": -0.4618930194133052,
      "rows": 9933
    },
    "3": {
      "candidate_mae": 1166909.1151054485,
      "control_mae": 1182811.8951608276,
      "delta": -15902.78005537903,
      "relative_change_pct": -1.3444893579817034,
      "rows": 10809
    },
    "4": {
      "candidate_mae": 1123982.9047178433,
      "control_mae": 1113684.87976537,
      "delta": 10298.024952473352,
      "relative_change_pct": 0.9246803238132253,
      "rows": 9935
    },
    "5": {
      "candidate_mae": 1230226.3257131653,
      "control_mae": 1196689.8803220654,
      "delta": 33536.4453910999,
      "relative_change_pct": 2.8024341095016387,
      "rows": 10413
    },
    "6": {
      "candidate_mae": 1265138.710313376,
      "control_mae": 1253733.2273133975,
      "delta": 11405.482999978587,
      "relative_change_pct": 0.9097216817344144,
      "rows": 8995
    },
    "7": {
      "candidate_mae": 1428138.0619514117,
      "control_mae": 1422678.507141905,
      "delta": 5459.55480950675,
      "relative_change_pct": 0.38375183023427706,
      "rows": 12434
    },
    "8": {
      "candidate_mae": 1668427.8508188964,
      "control_mae": 1646182.6876466258,
      "delta": 22245.163172270637,
      "relative_change_pct": 1.3513180122233095,
      "rows": 14629
    },
    "9": {
      "candidate_mae": 1750304.5099178324,
      "control_mae": 1717746.4536748976,
      "delta": 32558.056242934894,
      "relative_change_pct": 1.89539359393123,
      "rows": 14519
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1814037.9057478507,
      "control_mae": 1792277.4115217023,
      "delta": 21760.494226148352,
      "relative_change_pct": 1.214125340544965,
      "rows": 76096
    },
    "seen": {
      "candidate_mae": 818317.2615298014,
      "control_mae": 819552.3060656169,
      "delta": -1235.044535815483,
      "relative_change_pct": -0.15069746331927225,
      "rows": 57601
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 1449428.9633735274,
    "control_mae": 1436088.882157809,
    "delta": 13340.081215718295,
    "relative_change_pct": 0.9289175190656743
  }
}
```

完整城市／月份／新舊案／季度／價格帶／尾部：

```json
{
  "cities": {
    "南投縣": {
      "mae": 957599.4097196419,
      "rows": 1015
    },
    "嘉義市": {
      "mae": 1188793.7801414107,
      "rows": 689
    },
    "嘉義縣": {
      "mae": 1454205.8612920668,
      "rows": 2251
    },
    "基隆市": {
      "mae": 1821642.7963234452,
      "rows": 1100
    },
    "宜蘭縣": {
      "mae": 941079.1268643657,
      "rows": 2907
    },
    "屏東縣": {
      "mae": 759494.9418293188,
      "rows": 1013
    },
    "彰化縣": {
      "mae": 1079872.4289088557,
      "rows": 3295
    },
    "新北市": {
      "mae": 1318087.1221433086,
      "rows": 21822
    },
    "新竹市": {
      "mae": 2582720.248336649,
      "rows": 2131
    },
    "新竹縣": {
      "mae": 2027346.5540100844,
      "rows": 5589
    },
    "桃園市": {
      "mae": 1218734.7888823058,
      "rows": 26327
    },
    "澎湖縣": {
      "mae": 2630675.193336504,
      "rows": 30
    },
    "臺中市": {
      "mae": 1419643.5053029011,
      "rows": 27157
    },
    "臺北市": {
      "mae": 4116620.16977752,
      "rows": 4957
    },
    "臺南市": {
      "mae": 903570.1074147354,
      "rows": 9822
    },
    "臺東縣": {
      "mae": 542793.5782790222,
      "rows": 115
    },
    "花蓮縣": {
      "mae": 618745.6998222512,
      "rows": 583
    },
    "苗栗縣": {
      "mae": 1493117.4610945142,
      "rows": 4001
    },
    "金門縣": {
      "mae": 768414.159166023,
      "rows": 86
    },
    "雲林縣": {
      "mae": 898514.2559596185,
      "rows": 2364
    },
    "高雄市": {
      "mae": 1025447.2738559459,
      "rows": 16443
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 908250.1534932358,
      "rows": 26579
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1174382.5629296068,
      "rows": 31157
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1489799.1719515522,
      "rows": 37236
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1781075.2936729263,
      "rows": 38725
    }
  },
  "months": {
    "2023-08": {
      "mae": 664459.1123054287,
      "rows": 5076
    },
    "2023-09": {
      "mae": 811502.6084057463,
      "rows": 8602
    },
    "2023-10": {
      "mae": 1043195.3367916133,
      "rows": 9855
    },
    "2023-11": {
      "mae": 1173549.3462480968,
      "rows": 10336
    },
    "2023-12": {
      "mae": 1126326.259124352,
      "rows": 10416
    },
    "2024-01": {
      "mae": 1204058.4221089447,
      "rows": 10885
    },
    "2024-02": {
      "mae": 1189535.2204327958,
      "rows": 7388
    },
    "2024-03": {
      "mae": 1402430.0862294587,
      "rows": 12933
    },
    "2024-04": {
      "mae": 1608630.9032990239,
      "rows": 13856
    },
    "2024-05": {
      "mae": 1699669.0041339754,
      "rows": 14480
    },
    "2024-06": {
      "mae": 1803033.682357434,
      "rows": 14671
    },
    "2024-07": {
      "mae": 1765852.1198413707,
      "rows": 12090
    },
    "2024-08": {
      "mae": 1912937.9435570117,
      "rows": 3109
    }
  },
  "overall_mae": 1385049.1862731886,
  "price_bands": [
    {
      "absolute_error_share": 0.0895687227787396,
      "absolute_error_sum": 16586060307.189894,
      "band": "<10m",
      "mae": 648272.8281098258,
      "mean_signed_error_prediction_minus_truth": 195898.88556483053,
      "rows": 25585,
      "underprediction_fraction": 0.3918311510650772
    },
    {
      "absolute_error_share": 0.41774251282474634,
      "absolute_error_sum": 77356272319.57042,
      "band": "10-20m",
      "mae": 1043817.5163552392,
      "mean_signed_error_prediction_minus_truth": -216372.5635979499,
      "rows": 74109,
      "underprediction_fraction": 0.534037701223873
    },
    {
      "absolute_error_share": 0.4019505357347215,
      "absolute_error_sum": 74431962624.63431,
      "band": "20-50m",
      "mae": 2328982.841285219,
      "mean_signed_error_prediction_minus_truth": -1126460.5108136896,
      "rows": 31959,
      "underprediction_fraction": 0.6268030914609343
    },
    {
      "absolute_error_share": 0.059741995058371224,
      "absolute_error_sum": 11062838702.72164,
      "band": "50-100m",
      "mae": 6413239.827664719,
      "mean_signed_error_prediction_minus_truth": -1575173.658839392,
      "rows": 1725,
      "underprediction_fraction": 0.5385507246376812
    },
    {
      "absolute_error_share": 0.030996233603421495,
      "absolute_error_sum": 5739787103.050274,
      "band": ">=100m",
      "mae": 17993063.018966377,
      "mean_signed_error_prediction_minus_truth": -4569075.153200321,
      "rows": 319,
      "underprediction_fraction": 0.5141065830721003
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 681409.0820641867,
      "rows": 7252
    },
    "1": {
      "mae": 879886.3591392922,
      "rows": 9394
    },
    "10": {
      "mae": 1774335.421327419,
      "rows": 14072
    },
    "11": {
      "mae": 1814034.7502345564,
      "rows": 11312
    },
    "2": {
      "mae": 1100689.591141923,
      "rows": 9933
    },
    "3": {
      "mae": 1166909.1151054485,
      "rows": 10809
    },
    "4": {
      "mae": 1123982.9047178433,
      "rows": 9935
    },
    "5": {
      "mae": 1230226.3257131653,
      "rows": 10413
    },
    "6": {
      "mae": 1265138.710313376,
      "rows": 8995
    },
    "7": {
      "mae": 1428138.0619514117,
      "rows": 12434
    },
    "8": {
      "mae": 1668427.8508188964,
      "rows": 14629
    },
    "9": {
      "mae": 1750304.5099178324,
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
      "mae": 1814037.9057478507,
      "rows": 76096
    },
    "seen": {
      "mae": 818317.2615298014,
      "rows": 57601
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 774592.3581235185,
      "0.9": 3182044.973871236,
      "0.95": 4553462.172423154,
      "0.99": 8936153.191457573
    },
    "absolute_error_sum": 185176921057.1665,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 194067147.86302137,
    "mean_signed_error_prediction_minus_truth": -382943.0901887689,
    "top10_absolute_error_sum": 995088883.7998161,
    "top10_error_rows": 10,
    "top10_error_share": 0.005373719781703355,
    "top1pct_absolute_error_sum": 19637942932.999886,
    "top1pct_error_rows": 1337,
    "top1pct_error_share": 0.10604962443963198,
    "underprediction_fraction": 0.5290096262444184
  },
  "weighted_mae": 1449428.9633735274
}
```

## auxiliary完整診斷

prior／prepare實測10.219秒，native NN fit 14.047秒。

交易fold、初始化與fit價格因子變動：

```json
{
  "fold_policy": "transaction-permutation-5-fold",
  "fold_seed": 42,
  "folds": 5,
  "fold_rows": [
    26961,
    26961,
    26961,
    26961,
    26961
  ],
  "chronological_oof": false,
  "fit_factor_change_summary": {
    "changed_fraction": 0.9626052446125886,
    "changed_rows": 129764,
    "mean_absolute_change_million": 1.4378678189849716,
    "this_is_not_query_MAE": true
  },
  "original_initialization": {
    "initial_head_bias_float32": 0.48296213150024414,
    "initial_head_bias_formula_float64": 0.48296212332121224,
    "initial_training_median_ratio": 0.9635057554878078
  },
  "transaction_initialization": {
    "initial_head_bias_float32": 0.4913296103477478,
    "initial_head_bias_formula_float64": 0.4913296251758708,
    "initial_training_median_ratio": 0.9686888679446387
  },
  "initial_bias_changes_linked_to_fit_factor": true,
  "numeric_input_width": 144,
  "categorical_input_width": 23,
  "encoder_refit": false,
  "full_fit_state_and_query_prior_and_factor_exact_original": true,
  "query_prior_level_counts": {
    "address_city|address_district|transaction_type|building_type": 73757,
    "address_city|transaction_type|building_type": 22202,
    "global": 0,
    "transaction_type": 220,
    "transaction_type|building_type": 9332
  },
  "legacy_state_group_rule_string_is_not_transaction_fold_evidence": true
}
```

|層|總價MAE|
|---|---:|
|base|2,544,904.01|
|native|2,740,784.15|
|reference|2,447,249.94|
|spatial|2,448,562.43|
|median|2,304,247.27|

季度／至少500列群退步超5%（重疊群不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-5,105.09|[-22,274.45, +3,414.53]|
|month|weighted|-2,048.06|[-11,508.69, +4,356.91]|
|project|overall|-5,105.09|[-23,889.24, +6,794.93]|
|project|weighted|-2,048.06|[-16,615.91, +7,313.95]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R117",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Transaction OOF excludes each whole transaction fold, not all same-project peers; it is neither chronological OOF nor per-row as-of prediction. Prior factors and the linked initial ratio/head bias change together. Original group OOF may generalize better to new projects. Conditional single-seed bootstrap omits cumulative selection, retraining and future shift; not Private probability. R67/R100/R65/R102 contrary evidence and v24 comparisons remain; native improvement alone is insufficient.",
  "control": "v22",
  "difference": "R117 minus v22",
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
          "bootstrap_mean_delta": -5472.382553837837,
          "candidate_mae": 2304247.2722650957,
          "control_mae": 2309352.3576965467,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.6955,
          "percentile_2_5": -22274.450389541977,
          "percentile_97_5": 3414.526073622983,
          "point_delta": -5105.085431450978,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -1847.2112298060836,
          "candidate_mae": 2021117.6011854005,
          "control_mae": 2023165.66227504,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.6755,
          "percentile_2_5": -11508.687204917806,
          "percentile_97_5": 4356.905562555079,
          "point_delta": -2048.061089639552,
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
          "bootstrap_mean_delta": -5084.386402918718,
          "candidate_mae": 2304247.2722650957,
          "control_mae": 2309352.3576965467,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.693,
          "percentile_2_5": -23889.241407282276,
          "percentile_97_5": 6794.925488642079,
          "point_delta": -5105.085431450978,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -2050.8659297140834,
          "candidate_mae": 2021117.6011854005,
          "control_mae": 2023165.66227504,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.588,
          "percentile_2_5": -16615.914038956376,
          "percentile_97_5": 7313.945074256218,
          "point_delta": -2048.061089639552,
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

v24次對照與反證（不代替v22門檻）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R117",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 466689.4190578535,
      "control_mae": 466689.4190578535,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1104478.5472281838,
      "control_mae": 1118876.9223743458,
      "delta": -14398.375146161998,
      "relative_change_pct": -1.2868596052198038,
      "rows": 869
    },
    "嘉義市": {
      "candidate_mae": 2096563.641256515,
      "control_mae": 2076628.4947043534,
      "delta": 19935.146552161546,
      "relative_change_pct": 0.9599765486700443,
      "rows": 731
    },
    "嘉義縣": {
      "candidate_mae": 1348566.80541551,
      "control_mae": 1377314.8007233748,
      "delta": -28747.995307864854,
      "relative_change_pct": -2.087249428581339,
      "rows": 1560
    },
    "基隆市": {
      "candidate_mae": 3391363.2304859813,
      "control_mae": 2649701.530279867,
      "delta": 741661.7002061144,
      "relative_change_pct": 27.990386529602024,
      "rows": 1577
    },
    "宜蘭縣": {
      "candidate_mae": 1329332.9719090313,
      "control_mae": 1294116.4134795247,
      "delta": 35216.55842950661,
      "relative_change_pct": 2.72128210898886,
      "rows": 2330
    },
    "屏東縣": {
      "candidate_mae": 1429849.3412723618,
      "control_mae": 1418492.199859719,
      "delta": 11357.141412642784,
      "relative_change_pct": 0.8006488448625901,
      "rows": 1240
    },
    "彰化縣": {
      "candidate_mae": 1443663.0425275755,
      "control_mae": 1437729.9372029828,
      "delta": 5933.105324592674,
      "relative_change_pct": 0.41267175225795005,
      "rows": 2607
    },
    "新北市": {
      "candidate_mae": 1945563.1411664584,
      "control_mae": 1931141.0431383061,
      "delta": 14422.09802815225,
      "relative_change_pct": 0.7468174362197197,
      "rows": 17191
    },
    "新竹市": {
      "candidate_mae": 4272278.880169355,
      "control_mae": 4206008.947003277,
      "delta": 66269.93316607829,
      "relative_change_pct": 1.575601336114482,
      "rows": 1586
    },
    "新竹縣": {
      "candidate_mae": 2803887.330644159,
      "control_mae": 2839386.2871954134,
      "delta": -35498.95655125426,
      "relative_change_pct": -1.2502334293625874,
      "rows": 5042
    },
    "桃園市": {
      "candidate_mae": 1820116.130659272,
      "control_mae": 1801809.0342872667,
      "delta": 18307.096372005297,
      "relative_change_pct": 1.0160397702327513,
      "rows": 20723
    },
    "澎湖縣": {
      "candidate_mae": 958601.9126741473,
      "control_mae": 905431.1381984788,
      "delta": 53170.77447566844,
      "relative_change_pct": 5.872426099842495,
      "rows": 40
    },
    "臺中市": {
      "candidate_mae": 2880017.48222976,
      "control_mae": 2813849.3975944812,
      "delta": 66168.08463527868,
      "relative_change_pct": 2.3515147858248846,
      "rows": 20837
    },
    "臺北市": {
      "candidate_mae": 6482058.133167693,
      "control_mae": 6619596.082977112,
      "delta": -137537.949809419,
      "relative_change_pct": -2.0777393074346366,
      "rows": 4712
    },
    "臺南市": {
      "candidate_mae": 1784268.567281702,
      "control_mae": 1770047.7116755582,
      "delta": 14220.855606143828,
      "relative_change_pct": 0.8034165131448415,
      "rows": 8886
    },
    "臺東縣": {
      "candidate_mae": 1951486.0352800745,
      "control_mae": 1883004.2703493305,
      "delta": 68481.76493074396,
      "relative_change_pct": 3.6368353491858723,
      "rows": 235
    },
    "花蓮縣": {
      "candidate_mae": 885320.9339579509,
      "control_mae": 829143.651670074,
      "delta": 56177.28228787694,
      "relative_change_pct": 6.775337684214767,
      "rows": 691
    },
    "苗栗縣": {
      "candidate_mae": 1680485.9482371109,
      "control_mae": 1653807.183102974,
      "delta": 26678.76513413689,
      "relative_change_pct": 1.613172648342267,
      "rows": 2607
    },
    "金門縣": {
      "candidate_mae": 1622272.2377798625,
      "control_mae": 1495237.6344433622,
      "delta": 127034.60333650024,
      "relative_change_pct": 8.495947427366078,
      "rows": 35
    },
    "雲林縣": {
      "candidate_mae": 1077167.795774916,
      "control_mae": 1077147.497864338,
      "delta": 20.297910577850416,
      "relative_change_pct": 0.0018844132876969137,
      "rows": 1873
    },
    "高雄市": {
      "candidate_mae": 1824200.3943211327,
      "control_mae": 1798279.317019422,
      "delta": 25921.07730171061,
      "relative_change_pct": 1.4414377708950012,
      "rows": 10138
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1786776.6463395536,
      "control_mae": 1762875.3535570733,
      "delta": 23901.292782480363,
      "relative_change_pct": 1.355812975333344,
      "rows": 33694
    },
    "Q2": {
      "candidate_mae": 2396843.900101213,
      "control_mae": 2362943.1568230344,
      "delta": 33900.74327817839,
      "relative_change_pct": 1.4346829791605216,
      "rows": 29169
    },
    "Q3": {
      "candidate_mae": 2565619.874883177,
      "control_mae": 2551611.570300658,
      "delta": 14008.304582519457,
      "relative_change_pct": 0.5489983172034626,
      "rows": 25213
    },
    "Q4": {
      "candidate_mae": 2771394.7803108175,
      "control_mae": 2720040.168653072,
      "delta": 51354.61165774567,
      "relative_change_pct": 1.888009311391008,
      "rows": 17435
    }
  },
  "months": {
    "2021-08": {
      "candidate_mae": 1584443.2307265429,
      "control_mae": 1563207.6855669126,
      "delta": 21235.545159630245,
      "relative_change_pct": 1.3584596183666386,
      "rows": 5654
    },
    "2021-09": {
      "candidate_mae": 1638647.7327437128,
      "control_mae": 1621468.4185501335,
      "delta": 17179.31419357937,
      "relative_change_pct": 1.0594911376035665,
      "rows": 12352
    },
    "2021-10": {
      "candidate_mae": 1972218.4229448938,
      "control_mae": 1937066.7553911388,
      "delta": 35151.66755375499,
      "relative_change_pct": 1.8146853976984934,
      "rows": 11842
    },
    "2021-11": {
      "candidate_mae": 2059768.4544783684,
      "control_mae": 2042288.2736402748,
      "delta": 17480.180838093627,
      "relative_change_pct": 0.8559115313792649,
      "rows": 12524
    },
    "2021-12": {
      "candidate_mae": 2348496.9649890014,
      "control_mae": 2308971.313995403,
      "delta": 39525.650993598625,
      "relative_change_pct": 1.7118294521036792,
      "rows": 11482
    },
    "2022-01": {
      "candidate_mae": 2814786.000828522,
      "control_mae": 2773309.212675835,
      "delta": 41476.788152686786,
      "relative_change_pct": 1.4955702726227125,
      "rows": 8017
    },
    "2022-02": {
      "candidate_mae": 2495253.3512842325,
      "control_mae": 2452771.3776768353,
      "delta": 42481.97360739717,
      "relative_change_pct": 1.7319989133122697,
      "rows": 6204
    },
    "2022-03": {
      "candidate_mae": 2496722.16363674,
      "control_mae": 2450727.282373386,
      "delta": 45994.88126335386,
      "relative_change_pct": 1.8767849688607747,
      "rows": 9626
    },
    "2022-04": {
      "candidate_mae": 2619493.402936866,
      "control_mae": 2671623.6542875506,
      "delta": -52130.25135068456,
      "relative_change_pct": -1.9512572913113497,
      "rows": 7535
    },
    "2022-05": {
      "candidate_mae": 2669833.5149195874,
      "control_mae": 2633628.192055837,
      "delta": 36205.322863750625,
      "relative_change_pct": 1.3747317473651581,
      "rows": 6424
    },
    "2022-06": {
      "candidate_mae": 2731454.5315831676,
      "control_mae": 2695679.6338523687,
      "delta": 35774.897730798926,
      "relative_change_pct": 1.3271197838770394,
      "rows": 5629
    },
    "2022-07": {
      "candidate_mae": 2863653.940886687,
      "control_mae": 2793109.3935268214,
      "delta": 70544.54735986562,
      "relative_change_pct": 2.5256636035579683,
      "rows": 6566
    },
    "2022-08": {
      "candidate_mae": 2814179.317566522,
      "control_mae": 2766911.3766995794,
      "delta": 47267.94086694252,
      "relative_change_pct": 1.7083286897076027,
      "rows": 1656
    }
  },
  "overall_mae": {
    "candidate_mae": 2304247.2722651274,
    "control_mae": 2275409.1418580045,
    "delta": 28838.130407122895,
    "relative_change_pct": 1.2673821985076001
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 1113168.1603371084,
      "control_mae": 1107505.6060768394,
      "delta": 5662.55426026904,
      "relative_change_pct": 0.5112889929584852,
      "rows": 41132
    },
    {
      "band": "10-20m",
      "candidate_mae": 2070370.5111065102,
      "control_mae": 2024569.5907328655,
      "delta": 45800.92037364468,
      "relative_change_pct": 2.2622546828368293,
      "rows": 47793
    },
    {
      "band": "20-50m",
      "candidate_mae": 4698360.794945574,
      "control_mae": 4605603.27388477,
      "delta": 92757.5210608039,
      "relative_change_pct": 2.014014571918698,
      "rows": 15033
    },
    {
      "band": "50-100m",
      "candidate_mae": 13819189.47571636,
      "control_mae": 13891916.012976805,
      "delta": -72726.53726044483,
      "relative_change_pct": -0.5235169662162444,
      "rows": 1382
    },
    {
      "band": ">=100m",
      "candidate_mae": 50636290.34976392,
      "control_mae": 54572263.048466735,
      "delta": -3935972.698702812,
      "relative_change_pct": -7.212405128237388,
      "rows": 171
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 3391363.2304859813,
      "control_mae": 2649701.530279867,
      "delta": 741661.7002061144,
      "group": "基隆市",
      "relative_change_pct": 27.990386529602024,
      "rows": 1577,
      "section": "cities"
    },
    {
      "candidate_mae": 885320.9339579509,
      "control_mae": 829143.651670074,
      "delta": 56177.28228787694,
      "group": "花蓮縣",
      "relative_change_pct": 6.775337684214767,
      "rows": 691,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 1590205.6146122566,
      "control_mae": 1567209.9156508578,
      "delta": 22995.698961398797,
      "relative_change_pct": 1.4673017782591522,
      "rows": 9387
    },
    "1": {
      "candidate_mae": 1735568.388249336,
      "control_mae": 1715851.8997759607,
      "delta": 19716.488473375328,
      "relative_change_pct": 1.149078686566697,
      "rows": 12463
    },
    "10": {
      "candidate_mae": 2752818.0164119564,
      "control_mae": 2710524.938303519,
      "delta": 42293.07810843736,
      "relative_change_pct": 1.5603279464718014,
      "rows": 5992
    },
    "11": {
      "candidate_mae": 2886049.49882177,
      "control_mae": 2817934.7094697705,
      "delta": 68114.7893519993,
      "relative_change_pct": 2.4171883444672124,
      "rows": 6217
    },
    "2": {
      "candidate_mae": 1996454.1873225432,
      "control_mae": 1967431.6495803462,
      "delta": 29022.537742197048,
      "relative_change_pct": 1.4751484631441993,
      "rows": 11844
    },
    "3": {
      "candidate_mae": 2101445.705370602,
      "control_mae": 2077612.941580479,
      "delta": 23832.763790123165,
      "relative_change_pct": 1.1471224169403342,
      "rows": 12872
    },
    "4": {
      "candidate_mae": 2542874.840296187,
      "control_mae": 2505314.56244677,
      "delta": 37560.27784941718,
      "relative_change_pct": 1.499224026093339,
      "rows": 9987
    },
    "5": {
      "candidate_mae": 2768310.5503144003,
      "control_mae": 2719663.807003374,
      "delta": 48646.74331102613,
      "relative_change_pct": 1.7887042944703855,
      "rows": 6310
    },
    "6": {
      "candidate_mae": 2552037.8233358557,
      "control_mae": 2504077.440253743,
      "delta": 47960.38308211276,
      "relative_change_pct": 1.9152915285740062,
      "rows": 8284
    },
    "7": {
      "candidate_mae": 2460807.29618566,
      "control_mae": 2500338.7490206924,
      "delta": -39531.45283503225,
      "relative_change_pct": -1.5810438825745325,
      "rows": 9188
    },
    "8": {
      "candidate_mae": 2697067.014818299,
      "control_mae": 2655677.968591626,
      "delta": 41389.046226673294,
      "relative_change_pct": 1.5585114880710844,
      "rows": 7187
    },
    "9": {
      "candidate_mae": 2670240.001776139,
      "control_mae": 2628697.2045045653,
      "delta": 41542.79727157345,
      "relative_change_pct": 1.580356885547154,
      "rows": 5780
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 2485641.463117829,
      "control_mae": 2456616.5723295277,
      "delta": 29024.89078830136,
      "relative_change_pct": 1.1814986154220242,
      "rows": 90424
    },
    "seen": {
      "candidate_mae": 1217060.4018691112,
      "control_mae": 1189341.620617399,
      "delta": 27718.781251712237,
      "relative_change_pct": 2.3305987759280753,
      "rows": 15087
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 2021117.601185423,
    "control_mae": 1992570.9762837698,
    "delta": 28546.624901653267,
    "relative_change_pct": 1.4326528510865868
  }
}
```

完整城市／月份／新舊案／季度／價格帶／尾部：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 466689.4190578535,
      "rows": 1
    },
    "南投縣": {
      "mae": 1104478.5472281838,
      "rows": 869
    },
    "嘉義市": {
      "mae": 2096563.641256515,
      "rows": 731
    },
    "嘉義縣": {
      "mae": 1348566.80541551,
      "rows": 1560
    },
    "基隆市": {
      "mae": 3391363.2304859813,
      "rows": 1577
    },
    "宜蘭縣": {
      "mae": 1329332.9719090313,
      "rows": 2330
    },
    "屏東縣": {
      "mae": 1429849.3412723618,
      "rows": 1240
    },
    "彰化縣": {
      "mae": 1443663.0425275755,
      "rows": 2607
    },
    "新北市": {
      "mae": 1945563.1411664584,
      "rows": 17191
    },
    "新竹市": {
      "mae": 4272278.880169355,
      "rows": 1586
    },
    "新竹縣": {
      "mae": 2803887.330644159,
      "rows": 5042
    },
    "桃園市": {
      "mae": 1820116.130659272,
      "rows": 20723
    },
    "澎湖縣": {
      "mae": 958601.9126741473,
      "rows": 40
    },
    "臺中市": {
      "mae": 2880017.48222976,
      "rows": 20837
    },
    "臺北市": {
      "mae": 6482058.133167693,
      "rows": 4712
    },
    "臺南市": {
      "mae": 1784268.567281702,
      "rows": 8886
    },
    "臺東縣": {
      "mae": 1951486.0352800745,
      "rows": 235
    },
    "花蓮縣": {
      "mae": 885320.9339579509,
      "rows": 691
    },
    "苗栗縣": {
      "mae": 1680485.9482371109,
      "rows": 2607
    },
    "金門縣": {
      "mae": 1622272.2377798625,
      "rows": 35
    },
    "雲林縣": {
      "mae": 1077167.795774916,
      "rows": 1873
    },
    "高雄市": {
      "mae": 1824200.3943211327,
      "rows": 10138
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1786776.6463395536,
      "rows": 33694
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 2396843.900101213,
      "rows": 29169
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 2565619.874883177,
      "rows": 25213
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 2771394.7803108175,
      "rows": 17435
    }
  },
  "months": {
    "2021-08": {
      "mae": 1584443.2307265429,
      "rows": 5654
    },
    "2021-09": {
      "mae": 1638647.7327437128,
      "rows": 12352
    },
    "2021-10": {
      "mae": 1972218.4229448938,
      "rows": 11842
    },
    "2021-11": {
      "mae": 2059768.4544783684,
      "rows": 12524
    },
    "2021-12": {
      "mae": 2348496.9649890014,
      "rows": 11482
    },
    "2022-01": {
      "mae": 2814786.000828522,
      "rows": 8017
    },
    "2022-02": {
      "mae": 2495253.3512842325,
      "rows": 6204
    },
    "2022-03": {
      "mae": 2496722.16363674,
      "rows": 9626
    },
    "2022-04": {
      "mae": 2619493.402936866,
      "rows": 7535
    },
    "2022-05": {
      "mae": 2669833.5149195874,
      "rows": 6424
    },
    "2022-06": {
      "mae": 2731454.5315831676,
      "rows": 5629
    },
    "2022-07": {
      "mae": 2863653.940886687,
      "rows": 6566
    },
    "2022-08": {
      "mae": 2814179.317566522,
      "rows": 1656
    }
  },
  "overall_mae": 2304247.2722651274,
  "price_bands": [
    {
      "absolute_error_share": 0.18832751754213342,
      "absolute_error_sum": 45786832770.98595,
      "band": "<10m",
      "mae": 1113168.1603371084,
      "mean_signed_error_prediction_minus_truth": 44286.333448142446,
      "rows": 41132,
      "underprediction_fraction": 0.5401390644753477
    },
    {
      "absolute_error_share": 0.4069916923767985,
      "absolute_error_sum": 98949217837.31345,
      "band": "10-20m",
      "mae": 2070370.5111065102,
      "mean_signed_error_prediction_minus_truth": -1386604.5487003634,
      "rows": 47793,
      "underprediction_fraction": 0.7948444332852092
    },
    {
      "absolute_error_share": 0.29051275183409697,
      "absolute_error_sum": 70630457830.41682,
      "band": "20-50m",
      "mae": 4698360.794945574,
      "mean_signed_error_prediction_minus_truth": -3657939.4972114908,
      "rows": 15033,
      "underprediction_fraction": 0.8222576997272667
    },
    {
      "absolute_error_share": 0.07855318405810964,
      "absolute_error_sum": 19098119855.44001,
      "band": "50-100m",
      "mae": 13819189.47571636,
      "mean_signed_error_prediction_minus_truth": -9120243.88573878,
      "rows": 1382,
      "underprediction_fraction": 0.7691751085383502
    },
    {
      "absolute_error_share": 0.03561485418886145,
      "absolute_error_sum": 8658805649.809631,
      "band": ">=100m",
      "mae": 50636290.34976392,
      "mean_signed_error_prediction_minus_truth": -39198465.71502465,
      "rows": 171,
      "underprediction_fraction": 0.8888888888888888
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 1590205.6146122566,
      "rows": 9387
    },
    "1": {
      "mae": 1735568.388249336,
      "rows": 12463
    },
    "10": {
      "mae": 2752818.0164119564,
      "rows": 5992
    },
    "11": {
      "mae": 2886049.49882177,
      "rows": 6217
    },
    "2": {
      "mae": 1996454.1873225432,
      "rows": 11844
    },
    "3": {
      "mae": 2101445.705370602,
      "rows": 12872
    },
    "4": {
      "mae": 2542874.840296187,
      "rows": 9987
    },
    "5": {
      "mae": 2768310.5503144003,
      "rows": 6310
    },
    "6": {
      "mae": 2552037.8233358557,
      "rows": 8284
    },
    "7": {
      "mae": 2460807.29618566,
      "rows": 9188
    },
    "8": {
      "mae": 2697067.014818299,
      "rows": 7187
    },
    "9": {
      "mae": 2670240.001776139,
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
      "mae": 2485641.463117829,
      "rows": 90424
    },
    "seen": {
      "mae": 1217060.4018691112,
      "rows": 15087
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 1466931.4080328755,
      "0.9": 4831029.621409245,
      "0.95": 7123052.303711524,
      "0.99": 14611572.206531595
    },
    "absolute_error_sum": 243123433943.96585,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 351708033.7308078,
    "mean_signed_error_prediction_minus_truth": -1314984.4554367603,
    "top10_absolute_error_sum": 1619930932.7368574,
    "top10_error_rows": 10,
    "top10_error_share": 0.006662997912040897,
    "top1pct_absolute_error_sum": 26781622960.541527,
    "top1pct_error_rows": 1056,
    "top1pct_error_share": 0.11015648523093027,
    "underprediction_fraction": 0.6992730615765181
  },
  "weighted_mae": 2021117.601185423
}
```

## offset_2023_02_11完整診斷

prior／prepare實測15.171秒，native NN fit 40.344秒。

交易fold、初始化與fit價格因子變動：

```json
{
  "fold_policy": "transaction-permutation-5-fold",
  "fold_seed": 42,
  "folds": 5,
  "fold_rows": [
    54424,
    54424,
    54423,
    54423,
    54423
  ],
  "chronological_oof": false,
  "fit_factor_change_summary": {
    "changed_fraction": 0.9795823120201972,
    "changed_rows": 266561,
    "mean_absolute_change_million": 0.7431214101138455,
    "this_is_not_query_MAE": true
  },
  "original_initialization": {
    "initial_head_bias_float32": 0.3482004404067993,
    "initial_head_bias_formula_float64": 0.3482004386593577,
    "initial_training_median_ratio": 0.8823268937971015
  },
  "transaction_initialization": {
    "initial_head_bias_float32": 0.38292279839515686,
    "initial_head_bias_formula_float64": 0.38292279949831765,
    "initial_training_median_ratio": 0.9028264158890357
  },
  "initial_bias_changes_linked_to_fit_factor": true,
  "numeric_input_width": 144,
  "categorical_input_width": 23,
  "encoder_refit": false,
  "full_fit_state_and_query_prior_and_factor_exact_original": true,
  "query_prior_level_counts": {
    "address_city|address_district|transaction_type|building_type": 100663,
    "address_city|transaction_type|building_type": 5548,
    "global": 0,
    "transaction_type": 19,
    "transaction_type|building_type": 939
  },
  "legacy_state_group_rule_string_is_not_transaction_fold_evidence": true
}
```

|層|總價MAE|
|---|---:|
|base|1,325,946.46|
|native|1,521,471.68|
|reference|1,426,996.43|
|spatial|1,408,292.57|
|median|1,332,912.05|

季度／至少500列群退步超5%（重疊群不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-3,498.62|[-5,740.32, -1,167.84]|
|month|weighted|-3,435.06|[-5,791.79, -242.02]|
|project|overall|-3,498.62|[-9,094.68, +1,769.59]|
|project|weighted|-3,435.06|[-9,894.94, +2,595.50]|

完整條件抽樣與支援數：

```json
{
  "candidate": "R117",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Transaction OOF excludes each whole transaction fold, not all same-project peers; it is neither chronological OOF nor per-row as-of prediction. Prior factors and the linked initial ratio/head bias change together. Original group OOF may generalize better to new projects. Conditional single-seed bootstrap omits cumulative selection, retraining and future shift; not Private probability. R67/R100/R65/R102 contrary evidence and v24 comparisons remain; native improvement alone is insufficient.",
  "control": "v22",
  "difference": "R117 minus v22",
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
          "bootstrap_mean_delta": -3450.4293815226574,
          "candidate_mae": 1332912.0536606892,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 1.0,
          "percentile_2_5": -5740.321408735128,
          "percentile_97_5": -1167.8435822191598,
          "point_delta": -3498.6210036247503,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -3272.3387245730114,
          "candidate_mae": 1389455.2808739112,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.98,
          "percentile_2_5": -5791.7887124614845,
          "percentile_97_5": -242.01932531148037,
          "point_delta": -3435.0637128313538,
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
          "bootstrap_mean_delta": -3453.1058839839106,
          "candidate_mae": 1332912.0536606892,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.9015,
          "percentile_2_5": -9094.678997562005,
          "percentile_97_5": 1769.5944432792862,
          "point_delta": -3498.6210036247503,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -3368.961426379454,
          "candidate_mae": 1389455.2808739112,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.855,
          "percentile_2_5": -9894.936374674266,
          "percentile_97_5": 2595.497490362129,
          "point_delta": -3435.0637128313538,
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

v24次對照與反證（不代替v22門檻）：

```json
{
  "automatic_adoption": false,
  "both_point_metrics_improve": true,
  "candidate": "R117",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 748690.2509114165,
      "control_mae": 748690.2509114165,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1015235.0687331952,
      "control_mae": 1015289.1847550352,
      "delta": -54.11602184001822,
      "relative_change_pct": -0.005330109160285708,
      "rows": 899
    },
    "嘉義市": {
      "candidate_mae": 1168825.304045826,
      "control_mae": 1169659.4148716796,
      "delta": -834.1108258536551,
      "relative_change_pct": -0.07131228246858196,
      "rows": 604
    },
    "嘉義縣": {
      "candidate_mae": 705350.6090971753,
      "control_mae": 703921.1658472866,
      "delta": 1429.4432498887181,
      "relative_change_pct": 0.2030686558725855,
      "rows": 1532
    },
    "基隆市": {
      "candidate_mae": 817309.6284156322,
      "control_mae": 824405.9509523374,
      "delta": -7096.322536705178,
      "relative_change_pct": -0.8607801203409131,
      "rows": 842
    },
    "宜蘭縣": {
      "candidate_mae": 789938.0176927677,
      "control_mae": 785285.7426309958,
      "delta": 4652.275061771856,
      "relative_change_pct": 0.5924308578664659,
      "rows": 2576
    },
    "屏東縣": {
      "candidate_mae": 902713.1456897957,
      "control_mae": 920486.7394754328,
      "delta": -17773.59378563706,
      "relative_change_pct": -1.930890801943098,
      "rows": 691
    },
    "彰化縣": {
      "candidate_mae": 1016088.8793610954,
      "control_mae": 1011819.6970080789,
      "delta": 4269.182353016455,
      "relative_change_pct": 0.421931137102816,
      "rows": 2629
    },
    "新北市": {
      "candidate_mae": 1128223.2643581298,
      "control_mae": 1175179.3847952979,
      "delta": -46956.12043716805,
      "relative_change_pct": -3.9956555607335846,
      "rows": 20773
    },
    "新竹市": {
      "candidate_mae": 2390367.3138750237,
      "control_mae": 2294808.7567983773,
      "delta": 95558.55707664648,
      "relative_change_pct": 4.164118547724467,
      "rows": 1852
    },
    "新竹縣": {
      "candidate_mae": 2188343.7938903077,
      "control_mae": 2237487.2731571877,
      "delta": -49143.47926688008,
      "relative_change_pct": -2.1963691081709076,
      "rows": 4704
    },
    "桃園市": {
      "candidate_mae": 1066089.4195474435,
      "control_mae": 1066753.324847084,
      "delta": -663.9052996404935,
      "relative_change_pct": -0.062236065656103054,
      "rows": 21410
    },
    "澎湖縣": {
      "candidate_mae": 3899024.474707181,
      "control_mae": 3812375.149056576,
      "delta": 86649.32565060491,
      "relative_change_pct": 2.2728436279952002,
      "rows": 20
    },
    "臺中市": {
      "candidate_mae": 1617345.6920391656,
      "control_mae": 1612776.5130701174,
      "delta": 4569.178969048196,
      "relative_change_pct": 0.2833113535582314,
      "rows": 20055
    },
    "臺北市": {
      "candidate_mae": 4108264.247297786,
      "control_mae": 4030540.187754385,
      "delta": 77724.059543401,
      "relative_change_pct": 1.928378230281459,
      "rows": 3514
    },
    "臺南市": {
      "candidate_mae": 1268172.5145602003,
      "control_mae": 1290312.176866015,
      "delta": -22139.66230581468,
      "relative_change_pct": -1.7158376633776158,
      "rows": 8483
    },
    "臺東縣": {
      "candidate_mae": 721183.4504311245,
      "control_mae": 694696.0897249224,
      "delta": 26487.360706202104,
      "relative_change_pct": 3.81279830100818,
      "rows": 213
    },
    "花蓮縣": {
      "candidate_mae": 650572.0282504201,
      "control_mae": 651104.8750916787,
      "delta": -532.8468412585789,
      "relative_change_pct": -0.08183732938315837,
      "rows": 578
    },
    "苗栗縣": {
      "candidate_mae": 1061305.9756300526,
      "control_mae": 1025581.2731701825,
      "delta": 35724.70245987014,
      "relative_change_pct": 3.483361425803069,
      "rows": 2973
    },
    "金門縣": {
      "candidate_mae": 835058.4820508858,
      "control_mae": 893641.4111528273,
      "delta": -58582.929101941525,
      "relative_change_pct": -6.555529809923153,
      "rows": 65
    },
    "雲林縣": {
      "candidate_mae": 993284.7850658573,
      "control_mae": 996471.2687277427,
      "delta": -3186.4836618853733,
      "relative_change_pct": -0.3197767724857493,
      "rows": 1719
    },
    "高雄市": {
      "candidate_mae": 905345.1316515949,
      "control_mae": 908831.0314082025,
      "delta": -3485.8997566075996,
      "relative_change_pct": -0.38355861938454255,
      "rows": 11036
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 992541.0533059317,
      "control_mae": 993826.2791985838,
      "delta": -1285.225892652059,
      "relative_change_pct": -0.129320980895017,
      "rows": 22487
    },
    "Q2": {
      "candidate_mae": 1224410.2669405977,
      "control_mae": 1236271.5483622926,
      "delta": -11861.281421694905,
      "relative_change_pct": -0.9594398121842828,
      "rows": 27820
    },
    "Q3": {
      "candidate_mae": 1514318.4152688833,
      "control_mae": 1519816.2249750085,
      "delta": -5497.809706125176,
      "relative_change_pct": -0.3617417432305396,
      "rows": 27006
    },
    "Q4": {
      "candidate_mae": 1526286.3231521947,
      "control_mae": 1535913.1617296129,
      "delta": -9626.838577418122,
      "relative_change_pct": -0.6267827385877212,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "candidate_mae": 595835.1599566208,
      "control_mae": 596296.3427625896,
      "delta": -461.18280596879777,
      "relative_change_pct": -0.07734120988100943,
      "rows": 2896
    },
    "2023-03": {
      "candidate_mae": 919293.7517420347,
      "control_mae": 925522.7748700764,
      "delta": -6229.023128041765,
      "relative_change_pct": -0.6730275361312622,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1095103.169024976,
      "control_mae": 1094744.1862448624,
      "delta": 358.98278011358343,
      "relative_change_pct": 0.032791476275836506,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1212067.48077838,
      "control_mae": 1213455.5062566611,
      "delta": -1388.0254782810807,
      "relative_change_pct": -0.11438618648350307,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1259844.9223424182,
      "control_mae": 1276086.4061312978,
      "delta": -16241.483788879588,
      "relative_change_pct": -1.272757370570131,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1171754.0444758346,
      "control_mae": 1183156.183768008,
      "delta": -11402.13929217332,
      "relative_change_pct": -0.9637053373512217,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1299618.3941784063,
      "control_mae": 1306980.5584263823,
      "delta": -7362.1642479759175,
      "relative_change_pct": -0.5632956206204044,
      "rows": 7400
    },
    "2023-09": {
      "candidate_mae": 1361114.0515670378,
      "control_mae": 1365893.9517492831,
      "delta": -4779.900182245299,
      "relative_change_pct": -0.3499466540666456,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1594748.567095871,
      "control_mae": 1596675.0554627297,
      "delta": -1926.488366858568,
      "relative_change_pct": -0.12065625753138955,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1688853.6038028558,
      "control_mae": 1708449.945458693,
      "delta": -19596.34165583714,
      "relative_change_pct": -1.147024629426636,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1494174.149468435,
      "control_mae": 1506056.8881887118,
      "delta": -11882.738720276859,
      "relative_change_pct": -0.7889966716043417,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1523267.0001402746,
      "control_mae": 1526897.035515871,
      "delta": -3630.035375596257,
      "relative_change_pct": -0.23773936887432812,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1447628.236237483,
      "control_mae": 1437162.9778553827,
      "delta": 10465.25838210038,
      "relative_change_pct": 0.728188698383898,
      "rows": 2566
    }
  },
  "overall_mae": {
    "candidate_mae": 1332912.053660679,
    "control_mae": 1340328.1391452,
    "delta": -7416.085484520998,
    "relative_change_pct": -0.5533037222698792
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 706670.9752454957,
      "control_mae": 710632.9248020112,
      "delta": -3961.949556515552,
      "relative_change_pct": -0.5575240631609332,
      "rows": 25432
    },
    {
      "band": "10-20m",
      "candidate_mae": 1058178.0018754508,
      "control_mae": 1067452.758311166,
      "delta": -9274.756435715128,
      "relative_change_pct": -0.8688680940212162,
      "rows": 59481
    },
    {
      "band": "20-50m",
      "candidate_mae": 2243621.6448599743,
      "control_mae": 2241526.0750715095,
      "delta": 2095.5697884648107,
      "relative_change_pct": 0.09348853050473471,
      "rows": 20803
    },
    {
      "band": "50-100m",
      "candidate_mae": 6796760.500240136,
      "control_mae": 6794509.1460085185,
      "delta": 2251.3542316174135,
      "relative_change_pct": 0.033134906190243146,
      "rows": 1183
    },
    {
      "band": ">=100m",
      "candidate_mae": 26735856.1368816,
      "control_mae": 27434374.16274089,
      "delta": -698518.0258592889,
      "relative_change_pct": -2.546141645935407,
      "rows": 270
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 649740.9469612412,
      "control_mae": 645679.7496816863,
      "delta": 4061.197279554908,
      "relative_change_pct": 0.6289801223527667,
      "rows": 5055
    },
    "1": {
      "candidate_mae": 1011645.6023127495,
      "control_mae": 1023592.2057761417,
      "delta": -11946.603463392239,
      "relative_change_pct": -1.1671252864155695,
      "rows": 7516
    },
    "10": {
      "candidate_mae": 1489268.111677688,
      "control_mae": 1497637.3318924468,
      "delta": -8369.220214758767,
      "relative_change_pct": -0.5588282314105538,
      "rows": 9935
    },
    "11": {
      "candidate_mae": 1514133.7590139608,
      "control_mae": 1514286.76981743,
      "delta": -153.0108034692239,
      "relative_change_pct": -0.01010447997823237,
      "rows": 10413
    },
    "2": {
      "candidate_mae": 1155611.6195456786,
      "control_mae": 1151164.1928843837,
      "delta": 4447.426661294885,
      "relative_change_pct": 0.3863416434237161,
      "rows": 9057
    },
    "3": {
      "candidate_mae": 1180728.4847690961,
      "control_mae": 1198704.169185322,
      "delta": -17975.68441622588,
      "relative_change_pct": -1.4995930504223351,
      "rows": 9401
    },
    "4": {
      "candidate_mae": 1263378.708594006,
      "control_mae": 1271555.585854942,
      "delta": -8176.8772609359585,
      "relative_change_pct": -0.6430609366902479,
      "rows": 11490
    },
    "5": {
      "candidate_mae": 1197984.4920563244,
      "control_mae": 1206414.4495510287,
      "delta": -8429.957494704286,
      "relative_change_pct": -0.6987613168792469,
      "rows": 6914
    },
    "6": {
      "candidate_mae": 1338331.7873124317,
      "control_mae": 1342783.9583831823,
      "delta": -4452.17107075057,
      "relative_change_pct": -0.3315627240670446,
      "rows": 7252
    },
    "7": {
      "candidate_mae": 1404469.0936185643,
      "control_mae": 1412213.0920092391,
      "delta": -7743.998390674824,
      "relative_change_pct": -0.5483590567523332,
      "rows": 9394
    },
    "8": {
      "candidate_mae": 1704416.8789505681,
      "control_mae": 1708489.12857141,
      "delta": -4072.249620841816,
      "relative_change_pct": -0.2383538503547234,
      "rows": 9933
    },
    "9": {
      "candidate_mae": 1591424.0738032027,
      "control_mae": 1611253.9941320717,
      "delta": -19829.92032886902,
      "relative_change_pct": -1.2307134940292719,
      "rows": 10809
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1648241.633408694,
      "control_mae": 1658897.3265622966,
      "delta": -10655.693153602537,
      "relative_change_pct": -0.6423359048799084,
      "rows": 59352
    },
    "seen": {
      "candidate_mae": 941514.8263732248,
      "control_mae": 944909.8065107913,
      "delta": -3394.980137566454,
      "relative_change_pct": -0.35929144921279654,
      "rows": 47817
    }
  },
  "used_for_release_gate": false,
  "weighted_mae": {
    "candidate_mae": 1389455.2808738996,
    "control_mae": 1397452.2756174281,
    "delta": -7996.994743528543,
    "relative_change_pct": -0.5722553022424525
  }
}
```

完整城市／月份／新舊案／季度／價格帶／尾部：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 748690.2509114165,
      "rows": 1
    },
    "南投縣": {
      "mae": 1015235.0687331952,
      "rows": 899
    },
    "嘉義市": {
      "mae": 1168825.304045826,
      "rows": 604
    },
    "嘉義縣": {
      "mae": 705350.6090971753,
      "rows": 1532
    },
    "基隆市": {
      "mae": 817309.6284156322,
      "rows": 842
    },
    "宜蘭縣": {
      "mae": 789938.0176927677,
      "rows": 2576
    },
    "屏東縣": {
      "mae": 902713.1456897957,
      "rows": 691
    },
    "彰化縣": {
      "mae": 1016088.8793610954,
      "rows": 2629
    },
    "新北市": {
      "mae": 1128223.2643581298,
      "rows": 20773
    },
    "新竹市": {
      "mae": 2390367.3138750237,
      "rows": 1852
    },
    "新竹縣": {
      "mae": 2188343.7938903077,
      "rows": 4704
    },
    "桃園市": {
      "mae": 1066089.4195474435,
      "rows": 21410
    },
    "澎湖縣": {
      "mae": 3899024.474707181,
      "rows": 20
    },
    "臺中市": {
      "mae": 1617345.6920391656,
      "rows": 20055
    },
    "臺北市": {
      "mae": 4108264.247297786,
      "rows": 3514
    },
    "臺南市": {
      "mae": 1268172.5145602003,
      "rows": 8483
    },
    "臺東縣": {
      "mae": 721183.4504311245,
      "rows": 213
    },
    "花蓮縣": {
      "mae": 650572.0282504201,
      "rows": 578
    },
    "苗栗縣": {
      "mae": 1061305.9756300526,
      "rows": 2973
    },
    "金門縣": {
      "mae": 835058.4820508858,
      "rows": 65
    },
    "雲林縣": {
      "mae": 993284.7850658573,
      "rows": 1719
    },
    "高雄市": {
      "mae": 905345.1316515949,
      "rows": 11036
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 992541.0533059317,
      "rows": 22487
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1224410.2669405977,
      "rows": 27820
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1514318.4152688833,
      "rows": 27006
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1526286.3231521947,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "mae": 595835.1599566208,
      "rows": 2896
    },
    "2023-03": {
      "mae": 919293.7517420347,
      "rows": 7500
    },
    "2023-04": {
      "mae": 1095103.169024976,
      "rows": 8639
    },
    "2023-05": {
      "mae": 1212067.48077838,
      "rows": 8750
    },
    "2023-06": {
      "mae": 1259844.9223424182,
      "rows": 13180
    },
    "2023-07": {
      "mae": 1171754.0444758346,
      "rows": 6144
    },
    "2023-08": {
      "mae": 1299618.3941784063,
      "rows": 7400
    },
    "2023-09": {
      "mae": 1361114.0515670378,
      "rows": 8602
    },
    "2023-10": {
      "mae": 1594748.567095871,
      "rows": 9855
    },
    "2023-11": {
      "mae": 1688853.6038028558,
      "rows": 10336
    },
    "2023-12": {
      "mae": 1494174.149468435,
      "rows": 10416
    },
    "2024-01": {
      "mae": 1523267.0001402746,
      "rows": 10885
    },
    "2024-02": {
      "mae": 1447628.236237483,
      "rows": 2566
    }
  },
  "overall_mae": 1332912.053660679,
  "price_bands": [
    {
      "absolute_error_share": 0.12581345690205975,
      "absolute_error_sum": 17972056242.443447,
      "band": "<10m",
      "mae": 706670.9752454957,
      "mean_signed_error_prediction_minus_truth": 405329.9456851303,
      "rows": 25432,
      "underprediction_fraction": 0.2941962881409248
    },
    {
      "absolute_error_share": 0.4406221411373781,
      "absolute_error_sum": 62941485729.553696,
      "band": "10-20m",
      "mae": 1058178.0018754508,
      "mean_signed_error_prediction_minus_truth": 297570.79130750493,
      "rows": 59481,
      "underprediction_fraction": 0.36482238025588004
    },
    {
      "absolute_error_share": 0.3267419650076421,
      "absolute_error_sum": 46674061078.02205,
      "band": "20-50m",
      "mae": 2243621.6448599743,
      "mean_signed_error_prediction_minus_truth": -160983.6739875819,
      "rows": 20803,
      "underprediction_fraction": 0.4607989232322261
    },
    {
      "absolute_error_share": 0.05628802851468066,
      "absolute_error_sum": 8040567671.7840805,
      "band": "50-100m",
      "mae": 6796760.500240136,
      "mean_signed_error_prediction_minus_truth": -986236.0998630358,
      "rows": 1183,
      "underprediction_fraction": 0.4513947590870668
    },
    {
      "absolute_error_share": 0.05053440843823957,
      "absolute_error_sum": 7218681156.958033,
      "band": ">=100m",
      "mae": 26735856.1368816,
      "mean_signed_error_prediction_minus_truth": -11033047.087791305,
      "rows": 270,
      "underprediction_fraction": 0.7111111111111111
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 649740.9469612412,
      "rows": 5055
    },
    "1": {
      "mae": 1011645.6023127495,
      "rows": 7516
    },
    "10": {
      "mae": 1489268.111677688,
      "rows": 9935
    },
    "11": {
      "mae": 1514133.7590139608,
      "rows": 10413
    },
    "2": {
      "mae": 1155611.6195456786,
      "rows": 9057
    },
    "3": {
      "mae": 1180728.4847690961,
      "rows": 9401
    },
    "4": {
      "mae": 1263378.708594006,
      "rows": 11490
    },
    "5": {
      "mae": 1197984.4920563244,
      "rows": 6914
    },
    "6": {
      "mae": 1338331.7873124317,
      "rows": 7252
    },
    "7": {
      "mae": 1404469.0936185643,
      "rows": 9394
    },
    "8": {
      "mae": 1704416.8789505681,
      "rows": 9933
    },
    "9": {
      "mae": 1591424.0738032027,
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
      "mae": 1648241.633408694,
      "rows": 59352
    },
    "seen": {
      "mae": 941514.8263732248,
      "rows": 47817
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 786039.7213277854,
      "0.9": 2914204.4010223453,
      "0.95": 4196249.268591929,
      "0.99": 8655040.03782656
    },
    "absolute_error_sum": 142846851878.7613,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 594824419.7368984,
    "mean_signed_error_prediction_minus_truth": 191413.33806063927,
    "top10_absolute_error_sum": 2163450392.5596466,
    "top10_error_rows": 10,
    "top10_error_share": 0.015145243763550606,
    "top1pct_absolute_error_sum": 17729799806.22689,
    "top1pct_error_rows": 1072,
    "top1pct_error_share": 0.12411753968001157,
    "underprediction_fraction": 0.36852074760425124
  },
  "weighted_mae": 1389455.2808738996
}
```

## 工程失敗、實際成本與核驗範圍

原R117 train命令84.125秒exit1；development先驗及NN已完成，但動態R52父來源未列入AST來源閉包，metadata核驗因缺鍵停止。沒有stage seal、其餘三fit或query分數。原目錄全部保留，不把已花成本當0或刪除。另固定工程恢復規約及新source/目錄，補明列動態來源與GPU前守衛；四prior／NN全部重新訓練，不沿用原partial模型。科學假說、切分及門檻不改。

新v2四fit命令416.140秒，另evaluate 190.547秒。本研究合計五次實際NN fit＝原失敗批一個＋新完整批四個，prior也五次。以上程序時間不等於專案工時或下一次ETA。

[獨立postfit核驗](../outputs/research_round117_v2_20261008/postfit_review.json)說明真正重跑的檢查與限制；保存artifact核驗不是乾淨環境端到端重訓。

```json
[
  "All four origins are reused and overlapping, not fresh blind validation or four independent votes.",
  "This numerical postfit reviewer authored only the engineering-v2 driver; final source preflight/minimal-diff review is performed by root/another agent and bound separately here.",
  "CPU checkpoint constructors/tensors/config and saved ratio-to-TWD inverse were inspected without independent full-query NN inference.",
  "Full actual transaction OOF priors were replayed using frozen R60 estimator/key functions; independent full complement date/global median arithmetic and a 27-case artificial oracle supplement this shared-estimator replay.",
  "No fresh neural network retraining, GPU inference or separately independent full-query NN forward pass was performed; saved ratio/TWD reload exactness is original frozen trainer evidence.",
  "Composition and descriptive definitions reuse frozen helpers; independent MAE/group/gate arithmetic supplements replay.",
  "Bootstrap summaries use saved replicates without resampling, selection/refit/future-shift correction; not Private probability.",
  "Original R117 aborted batch retained one completed native before metadata failure; this engineering recovery trained four fresh priors/NNs, and all five historical native fits are counted.",
  "No fullfit/release/CSV/Kaggle upload or v22 replacement is authorized by audit pass."
]
```

Four reused overlapping origins are not independent votes or fresh blind validation. Transaction OOF excludes each whole transaction fold, not all same-project peers; it is neither chronological OOF nor per-row as-of prediction. Prior factors and the linked initial ratio/head bias change together. Original group OOF may generalize better to new projects. Conditional single-seed bootstrap omits cumulative selection, retraining and future shift; not Private probability. R67/R100/R65/R102 contrary evidence and v24 comparisons remain; native improvement alone is insufficient.

四個窗口已被反覆使用且重疊，有累積選型偏差；單一seed未估計重訓變異。條件bootstrap不含重訓或選型不確定性，不能稱Private勝率。沒有外部成交答案、Public搜參數、test ID人工調價或Kaggle操作。

- [原科學規約](round117_plan.md)
- [工程恢復規約](round117_engineering_recovery_v2.md)
- [先驗方法](round117_method_review.md)
- [四起點driver說明](round117_driver_review.md)
- [交易OOF先驗](../src/round117_transaction_prior.py)
- [原倍率NN trainer](../src/round117_transaction_prior_train.py)
- [v2研究流程](../src/round117_research_v2.py)
- [完整機器結果](../outputs/round117_v2/results.json)
