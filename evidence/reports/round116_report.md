# R116：單價CatBoost固定Depthwise比較

2026-10-08。combined探索門檻失敗，不全量fit、不產v31 CSV。
combined強推薦門檻為False。v22選定與上台材料保持，Public／Private沒有新增回報。

假說是讓不同分支可以在同一深度使用不同切分，可能改善房價非線性。原SymmetricTree每層共用切分；本輪僅新增grow_policy=Depthwise。保持Plain、原日期乘面積的單價MAE、95欄／23類別、depth8、2999樹、learning_rate .065、seed42及其他原參數。不搜尋leaf最小列數或事後改深度，不混Ordered或新NN。樹形與容量表現會連動，不能將差異全歸因於某個單一節點。

|起點|fit／query列|v22整體MAE|R116整體MAE|差vs v22|固定加權差|
|---|---:|---:|---:|---:|---:|
|development|240,316／81,234|1,407,387.07|1,398,534.63|-8,852.44|-8,839.06|
|confirmation|321,550／133,697|1,383,228.41|1,403,406.07|+20,177.67|+21,530.69|
|auxiliary|134,805／105,511|2,309,352.36|2,297,125.92|-12,226.44|-10,354.04|
|offset_2023_02_11|272,117／107,169|1,336,410.67|1,349,012.83|+12,602.16|+14,975.31|

單位新臺幣元，負值較好。這是離線MAE差，不是預測向量差或Kaggle分數。

## 原定判定與完整對照

```json
{
  "august": {
    "automatic_selection": false,
    "exploratory_submission": {
      "components": {
        "at_least_two_windows_overall_improve": true,
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
      "confirmation_fractional_improvement": -0.01458737070610251,
      "selected_submission": "v22"
    },
    "selected_submission": "v22"
  },
  "automatic_selection": false,
  "combined_exploratory_submission": {
    "components": {
      "fixed_extra_origin": false,
      "original_three_windows": false
    },
    "passed": false
  },
  "combined_promotion": {
    "components": {
      "fixed_extra_origin": false,
      "original_three_windows": false
    },
    "passed": false
  },
  "extra_origin": {
    "components": {
      "no_quarter_or_large_group_above_5pct": false,
      "overall_regression_at_most_0_25pct": false,
      "weighted_regression_at_most_0_25pct": false
    },
    "passed": false
  },
  "full_data_fit": false,
  "new_CSV": false,
  "private_score": null,
  "reserved_exploratory_version": 31,
  "selected_submission": "v22"
}
```

只替換原單價CatBoost槽，再沿.75新CatBoost+.25原LGB、原reference、原spatial各一次，最後與原N/H分支逐筆median。原v22/v24及其餘native由各時間起點自己的保存狀態重播，不是同seed重新訓練的matched control。全部四fit及五層封存後才讀query價格評分。

**診斷語意：** raw_unit_diagnostic_only.query_y_used_for_diagnostics=false只指封存時負單價／零投影及範圍統計；後加入的unprojected_TWD_overall_mae確實在封存後用query y計算。未投影單價先乘一次原價格因子才計總價MAE，不是直接拿單價減總價。原source/results不回寫；獨立語意補充另存於核驗目錄。

## development完整診斷

native fit實測39.297秒。

|層|總價MAE|
|---|---:|
|base|1,705,213.44|
|native|1,749,999.97|
|reference|1,518,854.19|
|spatial|1,502,297.63|
|median|1,398,534.63|

所有季度／至少500列群退步超5%（群重疊不能相加）：

```json
[
  {
    "candidate_mae": 1313060.020657954,
    "control_mae": 1229903.2131515937,
    "delta": 83156.80750636035,
    "group": "2022-12",
    "relative_change_pct": 6.761248089861744,
    "rows": 6122,
    "section": "months"
  },
  {
    "candidate_mae": 1307685.1214580873,
    "control_mae": 1220773.8640789841,
    "delta": 86911.25737910322,
    "group": "4",
    "relative_change_pct": 7.119357641611507,
    "rows": 5881,
    "section": "relative_forecast_months"
  }
]
```

原生CBM及葉數證據：

```json
{
  "actual_grow_policy": "Depthwise",
  "actual_boosting_type": "Plain",
  "actual_model_depth": 8,
  "actual_trees": 2999,
  "actual_min_data_in_leaf": 1,
  "tree_leaf_counts_count": 2999,
  "minimum_tree_leaf_count": 10,
  "maximum_tree_leaf_count": 253,
  "tree_leaf_counts_distribution": {
    "10": 36,
    "100": 3,
    "101": 1,
    "102": 1,
    "103": 2,
    "104": 2,
    "105": 1,
    "106": 1,
    "11": 135,
    "110": 2,
    "111": 1,
    "12": 33,
    "120": 1,
    "121": 1,
    "127": 1,
    "13": 40,
    "130": 1,
    "131": 1,
    "134": 1,
    "139": 1,
    "14": 66,
    "15": 78,
    "159": 1,
    "16": 84,
    "161": 1,
    "17": 206,
    "174": 1,
    "177": 1,
    "178": 1,
    "179": 1,
    "18": 161,
    "182": 1,
    "184": 1,
    "185": 1,
    "19": 142,
    "191": 1,
    "194": 3,
    "195": 1,
    "196": 1,
    "20": 161,
    "200": 2,
    "202": 1,
    "203": 1,
    "204": 2,
    "21": 171,
    "213": 1,
    "217": 2,
    "218": 1,
    "22": 162,
    "223": 1,
    "224": 1,
    "225": 1,
    "226": 1,
    "227": 1,
    "228": 1,
    "229": 1,
    "23": 87,
    "231": 1,
    "232": 3,
    "233": 1,
    "235": 2,
    "236": 1,
    "237": 1,
    "238": 2,
    "239": 1,
    "24": 143,
    "240": 3,
    "241": 1,
    "242": 2,
    "243": 2,
    "245": 2,
    "246": 1,
    "247": 2,
    "248": 1,
    "25": 119,
    "250": 1,
    "253": 1,
    "26": 76,
    "27": 80,
    "28": 162,
    "29": 81,
    "30": 53,
    "31": 60,
    "32": 55,
    "33": 49,
    "34": 50,
    "35": 21,
    "36": 25,
    "37": 21,
    "38": 26,
    "39": 25,
    "40": 23,
    "41": 12,
    "42": 15,
    "43": 15,
    "44": 19,
    "45": 14,
    "46": 21,
    "47": 18,
    "48": 15,
    "49": 16,
    "50": 10,
    "51": 9,
    "52": 13,
    "53": 6,
    "54": 6,
    "55": 8,
    "56": 10,
    "57": 3,
    "58": 6,
    "59": 3,
    "60": 3,
    "61": 8,
    "63": 2,
    "65": 3,
    "66": 1,
    "68": 4,
    "69": 4,
    "70": 4,
    "71": 2,
    "72": 4,
    "75": 1,
    "76": 1,
    "77": 1,
    "78": 2,
    "79": 5,
    "80": 5,
    "81": 2,
    "83": 2,
    "84": 4,
    "85": 2,
    "86": 1,
    "92": 1,
    "93": 1,
    "94": 1,
    "96": 2,
    "97": 1,
    "99": 1
  },
  "runtime_parameters_not_recovered_from_native_model": {
    "allow_writing_files": false,
    "thread_count": 4,
    "verbose": 300
  }
}
```

原單價逆轉換與未投影總價：

```json
{
  "extra_price_cap_applied": false,
  "inverse_applications_per_prediction": 1,
  "native_TWD_max": 3583018846.22417,
  "native_TWD_min": 49909.14752678778,
  "native_zero_rows": 0,
  "nonnegative_projection_after_TWD_reconstruction": true,
  "query_y_used_for_diagnostics": false,
  "raw_negative_unit_rows": 0,
  "raw_unit_max": 50.996918657267685,
  "raw_unit_min": 3.514458676779755,
  "raw_units": "unit",
  "unprojected_TWD_overall_mae": 1749999.9661696332
}
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-8,852.44|[-20,971.31, +6,033.31]|
|month|weighted|-8,839.06|[-23,619.92, +6,063.10]|
|project|overall|-8,852.44|[-32,423.94, +15,312.16]|
|project|weighted|-8,839.06|[-31,822.96, +15,108.90]|

完整條件抽樣及支援數：

```json
{
  "candidate": "R116",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R27 deeper trees, R84 extra trees, R92 shallow unit and R112 Ordered supplied contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R116 minus v22",
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
          "bootstrap_mean_delta": -8554.130710162488,
          "candidate_mae": 1398534.633916496,
          "control_mae": 1407387.0708058395,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.9145,
          "percentile_2_5": -20971.31244809786,
          "percentile_97_5": 6033.306431800711,
          "point_delta": -8852.43688934343,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -9068.971620803275,
          "candidate_mae": 1390878.6847519758,
          "control_mae": 1399717.744053713,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.9185,
          "percentile_2_5": -23619.915702401726,
          "percentile_97_5": 6063.101543006254,
          "point_delta": -8839.05930173723,
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
          "bootstrap_mean_delta": -9395.055393212739,
          "candidate_mae": 1398534.633916496,
          "control_mae": 1407387.0708058395,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.798,
          "percentile_2_5": -32423.944959165663,
          "percentile_97_5": 15312.162174040144,
          "point_delta": -8852.43688934343,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -9338.19965859827,
          "candidate_mae": 1390878.6847519758,
          "control_mae": 1399717.744053713,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.7995,
          "percentile_2_5": -31822.95679855087,
          "percentile_97_5": 15108.895320283005,
          "point_delta": -8839.05930173723,
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

v24次對照及反證（不以較弱對照替換v22）：

```json
{
  "both_point_metrics_improve": true,
  "candidate": "R116",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 279735.0033029721,
      "control_mae": 240685.11008695266,
      "delta": 39049.89321601947,
      "relative_change_pct": 16.224474044909492,
      "rows": 3
    },
    "南投縣": {
      "candidate_mae": 795148.7587618007,
      "control_mae": 815392.6297455619,
      "delta": -20243.870983761153,
      "relative_change_pct": -2.482714491799874,
      "rows": 659
    },
    "嘉義市": {
      "candidate_mae": 1487741.7298481574,
      "control_mae": 1432770.9123328016,
      "delta": 54970.81751535577,
      "relative_change_pct": 3.8366787769200075,
      "rows": 569
    },
    "嘉義縣": {
      "candidate_mae": 802425.0227985018,
      "control_mae": 798121.9796995061,
      "delta": 4303.043098995695,
      "relative_change_pct": 0.5391460463995486,
      "rows": 798
    },
    "基隆市": {
      "candidate_mae": 1335670.6145798536,
      "control_mae": 1573023.596581248,
      "delta": -237352.9820013945,
      "relative_change_pct": -15.088965131689617,
      "rows": 737
    },
    "宜蘭縣": {
      "candidate_mae": 833557.5667857141,
      "control_mae": 833869.7553878439,
      "delta": -312.18860212981235,
      "relative_change_pct": -0.03743853282993928,
      "rows": 2242
    },
    "屏東縣": {
      "candidate_mae": 792741.5474508124,
      "control_mae": 806099.9039080202,
      "delta": -13358.356457207818,
      "relative_change_pct": -1.65715891943985,
      "rows": 615
    },
    "彰化縣": {
      "candidate_mae": 1063738.7942077988,
      "control_mae": 1099316.8196964574,
      "delta": -35578.025488658575,
      "relative_change_pct": -3.236375979263408,
      "rows": 2246
    },
    "新北市": {
      "candidate_mae": 1145906.7904371824,
      "control_mae": 1175055.059966161,
      "delta": -29148.269528978504,
      "relative_change_pct": -2.48058755049469,
      "rows": 15979
    },
    "新竹市": {
      "candidate_mae": 4450860.708465353,
      "control_mae": 3988384.5966167687,
      "delta": 462476.1118485839,
      "relative_change_pct": 11.595574615369065,
      "rows": 1250
    },
    "新竹縣": {
      "candidate_mae": 2730845.3890422736,
      "control_mae": 2973954.3281137035,
      "delta": -243108.9390714299,
      "relative_change_pct": -8.174602305531273,
      "rows": 3639
    },
    "桃園市": {
      "candidate_mae": 986189.4194385294,
      "control_mae": 968876.870646954,
      "delta": 17312.548791575362,
      "relative_change_pct": 1.786867796732019,
      "rows": 15846
    },
    "澎湖縣": {
      "candidate_mae": 858929.3701140208,
      "control_mae": 875582.3630456572,
      "delta": -16652.9929316364,
      "relative_change_pct": -1.901933345677502,
      "rows": 28
    },
    "臺中市": {
      "candidate_mae": 1579950.4766610668,
      "control_mae": 1572125.9434656193,
      "delta": 7824.5331954474095,
      "relative_change_pct": 0.49770396754593876,
      "rows": 14719
    },
    "臺北市": {
      "candidate_mae": 4468936.3543950785,
      "control_mae": 4511801.4476898415,
      "delta": -42865.093294763006,
      "relative_change_pct": -0.9500660388482086,
      "rows": 2698
    },
    "臺南市": {
      "candidate_mae": 1221782.4282065793,
      "control_mae": 1239277.1706731338,
      "delta": -17494.74246655451,
      "relative_change_pct": -1.4116892395469494,
      "rows": 7530
    },
    "臺東縣": {
      "candidate_mae": 736286.8448863544,
      "control_mae": 642748.5280153912,
      "delta": 93538.3168709632,
      "relative_change_pct": 14.552863646344024,
      "rows": 265
    },
    "花蓮縣": {
      "candidate_mae": 560002.2152601547,
      "control_mae": 558290.847066315,
      "delta": 1711.3681938396767,
      "relative_change_pct": 0.30653703223552875,
      "rows": 535
    },
    "苗栗縣": {
      "candidate_mae": 1288980.35842659,
      "control_mae": 1309189.6028082767,
      "delta": -20209.244381686673,
      "relative_change_pct": -1.5436453465821025,
      "rows": 2402
    },
    "金門縣": {
      "candidate_mae": 672492.3941265355,
      "control_mae": 664978.0807066441,
      "delta": 7514.313419891405,
      "relative_change_pct": 1.130009189461743,
      "rows": 29
    },
    "雲林縣": {
      "candidate_mae": 935963.2250779648,
      "control_mae": 900864.7686889353,
      "delta": 35098.45638902951,
      "relative_change_pct": 3.8960849185066593,
      "rows": 1350
    },
    "高雄市": {
      "candidate_mae": 989113.9111499307,
      "control_mae": 970391.5011320283,
      "delta": 18722.41001790238,
      "relative_change_pct": 1.9293666521256012,
      "rows": 7095
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1220546.2395406482,
      "control_mae": 1257065.2649268839,
      "delta": -36519.02538623568,
      "relative_change_pct": -2.905101779927057,
      "rows": 17608
    },
    "Q2": {
      "candidate_mae": 1330257.1974723183,
      "control_mae": 1304412.7913579252,
      "delta": 25844.406114393147,
      "relative_change_pct": 1.9813057864518866,
      "rows": 14193
    },
    "Q3": {
      "candidate_mae": 1440132.0748636953,
      "control_mae": 1449548.9071540441,
      "delta": -9416.83229034883,
      "relative_change_pct": -0.6496388113483706,
      "rows": 22487
    },
    "Q4": {
      "candidate_mae": 1516091.2155628249,
      "control_mae": 1524529.5621492898,
      "delta": -8438.346586464904,
      "relative_change_pct": -0.5535049497215704,
      "rows": 26946
    }
  },
  "months": {
    "2022-08": {
      "candidate_mae": 941569.061774086,
      "control_mae": 950683.2454188134,
      "delta": -9114.183644727455,
      "relative_change_pct": -0.9586982508261516,
      "rows": 3937
    },
    "2022-09": {
      "candidate_mae": 1178225.8298355546,
      "control_mae": 1183834.4147522436,
      "delta": -5608.5849166889675,
      "relative_change_pct": -0.47376430747392567,
      "rows": 6480
    },
    "2022-10": {
      "candidate_mae": 1408621.3954841301,
      "control_mae": 1501876.047428305,
      "delta": -93254.6519441749,
      "relative_change_pct": -6.209210946792637,
      "rows": 5654
    },
    "2022-11": {
      "candidate_mae": 1493507.7339497113,
      "control_mae": 1518956.0042070919,
      "delta": -25448.270257380558,
      "relative_change_pct": -1.6753790226244751,
      "rows": 4732
    },
    "2022-12": {
      "candidate_mae": 1313060.020657954,
      "control_mae": 1228901.1954487665,
      "delta": 84158.82520918758,
      "relative_change_pct": 6.8482987502062524,
      "rows": 6122
    },
    "2023-01": {
      "candidate_mae": 1236109.5625675314,
      "control_mae": 1241965.4282693113,
      "delta": -5855.865701779956,
      "relative_change_pct": -0.4714998959302878,
      "rows": 3545
    },
    "2023-02": {
      "candidate_mae": 1157697.6943450356,
      "control_mae": 1181921.8231404028,
      "delta": -24224.1287953672,
      "relative_change_pct": -2.0495542362525248,
      "rows": 4227
    },
    "2023-03": {
      "candidate_mae": 1409153.7741510891,
      "control_mae": 1425446.287641478,
      "delta": -16292.513490388868,
      "relative_change_pct": -1.1429763177780774,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1507198.6227785894,
      "control_mae": 1508305.4461258438,
      "delta": -1106.8233472544234,
      "relative_change_pct": -0.07338191014939005,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1527395.3688208954,
      "control_mae": 1532629.9115433628,
      "delta": -5234.542722467333,
      "relative_change_pct": -0.3415399036024381,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1548695.562082522,
      "control_mae": 1543362.271613415,
      "delta": 5333.290469106985,
      "relative_change_pct": 0.34556309734924506,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1492392.966689555,
      "control_mae": 1539075.911771725,
      "delta": -46682.94508216996,
      "relative_change_pct": -3.0331801521362483,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1456887.3193033582,
      "control_mae": 1454058.0801178366,
      "delta": 2829.239185521612,
      "relative_change_pct": 0.19457539036489732,
      "rows": 2324
    }
  },
  "overall_mae": {
    "candidate_mae": 1398534.63391649,
    "control_mae": 1407340.7168109824,
    "delta": -8806.082894492429,
    "relative_change_pct": -0.625725013800987
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 728401.2733474829,
      "control_mae": 726223.9901538457,
      "delta": 2177.2831936371513,
      "relative_change_pct": 0.2998087674266872,
      "rows": 22139
    },
    {
      "band": "10-20m",
      "candidate_mae": 1104613.526103061,
      "control_mae": 1119987.3894261469,
      "delta": -15373.863323085941,
      "relative_change_pct": -1.3726818237625977,
      "rows": 43914
    },
    {
      "band": "20-50m",
      "candidate_mae": 2664381.1866821246,
      "control_mae": 2676120.810418359,
      "delta": -11739.623736234382,
      "relative_change_pct": -0.4386806339433952,
      "rows": 14197
    },
    {
      "band": "50-100m",
      "candidate_mae": 6379747.231114048,
      "control_mae": 6578422.082288335,
      "delta": -198674.8511742875,
      "relative_change_pct": -3.020098873089905,
      "rows": 802
    },
    {
      "band": ">=100m",
      "candidate_mae": 33141270.818477042,
      "control_mae": 31835906.574960757,
      "delta": 1305364.243516285,
      "relative_change_pct": 4.100289214138373,
      "rows": 182
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 1313060.020657954,
      "control_mae": 1228901.1954487665,
      "delta": 84158.82520918758,
      "group": "2022-12",
      "relative_change_pct": 6.8482987502062524,
      "rows": 6122,
      "section": "months"
    },
    {
      "candidate_mae": 4450860.708465353,
      "control_mae": 3988384.5966167687,
      "delta": 462476.1118485839,
      "group": "新竹市",
      "relative_change_pct": 11.595574615369065,
      "rows": 1250,
      "section": "cities"
    },
    {
      "candidate_mae": 1307685.1214580873,
      "control_mae": 1216912.2856501525,
      "delta": 90772.83580793487,
      "group": "4",
      "relative_change_pct": 7.45927515716042,
      "rows": 5881,
      "section": "relative_forecast_months"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 1054864.2273708566,
      "control_mae": 1056814.584844697,
      "delta": -1950.3574738404714,
      "relative_change_pct": -0.18455058264805116,
      "rows": 6145
    },
    "1": {
      "candidate_mae": 1178331.2676234997,
      "control_mae": 1196559.960172717,
      "delta": -18228.69254921726,
      "relative_change_pct": -1.523424914417665,
      "rows": 6235
    },
    "10": {
      "candidate_mae": 1555795.3463695482,
      "control_mae": 1557748.2179694306,
      "delta": -1952.8715998823754,
      "relative_change_pct": -0.12536503507787666,
      "rows": 11490
    },
    "11": {
      "candidate_mae": 1502679.718830837,
      "control_mae": 1536870.8291656736,
      "delta": -34191.110334836645,
      "relative_change_pct": -2.2247224481056804,
      "rows": 6914
    },
    "2": {
      "candidate_mae": 1465635.4351578613,
      "control_mae": 1564599.8870093762,
      "delta": -98964.45185151487,
      "relative_change_pct": -6.325224274474322,
      "rows": 5228
    },
    "3": {
      "candidate_mae": 1403569.6971430453,
      "control_mae": 1423037.1772045153,
      "delta": -19467.480061470065,
      "relative_change_pct": -1.3680232936509042,
      "rows": 5046
    },
    "4": {
      "candidate_mae": 1307685.1214580873,
      "control_mae": 1216912.2856501525,
      "delta": 90772.83580793487,
      "relative_change_pct": 7.45927515716042,
      "rows": 5881
    },
    "5": {
      "candidate_mae": 1257633.6535963858,
      "control_mae": 1278696.876809706,
      "delta": -21063.223213320132,
      "relative_change_pct": -1.6472413122547052,
      "rows": 3266
    },
    "6": {
      "candidate_mae": 1177868.8084269885,
      "control_mae": 1193017.763002033,
      "delta": -15148.954575044569,
      "relative_change_pct": -1.269801259029431,
      "rows": 5055
    },
    "7": {
      "candidate_mae": 1467942.694388997,
      "control_mae": 1487234.18211577,
      "delta": -19291.487726772903,
      "relative_change_pct": -1.2971385380161473,
      "rows": 7516
    },
    "8": {
      "candidate_mae": 1566344.0531067648,
      "control_mae": 1560492.9821896846,
      "delta": 5851.0709170801565,
      "relative_change_pct": 0.3749501589472021,
      "rows": 9057
    },
    "9": {
      "candidate_mae": 1467680.624361151,
      "control_mae": 1468927.3561406387,
      "delta": -1246.7317794878036,
      "relative_change_pct": -0.08487361708365096,
      "rows": 9401
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1688594.1743921083,
      "control_mae": 1696977.6628507746,
      "delta": -8383.488458666252,
      "relative_change_pct": -0.49402467941638795,
      "rows": 52253
    },
    "seen": {
      "candidate_mae": 875554.7102260551,
      "control_mae": 885122.7346358591,
      "delta": -9568.024409804028,
      "relative_change_pct": -1.0809827875159401,
      "rows": 28981
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 1390878.6847519693,
    "control_mae": 1399695.9217754127,
    "delta": -8817.23702344345,
    "relative_change_pct": -0.6299394665849583
  }
}
```

所有城市／月份／新舊案／價格帶／季度／誤差尾部：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 279735.0033029721,
      "rows": 3
    },
    "南投縣": {
      "mae": 795148.7587618007,
      "rows": 659
    },
    "嘉義市": {
      "mae": 1487741.7298481574,
      "rows": 569
    },
    "嘉義縣": {
      "mae": 802425.0227985018,
      "rows": 798
    },
    "基隆市": {
      "mae": 1335670.6145798536,
      "rows": 737
    },
    "宜蘭縣": {
      "mae": 833557.5667857141,
      "rows": 2242
    },
    "屏東縣": {
      "mae": 792741.5474508124,
      "rows": 615
    },
    "彰化縣": {
      "mae": 1063738.7942077988,
      "rows": 2246
    },
    "新北市": {
      "mae": 1145906.7904371824,
      "rows": 15979
    },
    "新竹市": {
      "mae": 4450860.708465353,
      "rows": 1250
    },
    "新竹縣": {
      "mae": 2730845.3890422736,
      "rows": 3639
    },
    "桃園市": {
      "mae": 986189.4194385294,
      "rows": 15846
    },
    "澎湖縣": {
      "mae": 858929.3701140208,
      "rows": 28
    },
    "臺中市": {
      "mae": 1579950.4766610668,
      "rows": 14719
    },
    "臺北市": {
      "mae": 4468936.3543950785,
      "rows": 2698
    },
    "臺南市": {
      "mae": 1221782.4282065793,
      "rows": 7530
    },
    "臺東縣": {
      "mae": 736286.8448863544,
      "rows": 265
    },
    "花蓮縣": {
      "mae": 560002.2152601547,
      "rows": 535
    },
    "苗栗縣": {
      "mae": 1288980.35842659,
      "rows": 2402
    },
    "金門縣": {
      "mae": 672492.3941265355,
      "rows": 29
    },
    "雲林縣": {
      "mae": 935963.2250779648,
      "rows": 1350
    },
    "高雄市": {
      "mae": 989113.9111499307,
      "rows": 7095
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1220546.2395406482,
      "rows": 17608
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1330257.1974723183,
      "rows": 14193
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1440132.0748636953,
      "rows": 22487
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1516091.2155628249,
      "rows": 26946
    }
  },
  "months": {
    "2022-08": {
      "mae": 941569.061774086,
      "rows": 3937
    },
    "2022-09": {
      "mae": 1178225.8298355546,
      "rows": 6480
    },
    "2022-10": {
      "mae": 1408621.3954841301,
      "rows": 5654
    },
    "2022-11": {
      "mae": 1493507.7339497113,
      "rows": 4732
    },
    "2022-12": {
      "mae": 1313060.020657954,
      "rows": 6122
    },
    "2023-01": {
      "mae": 1236109.5625675314,
      "rows": 3545
    },
    "2023-02": {
      "mae": 1157697.6943450356,
      "rows": 4227
    },
    "2023-03": {
      "mae": 1409153.7741510891,
      "rows": 7500
    },
    "2023-04": {
      "mae": 1507198.6227785894,
      "rows": 8639
    },
    "2023-05": {
      "mae": 1527395.3688208954,
      "rows": 8750
    },
    "2023-06": {
      "mae": 1548695.562082522,
      "rows": 13180
    },
    "2023-07": {
      "mae": 1492392.966689555,
      "rows": 6144
    },
    "2023-08": {
      "mae": 1456887.3193033582,
      "rows": 2324
    }
  },
  "overall_mae": 1398534.63391649,
  "price_bands": [
    {
      "absolute_error_share": 0.14194419366510316,
      "absolute_error_sum": 16126075790.639923,
      "band": "<10m",
      "mae": 728401.2733474829,
      "mean_signed_error_prediction_minus_truth": 386128.37370817206,
      "rows": 22139,
      "underprediction_fraction": 0.3293735037716247
    },
    {
      "absolute_error_share": 0.42697484536843155,
      "absolute_error_sum": 48507998385.28982,
      "band": "10-20m",
      "mae": 1104613.526103061,
      "mean_signed_error_prediction_minus_truth": -39153.11907692669,
      "rows": 43914,
      "underprediction_fraction": 0.47925490731885045
    },
    {
      "absolute_error_share": 0.33295219031972423,
      "absolute_error_sum": 37826219707.326126,
      "band": "20-50m",
      "mae": 2664381.1866821246,
      "mean_signed_error_prediction_minus_truth": -918076.1302859265,
      "rows": 14197,
      "underprediction_fraction": 0.5556807776290765
    },
    {
      "absolute_error_share": 0.04503672231161712,
      "absolute_error_sum": 5116557279.353466,
      "band": "50-100m",
      "mae": 6379747.231114048,
      "mean_signed_error_prediction_minus_truth": -1874874.5727688803,
      "rows": 802,
      "underprediction_fraction": 0.5149625935162094
    },
    {
      "absolute_error_share": 0.053092048335124,
      "absolute_error_sum": 6031711288.962822,
      "band": ">=100m",
      "mae": 33141270.818477042,
      "mean_signed_error_prediction_minus_truth": -5910379.946884836,
      "rows": 182,
      "underprediction_fraction": 0.45604395604395603
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 1054864.2273708566,
      "rows": 6145
    },
    "1": {
      "mae": 1178331.2676234997,
      "rows": 6235
    },
    "10": {
      "mae": 1555795.3463695482,
      "rows": 11490
    },
    "11": {
      "mae": 1502679.718830837,
      "rows": 6914
    },
    "2": {
      "mae": 1465635.4351578613,
      "rows": 5228
    },
    "3": {
      "mae": 1403569.6971430453,
      "rows": 5046
    },
    "4": {
      "mae": 1307685.1214580873,
      "rows": 5881
    },
    "5": {
      "mae": 1257633.6535963858,
      "rows": 3266
    },
    "6": {
      "mae": 1177868.8084269885,
      "rows": 5055
    },
    "7": {
      "mae": 1467942.694388997,
      "rows": 7516
    },
    "8": {
      "mae": 1566344.0531067648,
      "rows": 9057
    },
    "9": {
      "mae": 1467680.624361151,
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
      "mae": 1688594.1743921083,
      "rows": 52253
    },
    "seen": {
      "mae": 875554.7102260551,
      "rows": 28981
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 795573.0153380111,
      "0.9": 3096950.835904639,
      "0.95": 4572534.4252090985,
      "0.99": 8943542.202529123
    },
    "absolute_error_sum": 113608562451.57214,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 721361507.7495947,
    "mean_signed_error_prediction_minus_truth": -108133.77877467462,
    "top10_absolute_error_sum": 2653848454.626834,
    "top10_error_rows": 10,
    "top10_error_share": 0.02335958133224411,
    "top1pct_absolute_error_sum": 14288144366.359772,
    "top1pct_error_rows": 813,
    "top1pct_error_share": 0.12576643923692257,
    "underprediction_fraction": 0.45206440652928576
  },
  "weighted_mae": 1390878.6847519693
}
```

## confirmation完整診斷

native fit實測54.156秒。

|層|總價MAE|
|---|---:|
|base|1,848,073.86|
|native|1,825,496.61|
|reference|1,535,390.00|
|spatial|1,527,255.59|
|median|1,403,406.07|

所有季度／至少500列群退步超5%（群重疊不能相加）：

```json
[
  {
    "candidate_mae": 1987705.2387083306,
    "control_mae": 1823161.7663045553,
    "delta": 164543.4724037752,
    "group": "基隆市",
    "relative_change_pct": 9.025171295539803,
    "rows": 1100,
    "section": "cities"
  },
  {
    "candidate_mae": 1400196.2568464503,
    "control_mae": 1317138.3839079912,
    "delta": 83057.87293845904,
    "group": "新北市",
    "relative_change_pct": 6.305933678132111,
    "rows": 21822,
    "section": "cities"
  }
]
```

原生CBM及葉數證據：

```json
{
  "actual_grow_policy": "Depthwise",
  "actual_boosting_type": "Plain",
  "actual_model_depth": 8,
  "actual_trees": 2999,
  "actual_min_data_in_leaf": 1,
  "tree_leaf_counts_count": 2999,
  "minimum_tree_leaf_count": 12,
  "maximum_tree_leaf_count": 254,
  "tree_leaf_counts_distribution": {
    "100": 3,
    "101": 1,
    "103": 4,
    "104": 2,
    "106": 1,
    "107": 2,
    "111": 1,
    "112": 1,
    "113": 1,
    "117": 1,
    "118": 1,
    "12": 25,
    "120": 1,
    "124": 1,
    "129": 1,
    "13": 14,
    "131": 1,
    "136": 1,
    "138": 1,
    "14": 80,
    "141": 1,
    "143": 2,
    "148": 1,
    "149": 1,
    "15": 74,
    "156": 1,
    "16": 31,
    "17": 79,
    "174": 1,
    "177": 1,
    "178": 2,
    "18": 18,
    "181": 1,
    "183": 1,
    "184": 1,
    "185": 2,
    "189": 2,
    "19": 48,
    "190": 1,
    "191": 1,
    "193": 1,
    "20": 70,
    "207": 1,
    "21": 84,
    "211": 1,
    "213": 1,
    "214": 1,
    "218": 1,
    "22": 110,
    "227": 1,
    "229": 1,
    "23": 68,
    "233": 3,
    "234": 3,
    "235": 1,
    "236": 3,
    "237": 3,
    "238": 1,
    "239": 2,
    "24": 51,
    "240": 3,
    "241": 2,
    "242": 2,
    "243": 3,
    "244": 3,
    "245": 2,
    "246": 3,
    "247": 2,
    "248": 3,
    "249": 2,
    "25": 73,
    "251": 4,
    "254": 1,
    "26": 63,
    "27": 81,
    "28": 57,
    "29": 57,
    "30": 68,
    "31": 66,
    "32": 74,
    "33": 53,
    "34": 58,
    "35": 95,
    "36": 88,
    "37": 80,
    "38": 61,
    "39": 69,
    "40": 77,
    "41": 61,
    "42": 77,
    "43": 52,
    "44": 42,
    "45": 60,
    "46": 39,
    "47": 33,
    "48": 53,
    "49": 38,
    "50": 44,
    "51": 43,
    "52": 21,
    "53": 33,
    "54": 41,
    "55": 33,
    "56": 28,
    "57": 27,
    "58": 31,
    "59": 19,
    "60": 16,
    "61": 15,
    "62": 22,
    "63": 13,
    "64": 13,
    "65": 10,
    "66": 7,
    "67": 10,
    "68": 11,
    "69": 5,
    "70": 11,
    "71": 6,
    "72": 11,
    "73": 5,
    "74": 5,
    "75": 5,
    "76": 7,
    "77": 5,
    "78": 5,
    "79": 4,
    "80": 4,
    "81": 2,
    "82": 2,
    "83": 6,
    "84": 2,
    "85": 8,
    "86": 6,
    "87": 4,
    "88": 1,
    "89": 3,
    "90": 4,
    "91": 2,
    "92": 7,
    "93": 6,
    "94": 5,
    "95": 1,
    "96": 4,
    "97": 1,
    "98": 1
  },
  "runtime_parameters_not_recovered_from_native_model": {
    "allow_writing_files": false,
    "thread_count": 4,
    "verbose": 300
  }
}
```

原單價逆轉換與未投影總價：

```json
{
  "extra_price_cap_applied": false,
  "inverse_applications_per_prediction": 1,
  "native_TWD_max": 2110995536.6284583,
  "native_TWD_min": 313667.8120388093,
  "native_zero_rows": 0,
  "nonnegative_projection_after_TWD_reconstruction": true,
  "query_y_used_for_diagnostics": false,
  "raw_negative_unit_rows": 0,
  "raw_unit_max": 48.80370404891989,
  "raw_unit_min": 3.5897870554528613,
  "raw_units": "unit",
  "unprojected_TWD_overall_mae": 1825496.61478066
}
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+20,177.67|[+12,639.95, +26,503.14]|
|month|weighted|+21,530.69|[+13,579.64, +26,990.02]|
|project|overall|+20,177.67|[+4,686.31, +38,947.79]|
|project|weighted|+21,530.69|[+5,016.21, +41,808.83]|

完整條件抽樣及支援數：

```json
{
  "candidate": "R116",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R27 deeper trees, R84 extra trees, R92 shallow unit and R112 Ordered supplied contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R116 minus v22",
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
          "bootstrap_mean_delta": 19904.291652822198,
          "candidate_mae": 1403406.0709714661,
          "control_mae": 1383228.4054499408,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0,
          "percentile_2_5": 12639.954621055163,
          "percentile_97_5": 26503.14357544473,
          "point_delta": 20177.665521525312,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 20765.858489294627,
          "candidate_mae": 1468684.107214733,
          "control_mae": 1447153.4154201136,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0005,
          "percentile_2_5": 13579.639216746664,
          "percentile_97_5": 26990.016626117293,
          "point_delta": 21530.69179461943,
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
          "bootstrap_mean_delta": 20192.711661930978,
          "candidate_mae": 1403406.0709714661,
          "control_mae": 1383228.4054499408,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.002,
          "percentile_2_5": 4686.305198361591,
          "percentile_97_5": 38947.7926304691,
          "point_delta": 20177.665521525312,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 21538.435918038518,
          "candidate_mae": 1468684.107214733,
          "control_mae": 1447153.4154201136,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0035,
          "percentile_2_5": 5016.20814346255,
          "percentile_97_5": 41808.8315683963,
          "point_delta": 21530.69179461943,
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

v24次對照及反證（不以較弱對照替換v22）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R116",
  "cities": {
    "南投縣": {
      "candidate_mae": 966418.3656657404,
      "control_mae": 965273.7684037857,
      "delta": 1144.5972619547974,
      "relative_change_pct": 0.118577475056382,
      "rows": 1015
    },
    "嘉義市": {
      "candidate_mae": 1201842.2201706558,
      "control_mae": 1192323.4691625503,
      "delta": 9518.751008105464,
      "relative_change_pct": 0.7983362949981291,
      "rows": 689
    },
    "嘉義縣": {
      "candidate_mae": 1421408.9436800657,
      "control_mae": 1401441.997463974,
      "delta": 19966.946216091746,
      "relative_change_pct": 1.4247429613372227,
      "rows": 2251
    },
    "基隆市": {
      "candidate_mae": 1987705.2387083306,
      "control_mae": 1811344.9669182824,
      "delta": 176360.2717900481,
      "relative_change_pct": 9.736426523441157,
      "rows": 1100
    },
    "宜蘭縣": {
      "candidate_mae": 932534.8386796602,
      "control_mae": 940693.1296201872,
      "delta": -8158.290940526989,
      "relative_change_pct": -0.8672637955611484,
      "rows": 2907
    },
    "屏東縣": {
      "candidate_mae": 792337.687685576,
      "control_mae": 757610.0933880565,
      "delta": 34727.594297519536,
      "relative_change_pct": 4.583834692884915,
      "rows": 1013
    },
    "彰化縣": {
      "candidate_mae": 1044972.0088768986,
      "control_mae": 1068085.3267788775,
      "delta": -23113.317901978968,
      "relative_change_pct": -2.1639954526558203,
      "rows": 3295
    },
    "新北市": {
      "candidate_mae": 1400196.2568464503,
      "control_mae": 1308034.3659481003,
      "delta": 92161.89089834993,
      "relative_change_pct": 7.045831003954424,
      "rows": 21822
    },
    "新竹市": {
      "candidate_mae": 2658560.620383806,
      "control_mae": 2414804.346618525,
      "delta": 243756.27376528084,
      "relative_change_pct": 10.094245279399766,
      "rows": 2131
    },
    "新竹縣": {
      "candidate_mae": 2115630.520731424,
      "control_mae": 2023872.173760328,
      "delta": 91758.346971096,
      "relative_change_pct": 4.533801499953932,
      "rows": 5589
    },
    "桃園市": {
      "candidate_mae": 1209362.5452966832,
      "control_mae": 1236542.7211960873,
      "delta": -27180.1758994041,
      "relative_change_pct": -2.1980781928110957,
      "rows": 26327
    },
    "澎湖縣": {
      "candidate_mae": 2762868.9985404764,
      "control_mae": 2558605.6252502515,
      "delta": 204263.37329022493,
      "relative_change_pct": 7.983386391181188,
      "rows": 30
    },
    "臺中市": {
      "candidate_mae": 1446750.408696916,
      "control_mae": 1396562.2952675254,
      "delta": 50188.11342939059,
      "relative_change_pct": 3.5936895618234175,
      "rows": 27157
    },
    "臺北市": {
      "candidate_mae": 4124851.9708680087,
      "control_mae": 4016190.85969717,
      "delta": 108661.11117083859,
      "relative_change_pct": 2.7055763773893924,
      "rows": 4957
    },
    "臺南市": {
      "candidate_mae": 943452.3081407242,
      "control_mae": 924730.213914743,
      "delta": 18722.094225981156,
      "relative_change_pct": 2.02460068290872,
      "rows": 9822
    },
    "臺東縣": {
      "candidate_mae": 506049.2940368769,
      "control_mae": 511517.47384219157,
      "delta": -5468.179805314692,
      "relative_change_pct": -1.0690113407546429,
      "rows": 115
    },
    "花蓮縣": {
      "candidate_mae": 616369.2903519324,
      "control_mae": 617584.5960098031,
      "delta": -1215.3056578707183,
      "relative_change_pct": -0.19678367396511737,
      "rows": 583
    },
    "苗栗縣": {
      "candidate_mae": 1388524.3629003365,
      "control_mae": 1487508.760371721,
      "delta": -98984.39747138438,
      "relative_change_pct": -6.654374085612019,
      "rows": 4001
    },
    "金門縣": {
      "candidate_mae": 795427.7786771974,
      "control_mae": 790515.819334728,
      "delta": 4911.959342469461,
      "relative_change_pct": 0.6213612962993207,
      "rows": 86
    },
    "雲林縣": {
      "candidate_mae": 903691.405578652,
      "control_mae": 920111.576360945,
      "delta": -16420.17078229296,
      "relative_change_pct": -1.7845847399545802,
      "rows": 2364
    },
    "高雄市": {
      "candidate_mae": 993266.6539483925,
      "control_mae": 1000969.9302470884,
      "delta": -7703.276298695942,
      "relative_change_pct": -0.7695811897960207,
      "rows": 16443
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 919014.1035090482,
      "control_mae": 912993.6515105661,
      "delta": 6020.451998482109,
      "relative_change_pct": 0.6594188238353194,
      "rows": 26579
    },
    "Q2": {
      "candidate_mae": 1202412.4533103504,
      "control_mae": 1165407.6059651442,
      "delta": 37004.84734520619,
      "relative_change_pct": 3.1752707941664964,
      "rows": 31157
    },
    "Q3": {
      "candidate_mae": 1503326.255743214,
      "control_mae": 1475619.1470166727,
      "delta": 27707.10872654128,
      "relative_change_pct": 1.8776598814509842,
      "rows": 37236
    },
    "Q4": {
      "candidate_mae": 1801505.0831984873,
      "control_mae": 1757752.2708320958,
      "delta": 43752.81236639153,
      "relative_change_pct": 2.4891341682454238,
      "rows": 38725
    }
  },
  "months": {
    "2023-08": {
      "candidate_mae": 658953.8482824195,
      "control_mae": 661692.706617775,
      "delta": -2738.8583353555296,
      "relative_change_pct": -0.4139169599367556,
      "rows": 5076
    },
    "2023-09": {
      "candidate_mae": 811530.5388052458,
      "control_mae": 819211.9589262545,
      "delta": -7681.420121008763,
      "relative_change_pct": -0.9376596663794852,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1056163.5602143484,
      "control_mae": 1031792.2770912236,
      "delta": 24371.28312312474,
      "relative_change_pct": 2.3620338768023177,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1221869.6858885682,
      "control_mae": 1207667.0396223704,
      "delta": 14202.646266197786,
      "relative_change_pct": 1.176039901746334,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1149424.9024528484,
      "control_mae": 1122539.8496172,
      "delta": 26885.052835648414,
      "relative_change_pct": 2.3950199046222322,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1230691.8684673228,
      "control_mae": 1177153.3779831533,
      "delta": 53538.49048416945,
      "relative_change_pct": 4.548132085888272,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1205983.2565053233,
      "control_mae": 1170552.7720134521,
      "delta": 35430.484491871204,
      "relative_change_pct": 3.0268165040460073,
      "rows": 7388
    },
    "2024-03": {
      "candidate_mae": 1412005.3662833194,
      "control_mae": 1396897.8203355782,
      "delta": 15107.545947741251,
      "relative_change_pct": 1.0815068738608205,
      "rows": 12933
    },
    "2024-04": {
      "candidate_mae": 1621045.728359627,
      "control_mae": 1587211.1045129716,
      "delta": 33834.62384665548,
      "relative_change_pct": 2.131702818261058,
      "rows": 13856
    },
    "2024-05": {
      "candidate_mae": 1720119.8600881943,
      "control_mae": 1671790.446103836,
      "delta": 48329.41398435831,
      "relative_change_pct": 2.890877507823522,
      "rows": 14480
    },
    "2024-06": {
      "candidate_mae": 1816489.648314582,
      "control_mae": 1771497.5397129185,
      "delta": 44992.108601663494,
      "relative_change_pct": 2.539778215495277,
      "rows": 14671
    },
    "2024-07": {
      "candidate_mae": 1798965.285122343,
      "control_mae": 1756731.8339828255,
      "delta": 42233.451139517594,
      "relative_change_pct": 2.404092094338998,
      "rows": 12090
    },
    "2024-08": {
      "candidate_mae": 1917109.5193093626,
      "control_mae": 1893707.7653277898,
      "delta": 23401.75398157281,
      "relative_change_pct": 1.2357637440179214,
      "rows": 3109
    }
  },
  "overall_mae": {
    "candidate_mae": 1403406.0709714752,
    "control_mae": 1373195.9153073074,
    "delta": 30210.155664167833,
    "relative_change_pct": 2.1999887508700535
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 654914.6523791839,
      "control_mae": 652232.5472355905,
      "delta": 2682.1051435933914,
      "relative_change_pct": 0.41121915104684253,
      "rows": 25585
    },
    {
      "band": "10-20m",
      "candidate_mae": 1042980.130968683,
      "control_mae": 1040802.0332451425,
      "delta": 2178.0977235405007,
      "relative_change_pct": 0.20927108652443305,
      "rows": 74109
    },
    {
      "band": "20-50m",
      "candidate_mae": 2368214.973204605,
      "control_mae": 2283352.845939235,
      "delta": 84862.12726536999,
      "relative_change_pct": 3.716557754807395,
      "rows": 31959
    },
    {
      "band": "50-100m",
      "candidate_mae": 6444926.365292149,
      "control_mae": 6217166.085579387,
      "delta": 227760.27971276175,
      "relative_change_pct": 3.663409929502252,
      "rows": 1725
    },
    {
      "band": ">=100m",
      "candidate_mae": 21246693.588773828,
      "control_mae": 19039879.576983564,
      "delta": 2206814.0117902644,
      "relative_change_pct": 11.59048303256067,
      "rows": 319
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 1987705.2387083306,
      "control_mae": 1811344.9669182824,
      "delta": 176360.2717900481,
      "group": "基隆市",
      "relative_change_pct": 9.736426523441157,
      "rows": 1100,
      "section": "cities"
    },
    {
      "candidate_mae": 1400196.2568464503,
      "control_mae": 1308034.3659481003,
      "delta": 92161.89089834993,
      "group": "新北市",
      "relative_change_pct": 7.045831003954424,
      "rows": 21822,
      "section": "cities"
    },
    {
      "candidate_mae": 2658560.620383806,
      "control_mae": 2414804.346618525,
      "delta": 243756.27376528084,
      "group": "新竹市",
      "relative_change_pct": 10.094245279399766,
      "rows": 2131,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 672807.1169369243,
      "control_mae": 678922.6899683854,
      "delta": -6115.573031461099,
      "relative_change_pct": -0.900776056217222,
      "rows": 7252
    },
    "1": {
      "candidate_mae": 884064.5878782254,
      "control_mae": 889826.2099094184,
      "delta": -5761.622031192994,
      "relative_change_pct": -0.6474996990456721,
      "rows": 9394
    },
    "10": {
      "candidate_mae": 1796932.6520635574,
      "control_mae": 1753961.107622659,
      "delta": 42971.544440898346,
      "relative_change_pct": 2.449971339395462,
      "rows": 14072
    },
    "11": {
      "candidate_mae": 1841808.240990616,
      "control_mae": 1798487.2970159906,
      "delta": 43320.94397462532,
      "relative_change_pct": 2.408743394879822,
      "rows": 11312
    },
    "2": {
      "candidate_mae": 1131820.789953928,
      "control_mae": 1105797.1911768373,
      "delta": 26023.598777090665,
      "relative_change_pct": 2.353378990716663,
      "rows": 9933
    },
    "3": {
      "candidate_mae": 1204580.7455013779,
      "control_mae": 1182811.8951608276,
      "delta": 21768.8503405503,
      "relative_change_pct": 1.8404321456025243,
      "rows": 10809
    },
    "4": {
      "candidate_mae": 1143977.888866443,
      "control_mae": 1113684.87976537,
      "delta": 30293.009101073025,
      "relative_change_pct": 2.7200700711187826,
      "rows": 9935
    },
    "5": {
      "candidate_mae": 1255913.8772474872,
      "control_mae": 1196689.8803220654,
      "delta": 59223.99692542176,
      "relative_change_pct": 4.948984519655401,
      "rows": 10413
    },
    "6": {
      "candidate_mae": 1278912.61376261,
      "control_mae": 1253733.2273133975,
      "delta": 25179.38644921244,
      "relative_change_pct": 2.0083528058970646,
      "rows": 8995
    },
    "7": {
      "candidate_mae": 1437106.4362342206,
      "control_mae": 1422678.507141905,
      "delta": 14427.929092315724,
      "relative_change_pct": 1.0141384030114269,
      "rows": 12434
    },
    "8": {
      "candidate_mae": 1687449.572950416,
      "control_mae": 1646182.6876466258,
      "delta": 41266.885303790215,
      "relative_change_pct": 2.5068229433748415,
      "rows": 14629
    },
    "9": {
      "candidate_mae": 1760566.809847021,
      "control_mae": 1717746.4536748976,
      "delta": 42820.35617212346,
      "relative_change_pct": 2.4928216897502433,
      "rows": 14519
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1838380.2575731948,
      "control_mae": 1792277.4115217023,
      "delta": 46102.846051492495,
      "relative_change_pct": 2.5723052555993364,
      "rows": 76096
    },
    "seen": {
      "candidate_mae": 828766.8163813734,
      "control_mae": 819552.3060656169,
      "delta": 9214.510315756546,
      "relative_change_pct": 1.1243346211777718,
      "rows": 57601
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 1468684.1072147426,
    "control_mae": 1436088.882157809,
    "delta": 32595.225056933472,
    "relative_change_pct": 2.2697219832213453
  }
}
```

所有城市／月份／新舊案／價格帶／季度／誤差尾部：

```json
{
  "cities": {
    "南投縣": {
      "mae": 966418.3656657404,
      "rows": 1015
    },
    "嘉義市": {
      "mae": 1201842.2201706558,
      "rows": 689
    },
    "嘉義縣": {
      "mae": 1421408.9436800657,
      "rows": 2251
    },
    "基隆市": {
      "mae": 1987705.2387083306,
      "rows": 1100
    },
    "宜蘭縣": {
      "mae": 932534.8386796602,
      "rows": 2907
    },
    "屏東縣": {
      "mae": 792337.687685576,
      "rows": 1013
    },
    "彰化縣": {
      "mae": 1044972.0088768986,
      "rows": 3295
    },
    "新北市": {
      "mae": 1400196.2568464503,
      "rows": 21822
    },
    "新竹市": {
      "mae": 2658560.620383806,
      "rows": 2131
    },
    "新竹縣": {
      "mae": 2115630.520731424,
      "rows": 5589
    },
    "桃園市": {
      "mae": 1209362.5452966832,
      "rows": 26327
    },
    "澎湖縣": {
      "mae": 2762868.9985404764,
      "rows": 30
    },
    "臺中市": {
      "mae": 1446750.408696916,
      "rows": 27157
    },
    "臺北市": {
      "mae": 4124851.9708680087,
      "rows": 4957
    },
    "臺南市": {
      "mae": 943452.3081407242,
      "rows": 9822
    },
    "臺東縣": {
      "mae": 506049.2940368769,
      "rows": 115
    },
    "花蓮縣": {
      "mae": 616369.2903519324,
      "rows": 583
    },
    "苗栗縣": {
      "mae": 1388524.3629003365,
      "rows": 4001
    },
    "金門縣": {
      "mae": 795427.7786771974,
      "rows": 86
    },
    "雲林縣": {
      "mae": 903691.405578652,
      "rows": 2364
    },
    "高雄市": {
      "mae": 993266.6539483925,
      "rows": 16443
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 919014.1035090482,
      "rows": 26579
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1202412.4533103504,
      "rows": 31157
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1503326.255743214,
      "rows": 37236
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1801505.0831984873,
      "rows": 38725
    }
  },
  "months": {
    "2023-08": {
      "mae": 658953.8482824195,
      "rows": 5076
    },
    "2023-09": {
      "mae": 811530.5388052458,
      "rows": 8602
    },
    "2023-10": {
      "mae": 1056163.5602143484,
      "rows": 9855
    },
    "2023-11": {
      "mae": 1221869.6858885682,
      "rows": 10336
    },
    "2023-12": {
      "mae": 1149424.9024528484,
      "rows": 10416
    },
    "2024-01": {
      "mae": 1230691.8684673228,
      "rows": 10885
    },
    "2024-02": {
      "mae": 1205983.2565053233,
      "rows": 7388
    },
    "2024-03": {
      "mae": 1412005.3662833194,
      "rows": 12933
    },
    "2024-04": {
      "mae": 1621045.728359627,
      "rows": 13856
    },
    "2024-05": {
      "mae": 1720119.8600881943,
      "rows": 14480
    },
    "2024-06": {
      "mae": 1816489.648314582,
      "rows": 14671
    },
    "2024-07": {
      "mae": 1798965.285122343,
      "rows": 12090
    },
    "2024-08": {
      "mae": 1917109.5193093626,
      "rows": 3109
    }
  },
  "overall_mae": 1403406.0709714752,
  "price_bands": [
    {
      "absolute_error_share": 0.08930280803961349,
      "absolute_error_sum": 16755991381.121422,
      "band": "<10m",
      "mae": 654914.6523791839,
      "mean_signed_error_prediction_minus_truth": 230244.0423737188,
      "rows": 25585,
      "underprediction_fraction": 0.3757279656048466
    },
    {
      "absolute_error_share": 0.41194759804909714,
      "absolute_error_sum": 77294214525.95813,
      "band": "10-20m",
      "mae": 1042980.130968683,
      "mean_signed_error_prediction_minus_truth": -170510.9985970342,
      "rows": 74109,
      "underprediction_fraction": 0.5205575571118217
    },
    {
      "absolute_error_share": 0.4033752904789742,
      "absolute_error_sum": 75685782328.64597,
      "band": "20-50m",
      "mae": 2368214.973204605,
      "mean_signed_error_prediction_minus_truth": -1163998.507934452,
      "rows": 31959,
      "underprediction_fraction": 0.6339372320786008
    },
    {
      "absolute_error_share": 0.05925186790910133,
      "absolute_error_sum": 11117497980.128956,
      "band": "50-100m",
      "mae": 6444926.365292149,
      "mean_signed_error_prediction_minus_truth": -1799856.458442851,
      "rows": 1725,
      "underprediction_fraction": 0.5431884057971015
    },
    {
      "absolute_error_share": 0.036122435523213946,
      "absolute_error_sum": 6777695254.818851,
      "band": ">=100m",
      "mae": 21246693.588773828,
      "mean_signed_error_prediction_minus_truth": -4796921.345367907,
      "rows": 319,
      "underprediction_fraction": 0.4890282131661442
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 672807.1169369243,
      "rows": 7252
    },
    "1": {
      "mae": 884064.5878782254,
      "rows": 9394
    },
    "10": {
      "mae": 1796932.6520635574,
      "rows": 14072
    },
    "11": {
      "mae": 1841808.240990616,
      "rows": 11312
    },
    "2": {
      "mae": 1131820.789953928,
      "rows": 9933
    },
    "3": {
      "mae": 1204580.7455013779,
      "rows": 10809
    },
    "4": {
      "mae": 1143977.888866443,
      "rows": 9935
    },
    "5": {
      "mae": 1255913.8772474872,
      "rows": 10413
    },
    "6": {
      "mae": 1278912.61376261,
      "rows": 8995
    },
    "7": {
      "mae": 1437106.4362342206,
      "rows": 12434
    },
    "8": {
      "mae": 1687449.572950416,
      "rows": 14629
    },
    "9": {
      "mae": 1760566.809847021,
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
      "mae": 1838380.2575731948,
      "rows": 76096
    },
    "seen": {
      "mae": 828766.8163813734,
      "rows": 57601
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 783303.2516799187,
      "0.9": 3193208.6307810727,
      "0.95": 4638420.828836021,
      "0.99": 9039262.54501215
    },
    "absolute_error_sum": 187631181470.6733,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 514076332.12510395,
    "mean_signed_error_prediction_minus_truth": -363364.9549799879,
    "top10_absolute_error_sum": 2070620193.6153038,
    "top10_error_rows": 10,
    "top10_error_share": 0.011035586821900074,
    "top1pct_absolute_error_sum": 20879874604.550007,
    "top1pct_error_rows": 1337,
    "top1pct_error_share": 0.11128147486409941,
    "underprediction_fraction": 0.5201612601629056
  },
  "weighted_mae": 1468684.1072147426
}
```

## auxiliary完整診斷

native fit實測44.312秒。

|層|總價MAE|
|---|---:|
|base|2,388,319.93|
|native|2,347,396.21|
|reference|2,295,110.35|
|spatial|2,294,270.32|
|median|2,297,125.92|

所有季度／至少500列群退步超5%（群重疊不能相加）：

```json
[]
```

原生CBM及葉數證據：

```json
{
  "actual_grow_policy": "Depthwise",
  "actual_boosting_type": "Plain",
  "actual_model_depth": 8,
  "actual_trees": 2999,
  "actual_min_data_in_leaf": 1,
  "tree_leaf_counts_count": 2999,
  "minimum_tree_leaf_count": 14,
  "maximum_tree_leaf_count": 241,
  "tree_leaf_counts_distribution": {
    "100": 8,
    "101": 11,
    "102": 9,
    "103": 7,
    "104": 7,
    "105": 3,
    "106": 5,
    "107": 5,
    "108": 8,
    "109": 6,
    "110": 1,
    "111": 7,
    "112": 5,
    "113": 5,
    "114": 8,
    "115": 2,
    "116": 5,
    "117": 8,
    "118": 3,
    "119": 4,
    "120": 5,
    "121": 3,
    "122": 3,
    "123": 7,
    "124": 3,
    "125": 3,
    "126": 8,
    "127": 3,
    "128": 1,
    "129": 2,
    "130": 4,
    "131": 2,
    "133": 9,
    "135": 3,
    "136": 3,
    "137": 1,
    "138": 2,
    "139": 2,
    "14": 2,
    "141": 1,
    "142": 1,
    "143": 1,
    "144": 1,
    "145": 1,
    "146": 2,
    "149": 2,
    "150": 1,
    "154": 1,
    "156": 1,
    "164": 1,
    "169": 1,
    "17": 1,
    "170": 1,
    "175": 1,
    "177": 1,
    "178": 1,
    "180": 2,
    "182": 1,
    "185": 1,
    "188": 1,
    "189": 2,
    "19": 1,
    "190": 3,
    "194": 3,
    "195": 2,
    "200": 1,
    "201": 1,
    "204": 2,
    "206": 2,
    "208": 2,
    "209": 1,
    "21": 3,
    "212": 1,
    "214": 1,
    "215": 1,
    "216": 2,
    "218": 2,
    "22": 8,
    "222": 1,
    "223": 3,
    "227": 2,
    "229": 2,
    "23": 6,
    "231": 2,
    "235": 2,
    "236": 2,
    "237": 2,
    "24": 9,
    "241": 1,
    "25": 12,
    "26": 18,
    "27": 18,
    "28": 33,
    "29": 23,
    "30": 17,
    "31": 21,
    "32": 26,
    "33": 25,
    "34": 34,
    "35": 25,
    "36": 45,
    "37": 44,
    "38": 42,
    "39": 40,
    "40": 52,
    "41": 53,
    "42": 58,
    "43": 74,
    "44": 59,
    "45": 57,
    "46": 72,
    "47": 56,
    "48": 72,
    "49": 64,
    "50": 60,
    "51": 64,
    "52": 60,
    "53": 67,
    "54": 60,
    "55": 60,
    "56": 64,
    "57": 85,
    "58": 67,
    "59": 52,
    "60": 57,
    "61": 51,
    "62": 50,
    "63": 58,
    "64": 44,
    "65": 40,
    "66": 58,
    "67": 42,
    "68": 35,
    "69": 35,
    "70": 49,
    "71": 36,
    "72": 32,
    "73": 41,
    "74": 34,
    "75": 30,
    "76": 21,
    "77": 35,
    "78": 26,
    "79": 24,
    "80": 18,
    "81": 32,
    "82": 15,
    "83": 20,
    "84": 12,
    "85": 18,
    "86": 12,
    "87": 8,
    "88": 16,
    "89": 11,
    "90": 9,
    "91": 7,
    "92": 8,
    "93": 6,
    "94": 6,
    "95": 12,
    "96": 6,
    "97": 15,
    "98": 11,
    "99": 3
  },
  "runtime_parameters_not_recovered_from_native_model": {
    "allow_writing_files": false,
    "thread_count": 4,
    "verbose": 300
  }
}
```

原單價逆轉換與未投影總價：

```json
{
  "extra_price_cap_applied": false,
  "inverse_applications_per_prediction": 1,
  "native_TWD_max": 3874077339.9581394,
  "native_TWD_min": 94231.24061027414,
  "native_zero_rows": 0,
  "nonnegative_projection_after_TWD_reconstruction": true,
  "query_y_used_for_diagnostics": false,
  "raw_negative_unit_rows": 0,
  "raw_unit_max": 140.35705773183886,
  "raw_unit_min": 3.3730153146216253,
  "raw_units": "unit",
  "unprojected_TWD_overall_mae": 2347396.2085086433
}
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-12,226.44|[-23,341.49, -2,022.47]|
|month|weighted|-10,354.04|[-29,285.97, -764.16]|
|project|overall|-12,226.44|[-36,151.64, +9,793.38]|
|project|weighted|-10,354.04|[-29,627.09, +7,377.14]|

完整條件抽樣及支援數：

```json
{
  "candidate": "R116",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R27 deeper trees, R84 extra trees, R92 shallow unit and R112 Ordered supplied contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R116 minus v22",
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
          "bootstrap_mean_delta": -12754.923429420209,
          "candidate_mae": 2297125.915877968,
          "control_mae": 2309352.3576965467,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.992,
          "percentile_2_5": -23341.489292874,
          "percentile_97_5": -2022.4665850560207,
          "point_delta": -12226.441818578634,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -11888.35321943726,
          "candidate_mae": 2012811.6181801,
          "control_mae": 2023165.66227504,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.9865,
          "percentile_2_5": -29285.972208827163,
          "percentile_97_5": -764.1554077621549,
          "point_delta": -10354.044094939949,
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
          "bootstrap_mean_delta": -12226.273062668204,
          "candidate_mae": 2297125.915877968,
          "control_mae": 2309352.3576965467,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.845,
          "percentile_2_5": -36151.643609414015,
          "percentile_97_5": 9793.384093617997,
          "point_delta": -12226.441818578634,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -10459.180580242632,
          "candidate_mae": 2012811.6181801,
          "control_mae": 2023165.66227504,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.8665,
          "percentile_2_5": -29627.088294441375,
          "percentile_97_5": 7377.136063368816,
          "point_delta": -10354.044094939949,
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

v24次對照及反證（不以較弱對照替換v22）：

```json
{
  "both_point_metrics_improve": false,
  "candidate": "R116",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 466689.4190578535,
      "control_mae": 466689.4190578535,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1087458.0098139634,
      "control_mae": 1118876.9223743458,
      "delta": -31418.912560382392,
      "relative_change_pct": -2.808075842131855,
      "rows": 869
    },
    "嘉義市": {
      "candidate_mae": 2190833.1959067425,
      "control_mae": 2076628.4947043534,
      "delta": 114204.70120238909,
      "relative_change_pct": 5.499524902678764,
      "rows": 731
    },
    "嘉義縣": {
      "candidate_mae": 1352890.1707510625,
      "control_mae": 1377314.8007233748,
      "delta": -24424.629972312367,
      "relative_change_pct": -1.7733513035280235,
      "rows": 1560
    },
    "基隆市": {
      "candidate_mae": 3248561.6701048017,
      "control_mae": 2649701.530279867,
      "delta": 598860.1398249348,
      "relative_change_pct": 22.601041399620655,
      "rows": 1577
    },
    "宜蘭縣": {
      "candidate_mae": 1315526.0840677598,
      "control_mae": 1294116.4134795247,
      "delta": 21409.670588235138,
      "relative_change_pct": 1.654385213357305,
      "rows": 2330
    },
    "屏東縣": {
      "candidate_mae": 1398586.7036615438,
      "control_mae": 1418492.199859719,
      "delta": -19905.496198175242,
      "relative_change_pct": -1.403285559141163,
      "rows": 1240
    },
    "彰化縣": {
      "candidate_mae": 1436978.9912839201,
      "control_mae": 1437729.9372029828,
      "delta": -750.9459190627094,
      "relative_change_pct": -0.05223136137261144,
      "rows": 2607
    },
    "新北市": {
      "candidate_mae": 1913179.7880236295,
      "control_mae": 1931141.0431383061,
      "delta": -17961.255114676664,
      "relative_change_pct": -0.9300851006453545,
      "rows": 17191
    },
    "新竹市": {
      "candidate_mae": 4222157.355253105,
      "control_mae": 4206008.947003277,
      "delta": 16148.408249828033,
      "relative_change_pct": 0.3839366119592672,
      "rows": 1586
    },
    "新竹縣": {
      "candidate_mae": 2836324.85237765,
      "control_mae": 2839386.2871954134,
      "delta": -3061.434817763511,
      "relative_change_pct": -0.10782030016730922,
      "rows": 5042
    },
    "桃園市": {
      "candidate_mae": 1820610.7831219176,
      "control_mae": 1801809.0342872667,
      "delta": 18801.748834650964,
      "relative_change_pct": 1.0434928717120282,
      "rows": 20723
    },
    "澎湖縣": {
      "candidate_mae": 1097403.2644649458,
      "control_mae": 905431.1381984788,
      "delta": 191972.12626646692,
      "relative_change_pct": 21.202288961304184,
      "rows": 40
    },
    "臺中市": {
      "candidate_mae": 2865338.318592805,
      "control_mae": 2813849.3975944812,
      "delta": 51488.92099832371,
      "relative_change_pct": 1.8298392601374058,
      "rows": 20837
    },
    "臺北市": {
      "candidate_mae": 6787576.259169572,
      "control_mae": 6619596.082977112,
      "delta": 167980.17619245965,
      "relative_change_pct": 2.537619729161962,
      "rows": 4712
    },
    "臺南市": {
      "candidate_mae": 1749326.5246632611,
      "control_mae": 1770047.7116755582,
      "delta": -20721.1870122971,
      "relative_change_pct": -1.1706569758327057,
      "rows": 8886
    },
    "臺東縣": {
      "candidate_mae": 1923632.7320556408,
      "control_mae": 1883004.2703493305,
      "delta": 40628.46170631028,
      "relative_change_pct": 2.157640444372066,
      "rows": 235
    },
    "花蓮縣": {
      "candidate_mae": 779392.4920157843,
      "control_mae": 829143.651670074,
      "delta": -49751.15965428972,
      "relative_change_pct": -6.000306406987518,
      "rows": 691
    },
    "苗栗縣": {
      "candidate_mae": 1644400.7505841802,
      "control_mae": 1653807.183102974,
      "delta": -9406.432518793736,
      "relative_change_pct": -0.5687744384532671,
      "rows": 2607
    },
    "金門縣": {
      "candidate_mae": 1629342.8063585963,
      "control_mae": 1495237.6344433622,
      "delta": 134105.17191523407,
      "relative_change_pct": 8.96881999396423,
      "rows": 35
    },
    "雲林縣": {
      "candidate_mae": 1057200.4749378536,
      "control_mae": 1077147.497864338,
      "delta": -19947.022926484467,
      "relative_change_pct": -1.8518376513925399,
      "rows": 1873
    },
    "高雄市": {
      "candidate_mae": 1759679.3928310946,
      "control_mae": 1798279.317019422,
      "delta": -38599.92418832751,
      "relative_change_pct": -2.1464921396251935,
      "rows": 10138
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1790251.1609133754,
      "control_mae": 1762875.3535570733,
      "delta": 27375.80735630216,
      "relative_change_pct": 1.5529065796435428,
      "rows": 33694
    },
    "Q2": {
      "candidate_mae": 2374481.424163859,
      "control_mae": 2362943.1568230344,
      "delta": 11538.267340824474,
      "relative_change_pct": 0.4883006731460108,
      "rows": 29169
    },
    "Q3": {
      "candidate_mae": 2573900.863785542,
      "control_mae": 2551611.570300658,
      "delta": 22289.293484884314,
      "relative_change_pct": 0.8735378748207335,
      "rows": 25213
    },
    "Q4": {
      "candidate_mae": 2747021.437013157,
      "control_mae": 2720040.168653072,
      "delta": 26981.26836008532,
      "relative_change_pct": 0.9919437466780533,
      "rows": 17435
    }
  },
  "months": {
    "2021-08": {
      "candidate_mae": 1595853.5655748646,
      "control_mae": 1563207.6855669126,
      "delta": 32645.880007951986,
      "relative_change_pct": 2.088390449290341,
      "rows": 5654
    },
    "2021-09": {
      "candidate_mae": 1645439.1998062409,
      "control_mae": 1621468.4185501335,
      "delta": 23970.78125610738,
      "relative_change_pct": 1.478337843763945,
      "rows": 12352
    },
    "2021-10": {
      "candidate_mae": 1962482.4076290587,
      "control_mae": 1937066.7553911388,
      "delta": 25415.652237919858,
      "relative_change_pct": 1.3120689912820194,
      "rows": 11842
    },
    "2021-11": {
      "candidate_mae": 2063407.8960838038,
      "control_mae": 2042288.2736402748,
      "delta": 21119.62244352908,
      "relative_change_pct": 1.0341156396048061,
      "rows": 12524
    },
    "2021-12": {
      "candidate_mae": 2322456.672354941,
      "control_mae": 2308971.313995403,
      "delta": 13485.358359538019,
      "relative_change_pct": 0.584041831866815,
      "rows": 11482
    },
    "2022-01": {
      "candidate_mae": 2785991.023056833,
      "control_mae": 2773309.212675835,
      "delta": 12681.810380998068,
      "relative_change_pct": 0.4572807937547645,
      "rows": 8017
    },
    "2022-02": {
      "candidate_mae": 2462397.0731912036,
      "control_mae": 2452771.3776768353,
      "delta": 9625.695514368359,
      "relative_change_pct": 0.39244161123102406,
      "rows": 6204
    },
    "2022-03": {
      "candidate_mae": 2486567.3129399284,
      "control_mae": 2450727.282373386,
      "delta": 35840.03056654241,
      "relative_change_pct": 1.4624242699021752,
      "rows": 9626
    },
    "2022-04": {
      "candidate_mae": 2669954.6311241165,
      "control_mae": 2671623.6542875506,
      "delta": -1669.023163434118,
      "relative_change_pct": -0.06247224083210931,
      "rows": 7535
    },
    "2022-05": {
      "candidate_mae": 2670695.548672635,
      "control_mae": 2633628.192055837,
      "delta": 37067.35661679832,
      "relative_change_pct": 1.4074635413081287,
      "rows": 6424
    },
    "2022-06": {
      "candidate_mae": 2686362.3374456074,
      "control_mae": 2695679.6338523687,
      "delta": -9317.296406761277,
      "relative_change_pct": -0.34563811996628185,
      "rows": 5629
    },
    "2022-07": {
      "candidate_mae": 2850681.641924873,
      "control_mae": 2793109.3935268214,
      "delta": 57572.2483980516,
      "relative_change_pct": 2.0612242589380254,
      "rows": 6566
    },
    "2022-08": {
      "candidate_mae": 2786764.9419800546,
      "control_mae": 2766911.3766995794,
      "delta": 19853.565280475188,
      "relative_change_pct": 0.7175352795056584,
      "rows": 1656
    }
  },
  "overall_mae": {
    "candidate_mae": 2297125.9158779667,
    "control_mae": 2275409.1418580045,
    "delta": 21716.774019962177,
    "relative_change_pct": 0.9544118295239493
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 1116286.6244394088,
      "control_mae": 1107505.6060768394,
      "delta": 8781.01836256939,
      "relative_change_pct": 0.7928644617587749,
      "rows": 41132
    },
    {
      "band": "10-20m",
      "candidate_mae": 2034747.5373275145,
      "control_mae": 2024569.5907328655,
      "delta": 10177.946594648995,
      "relative_change_pct": 0.5027214989910385,
      "rows": 47793
    },
    {
      "band": "20-50m",
      "candidate_mae": 4657938.922856556,
      "control_mae": 4605603.27388477,
      "delta": 52335.64897178579,
      "relative_change_pct": 1.1363473112967741,
      "rows": 15033
    },
    {
      "band": "50-100m",
      "candidate_mae": 14219810.06415135,
      "control_mae": 13891916.012976805,
      "delta": 327894.05117454566,
      "relative_change_pct": 2.3603227291919358,
      "rows": 1382
    },
    {
      "band": ">=100m",
      "candidate_mae": 55764261.32341932,
      "control_mae": 54572263.048466735,
      "delta": 1191998.274952583,
      "relative_change_pct": 2.1842566321538563,
      "rows": 171
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 2190833.1959067425,
      "control_mae": 2076628.4947043534,
      "delta": 114204.70120238909,
      "group": "嘉義市",
      "relative_change_pct": 5.499524902678764,
      "rows": 731,
      "section": "cities"
    },
    {
      "candidate_mae": 3248561.6701048017,
      "control_mae": 2649701.530279867,
      "delta": 598860.1398249348,
      "group": "基隆市",
      "relative_change_pct": 22.601041399620655,
      "rows": 1577,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 1603042.2833962997,
      "control_mae": 1567209.9156508578,
      "delta": 35832.36774544185,
      "relative_change_pct": 2.286379596479312,
      "rows": 9387
    },
    "1": {
      "candidate_mae": 1728919.515870967,
      "control_mae": 1715851.8997759607,
      "delta": 13067.616095006233,
      "relative_change_pct": 0.7615818181459878,
      "rows": 12463
    },
    "10": {
      "candidate_mae": 2702554.9595222087,
      "control_mae": 2710524.938303519,
      "delta": -7969.978781310376,
      "relative_change_pct": -0.29403820155584615,
      "rows": 5992
    },
    "11": {
      "candidate_mae": 2880776.705288295,
      "control_mae": 2817934.7094697705,
      "delta": 62841.99581852462,
      "relative_change_pct": 2.2300728120968114,
      "rows": 6217
    },
    "2": {
      "candidate_mae": 2003161.1596820615,
      "control_mae": 1967431.6495803462,
      "delta": 35729.510101715336,
      "relative_change_pct": 1.8160483546829418,
      "rows": 11844
    },
    "3": {
      "candidate_mae": 2095207.6751498317,
      "control_mae": 2077612.941580479,
      "delta": 17594.733569352655,
      "relative_change_pct": 0.8468725438323469,
      "rows": 12872
    },
    "4": {
      "candidate_mae": 2501264.260466442,
      "control_mae": 2505314.56244677,
      "delta": -4050.301980327815,
      "relative_change_pct": -0.16166840048908515,
      "rows": 9987
    },
    "5": {
      "candidate_mae": 2743519.698514835,
      "control_mae": 2719663.807003374,
      "delta": 23855.891511460766,
      "relative_change_pct": 0.8771632526796048,
      "rows": 6310
    },
    "6": {
      "candidate_mae": 2537138.1705337,
      "control_mae": 2504077.440253743,
      "delta": 33060.730279956944,
      "relative_change_pct": 1.3202758728023538,
      "rows": 8284
    },
    "7": {
      "candidate_mae": 2505676.618051592,
      "control_mae": 2500338.7490206924,
      "delta": 5337.869030899368,
      "relative_change_pct": 0.2134858339890965,
      "rows": 9188
    },
    "8": {
      "candidate_mae": 2687738.0800081273,
      "control_mae": 2655677.968591626,
      "delta": 32060.11141650146,
      "relative_change_pct": 1.2072288807480585,
      "rows": 7187
    },
    "9": {
      "candidate_mae": 2652249.1671863776,
      "control_mae": 2628697.2045045653,
      "delta": 23551.962681812234,
      "relative_change_pct": 0.8959557092179854,
      "rows": 5780
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 2479279.067760944,
      "control_mae": 2456616.5723295277,
      "delta": 22662.495431416202,
      "relative_change_pct": 0.9225084486801338,
      "rows": 90424
    },
    "seen": {
      "candidate_mae": 1205390.2092519768,
      "control_mae": 1189341.620617399,
      "delta": 16048.588634577813,
      "relative_change_pct": 1.3493674446747126,
      "rows": 15087
    }
  },
  "used_for_release_gate": false,
  "used_for_research_gate": false,
  "weighted_mae": {
    "candidate_mae": 2012811.6181801,
    "control_mae": 1992570.9762837698,
    "delta": 20240.641896330286,
    "relative_change_pct": 1.0158053157072453
  }
}
```

所有城市／月份／新舊案／價格帶／季度／誤差尾部：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 466689.4190578535,
      "rows": 1
    },
    "南投縣": {
      "mae": 1087458.0098139634,
      "rows": 869
    },
    "嘉義市": {
      "mae": 2190833.1959067425,
      "rows": 731
    },
    "嘉義縣": {
      "mae": 1352890.1707510625,
      "rows": 1560
    },
    "基隆市": {
      "mae": 3248561.6701048017,
      "rows": 1577
    },
    "宜蘭縣": {
      "mae": 1315526.0840677598,
      "rows": 2330
    },
    "屏東縣": {
      "mae": 1398586.7036615438,
      "rows": 1240
    },
    "彰化縣": {
      "mae": 1436978.9912839201,
      "rows": 2607
    },
    "新北市": {
      "mae": 1913179.7880236295,
      "rows": 17191
    },
    "新竹市": {
      "mae": 4222157.355253105,
      "rows": 1586
    },
    "新竹縣": {
      "mae": 2836324.85237765,
      "rows": 5042
    },
    "桃園市": {
      "mae": 1820610.7831219176,
      "rows": 20723
    },
    "澎湖縣": {
      "mae": 1097403.2644649458,
      "rows": 40
    },
    "臺中市": {
      "mae": 2865338.318592805,
      "rows": 20837
    },
    "臺北市": {
      "mae": 6787576.259169572,
      "rows": 4712
    },
    "臺南市": {
      "mae": 1749326.5246632611,
      "rows": 8886
    },
    "臺東縣": {
      "mae": 1923632.7320556408,
      "rows": 235
    },
    "花蓮縣": {
      "mae": 779392.4920157843,
      "rows": 691
    },
    "苗栗縣": {
      "mae": 1644400.7505841802,
      "rows": 2607
    },
    "金門縣": {
      "mae": 1629342.8063585963,
      "rows": 35
    },
    "雲林縣": {
      "mae": 1057200.4749378536,
      "rows": 1873
    },
    "高雄市": {
      "mae": 1759679.3928310946,
      "rows": 10138
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1790251.1609133754,
      "rows": 33694
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 2374481.424163859,
      "rows": 29169
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 2573900.863785542,
      "rows": 25213
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 2747021.437013157,
      "rows": 17435
    }
  },
  "months": {
    "2021-08": {
      "mae": 1595853.5655748646,
      "rows": 5654
    },
    "2021-09": {
      "mae": 1645439.1998062409,
      "rows": 12352
    },
    "2021-10": {
      "mae": 1962482.4076290587,
      "rows": 11842
    },
    "2021-11": {
      "mae": 2063407.8960838038,
      "rows": 12524
    },
    "2021-12": {
      "mae": 2322456.672354941,
      "rows": 11482
    },
    "2022-01": {
      "mae": 2785991.023056833,
      "rows": 8017
    },
    "2022-02": {
      "mae": 2462397.0731912036,
      "rows": 6204
    },
    "2022-03": {
      "mae": 2486567.3129399284,
      "rows": 9626
    },
    "2022-04": {
      "mae": 2669954.6311241165,
      "rows": 7535
    },
    "2022-05": {
      "mae": 2670695.548672635,
      "rows": 6424
    },
    "2022-06": {
      "mae": 2686362.3374456074,
      "rows": 5629
    },
    "2022-07": {
      "mae": 2850681.641924873,
      "rows": 6566
    },
    "2022-08": {
      "mae": 2786764.9419800546,
      "rows": 1656
    }
  },
  "overall_mae": 2297125.9158779667,
  "price_bands": [
    {
      "absolute_error_share": 0.18944057683593465,
      "absolute_error_sum": 45915101436.44176,
      "band": "<10m",
      "mae": 1116286.6244394088,
      "mean_signed_error_prediction_minus_truth": 80590.56694215642,
      "rows": 41132,
      "underprediction_fraction": 0.531411066809297
    },
    {
      "absolute_error_share": 0.4012289702724752,
      "absolute_error_sum": 97246689051.4939,
      "band": "10-20m",
      "mae": 2034747.5373275145,
      "mean_signed_error_prediction_minus_truth": -1359926.3892535479,
      "rows": 47793,
      "underprediction_fraction": 0.7914966626911891
    },
    {
      "absolute_error_share": 0.28890622950166966,
      "absolute_error_sum": 70022795827.30261,
      "band": "20-50m",
      "mae": 4657938.922856556,
      "mean_signed_error_prediction_minus_truth": -3793457.7864058944,
      "rows": 15033,
      "underprediction_fraction": 0.8375573737776891
    },
    {
      "absolute_error_share": 0.0810810376242951,
      "absolute_error_sum": 19651777508.657166,
      "band": "50-100m",
      "mae": 14219810.06415135,
      "mean_signed_error_prediction_minus_truth": -10794424.715981131,
      "rows": 1382,
      "underprediction_fraction": 0.8024602026049205
    },
    {
      "absolute_error_share": 0.03934318576562534,
      "absolute_error_sum": 9535688686.304703,
      "band": ">=100m",
      "mae": 55764261.32341932,
      "mean_signed_error_prediction_minus_truth": -37925448.21308426,
      "rows": 171,
      "underprediction_fraction": 0.8888888888888888
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 1603042.2833962997,
      "rows": 9387
    },
    "1": {
      "mae": 1728919.515870967,
      "rows": 12463
    },
    "10": {
      "mae": 2702554.9595222087,
      "rows": 5992
    },
    "11": {
      "mae": 2880776.705288295,
      "rows": 6217
    },
    "2": {
      "mae": 2003161.1596820615,
      "rows": 11844
    },
    "3": {
      "mae": 2095207.6751498317,
      "rows": 12872
    },
    "4": {
      "mae": 2501264.260466442,
      "rows": 9987
    },
    "5": {
      "mae": 2743519.698514835,
      "rows": 6310
    },
    "6": {
      "mae": 2537138.1705337,
      "rows": 8284
    },
    "7": {
      "mae": 2505676.618051592,
      "rows": 9188
    },
    "8": {
      "mae": 2687738.0800081273,
      "rows": 7187
    },
    "9": {
      "mae": 2652249.1671863776,
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
      "mae": 2479279.067760944,
      "rows": 90424
    },
    "seen": {
      "mae": 1205390.2092519768,
      "rows": 15087
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 1446317.481764257,
      "0.9": 4745980.65463445,
      "0.95": 7061054.290470199,
      "0.99": 14870217.45142568
    },
    "absolute_error_sum": 242372052510.20013,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 674186206.2045565,
    "mean_signed_error_prediction_minus_truth": -1327921.3373685512,
    "top10_absolute_error_sum": 2274423523.833133,
    "top10_error_rows": 10,
    "top10_error_share": 0.009384017258909895,
    "top1pct_absolute_error_sum": 27986593879.975304,
    "top1pct_error_rows": 1056,
    "top1pct_error_share": 0.11546955843350586,
    "underprediction_fraction": 0.6969699841722664
  },
  "weighted_mae": 2012811.6181801
}
```

## offset_2023_02_11完整診斷

native fit實測53.406秒。

|層|總價MAE|
|---|---:|
|base|1,614,297.16|
|native|1,659,014.15|
|reference|1,478,395.83|
|spatial|1,456,950.95|
|median|1,349,012.83|

所有季度／至少500列群退步超5%（群重疊不能相加）：

```json
[
  {
    "candidate_mae": 2674211.014306638,
    "control_mae": 2407257.5414794474,
    "delta": 266953.47282719053,
    "group": "新竹市",
    "relative_change_pct": 11.08952690882949,
    "rows": 1852,
    "section": "cities"
  }
]
```

原生CBM及葉數證據：

```json
{
  "actual_grow_policy": "Depthwise",
  "actual_boosting_type": "Plain",
  "actual_model_depth": 8,
  "actual_trees": 2999,
  "actual_min_data_in_leaf": 1,
  "tree_leaf_counts_count": 2999,
  "minimum_tree_leaf_count": 10,
  "maximum_tree_leaf_count": 253,
  "tree_leaf_counts_distribution": {
    "10": 13,
    "100": 2,
    "101": 2,
    "102": 2,
    "103": 2,
    "104": 4,
    "105": 1,
    "106": 2,
    "107": 1,
    "108": 1,
    "109": 2,
    "11": 36,
    "111": 2,
    "115": 2,
    "117": 2,
    "119": 3,
    "12": 111,
    "121": 1,
    "126": 1,
    "128": 3,
    "13": 84,
    "130": 2,
    "133": 1,
    "134": 1,
    "136": 1,
    "138": 1,
    "139": 1,
    "14": 69,
    "140": 2,
    "142": 1,
    "149": 1,
    "15": 73,
    "151": 1,
    "16": 72,
    "166": 1,
    "17": 52,
    "173": 1,
    "176": 1,
    "18": 40,
    "182": 1,
    "186": 2,
    "187": 1,
    "19": 46,
    "190": 2,
    "191": 1,
    "192": 1,
    "194": 2,
    "20": 41,
    "203": 1,
    "209": 2,
    "21": 29,
    "215": 2,
    "217": 1,
    "218": 1,
    "22": 44,
    "223": 1,
    "225": 1,
    "226": 1,
    "228": 1,
    "229": 4,
    "23": 67,
    "230": 3,
    "231": 1,
    "232": 2,
    "233": 1,
    "234": 1,
    "235": 3,
    "236": 1,
    "237": 2,
    "238": 1,
    "239": 1,
    "24": 47,
    "240": 1,
    "241": 6,
    "242": 3,
    "244": 2,
    "246": 1,
    "25": 55,
    "253": 1,
    "26": 84,
    "27": 67,
    "28": 73,
    "29": 54,
    "30": 51,
    "31": 74,
    "32": 61,
    "33": 60,
    "34": 74,
    "35": 76,
    "36": 67,
    "37": 47,
    "38": 51,
    "39": 59,
    "40": 53,
    "41": 40,
    "42": 42,
    "43": 39,
    "44": 55,
    "45": 48,
    "46": 44,
    "47": 64,
    "48": 61,
    "49": 52,
    "50": 52,
    "51": 42,
    "52": 41,
    "53": 47,
    "54": 37,
    "55": 34,
    "56": 30,
    "57": 25,
    "58": 24,
    "59": 24,
    "60": 20,
    "61": 22,
    "62": 28,
    "63": 12,
    "64": 13,
    "65": 16,
    "66": 7,
    "67": 8,
    "68": 6,
    "69": 10,
    "70": 8,
    "71": 5,
    "72": 6,
    "73": 7,
    "74": 3,
    "75": 6,
    "76": 2,
    "77": 4,
    "78": 5,
    "79": 1,
    "80": 4,
    "81": 6,
    "82": 2,
    "83": 2,
    "84": 3,
    "85": 3,
    "86": 6,
    "87": 2,
    "88": 3,
    "89": 5,
    "90": 4,
    "91": 3,
    "92": 6,
    "93": 4,
    "94": 6,
    "95": 5,
    "96": 5,
    "97": 1,
    "98": 4,
    "99": 2
  },
  "runtime_parameters_not_recovered_from_native_model": {
    "allow_writing_files": false,
    "thread_count": 4,
    "verbose": 300
  }
}
```

原單價逆轉換與未投影總價：

```json
{
  "extra_price_cap_applied": false,
  "inverse_applications_per_prediction": 1,
  "native_TWD_max": 2980827047.591982,
  "native_TWD_min": 866904.3670017951,
  "native_zero_rows": 0,
  "nonnegative_projection_after_TWD_reconstruction": true,
  "query_y_used_for_diagnostics": false,
  "raw_negative_unit_rows": 0,
  "raw_unit_max": 52.387135078483595,
  "raw_unit_min": 3.9899500303244793,
  "raw_units": "unit",
  "unprojected_TWD_overall_mae": 1659014.1463303166
}
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+12,602.16|[+5,472.65, +19,301.34]|
|month|weighted|+14,975.31|[+5,864.07, +27,099.88]|
|project|overall|+12,602.16|[-2,191.26, +28,449.87]|
|project|weighted|+14,975.31|[-1,833.73, +33,214.22]|

完整條件抽樣及支援數：

```json
{
  "candidate": "R116",
  "caution": "Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R27 deeper trees, R84 extra trees, R92 shallow unit and R112 Ordered supplied contrary evidence; native improvement alone cannot establish complete-composition improvement.",
  "control": "v22",
  "difference": "R116 minus v22",
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
          "bootstrap_mean_delta": 12635.245683102523,
          "candidate_mae": 1349012.834450291,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0,
          "percentile_2_5": 5472.647875962075,
          "percentile_97_5": 19301.344017456653,
          "point_delta": 12602.1597859771,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 15504.71214124934,
          "candidate_mae": 1407865.6573956695,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0,
          "percentile_2_5": 5864.066940291319,
          "percentile_97_5": 27099.881674152875,
          "point_delta": 14975.312808926916,
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
          "bootstrap_mean_delta": 12695.436197062154,
          "candidate_mae": 1349012.834450291,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0465,
          "percentile_2_5": -2191.2600884607,
          "percentile_97_5": 28449.872212560094,
          "point_delta": 12602.1597859771,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 15117.329787556346,
          "candidate_mae": 1407865.6573956695,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.039,
          "percentile_2_5": -1833.7311455955612,
          "percentile_97_5": 33214.21714035927,
          "point_delta": 14975.312808926916,
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

v24次對照及反證（不以較弱對照替換v22）：

```json
{
  "automatic_adoption": false,
  "both_point_metrics_improve": false,
  "candidate": "R116",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 748690.2509114165,
      "control_mae": 748690.2509114165,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1030218.3670080245,
      "control_mae": 1015289.1847550352,
      "delta": 14929.182252989267,
      "relative_change_pct": 1.4704364507331296,
      "rows": 899
    },
    "嘉義市": {
      "candidate_mae": 1159843.8151848307,
      "control_mae": 1169659.4148716796,
      "delta": -9815.599686848931,
      "relative_change_pct": -0.8391844294200612,
      "rows": 604
    },
    "嘉義縣": {
      "candidate_mae": 690541.3848015437,
      "control_mae": 703921.1658472866,
      "delta": -13379.781045742915,
      "relative_change_pct": -1.9007499269663408,
      "rows": 1532
    },
    "基隆市": {
      "candidate_mae": 816839.2880068646,
      "control_mae": 824405.9509523374,
      "delta": -7566.662945472752,
      "relative_change_pct": -0.9178321598396874,
      "rows": 842
    },
    "宜蘭縣": {
      "candidate_mae": 793555.8244088602,
      "control_mae": 785285.7426309958,
      "delta": 8270.081777864369,
      "relative_change_pct": 1.0531302593316614,
      "rows": 2576
    },
    "屏東縣": {
      "candidate_mae": 902749.2206740001,
      "control_mae": 920486.7394754328,
      "delta": -17737.518801432685,
      "relative_change_pct": -1.9269716814759272,
      "rows": 691
    },
    "彰化縣": {
      "candidate_mae": 1008846.2349318272,
      "control_mae": 1011819.6970080789,
      "delta": -2973.4620762517443,
      "relative_change_pct": -0.29387272110279966,
      "rows": 2629
    },
    "新北市": {
      "candidate_mae": 1155669.8447824689,
      "control_mae": 1175179.3847952979,
      "delta": -19509.540012829006,
      "relative_change_pct": -1.6601329350435579,
      "rows": 20773
    },
    "新竹市": {
      "candidate_mae": 2674211.014306638,
      "control_mae": 2294808.7567983773,
      "delta": 379402.25750826066,
      "relative_change_pct": 16.53306648688177,
      "rows": 1852
    },
    "新竹縣": {
      "candidate_mae": 2227365.7237343397,
      "control_mae": 2237487.2731571877,
      "delta": -10121.549422848038,
      "relative_change_pct": -0.4523623237671475,
      "rows": 4704
    },
    "桃園市": {
      "candidate_mae": 1042587.645365084,
      "control_mae": 1066753.324847084,
      "delta": -24165.67948199995,
      "relative_change_pct": -2.2653484099019825,
      "rows": 21410
    },
    "澎湖縣": {
      "candidate_mae": 3795947.122518438,
      "control_mae": 3812375.149056576,
      "delta": -16428.026538138278,
      "relative_change_pct": -0.4309131681913207,
      "rows": 20
    },
    "臺中市": {
      "candidate_mae": 1634061.808679026,
      "control_mae": 1612776.5130701174,
      "delta": 21285.295608908636,
      "relative_change_pct": 1.3197920131159073,
      "rows": 20055
    },
    "臺北市": {
      "candidate_mae": 4136939.9424641095,
      "control_mae": 4030540.187754385,
      "delta": 106399.75470972434,
      "relative_change_pct": 2.63983857630272,
      "rows": 3514
    },
    "臺南市": {
      "candidate_mae": 1329130.792524088,
      "control_mae": 1290312.176866015,
      "delta": 38818.61565807299,
      "relative_change_pct": 3.008466970555753,
      "rows": 8483
    },
    "臺東縣": {
      "candidate_mae": 687705.0820271269,
      "control_mae": 694696.0897249224,
      "delta": -6991.007697795518,
      "relative_change_pct": -1.0063404416978559,
      "rows": 213
    },
    "花蓮縣": {
      "candidate_mae": 649593.2376776219,
      "control_mae": 651104.8750916787,
      "delta": -1511.6374140568078,
      "relative_change_pct": -0.2321649663341811,
      "rows": 578
    },
    "苗栗縣": {
      "candidate_mae": 1004351.8103258369,
      "control_mae": 1025581.2731701825,
      "delta": -21229.462844345602,
      "relative_change_pct": -2.069993222353119,
      "rows": 2973
    },
    "金門縣": {
      "candidate_mae": 912117.9234553765,
      "control_mae": 893641.4111528273,
      "delta": 18476.512302549207,
      "relative_change_pct": 2.0675532794204208,
      "rows": 65
    },
    "雲林縣": {
      "candidate_mae": 995256.2394789035,
      "control_mae": 996471.2687277427,
      "delta": -1215.0292488392442,
      "relative_change_pct": -0.12193319435998874,
      "rows": 1719
    },
    "高雄市": {
      "candidate_mae": 922706.050290238,
      "control_mae": 908831.0314082025,
      "delta": 13875.01888203551,
      "relative_change_pct": 1.526688504521753,
      "rows": 11036
    }
  },
  "control": "v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1010595.5680874448,
      "control_mae": 993826.2791985838,
      "delta": 16769.28888886096,
      "relative_change_pct": 1.6873460925569028,
      "rows": 22487
    },
    "Q2": {
      "candidate_mae": 1239322.7080604164,
      "control_mae": 1236271.5483622926,
      "delta": 3051.159698123811,
      "relative_change_pct": 0.24680335822382451,
      "rows": 27820
    },
    "Q3": {
      "candidate_mae": 1519490.8997093658,
      "control_mae": 1519816.2249750085,
      "delta": -325.3252656427212,
      "relative_change_pct": -0.021405566034674407,
      "rows": 27006
    },
    "Q4": {
      "candidate_mae": 1551907.9896780837,
      "control_mae": 1535913.1617296129,
      "delta": 15994.827948470833,
      "relative_change_pct": 1.0413888198248682,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "candidate_mae": 597639.9397583564,
      "control_mae": 596296.3427625896,
      "delta": 1343.5969957668567,
      "relative_change_pct": 0.22532370222867504,
      "rows": 2896
    },
    "2023-03": {
      "candidate_mae": 928890.7461655935,
      "control_mae": 925522.7748700764,
      "delta": 3367.9712955170544,
      "relative_change_pct": 0.3638993428324706,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1126828.8636983684,
      "control_mae": 1094744.1862448624,
      "delta": 32084.67745350604,
      "relative_change_pct": 2.9307922212915622,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1224013.5115126476,
      "control_mae": 1213455.5062566611,
      "delta": 10558.00525598647,
      "relative_change_pct": 0.8700776585172394,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1283934.9311979157,
      "control_mae": 1276086.4061312978,
      "delta": 7848.525066617876,
      "relative_change_pct": 0.6150465226263317,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1183767.0300092956,
      "control_mae": 1183156.183768008,
      "delta": 610.8462412876543,
      "relative_change_pct": 0.05162853811424007,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1304194.353904657,
      "control_mae": 1306980.5584263823,
      "delta": -2786.204521725187,
      "relative_change_pct": -0.21317872739283936,
      "rows": 7400
    },
    "2023-09": {
      "candidate_mae": 1376405.2225348132,
      "control_mae": 1365893.9517492831,
      "delta": 10511.270785530098,
      "relative_change_pct": 0.769552480415368,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1608907.6318663927,
      "control_mae": 1596675.0554627297,
      "delta": 12232.576403663028,
      "relative_change_pct": 0.7661281086474997,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1678734.0284951427,
      "control_mae": 1708449.945458693,
      "delta": -29715.916963550262,
      "relative_change_pct": -1.739349580743613,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1523640.7398725627,
      "control_mae": 1506056.8881887118,
      "delta": 17583.851683850866,
      "relative_change_pct": 1.1675423300243606,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1549714.5500939633,
      "control_mae": 1526897.035515871,
      "delta": 22817.514578092378,
      "relative_change_pct": 1.494371529143964,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1480066.4032096895,
      "control_mae": 1437162.9778553827,
      "delta": 42903.42535430682,
      "relative_change_pct": 2.985286012469496,
      "rows": 2566
    }
  },
  "overall_mae": {
    "candidate_mae": 1349012.8344502903,
    "control_mae": 1340328.1391452,
    "delta": 8684.695305090398,
    "relative_change_pct": 0.6479529192477522
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 702754.1093585587,
      "control_mae": 710632.9248020112,
      "delta": -7878.815443452564,
      "relative_change_pct": -1.1087039691620921,
      "rows": 25432
    },
    {
      "band": "10-20m",
      "candidate_mae": 1061303.1246469712,
      "control_mae": 1067452.758311166,
      "delta": -6149.633664194727,
      "relative_change_pct": -0.5761035901883059,
      "rows": 59481
    },
    {
      "band": "20-50m",
      "candidate_mae": 2298884.3272559415,
      "control_mae": 2241526.0750715095,
      "delta": 57358.252184432,
      "relative_change_pct": 2.55889292666837,
      "rows": 20803
    },
    {
      "band": "50-100m",
      "candidate_mae": 6954167.377937104,
      "control_mae": 6794509.1460085185,
      "delta": 159658.2319285851,
      "relative_change_pct": 2.3498125986389677,
      "rows": 1183
    },
    {
      "band": ">=100m",
      "candidate_mae": 27859526.373573545,
      "control_mae": 27434374.16274089,
      "delta": 425152.21083265543,
      "relative_change_pct": 1.5497062492136677,
      "rows": 270
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 2674211.014306638,
      "control_mae": 2294808.7567983773,
      "delta": 379402.25750826066,
      "group": "新竹市",
      "relative_change_pct": 16.53306648688177,
      "rows": 1852,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 651677.1693382852,
      "control_mae": 645679.7496816863,
      "delta": 5997.4196565989405,
      "relative_change_pct": 0.928853608860369,
      "rows": 5055
    },
    "1": {
      "candidate_mae": 1020313.910744743,
      "control_mae": 1023592.2057761417,
      "delta": -3278.2950313987676,
      "relative_change_pct": -0.32027354379012596,
      "rows": 7516
    },
    "10": {
      "candidate_mae": 1513972.4381046447,
      "control_mae": 1497637.3318924468,
      "delta": 16335.106212197803,
      "relative_change_pct": 1.0907250950773517,
      "rows": 9935
    },
    "11": {
      "candidate_mae": 1547478.8574612455,
      "control_mae": 1514286.76981743,
      "delta": 33192.08764381544,
      "relative_change_pct": 2.1919287882187097,
      "rows": 10413
    },
    "2": {
      "candidate_mae": 1191871.919954103,
      "control_mae": 1151164.1928843837,
      "delta": 40707.72706971923,
      "relative_change_pct": 3.536222488620064,
      "rows": 9057
    },
    "3": {
      "candidate_mae": 1196387.917381491,
      "control_mae": 1198704.169185322,
      "delta": -2316.2518038309645,
      "relative_change_pct": -0.1932296444255436,
      "rows": 9401
    },
    "4": {
      "candidate_mae": 1284496.4680978884,
      "control_mae": 1271555.585854942,
      "delta": 12940.88224294642,
      "relative_change_pct": 1.0177205296334333,
      "rows": 11490
    },
    "5": {
      "candidate_mae": 1202604.9343946637,
      "control_mae": 1206414.4495510287,
      "delta": -3809.5151563649997,
      "relative_change_pct": -0.31577167844621923,
      "rows": 6914
    },
    "6": {
      "candidate_mae": 1344635.000353003,
      "control_mae": 1342783.9583831823,
      "delta": 1851.0419698206242,
      "relative_change_pct": 0.13785106369973504,
      "rows": 7252
    },
    "7": {
      "candidate_mae": 1423871.0005262743,
      "control_mae": 1412213.0920092391,
      "delta": 11657.90851703519,
      "relative_change_pct": 0.8255063334987777,
      "rows": 9394
    },
    "8": {
      "candidate_mae": 1702025.7050026418,
      "control_mae": 1708489.12857141,
      "delta": -6463.423568768194,
      "relative_change_pct": -0.3783122444667075,
      "rows": 9933
    },
    "9": {
      "candidate_mae": 1600996.876792936,
      "control_mae": 1611253.9941320717,
      "delta": -10257.117339135846,
      "relative_change_pct": -0.6365922056044931,
      "rows": 10809
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1677222.5395024854,
      "control_mae": 1658897.3265622966,
      "delta": 18325.212940188823,
      "relative_change_pct": 1.1046622745582353,
      "rows": 59352
    },
    "seen": {
      "candidate_mae": 941628.3809241826,
      "control_mae": 944909.8065107913,
      "delta": -3281.425586608704,
      "relative_change_pct": -0.3472739476295433,
      "rows": 47817
    }
  },
  "used_for_release_gate": false,
  "weighted_mae": {
    "candidate_mae": 1407865.6573956683,
    "control_mae": 1397452.2756174281,
    "delta": 10413.381778240204,
    "relative_change_pct": 0.7451690451210092
  }
}
```

所有城市／月份／新舊案／價格帶／季度／誤差尾部：

```json
{
  "cities": {
    "__MISSING__": {
      "mae": 748690.2509114165,
      "rows": 1
    },
    "南投縣": {
      "mae": 1030218.3670080245,
      "rows": 899
    },
    "嘉義市": {
      "mae": 1159843.8151848307,
      "rows": 604
    },
    "嘉義縣": {
      "mae": 690541.3848015437,
      "rows": 1532
    },
    "基隆市": {
      "mae": 816839.2880068646,
      "rows": 842
    },
    "宜蘭縣": {
      "mae": 793555.8244088602,
      "rows": 2576
    },
    "屏東縣": {
      "mae": 902749.2206740001,
      "rows": 691
    },
    "彰化縣": {
      "mae": 1008846.2349318272,
      "rows": 2629
    },
    "新北市": {
      "mae": 1155669.8447824689,
      "rows": 20773
    },
    "新竹市": {
      "mae": 2674211.014306638,
      "rows": 1852
    },
    "新竹縣": {
      "mae": 2227365.7237343397,
      "rows": 4704
    },
    "桃園市": {
      "mae": 1042587.645365084,
      "rows": 21410
    },
    "澎湖縣": {
      "mae": 3795947.122518438,
      "rows": 20
    },
    "臺中市": {
      "mae": 1634061.808679026,
      "rows": 20055
    },
    "臺北市": {
      "mae": 4136939.9424641095,
      "rows": 3514
    },
    "臺南市": {
      "mae": 1329130.792524088,
      "rows": 8483
    },
    "臺東縣": {
      "mae": 687705.0820271269,
      "rows": 213
    },
    "花蓮縣": {
      "mae": 649593.2376776219,
      "rows": 578
    },
    "苗栗縣": {
      "mae": 1004351.8103258369,
      "rows": 2973
    },
    "金門縣": {
      "mae": 912117.9234553765,
      "rows": 65
    },
    "雲林縣": {
      "mae": 995256.2394789035,
      "rows": 1719
    },
    "高雄市": {
      "mae": 922706.050290238,
      "rows": 11036
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1010595.5680874448,
      "rows": 22487
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1239322.7080604164,
      "rows": 27820
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1519490.8997093658,
      "rows": 27006
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1551907.9896780837,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "mae": 597639.9397583564,
      "rows": 2896
    },
    "2023-03": {
      "mae": 928890.7461655935,
      "rows": 7500
    },
    "2023-04": {
      "mae": 1126828.8636983684,
      "rows": 8639
    },
    "2023-05": {
      "mae": 1224013.5115126476,
      "rows": 8750
    },
    "2023-06": {
      "mae": 1283934.9311979157,
      "rows": 13180
    },
    "2023-07": {
      "mae": 1183767.0300092956,
      "rows": 6144
    },
    "2023-08": {
      "mae": 1304194.353904657,
      "rows": 7400
    },
    "2023-09": {
      "mae": 1376405.2225348132,
      "rows": 8602
    },
    "2023-10": {
      "mae": 1608907.6318663927,
      "rows": 9855
    },
    "2023-11": {
      "mae": 1678734.0284951427,
      "rows": 10336
    },
    "2023-12": {
      "mae": 1523640.7398725627,
      "rows": 10416
    },
    "2024-01": {
      "mae": 1549714.5500939633,
      "rows": 10885
    },
    "2024-02": {
      "mae": 1480066.4032096895,
      "rows": 2566
    }
  },
  "overall_mae": 1349012.8344502903,
  "price_bands": [
    {
      "absolute_error_share": 0.12362282076203673,
      "absolute_error_sum": 17872442509.206863,
      "band": "<10m",
      "mae": 702754.1093585587,
      "mean_signed_error_prediction_minus_truth": 427996.3459475137,
      "rows": 25432,
      "underprediction_fraction": 0.2761088392576282
    },
    {
      "absolute_error_share": 0.4366489742918937,
      "absolute_error_sum": 63127371157.126495,
      "band": "10-20m",
      "mae": 1061303.1246469712,
      "mean_signed_error_prediction_minus_truth": 330509.9191725539,
      "rows": 59481,
      "underprediction_fraction": 0.3442780047410097
    },
    {
      "absolute_error_share": 0.3307941561755195,
      "absolute_error_sum": 47823690659.90535,
      "band": "20-50m",
      "mae": 2298884.3272559415,
      "mean_signed_error_prediction_minus_truth": -243424.82700186982,
      "rows": 20803,
      "underprediction_fraction": 0.46575013219247224
    },
    {
      "absolute_error_share": 0.05690423957811551,
      "absolute_error_sum": 8226780008.099594,
      "band": "50-100m",
      "mae": 6954167.377937104,
      "mean_signed_error_prediction_minus_truth": -1163325.9471626347,
      "rows": 1183,
      "underprediction_fraction": 0.46661031276415893
    },
    {
      "absolute_error_share": 0.05202980919243458,
      "absolute_error_sum": 7522072120.864858,
      "band": ">=100m",
      "mae": 27859526.373573545,
      "mean_signed_error_prediction_minus_truth": -10573310.946786324,
      "rows": 270,
      "underprediction_fraction": 0.7148148148148148
    }
  ],
  "relative_forecast_months": {
    "0": {
      "mae": 651677.1693382852,
      "rows": 5055
    },
    "1": {
      "mae": 1020313.910744743,
      "rows": 7516
    },
    "10": {
      "mae": 1513972.4381046447,
      "rows": 9935
    },
    "11": {
      "mae": 1547478.8574612455,
      "rows": 10413
    },
    "2": {
      "mae": 1191871.919954103,
      "rows": 9057
    },
    "3": {
      "mae": 1196387.917381491,
      "rows": 9401
    },
    "4": {
      "mae": 1284496.4680978884,
      "rows": 11490
    },
    "5": {
      "mae": 1202604.9343946637,
      "rows": 6914
    },
    "6": {
      "mae": 1344635.000353003,
      "rows": 7252
    },
    "7": {
      "mae": 1423871.0005262743,
      "rows": 9394
    },
    "8": {
      "mae": 1702025.7050026418,
      "rows": 9933
    },
    "9": {
      "mae": 1600996.876792936,
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
      "mae": 1677222.5395024854,
      "rows": 59352
    },
    "seen": {
      "mae": 941628.3809241826,
      "rows": 47817
    }
  },
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 793483.5640834402,
      "0.9": 2941017.494192319,
      "0.95": 4228005.112464451,
      "0.99": 8780375.620054333
    },
    "absolute_error_sum": 144572356455.20316,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 641967169.9038773,
    "mean_signed_error_prediction_minus_truth": 198274.57889123017,
    "top10_absolute_error_sum": 2345841228.906125,
    "top10_error_rows": 10,
    "top10_error_share": 0.016226070366591846,
    "top1pct_absolute_error_sum": 18149834943.017822,
    "top1pct_error_rows": 1072,
    "top1pct_error_share": 0.1255415308156901,
    "underprediction_fraction": 0.35396429937761853
  },
  "weighted_mae": 1407865.6573956683
}
```

## 執行與核驗界線

四fit命令601.844秒，另evaluate命令471.828秒。這是已測程序時間，不是整個專案工時或下一次ETA。18個可由native模型恢復的passed參數與實際Depthwise／Plain／GPU／2999樹逐項核對；allow_writing_files、thread_count、verbose是runtime宣告，不冒稱CBM能恢復。所有2999葉數保留SHA與分布，每樹1–256葉，未要求2的冪次。

[獨立核驗](../outputs/research_round116_20261008/postfit_review.json)保存完整實作範圍。CPU真載入四個CBM、全query raw單價及TWD重播、五層與分群／gates核對是保存模型核驗，不是新環境從零重訓。

```json
[
  "All four origins are reused and overlapping, not fresh blind validation or four independent votes.",
  "All four saved CBM models were reloaded on CPU and full-query raw unit predictions independently matched byte-for-byte, then original effective-area inverse-once and projection were independently reconstructed.",
  "Runtime-only allow_writing_files/thread_count/verbose are source/start metadata controls, not recoverable native model parameters.",
  "Only descriptive weight sum_squares and derived ESS permit at most 128 symmetric float64 ULP of reduction rounding; all weights dtype/shape/values SHA, predictions, metrics and gates remain exact.",
  "query_y_used_for_diagnostics=false applies to frozen raw-unit/projection counts only; unprojected_TWD_overall_mae was recomputed using postseal query labels and is not label-free.",
  "Composition and descriptive definitions reuse frozen helpers; independent MAE/group/gate arithmetic supplements replay.",
  "Bootstrap summaries use saved replicates without resampling, selection/refit/future-shift correction; not Private probability.",
  "No fullfit/release/CSV/Kaggle upload or v22 replacement is authorized by audit pass."
]
```

Four reused overlapping origins are not independent votes or fresh blind validation. Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap omits cumulative selection, retraining variation and future shift; not Private probability. R27 deeper trees, R84 extra trees, R92 shallow unit and R112 Ordered supplied contrary evidence; native improvement alone cannot establish complete-composition improvement.

四個已用且重疊的窗口有累積選型偏差，固定seed不代表重訓無變異；條件bootstrap不含重訓與選型不確定性，不是Private勝率。沒有外部成交答案、Public搜參數、test ID人工調價或Kaggle操作。

- [事前規約](round116_plan.md)
- [方法與工程參數說明](round116_method_review.md)
- [四起點driver說明](round116_driver_review.md)
- [單價trainer](../src/round116_depthwise_unit.py)
- [研究流程](../src/round116_research.py)
- [完整機器結果](../outputs/round116/results.json)
