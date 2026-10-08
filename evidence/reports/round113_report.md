# R113：2月起點的額外穩定性檢查

2026-10-08。fit截至2023-02-10，272,117列；query為2023-02-11至2024-02-10，107,169列。TWD MAE越低越好；本頁是本機離線分數，Public／Private沒有新增回報。

在R111／R112新分數出現前，已固定R107、R111、R112三者全部執行。原本三個8月切分的門檻保持，額外要求2月起點的整體與固定新舊案加權各不得退步超過0.25%，且任何季度或至少500列群不得退步超過5%。未達條件就不能發行新v28／v29，不會因其中一段較好就放寬條件。

|方法|整體MAE|差vs v22|固定加權MAE|差vs v22|額外穩定性|兩指標都改善|
|---|---:|---:|---:|---:|---|---|
|R97/v22|1,336,410.67|+0.00|1,392,890.34|+0.00|對照|—|
|R97/v24|1,340,328.14|+3,917.46|1,397,452.28|+4,561.93|對照|—|
|R107|1,336,107.62|-303.06|1,392,630.83|-259.51|True|True|
|R111|1,333,585.79|-2,824.88|1,389,632.46|-3,257.88|True|True|
|R112|1,341,197.54|+4,786.87|1,398,986.75|+6,096.41|False|False|

## 與原三窗門檻合看

|候選|原三窗探索門檻|本額外門檻|可進入獨立全量發行審查|
|---|---|---|---|
|R111|False|True|False|
|R112|False|False|False|

表中的True只代表數值門檻；獨立postfit、獨立final preflight、另立全量計畫、單次全量fit與CSV完整核對仍是必要步驟，不能把準備說成已發行。v22仍是原選定版本；v25是前輪R107已保存的探索檔，本輪只新增後續反證，不改其既有release或價格。R113本身不產生CSV，也不操作Kaggle。

## R107完整診斷

本次實測fit 34.047秒。

```json
{
  "components": {
    "no_quarter_or_large_group_above_5pct": true,
    "overall_regression_at_most_0_25pct": true,
    "weighted_regression_at_most_0_25pct": true
  },
  "passed": true
}
```

|層|MAE|
|---|---:|
|base|1,349,729.58|
|native|1,748,796.06|
|reference|1,468,491.54|
|spatial|1,449,238.18|
|median|1,336,107.62|

所有超5%季度／大型群（群可能重疊，不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-303.06|[-844.28, +163.84]|
|month|weighted|-259.51|[-877.80, +389.05]|
|project|overall|-303.06|[-2,558.33, +2,076.58]|
|project|weighted|-259.51|[-2,835.34, +2,450.07]|

完整paired summary與支援數：

```json
{
  "candidate": "R107",
  "caution": "The February origin was previously used for R97 and overlaps the August windows. This is not fresh blind validation or an independent fourth vote. Saved controls are not matched refits. Single-seed conditional bootstrap omits cumulative selection, refitting and future shift; it is not Private probability.",
  "control": "v22",
  "difference": "R107 minus v22",
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
          "bootstrap_mean_delta": -293.91836824224123,
          "candidate_mae": 1336107.617724857,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.8685,
          "percentile_2_5": -844.278284533415,
          "percentile_97_5": 163.83923110739562,
          "point_delta": -303.0569394568447,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -243.03825706459932,
          "candidate_mae": 1392630.8344366306,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.7605,
          "percentile_2_5": -877.7996136382687,
          "percentile_97_5": 389.05106942309067,
          "point_delta": -259.5101501119789,
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
          "bootstrap_mean_delta": -316.86844529151557,
          "candidate_mae": 1336107.617724857,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.623,
          "percentile_2_5": -2558.3283317768596,
          "percentile_97_5": 2076.5819156198227,
          "point_delta": -303.0569394568447,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -275.35072645709533,
          "candidate_mae": 1392630.8344366306,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.5975,
          "percentile_2_5": -2835.3392128462438,
          "percentile_97_5": 2450.0695477064055,
          "point_delta": -259.5101501119789,
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

與較強／較弱次對照v24的完整差異：

```json
{
  "automatic_adoption": false,
  "both_point_metrics_improve": true,
  "candidate": "R107",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 748690.2509114165,
      "control_mae": 748690.2509114165,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1020819.0017332863,
      "control_mae": 1015289.1847550352,
      "delta": 5529.816978251096,
      "relative_change_pct": 0.544654376436139,
      "rows": 899
    },
    "嘉義市": {
      "candidate_mae": 1151634.579084649,
      "control_mae": 1169659.4148716796,
      "delta": -18024.835787030635,
      "relative_change_pct": -1.5410328474984398,
      "rows": 604
    },
    "嘉義縣": {
      "candidate_mae": 705381.796217772,
      "control_mae": 703921.1658472866,
      "delta": 1460.6303704853635,
      "relative_change_pct": 0.20749914072085773,
      "rows": 1532
    },
    "基隆市": {
      "candidate_mae": 821027.7612473096,
      "control_mae": 824405.9509523374,
      "delta": -3378.189705027733,
      "relative_change_pct": -0.4097726006375033,
      "rows": 842
    },
    "宜蘭縣": {
      "candidate_mae": 782038.9076150676,
      "control_mae": 785285.7426309958,
      "delta": -3246.8350159282563,
      "relative_change_pct": -0.4134590556871905,
      "rows": 2576
    },
    "屏東縣": {
      "candidate_mae": 911014.3006452403,
      "control_mae": 920486.7394754328,
      "delta": -9472.438830192434,
      "relative_change_pct": -1.0290684725768662,
      "rows": 691
    },
    "彰化縣": {
      "candidate_mae": 1020316.5706109153,
      "control_mae": 1011819.6970080789,
      "delta": 8496.873602836393,
      "relative_change_pct": 0.8397616322316513,
      "rows": 2629
    },
    "新北市": {
      "candidate_mae": 1138008.290376746,
      "control_mae": 1175179.3847952979,
      "delta": -37171.09441855177,
      "relative_change_pct": -3.1630145065067263,
      "rows": 20773
    },
    "新竹市": {
      "candidate_mae": 2398326.395721676,
      "control_mae": 2294808.7567983773,
      "delta": 103517.63892329857,
      "relative_change_pct": 4.510948401108688,
      "rows": 1852
    },
    "新竹縣": {
      "candidate_mae": 2227909.1150439805,
      "control_mae": 2237487.2731571877,
      "delta": -9578.158113207202,
      "relative_change_pct": -0.4280765405066203,
      "rows": 4704
    },
    "桃園市": {
      "candidate_mae": 1065267.991430422,
      "control_mae": 1066753.324847084,
      "delta": -1485.333416661946,
      "relative_change_pct": -0.13923869577578907,
      "rows": 21410
    },
    "澎湖縣": {
      "candidate_mae": 3829890.6126322174,
      "control_mae": 3812375.149056576,
      "delta": 17515.46357564116,
      "relative_change_pct": 0.4594370409736722,
      "rows": 20
    },
    "臺中市": {
      "candidate_mae": 1614968.0069227377,
      "control_mae": 1612776.5130701174,
      "delta": 2191.493852620246,
      "relative_change_pct": 0.13588329411174702,
      "rows": 20055
    },
    "臺北市": {
      "candidate_mae": 4089691.2225895114,
      "control_mae": 4030540.187754385,
      "delta": 59151.03483512625,
      "relative_change_pct": 1.4675708981847975,
      "rows": 3514
    },
    "臺南市": {
      "candidate_mae": 1270497.013150316,
      "control_mae": 1290312.176866015,
      "delta": -19815.16371569899,
      "relative_change_pct": -1.5356875701062676,
      "rows": 8483
    },
    "臺東縣": {
      "candidate_mae": 700124.9858375425,
      "control_mae": 694696.0897249224,
      "delta": 5428.896112620016,
      "relative_change_pct": 0.7814778567084905,
      "rows": 213
    },
    "花蓮縣": {
      "candidate_mae": 654891.7729467346,
      "control_mae": 651104.8750916787,
      "delta": 3786.8978550558677,
      "relative_change_pct": 0.5816110430017367,
      "rows": 578
    },
    "苗栗縣": {
      "candidate_mae": 1055051.3150499007,
      "control_mae": 1025581.2731701825,
      "delta": 29470.041879718192,
      "relative_change_pct": 2.87349648932484,
      "rows": 2973
    },
    "金門縣": {
      "candidate_mae": 863219.6887724291,
      "control_mae": 893641.4111528273,
      "delta": -30421.72238039819,
      "relative_change_pct": -3.404242686241806,
      "rows": 65
    },
    "雲林縣": {
      "candidate_mae": 987757.4081819006,
      "control_mae": 996471.2687277427,
      "delta": -8713.860545842093,
      "relative_change_pct": -0.8744718306798374,
      "rows": 1719
    },
    "高雄市": {
      "candidate_mae": 913000.3888630108,
      "control_mae": 908831.0314082025,
      "delta": 4169.3574548083125,
      "relative_change_pct": 0.4587604637957879,
      "rows": 11036
    }
  },
  "control": "R97/v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 992346.7425776712,
      "control_mae": 993826.2791985838,
      "delta": -1479.5366209126078,
      "relative_change_pct": -0.14887276095231636,
      "rows": 22487
    },
    "Q2": {
      "candidate_mae": 1226760.7753274636,
      "control_mae": 1236271.5483622926,
      "delta": -9510.773034828948,
      "relative_change_pct": -0.7693110018933956,
      "rows": 27820
    },
    "Q3": {
      "candidate_mae": 1518894.1797347562,
      "control_mae": 1519816.2249750085,
      "delta": -922.0452402522787,
      "relative_change_pct": -0.060668206135741226,
      "rows": 27006
    },
    "Q4": {
      "candidate_mae": 1531574.0586844627,
      "control_mae": 1535913.1617296129,
      "delta": -4339.103045150172,
      "relative_change_pct": -0.28250965961277713,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "candidate_mae": 595773.4914272353,
      "control_mae": 596296.3427625896,
      "delta": -522.8513353542658,
      "relative_change_pct": -0.08768313636336265,
      "rows": 2896
    },
    "2023-03": {
      "candidate_mae": 920086.7748674467,
      "control_mae": 925522.7748700764,
      "delta": -5436.000002629706,
      "relative_change_pct": -0.587343731589188,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1094419.510414408,
      "control_mae": 1094744.1862448624,
      "delta": -324.67583045433275,
      "relative_change_pct": -0.029657689397558695,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1211870.823074571,
      "control_mae": 1213455.5062566611,
      "delta": -1584.6831820900552,
      "relative_change_pct": -0.13059260713881293,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1263928.113112186,
      "control_mae": 1276086.4061312978,
      "delta": -12158.293019111734,
      "relative_change_pct": -0.9527797616755393,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1172099.8557170806,
      "control_mae": 1183156.183768008,
      "delta": -11056.328050927259,
      "relative_change_pct": -0.9344774766519898,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1300700.4015810485,
      "control_mae": 1306980.5584263823,
      "delta": -6280.156845333753,
      "relative_change_pct": -0.48050881896017833,
      "rows": 7400
    },
    "2023-09": {
      "candidate_mae": 1364911.3979784846,
      "control_mae": 1365893.9517492831,
      "delta": -982.5537707984913,
      "relative_change_pct": -0.07193485039890155,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1598543.2623472977,
      "control_mae": 1596675.0554627297,
      "delta": 1868.2068845680915,
      "relative_change_pct": 0.11700607948851995,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1695528.4042164353,
      "control_mae": 1708449.945458693,
      "delta": -12921.541242257692,
      "relative_change_pct": -0.756331274241017,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1499254.01160144,
      "control_mae": 1506056.8881887118,
      "delta": -6802.876587271923,
      "relative_change_pct": -0.4517011701632023,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1531734.372966868,
      "control_mae": 1526897.035515871,
      "delta": 4837.337450997205,
      "relative_change_pct": 0.3168083595998916,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1446164.4023573238,
      "control_mae": 1437162.9778553827,
      "delta": 9001.424501941074,
      "relative_change_pct": 0.6263328961739272,
      "rows": 2566
    }
  },
  "overall_mae": {
    "candidate_mae": 1336107.6177248484,
    "control_mae": 1340328.1391452,
    "delta": -4220.521420351462,
    "relative_change_pct": -0.31488717554218615
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 704244.7930395937,
      "control_mae": 710632.9248020112,
      "delta": -6388.131762417499,
      "relative_change_pct": -0.8989355178269133,
      "rows": 25432
    },
    {
      "band": "10-20m",
      "candidate_mae": 1061640.414176624,
      "control_mae": 1067452.758311166,
      "delta": -5812.344134541927,
      "relative_change_pct": -0.5445059829849266,
      "rows": 59481
    },
    {
      "band": "20-50m",
      "candidate_mae": 2248631.7442572336,
      "control_mae": 2241526.0750715095,
      "delta": 7105.669185724109,
      "relative_change_pct": 0.31700140653047826,
      "rows": 20803
    },
    {
      "band": "50-100m",
      "candidate_mae": 6815707.116060606,
      "control_mae": 6794509.1460085185,
      "delta": 21197.97005208768,
      "relative_change_pct": 0.3119867763301279,
      "rows": 1183
    },
    {
      "band": ">=100m",
      "candidate_mae": 27000972.36166161,
      "control_mae": 27434374.16274089,
      "delta": -433401.8010792807,
      "relative_change_pct": -1.5797765187145816,
      "rows": 270
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 651085.9002468603,
      "control_mae": 645679.7496816863,
      "delta": 5406.150565174059,
      "relative_change_pct": 0.837280488948157,
      "rows": 5055
    },
    "1": {
      "candidate_mae": 1011498.148383031,
      "control_mae": 1023592.2057761417,
      "delta": -12094.057393110706,
      "relative_change_pct": -1.1815308210500053,
      "rows": 7516
    },
    "10": {
      "candidate_mae": 1492573.8074843227,
      "control_mae": 1497637.3318924468,
      "delta": -5063.524408124154,
      "relative_change_pct": -0.33810084059041023,
      "rows": 9935
    },
    "11": {
      "candidate_mae": 1521769.2796012457,
      "control_mae": 1514286.76981743,
      "delta": 7482.5097838155925,
      "relative_change_pct": 0.49412766016028264,
      "rows": 10413
    },
    "2": {
      "candidate_mae": 1154708.3028686382,
      "control_mae": 1151164.1928843837,
      "delta": 3544.1099842544645,
      "relative_change_pct": 0.30787180544369264,
      "rows": 9057
    },
    "3": {
      "candidate_mae": 1182196.697055164,
      "control_mae": 1198704.169185322,
      "delta": -16507.472130157985,
      "relative_change_pct": -1.3771097619003858,
      "rows": 9401
    },
    "4": {
      "candidate_mae": 1267078.9322702272,
      "control_mae": 1271555.585854942,
      "delta": -4476.653584714746,
      "relative_change_pct": -0.35206117880445054,
      "rows": 11490
    },
    "5": {
      "candidate_mae": 1197710.9787636457,
      "control_mae": 1206414.4495510287,
      "delta": -8703.470787382917,
      "relative_change_pct": -0.7214329031471685,
      "rows": 6914
    },
    "6": {
      "candidate_mae": 1338319.333273016,
      "control_mae": 1342783.9583831823,
      "delta": -4464.62511016638,
      "relative_change_pct": -0.332490203080929,
      "rows": 7252
    },
    "7": {
      "candidate_mae": 1410726.3332932042,
      "control_mae": 1412213.0920092391,
      "delta": -1486.7587160349358,
      "relative_change_pct": -0.10527863850345957,
      "rows": 9394
    },
    "8": {
      "candidate_mae": 1711680.5095630526,
      "control_mae": 1708489.12857141,
      "delta": 3191.380991642596,
      "relative_change_pct": 0.18679551062236716,
      "rows": 9933
    },
    "9": {
      "candidate_mae": 1595803.6192727669,
      "control_mae": 1611253.9941320717,
      "delta": -15450.374859304866,
      "relative_change_pct": -0.9589037430208179,
      "rows": 10809
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1651325.6031525012,
      "control_mae": 1658897.3265622966,
      "delta": -7571.723409795435,
      "relative_change_pct": -0.45643110568429107,
      "rows": 59352
    },
    "seen": {
      "candidate_mae": 944848.9049009144,
      "control_mae": 944909.8065107913,
      "delta": -60.9016098768916,
      "relative_change_pct": -0.006445229952875516,
      "rows": 47817
    }
  },
  "used_for_release_gate": false,
  "weighted_mae": {
    "candidate_mae": 1392630.8344366215,
    "control_mae": 1397452.2756174281,
    "delta": -4821.441180806607,
    "relative_change_pct": -0.345016517911238
  }
}
```

真價帶與誤差尾部（僅診斷，不能拿未知test真價路由）：

```json
{
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 1651325.6031525012,
      "rows": 59352
    },
    "seen": {
      "mae": 944848.9049009144,
      "rows": 47817
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 992346.7425776712,
      "rows": 22487
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1226760.7753274636,
      "rows": 27820
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1518894.1797347562,
      "rows": 27006
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1531574.0586844627,
      "rows": 29856
    }
  },
  "price_bands": [
    {
      "absolute_error_share": 0.12508163259878866,
      "absolute_error_sum": 17910353576.582947,
      "band": "<10m",
      "mae": 704244.7930395937,
      "mean_signed_error_prediction_minus_truth": 407644.66167331336,
      "rows": 25432,
      "underprediction_fraction": 0.2898710286253539
    },
    {
      "absolute_error_share": 0.4410065965355087,
      "absolute_error_sum": 63147433475.63977,
      "band": "10-20m",
      "mae": 1061640.414176624,
      "mean_signed_error_prediction_minus_truth": 308628.88392715534,
      "rows": 59481,
      "underprediction_fraction": 0.35870277903868464
    },
    {
      "absolute_error_share": 0.3266883805515928,
      "absolute_error_sum": 46778286175.783226,
      "band": "20-50m",
      "mae": 2248631.7442572336,
      "mean_signed_error_prediction_minus_truth": -134496.79302523003,
      "rows": 20803,
      "underprediction_fraction": 0.4557035043022641
    },
    {
      "absolute_error_share": 0.05630993757942325,
      "absolute_error_sum": 8062981518.299697,
      "band": "50-100m",
      "mae": 6815707.116060606,
      "mean_signed_error_prediction_minus_truth": -904921.6667955731,
      "rows": 1183,
      "underprediction_fraction": 0.44716821639898563
    },
    {
      "absolute_error_share": 0.0509134527346865,
      "absolute_error_sum": 7290262537.648635,
      "band": ">=100m",
      "mae": 27000972.36166161,
      "mean_signed_error_prediction_minus_truth": -10061086.941145882,
      "rows": 270,
      "underprediction_fraction": 0.7111111111111111
    }
  ],
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 791994.874374425,
      "0.9": 2907424.9455793584,
      "0.95": 4194436.814756115,
      "0.99": 8682178.733021962
    },
    "absolute_error_sum": 143189317283.95428,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 635540359.2647953,
    "mean_signed_error_prediction_minus_truth": 206587.92271379245,
    "top10_absolute_error_sum": 2317841349.779063,
    "top10_error_rows": 10,
    "top10_error_share": 0.01618725051382586,
    "top1pct_absolute_error_sum": 17826748410.262825,
    "top1pct_error_rows": 1072,
    "top1pct_error_share": 0.12449775408113131,
    "underprediction_fraction": 0.36306207951926395
  }
}
```

## R111完整診斷

本次實測fit 34.703秒。

```json
{
  "components": {
    "no_quarter_or_large_group_above_5pct": true,
    "overall_regression_at_most_0_25pct": true,
    "weighted_regression_at_most_0_25pct": true
  },
  "passed": true
}
```

|層|MAE|
|---|---:|
|base|1,357,615.53|
|native|1,776,152.84|
|reference|1,466,810.46|
|spatial|1,448,998.34|
|median|1,333,585.79|

所有超5%季度／大型群（群可能重疊，不能相加）：

```json
[]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|-2,824.88|[-4,393.02, -1,395.22]|
|month|weighted|-3,257.88|[-5,252.43, -1,544.77]|
|project|overall|-2,824.88|[-9,757.79, +3,678.84]|
|project|weighted|-3,257.88|[-11,130.26, +4,152.65]|

完整paired summary與支援數：

```json
{
  "candidate": "R111",
  "caution": "The February origin was previously used for R97 and overlaps the August windows. This is not fresh blind validation or an independent fourth vote. Saved controls are not matched refits. Single-seed conditional bootstrap omits cumulative selection, refitting and future shift; it is not Private probability.",
  "control": "v22",
  "difference": "R111 minus v22",
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
          "bootstrap_mean_delta": -2800.0634897741543,
          "candidate_mae": 1333585.792926543,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 1.0,
          "percentile_2_5": -4393.020525138365,
          "percentile_97_5": -1395.2161305992313,
          "point_delta": -2824.8817377709784,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -3252.781504542095,
          "candidate_mae": 1389632.4624466202,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 1.0,
          "percentile_2_5": -5252.432978054567,
          "percentile_97_5": -1544.7699205567362,
          "point_delta": -3257.882140122354,
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
          "bootstrap_mean_delta": -2838.0739804916325,
          "candidate_mae": 1333585.792926543,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.793,
          "percentile_2_5": -9757.789830077247,
          "percentile_97_5": 3678.843776532395,
          "point_delta": -2824.8817377709784,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": -3263.735746985007,
          "candidate_mae": 1389632.4624466202,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.796,
          "percentile_2_5": -11130.25615676252,
          "percentile_97_5": 4152.6515386235,
          "point_delta": -3257.882140122354,
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

與較強／較弱次對照v24的完整差異：

```json
{
  "automatic_adoption": false,
  "both_point_metrics_improve": true,
  "candidate": "R111",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 748690.2509114165,
      "control_mae": 748690.2509114165,
      "delta": 0.0,
      "relative_change_pct": 0.0,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 1020498.9508529847,
      "control_mae": 1015289.1847550352,
      "delta": 5209.7660979494685,
      "relative_change_pct": 0.5131312512903857,
      "rows": 899
    },
    "嘉義市": {
      "candidate_mae": 1149952.7956896366,
      "control_mae": 1169659.4148716796,
      "delta": -19706.61918204301,
      "relative_change_pct": -1.6848168733122175,
      "rows": 604
    },
    "嘉義縣": {
      "candidate_mae": 706548.9656447449,
      "control_mae": 703921.1658472866,
      "delta": 2627.7997974583413,
      "relative_change_pct": 0.37330881992947973,
      "rows": 1532
    },
    "基隆市": {
      "candidate_mae": 820978.6853600249,
      "control_mae": 824405.9509523374,
      "delta": -3427.2655923124403,
      "relative_change_pct": -0.41572547946231236,
      "rows": 842
    },
    "宜蘭縣": {
      "candidate_mae": 785947.6718394747,
      "control_mae": 785285.7426309958,
      "delta": 661.9292084788904,
      "relative_change_pct": 0.08429150977084905,
      "rows": 2576
    },
    "屏東縣": {
      "candidate_mae": 893833.1853122783,
      "control_mae": 920486.7394754328,
      "delta": -26653.554163154447,
      "relative_change_pct": -2.895593496365171,
      "rows": 691
    },
    "彰化縣": {
      "candidate_mae": 999270.1887317043,
      "control_mae": 1011819.6970080789,
      "delta": -12549.508276374661,
      "relative_change_pct": -1.2402909642383113,
      "rows": 2629
    },
    "新北市": {
      "candidate_mae": 1141113.089789558,
      "control_mae": 1175179.3847952979,
      "delta": -34066.2950057399,
      "relative_change_pct": -2.8988165931513374,
      "rows": 20773
    },
    "新竹市": {
      "candidate_mae": 2368601.164015043,
      "control_mae": 2294808.7567983773,
      "delta": 73792.4072166658,
      "relative_change_pct": 3.2156233933679914,
      "rows": 1852
    },
    "新竹縣": {
      "candidate_mae": 2211360.1642839657,
      "control_mae": 2237487.2731571877,
      "delta": -26127.108873222023,
      "relative_change_pct": -1.1676986585204385,
      "rows": 4704
    },
    "桃園市": {
      "candidate_mae": 1059216.5089617292,
      "control_mae": 1066753.324847084,
      "delta": -7536.815885354765,
      "relative_change_pct": -0.7065190901968967,
      "rows": 21410
    },
    "澎湖縣": {
      "candidate_mae": 3737360.408031527,
      "control_mae": 3812375.149056576,
      "delta": -75014.74102504924,
      "relative_change_pct": -1.9676642012424368,
      "rows": 20
    },
    "臺中市": {
      "candidate_mae": 1612033.9652430883,
      "control_mae": 1612776.5130701174,
      "delta": -742.5478270291351,
      "relative_change_pct": -0.046041582389838034,
      "rows": 20055
    },
    "臺北市": {
      "candidate_mae": 4117444.4230190795,
      "control_mae": 4030540.187754385,
      "delta": 86904.23526469432,
      "relative_change_pct": 2.156143623842962,
      "rows": 3514
    },
    "臺南市": {
      "candidate_mae": 1265818.0026421098,
      "control_mae": 1290312.176866015,
      "delta": -24494.174223905196,
      "relative_change_pct": -1.8983138083218023,
      "rows": 8483
    },
    "臺東縣": {
      "candidate_mae": 683270.8311585009,
      "control_mae": 694696.0897249224,
      "delta": -11425.258566421573,
      "relative_change_pct": -1.6446412662183851,
      "rows": 213
    },
    "花蓮縣": {
      "candidate_mae": 654826.5951387604,
      "control_mae": 651104.8750916787,
      "delta": 3721.7200470817043,
      "relative_change_pct": 0.5716007035821484,
      "rows": 578
    },
    "苗栗縣": {
      "candidate_mae": 1066883.7701558182,
      "control_mae": 1025581.2731701825,
      "delta": 41302.49698563572,
      "relative_change_pct": 4.027228077007027,
      "rows": 2973
    },
    "金門縣": {
      "candidate_mae": 837030.1965524926,
      "control_mae": 893641.4111528273,
      "delta": -56611.21460033476,
      "relative_change_pct": -6.3348915900511376,
      "rows": 65
    },
    "雲林縣": {
      "candidate_mae": 986096.1349632103,
      "control_mae": 996471.2687277427,
      "delta": -10375.133764532395,
      "relative_change_pct": -1.0411874471583087,
      "rows": 1719
    },
    "高雄市": {
      "candidate_mae": 909399.5299063923,
      "control_mae": 908831.0314082025,
      "delta": 568.4984981898451,
      "relative_change_pct": 0.06255271646138405,
      "rows": 11036
    }
  },
  "control": "R97/v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 990181.4766437166,
      "control_mae": 993826.2791985838,
      "delta": -3644.8025548672304,
      "relative_change_pct": -0.36674443322291495,
      "rows": 22487
    },
    "Q2": {
      "candidate_mae": 1221732.183841159,
      "control_mae": 1236271.5483622926,
      "delta": -14539.36452113348,
      "relative_change_pct": -1.1760656095656326,
      "rows": 27820
    },
    "Q3": {
      "candidate_mae": 1517915.9511502762,
      "control_mae": 1519816.2249750085,
      "delta": -1900.273824732285,
      "relative_change_pct": -0.1250331318685279,
      "rows": 27006
    },
    "Q4": {
      "candidate_mae": 1529723.2531361142,
      "control_mae": 1535913.1617296129,
      "delta": -6189.908593498636,
      "relative_change_pct": -0.4030116251187076,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "candidate_mae": 591802.9945153826,
      "control_mae": 596296.3427625896,
      "delta": -4493.348247207003,
      "relative_change_pct": -0.7535428150355086,
      "rows": 2896
    },
    "2023-03": {
      "candidate_mae": 921099.9218659087,
      "control_mae": 925522.7748700764,
      "delta": -4422.853004167788,
      "relative_change_pct": -0.47787619324534303,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1089675.027611344,
      "control_mae": 1094744.1862448624,
      "delta": -5069.158633518498,
      "relative_change_pct": -0.4630450380290647,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1209938.465403768,
      "control_mae": 1213455.5062566611,
      "delta": -3517.040852893144,
      "relative_change_pct": -0.28983682011899375,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1254592.9246711398,
      "control_mae": 1276086.4061312978,
      "delta": -21493.481460158015,
      "relative_change_pct": -1.6843280640626563,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1171031.0884617632,
      "control_mae": 1183156.183768008,
      "delta": -12125.095306244679,
      "relative_change_pct": -1.0248093592876117,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1301780.5913936803,
      "control_mae": 1306980.5584263823,
      "delta": -5199.967032701941,
      "relative_change_pct": -0.3978610851689144,
      "rows": 7400
    },
    "2023-09": {
      "candidate_mae": 1362152.7494132803,
      "control_mae": 1365893.9517492831,
      "delta": -3741.2023360028397,
      "relative_change_pct": -0.2739013765462194,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1602712.8410564405,
      "control_mae": 1596675.0554627297,
      "delta": 6037.78559371084,
      "relative_change_pct": 0.37814742411448515,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1687491.0041749943,
      "control_mae": 1708449.945458693,
      "delta": -20958.94128369866,
      "relative_change_pct": -1.2267811146244323,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1498848.3371997913,
      "control_mae": 1506056.8881887118,
      "delta": -7208.550988920499,
      "relative_change_pct": -0.4786373639305219,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1529726.28645665,
      "control_mae": 1526897.035515871,
      "delta": 2829.250940778991,
      "relative_change_pct": 0.18529415376218297,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1448090.3205992475,
      "control_mae": 1437162.9778553827,
      "delta": 10927.342743864749,
      "relative_change_pct": 0.7603412356315467,
      "rows": 2566
    }
  },
  "overall_mae": {
    "candidate_mae": 1333585.7929265413,
    "control_mae": 1340328.1391452,
    "delta": -6742.346218658611,
    "relative_change_pct": -0.5030369818959834
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 702051.6472218913,
      "control_mae": 710632.9248020112,
      "delta": -8581.277580119902,
      "relative_change_pct": -1.2075541789047735,
      "rows": 25432
    },
    {
      "band": "10-20m",
      "candidate_mae": 1057733.4656060957,
      "control_mae": 1067452.758311166,
      "delta": -9719.292705070227,
      "relative_change_pct": -0.910512678841851,
      "rows": 59481
    },
    {
      "band": "20-50m",
      "candidate_mae": 2248775.1073344587,
      "control_mae": 2241526.0750715095,
      "delta": 7249.032262949273,
      "relative_change_pct": 0.3233971865671031,
      "rows": 20803
    },
    {
      "band": "50-100m",
      "candidate_mae": 6833152.627035976,
      "control_mae": 6794509.1460085185,
      "delta": 38643.48102745786,
      "relative_change_pct": 0.5687457356674432,
      "rows": 1183
    },
    {
      "band": ">=100m",
      "candidate_mae": 26979799.876366135,
      "control_mae": 27434374.16274089,
      "delta": -454574.2863747552,
      "relative_change_pct": -1.6569515443589764,
      "rows": 270
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 647643.7235817157,
      "control_mae": 645679.7496816863,
      "delta": 1963.9739000294358,
      "relative_change_pct": 0.3041715186201295,
      "rows": 5055
    },
    "1": {
      "candidate_mae": 1012116.2184999982,
      "control_mae": 1023592.2057761417,
      "delta": -11475.98727614351,
      "relative_change_pct": -1.1211483646890228,
      "rows": 7516
    },
    "10": {
      "candidate_mae": 1492164.196171646,
      "control_mae": 1497637.3318924468,
      "delta": -5473.135720800841,
      "relative_change_pct": -0.365451341539735,
      "rows": 9935
    },
    "11": {
      "candidate_mae": 1521436.3822893735,
      "control_mae": 1514286.76981743,
      "delta": 7149.612471943488,
      "relative_change_pct": 0.4721438907377815,
      "rows": 10413
    },
    "2": {
      "candidate_mae": 1151314.0967134517,
      "control_mae": 1151164.1928843837,
      "delta": 149.90382906794548,
      "relative_change_pct": 0.013021932926209507,
      "rows": 9057
    },
    "3": {
      "candidate_mae": 1176011.6820922096,
      "control_mae": 1198704.169185322,
      "delta": -22692.48709311243,
      "relative_change_pct": -1.8930848558351956,
      "rows": 9401
    },
    "4": {
      "candidate_mae": 1259840.2378481585,
      "control_mae": 1271555.585854942,
      "delta": -11715.348006783519,
      "relative_change_pct": -0.9213398247868652,
      "rows": 11490
    },
    "5": {
      "candidate_mae": 1197889.7560268457,
      "control_mae": 1206414.4495510287,
      "delta": -8524.693524183007,
      "relative_change_pct": -0.70661401041371,
      "rows": 6914
    },
    "6": {
      "candidate_mae": 1339263.305152042,
      "control_mae": 1342783.9583831823,
      "delta": -3520.653231140226,
      "relative_change_pct": -0.2621905935917919,
      "rows": 7252
    },
    "7": {
      "candidate_mae": 1409535.834642437,
      "control_mae": 1412213.0920092391,
      "delta": -2677.2573668020777,
      "relative_change_pct": -0.18957885194174098,
      "rows": 9394
    },
    "8": {
      "candidate_mae": 1709346.4046581208,
      "control_mae": 1708489.12857141,
      "delta": 857.2760867108591,
      "relative_change_pct": 0.050177438789305556,
      "rows": 9933
    },
    "9": {
      "candidate_mae": 1591027.5069902914,
      "control_mae": 1611253.9941320717,
      "delta": -20226.487141780322,
      "relative_change_pct": -1.2553258030975836,
      "rows": 10809
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1646146.1757889776,
      "control_mae": 1658897.3265622966,
      "delta": -12751.15077331895,
      "relative_change_pct": -0.768652198610925,
      "rows": 59352
    },
    "seen": {
      "candidate_mae": 945625.781975387,
      "control_mae": 944909.8065107913,
      "delta": 715.9754645957146,
      "relative_change_pct": 0.0757718313073236,
      "rows": 47817
    }
  },
  "used_for_release_gate": false,
  "weighted_mae": {
    "candidate_mae": 1389632.4624466188,
    "control_mae": 1397452.2756174281,
    "delta": -7819.813170809299,
    "relative_change_pct": -0.559576402518241
  }
}
```

真價帶與誤差尾部（僅診斷，不能拿未知test真價路由）：

```json
{
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 1646146.1757889776,
      "rows": 59352
    },
    "seen": {
      "mae": 945625.781975387,
      "rows": 47817
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 990181.4766437166,
      "rows": 22487
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1221732.183841159,
      "rows": 27820
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1517915.9511502762,
      "rows": 27006
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1529723.2531361142,
      "rows": 29856
    }
  },
  "price_bands": [
    {
      "absolute_error_share": 0.12492789983071044,
      "absolute_error_sum": 17854577492.14714,
      "band": "<10m",
      "mae": 702051.6472218913,
      "mean_signed_error_prediction_minus_truth": 408321.70733084856,
      "rows": 25432,
      "underprediction_fraction": 0.28861277131173324
    },
    {
      "absolute_error_share": 0.44021452490706675,
      "absolute_error_sum": 62915044267.71619,
      "band": "10-20m",
      "mae": 1057733.4656060957,
      "mean_signed_error_prediction_minus_truth": 305212.23255150917,
      "rows": 59481,
      "underprediction_fraction": 0.35838334930482
    },
    {
      "absolute_error_share": 0.3273270193552714,
      "absolute_error_sum": 46781268557.878746,
      "band": "20-50m",
      "mae": 2248775.1073344587,
      "mean_signed_error_prediction_minus_truth": -149764.44421607454,
      "rows": 20803,
      "underprediction_fraction": 0.4560399942316012
    },
    {
      "absolute_error_share": 0.056560823958367014,
      "absolute_error_sum": 8083619557.78356,
      "band": "50-100m",
      "mae": 6833152.627035976,
      "mean_signed_error_prediction_minus_truth": -910479.4841366666,
      "rows": 1183,
      "underprediction_fraction": 0.4454775993237532
    },
    {
      "absolute_error_share": 0.0509697319485843,
      "absolute_error_sum": 7284545966.618856,
      "band": ">=100m",
      "mae": 26979799.876366135,
      "mean_signed_error_prediction_minus_truth": -10729804.000482466,
      "rows": 270,
      "underprediction_fraction": 0.7111111111111111
    }
  ],
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 788083.596029669,
      "0.9": 2901017.248193915,
      "0.95": 4164954.2255887873,
      "0.99": 8672176.611377303
    },
    "absolute_error_sum": 142919055842.1445,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 605782998.4127998,
    "mean_signed_error_prediction_minus_truth": 200142.5078366273,
    "top10_absolute_error_sum": 2231650476.3771667,
    "top10_error_rows": 10,
    "top10_error_share": 0.015614786028547841,
    "top1pct_absolute_error_sum": 17904655672.81623,
    "top1pct_error_rows": 1072,
    "top1pct_error_share": 0.12527829523721523,
    "underprediction_fraction": 0.3626328509177094
  }
}
```

## R112完整診斷

本次實測fit 310.016秒。

```json
{
  "components": {
    "no_quarter_or_large_group_above_5pct": false,
    "overall_regression_at_most_0_25pct": false,
    "weighted_regression_at_most_0_25pct": false
  },
  "passed": false
}
```

|層|MAE|
|---|---:|
|base|1,556,283.51|
|native|1,547,070.84|
|reference|1,402,861.37|
|spatial|1,385,023.85|
|median|1,341,197.54|

所有超5%季度／大型群（群可能重疊，不能相加）：

```json
[
  {
    "candidate_mae": 2628696.4722428187,
    "control_mae": 2407257.5414794474,
    "delta": 221438.9307633713,
    "group": "新竹市",
    "relative_change_pct": 9.198805152658482,
    "rows": 1852,
    "section": "cities"
  }
]
```

|條件抽樣|指標|點差vs v22|95%區間|
|---|---|---:|---|
|month|overall|+4,786.87|[-906.76, +10,192.01]|
|month|weighted|+6,096.41|[-728.82, +15,835.10]|
|project|overall|+4,786.87|[-8,052.73, +19,029.70]|
|project|weighted|+6,096.41|[-8,399.53, +22,404.34]|

完整paired summary與支援數：

```json
{
  "candidate": "R112",
  "caution": "The February origin was previously used for R97 and overlaps the August windows. This is not fresh blind validation or an independent fourth vote. Saved controls are not matched refits. Single-seed conditional bootstrap omits cumulative selection, refitting and future shift; it is not Private probability.",
  "control": "v22",
  "difference": "R112 minus v22",
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
          "bootstrap_mean_delta": 4856.7653369670825,
          "candidate_mae": 1341197.5418401144,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0595,
          "percentile_2_5": -906.7564318850636,
          "percentile_97_5": 10192.009977358664,
          "point_delta": 4786.867175800493,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 6609.459460823931,
          "candidate_mae": 1398986.7504868421,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.0485,
          "percentile_2_5": -728.8223408169113,
          "percentile_97_5": 15835.104580256408,
          "point_delta": 6096.405900099548,
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
          "bootstrap_mean_delta": 4943.853180801451,
          "candidate_mae": 1341197.5418401144,
          "control_mae": 1336410.674664314,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.2385,
          "percentile_2_5": -8052.732014224248,
          "percentile_97_5": 19029.697360009635,
          "point_delta": 4786.867175800493,
          "valid_replicates": 2000
        },
        "weighted": {
          "bootstrap_mean_delta": 6284.342857323445,
          "candidate_mae": 1398986.7504868421,
          "control_mae": 1392890.3445867426,
          "invalid_replicates": 0,
          "negative_delta_fraction": 0.216,
          "percentile_2_5": -8399.531503920245,
          "percentile_97_5": 22404.340312012853,
          "point_delta": 6096.405900099548,
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

與較強／較弱次對照v24的完整差異：

```json
{
  "automatic_adoption": false,
  "both_point_metrics_improve": false,
  "candidate": "R112",
  "cities": {
    "__MISSING__": {
      "candidate_mae": 576138.3117590193,
      "control_mae": 748690.2509114165,
      "delta": -172551.93915239722,
      "relative_change_pct": -23.047173239178882,
      "rows": 1
    },
    "南投縣": {
      "candidate_mae": 991555.8973514941,
      "control_mae": 1015289.1847550352,
      "delta": -23733.28740354115,
      "relative_change_pct": -2.337588911603291,
      "rows": 899
    },
    "嘉義市": {
      "candidate_mae": 1157156.7331653058,
      "control_mae": 1169659.4148716796,
      "delta": -12502.681706373813,
      "relative_change_pct": -1.0689164339129822,
      "rows": 604
    },
    "嘉義縣": {
      "candidate_mae": 675951.1498018821,
      "control_mae": 703921.1658472866,
      "delta": -27970.01604540448,
      "relative_change_pct": -3.9734585920197327,
      "rows": 1532
    },
    "基隆市": {
      "candidate_mae": 822762.5458694232,
      "control_mae": 824405.9509523374,
      "delta": -1643.4050829141634,
      "relative_change_pct": -0.19934415575430217,
      "rows": 842
    },
    "宜蘭縣": {
      "candidate_mae": 790232.0941571201,
      "control_mae": 785285.7426309958,
      "delta": 4946.3515261242865,
      "relative_change_pct": 0.6298791965268835,
      "rows": 2576
    },
    "屏東縣": {
      "candidate_mae": 876452.371590777,
      "control_mae": 920486.7394754328,
      "delta": -44034.36788465583,
      "relative_change_pct": -4.783813388745845,
      "rows": 691
    },
    "彰化縣": {
      "candidate_mae": 994677.0085480118,
      "control_mae": 1011819.6970080789,
      "delta": -17142.688460067147,
      "relative_change_pct": -1.6942434023331996,
      "rows": 2629
    },
    "新北市": {
      "candidate_mae": 1159712.3008372982,
      "control_mae": 1175179.3847952979,
      "delta": -15467.083957999712,
      "relative_change_pct": -1.3161466375360127,
      "rows": 20773
    },
    "新竹市": {
      "candidate_mae": 2628696.4722428187,
      "control_mae": 2294808.7567983773,
      "delta": 333887.7154444414,
      "relative_change_pct": 14.549696764721599,
      "rows": 1852
    },
    "新竹縣": {
      "candidate_mae": 2222438.5656453986,
      "control_mae": 2237487.2731571877,
      "delta": -15048.707511789165,
      "relative_change_pct": -0.6725717590587593,
      "rows": 4704
    },
    "桃園市": {
      "candidate_mae": 1039712.510200522,
      "control_mae": 1066753.324847084,
      "delta": -27040.81464656198,
      "relative_change_pct": -2.5348704350594082,
      "rows": 21410
    },
    "澎湖縣": {
      "candidate_mae": 3622626.7131888764,
      "control_mae": 3812375.149056576,
      "delta": -189748.4358676998,
      "relative_change_pct": -4.9771711452545695,
      "rows": 20
    },
    "臺中市": {
      "candidate_mae": 1628813.8027167162,
      "control_mae": 1612776.5130701174,
      "delta": 16037.289646598743,
      "relative_change_pct": 0.9943900792596365,
      "rows": 20055
    },
    "臺北市": {
      "candidate_mae": 4150279.27312882,
      "control_mae": 4030540.187754385,
      "delta": 119739.08537443494,
      "relative_change_pct": 2.9707949752796674,
      "rows": 3514
    },
    "臺南市": {
      "candidate_mae": 1284987.812507642,
      "control_mae": 1290312.176866015,
      "delta": -5324.364358372986,
      "relative_change_pct": -0.4126415648734798,
      "rows": 8483
    },
    "臺東縣": {
      "candidate_mae": 686946.2300448625,
      "control_mae": 694696.0897249224,
      "delta": -7749.859680059948,
      "relative_change_pct": -1.115575543706982,
      "rows": 213
    },
    "花蓮縣": {
      "candidate_mae": 632523.3498430937,
      "control_mae": 651104.8750916787,
      "delta": -18581.525248585036,
      "relative_change_pct": -2.8538452036576545,
      "rows": 578
    },
    "苗栗縣": {
      "candidate_mae": 1008454.9506469483,
      "control_mae": 1025581.2731701825,
      "delta": -17126.32252323418,
      "relative_change_pct": -1.6699137329501803,
      "rows": 2973
    },
    "金門縣": {
      "candidate_mae": 893980.8910051319,
      "control_mae": 893641.4111528273,
      "delta": 339.47985230456106,
      "relative_change_pct": 0.037988375210434876,
      "rows": 65
    },
    "雲林縣": {
      "candidate_mae": 978375.3126947552,
      "control_mae": 996471.2687277427,
      "delta": -18095.956032987568,
      "relative_change_pct": -1.8160037926724981,
      "rows": 1719
    },
    "高雄市": {
      "candidate_mae": 907277.4604631484,
      "control_mae": 908831.0314082025,
      "delta": -1553.5709450540598,
      "relative_change_pct": -0.17094167027361015,
      "rows": 11036
    }
  },
  "control": "R97/v24",
  "forecast_quarters": {
    "Q1": {
      "candidate_mae": 1002462.5202924679,
      "control_mae": 993826.2791985838,
      "delta": 8636.241093884106,
      "relative_change_pct": 0.8689890048840653,
      "rows": 22487
    },
    "Q2": {
      "candidate_mae": 1233666.7790521593,
      "control_mae": 1236271.5483622926,
      "delta": -2604.769310133299,
      "relative_change_pct": -0.2106955638980672,
      "rows": 27820
    },
    "Q3": {
      "candidate_mae": 1514827.903905822,
      "control_mae": 1519816.2249750085,
      "delta": -4988.321069186553,
      "relative_change_pct": -0.32821870086750654,
      "rows": 27006
    },
    "Q4": {
      "candidate_mae": 1539468.5323397345,
      "control_mae": 1535913.1617296129,
      "delta": 3555.3706101216376,
      "relative_change_pct": 0.231482527704749,
      "rows": 29856
    }
  },
  "months": {
    "2023-02": {
      "candidate_mae": 593810.2100035134,
      "control_mae": 596296.3427625896,
      "delta": -2486.132759076194,
      "relative_change_pct": -0.41692906375346106,
      "rows": 2896
    },
    "2023-03": {
      "candidate_mae": 924711.4056448479,
      "control_mae": 925522.7748700764,
      "delta": -811.3692252285546,
      "relative_change_pct": -0.08766604639658418,
      "rows": 7500
    },
    "2023-04": {
      "candidate_mae": 1118693.8545618749,
      "control_mae": 1094744.1862448624,
      "delta": 23949.66831701249,
      "relative_change_pct": 2.1876954103006896,
      "rows": 8639
    },
    "2023-05": {
      "candidate_mae": 1207214.7729895492,
      "control_mae": 1213455.5062566611,
      "delta": -6240.733267111937,
      "relative_change_pct": -0.5142943630758838,
      "rows": 8750
    },
    "2023-06": {
      "candidate_mae": 1279624.9495435038,
      "control_mae": 1276086.4061312978,
      "delta": 3538.543412205996,
      "relative_change_pct": 0.27729653691193007,
      "rows": 13180
    },
    "2023-07": {
      "candidate_mae": 1182631.9243346497,
      "control_mae": 1183156.183768008,
      "delta": -524.2594333582092,
      "relative_change_pct": -0.044310247501610105,
      "rows": 6144
    },
    "2023-08": {
      "candidate_mae": 1298973.5582983657,
      "control_mae": 1306980.5584263823,
      "delta": -8007.000128016574,
      "relative_change_pct": -0.6126334532211468,
      "rows": 7400
    },
    "2023-09": {
      "candidate_mae": 1365197.093289404,
      "control_mae": 1365893.9517492831,
      "delta": -696.8584598791786,
      "relative_change_pct": -0.05101848931878794,
      "rows": 8602
    },
    "2023-10": {
      "candidate_mae": 1602272.3009891952,
      "control_mae": 1596675.0554627297,
      "delta": 5597.245526465587,
      "relative_change_pct": 0.35055633313213247,
      "rows": 9855
    },
    "2023-11": {
      "candidate_mae": 1679241.6124912777,
      "control_mae": 1708449.945458693,
      "delta": -29208.33296741522,
      "relative_change_pct": -1.7096393748646364,
      "rows": 10336
    },
    "2023-12": {
      "candidate_mae": 1509944.7211921064,
      "control_mae": 1506056.8881887118,
      "delta": 3887.8330033945385,
      "relative_change_pct": 0.25814649060636186,
      "rows": 10416
    },
    "2024-01": {
      "candidate_mae": 1535771.411820042,
      "control_mae": 1526897.035515871,
      "delta": 8874.376304171048,
      "relative_change_pct": 0.5812033226702015,
      "rows": 10885
    },
    "2024-02": {
      "candidate_mae": 1470536.2452077477,
      "control_mae": 1437162.9778553827,
      "delta": 33373.26735236496,
      "relative_change_pct": 2.3221630299833125,
      "rows": 2566
    }
  },
  "overall_mae": {
    "candidate_mae": 1341197.5418401177,
    "control_mae": 1340328.1391452,
    "delta": 869.4026949177496,
    "relative_change_pct": 0.06486491401070002
  },
  "price_bands": [
    {
      "band": "<10m",
      "candidate_mae": 695677.6763399503,
      "control_mae": 710632.9248020112,
      "delta": -14955.248462060932,
      "relative_change_pct": -2.1044969829152795,
      "rows": 25432
    },
    {
      "band": "10-20m",
      "candidate_mae": 1051839.3330740635,
      "control_mae": 1067452.758311166,
      "delta": -15613.425237102434,
      "relative_change_pct": -1.4626806774854078,
      "rows": 59481
    },
    {
      "band": "20-50m",
      "candidate_mae": 2285992.4553199704,
      "control_mae": 2241526.0750715095,
      "delta": 44466.38024846092,
      "relative_change_pct": 1.9837547616769236,
      "rows": 20803
    },
    {
      "band": "50-100m",
      "candidate_mae": 6993563.698204264,
      "control_mae": 6794509.1460085185,
      "delta": 199054.55219574552,
      "relative_change_pct": 2.92963844654888,
      "rows": 1183
    },
    {
      "band": ">=100m",
      "candidate_mae": 28329564.530409507,
      "control_mae": 27434374.16274089,
      "delta": 895190.3676686175,
      "relative_change_pct": 3.2630245631204935,
      "rows": 270
    }
  ],
  "quarter_or_large_group_deterioration_above_5pct": [
    {
      "candidate_mae": 2628696.4722428187,
      "control_mae": 2294808.7567983773,
      "delta": 333887.7154444414,
      "group": "新竹市",
      "relative_change_pct": 14.549696764721599,
      "rows": 1852,
      "section": "cities"
    }
  ],
  "relative_forecast_months": {
    "0": {
      "candidate_mae": 652159.00260347,
      "control_mae": 645679.7496816863,
      "delta": 6479.252921783715,
      "relative_change_pct": 1.0034777960711827,
      "rows": 5055
    },
    "1": {
      "candidate_mae": 1014993.6283597843,
      "control_mae": 1023592.2057761417,
      "delta": -8598.577416357468,
      "relative_change_pct": -0.8400393601900839,
      "rows": 7516
    },
    "10": {
      "candidate_mae": 1494905.173547787,
      "control_mae": 1497637.3318924468,
      "delta": -2732.1583446599543,
      "relative_change_pct": -0.1824312392912602,
      "rows": 9935
    },
    "11": {
      "candidate_mae": 1536139.7892095037,
      "control_mae": 1514286.76981743,
      "delta": 21853.01939207362,
      "relative_change_pct": 1.4431229161903283,
      "rows": 10413
    },
    "2": {
      "candidate_mae": 1178951.642136092,
      "control_mae": 1151164.1928843837,
      "delta": 27787.44925170834,
      "relative_change_pct": 2.413856287701537,
      "rows": 9057
    },
    "3": {
      "candidate_mae": 1185921.7065181788,
      "control_mae": 1198704.169185322,
      "delta": -12782.462667143205,
      "relative_change_pct": -1.0663567371948475,
      "rows": 9401
    },
    "4": {
      "candidate_mae": 1278452.3242593657,
      "control_mae": 1271555.585854942,
      "delta": 6896.738404423697,
      "relative_change_pct": 0.5423859154208042,
      "rows": 11490
    },
    "5": {
      "candidate_mae": 1200430.2744023332,
      "control_mae": 1206414.4495510287,
      "delta": -5984.1751486954745,
      "relative_change_pct": -0.49602979729913765,
      "rows": 6914
    },
    "6": {
      "candidate_mae": 1340720.3791355207,
      "control_mae": 1342783.9583831823,
      "delta": -2063.5792476616334,
      "relative_change_pct": -0.1536791704114745,
      "rows": 7252
    },
    "7": {
      "candidate_mae": 1410375.8116997683,
      "control_mae": 1412213.0920092391,
      "delta": -1837.280309470836,
      "relative_change_pct": -0.13009936813833306,
      "rows": 9394
    },
    "8": {
      "candidate_mae": 1702978.2061763832,
      "control_mae": 1708489.12857141,
      "delta": -5510.922395026777,
      "relative_change_pct": -0.3225611625421845,
      "rows": 9933
    },
    "9": {
      "candidate_mae": 1596657.6716629453,
      "control_mae": 1611253.9941320717,
      "delta": -14596.322469126433,
      "relative_change_pct": -0.9058982955067231,
      "rows": 10809
    }
  },
  "strata": {
    "new_nonmissing": {
      "candidate_mae": 1663475.695646707,
      "control_mae": 1658897.3265622966,
      "delta": 4578.369084410369,
      "relative_change_pct": 0.2759886950868769,
      "rows": 59352
    },
    "seen": {
      "candidate_mae": 941175.5207026831,
      "control_mae": 944909.8065107913,
      "delta": -3734.285808108165,
      "relative_change_pct": -0.39520023841191004,
      "rows": 47817
    }
  },
  "used_for_release_gate": false,
  "weighted_mae": {
    "candidate_mae": 1398986.7504868458,
    "control_mae": 1397452.2756174281,
    "delta": 1534.4748694177251,
    "relative_change_pct": 0.10980517161058377
  }
}
```

真價帶與誤差尾部（僅診斷，不能拿未知test真價路由）：

```json
{
  "strata": {
    "missing": {
      "mae": null,
      "rows": 0
    },
    "new_nonmissing": {
      "mae": 1663475.695646707,
      "rows": 59352
    },
    "seen": {
      "mae": 941175.5207026831,
      "rows": 47817
    }
  },
  "forecast_quarters": {
    "Q1": {
      "horizon_days": [
        0,
        91
      ],
      "mae": 1002462.5202924679,
      "rows": 22487
    },
    "Q2": {
      "horizon_days": [
        92,
        183
      ],
      "mae": 1233666.7790521593,
      "rows": 27820
    },
    "Q3": {
      "horizon_days": [
        184,
        275
      ],
      "mae": 1514827.903905822,
      "rows": 27006
    },
    "Q4": {
      "horizon_days": [
        276,
        null
      ],
      "mae": 1539468.5323397345,
      "rows": 29856
    }
  },
  "price_bands": [
    {
      "absolute_error_share": 0.12309110071656808,
      "absolute_error_sum": 17692474664.677616,
      "band": "<10m",
      "mae": 695677.6763399503,
      "mean_signed_error_prediction_minus_truth": 407335.64897991647,
      "rows": 25432,
      "underprediction_fraction": 0.2900283107895565
    },
    {
      "absolute_error_share": 0.43527702162954696,
      "absolute_error_sum": 62564455370.57837,
      "band": "10-20m",
      "mae": 1051839.3330740635,
      "mean_signed_error_prediction_minus_truth": 287171.9860052786,
      "rows": 59481,
      "underprediction_fraction": 0.36144314991341775
    },
    {
      "absolute_error_share": 0.330855862736685,
      "absolute_error_sum": 47555501048.02135,
      "band": "20-50m",
      "mae": 2285992.4553199704,
      "mean_signed_error_prediction_minus_truth": -280701.4028309526,
      "rows": 20803,
      "underprediction_fraction": 0.4713743210113926
    },
    {
      "absolute_error_share": 0.05756007516432937,
      "absolute_error_sum": 8273385854.975644,
      "band": "50-100m",
      "mae": 6993563.698204264,
      "mean_signed_error_prediction_minus_truth": -1421469.027952723,
      "rows": 1183,
      "underprediction_fraction": 0.48520710059171596
    },
    {
      "absolute_error_share": 0.05321593975287045,
      "absolute_error_sum": 7648982423.2105665,
      "band": ">=100m",
      "mae": 28329564.530409507,
      "mean_signed_error_prediction_minus_truth": -9663490.767947089,
      "rows": 270,
      "underprediction_fraction": 0.6148148148148148
    }
  ],
  "tails": {
    "absolute_error_quantiles": {
      "0.5": 783072.8724749014,
      "0.9": 2920983.6845454792,
      "0.95": 4223085.260656307,
      "0.99": 8860345.241844548
    },
    "absolute_error_sum": 143734799361.46356,
    "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules",
    "maximum_absolute_error": 641967169.9038773,
    "mean_signed_error_prediction_minus_truth": 161524.93233986618,
    "top10_absolute_error_sum": 2353847221.159314,
    "top10_error_rows": 10,
    "top10_error_share": 0.016376321055278135,
    "top1pct_absolute_error_sum": 18235800011.08538,
    "top1pct_error_rows": 1072,
    "top1pct_error_share": 0.12687115501671994,
    "underprediction_fraction": 0.36783958047569726
  }
}
```

## 執行與可解釋範圍

三模型串行流程實測 581.485秒；獨立評分命令實測 78.125秒。這些是各程序的測量，不是整個專案總工時。

原R97四分支、兩final及兩個槽的v22還原全部bytes一致。encoder、group-OOF fit先驗、full-fit query先驗、日期權重、reference與spatial使用此起點自己的保存狀態，不用後期或全量模型回測。三個新模型、metadata及五層向量全部直接封存，確認成功退出後才開raw價格表作評分；原fit-label隔離擷取器曾解開整張價格pickle的歷史限制仍保留。

The February origin was previously used for R97 and overlaps the August windows. This is not fresh blind validation or an independent fourth vote. Saved controls are not matched refits. Single-seed conditional bootstrap omits cumulative selection, refitting and future shift; it is not Private probability.

這個窗口早已用過v22／v24，並與8月窗口重疊；不能說新盲測、四個獨立樣本或Private勝率。固定seed不包含重訓變異，保存模型可重播也不等於乾淨環境從原始資料端到端重訓。沒有查內政部答案、外部成交答案、test ID人工改價或Public搜參數。

- [事前固定計畫](round113_plan.md)
- [原三窗結果](round111_112_report.md)
- [機器結果](../outputs/round113/results.json)
- [額外發行數值門檻](../outputs/research_round113_20261008/round113_release_gate.json)
