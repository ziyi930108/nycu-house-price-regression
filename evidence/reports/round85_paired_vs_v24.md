# R85相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 9,504.75 | -24,491.98 | 51,807.51 |
| development | project | weighted | 9,372.05 | -24,032.20 | 50,873.35 |
| development | month | overall | 9,504.75 | -12,012.51 | 52,820.72 |
| development | month | weighted | 9,372.05 | -12,949.85 | 54,292.34 |
| confirmation | project | overall | 39,689.18 | 4,488.04 | 93,103.37 |
| confirmation | project | weighted | 41,672.06 | 3,930.24 | 102,296.01 |
| confirmation | month | overall | 39,689.18 | 12,723.32 | 69,052.80 |
| confirmation | month | weighted | 41,672.06 | 9,228.97 | 69,877.40 |
| auxiliary | project | overall | 65,831.96 | 32,762.58 | 106,419.60 |
| auxiliary | project | weighted | 60,795.40 | 32,664.29 | 94,969.04 |
| auxiliary | month | overall | 65,831.96 | 52,949.98 | 87,101.93 |
| auxiliary | month | weighted | 60,795.40 | 48,648.87 | 81,020.10 |

[完整結果](../outputs/round85/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
