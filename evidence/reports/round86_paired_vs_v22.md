# R86相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 32,262.95 | -2,971.04 | 84,597.21 |
| development | project | weighted | 31,779.22 | -3,065.56 | 82,974.08 |
| development | month | overall | 32,262.95 | 5,596.24 | 81,341.66 |
| development | month | weighted | 31,779.22 | 5,499.86 | 93,463.55 |
| confirmation | project | overall | -1,051.01 | -18,279.45 | 16,242.34 |
| confirmation | project | weighted | -1,682.07 | -20,870.55 | 17,325.25 |
| confirmation | month | overall | -1,051.01 | -9,594.81 | 8,701.05 |
| confirmation | month | weighted | -1,682.07 | -10,222.47 | 9,246.03 |
| auxiliary | project | overall | 12,956.06 | -12,649.66 | 38,268.69 |
| auxiliary | project | weighted | 6,821.64 | -14,639.26 | 27,338.67 |
| auxiliary | month | overall | 12,956.06 | -936.45 | 32,054.90 |
| auxiliary | month | weighted | 6,821.64 | -2,196.20 | 15,456.46 |

[完整結果](../outputs/round86/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
