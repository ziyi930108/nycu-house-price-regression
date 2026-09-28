# R95相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 15,020.22 | 2,233.09 | 27,907.08 |
| development | project | weighted | 14,755.78 | 2,191.07 | 27,329.47 |
| development | month | overall | 15,020.22 | 3,465.92 | 23,569.92 |
| development | month | weighted | 14,755.78 | 4,449.63 | 20,235.04 |
| confirmation | project | overall | 38,867.70 | 20,526.61 | 56,540.57 |
| confirmation | project | weighted | 43,900.90 | 23,673.08 | 63,768.22 |
| confirmation | month | overall | 38,867.70 | 19,252.01 | 54,787.80 |
| confirmation | month | weighted | 43,900.90 | 32,345.94 | 51,580.28 |
| auxiliary | project | overall | 1,790.34 | -30,775.23 | 32,841.02 |
| auxiliary | project | weighted | 4,455.44 | -21,387.00 | 28,552.99 |
| auxiliary | month | overall | 1,790.34 | -16,900.36 | 13,973.62 |
| auxiliary | month | weighted | 4,455.44 | -5,903.23 | 13,076.63 |

[完整結果](../outputs/round95/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
