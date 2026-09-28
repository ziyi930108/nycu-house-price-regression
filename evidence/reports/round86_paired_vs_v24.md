# R86相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 32,309.30 | -4,131.35 | 83,029.61 |
| development | project | weighted | 31,801.04 | -4,131.50 | 82,095.87 |
| development | month | overall | 32,309.30 | 5,948.96 | 75,773.44 |
| development | month | weighted | 31,801.04 | 5,998.26 | 84,122.77 |
| confirmation | project | overall | 8,981.48 | -6,758.87 | 25,102.40 |
| confirmation | project | weighted | 9,382.46 | -7,888.71 | 27,522.27 |
| confirmation | month | overall | 8,981.48 | 2,953.94 | 16,939.96 |
| confirmation | month | weighted | 9,382.46 | 2,675.82 | 17,963.52 |
| auxiliary | project | overall | 46,899.28 | 14,304.60 | 80,337.13 |
| auxiliary | project | weighted | 37,416.33 | 11,632.23 | 62,546.71 |
| auxiliary | month | overall | 46,899.28 | 26,342.67 | 74,534.43 |
| auxiliary | month | weighted | 37,416.33 | 23,928.65 | 52,163.18 |

[完整結果](../outputs/round86/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
