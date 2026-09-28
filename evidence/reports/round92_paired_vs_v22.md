# R92相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 1,812.80 | -10,795.24 | 13,145.67 |
| development | project | weighted | 1,859.52 | -10,433.73 | 13,043.69 |
| development | month | overall | 1,812.80 | -8,625.02 | 9,578.94 |
| development | month | weighted | 1,859.52 | -11,973.34 | 8,839.77 |
| confirmation | project | overall | 21,554.79 | 8,763.41 | 34,909.77 |
| confirmation | project | weighted | 23,266.58 | 9,502.80 | 37,930.13 |
| confirmation | month | overall | 21,554.79 | 10,676.69 | 31,994.77 |
| confirmation | month | weighted | 23,266.58 | 12,056.61 | 31,842.97 |
| auxiliary | project | overall | -6,753.95 | -31,833.07 | 15,250.40 |
| auxiliary | project | weighted | -2,545.01 | -22,180.22 | 14,901.39 |
| auxiliary | month | overall | -6,753.95 | -23,540.04 | 7,877.24 |
| auxiliary | month | weighted | -2,545.01 | -13,647.64 | 9,450.61 |

[完整結果](../outputs/round92/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
