# R89相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | -6,360.31 | -20,900.43 | 8,500.91 |
| development | project | weighted | -6,618.50 | -20,847.72 | 8,134.14 |
| development | month | overall | -6,360.31 | -10,691.30 | -13.54 |
| development | month | weighted | -6,618.50 | -11,898.35 | -156.65 |
| confirmation | project | overall | 19,939.52 | 6,733.50 | 32,708.46 |
| confirmation | project | weighted | 23,527.17 | 9,124.51 | 37,234.92 |
| confirmation | month | overall | 19,939.52 | 268.48 | 35,132.53 |
| confirmation | month | weighted | 23,527.17 | 5,969.67 | 32,729.74 |
| auxiliary | project | overall | 44,718.80 | 27,189.43 | 65,983.71 |
| auxiliary | project | weighted | 40,753.27 | 26,601.77 | 57,706.99 |
| auxiliary | month | overall | 44,718.80 | 33,450.36 | 59,222.08 |
| auxiliary | month | weighted | 40,753.27 | 31,391.54 | 56,179.29 |

[完整結果](../outputs/round89/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
