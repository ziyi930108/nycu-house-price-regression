# R88相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 10,977.52 | -671.28 | 22,551.22 |
| development | project | weighted | 10,740.71 | -610.07 | 21,877.10 |
| development | month | overall | 10,977.52 | 1,986.00 | 16,914.12 |
| development | month | weighted | 10,740.71 | 2,996.03 | 14,614.03 |
| confirmation | project | overall | 7,675.02 | -1,892.42 | 18,147.30 |
| confirmation | project | weighted | 8,949.94 | -1,736.24 | 20,233.10 |
| confirmation | month | overall | 7,675.02 | -2,221.59 | 16,953.26 |
| confirmation | month | weighted | 8,949.94 | -1,812.45 | 16,490.12 |
| auxiliary | project | overall | 16,617.14 | 10,800.92 | 21,839.43 |
| auxiliary | project | weighted | 13,647.83 | 8,994.88 | 17,917.81 |
| auxiliary | month | overall | 16,617.14 | 4,464.58 | 36,177.92 |
| auxiliary | month | weighted | 13,647.83 | 4,202.49 | 28,899.79 |

[完整結果](../outputs/round88/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
