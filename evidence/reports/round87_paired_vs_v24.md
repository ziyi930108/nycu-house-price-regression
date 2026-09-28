# R87相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | -688.71 | -20,682.83 | 21,214.72 |
| development | project | weighted | -726.00 | -20,250.73 | 20,864.25 |
| development | month | overall | -688.71 | -14,850.48 | 19,866.65 |
| development | month | weighted | -726.00 | -16,999.69 | 19,545.05 |
| confirmation | project | overall | 18,661.13 | 443.02 | 44,313.20 |
| confirmation | project | weighted | 20,062.53 | 39.78 | 48,730.91 |
| confirmation | month | overall | 18,661.13 | 3,483.87 | 34,207.23 |
| confirmation | month | weighted | 20,062.53 | 1,856.02 | 34,758.31 |
| auxiliary | project | overall | 24,125.69 | -7,664.39 | 58,752.53 |
| auxiliary | project | weighted | 24,931.54 | 50.45 | 53,406.41 |
| auxiliary | month | overall | 24,125.69 | 6,630.20 | 35,713.47 |
| auxiliary | month | weighted | 24,931.54 | 14,111.46 | 33,670.82 |

[完整結果](../outputs/round87/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
