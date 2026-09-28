# R91相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 47,037.46 | 10,574.22 | 99,389.38 |
| development | project | weighted | 46,823.32 | 10,803.78 | 98,954.58 |
| development | month | overall | 47,037.46 | 21,061.05 | 91,166.26 |
| development | month | weighted | 46,823.32 | 21,437.00 | 99,995.13 |
| confirmation | project | overall | -4,394.69 | -33,538.42 | 25,483.08 |
| confirmation | project | weighted | -8,937.30 | -39,291.47 | 21,670.36 |
| confirmation | month | overall | -4,394.69 | -15,826.94 | 11,064.59 |
| confirmation | month | weighted | -8,937.30 | -20,521.76 | 68.97 |
| auxiliary | project | overall | 37,857.15 | -4,967.79 | 79,067.07 |
| auxiliary | project | weighted | 23,692.65 | -9,513.79 | 56,150.43 |
| auxiliary | month | overall | 37,857.15 | 23,040.51 | 56,192.99 |
| auxiliary | month | weighted | 23,692.65 | -2,830.52 | 44,351.66 |

[完整結果](../outputs/round91/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
