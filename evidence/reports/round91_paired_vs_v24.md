# R91相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 47,083.81 | 10,272.23 | 95,399.28 |
| development | project | weighted | 46,845.15 | 10,484.67 | 95,267.79 |
| development | month | overall | 47,083.81 | 21,235.63 | 87,524.64 |
| development | month | weighted | 46,845.15 | 21,218.50 | 95,365.73 |
| confirmation | project | overall | 5,637.80 | -23,506.77 | 34,916.73 |
| confirmation | project | weighted | 2,127.24 | -28,035.47 | 33,656.32 |
| confirmation | month | overall | 5,637.80 | -1,458.59 | 13,820.58 |
| confirmation | month | weighted | 2,127.24 | -7,063.14 | 9,875.26 |
| auxiliary | project | overall | 71,800.36 | 27,953.49 | 113,534.72 |
| auxiliary | project | weighted | 54,287.33 | 21,534.33 | 86,561.11 |
| auxiliary | month | overall | 71,800.36 | 56,753.44 | 88,669.14 |
| auxiliary | month | weighted | 54,287.33 | 35,441.88 | 72,510.08 |

[完整結果](../outputs/round91/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
