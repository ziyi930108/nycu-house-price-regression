# R88相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 11,023.88 | -6,424.05 | 28,063.41 |
| development | project | weighted | 10,762.53 | -6,428.03 | 27,861.42 |
| development | month | overall | 11,023.88 | -5,117.66 | 21,644.19 |
| development | month | weighted | 10,762.53 | -5,757.02 | 18,620.13 |
| confirmation | project | overall | 17,707.51 | 2,391.49 | 32,760.18 |
| confirmation | project | weighted | 20,014.47 | 3,159.23 | 36,661.86 |
| confirmation | month | overall | 17,707.51 | 406.71 | 30,334.84 |
| confirmation | month | weighted | 20,014.47 | 3,738.85 | 28,610.30 |
| auxiliary | project | overall | 50,560.35 | 33,953.64 | 70,415.83 |
| auxiliary | project | weighted | 44,242.52 | 30,227.23 | 60,153.78 |
| auxiliary | month | overall | 50,560.35 | 32,388.18 | 77,625.42 |
| auxiliary | month | weighted | 44,242.52 | 30,174.68 | 65,123.81 |

[完整結果](../outputs/round88/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
