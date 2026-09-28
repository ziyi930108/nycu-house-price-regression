# R87相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | -735.06 | -22,096.42 | 22,246.34 |
| development | project | weighted | -747.83 | -21,308.57 | 21,654.80 |
| development | month | overall | -735.06 | -10,942.76 | 18,142.20 |
| development | month | weighted | -747.83 | -11,824.11 | 17,974.41 |
| confirmation | project | overall | 8,628.64 | -8,254.19 | 33,621.60 |
| confirmation | project | weighted | 8,997.99 | -9,518.14 | 36,453.50 |
| confirmation | month | overall | 8,628.64 | -5,237.33 | 23,274.91 |
| confirmation | month | weighted | 8,997.99 | -7,782.50 | 23,897.65 |
| auxiliary | project | overall | -9,817.53 | -33,282.85 | 9,791.16 |
| auxiliary | project | weighted | -5,663.14 | -24,151.30 | 10,383.13 |
| auxiliary | month | overall | -9,817.53 | -34,240.35 | 6,490.53 |
| auxiliary | month | weighted | -5,663.14 | -21,741.78 | 6,775.83 |

[完整結果](../outputs/round87/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
