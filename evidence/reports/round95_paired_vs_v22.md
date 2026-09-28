# R95相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 14,973.87 | 5,141.53 | 24,829.48 |
| development | project | weighted | 14,733.96 | 5,097.86 | 24,077.42 |
| development | month | overall | 14,973.87 | 10,223.08 | 19,651.72 |
| development | month | weighted | 14,733.96 | 11,849.15 | 17,262.90 |
| confirmation | project | overall | 28,835.21 | 16,657.53 | 42,819.45 |
| confirmation | project | weighted | 32,836.37 | 19,424.10 | 47,615.29 |
| confirmation | month | overall | 28,835.21 | 17,230.46 | 41,425.96 |
| confirmation | month | weighted | 32,836.37 | 24,246.32 | 39,971.57 |
| auxiliary | project | overall | -32,152.87 | -59,210.33 | -12,572.66 |
| auxiliary | project | weighted | -26,139.25 | -46,611.49 | -10,951.22 |
| auxiliary | month | overall | -32,152.87 | -58,230.26 | -16,083.30 |
| auxiliary | month | weighted | -26,139.25 | -43,209.78 | -14,064.46 |

[完整結果](../outputs/round95/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
