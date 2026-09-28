# R96相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 3,077.20 | -12,431.00 | 20,296.06 |
| development | project | weighted | 3,080.79 | -12,506.59 | 19,755.34 |
| development | month | overall | 3,077.20 | -856.03 | 8,106.68 |
| development | month | weighted | 3,080.79 | -3,200.56 | 7,746.76 |
| confirmation | project | overall | 14,995.79 | 2,959.90 | 27,433.45 |
| confirmation | project | weighted | 15,298.05 | 1,968.29 | 28,578.75 |
| confirmation | month | overall | 14,995.79 | 4,942.50 | 20,720.56 |
| confirmation | month | weighted | 15,298.05 | 1,567.83 | 20,548.56 |
| auxiliary | project | overall | 30,847.23 | 14,263.83 | 51,601.78 |
| auxiliary | project | weighted | 30,702.68 | 16,271.68 | 48,078.46 |
| auxiliary | month | overall | 30,847.23 | 26,914.56 | 36,115.38 |
| auxiliary | month | weighted | 30,702.68 | 25,106.13 | 35,082.01 |

[完整結果](../outputs/round96/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
