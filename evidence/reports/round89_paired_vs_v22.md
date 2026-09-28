# R89相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | -6,406.67 | -10,601.91 | -2,612.40 |
| development | project | weighted | -6,640.32 | -10,916.87 | -2,715.32 |
| development | month | overall | -6,406.67 | -9,398.38 | -1,708.05 |
| development | month | weighted | -6,640.32 | -13,238.76 | -408.61 |
| confirmation | project | overall | 9,907.03 | 5,640.20 | 13,926.63 |
| confirmation | project | weighted | 12,462.63 | 8,058.90 | 16,716.20 |
| confirmation | month | overall | 9,907.03 | -2,614.87 | 21,863.74 |
| confirmation | month | weighted | 12,462.63 | 952.80 | 20,730.71 |
| auxiliary | project | overall | 10,775.59 | 9,139.59 | 12,495.51 |
| auxiliary | project | weighted | 10,158.59 | 8,712.01 | 11,599.06 |
| auxiliary | month | overall | 10,775.59 | 6,518.56 | 16,699.91 |
| auxiliary | month | weighted | 10,158.59 | 6,180.29 | 17,181.04 |

[完整結果](../outputs/round89/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
