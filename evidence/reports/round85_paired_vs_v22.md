# R85相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 9,458.39 | -27,294.51 | 54,192.46 |
| development | project | weighted | 9,350.23 | -26,656.90 | 53,689.57 |
| development | month | overall | 9,458.39 | -10,385.74 | 51,863.09 |
| development | month | weighted | 9,350.23 | -10,799.51 | 52,587.75 |
| confirmation | project | overall | 29,656.69 | -5,935.68 | 84,824.00 |
| confirmation | project | weighted | 30,607.52 | -7,353.79 | 89,626.93 |
| confirmation | month | overall | 29,656.69 | 1,366.58 | 58,603.33 |
| confirmation | month | weighted | 30,607.52 | -3,433.86 | 59,193.80 |
| auxiliary | project | overall | 31,888.74 | 6,739.99 | 61,029.25 |
| auxiliary | project | weighted | 30,200.71 | 8,753.30 | 54,672.08 |
| auxiliary | month | overall | 31,888.74 | 17,928.81 | 46,773.42 |
| auxiliary | month | weighted | 30,200.71 | 18,159.98 | 43,740.04 |

[完整結果](../outputs/round85/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
