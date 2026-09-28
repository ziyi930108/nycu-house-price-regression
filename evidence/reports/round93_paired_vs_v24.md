# R93相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 713.26 | -17,953.50 | 18,729.64 |
| development | project | weighted | 619.88 | -17,725.30 | 18,570.26 |
| development | month | overall | 713.26 | -8,699.04 | 8,666.99 |
| development | month | weighted | 619.88 | -9,062.39 | 7,435.08 |
| confirmation | project | overall | 14,000.77 | -398.95 | 28,756.11 |
| confirmation | project | weighted | 15,940.04 | 144.55 | 32,238.66 |
| confirmation | month | overall | 14,000.77 | 864.85 | 23,076.18 |
| confirmation | month | weighted | 15,940.04 | 4,259.10 | 23,677.82 |
| auxiliary | project | overall | 23,402.68 | 1,195.22 | 46,290.17 |
| auxiliary | project | weighted | 22,563.43 | 5,074.56 | 41,395.58 |
| auxiliary | month | overall | 23,402.68 | 15,527.79 | 33,938.57 |
| auxiliary | month | weighted | 22,563.43 | 16,395.20 | 30,589.59 |

[完整結果](../outputs/round93/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
