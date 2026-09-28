# R96相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 3,030.85 | -6,058.85 | 13,958.00 |
| development | project | weighted | 3,058.97 | -6,120.58 | 13,848.97 |
| development | month | overall | 3,030.85 | -685.95 | 7,931.60 |
| development | month | weighted | 3,058.97 | -1,066.43 | 6,892.22 |
| confirmation | project | overall | 4,963.30 | -1,736.62 | 11,824.54 |
| confirmation | project | weighted | 4,233.52 | -2,599.33 | 11,319.86 |
| confirmation | month | overall | 4,963.30 | 2,436.96 | 7,494.41 |
| confirmation | month | weighted | 4,233.52 | -2,415.16 | 6,976.21 |
| auxiliary | project | overall | -3,095.98 | -8,268.69 | 2,492.77 |
| auxiliary | project | weighted | 107.99 | -5,833.20 | 6,578.91 |
| auxiliary | month | overall | -3,095.98 | -8,154.37 | 964.80 |
| auxiliary | month | weighted | 107.99 | -8,601.53 | 4,392.63 |

[完整結果](../outputs/round96/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
