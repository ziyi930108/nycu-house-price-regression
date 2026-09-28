# R92相對v24的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 1,859.16 | -16,678.11 | 19,985.27 |
| development | project | weighted | 1,881.34 | -16,586.64 | 19,598.04 |
| development | month | overall | 1,859.16 | -15,226.75 | 13,710.73 |
| development | month | weighted | 1,881.34 | -19,461.52 | 12,194.63 |
| confirmation | project | overall | 31,587.28 | 15,705.21 | 48,341.81 |
| confirmation | project | weighted | 34,331.12 | 16,950.50 | 52,825.06 |
| confirmation | month | overall | 31,587.28 | 14,039.69 | 44,768.06 |
| confirmation | month | weighted | 34,331.12 | 18,248.62 | 43,673.93 |
| auxiliary | project | overall | 27,189.27 | -991.57 | 55,355.45 |
| auxiliary | project | weighted | 28,049.68 | 7,064.90 | 48,695.66 |
| auxiliary | month | overall | 27,189.27 | 15,311.61 | 44,138.39 |
| auxiliary | month | weighted | 28,049.68 | 20,957.20 | 42,259.72 |

[完整結果](../outputs/round92/paired_vs_v24/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
