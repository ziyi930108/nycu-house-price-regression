# R93相對v22的直接成對診斷

固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。

| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |
|---|---|---|---:|---:|---:|
| development | project | overall | 666.90 | -16,487.57 | 16,361.20 |
| development | project | weighted | 598.05 | -16,035.54 | 16,344.68 |
| development | month | overall | 666.90 | -7,207.70 | 7,350.51 |
| development | month | weighted | 598.05 | -7,188.55 | 8,686.37 |
| confirmation | project | overall | 3,968.28 | -7,920.06 | 16,873.51 |
| confirmation | project | weighted | 4,875.51 | -8,322.49 | 18,618.98 |
| confirmation | month | overall | 3,968.28 | -3,547.78 | 10,717.94 |
| confirmation | month | weighted | 4,875.51 | -2,651.23 | 11,555.67 |
| auxiliary | project | overall | -10,540.53 | -33,144.47 | 10,104.14 |
| auxiliary | project | weighted | -8,031.26 | -26,367.49 | 8,788.41 |
| auxiliary | month | overall | -10,540.53 | -15,767.15 | -6,061.96 |
| auxiliary | month | weighted | -8,031.26 | -14,798.91 | -3,789.90 |

[完整結果](../outputs/round93/paired_vs_v22/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)
