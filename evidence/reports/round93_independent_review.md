# R93 獨立結果核對：建案密度權重

日期：2026-09-29（Asia/Taipei）。三期 main、generic 與兩組 paired 完成後核對。執行前契約見 [preflight](r93_validation_audit.md)。

**工程與保存證據通過；結果未達固定採用門檻，支持保留研究、不全量 fit／新增 CSV。** 中間及最新整體／加權 MAE 小幅退步；最舊年度整體、加權及新／舊案皆改善，不能概括成全面或顯著更差。對 v22 的 12 個區間為 2 個全負、10 個跨零、0 個全正。使用者選定 v22 的角色不變，Private 未知。

## 實際模型與權重保存

本審查獨立只讀載入全部 3 個 CBM；沒有 model.predict 或 fit。原生模型皆為 depth 8、2,999 trees、95 特徵／23 類別、每樹最多 256 葉，Plain／SymmetricTree、MAE loss/eval、seed 42；所存原生參數按固定容差符合 R79 配方。learning_rate 的保存值 0.06499999761581421 符合 .065 的 1e-7 容差。

| 窗口 | fit 秒 | query 列 | c | 真 CBM depth／trees／最大葉數 |
|---|---:|---:|---:|---|
| 中間／development | 155.593 | 81,234 | 6.438731 | 8／2999／256 |
| 最新／confirmation | 174.828 | 133,697 | 6.770329 | 8／2999／256 |
| 最舊／auxiliary | 143.063 | 105,511 | 7.235768 | 8／2999／256 |

三期每樹葉數向量 SHA 均為 `bce5085d2d84f5f83558dca4fdbdfc39a1a79a7f0e25a42307153fb73feb27bc`，與 metadata 相同。這是真 CBM 觀察，與合成 FakeModel 證據分開。

- 三期保存的 int64 counts、float64 q／Pool weights、float32 date weights，其維度、正有限及值 hash 均符合 wrapper 與 metadata。q 與 `1/sqrt(count)` 逐位相同。
- **獨立使用原 R79 保存的 w0 重算 c 及 `w0*q*c`，與本輪 Pool 向量逐位相同**；總 Pool 質量維持既定 rtol 2e-15。這比只相信完成旗標多一層核對，但本審查未再讀 fit-X 重做建案計數或面積；原 fit-X 計數由固定前置 audit 與 driver 在 fit 前獨立核對，其值 hash 亦已綁定。
- 日期向量與 R79 各期存檔逐位相同；wrapper 中 legacy feature state 也與 R79 各期 JSON 完全相同。未添加 query 密度或改特徵／單價 inverse。
- 主鏈 source／input／stage artifacts／protected 前後共 **398 個唯一檔案 SHA** 全部通過。generic 另核 80 個來源／輸入身分（與主鏈有重疊，不相加稱唯一檔案數），各期完整 metrics 與 main 完全一致。
- driver 完成三期舊 v22 base→reference→spatial→median 全向量與 control metrics 重播，新 CBM 保存後全 query raw／TWD 逐位重載一致；相關 flags、metadata、保存向量 SHA 相符。我未另跑 query predict，不將此稱為第二次獨立推論。

## 三期與新／既有建案

以下 Δ=R93−v22，單位 TWD MAE，負數較好。missing query 群三期皆 0 列；固定 weighted 指標不是 Private MAE。

| 窗口 | R93 整體 | Δ整體 | Δ加權 | Δ新案 | Δ既有案 |
|---|---:|---:|---:|---:|---:|
| 中間／development | 1,408,053.97 | +666.90 | +598.05 | +3,275.33 | -4,036.12 |
| 最新／confirmation | 1,387,196.68 | +3,968.28 | +4,875.51 | +10,013.51 | -4,018.01 |
| 最舊／auxiliary | 2,298,811.82 | -10,540.53 | -8,031.26 | -12,148.17 | -905.19 |

既有案三期皆改善，沒有達成主要動機期待的新案跨期改善：新案中間／最新退步。最新整體退步約 0.2869%，不符三期 overall／weighted 全改善且最新至少改善 0.5% 的固定門檻。最舊年度的有利結果仍應保留。

| 窗口 | ΔQ1 | ΔQ2 | ΔQ3 | ΔQ4 |
|---|---:|---:|---:|---:|
| 中間／development | +8,481.84 | -4,886.70 | -6,904.64 | +4,804.00 |
| 最新／confirmation | -9,683.42 | +13,856.34 | -5,828.66 | +14,802.74 |
| 最舊／auxiliary | -11,855.17 | -4,119.34 | -20,335.30 | -4,578.34 |

最舊四季全有利；中間 Q2／Q3 與最新 Q1／Q3 有利，其他季退步。按固定 test-X 的有支持 city×新／舊案組成估計，Δ 依序為 +16,145.06／+3,216.66／−1,495.41；這是歷史誤差重新組合、未見 test 真價，不能當 Public 或 Private 分數。

城市亦有相反結果：中間臺中改善 23,774.17 元而臺北退步 192,285.68 元；最新桃園改善 7,176.17 元而臺中退步 26,248.73 元；最舊基隆改善 632,912.43 元而臺北退步 82,062.35 元。這些是每期依整體誤差貢獻較大的兩側例子，均 ≥200 列；不是用它們選擇部署城市規則。完整所有城市見 [generic](round93_diagnostics.json)。

## 原生到最終 median 各層

比較保存的同窗 R79 單價分支，未重新 fit 控制。

| 窗口 | Δnative | Δbase | Δreference | Δspatial | Δ最終 median |
|---|---:|---:|---:|---:|---:|
| 中間／development | -32,309.58 | -19,213.77 | -24,549.98 | -25,254.86 | +666.90 |
| 最新／confirmation | +2,699.25 | +23,168.10 | -11,628.89 | -9,120.60 | +3,968.28 |
| 最舊／auxiliary | +140,495.26 | +85,010.11 | +88,884.84 | +79,491.81 | -10,540.53 |

中間 native／修正分支改善但最終 median 小幅退步；最舊 native／修正分支退步但 median 改善。median 的結果取決於逐列分支排序及誤差互補，不能只靠單一分支的平均 MAE 判斷集成會改善。這是保存結果的描述，不是本次估出了因果機制；GPU 非位元決定性，原控制也未同期重訓。

## 完整五個價帶與整體貢獻

價帶按已知回測真價事後診斷，未形成以未知 test 真價改價的規則。貢獻=價帶 ΔMAE×列數／全期 N；五帶貢獻和與整體 Δ 在 1e-6 元內一致。

| 窗口 | 價帶 | 列數 | Δ價帶 MAE | 對整體 Δ 貢獻 |
|---|---|---:|---:|---:|
| 中間／development | <10m | 22,139 | -8,686.17 | -2,367.28 |
| 中間／development | 10-20m | 43,914 | +3,192.20 | +1,725.66 |
| 中間／development | 20-50m | 14,197 | +12,950.76 | +2,263.36 |
| 中間／development | 50-100m | 802 | -260,041.47 | -2,567.31 |
| 中間／development | >=100m | 182 | +719,711.78 | +1,612.47 |
| 最新／confirmation | <10m | 25,585 | +513.09 | +98.19 |
| 最新／confirmation | 10-20m | 74,109 | +5,001.32 | +2,772.26 |
| 最新／confirmation | 20-50m | 31,959 | +13,967.96 | +3,338.91 |
| 最新／confirmation | 50-100m | 1,725 | -207,359.95 | -2,675.42 |
| 最新／confirmation | >=100m | 319 | +182,039.16 | +434.34 |
| 最舊／auxiliary | <10m | 41,132 | -16,510.10 | -6,436.23 |
| 最舊／auxiliary | 10-20m | 47,793 | -8,138.16 | -3,686.32 |
| 最舊／auxiliary | 20-50m | 15,033 | -19,392.71 | -2,763.04 |
| 最舊／auxiliary | 50-100m | 1,382 | -18,321.90 | -239.98 |
| 最舊／auxiliary | >=100m | 171 | +1,595,025.86 | +2,585.03 |

≥1 億元三期皆退步，5,000 萬至 1 億元三期皆改善。最舊整體改善由四個 <1 億價帶合計約 −13,125.57 元貢獻支撐，抵銷 ≥1 億群 +2,585.03 元代價；因此本輪最舊的改善並非只靠超高價尾端。最新 1,000 萬至 5,000 萬兩帶合計 +6,111.17 元貢獻，亦不能把最新退步全部歸咎於高價尾端。

## 全部 24 個配對區間

本審查只讀 12 份保存的 `(2000,4)` 配對 MAE 陣列，以第2−1欄及第4−3欄得到 overall／weighted delta，再取 2.5%／97.5% 分位數；**全部 24 個區間端點與 JSON 逐位一致**。seed=20260925、每項 2,000 次、無無效 replicate、point metrics 與 main 在 1e-6 元內一致，候選／對照 prediction SHA 與抽樣 source 身分通過。未執行新 bootstrap 或讀原始 query 標籤。

| 對照 | 窗口 | 抽法 | overall 95% 區間（元） | weighted 95% 區間（元） |
|---|---|---|---|---|
| v22 | 中間／development | project | [-16,487.57, +16,361.20] | [-16,035.54, +16,344.68] |
| v22 | 中間／development | month | [-7,207.70, +7,350.51] | [-7,188.55, +8,686.37] |
| v22 | 最新／confirmation | project | [-7,920.06, +16,873.51] | [-8,322.49, +18,618.98] |
| v22 | 最新／confirmation | month | [-3,547.78, +10,717.94] | [-2,651.23, +11,555.67] |
| v22 | 最舊／auxiliary | project | [-33,144.47, +10,104.14] | [-26,367.49, +8,788.41] |
| v22 | 最舊／auxiliary | month | [-15,767.15, -6,061.96] | [-14,798.91, -3,789.90] |
| v24 | 中間／development | project | [-17,953.50, +18,729.64] | [-17,725.30, +18,570.26] |
| v24 | 中間／development | month | [-8,699.04, +8,666.99] | [-9,062.39, +7,435.08] |
| v24 | 最新／confirmation | project | [-398.95, +28,756.11] | [+144.55, +32,238.66] |
| v24 | 最新／confirmation | month | [+864.85, +23,076.18] | [+4,259.10, +23,677.82] |
| v24 | 最舊／auxiliary | project | [+1,195.22, +46,290.17] | [+5,074.56, +41,395.58] |
| v24 | 最舊／auxiliary | month | [+15,527.79, +33,938.57] | [+16,395.20, +30,589.59] |

對 v22：最舊的兩個 month 指標全負，其餘 10 個跨零；沒有全正區間。因此本輪可以說未符合事先門檻，不能說統計上全面更差。对 v24 則 7 個全正、5 個跨零，沒有全負。這些相依區間條件於既有預測與重用窗口，不校正多輪選型、不含重訓和未來市場移轉；不是 Private 勝率、p 值或獨立盲測。

## 權重分布風險不能冒稱結果原因

[fit 分布盤點](round93_fit_distribution.md) 與 metadata 顯示三期總 Pool 質量保留，但單列集中度增加，有效權重列數約由 73,989→14,564、114,802→15,827、36,838→14,943。它只量化權重集中程度，不是獨立樣本量或泛化指標。原前置 audit 的最大 key 群占比（含缺名地理 key）與此處最大單列占比是不同統計量，不能混用。

缺名列 raw q=1，最終仍乘 c 約 6–7；但中間／最新缺名 Pool 占比只有 1.2783%／0.5675%，最舊為 9.8941%。因此不能只靠倍數推論前兩期是被缺名資料主導。fit 年份質量亦改變；日期向量未變不等於總加權時間分布未變。本次沒有識別這些因素對 MAE 的因果作用，也不依結果再調密度指數或門檻。

## 時間、工程事件與審查範圍

main 外層 549.828 秒、內部 549.656 秒，三次 fit 合計 473.484 秒（包含在 main 內）。generic 外層 26.906 秒、內部 26.781 秒。兩 paired 結果記錄 7.359／7.093 秒；沒有各自 execution.json，不能虛構外層時間或把並行時間相加成總牆鐘。未量測完整開發／聊天耗時。

獨立 CBM／權重審查分兩批：前兩期內部 0.125 秒、最後一期 0.047 秒；僅指相應已啟動 Python 程序內的讀取檢查，不含啟動與整份報告。paired 保存陣列端點核對內部 0.079 秒，同樣不是本輪總耗時。

本次第一次綜合審查在完成 398 個主鏈 SHA、80 個 generic 身分、完整 metrics 及價帶和之後，誤將 paired 的單一 source hash 字串視為字典，發生 AttributeError。讀原 paired source 確認兩個獨立 SHA 對應檔案後更正稽核命令，全部端點通過；沒有改來源、模型、預測或 replicate 陣列。這是審查命令的 schema 錯誤，不是研究訓練失敗。

獨立執行量：3 次 CBM 只讀載入、0 次 predict、0 次 fit、0 次新 bootstrap、0 次原始 query label 或 test array 反序列化。來源核對會 hash 整個已存在容器的 bytes，不能宣稱含標籤 bytes 完全未被讀取；只有聚合 metrics／已保存 bootstrap 數值進入本報告分析。主 driver 本來會載 canonical y、僅傳 fit y 訓練，query y 用於評分，此為已審計邊界。

只新增本報告；原 CSV／selector／decision／source 未修改。保存產物與重載一致不等於乾淨環境已從零重訓完整鏈。依 [主線 decision](../outputs/round93/decision_review.json) 保留研究、不啟動本候選全量 fit 或 CSV，v22 保持唯一提交選定。

## 依據 SHA-256

- [outputs/round93/plan.json](../outputs/round93/plan.json)：`7abab4e1091f78f4ca4a1326c0f4d654821d7cea6f73769b970e3c0791fe8025`
- [outputs/round93/results.json](../outputs/round93/results.json)：`0b6c41074bf9f21d5585edf60c3b545441fcc1f28b2afb7708b05e8aa5bb6df8`
- [outputs/round93/execution.json](../outputs/round93/execution.json)：`94ebaffe1444f8fbee7b89b0044e04bf408cf2f2a5fb9dd931512a9aca82789c`
- [reports/round93_diagnostics.json](../reports/round93_diagnostics.json)：`0eb222aabd06ecac63e8471999ebad60a884d11a6f8ebfc40156823e222ea9a2`
- [outputs/round93/diagnostics/execution.json](../outputs/round93/diagnostics/execution.json)：`09ead7d59d895f55eb164fba446ebd38745ee7e9d5cc968c4465e74baba33399`
- [outputs/round93/paired_vs_v22/results.json](../outputs/round93/paired_vs_v22/results.json)：`ad7bf082a22c3dd78df468cd0787b7c2dabe283a6d8689f2390d33a9767bef16`
- [outputs/round93/paired_vs_v24/results.json](../outputs/round93/paired_vs_v24/results.json)：`37fdb163905f473e9abca058c1e8c873e34cdf39799f4f3750d671c94edf4f23`
- [outputs/round93/decision_review.json](../outputs/round93/decision_review.json)：`60bece5be45e200f942eee36fa4ddbba03d2795afe3c1ba2b89c0295970581c4`
- [reports/round93_fit_distribution.md](../reports/round93_fit_distribution.md)：`27912cd8973a0007ca36f9c172bb4069e846653e0d5dafa6e12ff7d8afcc0a82`
- [outputs/round93_fit_distribution/results.json](../outputs/round93_fit_distribution/results.json)：`c3d8caf2af6638be08c200affa090f7863ddd49917f418866f30fed1dc06aca7`
- [src/round93_experiment.py](../src/round93_experiment.py)：`f92361e80f4fe0659e309733807fd65015ca421595918cb416f34a267590c4fb`
- [src/round93_project_density_unit.py](../src/round93_project_density_unit.py)：`7a3fcd56afea8702f84402b8cbf09e7e39b29f2bda9ea38c7b2877fc28773697`
- [reports/round93_helper_selfcheck.json](../reports/round93_helper_selfcheck.json)：`6c647aef7a2eec0f53f805ae6c1be5d3c957e3945835e793a37895e51fd1c81b`
- [reports/r93_validation_audit.md](../reports/r93_validation_audit.md)：`6a451dbecbaee2ae4f462593606a19a3b49e106c7d079f4f0830c33fce3db5d0`
