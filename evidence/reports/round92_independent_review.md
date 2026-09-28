# R92獨立結果核對：depth6近期加權單價樹

日期：2026-09-29（Asia/Taipei）。正式三期、generic及兩份paired完成後核對。先前實作合約見[preflight](r92_validation_audit.md)。

**工程與保存證據核對通過；結果不支持推進本候選。** 中間／最新年度整體及加權都退步，最新整體退步1.5583%，未達原三期皆改善且最新至少改善0.5%的固定門檻。最舊年度雖改善，四個vs v22配對區間均跨零，且主體價帶有代價。此報告不修改主線decision、模型、selector或v22 CSV。

## 實際模型容量與來源

本審查獨立載入3個保存CBM，只讀原生參數、feature names、category positions與leaf counts，**沒有predict或fit**。新wrapper由保存JSON核身分，嵌入legacy state和原R79各期R19 state完全相同。

| 期別 | 真CBM depth／trees／最大leaves | native欄／類別 | fit秒 | 全query列數 |
| --- | --- | --- | ---: | ---: |
| development | 6／2999／64 | 95／23 | 148.750 | 81,234 |
| confirmation | 6／2999／64 | 95／23 | 149.985 | 133,697 |
| auxiliary | 6／2999／64 | 95／23 | 105.641 | 105,511 |

三支都是Plain／SymmetricTree、MAE loss/eval、seed42，其餘原參數核對符合固定值；保存learning rate為0.06499999762，依既定1e-7容差符合.065。三期每樹葉數向量SHA均為`a81b92cbfc8a43baa8911edd2bdd1f437bc529a6827d3e13f19a4b3259e21224`，與各metadata精確一致；此處是實際CBM觀察，不是FakeModel證據。

- 主鏈source／input／各stage artifact／protected前後，共383個唯一檔案的SHA全部核對；v22 CSV與選定狀態未變。
- generic另核80個source／input身分（與上述可能重疊，不能當463個唯一檔），各stage完整metrics與main完全一致。
- 保存date weights為float32，Pool weights為float64，尺寸／正有限／values SHA符合metadata。driver內逐位核`原A/100 × saved date.astype(float64)`；本次未再讀X重算面積，明確區分獨立hash核對與driver數學核對。
- 正式driver三期原v22 base→reference→spatial→median向量和完整control metrics精確重播；新CBM存取後全query raw／TWD逐位一致、公開helper API等於顯式inverse。相關flag、metadata、向量檔SHA一致。本審查未另做全query模型predict，因此不把這項driver重載說成第二次獨立推論。

## 三期、新案與既有案

以下Δ＝R92−v22，TWD MAE，負數較好；missing三期均無樣本。加權是原固定組成指標，非Private真價。

| 窗口 | R92整體 | Δ整體 | Δ加權 | Δ新案 | Δ既有案 |
| --- | ---: | ---: | ---: | ---: | ---: |
| development | 1,409,199.87 | +1,812.80 | +1,859.52 | +43.04 | +5,003.70 |
| confirmation | 1,404,783.19 | +21,554.79 | +23,266.58 | +32,961.17 | +6,485.95 |
| auxiliary | 2,302,598.41 | −6,753.95 | −2,545.01 | −9,450.51 | +9,407.92 |

最新期四季全退步，Q2 +27,516.62、Q4 +41,365.74元；中間與最舊期都是Q1/Q2或Q2/Q3局部有利，並非全季穩定改善。既有案三期皆退步，新案只有最舊期改善。固定test-X組成估計中間／最舊小幅有利、最新退步；它僅對有支持city×seen/new質量重正規化，不是Private MAE。

原生到中位數各層的保存MAE差值也不一致：

| 窗口 | Δnative | Δbase | Δreference | Δspatial | Δ最終median |
| --- | ---: | ---: | ---: | ---: | ---: |
| development | +106,798.76 | +90,220.20 | +23,949.30 | +22,116.25 | +1,812.80 |
| confirmation | +146,464.99 | +141,780.51 | +65,701.32 | +64,238.83 | +21,554.79 |
| auxiliary | −43,181.27 | −26,046.32 | −24,201.04 | −22,102.56 | −6,753.95 |

中間／最新期在新native本身已較差，不能把退步只歸咎於median。這是保存預測的描述；未同期重訓depth8控制，不能將差值全當深度變動的純因果效果。

## 高價改善與代價都保留

| 窗口／價帶 | 列數 | Δ價帶MAE | 對整體Δ貢獻（以全體N為分母） |
| --- | ---: | ---: | ---: |
| development，1,000萬～2,000萬 | 43,914 | −3,545.37 | −1,916.58 |
| development，2,000萬～5,000萬 | 14,197 | +37,546.65 | +6,561.90 |
| development，≥1億元 | 182 | −1,392,944.02 | −3,120.81 |
| confirmation，1,000萬～2,000萬 | 74,109 | +7,032.82 | +3,898.33 |
| confirmation，2,000萬～5,000萬 | 31,959 | +49,204.33 | +11,761.83 |
| confirmation，5,000萬～1億元 | 1,725 | −50,332.93 | −649.41 |
| confirmation，≥1億元 | 319 | +2,449,462.82 | +5,844.40 |
| auxiliary，2,000萬～5,000萬 | 15,033 | +34,085.29 | +4,856.41 |
| auxiliary，5,000萬～1億元 | 1,382 | −101,359.68 | −1,327.63 |
| auxiliary，≥1億元 | 171 | −6,714,781.42 | −10,882.54 |

最舊期≥1億元的改善貢獻−10,882.54元，超過整體−6,753.95元；其餘四帶合計貢獻為+4,128.59元。最新退步也不是只靠319筆高價尾端解釋，2,000萬～5,000萬群貢獻+11,761.83元。完整五帶仍見[generic診斷](round92_diagnostics.json)，此表不刪除其餘帶；所有真價分群僅供事後診斷，沒有部署真價條件改價規則。

## 全部24個配對區間

本審查只讀12份保存的`(2000,4)`配對MAE陣列，按候選−對照計算overall及weighted差，再取原2.5%／97.5%分位數，**全部24個端點與JSON精確一致**。亦核seed20260925、2000replicates、各預測SHA、保存結果status完成、point值與main在1e-6元內一致；沒有新bootstrap或評分。

| 對照／窗口 | 全負區間 | 全正區間（偏退步） | 跨0 |
| --- | ---: | ---: | ---: |
| v22／development | 0 | 0 | 4 |
| v22／confirmation | 0 | 4 | 0 |
| v22／auxiliary | 0 | 0 | 4 |
| v24／development | 0 | 0 | 4 |
| v24／confirmation | 0 | 4 | 0 |
| v24／auxiliary | 0 | 3 | 1 |

這些區間以已保存預測和重用年度為條件，不校正多輪選型、GPU重訓或未來市場移轉；不是Private勝率、p值或新盲測。最舊單期小幅改善不能抵銷最新四區間均偏退步的反證。

## 時間與稽核範圍

main外層478.015秒、內部477.875秒，三次fit共404.376秒（包含在main內）。generic外層23.406秒、內部23.281秒。兩份paired完成結果分別記7.344／7.328秒；它們沒有各自execution.json，不把推測的外層時間當保存事實，也不把並行時間相加當總牆鐘。

本次3個CBM容量＋383檔hash步驟實測1.469秒；這不是整份獨立審查耗時。初次讀replicate陣列時誤以為前兩欄已是delta，斷言中止；核對原source的四欄配對MAE schema後，按第2−1與第4−3欄重算，全24端點通過。沒有改任何保存陣列、source或抽樣結果。

獨立執行量：3次CBM只讀載入、0次model predict、0次fit、0次新bootstrap、0次原始query label反序列化；只寫本報告。原v22仍由使用者選定，Private未知，無新CSV／Kaggle操作。

## 核對依據SHA-256

- [outputs/round92/plan.json](../outputs/round92/plan.json)：`450884f8b5a02e2da02abb849ded8e12a0236c3b649a826520217bedf9be0697`
- [outputs/round92/results.json](../outputs/round92/results.json)：`da261f7854fe1a7bf74963ec880a0af6e4e4b10982fe1228d811fc83373069ce`
- [outputs/round92/execution.json](../outputs/round92/execution.json)：`c41dc0e9fc6f371438a065546f874745bc5f9aa824fdaa724b91d1eb91206fc9`
- [reports/round92_diagnostics.json](../reports/round92_diagnostics.json)：`b46d52942a599aaa01410211b15f74558c87696e20979047365adff9064178bc`
- [outputs/round92/paired_vs_v22/results.json](../outputs/round92/paired_vs_v22/results.json)：`7367e3822f2117725f008b450664de83db2f6195e0b4110556fbeba92c901ce0`
- [outputs/round92/paired_vs_v24/results.json](../outputs/round92/paired_vs_v24/results.json)：`ccfcb57efaddf93ff32f45f7171e67396aed4c1935cd46185dc22d7c39934af7`
- [src/round92_shallow_unit.py](../src/round92_shallow_unit.py)：`21bbf98551de72aa3d03da367f8b46abd8b39746bbdfafe5e081fcb03786fd07`
- [src/round92_experiment.py](../src/round92_experiment.py)：`d84cb2acb42f188a049768ad55f1277cfa3ee6ab5305b4856b0f14c826817f43`
- [reports/round92_helper_selfcheck.json](../reports/round92_helper_selfcheck.json)：`590261e49d90a245bf07135a4f2ba9a248033ffbdff6c5979e39a8be2297b7ca`
- [reports/r92_validation_audit.md](../reports/r92_validation_audit.md)：`1021872ba89ec26a6ecbce6d2b395fed1949794cd3677a30a3d10e8bb6b5fdd8`
