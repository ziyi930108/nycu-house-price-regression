# v22 交付接手說明

新增[三種風格PDF投影備份與上台說明](teaching/v22_styles_20261001/projection/README.md)，各23頁；PPTX與講稿保持r2。PDF為固定图片、無講者備忘稿，原件仍可編輯。

更新日期：2026-10-01。[最新三種簡報與講稿](teaching/v22_styles_20261001/README.md)各23頁，r2補強加權算例及每支修正順序；20頁主講4,789漢字、配速16:45（未真人計時），使用者尚未選風格。製作來源及核驗副本已保存，原14頁報告方法仍適用，原配速屬r1。模型仍v22、Private未知、R106研究保持停止。原工作區C:/NYCU_ML保留完整研究與渲染。

## 當前成果

| 項目 | 狀態 |
| --- | --- |
| 主要版本 | v22，使用者已決定唯一提交；Private仍未知 |
| Public MAE | 1,998,568.5706 元，使用者截圖回報 |
| Private MAE | 未知，Public約30% test、Private其餘約70% |
| 方法 | 四個樹模型＋兩個NN，三分支各自修正後逐筆median |
| 推論重現 | 原test Excel＋17項保存模型／狀態，47,073筆與原v22位元一致 |
| CSV | 完整SHA與原v22相同，見config/v22.json |
| 本次驗證 | 隔離模式84.7364秒，4項格式測試通過 |
| 報告 | REPORT.md及docs/內11頁PDF |
| 簡報 | teaching/v22_styles_20261001內三款23頁r2及講稿／問答，主講1,005秒；原22頁與原零基礎版本保留 |
| 簡報重建 | scripts/rebuild_teaching.py接受runtime路徑，只讀repo內容，在全新目錄生成三款；現有環境69頁render相同，PPTX全檔SHA不同，未全新安裝測試；詳見docs/TEACHING_REBUILD.md |
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R106來源；歷史清單不回寫，新來源見本批複製清單，當前包身分以SOURCE_MANIFEST為準 |
| 最新研究 | R106六NN完成；T−v22 -63.26／+2,439.67／+422.80 元；研究門檻未全通過，不採用，無新CSV |
| 新切點R90 | 272,117 fit／107,169 query純X與helpers已在R97使用；七模型、encoder／prior、reference／spatial均已按新cutoff重建 |
| 本批發布 | 本頁於發布前核驗，base fc39e71e543e58af13feac88f3bd22255b40eb74；當前身分以Git main／SOURCE_MANIFEST為準 |
| 後續界線 | 使用者要求本輪跑完後暫停；R106完成後停止研究，不開R107或背景訓練；v22唯一提交 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [最新 R106](docs/RESEARCH_UPDATES_20260929_JOINT_DROPOUT.md)：六NN、四欄聯動、C重訓偏移、兩門檻與24CI；完成後暫停。
- [前批 R105](docs/RESEARCH_UPDATES_20260929_NATIVE_AVERAGE.md)：固定原生平均、兩端重播、兩項門檻及24CI。
- [前批 R103／R104](docs/RESEARCH_UPDATES_20260929_BASE_OBJECTIVE.md)：誤差互補、固定 base 目標、兩項門檻、24 CI 與地理時間限制。
- [前批 R101／R102](docs/RESEARCH_UPDATES_20260929_PRIOR_INPUT.md)：診斷、145 欄對照、兩項門檻、24 CI 與依賴界線。
- [前批 R100 時間前置 prior](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)：兩組相同 loss 列、三期完整模型、24區間、工程修正與依賴界線。
- [前批 R99 含median四欄組合](docs/RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)：六向量／兩對照、合法meta、12區間、工程修正與保存结果稽核。
- [前批 R98 三欄組合](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)：三窗結果、合法meta資料、完整反證、數值與保存結果稽核。
- [前批 R97 新切點](docs/RESEARCH_UPDATES_20260929_OFFSET.md)：完整重建、四窗反證、評分隔離、實測時間及独立審查。
- [歷史Attention／R96研究](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)：完成結果、反證、實測時間、工程恢復、程式及上游授權。
- [R93歷史研究](docs/RESEARCH_UPDATES_20260929_R93.md)：建案密度權重與當時尚未實作的Attention提案。
- [2026-09-29前批](docs/RESEARCH_UPDATES_20260929.md)：R89/R91/R92完整結果及R90準備。
- [2026-09-28研究](docs/RESEARCH_UPDATES_20260928.md)：R85–R88歷史結果與採用審查。
- presentation/：上台PPTX、講稿、來源與製作程式。
- docs/PUBLISHING.md：如何更新來源清單、測試及發布。

## 誠實界線

交付推論階段完成保存模型重現；後續R97另在新cutoff重建完整七模型，沒有從原始train/valid重訓正式test模型，也未重新下載OSM或驗證全新ML環境。原GPU與CPU的NN輸出有數值差，精確重播須匹配受測設定。固定seed不保證重訓位元一致。

v24在原三期本機整體MAE更低，但新R97窗口較高，且已回報Public較高；部分高價群退步。多輪重用年度窗口有選型偏差，bootstrap不是Private勝率。不可聲稱v22已證明Private最佳。

## 最新研究與後續工作

[R106完成頁](docs/RESEARCH_UPDATES_20260929_JOINT_DROPOUT.md)是最新入口。R106六次NN與三窗評分完成；完整T−v22 MAE為 **-63.26／+2,439.67／+422.80 元**（development／confirmation／auxiliary）。研究門檻未全通過，不採用；**唯一提交仍v22、Private未知，無新CSV。依使用者要求，本輪完成後暫停研究。**

C是原R63設定的新重訓，T只聯動四個類別遮罩；同23欄完整抽樣、其他19欄及eval不變。原v22五層先exact重播，六NN與全部query封存後才獨立評分。

C−v22重訓偏移、完整層次、群別正反證與24CI見[報告](evidence/reports/round106_report.md)。單seed及舊窗口並非盲測；未知Private、2023地理及乾淨環境尚未重訓的限制保留。

使用者要求跑完本輪後暫停；目前R106已完成，停止研究，不新增R107或排程／背景訓練。後續僅待新的明確指示。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。最新主講4,789個純漢字，按250–300字／分估計15.96–19.16分鐘；16:45是配速規劃，未真人排練，附錄及問答不計入。原22頁的4,143字／14:30屬歷史稿。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點已完成R97七模型比較；不因舊文件的待辦而重跑或自動開新訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
