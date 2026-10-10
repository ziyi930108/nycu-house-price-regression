# 最新接手狀態｜2026-10-10

使用者於2026-10-10明確改選 **v25**。Public MAE **1,995,836.2136元**，為使用者文字回報；比v22少2,732.3570元（約0.1367%），Private未知。原R107離線強門檻仍未通過，不因這次選定改寫。

先讀[現行選定](docs/V25_SELECTION_20261010.md)、[REPORT](REPORT.md)和[v25重現範圍](docs/V25_REPRODUCIBILITY_20261010.md)。本次發布v25選定、分數、報告入口和CSV檢查工具；原已發布研究至R118，R119後研究仍留原工作區，不能從此repo推斷已完成或仍在執行。

原v22報告／教學媒體、歷史來源、模型身分與不可變收據保留。原工作區須讀C:/NYCU_ML/HANDOFF.md與日誌。最新用量停止線剩餘20%，不重置／購買／使用額外點數；沒有背景自動續跑承諾。

下方為保留原 bytes 的歷史內容；選定、分數及重現範圍請以前述 2026-10-10 說明為準。

---

## 2026-10-08 四時間起點研究：完整保存未採用結果

R114 signed-additive NN、R116 Depthwise 單價樹、R117 固定交易五折先驗均完成四個時間起點的訓練、評分與獨立核驗，既定 combined 探索／強門檻皆未通過，因此不發行 v30–v32。R117 最新窗口比 v22 差約1,821元，其他三窗改善仍不能推翻事前固定的最新窗口不得退步規則。R115純X診斷與R118早期資料可行性盤點另列，不能當成價格改善。

[本批結果與限制](docs/RESEARCH_FOUR_ORIGINS_20261008.md) · [R114報告](evidence/reports/round114_report.md) · [R115純X報告](evidence/reports/round115_report.md) · [R116報告](evidence/reports/round116_report.md) · [R117報告](evidence/reports/round117_report.md)

[R114四窗程式](research/src/round114_research.py) · [R116四窗程式](research/src/round116_research.py) · [R117工程v2程式](research/src/round117_research_v2.py)

v22原選定、Public 1,998,568.5706元及上台材料保持；Private未知。v25仍只是小幅改善的探索候選。反覆使用歷史窗口有累積選型偏差，bootstrap不能當作Private勝率。這次保存原始工程失敗與重跑成本，只包含完成研究的程式及聚合證據，不包含交易表、模型、逐筆預測或提交CSV；沒有Kaggle操作，沒有全新環境從零重訓。後續研究仍在本機，不能把這個快照當成已完成的候選發行。

以下原日期正文逐bytes保留，依各自日期理解歷史狀態。

---

## 2026-10-08 追加研究：完整保留未採用方法

R111 殘差架構 NN 與 R112 Ordered 單價樹完成原三窗六 fit，兩者均未通過既定探索／強門檻，不產 v28／v29。額外 February 起點另完成 R107／R111／R112 三 fit：R107 整體改善約303元，R111改善約2,825元，R112退步約4,787元；額外起點改善不翻改原三窗失敗。

[本批結果與採用界線](docs/RESEARCH_CONTINUE_20261008.md) · [原三窗完整報告](evidence/reports/round111_112_report.md) · [額外起點完整報告](evidence/reports/round113_report.md) · [原三窗研究程式](research/src/round111_112_research.py) · [額外起點研究程式](research/src/round113_offset_research.py)

v25仍是小幅改善的探索候選，Public／Private未知；v22原選定、Public 1,998,568.5706元與上台材料保持。本次只發布完成的111–113程式與聚合證據，不包含未完成後續研究、提交CSV、交易表、模型或逐筆預測。重複歷史窗口存在選型偏差，並非Private保證。使用者最新停止線為Codex剩餘20%，不重置或使用額外點數。

以下所有原日期正文逐bytes保留，以其日期理解歷史狀態。

---

## 2026-10-08 研究恢復與新試交

使用者已重新授權模型優化與嘗試新提交。四方法、十二次回測及一次全量 prior NN fit 完成；v25為新增探索候選，最新年度離線 MAE 改善約0.1103%，早期略退步，Public／Private尚未知。R108／R109／R110未通過探索門檻，沒有v26／v27。原v22選定、Public 1,998,568.5706與約15分鐘簡報保留，Private未知。

[本輪方法、結果與限制](docs/RESEARCH_RESUME_20261008.md) · [R107／R108詳細報告](evidence/reports/round107_108_report.md) · [R109／R110詳細報告](evidence/reports/round109_110_report.md) · [v25訓練程式](research/src/round107_weighted_smooth_train.py) · [全量與CSV發行程式](research/src/round107_full_release.py)

CSV、交易資料及模型留在完整原工作區；此repo保存來源與彙總證據，乾淨環境端到端重訓尚未完成。以下原日期正文完整保留，歷史停止研究／唯一版本等敘述以當時狀態理解。

---

# v22 交付接手說明

先完整下載repo，再開[統一審查頁](teaching/v22_review_20261002/index.html)。首頁與指南已改指目前9檔練習工具及9檔答辯卡，舊計時／問答保留作歷史。三款PPTX、4,065字正文及15:00影片不改；[正式PDF](docs/臺灣房價預測_v22_正式報告_20261002.pdf)另更新為12頁，新增8.3同機新套件環境實測，原11頁保留。[報告更新与重建](docs/REPORT_UPDATE_20261002.md)。

導覽與Python說明已同步同機新套件環境的保存模型推論結果；新電腦、地理重建與從零重訓仍未驗證。變更身分見[導覽紀錄](teaching/v22_review_20261002/navigation_update_20261002.json)，首次複製provenance與所有舊發布收據保留。主講約15分鐘的配速仍是估計，尚未本人計時。

程式接手先看[前移的v22來源表](research/README.md)，需要辨識17項保存檔時看[DATA用途對照](DATA.md)。研究沿革留在來源表之後；歷史訓練原碼、保存模型與既有身分清單保持；2026-10-03更新可攜推論入口的環境檢查，詳見[本次CLI實測](docs/CLI_PREFLIGHT_20261003.md)。

## 當前成果

| 項目 | 狀態 |
| --- | --- |
| 主要版本 | v22，使用者已決定唯一提交；Private仍未知 |
| Public MAE | 1,998,568.5706 元，使用者截圖回報 |
| Private MAE | 未知，Public約30% test、Private其餘約70% |
| 方法 | 四個樹模型＋兩個NN，三分支各自修正後逐筆median |
| 推論重現 | 原test Excel＋17項保存模型／狀態，47,073筆與原v22位元一致 |
| CSV | 完整SHA與原v22相同，見config/v22.json |
| 原推論驗證 | 既有兩環境、隔離模式84.7364秒，4項格式測試通過 |
| 本次CLI驗證 | 9項單元測試通過、1項符號連結測試因本機權限跳過；真實錯環境提早拒絕，最終六模型重跑81.837秒，CSV與6比較向量相同，fit0 |
| 同機新venv | [後續CUDA重現](docs/CLEAN_GPU_REPLAY_20261002.md)：新ML與GPU環境、6模型／47,073筆／CSV及6比較向量逐bytes同，主程式76.930秒；不是從零重訓。上一階段CPU安裝紀錄保留 |
| 報告 | REPORT.md原文；[目前12頁正式PDF](docs/臺灣房價預測_v22_正式報告_20261002.pdf)，原11頁只作歷史保留 |
| 簡報 | v22_review_20261002內三款23頁、4,065字稿與排練HTML；主講規劃885秒，未真人計時；15:00合成影片另外附上 |
| 簡報重建 | [最新審查稿入口](docs/REVIEW_TEACHING_REBUILD.md)僅重建三款PPTX；舊--edition fifteen-minute仍產生4,051字稿。各自實測／限制見對應文件 |
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R106來源；歷史清單不回寫，新來源見本批複製清單，當前包身分以SOURCE_MANIFEST為準 |
| 最新研究 | R106六NN完成；T−v22 -63.26／+2,439.67／+422.80 元；研究門檻未全通過，不採用，無新CSV |
| 新切點R90 | 272,117 fit／107,169 query純X與helpers已在R97使用；七模型、encoder／prior、reference／spatial均已按新cutoff重建 |
| 本批發布 | 2026-10-03 CLI環境檢查更新；[本批實測](docs/CLI_PREFLIGHT_20261003.md)，當前身分以Git commit／SOURCE_MANIFEST為準 |
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

交付推論階段完成保存模型重現；後續R97另在新cutoff重建完整七模型，沒有從原始train/valid重訓正式test模型，也未重新下載OSM。後續同機新ML／GPU套件環境已完成保存模型推論並精確重播CSV；仍不能稱新電腦端到端重現或從零重訓。原GPU與CPU的NN輸出有數值差，精確重播須匹配受測設定。固定seed不保證重訓位元一致。

v24在原三期本機整體MAE更低，但新R97窗口較高，且已回報Public較高；部分高價群退步。多輪重用年度窗口有選型偏差，bootstrap不是Private勝率。不可聲稱v22已證明Private最佳。

## 最新研究與後續工作

[R106完成頁](docs/RESEARCH_UPDATES_20260929_JOINT_DROPOUT.md)是最新入口。R106六次NN與三窗評分完成；完整T−v22 MAE為 **-63.26／+2,439.67／+422.80 元**（development／confirmation／auxiliary）。研究門檻未全通過，不採用；**唯一提交仍v22、Private未知，無新CSV。依使用者要求，本輪完成後暫停研究。**

C是原R63設定的新重訓，T只聯動四個類別遮罩；同23欄完整抽樣、其他19欄及eval不變。原v22五層先exact重播，六NN與全部query封存後才獨立評分。

C−v22重訓偏移、完整層次、群別正反證與24CI見[報告](evidence/reports/round106_report.md)。單seed及舊窗口並非盲測；未知Private、2023地理及乾淨環境尚未重訓的限制保留。

使用者要求跑完本輪後暫停；目前R106已完成，停止研究，不新增R107或排程／背景訓練。簡報、影片與交接持續依使用者指示改良；不因交付工作自動恢復訓練。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。最新主講4,065個純漢字，按250–300字／分估計13.55–16.26分鐘，規劃14:45，未真人排練；附錄、問答和提示文字不計。4,789字／16:45、4,981字／17:20、4,051字／14:45及原22頁4,143字／14:30均為歷史稿。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點已完成R97七模型比較；不因舊文件的待辦而重跑或自動開新訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
