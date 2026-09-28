# v22 交付接手說明

更新日期：2026-09-29。這個GitHub專案整理保存模型推論、正式報告與上台簡報，原始研究工作區為C:/NYCU_ML。

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
| 簡報 | presentation/內22頁PPTX、講稿、排練指南及程式對照；20主講＋2附錄，規劃870秒 |
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R102來源；歷史清單不回寫，新來源見本批複製清單，當前包身分以SOURCE_MANIFEST為準 |
| 最新研究 | R101 診斷＋R102 六 NN 完成；T−v22 +934.33／+7,219.36／-9,619.90 元；研究門檻未全通過，不採用，無新 CSV |
| 新切點R90 | 272,117 fit／107,169 query純X與helpers已在R97使用；七模型、encoder／prior、reference／spatial均已按新cutoff重建 |
| 本批發布 | 本頁於發布前核驗，base 4217cc53e54c917a27a57a48f994eb16ea05b066；當前身分以 Git main／SOURCE_MANIFEST 為準 |
| 後續界線 | R101／R102 已完成，v22 唯一提交；後續研究需另固定計畫，本頁不新增訓練承諾 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [最新 R101／R102](docs/RESEARCH_UPDATES_20260929_PRIOR_INPUT.md)：診斷、145 欄對照、兩項門檻、24 CI 與依賴界線。
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

[R101／R102 完成頁](docs/RESEARCH_UPDATES_20260929_PRIOR_INPUT.md)是最新入口。R101 診斷與 R102 六次 NN／三期隔離評分均已完成。T 相對 v22 的 development／confirmation／auxiliary MAE 差為 **+934.33／+7,219.36／-9,619.90 元**。Primary gate 未通過，mechanism gate 未通過；**研究門檻未全通過，不採用；唯一提交仍是 v22，Private 未知。**

R101 的 fit／無真價 query 診斷已完成；R102 比較同一 R100T 七個共享 array 與同一 145 欄架構，只改末欄正零／標準化 log prior。全部六 NN 和 query 重載封存後才由隔離 scorer 讀真價。C 是同寬度機制對照，並非上一輪的 loss-mask C。

完整四向量、五層指標、兩项 gate、24 CI、所有群別退步及實測耗時見完成頁與[摘要](evidence/reports/round102_report.md)。窗口重用偏差、單 seed、未知 Private 與從零重訓未驗證的限制保留。

[R100](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)的當時下一方向已在 R101／R102 分別診斷與試驗，歷史頁不回寫。[下一方向](evidence/reports/round102_next_direction.md)只提先另固定誤差互補／median 路由診斷，尚未執行；最近窗在 base 已翻負，不能直接怪 R/S，也不能據 query 搜權重。本批不建立新的候選／調參／背景訓練承諾；部分價格狀態與模型依賴只在原工作區。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點已完成R97七模型比較；不因舊文件的待辦而重跑或自動開新訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
