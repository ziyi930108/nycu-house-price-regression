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
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R98來源；歷史清單不回寫，新來源見本批複製清單，當前包身分以SOURCE_MANIFEST為準 |
| 最新研究 | R98前置時間L1組合完成：3 CPU meta-fit，0 native fit；三窗差+8,798.64／−4,255.03／+5,190.67元，未過門檻，無新CSV |
| 新切點R90 | 272,117 fit／107,169 query純X與helpers已在R97使用；七模型、encoder／prior、reference／spatial均已按新cutoff重建 |
| 本批發布 | 本頁於發布前核驗，base 2eec4b00…1799f；當前發布身分以Git main歷史與SOURCE_MANIFEST為準，根工作區另存收據 |
| 後續界線 | R98 main／獨立評分／保存結果稽核已結束；四欄含median僅有限提案，沒有新plan或fit |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [最新 R98 時間前推組合](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)：三窗結果、合法meta資料、完整反證、數值與保存結果稽核。
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

[R98完成頁](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)是最新入口。保留原三支完成R/S的forward預測，三期只用各自cutoff以前的meta資料學非負／和為1的全域L1權重；meta列數105,511／137,312／186,745。三獨立程序只取得自身meta_y，全部query向量保存重載及封存後，另一支scorer才讀query真價。

development／offset／confirmation的整體差為+8,798.64／−4,255.03／+5,190.67元，加權方向相同。offset新案與億元群有利，confirmation中價帶也改善；但每期均有固定大群退超5%，三期門檻皆未过，12配對區間全部跨零。不能稱全面顯著更差，也不代表Private未知部分將改善。仍缺更早forward block，這三窗探索不是原三個8月窗口的完整採用驗證。

3次CPU meta-fit呼叫合2.828秒，包含於main 6.891秒；scorer26.609秒、保存結果稽核2.609秒。233唯一SHA、9保護檔與12保存區間通過。稽核者亦為helper作者，此角色明示；scorer與前置source審查由其他代理完成。沒有新樹／NN fit、GPU或CSV。原始失敗／修正／凍結時序及summary圖文修订均保存，不能美化成完整獨立重訓。

[前批R97](docs/RESEARCH_UPDATES_20260929_OFFSET.md)已完成新cutoff七模型与fit狀態重建；v22新窗較低而v24原三年較低，反證仍保留。[更早Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)等結果不回寫。[R98後續四欄含median提案](evidence/reports/round98_next_direction.md)尚未立項／fit，不把meta目標保證當query保證，不能按同窗真價挑比例。

本批以2eec4b006db5ecf5b0624f791b90faf02ae1799f為基礎；本頁於發布前核驗，當前發布身分以此儲存庫Git main歷史與SOURCE_MANIFEST為準，根工作區另存發布收據。prep及歷史收據、379原source manifest、正式REPORT／PDF／PPTX與v22保留。原資料／標籤／模型state／逐筆陣列不在文件程式包中，沒有背景自動研究承諾。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點已完成R97七模型比較；不因舊文件的待辦而重跑或自動開新訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
