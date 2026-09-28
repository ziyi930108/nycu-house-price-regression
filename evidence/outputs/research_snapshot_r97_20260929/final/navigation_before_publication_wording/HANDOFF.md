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
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R97來源；歷史清單不回寫，新來源見本批複製清單，發布前刷新SOURCE_MANIFEST |
| 最新研究 | R97完整v22／v24新切點完成：7 native fit共1,100.391秒；MAE 1,336,410.67／1,340,328.14元，沒有新CSV |
| 新切點R90 | 272,117 fit／107,169 query純X與helpers已在R97使用；七模型、encoder／prior、reference／spatial均已按新cutoff重建 |
| 本批發布 | 已發布base 40b2f36c…6faf；R97本機快照完成，尚未新package／commit／push／ZIP，發布另記收據 |
| 後續界線 | R97 main／評分／獨立審查已結束；時間外L1盤點尚未立項，後續訓練需另有固定計畫 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [最新 R97 新切點](docs/RESEARCH_UPDATES_20260929_OFFSET.md)：完整重建、四窗反證、評分隔離、實測時間及独立審查。
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

[R97完成頁](docs/RESEARCH_UPDATES_20260929_OFFSET.md)是最新入口。固定fit < 2023-02-11，query至2024-02-10，七native與全部learned states依新fit重建；先封存全部向量，獨立scorer才讀query y。新窗v22／v24整體MAE 1,336,410.67／1,340,328.14元，加權1,392,890.34／1,397,452.28元。v24的新案及四季較差，但seen和2千萬至1億元兩價帶較好；億元群270列的退步亦保留。原三個8月窗口均為v24整體及加權較低，不能刪除。

project兩區間跨零、month兩區間偏退；都不是Private勝率。7次fit呼叫合1,100.391秒，包含於main 1,410.656秒；scorer 29.578秒、獨立審查1.500秒。220個唯一SHA與4個保存区間核對通過，全query保存重載已完成。沒有重跑正式test模型、沒有新CSV。R90早期文件的「尚未fit」是當時時序，目前helpers已在R97使用。

[前批Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)均未通過各自事前門檻；[九假設總表](evidence/reports/regression_after_r95_20260929.md)與R96第十項對照及所有失敗來源仍保存。R94是fit前工程失敗，R95是同配置恢復；R97則是現有兩配方的時間穩定檢查，不冒充另一個新模型架構。

R97快照以已發布40b2f36c64b41514e1f654e9bfb4b3b2e7f66faf為基礎；本機程式／證據與四份導覽完成，尚未新package／commit／push／ZIP。原prep與失敗收據不可改，原379支manifest、正式報告簡報、v22均保持，最外SOURCE_MANIFEST由發布流程刷新。[下一方向盤點](evidence/reports/regression_direction_after_r97_inventory.md)寫於本輪結果未知時，未立項或訓練stacker；不可按同窗query標籤調比例再當外樣本。沒有背景自動研究承諾。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點已完成R97七模型比較；不因舊文件的待辦而重跑或自動開新訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
