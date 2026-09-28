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
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R92來源；歷史清單不回寫，新來源見本批複製清單，發布前刷新SOURCE_MANIFEST |
| 最新研究 | R89／R91／R92三項三期完成、六次native＋三次CPU wrapper fit；均未達門檻，沒有新CSV |
| 新切點R90 | pure-X 272,117 fit／107,169 query準備與重載完成；data/tree/NN helper保存，尚無新窗真模型fit或MAE |
| 本批發布 | 本機文件快照已整理；新commit／push／ZIP與SOURCE_MANIFEST刷新尚未由本項完成 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [2026-09-29研究](docs/RESEARCH_UPDATES_20260929.md)：R89/R91/R92完整結果、R90準備狀態與新程式入口。
- [2026-09-28研究](docs/RESEARCH_UPDATES_20260928.md)：R85–R88歷史結果與採用審查。
- presentation/：上台PPTX、講稿、來源與製作程式。
- docs/PUBLISHING.md：如何更新來源清單、測試及發布。

## 誠實界線

已完成的是保存模型推論重現，沒有在本次從原始train/valid重新訓練全部模型，也未重建OSM地理或驗證全新ML環境。原GPU與CPU的NN輸出有數值差，精確重播須匹配受測設定。固定seed不保證重訓位元一致。

v24在三期本機整體MAE更低，但已回報Public較高；部分高價群退步。多輪重用年度窗口有選型偏差，bootstrap不是Private勝率。不可聲稱v22已證明Private最佳。

## 最新研究與後續工作

使用者重新授權改良回歸。最新R89分群時間趨勢、R91先驗加法殘差、R92單價樹depth6均完成三期與完整診斷，沒有通過「三期整體／加權全改善、最新至少0.5%」門檻。R89三次CPU wrapper fit合計21.438秒；R91/R92六次native fit合計869.376秒。全部保留有利與不利分群，不全量、不新增test候選或CSV。詳見[最新研究](docs/RESEARCH_UPDATES_20260929.md)；舊窗口不是盲測，bootstrap不是Private勝率。

R90新切點純特徵資料已實際準備與重載核對，272,117 fit／107,169 query；尚未執行該窗口encoder/prior、reference/spatial或六native訓練，沒有新MAE。tree/NN helper可供後續明確決定後串接，不能將保存helper當成已完成新窗口驗證。本批文件由本機整理，發布仍需主流程刷新清單、核驗與推送，不能視為已發布或自動背景訓練保證。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點目前只完成純X資料準備與helper，模型比較仍未啟動；不因舊文件而自動訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
