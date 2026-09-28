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
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R93來源；歷史清單不回寫，新來源見本批複製清單，發布前刷新SOURCE_MANIFEST |
| 最新研究 | R93三期完成、3次native fit共473.484秒；採用門檻未過，對v22區間2偏改善／10跨零／0偏退，沒有新CSV |
| 新切點R90 | pure-X 272,117 fit／107,169 query準備與重載完成；data/tree/NN helper保存，尚無新窗真模型fit或MAE |
| 本批發布 | R93本機快照以已發布656143a3…d66為基礎；新commit／push／ZIP由發布流程另記，不由文件複製宣稱完成 |
| 下一方向 | 欄位Attention僅可行性盤點，未實作／smoke／fit；不把文獻建議當作模型成果 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [R93最新研究](docs/RESEARCH_UPDATES_20260929_R93.md)：建案密度權重完整結果、反證、程式與尚未實作的Attention方向。
- [2026-09-29前批](docs/RESEARCH_UPDATES_20260929.md)：R89/R91/R92完整結果及R90準備。
- [2026-09-28研究](docs/RESEARCH_UPDATES_20260928.md)：R85–R88歷史結果與採用審查。
- presentation/：上台PPTX、講稿、來源與製作程式。
- docs/PUBLISHING.md：如何更新來源清單、測試及發布。

## 誠實界線

已完成的是保存模型推論重現，沒有在本次從原始train/valid重新訓練全部模型，也未重建OSM地理或驗證全新ML環境。原GPU與CPU的NN輸出有數值差，精確重播須匹配受測設定。固定seed不保證重訓位元一致。

v24在三期本機整體MAE更低，但已回報Public較高；部分高價群退步。多輪重用年度窗口有選型偏差，bootstrap不是Private勝率。不可聲稱v22已證明Private最佳。

## 最新研究與後續工作

使用者重新授權改良回歸。最新R93只在原單價樹的Pool權重乘fit-only建案密度平方根折減並保留總尺度，3次native fit473.484秒、main549.828秒。中間／最新／最舊整體差額+666.90／+3,968.28／−10,540.53元，三期全改善及最新至少0.5%門檻均未過。對v22配對區間2個偏改善（最舊月份）、10個跨零、0個偏退；保留新案與價格带反證，不將未採用說成全面顯著更差。詳見[R93最新研究](docs/RESEARCH_UPDATES_20260929_R93.md)。

前批R89／R91／R92均未達門檻且已隨656143a3a55c3f8444546f8011338349ca8f1d66發布。此批R93文件在其基礎整理；發布仍需刷新SOURCE_MANIFEST、驗證與正常推送，以實際Git／收據為準。原CSV和選定保持，不全量、不新增test候選／Kaggle分數。

R90仍只完成272,117 fit／107,169 query純特徵資料及helper，尚未fit新窗口encoder/prior、reference/spatial或六native，沒有新MAE。Attention方向目前只完成文獻／歷史來源可行性盤點，沒有新架構程式、合成GPU smoke、訓練或分數。後續須另固定實作與驗證，不把提案或helper當作模型完成；也不能視為自動背景研究保證。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點目前只完成純X資料準備與helper，模型比較仍未啟動；不因舊文件而自動訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
