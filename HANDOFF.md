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
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R96來源；歷史清單不回寫，新來源見本批複製清單，發布前刷新SOURCE_MANIFEST |
| 最新研究 | R95 Attention三期3次NN fit共1,762.376秒；R96三期0 fit對照完成；兩者均未過門檻，沒有新CSV |
| 新切點R90 | pure-X 272,117 fit／107,169 query準備與重載完成；data/tree/NN helper保存，尚無新窗真模型fit或MAE |
| 本批發布 | 本機Attention／R96快照以已發布27c5e3ec…0c16為基礎；新commit／push／ZIP由發布流程另記，不由文件複製宣稱完成 |
| 後續界線 | 本輪訓練與評分已結束；若改做新時間切點，須重建全部fit狀態，沒有背景自動研究保證 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [最新Attention／R96研究](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)：完成結果、反證、實測時間、工程恢復、程式及上游授權。
- [R93歷史研究](docs/RESEARCH_UPDATES_20260929_R93.md)：建案密度權重與當時尚未實作的Attention提案。
- [2026-09-29前批](docs/RESEARCH_UPDATES_20260929.md)：R89/R91/R92完整結果及R90準備。
- [2026-09-28研究](docs/RESEARCH_UPDATES_20260928.md)：R85–R88歷史結果與採用審查。
- presentation/：上台PPTX、講稿、來源與製作程式。
- docs/PUBLISHING.md：如何更新來源清單、測試及發布。

## 誠實界線

已完成的是保存模型推論重現，沒有在本次從原始train/valid重新訓練全部模型，也未重建OSM地理或驗證全新ML環境。原GPU與CPU的NN輸出有數值差，精確重播須匹配受測設定。固定seed不保證重訓位元一致。

v24在三期本機整體MAE更低，但已回報Public較高；部分高價群退步。多輪重用年度窗口有選型偏差，bootstrap不是Private勝率。不可聲稱v22已證明Private最佳。

## 最新研究與後續工作

最新 Attention 候選 R95 完成三期真 NN 訓練，fit 1,762.376秒、main內計時1,903.922秒。中間／最新／較早年度對v22差額+14,973.87／+28,835.21／−32,152.87元；固定採用門檻未過。對v22配對區間4負／0跨零／8正，改善區間全在較早年度。既有建案三期改善，較早期五個價帶皆改善，但中間／最新的新建案退步。R94首次main在真fit前工程失敗、0fit；R95是同模型與同科學計畫的恢復，不算第二個模型假設。

R96固定把reference局部信任折半，趨勢不變、spatial同步使用新價格與confidence，再三分支median。0 fit、main內計時110.641秒；三期整體差額+3,030.85／+4,963.30／−3,095.98元，固定加權三期皆退，未過門檻。對v22區間0負／11跨零／1正，不能稱全面顯著更差。完整結果與48個配對區間、獨立審查及來源見[最新研究頁](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)。

[九項假設總表](evidence/reports/regression_after_r95_20260929.md)合計15次樹模型、3次NN、3次CPU趨勢wrapper fit；真fit共3,932.126秒已含於成功main共4,798.733秒，不能相加。R96另為第十個機制對照，0fit、另計110.641秒。沒有實測整段聊天時間。原Public、CSV和唯一選定保持，沒有全量fit、新test候選或Kaggle操作。

上一個已發布commit為27c5e3ec0ae6172f8a2c67cc45491987380e0c16。本批只準備GitHub本機快照，保留初始58檔與所有準備收據、失敗source和Apache-2.0授權；commit／push／ZIP以後續發布收據為準。379支歷史source manifest、正式報告與簡報保持，最外層SOURCE_MANIFEST待發布流程刷新。

R90仍只有272,117 fit／107,169 query純X資料與helpers；沒有新切點encoder／prior、reference／spatial或原生模型fit與MAE。Attention已完成本輪年度窗口研究，舊可行性文件的「未實作」是當時時序。新的錯開時間比較必須重建所有fit狀態，不能用後期模型回測；v22／v24／Attention完整比較需8個原生模型。沒有未經啟動證據的背景研究承諾。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點目前只完成純X資料準備與helper，模型比較仍未啟動；不因舊文件而自動訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
