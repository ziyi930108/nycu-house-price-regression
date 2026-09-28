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
| 研究程式 | research/src/保存379支歷史原碼與後續R85–R100來源；歷史清單不回寫，新來源見本批複製清單，當前包身分以SOURCE_MANIFEST為準 |
| 最新研究 | R100 時間前置 prior：6 NN、三期完整評分；T−v22 為 −5,097.81／+10,063.35／−11,676.48 元，未過門檻，無新 CSV |
| 新切點R90 | 272,117 fit／107,169 query純X與helpers已在R97使用；七模型、encoder／prior、reference／spatial均已按新cutoff重建 |
| 本批發布 | 本頁於發布前核驗，base c3484298…c118f；當前發布身分以 Git main 歷史與 SOURCE_MANIFEST 為準，根工作區另存收據 |
| 後續界線 | R100 六 NN、評分與保存結果稽核完成；下一方向僅提出 fit 側診斷，未執行、未另立新模型計畫 |
| GitHub | 私人ziyi930108/nycu-house-price-regression |
| 未交付資料 | 原交易表、模型、價格先驗與推論結果保留本機 |

## 先看哪些檔案

- README.md：實際可用命令與保存模型推論入口。
- REPORT.md：清理、特徵、日期權重、六模型與三分支、年度回測、Public、限制。
- evidence/reproduction/summary.json及integration_verified.json：全量實測結果。
- research/README.md：原訓練程式入口及限制。
- [最新 R100 時間前置 prior](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)：兩組相同 loss 列、三期完整模型、24區間、工程修正與依賴界線。
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

[R100 完成頁](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)是最新入口。原 group-OOF 並非外層 query 洩漏；本輪固定比較相同 eligible loss 列的原先驗 C 與日曆季前置、同 held fold 排除的先驗 T。保留原 R63 trainer、日期權重原值、完整 outer-fit encoder／query prior 與同窗 R/S。每期少 56 warmup 列，沒有重新分折。三期六個 NN 完成封存／重載後才由 scorer 讀 query 真價。

T 相對 v22，development／confirmation／auxiliary 整體差為 −5,097.81／+10,063.35／−11,676.48 元，加權方向相同。最近窗退 0.7275%；auxiliary 嘉義市 731 筆退 6.1285%，原固定門檻未過。相對同樣本 C 為 −3,704.14／+11,158.91／−4,972.80 元。C 三期點估計較原 v22 低，仍只是機制對照，不能事後改選。T 原生 prior NN 三期均退，完整 median 兩期改善，層差不能相加成因果。

24 個主要配對區間中 4 負、14 跨零、6 正；T−v22 的 auxiliary 四個區間全負，而 T−C 的 confirmation 四個全正。窗口多次重用有選型偏差，區間不是 Private 勝率。近期退步不限豪宅，auxiliary 收益也不限豪宅；完整價格／城市／新建案反證見[聚合摘要](evidence/reports/round100_summary.md)。

六次 NN fit 合計 118.203 秒、prior 生成 160.876 秒，均包含於 main 391.938 秒；scorer 68.985 秒；保存結果稽核內部 77.078 秒（外層程序 80.109 秒）。稽核通過 1,827 唯一 SHA、9 保護檔、705 prefix state 與 24 保存區間。稽核者參與純 X 政策／稽核且審閱 helper，其他核心模組由不同作者負責，不能稱完全獨立重作。

原 scorer 的純 X 計畫／科學計畫 SHA 接點在評分前發現，凍結原碼保留，另建 v2 窄修。42 人工檢查與真 metadata 前置核對後正式評分；MAE、配對、gate、公式保持，没有真模型失敗或重試。[v2 契約](evidence/reports/round100_evaluation_contract_v2.md)與原稿／diff 收據均保留。

[R99](docs/RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)結束時的 temporal prior 提案現已另立 R100 完成；歷史提案不回寫。[本輪有限下一方向](evidence/reports/round100_next_direction_inventory.md)只提 fit 側 prior／支持層級／ratio／bias 分布診斷，尚未執行或選定新模型，不掃季長、seed 或融合比例。[R98](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)、[R97](docs/RESEARCH_UPDATES_20260929_OFFSET.md)及[Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)完整反證保留。

本批以 c34842980f8abc4bcc15dc9d5940fe2d50ac118f 為基礎；本頁於發布前核驗，當前發布身分以 Git main 歷史與 SOURCE_MANIFEST 為準，根工作區另存發布收據。原 prep／歷史收據、379 source manifest、正式 REPORT／PDF／PPTX 與 v22 保持。交易表、模型、標籤、逐筆陣列、705 完整 prefix state 及含每個 prefix 價格的 temporal_prior_evidence 留在本機；只有其身分在 stage seal 保留。這是文件／程式快照，不是完整可重訓備份，沒有背景自動研究承諾。

使用者已選定v22，最新紀錄為evidence/selection/user_selected_v22_20260928.json。保留的上台要求是主講至少12分鐘並交代全部流程與設計理由。已擴充完整清理、特徵、驗證、模型表示、近期權重、OOF先驗、成交參考與空間修正、median、最終提交及保存模型重現。主講4,143個純漢字，按250–300字／分估計13.81–16.57分鐘；14:30是配速規劃，不是已完成的真人排練。內部版本比較只作技術研究沿革，不再列為待選模型。錯開時間切點已完成R97七模型比較；不因舊文件的待辦而重跑或自動開新訓練。

新切點必須重建encoder、OOF prior、NN、log、unit及reference/spatial。不能拿最終全量模型回測已看過的早期資料。

## 更新責任

每次實質修改同步更新本頁與相關報告、簡報或實測證據。若仍在原C:/NYCU_ML工作，同時更新其HANDOFF.md、HANDOFF_LOG.md並執行src/export_handoff.py。GitHub提交前刷新SOURCE_MANIFEST，保留原產物與失敗紀錄。

自動檢查設定目前只保留docs/source-checks.example.yml。發布登入缺workflow scope，未啟用GitHub Actions；本機來源/格式檢查仍已通過。
