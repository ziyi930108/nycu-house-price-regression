# R118 可行性盤點：完整時間外的條件融合

2026-10-08。這是有界可行性盤點，不是模型訓練／參數選擇／發行規約。先前R98／R99已真正做過fit-only forward-OOF全域L1融合，三窗只有一窗改善；不能把stacking寫成尚未做過。最強反證是R99在meta訓練目標改善約1.5–2.6萬元，下一期卻仍兩窗退步。

本次只查一件事：若日後以很少的pure-X／合法前綴支持／分支分歧讓凸權重隨交易改變，現有證據是否足夠建立**全部四個既定origin**的完整forward meta資料；尤其auxiliary之前缺的更早block與地理可用性。尚未選新cutoff、特徵清單、損失正則或模型形式，不能讀query誤差後在本盤點內挑模型。

## 允許讀取與固定輸出

可讀相關source／既有規約／aggregate metadata、原純X features的交易日期、canonical／fit／query索引，以及現有地理檔案名稱和來源日期紀錄。可核保存向量的檔案SHA與權威綁定，**不載入其逐筆預測值、價格標籤、模型或private prior/reference state**。不讀train價格、valid/test、query y，不計MAE、fit、GPU或bootstrap，不查外部成交價。

先列精確readset；只擴到回答缺口所需檔案，避免全面掃描歷史大檔。統計最早可用日期、每個候選完整年度block的純X列數及prefix列數，只用日期和索引。既有date invalid與canonical mapping沿原權威核驗，不重新去重、改資料或建立新訓練array。計算的日期候選只用來呈現覆蓋，不由本盤點自動選定新實驗切點。

輸出以下證據至新`outputs/research_round118_feasibility_20261008/`：

1. 四origin的meta覆蓋表、已有forward block及其權威來源、缺口，明示窗口重疊與歷史選型偏差。
2. 若補早期block，原v22所需4樹／2NN、encoder／OOF prior／reference／spatial的依賴，哪些現成、哪些要重建。不能用較晚模型、in-sample或group-OOF prior替代完整模型forward預測。
3. 地理快照的可核日期及早期block可用性；找不到早期快照就明確記缺，不把2021／2023地圖冒充更早當時可用資料。
4. 條件特徵的時間角色：每個meta row的新舊案、支持等須依產生該row forward預測的prefix建立，不能以outer fit把它事後變seen。列出可由純X前綴重建和需價格／模型重建的範圍。
5. 既有source是否支持乾淨隔離的新窗口；成本只列已有測量範圍，不相加冒充新ETA。給出「可再立單一規約」或「缺少哪些條件」的有界建議。

沒有模型採用gate、CSV預留、全量fit或Kaggle操作。若可行，主線必須另立固定單一模型／全部對照及門檻、獨立preflight後才能訓練。R116／R117按各自既定規約完成，不混改設定。v22選定、v25探索CSV與原簡報／旁白保持，Private未知；額度剩餘20%或以下保存停止，不重置／購買／使用額外點數。
