# R91 固定計畫：在分組OOF價格先驗上學加法總價殘差

日期：2026-09-28（Asia/Taipei）。此候選源於先前R85方向盤點的第二備案，並非從Public分數搜參數。R89三期未通過後，換一個預先有定義的目標表示；不搜尋R89群別例外或收縮比例。R90編號保留為錯開半年驗證基礎程式，尚未真fit。v22仍是使用者唯一選定提交，Private未知。

## 假設及唯一改動

既有R79直接學單價，再乘面積回總價。既有R86把價格先驗放進一個額外特徵，未帶來一致完整組合改善。R91則以同窗R60價格先驗當已知offset，讓CatBoost學總價的加法差額：

- A：原有效面積規則，正且有限建物面積，否則正且有限車位面積，否則1。
- fit的P：原R60相同時間窗、按建案分五折的OOF單價先驗，TWD/m²。
- B_fit = A * P_OOF；target = (y_TWD - B_fit) / 1e6，允許正負。
- sample_weight = 原R63保存float32日期權重轉float64；**不乘面積、不再次正規化**。
- query的P：同一fit的R60 full-fit prior所產生，不用query價格。
- prediction_TWD = max(A_query * P_query + 1e6 * model_output, 0)。
- nonnegative投影只在還原完整總價後一次套用；signed model_output不可預先截成0。
- 原95特徵／23類別不變；不把prior或offset新增進X。

對投影前價格，date_weight*abs(target-z) = date_weight*abs(y-(B+1e6*z))/1e6。因此優化仍是日期加權總價MAE，學習的是先驗以外的金額差異，不是log誤差或ratio。這個恆等式不代表樹模型與單價表示會學到同一函數，也不保證改善。

原CatBoost固定2999棵、depth8、lr.065、seed42及其他R79參數全保留。只替換v22單價槽的CatBoost；仍.75*新總價+.25*原LGB，沿原reference/spatial，最後與保存NN/log分支逐筆median。R89群斜率不併入。原模型、state、來源及CSV不可覆寫。

## 風險與防洩漏

P_fit必須是同期group OOF；禁止full-fit prior填回fit，禁止用in-sample v22預測當殘差offset。每期重播fit_oof_prior，完整核對OOF值、fold ID、full-fit state、query prior以及R63 factor，價格僅由該期fit提供。分組OOF不是時間OOF；fit內較後成交可進其他fold reference，此限制明載。

OOF與full-fit prior存在分布差異；offset仍乘面積，巨大面積風險沒有消失；模型未必能以加法補足比例誤差。日期權重相同，但面積權重移除會影響優化尺度，不能把最終差異全歸因單一機制。單seed GPU有數值非決定性，歷史v22控制未同期重訓，不稱純因果消融。

## 固定驗證

沿R79 development、confirmation、auxiliary三期；全部執行，不按第一期挑參數或停止差的年度。主對照v22、次對照v24。切點、fit/query indices、y、原X與地理來源、R60 prior、R63日期權重、原reference/spatial皆核SHA與列序。

helper只接fit X/y/geo/date/OOF prior，predict API不接query y。先合成檢查signed target、inverse及投影順序、date-only Pool权重、95欄不變、finite/shape/index和query API，再實際三次GPU fit。存native/state及全query raw輸出與總價，保存重載兩者必須逐位一致；另核公開predict_model與手算inverse相同。原v22 base→reference→spatial→median完整重播一致。

輸出整體、固定加權、新舊建案、全部城市／季度、真價五帶、尾部與已有固定test-X組成估計；組成估計不是test真實MAE。保留native/base/reference/spatial/median各層分數，不能把逐層差額相加成因果分解。兩對照各建案／三月月份區塊、overall／weighted、三期共24配對區間，2000次、seed20260925。

## 事前採用門檻

沿本批保守門檻：三期整體及固定加權MAE都低於v22、最新整體改善至少0.5%，再考慮進入R90固定錯開起點的重建比較。還要審查新建案、高價、季度退步與完整區間。不通過即保留研究，不能事後改群規則、混合比、seed或目標再冒稱原計畫。即使通過也不自動改Kaggle選定；全量fit／新CSV须另具體記錄。

既有三窗口反覆用於研究，R91不是盲測；bootstrap未校正選型偏差，不能換算Private勝率。禁止到內政部實價資料系統對答案，不拿Public反覆搜參數，不按test ID調價。原v22模型、選定、報告與簡報保留。
