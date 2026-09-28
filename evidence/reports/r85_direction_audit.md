# R85 研究方向盤點：縮放表示與加法殘差

2026-09-28。先讀 HANDOFF.md、HANDOFF_LOG.md 最新段，再查 R79–R84 及相關歷史來源。使用者最新要求重新研究更好的回歸模型；既有 v22 選定、CSV 及簡報不被本次研究自動取代。Private 仍未知。

本代理沒有讀取資料快取或真實預測陣列、沒有訓練、沒有查外部答案。只查程式／文字／小型證據，另製作新 R85 helper 並作人工數學檢查。主執行緒另負責凍結計畫、資料隔離與唯一 GPU。

## 已測方向與反證

| 已有研究 | 干預及結果 | 本次含意 |
| --- | --- | --- |
| R19／R79 | geo95 單價 CatBoost，以 A/100 權重對齊總價 MAE；R79 再乘保存365日期權重 | 單價、面積權重與近期權重本身都不是新方法 |
| R74／R75／R77／R78 | area NN 日期權重及逐層／組合／prior初始化敏感度 | 原生改善未必穿過 reference、spatial、median，不以原生準確直接宣稱組合改善 |
| R76／R80 | 移除單價 base 的 oldraw、再試移除 LGB 槽位 | 不把刪舊槽、改0.75/0.25說成尚未測 |
| R81 | 單價 LGB 乘日期權重 | 对v22開發+3,128.04、最新−5,451.75、較早−10,621.80元；三期億元群與臺北都退，保留研究 |
| R82 | R81單價reference trend改anchor | 最新+28,362.84、較早+22,126.85元對R81，未採；不等於整個時間修正「重複加價」已證明 |
| R83／v24 | 單價CatBoost seed42/137各半，純native base無LGB | 三期點估計優於v22，Public較差、Private未知；不能因Public丟棄離線反證 |
| R84 | 相同兩seed由2999增4499棵，三期六次fit | 對v24開發+930.08、最新−1,999.31、較早+7,114.59元；三期億元群都退，未final |
| R15／R20／R27 | raw目標深度6／半學習率雙樹數／深度10 | 不同總價表示下的歷史容量失敗，不能直接當sqrt反證 |
| R64 | NN log-ratio目標和log-space MAE | 原生三期皆退，較早巨額低估加重；不值得現在再掃log-ratio參數 |
| R65／R72／R73 | prior數值輸入、道路相對先驗、同寬零欄對照 | 道路δ未有一致收益；已有監督式OOF價格先驗，不是首次使用 |

不以舊報告中「v18 Public最佳／GitHub待發布」等歷史句子當目前狀態。

## 主要假設：固定平方根面積尺度

R79 用 A/100 的線性面積縮放，學有效面積單價再乘回總價。歷史純X盤點已看到test最大有效面積超過train支援，NN、unit和價格先驗也可能共享面積外推。本次不從未知真價斷定大筆預測必錯，不封頂、不刪列、不改個別ID。

唯一固定替代：

    A = 相同有效面積（正有限建物 → 正有限車位 → 1）
    F = sqrt(A / 100)
    t = (y_TWD / 1e6) / F
    w = F * float64(saved_R63_date_weight_float32)
    p_TWD = max((z * F) * 1e6, 0)

對未投影的逐筆輸出：

    w * abs(t-z) = date_weight * abs(y_TWD - (z*F)*1e6) / 1e6

因此仍對齊日期加權總價L1，沒有像log-target改成近似相對誤差。F在A=100時為1，A四倍時增至2而非4。固定在直接總價（面積冪0）和完整單價（面積冪1）之間，不搜尋冪次。

合理但未證實的動機：換一種目標／權重／逆轉換表示，可能減輕有限容量樹必須同時處理極寬總價與線性面積還原的難度。不能說平方根保證大房不被高估：z會重新學習，可能上升或造成低估；sample weight也影響優化及正則化。收益只能看完整三期總價MAE。

為隔離表示變動，95特徵、23類別、2999樹、depth8、lr=.065、seed42、保存日期權重、R6地理保持；只替換v22原unit CatBoost分量。原LGB、NN、log、reference／spatial及0.75/0.25、中位數不另調。

检索src Python和歷史計畫／報告的sqrt／power／平方根／residual：sqrt命中以embedding維度、地理距離和標準誤為主，在有限查閱來源內未找到此sqrt(A/100)回歸目標的已執行實驗；不是全歷史操作不存在的證明。

### 已完成helper

- [round85_sqrt_area.py](../src/round85_sqrt_area.py)
- [23項人工數學檢查](round85_helper_selfcheck.json)

介面：

    model, state, report = fit_model(X, y, geo, cats, saved_date_weights)
    pred_TWD = predict_model(model, state, Xquery, geoquery, cats)

另有scale_factor、transform_target、inverse_transform、sample_weights、check_state、check_model、augment_features、parameters、source_hashes。

state沿用R19純X feature state，歷史unit／area-only欄位只作provenance。實際R85規則以report頂層為準；不得用R19 predict_model逆轉換。metadata明列legacy_state_training_fields_are_provenance_only=True；fit_scale_factor_sha256與fit_sample_weights_sha256指實際F和F×date，原unit因子只放legacy_state_unit_factor_sha256。

數學檢查涵蓋面積備援、逆轉換和投影順序、weighted L1恆等式、日期float32→float64不正規化、index／dtype／shape、NaN／負標籤、inverse overflow拒絕。沒有真fit，不能替代主程式對native Pool、保存重載和三期query的驗證。

## 第二備案：OOF先驗上的加法總價殘差

尚未實作，也不是本次主要候選。可用當期R60建案分組OOF單價先驗形成B_fit；query只用同一期full-fit prior形成B_query：

    r_fit = (y_TWD - B_fit_TWD) / 1e6
    sample_weight = 原date_weight
    p_query = max(B_query_TWD + 1e6*CatBoost_r(Xquery), 0)

新樹學「比合法先驗高／低多少金額」，允許signed residual；不同於NN乘法ratio和既存reference的log單價殘差。仍是總價MAE，不採log objective的高額交易取捨。

應先固定是否把B作額外feature；最簡單版本不加欄，95特徵不變，只換signed target＋known offset。B provenance須由prior state／fold／索引核對，不可將full-fit prior填回fit，更不可把全fit推得in-sample v22拿來當OOF殘差。

最強反論：prior本身仍乘面積，極大面積外推可能仍在；加法修正可能無法補償大案的比例錯誤；OOF和full-fit prior存在分布差異。因此只是第二備案，不能說已勝過平方根或已成穩健模型。

應讀來源：
- src/round60_recency_price_prior.py
- src/round63_nn_experiment.py：精確prior/date NPZ和列序證據
- src/round63_recency_weighted_prior_train.py：價格因子語意
- reports/round65_plan.md：OOF／full-fit分離
- src/round2_neighborhood.py、src/round8_spatial_reference.py：與既有殘差差異

## 成本與評估建議

既有實測，不是新方法承諾值：

| 流程 | 實測秒數 |
| --- | --- |
| R79每期CatBoost fit | development165.062、confirmation190.125、auxiliary151.250 |
| R79三fit／完整main | 506.437／602.453 |
| R84六個4499樹fit／完整main | 1488.782／1605.906 |
| R81三次CPU LightGBM fit | 795.375 |
| R64三次NN fit／完整main | 63.420／116.125 |

平方根若沿同時點保存狀態，只新增三次2999棵CatBoost，計算量近R79。規劃可抓約10–15分鐘訓練／重載／組合，另留資料查核與審查；不同目標會改切分成本，實測後須如實記錄。

評估應看native→base→reference→spatial→median每層，以及三期整體／固定加權、新舊建案、所有季度／城市、價格群与大額群有號MAE貢獻。配對區間只描述重用窗口的條件差異，不叫Private勝率。v22是永久原對照，v24／R84可作已知強參考，不依Public單一分數宣稱成功。

新的時間切點須重建全部fit狀態，不可拿後期模型回測。效果不支持時保留失敗，不產出宣稱較好的CSV；支持時另存新版本，Private仍未知。

## 來源

- [R79報告](round79_report.md)
- [R81取捨](round81_delivery.md)
- [R82取捨](round82_delivery.md)
- [R83容量歷史](round83_capacity_history_audit.md)
- [R84報告](round84_report.md)
- [R64反證](round64_report.md)
- [R72道路先驗](round72_report.md)
- [R73機制對照](round73_report.md)
- [R79大面積核對](round79_input_tail_audit.md)
