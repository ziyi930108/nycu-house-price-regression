# R117：先驗NN僅改fit先驗的固定交易五折

2026-10-08。在R116新query分數與本輪任何價格先驗／NN真fit之前固定。R115純X結果已看過，但未用新房價誤差選種子、fold或參數。本輪只是一個受限假說比較，並非已有改善證據。必須先完成R115獨立結果核驗、本輪完整source／四起點preflight後才freeze及fit。

## 依據與最強反證

R115四起點純X診斷顯示，交易fold相對原建案group fold改變所選cell層級的fit列依development／confirmation／auxiliary／February為10290／10649／6176／10867（約3.31–4.58%），其中轉細10103／10450／6045／10628，仍有轉粗187／199／131／239。因此不是只有同層支持乘近似比例，值得一次固定比較。假說是：對已知過去交易的同案與粗層參考，改變fit先驗排除政策可能改變NN所學的比例校正。

反證同樣強：支持量比例中位只有約1.037–1.040，參考案的日期質量ESS中位約6.56–11.72，不代表大量獨立案；交易fold的同cluster reference peers均值114.32–133.14筆，而query新案仍約55–86%。原group OOF對未見案可能更合適；交易fold排除本列卻保留同案／同批近重複，可能降低fit誤差而傷害新案泛化。R67支持欄不穩定、R100時間先驗native三窗退步、R65／R102 native改善未穩定留到完整組合也保留。沒有假設交易OOF較正確、獨立、逐筆as-of或必定更好。

## 唯一科學改動與連動

每個起點以原canonical fit順序、重新建立`np.random.default_rng(42)`，一次`p=permutation(n)`，`folds[p]=np.arange(n)%5`；各fold大小差最多1。每列先驗排除自身完整transaction fold，不排除同案的其他fold。沒有改種子／重抽、按y或query分層、fold搜尋、時間OOF、子集或support shrinkage。

保持原R60四層city/district/type/building_type→city/type/building_type→type/building_type→type，raw支持≥30，不足向粗層、最後global不設30門檻。保持相同單價label `fit_y/effective_area`、365日期加權中位數／exact-half鄰值中點、invalid-date neutral1及每fold complement自己的日期anchor與mean_valid normalization。**這些先驗中位數權重須按原R60在各complement重建；不是把R115描述用的完整outer-fit日期權重當成先驗權重。**

完整fit的query prior/state依原R60重建並與此起點原full-fit state／query prior／float32 query factor exact；query先驗沒有改。fit factor仍為`effective_area * OOF_unit_prior / 1e6`，原float64運算後轉float32；唯一九array數值改動是`train_price_factor_million`。其餘八array、144num／23cat／encoder／vocabulary／fit_y與queryX、NN日期loss權重逐bytes保持。已bounded核原144num為72純X數值變換與72缺值旗標，沒有隱藏prior派生numeric欄；不能聲稱prior是純X。

NN保留原R63 Softplus ratio MLP 512→256→128、category/dropout0.1、seed42、30epochs、batch2048、AdamW .001／weight_decay .0001、cosine eta_min1e-5、日期weighted總價MAE；不混SmoothL1、skip、signed加法或Depthwise。原初始化公式`inverseSoftplus(median_fit(y_million/F_million))`保持，但F改變會連動初始bias，須分開保存原／新ratio median和bias，不能說初值相同或只隔離fold對price-estimator的因果效果。

舊R63 trainer的文字metadata硬寫group-fold。新trainer另檔、保留舊source，須準確記transaction-fold；只允許描述性provenance欄改寫，不改實際ML參數、loss、head、算術／batch／RNG。工程守衛可加九鍵前置檢查、checkpoint dtype/finite、全query重載和拒覆寫，但要把守衛與科學變因分開。若沿原R60 estimator內部legacy rule字串，須明示只代表原估計器／full-fit state，不能把其group-fold文字當成本輪排除政策證明；本輪fold evidence為獨立權威，原globals不改。

## 固定四起點與單一組合

四者一律執行：development 2022-08-11至2023-08-10（240316／81234）、confirmation 2023-08-11至2024-08-10（321550／133697）、auxiliary 2021-08-11至2022-08-10（134805／105511）、offset_2023_02_11 2023-02-11至2024-02-10（272117／107169）。各切點自己的fit labels、原NN八array、encoder、OOF與fullfit先驗、reference/spatial與原其他native狀態，不使用後期或全量模型回測。baseline v22/v24及prior槽須先完整精確回放。

僅替換v22的prior NN槽，N_base=.75*(.5*原areaNN+.5*新priorNN)+.25*原LGB，原R/S各一次，再與原H/U逐筆median；不改比例或Public選型。四模型／prepared九array／transaction fold／先驗evidence／metadata／五層直接sealed，training exit0後才另讀query價格評分。私有先驗state與逐筆F保留本機，不能上GitHub。工程失敗保留，無自動重試／參數改動。

## 評分及限制

沿R114／R116同門檻，不放寬：三August探索要求latest整體／固定加權不更差、至少兩窗整體改善、每窗兩指標退步≤0.25%、無季度或≥500列群退步>5%；強推薦要求三窗兩指標改善、latest整體至少0.5%、同群限制。兩者再AND February兩指標退步≤0.25%與同群限制。城市／月份／seen/new/missing（≥500列亦受限）／相對季度／五真價帶／尾部／五層／v24反證全保存，不只平均。

原project與3月循環月份各2000次bootstrap、seed20260925、August stage index0/1/2、offset index0保持；重疊已用窗口仍有累積選型偏差，非獨立四票或Private勝率，沒有重訓變異估計。保存模型重載不等於乾淨環境端到端重訓。

只有combined探索gate和獨立postfit通過，才另立單次全量計畫／獨立final preflight，預留v32；本研究不產CSV／fullfit、不上傳或勾選Kaggle。v22原選定、v25探索檔、簡報與旁白保持，Public／Private沒有新回報。禁止內政部／外部成交答案與test ID調價。Codex剩餘≤20%即保存停止，不重置／購買／使用額外點數。
