# R117 方法與作者準備核對

2026-10-08。依[固定規約](round117_plan.md)準備；plan SHA `c4084bee6bbe0f9ed97e019a88f48cbeee992f3a297a1b6ce251709f5377e8eb`。本文件不授權真先驗建立、array preparation、NN訓練、評分或發行。R115獨立結果核驗與完整四起點preflight仍由主線另外把關。

## 比較什麼及為什麼

R115純X診斷中，四起點約3–5% fit列的先驗路由層級改變，多數轉細，仍有少數轉粗。這提供一個受限表示假說：原先驗NN只把fit先驗排除政策從建案group五折改為交易五折，可能改變它所學的價格比例校正；尚沒有價格改善證據。

最強反證同時成立：支持比median僅約1.037–1.040、日期質量project ESS median約6.56–11.72；交易OOF允許同案／同批其他列進reference，query新案仍約55–86%。它可能只降低fit難度，傷害新案；排除本列不等於排除近重複，也不是獨立或逐筆as-of OOF。R67支持特徵不穩、R100時間prior native三窗退步、R65／R102 native改善未穩定留在完整median，均反對把先驗改動直接當收益。四個已反覆使用且重疊的窗口不是四票獨立盲測，bootstrap不是Private勝率。

## 先驗程式的精確界線

[新先驗](../src/round117_transaction_prior.py)直接呼叫原R60的`_dates`、`_weights`、`_build`、`_predict`及原cell keys／有效面積。唯一排除政策為每起點重新建立一次`np.random.default_rng(42)`，`folds[permutation(n)]=np.arange(n)%5`，不能重抽、搜尋、按y或query分層。canonical fit indices須排序、唯一、非負int64；這是工程身分守衛，沒有重新排序輸入。

各held fold的median只由其complement labels、area與dates建立，**各complement日期anchor與mean_valid normalization沿原R60重新計算**。不拿R115描述統計用的完整outer-fit日期權重替代。原四cell、raw30、global無30門檻、365日weighted median、exact-half相鄰中點、invalid-date neutral1與原underflow拒絕皆保持。

`fit_oof_prior(Xfit,yfit,source_indices)`回傳OOF float64單價、原R60完整fit state、int64 folds及獨立evidence。fullstate直接原估計器在全部fit上建立，原內部RULES不改；其legacy group文字只能作估計器／fullfit來源，不是本輪分折證明。新evidence的`fold_policy='transaction-permutation-5-fold'`、fold assignment、五個reference索引／label／date SHA才是政策權威。沒有修改任何父系RULES或globals。

`predict_prior(state,Xquery,return_diagnostics=False)`直接沿原R60，無query date／label輸入。人工資料上fullstate、query先驗與JSON重載／順序逐bytes一致；**真資料的原saved state／query先驗／float32 query factor exact仍須driver獨立核對**，作者人工測試不冒稱已做。

## NN的精確界線

[新trainer](../src/round117_transaction_prior_train.py)使用原`round47_nn_train.EmbeddingMLP`類別與原`round63_recency_weighted_prior_train.weighted_loss`函式本身，沒有新subclass、SmoothL1、skip、signed head或monkeypatch。原512→256→128、23 embeddings、Softplus ratio、dropout／category dropout .1、seed42／30epochs／2048、AdamW .001／.0001、cosine eta_min1e-5及CUDA setup全部保持。模型建構前沒有新增耗用Torch RNG的操作；原初始化tensor在人工seed42對照逐tensor一致。

`BASELINE_CONFIG`是完整原R63 **CONFIG**；唯一`PROVENANCE_CONFIG_OVERRIDES`為`train_price_factor_construction`描述，實際ML與loss configuration沒有改動。舊公式字串中的`group_prior`指階層cell先驗，不拿這個legacy字當建案排除證明。新metadata明確保存transaction政策及「trainer未驗分折／日期／query factor來源」的界線。

原bias公式保持`inverseSoftplus(median_float64(y_float32/F_float32))`，但F變動會連動median及bias；`initial_training_median_ratio`、`initial_head_bias`是float64公式值，`initial_head_bias_float32`是實際head parameter cast值。trainer只有新九array，**原／新F、median與bias比較由driver保存**，不能聲稱初始化值相同或純隔離fold對估計器的因果效果。

九key與重複key在物化前檢查；144／23、dtype、finite、positive factors／loss weights及category bounds再核。只有`train_price_factor_million`應變，其他八array exact由driver檢查。trainer保存全部九array identity，完全不接受query y，也不計query loss／early stopping。

CLI為`--data PATH --out PATH`；人工入口`--cpu-unit-check`（亦`--self-check`）。輸出run.json、model.pt、ratio_predictions.npy、predictions_twd.npy、training.json；checkpoint記actual原類別、configuration、architecture SHA與嚴格CPU tensor key／dtype／shape／finite。完整query ratio及TWD在保存PT模型重載後逐bytes exact，兩NPY另保存重載核對。工程失敗保留，既有output拒絕覆寫，無自動重試、CPU fallback或降參數。

## 作者人工檢查與待辦

先驗27項人工CPU checks：固定balanced folds、same-project可跨fold、held label/date invariance、原fullstate/query exact、五個complement日期權重、30支持與29-row global、invalid neutral、exact-half、拒絕錯labels／indices，以及父系source／RULES前後不變。

Trainer41項人工CPU checks：原weighted MAE值與autograd／factor-scaled ratio梯度、無batch重正規化、原bias公式及F連動、九key讀取前守衛、144／23／dtype／finite／vocabulary、原seed tensor初始化、Softplus、CPU PT完整query ratio／TWD重載、錯checkpoint拒絕與覆寫拒絕。零optimizer step、真fit、GPU、真價格／query y。

第一次trainer人工測試用sum(w)=n的權重，不能區分mean(w×error)與sum(w×error)/sum(w)，因此自查失敗；已保留[準備失敗](../outputs/research_continue_20261008/direction_review_b/R117_preparation_failure_attempt1.json)。只把該合成檢查的權重乘2，原loss alias及科學設定不改。這不是模型或真資料fit失敗。

最終作者命令／耗時／SHA與父系before/after收據另存`outputs/research_continue_20261008/direction_review_b/R117_author_selfcheck.json`。作者自查不能取代獨立preflight。待driver重播原四起點、核八array不變／query exact／原新bias，再完整凍結；本文件沒有真實新先驗、prepared、模型、MAE或CSV。v22、v25與旁白保持，Private未知。
