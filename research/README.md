# 原始訓練程式與重現界線

src/保留379支歷史研究程式，原始路徑與SHA256見source_manifest.json；後續R85–R98程式另有複製身分證據，沒有回寫歷史清單；當前發布包身分見最外層SOURCE_MANIFEST；本頁內容於發布前核驗，最新發布身分以Git main歷史為準。原碼維持bytes，未刪除歷史SHA檢查或拒絕覆寫保護，不在推論主入口的import路徑內。

| 分量 | 訓練入口／方法 |
| --- | --- |
| 近期單價CatBoost | round79_recency_unit.py、round79_final_unit.py |
| 直接總價CatBoost | train_final.py |
| LightGBM單價 | train_final.py（全量 refit）、train_lightgbm.py（候選與編碼） |
| log CatBoost | round46_recency_log.py、round46_trial_release.py |
| 面積NN | round55_final.py、round48_nn_train.py、round54_numeric_encode.py |
| 先驗倍率NN | round63_final.py、round63_recency_weighted_prior_train.py、round60_recency_price_prior.py |
| 參考成交與空間 | round2_neighborhood.py、round8_spatial_reference.py |
| 三期回測 | round79_experiment.py |

原driver依賴固定資料布局、canonical索引、地理、OOF先驗、SHA及歷史收據；不是複製research/到空資料夾就能從零訓練。R55最初產生v14階段產物，R63產生v17階段產物，v22後續重用native模型。

從已準備好的NPZ重訓單個NN，原始命令介面為：

~~~powershell
<torch-python> research/src/round48_nn_train.py --data <R55-final-prepared.npz> --out <new-model-dir>
<torch-python> research/src/round63_recency_weighted_prior_train.py --data <R63-final-prepared.npz> --out <new-model-dir>
~~~

這是命令介面說明，並非本次已執行命令。需另提供約1.34GB／396.7MB的原prepared，原trainer要求CUDA。它不是raw Excel至v22的完整重訓入口。

v22交付階段實测為../src/predict_v22.py保存模型推論，沒有重新訓練完整六模型鏈、重新下載地理或安裝全新環境。2026-09-28完成六次單價分支CatBoost研究fit，2026-09-29先完成R91/R92六次CatBoost與R89三次CPU趨勢wrapper fit，後續R93另完成三次CatBoost；都不是完整v22從零重訓。不得把原碼保存寫成從零重訓已驗證。

R79 target=y/(A*10000)，weight=(A/100)*saved_date_weight。日期權重2^(-days/365)，合法日期正規化、無效日期1；R79沿用R63保存的float32權重。NN價格先驗採建案分組五折OOF，不是時間OOF。歷史回測必須按切點重建全部fit狀態。

如後續要求從原始四表一鍵重訓，需串接原driver前置狀態並實際跑完，另保存收據。

## 後續固定實驗R85–R88

[完整摘要](../docs/RESEARCH_UPDATES_20260928.md)含各新driver與保存結果。R85／R86各三期新fit；R87／R88各三期CPU後處理評估。全數保留研究、不final、不產CSV。這些來源依賴原工作區的完整歷史artifact與精確SHA，複製程式不等於已取得必要資料或可在乾淨環境重跑。原training source bytes與拒絕覆寫保護仍保留。

## 2026-09-29：R89／R91／R92與R90準備

[該批完整摘要](../docs/RESEARCH_UPDATES_20260929.md)保存三項候選未過門檻的結果與反證。本次新增來源保留原bytes，[複製清單](../evidence/outputs/research_snapshot_20260929/copy_manifest.json)記錄路徑／SHA；原379支source_manifest不變。

| 程式 | 角色 |
| --- | --- |
| [round89_group_trend.py](src/round89_group_trend.py)／[driver](src/round89_experiment.py) | fit-only支持門檻、分群時間delta與完整三分支回測；三期完成未採用 |
| [round91_prior_residual.py](src/round91_prior_residual.py)／[driver](src/round91_experiment.py) | 分組OOF prior offset上的signed總價殘差；三期真fit完成未採用 |
| [round92_shallow_unit.py](src/round92_shallow_unit.py)／[driver](src/round92_experiment.py) | 原日期×面積權重單價樹只改depth6；三期真fit完成未採用 |
| [round90_offset_data.py](src/round90_offset_data.py) | 固定2023-02-11切點的純X/geo/indices準備，已真執行與完整重載 |
| [round90_offset_trees.py](src/round90_offset_trees.py)／[NN helper](src/round90_offset_nn.py) | 新切點四樹＋兩NN介面；R90當時未fit，後於R97完成使用 |

R90純X資料沒有載價格表；NN helper的合成OOF／encoder檢查並非新窗口真訓練。R97正式使用時，canonical fit_indices用於OOF群身分，fit_source_cache_indices只用來抽取raw cache標籤，兩者不能混用。所有來源仍依賴原工作區的完整私有資料／狀態，快照不等於可在空目錄直接重訓。

## 歷史追加：R93建案密度權重

[R93研究摘要](../docs/RESEARCH_UPDATES_20260929_R93.md)保留三期、兩對照與全部分群反證。僅用fit建案筆數q=n^(-1/2)，以c=sum(w0)/sum(w0*q)保留原Pool總權重；其他target／參數／特徵與組合保持。門檻未過，不新增CSV；對v22多數配對區間跨零，不能稱全面顯著退步。

- [round93_project_density_unit.py](src/round93_project_density_unit.py)：fit-only群計數、sum-preserving權重及原單價模型wrapper。
- [round93_experiment.py](src/round93_experiment.py)：三期來源／權重preflight、真fit、全query重載與完整組合評估。
- [round93_review.py](src/round93_review.py)：讀取完成结果／診斷／paired，整理報告和不可變採用審查。
- [本批原bytes複製清單](../evidence/outputs/research_snapshot_r93_20260929/copy_manifest.json)：舊379支manifest不改，各次发布包身分由SOURCE_MANIFEST記錄。

R93完成時僅有[Attention可行性研究](../evidence/reports/attention_regression_feasibility_after_r93.md)；這份歷史提案保留原時序。後續實際完成情形見下一節。

## 歷史追加：Attention R95 與零訓練 R96

[完成研究摘要](../docs/RESEARCH_UPDATES_20260929_ATTENTION.md)保存兩候選完整三期、分群、兩對照及獨立審查。R95只替换原prior NN，新增三次真NN fit共1,762.376秒；R96僅改reference信任及其傳給spatial的confidence，0 fit。兩者都未通過原固定門檻，沒有全量模型或新CSV。

|程式|角色|
|---|---|
|[Attention trainer v2](src/round94_attention_prior_train_v2.py)|固定168 tokens／d64／2blocks／4heads；同窗fit encoder、OOF prior、日期總價MAE|
|[原 trainer](src/round94_attention_prior_train.py)|保留v1；v2只硬化NPZ先驗keys的讀取契約，CONFIG不變|
|[R95 driver](src/round95_experiment.py)／[review](src/round95_review.py)|三期新NN、保存重載、原控制整鏈重播及完整集成診斷|
|[R94 failed driver](src/round94_experiment.py)|真fit前因JSON sidecar契約停止，0 fit；R95是同科學配置工程恢復|
|[R96 helper](src/round96_reference_shrink.py)／[driver](src/round96_experiment.py)／[review](src/round96_review.py)|原趨勢保留，局部confidence固定折半，重新套spatial後median，無新fit|
|[九假設彙整](src/summarize_research_after_r95.py)|讀完成聚合結果與雜湊，不重新fit或評分；R96另為第十個對照|

上游固定commit及完整原碼／metadata保留，[歸屬與改編說明](upstream/README.md)與[Apache-2.0授權全文](../docs/third_party/rtdl_LICENSE.txt)隨快照。CLS末層裁切、Softplus回歸頭與本題損失並非完整論文復刻。相同batch保存重載已驗，不把GPU固定seed稱為可跨硬體位元重現。

本批[準備58檔](../evidence/outputs/research_snapshot_attention_20260929_preparation/copy_manifest.json)與[完成複製清單](../evidence/outputs/research_snapshot_attention_20260929_final/copy_manifest.json)分開保留。原379支manifest及source bytes不改；各次发布包身分由最外層SOURCE_MANIFEST記錄。所有原driver仍依賴完整本機私有資料、模型、prior与歷史狀態，快照不含這些檔案；從零完整重訓未驗證。

R90在上述批次仍只有pure-X/helper；後續R97已完成v22／v24七native與全部新fit狀態，沒有加入Attention。舊文件保留當時時序，不能再當未完成待辦。

## 歷史追加：R97 完整新切點重建

[R97完成摘要](../docs/RESEARCH_UPDATES_20260929_OFFSET.md)保存四窗口比較、獨立評分、來源、工程修正及全部反證。固定2023-02-11起點，新fit 272,117／query 107,169列；5樹＋2NN共7次fit呼叫1,100.391秒，包含於main 1,410.656秒。新窗v22較低3,917.46元，但原三年v24都較低；Private未知、提交仍v22。

|程式|用途|
|---|---|
|[fit標籤擷取](src/round97_prepare_fit_labels.py)|只輸出fit列；原pickle需整表反序列化的界線明記|
|[主driver](src/round97_offset_experiment.py)|串行七native、四base與新R/S、保存重載、先封存預測|
|[獨立scorer](src/round97_offset_evaluate.py)|封存／來源／索引先核對，再讀query y；固定MAE、分群與2,000次配對|
|[完成後審查](src/round97_independent_review.py)|220唯一SHA、四base／兩median及4區間核對，不重新fit或score|
|[摘要v2](src/summarize_round97_stability_v2.py)／[原摘要](src/summarize_round97_stability.py)|只讀聚合結果；v1字型警告保留，v2 ASCII減號与完整群反證，不重新評分|

[本批複製清單](../evidence/outputs/research_snapshot_r97_20260929/final/copy_manifest.json)保存原bytes與明確排除项；模型、交易表、labels、encoder／prior／ramps／category states、逐筆預測及bootstrap陣列只保留本機身分。語法失敗原稿在evidence內，標註不可執行；不是訓練入口。現有來源仍依賴完整私有工作區，不把本機新窗口重建稱為全新環境重訓最終提交模型。

R97結束時，時間外L1 stacker只完成[有限盤點](../evidence/reports/regression_direction_after_r97_inventory.md)；後續R98執行情形見下一節。379原source_manifest與正式交付不改；當前發布包身分以外層SOURCE_MANIFEST及Git main歷史為準。

## 最新追加：R98 較早 forward 預測的 L1 組合

[R98 完成研究頁](../docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)記錄三個較晚窗口、各自合法的前置 meta 樣本、純數值求解、全預測封存後獨立評分，以及所有有利／不利分群。三次 CPU meta-fit 完成，0 原生模型 fit／載入與 0 GPU；整體差 +8,798.64／−4,255.03／+5,190.67 元，研究門檻未過，沒有新 CSV。

|程式|用途|
|---|---|
|[資料準備](src/round98_prepare_inputs.py)／[共用契約](src/round98_contract.py)|合法 forward 來源、canonical／raw mapping、日期邊界与每期只輸出 meta 標籤|
|[L1 helper](src/round98_l1_stacker.py)|非負／和為1／無截距的等權總價 MAE，等價 dual LP 與數值證書；不 clip 或 normalize|
|[每期 driver](src/round98_temporal_stack.py)|精確11陣列／自身stage路徑，獨立程序、求解／保存重載、先封存query預測|
|[scorer](src/round98_temporal_evaluate.py)|三期全封存核對後才讀query y；固定MAE／分群／12配對區間|
|[保存結果稽核](src/round98_independent_review.py)|233唯一SHA、9保護檔、公式與保存區間；不讀真價、不重做LP或評分|
|[摘要v2](src/summarize_round98_v2.py)／[原摘要](src/summarize_round98.py)|只讀聚合結果；保留首圖重疊與修訂收據，v2圖已目視核對，不重評分|

稽核作者亦為helper作者，故不稱完全獨立重作求解器；scorer與前置審查由其他代理處理。三個窗口不獨立，auxiliary以前forward block仍缺；本輪不是原三個8月窗口的完整採用門檻。多輪選型風險與Private未知保留。

[四欄含median的下一方向](../evidence/reports/round98_next_direction.md)只是一份有限提案，沒有新計畫／fit。原始資料、所有逐筆陣列、模型state與solver arrays不在本包；只有來源、聚合metadata及身分。原379來源清單不回寫，正式v22與交付報告不改。

[本批完成複製清單](../evidence/outputs/research_snapshot_r98_20260929/final/copy_manifest_v2.json)及[前置收據](../evidence/outputs/research_snapshot_r98_20260929/preparation/copy_manifest.json)分開保留；當前发布身分仍以 Git main 與外層 SOURCE_MANIFEST 為準。
