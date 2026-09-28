# 原始訓練程式與重現界線

src/保留379支歷史研究程式，原始路徑與SHA256見source_manifest.json；後續R85–R93程式另有複製身分證據，沒有回寫歷史清單；本批發布前還需刷新最外層SOURCE_MANIFEST。原碼維持bytes，未刪除歷史SHA檢查或拒絕覆寫保護，不在推論主入口的import路徑內。

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
| [round90_offset_trees.py](src/round90_offset_trees.py)／[NN helper](src/round90_offset_nn.py) | 新切點四樹＋兩NN介面；真窗口模型訓練未執行 |

R90純X資料沒有載價格表；NN helper的合成OOF／encoder檢查並非新窗口真訓練。若未來啟用，canonical fit_indices用於OOF群身分，fit_source_cache_indices只用來抽取raw cache標籤，兩者不能混用。所有來源仍依賴原工作區的完整私有資料／狀態，快照不等於可在空目錄直接重訓。

## 最新追加：R93建案密度權重

[R93研究摘要](../docs/RESEARCH_UPDATES_20260929_R93.md)保留三期、兩對照與全部分群反證。僅用fit建案筆數q=n^(-1/2)，以c=sum(w0)/sum(w0*q)保留原Pool總權重；其他target／參數／特徵與組合保持。門檻未過，不新增CSV；對v22多數配對區間跨零，不能稱全面顯著退步。

- [round93_project_density_unit.py](src/round93_project_density_unit.py)：fit-only群計數、sum-preserving權重及原單價模型wrapper。
- [round93_experiment.py](src/round93_experiment.py)：三期來源／權重preflight、真fit、全query重載與完整組合評估。
- [round93_review.py](src/round93_review.py)：讀取完成结果／診斷／paired，整理報告和不可變採用審查。
- [本批原bytes複製清單](../evidence/outputs/research_snapshot_r93_20260929/copy_manifest.json)：舊379支manifest不改，發布前另刷新SOURCE_MANIFEST。

後續Attention僅[可行性研究](../evidence/reports/attention_regression_feasibility_after_r93.md)，沒有新trainer或已測模型；R90仍尚無新切點六native fit。研究程式依賴原工作區歷史artifacts，不能把來源保存或單模型重載當全新環境完整重訓。
