# 原始訓練程式與重現界線

src/保留原研究程式，原始路徑與SHA256見source_manifest.json。原碼維持bytes，未刪除歷史SHA檢查或拒絕覆寫保護，不在推論主入口的import路徑內。

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

本次實測交付為../src/predict_v22.py保存模型推論，沒有重新訓練六模型、重新下載地理或安裝全新環境。不得把原碼保存寫成從零重訓已驗證。

R79 target=y/(A*10000)，weight=(A/100)*saved_date_weight。日期權重2^(-days/365)，合法日期正規化、無效日期1；R79沿用R63保存的float32權重。NN價格先驗採建案分組五折OOF，不是時間OOF。歷史回測必須按切點重建全部fit狀態。

如後續要求從原始四表一鍵重訓，需串接原driver前置狀態並實際跑完，另保存收據。
