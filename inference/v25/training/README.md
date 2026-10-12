# 六個基底模型的原始訓練來源

此批補上原 main `2fcf5cda93c1ff82d63c5b36d5aa2a87a70a57ce` 缺少的 22 支原始訓練／資料準備程式，以及 2 份原設定與 schema。它們依 [allowlist](source_additions_manifest.json)逐 bytes 保留於指定的 research/src 與 research/config。與既有檔案合計覆蓋盤點列出的 73 支保守 static source closure；這只證明來源可取得，尚未證成 raw→全部六模型重新訓練的完整執行鏈。

| 原生模型 | 原始訓練入口 | 命令證據 |
| --- | --- | --- |
| 直接總價 C_raw | [train_final.py](../../../research/src/train_final.py) | 由來源重建 CLI；所讀原 final_training 未保存完整 argv／OS exit 收據 |
| 單價 LightGBM L_unit | [train_final.py](../../../research/src/train_final.py) | 同一 driver 的 lightgbm_l1_unit63；不能換成 R2 final_spatial63 |
| 對數總價 C_log | [round46_trial_release.py](../../../research/src/round46_trial_release.py) | 原 execution.json 記 argv 與 exit 0；此次未重跑 |
| 日期加權單價 C_unit | [round79_final_unit.py](../../../research/src/round79_final_unit.py) | 原 execution.json 記 argv 與 exit 0；round79_final.py 是組合程式 |
| 面積網路 N_area | [round55_final.py](../../../research/src/round55_final.py)／[round48_nn_train.py](../../../research/src/round48_nn_train.py) | 原 child execution.json 記完整命令與 exit 0；依賴原 R55 prepared |
| 參考單價網路 N_prior | [round107_full_release.py](../../../research/src/round107_full_release.py)／[round107_weighted_smooth_train.py](../../../research/src/round107_weighted_smooth_train.py) | argv 由來源與 R63 prepared 路徑重建；原 fit_execution 記 exit 0，未直接存 argv |

[six_models.json](six_models.json)保存每個模型的完整參數、原命令證據類型、prepared 生成來源與前置依賴。[audit.json](audit.json)及[完整盤點](AUDIT.md)說明範圍。這些是原時點與本次只讀盤點的證據，不能將其中原 main 缺檔描述誤當此次草稿補齊後的檔案清單，也不能把歷史命令當成已在新 clone 成功執行的命令。

六個原生模型的原 metadata 各記 467,221 筆 fit。此數字有逐模型證據，但本次未打開實際 row-ID／label 陣列，不能只憑相同筆數保證全部訓練列逐 bytes 相同。原 C_raw／L_unit 訓練收據沒有 query_rows 欄位，後續 47,073 筆推論證據不能回填成該原收據也記錄了 query 筆數。

原 driver 保留所有歷史 SHA、完成收據、固定 ROOT／路徑與拒絕覆寫保護。它們仍可能依賴私有 prepared NPZ、合法 fit 價格狀態及保存預測向量；不能刪掉 guard 後宣稱已提供可攜一鍵訓練。兩份 config/schema 位於 research/config，是原樣快照，未自動接到 driver 原來的 models/final_config.json 等路徑。

完整重訓還需由合法課程 raw 表重建清理與特徵、非價格地理對齊、encoder、group OOF prior、日期權重，並在新路徑實際訓練六個 native，再由新 fit 標籤與模型輸出重建 reference／spatial。保存的價格 state 與預測不能冒充這些新 fit。此次沒有執行上述流程、載入模型、評分或修改已選 CSV；Private仍未知。
