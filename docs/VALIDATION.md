# 本次入口核對

2026-10-14完成設定入口的11項檢查：包括未知／缺少／空白／重複設定、裝置名稱、模式互斥、相對路徑、必要輸入及拒絕覆寫。

另透過`src/predict.py`讀取本機JSON設定，在既有ML／NN環境實際載入六個保存模型產生47,073筆預測。程序exit 0，實測75.971秒，輸出CSV與原選定答案逐bytes一致，沒有新增訓練或操作Kaggle。[彙總收據](entry_validation.json)

這次驗證沿用既有套件環境，沒有重新安裝全部套件或重訓模型。先前兩套新環境安裝證據見[原推論紀錄](../inference/v25/README.md)。來源清單可由`python scripts/verify_source.py`核對。
