# 設定檔

## 執行路徑

[predict.json](predict.json)是 `src/predict.py` 實際讀取的設定，內含示例路徑。先複製為 `config/predict.local.json`，再填入自己的檔案位置；本機設定已被Git排除。

| 欄位 | 意義 |
| --- | --- |
| `artifacts` | 保存六模型、編碼器、參考價格、地理與空間修正檔案的私人根目錄 |
| `test` | 原始測試Excel路徑 |
| `sample` | 範例提交CSV路徑，用來核對編號與順序 |
| `nn_python` | 獨立神經網路環境的Python執行檔；預設`.venv-nn/Scripts/python.exe` |
| `device` | 預設`cuda:0`。原入口也接受`cpu`，但CPU結果與選定答案的一致性未驗證 |
| `output_dir` | 尚不存在且位於公開儲存庫以外的新輸出資料夾 |

所有相對路徑皆從儲存庫根目錄計算。Windows絕對路徑可使用正斜線，例如 `D:/house-price-private/data/test.xlsx`。未知欄位、重複欄位或空字串會被拒絕，以免打錯設定卻悄悄沿用其他值。`--describe`不需檔案存在；`--check-inputs`會檢查路徑，但完整檔案指紋仍在正式推論時核對。

## 模型設定的來源

路徑設定與模型參數分開保存。模型結構、學習率、日期權重與融合方式屬於原已完成訓練的方法；本入口載入保存模型，不以新的參數重新訓練。

| 設定內容 | 實際來源與讀取方式 |
| --- | --- |
| 六模型推論、神經網路結構、執行批次、執行緒與固定答案身分 | [推論設定](../inference/v25/source/config/v25.json)，由原推論程式直接讀取並核對SHA |
| 六模型原訓練參數與前置條件 | [six_models.json](../inference/v25/training/six_models.json)，作為來源與參數導覽；原訓練入口另見[src說明](../src/README.md) |
| 早期總價與LightGBM模型的設定快照 | [original_v1_final_config.json](../research/config/original_v1_final_config.json)，保留原設定，尚未自動接至原driver預期的路徑 |
| 原始特徵欄位規格 | [original_feature_schema.json](../research/config/original_feature_schema.json) |
| 時間驗證期間、各期訓練與評估 | [年度流程](../research/src/round107_108_research.py)、[各期結果](../evidence/outputs/round107/results.json) |

日期權重與融合比例的設計理由見[正式報告](../report/README.md)。封存來源具有原SHA與拒絕覆寫檢查；若要研究不同參數，需另建實驗並重建相應訓練狀態，不能就地改掉核對條件。
