# 程式與執行順序

[predict.py](predict.py)是目前預測的統一入口，實際讀取[設定檔](../config/predict.json)後，呼叫已驗證的六模型推論程式。它不改變原模型或融合算法。

```powershell
python src/predict.py --describe
```

查看資訊只需Python標準庫；正式預測需要保存模型與兩套套件環境，步驟見[執行指南](../docs/USAGE.md)。下面按照製作過程列出原始程式位置，方便對照報告閱讀。

| 步驟 | 程式 | 工作 |
| --- | --- | --- |
| 1. 資料清理 | [prepare_data.py](../research/src/prepare_data.py)、[model_common.py](../research/src/model_common.py) | 前者讀取資料並排除誤植表頭；後者完全去重與合併訓練資料 |
| 2. 整理特徵 | [house_features.py](../research/src/house_features.py) | 解析日期、面積、樓層與房屋類別等條件 |
| 3. 文字與數值編碼 | [round47_nn_encode.py](../research/src/round47_nn_encode.py)、[round54_numeric_encode.py](../research/src/round54_numeric_encode.py) | 在訓練資料建立規則，再套用於驗證／測試資料 |
| 4. 模型訓練 | 下方六模型表 | 學習各種價格表示；參數與前置資料依模型而異 |
| 5. 時間驗證 | [round107_108_research.py](../research/src/round107_108_research.py) | 以較早資料訓練，對較晚資料產生預測並評分 |
| 6. 三路融合與預測 | [predict_v25.py](../inference/v25/source/src/predict_v25.py)、[tree_runtime.py](../inference/v25/source/src/tree_runtime.py) | 六模型輸出換算總價、三路融合、參考與空間修正、取中位數 |
| 7. 輸出CSV | [submission_io.py](../inference/v25/source/src/submission_io.py) | 檢查編號順序、將總價取整數、寫出提交格式 |

## 六個模型的原訓練來源

| 模型 | 原入口 |
| --- | --- |
| CatBoost直接總價 | [train_final.py](../research/src/train_final.py) |
| LightGBM單價 | [train_final.py](../research/src/train_final.py)的LightGBM分支 |
| CatBoost對數總價 | [round46_trial_release.py](../research/src/round46_trial_release.py) |
| CatBoost日期加權單價 | [round79_final_unit.py](../research/src/round79_final_unit.py) |
| 面積神經網路 | [round55_final.py](../research/src/round55_final.py)、[round48_nn_train.py](../research/src/round48_nn_train.py) |
| 參考單價神經網路 | [round107_full_release.py](../research/src/round107_full_release.py)、[round107_weighted_smooth_train.py](../research/src/round107_weighted_smooth_train.py) |

以上是原始訓練來源，保留當時資料路徑、檔案指紋與前置狀態。完整重訓仍需準備原始資料並依序重建各模型的特徵與狀態；目前未完成從全新下載到六模型全部重訓的整套驗證。詳細參數、來源與缺少的前置條件見[訓練導覽](../inference/v25/training/README.md)，不能只把上表當成可直接逐行執行的命令。

## 保存模型的預測流程

`predict.py`先檢查設定與封存程式身分，再把私人路徑傳給原推論入口。實際執行時會核對原測試資料及17項模型／轉換狀態，建立測試特徵，重新運算四個樹模型及兩個神經網路，最後輸出預測CSV。`--check-inputs`只做執行前路徑檢查，不載入模型或讀取交易表內容。
