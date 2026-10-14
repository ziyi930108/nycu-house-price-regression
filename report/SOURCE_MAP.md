# 報告來源對照

正式報告的數據來自課程資料、保存的驗證結果與選定紀錄。價格案例只來自已知答案的歷史驗證資料，沒有測試資料的真價。下列連結對應正文引用編號。

| 引用 | 來源 | 對照內容 |
| --- | --- | --- |
| [1] | [表頭排除](../research/src/prepare_data.py)、[資料集合與去重](../research/src/model_common.py)、[特徵解析](../research/src/house_features.py) | 表頭排除、完全去重及房屋條件整理 |
| [2] | [神經網路編碼](../research/src/round47_nn_encode.py)、[參考單價](../research/src/round60_recency_price_prior.py)、[固定訓練](../research/src/round107_weighted_smooth_train.py) | 僅用訓練資料建立規則、分組OOF、日期權重及SmoothL1 |
| [3] | [三年度結果](../evidence/outputs/round107/results.json)、[額外驗證期間](../evidence/outputs/round113/results.json) | 各期MAE、分群退步與重抽樣限制 |
| [4] | [單一網路曲線](source/assets/training_curve.png)、[30輪數據](../evidence/outputs/v25_training_curve_20261010/curve.json)、[曲線說明](../evidence/reports/v25_training_curve_20261010.md) | 固定設定的另一次診斷訓練，不是原提交軌跡或完整融合曲線 |
| [5] | [選定紀錄](../evidence/selection/user_selected_v25_20261010.json) | 已選定方法、使用者回報Public 1,995,836.2136元；Private未知 |
| [6] | [最終預測組合](../research/src/round107_final_candidate.py)、[發行收據](../evidence/outputs/round107/final/release.json) | 原全量階段僅新增一個參考單價網路，其餘五模型沿用保存預測 |
| [7] | [城市統計](source/assets/city_statistics.json) | 清理後455,255筆訓練交易、城市筆數與總價分布；前三縣市269,520筆占59.2% |
| [8] | [完整驗證彙總](source/assets/validation_summary.json)、[真價與估價圖](source/assets/fusion_validation.png)、[價格帶誤差圖](source/assets/error_by_price_band.png) | 133,697筆完整融合驗證的整體、分群、價格帶及大誤差交易占比；不附逐筆預測列表 |
| [9] | [六模型推論說明](../inference/v25/README.md)、[兩套環境推論收據](../inference/v25/evidence/reproducibility_receipt.json)、[原訓練來源](../inference/v25/training/README.md) | 保存狀態推論的驗證範圍；原始資料到六模型完整重訓仍未驗證 |
| [10] | [CatBoost官方介紹](https://catboost.ai/docs/en/)、[類別特徵文件](https://catboost.ai/docs/en/features/categorical-features) | 梯度提升決策樹與類別特徵處理 |
| [11] | [LightGBM官方Features](https://lightgbm.readthedocs.io/en/stable/Features.html) | 數值分箱、逐葉生長與類別特徵處理 |

圖表及數據皆沿用正式稿，未重新分析交易資料或改變模型。各驗證期間的訓練與驗證交易按日期隔離，但歷史期間已反覆參與方法比較，不能當作全新獨立考試。各價格帶只在預測完成後依已知驗證真價分組；案例是刻意選出的困難交易，原因解釋仍屬待驗證假設。

同建案訓練筆數是描述性統計，不等於修正程式實際採用的參考交易筆數。模型推論重現與重新訓練是不同驗證；同機新套件環境成功，不保證其他硬體、系統或套件版本完全相同。Public不代表未知Private的表現。

公開來源只含圖表、彙總數據及文件重建材料；不含課程原始交易表、模型權重、提交CSV或逐筆預測清單。Word與PDF身分及複製核對見[交付彙總](delivery.json)。
