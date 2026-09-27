# submission_v22_trial.csv：方法與程式碼導覽

整理日期：2026-09-27。這份說明以實際保存的程式、模型 metadata、發布收據及使用者提供的 Kaggle 截圖為依據。本次只整理說明與新增分數紀錄，未重新訓練或修改預測。

**v22 是提交版本名稱；R79 是產生它的實驗編號。** 它使用 CatBoost、LightGBM 與表格神經網路組成三條價格分支，經訓練資料的鄰近成交與時間趨勢修正後，逐筆取三條分支的中位數。

## 1. 這次 Kaggle 成績

| 提交版本 | 截圖中的 Public MAE（元） |
| --- | ---: |
| v19 | 2,387,305.3199 |
| v20 | 2,069,837.8954 |
| v21 | 2,071,041.3042 |
| **v22** | **1,998,568.5706** |
| v23 | 2,026,961.4299 |
| v24 | 2,083,045.2932 |

v22 是這六個版本中 Public MAE 最低者。意義是公開評分那部分測試資料的「預測總價與實際總價之絕對差」平均約 **199.86 萬元**；不是準確率，也不表示每筆都相差這個金額。相對 v20，MAE 少 71,269.3248 元，降低約 3.4432%。

分數來源是使用者截圖，沒有另登入平台核對上傳檔案 bytes。Public 約涵蓋 30% test；其餘約 70% 的 Private 分數仍未知。這次結果支持 v22 的 Public 表現，但不能由此證明 Private 最佳。截圖與本機 CSV 身分另存於 [本次分數紀錄](../outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json)，沒有改寫舊發布收據中當時尚未知的 Public 欄位。

## 2. 資料如何處理

1. 原始 train 有 455,787 列，移除混入資料區的 3 列英文表頭，得到 455,784 列。
2. 按「除編號以外的所有原欄位，包含目標價格」去除 529 列完全重複資料，得到 455,255 列；不是看見相同 ID 就刪除。
3. 最終訓練合併 11,994 列 valid，再移除 28 列跨集重複，得到 **467,221 列**。年度回測則各自只用其起點以前的 train，不將 valid 混入。
4. test 保留 **47,073 列**，不去重、不刪列，輸出沿 sample 的編號順序。
5. 解析民國日期、中文樓層、地址、交易筆棟數；數字文字轉數值，保留缺失與異常旗標，類別缺值用 `__MISSING__`。建立面積比例、格局、樓層、交易時間及有限備註旗標。
6. 編號和總價不作輸入特徵；建築完成年月在資料中不可用、管理組織為常數，因此排除；解約情形及完整備註也不納入，避免交易後資訊或文字價格進入模型。

共同基礎為 87 欄特徵。R79 單價 CatBoost 再加入 8 欄既有道路地理特徵，共 **95 欄，其中 23 欄類別**。地理欄包括位置、道路長度／涵蓋範圍、位置是否可用、道路片段數、匹配狀態與網格；道路位置不等同精確門牌定位。不同分支的編碼方式及輸入維度不同，不能把「95 欄」套用到所有神經網路。

地理資料沿既有道路地圖流程；價格標籤與鄰近成交參考使用提供的訓練資料，沒有到內政部實價登錄系統核對 test 答案。原始檔案保留。

對應程式：[原始資料讀取](../src/prepare_data.py)、[去重與資料集合](../src/model_common.py)、[基礎特徵](../src/house_features.py)、[特徵快取與檢查](../src/prepare_features.py)、[地理特徵](../src/round6_geo.py)。詳細歷史規則見 [資料處理紀錄](round3_preprocessing_record.md)；該頁也列有未採用的嘗試，不能全部當成 v22 已使用的方法。

## 3. v22 相對 v20 的主要改動

R79 保留 v20 的其他分量，重新訓練一支單價 CatBoost，將原來的面積權重乘上既有的近期成交權重。

設 `y` 為總價，`A` 為有效面積：優先採正且有限的建物面積，否則採正且有限的車位面積，再無法使用才取 1。這只作數值縮放，不是估計室內實際面積，也不改寫原面積特徵。

```python
target_unit = y / (A * 10_000)
sample_weight = (A / 100) * date_weight
predicted_total = np.maximum(predicted_unit * A * 10_000, 0)
```

以上為公式說明；實際實作在 [round19_unit_geo.py](../src/round19_unit_geo.py) 與 [round79_recency_unit.py](../src/round79_recency_unit.py)。

日期權重來自該次 fit 的交易日期，以最大合法交易日為基準，採 **365 天半衰期**。有效日期的原始權重為 `2 ** (-距基準日天數 / 365)`，再以有效 fit 列的平均值正規化；缺失／異常日期採中性權重 1。相差一年的兩筆有效日期，其日期權重比例為 1:2。R79 使用 R63 已保存、列序核對過的 float32 日期權重，再轉 float64 乘面積權重，不另重算或重新正規化。

面積權重的用途是讓單價學習對齊總價誤差。在非負輸出裁切前：

```text
(A/100) × date_weight × |y/(A×10,000) − predicted_unit|
= date_weight × |y − predicted_total| / 1,000,000
```

因此它保留單價表示法，同時以近期成交的總價誤差為重點；不是把總價 MAE 改成百分比誤差。

R79 單價模型的實際主要設定：

| 參數 | 值 |
| --- | --- |
| 模型／目標 | CatBoostRegressor／MAE |
| iterations | 2,999 |
| depth | 8 |
| learning_rate | 0.065 |
| l2_leaf_reg | 6.0 |
| random_seed | 42 |
| boosting_type | Plain |
| bootstrap_type／subsample | Bernoulli／0.85 |
| random_strength | 0.5 |
| 運算 | GPU、4 threads |

沒有在這一支使用 query 標籤、eval_set 或 early stopping。完整設定及實際欄名在 [模型 metadata](../outputs/round79/final_model/model/metadata.json)，保存模型為 [model.cbm](../outputs/round79/final_model/model/model.cbm)。

下面摘出訓練核心，省略前後的列序、雜湊及保存檢查；它不是可單獨執行的完整程式：

```python
transformed = transform_target(raw, X)
dates32 = _date_values(saved_date_weights, X)
area_weights = original.sample_weights(X)
weights = np.ascontiguousarray(
    area_weights * dates32.astype(np.float64), dtype=np.float64
)
state = fit_feature_state(X, geo, cats)
augmented = augment_features(X, geo)
model = CatBoostRegressor(**parameters())
model.fit(Pool(
    augmented, transformed, weight=weights,
    cat_features=state['categorical_columns'], thread_count=4
))
```

來源：[round79_recency_unit.py 的 fit_model](../src/round79_recency_unit.py#L78)。日期權重來源：[round60_recency_price_prior.py](../src/round60_recency_price_prior.py#L49)；全量訓練與權重來源核對：[round79_final_unit.py](../src/round79_final_unit.py)。

## 4. 三條分支如何組合

以下所有 `C`、`L`、`N` 都指已換算回「元」的預測，不能直接混合 log、單價或神經網路倍率輸出。

- `C_unit`：上述 R79 近期加權單價 CatBoost。
- `L_unit`：原有 LightGBM 單價模型換回總價的預測。
- `C_raw`：原有直接總價 CatBoost。
- `C_log`：R46 近期加權、對數總價 CatBoost，預測後換回元。
- `N_area`：面積連結的表格神經網路，學單價並按面積還原總價。
- `N_prior`：以區域成交單價先驗為基準、學價格調整倍率的表格神經網路，採近期加權總價 MAE。訓練用先驗由建案分組的 OOF 建立，避免直接把本列價格放進自己的先驗。

令 `R` 表示既有鄰近成交參考與時間趨勢修正，`S` 表示固定的空間修正。下式省略共同的特徵與保存狀態引數；空間修正同時使用原基準 `B` 與參考結果 `R(B)`：

```text
B_unit = 0.75 × C_unit + 0.25 × L_unit
B_nn   = 0.75 × (0.5 × N_area + 0.5 × N_prior) + 0.25 × L_unit
B_log  = 0.75 × (0.5 × C_raw  + 0.5 × C_log)   + 0.25 × L_unit

U = S(B_unit, R(B_unit))
N = S(B_nn, R(B_nn))
H = S(B_log, R(B_log))

v22_prediction = median(N, H, U)   # 每筆交易各自取中位數
```

兩個 NN 共用 PyTorch embedding MLP 架構：隱藏層 512→256→128，LayerNorm、ReLU、Dropout 0.1，Softplus 正值輸出；30 epochs、batch 2048、seed 42、AdamW，初始 learning rate 0.001。架構程式為 [round47_nn_train.py](../src/round47_nn_train.py#L15)。面積 NN 使用 603 維數值輸入（含分位節點的分段線性特徵），先驗倍率 NN 使用 144 維數值輸入；兩者另有類別 embedding。編碼見 [round47_nn_encode.py](../src/round47_nn_encode.py)、[round54_numeric_encode.py](../src/round54_numeric_encode.py)。

先驗的五折 OOF 是按建案分組，並非時間 OOF；它避免本建案標籤直接進入其訓練先驗，但不能被描述成所有先驗都只使用該筆成交當日以前的資料。年度 query 標籤不參與當期 fit 的先驗建立。

log CatBoost 的目標是 `log1p(y / 1,000,000)`，逆轉為 `max(expm1(z) * 1,000,000, 0)`。其原生 loss 為近期加權的 **log 空間 MAE**，不是直接的總價 MAE；完整組合仍以還原成元的總價 MAE 評估。目標轉換見 [round12_log_geo.py](../src/round12_log_geo.py#L103)，日期加權訓練見 [round46_recency_log.py](../src/round46_recency_log.py)。

這是三條分支，分支內仍包含多個模型。中位數的作用是減少單一分支極端輸出的影響；它不能保證避免過擬合，也可能擋掉某一分支的正確改善。

`R` 實際為 [round2_neighborhood.predict_reference](../src/round2_neighborhood.py) 的 `variant='trend'`：使用 fit 的同建案、地址、道路等可比交易，以及 fit 估計的時間趨勢。`S` 為 [round8_spatial_reference.predict](../src/round8_spatial_reference.py)，使用 R39 保存狀態補充較弱的參考來源，缺少支持時回退。這些價格參考來自 fit，不是上網查到的 test 真價。

NN 分支的最終組合見 [round71_final.py](../src/round71_final.py#L115)；log 分支見 [round46_trial_release.py](../src/round46_trial_release.py#L243)。v22 的實際最終兩行是：

```python
_, unit = branch(OUT / 'new_unit', .75 * native + .25 * oldunit)
pred = np.median(np.column_stack([nn, log, unit]), axis=1)
```

來源：[round79_final.py](../src/round79_final.py#L153)。這個程式沿用已保存的 NN／log 分支，只替換新單價分支，沒有在組合階段重新訓練全部模型。

## 5. 如何驗證與產生 CSV

R79 先在三個年度，分別用該年度以前的資料 fit、該年度資料評分：2021-08-11～2022-08-10、2022-08-11～2023-08-10、2023-08-11～2024-08-10。它對 v20 的三期總價 MAE 分別降低約 62,798、8,753、27,426 元；所有分群及不利結果保留在 [R79 報告](round79_report.md)。

這些年度已反覆參與研究，不能稱全新盲測。模型保存後重新載入核對全部 47,073 筆 raw／TWD 預測一致，是保存與推論檢查，並不等於證明泛化。

最後將未取整的 v22 總價向量只取整一次：`maximum(rint(pred), 1).astype(int64)`，按 sample 的 ID 順序寫出兩欄 `編號,總價元`。實際寫檔使用保存的 [CSV writer](../outputs/round79/csv_authoring/export.mjs)，再由 [round79_release.py](../src/round79_release.py) 核對全部編號、價格、列數、欄名、BOM 與來源向量；成品為 [submission_v22_trial.csv](../outputs/submission_v22_trial.csv)。

## 6. 程式碼入口與重現狀態

| 用途 | 實際程式 |
| --- | --- |
| 讀 Excel、清理與基礎特徵 | `prepare_data.py`、`model_common.py`、`house_features.py`、`prepare_features.py` |
| 95 欄地理單價目標／還原 | `round6_geo.py`、`round19_unit_geo.py` |
| v22 關鍵訓練改動 | [round79_recency_unit.py](../src/round79_recency_unit.py) |
| 三年度實驗 | [round79_experiment.py](../src/round79_experiment.py) |
| 全量單價模型訓練、保存與重載核對 | [round79_final_unit.py](../src/round79_final_unit.py) |
| NN 訓練與輸入轉換 | [round48_nn_train.py](../src/round48_nn_train.py)、[round54_numeric_encode.py](../src/round54_numeric_encode.py)、[round63_recency_weighted_prior_train.py](../src/round63_recency_weighted_prior_train.py) |
| NN 價格先驗與時間權重 | [round60_recency_price_prior.py](../src/round60_recency_price_prior.py) |
| log CatBoost | [round46_recency_log.py](../src/round46_recency_log.py) |
| v22 三分支組合及匯出資料準備 | [round79_final.py](../src/round79_final.py) |
| CSV 寫入／獨立檢查 | [export.mjs](../outputs/round79/csv_authoring/export.mjs)、[round79_release.py](../src/round79_release.py) |

**程式與模型都有保存，但目前屬跨多輪相依的研究版流程。** `round79_final.py` 需要前面已建立的模型預測、地理快取、reference 與 spatial 狀態；只複製這一個檔案不能從原始 Excel 重跑 v22。歷史入口亦固定了當時的檔案身分與拒絕覆寫檢查，現在直接重送原命令會被保護條件擋下。

目前已驗證的是當時這條產物鏈的訓練、模型重載、預測組合與 CSV 核對，尚未完成乾淨環境的單一入口重建。待使用者確定最終提交版本後，再整理最小相依程式、完整環境、資料／模型 manifest、單一執行入口與實際重現測試供 GitHub 報告使用。這次尚未把 v22 指定為最終 Private 版本，也未發布 GitHub。
