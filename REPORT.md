# 臺灣房屋總價預測：近期成交加權與多模型回歸

## 摘要

本研究將房屋成交總價預測建模為監督式回歸問題，使用課程提供的交易屬性與價格標籤，結合梯度提升樹、表格神經網路、訓練資料中的可比成交，以及非價格道路地理特徵。本文以提交版本 v22（實驗編號 R79）為主要研究交付對象。v22 的關鍵改動是對單價 CatBoost 同時使用面積權重與 365 日半衰期日期權重，再將各模型輸出還原為總價，經參考成交與空間修正後，逐筆取三條分支的中位數。

v22 在三個年度時間回測中，相對 v20 的整體 MAE 分別降低 2.65%、0.62% 與 1.94%。Kaggle Public 回報 MAE 為 **1,998,568.5706 元**，較 v20 降低約 3.44%。但部分高價群的誤差上升，歷史窗口也已反覆參與模型選擇。v24 在三期本機整體 MAE 均低於 v22，卻未在已回報的 Public 結果中取得更低誤差。因此，本報告將 v22 視為目前具競爭力的主要候選，不宣稱已證明其 Private 表現最佳。[E1–E5]

## 1. 問題定義與評估方法

輸入為每筆交易的房屋、土地、位置及交易時間等屬性，輸出為總價元。這是連續數值的回歸預測，不是價格區間分類。主要評估指標為平均絕對誤差：

$$
\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|.
$$

其中 $y_i$ 為實際總價，$\hat y_i$ 為預測總價，單位皆為新臺幣元。MAE 越小代表平均金額誤差越低。Public MAE 約 199.86 萬元，表示該評分集合的平均絕對誤差約為此金額，不是預測準確率，也不代表每筆交易都偏差同一金額。

Kaggle Public 約使用 test 的 30%，Private 使用其餘約 70%。此比例描述的是 test 的評分分割，與另外提供的 valid 檔案不同。Private 真實分數尚未知。本文分開呈現年度離線 MAE、平台回報 Public MAE 與未知 Private，避免混用不同集合的數字。[E1、E5]

## 2. 資料來源與前處理

### 2.1 原始資料與使用範圍

| 檔案 | 原始列數 | 用途 |
| --- | ---: | --- |
| `train-v2.xlsx` | 455,787 | 模型訓練與歷史時間回測 |
| `valid-v2.xlsx` | 11,994 | 最終全量訓練時併入的有標籤資料 |
| `test-reindex-test-v2.2.xlsx` | 47,073 | 無價格標籤的預測對象 |
| `test-reindex-test-sample-submission-v2.csv` | 47,073 | 提交欄位與編號順序範本 |

價格標籤與參考成交均來自課程提供的資料。研究遵守禁止到內政部不動產成交案件實際資訊資料供應系統核對答案的限制，未取得外部 test 成交真價。sample submission 中的價格僅是格式範例，不能當作答案。

後期模型另使用既有 OpenStreetMap／Geofabrik 道路地理快照，這些資料不含外部成交價格。因此不能將 v22 描述成只使用四份課程原始檔。地理來源標示為 © OpenStreetMap contributors，原來源聲明、快照 URL、檔案大小與 SHA256 均保留。[E6、E12]

課程交易資料的公開再散布權限未在本研究中確認。程式交付應將資料取得方式與必要檔名分開說明，不因程式整理至 GitHub 就假定可一併公開原始交易表。道路資料的來源與授權聲明亦須保留。

### 2.2 清理順序

| 步驟 | 移除列數 | 保留列數 |
| --- | ---: | ---: |
| 原始 train | — | 455,787 |
| 移除混入資料區的英文表頭 | 3 | 455,784 |
| 排除編號、包含目標價格的整列完全去重 | 529 | 455,255 |
| 併入 valid | 增加 11,994 | 467,249 |
| 移除合併後跨集完全重複 | 28 | **467,221** |

上述去重不是「相同編號只留一列」。年度回測僅使用 train 中早於當期起點的資料，不把 valid 加入回測的 fit。test 保留全部 47,073 列，不刪列、不去重，輸出依 sample 的編號順序對齊。[E1、E6]

數值與日期處理包括民國日期、中文樓層、地址、交易筆棟數與數字文字解析，並保留適當的缺失及異常旗標。類別缺值以 `__MISSING__` 表示。編號與目標總價不作模型特徵。建築完成年月因資料不可用而排除，管理組織因為常數而排除。完整備註與解約情形不納入，避免交易後資訊或文字價格進入模型，僅保留有限的非價格備註旗標。[E1、E6]

## 3. 特徵設計

| 類型 | 內容與目的 |
| --- | --- |
| 面積與交易組成 | 土地、建物及車位面積、面積比例、交易筆棟數，描述交易規模 |
| 建物與格局 | 樓層、總樓層、建物型態、用途、建材、房廳衛等，描述使用型態 |
| 時間 | 交易日期解析後的時間特徵，描述市場時序變化 |
| 類別與地址 | 地區、道路、建案等資訊，供類別模型及價格參考流程使用 |
| 道路地理 | 道路代表位置、長度、範圍、可用性、片段數、匹配狀態及網格 |

共同基礎特徵為 87 欄。R79 單價 CatBoost 加入 8 欄既有地理特徵後，共 **95 欄，其中 23 欄為類別特徵**。此數字只適用於該模型，不代表所有分支的輸入維度相同。面積 NN 與先驗倍率 NN 另有數值編碼與類別 embedding。[E1、E7、E8]

地理特徵使用固定年度快照，歷史時期各有對應來源。道路代表點是粗略位置，不等同住宅的精確門牌座標。地理狀態建立不使用 query 價格。[E7、E12]

## 4. 模型方法

### 4.1 近期加權的單價 CatBoost

令 $y$ 為總價，$A$ 為有效面積。$A$ 優先取正且有限的建物面積，否則取正且有限的車位面積，仍無可用面積時取 1。此規則用於數值縮放，不宣稱估計了實際室內面積。

```python
target_unit = y / (A * 10_000)
sample_weight = (A / 100) * date_weight
predicted_total = np.maximum(predicted_unit * A * 10_000, 0)
```

日期權重以 fit 的最大合法交易日為基準，原始權重為：

$$
w_i^{\mathrm{date,raw}}=2^{-\Delta t_i/365}.
$$

合法日期的權重再除以 fit 合法日期原始權重的平均值，無效日期採中性值 1。因此，在同一 fit 中相差一年的合法交易，其日期權重比例為 1:2。R79 使用已保存且列序一致的 R63 `float32` 日期權重，再轉為 `float64` 與面積權重相乘，不重新計算或再次正規化。[E7]

在非負裁切之前，有以下等價關係：

$$
\frac{A_i}{100}w_i^{\mathrm{date}}
\left|\frac{y_i}{10,000A_i}-z_i\right|
=\frac{w_i^{\mathrm{date}}}{1,000,000}|y_i-10,000A_i z_i|.
$$

因此，面積權重使單價誤差對應到總價誤差，日期權重則提高近期成交在訓練目標中的比重。這並未將評估指標改成百分比誤差。相對 v20，R79 的主要改動是在原有面積權重上乘入上述日期權重。

| 參數 | 設定 |
| --- | --- |
| 模型與 loss | `CatBoostRegressor`、`MAE` |
| iterations / depth | 2,999 / 8 |
| learning rate | 0.065 |
| L2 leaf regularization | 6.0 |
| random seed | 42 |
| boosting type | Plain |
| bootstrap / subsample | Bernoulli / 0.85 |
| random strength | 0.5 |
| 運算 | GPU、4 threads |

此模型沒有使用 query 標籤作訓練、`eval_set` 或 early stopping。完整參數以保存的 model metadata 與程式設定為準。[E7]

### 4.2 神經網路與其他樹模型

面積 NN 與價格先驗倍率 NN 皆使用 embedding MLP，隱藏層為 512→256→128，搭配 LayerNorm、ReLU、Dropout 0.1 與 Softplus 正值輸出。訓練設定為 30 epochs、batch size 2048、seed 42 與 AdamW，初始 learning rate 0.001。[E8]

- 面積 NN 由單價輸出按面積還原總價，使用總價 MAE。其數值輸入為 603 維，包含分段線性編碼，另有類別 embedding。
- 價格先驗倍率 NN 以區域單價先驗為基準學習調整倍率，使用近期加權的總價 MAE。其數值輸入為 144 維，另有類別 embedding。
- 價格先驗訓練採五折建案分組 OOF，避免本建案標籤直接進入其訓練先驗。這是建案分組 OOF，不是時間 OOF，不能宣稱每筆訓練資料的先驗都只來自其成交日以前。
- log CatBoost 以 `log1p(y / 1_000_000)` 為目標，還原方式為 `maximum(expm1(z) * 1_000_000, 0)`。其訓練 loss 是近期加權的 log 空間 MAE，完整組合才以元的 MAE 評估。
- 原有 LightGBM 單價預測與直接總價 CatBoost 亦保留在分支中，所有組合均使用已還原為元的預測。[E8、E9]

### 4.3 三分支組合

下式所有模型符號皆代表已還原為總價元的預測。`C_unit` 為 R79 單價 CatBoost，`L_unit` 為 LightGBM 單價還原總價，`C_raw` 為直接總價 CatBoost，`C_log` 為 log CatBoost 還原總價。`N_area` 與 `N_prior` 分別為上述兩個 NN。

```text
B_unit = 0.75 * C_unit + 0.25 * L_unit
B_nn   = 0.75 * (0.5 * N_area + 0.5 * N_prior) + 0.25 * L_unit
B_log  = 0.75 * (0.5 * C_raw  + 0.5 * C_log)   + 0.25 * L_unit

U = S(B_unit, R(B_unit))
N = S(B_nn,   R(B_nn))
H = S(B_log,  R(B_log))

prediction_v22 = median(N, H, U)  # 逐筆取中位數
```

`R` 為訓練資料內可比成交與時間趨勢參考，實際使用 `round2_neighborhood.predict_reference` 的 `variant='trend'`。`S` 為固定空間修正，使用 `round8_spatial_reference.predict` 與保存的 R39 狀態，在參考支持較弱時補充空間資訊，支持不足則回退。這些價格參考來自 fit，不是查詢外部 test 真價。[E9]

各分支先修正，最後才取中位數。不能先取中位數再重複修正，也不能在已還原總價後再乘面積。中位數可降低單一分支極端輸出的影響，但不保證消除共同偏差。[E9]

## 5. 驗證設計與結果

### 5.1 時間切分

每個回測以該期起點以前的 canonical train 資料作 fit，再以其後一年作 query。模型、encoder、prior、reference 與 spatial 狀態必須對應該次 fit。不能拿後期已看過 query 的模型評估較早年度。[E2]

| 期別 | query 期間，含起訖日 | fit 列數 | query 列數 |
| --- | --- | ---: | ---: |
| 較早年度 | 2021-08-11～2022-08-10 | 134,805 | 105,511 |
| 中間年度 | 2022-08-11～2023-08-10 | 240,316 | 81,234 |
| 最新年度 | 2023-08-11～2024-08-10 | 321,550 | 133,697 |

歷史程式將上述期別命名為 auxiliary、development 與 confirmation，但這些窗口已多次參與研究，不能因為命名為 confirmation 就視為未見過的新盲測。

### 5.2 年度整體 MAE

| query 期間 | v20 MAE | v22 MAE | v22 相對 v20 降低 | v24 MAE |
| --- | ---: | ---: | ---: | ---: |
| 2021-08-11～2022-08-10 | 2,372,150.72 | 2,309,352.36 | 2.65% | **2,275,409.14** |
| 2022-08-11～2023-08-10 | 1,416,140.05 | 1,407,387.07 | 0.62% | **1,407,340.72** |
| 2023-08-11～2024-08-10 | 1,410,654.84 | 1,383,228.41 | 1.94% | **1,373,195.92** |

全部 MAE 單位為元。v22 對 v20 的三期整體點估計都改善，支持日期加權作為候選方案。然而，v24 的三期整體與固定加權 MAE 點估計都比 v22 低。此反證必須與 Public 結果一起保留，不能宣稱 v22 在所有本機評估中最好。[E2、E3]

R79 對 v20 另以建案重抽樣及三個月循環區塊重抽樣計算配對區間。每種方式使用 2,000 次重抽樣。以整體 MAE 差值「v22 減 v20」為例：

| 年度 | 建案重抽樣的 95% 區間 | 三個月區塊的 95% 區間 |
| --- | ---: | ---: |
| 較早 | [-83,546.32, -40,840.43] | [-84,148.21, -49,070.97] |
| 中間 | [-31,419.13, +14,340.18] | [-31,351.44, +4,747.05] |
| 最新 | [-46,351.02, -10,140.17] | [-48,524.60, -8,874.84] |

中間年度的區間跨 0，因此不能說三期皆有一致的區間支持。上述區間未校正反覆選型造成的偏差，也不是 Private 的勝率。[E2]

### 5.3 Kaggle Public 與未知 Private

| 提交版本 | Public MAE（元） | Private MAE |
| --- | ---: | --- |
| v19 | 2,387,305.3199 | 未知 |
| v20 | 2,069,837.8954 | 未知 |
| v21 | 2,071,041.3042 | 未知 |
| **v22** | **1,998,568.5706** | **未知** |
| v23 | 2,026,961.4299 | 未知 |
| v24 | 2,083,045.2932 | 未知 |

數值來自保存的 Kaggle 提交成績截圖及其登錄 JSON。v22 是這六份回報中的 Public 最低值，相對 v20 減少 71,269.3248 元，約 3.4432%。本機 CSV 的雜湊有保存，但未獨立向平台驗證實際上傳的檔案 bytes，因此表格應描述為「平台截圖回報」，而非宣稱已獨立核實平台全部輸入。[E5]

## 6. 方法比較與取捨

本研究保留候選的有利與不利結果，方法比較以完整最終組合為主，不將單一分支的改善直接等同於最終模型的改善。

| 比較 | 固定改動 | 觀察與取捨 |
| --- | --- | --- |
| v20 → v22 | 單價 CatBoost 加入既有日期權重，其他分支保持 | 三期整體 MAE 都降低，但中間年度區間跨 0，部分高價群退步 |
| v22 → v23 | 單價 base 由 75% CatBoost＋25% LightGBM 改為純 CatBoost | 三期變化不一致，保留為對照 |
| v23 → v24 | 單價 base 改為 seed 42／137 CatBoost 各半 | 三期整體相對 v23 改善，對 v22 也有整體優勢，但有群組代價 |
| v24 → R84 | 固定兩個 seed，單價樹 iterations 由 2,999 增至 4,499 | 最新年度改善，另兩期退步，保留研究而未產生新提交 |

v24 的實際單價 base 為 `0.5*C42 + 0.5*C137`，不再包含原來的 25% LightGBM 項。NN 與 log 的完整分支仍沿用原配方，最後一樣取中位數。因此不能把 v24 簡化成「只替換 v22 的 CatBoost 為兩個 seed，仍保留 75%／25%」。[E3]

v22 最新年度真價 5,000 萬至 1 億群共有 1,725 筆，群內 MAE 對 v20 增加 **204,728.26 元**。另一方面，v24 在最新年度真價至少 1 億的 319 筆資料中，群內 MAE 對 v22 增加約 **909,727.59 元**，中間年度的 test 組成重加權估計亦比 v22 高約 9,367.11 元。這些是保留 v22 作主要候選的風險取捨，並非 v22 普遍較好的證明。[E2、E3]

R84 相對 v24 的整體 MAE 差值為：較早年度 +7,114.59 元、中間年度 +930.08 元、最新年度 −1,999.31 元。最新年度的配對區間支持改善，是 R84 的有利證據，但另外兩期與高價群的代價使其未被採用為新提交。這不表示增加樹數在所有情況下都無效。[E4]

上述是真實保存候選的比較，不是完整因子實驗。不同層級的 MAE 差值不能相加成各模組的因果貢獻。GPU 重訓亦非完全位元決定，固定 seed 不能排除全部訓練隨機性。[E2–E4]

## 7. 過擬合、洩漏與適用限制

1. **時間分離與狀態分離。** 年度 query 標籤不能進入同一期模型或價格參考。新切點必須重建對應的模型、encoder、prior、reference 與 spatial state，不能只替換評分索引。
2. **OOF 的界線。** 建案分組先驗避免直接使用本建案標籤，但不是時間 OOF。不能擴張成每筆訓練先驗均滿足嚴格的逐筆歷史可得性。
3. **窗口重用偏差。** 多輪研究重複使用同一批歷史期間，模型選擇可能對這些窗口適應。bootstrap 不會自動消除此偏差。
4. **Public 適應風險。** 公開榜僅代表約三成 test，不能據此推定 Private 最佳，也不應反覆搜尋混合比例、seed 或逐筆價格來追逐 Public。
5. **高價尾部風險。** 大面積物件與少量高價交易會放大總價外推。v22 的最大未取整 test 預測約 138.716 億元，這是模型輸出，不是已核實成交價。中位數組合也可能保留多分支共有的偏差，未據未知 test 真價加設逐筆價帽。[E10]
6. **群組診斷不等於推論規則。** 以真實價格分群只能用於有標籤回測的事後分析，不能在 test 推論時使用未知真價分流。
7. **組成重加權不是真值。** testmix 僅在有歷史支持的城市與新舊建案群內重新加權，無法觀察 test 真實誤差，不等同 test 或 Private MAE。
8. **粗略地理與資料範圍。** 道路特徵並非精確門牌定位，缺失地址、新建案或市場狀態改變仍可能導致分布差異。

## 8. 程式與重現範圍

### 8.1 實際方法入口

| 功能 | 保存程式 |
| --- | --- |
| 讀取與清理 | `prepare_data.py`、`model_common.py` |
| 基礎特徵 | `house_features.py`、`prepare_features.py` |
| 地理與單價表示 | `round6_geo.py`、`round19_unit_geo.py` |
| R79 近期加權訓練 | `round79_recency_unit.py`、`round79_final_unit.py` |
| NN 編碼與訓練 | `round47_nn_train.py`、`round48_nn_train.py`、`round54_numeric_encode.py`、`round63_recency_weighted_prior_train.py` |
| 日期權重與 OOF 先驗 | `round60_recency_price_prior.py` |
| log 模型 | `round12_log_geo.py`、`round46_recency_log.py` |
| 參考成交及空間修正 | `round2_neighborhood.py`、`round8_spatial_reference.py` |
| 三分支組合 | `round79_final.py` |
| 提交檔獨立核驗 | `round79_release.py` |

這些檔案具有跨輪相依，並非複製單一 `round79_final.py` 就能從原始 Excel 重建全部模型。交付應分清下列三種模式：

| 模式 | 實際工作 | 能證明什麼 |
| --- | --- | --- |
| 保存向量重組 | 讀取已保存分支，固定融合並輸出 CSV | 指定向量的組合與輸出一致性 |
| 保存模型推論 | 重載模型、encoder 與價格參考狀態，對指定資料重新預測 | 該組狀態可重播推論 |
| 從原始資料重訓 | 清理、地理、編碼、先驗、全部模型與參考狀態皆重建 | 所測環境下的完整方法可執行 |

保存的 R79 歷史證據已包含單價模型重載及 47,073 筆 raw／TWD 預測一致性核對。最終組合另保存未取整向量，CSV 已核對完整 ID、順序、欄名、列數、正整數、UTF-8 BOM 與一次取整結果。這些檢查支持指定產物鏈的可追溯性，不等於已完成乾淨環境從原始 Excel 重訓全部 v22 的驗證。[E7、E10、E11]

提交價格只做一次 `maximum(rint(prediction), 1).astype(int64)`，輸出兩欄 `編號,總價元`。原 v22 提交檔 SHA256 為：

```text
6490e2da78d4bf8173a27444de673a020152878f78e7ebec54d4fdc9bddfae35
```

本報告不把未執行的重訓流程記為完成。交付入口為 src/predict_v22.py，從原始 test Excel、保存地理及 fit 狀態重新推論六個模型。操作、環境與驗證範圍見 README.md 及 docs/REPRODUCIBILITY.md。[E11]

### 8.2 本次交付的實際重現測試

在獨立交付目錄保存必要模型副本，以 Python 隔離模式執行單一入口。原始 test Excel 重新生成 87 欄基礎特徵，encoder、quantile knots 與 prior 重新生成 NN query 輸入，再真正載入四個樹模型與兩個 NN，重算各分支參考／空間修正及逐筆中位數，沒有將舊預測向量當成推論輸入。

| 核對項目 | 實測結果 |
| --- | --- |
| 全量預測列數 | 47,073 |
| 六模型與三分支 | 與原保存輸出完全位元一致 |
| 最終未取整預測 | 最大絕對差 0 元 |
| 最終 CSV | 完整 SHA256 與原 v22 相同 |
| 單一入口執行時間 | 84.74 秒，不含先前模型複製及元件稽核 |
| 提交格式單元測試 | 4 項通過 |
| 本次新增訓練 | 0 次 |

實測為 Windows／Python 3.12.14 的既有 ML 與 Torch 兩環境，NN 為 Torch 2.13.0+cu126／CUDA 12.6／RTX 4060 Ti。CPU 面積 NN 的預測與原 GPU 有差異，平均絕對預測差約 565.28 元；這不是 test MAE。精確重現須匹配受測計算設定，不能承諾所有硬體位元一致。

首次整合已得到相同預測，但 CSV 使用 LF 而原檔為 CRLF，因此完整檔案雜湊核對失敗。修正換行及失敗收據狀態後，重新執行整條入口並通過。原失敗收據與更正說明均保留。這項格式修正沒有改動任何模型或價格。

此次驗證是「原始 test 加保存地理／模型／fit 狀態的推論重現」，尚非從原始 train／valid 重訓全部模型，也沒有重新建立地理資料或測試全新環境安裝。保存的必要 artifact 約 1.45 GB，留在本機私有目錄，公開程式包不包含原交易表、模型與價格參考狀態。

入口與命令見 [README](README.md)，驗證收據見 [重現摘要](evidence/reproduction/summary.json)及[完整收據](evidence/reproduction/integration_verified.json)。

## 9. 結論

v22 以單價表示、面積對齊的總價誤差、近期成交加權及三分支組合，取得比 v20 更低的三期歷史整體 MAE，也得到目前回報候選中最低的 Public MAE。它適合作為本次程式與技術報告的主要研究版本。

研究結果仍有明確界線：v24 的本機整體表現形成反證，部分高價群退步，年度窗口反覆參與選型，而 Private 尚未知。後續若進行研究，應固定 v22／v24 配方做時間起點穩定性比較，並補足清楚界定範圍的重現測試，而非單靠 Public 反覆調參。本次報告的完成不等於最終 Private 候選已定案。

## 附錄：證據與程式來源

下表顯示原研究來源名，連結指向本交付包內的副本。報告數字來自保存的結果及正式成績回報，未以未完成實驗填補。交付包可依 manifest 對應來源副本。

| 編號 | 支持內容 | 來源路徑 |
| --- | --- | --- |
| E1 | v22 方法、清理及程式導覽 | [reports/v22_method_and_code.md](evidence/reports/v22_method_and_code.md) |
| E2 | R79 三期 MAE、分群與配對區間 | [outputs/round79/results.json](evidence/outputs/round79/results.json)、[reports/round79_report.md](evidence/reports/round79_report.md)、[outputs/round79/paired_vs_v20/results.json](evidence/outputs/round79/paired_vs_v20/results.json) |
| E3 | v24 配方、三期優勢及群組代價 | [reports/round83_report.md](evidence/reports/round83_report.md)、[reports/round83_delivery.md](evidence/reports/round83_delivery.md)、[src/round83_final.py](research/src/round83_final.py) |
| E4 | 增加樹數的結果及未採用決策 | [reports/round84_report.md](evidence/reports/round84_report.md)、[outputs/round84/decision.json](evidence/outputs/round84/decision.json) |
| E5 | Public v19–v24、未知 Private 與 CSV 身分 | [outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json](evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json) 及同名 `.png` |
| E6 | 原始資料與前處理實作 | [src/prepare_data.py](research/src/prepare_data.py)、[src/model_common.py](research/src/model_common.py)、[src/house_features.py](research/src/house_features.py)、[src/prepare_features.py](research/src/prepare_features.py) |
| E7 | 單價、日期權重、CatBoost 設定及重載 | [src/round19_unit_geo.py](research/src/round19_unit_geo.py)、[src/round79_recency_unit.py](research/src/round79_recency_unit.py)、[src/round79_final_unit.py](research/src/round79_final_unit.py)、[outputs/round79/final_model/model/metadata.json](evidence/outputs/round79/final_model/model/metadata.json) |
| E8 | NN 架構、編碼、日期權重與先驗 | [src/round47_nn_train.py](research/src/round47_nn_train.py)、[src/round48_nn_train.py](research/src/round48_nn_train.py)、[src/round54_numeric_encode.py](research/src/round54_numeric_encode.py)、[src/round60_recency_price_prior.py](research/src/round60_recency_price_prior.py)、[src/round63_recency_weighted_prior_train.py](research/src/round63_recency_weighted_prior_train.py) |
| E9 | log 轉換、參考修正與最終融合 | [src/round12_log_geo.py](research/src/round12_log_geo.py)、[src/round46_recency_log.py](research/src/round46_recency_log.py)、[src/round71_final.py](research/src/round71_final.py)、[src/round46_trial_release.py](research/src/round46_trial_release.py)、[src/round2_neighborhood.py](research/src/round2_neighborhood.py)、[src/round8_spatial_reference.py](research/src/round8_spatial_reference.py)、[src/round79_final.py](research/src/round79_final.py) |
| E10 | 最終預測與 CSV 核驗 | [outputs/round79/final/trial_release.json](evidence/outputs/round79/final/trial_release.json)、[outputs/round79/final_export/release.json](evidence/outputs/round79/final_export/release.json)、[src/round79_release.py](research/src/round79_release.py) |
| E11 | 重現範圍與既有缺口 | [reports/github_reproducibility_audit_r83.md](evidence/reports/github_reproducibility_audit_r83.md)、[DELIVERY_REQUIREMENTS.md](evidence/DELIVERY_REQUIREMENTS.md) |
| E12 | 外部非價格地理資料的來源與快照 | [data_external/geography/ATTRIBUTION.md](evidence/data_external/geography/ATTRIBUTION.md)、`data_external/geography/*.download.json`、[README_round6.md](evidence/README_round6.md) |

