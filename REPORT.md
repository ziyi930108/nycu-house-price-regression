# v25 房屋總價回歸：目前選定方法與程式導覽

整理日期：2026-10-10。本文依保存的程式、訓練 metadata、離線結果、發行收據，以及使用者這次的選定與分數回報撰寫。本次只整理方法，沒有重訓或修改 v25 的價格。

**使用者已於 2026-10-10 選定 v25 作為提交資料。** v25 的研究編號是 R107；它沿用原本的六模型、三分支架構，只把其中一個「價格先驗神經網路」的訓練損失改成日期加權 SmoothL1。舊版交付已移出目前主分支，歷史結果保留於比較證據及 Git 歷史；原始 release 記錄的是發行當時尚未選定的狀態，不能回寫成當時就已選定。[最新選定紀錄](evidence/selection/user_selected_v25_20261010.json)、[原 v25 發行收據](evidence/outputs/round107/final/release.json)。

## 1. 要預測什麼，以及目前實際成績

這是監督式回歸：輸入房屋、土地、車位、位置及交易時間等資料，輸出每筆交易的總價元。官方指標是新臺幣元的平均絕對誤差（MAE）：

```text
MAE = 所有評分交易的 |預測總價 − 實際總價| 加總 ÷ 交易筆數
```

| 提交檔 | Public MAE（元） | 證據與角色 |
| --- | ---: | --- |
| v22 | 1,998,568.5706 | 先前使用者截圖回報；原選定基準，現在保留作對照 |
| **v25** | **1,995,836.2136** | 2026-10-10 使用者文字回報；目前選定提交 |

v25 的 Public MAE 比 v22 少 **2,732.3570 元，約降低 0.1367%**。這個分數表示公開評分交易的平均絕對總價誤差約 **199.58 萬元**；不是準確率，也不是每間房屋都差這個金額。這次分數來自使用者文字回報，沒有新截圖，也沒有另登入 Kaggle 核對實際上傳檔案的 bytes。[v25 分數紀錄](evidence/kaggle_feedback/public_v25_user_report_20261010.json)、[v22 原分數證據](evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json)。

Public 約涵蓋 test 的 30%，Private 約涵蓋其餘 70%；這不是把課程 valid 切成三成與七成。**Private 仍未知**。目前 Public 有小幅進步，與使用者改選 v25 的決定一致，但不能把它寫成已證明 Private 最佳。

## 2. 資料從原始表格走到模型

| 步驟 | 實際處理 | 為什麼這樣做 |
| --- | --- | --- |
| 原始 train | 455,787 列，去除混入資料區的 3 列英文表頭，剩 455,784 列 | 避免把欄名當作交易樣本 |
| train 完全去重 | 按除編號外的全部原欄位，包含目標價格，移除 529 列完全重複，剩 **455,255 列** | 避免同一完整紀錄被重複計入；不是只按相同 ID 刪除 |
| 最終訓練集 | 合併 11,994 列 valid，移除 28 列跨集完全重複，得到 **467,221 列** | 最終提交使用所有可用的合法標籤；年度回測則不混入 valid |
| test | 保留 **47,073 列**與 sample ID 順序，不去重、不刪列 | 每個測試 ID 都必須有一筆價格 |
| 欄位清理 | 解析民國日期、中文樓層、交易筆棟數、地址及數字文字；保留缺失、異常旗標與有限備註旗標 | 讓模型使用明確數值及結構，同時保留「原資料不完整」這件事 |
| 欄位排除 | 編號與總價不作特徵；建築完成年月不可用、管理組織為常數，排除；不納入解約情形與完整備註 | 避免識別碼記憶、無效欄位、交易後資訊及文字價格進入模型 |

基礎特徵共有 87 欄。使用道路地理資訊的模型再加入 8 欄，得到 95 欄，其中 23 欄為類別；地理資訊包含道路位置、道路範圍與匹配狀態等。它不是精確門牌定位。各模型的輸入表示不同，不能把「95 欄」等同所有神經網路的最終輸入維度。

prior NN 將 72 個數值欄位按 **fit 資料**的中位數、尺度與 `asinh` 轉換標準化，再各保留一個非有限值旗標，形成 **144 個數值輸入**。23 個類別欄位轉成 embedding；fit 中出現不到 5 次或 query 未見過的類別使用 0 編碼，不在 query 重新學詞表。另一個面積 NN 使用含分位節點的分段線性數值編碼，數值輸入為 603 維。

價格標籤、價格先驗與鄰近成交參考都來自課程提供的訓練資料；既有道路地理流程依保存來源使用非價格資訊。沒有到內政部不動產成交案件實際資訊資料供應系統核對 test 答案。

程式入口：[原始資料讀取](research/src/prepare_data.py)、[資料集合與去重](research/src/model_common.py)、[基礎特徵](research/src/house_features.py)、[特徵準備](research/src/prepare_features.py)、[道路地理特徵](research/src/round6_geo.py)、[prior NN 編碼](research/src/round47_nn_encode.py)、[面積 NN 編碼](research/src/round54_numeric_encode.py)。

## 3. 六個模型，為什麼組成三條分支

房價同時受面積、區域價格、建案及時間影響，因此保留不同表示法的模型。以下每個模型的輸出都先換回**新臺幣總價元**，才進行組合。

| 原生模型 | 學習方式與用途 |
| --- | --- |
| `C_unit`：R79 CatBoost | 學單價，以有效面積乘回總價；訓練權重為面積權重乘日期權重，讓單價誤差對齊近期總價誤差 |
| `L_unit`：LightGBM | 另一個單價模型，換回總價後作為三分支的共同分量 |
| `C_raw`：CatBoost | 直接學總價，保留原金額尺度的訊號 |
| `C_log`：R46 CatBoost | 學 `log1p(總價 / 1,000,000)`，逆轉回總價；其訓練 loss 是 log 空間 MAE，不能稱直接總價 MAE |
| `N_area`：面積 NN | 學與面積連結的價格表示，再還原總價 |
| **`N_prior`：R107 prior NN** | 以區域／交易與建物類型價格先驗及面積為基準，學正值價格調整倍率；這是 v25 唯一替換的模型 |

有效面積 `A` 優先採正且有限的建物面積，否則採正且有限的車位面積，再無法使用才取 1。它是數值縮放的回退規則，不是估計實際室內面積，也沒有改寫原面積特徵。

prior NN 的概念是：先建立「這類地點和房屋大約多少單價」的參考，再讓神經網路學調整倍率：

```text
price_factor_million = A × fit 建立的參考單價 / 1,000,000
prior NN 總價（元） = 正值倍率 × price_factor_million × 1,000,000
```

訓練列使用按建案分組的五折 OOF 先驗，避免把本建案標籤直接放入自己的訓練先驗；query 則用完整 fit 標籤建立的先驗。這是**建案分組 OOF，並非時間 OOF**，不能說每筆訓練先驗都只來自該筆成交當日以前。每個回測起點的 query 標籤均不參與該期先驗及模型的 fit。[先驗與日期權重](research/src/round60_recency_price_prior.py)。

六個模型形成三條固定分支：

```text
B_unit = 0.75 × C_unit + 0.25 × L_unit
B_nn   = 0.75 × (0.5 × N_area + 0.5 × N_prior_R107) + 0.25 × L_unit
B_log  = 0.75 × (0.5 × C_raw  + 0.5 × C_log)        + 0.25 × L_unit

U = S(B_unit, R(B_unit))
N = S(B_nn,   R(B_nn))
H = S(B_log,  R(B_log))

v25_prediction = median(N, H, U)  # 每筆交易各自取中位數
```

`R` 使用 fit 中同建案、地址、道路等可比交易，以及 fit 估計的時間趨勢修正價格。`S` 再依固定空間狀態補充參考較弱的情況；支持不足時回退。空間修正同時接收原本 `B` 與 `R(B)`，不是把兩次完全相同的校正相加。[R 實作](research/src/round2_neighborhood.py)、[S 實作](research/src/round8_spatial_reference.py)。

中位數使某一條分支的極端輸出較不容易直接決定最後價格，但也可能擋掉某個分支的正確改善。它不能保證避免過擬合。v25 不重新搜尋混合比例，也不重新 fit R／S；它只重算新的 NN 分支 R／S **各一次**，再與原本完整 log、unit 分支取中位數。[全量組合實作 `apply_prior`](research/src/round107_final_candidate.py)。

## 4. v25 唯一訓練改動：日期加權 SmoothL1

原 R63 prior NN 使用日期加權的總價絕對誤差。R107 保留九個 prepared 陣列、先驗、encoder、日期權重、初始化、架構及推論，唯一改動是訓練損失：

```python
per_row = torch.nn.functional.smooth_l1_loss(
    ratio_output * train_price_factor_million,
    train_y_million,
    beta=0.1,
    reduction="none",
)
loss = (train_sample_weight * per_row).mean()
```

這是訓練核心摘錄，不是可單獨執行的完整程式。目標、預測及 `beta` 都在「百萬元」尺度，因此 `beta=0.1` 代表 **10 萬元**，不是 0.1 元。

設誤差 `e` 的單位為百萬元，SmoothL1 在 `|e| < 0.1` 時使用 `e² / (2 × 0.1)`；在較大誤差時使用 `|e| − 0.1 / 2`。設計想法是：接近正確價格時把梯度變平滑，較大誤差時仍保持絕對誤差的線性尾部。它不是刪除高價房或異常交易，也不是改官方評分。官方仍用總價 MAE，因此損失變平滑是否有幫助，必須看保留的離線與 Kaggle 結果。

日期權重沿既有的 **365 天半衰期**：相差一年的有效成交日期，原始權重比例為 1:2；以該次 fit 的有效日期平均正規化，缺失／異常日期採中性權重 1。R107 直接使用原 prepared 中的正值 float32 權重，不重新計算、不按每個 batch 的權重和重新歸一。理由是近期價格可能較能代表待預測時期，但過強重視近期也可能減少有效歷史訊息。

| 保留設定 | 值 |
| --- | --- |
| 數值／類別輸入 | 144／23 |
| 隱藏層 | 512 → 256 → 128；LayerNorm、ReLU |
| Dropout／category dropout | 0.1／0.1 |
| 輸出 | Softplus 正值倍率 |
| epochs／batch／seed | **30／2,048／42** |
| optimizer | AdamW；learning rate 0.001、weight decay 0.0001 |
| scheduler | CosineAnnealingLR，最低 learning rate 0.00001 |
| 模型選擇 | 固定最後一個 epoch，不用 query y、eval_set 或 early stopping 選 epoch |

`beta=.1` 沿先前 R62 的固定值，沒有掃 beta；R62 已研究過未加日期權重的 SmoothL1，因此不能宣稱「平滑 loss 從未試過」。這次檢驗的是它與 R63 日期權重及既有完整組合的交互。R108 的 CatBoost CTR 改動是另一個未通過候選，**沒有混進 v25**。[訓練來源](research/src/round107_weighted_smooth_train.py)、[事前規約](evidence/reports/round107_108_plan.md)、[保存的全量設定](evidence/outputs/round107/final/model/training.json)。

## 5. 時間驗證與不利證據

先使用當時已存在的三個年度起點；每期只用起點以前的 train fit，該年 query 留作評分。三窗所有新模型與向量封存後，才讀 query 標籤評分。後續 R113 再固定檢查原本已有的 2 月起點，使用該起點自己的模型、encoder、prior、reference 與 spatial，沒有拿後期或全量模型回測。[三窗流程](research/src/round107_108_research.py)、[額外起點計畫](evidence/reports/round113_plan.md)。

| query 時期 | fit 列數 | query 列數 | v22 MAE（元） | v25 方法 MAE（元） | 差：v25 方法 − v22 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021-08-11～2022-08-10（auxiliary） | 134,805 | 105,511 | 2,309,352.36 | 2,310,236.02 | **+883.66** |
| 2022-08-11～2023-08-10（development） | 240,316 | 81,234 | 1,407,387.07 | 1,405,971.19 | −1,415.88 |
| 2023-08-11～2024-08-10（confirmation） | 321,550 | 133,697 | 1,383,228.41 | 1,381,702.87 | −1,525.54 |
| 2023-02-11～2024-02-10（額外起點） | 272,117 | 107,169 | 1,336,410.67 | 1,336,107.62 | −303.06 |

「v25 方法」指該起點各自 fit 的 R107 配方，並非拿最終全量模型去測歷史。負差代表 MAE 降低，正差代表退步；表中結果全部是**離線 MAE**，不能與 Public 混成一張排行榜。

除整體 MAE，另固定按 test 的既有／新建案組成做加權診斷：既有 17,237／47,073，約 36.62%；新建案 29,836／47,073，約 63.38%。使用的是已知特徵及 fit 是否見過該建案，不使用 test 真價。它是補充選型指標，官方 MAE 並未改成此加權值。

| 窗口 | 固定加權 MAE 差 vs v22 | 既有建案 MAE 差 | 新建案 MAE 差 |
| --- | ---: | ---: | ---: |
| auxiliary | **+755.96** | **+393.29** | **+965.48** |
| development | −1,398.03 | −196.46 | −2,092.20 |
| confirmation | −1,723.04 | **+213.10** | −2,841.61 |
| 額外 2 月起點 | −259.51 | −604.49 | −60.21 |

最近年度整體改善約 **0.1103%**，但既有建案小幅退步；較早年度兩個主要指標也都退步。因此 R107 **未通過原強推薦研究條件**（三窗整體及加權全改善，最近整體至少改善 0.5%，且季度或至少 500 列群無超過 5% 退步）。

它通過的是訓練前另外固定的**探索提交條件**：最近整體及加權不退步、各窗兩指標退步不超過 0.25%、至少兩窗整體改善、季度及大型群無超過 5% 退步；額外 2 月起點也通過後續固定的穩定性檢查。這些工程條件允許產出新試交，不能改稱原強條件通過；2026-10-10 的選定是使用者在後續 Public 回報後的決定。[原三窗結果與門檻](evidence/reports/round107_108_report.md)、[原機器結果](evidence/outputs/round107/results.json)、[2 月結果](evidence/reports/round113_report.md)、[2 月機器結果](evidence/outputs/round113/results.json)。

### 必須保留的次對照 v24

| 窗口 | v24 離線 MAE（元） | v25 方法 − v24（元） |
| --- | ---: | ---: |
| auxiliary | 2,275,409.14 | **+34,826.87** |
| development | 1,407,340.72 | −1,369.53 |
| confirmation | 1,373,195.92 | **+8,506.95** |
| 額外 2 月起點 | 1,340,328.14 | −4,220.52 |

v24 在最早及最近年度比 v25 方法好；原三個年度整體及固定加權也都比 v22 好。但 v24 既有 Public 回報為 2,083,045.2932，反而比 v22 差。這是本專案的重要反證：離線窗口較好，不能直接推出 Kaggle 較好；不能因目前選 v25，就刪掉不支持它的對照。

### 對過擬合與不確定性的實際限制

- 保留城市、月份、相對季度、新舊建案與五個真價帶的完整診斷，不只報整體勝出的時期。真價帶只用於有標籤的離線分析，不拿未知 test 真價分流。
- 各起點嚴格分開 fit 與 query 狀態；新起點不能沿用後期 encoder、prior、價格參考或模型。
- 年度窗口已多輪參與研究；2 月起點也早已用過，並與 8 月窗口重疊。它們不是四次獨立盲測，不能當成三比一的獨立投票。
- 每次 prior NN 固定 seed 42；保存 v22 是對照，沒有在本輪同期重新訓練一個 matched v22。小幅差異不等於已分離出 loss 的單獨因果效果，也不包含 GPU 重訓變異。
- 建案重抽樣與三個月循環月份區塊各 2,000 次，都是條件於既有預測的配對 bootstrap，未校正累積選型、重新 fit 或未來分布變動。三個 8 月起點的整體「按建案」95% 區間都跨 0；最近年度為約 **[−3,594.91, +523.35] 元**。2 月起點整體的建案區間約 **[−2,558.33, +2,076.58] 元**，月份區間約 **[−844.28, +163.84] 元**，也都跨 0。不能把重抽樣的負差比例稱為 Private 勝率。

## 6. 最終 v25 如何產生與檢查

探索條件通過並完成獨立審查後，先凍結全量計畫、source／input SHA 與受保護舊產物；重播原 v22 的保存向量組合，逐 bytes 核對原三分支鏈、未取整中位數及 CSV。

全量只新增**一次 R107 prior NN fit**，使用原 R63 final 的九鍵 `prepared.npz`，467,221 fit／47,073 query，固定 30 epochs／seed 42。這不是重新訓練全部六模型。新 NN 的全 query 總價輸出保存後重新載入模型，再推論核對逐 bytes 相同；之後替換 prior 槽、做一次原 R／S、與原 log／unit 分支逐筆取中位數。

全量 native fit 實測 **48.875 秒**，trainer 全程序 **51.468 秒**；含 context、組合與封存的 fit 程序 **80.234 秒**，發行核對程序 **38.468 秒**。這些各自有不同計時範圍，不是整個專案總工時。[訓練 metadata](evidence/outputs/round107/final/model/training.json)、[fit 執行收據](evidence/outputs/round107/final/fit_execution.json)、[五層封存](evidence/outputs/round107/final/seal.json)、[release](evidence/outputs/round107/final/release.json)。

最後只取整一次：

```python
rounded = np.maximum(np.rint(final_TWD), 1).astype(np.int64)
```

按 sample 的 47,073 個 ID 與順序，先用 Artifact Tool typed range 建立表格並核對全部值，再將該範圍以標準 comma／CRLF 編碼寫成 UTF-8 BOM CSV；當時 native CSV API 不可用，因此不是該 API 的直接匯出。Python `csv` reader 再獨立核對全部 ID、整數值、列數、兩欄名稱、順序及全檔 SHA。成品為 `outputs/submission_v25_trial.csv`（原工作區檔案，未納入GitHub）：

```text
SHA256  36231cd17a6cbeafefeb62a7989d207fa20d6c5f7aaf99857e4d9addd25597c1
欄位    編號,總價元
資料列  47,073
```

相對 v22，13,002 列取整價格不同；未取整預測的平均絕對差為 29,521.0062 元。這是**兩個預測向量的差**，不是 test MAE，也不是它對真價改善的金額。這些發行證據沒有使用 test 真價或按 test ID 人為改價。

## 7. 程式怎麼讀，以及目前能重現到哪裡

| 用途 | 來源與實際入口 | 範圍 |
| --- | --- | --- |
| 了解唯一 loss 改動 | [round107_weighted_smooth_train.py](research/src/round107_weighted_smooth_train.py)：`weighted_loss`、`run` | 九鍵 NPZ → 一支 prior NN fit → 保存模型／全 query reload；CLI 有 `--data`、`--out`、`--cpu-unit-check` |
| 了解原三窗研究 | [round107_108_research.py](research/src/round107_108_research.py)：`freeze`、`train`、`stage`、`evaluate` | R107／R108 獨立、共六次 fit；CLI 為 `--freeze`／`--train`／`--stage`／`--evaluate`，不是最終 v25 一鍵推論 |
| 了解全量資料身分與組合 | [round107_final_candidate.py](research/src/round107_final_candidate.py)：`specification`、`load_context`、`apply_prior` | 原 full-fit 資料、R／S 與保存向量的完整身分核對；CLI `--preflight` 是唯讀組合重播，沒有六模型 native 推論 |
| 了解最終發行 | [round107_full_release.py](research/src/round107_full_release.py)：`freeze`、`fit`、`release` | 單次全量 NN fit、五層封存、CSV；CLI `--freeze`／`--fit`／`--release`，綁定當時 manifest 與拒絕覆寫 |
| 了解 2 月後續檢查 | [round113_offset_research.py](research/src/round113_offset_research.py)、[round113_offset_composition.py](research/src/round113_offset_composition.py) | 自己起點的 R107／R111／R112；不產生新的 v25 CSV |
| 了解共用架構／先驗 | [round47_nn_train.py](research/src/round47_nn_train.py)、[round47_nn_encode.py](research/src/round47_nn_encode.py)、[round60_recency_price_prior.py](research/src/round60_recency_price_prior.py)、[round63_recency_weighted_prior_train.py](research/src/round63_recency_weighted_prior_train.py) | v25 直接沿用的架構、encoder、先驗與日期權重來源 |

目前 v25 已有真實訓練、保存模型全 query 重載、分支組合、CSV 全列核對與 source／artifact SHA 證據。原四支程式是研究與發行入口，依賴多輪保存資料、模型和狀態，原路徑也已有不可覆寫收據；不能現在直接重跑同一命令來覆蓋舊產物。

v25 原控制鏈是保存向量組合重播，沒有重新推論其他五個原生模型。舊版推論入口已移出主分支，不能更名當作 v25。

**尚未完成 v25 從原始 train／valid 到全部模型的乾淨環境單一入口重訓。** 固定 seed 與保存雜湊保證可追查這次產物，不保證任意硬體重訓逐 bytes 相同。GitHub 的 source／文件快照也不是含私有模型、交易表與逐筆價格的完整重訓備份；依後續新重現收據再更新實際完成範圍。

目前提供 v25 的 24 頁簡報與完整講稿，尚未製作 v25 配音影片，亦未完成真人排練計時。


## 本次實際訓練曲線與新版簡報

[實際曲線與白話說明](evidence/reports/v25_training_curve_20261010.md)、[完整簡報](teaching/v25_presentation_20261010/deliverables/臺灣房價預測_v25_含實際訓練曲線_r2.pptx)、[完整口語稿](teaching/v25_presentation_20261010/deliverables/完整口語講稿.md)。第19頁為另一次固定30輪的先驗NN診斷，未改原提交；圖的橫軸是epoch，架構固定。主講4,217漢字，以275字／分估15.33分鐘，尚未真人計時。


