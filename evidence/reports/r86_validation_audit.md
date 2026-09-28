# R86 來源與驗證程式審核

日期：2026-09-28。使用者要求繼續改善回歸模型。本審核只讀盤點既有資料與新程式，未執行 R86 driver、候選預測、計分或 GPU 訓練。本頁將已完成檢查落成可交接紀錄，沒有為撰寫文件重跑檢查。

## 假設及歷史區別

R86 保持 v22/R79 的單價目標、有效面積×日期權重、95個既有特徵、2999樹及原CatBoost參數，只新增數值欄：

    log_recency_oof_unit_prior = log1p(prior_TWD_per_m2 / 1e4)

完整候選維持 .75×新CatBoost總價 + .25×既有LGB總價，再以同窗 reference/spatial 修正，最後與固定 NN/log 分支取 median。不把 R85 平方根目標混入。

既有 [R65計畫](round65_plan.md) 已測「OOF單價prior作NN數值輸入」，所以不是首次使用價格先驗或監督式OOF特徵。本次查閱歷史來源與計畫，未找到把此R60 prior直接加入單價CatBoost的相同實驗。這是把已有資訊交給另一模型的固定假設，不能預先宣稱更準。

## 原料與固定列序

先驗來自 [R60程式](../src/round60_recency_price_prior.py)、[R60 driver](../src/round60_nn_experiment.py) 及 [R60結果](../outputs/round60/results.json)。R63 prepared保留R60原八個陣列，只追加fit日期權重；見 [R63 driver](../src/round63_nn_experiment.py)。

資料為canonical train 455,255列，不加入valid。fit嚴格早於各query起日；query含下列起訖日。三期不因早期成績更改順序或方法。

| 期別 | query日期 | fit列數 | query列數 |
| --- | --- | ---: | ---: |
| development | 2022-08-11～2023-08-10 | 240,316 | 81,234 |
| confirmation | 2023-08-11～2024-08-10 | 321,550 | 133,697 |
| auxiliary | 2021-08-11～2022-08-10 | 134,805 | 105,511 |

CPU只讀檢查實際載入canonical資料，逐一確認：
- R60、R63及R79的fit/query indices全列完全相同，內容SHA等於R79固定split_contract。
- R60六份prior向量的檔案SHA等於R60 results的artifact_sha256；float64、正有限、列數正確。
- OOF向量內容、fit標籤、有效面積、單價、canonical source indices及fold IDs之內容SHA與R60證據一致。
- 固定SHA256('42|'+group)取前8byte little-endian再mod5的分折身分重算一致；held與complement建案群互斥；各complement索引與標籤SHA一致。
- 使用保存的full-fit prior state重新查query，全列prior精確一致。
- R63 train/query_price_factor_million逐位等於A×對應R60 prior/1e6轉float32；train_y_million等於相同fit標籤。
- R63日期權重從fit日期計算後轉float32，與保存值完全相同；full-fit日期／異常旗標／anchor／權重SHA與R60 full_fit_date_support一致。

此檢查exit0；沒有建立新候選，沒有讀取R85其餘期別成績來修改R86。

## 六份prior檔案身分

以下是整份NPY檔案SHA256，不是裸陣列內容SHA。位置前綴為 outputs/round60/{stage}/。

| 期別 | 檔名 | SHA256 |
| --- | --- | --- |
| development | train_oof_unit_prior.npy | 6e293e5522464f74abd8c9ae2187971acfb09c37697306422701f2b9f12572d3 |
| development | query_unit_prior.npy | 37c97510280883515a904cfe671002a3c3855dde9acf983bcafea44ba2e404e0 |
| confirmation | train_oof_unit_prior.npy | c07add8250bfcb2d0980479d8bf96f5a01c3f081faa0a0809ec4ee0b703e14ac |
| confirmation | query_unit_prior.npy | 7ae1c9631ea0f7276bba30d5cf346451a639c28eace633dfb5473610a11ded4c |
| auxiliary | train_oof_unit_prior.npy | 6f3ba317361080cd8365309aea667f08c223fd9ffd50a3869eb28b43fd0a8aac |
| auxiliary | query_unit_prior.npy | 17407e755f33b0885f4e1d32c3d2d2e95d4cec5ac863b122ff02a00eef255a34 |

canonical索引內容SHA（ordered little-endian int64）：

| 期別 | fit | query |
| --- | --- | --- |
| development | 73750cf8018b2a28198db91ae397137a90366ae96e5f186cfdf862ab4a29e593 | 343b1b0762dfec06fe118f7c62c33e0e8988ea14ecf6846735b55d576cd7df43 |
| confirmation | 08ffe5fb2c639598390ffbeff10ad243f0150bc9191e06f1b2924633f91b6c1c | 8cffd2e1bb2bbc40c34c89350805d96a04386545015ccbb99bc4f8bca4ad4d83 |
| auxiliary | 105c98493c615bb92904ba2ec7597d7dd56d72b839d86b1e7fab5d2a7bd4e655 | 3ad65ebb87d7ced138d9b04ddc1f6658885d1c6afcecb846ccf2e230137e80bf |

各期完整證據：
- [development prior preparation](../outputs/round60/development/prior_preparation.json)
- [confirmation prior preparation](../outputs/round60/confirmation/prior_preparation.json)
- [auxiliary prior preparation](../outputs/round60/auxiliary/prior_preparation.json)
- [R79固定計畫](../outputs/round79/plan.json)

## R86 driver審查

審核對象：[R86計畫](round86_plan.md)、[R86 driver](../src/round86_experiment.py)、[R86 helper](../src/round86_prior_feature.py)。

driver在各期GPU fit以前：
1. 使用當期X、yfit、fi呼叫fit_oof_prior；重建OOF prior、fold IDs、full-fit state及evidence，與R60保存物精確比較。
2. query只經full-fit state的predict_prior查表，與query_unit_prior.npy精確相同；不用query標籤建立prior。
3. 核對R63保存價格因子，留下prior_verification.json。完整OOF重建是driver執行的再現核驗；本審核本身只重算fold身分、來源證據與query查表，沒有額外呼叫fit_oof_prior重建候選。
4. 以R6 tree geo保留原95欄，再追加1個float64數值欄，總96欄／23類別。legacy state僅描述原95欄，wrapper清楚表示新監督式特徵，不冒稱pure-X。
5. 延續R85框架的canonical索引、時間界限、日期權重、R39 query geo、reference/spatial fit-state與原v22逐層重組驗證。新base重新產生自己的reference diagnostics。
6. query標籤只交score；model.fit沒有query eval_set或early stopping。保存CBM、wrapper、全query raw/TWD重載相同證據與實測耗時。

新增feature會改變樹分裂與亂數使用，不保證其餘訓練路徑逐位一致；舊GPU控制未同期重訓，觀察微小差異不能完全歸因新資訊。

## 訓練前發現與修正

code review發現兩項實作問題，主執行緒在啟動R86以前完成修正：
- R60 plan只有split_contract，沒有splits欄位；原寫法會發生KeyError。現保留split_contract比較，改從R60 results各stage的dates核對R79 splits。
- plan與model metadata兩處round=85複製殘值改為86，避免產物實驗身分誤記。

這兩項是訓練前review修正，不是失敗fit或模型重試。本審核未執行有問題的driver。修正後未見剩餘開跑阻擋；主執行緒稍後通知已啟動R86（啟動PID6228），程序最終狀態與分數須以outputs/round86保存收據判定，本頁不宣稱三期已完成。

## 必須保留的限制

- prior使用fit標籤，是監督式OOF特徵；held fold prior、global fallback與日期anchor只用其complement。
- 五折建案群OOF不是時間前向OOF，不能說每筆訓練資料只見自身交易日以前的成交。
- OOF complement與query full-fit參考量不同，且train/query族群不同，可能有先驗分布差異。
- 30筆cell支持門檻不是30個獨立建案；建案別名、缺名及地址解析限制仍存在。
- NN也使用相關prior，新CatBoost可能提高分支相關性、減少median互補收益；必須看完整候選而非只看native。
- 三個年度已反覆使用；配對bootstrap只描述固定候選在這些窗口的條件不確定性，不涵蓋全部方法選擇或Private勝率。
- 沒有使用外部成交答案，沒有按test ID改價或查未知Private。結果再好也不自動改變已選v22或Kaggle勾選。
