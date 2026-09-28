# R93 固定建案密度權重：獨立執行前稽核

日期：2026-09-29（Asia/Taipei）。稽核者：`/root/r85_direction_audit`。

結論：依下列固定版本，未發現阻擋三期研究執行的數學、來源或介面問題。這是執行前工程審查，並非模型採用決定，也不代表已核實真實 CatBoost 產物或改善 Private。此稽核未訓練模型、未執行真實 query 推論或評分、未讀 test，未修改原始 source、計畫或選定的 v22。

## 1. 固定假設與數學

依 [固定計畫](round93_plan.md)，唯一改變為 R79 單價 CatBoost 的訓練 Pool 權重。對 fit 列的有效面積 A、原存檔 float32 日期權重 d、具名建案 fit 筆數 n：

- 原權重 `w0=(A/100)*d.astype(float64)`。
- 具名建案 `q=1/sqrt(n)`；缺名列的原始 q=1，count 使用 1 哨兵。
- `c=sum(w0)/sum(w0*q)`；新權重 `w=w0*q*c`，各步沿 float64 計算，保留原 Pool 權重總和。
- 單價目標仍為 `y/(A*10000)`，逆轉仍為 `max(z*A*10000,0)`。在逆轉非負投影前，新損失與 `d*q*c` 加權的總價 L1 僅差固定比例；因此確實改變原日期加權目標，並非只做無效果的權重縮放。

這個 c 不是 `1/mean(q)`。独立無標籤小例：w0=[1,9,1]、n=[2,2,1] 時，c=1.3628927740922494，新權重總和約 11，原總和 11；若除以 mean(q)，總和約 10.02944，不能保留原 Pool 尺度。

因 0<q≤1，c≥1；缺名與單例的最終乘數是 c，不能稱其「權重不變」。具名建案最終乘數為 c/sqrt(n)，只有 n>c² 時低於原權重。本方法不是每案等权：同案內面積／日期相對權重維持，但各案仍可能擁有不同總權重。若所有 q 相同，正規化後回到原權重（浮點容差內）；所有 q=1 時精確回復。

## 2. 分群、來源及洩漏邊界

[helper](../src/round93_project_density_unit.py) 沿用 [R5 project_keys](../src/round5_project_mask.py) 的完整 `city|district|project` 正規化。只有 project_name 的既有缺值規則決定缺名；同名不同城市／行政區分開計數。空字串、`nan`、`None`、`<NA>`、`NaT` 等沿原規則處理，未自行新增 `NaN`、`null`、`無` 黑名單。城市／行政區缺值但建案具名仍是合法具名 key。

只計各窗口 fit 的具名 key；缺名列不合併成一個大建案。helper 另拒絕三個正規化 key 欄位中的字面 `|`，避免串接碰撞。現有三期 fit 預檢均沒有此碰撞。

計數、q 與 c 不使用 y、query 特徵分布、query 標籤或 test。query prediction 不重新估計建案密度，也不接收訓練權重。模型 wrapper 保存目標 hash 是訓練追溯資訊，不能因此將整個 wrapper 稱為 pure-X；其中原 R19 feature state 與本次分群／權重計算仍有各自清楚邊界。

## 3. 純 fit 預檢與已知風險

主線 [fit-only audit](../outputs/round93_fit_audit/results.json) 沒有讀價格或 query 陣列、沒有 fit／候選分數。三期結果如下：

| 窗口 | fit 列 | 具名 key 數 | 缺名列 | 具名單例列 | 最大具名筆數 | c |
|---|---:|---:|---:|---:|---:|---:|
| development | 240,316 | 5,336 | 50,965 | 854 | 1,824 | 6.438731 |
| confirmation | 321,550 | 7,280 | 50,965 | 959 | 2,107 | 6.770329 |
| auxiliary | 134,805 | 1,797 | 50,965 | 587 | 1,226 | 7.235768 |

缺名／單例會相對上調約 6–7 倍，可能放大老資料、特殊交易或高價案例的影響；日期權重向量本身不變，並不保證整體加權日期分布不變。這是必須以三期反證檢驗的風險，並非已觀察到的模型結果。本輪不依此調整指數、截斷 q 或搜尋強度。

audit 的 `maximum_group_pool_weight_share` 按全部 key 字串聚合，包括缺名的地理 key；不得將此統計中的全部 group 都解讀為具名實體建案。其下降也不能直接當作獨立樣本數增加或泛化改善的證明。純 fit audit 實測 4.031 秒，與之後真實 fit 時間分開計。

## 4. 歷史方向是否重複

- [R5 研究特徵](round5_research_features.md) 曾提出尚未執行的備選密度權重 `min(1,sqrt(100/n))` 再除均值；實際 R5 採建案遮罩。本輪不能宣稱首次想到密度權重，但無 100 筆截點、保留面積與日期 Pool 尺度，並非已被該輪實測否決的同一模型。
- [R11](round11_report.md) 是使用未標註 query-X 的 domain importance weighting，來源、權重估計及原生總價模型都不同；其失敗是風險參考，不是 R93 相同實作的直接反證。
- R8 reference/spatial 的建案統計及 R89 的建案等權時間斜率位於不同估計步驟；R93 沒有沿用 R89 的新群趨勢，也不改 reference/spatial。

上述是相關歷史核對，非宣稱已窮盡所有過往實驗。

## 5. helper、driver 與測試收據

已讀完整 helper、[driver](../src/round93_experiment.py) 及 [self-check 收據](round93_helper_selfcheck.json)。`group_factors(X)` 回傳 int64 counts、float64 raw_q 和可序列化摘要；`weight_details(X,date_weights)` 回傳權重及摘要；`sample_weights` 只回權重。`fit_model(X,y,G,cats,w)` 保持既有 fit 契約；predict API 不含 y／weights／counts。

原 95 個特徵、23 類別欄位及 R19 legacy feature state 保留。CatBoost 參數全部等同 R79：depth 8、2,999 trees、seed 42、learning_rate .065，以及其餘既定參數。helper 使用獨立 R93 wrapper／原生模型檢查器，不修改舊 R79 source SHA；真模型需核實樹葉數不超過 256、特徵／類別順序與參數。preflight 只讀此檢查實作，尚未聲稱核過真實 CBM。

新版 driver 在 `fit_started` 前獨立用原 `project_keys` 重建 counts、q、c、w，逐位比對 helper，並核對純 fit audit 的值 hash 與 c。原 float32 日期來源、資料時間／索引／地理來源、舊 v22 三段完整向量及各對照指標均有既有重播檢查。候選只替換 native unit：`.75*native+.25*原LGB`，沿原同窗 reference/spatial，再與原 NN/log 取 median；未混入 R89、R91 或 R92。driver 會保存 counts、q、date／Pool weights、CBM、wrapper、raw/TWD vectors，並檢查全 query raw 與 TWD 重載逐位一致。此處是已審核的執行要求，須以實際 run 收據判定是否完成。

self-check 實測 53 項全部 passed，內部 0.328 秒、2 次 FakePool/FakeModel fit、0 次真實 fit。我未再重跑整套。涵蓋面積 fallback、exact sentinel／key、sum-preserving c、標籤擾動不影響權重、query 擾動不重算 fit 密度、錯誤輸入與 wrapper/native 參數拒絕、模擬重載及原輸入不變。收據保留首次 `KeyError: sample_weight` 工程失敗：合成斷言查錯 legacy 層級，修正為 `legacy.feature_rules.sample_weight`；候選數學未改。FakeModel 成功不等於真模型保存／重載已完成。

## 6. 評估限制與執行後待核

這是看過 R85–R92 結果後提出、重用既有三個窗口的適應性研究；confirmation 名称不代表全新盲測。固定三期／門檻及新案、高價、時間分群必須一起檢查，不能只挑最有利窗口。Bootstrap 只描述所用窗口的重抽樣，不是 Private 勝率。舊控制使用既有保存向量，亦非同時重新訓練的隨機性對照。

本審查不改 v22 的唯一提交角色，不生成新 CSV；即使研究通過，也必須另行判定新切點驗證與全量重建。真實模型容量、全 query 重載、完整三期結果與 protected hashes 待主線 run 結束後才能核實。

## 7. 本次核對的凍結身分

已逐檔核 SHA，self-check 所有 source bindings 一致；helper／driver／fit-audit 腳本 AST 可解析。本稽核未自行啟動訓練。

| 產物 | SHA-256 |
|---|---|
| reports/round93_plan.md | ac3ced3aa70ea3bb2548e4f4d4476da4c5b2e9ba2bd4795a5a673ee97f884054 |
| src/round93_experiment.py | f92361e80f4fe0659e309733807fd65015ca421595918cb416f34a267590c4fb |
| src/round93_project_density_unit.py | 7a3fcd56afea8702f84402b8cbf09e7e39b29f2bda9ea38c7b2877fc28773697 |
| reports/round93_helper_selfcheck.json | 6c647aef7a2eec0f53f805ae6c1be5d3c957e3945835e793a37895e51fd1c81b |
| outputs/round93_fit_audit/audit.py | 28363a05bb35aa43adf286a7d3229adbda5953a0e5c170126d9e950038e4a19e |
| outputs/round93_fit_audit/results.json | 0f9aa4b680fc2522b9491c988f47983f49a32c7c54e4cc8f1040f1108d06b8d4 |
