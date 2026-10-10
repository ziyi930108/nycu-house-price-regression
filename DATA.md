# 目前 v25 的資料範圍｜2026-10-10

現行提交已改v25；課程原表及非價格地理來源沿原流程。下方17項artifact與config/v22.json僅屬v22，不是完整v25artifact清單。v25還依賴新R107 prior模型及原完整工作區前置；請讀[專屬重現範圍](docs/V25_REPRODUCIBILITY_20261010.md)。所有交易表／模型／提交CSV仍未納Git。

下方為保留原 bytes 的歷史內容；選定、分數及重現範圍請以前述 2026-10-10 說明為準。

---

# 資料與模型取得

由原Kaggle課程取得train-v2.xlsx、valid-v2.xlsx、test-reindex-test-v2.2.xlsx及test-reindex-test-sample-submission-v2.csv。公開程式包不包含交易資料；其公開再散布權限未確認。

保存模型推論只需test和sample，SHA見config/v22.json。sample價格僅為格式範例，不用於模型計算。重訓另需train、valid及完整地理前置資料。

使用者限制為：禁止到內政部不動產成交案件實際資訊資料供應系統對答案，其他都可以。價格標籤與參考成交均來自課程資料。

## 地理資料

既有道路特徵取自OpenStreetMap／Geofabrik年度快照。© OpenStreetMap contributors。保存來源聲明見[evidence/data_external/geography/ATTRIBUTION.md](evidence/data_external/geography/ATTRIBUTION.md)，快照收據位於同目錄。道路代表點是粗略位置，並非精確門牌。

單價CatBoost、log CatBoost與兩個NN使用R6道路地理；直接總價CatBoost與LightGBM使用原87欄特徵。各分支最後的空間修正另使用R39地理。本次沒有重新下載道路或取得成交答案。

## 保存模型及狀態

[artifacts/manifest.json](artifacts/manifest.json)列出17檔、大小及SHA256，共1,447,418,242 bytes。包含六模型、類別映射、NN encoder／quantile knots／價格先驗、參考成交及空間狀態、兩套地理與feature schema；逐項用途見下表，原身分清單保持。

價格先驗與參考狀態衍生自課程價格，所以和模型一起保留於本機private/artifacts/，不放公開ZIP。保持原相對路徑，勿替換其他輪次狀態。取得合法副本後可移轉新目錄，不需要舊prediction.npy。

只載入經manifest核對的可信保存pickle。原工作區可用scripts/collect_artifacts.py複製；來源與目的SHA皆檢查，遇到不同內容不覆寫。

Public使用test約30%，Private使用其餘約70%；另附valid不是這個分割。

## 17項保存檔案各自用在哪裡

下表補充[身分清單](artifacts/manifest.json)中的具體用途，檔案大小與完整SHA256仍以原清單為準；本表不更改模型、狀態或雜湊。路徑都相對於 `private/artifacts/`（或 `--artifacts` 指定的根目錄）。推論入口會先核對17檔SHA，兩套地理均對應指定test的列序，不能任意交換或套給新房屋資料。

| 保存檔案 | 在v22推論中的用途 | 實際讀取／使用程式 |
| --- | --- | --- |
| `reports/feature_schema.json` | 87欄基礎特徵的欄位順序及類別欄名；四樹推論前核對輸入，並建立 CatBoost 類別索引。 | [predict_trees](src/tree_runtime.py#L96) |
| `outputs/round6/geography/final/test.pkl` | 對應原 test 列序的 R6 八欄道路地理；供單價／log CatBoost 加入地理特徵，也供兩個 NN 建立輸入。 | [main](src/predict_v22.py#L44) |
| `outputs/round39/geography/final/query.pkl` | 對應原 test 列序的 R39 道路地理；各分支最後的空間修正使用，不與 R6 特徵檔互換。 | [main](src/predict_v22.py#L45) |
| `outputs/round79/final_model/model/model.cbm` | 近期單價 CatBoost；單價輸出乘有效面積及10,000，還原總價 C_unit。 | [predict_trees](src/tree_runtime.py#L104) |
| `outputs/round79/final_model/model/feature_state.json` | 近期單價 CatBoost 的特徵狀態；核對 state 內容、增廣欄位／類別順序，以及原始欄位與 dtype。 | [_check_state / predict_trees](src/tree_runtime.py#L51) |
| `outputs/round46/trial/model/model.cbm` | log CatBoost；輸出以 expm1 再乘1,000,000還原總價 C_log。 | [predict_trees](src/tree_runtime.py#L106) |
| `outputs/round46/trial/model/feature_state.json` | log CatBoost 的特徵狀態；以 _check_state 核對特徵結構，和單價模型的狀態分開使用。 | [predict_trees / _check_state](src/tree_runtime.py#L106) |
| `models/final/catboost_raw.cbm` | 直接總價 CatBoost；原輸出乘1,000,000還原總價 C_raw。 | [predict_trees](src/tree_runtime.py#L108) |
| `models/final/lightgbm_l1_unit63/model.txt` | 單價 LightGBM，使用3,999棵保存樹；單價乘有效面積及10,000還原三路共用的總價 L_unit。 | [predict_trees](src/tree_runtime.py#L110) |
| `models/final/lightgbm_l1_unit63/category_maps.json` | LightGBM 保存的類別字串→整數映射；未知類別用−1，產生固定 f000 等欄位。 | [_encode_lightgbm / predict_trees](src/tree_runtime.py#L79) |
| `outputs/round2/neighborhood/final_reference.pkl` | 各分支的成交參考／時間趨勢狀態；傳入 predict_reference 的 trend 模式，產生第一步修正及診斷。 | [load_states / apply_branch](src/tree_runtime.py#L124) |
| `outputs/round39/trial/artifacts/spatial_state.pkl` | 空間修正的保存狀態；同時接收原分支初估、成交參考結果及診斷，輸出各分支修正後總價。 | [load_states / apply_branch](src/tree_runtime.py#L126) |
| `outputs/round48/final/encoder_state.json` | 兩個 NN 共用的保存編碼；將87欄加R6八欄轉為基礎數值、類別代碼及 cardinalities。 | [prepare_inputs](src/nn_prepare.py#L48) |
| `outputs/round55/final/numeric_state.json` | 面積 NN 的數值轉換狀態；對基礎數值編碼做 quantile-ramp 轉換，倍率 NN 仍使用未展開的基礎數值。 | [prepare_inputs](src/nn_prepare.py#L55) |
| `outputs/round60/final/price_prior_state.json` | 倍率 NN 的完整 fit 單價先驗；按保存層級查參考單價，結合有效面積建立總價縮放因子，並拒絕 uses_query_labels 狀態。 | [prior_for_query / prepare_inputs](src/nn_prepare.py#L26) |
| `outputs/round55/final/model/model.pt` | 面積 NN checkpoint（含結構設定及 state_dict）；讀第一組 ramp 輸入與面積縮放因子，輸出 N_area。 | [main → nn_predict.predict](src/predict_v22.py#L53) |
| `outputs/round63/final/model/model.pt` | 價格先驗倍率 NN checkpoint（含結構設定及 state_dict）；讀第二組基礎數值輸入與面積×先驗因子，輸出 N_prior。 | [main → nn_predict.predict](src/predict_v22.py#L53) |

兩個NN檔案由主入口依序交給 [nn_predict.predict](src/nn_predict.py#L34)，以保存的結構與 `state_dict` 載入模型，不從舊預測向量讀取答案。完整fit先驗用於待預測資料；訓練時的建案分組OOF流程是另一階段，不會在這個推論入口重新建立。

表中的私有檔案沒有隨GitHub程式包發布。`collect_artifacts.py` 只從你具備合法副本的原工作區逐檔核對並複製，並非上網下載模型或交易答案。
