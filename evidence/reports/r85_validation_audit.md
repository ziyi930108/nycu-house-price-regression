# R85 驗證與可重用狀態審核

日期：2026-09-28。使用者重新要求改善回歸模型。此文件只讀審核：未訓練、未推論、未評分，未改舊模型、預測、CSV、decision、selector。主執行緒已固定平方根面積尺度假設；本文是實作入口與公平比較要求，不是已完成結果。

## 固定假設及公平比較

R85 定義有效面積 A，沿用 v22 的有限正建物面積／車位面積／1 fallback；令 F=sqrt(A/100)，訓練 target=(y/1e6)/F，weight=F×原 R63 保存日期權重 d，預測回元 max(z×F×1e6,0)。

裁成非負之前，逐筆有：

    F*d*abs(y/(1e6*F)-z) = d*abs(y-z*F*1e6)/1e6

所以改的是模型學習的目標尺度和相應權重，仍對齊原有日期加權總價絕對誤差。它沒有使用 query 標籤、未知 test 真價或 test ID。這是合理、可執行的單一表示法假設；等式成立不預先證明泛化較好。

主對照是選定 v22/R79。維持 .75*new_native_TWD + .25*old_LGB_TWD、相同 reference/trend、spatial、NN/log，最後 median。v24/R83可當額外研究對照，但不能把 .5 seed mean 配方放入主比較。三年度全部完成，不因第一期成績而換 F 指數、種子、比例、切點或停止呈現不利期別。

特徵和地理編碼可保留舊 R19 fit-only state；其中歷史 unit-target／area-only weights 欄位只是來源，R85必須另有實際平方根尺度 wrapper／metadata，不能用舊欄位描述新模型。

## 最小合法資料入口

程式參考 [R79 driver](../src/round79_experiment.py)、[R84 driver](../src/round84_experiment.py)，不要執行其不可覆寫的 run。

- import round48_nn_experiment as old；old.previous.load_training_rows(plan) 實際在 [round39_city_road_experiment.py](../src/round39_city_road_experiment.py)。
- loader 讀 data_cache/train_features.pkl、data_cache/train.pkl，兩者455,784列，再只套一次 outputs/round6/geography/development/train_source_cache_indices.npy 得455,255列。loader本身沒有檢查所有計畫SHA，driver要先核對固定來源雜湊。
- 欄位／類別在 reports/feature_schema.json。outputs/round79/plan.json 的 bindings、split_contract、splits、target_group_weights 已提供完整期別身分；不要用名稱推測索引。
- b=plan['bindings'][stage]；fit/query indices 在 b['unit']['fit_indices'] 和 b['unit']['query_indices']。檢查int64、有序、不重疊、canonical範圍、列數及內容SHA。
- tree地理用 b['tree_geo']（R6），先核canonical全列index再按fi/qi切G/Gq。spatial才用 b['unit']['query_geo']（R39），二者不能混換。
- 日期權重只取 b['prepared'] 的 train_sample_weight，保持float32到helper，再轉float64乘F。核對R63 preparation、原fit_y_million、fit日期／旗標SHA、固定365日與無query/target使用；不要按query日期正規化。

| 期別 | query日期 | fit／query列數 |
| --- | --- | ---: |
| development | 2022-08-11～2023-08-10 | 240,316／81,234 |
| confirmation | 2023-08-11～2024-08-10 | 321,550／133,697 |
| auxiliary | 2021-08-11～2022-08-10 | 134,805／105,511 |

fit是query起日以前canonical train有效日期列，這三期不加入valid。期間已重複使用，confirmation名稱不是獨立盲測。

## 可合法共用的保存物

各stage位置皆由R79 plan input SHA綁定，driver仍需核對：

| 物件 | 保存位置／binding |
| --- | --- |
| 舊單價LGB | b['unit']['old_unit']：development/confirmation在outputs/round3/baseline/{stage}/lightgbm_l1_unit63.npy；auxiliary在outputs/round6/baseline/auxiliary/lightgbm_l1_unit63.npy |
| NN完整分支 | b['paths']['nn']＝outputs/round71/{stage}/seed42/nn_spatial.npy |
| log完整分支 | b['paths']['log']＝outputs/round46/{stage}/recency_log_spatial.npy |
| reference/trend | b['unit']['reference']：R3或R6相同期別reference_v2.pkl |
| spatial state | b['unit']['state']＝outputs/round39/evaluation/{stage}/spatial_state.pkl |
| spatial summary | b['unit']['state_summary']＝同目錄state_summary.json |
| v22完整預測 | outputs/round79/{stage}/predictions.npy |
| v24完整預測 | outputs/round83/{stage}/predictions.npy |
| v22 native與中間層 | outputs/round79/{stage}/model/native_predictions.npy、base.npy、reference.npy、unit_spatial.npy |

reference['fit_report']['fitting_rows']只檢查列數不足，須核spatial summary的fit_X_sha、fit_geo_sha、query_used_in_build=False、reference_residual SHA及相同split索引來源。先以舊native/LGB重組R79 base/reference/spatial/median，所有向量bitwise一致後才fit新候選。

後處理入口：

    p, diag, report = old.original.predict_reference(
        reference, Xq, base, variant='trend', return_diagnostics=True)
    branch, detail, spatial_report = old.spatial.predict(
        spatial_state, Xq, R39_query_geo, base, p, diag)
    candidate = np.median(np.column_stack([fixed_nn, fixed_log, branch]), axis=1)

predict_reference是 [round2_neighborhood.predict_reference](../src/round2_neighborhood.py)，spatial是 [round8_spatial_reference.predict](../src/round8_spatial_reference.py)。新base必重算自己的diag，不能用控制diag；未套spatial列須精確等於自身reference。

## 計分及既有CPU診斷相容

分數入口 [round3_evaluate.score](../src/round3_evaluate.py)：

    score = lambda v: old.original.score(
        yall[qi], v, X, Xq, plan['splits'][stage][0], plan['target_group_weights'])

輸出overall、固定seen/new weighted、strata、四季度、月份、城市。價格bands只作事後分析，不可按真價推論分流。

通用工具要求R85：
1. outputs/round85/results.json及.sha256；checked_stages順序為development、confirmation、auxiliary。
2. stages[s].metrics至少包含v11、v12、v5、R85完整score字典；v22/v24可一併提供。預設三對照向量在R79 bindings paths。
3. 每期保存fit_indices.npy、query_indices.npy、predictions.npy；提供prediction_sha256及artifact_sha256，float64有限非負。
4. 保存plan/input/source SHA、native模型和wrapper、raw/TWD全量重載精確檢查、fit及程序實測秒數。不能沿用上個模型的target/weight SHA描述新模型。

    .venv/Scripts/python.exe -B -X utf8 src/candidate_diagnostics_v3.py --round 85 --candidate R85 --extra-control v22:79:R79 --extra-control v24:83:R83
    .venv/Scripts/python.exe -B -X utf8 src/paired_candidate_bootstrap.py --round 85 --candidate R85 --control-round 79 --control-name v22 --control-key R79
    .venv/Scripts/python.exe -B -X utf8 src/paired_candidate_bootstrap.py --round 85 --candidate R85 --control-round 83 --control-name v24 --control-key R83

診斷含新舊案、城市×seen/new testmix支持率、五價格帶。testmix不是Private MAE；尾部對總誤差的貢獻可另列，但不能用它設定真價帽。Bootstrap固定2000、seed20260925，建案與三月循環月份區塊，各overall/weighted，三期每對照12區間，全部保留。

通用paired工具核對R48 indices與保存預測SHA，但不直接讀候選indices比較。必須先由R85 driver及candidate_diagnostics_v3驗完R85/R79/R83索引相同；不可把單獨paired當完整alignment稽核。

## 成本與額外切點

沿三期只新訓分支最直接。R79舊實測development165.062秒、confirmation190.125秒、auxiliary151.25秒，主程序602.453秒；R84六fit1488.782秒、主程序1605.906秒。這是舊量測，不是R85耗時保證。

半年錯開起點（fit<2023-02-11；query至2024-02-10）沒有現成合法六模型state，須重建tree/NN、encoder/prior、reference/spatial及該origin地理來源證據，不能套後期模型回測；通用三期diagnostics的硬索引也不能直接借用。它能檢查時間邊界敏感性，但與既有樣本重疊，不是盲測。建議先執行固定R85三期，再評估是否值得建立新origin工具；本次未新增切分。

## 不應省略的限制

- 舊v22是保存控制，未本輪同期重訓；GPU固定seed仍有重訓變異，差異不能完全歸因平方根表示。效果小時可另固定同期控制，不可反覆挑seed。
- 重用年度與bootstrap不校正整個研究搜尋，不代表Private勝率。
- 單支native改善經reference/spatial/median可能消失，以完整候選MAE作主結論。
- 同源模型可能有共同偏差，median不保證高價群改善。
- 新候選不自動取代選定v22；先保存全部正反證據，主執行緒再決定是否有理由final fit／新CSV。
