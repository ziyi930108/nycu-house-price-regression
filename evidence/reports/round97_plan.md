# R97：固定配方的錯開時間切點檢查

日期：2026-09-29（Asia/Taipei）。在新窗口模型 fit 與 query 評分之前固定。這是 R90 資料準備的完整執行，不是搜尋新比例／參數，也不是新的盲測。R95／R96已完成而未達门檻，本輪不另重訓它們。

## 目的與範圍

檢驗固定完整 v22 與既有 v24 的相對差異是否依賴每年8月的切點，為後續回歸改良提供較完整基準。兩配方都曾使用重疊歷史資料選型；新的起點不能消除累積選型偏差。使用者唯一提交仍為 v22，Public 1,998,568.5706，Private 未知。此檢查不自動替換提交、不訓全量test模型、不產生CSV。

## 資料與邊界

- canonical train；fit日期 < 2023-02-11，共272,117列；query 2023-02-11～2024-02-10（含），107,169列。不混valid，不讀test，不查外部成交答案。
- 使用既有純X preparation：outputs/round90/preparation/manifest.json，SHA256 `400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7`。原始R6地理供tree/NN，R39補位地理僅供spatial，來源2023 snapshot早於切點。
- 隔離標籤擷取程序核對原 data_cache/train.pkl SHA後，按 fit_source_cache_indices 只輸出 fit_y.npy。pickle底層會反序列化整張來源表，不能聲稱整個擷取程序未讀query價格；但不計算query統計、不輸出query標籤，模型driver只載fit-only檔。scorer須等兩完整預測及所有模型收據封存後才擷取query標籤。
- NN分组OOF source_indices傳canonical fit_indices，不能傳raw cache索引或0..n-1。新fit重建encoder、LGB maps、ramps、五折OOF與fullprior、日期權重、reference及spatial。不能切片後期保存模型或state來回測。

## 凍結模型與組合

使用原R90 helpers與R83 seed137 helper：C_raw、L_unit、C_log、C_unit42、C_unit137、N_area、N_prior，共5tree+2NN=7 native fits。GPU由主線串行，無warm start、query eval_set、early stopping、seed／輪數／比例搜尋。C_raw/log/unit均2999棵，LGB3999棵，兩NN seed42/30epochs；每支按原配方與目標還原TWD。C_unit42/137與prior NN共用新fit float32日期權重；area NN uniform；log維持原float64規則。

```text
B_N   = .75*(.5*N_area + .5*N_prior) + .25*L_unit
B_H   = .75*(.5*C_raw  + .5*C_log)   + .25*L_unit
B_U22 = .75*C_unit42 + .25*L_unit
B_U24 = .5*C_unit42 + .5*C_unit137
RS(B) = original_spatial(new_state, query_X, query_geo, B,
                        original_reference(new_fit_reference, query_X, B))
v22 = median(RS(B_N), RS(B_H), RS(B_U22))
v24 = median(RS(B_N), RS(B_H), RS(B_U24))
```

新reference/spatial共用一份，但每個base重新產生reference及其confidence/diag。U24不含LGB。保存每支native完整query重載、四支reference/spatial state重載與diag、兩完整median、來源／輸入SHA與計時。輸出全新outputs/round97，不覆寫、不自動重試；工程失敗保留原source/attempt/狀態，恢復另記錄且不據已知結果變更科學配置。

## 事前固定評估

所有完整預測封存後單次執行獨立scorer：TWD整體MAE、沿用固定group權重 seen=0.36617593949822613、new_nonmissing=0.6338240605017739、missing=0；新舊建案、四個相對季度、月份、城市、原五價帶（<1千萬／1–2千萬／2–5千萬／5千萬–1億／>=1億）、尾部誤差與各層比較。真價分帶僅診斷，不用來推論分流。沒有重新從test取得mix。

沿原methods的paired project與3月循環month bootstrap，每項2000replicates，seed20260925，這個單一窗口stage index=0、method index=0/1；保存兩指標共四個區間與replicates。候選−v22為delta，負數較好；不把bootstrap當Private勝率或已校正多重選型的檢定。

預先判讀：同時整體／加權下降才稱本窗口點估計一致支持；最近三個既有年度仍完整呈現，任何季度／>=500列分群退步超過5%需明列。區間跨零表示條件不確定性。無論結果如何，本輪只輸出穩定性證據，不宣稱Private改善或自動採用v24。下一新假設另訂計畫，不能看此窗口再搜比例。

## 執行與限制

root先核無活躍研究程序，再完成source/API前置獨立審查、fit-label與純X對齊、固定來源清單。原量測顯示七native總量可在現有本機資源執行；實際時間以此次收據為準，不將不同範圍歷史時間相加成承諾。GPU固定seed不保證重新fit位元一致，要求的是每支保存模型重載完整query一致。

本輪重建一個歷史窗口的全部模型，不等於乾淨環境端到端重現正式test提交；原v22模型、CSV、簡報、報告、選定與Public紀錄保持。
