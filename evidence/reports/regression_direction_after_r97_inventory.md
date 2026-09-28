# R97 執行中：時間外 L1 組合的有限盤點

2026-09-29。只讀程式、小型 metadata 與聚合報告；沒有載入標籤、陣列或模型，沒有 fit、predict、score 或外部查價。**R97 結果未知，未修改其固定比較。本文不是新實驗計畫。**

## 結論與歷史證據

本次有限搜尋未找到已實作的「以更早 forward 預測及其答案訓練、在下一時間窗評估」受約束 L1 線性 stacker／校準。這個方向尚未被既有混合實驗充分否證，但**目前不直接立項為可完成原三期採用門檻的新候選**：現存完整 forward blocks 只足以支援兩個較晚期的 meta-test，最早一期缺前置 block。

| 已做過的方法 | 為何不等同上述方法；保留的反證 |
|---|---|
| [R1 選型](../src/select_model.py) | 單模型／兩模型 25%、50%、75% 粗網格，在同一 temporal validation 的加權 MAE 選擇；另附 valid 破平手。没有獨立較早 meta-fit 年期。 |
| [R2 混合](../src/round2_blend.py)、[報告](optimization_round2_report.md) | 在 2024-04-01～2024-08-10 同窗按建案 GroupKFold 選全域／seen-unseen 比例；原生模型又曾用該 holdout early stopping。程式已明記不是 untouched test。新候選池全域／分群 crossfit 加權 MAE 約 174.09／178.86 萬，分群較差；不能外推成嚴格時間 stacker 已失敗。 |
| [R43 固定 median](round43_median_report.md)、[R55 固定等權](../src/round55_equal_blend.py) | 沒有學 meta 係數。R43 對原三支 mean 的 overall 差依中／新／早期為 +0.179%、+0.124%、−0.296%；聚合方式有年度取捨，但當時分支也不是目前 v22。 |

對 src 的 linprog、QuantileRegressor、stacker／stacking、SLSQP、nnls、LinearRegression／Ridge／Lasso 等有限靜態搜尋無命中；並讀上述實際組合程式，不能據此宣稱所有歷史二進位產物都已鑑識。既有 reference 是 fit 內價格鄰近與趨勢校正，並不是六模型 forward predictions 的 meta-fit。

## 現有保存資料能支援到哪裡

[年度設計審核](validation_design_audit_r84.md)及 [R79 固定規約](../outputs/round79/plan.json)記錄下表。各 block 原生模型只 fit 其起點以前資料；[R3 baseline](../src/round3_backtest.py)與 [R6 最早期 baseline](../src/round6_runner.py)無 query eval_set，後續來源核對見 [R79 driver](../src/round79_experiment.py)。配方仍受反覆回測選型影響，這是資料列外樣本，**不是從未見過的盲測**。

| 要評估的 query 起點 | 可作 meta-fit 的完整較早 forward block | 列數／缺口 |
|---|---|---:|
| auxiliary：2021-08-11 | 現有三期無更早 block | 缺；不可用本期 query 答案學係數再評本期 |
| development：2022-08-11 | auxiliary：2021-08-11～2022-08-10 | 105,511 |
| confirmation：2023-08-11 | auxiliary＋development（後者至 2023-08-10） | 186,745 |
| R97：2023-02-11 | 理論上 auxiliary＋development 中日期早於此起點的前段 | 前段列數本次未讀陣列核實；**不加入已凍結 R97** |

三個原 query blocks 本身不重疊；confirmation 的 133,697 列不能回流前兩期 meta-fit。若日後使用晚於 2024-08-10 的部署起點，三期皆在時間上較早，但這不提供另一個已知標籤的未來評估期。只按 cutoff／日期過濾，不能按 query 誤差或 Public 選子集。

已核 metadata 路徑與檔案存在，**未讀取或重算下列向量 SHA**；新任務仍須驗 artifact authority、canonical query index／日期／TWD dtype 與 fit cutoff，而非只信同長度：

- 三個完整分支（s 為上述 stage）：N = outputs/round71/s/seed42/nn_spatial.npy；H = outputs/round46/s/recency_log_spatial.npy；U = outputs/round79/s/unit_spatial.npy。各自已含同窗 fit-only reference／spatial，可直接作三欄歷史 forward 特徵；來源由 R79 plan 的 bindings 與各期 results 綁定。
- 六 native：raw／LGB 在 outputs/round3/baseline/s/{catboost_raw,lightgbm_l1_unit63}.npy；auxiliary 改為 outputs/round6/baseline/auxiliary。log 在 outputs/round46/s/model/predictions.npy；area NN／prior NN 分別為 outputs/round54、round63 的 s/model/predictions_twd.npy；unit CB 在 outputs/round79/s/model/native_predictions.npy。NN 混合身分見 [R71 source](../src/round71_missing_cell.py)。
- 不得將 R60 的 fit 內 group-OOF prior、native 的 in-sample fit 預測，或較晚期模型回推早期列，替代上述完整模型 forward blocks。三分支最後一層的 L1 組合與六 native 直接重組，改動層級不同，不能當同一假設搜尋兩者再挑較好者。

## 缺口、成本與下一步的界線

若只探索兩個較晚期，現有保存向量理論上可免新 native fit；仍需另寫依日期隔離的 meta-fit／seal／評分契約及係數保存重載。若要求原三期都可獨立比較，必須補一個 query 全部早於 2021-08-11 的更早 origin：固定 v22 配方至少重建 **4 樹＋2 NN**，連同 encoder、OOF prior、日期權重、reference／spatial；純 X 地理需核其來源早於新起點，不能直接把 2021 年地圖回用到更早 origin。未找到此完整早期 block 的保存證據；本次未選新 cutoff。

可參考 [既有重建成本盤點](r89_offset_validation_inventory.md)與 R90／R97 的新窗介面，不能直接改其 immutable 日期。舊各期實測單個 raw／LGB baseline 流程約 110～153／155～505 秒，單個 log／unit fit 約 110～135／151～190 秒，兩類 NN 各約 10～27 秒；口徑不同，**不可相加冒充新窗口 ETA**。更早起點準備、地理覆蓋、全部重載與線性求解工時尚未量測。

非負且和為 1 的權重可限制外插，L1 目標與 TWD MAE 對應；它仍會受歷史年份、價格尾端和高度相關分支支配，且取代逐列 median 會改掉其限制單支極端值的行為。不得在重用 query 上搜尋約束、截距、分群或衰減強度，再宣稱沒有過擬合。下一步由主線決定是否接受「兩期探索」的較弱證據，或先承擔早期 block 重建；本盤點不固定參數、不新 fit、不產 CSV，v22 唯一提交與未知 Private 保持。
