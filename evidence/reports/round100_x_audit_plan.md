# R100：時間前置 prior 的純 X 可行性稽核

2026-09-29。本計畫在任何本輪真實 X 統計前固定；只作支持量稽核，不估價格 prior、不讀標籤、不訓練、不計 MAE，也不授權新的回歸模型。

## 單一規則

只檢查原三個年度 fit，順序 auxiliary（cutoff 2021-08-11，134805列）、development（2022-08-11，240316列）、confirmation（2023-08-11，321550列）。由原始純特徵表以保存 canonical mapping 對齊，再讀保存 fit indices；不呼叫同時載入 y 的舊 loader。

分組完全沿 R60：`round52_price_prior.grouping.cluster_keys(Xfit, canonical_fit_indices)`；每鍵 `sha256('42|'+key)` 前八 bytes、little-endian、mod 5。canonical indices 不能換成 raw cache index 或本窗重新編號。結果須與保存 R60 folds 逐位相同。

固定日曆季（1/4/7/10月1日開始）。有效日期沿 R60：day有限、兩個日期旗標有限且等於0。季 reference 僅取有效日期且 day 嚴格小於該季起點、fold 不等於 held fold 的原 fit 列；整個 held fold 均排除。無效日期另記，不猜所屬季、不加入時間 prefix。

**整季 eligible 當且僅當五個 held-fold complements 的最小 reference 列數至少30；可訓練列還須日期有效。** 全五折都檢查，即使該季某折没有 held 列。只評此單一 mask，不掃其他門檻。30 是 root 本次預先固定的新暖身慣例，受既有 cell 30列啟發；原 R60 global fallback 沒有30列門檻，不得混稱原規則。

## 支持與輸出

依原 R52 四個 cell keys 與 raw count≥30，計算每季／每 held fold 的 prefix reference 列數、群數、最早／最晚 reference 日期、各層首個合格 cell 命中量與 global／零 reference 回退量。不讀任何單價或執行 prior `_build`／`predict_prior`。另列原無時間截斷 group-OOF 在相同 eligible 列的層級支持，僅作 X 支持比較。

保存每窗 fit indices、fold IDs、valid-date mask、每列季起點（無效為-1）、eligible mask，逐位重載核對。報最早 eligible 季、總保留／暖身排除／無效日期列數、每季筆數、各折統計與 cell 覆蓋。日期權重只呼叫原 R60 `_weights(_dates(Xfit))`，轉原 float32 後核保存 R63權重SHA；報原權重保留質量、掉列質量與ESS，不重正規化、不據此決定新 loss。

## 精確讀取契約

實際執行前生成 `outputs/round100_x_audit_preflight/request.json` 與 sidecar；逐檔保存以下 readset 的 SHA、來源閉包及本計畫 SHA，之後核對不變才載 X：

- `data_cache/train_features.pkl`、`reports/feature_schema.json`、`outputs/round6/geography/development/train_source_cache_indices.npy`。
- `outputs/round48/plan.json`、`outputs/round60/plan.json` 與各自 `.sha256`。
- 對明列的三個 stage，各讀 `outputs/round48/{stage}/fit_indices.npy`、`outputs/round60/{stage}/fit_indices.npy`、`outputs/round63/{stage}/fit_indices.npy`、`outputs/round60/{stage}/prior_fold_ids.npy`。
- 每 stage 的 `outputs/round60/{stage}/prior_preparation.json`、`outputs/round63/{stage}/preparation.json` 與各自 `.sha256`；只使用來源、fold、日期權重等身分欄，不用價格摘要作統計或門檻。

純 X/mapping/schema 的舊SHA須對 R48 plan；fit向量內容SHA對原 split contract；fold及日期權重內容SHA對舊準備metadata。明確不開 `data_cache/train.pkl`、任何 NPZ、標籤 NPY、prior state、模型、query indices／query資料、test或valid。完整純特徵pickle會含三窗外的 X，僅按保存 fit indices 統計，不能稱物理上只反序列化 fit 列。

新來源為 `src/round100_temporal_prior_x_audit.py`；讀取並固定 R60、R52、grouping 及其本機 import source 閉包，無 import-time fit。新來源 SHA 在 request 凍結，不在計畫中製造自我雜湊循環。只做有意義的人工日期邊界／五折／30列／冷啟動檢查後執行一次；失敗另存原碼與execution，不覆寫或降低保護。

輸出 `outputs/round100_x_audit/results.json`、`execution.json`、`report.md` 及上述純 X mask arrays。來源／readset 執行前後SHA一致，原 R99與v22不變。結果只描述資料支持，不能證明 temporal prior 較準或舊方法洩漏；新資料選擇和原方法比較須另固定 matched-sample 實驗。這次不自行選擇新訓練政策或啟動任何模型。
