# R89 後續錯開半年窗口：驗證成本與相依盤點

盤點日期：2026-09-28。**本文件是預備盤點，沒有啟動新窗口、fit、模型推論或候選評分。** 本次只讀 source、報告與 metadata，未讀取任何 query 價格向量，也未讀 test 或查外部資料。

若主執行緒決定比較完整 v22 與固定 R89，新窗口為 canonical train 中有效日期 fit < 2023-02-11，query 為 2023-02-11～2024-02-10（含末日）。不加入 valid，不搜索切點、seed、輪數或比例。此窗口與舊年度資料重疊、配方已受既有研究影響，應稱「錯開起點的時間穩定性檢查」，不是新盲測，也不能換算 Private 勝率。

## 可否直接沿用現有模型

**未找到完全相同 2023-02-11 cutoff 的已完成六模型／價格狀態。** 針對 src、outputs、reports、runs 的文字來源搜尋此起訖日期，命中原 R84 決策及 R84／R85 的待研究設計，未見對應訓練 metadata。沿 v22 實際相依核對，保存 native／prior／encoder／reference／spatial 均綁定原三個 8 月切點或最終全量 fit。本盤點不是對所有 binary 重新掃描／反序列化的鑑識證明；開跑前仍須以新索引 content SHA 為唯一重用資格。

- 原 confirmation 的 fit < 2023-08-11，已包含新窗口 2023-02-11～2023-08-10 的價格，不能用於新 query。
- 原 development 的 fit < 2022-08-11，雖未見新 query 價格，但少了六個月合法 fit 資料；直接用它不是本次固定新 cutoff 的完整 v22。原 auxiliary 同理。
- 最終全量 train+valid 模型與價格 state 更不能回測這個窗口。
- 此處 R89 只改 reference／spatial 的時間處理。**六個 native 可按新 fit 訓練一次，再由 v22 與 R89 共同使用；不是兩套各訓六個。**

## 六個 native 的最短合法重建

| 分量 | 新窗口須做的事 | 可用入口與重要設定 |
|---|---|---|
| C_raw：直接總價 CatBoost | 新 fit，87 欄，y/1e6，無額外樣本權重 | [round3_backtest.py](../src/round3_backtest.py) 的 run/catboost 分支或 [round6_runner.py](../src/round6_runner.py) 的 auxiliary_baseline/catboost 分支是既有配方範例；取 [final_config.json](../runs/v1/models/final_config.json) 的 component.iterations=2999 覆蓋 parameters.iterations=3000，固定 seed42、depth8；另寫新 driver 包装，不能直接跑固定舊 stage |
| L_unit：LightGBM 單價 | 新 fit、重新建立 fit-only category maps，87 欄；y/(A×1e4)、weight=A/100 | [train_lightgbm.py](../src/train_lightgbm.py) 的 encode_features、price_factor、effective_area；原 baseline 分支固定 component.iterations=3999，覆蓋 n_estimators=4000，不用新 query early stopping |
| C_log：近期加權 log CatBoost | 新 fit、95 欄 tree geography、新日期權重、feature state | [round46_recency_log.py](../src/round46_recency_log.py) 的 fit_model(X,y,geo,cats)，fit_weights 只用新 fit 日期；2999 樹、原 R12 log1p(y/1e6)／expm1 逆轉 |
| C_unit：近期加權單價 CatBoost | 新 fit、95 欄 tree geography、新 fit 日期權重；A/100 乘新保存 float32 date weights | [round79_recency_unit.py](../src/round79_recency_unit.py) 的 fit_model(X,y,geo,cats,saved_date_weights)，2999 樹、seed42；新 driver 先核日期、行序及權重 SHA，不能拿舊 R63 weights 切片 |
| N_area：R54 分位 ramp 面積 NN | 新 R47 基礎 encoder → 新 R54 ramp knots → 新 area NN | [round47_nn_encode.py](../src/round47_nn_encode.py) 的 fit_encoder/encode；[round54_numeric_encode.py](../src/round54_numeric_encode.py) 的 fit_state/transform；[round48_nn_train.py](../src/round48_nn_train.py) 的 run(data_path,out)，使用新 NPZ、30 epochs、seed42；不是 R74 近期加權 area NN |
| N_prior：R63 近期權重 prior-ratio NN | 新 encoder、新 R60 五折 OOF／full-fit prior、新日期 weights → 新 ratio NN | [round60_recency_price_prior.py](../src/round60_recency_price_prior.py) 的 fit_oof_prior/predict_prior；[round63_recency_weighted_prior_train.py](../src/round63_recency_weighted_prior_train.py) 的 run(data_path,out)，30 epochs、seed42；不是 R69 的 137／2026 seeds |

以上「可用入口」指可重用配方函數或隔離 trainer，不表示既有整支研究 driver 可直接接受新日期。R3/R6/R48/R54/R60/R63/R79 的驅動程式普遍硬綁三期 stage、固定 SHA、舊控制與拒絕覆寫保護，應保留其原檔，在新目錄寫單一新 origin driver。

**NN 的 ramp 寬度必須由新 fit 的 knots 自然決定。** 144 是基礎數值編碼固定寬度；area NN 舊 auxiliary 為590、development／confirmation 為599、final為603。不能硬寫「新窗口必為603」，也不能把後期603欄 state 複製過來。

## canonical 列、OOF、encoder 與權重

1. 保留清理後 train_features 的 87 欄和全局 canonical 455,255 列身分。沿用 [train_source_cache_indices.npy](../outputs/round6/geography/development/train_source_cache_indices.npy) 對原 455,784 列 cache 的映射與原 SHA；不要重複套去重或以 ID 去重。新 fit/query 索引依 [model_common.temporal_indices](../src/model_common.py) 的有限 trade_day_index 與固定起訖日產生。新索引、X frame、fit y、兩類地理 frame 各保存 SHA；必須互斥且完整覆蓋各自日期條件。本盤點未載入 feature table 計數，故不提供未驗證的新列數。
2. 基礎 deterministic X 可重用；但 R47 encoder 的 fit median／IQR／standard deviation／vocabulary、LGB category maps、R54 quantile knots 是學得狀態，全部重新用新 fit 建立，不能以「不含 y」為由直接沿用後期 state。area／prior NN 可共用同一份新 R47 encoder，之後 area 才加 ramps。
3. R60 的 fit_oof_prior(Xfit,yfit,source_indices) 在新 fit 重新建立五折 complement prior 和 full-fit prior。固定 sha256("42|"+group) 分折規則可重用，來源 index 必須為 canonical 原位置，不能改成0..n-1；重疊建案的 fold id 可能相同，但 prior 值仍須重新算，不能切片舊值。
4. 每個 OOF complement 的價格、日期 anchor、權重正規化都只用自身 reference fold；新 query 用全新 full-fit prior。這是 group OOF，不是 chronological OOF；無須先訓 R52/R60 的舊 NN，直接用 prior helper 建 state 即可。
5. 日期 weights 用 R60 _dates/_weights 的既有規則在新 full fit 計算，再保存為 float32，按 R63 配方供 N_prior 和 C_unit 共用；C_unit 再轉 float64 乘 A/100、不二次正規化。C_log 仍用 R46 自己的 float64 fit_weights 配方。365日半衰期規則可重用，舊日期 anchor／均值／向量不可重用。
6. 新 NPZ 僅有 fit labels。N_area 的 area factor=A/100；N_prior 的 factor=A×prior/1e6。NN 輸出 bias 亦由新 fit 目標／factor 初始化，不能複製舊 bias／checkpoint。

[round48_nn_experiment.prepare_npz](../src/round48_nn_experiment.py)、[round54_nn_experiment.prepare](../src/round54_nn_experiment.py)、[round60_nn_experiment.prepare](../src/round60_nn_experiment.py)、[round63_nn_experiment.py](../src/round63_nn_experiment.py) 可參考 dtype、NPZ鍵與證據格式；它們假設舊 stage 產物存在，因此新 driver 要組合純函數，而不是依序重跑所有歷史輪次。

## 純 X 地理可以如何節省重建

2023 年道路來源的實際 PBF header replication timestamp 為 **2023-01-01T21:21:53Z**，早於新 cutoff 2023-02-11（臺灣時間；UTC邊界2023-02-10T16:00:00Z）。來源：[osm_2023 audit](../outputs/geography/osm_2023/audit.json)、[R6 confirmation geography manifest](../outputs/round6/geography/confirmation/manifest.json)、[R39 confirmation geography manifest](../outputs/round39/geography/confirmation/manifest.json)。只讀現有來源證據即可，無須為本盤點下載外部資料。

| 地理項目 | 重用邊界 |
|---|---|
| 地址 parser／道路索引／2023 snapshot match | 固定規則、相同來源 snapshot、無價格 learned state；可按保存來源 SHA 重用。snapshot早於新origin，不代表道路實際完工日均已驗證 |
| R6 tree/NN geography | [confirmation/train.pkl](../outputs/round6/geography/confirmation/train.pkl) 是全部 canonical train 的455,255列純X八欄；可按新 fi/qi 重切。manifest記錄 SHA 0102788803d3f64f2935bf6797b37dae56fc889711c0b6d1c9871ea3414c8d3f。本盤點未反序列化或重新hash該大檔 |
| R39 spatial-only city/road fallback geography | 原 confirmation 的 fit.pkl／query.pkl 也是純X變換，不能照旧 partition 直接傳給新窗口。可用 manifest綁定的各自 canonical indices 把兩表映射回原位置，再取新 fi/qi；確認覆蓋、無重複、同一 snapshot／規則／dtype，且未改 R6 已有位置。新query跨越舊fit/query兩側，不能只取舊query.pkl |
| Geography feature state | R6八欄固定轉換及網格規則可沿用，但每個新模型的來源綁定、fit row hashes／schema manifest要重建。R47 encoder的地理欄統計仍必須 fit-only重算 |
| Reference／spatial價格state | 一律新fit重建。不能因其包含地理座標，就把內含成交殘差／global beta 的舊 spatial_state.pkl 視為純X cache |

R39原 confirmation 的 fit/query geo各為321,550／133,697列；新query在2024-02-10結束，位於原query終點之前，日期範圍理論上落於两表聯集內。**實際逐列覆蓋與SHA尚未在本盤點執行**，新driver需在fit前核驗；不通過就依相同2023道路規則重算所需列。

[round6_prepare_geo.verify_geography_chain](../src/round6_prepare_geo.py) 的現有介面把 origin硬綁year-08-11，且遍歷train/valid/test全部來源；不能僅呼叫confirmation就聲稱已驗證Feb origin，也不宜為新窗口無必要讀test。新wrapper應明確核2023 snapshot timestamp < Feb cutoff、只重綁所需train來源／索引，保留原source鏈。R39的 [prepare_partition](../src/round39_prepare_geo.py) 是可參考的純X補位實作，但同樣帶舊全局OUT／stage／預期值，不能就地改寫舊來源以通過。

## Reference、spatial、R89 與完整組合

- 原 v22：用新 Xfit/yfit 呼叫 [round2_neighborhood.fit_final_reference](../src/round2_neighborhood.py)，重新得到面積/單價mask、fill、案內global beta、fit residual及群索引；再用新 R39 fit geo 呼叫 [round8_spatial_reference.build](../src/round8_spatial_reference.py)。新query所有 reference/spatial state 均來自新 fit。
- R89：使用同一份新 global reference／spatial作共同起點，再呼叫 [round89_group_trend.fit_model](../src/round89_group_trend.py)(Xfit,yfit,Gfit,reference,spatial)。固定 type×use支持規則、delta IRLS、prior200／cap0.5不因新期結果改動；新期支持群與斜率可自然不同。它會重建一致 residual／spatial template，不能借用原三期R89 wrapper。
- 共享新六native生成相同三個base：U=.75*C_unit+.25*L；N=.75*(.5*N_area+.5*N_prior)+.25*L；H=.75*(.5*C_raw+.5*C_log)+.25*L。
- v22對三base各跑原R/S；R89對三base各呼叫完整 predict_model。保留 unsupported fallback及各支diag，最後各自取median。不能只改未來base的趨勢、卻留舊殘差／spatial beta，也不能共用另一base的confidence。
- 两完整未取整向量封存後，評估端才可取得新 query y；本次盤點尚未到此步。

## 最短依賴 DAG

~~~text
固定新origin／v22與R89配方／來源SHA
  ├─ canonical train X＋索引 → 新fi／qi → fit-only y → sealed query X
  ├─ 2023純X原地理＋R39補位地理 → canonical重映射與新geo manifest
  └─ frozen parameters／seed42
          ↓
     一份新fit資料契約
       ├─ C_raw ────────────────────────┐
       ├─ 新LGB category maps → L_unit ┤
       ├─ R46 fit weights → C_log ─────┤
       ├─ 新R47 encoder               │
       │    ├─ 新R54 knots → N_area ───┤
       │    └─ 新R60 OOF/full prior     │
       │         ＋新date32 → N_prior ─┤
       ├─ 新date32 → C_unit ───────────┘ → 共享U／N／H bases
       └─ 新R2 reference → 新R8 spatial
                  ├─ 原R/S × 三base → 完整v22
                  └─ 新R89 wrapper → R89 R/S × 三base → 完整R89
                             ↓
               封存兩向量＋來源／reload檢查
                             ↓
               同一query y的固定score／分群／paired
~~~

六native訓練、保存及全query重載核驗只做一次。GPU三CatBoost＋兩NN由主執行緒串行；LGB為CPU，可與來源整理排程，但不假定並行一定省時。OOF prior、encoder、reference與spatial不需要再訓歷史替代模型。

## 舊實測成本參考，非新窗口工時承諾

單位：秒；欄順為舊 development／confirmation／auxiliary。以下數值讀原metadata，不是本次新量測。

| 元件 | development | confirmation | auxiliary | 原量測範圍 |
|---|---:|---:|---:|---|
| C_raw | 149.883 | 153.221 | 109.984 | baseline elapsed，包含fit及該driver的預測／保存／評分等，非純fit |
| L_unit | 338.875 | 504.623 | 155.375 | baseline elapsed，同上 |
| C_log | 122.078 | 135.156 | 109.547 | R46 fit_metadata.elapsed_seconds，helper fit含其內部檢查 |
| C_unit | 165.062 | 190.125 | 151.250 | R79 model/execution.elapsed_seconds，外層fit段 |
| N_area | 18.516 | 24.093 | 10.344 | R54 training.fit_seconds |
| N_prior | 19.250 | 26.968 | 10.218 | R63 training.fit_seconds |
| 原global reference控制 | 8.460 | 9.623 | 5.141 | baseline v2_control.elapsed，包含建reference與預測／保存等 |
| R54 ramp準備 | 3.625 | 5.297 | 2.812 | preparation.elapsed，已有144欄輸入之後 |
| R60 OOF prior準備 | 8.453 | 11.453 | 5.063 | prior_preparation.prior_prepare_seconds，含該prepare內其他工作 |

來源：baseline的 [development目錄](../outputs/round3/baseline/development)、[confirmation目錄](../outputs/round3/baseline/confirmation)、[auxiliary目錄](../outputs/round6/baseline/auxiliary)下各component JSON；[R46 development fit metadata](../outputs/round46/development/model/fit_metadata.json)、[R79 development execution](../outputs/round79/development/model/execution.json)、[R54 development training](../outputs/round54/development/model/training.json)、[R63 development training](../outputs/round63/development/model/training.json)及各自同名confirmation/auxiliary；[R54 preparation](../outputs/round54/development/preparation.json)、[R60 prior preparation](../outputs/round60/development/prior_preparation.json)。

兩NN含subprocess啟動／保存／重載的實測外層時間另為R54 23.516／29.687／14.672秒、R63 22.875／30.750／13.672秒，不能與fit_seconds重複相加。未量測的新窗口資料重綁、encoder、地理驗證、R89wrapper、全部reload與診斷成本維持未知；不以這張不同scope的舊表加總成新ETA，也不把執行秒數等同開發聊天總耗時。

## 開跑前剩餘工作與最短安全落點

1. 在獨立新output目錄凍結新日期、兩配方、來源與評估項目，實際產生新 fi/qi並核驗列數／SHA；此步本盤點未做。
2. 寫新origin專用資料／state wrapper，重用上列純API及trainer；不要改舊plan/SHA/OUT來假裝新期已存在。使用新NPZ後直接調兩trainer即可，不用重跑R47、R48、R52、R54、R60、R63的全部歷史比較鏈。
3. 先完成2023純X地理重映射、fit/query隔離和train-only encoder/prior檢查，再安排六native；所有模型保留seed／參數／fit索引、時間、失敗日志及完整重載證據。
4. 新origin scorer可沿 [round3_evaluate.score](../src/round3_evaluate.py) 的公式，但 [candidate_diagnostics_v3.py](../src/candidate_diagnostics_v3.py) 與 [paired_candidate_bootstrap.py](../src/paired_candidate_bootstrap.py) 直接綁R48三期indices，**不能原封不動拿來評新期**。需新的alignment adapter，固定seen/new權重與cluster/month規則。此次不新增依賴test資料的統計。
5. 完整v22/R89同源同期比較後，再保留整體、群組、尾部與配對區間的正反證據。只有計畫或保存artifact不等於已完成乾淨環境從頭重訓。

本盤點沒有改原模型、CSV、state、根交接、GitHub包或正式簡報報告；沒有預先認定R89值得採用。下一窗口是否執行由主執行緒依完整R89證據決定。
