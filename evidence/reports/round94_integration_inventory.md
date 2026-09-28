# R94 整合契約盤點（實作前）

2026-09-29，Asia/Taipei。只讀 source、保存計畫／小型 metadata 與檔案路徑存在性；本盤點沒有載入模型、NPZ／NPY 陣列、query 價格或 test，沒有 fit、predict 或候選評分，沒有動 GitHub 快照。後續 driver 尚待實作及凍結，不能把下面步驟當已完成。

結論：可以只替換 v22 NN 分支中的 R63 prior NN；原 R54 area NN、LGB、R/S、log 與 R79 unit 分支可沿同窗保存產物。**不需要重新訓練或推論舊 R63 模型作控制**：綁定其保存 native TWD，重播 R71 seed42 中間向量及 v22 median 即可。舊 trainer 的全 query 重載證據已在 training.json，但本輪若只讀保存向量，不稱為重新跑過舊神經網路。

[正式計畫](round94_plan.md) SHA c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a。它將可行性稿的第二層 full attention 改為 CLS-only query，且 residual／FFN 只保留 CLS；此為真 fit 前明確改編，不能沿用舊稿宣稱兩層 full attention。

## 一、三個年度與九成員輸入

stage 依序 development→confirmation→auxiliary；fi/qi 都是去重後 canonical 455,255 列的原位置，不是 raw cache 映射或重設後 0..n-1。外層只用當期 cutoff 前 train，未併 valid。

| stage | query 日期 | fit／query 列 | R63 prepared 路徑與容器 SHA |
|---|---|---:|---|
| development | 2022-08-11 至 2023-08-10 | 240,316／81,234 | `outputs/round63/development/prepared.npz`；`e1411757d89ad74fdd5b6cd3622f6c8d26fdc9932859ae49eaa9be9705449bea` |
| confirmation | 2023-08-11 至 2024-08-10 | 321,550／133,697 | `outputs/round63/confirmation/prepared.npz`；`4d4b7866f4f94068e60585b6351fd1a669be184d9e1f8d14cc4a5eac652a0aff` |
| auxiliary | 2021-08-11 至 2022-08-10 | 134,805／105,511 | `outputs/round63/auxiliary/prepared.npz`；`b5faa0c320a67332f7bccd8d199be61cfc35ae9b098eb70656302c08923a19db` |

每期 prepared 嚴格只有以下 9 成員；任何 query_y／target 等額外成員應拒絕，不將 query labels 放進 trainer 子程序：

| 成員 | dtype／shape | 意義 |
|---|---|---|
| train_num、query_num | float32 [n,144] | 72 個尺度化值後接 72 非有限旗標 |
| train_cat、query_cat | int64 [n,23] | 原 fit 詞表碼；未知／稀有0 |
| cardinalities | int64 [23] | 對齊原類別順序，每欄含未知碼 |
| train_y_million | float32 [n_fit] | 只含 fit 總價／1e6 |
| train_price_factor_million | float32 [n_fit] | float32(A64×P_OOF64／1e6) |
| query_price_factor_million | float32 [n_query] | float32(A64×P_full-fit64／1e6) |
| train_sample_weight | float32 [n_fit] | 原全 fit 365 日衰減正規化後 cast32，不再正規化 |

這張表的 train/query 成員對合計九個，不是九種不同陣列對。既有 R63 僅將第九個日期權重加到原 R60 八成員，其餘 byte 不变；R60 前六成員又來自 R52→R48→R47。因子先乘 float64 再 cast32，不能改成先 cast prior／area 或多乘一次面積。推論 `float64(ratio)*float64(saved_query_factor32)*1e6`，無 clipping。新 loss 只乘原日期權重，不能加入 R93 的面積×密度權重。

## 二、encoder、prior、indices 與 date/factor 精確來源

以下 `{s}` 只允許上表三個 stage；各列已確認三期對應路徑存在。

| 角色 | 精確路徑／欄位 | 來源 authority／必要核對 |
|---|---|---|
| canonical fi/qi | `outputs/round63/{s}/fit_indices.npy`、`query_indices.npy` | R63 results stage artifact SHA；R48/R79 plan split_contract 的 ordered int64 content SHA；與 R47/R48/R54/R60/R71/R79/R83 同期相同 |
| fit-only encoder | `outputs/round47/{s}/encoder_state.json`；原樣複本 `outputs/round48/{s}/encoder_state.json` | R48 area_preparation 的 source_encoder_sha256，R47/R48 results；兩檔相同 bytes |
| encoder 值來源 | `outputs/round47/{s}/prepared.npz` → R48 → R52 → R60 → R63 | R60 prior_preparation.unchanged_six_array_sha256；R63 all_eight_R60_arrays_reused_without_changes |
| original NN geo | `outputs/round6/geography/{s}/train.pkl` | R48 plan bindings.original_geo；canonical 完整 geo 依 fi/qi 抽列；**不是 R39 補點 geo** |
| OOF train prior | `outputs/round60/{s}/train_oof_unit_prior.npy` | R60 results stage artifact SHA；prior_preparation.prior_evidence.oof_prior_sha256 |
| full-fit query prior | `outputs/round60/{s}/query_unit_prior.npy` | 同期 R60 artifact SHA；由同目錄 price_prior_state.json 對 Xq 查表，無 query labels |
| prior state | `outputs/round60/{s}/price_prior_state.json` | prior_preparation.prior_state_sha256（檔案 SHA）與 prior_evidence.state_content_sha256（內容 SHA），兩者不可混用 |
| prior fold IDs | `outputs/round60/{s}/prior_fold_ids.npy` | prior_evidence.fold_ids_sha256；5 折原 grouping 規則；與 R52 相同 |
| source_indices | **沒有另存 source_indices.npy**；呼叫 `fit_oof_prior(Xfit,yfit,fi)` | prior_evidence.source_indices_sha256 必須等於 canonical fi content SHA，不能傳 raw cache map／reset index |
| date weights | R63 prepared 成員 train_sample_weight | R63 preparation.train_sample_weight_sha256、date_weight_summary_float64 含日期／flags／rule hashes；training 同值 SHA |
| fit/query factors | R63 prepared 的兩個 *_price_factor_million 成員 | R60 prior_preparation 及 R63 training 的兩個 factor content SHA 相同；driver重建 A×P／1e6 並 cast32 精確比對 |

R47 encoder 的 feature_columns／input_dtypes／numeric_columns／categorical_columns／cardinalities／state_content_sha256／fit_rows 須核對。95 欄由原 87＋8 原 R6 地理形成，72 numeric／23 cat；numeric median 補缺後，以 IQR（退回 std，再退1）經 asinh，再用 fit 轉換後均值／std 標準化，無截尾。詞表只留 fit 至少5筆，排序後編碼，未知／稀有0。encoder 為 pure-X，prior 含 fit 價格，不能混稱全部前處理無標籤。

R60 的五折以 canonical fi 配合 cluster_keys，group hash `sha256('42|'+key)` 前8 bytes little-endian mod5；held fold 的 prior 只用 complement 價格與日期，最後另建 full-fit query state。這是 group OOF，不是時間 OOF；所有列已在外層 cutoff 以前。新 driver可沿 R86 的 fit-only 完整 replay：OOF、folds、state、evidence 全部相同，query prior由該full-fit state重播；不是換新 prior 候選。

日期的 formula 是 `exp2(-(max_valid_fit_day-day)/365)` 除有效 fit 列的均值；缺日期 neutral1，cast32後不再正規化；NN sample weights是全fit日期，與 prior各fold complement内部日期正規化分開。

| stage | date float32 content SHA | fit factor32 SHA | query factor32 SHA |
|---|---|---|---|
| development | `bd22ee426159dc495434d024ff111ae39a19cd77e92d7980e9b333b3acd00af6` | `6f24b6cfaf044eb4494c637da5a4a79b48b25e3e1dee9d9a7c5a27185b580365` | `d36bf6a2eceebab34522992939217b4d7e6e9c85e1878e0114966e4d14f6ef0f` |
| confirmation | `d1150f7935e78f75d28b6f6b8e004a97a5a90cc284e3a89c97ed0cf10b1e950e` | `3a5d9c3759af6a01cda823a44a5f7a8f3f2acc22a68ebece7dfbaf65f7638405` | `7b77243f2ac03041d6a574dd500044311d78764f50404b3d3fdfc0ce9fc94f32` |
| auxiliary | `1e0259dafd5d7b510f7a7490677202d9de5f065a6a5547636e280dde3cb7c39f` | `b8ae2fe88390340fb62ce94a2b22ef5544e4fcc182d95ca9a028304c23ab3624` | `f38d04a34b7510397cb2f936d1bfd91d023aaee71a982063b33f9e8bb451de2f` |

## 三、只替換 prior NN 的向量與 R/S 綁定

| 角色 | 路徑（各期 s） | 使用方式 |
|---|---|---|
| F：R54 area native | `outputs/round54/{s}/model/predictions_twd.npy` | 已是總價元，不再乘 A；authority R54 results stage artifact_sha256 |
| T_old：R63 prior native | `outputs/round63/{s}/model/predictions_twd.npy` | 已是總價元，不再乘 prior；authority R63 results，並核 R71 native_seed_bindings['42'] 相同 |
| R54 encoder／ramp | R48 encoder；`outputs/round54/{s}/numeric_state.json`、preparation.json、prepared.npz | F 自己用的 ramp 寬度為599／599／590；**R94 仍用原144，不能誤餵 R54 ramp** |
| 原 NN mean/base/R/S | `outputs/round71/{s}/seed42/native_mean.npy`、base.npy、reference.npy、nn_spatial.npy | authority R71 results stage artifacts；依原順序重播，不平均 seed137/2026 |
| 原 R71 diagnostics | 同 seed42 目錄 reference_diagnostics.pkl、spatial_diagnostics.pkl | 重播 R/S及逐列 fallback，與保存 diag 核對 |
| 保留 log 分支 | `outputs/round46/{s}/recency_log_spatial.npy` | R79 plan bindings.paths.log／input SHA |
| 保留 v22 unit 分支 | `outputs/round79/{s}/unit_spatial.npy` | R79 results stage artifact SHA；不是 R40 unit、R76或R93 |
| 完整 v22／v24 | `outputs/round79/{s}/predictions.npy`、`outputs/round83/{s}/predictions.npy` | 各自 results、indices及完整 metrics |

| stage | L：原 unit LGB | reference state | spatial state／query geo |
|---|---|---|---|
| development | `outputs/round3/baseline/development/lightgbm_l1_unit63.npy` | `outputs/round3/baseline/development/reference_v2.pkl` | `outputs/round39/evaluation/development/spatial_state.pkl`；`outputs/round39/geography/development/query.pkl` |
| confirmation | `outputs/round3/baseline/confirmation/lightgbm_l1_unit63.npy` | `outputs/round3/baseline/confirmation/reference_v2.pkl` | `outputs/round39/evaluation/confirmation/spatial_state.pkl`；`outputs/round39/geography/confirmation/query.pkl` |
| auxiliary | `outputs/round6/baseline/auxiliary/lightgbm_l1_unit63.npy` | `outputs/round6/baseline/auxiliary/reference_v2.pkl` | `outputs/round39/evaluation/auxiliary/spatial_state.pkl`；`outputs/round39/geography/auxiliary/query.pkl` |

這些檔案均由 R48 plan input_sha256 綁定；spatial 的 state_summary.fit_X_sha256、參數及 query_used_in_build=False，reference.fit_report.fitting_rows，query geo 的 R39 manifest frame SHA/index 需對齊。

控制重播：`M_old=.5*F+.5*T_old`，`B_old=.75*M_old+.25*L`，呼叫原 `predict_reference(...,variant='trend',return_diagnostics=True)` 及原 spatial.predict。與 R71 seed42 四層逐位比對；再 `median(N_old,R46_log,R79_unit)` 逐位比對完整 v22。**R63/{s}/base.npy 還含舊 raw 混合，不是此 B_old；R71/{s}/predictions.npy 外層仍含 R40 unit，也不是完整 v22。**

候選只將 T_old 換成新 T；`M_new=.5*F+.5*T`、`B_new=.75*M_new+.25*L`、沿同窗原R/S一次，再 `median(N_new,R46_log,R79_unit)`。保留其他分支 bytes；R/S須對新base重算，不平均已修正的 NN 向量。控制及候選各自一次 R/S，所以driver每期一般共2次reference＋2次spatial；舊 NN模型predict仍0。

## 四、trainer 環境、保存與標籤隔離

既有命令為 `C:/NYCU_ML/.venv_torch/Scripts/python.exe -u -B -X utf8 src/round63_recency_weighted_prior_train.py --data <R63 prepared> --out <新model目錄>`；driver 為一般 .venv，不 import torch，由 subprocess 串行調 trainer。保存 metadata顯示 PyTorch 2.13.0+cu126、CUDA12.6、numpy2.3.5，舊三期 fit19.250／26.968／10.218秒及峰值301.08／405.52／231.67MiB只是歷史MLP成本，不是attention ETA。

R63 save 是 `torch.save({numeric_count,cardinalities,configuration,state_dict:CPU tensors},model.pt)`；load `torch.load(...,map_location='cpu',weights_only=True)`，重建同architecture、load_state_dict再同batch全query predict。run.json記來源／配置／NPZ／factor/date hashes；training.json記30epoch、fit與elapsed、model reload、資源。舊 R63 只保存 predictions_twd.npy，沒有獨立 raw ratio 檔；不能以 TWD/factor 倒除的浮點值假稱原 raw ratio 已重播。

R94 應新檔使用自己的architecture/config/source identity，不能用 R63 EmbeddingMLP load新attention權重。依主線 trainer 契約新增 predictions_ratio.npy及predictions_twd.npy，保存與重載全query ratio和TWD逐位一致，推論batch256。不同batch／排列的語義測例容許已聲明的小浮點誤差，與同batch精確重載分開。

九成員 NPZ 僅 fit y，trainer fit/head初始化/loss沒有 query y或early stop。原共用 driver loader會反序列化 canonical y，再只傳 fit y 做來源重播，query y 只給 scorer；如果沿此設計，必須如實記錄這個程序邊界，不能說整個driver從未載過query價格。該資訊不跨 subprocess，query shape/encoded/factor不用于選epoch或超參數。

## 五、最少 main driver 整合步驟與 generic介面

1. 固定plan／trainer／architecture／upstream／CPU/smoke收據及SHA；驗resource門檻後才允許主線launch；新OUT拒絕覆寫，保留protected CSV／selector／v22選定。事前記命令、runtime、唯一seed、三fit順序。
2. 每期先核canonical fi/qi、cutoff／frames／geo與R63 prepared原容器SHA；同窗encoder/cardinalities/encoded arrays、R60 OOF/folds/state/factors及float32 date重播精確，全部來源檢查在fit_started以前。
3. 用保存T_old重播R71 seed42 mean→base→R/S→v22median完整向量／diag，並重播v11/v12/v5/v22/v24完整控制metrics；不再載舊NN。
4. .venv_torch只跑一個新attention trainer，核完整configuration與輸入身分、最後30epoch、無querylabels、raw/TWD全queryreload、輸出shape／正有限；失敗保存不靜默換batch或覆寫。
5. 保存候選 native ratio/TWD、mean、base、reference、nn_spatial、predictions及diag，完整三期scorer，不以分數跳過其餘期；input/source/protected前後SHA一致。只記結果，不在driver自動選型／全量／CSV。
6. generic不依賴unit或NN中間檔名：根results須`checked_stages`依三期順序；每stage `metrics`至少含v11/v12/v5/R94，建議同時v22/v24；`fit_indices.npy`、`query_indices.npy`、`predictions.npy`、`prediction_sha256`及artifact_sha256一致即可。完整score字典要保留rows/overall/weighted/strata/months/forecast_quarters/cities，非只overall。`new_NN_fits=3`與native樹fit0須標對。

既有可用後續命令（**此盤點未執行**）：

```text
.venv/Scripts/python.exe -B -X utf8 src/candidate_diagnostics_v3.py --round 94 --candidate R94 --extra-control v22:79:R79 --extra-control v24:83:R83
.venv/Scripts/python.exe -B -X utf8 src/paired_candidate_bootstrap.py --round 94 --candidate R94 --control-round 79 --control-name v22 --control-key R79
.venv/Scripts/python.exe -B -X utf8 src/paired_candidate_bootstrap.py --round 94 --candidate R94 --control-round 83 --control-name v24 --control-key R83
```

generic／paired只讀最終保存向量及standard metrics，不要求`unit_branch_metrics`。R94可另存`nn_branch_metrics`及旧NN层控制供分析，不能照抄R93的樹容量／Pool欄位。兩工具會在評分程序讀歷史querylabels，與trainer無querylabels不矛盾。主線另做decision/report，不能只將R93 review字串換94。

## 六、核對來源與範圍

本盤點只核source與metadata內相互一致的身分／路徑，不聲稱已在本輪重算NPZ/模型／OOF或完成真query重播；實作後由main收據記實際執行。R90新cutoff須重建其全部狀態，本表不可拿後期模型直接套到R90。

- [reports/round94_plan.md](../reports/round94_plan.md)：`c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a`
- [reports/attention_regression_feasibility_after_r93.md](../reports/attention_regression_feasibility_after_r93.md)：`93a55705f8c04677b5cc45a7f66ea83f244cf0c84127bfcd83f7f5eb2480a4aa`
- [src/round63_recency_weighted_prior_train.py](../src/round63_recency_weighted_prior_train.py)：`26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a`
- [src/round63_nn_experiment.py](../src/round63_nn_experiment.py)：`70d946c76df35761f55658f7b1ad0f04109ece2cf5901978bae6f0ce97ea513e`
- [src/round60_recency_price_prior.py](../src/round60_recency_price_prior.py)：`4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f`
- [src/round60_nn_experiment.py](../src/round60_nn_experiment.py)：`4f470c273052a953da944496b23a69fe248365199fbacd1104b78a66783aec3b`
- [src/round47_nn_encode.py](../src/round47_nn_encode.py)：`6bee3297f761d2cda0a0b9224aebc61e6748c9e56d8738d1fe344ea0abda4c47`
- [src/round47_nn_train.py](../src/round47_nn_train.py)：`68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7`
- [src/round48_nn_experiment.py](../src/round48_nn_experiment.py)：`9bedbda9ed7e47213aadeb9852deee4f3878a456f57620694ca750e9d2e7a9a0`
- [src/round54_nn_experiment.py](../src/round54_nn_experiment.py)：`57ebd44c8f27875476fd7b7122970be2c670b319d5b79690bf480d638d074918`
- [src/round71_missing_cell.py](../src/round71_missing_cell.py)：`6fad31c31699d973779031bd4c93e3b846455f6aabc1f4f95e1e92eff7dc60fd`
- [src/round93_experiment.py](../src/round93_experiment.py)：`f92361e80f4fe0659e309733807fd65015ca421595918cb416f34a267590c4fb`
- [src/candidate_diagnostics_v3.py](../src/candidate_diagnostics_v3.py)：`49cf3a89e8a1d60defb8f49bc63ef0f8f88ded528af1e3687684d70c8a4ae128`
- [src/paired_candidate_bootstrap.py](../src/paired_candidate_bootstrap.py)：`3761c1cb09abe4e26412bf3aea48273ce0a4bad10b5a408e23d8fae488903bb0`
