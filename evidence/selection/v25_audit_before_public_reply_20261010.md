# v25 最新選定／GitHub 更新只讀審查

審查日期：2026-10-10（依此次使用者訊息與客戶端日期）。審查者僅核對既有檔案，未新增 fit、query 真價、Kaggle 操作或 GitHub mutation。產物採拒絕覆寫。原 HANDOFF／LOG 已先讀；以下是對此次更新的建議，不改寫原研究結果。

## 結論與本次實查

- 使用者的最新原話「現在最好的版本是 v25所以我會以v25為繳交的資料，你同時也必須更新 GitHub」足以將現行提交選定改為 v25。這是使用者選擇，**沒有新增數值分數或 Private 證據**。
- `outputs/submission_v25_trial.csv`：47,073列、734,489 bytes；UTF-8 BOM、欄名 `編號,總價元`、全部 ID／順序與發行綁定 sample 完全相同，唯一且非空；所有價格是正整數，與已封存 R107 median 在最後一次 `maximum(rint(p),1)` 的全部值完全相同。
- CSV SHA256：`36231cd17a6cbeafefeb62a7989d207fa20d6c5f7aaf99857e4d9addd25597c1`。
- release SHA256：`661b0cc0f1cf4eef9234187a51929e4eb8240ccf43bc1606f71a1086f865f189`。release 內 Public/Private 均為 null；它記錄 2026-10-08「探索未選定、v22保留」，是正確的歷史狀態，**不得為本次改選而回寫 release/request/gates**。
- request 綁定的 44 個 source（43支 src Python＋1支 CSV builder）現存原檔 SHA 全部符合，且現有 GitHub 工作副本的 research/src 或 evidence 路徑全部逐 bytes 相同。
- 四支主要 R107 程式、三份方法／穩定性報告、final request/seal/release/fit_execution、model run/training 聚合 metadata、R107結果及R113的R107結果均已在現有 GitHub 工作副本，與根工作區相同。

## 選定更新必要清單

1. 新建獨立 2026-10-10 `user_selection.json`，保留原話、v25 CSV 路徑／SHA、release 路徑／SHA、Public/Private=null、選擇由使用者作出、無自動 Kaggle 操作。保留 2026-09-28 v22 selection 原檔。
2. 更新 HANDOFF、追加 LOG、當前報告／GitHub README 的最上層最新入口，明確指向 v25；保留舊 dated 選定、正式 v22 簡報／影片與原報告的歷史身分。使用者未要求此次重做所有媒體，不能只改標題便稱簡報已講 v25。
3. GitHub 新增當前 v25 方法／程式／重現範圍說明與選定證據，連到既有 R107 原碼和 R113 反證；舊 `predict_v22.py` 與17項 v22 manifest仍屬 v22。**不能叫使用者直接執行 predict_v22.py 然後把檔名改為 v25**。
4. 核對／刷新 SOURCE_MANIFEST，更新 exporter 新選定與關鍵文件白名單，執行新交接 ZIP 與相對連結／檔案檢查，推送既有私人 repo 並記錄真正 remote commit。這是授權的 repo 更新，不代表 Kaggle 已上傳或已勾選。

## v25 方法與反證應如何寫

v25＝R107 全量模型：保留 v22 三分支與同一 reference/spatial/median；只將 prior-ratio NN 的日期加權 L1 改為日期加權 SmoothL1，beta固定0.1百萬元（10萬元）。144數值維度＋23類別欄、512/256/128 MLP、Softplus倍率輸出、30 epochs／seed42均固定。全量467,221 fit列／47,073 test列，只有這支 NN 新 fit 一次。

NN 分支先作 `.75*(.5*saved_area + .5*new_prior) + .25*saved_LGB`，再套原R/S，與原log與unit完整分支取逐列中位數；最後一次轉正整數。價格標籤來自課程 fit，未查內政部成交系統、無 test 真價、無按 test ID 調價。

既有離線對照 v22：development -1,415.88元、confirmation -1,525.54元（0.1103%）、auxiliary +883.66元；後續 February -303.06元。原探索門檻 true，強推薦門檻 false；不得因使用者選定就改研究 gate。confirmation 既有案 +213.10元、auxiliary 新舊案均略退步；v24在原三窗仍有更好的離線點分數。三個8月窗的建案 bootstrap 整體區間都跨0，February兩種抽樣區間亦跨0。反覆重用且重疊時間窗、保存對照非同期 matched 重訓、單 seed，無法把條件 bootstrap 解釋為 Private 勝率。

## 真實重現範圍

- **已完成**：新 prior NN 全量 fit（training fit_seconds=48.875；driver80.234秒exit0）；training記錄 saved_model_reload_exact=true、reload_rows=47,073；保存新模型及五層預測、最後 CSV 全部 ID／值／SHA核對。此次審查另只讀重核 CSV、sample 與 sealed median。
- **已完成但範圍不同**：原 v22 六模型在同機新套件環境重播，屬 v22；v25 final 的原五個模型使用保存向量與既有R/S組合重播，不是本批重新跑五個 native 模型推論。
- **未完成**：v25 的乾淨環境六模型完整推論、raw Excel→所有模型從零重訓、新電腦端到端、原OSM地理重建。現行 GitHub production 僅有 predict_v22.py；v25 research driver 依賴完整本機私有 prepared/state/向量與歷史SHA，拒絕覆寫已存在輸出，不是 clone 後一鍵可跑的 production 推論入口。
- GitHub保存程式與聚合證據，沒有課程交易表、模型權重、逐筆價格、prepared arrays、price-derived private states、提交CSV。來源齊全可追溯不等於 GitHub snapshot 自帶全部訓練輸入。
- v25 Public **未知**，Private **未知**。v22 Public 1,998,568.5706元只能冠以v22；此次「最好」不構成v25新的平台分數或Private排名。

## 已核對 source 複本

| 原工作區 source | SHA256 | GitHub 工作副本 |
| --- | --- | --- |
| `outputs/research_resume_20261008/build_submission.mjs` | `1d7d693c89ab5abd279458cec5855dadb7b465e36e474de838ee65ef8bd29d47` | `evidence/outputs/research_resume_20261008/build_submission.mjs` |
| `src/house_features.py` | `108e42d4429961a10a704f85bd66791599d904e6e96381c4c2f6dd4275316388` | `research/src/house_features.py` |
| `src/model_common.py` | `efbe4aafe5cf7c0a7c78b002fb8e89e30db148c226e28284551e0318f3e7947f` | `research/src/model_common.py` |
| `src/prepare_data.py` | `1b2a2c4c947aebdeff5d268300fac6f2f56331a850ffcc6eb6149e7f255ebde6` | `research/src/prepare_data.py` |
| `src/round100_contract.py` | `f6483f2e1e92f8286ea7aed081080c287888b1b6c9f4a3a519964f63ffdd77f5` | `research/src/round100_contract.py` |
| `src/round100_temporal_prior_evaluate_v2.py` | `c12e69764470b9f47a33f1fc1dbae0063a31aff1bfe5568f487d48a08f8792bf` | `research/src/round100_temporal_prior_evaluate_v2.py` |
| `src/round104_composition.py` | `6f9998e71fc3e22b69bac29acb0501ca1146e5e9263b2c7d6015d7b6fd7a83f0` | `research/src/round104_composition.py` |
| `src/round107_108_research.py` | `acf6249f19c54cf5175269688a8fe57d376ae5af50b34be2b57a24c5e14103ea` | `research/src/round107_108_research.py` |
| `src/round107_final_candidate.py` | `8252533fc4f6fea412d9ad95e20f9cef4384b00232db528d2a1591b157afd085` | `research/src/round107_final_candidate.py` |
| `src/round107_full_release.py` | `594dd283282bde33058d7e2e72f856bcba26d6157d2ea02469d785fb2ed18fb6` | `research/src/round107_full_release.py` |
| `src/round107_weighted_smooth_train.py` | `f2330726165b1a102c893211268734c89e981b751f45167b82ae1c940917b36e` | `research/src/round107_weighted_smooth_train.py` |
| `src/round108_single_ctr_unit.py` | `0e46eae335d13e904b65a23eb120cd24f48bf5234b55b8c9c086c0a17820e6e8` | `research/src/round108_single_ctr_unit.py` |
| `src/round12_experiment.py` | `c47e35601c8a4ed4d442945a057e01012da84dc023a00879ea188f851532ff06` | `research/src/round12_experiment.py` |
| `src/round12_log_geo.py` | `e16fbe33928bd1a635e5de8bd717f223263847b718aef373fb933224b13d6c42` | `research/src/round12_log_geo.py` |
| `src/round19_unit_geo.py` | `fac832e6c81871833a136d05759c7f0926fb749d2320a9c39b581e3986dc75cf` | `research/src/round19_unit_geo.py` |
| `src/round2_neighborhood.py` | `cae244c40e6616d6c72f3aff7c3423a50a5bfbfbf35df33a7d044186eb491ae9` | `research/src/round2_neighborhood.py` |
| `src/round33_reproduction.py` | `2cb34f2a1047b57b78c14e7069e6ef606b621ec282d730c461935e43692d37b2` | `research/src/round33_reproduction.py` |
| `src/round34_trial_release.py` | `ba6a7aa57409c2e22abeec2465d0dc4ff2eca86480bf106fc36f9c1e639d89e3` | `research/src/round34_trial_release.py` |
| `src/round35_reproduction.py` | `495f4649da118011210dbc4769303f218b5bcd6be37367f2a049adf01604f421` | `research/src/round35_reproduction.py` |
| `src/round36_combination.py` | `114ec27f122b8935fe21fcf2d9151c3a6350a127e50d40cb45a214a5cc278059` | `research/src/round36_combination.py` |
| `src/round38_fixed_average.py` | `6058e972f831be9c6ae756966a807374c527d92c353f01a982b8c86ec41608bf` | `research/src/round38_fixed_average.py` |
| `src/round39_city_road_experiment.py` | `8c26637712304043ed49aca18f53099d5ec1249f71718458a17f1cbc35ace4fd` | `research/src/round39_city_road_experiment.py` |
| `src/round3_backtest.py` | `9d43e18b938aef9c729c546ed438d7b06948439281d33940691271d13e977c47` | `research/src/round3_backtest.py` |
| `src/round3_evaluate.py` | `e00b2b09d4f0ed8d6fbd547f00ebbb2d8096136a567a56eb85ddacc6ec1f4271` | `research/src/round3_evaluate.py` |
| `src/round3_reference.py` | `4d78963763458e5fd2dab2847a45cd7f356c58dc2288081c1f4006cd8dba3631` | `research/src/round3_reference.py` |
| `src/round46_experiment.py` | `e56c758f55b8dc27c4b5fae613c7ca0b169de3645c09b8244c613123a0ea4405` | `research/src/round46_experiment.py` |
| `src/round46_recency_log.py` | `48364428cdf6281205666470c8f61662fdb8aa65a18fc5a67d189d60cf7a8fa8` | `research/src/round46_recency_log.py` |
| `src/round46_trial_release.py` | `2089bdc626f55cba43c8863038ccd6087ad9d5b24085ac948b4a465ba12bdd19` | `research/src/round46_trial_release.py` |
| `src/round47_nn_encode.py` | `6bee3297f761d2cda0a0b9224aebc61e6748c9e56d8738d1fe344ea0abda4c47` | `research/src/round47_nn_encode.py` |
| `src/round47_nn_train.py` | `68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7` | `research/src/round47_nn_train.py` |
| `src/round48_final_branch.py` | `9775586eeadf382c6bc254a50cd3fbadd8f1d0e6e882cee4abf301efd7a61442` | `research/src/round48_final_branch.py` |
| `src/round48_nn_experiment.py` | `9bedbda9ed7e47213aadeb9852deee4f3878a456f57620694ca750e9d2e7a9a0` | `research/src/round48_nn_experiment.py` |
| `src/round63_recency_weighted_prior_train.py` | `26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a` | `research/src/round63_recency_weighted_prior_train.py` |
| `src/round6_common.py` | `ae3e6d7950514ff215e71fc8642af938b9496e41227b1cdbdc2bd2801cd90c0a` | `research/src/round6_common.py` |
| `src/round6_evaluate.py` | `836687b1213d92f23183df60d21e9281d2ae9c9b96aa3ead54134b406ed3ef26` | `research/src/round6_evaluate.py` |
| `src/round6_geo.py` | `df1454829b919e2a10ac264dd4e3f597367187174607a6d32abb20f40ef060a0` | `research/src/round6_geo.py` |
| `src/round6_prepare_geo.py` | `b8be469fc1217af9487534e1fb2ed048704ba9008e966a382d2f7ea8108344f0` | `research/src/round6_prepare_geo.py` |
| `src/round76_77_final.py` | `ed54e7d0779c2156414510704e8ffc90cf1b42f65819d3b6cd5c3f08c143f26b` | `research/src/round76_77_final.py` |
| `src/round79_final.py` | `c6ad4103c7e75ff1954954b234a671e842a4918618d5ceaadc6ecb4881b97684` | `research/src/round79_final.py` |
| `src/round79_recency_unit.py` | `9a124322a139e5621c28d26a6e7ab95385c73507f017a2765d9eaaa0eef39cd2` | `research/src/round79_recency_unit.py` |
| `src/round8_experiment.py` | `11d6b5f5b7a1d4f39b2607d8cc4257f311fe94afcddc89314cf28cf4b0d2bd41` | `research/src/round8_experiment.py` |
| `src/round8_spatial_reference.py` | `de86397255d654f363f3b746d579923fe689aa41d9dc16335d0922fea05e1c32` | `research/src/round8_spatial_reference.py` |
| `src/train_lightgbm.py` | `f889de86d3088bd17483245f2278c8da7c554640e518b155196a6db5f0dd8b6d` | `research/src/train_lightgbm.py` |
| `src/v3_cluster_stability.py` | `46622f859660e40bd78fbbd612db53b5a9bd5069f55eb9e7b169e19cc134d1c9` | `research/src/v3_cluster_stability.py` |

本審查不宣稱已讀遍43支依賴的全部語義；已全文／定點核對本次相關方法、final context、release driver前置與fit/release介面、trainer損失/契約、報告與聚合 metadata，並對全部44來源作bytes身分核對。
