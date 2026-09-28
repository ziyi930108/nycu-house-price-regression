# R90 新切點：Attention 後續驗證模型與工程缺口

日期：2026-09-29（Asia/Taipei）。本次只讀 source、小型 metadata、既有報告與目錄；沒有載入價格或預測陣列、模型，沒有 prepare、fit、predict 或評分。接獲任務時主線正在執行 R95 三期，本文件不判斷它是否過門檻，也不授權新切點訓練。原 R90／R94／R95 計畫及程式均未修改。

## 總數：完整三方比較需 8 次 native fit

新切點為 **fit < 2023-02-11；query 2023-02-11～2024-02-10，含末日**。完整 v22、完整 v24 與固定 Attention 候選共需 **5 個樹模型＋3 個 NN＝8 次 native fit**。共享分量只訓一次，不是三套各訓六個。只比較 v22 與 Attention 則為 7 次；[R94 計畫](round94_plan.md)「六模型＋候選」沒有包含完整 v24 額外的 seed 137 樹，不能把 7 次說成已涵蓋 v24。

| 新窗口元件 | 次數 | 使用者／既有入口 | 既有 R90 helper 能力與缺口 |
|---|---:|---|---|
| C_raw 總價 CatBoost | 1 | 三方共用；2999 棵、原 87 欄 | `round90_offset_trees.fit_trees` 已實作 |
| L_unit 單價 LightGBM | 1 | 三方共用；3999 棵、fit-only category maps | 同上；CPU fit |
| C_log 近期 log CatBoost | 1 | 三方共用；R46、2999 棵、95 欄 | 同上；新 fit 自算原 float64 日期權重 |
| C_unit42 近期單價 CatBoost | 1 | 三方共用；R79、2999 棵、95 欄 | 同上；新 float32 日期權重 × A/100 |
| N_area 面積 NN | 1 | 三方共用；R54 ramps＋R48 trainer、seed 42／30 epochs | `round90_offset_nn.prepare_nn/train_nn` 已實作 |
| N_prior 原 MLP ratio NN | 1 | v22／v24 共用；R63 trainer、seed 42／30 epochs | 同上；新 grouped OOF prior 與日期權重 |
| C_unit137 近期單價 CatBoost | 1 | 完整 v24 額外需要 | [round83_seed_unit.fit_model](../src/round83_seed_unit.py) 可直接接受新窗口；R90 四樹 helper **不含**這一支，仍缺新窗口保存／全 query 重載整合 |
| N_attention prior-ratio NN | 1 | 固定 R94／R95 候選額外需要 | [v2 trainer](../src/round94_attention_prior_train_v2.py) 可接新九成員 prior NPZ；R90 `train_nn` **只啟動兩個原 NN**，尚缺新窗口呼叫與收據整合 |

依現有配方，GPU 工作為四次 CatBoost＋三次 NN，LightGBM 為一次 CPU fit；GPU 仍由主線串行協調。此處是次數盤點，不是新切點成本實測或工期承諾。

## 完整組合不能混淆

令 `RS(B)` 表示使用同一份新窗口 reference／spatial state，依該分支的 B 重新產生 reference prediction、confidence／diagnostics，再做 spatial。所有分量先還原為 TWD：

```text
H       = RS(.75*(.5*C_raw + .5*C_log) + .25*L_unit)
N       = RS(.75*(.5*N_area + .5*N_prior) + .25*L_unit)
U22     = RS(.75*C_unit42 + .25*L_unit)
U24     = RS(.5*C_unit42 + .5*C_unit137)
N_att   = RS(.75*(.5*N_area + .5*N_attention) + .25*L_unit)
v22     = median(N, H, U22)
v24     = median(N, H, U24)
candidate = median(N_att, H, U22)
```

**v24 的 U24 不含 .25 LightGBM**，不是把兩棵樹均值塞回 U22。這由 [R83 driver](../src/round83_experiment.py) 的 `base=.5*seed42+.5*native` 核對。候選仍只替換 v22 的 prior NN，不順便套 v24 seed 混合。共需五種不同 base 的 R/S 推論，可共享 H、N、U22 的結果；不能平均已修正分支，也不能共用另一 base 的 confidence。R/S state 新建一次，不需為每個 base 另 fit。

## 尚須用新 fit 重建的非 native 狀態

| 狀態 | 新窗口要求與現有入口 |
|---|---|
| 原始特徵／兩種地理 | **已實際完成純 X 準備**：272,117 fit、107,169 query，見[準備報告](round90_preparation_report.md)。`OffsetData.tree_*` 是原 R6 八欄，供 geographic trees／NN；`spatial_*` 是 R39 補位八欄，僅供 spatial。重用保存純 X 與 canonical 映射，不重用舊價格 state |
| LGB vocabulary | 四樹 helper 在新 Xfit 建 category maps；query 不擴充 vocabulary |
| R47 encoder | NN prepare 在新 fit 建 median／尺度／vocabulary，供 area、原 prior MLP、attention 共用；不能借舊 confirmation encoder |
| R54 ramps | 新 fit 144 數值欄建立 quantile knots；只供 N_area，寬度自然決定，不硬寫舊 599／603。原 prior／attention 維持 144 數值欄 |
| R60 priors | 新 fit 五組 held-out complement prior＋一份 full-fit prior；這是 **分组 OOF、非時間 OOF**，不需另外 fit 五個 NN。query 只用新 full-fit prior。NN prepare 已實作，但未在此真窗口執行 |
| 日期權重／factors | `prepare_nn` 新算 fit-global date64 再 cast32，無效日期＝1；供 C_unit42、C_unit137、原 prior MLP、attention。area NN 仍 uniform。兩 ratio NN 共用同一新 prior NPZ／float32 factors，不另估先驗；log 按 R46 自己的 float64 權重配方 |
| Reference | [round2_neighborhood.fit_final_reference(Xfit,yfit)](../src/round2_neighborhood.py)：新 mask、fill、beta、殘差、群索引。R90 三個 helper 都未編排此步 |
| Spatial | [round8_spatial_reference.build(reference,Xfit,spatial_fit)](../src/round8_spatial_reference.py)：從上列新 reference residual／beta 與 R39 fit geo 建一次 state。三方共用，不能沿用舊 8 月窗口或最終全量 state |

Reference、spatial、encoder、ramp 與 prior 的建立不計入上面的 8 次 native fit，但必須保存身分、狀態與重載證據。沒有理由在這個比較額外訓 R89 分群趨勢、R52／R60 舊 NN 或多 seed NN。

## 目前完成程度與最少整合缺口

目前 `outputs/round90/` 僅有 `preparation/` 與 `preparation_launch/`。純 X manifest 已存在；沒有新切點模型、NN prepared、價格 state 或分數。[較早 preflight](round90_preflight_review.md)「尚未真 preparation」是當時紀錄；後來完成的是純 X，並非 NN 有標籤 preparation。

1. 另訂新切點執行／評估契約，確認是否完整比較三方，凍結來源、8 次 fit 清單與唯一輸出目錄。使用 `load_prepared(path, expected_manifest_sha)`，不重跑或覆寫既有 pure-X preparation。
2. 新 driver 核 raw-train 來源後，以 **fit_source_cache_indices** 抽對齊 fit y；`prepare_nn(..., source_indices=OffsetData.fit_indices)` 必須傳 **canonical 位置**。兩者不可互換，也不可用 reset 後的 0..n−1。query y 只在所有候選向量封存後交評估端；trainer 的 NPZ 僅含 fit labels。
3. 呼叫一次 `prepare_nn` 得到兩 NPZ、保存 date32、encoder／ramp／prior／folds。呼叫四樹 helper 與原兩 NN helper；新包裝另保存 seed137 CBM／wrapper／原生預測、檢查全 query 重載。attention 呼叫同一新 `prior_npz`，沿 v2 設定與完整 ratio/TWD reload；父程序 log／execution 放模型目錄之外，v2 trainer 的 out 必須尚不存在。
4. 新建一份 reference／spatial，保存重載後核五個 base 的 R/S／diagnostics／fallback，再保存三個完整未取整 median。新切點沒有可直接重播的舊完整預測：不能切舊 R71／R79／R83 向量當控制，也不能只補 attention 而借後期六模型。
5. 尚缺此新窗口的 scorer／generic／paired adapter。原 `candidate_diagnostics_v3.py` 與 `paired_candidate_bootstrap.py` 綁三個 8 月窗口及舊 R48 索引，不可僅改 round 參數冒用。新 adapter 應明確使用新 fi/qi／起日、固定 group weights、project/month 規則與三個完整控制，保留新案、價帶等正反證據；本次未實作。

R90 NN helper 已有 11 項小型合成、資料 helper 已有純 X 真實重映射核驗；四樹 helper 是只讀 source 審查。**新窗口的真正 NN prepare、8 次模型 fit、保存模型全 query 重載與端到端組合尚未執行。** JSON reader 須依實際 artifact 契約核驗：helper 自存 JSON 不保證各有 `.sha256` sidecar，應以父 manifest／artifact map 綁定，不能重犯 R94 對無旁檔 encoder JSON 使用 `locked` 的錯誤。

此窗口與舊年度資料重疊，配方也受先前研究影響，只能稱「錯開起點的時間穩定性檢查」，不是未接觸資料的盲測。後續是否執行仍取決於主線完整 R95 結果與另行固定的方案；v22 仍是唯一選定提交，Private 未知。

## 本次核對身分

| 檔案 | SHA-256 |
|---|---|
| `src/round90_offset_data.py` | `4f58575bff75b69fd900af84f9152dc3e900f34ad6909e0f2e0366bb67c928f1` |
| `src/round90_offset_trees.py` | `666e9fc61d22565aae2f738767bfef471e12679d4e1c0bbe86cf366afa540110` |
| `src/round90_offset_nn.py` | `b473b246925dffe3d084c909679892f45e33f19ed0a6c18c100cfc836bc35e06` |
| `src/round83_seed_unit.py` | `c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b` |
| `src/round83_experiment.py` | `8672b65b4e7258348a9124d0b20cdfeceadd404aa2a1a6f9bb0dee0a3ad44f5c` |
| `src/round94_attention_prior_train_v2.py` | `ae197952be777f3e9e2fdb66458887d2a3885fd736b75dd3677eeae43cfeef83` |
| `reports/round94_plan.md` | `c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a` |
| `reports/round95_plan.md` | `311c521cc9ae0251b68957cbe232fdc920d23e41d2683c8750cabfba2948a2fb` |
| `outputs/round90/preparation/manifest.json` | `400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7` |

補充來源：[最初錯開窗口盤點](r89_offset_validation_inventory.md)、[NN helper 合約](round90_nn_helper_review.md)。本次只新增本文件；未改 HANDOFF、GitHub、任何舊碼或 frozen plan，主線負責整批交接更新。
