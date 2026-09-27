# R83／v24 的 GitHub 重現性盤點

日期：2026-09-27。這是有限範圍的文件與程式碼稽核，不是交付包、不指定最終版本，也未執行訓練、模型推論、資料重建或發布。

**結論：目前已保存的程式、模型與證據支援本機既定產物鏈的追溯；本次未見已驗證「只提供原始 Excel，在乾淨環境用單一入口重建 v24」的流程。** 保存程式與成功產生 CSV，不等於端到端重現已通過。v24 是互補候選；v22 的有限本機優先、v20 備選及最終版本未選定的角色，不由本報告更動。

## 已有能力與實際入口

[主 README](../README.md) 的原始資料位置是 `台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression/`，包含 train-v2.xlsx、valid-v2.xlsx、test-reindex-test-v2.2.xlsx 與 sample-submission CSV。程式保留清理、特徵、早期模型訓練與推論入口，但 README 的 `models/final_config.json → train_final.py → predict.py → outputs/submission.csv` 是早期路線，不能直接當作 v24 配方。[R6 README](../README_round6.md) 與 [R8 README](../README_round8.md) 也屬當時版本；其中「報告重生」只讀既有 JSON，不是重建訓練與預測。

| 目前入口 | 實際作用與前提 |
|---|---|
| [prepare_data.py](../src/prepare_data.py)、[prepare_features.py](../src/prepare_features.py)、[house_features.py](../src/house_features.py) | 讀原始工作表、形成 data_cache 與原始 87 欄特徵；這是起始部分，不會自行完成後續 v24 所有分支。 |
| [round83_final_unit.py](../src/round83_final_unit.py) `--run --plan-sha <固定 SHA>` | 以原 467,221 列重新訓練一支 seed137 單價模型，保存後重載核對 47,073 列。它先要求已存在的 R79、R40、R63 權威資料、快取、地理特徵與日期權重，不從 Excel 開始。 |
| [round83_final.py](../src/round83_final.py) `--run --plan-sha <固定 SHA>` | 讀新 seed137 與保存的 seed42 原生向量，固定各半後套 reference／spatial，再與保存的 NN／log 分支取 median。產出 float64 向量及匯出 payload；本程式不重建那些舊模型。 |
| [CSV writer](../outputs/round83/csv_authoring/export.mjs) | 用 ArtifactTool 建立兩欄儲存格、全格讀回、預覽與 CSV 全量再匯入；需要上述 payload、特定套件環境及既有技能 marker 流程。 |
| [round83_release.py](../src/round83_release.py) `--run` | 獨立比較全部 ID、價格、BOM、標頭及一次取整結果，再發布 v24。只驗證成品及其證據，不從模型重新預測。 |

這些是**既有入口說明，不是現在應執行的重跑指令**：程式固定了當時的 plan／decision／selector SHA、旧 CSV 清單，且要求輸出目錄不存在。現在直接重送會因拒絕覆寫或當前 selector 已更新而停止；這是保存證據的設計。

## 仍依賴的歷史產物與環境

- 原始快取是 `data_cache/{train,valid,test}_features.pkl` 及 train／valid 標籤快取；原 R6 地理與 canonical 索引在 `outputs/round6/geography/final/`。程式按 train[KT]＋valid[KV] 再取 KF 對齊，不能只串接原始列或重複套 KF。
- seed137 訓練直接讀 `outputs/round63/final/prepared.npz` 的保存日期權重及其 preparation／training／indices 身分，不重新估計權重；原模型與 legacy state 承接 `outputs/round79/final_model/` 及 R40。
- 組裝直接使用 `outputs/round79/final_model/model/predictions_twd.npy`、`outputs/round71/final/nn_spatial.npy`、`outputs/round46/trial/recency_log_spatial.npy`，以及新 R83 native。另讀已保存 reference、`outputs/round39/trial/artifacts/spatial_state.pkl` 和 query geography。SHA 可以辨識這些輸入，**不會替缺失檔案自動建立它們**。
- R6 文件明列歷史 OSM／Geofabrik 快照；只有四份課程原始檔不足以重新建立這批地理輸入。將來需列清必要快照、manifest、取得／生成方法，而不是把主 README 當時的「四份本地資料」描述套用到所有後期版本。
- [requirements.txt](../requirements.txt) 固定 NumPy 2.3.5、pandas 3.0.1、CatBoost 1.2.10、LightGBM 4.7.0 等；[requirements-round2.txt](../requirements-round2.txt) 追加 SciPy／joblib／threadpoolctl。R83 實際 metadata 的 CatBoost／NumPy／pandas 與上述版本一致，但這兩份檔案未列 NN 程式使用的 PyTorch，也未涵蓋 Node 的 `@oai/artifact-tool`。重建完整候選還需整理 Torch／CUDA／GPU、Node／ArtifactTool 來源與版本；不能只宣稱一次 pip 安裝已覆盖全部環境。
- 程式有跨輪 imports（R83 → R79／R80 → 更早模組）、pickle state 與固定檔案布局。目前不是獨立的 v24 最小套件；抽出幾個最新 .py 即交付會缺依賴。GPU 固定種子也不承諾從頭重訓得到逐位相同的 CSV。

本次僅讀小證據：[final model results](../outputs/round83/final_model/results.json)／metadata 記錄完整重載 47,073 列；[release](../outputs/round83/final_export/release.json) 的狀態為 experimental_v24_trial_generated，列數 47,073，完整 ID／正整數／未取整向量比較與 BOM flags 為 true，[execution](../outputs/round83/final_export/execution.json) 為 exit 0。這支持**已完成這次指定產物的訓練、組裝與匯出檢查**；本稽核沒有重新讀陣列或模型，更沒有在乾淨環境重演整條流程。

## 選定版本後還要完成的整理

1. 以最後選定方法建立新交付目錄與單一明確入口，區分「從原始資料重建」及「從保存模型推論」，另把只依保存向量重新組裝標為第三種模式。保留全部原始程式與執行紀錄，不修改舊證據來迎合新目錄。
2. 從已選方法抽出必要依賴順序：清理／去重及索引 → 地理快照與特徵 → fit-only encoder／OOF prior／日期權重 → 各模型及 reference／spatial state → 固定融合 → CSV 驗證。交付流程不應要求使用者重新搜尋或評估所有 83 輪候選。
3. 產出模型／state／資料與 artifact manifest，列路徑、大小、SHA、用途及取得方式；明確哪些隨包提供、哪些另行保存。[現行 .gitignore](../.gitignore) 主要排除早期 .venv、data_cache、models 等，不能視為已完成後期 outputs 大模型與其他環境的 GitHub 收錄規劃，也不應直接把整個工作區全部加入版本庫。
4. 固定完整軟體與硬體環境，移除交付入口對使用者機器專用 runtime 路徑、node_modules junction 和可变 selector 狀態的隱性依賴；保留原 archival 程式的固定身分核對，不就地放寬它們。
5. 在另建的乾淨工作目錄實際走完選定模式，保存命令、輸入身分、真實耗時／exit、各步輸出與完整 CSV 核對。對保存模型重播與 GPU 從頭重訓分別定義可驗證的重現範圍；只有完成實測後才能寫「端到端已驗證」。
6. 最後再更新主 README／方法報告與精簡實驗沿革，區分離線、Public 與未知 Private，遵循 [DELIVERY_REQUIREMENTS.md](../DELIVERY_REQUIREMENTS.md)。此次未打包、未建立遠端、未推送 GitHub，也未更動推薦、selector、模型或 CSV。

範圍：只盤點根目錄 README／requirements／交付要求、代表性的 R6／R8 重現說明、prepare_data／prepare_features／house_features、R47／R48 trainer 的 imports、R83 final-unit／composer／writer／release 及必要 R80 承接來源與上述小 JSON。未遍歷全部歷史依賴、未清點所有模型大小，亦未測試 raw-to-v24 的完整可達性；「未見完整入口」是這次有限查核的結論，並非證明所有可能入口都不存在。
