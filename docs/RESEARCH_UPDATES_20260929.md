# 回歸改良研究：2026-09-29

本批完成三項固定假設、六次 CatBoost 訓練、三次 CPU 趨勢 wrapper fit，以及九組候選×年度完整評估。**三個候選都未達事前採用門檻，沒有替換 v22、全量訓練或新增 Kaggle CSV。** Public 仍為使用者回報的 1,998,568.5706 元，Private 未知；v22 是使用者唯一選定版本。

本批三項候選訓練及評估已結束；這不代表整個專案不再研究，後續新假設另批保存。這是本機文件快照。來源與複製核對見[複製清單](../evidence/outputs/research_snapshot_20260929/copy_manifest.json)；新 Git 提交、遠端發布、SOURCE_MANIFEST 更新及 ZIP 由後續發布流程另行處理，本頁不宣稱已完成。

## 改了什麼，為什麼這樣做

1. **分群時間趨勢（R89）**：不同建物型態／用途的漲幅可能不同。先只看 fit 資料盤點案內日期跨度和獨立案數，再在固定支持足夠的群估計時間斜率，以每案等權與 Huber 估計，再讓群斜率最多占 50%、至少一半退回全域斜率，以限制變動。不支持的群保留原 global 趨勢；reference 和 spatial 使用相同新斜率。原六個 native 模型不變。
2. **先驗上的加法總價殘差（R91）**：以同窗建案分組 OOF 單價乘面積形成基準價，CatBoost 學實際總價與基準的差額，希望直接處理金額誤差。target=(y−A×P_OOF)/1e6 可正可負，只用日期權重；query 先加回 full-fit prior offset，再投影成非負總價。95 欄／23 類別不新增 prior 特徵，其他模型參數保持。
3. **較淺單價樹（R92）**：只把近期加權單價 CatBoost 的 depth 8 改為 6，檢查較少的高階交互能否減輕稀疏建案的過擬合。仍為 2,999 棵樹、lr .065、seed 42、95 欄／23 類別；原單價 target、面積×日期權重及還原公式不變。R15 曾測 raw 總價的 depth 6，不能當作這個單價配方已經測過。

R91、R92 都只替換原 unit 槽的 CatBoost：75% 新 native 總價＋25% 原 LightGBM，重算固定 reference/spatial，再與保存 NN、log 分支取中位數。三個假設分開驗，不把其中有利部分事後拼在一起。

## 三期完整組合結果

以下是本機歷史回測 MAE，單位新臺幣元，越低越好。括號為候選−v22；負值改善。三欄是不同交易年度，不能把它們當成逐次提交的進步曲線。

| 方法 | 2021-08-11～2022-08-10 | 2022-08-11～2023-08-10 | 2023-08-11～2024-08-10 |
| --- | ---: | ---: | ---: |
| v22 對照 | 2,309,352.36 | 1,407,387.07 | 1,383,228.41 |
| R89 分群時間趨勢 | 2,320,127.94（+10,775.59） | 1,400,980.40（-6,406.67） | 1,393,135.44（+9,907.03） |
| R91 先驗加法殘差 | 2,347,209.50（+37,857.15） | 1,454,424.53（+47,037.46） | 1,378,833.71（-4,394.69） |
| R92 較淺單價樹 | 2,302,598.41（-6,753.95） | 1,409,199.87（+1,812.80） | 1,404,783.19（+21,554.79） |

三個候選均未符合「三期整體及固定加權 MAE 全改善、最新整體至少改善 0.5%」的固定門檻。沒有看結果後放寬門檻；v24 原有三期較低的整體及加權點估計也完整保留，不把 v22 描述成所有離線指標的最佳者。[機器摘要](../evidence/outputs/regression_research_20260929/summary.json)保存精確數值。

## 不採用的理由與保留反證

- **R89**：中間年度改善 6,407 元，但最新與最舊退步；新案及億元群三期都退步。中間已見建案及最新低於千萬元的群有改善，並非所有群都無效。支持3／6／2群來自預先固定門檻，沒有因支持少再降低要求。
- **R91**：最新改善 4,395 元、約 0.318%，低於 0.5%，其餘兩期退步；最新四個對 v22 配對區間都跨零。2,000–5,000 萬與 5,000 萬–1 億兩帶三期改善，最新新案亦改善；相反，中間／最新億元群的 MAE 增加約 1,370 萬／443 萬元，須完整保留。
- **R92**：最舊改善 6,754 元，中間及最新退步，最新四個對 v22 配對區間皆偏退；中間與最舊八個區間均跨零。中間／最舊億元群改善是有利證據，但三期 2,000–5,000 萬群均退步。更淺的樹並不保證泛化更好。

真價帶只是事後診斷，未知 test 真價不能用來決定逐筆選哪個模型。每個候選的兩對照、三年度、建案／月份、整體／加權共24個配對區間都保存，沒有只保留有利區間。

## R90 做到哪裡

R90 是**錯開時間切點的資料與訓練 helper 準備**，不是第四個新模型配方，也沒有新窗口分數。

固定 fit 日期早於 2023-02-11，共 **272,117 筆**；query 為 2023-02-11～2024-02-10，共 **107,169 筆**。純 X preparation 已真正執行：保留 canonical 索引，將 2023 年原 R6／R39 非價格地理資料按列身分重映射，保存後核对全部四組索引與六份 frame。該階段未讀價格表、test 或模型。[準備報告](../evidence/reports/round90_preparation_report.md)及[manifest](../evidence/outputs/round90/preparation/manifest.json)記錄來源與 SHA。

data、trees、NN 三個 helper 已保留；data／NN 有各自合成測試，tree 有只讀審查。**真實新窗口的 encoder、OOF prior、reference/spatial 及六個 native 模型尚未 fit，沒有新 MAE。** 歷史日期已更改，不能直接載入晚期模型來冒充這個窗口的驗證；完整重建是否啟動需後續另作決定。錯開切分仍與已研究資料重疊，不能稱為盲測。

## 做了哪些核驗

各期 fit/query 索引、日期、特徵、地理、fit 標籤、日期權重與保存 state 均綁定原 SHA。原 v22 base→reference→spatial→median 完整向量與控制組分數重播一致。R89 三分支 wrapper 保存重載一致；R91/R92 六個新 CatBoost 保存後，全 query raw 與還原 TWD 逐位一致。

R91 每期完整重播 R60 OOF 值、fold、full-fit state、query prior 及 R63 factor，價格來源只進 fit。這是 group OOF，不是時間 OOF；fit 內較後交易可進別折 reference，OOF 與 full-fit prior 也有分布落差。query 標籤只用評分／事後診斷，不進新模型訓練介面。

23／41／31項 helper 合成檢查與真模型訓練分開計數；FakeModel 通過不能代替 CBM 真重載。失敗的工程測例與修正保留，不把它們說成真 fit 失敗或刪除反證。各候選 main、generic、兩 paired 均 exit 0。

## 實測耗時與保存界線

| 實驗 | 新 native fit | 新 CPU wrapper fit | fit 秒數 | 主流程秒數 |
| --- | ---: | ---: | ---: | ---: |
| R89 | 0 | 3 | 21.438 | 226.437 |
| R91 | 3 | 0 | 465.000 | 592.328 |
| R92 | 3 | 0 | 404.376 | 478.015 |

六次 native fit 合計869.376秒。fit 已包含在主流程，不能重複相加；診斷／配對另計，並行程序也不能直接相加為牆鐘時間。R90 純 X prepare 9.500秒、外層10.234秒，另有4.266秒重載核驗；內外層同樣不能重複計算。這些不是整段聊天或程式開發耗時。

本快照只收原 bytes 程式、報告、彙總 JSON 及 SHA，排除課程表、模型、pickle state、NPY／NPZ、逐筆價格與 CSV。379支歷史 source_manifest 不改，新增來源另有複製清單，發布前再刷新外層 SOURCE_MANIFEST。原 v22 推論、正式 REPORT/PDF、22頁簡報與提交選定保持。

## 限制與接續

這三個假設受既有誤差啟發，年度窗口已反覆使用，仍有選型偏差。建案／月份 bootstrap 沒有校正多輪搜尋，不能解讀為 Private 勝率。GPU 同 seed 不保證位元決定性，原控制並未同期重訓，微小差異不能全作單項改動的因果效果。

保存程式、模型與重載證據不等於全新環境從原始資料重訓整條流程。下一假設須另固定可部署方法與 fit-only 規則；本頁不預告後續方法已實作或有效。沒有到內政部系統對答案，沒有新增 Public 或代操作 Kaggle。

## 詳細入口

- R89：[計畫](../evidence/reports/round89_plan.md)、[報告](../evidence/reports/round89_report.md)、[獨立審查](../evidence/reports/round89_independent_review.md)、[driver](../research/src/round89_experiment.py)、[決策](../evidence/outputs/round89/decision_review.json)。
- R91：[計畫](../evidence/reports/round91_plan.md)、[報告](../evidence/reports/round91_report.md)、[獨立審查](../evidence/reports/round91_independent_review.md)、[driver](../research/src/round91_experiment.py)、[決策](../evidence/outputs/round91/decision_review.json)。
- R92：[計畫](../evidence/reports/round92_plan.md)、[報告](../evidence/reports/round92_report.md)、[完成後獨立核驗](../evidence/reports/round92_independent_review.md)、[訓練前獨立審查](../evidence/reports/r92_validation_audit.md)、[driver](../research/src/round92_experiment.py)、[決策](../evidence/outputs/round92/decision_review.json)。
- R90：[data helper](../research/src/round90_offset_data.py)、[tree helper](../research/src/round90_offset_trees.py)、[NN helper](../research/src/round90_offset_nn.py)、[NN 合約](../evidence/reports/round90_nn_helper_review.md)。
- [本批工作區報告](../evidence/reports/regression_research_20260929.md)、[前批 R85–R88](RESEARCH_UPDATES_20260928.md)。

evidence 內歷史 source/report 均保留原 bytes，其相對路徑可能依原 C:/NYCU_ML 布局；其中 ../src/ 對應本專案 research/src/，未納入的資料／模型路徑仍需原工作區。請由本頁可用連結導覽，勿為修連結改寫凍結原始證據。
