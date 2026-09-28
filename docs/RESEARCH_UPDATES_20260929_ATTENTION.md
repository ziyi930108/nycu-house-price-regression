# Attention 與參考修正研究：三期完成，保留 v22

2026-09-29（Asia/Taipei）。**Attention 回歸與局部信任折半對照均已完成三期、完整診斷及兩組配對分析，但都沒有通過事前固定的採用門檻。** 不新增 test 預測或 CSV。使用者唯一提交仍為 v22，既有 Public MAE 為 1,998,568.5706 元，Private 仍未知；正式報告、PDF 與 22 頁簡報保持。

這份是完成研究後的本機文件快照。發布是否完成，請以 Git 歷史及另存的發布收據為準，不能只因本頁存在就推定已 push。[上一批 R93](RESEARCH_UPDATES_20260929_R93.md)與[更早研究](RESEARCH_UPDATES_20260929.md)保留歷史時序。

## 結果先看這張表

差額＝候選 MAE − v22 MAE，單位新臺幣元；負數較好。三個窗口已反覆使用，不是新盲測。

|期別|原 v22 MAE|Attention R95 MAE|R95 差額|信任折半 R96 MAE|R96 差額|
|---|---:|---:|---:|---:|---:|
|中間年度|1,407,387.07|1,422,360.94|+14,973.87|1,410,417.92|+3,030.85|
|最新年度|1,383,228.41|1,412,063.62|+28,835.21|1,388,191.71|+4,963.30|
|較早年度|2,309,352.36|2,277,199.48|-32,152.87|2,306,256.38|-3,095.98|

事前門檻要求三期整體與固定加權 MAE 都改善，且最新年度至少改善 0.5%。R95 最新年度反而退步約 2.08%；R96 約退步 0.36%。R96 固定加權差額三期皆為正：+3,058.97／+4,233.52／+107.99 元，所以較早期整體小幅改善仍不足以採用。

對 v22 的 12 個配對區間，R95 為 4 負／0 跨零／8 正，4 個負區間都在較早年度；R96 為 0 負／11 跨零／1 正，不能說信任折半在所有群都顯著更差。對 v24 兩候選皆為 0 負／4 跨零／8 正。這些區間互相依賴、未校正多輪選型，也不是 Private 勝率。[R95 完整報告](../evidence/reports/round95_report.md)與[R96 完整報告](../evidence/reports/round96_report.md)保留 48 個區間及價帶、建案、季度等診斷。

## Attention 為什麼試、如何做

原先驗 NN 使用欄位編碼後的多層感知器。這次把同一筆房屋的數值與類別欄位視為 token，讓欄位依地段、面積、屋況與建案資訊調整交互作用。它仍是監督式總價回歸；attention 發生在同一筆交易的欄位之間，不是跨成交資料找答案。

固定輸入為 144 數值＋23 類別＋CLS，共 168 tokens；維度 64、4 heads、2 blocks。最後一個 block 只計算 CLS 所需的 query、residual 與 FFN。Softplus 輸出正倍率，乘同窗價格 prior 及面積因子還原總價。固定 30 epochs、seed 42、AdamW，微批 256 累積成 2048 列一個 optimizer step；沒有用 query 挑 epoch 或搜索架構。

只替換 prior NN，保留同窗 area NN 與 LightGBM，仍以原比例組成 NN 分支，再套原 reference／spatial，與原 log／unit 分支取中位數。[固定科學計畫](../evidence/reports/round94_plan.md)及[整合契約](../evidence/reports/round94_integration_inventory.md)記錄 fit-only encoder、建案分組五折 OOF prior、日期權重與完整控制重播。prior 依賴 fit 價格；不能稱所有準備都純 X。query 標籤不進 trainer，driver 只在評分時使用。

有利反證也保留：R95 的既有建案三期都改善，較早年度五個價帶全改善，收益並非只出現在豪宅。但中間／最新年度新建案約退步 24,063／55,497 元；較早年度改善不足以支持替換。embedding 寬度、dropout 位置和架構同時改動，不能把結果全歸因 attention；舊對照沒有重新訓練，GPU 固定 seed 也不保證決定性。

## 信任折半：另一個零訓練機制對照

R96 檢驗局部參考成交是否修正過強。保留原三分支模型及整體時間趨勢，令 B_future 為原趨勢基準、P 為原參考預測、c 為原 confidence，固定：

~~~text
P_new = B_future + 0.5 × (P − B_future)
c_new = 0.5 × c
~~~

c=0 的列直接保留原 P；新的價格與 confidence 一起傳入原 spatial，再取三分支中位數。這不是把房價折半，也沒有按 test ID、未知真價或窗口改強度。降低局部信任會增加 spatial 的剩餘權重，不能直接稱更保守。[R96 計畫](../evidence/reports/round96_plan.md)與[方向及既有反證](../evidence/reports/regression_direction_after_attention_audit.md)在評分前固定規則。

原 base 全部不變；三期三分支的 reference、spatial 單獨 MAE 都退步，但較早期最後 median 的整體 MAE 仍小幅改善。這說明層差不能相加作因果解釋。既有建案三期皆退；新建案中間／最新略退、較早期改善。完整分群保留，沒有據此再挑強度。

## R94 工程失敗與 R95 恢復是一個模型假設

R94 初次 main 在真 fit 前因歷史 R48 encoder JSON 沒有個別 sidecar 停止，7.813 秒、0 fit、0 新分數。該檔原本由 R48 artifact SHA 管理；前置審查漏抓 reader 額外要求。原 source、失敗與審查皆保留，沒有刪除檢查。R95 在新目錄改用原 authority SHA 的 reader，其餘模型、trainer v2 與科學計畫相同，不能算成另一項調參實驗。

trainer v1 與 v2 也完整保留。v2 只把 NPZ 九鍵驗證移到陣列讀取之前並綁定新 source 路徑，不改 CONFIG。47 項 CPU 檢查通過；同權重上游 broadcast 參考路徑與末層 CLS 裁切版，在關 dropout 的 float64 測試下輸出及全參數梯度一致至數值容差。相同 batch 的保存重載要求逐位一致，不能把它擴張成跨硬體或所有 batch 的位元保證。

[R95 獨立完成審查](../evidence/reports/round95_independent_review.md)與[R96 獨立完成審查](../evidence/reports/round96_independent_review.md)另存來源、控制、結果與區間核對；審查本身沒有重新 fit。原[R94 preflight](../evidence/reports/r94_validation_audit.md)、[R95 恢復審查](../evidence/reports/r95_validation_audit.md)、[R96 preflight](../evidence/reports/r96_validation_audit.md)保留各自當時範圍，不回寫歷史判斷。

## 實測工作量與總研究表

|項目|實測或估算|
|---|---:|
|Attention v2 合成 GPU 每 2048 列均值|0.182333 秒|
|合成推估三期成本，非實際 fit|1,870.740 秒|
|R95 三次真 NN fit 合計|1,762.376 秒|
|R95 成功 main 內計時，含 fit|1,903.922 秒|
|R95 main 外程序計時|1,904.781 秒|
|R96 main 內計時，0 fit|110.641 秒|
|R96 main 外程序計時|111.547 秒|

同一 main 的內外計時，以及其中的 fit 時間，不能重複加總。合成、開發、審查、診斷與發布不在 main 總和；沒有實測整段聊天時間。

[九項假設總表](../evidence/reports/regression_after_r95_20260929.md)涵蓋 R85 至 R95 的九個候選，合計 15 次樹模型、3 次 NN、3 次 CPU 趨勢 wrapper fit；真 fit 共 3,932.126 秒，已包含於成功 main 共 4,798.733 秒。R90 是資料準備、R94 是同配置工程失敗，均不另計模型假設；R96 是其後第十個機制對照，0 fit，另記上表時間。

![九項固定假設在三個年度的 MAE 差額](../evidence/outputs/regression_after_r95_20260929/mae_delta_comparison.png)

圖表產生的首次摘要 reader schema 錯誤也[原樣保留](../evidence/outputs/regression_summary_20260929/attempt1.json)，只修彙整程式，沒有重跑模型或更改分數。

## 程式、證據與授權

- Attention：[trainer v2](../research/src/round94_attention_prior_train_v2.py)、[原 trainer](../research/src/round94_attention_prior_train.py)、[R95 driver](../research/src/round95_experiment.py)、[review](../research/src/round95_review.py)。
- 信任折半：[helper](../research/src/round96_reference_shrink.py)、[driver](../research/src/round96_experiment.py)、[review](../research/src/round96_review.py)、[39 項合成收據](../evidence/reports/round96_helper_selfcheck.json)；全部合成 attempts 保留。
- R95：[完整結果](../evidence/outputs/round95/results.json)、[診斷](../evidence/reports/round95_diagnostics.md)、[採用審查](../evidence/outputs/round95/decision_review.json)、[五步執行收據](../evidence/outputs/round95_launcher_20260929/pipeline_execution.json)。
- R96：[完整結果](../evidence/outputs/round96/results.json)、[診斷](../evidence/reports/round96_diagnostics.md)、[採用審查](../evidence/outputs/round96/decision_review.json)、[五步執行收據](../evidence/outputs/round96_launcher_20260929/pipeline_execution.json)。
- 失敗及修正：[R94 execution](../evidence/outputs/round94/execution.json)、[R94 原 driver](../research/src/round94_experiment.py)、[R95 工程 diff](../evidence/outputs/round95_launcher_20260929/driver_engineering_diff.patch)、[trainer v2 修訂](../evidence/reports/round94_trainer_revision.md)。
- 上游：[歸屬與改編說明](../research/upstream/README.md)、[完整 Apache-2.0 授權](third_party/rtdl_LICENSE.txt)、[固定來源 manifest](../evidence/outputs/round94_upstream/manifest.json)。本題改編不等於完整論文復刻。
- 快照：[原準備批次 58 檔](../evidence/outputs/research_snapshot_attention_20260929_preparation/copy_manifest.json)、[完成批次複製清單](../evidence/outputs/research_snapshot_attention_20260929_final/copy_manifest.json)；原準備收據不覆写，379 支歷史 source manifest 不改。

R90 仍只有 pure-X 272,117 fit／107,169 query 與 helpers，沒有新切點模型、prior／encoder／reference／spatial fit 或 MAE。[新窗口依賴盤點](../evidence/reports/round90_validation_inventory_after_attention.md)說明完整 v22／v24／Attention 比較需要 8 個原生模型及全部 fit 狀態重建；不能沿用看過該窗口的後期模型。

原報告的工作區路徑保留不改，程式連結映射與歷史依賴見[證據閱讀說明](../evidence/README.md)。

這是原碼、聚合報告、雜湊和授權快照；不含原交易表、模型、價格先驗、NPY／NPZ／CSV／pickle 或逐筆價格向量，也不是全新環境端到端重訓備份。全部本輪訓練與評分已結束，後續研究需另留計畫與執行證據。
