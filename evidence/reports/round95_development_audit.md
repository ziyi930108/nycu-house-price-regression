# R95 development：模型與四層結果只讀審查

2026-09-29（Asia/Taipei）。本文件只審已完成的 development：fit 240,316 列、query 81,234 列，query 期間 2022-08-11～2023-08-10。接獲任務時 confirmation 正由主線在單張 GPU 執行。**這不是完整三期結論，不據第一期選參數、改設定或取消後兩期。**

結論：**此次檢查未發現非預期工程錯誤；原生 NN、融合後 NN 分支及完整 median 在此期均比各自原對照差。** 工程完成不等於模型改善；新案退步與既有案小幅改善須同時保留。完整決策仍等三期與固定診斷完成。

## 審查範圍與身分

只讀 [stage results](../outputs/round95/development/results.json)、[training](../outputs/round95/development/model/training.json)、[run](../outputs/round95/development/model/run.json)、[子程序 execution](../outputs/round95/development/model/execution.json)、[父程序 execution](../outputs/round95/development/trainer_process/execution.json)、兩份來源／控制收據、原 R63／R79／R83 metadata 及凍結 source。另以 `torch.load(map_location='cpu', weights_only=True)` 讀保存 checkpoint，僅核 tensor 與身分；CUDA 未初始化，沒有建新模型、fit、predict、score、載入 query y、test 或預測陣列。

- stage 的 **24 個 artifact SHA**、trainer 的 **7 個 source SHA**、training 內 **4 個保存產物 SHA**全部吻合；這些集合重疊，不相加當獨立檔數。
- 啟動 request 的 **26 個凍結檔案**全部與當前 SHA 一致。
- R95 只是 R94 的無 sidecar JSON reader 工程恢復；[原科學配置](round94_plan.md)與 [R95 恢復範圍](round95_plan.md)保持，不是看過 R94 分數後另選架構。R94 沒有真 fit 或候選分數。

## 容量、固定訓練与資料邊界

保存 checkpoint 有 **61 個 tensor、1,246,657 個參數**，均為有限 float32、CPU。逐項核到 numeric weight／bias 為 144×64、23 組 category embeddings 總 cardinalities 17,860、類別 bias 23×64、2 個 blocks 各 Q/K/V/out 64×64、FFN 256×64 與 64×128、head 1×64；第一 block 無 attention norm，第二 block 有。容量符合 144 數值＋23 類別＋CLS＝168 tokens、d64／4 heads／2 blocks 的 source 配置。4 heads 和最後 CLS-only 路徑由固定 source 確認，不能只由權重矩陣形狀推知。

run 與 training 的共同欄位全等，CONFIG 與 trainer AST literal、configuration SHA 全等。固定 seed 42、30 epochs、AdamW lr .001／wd .0001、cosine eta_min 1e-5、microbatch 256／optimizer group 2048。30 筆 epoch 編號完整、兩種訓練 loss 有限，逐輪 learning rate 符合原 cosine 公式。**3540 次 optimizer updates＝30×ceil(240316/2048)**；每 epoch 118 組、最後 700 列，原 `train_group` 以當組實際列數作分母。沒有 early stopping／eval_set／梯度裁切／ratio 或總價 clipping，也沒有按這一期調 epoch。

第一／最後 epoch 的 training weighted MAE 為 2.3510036469／0.5204886198 百萬元，unweighted training MAE 為 3.3087019920／0.6984148026 百萬元。這些是訓練過程帶 dropout 的記錄，不能與完整模型 query MAE 當成同一測量，也不足以單獨證明退步原因就是過擬合。

[prepared verification](../outputs/round95/development/prepared_verification.json)記錄 encoder、完整 encoded fit/query、R60 OOF／fold／state、full-fit query prior、factor、date weight 的實际 CPU 重播相同。本審查核它與 trainer 的九個 array manifest 全等，並核原 R63 的 prepared SHA、兩個 factors、float32 date weight、行數、cardinalities 相同。NPZ 只有 fit labels；trainer 的 factor／weight provenance 標記 false 是因其交由 driver 核驗，不是漏驗：driver 收據為 true。OOF 按建案分組，非時間 OOF。

trainer 保存 model.pt 後以相同 batch 256 重載，**全部 81,234 query 的 raw ratio 與 TWD 逐位一致**；driver另核 factor 僅乘一次。這是主流程實際保存收據／source檢查，本審查没有再次推論。checkpoint identity、容量及有限值另做了上述 CPU 獨立檢查。

## 原 NN 分支四層：新值減原值

「native」只指 prior NN；「base」已與固定 R54 area NN／LightGBM 融合；「spatial」仍只是一支 NN 分支，**不是完整 median**。MAE 單位均為新臺幣元。

| 層 | 原 NN 路徑 MAE | R95 NN 路徑 MAE | Δ overall | Δ weighted |
|---|---:|---:|---:|---:|
| prior NN native | 1,834,776.3764 | 2,240,416.1406 | +405,639.7641 | +401,139.0061 |
| 融合 base | 1,479,752.8526 | 1,553,910.9619 | +74,158.1093 | +73,238.2308 |
| reference | 1,478,842.8619 | 1,568,853.7905 | +90,010.9286 | +88,814.3562 |
| spatial | 1,469,402.1087 | 1,558,782.8131 | +89,380.7044 | +88,167.2733 |

原 prior NN native MAE 也與原 R63 `native_MAE_diagnostic.R63_prior` 相同。原生退步約 **22.1084%**；融合及 median 減小了完整輸出的退步幅度，但沒有逆轉此期整體結果。這些層間差額不能相加成因果貢獻或獨立消融結果。

R95 的 reference MAE 高於其 base，spatial 再降低部分 MAE。固定 reference 並沒有保證對任意新 base 都改善，這個現象本身不是實作錯誤。空間收據為 9,656 列套用、71,578 列 fallback 原 reference 值逐位保存，沒有重新 fit reference 或額外對 G 套趨勢。

## 完整 median 與有利反證

主流程先重播原 R63 native→R71 seed42 mean/base/reference/spatial／diagnostics，以及 v22 完整 median，全向量一致；沒有重新訓或推論舊 NN。本審查另核保存的 v22、v24、v11、v12、v5 **完整 metrics dict** 各與原權威 results 相同。

| 完整模型 | overall MAE | weighted MAE | R95−該對照 overall | R95−該對照 weighted |
|---|---:|---:|---:|---:|
| v22 | 1,407,387.0708 | 1,399,717.7441 | +14,973.8663 | +14,733.9609 |
| v24 | 1,407,340.7168 | 1,399,695.9218 | +15,020.2203 | +14,755.7832 |
| R95 | 1,422,360.9371 | 1,414,451.7050 | — | — |

R95 完整 median 對 v22 整體退步 **1.06395%**。已見建案 28,981 列改善 1,414.1314 元；新建案 52,253 列退步 24,063.1159 元；缺建案群本期 0 列，不能填零 MAE。NN spatial 分支本身的 seen／new 則分別退步 6,490.9010／135,353.7470 元，顯示分支與最後 median 的群組表現不可混稱。

對 v22，Q1～Q4 分別退步 8,557.4854／11,677.3989／12,326.8061／23,112.0253 元。對 v24，Q1 改善 5,915.5098 元，其餘三季退步，故不能寫成所有對照的所有群組都更差。價格帶、更多誤差分解及配對區間尚待整批 generic／paired，這份單期檢查不作顯著性或 Private 勝率判斷。

## 實際時間與限制

本期 fit **614.032 秒**；trainer metadata elapsed **620.157 秒**；trainer 外層 execution **620.172 秒**；父程序 subprocess **622.328 秒**；含來源重播、控制與評分的 stage **655.219 秒**。各層互相包含，不相加。GPU allocated／reserved 峰值分別 805,418,496／1,004,535,808 bytes，未見 OOM／非有限值／自動重試；這是第一期實測，不拿三期合成估算當實耗。

本審查最初彙整誤把 metrics 的 `overall_mae` 當 `overall`，發生一次只讀腳本 KeyError；改讀正確 schema 後完成核驗，沒有修改訓練 source／artifact，也不是模型失敗。未量測整段人工審查時間。

後兩期仍照固定配置完成，之後再看三期門檻、泛化與有利反證。本文件沒有觸發新切點 fit、全量 fit、CSV、提交或使用者選型變更；v22 仍唯一選定提交，Private 未知。

## 關鍵 SHA-256

| 檔案 | SHA-256 |
|---|---|
| `src/round95_experiment.py` | `6893ee155b94d3bc5fd218695c903ba29cd34dc210432042615807d3f1c75912` |
| `src/round94_attention_prior_train_v2.py` | `ae197952be777f3e9e2fdb66458887d2a3885fd736b75dd3677eeae43cfeef83` |
| development `results.json` | `166eebc28ca2cf6b44918df5468edc4a2fd9bb53e3bb89beb8a05849b538281d` |
| development `model/training.json` | `843414a7a389a0b728d533b67f7b880831ae7b396433a7fd28f4d1bde4c2e72d` |
| development `model/model.pt` | `43d890d288ad74de9ae872a7ee3796799c9f5f3848812f0f7453aabe1e7cce1d` |
| `control_verification.json` | `b57607af186d939bfd76729085fe89cd01f8b477c7f7ff692585ed4ccf6f7f94` |
| `prepared_verification.json` | `46e0b8347b8e8e5d6c1df278c42553f56618eea795877c1ebb7b535f01c79f9a` |
