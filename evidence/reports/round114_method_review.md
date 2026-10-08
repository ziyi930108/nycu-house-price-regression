# R114作者方法檢查：參考總價＋有正負值的金額修正

2026-10-08。這是新程式的設計與作者CPU自查說明，**不是模型效果報告，也不是獨立預審或訓練授權**。目前沒有本方法的房價fit、驗證分數、Private分數或提交CSV。

固定科學規約：[round114_plan.md](round114_plan.md)，SHA256 `45b4095430f6adffab7977b37e60eb0e30f958e1a4634e6dd251739acd5b126a`。原三個8月起點及2023-02-11起點都須執行，不能看前三個分數才決定是否做第四個。四起點成功封存後才另讀query價格；研究與全量提交分開。

## 改的是什麼，為什麼值得測

原R63把參考總價F乘上NN輸出的正倍率；本方法保留同一份F，讓NN學習金額修正h：

```
原模型：total_million = F_million × Softplus(z)
本方法：raw_total_million = F_million + h
訓練：mean(w × abs(raw_total_million − y_million))
推論raw：float64(F_float32 + h_float32) × 1,000,000
推論native：max(raw_total_TWD, 0)
```

式中推論先把兩個float32值各自升為float64再相加、換成TWD；沒有先以float32合計再升精度。F是原有的有效面積乘參考單價；fit為既存建案group OOF，query為各外層fit自己的full-fit prior。F含fit標籤資訊，不能說成純X，亦未改成時間OOF。

原模型對head preactivation z的直接梯度，除不可微點外，為`w × F × sigmoid(z) × sign(error) / batch_size`；新模型對h為`w × sign(error) / batch_size`。這是參數化差異，不是舊梯度有害的證明，也不保證新的優化步幅更好。舊F尺度讓相同相對倍率改變對應不同金額；移除它也可能失去有用的尺度歸納。隱藏層梯度仍取決於head權重和網路表示，不能稱完全消除價格尺度效應。

這次同時更換head activation、輸出單位、bias、正值約束和梯度尺度；**不是純loss對照，也不是單獨隔離某一條梯度的因果實驗**。AdamW、learning rate、epochs照原樣，不嘗試補償步幅。

## 刻意保持的部分

- 原144數值欄與23類別embedding；原`min(32,max(4,ceil(sqrt(cardinality))))`維度、0.1獨立category dropout及unknown代碼0。
- 原MLP隱層512→256→128，各層Linear／LayerNorm／ReLU／Dropout0.1；原Linear1 head的隨機weight與建構RNG順序不變。
- seed42、固定30epochs、batch2048、AdamW lr0.001／weight_decay0.0001、cosine eta_min0.00001與原推論chunk4096。
- 原九array及float32日期權重；不重算權重、不做batch正規化、不用query y、不傳eval_set，不依training metric選epoch。
- 僅替換v22的prior NN槽，其他native、reference／spatial、比例與median由各起點原流程保留。

唯一初始化規則是head bias使用fit的`median(y_million−F_million)`。先以float64計算中位數，再依原parameter的float32型別賦值；metadata分開保存中位數與實際float32 bias。**weight沒有歸零，因此初始預測不保證等於prior或prior加上中位數**。CPU檢查比較同一RNG狀態下原模型與新模型的全部初始parameter，確保沒有額外初始化抽樣。

## 負值怎麼處理

訓練使用未clip總價；即使F+h低於0，仍保留MAE梯度。只在推論時作一次固定的`max(total,0)`，不設正的價格下限、上限、ratio cap或依ID例外。對非負真價，這個投影不會增加單筆絕對誤差，但不能由此推論新模型比原模型好。

保存兩份完整float64 TWD向量：`predictions_raw_twd.npy`是**未clip總價**，不是h；`predictions_twd.npy`是投影後native。診斷保存未clip總價負值筆數／比例、最低與最高值、零值及投影修正量，不讀query y。負的h可能只是下修過高prior，不能與負的總價混為一談。完整組合只接投影後native；原始未clip向量保留供審查。

## 最強反證及不能承諾的事

| 既有方法 | 觀察 | 與本方法的關係 |
| --- | --- | --- |
| [R91](round91_report.md) 加法prior CatBoost | 中間／早期整體MAE約退步47,037／37,857元；最新約改善4,395元。中間／最新億元群約退步13,695,997／4,427,483元 | 已直接反對「加法prior自然更好」或「這能解決高價房」；learner、目標尺度／weights與替換槽不同，不是完全相同方法 |
| [R104](round104_report.md) 教師offset NN | 中間／最新整體約退步39,238／11,007元；中間native約退步4.305百萬元 | 它仍用ratio head、145欄、教師子集及不同loss，沒有直接測本signed head；仍反對offset處理必然有效 |
| [R102](round102_report.md) 顯式prior feature | native三窗改善，但完整組合中間／最新約退步934／7,219元 | native改善不足以選模型，必須看固定完整組合與分群 |

[先前方向審查](../outputs/research_continue_20261008/direction_review_a/direction_review.md)也指出少數億元案並非總誤差的主要來源；本方法不宣稱針對少數豪宅即可大幅降低MAE。這些反證與本次結果無關，保留在訓練前固定的規約中。

## 程式與保存契約

新程式為[src/round114_additive_prior_train.py](../src/round114_additive_prior_train.py)，不修改舊source或globals。父系為原R63 trainer與原R47架構；原R63也核R52來源。各父系SHA與完整baseline/config差異保存於checkpoint／metadata。九NPZ鍵集合及數量須在任何array值讀取前完全吻合，拒絕query_y替換、額外鍵或重複鍵。

```
.venv_torch/Scripts/python.exe -B -X utf8 src/round114_additive_prior_train.py --self-check
# 上面只用人工CPU資料。
# --data PREPARED.npz --out NEW_MODEL_DIR 是研究driver將來使用的入口，須另有固定規約與獨立預審。
```

metadata用`input_arrays`保存全部九鍵dtype／shape／values SHA；記實際class、F／權重identity、初始bias、完整30epoch紀錄、負值診斷及source/input SHA。Trainer只驗數值與身分契約，**不自行證明OOF、dateweight或encoder的來源正確**；這由外層driver的fit-only回放與凍結負責，metadata明列這個限制。

保存`run.json`、`training.json`、`model.pt`及兩NPY。原有任一名稱存在即拒絕；工程失敗保留，不自動重試。`run.json`為started，成功才另寫completed的`training.json`。模型checkpoint記實際`AdditivePriorMLP`與完整config，載入前檢查state keys、dtype、shape、finite，避免框架默默轉型。完成fit時要求完整query的raw與native重載bytes相同，再保存NPY並核identity、source及輸入SHA。這不是GPU重新訓練可逐bit重現的聲明。

作者自查涵蓋原參數初始化、head／bias、未clip目標autograd、F尺度梯度、負值／零界線、九鍵先驗檢查、dtype／finite／factor／weights／cardinality、隱藏層梯度、in-memory checkpoint完整重載及拒絕silent cast／錯誤投影policy。自查只用人工CPU資料，不做optimizer step、房價fit、GPU呼叫、query評分或CSV。

## 接續門檻

本程式不代表已核准執行。需獨立source preflight、四起點來源與規約freeze、單GPU串行fit、四個modeldir與五層向量sealed，再分開eval與獨立postfit。三窗原探索／強門檻外，2月起點亦須整體／加權退步各≤0.25%、沒有季度或≥500列群退步>5%。只有探索全部通過與獨立postfit通過，才另立單次全量規約／final preflight；預留v30不代表一定產生CSV。

四窗口重疊，已參與過多輪研究，並非四個獨立盲測；原保存對照也不是同期matched refit。沒有新方法分數或Public／Private結果。v22選定、v25探索檔及原上台材料保持。
