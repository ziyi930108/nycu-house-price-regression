# R105 後有限方向盤點：四個識別類別的聯動 dropout

2026-09-29。僅讀既有 source／report；沒有讀逐筆陣列、價格或模型，沒有新計畫、實作、fit、預測或評分。唯一提交仍是 v22，Private 未知。

## 唯一保留的假設

可以考慮在原 v22／R63 prior NN 上，**只把四個類別的 dropout 改為聯動**：`project_name`、`address_raw`、`building_and_unit`、`building_block`。仍使用原 144 數值欄、23 類別欄、group-OOF prior／價格因子、全部 fit 列、保存日期權重、總價 MAE、seed／epoch 與完整組合；不沿用 R102 的時間 prior／145 欄，也不再調 R105 的 alpha。

[R47 forward](../src/round47_nn_train.py) 已對每列每類別獨立產生 `torch.rand(categorical.shape) < 0.1`，再把被遮罩的 code 設為 0；[R63](../src/round63_recency_weighted_prior_train.py) 直接沿用該 `EmbeddingMLP`。因此 **unknown embedding 原本就有訓練**，不能把本提案說成首次處理未知類別。[原 encoder](../src/round47_nn_encode.py) 的 0 也代表未見／少於五次的類別。

四欄聯動的動機是讓訓練偶爾同時失去這幾種識別線索，而非遮住案名卻仍依靠完整地址等替代線索。只談人工 dropout 事件，原四欄全遮機率為 `0.1^4`，聯動後為 `0.1`；每個欄位被人工遮罩的邊際機率仍為 0.1。原始 code 已為 0 的情況另當別論，這不是實際未知率或觀察到的效益。

可沿原完整 23 欄隨機 mask，再將四欄的 mask 複製為其中固定一欄（例如 `project_name`）的原 mask；原 source 的二維形狀容許這樣做。欄位位置須由當期凍結 encoder 的類別順序驗證，不能猜索引。這能保留其餘 19 欄的 mask 與該次隨機抽樣數量，且不新增參數或改變 eval forward。仍須以人工對照確認其他 dropout 的 RNG 消耗、初始化與保存重載；不能僅凭同 seed 聲稱 CUDA 重訓逐位確定。

## 與已做實驗的差異及反證

有限搜尋 `src` 與研究報告，未找到這個 **NN 每次 forward、四欄共用原 0.1 mask** 的已完成實驗；這不是聲稱所有歷史二進位產物都已鑑識。

- [R5](../src/round5_project_mask.py)／[R42](round42_experiment_report.md) 已做四欄聯合遮罩，不能說聯合遮罩本身全新。它們是樹模型中固定 SHA 選約半數建案，fit／query 採固定整案規則，新案一律遮罩；不是本提案的訓練時逐列隨機 dropout。**最強相近反證**是 R42 對 v7 開發／最新／最舊整體變化為 +0.8358%／−0.1141%／+0.9407%，新案三窗都退（+1.0479%／+0.1109%／+0.9591%）。
- [R18 永久移除四欄](round18_report.md) 開發期整體 +0.9562%、新案 +1.3706%，沒有執行另兩窗。移除識別資訊並不天然有利於冷啟動；以上方法不同，也不能直接證明新 NN 遮罩必敗。
- [既有冷啟動盤點](round17_cold_start_review.md) 已指出新案不等於所有類別未見。本提案**不遮數值 prior、其餘地址拆解欄、地理特徵或其他代理線索**，不等同真新案，更不是完整去識別。共享 mask 也會減少部分遮罩組合的多樣性，可能傷害已見案或既有分支互補。
- [R105](round105_report.md) 固定原生平均最終兩窗改善、最新窗 +2,010.70 元；兩門檻未通過。不能因接近而繼續搜尋 alpha，也不能由 native 改善推定完整 base／R／S／median 必然改善。

## 可行入口、成本與邊界

若主線另行固定計畫，可新建 trainer 類別，沿 R63 loss／九鍵 prepared 合約，只改訓練 forward 的四欄 mask 關係；原 source 不覆寫、不 monkeypatch。以同資料／初始化的原獨立 mask 作控制，三窗各 C／T 一次共六次 NN fit，可區分聯動變更與既有保存模型／執行環境差異；不能主張已消除所有 GPU 非確定性。query 不另遮罩，原 area／LGB／H／U 與同窗 R／S state 維持，仍要保存完整五層控制和固定全群／配對評估，不能挑新案子群採用。

資料／encoder／prior 可以重用相同切點的保存產物，不需要新教師或新 prior 建立；實作時仍須核來源 SHA 與逐列身份。本次未載入它們驗證新 driver。成本參考僅為已測量歷史：[R63](round63_report.md) 三次歷史 NN fit 合計 56.436 秒、主流程 111.735 秒；[R102](round102_report.md) 六次 NN fit 合計 107.155 秒、主流程 189.891 秒。它們不是本提案 runtime 預測或預算承諾，檔案驗證、重播與評分另有成本。

目前只保留這一個候選供下一輪決定，未立正式計畫、未啟動訓練，也沒有搜尋遮罩率、欄位集合、seed 或融合比例。反覆使用的年度仍非新盲測，2023 靜態地理可用性限制仍在，bootstrap 不能換成 Private 勝率。
