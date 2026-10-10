# v25 prior NN：真實訓練曲線的固定診斷規約

2026-10-10。老師要求以自己模型的訓練曲線取代欠擬合／過擬合示意。本規約在新增訓練與任何新 query 評分之前固定。原 R107 只存訓練模式的逐 epoch MAE／weighted SmoothL1，沒有逐 epoch query MAE；不能把舊訓練 loss 接上一條假想驗證曲線。

## 唯一資料與訓練

只新增 **一次 confirmation 起點的 R107 prior NN 診斷重跑**。fit 為 2023-08-11 以前的 321,550 列；query 為 2023-08-11～2024-08-10 的 133,697 列。它是原本已重用的歷史起點，不是 test、Private 或新盲測。

- 唯一 prepared：`outputs/round63/confirmation/prepared.npz`，SHA256 `4d4b7866f4f94068e60585b6351fd1a669be184d9e1f8d14cc4a5eac652a0aff`。九鍵 NPZ 不含 query y；144 數值／23 類別、fit-group OOF prior、完整 fit 的 query prior、原 float32 日期權重均保持，不 fit encoder／prior。
- 原 trainer：`src/round107_weighted_smooth_train.py`，SHA256 `f2330726165b1a102c893211268734c89e981b751f45167b82ae1c940917b36e`；原 confirmation training metadata：`outputs/round107/confirmation/model/training.json`，SHA256 `24fff85c1140e4abba01dd2c22c31163eb66400cf010a8e29f48c3121cdacf3b`。
- 新 trainer 另存 `src/round107_training_curve_train_20261010.py`。從原 source 複製，只在每 epoch 的 `schedule.step()` 後增加保存當期 checkpoint；30 個 snapshot 的 numeric_count、cardinalities、CONFIG 與 state_dict 使用原最終模型的同一格式。原 source 不動；保存精確 unified diff 與兩 source SHA，並由工具重建核對此唯一差異。
- weighted SmoothL1 beta=.1 百萬元、30 epochs、seed42、batch2048、原架構／初始化／AdamW／cosine schedule／dropout／浮點設定全保持。checkpoint 複製／保存不進行 forward、不使用 RNG、不選 epoch；原最後模型保存與全 query reload 檢查保留。
- 這次只增加診斷產物，不替換已選 v25 模型、CSV 或原收據；固定 seed 不保證與原 GPU 重訓逐 bytes 相同，不把差異硬修成相同。

## 嚴格先訓練，再看 query 標籤

外部入口為新工具 `src/v25_training_curve_20261010.py` 的 `--freeze`、`--train`、`--evaluate`，三個獨立程序。拒絕覆寫既有 request／執行收據／模型／curve。root 審查 source／人工檢查後，才可唯一真執行。

1. **freeze**：只讀來源與原 metadata、對檔案做 SHA，不反序列 NPZ、特徵表、價格表或模型。綁定此規約、source 閉包、原 R107 request／confirmation metadata、R63 preparation／prior-preparation、原 canonical／fit／query／source 索引及原 fit_y；綁定 raw 價格表 SHA 但不開啟。
2. **train**：驗來源／輸入 SHA；讀 pure-X 特徵表與固定 canonical mapping，按交易日期重新推導唯一 fit/query mask並逐 bytes 核原 canonical／raw 索引、列數、無交集。只讀該期保存的 fit_y，核對 NPZ train_y float32、原 factor／weight／cardinality 身分；不反序列 raw 價格表，不提供 query y。另起 Torch 子程序訓練一次；exit0、固定 30 epochs、全 query reload、30 snapshot 身分及最後 snapshot／final checkpoint 的完整結構與 tensor 值核對後才寫 seal。訓練失敗保留，不自動重跑。
3. **evaluate**：先核 train exit0、seal、全 30 snapshot 與原 NPZ／索引身分，才讀原 raw 價格 pickle。整份 raw 表會於此階段 materialize；只選這個 confirmation 的 query source rows 供評分，並用 fit source rows核保存 fit_y，不能宣稱全流程從未接觸整張 raw 價格表。query_y 私有保存，綁定索引、真價值 SHA、正值與列序。隔離 Torch 子程序依序載入 30 checkpoint，所有預測均 `model.eval()`、dropout off、`torch.inference_mode()`；將倍率乘原 price_factor 還原總價元後，分別對原 float64 fit_y 與該期 float64 query_y 算 unweighted TWD MAE。

fit 先驗是建案分組 OOF，query 先驗來自完整 outer-fit，兩者有不同建構方式；圖中 train/query 差距不能只歸因於過擬合。日期與先驗／encoder 的來源身分保持該時間起點，絕不使用 final 或較晚起點模型。

## 圖與採用界線

唯一曲線為 30 個 epoch 的兩條線：**train 總價 MAE（eval mode）**、**時間留出 query 總價 MAE（eval mode）**；相同新臺幣元單位與原生 prior NN 分量。舊 train-mode weighted SmoothL1／dropout MAE只留作訓練日誌，不能冒充同口徑的 Ein。

這是既有 v25 配方的一次診斷重跑，並非原 v25 每 epoch 的實際軌跡、v25 三分支 median 的學習曲線或 Private 曲線。無 early stopping、無 best epoch 欄位／挑選、無 beta／seed／epoch／比例搜尋、無新 CSV、無 Kaggle 操作、無選定改動。曲線即使未呈典型「先降後升」，也照實呈現，不改參數湊示意。

新產物位於 `outputs/v25_training_curve_20261010/`：freeze／train／evaluate 各自 execution 收據、來源 diff／SHA、完整 fit/query 身分證據、30 私有 `.pt` snapshots、私有 query_y、最後模型與預測，以及僅含 30 點聚合 MAE 的 `curve.json`。逐筆預測、標籤、模型、learned state 與九鍵 NPZ不進 GitHub；可發布新 source、規約、聚合曲線與不含逐筆資料的身分／執行證據。主線後續另用這 30 點繪圖，不能在真執行前填造數值。

人工檢查只使用合成陣列／假的依賴，驗 checkpoint 插入的精確 source 差異、固定30及loss配置、日期邊界與索引不符拒絕、MAE單位、30snapshot缺漏／identity／checkpoint結構拒絕、train-before-evaluate gate、拒絕覆寫及 root內路徑。人工檢查不讀真 NPZ／feature／價格／模型，不呼叫 setup／CUDA／真 fit。
