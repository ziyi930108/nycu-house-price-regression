# R116 方法與作者 CPU 核對

2026-10-08。狀態：只完成新 helper、合成檢查與零 fit 參數解析；沒有真實模型訓練、評分、CSV 或發行。此作者核對不是獨立 preflight。四 origin driver 由主流程另建／預審，訓練前仍需凍結完整來源、規約、輸入與保護清單。

## 固定假說與歷史反證

原 R79 同層共享一個條件的 SymmetricTree，改為每個葉節點可選自身切分的 Depthwise。不同地區／建物可能需要不同條件；更自由也可能擬合稀疏交易，沒有預期保證。這是固定一個成長政策的交互假說，沒有深度、最低葉筆數、種子或融合比例掃描。

|歷史方法|已保存反證|對本輪的界線|
|---|---|---|
|[R27 depth10](round27_report.md)|raw／uniform development 整體 +0.8988%、新案 +1.4659%；後兩窗依預先停止規則未跑|支持容量風險，並非同一 unit／date Depthwise 試驗|
|[R84 4499 trees](round84_report.md)|相對 v24 最終 median 三窗 +930.08／−1999.31／+7114.59 元；單支進步未必傳至 median|反對只看原生模型或單年度來採用；2999 舊控制未同期重訓|
|[R92 depth6](round92_report.md)|相對 v22 三窗 +1812.80／+21554.79／−6753.95 元；三窗 20–50m 群都退步|減少容量亦不保證改善；保留高價尾部有利結果|
|[R112 Ordered](round111_112_report.md)|相對 v22 三窗 +4288.21／+16317.73／+8689.27 元；最新新北市 21822 列退步 5.08765%|更換政策曾失敗，但 Ordered 不等於 Depthwise，不能推論後者已全盤證偽|

[CatBoost 官方 grow_policy 文件](https://catboost.ai/docs/en/references/training-parameters/common#grow_policy)於本日確認：Depthwise 每層各葉各自選切分，支援 CPU／GPU；只可匯出 JSON／CBM，且不支援 PredictionDiff。此處使用 CBM 與一般 predict，不使用 PredictionDiff。官方 min_data_in_leaf 適用 Depthwise／Lossguide、預設 1；max_leaves 僅適用 Lossguide，本輪不傳。相同 depth8 上限 256 葉不代表同樣假說容量或正則化，實際原生派生欄不宣稱全數相同。

## 精確傳參與狀態

[新 helper](../src/round116_depthwise_unit.py)獨立另存自 R112 已審查的保存／重載外殼，實際只匯入未改的 R79／R19 來源。implementation_parent 保存 R112 路徑及 SHA；R112、R114、舊原生檢查器與既有規約均未修改。

相對 R79，`changed_parameter_names=[]`，`added_parameter_names=['grow_policy']`，唯一新增 `grow_policy='Depthwise'`。原傳入參數逐欄完全保持：Plain、depth8、2999trees、lr .065、seed42、MAE、max_ctr_complexity2、95 欄／23 增強類別、GPU0／gpu_ram_part .75 等。ctr_target_border_count、min_data_in_leaf、max_leaves 不顯式傳入；實際 native 必須回報 global border1、min_data_in_leaf1。21 個原始類別與加地理後 23 個類別分開核對。

目標沿 y／(有效面積×10000)，Pool 權重沿有效面積／100×保存日期 float32 權重轉 float64；不重算、重正規化或讀 query 標籤。日期權重來源、切點、索引與檔案值 SHA 必須由 driver 核驗，helper 不冒稱已核到資料來源。R19 legacy feature state 僅描述純 X 特徵與歷史訓練來源，不能誤認為本輪 native 的 growth／weight 設定。

## Native、產物與失敗契約

- 所有 18 個可恢復 passed 參數均由 `get_all_params` 核對；有限 float rtol1e-7／atol0，其他型別和值精確。僅 `allow_writing_files`、`thread_count`、`verbose` 無法由 CBM 原生恢復，明列原建構值，沒有假稱獨立驗出。
- Native 必須 Depthwise／Plain／GPU／depth8／2999／global border1／min leaf1；逐樹葉數須 2999 個正整數、各 ≤256。接受 3／17 等非 2 的冪次，記錄完整向量 little-endian int64 SHA、最小／最大與精確頻數分布，沒有套原 SymmetricTree 檢查器。
- 完整 native_parameters 和派生欄名稱均保存，fit 與 CBM reload 皆獨立核對；只允許並列明原生有限浮點序列化容差。每個 raw unit 輸出只使用原 inverse 一次，再套原 TWD 非負投影，無 cap／reference／spatial 加在 helper。
- 產物：started.json、model.cbm、feature_state.json、raw.npy、native.npy、metadata.json。全部 query raw／TWD CBM 重載及 NPY 保存重載逐位相同，非抽樣核對。metadata 記錄全部 source/frame/target/date/area/combinedweight/model/state/prediction SHA 與真實計時。
- 不支援設定、資源或後驗核對失敗時，保留 started／partial artifacts 與 failed.json，原例外照常上拋；不自動重試、改參數或退回 CPU。舊輸出目錄拒絕覆寫，錯誤也不新增或改寫該目錄。失敗收據不捏造 fit 是否成功或 completed metadata。

## 本次作者執行證據

[R116 作者收據](../outputs/research_continue_20261008/direction_review_a/R116_helper_author_selfcheck.json)：80 個 CPU／FakeModel 檢查通過；2 個完成的 Fake fit 及 1 次模擬不支援的 Fake fit 呼叫，**0 真 fit／0 GPU／未載入真實訓練資料或 query 價格**。18 passed 參數逐一錯值／缺失拒絕；錯 min leaf、非法葉數／型別／形狀、錯 raw→TWD、異常重載、錯權重／索引、既有目錄覆寫與自動 fallback 均有檢查。相同程式的本機 CatBoost 1.2.10 `_check_train_params` 零 fit parser 接受固定設定；這不是 GPU kernel、記憶體或 MAE 效果測試。

source SHA：`ae9d61c6710ee83a654b6171e2fd5de782f51dc36642d69e4bfede16495e7cd2`。收據 SHA：`2a8fdf1be59a14cd43e8b90cad9039a96680e18130c65ca03f574b770dc1b1cf`。正式作者檢查內測量 0.563 秒，不含 Python 啟動／先前開發檢查；不當作全部開發時間。

四窗與組合／bootstrap／gates 只沿 [事前固定規約](round116_plan.md)，本 memo 不新增門檻或訓練。歷史窗口反覆使用且重疊，無新的獨立盲測；固定 seed 的 GPU 重訓不保證 bitwise，保存重載不等於乾淨環境從原始資料全鏈重訓。v22 保持選定，Public／Private 未知的新候選分數均不填，Private 仍未知。
