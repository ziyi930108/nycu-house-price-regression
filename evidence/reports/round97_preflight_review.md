# R97 主流程前置獨立審查

日期：2026-09-29（Asia/Taipei）。**最終主流程無剩餘 blocker，可依固定計畫執行七個原生模型。** 無阻擋訊息先傳給主線，書面隨後落盤；此時 `outputs/round97/started.json` 已存在。本文件不宣稱新窗口 fit、保存重載或評分已完成。scorer 由另一代理另行封存／審查，並非本文件放行範圍。

本次只讀 source、小型 metadata、既有訓練收據及檔案雜湊。未 prepare 新 NN、未載模型、未 fit／predict／score，未讀 query 價格或 test，也未修改 R90 helpers、原模型、舊計畫或 GitHub。只新增本文件。完整七 fit 由主線協調單 GPU 串行。

## 範圍與整合契約

[固定計畫](round97_plan.md)為 fit < 2023-02-11、query 至 2024-02-10（含）：272,117／107,169 列。完整 v22 與 v24 共需五樹＋兩 NN；無 Attention、R96、新提交或比例搜尋。既有 R90 純 X 完成不能當成新窗口模型已完成。

|元件|固定介面／邊界|保存與重載|
|---|---|---|
|C_raw|87 欄原 X，總價／1e6，2999 棵；不給 eval_set|CBM 重載後全 query TWD 逐位核對|
|L_unit|87 欄，fit-only category maps；未知 query 類別 -1，安全欄名；3999 棵，y/(A×1e4)、weight=A/100|category_state JSON 往返再編碼全等；LGB sklearn 與保存 Booster 全 query 相等|
|C_log|R46 原 95 欄、2999 棵；新 fit 自算 float64 日期權重|原 log state、CBM、metadata 保存，query 依原逆轉換重載|
|C_unit42|R79／R19 原單價轉換；新 float32 date × float64 A/100|直接 R19 feature state 與 CBM；完整 TWD 重載|
|C_unit137|R83 只改 seed 137，feature/target/date/Pool weight 必須與42相同|R83 wrapper 包含直接 R19 state；raw unit 與 TWD 全 query 重載|
|N_area|新 fit R47 encoder＋R54 ramps，原 R48 seed42／30epochs；面積 factor、uniform total-price loss|八鍵 NPZ；model.pt 完整 query TWD 重載|
|N_prior|同新 fit encoder，原144 numeric；新 R60 grouped OOF/full prior、float32 date，R63 seed42／30epochs|九鍵 NPZ；model.pt 完整 query TWD 重載|

LightGBM 的 mappings 僅由 train 建立，query 不擴充 vocabulary；JSON 還原保留 string keys／integer codes，query 編碼逐位重驗。R79 state 的 area-only 訓練字樣是歷史 provenance；真正 date×area 權重保存於 fit metadata，不能把兩者混稱。

R90 `prepare_nn` 保存新 encoder／ramps／五折 OOF prior／full prior／folds／canonical indices，prior 乘面積在 float64 完成後才 cast32。date 由 fit-global normalized64 cast32，不再正規化；無效日期權重維持1。只 N_area 加 ramps，N_prior 維持144欄。五折先驗是 group OOF，不是時間 OOF，不額外訓五個 NN。driver 傳 `data.fit_indices` 給 OOF，`fit_source_cache_indices` 只用於隔離標籤擷取。

新 R2 reference 與 R8 spatial 各建一次；四條 base 各自產生 reference／confidence／spatial，不能共用不同 base 的診斷或平均已修正分支。樹／NN 用原 R6 地理，spatial 用 R39 補位地理。核對實際 R2 `fit_report.fitting_rows`／`validation_targets_used_for_references` 和 R8 `state_summary`／`query_used_in_build` 都存在。

```text
N   = .75*(.5*N_area + .5*N_prior) + .25*L_unit
H   = .75*(.5*C_raw + .5*C_log) + .25*L_unit
U22 = .75*C_unit42 + .25*L_unit
U24 = .5*C_unit42 + .5*C_unit137
v22 = median(RS(N), RS(H), RS(U22))
v24 = median(RS(N), RS(H), RS(U24))
```

U24 沒有 .25 LightGBM。四分支原／重載 R/S 共8次 reference＋8次 spatial；保存 reference/spatial diagnostics，fallback、完整向量、state 使用前後及 caller frames 均設核驗。這些是待真執行的 runtime assertions，不是本次已完成推論。

## 真實 API／工程檢查

- R90 `fit_trees` 建立不存在的 trees 目錄；unit137 子目錄另建。NN preparation／training 目錄須空。原 R48/R63 trainer 允許已存在目錄與 stdout.log，只拒 run.json／model.pt／training.json／predictions_twd.npy；其本身不寫 execution.json，與父 helper 不衝突，不能套用 Attention trainer 的整個 out 不得存在契約。
- 原兩 NN trainers 都先驗 NPZ exact8／9鍵才讀內容，不含 query labels。`_expected_config` 以 AST 隔離純函數比對原 R54/R63 三期共六份 training.json，完全一致；預期 source／architecture／factor／weight／epoch／reload 欄位均存在。未 import Torch 或啟動 trainer。
- `.venv/Scripts/python.exe` 與 `.venv_torch/Scripts/python.exe` 實際存在；父環境 metadata 為 catboost1.2.10、lightgbm4.7.0、numpy2.3.5、pandas3.0.1。R90 NN 以後者 subprocess 串行啟動兩 trainer，避免與同程序 CatBoost 混合 GPU 任務。
- R90 NN 自存 prepared.json、encoder／prior／category JSON 等不保證個別 sidecar；driver 依 artifact map／固定 SHA 讀取，未錯用 locked。純 X preparation manifest 本身確有 sidecar，實核符合固定SHA。
- R90 八個 LOCKED_SOURCES 實際SHA一致；最終 driver 合成收據完整30項 source closure 另獨立核對一致。純X的25 input paths沒有 test／valid.pkl 子字串誤拒。
- [隔離擷取程序](../src/round97_prepare_fit_labels.py)真正已跑0.813秒、exit0；獨立核 fit_y檔、pureX manifest、兩indices、extractor共五檔SHA，兩索引內容身份對上pureX manifest。沒有重開原raw價格表或materialize價格陣列。pickle整表反序列化確實發生在隔離extractor，不能宣稱整個工作流從未讀query價格；它只按核實fit索引輸出fit_y、沒有query統計。模型driver只載此fit-only檔，原raw來源僅繼承宣告；scorer待封存後才讀query標籤。
- 封存向量與 metadata 使用新增目錄／拒覆寫，原 CSV、選定、正式報告／簡報保護 SHA 不變，無自動重試。訓練主流程不呼叫 scorer，主線另以完成／封存身分啟動評估。

## 啟動前抓到並修正的兩個契約問題

1. driver 初稿 `load_fit_labels` 沿代理草稿schema，與root實際 `fit_labels_sealed_no_training`／flat fit_y_path 欄位不同。root與作者在啟動前對實際manifest修正；本審查核對最終欄位存在。
2. 初稿 SHA `6d287909a2d6b61022462cfb2f67e6b3786e29c5fe7aaab12dc858749e69b26d` 把 R79 seed42 direct state 誤當 wrapper，使用 `seed42_state['legacy_feature_state']`，會在seed137 fit後 KeyError。本審查讀原 R79 JSON 確認並通知；最終抽出 `verify_unit_seed_pair`，改比 `state137['legacy_feature_state'] == state42`。保留首份合成收據，新增 direct接受／錯誤wrapper拒／不同weight拒 三例，最終18項合成passed、1.565秒、0真fit/GPU。

以上為未凍結整合草稿的修正，未修改任何 immutable helper／訓練科學配方，亦未先花真fit成本或看新分數才改程式。本審查沒有重跑合成整套；獨立讀最終增量、AST及收據/source身分。

## 已有成本證據與限制

|既有元件|三原窗口 fit 秒數範圍|證據|
|---|---:|---|
|R46 log|109.547–135.187|outputs/round46/results.json|
|R79 unit42|151.250–190.125|outputs/round79/results.json|
|R83 unit137|136.078–175.859|outputs/round83/results.json|
|R54 area NN|10.344–24.093|三期 model/training.json|
|R63 prior NN|10.218–26.968|三期 model/training.json|

既有 NN fit 列数134,805～321,550，area寬590／599、prior144；R54量得GPU allocated約0.683～1.265GB，R63約0.243～0.425GB。新窗口272,117列介於既有規模，但ramps/cardinalities、實際資源與資料準備時間仍應以本輪收據為準。表中不含 raw/LGB、prepare、reference/spatial、重載或評分，不能相加冒稱七模型總工期。未做新 benchmark，不承諾GPU重訓逐位一致，只要求保存模型全query重載一致。

錯開切點仍與舊資料重疊、配方受先前選型影響，不能稱盲測或 Private 保證。最終採用與分群結論需等新窗口封存後獨立scorer；本輪只是完整 v22/v24 穩定性證據，不自動換選定v22。

## 本次固定身分

|檔案|SHA-256|
|---|---|
|`reports/round97_plan.md`|`8372960e832bd66203374c942e0f7988e66522aedf358c894d892e82669a3a08`|
|`src/round97_offset_experiment.py`|`00d927b81170b9b8acfab984bb72635f413513f62da42733b85e5873528914f1`|
|`src/round90_offset_data.py`|`4f58575bff75b69fd900af84f9152dc3e900f34ad6909e0f2e0366bb67c928f1`|
|`src/round90_offset_trees.py`|`666e9fc61d22565aae2f738767bfef471e12679d4e1c0bbe86cf366afa540110`|
|`src/round90_offset_nn.py`|`b473b246925dffe3d084c909679892f45e33f19ed0a6c18c100cfc836bc35e06`|
|`src/round83_seed_unit.py`|`c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b`|
|`src/round97_prepare_fit_labels.py`|`bf7c8c01265735293221cd09339a96f0c0a0625afc8fbca8ab0e1b11792f9ef1`|
|`reports/round97_driver_selfcheck_v2.json`|`eb99076e36f882ba79b34344e8e34d1b999846104ee7e6e71546c99670894d77`|
|`outputs/round97_inputs/manifest.json`|`e1250558e16b463ad8eca37eda324b3b65deaa3fe014fa6668d7aba55e098a4b`|

[新driver](../src/round97_offset_experiment.py) · [最終18項合成](round97_driver_selfcheck_v2.json) · [R90 NN 11項合成](round90_nn_selfcheck.json) · [標籤擷取收據](../outputs/round97_inputs/manifest.json) · [模型數與舊介面盤點](round90_validation_inventory_after_attention.md)
