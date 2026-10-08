# R116 四 origin driver 作者核對

2026-10-08。僅完成 source 與 CPU 合成檢查，沒有執行 freeze／train／evaluate、真 fit、GPU、query 價格載入、fullfit 或 CSV。獨立 preflight 由其他代理及主流程協調；此作者收據不是獨立通過證據。

## 固定來源與介面

[driver](../src/round116_research.py) SHA `483d62747e08c4b11c0012b68fa960883522ab1625bf09d6b024396b2f541185`，獨立另存自凍結 R114 driver，原 R114 SHA `444113bfcb9ad321015c8af76f6218dfe3c7cd1fd1e659dde762ba765a39c95d` 保持；R116 helper SHA `ae9d61c6710ee83a654b6171e2fd5de782f51dc36642d69e4bfede16495e7cd2`，既有 [helper 說明](round116_method_review.md)與 [事前計畫](round116_plan.md)不改。

- request／seal／results 在 outputs/round116；正式 CPU execution、獨立 preflight／postfit 與主執行紀錄在 outputs/research_round116_20261008。
- CLI 為 --self-check／--freeze／--train／--stage ORIGIN／--evaluate，互斥。沒有 fullfit／CSV 功能。
- freeze 要求 preflight_review.json 的 status=passed，source_sha256 精確等於 `sources()` 的 45 個 Python 閉包加 PLAN 的 SHA，共 46 筆。規約 SHA 固定 `f9a4e0c8b7250d1fad5335d6fd02a92729d62bb8662d0376eeed286b8dd1cee6`。
- 另強制 R116_helper_selfcheck_execution.json／R116_driver_selfcheck_execution.json，命令各為 .venv Python、-B -X utf8、來源絕對路徑、--self-check，exit0 與該 entry 的完整 source closure 精確綁定。這些正式檔目前由主流程後續建立，作者不偽造。

## 四窗輸入與唯一替換

固定 development 240316／81234、confirmation 321550／133697、auxiliary 134805／105511、offset_2023_02_11 272117／107169，沒有修改共用 c.SPLITS。原始來源 raw 價格檔僅串流 SHA 預先綁定，直到獨立 evaluate 才反序列化。

August 使用實際 frozen API `round104_composition.specification/load_context`；沒有 context(spec,need_prior=False) 或 feature_arrays 這類不存在的 API。該 loader 僅載入純 X、保存 fit-only reference／spatial 和既有支線預測，完整重播原 v22。另取 R79 同切點 tree_geo，依 canonical fit／query 索引選列；對照 R79 bindings 的兩個 geography frame SHA。fit 價格由保存 fit_labels 載入且 canonical/source-cache 兩軸均精確匹配。

R63 NPZ 先核精確原九鍵、無 query_y 或重複鍵，August 只物化 train_sample_weight 與 train_y_million；float32 日期權重、索引及值 SHA 對照 R79 plan 與 R63 training，fit y_million 逐位匹配同一 fit y。單價 fit 不需要 NN 144 維編碼或 price factor。原 21 類別、地理增強後 95 欄／23 類別仍由原 helper 契約核對。

Offset 沿原已審 round113_offset_composition.load_context，保留原始純 X／tree fit-query、fit-label、保存 float32 日期權重、原九鍵 encoder／prior 的純 transform 重播與全部支線，沒有重訓 encoder／prior／R/S。固定 offset chain_unit；不套 NN metadata 或 signed residual 定義。

原 v22 unit42 native 重播 .75 unit＋.25 LGB、原 reference／spatial、與保存 NN／H median；原 v24 R83 native42/native137 各半，保持原 R/S 與 median，沒有加 LGB。候選只把 v22 unit42 native 槽換成 Depthwise；舊 NN／H／LightGBM／R/S／v24 皆保留。

## 保存與評分守衛

helper 只訓練一次，沒有 eval_set、early stop、選樹或 retry。每窗保存 raw.npy（**unit**，不是 signed TWD）、native.npy（原單次 inverse＋原非負 TWD）、CBM、started/state/metadata 六個原生產物；model_records 要求完整扁平集合，不硬套 NN training.json／model.pt。stage 封存六檔、train.log、全部五層、實際 frame／fit target／transformed target／date／area／Pool 權重 SHA、前後 state／舊支線／索引身分。日期來源由 driver 明確核到原切點，不將 helper 的 provenance_verified=false 塗改成 true。

validate 逐窗重新載入 CBM（CPU 推論物件，沒有 fit），由本輪 helper 核全部原生 passed／derived 參數、Depthwise／Plain／depth8／2999／global1／minleaf1、95／23 schema 與非對稱葉數 SHA／分布，精確匹配保存 reloaded_native_diagnostics；fit→reload 浮點差仍依 helper rtol1e-7。raw→TWD 單次 inverse、完整原生檔 SHA、五層與前後輸入身分皆重核。真正全 query raw/TWD 模型保存重載由 helper 在每個 fit 結束前執行。

--train 僅序列四個 stage，任一失敗保留 evidence、不重試；只有四個成功且全部 validate 後才寫封存。--evaluate 首先重核 request／source／input／protected、全部四個 model 與五層，並要求 training_execution exit0 且 failure=null，之後才讀 raw query 價格。仍使用原 seed／stage indices／配對方式；offset 不混作 development。保存五層、v22/v24、城市／月份／相對季度／新舊案／五真價帶與誤差尾部摘要。

兩個 combined gate 都是原三 August gate AND 原 offset 三條件；不修改任何門檻。只報 eligibility，預留 v31，無採用／fullfit／CSV／Kaggle 行為。歷史窗口反覆使用、重疊、舊控制未同期重訓、單 seed GPU 變異與 Private 未知等限制沿原規約保留。

## 作者 CPU 證據

[正式作者收據](../outputs/research_continue_20261008/direction_review_a/R116_driver_author_selfcheck.json) SHA `d38ac126f6c17ec2a1da69c8eec9c9f7b5e82704191c644ea709c44e460d2f0c`：40 合成 checks 通過，45 個 source memory compile，0 真 fit／GPU／真資料讀取。內部測量 2.750 秒，不含 Python 啟動或完整開發工時。

涵蓋原面積 fallback 與 unit→TWD／非法輸出、九鍵先驗再讀僅兩 fit arrays、日期 dtype／長度／非有限／零權重／fit-y 對齊、全部 combined gate 真值表、四窗及成功 exit 守衛、正式 CPU receipt command／closure／exit 身分、來源與 c.SPLITS 保持。一次中途 CPU fixture 失敗因 mock c.sha 範圍誤蓋 source closure；已限定只 mock receipt 路徑後重跑通過，並保存原因於作者收據，無真 fit 或研究分數。
