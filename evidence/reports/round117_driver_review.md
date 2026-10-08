# R117 四起點 driver：作者介面與守衛審查

2026-10-08。本文描述已完成的 source／人工 CPU 自檢；沒有真實價格先驗建立、九 array 準備、NN fit、GPU、query 評分或新 CSV。作者自檢不是獨立 preflight，也不是已完成實資料重播。

## 固定來源與接口

- 新 driver：`src/round117_research.py`，SHA256 `cd2e7464b6218909f6cebd9a06124f01f46db8df73ef3ec7b69ed138036e98c8`。
- 固定規約：`reports/round117_plan.md`，SHA256 `c4084bee6bbe0f9ed97e019a88f48cbeee992f3a297a1b6ce251709f5377e8eb`。
- 先驗 helper：`src/round117_transaction_prior.py`，SHA256 `193cabdbde1021eee7144d07e218c065dd2910426a16e0b90bef2816ad56b853`。
- NN trainer：`src/round117_transaction_prior_train.py`，SHA256 `5f479e4e03bac83bd67896dcdc3925d8d68dfc555bf5b2b8d0eaaa6cc6dc6c69`。
- 完整 research Python closure 50 筆，另加 plan 為 51 筆。Trainer helper 自述的四個來源包含以固定路徑驗證的 R52；本輪完整閉包亦包含它。正式單工具執行收據的 `source_sha256` 按 `c.source_hashes(entry)`，與該工具自檢回傳的額外父系列表分開，不猜兼容別名。

沿用 frozen R114 的 `august_specifications`、`load_context`、`prepared_arrays`、`verify_context` 及原 v24 重播；只調用 context／驗證函式，沒有改其 source、globals、split，也沒有使用其加法 NN／signed head。三 August 的單純 NN 八 array 為原 R63；February 用 R113 原 R97 context。四起點為 development、confirmation、auxiliary、offset_2023_02_11，列數／時間範圍與計畫相同。

先驗 API 為 `fit_oof_prior(Xfit, yfit, canonical_fit_indices)` 回傳 OOF 單價、原 R60 全 fit state、交易 folds、獨立 evidence。Trainer 命令為 `--data NEW_PREPARED --out NEW_MODEL_DIR`；無 query labels。模型仍為原 `round47_nn_train.EmbeddingMLP`、R63 ratio weighted MAE。AST 只解析新的 provenance 字串，原 R63 保存 CONFIG 合成後只有描述性 `train_price_factor_construction` 不同。

## 準備、封存及查詢分界

`specifications()` 僅讀 metadata／串流雜湊，先要求 R115 獨立 postfit passed、完整來源、request／results／正式 freeze/run／四個聚合檔 exact binding，以及零價格模型／評分範圍。再核本輪三份正式 CPU check 收據及全部舊輸入。獨立 preflight 必須綁定完整 source＋plan，且 `input_sha256` 與此函式返回的字典逐筆相等；freeze 才另外將 preflight 自身身份加入 request，避免自我雜湊循環。

真正 prepare 只在 frozen request 和主線 serial training marker 已存在時由 stage 執行；沒有獨立 prepare CLI。每個 stage：

1. 原 v22／v24、prior 槽、原九 array、fit labels 與 fit/query 狀態先重播。
2. 原 state／OOF／query prior／group folds 直接綁原 R60 artifact 身份，February 則綁原 R97 seal。用原 OOF/query 單價重播舊 F。
3. 新 folds 必須 exact 等於 seed42 的單次 permutation、`folds[p]=arange(n)%5`。新 evidence 的 canonical labels／indices／每 fold complement identities、OOF／area／unit／state SHA 均核對；禁止把 legacy group 字串當交易 fold 證據。
4. 新 full-fit state dict 必須等於原 state；保存時直接複製原 state 完整 bytes，SHA 相同。新 query prior 與原向量 exact，float32 query F 與原 array exact。
5. 新 archive 只准 `train_price_factor_million` 改變，另外八個 array 逐 dtype／shape／bytes 相同。保存原／新 fit F、fold／OOF／query prior、full state、evidence、九 array、初始化及私有 artifact 身份。另記 fit F 差異統計，明示不是 query MAE。
6. 原 median(y/F)／inverseSoftplus bias 需符合保存原 trainer metadata；新初始化依同公式重算，另存 f64 公式值與實際 f32 bias。不是保持初值不變的實驗。
7. 新 NN 完整 ratio／TWD 及 checkpoint 重載 exact；driver 驗證 ratio×原 query F×1e6 的單次逆變換。僅替 prior NN 槽，完整 native／base／reference／spatial／median 五層保存；原 R/S 各一次。

四 stage 全部新 prior／prepared／模型／log／五層封存且 training exit 0，獨立 `--evaluate` 才可讀 raw query y。原 August stage index／February index0、project／month 各2000次 paired 方法與 combined gates 沿原方法；不把 February 混進 development。預留 v32 不是已產檔，研究 driver 沒有 full fit、CSV、Kaggle 或參數搜尋。

私有 learned state、fold／OOF／逐筆因子、九 array、checkpoint／預測只保留本機，不能由研究打包廣泛複製至 GitHub。每 fold complement 的 estimator 在 helper 內建立；driver 保存該 fold evidence 與 full-fit state，不宣稱已逐份保存五套 complement 的完整 cells。

## CPU 檢查及待辦

作者正式收據 `outputs/research_continue_20261008/direction_review_a/R117_driver_author_selfcheck.json`，SHA256 `de1e4fe250fb90ed5f12e952fe02046210540cdaab211a6d3a7065ccd0d27a2b`；人工65項，實測命令2.297秒，零真 prior build／array prepare／NN fit／GPU／query 價格。覆蓋固定 fold／RNG、八 array 不變、ratio 逆變換、初始化連動、交易 fold／每 complement evidence、原模型 metadata、R115 拒絕、四起點完整封存與失敗訓練拒絕評分、combined gate 布林矩陣。沒有執行實資料 `specifications()`／context 或價格中位數建立；這些留給主線正式 metadata/context 審查及獨立完整 preflight。

正式 flags／預期 execution 檔為：

| 工具 | flags／環境 | 收據 |
| --- | --- | --- |
| prior helper | `.venv`、`-B -X utf8 ... --self-check` | `R117_prior_selfcheck_execution.json` |
| trainer | `.venv_torch`、`-B -X utf8 ... --cpu-unit-check` | `R117_trainer_selfcheck_execution.json` |
| driver | `.venv`、`-B -X utf8 ... --self-check` | `R117_driver_selfcheck_execution.json` |

三份均在 `outputs/research_round117_20261008/`，`command` 必須逐項等於 driver 定義，`source_sha256` 為該 entry 的 AST closure。完成後才能做 `specifications()` 純 metadata 庫存與四 context 獨立重播，並寫 `preflight_review.json`；作者沒有生成通過的實資料審查。

最強反證與科學限制以固定 plan 為準：交易 fold 可保留同案／同批 peers，可能只降低 fit 誤差、傷害新案；它不是時間 OOF、獨立樣本或逐筆 as-of。四起點重疊、已反覆使用，bootstrap 不含重訓或累積選型不確定性。Private未知；v22原選定與簡報保持。root統一後續正式檢查、GPU、交接更新與審查，剩餘≤20%停止新增工作。
