# R94 訓練前工程修正紀錄

2026-09-29。新架構未進行真fit前，保留v1 source／CPU／GPU合成結果，另存v2 source及收據。CONFIG、plan、架構、損失、optimizer與資料切分都未改；這不是看回測分數調參。

v1在NPZ materialize所有成員後才檢查九成員契約，標準R63輸入正確，但錯誤額外成員也可能先被反序列化。v2新增load_prepared：先驗NPZ目錄的成員集合，再只讀固定九成員。合成加入object型額外query_y測例，確認回報的是成員契約錯誤，未走到object反序列化；正確九成員NPZ完整roundtrip逐位相同。這是合成假資料，未讀真query價。

- 原source：[round94_attention_prior_train.py](../src/round94_attention_prior_train.py)，SHA 1f2a4140bdb9a439e098273b442fb524bef9d3fee70af8618f7cb997355ace94，保留。
- 實際待用source：[round94_attention_prior_train_v2.py](../src/round94_attention_prior_train_v2.py)，SHA ae197952be777f3e9e2fdb66458887d2a3885fd736b75dd3677eeae43cfeef83。
- [工程差異](../outputs/round94_launcher_20260929/trainer_v2_engineering_diff.patch)、[差異收據](../outputs/round94_launcher_20260929/trainer_v2_engineering.json)。
- [v2 CPU](round94_trainer_selfcheck_v2.json)：47項通過、1.250秒、0 GPU、0真fit；原45項加2項NPZ檢查。
- [v2 GPU合成](../outputs/round94_smoke_v2_20260929/smoke.json)：2暖身＋3量測group、micro256累積2048，用相同train_group及finite／gradient同步檢查。平均0.182333秒／group，10260組估三期fit 1870.740秒（約31.18分）；估算不含資料、重載或診斷，不是實際真fit時間。
- 峰值allocated 565207040 bytes、reserved 746586112 bytes。合成沒有全部真實prepared資料常駐，不能當真fit完整峰值。
- v1 CPU第一次float64測例因累積器原float32而失敗，attempt1保留；v1 prefit已改累積器沿numeric dtype，真正float32訓練設定保持。後續v1 passed与v1 GPU合成也都保留。

獨立審查亦發現driver先建model目錄會和trainer的拒覆寫守則衝突、雙方execution/log位置相衝；在真run前修成父程序收據與stdout放stage/trainer_process，model目錄完全由trainer建立。這是source preflight發現，沒有讓真fit失敗或覆寫模型。
