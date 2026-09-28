# R92 helper 完成與有限歷史查核

日期：2026-09-29（Asia/Taipei）。作者：子智能體 /root/r85_validation_audit。本文件記錄新 helper 與純合成檢查，不是候選回測結果，也不是作者自己的獨立審查。

## 有限歷史範圍

對 src/round*.py、reports/round*.md 與 reports/r*.md 搜尋 depth 6／depth6／depth=6 等固定字串，並讀相關 helper／report。未找到已執行的「R79 日期×面積權重、單價 target、原95欄／23類別、depth6、2999樹」同配方候選；這是有限已保存文字查核，不聲稱沒有任何未保存或範圍外實驗。

找到的 [R15](round15_report.md) 是 raw y/1e6、均一列權重、depth6；與此輪單價表示不同。[R20 helper](../src/round20_slow_geo.py) 實為 raw、depth8，改5998樹和learning_rate0.0325，其 depth6 出現於拒絕錯誤模型的合成測例，不是 depth6 正式候選。其他大量 depth6 命中同樣是既有 checker 的拒絕測例。[R79 helper](../src/round79_recency_unit.py) 固定depth8；[R92固定計畫](round92_plan.md)只指定depth8→6，不掃其他depth。

## 已實作 API 與身分

[新 helper](../src/round92_shallow_unit.py) SHA256：
21bbf98551de72aa3d03da367f8b46abd8b39746bbdfafe5e081fcb03786fd07。

- fit_model(X, y, geo, cats, saved_date_weights) → native model、獨立R92 wrapper、metadata。
- predict_model(model, state, X, geo, cats) → 非負float64 TWD；不接query y或日期權重。
- transform_target(y,X)、inverse_transform(z,X)、augment_features(X,geo)直接沿用原R19規則。
- sample_weights(X,w32)精確保留(A/100)×float64(saved_date_float32)，不重新估權或正規化。
- parameters()深複製原R79設定，鍵集合一致且差異键精確只有depth；其值8→6。
- wrapper包含legacy_feature_state，值與原R19 fit_feature_state完全相同；原depth8和area-only歷史欄位只作來源。新外層有R92 component、獨立parameters/weight/source身分、95欄／23類別 schema、實際fit target與weights hashes。
- check_model自己檢查native depth6／SymmetricTree／Plain、2999樹、每棵1..64葉、完整95names與23cat位置、seed、loss/eval、bootstrap、CTR、border與lr/L2/subsample/random_strength。未修改或呼叫原強制depth8的model checker，亦未monkeypatch真模型。

參數只有depth改動，沒有新增prior、offset、target轉換、特徵、校正或分支融合。source_hashes保留原R19/R6/prepare_data/R79來源與新helper SHA。日期權重的檔案、列序、日期與來源身分由driver驗，不在helper宣称已證明。

## 真實合成檢查

[自測收據](round92_helper_selfcheck.json) SHA256：
590261e49d90a245bf07135a4f2ba9a248033ffbdff6c5979e39a8be2297b7ca。

首輪31項passed、exit0、helper內部0.187秒；0真fit、0資料／真模型／configuration載入。只有1次FakeModel fit，CLI僅在self_check範圍將本helper的constructor／Pool換成假物件，並把讀config的函數換成固定合成字典，finally恢復；不改source或任何真模型。

重点覆盖实际FakePool unit labels、area×date权重、constructor只有depth差、原95值/dtype/order/index／23cat、legacy feature state完全一致、新checker拒depth8／>64葉／錯categories與其他參數、JSON wrapper＋FakeModel重載推論相同、query API無label及輸入未污染。另保留少量既有target/date對齊防呆檢查。沒有真CBM保存重載；這由主線三期driver逐query驗證。

檢查完後再核AST、收據source SHA及driver所需categorical_columns、legacy_feature_state、feature_count95/categorical_feature_count23、changed_parameter_names=['depth']與inverse/sample_weights API吻合。凍結後本代理未修改source，未啟動GPU。

## 接續與限制

主線應在R91全部結束後串行執行R92，保存真fit、native checks、完整query重載與三期控制／分群／paired結果。固定seed的GPU不是完全決定性；降低深度可能欠擬合，不能因較簡單就先稱改善泛化。反覆年度窗口不是新盲測，Private仍未知，v22唯一提交不變。
