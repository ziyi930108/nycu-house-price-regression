# 如何閱讀保存的研究證據

本目錄的 reports／outputs 保存來源文件原始 bytes、聚合結果與雜湊。價格向量、交易資料、模型權重及先驗狀態不在此快照內。這不是完整可直接重新訓練的工作區備份。

最新摘要見[Attention／R96完成研究](../docs/RESEARCH_UPDATES_20260929_ATTENTION.md)，原始來源見[research導覽](../research/README.md)。本輪[準備批次](outputs/research_snapshot_attention_20260929_preparation/copy_manifest.json)與[完成批次](outputs/research_snapshot_attention_20260929_final/copy_manifest.json)分開保存，已封存收據不改。

## 原始路徑的對照

為保留原碼與報告的 SHA，原報告內的工作區相對路徑不逐一改寫。從 reports 指向 ../src/NAME.py 的原連結，對應本儲存庫 research/src/NAME.py；個別原報告還引用未隨本次快照提供的歷史工作區產物。不要將「原工作區存在」或「有 SHA」誤認為檔案已隨 GitHub 上傳。

|本輪原報告引用的程式|本儲存庫實際位置|
|---|---|
|candidate_diagnostics_v3.py|[完整分群診斷](../research/src/candidate_diagnostics_v3.py)|
|paired_candidate_bootstrap.py|[分組配對分析](../research/src/paired_candidate_bootstrap.py)|
|round96_reference_shrink.py|[固定局部信任折半](../research/src/round96_reference_shrink.py)|
|round96_experiment.py|[R96 driver](../research/src/round96_experiment.py)|
|round96_review.py|[R96採用審查](../research/src/round96_review.py)|
|round94_attention_prior_train_v2.py|[Attention trainer v2](../research/src/round94_attention_prior_train_v2.py)|
|round83_seed_unit.py／round83_experiment.py|[固定單價模型](../research/src/round83_seed_unit.py)／[driver](../research/src/round83_experiment.py)|
|round2_neighborhood.py／round8_spatial_reference.py|[參考成交](../research/src/round2_neighborhood.py)／[空間修正](../research/src/round8_spatial_reference.py)|

完成批次的 link_verification.json 區分現行導覽的直接可用連結、原報告的來源映射與只在原工作區存在的歷史依賴；不把後兩者報成所有原連結都可直接點開。複製清單以來源／目的路徑與實際 SHA 為準。

舊報告中「未執行」「當時最佳」等字句是歷史狀態。現在模型選定與 Private 狀態以[交接入口](../HANDOFF.md)為準，未取得的 Private 分數始終未知。
