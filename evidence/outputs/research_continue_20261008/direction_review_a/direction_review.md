# R1–R110 歷史方向審查與 R112 helper

2026-10-08；主線最新停止線為Codex剩餘20%。本支只讀source、報告及聚合JSON，另寫新helper並跑CPU/Fake；0真fit／0GPU呼叫，不讀價格陣列／模型、不查成交答案、不操作Kaggle/GitHub。v22保留，Private未知。

先列round1–110報告標題，再精確詞檢索及深讀相關來源；不宣稱逐行深讀所有source，未保存實驗不在範圍。行號、SHA及命中表見direction_review.json。

|方向|是否重複／設計理由|最強反證|
|---|---|---|
|R111 prior-ratio殘差MLP|未見同獨立MLP真fit；R95 attention已有skip。保持R63九陣列／正ratio／日期總價MAE，固定skip架構可能改善優化訊號；容量、寬度、亂數也改，不能純歸因skip|R57三窗所有季度退步；R58加ramps仍未過；R95完整MAE中間+14,973.87／最新+28,835.21／最舊−32,152.87元|
|R112 unit/date×area Ordered|Ordered已做R17；本target×weight交互未見真fit。只改Plain→Ordered、保留95/23、maxCTR2、depth8、2999、seed42及完整設定|R17 raw/uniform完整MAE+6.0851%、新案+7.4351%、Q2+9.2054%；後兩窗未做。是強反證，不能說Ordered全盤證偽|
|signed additive prior-error NN頭（後備）|R91是CatBoost加法；R64是log-L1；R104仍ratio教師offset。以F+h(X)配原總價L1，輸出殘差梯度不再乘F；是代數推論，未證實prior梯度造成錯誤|R91完整MAE中間+47,037.46／最新−4,394.69／最舊+37,857.15元；億元群中間+1,369.60萬／最新+442.75萬元，不能稱大額解方|

R111/R112依主線reports/round111_112_plan.md固定；第三項只列後備，未立計畫、寫trainer或訓練。R111是內部skip，不能和價格offset混稱。R59 prior numeric ramps已真fit、unknown原有dropout訓練，不能重稱未做。

## 大額交易／先驗誤差

reports/r85_error_audit.md:13–25顯示各年整體偏差換號，不支持全體加價。100–200m²占絕對誤差62.24%／63.06%／67.64%，億元以上僅3.95%／5.02%／3.13%。億元群低估與factory/office及>500m²群高估可並存；少數極端值得監控，但不能期待只修幾筆帶來大幅整體收益。真價帶只作評分後診斷，test不依未知真價分流。

R64 logratio在最舊億元群進一步低估，目標壓縮不能自動稱抗過擬合；R85 sqrtarea金額等價L1已試過。加法NN後備若後續固定，需保留原144/23／F／日期／30epoch／seed，只換signed million-TWD head、median(y−F)初始化、F+h逆轉換；訓練不clip總價，query沿非負政策。這些是表示法固有聯動，不能說只改一個scalar。負值、先驗OOF/full mismatch與梯度不穩定都可能留存。

R60以30筆raw rows支持，不代表30個獨立建案；R101日期支持ESS=null。R67加support特徵、R93改全體loss密度權重都不同且跨年不穩。未固定唯一project-ESS收縮規則前，不加第四個模糊或掃threshold實驗。

## 平台依據與限制

[CatBoost官方設定](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type)把GPU超過50,000列的Plain列為預設，並非顯式Ordered禁止；Ordered支援CPU/GPU且可能較慢。[MAE官方表](https://catboost.ai/docs/en/concepts/loss-functions-regression#used-for-optimization)列GPU可優化。本機1.2.10零fit parser接受固定Ordered，R17曾fit240,316列；不保證新321,550／467,221列記憶體必足。OOM／非有限保留，不自動改參數重試。

[表格ResNet論文](https://arxiv.org/abs/2106.11959)支持它作有意義基準，也未宣稱通用優勝。本題固定適配非完整復刻，保留過往架構反證。

## R112 helper交接

src/round112_ordered_unit.py SHA c8b6b64577b57c3b5c4e5ff8bde0accfe72866d1e5b6d4338746dd6ebaff3b73。

- API fit_predict(X,y,G,Xq,Gq,cats,weights32,out)，CLI只--self-check，拒覆寫新目錄。
- Constructor只Plain→Ordered，不傳target8；native implicit1。fit-only label、area×date Pool；query無label/weight，無eval_set／earlystop／R/S／CSV／額外cap。
- Native核Ordered/GPU、maxCTR2/depth8/2999及可恢復的全部傳入設定。allow_writing_files/thread_count/verbose屬constructor runtime，未假稱CBM恢復。Float rtol1e-7／atol0，其餘型別、欄位、結構精確，序列化差留證。
- CBM全query raw/TWD精確重載、原95/23／順序／資料與source前後SHA保持，沿原非負還原。
- 38 CPU/Fake checks passed；2 Fake fits包含異常reload拒絕，0真fit／GPU。零fit native parser亦過。收據round112_helper_selfcheck.json。作者自查不是獨立審查，主線審後才fit。

第一次報告產生因encoding名utf8-sig拼錯失敗，helper收據已先保存，無模型或共享檔變更；改為utf-8-sig後產出，失敗另留direction_build_attempt1_failure.json。

重用年度有選型偏差、bootstrap不是Private機率；本報告不稱新方法已有效。根交接／quota／GitHub由主線更新。
