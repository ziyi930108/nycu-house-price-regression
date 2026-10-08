# 有界方法盤點：訓練內 OOF 融合

2026-10-08。相關報告9份（8全文、1節錄），只讀歷史方法及聚合JSON，無新fit／GPU／評分／逐筆價格／CSV。逐檔SHA、readset、舊六次fit身份與確切source行見同名JSON。

**真正fit-only時間forward-OOF的meta回歸已做；本次有界readset未找到同等嚴格的X條件權重學習。** 不建議再做全域比例搜尋，最多保留一個需另立可行性規約的方向。

| 方法 | 已做與界線 | 最強反證 |
| --- | --- | --- |
| R2 | 同2024 holdout按建案五折選粗比例及seen/new分流；native曾用該holdout early stopping，source明記非untouched test。 | 新池分群加權MAE 1,788,626，比全域1,740,945差；不能當嚴格時間OOF gate等價試驗。 |
| R98 | 更早完整forward N/H/U及合法過去答案學非負、和為1、無截距全域L1權重；3次真meta fit。 | development／offset／confirmation對v22 MAE差+8,798.64／−4,255.03／+5,190.67元。 |
| R99 | 再加入M=median(N,H,U)，仍全域一組4權重；3次真meta fit。 | meta目標更好，query仍兩窗退步；median-only已是可行解。 |
| 固定R/S與prior | R/S為fit支持程度固定規則；prior五折按建案，非完整模型時間OOF。 | 不能當已學習的條件meta回歸。 |

R98／R99 source嚴查meta IDs屬當窗fit、與同窗query無交集、meta日期早於cutoff。較早query在較晚origin可成合法fit標籤，不可回流同窗。證據：`src/round98_temporal_stack.py:50–72`、`src/round99_temporal_stack.py:103–130`、`src/round98_prepare_inputs.py:29–42,85–105`；R2 caveat在`src/round2_blend.py:125`。

R99最有力的反證：meta MAE比median改善25,912.38／18,033.98／14,836.22元，但query差為 **+10,449.82／−6,281.53／+2,821.52元**。offset嘉義縣退步6.080%；confirmation億元群319列對全窗不利貢獻9,615.26元，抵銷其他四帶合計改善6,793.74元。這是誤差位置，不是因果證明，不能推論時依未知真價分流。見`reports/round99_summary.md:20–34,45,55–61`。

唯一保留的未執行假說：**低容量、fit-only forward-OOF條件凸gate，保留原median作精確可行對照。** 可能理由是分支可靠度隨面積、其OOF生成前綴的新舊案支持與分歧而變；全域權重不能表達。只容許少量事前限定pure-X、合法前綴支持、OOF分歧；不能使用真價帶、query誤差、test ID或Public。這尚未驗證，增加容量也可能放大R99的轉移失敗。

目前不建議直接fit。最早auxiliary 2021-08-11之前仍缺完整forward block；四origin門檻至少需補4樹＋2NN及當時encoder／prior／reference／spatial，並核地理資料在該前綴可用。seen/support須依每列OOF生成前綴建立，不能拿完整outer-fit狀態。已有後期vectors可按hash重用，但不能聲稱完整四origin零native重訓。來源`reports/regression_direction_after_r97_inventory.md:24–38`；本次未選cutoff或估工時。

若另立規約，先固定唯一低容量形式、TWD-MAE、正則、小型特徵清單、完整OOF覆蓋和全部對照，再來源凍結與獨立預審。不是gate／seed／深度參數掃描。已有窗口反覆使用且重疊；bootstrap不含meta重fit／native重訓／選型不確定性，不能稱Private勝率。

本盤點只推薦先做單一可行性／覆蓋規約，**不推薦現在啟動新的融合訓練**。v22及簡報保留，Private未知。共享交接由root更新，本代理僅新增此目錄證據。
