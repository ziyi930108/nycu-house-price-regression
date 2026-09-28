# Attention 執行期間：下一個回歸假設的有限盤點

2026-09-29。僅讀交接、已完成 R85–R93 報告及相關 source；沒有 fit、predict、載入逐列 query y／預測或網路查價。接獲任務時主線回報 R95 第一期完整 MAE 1,422,360.94（v22 1,407,387.07），三期仍執行；這是當時進度，**不是 Attention 的最終判定**。

**結論：沒有足夠證據預期下一個模型一定改善。若仍要固定一個低成本、可否證的機制對照，優先考慮「降低局部 reference 信任強度、保留時間趨勢」，只做一次；不建議再排列新權重／深度／先驗組合。** 這是 R82／R84 已提出但未實測的延伸，不冒稱全新發現。

| 已完成方向 | 最重要的約束與反證 |
| --- | --- |
| [R85](round85_report.md)／[R87](round87_report.md)：面積表示／等權混合 | 原生或多數普通價帶改善仍可能在完整 median 轉差；R87 最新億元群抵銷低價帶收益。 |
| [R86](round86_report.md)／[R91](round91_report.md)：先驗欄／加法殘差 | 最新小幅改善，另兩期退；保留中高價帶收益，不能說先驗一概無效。 |
| [R88](round88_report.md)／[R89](round89_report.md)：撤 fallback 趨勢／群時間斜率 | 新案三期皆退；「修正更保守」不自動更穩。R89 中間整體仍有改善。 |
| [R92](round92_report.md)／[R93](round93_report.md)：depth6／建案密度權重 | 均未通固定門檻；最舊有利、R93 seen 及5千萬–1億三期有利，不能只列失敗。 |

**唯一保留假設與固定介面。** 舊 reference 的信任 c 是來源層級、距離、MAD 的工程公式，不是經校準的正確機率。較強 native 可能仍被過度拉向局部估值；這只是待驗機理，[R84 層間差](reference_blending_audit_r84.md)與 R85 最新 base 改善而 R/S 轉差不能證明因果。保留 v22 三支各自的 B、原 fit-only state、beta、鄰居與 clip；對每支固定 rho=0.5：

```text
Bf = 原 future_base（原 beta 與 366 日上限）
P_new = Bf + 0.5 * (P_original_reference - Bf)
c_new = 0.5 * c_original
U_new = 原 spatial(B, P_new, diagnostics[source 不變, confidence=c_new])
final_new = median(U_new_NN, U_new_log, U_new_unit)
```

因此 spatial 殘差份額由 `(1-c)*alpha` 改成 `(1-0.5*c)*alpha`，**不是只改 reference 價格而 spatial 有效強度不變**。c=0 的列保持原 reference，後續 S 也应逐位保持；不同於 R88 撤趨勢、R89 重估係數／殘差，也不改 unit 訓練權重或 NN 架構。0.5 是預先選定的機制消融強度，沒有最佳性證據。

**最強反證。** v22 的局部參考本來明顯降低單價 base MAE，減弱可能丟掉可靠同案／同路訊息；spatial 剩餘份額增加也可能放大粗地理誤差。全域趨勢仍保留，故此法不解決無支持廠辦的趨勢風險；面積外推與共同尾部偏差仍在。[既有誤差盤點](r85_error_audit.md)顯示100–200m²占62–68%總絕對誤差，億元群僅約3–5%，不能只以極端例子辯護。

**最少驗證與停止。** 若主線採用，先另凍結 plan／新 wrapper，不改舊 helper；rho=1 精確重播、c=0／fallback 不變、三支各用自己的 base、診斷不污染舊 state、合成算式與重載必過。再一次完成三期 CPU R/S＋median，0次新 native fit；對照 v22／v24，保留逐層、全部分群及24區間。沿既定三期整體／固定加權全改善、最新≥0.5%門檻，失敗不改 rho、挑 branch／城市或價格帶重試。R88 同類 CPU main 曾實測57.734秒，僅作成本參照；此候選尚未量測。bootstrap 不校正反覆選型，也不是 Private 勝率。

**不湊第二候選。** R18 面積申報狀態確有歷史支持（未 canonical 去重 train 中有車位卻面積0共9,735筆），但原95欄已有組成；[R30](round30_report.md)每車位面積在舊 raw 路線開發整體退1.49%、新案退2.05%，不是新 unit 狀態欄的直接否證，卻不足支持立即重訓。現階段不因還有未编码文字就新增備註／猜屋齡。

歷史範圍：[R82 後續審查](round82_followup_audit.md)、[R84 接續計畫](continuation_plan_after_r84.md)、[R85 方向](r85_direction_audit.md)、[R93 前盤點](next_regression_direction_after_r92.md)、[R18 純X審查](round18_feature_audit.md)。本次搜尋 src 的 confidence／neighborhood_base_weights／縮减詞及保存 plans/reports，未見上述完整三支固定政策已執行；這是有限查核，非全歷史不存在的證明。此文件只提案，不啟動後續實驗；v22唯一提交與未知Private不變。
