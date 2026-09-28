# R98：只用較早 forward 預測學習三分支 L1 組合

日期：2026-09-29（Asia/Taipei）。本計畫於任何真實 meta-fit／新 query 評分前固定；solver 工程數值規約以同樣 fit 前凍結的 helper parameters 與 launch request 綁定，不能看 query 結果換算法或門檻。

## 唯一假設與方法

固定 v22 的六個 native、原特徵、各分支 reference/spatial 不變，只將最後逐列 median(N,H,U) 改成全域 convex 線性組合 w_N*N+w_H*H+w_U*U。N、H、U 均是各自 cutoff 前資料訓練的完整修正後總價預測，不能用同一窗 in-sample prediction 或較晚模型回推。

三個權重非負、和為1、無截距；按 meta-fit 每一列等權最小化 TWD MAE（L1）。不加分群、正則化、時間衰減、額外 median shrink、候選網格或 Public 調參。除以共同1e6僅為求解數值尺度，還原後公式不變；所有列保留，不按價格或殘差截尾／刪樣本。只有這一套算法，各窗按當時可見 meta-fit 分別學自己的權重，不挑一窗係數套另一窗以求最好。

## 三個較晚評估與時間隔離

|query stage|query起訖（含）|可以學權重的完整 forward block|
|---|---|---|
|development|2022-08-11～2023-08-10|auxiliary：2021-08-11～2022-08-10|
|offset|2023-02-11～2024-02-10|auxiliary全段＋development中日期嚴格早於2023-02-11的前綴|
|confirmation|2023-08-11～2024-08-10|auxiliary＋development全段|

query列數依序81,234／107,169／133,697；meta-fit為105,511／待純X日期核出的固定前綴數／186,745。auxiliary本身缺更早forward block，這次不假裝它已被評估。offset和另外兩窗重疊，不能算三個獨立盲測，也不能取代原三個8月窗口的完整採用條件。

以保存canonical index與日期核對：meta與本stage query不重疊，所有meta日期<cutoff；每個meta預測的原生fit只早於其forward起點，所有learned state同origin。舊fit/query索引與R97的raw-cache mapping分清，不能只憑同長度配對。依日期決定offset前綴，不按殘差選擇。

時間前推允許較早query標籤成為更晚stage的meta-fit。為隔離實作，每個stage用獨立程序，只載入自己meta-y檔；在該程序中沒有本stage query-y。isolated preparer核原始pickle SHA後擷取並分別封存meta標籤；pickle仍會在擷取器反序列化整張原表，不能聲稱未觸及整表。preparer不計query統計、不輸出本stage query-y給driver。

全部三窗係數／完整query預測／v22控制重播封存成功後，獨立scorer才讀query標籤評分。三個stage都跑完，不因第一窗結果停止或改配置；沒有query early stopping。開發者先前已看過歷史分數，本輪仍有反覆選型偏差，不因程序隔離稱盲測。

## 控制、保存與評估

主要控制是同窗原v22 median，必須逐位重播其保存向量；另保留N/H/U本身，不混稱單模與完整模型。保存三個L1 model state、權重／來源／solver optimality診斷、meta目標、所有query預測及保存重載精確檢查；0 native fit／0 native model load，CPU meta-fit最多3次。

scorer固定報TWD overall與既有mix weighted MAE，權重seen=0.36617593949822613、new_nonmissing=0.6338240605017739、missing=0。新舊案定義按本query原模型fit群，不能改成meta-fit歷史案。所有城市、月份、相對季度、五價帶、尾端集中度與正反分群都保存，不只平均值。真价帶只作診斷，禁止用在predict分流。

每stage都做paired project與三個月循環month bootstrap各2000次，seed20260925，固定stage index development=0、offset=1、confirmation=2，family index project=0/month=1。保存12個區間與六份replicates；不挑有利抽法。與舊R97重抽使用不同stage index明記，不能混當同一收據。區間不包含meta／native重新訓練、累積選型或未來分布移轉，不是Private勝率。

## 事前固定研究判讀

只有三個較晚窗口的overall與weighted全部低於同窗v22，而且每窗沒有任何季度或至少500列分群MAE增加超過5%，才稱值得升級研究。bootstrap的全部結果另外揭露；跨零不能稱確定改善。無論通過與否，此輪不自動採用、不全量fit、不產新CSV、不替換使用者唯一提交v22。

即使探索全部通過，仍須另固定更早origin、確認當時可用非價格地理，重建至少六native及全部learned states，补出aux之前forward meta-fit，才能補原三期完整比較。這不是藉較晚三窗放寬原採用門檻。若不通過，保留全部結果，不用本次query搜尋約束、比例或重新裁樣本。

## 約束與執行

只有課程價格標籤，禁止查內政部成交答案；無test／valid使用與Public調參。原CSV、模型、selector、正式PPTX／PDF不變。所有來源、input／output SHA、合成驗證、失敗與實際時間保留新目錄；舊source和SHA guard不修改，不自動覆寫／重試。先核現有程序，root統籌，新研究source/API前置審查通過且launch frozen才真fit。完成後更新HANDOFF／LOG／程式快照；3個小型stacker不是完整從零重訓回歸pipeline。
