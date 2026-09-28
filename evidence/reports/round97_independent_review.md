# R97 完成後獨立審查

七個原生模型與獨立評分均完成後才執行本稽核。來源／保存產物與固定公式通過；本輪不自動採用 v24，唯一選定提交維持 v22，Private 未知。

本次只讀雜湊、模型 metadata、保存向量、prepared NPZ headers／factors及已存指標；沒有反序列化 raw 價格表或 CBM/PT、沒有 query y、模型 forward、重新 fit、逐列 MAE或新 bootstrap。讀模型／價格來源檔位元組核 SHA 不等於載模型或擷取價格。全 query 模型／state 重載是核主流程已保存收據，不是本稽核再次推論。

## 來源與執行

實核 220 個唯一來源／產物 SHA，main 保護清單與另指定 9 檔全部原樣。凍結計畫、main request、evaluation request、純 X 與隔離 fit-label 身分一致。完整四 base／兩 median 逐位相同；N/H由同一保存分支共享，U24沒有混入LGB。

模型 fit 呼叫收據合计 1100.391 秒（樹 1044.938、NN 55.453）；main execution 1410.656 秒、evaluation execution 29.578 秒。部分樹 fit 呼叫含helper前後檢查，不是純GPU計時；內外層不可重複相加。

五樹保留原2999／LGB3999配置，兩NN均seed42／30epochs；各支全107,169query保存重載收據通過。新fit encoder／OOF／date／R/S身分核對，group OOF不是時間OOF；canonical fit index與raw fit-label mapping分工正確。N_area ramps寬度由新fit決定，N_prior固定144。未讀prepared fit target數值或大numeric／categorical矩陣。

## 完整模型與分群

差額是v24−v22，負數較好。固定加權仍為既定新舊案組成，不是Private分數。

|指標|v22|v24|差額|
|---|---:|---:|---:|
|overall_mae|1,336,410.67|1,340,328.14|+3,917.46|
|weighted_mae|1,392,890.34|1,397,452.28|+4,561.93|

|分群／季度|筆數|v24−v22 MAE|
|---|---:|---:|
|missing|0|—|
|new_nonmissing|59,352|+7,511.52|
|seen|47,817|-543.59|
|Q1|22,487|+1,367.59|
|Q2|27,820|+9,958.48|
|Q3|27,006|+7.62|
|Q4|29,856|+3,745.55|

|真價帶|筆數|v24−v22 MAE|
|---|---:|---:|
|<10m|25,432|+5,383.52|
|10-20m|59,481|+5,488.88|
|20-50m|20,803|-9,081.26|
|50-100m|1,183|-18,630.65|
|>=100m|270|+619,964.60|

|unit比較層|unit42／U22 MAE|unit137／U24 MAE|差額|
|---|---:|---:|---:|
|native|1,571,600.41|1,591,114.21|+19,513.80|
|base|1,539,355.97|1,562,489.74|+23,133.77|
|reference|1,411,009.29|1,465,060.89|+54,051.60|
|spatial|1,394,719.31|1,446,823.95|+52,104.65|

native行比較兩個不同seed單模；base起才比較原U22含LGB與U24兩seed均值，不能將這些差額當成只改seed的因果效果，更不能相加取代完整median。其他21向量的全部指標保存於scorer JSON。

同時整體／固定加權改善：False。事前定義的季度或至少500列群退步超過5%共 0 項，全部保留如下（有利群也保留於上表及完整評分）。

沒有超過該固定門檻的群；不等於所有群皆改善。

## 保存抽樣區間

|抽法|指標|點差额|95%區間|有效／無效|
|---|---|---:|---|---:|
|project|overall|+3,917.46|[-6,109.91, +15,316.77]|2000/0|
|project|weighted|+4,561.93|[-6,746.06, +16,849.02]|2000/0|
|month|overall|+3,917.46|[+488.17, +7,176.21]|2000/0|
|month|weighted|+4,561.93|[+903.73, +9,072.55]|2000/0|

全部4區間由保存的兩份(2000,4)replicates直接重算端點；未重新抽樣。固定seed20260925、stage index0、family0/1。缺正權重層的weighted抽樣不重抽。兩種區間相依，條件是保存模型與重複使用資料，不是Private勝率、盲測或已校正多年選型的檢定。

## 三個既有年度的上下文

以下直接讀原R79/R83已存完整MAE，沒有重跑舊模型。

|舊窗口|v24−v22整體|v24−v22加權|
|---|---:|---:|
|development|-46.35|-21.82|
|confirmation|-10,032.49|-11,064.53|
|auxiliary|-33,943.22|-30,594.69|

新窗口與既有年度重疊，配方也由既有研究選出；錯開起點可補充時間穩定性證據，不能消除累積選型偏差。原始test／valid沒有參與這個fit；本次不是乾淨環境從零重訓正式提交，也不宣稱Private結果。真價带與各模型自己的誤差尾部只能作事後診斷，不用於未知test分流。

[逐項收據](../outputs/round97/independent_review.json) · [完整評估](../outputs/round97/evaluation/results.json) · [main封存](../outputs/round97/sealed_manifest.json) · [原計畫](round97_plan.md) · [main前置](round97_preflight_review.md) · [scorer前置](round97_evaluator_preflight_review.md)
