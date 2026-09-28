# R95：欄位Attention prior-ratio回歸

固定數值門檻未過，保留研究，不全量fit／新增CSV。 使用者選定仍為v22，Private未知。R95沿R94同一配置，原R94於真fit前因歷史JSON讀取契約錯誤停止；原失敗保存，新目錄只修工程讀取方式。

## 動機與做法

測試同一筆交易內的欄位交互，固定替換prior NN；不是跨成交資料的attention。沿同窗R63九個prepared成員、144數值／23類別、group-OOF fit prior與full-fit query prior及日期加權總價MAE。168tokens、d64、2blocks／4heads，最後CLS query/residual/FFN裁切，head Softplus ratio。固定30epochs、seed42、AdamW、micro256累積2048，不以query挑epoch或搜尋架構。

只把原NN分支變成 .75*(.5*R54area+.5*R95prior)+.25*LGB，再原reference/spatial，與保存R46log／R79unit取median。舊R71與v22整鏈重播，來源與保存模型重載依主流程收據核對。原始上游Apache-2.0 source/revision/license保留，末層計算裁切、Softplus頭、損失及optimizer都是本題改編，非完整論文復刻。

## 三期結果

單位TWD MAE；差額=R95−v22，負數較好。保留三期，不只展示有利窗口。

|期別|v22 MAE|R95 MAE|整體差額|固定加權差額|
|---|---:|---:|---:|---:|
|中間年度|1,407,387.07|1,422,360.94|+14,973.87|+14,733.96|
|最新年度|1,383,228.41|1,412,063.62|+28,835.21|+32,836.37|
|最舊年度|2,309,352.36|2,277,199.48|-32,152.87|-26,139.25|

最新相對改善-2.0846%；三期整體全改善=False、三期加權全改善=False、最新至少0.5%=False。

|期別|既有建案ΔMAE|新建案ΔMAE|
|---|---:|---:|
|中間年度|-1,414.13|+24,063.12|
|最新年度|-6,386.77|+55,496.55|
|最舊年度|-9,061.21|-36,005.66|

|期別|真價帶|筆數|群ΔMAE|
|---|---|---:|---:|
|中間年度|<10m|22,139|+4,836.07|
|中間年度|10-20m|43,914|+17,491.31|
|中間年度|20-50m|14,197|+14,298.55|
|中間年度|50-100m|802|+7,791.19|
|中間年度|>=100m|182|+725,071.15|
|最新年度|<10m|25,585|-3,432.65|
|最新年度|10-20m|74,109|+8,010.48|
|最新年度|20-50m|31,959|+78,886.08|
|最新年度|50-100m|1,725|+195,937.97|
|最新年度|>=100m|319|+1,536,815.20|
|最舊年度|<10m|41,132|-22,321.16|
|最舊年度|10-20m|47,793|-17,114.02|
|最舊年度|20-50m|15,033|-14,368.17|
|最舊年度|50-100m|1,382|-120,079.98|
|最舊年度|>=100m|171|-7,453,161.66|

真價分群僅事後診斷，未知test真價不能用作推論分流。所有季度／城市／testmix保留於完整diagnostics；testmix是歷史支持群重新加權，非Private。

## 原生分支與完整集成

|期別|層|MAE|
|---|---|---:|
|中間年度|base|1,553,910.96|
|中間年度|native|2,240,416.14|
|中間年度|reference|1,568,853.79|
|中間年度|spatial|1,558,782.81|
|最新年度|base|1,854,245.28|
|最新年度|native|2,309,053.86|
|最新年度|reference|1,557,488.83|
|最新年度|spatial|1,543,035.13|
|最舊年度|base|2,466,110.23|
|最舊年度|native|2,863,476.90|
|最舊年度|reference|2,374,459.34|
|最舊年度|spatial|2,377,071.15|

分支改善不保證median改善，層差也不能相加解釋因果。詳看結果中的完整分支與保存控制指標，不能只用native勝負取代完整組合。

## 全部配對區間

|對照|窗口|抽法|指標|點差額|95%區間|
|---|---|---|---|---:|---|
|v22|development|project|overall|+14,973.87|[+5,141.53, +24,829.48]|
|v22|development|project|weighted|+14,733.96|[+5,097.86, +24,077.42]|
|v22|development|month|overall|+14,973.87|[+10,223.08, +19,651.72]|
|v22|development|month|weighted|+14,733.96|[+11,849.15, +17,262.90]|
|v22|confirmation|project|overall|+28,835.21|[+16,657.53, +42,819.45]|
|v22|confirmation|project|weighted|+32,836.37|[+19,424.10, +47,615.29]|
|v22|confirmation|month|overall|+28,835.21|[+17,230.46, +41,425.96]|
|v22|confirmation|month|weighted|+32,836.37|[+24,246.32, +39,971.57]|
|v22|auxiliary|project|overall|-32,152.87|[-59,210.33, -12,572.66]|
|v22|auxiliary|project|weighted|-26,139.25|[-46,611.49, -10,951.22]|
|v22|auxiliary|month|overall|-32,152.87|[-58,230.26, -16,083.30]|
|v22|auxiliary|month|weighted|-26,139.25|[-43,209.78, -14,064.46]|
|v24|development|project|overall|+15,020.22|[+2,233.09, +27,907.08]|
|v24|development|project|weighted|+14,755.78|[+2,191.07, +27,329.47]|
|v24|development|month|overall|+15,020.22|[+3,465.92, +23,569.92]|
|v24|development|month|weighted|+14,755.78|[+4,449.63, +20,235.04]|
|v24|confirmation|project|overall|+38,867.70|[+20,526.61, +56,540.57]|
|v24|confirmation|project|weighted|+43,900.90|[+23,673.08, +63,768.22]|
|v24|confirmation|month|overall|+38,867.70|[+19,252.01, +54,787.80]|
|v24|confirmation|month|weighted|+43,900.90|[+32,345.94, +51,580.28]|
|v24|auxiliary|project|overall|+1,790.34|[-30,775.23, +32,841.02]|
|v24|auxiliary|project|weighted|+4,455.44|[-21,387.00, +28,552.99]|
|v24|auxiliary|month|overall|+1,790.34|[-16,900.36, +13,973.62]|
|v24|auxiliary|month|weighted|+4,455.44|[-5,903.23, +13,076.63]|

對v22負／跨零／正為4／0／8；對v24為0／4／8。這些區間相依、未校正多輪選型，不能称Private勝率或未來市場保證。窗口反覆重用，不是盲測。

## 時間與限制

三次真NN fit共1762.376秒，主流程1903.922秒包含fit；合成資源檢查、資料與診斷另計，不能重複加總。未測量整段聊天開發時間。GPU固定seed不保證決定性，初始化/dropout使shuffle亂數不與原MLP逐位相同；舊對照未同時重訓。

原編碼、prior與日期權重有保存來源核對；trainer不接受query標籤，主driver仍載canonical y而只用query部分score，不誇稱query bytes从未載入。embedding寬度、dropout位置與模型結構一起改，不能將結果全部歸因attention。保存重載並非全新ML環境從原始資料端到端重訓。

## 證據入口

- [固定計畫](round95_plan.md)、[上游來源](round94_upstream_review.md)、[整合契約](round94_integration_inventory.md)
- [CPU合成](round94_trainer_selfcheck_v2.json)、[GPU合成成本](../outputs/round94_smoke_v2_20260929/smoke.json)、[獨立preflight](r95_validation_audit.md)
- [完整結果](../outputs/round95/results.json)、[完整分群](round95_diagnostics.md)、[決策](../outputs/round95/decision_review.json)
- [對v22配對](round95_paired_vs_v22.md)、[對v24配對](round95_paired_vs_v24.md)

不改原CSV、selector或v22簡報。本輪没有test預測、Public/Private新分數或Kaggle操作。
