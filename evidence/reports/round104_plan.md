# R104：以合法較早教師預測訓練固定base組合目標

2026-09-29。R103三窗診斷完成後、任何R104教師／學生真資料準備與fit前固定。本輪是適應性研究，唯一提交仍v22，Private未知；不是新的盲測。

## 為什麼改訓練目標

R103確認最近期R102T原生項可改善23,837.92元，但固定base的誤差抵銷損失80,898.56元，合計退57,060.64元。development則在median把原spatial收益截住。這支持檢查NN是否該直接學其固定組合中的角色，但不保證能解決median問題。R98/99的較早全域L1後置權重仍有兩窗退步，是本輪最強反證之一；本輪不再搜尋最後權重。

## 唯一學生差異與匹配控制

學生沿各原8月outer cutoff、原R102T的145欄／categorical／factor與日期權重，僅用能取得合法forward教師的fit列。兩臂同列、同初始化、同seed42／30epochs／batch2048、AdamW及原架構／schedule，不earlystop或追加seed。

令p=ratio*train_price_factor_million，y為million TWD，w為保存日期權重，A/L為較早origin的area NN／unit LGB raw native總價（轉million）：

- C：`.375*mean(w*abs(p-y))`，梯度尺度匹配的native目標控制。
- T：`mean(w*abs(.375*p + .375*A + .25*L - y))`，固定base目標。

`.375/.375/.25`沿原base，不搜尋。兩臂bias均用相同子集y/F的中位數及原inverseSoftplus；不能為T以residual重新初始化。仍保留Softplus正價ratio。不把可能負的等價residual target clip為正、不刪此類列，只直接算上式。C的常數倍率雖不改理論L1最小值，也可能改數值訓練軌跡，故C不是原R102C/T；同時保存原v22/R102T作固定背景。

新trainer用精確10-key NPZ：原R63九鍵加`train_base_offset_million`（float32，由保存float64教師總價按`(.375*A_TWD+.25*L_TWD)/1e6`運算後轉float32）。兩臂同offset資料，C不使用offset進loss。其餘九鍵除fit子集外不另變形；query數值／categorical／factor保持R102T原bytes。保留原w，不對子集或batch重正規化。loss按原float32Torch訓練、query還原與base合成按既有float64順序，兩者數值精度差明載。

## 教師與價格時間隔離

新增一個`pre_auxiliary`教師origin 2020-08-11：只用更早canonical train fit，forward query 2020-08-11～2021-08-10。重新建立area NN encoder／quantile ramp、unit LGB類別狀態，原固定R48 area trainer（30epoch）與固定LGB 3999樹參數。只2個新native fit，不建立priorNN、H/U、reference或spatial。

另重用已封存auxiliary origin2021-08-11及development origin2022-08-11的raw area/LGB forward預測，保存source/seal/索引身份。原 R102 fixed_branches[area/lgb] 為相同原生來源，不用R/S後N或median作教師。

|學生窗口|教師forward資料|預測query日期|
|---|---|---|
|auxiliary|pre_auxiliary|2021-08-11～2022-08-10|
|development|pre_auxiliary＋auxiliary|2022-08-11～2023-08-10|
|confirmation|pre_auxiliary＋auxiliary＋development|2023-08-11～2024-08-10|

各teacher fit截止早於被預測列交易日，教師forward日期全小於該學生cutoff；每個學生canonical索引先與原R102 eligible fit列取交集，去重／遞增排序，僅按日期與索引決定，不能按殘差選樣。兩臂同子集，實際筆數由固定規則產生再保存，不臆造筆數。教師meta origin較早、query用本outer origin的原area/LGB，兩者fit規模／誤差分布可能不同，不能稱已匹配分布。

原R102145編碼／標準化與fit temporal prior可包含較晚但仍outer cutoff前的資料；沿用合法outer-fit狀態，不能稱每個學生fit-row的所有輸入都完全as-of。學生loss標籤直接取自身prepared子集，不另讀整張raw價格表；新增pre-aux教師fit y由隔離extractor依canonical→raw映射抽出。extractor反序列化整張raw pickle，僅保存早期fit標籤，不能宣稱完全未觸及query bytes。

## 非價格地理的明確限制

使用原有已證明不讀成交價格的2023 OSM靜態地理作相同特徵來源，按原R6 canonical映射取列；area NN仍需要原8欄地理，unit LGB只用原X。2023快照晚於2020／2021／2022 origin，含未來道路存在性／覆蓋的歷史可用性偏差。

因此本輪只稱**價格標籤與learned狀態按時間隔離的forward研究**，不稱完全historically-as-of回測，也不冒稱補完R98原先要求的全歷史可用地理。這是事前固定的限制，不在結果後隱藏。新thin wrapper保留來源／snapshot身份，不改R90原`snapshot < origin` guard，也不調用其不適用的高階loader。禁止內政部成交答案，沒有新增外部價格來源。

## 執行、完整預測與評估

主線單GPU协调：先完成2個早期teacher，封存／重載全forward預測；再按development、confirmation、auxiliary各C後T，最多6學生NN，總新native fit最多8（7NN＋1LGB）。各學生獨立程序只讀自己的10鍵prepared與同窗query特徵，不讀本窗query y／他窗NPZ。

query仍用原同窗 `B=.75*(.5*A+.5*P)+.25*L`，原R/S各一次，再median(N,H,U)，不改權重／修正或H/U。三窗六學生全部封存及保存模型全query重載成功後，隔離scorer才讀本輪query真價。保存native/base/reference/spatial/median、teacher raw、模型、encoder、loss曲線、來源／索引／SHA與真實時間。失敗不自動重試或覆寫；工程修復另存紀錄，不能根據query分數改計畫。

主比較T−v22與機制T−新C，所有原三窗overall／固定seen-new weighted、城市／月份／day-quarter／五真價帶／尾端全報。兩比較各stage×project/month×overall/weighted，2000配對replicates，seed20260925、stage dev0/conf1/aux2、family project0/month1，共24區間；不是Private勝率。原R102T與新C−v22作點差背景，不追加按結果挑選的bootstrap。

Primary保持：三窗完整overall／weighted均優於v22、confirmation至少改善0.5%、所有季度或至少500筆固定群無>5%退步。Mechanism：T相對同子集新C的base與完整median三窗overall／weighted均改善；本目標不要求原生NN自身MAE較好，但原生全部指標必須揭露。兩門檻分開，不能用base改善取代完整模型成績；任何通過亦不自動全量fit／替換v22／上傳Kaggle。

先做人工loss/gradient、資料映射、no-query-label與保存重載契約檢查，再凍結source與各實際input request才真執行。保留反覆使用窗口、教師／query分布移轉、單seed及靜態2023地理限制；source可追溯不等於乾淨環境從零重訓完成。正式v22報告／簡報不改。
