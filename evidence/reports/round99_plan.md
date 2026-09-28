# R99：保留逐筆中位數作可選基線的時間前置L1組合

日期2026-09-29（Asia/Taipei）。本計畫在任何R99真meta-fit或query評分前固定。R98已完整完成並發布77ec5b46a6b28ecef4cc2f64356083fa31c4ec30；本輪是看到其兩窗退步／一窗改善後提出的適應性候選，不是新盲測或獨立確證。

## 唯一改動與假設

保留v22六個原生模型、特徵、每支reference／spatial及各自合法fit日期；原完整三支總價N、H、U不重算。新增確定性第四欄M=median(N,H,U)，按每列三個未取整float64取中位數。只學全域四欄[N,H,U,M]非負、和為1、無截距的等列權TWD L1組合。

推論固定分項順序 `(((N*wN + H*wH) + U*wU) + M*wM)`，沒有clip／取整／調整權重。M权重1是可行解，舊三欄任一合法權重加上第四欄0亦可行。因此精確meta最優不應差於M或舊三欄解；**這不是query、固定群加權MAE或Private改善的保證**。若解退回原median應如實記錄，不能包裝為新改善。

不加正則、截距、分群／日期權重、比例網格、median權重下限、第二目標或額外縮減。不按真價、殘差、城市、ID刪列或分流，不用Public調參。多解接受固定runtime、固定列序求解結果，不能看query選另一解。

## 固定資料與隔離

重用R98已封存的11-key stage輸入，manifest為 `outputs/round98_inputs/manifest.json`，SHA e496b2afcf7ebc69a4e248d13b691e3b2e6e991bfab4e510386ff77cccc97fc7；來源原計畫SHA394a65459349e479bb3e8a487b60516156705469fa51bfbe75fd8697bae3642b。先核原provenance/來源/sidecar再derive第四欄；不能把原preparation plan身分改成R99。不重新反序列化raw價格pickle準備資料。

|stage|query起訖（含）|meta來源|meta／query列數|
|---|---|---|---:|
|development|2022-08-11～2023-08-10|auxiliary原forward全段|105511／81234|
|offset|2023-02-11～2024-02-10|auxiliary＋development嚴格早於2023-02-11之前31801列|137312／107169|
|confirmation|2023-08-11～2024-08-10|auxiliary＋development原forward全段|186745／133697|

原三欄float64、canonical/raw索引、列序和日期不變；meta日期<本stage cutoff、零重複／與自身query不相交。每個stage獨立程序，只讀自己的合法meta_y及本stage精確11-key readset。跨stage較早query可成為後期meta標籤，不能聲稱整體從未見過評估交易的價格。

三stage全部跑完，四欄來源／derived hashes、四權重、solver state／certificate、完整query保存重載和v22 median均封存成功後，另一scorer才讀query價格。無同窗early stopping、不依首窗分數改模型。原生模型fit及learned states保持原origin，沒有新native model load／fit，最多3次CPU meta fit。

## 固定求解與工程核验

沿R98等價dual LP，把三個價格約束擴為四個；P/y共同除1e6僅為數值尺度。固定SciPy1.18.1、HiGHS1.12.0、highs-ds、presolve=True、steepest-devex、threads=1、time_limit=300、maxiter=1000000、primal／dual feasibility 1e-8。

直接取四個負inequality marginals，不clip／normalize／fallback／自動重試。沿R98證書：weight sum 1e-10、權重嚴格>=0、dual box1e-8、dual constraint mean .01TWD、stationarity price max .10TWD及freevariable1e-10、objective identity mean .01TWD、primal-dual gap mean .10TWD、complementarity mean .10TWD、vertex optimality mean .10TWD。四vertex含M，另由driver比較同meta舊R98保存primal：R99不得高於它+.10TWD；這是工程／目標可行性檢查，不是新增query gate。

人工測例必須包括R98的median反例新增M後可達0loss、M損壞／欄位錯誤／索引错位被拒、數值證書與模型JSON保存重載、solver失敗完整保存。helper不讀真實檔案求fit。失敗保留原目錄/source/收據，任何修复另存新來源與執行，不削弱SHA或拒覆寫。

## 固定評估與判讀

主要控制為同窗v22，另外保留R98原保存完整預測作明示的舊候選對照，R98 source/model/predictions不動。scorer對N/H/U/v22/R98/R99六個向量報完整MAE、seen/new/missing、城市、calendar／relative月份、四季、五真價帶與尾端誤差。原v22及R98控制指標須匹配既有同窗結果至1e-6 TWD。

沿原mix權重seen=.36617593949822613、new_nonmissing=.6338240605017739、missing=0。分群依本query原native-fit群，不能改依較小meta。真價群只作事後診斷，絕不可用於test推論。

只為R99−v22做paired project／三個月循環month各2000次；seed20260925，stage indices development=0/offset=1/confirmation=2，familyproject=0/month=1，保存六份replicates及12區間。與R98固定同抽樣設定不代表實驗獨立；不用R98 bootstrap區間冒充R99。R99−R98報所有固定点差，不增加另一組bootstrap或按結果挑抽法。所有有利/不利證據均保留。

事前探索門檻沿R98：三窗overall與weighted全低於v22，且任何季度或至少500列固定分群不得有MAE>5%退步。十二區間完整報告，跨零不稱確定改善，bootstrap不是Private勝率。offset重疊，原歷史窗口已多輪選型、缺auxiliary以前forward block；即使全過也只支持另立較早block研究，不能說完整原三年度採用門檻已過。

此輪不全量fit、不產新CSV、不代上傳Kaggle、不更改使用者唯一選定v22或正式PPTX/PDF。價格只用課程標籤，禁止查內政部成交答案。完成後依實際結果更新HANDOFF/HANDOFF_LOG、source manifest與新文件包；保存來源和重載一致不等於乾淨環境從零重訓全鏈。
