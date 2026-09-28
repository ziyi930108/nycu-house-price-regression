# R102：時間價格先驗的明示輸入，同寬度對照

2026-09-29。R101三個隔離診斷程序全部exit0、64.359秒完成後，任何R102 prepared／模型／query MAE之前固定此計畫。不是新盲測，不以Public搜參數；Private未知，使用者唯一提交仍v22。

## 依據與可否證假設

R100 temporal-prior原生NN三窗均退；R101保存聚合確認，C→T fit target ratio的普通中位數依dev/conf/aux由0.893695→1.067085、0.881186→1.083431、0.963422→1.016911；有效ratio-loss權重wF下由1.002030→1.122437、0.998575→1.114349、0.993068→1.045264。query T/C native比值中位1.006247／0.969370／1.020182，方向不一致，不能用一個全局乘數修復或從分布斷定誤差原因。

假設只限：在R100T時間factor與loss列不變時，明示同源單價先驗可改善NN的ratio校正，且能保留到完整回歸組合。原輸入可能間接推知先驗，不能稱不可識別或已找到退步因果。R65舊OOF logprior曾改善原生卻使完整模型退；R71新base最新期A144三seed勝B145，最強既有反證保留。R101來源盤點範圍有限；此為適應性研究。

## 兩臂與唯一處理差異

兩arm都從各stage已封存R100T prepared九陣列出發，已eligible的列不再mask，train factor／y／weight/query factor/categorical/cardinalities均原bytes。兩者145 numeric欄：
- C（zero）：原144欄後加float32正零；固定機制控制，不事後另選為候選。
- T（log）：第145欄fit用保存temporal單價prior的自然log，再用全部eligible fit列的float64 mean/std(ddof0)標準化、轉float32；query用原R60 full-outer-fit query prior的log並套相同fit mean/std。

不clip、不按query重新標準化、不按價格區間做分流；prior與std需positive finite，否則報錯保留。兩arm前144欄逐位相同、其餘七個prepared arrays逐位相同，query_num與train_num僅最後欄不同。保存scale JSON與array identities。C最後欄正零，而不是隨機噪音；同寬度可排除144→145形狀差異，未聲稱保存初始RNG trace或完全消除訓練變異。

重用未改R63 trainer，SHA26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a。seed42、30epoch、batch2048、AdamW／架構hidden層／總價日期加權L1不變，numeric count145。兩臂相同fit y/F中位數與inverseSoftplus初始bias，原權重不重新normalization。無earlystop、epoch選擇、額外seed／參數grid；最多6次NN fit，不重建prior、encoder、R/S。

## 時間與執行隔離

stage順序development、confirmation、auxiliary，每期C再T，共六fit由主線单GPU串行，各stage獨立程序。
|stage|原outer fit列|eligible loss列|query日期（含）|query列|
|---|---:|---:|---|---:|
|development|240316|240260|2022-08-11–2023-08-10|81234|
|confirmation|321550|321494|2023-08-11–2024-08-10|133697|
|auxiliary|134805|134749|2021-08-11–2022-08-10|105511|

每個模型程序只讀自己R100T NPZ、saved temporal prior／query prior及索引、pure-X cache、原同窗R/S/geography／固定分支；不另讀完整fit_y、raw train價格pickle、valid/test、query y或其他stage NPZ。既有sealed metadata／R100核驗與R101可追溯復用；原705 prefixstates不重算。fit標準化是完整outer-fit eligible scope，不等於所有處理逐列as-of；query prior/R/S/encoder原fullfit scope也保留。

完整模型仍B=.75*(.5*saved_R54_area+.5*new_priorNN)+.25*saved_unitLGB；原同窗R與S各一次，再median(Nnew,saved_R46_H,saved_R79_U)。保存native/base/reference/spatial/median、診斷、模型與全query保存模型重載exact。v22完整鏈／原保存分支須精確重播；固定分支不重訓。

所有模型／預測／來源／輸入封存後，獨立scorer才讀query y；不能依首窗分數改後兩期。模型失敗不自動重試、不覆寫原資料。所有來源與request預先固定，失敗若需修工程另新source/目錄／收據，不改門檻或hide failure。

## 固定評估與選擇邊界

完整向量T、C、v22、v24四個，及native/base/reference/spatial/median五層：元MAE、seen/new/missing、固定weighted、城市、calendar月份、relative月份、四relative季度、五價格帶／尾端全部報。seen/new按原full outer fit定義，非eligible subset。weighted seen=.36617593949822613、new=.6338240605017739、missing=0。

主要完整比較T−v22、機制T−C；各三stage × project/month × overall/weighted，2000配對replicates、seed20260925、stage索引dev0/conf1/aux2、family project0/month1，沿原三月循環month／project方法，共24CI，全部保存。C−v22與T−v24完整點差，不額外挑control或追加bootstrap。舊R100T144/C的已保存原生MAE只作背景，不能混稱新145零欄C。

primary研究門檻沿R100：T完整三窗overall/weighted均低於v22、confirmation整體至少改善0.5%、所有季度或至少500列固定群不退超5%。另機制必要條件為T相對C原生NN三窗MAE皆低且完整三窗overall/weighted皆低；沒有通過不可稱明示prior機制成功。分開呈現primary和mechanism gates，不能用後處理收益替代原生改善。

這些是探索門檻，不是Private保證；單seed、反覆使用歷史窗口及兩級prior支持範圍差均保留。bootstrap條件於保存模型，不是Private勝率。無test預測／新CSV／Kaggle操作，不改選定v22或正式報告／PPTX。完成保存所有結果、source與真正耗時，更新交接；未做乾淨環境全鏈重訓不宣稱已做。
