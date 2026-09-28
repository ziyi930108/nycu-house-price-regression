# R100：時間前置價格先驗與相同訓練列對照

2026-09-29（Asia/Taipei）。在任何R100價格先驗計算、NN fit或query MAE前固定。本輪承接R99未跨窗改善及有限來源盤點；是適應性研究，不是盲測。R99已發布c34842980f8abc4bcc15dc9d5940fe2d50ac118f，使用者唯一提交仍v22，Private未知。

## 假設與唯一主要候選

現有R60價格先驗按建案五折排除held組，但其他折可含比訓練列更晚的fit成交。這不構成外層query價格洩漏；待驗的是train基準和預測未來時基準的時間方向不同，是否限制prior-ratio NN泛化。

R100T只把prior NN的fit價格因子改為「日曆季起點以前、且排除held建案折」的近期加權單價先驗。仍沿R60各層key、30原始列cell支持、365日加權中位數、global fallback與有效面積A，factor=float32(A*prior_TWD_per_m2/1e6)。每個quarter/fold獨立建reference state，anchor只由該reference有效日期求得。沒有同季價格、後期價格或held組價格進入該fit列prior。

## 固定時間與冷啟動規則

日曆季為1/4/7/10月1日起；day以2000-01-01為原點，reference必須有效日期且嚴格小於季起點。held fold沿原R60：原canonical fit indices進cluster_keys，sha256('42|'+key)前8bytes小端整數mod5；不得用raw cache indices或重設列序產fold。

整個quarter只有在五種held-fold reference各至少30列時才可作NN訓練；eligible row另須有效日期。無效日期不能猜時間，不進prefix或NN訓練。最早不合規季度只可作往後prior的歷史reference，不作本輪NN損失。這是事前固定的warmup慣例，30借用原cell支持尺度，並非舊global原有門檻或經最佳化的值。純X稽核只核此一規則及精確mask，不掃門檻／季長；若無可訓練列則停止並保留證據。

同一期仍只有一個outer fit/query切點。內部quarters只是fit先驗的生成，不是把NN／encoder當季度模型回測；encoder和R/S是原同窗完整outer-fit狀態，不能聲稱所有NN輸入均為逐列as-of生成。

## 相同訓練列控制與不變項

為分辨移除warmup列的影響，每個outer窗口新增兩個NN：
- R100C：相同eligible mask，但使用原R60分組OOF price factor。
- R100T：同mask，使用新的quarter-forward group-OOF price factor；唯一主要候選。

兩者的train_num/train_cat/train_y_million/train_sample_weight均取同一保存R63陣列的同mask切片，列序不變；原365日float32外層fit權重不重新正規化。query_num/query_cat/query_factor/cardinalities完全共用原R63，query full-fit prior仍用原完整outer fit（含warmup合法標籤）。encoder／query prior／R/S不縮為eligible子集，保留此範圍差異。

重用未改動src/round63_recency_weighted_prior_train.py和其架構，seed42、30epochs、batch2048、原AdamW/架構/總價加權L1不變；bias仍按各自fit y/factor中位数求得，公式相同、數值可不同。不early-stop、不按query選epoch、無另seed或超參數搜尋。兩個新NN依序執行，最多6次GPU fit；使用已驗證同架構，mask不增加資料量，不另做Attention或新GPU架構試驗。

完整模型保持：B=.75*(.5*saved_R54_area+.5*new_priorNN)+.25*saved_unitLGB；用原同窗reference/spatial各一次成Nnew，再median(Nnew,saved_R46_H,saved_R79_U)。原v22完整分支及N控制需重播吻合。R100C是機制控制，不自動當第二個可挑選提交候選；既有v22與v24亦保留為固定比較。

## 時間窗口、價格隔離與執行

固定順序development、confirmation、auxiliary；沿R79/R63原canonical索引：
|窗口|fit列數|query起訖（含）|query列數|
|---|---:|---|---:|
|development|240316|2022-08-11～2023-08-10|81234|
|confirmation|321550|2023-08-11～2024-08-10|133697|
|auxiliary|134805|2021-08-11～2022-08-10|105511|

前置只讀pure X／來源／fold，先保存mask與支持量。完整float64 fit y由隔離extractor按canonical→raw映射輸出每stage fit-only NPY；原raw pickle需反序列化整表的限制明記，不宣稱extractor從未接觸query bytes。各stage模型程序只讀自己fit-y及自身R63 NPZ，不能讀raw價格pickle或其他stage fit y；較早query可為後期fit資料，同stage query永不進其fit。

兩arm prepared各恰9個R63 keys，沒有query y。每stage保存prior provenance／prefix支持／新與控制NPZ／模型／全query重載／完整分支／封存；原input/source/保護SHA不變。全部六NN與三窗預測都封存成功後，另一scorer才取query y並評分，不依首窗改設定。任何真fit失敗保留完整失敗目錄，不自動重試或覆寫舊guard／hash。

## 固定評估與解讀

四個完整向量R100T、R100C、v22、v24及原生prior→base→reference→spatial→median層級都報元MAE、seen/new/missing、城市、calendar／relative月、季度、五價格帶及尾部。seen/new判定依原outer完整fit群，不改依mask。固定mix權重seen=.36617593949822613、new=.6338240605017739、missing=0。

配對比較預先固定R100T−v22為主要、R100T−R100C為機制對照；各stage project及三個月循環month各2000次，seed20260925、stage development0/confirmation1/auxiliary2、family project0/month1，同抽樣規則，合24個區間且全部報告。R100C−v22與R100T−v24報完整點差，不增加抽樣或看結果選control。

沿前期研究門檻：R100T三年度overall與固定weighted皆低於v22、confirmation整體至少改善0.5%，且無季度或至少500列固定群MAE退超5%，才值得另立採用研究；同時必須保留對R100C的完整反證，不能把單純mask效果稱先驗機制改善。這些是探索門檻，不是Private保證；不因一窗好而換門檻。

歷史窗口已多輪使用、單seed仍有訓練變異，bootstrap只條件於保存模型，不校正選型或未知未來分布。真價帶只作事後診斷，不能進test分流。不得查內政部成交答案、靠Public改參數或按test ID改價。本輪無test預測／新CSV／Kaggle操作，v22與正式PPTX/PDF不變；完成後更新交接、保存source/失败/數值證據，不能以可追溯冒稱乾淨環境全鏈重訓。
