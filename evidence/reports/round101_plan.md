# R101：R100價格先驗表示與支持分布診斷

2026-09-29，任何本輪真實fit-label陣列／prepared陣列載入或新統計前固定。前goal回合完成R100六NN與三窗評分、核驗、摘要、私人GitHub4217cc53…b066及3483檔交接ZIP，屬實質progress。本回合CIM查無Python模型程序，不重跑R100。此為面向後續改良的診斷，不是新模型、盲測或Private評分。

## 待釐清問題與界線

R100原生T NN三窗退步，最終組合兩好一退；完整控制C使用同eligible列。待檢查時間前置prior是否改變fit ratio表示、支持層級及其近期分布。這不先假定因果，不選補價係數，不重掃quarter／warmup／seed／融合比例。

原目標mean(w*abs(F*r-y_million))在F>0時數學上等於mean(w*F*abs(r-y_million/F))。因此同時報target ratio與其有效ratio-loss權重wF；只看未加權ratio平均不足以描述訓練目標。NN optimizer仍用原float32總價loss，本診斷的float64代數不聲稱浮點逐位等價或替換trainer。

## 固定stage及readset

依development、confirmation、auxiliary三個既有外層窗口，各獨立子程序；父只metadata。先驗R100 main/eval terminal exit0、來源與封存身分、9保護。每stage只可載自己的R100 C/T prepared.npz（精確9keys）、完整float64 fit-only y與eligible positions、canonical/raw索引及pure-X mask/folds。原pure-X features與canonical mapping、R60原OOF prior／full-query prior／fullfit prior JSON、R100 temporal prior／prefix JSON及metadata、三套native預測（原R63及C/T）可只讀。額外只读既有同stageR63 prepared的必要核對亦可，但不得載其他stage NPZ或fit_y。

不得讀raw train價格pickle、valid、test、query labels或query誤差陣列；不載NN模型，不推論模型／重建價格prior／score／bootstrap。原R100 query價格曾評分的歷史已知，不能宣稱新盲測；本診斷程序本身不用query真價。

每個輸入依保存權威SHA核對。原/新fit prior仍用同eligible列、原row order；C/T八個其他prepared陣列精確相同。fullprecision y/1e6轉float32必須與兩arm train_y對齊；訓練ratio使用實際saved float32 y及F轉float64計算。原fullfit query factor必須共用且和原R63吻合，先驗F=float32(A*P/1e6)逐位核對。

## 固定數列與統計

fit：C/T的prior單價、float32 factor F、target ratio y32/F32、T/C prior比值、T/C factor比值與其log、C/T有效ratio-loss權重wF。報一般權重、原w及各arm wF下的ratio描述；T/C目標ratio比值與C/T factor反比按浮點容差核對。prior比值和factor比值因float32有微小差別，不能冒稱逐位相等。初始化median與inverseSoftplus依原training JSON精確重播。

全體及每一個事前固定分組，報rows、weight mass fraction、ESS、mean、std(ddof0)、min/max及0/1/10/25/50/75/90/99/100百分位。一般分位數沿numpy linear；加權分位數用stable-sort後第一個累積權重>=q*sum的inverse-CDF，不插值，明示與一般median定義可能不同；空群保留rows0與null統計。

fit分組只有：(1)每個已出現calendar quarter，(2)離outer cutoff的固定age帶 [0,365)、[365,730)、[730,inf)，(3)原C與T所選支持層級0..4的25種transition（包括空群）。支持層級從純X原折排除／季prefix reference counts或已保存state cells.rows讀得，不能用query誤差分組。只報原始支持列數和reference中365日權重ESS的既有可用摘要，不將支持列數叫獨立建案數或可靠率。五層已包括global；所有eligible需被單一level指派。

query：只讀原fullfit prior、共用F以及原R63/C/T的已存native總價，除F*1e6反推模型輸出ratio（浮點反算，不稱原ratio bytes）。報全體、12個相對forecast月份、既有seen/new/missing及fullfit prior五支持層级的先驗與ratio分布，T/C及T/原native預測比值。seen/new由原完整outer-fit project membership定義，不用eligible子集。所有query rows保留，不因價格或誤差挑切片。train與query分布並非相同房子，觀察差異不等於因果、過擬合證明或Private預測。

對T/C prior比值固定報<0.9、[0.9,1.1]、>1.1列數及原w質量，屬事前描述尺度而非下游調價規則；不依結果改門檻。不增加具MAE最優性的評估或group排名。

## 執行、產物與後續

保留新src、固定source/request、人工統計/indices/反比與拒絕測例、3stage aggregate JSON、原始SHA及實測時間、完成摘要；不輸出逐列y／ratio或新CSV。所有原R100與v22資料／狀態不改。此輪無native fit／query MAE／Public操作。

結果只能支持進一步可否證假設。提供log prior給NN是待源碼核對的可能改法，R65曾試舊groupOOF logprior；不能因本診斷分布不同就自動把新舊組合視作會成功。任何後續模型另固定計畫／對照／窗口與完整驗證，不能偷偷在此診斷追加fit。完成更新HANDOFF/LOG/exporter與可交接快照；Private仍未知，使用者唯一提交v22不變。
