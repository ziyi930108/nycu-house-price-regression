# R89方向盤點：型態×用途時間斜率與一致的R/S正規化

2026-09-28。原方向盤點現已由主線正式凍結[round89_plan.md](round89_plan.md)，並完成新helper與23項純合成檢查。尚未在本代理擬合真實課程斜率或計算任何R89真實query候選分數；沒有test預測或CSV。下文的方案歷史保留，以正式計畫與helper為實作規格，不能把合成驗證當完成真實實驗。

## 歷史已做過什麼

|歷史方法|實際範圍|已保存結果與解讀|
|---|---|---|
|R2全域hedonic/reference|只用fit最近四年、同案至少8列，在案內中心化後以6輪Huber同時估面積、停車、樓層、時間四係數；所有型態／用途共用一組beta。局部reference殘差與query還原共用此beta，trend另外只作用在非局部參考份額。|這已經有within-project時間估計，不能把中心化或時間趨勢當新方法。|
|R3城市收縮future-only|每市至少30案／1,000列；四係數重新估後只取時間項；固定min(.5,案數/(案數+200))收縮至原全域beta；只改未來base額外趨勢，局部殘差與還原不變。|development v2=1,539,585.29、regional=1,530,413.10，約改善0.60%，未達原固定1%門檻。freeze選v2；confirmation只跑對照，沒有城市候選的confirmation分數。|
|R45再接R3至三分支median|原R3所有規則不改，共用同窗regional state，raw/log/unit各改reference預測，後接原R8；R8仍用全域beta/Bfuture。|三期相對當時v9：development−0.784%、confirmation+0.110%、auxiliary−0.024%。已是完整三期測過的城市方案，不能重做後稱首試；也不能把其限制混稱為全系統城市正規化。|
|R8 spatial|重用原R2殘差/beta，固定2km、同city/type/use、至少5案、最多40案，建案等票的median/MAD，補弱reference來源。|原R8 development整體改善0.725%未達當時1%門檻；當時未跑其後兩期。較後管線另有採用沿革；不能把舊R8未採用當作目前v22沒有spatial。|
|R82移除單價分支reference future trend|僅把R81單價reference改anchor，原spatial中的global beta/Bfuture保留。|已測過局部撤銷trend，並非全鏈移除；不應再用「只有date weight所以trend重複」推論。|
|R88無局部支持回退|原v22三分支都只對base_fallback且spatial未套用列回base。|三期整體皆退步；全面撤銷政策不成立，仍不能證明所有型態共用時間斜率最好。|

[原R3報告](optimization_round3_report.md)、[R3 development結果](../outputs/round3/evaluation/development/results.json)、[R3 freeze](../outputs/round3/frozen_candidates.json)、[R3 confirmation結果](../outputs/round3/evaluation/confirmation/results.json)、[R45報告](round45_regional_report.md)、[R45結果](../outputs/round45/results.json)、[R8報告](optimization_round8_report.md)、[R82方案](round82_plan.md)、[R88報告](round88_report.md)。

本次搜尋src／reports的regional API、effective_time_beta及型態／用途趨勢字樣，找到R3與R45，而沒有找到已實作的type×use一致重正規化。這是此次盤點範圍內的發現，不等於證明所有可能名稱都未曾出現。

## 單一候選建議

分組g=(building_type, main_use)，不再乘city或按query誤差挑特定類型。保留全域beta前三個係數、原面積／單價界限、fill、日期anchor、鄰居、幾何、比例、clip和三分支base。只有共享時間係數改為fit側的型態×用途收縮估計，並在R與S中一致使用。

1. 只取當期fit內原reference_residual有限、日期有限且位於原anchor往前4×365.25日至anchor、原具名建案的列。這沿用原面積15–1,500及fit單價0.001／0.999分位篩選，不能另用query價格篩選。
2. 中心化案鍵用原R2的city|district|type|use|project。每案在上述支持列至少8筆且日期max−min至少180日；不足的整案不用於新斜率估計。獨立案支持數用city|district|project，避免將同案拆分高估案數。
3. type×use組須非缺值，保留支持列至少1,000筆、至少30個獨立案，這些列最早至最晚日期跨度至少730日。未見、缺值、少案／少列／短跨度組直接沿用原global時間beta。180／730是本次評分前提出的固定支持規則，並非已證明最佳參數。
4. h沿用原matrices與原fill。令e=log(y/A)−h[:3]·beta_global[:3]，t=h[3]；對e與t各減同案均值，得到e_c,t_c。只估一個斜率b，不再每組重估另外三個hedonic係數。
5. 每案等初始總權重：每列w_base=1/n_project。6輪Huber IRLS，threshold=.10；每一輪w=w_base×HuberWeight，不是只有第一輪等案。單參數解可用b=(sum(w*t_c*e_c)+1e−5*b_global)/(sum(w*t_c²)+1e−5)，零／非有限時間資訊回退global。ridge是數值穩定项，主要收縮仍在下一步。
6. raw b截在原時間係數界限[-.08,.20]，alpha=min(.5,N_projects/(N_projects+200))，b_g=(1−alpha)*b_global+alpha*b_clipped。全部係數、支持數、跨度、Huber有效權重、回退原因均保存，不依query選beta或門檻。

步驟5的每案等權是相對R3的新增設計（R3交易列初始等權）；作用是減少大案交易筆數支配斜率，不能說已經證明優於舊權重。fit支持代理只提供支持量，沒有被要求估新候選MAE；支持量不足時應照規則回退，不能為獲得非零候選事後放寬。

## 只改future_base與完整一致正規化的差別

舊R2殘差是r_i=log(y_i/A_i)−h_i·beta_global。R2局部還原使用center+h_q·beta_global，額外future趨勢用B_future=B*exp(b_global*future_years)。R8的G亦使用同一殘差和query beta，並以B_future作clip尺度。

只改future_base是在測「剩餘base未來增幅」，並未改局部成交的日期調整或案間殘差；R3/R45已採這一路線。若只改reference的B_future而把spatial的B_future保持舊值，兩層使用不同基準；若把spatial B_future也改但仍用舊殘差/G，仍然是局部global正規化＋分群base外推的混合方案。這些可明確定義，但不能稱整個系統已使用分群時間趨勢。本候選不再把它當第二個平行備選。

建議採同源完整方案。令delta_g=b_g−b_global：

- 複製舊finite residual，對**全部原合格reference列**更新r_new=r_old−delta_g*t_i，包括早於斜率估計四年窗的reference列；不能只改拿來估斜率的列。日期缺失的舊合格reference仍使用原matrices填值語義。delta=0的列直接保留舊bytes。
- 對query用beta_g=[原前三係數,b_g]，局部還原與B_future皆由beta_g產生。R2的local clip仍相對未trend的原base，R8的G clip仍相對B_future；保留原本兩種clip定義，不能順手統一。
- 用新residual、同一fit X/geo及原門檻重建一次spatial保存結構；其每案median／MAD、G還原和B_future均使用同組beta_g。G不額外再乘一次future trend。
- 新residual會改變R2 dispersion/confidence及R8 MAD/alpha，因此必須重算diagnostics，不能借用舊source/confidence數值充當新診斷。鄰居距離、索引、eligible集合應不變，但數值權重可能變。
- 同一新trend/reference/spatial來源供NN、log、unit三支使用，各自原base不改、各自重算R/S後再median。不是只改單價一支，也不是在最終價上乘一個倍率。

此法是單一整體正規化介入，結果不能拆成「只有future項」的因果效果。用舊global前三係數仍可能有群組差異；即使案內中心化，也不能排除同案推出戶別／銷售進度與日期相關的偏差。

## 實作架構審查

主線提出的group-view方式可避免重寫R2／R8算法：

- 新wrapper保存只讀global_reference/global_spatial原state、新reference_template/spatial_template及type×use斜率mapping。wrapper有獨立family/format；template內的global beta只能視為fallback，不能直接拿template全query推論、不能冒稱舊v2未修改state。
- 新template共用原groups/reference_distance/fill，只有residual拷貝及新版身分；spatial從新residual重建。各supported且delta不為0的query組，reference/spatial各做淺拷貝view並複製beta陣列，只替換beta[3]，再呼叫舊公式。
- 必須直接從fit X逐列檢查：每個R2 groups[level][key]的所有row index同type×use；每個spatial group的row_positions也同type×use。不要拆字串key猜類別。R8會strip字串而R2未必，若有正規化碰撞導致同spatial組包含不同斜率組，應在評分前失敗，不能靜默混用beta。
- query以原位置分組／散回，X、geo、R/S diagnostics索引相同；原序、dtype、缺值及不可變state均核對。
- 為保證fallback bitwise，predict先用原global R/S對**全query**重播基準（與driver原v22對照），只對supported且delta不為0組重算並覆蓋。其餘prediction及diagnostics直接保留基準；這也避免子批BLAS造成舊行微小數值差。正式推論不依賴舊預測檔即可運作。
- 支援reload：pickle整個wrapper，driver對全query重載重播，參數／state／source及全部未取整reference、spatial、diagnostics核對。helper API不接受query y；slope只由fit_model的yfit估計。

建議介面：

    fit_model(Xfit, yfit, Gfit, reference, spatial_state) -> model, report
    predict_model(model, Xquery, Gquery, base_TWD) -> {
        reference, spatial, reference_diagnostics, spatial_diagnostics,
        global_reference_prediction, global_spatial_prediction, report
    }

model/state至少保存family/format、原state身分、fit X/geo/y與indices身分（indices由driver）、fill/global beta/anchor、固定參數、每組支持及raw/clipped/shrunk斜率、new residual SHA、new spatial content SHA、source SHA、零query參與聲明。原R8 build回傳的內層summary不等於新wrapper總體語義；外層必須清楚記錄本輪有新時間參數估計／新residual與spatial重建，不能沿用舊「不重建state」敘述。

必要小型合成檢查：兩組不同斜率、同案level差不影響中心化、每案權重总和、短跨度/少案/缺組回退、delta=0與所有不支持列bitwise保留、reference與spatial同組斜率、query無y介面、散回序／切片一致、state不變、pickle重載一致。真資料driver另核原三支R/S全向量及候選完整三期；此處未執行以上新檢查。

## 成本與最強反對理由

實測參考：

- R45三期新regional state各1.984／2.719／0.954秒；全三期3支reference＋3支spatial主流程62.891秒，零樹fit。見[run_execution](../outputs/round45/run_execution.json)及[results](../outputs/round45/results.json)。
- R8 development的建構＋推論／核對runner合計3.109秒，並非純build成本，不能當成新方案必定3秒。見[metadata](../outputs/round8/models/development/metadata.json)。
- R88三期9支原R/S重播等主流程57.734秒；R87主流程47.063秒。這些是歷史實測，不是R89已花時間。

R89沒有樹／NN訓練、沒有GPU必要；會有3次fit-only斜率估計及3次spatial重建，不能泛稱「零fit」。保守估計主流程約1–3分鐘CPU（含三期完整global控制＋新三支、state與source核驗、保存／重載），generic/paired另計；支持組數、分組dispatch及pickle大小可能增加耗時。這只是成本估計，R89尚未實測。

最強反證是：R3/R45城市收縮未呈現穩定跨期增益，R88撤銷global在三期均不利，v22三支base已用日期與近期權重，新增分群可能只改變noise；非住宅組可能支持少而全部回退、跨城市合併亦可能抹平地區市場差異。新residual會同時影響local confidence與spatial MAD，未必改善尾部。支持門檻與收縮不能消除反覆窗口的選型偏差；即使本輪改善，也不是新盲測或Private勝率。

## 本次讀取的核心來源SHA

|檔案|SHA256|
|---|---|
|src/round3_reference.py|4d78963763458e5fd2dab2847a45cd7f356c58dc2288081c1f4006cd8dba3631|
|src/round2_neighborhood.py|cae244c40e6616d6c72f3aff7c3423a50a5bfbfbf35df33a7d044186eb491ae9|
|src/round8_spatial_reference.py|de86397255d654f363f3b746d579923fe689aa41d9dc16335d0922fea05e1c32|
|src/round45_regional.py|1ce2cb215a8ca9a4cb722e3d8d98fa18cbaab315c4e1506fdcd34d32a444a2b8|
|outputs/round3/evaluation/development/results.json|f14e8c02f35f50a60445fc0488b8c200e90d78e02dd1f57d360a72a773b83fa0|
|outputs/round3/frozen_candidates.json|67edc169db83047e554060f7a1350e822f43ba95b0f61aa6cb8ab8ca02d66257|
|outputs/round3/evaluation/confirmation/results.json|29c4c9605fdba6bdb2fe00b411d2effb34512ee0a80dfc618430d5074b5de58c|
|outputs/round45/results.json|2d6a81310fc4eb369bbe1610ea7c00bb854ece55106071e1cd8bda627182ac5d|
|outputs/round45/run_execution.json|bee036a65cbfc235424bcd16de4007d087e3b019f9aa96b329206706b92c1e79|
|outputs/round8/models/development/metadata.json|a7dd065eec9760657a14879d1879bde9c3621e128d32a0da4e437b83a9bd1430|

本盤點只讀以上歷史證據與R85–R88交接，不載入query逐筆標籤／候選向量計分、不訓練、不修改原程式或state。正式freeze須另保存R89計畫與實際source；本文件是待主線確認的方案，不是已完成產物。

## 正式helper實作與合成檢查更新

主線正式規格使用centered_log_unit−centered_H@global_beta估delta，再raw_gamma=global_time+delta、clip和收縮，與先前時間gamma寫法等價但實際API以delta為準。已實作[src/round89_group_trend.py](../src/round89_group_trend.py)，source SHA 233b7919d738fa6eff199138fd2a301cf67ed55f2b0fc6e64c92c10f59de71b0。23項檢查通過，完整名單、來源SHA及實測耗時見[round89_helper_selfcheck.json](round89_helper_selfcheck.json)。

合成檢查曾兩次exit1：第一次是人工query赋值使pandas object變成StringDtype，修正合成fixture保持原schema；第二次是pickle重新序列化不保證bytes相同，改用state_identity值語義＋全合成query預測／diagnostics exact。predict前後原model pickle bytes未變仍有核驗。另將中性fixture單價設為1以讓log恰為0，沒有對實際delta加epsilon截斷。這些不是模型效果失敗或真資料重訓，失敗與修正列入收據。

helper回傳detached metadata，supported_groups保留與獨立fit支持審計對應的rows／獨立案／R2案鍵／總日期跨度。predict回傳完整R/S與global兩向量／四份diagnostics、affected_mask及fit_supported_mask；affected只由state與query群別決定。真正有spatial.applied且非零群斜率的合成案例已手算核對G和Bfuture，缺日／366日cap保持原規則。主線負責真資料支持量交叉核對與三期三支全query重載；本代理未讀真實query labels或計算候選MAE。
