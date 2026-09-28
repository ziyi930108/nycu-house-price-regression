# R89 候選前審查：建物型態 × 用途的 fit-only 時間斜率

日期：2026-09-28。這份獨立審查只讀既有 source、歷史報告與交接；未載入價格／模型向量，未執行新候選、query 評分、CPU 趨勢 fit、GPU 訓練或 synthetic。下列為實作前要求及待執行測項，不是已通過的收據。未改舊 helper、selector、CSV、root handoff 或 GitHub。

**結論：可作一個事前固定的後續假設，但必須把 fit 殘差、reference 局部價格、reference 剩餘 base 趨勢、spatial state 及 query 一起版本化。** 僅替換 `beta[3]`、沿用舊殘差或舊 confidence 都不是所宣稱的一致模型。是否優於 v22 尚無 R89 證據，Private 未知。

## 既有研究與最強反證

R3 已做城市分群、四年 fit 窗、每案至少8列、市別至少30案／1000列、最多50%局部權重的強收縮時間斜率；當時只變更 reference 剩餘 base 的未來趨勢，局部價格正規化不變。development 約改善0.60%，未達原1%接受條件，因此當時維持對照；不能改寫歷史門檻。

R45 已把同一 R3 方法接到當時三分支空間中位數，保留 R8 global beta 與原殘差。相對其當時 v9：development −0.783987%、confirmation +0.110240%、auxiliary −0.024089%（MAE差，負較好）。這是不同歷史基準，不可將數字當 R89−v22。R89 的差別是型態／用途分組及整條時間正規化一致，不是首次嘗試分群趨勢。

R85誤差盤點發現廠辦三期高估、單價分支 reference 後 MAE 較差，較早年度330列全部 base_fallback、無 spatial。這只提出風險；原 reference 分組本來就含型態／用途，不能說直接拿住宅成交當廠辦同組。廠辦只占全期絕對誤差約1.25–2.34%，一般100–200m²交易占約62–68%；研究不得只展示少數廠辦或億元案例。

R88 已測「無reference且未套spatial時三支各回退base」，三期完整 MAE 反而增加10,977.52／7,675.02／16,617.14元。這反對一律取消趨勢，亦提醒住宅主體、最新新案可能仍需要上漲外推；它不直接否定 fit-only 分群強收縮，但不能忽略。R89是看過既有誤差後的適應性研究，不是新的盲測。

## 主線已固定、待程式凍結的規則

本審查撰寫期間，主線確認以下單一規格：group為建物型態×用途；原finite reference、fit最近四年、named project；每案至少8列且日期span至少180天；群至少1000列、30個physical projects、日期span至少730天。response是`centered_log_unit−centered_hedonic @ global_beta`，以within-project centered time估一維delta；6輪IRLS、Huber .1、ridge 1e-5，每輪權重`1/n_project × Huber_weight`。`raw_gamma=global_beta_t+delta`先clip到[-.08,.20]，再以`alpha=min(.5,n_projects/(n_projects+200))`向global收縮。

這些是主線傳達的已固定設計，尚未在本報告執行、查閱新helper或確認plan/code SHA。180／730天及每案等權是本轮明确的新支持與權重規則，不是假稱R3原規約已有。min rows／project span／group span的計數必須全部基於最後合格fit列；physical project建議綁原城市＋行政區＋案名，不能將同案的地址／多筆列充當多案。同group內project權重以合格列數為分母，Huber每輪重算但1/n不變，是否按和歸一化會影響ridge，程式必須採唯一明定形式。

方向符合本審查的最小估計建議。仍需保存Huber後有效時間資訊、案別集中度及回退原因；支持門檻足夠不代表跨年gamma可靠。主線另指定fallback使用完整舊query路徑，僅supported group使用新state/group query view，這可以保住fallback bytes，但支持判定只能來自fit，內層視圖不得被用於其他gamma群。

## 一致公式與最小變更邊界

令 i 是 fit 列，q 是 query 列；`A` 為有效建物面積，`ell=log(y/A)`，`x` 為原三個非時間 hedonic 特徵，`t=(trade_day_index−8000)/365.25` 使用原 `matrices()` 的 fit-only 補值。原 global beta 寫成 `(b, beta_t)`。`g=(building_type,main_use)`，有效群斜率為 `gamma_g`；缺失、未知或不支持群回退同窗 `beta_t`。所有 y 只可取該期 fit。

一個最小且可解释的估計是保持 b 不變，對原殘差的 within-project 去均值量估時間 delta：

~~~text
r_old_i = ell_i − x_i @ b − beta_t*t_i
z_i = r_old_i − mean_project(r_old)
u_i = t_i − mean_project(t)
delta_hat_g = fixed robust one-dimensional slope of z on u, within group g
candidate_gamma_g = clip(beta_t + delta_hat_g, fixed lower, fixed upper)
gamma_g = (1−w_g)*beta_t + w_g*candidate_gamma_g
~~~

上式與本輪主線已固定的一維delta設計一致；上節列出目前規格，程式與plan仍須另凍結。歷史另一種設計是群內同時估四係數、只保留時間項如原R3；但該時間項以群自己的面積／車位／樓層係數為條件，最後卻採回 global 非時間項，估計目標與下游正規化不完全相同。若採此路徑須事前明講、保留四項原始／clip係數，不把它描述成完全等價的一維修正。不可看 query 分數後在兩種估計法間挑選。

取得 gamma 後，對原 finite reference mask 的全部 fit 列（不限趨勢估計的最近四年列）重新正規化：

~~~text
r_new_i = ell_i − x_i @ b − gamma_g(i)*t_i
        = r_old_i − (gamma_g(i)−beta_t)*t_i
~~~

原 finite mask、面積／單價資格、fill、group members、distance features、neighbor數、level順序不改。等式第二式可避免重算原 global 部分；delta為零的列用copy保留舊bytes，不要為「fallback」引入不必要浮點漂移。這項轉換仍須驗證原reference確屬同窗對齊fit列，不能只因長度與最晚日相同就相信。

Reference 局部與完整價格須同時使用新r、gamma：

~~~text
center_q = old weighted-median algorithm over r_new of selected neighbors
MAD_q = old weighted-median absolute deviation over the same new residuals
h_q = x_q @ b + gamma_g(q)*t_q
local_log_q = log(A_q) + center_q + h_q
c_q = old level weight / (1+(nearest_distance/2)^2) * exp(−MAD_q/.18)
L_q = max(B_q,1) * exp(clip(local_log_q−log(max(B_q,1)),log(.65),log(1.5)))
F_q = B_q * exp(gamma_g(q)*clip(query_day−fit_anchor,0,366)/365.25)
P_q = (1−c_q)*F_q + c_q*L_q
~~~

`B_q` 是各支原始 TWD base，不是 native，也不是已套過趨勢的 reference。local price 裁切相對 B，spatial price 裁切相對 F；保留這個既有差異，勿順便統一成同一底價。R2中source與nearest distance只依群與幾何特徵，應仍相同；**MAD與confidence依新殘差而變，須重算**，不能沿R45直接拿舊diag。

Spatial state 必須重新綁定新r，原每案一票、2km、排同案、至少5／最多40案等幾何規則不變。每個 query 仍用同一 gamma：

~~~text
C_q, M_q = original spatial project-center and MAD, computed from r_new
G_q = F_q * exp(clip(log(A_q)+C_q+h_q−log(F_q),log(.65),log(1.5)))
alpha_q = .30*exp(−M_q/.18)
S_q = P_q + (1−c_q)*alpha_q*(G_q−F_q)
~~~

只在原弱來源與原地理／日期／支持資格成立時套用。未applied列必須與輸入P逐位一致。不能把P整體再乘一次趨勢、給G再乘一次趨勢，或把R3 future-only結果交給舊global R8卻聲稱整条一致。新的MAD、c、alpha自然連動，是此時間表示的完整影響；不能當成只改一個外推倍率的純因果估計。

若主候選宣稱整條系統改斜率，就將同一新reference/spatial state依次用於三支各自base，再取median；若只改單價分支，NN/log須明確凍結舊預測，候選名稱與假設亦須限定單價。**此branch範圍必須在評分前固定**，不能評完決定哪幾支用新修正。

## 數值、支持與資料風險

| 風險 | 實作要求／證據 |
| --- | --- |
| 有很多成交但全在同一天 | rows／projects不足以識別時間斜率；支持至少包含事前固定的案內日期跨度／有效時間變異／真正有時間變化案數。零或近零denominator直接global回退，不能靠ridge造出可解就當支持。 |
| 單一大案支配斜率 | 保存案數、列數、最大案占比、時間資訊集中度；row-level fit可沿舊方式，但不能稱每案等權。若改權重須事前固定，不能看到廠辦收益才加。 |
| 固定非時間項仍存在混雜 | 同案去均值去掉案間價差，未消除同案房型／樓層銷售時序或未觀察升級；時間係數是預測正規化參數，不是因果市場漲幅。 |
| robust within-project計算 | 原法先普通均值demean再IRLS，非每輪重估robust案截距的完整固定效果Huber解；沿用時照實描述。若新增加權demean則是額外估計變更。 |
| 斜率clip／shrink順序 | 固定bounds、prior、support後權重、IRLS輪數、threshold及clip順序；先clip再shrink與先shrink再clip一般不等價。檢查effective gamma有限且在規約界內。 |
| 四年斜率套到更早reference列 | 舊R2可引用不在最近四年的reference列；全部finite r都需與新gamma一致。只重算最近四年r會混用座標；若刪舊row則改了覆蓋，不是最小變更。 |
| 日期原點與大相消 | 原時間原點8000保留最容易對照；改fit-anchor原點在數學上可等價，但r的平移也須依群同步。須測query最終價不受時間原點改變（數值容差），不要只比beta。 |
| horizon截斷並不涵蓋local | 原R2/R8只將F的未來horizon限制366天；local h與G使用未截query日期。缺日期的R2 local使用fit填值，F的horizon為0；spatial缺日期不applied。沿用時明講這些語義，改它們需另立變因。 |
| group字串和歧義 | R2用astype(str)，R8會strip／合併missing；gamma lookup須單一canonical函式，fit/query同用。tuple或明確序列化pair避免`|`碰撞；確認同一spatial group中不混入不同gamma。保留舊reference group規則，避免偷偷改覆蓋。 |
| missing／unseen回退 | 缺型態或用途、fit未見pair、支持失敗、非finite係數一律同窗global；不得以query筆數補支持、真價／ID挑群、臨時借用下一期係數。 |
| exp與非法base | 用原log-ratio裁切，檢查finite、非負、TWD及shape。spatial B=0／非finite F時保留P；禁止NaN在median被靜默忽略。不得新增尾端截價以使結果看起來正常。 |
| query標籤滲入 | fit函式僅接Xfit/yfit與已核同窗reference；predict無y、無load_data／檔案讀取、無eval_set。支援判定、fill、quantile、gamma、state都不得讀query特徵統計或y；query日期僅用已凍結映射作推論。 |
| 調參式支持門檻 | fit-only支持盤點可判斷可行性，但仍需在query評分前凍結一套規則；不能為已知query廠辦誤差專門放寬其門檻。缺支持而回退正是可接受結果。 |

## 新版本契約與來源保全

1. **新增helper與新state家族**，不修改 `round2_neighborhood.py`、`round3_reference.py`、`round8_spatial_reference.py`、任何舊`PARAMETERS`或保存pkl。舊API以format_version及整個參數字典嚴格檢查；不monkeypatch常數。若複用舊pure predict，可在新外層adapter建立明確的單群暫存視圖，內層beta為该群gamma且只能查该群；外層新state的format／family、參數與身分不可偽裝為原v1 artifact。視圖不得獨立保存成可接受任意query的舊reference，亦不得改原state。
2. 新reference要保存原reference身分、同窗fi索引／X／y身分、原global beta、fill、anchor、finite mask、群映射、每群支持／回退原因、raw/clip/effective gamma及新殘差SHA。new spatial要保存**完整新reference身分**、新殘差與gamma映射SHA、同窗fit geo SHA及原幾何規約。單有beta SHA不足以辨識row-specific slopes。
3. 舊R8 summary的`new_reference_fit_performed=False`、`raw_targets_read_by_helper=False`、`v2_refitted=False`不能原封不動用來概括整個新流程。新流程有CPU gamma估計／新殘差／state重建，需分別記錄；沒有GPU native refit也不等於零fit。
4. fit/query row順序須用canonical fi/qi與已保存來源清單綁定；length＋maxdate相同只是弱檢查。新函式應拒絕不同期reference、shuffle未同步y/geo、錯schema或hash。同窗沿用原global fit artifact不叫後期模型回測，前提是可驗證對應。
5. 明確核驗source閉包，包括直接匯入的`matrices`、group helper、`round6_geo` schema與其固定來源。複用純幾何helper可行，但每個被複用函式的舊版假設都需檢查；不要只hash新driver後漏掉實際執行的舊依賴。
6. state存tuple key、float64、固定排序及可重載內容；將run時間等非決定欄位與內容身分分開，區分輸入pkl文件SHA與記憶體pickle身分，不能宣稱跨Python版本pickle bytes必相同。
7. 新fit/predict前後hash原state與frames，禁止共享array就地覆寫。新目錄拒絕覆寫；保存失敗／已完成階段，不讓重跑默默取代現場。原v22與selector保留。

## 必要 synthetic 不變量（尚未執行）

| 測項 | 必須確認的內容 |
| --- | --- |
| 全global退化對照 | forced gamma=beta_t，原r/fill/groups完全一致；reference、完整diag與spatial應重播原API。若無法逐位一致，事前明列浮點容差及最大差，不把allclose稱bitwise。fallback未動值宜嚴格bytes一致。 |
| 兩群已知不同斜率 | 相同非時間生成機制、不同gamma、不同案截距且有足量案內时间；新fit方向／收縮符合公式，改query y不影響fit／predict（predict根本不接y）。不是需要queryMAE才過的測試。 |
| 只有案間時間差 | 每案內日期常數，即使案數／總列數很多仍回退；防止把案間價差當時間效應。再測極小span／全缺日期／少案與單一支配案。 |
| 群標籤隔離 | 缺／未知／低支持pair回退global；不同city同一type-use按規約共享gamma；R2鄰居與R8空間group內gamma一致。測空格、missing別名及含separator字串。 |
| 全有限舊殘差轉換 | 同時含最近四年與更早的finite r、原NaN r；全部finite正確轉換、NaN mask不變，原groups/distance/fill不變。 |
| 不只改coef | 人造不同日期neighbor以使MAD可改；核center、MAD、c、G、alpha全部由r_new算，驗證若誤用舊diag或舊spatial殘差會被測試抓到。 |
| 單次時間回加 | 當地理真值與F相同，G−F=0；不同設定用手算單筆公式驗P/S。不能對P或G再次乘趨勢。 |
| 時間平移 | 同一資料改用t−k並對r同步調整，局部價／G於容差內相同；固定舊origin對照仍須過。 |
| 日期邊界 | query在anchor以前、等於、+1、+366、+730與缺日；F按0–366cap，local按凍結語義，spatial缺日不套用。不將「F cap」錯稱全系統cap。 |
| 空間資格與精確保留 | 最少4／5案、排同案、最多40案、2km邊界、extent失敗、強來源、零base等；未applied列與輸入P逐位一致，原幾何row選擇不因新gamma改變。 |
| 查詢批次與順序 | 同一query換批次、重排行、先預測其他群再預測，還原順序後一致；cache不可只依日期／位置而漏group；新state不能被query改動。 |
| 身分與序列化 | 保存重載一致、舊檔SHA不變；錯期reference、錯gamma摘要、錯residualSHA、geo/index錯位拒絕；fit/query傳入目標或ID欄拒絕。 |
| 三支與median | 各支使用自己的原TWD base；source/c/幾何來自同新state但L/G裁切仍可隨base不同；最终median不用四捨五入／nanmedian。若候選單支，另外兩支bytes不變。 |

## 評分前凍結與後續審查

本報告不指定新候選分數或採用。主線已固定上節估計式及常數，允許評分前仍須把它們連同type-use canonical化、branch範圍、完整source SHA、三期fi/qi、全global對照容差及故障處理寫入plan並完成code freeze。synthetic通過後先驗舊v22三期完整向量與原diag，再執行一次固定候選；fit-only支持盤點不作query收益搜尋。

同窗state依三期重建；原三分支base可保留同窗已核保存向量，不需重訓native。保存每群gamma／支持／fallback及每支base→reference→spatial→median診斷，特別區分回退base趨勢、reference confidence與spatial權重變動。整體、加權、新舊案、季度、城市、型態／用途、面積、真價尾部及paired區間都要留存；真價分群只用於事後診斷。不能只因2/3期改善、廠辦改善或某一個CI未跨零就稱Private較佳。

所有窗口已反覆使用；若結果出現跨期支持不穩、最新新案／主體價帶退步，應保留反證及不採用可能。舊v22使用者選擇與未知Private維持，沒有本審查授權的test推論、提交檔或Kaggle動作。

## 已讀來源身分

以下SHA為本次審查所讀bytes；這是來源快照，不是新候選plan凍結或測試通過證據。

| 來源 | SHA256 |
| --- | --- |
| [src/round2_neighborhood.py](../src/round2_neighborhood.py) | `cae244c40e6616d6c72f3aff7c3423a50a5bfbfbf35df33a7d044186eb491ae9` |
| [src/round3_reference.py](../src/round3_reference.py) | `4d78963763458e5fd2dab2847a45cd7f356c58dc2288081c1f4006cd8dba3631` |
| [src/round8_spatial_reference.py](../src/round8_spatial_reference.py) | `de86397255d654f363f3b746d579923fe689aa41d9dc16335d0922fea05e1c32` |
| [reports/optimization_round3_report.md](optimization_round3_report.md) | `953882f4a2aab72e355b3131cbd6a758971f18a349214fb3e133dd6aaf13a27c` |
| [reports/round3_overfitting_controls.md](round3_overfitting_controls.md) | `e121fc6180d7470bd5602dcf8c7c4ef0d8db1686ffd838af55299734d82e71f7` |
| [reports/round45_regional_plan.md](round45_regional_plan.md) | `b24f88f538d9b1ba351ce5e0bc94c38a9ed94fe5a61b1b413c5474e28efd84c6` |
| [reports/round45_regional_report.md](round45_regional_report.md) | `451b9fe005a1279f84937f1c276bfef531a3ba46458008032e6a5a30f605789f` |
| [reports/round81_reference_audit.md](round81_reference_audit.md) | `5c36d821308e792f64f7cb71e55ace6284b6c79f318c1de1c6835365b38b5e05` |
| [reports/reference_blending_audit_r84.md](reference_blending_audit_r84.md) | `570dc019b6a6a003adee8f1a0782a0409c473227f5ff284d0355616de68fd2a3` |
| [reports/r85_error_audit.md](r85_error_audit.md) | `168ad4d36d65c8c12ac179f5a2a05abd783101904db4ea39d607b23145181ddc` |
| [reports/round88_report.md](round88_report.md) | `5661c7e7af8d67c4800c63e9ab81bd1d177fd71bf3321e340deb24bbf5b49953` |
