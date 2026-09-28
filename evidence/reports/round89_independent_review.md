# R89 獨立結果審查：不符合事前門檻，保留v22

2026-09-28。本次只讀已保存source、plan、metadata、結果與診斷，核對來源／產物SHA及彙總數字。沒有新fit、predict、query評分、bootstrap或test讀取；未修改任何source、模型、selector、CSV或root handoff。

**建議R89保留研究，不進final fit／test候選，不新增CSV。** 開發期改善是真實保存結果，但新建案三期皆退步，最新及最舊年度整體／加權皆退步，已直接違反事前門檻。不能將開發期4個有利區間推成Private勝率。

## 事前門檻與完整结果

固定計畫SHA為`34811ae50be55f3165e6debe8c24559dee2436939e45a90d29282697d61de391`。門檻是三期整體與固定加權同時優於v22，且最新期整體至少改善0.5%；通過亦不自動發布CSV。這份審查不降低門檻、不改為挑2/3期、不搜尋群別例外。

單位TWD元；差值＝R89−對照，負值較好。

|期別|指標|v22|v24|R89|差vs v22|差vs v24|
|---|---|---:|---:|---:|---:|---:|
|development|整體|1,407,387.07|1,407,340.72|1,400,980.40|-6,406.67|-6,360.31|
|development|固定加權|1,399,717.74|1,399,695.92|1,393,077.42|-6,640.32|-6,618.50|
|confirmation|整體|1,383,228.41|1,373,195.92|1,393,135.44|9,907.03|19,939.52|
|confirmation|固定加權|1,447,153.42|1,436,088.88|1,459,616.05|12,462.63|23,527.17|
|auxiliary|整體|2,309,352.36|2,275,409.14|2,320,127.94|10,775.59|44,718.80|
|auxiliary|固定加權|2,023,165.66|1,992,570.98|2,033,324.25|10,158.59|40,753.27|

最新期若只看0.5%門檻，R89整體MAE需不高於1,376,312.26元；實際1,393,135.44元，反而比v22高約0.7162%。另外最舊年度兩指標也退步；不是差一個尾數就能通過。

## 核驗範圍與來源保全

獨立讀取並重新hash了plan/result繼承的286項input、52項source、32項protected以及全部各期artifact，合併去重400個檔案，全部與保存SHA一致。原protected_before／after完全相同；text plan與machine plan身分一致。這是文件與產物身分核驗，沒有在本審查重新訓練或推論。

driver原執行的assert與完成收據表明：canonical fi/qi跨R46／R71／R79／R83逐位一致；同窗X、fit y、fit/query geo與reference residual身分核對；9支舊reference／spatial預測及全部diag重播精確、原v22三分支median精確。新wrapper保存重載後，18次wrapper query呼叫含3期×3支×原／重載，其全部價格與diag一致；模型推論前後值身分不變。每支不在affected mask列與原分支逐位相同，最終非union列保留v22。

本審查在source階段提出的call計數、affected語義、pickle值身分及實際spatial測路缺口均在正式freeze前修正；未變更斜率規則。23項synthetic exit0收據與5個helper source SHA已獨立讀取核對。synthetic曾遇到Pandas dtype與pickle bytes兩次工程失敗，收據保留；neutral合成用unit1取得精確log0，未對真fit添加epsilon clamp。

generic診斷中R89/v22/v24的三期完整metrics與主結果JSON逐項相同；兩組paired的候選向量SHA與主結果相同、point_metrics_reproduced均true。24個paired點差與保存直接MAE差最大差2.71e-8元以內，屬聚合順序舍入，不是新的分數。本審查未重算逐列價格誤差。

## 實際改了哪些群，沒有改哪些模型

R89是CPU時間正規化wrapper：固定type/use、案內中心化global殘差、一維delta、每輪等案初始權重×Huber、原global prior、先clip後強收縮；全部finite reference殘差與query還原一致，新spatial亦由新殘差build。三支原TWD base、原生模型與median規則不變。不是訓練新的CatBoost或把既有預測整體乘新倍率。

|期別|fit列|支持群／群斜率fit|重算reference殘差列|affected／最終改價列|query占比|wrapper fit與驗證秒|
|---|---:|---:|---:|---:|---:|---:|
|development|240,316|3|214,044|71,090|87.51%|7.188|
|confirmation|321,550|6|301,379|127,767|95.56%|10.578|
|auxiliary|134,805|2|119,471|81,983|77.70%|3.672|

支持群數3／6／2與獨立fit-only支持盤點的keys、rows、physical projects、R2 project數與日期span精確相同。主要是住宅大樓、華廈住家與部分透天；最新年度另支持住商／其他登記用途。**廠辦未獲足夠支持，維持global fallback，不能宣稱已解決廠辦誤差。** affected表示支持且delta非零的群，不是每列有獨立可靠成交，也不是重新訓練原生模型。

有效斜率是年化log-price係數，不能直接當因果市場漲幅：

|期別|型態／用途|global時間beta|effective gamma|delta|獨立案數|
|---|---|---:|---:|---:|---:|
|development|住宅大樓(11層含以上有電梯)／住家用|0.083895|0.077504|-0.006391|855|
|development|華廈(10層含以下有電梯)／住家用|0.083895|0.082776|-0.001119|257|
|development|透天厝／住家用|0.083895|0.085273|+0.001378|194|
|confirmation|住宅大樓(11層含以上有電梯)／住商用|0.078221|0.076213|-0.002008|45|
|confirmation|住宅大樓(11層含以上有電梯)／住家用|0.078221|0.070991|-0.007230|1291|
|confirmation|住宅大樓(11層含以上有電梯)／見其他登記事項|0.078221|0.075204|-0.003017|53|
|confirmation|華廈(10層含以下有電梯)／住家用|0.078221|0.066577|-0.011644|632|
|confirmation|華廈(10層含以下有電梯)／見其他登記事項|0.078221|0.072278|-0.005943|54|
|confirmation|透天厝／住家用|0.078221|0.068417|-0.009803|460|
|auxiliary|住宅大樓(11層含以上有電梯)／住家用|0.031601|0.027609|-0.003991|425|
|auxiliary|華廈(10層含以下有電梯)／住家用|0.031601|0.031562|-0.000039|55|

最新及最舊支持群斜率都低於global，但下游價格還經殘差中心、MAD/confidence、spatial與median，因此不能把MAE差全歸因於base外推幅度，亦不能假設每筆價格都只變低。

## 最強反證：新建案三期退步

|期別|群|列數|R89 MAE|差vs v22|差vs v24|
|---|---|---:|---:|---:|---:|
|development|new_nonmissing|52,253|1,700,399.24|2,445.79|3,421.57|
|development|seen|28,981|861,125.59|-22,367.73|-23,997.14|
|confirmation|new_nonmissing|76,096|1,836,122.88|26,936.05|43,845.47|
|confirmation|seen|57,601|807,909.96|-12,589.81|-11,642.35|
|auxiliary|new_nonmissing|90,424|2,503,875.99|11,170.88|47,259.42|
|auxiliary|seen|15,087|1,218,833.22|8,406.38|29,491.60|

三期缺建案名群皆0列，MAE是null／不適用，不是0元。開發期seen改善22,367.73元，抵過new退步2,445.79元；最新seen改善12,589.81元，卻不足抵銷76,096筆new退步26,936.05元。最舊seen與new都退步。保留seen收益，但不能藉它忽略新案泛化代價。

|期別|預測季度|列數|差vs v22|差vs v24|
|---|---|---:|---:|---:|
|development|Q1|17,608|192.35|-14,280.65|
|development|Q2|14,193|-2,209.81|10,190.85|
|development|Q3|22,487|-8,622.25|-8,496.30|
|development|Q4|26,946|-11,080.45|-8,120.04|
|confirmation|Q1|26,579|-4,833.48|-7,715.20|
|confirmation|Q2|31,157|-828.07|10,398.09|
|confirmation|Q3|37,236|7,795.26|22,856.67|
|confirmation|Q4|38,725|30,691.94|43,792.18|
|auxiliary|Q1|33,694|3,228.34|27,134.77|
|auxiliary|Q2|29,169|9,511.85|41,490.68|
|auxiliary|Q3|25,213|14,433.83|54,864.64|
|auxiliary|Q4|17,435|22,185.04|69,429.46|

最新Q1/Q2相對v22略改善，但Q3/Q4退步，Q4的38,725筆增加30,691.94元MAE；最舊四季皆退步。開發Q2–Q4改善、Q1微退步，不能稱所有時間距離都受益。

## 價帶、城市與分支反證

|期別|真價帶|列數|差vs v22|差vs v24|
|---|---|---:|---:|---:|
|development|<10m|22,139|-7,106.03|-9,030.87|
|development|10-20m|43,914|-6,105.30|-4,967.04|
|development|20-50m|14,197|-6,062.69|-1,920.09|
|development|50-100m|802|-25,844.17|-34,234.39|
|development|>=100m|182|64,770.57|-241,213.87|
|confirmation|<10m|25,585|-10,208.00|-15,429.10|
|confirmation|10-20m|74,109|5,621.77|8,190.73|
|confirmation|20-50m|31,959|34,050.44|72,922.17|
|confirmation|50-100m|1,725|25,611.24|218,315.32|
|confirmation|>=100m|319|115,017.62|-794,709.97|
|auxiliary|<10m|41,132|233.28|4,791.01|
|auxiliary|10-20m|47,793|11,929.47|48,470.39|
|auxiliary|20-50m|15,033|27,470.52|121,075.13|
|auxiliary|50-100m|1,382|66,655.41|22,572.59|
|auxiliary|>=100m|171|304,798.31|2,066,660.99|

最新1,000萬以下25,585筆改善10,208.00元，是重要有利反證；但1,000–2,000萬的74,109筆與2,000–5,000萬的31,959筆都退步，後者增加34,050.44元。最新≥1億元319筆相對v22增加115,017.62元，**相對v24卻改善794,709.97元**，兩個對照不能混稱；最舊五個價帶全部比v22差。真價帶只作診斷，不能在未知test分流。

城市反證亦不單一。最新臺南9,822筆改善25,544.38元，臺北4,957筆退步35,697.32元、新北21,822筆退步22,348.57元、桃園26,327筆退步13,382.32元；新竹市2,131筆退步53,977.86元。開發臺北改善50,157.03元而新竹市退步54,045.79元。完整城市指標在主結果／generic JSON保存，不依收益挑城市套斜率。

|期別|原base不變的分支|完整spatial分支MAE差vs原分支|
|---|---|---:|
|development|log|-6,492.48|
|development|nn|-10,696.85|
|development|unit|-4,026.53|
|confirmation|log|11,185.28|
|confirmation|nn|4,548.94|
|confirmation|unit|9,682.69|
|auxiliary|log|10,414.26|
|auxiliary|nn|11,620.00|
|auxiliary|unit|9,203.20|

開發三支自身均改善；最新與最舊三支自身皆退步，所以此次退步不是只到最外median才出現。以上不是可相加的因果分解；base、reference、spatial与median有裁切／權重互動，不能把差額直接歸給單一層。

## 兩對照的24個配對區間

每項2,000次、seed20260925，建案抽法與3個月循環區塊，固定overall／weighted。負差代表R89較低；所有24項皆2000有效、0無效replicate。

|對照|期別|抽法|指標|點差|2.5%|97.5%|
|---|---|---|---|---:|---:|---:|
|v22|development|project|overall|-6,406.67|-10,601.91|-2,612.40|
|v22|development|project|weighted|-6,640.32|-10,916.87|-2,715.32|
|v22|development|month|overall|-6,406.67|-9,398.38|-1,708.05|
|v22|development|month|weighted|-6,640.32|-13,238.76|-408.61|
|v22|confirmation|project|overall|9,907.03|5,640.20|13,926.63|
|v22|confirmation|project|weighted|12,462.63|8,058.90|16,716.20|
|v22|confirmation|month|overall|9,907.03|-2,614.87|21,863.74|
|v22|confirmation|month|weighted|12,462.63|952.80|20,730.71|
|v22|auxiliary|project|overall|10,775.59|9,139.59|12,495.51|
|v22|auxiliary|project|weighted|10,158.59|8,712.01|11,599.06|
|v22|auxiliary|month|overall|10,775.59|6,518.56|16,699.91|
|v22|auxiliary|month|weighted|10,158.59|6,180.29|17,181.04|
|v24|development|project|overall|-6,360.31|-20,900.43|8,500.91|
|v24|development|project|weighted|-6,618.50|-20,847.72|8,134.14|
|v24|development|month|overall|-6,360.31|-10,691.30|-13.54|
|v24|development|month|weighted|-6,618.50|-11,898.35|-156.65|
|v24|confirmation|project|overall|19,939.52|6,733.50|32,708.46|
|v24|confirmation|project|weighted|23,527.17|9,124.51|37,234.92|
|v24|confirmation|month|overall|19,939.52|268.48|35,132.53|
|v24|confirmation|month|weighted|23,527.17|5,969.67|32,729.74|
|v24|auxiliary|project|overall|44,718.80|27,189.43|65,983.71|
|v24|auxiliary|project|weighted|40,753.27|26,601.77|57,706.99|
|v24|auxiliary|month|overall|44,718.80|33,450.36|59,222.08|
|v24|auxiliary|month|weighted|40,753.27|31,391.54|56,179.29|

相對v22：開發4區間全負；最新3正、1跨零；最舊4全正。合計4負／7正／1跨零。相對v24：開發2負、2跨零，最新與最舊共8個全正。這支持保留開發期改善的證據，同時明確反對跨期替代；不是Private勝率、p值、新盲測或校正多輪選型後的顯著性。

固定city×seen/new test組成估計的R89−v22三期差為：
- development：-8,500.54元，涵蓋99.6580%，未支持161筆。
- confirmation：13,546.72元，涵蓋99.6580%，未支持161筆。
- auxiliary：11,421.98元，涵蓋94.7847%，未支持2455筆。

這沿用既有無標籤test組成，在歷史cell至少200筆的有支持質量內重正規化；不是完整test／Private MAE。本輪未讀新test價或搜尋Public。

## 實際執行時間與計數

主流程outer execution為226.437秒、exit0，內部run為226.203秒。三次CPU wrapper fit加其validate／支持核對合計21.438秒，已包含在主流程；共有11個群內時間係數fit、3次新spatial build、3個保存wrapper重載。原生CatBoost／NN fit、load、predict均0。不要將wrapper fit秒與主流程相加當總耗時。

外部控制重播R/S各9次；18次wrapper.predict（含重載核對）內，global R/S各18次、群別R/S各66次。合計R/S各93次，不能只以外部9次當全部推論成本。

generic內部21.438秒、outer execution21.579秒，exit0；paired v22/v24各6.078／6.094秒。部分程序並行，不把以上逐項相加當整段牆鐘；未量測本輪所有聊天與開發時間。

## 收據與限制

本輪未通過事前門檻，無理由重新用同一query搜尋例外群、比例或支持門檻。v22仍是使用者選定版本，Private未知。這不是證明v22是所有可能模型中最好；R89工程正確性、可追溯產物與泛化收益是不同問題。較新的時間切點若要研究，須重建全部受影響fit state，不用後期模型回測。

主要證據身分：

|檔案|SHA256|
|---|---|
|[reports/round89_plan.md](round89_plan.md)|`34811ae50be55f3165e6debe8c24559dee2436939e45a90d29282697d61de391`|
|[src/round89_experiment.py](../src/round89_experiment.py)|`1e111dbc2dc234cb41b593910de33b41cadf82a20807ac82008ac810ca5129d1`|
|[src/round89_group_trend.py](../src/round89_group_trend.py)|`233b7919d738fa6eff199138fd2a301cf67ed55f2b0fc6e64c92c10f59de71b0`|
|[reports/round89_helper_selfcheck.json](round89_helper_selfcheck.json)|`5341e48070f3cf54757e85d2162cb73f913d29b61c2f1ca1f171705e9a08f224`|
|[outputs/round89/plan.json](../outputs/round89/plan.json)|`85ba279fcf55cca7bded6d581c613f2e7e847c3d22de8ec4468120fe68b27f30`|
|[outputs/round89/results.json](../outputs/round89/results.json)|`a2f0a42d4ed22e27eae07d397839318f32703628cdfdacc94e619e12ba4038d1`|
|[outputs/round89/execution.json](../outputs/round89/execution.json)|`8a8e8d34a99beb4fbd790fca3b1f1fda6dcc1b436f02ddf362ed3e393b67f47e`|
|[reports/round89_diagnostics.json](round89_diagnostics.json)|`a7ed98b459f07279c2adfb709de7f3e1e8b677513c93f723e3563f529e0e402e`|
|[outputs/round89/paired_vs_v22/results.json](../outputs/round89/paired_vs_v22/results.json)|`b46232dfaa38d976da0917f0292074655fcb59adebd7ae5dac9c4b93ff666e8f`|
|[outputs/round89/paired_vs_v24/results.json](../outputs/round89/paired_vs_v24/results.json)|`dd0faffb7440d38486be31e483498cb882c288a4b1673514d2296aafab17db9a`|
