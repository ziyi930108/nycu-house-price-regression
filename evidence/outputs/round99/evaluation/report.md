# R99：較早 forward 預測學權重的三窗探索

Earlier forward blocks fit a four-column N/H/U/median stacker after observing R98; historically reused data, not a blind holdout. Offset overlaps other windows. Missing pre-auxiliary block prevents the original three-August-window adoption gate. Bootstrap is conditional, not Private win probability.

只改最後四欄 [N,H,U,M=median(N,H,U)] 組合；三窗各自學一組權重。負差代表 R99 較低。沒有新 CSV 或 Private 成績。

|窗口|v22 overall|R99 overall|差|weighted 差|研究門檻|
|---|---:|---:|---:|---:|---|
|development|1,407,387.07|1,417,836.89|10,449.82|10,336.51|False|
|offset|1,336,410.67|1,330,129.14|-6,281.53|-7,487.25|False|
|confirmation|1,383,228.41|1,386,049.92|2,821.52|2,732.32|False|

|窗口|權重 N / H / U / M|meta 列數|query 列數|
|---|---|---:|---:|
|development|0.181671907326 / 0.269222687355 / 0.473598723988 / 0.0755066813306|105511|81234|
|offset|0.200129741103 / 0.269518295634 / 0.406423003316 / 0.123928959946|137312|107169|
|confirmation|0.278766331011 / 0.251444670194 / 0.319509013231 / 0.150279985564|186745|133697|

## 固定 R98 對照（只報點差）

以下不是另一項選型門檻，也沒有額外 bootstrap。R98 預測原樣讀取，各窗控制 MAE 需匹配原結果。

|窗口|R98 overall|R99 overall|R99-R98 差|weighted 差|
|---|---:|---:|---:|---:|
|development|1,416,185.71|1,417,836.89|1,651.18|1,608.18|
|offset|1,332,155.65|1,330,129.14|-2,026.51|-1,679.39|
|confirmation|1,388,419.07|1,386,049.92|-2,369.15|-2,270.02|

## 全部配對區間

|窗口|抽樣|指標|點差|2.5%|97.5%|有效／無效|
|---|---|---|---:|---:|---:|---|
|development|project|overall|10,449.82|-6,867.65|31,136.90|2000 / 0|
|development|project|weighted|10,336.51|-6,884.78|30,405.41|2000 / 0|
|development|month|overall|10,449.82|-2,264.59|27,533.77|2000 / 0|
|development|month|weighted|10,336.51|-3,670.53|26,968.19|2000 / 0|
|offset|project|overall|-6,281.53|-21,606.26|6,248.27|2000 / 0|
|offset|project|weighted|-7,487.25|-25,078.69|6,636.31|2000 / 0|
|offset|month|overall|-6,281.53|-15,651.16|963.40|2000 / 0|
|offset|month|weighted|-7,487.25|-16,344.13|350.41|2000 / 0|
|confirmation|project|overall|2,821.52|-11,487.78|20,457.83|2000 / 0|
|confirmation|project|weighted|2,732.32|-13,338.09|21,991.51|2000 / 0|
|confirmation|month|overall|2,821.52|-6,130.28|13,891.84|2000 / 0|
|confirmation|month|weighted|2,732.32|-6,744.63|15,872.71|2000 / 0|

## development 正反分群

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|28981|2,709.78|
|strata/new_nonmissing|52253|14,742.67|
|forecast_quarters/Q1|17608|-6,212.33|
|forecast_quarters/Q2|14193|31,527.00|
|forecast_quarters/Q3|22487|14,261.25|
|forecast_quarters/Q4|26946|7,055.28|
|<10m|22139|9,728.78|
|10-20m|43914|1,012.52|
|20-50m|14197|12,849.21|
|50-100m|802|304,830.06|
|>=100m|182|890,865.42|

季度或 ≥500 列分群 >5% 退步項：[{"section": "months", "group": "2022-12", "rows": 6122, "control_mae": 1229903.2131515937, "candidate_mae": 1313368.8888412071, "delta": 83465.67568961345, "relative_change_pct": 6.786361300393298}, {"section": "cities", "group": "基隆市", "rows": 737, "control_mae": 1576679.1173530903, "candidate_mae": 1788256.8458456832, "delta": 211577.72849259293, "relative_change_pct": 13.419200277592758}, {"section": "relative_forecast_months", "group": "4", "rows": 5881, "control_mae": 1220773.8640789841, "candidate_mae": 1317057.5128349836, "delta": 96283.64875599951, "relative_change_pct": 7.887099452988449}]

城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。

### 同窗 R99 − R98 點差

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|28981|-1,285.77|
|strata/new_nonmissing|52253|3,280.09|
|forecast_quarters/Q1|17608|-2,845.75|
|forecast_quarters/Q2|14193|2,850.86|
|forecast_quarters/Q3|22487|1,985.36|
|forecast_quarters/Q4|26946|3,678.94|
|<10m|22139|-1,683.93|
|10-20m|43914|2,006.75|
|20-50m|14197|8,578.95|
|50-100m|802|33,301.59|
|>=100m|182|-358,326.44|

相對 R98 的季度或 ≥500 列群 >5% 退步（描述用、不新增門檻）：[]

## offset 正反分群

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|47817|2,064.52|
|strata/new_nonmissing|59352|-13,005.54|
|forecast_quarters/Q1|22487|2,665.74|
|forecast_quarters/Q2|27820|-3,642.76|
|forecast_quarters/Q3|27006|-20,709.30|
|forecast_quarters/Q4|29856|-2,428.76|
|<10m|25432|11,174.01|
|10-20m|59481|-4,906.95|
|20-50m|20803|-1,217.49|
|50-100m|1183|-19,982.93|
|>=100m|270|-2,283,428.37|

季度或 ≥500 列分群 >5% 退步項：[{"section": "cities", "group": "嘉義縣", "rows": 1532, "control_mae": 702698.0320022947, "candidate_mae": 745423.8750342249, "delta": 42725.84303193027, "relative_change_pct": 6.080256537816908}]

城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。

### 同窗 R99 − R98 點差

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|47817|-4,429.28|
|strata/new_nonmissing|59352|-90.71|
|forecast_quarters/Q1|22487|-980.40|
|forecast_quarters/Q2|27820|1,847.37|
|forecast_quarters/Q3|27006|-5,957.29|
|forecast_quarters/Q4|29856|-2,868.56|
|<10m|25432|-6,743.38|
|10-20m|59481|-1,691.09|
|20-50m|20803|1,474.95|
|50-100m|1183|-1,942.76|
|>=100m|270|98,227.63|

相對 R98 的季度或 ≥500 列群 >5% 退步（描述用、不新增門檻）：[]

## confirmation 正反分群

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|57601|3,606.71|
|strata/new_nonmissing|76096|2,227.16|
|forecast_quarters/Q1|26579|11,120.62|
|forecast_quarters/Q2|31157|-2,113.75|
|forecast_quarters/Q3|37236|-11,960.19|
|forecast_quarters/Q4|38725|15,309.53|
|<10m|25585|3,289.39|
|10-20m|74109|-8,217.36|
|20-50m|31959|-18,912.48|
|50-100m|1725|128,082.46|
|>=100m|319|4,029,877.65|

季度或 ≥500 列分群 >5% 退步項：[]

城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。

### 同窗 R99 − R98 點差

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|57601|-3,241.84|
|strata/new_nonmissing|76096|-1,708.57|
|forecast_quarters/Q1|26579|-7,417.15|
|forecast_quarters/Q2|31157|-2,237.83|
|forecast_quarters/Q3|37236|287.84|
|forecast_quarters/Q4|38725|-1,564.92|
|<10m|25585|-2,456.90|
|10-20m|74109|-448.21|
|20-50m|31959|1,147.66|
|50-100m|1725|-21,177.41|
|>=100m|319|-692,223.41|

相對 R98 的季度或 ≥500 列群 >5% 退步（描述用、不新增門檻）：[]

全部探索門檻通過：False
仍缺 auxiliary 之前的 forward meta block；即使通過，也只值得另立更早窗口研究。沒有全量候選或採用決策。
CI 條件於固定模型，不涵蓋重訓／反覆選型；12 個區間全部呈現，不是 Private 勝率。
第四欄納入 median 只保證精確 meta L1 最優不劣於可行的 median 解；不是 query、群加權 MAE 或 Private 改善保證。
meta-y 在評分器只核檔案 bytes，不反序列化；raw pickle 於三窗完整封存核驗後才反序列化，未保存逐列真價。
