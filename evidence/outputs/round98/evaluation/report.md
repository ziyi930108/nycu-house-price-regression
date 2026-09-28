# R98：較早 forward 預測學權重的三窗探索

Earlier forward blocks fit a new final three-branch stacker; historically reused data, not a blind holdout. Offset overlaps other windows. Missing pre-auxiliary block prevents the original three-August-window adoption gate. Bootstrap is conditional, not Private win probability.

只改最後組合；三窗各自學一組權重。負差代表 R98 較低。沒有新 CSV 或 Private 成績。

|窗口|v22 overall|R98 overall|差|weighted 差|研究門檻|
|---|---:|---:|---:|---:|---|
|development|1,407,387.07|1,416,185.71|8,798.64|8,728.33|False|
|offset|1,336,410.67|1,332,155.65|-4,255.03|-5,807.86|False|
|confirmation|1,383,228.41|1,388,419.07|5,190.67|5,002.34|False|

|窗口|權重 N / H / U|meta 列數|query 列數|
|---|---|---:|---:|
|development|0.228046465745 / 0.269965391162 / 0.501988143093|105511|81234|
|offset|0.259076767768 / 0.270462351656 / 0.470460880576|137312|107169|
|confirmation|0.331823933782 / 0.271627421448 / 0.39654864477|186745|133697|

## 全部配對區間

|窗口|抽樣|指標|點差|2.5%|97.5%|有效／無效|
|---|---|---|---:|---:|---:|---|
|development|project|overall|8,798.64|-8,571.79|30,064.02|2000 / 0|
|development|project|weighted|8,728.33|-8,525.89|29,238.51|2000 / 0|
|development|month|overall|8,798.64|-2,916.80|26,010.36|2000 / 0|
|development|month|weighted|8,728.33|-3,857.08|25,427.47|2000 / 0|
|offset|project|overall|-4,255.03|-19,833.84|8,738.66|2000 / 0|
|offset|project|weighted|-5,807.86|-23,737.39|9,075.75|2000 / 0|
|offset|month|overall|-4,255.03|-11,269.98|2,254.97|2000 / 0|
|offset|month|weighted|-5,807.86|-12,186.82|1,312.29|2000 / 0|
|confirmation|project|overall|5,190.67|-11,456.67|24,202.14|2000 / 0|
|confirmation|project|weighted|5,002.34|-13,329.07|26,446.20|2000 / 0|
|confirmation|month|overall|5,190.67|-5,539.87|17,999.12|2000 / 0|
|confirmation|month|weighted|5,002.34|-5,870.98|20,244.79|2000 / 0|

## development 正反分群

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|28981|3,995.54|
|strata/new_nonmissing|52253|11,462.58|
|forecast_quarters/Q1|17608|-3,366.58|
|forecast_quarters/Q2|14193|28,676.14|
|forecast_quarters/Q3|22487|12,275.90|
|forecast_quarters/Q4|26946|3,376.35|
|<10m|22139|11,412.71|
|10-20m|43914|-994.23|
|20-50m|14197|4,270.26|
|50-100m|802|271,528.47|
|>=100m|182|1,249,191.87|

季度或 ≥500 列分群 >5% 退步項：[{"section": "months", "group": "2022-12", "rows": 6122, "control_mae": 1229903.2131515937, "candidate_mae": 1307758.728681816, "delta": 77855.51553022233, "relative_change_pct": 6.330214824849484}, {"section": "cities", "group": "基隆市", "rows": 737, "control_mae": 1576679.1173530903, "candidate_mae": 1783975.1020141186, "delta": 207295.98466102825, "relative_change_pct": 13.147633045906906}, {"section": "relative_forecast_months", "group": "4", "rows": 5881, "control_mae": 1220773.8640789841, "candidate_mae": 1312068.1589271505, "delta": 91294.2948481664, "relative_change_pct": 7.478395265043098}]

城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。

## offset 正反分群

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|47817|6,493.80|
|strata/new_nonmissing|59352|-12,914.83|
|forecast_quarters/Q1|22487|3,646.14|
|forecast_quarters/Q2|27820|-5,490.14|
|forecast_quarters/Q3|27006|-14,752.01|
|forecast_quarters/Q4|29856|439.80|
|<10m|25432|17,917.38|
|10-20m|59481|-3,215.86|
|20-50m|20803|-2,692.44|
|50-100m|1183|-18,040.17|
|>=100m|270|-2,381,656.00|

季度或 ≥500 列分群 >5% 退步項：[{"section": "cities", "group": "嘉義縣", "rows": 1532, "control_mae": 702698.0320022947, "candidate_mae": 778373.7294748088, "delta": 75675.69747251412, "relative_change_pct": 10.769305452141497}]

城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。

## confirmation 正反分群

|分群|列數|MAE 差|
|---|---:|---:|
|strata/seen|57601|6,848.56|
|strata/new_nonmissing|76096|3,935.73|
|forecast_quarters/Q1|26579|18,537.78|
|forecast_quarters/Q2|31157|124.08|
|forecast_quarters/Q3|37236|-12,248.03|
|forecast_quarters/Q4|38725|16,874.45|
|<10m|25585|5,746.29|
|10-20m|74109|-7,769.15|
|20-50m|31959|-20,060.14|
|50-100m|1725|149,259.87|
|>=100m|319|4,722,101.06|

季度或 ≥500 列分群 >5% 退步項：[{"section": "cities", "group": "屏東縣", "rows": 1013, "control_mae": 756737.25172707, "candidate_mae": 794970.7035279837, "delta": 38233.45180091378, "relative_change_pct": 5.052407782708618}]

城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。

全部探索門檻通過：False
仍缺 auxiliary 之前的 forward meta block；即使通過，也只值得另立更早窗口研究。沒有全量候選或採用決策。
CI 條件於固定模型，不涵蓋重訓／反覆選型；12 個區間全部呈現，不是 Private 勝率。
meta-y 在評分器只核檔案 bytes，不反序列化；raw pickle 於三窗完整封存核驗後才反序列化，未保存逐列真價。
