# R100 時間前置 prior：完整三窗口評估

六次 NN 訓練與全部預測封存後才讀取 query 真價。C 只作相同 loss rows 的機制控制；T 是唯一研究候選。

|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T weighted−v22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,405,993.40|1,402,289.26|-5,097.81|-3,704.14|-4,948.79|
|confirmation|1,383,228.41|1,382,132.85|1,393,291.76|+10,063.35|+11,158.91|+11,056.78|
|auxiliary|2,309,352.36|2,302,648.68|2,297,675.88|-11,676.48|-4,972.80|-10,303.47|

## 固定配對區間

|窗口|控制|抽樣|指標|點差|95% 區間|有效次數|
|---|---|---|---|---:|---|---:|
|development|v22|project|overall|-5,097.81|-11859.809244867618 ～ 1407.9658558314136|2000|
|development|v22|project|weighted|-4,948.79|-11719.14652561586 ～ 1436.60305137958|2000|
|development|v22|month|overall|-5,097.81|-13388.798825735226 ～ 1067.4260967637688|2000|
|development|v22|month|weighted|-4,948.79|-17763.485521795472 ～ 1321.3948463059612|2000|
|development|C|project|overall|-3,704.14|-10845.494056214724 ～ 3002.642757030914|2000|
|development|C|project|weighted|-3,570.11|-10710.646349895023 ～ 3039.1735427884314|2000|
|development|C|month|overall|-3,704.14|-12204.748168817256 ～ 2336.9504062017527|2000|
|development|C|month|weighted|-3,570.11|-16429.548563964898 ～ 2074.684555803442|2000|
|confirmation|v22|project|overall|+10,063.35|-962.0070759051188 ～ 21018.28309130243|2000|
|confirmation|v22|project|weighted|+11,056.78|-1204.395027348696 ～ 23241.970745280592|2000|
|confirmation|v22|month|overall|+10,063.35|2720.938375597814 ～ 17887.69515118003|2000|
|confirmation|v22|month|weighted|+11,056.78|2572.458117101173 ～ 18036.460401862896|2000|
|confirmation|C|project|overall|+11,158.91|539.3168419319904 ～ 21378.45496189072|2000|
|confirmation|C|project|weighted|+12,171.04|507.1719884905499 ～ 23457.43010196284|2000|
|confirmation|C|month|overall|+11,158.91|5358.145394432353 ～ 17789.98417503465|2000|
|confirmation|C|month|weighted|+12,171.04|5764.281434130971 ～ 19172.679222903455|2000|
|auxiliary|v22|project|overall|-11,676.48|-18308.52287676786 ～ -5571.88071412767|2000|
|auxiliary|v22|project|weighted|-10,303.47|-16160.55012552474 ～ -4835.099731353648|2000|
|auxiliary|v22|month|overall|-11,676.48|-18130.881995062507 ～ -4436.9164329301475|2000|
|auxiliary|v22|month|weighted|-10,303.47|-18208.802071110997 ～ -3486.5523330131487|2000|
|auxiliary|C|project|overall|-4,972.80|-12700.505811443518 ～ 3116.0498608736993|2000|
|auxiliary|C|project|weighted|-5,252.08|-12106.142269717238 ～ 1811.256136570211|2000|
|auxiliary|C|month|overall|-4,972.80|-9820.745766570233 ～ 335.18951306481335|2000|
|auxiliary|C|month|weighted|-5,252.08|-11460.087204790208 ～ 187.55616235439126|2000|

## 固定門檻與反證

{"components": {"all_three_overall_improve": false, "all_three_weighted_improve": false, "confirmation_at_least_half_percent": false, "no_quarter_or_large_group_deterioration_above_5pct": false}, "adoption_research_gate_passed": false, "confirmation_fractional_improvement": -0.007275265775682378, "C_is_mechanism_control_not_alternative_candidate": true, "automatic_adoption": false, "new_CSV": false, "selected_submission": "v22", "Private_score": null}

### development

T 對 v22 的季度或至少 500 列固定群退步超過 5%：
[]

T 對 C 的相同反證（機制描述，不另改採用門檻）：
[]

### confirmation

T 對 v22 的季度或至少 500 列固定群退步超過 5%：
[]

T 對 C 的相同反證（機制描述，不另改採用門檻）：
[]

### auxiliary

T 對 v22 的季度或至少 500 列固定群退步超過 5%：
[{"section": "cities", "group": "嘉義市", "rows": 731, "control_mae": 2112509.041619518, "candidate_mae": 2241973.661560857, "delta": 129464.61994133936, "relative_change_pct": 6.128476488890556}]

T 對 C 的相同反證（機制描述，不另改採用門檻）：
[{"section": "cities", "group": "嘉義市", "rows": 731, "control_mae": 2131437.675296868, "candidate_mae": 2241973.661560857, "delta": 110535.986263989, "relative_change_pct": 5.185982566841579}]

Adaptive research after R99 on repeatedly used historical windows; not blind validation. Bootstrap is conditional on saved models, not Private improvement probability. C isolates the loss-row mask; T also excludes date-unidentifiable prior references. Neither is an automatically selected submission; v22 remains selected.

v22 為唯一提交；Private 未知。完整城市、月份、seen/new/missing、真價帶、尾部及所有層級均保存在 results.json。
原始價格 pickle 必須反序列化全表，之後僅依各期封存 raw-cache query 索引取價；未保存逐列真價、未載模型／test，未新增 CSV。
