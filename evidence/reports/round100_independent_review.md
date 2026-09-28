# R100 保存結果核驗

本作者編寫pure-X policy／audit並審查helper；helper、driver、scorer由其他作者實作。這是跨作者保存證據核驗，不聲稱完全獨立。

六NN與評分全部exit 0後才執行。沒有反序列化／重建y、NPZ或模型，没有重新建立prior、計MAE或重抽bootstrap。原始含價格檔僅按來源SHA讀位元組；prefix JSON是保存狀態，不是逐列價格表。

核 1827 個唯一檔案SHA、9保護檔、三窗相同8欄identity、factor公式、prefix合法性／支持／完整向量與24保存CI端點。六次實際fit合計 118.203秒；不把main時間當fit時間。

|窗口|T−v22 overall|T−C overall|T−v22 weighted|T−C weighted|
|---|---:|---:|---:|---:|
|development|-5,097.81|-3,704.14|-4,948.79|-3,570.11|
|confirmation|+10,063.35|+11,158.91|+11,056.78|+12,171.04|
|auxiliary|-11,676.48|-4,972.80|-10,303.47|-5,252.08|

固定門檻：{"C_is_mechanism_control_not_alternative_candidate": true, "Private_score": null, "adoption_research_gate_passed": false, "automatic_adoption": false, "components": {"all_three_overall_improve": false, "all_three_weighted_improve": false, "confirmation_at_least_half_percent": false, "no_quarter_or_large_group_deterioration_above_5pct": false}, "confirmation_fractional_improvement": -0.007275265775682378, "new_CSV": false, "selected_submission": "v22"}

分群的最強有利／不利證據（描述性、不得據此分流test）：
- development，T−v22：{"best": {"section": "price_bands", "group": "20-50m", "band": "20-50m", "candidate_mae": 2663502.070113866, "control_mae": 2680263.403665634, "delta": -16761.333551767748, "relative_change_pct": -0.6253614301058728, "rows": 14197}, "worst": {"section": "price_bands", "group": ">=100m", "band": ">=100m", "candidate_mae": 31745951.633506157, "control_mae": 31529922.141700063, "delta": 216029.4918060936, "relative_change_pct": 0.6851570734466949, "rows": 182}}
- development，T−C：{"best": {"section": "price_bands", "group": "20-50m", "band": "20-50m", "candidate_mae": 2663502.070113866, "control_mae": 2679786.662147763, "delta": -16284.59203389706, "relative_change_pct": -0.6076824048689563, "rows": 14197}, "worst": {"section": "price_bands", "group": ">=100m", "band": ">=100m", "candidate_mae": 31745951.633506157, "control_mae": 31461726.76430808, "delta": 284224.8691980764, "relative_change_pct": 0.9033988227261475, "rows": 182}}
- confirmation，T−v22：{"best": {"section": "price_bands", "group": "<10m", "band": "<10m", "candidate_mae": 639110.0818151181, "control_mae": 647011.4443501264, "delta": -7901.362535008346, "relative_change_pct": -1.2212090843222505, "rows": 25585}, "worst": {"section": "price_bands", "group": ">=100m", "band": ">=100m", "candidate_mae": 18413034.04207892, "control_mae": 18130151.987229306, "delta": 282882.05484961346, "relative_change_pct": 1.5602850712386342, "rows": 319}}
- confirmation，T−C：{"best": {"section": "price_bands", "group": "<10m", "band": "<10m", "candidate_mae": 639110.0818151181, "control_mae": 647868.7443819005, "delta": -8758.662566782441, "relative_change_pct": -1.3519192958040671, "rows": 25585}, "worst": {"section": "price_bands", "group": ">=100m", "band": ">=100m", "candidate_mae": 18413034.04207892, "control_mae": 18232390.780946102, "delta": 180643.26113281772, "relative_change_pct": 0.9907820828500469, "rows": 319}}
- auxiliary，T−v22：{"best": {"section": "price_bands", "group": ">=100m", "band": ">=100m", "candidate_mae": 55076705.679667614, "control_mae": 56334125.73125411, "delta": -1257420.0515864938, "relative_change_pct": -2.2320752035544213, "rows": 171}, "worst": {"section": "price_bands", "group": "<10m", "band": "<10m", "candidate_mae": 1116709.5051595306, "control_mae": 1112063.336947904, "delta": 4646.168211626587, "relative_change_pct": 0.41779708558490525, "rows": 41132}}
- auxiliary，T−C：{"best": {"section": "price_bands", "group": "50-100m", "band": "50-100m", "candidate_mae": 13746895.160857534, "control_mae": 13823182.564032264, "delta": -76287.40317473002, "relative_change_pct": -0.5518801681258904, "rows": 1382}, "worst": {"section": "price_bands", "group": ">=100m", "band": ">=100m", "candidate_mae": 55076705.679667614, "control_mae": 54863590.33412391, "delta": 213115.34554370493, "relative_change_pct": 0.3884458604437195, "rows": 171}}

|窗口|對照|抽樣|指標|點差|2.5%|97.5%|有效次數|
|---|---|---|---|---:|---:|---:|---:|
|development|v22|project|overall|-5097.809914367972|-11859.809244867618|1407.9658558314136|2000|
|development|v22|project|weighted|-4948.792486089049|-11719.14652561586|1436.60305137958|2000|
|development|v22|month|overall|-5097.809914367972|-13388.798825735226|1067.4260967637688|2000|
|development|v22|month|weighted|-4948.792486089049|-17763.485521795472|1321.3948463059612|2000|
|development|C|project|overall|-3704.143586222315|-10845.494056214724|3002.642757030914|2000|
|development|C|project|weighted|-3570.1050874551293|-10710.646349895023|3039.1735427884314|2000|
|development|C|month|overall|-3704.143586222315|-12204.748168817256|2336.9504062017527|2000|
|development|C|month|weighted|-3570.1050874551293|-16429.548563964898|2074.684555803442|2000|
|confirmation|v22|project|overall|10063.354278142098|-962.0070759051188|21018.28309130243|2000|
|confirmation|v22|project|weighted|11056.781980966683|-1204.395027348696|23241.970745280592|2000|
|confirmation|v22|month|overall|10063.354278142098|2720.938375597814|17887.69515118003|2000|
|confirmation|v22|month|weighted|11056.781980966683|2572.458117101173|18036.460401862896|2000|
|confirmation|C|project|overall|11158.909594442463|539.3168419319904|21378.45496189072|2000|
|confirmation|C|project|weighted|12171.039759247098|507.1719884905499|23457.43010196284|2000|
|confirmation|C|month|overall|11158.909594442463|5358.145394432353|17789.98417503465|2000|
|confirmation|C|month|weighted|12171.039759247098|5764.281434130971|19172.679222903455|2000|
|auxiliary|v22|project|overall|-11676.47816253407|-18308.52287676786|-5571.88071412767|2000|
|auxiliary|v22|project|weighted|-10303.472590743098|-16160.55012552474|-4835.099731353648|2000|
|auxiliary|v22|month|overall|-11676.47816253407|-18130.881995062507|-4436.9164329301475|2000|
|auxiliary|v22|month|weighted|-10303.472590743098|-18208.802071110997|-3486.5523330131487|2000|
|auxiliary|C|project|overall|-4972.7958148918115|-12700.505811443518|3116.0498608736993|2000|
|auxiliary|C|project|weighted|-5252.075240005506|-12106.142269717238|1811.256136570211|2000|
|auxiliary|C|month|overall|-4972.7958148918115|-9820.745766570233|335.18951306481335|2000|
|auxiliary|C|month|weighted|-5252.075240005506|-11460.087204790208|187.55616235439126|2000|

原pure-X計畫與科學計畫分別核SHA；已保留原scorer計畫身分誤比的來源，實際評分來源由保存evaluation_request綁定。
沒有用真價重核reference y／unit SHA或weighted median最優值，僅核來源封存、合法索引／日期、保存state與向量／支持。NN重載精確性採保存trainer收據，未重新載模型；八共享陣列依SHA身份核，未開含y的NPZ。
反覆使用歷史窗口與單seed限制仍在；bootstrap不是Private勝率。固定門檻不等於自動採用，v22唯一提交、Private未知。
