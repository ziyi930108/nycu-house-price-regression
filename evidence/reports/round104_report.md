# R104：以固定 base 為目標的配對學生實驗

Primary門檻 **False**；mechanism門檻 **False**。v22唯一提交不變，Private未知，無新CSV。

C、T使用同一教師覆蓋子集、145數值欄／23類別欄、factor、日期權重、初始化及30 epochs。C=.375×原生加權L1；T為加權|.375p+教師offset−y|。C是本輪重訓的梯度尺度對照，不能當成原R102T。教師offset不供query推論，也不裁切負的等價殘差目標。
本程式只讀封存聚合及訓練metadata；評分由另一作者的scorer完成。摘要作者曾實作trainer及main driver，並非全程獨立審查。以下差值前者減後者，負值較好；單位TWD。

|窗口|v22整體／加權|新C整體／加權|T整體／加權|原R102T整體／加權|
|---|---:|---:|---:|---:|
|development|1,407,387.07 / 1,399,717.74|1,405,023.09 / 1,397,406.10|1,446,625.39 / 1,438,827.94|1,408,321.41 / 1,400,662.07|
|confirmation|1,383,228.41 / 1,447,153.42|1,393,681.62 / 1,458,704.65|1,394,235.82 / 1,460,061.83|1,390,447.76 / 1,455,219.45|
|auxiliary|2,309,352.36 / 2,023,165.66|2,296,532.20 / 2,012,917.78|2,265,910.30 / 1,986,008.58|2,299,732.46 / 2,014,911.54|

|窗口|T−v22整體／加權|T−C|C−v22|T−原R102T（點比較）|
|---|---:|---:|---:|---:|
|development|39,238.32 / 39,110.20|41,602.30 / 41,421.84|-2,363.98 / -2,311.64|38,303.98 / 38,165.87|
|confirmation|11,007.42 / 12,908.42|554.21 / 1,357.18|10,453.21 / 11,551.24|3,788.06 / 4,842.38|
|auxiliary|-43,442.06 / -37,157.08|-30,621.90 / -26,909.19|-12,820.16 / -10,247.89|-33,822.16 / -28,902.96|

## 事前固定門檻

|門檻|條件|通過|
|---|---|---|
|primary|all_three_overall_improve|False|
|primary|all_three_weighted_improve|False|
|primary|confirmation_at_least_half_percent|False|
|primary|no_quarter_or_large_group_deterioration_above_5pct|False|
|mechanism|base_all_three_overall_mae_improve|False|
|mechanism|base_all_three_weighted_mae_improve|False|
|mechanism|median_all_three_overall_mae_improve|False|
|mechanism|median_all_three_weighted_mae_improve|False|

Primary要求三期完整模型整體及固定加權均改善、confirmation至少0.5%，且固定季度或至少500列群沒有>5%退步。Mechanism要求base及median的整體／加權指標三期均勝新C；不要求native改善。兩門檻分開，通過也不等於自動採用或Private保證。

## 五層結果

每格為整體／加權差。native是prior NN，spatial仍只是一支N，median才是完整模型。

|窗口|層級|T−v22原分量|T−C|C−v22原分量|T−原R102T|
|---|---|---:|---:|---:|---:|
|development|native|4,304,924.69 / 4,317,634.58|4,442,664.96 / 4,454,338.13|-137,740.27 / -136,703.55|4,460,257.46 / 4,472,611.38|
|development|base|775,811.05 / 786,910.07|793,761.94 / 804,977.47|-17,950.88 / -18,067.40|796,144.16 / 807,624.94|
|development|reference|833,097.79 / 833,446.81|872,722.94 / 872,527.55|-39,625.15 / -39,080.74|873,958.89 / 873,884.47|
|development|spatial|789,453.29 / 790,289.52|827,790.08 / 828,108.59|-38,336.79 / -37,819.07|828,247.37 / 828,677.97|
|development|median|39,238.32 / 39,110.20|41,602.30 / 41,421.84|-2,363.98 / -2,311.64|38,303.98 / 38,165.87|
|confirmation|native|664,045.54 / 627,400.71|698,427.25 / 652,682.41|-34,381.71 / -25,281.71|727,613.32 / 682,895.92|
|confirmation|base|190,241.18 / 190,177.49|121,238.08 / 112,225.69|69,003.10 / 77,951.80|133,180.54 / 124,939.58|
|confirmation|reference|84,547.72 / 84,965.49|57,328.66 / 54,132.61|27,219.06 / 30,832.88|66,011.62 / 63,558.49|
|confirmation|spatial|81,880.03 / 82,573.71|53,489.34 / 50,411.93|28,390.68 / 32,161.78|61,648.48 / 59,279.46|
|confirmation|median|11,007.42 / 12,908.42|554.21 / 1,357.18|10,453.21 / 11,551.24|3,788.06 / 4,842.38|
|auxiliary|native|1,055,283.18 / 1,373,954.96|1,172,226.09 / 1,429,545.03|-116,942.91 / -55,590.07|1,328,539.65 / 1,617,402.78|
|auxiliary|base|659.46 / 86,800.80|22,466.61 / 94,545.17|-21,807.15 / -7,744.37|41,196.51 / 125,803.25|
|auxiliary|reference|-8,288.28 / 36,579.47|16,277.74 / 49,242.29|-24,566.02 / -12,662.82|28,956.71 / 69,263.74|
|auxiliary|spatial|-9,203.07 / 32,733.61|14,962.91 / 45,135.86|-24,165.98 / -12,402.25|28,573.43 / 65,527.69|
|auxiliary|median|-43,442.06 / -37,157.08|-30,621.90 / -26,909.19|-12,820.16 / -10,247.89|-33,822.16 / -28,902.96|

## 固定分群與正反證

|窗口|比較|seen差|new差|>5%群數|最大相對退步群|
|---|---|---:|---:|---:|---|
|development|T-v22|30,486.48|44,092.33|6|cities/新北市，15979列，135,590.65元／11.705%|
|development|T-C|29,275.10|48,439.31|7|cities/新北市，15979列，145,511.16元／12.670%|
|development|C-v22|1,211.38|-4,346.98|0|無|
|confirmation|T-v22|-5,726.99|23,674.56|1|cities/嘉義縣，2251列，73,344.69元／5.158%|
|confirmation|T-C|-6,514.34|5,904.76|0|無|
|confirmation|C-v22|787.35|17,769.80|0|無|
|auxiliary|T-v22|-19,308.43|-47,468.69|3|cities/屏東縣，1240列，145,849.92元／10.291%|
|auxiliary|T-C|-16,365.53|-33,000.54|1|cities/花蓮縣，691列，51,079.91元／5.688%|
|auxiliary|C-v22|-2,942.90|-14,468.16|1|cities/屏東縣，1240列，71,599.81元／5.052%|

|窗口|比較|固定群（季度或至少500列）|列數|MAE差|相對差%|
|---|---|---|---:|---:|---:|
|development|T-v22|strata/new_nonmissing|52253|44,092.33|2.60|
|development|T-v22|strata/seen|28981|30,486.48|3.45|
|development|T-v22|forecast_quarters/Q1|17608|-5,144.93|-0.41|
|development|T-v22|forecast_quarters/Q2|14193|29,940.34|2.27|
|development|T-v22|forecast_quarters/Q3|22487|50,002.69|3.45|
|development|T-v22|forecast_quarters/Q4|26946|64,155.11|4.20|
|development|T-v22|cities/南投縣|659|9,381.72|1.16|
|development|T-v22|cities/嘉義市|569|3,656.42|0.25|
|development|T-v22|cities/嘉義縣|798|22,339.95|2.81|
|development|T-v22|cities/基隆市|737|-11,655.86|-0.74|
|development|T-v22|cities/宜蘭縣|2242|6,797.39|0.81|
|development|T-v22|cities/屏東縣|615|-19,539.04|-2.43|
|development|T-v22|cities/彰化縣|2246|64,914.24|5.95|
|development|T-v22|cities/新北市|15979|135,590.65|11.70|
|development|T-v22|cities/新竹市|1250|-317,604.53|-7.08|
|development|T-v22|cities/新竹縣|3639|-102,842.66|-3.44|
|development|T-v22|cities/桃園市|15846|35,028.06|3.62|
|development|T-v22|cities/臺中市|14719|30,146.14|1.91|
|development|T-v22|cities/臺北市|2698|212,130.81|4.84|
|development|T-v22|cities/臺南市|7530|5,096.69|0.42|
|development|T-v22|cities/花蓮縣|535|28,241.72|4.97|
|development|T-v22|cities/苗栗縣|2402|-32,777.09|-2.50|
|development|T-v22|cities/雲林縣|1350|19,555.28|2.14|
|development|T-v22|cities/高雄市|7095|2,504.03|0.26|
|development|T-C|strata/new_nonmissing|52253|48,439.31|2.86|
|development|T-C|strata/seen|28981|29,275.10|3.31|
|development|T-C|forecast_quarters/Q1|17608|-1,602.09|-0.13|
|development|T-C|forecast_quarters/Q2|14193|34,764.30|2.65|
|development|T-C|forecast_quarters/Q3|22487|51,926.20|3.59|
|development|T-C|forecast_quarters/Q4|26946|64,820.62|4.25|
|development|T-C|cities/南投縣|659|10,965.63|1.36|
|development|T-C|cities/嘉義市|569|23,799.12|1.66|
|development|T-C|cities/嘉義縣|798|20,458.66|2.56|
|development|T-C|cities/基隆市|737|-6,639.77|-0.42|
|development|T-C|cities/宜蘭縣|2242|-3,824.35|-0.45|
|development|T-C|cities/屏東縣|615|6,223.03|0.80|
|development|T-C|cities/彰化縣|2246|78,585.20|7.29|
|development|T-C|cities/新北市|15979|145,511.16|12.67|
|development|T-C|cities/新竹市|1250|-303,620.41|-6.79|
|development|T-C|cities/新竹縣|3639|-76,801.36|-2.59|
|development|T-C|cities/桃園市|15846|44,223.73|4.62|
|development|T-C|cities/臺中市|14719|39,225.14|2.51|
|development|T-C|cities/臺北市|2698|144,006.00|3.23|
|development|T-C|cities/臺南市|7530|-11,250.45|-0.91|
|development|T-C|cities/花蓮縣|535|31,055.27|5.49|
|development|T-C|cities/苗栗縣|2402|-51,989.88|-3.91|
|development|T-C|cities/雲林縣|1350|20,388.18|2.24|
|development|T-C|cities/高雄市|7095|-4,231.00|-0.43|
|confirmation|T-v22|strata/new_nonmissing|76096|23,674.56|1.31|
|confirmation|T-v22|strata/seen|57601|-5,726.99|-0.70|
|confirmation|T-v22|forecast_quarters/Q1|26579|6,489.04|0.71|
|confirmation|T-v22|forecast_quarters/Q2|31157|-59.57|-0.01|
|confirmation|T-v22|forecast_quarters/Q3|37236|8,736.09|0.59|
|confirmation|T-v22|forecast_quarters/Q4|38725|25,196.79|1.42|
|confirmation|T-v22|cities/南投縣|1015|28,214.55|2.91|
|confirmation|T-v22|cities/嘉義市|689|45,111.93|3.83|
|confirmation|T-v22|cities/嘉義縣|2251|73,344.69|5.16|
|confirmation|T-v22|cities/基隆市|1100|-4,991.65|-0.27|
|confirmation|T-v22|cities/宜蘭縣|2907|-24,625.00|-2.63|
|confirmation|T-v22|cities/屏東縣|1013|-33,730.21|-4.46|
|confirmation|T-v22|cities/彰化縣|3295|18,664.91|1.76|
|confirmation|T-v22|cities/新北市|21822|24,662.17|1.87|
|confirmation|T-v22|cities/新竹市|2131|-5,702.72|-0.22|
|confirmation|T-v22|cities/新竹縣|5589|-6,756.24|-0.33|
|confirmation|T-v22|cities/桃園市|26327|-22,342.29|-1.81|
|confirmation|T-v22|cities/臺中市|27157|10,028.90|0.71|
|confirmation|T-v22|cities/臺北市|4957|176,616.43|4.34|
|confirmation|T-v22|cities/臺南市|9822|-11,119.06|-1.22|
|confirmation|T-v22|cities/花蓮縣|583|4,044.14|0.65|
|confirmation|T-v22|cities/苗栗縣|4001|-8,222.83|-0.55|
|confirmation|T-v22|cities/雲林縣|2364|-21,591.48|-2.40|
|confirmation|T-v22|cities/高雄市|16443|26,522.95|2.62|
|confirmation|T-C|strata/new_nonmissing|76096|5,904.76|0.32|
|confirmation|T-C|strata/seen|57601|-6,514.34|-0.79|
|confirmation|T-C|forecast_quarters/Q1|26579|-3,327.15|-0.36|
|confirmation|T-C|forecast_quarters/Q2|31157|-4,514.23|-0.38|
|confirmation|T-C|forecast_quarters/Q3|37236|4,674.69|0.31|
|confirmation|T-C|forecast_quarters/Q4|38725|3,334.05|0.19|
|confirmation|T-C|cities/南投縣|1015|18,533.29|1.89|
|confirmation|T-C|cities/嘉義市|689|5,395.91|0.44|
|confirmation|T-C|cities/嘉義縣|2251|29,539.97|2.02|
|confirmation|T-C|cities/基隆市|1100|35,539.43|1.99|
|confirmation|T-C|cities/宜蘭縣|2907|-13,374.96|-1.45|
|confirmation|T-C|cities/屏東縣|1013|-21,027.76|-2.83|
|confirmation|T-C|cities/彰化縣|3295|6,152.22|0.57|
|confirmation|T-C|cities/新北市|21822|5,642.55|0.42|
|confirmation|T-C|cities/新竹市|2131|-17,314.35|-0.67|
|confirmation|T-C|cities/新竹縣|5589|280.93|0.01|
|confirmation|T-C|cities/桃園市|26327|-10,825.80|-0.89|
|confirmation|T-C|cities/臺中市|27157|1,408.49|0.10|
|confirmation|T-C|cities/臺北市|4957|14,758.68|0.35|
|confirmation|T-C|cities/臺南市|9822|-9,397.52|-1.03|
|confirmation|T-C|cities/花蓮縣|583|-873.38|-0.14|
|confirmation|T-C|cities/苗栗縣|4001|-10,094.14|-0.68|
|confirmation|T-C|cities/雲林縣|2364|-11,580.79|-1.30|
|confirmation|T-C|cities/高雄市|16443|13,775.02|1.35|
|auxiliary|T-v22|strata/new_nonmissing|90424|-47,468.69|-1.90|
|auxiliary|T-v22|strata/seen|15087|-19,308.43|-1.60|
|auxiliary|T-v22|forecast_quarters/Q1|33694|6,946.97|0.39|
|auxiliary|T-v22|forecast_quarters/Q2|29169|-64,731.00|-2.70|
|auxiliary|T-v22|forecast_quarters/Q3|25213|-74,631.39|-2.88|
|auxiliary|T-v22|forecast_quarters/Q4|17435|-60,101.32|-2.17|
|auxiliary|T-v22|cities/南投縣|869|7,966.21|0.71|
|auxiliary|T-v22|cities/嘉義市|731|-65,738.80|-3.11|
|auxiliary|T-v22|cities/嘉義縣|1560|-67,418.14|-4.97|
|auxiliary|T-v22|cities/基隆市|1577|109,076.69|3.23|
|auxiliary|T-v22|cities/宜蘭縣|2330|32,850.15|2.48|
|auxiliary|T-v22|cities/屏東縣|1240|145,849.92|10.29|
|auxiliary|T-v22|cities/彰化縣|2607|77,434.32|5.28|
|auxiliary|T-v22|cities/新北市|17191|-31,371.68|-1.61|
|auxiliary|T-v22|cities/新竹市|1586|-17,652.21|-0.41|
|auxiliary|T-v22|cities/新竹縣|5042|-103,096.21|-3.65|
|auxiliary|T-v22|cities/桃園市|20723|-19,677.76|-1.09|
|auxiliary|T-v22|cities/臺中市|20837|-97,783.30|-3.40|
|auxiliary|T-v22|cities/臺北市|4712|-128,228.68|-1.93|
|auxiliary|T-v22|cities/臺南市|8886|-21,813.20|-1.23|
|auxiliary|T-v22|cities/花蓮縣|691|70,024.75|7.97|
|auxiliary|T-v22|cities/苗栗縣|2607|-30,705.64|-1.84|
|auxiliary|T-v22|cities/雲林縣|1873|-9,736.53|-0.91|
|auxiliary|T-v22|cities/高雄市|10138|-73,774.36|-4.08|
|auxiliary|T-C|strata/new_nonmissing|90424|-33,000.54|-1.33|
|auxiliary|T-C|strata/seen|15087|-16,365.53|-1.36|
|auxiliary|T-C|forecast_quarters/Q1|33694|6,677.16|0.37|
|auxiliary|T-C|forecast_quarters/Q2|29169|-44,469.97|-1.87|
|auxiliary|T-C|forecast_quarters/Q3|25213|-36,062.95|-1.41|
|auxiliary|T-C|forecast_quarters/Q4|17435|-71,667.77|-2.58|
|auxiliary|T-C|cities/南投縣|869|12,162.78|1.09|
|auxiliary|T-C|cities/嘉義市|731|-168,783.41|-7.62|
|auxiliary|T-C|cities/嘉義縣|1560|-15,476.74|-1.19|
|auxiliary|T-C|cities/基隆市|1577|73,041.07|2.14|
|auxiliary|T-C|cities/宜蘭縣|2330|25,892.62|1.94|
|auxiliary|T-C|cities/屏東縣|1240|74,250.10|4.99|
|auxiliary|T-C|cities/彰化縣|2607|48,180.99|3.22|
|auxiliary|T-C|cities/新北市|17191|-19,439.79|-1.00|
|auxiliary|T-C|cities/新竹市|1586|-39,807.30|-0.93|
|auxiliary|T-C|cities/新竹縣|5042|-106,807.70|-3.77|
|auxiliary|T-C|cities/桃園市|20723|198.55|0.01|
|auxiliary|T-C|cities/臺中市|20837|-80,784.84|-2.83|
|auxiliary|T-C|cities/臺北市|4712|-14,562.09|-0.22|
|auxiliary|T-C|cities/臺南市|8886|-28,398.93|-1.59|
|auxiliary|T-C|cities/花蓮縣|691|51,079.91|5.69|
|auxiliary|T-C|cities/苗栗縣|2607|-41,707.63|-2.48|
|auxiliary|T-C|cities/雲林縣|1873|663.83|0.06|
|auxiliary|T-C|cities/高雄市|10138|-49,097.14|-2.75|

季度為原評分的預測期季度，不改成曆年季度；群重疊，不能把城市、新舊案與季度差額相加。完整月份、全部退步門檻清單與空群保留於摘要saved_stage_results。

## 五價帶與尾部

|窗口|真價帶|列數|T−v22群MAE差|T−C群MAE差|全窗貢獻T−v22|全窗貢獻T−C|
|---|---|---:|---:|---:|---:|---:|
|development|<10m|22139|39,698.33|41,554.83|10,819.13|11,325.09|
|development|10-20m|43914|47,381.03|51,677.67|25,613.54|27,936.25|
|development|20-50m|14197|-2,659.58|6,042.54|-464.81|1,056.03|
|development|50-100m|802|156,623.26|156,102.16|1,546.30|1,541.15|
|development|>=100m|182|769,559.34|-114,364.93|1,724.15|-256.23|
|confirmation|<10m|25585|-8,084.53|-5,270.46|-1,547.10|-1,008.58|
|confirmation|10-20m|74109|-1,802.76|1,139.02|-999.28|631.36|
|confirmation|20-50m|31959|29,779.05|-1,937.10|7,118.40|-463.05|
|confirmation|50-100m|1725|119,975.55|-42,679.47|1,547.96|-550.66|
|confirmation|>=100m|319|2,048,387.32|815,230.76|4,887.44|1,945.13|
|auxiliary|<10m|41132|17,966.36|16,936.92|7,003.94|6,602.62|
|auxiliary|10-20m|47793|-35,085.17|-28,226.29|-15,892.42|-12,785.58|
|auxiliary|20-50m|15033|-171,800.98|-151,446.40|-24,477.87|-21,577.79|
|auxiliary|50-100m|1382|-188,205.84|-338,336.03|-2,465.15|-4,431.58|
|auxiliary|>=100m|171|-4,695,890.60|968,985.41|-7,610.56|1,570.42|

全窗貢獻＝群列數×保存群MAE差÷全窗列數，五帶加總核至1e−6元。這是事後診斷，不是依未知真價分流；原始signed error、underprediction與各模型尾部摘要完整保留，模型各自最大誤差列不視為同一批樣本。

## 全部24個保存配對區間

|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|
|---|---|---|---:|---|---:|---|
|development|T−v22|project/overall|39,238.32|7,839.01～81,428.60|2000/0|positive|
|development|T−v22|project/weighted|39,110.20|7,568.20～79,194.34|2000/0|positive|
|development|T−v22|month/overall|39,238.32|12,642.49～56,711.39|2000/0|positive|
|development|T−v22|month/weighted|39,110.20|11,521.98～55,284.52|2000/0|positive|
|development|T−C|project/overall|41,602.30|10,407.14～83,275.73|2000/0|positive|
|development|T−C|project/weighted|41,421.84|10,330.96～81,777.13|2000/0|positive|
|development|T−C|month/overall|41,602.30|15,164.44～58,235.93|2000/0|positive|
|development|T−C|month/weighted|41,421.84|14,916.62～56,510.19|2000/0|positive|
|confirmation|T−v22|project/overall|11,007.42|-4,813.19～27,989.56|2000/0|crosses_zero|
|confirmation|T−v22|project/weighted|12,908.42|-4,523.57～31,650.34|2000/0|crosses_zero|
|confirmation|T−v22|month/overall|11,007.42|1,768.93～19,315.84|2000/0|positive|
|confirmation|T−v22|month/weighted|12,908.42|4,744.20～19,336.97|2000/0|positive|
|confirmation|T−C|project/overall|554.21|-6,626.37～7,847.31|2000/0|crosses_zero|
|confirmation|T−C|project/weighted|1,357.18|-6,331.79～9,163.13|2000/0|crosses_zero|
|confirmation|T−C|month/overall|554.21|-4,265.05～3,897.39|2000/0|crosses_zero|
|confirmation|T−C|month/weighted|1,357.18|-2,668.70～4,476.01|2000/0|crosses_zero|
|auxiliary|T−v22|project/overall|-43,442.06|-71,055.00～-19,996.52|2000/0|negative|
|auxiliary|T−v22|project/weighted|-37,157.08|-58,298.10～-18,607.93|2000/0|negative|
|auxiliary|T−v22|month/overall|-43,442.06|-83,379.26～-9,363.16|2000/0|negative|
|auxiliary|T−v22|month/weighted|-37,157.08|-71,925.22～-8,868.21|2000/0|negative|
|auxiliary|T−C|project/overall|-30,621.90|-49,578.71～-13,163.11|2000/0|negative|
|auxiliary|T−C|project/weighted|-26,909.19|-41,474.01～-12,877.16|2000/0|negative|
|auxiliary|T−C|month/overall|-30,621.90|-58,394.86～-7,965.74|2000/0|negative|
|auxiliary|T−C|month/weighted|-26,909.19|-53,818.87～-7,569.60|2000/0|negative|

方向計數：{"negative": 8, "crosses_zero": 6, "positive": 10, "unavailable": 0}。沿固定seed與每組2000次保存結果，摘要未重抽。R102T、C−v22仅點比較，沒有追加bootstrap；區間不是Private勝率。

## 樣本覆蓋與實測成本

|窗口|outer fit列|原eligible列|新學生列|eligible未有教師列|query列|保留日期權重質量比例|
|---|---:|---:|---:|---:|---:|---:|
|development|240316|240260|123352|116908|81234|0.925316|
|confirmation|321550|321494|204586|116908|133697|0.969324|
|auxiliary|134805|134749|17841|116908|105511|0.487263|

|窗口|C fit秒|T fit秒|C程序秒|T程序秒|準備秒|stage秒|
|---|---:|---:|---:|---:|---:|---:|
|development|10.625|10.188|14.313|13.969|6.937|50.609|
|confirmation|17.157|17.578|21.187|21.812|8.797|72.578|
|auxiliary|1.610|1.563|5.031|5.250|4.187|25.953|

新增早期教師origin 2020-08-11、query截至2021-08-10：fit 116964列／query 17841列。area NN fit 10.282秒，LGB fit 103.953秒；教師程序 131.000秒。較後aux/dev教師沿用原forward向量。
八次native fit＝7 NN＋1 LGB：教師fit 114.235秒＋六學生fit 58.721秒＝172.956秒。學生main程序 153.718秒，獨立scorer內層 58.563秒／execution 58.594秒。fit已含於各自程序耗時，不重複加總；0新prior/R/S建模，未量聊天開發工時。

## 解讀界線

教師使用固定2023非價格地圖，晚於早期origin；只稱價格標籤依時間先後的forward預測，不能稱完整歷史as-of。原R90 guard未削弱；學生encoder/scaler/factor仍沿outer-fit範圍，也不是每個fit row逐列as-of。教師offset分布與實際query原area/LGB分布可能不同，教師覆蓋子集也改變訓練母體。
單seed、30固定epochs，反覆利用既有年度窗口，並非盲測。matched-gradient C只控制部分因素，不證明因果；base loss也不保證reference、spatial或median受益。bootstrap僅條件於已保存預測，不含重新訓練、累積選型與未來分布變化。產物可追溯不等於乾淨環境端到端重現。
v22仍是使用者唯一選定提交；Private未知，未查外部價格答案、未生成或上傳新CSV。

[固定計畫](round104_plan.md) · [完整聚合結果](../outputs/round104/evaluation/results.json) · [摘要JSON](../outputs/round104_summary/summary.json) · [摘要來源](../src/summarize_round104.py)
