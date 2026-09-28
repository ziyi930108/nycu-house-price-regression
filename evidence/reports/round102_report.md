# R102：明示時間先驗與同寬零欄對照

Primary門檻：**False**；mechanism門檻：**False**。v22唯一提交不變，Private未知，無新CSV。

T為145欄log-prior候選，C為145欄正零控制；兩者factor、loss列與另七陣列皆同R100T。**本輪C不是R100原OOF先驗C**，零欄不另選為候選。僅讀保存聚合JSON，沒有新評分。差值為前者減後者，負值較好；金額為元。

|窗口|v22整體／加權|C整體／加權|T整體／加權|
|---|---:|---:|---:|
|development|1,407,387.07 / 1,399,717.74|1,404,025.68 / 1,396,470.47|1,408,321.41 / 1,400,662.07|
|confirmation|1,383,228.41 / 1,447,153.42|1,391,350.42 / 1,455,972.26|1,390,447.76 / 1,455,219.45|
|auxiliary|2,309,352.36 / 2,023,165.66|2,292,749.46 / 2,007,998.02|2,299,732.46 / 2,014,911.54|

|窗口|T−v22整體／加權|T−C|C−v22|T−v24|
|---|---:|---:|---:|---:|
|development|934.33 / 944.33|4,295.72 / 4,191.61|-3,361.39 / -3,247.28|980.69 / 966.15|
|confirmation|7,219.36 / 8,066.03|-902.66 / -752.81|8,122.01 / 8,818.85|17,251.85 / 19,130.57|
|auxiliary|-9,619.90 / -8,254.12|6,982.99 / 6,913.52|-16,602.89 / -15,167.65|24,323.32 / 22,340.57|

## 兩個門檻分開判斷

|門檻|條件|通過|
|---|---|---|
|primary|all_three_overall_improve|False|
|primary|all_three_weighted_improve|False|
|primary|confirmation_at_least_half_percent|False|
|primary|no_quarter_or_large_group_deterioration_above_5pct|False|
|mechanism|full_all_three_overall_improve|False|
|mechanism|full_all_three_weighted_improve|False|
|mechanism|native_all_three_overall_improve|True|

若native未改善，不用後處理或median收益宣稱ratio表示機制成功。即使探索門檻通過也不是採用、Private保證或乾淨環境重現。

## 由原生到完整模型

每格為整體／加權差；native是prior NN，spatial是N分支，median才是完整模型。

|窗口|層級|T−v22原分量|T−C|C−v22原分量|
|---|---|---:|---:|---:|
|development|native|-155,332.77 / -154,976.80|-304,378.87 / -306,003.27|149,046.11 / 151,026.47|
|development|base|-20,333.11 / -20,714.87|-29,343.32 / -30,561.02|9,010.21 / 9,846.15|
|development|reference|-40,861.10 / -40,437.66|-33,865.44 / -34,059.89|-6,995.66 / -6,377.77|
|development|spatial|-38,794.09 / -38,388.45|-31,222.64 / -31,445.02|-7,571.45 / -6,943.43|
|development|median|934.33 / 944.33|4,295.72 / 4,191.61|-3,361.39 / -3,247.28|
|confirmation|native|-63,567.78 / -55,495.22|-386,280.18 / -381,512.98|322,712.39 / 326,017.76|
|confirmation|base|57,060.64 / 65,237.91|-93,179.56 / -89,825.61|150,240.20 / 155,063.52|
|confirmation|reference|18,536.10 / 21,407.00|-46,906.23 / -47,190.82|65,442.33 / 68,597.81|
|confirmation|spatial|20,231.54 / 23,294.25|-45,502.72 / -45,699.91|65,734.26 / 68,994.16|
|confirmation|median|7,219.36 / 8,066.03|-902.66 / -752.81|8,122.01 / 8,818.85|
|auxiliary|native|-273,256.47 / -243,447.82|-298,184.14 / -319,842.16|24,927.67 / 76,394.33|
|auxiliary|base|-40,537.05 / -39,002.45|1,789.97 / -11,254.08|-42,327.02 / -27,748.37|
|auxiliary|reference|-37,244.98 / -32,684.27|-515.21 / -6,891.50|-36,729.77 / -25,792.77|
|auxiliary|spatial|-37,776.50 / -32,794.08|-2,618.23 / -8,016.80|-35,158.26 / -24,777.27|
|auxiliary|median|-9,619.90 / -8,254.12|6,982.99 / 6,913.52|-16,602.89 / -15,167.65|

## 分群正反證

|窗口|比較|seen差|new差|>5%群數|最大相對退步群|
|---|---|---:|---:|---:|---|
|development|T-v22|1,616.95|555.74|0|無|
|development|T-C|-2,816.62|8,240.43|0|無|
|development|C-v22|4,433.57|-7,684.69|0|無|
|development|T-v24|-12.47|1,531.52|1|cities/新竹市，1250列，513,761.06元／12.881%|
|confirmation|T-v22|-233.90|12,861.12|0|無|
|confirmation|T-C|-2,221.72|95.81|0|無|
|confirmation|C-v22|1,987.81|12,765.31|0|無|
|confirmation|T-v24|713.56|29,770.53|1|cities/新竹市，2131列，156,703.14元／6.489%|
|auxiliary|T-v22|-4,375.46|-10,494.92|2|cities/嘉義市，731列，137,478.12元／6.508%|
|auxiliary|T-C|6,716.24|7,027.50|0|無|
|auxiliary|C-v22|-11,091.70|-17,522.42|1|cities/嘉義市，731列，119,905.20元／5.676%|
|auxiliary|T-v24|16,709.77|25,593.62|4|cities/基隆市，1577列，809,055.59元／30.534%|

development城市正反端點（至少500列，按群內MAE差）：最低屏東縣 -17,584.72元；最高臺北市 54,998.72元。最低亦可能仍退步；非另設選型門檻。


confirmation城市正反端點（至少500列，按群內MAE差）：最低基隆市 -52,626.72元；最高臺北市 115,260.79元。最低亦可能仍退步；非另設選型門檻。


auxiliary城市正反端點（至少500列，按群內MAE差）：最低臺北市 -112,539.54元；最高嘉義市 137,478.12元。最低亦可能仍退步；非另設選型門檻。


## 五價帶及尾部貢獻

|窗口|真價帶|列數|T−v22群MAE差|T−C群MAE差|全窗貢獻T−v22|全窗貢獻T−C|
|---|---|---:|---:|---:|---:|---:|
|development|<10m|22139|-1,316.89|-1,637.73|-358.90|-446.34|
|development|10-20m|43914|-937.97|2,198.40|-507.05|1,188.43|
|development|20-50m|14197|1,182.58|16,168.15|206.68|2,825.65|
|development|50-100m|802|29,060.05|33,446.28|286.90|330.21|
|development|>=100m|182|583,235.67|177,543.62|1,306.71|397.78|
|confirmation|<10m|25585|-4,666.25|2,509.49|-892.96|480.23|
|confirmation|10-20m|74109|-3,971.71|-3,699.66|-2,201.54|-2,050.74|
|confirmation|20-50m|31959|28,908.31|-1,650.10|6,910.26|-394.44|
|confirmation|50-100m|1725|116,490.50|14,022.16|1,503.00|180.92|
|confirmation|>=100m|319|796,567.67|369,396.67|1,900.60|881.38|
|auxiliary|<10m|41132|2,786.28|2,617.56|1,086.19|1,020.42|
|auxiliary|10-20m|47793|-9,654.84|274.01|-4,373.32|124.12|
|auxiliary|20-50m|15033|-17,911.33|34,054.67|-2,551.97|4,852.04|
|auxiliary|50-100m|1382|2,322.07|126,843.07|30.41|1,661.41|
|auxiliary|>=100m|171|-2,351,607.76|-416,487.87|-3,811.21|-675.00|

全窗貢獻＝保存群列數×群內MAE差÷全窗列數；五带加總核至1e−6元。這是事後聚合診斷，不是因果證明或依未知真價推論分流。完整群及空群保存在原結果與摘要JSON。

## 全部24個保存配對區間

|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|
|---|---|---|---:|---|---:|---|
|development|T−v22|project/overall|934.33|-5,731.98～8,389.41|2000/0|crosses_zero|
|development|T−v22|project/weighted|944.33|-5,454.89～8,336.08|2000/0|crosses_zero|
|development|T−v22|month/overall|934.33|-4,132.07～4,748.25|2000/0|crosses_zero|
|development|T−v22|month/weighted|944.33|-4,730.17～4,386.26|2000/0|crosses_zero|
|development|T−C|project/overall|4,295.72|-3,390.60～12,084.68|2000/0|crosses_zero|
|development|T−C|project/weighted|4,191.61|-3,388.19～11,995.99|2000/0|crosses_zero|
|development|T−C|month/overall|4,295.72|-1,113.82～12,023.59|2000/0|crosses_zero|
|development|T−C|month/weighted|4,191.61|-1,180.49～15,136.27|2000/0|crosses_zero|
|confirmation|T−v22|project/overall|7,219.36|-4,511.40～19,656.88|2000/0|crosses_zero|
|confirmation|T−v22|project/weighted|8,066.03|-5,035.31～22,010.28|2000/0|crosses_zero|
|confirmation|T−v22|month/overall|7,219.36|1,723.40～12,831.38|2000/0|positive|
|confirmation|T−v22|month/weighted|8,066.03|1,939.22～12,670.80|2000/0|positive|
|confirmation|T−C|project/overall|-902.66|-8,171.17～7,032.59|2000/0|crosses_zero|
|confirmation|T−C|project/weighted|-752.81|-8,851.61～8,187.49|2000/0|crosses_zero|
|confirmation|T−C|month/overall|-902.66|-6,348.45～4,437.49|2000/0|crosses_zero|
|confirmation|T−C|month/weighted|-752.81|-6,542.00～7,241.50|2000/0|crosses_zero|
|auxiliary|T−v22|project/overall|-9,619.90|-22,624.34～365.78|2000/0|crosses_zero|
|auxiliary|T−v22|project/weighted|-8,254.12|-18,286.93～-245.83|2000/0|negative|
|auxiliary|T−v22|month/overall|-9,619.90|-18,607.78～-841.42|2000/0|negative|
|auxiliary|T−v22|month/weighted|-8,254.12|-13,598.07～-1,872.26|2000/0|negative|
|auxiliary|T−C|project/overall|6,982.99|-2,424.76～15,958.21|2000/0|crosses_zero|
|auxiliary|T−C|project/weighted|6,913.52|-1,012.83～14,725.31|2000/0|crosses_zero|
|auxiliary|T−C|month/overall|6,982.99|4,889.13～9,111.98|2000/0|positive|
|auxiliary|T−C|month/weighted|6,913.52|4,043.24～12,135.99|2000/0|positive|

區間方向計數：{"negative": 3, "crosses_zero": 17, "positive": 4, "unavailable": 0}。固定2000次，未重抽；不是Private勝率。

## 實測時間與限制

|窗口|C fit秒|T fit秒|stage秒|
|---|---:|---:|---:|
|development|18.500|17.531|61.453|
|confirmation|26.093|24.641|83.891|
|auxiliary|10.390|10.000|41.110|

六NN fit共107.155秒，已含於main 189.891秒，不重複相加；獨立評分另65.547秒。0新樹、0prior重建，未量開發聊天工時。
單seed、反覆年度窗口、沒有初始RNG trace；同寬不等於完全因果隔離。全eligible fit標準化、encoder/query prior/R/S的outer-fit範圍仍在，不能稱所有輸入逐列as-of。bootstrap不含重訓、累積選型或未來分布移轉。

[固定計畫](round102_plan.md) · [原完整結果](../outputs/round102/evaluation/results.json) · [摘要JSON](../outputs/round102_summary/summary.json) · [來源](../src/summarize_round102.py)
