# 第30輪：每車位申報面積回測

**未採用，維持既有v3。** 唯一候選在development整體MAE退步1.490025%、固定seen/new加權MAE退步1.471902%，未達事前兩項至少1%改善門檻。四季與大組最多5%退步檢查通過，但沒有季度改善；因此未執行confirmation、auxiliary或final，沒有新CSV、release或上傳。

原95欄值、型別、欄序和索引保持，只追加float32 `parking_area_per_slot`，共96欄／23類別。當停車面積有限且正、車位數有限且為正整數時，以float64計算面積÷數量再轉float32；其他NaN，溢位報錯，不截尾、不補值、不加旗標。這是申報移轉面積比值，不是核實淨車格大小；既有解析／面積語意限制仍存在。既存純X盤點的query可用率78.19%，但多車位使比值不同於原面積的只有7.13%，不能混稱新增資訊覆蓋。

完整fit 240,316筆，query 81,234筆，未用住宅子集或分流。所有R6傳入參數相同：depth8、2,999樹、learning_rate .065、seed42、Plain、raw MAE與uniform；target y/1e6、逆轉換max(z×1e6,0)，沒有eval_set、early stopping、校正或搜尋。原37.5%raw CatBoost＋37.5%新geo CatBoost＋25%unit LightGBM融合後，原87欄reference trend只用一次。建案分組保持原規則，完整列去重沿原始資料欄位（含價格、排除編號）；共用loader會反序列化query targets，但fit／state不使用。

下表直接轉錄保存摘要，單位TWD；正百分比表示MAE變差，負值表示改善。沒有重算逐筆誤差。

| 指標／組別 | 筆數 | v3 MAE | 本輪 MAE | 相對變化 |
|---|---:|---:|---:|---:|
| 整體 | 81,234 | 1,502,390.31 | 1,524,776.31 | +1.490025% |
| 固定 seen/new 加權 | 81,234 | 1,493,688.89 | 1,515,674.52 | +1.471902% |
| 已見建案 | 28,981 | 907,993.51 | 903,030.94 | -0.546543% |
| 新具名建案 | 52,253 | 1,832,059.69 | 1,869,613.98 | +2.049840% |
| 建案名稱缺失 | 0 | null | null | null |
| Q1 | 17,608 | 1,315,733.32 | 1,358,504.95 | +3.250782% |
| Q2 | 14,193 | 1,485,546.75 | 1,523,581.28 | +2.560305% |
| Q3 | 22,487 | 1,534,081.87 | 1,541,328.44 | +0.472372% |
| Q4 | 26,946 | 1,606,786.85 | 1,620,243.52 | +0.837489% |

保存的四個門檻：整體至少1%改善 **未過**；加權至少1%改善 **未過**；各預測季最多5%退步 **通過**；至少500筆建案組最多5%退步 **通過**。已見案略有改善，新案及四季變差；不能說全部組別退步。缺名組0筆，MAE為null。季度按距切點0–91、92–183、184–275、276天以上定義。

保存metadata記錄全部81,234筆raw與TWD在CBM重載後逐位一致，driver亦核對同窗v3控制與完整query。原生depth8／Plain／SymmetricTree／2,999樹、最大256葉；saved learning_rate=0.06499999762，fitted=0.06499999761581421，按既定rtol=1e-7對要求.065核對。native raw有5筆負值，按原非負投影成地理元件零值，沒有刪列；並非最終融合總價均為零。這些是driver內保存的重載證據，本輪沒有另跑獨立native verifier。

實際命令耗時：fit 177.2947919秒、evaluate 34.4450522秒、finalize 6.0630000秒，均exit0；helper fit elapsed 152.5620000秒、runner 164.1870000秒另列，口徑不同。

這些年度標籤已跨輪反覆使用，不是未見盲測；1%工程門檻不是顯著性證明。沒有以Public調整本輪方法，本輪無新Public分數，Private仍未知，不宣稱未來改善。

依據：[事前規約](round30_plan.md)、[工作紀錄](round30_work_log.md)、[既存純X研究](parking_ratio_research.md)、[保存回測結果](../outputs/round30/evaluation/development/results.json)、[模型metadata](../outputs/round30/models/development/metadata.json)、[終局決策](../outputs/round30/accepted_candidate.json)、[finalize執行紀錄](../outputs/round30/execution/finalize.json)。
