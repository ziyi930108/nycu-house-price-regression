# R102：明示價格先驗的同寬度對照評估

C=145欄正零對照；T=145欄 log-prior。两組同 R100T factor／loss 列，不是舊 R100 C。

|窗口|T MAE|C MAE|v22 MAE|T−v22|T−C|native T−C|
|---|---:|---:|---:|---:|---:|---:|
|development|1408321.41|1404025.68|1407387.07|934.33|4295.72|-304378.87|
|confirmation|1390447.76|1391350.42|1383228.41|7219.36|-902.66|-386280.18|
|auxiliary|2299732.46|2292749.46|2309352.36|-9619.90|6982.99|-298184.14|

Primary research gate: False；mechanism gate: False。兩門檻分開保留，不用後處理收益替代原生改善。

四完整向量／三支 NN 分支的五層指標、全部城市／月份／季度／價帶／尾部、24 CI 與條件抽樣陣列見 [完整結果](results.json)。

Adaptive study on reused windows, not blind validation. C is a 145-column zero-input control, not the old R100 mask control. C and T both retain R100T factors and loss rows. Conditional bootstrap is not Private probability; only v22 remains selected.

沒有全量 test 預測或 CSV；Private 未知，v22 唯一提交不變。
