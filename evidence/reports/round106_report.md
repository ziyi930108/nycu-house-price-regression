# R106：四個身分類別的聯動 dropout

C為原逐欄獨立0.1遮罩的重訓；T只令案名、完整門牌、棟及戶、棟別四欄共享案名原mask。沿原R63 144數值／23類別、價格因子及日期weighted MAE，三窗共六NN fit。唯一提交仍v22，Private未知，無新CSV。

所有值為新臺幣元；每格為整體／固定加權MAE，差值負數較好。C−v22展示原設定重訓偏移，不能隱去。

|窗口|v22|C|T|T−v22|T−C|C−v22|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07 / 1,399,717.74|1,407,387.07 / 1,399,717.74|1,407,323.81 / 1,399,653.26|-63.26 / -64.48|-63.26 / -64.48|0.00 / 0.00|
|confirmation|1,383,228.41 / 1,447,153.42|1,383,228.41 / 1,447,153.42|1,385,668.08 / 1,449,780.51|2,439.67 / 2,627.10|2,439.67 / 2,627.10|0.00 / 0.00|
|auxiliary|2,309,352.36 / 2,023,165.66|2,309,352.36 / 2,023,165.66|2,309,775.15 / 2,023,758.89|422.80 / 593.23|422.80 / 593.23|0.00 / 0.00|

## 固定門檻

```json
{
  "Private_score": null,
  "automatic_adoption": false,
  "both_research_gates_passed": false,
  "mechanism": {
    "causal_identification_claimed": false,
    "components": {
      "all_three_final_overall_mae_improve": false,
      "all_three_final_weighted_mae_improve": false,
      "no_quarter_or_large_group_deterioration_above_5pct": true
    },
    "control": "retrained original R63 independent dropout",
    "mechanism_gate_passed": false,
    "native_improvement_required": false
  },
  "new_CSV": false,
  "primary": {
    "C_is_mechanism_control_not_alternative_candidate": true,
    "Private_score": null,
    "adoption_research_gate_passed": false,
    "automatic_adoption": false,
    "components": {
      "all_three_overall_improve": false,
      "all_three_weighted_improve": false,
      "confirmation_at_least_half_percent": false,
      "no_quarter_or_large_group_deterioration_above_5pct": true
    },
    "confirmation_fractional_improvement": -0.0017637509762117394,
    "new_CSV": false,
    "selected_submission": "v22"
  },
  "selected_submission": "v22"
}
```

Primary為相對v22的跨窗採用研究條件，mechanism為相對同設定重訓C的條件；單seed不能證明跨seed因果。native不是門檻，仍完整列出。兩項全過亦不自動取代使用者選定。

## 五層與分群

|窗口|層|v22|C|T|T−v22|T−C|
|---|---|---:|---:|---:|---:|---:|
|development|native|1,834,776.38 / 1,826,042.86|1,834,776.38 / 1,826,042.86|1,857,153.80 / 1,848,159.34|22,377.42 / 22,116.48|22,377.42 / 22,116.48|
|development|base|1,479,752.85 / 1,470,202.73|1,479,752.85 / 1,470,202.73|1,509,429.97 / 1,499,495.41|29,677.12 / 29,292.68|29,677.12 / 29,292.68|
|development|reference|1,478,842.86 / 1,470,966.77|1,478,842.86 / 1,470,966.77|1,481,399.80 / 1,473,466.14|2,556.94 / 2,499.37|2,556.94 / 2,499.37|
|development|spatial|1,469,402.11 / 1,461,655.61|1,469,402.11 / 1,461,655.61|1,472,395.79 / 1,464,591.70|2,993.68 / 2,936.09|2,993.68 / 2,936.09|
|development|median|1,407,387.07 / 1,399,717.74|1,407,387.07 / 1,399,717.74|1,407,323.81 / 1,399,653.26|-63.26 / -64.48|-63.26 / -64.48|
|confirmation|native|1,776,517.58 / 1,834,121.89|1,776,517.58 / 1,834,121.89|1,774,816.12 / 1,830,215.68|-1,701.46 / -3,906.21|-1,701.46 / -3,906.21|
|confirmation|base|1,623,355.77 / 1,696,866.14|1,623,355.77 / 1,696,866.14|1,629,895.81 / 1,703,752.71|6,540.04 / 6,886.57|6,540.04 / 6,886.57|
|confirmation|reference|1,407,726.19 / 1,465,885.99|1,407,726.19 / 1,465,885.99|1,410,940.04 / 1,469,339.76|3,213.86 / 3,453.77|3,213.86 / 3,453.77|
|confirmation|spatial|1,403,580.19 / 1,460,855.82|1,403,580.19 / 1,460,855.82|1,406,615.16 / 1,464,107.79|3,034.97 / 3,251.97|3,034.97 / 3,251.97|
|confirmation|median|1,383,228.41 / 1,447,153.42|1,383,228.41 / 1,447,153.42|1,385,668.08 / 1,449,780.51|2,439.67 / 2,627.10|2,439.67 / 2,627.10|
|auxiliary|native|2,825,445.45 / 2,540,990.71|2,825,445.45 / 2,540,990.71|2,821,766.36 / 2,537,032.86|-3,679.09 / -3,957.85|-3,679.09 / -3,957.85|
|auxiliary|base|2,549,511.97 / 2,247,696.15|2,549,511.97 / 2,247,696.15|2,549,268.13 / 2,247,014.73|-243.84 / -681.42|-243.84 / -681.42|
|auxiliary|reference|2,450,073.42 / 2,135,883.50|2,450,073.42 / 2,135,883.50|2,451,139.86 / 2,136,383.76|1,066.44 / 500.26|1,066.44 / 500.26|
|auxiliary|spatial|2,451,856.28 / 2,137,962.02|2,451,856.28 / 2,137,962.02|2,452,211.21 / 2,137,700.93|354.93 / -261.09|354.93 / -261.09|
|auxiliary|median|2,309,352.36 / 2,023,165.66|2,309,352.36 / 2,023,165.66|2,309,775.15 / 2,023,758.89|422.80 / 593.23|422.80 / 593.23|

|窗口|比較|seen差|new差|>5%違例數|最大相對退步|
|---|---|---:|---:|---:|---|
|development|T-v22|-146.84|-16.90|0|無|
|development|T-C|-146.84|-16.90|0|無|
|development|C-v22|0.00|0.00|0|無|
|confirmation|T-v22|789.76|3,688.57|0|無|
|confirmation|T-C|789.76|3,688.57|0|無|
|confirmation|C-v22|0.00|0.00|0|無|
|auxiliary|T-v22|1,077.24|313.60|0|無|
|auxiliary|T-C|1,077.24|313.60|0|無|
|auxiliary|C-v22|0.00|0.00|0|無|

所有城市、月份、季度、固定群與空群保留在原評分JSON；互相重疊的群不能加總。五價帶的全窗貢獻另存摘要JSON，核對加總至1e−6元，只是事後診斷，不按未知真價調價。

## 全部24個條件配對區間

|窗口|比較|方法／指標|點差|2.5%～97.5%|有效／無效|
|---|---|---|---:|---|---:|
|development|T−v22|project/overall|-63.26|-3,216.48～3,378.36|2000/0|
|development|T−v22|project/weighted|-64.48|-3,142.55～3,337.27|2000/0|
|development|T−v22|month/overall|-63.26|-2,870.12～3,724.67|2000/0|
|development|T−v22|month/weighted|-64.48|-2,772.91～3,930.71|2000/0|
|development|T−C|project/overall|-63.26|-3,216.48～3,378.36|2000/0|
|development|T−C|project/weighted|-64.48|-3,142.55～3,337.27|2000/0|
|development|T−C|month/overall|-63.26|-2,870.12～3,724.67|2000/0|
|development|T−C|month/weighted|-64.48|-2,772.91～3,930.71|2000/0|
|confirmation|T−v22|project/overall|2,439.67|-910.44～6,019.66|2000/0|
|confirmation|T−v22|project/weighted|2,627.10|-1,129.72～6,608.33|2000/0|
|confirmation|T−v22|month/overall|2,439.67|-897.74～6,145.93|2000/0|
|confirmation|T−v22|month/weighted|2,627.10|-1,360.89～6,017.31|2000/0|
|confirmation|T−C|project/overall|2,439.67|-910.44～6,019.66|2000/0|
|confirmation|T−C|project/weighted|2,627.10|-1,129.72～6,608.33|2000/0|
|confirmation|T−C|month/overall|2,439.67|-897.74～6,145.93|2000/0|
|confirmation|T−C|month/weighted|2,627.10|-1,360.89～6,017.31|2000/0|
|auxiliary|T−v22|project/overall|422.80|-2,523.64～3,522.45|2000/0|
|auxiliary|T−v22|project/weighted|593.23|-1,922.88～3,201.19|2000/0|
|auxiliary|T−v22|month/overall|422.80|-589.15～1,747.16|2000/0|
|auxiliary|T−v22|month/weighted|593.23|-372.75～1,589.10|2000/0|
|auxiliary|T−C|project/overall|422.80|-2,523.64～3,522.45|2000/0|
|auxiliary|T−C|project/weighted|593.23|-1,922.88～3,201.19|2000/0|
|auxiliary|T−C|month/overall|422.80|-589.15～1,747.16|2000/0|
|auxiliary|T−C|month/weighted|593.23|-372.75～1,589.10|2000/0|

{"negative": 0, "crosses_zero": 24, "positive": 0, "unavailable": 0}；每類原固定2000次，本摘要不重抽。CI不是Private勝率。

## 時間與限制

|步驟|秒|
|---|---:|
|development C fit|16.391|
|development T fit|16.203|
|confirmation C fit|21.656|
|confirmation T fit|21.891|
|auxiliary C fit|9.141|
|auxiliary T fit|9.360|
|six_NN_fit_seconds|94.642|
|main_execution_seconds|167.750|
|evaluation_execution_seconds|51.422|

fit已包含於main，不能重複相加，以上非聊天或開發工時。原unknown embedding已有訓練，聯動不遮numeric prior或其他地址／地理，並不等同真新案；各欄邊際人工遮罩率不變，也不保證CUDA重訓逐位確定。
反覆年度選型、單seed、既有R42/R18遮罩反證與2023非價格地理的歷史可用性限制保留，不是新盲測或完整historical-as-of。無mask率／欄位／seed／alpha搜尋，未全量fit、未新CSV。保存模型重載不是乾淨環境從零重現。
本摘要只讀JSON metadata與聚合結果，不載模型、逐筆array或真價、不重算MAE或bootstrap。
使用者要求本輪完成後暫停；R106完成與文件交接後停止研究，不啟動R107或背景訓練。
[計畫](round106_plan.md) · [完整評分](../outputs/round106/evaluation/results.json) · [摘要JSON](../outputs/round106_summary/summary.json) · [摘要程式](../src/summarize_round106.py)
