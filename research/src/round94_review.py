"""Summarize completed R94 artifacts; no training, test prediction, or selection mutation."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
STAGES=("development","confirmation","auxiliary")
LABELS=("中間年度","最新年度","最舊年度")
C="R94"

def read(p):return json.loads((ROOT/p).read_text(encoding="utf8"))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def write(p,d):
 with (ROOT/p).open("x",encoding="utf8") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def main():
 r=read("outputs/round94/results.json");e=read("outputs/round94/execution.json")
 assert e["exit_code"]==0 and r["checked_stages"]==list(STAGES) and r["new_NN_fits"]==3
 g=read("reports/round94_diagnostics.json")
 assert read("outputs/round94/diagnostics/execution.json")["exit_code"]==0
 intervals=[]
 for control in ("v22","v24"):
  paired=read(f"outputs/round94/paired_vs_{control}/results.json")
  assert paired["status"]=="completed" and set(paired["stages"])==set(STAGES)
  assert (ROOT/f"outputs/round94/paired_vs_{control}/results.sha256").read_text().strip()==sha(f"outputs/round94/paired_vs_{control}/results.json")
  for s in STAGES:
   for method in ("project","month"):
    for metric,x in paired["stages"][s][method]["summary"].items():
     intervals.append(dict(control=control,stage=s,method=method,metric=metric,point_delta=x["point_delta"],
      lower=x["percentile_2_5"],upper=x["percentile_97_5"]))
 assert len(intervals)==24
 summary=[]
 for s in STAGES:
  x=r["stages"][s];m=x["metrics"]
  summary.append(dict(stage=s,metrics={c:{k:m[c][k] for k in ("overall_mae","weighted_mae")} for c in ("v22","v24",C)},
   delta_v22={k:m[C][k]-m["v22"][k] for k in ("overall_mae","weighted_mae")},
   group_deltas={k:m[C]["strata"][k]["mae"]-m["v22"]["strata"][k]["mae"] for k in ("seen","new_nonmissing")},
   price_bands=g["stages"][s]["price_bands"],fit_seconds=x["fit_seconds"],nn_branch_metrics=x["nn_branch_metrics"]))
 latest=-summary[1]["delta_v22"]["overall_mae"]/summary[1]["metrics"]["v22"]["overall_mae"]
 gate={"all_three_overall_improve":all(s["delta_v22"]["overall_mae"]<0 for s in summary),
       "all_three_weighted_improve":all(s["delta_v22"]["weighted_mae"]<0 for s in summary),
       "latest_at_least_half_percent":latest>=.005}
 passed=all(gate.values())
 evidence=["reports/round94_plan.md","reports/round94_upstream_review.md","reports/round94_integration_inventory.md",
  "reports/round94_trainer_selfcheck_v2.json","reports/r94_validation_audit.md","outputs/round94_upstream/manifest.json",
  "outputs/round94_smoke_v2_20260929/smoke.json","outputs/round94/results.json","outputs/round94/execution.json",
  "reports/round94_diagnostics.json","outputs/round94/paired_vs_v22/results.json","outputs/round94/paired_vs_v24/results.json"]
 decision={"round":94,"candidate":"fixed_small_FT_style_prior_ratio_NN","timestamp_utc":datetime.now(timezone.utc).isoformat(),
  "adoption_gate_passed":passed,"gate_components":gate,"latest_fractional_improvement":latest,
  "recommendation":"eligible_for_new_origin_check_no_automatic_adoption" if passed else "retain_research_no_final_fit_no_CSV",
  "new_model_fits":3,"new_tree_fits":0,"new_NN_fits":3,"fit_seconds":sum(s["fit_seconds"] for s in summary),
  "main_elapsed_seconds":e["elapsed_seconds"],"stage_summary":summary,"paired_intervals":intervals,
  "protected_before_after_equal":r["protected_before"]==r["protected_after"],"selector_changed":False,
  "selected_submission":"v22","private_score":None,"new_test_predictions":False,"new_CSV":False,
  "source_sha256":r["source_sha256"],"review_source_sha256":sha("src/round94_review.py"),
  "evidence_sha256":{p:sha(p) for p in evidence},
  "limitations":["Repeated-window architecture selection is not blind validation.",
   "Embedding width, dropout layout and architecture change together; no pure attention causal attribution.",
   "Native branches and final median can move in opposite directions.",
   "GPU controls not contemporaneously refit; same seed does not preserve RNG or guarantee determinism.",
   "Paired intervals do not correct model selection bias or represent Private win probabilities.",
   "Saved model reload is not clean-environment end-to-end retraining."]}
 assert decision["protected_before_after_equal"]
 write("outputs/round94/decision_review.json",decision)
 with (ROOT/"outputs/round94/decision_review.sha256").open("x",encoding="utf8") as f:f.write(sha("outputs/round94/decision_review.json")+"\n")
 table="\n".join(f'|{lab}|{s["metrics"]["v22"]["overall_mae"]:,.2f}|{s["metrics"][C]["overall_mae"]:,.2f}|{s["delta_v22"]["overall_mae"]:+,.2f}|{s["delta_v22"]["weighted_mae"]:+,.2f}|' for lab,s in zip(LABELS,summary))
 groups="\n".join(f'|{lab}|{s["group_deltas"]["seen"]:+,.2f}|{s["group_deltas"]["new_nonmissing"]:+,.2f}|' for lab,s in zip(LABELS,summary))
 bands="\n".join(f'|{lab}|{b["band"]}|{b["rows"]:,}|{b[C]-b["v22"]:+,.2f}|' for lab,s in zip(LABELS,summary) for b in s["price_bands"])
 count={c:{"negative":sum(i["control"]==c and i["upper"]<0 for i in intervals),
  "cross_zero":sum(i["control"]==c and i["lower"]<=0<=i["upper"] for i in intervals),
  "positive":sum(i["control"]==c and i["lower"]>0 for i in intervals)} for c in ("v22","v24")}
 interval_table="\n".join(f'|{i["control"]}|{i["stage"]}|{i["method"]}|{i["metric"]}|{i["point_delta"]:+,.2f}|[{i["lower"]:+,.2f}, {i["upper"]:+,.2f}]|' for i in intervals)
 layers="\n".join(f'|{lab}|{layer}|{m["overall_mae"]:,.2f}|' for lab,s in zip(LABELS,summary) for layer,m in s["nn_branch_metrics"].items())
 outcome="固定數值門檻通過，仍須分群審查及新切點重建，不自動採用。" if passed else "固定數值門檻未過，保留研究，不全量fit／新增CSV。"
 report=f"""# R94：欄位Attention prior-ratio回歸

{outcome} 使用者選定仍為v22，Private未知。

## 動機與做法

測試同一筆交易內的欄位交互，固定替換prior NN；不是跨成交資料的attention。沿同窗R63九個prepared成員、144數值／23類別、group-OOF fit prior與full-fit query prior及日期加權總價MAE。168tokens、d64、2blocks／4heads，最後CLS query/residual/FFN裁切，head Softplus ratio。固定30epochs、seed42、AdamW、micro256累積2048，不以query挑epoch或搜尋架構。

只把原NN分支變成 .75*(.5*R54area+.5*R94prior)+.25*LGB，再原reference/spatial，與保存R46log／R79unit取median。舊R71與v22整鏈重播，來源與保存模型重載依主流程收據核對。原始上游Apache-2.0 source/revision/license保留，末層計算裁切、Softplus頭、損失及optimizer都是本題改編，非完整論文復刻。

## 三期結果

單位TWD MAE；差額=R94−v22，負數較好。保留三期，不只展示有利窗口。

|期別|v22 MAE|R94 MAE|整體差額|固定加權差額|
|---|---:|---:|---:|---:|
{table}

最新相對改善{latest*100:.4f}%；三期整體全改善={gate["all_three_overall_improve"]}、三期加權全改善={gate["all_three_weighted_improve"]}、最新至少0.5%={gate["latest_at_least_half_percent"]}。

|期別|既有建案ΔMAE|新建案ΔMAE|
|---|---:|---:|
{groups}

|期別|真價帶|筆數|群ΔMAE|
|---|---|---:|---:|
{bands}

真價分群僅事後診斷，未知test真價不能用作推論分流。所有季度／城市／testmix保留於完整diagnostics；testmix是歷史支持群重新加權，非Private。

## 原生分支與完整集成

|期別|層|MAE|
|---|---|---:|
{layers}

分支改善不保證median改善，層差也不能相加解釋因果。詳看結果中的完整分支與保存控制指標，不能只用native勝負取代完整組合。

## 全部配對區間

|對照|窗口|抽法|指標|點差額|95%區間|
|---|---|---|---|---:|---|
{interval_table}

對v22負／跨零／正為{count["v22"]["negative"]}／{count["v22"]["cross_zero"]}／{count["v22"]["positive"]}；對v24為{count["v24"]["negative"]}／{count["v24"]["cross_zero"]}／{count["v24"]["positive"]}。這些區間相依、未校正多輪選型，不能称Private勝率或未來市場保證。窗口反覆重用，不是盲測。

## 時間與限制

三次真NN fit共{decision["fit_seconds"]:.3f}秒，主流程{e["elapsed_seconds"]:.3f}秒包含fit；合成資源檢查、資料與診斷另計，不能重複加總。未測量整段聊天開發時間。GPU固定seed不保證決定性，初始化/dropout使shuffle亂數不與原MLP逐位相同；舊對照未同時重訓。

原編碼、prior與日期權重有保存來源核對；trainer不接受query標籤，主driver仍載canonical y而只用query部分score，不誇稱query bytes从未載入。embedding寬度、dropout位置與模型結構一起改，不能將結果全部歸因attention。保存重載並非全新ML環境從原始資料端到端重訓。

## 證據入口

- [固定計畫](round94_plan.md)、[上游來源](round94_upstream_review.md)、[整合契約](round94_integration_inventory.md)
- [CPU合成](round94_trainer_selfcheck_v2.json)、[GPU合成成本](../outputs/round94_smoke_v2_20260929/smoke.json)、[獨立preflight](r94_validation_audit.md)
- [完整結果](../outputs/round94/results.json)、[完整分群](round94_diagnostics.md)、[決策](../outputs/round94/decision_review.json)
- [對v22配對](round94_paired_vs_v22.md)、[對v24配對](round94_paired_vs_v24.md)

不改原CSV、selector或v22簡報。本輪没有test預測、Public/Private新分數或Kaggle操作。
"""
 with (ROOT/"reports/round94_report.md").open("x",encoding="utf8") as f:f.write(report)
 print(json.dumps({"gate":gate,"recommendation":decision["recommendation"],"fit_seconds":decision["fit_seconds"],
  "main_seconds":e["elapsed_seconds"],"interval_counts":count},ensure_ascii=False))
if __name__=="__main__":main()
