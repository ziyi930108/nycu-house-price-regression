"""Summarize completed R93 artifacts without fitting or changing the selection."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
ROUND=93;C="R93"
STAGES=("development","confirmation","auxiliary")
LABELS=("中間年度","最新年度","最舊年度")
def read(p):return json.loads((ROOT/p).read_text(encoding="utf8"))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
 r=read("outputs/round93/results.json");e=read("outputs/round93/execution.json")
 assert e["exit_code"]==0 and r["checked_stages"]==list(STAGES)
 g=read("reports/round93_diagnostics.json")
 assert read("outputs/round93/diagnostics/execution.json")["exit_code"]==0
 intervals=[]
 for control in ("v22","v24"):
  paired=read(f"outputs/round93/paired_vs_{control}/results.json")
  for s in STAGES:
   for method in ("project","month"):
    for metric,x in paired["stages"][s][method]["summary"].items():
     intervals.append(dict(control=control,stage=s,method=method,metric=metric,point_delta=x["point_delta"],
      lower=x["percentile_2_5"],upper=x["percentile_97_5"]))
 stage_summary=[]
 for s in STAGES:
  x=r["stages"][s];m=x["metrics"]
  wm=read(f"outputs/round93/{s}/model/project_weight_check.json")
  stage_summary.append(dict(stage=s,metrics={c:{k:m[c][k] for k in ("overall_mae","weighted_mae")} for c in ("v22","v24",C)},
   delta_v22={k:m[C][k]-m["v22"][k] for k in ("overall_mae","weighted_mae")},
   group_deltas={k:m[C]["strata"][k]["mae"]-m["v22"]["strata"][k]["mae"] for k in ("seen","new_nonmissing")},
   price_bands=g["stages"][s]["price_bands"],fit_seconds=x["fit_seconds"],weight_check=wm))
 latest=-stage_summary[1]["delta_v22"]["overall_mae"]/stage_summary[1]["metrics"]["v22"]["overall_mae"]
 gate={"all_three_overall_improve":all(s["delta_v22"]["overall_mae"]<0 for s in stage_summary),
 "all_three_weighted_improve":all(s["delta_v22"]["weighted_mae"]<0 for s in stage_summary),
 "latest_at_least_half_percent":latest>=.005}
 passed=all(gate.values())
 decision={"round":93,"candidate":"fit_project_sqrt_count_adjusted_recency_unit_CatBoost",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"adoption_gate_passed":passed,"gate_components":gate,
 "latest_fractional_improvement":latest,
 "recommendation":"eligible_for_new_origin_check_no_automatic_adoption" if passed else "retain_research_no_final_fit_no_CSV",
 "new_native_fits":3,"new_NN_fits":0,"fit_seconds":sum(s["fit_seconds"] for s in stage_summary),
 "main_elapsed_seconds":e["elapsed_seconds"],"stage_summary":stage_summary,"paired_intervals":intervals,
 "protected_before_after_equal":r["protected_before"]==r["protected_after"],"selector_changed":False,
 "selected_submission":"v22","private_score":None,"new_test_predictions":False,"new_CSV":False,
 "source_sha256":r["source_sha256"],"review_source_sha256":sha("src/round93_review.py"),
 "evidence_sha256":{p:sha(p) for p in ["reports/round93_plan.md","reports/round93_helper_selfcheck.json","reports/r93_validation_audit.md",
 "outputs/round93_fit_audit/results.json","outputs/round93/results.json","outputs/round93/execution.json",
 "reports/round93_diagnostics.json","outputs/round93/paired_vs_v22/results.json","outputs/round93/paired_vs_v24/results.json"]},
 "limitations":["Repeated-window adaptive research; not a blind holdout.","Weights alter the learning objective from date-only TWD-L1.",
 "Normalization upweights missing-name and singleton rows; it does not give equal weight to physical projects.",
 "Fixed-seed GPU controls not contemporaneously refit; no pure causal attribution.",
 "Paired intervals do not correct selection bias or represent Private win probabilities.",
 "Saved model reload is not clean-environment end-to-end retraining."]}
 out=ROOT/"outputs/round93/decision_review.json"
 with out.open("x",encoding="utf8") as f:json.dump(decision,f,ensure_ascii=False,indent=2);f.write("\n")
 with out.with_suffix(".sha256").open("x",encoding="utf8") as f:f.write(sha("outputs/round93/decision_review.json")+"\n")
 rows="\n".join(f'|{lab}|{s["metrics"]["v22"]["overall_mae"]:,.2f}|{s["metrics"][C]["overall_mae"]:,.2f}|{s["delta_v22"]["overall_mae"]:+,.2f}|{s["delta_v22"]["weighted_mae"]:+,.2f}|' for lab,s in zip(LABELS,stage_summary))
 bandrows="\n".join(f'|{lab}|{b["band"]}|{b["rows"]:,}|{b[C]-b["v22"]:+,.2f}|' for lab,s in zip(LABELS,stage_summary) for b in s["price_bands"])
 grouprows="\n".join(f'|{lab}|{s["group_deltas"]["seen"]:+,.2f}|{s["group_deltas"]["new_nonmissing"]:+,.2f}|' for lab,s in zip(LABELS,stage_summary))
 icount={c:{"negative":sum(i["control"]==c and i["upper"]<0 for i in intervals),
 "cross_zero":sum(i["control"]==c and i["lower"]<=0<=i["upper"] for i in intervals),
 "positive":sum(i["control"]==c and i["lower"]>0 for i in intervals)} for c in ("v22","v24")}
 outcome="固定數值門檻通過，仍需獨立分群審查及新切點重建；不自動採用。" if passed else "固定數值門檻未過，保留研究，不全量fit、不新增CSV。"
 report=f"""# R93：建案樣本量平方根折減

{outcome} v22唯一提交選定不變，Private未知。

## 為什麼與怎麼做

大型建案有許多相似交易，可能在有限樹的split/leaf優化中占較大比重。R5曾提出密度折減但未實測；本輪在R79近期加權單價樹上固定測試其延伸，並未把概念包裝成全新或預設已過擬合。

fit-only city|district|完整project按R5原正規化計數n。具名raw q=1/sqrt(n)，缺案raw q=1；原w0=(A/100)*date64，c=sum(w0)/sum(w0*q)，新Pool w=w0*q*c。只保留總權重尺度，不保留各群權重；c約6–7，缺案／單例亦共同放大。改變的是訓練樣本的相對影響，query預測不乘q/c，也不重新估計建案密度。

原95欄／23cat、unit target及TWD inverse、depth8／2999樹／seed42皆保持；.75新native+.25原LGB，原reference/spatial後與NN/log取median。不併入其他新配方，不掃強度或參數。

## 三期結果

候選−v22，元，負值較好。實際三期、全部群及兩對照已完成，沒有只展示較有利年度。

|期別|v22 MAE|R93 MAE|整體差額|加權差額|
|---|---:|---:|---:|---:|
{rows}

最新相對改善{latest*100:.4f}%（負數代表退步）。固定門檻：三期整體全改善={gate["all_three_overall_improve"]}、三期加權全改善={gate["all_three_weighted_improve"]}、最新至少0.5%={gate["latest_at_least_half_percent"]}。

|期別|seen群MAE差|new_nonmissing群MAE差|
|---|---:|---:|
{grouprows}

|期別|真價帶|筆數|群MAE差|
|---|---|---:|---:|
{bandrows}

真價帶僅事後診斷，不能以未知test真價決定採哪個模型。完整城市／季度、native/base/reference/spatial/median每層、test-X組成估計在結果與diagnostics；X組成估計不是test MAE。

## 區間、驗證與時間

對v22的12區間：偏改善{icount["v22"]["negative"]}、跨零{icount["v22"]["cross_zero"]}、偏退{icount["v22"]["positive"]}；對v24：{icount["v24"]["negative"]}／{icount["v24"]["cross_zero"]}／{icount["v24"]["positive"]}。這些建案／月份區間不獨立、不校正反覆選型，不能稱Private勝率。

每期真fit前先獨立重建count/q/c/w並核對純X預檢的值SHA，總Pool尺度相同；fit/query日期、索引、原特徵／geo與原日期權重綁定。helper53項合成通過（2 FakeModel，0真fit），首次synthetic查錯legacy metadata層的KeyError保留，沒有改權重規則。

三次真native fit合計{decision["fit_seconds"]:.3f}秒；main {e["elapsed_seconds"]:.3f}秒（已包含fit）。新CBM保存後全query raw及TWD重載精確，原v22四層完整向量與指標重播一致。未量測整段開發聊天耗時。

fit計數／權重只用當期fit，但來源train_features pickle整表反序列化後再取fit；主driver也載完整canonical y再只傳fity給fit，queryy只評分。不能誇稱連query的bytes都沒載過。訓練目標已變為date*q*c加權TWD-L1，需以原交易列MAE判斷；模型更均衡不保證成績更好。GPU固定seed不保證位元決定性，舊控制未同期重訓。

## 追溯入口

- [固定計畫](round93_plan.md)、[歷史方向盤點](next_regression_direction_after_r92.md)
- [53項合成收據](round93_helper_selfcheck.json)、[訓練前審核](r93_validation_audit.md)
- [fit純X預檢](../outputs/round93_fit_audit/results.json)
- [完整結果](../outputs/round93/results.json)、[完整診斷](round93_diagnostics.md)
- [對v22區間](round93_paired_vs_v22.md)、[對v24區間](round93_paired_vs_v24.md)
- [不可變採用審查](../outputs/round93/decision_review.json)

原source、模型、CSV、selector與選定保持。沒有新test預測、Public分數或Kaggle操作；保存重載不等於乾淨環境從零重訓整條鏈。
"""
 with (ROOT/"reports/round93_report.md").open("x",encoding="utf8") as f:f.write(report)
 print(json.dumps({"gate":gate,"recommendation":decision["recommendation"],"fit_seconds":decision["fit_seconds"],"main_seconds":e["elapsed_seconds"],"interval_counts":icount},ensure_ascii=False))
if __name__=="__main__":main()
