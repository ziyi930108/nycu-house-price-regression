"""Audit saved R87 CPU results, write an immutable review and report; no fits."""
from pathlib import Path
import datetime, hashlib, json, math, time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/round87"
STAGES=("development","confirmation","auxiliary")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name):
 p=ROOT/name
 if p.with_suffix(p.suffix+".sha256").exists():
  assert p.with_suffix(p.suffix+".sha256").read_text().split()[0]==sha(p),name
 return json.loads(p.read_text(encoding="utf-8"))
def put(p,obj):
 with p.open("x",encoding="utf-8",newline="\n") as f:json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True);f.write("\n")
 with p.with_suffix(p.suffix+".sha256").open("x",encoding="ascii") as f:f.write(sha(p)+"\n")
def fmt(v):return f"{v:,.2f}"
def run():
 start=time.monotonic()
 names=["outputs/round87/results.json","outputs/round87/execution.json","reports/round87_diagnostics.json",
 "outputs/round87/diagnostics/execution.json","outputs/round87/paired_vs_v22/results.json",
 "outputs/round87/paired_vs_v24/results.json","outputs/round79/results.json",
 "outputs/round83/results.json","outputs/round85/results.json","reports/round87_plan.md"]
 inputs={n:sha(ROOT/n) for n in names}
 r=load(names[0]);exe=load(names[1]);d=load(names[2]);de=load(names[3])
 pairs={c:load(f"outputs/round87/paired_vs_{c}/results.json") for c in ("v22","v24")}
 assert r["checked_stages"]==d["checked_stages"]==list(STAGES)
 assert exe["exit_code"]==de["exit_code"]==0
 assert r["new_fits"]==r["models_loaded"]==r["model_predict_calls"]==0
 assert not r["new_CSV"] and not r["new_test_predictions"] and r["adaptive_after_R85_results"]
 for n,h in {**r["input_sha256"],**r["source_sha256"],**r["protected_before"]}.items():assert sha(ROOT/n)==h,n
 for s in STAGES:
  z=r["stages"][s]
  assert z["v22_base_reference_spatial_median_full_vectors_exact"] and z["controls_full_metrics_exact"]
  for n,h in z["artifact_sha256"].items():assert sha(ROOT/n)==h,n
  assert d["stages"][s]["all_full_saved_metrics_exact"]
  assert set(z["metrics"])=={"v11","v12","v5","v22","v24","R87"}
 table=[];bands=[];strata=[];quarters=[];mix=[];intervals=[]
 for s in STAGES:
  z=r["stages"][s];m=z["metrics"];a=m["R87"]
  table.append({"stage":s,"dates":z["dates"],"fit_rows":z["fit_rows"],"query_rows":z["query_rows"],
  "v22":{k:m["v22"][k] for k in ("overall_mae","weighted_mae")},"v24":{k:m["v24"][k] for k in ("overall_mae","weighted_mae")},
  "R87":{k:a[k] for k in ("overall_mae","weighted_mae")},
  "delta_vs_v22":{k:a[k]-m["v22"][k] for k in ("overall_mae","weighted_mae")},
  "delta_vs_v24":{k:a[k]-m["v24"][k] for k in ("overall_mae","weighted_mae")}})
  ds=d["stages"][s]
  for b in ds["price_bands"]:
   bands.append({"stage":s,"band":b["band"],"rows":b["rows"],"v22":b["v22"],"v24":b["v24"],"R87":b["R87"],
   "delta_vs_v22":b["R87"]-b["v22"],"delta_vs_v24":b["R87"]-b["v24"],
   "overall_delta_contribution_vs_v22":(b["R87"]-b["v22"])*b["rows"]/z["query_rows"]})
  subtotal=sum(b["overall_delta_contribution_vs_v22"] for b in bands if b["stage"]==s)
  assert math.isclose(subtotal,a["overall_mae"]-m["v22"]["overall_mae"],abs_tol=1e-7)
  for label in ("seen","new_nonmissing"):
   strata.append({"stage":s,"group":label,"rows":a["strata"][label]["rows"],"delta_vs_v22":a["strata"][label]["mae"]-m["v22"]["strata"][label]["mae"]})
  for q in ("Q1","Q2","Q3","Q4"):
   quarters.append({"stage":s,"quarter":q,"rows":a["forecast_quarters"][q]["rows"],"delta_vs_v22":a["forecast_quarters"][q]["mae"]-m["v22"]["forecast_quarters"][q]["mae"]})
  tc=ds["test_composition"];mix.append({"stage":s,**tc,"delta_vs_v22":tc["models"]["R87"]-tc["models"]["v22"],"delta_vs_v24":tc["models"]["R87"]-tc["models"]["v24"]})
 for c,p in pairs.items():
  assert p["status"]=="completed" and p["replicates"]==2000 and p["seed"]==20260925
  assert p["fits"]==p["model_predict_calls"]==0 and not p["new_CSV"]
  for s in STAGES:
   for group in ("project","month"):
    for metric in ("overall","weighted"):
     z=p["stages"][s][group]["summary"][metric]
     assert z["valid_replicates"]==2000 and z["invalid_replicates"]==0
     expected=r["stages"][s]["metrics"]["R87"][metric+"_mae"]-r["stages"][s]["metrics"][c][metric+"_mae"]
     assert math.isclose(z["point_delta"],expected,abs_tol=1e-6)
     intervals.append({"control":c,"stage":s,"unit":group,"metric":metric,"point_delta":z["point_delta"],"lower":z["percentile_2_5"],"upper":z["percentile_97_5"],"valid_replicates":z["valid_replicates"]})
 assert len(intervals)==24
 ci_counts={c:{"cross_zero":sum(x["lower"]<=0<=x["upper"] for x in intervals if x["control"]==c),
 "positive":sum(x["lower"]>0 for x in intervals if x["control"]==c),
 "negative":sum(x["upper"]<0 for x in intervals if x["control"]==c)} for c in pairs}
 costs={"main_execution_seconds":exe["elapsed_seconds"],"generic_seconds":de["elapsed_seconds"],
 "paired_vs_v22_seconds":pairs["v22"]["elapsed_seconds"],"paired_vs_v24_seconds":pairs["v24"]["elapsed_seconds"]}
 lines=["# R87：固定等權原生混合結果","",
 "2026-09-28。本輪不建議進行全量 fit／提交；v22選定與原CSV保留。R87在中間、最舊窗口小幅降低整體MAE，但最新窗口整體、新舊案均退步；固定test組成估計在中間、最新窗口也退步。這不證明任何版本的Private最好。","",
 "這是看過R85結果後立項的適應性研究，所有年度窗口已多輪重用，並非新盲測或獨立確認。固定一次0.5／0.5，不追加比例搜尋。","",
 "## 方法與可追溯性","",
 "先合成 base = 0.375×R79原生總價 + 0.375×R85平方根面積原生總價 + 0.25×原LightGBM單價分支總價；再使用該期fit-only reference與spatial，最後和固定NN、log分支取中位數。base按此加法順序計算，沒有平均已修正分支。","",
 "零新訓練、零樹或NN模型載入／predict；有載入保存的reference/spatial狀態，各期分別對原v22及R87重播修正。共6次reference及6次spatial呼叫。並未在本輪重新載入R79/R85 CBM；核對其先前全query重載一致證據與保存原生向量SHA。","",
 "各期索引、日期隔離、特徵／目標／日期權重雜湊、fit-only狀態皆通過；v22 base→reference→spatial→median全向量逐位重現，v11/v12/v5/v22/v24完整metrics一致。所有既有submission、研究selector及user_selection SHA未變。未生成test預測或CSV，未查外部成交答案。","",
 "計畫：[round87_plan.md](round87_plan.md)；程式：[round87_experiment.py](../src/round87_experiment.py)；原始結果：[results.json](../outputs/round87/results.json)。","",
 "## 三期整體結果","",
 "單位TWD，差值為R87減對照；負值改善。weighted沿用既有新舊案權重，並非Kaggle已知分數。","",
 "|窗口|v22 MAE|v24 MAE|R87 MAE|Δv22|Δv24|","|---|---:|---:|---:|---:|---:|"]
 for z in table:lines.append("|"+z["stage"]+"|"+"|".join(fmt(v) for v in (z["v22"]["overall_mae"],z["v24"]["overall_mae"],z["R87"]["overall_mae"],z["delta_vs_v22"]["overall_mae"],z["delta_vs_v24"]["overall_mae"]))+"|")
 lines+=["","|窗口|v22 weighted|v24 weighted|R87 weighted|Δv22|Δv24|","|---|---:|---:|---:|---:|---:|"]
 for z in table:lines.append("|"+z["stage"]+"|"+"|".join(fmt(v) for v in (z["v22"]["weighted_mae"],z["v24"]["weighted_mae"],z["R87"]["weighted_mae"],z["delta_vs_v22"]["weighted_mae"],z["delta_vs_v24"]["weighted_mae"]))+"|")
 lines+=["","|窗口|query日期|fit列|query列|","|---|---|---:|---:|"]
 for z in table:lines.append(f'|{z["stage"]}|{" ~ ".join(z["dates"])}|{z["fit_rows"]:,}|{z["query_rows"]:,}|')
 lines+=["","## 真價分帶及最強反證","",
 "以下分帶只用歷史query真價評分，不是預測時的門檻或調價規則。最後一欄是該群MAE差乘列數占比，五帶合計等於整體Δv22。","",
 "|窗口|真價帶|列數|Δv22群內MAE|Δv24群內MAE|對整體Δv22貢獻|","|---|---|---:|---:|---:|---:|"]
 for b in bands:lines.append(f'|{b["stage"]}|{b["band"]}|{b["rows"]:,}|{fmt(b["delta_vs_v22"])}|{fmt(b["delta_vs_v24"])}|{fmt(b["overall_delta_contribution_vs_v22"])}|')
 lines+=["",
 "中間與最新窗口的5,000萬以下三帶均優於v22，但兩期≥1億群明顯惡化；最新319筆≥1億群內MAE增加約560萬元，抵消多數一般價格交易的進步。最舊期≥1億群反而改善，說明收益不是跨期穩定；該期<1,000萬與2,000–5,000萬仍退步。","",
 "城市也非普遍改善：中間臺北市+343,002.42元、新竹市−404,141.10元；最新新北市+73,446.51元、臺北市+70,910.94元、新竹市−116,199.93元；最舊基隆市+319,322.86元、臺東縣+616,464.67元、臺北市−134,817.50元。各城樣本數、全部城市與月份保留在通用診斷，不能只引用有利城市。","",
 "## 新舊案、外推季度與組成估計","",
 "|窗口|seen Δv22|new Δv22|Q1|Q2|Q3|Q4|","|---|---:|---:|---:|---:|---:|---:|"]
 for s in STAGES:
  a=[x["delta_vs_v22"] for x in strata if x["stage"]==s]+[x["delta_vs_v22"] for x in quarters if x["stage"]==s]
  lines.append("|"+s+"|"+"|".join(fmt(v) for v in a)+"|")
 lines+=["","最新窗口new與seen均退步，且最遠Q4增加37,511.01元，不能用中間期的735元改善代替時間泛化證據。","",
 "固定組成估計只使用原有test-X聚合權重映射歷史誤差；不是test MAE，更不是Private估計。每格至少200歷史列，未覆蓋部分無法據此評估。","",
 "|窗口|R87組成MAE|Δv22|Δv24|test質量覆蓋|未覆蓋test列|","|---|---:|---:|---:|---:|---:|"]
 for z in mix:lines.append(f'|{z["stage"]}|{fmt(z["models"]["R87"])}|{fmt(z["delta_vs_v22"])}|{fmt(z["delta_vs_v24"])}|{z["test_mass_covered"]:.4%}|{z["unsupported_test_rows"]:,}|')
 lines+=["","## 配對不確定性：全部24個區間","",
 "各2,000次，seed20260925，建案群集及3月循環區塊；95%百分位区間，只描述這批保存模型／歷史窗口條件下的抽樣不確定性，不包含選型、重訓或未來移轉，不能解讀為Private勝率。","",
 "|對照|窗口|重抽單位|指標|點差|2.5%|97.5%|","|---|---|---|---|---:|---:|---:|"]
 for z in intervals:lines.append(f'|{z["control"]}|{z["stage"]}|{z["unit"]}|{z["metric"]}|{fmt(z["point_delta"])}|{fmt(z["lower"])}|{fmt(z["upper"])}|')
 lines+=["",
 "對v22的12區間全部跨0，沒有一致負區間支持優勢。對v24最新期4區間全部為正；最舊期3/4為正，其餘跨0。這些是不利的研究證據，同樣受窗口重用限制。","",
 "## 實際耗時與產物","",
 "|工作|實測秒|","|---|---:|"]
 for k,v in costs.items():lines.append(f"|{k}|{v:.3f}|")
 lines+=["",
 "两份paired並行執行，表中各自elapsed不能直接相加當wall-clock；以上不含先前R85訓練及人工審查。","",
 "- [通用分群報告](round87_diagnostics.md)／[完整JSON](round87_diagnostics.json)",
 "- [對v22配對報告](round87_paired_vs_v22.md)／[對v24配對報告](round87_paired_vs_v24.md)",
 "- [執行收據](../outputs/round87/execution.json)／[審查決策](../outputs/round87/decision_review.json)",
 "- [審查來源](../src/round87_review.py)；原生向量、R/S診斷、三期候選與SHA在outputs/round87。",
 "",
 "不推進本候選的理由是最新年度、新舊案、遠期與test組成反證，且相對v22沒有穩定的配對區間優勢；不因Public調比例。保留R87供後續方法分析，不改v22價格或使用者選定。"]
 report=ROOT/"reports/round87_report.md"
 with report.open("x",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines)+"\n")
 review={"round":87,"status":"completed_review","recommendation":"do_not_advance_R87_to_final_fit",
 "reason":"Latest-window overall/weighted/new/seen/Q4 degradation, first-two-window test-composition degradation, no paired CI wholly below zero versus v22.",
 "created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"adaptive_after_R85_results":True,"fresh_blind_validation":False,
 "Private_score":None,"public_used_for_tuning":False,"new_fits":0,"new_test_predictions":False,"new_CSV":False,"selector_changed":False,
 "source_sha256":{str(Path(__file__).relative_to(ROOT)).replace("\\","/"):sha(Path(__file__))},
 "input_sha256":inputs,"all_existing_protected_hashes_verified":True,
 "all_stage_artifact_hashes_verified":True,"v22_all_four_vectors_exact_all_stages":True,
 "table":table,"price_bands":bands,"strata_delta_vs_v22":strata,"quarters_delta_vs_v22":quarters,"test_composition":mix,
 "all_24_intervals":intervals,"interval_counts":ci_counts,"measured_costs":costs,
 "report_sha256":sha(report),"elapsed_seconds":time.monotonic()-start}
 put(OUT/"decision_review.json",review)
 print(json.dumps({"status":review["status"],"recommendation":review["recommendation"],"interval_counts":ci_counts,
 "report_sha256":review["report_sha256"],"elapsed_seconds":review["elapsed_seconds"]}))
if __name__=="__main__":run()
