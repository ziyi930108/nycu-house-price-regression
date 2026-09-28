"""Summarize frozen R88 aggregate evidence. No candidate, model, or GPU calls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
STAGES=("development","confirmation","auxiliary")
LABELS={"development":"開發／中間","confirmation":"確認／最新","auxiliary":"輔助／最舊"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();inputs={}
    def read(name):
        p=ROOT/name;h=sha(p);inputs[name]=h
        side=p.with_suffix(".sha256")
        if side.exists():assert side.read_text(encoding="utf8").strip()==h,name
        return json.loads(p.read_text(encoding="utf8"))
    r=read("outputs/round88/results.json");ex=read("outputs/round88/execution.json")
    d=read("reports/round88_diagnostics.json");de=read("outputs/round88/diagnostics/execution.json")
    pairs={c:read("outputs/round88/paired_vs_"+c+"/results.json") for c in ("v22","v24")}
    assert r["status"]=="three_periods_completed_no_adoption_decision" and r["checked_stages"]==list(STAGES)
    assert r["new_fits"]==r["models_loaded"]==r["model_predict_calls"]==r["new_reference_or_spatial_states"]==0
    assert ex["exit_code"]==de["exit_code"]==0 and ex["failure"] is None and de["failure"] is None
    assert r["protected_before"]==r["protected_after"] and not r["new_CSV"] and not r["new_test_predictions"] and not r["test_data_loaded"]
    assert r["reference_predict_calls"]==r["spatial_predict_calls"]==9
    assert all(p["status"]=="completed" and len(p["stages"])==3 for p in pairs.values())
    summary={}
    for s in STAGES:
        a=r["stages"][s];assert a["controls_full_metrics_exact"] and a["original_v22_median_exact"]
        assert a["all_original_reference_spatial_and_diagnostics_exact"] and a["query_labels_used_only_for_evaluation"]
        assert a["branch_masks_identical"] and a["changed_final_rows"]==a["union_mask_rows"]
        assert all(q["reference_replayed_exact"] and q["spatial_replayed_exact"] and q["diagnostics_replayed_exact"] and q["unmasked_unchanged"] and q["masked_equals_base"] for q in a["branch_policy_diagnostics"].values())
        assert d["stages"][s]["all_full_saved_metrics_exact"]
        for k in ("v22","v24","R88"):assert a["metrics"][k]==d["stages"][s]["metrics"][k]
        summary[s]={"metrics":{k:a["metrics"][k] for k in ("v22","v24","R88")},
           "differences":{k:a["differences"][k] for k in ("v22","v24")},
           "test_composition":d["stages"][s]["test_composition"],
           "price_bands":d["stages"][s]["price_bands"],
           "mask_rows":a["union_mask_rows"],"query_rows":a["query_rows"],
           "branch_policy_diagnostics":a["branch_policy_diagnostics"]}
    intervals=[]
    for control,p in pairs.items():
        for s in STAGES:
            for method in ("project","month"):
                for metric,z in p["stages"][s][method]["summary"].items():
                    intervals.append(dict(control=control,stage=s,method=method,metric=metric,
                         point_delta=z["point_delta"],percentile_2_5=z["percentile_2_5"],percentile_97_5=z["percentile_97_5"],
                         interpretation="worse" if z["percentile_2_5"]>0 else "better" if z["percentile_97_5"]<0 else "crosses_zero"))
    assert len(intervals)==24
    assert all(summary[s]["differences"][c][m]["delta_TWD"]>0 for s in STAGES for c in ("v22","v24") for m in ("overall","weighted"))
    assert all(summary[s]["test_composition"]["models"]["R88"]>summary[s]["test_composition"]["models"][c] for s in STAGES for c in ("v22","v24"))
    record=dict(round=88,candidate="no_trend_when_no_reference_or_spatial_support",review_type="recommendation_only_not_selection_mutation",
       timestamp_utc=datetime.now(timezone.utc).isoformat(),recommendation="retain_research_do_not_final_fit_or_export",
       reason="All three overall/weighted/test-composition MAEs worsen vs both v22 and v24; some seen/lower-price/upper-tail benefits do not compensate new-project and ordinary-price costs.",
       strongest_counterevidence_to_rejection=[
         "Development >=100m band improves by 1,101,240.07 TWD (182 rows), auxiliary by 404,226.87 (171 rows); latest >=100m worsens 488,430.52 (319 rows).",
         "Seen projects improve in development and confirmation; NN branch improves in development; several earlier forecast quarters improve.",
         "Many v22 project-bootstrap intervals and all latest-window v22 intervals cross zero; cannot call all deterioration statistically clear."],
       supports_rejection=[
         "Both controls have lower overall, weighted and fixed supported test-composition MAE in all three windows.",
         "New-project MAE worsens in all three windows; auxiliary all four forecast quarters worsen.",
         "Auxiliary v22 project/month overall/weighted intervals all lie in deterioration direction."],
       adaptive_after_error_audit=True,fresh_blind_validation=False,selector_changed=False,csv_generated=False,
       new_fits=0,models_loaded=0,model_predict_calls=0,test_data_loaded=False,private_score=None,
       prior_data_states_reused_not_retrained=True,stage_summary=summary,paired_intervals=intervals,
       timing_seconds=dict(main=r["elapsed_seconds"],main_wrapper=ex["elapsed_seconds"],diagnostics=de["elapsed_seconds"],
           paired_v22=pairs["v22"]["elapsed_seconds"],paired_v24=pairs["v24"]["elapsed_seconds"]),
       review_source_sha256=sha(Path(__file__)),input_sha256=inputs)
    f=lambda x:"—" if x is None else f"{x:,.2f}"
    table=lambda values:"| "+" | ".join(map(str,values))+" |"
    lines=["# R88：無局部支持時撤銷reference趨勢，三期完整組合未改善","",
       "**建議保留研究，不做final export、不新增CSV；原v22與使用者選擇不變。** 固定政策在三期整體、加權及有支持test組成估計都劣於v22與v24。這不是所有分群都退步；有利群組與不確定區間完整保留。","",
       "本輪是CPU後處理消融，0次fit、0次原生模型load/predict、0次新state、未讀test資料。R88源於已見誤差盤點（adaptive_after_error_audit），年度窗口不是盲測。","",
       "## 固定政策與原v22重播","",
       "每支分支僅當reference source＝base_fallback且spatial.applied＝False時改用自己的未修正base；其他列保留原完整分支。三支再逐筆median。沒有挑類型、真價、ID、城市、日期或比例，query價格只作評分。",
       "unit取R79 base；NN取R71 seed42 base；log取R46 base_recency_log，皆不是native單模型。9組reference、spatial及完整diagnostic重播逐位一致，三期v22原median精確重現。純政策函數通過全mask、兩類no-mask、混合mask及輸入不變檢查。","",
       table(["期別","query列","符合mask／改價列","比例"]),table(["---","---:","---:","---:"])]
    for s in STAGES:
        a=r["stages"][s];lines.append(table([LABELS[s],a["query_rows"],a["union_mask_rows"],f"{a['union_mask_rows']/a['query_rows']:.2%}"]))
    lines+=["","三支mask在每期皆相同，全部非mask的分支與最終價格逐位不變。mask中未修正base仍含原模型的時間特徵／訓練權重，不能說完全去掉所有時間資訊。","",
      "## 完整候選結果","",table(["期別","指標","v22","v24","R88","R88−v22","R88−v24"]),table(["---","---","---:","---:","---:","---:","---:"])]
    for s in STAGES:
        m=summary[s]["metrics"]
        for key,label in (("overall_mae","整體"),("weighted_mae","固定加權")):
            lines.append(table([LABELS[s],label,*[f(m[c][key]) for c in ("v22","v24","R88")],f(m["R88"][key]-m["v22"][key]),f(m["R88"][key]-m["v24"][key])]))
    lines+=["","負差代表改善，正差代表退步。主對照完整MAE分別增加約0.780%、0.555%、0.720%；這是歷史樣本上的政策效果，不能換算Private。","",
      table(["期別","test支持率／未支持列","v22 testmix","v24 testmix","R88 testmix"]),table(["---","---:","---:","---:","---:"])]
    for s in STAGES:
        m=summary[s]["test_composition"];lines.append(table([LABELS[s],f"{m['test_mass_covered']:.4%}／{m['unsupported_test_rows']}",*[f(m["models"][k]) for k in ("v22","v24","R88")]]))
    lines+=["","testmix沿原city×seen/new，歷史cell至少200列，在有支持test質量內重正規化；只讀已凍結組成統計，不讀新test特徵或真價。不是Private MAE。","",
      "## 有利與不利群組","",
      "開發與最新年度的已見建案有改善；新建案三期都退步。開發期NN分支自身改善，但另兩分支及最後median收益不成立。億元以上群在開發／較早年度改善，最新年度反而增加約48.84萬元MAE，不能只挑前兩期上台。","",
      table(["期別","群","列數","v22","R88","差額"]),table(["---","---","---:","---:","---:","---:"])]
    for s in STAGES:
        m=summary[s]["metrics"]
        for group in ("seen","new_nonmissing","missing"):
            a,b=m["v22"]["strata"][group],m["R88"]["strata"][group]
            lines.append(table([LABELS[s],group,a["rows"],f(a["mae"]),f(b["mae"]),f(b["mae"]-a["mae"]) if a["rows"] else "—"]))
    lines+=["",table(["期別","真價群","列數","v22","v24","R88","差vs v22"]),table(["---","---","---:","---:","---:","---:","---:"])]
    for s in STAGES:
        for b in summary[s]["price_bands"]:
            lines.append(table([LABELS[s],b["band"],b["rows"],*[f(b[c]) for c in ("v22","v24","R88")],f(b["R88"]-b["v22"]) if b["rows"] else "—"]))
    lines+=["","真價群只作事後诊斷，不用在未知test推論時分流或封頂。","",
       "## 分支影響及預測季度","",
       table(["期別","分支","符合mask列","原分支mask MAE","新mask MAE","整支MAE變化"]),table(["---","---","---:","---:","---:","---:"])]
    for s in STAGES:
        for b,v in summary[s]["branch_policy_diagnostics"].items():
            lines.append(table([LABELS[s],b,v["mask_rows"],f(v["masked_original_mae"]),f(v["masked_candidate_mae"]),f(v["candidate_metrics"]["overall_mae"]-v["original_metrics"]["overall_mae"])]))
    lines+=["","分支差不能相加當作最終median的因果分解。","",
       table(["期別","季度","列數","v22","v24","R88","差vs v22"]),table(["---","---","---:","---:","---:","---:","---:"])]
    for s in STAGES:
        m=summary[s]["metrics"]
        for q in ("Q1","Q2","Q3","Q4"):
            vals={c:m[c]["forecast_quarters"][q] for c in m}
            lines.append(table([LABELS[s],q,vals["v22"]["rows"],*[f(vals[c]["mae"]) for c in ("v22","v24","R88")],f(vals["R88"]["mae"]-vals["v22"]["mae"])]))
    lines+=["","## 全部24個配對區間","",
       "每一對照：建案群集及三月循環月份區塊，各overall／weighted，三期共12區間；2000次、seed20260925。差值為R88−控制。區間僅對固定保存預測及已見窗口有條件，不是p值、Private勝率，也未校正累積方法搜尋。",
       "相對v22：較早期四區間及開發期月份兩區間都落在退步方向；最新期四區間均跨0。不能把正點估計一律說成有明確統計支持。","",
       table(["控制","期別","抽法","指標","點差","2.5%","97.5%"]),table(["---","---","---","---","---:","---:","---:"])]
    for z in intervals:
        lines.append(table([z["control"],LABELS[z["stage"]],z["method"],z["metric"],f(z["point_delta"]),f(z["percentile_2_5"]),f(z["percentile_97_5"])]))
    lines+=["","## 所有城市","",
       table(["期別","城市","列數","v22","R88","差額"]),table(["---","---","---:","---:","---:","---:"])]
    for s in STAGES:
        m=summary[s]["metrics"]
        for city,a in m["v22"]["cities"].items():
            b=m["R88"]["cities"][city]
            lines.append(table([LABELS[s],city,a["rows"],f(a["mae"]),f(b["mae"]),f(b["mae"]-a["mae"])]))
    lines+=["","## 決定、耗時與保存","",
       "固定全面回退政策未獲支持，保留此失敗研究，不產生較差CSV；不據此宣稱全域時間趨勢在所有群都正確。若以後研究分群趨勢，必須另固定fit-only假設，不能從本輪query挑最佳類型或比例。",
       f"主CPU程序{r['elapsed_seconds']:.3f}秒；診斷{de['elapsed_seconds']:.3f}秒；兩組配對分別{pairs['v22']['elapsed_seconds']:.3f}及{pairs['v24']['elapsed_seconds']:.3f}秒。配對／診斷同時啟動，不能將加總當牆鐘總時間。沒有GPU模型耗時。",
       "舊CSV／selector／歷史decision的前後SHA相同。候選向量、各支mask、計畫、原state來源SHA、執行收據及全部分群保存在outputs/round88。","",
       "[固定計畫](round88_plan.md) · [driver](../src/round88_experiment.py) · [純政策自測](round88_policy_selfcheck.json) · [完整結果](../outputs/round88/results.json) · [診斷](round88_diagnostics.md) · [v22配對](round88_paired_vs_v22.md) · [v24配對](round88_paired_vs_v24.md) · [決策審查](../outputs/round88/decision_review.json)",""]
    text="\n".join(lines).replace("事後诊斷","事後診斷")
    report=ROOT/"reports/round88_report.md"
    assert not report.exists() and not (OUT/"decision_review.json").exists()
    report.write_text(text,encoding="utf8")
    record["report_sha256"]=sha(report);record["review_elapsed_seconds"]=time.perf_counter()-tick
    record["inputs_unchanged"]=all(sha(ROOT/name)==h for name,h in inputs.items());assert record["inputs_unchanged"]
    path=OUT/"decision_review.json";raw=(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    path.write_bytes(raw);path.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest()+"\n",encoding="ascii")
    print(json.dumps(dict(status="review_completed",recommendation=record["recommendation"],input_files=len(inputs),intervals=len(intervals),report_sha256=record["report_sha256"],review_seconds=record["review_elapsed_seconds"])),flush=True)
if __name__=="__main__":main()
