"""R98 stored-result audit by its helper author; scorer was written by another agent.

Never deserialize meta/query labels or raw price tables, fit an LP/native model,
compute label-based MAE, or draw bootstrap samples. Only stored evidence is checked.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import traceback
import numpy as np
from scipy.sparse import csc_matrix
import round98_contract as c

MAIN = c.ROOT / "outputs/round98"
EVAL = MAIN / "evaluation"
AUDIT = MAIN / "result_audit"
RECEIPT = MAIN / "independent_review.json"
REPORT = c.ROOT / "reports/round98_independent_review.md"
PINNED = {"src/round98_l1_stacker.py": "a1a501bc2c7874fc0d21983ad5a2be63e8d1c085f21d24a096220ff32dbfc864",
          "src/round98_temporal_evaluate.py": "6540f7045e8f47c6a0e8339a6680dd59a85839644bb0fcbbef192ea9ac7a23e1",
          c.PLAN: c.PLAN_SHA}


def close(a, b, name, tol=1e-6):
    c.need(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= tol, name)


def completion_gate():
    main = c.locked(MAIN / "execution.json")
    evaluation = c.locked(EVAL / "execution.json")
    c.need(main["exit_code"] == evaluation["exit_code"] == 0
           and main["failure"] is None and evaluation["failure"] is None, "Main/evaluation must first exit successfully")
    seal = c.locked(MAIN / "sealed_manifest.json")
    ev = c.locked(EVAL / "results.json")
    c.need(seal["status"] == "all_temporal_stack_predictions_sealed"
           and ev["status"] == "completed_three_later_window_evaluation", "Completion status differs")
    c.need(main["completed_stages"] == ev["checked_stages"] == list(c.STAGES), "All stages required")
    return seal, ev, main, evaluation


class Evidence:
    def __init__(self):
        self.hashes = {}
        self.loaded_arrays = []
    def bind(self, path, digest):
        name = c.rel(path)
        c.need(name not in self.hashes or self.hashes[name] == digest, "Conflicting authority: " + name)
        if name not in self.hashes:
            c.need(c.sha(name) == digest, "Changed recorded artifact: " + name)
            self.hashes[name] = digest
        return c.safe(name)
    def mapping(self, values):
        for name, digest in values.items():
            self.bind(name, digest)
    def document(self, record, locked=True):
        p = self.bind(record["path"], record["sha256"])
        if locked:
            value = c.locked(p)
            self.bind(p.with_suffix(".sha256"), c.sha(p.with_suffix(".sha256")))
            return value
        return c.read(p)
    def current(self, path, locked=True):
        return self.document({"path": c.rel(path), "sha256": c.sha(path)}, locked)
    def array(self, record):
        p = self.bind(record["path"], record["sha256"])
        c.need(p.suffix == ".npy" and p.name != "meta_y.npy" and "target" not in p.name.lower()
               and "labels" not in p.name.lower(), "Forbidden target-array deserialization")
        a = np.load(p, allow_pickle=False)
        c.need(str(a.dtype) == record["dtype"] and list(a.shape) == record["shape"]
               and c.values_sha(a) == record["values_sha256"], "Numeric array identity differs")
        self.loaded_arrays.append(c.rel(p))
        return a


def certificate_audit(P, state, meta, arrays):
    n = len(P)
    proof = meta["certificate"]
    tol = meta["parameters"]["certificate_tolerances"]
    c.need(meta["status"] == "completed" and proof["all_checks_passed"]
           and all(v is True for v in proof["checks"].values()), "Saved certificate did not pass")
    c.need(meta["solver_result"]["status"] == 0 and meta["solver_result"]["success"] is True
           and meta["weights_clipped_normalized_or_regularized"] is False, "Solver success/weight policy")
    w = np.asarray(state["weights"], dtype=np.float64)
    c.exact(w, -arrays["ineqlin_marginals"], "Weights are not original three solver marginals")
    c.need(w.shape == (3,) and (w >= 0).all() and abs(w.sum()-1.) <= tol["weight_sum_absolute"], "Raw simplex weights")
    x = arrays["x"]
    c.need(x.shape == (n+1,) and np.isfinite(x).all(), "Dual solution shape/finite")
    u, v = x[:-1], x[-1]
    data = np.r_[(P/1e6).reshape(-1), np.ones(3)]
    A = csc_matrix((data, np.tile(np.array([0,1,2], dtype=np.int32), n+1),
                    np.arange(n+2, dtype=np.int32)*3), shape=(3,n+1))
    slack = -np.asarray(A @ x)
    c.exact(slack, np.asarray(proof["raw_dual_constraint_slacks"]), "Saved dual constraints replay")
    bound_violation = max(float(np.max(-1-u)), float(np.max(u-1)), 0.)
    close(bound_violation, proof["dual_box_violation"], "Dual box record", 1e-12)
    c.need(bound_violation <= tol["dual_box_absolute"], "Dual solution outside box")
    dual_violation = max(float(-slack.min()), 0.)*1e6/n
    close(dual_violation, proof["dual_constraint_violation_mean_TWD"], "Dual constraint record")
    c.need(dual_violation <= tol["dual_constraint_violation_mean_TWD"], "Dual constraint tolerance")
    row_comp = float(np.abs(arrays["ineqlin_marginals"]*slack).sum())*1e6/n
    bound_comp = float(np.abs(arrays["lower_marginals"][:-1]*(u+1)).sum()
                       +np.abs(arrays["upper_marginals"][:-1]*(1-u)).sum())*1e6/n
    close(row_comp, proof["row_complementarity_mean_TWD"], "Row complementarity replay")
    close(bound_comp, proof["bound_complementarity_mean_TWD"], "Bound complementarity replay")
    c.need(row_comp <= tol["complementarity_mean_TWD"] and bound_comp <= tol["complementarity_mean_TWD"], "Complementarity tolerance")
    # Summary arithmetic only: no y reconstruction from multipliers or residuals.
    gap = proof["primal_MAE_TWD"]-proof["dual_objective_MAE_TWD"]
    close(gap, proof["primal_minus_dual_MAE_TWD"], "Stored primal/dual/gap summary")
    close(gap, state["certificate_gap_MAE_TWD"], "State gap summary")
    close(state["primal_MAE_TWD"], proof["primal_MAE_TWD"], "State objective summary")
    c.need(abs(gap) <= tol["primal_dual_gap_mean_TWD"], "Stored optimality gap tolerance")
    return {"weights": w.tolist(), "meta_rows": n, "raw_solver_marginals_exact": True,
            "dual_constraints_replayed_exact": True, "dual_box_verified": True,
            "complementarity_replayed": True, "stored_primal_dual_gap_summary_verified": True,
            "gap_MAE_TWD": gap, "primal_meta_MAE_TWD": proof["primal_MAE_TWD"],
            "meta_labels_loaded_or_reconstructed": False, "primal_loss_or_y_stationarity_recomputed": False,
            "solver_seconds": meta["solver_seconds"], "build_seconds": meta["build_seconds"],
            "helper_elapsed_seconds": meta["elapsed_seconds"], "warnings": meta["warnings"]}


def metric_audit(stage, item, registry):
    metrics = item["metrics"]
    c.need(set(metrics) == {"N","H","U","v22","R98"}, "All five fixed vectors required")
    for name, m in metrics.items():
        n = m["rows"]
        c.need(n == item["query_rows"], "Metric support count")
        for group in ("strata", "forecast_quarters", "cities", "months", "relative_forecast_months"):
            parts = m[group].values()
            c.need(sum(p["rows"] for p in parts) == n, "Incomplete metric partition: "+group)
            average = sum(p["rows"]*p["mae"] for p in parts if p["rows"])/n
            close(average, m["overall_mae"], "Aggregate partition arithmetic: "+name+"/"+group)
        bands = m["price_bands"]
        c.need(sum(b["rows"] for b in bands) == n, "Price-band partition")
        close(sum(b["rows"]*b["mae"] for b in bands if b["rows"])/n, m["overall_mae"], "Price-band arithmetic")
        close(sum(c.WEIGHTS[g]*m["strata"][g]["mae"] for g in c.WEIGHTS if c.WEIGHTS[g]>0), m["weighted_mae"], "Fixed weighted arithmetic")
        close(m["tails"]["absolute_error_sum"]/n, m["overall_mae"], "Tail error total")
    control, candidate = metrics["v22"], metrics["R98"]
    comparison = item["comparison"]
    adverse = []
    for key in ("overall_mae", "weighted_mae"):
        close(candidate[key]-control[key], comparison[key]["delta"], "Complete model difference")
    for section in ("strata", "forecast_quarters", "months", "cities", "relative_forecast_months"):
        for key, old in control[section].items():
            new = candidate[section][key]
            c.need(old["rows"] == new["rows"], "Pair group support differs")
            if old["rows"]:
                close(new["mae"]-old["mae"], comparison[section][key]["delta"], "Group delta")
                if (section=="forecast_quarters" or old["rows"]>=500) and new["mae"]>1.05*old["mae"]:
                    adverse.append((section,key))
    for old,new in zip(control["price_bands"],candidate["price_bands"]):
        c.need(old["band"]==new["band"] and old["rows"]==new["rows"], "Pair price support")
        if old["rows"]>=500 and new["mae"]>1.05*old["mae"]:
            adverse.append(("price_bands",old["band"]))
    declared=[(x["section"],x.get("group",x.get("band"))) for x in comparison["quarter_or_large_group_deterioration_above_5pct"]]
    c.need(declared==adverse, "Fixed adverse-group list incomplete")
    improve=all(candidate[k]<control[k] for k in ("overall_mae","weighted_mae"))
    c.need(comparison["both_point_metrics_improve"]==improve and comparison["stage_research_gate_passed"]==(improve and not adverse), "Fixed stage gate")
    intervals=[]
    paired=item["paired"]
    c.need(paired["stage_index"]==c.STAGES.index(stage) and paired["seed"]==20260925 and paired["replicates"]==2000, "Fixed bootstrap parameters")
    for fi,family in enumerate(("project","month")):
        b=registry.array(item["replicate_artifacts"][family])
        c.need(b.dtype==np.float64 and b.shape==(2000,4), "Saved replicates shape")
        c.need(paired["methods"][family]["family_index"]==fi, "Family index")
        for metric,left in (("overall",0),("weighted",2)):
            s=paired["methods"][family]["summary"][metric]
            delta=b[:,left+1]-b[:,left]
            valid=delta[np.isfinite(delta)]
            bounds=np.quantile(valid,[.025,.975],method="linear").tolist() if len(valid) else [None,None]
            c.need(bounds==[s["percentile_2_5"],s["percentile_97_5"]], "CI endpoints differ")
            c.need(s["valid_replicates"]==len(valid) and s["invalid_replicates"]==2000-len(valid), "Replicate validity counts")
            c.need(s["negative_delta_fraction"]==(float(np.mean(valid<0)) if len(valid) else None)
                   and s["bootstrap_mean_delta"]==(float(valid.mean()) if len(valid) else None), "Stored bootstrap summary differs")
            close(s["control_mae"],control[metric+"_mae"],"Paired control point")
            close(s["candidate_mae"],candidate[metric+"_mae"],"Paired candidate point")
            close(s["point_delta"],comparison[metric+"_mae"]["delta"],"Paired delta")
            intervals.append({"stage":stage,"family":family,"metric":metric,"point_delta":s["point_delta"],
                              "lower":bounds[0],"upper":bounds[1],"valid":len(valid),"invalid":2000-len(valid)})
    return {"vectors_verified":5,"all_aggregate_partitions_verified":True,"stage_gate_passed":improve and not adverse,
            "adverse_groups":declared,"intervals":intervals,"new_MAE_or_bootstrap_computed":False}


def render(review, evaluation):
    lines=["# R98 保存結果稽核", "", "結論：三個較晚窗口未全部通過事前研究門檻，沒有足夠證據將這組全域 L1 組合升級為提交候選。v22 仍是唯一選定提交，Private 未知。", "",
           "**角色揭露：本稽核作者同時編寫 R98 的純數值 helper，因此這是保存結果稽核，不聲稱對自己開發的求解器完全獨立。評分程式由另一代理獨立編寫，前置另有代理審查。**", "",
           "所有三期 main 與 scorer 真正 exit 0 後才執行本稽核。沒有載入或重建 meta/query y、沒有重新求 LP、模型推論、逐列 MAE或新 bootstrap；raw/meta-y 檔案只讀位元組核 SHA。讀 P 是讀已保存的三分支預測，並非真價。", "",
           f"實核 {review['unique_hashed_files']} 個唯一來源／產物 SHA，包含凍結 helper／scorer／計畫、solver 原碼與 binary、各階段來源、輸入及 9 個保護檔。", "",
           "## 求解與保存", "", "|窗口|meta列數|N / H / U權重|LP秒數|helper呼叫秒數|primal−dual（平均TWD）|", "|---|---:|---|---:|---:|---:|"]
    for s in c.STAGES:
        a=review["stages"][s]
        lines.append(f"|{s}|{a['certificate']['meta_rows']:,}|{' / '.join(f'{w:.10f}' for w in a['certificate']['weights'])}|{a['certificate']['solver_seconds']:.3f}|{a['fit_call_seconds']:.3f}|{a['certificate']['gap_MAE_TWD']:+.9f}|")
    lines += ["", f"driver三次fit呼叫合計 {review['timings']['fit_calls_total_seconds']:.3f} 秒；main內層 {review['timings']['main_seconds']:.3f} 秒、評估execution {review['timings']['evaluation_seconds']:.3f} 秒。fit呼叫已包含於main，不能重複相加。", "",
              "SciPy 1.18.1／HiGHS 1.12.0；固定單執行緒選項、300秒上限、無截距與每列等權L1。三個權重逐位等於負的solver inequality marginals，未clip／normalize。用保存P及u重算三個dual constraints與互補性，保存primal／dual／gap摘要一致；未讀y，所以沒有重算primal loss或含y的stationarity，這個範圍限制明列。", "",
              "三個meta P/y身分、canonical/raw mapping、日期、合法forward block與完整query公式均核對。query公式是固定分項相加，三期與保存向量逐位一致；v22原median亦逐位重現。原v22保存MAE按R79與R97權威結果核對，未重讀真價重算。", "",
              "## 完整模型與所有五向量", "", "差值為R98−v22，負數較好。", "", "|窗口|v22 MAE|R98 MAE|整體差|固定加權差|研究門檻|", "|---|---:|---:|---:|---:|---|"]
    for s in c.STAGES:
        a=evaluation["stages"][s];m=a["metrics"];d=a["comparison"]
        lines.append(f"|{s}|{m['v22']['overall_mae']:,.2f}|{m['R98']['overall_mae']:,.2f}|{d['overall_mae']['delta']:+,.2f}|{d['weighted_mae']['delta']:+,.2f}|{d['stage_research_gate_passed']}|")
    lines += ["", "|窗口|N MAE|H MAE|U MAE|v22 median MAE|R98 MAE|", "|---|---:|---:|---:|---:|---:|"]
    for s in c.STAGES:
        m=evaluation["stages"][s]["metrics"]
        lines.append("|"+s+"|"+"|".join(f"{m[k]['overall_mae']:,.2f}" for k in ("N","H","U","v22","R98"))+"|")
    lines += ["", "## 必須保留的有利反證", "",
              "offset 的整體與固定加權點估計確實改善；新案約下降12,915元，>=1億的270列下降約238萬元。confirmation 的2,000–5,000萬元帶也改善約20,060元。不能把本輪說成所有窗口、所有群都退步。", "",
              "不利證據是development與confirmation整體／加權都退步，且每期至少有一個事前定義的大群退步超過5%；offset因此也未通過研究門檻。尾部少量高價交易改善不能取代全體與分群門檻。", ""]
    for s in c.STAGES:
        a=evaluation["stages"][s]
        lines += ["### "+s, "", "|群／季度／真價帶|列數|R98−v22 MAE|", "|---|---:|---:|"]
        for section in ("strata","forecast_quarters"):
            for k,v in a["comparison"][section].items():
                lines.append(f"|{section}/{k}|{v['rows']:,}|{v['delta']:+,.2f}|")
        for v in a["comparison"]["price_bands"]:
            lines.append(f"|{v['band']}|{v['rows']:,}|"+(f"{v['delta']:+,.2f}" if v['rows'] else "—")+"|")
        lines += ["", "|超過5%的季度或>=500列群|列數|MAE差|相對變化|", "|---|---:|---:|---:|"]
        for v in a["comparison"]["quarter_or_large_group_deterioration_above_5pct"]:
            lines.append(f"|{v['section']}/{v.get('group',v.get('band'))}|{v['rows']:,}|{v['delta']:+,.2f}|{v['relative_change_pct']:+.3f}%|")
    lines += ["", "## 十二個保存配對區間", "", "|窗口|抽樣|指標|點差|2.5%|97.5%|有效／無效|", "|---|---|---|---:|---:|---:|---:|"]
    for a in review["intervals"]:
        lines.append(f"|{a['stage']}|{a['family']}|{a['metric']}|{a['point_delta']:+,.2f}|{a['lower']:+,.2f}|{a['upper']:+,.2f}|{a['valid']}/{a['invalid']}|")
    lines += ["", "12個區間全部跨零，由六份已保存(2000,4)replicates重新計算端點，未重新抽樣。stage index依序0／1／2、family0／1；正權重層缺席時保留無效，未重抽。它們不是Private勝率或已修正選型的顯著性檢定。", "",
              "## 學術與重現界線", "",
              "這是對早期forward預測學權重的時間前推檢查，不是新盲測；offset和其他窗口重疊，歷史資料也已反覆用於選型。較早query標籤成為後來stage meta-label是允許的角色轉換，但同stage query沒有進入該stage meta-fit。缺少auxiliary以前的forward block，本輪不能冒稱完成原三個8月窗口的採用檢查。", "",
              "此次0 native fit／0 native model load，只新增3個CPU線性組合；沒有全量test fit、新CSV、Kaggle操作或替換提交。保存模型／求解器證據可追溯，不等於全新環境端到端重訓。真價帶和誤差尾部只作診斷，不能用來依未知真价分流。", "",
              "[逐項稽核JSON](../outputs/round98/independent_review.json) · [完整評估](../outputs/round98/evaluation/results.json) · [固定計畫](round98_plan.md) · [求解器設計](round98_helper_design.md) · [主流程封存](../outputs/round98/sealed_manifest.json)", ""]
    return "\n".join(lines)


def run():
    seal, ev, main_execution, eval_execution = completion_gate()
    c.need(not AUDIT.exists() and not RECEIPT.exists() and not REPORT.exists(), "Preserve previous audit attempts")
    AUDIT.mkdir()
    started,tick,code,failure=c.now(),time.monotonic(),1,None
    try:
        reg=Evidence();reg.mapping(PINNED)
        for path in (MAIN/"sealed_manifest.json",MAIN/"execution.json",EVAL/"results.json",EVAL/"execution.json",EVAL/"prelabel_gate.json"):
            reg.current(path)
        launch=reg.document(seal["launch_request"])
        er=reg.document(ev["evaluation_request"])
        prep=reg.document(seal["preparation"])
        inv=reg.document(prep["inventory"],False)
        c.need(launch["helper_parameters"]==er["model_parameters"] and er["parameters"]==ev["parameters"]==c.evaluation_parameters(),"Frozen parameter authorities")
        c.need(seal["source_sha256"]==launch["source_sha256"],"Main source freeze")
        for mapping in (seal["source_sha256"],ev["source_sha256"],ev["input_sha256"],ev["artifact_sha256"],prep["input_sha256"],prep["source_sha256"]):reg.mapping(mapping)
        c.need(seal["protected_before"]==seal["protected_after"]==launch["protected_before"]==c.protected_hashes() and len(seal["protected_before"])==9,"Nine protected artifacts")
        reg.mapping(seal["protected_before"])
        gate=reg.current(EVAL/"prelabel_gate.json")
        c.need(gate["checked_stages"]==list(c.STAGES) and gate["status"]=="all_stages_verified_before_price_deserialization","Prelabel completion gate")
        mapping_rec=inv["canonical_mapping"]
        mapping=reg.array({"path":mapping_rec["path"],"sha256":mapping_rec["file_sha256"],**{k:mapping_rec[k] for k in ("dtype","shape","values_sha256")}})
        histories={"annual":reg.current("outputs/round79/results.json"),"offset":reg.current("outputs/round97/evaluation/results.json")}
        stages={};intervals=[]
        for stage in c.STAGES:
            b=seal["stages"][stage];st=reg.document(b["stage_manifest"]);ex=reg.document(b["execution"])
            c.need(ex["exit_code"]==0 and st["all_query_reload_exact"] and st["query_labels_available"] is False,"Stage completion")
            c.need(st["launch_request"]==seal["launch_request"] and st["source_sha256"]==launch["driver_source_sha256"],"Stage/source authority")
            reg.mapping(st["source_sha256"]);reg.mapping(st["model_artifact_sha256"])
            records=prep["stages"][stage]["arrays"]
            c.need(st["input_arrays"]==records and st["actual_stage_data_readset"]=={r["path"]:r["sha256"] for r in records.values()},"Stage actual readset")
            reg.bind(records["meta_y"]["path"],records["meta_y"]["sha256"])
            arr={k:reg.array(r) for k,r in records.items() if k!="meta_y"}
            ids,qids=arr["meta_indices"],arr["query_indices"]
            c.need(len(np.unique(ids))==len(ids) and not np.intersect1d(ids,qids).size and np.isin(ids,arr["fit_indices"]).all(),"Meta/query time separation")
            start,end,n=c.SPLITS[stage]
            day=lambda d:(np.datetime64(d)-np.datetime64("2000-01-01")).astype(int)
            c.need(np.isfinite(arr["meta_trade_days"]).all() and (arr["meta_trade_days"]<day(start)).all()
                   and ((arr["query_trade_days"]>=day(start))&(arr["query_trade_days"]<=day(end))).all(),"Saved date boundary")
            for role in ("meta","query"):c.exact(mapping[arr[role+"_indices"]],arr[role+"_source_cache_indices"],"Raw source mapping")
            c.need(c.values_sha(arr["meta_predictions"])==inv["meta_fit"][stage]["prediction_matrix_values_sha256"],"Earlier forward matrix identity")
            for j,k in enumerate(c.BRANCHES):c.need(c.values_sha(arr["query_predictions"][:,j])==inv["blocks"][stage]["branches"][k]["values_sha256"],"Original branch identity")
            state=reg.document(b["model_state"],False)
            folder=c.safe(b["model_state"]["path"]).parent
            meta=reg.current(folder/"fit_metadata.json",False);reg.mapping(state["source_sha256"])
            c.need(state["parameters"]==meta["parameters"]==launch["helper_parameters"] and state["input_arrays"]==meta["input_arrays"],"Model parameter/input identity")
            for k,rk in (("P","meta_predictions"),("y","meta_y")):
                c.need(state["input_arrays"][k]=={f:records[rk][f] for f in ("dtype","shape","values_sha256")},"P/y stored identity")
            c.need(state["input_arrays"]["row_ids_values_sha256"]==c.values_sha(ids),"Model row identity")
            with np.load(folder/"solver_arrays.npz",allow_pickle=False) as z:
                c.need(set(z.files)==set(meta["solver_arrays"]),"Solver evidence schema")
                arrays={k:z[k] for k in z.files}
            for k,a in arrays.items():c.need({"dtype":str(a.dtype),"shape":list(a.shape),"values_sha256":c.values_sha(a)}==meta["solver_arrays"][k],"Solver evidence numerical identity")
            cert=certificate_audit(arr["meta_predictions"],state,meta,arrays)
            control=reg.array(b["predictions"]["v22"]);candidate=reg.array(b["predictions"]["R98"])
            P=arr["query_predictions"];w=np.asarray(state["weights"])
            c.exact(np.median(P,axis=1),control,"Original full median replay")
            c.exact((P[:,0]*w[0]+P[:,1]*w[1])+P[:,2]*w[2],candidate,"Saved query weight formula replay")
            c.exact(control,arr["control_v22"],"Saved original control")
            item=reg.current(EVAL/stage/"results.json");c.need(item==ev["stages"][stage],"Stage/full result identity")
            old=histories["offset"]["metrics"]["v22"] if stage=="offset" else histories["annual"]["stages"][stage]["metrics"]["R79"]
            for k in ("overall_mae","weighted_mae"):close(item["metrics"]["v22"][k],old[k],"Original v22 saved score replay")
            ma=metric_audit(stage,item,reg);intervals+=ma["intervals"]
            stages[stage]={"certificate":cert,"metrics_audit":ma,"full_query_formula_exact":True,
                           "control_median_exact":True,"original_v22_saved_metrics_replayed":True,
                           "fit_call_seconds":st["fit_call_elapsed_seconds"]}
        expected_gate=all(stages[s]["metrics_audit"]["stage_gate_passed"] for s in c.STAGES)
        c.need(ev["research_gate"]["all_three_later_windows_passed"]==expected_gate
               and ev["research_gate"]["automatic_adoption"] is False,"Complete research gate")
        c.need(len(intervals)==12,"All intervals required")
        # Check all input identities again, without deserializing forbidden arrays.
        reg.mapping(reg.hashes.copy())
        c.verify_hashes(reg.hashes)
        review={"round":98,"status":"passed_stored_result_audit","author_role":"Helper author performed this result audit; scorer developed independently by another agent",
                "unique_hashed_files":len(reg.hashes),"all_verified_sha256":reg.hashes,"loaded_numeric_arrays":reg.loaded_arrays,
                "protected_files_verified":9,"stages":stages,"intervals":intervals,
                "all_intervals_cross_zero":all(x["lower"]<=0<=x["upper"] for x in intervals),
                "research_gate_passed":expected_gate,"selected_submission":"v22","Private_score":None,
                "meta_or_query_labels_deserialized_or_reconstructed":False,"label_based_MAE_recomputed":False,
                "new_bootstrap_draws":0,"LP_solves":0,"native_model_load_fit_predict_calls":0,"GPU_calls":0,
                "timings":{"fit_calls_total_seconds":sum(a["fit_call_seconds"] for a in stages.values()),
                           "main_seconds":main_execution["elapsed_seconds"],"evaluation_seconds":eval_execution["elapsed_seconds"]},
                "source_sha256":c.sha(__file__),"completed_at_utc":c.now(),"elapsed_seconds_before_write":time.monotonic()-tick}
        c.write(RECEIPT,review)
        text=render(review,ev)
        with REPORT.open("x",encoding="utf8",newline="\n") as f:f.write(text)
        for link in re.findall(r"\]\(([^)]+)\)",text):c.need((REPORT.parent/link).resolve().is_file(),"Broken report link")
        code=0
        print(json.dumps({"status":review["status"],"unique_hashes":len(reg.hashes),"intervals":len(intervals),
                          "report_sha256":c.sha(REPORT),"receipt_sha256":c.sha(RECEIPT)}))
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        c.write(AUDIT/"execution.json",{"exit_code":code,"failure":failure,"started_at_utc":started,"completed_at_utc":c.now(),
                "elapsed_seconds":time.monotonic()-tick,"source_sha256":c.sha(__file__),"query_or_meta_y_loaded":False,
                "new_fit_score_or_bootstrap_calls":0,"automatic_retry":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",required=True)
    parser.parse_args()
    run()
