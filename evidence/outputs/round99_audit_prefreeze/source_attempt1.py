"""R99 stored-result audit by its helper author; scorer and driver were written by other agents.

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
import round99_contract as c

MAIN = c.ROOT / "outputs/round99"
EVAL = MAIN / "evaluation"
AUDIT = MAIN / "result_audit"
RECEIPT = MAIN / "independent_review.json"
REPORT = c.ROOT / "reports/round99_independent_review.md"
PINNED = {"src/round99_l1_stacker.py": "f07d818aa9b12cf03726267188e99d111ce2529dd6d6cd5f9d921fee259fc09e",
          "src/round99_temporal_evaluate.py": "1a6d94ba0730124933754137179430256ced82af52c108130a57f9a06295a1f2",
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
    c.exact(w, -arrays["ineqlin_marginals"], "Weights are not original four solver marginals")
    c.need(w.shape == (4,) and (w >= 0).all() and abs(w.sum()-1.) <= tol["weight_sum_absolute"], "Raw simplex weights")
    x = arrays["x"]
    c.need(x.shape == (n+1,) and np.isfinite(x).all(), "Dual solution shape/finite")
    u, v = x[:-1], x[-1]
    data = np.r_[(P/1e6).reshape(-1), np.ones(4)]
    A = csc_matrix((data, np.tile(np.array([0,1,2,3], dtype=np.int32), n+1),
                    np.arange(n+2, dtype=np.int32)*4), shape=(4,n+1))
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
    c.need(state["format"] == "R99_convex_L1_stacker_v1" and state["branch_order"] == list(c.STACKER_BRANCHES)
           and state["query_dependent_state"] is False, "Four-input model state identity")
    vertices = proof["native_vertex_MAE_TWD"]
    c.need(set(vertices) == set(c.STACKER_BRANCHES), "All four vertex summaries required")
    close(proof["median_vertex_MAE_TWD"], vertices["M"], "Median vertex summary identity")
    close(proof["primal_minus_median_vertex_MAE_TWD"], proof["primal_MAE_TWD"]-vertices["M"], "Median objective delta arithmetic")
    close(proof["primal_minus_best_vertex_MAE_TWD"], proof["primal_MAE_TWD"]-min(vertices.values()), "Best vertex delta arithmetic")
    c.need(proof["primal_minus_median_vertex_MAE_TWD"] <= tol["vertex_optimality_mean_TWD"]
           and proof["primal_minus_best_vertex_MAE_TWD"] <= tol["vertex_optimality_mean_TWD"], "Saved vertex feasibility check")
    for key, limit in (("stationarity_price_max_TWD", "stationarity_price_max_TWD"),
                       ("stationarity_free_variable_absolute", "stationarity_free_variable_absolute"),
                       ("objective_identity_mean_TWD", "objective_identity_mean_TWD")):
        c.need(np.isfinite(proof[key]) and 0 <= proof[key] <= tol[limit], "Saved certificate range: "+key)
    return {"weights": w.tolist(), "meta_rows": n,
            "four_vertex_summary_verified": True, "median_meta_MAE_TWD": vertices["M"],
            "primal_minus_median_meta_MAE_TWD": proof["primal_minus_median_vertex_MAE_TWD"], "raw_solver_marginals_exact": True,
            "dual_constraints_replayed_exact": True, "dual_box_verified": True,
            "complementarity_replayed": True, "stored_primal_dual_gap_summary_verified": True,
            "gap_MAE_TWD": gap, "primal_meta_MAE_TWD": proof["primal_MAE_TWD"],
            "meta_labels_loaded_or_reconstructed": False, "primal_loss_or_y_stationarity_recomputed": False,
            "solver_seconds": meta["solver_seconds"], "build_seconds": meta["build_seconds"],
            "helper_elapsed_seconds": meta["elapsed_seconds"], "warnings": meta["warnings"]}


def metric_audit(stage, item, registry):
    metrics = item["metrics"]
    c.need(set(metrics) == {"N","H","U","v22","R98","R99"}, "All six fixed vectors required")
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
    candidate = metrics["R99"]
    adverse_by_control = {}
    for control_name, comparison_name in (("v22", "comparison"), ("R98", "comparison_vs_R98")):
        control, comparison = metrics[control_name], item[comparison_name]
        adverse = []
        for key in ("overall_mae", "weighted_mae"):
            close(candidate[key]-control[key], comparison[key]["delta"], "Complete model difference")
        for section in ("strata", "forecast_quarters", "months", "cities", "relative_forecast_months"):
            for key, previous in control[section].items():
                new = candidate[section][key]
                c.need(previous["rows"] == new["rows"], "Pair group support differs")
                if previous["rows"]:
                    close(new["mae"]-previous["mae"], comparison[section][key]["delta"], "Group delta")
                    if (section=="forecast_quarters" or previous["rows"]>=500) and new["mae"]>1.05*previous["mae"]:
                        adverse.append((section,key))
        for previous,new in zip(control["price_bands"],candidate["price_bands"]):
            c.need(previous["band"]==new["band"] and previous["rows"]==new["rows"], "Pair price support")
            saved = next(x for x in comparison["price_bands"] if x["band"] == previous["band"])
            if previous["rows"]:
                close(new["mae"]-previous["mae"], saved["delta"], "Price-band delta")
            if previous["rows"]>=500 and new["mae"]>1.05*previous["mae"]:
                adverse.append(("price_bands",previous["band"]))
        declared=[(x["section"],x.get("group",x.get("band"))) for x in comparison["quarter_or_large_group_deterioration_above_5pct"]]
        c.need(declared==adverse, "Fixed adverse-group list incomplete")
        improve=all(candidate[k]<control[k] for k in ("overall_mae","weighted_mae"))
        c.need(comparison["both_point_metrics_improve"]==improve, "Saved point improvement flag")
        if control_name == "v22":
            c.need(comparison["stage_research_gate_passed"]==(improve and not adverse), "Fixed stage gate")
        else:
            c.need(comparison["used_for_research_gate"] is False and "stage_research_gate_passed" not in comparison, "R98 must not add a gate")
        adverse_by_control[control_name] = declared
    control, comparison = metrics["v22"], item["comparison"]
    improve=all(candidate[k]<control[k] for k in ("overall_mae","weighted_mae"))
    adverse=adverse_by_control["v22"]
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
    return {"vectors_verified":6,"all_aggregate_partitions_verified":True,"stage_gate_passed":improve and not adverse,
            "adverse_groups":adverse,"adverse_groups_vs_R98":adverse_by_control["R98"],"intervals":intervals,"new_MAE_or_bootstrap_computed":False}


def render(review, evaluation):
    passed = review["research_gate_passed"]
    verdict = ("三個較晚窗口通過固定探索門檻，但仍未完成原三年度採用證據；只支持另立更早 forward block 研究。"
               if passed else "三個較晚窗口沒有全部通過固定探索門檻，不能升級為提交候選。")
    lines=["# R99 保存結果稽核", "", "結論："+verdict+" v22 仍是唯一選定提交，Private 未知。", "",
           "**角色揭露：本作者同時編寫 R99 純數值 helper；此為保存結果稽核，不聲稱完全獨立驗證自己的求解器。driver 與 scorer 各由其他代理編寫；本作者另作了執行前交叉來源審查。**", "",
           "main 與評分 execution 均真正 exit 0 後才執行。沒有反序列化或重建 meta/query y，沒有 LP、native model load/fit/predict、逐筆 MAE 或新 bootstrap。只有 SHA 讀取價格檔案位元組；P 是保存的分支預測，不是真價。", "",
           f"核對 {review['frozen_source_count']} 個 freeze 身分與共 {review['unique_hashed_files']} 個唯一來源／产物 SHA，含 9 個保護檔、原 R98 preparation 身分與新 R99 plan 的區別。", "",
           "## 四欄模型與可行基線", "", "|窗口|N / H / U / M 權重|meta 列數|meta R99−M|meta R99−R98|fit 呼叫秒數|", "|---|---|---:|---:|---:|---:|"]
    for stage in c.STAGES:
        a=review["stages"][stage]; proof=a["certificate"]
        lines.append(f"|{stage}|{' / '.join(f'{w:.10f}' for w in proof['weights'])}|{proof['meta_rows']:,}|{proof['primal_minus_median_meta_MAE_TWD']:+,.6f}|{a['meta_objective_delta_vs_R98']:+,.6f}|{a['fit_call_seconds']:.3f}|")
    lines += ["", "四權重逐位等於負的四個 solver inequality marginals，未 clip/normalize。原前三欄來源 SHA、derived M=median 的 bytes、四項固定 float64 順序與模型 JSON 重新讀取後的全 query 保存向量全部一致。四個 dual constraints、u box 與互補性可用保存 P/u 重算。", "",
              "meta 的 primal/dual/gap、四 vertex、M 及舊 R98 objective 僅核保存摘要、身分與固定容差；未讀 y，因此未獨立重算 primal loss、含 y 的 stationarity 或 label-based optimality。primal−dual 容差 0.10 元；M／舊 R98 可行基線容差 0.10 元，都是 meta 工程證書，不是 query 或固定加權 MAE 保證。", "",
              f"三次 fit 呼叫合計 {review['timings']['fit_calls_total_seconds']:.3f} 秒，已包含在 main {review['timings']['main_seconds']:.3f} 秒內；評分 execution {review['timings']['evaluation_seconds']:.3f} 秒。它們不是整段研究聊天工時，亦不能把 fit 時間再次加進 main。", "",
              "## 完整模型與兩個固定對照", "", "負差較好。R98 僅點差，不新增一組 bootstrap 或門檻。", "",
              "|窗口|v22 MAE|R98 MAE|R99 MAE|R99−v22|固定加權差|R99−R98|研究門檻|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for stage in c.STAGES:
        a=evaluation["stages"][stage];m=a["metrics"];d=a["comparison"];d98=a["comparison_vs_R98"]
        lines.append(f"|{stage}|{m['v22']['overall_mae']:,.2f}|{m['R98']['overall_mae']:,.2f}|{m['R99']['overall_mae']:,.2f}|{d['overall_mae']['delta']:+,.2f}|{d['weighted_mae']['delta']:+,.2f}|{d98['overall_mae']['delta']:+,.2f}|{d['stage_research_gate_passed']}|")
    lines += ["", "|窗口|N|H|U|v22 median|R98|R99|", "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in c.STAGES:
        m=evaluation["stages"][stage]["metrics"]
        lines.append("|"+stage+"|"+"|".join(f"{m[k]['overall_mae']:,.2f}" for k in ("N","H","U","v22","R98","R99"))+"|")
    off=evaluation["stages"]["offset"]["comparison"]
    conf=evaluation["stages"]["confirmation"]["comparison"]
    ob={v["band"]:v for v in off["price_bands"]}
    cb={v["band"]:v for v in conf["price_bands"]}
    lines += ["", "**有利反證與代價：** "
              +f"offset 整體下降 {-off['overall_mae']['delta']:,.2f} 元、新案下降 {-off['strata']['new_nonmissing']['delta']:,.2f} 元；億元以上 {ob['>=100m']['rows']} 列下降 {-ob['>=100m']['delta']:,.2f} 元。"
              +f"confirmation 的 1,000–2,000 萬與 2,000–5,000 萬亦分別下降 {-cb['10-20m']['delta']:,.2f}／{-cb['20-50m']['delta']:,.2f} 元，不能稱全部都退步。"
              +f"但 confirmation 億元以上 {cb['>=100m']['rows']} 列增加 {cb['>=100m']['delta']:,.2f} 元，且 development/confirmation 整體仍不及原 median。兩個窗口較 R98 改善也不能替代相對 v22 的門檻。"]
    lines += ["", "六向量各自的群組／季度／城市／月份／五價帶筆數與誤差總和、固定 seen/new 權重和尾部誤差總額均從保存聚合值核算，沒有重算逐筆 MAE。v22 和 R98 的 overall/weighted 同窗舊控制亦核對至 1e-6 元。", "",
              "## 完整正反群組", "", "以下列完整 seen/new/missing、四季度與五價帶，不能只用整體結果概括所有群。真價群只作事後診斷，不能用作 test 分流。"]
    for stage in c.STAGES:
        a=evaluation["stages"][stage]
        lines += ["", "### "+stage, "", "|群／季度／價帶|列數|R99−v22|R99−R98|", "|---|---:|---:|---:|"]
        for section in ("strata","forecast_quarters"):
            for name,v in a["comparison"][section].items():
                d98=a["comparison_vs_R98"][section][name]["delta"]
                lines.append(f"|{section}/{name}|{v['rows']:,}|{v['delta']:+,.2f}|{d98:+,.2f}|")
            if section == "strata" and "missing" not in a["comparison"][section]:
                c.need(a["metrics"]["v22"]["strata"]["missing"]["rows"] == 0, "Missing stratum unexpectedly omitted")
                lines.append("|strata/missing|0|—|—|")
        for v, v98 in zip(a["comparison"]["price_bands"],a["comparison_vs_R98"]["price_bands"]):
            lines.append(f"|{v['band']}|{v['rows']:,}|"+(f"{v['delta']:+,.2f}|{v98['delta']:+,.2f}|" if v['rows'] else "—|—|"))
        lines += ["", "|相對 v22 超過 5% 的季度或至少 500 列群|列數|MAE 差|相對變化|", "|---|---:|---:|---:|"]
        adverse=a["comparison"]["quarter_or_large_group_deterioration_above_5pct"]
        if not adverse:lines.append("|無|—|—|—|")
        for v in adverse:
            lines.append(f"|{v['section']}/{v.get('group',v.get('band'))}|{v['rows']:,}|{v['delta']:+,.2f}|{v['relative_change_pct']:+.3f}%|")
    lines += ["", "## 十二個保存配對區間", "", "|窗口|抽樣|指標|點差|2.5%|97.5%|有效／無效|", "|---|---|---|---:|---:|---:|---:|"]
    for a in review["intervals"]:
        lines.append(f"|{a['stage']}|{a['family']}|{a['metric']}|{a['point_delta']:+,.2f}|{a['lower']:+,.2f}|{a['upper']:+,.2f}|{a['valid']}/{a['invalid']}|")
    crossed=sum(x['lower']<=0<=x['upper'] for x in review['intervals'])
    lines += ["", f"12 個區間中 {crossed} 個跨零；由六份已保存 (2000,4) replicates 核端點、平均差、負差比例與有效／無效筆數，未重新抽樣。seed20260925，stage0/1/2、family project0/month1。缺正權重層的樣本保留無效，沒有補抽。CI 條件於固定模型，不是 Private 勝率，也沒有消除多輪選型偏差。", "",
              "## 界線", "", "本輪是在觀看 R98 結果後的適應性候選；歷史窗口多次選型，offset 與年度窗口重疊，不是三個獨立盲測。較早 query 可成為較晚 meta label，但同 stage query 不能進本 stage fit。缺 auxiliary 以前 forward block，即使探索門檻通過也不是完整原三年度採用驗證。", "",
              "0 native 重訓、0 GPU，只新增 3 個 CPU meta-fit；本稽核本身 0 fit/score。沒有全量 fit、新 CSV、Kaggle 操作或改 v22 選定。可追溯保存與重載一致仍不等於乾淨環境從零重訓全鏈。", "",
              "[逐項稽核 JSON](../outputs/round99/independent_review.json) · [完整評估](../outputs/round99/evaluation/results.json) · [固定計畫](round99_plan.md) · [helper 設計](round99_helper_design.md) · [主流程封存](../outputs/round99/sealed_manifest.json)", ""]
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
        c.need(seal["source_sha256"]==launch["source_sha256"] and len(launch["source_sha256"])==32,"Main 32-source freeze")
        c.need(prep["plan_sha256"]==c.PREPARATION_PLAN_SHA and seal["plan_sha256"]==c.PLAN_SHA
               and seal["preparation"]=={"path":c.PREPARATION,"sha256":c.PREPARATION_SHA}, "Original preparation/new plan separation")
        for mapping in (seal["source_sha256"],ev["source_sha256"],ev["input_sha256"],ev["artifact_sha256"],prep["input_sha256"],prep["source_sha256"]):reg.mapping(mapping)
        c.need(seal["protected_before"]==seal["protected_after"]==launch["protected_before"]==c.protected_hashes() and len(seal["protected_before"])==9,"Nine protected artifacts")
        reg.mapping(seal["protected_before"])
        gate=reg.current(EVAL/"prelabel_gate.json")
        c.need(gate["checked_stages"]==list(c.STAGES) and gate["status"]=="all_stages_verified_before_price_deserialization","Prelabel completion gate")
        mapping_rec=inv["canonical_mapping"]
        mapping=reg.array({"path":mapping_rec["path"],"sha256":mapping_rec["file_sha256"],**{k:mapping_rec[k] for k in ("dtype","shape","values_sha256")}})
        histories={"annual":reg.current("outputs/round79/results.json"),"offset":reg.current("outputs/round97/evaluation/results.json"),
                   "R98":reg.current("outputs/round98/evaluation/results.json")}
        old_seal=reg.document({"path":c.R98_SEALED_MANIFEST,"sha256":c.R98_SEALED_MANIFEST_SHA})
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
            P3,Q3=arr["meta_predictions"],arr["query_predictions"]
            P4,Q4=np.column_stack([P3,np.median(P3,axis=1)]),np.column_stack([Q3,np.median(Q3,axis=1)])
            c.exact(P4[:,:3],P3,"Derived meta preserves original three branches")
            c.exact(Q4[:,:3],Q3,"Derived query preserves original three branches")
            c.need(st["derived_meta_predictions"]==c.matrix_identity(P4)
                   and st["derived_query_predictions"]==c.matrix_identity(Q4), "Derived four-column identities")
            c.need(st["source_branch_order"]==list(c.BRANCHES) and st["stacker_branch_order"]==list(c.STACKER_BRANCHES),"Saved branch orders")
            state=reg.document(b["model_state"],False)
            folder=c.safe(b["model_state"]["path"]).parent
            meta=reg.current(folder/"fit_metadata.json",False);reg.mapping(state["source_sha256"])
            c.need(state["parameters"]==meta["parameters"]==launch["helper_parameters"] and state["input_arrays"]==meta["input_arrays"],"Model parameter/input identity")
            for k,rk in (("P","meta_predictions"),("y","meta_y")):
                expected=c.matrix_identity(P4) if k=="P" else {f:records[rk][f] for f in ("dtype","shape","values_sha256")}
                c.need(state["input_arrays"][k]==expected,"Derived P/original y stored identity")
            c.need(state["input_arrays"]["row_ids_values_sha256"]==c.values_sha(ids),"Model row identity")
            with np.load(folder/"solver_arrays.npz",allow_pickle=False) as z:
                c.need(set(z.files)==set(meta["solver_arrays"]),"Solver evidence schema")
                arrays={k:z[k] for k in z.files}
            for k,a in arrays.items():c.need({"dtype":str(a.dtype),"shape":list(a.shape),"values_sha256":c.values_sha(a)}==meta["solver_arrays"][k],"Solver evidence numerical identity")
            cert=certificate_audit(P4,state,meta,arrays)
            saved_receipt=reg.current(folder/"receipt.json",False)
            c.need(saved_receipt["state_reload_exact"] is True and saved_receipt["solver_arrays_reload_exact"] is True,"Saved model reload receipt")
            reg.mapping(saved_receipt["artifact_sha256"])
            restored=c.read(folder/"state.json")
            c.need(restored==state,"Model JSON full reload")
            old_stage=reg.document(old_seal["stages"][stage]["stage_manifest"])
            c.need(old_stage["input_arrays"]==records,"Old R98 same meta source identity")
            old_meta_path=f"outputs/round98/{stage}/model/fit_metadata.json"
            old_meta=reg.document({"path":old_meta_path,"sha256":old_stage["model_artifact_sha256"][old_meta_path]},False)
            old_state=reg.document(old_stage["model_state"],False)
            c.need(old_state["format"]=="R98_convex_L1_stacker_v1" and old_meta["status"]=="completed"
                   and old_meta["certificate"]["all_checks_passed"] is True,"Original R98 certified fit")
            c.need(old_state["input_arrays"]==old_meta["input_arrays"] and old_state["input_arrays"]["P"]==c.matrix_identity(P3)
                   and old_state["input_arrays"]["y"]==state["input_arrays"]["y"]
                   and old_state["input_arrays"]["row_ids_values_sha256"]==c.values_sha(ids),"Old R98 identical meta P/y/indices")
            old_objective=old_meta["certificate"]["primal_MAE_TWD"]
            close(old_objective,old_state["primal_MAE_TWD"],"Old R98 saved primal identity")
            close(old_objective,st["old_R98_meta_objective_TWD"],"Old R98 stage objective authority")
            close(cert["primal_meta_MAE_TWD"],st["R99_meta_objective_TWD"],"New saved primal authority")
            c.need(st["R99_not_worse_than_R98_meta_verified"] is True
                   and st["R99_not_worse_than_R98_meta_tolerance_TWD"]==.10
                   and cert["primal_meta_MAE_TWD"]<=old_objective+.10,"Saved old feasible solution comparison")
            control=reg.array(b["predictions"]["v22"]);candidate=reg.array(b["predictions"]["R99"])
            P=Q4;w=np.asarray(restored["weights"])
            c.exact(np.median(P[:,:3],axis=1),P[:,3],"Fourth input M exact")
            c.exact(P[:,3],control,"Original full median replay")
            c.exact(((P[:,0]*w[0]+P[:,1]*w[1])+P[:,2]*w[2])+P[:,3]*w[3],candidate,"Saved four-term query formula after JSON reload")
            c.exact(control,arr["control_v22"],"Saved original control")
            item=reg.current(EVAL/stage/"results.json");c.need(item==ev["stages"][stage],"Stage/full result identity")
            old=histories["offset"]["metrics"]["v22"] if stage=="offset" else histories["annual"]["stages"][stage]["metrics"]["R79"]
            for k in ("overall_mae","weighted_mae"):close(item["metrics"]["v22"][k],old[k],"Original v22 saved score replay")
            for k in ("overall_mae","weighted_mae"):
                close(item["metrics"]["R98"][k],histories["R98"]["stages"][stage]["metrics"]["R98"][k],"Original R98 saved score replay")
            ma=metric_audit(stage,item,reg);intervals+=ma["intervals"]
            stages[stage]={"certificate":cert,"metrics_audit":ma,"full_query_formula_exact":True,
                           "control_median_exact":True,"original_v22_saved_metrics_replayed":True,
                           "fit_call_seconds":st["fit_call_elapsed_seconds"],"derived_median_and_JSON_reload_verified":True,
                           "old_R98_same_meta_objective_TWD":old_objective,
                           "meta_objective_delta_vs_R98":cert["primal_meta_MAE_TWD"]-old_objective}
        expected_gate=all(stages[s]["metrics_audit"]["stage_gate_passed"] for s in c.STAGES)
        c.need(ev["research_gate"]["all_three_later_windows_passed"]==expected_gate
               and ev["research_gate"]["automatic_adoption"] is False,"Complete research gate")
        c.need(len(intervals)==12,"All intervals required")
        # Check all input identities again, without deserializing forbidden arrays.
        reg.mapping(reg.hashes.copy())
        c.verify_hashes(reg.hashes)
        review={"round":99,"status":"passed_stored_result_audit","author_role":"Helper author performed this result audit; scorer and driver developed by other agents",
                "frozen_source_count":len(launch["source_sha256"]),"unique_hashed_files":len(reg.hashes),"all_verified_sha256":reg.hashes,"loaded_numeric_arrays":reg.loaded_arrays,
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
