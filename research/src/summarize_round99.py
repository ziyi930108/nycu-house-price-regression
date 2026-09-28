"""Concise R99 summary from saved aggregate JSON only; no arrays or rescoring."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round99_summary"
REPORT = ROOT / "reports/round99_summary.md"
STAGES = ("development", "offset", "confirmation")
BASE = ("outputs/round99/evaluation/results.json", "outputs/round99/sealed_manifest.json",
        "outputs/round99/execution.json", "outputs/round99/evaluation/execution.json")
ALLOWED = set(BASE) | {f"outputs/round99/{s}/{p}" for s in STAGES for p in ("stage_seal.json", "model/fit_metadata.json")}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    with path.with_suffix(".sha256").open("x", encoding="ascii") as f: f.write(sha(path) + "\n")


def collect(inputs):
    def read(name, expected=None):
        need(name in ALLOWED, "Only saved aggregate JSON may be read")
        p = ROOT / name; digest = sha(p)
        if expected is not None: need(digest == expected, "Bound JSON changed: " + name)
        sidecar = p.with_suffix(".sha256")
        if sidecar.is_file():
            need(sidecar.read_text(encoding="ascii").strip() == digest, "JSON sidecar differs")
            inputs[sidecar.relative_to(ROOT).as_posix()] = sha(sidecar)
        inputs[name] = digest
        return json.loads(p.read_text(encoding="utf-8-sig"))
    result, seal, main, evaluation = (read(p) for p in BASE)
    need(main["exit_code"] == evaluation["exit_code"] == 0 and main["failure"] is None and evaluation["failure"] is None, "Incomplete run")
    need(result["status"] == "completed_three_later_window_evaluation" and result["checked_stages"] == list(STAGES), "Incomplete scores")
    need(result["sealed_manifest"]["sha256"] == inputs[BASE[1]] and seal["new_meta_fits"] == 3 and seal["new_native_fits"] == 0, "Seal identity/fit count")
    need(result["research_gate"]["all_three_later_windows_passed"] is False and result["new_CSV"] is False and result["Private_score"] is None, "Saved decision differs")
    stages, intervals = {}, []
    for s in STAGES:
        r = result["stages"][s]; binding = seal["stages"][s]["stage_manifest"]
        ss = read(binding["path"], binding["sha256"])
        mp = f"outputs/round99/{s}/model/fit_metadata.json"
        meta = read(mp, ss["model_artifact_sha256"][mp]); cert = meta["certificate"]
        need(meta["status"] == "completed" and cert["all_checks_passed"] is True and ss["all_query_reload_exact"] is True, "Fit certificate/reload failed")
        need(r["weights"] == cert["weights"] and len(r["weights"]) == 4, "Four saved weights differ")
        need(ss["R99_meta_objective_TWD"] == cert["primal_MAE_TWD"] and ss["R99_not_worse_than_R98_meta_verified"] is True, "Meta objective identity")
        need(r["original_v22_overall_weighted_replay_verified"] and r["original_R98_overall_weighted_replay_verified"], "Controls not verified")
        comparisons = {"v22": r["comparison"], "R98": r["comparison_vs_R98"]}
        decompositions = {}
        for control, d in comparisons.items():
            bands = [{**b, "row_share": b["rows"] / r["query_rows"],
                      "overall_delta_contribution_TWD": b["rows"] * b["delta"] / r["query_rows"] if b["rows"] else 0.}
                     for b in d["price_bands"]]
            total = sum(b["overall_delta_contribution_TWD"] for b in bands)
            need(sum(b["rows"] for b in bands) == r["query_rows"] and abs(total - d["overall_mae"]["delta"]) <= 1e-6, "Band decomposition identity")
            need(bands[-1]["band"] == ">=100m", "Tail band identity")
            decompositions[control] = {"formula": "band_rows * saved_band_MAE_delta / all_query_rows", "bands": bands,
                "sum_TWD": total, "at_least_100m_contribution_TWD": bands[-1]["overall_delta_contribution_TWD"],
                "other_bands_combined_contribution_TWD": total - bands[-1]["overall_delta_contribution_TWD"],
                "post_hoc_aggregate_diagnostic": True, "not_a_selection_gate_or_inference_rule": True}
        for family in ("project", "month"):
            for metric in ("overall", "weighted"):
                v = r["paired"]["methods"][family]["summary"][metric]
                need(v["valid_replicates"] + v["invalid_replicates"] == 2000, "Replicate count differs")
                lo, hi = v["percentile_2_5"], v["percentile_97_5"]
                need(lo is not None and hi is not None and lo <= 0 <= hi, "Expected saved interval does not cross zero")
                intervals.append({"stage": s, "family": family, "metric": metric, **v})
        stages[s] = {"cutoff": r["cutoff"], "query_end": r["query_end"], "meta_rows": r["meta_rows"], "query_rows": r["query_rows"],
            "weights_N_H_U_M": r["weights"], "comparisons": comparisons,
            "metrics": {name: {k: r["metrics"][name][k] for k in ("overall_mae", "weighted_mae")} for name in ("N", "H", "U", "v22", "R98", "R99")},
            "meta_objective": {"R98_MAE_TWD": ss["old_R98_meta_objective_TWD"], "median_MAE_TWD": cert["median_vertex_MAE_TWD"],
                "R99_MAE_TWD": cert["primal_MAE_TWD"], "R99_minus_R98_TWD": cert["primal_MAE_TWD"] - ss["old_R98_meta_objective_TWD"],
                "R99_minus_median_TWD": cert["primal_MAE_TWD"] - cert["median_vertex_MAE_TWD"], "engineering_tolerance_TWD": .10},
            "decompositions": decompositions, "tails": {k: r["metrics"][k]["tails"] for k in ("v22", "R98", "R99")},
            "fit_call_seconds": ss["fit_call_elapsed_seconds"], "solver_seconds": meta["solver_seconds"],
            "full_query_reload_exact": True, "meta_certificate_passed": True}
    need(len(intervals) == 12, "All twelve primary intervals required")
    fit_seconds = sum(r["fit_call_seconds"] for r in stages.values())
    need(fit_seconds < main["elapsed_seconds"], "Fit/main time hierarchy")
    return {"round": 99, "status": "aggregate_summary_completed", "stage_order": list(STAGES), "stages": stages,
            "primary_intervals_vs_v22": intervals, "all_12_intervals_cross_zero": True,
            "R98_point_comparisons_only": True, "new_R98_bootstrap_calls": 0,
            "timing": {"main_execution_seconds": main["elapsed_seconds"], "meta_fit_calls_total_seconds": fit_seconds,
                "meta_fit_time_already_included_in_main": True, "evaluation_execution_seconds": evaluation["elapsed_seconds"],
                "developer_chat_time_measured": False, "new_meta_fits": 3, "new_native_fits": 0},
            "research_gate": result["research_gate"], "selected_submission": "v22", "Private_score": None,
            "new_CSV": False, "new_fit_predict_score_bootstrap_calls_by_summary": 0, "reads_row_arrays_or_labels": False,
            "input_sha256": inputs,
            "limitations": ["Adaptive research after observed R98; reused and overlapping later query windows, not blind validation.",
                "Missing earlier forward meta block for auxiliary; original adoption gate still unavailable.",
                "Meta optimality within the enlarged convex class is not future, weighted-group or Private optimality.",
                "Saved bootstrap excludes retraining and cumulative selection; not Private probability.",
                "True-price-band arithmetic is post-hoc aggregation, not causal proof, a new gate or an inference route."]}


def render(r):
    f = lambda x: f"{x:,.2f}"
    lines = ["# R99 四欄 L1 組合：完成摘要", "", "2026-09-29。**固定探索門檻未通過；v22 唯一提交不變，Private 未知，沒有新 CSV 或採用。**",
             "只讀保存聚合 JSON／模型 metadata，沒有讀逐筆標籤或向量，沒有新 fit、score、predict 或 bootstrap。", "",
             "唯一改動是在完整 N/H/U 分支後加入 M=median(N,H,U)，各窗只用較早 forward block 學四個非負、和為 1 的全域 L1 權重。六個原生模型與 reference/spatial 沒有重新訓練。", "",
             "|窗口|N|H|U|M|meta／query 列數|", "|---|---:|---:|---:|---:|---:|"]
    for s in STAGES:
        x=r["stages"][s]; w=x["weights_N_H_U_M"]
        lines.append(f'|{s}|{w[0]:.9f}|{w[1]:.9f}|{w[2]:.9f}|{w[3]:.9f}|{x["meta_rows"]:,} / {x["query_rows"]:,}|')
    lines += ["", "## 較晚窗口結果", "", "差為 R99 − 對照，負數較好，單位新臺幣元。R98 只報固定點差，不另做 bootstrap。", "",
              "|窗口|v22 overall|R98 overall|R99 overall|對 v22 整體差／加權差|對 R98 整體差／加權差|", "|---|---:|---:|---:|---:|---:|"]
    for s in STAGES:
        x=r["stages"][s]; m=x["metrics"]; a=x["comparisons"]["v22"]; b=x["comparisons"]["R98"]
        lines.append(f'|{s}|{f(m["v22"]["overall_mae"])}|{f(m["R98"]["overall_mae"])}|{f(m["R99"]["overall_mae"])}|{f(a["overall_mae"]["delta"])} / {f(a["weighted_mae"]["delta"])}|{f(b["overall_mae"]["delta"])} / {f(b["weighted_mae"]["delta"])}|')
    lines += ["", "相對 v22 仍是一窗改善、兩窗退步；offset 兩指標改善，但嘉義縣仍超過固定 5% 退步門檻。相對 R98，offset 與 confirmation 改善，development 反而退步，不能說新增 median 全面有效。", "",
              "## meta 目標改善不等於 query 改善", "",
              "|窗口|meta：R98 MAE|meta：原 median MAE|meta：R99 MAE|R99−R98|R99−median|", "|---|---:|---:|---:|---:|---:|"]
    for s in STAGES:
        m=r["stages"][s]["meta_objective"]
        lines.append(f'|{s}|{f(m["R98_MAE_TWD"])}|{f(m["median_MAE_TWD"])}|{f(m["R99_MAE_TWD"])}|{f(m["R99_minus_R98_TWD"])}|{f(m["R99_minus_median_TWD"])}|')
    lines += ["", "M 權重 1 與舊 R98 權重加上 M 權重 0 都是可行解；本輪三個數值證書與不劣於舊 meta 目標（容差 0.10 元）檢查通過。這只是訓練目標的可行性與最優性檢查，不保證較晚 query、既定群加權 MAE 或 Private 改善。", "",
              "## 有利與不利群"]
    lines += ["", "offset 新案／Q2–Q4 相對 v22 改善，但 seen／Q1 退步；confirmation 的 Q2、Q3 與 1,000–5,000 萬群改善，但 seen、新案與 Q1、Q4 仍較高。development Q1 改善，其餘季度與新舊案較高。", "",
              "|窗口|對 v22 超過 5% 的固定大群項目|列數|MAE 差|增幅|", "|---|---|---:|---:|---:|"]
    for s in STAGES:
        adverse=r["stages"][s]["comparisons"]["v22"]["quarter_or_large_group_deterioration_above_5pct"]
        for a in adverse: lines.append(f'|{s}|{a["section"]}/{a.get("group",a.get("band"))}|{a["rows"]:,}|{f(a["delta"])}|{a["relative_change_pct"]:.3f}%|')
        if not adverse: lines.append(f'|{s}|無；但兩個整體點指標仍未改善|—|—|—|')
    lines += ["", "以上群可能重疊，不能相加作不重複列數。相對 R98 三窗均無該 >5% 大群項目，但不是本輪另設通過條件。所有城市、月份、價格帶与六個向量完整指標保留在原結果 JSON。", "",
              "## 億元群對整體差額的貢獻", "",
              "貢獻 = 保存的群列數 × 群 MAE 差 ÷ 全窗列數。這是事後聚合診斷，五價帶加總與保存 overall 差一致至 0.000001 元；不是新門檻、因果證明，也不能依未知真價在推論時分流。", "",
              "|窗口|億元群列數|帶內差：對 v22|對全窗貢獻：對 v22|其他四帶貢獻：對 v22|億元群貢獻：對 R98|", "|---|---:|---:|---:|---:|---:|"]
    for s in STAGES:
        a=r["stages"][s]["decompositions"]["v22"]; b=r["stages"][s]["decompositions"]["R98"]; high=a["bands"][-1]
        lines.append(f'|{s}|{high["rows"]:,}|{f(high["delta"])}|{f(a["at_least_100m_contribution_TWD"])}|{f(a["other_bands_combined_contribution_TWD"])}|{f(b["at_least_100m_contribution_TWD"])}|')
    off=r["stages"]["offset"]["decompositions"]["v22"]; conf=r["stages"]["confirmation"]["decompositions"]["v22"]; mid=conf["bands"][2]
    lines += ["", f'offset 的 270 筆億元群仍是主要有利貢獻（{f(off["at_least_100m_contribution_TWD"])} 元），其他價格帶合計 {f(off["other_bands_combined_contribution_TWD"])} 元。相較 R98，這個億元群反而稍退，而全窗仍改善，表示不能把兩個對照混為相同尾端機制。',
              f'confirmation 的 2,000–5,000 萬 {mid["rows"]:,} 列相對 v22 改善 {f(-mid["delta"])} 元，對整體貢獻 {f(mid["overall_delta_contribution_TWD"])} 元；億元群不利貢獻仍抵銷部分其他群的改善。這些是誤差貢獻位置，不是已證明的原因。', "",
              "## 十二個主比較區間", "", "全為 R99−v22；每族 2,000 次，所有保存區間均經核對跨零，沒有重新抽樣。完整端點見[獨立稽核](round99_independent_review.md)及聚合摘要 JSON，不重複列長表。"]
    t=r["timing"]
    lines += ["", "區間未含重新訓練、累積選型或未來分布移轉；不能稱確定改善／全面顯著退步，bootstrap 不是 Private 勝率。", "",
              "## 實测時間與保留限制", "", "|窗口|meta-fit 呼叫秒數|其中 solver 秒數|", "|---|---:|---:|"]
    for s in STAGES:
        x=r["stages"][s];lines.append(f'|{s}|{x["fit_call_seconds"]:.3f}|{x["solver_seconds"]:.3f}|')
    lines += ["", f'三個 CPU meta-fit 合計 **{t["meta_fit_calls_total_seconds"]:.3f} 秒**，已包含在主流程 **{t["main_execution_seconds"]:.3f} 秒**，不可再相加；獨立評分／既定 bootstrap **{t["evaluation_execution_seconds"]:.3f} 秒**。新 native fit **0 次**。開發、聊天與等待時間沒有量成模型訓練工時。',
              "本輪是看過 R98 後的適應性研究；窗口已重複使用，offset 與另兩窗重疊，缺 auxiliary 之前的 forward meta block，不能稱獨立盲測或原三期完整採用驗證。三窗固定探索門檻未過，保留正反證據而不搜尋新比例；沒有啟動更早 block、全量候選或新 CSV。使用者唯一選定仍為 v22，Private 未知。", "",
              "[固定計畫](round99_plan.md) · [原評估結果](../outputs/round99/evaluation/results.json) · [聚合摘要](../outputs/round99_summary/summary.json) · [摘要程式](../src/summarize_round99.py)", ""]
    return "\n".join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(), "Existing summary preserved; no overwrite")
    OUT.mkdir(); start=datetime.now(timezone.utc).isoformat();tick=time.monotonic();inputs={};code=1;failure=None
    try:
        result=collect(inputs);result["source_sha256"]=sha(__file__);result["created_at_utc"]=datetime.now(timezone.utc).isoformat()
        write(OUT/"summary.json",result)
        with REPORT.open("x",encoding="utf8",newline="\n") as f:f.write(render(result))
        need(all(sha(ROOT/p)==h for p,h in inputs.items()),"Input JSON changed")
        artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in (REPORT,OUT/"summary.json",OUT/"summary.sha256")}
        write(OUT/"receipt.json",{"status":"completed","source_sha256":sha(__file__),"input_sha256":inputs,"artifact_sha256":artifacts,
              "all_12_saved_intervals_cross_zero":True,"aggregate_band_identities_verified":True,"reads_labels_or_row_vectors":False,
              "new_fit_score_predict_bootstrap_calls":0,"selected_submission":"v22","Private_score":None,"new_CSV":False,
              "started_at_utc":start,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-tick})
        code=0;print(json.dumps({"status":"completed","artifacts":artifacts,"timing":result["timing"],"elapsed_seconds":time.monotonic()-tick}))
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        write(OUT/"execution.json",{"exit_code":code,"failure":failure,"source_sha256":sha(__file__),"input_sha256":inputs,
              "started_at_utc":start,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-tick,
              "automatic_retry":False,"new_fit_score_predict_bootstrap_calls":0})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true");args=parser.parse_args()
    if not args.run:parser.error("--run required")
    run()
