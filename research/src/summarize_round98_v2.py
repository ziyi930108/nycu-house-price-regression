"""Summarize R98 saved aggregates only; no labels, vectors, models or rescoring.

The price-band contribution identity is post-hoc aggregate arithmetic, not a
new candidate, scoring run, selection gate or a rule for price-based inference.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round98_summary_v2"
REPORT = ROOT / "reports/round98_summary_v2.md"
STAGES = ("development", "offset", "confirmation")
NAMES = {"development": "開發窗", "offset": "錯開半年窗", "confirmation": "最近窗"}
BANDS = ("<10m", "10-20m", "20-50m", "50-100m", ">=100m")
BASE = ("outputs/round98/evaluation/results.json", "outputs/round98/execution.json",
        "outputs/round98/evaluation/execution.json", "outputs/round98/sealed_manifest.json")
ALLOWED = set(BASE) | {f"outputs/round98/{s}/{name}" for s in STAGES
                       for name in ("stage_seal.json", "execution.json", "model/fit_metadata.json")}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    with path.with_suffix(".sha256").open("x", encoding="ascii") as f:
        f.write(sha(path) + "\n")


def collect():
    inputs = {}
    def read(name, expected=None):
        need(name in ALLOWED and name.endswith(".json"), "Only assigned aggregate JSON inputs may be read")
        path = ROOT / name
        digest = sha(path)
        if expected is not None:
            need(digest == expected, "Bound aggregate changed: " + name)
        receipt = path.with_suffix(".sha256")
        if receipt.is_file():
            need(receipt.read_text(encoding="ascii").strip() == digest, "JSON receipt differs: " + name)
            inputs[receipt.relative_to(ROOT).as_posix()] = sha(receipt)
        inputs[name] = digest
        return json.loads(path.read_text(encoding="utf-8-sig"))
    ev, main_exec, eval_exec, seal = (read(p) for p in BASE)
    need(main_exec["exit_code"] == eval_exec["exit_code"] == 0, "R98 main/evaluation incomplete")
    need(main_exec["failure"] is None and eval_exec["failure"] is None, "R98 recorded failure")
    need(ev["status"] == "completed_three_later_window_evaluation" and ev["checked_stages"] == list(STAGES), "Missing stage scores")
    need(seal["status"] == "all_temporal_stack_predictions_sealed" and seal["new_meta_fits"] == 3
         and seal["new_native_fits"] == 0 and seal["all_query_reload_exact"] is True, "Model seal incomplete")
    need(ev["sealed_manifest"]["sha256"] == inputs["outputs/round98/sealed_manifest.json"], "Evaluation/seal identity differs")
    need(ev["research_gate"]["all_three_later_windows_passed"] is False and ev["new_CSV"] is False
         and ev["selection_changed"] is False and ev["Private_score"] is None, "Frozen result status differs")
    stages, all_intervals = {}, []
    for s in STAGES:
        saved = ev["stages"][s]
        sr = seal["stages"][s]
        ss = read(sr["stage_manifest"]["path"], sr["stage_manifest"]["sha256"])
        sx = read(sr["execution"]["path"], sr["execution"]["sha256"])
        meta_name = f"outputs/round98/{s}/model/fit_metadata.json"
        meta = read(meta_name, ss["model_artifact_sha256"][meta_name])
        need(sx["exit_code"] == 0 and sx["failure"] is None and meta["status"] == "completed", "Stage fit unfinished")
        need(meta["certificate"]["all_checks_passed"] is True and ss["all_query_reload_exact"] is True, "Numerical certificate/reload failed")
        need(saved["weights"] == meta["certificate"]["weights"] and saved["meta_rows"] == meta["fit_rows"], "Weight/row authority differs")
        need(saved["original_v22_overall_weighted_replay_verified"] is True, "Original control did not replay")
        comparison = saved["comparison"]
        n = saved["query_rows"]
        contributions = []
        for b in comparison["price_bands"]:
            contribution = b["rows"] * b["delta"] / n if b["rows"] else 0.
            contributions.append({**b, "row_share": b["rows"] / n,
                                  "contribution_to_overall_delta_TWD": contribution})
        need(tuple(b["band"] for b in contributions) == BANDS and sum(b["rows"] for b in contributions) == n, "Price-band partition differs")
        total = sum(b["contribution_to_overall_delta_TWD"] for b in contributions)
        delta = comparison["overall_mae"]["delta"]
        need(abs(total - delta) <= 1e-6, "Aggregate contribution identity failed")
        intervals = []
        for family in ("project", "month"):
            fm = saved["paired"]["methods"][family]
            for metric in ("overall", "weighted"):
                v = fm["summary"][metric]
                lo, hi = v["percentile_2_5"], v["percentile_97_5"]
                direction = "unavailable" if lo is None or hi is None else "negative" if hi < 0 else "positive" if lo > 0 else "crosses_zero"
                item = {"stage": s, "family": family, "metric": metric, **v, "interval_direction": direction}
                intervals.append(item); all_intervals.append(item)
        high = contributions[-1]["contribution_to_overall_delta_TWD"]
        stages[s] = {"stage": s, "label": NAMES[s], "cutoff": saved["cutoff"], "query_end": saved["query_end"],
                     "meta_rows": saved["meta_rows"], "query_rows": n, "weights_N_H_U": saved["weights"],
                     "metrics": {k: {x: saved["metrics"][k][x] for x in ("overall_mae", "weighted_mae")} for k in ("N", "H", "U", "v22", "R98")},
                     "comparison": comparison, "paired_intervals": intervals,
                     "paired_support": {f: saved["paired"]["methods"][f]["support"] for f in ("project", "month")},
                     "price_band_contributions": contributions,
                     "aggregate_decomposition": {"formula": "band_rows * band_MAE_delta / all_query_rows", "sum_contributions_TWD": total,
                         "overall_delta_TWD": delta, "absolute_identity_error_TWD": abs(total - delta),
                         "at_least_100m_contribution_TWD": high, "all_other_price_bands_combined_contribution_TWD": total - high,
                         "at_least_100m_contribution_divided_by_overall_delta": high / delta if delta else None,
                         "post_hoc_descriptive_only": True, "new_selection_threshold": False, "permitted_prediction_routing_rule": False},
                     "tails": {k: saved["metrics"][k]["tails"] for k in ("v22", "R98")},
                     "timing": {"fit_call_elapsed_seconds": ss["fit_call_elapsed_seconds"], "solver_seconds": meta["solver_seconds"],
                                "stage_execution_elapsed_seconds": sx["elapsed_seconds"]},
                     "meta_fit_MAE_TWD": meta["certificate"]["primal_MAE_TWD"], "solver_runtime": meta["runtime"],
                     "certificate_passed": True, "full_query_reload_exact": True}
    need(len(all_intervals) == 12, "Twelve saved intervals required")
    fit_total = sum(stages[s]["timing"]["fit_call_elapsed_seconds"] for s in STAGES)
    need(fit_total <= main_exec["elapsed_seconds"], "Fit time cannot exceed sequential main elapsed time")
    return {"round": 98, "status": "summarized_completed_aggregate_evidence", "stages": stages,
            "stage_order": list(STAGES), "difference": "R98 minus v22; negative favors R98",
            "research_gate": ev["research_gate"], "all_intervals": all_intervals,
            "interval_counts": {k: sum(x["interval_direction"] == k for x in all_intervals) for k in ("negative", "crosses_zero", "positive", "unavailable")},
            "timing": {"main_execution_seconds": main_exec["elapsed_seconds"], "meta_fit_calls_total_seconds": fit_total,
                       "fit_call_time_included_in_main": True, "evaluation_execution_seconds": eval_exec["elapsed_seconds"],
                       "development_chat_time_measured": False, "three_meta_fits": 3, "native_fits": 0},
            "limitations": ["Three later, historically reused windows; offset overlaps both August-origin windows.",
                "No pre-auxiliary forward block; not the original three-period adoption test.",
                "All intervals are conditional on saved fixed predictions; no retraining or selection correction, not Private probability.",
                "Global convex combinations do not contain the row-specific median rule; optimal meta L1 need not beat v22.",
                "Price-band contribution arithmetic is post-hoc explanation, not causal proof or a rule for inference.",
                "Chat/development duration is not measured model fit time."],
            "selected_submission": "v22", "Private_score": None, "new_CSV": False,
            "summary_reads_labels_or_row_vectors": False, "new_fit_score_predict_bootstrap_calls": 0,
            "input_sha256": inputs}


def figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter
    font = font_manager.findfont(font_manager.FontProperties(family="Microsoft JhengHei"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Microsoft JhengHei", "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, (a, b) = plt.subplots(2, 1, figsize=(12.8, 11.0), gridspec_kw={"height_ratios": [1, 1.12]})
    fig.subplots_adjust(left=.18, right=.96, top=.85, bottom=.25, hspace=.53)
    fig.suptitle("R98 時間外 L1 組合：改善未跨窗口保持", x=.08, y=.98, ha="left", fontsize=20, fontweight="bold")
    fig.text(.08, .925, "差值 = R98 - v22；低於零較好。三個較晚窗口皆未通過固定研究門檻。", fontsize=12, color="#485362")
    labels = [NAMES[s] + "\n" + summary["stages"][s]["cutoff"] for s in STAGES]
    for j, (key, label, color) in enumerate((("overall_mae", "整體 MAE", "#3569a8"), ("weighted_mae", "既定加權 MAE", "#e89940"))):
        values = [summary["stages"][s]["comparison"][key]["delta"] / 1000 for s in STAGES]
        ys = [i + (j - .5) * .28 for i in range(3)]
        a.barh(ys, values, height=.25, color=color, label=label)
        for y, x in zip(ys, values):
            a.text(x + (.14 if x >= 0 else -.14), y, f"{x:+.2f}", ha="left" if x >= 0 else "right", va="center", fontsize=10)
    a.set_yticks(range(3), labels); a.invert_yaxis(); a.axvline(0, color="#222222", lw=1)
    a.set_xlim(-7.9, 10.8); a.set_xlabel("平均誤差差額（千元／筆）")
    a.set_title("A. 整體與既定加權 MAE 差額", loc="left", fontsize=13, pad=12)
    a.legend(loc="lower right", frameon=False); a.grid(axis="x", alpha=.15); a.set_axisbelow(True)
    colors = ("#6a8fc5", "#9ac2cf", "#63a684", "#e5b86f", "#b86179")
    band_labels = ("低於 1 千萬", "1–2 千萬", "2–5 千萬", "5 千萬–1 億", "至少 1 億")
    for i, s in enumerate(STAGES):
        positive = negative = 0.
        for j, row in enumerate(summary["stages"][s]["price_band_contributions"]):
            v = row["contribution_to_overall_delta_TWD"] / 1000
            start = positive if v >= 0 else negative
            b.barh(i, v, left=start, height=.49, color=colors[j], label=band_labels[j] if i == 0 else None)
            if v >= 0: positive += v
            else: negative += v
        total = summary["stages"][s]["comparison"]["overall_mae"]["delta"] / 1000
        b.scatter([total], [i], marker="D", color="#111111", s=37, zorder=5, label="整體差額（正負貢獻相加）" if i == 0 else None)
    b.set_yticks(range(3), labels); b.invert_yaxis(); b.axvline(0, color="#222222", lw=1)
    b.set_title("B. 五個真價帶對整體差額的貢獻（事後聚合診斷）", loc="left", fontsize=13, pad=12)
    b.set_xlabel("價格帶列數 × 該帶 MAE 差 ／ 全窗列數（千元／筆）")
    b.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    fig.legend(*b.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.57, .16), ncol=3, frameon=False, fontsize=10)
    b.grid(axis="x", alpha=.15); b.set_axisbelow(True)
    fig.text(.08, .035, "錯開半年窗：270 筆億元群貢獻 -6,000.31 元；其他價格帶合計 +1,745.28 元。\n"
             "不能依未知真價分流預測；12 個保存區間均跨零。v22 唯一提交，Private 未知。", fontsize=10.5, color="#485362")
    with path.open("xb") as f:
        fig.savefig(f, format="png", dpi=160)
    plt.close(fig)
    return {"font_family": "Microsoft JhengHei", "font_path": font, "axes_unicode_minus": False,
            "matplotlib_version": matplotlib.__version__, "size_pixels": [2048, 1760], "visual_review_performed_by_script": False}


def render(s):
    f = lambda x: "無" if x is None else f"{x:,.2f}"
    lines = ["# R98：時間外 L1 三分支組合完成摘要", "", "2026-09-29。**固定研究門檻未通過；保留 v22 唯一提交，不產生新 CSV，Private 未知。**",
             "本頁只讀保存的聚合 JSON 與 meta-fit 收據；沒有讀真價／逐筆預測陣列、載入原生模型，沒有新 fit、predict、score 或 bootstrap。", "",
             "![R98 MAE 差與價帶贡献](../outputs/round98_summary_v2/mae_delta.png)", "",
             "## 改了什麼、權重如何得到", "",
             "六個原生模型、特徵、各分支 reference/spatial 都不變。N、H、U 是完成修正後的 NN、log 與 unit 分支；只將逐筆 median(N,H,U) 改成全域非負、和為 1 的線性組合，無截距、分群、衰減或比例搜尋。",
             "每窗獨立用當時較早 forward 預測與標籤最小化等列權 TWD L1：development 用 auxiliary；offset 用 auxiliary＋development 的 2023-02-11 前綴；confirmation 用 auxiliary＋development。每窗独立程序、係數／完整預測封存後才交 scorer。較早 query 合法成為較晚 meta-fit 的角色重疊明載，不能稱整批未見標籤的盲測。", "",
             "|窗口／起日|meta 列數|query 列數|N 權重|H 權重|U 權重|", "|---|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        r = s["stages"][stage]; w = r["weights_N_H_U"]
        lines.append(f'|{stage} / {r["cutoff"]}|{r["meta_rows"]:,}|{r["query_rows"]:,}|{w[0]:.9f}|{w[1]:.9f}|{w[2]:.9f}|')
    lines += ["", "meta-fit 的最優是固定全域凸組合類別內的解；**這個類別一般不能表示逐筆 median 函數**。所以即使求解最佳、或優於每一個單支，也不保證優於 v22 median，更不保證未來改善。N 權重隨較晚 meta-fit 增加、U 減少只是保存結果，不能據此斷言市場機制。", "",
              "## 完整模型結果", "", "差值為 R98 − v22，负值较好，单位新臺幣元。", "",
              "|窗口|v22 整體|R98 整體|整體差|v22 加權|R98 加權|加權差|", "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        r = s["stages"][stage]; m = r["metrics"]; d = r["comparison"]
        lines.append(f'|{stage}|{f(m["v22"]["overall_mae"])}|{f(m["R98"]["overall_mae"])}|{f(d["overall_mae"]["delta"])}|{f(m["v22"]["weighted_mae"])}|{f(m["R98"]["weighted_mae"])}|{f(d["weighted_mae"]["delta"])}|')
    lines += ["", "development／confirmation 整體與加權都較高；offset 的兩者較低，但嘉義縣 1,532 列增加 10.77%，仍未通過事前固定分群條件。並非所有窗口或分群都退步。",
              "門檻是三個較晚窗口的兩個點指標全改善，且任何季度或至少 500 列分群不得超過 5% 退步。這只是升級研究條件，並不是原三期完整採用條件。", "",
              "|窗口|N 單支整體 MAE|H 單支整體 MAE|U 單支整體 MAE|", "|---|---:|---:|---:|"]
    for stage in STAGES:
        m = s["stages"][stage]["metrics"]
        lines.append(f'|{stage}|{f(m["N"]["overall_mae"])}|{f(m["H"]["overall_mae"])}|{f(m["U"]["overall_mae"])}|')
    lines += ["", "R98 三窗皆比每支单獨分支低，但與逐筆 median 的比較有得有失；分支與完整模型不能混稱。", "",
              "## 新舊案、季度與 >5% 反證", "", "|窗口|群|列數|MAE 差|", "|---|---|---:|---:|"]
    for stage in STAGES:
        for section in ("strata", "forecast_quarters"):
            for name, v in s["stages"][stage]["comparison"][section].items():
                lines.append(f'|{stage}|{section}/{name}|{v["rows"]:,}|{f(v["delta"])}|')
    lines += ["", "|窗口|超標項目|列數|MAE 差|相對增加|", "|---|---|---:|---:|---:|"]
    for stage in STAGES:
        for v in s["stages"][stage]["comparison"]["quarter_or_large_group_deterioration_above_5pct"]:
            lines.append(f'|{stage}|{v["section"]}/{v.get("group", v.get("band"))}|{v["rows"]:,}|{f(v["delta"])}|{v["relative_change_pct"]:.3f}%|')
    lines += ["", "各超標項目可能重疊，不能相加當不重複退步列數。小於 500 列群仍完整保留，但不改寫原先門檻。", "",
              "## 五個價格帶對整體差值的分解", "",
              "每帶貢獻 = 該帶列數 × 該帶 MAE 差 ÷ 全窗列數。五帶合計等於保存的 overall 差（算術誤差核至 0.000001 元）。**這是事後聚合診斷，不是新選型門檻；不得依未知真價在推論時分流、裁價或挑列。**", "",
              "|窗口|真價帶|列數|帶內 MAE 差|對全窗整體差的貢獻|", "|---|---|---:|---:|---:|"]
    for stage in STAGES:
        for b in s["stages"][stage]["price_band_contributions"]:
            lines.append(f'|{stage}|{b["band"]}|{b["rows"]:,}|{f(b.get("delta"))}|{f(b["contribution_to_overall_delta_TWD"])}|')
    off = s["stages"]["offset"]; offd = off["aggregate_decomposition"]; high = off["price_band_contributions"][-1]
    conf = s["stages"]["confirmation"]; mid = conf["price_band_contributions"][2]; chigh = conf["price_band_contributions"][-1]
    lines += ["", f'offset 的億元以上 **{high["rows"]:,} 列（{high["row_share"]:.3%}）** 帶內 MAE 降低 {f(-high["delta"])} 元，對全窗貢獻 {f(offd["at_least_100m_contribution_TWD"])} 元；其他四帶合計反而是 +{f(offd["all_other_price_bands_combined_contribution_TWD"])} 元，因此全窗淨改善 {f(-offd["overall_delta_TWD"])} 元高度依賴這個小群。貢獻約占淨改善 {offd["at_least_100m_contribution_divided_by_overall_delta"]:.1%}，超過 100% 是其他價格帶抵銷所致，不是改善機率。',
              f'confirmation 的 **2,000–5,000 萬 {mid["rows"]:,} 列** 改善 {f(-mid["delta"])} 元、貢獻 {f(mid["contribution_to_overall_delta_TWD"])} 元；1,000–2,000 萬也改善，都是必須保留的有利反證。但億元群 {chigh["rows"]:,} 列對全窗貢獻 +{f(chigh["contribution_to_overall_delta_TWD"])} 元，與部分其他群抵銷上述改善後，最近窗整體仍退步。',
              "全域加權會改變逐筆 median 對極端分支的反應，可能與尾端年度差異、分支相關性和市場移轉有關；上述分解只定位誤差貢獻，**不能證明這些機制的因果關係**，也不能保證換另一組權重會改善。", "",
              "## 全部十二個配對區間", "", "兩族各 2,000 次；stage index development=0、offset=1、confirmation=2，seed20260925。", "",
              "|窗口|抽樣|指標|點差|2.5%|97.5%|有效／無效|", "|---|---|---|---:|---:|---:|---|"]
    for v in s["all_intervals"]:
        lines.append(f'|{v["stage"]}|{v["family"]}|{v["metric"]}|{f(v["point_delta"])}|{f(v["percentile_2_5"])}|{f(v["percentile_97_5"])}|{v["valid_replicates"]} / {v["invalid_replicates"]}|')
    lines += ["", "十二個保存區間全部跨零，不能稱確定改善，也不能說全面顯著退步。project 按既定新舊案層抽整個地理名稱建案鍵；month 抽四段三個月循環區塊。區間條件於固定模型，不含重新 meta-fit／native 訓練或累積選型偏差，不是 Private 勝率。", "",
              "## 實測運算時間與結論", "", "|窗口|meta-fit 呼叫秒數|其中 solver 秒數|", "|---|---:|---:|"]
    for stage in STAGES:
        t = s["stages"][stage]["timing"]
        lines.append(f'|{stage}|{t["fit_call_elapsed_seconds"]:.3f}|{t["solver_seconds"]:.3f}|')
    t = s["timing"]
    lines += ["", f'三次 CPU meta-fit 呼叫合計 **{t["meta_fit_calls_total_seconds"]:.3f} 秒**，已包含於主流程 **{t["main_execution_seconds"]:.3f} 秒**，不可再相加；獨立評分／既定 bootstrap 程序 **{t["evaluation_execution_seconds"]:.3f} 秒**。本輪原生模型新訓練 **0 次**。這些是收據量測，沒有把開發、聊天、查核與等待時間冒充模型訓練時間，也未量整段開發工時。',
              "三窗全完成，固定門檻未過：保留全部權重、求解證書、預測與反證，不因結果重新搜比例。即使曾通過，仍缺 auxiliary 之前的 forward block，需另固定更早起點並重建至少六 native 與 learned states，才有原三期完整比較；目前沒有啟動這項補驗證。",
              "使用者選定仍是 v22，Private 未知；沒有全量候選、新 CSV 或 Kaggle 操作。保存重載通過也不等於全新環境從零重訓驗證。", "",
              "[固定計畫](round98_plan.md) · [契約審查](round98_validation_contract_review.md) · [完整評估結果](../outputs/round98/evaluation/results.json) · [聚合摘要](../outputs/round98_summary_v2/summary.json) · [摘要程式](../src/summarize_round98_v2.py)", ""]
    return "\n".join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(), "Summary output already exists; no overwrite")
    tick = time.monotonic(); start = datetime.now(timezone.utc).isoformat()
    summary = collect()
    OUT.mkdir()
    code, failure = 1, None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            graphic = figure(summary, OUT / "mae_delta.png")
        summary["figure"] = graphic
        summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["source_sha256"] = sha(__file__)
        write_json(OUT / "summary.json", summary)
        with REPORT.open("x", encoding="utf8", newline="\n") as f: f.write(render(summary))
        need(all(sha(ROOT / p) == h for p, h in summary["input_sha256"].items()), "Aggregate inputs changed")
        artifacts = {p.relative_to(ROOT).as_posix(): sha(p) for p in (REPORT, OUT / "summary.json", OUT / "summary.sha256", OUT / "mae_delta.png")}
        write_json(OUT / "receipt.json", {"status": "completed", "source_sha256": sha(__file__), "input_sha256": summary["input_sha256"],
                "artifact_sha256": artifacts, "read_scope": "saved aggregate JSON only; no row labels/vectors/model objects",
                "new_fit_score_predict_bootstrap_calls": 0, "new_CSV": False, "selected_submission": "v22", "Private_score": None,
                "decomposition_identity_all_stages_verified": True, "saved_CI_count": len(summary["all_intervals"]),
                "plot": graphic, "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in captured],
                "started_at_utc": start, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - tick})
        code = 0
        print(json.dumps({"status": "completed", "interval_counts": summary["interval_counts"], "artifacts": artifacts,
                          "elapsed_seconds": time.monotonic() - tick}, ensure_ascii=False))
    except Exception:
        import traceback
        failure = traceback.format_exc()
        raise
    finally:
        write_json(OUT / "execution.json", {"exit_code": code, "failure": failure, "started_at_utc": start,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - tick,
                "source_sha256": sha(__file__), "automatic_retry": False, "new_fit_score_predict_bootstrap_calls": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not args.run: p.error("--run required")
    run()
