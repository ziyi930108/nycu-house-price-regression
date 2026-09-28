"""Review fixed reference attenuation from completed receipts; no fit or prediction."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("development", "confirmation", "auxiliary")
LABELS = ("中間年度", "最新年度", "較早年度")


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def read(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    r = read("outputs/round96/results.json")
    e = read("outputs/round96/execution.json")
    g = read("reports/round96_diagnostics.json")
    assert r["status"] == "completed" and e["exit_code"] == 0
    assert set(r["checked_stages"]) == set(STAGES)
    assert r["new_fits"] == 0 and r["new_NN_fits"] == 0
    assert r["protected_before"] == r["protected_after"]
    assert read("outputs/round96/diagnostics/execution.json")["exit_code"] == 0
    intervals = []
    for control in ("v22", "v24"):
        name = f"outputs/round96/paired_vs_{control}/results.json"
        p = read(name)
        assert p["status"] == "completed" and set(p["stages"]) == set(STAGES)
        assert (ROOT / name).with_suffix(".sha256").read_text().strip() == sha(name)
        for s in STAGES:
            for method in ("project", "month"):
                for metric, x in p["stages"][s][method]["summary"].items():
                    intervals.append({"control": control, "stage": s, "method": method,
                                      "metric": metric, "point_delta": x["point_delta"],
                                      "lower": x["percentile_2_5"], "upper": x["percentile_97_5"]})
    assert len(intervals) == 24
    stages = []
    layer_rows = []
    for s, label in zip(STAGES, LABELS):
        x = r["stages"][s]
        m = x["metrics"]
        stages.append({"stage": s, "metrics": {c: {k: m[c][k] for k in ("overall_mae", "weighted_mae")} for c in ("R96", "v22", "v24")},
                       "delta_v22": {k: m["R96"][k] - m["v22"][k] for k in ("overall_mae", "weighted_mae")},
                       "group_deltas": {k: m["R96"]["strata"][k]["mae"] - m["v22"]["strata"][k]["mae"] for k in ("seen", "new_nonmissing")},
                       "price_bands": g["stages"][s]["price_bands"]})
        for branch, layers in x["branch_metrics"].items():
            for layer, new in layers.items():
                old = x["original_branch_metrics"][branch][layer]
                layer_rows.append({"stage": s, "label": label, "branch": branch, "layer": layer,
                                   "old_mae": old["overall_mae"], "new_mae": new["overall_mae"],
                                   "delta": new["overall_mae"] - old["overall_mae"]})
    latest = -stages[1]["delta_v22"]["overall_mae"] / stages[1]["metrics"]["v22"]["overall_mae"]
    gates = {"all_three_overall_improve": all(x["delta_v22"]["overall_mae"] < 0 for x in stages),
             "all_three_weighted_improve": all(x["delta_v22"]["weighted_mae"] < 0 for x in stages),
             "latest_at_least_half_percent": latest >= .005}
    passed = all(gates.values())
    counts = {c: {"negative": sum(i["control"] == c and i["upper"] < 0 for i in intervals),
                  "cross_zero": sum(i["control"] == c and i["lower"] <= 0 <= i["upper"] for i in intervals),
                  "positive": sum(i["control"] == c and i["lower"] > 0 for i in intervals)} for c in ("v22", "v24")}
    evidence = ["reports/round96_plan.md", "reports/round96_helper_selfcheck.json", "reports/r96_validation_audit.md",
                "reports/regression_direction_after_attention_audit.md", "outputs/round96/results.json",
                "outputs/round96/execution.json", "reports/round96_diagnostics.json",
                "outputs/round96/paired_vs_v22/results.json", "outputs/round96/paired_vs_v24/results.json"]
    decision = {"round": 96, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "candidate": "fixed_half_reference_confidence_with_unchanged_global_trend",
                "adoption_gate_passed": passed, "gate_components": gates,
                "recommendation": "eligible_for_new_origin_check_no_automatic_adoption" if passed else "retain_research_no_final_fit_no_CSV",
                "latest_fractional_improvement": latest, "stage_summary": stages,
                "branch_layer_comparison": layer_rows, "paired_intervals": intervals, "interval_counts": counts,
                "main_elapsed_seconds": e["elapsed_seconds"], "new_fits": 0, "new_tree_fits": 0, "new_NN_fits": 0,
                "selected_submission": "v22", "selector_changed": False, "private_score": None,
                "new_CSV": False, "new_test_predictions": False, "protected_before_after_equal": True,
                "source_sha256": r["source_sha256"], "review_source_sha256": sha("src/round96_review.py"),
                "evidence_sha256": {p: sha(p) for p in evidence},
                "limitations": ["Repeated historical windows; not a blind holdout.",
                                "Reference and spatial corrections change together; not an isolated reference causal estimate.",
                                "No native retraining; same-window saved components and fitted states are reused.",
                                "Bootstrap does not correct adaptive model selection or represent Private win probability.",
                                "No new-origin model rebuild or test evaluation performed."]}
    out = ROOT / "outputs/round96/decision_review.json"
    assert not out.exists()
    with out.open("x", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    with out.with_suffix(".sha256").open("x", encoding="utf-8") as f:
        f.write(sha("outputs/round96/decision_review.json") + "\n")

    lines = ["# R96：降低局部參考成交的修正力度", "",
             ("固定三期數值門檻通過，仍需分群審查及新切點重建，不自動採用。" if passed else "固定三期數值門檻未過，保留研究，不產生新的提交檔。") + "v22 仍為唯一選定版本，Private 未知。", "",
             "## 為什麼測試", "",
             "歷史分支分析提示，參考成交修正有時會抵銷原生模型的改善，但層差本身不能證明修正過強。這次固定把局部信任程度乘上 0.5；整體時間趨勢保持，空間修正同步使用新的參考價格及 confidence，再由原三分支取中位數。", "",
             "令 B_future 為原趨勢修正後基準，P 為原參考預測，c 為原 confidence，則 P_new = B_future + 0.5(P − B_future)、c_new = 0.5c。這不是把全部價格折半，也不是以未知 test 真價調整個別房屋。原 local_price 使用 max(base,1)，原 P 已 max(...,0)；本候選不另裁價，c=0 列直接保留原 P。exact 算式及 fallback 以固定 helper／計畫為準。", "",
             "最強反論是局部參考本來就可能有價值；降低 confidence 也會增加空間修正剩餘份額，未必更保守。R88 已出現撤減部分修正傷害新建案的結果，因此不把此假設當作必然改良。", "",
             "## 完整結果", "", "單位 TWD MAE；差額為 R96 減 v22，負數較好。", "",
             "|期別|v22|R96|整體差額|固定加權差額|", "|---|---:|---:|---:|---:|"]
    for label, x in zip(LABELS, stages):
        lines.append(f'|{label}|{x["metrics"]["v22"]["overall_mae"]:,.2f}|{x["metrics"]["R96"]["overall_mae"]:,.2f}|{x["delta_v22"]["overall_mae"]:+,.2f}|{x["delta_v22"]["weighted_mae"]:+,.2f}|')
    lines += ["", f'最新年度相對改善 {latest * 100:.4f}%；門檻：{json.dumps(gates, ensure_ascii=False)}。', "",
              "|期別|既有建案差額|新建案差額|", "|---|---:|---:|"]
    for label, x in zip(LABELS, stages):
        lines.append(f'|{label}|{x["group_deltas"]["seen"]:+,.2f}|{x["group_deltas"]["new_nonmissing"]:+,.2f}|')
    lines += ["", "|期別|真價帶|筆數|群差額|", "|---|---|---:|---:|"]
    for label, x in zip(LABELS, stages):
        for b in x["price_bands"]:
            lines.append(f'|{label}|{b["band"]}|{b["rows"]:,}|{b["R96"]-b["v22"]:+,.2f}|')
    lines += ["", "真價帶只作歷史事後診斷，不能用於未知 test 分流。完整城市／季度／月份／testmix 見分群檔；testmix 不是 Private 分數。", "",
              "## 分支與全部配對區間", "", "|期別|分支|層|原 MAE|新 MAE|差額|", "|---|---|---|---:|---:|---:|"]
    for x in layer_rows:
        lines.append(f'|{x["label"]}|{x["branch"]}|{x["layer"]}|{x["old_mae"]:,.2f}|{x["new_mae"]:,.2f}|{x["delta"]:+,.2f}|')
    lines += ["", "層差不可相加作因果分析；降低 reference 信任同時影響 spatial，完整 median 才是本候選輸出。", "",
              "|對照|期別|抽法|指標|點差额|95%區間|", "|---|---|---|---|---:|---|"]
    for i in intervals:
        lines.append(f'|{i["control"]}|{i["stage"]}|{i["method"]}|{i["metric"]}|{i["point_delta"]:+,.2f}|[{i["lower"]:+,.2f}, {i["upper"]:+,.2f}]|')
    lines += ["", f'負／跨零／正區間數：v22={counts["v22"]}；v24={counts["v24"]}。區間相依，未校正多輪選型，不是 Private 勝率。', "",
              "## 執行、限制與證據", "", f'主流程實測 {e["elapsed_seconds"]:.3f} 秒；0 樹模型／NN／wrapper fit。CPU 套用既有同窗 reference／spatial 狀態不是新訓練。合成、診斷及審查另計，沒有量測整段聊天時間。', "",
              "原三分支與完整控制重播、來源 SHA、fallback 和保護檔案依主結果核對。沒有 query 標籤進修正函式，價格只供評分；沒有 new test prediction、CSV、Kaggle 操作或 Private 答案。新切點若要比較 v22／v24／R96，仍需重建七個原生模型及全部 fit 狀態，不能沿後期模型回測。", "",
              "- [事前計畫](round96_plan.md)、[合成](round96_helper_selfcheck.json)、[獨立 preflight](r96_validation_audit.md)",
              "- [完整結果](../outputs/round96/results.json)、[分群](round96_diagnostics.md)、[決策](../outputs/round96/decision_review.json)",
              "- [對 v22 配對](round96_paired_vs_v22.md)、[對 v24 配對](round96_paired_vs_v24.md)",
              "- [方向與歷史反證](regression_direction_after_attention_audit.md)"]
    with (ROOT / "reports/round96_report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"gate": gates, "recommendation": decision["recommendation"], "interval_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
