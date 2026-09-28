"""Post-completion R97 artifact audit; never load labels/models or score rows.

Only --run performs the audit. It requires completed main and evaluation receipts.
It checks stored metadata, file hashes, saved prediction arithmetic and stored
bootstrap quantiles. It does not import a training/evaluation module or parse
archived failure sources. Native model files are hashed, never deserialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "outputs/round97"
EVAL = MAIN / "evaluation"
AUDIT = ROOT / "outputs/round97_independent_review"
REPORT = ROOT / "reports/round97_independent_review.md"
RECEIPT = MAIN / "independent_review.json"
PROTECTED9 = ROOT / "outputs/github_publish_r97_20260929/protected_before.json"
CORE = {
    "src/round97_offset_experiment.py": "00d927b81170b9b8acfab984bb72635f413513f62da42733b85e5873528914f1",
    "src/round97_offset_evaluate.py": "fbe0b0f841fce7f13afa6e499b6f88498ae061c816ab381fc89a5357eb87d0d8",
    "reports/round97_plan.md": "8372960e832bd66203374c942e0f7988e66522aedf358c894d892e82669a3a08",
}
NATIVES = ("raw", "light", "log", "unit42", "unit137", "N_area", "N_prior")
BRANCHES = ("N", "H", "U22", "U24")
WEIGHTS = {"seen": .36617593949822613, "new_nonmissing": .6338240605017739, "missing": 0.0}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def safe(path):
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    need(p.is_relative_to(ROOT), "Path outside workspace")
    return p


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding="utf-8-sig"))


def locked(path):
    p = safe(path)
    need(p.with_suffix(".sha256").read_text(encoding="ascii").strip() == sha(p), "JSON sidecar: " + rel(p))
    return read(p)


def write(path, value):
    p = safe(path)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    with p.with_suffix(".sha256").open("x", encoding="ascii") as f:
        f.write(sha(p) + "\n")


def exact(a, b, message):
    a, b = np.asarray(a), np.asarray(b)
    need(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), message)


def vsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def close(a, b, message):
    need(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-6, message)


def merged(*maps):
    result = {}
    for m in maps:
        for path, digest in m.items():
            key = rel(safe(path))
            need(key not in result or result[key] == digest, "Conflicting authority: " + key)
            need(isinstance(digest, str) and len(digest) == 64, "Invalid SHA: " + key)
            result[key] = digest
    return result


def load_vector(rec, rows, dtype):
    p = safe(rec["path"])
    need(sha(p) == rec["sha256"], "Vector file SHA")
    a = np.load(p, allow_pickle=False)
    need(a.shape == (rows,) and a.dtype == np.dtype(dtype), "Vector shape/dtype")
    need(rec["shape"] == list(a.shape) and rec["dtype"] == str(a.dtype)
         and rec["values_sha256"] == vsha(a), "Vector semantic SHA")
    need(np.isfinite(a).all() and (a >= 0).all(), "Invalid vector values")
    return a


def npz_headers(path, records):
    """Read NPY headers only, including the fit-target header; not its values."""
    with zipfile.ZipFile(safe(path)) as z:
        need(set(z.namelist()) == {k + ".npy" for k in records}, "Prepared exact NPZ keys")
        for name, rec in records.items():
            with z.open(name + ".npy") as f:
                version = np.lib.format.read_magic(f)
                need(version in ((1, 0), (2, 0)), "Unexpected prepared NPY format")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(f)
            need(list(shape) == rec["shape"] and str(dtype) == rec["dtype"], "Prepared header metadata")


def completion_gate():
    # No model, prediction, label or preparation array is read before this gate.
    seal = locked(MAIN / "sealed_manifest.json")
    me = locked(MAIN / "execution.json")
    ev = locked(EVAL / "results.json")
    ee = locked(EVAL / "execution.json")
    need(seal["status"] == "sealed_predictions_without_query_labels"
         and seal["all_fits_complete"] and seal["predictions_sealed"], "Main not sealed")
    need(me["exit_code"] == 0 and me["sealed_manifest_created"], "Main not complete")
    need(ev["status"] == "completed_sealed_prediction_evaluation" and ee["exit_code"] == 0,
         "Evaluation not complete")
    need(seal["new_native_tree_fits"] == 5 and seal["new_NN_fits"] == 2 and seal["new_native_fits"] == 7,
         "Seven native fits required")
    need(seal["query_labels_available"] is False and seal["scores"] is None
         and seal["new_query_metrics_computed"] is False, "Seal/scorer boundary")
    need(ev["sealed_manifest_sha256"] == sha(MAIN / "sealed_manifest.json"), "Evaluation seal authority")
    need(seal["all_query_saved_model_and_state_reload_exact"] is True
         and seal["input_frames_unchanged"] is True, "Full query reload receipt")
    return seal, me, ev, ee


def review_native_metadata(seal, prepared):
    rows, query_rows = seal["fit_rows"], seal["query_rows"]
    need(rows == 272117 and query_rows == 107169, "Fixed row counts")
    details = {}
    for key in ("raw", "light", "log", "unit"):
        meta = read(MAIN / "trees" / key / "metadata.json")
        need(meta == seal["tree_receipts"][key], "Tree receipt binding")
        need(meta["status"] == "completed" and meta["new_native_fits"] == 1
             and meta["saved_model_reload_exact"] and meta["reload_rows"] == query_rows
             and meta["fit_rows"] == rows, "Tree completion/reload")
        need(meta["iterations"] == (3999 if key == "light" else 2999), "Tree iterations")
        need(meta["feature_count"] == (87 if key in ("raw", "light") else 95), "Tree feature count")
        need(meta["query_labels_used"] is False and meta["eval_set_used"] is False
             and meta["early_stopping_used"] is False, "Tree training boundary")
        details[key] = {k: meta[k] for k in ("fit_seconds", "iterations", "feature_count", "saved_model_reload_exact")}
    um = read(MAIN / "trees/unit/fit_metadata.json")
    sm = read(MAIN / "trees/unit137/fit_metadata.json")
    us = read(MAIN / "trees/unit/feature_state.json")
    ss = read(MAIN / "trees/unit137/feature_state.json")
    need(ss["legacy_feature_state"] == us, "Seed137 wrapper/direct seed42 state")
    need(sm["parameters"] == {**um["parameters"], "random_seed": 137}, "Only seed difference")
    for key in ("fit_target_values_sha256", "fit_transformed_target_values_sha256", "fit_date_weights_sha256",
                "fit_sample_weights_sha256", "fit_original_feature_values_sha256",
                "fit_geo_feature_values_sha256", "fit_model_feature_values_sha256"):
        need(sm[key] == um[key], "Unit shared fit identity: " + key)
    need(um["fit_date_weights_sha256"] == prepared["prior_arrays"]["train_sample_weight"]["values_sha256"],
         "Unit/prior saved date weight identity")
    seed = read(MAIN / "trees/unit137/metadata.json")
    need(seed == seal["seed137_receipt"] and seed["new_native_fits"] == 1
         and seed["saved_model_reload_exact"] and seed["raw_reload_exact"] and seed["TWD_reload_exact"]
         and seed["reload_rows"] == query_rows and seed["seed"] == 137, "Seed137 completion/reload")
    details["unit137"] = {"fit_seconds": seed["fit_seconds"], "iterations": sm["actual_iterations"],
                          "feature_count": sm["feature_count"], "saved_model_reload_exact": True}
    need(details["unit137"]["iterations"] == 2999 and details["unit137"]["feature_count"] == 95, "Seed137 capacity")
    nr = read(MAIN / "nn_models/receipt.json")
    need(nr == seal["NN_receipt"] and nr["new_NN_fits"] == 2
         and nr["status"] == "two_native_NN_fits_completed" and nr["serial_execution"]
         and nr["full_query_saved_model_reload_exact"], "NN receipt identity")
    for key in ("N_area", "N_prior"):
        m = read(MAIN / "nn_models" / key / "training.json")
        x = read(MAIN / "nn_models" / key / "execution.json")
        need(m["status"] == "completed" and x["exit_code"] == 0 and m["saved_model_reload_exact"]
             and m["reload_rows"] == query_rows and m["fit_rows"] == rows, "NN complete/full query reload")
        need(m["configuration"]["seed"] == 42 and m["configuration"]["epochs"] == 30
             and [v["epoch"] for v in m["epochs"]] == list(range(1, 31)), "NN fixed seed/epochs")
        need(m["query_labels_available"] is False and m["query_labels_used"] is False, "NN query label boundary")
        kind = "area" if key == "N_area" else "prior"
        need(m["input_sha256"] == sha(prepared[kind + "_npz"])
             and m["numeric_count"] == prepared[kind + "_numeric_width"]
             and m["cardinalities"] == prepared["cardinalities"], "NN preparation binding")
        need(m["extra_sample_weights"] is (key == "N_prior"), "NN loss weights role")
        factors = ("train_area_factor", "query_area_factor") if key == "N_area" else (
            "train_price_factor_million", "query_price_factor_million", "train_sample_weight")
        for name in factors:
            need(m[name + "_sha256"] == prepared[kind + "_arrays"][name]["values_sha256"], "NN factor/weight SHA")
        if key == "N_prior":
            need(m["sample_weights_recomputed_in_trainer"] is False
                 and m["sample_weights_batch_renormalized"] is False
                 and m["epoch_selection_uses_training_metrics"] is False, "Fixed prior loss")
        details[key] = {k: m[k] for k in ("fit_seconds", "numeric_count", "parameter_count", "saved_model_reload_exact")}
    close(sum(details[k]["fit_seconds"] for k in ("raw", "light", "log", "unit", "unit137")),
          seal["fit_seconds"]["trees"], "Tree fit timing sum")
    close(details["N_area"]["fit_seconds"] + details["N_prior"]["fit_seconds"], seal["fit_seconds"]["NN"], "NN fit timing sum")
    return details


def review_preparation(seal, prepared, pure, labels):
    need(prepared == read(prepared["prepared_metadata"])
         and prepared["status"] == "prepared_no_NN_training" and prepared["new_NN_fits"] == 0, "NN prepared receipt")
    need(prepared["source_indices_contract"]["caller_field"] == "round90_offset_data.OffsetData.fit_indices"
         and prepared["chronological_oof"] is False and prepared["query_labels_available"] is False, "NN OOF role")
    need(prepared["input_sha256"]["fit_target"] == labels["fit_y_values_sha256"], "Fit-only label binding")
    for name, role in (("fit_features", "Xfit"), ("query_features", "Xquery"),
                       ("fit_tree_geo", "tree_fit"), ("query_tree_geo", "tree_query")):
        need(prepared["input_sha256"][name] == pure["frames"][role]["frame_sha256"], "Encoder pure-X identity")
    fi = load_vector(seal["arrays"]["fit_indices"], seal["fit_rows"], "int64")
    source_indices = np.load(safe(prepared["source_indices"]), allow_pickle=False)
    exact(fi, source_indices, "NN canonical fit indices")
    need(vsha(fi) == labels["fit_canonical_values_sha256"] == prepared["input_sha256"]["source_indices"], "Fit axis SHA")
    dates = np.load(safe(prepared["saved_date_weights"]), allow_pickle=False)
    need(dates.dtype == np.float32 and dates.shape == (seal["fit_rows"],)
         and np.isfinite(dates).all() and (dates > 0).all(), "Saved fit date weights")
    for kind, count in (("area", 8), ("prior", 9)):
        records = prepared[kind + "_arrays"]
        need(len(records) == count and not any("query_y" in n or "query_target" in n for n in records), "No query target NPZ")
        npz_headers(prepared[kind + "_npz"], records)
        # Load only factors/weights and cardinalities, never fit-target or numeric data.
        fields = ("train_area_factor", "query_area_factor") if kind == "area" else (
            "train_price_factor_million", "query_price_factor_million", "train_sample_weight")
        with np.load(safe(prepared[kind + "_npz"]), allow_pickle=False) as z:
            need(z["cardinalities"].tolist() == prepared["cardinalities"], "Saved cardinality values")
            for name in fields:
                a = z[name]
                need(a.dtype == np.float32 and np.isfinite(a).all() and (a > 0).all()
                     and vsha(a) == records[name]["values_sha256"], "Saved factor/weight values")
            if kind == "prior":
                exact(dates, z["train_sample_weight"], "Identical saved prior/tree dates")
    need(prepared["prior_numeric_width"] == 144 and prepared["area_numeric_width"] >= 144
         and len(prepared["categorical_columns"]) == 23, "New encoder widths/categories")
    return {"fit_only_target_binding": True, "canonical_fit_indices_exact": True,
            "NPZ_exact_keys_and_all_headers_verified": True, "NPZ_factor_weight_values_verified": True,
            "fit_target_array_values_loaded": False, "numeric_or_categorical_feature_arrays_loaded": False,
            "chronological_OOF": False, "area_numeric_width": prepared["area_numeric_width"],
            "prior_numeric_width": prepared["prior_numeric_width"], "category_count": 23}


def review_formulas(seal):
    n = seal["query_rows"]
    native = {k: load_vector(seal["native_predictions"][k], n, "float64") for k in NATIVES}
    branch = {k: {layer: load_vector(rec, n, "float64") for layer, rec in seal["branches"][k].items()} for k in BRANCHES}
    full = {k: load_vector(seal["predictions"][k], n, "float64") for k in ("v22", "v24")}
    expected = {"N": .75*(.5*native["N_area"]+.5*native["N_prior"])+.25*native["light"],
                "H": .75*(.5*native["raw"]+.5*native["log"])+.25*native["light"],
                "U22": .75*native["unit42"]+.25*native["light"],
                "U24": .5*native["unit42"]+.5*native["unit137"]}
    for key in BRANCHES:
        exact(branch[key]["base"], expected[key], "Fixed base formula: " + key)
    for version, unit in (("v22", "U22"), ("v24", "U24")):
        exact(full[version], np.median(np.column_stack([branch[k]["spatial"] for k in ("N", "H", unit)]), axis=1), "Full median: " + version)
    rs = read(MAIN / "reference_spatial/results.json")
    need(rs == seal["reference_spatial_receipt"] and rs["new_reference_fits"] == 1
         and rs["new_spatial_state_builds"] == 1 and rs["all_query_reload_exact"]
         and rs["fit_state_unchanged_during_all_predictions"]
         and rs["reference_predict_calls"] == rs["spatial_predict_calls"] == 8, "Fresh R/S completion receipt")
    for key in BRANCHES:
        need(rs["reports"][key]["all_query_reload_exact"] and rs["reports"][key]["spatial_fallback_exact"], "Branch reload/fallback receipt")
    ref = read(MAIN / "reference_spatial/reference_fit_report.json")
    sp = read(MAIN / "reference_spatial/spatial_summary.json")
    need(ref["fitting_rows"] == sp["fit_rows"] == seal["fit_rows"]
         and ref["validation_targets_used_for_references"] is False and sp["query_used_in_build"] is False, "R/S fit-only receipt")
    need(sp["fit_X_sha256"] == seal["frames"]["Xfit"]["frame_sha256"]
         and sp["fit_geo_sha256"] == seal["frames"]["spatial_fit"]["frame_sha256"], "Spatial new fit identity")
    return {"native_vectors": 7, "branch_vectors": 12, "complete_vectors": 2,
            "four_bases_exact": True, "two_medians_exact": True,
            "N_H_shared_without_separate_reweighting": True, "U24_has_no_LGB": True,
            "R_S_saved_full_query_reload_receipts_verified": True,
            "R_S_repredicted_by_this_audit": False}


def review_metrics(ev):
    metrics = ev["metrics"]
    expected = {"v22", "v24"} | {"native/"+k for k in NATIVES} | {b+"/"+l for b in BRANCHES for l in ("base", "reference", "spatial")}
    need(set(metrics) == expected, "All 21 vector metrics")
    need(ev["parameters"]["fixed_group_weights"] == WEIGHTS and ev["parameters"]["stage_index"] == 0
         and ev["parameters"]["seed"] == 20260925 and ev["parameters"]["replicates"] == 2000, "Fixed evaluator parameters")
    for name, m in metrics.items():
        need(m["rows"] == 107169 and sum(x["rows"] for x in m["strata"].values()) == 107169
             and sum(x["rows"] for x in m["price_bands"]) == 107169, "Metric row partitions")
        weighted = sum(WEIGHTS[k]*m["strata"][k]["mae"] for k in WEIGHTS if WEIGHTS[k] > 0)
        close(weighted, m["weighted_mae"], "Saved aggregate weighted arithmetic")
    delta = {k: metrics["v24"][k]-metrics["v22"][k] for k in ("overall_mae", "weighted_mae")}
    need(ev["comparison"]["both_point_metrics_improve"] == all(x < 0 for x in delta.values()), "Point interpretation")
    for key, value in delta.items():
        need(ev["comparison"][key]["delta"] == value, "Saved metric delta")
    adverse = []
    for section in ("strata", "forecast_quarters", "months", "cities", "relative_forecast_months"):
        for key, a in metrics["v22"][section].items():
            b = metrics["v24"][section][key]
            need(a["rows"] == b["rows"], "Paired group supports")
            if a["rows"] and (section == "forecast_quarters" or a["rows"] >= 500) and b["mae"] > 1.05*a["mae"]:
                adverse.append((section, key))
    for a, b in zip(metrics["v22"]["price_bands"], metrics["v24"]["price_bands"]):
        need(a["band"] == b["band"] and a["rows"] == b["rows"], "Price group supports")
        if a["rows"] >= 500 and b["mae"] > 1.05*a["mae"]:
            adverse.append(("price_bands", a["band"]))
    declared = [(x["section"], x.get("group", x.get("band"))) for x in ev["comparison"]["quarter_or_large_group_deterioration_above_5pct"]]
    need(adverse == declared, "All predefined adverse groups retained")
    intervals = []
    paired = ev["paired"]
    need(paired["control"] == "v22" and paired["candidate"] == "v24" and paired["stage_index"] == 0
         and paired["seed"] == 20260925 and paired["replicates"] == 2000, "Paired schema")
    for index, name in enumerate(("project", "month")):
        p = paired["methods"][name]
        need(p["family_index"] == index, "Paired family index")
        a = np.load(EVAL / (name + "_replicates.npy"), allow_pickle=False)
        need(a.dtype == np.float64 and a.shape == (2000, 4), "Stored replicate shape")
        for metric, col in (("overall", 0), ("weighted", 2)):
            diffs = a[:, col+1]-a[:, col]
            finite = diffs[np.isfinite(diffs)]
            summary = p["summary"][metric]
            need(summary["valid_replicates"] == len(finite) and summary["invalid_replicates"] == 2000-len(finite), "Invalid replicate preservation")
            bounds = np.quantile(finite, [.025, .975], method="linear").tolist() if len(finite) else [None, None]
            need(bounds == [summary["percentile_2_5"], summary["percentile_97_5"]], "Stored replicate quantile endpoints")
            close(summary["control_mae"], metrics["v22"][metric+"_mae"], "Paired control point")
            close(summary["candidate_mae"], metrics["v24"][metric+"_mae"], "Paired candidate point")
            intervals.append({"method": name, "metric": metric, "lower": bounds[0], "upper": bounds[1],
                              "point_delta": summary["point_delta"], "valid": len(finite), "invalid": 2000-len(finite)})
    return {"vectors_checked": 21, "aggregate_weight_arithmetic_verified": True, "delta_v24_minus_v22": delta,
            "both_point_metrics_improve": all(x < 0 for x in delta.values()),
            "all_predefined_adverse_groups_retained": True, "adverse_groups": declared,
            "four_intervals_recomputed_from_saved_replicates": intervals,
            "new_MAE_or_bootstrap_computed": False}


def report_text(result, seal, ev):
    lines = ["# R97 完成後獨立審查", "", "七個原生模型與獨立評分均完成後才執行本稽核。來源／保存產物與固定公式通過；本輪不自動採用 v24，唯一選定提交維持 v22，Private 未知。", "",
             "本次只讀雜湊、模型 metadata、保存向量、prepared NPZ headers／factors及已存指標；沒有反序列化 raw 價格表或 CBM/PT、沒有 query y、模型 forward、重新 fit、逐列 MAE或新 bootstrap。讀模型／價格來源檔位元組核 SHA 不等於載模型或擷取價格。全 query 模型／state 重載是核主流程已保存收據，不是本稽核再次推論。", "",
             "## 來源與執行", "", f"實核 {result['unique_hashed_files']} 個唯一來源／產物 SHA，main 保護清單與另指定 9 檔全部原樣。凍結計畫、main request、evaluation request、純 X 與隔離 fit-label 身分一致。完整四 base／兩 median 逐位相同；N/H由同一保存分支共享，U24沒有混入LGB。", "",
             f"模型 fit 呼叫收據合计 {seal['fit_seconds']['native_total']:.3f} 秒（樹 {seal['fit_seconds']['trees']:.3f}、NN {seal['fit_seconds']['NN']:.3f}）；main execution {result['timings']['main_execution_seconds']:.3f} 秒、evaluation execution {result['timings']['evaluation_execution_seconds']:.3f} 秒。部分樹 fit 呼叫含helper前後檢查，不是純GPU計時；內外層不可重複相加。", "",
             "五樹保留原2999／LGB3999配置，兩NN均seed42／30epochs；各支全107,169query保存重載收據通過。新fit encoder／OOF／date／R/S身分核對，group OOF不是時間OOF；canonical fit index與raw fit-label mapping分工正確。N_area ramps寬度由新fit決定，N_prior固定144。未讀prepared fit target數值或大numeric／categorical矩陣。", "",
             "## 完整模型與分群", "", "差額是v24−v22，負數較好。固定加權仍為既定新舊案組成，不是Private分數。", "", "|指標|v22|v24|差額|", "|---|---:|---:|---:|"]
    m = ev["metrics"]
    for key in ("overall_mae", "weighted_mae"):
        lines.append(f"|{key}|{m['v22'][key]:,.2f}|{m['v24'][key]:,.2f}|{m['v24'][key]-m['v22'][key]:+,.2f}|")
    lines += ["", "|分群／季度|筆數|v24−v22 MAE|", "|---|---:|---:|"]
    for section in ("strata", "forecast_quarters"):
        for key, a in m["v22"][section].items():
            b = m["v24"][section][key]
            value = f"{b['mae']-a['mae']:+,.2f}" if a["rows"] else "—"
            lines.append(f"|{key}|{a['rows']:,}|{value}|")
    lines += ["", "|真價帶|筆數|v24−v22 MAE|", "|---|---:|---:|"]
    for a, b in zip(m["v22"]["price_bands"], m["v24"]["price_bands"]):
        value = f"{b['mae']-a['mae']:+,.2f}" if a["rows"] else "—"
        lines.append(f"|{a['band']}|{a['rows']:,}|{value}|")
    lines += ["", "|unit比較層|unit42／U22 MAE|unit137／U24 MAE|差額|", "|---|---:|---:|---:|"]
    for label, a, b in [("native", "native/unit42", "native/unit137")] + [(l, "U22/"+l, "U24/"+l) for l in ("base", "reference", "spatial")]:
        lines.append(f"|{label}|{m[a]['overall_mae']:,.2f}|{m[b]['overall_mae']:,.2f}|{m[b]['overall_mae']-m[a]['overall_mae']:+,.2f}|")
    lines += ["", "native行比較兩個不同seed單模；base起才比較原U22含LGB與U24兩seed均值，不能將這些差額當成只改seed的因果效果，更不能相加取代完整median。其他21向量的全部指標保存於scorer JSON。", "",
              f"同時整體／固定加權改善：{result['metrics_audit']['both_point_metrics_improve']}。事前定義的季度或至少500列群退步超過5%共 {len(result['metrics_audit']['adverse_groups'])} 項，全部保留如下（有利群也保留於上表及完整評分）。", ""]
    adverse = ev["comparison"]["quarter_or_large_group_deterioration_above_5pct"]
    if adverse:
        lines += ["|類別|群|列數|差額|相對變化|", "|---|---|---:|---:|---:|"]
        for x in adverse:
            lines.append(f"|{x['section']}|{x.get('group',x.get('band'))}|{x['rows']:,}|{x['delta']:+,.2f}|{x['relative_change_pct']:+.2f}%|")
    else:
        lines += ["沒有超過該固定門檻的群；不等於所有群皆改善。"]
    lines += ["", "## 保存抽樣區間", "", "|抽法|指標|點差额|95%區間|有效／無效|", "|---|---|---:|---|---:|"]
    for x in result["metrics_audit"]["four_intervals_recomputed_from_saved_replicates"]:
        bounds = "無有效抽樣" if x["lower"] is None else f"[{x['lower']:+,.2f}, {x['upper']:+,.2f}]"
        lines.append(f"|{x['method']}|{x['metric']}|{x['point_delta']:+,.2f}|{bounds}|{x['valid']}/{x['invalid']}|")
    lines += ["", "全部4區間由保存的兩份(2000,4)replicates直接重算端點；未重新抽樣。固定seed20260925、stage index0、family0/1。缺正權重層的weighted抽樣不重抽。兩種區間相依，條件是保存模型與重複使用資料，不是Private勝率、盲測或已校正多年選型的檢定。", "",
              "## 三個既有年度的上下文", "", "以下直接讀原R79/R83已存完整MAE，沒有重跑舊模型。", "", "|舊窗口|v24−v22整體|v24−v22加權|", "|---|---:|---:|"]
    for x in result["historical_context"]:
        lines.append(f"|{x['stage']}|{x['overall_delta']:+,.2f}|{x['weighted_delta']:+,.2f}|")
    lines += ["", "新窗口與既有年度重疊，配方也由既有研究選出；錯開起點可補充時間穩定性證據，不能消除累積選型偏差。原始test／valid沒有參與這個fit；本次不是乾淨環境從零重訓正式提交，也不宣稱Private結果。真價带與各模型自己的誤差尾部只能作事後診斷，不用於未知test分流。", "",
              "[逐項收據](../outputs/round97/independent_review.json) · [完整評估](../outputs/round97/evaluation/results.json) · [main封存](../outputs/round97/sealed_manifest.json) · [原計畫](round97_plan.md) · [main前置](round97_preflight_review.md) · [scorer前置](round97_evaluator_preflight_review.md)", ""]
    return "\n".join(lines)


def run():
    seal, main_execution, ev, eval_execution = completion_gate()
    need(not AUDIT.exists() and not RECEIPT.exists() and not REPORT.exists(), "Preserve existing audit; no overwrite/retry")
    AUDIT.mkdir()
    tick, started, code, failure = time.monotonic(), datetime.now(timezone.utc).isoformat(), 1, None
    try:
        request = read(ROOT / "outputs/round97_launcher_20260929/research_request.json")
        er = locked(ROOT / "outputs/round97_launcher_20260929/evaluation_request.json")
        need(sha(ROOT / "outputs/round97_launcher_20260929/research_request.json") == seal["request_sha256"], "Main request SHA")
        need(er["parameters"] == ev["parameters"] and er["source_sha256"] == ev["source_sha256"]
             and er["status"] == "evaluation_frozen_before_query_score", "Separate scoring freeze")
        need(seal["plan_sha256"] == er["plan_sha256"] == CORE["reports/round97_plan.md"], "Fixed plan identity")
        need(all(request["frozen_source_and_plan"].get(k) == v for k, v in seal["source_sha256"].items()), "Main source freeze")
        need(seal["protected_before"] == seal["protected_after"], "Main protected before/after")
        protected9 = read(PROTECTED9)
        need(len(protected9) == 9, "Authoritative nine-file protection map")
        prepared = seal["NN_preparation"]
        pure = read(seal["preparation_manifest"]["path"])
        labels = read(seal["fit_label_manifest"]["path"])
        histories = {n: locked(ROOT / f"outputs/round{n}/results.json") for n in (79, 83)}
        maps = [CORE, request["frozen_source_and_plan"], er["source_sha256"], seal["source_sha256"],
                seal["input_sha256"], seal["artifact_sha256"], seal["protected_before"], protected9,
                ev["source_sha256"], ev["input_sha256"], ev["artifact_sha256"], prepared["artifact_sha256"]]
        evidence = [MAIN/"sealed_manifest.json", MAIN/"execution.json", EVAL/"results.json", EVAL/"execution.json",
                    EVAL/"prelabel_gate.json", ROOT/"outputs/round97_launcher_20260929/research_request.json",
                    ROOT/"outputs/round97_launcher_20260929/evaluation_request.json", PROTECTED9,
                    ROOT/"outputs/round79/results.json", ROOT/"outputs/round83/results.json", Path(__file__)]
        hashes = merged(*maps, {rel(p): sha(p) for p in evidence})
        for path, digest in hashes.items():
            need(sha(path) == digest, "Changed bound artifact: " + path)
        gate = locked(EVAL/"prelabel_gate.json")
        need(gate["status"] == "verified_without_price_deserialization"
             and gate["sealed_manifest_sha256"] == ev["sealed_manifest_sha256"], "Pre-label gate authority")
        need(ev["model_fit_load_predict_calls"] == 0 and ev["GPU_calls"] == 0 and ev["test_files_read"] == 0
             and ev["query_labels_used_in_training_or_predictions"] is False, "Scorer role")
        prep_audit = review_preparation(seal, prepared, pure, labels)
        native_audit = review_native_metadata(seal, prepared)
        formula_audit = review_formulas(seal)
        metric_audit = review_metrics(ev)
        context = []
        for stage in ("development", "confirmation", "auxiliary"):
            # Research keys predate release labels: R79 became v22; R83 became v24.
            a = histories[79]["stages"][stage]["metrics"]["R79"]
            b = histories[83]["stages"][stage]["metrics"]["R83"]
            need(histories[83]["stages"][stage]["metrics"]["v22"] == a, "Historical release/control identity")
            context.append({"stage": stage, "overall_delta": b["overall_mae"]-a["overall_mae"],
                            "weighted_delta": b["weighted_mae"]-a["weighted_mae"]})
        result = {"status": "passed_independent_stored_artifact_review", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "round": 97, "unique_hashed_files": len(hashes), "protected_nine_files_verified": 9,
                  "protected_main_files_verified": len(seal["protected_before"]), "new_fits": 0,
                  "native_models_deserialized": 0, "model_predict_calls": 0, "new_score_calls": 0,
                  "new_bootstrap_calls": 0, "query_labels_loaded": False, "test_data_loaded": False,
                  "preparation_audit": prep_audit, "native_metadata_audit": native_audit,
                  "formula_audit": formula_audit, "metrics_audit": metric_audit,
                  "historical_context": context, "selected_submission": "v22", "Private_score": None,
                  "timings": {"main_execution_seconds": main_execution["elapsed_seconds"],
                              "evaluation_execution_seconds": eval_execution["elapsed_seconds"],
                              "native_fit_receipt_seconds": seal["fit_seconds"]},
                  "evidence_sha256": {rel(p): sha(p) for p in evidence},
                  "all_verified_artifact_sha256": hashes,
                  "limitations": ["No model forward or label-based metric recomputation in this audit.",
                                  "Full-query model reload is verified through frozen completed-run receipts.",
                                  "Overlapping previously used historical data are not a blind holdout.",
                                  "Bootstrap does not remove adaptive selection, retraining or future-shift uncertainty.",
                                  "Not clean-environment end-to-end retraining of a formal test submission."]}
        result["elapsed_seconds_before_write"] = time.monotonic()-tick
        write(RECEIPT, result)
        text = report_text(result, seal, ev)
        with REPORT.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            need((REPORT.parent/target).resolve().is_file(), "Broken report link: " + target)
        code = 0
        print(json.dumps({"status": result["status"], "unique_hashed_files": len(hashes),
                          "report_sha256": sha(REPORT), "receipt_sha256": sha(RECEIPT)}, ensure_ascii=False))
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        write(AUDIT/"execution.json", {"exit_code": code, "failure": failure, "started_at_utc": started,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-tick,
              "source_sha256": sha(__file__), "query_labels_loaded": False, "new_fits": 0,
              "new_model_predict_score_or_bootstrap_calls": 0, "automatic_retry": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    run()
