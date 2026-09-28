"""R100 isolated evaluation: all six NN fits and three windows seal before labels.

No model loading, training, or native inference. C is a matched-loss-row
mechanism control; T alone is the research candidate. Synthetic checks use
artificial arrays only.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import v3_cluster_stability as methods
import round100_contract as contract

ROOT = Path(__file__).resolve().parents[1]
PLAN = "reports/round100_plan.md"
PLAN_SHA = "bbb29bcc4fb83eaa3fdac9f15ade8221ced3437abacb04e891038ffb3462e107"
STAGES = ("development", "confirmation", "auxiliary")
SPLITS = {"development": ("2022-08-11", "2023-08-10", 81234),
          "confirmation": ("2023-08-11", "2024-08-10", 133697),
          "auxiliary": ("2021-08-11", "2022-08-10", 105511)}
FIT_ROWS = {"development": 240316, "confirmation": 321550, "auxiliary": 134805}
WEIGHTS = {"seen": 0.36617593949822613, "new_nonmissing": 0.6338240605017739, "missing": 0.0}
RAW = "data_cache/train.pkl"
FEATURES = "data_cache/train_features.pkl"
CANONICAL = "outputs/round6/geography/development/train_source_cache_indices.npy"
AUTHORITY = "outputs/round79/plan.json"
AUTHORITY_SHA = "7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f"
CONTROL_RESULTS = {"outputs/round79/results.json": "7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6",
                   "outputs/round83/results.json": "8db07138c15ef2725d848c8b0ca65a6095bcc183b8550611377b3e165c59583b",
                   "outputs/round63/results.json": "83779882a45379a955aa88197e03dd1d566262c1bc75bbeb0935d735bd387506"}
BANDS = (("<10m", 0., 1e7), ("10-20m", 1e7, 2e7), ("20-50m", 2e7, 5e7),
         ("50-100m", 5e7, 1e8), (">=100m", 1e8, np.inf))
CAUTION = ("Adaptive research after R99 on repeatedly used historical windows; not blind validation. "
           "Bootstrap is conditional on saved models, not Private improvement probability. "
           "C isolates the loss-row mask; T also excludes date-unidentifiable prior references. "
           "Neither is an automatically selected submission; v22 remains selected.")

def need(ok, msg):
    if not ok:
        raise ValueError(msg)

def now():
    return datetime.now(timezone.utc).isoformat()

def safe(name):
    p = Path(name)
    p = ((ROOT / p) if not p.is_absolute() else p).resolve()
    need(p.is_relative_to(ROOT), "Path outside workspace: " + str(name))
    return p

def rel(path):
    return safe(path).relative_to(ROOT).as_posix()

def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def read(path):
    return json.loads(safe(path).read_text(encoding="utf-8-sig"))

def write(path, value):
    p = safe(path)
    with p.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    with p.with_suffix(".sha256").open("x", encoding="ascii") as f:
        f.write(sha(p) + "\n")

def exact(a, b, name):
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name)

def array_record(path, a):
    return {"path": rel(path), "sha256": sha(path), "dtype": str(a.dtype),
            "shape": list(a.shape), "values_sha256": values_sha(a)}

def save_array(path, a):
    p = safe(path)
    need(isinstance(a, np.ndarray) and not a.dtype.hasobject, "Only numeric arrays")
    with p.open("xb") as f:
        np.save(f, a, allow_pickle=False)
    exact(a, np.load(p, allow_pickle=False), "Saved array differs: " + rel(p))
    return array_record(p, a)

def load_array(record):
    p = safe(record["path"])
    need(sha(p) == record["sha256"], "Array file identity differs: " + rel(p))
    a = np.load(p, allow_pickle=False)
    need(not a.dtype.hasobject and str(a.dtype) == record["dtype"] and list(a.shape) == record["shape"], "Array schema differs")
    need(values_sha(a) == record["values_sha256"], "Array values differ")
    return a

def source_hashes(*entries):
    pending = [safe(p) for p in (entries or (__file__,))]
    found = set()
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig"))):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                q = ROOT / "src" / (name.split(".")[0] + ".py")
                if q.is_file() and q not in found:
                    pending.append(q)
    return {rel(p): sha(p) for p in sorted(found)}

def verify_hashes(mapping, exclude=()):
    for name, expected in mapping.items():
        if name not in exclude:
            need(sha(name) == expected, "Immutable source changed: " + name)

def protected_hashes():
    names = read("outputs/github_publish_r97_20260929/protected_before.json")
    return {name: sha(name) for name in names}

def parameters():
    return contract.evaluation_parameters()


def check_header(seal):
    need(seal.get("schema_version") == 1 and seal.get("status") == "all_temporal_prior_predictions_sealed", "All predictions must be sealed")
    need(seal.get("new_NN_fits") == 6 and seal.get("new_native_tree_fits") == 0, "Exactly six NN fits required")
    need(seal.get("query_scores_computed") is False and seal.get("all_query_reload_exact") is True, "Unscored reload gate failed")
    need(seal.get("plan_sha256") == PLAN_SHA and set(seal.get("stages", {})) == set(STAGES), "Stage/plan differs")
    need(seal.get("protected_before") == seal.get("protected_after") and len(seal.get("protected_before", {})) == 9, "Protected artifacts differ")


def temporal_masks(days, start, end):
    return (np.flatnonzero(np.isfinite(days) & (days < day(start))).astype(np.int64),
            np.flatnonzero(np.isfinite(days) & (days >= day(start)) & (days <= day(end))).astype(np.int64))


def check_temporal(stage, fi, qi, raw_qi, days, mapping):
    start, end, n = SPLITS[stage]
    index(fi, len(days), "canonical fit"); index(qi, len(days), "canonical query")
    need(len(fi) == FIT_ROWS[stage] and len(qi) == n, "Fixed stage rows differ")
    expected_fit, expected_query = temporal_masks(days, start, end)
    exact(fi, expected_fit, "Fit date mask differs")
    exact(qi, expected_query, "Query date mask differs")
    need(not np.intersect1d(fi, qi).size, "Fit/query overlap")
    exact(raw_qi, mapping[qi], "Canonical/raw query mapping differs")


def check_formulas(predictions, branches, saved):
    # Identical floating-point operation order to the frozen scientific plan.
    for name in ("v22", "C", "T"):
        base = .75 * (.5 * saved["area"] + .5 * branches[name]["native"]) + .25 * saved["lgb"]
        exact(base, branches[name]["base"], "Prior/base formula differs: " + name)
        total = np.median(np.vstack([branches[name]["spatial"], saved["H"], saved["U"]]), axis=0)
        exact(total, predictions[name], "Final three-branch median differs: " + name)


def research_gate(stages):
    all_overall = all(stages[s]["metrics"]["T"]["overall_mae"] < stages[s]["metrics"]["v22"]["overall_mae"] for s in STAGES)
    all_weighted = all(stages[s]["metrics"]["T"]["weighted_mae"] < stages[s]["metrics"]["v22"]["weighted_mae"] for s in STAGES)
    latest = stages["confirmation"]["metrics"]
    fraction = (latest["v22"]["overall_mae"] - latest["T"]["overall_mae"]) / latest["v22"]["overall_mae"]
    no_adverse = all(not stages[s]["comparison"]["quarter_or_large_group_deterioration_above_5pct"] for s in STAGES)
    parts = {"all_three_overall_improve": all_overall, "all_three_weighted_improve": all_weighted,
             "confirmation_at_least_half_percent": fraction >= .005,
             "no_quarter_or_large_group_deterioration_above_5pct": no_adverse}
    return {"components": parts, "adoption_research_gate_passed": all(parts.values()),
            "confirmation_fractional_improvement": fraction,
            "C_is_mechanism_control_not_alternative_candidate": True, "automatic_adoption": False,
            "new_CSV": False, "selected_submission": "v22", "Private_score": None}


def day(date):
    return (pd.Timestamp(date) - pd.Timestamp("2000-01-01")).days

def index(a, rows, name, sorted_required=True):
    need(a.dtype == np.int64 and a.ndim == 1 and len(a) > 0 and (a >= 0).all() and (a < rows).all(), "Index schema/bounds: " + name)
    need(len(np.unique(a)) == len(a), "Repeated indices: " + name)
    if sorted_required:
        need((np.diff(a) > 0).all(), "Unsorted indices: " + name)

def prices(a, shape, name):
    need(a.dtype == np.float64 and a.shape == shape and np.isfinite(a).all() and (a >= 0).all(), "Invalid total prices: " + name)

def select_query_y(raw, raw_indices, expected_raw_rows=455784):
    need(isinstance(raw, pd.DataFrame) and len(raw) == expected_raw_rows and raw.index.equals(pd.RangeIndex(expected_raw_rows)), "Raw price row axis differs")
    index(raw_indices, len(raw), "raw query indices")
    y = raw.iloc[raw_indices]["總價元"].to_numpy(dtype=np.float64, copy=True)
    need(np.isfinite(y).all() and (y > 0).all(), "Invalid query target")
    return y

def descriptive_metrics(y, prediction, fit, query, start):
    need(y.shape == prediction.shape == (len(query),), "Scoring alignment differs")
    need(np.isfinite(y).all() and (y > 0).all(), "Invalid scoring targets")
    prices(prediction, y.shape, "score")
    result = methods.original_score(y, prediction, fit, query, start, WEIGHTS)
    residual = prediction - y
    error = np.abs(residual)
    total = float(error.sum())
    bands = []
    for name, lo, hi in BANDS:
        mask = (y >= lo) & (y < hi)
        count = int(mask.sum())
        bands.append({"band": name, "rows": count, "mae": float(error[mask].mean()) if count else None,
                      "mean_signed_error_prediction_minus_truth": float(residual[mask].mean()) if count else None,
                      "absolute_error_sum": float(error[mask].sum()),
                      "absolute_error_share": float(error[mask].sum() / total) if total else None,
                      "underprediction_fraction": float((residual[mask] < 0).mean()) if count else None})
    need(sum(b["rows"] for b in bands) == len(y), "Price bands not partition")
    descending = np.sort(error)[::-1]
    top = min(10, len(error)); top_fraction = max(1, int(np.ceil(.01 * len(error))))
    result["price_bands"] = bands
    result["tails"] = {"mean_signed_error_prediction_minus_truth": float(residual.mean()),
                       "underprediction_fraction": float((residual < 0).mean()), "absolute_error_sum": total,
                       "maximum_absolute_error": float(error.max()),
                       "absolute_error_quantiles": {str(q): float(np.quantile(error, q, method="linear")) for q in (.5, .9, .95, .99)},
                       "top10_error_rows": top, "top10_absolute_error_sum": float(descending[:top].sum()),
                       "top10_error_share": float(descending[:top].sum() / total) if total else None,
                       "top1pct_error_rows": top_fraction, "top1pct_absolute_error_sum": float(descending[:top_fraction].sum()),
                       "top1pct_error_share": float(descending[:top_fraction].sum() / total) if total else None,
                       "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules"}
    months = methods.relative_months(query.trade_day_index, start)
    result["relative_forecast_months"] = {str(i): {"rows": int((months == i).sum()),
        "mae": float(error[months == i].mean()) if (months == i).any() else None} for i in range(12)}
    return result

def paired(y, predictions, fit, query, query_indices, stage, control_name):
    start = SPLITS[stage][0]
    stage_index = STAGES.index(stage)
    labels = methods.group_labels(fit, query)
    errors = np.abs(y[:, None] - np.column_stack([predictions[k] for k in (control_name, "T")]))
    point = methods.point_values(errors, labels, WEIGHTS)
    keys, fallback = methods.cluster_keys(query, query_indices)
    months = methods.relative_months(query.trade_day_index, start)
    result = {"control": control_name, "candidate": "T", "difference": "T minus " + control_name, "seed": 20260925,
              "replicates": 2000, "stage_index": stage_index, "fallback": fallback, "methods": {}, "caution": CAUTION}
    arrays = {}
    for family_index, (name, fn, groups) in enumerate((("project", methods.bootstrap_projects, keys), ("month", methods.bootstrap_months, months))):
        samples, support = fn(errors, labels, groups, WEIGHTS,
            np.random.default_rng(np.random.SeedSequence([20260925, stage_index, family_index])), 2000)
        summary = {}
        for metric, item in methods.summarize(samples, point).items():
            translated = {k: v for k, v in item.items() if k not in ("v2_mae", "v3_mae", "point_delta_v3_minus_v2")}
            translated.update(control_mae=item["v2_mae"], candidate_mae=item["v3_mae"], point_delta=item["point_delta_v3_minus_v2"])
            summary[metric] = translated
        result["methods"][name] = {"family_index": family_index, "summary": summary, "support": support}
        arrays[name] = samples
    return result, arrays

def comparisons(metrics, control_name="v22", candidate_name="T"):
    control, candidate = metrics[control_name], metrics[candidate_name]
    def difference(a, b):
        return {"control_mae": a, "candidate_mae": b, "delta": b - a,
                "relative_change_pct": 100 * (b - a) / a if a else None}
    result = {k: difference(control[k], candidate[k]) for k in ("overall_mae", "weighted_mae")}
    adverse = []
    for section in ("strata", "forecast_quarters", "months", "cities", "relative_forecast_months"):
        result[section] = {}
        for key, a in control[section].items():
            b = candidate[section][key]
            need(a["rows"] == b["rows"], "Group support differs")
            if a["rows"]:
                item = {"rows": a["rows"], **difference(a["mae"], b["mae"])}
                result[section][key] = item
                if (section == "forecast_quarters" or a["rows"] >= 500) and b["mae"] > 1.05 * a["mae"]:
                    adverse.append({"section": section, "group": key, **item})
    result["price_bands"] = []
    for a, b in zip(control["price_bands"], candidate["price_bands"]):
        need(a["band"] == b["band"] and a["rows"] == b["rows"], "Price support differs")
        item = {"band": a["band"], "rows": a["rows"], **(difference(a["mae"], b["mae"]) if a["rows"] else {})}
        result["price_bands"].append(item)
        if a["rows"] >= 500 and b["mae"] > 1.05 * a["mae"]:
            adverse.append({"section": "price_bands", **item})
    result["both_point_metrics_improve"] = all(candidate[k] < control[k] for k in ("overall_mae", "weighted_mae"))
    result["quarter_or_large_group_deterioration_above_5pct"] = adverse
    if control_name == "v22" and candidate_name == "T":
        result["stage_research_gate_passed"] = result["both_point_metrics_improve"] and not adverse
    result["control"] = control_name
    result["candidate"] = candidate_name
    result["used_for_research_gate"] = control_name == "v22" and candidate_name == "T"
    return result



def preflight(sealed_path, request_path):
    inputs = {}
    def checked(path, digest):
        name = rel(path)
        need(name not in inputs or inputs[name] == digest, "Conflicting authority: " + name)
        if name not in inputs:
            need(sha(name) == digest, "Immutable input changed: " + name)
        inputs[name] = digest
        return safe(name)
    def document(record):
        return read(checked(record["path"], record["sha256"]))
    def array(record):
        checked(record["path"], record["sha256"])
        return load_array(record)
    def bind_map(mapping):
        for name, digest in mapping.items(): checked(name, digest)

    checked(PLAN, PLAN_SHA)
    sources = source_hashes()
    request_record = {"path": rel(request_path), "sha256": sha(request_path)}
    request = document(request_record)
    need(request.get("status") == "evaluation_frozen_before_query_score" and request.get("parameters") == parameters(), "Scorer was not frozen")
    need(request.get("plan_sha256") == PLAN_SHA and request.get("source_sha256") == sources, "Evaluation source or plan changed")
    bind_map(sources)
    seal_record = {"path": rel(sealed_path), "sha256": sha(sealed_path)}
    seal = document(seal_record); check_header(seal)
    need(request["research_request"] == seal["request"], "Evaluation/main request binding differs")
    launch = document(seal["request"])
    need(launch["plan_sha256"] == PLAN_SHA and launch["status"] == "frozen_before_real_prior_or_fit" and launch["approval_for_six_fits"] is True, "Main plan/approval differs")
    bind_map(launch["frozen_source_and_plan"]); bind_map(launch["input_authority_sha256"])
    bind_map(launch["source_sha256"])
    for key in ("source_sha256",):
        bind_map(seal[key])
    for name, digest in seal["source_sha256"].items():
        need(launch["source_sha256"].get(name) == digest, "Main source not frozen")
    need(seal["protected_before"] == request["protected_sha256"] == protected_hashes(), "Protected selection changed")
    main_execution = document({"path": rel(safe(sealed_path).parent / "execution.json"), "sha256": sha(safe(sealed_path).parent / "execution.json")})
    need(main_execution["exit_code"] == 0 and main_execution.get("failure") is None, "Main has not completed")
    labels = document(seal["fit_label_manifest"])
    need(labels["status"] == "fit_only_labels_sealed" and labels["plan_sha256"] == PLAN_SHA, "Fit label preparation differs")
    need(labels["query_statistics_computed"] is False and labels["all_arrays_saved_reload_exact"] is True, "Fit preparation boundary failed")
    bind_map(labels["input_sha256"]); bind_map(labels["source_sha256"])
    audit = document(seal["pure_X_audit"])
    need(audit["status"] == "completed_pure_X_support_audit" and audit["stage_order"] == list(STAGES), "Pure-X eligibility audit incomplete")
    original_plan = document({"path": AUTHORITY, "sha256": AUTHORITY_SHA})
    old = {int(Path(name).parent.name[5:]): document({"path": name, "sha256": digest}) for name, digest in CONTROL_RESULTS.items()}
    for key, path in (("raw_train", RAW), ("features", FEATURES), ("canonical_mapping", CANONICAL)):
        record = seal[key]
        need(record["path"] == path and record["sha256"] == original_plan["input_sha256"][path], "Original source differs: " + key)
        need(labels[key] == record, "Extractor/driver source differs")
        checked(record["path"], record["sha256"])
    need(seal["raw_train"]["rows"] == 455784, "Raw population differs")
    # Pure X and canonical mapping only. Fit y / NPZ / model bytes are never deserialized here.
    mapping = np.load(safe(CANONICAL), allow_pickle=False)
    index(mapping, 455784, "canonical mapping")
    need(len(mapping) == 455255, "Canonical population differs")
    Xraw = pd.read_pickle(safe(FEATURES))
    need(isinstance(Xraw, pd.DataFrame) and len(Xraw) == 455784 and Xraw.index.equals(pd.RangeIndex(455784)), "Pure-X row axis differs")
    need(not {"總價元", "y", "target", "total_price", "編號"}.intersection(Xraw.columns), "Price field in pure X")
    X = Xraw.iloc[mapping].reset_index(drop=True); del Xraw
    days = X.trade_day_index.to_numpy(dtype=np.float64)
    stages = {}
    for stage in STAGES:
        root_stage = seal["stages"][stage]
        st = document(root_stage["stage_manifest"])
        need(st["status"] == "temporal_prior_stage_predictions_sealed" and st["stage"] == stage and st["cutoff"] == SPLITS[stage][0] and st["query_end"] == SPLITS[stage][1], "Stage dates differ")
        need(st["fit_rows"] == FIT_ROWS[stage] and st["query_rows"] == SPLITS[stage][2], "Stage row count differs")
        need(st["plan_sha256"] == PLAN_SHA and st["new_NN_fits"] == 2 and st["new_native_tree_fits"] == 0, "Stage fit contract differs")
        for key in ("query_labels_available", "query_scores_computed", "raw_price_pickle_opened", "other_stage_fit_labels_opened"):
            need(st[key] is False, "Stage target isolation failed: " + key)
        need(st["all_query_reload_exact"] is True and st["original_v22_full_chain_and_diagnostics_exact"] is True
             and st["original_R63_configuration_exact"] is True, "Stage query/config/control replay incomplete")
        need(st["request"] == seal["request"] and st["predictions"] == root_stage["predictions"], "Stage/root binding differs")
        need(st["protected_before"] == st["protected_after"] == seal["protected_before"], "Stage protected change")
        ex = document(root_stage["execution"])
        need(ex["exit_code"] == 0 and ex.get("failure") is None, "Stage execution failed")
        bind_map(st["source_sha256"]); bind_map(st["original_input_sha256"])
        for name, digest in st["source_sha256"].items():
            need(launch["source_sha256"].get(name) == digest, "Stage source not in request")
        fi, qi, rf, rq = (array(st[k]) for k in ("fit_indices", "query_indices", "fit_source_cache_indices", "query_source_cache_indices"))
        check_temporal(stage, fi, qi, rq, days, mapping); exact(rf, mapping[fi], "Raw fit mapping differs")
        lp = labels["stages"][stage]
        for k, values in (("fit_indices", fi), ("fit_source_cache_indices", rf)):
            exact(values, array(lp["arrays"][k]), "Extractor canonical/raw index mismatch")
        checked(lp["arrays"]["fit_y"]["path"], lp["arrays"]["fit_y"]["sha256"])
        eligible = array(st["eligible_positions"])
        index(eligible, len(fi), "eligible positions")
        exact(fi, array(audit["stages"][stage]["arrays"]["fit_indices"]), "Audit canonical indices differ")
        mask = array(audit["stages"][stage]["arrays"]["eligible_mask"])
        need(mask.dtype == np.bool_ and mask.shape == fi.shape, "Eligibility mask shape")
        exact(eligible, np.flatnonzero(mask).astype(np.int64), "Saved pure-X mask differs")
        need(st["eligible_rows"] == len(eligible) == audit["stages"][stage]["eligible_rows"], "Eligible counts differ")
        for name, record in st.get("artifact_sha256", {}).items():
            checked(name, record)
        need(set(st["layers"]) == {"v22", "C", "T"} and set(st["fixed_branches"]) == {"area", "lgb", "H", "U"}, "Layer/fixed branch schema")
        layers = {name: {layer: array(record) for layer, record in value.items()} for name, value in st["layers"].items()}
        fixed = {name: array(record) for name, record in st["fixed_branches"].items()}
        need(set(st["predictions"]) == {"v22", "v24", "C", "T"} and set(st["arms"]) == {"C", "T"}, "Fixed arm/control schema")
        predictions = {name: array(record) for name, record in st["predictions"].items()}
        for arm, values in layers.items():
            need(set(values) == {"native", "base", "reference", "spatial"}, "Four saved layers required")
            for name, vector in values.items(): prices(vector, (len(qi),), arm + "/" + name)
        for name, vector in {**fixed, **predictions}.items(): prices(vector, (len(qi),), name)
        check_formulas(predictions, layers, fixed)
        for name, number in (("v22", 79), ("v24", 83)):
            original_path = f"outputs/round{number}/{stage}/predictions.npy"
            digest = old[number]["stages"][stage]["artifact_sha256"][original_path]
            checked(original_path, digest)
            exact(predictions[name], np.load(safe(original_path), allow_pickle=False), "Original full control differs")
        # Original prior native values have their own historical authority.
        old_native = f"outputs/round63/{stage}/model/predictions_twd.npy"
        checked(old_native, old[63]["stages"][stage]["artifact_sha256"][old_native])
        exact(layers["v22"]["native"], np.load(safe(old_native), allow_pickle=False), "Original prior native differs")
        training_path = f"outputs/round63/{stage}/model/training.json"
        old_training = document({"path": training_path, "sha256": old[63]["stages"][stage]["artifact_sha256"][training_path]})
        check_arms(st, old_training, checked, document, len(eligible), len(qi))
        for arm in ("C", "T"):
            exact(layers[arm]["native"], array(st["arms"][arm]["native_prediction"]), "Trainer/native saved layer differs")
        stages[stage] = {"fit": X.iloc[fi].reset_index(drop=True), "query": X.iloc[qi].reset_index(drop=True),
                         "query_indices": qi, "query_source_cache_indices": rq, "eligible_rows": len(eligible),
                         "predictions": predictions, "layers": layers}
    need(source_hashes() == sources, "Scorer changed during prelabel gate")
    return {"stages": stages, "source_sha256": sources, "input_sha256": inputs, "sealed_manifest": seal_record,
            "evaluation_request": request_record, "raw_train": seal["raw_train"], "protected_before": seal["protected_before"],
            "original_control_metrics": {s: {"v22": old[79]["stages"][s]["metrics"]["R79"],
                                             "v24": old[83]["stages"][s]["metrics"]["R83"]} for s in STAGES}}


def check_arms(stage, old_training, checked, document, fit_rows, query_rows):
    expected_names = {"train_num", "train_cat", "train_y_million", "query_num", "query_cat", "cardinalities",
                      "train_price_factor_million", "query_price_factor_million", "train_sample_weight"}
    arm_arrays = {}
    for arm in ("C", "T"):
        r = stage["arms"][arm]
        for key in ("prepared", "training", "model", "execution"):
            checked(r[key]["path"], r[key]["sha256"])
        need(r["saved_model_reload_exact"] is True and r["reload_rows"] == query_rows, "Arm reload not full query")
        training = document(r["training"]); execution = document(r["execution"])
        need(execution["exit_code"] == 0 and execution.get("failure") is None, "NN process not completed")
        need(training["status"] == "completed" and training["fit_rows"] == fit_rows and training["query_rows"] == query_rows, "NN training counts/status differ")
        need(training["configuration"] == old_training["configuration"], "Immutable R63 training config differs")
        need(training["query_labels_available"] is False and training["query_labels_used"] is False, "NN trained with query labels")
        need(training["saved_model_reload_exact"] is True and training["reload_rows"] == query_rows, "NN reload contract differs")
        need(training["sample_weights_recomputed_in_trainer"] is False and training["sample_weights_batch_renormalized"] is False, "Loss weights changed")
        a = r["all_input_arrays"]
        need(set(a) == expected_names, "NN prepared must have exact nine keys")
        for name, record in a.items():
            need(set(record) == {"dtype", "shape", "values_sha256"}, "NN array identity schema")
            if name.startswith("train_"): need(record["shape"][0] == fit_rows, "Train input row count")
            if name.startswith("query_"): need(record["shape"][0] == query_rows, "Query input row count")
            need(record["dtype"] == ("int64" if name in ("train_cat", "query_cat", "cardinalities") else "float32"), "NN dtype differs")
        need(training["input_sha256"] == r["prepared"]["sha256"], "Trainer NPZ identity differs")
        for name, key in (("train_price_factor_million", "train_price_factor_million_sha256"),
                          ("query_price_factor_million", "query_price_factor_million_sha256"),
                          ("train_sample_weight", "train_sample_weight_sha256")):
            need(training[key] == a[name]["values_sha256"], "Trainer factor/weight identity differs")
        need(a["query_price_factor_million"]["values_sha256"] == old_training["query_price_factor_million_sha256"], "Full-query factor changed")
        need(training["source_sha256"] == old_training["source_sha256"] == contract.TRAINER_SHA, "Trainer source differs")
        checked(contract.TRAINER, training["source_sha256"])
        need(training["parent_trainer_source_sha256"] == old_training["parent_trainer_source_sha256"]
             and training["architecture_source_sha256"] == old_training["architecture_source_sha256"], "Parent architecture identity differs")
        arm_arrays[arm] = a
    for name in expected_names - {"train_price_factor_million"}:
        need(arm_arrays["C"][name] == arm_arrays["T"][name], "Matched arm input differs: " + name)


def render(result):
    lines = ["# R100 時間前置 prior：完整三窗口評估", "",
             "六次 NN 訓練與全部預測封存後才讀取 query 真價。C 只作相同 loss rows 的機制控制；T 是唯一研究候選。", "",
             "|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T weighted−v22|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        r = result["stages"][stage]; m = r["metrics"]
        lines.append(f'|{stage}|{m["v22"]["overall_mae"]:,.2f}|{m["C"]["overall_mae"]:,.2f}|{m["T"]["overall_mae"]:,.2f}|{r["comparison"]["overall_mae"]["delta"]:+,.2f}|{r["comparison_vs_C"]["overall_mae"]["delta"]:+,.2f}|{r["comparison"]["weighted_mae"]["delta"]:+,.2f}|')
    lines += ["", "## 固定配對區間", "", "|窗口|控制|抽樣|指標|點差|95% 區間|有效次數|",
              "|---|---|---|---|---:|---|---:|"]
    for stage in STAGES:
        for control, paired_result in result["stages"][stage]["paired"].items():
            for family, f in paired_result["methods"].items():
                for metric, r in f["summary"].items():
                    lines.append(f'|{stage}|{control}|{family}|{metric}|{r["point_delta"]:+,.2f}|{r["percentile_2_5"]} ～ {r["percentile_97_5"]}|{r["valid_replicates"]}|')
    lines += ["", "## 固定門檻與反證", "", json.dumps(result["research_gate"], ensure_ascii=False), ""]
    for stage in STAGES:
        r = result["stages"][stage]
        lines += [f"### {stage}", "", "T 對 v22 的季度或至少 500 列固定群退步超過 5%：",
                  json.dumps(r["comparison"]["quarter_or_large_group_deterioration_above_5pct"], ensure_ascii=False),
                  "", "T 對 C 的相同反證（機制描述，不另改採用門檻）：",
                  json.dumps(r["comparison_vs_C"]["quarter_or_large_group_deterioration_above_5pct"], ensure_ascii=False), ""]
    lines += [CAUTION, "", "v22 為唯一提交；Private 未知。完整城市、月份、seen/new/missing、真價帶、尾部及所有層級均保存在 results.json。",
              "原始價格 pickle 必須反序列化全表，之後僅依各期封存 raw-cache query 索引取價；未保存逐列真價、未載模型／test，未新增 CSV。", ""]
    return "\n".join(lines)


def run(sealed_path, request_path, out):
    out = safe(out)
    need(not out.exists(), "Evaluation directory exists; preserve evidence")
    out.mkdir(parents=True, exist_ok=False)
    started, tick, code, failure, labels_opened = now(), time.monotonic(), 1, None, False
    try:
        checked = preflight(sealed_path, request_path)
        write(out / "prelabel_gate.json", {"status": "all_six_fits_and_three_stages_verified_before_price_deserialization",
              "checked_stages": list(STAGES), "sealed_manifest": checked["sealed_manifest"],
              "verified_input_count": len(checked["input_sha256"]), "fit_labels_and_prepared_NPZ_hashed_only": True,
              "source_sha256": checked["source_sha256"], "completed_at_utc": now()})
        need(sha(checked["raw_train"]["path"]) == checked["raw_train"]["sha256"], "Raw changed after gate")
        raw = pd.read_pickle(safe(checked["raw_train"]["path"])); labels_opened = True
        stages = {}
        for stage in STAGES:
            a = checked["stages"][stage]
            y = select_query_y(raw, a["query_source_cache_indices"])
            score = lambda p: descriptive_metrics(y, p, a["fit"], a["query"], SPLITS[stage][0])
            metrics = {name: score(p) for name, p in a["predictions"].items()}
            layer_metrics = {arm: {layer: score(p) for layer, p in layers.items()} for arm, layers in a["layers"].items()}
            for name in ("v22", "v24"):
                original = checked["original_control_metrics"][stage][name]
                need({k: metrics[name][k] for k in original} == original, "Full original control score replay differs: " + stage + "/" + name)
            boot, reps = {}, {}
            folder = out / stage; folder.mkdir()
            for control in ("v22", "C"):
                item, samples = paired(y, a["predictions"], a["fit"], a["query"], a["query_indices"], stage, control)
                for f in item["methods"].values():
                    for metric, key in (("overall", "overall_mae"), ("weighted", "weighted_mae")):
                        r = f["summary"][metric]
                        need(abs(r["control_mae"]-metrics[control][key]) <= 1e-6
                             and abs(r["candidate_mae"]-metrics["T"][key]) <= 1e-6, "Paired point differs")
                boot[control] = item
                reps[control] = {family: save_array(folder / ("T_vs_" + control + "_" + family + "_replicates.npy"), samples[family])
                                 for family in ("project", "month")}
            item = {"stage": stage, "cutoff": SPLITS[stage][0], "query_end": SPLITS[stage][1],
                    "fit_rows": len(a["fit"]), "query_rows": len(y), "eligible_rows": a["eligible_rows"],
                    "metrics": metrics, "layer_metrics": layer_metrics,
                    "comparison": comparisons(metrics), "comparison_vs_C": comparisons(metrics, "C"),
                    "comparison_vs_v24": comparisons(metrics, "v24"), "C_vs_v22": comparisons(metrics, "v22", "C"),
                    "paired": boot, "replicate_artifacts": reps, "query_target_values_sha256": values_sha(y),
                    "query_target_values_persisted": False, "group_membership_uses_original_outer_fit": True,
                    "full_original_v22_and_v24_metrics_replay_exact": True}
            write(folder / "results.json", item); stages[stage] = item
            print(json.dumps({"stage": stage, "T_minus_v22": item["comparison"]["overall_mae"]["delta"],
                              "T_minus_C": item["comparison_vs_C"]["overall_mae"]["delta"]}), flush=True)
        del raw
        verify_hashes(checked["input_sha256"])
        need(source_hashes() == checked["source_sha256"] and protected_hashes() == checked["protected_before"], "Source/protected changed")
        result = {"round": 100, "status": "completed_three_window_six_fit_evaluation", "created_at_utc": now(),
                  "parameters": parameters(), "checked_stages": list(STAGES), "stages": stages,
                  "research_gate": research_gate(stages), "sealed_manifest": checked["sealed_manifest"],
                  "evaluation_request": checked["evaluation_request"], "input_sha256": checked["input_sha256"],
                  "source_sha256": checked["source_sha256"], "protected_before": checked["protected_before"],
                  "protected_after": protected_hashes(), "query_labels_used_in_training_or_predictions": False,
                  "raw_pickle_whole_table_deserialized_after_all_seals": True, "fit_labels_and_prepared_NPZ_hashed_only": True,
                  "query_target_values_persisted": False, "query_y_mapping": "raw_train.iloc[canonical_mapping[query_indices]]['總價元']",
                  "C_vs_v22_and_T_vs_v24_additional_bootstrap_calls": 0, "conditional_intervals_reported": 24,
                  "model_fit_load_predict_calls": 0, "new_fits_in_scorer": 0, "test_files_read": 0,
                  "selection_changed": False, "new_CSV": False, "Private_score": None, "fresh_blind_validation": False,
                  "caution": CAUTION, "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
                  "elapsed_seconds": time.monotonic() - tick}
        with (out / "report.md").open("x", encoding="utf8", newline="\n") as f: f.write(render(result))
        result["artifact_sha256"] = {rel(p): sha(p) for p in out.rglob("*") if p.is_file()}
        write(out / "results.json", result); code = 0
        return result
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        write(out / "execution.json", {"exit_code": code, "failure": failure, "started_at_utc": started,
              "completed_at_utc": now(), "elapsed_seconds": time.monotonic() - tick,
              "source_sha256": sha(__file__), "raw_prices_deserialized": labels_opened,
              "new_fits": 0, "native_model_load_predict_calls": 0, "automatic_retry": False})



def self_check():
    tick = time.monotonic(); checks = []
    def mark(name, ok):
        need(bool(ok), "Synthetic check failed: " + name); checks.append(name)
    def rejects(name, fn):
        try: fn()
        except (ValueError, KeyError, AssertionError): checks.append(name)
        else: raise AssertionError("Synthetic accepted: " + name)
    mark("contract_parameters_identical", parameters() == contract.evaluation_parameters())
    header = {"schema_version": 1, "status": "all_temporal_prior_predictions_sealed", "new_NN_fits": 6,
              "new_native_tree_fits": 0, "query_scores_computed": False, "all_query_reload_exact": True,
              "plan_sha256": PLAN_SHA, "stages": dict.fromkeys(STAGES),
              "protected_before": dict.fromkeys(range(9), "same"), "protected_after": dict.fromkeys(range(9), "same")}
    check_header(header); mark("complete_six_fit_header", True)
    for key, value in (("new_NN_fits", 5), ("new_native_tree_fits", 1), ("query_scores_computed", True),
                       ("all_query_reload_exact", False), ("status", "running"), ("plan_sha256", "wrong")):
        rejects("header_reject_" + key, lambda key=key, value=value: check_header({**header, key: value}))
    rejects("missing_stage_rejected", lambda: check_header({**header, "stages": {"development": {}}}))
    a, b = day("2022-08-11"), day("2023-08-10")
    fi, qi = temporal_masks(np.array([a-1., a, b, b+1., np.nan]), "2022-08-11", "2023-08-10")
    mark("cutoff_in_query_and_end_included", np.array_equal(fi, [0]) and np.array_equal(qi, [1, 2]))
    raw = pd.DataFrame({"總價元": [10., 20., 30., 40., 50.]})
    mark("raw_canonical_nonidentity_selection", np.array_equal(select_query_y(raw, np.array([1, 4], dtype=np.int64), 5), [20., 50.]))
    rejects("duplicate_raw_indices", lambda: select_query_y(raw, np.array([1, 1], dtype=np.int64), 5))
    rejects("raw_index_out_of_bounds", lambda: select_query_y(raw, np.array([1, 5], dtype=np.int64), 5))
    saved = {"area": np.array([100., 150., 200.]), "lgb": np.array([80., 160., 220.]),
             "H": np.array([110., 155., 210.]), "U": np.array([115., 165., 200.])}
    layers, predictions = {}, {"v24": np.array([112., 157., 205.])}
    for i, arm in enumerate(("v22", "C", "T")):
        native = np.array([90., 140., 210.]) + i * 5.
        base = .75 * (.5 * saved["area"] + .5 * native) + .25 * saved["lgb"]
        spatial = base + 10.
        layers[arm] = {"native": native, "base": base, "reference": base+5., "spatial": spatial}
        predictions[arm] = np.median(np.vstack([spatial, saved["H"], saved["U"]]), axis=0)
    check_formulas(predictions, layers, saved); mark("fixed_three_arm_base_and_median", True)
    bad = {k: v.copy() for k, v in predictions.items()}; bad["T"][0] += 1.
    rejects("changed_final_rejected", lambda: check_formulas(bad, layers, saved))
    old = {"configuration": {"seed": 42, "epochs": 30, "batch_size": 2048},
           "source_sha256": contract.TRAINER_SHA, "parent_trainer_source_sha256": "parent",
           "architecture_source_sha256": "architecture", "query_price_factor_million_sha256": "qfactor"}
    expected_names = ("train_num", "train_cat", "train_y_million", "query_num", "query_cat", "cardinalities",
                      "train_price_factor_million", "query_price_factor_million", "train_sample_weight")
    records, documents = {}, {}
    for arm in ("C", "T"):
        identities = {}
        for name in expected_names:
            nr = 6 if name.startswith("train") else 3
            shape = [2] if name == "cardinalities" else ([nr, 2] if name.endswith(("_num", "_cat")) else [nr])
            identities[name] = {"dtype": "int64" if name in ("train_cat", "query_cat", "cardinalities") else "float32",
                                "shape": shape, "values_sha256": ("qfactor" if name == "query_price_factor_million" else name)}
        identities["train_price_factor_million"]["values_sha256"] = arm + "_factor"
        records[arm] = {k: {"path": arm+"/"+k, "sha256": arm+"_"+k} for k in ("prepared", "training", "model", "execution")}
        records[arm].update(saved_model_reload_exact=True, reload_rows=3, all_input_arrays=identities)
        documents[arm+"/execution"] = {"exit_code": 0, "failure": None}
        documents[arm+"/training"] = {**old, "status": "completed", "fit_rows": 6, "query_rows": 3,
              "query_labels_available": False, "query_labels_used": False, "saved_model_reload_exact": True,
              "reload_rows": 3, "sample_weights_recomputed_in_trainer": False, "sample_weights_batch_renormalized": False,
              "input_sha256": arm+"_prepared", "train_price_factor_million_sha256": arm+"_factor",
              "train_sample_weight_sha256": "train_sample_weight"}
    reader = lambda r: documents[r["path"]]
    check_arms({"arms": records}, old, lambda p, h: None, reader, 6, 3)
    mark("actual_R63_scalar_source_and_nine_keys_contract", True)
    documents["T/training"]["query_labels_used"] = True
    rejects("NN_query_label_rejected", lambda: check_arms({"arms": records}, old, lambda p, h: None, reader, 6, 3))
    documents["T/training"]["query_labels_used"] = False
    records["T"]["all_input_arrays"]["train_y_million"]["values_sha256"] = "different"
    rejects("matched_loss_labels_required", lambda: check_arms({"arms": records}, old, lambda p, h: None, reader, 6, 3))
    records["T"]["all_input_arrays"]["train_y_million"]["values_sha256"] = "train_y_million"
    records["T"]["all_input_arrays"]["train_sample_weight"]["values_sha256"] = "different"
    rejects("matched_date_weights_required", lambda: check_arms({"arms": records}, old, lambda p, h: None, reader, 6, 3))
    records["T"]["all_input_arrays"]["train_sample_weight"]["values_sha256"] = "train_sample_weight"
    for flag in ("sample_weights_recomputed_in_trainer", "sample_weights_batch_renormalized"):
        documents["T/training"][flag] = True
        rejects("no_" + flag, lambda: check_arms({"arms": records}, old, lambda p, h: None, reader, 6, 3))
        documents["T/training"][flag] = False
    dates = pd.date_range("2022-08-11", periods=12, freq="MS") - pd.Timedelta(days=1)
    q = pd.DataFrame({"trade_day_index": np.array([(d-pd.Timestamp("2000-01-01")).days for d in dates], dtype=float),
                     "project_name": ["seen"]*4+["new"]*4+[None]*4, "address_city": ["city"]*12,
                     "address_district": ["district"]*12, "address_raw": ["road"]*12,
                     "address_road": ["road"]*12, "address_section": ["section"]*12})
    fit = q.iloc[:1].copy()
    y = np.linspace(1e6, 12e6, 12)
    ps = {"v22": y+100., "C": y+99., "T": y+90., "v24": y+95.}
    metrics = {name: descriptive_metrics(y, pred, fit, q, "2022-08-11") for name, pred in ps.items()}
    mark("strata_use_original_fit", metrics["T"]["strata"]["seen"]["rows"] == 4 and metrics["T"]["strata"]["new_nonmissing"]["rows"] == 4)
    mark("all_five_price_bands_and_tails", len(metrics["T"]["price_bands"]) == 5 and "tails" in metrics["T"])
    comp = comparisons(metrics)
    mark("T_primary_comparison", comp["overall_mae"]["delta"] == -10. and comp["used_for_research_gate"])
    mark("C_control_not_candidate_gate", not comparisons(metrics, "v22", "C")["used_for_research_gate"])
    mark("T_vs_v24_point_only_gate", not comparisons(metrics, "v24")["used_for_research_gate"])
    for control in ("v22", "C"):
        boot, arrays = paired(y, ps, fit, q, np.arange(12, dtype=np.int64), "development", control)
        for family, vals in arrays.items():
            valid = np.isfinite(vals[:, 2:]).all(axis=1)
            delta = -10. if control == "v22" else -9.
            mark(control+"_"+family+"_paired_2000", vals.shape == (2000, 4) and np.allclose(vals[valid,1]-vals[valid,0], delta) and np.allclose(vals[valid,3]-vals[valid,2], delta))
    artificial = {stage: {"metrics": metrics, "comparison": comp} for stage in STAGES}
    gate = research_gate(artificial)
    mark("three_window_gate_pass_without_auto_adoption", gate["adoption_research_gate_passed"] and not gate["automatic_adoption"])
    less = json.loads(json.dumps(artificial))
    less["confirmation"]["metrics"]["T"]["overall_mae"] = 99.6
    mark("confirmation_half_percent_gate", not research_gate(less)["components"]["confirmation_at_least_half_percent"])
    less = json.loads(json.dumps(artificial)); less["auxiliary"]["metrics"]["T"]["weighted_mae"] = 101.
    mark("all_weighted_required", not research_gate(less)["adoption_research_gate_passed"])
    bad_metrics = json.loads(json.dumps(metrics)); bad_metrics["T"]["forecast_quarters"]["Q1"]["mae"] = 105.1
    mark("small_quarter_over5pct_rejected", any(r["section"] == "forecast_quarters" for r in comparisons(bad_metrics)["quarter_or_large_group_deterioration_above_5pct"]))
    mark("stage_seed_order", parameters()["stage_indices"] == {"development": 0, "confirmation": 1, "auxiliary": 2})
    mark("complete24interval_contract", len(STAGES)*2*2*2 == 24)
    return {"status": "passed", "exit_code": 0, "check_count": len(checks), "checks": checks,
            "synthetic_only": True, "real_data_read": False, "real_query_prices_read": False,
            "new_fits": 0, "new_real_scores": 0, "source_sha256": sha(__file__), "source_hashes": source_hashes(),
            "parameters": parameters(), "plan_sha256": PLAN_SHA, "elapsed_seconds": time.monotonic()-tick}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-manifest")
    parser.add_argument("--evaluation-request")
    parser.add_argument("--out")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        need(args.out is not None and args.sealed_manifest is None and args.evaluation_request is None, "Self-check needs only output receipt")
        tick = time.monotonic()
        try:
            result = self_check()
        except Exception:
            write(args.out, {"status": "failed", "exit_code": 1, "synthetic_only": True,
                  "real_data_read": False, "source_sha256": sha(__file__), "failure": traceback.format_exc(),
                  "elapsed_seconds": time.monotonic()-tick})
            raise
        write(args.out, result)
        print(json.dumps({"status": result["status"], "check_count": result["check_count"], "elapsed_seconds": result["elapsed_seconds"]}))
    else:
        need(all((args.sealed_manifest, args.evaluation_request, args.out)), "Evaluation requires sealed manifest, request and new output")
        run(args.sealed_manifest, args.evaluation_request, args.out)
