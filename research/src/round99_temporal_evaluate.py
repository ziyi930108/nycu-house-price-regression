"""Evaluate three fully sealed R99 forward stackers; never fit or select weights.

The gate verifies every stage before deserializing the course price table.
Meta target files are hashed as bytes, never deserialized by this evaluator.
Synthetic checks use artificial inputs only. No native model load or inference.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round99_contract as c
import v3_cluster_stability as methods

BANDS = (("<10m", 0., 1e7), ("10-20m", 1e7, 2e7), ("20-50m", 2e7, 5e7),
         ("50-100m", 5e7, 1e8), (">=100m", 1e8, np.inf))
ARRAY_KEYS = {"meta_predictions", "meta_y", "meta_indices", "meta_source_cache_indices", "meta_trade_days",
              "query_predictions", "query_indices", "query_source_cache_indices", "query_trade_days", "fit_indices", "control_v22"}
META_ROWS = {"development": 105511, "offset": 137312, "confirmation": 186745}
FIT_ROWS = {"development": 240316, "offset": 272117, "confirmation": 321550}
NEED_COLUMNS = ("trade_day_index", "project_name", "address_city", "address_district", "address_raw", "address_road", "address_section")
CONTROL_RESULTS = {"outputs/round79/results.json": "7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6",
                   "outputs/round97/evaluation/results.json": "0c1fe34c720e72148de78577834661a2695aac7aa033a568ac4a51ec96d79abf",
                   "outputs/round98/evaluation/results.json": "f76032d2f067e7a317aafc268ffbeacd8a18b51b7ca368101807b1140a80cd1e"}
PREPARATION_PATH = "outputs/round98_inputs/manifest.json"
PREPARATION_SHA = "e496b2afcf7ebc69a4e248d13b691e3b2e6e991bfab4e510386ff77cccc97fc7"
PREPARATION_PLAN_SHA = "394a65459349e479bb3e8a487b60516156705469fa51bfbe75fd8697bae3642b"
R98_SEAL = {"path": "outputs/round98/sealed_manifest.json", "sha256": "d0ff684e01bf18d590f9d8cdd402a6067aa7d85df1ea223fc6cf03df097bcbcc"}


def parameters():
    return c.evaluation_parameters()


def source_hashes():
    return c.source_hashes(__file__)


def day(date):
    return (pd.Timestamp(date) - pd.Timestamp("2000-01-01")).days


def check_header(seal):
    c.need(seal.get("schema_version") == 1 and seal.get("status") == "all_temporal_stack_predictions_sealed", "All stages must be sealed")
    c.need(seal.get("new_meta_fits") == 3 and seal.get("new_native_fits") == 0, "Fixed three meta fits required")
    c.need(seal.get("query_scores_computed") is False and seal.get("all_query_reload_exact") is True, "Unscored full-query reload gate")
    c.need(seal.get("plan_sha256") == c.PLAN_SHA and set(seal.get("stages", {})) == set(c.STAGES), "Stage/plan contract differs")
    c.need(seal.get("protected_before") == seal.get("protected_after") and len(seal.get("protected_before", {})) == 9, "Protected artifacts differ")


def check_stage_header(seal, stage, prep, model_parameters):
    start, end, n = c.SPLITS[stage]
    c.need(seal.get("schema_version") == 1 and seal.get("status") == "temporal_stack_stage_predictions_sealed", "Stage not sealed")
    c.need(seal.get("stage") == stage and seal.get("cutoff") == start and seal.get("query_end") == end, "Stage dates differ")
    c.need(seal.get("query_rows") == n and seal.get("meta_rows") == META_ROWS[stage] == prep["meta_rows"], "Stage rows differ")
    for key, expected in (("query_labels_available", False), ("query_scores_computed", False), ("all_query_reload_exact", True)):
        c.need(seal.get(key) is expected, "Stage gate: " + key)
    c.need(seal.get("new_meta_fits") == 1 and seal.get("new_native_fits") == 0, "Wrong stage fit count")
    c.need(seal.get("parameters") == model_parameters and seal.get("plan_sha256") == c.PLAN_SHA, "Stage algorithm changed")
    c.need(seal.get("protected_before") == seal.get("protected_after") and len(seal.get("protected_before", {})) == 9, "Stage protection changed")


def index(a, rows, name, sorted_required=True):
    c.need(a.dtype == np.int64 and a.ndim == 1 and len(a) > 0 and (a >= 0).all() and (a < rows).all(), "Index schema/bounds: " + name)
    c.need(len(np.unique(a)) == len(a), "Repeated indices: " + name)
    if sorted_required:
        c.need((np.diff(a) > 0).all(), "Unsorted indices: " + name)


def prices(a, shape, name):
    c.need(a.dtype == np.float64 and a.shape == shape and np.isfinite(a).all() and (a >= 0).all(), "Invalid total prices: " + name)


def check_temporal(stage, arrays, days, mapping):
    start, end, n = c.SPLITS[stage]
    fi, qi, mi = (arrays[k] for k in ("fit_indices", "query_indices", "meta_indices"))
    for k in ("fit_indices", "query_indices", "meta_indices"):
        index(arrays[k], len(days), k, k != "meta_indices")
    c.need(len(fi) == FIT_ROWS[stage] and len(qi) == n and len(mi) == META_ROWS[stage], "Fixed stage row counts")
    c.exact(fi, np.flatnonzero(np.isfinite(days) & (days < day(start))).astype(np.int64), "Original fit date mask differs")
    c.exact(qi, np.flatnonzero(np.isfinite(days) & (days >= day(start)) & (days <= day(end))).astype(np.int64), "Query date mask differs")
    aux = np.flatnonzero(np.isfinite(days) & (days >= day("2021-08-11")) & (days <= day("2022-08-10"))).astype(np.int64)
    dev = np.flatnonzero(np.isfinite(days) & (days >= day("2022-08-11")) & (days <= day("2023-08-10")) & (days < day(start))).astype(np.int64)
    expected_meta = aux if stage == "development" else np.concatenate([aux, dev])
    c.exact(mi, expected_meta, "Meta origin/prefix order differs")
    c.need(not np.intersect1d(mi, qi).size and np.isin(mi, fi).all(), "Meta/query leakage")
    for role in ("meta", "query"):
        ids = arrays[role + "_indices"]
        c.exact(arrays[role + "_source_cache_indices"], mapping[ids], "Canonical/raw axis differs")
        c.exact(arrays[role + "_trade_days"], days[ids], "Saved transaction dates differ")


def check_state(state, model_parameters, prep_arrays, fit_rows, derived_meta):
    c.need(state.get("format") == "R99_convex_L1_stacker_v1" and state.get("parameters") == model_parameters, "Stacker state algorithm differs")
    c.need(state.get("branch_order") == list(c.STACKER_BRANCHES) and state.get("fit_rows") == fit_rows, "State axis/fit rows differ")
    c.need(state.get("solver_status") == 0 and state.get("certificate_passed") is True and state.get("query_dependent_state") is False, "Uncertified/query-dependent state")
    w = np.asarray(state.get("weights"), dtype=np.float64)
    tol = model_parameters["certificate_tolerances"]["weight_sum_absolute"]
    c.need(w.shape == (4,) and np.isfinite(w).all() and (w >= 0).all() and abs(float(w.sum()) - 1.) <= tol, "Invalid raw convex weights")
    c.need(state["runtime"]["scipy"] == model_parameters["required_scipy"] and state["runtime"]["highs"] == model_parameters["required_highs"], "Solver runtime differs")
    c.need(state["input_arrays"].get("row_ids_provided") is True
           and state["input_arrays"].get("row_ids_values_sha256") == prep_arrays["meta_indices"]["values_sha256"], "Fitted row order differs")
    for key, name in (("P", "meta_predictions"), ("y", "meta_y")):
        r = derived_meta if key == "P" else prep_arrays[name]
        expected = {k: r[k] for k in ("dtype", "shape", "values_sha256")}
        c.need(state["input_arrays"][key] == expected, "Fit P/y identity differs: " + key)
    return w


def augment(P):
    prices(P, (len(P), 3), "immutable original matrix")
    return np.column_stack([P, np.median(P, axis=1)])


def derived_record(P):
    return {"dtype": str(P.dtype), "shape": list(P.shape), "values_sha256": c.values_sha(P)}


def check_formulas(P, control, candidate, w):
    prices(P, (len(control), 4), "four-column query matrix")
    prices(control, (len(P),), "v22")
    prices(candidate, (len(P),), "R99")
    c.exact(np.median(P[:, :3], axis=1), P[:, 3], "Fourth column must equal original median exactly")
    c.exact(P[:, 3], control, "Original median must replay exactly")
    expected = ((P[:, 0] * w[0] + P[:, 1] * w[1]) + P[:, 2] * w[2]) + P[:, 3] * w[3]
    c.exact(expected, candidate, "Fixed ordered convex formula differs")


def preflight(sealed_path, request_path):
    inputs = {}
    def checked(path, digest):
        name = c.rel(path)
        c.need(name not in inputs or inputs[name] == digest, "Conflicting input authority: " + name)
        if name not in inputs:
            c.need(c.sha(name) == digest, "Changed input: " + name)
        inputs[name] = digest
        return c.safe(name)
    def document(record, sidecar=True):
        path = checked(record["path"], record["sha256"])
        if sidecar:
            checked(path.with_suffix(".sha256"), c.sha(path.with_suffix(".sha256")))
            return c.locked(path)
        return c.read(path)
    def array(record):
        checked(record["path"], record["sha256"])
        return c.load_array(record)
    c.need(c.sha(c.PLAN) == c.PLAN_SHA, "Fixed plan changed")
    request = document({"path": c.rel(request_path), "sha256": c.sha(request_path)})
    sources = source_hashes()
    c.need(request.get("status") == "evaluation_frozen_before_query_score" and request.get("parameters") == parameters(), "Evaluation not frozen")
    c.need(request.get("plan_sha256") == c.PLAN_SHA and request.get("source_sha256") == sources, "Evaluation source/plan changed")
    c.verify_hashes(sources)
    checked(c.PLAN, c.PLAN_SHA)
    model_parameters = request["model_parameters"]
    seal_record = {"path": c.rel(sealed_path), "sha256": c.sha(sealed_path)}
    seal = document(seal_record)
    check_header(seal)
    launch = document(seal["launch_request"])
    c.need(launch["status"] == "research_frozen_before_meta_fit" and launch["plan_sha256"] == c.PLAN_SHA
           and launch["helper_parameters"] == model_parameters and launch["source_sha256"] == seal["source_sha256"], "Model launch was not frozen consistently")
    main_execution = document({"path": c.rel(c.safe(sealed_path).parent / "execution.json"), "sha256": c.sha(c.safe(sealed_path).parent / "execution.json")})
    c.need(main_execution["exit_code"] == 0 and main_execution["failure"] is None, "Main execution incomplete")
    c.need(seal["protected_before"] == c.protected_hashes(), "Protected selection changed")
    c.verify_hashes(seal["source_sha256"])
    for name, digest in seal["source_sha256"].items(): checked(name, digest)
    preparation_record = seal["preparation"]
    c.need(preparation_record == {"path": PREPARATION_PATH, "sha256": PREPARATION_SHA}, "Original R98 preparation must remain unchanged")
    prep = document(preparation_record)
    c.need(launch["preparation"] == preparation_record, "Model launch preparation differs")
    c.need(prep["schema_version"] == 1 and prep["status"] == "prepared_temporal_meta_inputs" and prep["plan_sha256"] == PREPARATION_PLAN_SHA, "Preparation contract")
    c.need(prep["branch_order"] == list(c.BRANCHES) and prep["stage_order"] == list(c.STAGES) and set(prep["stages"]) == set(c.STAGES), "Preparation stages/branches")
    c.need(prep["query_statistics_computed"] is False and prep["all_arrays_saved_reload_exact"] is True and prep["test_read"] is False and prep["valid_read"] is False, "Preparation leakage/saving contract")
    inv = document(prep["inventory"], sidecar=False)
    c.need(inv["status"] == "passed" and inv["query_or_fit_labels_loaded"] is False, "Original forward inventory unavailable")
    for name, digest in prep["input_sha256"].items(): checked(name, digest)
    for name, digest in prep["source_sha256"].items(): checked(name, digest)
    for key in ("raw_train", "features", "canonical_mapping"):
        checked(prep[key]["path"], prep[key]["sha256"])
    old79 = document({"path": "outputs/round79/results.json", "sha256": CONTROL_RESULTS["outputs/round79/results.json"]})
    old97 = document({"path": "outputs/round97/evaluation/results.json", "sha256": CONTROL_RESULTS["outputs/round97/evaluation/results.json"]})
    old98 = document({"path": "outputs/round98/evaluation/results.json", "sha256": CONTROL_RESULTS["outputs/round98/evaluation/results.json"]})
    old98_seal = document(R98_SEAL)
    c.need(old98["status"] == "completed_three_later_window_evaluation" and old98["sealed_manifest"] == R98_SEAL
           and old98_seal["status"] == "all_temporal_stack_predictions_sealed", "Original R98 comparator incomplete")
    c.need(old98_seal["preparation"] == preparation_record and old98_seal["plan_sha256"] == PREPARATION_PLAN_SHA, "Original R98 source identity differs")
    c.need(inv["verified_input_sha256"]["outputs/round79/results.json"] == CONTROL_RESULTS["outputs/round79/results.json"], "Annual control authority differs")
    original_metrics = {s: old79["stages"][s]["metrics"]["R79"] for s in ("development", "confirmation")}
    original_metrics["offset"] = old97["metrics"]["v22"]
    c.need(prep["raw_train"]["path"] == c.RAW and prep["raw_train"]["rows"] == 455784 and prep["features"]["path"] == c.FEATURES and prep["canonical_mapping"]["path"] == c.CANONICAL, "Unexpected source files")
    # Pure features only: no raw price table or meta target file is deserialized here.
    mapping = np.load(c.safe(prep["canonical_mapping"]["path"]), allow_pickle=False)
    index(mapping, 455784, "canonical raw mapping")
    c.need(len(mapping) == 455255, "Canonical population differs")
    Xraw = pd.read_pickle(c.safe(prep["features"]["path"]))
    c.need(isinstance(Xraw, pd.DataFrame) and len(Xraw) == 455784 and Xraw.index.equals(pd.RangeIndex(455784)), "Pure-X row axis")
    c.need(not {"總價元", "target", "y", "total_price", "編號"}.intersection(Xraw.columns) and set(NEED_COLUMNS).issubset(Xraw.columns), "Pure-X target/schema")
    X = Xraw.iloc[mapping].reset_index(drop=True)
    del Xraw
    days = X.trade_day_index.to_numpy(dtype=np.float64)
    all_stages = {}
    for stage in c.STAGES:
        b, p = seal["stages"][stage], prep["stages"][stage]
        st = document(b["stage_manifest"])
        check_stage_header(st, stage, p, model_parameters)
        c.need(st["preparation"] == preparation_record and st["model_state"] == b["model_state"] and st["predictions"] == b["predictions"], "Stage/root bindings differ")
        c.need(st["launch_request"] == seal["launch_request"] and st["source_sha256"] == launch["driver_source_sha256"], "Stage launch source differs")
        c.need(st["raw_price_pickle_opened"] is False and st["other_stage_meta_targets_opened"] is False and st["caller_arrays_unchanged"] is True, "Stage label isolation differs")
        c.need(st["protected_before"] == seal["protected_before"], "Protection differs across stages")
        ex = document(b["execution"])
        c.need(ex["exit_code"] == 0 and ex["failure"] is None, "Stage execution incomplete")
        for name, digest in st["source_sha256"].items(): checked(name, digest)
        for name, digest in st["model_artifact_sha256"].items(): checked(name, digest)
        records = p["arrays"]
        c.need(st["input_arrays"] == records and st["actual_stage_data_readset"] == {r["path"]: r["sha256"] for r in records.values()}, "Stage prepared readset differs")
        c.need(set(records) == ARRAY_KEYS and p["cutoff"] == c.SPLITS[stage][0] and p["query_end"] == c.SPLITS[stage][1] and p["query_rows"] == c.SPLITS[stage][2], "Stage preparation arrays/dates")
        for key, r in records.items():
            c.need(r["path"] == f"outputs/round98_inputs/{stage}/{key}.npy", "Cross-stage/role array path")
        # Hash only: target bytes enter no ndarray, JSON output or scoring function.
        checked(records["meta_y"]["path"], records["meta_y"]["sha256"])
        a = {k: array(r) for k, r in records.items() if k != "meta_y"}
        check_temporal(stage, a, days, mapping)
        P4, Q4 = augment(a["meta_predictions"]), augment(a["query_predictions"])
        derived_meta = derived_record(P4)
        c.need(st["derived_meta_predictions"] == derived_meta and st["derived_query_predictions"] == derived_record(Q4), "Derived four-column identity differs")
        state = document(b["model_state"], sidecar=False)
        w = check_state(state, model_parameters, records, p["meta_rows"], derived_meta)
        meta_path = c.safe(b["model_state"]["path"]).parent / "fit_metadata.json"
        meta = document({"path": c.rel(meta_path), "sha256": st["model_artifact_sha256"][c.rel(meta_path)]}, sidecar=False)
        c.need(meta["status"] == "completed" and meta["parameters"] == model_parameters and meta["input_arrays"] == state["input_arrays"], "Fit metadata/state identity differs")
        c.need(meta["certificate"]["all_checks_passed"] is True and all(v is True for v in meta["certificate"]["checks"].values())
               and meta["certificate"]["weights"] == state["weights"] and meta["weights_clipped_normalized_or_regularized"] is False, "Fit numerical certificate failed")
        cert = meta["certificate"]
        c.need(set(cert["native_vertex_MAE_TWD"]) == set(c.STACKER_BRANCHES) and cert["checks"]["median_vertex_optimality"] is True,
               "Fourth median vertex certificate missing")
        c.need(cert["median_vertex_MAE_TWD"] == cert["native_vertex_MAE_TWD"]["M"]
               and cert["primal_minus_median_vertex_MAE_TWD"] <= model_parameters["certificate_tolerances"]["vertex_optimality_mean_TWD"], "Median vertex certificate differs")
        old_stage = document(old98_seal["stages"][stage]["stage_manifest"])
        c.need(old_stage["input_arrays"] == records and old_stage["preparation"] == preparation_record and old_stage["all_query_reload_exact"] is True, "R98 comparator row/input alignment differs")
        old_meta_path = f"outputs/round98/{stage}/model/fit_metadata.json"
        old_meta = document({"path": old_meta_path, "sha256": old_stage["model_artifact_sha256"][old_meta_path]}, sidecar=False)
        old_objective = old_meta["certificate"]["primal_MAE_TWD"]
        c.need(old_meta["status"] == "completed" and old_meta["certificate"]["all_checks_passed"] is True, "Original R98 meta objective unavailable")
        c.need(st["old_R98_meta_objective_TWD"] == old_objective and st["R99_meta_objective_TWD"] == cert["primal_MAE_TWD"]
               and st["R99_not_worse_than_R98_meta_tolerance_TWD"] == .10 and st["R99_not_worse_than_R98_meta_verified"] is True
               and cert["primal_MAE_TWD"] <= old_objective + .10, "Meta feasible R98 objective comparison failed")
        for name, digest in state["source_sha256"].items(): checked(name, digest)
        c.need(b["model_state"]["sha256"] == st["model_artifact_sha256"][b["model_state"]["path"]], "State artifact authority differs")
        control, candidate = (array(b["predictions"][k]) for k in ("v22", "R99"))
        r98_control = array(old98_seal["stages"][stage]["predictions"]["R98"])
        prices(r98_control, (len(control),), "R98 fixed comparator")
        c.exact(control, a["control_v22"], "Stage control differs from original source")
        check_formulas(Q4, control, candidate, w)
        all_stages[stage] = {"arrays": a, "fit": X.iloc[a["fit_indices"]].reset_index(drop=True),
                             "query": X.iloc[a["query_indices"]].reset_index(drop=True),
                             "predictions": {**{k: a["query_predictions"][:, j].copy() for j, k in enumerate(c.BRANCHES)}, "v22": control, "R98": r98_control, "R99": candidate},
                             "weights": state["weights"], "meta_rows": p["meta_rows"]}
    c.need(source_hashes() == sources, "Evaluator changed during gate")
    return {"stages": all_stages, "source_sha256": sources, "input_sha256": inputs,
            "sealed_manifest": seal_record, "preparation": preparation_record, "raw_train": prep["raw_train"],
            "original_control_metrics": original_metrics,
            "original_R98_metrics": {s: old98["stages"][s]["metrics"]["R98"] for s in c.STAGES},
            "protected_before": seal["protected_before"], "evaluation_request": {"path": c.rel(request_path), "sha256": c.sha(request_path)}}


def select_query_y(raw, raw_indices, expected_raw_rows=455784):
    c.need(isinstance(raw, pd.DataFrame) and len(raw) == expected_raw_rows and raw.index.equals(pd.RangeIndex(expected_raw_rows)), "Raw price row axis differs")
    index(raw_indices, len(raw), "raw query indices")
    y = raw.iloc[raw_indices]["總價元"].to_numpy(dtype=np.float64, copy=True)
    c.need(np.isfinite(y).all() and (y > 0).all(), "Invalid query target")
    return y


def descriptive_metrics(y, prediction, fit, query, start):
    c.need(y.shape == prediction.shape == (len(query),), "Scoring alignment differs")
    c.need(np.isfinite(y).all() and (y > 0).all(), "Invalid scoring targets")
    prices(prediction, y.shape, "score")
    result = methods.original_score(y, prediction, fit, query, start, c.WEIGHTS)
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
    c.need(sum(b["rows"] for b in bands) == len(y), "Price bands not partition")
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


def paired(y, predictions, fit, query, query_indices, stage):
    start = c.SPLITS[stage][0]
    stage_index = c.STAGES.index(stage)
    labels = methods.group_labels(fit, query)
    errors = np.abs(y[:, None] - np.column_stack([predictions[k] for k in ("v22", "R99")]))
    point = methods.point_values(errors, labels, c.WEIGHTS)
    keys, fallback = methods.cluster_keys(query, query_indices)
    months = methods.relative_months(query.trade_day_index, start)
    result = {"control": "v22", "candidate": "R99", "difference": "R99 minus v22", "seed": 20260925,
              "replicates": 2000, "stage_index": stage_index, "fallback": fallback, "methods": {}, "caution": c.CAUTION}
    arrays = {}
    for family_index, (name, fn, groups) in enumerate((("project", methods.bootstrap_projects, keys), ("month", methods.bootstrap_months, months))):
        samples, support = fn(errors, labels, groups, c.WEIGHTS,
            np.random.default_rng(np.random.SeedSequence([20260925, stage_index, family_index])), 2000)
        summary = {}
        for metric, item in methods.summarize(samples, point).items():
            translated = {k: v for k, v in item.items() if k not in ("v2_mae", "v3_mae", "point_delta_v3_minus_v2")}
            translated.update(control_mae=item["v2_mae"], candidate_mae=item["v3_mae"], point_delta=item["point_delta_v3_minus_v2"])
            summary[metric] = translated
        result["methods"][name] = {"family_index": family_index, "summary": summary, "support": support}
        arrays[name] = samples
    return result, arrays


def comparisons(metrics, control_name="v22"):
    control, candidate = metrics[control_name], metrics["R99"]
    def difference(a, b):
        return {"control_mae": a, "candidate_mae": b, "delta": b - a,
                "relative_change_pct": 100 * (b - a) / a if a else None}
    result = {k: difference(control[k], candidate[k]) for k in ("overall_mae", "weighted_mae")}
    adverse = []
    for section in ("strata", "forecast_quarters", "months", "cities", "relative_forecast_months"):
        result[section] = {}
        for key, a in control[section].items():
            b = candidate[section][key]
            c.need(a["rows"] == b["rows"], "Group support differs")
            if a["rows"]:
                item = {"rows": a["rows"], **difference(a["mae"], b["mae"])}
                result[section][key] = item
                if (section == "forecast_quarters" or a["rows"] >= 500) and b["mae"] > 1.05 * a["mae"]:
                    adverse.append({"section": section, "group": key, **item})
    result["price_bands"] = []
    for a, b in zip(control["price_bands"], candidate["price_bands"]):
        c.need(a["band"] == b["band"] and a["rows"] == b["rows"], "Price support differs")
        item = {"band": a["band"], "rows": a["rows"], **(difference(a["mae"], b["mae"]) if a["rows"] else {})}
        result["price_bands"].append(item)
        if a["rows"] >= 500 and b["mae"] > 1.05 * a["mae"]:
            adverse.append({"section": "price_bands", **item})
    result["both_point_metrics_improve"] = all(candidate[k] < control[k] for k in ("overall_mae", "weighted_mae"))
    result["quarter_or_large_group_deterioration_above_5pct"] = adverse
    if control_name == "v22":
        result["stage_research_gate_passed"] = result["both_point_metrics_improve"] and not adverse
    result["control"] = control_name
    result["candidate"] = "R99"
    result["used_for_research_gate"] = control_name == "v22"
    return result


def research_gate(stages):
    passed = all(stages[s]["comparison"]["stage_research_gate_passed"] for s in c.STAGES)
    return {"all_three_later_windows_passed": passed, "worth_earlier_forward_block_research": passed,
            "original_three_period_adoption_complete": False, "automatic_adoption": False,
            "full_fit_authorized_by_gate": False, "new_CSV": False, "selected_submission": "v22", "Private_score": None}


def render(result):
    f = lambda x: "NA" if x is None else f"{x:,.2f}"
    lines = ["# R99：較早 forward 預測學權重的三窗探索", "", c.CAUTION, "",
             "只改最後四欄 [N,H,U,M=median(N,H,U)] 組合；三窗各自學一組權重。負差代表 R99 較低。沒有新 CSV 或 Private 成績。", "",
             "|窗口|v22 overall|R99 overall|差|weighted 差|研究門檻|", "|---|---:|---:|---:|---:|---|"]
    for s in c.STAGES:
        r = result["stages"][s]; m = r["metrics"]; d = r["comparison"]
        lines.append(f'|{s}|{f(m["v22"]["overall_mae"])}|{f(m["R99"]["overall_mae"])}|{f(d["overall_mae"]["delta"])}|{f(d["weighted_mae"]["delta"])}|{d["stage_research_gate_passed"]}|')
    lines += ["", "|窗口|權重 N / H / U / M|meta 列數|query 列數|", "|---|---|---:|---:|"]
    for s in c.STAGES:
        r = result["stages"][s]
        lines.append(f'|{s}|{" / ".join(f"{v:.12g}" for v in r["weights"])}|{r["meta_rows"]}|{r["query_rows"]}|')
    lines += ["", "## 固定 R98 對照（只報點差）", "",
              "以下不是另一項選型門檻，也沒有額外 bootstrap。R98 預測原樣讀取，各窗控制 MAE 需匹配原結果。", "",
              "|窗口|R98 overall|R99 overall|R99-R98 差|weighted 差|", "|---|---:|---:|---:|---:|"]
    for s in c.STAGES:
        r = result["stages"][s]; m = r["metrics"]; d = r["comparison_vs_R98"]
        lines.append(f'|{s}|{f(m["R98"]["overall_mae"])}|{f(m["R99"]["overall_mae"])}|{f(d["overall_mae"]["delta"])}|{f(d["weighted_mae"]["delta"])}|')
    lines += ["", "## 全部配對區間", "", "|窗口|抽樣|指標|點差|2.5%|97.5%|有效／無效|", "|---|---|---|---:|---:|---:|---|"]
    for s in c.STAGES:
        for family, b in result["stages"][s]["paired"]["methods"].items():
            for metric, v in b["summary"].items():
                lines.append(f'|{s}|{family}|{metric}|{f(v["point_delta"])}|{f(v["percentile_2_5"])}|{f(v["percentile_97_5"])}|{v["valid_replicates"]} / {v["invalid_replicates"]}|')
    for s in c.STAGES:
        r = result["stages"][s]
        lines += ["", "## " + s + " 正反分群", "", "|分群|列數|MAE 差|", "|---|---:|---:|"]
        for section in ("strata", "forecast_quarters"):
            for name, v in r["comparison"][section].items(): lines.append(f'|{section}/{name}|{v["rows"]}|{f(v["delta"])}|')
        for v in r["comparison"]["price_bands"]: lines.append(f'|{v["band"]}|{v["rows"]}|{f(v.get("delta"))}|')
        lines += ["", "季度或 ≥500 列分群 >5% 退步項：" + json.dumps(r["comparison"]["quarter_or_large_group_deterioration_above_5pct"], ensure_ascii=False),
                  "", "城市／月份／所有單支、尾端與完整聚合值均保存於 results.json。真價帶只用於這份診斷，不進推論。"]
        lines += ["", "### 同窗 R99 − R98 點差", "", "|分群|列數|MAE 差|", "|---|---:|---:|"]
        for section in ("strata", "forecast_quarters"):
            for name, v in r["comparison_vs_R98"][section].items(): lines.append(f'|{section}/{name}|{v["rows"]}|{f(v["delta"])}|')
        for v in r["comparison_vs_R98"]["price_bands"]: lines.append(f'|{v["band"]}|{v["rows"]}|{f(v.get("delta"))}|')
        lines += ["", "相對 R98 的季度或 ≥500 列群 >5% 退步（描述用、不新增門檻）：" + json.dumps(r["comparison_vs_R98"]["quarter_or_large_group_deterioration_above_5pct"], ensure_ascii=False)]
    lines += ["", "全部探索門檻通過：" + str(result["research_gate"]["all_three_later_windows_passed"]),
              "仍缺 auxiliary 之前的 forward meta block；即使通過，也只值得另立更早窗口研究。沒有全量候選或採用決策。",
              "CI 條件於固定模型，不涵蓋重訓／反覆選型；12 個區間全部呈現，不是 Private 勝率。",
              "第四欄納入 median 只保證精確 meta L1 最優不劣於可行的 median 解；不是 query、群加權 MAE 或 Private 改善保證。",
              "meta-y 在評分器只核檔案 bytes，不反序列化；raw pickle 於三窗完整封存核驗後才反序列化，未保存逐列真價。", ""]
    return "\n".join(lines)


def run(sealed_path, request_path, out):
    out = c.safe(out)
    c.need(not out.exists(), "Evaluation output exists; preserve attempt")
    out.mkdir(parents=True, exist_ok=False)
    started, tick, code, failure, labels_opened = c.now(), time.monotonic(), 1, None, False
    try:
        checked = preflight(sealed_path, request_path)
        c.write(out / "prelabel_gate.json", {"status": "all_stages_verified_before_price_deserialization", "checked_stages": list(c.STAGES),
                "sealed_manifest": checked["sealed_manifest"], "parameters": parameters(), "source_sha256": checked["source_sha256"],
                "verified_input_count": len(checked["input_sha256"]), "meta_target_files_hashed_only": True, "completed_at_utc": c.now()})
        # This is the sole real price deserialization in the scorer, after the complete gate.
        c.need(c.sha(checked["raw_train"]["path"]) == checked["raw_train"]["sha256"], "Raw prices changed after gate")
        raw = pd.read_pickle(c.safe(checked["raw_train"]["path"])); labels_opened = True
        stages, artifacts = {}, {}
        for stage in c.STAGES:
            a = checked["stages"][stage]
            y = select_query_y(raw, a["arrays"]["query_source_cache_indices"])
            metrics = {name: descriptive_metrics(y, p, a["fit"], a["query"], c.SPLITS[stage][0]) for name, p in a["predictions"].items()}
            for key in ("overall_mae", "weighted_mae"):
                c.need(abs(metrics["v22"][key] - checked["original_control_metrics"][stage][key]) <= 1e-6, "Original v22 score replay differs: " + stage + "/" + key)
                c.need(abs(metrics["R98"][key] - checked["original_R98_metrics"][stage][key]) <= 1e-6, "Original R98 score replay differs: " + stage + "/" + key)
            boot, samples = paired(y, a["predictions"], a["fit"], a["query"], a["arrays"]["query_indices"], stage)
            for family in boot["methods"].values():
                for metric, key in (("overall", "overall_mae"), ("weighted", "weighted_mae")):
                    r = family["summary"][metric]
                    c.need(abs(r["control_mae"] - metrics["v22"][key]) <= 1e-6 and abs(r["candidate_mae"] - metrics["R99"][key]) <= 1e-6, "Paired point differs")
            folder = out / stage; folder.mkdir()
            reps = {family: c.save_array(folder / (family + "_replicates.npy"), b) for family, b in samples.items()}
            item = {"stage": stage, "cutoff": c.SPLITS[stage][0], "query_end": c.SPLITS[stage][1], "query_rows": len(y),
                    "meta_rows": a["meta_rows"], "weights": a["weights"], "metrics": metrics, "comparison": comparisons(metrics), "comparison_vs_R98": comparisons(metrics, "R98"),
                    "paired": boot, "replicate_artifacts": reps, "query_target_values_sha256": c.values_sha(y),
                    "query_target_values_persisted": False, "group_membership_uses_original_native_fit_not_meta_fit": True,
                    "original_v22_overall_weighted_replay_verified": True, "original_R98_overall_weighted_replay_verified": True,
                    "control_replay_tolerance_TWD": 1e-6, "R98_additional_bootstrap_calls": 0}
            c.write(folder / "results.json", item)
            stages[stage] = item
            print(json.dumps({"stage": stage, "overall_delta": item["comparison"]["overall_mae"]["delta"],
                              "weighted_delta": item["comparison"]["weighted_mae"]["delta"]}), flush=True)
        del raw
        c.verify_hashes(checked["input_sha256"])
        c.need(source_hashes() == checked["source_sha256"] and c.protected_hashes() == checked["protected_before"], "Source/protected changed")
        result = {"round": 99, "status": "completed_three_later_window_evaluation", "created_at_utc": c.now(), "parameters": parameters(),
                  "checked_stages": list(c.STAGES), "stages": stages, "research_gate": research_gate(stages),
                  "sealed_manifest": checked["sealed_manifest"], "evaluation_request": checked["evaluation_request"],
                  "source_sha256": checked["source_sha256"], "input_sha256": checked["input_sha256"],
                  "meta_target_files_hashed_only": True, "raw_pickle_whole_table_deserialized_after_all_seals": True,
                  "query_target_values_persisted": False, "query_y_mapping": "raw_train.iloc[query_source_cache_indices]['總價元']",
                  "query_labels_used_in_training_or_predictions": False, "cross_stage_earlier_query_labels_are_legal_later_meta_labels": True,
                  "model_fit_load_predict_calls": 0, "new_meta_fits_in_scorer": 0, "GPU_calls": 0, "test_files_read": 0,
                  "additional_control": "R98", "R98_comparisons_point_metrics_only": True, "R98_additional_bootstrap_calls": 0,
                  "selection_changed": False, "new_CSV": False, "Private_score": None, "fresh_blind_validation": False,
                  "caution": c.CAUTION, "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
                  "elapsed_seconds": time.monotonic() - tick}
        with (out / "report.md").open("x", encoding="utf8", newline="\n") as f: f.write(render(result))
        result["artifact_sha256"] = {c.rel(p): c.sha(p) for p in out.rglob("*") if p.is_file()}
        c.write(out / "results.json", result)
        code = 0
        return result
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        c.write(out / "execution.json", {"exit_code": code, "failure": failure, "started_at_utc": started, "completed_at_utc": c.now(),
                "elapsed_seconds": time.monotonic() - tick, "source_sha256": c.sha(__file__), "raw_prices_deserialized": labels_opened,
                "new_fits": 0, "native_model_load_predict_calls": 0, "automatic_retry": False})


def self_check():
    tick = time.monotonic(); checks = []
    def mark(name, ok):
        c.need(bool(ok), name); checks.append(name)
    def rejects(name, fn):
        try: fn()
        except (ValueError, KeyError): checks.append(name); return
        raise AssertionError("Expected rejection: " + name)
    header = {"schema_version": 1, "status": "all_temporal_stack_predictions_sealed", "new_meta_fits": 3, "new_native_fits": 0,
              "query_scores_computed": False, "all_query_reload_exact": True, "plan_sha256": c.PLAN_SHA,
              "stages": dict.fromkeys(c.STAGES), "protected_before": dict.fromkeys(range(9), "same"), "protected_after": dict.fromkeys(range(9), "same")}
    check_header(header); mark("all_stage_header_passes", True)
    for key, value in (("status", "running"), ("new_meta_fits", 2), ("new_native_fits", 1), ("query_scores_computed", True),
                       ("all_query_reload_exact", False), ("plan_sha256", "bad"), ("stages", {"development": {}})):
        rejects("reject_header_" + key, lambda k=key, v=value: check_header({**header, k: v}))
    P3 = np.array([[1., 3., 8.], [0., 0., 0.], [9., 4., 2.]], dtype=np.float64) * 1e6
    P = augment(P3)
    w = np.array([.1, .2, .3, .4]); v22 = np.median(P3, axis=1)
    pred = ((P[:, 0] * w[0] + P[:, 1] * w[1]) + P[:, 2] * w[2]) + P[:, 3] * w[3]
    check_formulas(P, v22, pred, w); mark("ordered_formula_and_original_median", True)
    mp = {"certificate_tolerances": {"weight_sum_absolute": 1e-10}, "required_scipy": "synthetic", "required_highs": "synthetic"}
    def rec(a): return {"dtype": str(a.dtype), "shape": list(a.shape), "values_sha256": c.values_sha(a)}
    prep = {"meta_predictions": rec(P3), "meta_y": rec(v22), "meta_indices": rec(np.array([3, 1, 8], dtype=np.int64))}
    dm = rec(P)
    state = {"format": "R99_convex_L1_stacker_v1", "parameters": mp, "branch_order": list(c.STACKER_BRANCHES), "fit_rows": 3,
             "solver_status": 0, "certificate_passed": True, "query_dependent_state": False, "weights": w.tolist(),
             "runtime": {"scipy": "synthetic", "highs": "synthetic"},
             "input_arrays": {"P": dm, "y": prep["meta_y"], "row_ids_provided": True, "row_ids_values_sha256": prep["meta_indices"]["values_sha256"]}}
    mark("state_binds_derived_four_columns_and_original_target_row_order", np.array_equal(check_state(state, mp, prep, 3, dm), w))
    for key, value in (("weights", [-.1, .2, .3, .6]), ("weights", [.1, .2, .3, .1]), ("certificate_passed", False),
                       ("query_dependent_state", True), ("branch_order", ["H", "N", "U"]), ("fit_rows", 4)):
        rejects("state_reject_" + key + "_" + str(value), lambda k=key, v=value: check_state({**state, k: v}, mp, prep, 3, dm))
    changed = copy.deepcopy(state); changed["input_arrays"]["row_ids_values_sha256"] = "bad"
    rejects("fitted_row_alignment_hash_rejected", lambda: check_state(changed, mp, prep, 3, dm))
    rejects("state_original_three_column_hash_not_accepted", lambda: check_state(state, mp, prep, 3, rec(P3)))
    corrupted = P.copy(); corrupted[:, 3] += 1.
    rejects("corrupt_median_column_rejected", lambda: check_formulas(corrupted, v22, pred, w))
    check_formulas(P, v22, v22, np.array([0., 0., 0., 1.])); mark("median_vertex_preserves_v22_exactly", True)
    check_formulas(P[::-1], v22[::-1], pred[::-1], w); mark("query_permutation_preserves_rows", True)
    stage_header = {"schema_version": 1, "status": "temporal_stack_stage_predictions_sealed", "stage": "offset", "cutoff": "2023-02-11",
                    "query_end": "2024-02-10", "query_rows": 107169, "meta_rows": 137312, "query_labels_available": False,
                    "query_scores_computed": False, "all_query_reload_exact": True, "new_meta_fits": 1, "new_native_fits": 0,
                    "parameters": mp, "plan_sha256": c.PLAN_SHA, "protected_before": header["protected_before"], "protected_after": header["protected_after"]}
    check_stage_header(stage_header, "offset", {"meta_rows": 137312}, mp); mark("complete_stage_header_accepts", True)
    rejects("stage_query_labels_blocked", lambda: check_stage_header({**stage_header, "query_labels_available": True}, "offset", {"meta_rows": 137312}, mp))
    rejects("stage_wrong_origin_blocked", lambda: check_stage_header({**stage_header, "cutoff": "2023-08-11"}, "offset", {"meta_rows": 137312}, mp))
    rejects("changed_control_rejected", lambda: check_formulas(P, v22 + 1., pred, w))
    rejects("changed_candidate_rejected", lambda: check_formulas(P, v22, pred + 1., w))
    rejects("wrong_branch_count_rejected", lambda: check_formulas(P[:, :2], v22, pred, w))
    rejects("float32_query_rejected", lambda: check_formulas(P.astype(np.float32), v22, pred, w))
    raw = pd.DataFrame({"總價元": np.arange(1., 9.)})
    ids = np.array([1, 4, 7], dtype=np.int64)
    mark("raw_not_canonical_label_mapping", np.array_equal(select_query_y(raw, ids, 8), [2., 5., 8.]))
    rejects("duplicate_raw_indices_rejected", lambda: select_query_y(raw, np.array([1, 1], dtype=np.int64), 8))
    rejects("raw_index_bounds_rejected", lambda: select_query_y(raw, np.array([8], dtype=np.int64), 8))
    for s, (start, end, _) in c.SPLITS.items():
        dates = pd.to_datetime([start, (pd.Timestamp(start) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                (pd.Timestamp(start) + pd.DateOffset(months=1)).strftime("%Y-%m-%d"), end])
        ds = np.asarray((dates - pd.Timestamp("2000-01-01")).days, dtype=np.float64)
        mark("relative_month_boundaries_" + s, np.array_equal(methods.relative_months(ds, start), [0, 0, 1, 11]))
    fit = pd.DataFrame({"project_name": ["seen"], "address_city": ["city"], "address_district": ["district"]})
    start = c.SPLITS["offset"][0]
    dates = [pd.Timestamp(start) + pd.DateOffset(months=i) for i in range(12)]
    q = pd.DataFrame({"project_name": ["seen", "new"] * 6, "address_city": ["city"] * 12, "address_district": ["district"] * 12,
                      "address_raw": ["road1"] * 12, "address_road": ["road"] * 12, "address_section": ["one"] * 12,
                      "trade_day_index": np.array([(d - pd.Timestamp("2000-01-01")).days for d in dates], dtype=np.float64)})
    y = np.linspace(1e6, 1.2e8, 12); pred0 = y + 10.; pred1 = y + 7.
    metrics = {"v22": descriptive_metrics(y, pred0, fit, q, start), "R99": descriptive_metrics(y, pred1, fit, q, start)}
    metrics["R98"] = descriptive_metrics(y, pred0 + 1., fit, q, start)
    comp = comparisons(metrics)
    secondary = comparisons(metrics, "R98")
    mark("R98_full_point_comparison_has_no_extra_gate", secondary["overall_mae"]["delta"] == -4.
         and not secondary["used_for_research_gate"] and "stage_research_gate_passed" not in secondary)
    mark("metric_delta_sign_and_weighted", comp["overall_mae"]["delta"] == -3. and abs(comp["weighted_mae"]["delta"] + 3.) < 1e-10)
    mark("all_price_bands_partition", sum(x["rows"] for x in metrics["v22"]["price_bands"]) == 12)
    mark("original_fit_defines_seen_group", metrics["v22"]["strata"]["seen"]["rows"] == 6)
    mark("zero_support_missing_retained", metrics["v22"]["strata"]["missing"] == {"rows": 0, "mae": None})
    both = {"v22": pred0, "R99": pred1}
    boot, samples = paired(y, both, fit, q, np.arange(12, dtype=np.int64), "offset")
    mark("offset_fixed_stage_index_one", boot["stage_index"] == 1 and boot["replicates"] == 2000)
    for family, a in samples.items():
        valid = np.isfinite(a).all(axis=1)
        mark(family + "_paired_constant_delta", a.shape == (2000, 4) and np.allclose(a[valid, 1] - a[valid, 0], -3.) and np.allclose(a[valid, 3] - a[valid, 2], -3.))
        s = boot["methods"][family]["summary"]
        mark(family + "_all_replicates_accounted", all(v["valid_replicates"] + v["invalid_replicates"] == 2000 for v in s.values()))
    worse = copy.deepcopy(metrics)
    worse["R99"]["forecast_quarters"]["Q1"]["mae"] = 10.51
    mark("small_quarter_adverse_gate", any(x["section"] == "forecast_quarters" for x in comparisons(worse)["quarter_or_large_group_deterioration_above_5pct"]))
    artificial_stages = {s: {"comparison": comp} for s in c.STAGES}
    gate = research_gate(artificial_stages)
    mark("pass_only_upgrades_research_not_adoption", gate["all_three_later_windows_passed"] and not gate["automatic_adoption"] and not gate["original_three_period_adoption_complete"] and not gate["new_CSV"])
    artificial_stages["offset"] = {"comparison": {**comp, "stage_research_gate_passed": False}}
    mark("one_failed_window_blocks_upgrade", not research_gate(artificial_stages)["all_three_later_windows_passed"])
    return {"status": "passed", "exit_code": 0, "synthetic_only": True, "real_data_read": False, "real_query_prices_read": False,
            "real_meta_fits": 0, "native_fit_load_predict_calls": 0, "GPU_calls": 0, "check_count": len(checks), "checks": checks,
            "source_sha256": c.sha(__file__), "source_hashes": source_hashes(), "plan_sha256": c.PLAN_SHA,
            "parameters": parameters(), "elapsed_seconds": time.monotonic() - tick, "completed_at_utc": c.now()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--sealed-manifest"); ap.add_argument("--evaluation-request"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.self_check:
        out = c.safe(args.out)
        c.need(not out.exists(), "Synthetic receipt exists; preserve it")
        try:
            result = self_check()
        except Exception:
            c.write(out, {"status": "failed", "exit_code": 1, "synthetic_only": True, "real_data_read": False,
                          "source_sha256": c.sha(__file__), "failure": traceback.format_exc(), "completed_at_utc": c.now()})
            raise
        c.write(out, result)
        print(json.dumps({"status": result["status"], "check_count": result["check_count"], "elapsed_seconds": result["elapsed_seconds"]}))
    else:
        if not args.sealed_manifest or not args.evaluation_request: ap.error("--sealed-manifest and --evaluation-request required")
        run(args.sealed_manifest, args.evaluation_request, args.out)
