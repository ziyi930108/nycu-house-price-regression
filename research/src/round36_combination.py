"""R36: one predeclared log-geography + unchanged R8 spatial combination.

No tree fit/load, parameter search, CSV or adoption decision. R35 must have
completed both periods before freeze; run always evaluates all three periods.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import pickle
from pathlib import Path
import json
import time
import traceback

import numpy as np
import round33_reproduction as spatial_run
import round35_reproduction as log_run

c6, e6 = log_run.c6, log_run.e6
spatial = spatial_run.component
ROOT = c6.ROOT
OUT = ROOT / "outputs/round36"
STAGES = ("development", "confirmation", "auxiliary")
V3, SPATIAL, LOG, COMBINED = "v3_control", "spatial_v3", "log_geography_blend", "log_spatial_combination"
DOCUMENT = ROOT / "reports/round36_combination_plan.md"
DOCUMENT_SHA = "310741147932c8097903593d6f4179b2f2f2303c0918efc96179485be7e7af4d"
require, sha, rel, locked, locked_write = log_run.require, log_run.sha, log_run.rel, log_run.locked, log_run.locked_write
merge, now, save_array = log_run.merge, log_run.now, log_run.save_array
FORMULA = "B_log=.75*(.5*old_raw+.5*log_geo_TWD)+.25*old_unit; p_log,diag=reference_trend_once(B_log); R8.predict(saved_state,Xq,geoq,B_log,p_log,diag)"
PROTECTED = tuple(dict.fromkeys(log_run.PROTECTED + (
    "outputs/round8/candidate_plan.json", "outputs/round8/frozen_candidate.json", "outputs/round8/accepted_candidate.json",
    "outputs/round33/reproduction_plan.json", "outputs/round33/results.json",
    "outputs/round35/reproduction_plan.json", "outputs/round35/results.json",
)))


def sources():
    return merge(log_run.sources(), spatial_run.sources(), {rel(Path(__file__)): sha(Path(__file__))})


def small(mapping, path, expected=None, receipt=True):
    path = Path(path)
    digest = sha(path)
    require(expected is None or digest == expected, "Authority SHA differs: " + rel(path))
    mapping.update(merge(mapping, {rel(path): digest}))
    if receipt:
        value = locked(path)
        mapping[rel(path.with_suffix(".sha256"))] = sha(path.with_suffix(".sha256"))
        return value
    return c6.read(path)


def completed_reproduction(module, mapping):
    specification = module.read_plan()
    path = module.OUT / "results.json"
    value = small(mapping, path)
    small(mapping, module.OUT / "reproduction_plan.json")
    require(value["status"] == "both_periods_completed_no_adoption_decision"
            and value["checked_stages"] == ["confirmation", "auxiliary"]
            and value["plan_sha256"] == sha(module.OUT / "reproduction_plan.json")
            and value["source_sha256"] == specification["source_sha256"]
            and value["new_formal_acceptance_decision"] is None, "Required two-period reproduction is incomplete")
    return specification, value


def stage_binding(stage, shared, spatial_summary, log_summary):
    """Resolve only needed arrays/states from saved authoritative result maps."""
    evidence = {}
    if stage == "development":
        sr_path, lr_path = spatial_run.OLD_DEV, log_run.OLD_DEV
        sr = small(evidence, sr_path, spatial_run.OLD_DEV_SHA)
        lr = small(evidence, lr_path, log_run.OLD_DEV_SHA)
        require(sr["plan_sha256"] == spatial_run.R8_PLAN_SHA
                and lr["plan_sha256"] == log_run.R12_PLAN_SHA, "Original development method differs")
        sm = ROOT / "outputs/round8/models/development"
        lm = ROOT / "outputs/round12/models/development"
        sb = merge(sr["input_artifact_sha256"], sr["output_artifact_sha256"])
        lb = merge(lr["input_artifact_sha256"], lr["output_artifact_sha256"])
        spatial_prediction = sm / "predictions.npy"
        log_prediction = ROOT / "outputs/round12/evaluation/development/log_geography_blend.npy"
        small(evidence, sm / "metadata.json", sb[rel(sm / "metadata.json")])
    else:
        sr_path, lr_path = spatial_run.OUT / stage / "results.json", log_run.OUT / stage / "results.json"
        sr = small(evidence, sr_path, spatial_summary["stage_result_sha256"][stage])
        lr = small(evidence, lr_path, log_summary["stage_result_sha256"][stage])
        require(sr["status"] == lr["status"] == "completed"
                and sr["stage"] == lr["stage"] == stage
                and sr["plan_sha256"] == spatial_summary["plan_sha256"]
                and lr["plan_sha256"] == log_summary["plan_sha256"], "Reproduced stage differs")
        sm, lm = spatial_run.OUT / stage, log_run.OUT / stage / "model"
        sb, lb = sr["artifact_sha256"], lr["artifact_sha256"]
        spatial_prediction = sm / "predictions.npy"
        log_prediction = log_run.OUT / stage / (LOG + ".npy")
    meta = small(evidence, lm / "metadata.json", lb[rel(lm / "metadata.json")])
    require(meta["component"] == log_run.component.COMPONENT and meta["parameters"] == log_run.component.parameters()
            and meta["feature_count"] == 95 and meta["categorical_feature_count"] == 23
            and meta["reload_all_query_rows_verified"] is True and meta["query_labels_used"] is False
            and meta["fit_uses_eval_set"] is False and meta["early_stopping_used"] is False,
            "Saved log component contract/reload differs")
    require(sr["fit_rows"] == lr["fit_rows"] and sr["query_rows"] == lr["query_rows"], "Saved period rows differ")
    roles = {"spatial_state": rel(sm / "reference.pkl"), "spatial_state_summary": rel(sm / "state_summary.json"),
             "spatial_prediction": rel(spatial_prediction), "log_component": rel(lm / "predictions.npy"),
             "log_prediction": rel(log_prediction), "spatial_result": rel(sr_path), "log_result": rel(lr_path),
             "spatial_indices_folder": rel(sm), "log_indices_folder": rel(lm), "log_metadata": rel(lm / "metadata.json")}
    for path in (sm / "reference.pkl", sm / "state_summary.json", spatial_prediction,
                 sm / "fit_indices.npy", sm / "query_indices.npy"):
        name = rel(path)
        evidence[name] = sb[name]
    for path in (lm / "predictions.npy", log_prediction, lm / "fit_indices.npy", lm / "query_indices.npy"):
        name = rel(path)
        evidence[name] = lb[name]
    summary = small(evidence, sm / "state_summary.json", sb[rel(sm / "state_summary.json")], receipt=stage != "development")
    require(summary["parameters"] == spatial.parameters() and summary["fit_rows"] == sr["fit_rows"]
            and summary["query_used_in_build"] is False, "Saved spatial state contract differs")
    return roles, merge(shared, evidence)


def freeze():
    require(not (OUT / "combination_plan.json").exists(), "R36 plan already exists")
    require(sha(DOCUMENT) == DOCUMENT_SHA, "Predeclared method document changed")
    old_log, old_spatial = log_run.original_contract(), spatial_run.original_contract()
    require(old_log["target_group_weights"] == old_spatial["target_group_weights"]
            and old_log["split_contract"] == old_spatial["split_contract"], "Original methods use different partitions/weights")
    dependencies = {}
    dependencies[rel(DOCUMENT)] = sha(DOCUMENT)
    lp, log_summary = completed_reproduction(log_run, dependencies)
    sp, spatial_summary = completed_reproduction(spatial_run, dependencies)
    for module in (log_run, spatial_run):
        path = module.R12_PLAN if module is log_run else module.R8_PLAN
        small(dependencies, path)
    r6_path = c6.OUT / "candidate_plan.json"
    r6 = small(dependencies, r6_path, old_log["old_plan_sha256"])
    snapshot_path = ROOT / "runs/v3/snapshot_manifest.json"
    snapshot = small(dependencies, snapshot_path, old_log["v3_snapshot_sha256"])
    known = {n: v["original_sha256"] for n, v in snapshot["copied_files"].items()}
    known = merge(known, {v["original_path"]: v["sha256"] for v in snapshot["external_dependencies"]})
    require(known.get(rel(r6_path)) == sha(r6_path), "Original R6 plan not bound to v3")
    dependencies = merge(dependencies, r6["data_sha256"])
    bindings, stages = {}, {}
    for stage in STAGES:
        inherited = merge(old_log["stage_dependency_sha256"][stage], old_spatial["stage_dependency_sha256"][stage])
        bindings[stage], stages[stage] = stage_binding(stage, inherited, spatial_summary, log_summary)
    value = {"round": 36, "kind": "fixed_log_plus_spatial_combination_research", "created_at_utc": now(),
             "stage_order": list(STAGES), "methods": [V3, SPATIAL, LOG, COMBINED], "unique_new_combination": COMBINED,
             "formula": FORMULA, "spatial_parameters": spatial.parameters(), "log_parameters": log_run.component.parameters(),
             "source_sha256": sources(), "dependency_sha256": dependencies, "stage_dependency_sha256": stages,
             "bindings": bindings, "splits": copy.deepcopy(old_log["splits"]),
             "split_contract": copy.deepcopy(old_log["split_contract"]), "target_group_weights": old_log["target_group_weights"],
             "quarter_definition": old_log["quarter_definition"], "runtime": spatial_run.original.runtime_versions(),
             "R35_results_sha256": sha(log_run.OUT / "results.json"), "R33_results_sha256": sha(spatial_run.OUT / "results.json"),
             "protected_sha256": {n: sha(ROOT / n) for n in PROTECTED},
             "new_models_trained": 0, "tree_models_loaded": 0, "new_acceptance_thresholds": None,
             "formal_adoption_decision": None, "automatic_release_allowed": False,
             "all_three_periods_regardless_of_scores": True, "historically_untouched": False}
    require(value["source_sha256"] == sources() and known.get(rel(r6_path)) == sha(r6_path)
            and sha(DOCUMENT) == DOCUMENT_SHA, "Source/authority changed during freeze")
    locked_write(OUT / "combination_plan.json", value)
    print(json.dumps({"status": "combination_plan_frozen", "plan_sha256": sha(OUT / "combination_plan.json")}), flush=True)


def read_plan():
    value = locked(OUT / "combination_plan.json")
    require(value["round"] == 36 and value["stage_order"] == list(STAGES)
            and value["methods"] == [V3, SPATIAL, LOG, COMBINED] and value["formula"] == FORMULA
            and value["spatial_parameters"] == spatial.parameters() and value["log_parameters"] == log_run.component.parameters()
            and value["source_sha256"] == sources() and value["runtime"] == spatial_run.original.runtime_versions(), "Frozen combination changed")
    require(value["all_three_periods_regardless_of_scores"] is True
            and value["formal_adoption_decision"] is None and value["new_acceptance_thresholds"] is None
            and value["automatic_release_allowed"] is False, "Research scope changed")
    require(sha(DOCUMENT) == DOCUMENT_SHA and sha(log_run.OUT / "results.json") == value["R35_results_sha256"]
            and sha(spatial_run.OUT / "results.json") == value["R33_results_sha256"], "Required completed evidence changed")
    return value


def evaluate(stage, plan):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    binding = plan["bindings"][stage]
    data, X, y, Xq, yq, fi, qi = c6.partition(stage)
    del y
    weights = log_run.original.target_weights(data)
    require(weights == plan["target_group_weights"], "Original scoring weights differ")
    del data
    for name, values in (("fit", fi), ("query", qi)):
        require(log_run.original.value_sha(values, "<i8") == plan["split_contract"][stage][name + "_indices_content_sha256"], "Period index differs")
        save_array(folder / (name + "_indices.npy"), values)
    for name in ("spatial_indices_folder", "log_indices_folder"):
        e6.verify_indices(ROOT / binding[name], fi, qi, "query_indices.npy")
    e6.verify_indices(c6.baseline_folder(stage), fi, qi, "check_indices.npy")
    e6.verify_indices(c6.OUT / "models" / stage / "geography_catboost", fi, qi, "query_indices.npy")
    geo, geoq, _ = log_run.original.load_geographic_partition(stage, fi, qi)
    before = log_run.original.feature_hashes(X, Xq, geo, geoq)
    summary = c6.read(ROOT / binding["spatial_state_summary"])
    require(summary["fit_X_sha256"] == spatial.frame_sha(X) and summary["fit_geo_sha256"] == spatial.frame_sha(geo), "Spatial state fit frames differ")
    meta = locked(ROOT / binding["log_metadata"])
    require(meta["fit_original_feature_values_sha256"] == spatial.frame_sha(X)
            and meta["fit_geo_feature_values_sha256"] == spatial.frame_sha(geo), "Log fit frames differ")
    reference, reference_path = c6.load_reference(stage, X)
    with (ROOT / binding["spatial_state"]).open("rb") as stream:
        state = pickle.load(stream)
    require(state["parameters"] == spatial.parameters() and state["state_summary"] == summary
            and summary["existing_reference_residual_sha256"] == hashlib.sha256(np.asarray(reference["reference_residual"], dtype='<f8').tobytes()).hexdigest(),
            "Saved spatial state and period reference differ")
    base_folder = c6.baseline_folder(stage)
    old = e6.vector(base_folder / "catboost_raw.npy", len(qi))
    unit = e6.vector(base_folder / "lightgbm_l1_unit63.npy", len(qi))
    raw_geo = e6.vector(c6.OUT / "models" / stage / "geography_catboost/predictions.npy", len(qi))
    control = e6.vector(c6.OUT / "evaluation" / stage / "geography_blend.npy", len(qi))
    old_spatial = e6.vector(ROOT / binding["spatial_prediction"], len(qi))
    log_component = e6.vector(ROOT / binding["log_component"], len(qi))
    saved_log = e6.vector(ROOT / binding["log_prediction"], len(qi))
    v3_replay = log_run.original.blend(old, raw_geo, unit, reference, Xq)
    require(np.allclose(v3_replay, control, rtol=1e-12, atol=1e-5), "Original v3 control replay differs")
    base_log = .75 * (.5 * old + .5 * log_component) + .25 * unit
    p_log, diagnostics, reference_report = log_run.original.predict_reference(reference, Xq, base_log, variant="trend", return_diagnostics=True)
    require(np.allclose(p_log, saved_log, rtol=1e-12, atol=1e-5), "Saved log-only fusion replay differs")
    predictions = {V3: control, SPATIAL: old_spatial, LOG: p_log}
    metrics = {n: log_run.original.score(yq, p, X, Xq, c6.SPLITS[stage][0], weights) for n, p in predictions.items()}
    prior_spatial, prior_log = locked(ROOT / binding["spatial_result"]), locked(ROOT / binding["log_result"])
    prior_v3 = locked(c6.OUT / "evaluation" / stage / "results.json")
    require(metrics[V3] == prior_v3["metrics"]["geography_blend"]
            and metrics[SPATIAL] == prior_spatial["metrics"][spatial_run.CANDIDATE]
            and metrics[LOG] == prior_log["metrics"][log_run.CANDIDATE], "One of the three original metric records does not replay")
    combined, detail, prediction_report = spatial.predict(state, Xq, geoq, base_log, p_log, diagnostics)
    require(combined.dtype == np.dtype("float64") and combined.shape == (len(qi),)
            and np.isfinite(combined).all() and (combined >= 0).all(), "Invalid combination TWD output")
    fallback = ~detail["applied"].to_numpy(dtype=bool)
    require(combined[fallback].tobytes() == p_log[fallback].tobytes(), "Fallback must preserve log fusion, not original v3")
    predictions[COMBINED] = combined
    metrics[COMBINED] = log_run.original.score(yq, combined, X, Xq, c6.SPLITS[stage][0], weights)
    for name, values in predictions.items():
        save_array(folder / (name + ".npy"), values)
        restored = np.load(folder / (name + ".npy"), allow_pickle=False)
        require(restored.dtype == values.dtype and restored.tobytes() == values.tobytes(), "Saved prediction vector differs")
    with (folder / "diagnostics.pkl").open("xb") as stream:
        pickle.dump(detail, stream, protocol=5)
    locked_write(folder / "prediction_report.json", prediction_report)
    require(before == log_run.original.feature_hashes(X, Xq, geo, geoq), "Source feature frames changed")
    result = {"round": 36, "stage": stage, "status": "completed", "created_at_utc": now(),
              "plan_sha256": sha(OUT / "combination_plan.json"), "source_sha256": plan["source_sha256"],
              "input_sha256": plan["stage_dependency_sha256"][stage], "fit_rows": len(fi), "query_rows": len(qi),
              "dates": plan["splits"][stage], "metrics": metrics,
              "combination_vs_v3": log_run.differences(metrics[V3], metrics[COMBINED]),
              "combination_vs_spatial_v3": log_run.differences(metrics[SPATIAL], metrics[COMBINED]),
              "existing_three_metric_records_reproduced": True, "log_reference_replay_report": reference_report,
              "prediction_report": prediction_report, "reference_path": rel(reference_path),
              "reference_sha256": sha(reference_path), "input_frame_sha256": before,
              "fallback_baseline": LOG, "fallback_bitwise_preserved": True,
              "raw_base_and_future_clips_use_log_fusion": True, "reference_trend_applications": 1,
              "new_models_trained": 0, "tree_models_loaded": 0, "spatial_state_rebuilt": False,
              "formal_adoption_decision": None, "elapsed_seconds": time.monotonic() - started,
              "artifact_sha256": {rel(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    locked_write(folder / "results.json", result)
    print(json.dumps({"event": "period_completed", "stage": stage, "combination_vs_v3": result["combination_vs_v3"]}, ensure_ascii=False), flush=True)
    return result


def run():
    plan = read_plan()
    require(not any((OUT / s).exists() for s in STAGES) and not (OUT / "results.json").exists(), "Prior attempt exists; no automatic retry")
    locked_write(OUT / "run_claim.json", {"created_at_utc": now(), "plan_sha256": sha(OUT / "combination_plan.json"), "stage_order": list(STAGES)})
    protected = plan["protected_sha256"]
    record = {"round": 36, "status": "incomplete", "stage_order": list(STAGES), "checked_stages": [],
              "plan_sha256": sha(OUT / "combination_plan.json"), "source_sha256": plan["source_sha256"],
              "stage_result_sha256": {}, "stages": {}, "new_models_trained": 0, "tree_models_loaded": 0,
              "formal_adoption_decision": None, "new_submission_generated": False,
              "registry_updated": False, "recommended_version": "v3", "protected_before": protected}
    phase = "verify_inputs_before_deserialization"
    try:
        c6.verify_hashes(merge(plan["dependency_sha256"], *plan["stage_dependency_sha256"].values(), protected))
        for stage in STAGES:
            phase = stage
            result = evaluate(stage, plan)
            record["checked_stages"].append(stage)
            record["stage_result_sha256"][stage] = sha(OUT / stage / "results.json")
            record["stages"][stage] = {k: result[k] for k in ("metrics", "combination_vs_v3", "combination_vs_spatial_v3")}
        require(plan["source_sha256"] == sources(), "Source changed during research")
        record["protected_after"] = {n: sha(ROOT / n) for n in PROTECTED}
        require(record["protected_after"] == protected, "Prior evidence or submission changed")
        record["status"] = "three_periods_completed_no_adoption_decision"
    except Exception as exc:
        record.update(status="incomplete_technical_failure", failure_phase=phase,
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        record["protected_after"] = {n: sha(ROOT / n) for n in PROTECTED}
    record["completed_at_utc"] = now()
    locked_write(OUT / "results.json", record)
    print(json.dumps({"status": record["status"], "checked_stages": record["checked_stages"], "results_sha256": sha(OUT / "results.json")}), flush=True)
    return 0 if record["status"] == "three_periods_completed_no_adoption_decision" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        raise SystemExit(run())
