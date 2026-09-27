"""R35: fixed R12 method replication on confirmation and auxiliary, no selection.

Freeze a separate research plan, then execute both fixed periods once. Reuse
the original helper and scorer; do not call R12 authorization or acceptance.
One new fixed 2999-tree fit per period. No development rerun, parameter search,
CSV, automatic adoption, or registry writes. Query labels only enter scoring.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import round12_experiment as original


c6, e6, component = original.c6, original.e6, original.component
ROOT = c6.ROOT
OUT = ROOT / "outputs/round35"
STAGES = ("confirmation", "auxiliary")
CONTROL, CANDIDATE = original.CONTROL, original.CANDIDATE
DOCUMENT = ROOT / "reports/round35_reproduction_plan.md"
R12_PLAN = ROOT / "outputs/round12/candidate_plan.json"
R12_PLAN_SHA = "5b37d819e1fffb624b26740815fae10fb6597d5da84613026a6436132df3e25b"
HELPER_SHA = "e16fbe33928bd1a635e5de8bd717f223263847b718aef373fb933224b13d6c42"
OLD_DEV = ROOT / "outputs/round12/evaluation/development/results.json"
OLD_DEV_SHA = "e3a140a1b5d47ed8a40b4230fd16a5468b7474af2c2103b011276856914dc640"
require, sha, rel, locked, locked_write = c6.require, c6.sha256, c6.rel, c6.read_locked, c6.locked_write
PROTECTED = (
    "outputs/submission_v2.csv", "outputs/submission_v3.csv", "outputs/submission_v4_trial.csv", "runs/versions.json",
    "outputs/round6/candidate_plan.json", "outputs/round6/accepted_candidate.json",
    "outputs/round12/candidate_plan.json", "outputs/round12/frozen_candidate.json",
    "outputs/round12/accepted_candidate.json",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def merge(*mappings):
    result = {}
    for mapping in mappings:
        for name, digest in mapping.items():
            require(name not in result or result[name] == digest, "Conflicting source/input SHA: " + name)
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT.resolve()) and path.is_file(), "Missing/unsafe input: " + name)
            result[name] = digest
    return result


def sources():
    return merge(original.sources(), {rel(Path(__file__)): sha(Path(__file__))})


def original_contract():
    require(sha(R12_PLAN) == R12_PLAN_SHA, "Original R12 plan identity changed")
    value = locked(R12_PLAN)
    require(value["round"] == 12 and value["candidate"] == CANDIDATE
            and value["parameters"] == component.parameters(), "Original R12 method differs")
    require(sha(ROOT / "src/round12_log_geo.py") == HELPER_SHA
            and value["source_sha256"] == original.sources(), "Original R12 source closure changed")
    require(value["runtime"] == original.runtime_versions(), "Original R12 runtime differs")
    return value


def freeze():
    require(not (OUT / "reproduction_plan.json").exists(), "Research plan already exists")
    old = original_contract()
    old_r6_path = c6.OUT / "candidate_plan.json"
    snapshot_path = ROOT / "runs/v3/snapshot_manifest.json"
    require(sha(old_r6_path) == old["old_plan_sha256"]
            and sha(snapshot_path) == old["v3_snapshot_sha256"] == original.SNAPSHOT_SHA,
            "Original R6 plan or released v3 snapshot differs")
    r6, snapshot = locked(old_r6_path), locked(snapshot_path)
    known = {k: v["original_sha256"] for k, v in snapshot["copied_files"].items()}
    known.update({v["original_path"]: v["sha256"] for v in snapshot["external_dependencies"]})
    require(known.get(rel(old_r6_path)) == sha(old_r6_path), "R6 plan not bound to released v3")
    require(sha(OLD_DEV) == OLD_DEV_SHA, "Original development record changed")
    locked(OLD_DEV)
    paths = [R12_PLAN, R12_PLAN.with_suffix(".sha256"), old_r6_path,
             old_r6_path.with_suffix(".sha256"), snapshot_path, snapshot_path.with_suffix(".sha256"),
             OLD_DEV, OLD_DEV.with_suffix(".sha256"), DOCUMENT]
    small = {rel(path): sha(path) for path in paths}
    shared = merge(r6["data_sha256"], small)
    stages = {s: copy.deepcopy(old["stage_dependency_sha256"][s]) for s in STAGES}
    # These expected binary hashes inherit the immutable R12 plan; run verifies
    # the bytes before loading them. No targets/arrays are loaded by freeze.
    merge(shared, *stages.values())
    value = {
        "round": 35, "kind": "fixed_R12_method_two_period_reproduction", "created_at_utc": now(),
        "stage_order": list(STAGES), "candidate": CANDIDATE, "control": CONTROL,
        "parameters": component.parameters(), "formulas": copy.deepcopy(original.FORMULAS),
        "source_sha256": sources(), "shared_dependency_sha256": shared,
        "stage_dependency_sha256": stages, "original_R12_plan_sha256": R12_PLAN_SHA,
        "original_R12_development_result_sha256": OLD_DEV_SHA,
        "splits": {s: list(c6.SPLITS[s]) for s in STAGES},
        "split_contract": {s: old["split_contract"][s] for s in STAGES},
        "target_group_weights": old["target_group_weights"], "quarter_definition": old["quarter_definition"],
        "runtime": original.runtime_versions(), "development_recomputed": False,
        "execute_both_regardless_of_first_metrics": True, "new_tree_models_planned": 2,
        "feature_count": 95, "categorical_count": 23, "sample_weight": "uniform",
        "target_formula": component.TARGET_FORMULA, "inverse_formula": component.INVERSE_FORMULA,
        "feature_rules": copy.deepcopy(component.FEATURE_RULES),
        "shared_loader_deserializes_query_labels": True,
        "protected_sha256": {name: sha(ROOT / name) for name in PROTECTED},
        "new_acceptance_thresholds": None, "formal_adoption_decision": None,
        "automatic_release_allowed": False, "query_labels_passed_to_component": False, "fit_uses_eval_set": False, "early_stopping_used": False,
        "historically_untouched": False, "old_R12_and_other_decisions_unchanged": True,
        "large_input_identity_at_freeze": "inherited immutable expected SHA; verified before run reads",
    }
    require(value["source_sha256"] == sources() and sha(R12_PLAN) == R12_PLAN_SHA
            and sha(old_r6_path) == old["old_plan_sha256"]
            and known.get(rel(old_r6_path)) == sha(old_r6_path), "Sources/contracts changed during freeze")
    locked_write(OUT / "reproduction_plan.json", value)
    print(json.dumps({"status": "research_plan_frozen", "plan_sha256": sha(OUT / "reproduction_plan.json")}), flush=True)


def read_plan():
    value = locked(OUT / "reproduction_plan.json")
    require(value["round"] == 35 and value["stage_order"] == list(STAGES)
            and value["candidate"] == CANDIDATE and value["control"] == CONTROL,
            "Wrong reproduction identity")
    require(value["parameters"] == component.parameters() and value["formulas"] == original.FORMULAS
            and value["source_sha256"] == sources() and value["runtime"] == original.runtime_versions(),
            "Frozen method/source/runtime changed")
    old = original_contract()
    require(value["target_group_weights"] == old["target_group_weights"]
            and value["stage_dependency_sha256"] == {s: old["stage_dependency_sha256"][s] for s in STAGES}
            and value["split_contract"] == {s: old["split_contract"][s] for s in STAGES}
            and value["splits"] == {s: list(c6.SPLITS[s]) for s in STAGES}, "Original period contract changed")
    require(value["execute_both_regardless_of_first_metrics"] is True
            and value["new_acceptance_thresholds"] is None and value["formal_adoption_decision"] is None
            and value["automatic_release_allowed"] is False
            and value["new_tree_models_planned"] == 2 and value["feature_count"] == 95
            and value["categorical_count"] == 23 and value["sample_weight"] == "uniform"
            and value["target_formula"] == component.TARGET_FORMULA
            and value["inverse_formula"] == component.INVERSE_FORMULA
            and value["feature_rules"] == component.FEATURE_RULES, "Replication scope changed")
    return value


def differences(control, candidate):
    def item(a, b):
        return {"delta_TWD": None if a is None or b is None else b - a,
                "relative_change_pct": None if a is None or b is None or a == 0 else 100.0 * (b / a - 1.0),
                "direction": "no_samples" if a is None or b is None else "lower_MAE" if b < a else "higher_MAE" if b > a else "equal"}
    return {"overall": item(control["overall_mae"], candidate["overall_mae"]),
            "weighted": item(control["weighted_mae"], candidate["weighted_mae"]),
            "strata": {k: item(v["mae"], candidate["strata"][k]["mae"]) for k, v in control["strata"].items()},
            "forecast_quarters": {k: item(v["mae"], candidate["forecast_quarters"][k]["mae"])
                                  for k, v in control["forecast_quarters"].items()}}


def save_array(path, array):
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def reproduce(stage, plan):
    require(stage in STAGES, "Only the two registered periods are allowed")
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=False)
    model_folder = folder / "model"
    model_folder.mkdir()
    started = time.monotonic()
    data, X, y, Xq, yq, fi, qi = c6.partition(stage)
    # The shared loader reads all labels. Only y is supplied to fit_model;
    # yq is retained for the later full-query scorer, never an eval_set.
    cats = data["categorical_columns"]
    weights = original.target_weights(data)
    require(weights == plan["target_group_weights"], "Original test-X scoring weights differ")
    del data
    for name, values in (("fit", fi), ("query", qi)):
        require(original.value_sha(values, "<i8") == plan["split_contract"][stage][name + "_indices_content_sha256"],
                "Canonical period index differs")
        save_array(model_folder / (name + "_indices.npy"), values)
    e6.verify_indices(c6.baseline_folder(stage), fi, qi, "check_indices.npy")
    e6.verify_indices(c6.OUT / "models" / stage / "geography_catboost", fi, qi, "query_indices.npy")
    geo, geoq, _ = original.load_geographic_partition(stage, fi, qi)
    before = original.feature_hashes(X, Xq, geo, geoq)
    locked_write(folder / "fit_started.json", {
        "round": 35, "stage": stage, "event": "fit_started", "created_at_utc": now(),
        "plan_sha256": sha(OUT / "reproduction_plan.json"), "fit_rows": len(fi), "query_rows": len(qi),
        "fit_target_values_sha256": original.value_sha(y, "<f8"),
        "fit_indices_content_sha256": original.value_sha(fi, "<i8"),
        "query_indices_content_sha256": original.value_sha(qi, "<i8"),
        "source_sha256": plan["source_sha256"], "parameters": plan["parameters"],
        "query_labels_used_for_training": False, "automatic_retrain": False})
    print(json.dumps({"event": "fit_started", "stage": stage, "fit_rows": len(X), "query_rows": len(Xq)}), flush=True)
    model, state, metadata = component.fit_model(X, y, geo, cats)
    # Persist fitted artifacts before inference/reload/scoring. A later failure
    # must not require discarding this already trained model or silently refitting.
    model.save_model(str(model_folder / "model.cbm"))
    locked_write(model_folder / "feature_state.json", state)
    metadata["component_source_sha256"] = metadata["source_sha256"]
    metadata.update({"round": 35, "stage": stage, "event": "fit_completed_model_saved",
        "created_at_utc": now(), "plan_sha256": sha(OUT / "reproduction_plan.json"),
        "source_sha256": plan["source_sha256"], "outer_fit_rows": len(X), "query_rows": len(Xq),
        "fit_indices_content_sha256": original.value_sha(fi, "<i8"),
        "query_indices_content_sha256": original.value_sha(qi, "<i8"),
        "geo_manifest_sha256": sha(c6.OUT / "geography" / stage / "manifest.json"),
        "model_sha256": sha(model_folder / "model.cbm"),
        "feature_state_sha256": sha(model_folder / "feature_state.json"),
        "shared_loader_deserialized_query_labels": True, "query_labels_used_for_training": False,
        "reload_all_query_rows_verified": False, "price_scoring_completed": False, **before})
    locked_write(model_folder / "fit_metadata.json", metadata)
    require(metadata["parameters"] == plan["parameters"] and metadata["feature_count"] == 95
            and metadata["categorical_feature_count"] == 23 and metadata["iterations"] == 2999
            and metadata["sample_weight"] == "uniform" and metadata["feature_rules"] == component.FEATURE_RULES,
            "Fixed component contract differs")
    require(metadata["fit_target_values_sha256"] == metadata["fit_original_target_values_sha256"] == original.value_sha(y, "<f8")
            and metadata["fit_transformed_target_values_sha256"] == original.value_sha(component.transform_target(y), "<f8"),
            "Actual fit target identity differs")
    for name in ("weight_argument_passed_to_pool", "query_labels_used", "fit_uses_eval_set", "early_stopping_used",
                 "query_used_for_geographic_state", "target_used_for_geographic_state", "calibration_applied",
                 "extra_price_cap_applied", "reference_or_trend_applied_here"):
        require(metadata[name] is False, "Forbidden component behavior: " + name)
    print(json.dumps({"event": "fit_completed_model_saved", "stage": stage}), flush=True)
    component_prediction = component.predict_model(model, state, Xq, geoq, cats)
    z = original.raw_predict(model, state, Xq, geoq, cats)
    raw_report = original.raw_diagnostics(z, component_prediction)
    for name, values in (("predictions", component_prediction), ("raw_z", z)):
        save_array(model_folder / (name + ".npy"), values)
    del model
    reloaded = original.load_model(model_folder / "model.cbm")
    restored = locked(model_folder / "feature_state.json")
    require(restored == state and int(reloaded.tree_count_) == 2999, "Saved state/native iteration count differs")
    # One complete native query replay yields both raw z and its fixed inverse.
    # raw_predict checks the same fit-only state/schema/Pool as predict_model.
    replay_z = original.raw_predict(reloaded, restored, Xq, geoq, cats)
    replay = component.inverse_transform(replay_z)
    require(replay_z.dtype == z.dtype and replay_z.tobytes() == z.tobytes()
            and replay.dtype == component_prediction.dtype and replay.tobytes() == component_prediction.tobytes(),
            "Saved native model full-query raw/TWD replay differs")
    for name, values in (("predictions", component_prediction), ("raw_z", z)):
        saved = np.load(model_folder / (name + ".npy"), allow_pickle=False)
        require(saved.dtype == values.dtype and saved.shape == values.shape and saved.tobytes() == values.tobytes(),
                "Saved component vector differs")
    del reloaded
    metadata.update({"event": "complete_query_reload_verified", "actual_iterations": 2999,
        "reload_all_query_rows_verified": True, "reload_query_rows": len(Xq), "reload_exact_equality": True,
        "raw_z_reload_bitwise_equal": True, "raw_z_diagnostics": raw_report, "prediction_units": "TWD",
        "artifact_sha256": {rel(p): sha(p) for p in sorted(model_folder.iterdir()) if p.is_file()},
        "fit_and_reload_elapsed_seconds": time.monotonic() - started})
    locked_write(model_folder / "metadata.json", metadata)
    reference, reference_path = c6.load_reference(stage, X)
    base_folder = c6.baseline_folder(stage)
    old = e6.vector(base_folder / "catboost_raw.npy", len(qi))
    unit = e6.vector(base_folder / "lightgbm_l1_unit63.npy", len(qi))
    geographic = e6.vector(c6.OUT / "models" / stage / "geography_catboost/predictions.npy", len(qi))
    control = e6.vector(c6.OUT / "evaluation" / stage / "geography_blend.npy", len(qi))
    control_replay = original.blend(old, geographic, unit, reference, Xq)
    require(np.allclose(control, control_replay, rtol=1e-12, atol=1e-5), "Original v3 control does not replay")
    prediction = original.blend(old, component_prediction, unit, reference, Xq)
    for name, values in ((CONTROL, control), (CANDIDATE, prediction)):
        save_array(folder / (name + ".npy"), values)
    metrics = {name: original.score(yq, values, X, Xq, c6.SPLITS[stage][0], weights)
               for name, values in ((CONTROL, control), (CANDIDATE, prediction))}
    require(metrics[CONTROL] == locked(c6.OUT / "evaluation" / stage / "results.json")["metrics"]["geography_blend"],
            "Original v3 control metrics do not replay")
    require(before == original.feature_hashes(X, Xq, geo, geoq), "Feature frames changed")
    result = {"round": 35, "stage": stage, "status": "completed", "created_at_utc": now(),
        "plan_sha256": sha(OUT / "reproduction_plan.json"), "source_sha256": plan["source_sha256"],
        "fit_rows": len(fi), "query_rows": len(qi), "dates": plan["splits"][stage], "frame_sha256": before,
        "model_metadata_sha256": sha(model_folder / "metadata.json"), "model_sha256": sha(model_folder / "model.cbm"),
        "original_reference_path": rel(reference_path), "original_reference_sha256": sha(reference_path),
        "input_sha256": plan["stage_dependency_sha256"][stage], "target_group_weights": weights,
        "metrics": metrics, "changes": differences(metrics[CONTROL], metrics[CANDIDATE]),
        "new_tree_models_trained": 1, "v2_reference_refitted": False, "query_labels_passed_to_component": False,
        "full_query_scored": True, "reload_rows_verified": len(Xq), "raw_and_TWD_reload_bitwise_equal": True,
        "raw_z_diagnostics": raw_report, "candidate_component_separately_scored": False,
        "new_formal_acceptance_decision": None, "elapsed_seconds": time.monotonic() - started,
        "artifact_sha256": {rel(p): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}}
    locked_write(folder / "results.json", result)
    print(json.dumps({"event": "period_completed", "stage": stage, "changes": result["changes"]}, ensure_ascii=False), flush=True)
    return result


def run():
    plan = read_plan()
    require(not any((OUT / s).exists() for s in STAGES) and not (OUT / "results.json").exists(),
            "Prior attempt exists; preserve it and do not automatically retry")
    locked_write(OUT / "run_claim.json", {"created_at_utc": now(), "plan_sha256": sha(OUT / "reproduction_plan.json"),
                                         "stage_order": list(STAGES), "automatic_retry": False})
    protected = plan["protected_sha256"]
    record = {"round": 35, "kind": plan["kind"], "status": "incomplete", "stage_order": list(STAGES),
              "plan_sha256": sha(OUT / "reproduction_plan.json"), "source_sha256": plan["source_sha256"],
              "checked_stages": [], "stage_result_sha256": {}, "new_formal_acceptance_decision": None,
              "new_submission_generated": False, "registry_updated": False, "recommendation_changed": False,
              "development_recomputed": False, "historically_untouched": False, "protected_before": protected,
              "formal_acceptance": False, "recommended_version": "v3", "trial_submission_modified": False}
    phase = "verify_frozen_inputs"
    try:
        c6.verify_hashes(merge(plan["shared_dependency_sha256"], *plan["stage_dependency_sha256"].values(), protected))
        historical = locked(OLD_DEV)
        require(sha(OLD_DEV) == OLD_DEV_SHA, "Historical R12 development identity changed")
        record["original_R12_development"] = {"path": rel(OLD_DEV), "sha256": OLD_DEV_SHA,
            "metrics": historical["metrics"], "historical_decision_only": historical["decision"], "recomputed": False}
        record["stages"] = {}
        for stage in STAGES:  # No branch or gate depends on the first period's metrics.
            phase = stage
            result = reproduce(stage, plan)
            record["checked_stages"].append(stage)
            record["stage_result_sha256"][stage] = sha(OUT / stage / "results.json")
            record["stages"][stage] = {"metrics": result["metrics"], "changes": result["changes"]}
        require(plan["source_sha256"] == sources(), "Source changed during reproduction")
        record["protected_after"] = {name: sha(ROOT / name) for name in PROTECTED}
        require(record["protected_after"] == protected, "Protected old artifacts changed")
        record["new_tree_models_trained"] = 2
        record["status"] = "both_periods_completed_no_adoption_decision"
    except Exception as exc:
        record.update(status="incomplete_technical_failure", failure_phase=phase,
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        record["protected_after"] = {name: sha(ROOT / name) for name in PROTECTED}
    record["completed_at_utc"] = now()
    locked_write(OUT / "results.json", record)
    print(json.dumps({"status": record["status"], "checked_stages": record["checked_stages"],
                      "results_sha256": sha(OUT / "results.json")}), flush=True)
    return 0 if record["status"] == "both_periods_completed_no_adoption_decision" else 1


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
