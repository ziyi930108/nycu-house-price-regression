"""R33: fixed R8 method replication on confirmation and auxiliary, no selection.

Freeze a separate research plan, then execute both fixed periods once. Reuse
the original helper and scorer; do not call R8 authorization or acceptance.
No development rerun, tree fitting, parameter search, CSV, or registry writes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import round8_experiment as original


c6, e6, component = original.c6, original.e6, original.component
ROOT = c6.ROOT
OUT = ROOT / "outputs/round33"
STAGES = ("confirmation", "auxiliary")
CONTROL, CANDIDATE = original.CONTROL, original.CANDIDATE
DOCUMENT = ROOT / "reports/round33_reproduction_plan.md"
R8_PLAN = ROOT / "outputs/round8/candidate_plan.json"
R8_PLAN_SHA = "ea32d46733371ffcef27946be47e40dd1c32ac5c4ab1019eaa31a65773c4e08d"
HELPER_SHA = "de86397255d654f363f3b746d579923fe689aa41d9dc16335d0922fea05e1c32"
OLD_DEV = ROOT / "outputs/round8/evaluation/development/results.json"
OLD_DEV_SHA = "840444a112e6c6cde618ce8f0840173640829483ca46f14775db8992499fe01f"
require, sha, rel, locked, locked_write = c6.require, c6.sha256, c6.rel, c6.read_locked, c6.locked_write
PROTECTED = (
    "outputs/submission_v2.csv", "outputs/submission_v3.csv", "runs/versions.json",
    "outputs/round8/candidate_plan.json", "outputs/round8/frozen_candidate.json",
    "outputs/round8/accepted_candidate.json",
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
    require(sha(R8_PLAN) == R8_PLAN_SHA, "Original R8 plan identity changed")
    value = locked(R8_PLAN)
    require(value["round"] == 8 and value["candidate"] == CANDIDATE
            and value["parameters"] == component.parameters(), "Original R8 method differs")
    require(sha(ROOT / "src/round8_spatial_reference.py") == HELPER_SHA
            and value["source_sha256"] == original.sources(), "Original R8 source closure changed")
    require(value["runtime"] == original.runtime_versions(), "Original R8 runtime differs")
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
    paths = [R8_PLAN, R8_PLAN.with_suffix(".sha256"), old_r6_path,
             old_r6_path.with_suffix(".sha256"), snapshot_path, snapshot_path.with_suffix(".sha256"),
             OLD_DEV, OLD_DEV.with_suffix(".sha256"), DOCUMENT]
    small = {rel(path): sha(path) for path in paths}
    shared = merge(r6["data_sha256"], small)
    stages = {s: copy.deepcopy(old["stage_dependency_sha256"][s]) for s in STAGES}
    # These expected binary hashes inherit the immutable R8 plan; run verifies
    # the bytes before loading them. No targets/arrays are loaded by freeze.
    merge(shared, *stages.values())
    value = {
        "round": 33, "kind": "fixed_R8_method_two_period_reproduction", "created_at_utc": now(),
        "stage_order": list(STAGES), "candidate": CANDIDATE, "control": CONTROL,
        "parameters": component.parameters(), "formulas": copy.deepcopy(original.FORMULAS),
        "source_sha256": sources(), "shared_dependency_sha256": shared,
        "stage_dependency_sha256": stages, "original_R8_plan_sha256": R8_PLAN_SHA,
        "original_R8_development_result_sha256": OLD_DEV_SHA,
        "splits": {s: list(c6.SPLITS[s]) for s in STAGES},
        "split_contract": {s: old["split_contract"][s] for s in STAGES},
        "target_group_weights": old["target_group_weights"], "quarter_definition": old["quarter_definition"],
        "runtime": original.runtime_versions(), "development_recomputed": False,
        "execute_both_regardless_of_first_metrics": True, "new_tree_models_trained": 0,
        "new_acceptance_thresholds": None, "formal_adoption_decision": None,
        "automatic_release_allowed": False, "query_labels_passed_to_component": False,
        "historically_untouched": False, "old_R8_and_R32_decisions_unchanged": True,
        "large_input_identity_at_freeze": "inherited immutable expected SHA; verified before run reads",
    }
    require(value["source_sha256"] == sources() and sha(R8_PLAN) == R8_PLAN_SHA
            and sha(old_r6_path) == old["old_plan_sha256"]
            and known.get(rel(old_r6_path)) == sha(old_r6_path), "Sources/contracts changed during freeze")
    locked_write(OUT / "reproduction_plan.json", value)
    print(json.dumps({"status": "research_plan_frozen", "plan_sha256": sha(OUT / "reproduction_plan.json")}), flush=True)


def read_plan():
    value = locked(OUT / "reproduction_plan.json")
    require(value["round"] == 33 and value["stage_order"] == list(STAGES)
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
            and value["automatic_release_allowed"] is False, "Replication scope changed")
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
    started = time.monotonic()
    data, X, y, Xq, yq, fi, qi = c6.partition(stage)
    del y
    weights = original.target_weights(data)
    require(weights == plan["target_group_weights"], "Original test-X scoring weights differ")
    del data
    for name, values in (("fit", fi), ("query", qi)):
        require(original.value_sha(values, "<i8") == plan["split_contract"][stage][name + "_indices_content_sha256"],
                "Canonical period index differs")
    e6.verify_indices(c6.baseline_folder(stage), fi, qi, "check_indices.npy")
    e6.verify_indices(c6.OUT / "models" / stage / "geography_catboost", fi, qi, "query_indices.npy")
    geo, geoq, _ = original.load_geographic_partition(stage, fi, qi)
    reference, reference_path = c6.load_reference(stage, X)
    raw_base, control = original.controls(stage, Xq, len(qi))
    diagnostics, control_report = original.replay_control(reference, Xq, raw_base, control)
    before = {k: original.frame_sha(f) for k, f in (("fit_X", X), ("query_X", Xq), ("fit_geo", geo), ("query_geo", geoq))}
    model, state = component.build(reference, X, geo)
    prediction, detail, prediction_report = component.predict(model, Xq, geoq, raw_base, control, diagnostics)
    require(prediction.shape == (len(qi),) and np.isfinite(prediction).all() and (prediction >= 0).all(),
            "Invalid reproduced TWD prediction")
    strong = ~diagnostics["source"].isin(["district_style", "base_fallback"]).to_numpy()
    require(prediction[strong].tobytes() == control[strong].tobytes(), "Strong-source fallback changed")
    with (folder / "reference.pkl").open("xb") as stream:
        pickle.dump(model, stream, protocol=5)
    locked_write(folder / "state_summary.json", state)
    locked_write(folder / "prediction_report.json", prediction_report)
    with (folder / "diagnostics.pkl").open("xb") as stream:
        pickle.dump(detail, stream, protocol=5)
    with (folder / "reference.pkl").open("rb") as stream:
        reloaded = pickle.load(stream)
    n = min(64, len(qi))
    replay, _, _ = component.predict(reloaded, Xq.iloc[:n], geoq.iloc[:n], raw_base[:n], control[:n], diagnostics.iloc[:n])
    require(np.allclose(replay, prediction[:n], rtol=1e-12, atol=1e-5), "R8 reference reload differs")
    for name, values in (("fit_indices", fi), ("query_indices", qi), ("v3_control", control), ("predictions", prediction)):
        save_array(folder / (name + ".npy"), values)
    saved = np.load(folder / "predictions.npy", allow_pickle=False)
    require(saved.dtype == prediction.dtype and saved.shape == prediction.shape
            and saved.tobytes() == prediction.tobytes(), "Saved TWD vector changed")
    metrics = {name: original.score(yq, values, X, Xq, c6.SPLITS[stage][0], weights)
               for name, values in ((CONTROL, control), (CANDIDATE, prediction))}
    require(metrics[CONTROL] == locked(c6.OUT / "evaluation" / stage / "results.json")["metrics"]["geography_blend"],
            "Original v3 control metrics do not replay")
    after = {k: original.frame_sha(f) for k, f in (("fit_X", X), ("query_X", Xq), ("fit_geo", geo), ("query_geo", geoq))}
    require(after == before, "Feature frames changed")
    result = {
        "round": 33, "stage": stage, "status": "completed", "created_at_utc": now(),
        "plan_sha256": sha(OUT / "reproduction_plan.json"), "source_sha256": plan["source_sha256"],
        "fit_rows": len(fi), "query_rows": len(qi), "dates": plan["splits"][stage],
        "frame_sha256": before, "original_reference_path": rel(reference_path),
        "original_reference_sha256": sha(reference_path), "input_sha256": plan["stage_dependency_sha256"][stage],
        "metrics": metrics, "changes": differences(metrics[CONTROL], metrics[CANDIDATE]),
        "prediction_report": prediction_report, "control_replay_report": control_report,
        "new_tree_models_trained": 0, "v2_reference_refitted": False, "query_labels_passed_to_component": False,
        "full_query_scored": True, "reload_rows_verified": n, "strong_sources_unchanged": True,
        "new_formal_acceptance_decision": None, "elapsed_seconds": time.monotonic() - started,
        "artifact_sha256": {rel(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    }
    locked_write(folder / "results.json", result)
    print(json.dumps({"event": "period_completed", "stage": stage, "changes": result["changes"]}, ensure_ascii=False), flush=True)
    return result


def run():
    plan = read_plan()
    require(not any((OUT / s).exists() for s in STAGES) and not (OUT / "results.json").exists(),
            "Prior attempt exists; preserve it and do not automatically retry")
    locked_write(OUT / "run_claim.json", {"created_at_utc": now(), "plan_sha256": sha(OUT / "reproduction_plan.json"),
                                         "stage_order": list(STAGES), "automatic_retry": False})
    protected = {name: sha(ROOT / name) for name in PROTECTED}
    record = {"round": 33, "kind": plan["kind"], "status": "incomplete", "stage_order": list(STAGES),
              "plan_sha256": sha(OUT / "reproduction_plan.json"), "source_sha256": plan["source_sha256"],
              "checked_stages": [], "stage_result_sha256": {}, "new_formal_acceptance_decision": None,
              "new_submission_generated": False, "registry_updated": False, "recommendation_changed": False,
              "development_recomputed": False, "historically_untouched": False, "protected_before": protected}
    phase = "verify_frozen_inputs"
    try:
        c6.verify_hashes(merge(plan["shared_dependency_sha256"], *plan["stage_dependency_sha256"].values()))
        historical = locked(OLD_DEV)
        require(sha(OLD_DEV) == OLD_DEV_SHA, "Historical R8 development identity changed")
        record["original_R8_development"] = {"path": rel(OLD_DEV), "sha256": OLD_DEV_SHA,
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
