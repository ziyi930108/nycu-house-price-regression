"""Fit one frozen R99 temporal stacker in an isolated stage process.

Reads its own pre-cutoff meta targets and saved forward matrices only. This
process never opens raw train prices, another stage's meta targets or query y.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
import numpy as np
import pandas as pd
import round99_contract as c
import round99_l1_stacker as helper

PREPARATION = c.PREPARATION
REQUEST = "outputs/round99_launcher_20260929/research_request.json"
OUT = c.ROOT / "outputs/round99"



def verify_saved_r98_meta(stage, P, y, ids, spec):
    """Read bound R98 JSON evidence only; never load another target or model."""
    c.need(c.sha(c.R98_SEALED_MANIFEST) == c.R98_SEALED_MANIFEST_SHA, "R98 root seal identity")
    sealed = c.locked(c.R98_SEALED_MANIFEST)
    c.need(sealed["status"] == "all_temporal_stack_predictions_sealed"
           and sealed["plan_sha256"] == c.PREPARATION_PLAN_SHA
           and sealed["preparation"] == {"path": c.PREPARATION, "sha256": c.PREPARATION_SHA},
           "R98 authority origin")
    c.verify_hashes(sealed["source_sha256"])
    record = sealed["stages"][stage]["stage_manifest"]
    c.need(c.sha(record["path"]) == record["sha256"], "R98 stage authority hash")
    old = c.locked(record["path"])
    c.need(old["stage"] == stage and old["status"] == "temporal_stack_stage_predictions_sealed"
           and old["input_arrays"] == spec["arrays"] and old["all_query_reload_exact"] is True,
           "R98 stage/meta provenance")
    paths = {"state": f"outputs/round98/{stage}/model/state.json",
             "metadata": f"outputs/round98/{stage}/model/fit_metadata.json",
             "receipt": f"outputs/round98/{stage}/model/receipt.json"}
    for path in paths.values():
        c.need(c.sha(path) == old["model_artifact_sha256"][path], "R98 model evidence hash")
    state, metadata, receipt = (c.read(paths[k]) for k in ("state", "metadata", "receipt"))
    c.need(receipt["status"] == "saved_model_and_solver_evidence"
           and receipt["state_reload_exact"] is True and receipt["solver_arrays_reload_exact"] is True,
           "R98 model evidence not saved")
    c.need(receipt["artifact_sha256"][paths["state"]] == c.sha(paths["state"])
           and receipt["artifact_sha256"][paths["metadata"]] == c.sha(paths["metadata"]),
           "R98 receipt differs from stage authority")
    c.need(state["format"] == "R98_convex_L1_stacker_v1" and state["branch_order"] == list(c.BRANCHES)
           and state["certificate_passed"] is True and state["query_dependent_state"] is False,
           "R98 baseline is not fixed three-branch fit")
    old_weights = np.asarray(state["weights"], dtype=np.float64)
    c.need(old_weights.shape == (3,) and np.isfinite(old_weights).all() and (old_weights >= 0).all()
           and abs(float(old_weights.sum()) - 1.0) <= 1e-10, "Saved R98 weights not feasible")
    expected = {"P": c.matrix_identity(P), "y": c.matrix_identity(y)}
    for k, identity in expected.items():
        c.need(state["input_arrays"][k] == metadata["input_arrays"][k] == identity, "R98 meta input differs")
    c.need(state["input_arrays"]["row_ids_values_sha256"] == metadata["input_arrays"]["row_ids_values_sha256"]
           == c.values_sha(ids), "R98 meta row order differs")
    c.need(state["fit_rows"] == len(y) and metadata["status"] == "completed"
           and metadata["certificate"]["all_checks_passed"] is True, "R98 saved fit not valid")
    c.verify_hashes(state["source_sha256"])
    objective = float(state["primal_MAE_TWD"])
    c.need(np.isfinite(objective) and objective == metadata["certificate"]["primal_MAE_TWD"],
           "R98 objective identity differs")
    return {"root_seal": {"path": c.R98_SEALED_MANIFEST, "sha256": c.R98_SEALED_MANIFEST_SHA},
            "stage_seal": record, "metadata_readset": {p: c.sha(p) for p in paths.values()},
            "same_original_meta_P_y_indices_verified": True, "primal_MAE_TWD": objective,
            "raw_or_query_targets_read": False, "native_model_load_calls": 0}


def run_stage(stage, request_sha):
    c.need(stage in c.STAGES, "Unknown stage")
    out = OUT / stage
    c.need(not out.exists(), "Stage already exists; no retry or overwrite")
    out.mkdir(parents=True)
    start, tick, code, failure = c.now(), time.monotonic(), 1, None
    fit_started = False
    try:
        request = c.locked(REQUEST)
        c.need(c.sha(REQUEST) == request_sha, "Unexpected launch request")
        c.need(request["status"] == "research_frozen_before_meta_fit" and request["plan_sha256"] == c.PLAN_SHA == c.sha(c.PLAN), "Frozen plan")
        c.need(request["stage_order"] == list(c.STAGES) and request["helper_parameters"] == helper.parameters(), "Frozen method")
        sources = c.source_hashes(__file__)
        c.need(request["driver_source_sha256"] == sources, "Driver import closure differs")
        c.verify_hashes(request["source_sha256"])
        c.need(request["preparation"]["path"] == PREPARATION and request["preparation"]["sha256"] == c.PREPARATION_SHA == c.sha(PREPARATION), "Prepared input identity")
        prep = c.locked(PREPARATION)
        c.need(prep["status"] == "prepared_temporal_meta_inputs" and prep["plan_sha256"] == c.PREPARATION_PLAN_SHA == c.sha(c.PREPARATION_PLAN)
               and prep["all_arrays_saved_reload_exact"] is True, "Inputs not prepared")
        c.verify_hashes(prep["source_sha256"])
        c.need(prep["branch_order"] == list(c.BRANCHES), "Original prepared branch order")
        c.need(c.locked(c.safe(PREPARATION).parent / "execution.json")["exit_code"] == 0, "Preparation failed")
        c.need(prep["protected_before"] == prep["protected_after"], "Protected preparation inputs changed")
        protected = c.protected_hashes()
        c.need(protected == request["protected_before"], "Selected outputs changed before stage")
        spec = prep["stages"][stage]
        cutoff, query_end, n = c.SPLITS[stage]
        c.need((spec["cutoff"], spec["query_end"], spec["query_rows"]) == (cutoff, query_end, n), "Stage dates/count")
        # No loops over other stage data: stage-local memory contains only legal labels.
        needed = ("meta_predictions", "meta_y", "meta_indices", "meta_source_cache_indices", "meta_trade_days",
                  "query_predictions", "query_indices", "query_source_cache_indices", "query_trade_days", "fit_indices", "control_v22")
        c.need(set(spec["arrays"]) == set(needed), "Unexpected stage array; query targets are forbidden")
        for key in needed:
            c.need(spec["arrays"][key]["path"] == f"outputs/round98_inputs/{stage}/{key}.npy", "Array belongs to another stage or role")
        arrays = {k: c.load_array(spec["arrays"][k]) for k in needed}
        P, y, Q = arrays["meta_predictions"], arrays["meta_y"], arrays["query_predictions"]
        ids, qids, fitids = arrays["meta_indices"], arrays["query_indices"], arrays["fit_indices"]
        c.need(P.dtype == y.dtype == Q.dtype == np.float64 and P.shape == (spec["meta_rows"], 3)
               and y.shape == (spec["meta_rows"],) and Q.shape == (n, 3), "Meta/query matrices")
        c.need(ids.dtype == qids.dtype == fitids.dtype == np.int64 and len(np.unique(ids)) == len(ids)
               and not np.intersect1d(ids, qids).size and np.isin(ids, fitids).all(), "Meta/query source separation")
        cut = (pd.Timestamp(cutoff) - pd.Timestamp("2000-01-01")).days
        end = (pd.Timestamp(query_end) - pd.Timestamp("2000-01-01")).days
        c.need(np.isfinite(arrays["meta_trade_days"]).all() and (arrays["meta_trade_days"] < cut).all(), "Future meta target")
        c.need(((arrays["query_trade_days"] >= cut) & (arrays["query_trade_days"] <= end)).all(), "Query origin")
        c.need(np.isfinite(P).all() and np.isfinite(Q).all() and (P >= 0).all() and (Q >= 0).all()
               and np.isfinite(y).all() and (y > 0).all(), "Finite nonnegative prices")
        c.exact(np.median(Q, axis=1), arrays["control_v22"], "v22 median control differs")
        arrays_before = {k: c.values_sha(a) for k, a in arrays.items()}
        P4, Q4 = c.stacker_matrix(P), c.stacker_matrix(Q)
        derived_meta, derived_query = c.matrix_identity(P4), c.matrix_identity(Q4)
        c.exact(Q4[:, 3], arrays["control_v22"], "Derived median control differs")
        old_meta = verify_saved_r98_meta(stage, P, y, ids, spec)
        c.write(out / "fit_started.json", {"stage": stage, "started_at_utc": c.now(), "new_native_fits": 0,
                "meta_rows": len(y), "query_rows": len(Q), "query_labels_available": False,
                "request_sha256": request_sha, "source_sha256": sources,
                "derived_meta_predictions": derived_meta, "derived_query_predictions": derived_query})
        fit_tick = time.monotonic(); fit_started = True
        state, solver_arrays, metadata = helper.fit_stacker(P4, y, prediction_row_ids=ids, target_row_ids=ids.copy())
        fit_seconds = time.monotonic() - fit_tick
        # Save numerical evidence before the extra feasible-baseline engineering check.
        helper.save_artifacts(out / "model", state, solver_arrays, metadata)
        c.need(state["input_arrays"]["P"] == derived_meta and metadata["input_arrays"]["P"] == derived_meta,
               "Fitted state must bind derived four-column meta matrix")
        current_objective = float(state["primal_MAE_TWD"])
        c.need(np.isfinite(current_objective) and current_objective == metadata["certificate"]["primal_MAE_TWD"],
               "Candidate objective evidence differs")
        c.need(current_objective <= old_meta["primal_MAE_TWD"] + c.OLD_META_OBJECTIVE_TOLERANCE_TWD,
               "Four-column meta objective worse than feasible saved R98 solution")
        pred = helper.predict(state, Q4)
        c.need(pred.dtype == np.float64 and pred.shape == (n,) and np.isfinite(pred).all() and (pred >= 0).all(), "Invalid candidate predictions")
        helper.save_artifacts(out / "model", state, solver_arrays, metadata)
        state_path = out / "model/state.json"
        restored_state = helper.load_state(state_path)
        restored = helper.predict(restored_state, Q4)
        c.exact(pred, restored, "All-query reloaded weights prediction differs")
        pred_rec = c.save_array(out / "predictions.npy", pred)
        control_rec = c.save_array(out / "control_v22.npy", arrays["control_v22"])
        c.need(arrays_before == {k: c.values_sha(a) for k, a in arrays.items()}, "Caller arrays mutated")
        c.need(c.matrix_identity(P4) == derived_meta and c.matrix_identity(Q4) == derived_query,
               "Derived matrices mutated")
        c.need(c.source_hashes(__file__) == sources and helper.parameters() == request["helper_parameters"], "Source/parameters changed")
        c.verify_hashes(request["source_sha256"])
        c.need(protected == c.protected_hashes(), "Protected artifacts changed")
        artifacts = {c.rel(p): c.sha(p) for p in sorted((out / "model").rglob("*")) if p.is_file()}
        seal = {"schema_version": 1, "status": "temporal_stack_stage_predictions_sealed", "stage": stage,
                "cutoff": cutoff, "query_end": query_end, "meta_rows": len(y), "query_rows": n,
                "query_labels_available": False, "new_meta_fits": 1, "new_native_fits": 0,
                "all_query_reload_exact": True, "query_scores_computed": False,
                "parameters": helper.parameters(), "model_state": {"path": c.rel(state_path), "sha256": c.sha(state_path)},
                "model_artifact_sha256": artifacts, "source_sha256": sources, "plan_sha256": c.PLAN_SHA,
                "preparation": {"path": PREPARATION, "sha256": c.sha(PREPARATION)},
                "launch_request": {"path": REQUEST, "sha256": request_sha},
                "predictions": {"v22": control_rec, "R99": pred_rec}, "input_arrays": spec["arrays"],
                "source_branch_order": list(c.BRANCHES), "stacker_branch_order": list(c.STACKER_BRANCHES),
                "derived_meta_predictions": derived_meta, "derived_query_predictions": derived_query,
                "original_preparation_plan_sha256": c.PREPARATION_PLAN_SHA,
                "r98_meta_authority": old_meta,
                "old_R98_meta_objective_TWD": old_meta["primal_MAE_TWD"],
                "R99_meta_objective_TWD": current_objective,
                "R99_not_worse_than_R98_meta_tolerance_TWD": c.OLD_META_OBJECTIVE_TOLERANCE_TWD,
                "R99_not_worse_than_R98_meta_verified": True,
                "actual_stage_data_readset": {r["path"]: r["sha256"] for r in spec["arrays"].values()},
                "raw_price_pickle_opened": False, "other_stage_meta_targets_opened": False,
                "caller_arrays_unchanged": True, "fit_call_elapsed_seconds": fit_seconds,
                "protected_before": protected, "protected_after": c.protected_hashes(), "selected_submission": "v22",
                "new_CSV": False, "started_at_utc": start, "completed_at_utc": c.now(),
                "elapsed_seconds": time.monotonic() - tick, "caution": c.CAUTION}
        c.write(out / "stage_seal.json", seal)
        code = 0
        print(json.dumps({"stage": stage, "meta_rows": len(y), "query_rows": n, "fit_call_seconds": fit_seconds,
                          "elapsed_seconds": seal["elapsed_seconds"], "all_query_reload_exact": True}))
    except Exception as exc:
        failure = traceback.format_exc()
        if isinstance(exc, helper.StackerFitError):
            c.write(out / "solver_failure.json", exc.metadata)
            if exc.solver_arrays:
                with (out / "solver_failure_arrays.npz").open("xb") as f:
                    np.savez(f, **exc.solver_arrays)
        raise
    finally:
        c.write(out / "execution.json", {"stage": stage, "exit_code": code, "failure": failure,
                "started_at_utc": start, "completed_at_utc": c.now(), "elapsed_seconds": time.monotonic() - tick,
                "meta_fit_started": fit_started, "query_labels_available": False, "new_native_fits": 0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=c.STAGES, required=True)
    ap.add_argument("--request-sha", required=True)
    args = ap.parse_args()
    run_stage(args.stage, args.request_sha)
