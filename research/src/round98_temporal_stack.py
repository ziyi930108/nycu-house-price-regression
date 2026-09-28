"""Fit one frozen R98 temporal stacker in an isolated stage process.

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
import round98_contract as c
import round98_l1_stacker as helper

PREPARATION = "outputs/round98_inputs/manifest.json"
REQUEST = "outputs/round98_launcher_20260929/research_request.json"
OUT = c.ROOT / "outputs/round98"


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
        c.need(request["preparation"]["path"] == PREPARATION and request["preparation"]["sha256"] == c.sha(PREPARATION), "Prepared input identity")
        prep = c.locked(PREPARATION)
        c.need(prep["status"] == "prepared_temporal_meta_inputs" and prep["plan_sha256"] == c.PLAN_SHA
               and prep["all_arrays_saved_reload_exact"] is True, "Inputs not prepared")
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
        c.write(out / "fit_started.json", {"stage": stage, "started_at_utc": c.now(), "new_native_fits": 0,
                "meta_rows": len(y), "query_rows": len(Q), "query_labels_available": False,
                "request_sha256": request_sha, "source_sha256": sources})
        fit_tick = time.monotonic(); fit_started = True
        state, solver_arrays, metadata = helper.fit_stacker(P, y, prediction_row_ids=ids, target_row_ids=ids.copy())
        fit_seconds = time.monotonic() - fit_tick
        pred = helper.predict(state, Q)
        c.need(pred.dtype == np.float64 and pred.shape == (n,) and np.isfinite(pred).all() and (pred >= 0).all(), "Invalid candidate predictions")
        helper.save_artifacts(out / "model", state, solver_arrays, metadata)
        state_path = out / "model/state.json"
        restored_state = helper.load_state(state_path)
        restored = helper.predict(restored_state, Q)
        c.exact(pred, restored, "All-query reloaded weights prediction differs")
        pred_rec = c.save_array(out / "predictions.npy", pred)
        control_rec = c.save_array(out / "control_v22.npy", arrays["control_v22"])
        c.need(arrays_before == {k: c.values_sha(a) for k, a in arrays.items()}, "Caller arrays mutated")
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
                "predictions": {"v22": control_rec, "R98": pred_rec}, "input_arrays": spec["arrays"],
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
