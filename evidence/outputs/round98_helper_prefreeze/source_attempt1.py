"""Pure-numeric, unregularized convex three-branch L1 stacker for R98.

No dataset loader, native model, GPU, test, split selection or query target API.
Solve the equivalent three-row dual LP with fixed HiGHS dual simplex settings.
All three branch columns are already corrected TWD predictions, in N/H/U order.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback
import warnings

import numpy as np
import scipy
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csc_matrix
from scipy.optimize._highspy import _core

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "R98_convex_L1_stacker_v1"
BRANCH_ORDER = ["N", "H", "U"]
SCALE = 1e6
EXPECTED_SCIPY = "1.18.1"
EXPECTED_HIGHS = "1.12.0"
SOLVER_OPTIONS = {"presolve": True, "simplex_dual_edge_weight_strategy": "steepest-devex",
                  "threads": 1, "time_limit": 300.0, "maxiter": 1000000,
                  "primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8}
TOLERANCES = {"weight_sum_absolute": 1e-10, "weight_minimum": 0.0,
              "dual_box_absolute": 1e-8, "dual_constraint_violation_mean_TWD": .01,
              "stationarity_price_max_TWD": .10, "stationarity_free_variable_absolute": 1e-10,
              "objective_identity_mean_TWD": .01, "primal_dual_gap_mean_TWD": .10,
              "complementarity_mean_TWD": .10, "vertex_optimality_mean_TWD": .10}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "highs": ".".join(str(getattr(_core, "HIGHS_VERSION_" + k)) for k in ("MAJOR", "MINOR", "PATCH"))}


def parameters():
    return {"format": FORMAT, "branch_order": BRANCH_ORDER.copy(), "intercept": False,
            "nonnegative_weights": True, "weight_sum": 1.0, "objective": "equal-row TWD MAE",
            "common_price_scale_TWD": SCALE, "regularization": None, "time_decay": None,
            "formulation": "min -y_scaled@u-v; P_scaled.T@u+v<=0; -1<=u<=1; v free",
            "method": "highs-ds", "solver_options": SOLVER_OPTIONS.copy(),
            "required_scipy": EXPECTED_SCIPY, "required_highs": EXPECTED_HIGHS,
            "certificate_tolerances": TOLERANCES.copy(),
            "weight_extraction": "negative of three raw inequality marginals; no clipping or normalization",
            "multiple_optima": "accept solver solution for fixed runtime and supplied row order; no second objective",
            "automatic_retry": False, "fallback_weights": None,
            "prediction": "(N*w_N + H*w_H) + U*w_U; unrounded float64 TWD"}


def source_hashes():
    import scipy.optimize._linprog as lp
    import scipy.optimize._linprog_highs as highs
    import scipy.optimize._highspy._highs_wrapper as wrapper
    paths = [Path(__file__).resolve(), Path(lp.__file__), Path(highs.__file__), Path(wrapper.__file__), Path(_core.__file__)]
    return {str(p.resolve().relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def resource_estimate(rows):
    """Exact allocated CSC payload size; solver peak RAM and runtime are not measured."""
    need(isinstance(rows, int) and rows > 0 and 3 * (rows + 1) < np.iinfo(np.int32).max, "Unsupported LP dimensions")
    nnz = 3 * (rows + 1)
    return {"rows": rows, "variables": rows + 1, "inequality_rows": 3, "nonzeros": nnz,
            "CSC_float64_data_int32_indices_indptr_bytes": nnz * 12 + (rows + 2) * 4,
            "input_prediction_plus_target_float64_bytes": rows * 4 * 8,
            "solver_peak_RAM_measured": False, "solver_runtime_estimate_seconds": None,
            "caution": "Payload counts exclude solver workspaces, copies, Python overhead and certificate arrays; not a peak-RAM bound."}


def check_predictions(P):
    need(isinstance(P, np.ndarray) and P.dtype == np.float64 and P.ndim == 2
         and P.shape[1] == 3 and P.shape[0] > 0, "P must be nonempty float64 n-by-3 in N/H/U order")
    need(np.isfinite(P).all() and (P >= 0).all(), "Predictions must be finite nonnegative TWD")


def check_alignment(P, y, prediction_row_ids=None, target_row_ids=None):
    check_predictions(P)
    need(isinstance(y, np.ndarray) and y.dtype == np.float64 and y.shape == (len(P),)
         and np.isfinite(y).all() and (y >= 0).all(), "y must be an aligned finite nonnegative float64 TWD vector")
    need((prediction_row_ids is None) == (target_row_ids is None), "Both row-identity vectors must be supplied together")
    if prediction_row_ids is None:
        return {"row_ids_provided": False, "row_alignment_authority": "caller must bind P/y to the same ordered source manifest"}
    for a in (prediction_row_ids, target_row_ids):
        need(isinstance(a, np.ndarray) and a.dtype == np.int64 and a.shape == (len(P),)
             and (a >= 0).all() and len(np.unique(a)) == len(a), "Row IDs must be unique aligned nonnegative int64")
    need(prediction_row_ids.tobytes() == target_row_ids.tobytes(), "Prediction/target row identities or order differ")
    return {"row_ids_provided": True, "row_ids_values_sha256": values_sha(prediction_row_ids),
            "row_alignment_authority": "caller-supplied prediction and target row IDs matched exactly"}


def build_dual(P, y):
    """Three sparse rows rather than a dense n-by-n absolute-residual identity."""
    n = len(P)
    resource_estimate(n)
    ps, ys = P / SCALE, y / SCALE
    coefficients = np.empty(3 * (n + 1), dtype=np.float64)
    coefficients[:-3] = ps.reshape(-1)
    coefficients[-3:] = 1.0
    indices = np.tile(np.array([0, 1, 2], dtype=np.int32), n + 1)
    indptr = np.arange(n + 2, dtype=np.int32) * 3
    A = csc_matrix((coefficients, indices, indptr), shape=(3, n + 1), copy=False)
    c = np.r_[-ys, -1.0]
    bounds = np.empty((n + 1, 2), dtype=np.float64)
    bounds[:-1, 0], bounds[:-1, 1] = -1.0, 1.0
    bounds[-1, 0], bounds[-1, 1] = -np.inf, np.inf
    return ps, ys, A, c, np.zeros(3), bounds


def _array_record(a):
    return {"dtype": str(a.dtype), "shape": list(a.shape), "values_sha256": values_sha(a)}


def _solver_arrays(res):
    result = {"x": np.asarray(res.x, dtype=np.float64).copy()}
    for section in ("ineqlin", "lower", "upper"):
        for field in ("residual", "marginals"):
            result[section + "_" + field] = np.asarray(getattr(getattr(res, section), field), dtype=np.float64).copy()
    return result


def certificate(P, y, ps, ys, A, c, result, arrays):
    """Check raw solver primal/dual/KKT quantities in explicit fixed TWD tolerances."""
    n = len(P)
    z = arrays["x"]
    multipliers = arrays["ineqlin_marginals"]
    lower, upper = arrays["lower_marginals"], arrays["upper_marginals"]
    need(z.shape == (n + 1,) and multipliers.shape == (3,) and lower.shape == upper.shape == z.shape,
         "Solver vector schema differs")
    need(all(np.isfinite(a).all() for a in (z, multipliers, lower, upper)), "Nonfinite solver solution or marginals")
    w = -multipliers.copy()
    u, v = z[:-1], float(z[-1])
    prediction = (P[:, 0] * w[0] + P[:, 1] * w[1]) + P[:, 2] * w[2]
    primal = float(np.abs(prediction - y).mean())
    dual_sum = float(ys @ u + v)
    dual = dual_sum * SCALE / n
    slack = -(np.asarray(A @ z))
    stationarity = c - np.asarray(A.T @ multipliers) - lower - upper
    objective_mean_difference = abs(float(result.fun) + dual_sum) * SCALE / n
    box_violation = max(float(np.max(-1.0 - u)), float(np.max(u - 1.0)), 0.0)
    dual_violation = max(float(-slack.min()), 0.0) * SCALE / n
    row_complementarity = float(np.abs(multipliers * slack).sum()) * SCALE / n
    bound_complementarity = float(np.abs(lower[:-1] * (u + 1.0)).sum()
                                  + np.abs(upper[:-1] * (1.0 - u)).sum()) * SCALE / n
    native_mae = np.abs(P - y[:, None]).mean(axis=0)
    metrics = {"weights": w.tolist(), "weight_sum": float(w.sum()), "minimum_weight": float(w.min()),
               "weight_sum_error": abs(float(w.sum()) - 1.0), "primal_MAE_TWD": primal,
               "dual_objective_MAE_TWD": dual, "primal_minus_dual_MAE_TWD": primal - dual,
               "dual_box_violation": box_violation, "dual_constraint_violation_mean_TWD": dual_violation,
               "stationarity_price_max_TWD": float(np.max(np.abs(stationarity[:-1]))) * SCALE,
               "stationarity_free_variable_absolute": abs(float(stationarity[-1])),
               "objective_identity_mean_TWD": objective_mean_difference,
               "row_complementarity_mean_TWD": row_complementarity,
               "bound_complementarity_mean_TWD": bound_complementarity,
               "native_vertex_MAE_TWD": dict(zip(BRANCH_ORDER, native_mae.tolist())),
               "primal_minus_best_vertex_MAE_TWD": primal - float(native_mae.min()),
               "solver_dual_objective_sum_scaled": float(result.fun),
               "raw_dual_constraint_slacks": slack.tolist(),
               "saved_solver_constraint_residual_max_difference": float(np.max(np.abs(slack - arrays["ineqlin_residual"]))) }
    checks = {"nonnegative_raw_weights": metrics["minimum_weight"] >= TOLERANCES["weight_minimum"],
              "weight_sum_one": metrics["weight_sum_error"] <= TOLERANCES["weight_sum_absolute"],
              "dual_box": box_violation <= TOLERANCES["dual_box_absolute"],
              "dual_constraints": dual_violation <= TOLERANCES["dual_constraint_violation_mean_TWD"],
              "stationarity_prices": metrics["stationarity_price_max_TWD"] <= TOLERANCES["stationarity_price_max_TWD"],
              "stationarity_free_variable": metrics["stationarity_free_variable_absolute"] <= TOLERANCES["stationarity_free_variable_absolute"],
              "objective_identity": objective_mean_difference <= TOLERANCES["objective_identity_mean_TWD"],
              "primal_dual_gap": abs(primal - dual) <= TOLERANCES["primal_dual_gap_mean_TWD"],
              "row_complementarity": row_complementarity <= TOLERANCES["complementarity_mean_TWD"],
              "bound_complementarity": bound_complementarity <= TOLERANCES["complementarity_mean_TWD"],
              "vertex_optimality": metrics["primal_minus_best_vertex_MAE_TWD"] <= TOLERANCES["vertex_optimality_mean_TWD"],
              "marginal_signs": bool((multipliers <= 0).all() and (lower >= -1e-8).all() and (upper <= 1e-8).all()),
              "free_bound_marginals_zero": abs(float(lower[-1])) <= 1e-10 and abs(float(upper[-1])) <= 1e-10}
    metrics.update(checks=checks, all_checks_passed=all(checks.values()), tolerance_parameters=TOLERANCES.copy(),
                   numerical_certificate_only=True, weights_modified=False)
    return w, metrics


class StackerFitError(RuntimeError):
    """Carries numerical solver evidence so the caller can preserve a failed attempt."""
    def __init__(self, message, metadata, solver_arrays=None):
        super().__init__(message)
        self.metadata = metadata
        self.solver_arrays = solver_arrays or {}


def fit_stacker(P, y, *, prediction_row_ids=None, target_row_ids=None):
    """Return (JSON state, solver arrays, JSON metadata); never load data or retry."""
    alignment = check_alignment(P, y, prediction_row_ids, target_row_ids)
    rt = runtime()
    need(rt["scipy"] == EXPECTED_SCIPY and rt["highs"] == EXPECTED_HIGHS, "Frozen SciPy/HiGHS version differs")
    inputs = {"P": _array_record(P), "y": _array_record(y), **alignment}
    sources = source_hashes()
    tick = time.monotonic()
    ps, ys, A, c, b, bounds = build_dual(P, y)
    build_seconds = time.monotonic() - tick
    solve_tick = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds", options=SOLVER_OPTIONS.copy())
    solver_seconds = time.monotonic() - solve_tick
    warnings_record = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
    metadata = {"format": FORMAT, "status": "solver_returned", "fit_rows": len(P), "parameters": parameters(),
                "runtime": rt, "source_sha256": sources, "input_arrays": inputs,
                "resource_estimate": resource_estimate(len(P)), "build_seconds": build_seconds,
                "solver_seconds": solver_seconds, "solver_result": {"status": int(result.status),
                "success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
                "crossover_iterations": int(result.crossover_nit), "fun": float(result.fun) if result.fun is not None else None},
                "warnings": warnings_record, "price_unit": "TWD", "rows_equal_weight": True,
                "query_targets_available": False, "native_model_fit_load_predict_calls": 0, "GPU_calls": 0,
                "weights_clipped_normalized_or_regularized": False, "automatic_retry": False}
    if not result.success or result.status != 0:
        metadata.update(status="failed_solver", elapsed_seconds=time.monotonic() - tick)
        raise StackerFitError("LP did not reach an optimal solution; no fallback/retry", metadata,
                              {"x": np.asarray(result.x).copy()} if result.x is not None else {})
    arrays = _solver_arrays(result)
    w, proof = certificate(P, y, ps, ys, A, c, result, arrays)
    metadata["certificate"] = proof
    if not proof["all_checks_passed"]:
        metadata.update(status="failed_certificate", elapsed_seconds=time.monotonic() - tick)
        raise StackerFitError("LP numerical certificate failed; weights not repaired", metadata, arrays)
    need(values_sha(P) == inputs["P"]["values_sha256"] and values_sha(y) == inputs["y"]["values_sha256"], "Caller input changed during fit")
    need(source_hashes() == sources, "Stacker/solver source changed during fit")
    state = {"format": FORMAT, "parameters": parameters(), "weights": w.tolist(), "branch_order": BRANCH_ORDER.copy(),
             "fit_rows": len(P), "input_arrays": inputs, "runtime": rt, "source_sha256": sources,
             "solver_status": 0, "certificate_passed": True, "primal_MAE_TWD": proof["primal_MAE_TWD"],
             "certificate_gap_MAE_TWD": proof["primal_minus_dual_MAE_TWD"], "query_dependent_state": False}
    check_state(state)
    metadata.update(status="completed", elapsed_seconds=time.monotonic() - tick,
                    solver_arrays={k: _array_record(a) for k, a in arrays.items()}, completed_at_utc=now())
    return state, arrays, metadata


def check_state(state):
    need(isinstance(state, dict) and state.get("format") == FORMAT and state.get("parameters") == parameters(), "Unknown/changed stacker state")
    need(state.get("branch_order") == BRANCH_ORDER and state.get("solver_status") == 0
         and state.get("certificate_passed") is True and state.get("query_dependent_state") is False, "Invalid fitted state contract")
    w = np.asarray(state.get("weights"), dtype=np.float64)
    need(w.shape == (3,) and np.isfinite(w).all() and (w >= 0).all()
         and abs(float(w.sum()) - 1.0) <= TOLERANCES["weight_sum_absolute"], "Invalid raw convex weights")
    need(state["runtime"]["scipy"] == EXPECTED_SCIPY and state["runtime"]["highs"] == EXPECTED_HIGHS, "State solver identity differs")
    return w


def predict(state, P):
    """Fixed ordered arithmetic; no price target, fit state update or coefficient repair."""
    w = check_state(state)
    check_predictions(P)
    values = (P[:, 0] * w[0] + P[:, 1] * w[1]) + P[:, 2] * w[2]
    need(values.dtype == np.float64 and np.isfinite(values).all() and (values >= 0).all(), "Invalid linear predictions")
    return values


def safe(path):
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    need(p.is_relative_to(ROOT), "Artifact outside workspace")
    return p


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def load_state(path):
    state = json.loads(safe(path).read_text(encoding="utf-8-sig"))
    check_state(state)
    return state


def save_artifacts(out, state, solver_arrays, metadata):
    """Persist solver evidence and compact model; caller verifies its full query reload."""
    out = safe(out)
    need(not out.exists(), "Output exists; preserve old attempt")
    check_state(state)
    need(metadata["status"] == "completed" and metadata["certificate"]["all_checks_passed"], "Cannot save an uncertified model as success")
    need(state["weights"] == metadata["certificate"]["weights"] and state["input_arrays"] == metadata["input_arrays"], "State/fit identity differs")
    need(set(solver_arrays) == set(metadata["solver_arrays"]), "Solver array schema differs")
    need(all(_array_record(a) == metadata["solver_arrays"][k] for k, a in solver_arrays.items()), "Solver evidence changed")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "state.json", state)
    write_json(out / "fit_metadata.json", metadata)
    with (out / "solver_arrays.npz").open("xb") as f:
        np.savez_compressed(f, **solver_arrays)
    restored = load_state(out / "state.json")
    need(restored == state, "State JSON reload differs")
    with np.load(out / "solver_arrays.npz", allow_pickle=False) as loaded:
        need(set(loaded.files) == set(solver_arrays) and all(loaded[k].dtype == solver_arrays[k].dtype
             and loaded[k].shape == solver_arrays[k].shape and loaded[k].tobytes() == solver_arrays[k].tobytes()
             for k in solver_arrays), "Solver evidence reload differs")
    receipt = {"status": "saved_model_and_solver_evidence", "state_reload_exact": True,
               "solver_arrays_reload_exact": True, "full_query_prediction_reload_checked_by_caller": False,
               "artifact_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "receipt.json", receipt)
    return receipt


def self_check():
    """Artificial mathematical and API checks; no true arrays, targets, fits or GPU."""
    start = time.monotonic()
    checks, fits = [], []
    def mark(name, ok):
        need(bool(ok), name)
        checks.append(name)
    def rejects(name, f):
        try:
            f()
        except (ValueError, FileExistsError):
            checks.append(name)
            return
        raise AssertionError("Expected rejection: " + name)
    def fit_case(name, P, y, weights=None, objective=None):
        ids = np.arange(len(y), dtype=np.int64) + 100
        state, arrays, meta = fit_stacker(P, y, prediction_row_ids=ids, target_row_ids=ids.copy())
        fits.append({"name": name, "rows": len(P), "weights": state["weights"],
                     "solver_seconds": meta["solver_seconds"], "certificate": meta["certificate"], "warnings": meta["warnings"]})
        mark(name + "_certificate", meta["certificate"]["all_checks_passed"])
        if weights is not None:
            mark(name + "_known_optimal_weights", np.allclose(state["weights"], weights, rtol=0, atol=1e-10))
        if objective is not None:
            mark(name + "_known_L1_optimum", abs(meta["certificate"]["primal_MAE_TWD"] - objective) <= .10)
        return state, arrays, meta
    P = np.eye(3, dtype=np.float64) * SCALE
    y = np.array([.2, .3, .5], dtype=np.float64) * SCALE
    original_P, original_y = P.copy(), y.copy()
    state, arrays, meta = fit_case("unique_interior", P, y, [.2, .3, .5], 0.)
    mark("caller_inputs_unchanged", P.tobytes() == original_P.tobytes() and y.tobytes() == original_y.tobytes())
    fit_case("boundary_vertex", P, np.array([1., 0., 0.]) * SCALE, [1., 0., 0.], 0.)
    outlier_P = np.tile(np.array([0., 1., 1.]), (5, 1)) * SCALE
    fit_case("L1_not_squared_error_with_outlier", outlier_P, np.array([0., 0., 0., 0., 100.]) * SCALE, [1., 0., 0.], 20. * SCALE)
    same = np.tile(np.arange(1., 6.)[:, None], (1, 3)) * SCALE
    tied, _, _ = fit_case("nonunique_identical_branches", same, np.arange(1., 6.) * SCALE, objective=0.)
    mark("nonunique_no_forced_weight_tiebreak", np.array_equal(predict(tied, same), same[:, 0]))
    median_P = np.array([[0., 0., 3.], [0., 3., 0.], [3., 0., 0.]]) * SCALE
    _, _, median_meta = fit_case("median_can_outperform_all_global_weights", median_P, np.zeros(3), objective=SCALE)
    mark("no_claim_L1_stacker_beats_rowwise_median", np.median(median_P, axis=1).mean() == 0 and median_meta["certificate"]["primal_MAE_TWD"] > 0)
    shifted, _, shifted_meta = fit_case("common_large_TWD_offset", P + 1e9, y + 1e9, [.2, .3, .5], 0.)
    mark("common_offset_no_intercept_needed", np.allclose(shifted["weights"], state["weights"], rtol=0, atol=1e-10))
    query = np.array([[7., 5., 9.], [2., 20., 1.], [0., 0., 0.]]) * SCALE
    before = json.dumps(state, sort_keys=True)
    predicted = predict(state, query)
    mark("predict_uses_ordered_nonnegative_convex_weights", np.allclose(predicted, (query[:, 0] * .2 + query[:, 1] * .3) + query[:, 2] * .5, rtol=0, atol=.01))
    mark("query_changes_do_not_modify_fitted_state", json.dumps(state, sort_keys=True) == before and np.array_equal(predict(state, query[::-1]), predicted[::-1]))
    ids = np.array([3, 7, 5], dtype=np.int64)
    mark("matching_unsorted_source_indices_allowed", check_alignment(P, y, ids, ids.copy())["row_ids_provided"])
    rejects("row_order_misalignment_rejected", lambda: check_alignment(P, y, ids, ids[::-1]))
    rejects("duplicate_row_ids_rejected", lambda: check_alignment(P, y, np.array([1, 1, 2]), np.array([1, 1, 2])))
    rejects("only_one_row_identity_rejected", lambda: check_alignment(P, y, ids, None))
    rejects("target_length_misalignment_rejected", lambda: check_alignment(P, y[:-1]))
    rejects("nonfinite_predictions_rejected", lambda: check_alignment(np.full((3, 3), np.nan), y))
    rejects("nonfinite_target_rejected", lambda: check_alignment(P, np.full(3, np.inf)))
    rejects("wrong_branch_count_rejected", lambda: check_alignment(P[:, :2], y))
    rejects("implicit_float32_conversion_rejected", lambda: check_alignment(P.astype(np.float32), y))
    rejects("negative_prediction_rejected", lambda: predict(state, -np.ones((1, 3))))
    rejects("nonconvex_reloaded_state_rejected", lambda: predict({**state, "weights": [-.1, .3, .8]}, query))
    rejects("unnormalized_reloaded_state_rejected", lambda: predict({**state, "weights": [.2, .3, .6]}, query))
    with tempfile.TemporaryDirectory(prefix="round98_synthetic_", dir=ROOT / "outputs") as folder:
        out = Path(folder) / "model"
        receipt = save_artifacts(out, state, arrays, meta)
        reload_prediction = predict(load_state(out / "state.json"), query)
        mark("full_artificial_query_json_reload_prediction_bitwise", reload_prediction.tobytes() == predicted.tobytes())
        mark("solver_evidence_full_NPZ_reload", receipt["solver_arrays_reload_exact"])
        rejects("saved_model_refuses_overwrite", lambda: save_artifacts(out, state, arrays, meta))
    estimate = resource_estimate(186745)
    mark("sparse_three_constraints_linear_memory", estimate["inequality_rows"] == 3 and estimate["variables"] == 186746 and estimate["nonzeros"] == 560238)
    return {"status": "passed", "synthetic_only": True, "checks": checks, "check_count": len(checks),
            "synthetic_LP_fits": fits, "synthetic_fit_count": len(fits), "real_arrays_or_labels_loaded": 0,
            "real_meta_fits": 0, "GPU_calls": 0, "parameters": parameters(), "runtime": runtime(),
            "source_sha256": sha(__file__), "source_hashes": source_hashes(),
            "max_planned_meta_rows_resource_estimate": estimate, "elapsed_seconds": time.monotonic() - start,
            "completed_at_utc": now()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = safe(args.out)
    need(not out.exists(), "Selfcheck receipt exists; preserve prior attempt")
    try:
        result = self_check()
    except Exception as exc:
        result = {"status": "failed", "synthetic_only": True, "failure": traceback.format_exc(),
                  "source_sha256": sha(__file__), "parameters": parameters(), "real_meta_fits": 0, "GPU_calls": 0}
        if isinstance(exc, StackerFitError):
            result["solver_failure_metadata"] = exc.metadata
        write_json(out, result)
        raise
    write_json(out, result)
    print(json.dumps({"status": result["status"], "checks": result["check_count"],
                      "synthetic_LP_fits": result["synthetic_fit_count"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
