"""Fixed-round, fit-date-only CatBoost recency candidate for frozen year backtests.

The development/confirmation dates come exclusively from round3_backtest.SPLITS.
No evaluation set, early stopping, or query labels enter model fitting. Predictions
are this one raw-price component; the parent evaluator owns any 50/50 blend.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model_common import ROOT, load_data, temporal_indices, full_training_data, mae
from prepare_data import sha256, ID
from round3_backtest import SPLITS
from round2_catboost import fit_final_model, predict_raw, guard_shared_cache

OUT = ROOT / "outputs/round3/recency"
CANDIDATE = "catboost_recency2y_d8"
ITERATIONS = 2800
DEPENDENCIES = ("src/round2_catboost.py", "src/train_catboost.py", "src/model_common.py", "src/round3_backtest.py")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def frame_hash(frame):
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(rows.tobytes())
    return digest.hexdigest()


def date_bounds(frame):
    days = frame["trade_day_index"].to_numpy(dtype="float64")
    finite = days[np.isfinite(days)]
    def date(day):
        return (pd.Timestamp("2000-01-01") + pd.Timedelta(days=float(day))).date().isoformat()
    return {"minimum": date(finite.min()) if len(finite) else None,
            "maximum": date(finite.max()) if len(finite) else None,
            "nonfinite_rows": int((~np.isfinite(days)).sum())}


def run(split, overwrite=False):
    frozen_path = ROOT / "outputs/round3/frozen_candidates.json"
    if split == "confirmation" and not frozen_path.is_file():
        raise RuntimeError("Freeze candidates and acceptance rules before confirmation")
    folder = OUT / split
    if not overwrite and any((folder / name).exists() for name in ("model.cbm", "predictions.npy", "metadata.json")):
        raise FileExistsError(f"Completed or partial artifacts exist in {folder}; use --overwrite for an intentional rerun")
    guard_shared_cache()
    d = load_data()
    if split in SPLITS:
        start, end = SPLITS[split]
        fit_indices, query_indices = temporal_indices(d, start, end=end)
        if np.intersect1d(fit_indices, query_indices).size or min(len(fit_indices), len(query_indices)) < 10000:
            raise ValueError("Invalid chronological backtest indices")
        Xfit = d["X_train"].iloc[fit_indices].reset_index(drop=True)
        yfit = d["y_train"][fit_indices]
        Xquery = d["X_train"].iloc[query_indices].reset_index(drop=True)
        yquery = d["y_train"][query_indices]
        fit_space = query_space = "deduplicated original train, zero-based row positions returned by load_data"
        cutoff = (pd.Timestamp(start) - pd.Timestamp("2000-01-01")).days
        if not float(Xfit["trade_day_index"].max()) < cutoff:
            raise ValueError("Fit data reaches the query period")
        baseline_folder = ROOT / "outputs/round3/baseline" / split
        for name, indices in (("fit_indices", fit_indices), ("check_indices", query_indices)):
            path = baseline_folder / f"{name}.npy"
            if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), indices):
                raise ValueError(f"Index mismatch with frozen baseline: {path}")
    else:
        start = end = None
        Xfit, yfit = full_training_data(d)
        raw_concat = pd.concat([d["raw_train"], d["raw_valid"]], ignore_index=True)
        keep = ~raw_concat.duplicated(subset=[c for c in raw_concat if c != ID]).to_numpy()
        fit_indices = np.flatnonzero(keep)
        expected = pd.concat([d["X_train"], d["X_valid"]], ignore_index=True).iloc[fit_indices].reset_index(drop=True)
        if not Xfit.equals(expected):
            raise ValueError("Final indices do not reproduce full_training_data")
        Xquery = d["X_test"]
        query_indices = np.arange(len(Xquery), dtype="int64")
        yquery = None
        fit_space = "concat(load_data deduplicated original train, original valid), before combined exact-row dedup"
        query_space = "original test, zero-based row positions"
    if list(Xfit.columns) != list(Xquery.columns):
        raise ValueError("Fit/query feature schema mismatch")
    script_hash = sha256(Path(__file__))
    dependency_hashes = {name: sha256(ROOT / name) for name in DEPENDENCIES}
    schema_hash = sha256(ROOT / "reports/feature_schema.json")
    feature_hash = sha256(ROOT / "src/house_features.py")
    fit_fingerprint, query_fingerprint = frame_hash(Xfit), frame_hash(Xquery)
    expected_anchor = float(np.nanmax(Xfit["trade_day_index"].to_numpy(dtype="float64")))
    folder.mkdir(parents=True, exist_ok=True)
    for name, indices in (("fit_indices", fit_indices), ("query_indices", query_indices)):
        path = folder / f"{name}.npy"
        if path.exists() and not overwrite:
            if not np.array_equal(np.load(path, allow_pickle=False), indices):
                raise ValueError(f"Existing index mismatch: {path}")
        else:
            np.save(path, indices.astype("int64"))
    began = time.perf_counter()
    print(json.dumps({"event": "started", "split": split, "candidate": CANDIDATE,
                      "iterations": ITERATIONS, "fit_rows": len(Xfit), "query_rows": len(Xquery),
                      "recency_anchor_day_index": expected_anchor, "eval_set_used": False}), flush=True)
    model, training = fit_final_model(Xfit, yfit, d["categorical_columns"], candidate=CANDIDATE, iterations=ITERATIONS)
    recency = training["recency"]
    if (training["iterations"] != ITERATIONS or recency["half_life_days"] != 730.0
            or recency["minimum_weight"] != 0.1
            or recency["anchor_day_index_from_training_only"] != expected_anchor):
        raise ValueError("The fixed candidate configuration changed")
    prediction = predict_raw(model, Xquery, d["categorical_columns"])
    if prediction.shape != (len(Xquery),) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid price predictions")
    model_path, prediction_path = folder / "model.cbm", folder / "predictions.npy"
    model.save_model(str(model_path))
    np.save(prediction_path, prediction)
    if dependency_hashes != {name: sha256(ROOT / name) for name in DEPENDENCIES} or sha256(Path(__file__)) != script_hash:
        raise RuntimeError("Source code changed while the model was fitting")
    known = set(Xfit["project_name"].astype(str)) - {"__MISSING__"}
    seen = Xquery["project_name"].isin(known).to_numpy()
    params = training["parameters"]
    fit_indices_hash = sha256(folder / "fit_indices.npy")
    query_indices_hash = sha256(folder / "query_indices.npy")
    metadata = {
        "schema_version": 1, "split": split, "candidate": CANDIDATE, "component": "catboost_recency_raw",
        "family": "catboost", "target_mode": "raw", "iterations": training["iterations"],
        "parameters": params, "params": params, "parameters_sha256": canonical_hash(params),
        "parameters_hash_serialization": "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False), UTF-8",
        "recency": recency, "fit_only_recency_anchor": {"day_index": expected_anchor,
            "date": (pd.Timestamp("2000-01-01") + pd.Timedelta(days=expected_anchor)).date().isoformat()},
        "start": start, "end_inclusive": end, "dates": {"fit": date_bounds(Xfit), "query": date_bounds(Xquery)},
        "fit_rows": len(Xfit), "query_rows": len(Xquery), "check_rows": len(Xquery), "training_rows": len(Xfit),
        "seen_rows": int(seen.sum()), "unseen_rows": int((~seen).sum()),
        "fit_indices_space": fit_space, "query_indices_space": query_space,
        "fit_indices_path": (folder / "fit_indices.npy").relative_to(ROOT).as_posix(),
        "query_indices_path": (folder / "query_indices.npy").relative_to(ROOT).as_posix(),
        "fit_indices_sha256": fit_indices_hash, "query_indices_sha256": query_indices_hash,
        "check_indices_sha256": query_indices_hash,
        "source_sha256": d["metadata"]["source_sha256"], "feature_source_sha256": feature_hash,
        "feature_schema_sha256": schema_hash, "script_sha256": script_hash,
        "dependency_script_sha256": dependency_hashes, "fit_feature_values_sha256": fit_fingerprint,
        "query_feature_values_sha256": query_fingerprint,
        "fit_target_values_sha256": hashlib.sha256(np.asarray(yfit, dtype="float64").tobytes()).hexdigest(),
        "feature_columns": list(Xfit.columns), "categorical_columns": list(d["categorical_columns"]),
        "model_path": model_path.relative_to(ROOT).as_posix(), "model_sha256": sha256(model_path),
        "prediction_path": prediction_path.relative_to(ROOT).as_posix(), "prediction_sha256": sha256(prediction_path),
        "fit_uses_eval_set": False, "early_stopping_used": False, "iterations_selected_from_confirmation": False,
        "test_labels_used": False, "external_property_data_used": False, "submission_written": False,
        "prediction_is_raw_component_only": True,
        "planned_parent_candidate": "50 percent frozen original raw CatBoost plus 50 percent this recency raw component; no blend selected or written by this script",
        "elapsed_seconds": time.perf_counter() - began,
    }
    if frozen_path.exists():
        metadata["frozen_candidates_sha256"] = sha256(frozen_path)
    if yquery is not None:
        metadata.update({"mae": mae(yquery, prediction),
            "seen_mae": mae(yquery[seen], prediction[seen]) if seen.any() else None,
            "unseen_mae": mae(yquery[~seen], prediction[~seen]) if (~seen).any() else None,
            "mean_signed_error_pred_minus_actual": float(np.mean(prediction - yquery))})
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: metadata.get(k) for k in ("split", "iterations", "fit_rows", "query_rows", "mae", "seen_mae", "unseen_mae", "elapsed_seconds", "prediction_sha256", "model_sha256")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=[*SPLITS, "final"], required=True)
    parser.add_argument("--overwrite", action="store_true", help="Intentionally replace this split's own recency artifacts")
    args = parser.parse_args()
    run(args.split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
