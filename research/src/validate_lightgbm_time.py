"""Evaluate the two LightGBM candidates on a later transaction period."""
from __future__ import annotations
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from model_common import load_data, temporal_indices
from train_lightgbm import encode_features, effective_area, price_factor, json_write

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"


def main():
    data = load_data()
    fit_indices, valid_indices = temporal_indices(data, cutoff="2024-04-01", end="2024-08-10")
    if len(fit_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("Temporal split is empty")
    frames = {
        "train": data["X_train"].iloc[fit_indices],
        "valid": data["X_train"].iloc[valid_indices],
    }
    encoded, maps, safe_names, cats = encode_features(frames, data["categorical_columns"])
    y_fit, y_valid = data["y_train"][fit_indices], data["y_train"][valid_indices]
    fit_projects = set(frames["train"]["project_name"]) - {"__MISSING__"}
    seen = frames["valid"]["project_name"].isin(fit_projects).to_numpy()
    for mode in ("raw", "unit"):
        original_name = f"lightgbm_l1_{mode}63"
        original = json.loads((EXPERIMENTS / f"{original_name}.json").read_text(encoding="utf-8"))
        name = f"{original_name}_temporal"
        model_dir = ROOT / "models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        params = dict(original["params"])
        params["n_estimators"] = min(4000, int(original["best_iteration"]))
        params["n_jobs"] = 4
        factors = {s:price_factor(frames[s], mode) for s in ("train", "valid")}
        weight = effective_area(frames["train"]) / 100 if mode == "unit" else None
        def price_mae(y_true, prediction):
            return "price_mae", float(np.abs((y_true - prediction) * factors["valid"]).mean()), False
        model = lgb.LGBMRegressor(**params)
        started = time.perf_counter()
        print(f"Starting {name}: {len(y_fit):,} fit / {len(y_valid):,} valid; "
              f"{int((~seen).sum()):,} unseen-project rows", flush=True)
        model.fit(
            encoded["train"], y_fit / factors["train"], sample_weight=weight,
            eval_set=[(encoded["valid"], y_valid / factors["valid"])],
            eval_metric=price_mae, categorical_feature=cats,
            callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=True),
                       lgb.log_evaluation(250)],
        )
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
        predictions = np.maximum(model.predict(encoded["valid"], num_iteration=best_iteration) * factors["valid"], 0)
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite temporal predictions")
        error = np.abs(y_valid - predictions)
        result = {
            "name": name, "family": "lightgbm", "target_mode": mode,
            "mae": float(error.mean()),
            "seen_project_mae": float(error[seen].mean()) if seen.any() else None,
            "unseen_project_mae": float(error[~seen].mean()) if (~seen).any() else None,
            "rows": len(y_valid), "fit_rows": len(y_fit),
            "seen_project_rows": int(seen.sum()), "unseen_project_rows": int((~seen).sum()),
            "median_baseline_mae": float(np.abs(y_valid - np.median(y_fit)).mean()),
            "best_iteration": best_iteration,
            "iteration_cap_from_original_validation": params["n_estimators"],
            "params": params, "cutoff": "2024-04-01", "end": "2024-08-10",
            "category_mapping_fit_split": "temporal_fit_only",
            "category_maps_path": str((model_dir / "category_maps.json").relative_to(ROOT)),
            "safe_feature_names": safe_names,
            "model_path": str((model_dir / "model.txt").relative_to(ROOT)),
            "elapsed_seconds": time.perf_counter() - started,
            "external_data": False,
            "source_sha256": data["metadata"]["source_sha256"],
        }
        np.save(EXPERIMENTS / f"{name}_valid.npy", predictions)
        np.save(EXPERIMENTS / f"{name}_indices.npy", valid_indices)
        np.save(EXPERIMENTS / f"{name}_seen_project.npy", seen)
        model.booster_.save_model(str(model_dir / "model.txt"), num_iteration=best_iteration)
        json_write(model_dir / "category_maps.json", maps)
        json_write(EXPERIMENTS / f"{name}.json", result)
        print(json.dumps({k:result[k] for k in ("name","mae","seen_project_mae","unseen_project_mae","best_iteration","elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()

