"""Fit reproducible LightGBM MAE models using only supplied local records.

Training-only category maps encode unseen values as -1. The validation split
selects the stopping iteration; all saved predictions are in original dollars.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from model_common import load_data, save_experiment

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def encode_features(frames, categorical_columns, category_maps=None):
    """Learn categorical codes from train only; preserve row and column order."""
    feature_names = list(frames["train"].columns)
    categorical_columns = [c for c in categorical_columns if c in feature_names]
    if category_maps is None:
        category_maps = {}
        for column in categorical_columns:
            values = frames["train"][column].fillna("__MISSING__").astype(str)
            category_maps[column] = {value: i for i, value in enumerate(pd.unique(values))}
    safe_names = {name: f"f{i:03d}" for i, name in enumerate(feature_names)}
    encoded = {}
    for split, frame in frames.items():
        if list(frame.columns) != feature_names:
            raise ValueError(f"{split} feature schema differs from training")
        columns = {}
        for column in feature_names:
            if column in category_maps:
                values = frame[column].fillna("__MISSING__").astype(str)
                columns[safe_names[column]] = values.map(category_maps[column]).fillna(-1).astype("int32")
            else:
                columns[safe_names[column]] = pd.to_numeric(frame[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                ).astype("float32")
        encoded[split] = pd.DataFrame(columns, index=frame.index)
    return encoded, category_maps, safe_names, [safe_names[c] for c in categorical_columns]


def effective_area(X):
    building = X["building_area"].to_numpy(dtype="float64")
    parking = X["parking_area"].to_numpy(dtype="float64")
    return np.where(np.isfinite(building) & (building > 0), building,
                    np.where(np.isfinite(parking) & (parking > 0), parking, 1.0))


def price_factor(X, mode):
    return effective_area(X) * 1e4 if mode == "unit" else np.full(len(X), 1e6)


def candidate_params(leaves, iterations, threads):
    return {
        "objective": "regression_l1", "metric": "None",
        "n_estimators": iterations, "learning_rate": 0.04,
        "num_leaves": leaves, "max_depth": -1, "min_child_samples": 40,
        "colsample_bytree": 0.9, "subsample": 0.9, "subsample_freq": 1,
        "reg_lambda": 2.0, "cat_smooth": 20.0, "cat_l2": 10.0,
        "max_cat_to_onehot": 4, "max_cat_threshold": 64,
        "random_state": SEED, "n_jobs": min(4, max(1, threads)),
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
    }


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train_candidate(data, encoded, maps, safe_names, categorical_features, mode, args):
    name = f"lightgbm_l1_{mode}{args.leaves}"
    output_dir = ROOT / "outputs" / "experiments" / name
    model_dir = ROOT / "models" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    params = candidate_params(args.leaves, args.iterations, args.threads)
    factors = {s: price_factor(data[f"X_{s}"], mode) for s in ("train", "valid", "test")}
    y_train = np.asarray(data["y_train"], dtype="float64")
    y_valid = np.asarray(data["y_valid"], dtype="float64")
    weights = effective_area(data["X_train"]) / 100 if mode == "unit" else None
    def price_mae(y_true, prediction):
        return "price_mae", float(np.mean(np.abs((y_true - prediction) * factors["valid"]))), False
    model = lgb.LGBMRegressor(**params)
    started = time.perf_counter()
    print(f"Starting {name}: {len(y_train):,} train / {len(y_valid):,} valid", flush=True)
    model.fit(
        encoded["train"], y_train / factors["train"], sample_weight=weights,
        eval_set=[(encoded["valid"], y_valid / factors["valid"])],
        eval_metric=price_mae, categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(args.patience, first_metric_only=True, verbose=True),
                   lgb.log_evaluation(100)],
    )
    best_iteration = int(model.best_iteration_ or args.iterations)
    valid_raw = model.predict(encoded["valid"], num_iteration=best_iteration) * factors["valid"]
    test_raw = model.predict(encoded["test"], num_iteration=best_iteration) * factors["test"]
    valid_prediction = np.maximum(valid_raw, 0.0)
    test_prediction = np.maximum(test_raw, 0.0)
    if not np.isfinite(valid_prediction).all() or not np.isfinite(test_prediction).all():
        raise ValueError("Predictions contain non-finite values")
    np.save(output_dir / "valid_predictions.npy", valid_prediction)
    np.save(output_dir / "test_predictions.npy", test_prediction)
    model_path = model_dir / "model.txt"
    model.booster_.save_model(str(model_path), num_iteration=best_iteration)
    category_path = model_dir / "category_maps.json"
    json_write(category_path, maps)
    metadata = {
        "name": name, "family": "lightgbm", "target_mode": mode,
        "objective": "absolute_error_original_price",
        "validation_mae": float(mean_absolute_error(y_valid, valid_prediction)),
        "validation_mae_before_nonnegative_clip": float(mean_absolute_error(y_valid, valid_raw)),
        "best_iteration": best_iteration, "elapsed_seconds": time.perf_counter() - started,
        "params": params, "seed": SEED, "train_rows": len(y_train),
        "valid_rows": len(y_valid), "test_rows": len(test_prediction),
        "features": list(safe_names), "safe_feature_names": safe_names,
        "categorical_features": categorical_features,
        "category_maps_path": str(category_path.relative_to(ROOT)),
        "category_mapping_fit_split": "train_only", "unseen_category_code": -1,
        "model_path": str(model_path.relative_to(ROOT)),
        "target_factor": "effective_area * 1e4" if mode == "unit" else "1e6",
        "sample_weight": "effective_area / 100" if mode == "unit" else None,
        "effective_area": "positive building_area, else positive parking_area, else 1",
        "lightgbm_version": lgb.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "validation_evaluation": model.evals_result_,
        "external_data": False,
    }
    json_write(output_dir / "metrics.json", metadata)
    json_write(model_dir / "metadata.json", {k:v for k,v in metadata.items() if k != "validation_evaluation"})
    save_experiment(name, data, valid_prediction, test_prediction,
                    {k:v for k,v in metadata.items() if k != "validation_evaluation"})
    pd.DataFrame({
        "feature": list(safe_names),
        "gain": model.booster_.feature_importance("gain", iteration=best_iteration),
        "split": model.booster_.feature_importance("split", iteration=best_iteration),
    }).sort_values("gain", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k:metadata[k] for k in ("name","validation_mae","best_iteration","elapsed_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["raw","unit"], default=["raw","unit"])
    parser.add_argument("--leaves", type=int, default=63, choices=[63,127])
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    data = load_data()
    frames = {s:data[f"X_{s}"] for s in ("train","valid","test")}
    categorical_columns = data.get("categorical_columns")
    if categorical_columns is None:
        categorical_columns = list(frames["train"].select_dtypes(include=["object","string","category"]).columns)
    encoded, maps, safe_names, cats = encode_features(frames, categorical_columns)
    for mode in args.modes:
        train_candidate(data, encoded, maps, safe_names, cats, mode, args)


if __name__ == "__main__":
    main()

