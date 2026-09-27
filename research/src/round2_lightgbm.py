"""Bounded round-two LightGBM generalization experiments.

Only local supplied records are used. Each categorical vocabulary is learned
from the current fit partition, never from holdout targets or vocabulary.
Run temporal candidates first; only promising candidates proceed to the
original validation split. Only the explicitly selected spatial63 candidate has an optional final refit; no test prediction is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from model_common import load_data, temporal_indices, full_training_data
from train_lightgbm import encode_features, effective_area, price_factor, candidate_params

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "round2" / "lightgbm"
EXPERIMENTS = ROOT / "outputs" / "experiments"
CONFIGS = {
    "lightgbm_unit_smooth31": {
        "drop_features": [],
        "overrides": {"num_leaves": 31, "min_child_samples": 100,
                      "cat_smooth": 100.0, "cat_l2": 30.0, "reg_lambda": 5.0},
        "hypothesis": "Smaller trees and stronger category regularization may generalize better to new projects.",
    },
    "lightgbm_unit_spatial63": {
        "drop_features": ["project_name", "address_raw", "building_and_unit", "building_block"],
        "overrides": {"num_leaves": 63, "min_child_samples": 80,
                      "cat_smooth": 100.0, "cat_l2": 30.0, "reg_lambda": 5.0},
        "hypothesis": "Removing project, full-address, and unit identifiers forces transferable geographic and structural patterns.",
    },
}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest():
    records = []
    for name in CONFIGS:
        summary_path = OUT / f"{name}.json"
        temporal_path = OUT / f"{name}_temporal_valid.npy"
        valid_path = OUT / f"{name}_valid.npy"
        if not summary_path.exists() or not temporal_path.exists():
            continue
        records.append({
            "name": name,
            "valid_predictions": relative(valid_path) if valid_path.exists() else None,
            "temporal_predictions": relative(temporal_path),
            "temporal_indices": relative(OUT / f"{name}_temporal_indices.npy"),
            "metadata_path": relative(summary_path),
        })
    atomic_json(OUT / "candidates.json", records)


def grouped_metrics(y, prediction, seen):
    error = np.abs(np.asarray(y, dtype="float64") - prediction)
    return {
        "mae": float(error.mean()),
        "seen_project_mae": float(error[seen].mean()) if seen.any() else None,
        "unseen_project_mae": float(error[~seen].mean()) if (~seen).any() else None,
        "rows": len(y), "seen_project_rows": int(seen.sum()),
        "unseen_project_rows": int((~seen).sum()),
    }


def temporal_screen(y, prediction, indices, metrics):
    base = load_json(EXPERIMENTS / "lightgbm_l1_unit63_temporal.json")
    reference_indices = np.load(EXPERIMENTS / "selected_temporal_indices.npy")
    if not np.array_equal(indices, reference_indices):
        raise ValueError("Temporal indices differ from the established reference ensemble")
    reference = np.load(EXPERIMENTS / "selected_temporal_valid.npy")
    base_mae = float(np.abs(y - reference).mean())
    blends = []
    for weight in (0.0, 0.1, 0.2, 0.3, 0.5):
        mixed = (1 - weight) * reference + weight * prediction
        blends.append({"candidate_weight": weight, "mae": float(np.abs(y - mixed).mean())})
    best_blend = min(blends, key=lambda row: row["mae"])
    reasons = []
    if metrics["mae"] < base["mae"]:
        reasons.append("temporal_mae_better_than_original_lightgbm_unit63")
    if metrics["unseen_project_mae"] < base["unseen_project_mae"]:
        reasons.append("unseen_project_mae_better_than_original_lightgbm_unit63")
    if best_blend["mae"] < 0.998 * base_mae:
        reasons.append("coarse_blend_improves_reference_ensemble_by_at_least_0.2_percent")
    return {
        "passed": bool(reasons), "reasons": reasons,
        "original_lightgbm_temporal_mae": base["mae"],
        "original_lightgbm_unseen_mae": base["unseen_project_mae"],
        "reference_ensemble_mae": base_mae,
        "coarse_blends": blends, "best_coarse_blend": best_blend,
    }


def train_stage(data, name, stage, fit_indices, valid_indices, args):
    config = CONFIGS[name]
    prefix = name + ("_temporal" if stage == "temporal" else "_original")
    metrics_path = OUT / f"{prefix}.json"
    prediction_path = OUT / (f"{name}_temporal_valid.npy" if stage == "temporal" else f"{name}_valid.npy")
    if metrics_path.exists() and prediction_path.exists() and not args.overwrite:
        print(f"Reusing completed {prefix}", flush=True)
        return load_json(metrics_path), np.load(prediction_path)

    if stage == "temporal":
        original_fit = data["X_train"].iloc[fit_indices]
        original_valid = data["X_train"].iloc[valid_indices]
        y_fit, y_valid = data["y_train"][fit_indices], data["y_train"][valid_indices]
    else:
        original_fit, original_valid = data["X_train"], data["X_valid"]
        y_fit, y_valid = data["y_train"], data["y_valid"]
    frames = {
        "train": original_fit.drop(columns=config["drop_features"]),
        "valid": original_valid.drop(columns=config["drop_features"]),
    }
    encoded, maps, safe_names, cats = encode_features(frames, data["categorical_columns"])
    params = candidate_params(63, args.iterations, 4)
    params.update(config["overrides"])
    factors = {key: price_factor(frame, "unit") for key, frame in frames.items()}
    weights = effective_area(frames["train"]) / 100
    seen_projects = set(original_fit["project_name"]) - {"__MISSING__"}
    seen = original_valid["project_name"].isin(seen_projects).to_numpy()
    def price_mae(y_true, prediction):
        return "price_mae", float(np.abs((y_true - prediction) * factors["valid"]).mean()), False

    print(f"Starting {prefix}: {len(y_fit):,} fit / {len(y_valid):,} holdout; "
          f"{len(frames['train'].columns)} features", flush=True)
    started = time.perf_counter()
    model = lgb.LGBMRegressor(**params)
    model.fit(
        encoded["train"], y_fit / factors["train"], sample_weight=weights,
        eval_set=[(encoded["valid"], y_valid / factors["valid"])],
        eval_metric=price_mae, categorical_feature=cats,
        callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=True),
                   lgb.log_evaluation(250)],
    )
    best_iteration = int(model.best_iteration_ or args.iterations)
    prediction = np.maximum(model.predict(encoded["valid"], num_iteration=best_iteration) * factors["valid"], 0)
    if prediction.shape != y_valid.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction alignment or finite-value check failed")
    metrics = grouped_metrics(y_valid, prediction, seen)
    model_dir = OUT / "models" / prefix
    model_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_dir / "model.txt"), num_iteration=best_iteration)
    atomic_json(model_dir / "category_maps.json", maps)
    np.save(prediction_path, prediction)
    if stage == "temporal":
        np.save(OUT / f"{name}_temporal_indices.npy", valid_indices)
        np.save(OUT / f"{name}_temporal_seen_project.npy", seen)
        metrics["screening"] = temporal_screen(y_valid, prediction, valid_indices, metrics)
    else:
        clean = data["valid_clean_mask"]
        metrics["clean_mae"] = float(np.abs(y_valid[clean] - prediction[clean]).mean())
    metrics.update({
        "name": prefix, "base_name": name, "family": "lightgbm", "stage": stage,
        "target_mode": "unit", "hypothesis": config["hypothesis"],
        "drop_features": config["drop_features"], "feature_count": len(safe_names),
        "features": list(safe_names), "safe_feature_names": safe_names,
        "categorical_features": cats, "params": params, "best_iteration": best_iteration,
        "fit_rows": len(y_fit), "elapsed_seconds": time.perf_counter() - started,
        "category_mapping_fit_split": "current_fit_only", "unseen_category_code": -1,
        "target_factor": "effective_area * 1e4", "sample_weight": "effective_area / 100",
        "effective_area": "positive building_area, else positive parking_area, else 1",
        "cutoff": "2024-04-01" if stage == "temporal" else None,
        "end": "2024-08-10" if stage == "temporal" else None,
        "model_path": relative(model_dir / "model.txt"),
        "category_maps_path": relative(model_dir / "category_maps.json"),
        "predictions_path": relative(prediction_path),
        "source_sha256": data["metadata"]["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_source_sha256": hashlib.sha256((ROOT / "src" / "house_features.py").read_bytes()).hexdigest(),
        "lightgbm_version": lgb.__version__, "external_data": False,
    })
    atomic_json(metrics_path, metrics)
    pd.DataFrame({
        "feature": list(safe_names),
        "gain": model.booster_.feature_importance("gain", iteration=best_iteration),
    }).sort_values("gain", ascending=False).to_csv(OUT / f"{prefix}_importance.csv", index=False)
    print(json.dumps({key:metrics[key] for key in (
        "name", "mae", "seen_project_mae", "unseen_project_mae",
        "best_iteration", "elapsed_seconds")}), flush=True)
    return metrics, prediction


def update_summary(name, temporal=None, valid=None):
    path = OUT / f"{name}.json"
    summary = load_json(path) if path.exists() else {
        "name": name, "family": "lightgbm", "target_mode": "unit",
        **CONFIGS[name], "temporal": None, "valid": None, "external_data": False,
    }
    if temporal is not None:
        summary["temporal"] = temporal
        summary["screening"] = temporal["screening"]
    if valid is not None:
        summary["valid"] = valid
    atomic_json(path, summary)
    update_manifest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_spatial_fit(data, args):
    """Refit the selected spatial63 candidate only; never predict test prices."""
    name = "lightgbm_unit_spatial63"
    if args.candidates != [name]:
        raise ValueError("Final refit is authorized only for --candidates lightgbm_unit_spatial63")
    validation_path = OUT / f"{name}_original.json"
    if not validation_path.exists():
        raise ValueError("Complete this candidate's original validation before final refit")
    validation = load_json(validation_path)
    selected_iterations = int(validation["best_iteration"])
    if selected_iterations < 1:
        raise ValueError("Invalid validation-selected iteration count")
    destination = OUT / "final_spatial63"
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        existing = load_json(metadata_path)
        if (existing["validation_best_iteration"] != selected_iterations
                or file_sha256(destination / "model.txt") != existing["model_sha256"]
                or file_sha256(destination / "category_maps.json") != existing["category_maps_sha256"]):
            raise ValueError("Existing final artifact differs; review before explicitly using --overwrite")
        print(f"Reusing verified completed final model: {metadata_path}", flush=True)
        return existing

    full_X, full_y = full_training_data(data)
    if len(full_X) != 467221 or len(full_y) != 467221:
        raise ValueError("Expected 467,221 deduplicated train+valid rows")
    full_X = full_X.drop(columns=CONFIGS[name]["drop_features"])
    categorical_columns = [column for column in data["categorical_columns"] if column in full_X]
    encoded, maps, safe_names, cats = encode_features({"train": full_X}, categorical_columns)
    factor = price_factor(full_X, "unit")
    weights = effective_area(full_X) / 100
    params = dict(validation["params"])
    params["n_estimators"] = selected_iterations
    params["n_jobs"] = 4
    model = lgb.LGBMRegressor(**params)

    def training_progress(environment):
        iteration = environment.iteration + 1
        if iteration % 250 == 0 or iteration == selected_iterations:
            print(f"Final spatial63 iteration {iteration}/{selected_iterations}", flush=True)

    print(f"Starting final spatial63: {len(full_y):,} rows / "
          f"{selected_iterations} validation-selected iterations / 4 threads", flush=True)
    started = time.perf_counter()
    model.fit(
        encoded["train"], full_y / factor, sample_weight=weights,
        categorical_feature=cats, callbacks=[training_progress],
    )
    actual_iterations = int(model.booster_.current_iteration())
    model_path = destination / "model.txt"
    maps_path = destination / "category_maps.json"
    model.booster_.save_model(str(model_path), num_iteration=actual_iterations)
    atomic_json(maps_path, maps)
    metadata = {
        "name": f"{name}_final", "family": "lightgbm", "target_mode": "unit",
        "iterations": actual_iterations, "actual_iterations": actual_iterations,
        "requested_iterations": selected_iterations,
        "validation_best_iteration": selected_iterations,
        "validation_mae": validation["mae"], "validation_clean_mae": validation["clean_mae"],
        "validation_metadata_path": relative(validation_path),
        "params": params, "training_rows": len(full_y),
        "training_definition": "train+valid with exact duplicate rows excluding ID removed by model_common.full_training_data",
        "drop_features": CONFIGS[name]["drop_features"],
        "feature_columns": list(full_X.columns),
        "categorical_columns": categorical_columns,
        "safe_feature_names": safe_names, "categorical_features": cats,
        "category_mapping_fit_split": "deduplicated_train_plus_valid_only",
        "unseen_category_code": -1,
        "target_factor": "effective_area * 1e4", "sample_weight": "effective_area / 100",
        "effective_area": "positive building_area, else positive parking_area, else 1",
        "model_path": relative(model_path), "category_maps_path": relative(maps_path),
        "model_sha256": file_sha256(model_path), "category_maps_sha256": file_sha256(maps_path),
        "source_sha256": data["metadata"]["source_sha256"],
        "training_source_sha256": {split: data["metadata"]["source_sha256"][split] for split in ("train", "valid")},
        "script_sha256": file_sha256(Path(__file__)),
        "feature_source_sha256": file_sha256(ROOT / "src" / "house_features.py"),
        "common_source_sha256": file_sha256(ROOT / "src" / "model_common.py"),
        "encoding_source_sha256": file_sha256(ROOT / "src" / "train_lightgbm.py"),
        "lightgbm_version": lgb.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "eval_set_used": False, "test_predictions_produced": False, "external_data": False,
        "elapsed_seconds": time.perf_counter() - started,
        "reproduce": ".venv/Scripts/python.exe src/round2_lightgbm.py --stage final --candidates lightgbm_unit_spatial63",
    }
    atomic_json(metadata_path, metadata)
    summary_path = OUT / f"{name}.json"
    summary = load_json(summary_path)
    summary["final_model_metadata"] = relative(metadata_path)
    atomic_json(summary_path, summary)
    update_manifest()
    print(json.dumps({key: metadata[key] for key in (
        "name", "training_rows", "actual_iterations", "elapsed_seconds",
        "model_sha256", "category_maps_sha256")}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "temporal", "valid", "final"], default="all")
    parser.add_argument("--candidates", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--force-valid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.iterations > 4000 or args.iterations < 1:
        raise ValueError("This bounded experiment permits 1 to 4000 iterations")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "round2_lightgbm.py").write_bytes(Path(__file__).read_bytes())
    data = load_data()
    if args.stage == "final":
        final_spatial_fit(data, args)
        return
    fit_indices, valid_indices = temporal_indices(data, cutoff="2024-04-01", end="2024-08-10")
    if len(valid_indices) != 58206 or len(data["y_valid"]) != 11994:
        raise ValueError("Unexpected validation split size")
    if args.stage in ("all", "temporal"):
        for name in args.candidates:
            temporal, _ = train_stage(data, name, "temporal", fit_indices, valid_indices, args)
            update_summary(name, temporal=temporal)
            print(f"Manifest updated for {name}; screening_passed={temporal['screening']['passed']}", flush=True)
    if args.stage in ("all", "valid"):
        for name in args.candidates:
            summary_path = OUT / f"{name}.json"
            if not summary_path.exists():
                raise ValueError(f"Temporal screening must run first for {name}")
            summary = load_json(summary_path)
            if not args.force_valid and not summary["screening"]["passed"]:
                print(f"Skipping original validation for rejected temporal candidate {name}", flush=True)
                continue
            valid, _ = train_stage(data, name, "valid", fit_indices, valid_indices, args)
            update_summary(name, valid=valid)
    print(f"Completed requested stage. Manifest: {OUT / 'candidates.json'}", flush=True)


if __name__ == "__main__":
    main()

