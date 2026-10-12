"""Shared, leakage-aware local data loading and MAE evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_data import ROOT, CACHE_DIR, REPORT_DIR, TARGET, ID, sha256

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
MODEL_DIR = ROOT / "models"

def _canonical(value):
    if pd.isna(value):
        return "m:"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "n:" + format(float(value), ".17g")
    return "s:" + str(value)

def feature_hashes(frame):
    normalized = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
    for name in normalized:
        normalized[name] = normalized[name].map(_canonical)
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy()

def load_data():
    schema = json.loads((REPORT_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["feature_source_sha256"] != sha256(ROOT / "src" / "house_features.py"):
        raise RuntimeError("Feature code changed. Run src/prepare_features.py again.")
    data = {}
    for split in ("train", "valid", "test"):
        raw = pd.read_pickle(CACHE_DIR / f"{split}.pkl")
        features = pd.read_pickle(CACHE_DIR / f"{split}_features.pkl")
        metadata = json.loads((CACHE_DIR / f"{split}.json").read_text(encoding="utf-8"))
        if schema["splits"][split]["input_source_sha256"] != metadata["source_sha256"]:
            raise RuntimeError(f"Stale feature cache: {split}")
        if len(raw) != len(features) or list(features.columns) != schema["columns"]:
            raise ValueError(f"Feature alignment/schema problem: {split}")
        data[f"raw_{split}"] = raw
        data[f"X_{split}"] = features
        data[f"ids_{split}"] = raw[ID].to_numpy()
        if TARGET in raw:
            data[f"y_{split}"] = raw[TARGET].to_numpy(dtype="float64")
    raw = data["raw_train"]
    keep = ~raw.duplicated(subset=[c for c in raw if c != ID]).to_numpy()
    for key in ("raw_train", "X_train"):
        data[key] = data[key].loc[keep].reset_index(drop=True)
    data["y_train"] = data["y_train"][keep]
    data["ids_train"] = data["ids_train"][keep]
    mask_path = CACHE_DIR / "valid_clean_mask.npy"
    expected_signature = {
        split: schema["splits"][split]["input_source_sha256"] for split in ("train", "valid")}
    signature_path = CACHE_DIR / "valid_clean_mask.json"
    reuse = mask_path.exists() and signature_path.exists()
    if reuse:
        reuse = json.loads(signature_path.read_text(encoding="utf-8")) == expected_signature
    if reuse:
        mask = np.load(mask_path)
    else:
        hashes = feature_hashes(data["raw_train"])
        valid_hashes = feature_hashes(data["raw_valid"])
        mask = ~pd.Series(valid_hashes).isin(hashes).to_numpy()
        np.save(mask_path, mask)
        signature_path.write_text(json.dumps(expected_signature, indent=2), encoding="utf-8")
    if len(mask) != len(data["y_valid"]) or not mask.any():
        raise ValueError("Invalid clean-validation mask")
    data["valid_clean_mask"] = mask.astype(bool)
    data["categorical_columns"] = schema["categorical_columns"]
    data["metadata"] = {
        "train_rows_before_dedup": int(len(keep)), "train_rows": int(keep.sum()),
        "removed_exact_train_duplicates_without_id": int((~keep).sum()),
        "valid_rows": len(mask), "clean_valid_rows": int(mask.sum()),
        "overlapping_valid_rows": int((~mask).sum()),
        "test_rows": len(data["X_test"]), "features": len(schema["columns"]),
        "source_sha256": {k: schema["splits"][k]["input_source_sha256"] for k in schema["splits"]},
        "external_property_data_used": False}
    MODEL_DIR.mkdir(exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    return data

def mae(y, prediction):
    y, prediction = np.asarray(y, dtype="float64"), np.asarray(prediction, dtype="float64")
    if y.shape != prediction.shape or y.ndim != 1 or not len(y):
        raise ValueError("MAE requires aligned, nonempty one-dimensional arrays")
    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
        raise ValueError("Nonfinite target or prediction")
    return float(np.abs(y - prediction).mean())

def evaluate(data, prediction):
    prediction = np.asarray(prediction, dtype="float64")
    mask = data["valid_clean_mask"]
    error = np.abs(data["y_valid"] - prediction)
    return {"mae": mae(data["y_valid"], prediction),
            "clean_mae": mae(data["y_valid"][mask], prediction[mask]),
            "median_absolute_error": float(np.median(error)),
            "p90_absolute_error": float(np.quantile(error, .9))}

def save_experiment(name, data, valid_prediction, test_prediction, metadata):
    valid_prediction = np.maximum(np.asarray(valid_prediction, dtype="float64"), 0.0)
    test_prediction = np.maximum(np.asarray(test_prediction, dtype="float64"), 0.0)
    if test_prediction.shape != (len(data["X_test"]),) or not np.isfinite(test_prediction).all():
        raise ValueError("Invalid test predictions")
    result = {"name": name, **metadata, **evaluate(data, valid_prediction), "data": data["metadata"]}
    np.save(EXPERIMENT_DIR / f"{name}_valid.npy", valid_prediction)
    np.save(EXPERIMENT_DIR / f"{name}_test.npy", test_prediction)
    (EXPERIMENT_DIR / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result

def full_training_data(data):
    raw = pd.concat([data["raw_train"], data["raw_valid"]], ignore_index=True)
    features = pd.concat([data["X_train"], data["X_valid"]], ignore_index=True)
    target = np.concatenate([data["y_train"], data["y_valid"]])
    keep = ~raw.duplicated(subset=[c for c in raw if c != ID]).to_numpy()
    return features.loc[keep].reset_index(drop=True), target[keep]

def temporal_indices(data, cutoff="2024-05-01", end="2024-08-10"):
    days = data["X_train"]["trade_day_index"].to_numpy()
    cutoff_days = (pd.Timestamp(cutoff) - pd.Timestamp("2000-01-01")).days
    end_days = (pd.Timestamp(end) - pd.Timestamp("2000-01-01")).days
    fit = np.flatnonzero(np.isfinite(days) & (days < cutoff_days))
    valid = np.flatnonzero(np.isfinite(days) & (days >= cutoff_days) & (days <= end_days))
    return fit, valid
