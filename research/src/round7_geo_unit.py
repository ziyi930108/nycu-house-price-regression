"""One fixed R7 unit-price LightGBM with the unchanged R6 geography features.

No data loading, training, file writes, scoring, or query-fitted state on import.
The caller owns stage authorization, data partitions, and artifact publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prepare_data import ROOT, sha256
from round6_geo import (
    FEATURE_RULES as R6_FEATURE_RULES,
    augment_features,
    categories,
    content_sha,
    encode_geo_features,
    frame_sha,
    need,
)
from train_lightgbm import candidate_params, effective_area, encode_features, price_factor

ITERATIONS = 3999
BASE_CONFIGURATION = ROOT / "runs/v1/models/final_config.json"
FEATURE_RULES = {
    **{k: v for k, v in R6_FEATURE_RULES.items()
       if k not in {"fit_target", "sample_weight", "iterations"}},
    "component": "geography_unit_lightgbm",
    "fit_target": "raw TWD / (effective_area * 1e4)",
    "sample_weight": "effective_area / 100",
    "effective_area": "positive finite building_area; else positive finite parking_area; else 1",
    "prediction": "max(unit_prediction * effective_area * 1e4, 0), float64 TWD",
    "iterations": ITERATIONS,
    "num_leaves": 63,
    "threads": 4,
    "category_mapping_fit_scope": "fit X plus fit geography only",
    "unseen_category_code": -1,
    "feature_exclusions": [],
    "trend_applications_in_component": 0,
}


def parameters():
    """Return every original v1 LGB setting, with its selected 3999 rounds."""
    configuration = json.loads(BASE_CONFIGURATION.read_text(encoding="utf-8"))
    matches = [c for c in configuration["components"]
               if c["name"] == "lightgbm_l1_unit63"]
    need(len(matches) == 1, "Expected exactly one saved v1 unit LightGBM")
    component = matches[0]
    need(component["family"] == "lightgbm" and component["target_mode"] == "unit"
         and component["iterations"] == ITERATIONS, "Unexpected v1 component")
    result = {**component["parameters"], "n_estimators": ITERATIONS, "n_jobs": 4}
    need(result == candidate_params(63, ITERATIONS, 4), "Original v1 parameters changed")
    return result


def source_hashes():
    return {name: sha256(ROOT / name) for name in (
        "src/round7_geo_unit.py", "src/round6_geo.py", "src/train_lightgbm.py")}


def _fit_encoding(X, geo, categorical_columns):
    augmented = augment_features(X, geo)
    categorical_names = categories(X, categorical_columns)
    encoded, maps, safe_names, encoded_cats = encode_features(
        {"train": augmented}, categorical_names)
    need(len(augmented.columns) == 95 and len(categorical_names) == 23,
         "Expected 95 features and 23 categorical features")
    state = {
        "format_version": 1,
        "feature_rules": FEATURE_RULES,
        "source_sha256": sha256(Path(__file__)),
        "dependency_source_sha256": source_hashes(),
        "base_configuration_sha256": sha256(BASE_CONFIGURATION),
        "parameters": parameters(),
        "fit_rows": len(X),
        "original_columns": list(X.columns),
        "original_dtypes": {c: str(X[c].dtype) for c in X},
        "all_feature_names": list(augmented.columns),
        "feature_columns": list(augmented.columns),
        "model_feature_columns": list(encoded["train"].columns),
        "original_categorical_columns": list(categorical_columns),
        "categorical_columns": categorical_names,
        "categorical_features": encoded_cats,
        "safe_feature_names": safe_names,
        "category_maps": maps,
        "fit_original_feature_values_sha256": frame_sha(X),
        "fit_geo_feature_values_sha256": frame_sha(geo),
        "fit_augmented_feature_values_sha256": frame_sha(augmented),
        "fit_encoded_feature_values_sha256": frame_sha(encoded["train"]),
        "fit_position_rows": int(geo.geo_has_position.sum()),
        "fit_state_uses_y": False,
        "fit_state_uses_query": False,
    }
    state["state_content_sha256"] = content_sha(state)
    return encoded["train"], state


def fit_feature_state(X, geo, categorical_columns):
    """Reconstruct the complete target-free fit state without fitting a model."""
    return _fit_encoding(X, geo, categorical_columns)[1]


def check_state(state, X, categorical_columns):
    need(state["state_content_sha256"] == content_sha(
        {k: v for k, v in state.items() if k != "state_content_sha256"}),
        "Feature state content changed")
    need(state["feature_rules"] == FEATURE_RULES
         and state["source_sha256"] == sha256(Path(__file__))
         and state["dependency_source_sha256"] == source_hashes(),
         "Feature rules or sources changed")
    need(state["base_configuration_sha256"] == sha256(BASE_CONFIGURATION)
         and state["parameters"] == parameters(), "Base configuration changed")
    need(list(X.columns) == state["original_columns"]
         and {c: str(X[c].dtype) for c in X} == state["original_dtypes"],
         "Original feature schema changed")
    need(list(categorical_columns) == state["original_categorical_columns"]
         and categories(X, categorical_columns) == state["categorical_columns"],
         "Categorical schema changed")
    need(set(state["category_maps"]) == set(state["categorical_columns"]),
         "Category mapping inventory differs")
    need(state["fit_state_uses_y"] is False and state["fit_state_uses_query"] is False,
         "Unexpected target or query feature state")


def encode_query(state, X, geo, categorical_columns):
    """Apply saved maps; never learn a category or change the caller's frames."""
    check_state(state, X, categorical_columns)
    augmented = augment_features(X, geo)
    need(list(augmented.columns) == state["feature_columns"] == state["all_feature_names"],
         "Augmented feature schema differs")
    encoded, maps, safe_names, cats = encode_features(
        {"train": augmented}, state["categorical_columns"],
        category_maps=state["category_maps"])
    need(maps == state["category_maps"] and safe_names == state["safe_feature_names"]
         and cats == state["categorical_features"]
         and list(encoded["train"].columns) == state["model_feature_columns"],
         "Saved category or model schema differs")
    return encoded["train"]


def fit(X, y, geo, categorical_columns):
    """Fit only the supplied rows; return native Booster, JSON state, metadata."""
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_before = source_hashes()
    y = np.asarray(y, dtype="float64")
    need(y.shape == (len(X),) and len(y) > 0 and np.isfinite(y).all() and (y >= 0).all(),
         "Invalid fitting targets")
    encoded, state = _fit_encoding(X, geo, categorical_columns)
    area = effective_area(X)
    target = y / price_factor(X, "unit")
    weights = area / 100.0
    need(np.isfinite(target).all() and np.isfinite(weights).all() and (weights > 0).all(),
         "Invalid unit target or area weight")
    config = parameters()
    estimator = lgb.LGBMRegressor(**config)
    train_started = time.monotonic()
    estimator.fit(encoded, target, sample_weight=weights,
                  categorical_feature=state["categorical_features"])
    training_seconds = time.monotonic() - train_started
    model = estimator.booster_
    need(model.current_iteration() == ITERATIONS and model.num_trees() == ITERATIONS,
         "Unexpected trained iteration count")
    need(model.feature_name() == state["model_feature_columns"], "Model feature schema differs")
    need(source_hashes() == source_before
         and state["base_configuration_sha256"] == sha256(BASE_CONFIGURATION),
         "Source or base configuration changed during fit")
    metadata = {
        "family": "lightgbm", "component": "geography_unit_lightgbm",
        "target_mode": "unit", "parameters": config, "iterations": ITERATIONS,
        "actual_iterations": int(model.current_iteration()), "actual_trees": int(model.num_trees()),
        "features": 95, "feature_count": 95, "categorical_feature_count": 23,
        "feature_columns": state["feature_columns"],
        "model_feature_columns": state["model_feature_columns"],
        "categorical_columns": state["categorical_columns"],
        "categorical_features": state["categorical_features"],
        "sample_weight": FEATURE_RULES["sample_weight"],
        "target_factor": "effective_area * 1e4", "prediction_units": "TWD",
        "geo_feature_rules": FEATURE_RULES,
        "feature_state_content_sha256": state["state_content_sha256"],
        "script_sha256": source_before["src/round7_geo_unit.py"],
        "dependency_source_sha256": source_before,
        "base_configuration_sha256": state["base_configuration_sha256"],
        "fit_rows": len(X),
        "fit_target_values_sha256": hashlib.sha256(y.astype("<f8").tobytes()).hexdigest(),
        "fit_transformed_target_values_sha256": hashlib.sha256(target.astype("<f8").tobytes()).hexdigest(),
        "fit_sample_weight_values_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),
        "fit_original_feature_values_sha256": state["fit_original_feature_values_sha256"],
        "fit_geo_feature_values_sha256": state["fit_geo_feature_values_sha256"],
        "target_used_for_geographic_state": False, "query_used_for_geographic_state": False,
        "reference_features_changed": False, "external_geography_used": True,
        "external_property_answers_used": False, "fit_uses_eval_set": False,
        "early_stopping_used": False, "category_maps_fit_only": True,
        "fit_started_at_utc": started_utc,
        "fit_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": training_seconds, "elapsed_seconds": time.monotonic() - started,
        "lightgbm_version": lgb.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    return model, state, metadata


def predict(model, state, X, geo, categorical_columns):
    """Predict from a fitted estimator or reloaded native Booster, in total TWD."""
    encoded = encode_query(state, X, geo, categorical_columns)
    booster = model if isinstance(model, lgb.Booster) else model.booster_
    need(booster.feature_name() == state["model_feature_columns"]
         and booster.current_iteration() == ITERATIONS and booster.num_trees() == ITERATIONS,
         "Saved model schema or iteration count differs")
    raw = np.asarray(booster.predict(encoded, num_iteration=ITERATIONS, num_threads=4),
                     dtype="float64")
    need(raw.shape == (len(X),) and np.isfinite(raw).all(), "Invalid unit predictions")
    result = np.maximum(raw * price_factor(X, "unit"), 0.0)
    need(result.shape == (len(X),) and np.isfinite(result).all(), "Invalid total-price predictions")
    return result


# Same descriptive aliases as the R6 helper, for callers using that convention.
fit_model = fit
predict_model = predict


def self_check():
    """Bounded synthetic preprocessing/loss checks; no model fitting or file output."""
    config = json.loads(BASE_CONFIGURATION.read_text(encoding="utf-8"))
    cats = config["categorical_columns"]
    X = pd.DataFrame({c: (["A", "B", "A", "B", "A", "B"] if c in cats
                          else np.zeros(6, dtype="float32")) for c in config["feature_columns"]})
    X["building_area"] = np.asarray([100, 0, -2, np.nan, 400, 25], dtype="float32")
    X["parking_area"] = np.asarray([20, 80, 0, np.nan, 30, 0], dtype="float32")
    matched = pd.DataFrame({
        "geo_latitude": [25.0, 25.1, np.nan, np.nan, 24.0, 24.1],
        "geo_longitude": [121.5, 121.6, np.nan, np.nan, 120.5, 120.6],
        "geo_road_length_m": [100, 200, np.nan, np.nan, 500, 600],
        "geo_extent_diagonal_m": [90, 180, np.nan, np.nan, 400, 500],
        "geo_nearby_fragment_grouping": [0, 1, np.nan, np.nan, 0, 0],
        "geo_match_status": ["road_level", "planar_intersection", "ineligible_address",
                             "ineligible_address", "road_level", "road_level"],
    })
    geo = encode_geo_features(matched)
    x_before, geo_before = frame_sha(X), frame_sha(geo)
    encoded, state = _fit_encoding(X, geo, cats)
    saved = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    need(encoded.equals(encode_query(saved, X, geo, cats)), "JSON state replay differs")
    query = X.copy(deep=True); query.loc[0, cats[0]] = "UNSEEN_QUERY_CATEGORY"
    query_geo = geo.copy(deep=True); query_geo.loc[0, "geo_grid_002"] = "UNSEEN_GRID"
    state_before = content_sha(saved)
    query_encoded = encode_query(saved, query, query_geo, cats)
    for name in (cats[0], "geo_grid_002"):
        need(query_encoded.loc[0, saved["safe_feature_names"][name]] == -1,
             "Unseen query category was learned")
    need(content_sha(saved) == state_before, "Query changed saved fit state")
    need(frame_sha(X) == x_before and frame_sha(geo) == geo_before, "Input frames were mutated")
    area = effective_area(X)
    need(np.array_equal(area, [100, 80, 1, 1, 400, 25]), "Area fallback differs")
    y = np.asarray([1e7, 8e6, 2e5, 3e5, 3e7, 2e6], dtype="float64")
    prediction_unit = np.asarray([9, 11, 19, 32, 8, 7], dtype="float64")
    need(np.allclose((area / 100) * np.abs(y / price_factor(X, "unit") - prediction_unit),
                     np.abs(y - prediction_unit * price_factor(X, "unit")) / 1e6,
                     rtol=1e-13, atol=1e-13), "Unit MAE no longer equals total-price MAE")
    rejected = False
    try:
        encode_query(saved, query_geo.reindex(columns=query_geo.columns[::-1]), X, cats)
    except ValueError:
        rejected = True
    need(rejected, "Wrong original/geography schema was accepted")
    need(len(saved["model_feature_columns"]) == 95 and len(saved["categorical_features"]) == 23,
         "Feature counts differ")
    return {"status": "passed", "synthetic_rows": len(X), "model_training_performed": False,
            "file_outputs_written": False, "real_data_read": False,
            "checks": ["v1_parameter_identity", "95_features_23_categories", "JSON_state_replay",
                       "unseen_categories_minus_one", "query_does_not_mutate_state",
                       "original_frames_unchanged", "area_fallback", "exact_MAE_scaling",
                       "schema_rejection"], "parameters": parameters(),
            "script_sha256": sha256(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False), flush=True)
