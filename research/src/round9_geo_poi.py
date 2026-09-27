"""One fixed R9 historical-POI enhancement of the R6 raw CatBoost component.

Pure feature functions never accept targets. Only fit_model receives fit y.
The caller owns annual-map provenance, partitions, authorization and files.
No real data, training, predictions or artifact writes occur on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from prepare_data import ROOT, sha256
from round6_geo import (
    FEATURES as GEO_FEATURES,
    FEATURE_RULES as R6_FEATURE_RULES,
    augment_features as augment_r6_features,
    categories as r6_categories,
    content_sha,
    frame_sha,
    need,
    parameters as r6_parameters,
)

POI_KEYS = (
    "railway_station", "subway_entrance", "bus_station", "school",
    "kindergarten", "higher_education", "hospital", "clinic", "pharmacy",
    "supermarket", "convenience", "park", "motorway_junction",
)
POI_DISTANCE_COLUMNS = ["poi_log_distance_" + name for name in POI_KEYS]
POI_POSITION_COLUMN = "poi_position_usable"
POI_FEATURES = POI_DISTANCE_COLUMNS + [POI_POSITION_COLUMN]
ITERATIONS = 2999
FEATURE_COUNT = 109
BASE_CONFIGURATION = ROOT / "runs/v1/models/final_config.json"
FEATURE_RULES = {
    "format_version": 1,
    "component": "geography_poi_raw_catboost",
    "original_feature_count": 87,
    "existing_geo_feature_count": 8,
    "existing_95_values_dtypes_order_and_index_unchanged": True,
    "poi_keys": list(POI_KEYS),
    "poi_distance_columns": POI_DISTANCE_COLUMNS,
    "poi_position_column": POI_POSITION_COLUMN,
    "poi_columns": POI_FEATURES,
    "poi_dtype": "float32",
    "distance_input": "already log1p(nonnegative distance in metres); do not transform again",
    "missing_distance": "NaN is preserved; never impute zero or city coordinates",
    "position_flag": "exactly 0 or 1; 1 requires geo_has_position=1",
    "flag_zero_distances": "all 13 distances must be NaN",
    "flag_one_distances": "each distance is finite nonnegative or NaN for unavailable source category",
    "annual_map_provenance": "caller must bind the same as-of annual snapshot as the R6 geography",
    "original_geo_rules": R6_FEATURE_RULES,
    "feature_count": FEATURE_COUNT,
    "categorical_feature_count": 23,
    "target_formula": "y_fit / 1e6",
    "prediction_formula": "max(model_output * 1e6, 0), unrounded float64 TWD",
    "sample_weight": "uniform",
    "iterations": ITERATIONS,
    "threads": 4,
    "seed": 42,
    "no_eval_set": True,
    "no_early_stopping": True,
    "trend_applications_in_component": 0,
    "feature_state_uses_y": False,
    "feature_state_uses_query": False,
    "missing_position_predictions_equal_old_model_guaranteed": False,
}


def source_hashes():
    return {rel: sha256(ROOT / rel) for rel in (
        "src/round9_geo_poi.py", "src/round6_geo.py", "src/prepare_data.py")}


def parameters():
    """Use the complete frozen R6 raw configuration without new tuning."""
    config = r6_parameters()
    need(config.get("iterations") == ITERATIONS and config.get("thread_count") == 4
         and config.get("random_seed") == 42 and config.get("loss_function") == "MAE"
         and config.get("task_type") == "GPU", "Unexpected original CatBoost parameters")
    return config


def check_poi(poi, geo):
    """Validate prepared pure-X POI values; never infer a missing distance."""
    need(isinstance(poi, pd.DataFrame) and isinstance(geo, pd.DataFrame),
         "Expected POI and geography DataFrames")
    need(poi.columns.is_unique and list(poi.columns) == POI_FEATURES,
         "POI columns or their order differ from the fixed 14-column schema")
    need(len(poi) == len(geo) and poi.index.equals(geo.index),
         "POI and geography row indices differ")
    need(all(poi[c].dtype == np.dtype("float32") for c in POI_FEATURES),
         "Every POI column must be float32")
    flag = poi[POI_POSITION_COLUMN]
    need(flag.isin([0.0, 1.0]).all(), "POI position flag must be 0 or 1 without missing values")
    need("geo_has_position" in geo, "Missing original geography position flag")
    usable = flag.eq(1.0)
    need(geo.loc[usable, "geo_has_position"].eq(1.0).all(),
         "POI position cannot be usable when original geography has no position")
    distances = poi[POI_DISTANCE_COLUMNS].to_numpy(dtype="float32", copy=False)
    allowed = np.isnan(distances) | (np.isfinite(distances) & (distances >= 0.0))
    need(allowed.all(), "POI log distances must be nonnegative finite numbers or NaN")
    need(poi.loc[~usable, POI_DISTANCE_COLUMNS].isna().all().all(),
         "Unusable POI positions must have all distances missing")


def augment_features(X, geo, poi):
    """Return independent 109-column data, preserving all original 95 values."""
    need(isinstance(X, pd.DataFrame) and X.columns.is_unique, "Expected unique original feature columns")
    need(not {"編號", "總價元", "y"}.intersection(X.columns), "IDs and targets cannot be features")
    need(isinstance(geo, pd.DataFrame), "Expected a geography DataFrame")
    need(not set(POI_FEATURES).intersection(X.columns), "Features already augmented")
    base = augment_r6_features(X, geo)
    check_poi(poi, geo)
    result = pd.concat([base, poi.copy(deep=True)], axis=1)
    need(len(result.columns) == FEATURE_COUNT and result.columns.is_unique,
         "Expected 109 unique features")
    need(result[base.columns].equals(base) and result[base.columns].dtypes.equals(base.dtypes)
         and result.index.equals(X.index), "Original 95 feature values, types or indices changed")
    need(result[POI_FEATURES].equals(poi) and result[POI_FEATURES].dtypes.equals(poi.dtypes),
         "Prepared POI values or missing values changed")
    return result


def _categories(X, supplied):
    actual = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    need(list(supplied) == actual and len(actual) == 21, "Original categorical order or count differs")
    combined = r6_categories(X, supplied)
    need(len(combined) == 23 and not set(combined).intersection(POI_FEATURES),
         "Expected unchanged 23 categorical features")
    return combined


def fit_feature_state(X, geo, poi, categorical_columns):
    """Save fit-X schemas and hashes only; no price/query-derived parameters."""
    augmented = augment_features(X, geo, poi)
    categorical = _categories(X, categorical_columns)
    state = {
        "format_version": 1,
        "feature_rules": json.loads(json.dumps(FEATURE_RULES)),
        "source_sha256": sha256(Path(__file__)),
        "dependency_source_sha256": source_hashes(),
        "base_configuration_sha256": sha256(BASE_CONFIGURATION),
        "parameters": parameters(),
        "fit_rows": len(X),
        "original_columns": list(X.columns),
        "original_dtypes": {c: str(X[c].dtype) for c in X},
        "geo_columns": list(geo.columns),
        "geo_dtypes": {c: str(geo[c].dtype) for c in geo},
        "poi_columns": list(poi.columns),
        "poi_dtypes": {c: str(poi[c].dtype) for c in poi},
        "all_feature_names": list(augmented.columns),
        "feature_columns": list(augmented.columns),
        "original_categorical_columns": list(categorical_columns),
        "categorical_columns": categorical,
        "fit_original_feature_values_sha256": frame_sha(X),
        "fit_geo_feature_values_sha256": frame_sha(geo),
        "fit_poi_feature_values_sha256": frame_sha(poi),
        "fit_augmented_feature_values_sha256": frame_sha(augmented),
        "fit_poi_position_rows": int(poi[POI_POSITION_COLUMN].sum()),
        "fit_poi_nonmissing_counts": {c: int(poi[c].notna().sum()) for c in POI_DISTANCE_COLUMNS},
        "fit_state_uses_y": False,
        "fit_state_uses_query": False,
        "fitted_feature_statistics": "none; supplied distances and missing values are retained",
    }
    state["state_content_sha256"] = content_sha(state)
    return state


def check_state(state, X, geo, poi, categorical_columns):
    need(state["state_content_sha256"] == content_sha(
        {k: v for k, v in state.items() if k != "state_content_sha256"}), "Feature state content changed")
    need(state["feature_rules"] == FEATURE_RULES and state["source_sha256"] == sha256(Path(__file__))
         and state["dependency_source_sha256"] == source_hashes(), "Feature rules or sources changed")
    need(state["base_configuration_sha256"] == sha256(BASE_CONFIGURATION)
         and state["parameters"] == parameters(), "Base model configuration changed")
    for label, frame in (("original", X), ("geo", geo), ("poi", poi)):
        need(isinstance(frame, pd.DataFrame), "Expected a feature DataFrame")
        need(list(frame.columns) == state[label + "_columns"]
             and {c: str(frame[c].dtype) for c in frame} == state[label + "_dtypes"],
             label + " feature schema or dtypes changed")
    need(list(categorical_columns) == state["original_categorical_columns"]
         and _categories(X, categorical_columns) == state["categorical_columns"],
         "Categorical schema changed")
    need(state["fit_state_uses_y"] is False and state["fit_state_uses_query"] is False,
         "Unexpected query or target feature state")


def transform_features(state, X, geo, poi, categorical_columns):
    """Apply the immutable contract to fit or query frames without learning state."""
    check_state(state, X, geo, poi, categorical_columns)
    result = augment_features(X, geo, poi)
    need(list(result.columns) == state["feature_columns"] == state["all_feature_names"],
         "Saved model feature order differs")
    return result


def _targets(y, rows):
    target = np.asarray(y, dtype="float64")
    need(rows > 0 and target.shape == (rows,) and np.isfinite(target).all()
         and (target >= 0).all(), "Fit targets must be aligned, finite and nonnegative")
    return target, target / 1e6


def _prices(values, rows):
    prices = np.asarray(values, dtype="float64") * 1e6
    need(prices.shape == (rows,) and np.isfinite(prices).all(), "Invalid raw price predictions")
    return np.maximum(prices, 0.0)


def fit_model(X, y, geo, poi, categorical_columns):
    """Fit the fixed raw-MAE model on supplied rows; no eval_set or early stop."""
    from catboost import CatBoostRegressor, Pool, __version__ as catboost_version
    began = time.monotonic()
    began_utc = datetime.now(timezone.utc).isoformat()
    sources_before = source_hashes()
    target, scaled = _targets(y, len(X))
    state = fit_feature_state(X, geo, poi, categorical_columns)
    augmented = transform_features(state, X, geo, poi, categorical_columns)
    config = parameters()
    model = CatBoostRegressor(**config)
    train_began = time.monotonic()
    model.fit(Pool(augmented, scaled, cat_features=state["categorical_columns"], thread_count=4))
    training_seconds = time.monotonic() - train_began
    need(int(model.tree_count_) == ITERATIONS and list(model.feature_names_) == state["feature_columns"],
         "Trained tree count or model feature schema differs")
    need(source_hashes() == sources_before and state["base_configuration_sha256"] == sha256(BASE_CONFIGURATION),
         "Source or base configuration changed during fit")
    metadata = {
        "family": "catboost", "component": FEATURE_RULES["component"], "target_mode": "raw",
        "parameters": config, "iterations": ITERATIONS, "actual_iterations": int(model.tree_count_),
        "features": FEATURE_COUNT, "feature_count": FEATURE_COUNT, "categorical_feature_count": 23,
        "feature_columns": state["feature_columns"], "categorical_columns": state["categorical_columns"],
        "categorical_features": state["categorical_columns"], "sample_weight": "uniform",
        "target_formula": FEATURE_RULES["target_formula"], "prediction_formula": FEATURE_RULES["prediction_formula"],
        "prediction_units": "TWD", "feature_rules": state["feature_rules"],
        "feature_state_content_sha256": state["state_content_sha256"],
        "script_sha256": sources_before["src/round9_geo_poi.py"], "dependency_source_sha256": sources_before,
        "base_configuration_sha256": state["base_configuration_sha256"], "fit_rows": len(X),
        "fit_target_values_sha256": hashlib.sha256(target.astype("<f8").tobytes()).hexdigest(),
        "fit_transformed_target_values_sha256": hashlib.sha256(scaled.astype("<f8").tobytes()).hexdigest(),
        "fit_original_feature_values_sha256": state["fit_original_feature_values_sha256"],
        "fit_geo_feature_values_sha256": state["fit_geo_feature_values_sha256"],
        "fit_poi_feature_values_sha256": state["fit_poi_feature_values_sha256"],
        "fit_augmented_feature_values_sha256": state["fit_augmented_feature_values_sha256"],
        "target_used_for_geographic_state": False, "query_used_for_geographic_state": False,
        "reference_features_changed": False, "external_geography_used": True,
        "external_property_answers_used": False, "fit_uses_eval_set": False, "early_stopping_used": False,
        "missing_position_predictions_equal_old_model_guaranteed": False,
        "fit_started_at_utc": began_utc, "fit_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": training_seconds, "elapsed_seconds": time.monotonic() - began,
        "runtime": {"catboost": catboost_version, "numpy": np.__version__, "pandas": pd.__version__},
    }
    return model, state, metadata


def predict_model(model, state, X, geo, poi, categorical_columns):
    """Predict unrounded total-price TWD from a loaded or freshly fitted model."""
    from catboost import Pool
    augmented = transform_features(state, X, geo, poi, categorical_columns)
    need(list(model.feature_names_) == state["feature_columns"] and int(model.tree_count_) == ITERATIONS,
         "Saved model feature names or tree count differs")
    values = model.predict(Pool(augmented, cat_features=state["categorical_columns"], thread_count=4), thread_count=4)
    return _prices(values, len(X))


fit = fit_model
predict = predict_model


def self_check():
    """Bounded synthetic feature/scale checks; never train or read real data."""
    from house_features import build_features, get_categorical_columns
    from round6_geo import encode_geo_features
    import inspect
    checks = []

    def ok(name, condition):
        assert condition, name
        checks.append(name)

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError("Expected rejection: " + name)

    X = build_features(pd.DataFrame([
        {"鄉鎮市區": "大安區", "土地位置建物門牌": "臺北市大安區和平東路", "建案名稱": "合成甲", "建物型態": "住宅大樓"},
        {"鄉鎮市區": "中西區", "土地位置建物門牌": "臺南市中西區中正路", "建案名稱": "合成乙", "建物型態": "住宅大樓"},
        {"鄉鎮市區": "__MISSING__", "土地位置建物門牌": "未知", "建案名稱": "合成丙", "建物型態": "住宅大樓"},
    ], index=[17, 31, 46]))
    matched = pd.DataFrame({
        "geo_latitude": [25.0, 23.0, np.nan], "geo_longitude": [121.5, 120.2, np.nan],
        "geo_road_length_m": [1000.0, 20.0, np.nan], "geo_extent_diagonal_m": [900.0, 0.0, np.nan],
        "geo_nearby_fragment_grouping": [0.0, 0.0, np.nan],
        "geo_match_status": ["road_level", "planar_intersection", "ineligible_address"],
    }, index=X.index)
    geo = encode_geo_features(matched)
    poi = pd.DataFrame(np.nan, index=X.index, columns=POI_FEATURES, dtype="float32")
    poi.loc[:, POI_POSITION_COLUMN] = np.asarray([1, 1, 0], dtype="float32")
    poi.loc[17, POI_DISTANCE_COLUMNS] = np.log1p(np.arange(13, dtype="float64") * 100).astype("float32")
    originals = [frame.copy(deep=True) for frame in (X, geo, poi)]
    cats = get_categorical_columns(X)
    out = augment_features(X, geo, poi)
    base = augment_r6_features(X, geo)
    ok("109_features_and_23_unchanged_categories", len(out.columns) == 109 and len(_categories(X, cats)) == 23)
    pd.testing.assert_frame_equal(out[base.columns], base, check_exact=True)
    checks.append("original_95_values_dtypes_order_and_indices_unchanged")
    ok("all_14_added_columns_float32", all(out[c].dtype == np.dtype("float32") for c in POI_FEATURES))
    ok("zero_distance_retained_and_unavailable_categories_nan", out.loc[17, POI_DISTANCE_COLUMNS[0]] == 0 and out.loc[31, POI_DISTANCE_COLUMNS].isna().all())
    ok("missing_position_all_distances_remain_nan", out.loc[46, POI_DISTANCE_COLUMNS].isna().all())
    for frame, original in zip((X, geo, poi), originals):
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
    checks.append("augmentation_does_not_mutate_any_input")
    state = fit_feature_state(X, geo, poi, cats)
    frozen = json.dumps(state, sort_keys=True)
    reloaded = json.loads(frozen)
    pd.testing.assert_frame_equal(transform_features(reloaded, X, geo, poi, cats), out, check_exact=True)
    checks.append("json_roundtrip_preserves_transform")
    seq = pd.concat([transform_features(state, X.iloc[[i]], geo.iloc[[i]], poi.iloc[[i]], cats) for i in range(3)])
    pd.testing.assert_frame_equal(seq, out, check_exact=True)
    order = [2, 0, 1]
    pd.testing.assert_frame_equal(transform_features(state, X.iloc[order], geo.iloc[order], poi.iloc[order], cats), out.iloc[order], check_exact=True)
    checks.append("batch_single_rows_and_query_reordering_equal")
    ok("query_does_not_mutate_state", json.dumps(state, sort_keys=True) == frozen)
    rejected("poi_index_misalignment_rejected", lambda: augment_features(X, geo, poi.reset_index(drop=True)))
    rejected("geo_index_misalignment_rejected", lambda: augment_features(X, geo.reset_index(drop=True), poi))
    rejected("poi_column_reordering_rejected", lambda: augment_features(X, geo, poi[POI_FEATURES[::-1]]))
    rejected("missing_poi_column_rejected", lambda: augment_features(X, geo, poi.drop(columns=POI_DISTANCE_COLUMNS[0])))
    rejected("extra_poi_column_rejected", lambda: augment_features(X, geo, poi.assign(extra=np.float32(0))))
    bad = poi.copy(); bad[POI_DISTANCE_COLUMNS[0]] = bad[POI_DISTANCE_COLUMNS[0]].astype("float64")
    rejected("wrong_poi_dtype_rejected", lambda: augment_features(X, geo, bad))
    for label, value in [("negative_distance_rejected", -1.0), ("infinite_distance_rejected", np.inf)]:
        bad = poi.copy(); bad.loc[17, POI_DISTANCE_COLUMNS[0]] = np.float32(value)
        rejected(label, lambda: augment_features(X, geo, bad))
    for label, value in [("nonbinary_flag_rejected", 2.0), ("missing_flag_rejected", np.nan)]:
        bad = poi.copy(); bad.loc[17, POI_POSITION_COLUMN] = np.float32(value)
        rejected(label, lambda: augment_features(X, geo, bad))
    bad = poi.copy(); bad.loc[46, POI_DISTANCE_COLUMNS[0]] = np.float32(0)
    rejected("unusable_position_with_zero_distance_rejected", lambda: augment_features(X, geo, bad))
    bad = poi.copy(); bad.loc[46, POI_POSITION_COLUMN] = np.float32(1)
    rejected("poi_position_without_geo_position_rejected", lambda: augment_features(X, geo, bad))
    bad_state = json.loads(frozen); bad_state["fit_rows"] += 1
    rejected("changed_state_content_rejected", lambda: transform_features(bad_state, X, geo, poi, cats))
    rejected("categorical_order_changed_rejected", lambda: transform_features(state, X, geo, poi, cats[::-1]))
    target, scaled = _targets([0, 1000000, 25300000], 3)
    ok("raw_target_divisor_is_one_million", np.array_equal(scaled, [0, 1, 25.3]))
    ok("prediction_total_price_units_and_nonnegative_clip", np.array_equal(_prices([-2, 1, 25.3], 3), [0, 1000000, 25300000]))
    rejected("nonfinite_prediction_rejected", lambda: _prices([np.nan], 1))
    rejected("negative_fit_target_rejected", lambda: _targets([-1], 1))
    rejected("unaligned_fit_target_rejected", lambda: _targets([1], 2))
    config = parameters()
    ok("fixed_2999_original_raw_parameters_without_earlystop", config == r6_parameters() and config["iterations"] == 2999)
    ok("pure_feature_and_query_apis_have_no_y", all("y" not in inspect.signature(function).parameters for function in (augment_features, fit_feature_state, transform_features, predict_model)))
    ok("missing_feature_is_not_a_prediction_equality_claim", FEATURE_RULES["missing_position_predictions_equal_old_model_guaranteed"] is False)
    ok("fit_state_no_target_or_query_statistics", not state["fit_state_uses_y"] and not state["fit_state_uses_query"])
    return {"status": "passed", "checks": checks, "passed_count": len(checks), "failed_count": 0,
            "synthetic_rows": 3, "feature_count": 109, "categorical_feature_count": 23,
            "model_training_performed": False, "real_data_read": False, "real_scoring_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, indent=2))