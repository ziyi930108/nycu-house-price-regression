"""R32 raw-price XGBoost over the unchanged R6 95-feature information.

The fit vocabulary is target-free. Training configuration is checked separately
from the objective/schema/trees that persist in a UBJSON model. No data files,
old models, or query labels are read by this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import xgboost_geo_features as encoder


COMPONENT = "geography_xgboost_raw"
TARGET_MODE = "raw"
TARGET_FORMULA = "y_TWD / 1e6"
INVERSE_FORMULA = "max(z * 1e6, 0)"
NUM_ROUNDS = 2999
XGBOOST_VERSION = "3.4.1"
PARAMETERS = {
    "booster": "gbtree", "tree_method": "hist", "device": "cuda:0",
    "objective": "reg:absoluteerror", "eval_metric": "mae",
    "seed": 42, "nthread": 4, "eta": .065, "max_depth": 8,
    "max_cat_to_onehot": 4, "max_cat_threshold": 64,
    "subsample": 1.0, "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
}
FEATURE_RULES = {
    "feature_count": 95, "categorical_feature_count": 23,
    "numeric_feature_count": 72, "original_information_preserved": True,
    "added_feature_names": [], "removed_feature_names": [],
    "encoder_rules": copy.deepcopy(encoder.FEATURE_RULES),
    "target_formula": TARGET_FORMULA, "inverse_formula": INVERSE_FORMULA,
    "sample_weight": "uniform", "fit_uses_eval_set": False,
    "early_stopping_used": False, "query_labels_used": False,
    "model_format": "UBJSON", "native_float_rtol": 1e-7,
    "native_float_atol": 0.0,
    "reload_training_configuration_equality_required": False,
    "reference_features_changed": False, "trend_applied_here": False,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def content_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def values_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def source_hashes():
    return {**encoder.source_hashes(),
            "src/round32_xgboost_geo.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def parameters():
    return dict(PARAMETERS)


def _version():
    need(xgb.__version__ == XGBOOST_VERSION, "XGBoost version differs from fixed 3.4.1")


def transform_target(y):
    values = np.asarray(y, dtype="float64")
    need(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
         and (values >= 0).all(), "Expected nonempty finite nonnegative raw target")
    transformed = values / 1e6
    need(np.isfinite(transformed).all(), "Nonfinite transformed target")
    return transformed


def inverse_transform(z):
    values = np.asarray(z, dtype="float64")
    need(values.ndim == 1 and np.isfinite(values).all(), "Expected finite raw prediction vector")
    with np.errstate(over="ignore", invalid="ignore"):
        prices = values * 1e6
    need(np.isfinite(prices).all(), "Raw-to-TWD multiplication overflow")
    return np.maximum(prices, 0.0)


def fit_feature_state(X, geo, cats):
    """No target or query argument; preserve the original encoder's fit scope."""
    saved = encoder.fit_feature_state(X, geo, cats)
    columns = list(saved["model_feature_columns"])
    categorical = list(saved["categorical_columns"])
    state = {
        "format_version": 1, "component": COMPONENT, "target_mode": TARGET_MODE,
        "target_formula": TARGET_FORMULA, "inverse_formula": INVERSE_FORMULA,
        "parameters": parameters(), "num_boost_rounds": NUM_ROUNDS,
        "feature_rules": copy.deepcopy(FEATURE_RULES), "source_sha256": source_hashes(),
        "encoder_state": saved, "all_feature_names": columns,
        "categorical_columns": categorical,
        "feature_types": ["c" if c in categorical else "float" for c in columns],
        "fit_rows": len(X), "fit_state_uses_y": False, "fit_state_uses_query": False,
        "query_labels_used": False,
    }
    state["state_content_sha256"] = content_sha(state)
    return state


def _check_state_contract(state):
    need(state["state_content_sha256"] == content_sha(
        {k: v for k, v in state.items() if k != "state_content_sha256"}), "State content changed")
    need(state["format_version"] == 1 and state["component"] == COMPONENT
         and state["target_mode"] == TARGET_MODE and state["target_formula"] == TARGET_FORMULA
         and state["inverse_formula"] == INVERSE_FORMULA, "Model or target identity changed")
    need(state["parameters"] == parameters() and state["num_boost_rounds"] == NUM_ROUNDS
         and state["feature_rules"] == FEATURE_RULES and state["source_sha256"] == source_hashes(),
         "Fixed source or configuration changed")
    saved = state["encoder_state"]
    need(state["all_feature_names"] == saved["model_feature_columns"]
         and len(state["all_feature_names"]) == 95
         and state["categorical_columns"] == saved["categorical_columns"]
         and len(state["categorical_columns"]) == 23, "Unexpected feature schema")
    expected = ["c" if c in state["categorical_columns"] else "float"
                for c in state["all_feature_names"]]
    need(state["feature_types"] == expected and expected.count("c") == 23
         and expected.count("float") == 72, "Native feature type contract changed")
    need(state["fit_rows"] == saved["fit_rows"] > 0
         and state["fit_state_uses_y"] is False and state["fit_state_uses_query"] is False
         and state["query_labels_used"] is False, "Unexpected fit-state scope")


def check_state(state, X, cats):
    _check_state_contract(state)
    encoder.check_state(state["encoder_state"], X, cats)


def _configuration(model_or_config):
    if isinstance(model_or_config, dict):
        return copy.deepcopy(model_or_config)
    if isinstance(model_or_config, str):
        return json.loads(model_or_config)
    return json.loads(model_or_config.save_config())


def check_fitted_configuration(model_or_config):
    """Check all requested settings only on the fitted native configuration.

    The full dict, including version-specific defaults, must separately be saved
    as metadata.fitted_config. This check is intentionally not used after load.
    """
    config = _configuration(model_or_config)
    need(config["version"] == [3, 4, 1], "Native configuration version differs")
    learner = config["learner"]
    generic = learner["generic_param"]
    booster = learner["gradient_booster"]
    tree = booster["tree_train_param"]
    metric_names = [m["name"] for m in learner["metrics"]]
    need(metric_names == ["mae"], "Expected only the fixed MAE evaluation metric")
    actual = {
        "booster": booster["name"],
        "tree_method": booster["gbtree_train_param"]["tree_method"],
        "device": generic["device"], "objective": learner["objective"]["name"],
        "eval_metric": metric_names[0], "seed": int(generic["seed"]),
        "nthread": int(generic["nthread"]),
    }
    for key in ("max_depth", "max_cat_to_onehot", "max_cat_threshold"):
        actual[key] = int(tree[key])
    for key in ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"):
        actual[key] = float(tree[key])
    actual["eta"] = float(tree["eta"] if "eta" in tree else tree["learning_rate"])
    for key, expected in PARAMETERS.items():
        if isinstance(expected, float):
            need(np.isfinite(actual[key]) and np.isclose(actual[key], expected, rtol=1e-7, atol=0),
                 f"Native fitted parameter differs: {key}")
        else:
            need(actual[key] == expected, f"Native fitted parameter differs: {key}")
    if "learning_rate" in tree:
        need(np.isclose(float(tree["learning_rate"]), PARAMETERS["eta"], rtol=1e-7, atol=0),
             "Native learning-rate alias differs")
    return actual


def check_model(model, state):
    """Validate persisted model content; never require training settings after load."""
    _version()
    _check_state_contract(state)
    config = _configuration(model)
    learner = config["learner"]
    booster = learner["gradient_booster"]
    need(booster["name"] == "gbtree"
         and learner["objective"]["name"] == "reg:absoluteerror", "Persisted learner differs")
    model_param = booster["gbtree_model_param"]
    trees, parallel = int(model_param["num_trees"]), int(model_param["num_parallel_tree"])
    need(trees == NUM_ROUNDS and parallel == 1
         and model.num_boosted_rounds() == NUM_ROUNDS, "Persisted tree count differs")
    need(model.num_features() == 95 and model.feature_names == state["all_feature_names"]
         and model.feature_types == state["feature_types"], "Persisted feature schema differs")
    return {"actual_booster": booster["name"], "actual_objective": learner["objective"]["name"],
            "actual_trees": trees, "actual_num_parallel_tree": parallel,
            "actual_boosted_rounds": int(model.num_boosted_rounds()),
            "feature_count": 95, "categorical_feature_count": 23}


def _matrix(encoded, label=None):
    return xgb.DMatrix(encoded, label=label, enable_categorical=True,
                       missing=np.nan, nthread=PARAMETERS["nthread"])


def fit_model(X, y, geo, cats):
    """Fit on supplied rows only; caller owns stage authorization and artifact IO."""
    _version()
    started = time.monotonic()
    sources = source_hashes()
    if isinstance(y, pd.Series):
        need(y.index.equals(X.index), "Target index differs from fit features")
    original_target = np.asarray(y, dtype="float64")
    target = transform_target(original_target)
    need(target.shape == (len(X),), "Target length differs from fit features")
    state = fit_feature_state(X, geo, cats)
    encoded = encoder.encode_query(state["encoder_state"], X, geo, cats)
    dtrain = _matrix(encoded, target)
    processing_seconds = time.monotonic() - started
    training_started = time.monotonic()
    model = xgb.train(parameters(), dtrain, num_boost_round=NUM_ROUNDS,
                      evals=(), verbose_eval=False)
    training_seconds = time.monotonic() - training_started
    fitted_config = _configuration(model)
    actual = check_fitted_configuration(fitted_config)
    persisted = check_model(model, state)
    need(source_hashes() == sources == state["source_sha256"], "Source changed during fit")
    saved = state["encoder_state"]
    metadata = {
        "family": "xgboost", "component": COMPONENT, "target_mode": TARGET_MODE,
        "parameters": parameters(), "num_boost_rounds": NUM_ROUNDS,
        "iterations": NUM_ROUNDS, "actual_iterations": NUM_ROUNDS,
        "feature_count": 95, "categorical_feature_count": 23,
        "categorical_features": list(state["categorical_columns"]),
        "feature_names": list(state["all_feature_names"]), "feature_types": list(state["feature_types"]),
        "baseline_feature_count": 95, "added_feature_names": [], "removed_feature_names": [],
        "target_formula": TARGET_FORMULA, "inverse_formula": INVERSE_FORMULA,
        "raw_z_units": "million_TWD", "sample_weight": "uniform",
        "weight_argument_passed": False, "fit_uses_eval_set": False,
        "early_stopping_used": False, "query_labels_used": False,
        "reference_features_changed": False, "trend_applied_here": False,
        "calibration_applied": False, "extra_price_cap_applied": False,
        "fit_rows": len(X), "fit_original_target_values_sha256": values_sha(original_target),
        "fit_transformed_target_values_sha256": values_sha(target),
        "fit_target_values_sha256": values_sha(original_target),
        "fit_original_feature_values_sha256": saved["fit_original_feature_values_sha256"],
        "fit_geo_feature_values_sha256": saved["fit_geo_feature_values_sha256"],
        "fit_model_feature_values_sha256": saved["fit_encoded_feature_values_sha256"],
        "feature_state_content_sha256": state["state_content_sha256"],
        "source_sha256": sources, "feature_rules": copy.deepcopy(FEATURE_RULES),
        "fitted_config": fitted_config, "fitted_config_sha256": content_sha(fitted_config),
        "actual_fitted_parameters": actual, "persistent_model_checks": persisted,
        "runtime": {"xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "native_build_info": xgb.build_info(),
        "feature_processing_seconds": processing_seconds, "training_seconds": training_seconds,
        "elapsed_seconds": time.monotonic() - started,
    }
    return model, state, metadata


def raw_predict(model, state, X, geo, cats):
    check_state(state, X, cats)
    check_model(model, state)
    encoded = encoder.encode_query(state["encoder_state"], X, geo, cats)
    if len(encoded) == 0:
        return np.empty(0, dtype="float64")
    result = np.asarray(model.predict(_matrix(encoded)), dtype="float64")
    need(result.shape == (len(X),) and np.isfinite(result).all(), "Invalid raw query predictions")
    return result


def predict_model(model, state, X, geo, cats):
    return inverse_transform(raw_predict(model, state, X, geo, cats))


def load_model(path):
    _version()
    path = Path(path)
    need(path.suffix.lower() == ".ubj", "Production model must use UBJSON")
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({"device": PARAMETERS["device"], "nthread": PARAMETERS["nthread"]})
    return model


def self_check():
    """Only small target/inverse/configuration checks; no encoder or model execution."""
    checks = []
    def ok(name, condition):
        need(condition, name); checks.append(name)
    def reject(name, fn):
        try:
            fn()
        except (ValueError, OverflowError):
            checks.append(name); return
        raise ValueError(f"Expected rejection: {name}")
    ok("raw_scale", np.array_equal(transform_target([0, 1e6, 2.5e6]), [0, 1, 2.5]))
    ok("float64_nonnegative_inverse", np.array_equal(inverse_transform([-1, 0, 2.5]), [0, 0, 2.5e6])
       and inverse_transform([1]).dtype == np.dtype("float64"))
    ok("empty_inverse", inverse_transform([]).shape == (0,))
    reject("negative_target", lambda: transform_target([-1]))
    reject("nonfinite_target", lambda: transform_target([np.nan]))
    reject("target_matrix", lambda: transform_target([[1]]))
    reject("nonfinite_raw", lambda: inverse_transform([np.inf]))
    reject("TWD_overflow", lambda: inverse_transform([np.finfo("float64").max]))
    changed = parameters(); changed["eta"] = 1
    ok("parameter_copy_and_fixed_budget", parameters()["eta"] == .065 and NUM_ROUNDS == 2999
       and PARAMETERS["objective"] == "reg:absoluteerror" and PARAMETERS["eval_metric"] == "mae")
    return {"status": "passed", "checks": checks, "check_count": len(checks),
            "model_training_performed": False, "encoder_self_check_executed": False,
            "real_data_read": False, "file_outputs_written": False}
