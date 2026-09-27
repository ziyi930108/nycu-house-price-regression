"""Target-free, fit-vocabulary categorical view of the original R6 95 features.

This module does not import XGBoost, fit a model, read data caches, or define a
target, loss, sample weight, or training parameter.  The caller supplies the
original 87 features and the already aligned R6 geography features.  Importing
round6_geo reuses its existing schema/encoding code, not its model functions.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import round6_geo as base


FEATURE_RULES = {
    "version": 1,
    "encoder": "r6_geo95_fit_only_pandas_categorical",
    "feature_count": 95,
    "categorical_feature_count": 23,
    "numeric_feature_count": 72,
    "original_feature_names_order_index_preserved": True,
    "numeric_dtype": "float32",
    "numeric_policy": "preserve input values and NaN; reject infinity or other dtype",
    "category_input": "strings only; no new normalization or missing-token replacement",
    "vocabulary_fit_scope": "supplied fit X and fit geography only",
    "vocabulary_order": "unique strings sorted by Python string order",
    "categorical_dtype": "pandas Categorical with saved fit categories; ordered=False",
    "unseen_category": "NaN, categorical code -1; never extend the fit vocabulary",
    "existing_missing_string": "a normal category if present in fit, otherwise unseen",
    "state_uses_y": False,
    "state_uses_query": False,
    "geo_state_rules_scope": "original R6 encoder provenance; not a new training declaration",
    "target_encoding": False,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def content_sha(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def source_hashes():
    """Hash only this encoder and its local source dependencies, not data."""
    root = Path(__file__).resolve().parents[1]
    names = ("src/xgboost_geo_features.py", "src/round6_geo.py", "src/prepare_data.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _view(X, geo, categorical_columns):
    augmented = base.augment_features(X, geo)
    need(augmented.columns.is_unique, "Duplicate feature names are not supported")
    need(all(isinstance(c, str) for c in augmented.columns), "Feature names must be strings")
    cats = base.categories(X, categorical_columns)
    need(len(cats) == 23 and len(set(cats)) == 23, "Expected 23 distinct categorical columns")
    numeric = [c for c in augmented.columns if c not in cats]
    need(len(numeric) == 72, "Expected 72 numeric columns")
    for column in cats:
        need(augmented[column].map(lambda v: isinstance(v, str)).all(),
             f"Categorical input must contain strings, including explicit missing tokens: {column}")
    for column in numeric:
        need(augmented[column].dtype == np.dtype("float32"),
             f"Numeric input must already be float32: {column}")
        need(not np.isinf(augmented[column].to_numpy(copy=False)).any(),
             f"Infinite numeric input is not allowed: {column}")
    return augmented, cats, numeric


def _encode(augmented, cats, vocabularies):
    result = augmented.copy(deep=True)
    for column in cats:
        dtype = pd.CategoricalDtype(categories=vocabularies[column], ordered=False)
        result[column] = pd.Series(
            pd.Categorical(augmented[column], dtype=dtype), index=augmented.index,
        )
    need(result.index.equals(augmented.index)
         and list(result.columns) == list(augmented.columns),
         "Encoding changed row or feature order")
    return result


def fit_feature_state(X, geo, categorical_columns):
    """Build JSON-serializable state from fit features only; no target argument."""
    need(len(X) > 0, "Cannot learn a categorical vocabulary from an empty fit set")
    augmented, cats, numeric = _view(X, geo, categorical_columns)
    vocabularies = {c: sorted(set(augmented[c].tolist())) for c in cats}
    encoded = _encode(augmented, cats, vocabularies)
    state = {
        "format_version": 1,
        "feature_rules": copy.deepcopy(FEATURE_RULES),
        "source_sha256": source_hashes(),
        "geo_state": copy.deepcopy(base.fit_feature_state(X, geo)),
        "fit_rows": len(X),
        "feature_columns": list(augmented.columns),
        "model_feature_columns": list(augmented.columns),
        "original_categorical_columns": list(categorical_columns),
        "categorical_columns": list(cats),
        "numeric_columns": numeric,
        "category_vocabularies": vocabularies,
        "vocabulary_content_sha256": content_sha(vocabularies),
        "fit_original_feature_values_sha256": base.frame_sha(X),
        "fit_geo_feature_values_sha256": base.frame_sha(geo),
        "fit_augmented_feature_values_sha256": base.frame_sha(augmented),
        "fit_encoded_feature_values_sha256": base.frame_sha(encoded),
        "fit_state_uses_y": False,
        "fit_state_uses_query": False,
    }
    state["state_content_sha256"] = content_sha(state)
    return state


def check_state(state, X, categorical_columns):
    """Validate saved state and the original schema, without fitting a vocabulary."""
    need(state["state_content_sha256"] == content_sha(
        {k: v for k, v in state.items() if k != "state_content_sha256"}),
        "Feature state content changed")
    need(state["format_version"] == 1 and state["feature_rules"] == FEATURE_RULES,
         "Encoder rules changed")
    need(state["source_sha256"] == source_hashes(), "Encoder source identity changed")
    base.check_state(state["geo_state"], X)
    need(list(categorical_columns) == state["original_categorical_columns"],
         "Original categorical column order changed")
    cats = base.categories(X, categorical_columns)
    need(cats == state["categorical_columns"] and len(cats) == 23,
         "Categorical feature schema changed")
    columns = list(X.columns) + list(base.FEATURES)
    need(columns == state["feature_columns"] == state["model_feature_columns"]
         == state["geo_state"]["all_feature_names"] and len(columns) == 95,
         "Feature names or order changed")
    need(state["numeric_columns"] == [c for c in columns if c not in cats]
         and len(state["numeric_columns"]) == 72, "Numeric schema changed")
    vocabularies = state["category_vocabularies"]
    need(set(vocabularies) == set(cats)
         and state["vocabulary_content_sha256"] == content_sha(vocabularies),
         "Saved vocabulary identity changed")
    for column in cats:
        values = vocabularies[column]
        need(isinstance(values, list) and len(values) > 0
             and all(isinstance(v, str) for v in values), "Invalid saved vocabulary")
        need(values == sorted(set(values)), "Saved vocabulary order or uniqueness changed")
    need(state["fit_rows"] == state["geo_state"]["fit_rows"] > 0
         and state["fit_state_uses_y"] is False and state["fit_state_uses_query"] is False,
         "Unexpected fit-state scope")
    need(state["fit_original_feature_values_sha256"]
         == state["geo_state"]["fit_original_feature_values_sha256"]
         and state["fit_geo_feature_values_sha256"]
         == state["geo_state"]["fit_geo_feature_values_sha256"],
         "Original and geography fit fingerprints disagree")


def encode_query(state, X, geo, categorical_columns):
    """Apply fit categories to any aligned batch; unseen strings become NaN."""
    check_state(state, X, categorical_columns)
    augmented, cats, numeric = _view(X, geo, categorical_columns)
    need(list(augmented.columns) == state["feature_columns"]
         and numeric == state["numeric_columns"], "Augmented feature schema changed")
    return _encode(augmented, cats, state["category_vocabularies"])


# Same signature as encode_query; callers never pass targets or model parameters.
transform_features = encode_query


def self_check():
    """Small in-memory encoder checks for a future preflight; no XGBoost or fit."""
    index = pd.Index([7, 2, 19, 30], name="source_row")
    cats = [f"category_{i:02d}" for i in range(21)]
    numeric = [f"numeric_{i:02d}" for i in range(66)]
    X = pd.DataFrame({c: ["__MISSING__", "z", "a", "z"] for c in cats}, index=index)
    for column in numeric:
        X[column] = np.asarray([1.25, np.nan, -0.0, 3.0], dtype="float32")
    matched = pd.DataFrame({
        "geo_latitude": [25.0, 25.1, 25.2, 25.3],
        "geo_longitude": [121.0, 121.1, 121.2, 121.3],
        "geo_road_length_m": [100.0] * 4,
        "geo_extent_diagonal_m": [90.0] * 4,
        "geo_nearby_fragment_grouping": [0, 1, 0, 1],
        "geo_match_status": ["road_level"] * 4,
    }, index=index)
    geo = base.encode_geo_features(matched)
    before = (base.frame_sha(X), base.frame_sha(geo))
    state = fit_feature_state(X, geo, cats)
    encoded = encode_query(state, X, geo, cats)
    restored = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    pd.testing.assert_frame_equal(encoded, encode_query(restored, X, geo, cats))
    need(encoded.shape == (4, 95)
         and sum(isinstance(d, pd.CategoricalDtype) for d in encoded.dtypes) == 23,
         "Encoded feature counts differ")
    augmented = base.augment_features(X, geo)
    for column in state["numeric_columns"]:
        need(encoded[column].dtype == np.dtype("float32") and np.array_equal(
            encoded[column].to_numpy().view("uint32"),
            augmented[column].to_numpy().view("uint32")), "Numeric bits changed")
    need(not pd.isna(encoded.loc[7, cats[0]])
         and "__MISSING__" in encoded[cats[0]].cat.categories,
         "A fitted missing string was converted to unseen")
    query, query_geo = X.copy(deep=True), geo.copy(deep=True)
    query.loc[7, cats[0]] = "ONLY_IN_QUERY"
    query_geo.loc[7, "geo_grid_002"] = "ONLY_IN_QUERY_GRID"
    state_before = content_sha(state)
    output = encode_query(state, query, query_geo, cats)
    need(pd.isna(output.loc[7, cats[0]]) and output[cats[0]].cat.codes.loc[7] == -1
         and pd.isna(output.loc[7, "geo_grid_002"]), "Unseen query strings were learned")
    need(list(output[cats[0]].cat.categories) == state["category_vocabularies"][cats[0]]
         and content_sha(state) == state_before, "Query mutated the fit vocabulary")
    pd.testing.assert_frame_equal(
        output.iloc[[2, 0]], encode_query(state, query.iloc[[2, 0]], query_geo.iloc[[2, 0]], cats))
    need(encode_query(state, X.iloc[:0], geo.iloc[:0], cats).shape == (0, 95),
         "Empty query batch schema changed")
    failures = []
    bad_inf = query.copy(deep=True); bad_inf.loc[7, numeric[0]] = np.float32(np.inf)
    bad_dtype = query.copy(deep=True); bad_dtype[numeric[0]] = bad_dtype[numeric[0]].astype("float64")
    bad_string = query.copy(deep=True); bad_string.loc[7, cats[0]] = None
    bad_state = copy.deepcopy(state); bad_state["category_vocabularies"][cats[0]].append("tamper")
    for name, saved, features, geography in (
        ("misaligned_geo_index", state, query, query_geo.iloc[::-1]),
        ("infinite_numeric", state, bad_inf, query_geo),
        ("changed_numeric_dtype", state, bad_dtype, query_geo),
        ("nonstring_category", state, bad_string, query_geo),
        ("state_mutation", bad_state, query, query_geo),
    ):
        rejected = False
        try:
            encode_query(saved, features, geography, cats)
        except ValueError:
            rejected = True
        need(rejected, f"Expected rejection: {name}")
        failures.append(name)
    need(before == (base.frame_sha(X), base.frame_sha(geo)), "Fit frames were mutated")
    return {"status": "passed", "real_data_read": False, "model_training_performed": False,
            "xgboost_imported_here": False, "file_outputs_written": False,
            "checks": ["95_features_23_categories_72_numeric", "JSON_state_roundtrip",
                       "numeric_float32_bits_preserved", "fitted_missing_string_preserved",
                       "query_unknown_is_NaN", "fit_vocabulary_immutable",
                       "sliced_index_and_batch_consistency", "empty_query", "input_frames_unchanged"]
                      + failures}
