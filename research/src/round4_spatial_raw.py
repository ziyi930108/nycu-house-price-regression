"""Pure fit/predict API for the fixed round-four raw-price spatial LightGBM.

No datasets are loaded and no model is trained on import. The caller owns split
selection, artifact paths, model serialization, file hashes, and evaluation.
The returned maps bundle is JSON serializable and preserves feature order for
fresh-process inference after loading a LightGBM Booster from its saved file.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

NAME = 'lightgbm_l1_raw_spatial63'
DROP_FEATURES = ['project_name', 'address_raw', 'building_and_unit', 'building_block']
PARAMETERS = {
    'objective': 'regression_l1', 'metric': 'None', 'n_estimators': 4000,
    'learning_rate': 0.04, 'num_leaves': 63, 'max_depth': -1,
    'min_child_samples': 80, 'colsample_bytree': 0.9,
    'subsample': 0.9, 'subsample_freq': 1, 'reg_lambda': 5.0,
    'cat_smooth': 100.0, 'cat_l2': 30.0, 'max_cat_to_onehot': 4,
    'max_cat_threshold': 64, 'random_state': 42, 'n_jobs': 4,
    'verbosity': -1, 'deterministic': True, 'force_col_wise': True,
}


def _content_hash(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _frame_hash(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), ensure_ascii=False).encode('utf-8'))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype='uint64').tobytes())
    return digest.hexdigest()


def _source_hashes():
    source = Path(__file__).resolve()
    root = source.parents[1]
    paths = [source, source.with_name('train_lightgbm.py')]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _features(X, categorical_columns):
    if not isinstance(X, pd.DataFrame) or not len(X):
        raise ValueError('X must be a nonempty pandas DataFrame')
    if not X.columns.is_unique:
        raise ValueError('Duplicate feature names are not allowed')
    if any(name in X.columns for name in ['編號', '總價元']):
        raise ValueError('Pass feature columns only, without ID or target')
    unknown_categories = set(categorical_columns) - set(X.columns) - set(DROP_FEATURES)
    if unknown_categories:
        raise ValueError(f'Categorical columns absent from X: {sorted(unknown_categories)}')
    features = X.drop(columns=DROP_FEATURES, errors='ignore')
    if len(features.columns) != 83:
        raise ValueError('This fixed candidate requires the existing 83 spatial features after the four exclusions')
    categorical = [name for name in features.columns if name in set(categorical_columns)]
    if len(categorical) != 17:
        raise ValueError('This fixed candidate requires the existing 17 spatial categorical features')
    return features, categorical


def fit_spatial_raw(X, y, categorical_columns):
    """Return (LGBMRegressor, JSON maps bundle, metadata) from fitting rows only.

    Target: total price / 1e6. All sample weights are one. Training always uses
    4,000 rounds and four CPU threads; there is no evaluation set or early stop.
    The API does not save artifacts, predict a holdout, or calculate any score.
    """
    import lightgbm as lgb
    from train_lightgbm import encode_features

    features, categorical = _features(X, categorical_columns)
    target = np.asarray(y, dtype='float64')
    if target.shape != (len(features),) or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError('y must be an aligned finite nonnegative total-price vector')
    sources_before = _source_hashes()
    parameters = dict(PARAMETERS)
    encoded, category_maps, safe_names, safe_categories = encode_features(
        {'train': features}, categorical)
    schema = {
        'feature_columns': list(features.columns), 'categorical_columns': categorical,
        'safe_feature_names': safe_names, 'safe_categorical_features': safe_categories,
        'excluded_features': list(DROP_FEATURES), 'numeric_dtype': 'float32',
        'categorical_dtype': 'int32', 'unseen_category_code': -1,
    }
    maps = {
        'format_version': 1, 'candidate': NAME, 'schema': schema,
        'schema_content_sha256': _content_hash(schema),
        'category_maps': category_maps,
        'category_maps_content_sha256': _content_hash(category_maps),
    }
    started = time.perf_counter()
    model = lgb.LGBMRegressor(**parameters)
    model.fit(encoded['train'], target / 1e6,
              sample_weight=np.ones(len(target), dtype='float64'),
              categorical_feature=safe_categories)
    elapsed = time.perf_counter() - started
    actual_iterations = int(model.booster_.current_iteration())
    if actual_iterations != PARAMETERS['n_estimators']:
        raise RuntimeError(f'Expected 4000 fixed boosting rounds, got {actual_iterations}')
    if sources_before != _source_hashes():
        raise RuntimeError('Training API or encoding source changed during fitting')
    metadata = {
        'name': NAME, 'family': 'lightgbm', 'target_mode': 'raw',
        'target_formula': 'y / 1e6', 'prediction_formula': 'max(model_output * 1e6, 0)',
        'prediction_units': 'TWD', 'sample_weight': 'uniform ones',
        'loss': 'regression_l1', 'parameters': parameters,
        'parameters_content_sha256': _content_hash(parameters),
        'requested_iterations': 4000, 'actual_iterations': actual_iterations,
        'iterations': actual_iterations, 'training_rows': len(features),
        'feature_columns': list(features.columns), 'categorical_columns': categorical,
        'feature_count': len(features.columns), 'categorical_feature_count': len(categorical),
        'feature_exclusions': list(DROP_FEATURES), 'schema': schema,
        'schema_content_sha256': _content_hash(schema),
        'maps_content_sha256': _content_hash(maps),
        'category_maps_content_sha256': _content_hash(category_maps),
        'fit_feature_values_sha256': _frame_hash(features),
        'fit_encoded_feature_values_sha256': _frame_hash(encoded['train']),
        'fit_target_values_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'source_sha256': sources_before,
        'lightgbm_version': lgb.__version__, 'elapsed_seconds': elapsed,
        'category_mapping_fit_side_only': True, 'unseen_category_code': -1,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'iterations_fixed_before_evaluation': True,
        'external_property_data_used': False, 'query_labels_used': False,
        'api_saves_artifacts': False,
        'hash_note': 'Content hashes use canonical JSON or dataframe values; the caller records model/map/metadata file SHA256 after saving.',
    }
    json.dumps(maps, ensure_ascii=False, allow_nan=False)
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    return model, maps, metadata


def predict_spatial_raw(model, maps, X, categorical_columns):
    """Predict float64 total prices in TWD from an in-memory regressor or Booster.

    Maps may be freshly returned or loaded from JSON. Inference never adds new
    categories: unknown values keep code -1. Feature order is restored from the
    fitted schema, and changed, missing, or extra feature columns are rejected.
    """
    from train_lightgbm import encode_features

    if maps.get('format_version') != 1 or maps.get('candidate') != NAME:
        raise ValueError('Wrong category-map bundle format or candidate')
    schema = maps['schema']
    if maps['schema_content_sha256'] != _content_hash(schema):
        raise ValueError('Feature schema content hash mismatch')
    if maps['category_maps_content_sha256'] != _content_hash(maps['category_maps']):
        raise ValueError('Category map content hash mismatch')
    if schema['excluded_features'] != DROP_FEATURES:
        raise ValueError('Feature exclusion contract changed')
    features, categorical = _features(X, categorical_columns)
    columns = schema['feature_columns']
    if set(features.columns) != set(columns) or set(categorical) != set(schema['categorical_columns']):
        raise ValueError('Inference feature or categorical schema differs from fitting')
    features = features.loc[:, columns]
    encoded, _, safe_names, safe_categories = encode_features(
        {'train': features}, schema['categorical_columns'], category_maps=maps['category_maps'])
    if safe_names != schema['safe_feature_names'] or safe_categories != schema['safe_categorical_features']:
        raise ValueError('Safe feature mapping differs from fitting')
    booster = model.booster_ if hasattr(model, 'booster_') else model
    if list(booster.feature_name()) != list(encoded['train'].columns):
        raise ValueError('Loaded model feature names differ from the saved schema')
    if int(booster.current_iteration()) != 4000:
        raise ValueError('Loaded model does not contain the fixed 4000 boosting rounds')
    prediction = np.maximum(
        np.asarray(booster.predict(encoded['train'], num_iteration=4000, num_threads=4), dtype='float64') * 1e6,
        0.0)
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid total-price predictions')
    return prediction
