"""Fixed R14 raw-price LightGBM on the unchanged 95-column R6 geography view.

This module loads no training/query data and writes no artifacts. The caller owns
stage authorization, model publication and the original-87 reference/trend.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import round6_geo as base
from train_lightgbm import candidate_params, encode_features

ROOT = base.ROOT
COMPONENT = 'geography_raw_lightgbm'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ITERATIONS = 3999


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    **_copy_json({k: v for k, v in base.FEATURE_RULES.items()
                 if k not in {'fit_target', 'sample_weight', 'iterations'}}),
    'component': COMPONENT,
    'feature_count': 95,
    'categorical_feature_count': 23,
    'existing_95_values_dtypes_order_and_index_preserved_before_encoding': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'sample_weight': 'uniform',
    'sample_weight_argument_passed': False,
    'area_used_for_target_or_prediction_scaling': False,
    'iterations': ITERATIONS,
    'num_leaves': 63,
    'threads': 4,
    'seed': 42,
    'category_mapping_fit_scope': 'fit X plus fit geography only',
    'unseen_category_code': -1,
    'feature_exclusions': [],
    'trend_applications_in_component': 0,
    'extra_price_cap_applied': False,
}


def source_hashes():
    return {name: base.sha256(ROOT / name) for name in (
        'src/round14_geo_raw_lgb.py', 'src/round6_geo.py',
        'src/train_lightgbm.py', 'src/prepare_data.py')}


def parameters():
    """The fixed R7 numerical recipe, independently of its unit-price target."""
    result = candidate_params(63, ITERATIONS, 4)
    base.need(result['objective'] == 'regression_l1' and result['metric'] == 'None'
              and result['n_estimators'] == ITERATIONS and result['num_leaves'] == 63
              and result['n_jobs'] == 4 and result['random_state'] == 42,
              'Fixed raw LightGBM parameter recipe differs')
    return _copy_json(result)


def values_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes(order='C')).hexdigest()


def _check_X(X):
    base.need(isinstance(X, pd.DataFrame), 'X must be a DataFrame')
    base.need(X.columns.is_unique and X.index.is_unique and len(X.columns) == 87,
              'Expected 87 unique original columns and unique row indices')
    base.need(all(isinstance(c, str) for c in X.columns), 'Feature names must be strings')
    base.need(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
              'IDs and targets cannot enter model features')


def _float_vector(values, name):
    raw = values.to_numpy(copy=True) if isinstance(values, pd.Series) else np.asarray(values)
    base.need(raw.ndim == 1 and raw.dtype.kind in 'iuf', name + ' must be a real numeric vector')
    result = np.array(raw, dtype=np.float64, order='C', copy=True)
    base.need(np.isfinite(result).all(), name + ' must be finite')
    return result


def transform_target(y):
    values = _float_vector(y, 'Fit target')
    base.need((values >= 0).all(), 'Fit prices must be nonnegative')
    return values / 1e6


def inverse_transform(z):
    values = _float_vector(z, 'Model output')
    try:
        with np.errstate(over='raise', invalid='raise'):
            prices = values * 1e6
    except FloatingPointError as exc:
        raise ValueError('Raw TWD scaling overflowed') from exc
    base.need(np.isfinite(prices).all(), 'TWD predictions must be finite')
    return np.maximum(prices, 0.0).astype(np.float64, copy=False)


def _fit_encoding(X, geo, categorical_columns):
    _check_X(X)
    base.need(len(X) > 0, 'Cannot fit an empty frame')
    augmented = base.augment_features(X, geo)
    names = base.categories(X, categorical_columns)
    encoded, maps, safe_names, encoded_cats = encode_features({'train': augmented}, names)
    values = encoded['train']
    base.need(len(augmented.columns) == 95 and len(names) == 23, 'Expected 95 features / 23 categories')
    base.need(list(values.columns) == [f'f{i:03d}' for i in range(95)], 'Safe feature order differs')
    state = {
        'format_version': 1, 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'feature_rules': _copy_json(FEATURE_RULES), 'source_sha256': source_hashes(),
        'parameters': parameters(), 'fit_rows': int(len(X)),
        'original_columns': list(X.columns), 'original_dtypes': {c: str(X[c].dtype) for c in X},
        'all_feature_names': list(augmented.columns), 'feature_columns': list(augmented.columns),
        'model_feature_columns': list(values.columns),
        'original_categorical_columns': list(categorical_columns),
        'categorical_columns': list(names), 'categorical_features': list(encoded_cats),
        'safe_feature_names': _copy_json(safe_names), 'category_maps': _copy_json(maps),
        'fit_model_feature_dtypes': {c: str(values[c].dtype) for c in values},
        'fit_original_feature_values_sha256': base.frame_sha(X),
        'fit_geo_feature_values_sha256': base.frame_sha(geo),
        'fit_augmented_feature_values_sha256': base.frame_sha(augmented),
        'fit_encoded_feature_values_sha256': base.frame_sha(values),
        'fit_model_feature_values_sha256': base.frame_sha(values),
        'fit_position_rows': int(geo.geo_has_position.sum()),
        'fit_state_uses_y': False, 'fit_state_uses_query': False, 'query_labels_used': False,
    }
    state['state_content_sha256'] = base.content_sha(state)
    return values, state


def fit_feature_state(X, geo, categorical_columns):
    """Learn category maps from the supplied fit X/geo only; no y or query API."""
    return _fit_encoding(X, geo, categorical_columns)[1]


def check_state(state, X, categorical_columns):
    _check_X(X)
    base.need(state['state_content_sha256'] == base.content_sha(
        {k: v for k, v in state.items() if k != 'state_content_sha256'}), 'Feature state content changed')
    base.need(state['component'] == COMPONENT and state['target_mode'] == TARGET_MODE
              and state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Model target/component contract changed')
    base.need(state['feature_rules'] == FEATURE_RULES and state['source_sha256'] == source_hashes()
              and state['parameters'] == parameters(), 'Rules, sources or parameters changed')
    base.need(list(X.columns) == state['original_columns']
              and {c: str(X[c].dtype) for c in X} == state['original_dtypes'], 'Original schema changed')
    base.need(list(categorical_columns) == state['original_categorical_columns']
              and base.categories(X, categorical_columns) == state['categorical_columns'],
              'Category schema changed')
    base.need(set(state['category_maps']) == set(state['categorical_columns']), 'Category maps incomplete')
    expected = list(X.columns) + list(base.FEATURES)
    expected_safe = {c: f'f{i:03d}' for i, c in enumerate(expected)}
    base.need(state['feature_columns'] == state['all_feature_names'] == expected
              and state['model_feature_columns'] == list(expected_safe.values())
              and state['safe_feature_names'] == expected_safe
              and state['categorical_features'] == [expected_safe[c] for c in state['categorical_columns']],
              'Saved encoded feature names/order differ')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'Unexpected feature-state data use')


def encode_query(state, X, geo, categorical_columns):
    """Apply immutable fit maps; unknown categories become -1, never query codes."""
    check_state(state, X, categorical_columns)
    augmented = base.augment_features(X, geo)
    encoded, maps, safe_names, cats = encode_features(
        {'train': augmented}, state['categorical_columns'], category_maps=state['category_maps'])
    values = encoded['train']
    base.need(maps == state['category_maps'] and safe_names == state['safe_feature_names']
              and cats == state['categorical_features']
              and list(values.columns) == state['model_feature_columns']
              and {c: str(values[c].dtype) for c in values} == state['fit_model_feature_dtypes'],
              'Query encoder differs from saved fit schema')
    return values


def _check_model(model, state):
    base.need(model.current_iteration() == ITERATIONS and model.num_trees() == ITERATIONS,
              'Expected all fixed 3999 LightGBM trees/iterations')
    base.need(model.feature_name() == state['model_feature_columns'], 'Booster feature names/order differ')


def fit_model(X, y, geo, categorical_columns):
    """Fit uniform-weight raw prices; return native Booster, JSON state, metadata."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fit target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fit target length differs')
    target = transform_target(raw)
    raw_sha, target_sha = values_sha256(raw), values_sha256(target)
    encoded, state = _fit_encoding(X, geo, categorical_columns)
    config = parameters()
    estimator = lgb.LGBMRegressor(**config)
    train_started = time.monotonic()
    estimator.fit(encoded, target, categorical_feature=state['categorical_features'])
    training_seconds = time.monotonic() - train_started
    model = estimator.booster_
    _check_model(model, state)
    check_state(state, X, categorical_columns)
    base.need(base.frame_sha(X) == state['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['fit_geo_feature_values_sha256']
              and base.frame_sha(encoded) == state['fit_encoded_feature_values_sha256'],
              'Fitting changed feature values')
    base.need(values_sha256(raw) == raw_sha and values_sha256(target) == target_sha,
              'Fitting changed target values')
    metadata = {
        'family': 'lightgbm', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'parameters': config, 'parameters_sha256': base.content_sha(config),
        'iterations': ITERATIONS, 'actual_iterations': int(model.current_iteration()),
        'actual_trees': int(model.num_trees()), 'features': 95, 'feature_count': 95,
        'categorical_feature_count': 23, 'feature_columns': list(state['feature_columns']),
        'model_feature_columns': list(state['model_feature_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'categorical_features': list(state['categorical_features']),
        'sample_weight': 'uniform', 'sample_weight_argument_passed': False,
        'target_factor': '1e6', 'prediction_units': 'TWD',
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(FEATURE_RULES),
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round14_geo_raw_lgb.py'],
        'source_sha256': _copy_json(state['source_sha256']), 'fit_rows': int(len(X)),
        'fit_target_values_sha256': raw_sha, 'fit_original_target_values_sha256': raw_sha,
        'fit_transformed_target_values_sha256': target_sha,
        'target_value_hash_encoding': 'ordered C-contiguous little-endian float64',
        'fit_original_feature_values_sha256': state['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['fit_geo_feature_values_sha256'],
        'fit_augmented_feature_values_sha256': state['fit_augmented_feature_values_sha256'],
        'fit_model_feature_values_sha256': state['fit_model_feature_values_sha256'],
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'query_labels_used': False, 'fit_uses_eval_set': False, 'early_stopping_used': False,
        'calibration_applied': False, 'extra_price_cap_applied': False,
        'reference_or_trend_applied_here': False, 'reference_features_changed': False,
        'category_maps_fit_only': True, 'external_geography_used': True,
        'external_property_answers_used': False, 'training_seconds': training_seconds,
        'elapsed_seconds': time.monotonic() - started,
        'runtime_versions': {'lightgbm': lgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
    }
    return model, state, metadata


def predict_model(model, state, X, geo, categorical_columns):
    """Return unrounded float64 total TWD; the caller alone applies any trend."""
    encoded = encode_query(state, X, geo, categorical_columns)
    _check_model(model, state)
    raw = _float_vector(model.predict(encoded, num_iteration=ITERATIONS, num_threads=4), 'Model output')
    base.need(raw.shape == (len(X),), 'Prediction row count differs')
    return inverse_transform(raw)


def self_check():
    """In-memory fixtures and fake estimators only; no real data/model IO."""
    checks = []
    def ok(name, value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def rejects(name, function):
        try: function()
        except (ValueError, KeyError, TypeError): checks.append(name)
        else: raise AssertionError(name + ' did not reject')
    index = pd.Index([41, 8, 93, 2, 17, 65], name='source_row')
    cats = ['cat_' + str(i) for i in range(21)]
    X = pd.DataFrame({c: pd.Series(['B', 'A', 'B', 'A', 'B', 'A'], index=index, dtype=object) for c in cats}, index=index)
    for c in ['building_area', 'parking_area'] + ['num_' + str(i) for i in range(64)]:
        X[c] = np.arange(len(index), dtype='float32')
    geo = pd.DataFrame({c: np.full(len(index), np.nan, dtype='float32') for c in base.NUMERIC}, index=index)
    geo['geo_has_position'] = np.zeros(len(index), dtype='float32')
    geo['geo_match_status'] = pd.Series(['ineligible_address'] * len(index), index=index, dtype=object)
    geo['geo_grid_002'] = pd.Series(['__MISSING__'] * len(index), index=index, dtype=object)
    geo = geo[base.FEATURES]
    before_X, before_geo = X.copy(deep=True), geo.copy(deep=True)
    encoded, state = _fit_encoding(X, geo, cats)
    saved = _copy_json(state)
    ok('95_columns_23_categorical', encoded.shape == (6, 95) and len(state['categorical_features']) == 23)
    ok('safe_names_exact_order', list(encoded.columns) == [f'f{i:03d}' for i in range(95)])
    ok('categorical_int32_numeric_float32', all(str(encoded[c].dtype) == ('int32' if c in state['categorical_features'] else 'float32') for c in encoded))
    ok('first_fit_occurrence_category_codes', state['category_maps']['cat_0'] == {'B': 0, 'A': 1})
    ok('JSON_state_exact_encoding_replay', encoded.equals(encode_query(saved, X, geo, cats)))
    query = X.copy(); query.loc[index[0], 'cat_0'] = 'UNSEEN'
    query_geo = geo.copy(); query_geo.loc[index[0], 'geo_grid_002'] = 'UNSEEN_GRID'
    # Use a usable geometry row for the novel grid while preserving R6 validity.
    query_geo.loc[index[0], base.NUMERIC] = np.asarray([25, 121, 1, 1, 1, 0], dtype='float32')
    query_geo.loc[index[0], 'geo_match_status'] = 'road_level'
    state_before = base.content_sha(saved)
    q = encode_query(saved, query, query_geo, cats)
    ok('unseen_category_and_grid_minus_one', q.loc[index[0], state['safe_feature_names']['cat_0']] == -1 and q.loc[index[0], state['safe_feature_names']['geo_grid_002']] == -1)
    ok('query_preserves_fit_maps', base.content_sha(saved) == state_before)
    ok('fit_state_has_no_targets_or_query', not state['fit_state_uses_y'] and not state['fit_state_uses_query'] and not state['query_labels_used'])
    altered = _copy_json(state); altered['feature_rules']['additional_numeric'].append('bad')
    ok('rules_deep_copied', 'bad' not in FEATURE_RULES['additional_numeric'] and 'bad' not in base.FEATURE_RULES['additional_numeric'])
    rejects('mutated_state_rejected', lambda: check_state(altered, X, cats))
    rejects('geo_index_misalignment_rejected', lambda: encode_query(saved, X, geo.iloc[::-1], cats))
    rejects('X_column_order_change_rejected', lambda: encode_query(saved, X.iloc[:, ::-1], geo, cats))
    changed_dtype = X.copy(); changed_dtype['num_0'] = changed_dtype['num_0'].astype('float64')
    rejects('X_dtype_change_rejected', lambda: encode_query(saved, changed_dtype, geo, cats))
    duplicate = X.copy(); duplicate.index = [1] * len(X)
    rejects('duplicate_row_index_rejected', lambda: fit_feature_state(duplicate, geo, cats))
    target_column = X.rename(columns={'num_0': '總價元'})
    rejects('target_feature_rejected', lambda: fit_feature_state(target_column, geo, cats))
    ok('target_and_inverse_raw_scale', np.array_equal(transform_target([0., 1e6, 3e6]), [0., 1., 3.]) and np.array_equal(inverse_transform([-1., 0., 3.]), [0., 0., 3e6]))
    for label, values in [('negative', [-1.]), ('nan', [np.nan]), ('inf', [np.inf]), ('complex', [1j]), ('matrix', [[1.]])]:
        rejects('invalid_target_' + label, lambda values=values: transform_target(values))
    for label, values in [('nan', [np.nan]), ('inf', [np.inf]), ('matrix', [[1.]]), ('overflow', [np.finfo('float64').max])]:
        rejects('invalid_inverse_' + label, lambda values=values: inverse_transform(values))
    ok('fixed_3999_63_full_parameters', parameters() == candidate_params(63, 3999, 4))
    ok('feature_and_query_APIs_without_y', list(inspect.signature(fit_feature_state).parameters) == ['X', 'geo', 'categorical_columns'] and list(inspect.signature(predict_model).parameters) == ['model', 'state', 'X', 'geo', 'categorical_columns'])
    calls = []
    class FakeBooster:
        def __init__(self, names, iterations=ITERATIONS): self.names, self.iterations = list(names), iterations
        def current_iteration(self): return self.iterations
        def num_trees(self): return self.iterations
        def feature_name(self): return list(self.names)
        def predict(self, frame, **kwargs):
            if kwargs != {'num_iteration': ITERATIONS, 'num_threads': 4}: raise AssertionError('Prediction configuration differs')
            return frame['f023'].to_numpy(dtype='float64') + .25
    class FakeLGBM:
        def __init__(self, **kwargs): self.config = kwargs
        def fit(self, frame, target, **kwargs):
            if set(kwargs) != {'categorical_feature'}: raise AssertionError('Unexpected weights/eval/early-stop argument')
            calls.append({'frame': frame.copy(deep=True), 'target': target.copy(), 'kwargs': _copy_json(kwargs), 'config': self.config})
            self.booster_ = FakeBooster(frame.columns)
            return self
    original = lgb.LGBMRegressor
    lgb.LGBMRegressor = FakeLGBM
    try:
        y = pd.Series(np.arange(1, 7, dtype='float64') * 1e6, index=index)
        before_y = y.copy()
        model, model_state, metadata = fit_model(X, y, geo, cats)
        ok('returns_native_Booster_interface', isinstance(model, FakeBooster))
        ok('actual_uniform_fit_without_sample_weight_or_eval', set(calls[-1]['kwargs']) == {'categorical_feature'})
        ok('actual_fit_y_divided_once_by_million', np.array_equal(calls[-1]['target'], y.to_numpy() / 1e6))
        ok('actual_categorical_safe_names_passed', calls[-1]['kwargs']['categorical_feature'] == model_state['categorical_features'])
        ok('actual_full_parameters_unchanged', calls[-1]['config'] == candidate_params(63, 3999, 4))
        prediction = predict_model(model, _copy_json(model_state), X, geo, cats)
        ok('prediction_float64_TWD_once', prediction.dtype == np.dtype('float64') and np.array_equal(prediction, (np.arange(6) + .25) * 1e6))
        pieces = [predict_model(model, model_state, X.iloc[a:b], geo.iloc[a:b], cats) for a, b in [(0, 2), (2, 6)]]
        ok('nonrange_index_and_batch_consistency', np.array_equal(np.concatenate(pieces), prediction))
        order = [5, 1, 3]
        ok('reordered_query_preserved', np.array_equal(predict_model(model, model_state, X.iloc[order], geo.iloc[order], cats), prediction[order]))
        area_changed = X.copy(); area_changed['building_area'] = np.full(6, 1e5, dtype='float32'); area_changed['parking_area'] = np.full(6, np.nan, dtype='float32')
        ok('no_postmodel_area_multiplier', np.array_equal(predict_model(model, model_state, area_changed, geo, cats), prediction))
        rejects('wrong_Booster_iteration_rejected', lambda: predict_model(FakeBooster(model.names, 12), model_state, X, geo, cats))
        rejects('wrong_Booster_feature_order_rejected', lambda: predict_model(FakeBooster(model.names[::-1]), model_state, X, geo, cats))
        rejects('Series_y_index_misalignment_rejected', lambda: fit_model(X, y.iloc[::-1], geo, cats))
        rejects('target_row_count_mismatch_rejected', lambda: fit_model(X, [1.], geo, cats))
        ok('raw_target_hashes_and_transformed_hash', metadata['fit_target_values_sha256'] == metadata['fit_original_target_values_sha256'] == values_sha256(y.to_numpy()) and metadata['fit_transformed_target_values_sha256'] == values_sha256(y.to_numpy() / 1e6))
        ok('metadata_raw_uniform95_23_3999', metadata['geo_feature_rules'] == FEATURE_RULES and metadata['target_mode'] == 'raw' and metadata['sample_weight'] == 'uniform' and metadata['sample_weight_argument_passed'] is False and metadata['feature_count'] == 95 and metadata['categorical_feature_count'] == 23 and metadata['iterations'] == metadata['actual_iterations'] == metadata['actual_trees'] == 3999)
        ok('metadata_no_query_y_eval_calibration_or_trend', all(metadata[k] is False for k in ['query_labels_used', 'fit_uses_eval_set', 'early_stopping_used', 'calibration_applied', 'extra_price_cap_applied', 'reference_or_trend_applied_here', 'reference_features_changed']))
        metadata['feature_rules']['additional_numeric'].append('metadata_only')
        ok('metadata_rules_independent_of_state_and_constant', 'metadata_only' not in model_state['feature_rules']['additional_numeric'] and 'metadata_only' not in FEATURE_RULES['additional_numeric'])
        pd.testing.assert_frame_equal(X, before_X); pd.testing.assert_frame_equal(geo, before_geo); pd.testing.assert_series_equal(y, before_y)
        checks.append('input_frames_and_target_unchanged')
    finally:
        lgb.LGBMRegressor = original
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks),
            'real_data_read': False, 'real_model_training_performed': False,
            'file_outputs_written': False, 'source_sha256': source_hashes()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check-only', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False, indent=2))
