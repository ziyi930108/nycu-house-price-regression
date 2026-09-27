"""R12 prototype: original geographic CatBoost trained on log1p(TWD / 1e6).

The only target accepted is the supplied fit target. Geographic state is fitted
without targets or query data. This module does not apply the reference/trend,
calibrate predictions, save a model, or select a candidate. The caller must
verify the row/index, geographic-manifest, split, and plan provenance on disk.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import round6_geo as base
from prepare_data import ROOT, sha256

COMPONENT = 'geography_log1p_raw_catboost'
TARGET_MODE = 'log1p_raw'
TARGET_FORMULA = 'log1p(y_TWD / 1e6)'
INVERSE_FORMULA = 'max(expm1(z) * 1e6, 0)'
LEGACY_TRAINING_FIELDS = {
    'fit_target', 'sample_weight', 'iterations', 'seed',
    'no_eval_set', 'no_early_stopping',
}
FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'feature_count': 95,
    'categorical_feature_count': 23,
    'geo_encoder_rules': {
        k: v for k, v in base.FEATURE_RULES.items()
        if k not in LEGACY_TRAINING_FIELDS
    },
    'legacy_geo_state_training_fields_are_not_this_models_training_declaration': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'target_dtype': 'float64',
    'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': 'uniform',
    'weight_argument_passed_to_pool': False,
    'query_labels_used': False,
    'query_used_for_geographic_state': False,
    'target_used_for_geographic_state': False,
    'iterations': 2999,
    'seed': 42,
    'no_eval_set': True,
    'no_early_stopping': True,
    'calibration_applied': False,
    'extra_price_cap_applied': False,
    'nonfinite_or_inverse_overflow': 'raise ValueError',
    'finite_negative_model_prediction': 'project price to zero after finite inverse',
    'reference_or_trend_applied_here': False,
}


def parameters():
    """Exactly the fixed R6 parameters, including all 2999 iterations."""
    return base.parameters()


def values_sha256(values):
    """Ordered float64 value SHA; row identity is checked separately."""
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes(order='C')).hexdigest()


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _sources():
    return {
        'src/round12_log_geo.py': sha256(Path(__file__)),
        'src/round6_geo.py': sha256(Path(base.__file__)),
        'src/prepare_data.py': sha256(ROOT / 'src/prepare_data.py'),
    }


def _check_X(X):
    base.need(isinstance(X, pd.DataFrame), 'X must be a DataFrame')
    base.need(X.columns.is_unique and X.index.is_unique,
              'Feature columns and row indices must be unique')
    base.need(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
              'Identifiers/targets must not enter model features')


def _float_vector(values, name):
    raw = values.to_numpy(copy=True) if isinstance(values, pd.Series) else np.asarray(values)
    base.need(raw.ndim == 1, name + ' must be a one-dimensional vector')
    base.need(raw.dtype.kind in 'iuf', name + ' must contain real numeric values')
    result = np.array(raw, dtype=np.float64, order='C', copy=True)
    base.need(np.isfinite(result).all(), name + ' must be finite')
    return result


def transform_target(y):
    """Fixed forward transform; no data-dependent scale or clipping."""
    values = _float_vector(y, 'Fit target')
    base.need((values >= 0).all(), 'Fit target must be nonnegative')
    result = np.log1p(values / 1e6)
    base.need(np.isfinite(result).all(), 'Transformed fit target must be finite')
    return np.asarray(result, dtype=np.float64)


def inverse_transform(z):
    """Exact nonnegative inverse, rejecting overflow instead of capping it."""
    values = _float_vector(z, 'Model prediction')
    try:
        with np.errstate(over='raise', invalid='raise'):
            unscaled = np.expm1(values)
            base.need(np.isfinite(unscaled).all(), 'Inverse expm1 is nonfinite')
            prices = unscaled * 1e6
            base.need(np.isfinite(prices).all(), 'Inverse TWD scaling is nonfinite')
    except FloatingPointError as exc:
        raise ValueError('Inverse target transformation overflowed') from exc
    return np.maximum(prices, 0.0).astype(np.float64, copy=False)


def fit_feature_state(X, geo, cats):
    """Build a label-free wrapper around the unchanged R6 geographic state."""
    _check_X(X)
    base.need(len(X) > 0, 'Cannot fit an empty feature frame')
    geo_state = _copy_json(base.fit_feature_state(X, geo))
    categories = base.categories(X, cats)
    base.need(len(categories) == 23 and len(set(categories)) == 23,
              'Expected 23 distinct categorical features')
    state = {
        'format_version': 1,
        'component': COMPONENT,
        'target_mode': TARGET_MODE,
        'feature_rules': _copy_json(FEATURE_RULES),
        'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA,
        'source_sha256': _sources(),
        'base_configuration_sha256': sha256(ROOT / 'runs/v1/models/final_config.json'),
        'parameters': parameters(),
        'geo_state': geo_state,
        'original_categorical_columns': list(cats),
        'categorical_columns': categories,
        'all_feature_names': list(geo_state['all_feature_names']),
        'fit_rows': int(len(X)),
        'fit_state_uses_y': False,
        'fit_state_uses_query': False,
        'query_labels_used': False,
    }
    state['state_content_sha256'] = base.content_sha(state)
    return state


def check_state(state, X, cats):
    _check_X(X)
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'
    }), 'Log model state content changed')
    base.need(state['component'] == COMPONENT and state['target_mode'] == TARGET_MODE
              and state['feature_rules'] == FEATURE_RULES, 'Log model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Target transformation changed')
    base.need(state['source_sha256'] == _sources(), 'Helper/dependency source changed')
    base.need(state['base_configuration_sha256'] == sha256(ROOT / 'runs/v1/models/final_config.json'),
              'Base model configuration changed')
    base.need(state['parameters'] == parameters(), 'Original R6 parameters changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns'], 'Original category order changed')
    base.need(base.categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names']
              and len(state['all_feature_names']) == 95, 'Wrapped geography schema changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declarations changed')


def fit_model(X, y, geo, cats):
    """Fit once with uniform log-space MAE; accepts only fit labels."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fitting target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fitting target length differs from X')
    transformed = transform_target(raw)
    raw_sha = values_sha256(raw)
    transformed_sha = values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    augmented = base.augment_features(X, geo)
    categories = base.categories(X, cats)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, cat_features=categories, thread_count=4))
    base.need(int(model.tree_count_) == 2999, 'Expected all fixed 2999 trees')
    check_state(state, X, cats)
    base.need(parameters() == config, 'Base parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256'],
              'Fit feature inputs changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Target values passed to Pool changed')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Model feature names differ')
    metadata = {
        'family': 'catboost',
        'component': COMPONENT,
        'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA,
        'parameters': config,
        'parameters_sha256': base.content_sha(config),
        'iterations': 2999,
        'features': 95,
        'feature_count': 95,
        'categorical_feature_count': 23,
        'feature_columns': list(augmented.columns),
        'categorical_features': categories,
        'categorical_columns': categories,
        'sample_weight': 'uniform',
        'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES),
        'geo_feature_rules': _copy_json(FEATURE_RULES['geo_encoder_rules']),
        'legacy_geo_state_training_fields_are_not_this_models_training_declaration': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round12_log_geo.py'],
        'source_sha256': _copy_json(state['source_sha256']),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_rows': int(len(X)),
        'fit_original_feature_values_sha256': state['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['geo_state']['fit_geo_feature_values_sha256'],
        'fit_target_values_sha256': raw_sha,
        'fit_original_target_values_sha256': raw_sha,
        'fit_transformed_target_values_sha256': transformed_sha,
        'target_value_hash_encoding': 'ordered C-contiguous little-endian float64',
        'target_used_for_geographic_state': False,
        'query_used_for_geographic_state': False,
        'query_labels_used': False,
        'fit_uses_eval_set': False,
        'early_stopping_used': False,
        'calibration_applied': False,
        'extra_price_cap_applied': False,
        'reference_or_trend_applied_here': False,
        'reference_features_changed': False,
        'external_geography_used': True,
        'external_property_answers_used': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    return model, state, metadata


def predict_model(model, state, X, geo, cats):
    """Return unrounded float64 TWD. This API has no query-label argument."""
    check_state(state, X, cats)
    augmented = base.augment_features(X, geo)
    categories = base.categories(X, cats)
    base.need(list(model.feature_names_) == list(augmented.columns) == state['all_feature_names'],
              'Saved model feature names differ')
    raw = model.predict(Pool(augmented, cat_features=categories, thread_count=4), thread_count=4)
    result = inverse_transform(raw)
    base.need(result.shape == (len(X),), 'Prediction shape differs from query rows')
    return result


def self_check():
    """Synthetic only: replace CatBoost and Pool, never fit a real estimator."""
    from house_features import build_features, get_categorical_columns

    checks = []

    def ok(name, condition):
        base.need(bool(condition), 'Synthetic check failed: ' + name)
        checks.append(name)

    def rejected(name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    artificial = np.array([0.0, 0.001, 1.0, 1e6, 1e9, 1e15], dtype=np.float64)
    ok('forward_inverse_zero_small_large', np.allclose(
        inverse_transform(transform_target(artificial)), artificial, rtol=1e-13, atol=1e-12))
    ok('target_transform_is_fixed_log1p', np.array_equal(
        transform_target(artificial), np.log1p(artificial / 1e6)))
    rejected('negative_target_rejected', lambda: transform_target(np.array([-1.])))
    rejected('nan_target_rejected', lambda: transform_target(np.array([np.nan])))
    rejected('infinite_target_rejected', lambda: transform_target(np.array([np.inf])))
    rejected('matrix_target_rejected', lambda: transform_target(np.array([[1.]])))
    rejected('complex_target_rejected', lambda: transform_target(np.array([1+1j])))
    rejected('text_target_rejected', lambda: transform_target(np.array(['1'])))
    rejected('inverse_nan_rejected', lambda: inverse_transform(np.array([np.nan])))
    rejected('inverse_infinity_rejected', lambda: inverse_transform(np.array([np.inf])))
    rejected('inverse_exponential_overflow_rejected', lambda: inverse_transform(np.array([1000.])))
    rejected('inverse_scaling_overflow_rejected', lambda: inverse_transform(np.array([700.])))
    ok('finite_negative_prediction_projects_zero', np.array_equal(
        inverse_transform(np.array([-1e6, -0.1, 0.])), np.zeros(3)))

    raw = pd.DataFrame({
        '鄉鎮市區': ['中山區', '新店區', '前鎮區', '新市區'],
        '土地位置建物門牌': ['臺北市中山區中山路1號', '新北市新店區及人路2號',
                     '高雄市前鎮區中華路3號', '臺南市新市區和平路4號'],
        '交易年月日': [1110101, 1110201, 1110301, 1110401],
        '建物移轉總面積平方公尺': [80, 100, 120, 140],
        '建案名稱': ['甲案', '乙案', '丙案', None],
    }, index=pd.Index([7, 11, 19, 31], name='source_position'))
    X = build_features(raw)
    cats = get_categorical_columns(X)
    matched = pd.DataFrame({
        'geo_latitude': [25., np.nan, 22.6, np.nan],
        'geo_longitude': [121.5, np.nan, 120.3, np.nan],
        'geo_road_length_m': [1000., np.nan, 700., np.nan],
        'geo_extent_diagonal_m': [900., np.nan, 0., np.nan],
        'geo_nearby_fragment_grouping': [0., np.nan, 1., np.nan],
        'geo_match_status': ['road_level', 'ineligible_address', 'planar_intersection', 'not_found'],
    }, index=X.index)
    geo = base.encode_geo_features(matched)
    X_before, geo_before = X.copy(deep=True), geo.copy(deep=True)
    y = pd.Series([0., 1e3, 1e6, 1e12], index=X.index)
    y_before = y.copy(deep=True)
    pools, models = [], []

    class FakePool:
        def __init__(self, data, label=None, cat_features=None, thread_count=None):
            self.data = data.copy(deep=True)
            self.label = None if label is None else np.array(label, copy=True)
            self.cat_features = list(cat_features)
            self.thread_count = thread_count
            pools.append(self)

    class FakeCat:
        def __init__(self, **config):
            self.config = config
            self.tree_count_ = config['iterations']
            self.output = None
            models.append(self)

        def fit(self, pool):
            self.fit_pool = pool
            self.feature_names_ = list(pool.data.columns)
            return self

        def predict(self, pool, thread_count=None):
            return np.zeros(len(pool.data)) if self.output is None else self.output.copy()

    original_factories = (globals()['CatBoostRegressor'], globals()['Pool'])
    try:
        globals()['CatBoostRegressor'], globals()['Pool'] = FakeCat, FakePool
        model, state, metadata = fit_model(X, y, geo, cats)
        fit_pool = model.fit_pool
        ok('original_87_21_and_augmented_95_23',
           X.shape[1] == 87 and len(cats) == 21 and fit_pool.data.shape[1] == 95
           and len(fit_pool.cat_features) == 23)
        ok('all_original_features_dtypes_indices_preserved',
           fit_pool.data[X.columns].equals(X) and fit_pool.data[X.columns].dtypes.equals(X.dtypes)
           and fit_pool.data.index.equals(X.index))
        ok('all_eight_geo_features_unchanged', fit_pool.data[base.FEATURES].equals(geo))
        ok('log_labels_reach_pool', np.array_equal(fit_pool.label, np.log1p(y.to_numpy() / 1e6)))
        ok('exact_R6_parameters_2999_seed42', model.config == base.parameters()
           and model.config['iterations'] == 2999 and model.config['random_seed'] == 42)
        ok('uniform_no_weight_eval_or_earlystop_arguments',
           list(inspect.signature(FakeCat.fit).parameters) == ['self', 'pool']
           and 'weight' not in inspect.signature(FakePool).parameters
           and metadata['sample_weight'] == 'uniform' and metadata['fit_uses_eval_set'] is False
           and metadata['early_stopping_used'] is False)
        ok('raw_and_transformed_label_hashes', metadata['fit_target_values_sha256'] == values_sha256(y)
           and metadata['fit_transformed_target_values_sha256'] == values_sha256(fit_pool.label)
           and metadata['fit_target_values_sha256'] != metadata['fit_transformed_target_values_sha256'])
        ok('current_log_contract_separate_from_legacy_geo_state',
           state['feature_rules']['fit_target'] == TARGET_FORMULA
           and state['geo_state']['feature_rules']['fit_target'] == 'raw TWD / 1e6'
           and 'fit_target' not in metadata['geo_feature_rules']
           and metadata['legacy_geo_state_training_fields_are_not_this_models_training_declaration'])
        _, alternative_state, _ = fit_model(X, y + 100., geo, cats)
        ok('geographic_state_independent_of_fit_targets', state == alternative_state)
        ok('input_frames_and_labels_not_mutated', X.equals(X_before) and X.dtypes.equals(X_before.dtypes)
           and geo.equals(geo_before) and geo.dtypes.equals(geo_before.dtypes) and y.equals(y_before))
        rejected('target_series_index_mismatch_rejected',
                 lambda: fit_model(X, y.iloc[::-1], geo, cats))
        rejected('target_length_mismatch_rejected',
                 lambda: fit_model(X, y.to_numpy()[:2], geo, cats))
        rejected('target_matrix_fit_rejected',
                 lambda: fit_model(X, y.to_numpy()[:, None], geo, cats))
        rejected('empty_fit_rejected', lambda: fit_model(X.iloc[:0], y.iloc[:0], geo.iloc[:0], cats))
        duplicate = X.copy()
        duplicate.index = pd.Index([7, 7, 19, 31])
        rejected('duplicate_feature_indices_rejected',
                 lambda: fit_model(duplicate, y.to_numpy(), geo, cats))
        rejected('geo_row_order_mismatch_rejected',
                 lambda: fit_model(X, y, geo.iloc[::-1], cats))
        rejected('category_schema_mismatch_rejected',
                 lambda: fit_model(X, y, geo, cats[:-1]))
        restored = _copy_json(state)
        ok('state_JSON_roundtrip', restored == state)
        state_before = _copy_json(restored)
        model.output = np.array([-0.3, 0., np.log1p(2.), np.log1p(100.)])
        prediction = predict_model(model, restored, X, geo, cats)
        ok('inverse_TWD_float64_without_rounding', prediction.dtype == np.float64
           and np.allclose(prediction, [0., 0., 2e6, 100e6], rtol=1e-14, atol=1e-9))
        ok('prediction_pool_has_no_label', pools[-1].label is None)
        ok('prediction_API_has_no_query_target',
           list(inspect.signature(predict_model).parameters) == ['model', 'state', 'X', 'geo', 'cats'])
        ok('query_preserves_state', restored == state_before)
        model.output = np.array([np.log1p(0.125), 0.])
        subset_prediction = predict_model(model, restored, X.iloc[[2, 0]], geo.iloc[[2, 0]], cats)
        ok('JSON_state_supports_nonrange_query_subset_order',
           np.allclose(subset_prediction, [125000., 0.], rtol=1e-14)
           and pools[-1].data.index.equals(X.iloc[[2, 0]].index))
        for name, bad in [('nonfinite', np.array([np.nan]*4)),
                          ('exponential_overflow', np.array([1000.]*4)),
                          ('scaling_overflow', np.array([700.]*4)),
                          ('shape', np.array([0., 0.]))]:
            model.output = bad
            rejected('predict_' + name + '_rejected',
                     lambda: predict_model(model, restored, X, geo, cats))
        changed = _copy_json(state)
        changed['target_formula'] = 'raw TWD / 1e6'
        rejected('state_tampering_rejected', lambda: check_state(changed, X, cats))
        rejected('prediction_category_order_change_rejected',
                 lambda: check_state(state, X, cats[::-1]))
        rejected('prediction_original_feature_order_change_rejected',
                 lambda: check_state(state, X[X.columns[::-1]], cats))
        ok('no_reference_calibration_or_cap', not any(metadata[k] for k in [
            'reference_or_trend_applied_here', 'calibration_applied', 'extra_price_cap_applied']))
        ok('source_and_parameter_hashes_bound', state['source_sha256'] == _sources()
           and metadata['parameters_sha256'] == base.content_sha(parameters()))
    finally:
        globals()['CatBoostRegressor'], globals()['Pool'] = original_factories
    return checks


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    verified = self_check()
    print(json.dumps({
        'status': 'passed', 'synthetic_checks': len(verified), 'checks': verified,
        'script_sha256': sha256(Path(__file__)), 'real_model_fitted': False,
        'real_target_or_prediction_data_read': False,
    }, ensure_ascii=False, indent=2))
