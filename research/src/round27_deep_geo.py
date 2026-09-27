"""R27 fixed depth-10 raw geographic CatBoost: only passed depth changes 8 to 10.
Original 95/23 features, seed42, 2999 rounds, .065 and uniform raw MAE remain.
Native max_leaves is recorded independently; coupled defaults are not claimed unchanged.
No real-data, model or configuration reads occur in the FakePool self-check.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round6_geo as base
from prepare_data import ROOT, sha256

COMPONENT = 'geography_deep_raw_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ADDED_FEATURES = []
ADDED_CATEGORICAL_FEATURES = []


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'existing_geo_features_unchanged': 8,
    'existing_95_values_dtypes_order_and_index_unchanged': True,
    'feature_count': 95,
    'categorical_feature_count': 23,
    'geo_encoder_rules': _copy_json(base.FEATURE_RULES),
    'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'additional_categorical_features': list(ADDED_CATEGORICAL_FEATURES),
    'original_coarse_grid_policy': 'preserve existing geo_grid_002 unchanged',
    'fit_vocabulary_or_support_filter': False,
    'target_dtype': 'float64',
    'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': 'uniform',
    'weight_argument_passed_to_pool': False,
    'baseline_depth': 8,
    'depth': 10,
    'baseline_boosting_type': 'Plain',
    'boosting_type': 'Plain',
    'changed_parameter_names': ['depth'],
    'removed_feature_names': [],
    'added_feature_names': list(ADDED_FEATURES),
    'baseline_feature_count': 95,
    'original95_preserved': True,
    'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 1024,
    'baseline_iterations': 2999,
    'baseline_learning_rate': 0.065,
    'iterations': 2999,
    'learning_rate': 0.065,
    'native_learning_rate_comparison_rtol': 1e-7,
    'native_learning_rate_comparison_atol': 0.0,
    'threads': 4,
    'seed': 42,
    'no_eval_set': True,
    'no_early_stopping': True,
    'query_labels_used': False,
    'query_used_for_geographic_state': False,
    'target_used_for_geographic_state': False,
    'reference_scoring_grouping_or_dedup_keys_changed': False,
    'reference_or_trend_applied_here': False,
    'calibration_applied': False,
    'extra_price_cap_applied': False,
    'nonfinite_or_inverse_overflow': 'raise ValueError',
}


def source_hashes():
    return {name: sha256(ROOT / name) for name in (
        'src/round27_deep_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['learning_rate'] == 0.065 and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == 'MAE' and result['boosting_type'] == 'Plain',
              'Expected the fixed R6 depth-8 Plain recipe')
    return result


def parameters():
    """Deep-copy the baseline; depth is the only permitted changed parameter."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    result['depth'] = 10
    base.need(set(result) == set(baseline)
              and sorted(k for k in result if result[k] != baseline[k]) == ['depth'],
              'Only depth may change from the R6 baseline')
    return result


def _check_X(X):
    base.need(isinstance(X, pd.DataFrame), 'X must be a DataFrame')
    base.need(X.columns.is_unique and X.index.is_unique, 'X columns and indices must be unique')
    base.need(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
              'Identifiers and targets cannot enter model features')


def _float_vector(values, name):
    raw = values.to_numpy(copy=True) if isinstance(values, pd.Series) else np.asarray(values)
    base.need(raw.ndim == 1 and raw.dtype.kind in 'iuf', name + ' must be real numeric and one dimensional')
    result = np.array(raw, dtype=np.float64, order='C', copy=True)
    base.need(np.isfinite(result).all(), name + ' must be finite')
    return result


def values_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes(order='C')).hexdigest()


def transform_target(y):
    result = _float_vector(y, 'Fit target')
    base.need((result >= 0).all(), 'Fit prices must be nonnegative')
    return result / 1e6


def inverse_transform(z):
    result = _float_vector(z, 'Model output')
    try:
        with np.errstate(over='raise', invalid='raise'):
            prices = result * 1e6
    except FloatingPointError as exc:
        raise ValueError('Raw TWD scaling overflowed') from exc
    base.need(np.isfinite(prices).all(), 'Raw TWD predictions must be finite')
    return np.maximum(prices, 0).astype(np.float64, copy=False)


def augment_features(X, geo):
    """Use the original R6 95-column view, without adding or changing features."""
    _check_X(X)
    result = base.augment_features(X, geo)
    base.need(len(result.columns) == 95 and result.columns.is_unique
              and result.index.equals(X.index), 'Expected unchanged R6 95-feature schema')
    return result


def categories(X, supplied):
    result = base.categories(X, supplied)
    base.need(len(result) == len(set(result)) == 23, 'Expected 23 unique categorical columns')
    return result


def fit_feature_state(X, geo, cats):
    """Accept fit X and geography only; neither target nor query is an input."""
    _check_X(X)
    base.need(len(X) > 0, 'Cannot fit an empty feature frame')
    geo_state = _copy_json(base.fit_feature_state(X, geo))
    augmented = augment_features(X, geo)
    state = {
        'format_version': 1,
        'component': COMPONENT,
        'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA,
        'feature_rules': _copy_json(FEATURE_RULES),
        'source_sha256': source_hashes(),
        'base_configuration_sha256': _base_configuration_sha256(),
        'baseline_parameters': _baseline_parameters(),
        'parameters': parameters(),
        'changed_parameter_names': ['depth'],
        'removed_feature_names': [],
        'added_feature_names': list(ADDED_FEATURES),
        'baseline_feature_count': 95,
        'original95_preserved': True,
        'geo_state': geo_state,
        'original_categorical_columns': list(cats),
        'categorical_columns': categories(X, cats),
        'all_feature_names': list(augmented.columns),
        'fit_rows': int(len(X)),
        'fit_model_feature_values_sha256': base.frame_sha(augmented),
        'fit_model_feature_dtypes': {c: str(augmented[c].dtype) for c in augmented},
        'fit_state_uses_y': False,
        'fit_state_uses_query': False,
        'query_labels_used': False,
    }
    state['state_content_sha256'] = base.content_sha(state)
    return state


def check_state(state, X, cats):
    _check_X(X)
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['target_mode'] == TARGET_MODE
              and state['feature_rules'] == FEATURE_RULES, 'Depth-only raw model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == ['depth']
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True, 'Depth-only R6 configuration or original feature schema changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['geo_state']['all_feature_names']) == 95
              and len(state['all_feature_names']) == 95, 'Original R6 or added model columns changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_model(model, state):
    """Check native trees, learning rate, policies and categorical positions."""
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 10
              and state['parameters']['boosting_type'] == 'Plain'
              and state['parameters']['iterations'] == 2999 and state['parameters']['learning_rate'] == 0.065
              and state['changed_parameter_names'] == ['depth']
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True
              and state['parameters'] == dict(state['baseline_parameters'], depth=10)
              and state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['all_feature_names']) == 95, 'Wrong depth-only raw model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    category_indices = np.asarray(model.get_cat_feature_indices())
    expected_categories = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(category_indices.shape == (23,) and category_indices.dtype.kind in 'iu'
              and len(expected_categories) == len(set(expected_categories)) == 23
              and np.array_equal(category_indices, expected_categories), 'Native categorical positions differ')
    actual = model.get_all_params()
    base.need(actual['random_seed'] == 42 and actual['loss_function'] == 'MAE'
              and actual['eval_metric'] == 'MAE', 'Native seed or loss differs from R6')
    native_learning_rate = float(actual['learning_rate'])
    base.need(actual['depth'] == 10 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain', 'Native depth/grow/boosting policy differs')
    base.need(np.isclose(native_learning_rate, 0.065, rtol=1e-7, atol=0),
              'Native learning rate differs from the fixed 0.065 within storage tolerance')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 1024).all(),
              'Expected 2999 positive integer leaf counts, each at most 1024')
    return {
        'actual_model_depth': int(actual['depth']),
        'actual_boosting_type': actual['boosting_type'],
        'actual_grow_policy': actual['grow_policy'],
        'actual_trees': int(model.tree_count_),
        'maximum_tree_leaf_count': int(leaves.max()),
        'tree_leaf_counts_sha256': hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest(),
        'actual_learning_rate': native_learning_rate,
        'actual_max_leaves_parameter': actual.get('max_leaves'),
    }


def fit_model(X, y, geo, cats):
    """Train once on fit labels only, with uniform raw-price MAE."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fit target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fit target length differs from X')
    transformed = transform_target(raw)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    prepared_at = time.monotonic()
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    fit_feature_preparation_seconds = time.monotonic() - prepared_at
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'],
              'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Fit target changed')
    metadata = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'parameters': _copy_json(config), 'parameters_sha256': base.content_sha(config),
        'baseline_parameters': _copy_json(state['baseline_parameters']),
        'changed_parameter_names': ['depth'],
        'removed_feature_names': [],
        'added_feature_names': list(ADDED_FEATURES),
        'baseline_feature_count': 95,
        'original95_preserved': True,
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 95, 'feature_count': 95, 'categorical_feature_count': 23,
        'native_categorical_feature_count': 23,
        'native_learning_rate_comparison_rtol': 1e-7,
        'native_learning_rate_comparison_atol': 0.0,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round27_deep_geo.py'],
        'source_sha256': _copy_json(state['source_sha256']),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_rows': int(len(X)),
        'fit_original_feature_values_sha256': state['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256': state['fit_model_feature_values_sha256'],
        'fit_target_values_sha256': raw_sha, 'fit_original_target_values_sha256': raw_sha,
        'fit_transformed_target_values_sha256': transformed_sha,
        'target_value_hash_encoding': 'ordered C-contiguous little-endian float64',
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'query_labels_used': False, 'fit_uses_eval_set': False, 'early_stopping_used': False,
        'calibration_applied': False, 'extra_price_cap_applied': False,
        'reference_or_trend_applied_here': False, 'reference_features_changed': False,
        'external_geography_used': True, 'external_property_answers_used': False,
        'fit_feature_preparation_seconds': fit_feature_preparation_seconds,
        'elapsed_seconds': time.monotonic() - started,
        **actual,
    }
    return model, state, metadata


def predict_model(model, state, X, geo, cats):
    """Return unrounded nonnegative float64 TWD; no query labels or trend."""
    check_state(state, X, cats)
    augmented = augment_features(X, geo)
    base.need(list(augmented.columns) == state['all_feature_names'], 'Query feature order differs')
    check_model(model, state)
    raw = model.predict(Pool(augmented, cat_features=state['categorical_columns'], thread_count=4),
                        thread_count=4)
    result = inverse_transform(raw)
    base.need(result.shape == (len(X),), 'Prediction rows differ')
    return result



def self_check():
    """Bounded in-memory FakeCatBoost/Pool checks; no real configuration/data/model IO."""
    checks = []
    def ok(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, operation):
        try:
            operation()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name + ' did not reject')

    idx = pd.Index([41, 9, 77, 6], name='source_row')
    cats = ['category_' + str(i) for i in range(21)]
    X = pd.DataFrame({c: pd.Series(['原字串'] * len(idx), index=idx, dtype=object)
                      for c in cats}, index=idx)
    for i in range(66):
        X['numeric_' + str(i)] = np.arange(len(idx), dtype='float32')
    geo = pd.DataFrame({c: np.full(len(idx), np.nan, dtype='float32') for c in base.NUMERIC}, index=idx)
    geo['geo_has_position'] = np.zeros(len(idx), dtype='float32')
    geo['geo_match_status'] = pd.Series(['ineligible_address'] * len(idx), index=idx, dtype=object)
    geo['geo_grid_002'] = pd.Series(['__MISSING__'] * len(idx), index=idx, dtype=object)
    geo = geo[base.FEATURES]
    before_X, before_geo = X.copy(deep=True), geo.copy(deep=True)
    augmented = augment_features(X, geo)
    expected = base.augment_features(X, geo)
    ok('R6_95_values_dtypes_order_index_exact', augmented.equals(expected)
       and augmented.dtypes.equals(expected.dtypes) and augmented.index.equals(idx))
    ok('95_features_23_categories', len(augmented.columns) == 95 and len(categories(X, cats)) == 23)
    augmented.iloc[0, 0] = 'only output'
    ok('independent_feature_copy', X.equals(before_X) and geo.equals(before_geo))
    rejects('geo_index_misalignment_rejected', lambda: augment_features(X, geo.iloc[::-1]))
    rejects('target_column_rejected', lambda: augment_features(X.rename(columns={'numeric_0':'target'}), geo))
    rejects('non87_input_rejected', lambda: augment_features(X.assign(extra=np.float32(1)), geo))
    duplicate = X.copy(); duplicate.index = [0] * len(X)
    rejects('duplicate_index_rejected', lambda: augment_features(duplicate, geo))
    ok('raw_target_scale', np.array_equal(transform_target([0.0, 1e6, 3e6]), [0.0, 1.0, 3.0]))
    ok('raw_inverse_scale_and_zero_projection', np.array_equal(inverse_transform([-1.0, 0.0, 3.0]), [0.0, 0.0, 3e6]))
    for label, values in [('negative', [-1.0]), ('nan', [np.nan]), ('infinite', [np.inf])]:
        rejects('invalid_target_' + label, lambda v=values: transform_target(v))
    rejects('inverse_overflow_rejected', lambda: inverse_transform([np.finfo('float64').max]))
    ok('query_prediction_and_feature_state_have_no_y',
       list(inspect.signature(fit_feature_state).parameters) == ['X','geo','cats']
       and list(inspect.signature(predict_model).parameters) == ['model','state','X','geo','cats'])

    fake_baseline = {'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,
                     'loss_function':'MAE','task_type':'GPU','verbose':300,'learning_rate':0.065,'boosting_type':'Plain','eval_metric':'MAE'}
    saved_baseline = _copy_json(fake_baseline)
    class FakePool:
        def __init__(self, data, label=None, **kwargs):
            self.data, self.label, self.kwargs = data, label, kwargs
    class FakeModel:
        def __init__(self, **kwargs):
            self.config = _copy_json(kwargs)
            self.actual = {**kwargs, 'grow_policy':'SymmetricTree','max_leaves':1024}
            self.tree_count_ = 2999
            self.leaf_counts = np.full(2999, 1024, dtype=np.uint32)
        def fit(self, pool, **kwargs):
            if kwargs:
                raise AssertionError('Fit cannot receive eval_set/early-stop/weight kwargs')
            self.feature_names_ = list(pool.data.columns)
            self.fit_pool = pool
            self.cat_indices = sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self, pool, **kwargs):
            if pool.label is not None:
                raise AssertionError('Prediction cannot receive query labels')
            return pool.data['numeric_0'].to_numpy(dtype='float64') - 0.25
        def get_cat_feature_indices(self):
            return list(self.cat_indices)
        def get_all_params(self):
            return dict(self.actual)
        def get_tree_leaf_counts(self):
            return self.leaf_counts.copy()

    original_model, original_pool = globals()['CatBoostRegressor'], globals()['Pool']
    original_parameters = base.parameters
    original_config_sha = globals()['_base_configuration_sha256']
    globals()['CatBoostRegressor'], globals()['Pool'] = FakeModel, FakePool
    base.parameters = lambda: fake_baseline
    globals()['_base_configuration_sha256'] = lambda: 'a' * 64
    try:
        config = parameters()
        ok('only_depth_8_to_10', set(config) == set(fake_baseline)
           and [k for k in config if config[k] != fake_baseline[k]] == ['depth'] and config['depth'] == 10)
        config['depth'] = 99
        ok('parameters_are_independent_copies', fake_baseline == saved_baseline and parameters()['depth'] == 10)
        fake_baseline['depth'] = 10
        rejects('wrong_baseline_depth_rejected', parameters)
        fake_baseline['depth'] = 8
        y = pd.Series([0.0,1e6,2e6,3e6], index=idx, dtype='float64')
        model, state, meta = fit_model(X, y, geo, cats)
        ok('Pool_uniform_raw_fit_y_only', np.array_equal(model.fit_pool.label, y.to_numpy()/1e6)
           and set(model.fit_pool.kwargs) == {'cat_features','thread_count'}
           and model.fit_pool.kwargs['cat_features'] == categories(X,cats))
        ok('actual_constructor_depth_10_other_parameters_unchanged', model.config == parameters())
        ok('label_free_state_and_explicit_parameter_delta', state['fit_rows'] == len(X)
           and state['baseline_parameters'] == saved_baseline and state['changed_parameter_names'] == ['depth']
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        native = check_model(model,state)
        leaf_sha = hashlib.sha256(np.full(2999,1024,dtype='<i8').tobytes()).hexdigest()
        ok('native_depth_policy_leaf_cap_and_LE_int64_hash', native == {
            'actual_model_depth':10,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
            'maximum_tree_leaf_count':1024,'tree_leaf_counts_sha256':leaf_sha,
            'actual_boosting_type':'Plain','actual_learning_rate':.065,'actual_max_leaves_parameter':1024})
        ok('metadata_records_native_capacity_raw95_23_and_no_trend',
           all(meta[k] == v for k,v in native.items()) and meta['feature_count'] == 95
           and meta['categorical_feature_count'] == 23 and meta['target_mode'] == 'raw'
           and meta['sample_weight'] == 'uniform' and not meta['weight_argument_passed_to_pool']
           and not meta['reference_or_trend_applied_here'] and not meta['calibration_applied'])
        ok('raw_original_transformed_target_hashes',
           meta['fit_target_values_sha256'] == meta['fit_original_target_values_sha256'] == values_sha256(y)
           and meta['fit_transformed_target_values_sha256'] == values_sha256(y.to_numpy()/1e6))
        restored = _copy_json(state)
        prediction = predict_model(model,restored,X,geo,cats)
        ok('JSON_roundtrip_and_float64_TWD_prediction', prediction.dtype == np.dtype('float64')
           and np.array_equal(prediction, np.maximum(np.arange(len(X))-.25,0)*1e6))
        pieces = [predict_model(model,restored,X.iloc[a:b],geo.iloc[a:b],cats) for a,b in [(0,2),(2,4)]]
        ok('batch_and_full_equivalence', np.array_equal(prediction,np.concatenate(pieces)))
        query = X.iloc[[3,1]].copy(); query.iloc[0,0] = 'unseen category'
        predict_model(model,restored,query,geo.loc[query.index],cats)
        ok('query_unknown_category_preserved_without_refitting',
           augment_features(query,geo.loc[query.index]).iloc[0,0] == 'unseen category')
        bad = _copy_json(state); bad['feature_rules']['geo_encoder_rules']['usable_statuses'].append('fake')
        ok('state_rules_deep_copy', 'fake' not in FEATURE_RULES['geo_encoder_rules']['usable_statuses']
           and 'fake' not in base.FEATURE_RULES['usable_statuses'])
        rejects('modified_state_rejected', lambda: check_state(bad,X,cats))
        rejects('y_Series_index_misalignment_rejected', lambda: fit_model(X,y.iloc[::-1],geo,cats))
        model.actual['learning_rate'] = .0325
        rejects('wrong_native_learning_rate_rejected', lambda: check_model(model,state))
        model.actual['learning_rate'] = float(np.float32(.065))
        ok('native_float32_learning_rate_allowed',check_model(model,state)['actual_learning_rate']==float(np.float32(.065)))
        model.actual['learning_rate'] = .065
        previous_indices = model.cat_indices
        model.cat_indices = list(range(23))
        if model.cat_indices == previous_indices:model.cat_indices[-1]=94
        rejects('wrong_native_categorical_positions_rejected', lambda: check_model(model,state))
        model.cat_indices = previous_indices
        model.actual.pop('max_leaves')
        ok('absent_native_max_leaves_is_null_not_inferred',check_model(model,state)['actual_max_leaves_parameter'] is None)
        model.actual['max_leaves'] = 1024
        model.actual['depth'] = 8
        rejects('wrong_native_depth_rejected', lambda: check_model(model,state))
        rejects('predict_calls_native_model_guard', lambda: predict_model(model,state,X,geo,cats))
        model.actual['depth'] = 10
        model.actual['grow_policy'] = 'Lossguide'
        rejects('wrong_native_grow_policy_rejected', lambda: check_model(model,state))
        model.actual['grow_policy'] = 'SymmetricTree'
        for name,value in [('over1024',2048),('zero',0)]:
            model.leaf_counts[0] = value
            rejects('invalid_leaf_count_'+name, lambda: check_model(model,state))
        model.leaf_counts[0] = 1024
        saved_leaves = model.leaf_counts
        model.leaf_counts = saved_leaves.astype('float64')
        rejects('noninteger_leaf_dtype_rejected', lambda: check_model(model,state))
        model.leaf_counts = saved_leaves[:-1]
        rejects('wrong_leaf_vector_length_rejected', lambda: check_model(model,state))
        model.leaf_counts = saved_leaves
        model.tree_count_ = 2998
        rejects('wrong_native_tree_count_rejected', lambda: check_model(model,state))
        model.tree_count_ = 2999
        model.feature_names_ = list(reversed(model.feature_names_))
        rejects('wrong_native_feature_order_rejected', lambda: check_model(model,state))
        ok('all_input_frames_and_baseline_unchanged', X.equals(before_X) and geo.equals(before_geo)
           and fake_baseline == saved_baseline)
    finally:
        globals()['CatBoostRegressor'], globals()['Pool'] = original_model, original_pool
        base.parameters = original_parameters
        globals()['_base_configuration_sha256'] = original_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),
            'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'files_written':False,
            'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=True,allow_nan=False),flush=True)
