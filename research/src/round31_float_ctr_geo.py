"""R31: unchanged R6 raw geography plus one fixed FloatTargetMeanValue CTR.
No real data/config/model reads occur before an explicit API call. This helper
verifies native configuration, not whether trees selected the added statistic.
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

COMPONENT = 'geography_float_ctr_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ADDED_FEATURES = []
ADDED_CATEGORICAL_FEATURES = []
BASELINE_CTR_EVIDENCE_SHA = '68d062a1c15b2bcd1215fe31563788a22025f665356fdc059d65f310e422d93e'
BASELINE_CTR_EVIDENCE_PATH = 'outputs/round23_research/baseline_categorical_config.json'
BORDERS_CTR = 'Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1'
EXTRA_SIMPLE_CTR = 'FloatTargetMeanValue:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1'
PASSED_SIMPLE_CTR = [BORDERS_CTR, 'FeatureFreq:CtrBorderCount=15:CtrBorderType=MinEntropy:Prior=0/1', EXTRA_SIMPLE_CTR]
CTR_CONTRACT = {
    'simple_ctr': [BORDERS_CTR, 'FeatureFreq:CtrBorderCount=15:CtrBorderType=MinEntropy:Prior=0/1',
                   'FloatTargetMeanValue:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1'],
    'combinations_ctr': [BORDERS_CTR, 'FeatureFreq:CtrBorderCount=15:CtrBorderType=Median:Prior=0/1'],
    'ctr_target_border_count': 1,
    'one_hot_max_size': 8,
    'max_ctr_complexity': 2,
    'counter_calc_method': 'SkipTest',
    'ctr_history_unit': 'Sample',
}


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1,
    'passed_simple_ctr': _copy_json(PASSED_SIMPLE_CTR),
    'ctr_contract': _copy_json(CTR_CONTRACT),
    'baseline_ctr_evidence_sha256': BASELINE_CTR_EVIDENCE_SHA,
    'actual_split_usage_checked': False,
    'actual_float_ctr_split_used': None,
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
    'depth': 8,
    'baseline_boosting_type': 'Plain',
    'boosting_type': 'Plain',
    'changed_parameter_names': ['simple_ctr'],
    'removed_feature_names': [],
    'added_feature_names': list(ADDED_FEATURES),
    'baseline_feature_count': 95,
    'original95_preserved': True,
    'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 256,
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
        'src/round31_float_ctr_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['learning_rate'] == 0.065 and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == result['eval_metric'] == 'MAE' and result['boosting_type'] == 'Plain',
              'Expected the fixed R6 depth-8 Plain recipe')
    return result


def parameters():
    """Preserve every baseline parameter and append the single simple-CTR setting."""
    baseline = _baseline_parameters()
    base.need('simple_ctr' not in baseline, 'Unexpected baseline simple CTR override')
    result = _copy_json(baseline)
    result['simple_ctr'] = _copy_json(PASSED_SIMPLE_CTR)
    base.need(sorted(k for k in set(result) | set(baseline) if result.get(k) != baseline.get(k))
              == ['simple_ctr'], 'Only simple_ctr may change')
    return result


def _baseline_ctr_evidence_sha256():
    path = ROOT / BASELINE_CTR_EVIDENCE_PATH
    actual = sha256(path)
    base.need(actual == BASELINE_CTR_EVIDENCE_SHA, 'Baseline categorical evidence changed')
    base.need(path.with_suffix('.sha256').read_text(encoding='ascii').strip().split()[0] == actual,
              'Baseline categorical evidence receipt differs')
    return actual


def native_ctr_configuration(model):
    """Return and validate saved/fitted settings; this does not inspect tree usage."""
    native = model.get_all_params()
    observed = {key: _copy_json(native[key]) for key in CTR_CONTRACT}
    base.need(observed == CTR_CONTRACT, 'Native simple/combinations/global/counter CTR configuration differs')
    base.need(native.get('per_feature_ctr') in (None, {}, []), 'Unexpected per-feature CTR override')
    return observed



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
    base.need(len(X) > 0, 'Nonempty fit frame required')
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
        'baseline_ctr_evidence_sha256': _baseline_ctr_evidence_sha256(),
        'ctr_contract': _copy_json(CTR_CONTRACT),
        'baseline_parameters': _baseline_parameters(),
        'parameters': parameters(),
        'changed_parameter_names': ['simple_ctr'],
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
    base.need(state['fit_rows'] == state['geo_state']['fit_rows'] and state['fit_rows'] > 0,
              'Complete fit state row count differs')
    base.need(state['ctr_contract'] == CTR_CONTRACT
              and state['baseline_ctr_evidence_sha256'] == _baseline_ctr_evidence_sha256(),
              'CTR contract or baseline evidence changed')
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['target_mode'] == TARGET_MODE
              and state['feature_rules'] == FEATURE_RULES, 'Float-CTR raw model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == ['simple_ctr']
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True, 'Float-CTR configuration or original feature schema changed')
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
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 8
              and state['parameters']['boosting_type'] == 'Plain'
              and state['parameters']['iterations'] == 2999 and state['parameters']['learning_rate'] == 0.065
              and state['changed_parameter_names'] == ['simple_ctr']
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True
              and state['parameters'] == dict(state['baseline_parameters'], simple_ctr=PASSED_SIMPLE_CTR)
              and state['ctr_contract'] == CTR_CONTRACT
              and state['baseline_ctr_evidence_sha256'] == BASELINE_CTR_EVIDENCE_SHA
              and state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['all_feature_names']) == 95, 'Wrong Float-CTR raw model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    category_indices = np.asarray(model.get_cat_feature_indices())
    expected_categories = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(category_indices.shape == (23,) and category_indices.dtype.kind in 'iu'
              and len(expected_categories) == len(set(expected_categories)) == 23
              and np.array_equal(category_indices, expected_categories), 'Native categorical positions differ')
    native_ctr_configuration(model)
    actual = model.get_all_params()
    base.need(actual['random_seed'] == 42 and actual['loss_function'] == 'MAE'
              and actual['eval_metric'] == 'MAE', 'Native seed or loss differs from R6')
    native_learning_rate = float(actual['learning_rate'])
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain', 'Native depth/grow/boosting policy differs')
    base.need(np.isclose(native_learning_rate, 0.065, rtol=1e-7, atol=0),
              'Native learning rate differs from the fixed 0.065 within storage tolerance')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 256).all(),
              'Expected 2999 positive integer leaf counts, each at most 256')
    return {
        'actual_model_depth': int(actual['depth']),
        'actual_boosting_type': actual['boosting_type'],
        'actual_grow_policy': actual['grow_policy'],
        'actual_trees': int(model.tree_count_),
        'maximum_tree_leaf_count': int(leaves.max()),
        'tree_leaf_counts_sha256': hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest(),
        'actual_learning_rate': native_learning_rate,
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
        'changed_parameter_names': ['simple_ctr'],
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
        'script_sha256': state['source_sha256']['src/round31_float_ctr_geo.py'],
        'source_sha256': _copy_json(state['source_sha256']),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'baseline_ctr_evidence_sha256': state['baseline_ctr_evidence_sha256'],
        'ctr_contract': _copy_json(CTR_CONTRACT),
        'native_ctr_configuration': native_ctr_configuration(model),
        'actual_split_usage_checked': False,
        'actual_float_ctr_split_used': None,
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
    """Return full-query TWD with the raw inverse; no routing, reference or trend."""
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
    """Bounded FakePool checks, reserved for the formal CPU preflight."""
    global CatBoostRegressor, Pool, _base_configuration_sha256, _baseline_ctr_evidence_sha256
    checks = []
    def ok(name, value):
        base.need(bool(value), name); checks.append(name)
    def rejects(name, call):
        try:
            call()
        except (ValueError, KeyError):
            checks.append(name); return
        raise AssertionError('Expected rejection: ' + name)
    saved = (CatBoostRegressor, Pool, base.parameters, _base_configuration_sha256, _baseline_ctr_evidence_sha256)
    cfg = {'iterations': 2999, 'learning_rate': .065, 'thread_count': 4, 'depth': 8, 'random_seed': 42,
           'loss_function': 'MAE', 'eval_metric': 'MAE', 'boosting_type': 'Plain', 'task_type': 'GPU',
           'l2_leaf_reg': 6., 'max_ctr_complexity': 2, 'one_hot_max_size': 8, 'allow_writing_files': False}
    calls = []
    class FakePool:
        def __init__(self, data, label=None, **kwargs):
            base.need(set(kwargs) == {'cat_features', 'thread_count'}, 'Unexpected weight/Pool argument')
            self.data, self.label, self.cats = data, label, kwargs['cat_features']
    class FakeModel:
        def __init__(self, **kwargs):
            self.params = kwargs; self.tree_count_ = 2999; self.native_override = {}
        def fit(self, pool, **kwargs):
            base.need(not kwargs, 'Unexpected eval/early stopping')
            self.feature_names_ = list(pool.data.columns); self.cats = pool.cats
            self.y = np.asarray(pool.label); calls.append(len(pool.data))
        def predict(self, pool, thread_count=4):
            return np.arange(len(pool.data), dtype='float64') - 1
        def get_all_params(self):
            result = dict(self.params, grow_policy='SymmetricTree')
            result.update(_copy_json(CTR_CONTRACT)); result.update(self.native_override)
            return result
        def get_cat_feature_indices(self):
            return sorted(self.feature_names_.index(c) for c in self.cats)
        def get_tree_leaf_counts(self):
            return np.full(2999, 256, dtype='int64')
    try:
        CatBoostRegressor, Pool = FakeModel, FakePool
        base.parameters = lambda: dict(cfg)
        _base_configuration_sha256 = lambda: 'b'*64
        _baseline_ctr_evidence_sha256 = lambda: BASELINE_CTR_EVIDENCE_SHA
        idx = pd.Index([41, 9, 77, 6], name='source_row')
        X = pd.DataFrame({'main_use': ['住家用', '一般事務所', None, '其他'],
                          **{f'cat{i}': ['a', 'b', 'c', 'a'] for i in range(20)},
                          **{f'num{i}': np.array([1, 2, np.nan, 4], dtype='float32') for i in range(66)}}, index=idx)
        cats = list(X.columns[:21])
        geo = pd.DataFrame({c: np.ones(4, dtype='float32') for c in base.NUMERIC}, index=idx)
        geo['geo_latitude'] = np.full(4, 25, dtype='float32'); geo['geo_longitude'] = np.full(4, 121, dtype='float32')
        geo['geo_match_status'] = 'road_level'; geo['geo_grid_002'] = '1250:6050'
        before = (base.frame_sha(X), base.frame_sha(geo))
        y = pd.Series([1e6, 2e6, 0., 4e6], index=idx)
        model, state, meta = fit_model(X, y, geo, cats)
        ok('full_fit_all_uses_raw_target', calls == [4] and np.array_equal(model.y, [1., 2., 0., 4.]) and meta['fit_rows'] == 4)
        wanted = dict(cfg, simple_ctr=PASSED_SIMPLE_CTR)
        ok('only_simple_ctr_added', parameters() == wanted == meta['parameters'] and meta['baseline_parameters'] == cfg
           and meta['changed_parameter_names'] == ['simple_ctr'])
        ok('original_95_23_intact', meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23
           and state['all_feature_names'] == state['geo_state']['all_feature_names'] and meta['original95_preserved'])
        output = predict_model(model, state, X, geo, cats)
        ok('raw_inverse_full_query_order', np.array_equal(output, [0., 0., 1e6, 2e6]))
        restored = json.loads(json.dumps(state)); check_state(restored, X, cats)
        ok('state_roundtrip_no_query_or_input_mutation', restored == state and before == (base.frame_sha(X), base.frame_sha(geo)))
        ok('native_capacity_seven_and_categories', len(check_model(model, state)) == 7 and meta['native_categorical_feature_count'] == 23)
        ok('configuration_not_claimed_as_split_usage', meta['native_ctr_configuration'] == CTR_CONTRACT
           and meta['actual_float_ctr_split_used'] is None and meta['actual_split_usage_checked'] is False)
        for field, value in [('simple_ctr', CTR_CONTRACT['simple_ctr'][:2]),
                             ('combinations_ctr', []), ('ctr_target_border_count', 8),
                             ('counter_calc_method', 'Full')]:
            model.native_override = {field: value}
            rejects('reject_native_' + field, lambda: check_model(model, state))
        model.native_override = {'learning_rate': float(np.float32(.065))}
        ok('native_float32_lr_accepted', np.isfinite(check_model(model, state)['actual_learning_rate']))
        model.native_override = {'learning_rate': .0325}
        rejects('wrong_native_lr', lambda: check_model(model, state))
        model.native_override = {}
        rejects('target_index_mismatch', lambda: fit_model(X, y.reset_index(drop=True), geo, cats))
        rejects('geo_index_mismatch', lambda: augment_features(X, geo.reset_index(drop=True)))
        rejects('negative_target', lambda: transform_target(np.array([-1.])))
        rejects('nonfinite_target', lambda: transform_target(np.array([np.nan])))
        rejects('inverse_overflow', lambda: inverse_transform(np.array([np.finfo('float64').max])))
        ok('no_weights_eval_query_y_or_trend', not meta['weight_argument_passed_to_pool'] and not meta['fit_uses_eval_set']
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'] and not meta['reference_or_trend_applied_here'])
    finally:
        CatBoostRegressor, Pool, base.parameters, _base_configuration_sha256, _baseline_ctr_evidence_sha256 = saved
    return {'synthetic_only': True, 'real_model_training_or_config_or_data_read': False,
            'checks': checks, 'check_count': len(checks)}



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check-only',action='store_true',required=True)
    parser.parse_args();print(json.dumps(self_check(),ensure_ascii=False),flush=True)
