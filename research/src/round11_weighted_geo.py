"""R11: original 95 geographic features with prepared, pure-X importance weights.

No query labels or domain-classifier fitting enter this module. The caller verifies
query/indices/manifest/plan hashes against their actual files. This module verifies
only supplied fit-X and weight values, and preserves the other contract identities.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import round6_geo as base
from prepare_data import ROOT, sha256

COMPONENT = 'geography_raw_importance_weighted'
CONTRACT_HASH_FIELDS = [
    'fit_X_sha256', 'query_X_sha256', 'fit_indices_content_sha256',
    'query_indices_content_sha256', 'weights_values_sha256',
    'domain_plan_sha256', 'preparation_manifest_sha256',
]
LEGACY_TRAINING_FIELDS = {
    'fit_target', 'sample_weight', 'iterations', 'seed', 'no_eval_set',
    'no_early_stopping',
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
    'fit_target': 'raw TWD / 1e6',
    'prediction': 'max(raw model prediction * 1e6, 0)',
    'sample_weight': 'importance_clip_then_mean_one',
    'prepared_clipping_before_mean_normalization': [0.25, 4.0],
    'weights_dtype': 'float64',
    'weights_mean': 1.0,
    'helper_reclips_or_renormalizes_weights': False,
    'final_normalized_weights_need_not_be_within_pre_normalization_clip_bounds': True,
    'query_X_used_for_training_weights': True,
    'query_labels_used': False,
    'query_used_for_geographic_state': False,
    'target_used_for_geographic_state': False,
    'iterations': 2999,
    'seed': 42,
    'no_eval_set': True,
    'no_early_stopping': True,
    'reference_or_trend_applied_here': False,
}


def parameters():
    """Exactly the original R6 raw parameters; no search or overrides."""
    return base.parameters()


def weights_values_sha256(weights):
    """Canonical value identity, independent of NPY/Series file encoding."""
    return hashlib.sha256(np.asarray(weights, dtype='<f8').tobytes(order='C')).hexdigest()


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _sources():
    return {
        'src/round11_weighted_geo.py': sha256(Path(__file__)),
        'src/round6_geo.py': sha256(Path(base.__file__)),
        'src/prepare_data.py': sha256(ROOT / 'src/prepare_data.py'),
    }


def _check_X(X):
    base.need(isinstance(X, pd.DataFrame), 'X must be a DataFrame')
    base.need(X.columns.is_unique, 'Feature column names must be unique')
    base.need(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
              'Identifiers/targets must not enter model features')


def _check_contract_structure(contract):
    base.need(isinstance(contract, dict), 'Weight contract must be a JSON object')
    base.need(contract.get('phase') in {'development', 'confirmation', 'auxiliary', 'final'},
              'Unknown weight preparation phase')
    for key in CONTRACT_HASH_FIELDS:
        value = contract.get(key)
        base.need(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
                  'Missing or invalid weight contract SHA256: ' + key)
    base.need(contract.get('query_X_used_for_weights') is True,
              'Weight contract must declare use of query X')
    base.need(contract.get('query_labels_used_for_weights') is False,
              'Weight contract must forbid query labels')
    return _copy_json(contract)


def validate_weights(X, weights, weight_contract):
    """Validate, copy and return prepared weights plus a detached contract.

    For arrays, ordered value SHA is the row-identity guard. A Series additionally
    has to match X.index. Clipping was performed by the preparation process; after
    mean normalization a valid weight can lie outside the earlier clip bounds.
    """
    _check_X(X)
    contract = _check_contract_structure(weight_contract)
    if isinstance(weights, pd.Series):
        base.need(weights.index.equals(X.index), 'Weight Series index differs from X')
        values = weights.to_numpy(copy=True)
    else:
        values = np.asarray(weights)
    base.need(values.dtype == np.dtype('float64'), 'Prepared weights must be float64')
    base.need(values.shape == (len(X),) and len(X) > 0, 'Weight length/shape differs from X')
    base.need(np.isfinite(values).all() and (values > 0).all(),
              'Weights must be finite and strictly positive')
    base.need(np.isclose(values.mean(dtype=np.float64), 1.0, rtol=0, atol=1e-10),
              'Prepared weights must already have mean one')
    base.need(contract['fit_X_sha256'] == base.frame_sha(X), 'Weight contract fit-X identity differs')
    base.need(contract['weights_values_sha256'] == weights_values_sha256(values),
              'Weight value/order identity differs')
    return np.array(values, dtype=np.float64, order='C', copy=True), contract


def _weight_summary(values):
    total = float(values.sum(dtype=np.float64))
    return {
        'rows': int(len(values)), 'dtype': str(values.dtype),
        'min': float(values.min()), 'max': float(values.max()),
        'mean': float(values.mean(dtype=np.float64)), 'sum': total,
        'effective_sample_size_rows': float(total ** 2 / np.dot(values, values)),
    }


def fit_feature_state(X, geo, cats, weights, weight_contract):
    """Build a label-free wrapper state; no query frame is fitted here."""
    values, contract = validate_weights(X, weights, weight_contract)
    geo_state = _copy_json(base.fit_feature_state(X, geo))
    categories = base.categories(X, cats)
    base.need(len(categories) == 23 and len(set(categories)) == 23,
              'Expected 23 distinct categorical features')
    state = {
        'format_version': 1, 'component': COMPONENT,
        'feature_rules': _copy_json(FEATURE_RULES),
        'source_sha256': _sources(),
        'base_configuration_sha256': sha256(ROOT / 'runs/v1/models/final_config.json'),
        'parameters': parameters(),
        'geo_state': geo_state,
        'original_categorical_columns': list(cats),
        'categorical_columns': categories,
        'all_feature_names': list(geo_state['all_feature_names']),
        'fit_rows': int(len(X)),
        'weight_contract': contract,
        'weight_contract_sha256': base.content_sha(contract),
        'fit_sample_weight_values_sha256': weights_values_sha256(values),
        'fit_sample_weight_summary': _weight_summary(values),
        'fit_state_uses_y': False,
        'geographic_feature_state_uses_query': False,
        'query_X_used_for_training_weights': True,
        'query_labels_used': False,
        'helper_verified_contract_fields': ['fit_X_sha256', 'weights_values_sha256'],
        'other_contract_hashes_must_be_verified_by_caller': [
            k for k in CONTRACT_HASH_FIELDS if k not in {'fit_X_sha256', 'weights_values_sha256'}
        ],
    }
    state['state_content_sha256'] = base.content_sha(state)
    return state


def check_state(state, X, cats):
    _check_X(X)
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'
    }), 'Weighted model state content changed')
    base.need(state['feature_rules'] == FEATURE_RULES and state['component'] == COMPONENT,
              'Weighted model rules changed')
    base.need(state['source_sha256'] == _sources(), 'Weighted helper/dependency source changed')
    base.need(state['base_configuration_sha256'] == sha256(ROOT / 'runs/v1/models/final_config.json'),
              'Base model configuration changed')
    base.need(state['parameters'] == parameters(), 'Original R6 parameters changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns'], 'Original category order changed')
    base.need(base.categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names'],
              'Wrapped geography schema changed')
    contract = _check_contract_structure(state['weight_contract'])
    base.need(base.content_sha(contract) == state['weight_contract_sha256'], 'Weight contract changed')
    base.need(contract['fit_X_sha256'] == state['geo_state']['fit_original_feature_values_sha256'],
              'Fit-X and geo-state identities differ')
    base.need(contract['weights_values_sha256'] == state['fit_sample_weight_values_sha256'],
              'Stored weight identities differ')
    base.need(state['query_X_used_for_training_weights'] is True and state['query_labels_used'] is False
              and state['geographic_feature_state_uses_query'] is False and state['fit_state_uses_y'] is False,
              'State data-use declarations changed')


def fit_model(X, y, geo, cats, weights, weight_contract):
    """Fit raw R6 CatBoost with already prepared pure-X importance weights."""
    started = time.monotonic()
    values, contract = validate_weights(X, weights, weight_contract)
    state = fit_feature_state(X, geo, cats, values, contract)
    y_values = np.asarray(y, dtype=np.float64)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fitting target Series index differs from X')
    base.need(y_values.shape == (len(X),) and np.isfinite(y_values).all(), 'Invalid fitting target')
    augmented = base.augment_features(X, geo)
    categories = base.categories(X, cats)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, y_values / 1e6, weight=values,
                   cat_features=categories, thread_count=4))
    base.need(int(model.tree_count_) == 2999, 'Expected all fixed 2999 trees')
    base.need(_sources() == state['source_sha256'], 'Sources changed during fit')
    base.need(parameters() == config and sha256(ROOT / 'runs/v1/models/final_config.json')
              == state['base_configuration_sha256'], 'Base parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256'],
              'Fit feature inputs changed')
    base.need(weights_values_sha256(values) == state['fit_sample_weight_values_sha256'],
              'Weights passed to Pool changed')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Model feature names differ')
    metadata = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': 'raw',
        'parameters': config, 'iterations': 2999,
        'features': 95, 'feature_count': 95, 'categorical_feature_count': 23,
        'categorical_features': categories,
        'sample_weight': FEATURE_RULES['sample_weight'],
        'fit_sample_weight_values_sha256': state['fit_sample_weight_values_sha256'],
        'fit_sample_weight_summary': state['fit_sample_weight_summary'],
        'weight_contract': _copy_json(contract),
        'weight_contract_sha256': state['weight_contract_sha256'],
        'feature_rules': _copy_json(FEATURE_RULES),
        'geo_encoder_rules': _copy_json(FEATURE_RULES['geo_encoder_rules']),
        'legacy_geo_state_training_fields_are_not_this_models_training_declaration': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'geo_state_content_sha256': state['geo_state']['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round11_weighted_geo.py'],
        'source_sha256': state['source_sha256'],
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_target_values_sha256': hashlib.sha256(y_values.astype('<f8').tobytes()).hexdigest(),
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'query_X_used_for_training_weights': True, 'query_labels_used': False,
        'helper_verified_contract_fields': state['helper_verified_contract_fields'],
        'other_contract_hashes_must_be_verified_by_caller': state['other_contract_hashes_must_be_verified_by_caller'],
        'reference_features_changed': False, 'reference_or_trend_applied_here': False,
        'external_geography_used': True, 'external_property_answers_used': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    return model, state, metadata


def predict_model(model, state, X, geo, cats):
    """Predict from X/geo only; the caller binds the full transductive query batch."""
    check_state(state, X, cats)
    augmented = base.augment_features(X, geo)
    base.need(list(model.feature_names_) == list(augmented.columns) == state['all_feature_names'],
              'Saved model feature names differ')
    prediction = np.maximum(np.asarray(model.predict(
        Pool(augmented, cat_features=state['categorical_columns'], thread_count=4),
        thread_count=4), dtype=np.float64) * 1e6, 0)
    base.need(prediction.shape == (len(X),) and np.isfinite(prediction).all(),
              'Invalid price predictions')
    return prediction


def self_check():
    """Synthetic-only contract checks. Fake factories never train CatBoost."""
    from house_features import build_features, get_categorical_columns

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
        'geo_latitude': [25.0, np.nan, 22.6, np.nan],
        'geo_longitude': [121.5, np.nan, 120.3, np.nan],
        'geo_road_length_m': [1000.0, np.nan, 700.0, np.nan],
        'geo_extent_diagonal_m': [900.0, np.nan, 0.0, np.nan],
        'geo_nearby_fragment_grouping': [0.0, np.nan, 1.0, np.nan],
        'geo_match_status': ['road_level', 'ineligible_address', 'planar_intersection', 'not_found'],
    }, index=X.index)
    geo = base.encode_geo_features(matched)
    weights = np.array([0.25, 0.5, 1.0, 4.0], dtype=np.float64)
    weights /= weights.mean()
    y = np.array([1000000, 2000000, 3000000, 4000000], dtype=np.float64)
    contract = {k: '1' * 64 for k in CONTRACT_HASH_FIELDS}
    contract.update({
        'phase': 'development', 'fit_X_sha256': base.frame_sha(X),
        'weights_values_sha256': weights_values_sha256(weights),
        'query_X_used_for_weights': True, 'query_labels_used_for_weights': False,
    })
    checks = []

    def check(name, condition):
        base.need(bool(condition), 'Synthetic check failed: ' + name)
        checks.append(name)

    def rejects(name, operation):
        try:
            operation()
        except (ValueError, TypeError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    class FakePool:
        def __init__(self, data, label=None, *, weight=None, cat_features, thread_count):
            self.data = data.copy(deep=True)
            self.label = None if label is None else np.asarray(label).copy()
            self.weight = None if weight is None else np.asarray(weight).copy()
            self.cat_features = list(cat_features)
            self.thread_count = thread_count

    class FakeCat:
        def __init__(self, **config):
            self.config = config
            self.fit_calls = 0
            self.predict_pools = []

        def fit(self, pool):
            self.fit_calls += 1
            self.fit_pool = pool
            self.feature_names_ = list(pool.data.columns)
            self.tree_count_ = 2999
            return self

        def predict(self, pool, *, thread_count):
            self.predict_pools.append(pool)
            self.prediction_thread_count = thread_count
            return np.arange(len(pool.data), dtype=np.float64) - 1.0

    global CatBoostRegressor, Pool
    original_cat, original_pool = CatBoostRegressor, Pool
    CatBoostRegressor, Pool = FakeCat, FakePool
    before_X, before_geo = X.copy(deep=True), geo.copy(deep=True)
    before_weights, before_contract = weights.copy(), _copy_json(contract)
    try:
        model, state, metadata = fit_model(X, y, geo, cats, weights, contract)
        check('fixture_87_original_and_21_categories', X.shape == (4, 87) and len(cats) == 21)
        check('exact_original_95_values_dtypes_order_index',
              model.fit_pool.data.equals(base.augment_features(X, geo))
              and model.fit_pool.data.dtypes.equals(base.augment_features(X, geo).dtypes)
              and model.fit_pool.data.index.equals(X.index))
        check('nonuniform_float64_weights_reach_pool',
              model.fit_pool.weight.dtype == np.dtype('float64')
              and np.array_equal(model.fit_pool.weight, weights)
              and not np.all(model.fit_pool.weight == 1))
        check('raw_target_scaling', np.array_equal(model.fit_pool.label, y / 1e6))
        check('exact_R6_parameters_2999_no_eval_set',
              model.config == base.parameters() and model.config['iterations'] == 2999
              and model.fit_calls == 1 and model.fit_pool.thread_count == 4)
        check('all_23_categorical_columns_preserved',
              model.fit_pool.cat_features == cats + base.CATEGORICAL
              and len(model.fit_pool.cat_features) == 23)
        check('wrapper_geo_state_exactly_original_fit_only',
              state['geo_state'] == base.fit_feature_state(X, geo)
              and state['geo_state']['fit_state_uses_y'] is False
              and state['geo_state']['fit_state_uses_query'] is False)
        check('metadata_distinguishes_training_weights_from_geo_state',
              metadata['sample_weight'] == 'importance_clip_then_mean_one'
              and metadata['query_X_used_for_training_weights'] is True
              and metadata['query_labels_used'] is False
              and metadata['query_used_for_geographic_state'] is False
              and 'sample_weight' not in metadata['geo_encoder_rules']
              and metadata['fit_uses_eval_set'] is False and metadata['early_stopping_used'] is False)
        check('caller_only_identities_not_claimed_verified',
              state['helper_verified_contract_fields'] == ['fit_X_sha256', 'weights_values_sha256']
              and 'query_X_sha256' in state['other_contract_hashes_must_be_verified_by_caller'])
        check('source_and_state_content_identity', state['source_sha256'] == _sources()
              and metadata['feature_state_content_sha256'] == state['state_content_sha256'])
        roundtripped = _copy_json(state)
        check_state(roundtripped, X, cats)
        check('state_JSON_roundtrip', roundtripped == state)
        predictions = predict_model(model, roundtripped, X, geo, cats)
        check('predict_original_TWD_nonnegative_float64',
              predictions.dtype == np.dtype('float64')
              and np.array_equal(predictions, [0, 0, 1000000, 2000000]))
        check('predict_has_no_labels_or_weights',
              list(inspect.signature(predict_model).parameters) == ['model', 'state', 'X', 'geo', 'cats']
              and model.predict_pools[-1].label is None and model.predict_pools[-1].weight is None)
        predict_model(model, state, X.iloc[[1, 3]], geo.iloc[[1, 3]], cats)
        check('query_slices_keep_nonrange_index',
              model.predict_pools[-1].data.index.equals(X.iloc[[1, 3]].index))
        check('no_input_mutation', X.equals(before_X) and X.dtypes.equals(before_X.dtypes)
              and geo.equals(before_geo) and geo.dtypes.equals(before_geo.dtypes)
              and np.array_equal(weights, before_weights) and contract == before_contract)
        series_values, _ = validate_weights(X, pd.Series(weights, index=X.index), contract)
        check('matching_weight_Series_accepted', np.array_equal(series_values, weights))
        check('mean_normalized_value_below_initial_clip_accepted', weights.min() < 0.25)
        rejects('wrong_weight_Series_index_rejected', lambda: validate_weights(
            X, pd.Series(weights, index=X.index[::-1]), contract))
        rejects('wrong_weight_shape_rejected', lambda: validate_weights(X, weights[:, None], contract))
        rejects('wrong_weight_length_rejected', lambda: validate_weights(X, weights[:-1], contract))
        rejects('non_float64_weight_rejected', lambda: validate_weights(X, weights.astype('float32'), contract))
        rejects('wrong_weight_order_rejected', lambda: validate_weights(X, weights[::-1].copy(), contract))
        rejects('wrong_fit_row_order_rejected', lambda: validate_weights(X.iloc[::-1], weights, contract))
        for label, bad_value in [('zero', 0.0), ('negative', -0.1), ('nan', np.nan), ('inf', np.inf)]:
            bad = weights.copy()
            bad[0] = bad_value
            rejects(label + '_weight_rejected', lambda bad=bad: validate_weights(X, bad, contract))
        rejects('non_normalized_weight_rejected', lambda: validate_weights(X, weights * 2, contract))
        bad_contract = dict(contract, query_labels_used_for_weights=True)
        rejects('query_label_use_rejected', lambda: validate_weights(X, weights, bad_contract))
        bad_contract = dict(contract, query_indices_content_sha256='missing')
        rejects('missing_query_indices_identity_rejected', lambda: validate_weights(X, weights, bad_contract))
        bad_geo = geo.copy()
        bad_geo.index = bad_geo.index[::-1]
        rejects('geo_index_misalignment_rejected', lambda: fit_feature_state(X, bad_geo, cats, weights, contract))
        changed = _copy_json(state)
        changed['weight_contract']['phase'] = 'confirmation'
        rejects('state_tampering_rejected', lambda: check_state(changed, X, cats))
        rejects('category_order_change_rejected', lambda: predict_model(model, state, X, geo, cats[::-1]))
        target_series = pd.Series(y, index=X.index[::-1])
        rejects('target_Series_misalignment_rejected', lambda: fit_model(
            X, target_series, geo, cats, weights, contract))
        check('state_independent_of_target', fit_feature_state(X, geo, cats, weights, contract) == state)
    finally:
        CatBoostRegressor, Pool = original_cat, original_pool
    check('real_factories_restored_without_training',
          CatBoostRegressor is original_cat and Pool is original_pool)
    return checks


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    result = self_check()
    print(json.dumps({'synthetic_only': True, 'checks': result,
                      'checks_passed': len(result), 'script_sha256': sha256(Path(__file__))},
                     ensure_ascii=False, indent=2))
