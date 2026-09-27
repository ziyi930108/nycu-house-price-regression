"""R13 prototype: one additive city alias on the unchanged R6 raw component.

Only a complete address_city value equal to 桃園縣 becomes 桃園市 in the
new model-only column. Original X, geographic features, reference and scoring
keys stay unchanged. No data loading, training or artifact writes on import.
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

COMPONENT = 'geography_city_alias_raw_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ALIAS_COLUMN = 'r13_address_city_canonical'
CITY_ALIAS = {'桃園縣': '桃園市'}
MAPPING_EVIDENCE_SHA256 = {
    'reports/round13_city_alias_feasibility.md': 'c1136006cfb90160ecd94644fa12fab6fff639e967914600401d88cf98975a36',
    'reports/round13_official_name_evidence.md': 'efb85ba2cc96c5a3cd6a77b0fae8592e4d37194c0102988397d9f951fbb6b0ce',
}
FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'existing_geo_features_unchanged': 8,
    'existing_95_values_dtypes_order_and_index_unchanged': True,
    'new_categorical_column': ALIAS_COLUMN,
    'alias_input_column': 'address_city',
    'alias_mapping': CITY_ALIAS,
    'matching': 'complete existing string equality only; no stripping or substring replacement',
    'unmatched_and_missing': 'preserve original value in new column',
    'alias_dtype': 'object',
    'alias_position': 'append after all original 95 R6 model columns',
    'district_or_address_parsing_changed': False,
    'transaction_date_used_for_alias': False,
    'query_vocabulary_or_target_used_for_alias': False,
    'reference_scoring_grouping_or_dedup_keys_changed': False,
    'historical_boundary_identity_claimed': False,
    'feature_count': 96,
    'categorical_feature_count': 24,
    'geo_encoder_rules': base.FEATURE_RULES,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'sample_weight': 'uniform',
    'weight_argument_passed_to_pool': False,
    'iterations': 2999,
    'threads': 4,
    'seed': 42,
    'no_eval_set': True,
    'no_early_stopping': True,
    'reference_or_trend_applied_here': False,
    'extra_price_cap_applied': False,
    'unchanged_alias_rows_have_old_model_predictions_guaranteed': False,
}


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def source_hashes():
    return {name: sha256(ROOT / name) for name in (
        'src/round13_city_alias_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


def evidence_hashes():
    actual = {name: sha256(ROOT / name) for name in MAPPING_EVIDENCE_SHA256}
    base.need(actual == MAPPING_EVIDENCE_SHA256, 'Frozen city-alias evidence changed')
    return actual


def parameters():
    result = base.parameters()
    base.need(result['iterations'] == 2999 and result['thread_count'] == 4
              and result['random_seed'] == 42 and result['loss_function'] == 'MAE',
              'Fixed R6 raw parameters differ')
    return result


def values_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes(order='C')).hexdigest()


def _check_X(X):
    base.need(isinstance(X, pd.DataFrame), 'X must be a DataFrame')
    base.need(X.columns.is_unique and X.index.is_unique, 'X columns and indices must be unique')
    base.need('address_city' in X and ALIAS_COLUMN not in X, 'Expected unaugmented address_city input')
    base.need(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
              'Identifiers and targets cannot enter model features')


def canonical_city(city):
    """Fixed full-string projection, preserving every other value including NA."""
    base.need(isinstance(city, pd.Series), 'City input must be a Series')
    result = city.astype(object).copy(deep=True)
    for old, new in CITY_ALIAS.items():
        result.loc[result.eq(old).fillna(False)] = new
    result.name = ALIAS_COLUMN
    return result


def augment_features(X, geo):
    """Return a new 96-column model view; never modify either input frame."""
    _check_X(X)
    original = base.augment_features(X, geo)
    alias = canonical_city(X['address_city'])
    result = pd.concat([original, alias], axis=1)
    base.need(len(result.columns) == 96 and result.columns.is_unique
              and list(result.columns) == list(original.columns) + [ALIAS_COLUMN],
              'Expected 95 unchanged columns followed by one city alias')
    base.need(result[original.columns].equals(original)
              and result[original.columns].dtypes.equals(original.dtypes)
              and result.index.equals(X.index), 'Original 95 features changed')
    base.need(result[ALIAS_COLUMN].dtype == object, 'Alias must be an object category')
    return result


def categories(X, supplied):
    result = base.categories(X, supplied) + [ALIAS_COLUMN]
    base.need(len(result) == len(set(result)) == 24, 'Expected 24 unique categorical columns')
    return result


def _float_vector(values, name):
    raw = values.to_numpy(copy=True) if isinstance(values, pd.Series) else np.asarray(values)
    base.need(raw.ndim == 1 and raw.dtype.kind in 'iuf', name + ' must be real numeric and one dimensional')
    result = np.array(raw, dtype=np.float64, order='C', copy=True)
    base.need(np.isfinite(result).all(), name + ' must be finite')
    return result


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


def fit_feature_state(X, geo, cats):
    """Record fit-X identities and fixed rules only; accepts neither y nor query."""
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
        'mapping_evidence_sha256': evidence_hashes(),
        'base_configuration_sha256': sha256(ROOT / 'runs/v1/models/final_config.json'),
        'parameters': parameters(),
        'geo_state': geo_state,
        'original_categorical_columns': list(cats),
        'categorical_columns': categories(X, cats),
        'all_feature_names': list(augmented.columns),
        'fit_rows': int(len(X)),
        'fit_alias_changed_rows': int(X['address_city'].eq('桃園縣').sum()),
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
              and state['feature_rules'] == FEATURE_RULES, 'Model feature rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['mapping_evidence_sha256'] == evidence_hashes(), 'Mapping evidence changed')
    base.need(state['base_configuration_sha256'] == sha256(ROOT / 'runs/v1/models/final_config.json')
              and state['parameters'] == parameters(), 'Base configuration changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    expected = list(state['geo_state']['all_feature_names']) + [ALIAS_COLUMN]
    base.need(state['all_feature_names'] == expected and len(expected) == 96,
              'Model schema must be unchanged R6 features plus the final alias')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def fit_model(X, y, geo, cats):
    """Train the fixed raw component once; no evaluation set or early stopping."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fit target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fit target length differs from X')
    transformed = transform_target(raw)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, cat_features=state['categorical_columns'], thread_count=4))
    base.need(int(model.tree_count_) == 2999, 'Expected all fixed 2999 trees')
    check_state(state, X, cats)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'],
              'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Fit target changed')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Model names differ')
    metadata = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'parameters': config, 'parameters_sha256': base.content_sha(config),
        'iterations': 2999, 'features': 96, 'feature_count': 96, 'categorical_feature_count': 24,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round13_city_alias_geo.py'],
        'source_sha256': _copy_json(state['source_sha256']),
        'mapping_evidence_sha256': _copy_json(state['mapping_evidence_sha256']),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_rows': int(len(X)), 'fit_alias_changed_rows': state['fit_alias_changed_rows'],
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
        'elapsed_seconds': time.monotonic() - started,
    }
    return model, state, metadata


def predict_model(model, state, X, geo, cats):
    """Return unrounded, nonnegative float64 TWD; never apply a trend."""
    check_state(state, X, cats)
    augmented = augment_features(X, geo)
    base.need(list(model.feature_names_) == list(augmented.columns) == state['all_feature_names'],
              'Model feature names differ')
    base.need(int(model.tree_count_) == 2999, 'Model tree count differs')
    raw = model.predict(Pool(augmented, cat_features=state['categorical_columns'], thread_count=4),
                        thread_count=4)
    result = inverse_transform(raw)
    base.need(result.shape == (len(X),), 'Prediction rows differ')
    return result


def self_check():
    """Small in-memory fixtures and a fake estimator; no real fitting or IO writes."""
    checks = []

    def ok(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, fn):
        try:
            fn()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name + ' did not reject')

    cities = ['桃園縣', '桃園市', '桃園縣桃園市', '新竹縣', '新竹市',
              '嘉義縣', '嘉義市', '__MISSING__', ' 桃園縣', '桃園縣路']
    idx = pd.Index([41, 9, 77, 6, 13, 99, 2, 55, 8, 70], name='source_row')
    cats = ['address_city'] + ['cat_' + str(i) for i in range(20)]
    X = pd.DataFrame({name: pd.Series(['原字串'] * len(idx), index=idx, dtype=object)
                      for name in cats}, index=idx)
    X['address_city'] = pd.Series(cities, index=idx, dtype=object)
    X['cat_0'] = pd.Series(['桃園市'] * len(idx), index=idx, dtype=object)
    for n in range(66):
        X['numeric_' + str(n)] = np.arange(len(idx), dtype='float32')
    geo = pd.DataFrame({c: np.full(len(idx), np.nan, dtype='float32') for c in base.NUMERIC}, index=idx)
    geo['geo_has_position'] = np.zeros(len(idx), dtype='float32')
    geo['geo_match_status'] = pd.Series(['ineligible_address'] * len(idx), index=idx, dtype=object)
    geo['geo_grid_002'] = pd.Series(['__MISSING__'] * len(idx), index=idx, dtype=object)
    geo = geo[base.FEATURES]
    before_X, before_geo = X.copy(deep=True), geo.copy(deep=True)
    result = augment_features(X, geo)
    expected = ['桃園市'] + cities[1:]
    ok('full_string_only_and_non_Taoyuan_counties_preserved', result[ALIAS_COLUMN].tolist() == expected)
    ok('literal_modern_city_unchanged', result[ALIAS_COLUMN].iloc[1] == '桃園市')
    ok('county_child_string_not_replaced', result[ALIAS_COLUMN].iloc[2] == '桃園縣桃園市')
    ok('missing_sentinel_preserved', result[ALIAS_COLUMN].iloc[7] == '__MISSING__')
    probe = pd.Series([None, np.nan, pd.NA, '__MISSING__', ''], dtype=object)
    pd.testing.assert_series_equal(canonical_city(probe), probe.rename(ALIAS_COLUMN))
    checks.append('literal_NA_and_empty_values_preserved')
    ok('original_95_values_dtypes_order_index_preserved',
       result.iloc[:, :95].equals(base.augment_features(X, geo))
       and result.iloc[:, :95].dtypes.equals(base.augment_features(X, geo).dtypes)
       and result.index.equals(idx))
    ok('exact_96_features_24_categories', len(result.columns) == 96 and len(categories(X, cats)) == 24)
    ok('alias_last_object_column', result.columns[-1] == ALIAS_COLUMN and result[ALIAS_COLUMN].dtype == object)
    result.loc[idx[0], 'address_city'] = 'only_output_changed'
    result.loc[idx[0], 'geo_match_status'] = 'only_output_changed'
    ok('output_is_independent_copy', X.equals(before_X) and geo.equals(before_geo))
    ok('canonical_projection_idempotent', canonical_city(canonical_city(X.address_city)).equals(canonical_city(X.address_city)))
    rejects('misaligned_geo_rejected', lambda: augment_features(X, geo.iloc[::-1]))
    rejects('already_augmented_rejected', lambda: augment_features(X.assign(**{ALIAS_COLUMN: 'x'}), geo))
    duplicate = X.copy(); duplicate.index = [0] * len(X)
    rejects('duplicate_row_indices_rejected', lambda: augment_features(duplicate, geo))
    rejects('missing_city_column_rejected', lambda: augment_features(X.rename(columns={'address_city': 'old_city'}), geo))
    for label, values in [('negative', [-1.0]), ('nan', [np.nan]), ('infinite', [np.inf])]:
        rejects('invalid_fit_target_' + label, lambda v=values: transform_target(v))
    ok('raw_target_scale_without_log', np.array_equal(transform_target([0.0, 1e6, 3e6]), [0.0, 1.0, 3.0]))
    ok('raw_inverse_scale_and_zero_projection', np.array_equal(inverse_transform([-1.0, 0.0, 3.0]), [0.0, 0.0, 3e6]))
    rejects('raw_inverse_overflow_rejected', lambda: inverse_transform([np.finfo('float64').max]))
    ok('feature_and_prediction_apis_have_no_query_y',
       list(inspect.signature(fit_feature_state).parameters) == ['X', 'geo', 'cats']
       and list(inspect.signature(predict_model).parameters) == ['model', 'state', 'X', 'geo', 'cats'])

    calls = []
    class FakePool:
        def __init__(self, data, label=None, **kwargs):
            self.data, self.label, self.kwargs = data, label, kwargs
            calls.append(self)
    class FakeModel:
        def __init__(self, **kwargs):
            self.config, self.tree_count_ = kwargs, 2999
        def fit(self, pool, **kwargs):
            if kwargs:
                raise AssertionError('No fit kwargs, weights or evaluation set are allowed')
            self.feature_names_ = list(pool.data.columns)
            self.fit_pool = pool
            return self
        def predict(self, pool, **kwargs):
            if pool.label is not None:
                raise AssertionError('Prediction must not receive labels')
            return pool.data['numeric_0'].to_numpy(dtype='float64') + 0.25

    original_model, original_pool = globals()['CatBoostRegressor'], globals()['Pool']
    globals()['CatBoostRegressor'], globals()['Pool'] = FakeModel, FakePool
    try:
        y = pd.Series(np.arange(1, len(X) + 1, dtype='float64') * 1e6, index=idx)
        model, state, meta = fit_model(X, y, geo, cats)
        ok('fake_fit_uses_only_uniform_raw_scaled_fit_y',
           np.array_equal(model.fit_pool.label, y.to_numpy() / 1e6)
           and set(model.fit_pool.kwargs) == {'cat_features', 'thread_count'})
        ok('all_original_parameters_retained', model.config == base.parameters())
        ok('fit_state_only_fit_rows_and_fixed_alias', state['fit_rows'] == len(X)
           and state['fit_alias_changed_rows'] == 1 and not state['fit_state_uses_y']
           and not state['fit_state_uses_query'])
        ok('metadata_raw_96_24_2999_no_trend', meta['target_mode'] == 'raw'
           and meta['feature_count'] == 96 and meta['categorical_feature_count'] == 24
           and meta['iterations'] == 2999 and not meta['reference_or_trend_applied_here']
           and meta['calibration_applied'] is False and meta['extra_price_cap_applied'] is False)
        reloaded = _copy_json(state)
        prediction = predict_model(model, reloaded, X, geo, cats)
        ok('json_state_reload_and_raw_TWD_prediction',
           np.array_equal(prediction, (np.arange(len(X)) + 0.25) * 1e6)
           and prediction.dtype == np.dtype('float64'))
        pieces = [predict_model(model, reloaded, X.iloc[a:b], geo.iloc[a:b], cats)
                  for a, b in [(0, 3), (3, len(X))]]
        ok('batch_and_full_prediction_equal', np.array_equal(prediction, np.concatenate(pieces)))
        q = X.iloc[[3, 1]].copy(); q.loc[q.index[0], 'address_city'] = '未見城市'
        ok('unseen_city_is_preserved_without_learning_query_vocab',
           augment_features(q, geo.loc[q.index])[ALIAS_COLUMN].iloc[0] == '未見城市')
        predict_model(model, reloaded, q, geo.loc[q.index], cats)
        checks.append('query_row_order_and_changed_city_values_allowed')
        bad = _copy_json(state); bad['feature_rules']['alias_mapping']['桃園縣'] = '其他'
        rejects('changed_state_rejected', lambda: check_state(bad, X, cats))
        wrong_y = y.iloc[::-1]
        rejects('fit_target_series_index_misalignment_rejected', lambda: fit_model(X, wrong_y, geo, cats))
        ok('inputs_unchanged_after_fake_fit_predict', X.equals(before_X) and geo.equals(before_geo))
    finally:
        globals()['CatBoostRegressor'], globals()['Pool'] = original_model, original_pool
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks),
            'real_data_loaded': False, 'real_model_training_performed': False,
            'files_written': False, 'source_sha256': source_hashes(),
            'mapping_evidence_sha256': evidence_hashes()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check-only', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False, indent=2))
