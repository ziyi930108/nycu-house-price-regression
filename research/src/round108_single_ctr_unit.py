"""R108: fixed R79 unit CatBoost with max_ctr_complexity 2 -> 1 only.

The caller owns temporal row/label/date-weight provenance. This module never
loads labels, selects parameters, applies R/S, scores, or writes a submission.
The original pure-X feature state remains unchanged inside an explicit wrapper.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round79_recency_unit as baseline

original = baseline.original
base = baseline.base
COMPONENT = 'geography_recency_single_ctr_unit_catboost'
TARGET_MODE = original.TARGET_MODE
TARGET_FORMULA = original.TARGET_FORMULA
INVERSE_FORMULA = original.INVERSE_FORMULA
WEIGHT_FORMULA = baseline.WEIGHT_FORMULA
RELOAD_FLOAT_RTOL = 1e-7
RELOAD_FLOAT_ATOL = 0.


def parameters():
    prior = baseline.parameters()
    base.need(prior['max_ctr_complexity'] == 2 and prior['depth'] == 8
              and prior['iterations'] == 2999 and prior['random_seed'] == 42,
              'Expected original fixed R79 recipe')
    result = original._copy_json(prior)
    result['max_ctr_complexity'] = 1
    base.need(set(result) == set(prior)
              and [k for k in result if result[k] != prior[k]] == ['max_ctr_complexity'],
              'Only max_ctr_complexity may differ from R79')
    return result


def source_hashes():
    return {**baseline.source_hashes(),
            'src/round108_single_ctr_unit.py': original.sha256(Path(__file__))}


def _date_sha(a):
    return hashlib.sha256(np.asarray(a, dtype='<f4').tobytes(order='C')).hexdigest()


def sample_weights(X, weights32):
    dates = baseline._date_values(weights32, X)
    try:
        with np.errstate(over='raise', invalid='raise'):
            value = original.sample_weights(X) * dates.astype(np.float64)
    except FloatingPointError as exc:
        raise ValueError('Area-date Pool weights overflowed') from exc
    base.need(np.isfinite(value).all() and (value > 0).all(),
              'Pool weights must be finite and positive')
    return np.ascontiguousarray(value, dtype=np.float64)


def _check_state(state, X, cats):
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}),
        'R108 wrapper content changed')
    base.need(state['component'] == COMPONENT and state['parameters'] == parameters()
              and state['baseline_parameters'] == baseline.parameters()
              and state['changed_parameter_names'] == ['max_ctr_complexity']
              and state['source_sha256'] == source_hashes(), 'R108 recipe/source differs')
    base.need(state['legacy_state_parameters_are_provenance_not_training'] is True
              and state['legacy_state_training_fields_are_provenance_only'] is True,
              'Legacy feature-state scope not declared')
    legacy = state['legacy_feature_state']
    original.check_state(legacy, X, cats)
    base.need(state['feature_columns'] == legacy['all_feature_names']
              and len(state['feature_columns']) == 95
              and state['categorical_columns'] == legacy['categorical_columns']
              and len(state['categorical_columns']) == 23, 'Original feature schema differs')
    return legacy


def check_model(model, state):
    """Check the fitted native model directly; original checker fixes CTR=2."""
    base.need(int(model.tree_count_) == 2999, 'Expected 2999 native trees')
    names = state['feature_columns']
    base.need(len(names) == 95 and list(model.feature_names_) == names,
              'Native 95-feature schema differs')
    indices = np.asarray(model.get_cat_feature_indices())
    expected = sorted(names.index(k) for k in state['categorical_columns'])
    base.need(len(expected) == len(set(expected)) == 23 and indices.ndim == 1
              and indices.dtype.kind in 'iu' and indices.tolist() == expected,
              'Native 23-category positions differ')
    actual = model.get_all_params()
    fixed = {'depth': 8, 'grow_policy': 'SymmetricTree', 'boosting_type': 'Plain',
             'iterations': 2999, 'random_seed': 42, 'loss_function': 'MAE',
             'eval_metric': 'MAE', 'bootstrap_type': 'Bernoulli',
             'one_hot_max_size': 8, 'max_ctr_complexity': 1, 'border_count': 128}
    base.need(all(actual.get(k) == value for k, value in fixed.items()),
              'Native R108 fixed parameter differs')
    floats = {'learning_rate': .065, 'l2_leaf_reg': 6., 'subsample': .85,
              'random_strength': .5}
    base.need(all(np.isfinite(float(actual[k])) and np.isclose(float(actual[k]), value,
              rtol=1e-7, atol=0) for k, value in floats.items()),
              'Native R108 float parameter differs')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.dtype.kind in 'iu' and leaves.shape == (2999,)
              and (leaves > 0).all() and (leaves <= 256).all(),
              'Native depth-8 leaf counts differ')
    return {'actual_model_depth': int(actual['depth']), 'actual_trees': int(model.tree_count_),
            'actual_max_ctr_complexity': int(actual['max_ctr_complexity']),
            'actual_learning_rate': float(actual['learning_rate']),
            'actual_loss_function': actual['loss_function'], 'actual_seed': int(actual['random_seed']),
            'maximum_tree_leaf_count': int(leaves.max()),
            'tree_leaf_counts_sha256': hashlib.sha256(leaves.astype('<i8').tobytes()).hexdigest(),
            'native_feature_count': 95, 'native_categorical_feature_count': 23,
            'native_parameters': original._copy_json(actual)}


def compare_reload_diagnostics(fitted, reloaded):
    """Permit declared native float serialization rounding, recording every delta.

    Both models must independently pass check_model first. Outside the top-level
    learning rate and finite floats in native_parameters, values remain exact.
    Dictionary keys, list structure and scalar types must also match exactly.
    """
    changes = []

    def compare_native(left, right, field):
        base.need(type(left) is type(right), 'Reloaded native parameter type differs: ' + field)
        if isinstance(left, dict):
            base.need(set(left) == set(right), 'Reloaded native parameter keys differ: ' + field)
            for key in sorted(left):
                compare_native(left[key], right[key], field + '.' + key)
        elif isinstance(left, list):
            base.need(len(left) == len(right), 'Reloaded native parameter length differs: ' + field)
            for i, (a, b) in enumerate(zip(left, right)):
                compare_native(a, b, field + '[' + str(i) + ']')
        elif isinstance(left, float):
            base.need(np.isfinite(left) and np.isfinite(right)
                      and np.isclose(left, right, rtol=RELOAD_FLOAT_RTOL, atol=RELOAD_FLOAT_ATOL),
                      'Reloaded native float differs beyond serialization tolerance: ' + field)
            if left != right:
                changes.append({'field': field, 'fit_value': left, 'reload_value': right})
        else:
            base.need(left == right, 'Reloaded native parameter differs: ' + field)

    base.need(set(fitted) == set(reloaded), 'Reloaded native diagnostic keys differ')
    exact_fields = set(fitted) - {'actual_learning_rate', 'native_parameters'}
    base.need(all(type(fitted[k]) is type(reloaded[k]) and fitted[k] == reloaded[k]
                  for k in exact_fields), 'Reloaded native non-floating diagnostics differ')
    for name, diagnostic in [('fit', fitted), ('reload', reloaded)]:
        rate = diagnostic['actual_learning_rate']
        base.need(rate == diagnostic['native_parameters']['learning_rate']
                  and np.isfinite(rate) and np.isclose(rate, .065,
                  rtol=RELOAD_FLOAT_RTOL, atol=RELOAD_FLOAT_ATOL),
                  name + ' native learning-rate diagnostic differs')
    compare_native(fitted['actual_learning_rate'], reloaded['actual_learning_rate'],
                   'actual_learning_rate')
    compare_native(fitted['native_parameters'], reloaded['native_parameters'], 'native_parameters')
    return {'status': 'passed', 'native_float_rtol': RELOAD_FLOAT_RTOL,
            'native_float_atol': RELOAD_FLOAT_ATOL,
            'scalar_types_keys_structures_and_nonfloat_values_exact': True,
            'other_top_level_diagnostics_exact': True,
            'both_models_independently_checked_against_fixed_recipe': True,
            'serialization_float_differences': changes}


def _write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def _save_array(path, value):
    with Path(path).open('xb') as f:
        np.save(f, value, allow_pickle=False)
    restored = np.load(path, allow_pickle=False)
    _exact(value, restored, 'Saved prediction array reload differs')


def _exact(a, b, message):
    base.need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), message)


def _raw_predict(model, state, X, geo, cats):
    _check_state(state, X, cats)
    check_model(model, state)
    frame = original.augment_features(X, geo)
    base.need(list(frame) == state['feature_columns']
              and {k: str(frame[k].dtype) for k in frame} == state['feature_dtypes'],
              'Query feature schema differs')
    raw = np.asarray(model.predict(Pool(frame, cat_features=state['categorical_columns'],
                         thread_count=4), thread_count=4), dtype=np.float64)
    base.need(raw.shape == (len(X),) and np.isfinite(raw).all(), 'Raw predictions invalid')
    return raw


def fit_predict(X, y, G, Xq, Gq, cats, weights32, out):
    """Fit once with fit labels only; save and reload every query prediction."""
    tick = time.monotonic()
    out = Path(out)
    base.need(not out.exists(), 'Refuse to overwrite R108 output')
    original._check_X(X)
    original._check_X(Xq)
    base.need(len(X) > 0 and len(Xq) > 0, 'Fit/query must be nonempty')
    raw_y = original._aligned_vector(y, X, 'Fit target')
    target = original.transform_target(raw_y, X)
    dates32 = baseline._date_values(weights32, X)
    weights = sample_weights(X, dates32)
    legacy = original.fit_feature_state(X, G, cats)
    fit_frame = original.augment_features(X, G)
    query_frame = original.augment_features(Xq, Gq)
    config = parameters()
    sources = source_hashes()
    frames = {'fit_X': base.frame_sha(X), 'query_X': base.frame_sha(Xq),
              'fit_geo': base.frame_sha(G), 'query_geo': base.frame_sha(Gq),
              'fit_augmented': base.frame_sha(fit_frame), 'query_augmented': base.frame_sha(query_frame)}
    hashes = {'fit_target_values_sha256': original.values_sha256(raw_y),
              'fit_transformed_target_values_sha256': original.values_sha256(target),
              'fit_date_weights_sha256': _date_sha(dates32),
              'fit_area_weights_sha256': original.values_sha256(original.sample_weights(X)),
              'fit_sample_weights_sha256': original.values_sha256(weights)}
    state = {'format_version': 1, 'component': COMPONENT,
             'parameters': original._copy_json(config), 'baseline_parameters': baseline.parameters(),
             'changed_parameter_names': ['max_ctr_complexity'], 'source_sha256': sources,
             'target_mode': TARGET_MODE, 'target_formula': TARGET_FORMULA,
             'inverse_formula': INVERSE_FORMULA, 'sample_weight': WEIGHT_FORMULA,
             'feature_columns': list(fit_frame), 'categorical_columns': list(legacy['categorical_columns']),
             'feature_dtypes': {k: str(fit_frame[k].dtype) for k in fit_frame},
             'legacy_feature_state': legacy,
             'legacy_state_parameters_are_provenance_not_training': True,
             'legacy_state_training_fields_are_provenance_only': True,
             'fit_rows': len(X), 'query_labels_used': False, **hashes}
    state['state_content_sha256'] = base.content_sha(state)
    _check_state(state, X, cats)
    _check_state(state, Xq, cats)
    base.need(list(query_frame) == state['feature_columns']
              and {k: str(query_frame[k].dtype) for k in query_frame} == state['feature_dtypes'],
              'Fit/query augmented schema differs')
    out.mkdir(parents=True, exist_ok=False)
    _write(out / 'started.json', {'component': COMPONENT, 'parameters': config,
        'fit_rows': len(X), 'query_rows': len(Xq), 'query_labels_available': False,
        'source_sha256': sources, 'frame_sha256': frames, **hashes})
    model = CatBoostRegressor(**config)
    fit_tick = time.monotonic()
    model.fit(Pool(fit_frame, target, weight=weights,
                   cat_features=state['categorical_columns'], thread_count=4))
    fit_seconds = time.monotonic() - fit_tick
    actual = check_model(model, state)
    model_path = out / 'model.cbm'
    model.save_model(str(model_path))
    _write(out / 'feature_state.json', state)
    raw = _raw_predict(model, state, Xq, Gq, cats)
    native = original.inverse_transform(raw, Xq)
    restored_model = CatBoostRegressor()
    restored_model.load_model(str(model_path))
    restored_state = json.loads((out / 'feature_state.json').read_text(encoding='utf8'))
    restored_raw = _raw_predict(restored_model, restored_state, Xq, Gq, cats)
    _exact(raw, restored_raw, 'Full-query raw native-model reload differs')
    _exact(native, original.inverse_transform(restored_raw, Xq),
           'Full-query TWD native-model reload differs')
    reloaded_actual = check_model(restored_model, restored_state)
    reload_diagnostics = compare_reload_diagnostics(actual, reloaded_actual)
    _save_array(out / 'raw.npy', raw)
    _save_array(out / 'native.npy', native)
    after = {'fit_X': base.frame_sha(X), 'query_X': base.frame_sha(Xq),
             'fit_geo': base.frame_sha(G), 'query_geo': base.frame_sha(Gq),
             'fit_augmented': base.frame_sha(original.augment_features(X, G)),
             'query_augmented': base.frame_sha(original.augment_features(Xq, Gq))}
    base.need(frames == after and source_hashes() == sources and parameters() == config,
              'Features/source/config changed during fit or predict')
    current = {'fit_target_values_sha256': original.values_sha256(original._aligned_vector(y, X, 'Fit target')),
               'fit_transformed_target_values_sha256': original.values_sha256(target),
               'fit_date_weights_sha256': _date_sha(baseline._date_values(weights32, X)),
               'fit_area_weights_sha256': original.values_sha256(original.sample_weights(X)),
               'fit_sample_weights_sha256': original.values_sha256(sample_weights(X, weights32))}
    base.need(hashes == current and original.values_sha256(weights) == hashes['fit_sample_weights_sha256'],
              'Fit labels or weights changed')
    metadata = {'status': 'completed', 'family': 'catboost', 'component': COMPONENT,
        'parameters': config, 'baseline_parameters': baseline.parameters(),
        'changed_parameter_names': ['max_ctr_complexity'],
        'target_mode': TARGET_MODE, 'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA, 'prediction_units': 'TWD',
        'fit_rows': len(X), 'query_rows': len(Xq), 'new_native_fits': 1,
        'feature_count': 95, 'categorical_feature_count': 23,
        'feature_columns': state['feature_columns'], 'categorical_columns': state['categorical_columns'],
        'source_sha256': sources, 'frame_sha256': frames, **hashes, **actual,
        'fit_native_diagnostics': actual, 'reloaded_native_diagnostics': reloaded_actual,
        'reload_diagnostics_comparison': reload_diagnostics,
        'sample_weight': WEIGHT_FORMULA, 'weight_dtype': 'float64', 'weight_argument_passed_to_pool': True,
        'date_weights_recomputed': False, 'date_weights_renormalized': False,
        'weights_normalized_to_mean_one': False, 'weights_use_fit_target': False, 'weights_use_query': False,
        'date_weight_provenance_verified_in_helper': False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks': True,
        'fit_date_weights_summary': baseline._summary(dates32),
        'fit_sample_weights_summary': baseline._summary(weights),
        'legacy_state_parameters_are_provenance_not_training': True,
        'legacy_state_training_fields_are_provenance_only': True,
        'legacy_state_weight_policy': original.WEIGHT_FORMULA,
        'query_labels_available': False, 'query_labels_used': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'saved_model_reload_exact': True, 'raw_and_TWD_reload_exact': True, 'reload_rows': len(Xq),
        'reference_or_trend_applied_here': False, 'spatial_applied_here': False,
        'extra_price_cap_applied': False, 'new_CSV': False,
        'fit_seconds': fit_seconds, 'elapsed_seconds': time.monotonic() - tick,
        'model_sha256': original.sha256(model_path),
        'feature_state_sha256': original.sha256(out / 'feature_state.json'),
        'raw_prediction_sha256': original.sha256(out / 'raw.npy'),
        'prediction_sha256': original.sha256(out / 'native.npy')}
    _write(out / 'metadata.json', metadata)
    return native, metadata


def self_check():
    """Synthetic API/serialization checks; FakeModel only, no CUDA calls."""
    import tempfile
    from unittest.mock import patch
    checks = []

    def ok(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    def reject(name, function):
        try:
            function()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name + ' did not reject')

    index = pd.Index([9, 41, 7, 23, 8, 102, 6, 55], name='synthetic_row')
    cats = ['category_' + str(i) for i in range(21)]
    X = pd.DataFrame({c: pd.Series(['category'] * len(index), index=index,
                                  dtype=object) for c in cats}, index=index)
    X['building_area'] = np.array([100, np.nan, 0, -1, np.inf, 50, 0, 0], dtype=np.float32)
    X['parking_area'] = np.array([30, 20, 0, -1, 5, 70, np.nan, np.inf], dtype=np.float32)
    for i in range(64):
        X['numeric_' + str(i)] = np.arange(len(index), dtype=np.float32)
    G = pd.DataFrame({c: np.full(len(index), np.nan, dtype=np.float32)
                      for c in base.NUMERIC}, index=index)
    G['geo_has_position'] = np.zeros(len(index), dtype=np.float32)
    G['geo_match_status'] = pd.Series(['ineligible_address'] * len(index), index=index, dtype=object)
    G['geo_grid_002'] = pd.Series(['__MISSING__'] * len(index), index=index, dtype=object)
    G = G[base.FEATURES]
    y = pd.Series([1e6, 2e6, 3e6, 0, 4e6, 5e6, 6e6, 7e6], index=index, dtype=np.float64)
    w = pd.Series([.125, .25, .5, 1., 2., 4., 1.5, .75], index=index, dtype=np.float32)
    area = np.array([100, 20, 1, 1, 5, 50, 1, 1], dtype=np.float64)
    expected = area / 100 * w.to_numpy(dtype=np.float64)
    Xq, Gq = X.iloc[:4].copy(), G.iloc[:4].copy()
    before = {name: base.frame_sha(frame) for name, frame in [('X', X), ('G', G), ('Xq', Xq), ('Gq', Gq)]}
    prior, config = baseline.parameters(), parameters()
    ok('only_max_ctr_complexity_2_to_1', set(config) == set(prior)
       and [k for k in config if config[k] != prior[k]] == ['max_ctr_complexity']
       and config['max_ctr_complexity'] == 1)
    ok('exact_original_area_date_weights', np.array_equal(sample_weights(X, w), expected))
    reject('wrong_date_dtype', lambda: sample_weights(X, w.astype(np.float64)))
    reject('misordered_date_index', lambda: sample_weights(X, w.iloc[::-1]))
    reject('nonpositive_date_weight', lambda: sample_weights(X, np.zeros(len(X), dtype=np.float32)))
    reject('negative_fit_price', lambda: original.transform_target(-np.ones(len(X)), X))

    class FakePool:
        def __init__(self, data, label=None, **kwargs):
            self.data, self.label, self.kwargs = data, label, kwargs

    class FakeModel:
        fits = 0
        reload_learning_rate = .06499999762

        def __init__(self, **kwargs):
            self.config = dict(kwargs)
            if 'learning_rate' in self.config:
                self.config['learning_rate'] = .06499999761581421
            self.tree_count_ = 2999

        def fit(self, pool, **kwargs):
            if kwargs or pool.label is None:
                raise AssertionError('Unexpected eval_set/early stopping or missing fit label')
            if not np.array_equal(pool.label, y.to_numpy() / (area * 1e4)):
                raise AssertionError('Wrong fit unit label')
            if not np.array_equal(pool.kwargs['weight'], expected):
                raise AssertionError('Wrong Pool weight')
            self.names = list(pool.data)
            self.cats = sorted(self.names.index(c) for c in pool.kwargs['cat_features'])
            FakeModel.fits += 1
            return self

        @property
        def feature_names_(self):
            return self.names

        def get_all_params(self):
            return {**self.config, 'grow_policy': 'SymmetricTree'}

        def get_cat_feature_indices(self):
            return list(self.cats)

        def get_tree_leaf_counts(self):
            return np.full(2999, 256, dtype=np.int64)

        def predict(self, pool, **kwargs):
            if pool.label is not None or 'weight' in pool.kwargs:
                raise AssertionError('Query Pool contains label or fit weights')
            return pool.data['numeric_0'].to_numpy(dtype=np.float64) - .5

        def save_model(self, path):
            _write(path, {'config': self.config, 'names': self.names, 'cats': self.cats})

        def load_model(self, path):
            doc = json.loads(Path(path).read_text(encoding='utf8'))
            self.config, self.names, self.cats = doc['config'], doc['names'], doc['cats']
            self.config['learning_rate'] = self.reload_learning_rate
            self.config['subsample'] = .85000000001

    with tempfile.TemporaryDirectory(prefix='r108_synthetic_') as directory:
        out = Path(directory) / 'model'
        with patch(__name__ + '.Pool', FakePool), patch(__name__ + '.CatBoostRegressor', FakeModel):
            native, meta = fit_predict(X, y, G, Xq, Gq, cats, w, out)
            ok('one_fake_fit_and_no_true_fit', FakeModel.fits == 1)
            ok('expected_TWD_inverse', np.array_equal(native,
               np.maximum((np.arange(4) - .5) * area[:4] * 1e4, 0)))
            ok('raw_and_TWD_full_query_reload', meta['raw_and_TWD_reload_exact'] and meta['reload_rows'] == 4)
            ok('actual_fitted_ctr_checked', meta['actual_max_ctr_complexity'] == 1 and meta['actual_trees'] == 2999)
            differences = meta['reload_diagnostics_comparison']['serialization_float_differences']
            ok('native_LR_subsample_serialization_rounding_recorded',
               {item['field'] for item in differences} == {
               'actual_learning_rate', 'native_parameters.learning_rate', 'native_parameters.subsample'}
               and meta['fit_native_diagnostics']['actual_learning_rate']
               != meta['reloaded_native_diagnostics']['actual_learning_rate'])
            state = json.loads((out / 'feature_state.json').read_text(encoding='utf8'))
            ok('legacy_params_explicit_provenance', state['parameters']['max_ctr_complexity'] == 1
               and state['legacy_feature_state']['parameters']['max_ctr_complexity'] == 2
               and state['legacy_state_parameters_are_provenance_not_training'])
            fake = FakeModel()
            fake.load_model(out / 'model.cbm')
            fake.config['max_ctr_complexity'] = 2
            reject('native_wrong_ctr_rejected', lambda: check_model(fake, state))
            fake.config['max_ctr_complexity'] = 1
            fake.config['random_seed'] = 137
            reject('native_wrong_seed_rejected', lambda: check_model(fake, state))
            fake.config['random_seed'] = 42
            fake.tree_count_ = 2998
            reject('native_wrong_tree_count_rejected', lambda: check_model(fake, state))
            bad = original._copy_json(meta['reloaded_native_diagnostics'])
            bad['native_parameters']['subsample'] = .86
            reject('native_float_rounding_beyond_tolerance_rejected', lambda:
                   compare_reload_diagnostics(meta['fit_native_diagnostics'], bad))
            bad = original._copy_json(meta['reloaded_native_diagnostics'])
            bad['native_parameters']['random_seed'] = 137
            reject('nonfloat_native_diagnostic_difference_rejected', lambda:
                   compare_reload_diagnostics(meta['fit_native_diagnostics'], bad))
            bad = original._copy_json(meta['reloaded_native_diagnostics'])
            bad['native_parameters']['subsample'] = float('nan')
            reject('nonfinite_native_float_rejected', lambda:
                   compare_reload_diagnostics(meta['fit_native_diagnostics'], bad))
            bad = original._copy_json(meta['reloaded_native_diagnostics'])
            bad['native_parameters']['depth'] = 8.
            reject('native_scalar_type_change_rejected', lambda:
                   compare_reload_diagnostics(meta['fit_native_diagnostics'], bad))
            FakeModel.reload_learning_rate = .066
            reject('FakeModel_abnormal_reload_LR_rejected', lambda:
                   fit_predict(X, y, G, Xq, Gq, cats, w, Path(directory) / 'bad_reload'))
            FakeModel.reload_learning_rate = .06499999762
            reject('existing_output_rejected', lambda: fit_predict(X, y, G, Xq, Gq, cats, w, out))
            ok('saved_artifacts_exist', all((out / name).is_file() for name in [
               'started.json', 'model.cbm', 'feature_state.json', 'raw.npy', 'native.npy', 'metadata.json']))
            ok('no_query_labels_earlystop_or_corrections', not meta['query_labels_available']
               and not meta['fit_uses_eval_set'] and not meta['early_stopping_used']
               and not meta['reference_or_trend_applied_here'] and not meta['spatial_applied_here'])
    ok('all_input_frames_unchanged', before == {name: base.frame_sha(frame)
       for name, frame in [('X', X), ('G', G), ('Xq', Xq), ('Gq', Gq)]})
    return {'status': 'passed', 'synthetic_checks': checks, 'check_count': len(checks),
            'true_model_fits': 0, 'fake_model_fits': FakeModel.fits, 'GPU_work': False,
            'source_sha256': source_hashes(), 'parameters': parameters()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, allow_nan=False))
