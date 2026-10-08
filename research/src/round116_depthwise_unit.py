"""R116: fixed R79 unit/date-area CatBoost with implicit SymmetricTree to Depthwise.

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
COMPONENT = 'geography_recency_depthwise_unit_catboost'
IMPLEMENTATION_PARENT = {'path': 'src/round112_ordered_unit.py',
                         'sha256': 'c8b6b64577b57c3b5c4e5ff8bde0accfe72866d1e5b6d4338746dd6ebaff3b73'}
NATIVE_RUNTIME_ONLY_PARAMETER_NAMES = ['allow_writing_files', 'thread_count', 'verbose']
TARGET_MODE = original.TARGET_MODE
TARGET_FORMULA = original.TARGET_FORMULA
INVERSE_FORMULA = original.INVERSE_FORMULA
WEIGHT_FORMULA = baseline.WEIGHT_FORMULA
RELOAD_FLOAT_RTOL = 1e-7
RELOAD_FLOAT_ATOL = 0.


def parameters():
    """Deep-copy R79: add grow_policy=Depthwise; no other passed parameter changes."""
    prior = baseline.parameters()
    base.need(prior['max_ctr_complexity'] == 2 and prior['depth'] == 8
              and prior['iterations'] == 2999 and prior['random_seed'] == 42
              and prior['boosting_type'] == 'Plain' and prior['task_type'] == 'GPU',
              'Expected original fixed R79 Plain/GPU recipe')
    base.need('ctr_target_border_count' not in prior
              and not any(k in prior for k in ('simple_ctr', 'combinations_ctr', 'per_feature_ctr')),
              'Baseline must have implicit target borders and no per-CTR override')
    result = original._copy_json(prior)
    base.need(not any(k in prior for k in ('grow_policy', 'min_data_in_leaf', 'max_leaves')),
              'Original R79 must retain implicit growth and leaf defaults')
    result['grow_policy'] = 'Depthwise'
    base.need(set(result) - set(prior) == {'grow_policy'}
              and not (set(prior) - set(result))
              and all(result[k] == prior[k] for k in prior),
              'Only grow_policy=Depthwise may be added to original R79')
    return result


def source_hashes():
    return {**baseline.source_hashes(),
            'src/round116_depthwise_unit.py': original.sha256(Path(__file__))}


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
        'R116 wrapper content changed')
    base.need(state['component'] == COMPONENT and state['parameters'] == parameters()
              and state['baseline_parameters'] == baseline.parameters()
              and state['changed_parameter_names'] == []
              and state['added_parameter_names'] == ['grow_policy']
              and state['input_categorical_feature_count'] == len(cats) == 21
              and state['source_sha256'] == source_hashes(), 'R116 recipe/source differs')
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
    """Check all recoverable passed parameters and native Depthwise/GPU capacity.

    CatBoost does not retain three runtime controls in get_all_params after CBM
    loading. Their unchanged constructor values are recorded, not misrepresented
    as independently recovered model properties. Other passed parameters must be
    native and match, with only documented float serialization tolerance.
    """
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}),
        'R116 wrapper state content changed')
    base.need(state['parameters'] == parameters()
              and state['baseline_parameters'] == baseline.parameters()
              and state['changed_parameter_names'] == []
              and state['added_parameter_names'] == ['grow_policy']
              and state['input_categorical_feature_count'] == 21,
              'Native model state differs from fixed R116 recipe')
    base.need(not isinstance(model.tree_count_, bool)
              and isinstance(model.tree_count_, (int, np.integer))
              and model.tree_count_ == 2999, 'Expected 2999 integer native trees')
    names = state['feature_columns']
    base.need(len(names) == 95 and list(model.feature_names_) == names,
              'Native 95-feature schema differs')
    indices = np.asarray(model.get_cat_feature_indices())
    expected = sorted(names.index(k) for k in state['categorical_columns'])
    base.need(len(expected) == len(set(expected)) == 23 and indices.ndim == 1
              and indices.dtype.kind in 'iu' and indices.tolist() == expected,
              'Native 23-category positions differ')
    actual = model.get_all_params()
    config = parameters()
    checked = []
    for key, value in config.items():
        if key in NATIVE_RUNTIME_ONLY_PARAMETER_NAMES:
            continue
        base.need(key in actual, 'Native parameter missing: ' + key)
        observed = actual[key]
        if isinstance(value, float):
            base.need(not isinstance(observed, bool) and isinstance(observed, (int, float))
                      and np.isfinite(float(observed))
                      and np.isclose(float(observed), value, rtol=RELOAD_FLOAT_RTOL,
                                     atol=RELOAD_FLOAT_ATOL),
                      'Native R116 float parameter differs: ' + key)
        else:
            base.need(type(observed) is type(value) and observed == value,
                      'Native R116 fixed parameter differs: ' + key)
        checked.append(key)
    base.need(actual.get('grow_policy') == 'Depthwise'
              and actual.get('boosting_type') == 'Plain'
              and type(actual.get('ctr_target_border_count')) is int
              and actual['ctr_target_border_count'] == 1
              and type(actual.get('min_data_in_leaf')) is int
              and actual['min_data_in_leaf'] == 1
              and actual.get('use_best_model') is False,
              'Native Depthwise/Plain, implicit target border/min leaf or no-eval policy differs')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.dtype.kind in 'iu' and leaves.shape == (2999,)
              and (leaves > 0).all() and (leaves <= 256).all(),
              'Native depth-8 nonsymmetric leaf counts must be 2999 positive integers at most 256')
    leaf_values, leaf_frequencies = np.unique(leaves, return_counts=True)
    leaf_distribution = {str(int(n)): int(count) for n, count in zip(leaf_values, leaf_frequencies)}
    return {'actual_model_depth': int(actual['depth']), 'actual_trees': int(model.tree_count_),
            'actual_boosting_type': actual['boosting_type'], 'actual_task_type': actual['task_type'],
            'actual_max_ctr_complexity': int(actual['max_ctr_complexity']),
            'actual_ctr_target_border_count': int(actual['ctr_target_border_count']),
            'actual_learning_rate': float(actual['learning_rate']),
            'actual_loss_function': actual['loss_function'], 'actual_seed': int(actual['random_seed']),
            'actual_grow_policy': actual['grow_policy'],
            'actual_min_data_in_leaf': int(actual['min_data_in_leaf']),
            'minimum_tree_leaf_count': int(leaves.min()),
            'maximum_tree_leaf_count': int(leaves.max()),
            'tree_leaf_counts_count': int(len(leaves)),
            'tree_leaf_counts_distribution': leaf_distribution,
            'tree_leaf_counts_sha256_definition': 'SHA256 of C-order little-endian int64 per-tree leaf counts',
            'nonsymmetric_tree_leaf_bounds_checked': True,
            'power_of_two_leaf_counts_required': False,
            'tree_leaf_counts_sha256': hashlib.sha256(leaves.astype('<i8').tobytes()).hexdigest(),
            'native_feature_count': 95, 'native_categorical_feature_count': 23,
            'native_checked_passed_parameter_names': sorted(checked),
            'native_derived_parameter_names': sorted(set(actual) - set(config)),
            'native_derived_parameter_values_retained_in_native_parameters': True,
            'runtime_parameters_not_recovered_from_native_model': {
                key: config[key] for key in NATIVE_RUNTIME_ONLY_PARAMETER_NAMES},
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


def _fit_predict_once(X, y, G, Xq, Gq, cats, weights32, out):
    """Fit once with fit labels only; save and reload every query prediction."""
    tick = time.monotonic()
    out = Path(out)
    base.need(not out.exists(), 'Refuse to overwrite R116 output')
    original._check_X(X)
    original._check_X(Xq)
    base.need(len(X) > 0 and len(Xq) > 0, 'Fit/query must be nonempty')
    base.need(len(cats) == len(set(cats)) == 21, 'Expected 21 unique input categorical columns')
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
             'changed_parameter_names': [], 'added_parameter_names': ['grow_policy'],
             'input_categorical_feature_count': 21, 'source_sha256': sources,
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
        'changed_parameter_names': [],
        'added_parameter_names': ['grow_policy'],
        'baseline_grow_policy_implicit': 'SymmetricTree',
        'min_data_in_leaf_implicit': True,
        'native_derived_defaults_may_differ_from_SymmetricTree': True,
        'baseline_global_target_border_count_implicit': True,
        'implementation_parent': IMPLEMENTATION_PARENT,
        'target_mode': TARGET_MODE, 'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA, 'prediction_units': 'TWD',
        'fit_rows': len(X), 'query_rows': len(Xq), 'new_native_fits': 1,
        'feature_count': 95, 'input_categorical_feature_count': 21,
        'categorical_feature_count': 23,
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
        'inverse_applications_per_prediction': 1,
        'nonnegative_TWD_projection': 'original inverse_transform, after area times 1e4 reconstruction',
        'automatic_fallback_used': False, 'parameters_retuned_after_failure': False,
        'fit_seconds': fit_seconds, 'elapsed_seconds': time.monotonic() - tick,
        'model_sha256': original.sha256(model_path),
        'feature_state_sha256': original.sha256(out / 'feature_state.json'),
        'raw_prediction_sha256': original.sha256(out / 'raw.npy'),
        'prediction_sha256': original.sha256(out / 'native.npy')}
    _write(out / 'metadata.json', metadata)
    return native, metadata


def fit_predict(X, y, G, Xq, Gq, cats, weights32, out):
    """Fit once or preserve failure; never retry, retune, or switch devices.

    A fresh output directory's started/partial artifacts are retained if native
    Depthwise is unsupported or validation fails. No completed metadata is
    fabricated. An already existing directory is never touched, even on error.
    """
    out = Path(out)
    existed_before = out.exists()
    try:
        return _fit_predict_once(X, y, G, Xq, Gq, cats, weights32, out)
    except Exception as exc:
        if not existed_before and out.is_dir() and not (out / 'failed.json').exists():
            failure = {'status': 'failed', 'component': COMPONENT,
                       'parameters': parameters(), 'source_sha256': source_hashes(),
                       'exception_type': type(exc).__name__, 'exception_message': str(exc),
                       'automatic_retry_used': False, 'automatic_fallback_used': False,
                       'parameters_retuned_after_failure': False,
                       'query_labels_available': False, 'new_CSV': False,
                       'fit_completion_not_inferred_from_failure': True,
                       'partial_artifacts_preserved': True}
            try:
                _write(out / 'failed.json', failure)
            except Exception:
                # Preserve the original training/validation exception and all
                # existing evidence even if a failure receipt cannot be written.
                pass
        raise


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
    ok('only_grow_policy_added_all_original_parameters_unchanged',
       set(config) - set(prior) == {'grow_policy'} and set(prior) <= set(config)
       and all(config[k] == prior[k] for k in prior)
       and config['grow_policy'] == 'Depthwise'
       and config['boosting_type'] == prior['boosting_type'] == 'Plain'
       and config['task_type'] == 'GPU' and config['max_ctr_complexity'] == 2
       and not any(k in config for k in ('ctr_target_border_count', 'min_data_in_leaf', 'max_leaves')))
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
        fit_calls = 0
        fit_exception = None
        reload_learning_rate = .06499999762

        def __init__(self, **kwargs):
            self.config = dict(kwargs)
            if 'learning_rate' in self.config:
                self.config['learning_rate'] = .06499999761581421
            self.tree_count_ = 2999

        def fit(self, pool, **kwargs):
            FakeModel.fit_calls += 1
            if self.fit_exception is not None:
                raise self.fit_exception
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
            return {**{k:v for k,v in self.config.items() if k not in NATIVE_RUNTIME_ONLY_PARAMETER_NAMES},
                    'grow_policy': self.config['grow_policy'], 'min_data_in_leaf': 1,
                    'ctr_target_border_count': 1, 'use_best_model': False,
                    'simple_ctr': ['Borders:TargetBorderCount=1'],
                    'combinations_ctr': ['Borders:TargetBorderCount=1']}

        def get_cat_feature_indices(self):
            return list(self.cats)

        def get_tree_leaf_counts(self):
            return np.resize(np.array([1, 3, 17, 128, 256], dtype=np.int64), 2999)

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

    with tempfile.TemporaryDirectory(prefix='r116_synthetic_') as directory:
        out = Path(directory) / 'model'
        with patch(__name__ + '.Pool', FakePool), patch(__name__ + '.CatBoostRegressor', FakeModel):
            native, meta = fit_predict(X, y, G, Xq, Gq, cats, w, out)
            ok('one_fake_fit_and_no_true_fit', FakeModel.fits == 1)
            ok('expected_TWD_inverse', np.array_equal(native,
               np.maximum((np.arange(4) - .5) * area[:4] * 1e4, 0)))
            ok('raw_and_TWD_full_query_reload', meta['raw_and_TWD_reload_exact'] and meta['reload_rows'] == 4)
            ok('native_Depthwise_Plain_GPU_original1_complexity2_checked',
               meta['actual_grow_policy'] == 'Depthwise'
               and meta['actual_min_data_in_leaf'] == 1
               and meta['actual_boosting_type'] == 'Plain' and meta['actual_task_type'] == 'GPU'
               and meta['actual_max_ctr_complexity'] == 2
               and meta['actual_ctr_target_border_count'] == 1 and meta['actual_trees'] == 2999)
            ok('all_effective_passed_native_parameters_checked',
               set(meta['native_checked_passed_parameter_names'])
               == set(config)-set(NATIVE_RUNTIME_ONLY_PARAMETER_NAMES))
            differences = meta['reload_diagnostics_comparison']['serialization_float_differences']
            ok('native_LR_subsample_serialization_rounding_recorded',
               {item['field'] for item in differences} == {
               'actual_learning_rate', 'native_parameters.learning_rate', 'native_parameters.subsample'}
               and meta['fit_native_diagnostics']['actual_learning_rate']
               != meta['reloaded_native_diagnostics']['actual_learning_rate'])
            state = json.loads((out / 'feature_state.json').read_text(encoding='utf8'))
            ok('legacy_params_explicit_provenance', state['parameters']['boosting_type'] == 'Plain'
               and state['parameters']['grow_policy'] == 'Depthwise'
               and state['changed_parameter_names'] == []
               and state['added_parameter_names'] == ['grow_policy']
               and state['input_categorical_feature_count'] == 21
               and state['parameters']['max_ctr_complexity'] == 2
               and state['legacy_feature_state']['parameters']['boosting_type'] == 'Plain'
               and 'ctr_target_border_count' not in state['parameters']
               and state['legacy_state_parameters_are_provenance_not_training'])
            fake = FakeModel()
            fake.load_model(out / 'model.cbm')
            for field, fixed_value in config.items():
                if field in NATIVE_RUNTIME_ONLY_PARAMETER_NAMES:
                    continue
                good = fake.config[field]
                if isinstance(fixed_value, str):
                    bad_value = fixed_value + '_INVALID_SYNTHETIC'
                elif isinstance(fixed_value, float):
                    bad_value = fixed_value + .1
                elif isinstance(fixed_value, int) and not isinstance(fixed_value, bool):
                    bad_value = fixed_value + 1
                else:
                    raise AssertionError('Unsupported synthetic passed parameter type')
                fake.config[field] = bad_value
                reject('native_passed_wrong_' + field + '_rejected', lambda: check_model(fake, state))
                fake.config[field] = good
            fake.tree_count_ = 2998
            reject('native_wrong_tree_count_rejected', lambda: check_model(fake, state))
            fake.tree_count_ = 2999
            original_get_all = fake.get_all_params
            for field, bad_value in [('ctr_target_border_count',8), ('grow_policy','Lossguide'),
                                     ('min_data_in_leaf',30), ('min_data_in_leaf',True),
                                     ('use_best_model',True)]:
                fake.get_all_params = lambda field=field,bad_value=bad_value: {
                    **original_get_all(),field:bad_value}
                reject('native_implicit_wrong_' + field + '_' + type(bad_value).__name__ + '_rejected',
                       lambda: check_model(fake, state))
            for field in config:
                if field in NATIVE_RUNTIME_ONLY_PARAMETER_NAMES:
                    continue
                fake.get_all_params = lambda field=field: {
                    key: value for key, value in original_get_all().items() if key != field}
                reject('native_missing_' + field + '_rejected', lambda: check_model(fake, state))
            fake.get_all_params = original_get_all
            valid_leaves = fake.get_tree_leaf_counts
            ok('nonsymmetric_non_power_of_two_leaf_counts_accepted_and_hashed',
               meta['minimum_tree_leaf_count'] == 1 and meta['maximum_tree_leaf_count'] == 256
               and meta['tree_leaf_counts_count'] == 2999
               and sum(meta['tree_leaf_counts_distribution'].values()) == 2999
               and meta['tree_leaf_counts_distribution'].get('3', 0) > 0
               and meta['tree_leaf_counts_sha256'] == hashlib.sha256(
                   valid_leaves().astype('<i8').tobytes(order='C')).hexdigest()
               and meta['power_of_two_leaf_counts_required'] is False)
            for name, bad_leaves in [
                    ('zero', np.zeros(2999, dtype=np.int64)),
                    ('negative', np.full(2999, -1, dtype=np.int64)),
                    ('above_depth8', np.full(2999, 257, dtype=np.int64)),
                    ('float_dtype', np.full(2999, 256., dtype=np.float64)),
                    ('boolean_dtype', np.ones(2999, dtype=bool)),
                    ('wrong_length', np.ones(2998, dtype=np.int64)),
                    ('wrong_rank', np.ones((2999, 1), dtype=np.int64))]:
                fake.get_tree_leaf_counts = lambda bad_leaves=bad_leaves: bad_leaves
                reject('native_leaf_counts_' + name + '_rejected', lambda: check_model(fake, state))
            fake.get_tree_leaf_counts = valid_leaves
            ok('derived_native_defaults_fully_retained_runtime_three_explicit',
               meta['native_parameters']['min_data_in_leaf'] == 1
               and 'min_data_in_leaf' in meta['native_derived_parameter_names']
               and set(meta['runtime_parameters_not_recovered_from_native_model'])
                   == set(NATIVE_RUNTIME_ONLY_PARAMETER_NAMES)
               and meta['native_derived_parameter_values_retained_in_native_parameters'] is True)
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
            failed_reload = json.loads((Path(directory) / 'bad_reload' / 'failed.json').read_text(encoding='utf8'))
            ok('validation_failure_preserved_no_completed_metadata',
               failed_reload['status'] == 'failed'
               and not (Path(directory) / 'bad_reload' / 'metadata.json').exists()
               and not failed_reload['automatic_retry_used']
               and not failed_reload['parameters_retuned_after_failure'])
            FakeModel.fit_exception = ValueError('Synthetic unsupported GPU Depthwise configuration')
            prior_calls = FakeModel.fit_calls
            unsupported_out = Path(directory) / 'unsupported'
            reject('unsupported_native_configuration_raises_without_fallback', lambda:
                   fit_predict(X, y, G, Xq, Gq, cats, w, unsupported_out))
            FakeModel.fit_exception = None
            failed_fit = json.loads((unsupported_out / 'failed.json').read_text(encoding='utf8'))
            ok('unsupported_failure_preserved_once_without_retuning_or_CPU_fallback',
               FakeModel.fit_calls == prior_calls + 1
               and failed_fit['status'] == 'failed'
               and failed_fit['parameters'] == config
               and not failed_fit['automatic_retry_used'] and not failed_fit['automatic_fallback_used']
               and failed_fit['partial_artifacts_preserved']
               and (unsupported_out / 'started.json').is_file()
               and not (unsupported_out / 'metadata.json').exists())
            protected_before = {p.name: original.sha256(p) for p in out.iterdir()}
            reject('existing_output_rejected', lambda: fit_predict(X, y, G, Xq, Gq, cats, w, out))
            ok('refused_existing_output_preserves_all_bytes_and_adds_no_failure',
               protected_before == {p.name: original.sha256(p) for p in out.iterdir()}
               and not (out / 'failed.json').exists())
            reject('input21_vs_augmented23_contract_rejects_20', lambda:
                   fit_predict(X, y, G, Xq, Gq, cats[:-1], w, Path(directory) / 'bad_cats'))
            ok('saved_artifacts_exist', all((out / name).is_file() for name in [
               'started.json', 'model.cbm', 'feature_state.json', 'raw.npy', 'native.npy', 'metadata.json']))
            ok('no_query_labels_earlystop_or_corrections', not meta['query_labels_available']
               and not meta['fit_uses_eval_set'] and not meta['early_stopping_used']
               and not meta['reference_or_trend_applied_here'] and not meta['spatial_applied_here'])
    ok('all_input_frames_unchanged', before == {name: base.frame_sha(frame)
       for name, frame in [('X', X), ('G', G), ('Xq', Xq), ('Gq', Gq)]})
    ok('original_R79_configuration_unchanged', baseline.parameters() == prior)
    ok('synthetic_check_names_unique', len(checks) == len(set(checks)))
    return {'status': 'passed', 'synthetic_checks': checks, 'check_count': len(checks),
            'true_model_fits': 0, 'fake_model_fits': FakeModel.fits,
            'fake_model_fit_calls': FakeModel.fit_calls, 'GPU_work': False,
            'training_data_loaded': False, 'query_prices_loaded': False,
            'source_sha256': source_hashes(), 'parameters': parameters(),
            'implementation_parent': IMPLEMENTATION_PARENT}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, allow_nan=False))
