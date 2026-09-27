"""R17 independent price replay. No fit, tuning, or additional window scoring.

Independent raw CatBoost replay, native Ordered/depth/leaf capacity, errors, grouping and
fixed gates. The driver supplies source/receipt/provenance/partition integrity;
read_result adds a saved-gate integrity check but does not calculate our decision.
Frozen R6 augmentation and reference are explicitly reused. The R17 helper is not
called to construct state, predict, invert, score or check native model capacity.
Only the explicitly named saved stage is audited; no fitting is performed.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round17'
CONTROL = 'v3_control'
CANDIDATE = 'ordered_geography_blend'
EXPECTED_HELPER_SHA256 = 'a04d0ef7b3765777dd411f5b9070143603f19a32e2986fd02efbf47abe3b834f'
HELPER_SOURCES = ('src/round17_ordered_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')

GROUPS = ('seen', 'new_nonmissing', 'missing')
QUARTERS = {'Q1': [0,91], 'Q2': [92,183], 'Q3': [184,275], 'Q4': [276,None]}
THRESHOLDS = {
    'development_min_overall_improvement': .01, 'development_min_weighted_improvement': .01,
    'development_max_quarter_deterioration': .05, 'large_stratum_min_rows': 500,
    'max_large_stratum_deterioration': .05, 'confirmation_max_overall_deterioration': .0,
    'confirmation_max_weighted_deterioration': .0, 'confirmation_min_improved_quarters': 2,
    'auxiliary_max_overall_deterioration': .03, 'auxiliary_max_weighted_deterioration': .03,
    'auxiliary_quarters_diagnostic_only': True,
}
SPLITS = {'development': ('2022-08-11','2023-08-10'),
          'confirmation': ('2023-08-11','2024-08-10'),
          'auxiliary': ('2021-08-11','2022-08-10')}
COUNTS = {'development': (240316,81234), 'confirmation': (321550,133697), 'auxiliary': (134805,105511)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def norm(values):
    values = values.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
    return values.mask(values.isin(['', 'nan', 'None', '<NA>', 'NaT']), '__MISSING__')

def labels(fit, query):
    def keys(frame):
        project = norm(frame['project_name'])
        return norm(frame['address_city']) + '|' + norm(frame['address_district']) + '|' + project, project.eq('__MISSING__')
    fit_key, fit_missing = keys(fit)
    query_key, query_missing = keys(query)
    known = set(fit_key[~fit_missing])
    return np.where(query_missing, 'missing', np.where(query_key.isin(known), 'seen', 'new_nonmissing'))

def scoring_weights(data):
    columns = ['address_city', 'address_district', 'project_name']
    fit = pd.concat([data['X_train'][columns], data['X_valid'][columns]], ignore_index=True)
    group = labels(fit, data['X_test'])
    return {key: float(np.mean(group == key)) for key in GROUPS}

def independent_score(y, prediction, fit, query, start, weights):
    y, prediction = np.asarray(y, dtype='float64'), np.asarray(prediction, dtype='float64')
    require(y.shape == prediction.shape == (len(query),) and len(y) > 0, 'Unaligned score arrays')
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), 'Nonfinite score arrays')
    require(set(weights) == set(GROUPS) and all(np.isfinite(v) and v >= 0 for v in weights.values()), 'Invalid scoring weights')
    require(abs(sum(weights.values()) - 1.) < 1e-12, 'Scoring weights do not sum to one')
    errors = np.abs(y - prediction)
    group = labels(fit, query)
    horizon = query['trade_day_index'].to_numpy(dtype='float64') - (pd.Timestamp(start) - pd.Timestamp('2000-01-01')).days
    require(np.isfinite(horizon).all() and (horizon >= 0).all(), 'Invalid forecast horizon')
    month = pd.to_datetime(query['trade_day_index'], unit='D', origin='2000-01-01').dt.strftime('%Y-%m').to_numpy()
    city = norm(query['address_city']).to_numpy()
    def part(mask):
        n = int(np.sum(mask))
        return {'rows': n, 'mae': float(errors[mask].mean()) if n else None}
    strata = {key: part(group == key) for key in GROUPS}
    weighted = 0.0
    for key, weight in weights.items():
        if weight > 0:
            require(strata[key]['rows'] > 0, 'Positive scoring weight has no stratum rows')
            weighted += weight * strata[key]['mae']
    quarters = {}
    for name, (low, high) in QUARTERS.items():
        mask = horizon >= low
        if high is not None:
            mask &= horizon <= high
        quarters[name] = {**part(mask), 'horizon_days': [low, high]}
    require(sum(v['rows'] for v in quarters.values()) == len(query), 'Quarter masks do not partition rows')
    return {'rows': len(errors), 'overall_mae': float(errors.mean()), 'weighted_mae': float(weighted),
            'strata': strata, 'forecast_quarters': quarters,
            'months': {str(key): part(month == key) for key in sorted(set(month))},
            'cities': {str(key): part(city == key) for key in sorted(set(city))}}

def independent_blend(old, geographic, unit, reference, query, apply_reference):
    # Preserve the frozen arithmetic order as well as its 37.5/37.5/25 meaning.
    base = .75 * (.5 * np.asarray(old, dtype='float64') + .5 * np.asarray(geographic, dtype='float64')) + .25 * np.asarray(unit, dtype='float64')
    result = np.asarray(apply_reference(reference, query, base, variant='trend'), dtype='float64')
    require(result.shape == (len(query),) and np.isfinite(result).all() and (result >= 0).all(), 'Invalid blended prices')
    return result

def compare_vector(saved, replay, name):
    saved, replay = np.asarray(saved), np.asarray(replay)
    require(saved.dtype == replay.dtype == np.dtype('float64') and saved.ndim == 1
            and saved.shape == replay.shape and len(saved) > 0
            and np.isfinite(saved).all() and np.isfinite(replay).all(), name + ' shape/dtype/finite mismatch')
    require(np.array_equal(saved.view('uint64'), replay.view('uint64')), name + ' bitwise full-query replay mismatch')
    return {'rows': len(saved), 'bitwise_equal': True,
            'maximum_absolute_difference_twd': float(np.max(np.abs(saved - replay))),
            'rtol': 0., 'atol_twd': 0.}

def content_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()

def frame_sha(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()

def validate_category_counts(original_count, augmented_count):
    require(original_count == 21, 'Original 87 features require 21 categorical columns')
    require(augmented_count == 23, 'Augmented 95 features require 23 categorical columns')

def independent_model_view(X, geo, original_categories):
    """Use only immutable R6 augmentation; preserve its exact original 95 view."""
    from round6_geo import augment_features as r6_augment, categories as r6_categories
    require(isinstance(X, pd.DataFrame) and X.columns.is_unique and X.index.is_unique,
            'Independent model view needs unique X columns and indices')
    require(len(X.columns) == 87 and len(set(original_categories)) == 21,
            'Expected unmodified original 87/21 schema')
    require(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
            'IDs or labels are not original model features')
    result = r6_augment(X, geo)
    cats = r6_categories(X, original_categories)
    validate_category_counts(len(original_categories), len(cats))
    require(len(result.columns) == 95 and result.columns.is_unique
            and result.iloc[:, :87].equals(X) and result.iloc[:, 87:].equals(geo)
            and result.iloc[:, :87].dtypes.equals(X.dtypes)
            and result.iloc[:, 87:].dtypes.equals(geo.dtypes)
            and result.index.equals(X.index), 'R6 95 values/dtypes/order/index changed')
    return result, cats

def independent_parameters(configuration):
    """Reproduce the fixed R6 recipe from its immutable v1 JSON, then only boosting_type."""
    matches = [v for v in configuration['components'] if v['family'] == 'catboost']
    require(len(matches) > 0, 'No CatBoost component in fixed v1 configuration')
    baseline = {**copy.deepcopy(matches[0]['parameters']), 'iterations': 2999,
                'thread_count': 4, 'verbose': 300}
    require(baseline['depth'] == 8 and baseline['random_seed'] == 42
            and baseline['loss_function'] == 'MAE' and baseline['boosting_type'] == 'Plain', 'Wrong original R6 raw CatBoost recipe')
    ordered = {**baseline, 'boosting_type': 'Ordered'}
    require(set(baseline) == set(ordered)
            and sorted(k for k in ordered if ordered[k] != baseline[k]) == ['boosting_type'],
            'Only boosting_type may change')
    return baseline, ordered


def independent_state_check(state, X, geo, original_categories, specification,
                            baseline, parameters, configuration_sha):
    from round6_geo import FEATURE_RULES as r6_rules
    require(state['state_content_sha256'] == content_sha({k: v for k, v in state.items()
            if k != 'state_content_sha256'}), 'Saved state content changed')
    require(state['format_version'] == 1 and state['component'] == 'geography_ordered_raw_catboost'
            and state['target_mode'] == 'raw' and state['target_formula'] == 'y_TWD / 1e6'
            and state['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong ordered raw state semantics')
    require(state['source_sha256'] == {n: specification['source_sha256'][n] for n in HELPER_SOURCES},
            'Saved component source map differs')
    require(state['base_configuration_sha256'] == configuration_sha
            and state['baseline_parameters'] == specification['baseline_parameters'] == baseline
            and state['parameters'] == specification['parameters'] == parameters
            and state['changed_parameter_names'] == specification['changed_parameter_names'] == ['boosting_type'],
            'Saved boosting-only recipe differs')
    rules = state['feature_rules']
    require(rules == specification['feature_rules'] and rules['geo_encoder_rules'] == r6_rules
            and rules['original_features_unchanged'] == 87 and rules['existing_geo_features_unchanged'] == 8
            and rules['feature_count'] == 95 and rules['categorical_feature_count'] == 23
            and rules['baseline_depth'] == 8 and rules['depth'] == 8
            and rules['baseline_boosting_type'] == 'Plain' and rules['boosting_type'] == 'Ordered'
            and rules['changed_parameter_names'] == ['boosting_type']
            and rules['required_actual_grow_policy'] == 'SymmetricTree'
            and rules['maximum_allowed_tree_leaf_count'] == 256 and rules['iterations'] == 2999
            and rules['sample_weight'] == 'uniform' and rules['weight_argument_passed_to_pool'] is False,
            'Saved model feature/capacity rules differ')
    view, cats = independent_model_view(X, geo, original_categories)
    geographic = {
        'feature_rules': copy.deepcopy(r6_rules), 'source_sha256': specification['source_sha256']['src/round6_geo.py'],
        'fit_rows': len(X), 'original_columns': list(X), 'original_dtypes': {c: str(X[c].dtype) for c in X},
        'all_feature_names': list(view), 'fit_original_feature_values_sha256': frame_sha(X),
        'fit_geo_feature_values_sha256': frame_sha(geo), 'fit_position_rows': int(geo.geo_has_position.sum()),
        'fit_state_uses_y': False, 'fit_state_uses_query': False,
    }
    geographic['state_content_sha256'] = content_sha(geographic)
    require(state['geo_state'] == geographic, 'Independent fit-only R6 geography state differs')
    expected = {
        'original_categorical_columns': list(original_categories), 'categorical_columns': cats,
        'all_feature_names': list(view), 'fit_rows': len(X),
        'fit_model_feature_values_sha256': frame_sha(view),
        'fit_model_feature_dtypes': {c: str(view[c].dtype) for c in view},
        'fit_state_uses_y': False, 'fit_state_uses_query': False, 'query_labels_used': False,
    }
    for name, value in expected.items():
        require(state[name] == value, 'Independent fit-only state differs: ' + name)
    return {'fit_rows': len(X), 'feature_count': 95, 'categorical_count': 23,
            'fit_state_reconstructed_independently': True,
            'fit_model_feature_values_sha256': frame_sha(view)}


def independent_query_view(state, X, geo, original_categories):
    original = state['geo_state']
    require(list(X) == original['original_columns']
            and {c: str(X[c].dtype) for c in X} == original['original_dtypes']
            and list(original_categories) == state['original_categorical_columns'], 'Query original schema differs')
    view, cats = independent_model_view(X, geo, original_categories)
    require(list(view) == state['all_feature_names'] == original['all_feature_names']
            and cats == state['categorical_columns']
            and {c: str(view[c].dtype) for c in view} == state['fit_model_feature_dtypes'],
            'Query 95/23 schema differs')
    return view, cats


def independent_capacity(model, feature_names):
    """Inspect loaded native CatBoost rather than trusting requested parameters."""
    require(model.tree_count_ == 2999 and list(model.feature_names_) == list(feature_names)
            and len(feature_names) == 95, 'Native CatBoost tree count/feature order differs')
    native = model.get_all_params()
    require(native['depth'] == 8 and native['grow_policy'] == 'SymmetricTree'
            and native['boosting_type'] == 'Ordered',
            'Native CatBoost depth/grow/boosting policy differs')
    leaves = np.asarray(model.get_tree_leaf_counts())
    require(leaves.dtype.kind in 'iu' and leaves.shape == (2999,)
            and (leaves > 0).all() and (leaves <= 256).all(), 'Invalid native tree leaf capacities')
    return {'actual_model_depth': int(native['depth']), 'actual_grow_policy': native['grow_policy'],
            'actual_boosting_type': native['boosting_type'],
            'actual_trees': int(model.tree_count_), 'maximum_tree_leaf_count': int(leaves.max()),
            'tree_leaf_counts_sha256': hashlib.sha256(np.asarray(leaves, dtype='<i8').tobytes(order='C')).hexdigest()}


def native_raw_replay(model, view, cats, pool_class):
    require(view.shape[1] == 95 and len(cats) == 23
            and model.feature_names_ == list(view), 'Wrong native query features')
    require(model.get_cat_feature_indices() == sorted(list(view).index(c) for c in cats),
            'Wrong native categorical feature indices')
    raw = np.asarray(model.predict(pool_class(view, cat_features=cats, thread_count=4),
                                   thread_count=4), dtype='float64')
    require(raw.shape == (len(view),) and np.isfinite(raw).all(), 'Invalid raw million-TWD native output')
    return raw


def ordered_self_checks():
    """Synthetic feature/state fixtures and fake native CatBoost; no real model IO."""
    from round6_geo import NUMERIC, FEATURES, FEATURE_RULES as r6_rules
    checks = []
    def ok(name, value): require(value, name); checks.append(name)
    def rejects(name, operation):
        try: operation()
        except (RuntimeError, ValueError, KeyError, FloatingPointError): checks.append(name)
        else: raise AssertionError('Expected rejection: ' + name)
    idx = pd.Index([41, 3, 99, 12, 8], name='source_row')
    cats = ['category_' + str(i) for i in range(21)]
    X = pd.DataFrame({c: pd.Series(['known', 'new', '__MISSING__', 'literal', 'known'], index=idx, dtype=object) for c in cats}, index=idx)
    for j in range(66): X['numeric_' + str(j)] = np.arange(5, dtype='float32')
    geo = pd.DataFrame({c: np.full(5, np.nan, dtype='float32') for c in NUMERIC}, index=idx)
    geo['geo_has_position'] = np.zeros(5, dtype='float32')
    geo['geo_match_status'] = pd.Series(['ineligible_address'] * 5, index=idx, dtype=object)
    geo['geo_grid_002'] = pd.Series(['__MISSING__'] * 5, index=idx, dtype=object)
    geo = geo[FEATURES]
    before_X, before_geo = X.copy(deep=True), geo.copy(deep=True)
    view, all_cats = independent_model_view(X, geo, cats)
    ok('exact_original_R6_95_columns_23_categories', view.shape == (5, 95) and len(all_cats) == 23)
    ok('original95_values_dtypes_order_index_preserved', view.iloc[:, :87].equals(X)
       and view.iloc[:, 87:].equals(geo) and view.iloc[:, :87].dtypes.equals(X.dtypes)
       and view.iloc[:, 87:].dtypes.equals(geo.dtypes) and view.index.equals(idx))
    config = {'components': [{'family': 'catboost', 'parameters': {'depth': 8, 'random_seed': 42,
               'loss_function': 'MAE', 'boosting_type': 'Plain', 'learning_rate': .04, 'iterations': 4000, 'thread_count': 1, 'verbose': False}}]}
    baseline, parameters = independent_parameters(config)
    ok('only_boosting_type_changed_from_fixed_R6_recipe', baseline['iterations'] == 2999 and baseline['thread_count'] == 4
       and baseline['verbose'] == 300 and sorted(k for k in parameters if parameters[k] != baseline[k]) == ['boosting_type'])
    nonplain = copy.deepcopy(config); nonplain['components'][0]['parameters']['boosting_type'] = 'Ordered'
    rejects('nonPlain_baseline_rejected', lambda: independent_parameters(nonplain))
    require(config['components'][0]['parameters']['boosting_type'] == 'Plain', 'Input configuration mutated')
    checks.append('fixed_input_configuration_not_mutated')
    rules = {'geo_encoder_rules': copy.deepcopy(r6_rules), 'original_features_unchanged': 87,
             'existing_geo_features_unchanged': 8, 'feature_count': 95, 'categorical_feature_count': 23,
             'baseline_depth': 8, 'depth': 8, 'baseline_boosting_type': 'Plain',
             'boosting_type': 'Ordered', 'changed_parameter_names': ['boosting_type'],
             'required_actual_grow_policy': 'SymmetricTree', 'maximum_allowed_tree_leaf_count': 256,
             'iterations': 2999, 'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False}
    sources = {name: 'a' * 64 for name in HELPER_SOURCES}
    gs = {'feature_rules': copy.deepcopy(r6_rules), 'source_sha256': sources['src/round6_geo.py'],
          'fit_rows': 5, 'original_columns': list(X), 'original_dtypes': {c: str(X[c].dtype) for c in X},
          'all_feature_names': list(view), 'fit_original_feature_values_sha256': frame_sha(X),
          'fit_geo_feature_values_sha256': frame_sha(geo), 'fit_position_rows': 0,
          'fit_state_uses_y': False, 'fit_state_uses_query': False}
    gs['state_content_sha256'] = content_sha(gs)
    state = {'format_version': 1, 'component': 'geography_ordered_raw_catboost', 'target_mode': 'raw',
             'target_formula': 'y_TWD / 1e6', 'inverse_formula': 'max(z * 1e6, 0)',
             'feature_rules': rules, 'source_sha256': sources, 'base_configuration_sha256': 'b' * 64,
             'baseline_parameters': baseline, 'parameters': parameters, 'changed_parameter_names': ['boosting_type'],
             'geo_state': gs, 'original_categorical_columns': cats, 'categorical_columns': all_cats,
             'all_feature_names': list(view), 'fit_rows': 5, 'fit_model_feature_values_sha256': frame_sha(view),
             'fit_model_feature_dtypes': {c: str(view[c].dtype) for c in view}, 'fit_state_uses_y': False,
             'fit_state_uses_query': False, 'query_labels_used': False}
    state['state_content_sha256'] = content_sha(state)
    state = json.loads(json.dumps(state))
    spec = {'source_sha256': sources, 'baseline_parameters': baseline, 'parameters': parameters,
            'changed_parameter_names': ['boosting_type'], 'feature_rules': rules}
    proof = independent_state_check(state, X, geo, cats, spec, baseline, parameters, 'b' * 64)
    ok('JSON_state_fit_only_geo_and95_hashes_independent', proof['fit_state_reconstructed_independently'])
    replay, replay_cats = independent_query_view(state, X, geo, cats)
    pd.testing.assert_frame_equal(replay, view)
    ok('JSON_state_query_view_replayed', replay_cats == all_cats)
    pieces = [independent_query_view(state, X.iloc[a:b], geo.iloc[a:b], cats)[0] for a, b in [(0, 2), (2, 5)]]
    pd.testing.assert_frame_equal(pd.concat(pieces), view)
    checks.append('nonrange_index_batch_view_equality')
    rejects('misaligned_geo_rejected', lambda: independent_query_view(state, X, geo.iloc[::-1], cats))
    changed = X.copy(); changed['numeric_0'] = changed['numeric_0'].astype('float64')
    rejects('changed_original_dtype_rejected', lambda: independent_query_view(state, changed, geo, cats))
    rejects('changed_original_column_order_rejected', lambda: independent_query_view(state, X.iloc[:, ::-1], geo, cats))
    bad = copy.deepcopy(state); bad['fit_state_uses_query'] = True
    bad['state_content_sha256'] = content_sha({k: v for k, v in bad.items() if k != 'state_content_sha256'})
    rejects('rehashed_query_in_state_rejected', lambda: independent_state_check(bad, X, geo, cats, spec, baseline, parameters, 'b' * 64))
    class FakePool:
        def __init__(self, frame, **kwargs):
            require(kwargs == {'cat_features': all_cats, 'thread_count': 4}, 'Pool config or unexpected label/weight differs')
            self.frame = frame
    class FakeCatBoost:
        def __init__(self, depth=8, policy='SymmetricTree', tree_count=2999, leaves=None, names=None, cat_indices=None, boosting_type='Ordered'):
            self.depth, self.policy, self.tree_count_, self.boosting_type = depth, policy, tree_count, boosting_type
            self.leaves = np.full(2999, 256, dtype='int64') if leaves is None else leaves
            self.feature_names_ = list(view) if names is None else names
            self.cat_indices = sorted(list(view).index(c) for c in all_cats) if cat_indices is None else cat_indices
        def get_all_params(self): return {'depth': self.depth, 'grow_policy': self.policy, 'boosting_type': self.boosting_type}
        def get_tree_leaf_counts(self): return self.leaves
        def get_cat_feature_indices(self): return self.cat_indices
        def predict(self, pool, **kwargs):
            require(kwargs == {'thread_count': 4}, 'Prediction threading differs')
            return pool.frame['numeric_0'].to_numpy(dtype='float64') - .5
    model = FakeCatBoost()
    capacity = independent_capacity(model, view.columns)
    ok('native_capacity_Ordered_8_symmetric_2999_256', capacity['actual_model_depth'] == 8
       and capacity['actual_boosting_type'] == 'Ordered'
       and capacity['actual_grow_policy'] == 'SymmetricTree' and capacity['actual_trees'] == 2999
       and capacity['maximum_tree_leaf_count'] == 256)
    ok('native_leaf_hash_little_endian_int64', capacity['tree_leaf_counts_sha256'] == hashlib.sha256(np.full(2999, 256, dtype='<i8').tobytes()).hexdigest())
    for label, kwargs in [('Plain', {'boosting_type': 'Plain'}), ('depth6', {'depth': 6}), ('asymmetric', {'policy': 'Lossguide'}),
                          ('wrong_trees', {'tree_count': 2998}), ('feature_order', {'names': list(view)[::-1]}),
                          ('too_many_leaves', {'leaves': np.full(2999, 257, dtype='int64')}),
                          ('zero_leaves', {'leaves': np.zeros(2999, dtype='int64')}),
                          ('negative_leaves', {'leaves': np.full(2999, -1, dtype='int64')}),
                          ('floating_leaf_counts', {'leaves': np.full(2999, 256.)}),
                          ('wrong_leaf_vector_size', {'leaves': np.full(2998, 256, dtype='int64')})]:
        rejects('native_capacity_rejects_' + label, lambda kwargs=kwargs: independent_capacity(FakeCatBoost(**kwargs), view.columns))
    varied = np.full(2999, 256, dtype='int64'); varied[0] = 32
    other = independent_capacity(FakeCatBoost(leaves=varied), view.columns)
    ok('valid_smaller_tree_allowed_but_leaf_digest_changes', other['maximum_tree_leaf_count'] == 256
       and other['tree_leaf_counts_sha256'] != capacity['tree_leaf_counts_sha256'])
    raw = native_raw_replay(model, replay, all_cats, FakePool)
    ok('native_Pool_prediction_raw_millions', np.array_equal(raw, np.arange(5) - .5))
    prices = inverse_replay(raw)
    ok('raw_inverse_once_negative_projection_no_log', np.array_equal(prices, [0., .5e6, 1.5e6, 2.5e6, 3.5e6]))
    rejects('wrong_native_categorical_indices_rejected', lambda: native_raw_replay(FakeCatBoost(cat_indices=[0]), replay, all_cats, FakePool))
    split_raw = np.concatenate([native_raw_replay(model, replay.iloc[a:b], all_cats, FakePool) for a, b in [(0, 2), (2, 5)]])
    ok('native_prediction_batch_and_index_order_consistent', np.array_equal(raw, split_raw))
    ok('bitwise_total_price_replay', compare_vector(prices.copy(), prices, 'synthetic')['bitwise_equal'])
    altered = prices.copy(); altered[-1] = np.nextafter(altered[-1], np.inf)
    rejects('one_ULP_price_change_rejected', lambda: compare_vector(altered, prices, 'synthetic'))
    for name, values in [('nan', [np.nan]), ('inf', [np.inf]), ('matrix', [[1.]]), ('overflow', [np.finfo('float64').max])]:
        rejects('invalid_raw_inverse_' + name, lambda values=values: inverse_replay(values))
    pd.testing.assert_frame_equal(X, before_X); pd.testing.assert_frame_equal(geo, before_geo)
    checks.append('original_X_geo_not_mutated')
    return checks


def self_check():
    import copy
    checks = []
    def check(name, truth):
        require(truth, name); checks.append(name)
    def rejected(name, operation):
        try:
            operation()
        except RuntimeError:
            checks.append(name)
        else:
            raise AssertionError(name)
    validate_category_counts(21, 23)
    checks.append('original_87_21cats_and_augmented_95_23cats_are_distinct')
    rejected('original_23cats_schema_rejected', lambda: validate_category_counts(23, 23))
    rejected('augmented_21cats_schema_rejected', lambda: validate_category_counts(21, 21))
    start = pd.Timestamp('2022-08-11')
    day = (start - pd.Timestamp('2000-01-01')).days
    fit = pd.DataFrame({'address_city': ['A'], 'address_district': ['D'], 'project_name': ['known']})
    query = pd.DataFrame({'address_city': ['A'] * 4, 'address_district': ['D'] * 4,
                          'project_name': ['known', 'new', None, 'known'], 'trade_day_index': np.array([0, 92, 184, 276]) + day})
    weights = {'seen': .2, 'new_nonmissing': .3, 'missing': .5}
    result = independent_score([11, 12, 13, 14], np.repeat(10., 4), fit, query, str(start.date()), weights)
    check('overall_mae_in_twd', result['overall_mae'] == 2.5)
    check('fixed_group_weighted_mae', result['weighted_mae'] == .2 * 2.5 + .3 * 2 + .5 * 3)
    check('four_nonoverlapping_forecast_quarters', [v['mae'] for v in result['forecast_quarters'].values()] == [1., 2., 3., 4.])
    check('seen_new_missing_semantics', labels(fit, query).tolist() == ['seen', 'new_nonmissing', 'missing', 'seen'])
    subset = query.iloc[:1].copy()
    absent = independent_score([11], [10], fit, subset, str(start.date()), {'seen': 1., 'new_nonmissing': 0., 'missing': 0.})
    check('empty_strata_and_quarters_are_null', absent['strata']['missing']['mae'] is None and absent['forecast_quarters']['Q2']['mae'] is None)
    rejected('empty_positive_weight_stratum_rejected', lambda: independent_score([11], [10], fit, subset, str(start.date()), weights))
    rejected('unmatched_prediction_rows_rejected', lambda: independent_score([11], [10, 12], fit, subset, str(start.date()), weights))
    base = {'overall_mae': 100., 'weighted_mae': 100., 'strata': {'seen': {'rows': 500, 'mae': 100.}, 'missing': {'rows': 0, 'mae': None}},
            'forecast_quarters': {q: {'rows': 1, 'mae': 100.} for q in QUARTERS}}
    metrics = {CONTROL: copy.deepcopy(base), CANDIDATE: copy.deepcopy(base)}
    metrics[CANDIDATE]['overall_mae'] = metrics[CANDIDATE]['weighted_mae'] = 99.
    check('exact_one_percent_passes', independent_development_decision(metrics)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['weighted_mae'] = 99.00001
    check('both_mae_gates_required', not independent_development_decision(bad)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['strata']['seen']['mae'] = 105.00001
    check('large_group_deterioration_rejected', not independent_development_decision(bad)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 105.00001
    check('quarter_deterioration_rejected', not independent_development_decision(bad)['passed'])
    calls = []
    def reference_spy(ref, q, p, variant):
        require(ref == 'fit_reference' and q is query and variant == 'trend', 'Wrong reference identity')
        calls.append(p.copy()); return p * 1.1
    got = independent_blend([1] * 4, [3] * 4, [5] * 4, 'fit_reference', query, reference_spy)
    check('single_trend_fixed_slot_original_query', len(calls) == 1 and np.array_equal(got, np.repeat(1.1 * (.375 + 1.125 + 1.25), 4)))
    return checks

def independent_decision(stage, metrics):
    require(stage in SPLITS and set(metrics) == {CONTROL, CANDIDATE}, 'Unexpected stage/candidate')
    base, value = metrics[CONTROL], metrics[CANDIDATE]
    improved = [q for q, item in base['forecast_quarters'].items()
                if item['rows'] and value['forecast_quarters'][q]['mae'] < item['mae']]
    large = all(value['strata'][g]['mae'] <= 1.05 * item['mae']
                for g,item in base['strata'].items() if item['rows'] >= 500)
    if stage == 'development':
        checks = {
            'overall_improves_at_least_1pct': value['overall_mae'] <= .99 * base['overall_mae'],
            'weighted_improves_at_least_1pct': value['weighted_mae'] <= .99 * base['weighted_mae'],
            'quarters_deteriorate_at_most_5pct': all(value['forecast_quarters'][q]['mae'] <= 1.05 * item['mae'] for q,item in base['forecast_quarters'].items() if item['rows']),
            'large_strata_deteriorate_at_most_5pct': large,
        }
    else:
        bound = 1. if stage == 'confirmation' else 1.03
        checks = {'overall_within_limit': value['overall_mae'] <= bound * base['overall_mae'],
                  'weighted_within_limit': value['weighted_mae'] <= bound * base['weighted_mae'],
                  'large_strata_deteriorate_at_most_5pct': large}
        if stage == 'confirmation': checks['at_least_two_quarters_improve'] = len(improved) >= 2
    passed = all(checks.values())
    return {'passed': passed, 'selected_candidate': CANDIDATE if passed else CONTROL,
            'checks': checks, 'improved_quarters': improved, 'candidate_reselection_performed': False}


def independent_development_decision(metrics):
    return independent_decision('development', metrics)


def extra_self_checks():
    checks = self_check()
    base = {'overall_mae': 100., 'weighted_mae': 100.,
            'strata': {'seen': {'rows':500,'mae':100.}, 'missing': {'rows':0,'mae':None}},
            'forecast_quarters': {q: {'rows': 1,'mae':100.} for q in QUARTERS}}
    metrics = {CONTROL: copy.deepcopy(base), CANDIDATE: copy.deepcopy(base)}
    require(not independent_decision('confirmation',metrics)['passed'], 'Confirmation requires 2 strict improvements')
    checks.append('confirmation_two_quarters_required')
    for q in ('Q1','Q2'): metrics[CANDIDATE]['forecast_quarters'][q]['mae'] = 99.
    require(independent_decision('confirmation',metrics)['passed'], 'Confirmation equal overall plus 2 quarters allowed')
    checks.append('confirmation_equal_means_and_two_quarters_pass')
    metrics[CANDIDATE]['weighted_mae'] = 100.00001
    require(not independent_decision('confirmation',metrics)['passed'], 'Confirmation no tolerance beyond zero')
    checks.append('confirmation_tiny_weighted_deterioration_fails')
    metrics[CANDIDATE]['overall_mae'] = metrics[CANDIDATE]['weighted_mae'] = 103.
    metrics[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 9999.
    require(independent_decision('auxiliary',metrics)['passed'], 'Auxiliary exact 3 percent / diagnostic quarters')
    checks.append('auxiliary_exact_three_percent_and_diagnostic_quarters')
    metrics[CANDIDATE]['overall_mae'] = 103.00001
    require(not independent_decision('auxiliary',metrics)['passed'], 'Auxiliary >3 percent must fail')
    checks.append('auxiliary_over_three_percent_fails')
    checks.extend(ordered_self_checks())
    return checks


def inverse_replay(raw):
    raw = np.asarray(raw, dtype='float64')
    require(raw.ndim == 1 and np.isfinite(raw).all(), 'Invalid raw million-TWD predictions')
    with np.errstate(over='raise', invalid='raise'):
        result = raw * 1e6
    require(np.isfinite(result).all(), 'Nonfinite inverse prices')
    return np.maximum(result, 0.)


def verify(stage):
    """Read only the explicitly named saved stage; never fit or choose candidates."""
    require(stage in SPLITS, 'Unexpected stage')
    import round17_experiment as driver
    from catboost import CatBoostRegressor, Pool
    proof_path = OUT / 'audit' / (stage + '_replay.json')
    for p in (proof_path, proof_path.with_suffix('.sha256')):
        require(not p.exists(), 'Refusing to replace replay evidence')
    checks = extra_self_checks()
    driver.require_no_staging()
    plan = driver.authorize(stage)
    result = driver.read_result(stage)
    require(plan['thresholds'] == THRESHOLDS and plan['splits'] == {k:list(v) for k,v in SPLITS.items()}, 'Different fixed protocol')
    require(driver.CONTROL == CONTROL and driver.CANDIDATE == CANDIDATE, 'Different identities')
    require(plan['source_sha256']['src/round17_ordered_geo.py']==EXPECTED_HELPER_SHA256, 'Unexpected ordered raw CatBoost helper identity')
    require(result['evaluated_candidates'] == [CONTROL,CANDIDATE], 'Unexpected scored candidate')
    require(result['query_labels_used_for_training'] is False and result['public_score_used'] is False
            and result['historically_untouched'] is False and result['private_score_known'] is False, 'Incorrect exposure declaration')
    require(result['source_sha256'] == plan['source_sha256'], 'Result source differs')
    configuration_path=ROOT/'runs/v1/models/final_config.json'
    configuration_sha=sha(configuration_path)
    require(plan['dependency_sha256']['runs/v1/models/final_config.json']==configuration_sha,
            'Fixed v1 baseline configuration identity differs')
    baseline,parameters=independent_parameters(read(configuration_path))
    require(plan['baseline_parameters']==baseline and plan['changed_parameter_names']==['boosting_type']
            and plan['capacity_constraint']=={'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Ordered'},
            'Frozen boosting-only recipe differs')
    require(plan['round'] == 17 and plan['candidate'] == CANDIDATE
            and plan['control'] == CONTROL and plan['parameters'] == parameters
            and plan['feature_count'] == 95 and plan['categorical_count'] == 23,
            'Frozen independent model recipe differs')
    captured = {}
    def capture(p, expected=None):
        p=Path(p).resolve()
        require(p.is_relative_to(ROOT) and p.is_file(), 'Missing/unsafe audit input')
        digest=sha(p); name=rel(p)
        require(expected is None or digest==expected, 'Input identity changed: '+name)
        require(name not in captured or captured[name]==digest, 'Input changed during capture: '+name)
        captured[name]=digest
        return digest
    def nested(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if isinstance(key,str) and '/' in key and isinstance(item,str) and len(item)==64:
                    capture(ROOT/key,item)
                else: nested(item)
        elif isinstance(value,list):
            for item in value: nested(item)
    nested(plan); nested(result)
    for path in [Path(__file__), OUT/'candidate_plan.json',OUT/'evaluation'/stage/'results.json']:
        capture(path)
        if path.suffix=='.json': capture(path.with_suffix('.sha256'))
    freeze=driver.locked(OUT/'frozen_candidate.json')
    nested(freeze)
    for path in [OUT/'frozen_candidate.json',OUT/'frozen_candidate.sha256']:
        capture(path)
    if stage=='development':
        require(freeze['result_sha256']==sha(OUT/'evaluation/development/results.json')
                and freeze['passed']==result['decision']['passed']
                and freeze['later_stages_allowed']==result['decision']['passed'], 'Development freeze does not match result')
    else:
        require(freeze['passed'] is True and freeze['later_stages_allowed'] is True, 'No own development authorization')
    folder=OUT/'models'/stage
    meta=driver.locked(folder/'metadata.json')
    nested(meta)
    for path in folder.iterdir():
        require(path.is_file(), 'Unexpected nested model artifact')
        capture(path, meta.get('artifact_sha256',{}).get(path.name))
    required={'model.cbm','feature_state.json','predictions.npy','raw_z.npy','fit_indices.npy','query_indices.npy'}
    require(set(meta['artifact_sha256'])==required, 'Unexpected model artifact inventory')
    for split in ('train','valid','test'):
        for suffix in ('.pkl','_features.pkl','.json'): capture(ROOT/'data_cache'/(split+suffix))
    for name in ('valid_clean_mask.npy','valid_clean_mask.json'): capture(ROOT/'data_cache'/name)
    schema=read(ROOT/'reports/feature_schema.json')
    require(read(ROOT/'data_cache/valid_clean_mask.json')=={s:schema['splits'][s]['input_source_sha256'] for s in ('train','valid')}, 'Canonical mask would be rewritten')
    protected={rel(path):capture(path) for path in [ROOT/'outputs/submission_v2.csv',ROOT/'outputs/submission_v3.csv',ROOT/'runs/versions.json']}
    print(json.dumps({'event':'independent_replay_started','stage':stage,'training':False,'additional_windows':False}),flush=True)
    # The shared loader deserializes canonical train/valid targets; only this stage's query errors are calculated.
    data,X,y,Xq,yq,fi,qi=driver.partition(stage)
    require((len(X),len(Xq))==COUNTS[stage] and len(X.columns)==87, 'Canonical shape differs')
    require((result['fit_rows'],result['query_rows'])==COUNTS[stage], 'Saved result counts differ')
    validate_category_counts(len(data['categorical_columns']),23)
    require(not np.intersect1d(fi,qi).size, 'Fit and query overlap')
    start,end=SPLITS[stage]
    start_day=(pd.Timestamp(start)-pd.Timestamp('2000-01-01')).days
    end_day=(pd.Timestamp(end)-pd.Timestamp('2000-01-01')).days
    require(np.isfinite(X['trade_day_index']).all() and (X['trade_day_index']<start_day).all(), 'Fit time leakage')
    require(np.isfinite(Xq['trade_day_index']).all() and Xq['trade_day_index'].between(start_day,end_day).all(), 'Wrong query dates')
    for name,values in [('fit',fi),('query',qi)]:
        a=np.load(folder/(name+'_indices.npy'),allow_pickle=False)
        require(np.array_equal(a,values),'Model row indices differ')
        require(plan['split_contract'][stage][name+'_indices_content_sha256']==hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest(),'Plan row indices differ')
    weights=scoring_weights(data)
    require(weights==plan['target_group_weights']==result['target_group_weights'], 'Independent test-X group weights differ')
    geo,geoq,_=driver.load_geographic_partition(stage,fi,qi)
    state=read(folder/'feature_state.json')
    state_evidence=independent_state_check(state,X,geo,data['categorical_columns'],plan,baseline,parameters,configuration_sha)
    require(meta['component_source_sha256']==state['source_sha256']
            and meta['feature_state_content_sha256']==state['state_content_sha256'],
            'Component source/state metadata differs')
    require(meta['base_configuration_sha256']==configuration_sha
            and meta['baseline_parameters']==baseline and meta['changed_parameter_names']==['boosting_type'],
            'Metadata original recipe identity differs')
    require(meta['feature_columns']==state['all_feature_names']
            and meta['categorical_columns']==meta['categorical_features']==state['categorical_columns'],
            'Model metadata category/feature schema differs')
    for field in ('fit_original_feature_values_sha256','fit_geo_feature_values_sha256'):
        require(meta[field]==state['geo_state'][field], 'Metadata fit frame differs: '+field)
    require(meta['fit_model_feature_values_sha256']==state['fit_model_feature_values_sha256'],
            'Metadata 95-column fit view differs')
    for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]:
        require(meta[key]==frame_sha(frame),'Saved frame differs: '+key)
    y64=np.asarray(y,dtype='float64')
    require(y64.shape==(len(X),) and np.isfinite(y64).all() and (y64>=0).all(), 'Invalid raw fit labels')
    raw_y_sha=hashlib.sha256(y64.astype('<f8').tobytes()).hexdigest()
    transformed_sha=hashlib.sha256((y64/1e6).astype('<f8').tobytes()).hexdigest()
    require(meta['fit_target_sha256']==meta['fit_target_values_sha256']==meta['fit_original_target_values_sha256']==raw_y_sha,'Wrong target hash')
    require(meta['fit_transformed_target_values_sha256']==transformed_sha,'Wrong million-TWD target hash')
    require(meta['target_mode']=='raw' and meta['component']=='geography_ordered_raw_catboost' and meta['family']=='catboost'
            and meta['feature_count']==95 and meta['categorical_feature_count']==23
            and meta['calibration_applied'] is False and meta['extra_price_cap_applied'] is False
            and meta['reference_features_changed'] is False and meta['reference_or_trend_applied_here'] is False,
            'Wrong ordered raw geographic CatBoost semantics')
    require(meta['parameters']==plan['parameters']==parameters and meta['sample_weight']=='uniform'
            and meta['parameters_sha256']==content_sha(parameters)
            and meta['weight_argument_passed_to_pool'] is False
            and meta['iterations']==meta['actual_iterations']==meta['actual_trees']==2999
            and meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False
            and meta['query_labels_used'] is False and meta['prediction_units']=='TWD','Wrong training contract')
    require(meta['target_formula']=='y_TWD / 1e6' and meta['inverse_formula']=='max(z * 1e6, 0)','Wrong target units/formula')
    require(meta['plan_sha256']==sha(OUT/'candidate_plan.json') and meta['stage']==stage and meta['round']==17,'Wrong model provenance')
    require(meta['geo_manifest_sha256']==sha(driver.c6.OUT/'geography'/stage/'manifest.json'),'Wrong map manifest')
    # Direct native CBM/Pool replay; no R17 helper prediction or capacity calls.
    model=CatBoostRegressor();model.load_model(str(folder/'model.cbm'))
    view,cats=independent_query_view(state,Xq,geoq,data['categorical_columns'])
    native_capacity=independent_capacity(model,state['all_feature_names'])
    require(all(meta[key]==value for key,value in native_capacity.items()),
            'Independent native boosting/depth/tree/leaf diagnostics differ')
    raw=native_raw_replay(model,view,cats,Pool)
    for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]:
        require(meta[key]==frame_sha(frame),'Replay changed input frame: '+key)
    saved_raw=np.load(folder/'raw_z.npy',allow_pickle=False)
    require(saved_raw.dtype==np.dtype('float64') and saved_raw.shape==raw.shape==(len(Xq),) and np.isfinite(raw).all() and np.array_equal(saved_raw.view('uint64'),raw.view('uint64')),'All-row raw z replay differs')
    prices=inverse_replay(raw)
    saved_prices=np.load(folder/'predictions.npy',allow_pickle=False)
    require(saved_prices.dtype==np.dtype('float64'), 'Saved TWD predictions must be float64')
    replays={'component_TWD':compare_vector(saved_prices,prices,'component_TWD')}
    raw_diagnostic={'rows':len(raw),'negative_count':int(np.sum(raw<0)),'zero_count':int(np.sum(raw==0)), 'min':float(raw.min()),'max':float(raw.max()),'positive_count':int(np.sum(raw>0)),
                    'zero_price_count':int(np.sum(prices==0)),
                    'raw_values_sha256':hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest(),
                    'TWD_values_sha256':hashlib.sha256(prices.astype('<f8').tobytes()).hexdigest(),
                    'inverse_matches_component_all_rows':True,'used_for_tuning':False}
    # Driver read_result checks saved metadata integrity; these diagnostics are independently recomputed.
    require(all(meta['raw_z_diagnostics'][k]==v for k,v in raw_diagnostic.items()),'Raw z diagnostics differ')
    reference,_=driver.c6.load_reference(stage,X)
    base=driver.c6.baseline_folder(stage)
    old=np.load(base/'catboost_raw.npy',allow_pickle=False)
    unit=np.load(base/'lightgbm_l1_unit63.npy',allow_pickle=False)
    old_geo=np.load(driver.c6.OUT/'models'/stage/'geography_catboost/predictions.npy',allow_pickle=False)
    control=independent_blend(old,old_geo,unit,reference,Xq,driver.predict_reference)
    candidate=independent_blend(old,prices,unit,reference,Xq,driver.predict_reference)
    original=np.load(driver.c6.OUT/'evaluation'/stage/'geography_blend.npy',allow_pickle=False)
    replays['original_v3_TWD']=compare_vector(original,control,'original_v3_TWD')
    for name,values in [(CONTROL,control),(CANDIDATE,candidate)]:
        saved=np.load(OUT/'evaluation'/stage/(name+'.npy'),allow_pickle=False)
        replays[name]=compare_vector(saved,values,name)
    metrics={name:independent_score(yq,values,X,Xq,start,weights) for name,values in [(CONTROL,control),(CANDIDATE,candidate)]}
    require(metrics==result['metrics'],'Independent all metrics differ')
    original_result=driver.locked(driver.c6.OUT/'evaluation'/stage/'results.json')
    require(metrics[CONTROL]==original_result['metrics']['geography_blend'],
            'Independent control metrics differ from the archived same-window v3 result')
    decision=independent_decision(stage,metrics)
    require(decision==result['decision'],'Independent decision differs')
    require(freeze['round']==17 and freeze['candidate']==CANDIDATE
            and freeze['plan_sha256']==sha(OUT/'candidate_plan.json')
            and freeze['source_sha256']==plan['source_sha256']
            and freeze['parameters']==parameters and freeze['thresholds']==THRESHOLDS
            and freeze['target_group_weights']==weights, 'Own freeze identity/configuration differs')
    if stage=='development':
        require(freeze['selected_candidate']==decision['selected_candidate'], 'Independent selection differs from freeze')
        if not decision['passed']:
            for later in ('confirmation','auxiliary','final'):
                require(not (OUT/'models'/later).exists() and not (OUT/'evaluation'/later).exists(),
                        'Development rejection has unauthorized later artifacts')
    acceptance_sha=None
    acceptance_path=OUT/'accepted_candidate.json'
    if acceptance_path.exists():
        acceptance=driver.locked(acceptance_path)
        acceptance_sha=capture(acceptance_path);capture(acceptance_path.with_suffix('.sha256'));nested(acceptance)
        require(acceptance['round']==17 and acceptance['candidate']==CANDIDATE
                and acceptance['plan_sha256']==sha(OUT/'candidate_plan.json')
                and acceptance['frozen_candidate_sha256']==sha(OUT/'frozen_candidate.json')
                and acceptance['source_sha256']==plan['source_sha256']
                and acceptance['stage_result_sha256'][stage]==sha(OUT/'evaluation'/stage/'results.json'),
                'Finalized acceptance is not bound to the audited stage')
        if stage=='development' and not decision['passed']:
            require(acceptance['accepted'] is False and acceptance['deployment_decision']=='keep_v3'
                    and acceptance['stopping_reason']=='development_price_gates_failed'
                    and set(acceptance['stage_result_sha256'])=={'development'}
                    and acceptance['unevaluated_stages']==['confirmation','auxiliary']
                    and all(acceptance['stage_status'][q]=='not_executed_after_development_rejection'
                            for q in ('confirmation','auxiliary')), 'Development-stop receipt differs')
    driver.require_no_staging(); driver.authorize(stage); driver.read_result(stage)
    for name,digest in captured.items(): require(sha(ROOT/name)==digest,'Audited input changed: '+name)
    proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),'round':17,'stage':stage,'status':'passed',
           'candidate_accepted_in_this_stage':decision['passed'],'fit_rows':len(X),'query_rows':len(Xq),
           'source_sha256':sha(Path(__file__)),'plan_sha256':sha(OUT/'candidate_plan.json'),
           'result_sha256':sha(OUT/'evaluation'/stage/'results.json'),
           'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),
           'accepted_candidate_sha256':acceptance_sha,
           'independent_fit_state_evidence':state_evidence,
           'saved_raw_z_bitwise_equal':True,'raw_z_diagnostics':raw_diagnostic,'replays':replays,
           'independent_metrics':metrics,'independent_decision':decision,'synthetic_checks':checks,
           'source_input_sha256':captured,'protected_artifacts_unchanged':protected,
           'fit_performed':False,'new_window_scored':False,'existing_query_errors_recomputed':True,
           'shared_canonical_loader_deserializes_train_valid_targets':True,
           'model_prediction_reimplemented_directly':True,'inverse_reimplemented_independently':True,
           'model_view_reimplemented_independently':True,'raw_z_units':'million_TWD',
           'model_family':'catboost','model_format':'native_CatBoost_cbm',
           'feature_count':95,'categorical_count':23,'actual_iterations':2999,
           'native_model_capacity':native_capacity,
           'baseline_parameters':baseline,'changed_parameter_names':['boosting_type'],
           'fit_feature_state_reconstructed_independently':True,
           'query_model_view_checked_independently':True,
           'R17_helper_state_prediction_and_capacity_not_called':True,
           'R17_driver_reused_for_provenance_partition_and_saved_integrity':True,
           'driver_read_result_secondary_gate_integrity_check':True,
           'independent_metrics_and_gates_do_not_call_driver_scoring_or_decision':True,
           'R6_geographic_augmentation_reused':True,
           'archived_same_window_v3_metrics_equal':True,
           'original_95_values_dtypes_order_index_preserved':True,
           'original_87_reference_and_scoring_keys_unchanged':True,
           'legacy_reference_function_reused':True,'public_score_used':False,'historically_untouched':False}
    driver.locked_write(proof_path,proof)
    print(json.dumps({'event':'independent_replay_passed','stage':stage,'query_rows':len(Xq),'proof_sha256':sha(proof_path),'raw_z_diagnostics':raw_diagnostic,'replays':replays}),flush=True)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['self-check','verify'])
    parser.add_argument('--stage',choices=list(SPLITS))
    args=parser.parse_args()
    if args.action=='self-check':
        checks=extra_self_checks()
        z=np.array([-1.,0.,2.])
        require(np.allclose(inverse_replay(z),[0.,0.,2e6],rtol=1e-14),'Inverse unit check')
        print(json.dumps({'checks':checks+['independent_inverse_units_and_zero_projection'],'real_data_read':False,'model_fit':False}),flush=True)
    else:
        require(args.stage is not None,'Verify requires an explicit saved stage')
        verify(args.stage)


if __name__=='__main__': main()
