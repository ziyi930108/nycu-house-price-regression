"""Fixed R16 raw geography CatBoost: append one deterministic unit-code prefix.

No dataset loader, query-label API, trend, validation set or early stopping.
The CLI runs synthetic FakeCatBoost/Pool checks only.
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
import round16_unit_prefix as prefix_parser
from prepare_data import ROOT, sha256

COMPONENT = 'geography_prefix_raw_catboost'
PREFIX_COLUMN = 'unit_code_prefix'
PREFIX_SOURCE_COLUMN = 'building_and_unit'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'existing_geo_features_unchanged': 8,
    'existing_95_values_dtypes_order_and_index_unchanged': True,
    'feature_count': 96,
    'categorical_feature_count': 24,
    'geo_encoder_rules': _copy_json(base.FEATURE_RULES),
    'prefix_column': PREFIX_COLUMN,
    'prefix_source_column': PREFIX_SOURCE_COLUMN,
    'prefix_parser_rules': _copy_json(prefix_parser.RULES),
    'prefix_missing_value': prefix_parser.MISSING,
    'prefix_dtype': 'object',
    'prefix_position': 'appended after the original R6 95 features',
    'prefix_is_physical_block_or_household_identifier': False,
    'prefix_rule_uses_fit_statistics': False,
    'prefix_rule_uses_query_statistics': False,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'target_dtype': 'float64',
    'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': 'uniform',
    'weight_argument_passed_to_pool': False,
    'baseline_depth': 8,
    'depth': 8,
    'changed_parameter_names': [],
    'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 256,
    'iterations': 2999,
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
        'src/round16_prefix_geo.py', 'src/round16_unit_prefix.py',
        'src/round6_geo.py', 'src/prepare_data.py')}


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == 'MAE', 'Expected the fixed R6 depth-8 recipe')
    return result


def parameters():
    """An independent copy of the original R6 recipe; no parameter is changed."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    base.need(result == baseline, 'No R6 model parameter may change')
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
    """Append a deterministic category; preserve all original R6 95 values."""
    _check_X(X)
    base.need(PREFIX_SOURCE_COLUMN in X and PREFIX_COLUMN not in X,
              'Expected the original unit string and no existing prefix feature')
    base.need(prefix_parser.MISSING == '__MISSING__'
              and _copy_json(prefix_parser.RULES) == FEATURE_RULES['prefix_parser_rules'],
              'Prefix parser contract changed')
    original = base.augment_features(X, geo)
    base.need(len(original.columns) == 95 and original.columns.is_unique
              and original.index.equals(X.index), 'Expected the original R6 95-feature schema')
    # This per-call memo is deterministic parsing only, never learned fit/query statistics.
    strings = X[PREFIX_SOURCE_COLUMN].map(
        lambda value: prefix_parser.MISSING if pd.isna(value) else str(value))
    memo = {value: prefix_parser.extract_prefix(value) for value in pd.unique(strings)}
    base.need(all(isinstance(value, str) and value for value in memo.values()),
              'The parser must return nonempty strings, including its missing sentinel')
    result = original.copy(deep=True)
    result[PREFIX_COLUMN] = strings.map(memo).astype(object)
    base.need(len(result.columns) == 96 and result.columns[-1] == PREFIX_COLUMN
              and result.iloc[:, :95].equals(original)
              and result.iloc[:, :95].dtypes.equals(original.dtypes)
              and result.index.equals(X.index), 'The prefix must not alter the original 95 features')
    return result


def categories(X, supplied):
    original = base.categories(X, supplied)
    base.need(PREFIX_SOURCE_COLUMN in original and PREFIX_COLUMN not in original
              and len(original) == len(set(original)) == 23,
              'Expected original 23 categories, including the unmodified unit string')
    return original + [PREFIX_COLUMN]


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
        'changed_parameter_names': [],
        'geo_state': geo_state,
        'prefix_parser_rules': _copy_json(prefix_parser.RULES),
        'prefix_parser_sha256': source_hashes()['src/round16_unit_prefix.py'],
        'base_feature_names': list(geo_state['all_feature_names']),
        'fit_prefix_nonmissing_rows': int((augmented[PREFIX_COLUMN] != prefix_parser.MISSING).sum()),
        'fit_prefix_unique_values': int(augmented.loc[augmented[PREFIX_COLUMN] != prefix_parser.MISSING, PREFIX_COLUMN].nunique()),
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
              and state['feature_rules'] == FEATURE_RULES, 'Prefix model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == []
              and state['baseline_parameters'] == state['parameters'], 'Unchanged R6 configuration required')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['base_feature_names'] == state['geo_state']['all_feature_names']
              and len(state['base_feature_names']) == 95
              and state['all_feature_names'] == state['base_feature_names'] + [PREFIX_COLUMN],
              'Original R6 model columns or appended prefix changed')
    base.need(state['prefix_parser_rules'] == prefix_parser.RULES
              and state['prefix_parser_sha256'] == source_hashes()['src/round16_unit_prefix.py'],
              'Prefix parser source or rules changed')
    base.need(0 <= state['fit_prefix_unique_values'] <= state['fit_prefix_nonmissing_rows'] <= state['fit_rows'],
              'Invalid fit-only prefix evidence')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_model(model, state):
    """Check native model capacity, not just the supplied metadata declaration."""
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 8
              and len(state['all_feature_names']) == 96
              and state['changed_parameter_names'] == []
              and state['parameters'] == state['baseline_parameters'], 'Wrong prefix model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    actual = model.get_all_params()
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree',
              'Native depth/grow policy differs from the fixed original R6 model')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 256).all(),
              'Expected 2999 positive integer leaf counts, each at most 256')
    return {
        'actual_model_depth': int(actual['depth']),
        'actual_grow_policy': actual['grow_policy'],
        'actual_trees': int(model.tree_count_),
        'maximum_tree_leaf_count': int(leaves.max()),
        'tree_leaf_counts_sha256': hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest(),
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
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
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
        'changed_parameter_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 96, 'feature_count': 96, 'categorical_feature_count': 24,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round16_prefix_geo.py'],
        'source_sha256': _copy_json(state['source_sha256']),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_rows': int(len(X)),
        'prefix_parser_rules': _copy_json(state['prefix_parser_rules']),
        'prefix_parser_sha256': state['prefix_parser_sha256'],
        'fit_prefix_nonmissing_rows': state['fit_prefix_nonmissing_rows'],
        'fit_prefix_unique_values': state['fit_prefix_unique_values'],
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
    """Bounded synthetic parser integration and FakeCatBoost/Pool; no real ML IO."""
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
    cats = [PREFIX_SOURCE_COLUMN] + ['category_' + str(i) for i in range(20)]
    X = pd.DataFrame({c: pd.Series(['原字串'] * len(idx), index=idx, dtype=object)
                      for c in cats}, index=idx)
    X[PREFIX_SOURCE_COLUMN] = pd.Series(['A1-5F號', 'A1棟6F號', None, 'A1號'], index=idx, dtype=object)
    for i in range(66):
        X['numeric_' + str(i)] = np.arange(len(idx), dtype='float32')
    geo = pd.DataFrame({c: np.full(len(idx), np.nan, dtype='float32') for c in base.NUMERIC}, index=idx)
    geo['geo_has_position'] = np.zeros(len(idx), dtype='float32')
    geo['geo_match_status'] = pd.Series(['ineligible_address'] * len(idx), index=idx, dtype=object)
    geo['geo_grid_002'] = pd.Series(['__MISSING__'] * len(idx), index=idx, dtype=object)
    geo = geo[base.FEATURES]
    before_X, before_geo = X.copy(deep=True), geo.copy(deep=True)
    original = base.augment_features(X, geo)
    augmented = augment_features(X, geo)
    ok('original_R6_95_values_dtypes_order_index_exact', augmented.iloc[:, :95].equals(original)
       and augmented.iloc[:, :95].dtypes.equals(original.dtypes) and augmented.index.equals(idx))
    ok('96_features_24_categories_prefix_last', len(augmented.columns) == 96
       and augmented.columns[-1] == PREFIX_COLUMN and categories(X,cats) == base.categories(X,cats) + [PREFIX_COLUMN])
    ok('real_parser_integration_missing_and_cross_format_code', augmented[PREFIX_COLUMN].tolist()
       == ['A1', 'A1', prefix_parser.MISSING, prefix_parser.MISSING]
       and augmented[PREFIX_COLUMN].dtype == np.dtype(object))
    saved_extract = prefix_parser.extract_prefix
    seen = []
    def stub_extract(value):
        seen.append(value)
        return 'STUB_CODE' if value == 'A1-5F號' else prefix_parser.MISSING
    prefix_parser.extract_prefix = stub_extract
    try:
        stub_view = augment_features(X,geo)
        ok('delegation_to_external_parser_no_embedded_regex', stub_view[PREFIX_COLUMN].tolist()
           == ['STUB_CODE',prefix_parser.MISSING,prefix_parser.MISSING,prefix_parser.MISSING]
           and set(seen) == {'A1-5F號','A1棟6F號',prefix_parser.MISSING,'A1號'})
    finally:
        prefix_parser.extract_prefix = saved_extract
    augmented.iloc[0,0] = 'output only'; augmented.iloc[1,-1] = 'output only'
    ok('independent_output_copy', X.equals(before_X) and geo.equals(before_geo))
    rejects('geo_index_misalignment_rejected', lambda: augment_features(X,geo.iloc[::-1]))
    rejects('target_column_rejected', lambda: augment_features(X.rename(columns={'numeric_0':'target'}),geo))
    rejects('non87_input_rejected', lambda: augment_features(X.assign(extra=np.float32(1)),geo))
    rejects('missing_unit_source_rejected', lambda: augment_features(X.rename(columns={PREFIX_SOURCE_COLUMN:'other'}),geo))
    rejects('preexisting_prefix_rejected', lambda: augment_features(X.rename(columns={'numeric_0':PREFIX_COLUMN}),geo))
    duplicate=X.copy(); duplicate.index=[0]*len(X)
    rejects('duplicate_index_rejected', lambda: augment_features(duplicate,geo))
    ok('raw_target_scale', np.array_equal(transform_target([0.0,1e6,3e6]),[0.0,1.0,3.0]))
    ok('raw_inverse_scale_and_zero_projection', np.array_equal(inverse_transform([-1.0,0.0,3.0]),[0.0,0.0,3e6]))
    for label,values in [('negative',[-1.0]),('nan',[np.nan]),('infinite',[np.inf])]:
        rejects('invalid_target_'+label, lambda v=values: transform_target(v))
    rejects('inverse_overflow_rejected', lambda: inverse_transform([np.finfo('float64').max]))
    ok('no_target_or_query_in_feature_state_API', list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats']
       and list(inspect.signature(predict_model).parameters)==['model','state','X','geo','cats'])

    fake_baseline={'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,
                   'loss_function':'MAE','task_type':'GPU','verbose':300,'learning_rate':0.05}
    saved_baseline=_copy_json(fake_baseline)
    class FakePool:
        def __init__(self,data,label=None,**kwargs):
            self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=_copy_json(kwargs)
            self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999
            self.leaf_counts=np.full(2999,256,dtype=np.uint32)
        def fit(self,pool,**kwargs):
            if kwargs:
                raise AssertionError('No eval_set, early stopping, or weights')
            self.feature_names_=list(pool.data.columns)
            self.fit_pool=pool
            return self
        def predict(self,pool,**kwargs):
            if pool.label is not None:
                raise AssertionError('No query labels')
            return pool.data['numeric_0'].to_numpy(dtype='float64')-.25
        def get_all_params(self):
            return dict(self.actual)
        def get_tree_leaf_counts(self):
            return self.leaf_counts.copy()

    original_model,original_pool=globals()['CatBoostRegressor'],globals()['Pool']
    original_parameters=base.parameters
    original_config_sha=globals()['_base_configuration_sha256']
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:fake_baseline
    globals()['_base_configuration_sha256']=lambda:'a'*64
    try:
        config=parameters()
        ok('R6_parameters_exactly_unchanged',config==saved_baseline)
        config['depth']=99
        ok('independent_parameter_copy',fake_baseline==saved_baseline and parameters()['depth']==8)
        fake_baseline['depth']=6
        rejects('wrong_baseline_depth_rejected',parameters)
        fake_baseline['depth']=8
        y=pd.Series([0.0,1e6,2e6,3e6],index=idx,dtype='float64')
        model,state,meta=fit_model(X,y,geo,cats)
        ok('uniform_Pool_raw_fit_y_only',np.array_equal(model.fit_pool.label,y.to_numpy()/1e6)
           and set(model.fit_pool.kwargs)=={'cat_features','thread_count'}
           and model.fit_pool.kwargs['cat_features']==categories(X,cats))
        ok('constructor_and_metadata_have_no_parameter_delta',model.config==saved_baseline
           and state['baseline_parameters']==state['parameters']==saved_baseline
           and meta['baseline_parameters']==meta['parameters']==saved_baseline
           and state['changed_parameter_names']==meta['changed_parameter_names']==[])
        ok('fit_only_prefix_evidence',state['fit_prefix_nonmissing_rows']==meta['fit_prefix_nonmissing_rows']==2
           and state['fit_prefix_unique_values']==meta['fit_prefix_unique_values']==1
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        ok('parser_rules_and_source_bound',state['prefix_parser_rules']==prefix_parser.RULES
           and meta['prefix_parser_rules']==prefix_parser.RULES
           and state['prefix_parser_sha256']==meta['prefix_parser_sha256']==source_hashes()['src/round16_unit_prefix.py'])
        native=check_model(model,state)
        leaf_sha=hashlib.sha256(np.full(2999,256,dtype='<i8').tobytes()).hexdigest()
        ok('native_depth_policy_leaf_cap_and_LE_int64_hash',native=={
            'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
            'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':leaf_sha})
        ok('metadata_raw96_24_no_calibration_or_trend',all(meta[k]==v for k,v in native.items())
           and meta['feature_count']==96 and meta['categorical_feature_count']==24 and meta['target_mode']=='raw'
           and meta['sample_weight']=='uniform' and not meta['weight_argument_passed_to_pool']
           and not meta['reference_or_trend_applied_here'] and not meta['calibration_applied']
           and not meta['extra_price_cap_applied'])
        ok('original_and_transformed_target_hashes',meta['fit_target_values_sha256']
           ==meta['fit_original_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(y.to_numpy()/1e6))
        restored=_copy_json(state)
        prediction=predict_model(model,restored,X,geo,cats)
        ok('JSON_roundtrip_float64_TWD_prediction',prediction.dtype==np.dtype('float64')
           and np.array_equal(prediction,np.maximum(np.arange(len(X))-.25,0)*1e6))
        pieces=[predict_model(model,restored,X.iloc[a:b],geo.iloc[a:b],cats) for a,b in [(0,2),(2,4)]]
        ok('batch_and_full_equivalence',np.array_equal(prediction,np.concatenate(pieces)))
        query=X.iloc[[3,1]].copy(); query.loc[query.index[0],PREFIX_SOURCE_COLUMN]='ZZZ99-10F號'
        before_state=_copy_json(restored)
        predict_model(model,restored,query,geo.loc[query.index],cats)
        ok('unseen_query_prefix_preserved_without_fit_vocabulary',
           augment_features(query,geo.loc[query.index])[PREFIX_COLUMN].iloc[0]=='ZZZ99'
           and restored==before_state)
        bad=_copy_json(state); bad['prefix_parser_rules']['version']=999
        ok('prefix_rule_deep_copy',FEATURE_RULES['prefix_parser_rules']['version']==prefix_parser.RULES['version']==1)
        rejects('modified_state_rejected',lambda:check_state(bad,X,cats))
        rejects('y_index_misalignment_rejected',lambda:fit_model(X,y.iloc[::-1],geo,cats))
        model.actual['depth']=6
        rejects('wrong_native_depth_rejected',lambda:check_model(model,state))
        rejects('prediction_native_capacity_guard',lambda:predict_model(model,state,X,geo,cats))
        model.actual['depth']=8; model.actual['grow_policy']='Lossguide'
        rejects('wrong_native_grow_policy_rejected',lambda:check_model(model,state))
        model.actual['grow_policy']='SymmetricTree'
        for name,value in [('over256',257),('zero',0)]:
            model.leaf_counts[0]=value
            rejects('invalid_leaf_count_'+name,lambda:check_model(model,state))
        model.leaf_counts[0]=256
        saved_leaves=model.leaf_counts
        model.leaf_counts=saved_leaves.astype('float64')
        rejects('noninteger_leaf_dtype_rejected',lambda:check_model(model,state))
        model.leaf_counts=saved_leaves[:-1]
        rejects('wrong_leaf_vector_length_rejected',lambda:check_model(model,state))
        model.leaf_counts=saved_leaves; model.tree_count_=2998
        rejects('wrong_native_tree_count_rejected',lambda:check_model(model,state))
        model.tree_count_=2999; model.feature_names_=list(reversed(model.feature_names_))
        rejects('wrong_native_feature_order_rejected',lambda:check_model(model,state))
        ok('all_inputs_and_baseline_unchanged',X.equals(before_X) and geo.equals(before_geo)
           and fake_baseline==saved_baseline)
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=original_model,original_pool
        base.parameters=original_parameters
        globals()['_base_configuration_sha256']=original_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),'real_data_loaded':False,
            'real_model_training_performed':False,'real_model_or_configuration_loaded':False,
            'files_written':False,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=False),flush=True)
