"""Fixed R19 geography CatBoost: area-weighted unit target, original R6 parameters.

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
from prepare_data import ROOT, sha256

COMPONENT = 'geography_unit_catboost'
TARGET_MODE = 'unit'
TARGET_FORMULA = 'y_TWD / (effective_area(X) * 1e4)'
INVERSE_FORMULA = 'max(z * (effective_area(X) * 1e4), 0)'
AREA_RULE = 'finite positive building_area; else finite positive parking_area; else 1'
WEIGHT_FORMULA = 'effective_area / 100'


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
    'geo_rule_target_weight_fields_are_provenance_only': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'target_dtype': 'float64',
    'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': WEIGHT_FORMULA,
    'weight_argument_passed_to_pool': True,
    'baseline_depth': 8,
    'depth': 8,
    'baseline_boosting_type': 'Plain',
    'effective_area_rule': AREA_RULE,
    'factor': 'effective_area * 1e4',
    'weight_dtype': 'float64',
    'weights_normalized_to_mean_one': False,
    'loss_equivalence': '(A/100)*abs(y/(A*1e4)-z) = abs(y-A*1e4*z)/1e6 before nonnegative prediction clipping',
    'removed_feature_names': [],
    'boosting_type': 'Plain',
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
        'src/round19_unit_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == 'MAE' and result['boosting_type'] == 'Plain',
              'Expected the fixed R6 depth-8 Plain recipe')
    return result


def parameters():
    """All passed CatBoost parameters are unchanged from R6; only Pool labels/weights differ."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    base.need(result == baseline, 'Passed CatBoost parameters must remain the R6 recipe')
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


def _area_values(X):
    _check_X(X)
    base.need({'building_area', 'parking_area'}.issubset(X.columns), 'Required area columns are missing')
    arrays = []
    for name in ('building_area', 'parking_area'):
        raw = X[name].to_numpy(copy=True)
        base.need(raw.ndim == 1 and raw.dtype.kind in 'iuf', 'Area columns must be real numeric')
        arrays.append(np.array(raw, dtype=np.float64, order='C', copy=True))
    return arrays


def effective_area(X):
    """Exact R1 numerical denominator fallback; not a net-indoor-area estimate."""
    building, parking = _area_values(X)
    building_ok = np.isfinite(building) & (building > 0)
    parking_ok = np.isfinite(parking) & (parking > 0)
    area = np.where(building_ok, building, np.where(parking_ok, parking, 1.0))
    base.need(np.isfinite(area).all() and (area > 0).all(), 'Effective area must be finite and positive')
    return np.asarray(area, dtype=np.float64)


def sample_weights(X):
    with np.errstate(over='raise', invalid='raise'):
        weights = effective_area(X) / 100.0
    base.need(np.isfinite(weights).all() and (weights > 0).all(), 'Pool weights must be finite and positive')
    return np.asarray(weights, dtype=np.float64)


def _price_factor(X):
    try:
        with np.errstate(over='raise', invalid='raise'):
            factor = effective_area(X) * 1e4
    except FloatingPointError as exc:
        raise ValueError('Area factor overflowed') from exc
    base.need(np.isfinite(factor).all() and (factor > 0).all(), 'Price factor must be finite and positive')
    return np.asarray(factor, dtype=np.float64)


def _aligned_vector(values, X, name):
    _check_X(X)
    if isinstance(values, pd.Series):
        base.need(values.index.equals(X.index), name + ' Series index differs from X')
    result = _float_vector(values, name)
    base.need(result.shape == (len(X),), name + ' length differs from X')
    return result


def transform_target(y, X):
    raw = _aligned_vector(y, X, 'Fit target')
    base.need((raw >= 0).all(), 'Fit prices must be nonnegative')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = raw / _price_factor(X)
    except FloatingPointError as exc:
        raise ValueError('Unit target transformation overflowed') from exc
    base.need(np.isfinite(result).all(), 'Unit target must be finite')
    return np.asarray(result, dtype=np.float64)


def inverse_transform(z, X):
    raw = _aligned_vector(z, X, 'Model output')
    try:
        with np.errstate(over='raise', invalid='raise'):
            prices = raw * _price_factor(X)
    except FloatingPointError as exc:
        raise ValueError('Unit prediction TWD reconstruction overflowed') from exc
    base.need(np.isfinite(prices).all(), 'Unit reconstruction must be finite')
    return np.maximum(prices, 0).astype(np.float64, copy=False)


def weight_diagnostics(X):
    """Fit-only non-price summaries, with mutually exclusive denominator sources."""
    building, parking = _area_values(X)
    b = np.isfinite(building) & (building > 0)
    p = ~b & np.isfinite(parking) & (parking > 0)
    default = ~b & ~p
    def summary(values):
        base.need(len(values) > 0, 'Cannot summarize empty fit weights')
        # Scale before summation so an otherwise-finite factor mean does not overflow.
        return {'min': float(values.min()), 'max': float(values.max()),
                'mean': float(np.sum(values / len(values), dtype=np.float64))}
    return {'rows': int(len(X)), 'effective_area_rule': AREA_RULE,
            'effective_area_source_counts': {'building_area': int(b.sum()),
                'parking_area_fallback': int(p.sum()), 'default_one_fallback': int(default.sum())},
            'effective_area': summary(effective_area(X)), 'sample_weight': summary(sample_weights(X)),
            'price_factor': summary(_price_factor(X)), 'weights_normalized_to_mean_one': False}


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
        'changed_parameter_names': [],
        'removed_feature_names': [],
        'geo_state': geo_state,
        'fit_effective_area_sha256': values_sha256(effective_area(X)),
        'fit_sample_weights_sha256': values_sha256(sample_weights(X)),
        'fit_factor_sha256': values_sha256(_price_factor(X)),
        'fit_weight_diagnostics': weight_diagnostics(X),
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
              and state['feature_rules'] == FEATURE_RULES, 'Unit model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Unit target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == [] and state['removed_feature_names'] == []
              and state['parameters'] == state['baseline_parameters'], 'Original R6 configuration changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names']
              and len(state['all_feature_names']) == 95, 'Original R6 model columns changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_model(model, state):
    """Check native model capacity, not just the supplied metadata declaration."""
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 8
              and state['parameters']['boosting_type'] == 'Plain'
              and state['parameters'] == state['baseline_parameters']
              and state['changed_parameter_names'] == [] and state['removed_feature_names'] == []
              and len(state['all_feature_names']) == 95, 'Wrong unit model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    category_indices = np.asarray(model.get_cat_feature_indices())
    expected_categories = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(category_indices.ndim == 1 and category_indices.dtype.kind in 'iu'
              and len(expected_categories) == len(set(expected_categories)) == 23
              and category_indices.tolist() == expected_categories, 'Native 23-category indices differ')
    actual = model.get_all_params()
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain',
              'Native depth/grow/boosting policy differs from the fixed R6 recipe')
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
    }


def fit_model(X, y, geo, cats):
    """Train once on fit labels only, with area-weighted unit MAE."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fit target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fit target length differs from X')
    transformed = transform_target(raw, X)
    weights = sample_weights(X)
    weights_sha = values_sha256(weights)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, cat_features=state['categorical_columns'], weight=weights, thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'],
              'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Fit target changed')
    base.need(values_sha256(weights) == weights_sha == state['fit_sample_weights_sha256']
              and values_sha256(effective_area(X)) == state['fit_effective_area_sha256']
              and values_sha256(_price_factor(X)) == state['fit_factor_sha256'], 'Fit area/weights/factor changed')
    metadata = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'parameters': _copy_json(config), 'parameters_sha256': base.content_sha(config),
        'baseline_parameters': _copy_json(state['baseline_parameters']),
        'removed_feature_names': [],
        'fit_effective_area_sha256': state['fit_effective_area_sha256'],
        'fit_sample_weights_sha256': state['fit_sample_weights_sha256'],
        'fit_factor_sha256': state['fit_factor_sha256'],
        'fit_weight_diagnostics': _copy_json(state['fit_weight_diagnostics']),
        'native_categorical_feature_count': 23,
        'changed_parameter_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 95, 'feature_count': 95, 'categorical_feature_count': 23,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': WEIGHT_FORMULA, 'weight_argument_passed_to_pool': True,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'geo_rule_target_weight_fields_are_provenance_only': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round19_unit_geo.py'],
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
    result = inverse_transform(raw, X)
    base.need(result.shape == (len(X),), 'Prediction rows differ')
    return result


def self_check():
    """Bounded unit-target numerical and FakePool checks, without real data/config/model IO."""
    checks=[]
    def ok(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name+' did not reject')
    idx=pd.Index([9,41,7,23,8,102,6,55],name='synthetic_row')
    cats=['category_'+str(i) for i in range(21)]
    X=pd.DataFrame({c:pd.Series(['category']*len(idx),index=idx,dtype=object) for c in cats},index=idx)
    X['building_area']=np.array([100,np.nan,0,-1,np.inf,50,0,0],dtype='float32')
    X['parking_area']=np.array([30,20,0,-1,5,70,np.nan,np.inf],dtype='float32')
    for i in range(64):X['numeric_'+str(i)]=np.arange(len(idx),dtype='float32')
    geo=pd.DataFrame({c:np.full(len(idx),np.nan,dtype='float32') for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(len(idx),dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo=geo[base.FEATURES]
    X_before,geo_before=X.copy(deep=True),geo.copy(deep=True)
    area=np.array([100,20,1,1,5,50,1,1],dtype='float64');factor=area*1e4
    y=pd.Series([1e6,2e6,3e6,0,4e6,5e6,6e6,7e6],index=idx,dtype='float64')
    ok('R1_effective_area_precedence_and_all_fallbacks',np.array_equal(effective_area(X),area))
    ok('exact_positive_float64_weights_not_normalized',np.array_equal(sample_weights(X),area/100)
       and sample_weights(X).dtype==np.dtype('float64') and not np.isclose(sample_weights(X).mean(),1))
    ok('unit_target_and_TWD_inverse_exact_scale',np.array_equal(transform_target(y,X),y.to_numpy()/factor)
       and np.array_equal(inverse_transform(np.ones(len(X)),X),factor))
    z=np.array([2,3,4,5,6,7,8,9],dtype='float64')
    loss=sample_weights(X)*np.abs(transform_target(y,X)-z)
    reference=np.abs(y.to_numpy()-factor*z)/1e6
    ok('unclipped_unit_weighted_L1_equals_TWD_L1_up_to_1e6',np.allclose(loss,reference,rtol=2e-15,atol=1e-13))
    ok('inverse_negative_predictions_clipped_after_factor',np.array_equal(inverse_transform(-z,X),np.zeros(len(X))))
    stats=weight_diagnostics(X)
    ok('fit_denominator_source_counts_are_exclusive',stats['effective_area_source_counts']=={
       'building_area':2,'parking_area_fallback':2,'default_one_fallback':4} and stats['rows']==8)
    aug=augment_features(X,geo);expected=base.augment_features(X,geo)
    ok('original_R6_95_values_dtypes_order_index_exact',aug.equals(expected)
       and aug.dtypes.equals(expected.dtypes) and aug.index.equals(idx))
    ok('95_features_23_categories_no_drop4',len(aug.columns)==95 and categories(X,cats)==base.categories(X,cats))
    ok('state_API_fit_X_only',list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats'])
    rejects('misaligned_target_index_rejected',lambda:transform_target(y.iloc[::-1],X))
    rejects('misaligned_output_index_rejected',lambda:inverse_transform(pd.Series(z,index=idx[::-1]),X))
    rejects('negative_target_rejected',lambda:transform_target(-np.ones(len(X)),X))
    rejects('nonfinite_target_rejected',lambda:transform_target(np.full(len(X),np.nan),X))
    rejects('nonfinite_output_rejected',lambda:inverse_transform(np.full(len(X),np.inf),X))
    rejects('wrong_target_length_rejected',lambda:transform_target(np.ones(len(X)-1),X))
    huge=X.astype({'building_area':'float64'});huge.loc[:,'building_area']=np.finfo('float64').max
    rejects('area_factor_overflow_rejected',lambda:inverse_transform(z,huge))
    tiny=X.astype({'building_area':'float64'});tiny.loc[:,'building_area']=np.nextafter(0.,1.)
    rejects('underflow_zero_weights_rejected',lambda:sample_weights(tiny))
    rejects('unit_target_division_overflow_rejected',lambda:transform_target(y,tiny))
    rejects('TWD_inverse_overflow_rejected',lambda:inverse_transform(np.full(len(X),np.finfo('float64').max),X))
    fake_base={'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,'loss_function':'MAE',
               'boosting_type':'Plain','task_type':'GPU','learning_rate':0.065,
               'one_hot_max_size':8,'max_ctr_complexity':2,'verbose':300}
    saved_base=_copy_json(fake_base)
    class FakePool:
        def __init__(self,data,label=None,**kwargs):self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=_copy_json(kwargs);self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999;self.leaves=np.full(2999,256,dtype=np.uint32)
        def fit(self,pool,**kwargs):
            if kwargs:raise AssertionError('No eval_set/earlystop')
            self.feature_names_=list(pool.data.columns);self.fit_pool=pool
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features']);return self
        def predict(self,pool,**kwargs):
            if pool.label is not None or 'weight' in pool.kwargs:raise AssertionError('Query label/weight forbidden')
            return pool.data['numeric_0'].to_numpy(dtype='float64')-.5
        def get_all_params(self):return dict(self.actual)
        def get_tree_leaf_counts(self):return self.leaves.copy()
        def get_cat_feature_indices(self):return list(self.cat_indices)
    original_model,original_pool=globals()['CatBoostRegressor'],globals()['Pool']
    original_params=base.parameters;original_config_sha=globals()['_base_configuration_sha256']
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:fake_base;globals()['_base_configuration_sha256']=lambda:'a'*64
    try:
        params=parameters()
        ok('all_passed_R6_parameters_exactly_unchanged',params==saved_base)
        params['depth']=99
        ok('baseline_parameters_deepcopy_unchanged',fake_base==saved_base and parameters()['depth']==8)
        model,state,meta=fit_model(X,y,geo,cats)
        ok('fit_Pool_weighted_unit_labels_and_23cats',model.config==saved_base
           and np.array_equal(model.fit_pool.label,y.to_numpy()/factor)
           and np.array_equal(model.fit_pool.kwargs['weight'],area/100)
           and set(model.fit_pool.kwargs)=={'cat_features','thread_count','weight'}
           and model.fit_pool.kwargs['cat_features']==categories(X,cats))
        ok('fit_only_state_no_parameter_or_feature_changes',state['baseline_parameters']==state['parameters']==saved_base
           and state['changed_parameter_names']==state['removed_feature_names']==[]
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        actual=check_model(model,state)
        ok('native_Plain_capacity_six_fields_plus_23cat_indices',actual['actual_boosting_type']=='Plain'
           and actual['actual_model_depth']==8 and actual['actual_trees']==2999 and len(actual)==6
           and actual['maximum_tree_leaf_count']==256 and len(model.get_cat_feature_indices())==23)
        ok('metadata_unit_weight_contract_and_hashes',meta['target_mode']=='unit'
           and meta['sample_weight']==WEIGHT_FORMULA and meta['weight_argument_passed_to_pool'] is True
           and meta['fit_effective_area_sha256']==values_sha256(area)
           and meta['fit_sample_weights_sha256']==values_sha256(area/100)
           and meta['fit_factor_sha256']==values_sha256(factor)
           and meta['fit_weight_diagnostics']==stats and meta['native_categorical_feature_count']==23
           and meta['fit_transformed_target_values_sha256']==values_sha256(y.to_numpy()/factor))
        restored=_copy_json(state)
        loaded=FakeModel(**_copy_json(model.config));loaded.feature_names_=list(model.feature_names_);loaded.cat_indices=list(model.cat_indices)
        pred=predict_model(loaded,restored,X,geo,cats)
        ok('JSON_state_simulated_reload_unit_to_TWD_exact',np.array_equal(pred,predict_model(model,state,X,geo,cats))
           and pred.dtype==np.dtype('float64') and np.array_equal(pred,np.maximum((np.arange(len(X))-.5)*factor,0)))
        ok('nested_R6_geo_target_and_weight_are_provenance_not_active_policy',
           meta['geo_rule_target_weight_fields_are_provenance_only'] is True
           and FEATURE_RULES['geo_rule_target_weight_fields_are_provenance_only'] is True
           and meta['sample_weight']==WEIGHT_FORMULA and meta['target_mode']=='unit')
        ok('metadata_no_trend_calibration_cap_or_query_labels',all(meta[k] is False for k in
           ('calibration_applied','extra_price_cap_applied','reference_or_trend_applied_here','query_labels_used','fit_uses_eval_set','early_stopping_used')))
        loaded.cat_indices=loaded.cat_indices[:-1]
        rejects('native_wrong_category_count_rejected',lambda:check_model(loaded,restored))
        loaded.cat_indices=list(model.cat_indices);loaded.cat_indices[0]=50
        rejects('native_wrong_category_positions_rejected',lambda:check_model(loaded,restored))
        loaded.cat_indices=list(model.cat_indices);loaded.actual['boosting_type']='Ordered'
        rejects('native_Ordered_rejected',lambda:check_model(loaded,restored))
        loaded.actual['boosting_type']='Plain';loaded.actual['depth']=6
        rejects('native_depth6_rejected',lambda:check_model(loaded,restored))
        loaded.actual['depth']=8;loaded.leaves[0]=257
        rejects('leaf_over256_rejected',lambda:check_model(loaded,restored))
        bad=_copy_json(state);bad['removed_feature_names']=['project_name']
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        rejects('rehashed_unexpected_feature_removal_rejected',lambda:check_state(bad,X,cats))
        ok('original_X_geo_and_baseline_unchanged',X.equals(X_before) and geo.equals(geo_before)
           and X.dtypes.equals(X_before.dtypes) and geo.dtypes.equals(geo_before.dtypes) and fake_base==saved_base)
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=original_model,original_pool
        base.parameters=original_params;globals()['_base_configuration_sha256']=original_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),'synthetic_only':True,
            'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'files_written':False,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=False),flush=True)
