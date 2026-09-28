"""R91: signed additive total-price residual on a fit-only group-OOF unit prior.

The driver binds train prior to same-window R60 group OOF and query prior to
that window's full-fit state. This helper cannot establish that provenance from
an arbitrary vector. It neither fits priors nor loads course/query labels.
CLI self-check replaces CatBoost and Pool with synthetic fakes: no real fit.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round79_recency_unit as baseline

original = baseline.original
base = baseline.base
COMPONENT = 'geography_recency_prior_total_residual_catboost'
TARGET_MODE = 'signed_additive_total_residual_million'
OFFSET_FORMULA = 'effective_area(X) * prior_TWD_per_m2'
TARGET_FORMULA = '(y_TWD - effective_area(X) * fit_group_OOF_prior_TWD_per_m2) / 1e6'
INVERSE_FORMULA = 'max(effective_area(X) * query_full_fit_prior_TWD_per_m2 + 1e6 * signed_z, 0)'
WEIGHT_FORMULA = 'float64(saved_R63_train_sample_weight_float32); no area multiplier or normalization'
LOSS_EQUIVALENCE = 'date_weight*abs((y-B)/1e6-z) = date_weight*abs(y-(B+1e6*z))/1e6 before final nonnegative projection'
parameters = baseline.parameters
effective_area = original.effective_area
augment_features = original.augment_features
categories = original.categories
values_sha256 = original.values_sha256
FEATURE_RULES = {
    'version': 1, 'component': COMPONENT, 'feature_count': 95,
    'categorical_feature_count': 23, 'existing_95_values_dtypes_order_and_index_unchanged': True,
    'added_feature_names': [], 'removed_feature_names': [], 'prior_as_feature': False,
    'target_mode': TARGET_MODE, 'target_formula': TARGET_FORMULA,
    'target_domain': 'finite signed million TWD residual', 'offset_formula': OFFSET_FORMULA,
    'inverse_formula': INVERSE_FORMULA, 'sample_weight': WEIGHT_FORMULA,
    'loss_equivalence': LOSS_EQUIVALENCE, 'effective_area_rule': original.AREA_RULE,
    'date_weights_recomputed': False, 'date_weights_renormalized': False,
    'weights_normalized_to_mean_one': False, 'area_multiplied_into_weights': False,
    'fit_prior_required_source': 'same-window R60 group-OOF unit prior',
    'query_prior_required_source': 'same-window R60 full-fit unit prior',
    'prior_provenance_verified_in_helper': False, 'prior_fitted_in_helper': False,
    'legacy_state_training_fields_are_provenance_only': True,
    'signed_z_clipped_before_offset': False, 'nonnegative_projection_after_offset': True,
    'query_labels_used': False, 'no_eval_set': True, 'no_early_stopping': True,
    'nonfinite_or_overflow': 'raise ValueError',
}


def source_hashes():
    return {**baseline.source_hashes(), 'src/round91_prior_residual.py': original.sha256(Path(__file__))}


def prior_values(prior, X):
    values = original._aligned_vector(prior, X, 'Unit prior')
    base.need((values > 0).all(), 'Unit prior must be finite and strictly positive')
    return values


def offset_values(X, prior):
    try:
        with np.errstate(over='raise', invalid='raise'):
            values = effective_area(X) * prior_values(prior, X)
    except FloatingPointError as exc:
        raise ValueError('Prior total-price offset overflowed') from exc
    base.need(np.isfinite(values).all() and (values > 0).all(), 'Offset must be finite and positive')
    return np.ascontiguousarray(values, dtype=np.float64)


def transform_target(y, X, train_prior):
    raw = original._aligned_vector(y, X, 'Fit target')
    base.need((raw >= 0).all(), 'Fit total prices must be nonnegative')
    try:
        with np.errstate(over='raise', invalid='raise'):
            transformed = (raw - offset_values(X, train_prior)) / 1e6
    except FloatingPointError as exc:
        raise ValueError('Signed residual target overflowed') from exc
    base.need(np.isfinite(transformed).all(), 'Signed residual target must be finite')
    return np.ascontiguousarray(transformed, dtype=np.float64)


def inverse_transform(z, X, query_prior):
    raw = original._aligned_vector(z, X, 'Signed model output')
    try:
        with np.errstate(over='raise', invalid='raise'):
            total = offset_values(X, query_prior) + 1e6 * raw
    except FloatingPointError as exc:
        raise ValueError('Residual total-price reconstruction overflowed') from exc
    base.need(np.isfinite(total).all(), 'Reconstructed total must be finite before projection')
    return np.maximum(total, 0.).astype(np.float64, copy=False)


def sample_weights(X, saved_date_weights):
    original._check_X(X)
    dates32 = baseline._date_values(saved_date_weights, X)
    return np.ascontiguousarray(dates32, dtype=np.float64)


def _date_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes(order='C')).hexdigest()


def _check_wrapper(state):
    base.need(state.get('state_content_sha256') == base.content_sha({
        k:v for k,v in state.items() if k != 'state_content_sha256'}), 'Residual wrapper state changed')
    base.need(state.get('format_version') == 1 and state.get('component') == COMPONENT,
              'Expected R91 residual wrapper, not an old unit feature state')
    base.need(state['feature_rules'] == FEATURE_RULES and state['target_mode'] == TARGET_MODE
              and state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA
              and state['offset_formula'] == OFFSET_FORMULA and state['sample_weight'] == WEIGHT_FORMULA,
              'Residual target, inverse, offset or weight rules changed')
    base.need(state['source_sha256'] == source_hashes() and state['parameters'] == parameters(),
              'Residual source or parameters changed')
    legacy = state['legacy_feature_state']
    base.need(state['base_configuration_sha256'] == legacy['base_configuration_sha256']
              and state['all_feature_names'] == state['feature_names'] == legacy['all_feature_names']
              and len(state['all_feature_names']) == 95
              and state['categorical_columns'] == legacy['categorical_columns']
              and len(state['categorical_columns']) == 23
              and state['feature_dtypes'] == legacy['fit_model_feature_dtypes'], 'Original feature contract changed')
    base.need(state['added_feature_names'] == state['removed_feature_names'] == state['changed_parameter_names'] == []
              and state['prior_as_feature'] is False and state['query_labels_used'] is False
              and state['prior_estimated_in_helper'] is False, 'Unexpected features, prior fit or query labels')
    return legacy


def check_state(state, X, cats):
    legacy = _check_wrapper(state)
    baseline.check_state(legacy, X, cats)
    base.need(state['original_categorical_columns'] == list(cats), 'Original categories changed')
    return True


def check_model(model, state):
    legacy = _check_wrapper(state)
    actual_report = baseline.check_model(model, legacy)
    actual = model.get_all_params()
    fixed = {'depth': 8, 'grow_policy': 'SymmetricTree', 'boosting_type': 'Plain',
             'random_seed': 42, 'loss_function': 'MAE', 'eval_metric': 'MAE',
             'bootstrap_type': 'Bernoulli', 'one_hot_max_size': 8,
             'max_ctr_complexity': 2, 'border_count': 128}
    base.need(all(actual.get(k) == v for k,v in fixed.items()), 'Native residual fixed recipe differs')
    floats = {'learning_rate': .065, 'l2_leaf_reg': 6., 'subsample': .85, 'random_strength': .5}
    base.need(all(np.isfinite(float(actual[k])) and np.isclose(float(actual[k]),v,rtol=1e-7,atol=0)
                  for k,v in floats.items()), 'Native residual float parameters differ')
    return {**actual_report, 'native_feature_count': 95, 'native_categorical_feature_count': 23}


def fit_model(X, y, geo, cats, saved_date_weights, train_prior):
    """One native fit; caller must prove OOF/date source identity before calling."""
    started = time.monotonic()
    original._check_X(X); base.need(len(X) > 0, 'Cannot fit empty X')
    raw = original._aligned_vector(y, X, 'Fit target')
    prior = prior_values(train_prior, X)
    offset = offset_values(X, prior)
    transformed = transform_target(raw, X, prior)
    dates32 = baseline._date_values(saved_date_weights, X)
    weights = sample_weights(X, dates32)
    legacy = baseline.fit_feature_state(X, geo, cats)
    frame = augment_features(X, geo)
    config = parameters()
    state = {
        'format_version': 1, 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'offset_formula': OFFSET_FORMULA, 'sample_weight': WEIGHT_FORMULA,
        'feature_rules': original._copy_json(FEATURE_RULES), 'legacy_feature_state': legacy,
        'legacy_feature_state_pure_X_95_columns_only': True,
        'legacy_state_training_fields_are_provenance_only': True,
        'source_sha256': source_hashes(), 'parameters': original._copy_json(config),
        'base_configuration_sha256': legacy['base_configuration_sha256'],
        'original_categorical_columns': list(cats), 'categorical_columns': list(legacy['categorical_columns']),
        'all_feature_names': list(frame.columns), 'feature_names': list(frame.columns),
        'feature_dtypes': {c:str(frame[c].dtype) for c in frame},
        'fit_rows': len(X), 'fit_model_feature_values_sha256': base.frame_sha(frame),
        'fit_prior_values_sha256': values_sha256(prior), 'fit_offset_TWD_sha256': values_sha256(offset),
        'fit_date_weights_sha256': _date_sha(dates32), 'fit_sample_weights_sha256': values_sha256(weights),
        'fit_target_values_sha256': values_sha256(raw),
        'fit_transformed_target_values_sha256': values_sha256(transformed),
        'fit_effective_area_sha256': values_sha256(effective_area(X)),
        'fit_prior_uses_fit_labels': True, 'fit_wrapper_contains_target_derived_identities': True,
        'query_labels_used': False, 'prior_provenance_verified_in_helper': False,
        'prior_estimated_in_helper': False, 'prior_as_feature': False,
        'added_feature_names': [], 'removed_feature_names': [], 'changed_parameter_names': [],
    }
    state['state_content_sha256'] = base.content_sha(state)
    # Label and weight argument identities below are the actual Pool inputs.
    model = CatBoostRegressor(**config)
    model.fit(Pool(frame, transformed, weight=weights,
                   cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config and base.frame_sha(augment_features(X,geo)) == state['fit_model_feature_values_sha256']
              and base.frame_sha(X) == legacy['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == legacy['geo_state']['fit_geo_feature_values_sha256'], 'Fit features/config changed')
    for name, values in [('fit_target_values_sha256', original._aligned_vector(y,X,'Fit target')),
                         ('fit_transformed_target_values_sha256', transformed),
                         ('fit_prior_values_sha256', prior_values(train_prior,X)),
                         ('fit_offset_TWD_sha256', offset_values(X,train_prior)),
                         ('fit_sample_weights_sha256', weights),
                         ('fit_effective_area_sha256', effective_area(X))]:
        base.need(values_sha256(values) == state[name], 'Fit values changed: ' + name)
    base.need(_date_sha(baseline._date_values(saved_date_weights,X)) == state['fit_date_weights_sha256']
              and values_sha256(sample_weights(X,saved_date_weights)) == state['fit_sample_weights_sha256'], 'Date weights changed')
    report = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA, 'offset_formula': OFFSET_FORMULA,
        'prediction_units': 'TWD', 'raw_output_units': 'signed million TWD residual',
        'parameters': original._copy_json(config), 'baseline_parameters': original._copy_json(legacy['baseline_parameters']),
        'changed_parameter_names': [], 'added_feature_names': [], 'removed_feature_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'], 'seed': 42,
        'fit_rows': len(X), 'feature_count': 95, 'categorical_feature_count': 23,
        'feature_columns': list(frame.columns), 'categorical_columns': list(state['categorical_columns']),
        'sample_weight': WEIGHT_FORMULA, 'weight_argument_passed_to_pool': True,
        'weight_dtype': 'float64', 'weights_normalized_to_mean_one': False,
        'area_multiplied_into_weights': False, 'date_weights_recomputed': False, 'date_weights_renormalized': False,
        'date_weight_provenance_verified_in_helper': False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks': True,
        'fit_date_weights_summary': baseline._summary(dates32), 'fit_sample_weights_summary': baseline._summary(weights),
        'fit_date_weights_hash_encoding': 'ordered little-endian float32',
        'fit_sample_weights_hash_encoding': 'ordered little-endian float64',
        'train_prior_required_source': 'R60 same-window group-OOF unit prior',
        'query_prior_required_source': 'R60 same-window full-fit unit prior',
        'prior_provenance_verified_in_helper': False, 'prior_uses_fit_labels': True,
        'prior_estimated_in_helper': False, 'prior_as_feature': False, 'query_labels_used': False,
        'signed_residual_negative_rows': int((transformed < 0).sum()),
        'signed_residual_zero_rows': int((transformed == 0).sum()),
        'signed_residual_positive_rows': int((transformed > 0).sum()),
        'loss_equivalence_before_final_nonnegative_projection': LOSS_EQUIVALENCE,
        'signed_z_clipped_before_offset': False, 'nonnegative_projection_after_offset': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'legacy_feature_state_content_sha256': legacy['state_content_sha256'],
        'legacy_state_training_fields_are_provenance_only': True,
        'fit_original_feature_values_sha256': legacy['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': legacy['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256': state['fit_model_feature_values_sha256'],
        'feature_rules': original._copy_json(FEATURE_RULES), 'source_sha256': source_hashes(),
        'base_configuration_sha256': legacy['base_configuration_sha256'],
        'weights_use_fit_target': False, 'weights_use_query': False,
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'reference_or_trend_applied_here': False, 'spatial_applied_here': False,
        'extra_price_cap_applied': False, 'calibration_applied': False,
        'elapsed_seconds': time.monotonic()-started, **actual,
    }
    for k in ('fit_prior_values_sha256','fit_offset_TWD_sha256','fit_date_weights_sha256',
              'fit_sample_weights_sha256','fit_target_values_sha256','fit_transformed_target_values_sha256',
              'fit_effective_area_sha256'):
        report[k] = state[k]
    return model, state, report


def predict_model(model, state, X, geo, cats, query_prior):
    """Return signed residual plus supplied fit-only offset, then project once."""
    check_state(state, X, cats); check_model(model, state)
    prior_values(query_prior, X)
    frame = augment_features(X, geo)
    base.need(list(frame.columns) == state['all_feature_names']
              and {c:str(frame[c].dtype) for c in frame} == state['feature_dtypes'], 'Query feature schema changed')
    raw = model.predict(Pool(frame, cat_features=state['categorical_columns'], thread_count=4), thread_count=4)
    return inverse_transform(raw, X, query_prior)



def self_check():
    """Bounded FakePool/FakeModel checks; no actual training/config/data reads."""
    checks = []
    def ok(name, value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def reject(name, operation):
        try: operation()
        except (ValueError, KeyError): checks.append(name)
        else: raise AssertionError(name + ' did not reject')
    idx = pd.Index([9,41,7,23,8,102,6,55], name='synthetic_row')
    cats = ['category_'+str(i) for i in range(21)]
    X = pd.DataFrame({c:pd.Series(['category']*len(idx),index=idx,dtype=object) for c in cats},index=idx)
    X['building_area'] = np.array([100,np.nan,0,-1,np.inf,50,0,0],dtype=np.float32)
    X['parking_area'] = np.array([30,20,0,-1,5,70,np.nan,np.inf],dtype=np.float32)
    for i in range(64): X['numeric_'+str(i)] = np.arange(len(idx),dtype=np.float32)
    geo = pd.DataFrame({c:np.full(len(idx),np.nan,dtype=np.float32) for c in base.NUMERIC},index=idx)
    geo['geo_has_position'] = np.zeros(len(idx),dtype=np.float32)
    geo['geo_match_status'] = pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002'] = pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo = geo[base.FEATURES]
    beforeX,beforeG = X.copy(deep=True),geo.copy(deep=True)
    area = np.array([100,20,1,1,5,50,1,1],dtype=np.float64)
    pri = pd.Series([20000,100000,1e6,2e6,300000,20000,8e6,1e6],index=idx,dtype=np.float64)
    y = pd.Series([1e6,2e6,3e6,0,4e6,5e6,6e6,7e6],index=idx,dtype=np.float64)
    w32 = pd.Series([.125,.25,.5,1.,2.,4.,1.5,.75],index=idx,dtype=np.float32)
    offset = area*pri.to_numpy(); z = np.array([-.5,-3,-2,-2,-2,-2,-2,-2.])
    target = (y.to_numpy()-offset)/1e6
    ok('unchanged_effective_area_all_fallbacks',np.array_equal(effective_area(X),area))
    ok('signed_target_has_negative_zero_positive_values',np.array_equal(transform_target(y,X,pri),target)
       and (target<0).any() and (target==0).any() and (target>0).any())
    ok('offset_is_area_times_prior_in_TWD',np.array_equal(offset_values(X,pri),offset))
    ok('date_only_float32_to_float64_weights_without_area_or_renormalization',
       np.array_equal(sample_weights(X,w32),w32.to_numpy(dtype=np.float64))
       and sample_weights(X,w32).dtype==np.dtype('float64')
       and not np.array_equal(sample_weights(X,w32),area/100*w32.to_numpy())
       and not np.isclose(sample_weights(X,w32).mean(),1.))
    ok('unprojected_date_residual_L1_equals_date_total_L1',np.allclose(
       sample_weights(X,w32)*np.abs(target-z),
       sample_weights(X,w32)*np.abs(y.to_numpy()-(offset+1e6*z))/1e6,rtol=1e-15,atol=1e-14))
    inverse = inverse_transform(z,X,pri)
    ok('signed_z_not_clipped_before_adding_offset',inverse[0]==1.5e6 and inverse[1]==0.
       and np.array_equal(inverse,np.maximum(offset+1e6*z,0)))
    ok('target_inverse_roundtrip_nonnegative_prices',np.array_equal(inverse_transform(target,X,pri),y.to_numpy()))
    ok('residual_inverse_differs_from_old_unit_inverse',not np.array_equal(inverse,original.inverse_transform(z,X)))
    reject('misordered_prior_index_rejected',lambda:prior_values(pri.iloc[::-1],X))
    reject('misordered_label_index_rejected',lambda:transform_target(y.iloc[::-1],X,pri))
    reject('misordered_signed_output_index_rejected',lambda:inverse_transform(pd.Series(z,index=idx[::-1]),X,pri))
    reject('misordered_date_index_rejected',lambda:sample_weights(X,w32.iloc[::-1]))
    reject('wrong_prior_length_rejected',lambda:prior_values(pri.to_numpy()[:-1],X))
    reject('matrix_prior_rejected',lambda:prior_values(pri.to_numpy()[:,None],X))
    reject('zero_prior_rejected',lambda:prior_values(np.zeros(len(X)),X))
    reject('negative_prior_rejected',lambda:prior_values(-pri,X))
    reject('nonfinite_prior_rejected',lambda:prior_values(np.full(len(X),np.nan),X))
    reject('negative_raw_price_rejected_but_signed_target_allowed',lambda:transform_target(-np.ones(len(X)),X,pri))
    reject('nonfinite_label_rejected',lambda:transform_target(np.full(len(X),np.inf),X,pri))
    reject('nonfinite_signed_z_rejected',lambda:inverse_transform(np.full(len(X),np.inf),X,pri))
    reject('date_weight_float64_input_rejected',lambda:sample_weights(X,w32.to_numpy(dtype=np.float64)))
    reject('date_weight_zero_rejected',lambda:sample_weights(X,np.zeros(len(X),dtype=np.float32)))
    huge = X.astype({'building_area':'float64'});huge.loc[:,'building_area']=np.finfo(np.float64).max
    reject('offset_overflow_rejected',lambda:offset_values(huge,pri))
    reject('signed_reconstruction_overflow_rejected',lambda:inverse_transform(np.full(len(X),np.finfo(np.float64).max),X,pri))
    aug = augment_features(X,geo)
    expected = baseline.augment_features(X,geo)
    ok('95_features_values_dtypes_order_index_exact_no_prior_column',aug.equals(expected)
       and aug.dtypes.equals(expected.dtypes) and len(aug.columns)==95 and aug.index.equals(idx))
    reject('geography_index_misalignment_rejected',lambda:augment_features(X,geo.iloc[::-1]))
    reject('target_column_in_X_rejected',lambda:augment_features(X.assign(y=1.),geo))
    ok('augment_API_cannot_receive_prior',list(inspect.signature(augment_features).parameters)==['X','geo'])
    ok('predict_API_prior_but_no_query_labels',list(inspect.signature(predict_model).parameters)
       ==['model','state','X','geo','cats','query_prior'])
    fake_config = {'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,'loss_function':'MAE',
        'eval_metric':'MAE','boosting_type':'Plain','task_type':'GPU','learning_rate':.065,
        'one_hot_max_size':8,'max_ctr_complexity':2,'verbose':300,'bootstrap_type':'Bernoulli',
        'border_count':128,'l2_leaf_reg':6.,'subsample':.85,'random_strength':.5}
    class FakePool:
        def __init__(self,data,label=None,**kwargs):self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=original._copy_json(kwargs);self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999;self.leaves=np.full(2999,256,dtype=np.uint32)
        def fit(self,pool,**kwargs):
            if kwargs:raise AssertionError('eval_set or early stopping supplied')
            self.pool=pool;self.feature_names_=list(pool.data.columns)
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self,pool,**kwargs):
            if pool.label is not None or 'weight' in pool.kwargs:raise AssertionError('query labels/weights supplied')
            return pool.data['numeric_0'].to_numpy(dtype=np.float64)-.5
        def get_all_params(self):return dict(self.actual)
        def get_tree_leaf_counts(self):return self.leaves.copy()
        def get_cat_feature_indices(self):return list(self.cat_indices)
    old_model,old_pool = globals()['CatBoostRegressor'],globals()['Pool']
    old_params,old_config_sha = base.parameters,original._base_configuration_sha256
    globals()['CatBoostRegressor'],globals()['Pool'] = FakeModel,FakePool
    base.parameters=lambda:original._copy_json(fake_config)
    original._base_configuration_sha256=lambda:'a'*64
    try:
        model,state,meta = fit_model(X,y,geo,cats,w32,pri)
        ok('actual_FakePool_signed_labels_date_only_weights_and_23_categories',
           np.array_equal(model.pool.label,target) and np.array_equal(model.pool.kwargs['weight'],w32.to_numpy(dtype=np.float64))
           and (model.pool.label<0).any() and set(model.pool.kwargs)=={'weight','cat_features','thread_count'}
           and len(model.pool.kwargs['cat_features'])==23 and model.config==fake_config)
        ok('actual_FakePool_95_columns_no_prior_feature',model.pool.data.equals(aug) and len(model.feature_names_)==95)
        legacy = original.fit_feature_state(X,geo,cats)
        ok('legacy_R19_feature_state_exact_and_own_target_identity',state['legacy_feature_state']==legacy
           and state['target_mode']==TARGET_MODE and legacy['target_mode']=='unit')
        ok('fit_metadata_records_signed_offset_and_date_weight_hashes',
           meta['fit_offset_TWD_sha256']==values_sha256(offset)
           and meta['fit_transformed_target_values_sha256']==values_sha256(target)
           and meta['fit_sample_weights_sha256']==values_sha256(w32.to_numpy(dtype=np.float64))
           and meta['fit_date_weights_sha256']==_date_sha(w32)
           and meta['signed_residual_negative_rows']==int((target<0).sum())
           and not meta['prior_provenance_verified_in_helper'] and not meta['date_weight_provenance_verified_in_helper'])
        restored=original._copy_json(state)
        loaded=FakeModel(**model.config);loaded.feature_names_=list(model.feature_names_);loaded.cat_indices=list(model.cat_indices)
        pred=predict_model(model,state,X,geo,cats,pri);replay=predict_model(loaded,restored,X,geo,cats,pri)
        ok('JSON_state_simulated_native_reload_signed_raw_plus_offset_exact',
           np.array_equal(pred,replay) and np.array_equal(pred,np.maximum(offset+1e6*(np.arange(len(X))-.5),0)))
        second=predict_model(model,state,X,geo,cats,pri*2)
        ok('query_prior_changes_only_offset_not_model_features',np.array_equal(second,np.maximum(2*offset+1e6*(np.arange(len(X))-.5),0)))
        reject('old_unit_state_cannot_enter_residual_predict',lambda:predict_model(model,legacy,X,geo,cats,pri))
        bad=original._copy_json(state);bad['inverse_formula']=original.INVERSE_FORMULA
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        reject('rehashed_old_unit_inverse_rejected',lambda:check_state(bad,X,cats))
        loaded.actual['random_seed']=137
        reject('native_seed_change_rejected',lambda:check_model(loaded,state))
        loaded.actual['random_seed']=42;loaded.feature_names_.append('prior_feature')
        reject('native_96_feature_model_rejected',lambda:check_model(loaded,state))
        ok('original_X_geo_y_prior_date_inputs_unchanged',X.equals(beforeX) and geo.equals(beforeG)
           and values_sha256(pri)==state['fit_prior_values_sha256'] and _date_sha(w32)==state['fit_date_weights_sha256'])
        ok('no_eval_set_no_earlystop_no_prior_fit_no_query_labels',all(meta[k] is False for k in
           ['fit_uses_eval_set','early_stopping_used','prior_estimated_in_helper','query_labels_used']))
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=old_model,old_pool
        base.parameters=old_params;original._base_configuration_sha256=old_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),'synthetic_only':True,
        'real_data_loaded':False,'real_model_training_performed':False,'real_model_or_configuration_loaded':False,
        'fake_native_fit_calls':1,'new_real_native_fits':0,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    tick=time.monotonic();receipt=self_check();receipt['elapsed_seconds']=time.monotonic()-tick
    receipt['exit_code']=0
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
