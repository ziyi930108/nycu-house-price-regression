"""R92: fixed R79 date-area weighted unit CatBoost with only depth 8 -> 6.

No data loader, date-weight estimation, query labels or corrections. The driver
must bind saved float32 date weights to the same-window R63 provenance. The
original R19 pure-X feature state is embedded unchanged, never rewritten.
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
COMPONENT = 'geography_recency_shallow_unit_catboost'
TARGET_MODE = original.TARGET_MODE
TARGET_FORMULA = original.TARGET_FORMULA
INVERSE_FORMULA = original.INVERSE_FORMULA
WEIGHT_FORMULA = baseline.WEIGHT_FORMULA
effective_area = original.effective_area
transform_target = original.transform_target
inverse_transform = original.inverse_transform
augment_features = original.augment_features
categories = original.categories
values_sha256 = original.values_sha256
FEATURE_RULES = original._copy_json(baseline.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT, depth=6, baseline_depth=8,
    changed_parameter_names=['depth'], maximum_allowed_tree_leaf_count=64,
    legacy_state_parameter_and_training_fields_are_provenance_only=True)


def source_hashes():
    return {**baseline.source_hashes(),
            'src/round92_shallow_unit.py': original.sha256(Path(__file__))}


def parameters():
    old = baseline.parameters()
    base.need(old['depth'] == 8 and old['iterations'] == 2999
              and old['learning_rate'] == .065 and old['random_seed'] == 42,
              'Expected original fixed R79 recipe')
    result = original._copy_json(old)
    result['depth'] = 6
    base.need(set(result) == set(old)
              and [k for k in result if result[k] != old[k]] == ['depth'],
              'Only depth may change from R79')
    return result


def sample_weights(X, saved_date_weights):
    original._check_X(X)
    dates = baseline._date_values(saved_date_weights, X)
    try:
        with np.errstate(over='raise', invalid='raise'):
            weights = original.sample_weights(X) * dates.astype(np.float64)
    except FloatingPointError as exc:
        raise ValueError('Area-date Pool weights overflowed') from exc
    base.need(np.isfinite(weights).all() and (weights > 0).all(),
              'Combined Pool weights must be finite and positive')
    return np.ascontiguousarray(weights, dtype=np.float64)


def _date_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes(order='C')).hexdigest()


def _check_wrapper(state):
    base.need(state.get('state_content_sha256') == base.content_sha({
        k:v for k,v in state.items() if k != 'state_content_sha256'}), 'R92 wrapper content changed')
    base.need(state.get('format_version') == 1 and state.get('component') == COMPONENT,
              'Expected independent R92 wrapper')
    base.need(state['feature_rules'] == FEATURE_RULES and state['target_mode'] == TARGET_MODE
              and state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA
              and state['sample_weight'] == WEIGHT_FORMULA, 'R92 target/weight/feature rules changed')
    base.need(state['source_sha256'] == source_hashes() and state['parameters'] == parameters()
              and state['baseline_parameters'] == baseline.parameters()
              and state['changed_parameter_names'] == ['depth'], 'R92 source or parameters changed')
    legacy = state['legacy_feature_state']
    base.need(state['base_configuration_sha256'] == legacy['base_configuration_sha256']
              and state['all_feature_names'] == state['feature_names'] == legacy['all_feature_names']
              and len(state['all_feature_names']) == 95
              and state['categorical_columns'] == legacy['categorical_columns']
              and len(state['categorical_columns']) == 23
              and state['feature_dtypes'] == legacy['fit_model_feature_dtypes'],
              'Original feature contract changed')
    base.need(state['added_feature_names'] == state['removed_feature_names'] == []
              and state['query_labels_used'] is False, 'Unexpected features or query labels')
    return legacy


def check_state(state, X, cats):
    legacy = _check_wrapper(state)
    original.check_state(legacy, X, cats)
    base.need(state['original_categorical_columns'] == list(cats), 'Original categories changed')
    return True


def check_model(model, state):
    """Validate native depth-6 capacity directly; never call the depth-8 checker."""
    _check_wrapper(state)
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    cat_indices = np.asarray(model.get_cat_feature_indices())
    expected = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(cat_indices.ndim == 1 and cat_indices.dtype.kind in 'iu'
              and len(expected) == len(set(expected)) == 23 and cat_indices.tolist() == expected,
              'Native 23-category indices differ')
    actual = model.get_all_params()
    fixed = {'depth':6, 'grow_policy':'SymmetricTree', 'boosting_type':'Plain',
             'iterations':2999, 'random_seed':42, 'loss_function':'MAE', 'eval_metric':'MAE',
             'bootstrap_type':'Bernoulli', 'one_hot_max_size':8,
             'max_ctr_complexity':2, 'border_count':128}
    base.need(all(actual.get(k) == v for k,v in fixed.items()), 'Native shallow recipe differs')
    floats = {'learning_rate':.065, 'l2_leaf_reg':6., 'subsample':.85, 'random_strength':.5}
    base.need(all(np.isfinite(float(actual[k])) and np.isclose(float(actual[k]),v,rtol=1e-7,atol=0)
                  for k,v in floats.items()), 'Native shallow float parameters differ')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 64).all(),
              'Expected 2999 positive integer leaf counts, each at most 64')
    return {'actual_model_depth':int(actual['depth']), 'actual_boosting_type':actual['boosting_type'],
            'actual_grow_policy':actual['grow_policy'], 'actual_trees':int(model.tree_count_),
            'maximum_tree_leaf_count':int(leaves.max()),
            'tree_leaf_counts_sha256':hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest(),
            'actual_learning_rate':float(actual['learning_rate']),
            'native_feature_count':95, 'native_categorical_feature_count':23}


def fit_model(X, y, geo, cats, saved_date_weights):
    """One fit using same-window fit labels and saved float32 date weights only."""
    tick = time.monotonic()
    original._check_X(X)
    base.need(len(X) > 0, 'Cannot fit empty X')
    raw = original._aligned_vector(y, X, 'Fit target')
    transformed = transform_target(raw, X)
    dates32 = baseline._date_values(saved_date_weights, X)
    weights = sample_weights(X, dates32)
    legacy = original.fit_feature_state(X, geo, cats)
    frame = augment_features(X, geo)
    config = parameters()
    state = {
        'format_version':1, 'component':COMPONENT, 'target_mode':TARGET_MODE,
        'target_formula':TARGET_FORMULA, 'inverse_formula':INVERSE_FORMULA,
        'sample_weight':WEIGHT_FORMULA, 'feature_rules':original._copy_json(FEATURE_RULES),
        'legacy_feature_state':legacy, 'legacy_feature_state_pure_X_95_columns_only':True,
        'legacy_state_parameter_and_training_fields_are_provenance_only':True,
        'source_sha256':source_hashes(), 'parameters':original._copy_json(config),
        'baseline_parameters':original._copy_json(legacy['parameters']),
        'base_configuration_sha256':legacy['base_configuration_sha256'],
        'original_categorical_columns':list(cats), 'categorical_columns':list(legacy['categorical_columns']),
        'all_feature_names':list(frame.columns), 'feature_names':list(frame.columns),
        'feature_dtypes':{c:str(frame[c].dtype) for c in frame},
        'fit_rows':len(X), 'fit_model_feature_values_sha256':base.frame_sha(frame),
        'fit_date_weights_sha256':_date_sha(dates32), 'fit_sample_weights_sha256':values_sha256(weights),
        'fit_area_weights_sha256':values_sha256(original.sample_weights(X)),
        'fit_target_values_sha256':values_sha256(raw),
        'fit_transformed_target_values_sha256':values_sha256(transformed),
        'fit_effective_area_sha256':values_sha256(effective_area(X)),
        'fit_factor_sha256':values_sha256(original._price_factor(X)),
        'fit_wrapper_contains_target_derived_identities':True,
        'query_labels_used':False, 'added_feature_names':[], 'removed_feature_names':[],
        'changed_parameter_names':['depth'],
    }
    state['state_content_sha256'] = base.content_sha(state)
    model = CatBoostRegressor(**config)
    model.fit(Pool(frame, transformed, weight=weights,
                   cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config and base.frame_sha(augment_features(X,geo)) == state['fit_model_feature_values_sha256']
              and base.frame_sha(X) == legacy['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == legacy['geo_state']['fit_geo_feature_values_sha256'], 'Fit inputs/config changed')
    for k,v in [('fit_target_values_sha256',original._aligned_vector(y,X,'Fit target')),
                ('fit_transformed_target_values_sha256',transformed),('fit_sample_weights_sha256',weights),
                ('fit_area_weights_sha256',original.sample_weights(X)),
                ('fit_effective_area_sha256',effective_area(X)),('fit_factor_sha256',original._price_factor(X))]:
        base.need(values_sha256(v) == state[k], 'Fit values changed: '+k)
    base.need(_date_sha(baseline._date_values(saved_date_weights,X)) == state['fit_date_weights_sha256']
              and values_sha256(sample_weights(X,saved_date_weights)) == state['fit_sample_weights_sha256'],
              'Saved date weights changed')
    metadata = {
        'family':'catboost', 'component':COMPONENT, 'target_mode':TARGET_MODE,
        'target_formula':TARGET_FORMULA, 'inverse_formula':INVERSE_FORMULA, 'prediction_units':'TWD',
        'parameters':original._copy_json(config), 'baseline_parameters':original._copy_json(legacy['parameters']),
        'changed_parameter_names':['depth'], 'added_feature_names':[], 'removed_feature_names':[],
        'iterations':2999, 'actual_iterations':actual['actual_trees'], 'seed':42,
        'fit_rows':len(X), 'feature_count':95, 'categorical_feature_count':23,
        'feature_columns':list(frame.columns), 'categorical_columns':list(state['categorical_columns']),
        'sample_weight':WEIGHT_FORMULA, 'weight_argument_passed_to_pool':True, 'weight_dtype':'float64',
        'weights_normalized_to_mean_one':False, 'date_weights_recomputed':False, 'date_weights_renormalized':False,
        'date_weight_provenance_verified_in_helper':False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks':True,
        'fit_date_weights_summary':baseline._summary(dates32), 'fit_sample_weights_summary':baseline._summary(weights),
        'fit_date_weights_hash_encoding':'ordered little-endian float32',
        'fit_sample_weights_hash_encoding':'ordered little-endian float64',
        'loss_equivalence_before_nonnegative_inverse_clipping':
            '(A/100)*date_weight*abs(y/(A*1e4)-z) = date_weight*abs(y-A*1e4*z)/1e6',
        'feature_rules':original._copy_json(FEATURE_RULES),
        'legacy_state_parameter_and_training_fields_are_provenance_only':True,
        'legacy_state_weight_policy':original.WEIGHT_FORMULA,
        'feature_state_content_sha256':state['state_content_sha256'],
        'legacy_feature_state_content_sha256':legacy['state_content_sha256'],
        'fit_original_feature_values_sha256':legacy['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256':legacy['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256':state['fit_model_feature_values_sha256'],
        'source_sha256':source_hashes(), 'base_configuration_sha256':legacy['base_configuration_sha256'],
        'query_labels_used':False, 'weights_use_fit_target':False, 'weights_use_query':False,
        'target_used_for_geographic_state':False, 'query_used_for_geographic_state':False,
        'fit_uses_eval_set':False, 'early_stopping_used':False,
        'reference_or_trend_applied_here':False, 'spatial_applied_here':False,
        'extra_price_cap_applied':False, 'calibration_applied':False,
        'elapsed_seconds':time.monotonic()-tick, **actual,
    }
    for k in ('fit_date_weights_sha256','fit_sample_weights_sha256','fit_area_weights_sha256',
              'fit_target_values_sha256','fit_transformed_target_values_sha256',
              'fit_effective_area_sha256','fit_factor_sha256'):
        metadata[k] = state[k]
    return model, state, metadata


def predict_model(model, state, X, geo, cats):
    check_state(state, X, cats)
    check_model(model, state)
    frame = augment_features(X, geo)
    base.need(list(frame.columns) == state['all_feature_names']
              and {c:str(frame[c].dtype) for c in frame} == state['feature_dtypes'], 'Query schema changed')
    raw = model.predict(Pool(frame,cat_features=state['categorical_columns'],thread_count=4),thread_count=4)
    return inverse_transform(raw,X)


def self_check():
    """Synthetic fake fit only; no data/model/config reads or real-model patch."""
    checks = []
    def ok(name, value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def reject(name, operation):
        try: operation()
        except (ValueError, KeyError): checks.append(name)
        else: raise AssertionError(name+' did not reject')
    idx = pd.Index([9,41,7,23,8,102,6,55],name='synthetic_row')
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
    bx,bg = X.copy(deep=True),geo.copy(deep=True)
    area = np.array([100,20,1,1,5,50,1,1],dtype=np.float64)
    y = pd.Series([1e6,2e6,3e6,0,4e6,5e6,6e6,7e6],index=idx,dtype=np.float64)
    dates = pd.Series([.125,.25,.5,1.,2.,4.,1.5,.75],index=idx,dtype=np.float32)
    target = y.to_numpy()/(area*1e4)
    weights = area/100*dates.to_numpy(dtype=np.float64)
    z = np.arange(len(idx),dtype=np.float64)-.5
    ok('effective_area_all_original_fallbacks',np.array_equal(effective_area(X),area))
    ok('unit_target_and_TWD_inverse_unchanged',np.array_equal(transform_target(y,X),target)
       and np.array_equal(inverse_transform(z,X),np.maximum(z*area*1e4,0)))
    ok('area_times_float32_date_weights_float64_no_renormalization',np.array_equal(sample_weights(X,dates),weights)
       and sample_weights(X,dates).dtype==np.dtype('float64'))
    ok('weighted_unit_L1_matches_date_weighted_total_L1',np.allclose(
       weights*np.abs(target-z),dates.to_numpy(dtype=np.float64)*np.abs(y.to_numpy()-area*1e4*z)/1e6,
       rtol=2e-15,atol=1e-13))
    reject('misordered_date_index_rejected',lambda:sample_weights(X,dates.iloc[::-1]))
    reject('wrong_date_dtype_rejected',lambda:sample_weights(X,dates.astype(np.float64)))
    reject('nonpositive_date_rejected',lambda:sample_weights(X,np.zeros(len(X),dtype=np.float32)))
    reject('misordered_target_rejected',lambda:transform_target(y.iloc[::-1],X))
    reject('nonfinite_target_rejected',lambda:transform_target(np.full(len(X),np.nan),X))
    reject('negative_price_rejected',lambda:transform_target(-np.ones(len(X)),X))
    frame = augment_features(X,geo)
    ok('original95_values_dtypes_order_index_and23categories',frame.equals(original.augment_features(X,geo))
       and frame.dtypes.equals(original.augment_features(X,geo).dtypes) and len(frame.columns)==95
       and len(categories(X,cats))==23)
    reject('misordered_geography_rejected',lambda:augment_features(X,geo.iloc[::-1]))
    ok('query_API_has_no_y_or_date_weights',list(inspect.signature(predict_model).parameters)
       ==['model','state','X','geo','cats'])
    fake_config = {'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,'loss_function':'MAE',
        'eval_metric':'MAE','boosting_type':'Plain','task_type':'GPU','learning_rate':.065,
        'one_hot_max_size':8,'max_ctr_complexity':2,'verbose':300,'bootstrap_type':'Bernoulli',
        'border_count':128,'l2_leaf_reg':6.,'subsample':.85,'random_strength':.5,
        'gpu_ram_part':.75,'allow_writing_files':False}
    class FakePool:
        def __init__(self,data,label=None,**kwargs): self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=original._copy_json(kwargs)
            self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999
            self.leaves=np.full(2999,64,dtype=np.uint32)
        def fit(self,pool,**kwargs):
            if kwargs: raise AssertionError('eval_set or early stopping supplied')
            self.pool=pool
            self.feature_names_=list(pool.data.columns)
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self,pool,**kwargs):
            if pool.label is not None or 'weight' in pool.kwargs: raise AssertionError('query y or weight supplied')
            return pool.data['numeric_0'].to_numpy(dtype=np.float64)-.5
        def get_all_params(self): return dict(self.actual)
        def get_tree_leaf_counts(self): return self.leaves.copy()
        def get_cat_feature_indices(self): return list(self.cat_indices)
    old_model,old_pool=globals()['CatBoostRegressor'],globals()['Pool']
    old_params,old_config_sha=base.parameters,original._base_configuration_sha256
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:original._copy_json(fake_config)
    original._base_configuration_sha256=lambda:'a'*64
    try:
        config=parameters()
        ok('parameters_only_depth8_to6_and_baseline_unmutated',config=={**fake_config,'depth':6}
           and fake_config['depth']==8)
        model,state,meta=fit_model(X,y,geo,cats,dates)
        ok('actual_FakePool_unit_labels_area_date_weights',np.array_equal(model.pool.label,target)
           and np.array_equal(model.pool.kwargs['weight'],weights)
           and set(model.pool.kwargs)=={'weight','cat_features','thread_count'})
        ok('actual_constructor_only_depth_changed',model.config=={**fake_config,'depth':6})
        ok('actual_FakePool95cols23cats',model.pool.data.equals(frame) and len(model.cat_indices)==23)
        legacy=original.fit_feature_state(X,geo,cats)
        ok('legacy_R19_feature_state_exact_own_depth6_wrapper',state['legacy_feature_state']==legacy
           and state['parameters']['depth']==6 and legacy['parameters']['depth']==8)
        ok('metadata_target_weight_and_date_hashes',meta['fit_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(target)
           and meta['fit_sample_weights_sha256']==values_sha256(weights)
           and meta['fit_date_weights_sha256']==_date_sha(dates)
           and meta['changed_parameter_names']==['depth'])
        loaded=FakeModel(**model.config)
        loaded.feature_names_=list(model.feature_names_)
        loaded.cat_indices=list(model.cat_indices)
        restored=original._copy_json(state)
        ok('JSON_wrapper_simulated_model_reload_predictions_exact',
           np.array_equal(predict_model(loaded,restored,X,geo,cats),predict_model(model,state,X,geo,cats))
           and np.array_equal(predict_model(loaded,restored,X,geo,cats),inverse_transform(z,X)))
        for name,key,value in [('depth8','depth',8),('seed','random_seed',137),('lr','learning_rate',.07),
                               ('iterations','iterations',2998),('L2','l2_leaf_reg',7.)]:
            previous=loaded.actual[key];loaded.actual[key]=value
            reject('native_'+name+'_change_rejected',lambda:check_model(loaded,state))
            loaded.actual[key]=previous
        loaded.leaves[0]=65
        reject('native_leaf_count_over64_rejected',lambda:check_model(loaded,state))
        loaded.leaves[0]=64;loaded.cat_indices=loaded.cat_indices[:-1]
        reject('native_category_count_changed_rejected',lambda:check_model(loaded,state))
        loaded.cat_indices=list(model.cat_indices);loaded.feature_names_.append('extra')
        reject('native_feature_count_changed_rejected',lambda:check_model(loaded,state))
        reject('legacy_state_cannot_enter_depth6_predict',lambda:predict_model(model,legacy,X,geo,cats))
        ok('original_inputs_unchanged',X.equals(bx) and X.dtypes.equals(bx.dtypes)
           and geo.equals(bg) and values_sha256(y)==state['fit_target_values_sha256']
           and _date_sha(dates)==state['fit_date_weights_sha256'])
        ok('no_eval_set_no_earlystop_no_query_y_no_corrections',all(meta[k] is False for k in
           ['fit_uses_eval_set','early_stopping_used','query_labels_used','reference_or_trend_applied_here','spatial_applied_here']))
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=old_model,old_pool
        base.parameters=old_params;original._base_configuration_sha256=old_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),'synthetic_only':True,
            'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'fake_native_fit_calls':1,
            'new_real_native_fits':0,'source_sha256':source_hashes()}


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

