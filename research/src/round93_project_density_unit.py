"""R93: fit-project sqrt-count reduction of the original R79 unit Pool weights.

Only fit X and saved same-window float32 dates determine counts, q and c. The
single global c preserves sum(original area-date Pool weights). No loader or
query labels/counts/weights enter prediction. Legacy pure-X state is unchanged.
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
import round5_project_mask as project_identity

original = baseline.original
base = baseline.base
COMPONENT = 'geography_recency_project_density_unit_catboost'
TARGET_MODE = original.TARGET_MODE
TARGET_FORMULA = original.TARGET_FORMULA
INVERSE_FORMULA = original.INVERSE_FORMULA
WEIGHT_FORMULA = 'w0 * q * c; w0=(effective_area/100)*float64(saved_date32); q=named_fit_count**(-0.5), missing=1; c=sum(w0)/sum(w0*q)'
GROUP_RULES = {
    'key_source':'src/round5_project_mask.py::project_keys',
    'key_fields':['address_city','address_district','project_name'],
    'key_normalization':'existing R5 fill/str/strip/sentinels',
    'missing_name_only_is_missing':True,
    'missing_location_named_project_remains_named':True,
    'normalized_literal_pipe_in_any_key_field':'reject',
    'fit_named_count':'count every fit transaction row within full geographic project key',
    'missing_count_sentinel':1,'missing_raw_q':1.0,
    'raw_q':'1/sqrt(float64(named_fit_count)); missing=1',
    'normalization':'c=sum_float64(w0)/sum_float64(w0*q)',
    'normalization_uses_fit_area_and_saved_dates':True,
    'preserves_original_pool_weight_sum':True,
    'missing_final_weight_unchanged':False,
    'q_clipping_or_count_threshold':False,
    'counts_use_target_or_query':False,'query_prediction_uses_counts_or_weights':False,
}
effective_area = original.effective_area
transform_target = original.transform_target
inverse_transform = original.inverse_transform
augment_features = original.augment_features
categories = original.categories
values_sha256 = original.values_sha256
FEATURE_RULES = original._copy_json(baseline.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT, sample_weight=WEIGHT_FORMULA,
    depth=8, baseline_depth=8, changed_parameter_names=[], maximum_allowed_tree_leaf_count=256,
    group_weight_rules=original._copy_json(GROUP_RULES),
    loss_equivalence='date*q*c*abs(y-A*1e4*z)/1e6 before nonnegative inverse projection',
    weights_normalized_to_mean_one=False, original_pool_weight_sum_preserved=True,
    legacy_state_parameter_and_training_fields_are_provenance_only=True)



def source_hashes():
    return {**baseline.source_hashes(),
            'src/round5_project_mask.py':original.sha256(Path(project_identity.__file__)),
            'src/round93_project_density_unit.py': original.sha256(Path(__file__))}


def parameters():
    config = baseline.parameters()
    base.need(config['depth'] == 8 and config['iterations'] == 2999
              and config['learning_rate'] == .065 and config['random_seed'] == 42,
              'Expected unchanged original fixed R79 recipe')
    return original._copy_json(config)


def _counts_sha(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes(order='C')).hexdigest()


def group_factors(X):
    """Fit-only row-aligned (int64 counts, float64 raw_q, JSON summary).

    Missing names use count sentinel 1, not a shared physical project. This API
    cannot infer whether the caller provided fit rows: the driver owns that bind.
    """
    original._check_X(X)
    base.need(len(X)>0,'Cannot count an empty fit frame')
    keys, missing = project_identity.project_keys(X)
    for name in GROUP_RULES['key_fields']:
        normalized = project_identity._normal(X[name])
        base.need(not normalized.str.contains('|',regex=False).any(),
                  'Literal pipe in normalized project-key field: '+name)
    named_counts = keys.loc[~missing].value_counts(dropna=True)
    counts = np.ascontiguousarray(keys.map(named_counts).where(~missing,1).to_numpy(dtype=np.int64))
    base.need(counts.shape==(len(X),) and (counts>0).all(),'Invalid fit project counts')
    raw_q = np.ascontiguousarray(1.0/np.sqrt(counts.astype(np.float64)),dtype=np.float64)
    base.need(np.isfinite(raw_q).all() and (raw_q>0).all() and (raw_q<=1).all(),
              'Invalid raw project factors')
    summary = {
        'rows':len(X),'named_rows':int((~missing).sum()),'missing_name_rows':int(missing.sum()),
        'named_projects':len(named_counts),'named_singleton_rows':int((counts[~missing.to_numpy()] == 1).sum()),
        'max_named_project_count':int(named_counts.max()) if len(named_counts) else 0,
        'missing_count_is_neutral_sentinel_not_group_size':True,
        'counts_sha256':_counts_sha(counts),'raw_q_sha256':values_sha256(raw_q),
        'raw_q_summary':baseline._summary(raw_q),
        'ordered_keys_and_index_sha256':base.frame_sha(keys.to_frame(name='project_key')),
        'ordered_missing_flags_sha256':hashlib.sha256(missing.to_numpy(dtype=np.uint8).tobytes()).hexdigest(),
        'fit_only':True,'uses_target':False,'uses_query':False,
        'rules':original._copy_json(GROUP_RULES),
    }
    return counts,raw_q,summary


def weight_details(X, saved_date_weights):
    """Fit-only (float64 Pool weights, JSON summary); never mean-normalize q."""
    original._check_X(X)
    dates = baseline._date_values(saved_date_weights,X)
    counts,q,groups = group_factors(X)
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            w0 = original.sample_weights(X)*dates.astype(np.float64)
            base.need(np.isfinite(w0).all() and (w0>0).all(),'Original Pool weights invalid')
            weighted_q = w0*q
            total = float(w0.sum(dtype=np.float64))
            reduced_total = float(weighted_q.sum(dtype=np.float64))
            base.need(np.isfinite(total) and np.isfinite(reduced_total)
                      and total>0 and reduced_total>0,'Pool sums invalid')
            c = float(total/reduced_total)
            weights = np.ascontiguousarray(weighted_q*c,dtype=np.float64)
            multipliers = q*c
    except FloatingPointError as exc:
        raise ValueError('Project Pool weight arithmetic overflowed') from exc
    base.need(np.isfinite(c) and c>0 and np.isfinite(weights).all() and (weights>0).all(),
              'Combined Pool weights must be finite and positive')
    base.need(np.isclose(weights.sum(dtype=np.float64),total,rtol=2e-15,atol=0),
              'Original Pool total weight not preserved')
    summary = {
        'normalization_c':c,'original_pool_weight_sum':total,
        'raw_reduced_pool_weight_sum':reduced_total,'new_pool_weight_sum':float(weights.sum(dtype=np.float64)),
        'pool_sum_preserved_rtol':2e-15,'pool_sum_preserved_atol':0,
        'original_pool_weights_sha256':values_sha256(w0),
        'pool_weights_sha256':values_sha256(weights),'raw_q_sha256':values_sha256(q),
        'counts_sha256':_counts_sha(counts),'multipliers_sha256':values_sha256(multipliers),
        'original_pool_weights_summary':baseline._summary(w0),
        'pool_weights_summary':baseline._summary(weights),
        'multiplier_summary':baseline._summary(multipliers),'group_summary':groups,
        'missing_raw_q_is_one_but_final_multiplier_is_c':True,
        'uses_fit_target':False,'uses_query':False,'date_weights_recomputed':False,
        'date_weights_renormalized':False,'q_normalized_by_its_mean':False,
    }
    return weights,summary


def sample_weights(X, saved_date_weights):
    return weight_details(X,saved_date_weights)[0]


def _date_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes(order='C')).hexdigest()


def _check_wrapper(state):
    base.need(state.get('state_content_sha256') == base.content_sha({
        k:v for k,v in state.items() if k != 'state_content_sha256'}), 'R93 wrapper content changed')
    base.need(state.get('format_version') == 1 and state.get('component') == COMPONENT,
              'Expected independent R93 wrapper')
    base.need(state['feature_rules'] == FEATURE_RULES and state['target_mode'] == TARGET_MODE
              and state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA
              and state['sample_weight'] == WEIGHT_FORMULA, 'R93 target/weight/feature rules changed')
    base.need(state['source_sha256'] == source_hashes() and state['parameters'] == parameters()
              and state['baseline_parameters'] == baseline.parameters()
              and state['changed_parameter_names'] == [], 'R93 source or parameters changed')
    weight_summary = state['group_weight_summary']
    base.need(weight_summary['pool_weights_sha256'] == state['fit_sample_weights_sha256']
              and weight_summary['counts_sha256'] == state['fit_project_counts_sha256']
              and weight_summary['raw_q_sha256'] == state['fit_project_raw_q_sha256']
              and weight_summary['normalization_c'] == state['fit_project_normalization_c']
              and weight_summary['original_pool_weights_sha256'] == state['fit_original_pool_weights_sha256']
              and weight_summary['group_summary']['rules'] == GROUP_RULES,
              'R93 fit project weight identities changed')
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
    """Validate unchanged original depth-8 capacity and all fixed parameters."""
    _check_wrapper(state)
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    cat_indices = np.asarray(model.get_cat_feature_indices())
    expected = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(cat_indices.ndim == 1 and cat_indices.dtype.kind in 'iu'
              and len(expected) == len(set(expected)) == 23 and cat_indices.tolist() == expected,
              'Native 23-category indices differ')
    actual = model.get_all_params()
    fixed = {'depth':8, 'grow_policy':'SymmetricTree', 'boosting_type':'Plain',
             'iterations':2999, 'random_seed':42, 'loss_function':'MAE', 'eval_metric':'MAE',
             'bootstrap_type':'Bernoulli', 'one_hot_max_size':8,
             'max_ctr_complexity':2, 'border_count':128}
    base.need(all(actual.get(k) == v for k,v in fixed.items()), 'Native original recipe differs')
    floats = {'learning_rate':.065, 'l2_leaf_reg':6., 'subsample':.85, 'random_strength':.5}
    base.need(all(np.isfinite(float(actual[k])) and np.isclose(float(actual[k]),v,rtol=1e-7,atol=0)
                  for k,v in floats.items()), 'Native original float parameters differ')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 256).all(),
              'Expected 2999 positive integer leaf counts, each at most 256')
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
    weights,weight_summary = weight_details(X, dates32)
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
        'group_weight_summary':weight_summary, 'fit_project_counts_sha256':weight_summary['counts_sha256'],
        'fit_project_raw_q_sha256':weight_summary['raw_q_sha256'],
        'fit_project_normalization_c':weight_summary['normalization_c'],
        'fit_original_pool_weights_sha256':weight_summary['original_pool_weights_sha256'],
        'fit_date_weights_sha256':_date_sha(dates32), 'fit_sample_weights_sha256':values_sha256(weights),
        'fit_area_weights_sha256':values_sha256(original.sample_weights(X)),
        'fit_target_values_sha256':values_sha256(raw),
        'fit_transformed_target_values_sha256':values_sha256(transformed),
        'fit_effective_area_sha256':values_sha256(effective_area(X)),
        'fit_factor_sha256':values_sha256(original._price_factor(X)),
        'fit_wrapper_contains_target_derived_identities':True,
        'query_labels_used':False, 'added_feature_names':[], 'removed_feature_names':[],
        'changed_parameter_names':[],
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
        'changed_parameter_names':[], 'added_feature_names':[], 'removed_feature_names':[],
        'iterations':2999, 'actual_iterations':actual['actual_trees'], 'seed':42,
        'fit_rows':len(X), 'feature_count':95, 'categorical_feature_count':23,
        'group_weight_summary':weight_summary, 'fit_project_counts_sha256':weight_summary['counts_sha256'],
        'fit_project_raw_q_sha256':weight_summary['raw_q_sha256'],
        'fit_project_normalization_c':weight_summary['normalization_c'],
        'fit_original_pool_weights_sha256':weight_summary['original_pool_weights_sha256'],
        'feature_columns':list(frame.columns), 'categorical_columns':list(state['categorical_columns']),
        'sample_weight':WEIGHT_FORMULA, 'weight_argument_passed_to_pool':True, 'weight_dtype':'float64',
        'weights_normalized_to_mean_one':False, 'date_weights_recomputed':False, 'date_weights_renormalized':False,
        'original_pool_weight_sum_preserved':True, 'q_normalized_by_its_mean':False,
        'date_weight_provenance_verified_in_helper':False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks':True,
        'fit_date_weights_summary':baseline._summary(dates32), 'fit_sample_weights_summary':baseline._summary(weights),
        'fit_date_weights_hash_encoding':'ordered little-endian float32',
        'fit_sample_weights_hash_encoding':'ordered little-endian float64',
        'loss_equivalence_before_nonnegative_inverse_clipping':
            '(A/100)*date_weight*q*c*abs(y/(A*1e4)-z) = date_weight*q*c*abs(y-A*1e4*z)/1e6',
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
    """Synthetic FakeModel/Pool only; no data loader, real model or config read."""
    checks=[]
    def ok(name,value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def reject(name,operation):
        try: operation()
        except (ValueError,KeyError): checks.append(name)
        else: raise AssertionError(name+' did not reject')
    idx=pd.Index([9,41,7,23,8,102,6,55,77,13,17,44],name='synthetic_row')
    cats=list(project_identity.IDENTIFIER_COLUMNS)+['address_city','address_district']+['category_'+str(i) for i in range(15)]
    X=pd.DataFrame({c:pd.Series(['category']*len(idx),index=idx,dtype=object) for c in cats},index=idx)
    X['project_name']=['Alpha',' Alpha ','Alpha','Beta','Beta','Alpha','Alpha',None,'','nan','None','__MISSING__']
    X['address_city']='City';X['address_district']='District'
    X.loc[idx[5],'address_city']='OtherCity';X.loc[idx[6],'address_district']='OtherDistrict'
    X['building_area']=np.array([100,np.nan,0,-1,np.inf,50,0,0,80,60,25,200],dtype=np.float32)
    X['parking_area']=np.array([30,20,0,-1,5,70,np.nan,np.inf,20,0,10,0],dtype=np.float32)
    for i in range(64): X['numeric_'+str(i)]=np.arange(len(idx),dtype=np.float32)
    geo=pd.DataFrame({c:np.full(len(idx),np.nan,dtype=np.float32) for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(len(idx),dtype=np.float32)
    geo['geo_match_status']=pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo=geo[base.FEATURES]
    bx,bg=X.copy(deep=True),geo.copy(deep=True)
    area=np.array([100,20,1,1,5,50,1,1,80,60,25,200],dtype=np.float64)
    y=pd.Series([1e6,2e6,3e6,0,4e6,5e6,6e6,7e6,8e6,9e6,1e7,2e8],index=idx,dtype=np.float64)
    dates=pd.Series([.125,.25,.5,1.,2.,4.,1.5,.75,.25,1.,.5,2.],index=idx,dtype=np.float32)
    expected_counts=np.array([3,3,3,2,2,1,1,1,1,1,1,1],dtype=np.int64)
    q=1/np.sqrt(expected_counts.astype(np.float64));w0=area/100*dates.to_numpy(dtype=np.float64)
    c=float(w0.sum(dtype=np.float64)/(w0*q).sum(dtype=np.float64));weights=w0*q*c
    counts,q_actual,group=group_factors(X)
    actual,ws=weight_details(X,dates)
    target=y.to_numpy()/(area*1e4);z=np.arange(len(idx),dtype=np.float64)-.5
    ok('effective_area_original_building_parking_one_fallbacks',np.array_equal(effective_area(X),area))
    ok('R5_normalization_full_key_counts_int64_and_missing_count1',np.array_equal(counts,expected_counts)
       and counts.dtype==np.dtype('int64') and group['named_projects']==4 and group['missing_name_rows']==5)
    ok('raw_q_exact_float64_no_threshold_or_clipping',np.array_equal(q_actual,q) and q_actual.dtype==np.dtype('float64'))
    ok('same_name_different_city_and_district_are_distinct',counts[5]==counts[6]==1 and counts[0]==3)
    ok('nondefault_index_preserved_and_no_X_mutation',X.index.equals(idx) and X.equals(bx)
       and project_identity.project_keys(X)[0].index.equals(idx))
    ok('weights_exact_float64_area_date_q_c_order',np.array_equal(actual,weights)
       and np.array_equal(sample_weights(X,dates),weights) and actual.dtype==np.dtype('float64'))
    ok('c_preserves_original_Pool_total_not_q_mean',ws['normalization_c']==c
       and np.isclose(weights.sum(),w0.sum(),rtol=2e-15,atol=0) and not np.isclose(c,1/q.mean()))
    ok('missing_raw1_but_final_multiplier_is_c',np.array_equal(q_actual[7:],np.ones(5))
       and np.allclose(weights[7:]/w0[7:],c,rtol=2e-15,atol=0) and c!=1)
    ok('unit_target_and_TWD_inverse_unchanged',np.array_equal(transform_target(y,X),target)
       and np.array_equal(inverse_transform(z,X),np.maximum(z*area*1e4,0)))
    ok('new_loss_is_date_q_c_total_L1_before_inverse_projection',np.allclose(
       weights*np.abs(target-z),dates.to_numpy(dtype=np.float64)*q*c*np.abs(y.to_numpy()-area*1e4*z)/1e6,
       rtol=2e-15,atol=1e-13))
    allmissing=X.copy();allmissing['project_name']='__MISSING__'
    wm,sm=weight_details(allmissing,dates)
    ok('all_missing_q1_c1_weights_exact_old',np.array_equal(wm,w0) and sm['normalization_c']==1
       and sm['group_summary']['named_projects']==0)
    uniform=X.copy();uniform['project_name']='One';uniform['address_city']='City';uniform['address_district']='D'
    wu,su=weight_details(uniform,dates)
    ok('all_q_equal_normalization_recovers_old_weights',np.allclose(wu,w0,rtol=2e-15,atol=0)
       and su['group_summary']['named_projects']==1)
    sentinels=X.copy();sentinels['project_name']=[None,np.nan,pd.NA,'',' ','nan','None','<NA>','NaT','__MISSING__','NaN','null']
    _,_,ss=group_factors(sentinels)
    ok('exact_R5_sentinels_no_added_NaN_null_blacklist',ss['missing_name_rows']==10 and ss['named_projects']==2)
    literal=X.copy();literal['project_name']='無'
    ok('literal_nonmissing_project_not_blacklisted',group_factors(literal)[2]['missing_name_rows']==0)
    missing_location=X.copy();missing_location.loc[idx[:3],'address_city']=None
    ok('missing_city_still_valid_named_group',group_factors(missing_location)[0][0]==3
       and group_factors(missing_location)[2]['missing_name_rows']==5)
    perm=np.array([11,7,3,9,0,6,2,5,1,10,8,4])
    pc,pq,_=group_factors(X.iloc[perm]);pw=sample_weights(X.iloc[perm],dates.iloc[perm])
    ok('row_permutation_preserves_aligned_counts_q_and_weights_tolerance',np.array_equal(pc,counts[perm])
       and np.array_equal(pq,q[perm]) and np.allclose(pw,weights[perm],rtol=2e-15,atol=0))
    for field in GROUP_RULES['key_fields']:
        bad=X.copy();bad.loc[idx[0],field]=' collision|field '
        reject('literal_pipe_in_'+field+'_rejected',lambda bad=bad:group_factors(bad))
    reject('empty_fit_rejected',lambda:sample_weights(X.iloc[:0],dates.iloc[:0]))
    duplicate=X.copy();duplicate.index=pd.Index([0]*len(X))
    reject('duplicate_fit_index_rejected',lambda:group_factors(duplicate))
    reject('misordered_date_index_rejected',lambda:sample_weights(X,dates.iloc[::-1]))
    reject('wrong_date_float64_rejected',lambda:sample_weights(X,dates.astype(np.float64)))
    reject('nonpositive_date_rejected',lambda:sample_weights(X,np.zeros(len(X),dtype=np.float32)))
    reject('nonfinite_date_rejected',lambda:sample_weights(X,np.full(len(X),np.inf,dtype=np.float32)))
    reject('misordered_target_rejected',lambda:transform_target(y.iloc[::-1],X))
    reject('nonfinite_target_rejected',lambda:transform_target(np.full(len(X),np.nan),X))
    reject('negative_price_rejected',lambda:transform_target(-np.ones(len(X)),X))
    reject('nonfinite_query_output_rejected',lambda:inverse_transform(np.full(len(X),np.inf),X))
    frame=augment_features(X,geo)
    ok('original95_values_dtypes_order_index_and23categories',frame.equals(original.augment_features(X,geo))
       and frame.dtypes.equals(original.augment_features(X,geo).dtypes) and len(frame.columns)==95
       and len(categories(X,cats))==23 and not set(['prior','project_count','project_q']) & set(frame.columns))
    reject('misordered_geography_rejected',lambda:augment_features(X,geo.iloc[::-1]))
    ok('query_API_has_no_label_weight_or_count_inputs',list(inspect.signature(predict_model).parameters)
       ==['model','state','X','geo','cats'])
    fake_config={'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,'loss_function':'MAE',
        'eval_metric':'MAE','boosting_type':'Plain','task_type':'GPU','learning_rate':.065,
        'one_hot_max_size':8,'max_ctr_complexity':2,'verbose':300,'bootstrap_type':'Bernoulli',
        'border_count':128,'l2_leaf_reg':6.,'subsample':.85,'random_strength':.5,
        'gpu_ram_part':.75,'allow_writing_files':False}
    class FakePool:
        def __init__(self,data,label=None,**kwargs): self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=original._copy_json(kwargs);self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999;self.leaves=np.full(2999,256,dtype=np.uint32)
        def fit(self,pool,**kwargs):
            if kwargs: raise AssertionError('eval_set or early stopping supplied')
            self.pool=pool;self.feature_names_=list(pool.data.columns)
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self,pool,**kwargs):
            if pool.label is not None or 'weight' in pool.kwargs: raise AssertionError('query y or weights supplied')
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
        ok('all_parameters_equal_R79_depth8_seed42_2999',parameters()==fake_config)
        model,state,meta=fit_model(X,y,geo,cats,dates)
        ok('actual_FakePool_receives_unit_labels_and_new_weights',np.array_equal(model.pool.label,target)
           and np.array_equal(model.pool.kwargs['weight'],weights)
           and set(model.pool.kwargs)=={'weight','cat_features','thread_count'})
        ok('actual_constructor_original_parameters_no_changes',model.config==fake_config and meta['changed_parameter_names']==[])
        ok('actual_FakePool_exact95_features_23cats',model.pool.data.equals(frame) and len(model.cat_indices)==23)
        legacy=original.fit_feature_state(X,geo,cats)
        ok('legacy_R19_pure_X_state_exact_in_independent_weight_wrapper',state['legacy_feature_state']==legacy
           and state['sample_weight']==WEIGHT_FORMULA and legacy['feature_rules']['sample_weight']==original.WEIGHT_FORMULA)
        ok('metadata_target_date_counts_q_c_weight_identity',meta['fit_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(target)
           and meta['fit_sample_weights_sha256']==values_sha256(weights)
           and meta['fit_date_weights_sha256']==_date_sha(dates)
           and meta['fit_project_counts_sha256']==_counts_sha(expected_counts)
           and meta['fit_project_raw_q_sha256']==values_sha256(q) and meta['fit_project_normalization_c']==c)
        model2,state2,meta2=fit_model(X,y+123456,geo,cats,dates)
        ok('changing_fit_y_does_not_change_counts_weights_or_pure_X_state',np.array_equal(model2.pool.kwargs['weight'],weights)
           and state2['group_weight_summary']==state['group_weight_summary']
           and state2['legacy_feature_state']==state['legacy_feature_state']
           and state2['fit_target_values_sha256']!=state['fit_target_values_sha256'])
        loaded=FakeModel(**model.config);loaded.feature_names_=list(model.feature_names_);loaded.cat_indices=list(model.cat_indices)
        restored=original._copy_json(state)
        ok('JSON_wrapper_simulated_model_reload_predictions_exact',np.array_equal(predict_model(loaded,restored,X,geo,cats),predict_model(model,state,X,geo,cats))
           and np.array_equal(predict_model(loaded,restored,X,geo,cats),inverse_transform(z,X)))
        frozen=base.content_sha(state)
        query=X.iloc[[0,5,7]].copy();query['project_name']=['EntireNew','ExistingChanged',None]
        query['building_area']=np.array([999,5,22],dtype=np.float32)
        old_group=globals()['group_factors']
        def forbidden_query_count(*args,**kwargs): raise AssertionError('prediction attempted project count')
        globals()['group_factors']=forbidden_query_count
        try:
            result=predict_model(loaded,restored,query,geo.loc[query.index],cats)
        finally: globals()['group_factors']=old_group
        ok('changed_query_rows_do_not_count_or_mutate_fit_state_weights',base.content_sha(state)==frozen
           and np.array_equal(model.pool.kwargs['weight'],weights)
           and np.array_equal(result,inverse_transform(query['numeric_0'].to_numpy(dtype=np.float64)-.5,query)))
        for name,key,value in [('depth6','depth',6),('seed','random_seed',137),('lr','learning_rate',.07),
                               ('iterations','iterations',2998),('L2','l2_leaf_reg',7.)]:
            previous=loaded.actual[key];loaded.actual[key]=value
            reject('native_'+name+'_change_rejected',lambda:check_model(loaded,state));loaded.actual[key]=previous
        loaded.leaves[0]=257
        reject('native_leaf_count_over256_rejected',lambda:check_model(loaded,state));loaded.leaves[0]=256
        loaded.cat_indices=loaded.cat_indices[:-1]
        reject('native_category_count_changed_rejected',lambda:check_model(loaded,state));loaded.cat_indices=list(model.cat_indices)
        loaded.feature_names_.append('extra')
        reject('native_feature_count_changed_rejected',lambda:check_model(loaded,state))
        reject('legacy_state_not_accepted_as_R93_wrapper',lambda:predict_model(model,legacy,X,geo,cats))
        tampered=original._copy_json(state);tampered['fit_project_normalization_c']=c+1
        reject('wrapper_c_tamper_rejected',lambda:check_state(tampered,X,cats))
        ok('original_inputs_unchanged',X.equals(bx) and X.dtypes.equals(bx.dtypes) and geo.equals(bg)
           and values_sha256(y)==state['fit_target_values_sha256'] and _date_sha(dates)==state['fit_date_weights_sha256'])
        ok('no_eval_earlystop_query_y_or_corrections',all(meta[k] is False for k in
           ['fit_uses_eval_set','early_stopping_used','query_labels_used','reference_or_trend_applied_here','spatial_applied_here']))
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=old_model,old_pool
        base.parameters=old_params;original._base_configuration_sha256=old_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),'synthetic_only':True,
            'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'fake_native_fit_calls':2,
            'new_real_native_fits':0,'source_sha256':source_hashes(),
            'pre_freeze_engineering_failures':[{'attempt':1,'exception':'KeyError: sample_weight',
                'location':'synthetic legacy-state assertion',
                'resolution':'read legacy feature_rules.sample_weight; candidate math unchanged',
                'real_data_loaded':False,'real_native_fits':0}]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    tick=time.monotonic();receipt=self_check();receipt['elapsed_seconds']=time.monotonic()-tick;receipt['exit_code']=0
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
