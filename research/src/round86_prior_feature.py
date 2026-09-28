"""R86: one supervised OOF-prior numeric feature on unchanged R79 unit CatBoost.

The driver must bind train priors to the R60 group-OOF source and query priors to
that period's full-fit source. This helper verifies values/order, not provenance.
No data loader, new prior fit, query labels, parameter search or postprocessing.
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
COMPONENT = 'geography_recency_unit_with_oof_prior_catboost'
TARGET_MODE = baseline.TARGET_MODE
TARGET_FORMULA = baseline.TARGET_FORMULA
INVERSE_FORMULA = baseline.INVERSE_FORMULA
WEIGHT_FORMULA = baseline.WEIGHT_FORMULA
PRIOR_FEATURE = 'log_recency_oof_unit_prior'
PRIOR_FORMULA = 'log1p(prior_TWD_per_m2 / 1e4)'
parameters = baseline.parameters
effective_area = baseline.effective_area
transform_target = baseline.transform_target
inverse_transform = baseline.inverse_transform
categories = baseline.categories
values_sha256 = baseline.values_sha256
FEATURE_RULES = original._copy_json(baseline.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT, feature_count=96,
    additional_numeric_feature=PRIOR_FEATURE, prior_feature_formula=PRIOR_FORMULA,
    prior_feature_dtype='float64', fit_prior='driver-supplied R60 group-OOF unit prior',
    query_prior='driver-supplied same-period full-fit R60 unit prior',
    prior_uses_fit_labels=True, prior_provenance_verified_in_helper=False,
    prior_uses_query_labels=False, existing_95_values_dtypes_order_and_index_unchanged=True,
    added_feature_names=[PRIOR_FEATURE], categorical_feature_count=23)


def source_hashes():
    return {**baseline.source_hashes(),
            'src/round86_prior_feature.py': original.sha256(Path(__file__))}


def prior_values(prior, X):
    result = original._aligned_vector(prior, X, 'Unit price prior')
    base.need((result > 0).all(), 'Unit priors must be strictly positive and finite')
    return result


def prior_feature_values(prior, X):
    values = prior_values(prior, X)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = np.log1p(values / 1e4)
    base.need(np.isfinite(result).all(), 'Prior feature must be finite')
    return np.ascontiguousarray(result, dtype=np.float64)


def augment_features(X, geo, prior):
    frame = baseline.augment_features(X, geo)
    base.need(PRIOR_FEATURE not in frame, 'Duplicate prior feature')
    result = frame.copy(deep=True)
    result[PRIOR_FEATURE] = prior_feature_values(prior, X)
    base.need(len(result.columns) == 96 and result.index.equals(X.index),
              'Expected aligned 96-feature view')
    base.need(result.iloc[:, :95].equals(frame) and result.iloc[:, :95].dtypes.equals(frame.dtypes),
              'Existing 95 columns changed')
    return result


def sample_weights(X, saved_date_weights):
    dates32 = baseline._date_values(saved_date_weights, X)
    weights = original.sample_weights(X) * dates32.astype(np.float64)
    base.need(np.isfinite(weights).all() and (weights > 0).all(), 'Invalid area/date weights')
    return np.ascontiguousarray(weights, dtype=np.float64)


def fit_feature_state(X, geo, cats, train_prior):
    legacy = baseline.fit_feature_state(X, geo, cats)
    frame = augment_features(X, geo, train_prior)
    state = {
        'format_version': 1, 'component': COMPONENT,
        'target_mode': TARGET_MODE, 'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA, 'sample_weight': WEIGHT_FORMULA,
        'feature_rules': original._copy_json(FEATURE_RULES),
        'legacy_feature_state': legacy,
        'legacy_feature_state_pure_X_95_columns_only': True,
        'source_sha256': source_hashes(), 'parameters': parameters(),
        'base_configuration_sha256': legacy['base_configuration_sha256'],
        'original_categorical_columns': list(cats),
        'categorical_columns': list(legacy['categorical_columns']),
        'all_feature_names': list(frame.columns),
        'feature_names': list(frame.columns),
        'feature_dtypes': {c: str(frame[c].dtype) for c in frame},
        'prior_feature_schema': {'name': PRIOR_FEATURE, 'formula': PRIOR_FORMULA,
                                 'dtype': 'float64', 'categorical': False},
        'fit_prior_values_sha256': values_sha256(prior_values(train_prior, X)),
        'fit_prior_feature_sha256': values_sha256(frame[PRIOR_FEATURE]),
        'fit_model_feature_values_sha256': base.frame_sha(frame),
        'fit_rows': len(X), 'added_feature_names': [PRIOR_FEATURE],
        'removed_feature_names': [], 'changed_parameter_names': [],
        'fit_prior_uses_fit_labels': True, 'query_labels_used': False,
        'prior_provenance_verified_in_helper': False,
        'prior_estimated_in_helper': False,
    }
    state['state_content_sha256'] = base.content_sha(state)
    return state


def check_state(state, X, cats):
    base.need(state['state_content_sha256'] == base.content_sha({
        k:v for k,v in state.items() if k != 'state_content_sha256'}), 'Wrapper state changed')
    base.need(state['component'] == COMPONENT and state['feature_rules'] == FEATURE_RULES,
              'Wrong R86 feature rules')
    base.need(state['target_mode'] == TARGET_MODE and state['target_formula'] == TARGET_FORMULA
              and state['inverse_formula'] == INVERSE_FORMULA
              and state['sample_weight'] == WEIGHT_FORMULA, 'Target/weight rules changed')
    base.need(state['source_sha256'] == source_hashes() and state['parameters'] == parameters(),
              'Helper source/configuration changed')
    legacy = state['legacy_feature_state']
    baseline.check_state(legacy, X, cats)
    base.need(state['all_feature_names'] == state['feature_names']
              == legacy['all_feature_names'] + [PRIOR_FEATURE]
              and len(state['all_feature_names']) == 96, 'Wrong 96-column schema')
    base.need(state['categorical_columns'] == legacy['categorical_columns']
              and state['original_categorical_columns'] == list(cats)
              and PRIOR_FEATURE not in state['categorical_columns'], 'Category schema changed')
    base.need(state['prior_feature_schema'] == {'name':PRIOR_FEATURE,'formula':PRIOR_FORMULA,
              'dtype':'float64','categorical':False}, 'Prior feature schema changed')
    expected_dtypes = {**legacy['fit_model_feature_dtypes'], PRIOR_FEATURE:'float64'}
    base.need(state['feature_dtypes'] == expected_dtypes, 'Feature dtypes changed')
    base.need(state['added_feature_names'] == [PRIOR_FEATURE]
              and state['removed_feature_names'] == state['changed_parameter_names'] == [],
              'Unexpected feature/configuration change')
    base.need(state['fit_prior_uses_fit_labels'] is True and state['query_labels_used'] is False
              and state['prior_estimated_in_helper'] is False, 'Label provenance declarations changed')


def check_model(model, state):
    base.need(state['state_content_sha256'] == base.content_sha({
        k:v for k,v in state.items() if k != 'state_content_sha256'}), 'Wrapper state changed')
    base.need(state['component'] == COMPONENT and state['parameters'] == parameters(),
              'Wrong model/configuration')
    names = state['all_feature_names']
    base.need(len(names) == 96 and names[-1] == PRIOR_FEATURE
              and list(model.feature_names_) == names, 'Native 96-column names/order differ')
    actual_indices = np.asarray(model.get_cat_feature_indices())
    expected = sorted(names.index(c) for c in state['categorical_columns'])
    base.need(actual_indices.ndim == 1 and actual_indices.dtype.kind in 'iu'
              and len(expected) == len(set(expected)) == 23 and actual_indices.tolist() == expected,
              'Native 23-category positions differ')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    actual = model.get_all_params()
    required = {'depth':8,'grow_policy':'SymmetricTree','boosting_type':'Plain',
                'random_seed':42,'loss_function':'MAE','eval_metric':'MAE',
                'bootstrap_type':'Bernoulli','one_hot_max_size':8,
                'max_ctr_complexity':2,'border_count':128}
    base.need(all(actual.get(k) == v for k,v in required.items()), 'Native fixed recipe differs')
    float_params = {'learning_rate':.065,'l2_leaf_reg':6.,'subsample':.85,'random_strength':.5}
    base.need(all(np.isfinite(float(actual[k])) and np.isclose(float(actual[k]),v,rtol=1e-7,atol=0)
                  for k,v in float_params.items()), 'Native float parameter differs')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 256).all(), 'Native leaf capacity differs')
    return {'actual_model_depth':int(actual['depth']), 'actual_boosting_type':actual['boosting_type'],
            'actual_grow_policy':actual['grow_policy'], 'actual_trees':int(model.tree_count_),
            'maximum_tree_leaf_count':int(leaves.max()),
            'tree_leaf_counts_sha256':hashlib.sha256(leaves.astype('<i8').tobytes()).hexdigest(),
            'actual_learning_rate':float(actual['learning_rate']),
            'native_feature_count':96,'native_categorical_feature_count':23}


def fit_model(X, y, geo, cats, saved_date_weights, train_prior):
    started = time.monotonic()
    base.need(len(X) > 0, 'Cannot fit empty X')
    raw = original._aligned_vector(y, X, 'Fit target')
    transformed = transform_target(raw, X)
    dates32 = baseline._date_values(saved_date_weights, X)
    dates_sha = hashlib.sha256(dates32.astype('<f4',copy=False).tobytes()).hexdigest()
    weights = sample_weights(X, dates32)
    raw_sha, target_sha, weight_sha = map(values_sha256, (raw, transformed, weights))
    state = fit_feature_state(X, geo, cats, train_prior)
    frame = augment_features(X, geo, train_prior)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(frame, transformed, cat_features=state['categorical_columns'],
                   weight=weights, thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(base.frame_sha(frame) == state['fit_model_feature_values_sha256']
              and base.frame_sha(augment_features(X,geo,train_prior)) == state['fit_model_feature_values_sha256'],
              'Input features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == target_sha
              and values_sha256(sample_weights(X,saved_date_weights)) == weight_sha,
              'Labels/weights changed')
    legacy = state['legacy_feature_state']
    report = {
        'family':'catboost','component':COMPONENT,'target_mode':TARGET_MODE,
        'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
        'prediction_units':'TWD','parameters':original._copy_json(config),
        'baseline_parameters':original._copy_json(legacy['baseline_parameters']),
        'changed_parameter_names':[],'added_feature_names':[PRIOR_FEATURE],'removed_feature_names':[],
        'iterations':2999,'actual_iterations':actual['actual_trees'],'seed':42,
        'fit_rows':len(X),'feature_count':96,'categorical_feature_count':23,
        'feature_columns':list(frame.columns),'categorical_columns':list(state['categorical_columns']),
        'sample_weight':WEIGHT_FORMULA,'weight_argument_passed_to_pool':True,
        'weight_dtype':'float64','weights_normalized_to_mean_one':False,
        'date_weights_recomputed':False,'date_weights_renormalized':False,
        'date_weight_provenance_verified_in_helper':False,
        'fit_date_weights_sha256':dates_sha,'fit_date_weights_hash_encoding':'ordered little-endian float32',
        'fit_date_weights_summary':baseline._summary(dates32),
        'fit_sample_weights_sha256':weight_sha,'fit_sample_weights_summary':baseline._summary(weights),
        'fit_area_weights_sha256':values_sha256(original.sample_weights(X)),
        'fit_effective_area_sha256':legacy['fit_effective_area_sha256'],
        'fit_factor_sha256':legacy['fit_factor_sha256'],
        'fit_target_values_sha256':raw_sha,'fit_transformed_target_values_sha256':target_sha,
        'fit_prior_values_sha256':state['fit_prior_values_sha256'],
        'fit_prior_feature_sha256':state['fit_prior_feature_sha256'],
        'prior_feature_schema':state['prior_feature_schema'],
        'train_prior_required_source':'R60 same-period group-OOF unit prior',
        'query_prior_required_source':'R60 same-period full-fit unit prior',
        'prior_provenance_verified_in_helper':False,'prior_uses_fit_labels':True,
        'prior_estimated_in_helper':False,'query_labels_used':False,
        'feature_state_content_sha256':state['state_content_sha256'],
        'legacy_feature_state_content_sha256':legacy['state_content_sha256'],
        'legacy_feature_state_is_not_the_96_feature_state':True,
        'fit_original_feature_values_sha256':legacy['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256':legacy['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256':state['fit_model_feature_values_sha256'],
        'feature_rules':original._copy_json(FEATURE_RULES),'source_sha256':source_hashes(),
        'base_configuration_sha256':legacy['base_configuration_sha256'],
        'weights_use_fit_target':False,'weights_use_query':False,
        'fit_uses_eval_set':False,'early_stopping_used':False,
        'reference_or_trend_applied_here':False,'spatial_applied_here':False,
        'extra_price_cap_applied':False,'calibration_applied':False,
        'elapsed_seconds':time.monotonic()-started, **actual}
    return model,state,report


def predict_model(model, state, X, geo, cats, query_prior):
    check_state(state, X, cats)
    check_model(model, state)
    frame = augment_features(X, geo, query_prior)
    base.need(list(frame.columns) == state['all_feature_names']
              and {c:str(frame[c].dtype) for c in frame} == state['feature_dtypes'],
              'Query feature schema differs')
    z = model.predict(Pool(frame,cat_features=state['categorical_columns'],thread_count=4),thread_count=4)
    return inverse_transform(z, X)


def self_check():
    """Small synthetic value and feature-isolation checks; zero model fitting."""
    checks=[]
    def ok(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name+' did not reject')
    idx=pd.Index([3,8,5])
    X=pd.DataFrame({'building_area':[100.,200.,0.],'parking_area':[0.,0.,25.]},index=idx)
    pri=np.array([1e4,2e4,5e4],dtype=np.float64)
    ok('fixed_log1p_unit_TWD_scale',np.array_equal(prior_feature_values(pri,X),np.log1p(pri/1e4)))
    reject('wrong_length',lambda:prior_values(pri[:2],X))
    reject('matrix_prior',lambda:prior_values(pri[:,None],X))
    reject('zero_prior',lambda:prior_values(np.array([0.,1.,2.]),X))
    reject('negative_prior',lambda:prior_values(-pri,X))
    reject('NaN_prior',lambda:prior_values(np.array([np.nan,1.,2.]),X))
    reject('infinite_prior',lambda:prior_values(np.array([np.inf,1.,2.]),X))
    reject('misordered_prior_series',lambda:prior_values(pd.Series(pri,index=idx[::-1]),X))
    dates=np.array([.5,1.,2.],dtype=np.float32)
    ok('original_area_times_saved_dates_exact',np.array_equal(sample_weights(X,dates),
       original.sample_weights(X)*dates.astype(np.float64)))
    y=np.array([1e6,4e6,3e6])
    ok('original_unit_target_inverse_unchanged',np.array_equal(inverse_transform(transform_target(y,X),X),y))
    ok('predict_API_has_query_prior_no_y',list(inspect.signature(predict_model).parameters)
       ==['model','state','X','geo','cats','query_prior'])
    # Real original augment, but artificial values only. No fit state/config read.
    cats=['cat'+str(i) for i in range(21)]
    full=pd.DataFrame({c:['a','b','c'] for c in cats},index=idx)
    full['building_area']=X.building_area.astype('float32')
    full['parking_area']=X.parking_area.astype('float32')
    for i in range(64):full['num'+str(i)]=np.arange(3,dtype=np.float32)
    geo=pd.DataFrame({c:np.full(3,np.nan,dtype=np.float32) for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(3,dtype=np.float32)
    geo['geo_match_status']=['ineligible_address']*3
    geo['geo_grid_002']=['__MISSING__']*3
    geo=geo[base.FEATURES]
    before=full.copy(deep=True);g_before=geo.copy(deep=True)
    old=baseline.augment_features(full,geo)
    new=augment_features(full,geo,pri)
    ok('96_features_and_existing95_exact',new.shape==(3,96) and new.iloc[:,:95].equals(old)
       and new.iloc[:,:95].dtypes.equals(old.dtypes) and new.columns[-1]==PRIOR_FEATURE)
    ok('prior_numeric_not_new_category',str(new[PRIOR_FEATURE].dtype)=='float64'
       and len(categories(full,cats))==23 and PRIOR_FEATURE not in categories(full,cats))
    ok('input_frames_unchanged',full.equals(before) and geo.equals(g_before))
    return {'status':'passed','checks':checks,'check_count':len(checks),
        'synthetic_only':True,'real_data_loaded':False,'real_model_training_performed':False,
        'real_model_or_configuration_loaded':False,'files_written':False,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=False),flush=True)
