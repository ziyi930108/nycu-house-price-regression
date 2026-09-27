"""R24 fixed seeds: unchanged R6 geography features and raw-price recipe.
Only new seeds 137 and 2026 may be fitted; saved seed42 is reused by the driver.
No labels/query fit-state, validation set, early stopping or reference here.
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

COMPONENT = 'geography_seed_ensemble'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
SEEDS = (42, 137, 2026)
NEW_SEEDS = (137, 2026)
ENSEMBLE_FORMULA = '(TWD_seed42 + TWD_seed137 + TWD_seed2026) / 3; each branch clipped before mean'


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
    'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'target_dtype': 'float64',
    'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': 'uniform',
    'weight_argument_passed_to_pool': False,
    'baseline_depth': 8,
    'depth': 8,
    'baseline_boosting_type': 'Plain',
    'boosting_type': 'Plain',
    'changed_parameter_names': ['random_seed'],
    'removed_feature_names': [],
    'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 256,
    'baseline_iterations': 2999,
    'baseline_learning_rate': 0.065,
    'iterations': 2999,
    'learning_rate': 0.065,
    'native_learning_rate_comparison_rtol': 1e-7,
    'native_learning_rate_comparison_atol': 0.0,
    'threads': 4,
    'seeds': list(SEEDS),
    'newly_fitted_seeds': list(NEW_SEEDS),
    'seed42_reused_not_retrained': True,
    'ensemble_formula': ENSEMBLE_FORMULA,
    'seed_or_weight_selection': False,
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
        'src/round24_seed_ensemble.py', 'src/round6_geo.py', 'src/prepare_data.py')}


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['learning_rate'] == 0.065 and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == 'MAE' and result['boosting_type'] == 'Plain',
              'Expected the fixed R6 depth-8 Plain recipe')
    return result


def parameters(seed=137):
    base.need(type(seed) is int and seed in SEEDS, 'Only the three fixed seeds are allowed')
    baseline = _baseline_parameters()
    result = dict(baseline, random_seed=seed)
    base.need(set(result)==set(baseline) and sorted(k for k in result if result[k]!=baseline[k])
              == ([] if seed==42 else ['random_seed']), 'Only random_seed may differ from R6')
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


def fit_feature_state(X, geo, cats, seed=137):
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
        'seed': seed,
        'parameters': parameters(seed),
        'changed_parameter_names': ['random_seed'],
        'removed_feature_names': [],
        'geo_state': geo_state,
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
              and state['feature_rules'] == FEATURE_RULES, 'Fixed seed raw model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters(state['seed'])
              and state['changed_parameter_names'] == ['random_seed']
              and state['removed_feature_names'] == [], 'Fixed seed-only configuration changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names']
              and len(state['all_feature_names']) == 95, 'Original R6 model columns changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_native_model(model, names, cats, seed):
    base.need(type(seed) is int and seed in SEEDS, 'Unplanned native seed')
    base.need(len(names)==95 and len(set(names))==95 and len(cats)==len(set(cats))==23,
              'Native schema must be original95/23')
    base.need(int(model.tree_count_)==2999 and list(model.feature_names_)==list(names),
              'Native trees/features differ')
    positions=np.asarray(model.get_cat_feature_indices())
    expected=sorted(names.index(c) for c in cats)
    base.need(positions.shape==(23,) and positions.dtype.kind in 'iu'
              and np.array_equal(positions,expected), 'Native category positions differ')
    actual=model.get_all_params();rate=float(actual['learning_rate'])
    base.need(actual['depth']==8 and actual['grow_policy']=='SymmetricTree'
              and actual['boosting_type']=='Plain' and actual['random_seed']==seed
              and actual['loss_function']==actual['eval_metric']=='MAE', 'Native fixed recipe/seed differs')
    base.need(np.isclose(rate,.065,rtol=1e-7,atol=0),'Native learning rate differs')
    leaves=np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape==(2999,) and leaves.dtype.kind in 'iu' and (leaves>0).all()
              and (leaves<=256).all(),'Native leaf capacity differs')
    return {'actual_model_depth':int(actual['depth']),'actual_boosting_type':actual['boosting_type'],
        'actual_grow_policy':actual['grow_policy'],'actual_trees':int(model.tree_count_),
        'maximum_tree_leaf_count':int(leaves.max()),
        'tree_leaf_counts_sha256':hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest(),
        'actual_learning_rate':rate,'actual_random_seed':int(actual['random_seed'])}


def check_model(model, state):
    base.need(state['state_content_sha256']==base.content_sha({k:v for k,v in state.items()
              if k!='state_content_sha256'}),'State content changed')
    base.need(state['component']==COMPONENT and state['parameters']==parameters(state['seed'])
              and state['changed_parameter_names']==['random_seed'] and state['removed_feature_names']==[],
              'Wrong fixed seed model state')
    return check_native_model(model,state['all_feature_names'],state['categorical_columns'],state['seed'])


def fit_model(X, y, geo, cats, seed=137):
    """Train once on fit labels only, with uniform raw-price MAE."""
    started = time.monotonic()
    base.need(type(seed) is int and seed in NEW_SEEDS, 'Saved seed42 must not be refitted')
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fit target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fit target length differs from X')
    transformed = transform_target(raw)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    state = fit_feature_state(X, geo, cats, seed)
    augmented = augment_features(X, geo)
    config = parameters(seed)
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters(seed) == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'],
              'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Fit target changed')
    metadata = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE, 'seed': seed,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'parameters': _copy_json(config), 'parameters_sha256': base.content_sha(config),
        'baseline_parameters': _copy_json(state['baseline_parameters']),
        'changed_parameter_names': ['random_seed'],
        'removed_feature_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 95, 'feature_count': 95, 'categorical_feature_count': 23,
        'native_categorical_feature_count': 23,
        'native_learning_rate_comparison_rtol': 1e-7,
        'native_learning_rate_comparison_atol': 0.0,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round24_seed_ensemble.py'],
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
    result = inverse_transform(raw)
    base.need(result.shape == (len(X),), 'Prediction rows differ')
    return result




def average_predictions(predictions):
    base.need(isinstance(predictions,dict) and set(predictions)==set(SEEDS), 'All three fixed seeds required')
    vectors=[_float_vector(predictions[s], 'TWD seed '+str(s)) for s in SEEDS]
    base.need(len(vectors[0])>0 and all(v.shape==vectors[0].shape and (v>=0).all() for v in vectors),
              'Per-seed predictions must be aligned nonnegative TWD')
    with np.errstate(over='raise',invalid='raise'):
        result=(vectors[0]+vectors[1]+vectors[2])/3.
    base.need(np.isfinite(result).all(), 'Mean overflow')
    return result.astype('float64',copy=False)


def self_check():
    global CatBoostRegressor, Pool, _base_configuration_sha256
    checks=[]
    def good(name,condition):base.need(condition,name);checks.append(name)
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,FloatingPointError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    originals=(CatBoostRegressor,Pool,base.parameters,_base_configuration_sha256)
    fakeparams={'iterations':2999,'learning_rate':.065,'thread_count':4,'depth':8,'random_seed':42,
        'loss_function':'MAE','eval_metric':'MAE','boosting_type':'Plain','task_type':'GPU',
        'l2_leaf_reg':6.,'max_ctr_complexity':2,'one_hot_max_size':8,'allow_writing_files':False}
    calls=[]
    class FakePool:
        def __init__(self,data,label=None,**kw):
            base.need(set(kw)=={'cat_features','thread_count'},'No weights/extra Pool arguments')
            self.data=data;self.label=label;self.cats=kw['cat_features'];calls.append(('pool',label is not None,kw))
    class FakeModel:
        def __init__(self,**kw):self.params=kw;self.tree_count_=2999
        def fit(self,pool,**kw):
            base.need(not kw,'No eval/earlystop arguments');self.feature_names_=list(pool.data.columns)
            self.cats=list(pool.cats);self.y=np.asarray(pool.label);calls.append(('fit',dict(self.params)))
        def predict(self,pool,thread_count=4):return np.arange(len(pool.data),dtype='float64')-1.
        def get_all_params(self):return {**self.params,'grow_policy':'SymmetricTree'}
        def get_cat_feature_indices(self):return sorted(self.feature_names_.index(c) for c in self.cats)
        def get_tree_leaf_counts(self):return np.full(2999,256,dtype='int64')
    try:
        CatBoostRegressor,Pool=FakeModel,FakePool;base.parameters=lambda:dict(fakeparams)
        _base_configuration_sha256=lambda:'b'*64
        X=pd.DataFrame({**{f'cat{i}':pd.Series(['a','b','a'],dtype='str') for i in range(21)},
                        **{f'num{i}':np.array([1,2,np.nan],dtype='float32') for i in range(66)}})
        cats=list(X.columns[:21]);geo=pd.DataFrame({c:np.array([1,1,0],dtype='float32') for c in base.NUMERIC})
        geo['geo_latitude']=np.array([25,24,np.nan],dtype='float32');geo['geo_longitude']=np.array([121,120,np.nan],dtype='float32')
        for c in ['geo_log_road_length','geo_log_extent','geo_nearby_fragments']:geo.loc[2,c]=np.nan
        geo['geo_match_status']=pd.Series(['road_level','road_level','unmatched'],dtype='str')
        geo['geo_grid_002']=pd.Series(['1250:6050','1200:6000','__MISSING__'],dtype='str')
        before=(base.frame_sha(X),base.frame_sha(geo));y=np.array([0.,1e6,2e6])
        for seed in NEW_SEEDS:
            good('seed_only_params_'+str(seed),parameters(seed)==dict(fakeparams,random_seed=seed))
            model,state,meta=fit_model(X,y,geo,cats,seed)
            good('fit95_23_'+str(seed),meta['feature_count']==95 and meta['categorical_feature_count']==23)
            good('native_seed_'+str(seed),meta['actual_random_seed']==seed and meta['actual_trees']==2999)
            good('raw_target_'+str(seed),np.array_equal(model.y,[0.,1.,2.]))
            good('uniform_'+str(seed),meta['sample_weight']=='uniform' and meta['weight_argument_passed_to_pool'] is False)
            good('no_eval_'+str(seed),meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False)
            prediction=predict_model(model,state,X,geo,cats);good('clip_before_mean_'+str(seed),np.array_equal(prediction,[0.,0.,1e6]))
            restored=json.loads(json.dumps(state));check_state(restored,X,cats)
            good('json_state_'+str(seed),restored==state)
            good('fit_state_no_y_query_'+str(seed),not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
            statehash=state['state_content_sha256'];predict_model(model,state,X.iloc[:1],geo.iloc[:1],cats)
            good('query_no_state_mutation_'+str(seed),statehash==state['state_content_sha256'])
            model.params['random_seed']=42;bad('wrong_native_seed_'+str(seed),lambda:check_model(model,state));model.params['random_seed']=seed
            model.params['learning_rate']=.0325;bad('wrong_LR_'+str(seed),lambda:check_model(model,state));model.params['learning_rate']=float(np.float32(.065))
            good('float32_LR_'+str(seed),np.isfinite(check_model(model,state)['actual_learning_rate']))
            model.params['learning_rate']=.065
        good('original_X_geo_unchanged',before==(base.frame_sha(X),base.frame_sha(geo)))
        good('base_params_not_mutated',fakeparams['random_seed']==42)
        good('seed42_params_only_query',parameters(42)==fakeparams)
        bad('seed42_refit_forbidden',lambda:fit_model(X,y,geo,cats,42))
        bad('unplanned_seed',lambda:parameters(999))
        bad('boolean_seed',lambda:parameters(True))
        bad('negative_y',lambda:transform_target([-1.]))
        bad('nan_y',lambda:transform_target([np.nan]))
        bad('inverse_overflow',lambda:inverse_transform([1e308]))
        bad('misaligned_y_index',lambda:fit_model(X,pd.Series(y,index=[1,2,3]),geo,cats,137))
        bad('misaligned_geo_index',lambda:augment_features(X,geo.set_axis([1,2,3])))
        p={42:np.array([0.,3.]),137:np.array([3.,6.]),2026:np.array([6.,9.])}
        good('fixed_TWD_mean',np.array_equal(average_predictions(p),[3.,6.]))
        good('dictionary_order_irrelevant',np.array_equal(average_predictions(dict(reversed(list(p.items())))),[3.,6.]))
        bad('missing_seed',lambda:average_predictions({42:p[42],137:p[137]}))
        bad('negative_component',lambda:average_predictions({**p,42:np.array([-1.,3.])}))
        bad('unequal_length',lambda:average_predictions({**p,42:np.array([1.])}))
        bad('nonfinite_component',lambda:average_predictions({**p,42:np.array([np.inf,1.])}))
        good('clip_mean_not_mean_clip',not np.array_equal(average_predictions({42:inverse_transform([-3.]),137:inverse_transform([1.]),2026:inverse_transform([1.])}),inverse_transform([(-3.+1.+1.)/3.])))
        good('exactly_two_fake_fits',len([x for x in calls if x[0]=='fit'])==2)
    finally:CatBoostRegressor,Pool,base.parameters,_base_configuration_sha256=originals
    return {'synthetic_only':True,'real_training_or_config_or_data_read':False,'checks':checks,'check_count':len(checks)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check-only',action='store_true',required=True)
    parser.parse_args();print(json.dumps(self_check(),ensure_ascii=False),flush=True)
