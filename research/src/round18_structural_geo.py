"""Fixed R18 raw geography CatBoost: omit four identifiers from the model view only.

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

COMPONENT = 'geography_structural_raw_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
REMOVED_FEATURE_NAMES = ['project_name', 'address_raw', 'building_and_unit', 'building_block']


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'existing_geo_features_unchanged': 8,
    'baseline_95_view_values_dtypes_order_and_index_unchanged': True,
    'remaining_91_values_dtypes_order_and_index_unchanged': True,
    'removed_feature_names': list(REMOVED_FEATURE_NAMES),
    'removal_scope': 'model view only; original X87 and nested geography state remain unchanged',
    'feature_count': 91,
    'categorical_feature_count': 19,
    'geo_encoder_rules': _copy_json(base.FEATURE_RULES),
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
        'src/round18_structural_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


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
    """All passed parameters are the original R6 recipe, without any changed value."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    base.need(result == baseline, 'R18 cannot alter original R6 parameters')
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
    """Construct unchanged R6 95 first, then omit only the four fixed model columns."""
    _check_X(X)
    original = base.augment_features(X, geo)
    base.need(len(original.columns) == 95 and original.columns.is_unique
              and original.index.equals(X.index), 'Expected the original R6 95-feature view')
    base.need(set(REMOVED_FEATURE_NAMES).issubset(original.columns), 'A removed identifier is missing')
    names = [c for c in original if c not in REMOVED_FEATURE_NAMES]
    result = original.loc[:, names].copy(deep=True)
    base.need(len(names) == 91 and not set(REMOVED_FEATURE_NAMES) & set(names),
              'Expected precisely four omitted identifiers and 91 remaining features')
    base.need(result.equals(original[names]) and result.dtypes.equals(original[names].dtypes)
              and result.index.equals(X.index), 'Remaining model features changed')
    return result


def categories(X, supplied):
    original = base.categories(X, supplied)
    base.need(len(original) == len(set(original)) == 23
              and set(REMOVED_FEATURE_NAMES).issubset(original), 'Expected four original categorical identifiers')
    result = [c for c in original if c not in REMOVED_FEATURE_NAMES]
    base.need(len(result) == len(set(result)) == 19, 'Expected 19 remaining categorical columns')
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
        'geo_state': geo_state,
        'removed_feature_names': list(REMOVED_FEATURE_NAMES),
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
              and state['feature_rules'] == FEATURE_RULES, 'Structural model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == []
              and state['parameters'] == state['baseline_parameters'], 'Original R6 configuration changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    baseline_names = state['geo_state']['all_feature_names']
    expected = [c for c in baseline_names if c not in REMOVED_FEATURE_NAMES]
    base.need(len(baseline_names) == 95 and set(REMOVED_FEATURE_NAMES).issubset(baseline_names)
              and state['all_feature_names'] == expected and len(expected) == 91
              and state['removed_feature_names'] == REMOVED_FEATURE_NAMES,
              'Model feature exclusion or original R6 geography schema changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_model(model, state):
    """Check native model capacity, not just the supplied metadata declaration."""
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 8
              and state['parameters']['boosting_type'] == 'Plain'
              and state['parameters'] == state['baseline_parameters']
              and state['changed_parameter_names'] == []
              and state['removed_feature_names'] == REMOVED_FEATURE_NAMES
              and len(state['all_feature_names']) == 91
              and len(state['categorical_columns']) == 19, 'Wrong structural model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    actual = model.get_all_params()
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain',
              'Native depth/grow/boosting policy differs from the original R6 recipe')
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
        'removed_feature_names': list(REMOVED_FEATURE_NAMES),
        'changed_parameter_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 91, 'feature_count': 91, 'categorical_feature_count': 19,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round18_structural_geo.py'],
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


def self_check():
    """Bounded feature-removal delta using synthetic FakePool/Model only."""
    checks=[]
    def ok(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name+' did not reject')
    idx=pd.Index([9,41,7,23],name='synthetic_row')
    cats=list(REMOVED_FEATURE_NAMES)+['category_'+str(i) for i in range(17)]
    X=pd.DataFrame({c:pd.Series(['x','y','__MISSING__','z'],index=idx,dtype=object) for c in cats},index=idx)
    for i in range(66):X['numeric_'+str(i)]=np.arange(len(idx),dtype='float32')
    X.loc[idx[2],'numeric_1']=np.nan
    geo=pd.DataFrame({c:np.full(len(idx),np.nan,dtype='float32') for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(len(idx),dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo=geo[base.FEATURES]
    X_before,geo_before=X.copy(deep=True),geo.copy(deep=True)
    baseline=base.augment_features(X,geo);aug=augment_features(X,geo)
    keep=[c for c in baseline if c not in REMOVED_FEATURE_NAMES]
    ok('remaining_91_values_dtypes_order_index_exact',aug.equals(baseline[keep])
       and aug.dtypes.equals(baseline[keep].dtypes) and aug.index.equals(idx) and list(aug)==keep)
    ok('four_categorical_identifiers_only_removed',len(aug.columns)==91 and len(categories(X,cats))==19
       and categories(X,cats)==[c for c in base.categories(X,cats) if c not in REMOVED_FEATURE_NAMES]
       and not set(REMOVED_FEATURE_NAMES)&set(aug))
    ok('original87_and_all8_geo_remain_unmodified',X.equals(X_before) and geo.equals(geo_before)
       and aug[base.FEATURES].equals(geo))
    ok('missing_numerics_preserved',pd.isna(aug.loc[idx[2],'numeric_1']) and aug.geo_latitude.isna().all())
    ok('state_API_fit_X_only',list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats'])
    rejects('geo_row_alignment_rejected',lambda:augment_features(X,geo.iloc[::-1]))
    bad=X.rename(columns={REMOVED_FEATURE_NAMES[0]:'unrecognized_category'})
    rejects('missing_required_removed_identifier_rejected',lambda:augment_features(bad,geo))
    rejects('wrong_original_category_list_rejected',lambda:categories(X,cats[:-1]))
    changed=X.copy(deep=True)
    for c in REMOVED_FEATURE_NAMES:changed.loc[:,c]='unseen-other-identifier'
    ok('excluded_values_cannot_enter_model_view',augment_features(changed,geo).equals(aug))
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
            if kwargs:raise AssertionError('No eval_set/earlystop/weights')
            self.feature_names_=list(pool.data.columns);self.fit_pool=pool;return self
        def predict(self,pool,**kwargs):
            if pool.label is not None:raise AssertionError('Query labels forbidden')
            return pool.data['numeric_0'].to_numpy(dtype='float64')-.5
        def get_all_params(self):return dict(self.actual)
        def get_tree_leaf_counts(self):return self.leaves.copy()
    original_model,original_pool=globals()['CatBoostRegressor'],globals()['Pool']
    original_params=base.parameters;original_config_sha=globals()['_base_configuration_sha256']
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:fake_base
    globals()['_base_configuration_sha256']=lambda:'a'*64
    try:
        params=parameters()
        ok('passed_parameters_exact_R6_no_changes',params==saved_base)
        params['depth']=99
        ok('baseline_parameters_deepcopy_unchanged',fake_base==saved_base and parameters()['depth']==8)
        fake_base['boosting_type']='Ordered'
        rejects('nonPlain_baseline_rejected',parameters)
        fake_base['boosting_type']='Plain'
        y=pd.Series([0.0,1e6,2e6,3e6],index=idx,dtype='float64')
        model,state,meta=fit_model(X,y,geo,cats)
        ok('raw_uniform_Pool_91_19_no_evalset',model.config==saved_base
           and np.array_equal(model.fit_pool.label,y.to_numpy()/1e6)
           and model.fit_pool.data.equals(aug) and set(model.fit_pool.kwargs)=={'cat_features','thread_count'}
           and model.fit_pool.kwargs['cat_features']==categories(X,cats))
        ok('fit_only_state_nested_original95_and_model91',len(state['geo_state']['all_feature_names'])==95
           and state['geo_state']['original_columns']==list(X) and state['all_feature_names']==list(aug)
           and state['removed_feature_names']==REMOVED_FEATURE_NAMES
           and state['baseline_parameters']==state['parameters']==saved_base and state['changed_parameter_names']==[]
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        actual=check_model(model,state)
        leaf_sha=hashlib.sha256(np.full(2999,256,dtype='<i8').tobytes()).hexdigest()
        ok('native_six_fields_Plain_depth8_tree2999_leaf256',actual=={
            'actual_model_depth':8,'actual_boosting_type':'Plain','actual_grow_policy':'SymmetricTree',
            'actual_trees':2999,'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':leaf_sha})
        ok('metadata_raw91_19_removal_and_unchanged_parameters',all(meta[k]==v for k,v in actual.items())
           and meta['baseline_parameters']==meta['parameters']==saved_base and meta['changed_parameter_names']==[]
           and meta['removed_feature_names']==REMOVED_FEATURE_NAMES and meta['feature_count']==91
           and meta['features']==91 and meta['categorical_feature_count']==19 and meta['target_mode']=='raw'
           and not meta['calibration_applied'] and not meta['extra_price_cap_applied']
           and not meta['reference_or_trend_applied_here'] and not meta['reference_features_changed'])
        restored=_copy_json(state)
        loaded=FakeModel(**_copy_json(model.config));loaded.feature_names_=list(model.feature_names_)
        pred=predict_model(loaded,restored,X,geo,cats)
        ok('JSON_state_simulated_reload_TWD_exact',np.array_equal(pred,predict_model(model,state,X,geo,cats))
           and pred.dtype==np.dtype('float64') and np.array_equal(pred,np.maximum(np.arange(len(X))-.5,0)*1e6))
        ok('query_removed_identifiers_do_not_change_predictions',np.array_equal(pred,predict_model(loaded,restored,changed,geo,cats)))
        ok('fit_target_hashes_raw_scale',meta['fit_original_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(y.to_numpy()/1e6))
        rejects('misaligned_target_index_rejected',lambda:fit_model(X,y.iloc[::-1],geo,cats))
        rejects('negative_target_rejected',lambda:transform_target(np.array([-1.])))
        rejects('nonfinite_target_rejected',lambda:transform_target(np.array([np.nan])))
        rejects('inverse_overflow_rejected',lambda:inverse_transform(np.array([np.finfo('float64').max])))
        loaded.actual['boosting_type']='Ordered'
        rejects('native_Ordered_rejected',lambda:check_model(loaded,restored))
        loaded.actual['boosting_type']='Plain';loaded.actual['depth']=6
        rejects('native_depth6_rejected',lambda:check_model(loaded,restored))
        loaded.actual['depth']=8;loaded.leaves[0]=257
        rejects('native_leaf_over256_rejected',lambda:check_model(loaded,restored))
        loaded.leaves[0]=256;loaded.feature_names_=list(baseline)
        rejects('native_old95_model_rejected',lambda:check_model(loaded,restored))
        bad=_copy_json(state);bad['removed_feature_names']=REMOVED_FEATURE_NAMES[:-1]
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        rejects('rehashed_wrong_removal_state_rejected',lambda:check_state(bad,X,cats))
        ok('original_X_geo_baseline_unchanged_after_all_operations',X.equals(X_before) and geo.equals(geo_before)
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
