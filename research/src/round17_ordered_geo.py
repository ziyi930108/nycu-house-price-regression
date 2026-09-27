"""Fixed R17 raw geography CatBoost: only Plain changes to Ordered boosting.

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

COMPONENT = 'geography_ordered_raw_catboost'
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
    'feature_count': 95,
    'categorical_feature_count': 23,
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
    'boosting_type': 'Ordered',
    'changed_parameter_names': ['boosting_type'],
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
        'src/round17_ordered_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


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
    """Deep-copy R6; boosting_type is the only permitted changed parameter."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    result['boosting_type'] = 'Ordered'
    base.need(set(result) == set(baseline)
              and sorted(k for k in result if result[k] != baseline[k]) == ['boosting_type'],
              'Only boosting_type may change from Plain to Ordered')
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
        'changed_parameter_names': ['boosting_type'],
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
              and state['feature_rules'] == FEATURE_RULES, 'Ordered model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == ['boosting_type'], 'Fixed boosting-only configuration changed')
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
              and state['parameters']['boosting_type'] == 'Ordered'
              and len(state['all_feature_names']) == 95, 'Wrong Ordered model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    actual = model.get_all_params()
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Ordered',
              'Native depth/grow/boosting policy differs from the fixed Ordered model')
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
        'changed_parameter_names': ['boosting_type'],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 95, 'feature_count': 95, 'categorical_feature_count': 23,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round17_ordered_geo.py'],
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
    """Bounded R17 delta with FakePool/Model; no real data/config/model IO."""
    checks=[]
    def ok(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name+' did not reject')
    idx=pd.Index([9,41,7,23],name='synthetic_row')
    cats=['category_'+str(i) for i in range(21)]
    X=pd.DataFrame({c:pd.Series(['category']*len(idx),index=idx,dtype=object) for c in cats},index=idx)
    for i in range(66):X['numeric_'+str(i)]=np.arange(len(idx),dtype='float32')
    geo=pd.DataFrame({c:np.full(len(idx),np.nan,dtype='float32') for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(len(idx),dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo=geo[base.FEATURES]
    X_before,geo_before=X.copy(deep=True),geo.copy(deep=True)
    aug=augment_features(X,geo);expected=base.augment_features(X,geo)
    ok('original_R6_95_values_dtypes_order_index_exact',aug.equals(expected)
       and aug.dtypes.equals(expected.dtypes) and aug.index.equals(idx))
    ok('95_features_23_categories_no_prefix_or_alias',len(aug.columns)==95 and len(categories(X,cats))==23
       and categories(X,cats)==base.categories(X,cats))
    ok('state_API_fit_X_only',list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats'])
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
        ok('only_Plain_to_Ordered_parameter_delta',params['boosting_type']=='Ordered'
           and sorted(k for k in params if params[k]!=fake_base[k])==['boosting_type'] and set(params)==set(fake_base))
        params['depth']=99
        ok('baseline_parameters_are_not_mutated',fake_base==saved_base and parameters()['depth']==8)
        fake_base['boosting_type']='Ordered'
        rejects('nonPlain_baseline_rejected',parameters)
        fake_base['boosting_type']='Plain'
        y=pd.Series([0.0,1e6,2e6,3e6],index=idx,dtype='float64')
        model,state,meta=fit_model(X,y,geo,cats)
        ok('constructor_exact_Order_recipe_and_uniform_raw_Pool',model.config==parameters()
           and np.array_equal(model.fit_pool.label,y.to_numpy()/1e6)
           and set(model.fit_pool.kwargs)=={'cat_features','thread_count'}
           and model.fit_pool.kwargs['cat_features']==categories(X,cats))
        ok('fit_only_state_records_boosting_only_delta',state['baseline_parameters']==saved_base
           and state['parameters']==parameters() and state['changed_parameter_names']==['boosting_type']
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        actual=check_model(model,state)
        leaf_sha=hashlib.sha256(np.full(2999,256,dtype='<i8').tobytes()).hexdigest()
        ok('native_six_fields_Ordered_depth8_tree2999_leaf256',actual=={
            'actual_model_depth':8,'actual_boosting_type':'Ordered','actual_grow_policy':'SymmetricTree',
            'actual_trees':2999,'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':leaf_sha})
        ok('metadata_records_native_boosting_and_unchanged_raw95_23',all(meta[k]==v for k,v in actual.items())
           and meta['baseline_parameters']==saved_base and meta['parameters']==parameters()
           and meta['changed_parameter_names']==['boosting_type'] and meta['feature_count']==95
           and meta['categorical_feature_count']==23 and meta['target_mode']=='raw'
           and not meta['calibration_applied'] and not meta['extra_price_cap_applied']
           and not meta['reference_or_trend_applied_here'])
        restored=_copy_json(state)
        loaded=FakeModel(**_copy_json(model.config));loaded.feature_names_=list(model.feature_names_)
        pred=predict_model(loaded,restored,X,geo,cats)
        ok('JSON_state_and_simulated_model_reload_TWD_exact',np.array_equal(pred,predict_model(model,state,X,geo,cats))
           and pred.dtype==np.dtype('float64') and np.array_equal(pred,np.maximum(np.arange(len(X))-.5,0)*1e6))
        ok('ordered_model_preserves_raw_target_hashes',meta['fit_original_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(y.to_numpy()/1e6))
        loaded.actual['boosting_type']='Plain'
        rejects('native_Plain_reload_rejected',lambda:check_model(loaded,restored))
        rejects('predict_checks_native_Ordered',lambda:predict_model(loaded,restored,X,geo,cats))
        loaded.actual['boosting_type']='Ordered';loaded.actual['depth']=6
        rejects('native_depth6_rejected',lambda:check_model(loaded,restored))
        loaded.actual['depth']=8;loaded.actual['grow_policy']='Lossguide'
        rejects('native_nonSymmetricTree_rejected',lambda:check_model(loaded,restored))
        loaded.actual['grow_policy']='SymmetricTree';loaded.leaves[0]=257
        rejects('leaf_over256_rejected',lambda:check_model(loaded,restored))
        loaded.leaves[0]=0
        rejects('zero_leaf_rejected',lambda:check_model(loaded,restored))
        loaded.leaves[0]=256;loaded.tree_count_=2998
        rejects('wrong_tree_count_rejected',lambda:check_model(loaded,restored))
        bad=_copy_json(state);bad['parameters']['boosting_type']='Plain'
        rejects('changed_state_rejected',lambda:check_state(bad,X,cats))
        ok('original_X_geo_and_baseline_unchanged',X.equals(X_before) and geo.equals(geo_before) and fake_base==saved_base)
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
