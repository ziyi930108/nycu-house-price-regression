"""R30: one fixed parking-area-per-slot numeric feature on original R6 geo95.
All passed CatBoost parameters, full fit rows and raw uniform target remain R6.
The frozen target-free module supplies float32 values or NaN; no caps or flags.
This module never applies the ensemble, reference, trend, or query labels.
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
import parking_ratio_features as parking
from prepare_data import ROOT, sha256

COMPONENT = 'geography_parking_ratio_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ADDED_FEATURES = [parking.FEATURE_NAME]
PARKING_SOURCE_SHA = '9e494df1275e14d426d4fbb97853f5388f9ca2079e79f898572015e9bd07f97a'


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1, 'component': COMPONENT,
    'original_features_unchanged': 87, 'existing_geo_features_unchanged': 8,
    'existing_95_values_dtypes_order_and_index_unchanged': True,
    'feature_count': 96, 'categorical_feature_count': 23,
    'geo_encoder_rules': _copy_json(base.FEATURE_RULES),
    'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
    'parking_ratio_rules': _copy_json(parking.RULES),
    'parking_ratio_source_sha256': PARKING_SOURCE_SHA,
    'fit_target': TARGET_FORMULA, 'prediction': INVERSE_FORMULA,
    'target_dtype': 'float64', 'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
    'baseline_depth': 8, 'depth': 8, 'baseline_boosting_type': 'Plain', 'boosting_type': 'Plain',
    'changed_parameter_names': [], 'removed_feature_names': [],
    'added_feature_names': list(ADDED_FEATURES), 'baseline_feature_count': 95,
    'original95_preserved': True, 'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 256,
    'baseline_iterations': 2999, 'baseline_learning_rate': 0.065,
    'iterations': 2999, 'learning_rate': 0.065,
    'native_learning_rate_comparison_rtol': 1e-7, 'native_learning_rate_comparison_atol': 0.0,
    'threads': 4, 'seed': 42, 'no_eval_set': True, 'no_early_stopping': True,
    'query_labels_used': False, 'query_used_for_geographic_state': False,
    'target_used_for_geographic_state': False,
    'reference_scoring_grouping_or_dedup_keys_changed': False,
    'reference_or_trend_applied_here': False, 'calibration_applied': False,
    'extra_price_cap_applied': False, 'nonfinite_or_inverse_overflow': 'raise ValueError',
}


def source_hashes():
    paths = ('src/round30_parking_geo.py', 'src/parking_ratio_features.py',
             'src/round6_geo.py', 'src/prepare_data.py')
    result = {name: sha256(ROOT / name) for name in paths}
    base.need(result['src/parking_ratio_features.py'] == PARKING_SOURCE_SHA,
              'Frozen parking ratio module changed')
    return result


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['learning_rate'] == 0.065 and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == 'MAE' and result['boosting_type'] == 'Plain',
              'Expected the fixed R6 depth-8 Plain recipe')
    return result


def parameters():
    """Return an independent deep copy of the exact original R6 recipe."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    base.need(result == baseline, 'R30 must not change any passed model parameter')
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
    """Append one float32 numeric column without changing the original 95."""
    _check_X(X)
    baseline = base.augment_features(X, geo)
    base.need(len(baseline.columns) == 95 and baseline.index.equals(X.index),
              'Expected original R6 95-feature schema')
    base.need(not set(ADDED_FEATURES) & set(baseline.columns), 'Added feature collision')
    added = parking.features_from_X(X)
    base.need(list(added.columns) == ADDED_FEATURES and added.index.equals(X.index)
              and str(added.iloc[:, 0].dtype) == 'float32'
              and not np.isinf(added.iloc[:, 0].to_numpy()).any(), 'Invalid ratio schema')
    result = pd.concat([baseline, added], axis=1)
    preserved = result.iloc[:, :95]
    base.need(len(result.columns) == 96 and result.columns.is_unique
              and list(result.columns) == list(baseline.columns) + ADDED_FEATURES
              and preserved.equals(baseline) and preserved.dtypes.equals(baseline.dtypes)
              and result.index.equals(X.index), 'Original 95 features changed')
    return result


def categories(X, supplied):
    result = base.categories(X, supplied)
    base.need(len(result) == len(set(result)) == 23
              and not set(result) & set(ADDED_FEATURES), 'Ratio must remain numeric')
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
        'added_feature_names': list(ADDED_FEATURES),
        'baseline_feature_count': 95,
        'original95_preserved': True,
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
              and state['feature_rules'] == FEATURE_RULES, 'Parking-ratio raw model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == []
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True, 'Original R6 configuration or added numeric schema changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['geo_state']['all_feature_names']) == 95
              and len(state['all_feature_names']) == 96, 'Original R6 or added model columns changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_model(model, state):
    """Check native trees, learning rate, policies and categorical positions."""
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 8
              and state['parameters']['boosting_type'] == 'Plain'
              and state['parameters']['iterations'] == 2999 and state['parameters']['learning_rate'] == 0.065
              and state['changed_parameter_names'] == []
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True
              and state['parameters'] == state['baseline_parameters']
              and state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['all_feature_names']) == 96, 'Wrong parking-ratio raw model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    category_indices = np.asarray(model.get_cat_feature_indices())
    expected_categories = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(category_indices.shape == (23,) and category_indices.dtype.kind in 'iu'
              and len(expected_categories) == len(set(expected_categories)) == 23
              and np.array_equal(category_indices, expected_categories), 'Native categorical positions differ')
    actual = model.get_all_params()
    native_learning_rate = float(actual['learning_rate'])
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain'
              and actual['random_seed'] == 42 and actual['loss_function'] == 'MAE', 'Native depth/grow/boosting policy differs')
    base.need(np.isclose(native_learning_rate, 0.065, rtol=1e-7, atol=0),
              'Native learning rate differs from the fixed 0.065 within storage tolerance')
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
        'actual_learning_rate': native_learning_rate,
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
        'removed_feature_names': [],
        'added_feature_names': list(ADDED_FEATURES),
        'baseline_feature_count': 95,
        'original95_preserved': True,
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 96, 'feature_count': 96, 'categorical_feature_count': 23,
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
        'script_sha256': state['source_sha256']['src/round30_parking_geo.py'],
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
    """Small additive-feature/FakePool checks; never real config, data or GPU."""
    checks=[]
    def ok(name,value):
        base.need(bool(value),name);checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,RuntimeError,KeyError):checks.append(name);return
        raise AssertionError(name+' did not reject')
    idx=pd.Index([9,41,7,23],name='synthetic_row')
    cats=['category_'+str(i) for i in range(21)]
    X=pd.DataFrame({c:pd.Series(['category']*4,index=idx,dtype=object) for c in cats},index=idx)
    for i in range(64):X['numeric_'+str(i)]=np.arange(4,dtype='float32')
    X['parking_area']=np.array([20,60,0,50],dtype='float32')
    X['parking_count']=np.array([1,2,1,0],dtype='float32')
    geo=pd.DataFrame({c:np.full(4,np.nan,dtype='float32') for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(4,dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*4,index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*4,index=idx,dtype=object)
    geo=geo[base.FEATURES]
    before=X.copy(deep=True);gbefore=geo.copy(deep=True)
    view=augment_features(X,geo);original=base.augment_features(X,geo)
    ok('original95_exact_and_ratio_float32',view.iloc[:,:95].equals(original)
       and view.iloc[:,:95].dtypes.equals(original.dtypes)
       and view.iloc[:,95].dtype==np.dtype('float32')
       and np.array_equal(view.iloc[:2,95].to_numpy(),[20,30]) and view.iloc[2:,95].isna().all())
    ok('96_columns_23_cats_ratio_numeric',len(view.columns)==96 and len(categories(X,cats))==23
       and ADDED_FEATURES==['parking_area_per_slot'] and ADDED_FEATURES[0] not in categories(X,cats))
    fake={'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,
          'loss_function':'MAE','boosting_type':'Plain','task_type':'GPU','learning_rate':.065}
    class FakePool:
        def __init__(self,data,label=None,**kwargs):self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=_copy_json(kwargs);self.tree_count_=2999
            self.actual={**kwargs,'grow_policy':'SymmetricTree'}
        def fit(self,pool,**kwargs):
            assert not kwargs
            self.fit_pool=pool;self.feature_names_=list(pool.data.columns)
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self,pool,**kwargs):
            assert pool.label is None
            return pool.data['numeric_0'].to_numpy(dtype='float64')-.5
        def get_all_params(self):return dict(self.actual)
        def get_tree_leaf_counts(self):return np.full(2999,256,dtype=np.uint32)
        def get_cat_feature_indices(self):return self.cat_indices
    model_cls,pool_cls=globals()['CatBoostRegressor'],globals()['Pool']
    old_params,old_config=base.parameters,globals()['_base_configuration_sha256']
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:_copy_json(fake)
    globals()['_base_configuration_sha256']=lambda:'a'*64
    try:
        y=pd.Series([0,1e6,2e6,3e6],index=idx,dtype='float64')
        model,state,meta=fit_model(X,y,geo,cats)
        ok('full_fit_raw_uniform_original_parameters',meta['fit_rows']==4 and model.config==fake
           and np.array_equal(model.fit_pool.label,y.to_numpy()/1e6)
           and set(model.fit_pool.kwargs)=={'cat_features','thread_count'}
           and meta['changed_parameter_names']==[] and meta['weight_argument_passed_to_pool'] is False)
        ok('nested_original95_fit_state_and_96_model_state',len(state['geo_state']['all_feature_names'])==95
           and len(state['all_feature_names'])==96 and state['added_feature_names']==ADDED_FEATURES
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        restored=_copy_json(state);p=predict_model(model,restored,X,geo,cats)
        ok('JSON_roundtrip_raw_inverse',np.array_equal(p,[0,500000,1500000,2500000])
           and p.dtype==np.dtype('float64') and meta['native_categorical_feature_count']==23)
        q=X.copy();q.loc[q.index[0],'parking_area']=np.float32(80)
        stamp=json.dumps(restored,sort_keys=True)
        predict_model(model,restored,q,geo,cats)
        ok('query_ratio_computed_without_fit_state_update',augment_features(q,geo).iloc[0,-1]==80
           and json.dumps(restored,sort_keys=True)==stamp)
        reject('target_series_index_mismatch',lambda:fit_model(X,y.iloc[::-1],geo,cats))
        model.actual['depth']=10
        reject('native_depth_change_rejected',lambda:check_model(model,state))
        model.actual['depth']=8
        model.cat_indices.append(95)
        reject('ratio_categorical_position_rejected',lambda:check_model(model,state))
        ok('source_frames_unchanged',X.equals(before) and X.dtypes.equals(before.dtypes)
           and geo.equals(gbefore) and geo.dtypes.equals(gbefore.dtypes))
        ok('state_and_query_API_no_query_labels',list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats']
           and list(inspect.signature(predict_model).parameters)==['model','state','X','geo','cats'])
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=model_cls,pool_cls
        base.parameters=old_params;globals()['_base_configuration_sha256']=old_config
    return {'status':'passed','checks':checks,'check_count':len(checks),'synthetic_only':True,
            'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'files_written':False,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=False),flush=True)
