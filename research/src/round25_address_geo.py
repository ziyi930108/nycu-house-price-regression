"""R25 raw single-seed CatBoost: unchanged R6 95 plus fixed address detail pack.
All passed model parameters remain R6. No training/query vocabulary or labels
enter feature construction; no evaluation set, early stopping or trend here.
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
import round25_address_features as address_pack
from prepare_data import ROOT, sha256

COMPONENT = 'geography_address_detail_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ADDED_FEATURES = list(address_pack.FEATURE_COLUMNS)
ADDED_CATEGORICAL_FEATURES = list(address_pack.CATEGORICAL_COLUMNS)


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'existing_geo_features_unchanged': 8,
    'existing_95_values_dtypes_order_and_index_unchanged': True,
    'feature_count': 98,
    'categorical_feature_count': 25,
    'geo_encoder_rules': _copy_json(base.FEATURE_RULES),
    'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'address_feature_rules': _copy_json(address_pack.RULES),
    'additional_categorical_features': list(ADDED_CATEGORICAL_FEATURES),
    'original_coarse_grid_policy': 'preserve existing geo_grid_002 unchanged',
    'fit_vocabulary_or_support_filter': False,
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
    'removed_feature_names': [],
    'added_feature_names': list(ADDED_FEATURES),
    'baseline_feature_count': 95,
    'original95_preserved': True,
    'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 256,
    'baseline_iterations': 2999,
    'baseline_learning_rate': 0.065,
    'iterations': 2999,
    'learning_rate': 0.065,
    'native_learning_rate_comparison_rtol': 1e-7,
    'native_learning_rate_comparison_atol': 0.0,
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
        'src/round25_address_geo.py', 'src/round25_address_features.py', 'src/geo_address_audit.py',
        'src/round6_geo.py', 'src/prepare_data.py')}


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
    base.need(result == baseline, 'R25 must not change any passed model parameter')
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
    """Append the fixed three-column pack; retain every original R6 value."""
    _check_X(X)
    baseline = base.augment_features(X, geo)
    base.need(len(baseline.columns) == 95 and baseline.columns.is_unique
              and baseline.index.equals(X.index), 'Expected original R6 95-feature schema')
    base.need(not set(ADDED_FEATURES) & set(baseline.columns), 'Address feature name collision')
    added = address_pack.features_from_X(X)
    base.need(list(added.columns) == ADDED_FEATURES and added.index.equals(X.index)
              and added.dtypes.astype(str).tolist() == ['object','object','float32'],
              'Address feature pack schema or row alignment differs')
    result = pd.concat([baseline, added], axis=1)
    preserved = result.iloc[:, :95]
    base.need(list(result.columns) == list(baseline.columns) + ADDED_FEATURES
              and len(result.columns) == 98 and result.columns.is_unique
              and preserved.equals(baseline) and preserved.dtypes.equals(baseline.dtypes)
              and preserved.index.equals(baseline.index), 'Original 95 features or added address schema changed')
    return result


def categories(X, supplied):
    baseline = base.categories(X, supplied)
    result = baseline + list(ADDED_CATEGORICAL_FEATURES)
    base.need(len(baseline) == 23 and len(result) == len(set(result)) == 25,
              'Expected 23 original and two additional unique categorical columns')
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
              and state['feature_rules'] == FEATURE_RULES, 'Address-detail raw model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == []
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True, 'Original R6 configuration or added address schema changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['geo_state']['all_feature_names']) == 95
              and len(state['all_feature_names']) == 98, 'Original R6 or added model columns changed')
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
              and len(state['all_feature_names']) == 98, 'Wrong address-detail raw model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    category_indices = np.asarray(model.get_cat_feature_indices())
    expected_categories = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(category_indices.shape == (25,) and category_indices.dtype.kind in 'iu'
              and len(expected_categories) == len(set(expected_categories)) == 25
              and np.array_equal(category_indices, expected_categories), 'Native categorical positions differ')
    actual = model.get_all_params()
    base.need(actual['random_seed'] == 42 and actual['loss_function'] == 'MAE'
              and actual['eval_metric'] == 'MAE', 'Native seed or loss differs from R6')
    native_learning_rate = float(actual['learning_rate'])
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain', 'Native depth/grow/boosting policy differs')
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
    prepared_at = time.monotonic()
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    fit_feature_preparation_seconds = time.monotonic() - prepared_at
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
        'features': 98, 'feature_count': 98, 'categorical_feature_count': 25,
        'native_categorical_feature_count': 25,
        'native_learning_rate_comparison_rtol': 1e-7,
        'native_learning_rate_comparison_atol': 0.0,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round25_address_geo.py'],
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
        'fit_feature_preparation_seconds': fit_feature_preparation_seconds,
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
    """Bounded integration checks; artificial frames/FakePool only, no config read."""
    checks=[]
    def ok(name,value):
        base.need(bool(value),'Synthetic check failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    index=pd.Index([7,12,25,40],name='artificial')
    cats=['address_raw','district']+['cat_'+str(i) for i in range(19)]
    X=pd.DataFrame({c:pd.Series(['x']*4,index=index,dtype=object) for c in cats},index=index)
    X.loc[:,'address_raw']=['臺北市大安區中山路101號','__MISSING__','臺北市大安區中山路12巷3弄10號','臺北市大安區中山路10號旁']
    X.loc[:,'district']='大安區'
    for i in range(66):X['numeric_'+str(i)]=np.arange(4,dtype='float32')
    geo=pd.DataFrame({c:np.full(4,np.nan,dtype='float32') for c in base.NUMERIC},index=index)
    geo['geo_has_position']=np.zeros(4,dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*4,index=index,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*4,index=index,dtype=object)
    geo=geo[base.FEATURES]
    before_X,before_geo=X.copy(deep=True),geo.copy(deep=True)
    view=augment_features(X,geo);baseline=base.augment_features(X,geo)
    ok('original95_values_dtypes_index_order_exact',view.iloc[:,:95].equals(baseline)
       and view.iloc[:,:95].dtypes.equals(baseline.dtypes) and view.index.equals(index))
    ok('exact98_25_two_categories_one_flag',len(view.columns)==98 and len(categories(X,cats))==25
       and view.columns[-3:].tolist()==ADDED_FEATURES
       and categories(X,cats)==base.categories(X,cats)+ADDED_CATEGORICAL_FEATURES
       and view[ADDED_FEATURES].dtypes.astype(str).tolist()==['object','object','float32'])
    ok('pack_delegation_values_exact',view[ADDED_FEATURES].equals(address_pack.features_from_X(X)))
    ok('missing_syntax_not_imputed',view.addr_detail_eligible.tolist()==[1.,0.,1.,0.]
       and view.loc[index[1],ADDED_CATEGORICAL_FEATURES].eq('__MISSING__').all())
    ok('old_coarse_grid_preserved',view.geo_grid_002.equals(geo.geo_grid_002))
    ok('original_inputs_unchanged',X.equals(before_X) and X.dtypes.equals(before_X.dtypes)
       and geo.equals(before_geo) and geo.dtypes.equals(before_geo.dtypes))
    ok('fit_state_signature_excludes_targets_query',list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats'])
    reject('added_name_collision',lambda:augment_features(X.rename(columns={'numeric_1':ADDED_FEATURES[0]}),geo))
    reject('misaligned_geography',lambda:augment_features(X,geo.iloc[::-1]))
    reject('target_column_forbidden',lambda:augment_features(X.assign(y=1),geo))
    ok('raw_scale_and_nonnegative_inverse',np.array_equal(transform_target([0,1e6,2.5e6]),[0,1,2.5])
       and np.array_equal(inverse_transform([-1,0,2.5]),[0,0,2.5e6]))
    reject('negative_target',lambda:transform_target([-1]))
    reject('nonfinite_target',lambda:transform_target([np.nan]))
    reject('inverse_overflow',lambda:inverse_transform([np.finfo('float64').max]))
    fake={'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,'loss_function':'MAE','eval_metric':'MAE',
          'boosting_type':'Plain','task_type':'GPU','devices':'0','learning_rate':.065,
          'l2_leaf_reg':6,'bootstrap_type':'Bernoulli','subsample':.85,'random_strength':.5,
          'one_hot_max_size':8,'max_ctr_complexity':2,'border_count':128,'gpu_ram_part':.75,
          'allow_writing_files':False,'verbose':300}
    class FakePool:
        def __init__(self,data,label=None,**kwargs):self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=_copy_json(kwargs);self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999;self.leaves=np.full(2999,256,dtype=np.int64)
        def fit(self,pool,**kwargs):
            if kwargs:raise AssertionError('Unexpected fit arguments')
            self.pool=pool;self.feature_names_=list(pool.data.columns)
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self,pool,**kwargs):
            if pool.label is not None:raise AssertionError('Query labels forbidden')
            self.query_pool=pool;return pool.data.numeric_0.to_numpy(dtype='float64')-.5
        def get_all_params(self):return dict(self.actual)
        def get_tree_leaf_counts(self):return self.leaves.copy()
        def get_cat_feature_indices(self):return list(self.cat_indices)
    previous=(globals()['CatBoostRegressor'],globals()['Pool'],base.parameters,globals()['_base_configuration_sha256'])
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:_copy_json(fake)
    globals()['_base_configuration_sha256']=lambda:'a'*64
    try:
        ok('parameters_exact_R6_seed42',parameters()==fake)
        y=pd.Series([0,1e6,2e6,3e6],index=index)
        model,state,meta=fit_model(X,y,geo,cats)
        ok('uniform_raw_Pool_without_eval',np.array_equal(model.pool.label,y.to_numpy()/1e6)
           and set(model.pool.kwargs)=={'cat_features','thread_count'} and model.config==fake)
        ok('state_fit_only_R6_geo95_plus_pack98',fit_feature_state(X,geo,cats)==state
           and len(state['geo_state']['all_feature_names'])==95 and len(state['all_feature_names'])==98
           and state['feature_rules']['address_feature_rules']==address_pack.RULES
           and not state['fit_state_uses_y'] and not state['fit_state_uses_query'])
        ok('metadata_schema_and_no_parameter_changes',meta['feature_count']==98
           and meta['categorical_feature_count']==25 and meta['native_categorical_feature_count']==25
           and meta['parameters']==meta['baseline_parameters']==fake and meta['changed_parameter_names']==[]
           and meta['removed_feature_names']==[] and meta['added_feature_names']==ADDED_FEATURES
           and meta['fit_feature_preparation_seconds']>=0 and meta['sample_weight']=='uniform'
           and meta['weight_argument_passed_to_pool'] is False and meta['fit_uses_eval_set'] is False)
        serialized=json.dumps(state,sort_keys=True)
        restored=json.loads(serialized)
        query=X.copy(deep=True);query.loc[index[0],'address_raw']='臺北市大安區中山路9999巷9999號'
        prediction=predict_model(model,restored,query,geo,cats)
        ok('query_unseen_key_kept_no_state_mutation',model.query_pool.data.addr_lane_path.iloc[0]
           ==address_pack.features_from_X(query).addr_lane_path.iloc[0]
           and json.dumps(restored,sort_keys=True)==serialized)
        ok('predict_raw_TWD_and_batch_sequence',np.array_equal(prediction,np.array([0,.5,1.5,2.5])*1e6)
           and np.array_equal(prediction,np.concatenate([predict_model(model,restored,query.iloc[[i]],geo.iloc[[i]],cats) for i in range(4)])))
        reject('target_index_mismatch',lambda:fit_model(X,y.iloc[::-1],geo,cats))
        for key,value in [('random_seed',137),('loss_function','Huber:delta=1'),('eval_metric','RMSE'),('depth',6),('learning_rate',.0325)]:
            original=model.actual[key];model.actual[key]=value
            reject('wrong_native_'+key,lambda:check_model(model,state));model.actual[key]=original
        model.actual['learning_rate']=float(np.float32(.065))
        ok('native_float32_learning_rate_accepted',check_model(model,state)['actual_learning_rate']==float(np.float32(.065)))
        original=model.cat_indices;model.cat_indices=original[:-1]
        reject('native_category_count_or_positions',lambda:check_model(model,state));model.cat_indices=original
        damaged=_copy_json(state);damaged['all_feature_names'][-1]='wrong';damaged['state_content_sha256']=base.content_sha({k:v for k,v in damaged.items() if k!='state_content_sha256'})
        reject('state_schema_mismatch',lambda:check_state(damaged,X,cats))
    finally:
        globals()['CatBoostRegressor'],globals()['Pool'],base.parameters,globals()['_base_configuration_sha256']=previous
    return {'all_checks_passed':True,'synthetic_only':True,'real_data_config_model_read':False,
            'real_model_fit_or_score':False,'check_count':len(checks),'checks':checks}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check-only',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=True,allow_nan=False))
