"""R44 fixed fit-normalized weighted log CatBoost. State stays the unchanged R12 X-only encoder.

Only the training Pool differs: weights=(1+y_TWD/1e6)/mean_fit(1+y_TWD/1e6).
Original R12 state training fields describe its historical provenance, not this Pool.
No query target, cap, calibration, reference, or spatial operation is added here.
"""
from __future__ import annotations
import copy, hashlib, inspect, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round12_log_geo as original
from prepare_data import ROOT, sha256

base=original.base
COMPONENT='geography_weighted_log_catboost'
TARGET_MODE=original.TARGET_MODE
TARGET_FORMULA=original.TARGET_FORMULA
INVERSE_FORMULA=original.INVERSE_FORMULA
WEIGHT_FORMULA='(1 + y_TWD / 1e6) / mean_fit(1 + y_TWD / 1e6)'
parameters=original.parameters
fit_feature_state=original.fit_feature_state
check_state=original.check_state
augment_features=base.augment_features
categories=base.categories
transform_target=original.transform_target
inverse_transform=original.inverse_transform
predict_model=original.predict_model
values_sha256=original.values_sha256
_copy_json=original._copy_json
_check_X=original._check_X
_float_vector=original._float_vector
FEATURE_RULES=copy.deepcopy(original.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT,sample_weight=WEIGHT_FORMULA,weight_argument_passed_to_pool=True,
                     weight_normalization_fit_only=True,weights_clipped=False,
                     legacy_state_training_fields_are_provenance_only=True)


def _sources():
    return {**original._sources(),'src/round44_weighted_log.py':sha256(Path(__file__))}


def fit_weights(y):
    raw=_float_vector(y,'Fit target for weights')
    base.need(len(raw)>0 and (raw>=0).all(),'Weights require nonempty nonnegative fit labels')
    unnormalized=1.0+raw/1e6
    mean=float(np.mean(unnormalized))
    base.need(np.isfinite(unnormalized).all() and (unnormalized>0).all() and np.isfinite(mean) and mean>0,
              'Unnormalized fit weights/mean must be positive finite')
    weights=np.ascontiguousarray(unnormalized/mean,dtype=np.float64)
    base.need(np.isfinite(weights).all() and (weights>0).all() and abs(float(weights.mean())-1.0)<1e-12,
              'Normalized fit weights must be positive finite with mean one')
    summary={'rows':len(weights),'min':float(weights.min()),'max':float(weights.max()),
             'mean':float(weights.mean()),'sum':float(weights.sum()),'normalization_mean_fit_only':mean}
    return weights,summary


def fit_model(X, y, geo, cats):
    """Fit the fixed log target once; supplied fit labels alone determine weights."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fitting target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fitting target length differs from X')
    transformed = transform_target(raw)
    weights, weight_summary = fit_weights(raw)
    weight_sha = values_sha256(weights)
    raw_sha = values_sha256(raw)
    transformed_sha = values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    augmented = base.augment_features(X, geo)
    categories = base.categories(X, cats)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, weight=weights, cat_features=categories, thread_count=4))
    base.need(int(model.tree_count_) == 2999, 'Expected all fixed 2999 trees')
    check_state(state, X, cats)
    base.need(parameters() == config, 'Base parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256'],
              'Fit feature inputs changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Target values passed to Pool changed')
    base.need(values_sha256(weights) == weight_sha, 'Fit weights changed during training')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Model feature names differ')
    metadata = {
        'family': 'catboost',
        'component': COMPONENT,
        'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA,
        'parameters': config,
        'parameters_sha256': base.content_sha(config),
        'iterations': 2999,
        'features': 95,
        'feature_count': 95,
        'categorical_feature_count': 23,
        'feature_columns': list(augmented.columns),
        'categorical_features': categories,
        'categorical_columns': categories,
        'sample_weight': WEIGHT_FORMULA,
        'weight_argument_passed_to_pool': True,
        'weight_formula': WEIGHT_FORMULA,
        'fit_weights_sha256': weight_sha,
        'fit_weight_summary': weight_summary,
        'weight_value_hash_encoding': 'ordered C-contiguous little-endian float64',
        'weight_normalization_fit_only': True,
        'weights_clipped': False,
        'legacy_state_training_fields_are_provenance_only': True,
        'feature_rules': _copy_json(FEATURE_RULES),
        'geo_feature_rules': _copy_json(FEATURE_RULES['geo_encoder_rules']),
        'legacy_geo_state_training_fields_are_not_this_models_training_declaration': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': sha256(Path(__file__)),
        'source_sha256': _sources(),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_rows': int(len(X)),
        'fit_original_feature_values_sha256': state['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['geo_state']['fit_geo_feature_values_sha256'],
        'fit_target_values_sha256': raw_sha,
        'fit_original_target_values_sha256': raw_sha,
        'fit_transformed_target_values_sha256': transformed_sha,
        'target_value_hash_encoding': 'ordered C-contiguous little-endian float64',
        'target_used_for_geographic_state': False,
        'query_used_for_geographic_state': False,
        'query_labels_used': False,
        'fit_uses_eval_set': False,
        'early_stopping_used': False,
        'calibration_applied': False,
        'extra_price_cap_applied': False,
        'reference_or_trend_applied_here': False,
        'reference_features_changed': False,
        'external_geography_used': True,
        'external_property_answers_used': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    return model, state, metadata


def self_check():
    """One tiny FakePool fit and numerical/index checks; never a real estimator fit."""
    from house_features import build_features,get_categorical_columns
    raw=pd.DataFrame({'鄉鎮市區':['中山區']*4,'土地位置建物門牌':['臺北市中山區中山路1號']*4,
        '交易年月日':[1110101,1110201,1110301,1110401],'建物移轉總面積平方公尺':[80,100,120,140],
        '建案名稱':['甲','乙','丙','丁']},index=pd.Index([7,11,19,31]))
    X=build_features(raw);cats=get_categorical_columns(X)
    matched=pd.DataFrame({'geo_latitude':[25.]*4,'geo_longitude':[121.5]*4,'geo_road_length_m':[1000.]*4,
        'geo_extent_diagonal_m':[900.]*4,'geo_nearby_fragment_grouping':[0.]*4,'geo_match_status':['road_level']*4},index=X.index)
    geo=base.encode_geo_features(matched);y=pd.Series([0.,1e6,3e6,7e6],index=X.index)
    class FakePool:
        def __init__(self,data,label=None,weight=None,cat_features=None,thread_count=None):
            self.data=data.copy(deep=True);self.label=np.array(label,copy=True);self.weight=np.array(weight,copy=True)
            self.categories=list(cat_features)
    class FakeCat:
        def __init__(self,**config):self.config=config;self.tree_count_=config['iterations']
        def fit(self,pool):self.pool=pool;self.feature_names_=list(pool.data.columns);return self
    checks=[]
    def ok(name,value):base.need(bool(value),'Synthetic: '+name);checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,TypeError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    saved=globals()['Pool'],globals()['CatBoostRegressor']
    try:
        globals()['Pool'],globals()['CatBoostRegressor']=FakePool,FakeCat
        model,state,meta=fit_model(X,y,geo,cats);expected=np.array([1.,2.,4.,8.])/3.75
        ok('log_labels_and_normalized_fit_weights_reach_Pool',np.array_equal(model.pool.label,np.log1p(y.to_numpy()/1e6))
           and np.array_equal(model.pool.weight,expected) and model.pool.weight.dtype==np.float64)
        ok('all95_values_dtypes_indices_and23_categories_unchanged',model.pool.data.equals(base.augment_features(X,geo))
           and model.pool.data.dtypes.equals(base.augment_features(X,geo).dtypes) and model.pool.data.index.equals(X.index)
           and len(model.pool.data.columns)==95 and len(model.pool.categories)==23)
        ok('parameters_exactly_original_R6',model.config==original.parameters() and model.config['iterations']==2999)
        ok('state_unchanged_X_only_R12',state==original.fit_feature_state(X,geo,cats) and state['fit_state_uses_y'] is False)
        ok('weight_and_both_target_hashes',meta['fit_weights_sha256']==values_sha256(expected)
           and meta['fit_original_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(model.pool.label)
           and meta['fit_weight_summary']['normalization_mean_fit_only']==3.75)
        ok('weighted_metadata_not_legacy_uniform_declaration',meta['weight_argument_passed_to_pool'] is True
           and meta['sample_weight']==WEIGHT_FORMULA and meta['legacy_state_training_fields_are_provenance_only'] is True)
        ok('inverse_TWD_original_and_query_API_has_no_label',np.allclose(inverse_transform(model.pool.label),y,rtol=1e-14)
           and list(inspect.signature(predict_model).parameters)==['model','state','X','geo','cats'])
        reject('Series_index_mismatch',lambda:fit_model(X,y.iloc[::-1],geo,cats))
        reject('negative_fit_weight_label',lambda:fit_weights(np.array([-1.])))
        reject('nonfinite_fit_weight_label',lambda:fit_weights(np.array([np.inf])))
        reject('empty_fit_weight_label',lambda:fit_weights(np.array([])))
    finally:globals()['Pool'],globals()['CatBoostRegressor']=saved
    return checks


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true',required=True);parser.parse_args()
    checks=self_check();print(json.dumps({'status':'passed','checks':checks,'check_count':len(checks),'real_fit':False,
                                         'source_sha256':sha256(Path(__file__))},ensure_ascii=False,indent=2))
