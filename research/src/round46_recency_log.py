"""R46 fixed fit-date recency-weighted log CatBoost. State stays the unchanged R12 X-only encoder.

Only the training Pool differs: 365-day half-life weights, normalized on fit dates alone.
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
COMPONENT='geography_recency_log_catboost'
TARGET_MODE=original.TARGET_MODE
TARGET_FORMULA=original.TARGET_FORMULA
INVERSE_FORMULA=original.INVERSE_FORMULA
WEIGHT_FORMULA='2**(-(max_fit(trade_day_index) - trade_day_index) / 365) / mean_fit(2**(-(max_fit(trade_day_index) - trade_day_index) / 365))'
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
                     weight_normalization_fit_only=True,weights_clipped=False,weight_half_life_days=365,
                     weights_use_fit_target=False,weights_use_query=False,weight_floor=None,
                     legacy_state_training_fields_are_provenance_only=True)


def _sources():
    return {**original._sources(),'src/round46_recency_log.py':sha256(Path(__file__))}


def fit_weights(X):
    """Ordered weights from this fit frame only; no target/query input or floor."""
    _check_X(X)
    base.need(len(X)>0 and {'trade_day_index','trade_date_invalid','trade_date_out_of_range'}<=set(X.columns),
              'Fit recency requires nonempty date features')
    days=_float_vector(X['trade_day_index'],'Fit transaction day index')
    for flag in ('trade_date_invalid','trade_date_out_of_range'):
        values=_float_vector(X[flag],'Fit date flag '+flag)
        base.need((values==0).all(),'Fit dates must be valid/in-range; no replacement policy')
    maximum=float(days.max())
    with np.errstate(over='raise',invalid='raise'):
        age=maximum-days
        base.need(np.isfinite(age).all() and (age>=0).all(),'Fit date ages must be finite nonnegative')
        unnormalized=np.exp2(-age/365.0)
    mean=float(np.mean(unnormalized))
    base.need(np.isfinite(unnormalized).all() and (unnormalized>0).all() and np.isfinite(mean) and mean>0,
              'Fit recency weights/mean must be positive finite; no floor or clipping')
    weights=np.ascontiguousarray(unnormalized/mean,dtype=np.float64)
    base.need(np.isfinite(weights).all() and (weights>0).all() and abs(float(weights.mean())-1.0)<1e-12,
              'Normalized fit weights must be positive finite with mean one')
    summary={'rows':len(weights),'min':float(weights.min()),'max':float(weights.max()),
             'mean':float(weights.mean()),'sum':float(weights.sum()),'normalization_mean_fit_only':mean,
             'max_fit_trade_day_index':maximum,'min_fit_trade_day_index':float(days.min())}
    return weights,summary


def fit_model(X, y, geo, cats):
    """Fit the fixed log target once; supplied fit dates alone determine weights."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fitting target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fitting target length differs from X')
    transformed = transform_target(raw)
    weights, weight_summary = fit_weights(X)
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
        'fit_weight_date_values_sha256': values_sha256(X['trade_day_index']),
        'weight_half_life_days': 365,
        'weight_floor': None,
        'weights_use_fit_target': False,
        'weights_use_query': False,
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
    """One small fake fit checks weights actually reach Pool and preserve row order."""
    from house_features import build_features,get_categorical_columns
    raw=pd.DataFrame({'鄉鎮市區':['中山區']*4,'土地位置建物門牌':['臺北市中山區中山路1號']*4,
        '交易年月日':[1110101,1110201,1110301,1110401],'建物移轉總面積平方公尺':[80,100,120,140],
        '建案名稱':['甲','乙','丙','丁']},index=pd.Index([7,11,19,31]))
    X=build_features(raw);X['trade_day_index']=np.array([1000,1365,1730,2095],dtype=np.float32)
    cats=get_categorical_columns(X)
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
    saved=globals()['Pool'],globals()['CatBoostRegressor']
    try:
        globals()['Pool'],globals()['CatBoostRegressor']=FakePool,FakeCat
        model,state,meta=fit_model(X,y,geo,cats)
        expected=np.array([.125,.25,.5,1.],dtype=np.float64)/.46875
        reordered,_=fit_weights(X.iloc[::-1])
        base.need(np.array_equal(model.pool.weight,expected) and np.array_equal(reordered,expected[::-1])
            and model.pool.weight.dtype==np.float64 and model.pool.data.index.equals(X.index)
            and np.array_equal(model.pool.label,np.log1p(y.to_numpy()/1e6))
            and model.pool.data.equals(base.augment_features(X,geo)) and len(model.pool.categories)==23
            and model.config==original.parameters() and state==original.fit_feature_state(X,geo,cats)
            and meta['fit_weights_sha256']==values_sha256(expected)
            and meta['fit_weight_summary']['max_fit_trade_day_index']==2095
            and meta['fit_weight_summary']['normalization_mean_fit_only']==.46875,
            'FakePool fit-date weights/log-label/index contract differs')
        try:fit_model(X,y.iloc[::-1],geo,cats)
        except ValueError:pass
        else:raise AssertionError('Target Series index mismatch was accepted')
    finally:globals()['Pool'],globals()['CatBoostRegressor']=saved
    return ['fit_date_half_life_weights_and_log_labels_reach_Pool_with_original95_and_state',
            'row_order_preserved_and_target_index_mismatch_rejected']


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true',required=True);parser.parse_args()
    checks=self_check();print(json.dumps({'status':'passed','checks':checks,'check_count':len(checks),'real_fit':False,
                                         'source_sha256':sha256(Path(__file__))},ensure_ascii=False,indent=2))
