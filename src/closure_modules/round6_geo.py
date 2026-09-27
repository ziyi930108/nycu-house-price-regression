"""Fixed, target-free geography transforms and one raw CatBoost component."""
from __future__ import annotations
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from prepare_data import ROOT,sha256
NUMERIC=['geo_latitude','geo_longitude','geo_log_road_length','geo_log_extent','geo_has_position','geo_nearby_fragments']
CATEGORICAL=['geo_match_status','geo_grid_002']
FEATURES=NUMERIC+CATEGORICAL
FEATURE_RULES={'version':1,'original_features_unchanged':87,'additional_numeric':NUMERIC,'additional_categorical':CATEGORICAL,'numeric_dtype':'float32','usable_statuses':['road_level','planar_intersection'],'coordinate_crs':'EPSG:4326','coordinate_order':'latitude, longitude as named','grid_degrees':0.02,'grid_encoding':'floor(raw_float64_latitude/0.02):floor(raw_float64_longitude/0.02)','missing_grid':'__MISSING__','missing_position_numeric':'NaN for latitude, longitude, log lengths and fragments; has_position=0','distance_transform':'log1p of nonnegative metres','intersection_semantics':'planar location only; no grade connectivity or exact house position claim','query_or_target_fitted_state':False,'fit_target':'raw TWD / 1e6','sample_weight':'uniform','iterations':2999,'seed':42,'no_eval_set':True,'no_early_stopping':True}

def need(ok,message):
    if not ok:raise ValueError(message)

def content_sha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
def frame_sha(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()

def encode_geo_features(matched):
    required=['geo_latitude','geo_longitude','geo_road_length_m','geo_extent_diagonal_m','geo_nearby_fragment_grouping','geo_match_status']
    need(set(required)<=set(matched.columns),'Missing geographic columns')
    need('總價元' not in matched and 'y' not in matched,'Geography encoder refuses targets')
    status=matched.geo_match_status.fillna('ineligible_address').astype(str)
    usable=status.isin(FEATURE_RULES['usable_statuses'])
    lat=pd.to_numeric(matched.geo_latitude,errors='raise').astype('float64')
    lon=pd.to_numeric(matched.geo_longitude,errors='raise').astype('float64')
    need(lat[~usable].isna().all() and lon[~usable].isna().all(),'Rejected rows have coordinates')
    need(np.isfinite(lat[usable]).all() and np.isfinite(lon[usable]).all(),'Usable coordinates not finite')
    need(lat[usable].between(20,28).all() and lon[usable].between(116,125).all(),'Coordinates out of Taiwan study bounds')
    result=pd.DataFrame(index=matched.index)
    result['geo_latitude']=lat;result['geo_longitude']=lon
    for source,target in [('geo_road_length_m','geo_log_road_length'),('geo_extent_diagonal_m','geo_log_extent')]:
        values=pd.to_numeric(matched[source],errors='raise').where(usable).astype('float64')
        need(np.isfinite(values[usable]).all() and values[usable].ge(0).all(),'Invalid usable road geometry size')
        result[target]=np.log1p(values)
    result['geo_has_position']=usable.astype('float64')
    flags=pd.to_numeric(matched.geo_nearby_fragment_grouping,errors='raise').where(usable).astype('float64')
    need(flags[usable].isin([0,1]).all(),'Unexpected fragmentation flag')
    result['geo_nearby_fragments']=flags
    result['geo_match_status']=status
    grid=pd.Series('__MISSING__',index=matched.index,dtype=object)
    grid.loc[usable]=[str(int(a))+':'+str(int(b)) for a,b in zip(np.floor(lat[usable].to_numpy()/0.02),np.floor(lon[usable].to_numpy()/0.02))]
    result['geo_grid_002']=grid
    for col in NUMERIC:result[col]=result[col].astype('float32')
    for col in CATEGORICAL:result[col]=result[col].astype(str)
    return result[FEATURES]

def check_geo(geo):
    need(list(geo.columns)==FEATURES,'Unexpected geographic feature schema')
    need(all(str(geo[c].dtype)=='float32' for c in NUMERIC),'Geographic numeric dtype differs')
    need(all(geo[c].map(lambda v:isinstance(v,str)).all() for c in CATEGORICAL),'Geographic categories must be strings')
    usable=geo.geo_has_position.eq(1)
    need(geo.geo_has_position.isin([0,1]).all(),'Position flag differs')
    need(usable.equals(geo.geo_match_status.isin(FEATURE_RULES['usable_statuses'])),'Status and position availability disagree')
    need(geo.loc[~usable,[c for c in NUMERIC if c!='geo_has_position']].isna().all().all(),'Missing position has nonmissing geometry features')
    need(geo.loc[~usable,'geo_grid_002'].eq('__MISSING__').all(),'Missing position has spatial grid')
    need(np.isfinite(geo.loc[usable,NUMERIC].to_numpy()).all(),'Usable geometry is nonfinite')

def augment_features(X,geo):
    need(len(X.columns)==87 and not set(FEATURES)&set(X.columns),'Expected original 87 features')
    need(len(X)==len(geo) and X.index.equals(geo.index),'Geographic rows/index differ')
    check_geo(geo)
    result=pd.concat([X.copy(deep=True),geo.copy(deep=True)],axis=1)
    need(result[X.columns].equals(X) and result[X.columns].dtypes.equals(X.dtypes),'Original features were changed')
    need(len(result.columns)==95,'Expected 95 total features')
    return result

def fit_feature_state(X,geo):
    augmented=augment_features(X,geo)
    state={'feature_rules':FEATURE_RULES,'source_sha256':sha256(Path(__file__)),'fit_rows':len(X),'original_columns':list(X.columns),'original_dtypes':{c:str(X[c].dtype) for c in X},'all_feature_names':list(augmented.columns),'fit_original_feature_values_sha256':frame_sha(X),'fit_geo_feature_values_sha256':frame_sha(geo),'fit_position_rows':int(geo.geo_has_position.sum()),'fit_state_uses_y':False,'fit_state_uses_query':False}
    state['state_content_sha256']=content_sha(state)
    return state

def check_state(state,X):
    need(state['feature_rules']==FEATURE_RULES and state['source_sha256']==sha256(Path(__file__)),'Geography rules/source changed')
    need(state['state_content_sha256']==content_sha({k:v for k,v in state.items() if k!='state_content_sha256'}),'Feature state changed')
    need(list(X.columns)==state['original_columns'] and {c:str(X[c].dtype) for c in X}==state['original_dtypes'],'Original feature schema changed')

def parameters():
    cfg=json.loads((ROOT/'runs/v1/models/final_config.json').read_text(encoding='utf-8'))
    component=next(c for c in cfg['components'] if c['family']=='catboost')
    return {**component['parameters'],'iterations':2999,'thread_count':4,'verbose':300}

def categories(X,supplied):
    actual=[c for c in X if pd.api.types.is_object_dtype(X[c].dtype) or pd.api.types.is_string_dtype(X[c].dtype)]
    need(set(actual)==set(supplied) and len(supplied)==21,'Original category schema differs')
    return list(supplied)+CATEGORICAL

def fit_model(X,y,geo,categorical_columns):
    started=time.monotonic();source=sha256(Path(__file__))
    y=np.asarray(y,dtype='float64');need(y.shape==(len(X),) and np.isfinite(y).all(),'Invalid fitting target')
    state=fit_feature_state(X,geo);augmented=augment_features(X,geo);cats=categories(X,categorical_columns)
    config=parameters();model=CatBoostRegressor(**config)
    model.fit(Pool(augmented,y/1e6,cat_features=cats,thread_count=4))
    need(int(model.tree_count_)==2999 and source==sha256(Path(__file__)),'Tree count/source changed during fit')
    metadata={'family':'catboost','component':'geography_raw','target_mode':'raw','parameters':config,'iterations':2999,'features':95,'feature_count':95,'categorical_feature_count':23,'categorical_features':cats,'sample_weight':'uniform','geo_feature_rules':FEATURE_RULES,'feature_state_content_sha256':state['state_content_sha256'],'script_sha256':source,'base_configuration_sha256':sha256(ROOT/'runs/v1/models/final_config.json'),'fit_target_values_sha256':hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),'target_used_for_geographic_state':False,'query_used_for_geographic_state':False,'reference_features_changed':False,'external_geography_used':True,'external_property_answers_used':False,'fit_uses_eval_set':False,'early_stopping_used':False,'elapsed_seconds':time.monotonic()-started}
    return model,state,metadata

def predict_model(model,state,X,geo,categorical_columns):
    check_state(state,X);augmented=augment_features(X,geo);cats=categories(X,categorical_columns)
    need(list(model.feature_names_)==list(augmented.columns)==state['all_feature_names'],'Saved model feature names differ')
    pred=np.maximum(np.asarray(model.predict(Pool(augmented,cat_features=cats,thread_count=4),thread_count=4),dtype='float64')*1e6,0)
    need(pred.shape==(len(X),) and np.isfinite(pred).all(),'Invalid price predictions')
    return pred
