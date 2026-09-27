"""Rebuild v22 neural query inputs from 87 features and archived R6 geography.
Only frozen encoder, quantile-ramp and full-fit prior states are read. No fit,
train labels, or saved query predictions are needed.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import round47_nn_encode as encoder
import round54_numeric_encode as ramp_encoder

GEO_NUM=['geo_latitude','geo_longitude','geo_log_road_length','geo_log_extent','geo_has_position','geo_nearby_fragments']
GEO_CAT=['geo_match_status','geo_grid_002']
LEVELS=[['address_city','address_district','transaction_type','building_type'],
        ['address_city','transaction_type','building_type'],['transaction_type','building_type'],['transaction_type']]

def effective_area(X):
    building=X['building_area'].to_numpy(dtype=np.float64,copy=True)
    parking=X['parking_area'].to_numpy(dtype=np.float64,copy=True)
    return np.where(np.isfinite(building)&(building>0),building,np.where(np.isfinite(parking)&(parking>0),parking,1.0))

def norm(s):
    v=s.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
    return v.mask(v.isin(['','nan','None','<NA>','NaT']),'__MISSING__')

def prior_for_query(state,X):
    if state['state_role']!='full_fit_for_query' or state['uses_query_labels']:
        raise ValueError('Full-fit, query-label-free prior state required')
    if state['rules']['levels']!=LEVELS or len(state['levels'])!=len(LEVELS):
        raise ValueError('Unexpected prior hierarchy')
    columns=list(dict.fromkeys(c for level in LEVELS for c in level))
    normalized={c:norm(X[c]).to_numpy() for c in columns}
    keys=[np.asarray([json.dumps(row,ensure_ascii=False,separators=(',',':'))
          for row in zip(*(normalized[c] for c in level))],dtype=object) for level in LEVELS]
    result=np.full(len(X),float(state['global_median']),dtype=np.float64)
    available=np.ones(len(X),dtype=bool)
    for columns,key,level in zip(LEVELS,keys,state['levels']):
        if level['columns']!=columns:raise ValueError('Prior level changed')
        for i in np.flatnonzero(available):
            cell=level['cells'].get(key[i])
            if cell is not None:
                if cell['rows']<30 or not np.isfinite(cell['median']) or cell['median']<=0 or cell['weight_sum']<=0:
                    raise ValueError('Invalid prior cell')
                result[i]=cell['median'];available[i]=False
    if not np.isfinite(result).all() or not (result>0).all():raise ValueError('Invalid prior values')
    return result

def prepare_inputs(X,geo,encoder_state,numeric_state,prior_state):
    if len(X.columns)!=87 or list(geo.columns)!=GEO_NUM+GEO_CAT or not X.index.equals(geo.index):
        raise ValueError('Expected aligned original 87 features + saved R6 geo8')
    if any(str(geo[c].dtype)!='float32' for c in GEO_NUM):
        raise ValueError('Frozen geography numeric dtypes differ')
    for c in GEO_CAT:
        if not geo[c].map(lambda v:isinstance(v,str)).all():raise ValueError('Invalid geo category')
    X95=pd.concat([X.copy(deep=True),geo.copy(deep=True)],axis=1)
    qn,qc=encoder.encode(encoder_state,X95)
    ramps=ramp_encoder.transform(numeric_state,qn)
    cardinalities=np.asarray(encoder_state['cardinalities'],dtype=np.int64)
    area=effective_area(X)
    area_factor=np.asarray(area/100,dtype=np.float32)
    prior_unit=prior_for_query(prior_state,X)
    prior_factor=np.asarray(area*prior_unit/1e6,dtype=np.float32)
    return (
      {'query_num':ramps,'query_cat':qc,'cardinalities':cardinalities,'query_factor_million':area_factor},
      {'query_num':qn,'query_cat':qc,'cardinalities':cardinalities,'query_factor_million':prior_factor},
    )

