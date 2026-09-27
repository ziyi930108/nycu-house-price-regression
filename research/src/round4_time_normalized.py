"""Fit-only time normalization: CatBoost predicts total price at the fit anchor.

No query dates or labels enter fitting. The caller adds the existing v2 future
trend exactly once when composing this anchor-date prediction with the baseline.
This module does not load competition data, train on import, or write artifacts.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from catboost import CatBoostRegressor,Pool
from prepare_data import ROOT,sha256,ID,TARGET

DROP_ABSOLUTE_TIME=('trade_year','trade_month_index','trade_day_index')
LOOKBACK_YEARS=4.0
ITERATIONS=2999
CONFIG=ROOT/'runs/v1/models/final_config.json'

def parameters():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    matches=[c for c in cfg['components'] if c['family']=='catboost' and c['target_mode']=='raw']
    if len(matches)!=1:raise ValueError('Exactly one frozen raw CatBoost configuration is required')
    return {**matches[0]['parameters'],'iterations':ITERATIONS,'thread_count':4,'verbose':300}

def prepare_anchor_target(X,y,reference):
    y=np.asarray(y,dtype='float64')
    if y.shape!=(len(X),) or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Fit targets must be finite, nonnegative, and aligned')
    days=X['trade_day_index'].to_numpy(dtype='float64')
    finite=np.isfinite(days)
    if not finite.any():raise ValueError('Fit requires at least one finite transaction date')
    anchor=float(days[finite].max())
    if reference['fit_report']['fitting_rows']!=len(X):raise ValueError('Reference and outer fit rows differ')
    if float(reference['fit_report']['hedonic']['maximum_fit_day_index'])!=anchor:
        raise ValueError('Reference and outer fit anchors differ')
    beta=float(np.asarray(reference['beta'])[3])
    if not np.isfinite(beta) or not -.08<=beta<=.20:raise ValueError('Unexpected frozen hedonic coefficient')
    selected=np.flatnonzero(finite & (days>=anchor-LOOKBACK_YEARS*365.25))
    if len(selected)<1000:raise ValueError('Insufficient recent fit rows')
    factor=np.exp(beta*(anchor-days[selected])/365.25)
    return selected,y[selected]*factor/1e6,1.0/factor,{
      'anchor_day_index':anchor,'annual_log_slope':beta,'lookback_years':LOOKBACK_YEARS,
      'outer_fit_rows':len(X),'nonfinite_fit_dates_excluded':int((~finite).sum()),'selected_fit_rows':len(selected),'selected_minimum_day_index':float(days[selected].min()),
      'normalization_factor_min':float(factor.min()),'normalization_factor_max':float(factor.max()),
      'sample_weight_min':float((1/factor).min()),'sample_weight_max':float((1/factor).max()),
      'target_formula':'(price_TWD / 1e6) * exp(beta_time * (anchor_day - trade_day) / 365.25)',
      'weight_formula':'inverse normalization factor; no price-derived sample weights',
      'weighted_loss_identity':'abs(y*G/1e6-p_anchor)/G = abs(y/1e6-p_anchor/G)',
      'prediction_semantics':'total price at training anchor date, in TWD; future trend must be added once by the caller',
      'reference_hedonic_fit_only':True,'query_rows_used_for_normalization':False}

def model_features(X):
    if ID in X or TARGET in X:raise ValueError('Identifiers or target are not model features')
    missing=[c for c in DROP_ABSOLUTE_TIME if c not in X]
    if missing:raise ValueError('Absolute-time feature schema differs: '+str(missing))
    return X.drop(columns=list(DROP_ABSOLUTE_TIME))

def fit_anchor_model(X,y,categorical_columns,reference):
    selected,target,weights,normalization=prepare_anchor_target(X,y,reference)
    features=model_features(X.iloc[selected]).reset_index(drop=True)
    cats=[c for c in categorical_columns if c in features]
    config=parameters()
    train=Pool(features,target,weight=weights,cat_features=cats,thread_count=4)
    model=CatBoostRegressor(**config)
    model.fit(train)
    meta={'family':'catboost','candidate_component':'time_normalized_raw','parameters':config,
      'normalization':normalization,'features':list(features.columns),'categorical_columns':cats,
      'dropped_absolute_time_columns':list(DROP_ABSOLUTE_TIME),'tree_count':int(model.tree_count_),
      'fit_uses_eval_set':False,'early_stopping_used':False,'test_labels_used':False,
      'source_sha256':sha256(Path(__file__)),'base_configuration_sha256':sha256(CONFIG),
      'normalization_no_future_rows':True,'extra_query_time_growth_applied_in_component':False}
    return model,meta,selected

def predict_anchor_model(model,X,categorical_columns):
    features=model_features(X)
    cats=[c for c in categorical_columns if c in features]
    pred=np.maximum(np.asarray(model.predict(Pool(features,cat_features=cats,thread_count=4),thread_count=4),dtype='float64')*1e6,0)
    if pred.shape!=(len(X),) or not np.isfinite(pred).all():raise ValueError('Invalid anchor-price predictions')
    return pred
