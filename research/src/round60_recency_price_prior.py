"""R60 fixed 365-day weighted-median unit priors; group OOF, fit labels/dates only."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import round52_price_prior as original

LEVELS=copy.deepcopy(original.LEVELS)
ROOT=Path(__file__).resolve().parents[1]
need,values_sha=original.need,original.values_sha
DATE_COLUMNS=('trade_day_index','trade_date_invalid','trade_date_out_of_range')
WEIGHT_RULES={'half_life_days':365,'anchor':'max valid reference trade_day_index; each fold complement separately',
    'valid':'finite trade_day_index and both finite date flags equal zero',
    'formula':'exp2(-(anchor-day)/365) / mean_valid_reference(exp2(-(anchor-day)/365))',
    'invalid_or_missing_date_weight':1.0,'normalization':'valid reference rows only; neutral rows remain 1',
    'floor':None,'clipping':False,'weights_use_targets':False,'weights_use_query':False,
    'no_valid_reference_dates':'raise','nonfinite_date_flags':'raise'}
MEDIAN_RULE='stable ascending unit sort; first cumulative weight >= half total; exact half uses adjacent midpoint; no tolerance'
RULES=copy.deepcopy(original.RULES)
RULES.update(cell_estimator='365-day recency weighted median; fallback if fewer than 30 raw rows',
    global_estimator='same weighted median using all reference rows only; positive finite',
    global_uses_full_fit_for_oof=False,weights=WEIGHT_RULES,weighted_median=MEDIAN_RULE,
    support_measure='original reference row count, including neutral-date rows')
RULES['global']='weighted median of this reference only; no 30-row threshold'


def weighted_median(values,weights):
    v=np.asarray(values,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    need(v.ndim==w.ndim==1 and v.shape==w.shape and len(v)>0 and np.isfinite(v).all()
         and (v>0).all() and np.isfinite(w).all() and (w>0).all(),'Median needs positive finite values/weights')
    order=np.argsort(v,kind='stable');v=v[order];w=w[order]/w.max()
    need((w>0).all() and np.isfinite(w).all(),'Weight scaling underflow/nonfinite; no floor')
    cumulative=np.cumsum(w,dtype=np.float64);total=float(cumulative[-1]);half=total/2.0
    need(np.isfinite(total) and total>0,'Invalid cumulative weight')
    j=int(np.searchsorted(cumulative,half,side='left'));value=float(v[j])
    if cumulative[j]==half and j+1<len(v):value=float(v[j]+(v[j+1]-v[j])/2.0)
    need(np.isfinite(value) and value>0,'Invalid weighted median')
    return value


def _dates(X):
    need(set(DATE_COLUMNS)<=set(X.columns),'Missing fit date columns')
    dates=tuple(X[c].to_numpy(dtype=np.float64,copy=True) for c in DATE_COLUMNS)
    need(np.isfinite(dates[1]).all() and np.isfinite(dates[2]).all(),'Date flags must be finite')
    return dates


def _weights(dates):
    days,invalid,outside=dates
    need(days.ndim==1 and len(days)>0 and invalid.shape==outside.shape==days.shape
         and np.isfinite(invalid).all() and np.isfinite(outside).all(),'Invalid reference date arrays/flags')
    valid=np.isfinite(days)&(invalid==0)&(outside==0)
    need(valid.any(),'No valid reference dates; no guessed anchor')
    anchor=float(days[valid].max());age=anchor-days[valid]
    need(np.isfinite(age).all() and (age>=0).all(),'Invalid reference date ages')
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        u=np.exp2(-age/365.0)
    denominator=float(u.mean(dtype=np.float64))
    need(np.isfinite(u).all() and (u>0).all() and np.isfinite(denominator) and denominator>0,
         'Recency weight underflow/nonfinite; no floor')
    w=np.ones(len(days),dtype=np.float64);w[valid]=u/denominator
    need(np.isfinite(w).all() and (w>0).all() and abs(float(w.mean())-1)<1e-12,'Invalid normalized weights')
    summary={'reference_rows':len(w),'valid_date_rows':int(valid.sum()),'neutral_date_rows':int((~valid).sum()),
        'invalid_flag_rows':int((invalid!=0).sum()),'out_of_range_flag_rows':int((outside!=0).sum()),
        'max_valid_reference_trade_day_index':anchor,'min_valid_reference_trade_day_index':float(days[valid].min()),
        'normalization_mean_valid_reference_only':denominator,'weight_min':float(w.min()),'weight_max':float(w.max()),
        'weight_mean':float(w.mean()),'weight_sum':float(w.sum(dtype=np.float64)),
        'weights_sha256':values_sha(w),'date_values_sha256':values_sha(days),
        'invalid_flags_sha256':values_sha(invalid),'out_of_range_flags_sha256':values_sha(outside),
        'valid_mask_sha256':values_sha(valid,'u1'),'weight_rules':copy.deepcopy(WEIGHT_RULES)}
    return w,valid,summary


def _build(keys,units,dates,mask,role):
    values=units[mask];weights,valid,summary=_weights(tuple(a[mask] for a in dates))
    global_median=weighted_median(values,weights);levels=[]
    for columns,key in zip(LEVELS,keys):
        positions=pd.DataFrame({'key':key[mask]}).groupby('key',sort=True).indices
        cells={}
        for k,ix in positions.items():
            if len(ix)>=30:
                cells[str(k)]={'rows':len(ix),'median':weighted_median(values[ix],weights[ix]),
                    'weight_sum':float(weights[ix].sum(dtype=np.float64)),
                    'valid_date_rows':int(valid[ix].sum()),'neutral_date_rows':int((~valid[ix]).sum())}
        levels.append({'columns':list(columns),'cell_count':len(cells),'cells':cells,
            'minimum_observed_support':min((c['rows'] for c in cells.values()),default=None)})
    return {'rules':copy.deepcopy(RULES),'fit_rows':int(mask.sum()),'global_median':global_median,
        'global_raw_rows':int(mask.sum()),'global_weight_sum':summary['weight_sum'],'date_support':summary,
        'levels':levels,'uses_fit_labels':True,'uses_query_labels':False,'state_role':role}


def _predict(state,keys,return_diagnostics=False):
    need(state['rules']==RULES and len(state['levels'])==len(LEVELS),'R60 rule/schema differs')
    value=float(state['global_median']);need(np.isfinite(value) and value>0,'Invalid global prior')
    result=np.full(len(keys[0]),value,dtype=np.float64);unassigned=np.ones(len(result),dtype=bool)
    counts={**{'|'.join(c):0 for c in LEVELS},'global':0}
    for columns,key,level in zip(LEVELS,keys,state['levels']):
        need(level['columns']==columns,'Prior hierarchy differs')
        for i in np.flatnonzero(unassigned):
            cell=level['cells'].get(key[i])
            if cell is not None:
                need(cell['rows']>=30 and np.isfinite(cell['median']) and cell['median']>0
                     and np.isfinite(cell['weight_sum']) and cell['weight_sum']>0,'Invalid weighted prior cell')
                result[i]=cell['median'];unassigned[i]=False;counts['|'.join(columns)]+=1
    counts['global']=int(unassigned.sum())
    need(np.isfinite(result).all() and (result>0).all(),'Invalid prediction prior')
    return (result,counts) if return_diagnostics else result


def predict_prior(state,Xquery,return_diagnostics=False):
    """Float64 TWD/m2, original row order; no query dates, labels or fitting."""
    need(state['state_role']=='full_fit_for_query','Query requires the full-fit state')
    return _predict(state,original._keys(Xquery),return_diagnostics)


def fit_oof_prior(Xfit,yfit,source_indices):
    """Return (OOF unit float64, JSON-safe full-fit state, folds int64, evidence)."""
    keys=original._keys(Xfit);n=len(Xfit);indices=np.asarray(source_indices);dates=_dates(Xfit)
    need(n>0 and indices.dtype.kind in 'iu' and indices.shape==(n,) and (indices>=0).all()
         and len(np.unique(indices))==n,'Invalid ordered canonical source indices')
    if isinstance(yfit,pd.Series):need(yfit.index.equals(Xfit.index),'Fit target Series index differs')
    y=np.asarray(yfit,dtype=np.float64)
    need(y.shape==(n,) and np.isfinite(y).all() and (y>0).all(),'Fit prices must be positive finite and aligned')
    area=original.effective_area(Xfit)
    with np.errstate(over='raise',invalid='raise',divide='raise'):units=y/area
    need(np.isfinite(units).all() and (units>0).all(),'Invalid fit unit prices')
    groups,fallback=original.grouping.cluster_keys(Xfit,indices)
    assignments={key:int.from_bytes(hashlib.sha256(('42|'+key).encode('utf-8')).digest()[:8],'little')%5
                 for key in np.unique(groups)}
    folds=np.asarray([assignments[key] for key in groups],dtype=np.int64)
    oof=np.empty(n,dtype=np.float64);records=[]
    support=lambda st:[{k:level[k] for k in ('columns','cell_count','minimum_observed_support')} for level in st['levels']]
    for fold in range(5):
        held=folds==fold;reference=~held
        record={'fold':fold,'held_rows':int(held.sum()),'query_rows':int(held.sum()),'fit_rows':int(reference.sum()),
            'reference_rows':int(reference.sum()),'held_groups':len(np.unique(groups[held])),
            'reference_groups':len(np.unique(groups[reference])),
            'reference_indices_sha256':values_sha(indices[reference],'<i8'),
            'reference_target_sha256':values_sha(y[reference]),'reference_unit_sha256':values_sha(units[reference])}
        if held.any():
            st=_build(keys,units,dates,reference,'complement_fit_for_held_fold')
            oof[held],record['held_prior_level_counts']=_predict(st,[key[held] for key in keys],True)
            record.update(reference_global_median=st['global_median'],date_support=st['date_support'],
                          reference_level_support=support(st))
        else:
            record.update(reference_global_median=None,date_support=None,reference_level_support=[],
                          held_prior_level_counts={**{'|'.join(c):0 for c in LEVELS},'global':0})
        records.append(record)
    state=_build(keys,units,dates,np.ones(n,dtype=bool),'full_fit_for_query')
    source_paths=[Path(__file__),Path(original.__file__),Path(original.grouping.__file__),ROOT/'src/round19_unit_geo.py']
    evidence={'rules':copy.deepcopy(RULES),'fit_rows':n,'groups':len(assignments),'fallback_counts':fallback,
        'folds':records,'full_state_level_support':support(state),'full_fit_date_support':state['date_support'],
        'full_fit_global_median':state['global_median'],'training_prior_role':'group-fold OOF; complement labels and dates only',
        'query_prior_role':'full fit state','uses_fit_labels':True,'uses_query_labels':False,'chronological_oof':False,
        'fold_assignment_identical_to_R52_rule':True,'fit_target_sha256':values_sha(y),'effective_area_sha256':values_sha(area),
        'fit_unit_sha256':values_sha(units),'source_indices_sha256':values_sha(indices,'<i8'),
        'fold_ids_sha256':values_sha(folds,'<i8'),'oof_prior_sha256':values_sha(oof),
        'group_keys_sha256':hashlib.sha256(json.dumps(groups.tolist(),ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        'state_content_sha256':hashlib.sha256(json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        'source_sha256':{p.resolve().relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    return oof,state,folds,evidence


def self_check():
    for v in ([1.,3.],[1.,2.,3.],[1.,2.,2.,9.]):
        need(weighted_median(v,np.ones(len(v)))==np.median(v),'Equal-weight ordinary median')
    w,_,summary=_weights((np.array([1000.,1365.,np.nan]),np.array([0.,0.,1.]),np.zeros(3)))
    need(np.array_equal(w,np.array([2/3,4/3,1.])) and summary['neutral_date_rows']==1,'Invalid date neutral weight')
    need(weighted_median([1.,9.],w[:2])==9.,'Recency direction')
    n=360
    X=pd.DataFrame({'project_name':[f'p{i//6}' for i in range(n)],'address_city':['A']*180+['B']*180,
        'address_district':['D']*n,'address_raw':['__MISSING__']*n,'address_road':['__MISSING__']*n,
        'address_section':['__MISSING__']*n,'transaction_type':['house']*n,'building_type':['apartment']*n,
        'building_area':np.full(n,100,dtype=np.float32),'parking_area':np.zeros(n,dtype=np.float32),
        'trade_day_index':np.arange(n,dtype=np.float64)+1000,'trade_date_invalid':np.zeros(n),
        'trade_date_out_of_range':np.zeros(n)})
    y=pd.Series((np.arange(n,dtype=np.float64)+100)*100,index=X.index);indices=np.arange(1000,1000+n,dtype=np.int64)
    before=X.copy(deep=True);yb=y.copy();oof,state,folds,evidence=fit_oof_prior(X,y,indices)
    need(set(folds)==set(range(5)),'Fixture five folds')
    keys,_=original.grouping.cluster_keys(X,indices)
    expected=np.array([int.from_bytes(hashlib.sha256(('42|'+k).encode()).digest()[:8],'little')%5 for k in keys],dtype=np.int64)
    need(np.array_equal(expected,folds),'R52 canonical fold rule')
    revisedX=X.copy();revisedY=y.copy();held=folds==0
    revisedX.loc[held,'trade_day_index']+=3650;revisedY.loc[held]*=7
    changed,_,same,e2=fit_oof_prior(revisedX,revisedY,indices)
    need(np.array_equal(same,folds) and oof[held].tobytes()==changed[held].tobytes()
         and evidence['folds'][0]==e2['folds'][0],'Held-fold labels or dates leaked')
    q=X.iloc[[0,359,30]].drop(columns=list(DATE_COLUMNS)).copy()
    restored=json.loads(json.dumps(state,allow_nan=False));a,counts=predict_prior(restored,q,True)
    need(a[0]!=a[1] and a.tobytes()==predict_prior(state,q).tobytes()
         and a[::-1].tobytes()==predict_prior(state,q.iloc[::-1]).tobytes() and sum(counts.values())==3,'Reload/query order')
    q.loc[:,['address_city','address_district','transaction_type','building_type']]='unknown'
    need(np.array_equal(predict_prior(state,q),np.full(3,state['global_median'])),'Unknown global fallback')
    small=X.iloc[:31].copy();small.loc[:,'trade_day_index']=0.;small.loc[30,'trade_day_index']=3650.
    small.loc[0,'trade_day_index']=np.nan;small.loc[0,'trade_date_invalid']=1.;small.loc[30,'address_district']='rare'
    smallkeys=original._keys(small)
    st=_build(smallkeys,np.arange(1.,32.),_dates(small),np.ones(31,dtype=bool),'full_fit_for_query')
    cell=st['levels'][0]['cells'][smallkeys[0][0]]
    need(cell['rows']==30 and cell['neutral_date_rows']==1 and cell['weight_sum']<30
         and smallkeys[0][30] not in st['levels'][0]['cells'],'Support must count raw rows including neutral')
    rejected=0
    badflags=X.copy();badflags.loc[0,'trade_date_invalid']=np.nan
    calls=[lambda:weighted_median([0.],[1.]),lambda:fit_oof_prior(X,np.zeros(n),indices),
           lambda:fit_oof_prior(badflags,y,indices),
           lambda:_weights((np.full(2,np.nan),np.ones(2),np.zeros(2))),
           lambda:_weights((np.array([0.,1e9]),np.zeros(2),np.zeros(2)))]
    for call in calls:
        try:call()
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid values/dates accepted')
    need(X.equals(before) and y.equals(yb),'Inputs mutated')
    return {'status':'passed','equal_weights_ordinary_median':True,'recency_direction':True,
        'held_fold_label_and_date_invariance':True,'original_fold_rule':True,'invalid_date_neutral_one':True,
        'raw_rows_support_includes_neutral':True,'JSON_roundtrip_query_order':True,'unknown_global_fallback':True,
        'invalid_value_date_rejections':rejected,'real_data_read':False,'GPU_calls':0}


if __name__=='__main__':print(json.dumps(self_check(),ensure_ascii=False))
