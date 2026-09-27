"""R72 road-relative supervised input: saved R60 folds/base priors remain fixed.

Only fit labels build five complement road tables and one full-fit query table.
No loaders, model fitting, query labels, base-prior refitting, scaling or clipping.
"""
from __future__ import annotations
import copy, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import round60_recency_price_prior as prior

base=prior.original
need,values_sha=prior.need,prior.values_sha
ROOT=Path(__file__).resolve().parents[1]
COLUMNS=['address_city','address_district','address_road','address_section','transaction_type','building_type']
MANDATORY=COLUMNS[:3]
RULES={'columns':COLUMNS,'mandatory_known_columns':MANDATORY,'minimum_cell_rows':30,
       'unit':'TWD per square metre of effective area','area':base.RULES['area'],
       'folds':5,'seed':42,'fold_hash':base.RULES['fold_hash'],
       'weights':copy.deepcopy(prior.WEIGHT_RULES),'weighted_median':prior.MEDIAN_RULE,
       'weight_population':'whole fold complement/full fit first; then select road-cell rows',
       'feature':'log(road_weighted_median)-log(saved_R60_base_unit_prior)',
       'dtype':'float32','fallback':0.0,'support_fallback':0,'coarser_road_backoff':False,
       'center':False,'standardize':False,'clipping':False,
       'training':'complete saved project/fallback fold excluded',
       'query':'full fit road state plus saved full-fit base prior',
       'base_prior_refitted':False,'encoder_or_training_weights_changed':False}


def _hash_json(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sources():
    paths=[Path(__file__),Path(prior.__file__),Path(base.__file__),
           Path(base.grouping.__file__),ROOT/'src/round19_unit_geo.py']
    return {p.resolve().relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _keys(X):
    need(isinstance(X,pd.DataFrame) and X.columns.is_unique and X.index.is_unique,'Unique X frame/index required')
    need(set(COLUMNS)<=set(X.columns),'Missing road-key columns')
    normalized={c:base.grouping.norm(X[c]).to_numpy() for c in COLUMNS}
    eligible=np.ones(len(X),dtype=bool)
    for c in MANDATORY:eligible &= normalized[c]!='__MISSING__'
    keys=np.asarray([json.dumps(row,ensure_ascii=False,separators=(',',':'))
                     for row in zip(*(normalized[c] for c in COLUMNS))],dtype=object)
    return keys,eligible


def _folds(X,indices):
    groups,fallback=base.grouping.cluster_keys(X,indices)
    assignment={key:int.from_bytes(hashlib.sha256(('42|'+key).encode('utf-8')).digest()[:8],'little')%5
                for key in np.unique(groups)}
    return np.asarray([assignment[key] for key in groups],dtype=np.int64),groups,fallback


def _base_prior(values,X):
    if isinstance(values,pd.Series):need(values.index.equals(X.index),'Base prior Series index differs')
    result=np.asarray(values,dtype=np.float64)
    need(result.shape==(len(X),) and np.isfinite(result).all() and (result>0).all(),'Base prior must be aligned positive finite TWD/m2')
    return result


def _build(keys,eligible,units,dates,mask,role):
    # Date normalization includes ALL reference rows, even rows with missing roads.
    weights,valid,date_summary=prior._weights(tuple(a[mask] for a in dates))
    k,e,u=keys[mask],eligible[mask],units[mask]
    positions=pd.DataFrame({'key':k[e],'position':np.flatnonzero(e)}).groupby('key',sort=True)['position']
    cells={}
    for key,positions_in_reference in positions:
        ix=positions_in_reference.to_numpy(dtype=np.int64)
        if len(ix)>=30:
            cells[str(key)]={'rows':len(ix),'median':prior.weighted_median(u[ix],weights[ix]),
                            'weight_sum':float(weights[ix].sum(dtype=np.float64)),
                            'valid_date_rows':int(valid[ix].sum()),'neutral_date_rows':int((~valid[ix]).sum())}
    return {'rules':copy.deepcopy(RULES),'state_role':role,'fit_rows':int(mask.sum()),
            'eligible_reference_rows':int(e.sum()),'cell_count':len(cells),
            'minimum_observed_support':min((c['rows'] for c in cells.values()),default=None),
            'cells':cells,'date_support':date_summary,'uses_fit_labels':True,'uses_query_labels':False}


def _lookup(state,keys,eligible,base_values):
    need(state['rules']==RULES and state['cell_count']==len(state['cells']),'Road state rules/cell schema changed')
    n=len(keys);delta=np.zeros(n,dtype=np.float32);support=np.zeros(n,dtype=np.int64)
    for i in np.flatnonzero(eligible):
        cell=state['cells'].get(keys[i])
        if cell is None:continue
        need(isinstance(cell['rows'],int) and 30<=cell['rows']<=state['fit_rows']
             and np.isfinite(cell['median']) and cell['median']>0
             and np.isfinite(cell['weight_sum']) and cell['weight_sum']>0,'Invalid saved road cell')
        value=np.log(np.float64(cell['median']))-np.log(base_values[i])
        need(np.isfinite(value),'Nonfinite road-relative log value')
        delta[i]=value;support[i]=cell['rows']
    need(np.isfinite(delta).all() and (delta[support==0]==0).all(),'Invalid feature or fallback')
    diagnostics={'rows':n,'selected_rows':int((support>0).sum()),'fallback_rows':int((support==0).sum()),
                 'reason_counts':{'selected':int((support>0).sum()),'missing_required_key':int((~eligible).sum()),
                                  'unsupported_key_or_below_30':int((eligible&(support==0)).sum())},
                 'base_unit_prior_sha256':values_sha(base_values),'delta_sha256':values_sha(delta,'<f4'),
                 'selected_support_sha256':values_sha(support,'<i8'),'query_labels_used':False}
    return delta,support,diagnostics


def predict_road_prior(state,Xquery,base_unit_prior,return_diagnostics=False):
    """Return (delta float32, selected raw support int64[, diagnostics]) in X row order."""
    need(state['state_role']=='full_fit_for_query' and not state['uses_query_labels'],'Query requires full-fit road state')
    keys,eligible=_keys(Xquery)
    result=_lookup(state,keys,eligible,_base_prior(base_unit_prior,Xquery))
    return result if return_diagnostics else result[:2]


def fit_oof_road_prior(Xfit,yfit,source_indices,saved_fold_ids,base_unit_prior):
    """Return (OOF delta, selected support, full-fit JSON state, verified folds, evidence).

    Caller binds saved R60 OOF base priors/folds to these canonical fit rows.
    Support is raw rows in the selected cell, or zero on fallback; not ESS/projects.
    """
    started=time.perf_counter();keys,eligible=_keys(Xfit);n=len(Xfit)
    indices=np.asarray(source_indices);saved=np.asarray(saved_fold_ids)
    need(n>0 and indices.shape==(n,) and indices.dtype.kind in 'iu' and (indices>=0).all()
         and len(np.unique(indices))==n,'Invalid canonical source indices')
    need(saved.shape==(n,) and saved.dtype.kind in 'iu' and ((saved>=0)&(saved<5)).all(),'Invalid saved fold IDs')
    folds,groups,fallback=_folds(Xfit,indices)
    need(np.array_equal(folds,saved),'Saved folds differ from original R60 project/hash rule')
    if isinstance(yfit,pd.Series):need(yfit.index.equals(Xfit.index),'Fit target Series index differs')
    y=np.asarray(yfit,dtype=np.float64)
    need(y.shape==(n,) and np.isfinite(y).all() and (y>0).all(),'Fit targets must be positive finite TWD')
    base_values=_base_prior(base_unit_prior,Xfit);area=base.effective_area(Xfit);dates=prior._dates(Xfit)
    with np.errstate(over='raise',invalid='raise',divide='raise'):units=y/area
    need(np.isfinite(units).all() and (units>0).all(),'Invalid fit unit prices')
    delta=np.zeros(n,dtype=np.float32);support=np.zeros(n,dtype=np.int64);records=[]
    for fold in range(5):
        held=folds==fold;reference=~held
        hg,rg=set(groups[held]),set(groups[reference])
        need(not hg.intersection(rg),'Project/fallback group crosses held/reference boundary')
        record={'fold':fold,'held_rows':int(held.sum()),'query_rows':int(held.sum()),
                'reference_rows':int(reference.sum()),'fit_rows':int(reference.sum()),
                'held_groups':len(hg),'reference_groups':len(rg),'held_reference_group_overlap':0,
                'held_indices_sha256':values_sha(indices[held],'<i8'),
                'reference_indices_sha256':values_sha(indices[reference],'<i8'),
                'reference_target_sha256':values_sha(y[reference]),'reference_unit_sha256':values_sha(units[reference])}
        if held.any():
            st=_build(keys,eligible,units,dates,reference,'complement_fit_for_held_fold')
            delta[held],support[held],diag=_lookup(st,keys[held],eligible[held],base_values[held])
            record.update(date_support=st['date_support'],cell_count=st['cell_count'],
                          eligible_reference_rows=st['eligible_reference_rows'],
                          minimum_observed_support=st['minimum_observed_support'],held_diagnostics=diag)
        else:record.update(date_support=None,cell_count=None,eligible_reference_rows=None,
                           minimum_observed_support=None,held_diagnostics=None)
        records.append(record)
    state=_build(keys,eligible,units,dates,np.ones(n,dtype=bool),'full_fit_for_query')
    source=_sources();state['source_sha256']=source
    used=list(dict.fromkeys(COLUMNS+list(prior.DATE_COLUMNS)+['project_name','address_raw','building_area','parking_area']))
    fit_hash=hashlib.sha256(pd.util.hash_pandas_object(Xfit[used],index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()
    evidence={'rules':copy.deepcopy(RULES),'fit_rows':n,'groups':len(set(groups)),'fallback_group_counts':fallback,
              'folds':records,'saved_fold_ids_exact':True,'whole_group_exclusion_verified':True,
              'training_role':'road and base priors use matching whole-group OOF folds',
              'query_role':'full-fit road state and saved full-fit base prior','chronological_oof':False,
              'uses_fit_labels':True,'uses_query_labels':False,'base_prior_refitted':False,
              'source_indices_sha256':values_sha(indices,'<i8'),'fold_ids_sha256':values_sha(folds,'<i8'),
              'group_keys_sha256':_hash_json(groups.tolist()),'fit_target_sha256':values_sha(y),
              'effective_area_sha256':values_sha(area),'fit_unit_sha256':values_sha(units),
              'fit_base_unit_prior_sha256':values_sha(base_values),'used_fit_columns':used,
              'used_fit_feature_values_sha256':fit_hash,'train_delta_sha256':values_sha(delta,'<f4'),
              'train_selected_support_sha256':values_sha(support,'<i8'),
              'train_selected_rows':int((support>0).sum()),'train_fallback_rows':int((support==0).sum()),
              'full_state_cell_count':state['cell_count'],'full_state_minimum_observed_support':state['minimum_observed_support'],
              'full_fit_date_support':state['date_support'],'state_content_sha256':_hash_json(state),
              'support_is_raw_rows_not_independent_projects_or_ESS':True,'source_sha256':source,
              'elapsed_seconds':time.perf_counter()-started}
    return delta,support,state,folds,evidence


def self_check():
    """One bounded CPU-only artificial check; no local housing data or models."""
    tick=time.perf_counter();n=300
    X=pd.DataFrame({'address_city':['A']*n,'address_district':['D']*n,'address_road':['Road']*n,
        'address_section':['__MISSING__']*n,'transaction_type':['house']*n,'building_type':['tower']*n,
        'project_name':['P'+str(i//10) for i in range(n)],'address_raw':['synthetic']*n,
        'building_area':np.full(n,100.),'parking_area':np.zeros(n),'trade_day_index':np.arange(n,dtype=float),
        'trade_date_invalid':np.zeros(n),'trade_date_out_of_range':np.zeros(n)},index=np.arange(1000,1000+n))
    X.loc[X.index[0],['trade_day_index','trade_date_invalid']]=[np.nan,1.]
    indices=np.arange(2000,2000+n,dtype=np.int64);folds,_,_=_folds(X,indices)
    assert len(np.unique(folds))==5
    y=1e6+np.arange(n)*1000.;base_values=np.full(n,12000.);before=X.copy(deep=True)
    delta,support,state,got,evidence=fit_oof_road_prior(X,y,indices,folds,base_values)
    held=folds==int(folds[0]);altered=y.copy();altered[held]*=100
    X2=X.copy();X2.loc[held,'trade_day_index']=1000.;X2.loc[held,'trade_date_invalid']=0.
    other=fit_oof_road_prior(X2,altered,indices,folds,base_values)
    assert delta[held].tobytes()==other[0][held].tobytes() and np.array_equal(support[held],other[1][held])
    assert all(r['held_reference_group_overlap']==0 for r in evidence['folds'])
    assert np.array_equal(got,folds) and (support>=30).all()
    restored=json.loads(json.dumps(state,allow_nan=False));order=[99,0,201]
    expected=predict_road_prior(state,X,base_values)
    actual=predict_road_prior(restored,X.iloc[order],base_values[order])
    assert actual[0].tobytes()==expected[0][order].tobytes() and np.array_equal(actual[1],expected[1][order])
    # Exactly 30 eligible raw rows qualifies; 29, unknown and missing mandatory keys fall back.
    T=X.iloc[:62].copy();T['address_road']=['yes']*30+['no']*29+['yes']*3
    T.loc[T.index[59],'address_city']='__MISSING__';T.loc[T.index[60],'address_district']='__MISSING__'
    T.loc[T.index[61],'address_road']='__MISSING__';T.loc[T.index[61],'trade_day_index']=730.
    # Row 0 is neutral; the latest valid row cannot form a road key but MUST enter normalization.
    k,e=_keys(T);threshold=_build(k,e,np.full(62,20000.),prior._dates(T),np.ones(62,dtype=bool),'full_fit_for_query')
    dates=T.trade_day_index.to_numpy();valid=np.isfinite(dates)
    raw=np.exp2(-(730.-dates[valid])/365.);denominator=float(raw.mean())
    expected_weights=np.ones(62);expected_weights[valid]=raw/denominator
    cell=threshold['cells'][k[0]]
    assert threshold['date_support']['max_valid_reference_trade_day_index']==730.
    assert threshold['date_support']['normalization_mean_valid_reference_only']==denominator
    assert cell['valid_date_rows']==29 and cell['neutral_date_rows']==1 and cell['rows']==30
    assert cell['weight_sum']==float(expected_weights[:30].sum())
    Q=T.iloc[[0,30,59,60,61,0]][COLUMNS].copy();Q.index=np.arange(6);Q.loc[5,'address_road']='unknown'
    d,s,diag=predict_road_prior(threshold,Q,np.full(6,10000.),True)
    assert np.array_equal(s,[30,0,0,0,0,0]) and d[0]==np.float32(np.log(20000.)-np.log(10000.)) and (d[1:]==0).all()
    assert threshold['date_support']['reference_rows']==62 and threshold['eligible_reference_rows']==59
    assert diag['reason_counts']=={'selected':1,'missing_required_key':3,'unsupported_key_or_below_30':2}
    try:fit_oof_road_prior(X,y,indices,(folds+1)%5,base_values)
    except ValueError:pass
    else:raise AssertionError('Mismatched saved folds accepted')
    pd.testing.assert_frame_equal(X,before);assert np.array_equal(y,1e6+np.arange(n)*1000.)
    assert np.array_equal(base_values,np.full(n,12000.)) and _hash_json(state)==evidence['state_content_sha256']
    return {'status':'passed','own_fold_labels_and_dates_do_not_change_held_feature':True,
            'whole_group_exclusion':True,'threshold_30_29_and_fallback':True,
            'whole_reference_date_population_and_neutral_weight':True,'query_needs_only_six_keys':True,
            'JSON_roundtrip_and_query_order':True,
            'saved_fold_mismatch_rejected':True,'inputs_unchanged':True,
            'dtype_float32_support_int64':delta.dtype==np.float32 and support.dtype==np.int64,
            'real_data_read':False,'model_fits':0,'elapsed_seconds':time.perf_counter()-tick}


if __name__=='__main__':print(json.dumps(self_check(),ensure_ascii=False))

