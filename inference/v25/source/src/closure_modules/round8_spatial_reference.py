"""One prespecified R8 spatial residual supplement to the saved v3 prediction.

Pure input APIs: no data/model file reads, query labels, training or v2 refits.
Geometry denotes coarse support, not exact house position or an error bound.
"""
from __future__ import annotations
import copy,hashlib,json,pickle
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
from round2_neighborhood import matrices, PARAMETERS as V2_PARAMETERS
from round6_geo import check_geo, frame_sha, FEATURES as GEO_COLUMNS

MISSING='__MISSING__'
EARTH_RADIUS_M=6371008.8
GROUP_COLUMNS=('address_city','building_type','main_use')
WEAK_SOURCES=('district_style','base_fallback')
PARAMETERS={
    'format_version':1,'radius_m':2000.0,'maximum_extent_m_each':2000.0,
    'maximum_projects':40,'minimum_projects':5,'same_nonmissing_group':list(GROUP_COLUMNS),
    'fit_project_name_required':True,'project_key':'normalized original city|address_district|project_name',
    'exclude_query_project':True,'fit_area_bounds_m2':[15.0,1500.0],
    'query_area_bounds_m2':[15.0,1500.0],'fit_recent_years':4.0,'days_per_year':365.25,
    'fit_reference_residual_required':'finite existing v2 reference_residual only',
    'fit_dates':'finite and anchor - 4*365.25 <= trade_day_index <= anchor',
    'project_vote':'median of all eligible actual fit rows of this project inside query radius',
    'project_order':'minimum actual fit-position distance, then project key lexicographically',
    'spatial_center':'unweighted median of at most 40 project medians',
    'spatial_MAD':'unweighted median absolute deviation of project medians from center',
    'maximum_weight':0.30,'MAD_scale':0.18,'eligible_v2_sources':list(WEAK_SOURCES),
    'geographic_price_ratio_bounds':[0.65,1.5],'future_day_cap':366.0,
    'G':'area * exp(center + query_hedonic @ saved_v2_beta), then clip to [0.65,1.5] * Bfuture',
    'Bfuture':'raw_v3_blend * exp(saved_v2_beta[3] * clipped_future_days / 365.25)',
    'adjustment':'p8 = p3 + (1-c) * 0.30 * exp(-MAD/0.18) * (G - Bfuture)',
    'v2_refitted':False,'additional_trend_applied_to_G':False,'Dq_plus_Dr_distance_hard_gate':False,
    'geometry_semantics':'road representative or planar intersection; extent is support, not house error bound',
    'location_compression':'same group, exact location, project: keep every row residual and source row position',
    'query_cache':'same group, exact query location, query project; no query fitted state',
    'cpu_threads':1,
}


def parameters():
    return copy.deepcopy(PARAMETERS)


def need(ok,message):
    if not ok:raise ValueError(message)


def _numeric_sha(values):
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()


def _norm(series):
    values=series.fillna(MISSING).astype(str).str.strip()
    return values.mask(values.isin(['','nan','None','<NA>','NaT']),MISSING).to_numpy()


def _project_keys(X):
    city=_norm(X.address_city);district=_norm(X.address_district);project=_norm(X.project_name)
    return np.asarray([a+'|'+b+'|'+c for a,b,c in zip(city,district,project)],dtype=object),project!=MISSING


def _group_keys(X):
    values=[_norm(X[c]) for c in GROUP_COLUMNS]
    return list(zip(*values)),np.logical_and.reduce([v!=MISSING for v in values])


def _xyz(latitude,longitude):
    lat=np.radians(np.asarray(latitude,dtype='float64'));lon=np.radians(np.asarray(longitude,dtype='float64'))
    return EARTH_RADIUS_M*np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])


def _schema(X,geo):
    need(isinstance(X,pd.DataFrame) and isinstance(geo,pd.DataFrame),'Expected X and geo DataFrames')
    need(len(X.columns)==87 and X.columns.is_unique and '編號' not in X and '總價元' not in X and 'y' not in X,'Expected original target-free 87-column X')
    need(len(X)==len(geo) and X.index.equals(geo.index),'X and geographic row order/index differ')
    check_geo(geo)
    need(list(geo.columns)==list(GEO_COLUMNS),'Geographic schema differs')
    needed={'address_city','address_district','project_name','building_type','main_use','building_area',
        'parking_fraction','floor_mean','total_floors','rooms','land_building_ratio','trade_day_index','parking_count'}
    need(needed<=set(X.columns),'Required original features absent')
    cats=[c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    need(len(cats)==21,'Expected original 21 categorical features')
    usable=geo.geo_has_position.eq(1)
    need(geo.loc[usable,'geo_latitude'].between(20,28).all() and geo.loc[usable,'geo_longitude'].between(116,125).all(),'Geographic coordinates outside study bounds')
    need(geo.loc[usable,'geo_log_extent'].ge(0).all(),'Negative geometry extent')
    return {'feature_columns':list(X.columns),'feature_dtypes':{c:str(X[c].dtype) for c in X},
            'geo_columns':list(geo.columns),'geo_dtypes':{c:str(geo[c].dtype) for c in geo},'categorical_columns':cats}


def _eligible_geo(X,geo):
    groups,known=_group_keys(X)
    with np.errstate(over='ignore',invalid='ignore'):
        extent=np.expm1(geo.geo_log_extent.to_numpy(dtype='float64'))
    return geo.geo_has_position.eq(1).to_numpy()&np.isfinite(extent)&(extent>=0)&(extent<=2000)&known,groups


def build(reference,Xfit,geo):
    """Reuse fit-only v2 residuals; return pickleable dict and JSON state summary."""
    schema=_schema(Xfit,geo)
    need(reference.get('format_version')==1 and reference.get('parameters')==V2_PARAMETERS,'Unexpected v2 reference contract')
    need(reference['fit_report']['fitting_rows']==len(Xfit) and len(Xfit)>0,'Reference and fit rows differ')
    residual=np.asarray(reference['reference_residual'],dtype='float64')
    beta=np.asarray(reference['beta'],dtype='float64').copy();fill=np.asarray(reference['fill_values'],dtype='float64').copy()
    anchor=float(reference['fit_report']['hedonic']['maximum_fit_day_index'])
    need(residual.shape==(len(Xfit),) and beta.shape==(4,) and fill.shape==(8,),'Reference array schema differs')
    need(np.isfinite(beta).all() and np.isfinite(fill).all() and np.isfinite(anchor),'Nonfinite saved v2 normalization')
    dates=Xfit.trade_day_index.to_numpy(dtype='float64');area=Xfit.building_area.to_numpy(dtype='float64')
    need(np.isfinite(dates).any() and float(np.nanmax(dates))==anchor,'Reference date anchor differs from fit X')
    geometric,groups=_eligible_geo(Xfit,geo);projects,named=_project_keys(Xfit)
    finite_residual=np.isfinite(residual)
    area_ok=np.isfinite(area)&(area>=15)&(area<=1500)
    date_ok=np.isfinite(dates)&(dates>=anchor-4*365.25)&(dates<=anchor)
    selected=geometric&named&finite_residual&area_ok&date_ok
    latitude=geo.geo_latitude.to_numpy(dtype='float64');longitude=geo.geo_longitude.to_numpy(dtype='float64')
    # Compression keeps full row multiplicity; never take a median of location medians.
    locations={}
    for row in np.flatnonzero(selected):
        key=(groups[row],float(latitude[row]),float(longitude[row]),str(projects[row]))
        locations.setdefault(key,[]).append(int(row))
    grouped={}
    for (group,lat,lon,project),rows in locations.items():
        grouped.setdefault(group,[]).append((lat,lon,project,np.asarray(rows,dtype='int64')))
    model_groups={}
    for group,items in grouped.items():
        coords=np.asarray([[v[0],v[1]] for v in items],dtype='float64')
        rows=[v[3].copy() for v in items]
        model_groups[group]={'xyz':_xyz(coords[:,0],coords[:,1]),'latitude_longitude':coords,
            'projects':[v[2] for v in items],'row_positions':rows,
            'residual_arrays':[residual[v].copy() for v in rows]}
    summary={'format_version':1,'parameters':parameters(),'schema':schema,'fit_rows':len(Xfit),
        'reference_rows_selected':int(selected.sum()),'position_project_groups':len(locations),
        'unique_projects':int(len(set(projects[selected]))),'city_style_use_groups':len(model_groups),
        'fit_X_sha256':frame_sha(Xfit),'fit_geo_sha256':frame_sha(geo),
        'existing_reference_residual_sha256':_numeric_sha(residual),'saved_beta_sha256':_numeric_sha(beta),
        'saved_fill_values_sha256':_numeric_sha(fill),'saved_beta':beta.tolist(),'saved_fill_values':fill.tolist(),
        'anchor_day_index':anchor,'oldest_eligible_day_index':anchor-4*365.25,
        'selected_fit_row_positions_sha256':hashlib.sha256(np.flatnonzero(selected).astype('<i8').tobytes()).hexdigest(),
        'individual_filter_pass_counts':{'geography_and_group':int(geometric.sum()),'named_project':int(named.sum()),
            'finite_existing_residual':int(finite_residual.sum()),'area_15_to_1500':int(area_ok.sum()),'finite_date_recent_4_years':int(date_ok.sum())},
        'filters_are_intersected_not_summed':True,'query_used_in_build':False,'new_reference_fit_performed':False,
        'input_frames_mutated':False,'raw_targets_read_by_helper':False}
    summary['runtime']={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    summary['state_content_sha256']=hashlib.sha256(json.dumps(summary,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
    model={'format_version':1,'parameters':parameters(),'schema':schema,'beta':beta,'fill_values':fill,
           'anchor_day_index':anchor,'groups':model_groups,'state_summary':copy.deepcopy(summary)}
    return model,summary


def _neighborhood(group,tree,point,query_project):
    chord=2*EARTH_RADIUS_M*np.sin(2000/(2*EARTH_RADIUS_M))
    ids=np.asarray(tree.query_ball_point(point,chord,workers=1),dtype='int64')
    if len(ids)==0:return {'projects':(), 'count':0,'available_count':0,'center':np.nan,'MAD':np.nan,'nearest_distance_m':np.nan,'farthest_distance_m':np.nan,'reference_rows':0}
    distances=2*EARTH_RADIUS_M*np.arcsin(np.clip(np.linalg.norm(group['xyz'][ids]-point,axis=1)/(2*EARTH_RADIUS_M),0,1))
    per_project={}
    for position,distance in zip(ids,distances):
        project=group['projects'][int(position)]
        if distance>2000 or project==query_project:continue
        entry=per_project.setdefault(project,{'distance':float(distance),'arrays':[]})
        entry['distance']=min(entry['distance'],float(distance))
        entry['arrays'].append(group['residual_arrays'][int(position)])
    keys=sorted(per_project,key=lambda key:(per_project[key]['distance'],key))[:40]
    medians=[];rows=0
    for key in keys:
        values=np.concatenate(per_project[key]['arrays'])
        rows+=len(values);medians.append(float(np.median(values)))
    center=float(np.median(medians)) if medians else np.nan
    mad=float(np.median(np.abs(np.asarray(medians)-center))) if medians else np.nan
    return {'projects':tuple(keys),'count':len(keys),'available_count':len(per_project),'center':center,'MAD':mad,
        'nearest_distance_m':min((per_project[k]['distance'] for k in keys),default=np.nan),
        'farthest_distance_m':max((per_project[k]['distance'] for k in keys),default=np.nan),'reference_rows':rows}


def predict(model,Xquery,geoq,raw_base,v3_prediction,v2_diagnostics):
    """Supplement weak v2 sources only. No labels, files, refit or additional trend."""
    need(model.get('format_version')==1 and model.get('parameters')==PARAMETERS,'Spatial model contract differs')
    need(_schema(Xquery,geoq)==model['schema'],'Query schema differs from fit')
    n=len(Xquery)
    baseline=np.asarray(raw_base,dtype='float64');original=np.asarray(v3_prediction,dtype='float64')
    need(baseline.shape==original.shape==(n,) and np.isfinite(baseline).all() and np.isfinite(original).all() and (baseline>=0).all() and (original>=0).all(),'Invalid aligned base or v3 predictions')
    if isinstance(v2_diagnostics,pd.DataFrame):
        need(v2_diagnostics.index.equals(Xquery.index),'v2 diagnostics and query indices differ')
    sources=np.asarray(v2_diagnostics['source'],dtype=object);confidence=np.asarray(v2_diagnostics['confidence'],dtype='float64')
    need(sources.shape==confidence.shape==(n,) and np.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all(),'Invalid v2 diagnostics')
    beta=np.asarray(model['beta'],dtype='float64');fill=np.asarray(model['fill_values'],dtype='float64');anchor=float(model['anchor_day_index'])
    need(beta.shape==(4,) and fill.shape==(8,) and np.isfinite(beta).all() and np.isfinite(fill).all() and np.isfinite(anchor),'Invalid saved normalization')
    geometric,groups=_eligible_geo(Xquery,geoq);projects,_=_project_keys(Xquery)
    area=Xquery.building_area.to_numpy(dtype='float64');dates=Xquery.trade_day_index.to_numpy(dtype='float64')
    area_ok=np.isfinite(area)&(area>=15)&(area<=1500);date_ok=np.isfinite(dates)
    weak=np.isin(sources,WEAK_SOURCES);eligible=weak&area_ok&date_ok&geometric
    _,hedonic,_=matrices(Xquery,fill)
    future_days=np.minimum(np.maximum(np.nan_to_num(dates-anchor,nan=0.0),0.0),366.0)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        future_base=baseline*np.exp(beta[3]*future_days/365.25)
    valid_base_future=np.isfinite(future_base)&(future_base>0)
    eligible=eligible&valid_base_future
    output=original.copy();applied=np.zeros(n,dtype=bool)
    reason=np.full(n,'not_weak_source',dtype=object);reason[weak]='area_out_of_bounds'
    reason[weak&area_ok]='invalid_date';reason[weak&area_ok&date_ok]='invalid_geography_or_group'
    reason[eligible]='fewer_than_5_projects'
    reason[weak&~valid_base_future]='invalid_base_future'
    count=np.zeros(n,dtype='int64');available=count.copy();fit_rows=count.copy()
    center=np.full(n,np.nan);mad=center.copy();nearest=center.copy();farthest=center.copy();G=center.copy()
    alpha=np.zeros(n,dtype='float64');effective=alpha.copy()
    lat=geoq.geo_latitude.to_numpy(dtype='float64');lon=geoq.geo_longitude.to_numpy(dtype='float64')
    points=_xyz(lat,lon);trees={};cache={}
    for row in np.flatnonzero(eligible):
        group_key=groups[row];key=(group_key,float(lat[row]),float(lon[row]),str(projects[row]))
        if key not in cache:
            group=model['groups'].get(group_key)
            if group is None:
                answer={'projects':(),'count':0,'available_count':0,'center':np.nan,'MAD':np.nan,'nearest_distance_m':np.nan,'farthest_distance_m':np.nan,'reference_rows':0}
            else:
                if group_key not in trees:trees[group_key]=cKDTree(group['xyz'])
                answer=_neighborhood(group,trees[group_key],points[row],str(projects[row]))
            cache[key]=answer
        answer=cache[key];count[row]=answer['count'];available[row]=answer['available_count'];fit_rows[row]=answer['reference_rows']
        center[row]=answer['center'];mad[row]=answer['MAD'];nearest[row]=answer['nearest_distance_m'];farthest[row]=answer['farthest_distance_m']
        if count[row]<5:continue
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            log_ratio=np.log(area[row])+center[row]+float(hedonic[row]@beta)-np.log(future_base[row])
            G[row]=future_base[row]*np.exp(np.clip(log_ratio,np.log(0.65),np.log(1.5)))
        if not np.isfinite(G[row]):
            reason[row]='invalid_geographic_price';continue
        alpha[row]=0.30*np.exp(-mad[row]/0.18);effective[row]=(1-confidence[row])*alpha[row]
        output[row]=original[row]+effective[row]*(G[row]-future_base[row])
        applied[row]=True;reason[row]='spatial_residual'
    need(np.isfinite(output).all() and (output>=0).all(),'Invalid spatially supplemented prices')
    need(np.array_equal(output[~applied],original[~applied]),'Fallback rows changed')
    diagnostic=pd.DataFrame({'v2_source':sources.copy(),'v2_confidence':confidence.copy(),'eligible':eligible,
        'applied':applied,'reason':reason,'project_count':count,'available_project_count_before_cap':available,
        'reference_row_count':fit_rows,'residual_center':center,'project_MAD':mad,'nearest_distance_m':nearest,
        'farthest_distance_m':farthest,'spatial_weight':alpha,'effective_remaining_weight':effective,
        'geographic_price':G,'base_future':future_base,'price_adjustment':output-original},index=Xquery.index.copy())
    report={'query_rows':n,'weak_source_rows':int(weak.sum()),'eligible_query_rows':int(eligible.sum()),
        'applied_rows':int(applied.sum()),'changed_rows':int(np.count_nonzero(output!=original)),
        'fallback_rows':int((~applied).sum()),'fallback_bitwise_values_preserved':True,
        'reason_counts':{str(k):int(v) for k,v in pd.Series(reason).value_counts().items()},
        'applied_source_counts':{str(k):int(v) for k,v in pd.Series(sources[applied]).value_counts().items()},
        'neighborhood_cache_entries':len(cache),'neighborhood_cache_reuses':int(eligible.sum())-len(cache),
        'maximum_projects_used':int(count.max()) if n else 0,'query_labels_used':False,
        'reference_refitted':False,'additional_trend_applications_to_G':0,'parameters':parameters(),
        'geometry_is_exact_house_position':False,'distance_plus_extents_used_as_gate':False}
    return output,diagnostic,report

def self_check():
    """Return passed synthetic check names; no files, real data or model fits."""
    from house_features import build_features
    from round6_geo import encode_geo_features
    import inspect
    checks=[]
    def passed(name,condition):
        need(condition,'Synthetic check failed: '+name);checks.append(name)
    def features(projects):
        n=len(projects)
        raw=pd.DataFrame({'鄉鎮市區':['大安區']*n,'土地位置建物門牌':['臺北市大安區合成路1號']*n,
            '建案名稱':projects,'建物型態':['住宅大樓']*n,'主要用途':['住家用']*n,
            '建物移轉總面積平方公尺':[100.]*n,'土地移轉總面積平方公尺':[20.]*n,
            '車位移轉總面積平方公尺':[0.]*n,'交易筆棟數':['土地1建物1車位0']*n,
            '建物現況格局-房':[2]*n,'建物現況格局-廳':[1]*n,'建物現況格局-衛':[1]*n,
            '移轉層次':['五層']*n,'總樓層數':[10]*n,'交易年月日':['1130101']*n})
        X=build_features(raw);X['trade_day_index']=np.full(n,9000.,dtype='float32')
        return X
    def geography(longitudes,extent=0.0):
        n=len(longitudes)
        return encode_geo_features(pd.DataFrame({'geo_latitude':[25.]*n,'geo_longitude':longitudes,
            'geo_road_length_m':[100.]*n,'geo_extent_diagonal_m':[extent]*n,
            'geo_nearby_fragment_grouping':[0]*n,'geo_match_status':['road_level']*n}))
    def reference(X,offsets,beta_time=.07):
        anchor=float(np.nanmax(X.trade_day_index.to_numpy(dtype='float64')))
        beta=np.asarray([0.,0.,0.,beta_time]);fill=np.zeros(8,dtype='float64')
        residual=np.log(10000.)+np.asarray(offsets,dtype='float64')-beta_time*(anchor-8000)/365.25
        return {'format_version':1,'parameters':copy.deepcopy(V2_PARAMETERS),'reference_residual':residual,
            'beta':beta,'fill_values':fill,'fit_report':{'fitting_rows':len(X),'hedonic':{'maximum_fit_day_index':anchor}}}
    def query_inputs(project='Q',days=365.25,extent=0.0):
        X=features([project]);X['trade_day_index']=np.asarray([9000.+days],dtype='float32')
        geo=geography([121.],extent)
        diagnostics=pd.DataFrame({'source':['base_fallback'],'confidence':[0.]},index=X.index)
        raw=np.asarray([1e6]);p3=raw*np.exp(.07*min(max(days,0),366)/365.25)
        return X,geo,raw,p3,diagnostics
    projects=['A']*20+['A']+['A']*30+['B','C','D','E']
    fit=features(projects)
    fit_geo=geography([121.]*20+[121.001]+[121.1]*30+[121.]*4)
    offsets=[.4]*20+[-.4]+[-2.]*30+[0.,.1,.2,.3]
    old=reference(fit,offsets)
    fit_before,geo_before=frame_sha(fit),frame_sha(fit_geo);reference_before=pickle.dumps(old,protocol=5)
    model,state=build(old,fit,fit_geo)
    passed('build_no_input_or_reference_mutation',frame_sha(fit)==fit_before and frame_sha(fit_geo)==geo_before and pickle.dumps(old,protocol=5)==reference_before)
    passed('pure_build_and_predict_APIs_have_no_query_y',list(inspect.signature(build).parameters)==['reference','Xfit','geo'] and list(inspect.signature(predict).parameters)==['model','Xquery','geoq','raw_base','v3_prediction','v2_diagnostics'])
    X,g,raw,p3,d=query_inputs();model_before=pickle.dumps(model,protocol=5)
    query_before,query_geo_before=frame_sha(X),frame_sha(g);d_before=d.copy(deep=True)
    prediction,diag,report=predict(model,X,g,raw,p3,d)
    expected_center=np.log(10000.)+.2-.07*(9000-8000)/365.25
    expected_mad=.1;expected_G=1e6*np.exp(.2+.07);expected_alpha=.30*np.exp(-expected_mad/.18)
    expected=p3[0]+expected_alpha*(expected_G-p3[0])
    passed('all_actual_in_radius_rows_not_location_or_global_project_medians',np.isclose(diag.residual_center.iloc[0],expected_center,rtol=0,atol=1e-12) and diag.reference_row_count.iloc[0]==25)
    passed('each_project_equal_vote_and_project_MAD',diag.project_count.iloc[0]==5 and np.isclose(diag.project_MAD.iloc[0],expected_mad,atol=1e-12))
    passed('only_one_future_trend_and_exact_fixed_adjustment',np.isclose(prediction[0],expected,rtol=1e-12) and np.isclose(diag.geographic_price.iloc[0],expected_G,rtol=1e-12))
    passed('query_does_not_mutate_model_frames_or_diagnostics',pickle.dumps(model,protocol=5)==model_before and frame_sha(X)==query_before and frame_sha(g)==query_geo_before and d.equals(d_before))
    reloaded=pickle.loads(pickle.dumps(model,protocol=5))
    replay,replay_diag,replay_report=predict(reloaded,X,g,raw,p3,d)
    passed('plain_pickle_roundtrip_and_deterministic_report',np.array_equal(replay,prediction) and replay_diag.equals(diag) and replay_report==report)
    passed('JSON_serializable_state_summary',json.loads(json.dumps(state,ensure_ascii=False,allow_nan=False))==state)
    Xs,gs,rs,ps,ds=query_inputs('A')
    fallback,fd,_=predict(model,Xs,gs,rs,ps,ds)
    passed('exclude_query_project_and_require_five_distinct_projects',fd.project_count.iloc[0]==4 and not fd.applied.iloc[0] and np.array_equal(fallback,ps))
    fmissing=fit.copy(deep=True);fmissing.loc[fmissing.project_name=='E','project_name']=MISSING
    m,_=build(old,fmissing,fit_geo);fallback,fd,_=predict(m,X,g,raw,p3,d)
    passed('missing_fit_name_never_counts_as_an_extra_project',fd.project_count.iloc[0]==4 and np.array_equal(fallback,p3))
    invalid_queries=pd.concat([X]*7,ignore_index=True);invalid_geo=pd.concat([g]*7,ignore_index=True)
    invalid_d=pd.DataFrame({'source':['project']+['base_fallback']*6,'confidence':[0.]*7})
    invalid_queries.loc[1,'building_area']=np.float32(14.)
    invalid_queries.loc[2,'trade_day_index']=np.float32(np.nan)
    invalid_geo.loc[3,'geo_log_extent']=np.float32(np.log1p(2001.))
    invalid_queries.loc[4,'address_city']='臺中市'
    invalid_queries.loc[5,'building_type']='辦公商業大樓'
    invalid_queries.loc[6,'main_use']='商業用'
    fallback,fd,_=predict(model,invalid_queries,invalid_geo,np.repeat(raw,7),np.repeat(p3,7),invalid_d)
    passed('strong_source_bad_area_bad_date_bad_geo_and_other_groups_fallback_exactly',not fd.applied.any() and np.array_equal(fallback,np.repeat(p3,7)))
    f_old=fit.copy(deep=True);f_old.loc[f_old.project_name=='E','trade_day_index']=np.float32(9000-4*365.25-1)
    m,_=build(old,f_old,fit_geo);fallback,fd,_=predict(m,X,g,raw,p3,d)
    passed('finite_recent_four_year_fit_date_filter',fd.project_count.iloc[0]==4 and np.array_equal(fallback,p3))
    bad_res=copy.deepcopy(old);bad_res['reference_residual'][np.asarray(projects)=='E']=np.nan
    m,_=build(bad_res,fit,fit_geo);fallback,fd,_=predict(m,X,g,raw,p3,d)
    passed('reuse_only_finite_existing_reference_residuals',fd.project_count.iloc[0]==4 and np.array_equal(fallback,p3))
    f_area=fit.copy(deep=True);f_area.loc[f_area.project_name=='E','building_area']=np.float32(1501.)
    m,_=build(old,f_area,fit_geo);fallback,fd,_=predict(m,X,g,raw,p3,d)
    passed('fit_area_bounds_enforced_without_new_price_filter',fd.project_count.iloc[0]==4 and np.array_equal(fallback,p3))
    f_ext=fit_geo.copy(deep=True);f_ext.loc[np.asarray(projects)=='E','geo_log_extent']=np.float32(np.log1p(2001.))
    m,_=build(old,fit,f_ext);fallback,fd,_=predict(m,X,g,raw,p3,d)
    passed('fit_extent_condition_enforced',fd.project_count.iloc[0]==4 and np.array_equal(fallback,p3))
    cross_district=X.copy(deep=True);cross_district.loc[:,'address_district']='中山區'
    _,cd,_=predict(model,cross_district,g,raw,p3,d)
    passed('same_city_can_cross_district',cd.applied.iloc[0])
    named_query=X.copy(deep=True);named_query.loc[:,'project_name']=MISSING
    _,nd,_=predict(model,named_query,g,raw,p3,d)
    passed('missing_query_project_still_can_use_named_fit_projects',nd.applied.iloc[0] and nd.project_count.iloc[0]==5)
    efit=geography([121.]*20+[121.001]+[121.1]*30+[121.]*4,1900.)
    m,_=build(old,fit,efit);eg=geography([121.],1900.)
    _,ed,_=predict(m,X,eg,raw,p3,d)
    passed('no_unapproved_distance_plus_extent_hard_gate',ed.applied.iloc[0])
    confidence=.12;cd=d.copy();cd['source']='district_style';cd['confidence']=confidence
    p3c=(1-confidence)*p3+confidence*raw*1.1
    pc,dc,_=predict(model,X,g,raw,p3c,cd)
    passed('existing_reference_weight_is_preserved',np.isclose(pc[0],p3c[0]+(1-confidence)*expected_alpha*(expected_G-p3[0]),rtol=1e-12))
    X2=pd.concat([X,X],ignore_index=True);X2.loc[1,'trade_day_index']=np.float32(9000+730.5)
    g2=pd.concat([g,g],ignore_index=True);d2=pd.concat([d,d],ignore_index=True)
    X2.index=g2.index=d2.index=pd.Index([100,101])
    raw2=np.repeat(raw,2);p32=raw2*np.exp(.07*np.array([365.25,366])/365.25)
    pfull,dfull,rfull=predict(model,X2,g2,raw2,p32,d2)
    pslice,dslice,_=predict(model,X2.iloc[1:],g2.iloc[1:],raw2[1:],p32[1:],d2.iloc[1:])
    passed('sliced_nonzero_indices_and_cached_neighborhood_preserve_row_math',np.array_equal(pfull[1:],pslice) and dslice.index.equals(pd.Index([101])) and rfull['neighborhood_cache_entries']==1 and rfull['neighborhood_cache_reuses']==1)
    passed('future_base_capped_at_366_days',np.isclose(dfull.base_future.loc[101],raw[0]*np.exp(.07*366/365.25),rtol=1e-12))
    for label,offset,bound in [('upper',3.,1.5),('lower',-3.,.65)]:
        rr=reference(fit,[offset]*len(fit));m,_=build(rr,fit,fit_geo)
        _,dd,_=predict(m,X,g,raw,p3,d)
        passed('geographic_price_'+label+'_ratio_clip',np.isclose(dd.geographic_price.iloc[0],bound*p3[0],rtol=1e-12))
    zero,zd,_=predict(model,X,g,np.zeros(1),p3,d)
    passed('zero_base_future_falls_back_exactly',np.array_equal(zero,p3) and not zd.applied.iloc[0] and zd.reason.iloc[0]=='invalid_base_future')
    huge_base,hd,_=predict(model,X,g,np.asarray([np.finfo('float64').max]),p3,d)
    passed('nonfinite_base_future_falls_back_exactly',np.array_equal(huge_base,p3) and not hd.applied.iloc[0] and hd.reason.iloc[0]=='invalid_base_future')
    huge_ref=reference(fit,[1000.]*len(fit));huge_model,_=build(huge_ref,fit,fit_geo)
    huge_pred,huge_diag,_=predict(huge_model,X,g,raw,p3,d)
    passed('finite_huge_log_price_clips_without_exponential_overflow',np.isfinite(huge_pred).all() and huge_diag.applied.iloc[0] and np.isclose(huge_diag.geographic_price.iloc[0],1.5*p3[0],rtol=1e-12))
    tie_names=[f'P{i:02d}' for i in range(44,-1,-1)]
    tf=features(tie_names);tg=geography([121.]*45);tm,_=build(reference(tf,[0.]*45),tf,tg)
    group=next(iter(tm['groups'].values()));answer=_neighborhood(group,cKDTree(group['xyz']),_xyz([25.],[121.])[0],'not_in_fit')
    expected_keys=sorted(group['projects'])[:40]
    passed('maximum_40_projects_and_lexical_project_ties',answer['available_count']==45 and list(answer['projects'])==expected_keys)
    removed=_neighborhood(group,cKDTree(group['xyz']),_xyz([25.],[121.])[0],expected_keys[0])
    passed('query_project_exclusion_precedes_nearest_40_limit',removed['count']==40 and expected_keys[0] not in removed['projects'] and sorted(group['projects'])[40] in removed['projects'])
    schema_failed=False
    try:predict(model,X,g[g.columns[::-1]],raw,p3,d)
    except ValueError:schema_failed=True
    passed('original_geography_schema_rejected_if_reordered',schema_failed)
    return checks


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps({'status':'passed','checks':self_check(),'real_data_read':False,'models_fitted_or_scored':False},ensure_ascii=True),flush=True)