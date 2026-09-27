"""Fixed R8 geography feasibility audit. X/indices only; no targets or predictions.

Reads canonical alignment indices instead of reproducing target-containing dedup.
No model fitting, scoring, external downloads or shared-cache writes.
"""
from __future__ import annotations
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round8/audit/location_feasibility.json'
REPORT=ROOT/'reports/round8_location_feasibility.md'
MISSING='__MISSING__'
EARTH_RADIUS_M=6371008.8
RULES={'radius_m':2000.0,'maximum_unique_projects':40,'maximum_extent_m':2000.0,
       'same_group':['address_city','building_type','main_use'],
       'all_group_values_required':True,'fit_named_project_required':True,
       'project_key':'original normalized city|address_district|project_name',
       'candidate_position_dedup':['city','building_type','main_use','latitude','longitude','project_key'],
       'candidate_position_representative':'first actual canonical fit row in original sorted index order',
       'project_representative_per_query':'nearest actual eligible candidate row; canonical fit row breaks distance ties',
       'query_cache_key':'city, building_type, main_use, latitude, longitude',
       'max40_order':'deduplicate fit projects by nearest row, then take closest 40; diagnostics never search outside those 40',
       'distance':'spherical great-circle metres; KD-tree chord radius is exactly equivalent to 2000m',
       'cross_road_definition':'both original address_road values nonmissing and original location_road keys differ',
       'same_project_exclusion':'reported separately; not used to expand or refill the fixed nearest-40 pool',
       'scope':'research availability only; not model schema, confidence calibration or acceptance threshold',
       'additional_diagnostic':'D_query + D_reference + centre_distance <= 2000m within unchanged nearest-40 pool; D is geometry extent, not proven home-position error',
       'threads':1}
LEVELS=('project','address','street_section','locality','street','district_style')
MINIMUMS={'project':6,'address':8,'street_section':20,'locality':20,'street':20,'district_style':80}

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read_locked(path):
    need(path.with_suffix('.sha256').read_text(encoding='ascii').strip()==sha(path),'Receipt differs: '+str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))

def norm(series):
    value=series.fillna(MISSING).astype(str).str.strip().reset_index(drop=True)
    return value.mask(value.isin(['','nan','None','<NA>','NaT']),MISSING)

def project_keys(X):
    p=norm(X.project_name)
    return norm(X.address_city)+'|'+norm(X.address_district)+'|'+p,p.eq(MISSING).to_numpy()

def xyz(latitude,longitude):
    la=np.radians(np.asarray(latitude,dtype='float64'))
    lo=np.radians(np.asarray(longitude,dtype='float64'))
    return EARTH_RADIUS_M*np.column_stack((np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)))

def geo_mask(X,geo):
    extent=np.expm1(geo.geo_log_extent.to_numpy(dtype='float64'))
    lat=geo.geo_latitude.to_numpy(dtype='float64');lon=geo.geo_longitude.to_numpy(dtype='float64')
    position=geo.geo_has_position.to_numpy()==1
    finite=np.isfinite(lat)&np.isfinite(lon)&np.isfinite(extent)&(extent>=0)
    group=np.ones(len(X),dtype=bool)
    for c in RULES['same_group']:group &= norm(X[c]).ne(MISSING).to_numpy()
    eligible=position&finite&(extent<=RULES['maximum_extent_m'])&group
    return eligible,{'position_rows':int(position.sum()),'position_with_extent_at_most_2000_rows':int((position&finite&(extent<=2000)).sum()),'missing_group_rows':int((~group).sum()),'eligible_rows':int(eligible.sum())}

def reference_keys(X):
    # Same X-only expressions as frozen round2_neighborhood.group_keys.
    region=X.address_city.astype(str)+'|'+X.address_district.astype(str)
    prefix=region+'|'+X.building_type.astype(str)+'|'+X.main_use.astype(str)+'|'
    project=X.project_name.astype(str);road=X.address_road.astype(str)
    locality=X.address_locality.astype(str);address=X.address_raw.astype(str)
    return {'project':(prefix+project).where(project!=MISSING),
            'address':(prefix+address).where(address!=MISSING),
            'street_section':(prefix+road+'|'+X.address_section.astype(str)).where(road!=MISSING),
            'locality':(prefix+locality).where(locality!=MISSING),
            'street':(prefix+road).where(road!=MISSING),'district_style':region+'|'+X.building_type.astype(str)+'|'+X.main_use.astype(str)}

def reference_upper_bound(fit,query):
    area=fit.building_area.to_numpy(dtype='float64')
    valid=np.isfinite(area)&(area>=15)&(area<=1500)
    fk,qk=reference_keys(fit),reference_keys(query)
    selected=np.full(len(query),'no_X_supported_level',dtype=object)
    support={}
    for level in LEVELS:
        counts=fk[level].loc[valid].value_counts(dropna=True)
        values=qk[level].map(counts).fillna(0).to_numpy(dtype='int64')
        support[level]=values
        chosen=(selected=='no_X_supported_level')&(values>=MINIMUMS[level])
        selected[chosen]=level
    return selected,support,{'fit_area_eligible_rows':int(valid.sum()),'minimum_fit_rows_by_level':MINIMUMS,
        'fit_area_bounds_m2':[15,1500],'target_quantile_filter_applied':False,
        'local_price_dispersion_or_confidence_computed':False,'actual_v2_reference_assignments_loaded':False,
        'meaning':'optimistic X-only upper bound before v2 fit-only price-quantile filtering; not actual reference selection'}

def nearest_projects(tree,positions,point,projects,fit_indices):
    chord=2*EARTH_RADIUS_M*np.sin(RULES['radius_m']/(2*EARTH_RADIUS_M))
    local=np.asarray(tree.query_ball_point(point,chord,workers=1),dtype='int64')
    if len(local)==0:return np.empty(0,dtype='int64'),np.empty(0),0
    distances=2*EARTH_RADIUS_M*np.arcsin(np.clip(np.linalg.norm(tree.data[local]-point,axis=1)/(2*EARTH_RADIUS_M),0,1))
    keep=distances<=RULES['radius_m'];local=local[keep];distances=distances[keep]
    global_rows=positions[local]
    order=np.lexsort((fit_indices[global_rows],distances))
    picked=[];picked_dist=[];seen=set()
    for i in order:
        key=projects[global_rows[i]]
        if key in seen:continue
        seen.add(key)
        if len(picked)<RULES['maximum_unique_projects']:
            picked.append(int(global_rows[i]));picked_dist.append(float(distances[i]))
    return np.asarray(picked,dtype='int64'),np.asarray(picked_dist,dtype='float64'),len(seen)

def extent_summary(values):
    a=np.asarray(values,dtype='float64');a=a[np.isfinite(a)&(a>=0)]
    if not len(a):return {'rows':0,'quantiles':None,'bins':{}}
    return {'rows':len(a),'quantiles':{str(q):float(np.quantile(a,q)) for q in [0,.25,.5,.75,.9,.95,1]},
        'bins':{'0':int((a==0).sum()),'(0,100]':int(((a>0)&(a<=100)).sum()),'(100,500]':int(((a>100)&(a<=500)).sum()),
                '(500,1000]':int(((a>500)&(a<=1000)).sum()),'(1000,2000]':int(((a>1000)&(a<=2000)).sum()),'>2000':int((a>2000).sum())}}


def count_summary(values):
    a=np.asarray(values,dtype='int64')
    if not len(a):return {'rows':0,'quantiles':None,'bins':{},'at_least':{}}
    bins={'0':int((a==0).sum()),'1':int((a==1).sum()),'2':int((a==2).sum()),'3-4':int(((a>=3)&(a<=4)).sum()),
          '5-9':int(((a>=5)&(a<=9)).sum()),'10-19':int(((a>=10)&(a<=19)).sum()),'20-39':int(((a>=20)&(a<=39)).sum()),'40':int((a==40).sum())}
    return {'rows':len(a),'quantiles':{str(q):float(np.quantile(a,q)) for q in [0,.25,.5,.75,.9,.95,1]},'bins':bins,
            'at_least':{str(k):int((a>=k).sum()) for k in [1,3,5,10,20,40]}}

def cohort(mask,eligible,counts,other,cross,nearest):
    mask=np.asarray(mask,dtype=bool)
    n=int(mask.sum());valid=mask&eligible;found=mask&(counts>0)
    return {'query_rows':n,'geo_eligible_rows':int(valid.sum()),'geo_supported_rows':int(found.sum()),
            'cross_project_supported_rows':int((mask&(other>0)).sum()),'cross_road_cross_project_supported_rows':int((mask&(cross>0)).sum()),
            'geo_supported_fraction_all_rows':float(found.sum()/n) if n else None,
            'unique_project_count_all_rows':count_summary(counts[mask]),'unique_project_count_geo_eligible':count_summary(counts[valid]),
            'nonself_project_count_geo_eligible':count_summary(other[valid]),'cross_road_nonself_count_geo_eligible':count_summary(cross[valid]),
            'nearest_distance_m_quantiles_supported':{str(q):float(np.quantile(nearest[found],q)) for q in [.25,.5,.75,.9,1]} if found.any() else None}

def self_check():
    pts=xyz([25]*6,[121,121.0001,121.001,121.002,121.003,121.03])
    tree=cKDTree(pts);projects=np.array(['A','A','B','C','D','E'],dtype=object)
    rows,dist,n=nearest_projects(tree,np.arange(6),pts[0],projects,np.arange(6))
    need(list(projects[rows])==['A','B','C','D'] and n==4,'Project dedup/radius failed')
    need(rows[0]==0 and dist[0]==0 and np.all(dist<=2000),'Actual row or distance failed')
    many=xyz(np.repeat(25.,50),121+np.arange(50)*.00001)
    rows,dist,n=nearest_projects(cKDTree(many),np.arange(50),many[0],np.array([str(i) for i in range(50)]),np.arange(50))
    need(len(rows)==40 and n==50,'Fixed nearest-40 limit failed')
    return {'status':'passed','checks':['same_project_once','actual_fit_row_preserved','radius_2000m','fixed_cap_40','uncapped_count_descriptive_only'],'target_or_model_used':False}


def run():
    started=time.monotonic();synthetic=self_check()
    plan_path=ROOT/'outputs/round6/candidate_plan.json'
    geo_path=ROOT/'outputs/round6/geography/development/manifest.json'
    plan=read_locked(plan_path);manifest=read_locked(geo_path)
    need(manifest['stage']=='development' and manifest['snapshot_year']==2022,'Wrong geographic snapshot')
    names=['data_cache/train_features.pkl','outputs/round6/geography/development/train.pkl',
           'outputs/round6/geography/development/train_source_cache_indices.npy',
           'outputs/round6/models/development/geography_catboost/fit_indices.npy',
           'outputs/round6/models/development/geography_catboost/query_indices.npy']
    paths=[ROOT/name for name in names]
    inputs={name:sha(path) for name,path in zip(names,paths)}
    inputs.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [plan_path,plan_path.with_suffix('.sha256'),geo_path,geo_path.with_suffix('.sha256'),Path(__file__),ROOT/'src/round2_neighborhood.py',ROOT/'src/house_features.py']})
    need(inputs[names[0]]==plan['data_sha256'][names[0]],'Original X cache changed')
    need(inputs[names[1]]==manifest['features']['train']['sha256'],'Aligned geography changed')
    need(inputs[names[2]]==manifest['index_files_sha256'][names[2]],'Canonical alignment changed')
    cache=pd.read_pickle(paths[0]);geo_all=pd.read_pickle(paths[1])
    need('總價元' not in cache and '編號' not in cache and cache.shape==(455784,87),'X cache is not expected target-free schema')
    source_indices=np.load(paths[2],allow_pickle=False)
    need(source_indices.shape==(455255,) and np.all(np.diff(source_indices)>0),'Bad canonical cache mapping')
    X=cache.iloc[source_indices].reset_index(drop=True);del cache
    need(len(X)==len(geo_all)==455255,'Canonical X/geography row mismatch')
    fit_idx=np.load(paths[3],allow_pickle=False);query_idx=np.load(paths[4],allow_pickle=False)
    spec=plan['split_contract']['development']
    for role,values in [('fit',fit_idx),('query',query_idx)]:
        need(len(values)==spec[role+'_rows'] and hashlib.sha256(values.astype('<i8').tobytes()).hexdigest()==spec[role+'_indices_content_sha256'],'Frozen split differs')
    need(len(fit_idx)==240316 and len(query_idx)==81234 and not np.intersect1d(fit_idx,query_idx).size,'Unexpected or overlapping fixed split')
    fit=X.iloc[fit_idx].reset_index(drop=True);query=X.iloc[query_idx].reset_index(drop=True)
    fit_geo=geo_all.iloc[fit_idx].reset_index(drop=True);query_geo=geo_all.iloc[query_idx].reset_index(drop=True)
    fit_dates=pd.Timestamp('2000-01-01')+pd.to_timedelta(fit.trade_day_index,unit='D')
    query_dates=pd.Timestamp('2000-01-01')+pd.to_timedelta(query.trade_day_index,unit='D')
    need(fit_dates.lt('2022-08-11').all() and query_dates.between('2022-08-11','2023-08-10').all(),'Temporal X-only dates differ')
    fit_ok,fit_quality=geo_mask(fit,fit_geo);query_ok,query_quality=geo_mask(query,query_geo)
    fp,fmissing=project_keys(fit);qp,qmissing=project_keys(query)
    known=set(fp[~fmissing]);labels=np.where(qmissing,'missing',np.where(qp.isin(known),'seen','new_nonmissing'))
    fit_quality['eligible_but_missing_project_rows']=int((fit_ok&fmissing).sum());fit_ok &= ~fmissing
    work=pd.DataFrame({c:norm(fit[c]) for c in RULES['same_group']})
    work['latitude']=fit_geo.geo_latitude.to_numpy(dtype='float64');work['longitude']=fit_geo.geo_longitude.to_numpy(dtype='float64')
    fit_extent=np.expm1(fit_geo.geo_log_extent.to_numpy(dtype='float64'));query_extent=np.expm1(query_geo.geo_log_extent.to_numpy(dtype='float64'))
    work['geometry_extent_m']=fit_extent
    work['project_key']=fp.to_numpy();work['fit_canonical_row_index']=fit_idx;work['feature_cache_row_index']=source_indices[fit_idx]
    work['road_key']=norm(fit.location_road).to_numpy();work['road_known']=norm(fit.address_road).ne(MISSING).to_numpy()
    candidate=work.loc[fit_ok].drop_duplicates(subset=RULES['same_group']+['latitude','longitude','project_key'],keep='first').reset_index(drop=True)
    positions=xyz(candidate.latitude,candidate.longitude)
    trees={key:(cKDTree(positions[np.asarray(rows)]),np.asarray(rows,dtype='int64')) for key,rows in candidate.groupby(RULES['same_group'],sort=False).indices.items()}
    candidate_projects=candidate.project_key.to_numpy();candidate_fit_rows=candidate.fit_canonical_row_index.to_numpy(dtype='int64')
    candidate_extent=candidate.geometry_extent_m.to_numpy(dtype='float64')
    candidate_roads=candidate.road_key.to_numpy();candidate_road_known=candidate.road_known.to_numpy(dtype=bool)
    query_groups=list(zip(*(norm(query[c]).tolist() for c in RULES['same_group'])))
    query_positions=xyz(query_geo.geo_latitude,query_geo.geo_longitude)
    query_projects=qp.to_numpy();query_roads=norm(query.location_road).to_numpy();query_road_known=norm(query.address_road).ne(MISSING).to_numpy()
    counts=np.zeros(len(query),dtype='int64');other=counts.copy();cross=counts.copy();uncapped=counts.copy();nearest=np.full(len(query),np.nan)
    budget_counts=counts.copy();budget_other=counts.copy();budget_cross=counts.copy();budget_nearest=np.full(len(query),np.nan)
    cache_neighbors={};missing_groups=set()
    for row in np.flatnonzero(query_ok):
        group=query_groups[row];key=(*group,float(query_geo.geo_latitude.iloc[row]),float(query_geo.geo_longitude.iloc[row]))
        if key not in cache_neighbors:
            if group not in trees:
                answer=(np.empty(0,dtype='int64'),np.empty(0),0);missing_groups.add(group)
            else:
                tree,group_rows=trees[group]
                answer=nearest_projects(tree,group_rows,query_positions[row],candidate_projects,candidate_fit_rows)
            cache_neighbors[key]=answer
        selected,dist,total=cache_neighbors[key];counts[row]=len(selected);uncapped[row]=total
        if len(selected):
            nearest[row]=dist[0]
            nonself=candidate_projects[selected]!=query_projects[row]
            other[row]=int(nonself.sum())
            different_road=nonself&query_road_known[row]&candidate_road_known[selected]&(candidate_roads[selected]!=query_roads[row])
            cross[row]=int(different_road.sum())
            conservative=(query_extent[row]+candidate_extent[selected]+dist)<=2000.0
            budget_counts[row]=int(conservative.sum());budget_other[row]=int((conservative&nonself).sum());budget_cross[row]=int((conservative&different_road).sum())
            if conservative.any():budget_nearest[row]=float(dist[conservative].min())
    upper,support,upper_policy=reference_upper_bound(fit,query)
    cohorts={'all_queries':cohort(np.ones(len(query),dtype=bool),query_ok,counts,other,cross,nearest)}
    for group in ('seen','new_nonmissing','missing'):cohorts[group]=cohort(labels==group,query_ok,counts,other,cross,nearest)
    first_support={level:int((upper==level).sum()) for level in (*LEVELS,'no_X_supported_level')}
    by_v2={level:cohort(upper==level,query_ok,counts,other,cross,nearest) for level in (*LEVELS,'no_X_supported_level')}
    weak=~((support['project']>=6)|(support['address']>=8)|(support['street']>=20))
    budget_cohorts={label:cohort(mask,query_ok,budget_counts,budget_other,budget_cross,budget_nearest) for label,mask in {'all_queries':np.ones(len(query),dtype=bool),'seen':labels=='seen','new_nonmissing':labels=='new_nonmissing','missing':labels=='missing','no_project_address_street_X_support':weak,'new_and_no_project_address_street_X_support':weak&(labels=='new_nonmissing')}.items()}
    geometry_extents={'fit_position_rows':extent_summary(fit_extent[fit_geo.geo_has_position.eq(1)]),'query_position_rows':extent_summary(query_extent[query_geo.geo_has_position.eq(1)]),'deduplicated_fit_candidates':extent_summary(candidate_extent)}
    for name,frame,extents in [('fit',fit_geo,fit_extent),('query',query_geo,query_extent)]:
        for quality in ('road_level','planar_intersection'):
            geometry_extents[name+'_'+quality]=extent_summary(extents[frame.geo_match_status.eq(quality)])
    text_coverage={'policy':upper_policy,'first_X_supported_level_counts':first_support,'by_first_X_supported_level':by_v2,
        'project_address_or_street_any_X_supported_rows':int(((support['project']>=6)|(support['address']>=8)|(support['street']>=20)).sum()),
        'no_project_address_street_X_support':cohort(~((support['project']>=6)|(support['address']>=8)|(support['street']>=20)),query_ok,counts,other,cross,nearest)}
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'rules':RULES,'synthetic_checks':synthetic,
        'source_sha256':inputs,'stage':'development','fit_rows':len(fit),'query_rows':len(query),'query_dates':['2022-08-11','2023-08-10'],
        'canonical_alignment':{'raw_X_cache_rows':455784,'canonical_rows':455255,'removed_rows_in_existing_mapping':529,'target_containing_dedup_rule_reexecuted':False,'preexisting_target_containing_dedup_indices_reused':True},
        'fit_quality':fit_quality,'query_quality':query_quality,
        'fit_named_geographic_rows':int(fit_ok.sum()),'fit_candidate_position_rows_after_X_dedup':len(candidate),
        'fit_named_unique_projects':int(candidate.project_key.nunique()),'fit_city_style_use_groups':len(trees),
        'query_eligible_unique_locations_cached':len(cache_neighbors),'query_cached_reuses':int(query_ok.sum())-len(cache_neighbors),
        'query_groups_with_no_fit_candidates':len(missing_groups),'query_rows_over_40_distinct_projects_before_cap':int((uncapped>40).sum()),
        'selected_candidate_canonical_row_indices_sha256':hashlib.sha256(candidate_fit_rows.astype('<i8').tobytes()).hexdigest(),
        'cohorts':cohorts,'v2_X_only_upper_bound':text_coverage,
        'geometry_extent_distribution_m':geometry_extents,'distance_plus_extent_diagnostic':{'computed':True,'condition':'D_query + D_reference + centre_distance <= 2000 metres','candidate_pool_changed':False,'positions_are_house_error_bounds':False,'cohorts':budget_cohorts},
        'cross_road_semantics':'different original textual location_road keys among the fixed nearest 40, both address_road known, excluding same project; not proof of distinct physical roads',
        'runtime_seconds':time.monotonic()-started,'targets_read':False,'predictions_or_metrics_read':False,'models_fitted_or_scored':False,'external_downloads':False,
        'candidate_or_threshold_selected_from_this_audit':False,'dependencies':{'numpy':np.__version__,'pandas':pd.__version__}}
    for name,digest in inputs.items():need(sha(ROOT/name)==digest,'Input changed during audit: '+name)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    write_report(result)
    print(json.dumps({'status':'completed','json':str(OUT.relative_to(ROOT)),'report':str(REPORT.relative_to(ROOT)),
        'fit_candidate_positions':len(candidate),'fit_unique_projects':result['fit_named_unique_projects'],
        'query_eligible':int(query_ok.sum()),'geo_supported':int((counts>0).sum()),'cross_project_supported':int((other>0).sum()),'cross_road_cross_project_supported':int((cross>0).sum()),
        'source_sha256':sha(Path(__file__)),'audit_sha256':sha(OUT),'report_sha256':sha(REPORT),'targets_read':False,'models_fitted_or_scored':False},ensure_ascii=True),flush=True)

def write_report(result):
    def number(value):return f'{value:,}'
    all_rows=result['cohorts']['all_queries']
    lines=['# 第八輪粗略地理近鄰可行性稽核','',
        '**這是純 X 可用性診斷，沒有訓練、預測、計算 MAE 或選定新模型。** 半徑、鄰居上限與位置品質條件在執行前固定；未因覆蓋率更動規則。',
        '', '## 資料與固定規則','',
        f"使用第六輪 development 原切分：fit {number(result['fit_rows'])} 列；query {number(result['query_rows'])} 列，日期為 2022-08-11～2023-08-10，地圖為既有 2022 年版本。",
        '只讀原 87 欄無價格特徵、8 欄地理資料與保存的來源／切分索引。原特徵快取 455,784 列，沿已保存 source-cache 索引對齊為 455,255 列；重用既有移除 529 列的規則結果，沒有讀價格重算去重。既有去重曾包含目標欄，這個來源限制仍須保留。',
        '',
        '- fit 與 query 都須有位置，座標及 extent 有限，且 `expm1(geo_log_extent) <= 2000` 公尺。城市、建物型態、主要用途都必須非缺失且相同。',
        '- 使用固定球面距離 2,000 公尺；KD-tree 的弦長查詢與此半徑等價，沒有新增 nearest-road 猜测、距離容忍或外部查詢。',
        '- fit 只接受具名地理建案；鍵沿原城市＋行政區＋完整案名。缺案名不當成許多獨立建案，亦不把所有缺名當一個有效建案。',
        '- 同城市／型態／用途／座標／建案的候選位置先合併，保留排序最前的實際 canonical fit 列索引。對每個 query，各案只留半徑內最近的一個實際 fit 位置，同距離以原列序決勝；再取最近至多 40 案。',
        '- 同城市／型態／用途與完全相同座標的 query 共用一次鄰域查詢。道路或同案診斷只在固定的 40 案內計算，不為了提高跨道路覆蓋再向外補案。',
        '', '## 可用位置與去重','',
        '| 項目 | 列數或個數 |','|---|---:|',
        f"| fit 原有位置 | {number(result['fit_quality']['position_rows'])} |",
        f"| fit 有位置且 extent 不超過 2 km | {number(result['fit_quality']['position_with_extent_at_most_2000_rows'])} |",
        f"| fit 通過位置／同類欄位但缺案名，因此排除 | {number(result['fit_quality']['eligible_but_missing_project_rows'])} |",
        f"| fit 具名且通過所有地理條件 | {number(result['fit_named_geographic_rows'])} |",
        f"| 合併相同候選位置後 | {number(result['fit_candidate_position_rows_after_X_dedup'])} |",
        f"| 候選中的不同地理建案 | {number(result['fit_named_unique_projects'])} |",
        f"| fit 城市×型態×用途組 | {number(result['fit_city_style_use_groups'])} |",
        f"| query 通過地理與分組條件 | {number(all_rows['geo_eligible_rows'])} |",
        f"| 實際查詢並快取的不同 query 位置 | {number(result['query_eligible_unique_locations_cached'])} |",
        f"| 使用已快取鄰域的 query 列 | {number(result['query_cached_reuses'])} |",
        f"| 半徑內原有超過 40 案、實際仍截為 40 的 query 列 | {number(result['query_rows_over_40_distinct_projects_before_cap'])} |",
        '', '## 嚴格地理支持與建案組別','',
        '下表的「有地理支持」指固定條件內至少有 1 個具名 fit 建案；不是價格可靠度、模型採用門檻或改善證據。另列排除 query 自己建案後的支持，以及其中不同文字道路鍵的支持。不同文字道路鍵要求雙方原 address_road 都非缺失、location_road 不同；它不是已證實不同實體道路。',
        '', '| query 組別 | 全部列 | 地理條件合格 | 至少 1 案 | 排除自身案仍有支持 | 不同文字道路鍵且不同案 |',
        '|---|---:|---:|---:|---:|---:|']
    labels={'all_queries':'全部','seen':'已見具名案','new_nonmissing':'真正新具名案','missing':'案名缺失'}
    for group,label in labels.items():
        c=result['cohorts'][group]
        lines.append('| '+label+' | '+' | '.join(number(c[k]) for k in ['query_rows','geo_eligible_rows','geo_supported_rows','cross_project_supported_rows','cross_road_cross_project_supported_rows'])+' |')
    lines += ['', '## 鄰近不同建案數分布','',
        '下表分母限於地理條件合格 query；建案已去重，上限 40。0 案仍保留，沒有以缺失或 0 MAE 混淆。',
        '', '| 組別 | 0 案 | 1 案 | 2 案 | 3–4 | 5–9 | 10–19 | 20–39 | 40 | 中位數 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group,label in labels.items():
        d=result['cohorts'][group]['unique_project_count_geo_eligible'];bins=d['bins']
        median='—' if d['quantiles'] is None else str(d['quantiles']['0.5'])
        lines.append('| '+label+' | '+' | '.join(number(bins.get(k,0)) for k in ['0','1','2','3-4','5-9','10-19','20-39','40'])+' | '+median+' |')
    lines += ['', '完整 JSON 另列 ≥1／3／5／10／20／40 案的列數、排除自身案後與不同文字道路鍵的分布，以及最近代表位置距離。這些是描述性切面，不是掃描接受門檻。',
        '', '## 與既有 v2 文字參考的純 X 重疊','',
        '依 frozen v2 的 X 分組式、15～1,500 m² 面積條件及既有最少列數（建案 6、完整地址 8、路段／里村地段／道路 20、行政區屋型 80），計算每個 query 可能先有支持的層級。**沒有套用單價分位數、價格離散度或 reference 信心**；因此只是價格篩選前的樂觀可用性上界，不能宣稱實際 v2 已選到該層。',
        '', '| 第一個 X 支持層級 | query 列數 | 有地理支持 | 不同文字道路鍵且不同案 |',
        '|---|---:|---:|---:|']
    upper=result['v2_X_only_upper_bound']
    for level in (*LEVELS,'no_X_supported_level'):
        c=upper['by_first_X_supported_level'][level]
        lines.append(f"| {level} | {number(c['query_rows'])} | {number(c['geo_supported_rows'])} | {number(c['cross_road_cross_project_supported_rows'])} |")
    gap=upper['no_project_address_street_X_support']
    lines += ['',
        f"在 project／address／street 三種鍵皆不足既有最少列數的 {number(gap['query_rows'])} 筆 query 中，{number(gap['geo_supported_rows'])} 筆仍有至少一個符合固定條件的近鄰建案；其中 {number(gap['cross_road_cross_project_supported_rows'])} 筆另有不同文字道路鍵且不同案的支持。這只能表示資訊可能互補，不能推論房價補正必然有效。",
        '', '## 限制與後續用途','',
        '- 沿用原 address_city／address_district／location_road 鍵，保留既有行政區截斷及道路文字粒度限制；不同鍵可能仍是同一實體道路，同鍵也可能涵蓋較長路段。',
        '- 道路代表點與雙路平面交點均為粗略位置；extent 是幾何支持範圍，單點為 0 不等於住宅定位誤差為 0。即使 extent 不超過 2 km，也不保證真實住宅位於代表點。',
        '- 具名案去重避免同案大量交易膨脹案數；同址異名、跨案共用名稱或文字歧義仍可能存在。沒有使用外部企業、地籍或成交答案校正。',
        '- 只查 development 一個年度，沒有查看確認期、輔助期或 test 分數，也沒有讀取價格、模型預測或既有 MAE。',
        '- 研究結果不改動 95 欄、reference、模型或凍結規約；正式候選需由主流程另外預先固定。可用性不等同價格預測品質。',
        '', '## 重現與輸出','',
        '```text',r'.venv\Scripts\python.exe src\round8_location_feasibility.py','```','',
        '- [完整純 X 結果及來源 SHA](../outputs/round8/audit/location_feasibility.json)',
        '- [本次可重跑程式](../src/round8_location_feasibility.py)',
        '- [既有 development 地理 manifest](../outputs/round6/geography/development/manifest.json)',
        '- [既有 R6 切分計畫](../outputs/round6/candidate_plan.json)',
        '- [v2 原參考分組來源](../src/round2_neighborhood.py)','',
        '所有讀入檔案在產出前再次核對 SHA；JSON 記錄來源、規則、依賴版本與合成檢查。既有資料與程式保持不變。','']
    lines += extent_report(result)
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def extent_report(result):
    title='\u5e7e\u4f55\u7bc4\u570d\u8207\u8ddd\u96e2\u52a0\u7e3d\u8a3a\u65b7'
    lines=['','## '+title,'',
        '\u4ee5\u4e0b\u53ea\u9644\u52a0\u63cf\u8ff0\u6027\u7d71\u8a08\uff0c\u4e0d\u6539\u539f\u672c 2 km / 40 \u6848\u5019\u9078\u6c60\u3002Dq\u3001Dr \u70ba\u67e5\u8a62\u8207\u53c3\u8003\u7684\u5e7e\u4f55 bbox \u5c0d\u89d2\u7dda\uff0c\u4e0d\u662f\u5df2\u9a57\u8b49\u7684\u4f4f\u5b85\u5b9a\u4f4d\u8aa4\u5dee\u4e0a\u754c\u3002',
        '', '| extent cohort | rows | median m | p90 m | <= 2000 m |','|---|---:|---:|---:|---:|']
    for name,value in result['geometry_extent_distribution_m'].items():
        q=value['quantiles']
        median='n/a' if q is None else f"{q['0.5']:.2f}"
        p90='n/a' if q is None else f"{q['0.9']:.2f}"
        lines.append(f"| {name} | {value['rows']:,} | {median} | {p90} | {value['rows']-value['bins'].get('>2000',0):,} |")
    lines += ['',
        '\u53e6\u5728\u540c\u4e00\u6279\u5df2\u9078\u7684\u6700\u591a 40 \u6848\u5167\uff0c\u6838\u5c0d `Dq + Dr + distance <= 2000 m`\u3002\u4e0d\u6eff\u8db3\u6642\u4e0d\u5f80\u5916\u88dc\u6848\uff0c\u4e0d\u6539\u53c3\u8003\u4ee3\u8868\u5217\u3002\u9019\u662f\u5e7e\u4f55\u652f\u6301\u7bc4\u570d\u7684\u984d\u5916\u7d04\u675f\uff0c\u4e0d\u662f\u5df2\u9078\u7528\u7684\u6a21\u578b\u9580\u6abb\u3002',
        '', '| query cohort | all rows | >=1 budget-supported project | >=1 other project | >=1 other project / other text road | >=3 projects | >=5 projects |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in result['distance_plus_extent_diagnostic']['cohorts'].items():
        a=c['unique_project_count_geo_eligible']['at_least']
        lines.append(f"| {name} | {c['query_rows']:,} | {c['geo_supported_rows']:,} | {c['cross_project_supported_rows']:,} | {c['cross_road_cross_project_supported_rows']:,} | {a.get('3',0):,} | {a.get('5',0):,} |")
    lines += ['', '\u7d50\u679c\u50c5\u8b49\u660e\u90e8\u5206\u67e5\u8a62\u6709\u6b64\u985e\u5e7e\u4f55\u652f\u6301\uff0c\u4e0d\u80fd\u5f9e\u8986\u84cb\u7387\u63a8\u4f30\u623f\u50f9\u6a21\u578b\u7684 MAE \u6539\u5584\u3002', '']
    return lines


if __name__=='__main__':
    run()