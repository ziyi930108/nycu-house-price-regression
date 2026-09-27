"""Fixed R21 geo8-only support audit. Never load target/X87/model/predictions."""
from __future__ import annotations
import argparse,hashlib,json,sys,uuid
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round21_research'
NUMERIC=['geo_latitude','geo_longitude','geo_log_road_length','geo_log_extent','geo_has_position','geo_nearby_fragments']
CATEGORICAL=['geo_match_status','geo_grid_002']
COLUMNS=NUMERIC+CATEGORICAL
MISSING='__MISSING__'
RULES={
 'grid_degrees':.005,'shift_degrees_latitude':.0025,'shift_degrees_longitude':.0025,
 'arithmetic':'promote stored float32 coordinates to float64 before all arithmetic',
 'regular':'floor(lat64 / .005):floor(lon64 / .005)',
 'shifted':'floor((lat64 + .0025) / .005):floor((lon64 + .0025) / .005)',
 'original_geo_grid_002':'preserve original category; do not regenerate',
 'usable_statuses':['road_level','planar_intersection'],
 'missing_position_grid':MISSING,'support_bins':['0','1-4','5-19','20+'],
 'support_unit':'actual fit rows; distinct stored latitude/longitude pairs reported separately',
 'boundary_stress':'each stored coordinate moved to its adjacent float32 values in both directions; one-ULP stress, not recovered original-coordinate uncertainty',
 'extent':'expm1(float64(stored geo_log_extent)); geometry support span, not house-position error',
 'support_filters_or_thresholds':False,'search_performed':False,'target_fitted_state':False,
 'price_candidate':'original95 plus regular and shifted categorical grids; original R6 2999/.065/Plain parameters unchanged',
}
PATHS={
 'snapshot':'runs/v3/snapshot_manifest.json',
 'manifest':'outputs/round6/geography/development/manifest.json',
 'geo':'outputs/round6/geography/development/train.pkl',
 'fit_indices':'outputs/round6/models/development/geography_catboost/fit_indices.npy',
 'query_indices':'outputs/round6/models/development/geography_catboost/query_indices.npy',
}
EXPECTED={
 'snapshot':'77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525',
 'manifest':'4b21f0620aedb8e9527044e9a44a4e5d6f8cae49e71210252c56ba614a5151c6',
 'geo':'35289a9eddb1bbe140280181ccda3342b49359ab06f8b83d26150e520e9abefb',
 'fit_indices':'8ce930622540ef9321c94b347c58adad4005c7e4dd209484874594b2bb5fe0ac',
 'query_indices':'2e1d097650f727b3e03a19d1d6aca811b8aab546573a7900a6edafeb61ed080d',
}
HISTORY=[
 'src/round4_spatial_raw.py','src/round4_augmented.py','src/round6_geo.py',
 'src/round8_spatial_reference.py','src/round9_geo_poi.py',
 *['reports/round'+str(i)+'_plan.md' for i in range(13,21)],
]
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def content_sha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
def save_new(path,value):
 path=Path(path);receipt=path.with_suffix('.sha256')
 require(not path.exists() and not receipt.exists(),'Refusing overwrite: '+str(path))
 path.parent.mkdir(parents=True,exist_ok=True)
 body=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
 with tmp.open('xb') as f:f.write(body)
 digest=hashlib.sha256(body).hexdigest();tmp.rename(path)
 with receipt.open('xb') as f:f.write((digest+'\n').encode())
 return digest

def validate_geo(geo):
 require(isinstance(geo,pd.DataFrame) and list(geo)==COLUMNS,'Expected native geo8 only')
 require(geo.index.is_unique,'Duplicate geo frame index')
 require(all(str(geo[c].dtype)=='float32' for c in NUMERIC),'Numeric dtype must remain float32')
 require(all(geo[c].map(lambda v:isinstance(v,str)).all() for c in CATEGORICAL),'Categories must be strings')
 usable=geo.geo_has_position.eq(1)
 require(geo.geo_has_position.isin([0,1]).all(),'Invalid position flag')
 require(usable.equals(geo.geo_match_status.isin(RULES['usable_statuses'])),'Status/position mismatch')
 require(geo.loc[~usable,[c for c in NUMERIC if c!='geo_has_position']].isna().all().all(),'Missing position has geometry values')
 require(geo.loc[~usable,'geo_grid_002'].eq(MISSING).all(),'Missing position has grid')
 require(np.isfinite(geo.loc[usable,NUMERIC].to_numpy()).all(),'Nonfinite usable geometry')
 require(geo.loc[usable,'geo_latitude'].between(20,28).all() and geo.loc[usable,'geo_longitude'].between(116,125).all(),'Coordinate range')
 require(geo.loc[usable,['geo_log_extent','geo_log_road_length']].ge(0).all().all(),'Negative log geometry span')
 require(geo.loc[usable,'geo_nearby_fragments'].isin([0,1]).all(),'Invalid fragment flag')
 require(geo.loc[usable,'geo_grid_002'].str.fullmatch(r'\d+:\d+').all(),'Malformed preserved coarse grid')
 return usable.to_numpy()

def indices_ok(fi,qi,n,expected_counts=None):
 for name,a in [('fit',fi),('query',qi)]:
  require(a.dtype==np.dtype('int64') and a.ndim==1,'Wrong '+name+' index dtype/shape')
  require(len(np.unique(a))==len(a) and ((a>=0)&(a<n)).all(),'Invalid '+name+' index positions')
  require(len(a)<2 or np.all(a[1:]>a[:-1]),'Source indices not strictly increasing')
 require(not np.intersect1d(fi,qi).size,'Fit/query overlap')
 if expected_counts is not None:require((len(fi),len(qi))==expected_counts,'Unexpected split sizes')

def grid_pairs(coords,shift):return np.floor((coords+float(shift))/.005).astype('int64')
def keys(pairs):return np.array([str(int(a))+':'+str(int(b)) for a,b in pairs],dtype=object)
def grids(geo):
 usable=validate_geo(geo);coords=geo.loc[usable,['geo_latitude','geo_longitude']].to_numpy(dtype='float64')
 result=pd.DataFrame(index=geo.index)
 for name,shift in [('regular',0.),('shifted',.0025)]:
  values=np.full(len(geo),MISSING,dtype=object);values[usable]=keys(grid_pairs(coords,shift));result[name]=values
 return result

def distribution(values):
 a=np.asarray(values,dtype='float64')
 if not len(a):return {'count':0,'min':None,'p25':None,'median':None,'p75':None,'p90':None,'p99':None,'max':None}
 require(np.isfinite(a).all(),'Nonfinite descriptive values')
 v=np.quantile(a,[0,.25,.5,.75,.9,.99,1])
 return dict(count=len(a),**dict(zip(['min','p25','median','p75','p90','p99','max'],map(float,v))))
def bins(values):
 a=np.asarray(values)
 return {'0':int(np.sum(a==0)),'1-4':int(np.sum((a>=1)&(a<=4))),
         '5-19':int(np.sum((a>=5)&(a<=19))),'20+':int(np.sum(a>=20))}

def frame_summary(geo,g):
 u=validate_geo(geo);subset=geo.loc[u];coords=subset[['geo_latitude','geo_longitude']].to_numpy(dtype='float64')
 coarse=keys(grid_pairs(coords,0.)//4)
 mismatch=int(np.sum(coarse!=subset.geo_grid_002.to_numpy()))
 result={'rows':len(geo),'position_rows':int(u.sum()),'missing_position_rows':int((~u).sum()),
  'position_rate':float(u.mean()) if len(u) else None,'distinct_coordinates':len(np.unique(coords,axis=0)),
  'status_counts':{str(k):int(v) for k,v in geo.geo_match_status.value_counts().sort_index().items()},
  'coarse_parent_mismatch_rows':mismatch,'coarse_parent_comparison':'floor(regular_lat_index/4):floor(regular_lon_index/4) vs preserved original geo_grid_002',
  'extent_metres':distribution(np.expm1(subset.geo_log_extent.to_numpy(dtype='float64'))),
  'by_usable_status':{},'float32_boundary_stress':{}}
 for status in RULES['usable_statuses']:
  s=geo.loc[geo.geo_match_status.eq(status)]
  result['by_usable_status'][status]={'rows':len(s),'distinct_coordinates':len(s[['geo_latitude','geo_longitude']].drop_duplicates()),
   'extent_metres':distribution(np.expm1(s.geo_log_extent.to_numpy(dtype='float64')))}
 c32=coords.astype('float32');lower=np.nextafter(c32,np.float32(-np.inf)).astype('float64');upper=np.nextafter(c32,np.float32(np.inf)).astype('float64')
 for name,shift in [('regular',0.),('shifted',.0025)]:
  sensitive=np.any(grid_pairs(lower,shift)!=grid_pairs(upper,shift),axis=1)
  result['float32_boundary_stress'][name]={'rows':int(sensitive.sum()),'distinct_coordinates':len(np.unique(coords[sensitive],axis=0)),
   'denominator_position_rows':int(u.sum()),'interpretation':RULES['boundary_stress']}
 return result

def fit_support(geo,g):
 u=validate_geo(geo);sub=geo.loc[u];details={};tables={}
 for name in ('regular','shifted'):
  table=pd.DataFrame({'grid':g.loc[u,name].to_numpy(),'lat':sub.geo_latitude.to_numpy(),'lon':sub.geo_longitude.to_numpy()})
  grouped=table.groupby(['grid','lat','lon'],sort=True).size().rename('rows').reset_index()
  stats=grouped.groupby('grid',sort=True).agg(fit_rows=('rows','sum'),distinct_coordinates=('rows','size'),largest_coordinate_rows=('rows','max'))
  only=stats.distinct_coordinates.eq(1)
  details[name]={'occupied_cells':len(stats),'fit_rows_per_cell':distribution(stats.fit_rows),
   'distinct_coordinates_per_cell':distribution(stats.distinct_coordinates),
   'fit_row_support_bins_per_cell':bins(stats.fit_rows),'distinct_coordinate_bins_per_cell':bins(stats.distinct_coordinates),
   'one_coordinate_cells':int(only.sum()),'rows_in_one_coordinate_cells':int(stats.loc[only,'fit_rows'].sum()),
   'one_coordinate_cell_fraction':float(only.mean()) if len(stats) else None,
   'row_fraction_in_one_coordinate_cells':float(stats.loc[only,'fit_rows'].sum()/len(table)) if len(table) else None,
   'largest_single_coordinate_share_per_cell':distribution(stats.largest_coordinate_rows/stats.fit_rows)}
  tables[name]=stats
 return details,tables

def query_support(geo,g,tables):
 u=validate_geo(geo);q=g.loc[u];result={'denominator_position_rows':len(q),'missing_positions_excluded':int((~u).sum())};unseen={}
 for name in ('regular','shifted'):
  n=q[name].map(tables[name].fit_rows).fillna(0).to_numpy(dtype='int64')
  c=q[name].map(tables[name].distinct_coordinates).fillna(0).to_numpy(dtype='int64')
  unseen[name]=n==0
  result[name]={'fit_row_support_bins':bins(n),'fit_distinct_coordinate_support_bins':bins(c),
   'unseen_rows':int((n==0).sum()),'unseen_rate':float((n==0).mean()) if len(n) else None,
   'query_rows_supported_by_only_one_fit_coordinate':int((c==1).sum())}
 regular,shifted=unseen['regular'],unseen['shifted']
 for label,mask in [('any_unseen',regular|shifted),('both_unseen',regular&shifted),('regular_unseen_shifted_seen',regular&~shifted),('both_seen',~regular&~shifted)]:
  result[label]={'rows':int(mask.sum()),'rate':float(mask.mean()) if len(mask) else None}
 result['any_unseen_definition']='at least one of the two grids absent in fit'
 result['both_unseen_definition']='both grids absent in fit'
 return result

def self_check():
 checks=[]
 def ok(name,v):require(v,name);checks.append(name)
 def rejects(name,fn):
  try:fn()
  except (ValueError,KeyError,TypeError):checks.append(name)
  else:raise AssertionError(name)
 idx=pd.Index([7,2,10,9,18]);g=pd.DataFrame(index=idx)
 for c in NUMERIC:g[c]=np.full(5,np.nan,dtype='float32')
 g.loc[idx[:4],'geo_latitude']=np.array([24.001,24.004,24.004,24.06],dtype='float32')
 g.loc[idx[:4],'geo_longitude']=np.array([121.001,121.004,121.004,121.06],dtype='float32')
 g.loc[idx[:4],['geo_log_road_length','geo_log_extent','geo_nearby_fragments']]=np.float32(0)
 g['geo_has_position']=np.array([1,1,1,1,0],dtype='float32')
 g['geo_match_status']=pd.Series(['road_level','road_level','road_level','planar_intersection','ineligible_address'],index=idx)
 g['geo_grid_002']=pd.Series(['1200:6050','1200:6050','1200:6050','1203:6053',MISSING],index=idx)
 original=g.copy(deep=True);out=grids(g)
 ok('both_axis_plus_half_cell_not_minus',out.loc[2,'shifted']=='4801:24201' and out.loc[2,'regular']=='4800:24200')
 ok('missing_position_not_zero_grid',out.loc[18].eq(MISSING).all())
 ok('index_and_input_unchanged',out.index.equals(idx) and g.equals(original))
 ok('float64_promotion_before_division',out.loc[9,'regular']==keys(grid_pairs(g.loc[[9],['geo_latitude','geo_longitude']].to_numpy(dtype='float64'),0))[0])
 summary=frame_summary(g,out);ok('old_float64_coarse_parent_mismatch_is_descriptive',summary['coarse_parent_mismatch_rows']>=1)
 ok('one_ULP_boundary_stress_is_reported',summary['float32_boundary_stress']['regular']['rows']>=1)
 details,tables=fit_support(g,out);ok('row_and_coordinate_support_not_conflated',details['regular']['fit_rows_per_cell']['max']==3 and details['regular']['distinct_coordinates_per_cell']['max']==2)
 q=query_support(g,out,tables);ok('missing_excluded_from_unseen',q['denominator_position_rows']==4 and q['missing_positions_excluded']==1 and q['both_unseen']['rows']==0)
 ok('fixed_support_bins',bins([0,1,4,5,19,20])=={'0':1,'1-4':2,'5-19':2,'20+':1})
 bad=g.copy();bad['geo_latitude']=bad.geo_latitude.astype('float64');rejects('wrong_dtype_rejected',lambda:grids(bad))
 bad=g.copy();bad.loc[18,'geo_latitude']=np.float32(0);rejects('geometry_on_missing_rejected',lambda:grids(bad))
 bad=g.copy();bad.loc[7,'geo_match_status']='rejected';rejects('status_position_mismatch_rejected',lambda:grids(bad))
 rejects('indices_overlap_rejected',lambda:indices_ok(np.array([0,1],dtype='int64'),np.array([1,2],dtype='int64'),5))
 rejects('indices_out_of_bounds_rejected',lambda:indices_ok(np.array([0],dtype='int64'),np.array([5],dtype='int64'),5))
 return checks

def verify_inputs():
 inputs={}
 for key,name in PATHS.items():
  path=ROOT/name;digest=sha(path);require(digest==EXPECTED[key],'Input identity differs: '+name)
  inputs[key]={'path':name,'sha256':digest,'bytes':path.stat().st_size}
  if key in ('snapshot','manifest'):
   receipt=path.with_suffix('.sha256');require(receipt.read_text(encoding='utf-8').strip()==digest,'Receipt differs: '+name)
   inputs[key+'_receipt']={'path':rel(receipt),'sha256':sha(receipt),'bytes':receipt.stat().st_size}
 snap=read(ROOT/PATHS['snapshot']);manifest=read(ROOT/PATHS['manifest'])
 require(snap['version']=='v3' and snap['is_release'] is True,'Snapshot identity')
 def snapshot_binding(name,digest,size):
  records=[]
  if name in snap['copied_files']:
   item=snap['copied_files'][name];records.append((item['original_sha256'],item['original_bytes']))
  records.extend((v['sha256'],v['bytes']) for v in snap['external_dependencies'] if v['original_path']==name)
  require(len(records)>=1 and all(x==(digest,size) for x in records),'Snapshot input binding: '+name)
 for key in ('manifest','geo','fit_indices','query_indices'):
  item=inputs[key];snapshot_binding(item['path'],item['sha256'],item['bytes'])
 require(manifest['stage']=='development' and manifest['snapshot_year']==2022 and manifest['geography_provenance_chain']['verified'] is True,'Wrong saved geographic stage')
 require(manifest['features']['train']=={'path':PATHS['geo'],'rows':455255,'sha256':EXPECTED['geo']},'Declared native geography differs')
 require(manifest['alignment']['train']['canonical_rows']==455255 and manifest['alignment']['train']['removed_duplicate_rows']==529,'Canonical geography alignment differs')
 require(manifest['feature_rules']['additional_numeric']==NUMERIC and manifest['feature_rules']['additional_categorical']==CATEGORICAL,'Declared geo8 schema differs')
 require(manifest['feature_rules']['grid_encoding']=='floor(raw_float64_latitude/0.02):floor(raw_float64_longitude/0.02)','Original coarse grid provenance differs')
 for name in HISTORY:
  if name=='src/round6_geo.py':require(sha(ROOT/name)==manifest['source_sha256'][name],'Original R6 encoder differs')
 return inputs,manifest

def audit():
 target=OUT/'grid_audit.json';require(not target.exists() and not target.with_suffix('.sha256').exists(),'Audit already exists')
 pre=read(OUT/'self_check.json');require(sha(OUT/'self_check.json')==(OUT/'self_check.sha256').read_text().strip(),'Self-check receipt differs')
 require(pre['script_sha256']==sha(Path(__file__)) and pre['status']=='passed' and pre['synthetic_only'] is True,'Run fixed source synthetic check first')
 method=ROOT/'reports/round21_method_review.md';before_method=method.read_bytes();require(b'<!-- R21_AUDIT_RESULTS -->' not in before_method,'Method already has results')
 preserved=OUT/'method_before_audit.md'
 with preserved.open('xb') as f:f.write(before_method)
 inputs,manifest=verify_inputs()
 print(json.dumps({'event':'verified_target_free_inputs','paths':[v['path'] for v in inputs.values()]},ensure_ascii=False),flush=True)
 geo=pd.read_pickle(ROOT/PATHS['geo']);fi=np.load(ROOT/PATHS['fit_indices'],allow_pickle=False);qi=np.load(ROOT/PATHS['query_indices'],allow_pickle=False)
 require(len(geo)==455255,'Native canonical geography row count differs');validate_geo(geo);indices_ok(fi,qi,len(geo),(240316,81234))
 fit=geo.iloc[fi].copy();query=geo.iloc[qi].copy();fg=grids(fit);qg=grids(query)
 fit_summary=frame_summary(fit,fg);query_summary=frame_summary(query,qg)
 support,tables=fit_support(fit,fg);query_summary['support']=query_support(query,qg,tables)
 for status in RULES['usable_statuses']:
  mask=query.geo_match_status.eq(status);query_summary['by_usable_status'][status]['support']=query_support(query.loc[mask],qg.loc[mask],tables)
 evidence={name:sha(ROOT/name) for name in HISTORY}
 evidence[rel(preserved)]=sha(preserved)
 result={'round':21,'status':'completed_pure_X_only','rules':RULES,'source_sha256':{rel(Path(__file__)):sha(Path(__file__))},
  'inputs':inputs,'evidence_sha256':evidence,'synthetic_evidence':{'path':rel(OUT/'self_check.json'),'sha256':sha(OUT/'self_check.json'),'checks':pre['checks']},
  'scope':{'deserialized_frames':[PATHS['geo']],'deserialized_arrays':[PATHS['fit_indices'],PATHS['query_indices']],
   'X87_or_projects_read':False,'target_cache_or_Excel_read':False,'model_or_predictions_read':False,'prices_or_scores_computed':False,
   'query_X_observed':True,'historically_untouched_blind_data':False,'existing_target_containing_dedup_reused_not_recomputed':True},
  'alignment':{'canonical_rows':len(geo),'fit_rows':len(fi),'query_rows':len(qi),'fit_before':'2022-08-11','query_inclusive':['2022-08-11','2023-08-10'],
   'dates_not_independently_recomputed':'native geo8 contains no transaction dates; exact historical saved index bytes and counts are verified',
   'fit_indices_values_sha256':hashlib.sha256(fi.astype('<i8').tobytes()).hexdigest(),'query_indices_values_sha256':hashlib.sha256(qi.astype('<i8').tobytes()).hexdigest()},
  'fit':fit_summary,'fit_grid_support':support,'query':query_summary,
  'limitations':['Road representative/planar intersection is not a residence address.','Extent is geometry support span, not a calibrated error bound.',
   'Distinct coordinates are not independent projects or physical facilities. No X87/project values were read.',
   'Two offset grids do not make all nearby pairs co-cell, and add sparse categorical capacity.',
   'The original coarse grid used pre-cast float64; exact hierarchy is not assumed.',
   'Historical method text exposed an old R8 aggregate summary during source review; no result JSON/model/arrays were opened or score used to set these rules.'],
  'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}}
 for item in inputs.values():require(sha(ROOT/item['path'])==item['sha256'],'Input changed during audit')
 require(method.read_bytes()==before_method,'Method changed during audit')
 digest=save_new(target,result)
 completion=['','<!-- R21_AUDIT_RESULTS -->','## 固定盤點完成結果','',
  f'- Fit {len(fi):,} 列：位置可用 {fit_summary["position_rows"]:,}；query {len(qi):,} 列：位置可用 {query_summary["position_rows"]:,}。',
  f'- 正網格／偏移網格占用 fit cells：{support["regular"]["occupied_cells"]:,}／{support["shifted"]["occupied_cells"]:,}；只含單一座標 cells：{support["regular"]["one_coordinate_cells"]:,}／{support["shifted"]["one_coordinate_cells"]:,}。',
  f'- 有位置 query 的正網格未知 {query_summary["support"]["regular"]["unseen_rows"]:,}，偏移未知 {query_summary["support"]["shifted"]["unseen_rows"]:,}；兩者皆未知 {query_summary["support"]["both_unseen"]["rows"]:,}；正網格未知但偏移已見 {query_summary["support"]["regular_unseen_shifted_seen"]["rows"]:,}。',
  f'- 原 coarse parent 比對不一致：fit {fit_summary["coarse_parent_mismatch_rows"]:,}，query {query_summary["coarse_parent_mismatch_rows"]:,}；屬描述性量化，不修改原格網。',
  '- 固定 bins、distinct coordinate 支持、單座標集中、status／extent 分布及一個 float32 ULP 擾動的敏感列數完整保存在 JSON；不以結果新增篩選、尺寸或接受門檻。',
  f'- [完整純X結果](../outputs/round21_research/grid_audit.json) SHA `{digest}`；[合成檢查](../outputs/round21_research/self_check.json)；執行前固定本文保留於 [method_before_audit.md](../outputs/round21_research/method_before_audit.md)。',
  '- 本次成功盤點未證明價格改善，也未訓練或選型。','']
 with method.open('ab') as f:f.write(('\n'.join(completion)).encode('utf-8'))
 print(json.dumps({'event':'audit_completed','path':rel(target),'sha256':digest,'fit_position_rows':fit_summary['position_rows'],
  'query_position_rows':query_summary['position_rows'],'support':query_summary['support'],'method_sha256':sha(method)},ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['self-check','audit']);args=p.parse_args()
 if args.action=='self-check':
  checks=self_check();value={'status':'passed','synthetic_only':True,'script_sha256':sha(Path(__file__)),'checks':checks,'check_count':len(checks),'rules':RULES}
  print(json.dumps({'event':'self_check_passed','checks':len(checks),'receipt_sha256':save_new(OUT/'self_check.json',value)},ensure_ascii=False),flush=True)
 else:audit()
