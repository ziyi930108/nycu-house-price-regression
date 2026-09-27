"""Fixed target-free city/road fallback for spatial-only geography; no model work."""
from __future__ import annotations
import argparse, collections, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import round6_geo as encoder
import geography_city_road_support as previous

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round39/geography'
DOC=ROOT/'reports/round39_geography_plan.md'
REPORT=ROOT/'reports/round39_geography_report.md'
STAGES={'auxiliary':2021,'development':2022,'confirmation':2023,'final':2024}
MANIFEST_SHA={
 'auxiliary':'b17bd7c3a65f105c76709c2f23cb281be691990052bb802d13f43db4338b1191',
 'development':'4b21f0620aedb8e9527044e9a44a4e5d6f8cae49e71210252c56ba614a5151c6',
 'confirmation':'df5354df87cb3ff52db6f97eafd983e791ea31777458f1f33cdba3bf32a535fc',
 'final':'f8c2b44161ea9588d460cbdbb82f003b63f7d1c4001660a8ad32f42c36b5a6ee'}
R6_PLAN_SHA='606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8'
SOURCE_EXPECTED={
 'src/geography_city_road_support.py':'1ab904ffd435afd80e4abb33d16ff1e16a94d1d2124e23c630e3c07fac6e1a97',
 'src/round6_geo.py':'df1454829b919e2a10ac264dd4e3f597367187174607a6d32abb20f40ef060a0',
 'src/prepare_data.py':'1b2a2c4c947aebdeff5d268300fac6f2f56331a850ffcc6eb6149e7f255ebde6'}
RULES={k:v for k,v in previous.RULES.items() if k not in ['expected_test_rows','expected_candidate_rows','expected_city_road_pairs','other_years_processed']}
RULES.update({'years':STAGES,'original_X_read':False,'original_geography_files_modified':False,
 'new_frames_use':'R8 spatial only; never new tree-model inputs',
 'update_only':'original has_position=0 and both coordinate fields NaN',
 'encoder':'unchanged round6_geo.encode_geo_features on original float64 road record',
 'inferred_district':'provenance only, never original X',
 'R8_2km_count':'geometry-only necessary-condition upper bound, expm1(encoded float32 log extent)<=2000; no groups/residuals/neighborhood eligibility'})
need=previous.need
sha=previous.sha
read=previous.read
frame_sha=encoder.frame_sha
INPUTS={}

def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def bind(path,expected=None):
 path=Path(path);actual=sha(path)
 need(expected is None or actual==expected,'Input identity differs: '+rel(path))
 key=rel(path);need(key not in INPUTS or INPUTS[key]==actual,'Conflicting input identity')
 INPUTS[key]=actual;return actual
def small(path,expected=None,receipt=False):
 bind(path,expected)
 if receipt:
  rp=Path(path).with_suffix('.sha256');words=rp.read_text(encoding='utf-8-sig').split()
  need(len(words) in (1,2) and words[0]==sha(path) and (len(words)==1 or words[1].lstrip('*')==Path(path).name),'Receipt differs')
  bind(rp)
 return read(path)
def lock_write(path,value):
 previous.write(path,value)
 with Path(path).with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')
def index(path,expected=None):
 bind(path,expected);a=np.load(path,allow_pickle=False)
 need(a.dtype==np.dtype('int64') and a.ndim==1 and np.array_equal(a,np.unique(a)),'Index must be sorted unique int64')
 return a
def numeric_sha(a):return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()
def info(path,**extra):return {'path':rel(path),'sha256':sha(path),**extra}

def candidates(p):
 # Same predicate as the immutable 2024 study, with only cohort-count assertions removed.
 need(not {'y','總價元'}.intersection(p.columns),'Target forbidden')
 flags=['is_embedded_header','district_prefix_conflict',*previous.EXCLUDE]
 need(not p[flags+['road_count','road_mentions_count']].isna().any().any(),'Missing parser gate')
 need(not p.is_embedded_header.any(),'Header in canonical geography')
 explicit=p.city.fillna('').ne('');absent=p.district_full.fillna('').eq('')
 blank=p.supplied_district_normalized.fillna('').eq('');city_only=p.supplied_district_is_city.astype(bool)
 admin=explicit & absent & p.district_in_address.fillna('').eq('') & ~p.district_prefix_conflict.astype(bool) & (blank|city_only)
 single=admin & p.road_count.eq(1) & p.road_mentions_count.eq(1) & p.road1_name.fillna('').ne('')
 clean=single.copy()
 for flag in previous.EXCLUDE:clean &= ~p[flag].astype(bool)
 clean &= ~p.road1_base.fillna('').astype(str).str.match(r'^[0-9'+previous.CN+r']+鄰')
 context=pd.Series(False,index=p.index)
 for i,row in p.loc[clean].iterrows():
  cs=json.loads(row.road_candidates_json)
  context.loc[i]=len(cs)==1 and cs[0].get('road_name')==row.road1_name and not cs[0].get('context_uncertain',False) and not cs[0].get('section_unsupported',False)
 clean &= context
 need(not p.loc[clean,'road_query_eligible'].any(),'Missing-district candidate originally eligible')
 return clean.to_numpy(dtype=bool)

def parsed_cache(split,m):
 path=ROOT/f'outputs/geography/address_audit/{split}_parsed.pkl';bind(path,m['source_sha256'][rel(path)])
 p=pd.read_pickle(path);a=m['alignment'][split]
 need(len(p)==a['original_workbook_rows'] and not {'y','總價元'}.intersection(p.columns),'Parsed schema/row count differs')
 need(np.array_equal(p.source_row_index.to_numpy(),np.arange(len(p))) and np.array_equal(p.source_excel_row.to_numpy(),np.arange(len(p))+2),'Original row provenance differs')
 valid=p.existing_feature_cache_row_index.ge(0)
 need(np.array_equal(~valid.to_numpy(),p.is_embedded_header.to_numpy(dtype=bool)),'Header/cache mapping differs')
 p=p.loc[valid].sort_values('existing_feature_cache_row_index').reset_index(drop=True)
 need(np.array_equal(p.existing_feature_cache_row_index.to_numpy(),np.arange(a['header_removed_cache_rows'])),'Cache index not complete')
 p=p.copy();p['source_dataset']=split;p['_candidate']=candidates(p)
 return p

def scan_index(m,wanted):
 ap=ROOT/m['geo_index_audit_path'];audit=small(ap,m['geo_index_audit_sha256'],True)
 path=ap.with_name('road_index.jsonl');expected=m['source_sha256'][rel(path)]
 need(audit['output_sha256']['road_index.jsonl']==expected,'Index audit/manifest differ')
 allkeys=set();records={pair:[] for pair in wanted};digest=hashlib.sha256();stat=path.stat();rows=0
 with path.open('rb') as f:
  for line in f:
   digest.update(line);r=json.loads(line);rows+=1;key=r['key']
   need(key=='|'.join([r['city'],r['district'],r['road_name']]) and key not in allkeys,'Malformed/duplicate full original road key')
   allkeys.add(key);pair=(r['city'],r['road_name'])
   if pair in records:
    meta=previous.inspect_record(r)
    records[pair].append({**meta,**{k:r[k] for k in ['latitude','longitude','road_length_m']}})
 need(rows==audit['road_keys'] and digest.hexdigest()==expected,'Index full-stream SHA/count differs')
 need(path.stat().st_size==stat.st_size and path.stat().st_mtime_ns==stat.st_mtime_ns,'Index changed during read')
 INPUTS[rel(path)]=expected
 return records,{'path':rel(path),'sha256':expected,'bytes':stat.st_size,'rows':rows,'full_streams':1,'all_original_keys_unique':True,'quality_filtered_before_uniqueness':False}

def capacity(g):
 e=np.expm1(g.geo_log_extent.to_numpy(dtype='float64'))
 return g.geo_has_position.eq(1).to_numpy() & np.isfinite(e) & (e>=0) & (e<=2000)

def prepare_partition(stage,which,p,original,records,original_path,original_sha):
 p=p.reset_index(drop=True);original=original.reset_index(drop=True);encoder.check_geo(original)
 need(len(p)==len(original) and original.index.equals(p.index),'Parsed/geo partition alignment differs')
 before=frame_sha(original);updated=original.copy(deep=True);take=p._candidate.to_numpy(dtype=bool)
 pairs=collections.Counter(zip(p.loc[take,'city'],p.loc[take,'road1_name']))
 classified={pair:previous.classify(pair,count,records[pair]) for pair,count in pairs.items()}
 for v in classified.values():
  if v['status']=='zero':v['reason']='exact_city_full_road_not_in_annual_index'
 provenance=p[['source_dataset','source_row_index','source_excel_row','existing_feature_cache_row_index']].copy()
 provenance['partition_row']=np.arange(len(p),dtype='int64');provenance['canonical_position']=p.canonical_position.to_numpy(copy=True);provenance['candidate']=take
 provenance['city']=p.city.fillna('');provenance['road_name']=p.road1_name.fillna('')
 provenance['original_district']=p.district_full.fillna('');provenance['inferred_district']='';provenance['selected_road_key']=''
 provenance['status']='not_candidate';provenance['reason']='original_predicate_rejected';provenance['matching_key_count']=0
 provenance['matching_records_json']='[]';provenance['original_has_position']=original.geo_has_position.to_numpy(copy=True)
 provenance['added']=False;to_encode=[];positions=[]
 missing=original.geo_has_position.eq(0) & original[['geo_latitude','geo_longitude']].isna().all(axis=1)
 for i in np.flatnonzero(take):
  v=classified[(p.at[i,'city'],p.at[i,'road1_name'])]
  provenance.at[i,'status']=v['status'];provenance.at[i,'reason']=v['reason'];provenance.at[i,'matching_key_count']=v['matching_original_key_count']
  provenance.at[i,'matching_records_json']=json.dumps(v['records'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
  if v['unique_usable']:
   r=records[(p.at[i,'city'],p.at[i,'road1_name'])][0]
   provenance.at[i,'inferred_district']=r['district'];provenance.at[i,'selected_road_key']=r['key']
   if bool(missing.iloc[i]):
    positions.append(int(i));to_encode.append({'geo_match_status':'road_level','geo_latitude':r['latitude'],'geo_longitude':r['longitude'],
     'geo_road_length_m':r['road_length_m'],'geo_extent_diagonal_m':r['extent_diagonal_m'],'geo_nearby_fragment_grouping':r['nearby_fragment_grouping']})
    provenance.at[i,'added']=True
   else:provenance.at[i,'reason']='unique_usable_but_original_position_preserved'
 added=np.asarray(positions,dtype='int64')
 if len(added):
  encoded=encoder.encode_geo_features(pd.DataFrame(to_encode,index=added))
  for c in original.columns:updated.loc[added,c]=encoded[c].astype(original[c].dtype).array
 keep=~provenance.added.to_numpy(dtype=bool)
 pd.testing.assert_frame_equal(updated.loc[keep],original.loc[keep],check_exact=True)
 for c in encoder.NUMERIC:need(updated.loc[keep,c].to_numpy().tobytes()==original.loc[keep,c].to_numpy().tobytes(),'Original numeric bytes changed')
 need(updated.index.equals(original.index) and updated.dtypes.equals(original.dtypes) and list(updated)==list(original),'Original schema/index changed')
 need(frame_sha(original)==before,'Input geo mutated');encoder.check_geo(updated)
 stats=previous.totals(list(classified.values()))
 stats.update({'rows':len(p),'added_rows':len(added),'original_position_rows':int(original.geo_has_position.sum()),'updated_position_rows':int(updated.geo_has_position.sum()),
  'original_extent_lte_2km_rows':int(capacity(original).sum()),'updated_extent_lte_2km_rows':int(capacity(updated).sum()),
  'added_extent_lte_2km_rows':int(capacity(updated)[added].sum()),'R8_actual_eligibility_not_computed':True,
  'all_unmodified_rows_exact':True,'all_existing_position_rows_exact':True,'input_geo_unmutated':True,'dtype_order_index_preserved':True,
  'by_city':{city:previous.totals([v for v in classified.values() if v['city']==city]) for city in sorted({v['city'] for v in classified.values()})}})
 for city,v in stats['by_city'].items():
  mask=provenance.city.eq(city).to_numpy();v['added_rows']=int((mask&provenance.added.to_numpy()).sum());v['added_extent_lte_2km_rows']=int((mask&provenance.added.to_numpy()&capacity(updated)).sum())
 folder=OUT/stage;folder.mkdir(exist_ok=True)
 path=folder/f'{which}.pkl'
 with path.open('xb') as f:updated.to_pickle(f)
 pp=folder/f'{which}_provenance.pkl'
 with pp.open('xb') as f:provenance.to_pickle(f)
 ip=folder/f'{which}_added_positions.npy'
 with ip.open('xb') as f:np.save(f,added,allow_pickle=False)
 return stats,info(path,rows=len(updated),frame_sha256=frame_sha(updated),dtypes={c:str(updated[c].dtype) for c in updated},original_geo_path=rel(original_path),original_geo_sha256=original_sha,original_partition_frame_sha256=before),info(pp,rows=len(provenance)),info(ip,rows=len(added),content_sha256=numeric_sha(added))

def render(summary):
 lines=['# R39：缺區單路的同年度 spatial 專用地理補充','',
  '固定規則先寫後跑；只讀無目標parsed、保存位置索引與原八欄geo／道路索引。沒有讀價格、X87、模型或誤差，也沒有重新去重、改原地理檔、發布CSV或更新推薦。','',
  '| 年度／期別 | 集合 | 列數 | 候選／鍵 | 零鍵 | 多鍵 | 唯一原品質可用 | 真正新增 | 新增且extent≤2km |',
  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 for stage,record in summary['stages'].items():
  for which,v in record['counts'].items():lines.append(f"| {record['snapshot_year']}/{stage} | {which} | {v['rows']:,} | {v['candidate_rows']:,}/{v['city_road_pairs']} | {v['zero_rows']:,} | {v['multiple_rows']:,} | {v['unique_usable_rows']:,} | {v['added_rows']:,} | {v['added_extent_lte_2km_rows']:,} |")
 lines += ['', '原有座標及全部未新增列的八欄數值（含float32位元）、類別、dtype、順序與index均保持；推定行政區只在旁表。原 .02 grid 由原encoder使用道路原float64座標產生，沒有新grid或尾文規則。四年各掃一次全部品質索引，先排歧再品質，沒有以排除低品質道路假造唯一。',
  '2km欄只是既有R8的單端extent必要條件上限，不代表已滿足同city/style/use、五個建案、距離、時間、面積、殘差或弱source等條件；本次不讀那些價格reference。道路代表點不是門牌位置，索引缺漏可能造成表面唯一，字串parser也不是完整道路實體認證。',
  'train的529筆及final接合的28筆去重沿原保存位置；未讀價格重算。final移除的28筆在valid段。2024 query重現舊710/72候選與327/47品質可用；各年用各自預測時點前OSM，不借2024路網補早年。','',
  f"開始：{summary['started_at_utc']}；結束：{summary['completed_at_utc']}；程式內耗時{summary['elapsed_seconds']:.3f}秒。來源、各期時間戳、城市數及输出身份見summary與stage manifests；旁表保留零鍵／多鍵及品質原因。",'',
  '[事前規約](round39_geography_plan.md) · [完整summary](../outputs/round39/geography/summary.json) · [程式](../src/round39_prepare_geo.py) · [舊2024盤點](geography_city_road_support_2024.md)','']
 return '\n'.join(lines)

def run():
 started=time.perf_counter();need(not OUT.exists() and not REPORT.exists(),'No overwrite or repeat')
 OUT.mkdir(parents=True);lock_write(OUT/'run_claim.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'rules':RULES})
 sources={n:bind(ROOT/n,h) for n,h in SOURCE_EXPECTED.items()}
 sources[rel(Path(__file__))]=bind(Path(__file__));sources[rel(DOC)]=bind(DOC)
 r6=small(ROOT/'outputs/round6/candidate_plan.json',R6_PLAN_SHA,True)
 manifests={stage:small(ROOT/f'outputs/round6/geography/{stage}/manifest.json',h,True) for stage,h in MANIFEST_SHA.items()}
 first=manifests['final'];cache={split:parsed_cache(split,first) for split in ['train','valid','test']}
 summary={'status':'completed_pure_X_spatial_geography','started_at_utc':read(OUT/'run_claim.json')['started_at_utc'],'rules':RULES,'stages':{},
  'price_data_read':False,'models_loaded':False,'original_X_read':False,'original_geography_modified':False,'dedup_recomputed':False,'inherited_price_containing_dedup_positions_used':True}
 for stage,year in STAGES.items():
  m=manifests[stage];need(m['stage']==stage and m['snapshot_year']==year,'Annual geography contract differs')
  expected_index=f'outputs/geography/road_index_{year}_'+('parent_verified' if stage=='development' else 'annual_v3')+'/audit.json'
  need(m['geo_index_audit_path']==expected_index,'Wrong annual road index')
  need(pd.Timestamp(m['replication_timestamp'])<pd.Timestamp(m['forecast_origin_utc']),'Map after forecast origin')
  for split in cache:need(m['source_sha256'][f'outputs/geography/address_audit/{split}_parsed.pkl']==first['source_sha256'][f'outputs/geography/address_audit/{split}_parsed.pkl'],'Parsed source differs by year')
  maps={}
  for split,bound,length in [('train',455784,455255),('valid',11994,11994),('test',47073,47073),('full',467249,467221)]:
   path=ROOT/f'outputs/round6/geography/{stage}/{split}_source_cache_indices.npy';a=index(path,m['index_files_sha256'][rel(path)])
   need(len(a)==length and (a>=0).all() and (a<bound).all(),'Source map length/range differs');maps[split]=a
  fi_path=ROOT/f'outputs/round6/models/{stage}/geography_catboost/fit_indices.npy';qi_path=fi_path.with_name('query_indices.npy');fi=index(fi_path);qi=index(qi_path)
  train=cache['train'].iloc[maps['train']].reset_index(drop=True)
  if stage=='final':
   joined=pd.concat([train,cache['valid'].iloc[maps['valid']]],ignore_index=True)
   removed=np.setdiff1d(np.arange(len(joined)),maps['full']);need(len(removed)==28 and (removed>=455255).all(),'Final overlap mapping differs')
   partitions={'fit':joined.iloc[maps['full']].reset_index(drop=True),'query':cache['test'].iloc[maps['test']].reset_index(drop=True)}
   need(np.array_equal(fi,np.arange(467221)) and np.array_equal(qi,np.arange(47073)),'Final indices differ')
   geo_roles={'fit':'full','query':'test'}
  else:
   contract=r6['split_contract'][stage]
   for label,a in [('fit',fi),('query',qi)]:need(len(a)==contract[label+'_rows'] and numeric_sha(a)==contract[label+'_indices_content_sha256'],'Frozen split indices differ')
   need(not np.intersect1d(fi,qi).size and min(fi.min(),qi.min())>=0 and max(fi.max(),qi.max())<len(train),'Split overlap/range invalid')
   baseline=ROOT/('outputs/round6/baseline/auxiliary' if stage=='auxiliary' else f'outputs/round3/baseline/{stage}')
   need(np.array_equal(fi,index(baseline/'fit_indices.npy')) and np.array_equal(qi,index(baseline/'check_indices.npy')),'Original baseline indices differ')
   partitions={'fit':train.iloc[fi].reset_index(drop=True),'query':train.iloc[qi].reset_index(drop=True)};geo_roles={'fit':'train','query':'train'}
  for which,p in partitions.items():p['canonical_position']=fi if which=='fit' else qi
  wanted=set()
  for p in partitions.values():wanted.update(zip(p.loc[p._candidate,'city'],p.loc[p._candidate,'road1_name']))
  records,scan=scan_index(m,wanted);geos={};counts={};features={};provenance={};added_positions={}
  for which,p in partitions.items():
   role=geo_roles[which];item=m['features'][role];path=ROOT/item['path'];digest=bind(path,item['sha256'])
   if role not in geos:
    geos[role]=pd.read_pickle(path);need(len(geos[role])==item['rows'],'Original geo rows differ')
   g=geos[role] if stage=='final' else geos[role].iloc[fi if which=='fit' else qi].reset_index(drop=True)
   counts[which],features[which],provenance[which],added_positions[which]=prepare_partition(stage,which,p,g,records,path,digest)
  if stage=='final':
   q=counts['query'];need((q['candidate_rows'],q['city_road_pairs'],q['unique_usable_rows'],q['unique_usable_pairs'],q['zero_rows'],q['multiple_rows'],q['unique_unusable_rows'])==(710,72,327,47,9,324,50),'2024 fixed support not reproduced')
  stage_value={'round':39,'status':'prepared_spatial_only','stage':stage,'snapshot_year':year,'replication_timestamp':m['replication_timestamp'],'forecast_origin_utc':m['forecast_origin_utc'],
   'rules':RULES,'encoder_feature_rules':encoder.FEATURE_RULES,'source_sha256':sources,'input_sha256':dict(INPUTS),'original_manifest_sha256':MANIFEST_SHA[stage],
   'inherited_geography_source_sha256':m['source_sha256'],'inherited_sources_not_all_rehashed':True,'PBF_rehashed':False,
   'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'index_scan':scan,
   'indices':{'fit':info(fi_path,rows=len(fi),content_sha256=numeric_sha(fi)),'query':info(qi_path,rows=len(qi),content_sha256=numeric_sha(qi))},
   'features':features,'provenance':provenance,'added_positions':added_positions,'counts':counts,'spatial_only':True}
  manifest_path=OUT/stage/'manifest.json';lock_write(manifest_path,stage_value)
  summary['stages'][stage]={'snapshot_year':year,'counts':counts,'manifest':info(manifest_path),'replication_timestamp':m['replication_timestamp'],'forecast_origin_utc':m['forecast_origin_utc']}
  print(json.dumps({'event':'stage_prepared','stage':stage,'counts':counts,'manifest_sha256':sha(manifest_path)},ensure_ascii=False),flush=True)
 # Recheck every actually-read source/map/parsed/geo; each large road index was already
 # streamed exactly once with SHA and stat checks, so do not scan it again.
 for name,digest in INPUTS.items():
  if not name.endswith('/road_index.jsonl'):need(sha(ROOT/name)==digest,'Read input changed: '+name)
 summary.update({'source_sha256':sources,'input_sha256':dict(INPUTS),'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,
  'original_inputs_unchanged':True,'index_full_streams':4})
 with REPORT.open('x',encoding='utf-8') as f:f.write(render(summary))
 summary['report']=info(REPORT);lock_write(OUT/'summary.json',summary)
 print(json.dumps({'status':summary['status'],'summary_sha256':sha(OUT/'summary.json'),'report_sha256':sha(REPORT),'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True);parser.parse_args();run()
