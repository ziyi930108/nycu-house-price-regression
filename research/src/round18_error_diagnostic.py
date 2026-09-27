from pathlib import Path
from zipfile import ZipFile
from datetime import datetime,timezone
import sys,json,hashlib,time
import numpy as np
import pandas as pd
root=Path(r'C:\NYCU_ML');sys.path.insert(0,str(root/'src'))
from round10_remarks_inventory import sheet_cells,decode_selected
from house_features import _date_parts
out=root/'outputs/round18_research/error_diagnostic.json'
assert not out.exists()
started=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
sources={}
def capture(name,expected=None):
 p=root/name;h=sha(p)
 assert expected is None or h==expected,(name,'hash mismatch')
 sources[name]=h;return p
# Fixed interpretable one-way bands are specified before any target/prediction decoding.
def positive_bands(series,edges):
 a=pd.to_numeric(series,errors='coerce').to_numpy(dtype='float64');labels=np.full(len(a),'missing_or_nonfinite',dtype=object)
 labels[np.isfinite(a)&(a<=0)]='nonpositive'
 order=['missing_or_nonfinite','nonpositive']
 for lo,hi in zip([0.]+edges,edges+[float('inf')]):
  label=f'({lo:g},{hi:g}]' if np.isfinite(hi) else f'({lo:g},inf)'
  labels[np.isfinite(a)&(a>lo)&(a<=hi)]=label;order.append(label)
 return labels,order
band_specs={'building_area_m2':[30,60,100,150,250,500], 'land_area_m2':[30,100,300,1000],
 'floor_mean':['missing','negative','zero','(0,3]','(3,6]','(6,12]','(12,20]','(20,inf)'],
 'total_floors':[3,6,12,20,30],
 'parking_area_m2':['missing','negative','zero','(0,15]','(15,30]','(30,60]','(60,inf)'],
 'relative_floor':['missing','negative','[0,1/3]','(1/3,2/3]','(2/3,1]','(1,inf)'],
 'geo_extent_m':[250,1000,2000]}
resultpath=capture('outputs/round6/evaluation/development/results.json');saved=read(resultpath)
assert (resultpath.with_suffix('.sha256')).read_text().strip()==sha(resultpath)
capture('outputs/round6/evaluation/development/results.sha256')
manifestpath=capture('outputs/round6/geography/development/manifest.json');manifest=read(manifestpath)
known={**manifest['source_sha256'],**manifest['index_files_sha256'],**saved['input_artifact_sha256'],**saved['output_artifact_sha256']}
indexpath=capture('outputs/round6/geography/development/train_source_cache_indices.npy',known['outputs/round6/geography/development/train_source_cache_indices.npy'])
checkname='outputs/round3/baseline/development/check_indices.npy'
checkpath=capture(checkname,known.get(checkname));fi_name='outputs/round3/baseline/development/fit_indices.npy';fi_path=capture(fi_name,known.get(fi_name))
cachepath=capture('data_cache/train_features.pkl',known['data_cache/train_features.pkl']);meta_path=capture('data_cache/train.json',known.get('data_cache/train.json'))
meta=read(meta_path);workbook=Path(meta['source']).as_posix();bookpath=capture(workbook,meta['source_sha256'])
for name in ['src/house_features.py','src/model_common.py','src/round6_prepare_geo.py','src/round10_remarks_inventory.py']:capture(name)
cache_map=np.load(indexpath,allow_pickle=False);query=np.load(checkpath,allow_pickle=False);fit=np.load(fi_path,allow_pickle=False)
assert len(cache_map)==455255 and len(query)==81234 and len(fit)==240316 and np.unique(cache_map).size==len(cache_map) and np.all(np.diff(cache_map)>0) and not np.intersect1d(query,fit).size
features=pd.read_pickle(cachepath)
assert len(features)==455784 and features.shape[1]==87 and not {'y','target','總價元','編號'}&set(features)
X=features.iloc[cache_map].reset_index(drop=True).iloc[query].copy();del features
start=(pd.Timestamp('2022-08-11')-pd.Timestamp('2000-01-01')).days;end=(pd.Timestamp('2023-08-10')-pd.Timestamp('2000-01-01')).days
assert X.trade_day_index.between(start,end).all()
# Reuse the frozen raw-cache map. Reconstruct workbook row numbers from saved removed header rows.
headers=sorted(int(v['excel_row']) for v in meta['rejected_rows'])
assert headers==[151490,225665,437510]
all_excel=np.setdiff1d(np.arange(2,meta['raw_rows']+2,dtype=np.int64),np.asarray(headers,dtype=np.int64),assume_unique=True)
assert len(all_excel)==meta['clean_rows']==455784
excel=all_excel[cache_map[query]];assert np.unique(excel).size==81234 and not set(excel)&set(headers)
with ZipFile(bookpath) as z:
 # Decode only selected B(total price) and J(date) cells. Other rows' cell bytes are never parsed/decoded.
 tokens=sheet_cells(z,columns={'B','J'},eligible=set(map(int,excel)))
 assert set(tokens)==set(map(int,excel)) and all(set(v)=={'B','J'} for v in tokens.values())
 values,shared_count=decode_selected(z,tokens)
 y=np.array([float(values[int(r)]['B']) for r in excel],dtype='float64')
 dates=_date_parts(pd.Series([values[int(r)]['J'] for r in excel]))
assert np.isfinite(y).all() and (y>=0).all()
assert np.array_equal(((dates-pd.Timestamp('2000-01-01')).dt.days).to_numpy(),X.trade_day_index.to_numpy())
del tokens,values,dates
predname='outputs/round6/evaluation/development/geography_blend.npy';predpath=capture(predname,saved['output_artifact_sha256'][predname]);p=np.load(predpath,allow_pickle=False)
assert p.shape==y.shape and p.dtype==np.dtype('float64') and np.isfinite(p).all()
error=np.abs(y-p);total_error=float(error.sum());mae=float(error.mean())
assert mae==saved['metrics']['geography_blend']['overall_mae']
geospec=manifest['features']['train'];geopath=capture(geospec['path'],geospec['sha256']);all_geo=pd.read_pickle(geopath)
assert len(all_geo)==len(cache_map) and all_geo.shape[1]==8
geo=all_geo.iloc[query].copy();del all_geo
assert geo.index.equals(X.index)
series={};orders={}
for c in ['building_type','transaction_type','main_use']:
 v=X[c].fillna('__MISSING__').astype(str).str.strip().replace({'':'__MISSING__','nan':'__MISSING__','None':'__MISSING__'})
 series[c]=v.to_numpy();orders[c]=sorted(set(series[c]))
for outname,column,edges in [('building_area_m2','building_area',band_specs['building_area_m2']),('land_area_m2','land_area',band_specs['land_area_m2']),('total_floors','total_floors',band_specs['total_floors'])]:series[outname],orders[outname]=positive_bands(X[column],edges)
for outname,column,edges in [('floor_mean','floor_mean',[3,6,12,20]),('parking_area_m2','parking_area',[15,30,60])]:
 a=X[column].to_numpy(dtype='float64');v,order=positive_bands(X[column],edges);v[np.isfinite(a)&(a<0)]='negative';v[np.isfinite(a)&(a==0)]='zero';order=[order[0],'negative','zero']+order[2:];series[outname],orders[outname]=v,order
r=X.relative_floor.to_numpy(dtype='float64');labels=np.full(len(r),'missing_or_nonfinite',object)
for label,mask in [('negative',r<0),('[0,1/3]',(r>=0)&(r<=1/3)),('(1/3,2/3]',(r>1/3)&(r<=2/3)),('(2/3,1]',(r>2/3)&(r<=1)),('(1,inf)',r>1)]:labels[np.isfinite(r)&mask]=label
series['relative_floor']=labels;orders['relative_floor']=['missing_or_nonfinite','negative','[0,1/3]','(1/3,2/3]','(2/3,1]','(1,inf)']
p_missing=X.parking_type.fillna('__MISSING__').astype(str).isin(['__MISSING__','','nan','None'])
series['parking_type_missingness']=np.where(p_missing,'missing_type','recorded_type');orders['parking_type_missingness']=['missing_type','recorded_type']
series['geo_match_status']=geo.geo_match_status.fillna('__MISSING__').astype(str).to_numpy();orders['geo_match_status']=sorted(set(series['geo_match_status']))
extent=np.expm1(geo.geo_log_extent.to_numpy(dtype='float64'));glabels=np.full(len(extent),'unavailable',object)
usable=(geo.geo_has_position.to_numpy()==1)&np.isfinite(extent)&(extent>=0)
for label,mask in [('0..250m',extent<=250),('(250,1000]m',(extent>250)&(extent<=1000)),('(1000,2000]m',(extent>1000)&(extent<=2000)),('>2000m',extent>2000)]:glabels[usable&mask]=label
series['geo_extent_m']=glabels;orders['geo_extent_m']=['unavailable','0..250m','(250,1000]m','(1000,2000]m','>2000m']
tables={}
for name,v in series.items():
 rows=[]
 for label in orders[name]:
  mask=v==label;n=int(mask.sum());ae=float(error[mask].sum())
  rows.append({'group':label,'rows':n,'row_share':n/len(y),'mae_twd':float(error[mask].mean()) if n else None,'absolute_error_sum_twd':ae,'absolute_error_share':ae/total_error,'mae_to_overall_ratio':float(error[mask].mean())/mae if n else None})
 assert sum(z['rows'] for z in rows)==len(y)
 assert abs(sum(z['absolute_error_sum_twd'] for z in rows)-total_error)<1e-3
 tables[name]=rows
# Publish aggregate-only evidence once, with no individual prices/errors or address identifiers.
value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'exploratory_diagnostic_not_candidate','stage':'development','historically_untouched':False,'query_dates':['2022-08-11','2023-08-10'],'rows':len(y),'fit_rows_from_existing_indices':len(fit),'baseline':'R6 geography_blend / released v3 same-window predictions','overall_mae_twd':mae,'absolute_error_sum_twd':total_error,'saved_overall_mae_exact_match':True,'source_sha256':sources,'bands_fixed_before_target_and_prediction_decode':band_specs,'grouping':'12 one-way feature axes; no cross-products, rank search or adaptive band thresholds','tables':tables,'alignment':{'canonical_train_rows':len(cache_map),'header_cleaned_cache_rows':455784,'removed_header_excel_rows':headers,'raw_full_row_dedup_removed':529,'frozen_map_reused_without_recomputing_target_dedup':True,'selected_workbook_rows':len(excel),'selected_original_date_matches_feature_date_all_rows':True,'query_indices_content_sha256':hashlib.sha256(query.astype('<i8').tobytes()).hexdigest(),'source_cache_rows_content_sha256':hashlib.sha256(cache_map[query].astype('<i8').tobytes()).hexdigest(),'source_excel_rows_content_sha256':hashlib.sha256(excel.astype('<i8').tobytes()).hexdigest()},'read_scope':{'target_cells_decoded':len(y),'target_source':'only selected development workbook B cells','selected_date_cells_decoded':len(y),'shared_string_entries_selected':shared_count,'raw_price_pickle_deserialized':False,'common_load_data_called':False,'confirmation_auxiliary_valid_test_targets_decoded':False,'workbook_bytes_streamed_beyond_selected_rows':True,'outside_selected_row_column_cell_XML_values_decoded':False,'new_predictions_generated':False,'model_loaded_or_fitted':False,'individual_records_published':False},'limitations':['Development labels were repeatedly used in historical selection; exploratory slices are not an independent holdout.','Single-axis slices overlap across tables; do not add their error shares.','TWD MAE rises with price scale; concentration is not proof of a feature defect or a useful model change.','Floor parser and zero/missing parking semantics are inherited unchanged; no repair was applied.','Road geometry extent is not a calibrated location error or exact house position.'],'elapsed_seconds':time.monotonic()-started}
for name,h in sources.items():assert sha(root/name)==h,name
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':'written','path':str(out),'sha256':sha(out),'rows':len(y),'overall_mae_twd':mae,'axes':len(tables),'table_rows':sum(map(len,tables.values())),'source_count':len(sources),'elapsed_seconds':value['elapsed_seconds']},ensure_ascii=False),flush=True)
for name,rows in tables.items():print(name,json.dumps([r for r in rows if r['rows']],ensure_ascii=False))