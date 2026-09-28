"""R93 fit-only Pool distribution audit; no fit, model, labels or query analysis.

Whole train-feature pickle must be deserialized; analysis uses canonical fit rows
only. Within prepared NPZ only train_sample_weight is deserialized.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time,traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
import round5_project_mask as identity
import round79_recency_unit as old
LOCKS={
 'outputs/round79/plan.json':'7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f',
 'reports/round93_plan.md':'ac3ced3aa70ea3bb2548e4f4d4476da4c5b2e9ba2bd4795a5a673ee97f884054',
 'outputs/round93_fit_audit/audit.py':'28363a05bb35aa43adf286a7d3229adbda5953a0e5c170126d9e950038e4a19e',
 'outputs/round93_fit_audit/results.json':'0f9aa4b680fc2522b9491c988f47983f49a32c7c54e4cc8f1040f1108d06b8d4'}
sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
values_sha=lambda v:hashlib.sha256(np.asarray(v,dtype='<f8').tobytes(order='C')).hexdigest()
int_sha=lambda v:hashlib.sha256(np.asarray(v,dtype='<i8').tobytes(order='C')).hexdigest()

def concentration(v):
 v=np.asarray(v,dtype=np.float64)
 if not len(v):return {'rows':0,'weight_sum':0.,'maximum_single_row_share':None,'weight_effective_rows':None}
 total=float(v.sum(dtype=np.float64));squares=float(np.dot(v,v))
 assert np.isfinite(v).all() and (v>0).all() and np.isfinite(squares)
 return {'rows':len(v),'weight_sum':total,'maximum_single_row_share':float(v.max()/total),
         'weight_effective_rows':total*total/squares}

def group_record(mask,w0,w1,total0,total1,n):
 count=int(mask.sum());c0,c1=concentration(w0[mask]),concentration(w1[mask])
 return {'rows':count,'raw_row_fraction':count/n,'w0_mass':c0['weight_sum'],'w1_mass':c1['weight_sum'],
  'w0_mass_fraction':c0['weight_sum']/total0,'w1_mass_fraction':c1['weight_sum']/total1,
  'pool_mass_fraction_change':c1['weight_sum']/total1-c0['weight_sum']/total0,
  'relative_mass_multiplier':c1['weight_sum']/c0['weight_sum'] if count else None,
  'within_group_w0_concentration':c0,'within_group_w1_concentration':c1}

def run():
 tick=time.monotonic()
 sources={**old.source_hashes(),'src/round5_project_mask.py':sha('src/round5_project_mask.py'),
  'outputs/round93_fit_distribution/audit.py':sha(Path(__file__).relative_to(ROOT))}
 for p,h in LOCKS.items():assert sha(p)==h,p
 plan=json.loads((ROOT/'outputs/round79/plan.json').read_text(encoding='utf-8'))
 prior=json.loads((ROOT/'outputs/round93_fit_audit/results.json').read_text(encoding='utf-8'))
 for p,h in prior['source_sha256'].items():assert sha(p)==h,p
 inputs={**LOCKS}
 paths=('data_cache/train_features.pkl','outputs/round6/geography/development/train_source_cache_indices.npy')
 for p in paths:
  inputs[p]=plan['input_sha256'][p];assert sha(p)==inputs[p],p
 Xraw=pd.read_pickle(ROOT/paths[0]);ids=np.load(ROOT/paths[1],allow_pickle=False)
 assert len(Xraw)==455784 and len(Xraw.columns)==87 and ids.shape==(455255,) and ids.dtype==np.int64
 assert not {'總價元','target','y','y_true','y_pred'}.intersection(Xraw.columns)
 assert len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<len(Xraw)
 records={}
 for stage in ('development','confirmation','auxiliary'):
  b=plan['bindings'][stage];spec=plan['split_contract'][stage]
  fitpath=b['unit']['fit_indices'];inputs[fitpath]=plan['input_sha256'][fitpath]
  assert sha(fitpath)==inputs[fitpath]
  fi=np.load(ROOT/fitpath,allow_pickle=False)
  assert fi.dtype==np.int64 and fi.shape==(spec['fit_rows'],) and (np.diff(fi)>0).all()
  assert int_sha(fi)==spec['fit_indices_content_sha256']
  X=Xraw.iloc[ids[fi]].reset_index(drop=True);fit_hash=old.base.frame_sha(X)
  assert fit_hash==b['R19_state']['geo_state']['fit_original_feature_values_sha256']
  cutoff=(pd.Timestamp(plan['splits'][stage][0])-pd.Timestamp('2000-01-01')).days
  days=X.trade_day_index.to_numpy(dtype=np.float64)
  assert np.isfinite(days).all() and (days<cutoff).all()
  for col in ('address_city','address_district','project_name'):
   assert not identity._normal(X[col]).str.contains('|',regex=False).any()
  keys,missing=identity.project_keys(X);frequency=keys[~missing].value_counts()
  counts=keys.map(frequency).where(~missing,1).to_numpy(dtype=np.int64)
  q=1/np.sqrt(counts.astype(np.float64))
  prep_path=b['prepared'];inputs[prep_path]=plan['input_sha256'][prep_path]
  assert sha(prep_path)==inputs[prep_path]
  with np.load(ROOT/prep_path,allow_pickle=False) as z:dates=z['train_sample_weight'].copy()
  dates=old._date_values(dates,X)
  date_sha=hashlib.sha256(dates.astype('<f4').tobytes(order='C')).hexdigest()
  assert date_sha==b['preparation']['train_sample_weight_sha256']
  dc=b['preparation']['date_weight_summary_float64']
  for col,key in [('trade_day_index','date_values_sha256'),('trade_date_invalid','invalid_flags_sha256'),
                  ('trade_date_out_of_range','out_of_range_flags_sha256')]:
   assert values_sha(X[col].to_numpy(dtype=np.float64))==dc[key]
  w0=old.original.sample_weights(X)*dates.astype(np.float64)
  c=float(w0.sum(dtype=np.float64)/(w0*q).sum(dtype=np.float64));w1=w0*q*c
  assert np.isclose(w0.sum(),w1.sum(),rtol=2e-15,atol=0)
  known=prior['stages'][stage]
  assert int_sha(counts)==known['counts_values_sha256']
  assert values_sha(q)==known['q_values_sha256'] and values_sha(w1)==known['weights_values_sha256']
  assert c==known['normalization_c']
  years=pd.to_numeric(X.trade_year,errors='coerce').to_numpy(dtype=np.float64)
  dateyears=pd.to_datetime(days,unit='D',origin='2000-01-01').year.to_numpy()
  assert np.isfinite(years).all() and np.array_equal(years,dateyears);years=years.astype(np.int64)
  masks={'named':~missing.to_numpy(dtype=bool),'missing_name':missing.to_numpy(dtype=bool)}
  t0,t1,n=float(w0.sum()),float(w1.sum()),len(X)
  statuses={name:group_record(mask,w0,w1,t0,t1,n) for name,mask in masks.items()}
  byyear=[];joint=[]
  for year in sorted(np.unique(years).tolist()):
   mask=years==year
   byyear.append({'trade_year':year,**group_record(mask,w0,w1,t0,t1,n)})
   for name,sub in masks.items():
    joint.append({'trade_year':year,'name_status':name,**group_record(mask&sub,w0,w1,t0,t1,n)})
  for rows in (list(statuses.values()),byyear,joint):
   assert sum(z['rows'] for z in rows)==n
   for k in ('raw_row_fraction','w0_mass_fraction','w1_mass_fraction'):
    assert np.isclose(sum(z[k] for z in rows),1.,rtol=2e-15,atol=0)
  records[stage]={'fit_rows':n,'fit_cutoff_exclusive':plan['splits'][stage][0],
   'normalization_c':c,'named_project_count':len(frequency),'by_name_status':statuses,
   'by_trade_year':byyear,'by_trade_year_and_name_status':joint,
   'w0_concentration':concentration(w0),'w1_concentration':concentration(w1),
   'trade_year_mass_total_variation':.5*sum(abs(z['pool_mass_fraction_change']) for z in byyear),
   'fit_feature_values_sha256':fit_hash,'fit_indices_values_sha256':int_sha(fi),
   'date_weights_values_sha256':date_sha,'w0_values_sha256':values_sha(w0),'w1_values_sha256':values_sha(w1),
   'counts_values_sha256':int_sha(counts),'q_values_sha256':values_sha(q),'fixed_R93_fit_audit_replay_exact':True}
 for p,h in {**inputs,**sources}.items():assert sha(p)==h,p
 return {'status':'passed','exit_code':0,'stages':records,'source_sha256':sources,'input_sha256':inputs,
  'formula':'w0=(A/100)*float64(saved_date32);q=named_count**(-0.5),missing=1;c=sum(w0)/sum(w0*q);w1=w0*q*c',
  'audit_scope':{'new_fits':0,'model_loads':0,'model_predicts':0,'scores':0,'price_arrays_deserialized':False,
   'raw_price_table_opened':False,'test_accessed':False,'query_subset_materialized_or_analyzed':False,
   'query_arrays_accessed':False,'full_training_X_pickle_deserialized_before_fit_selection':True,
   'prepared_NPZ_member_arrays_accessed':['train_sample_weight'],'NPZ_container_bytes_hashed_for_source_identity':True,
   'row_price_vectors_or_project_names_exported':False,'new_strength_or_threshold_searched':False,
   'hypothesis_or_plan_changed':False},
  'interpretation':'weight_effective_rows=(sum(w))^2/sum(w^2) describes concentration, not statistically independent samples; no performance inference',
  'elapsed_seconds':time.monotonic()-tick}

if __name__=='__main__':
 out=Path(__file__).with_name('results.json')
 if out.exists():raise FileExistsError('Preserve prior result: '+str(out))
 started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
 try:result=run()
 except Exception:
  result={'status':'failed','exit_code':1,'failure':traceback.format_exc(),'elapsed_seconds':time.monotonic()-tick}
 result.update(started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat())
 with out.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 if result['exit_code']:print(result['failure']);sys.exit(1)
 print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],
  'stages':{k:{'fit_rows':v['fit_rows'],'by_name_status':v['by_name_status'],
  'w0_concentration':v['w0_concentration'],'w1_concentration':v['w1_concentration'],
  'trade_year_mass_total_variation':v['trade_year_mass_total_variation']} for k,v in result['stages'].items()}},ensure_ascii=False))

