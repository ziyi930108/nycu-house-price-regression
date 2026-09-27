"""One saved-vector R43 median trial release; no training or prediction calls."""
import contextlib
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round43/trial'
CSV=ROOT/'outputs/submission_v9_trial.csv'
AUTHORITIES={
 'outputs/round43/results.json':'6b5f0ea14752b9c70b1c6e89df96f4ebd7ac083054518f0d3485451636bd5387',
 'outputs/round39/trial/trial_release.json':'5e385afdf721e6c4925745c7120ee6a98b6284dac0bf7e93d8f4e9a6d31d560c',
 'outputs/round40/trial/trial_release.json':'c5f33956319569f951b99c7a79de089a4c85ae464803d18fb25d128730e5ca2d',
 'outputs/round37/trial/trial_release.json':'e7794550b976eca72fe3e6fb0bcdbdf02a24ce0b8e008139d328a2422bb827fe'}
PATHS={'raw':'outputs/round39/trial/artifacts/raw_predictions.npy','log':'outputs/round39/trial/artifacts/log_predictions.npy',
 'unit':'outputs/round40/trial/unit_spatial.npy','v5':'outputs/round37/trial/predictions.npy',
 'v7':'outputs/round39/trial/artifacts/predictions.npy','v8':'outputs/round40/trial/predictions.npy'}
ID,TARGET='編號','總價元'
PROTECTED=tuple('outputs/submission_'+v+('.csv' if v in ('v2','v3') else '_trial.csv') for v in ('v2','v3','v4','v5','v6','v7','v8'))+('runs/versions.json',)
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
 with path.open('x',encoding='utf8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def rounded(a):return np.maximum(np.rint(a),1).astype('int64')
def exact(a,b):return a.dtype==b.dtype==np.dtype('float64') and a.shape==b.shape==(47073,) and a.tobytes()==b.tobytes()
def csvcheck(path,ids,prices):
 f=pd.read_csv(path,encoding='utf-8-sig',dtype={ID:str},keep_default_na=False)
 need(list(f)==[ID,TARGET] and len(f)==47073 and f[ID].is_unique and f[ID].tolist()==ids
  and f[TARGET].dtype==np.dtype('int64') and (f[TARGET]>0).all() and np.array_equal(f[TARGET].to_numpy(),prices),'CSV differs: '+str(path))
def main():
 inputs={};docs={};known={};before={n:sha(ROOT/n) for n in PROTECTED}
 def bind(name,digest):
  need((ROOT/name).resolve().is_relative_to(ROOT) and sha(ROOT/name)==digest,'Input changed: '+name)
  need(name not in inputs or inputs[name]==digest,'Identity conflict');inputs[name]=digest
 for name,digest in AUTHORITIES.items():
  bind(name,digest);receipt=str(Path(name).with_suffix('.sha256')).replace('\\','/')
  need((ROOT/receipt).read_text(encoding='ascii').strip()==digest,'Receipt differs');bind(receipt,sha(ROOT/receipt))
  d=read(ROOT/name);docs[name]=d
  for key in ('input_sha256','artifact_sha256'):
   for k,v in d.get(key,{}).items():need(k not in known or known[k]==v,'Historical identity conflict');known[k]=v
 r=docs['outputs/round43/results.json']
 need(r['status']=='three_periods_completed_no_adoption_decision' and r['checked_stages']==['development','confirmation','auxiliary'],'R43 incomplete')
 for n in PATHS.values():bind(n,known[n])
 sample=next(k for k in known if k.endswith('test-reindex-test-sample-submission-v2.csv'));bind(sample,known[sample])
 xfile='data_cache/test_features.pkl';kfile='outputs/round6/geography/final/test_source_cache_indices.npy'
 for n in (xfile,kfile):bind(n,known[n])
 vectors={n:np.load(ROOT/path,allow_pickle=False) for n,path in PATHS.items()}
 for a in vectors.values():need(a.shape==(47073,) and a.dtype==np.dtype('float64') and np.isfinite(a).all() and (a>0).all(),'Invalid vector')
 need(exact(.5*vectors['raw']+.5*vectors['log'],vectors['v7']),'v7 mean differs')
 need(exact((vectors['raw']+vectors['log']+vectors['unit'])/3.0,vectors['v8']),'v8 mean differs')
 frame=pd.read_csv(ROOT/sample,encoding='utf-8-sig',usecols=[ID],dtype={ID:str},keep_default_na=False);ids=frame[ID].tolist()
 need(len(ids)==47073 and frame[ID].is_unique and all(ids),'Sample IDs invalid')
 for n,rnd in (('v5',37),('v7',39),('v8',40)):
  path='outputs/submission_'+n+'_trial.csv';proof=docs[f'outputs/round{rnd}/trial/trial_release.json']
  bind(path,proof['sha256']);csvcheck(ROOT/path,ids,rounded(vectors[n]))
 values=np.median(np.column_stack([vectors[n] for n in ('raw','log','unit')]),axis=1)
 need(values.shape==(47073,) and values.dtype==np.dtype('float64') and np.isfinite(values).all() and (values>0).all(),'Invalid median')
 prices=rounded(values)
 with (OUT/'predictions.npy').open('xb') as f:np.save(f,values,allow_pickle=False)
 staged=OUT/CSV.name;pd.DataFrame({ID:ids,TARGET:prices}).to_csv(staged,encoding='utf-8-sig',index=False,mode='x')
 with staged.open('rb') as f:need(f.read(3)==b'\xef\xbb\xbf','BOM absent')
 csvcheck(staged,ids,prices)
 X=pd.read_pickle(ROOT/xfile);k=np.load(ROOT/kfile,allow_pickle=False)
 need(len(X)==47073 and X.index.equals(pd.RangeIndex(47073)) and k.dtype==np.dtype('int64')
      and np.array_equal(k,np.arange(47073)) and len(X.columns)==87 and TARGET not in X,'Pure X alignment differs')
 fields=['address_city','address_district','transaction_type','building_type','main_use','building_area','land_area','parking_area','parking_count']
 def scalar(v):
  if pd.isna(v):return None
  return v.item() if isinstance(v,np.generic) else v
 def describe(i):
  return {'position':int(i),'ID':ids[i],'pure_X':{c:scalar(X.iloc[i][c]) for c in fields},
    'complete_spatial_branches_TWD':{n:float(vectors[n][i]) for n in ('raw','log','unit')},
    'v5_TWD':float(vectors['v5'][i]),'v7_TWD':float(vectors['v7'][i]),'v8_mean_TWD':float(vectors['v8'][i]),
    'v9_median_TWD':float(values[i]),'v9_rounded_TWD':int(prices[i]),'median_minus_v8_TWD':float(values[i]-vectors['v8'][i]),
    'true_price_unknown':True}
 top=np.argsort(-np.abs(values-vectors['v8']),kind='stable')[:12]
 changes={}
 for n in ('v5','v7','v8'):
  diff=values-vectors[n];changes[n]={'changed_unrounded_rows':int(np.count_nonzero(diff)),
    'changed_rounded_rows':int(np.count_nonzero(prices!=rounded(vectors[n]))),
    'mean_absolute_prediction_change_TWD':float(np.abs(diff).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(diff).max()),
    'minimum_signed_change_TWD':float(diff.min()),'maximum_signed_change_TWD':float(diff.max())}
 after={n:sha(ROOT/n) for n in PROTECTED};need(before==after,'Protected files changed')
 need(not CSV.exists(),'Existing v9 CSV');os.link(staged,CSV)
 proof={'round':43,'status':'experimental_trial_generated','trial_only':True,'path':str(CSV.relative_to(ROOT)).replace('\\','/'),
    'sha256':sha(CSV),'query_rows':47073,'formula':'median([saved_raw_spatial,saved_log_spatial,saved_unit_spatial],axis=1)',
    'source_sha256':{str(Path(__file__).relative_to(ROOT)).replace('\\','/'):sha(__file__)},'input_sha256':inputs,
    'new_fits':0,'model_loads':0,'tree_predictions':0,'reference_or_spatial_predictions':0,'test_labels_loaded':False,
    'v7_and_v8_mean_vectors_bitwise_equal':True,'v5_v7_v8_CSVs_exact':True,'ID_order_verified':True,'UTF8_BOM':True,
    'rounding':'max(rint(median),1), once','finite_positive_integer_prices':True,'prediction_changes_vs_controls':changes,
    'top12_order':'descending absolute unrounded v9 minus v8 prediction change, stable ties',
    'top12_pure_X_metadata':[describe(int(i)) for i in top],
    'ID30528':describe(ids.index('30528')),'no_ID_or_area_specific_rule':True,'prediction_changes_are_not_MAE':True,
    'new_public_mae':None,'private_mae':None,'uploaded':False,'registry_updated':False,
    'protected_before':before,'protected_after':after,'runtime':{'numpy':np.__version__,'pandas':pd.__version__},
    'predictions_sha256':sha(OUT/'predictions.npy'),'completed_at_utc':now()}
 write(OUT/'release.json',proof);print(json.dumps({'path':proof['path'],'sha256':proof['sha256'],'differences':changes,'ID30528':proof['ID30528']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 need(not OUT.exists() and not CSV.exists(),'Existing release attempt; no retry')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();stamp=now();exit_code=1;error=None
 write(OUT/'run_claim.json',{'started_at_utc':stamp,'source_sha256':sha(__file__),'automatic_retry':False})
 with (OUT/'execution.log').open('x',encoding='utf8') as f:
  with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
   try:main();exit_code=0
   except Exception:error=traceback.format_exc();print(error,flush=True)
 event={'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-start,
  'exit_code':exit_code,'failure':error,'source_sha256':sha(__file__),'log_sha256':sha(OUT/'execution.log'),'automatic_retry':False}
 write(OUT/'execution.json',event);print(json.dumps(event,ensure_ascii=False),flush=True);raise SystemExit(exit_code)
