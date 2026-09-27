"""R56 fit-only quantile ramps appended to R48 NN numeric inputs."""
from pathlib import Path
import contextlib,json,pickle,subprocess,sys,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round56_numeric_encode as encoder
ROOT=old.ROOT;OUT=ROOT/'outputs/round56';STAGES=old.STAGES
sha,read,write,spatial=old.sha,old.locked,old.write,old.spatial
ENCODER_SHA='ec5525fb354d23675559c67721027bba2c663abce0d4a1e2a7de6a56844e57c3'
FORMULA='median(spatial(reference(.75*(.5*oldraw+.5*(.5*R56_ramp_area_NN_TWD+.5*R52_native_TWD))+.25*oldunit)),R46_recency_log_spatial,R40_unit_spatial)'
rel=lambda p:Path(p).relative_to(ROOT).as_posix()

def prepare(folder,path,y):
 tick=time.monotonic()
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)=={'train_num','query_num','train_cat','query_cat','cardinalities','train_y_million','train_area_factor','query_area_factor'}
  arrays={k:z[k] for k in z.files}
 assert np.array_equal(arrays['train_y_million'],np.asarray(y/1e6,dtype=np.float32))
 assert arrays['train_num'].shape[1]==arrays['query_num'].shape[1]==144
 original={k:sha_array(a) for k,a in arrays.items()}
 mapping=json.loads((path.parent/'encoder_state.json').read_text(encoding='utf8'))
 assert mapping['fit_rows']==len(y) and mapping['query_used_for_state'] is False
 state=encoder.fit_state(arrays['train_num'],mapping['numeric_columns']);write(folder/'numeric_state.json',state)
 state2=read(folder/'numeric_state.json');assert state2==state and state['fit_rows']==len(y)
 for k in ('train_num','query_num'):
  num=encoder.transform(state2,arrays[k])
  assert num.dtype==np.float32 and num.shape[1]==state['output_width'] and num.shape[0]==len(arrays[k])
  assert num[:,:144].tobytes()==arrays[k].tobytes() and np.isfinite(num).all()
  arrays[k]=num
 unchanged={k:sha_array(a) for k,a in arrays.items() if k not in ('train_num','query_num')}
 assert all(original[k]==v for k,v in unchanged.items())
 path=folder/'prepared.npz'
 with path.open('xb') as f:np.savez(f,**arrays)
 evidence={'source_prepared_arrays_sha256':original,'unchanged_six_array_sha256':unchanged,'original_numeric_prefix_bitwise_equal':True,
  'fit_rows':len(y),'query_rows':len(arrays['query_num']),'numeric_count':state['output_width'],'query_labels_used':False,
  'state_fit_only':True,'excluded_ramp_columns':state['excluded_numeric_names'],'ramp_quantile_count':17,'extra_ramp_columns':state['ramp_width'],'elapsed_seconds':time.monotonic()-tick}
 write(folder/'preparation.json',evidence)
 return path,evidence

def sha_array(a):
 import hashlib
 return hashlib.sha256(a.tobytes()).hexdigest()

def train(folder,path,baseline,prep):
 out=folder/'model';out.mkdir()
 cmd=[str(old.TORCH_PYTHON),'-u','-B','-X','utf8',str(old.TRAINER),'--data',str(path),'--out',str(out)]
 write(out/'fit_started.json',{'command':cmd,'started_at_utc':old.now(),'input_sha256':sha(path),'automatic_retry':False})
 tick=time.monotonic()
 with (out/'trainer.log').open('x',encoding='utf8') as f:
  done=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
 write(out/'execution.json',{'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-tick,'command':cmd,'automatic_retry':False})
 assert done.returncode==0,'Trainer failed; preserve existing outputs'
 m=json.loads((out/'training.json').read_text(encoding='utf8'))
 assert m['status']=='completed' and m['configuration']==baseline['configuration']
 assert m['source_sha256']==old.TRAINER_SHA and m['architecture_source_sha256']==old.ARCHITECTURE_SHA
 assert m['input_sha256']==sha(path) and m['numeric_count']==prep['numeric_count'] and m['cardinalities']==baseline['cardinalities']
 assert m['fit_rows']==prep['fit_rows'] and m['query_rows']==prep['query_rows']
 assert m['saved_model_reload_exact'] and m['reload_rows']==prep['query_rows'] and m['prediction_units']=='TWD'
 assert not m['query_labels_available'] and not m['query_labels_used'] and not m['extra_sample_weights'] and not m['price_or_unit_clipping']
 assert [e['epoch'] for e in m['epochs']]==list(range(1,31))
 for k in ('train_area_factor_sha256','query_area_factor_sha256','initial_training_median_unit_1e4_twd_per_m2'):
  assert m[k]==baseline[k],k
 return old.vector(out/'predictions_twd.npy',prep['query_rows']),m

def run():
 start=time.monotonic()
 assert sha(Path(encoder.__file__))==ENCODER_SHA
 assert sha(ROOT/'src/round54_numeric_encode.py')=='fe73f4154bcd0811b7cdad09c58e1dfd7cba5350446193e90f70bd8a7ef0206e'
 assert sha(old.TRAINER)==old.TRAINER_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
 protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
 protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 p48=read(ROOT/'outputs/round48/plan.json');p52=read(ROOT/'outputs/round52/plan.json')
 records={r:read(ROOT/f'outputs/round{r}/results.json') for r in (48,49,50,52,53,54)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 Xall,yall=old.previous.load_training_rows(p48)
 source={rel(p):sha(p) for p in (Path(__file__),Path(encoder.__file__),old.TRAINER,old.ARCHITECTURE,Path(old.__file__),ROOT/'src/round54_numeric_encode.py')}
 result={'round':54,'candidate':'R56','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},
   'new_NN_fits':0,'new_submission_generated':False,'fresh_blind_validation':False,'protected_before':protected,
   'source_sha256':source,'plan_sha256':sha(ROOT/'reports/round56_plan.md')}
 write(OUT/'plan.json',{'round':54,'formula':FORMULA,'source_sha256':source,'text_plan_sha256':result['plan_sha256'],
   'stage_order':list(STAGES),'split_contract':p48['split_contract'],'primary_control':'v15=R54',
   'query_labels_in_trainer':False,'seed':42,'epochs':30,'ramp_quantile_count':17,'new_gate':None})
 inputs={}
 def checked(name,digest):
  path=ROOT/name;assert sha(path)==digest,name;inputs[name]=digest;return path
 def vector(name,digest,n):return old.vector(checked(name,digest),n)
 for s in STAGES:
  tick=time.monotonic();folder=OUT/s;folder.mkdir();b=p48['bindings'][s];bs=p52['bindings'][s]
  a48=records[48]['stages'][s];a52=records[52]['stages'][s]
  ind={}
  for k in ('fit','query'):
   name=f'outputs/round48/{s}/{k}_indices.npy';ind[k]=np.load(checked(name,a48['artifact_sha256'][name]),allow_pickle=False)
   assert old.original.value_sha(ind[k],'<i8')==p48['split_contract'][s][k+'_indices_content_sha256']
   old.previous.save_array(folder/f'{k}_indices.npy',ind[k])
  fi,qi=ind['fit'],ind['query'];X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);n=len(qi)
  assert not np.intersect1d(fi,qi).size
  assert spatial.frame_sha(X)==a48['original_frame_sha256']['fit_X']==a52['original_frame_sha256']['fit_X']
  assert spatial.frame_sha(Xq)==a48['original_frame_sha256']['query_X']==a52['original_frame_sha256']['query_X']
  name=f'outputs/round48/{s}/model/training.json'
  baseline=json.loads(checked(name,a48['artifact_sha256'][name]).read_text(encoding='utf8'))
  assert baseline['status']=='completed' and baseline['configuration']['seed']==42 and baseline['configuration']['epochs']==30
  name=f'outputs/round48/{s}/encoder_state.json';checked(name,a48['artifact_sha256'][name])
  name=f'outputs/round48/{s}/prepared.npz';data=checked(name,a48['artifact_sha256'][name])
  assert baseline['input_sha256']==a48['artifact_sha256'][name]
  prepared,prep=prepare(folder,data,yall[fi])
  new_native,training=train(folder,prepared,baseline,prep)
  name=f'outputs/round52/{s}/model/predictions_twd.npy';prior_native=vector(name,a52['artifact_sha256'][name],n)
  mean=.5*new_native+.5*prior_native
  raw,unit=[vector(b[k],p48['input_sha256'][b[k]],n) for k in ('old_raw','old_unit')]
  B=.75*(.5*raw+.5*mean)+.25*unit
  with checked(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f:reference=pickle.load(f)
  with checked(b['state'],p48['input_sha256'][b['state']]).open('rb') as f:state=pickle.load(f)
  assert state['state_summary']['fit_X_sha256']==a48['original_frame_sha256']['fit_X'] and not state['state_summary']['query_used_in_build']
  assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
  geo=pd.read_pickle(checked(b['query_geo'],p48['input_sha256'][b['query_geo']]))
  assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
  p,diag,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
  branch,detail,sr=spatial.predict(state,Xq,geo,B,p,diag)
  fallback=~detail['applied'].to_numpy(dtype=bool);old.exact(branch[fallback],p[fallback],'spatial fallback')
  log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n)
  ub=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
  candidate=np.median(np.column_stack([branch,log,ub]),axis=1)
  assert candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
  preds={'R56':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
  for label,r in (('v11',49),('v12',50),('v13',53),('v15',54)):
   name=f'outputs/round{r}/{s}/predictions.npy';preds[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n)
  score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
  metrics={k:score(v) for k,v in preds.items()}
  for label,r,key in [('v11',49,'R49'),('v12',50,'R50'),('v13',53,'R53'),('v15',54,'R54'),('v5',53,'v5')]:
   assert metrics[label]==records[r]['stages'][s]['metrics'][key]
  for name,v in [('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)]:
   old.previous.save_array(folder/f'{name}.npy',v)
  for name,v in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
   with (folder/f'{name}.pkl').open('xb') as f:pickle.dump(v,f,protocol=5)
  item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,
    'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R56']) for k in ('v15','v13','v11','v12','v5')},
    'preparation':prep,'new_NN_fits':1,'fit_seconds':training['fit_seconds'],'numeric_count':training['numeric_count'],
    'parameter_count':training['parameter_count'],'reference_predict_calls':1,'spatial_predict_calls':1,'controls_full_metrics_exact':True,
    'full_query_reload_exact':True,'native_MAE_diagnostic':{'R56':float(np.abs(new_native-yall[qi]).mean()),
    'R48':float(np.abs(old.vector(ROOT/f'outputs/round48/{s}/model/predictions_twd.npy',n)-yall[qi]).mean()),
    'R54':float(np.abs(vector(f'outputs/round54/{s}/model/predictions_twd.npy',records[54]['stages'][s]['artifact_sha256'][f'outputs/round54/{s}/model/predictions_twd.npy'],n)-yall[qi]).mean())},
    'reference_report':rr,'spatial_report':sr,'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
    'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
  write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s);result['new_NN_fits']+=1
  print(json.dumps({'stage':s,'numeric_count':training['numeric_count'],'fit_seconds':training['fit_seconds'],
   'vs_v15':{k:item['differences']['v15'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
 assert all(sha(ROOT/n)==h for n,h in source.items())
 result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,elapsed_seconds=time.monotonic()-start,
   protected_after={n:sha(ROOT/n) for n in protected})
 assert result['protected_before']==result['protected_after'];write(OUT/'results.json',result)
 print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':
 assert not OUT.exists(),'Existing output: no overwrite';OUT.mkdir();tick=time.monotonic();code=0;failure=None
 write(OUT/'started.json',{'started_at_utc':old.now(),'command':sys.argv,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 try:run()
 except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
   'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 raise SystemExit(code)
