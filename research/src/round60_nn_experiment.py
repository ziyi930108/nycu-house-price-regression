"""R60 recency-weighted group OOF unit priors, original 144 numeric, fixed R54 area NN."""
from pathlib import Path
import contextlib,json,pickle,subprocess,sys,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round60_recency_price_prior as price_prior
ROOT=old.ROOT;OUT=ROOT/'outputs/round60';STAGES=old.STAGES
sha,read,write,spatial=old.sha,old.locked,old.write,old.spatial
TRAINER=ROOT/'src/round52_nn_train.py'
TRAINER_SHA='0d27f93d9c48c7dc13cf0252a16eadd1a2aa8a7f53b0740021fc3d8b37c0740f'
PRIOR_SHA='4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f'
FORMULA='median(spatial(reference(.75*(.5*oldraw+.5*(.5*R54_ramp_area_NN_TWD+.5*R60_recency_prior_NN_TWD))+.25*oldunit)),R46_recency_log_spatial,R40_unit_spatial)'
rel=lambda p:Path(p).relative_to(ROOT).as_posix()

def prepare(folder,prior_data,X,Xq,y,fi,prior_evidence,old_folds):
 tick=time.monotonic()
 names=('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities')
 with np.load(prior_data,allow_pickle=False) as saved:
  assert set(saved.files)==set(names)|{'train_price_factor_million','query_price_factor_million'}
  arrays={name:saved[name] for name in names}
 assert arrays['train_num'].shape==(len(X),144) and arrays['query_num'].shape==(len(Xq),144)
 assert arrays['train_y_million'].tobytes()==np.asarray(y/1e6,dtype=np.float32).tobytes()
 unchanged={name:old.original.value_sha(value,'<i8' if value.dtype==np.int64 else '<f4') for name,value in arrays.items()}
 train_unit,state,folds,evidence=price_prior.fit_oof_prior(X,y,fi)
 assert train_unit.dtype==np.float64 and train_unit.shape==(len(X),) and np.isfinite(train_unit).all() and (train_unit>0).all()
 assert folds.dtype==np.int64 and np.array_equal(folds,old_folds) and set(np.unique(folds))==set(range(5))
 identity=('fit_rows','groups','fallback_counts','fit_target_sha256','effective_area_sha256','fit_unit_sha256',
   'source_indices_sha256','fold_ids_sha256','group_keys_sha256','full_state_level_support')
 for key in identity:assert evidence[key]==prior_evidence['prior_evidence'][key],key
 assert not evidence['uses_query_labels'] and not evidence['chronological_oof']
 write(folder/'price_prior_state.json',state);restored=read(folder/'price_prior_state.json');assert restored==state
 query_unit,counts=price_prior.predict_prior(restored,Xq,return_diagnostics=True)
 assert query_unit.dtype==np.float64 and query_unit.shape==(len(Xq),) and sum(counts.values())==len(Xq)
 factors={name:np.asarray(old.area_helper.effective_area(frame)*unit/1e6,dtype=np.float32)
  for name,frame,unit in [('train_price_factor_million',X,train_unit),('query_price_factor_million',Xq,query_unit)]}
 for name,f in factors.items():assert np.isfinite(f).all() and (f>0).all(),name
 median=float(np.median(arrays['train_y_million'].astype(np.float64)/factors['train_price_factor_million'].astype(np.float64)))
 bias=median+float(np.log(-np.expm1(-median)))
 assert np.isfinite(median) and median>0 and np.isfinite(np.float32(bias))
 for name,v in [('train_oof_unit_prior',train_unit),('query_unit_prior',query_unit),('prior_fold_ids',folds)]:
  old.previous.save_array(folder/(name+'.npy'),v)
 path=folder/'prepared.npz'
 with path.open('xb') as f:np.savez(f,**arrays,**factors)
 prep={'fit_rows':len(X),'query_rows':len(Xq),'numeric_count':144,'source_prepared_sha256':sha(prior_data),
   'original_six_R52_encoding_arrays_reused':True,'unchanged_six_array_sha256':unchanged,
   'encoder_refit':False,'fit_labels_used_for_prior':True,'query_labels_used':False,
   'prior_evidence':evidence,'same_R52_fold_assignments':True,'same_R52_prior_identity_fields':list(identity),
   'query_prior_level_counts':counts,'prior_state_sha256':sha(folder/'price_prior_state.json'),
   'price_factor_formula':'effective_area_A * group_prior_TWD_per_m2 / 1e6',
   'initial_training_median_ratio':median,'initial_head_bias':bias,
   **{name+'_sha256':old.original.value_sha(v,'<f4') for name,v in factors.items()},
   'price_factor_million_summary':{name.replace('_price_factor_million',''):{'rows':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean(dtype=np.float64))} for name,v in factors.items()},
   'prior_prepare_seconds':time.monotonic()-tick}
 write(folder/'prior_preparation.json',prep)
 return path,prep

def train(folder,path,baseline,prep):
 out=folder/'model';out.mkdir()
 cmd=[str(old.TORCH_PYTHON),'-u','-B','-X','utf8',str(TRAINER),'--data',str(path),'--out',str(out)]
 write(out/'fit_started.json',{'command':cmd,'started_at_utc':old.now(),'input_sha256':sha(path),'automatic_retry':False})
 tick=time.monotonic()
 with (out/'trainer.log').open('x',encoding='utf8') as f:
  done=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
 write(out/'execution.json',{'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-tick,'command':cmd,'automatic_retry':False})
 assert done.returncode==0,'Trainer failed; preserve existing outputs'
 m=json.loads((out/'training.json').read_text(encoding='utf8'))
 assert m['status']=='completed' and m['configuration']==baseline['configuration']
 assert m['source_sha256']==TRAINER_SHA and m['architecture_source_sha256']==old.ARCHITECTURE_SHA
 assert m['input_sha256']==sha(path) and m['numeric_count']==prep['numeric_count'] and m['cardinalities']==baseline['cardinalities']
 assert m['fit_rows']==prep['fit_rows'] and m['query_rows']==prep['query_rows']
 assert m['saved_model_reload_exact'] and m['reload_rows']==prep['query_rows'] and m['prediction_units']=='TWD'
 assert not m['query_labels_available'] and not m['query_labels_used'] and not m['extra_sample_weights'] and not m['price_or_ratio_clipping']
 assert m['price_factor_uses_fit_labels'] and not m['price_factor_provenance_verified_in_trainer']
 assert [e['epoch'] for e in m['epochs']]==list(range(1,31))
 for key in ('price_factor_formula','train_price_factor_construction','query_price_factor_construction','output_units'):
  assert m[key]==baseline[key],key
 for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary',
   'initial_training_median_ratio','initial_head_bias'):
  assert m[key]==prep[key],key
 return old.vector(out/'predictions_twd.npy',prep['query_rows']),m

def run():
 start=time.monotonic()
 assert sha(TRAINER)==TRAINER_SHA and sha(Path(price_prior.__file__))==PRIOR_SHA
 assert sha(old.TRAINER)==old.TRAINER_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
 protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
 protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 p48=read(ROOT/'outputs/round48/plan.json');p52=read(ROOT/'outputs/round52/plan.json')
 records={r:read(ROOT/f'outputs/round{r}/results.json') for r in (48,49,50,52,53,54)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 Xall,yall=old.previous.load_training_rows(p48)
 source={rel(p):sha(p) for p in (Path(__file__),TRAINER,old.TRAINER,old.ARCHITECTURE,Path(old.__file__),Path(price_prior.__file__),ROOT/'src/round52_price_prior.py')}
 result={'round':60,'candidate':'R60','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},
   'new_NN_fits':0,'new_submission_generated':False,'fresh_blind_validation':False,'protected_before':protected,
   'source_sha256':source,'plan_sha256':sha(ROOT/'reports/round60_plan.md')}
 write(OUT/'plan.json',{'round':60,'formula':FORMULA,'source_sha256':source,'text_plan_sha256':result['plan_sha256'],
   'stage_order':list(STAGES),'split_contract':p48['split_contract'],'primary_control':'v15=R54',
   'query_labels_in_trainer':False,'seed':42,'epochs':30,'batch_size':2048,'numeric_representation':'unchanged original R52 144 columns','prior_rules':price_prior.RULES,'three_periods_regardless_of_scores':True,'new_gate':None})
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
  name=f'outputs/round52/{s}/model/training.json'
  baseline=json.loads(checked(name,a52['artifact_sha256'][name]).read_text(encoding='utf8'))
  assert baseline['status']=='completed' and baseline['configuration']['seed']==42 and baseline['configuration']['epochs']==30
  a54=records[54]['stages'][s]
  for number,record in ((52,a52),(54,a54)):
   for k in ('fit','query'):
    name=f'outputs/round{number}/{s}/{k}_indices.npy'
    assert np.array_equal(np.load(checked(name,record['artifact_sha256'][name]),allow_pickle=False),ind[k])
  name=f'outputs/round52/{s}/prior_preparation.json';prior_evidence=read(checked(name,a52['artifact_sha256'][name]))
  assert prior_evidence['fit_labels_used_for_prior'] and prior_evidence['query_labels_used'] is False and prior_evidence['encoder_refit'] is False
  for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary'):
   assert prior_evidence[key]==baseline[key],key
  name=f'outputs/round52/{s}/prepared.npz';prior_data=checked(name,a52['artifact_sha256'][name])
  assert baseline['input_sha256']==a52['artifact_sha256'][name]
  name=f'outputs/round52/{s}/prior_fold_ids.npy';old_folds=np.load(checked(name,a52['artifact_sha256'][name]),allow_pickle=False)
  prepared,prep=prepare(folder,prior_data,X,Xq,yall[fi],fi,prior_evidence,old_folds)
  new_native,training=train(folder,prepared,baseline,prep)
  name=f'outputs/round54/{s}/model/predictions_twd.npy';area_native=vector(name,a54['artifact_sha256'][name],n)
  name=f'outputs/round52/{s}/model/predictions_twd.npy';prior_native=vector(name,a52['artifact_sha256'][name],n)
  mean=.5*area_native+.5*new_native
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
  preds={'R60':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
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
    'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R60']) for k in ('v15','v13','v11','v12','v5')},
    'original_frame_sha256':{'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)},'preparation':prep,'new_NN_fits':1,'fit_seconds':training['fit_seconds'],'numeric_count':training['numeric_count'],
    'parameter_count':training['parameter_count'],'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],'reference_predict_calls':1,'spatial_predict_calls':1,'controls_full_metrics_exact':True,
    'full_query_reload_exact':True, 'native_MAE_diagnostic':{'R60_prior':float(np.abs(new_native-yall[qi]).mean()),
    'R52_prior':float(np.abs(prior_native-yall[qi]).mean()),'R54_area':float(np.abs(area_native-yall[qi]).mean())},
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
