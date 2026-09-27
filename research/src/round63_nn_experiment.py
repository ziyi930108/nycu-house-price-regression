"""R63 fit-date weighted total-price MAE on unchanged R60 prepared priors; three fits."""
from pathlib import Path
import contextlib,json,pickle,subprocess,sys,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round60_recency_price_prior as date_weights
ROOT=old.ROOT;OUT=ROOT/'outputs/round63';STAGES=old.STAGES
sha,read,write,spatial=old.sha,old.locked,old.write,old.spatial
TRAINER=ROOT/'src/round63_recency_weighted_prior_train.py'
DATE_WEIGHT_SOURCE=ROOT/'src/round60_recency_price_prior.py'
DATE_WEIGHT_SHA='4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f'
PARENT=ROOT/'src/round52_nn_train.py'
PARENT_SHA='0d27f93d9c48c7dc13cf0252a16eadd1a2aa8a7f53b0740021fc3d8b37c0740f'
TRAINER_SHA='26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
LOSS_OVERRIDES={'loss':'fit-recency weighted MAE in million TWD','loss_formula':'mean(train_sample_weight * abs(ratio_output * train_price_factor_million - train_y_million))','extra_sample_weights':True,'sample_weight_normalization':'driver-supplied fit-global normalization; no batch renormalization'}
FORMULA='median(spatial(reference(.75*(.5*oldraw+.5*(.5*R54_ramp_area_NN_TWD+.5*R63_weighted_MAE_recency_prior_NN_TWD))+.25*oldunit)),R46_recency_log_spatial,R40_unit_spatial)'
rel=lambda p:Path(p).relative_to(ROOT).as_posix()

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
 assert m['status']=='completed' and m['configuration']=={**baseline['configuration'],**LOSS_OVERRIDES}
 assert m['parent_trainer_source_sha256']==PARENT_SHA and m['loss_configuration_overrides']==LOSS_OVERRIDES
 assert m['changed_configuration_fields']==['loss','loss_formula','extra_sample_weights'] and m['added_configuration_fields']==['sample_weight_normalization']
 assert m['training_objective_name']=='weighted MAE' and m['epoch_selection_uses_training_metrics'] is False
 assert m['train_sample_weight_sha256']==prep['train_sample_weight_sha256']
 for key in ('rows','min','max'):
  assert m['train_sample_weight_summary'][key]==prep['train_sample_weight_summary'][key],key
 for key in ('mean','sum','sum_squares','effective_sample_size'):
  assert np.isclose(m['train_sample_weight_summary'][key],prep['train_sample_weight_summary'][key],rtol=1e-12,atol=1e-12),key
 assert m['train_sample_weight_shape']==[prep['fit_rows']] and m['train_sample_weight_dtype']=='float32'
 assert not m['sample_weight_provenance_verified_in_trainer'] and not m['sample_weights_recomputed_in_trainer'] and not m['sample_weights_batch_renormalized']
 assert m['source_sha256']==TRAINER_SHA and m['architecture_source_sha256']==old.ARCHITECTURE_SHA
 assert m['input_sha256']==sha(path) and m['numeric_count']==prep['numeric_count'] and m['cardinalities']==baseline['cardinalities']
 assert m['fit_rows']==prep['fit_rows'] and m['query_rows']==prep['query_rows']
 assert m['saved_model_reload_exact'] and m['reload_rows']==prep['query_rows'] and m['prediction_units']=='TWD'
 assert not m['query_labels_available'] and not m['query_labels_used'] and m['extra_sample_weights'] is True and not m['price_or_ratio_clipping']
 assert m['price_factor_uses_fit_labels'] and not m['price_factor_provenance_verified_in_trainer']
 assert [e['epoch'] for e in m['epochs']]==list(range(1,31))
 for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary',
   'initial_training_median_ratio','initial_head_bias','price_factor_formula','train_price_factor_construction','query_price_factor_construction','output_units'):
  assert m[key]==baseline[key],key
 for e in m['epochs']:
  assert np.isfinite(e['training_mae_million_with_dropout']) and np.isfinite(e['training_weighted_mae_million_with_dropout'])
 return old.vector(out/'predictions_twd.npy',prep['query_rows']),m

def run():
 start=time.monotonic()
 assert sha(TRAINER)==TRAINER_SHA and sha(PARENT)==PARENT_SHA
 assert sha(DATE_WEIGHT_SOURCE)==DATE_WEIGHT_SHA and Path(date_weights.__file__).resolve()==DATE_WEIGHT_SOURCE
 assert sha(old.TRAINER)==old.TRAINER_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
 protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
 protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 p48=read(ROOT/'outputs/round48/plan.json');p52=read(ROOT/'outputs/round52/plan.json')
 records={r:read(ROOT/f'outputs/round{r}/results.json') for r in (48,49,50,53,54,60,62)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 Xall,yall=old.previous.load_training_rows(p48)
 source={rel(p):sha(p) for p in (Path(__file__),TRAINER,old.TRAINER,old.ARCHITECTURE,Path(old.__file__),PARENT,DATE_WEIGHT_SOURCE,Path(date_weights.original.__file__))}
 result={'round':63,'candidate':'R63','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},
   'new_NN_fits':0,'new_submission_generated':False,'fresh_blind_validation':False,'protected_before':protected,
   'source_sha256':source,'plan_sha256':sha(ROOT/'reports/round63_plan.md')}
 write(OUT/'plan.json',{'round':63,'formula':FORMULA,'source_sha256':source,'text_plan_sha256':result['plan_sha256'],
   'stage_order':list(STAGES),'split_contract':p48['split_contract'],'primary_control':'v16=R60',
   'query_labels_in_trainer':False,'seed':42,'epochs':30,'batch_size':2048,'numeric_representation':'unchanged original R60 144 columns','loss_overrides':LOSS_OVERRIDES,'original_eight_arrays_reused_without_changes':True,'extra_array':'train_sample_weight','date_weight_source_sha256':DATE_WEIGHT_SHA,'date_weight_rules':date_weights.WEIGHT_RULES,'date_weight_scope':'full canonical stage fit rows only; no query dates or labels','no_batch_weight_normalization':True,'three_periods_regardless_of_scores':True,'new_gate':None})
 inputs={}
 def checked(name,digest):
  path=ROOT/name;assert sha(path)==digest,name;inputs[name]=digest;return path
 def vector(name,digest,n):return old.vector(checked(name,digest),n)
 for s in STAGES:
  tick=time.monotonic();folder=OUT/s;folder.mkdir();b=p48['bindings'][s];bs=p52['bindings'][s]
  a48=records[48]['stages'][s];a60=records[60]['stages'][s]
  ind={}
  for k in ('fit','query'):
   name=f'outputs/round48/{s}/{k}_indices.npy';ind[k]=np.load(checked(name,a48['artifact_sha256'][name]),allow_pickle=False)
   assert old.original.value_sha(ind[k],'<i8')==p48['split_contract'][s][k+'_indices_content_sha256']
   old.previous.save_array(folder/f'{k}_indices.npy',ind[k])
  fi,qi=ind['fit'],ind['query'];X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);n=len(qi)
  assert not np.intersect1d(fi,qi).size
  assert spatial.frame_sha(X)==a48['original_frame_sha256']['fit_X']==a60['original_frame_sha256']['fit_X']
  assert spatial.frame_sha(Xq)==a48['original_frame_sha256']['query_X']==a60['original_frame_sha256']['query_X']
  name=f'outputs/round60/{s}/model/training.json'
  baseline=json.loads(checked(name,a60['artifact_sha256'][name]).read_text(encoding='utf8'))
  assert baseline['status']=='completed' and baseline['configuration']['seed']==42 and baseline['configuration']['epochs']==30
  a54=records[54]['stages'][s]
  for number,record in ((60,a60),(54,a54),(62,records[62]['stages'][s])):
   for k in ('fit','query'):
    name=f'outputs/round{number}/{s}/{k}_indices.npy'
    assert np.array_equal(np.load(checked(name,record['artifact_sha256'][name]),allow_pickle=False),ind[k])
  name=f'outputs/round60/{s}/prior_preparation.json';prior_evidence=read(checked(name,a60['artifact_sha256'][name]))
  assert prior_evidence['fit_labels_used_for_prior'] and prior_evidence['query_labels_used'] is False and prior_evidence['encoder_refit'] is False
  for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary'):
   assert prior_evidence[key]==baseline[key],key
  name=f'outputs/round60/{s}/prepared.npz';prior_data=checked(name,a60['artifact_sha256'][name])
  assert baseline['input_sha256']==a60['artifact_sha256'][name]
  with np.load(prior_data,allow_pickle=False) as z:
   expected={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million'}
   assert set(z.files)==expected and z['train_num'].shape==(len(fi),144) and z['query_num'].shape==(n,144)
   assert z['train_y_million'].tobytes()==np.asarray(yall[fi]/1e6,dtype=np.float32).tobytes()
   for key in ('train_price_factor_million','query_price_factor_million'):
    assert old.original.value_sha(z[key],'<f4')==baseline[key+'_sha256']
   arrays={k:z[k].copy() for k in z.files}
  weights64,valid_dates,date_summary=date_weights._weights(date_weights._dates(X))
  assert weights64.shape==(len(fi),) and valid_dates.shape==weights64.shape
  weights32=weights64.astype(np.float32)
  assert np.isfinite(weights32).all() and (weights32>0).all() and np.array_equal(weights32[~valid_dates],np.ones(int((~valid_dates).sum()),dtype=np.float32))
  # Preserve fit-global normalization; no renormalization after float32 casting.
  v=weights32.astype(np.float64);total=float(v.sum(dtype=np.float64));squares=float(np.dot(v,v))
  summary={'rows':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),
    'sum':total,'sum_squares':squares,'effective_sample_size':total*total/squares}
  assert np.isfinite(list(summary.values())).all() and 0<summary['effective_sample_size']<=len(fi)*(1+1e-12)
  prepared=folder/'prepared.npz'
  with prepared.open('xb') as f:np.savez(f,**arrays,train_sample_weight=weights32)
  with np.load(prepared,allow_pickle=False) as z:
   assert set(z.files)==expected|{'train_sample_weight'}
   for k,a in arrays.items():assert z[k].dtype==a.dtype and z[k].shape==a.shape and z[k].tobytes()==a.tobytes(),k
   assert z['train_sample_weight'].dtype==np.float32 and z['train_sample_weight'].tobytes()==weights32.tobytes()
  del arrays
  prep={'fit_rows':len(fi),'query_rows':n,'numeric_count':144,'prepared_path':rel(prepared),'prepared_sha256':sha(prepared),
   'all_eight_R60_arrays_reused_without_changes':True,'exact_NPZ_keys':sorted(expected|{'train_sample_weight'}),
   'date_weight_summary_float64':date_summary,'train_sample_weight_sha256':old.original.value_sha(weights32,'<f4'),
   'train_sample_weight_summary':summary,'neutral_weights_exactly_one':True,'weights_use_fit_X_only':True,
   'weights_use_targets':False,'weights_use_query_dates':False,'float32_weights_renormalized':False,
   'source_prepared_path':rel(prior_data),'source_prepared_sha256':sha(prior_data),'encoder_refit':False,'prior_refit':False,
   'fit_labels_used_for_prior':True,'query_labels_used':False,'prior_preparation_path':f'outputs/round60/{s}/prior_preparation.json',
   'prior_preparation_sha256':sha(ROOT/f'outputs/round60/{s}/prior_preparation.json')}
  write(folder/'preparation.json',prep)
  new_native,training=train(folder,prepared,baseline,prep)
  name=f'outputs/round54/{s}/model/predictions_twd.npy';area_native=vector(name,a54['artifact_sha256'][name],n)
  name=f'outputs/round60/{s}/model/predictions_twd.npy';prior_native=vector(name,a60['artifact_sha256'][name],n)
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
  preds={'R63':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
  for label,r in (('v11',49),('v12',50),('v13',53),('v15',54),('v16',60),('R62',62)):
   name=f'outputs/round{r}/{s}/predictions.npy';preds[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n)
  score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
  metrics={k:score(v) for k,v in preds.items()}
  for label,r,key in [('R62',62,'R62'),('v16',60,'R60'),('v11',49,'R49'),('v12',50,'R50'),('v13',53,'R53'),('v15',54,'R54'),('v5',53,'v5')]:
   assert metrics[label]==records[r]['stages'][s]['metrics'][key]
  for name,v in [('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)]:
   old.previous.save_array(folder/f'{name}.npy',v)
  for name,v in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
   with (folder/f'{name}.pkl').open('xb') as f:pickle.dump(v,f,protocol=5)
  item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,
    'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R63']) for k in ('v16','v15','v13','v11','v12','v5','R62')},
    'original_frame_sha256':{'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)},'preparation':prep,'training_loss_comparison':{'R60_final_training_MAE_million_with_dropout':baseline['epochs'][-1]['training_mae_million_with_dropout'],'R63_final_training_MAE_million_with_dropout':training['epochs'][-1]['training_mae_million_with_dropout'],'R63_final_training_weighted_MAE_million_with_dropout':training['epochs'][-1]['training_weighted_mae_million_with_dropout']},'new_NN_fits':1,'fit_seconds':training['fit_seconds'],'numeric_count':training['numeric_count'],
    'parameter_count':training['parameter_count'],'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],'reference_predict_calls':1,'spatial_predict_calls':1,'controls_full_metrics_exact':True,
    'full_query_reload_exact':True, 'native_MAE_diagnostic':{'R63_prior':float(np.abs(new_native-yall[qi]).mean()),
    'R60_prior':float(np.abs(prior_native-yall[qi]).mean()),'R54_area':float(np.abs(area_native-yall[qi]).mean())},
    'reference_report':rr,'spatial_report':sr,'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
    'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
  write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s);result['new_NN_fits']+=1
  print(json.dumps({'stage':s,'numeric_count':training['numeric_count'],'fit_seconds':training['fit_seconds'],
   'vs_v16':{k:item['differences']['v16'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
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
