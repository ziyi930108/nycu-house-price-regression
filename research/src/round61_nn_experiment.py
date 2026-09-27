"""Paired ordinary/recency prior seeds and fixed three-seed means; 12 new fits."""
from pathlib import Path
import json,pickle,subprocess,sys,time,traceback
import numpy as np
import pandas as pd
import round60_nn_experiment as previous

old=previous.old;ROOT=old.ROOT;OUT=ROOT/'outputs/round61';STAGES=old.STAGES
sha,read,write,spatial=old.sha,old.locked,old.write,old.spatial
WRAPPER=ROOT/'src/round61_prior_seed_train.py'
WRAPPER_SHA='5e4bf126fe92bec8ae1308aded202ff11566135454050d1be78d22e5fd7d3641'
SEEDS=(42,137,2026);NEW_SEEDS=(137,2026);METHODS={'ordinary':52,'recency':60}
FORMULA='median(spatial(reference(.75*(.5*oldraw+.5*(.5*saved_R54_area_TWD+.5*((R60prior42+R60prior137+R60prior2026)/3)))+.25*oldunit)),R46_recency_log_spatial,R40_unit_spatial)'
rel=previous.rel

def train(path,out,seed,baseline):
 out.mkdir(parents=True)
 command=[str(old.TORCH_PYTHON),'-u','-B','-X','utf8',str(WRAPPER),'--data',str(path),'--out',str(out),'--seed',str(seed)]
 write(out/'fit_started.json',{'command':command,'seed':seed,'started_at_utc':old.now(),'input_sha256':sha(path),'automatic_retry':False})
 tick=time.monotonic()
 with (out/'trainer.log').open('x',encoding='utf8') as f:
  done=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
 write(out/'execution.json',{'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-tick,'command':command,'automatic_retry':False})
 assert done.returncode==0,'Trainer failed; preserve outputs, no automatic retry'
 m=json.loads((out/'training.json').read_text(encoding='utf8'))
 assert m['status']=='completed' and m['configuration']=={**baseline['configuration'],'seed':seed}
 assert m['requested_seed']==m['actual_seed']==m['effective_seed']==seed and m['seed_setup_completed']
 assert m['seeded_configurations']=={'round52_nn_train':seed,'round47_nn_train':seed}
 assert m['baseline_seed']==42 and m['changed_configuration_fields']==['seed']
 assert m['seed_entrypoint_source_sha256']==WRAPPER_SHA
 assert m['source_sha256']==m['underlying_trainer_source_sha256']==previous.TRAINER_SHA
 assert m['architecture_source_sha256']==old.ARCHITECTURE_SHA
 assert m['seed_source_sha256']=={rel(p):sha(p) for p in (WRAPPER,previous.TRAINER,old.ARCHITECTURE)}
 assert m['input_sha256']==baseline['input_sha256']==sha(path)
 assert m['numeric_count']==144 and m['cardinalities']==baseline['cardinalities']
 assert m['fit_rows']==baseline['fit_rows'] and m['query_rows']==baseline['query_rows']
 assert m['saved_model_reload_exact'] and m['reload_rows']==baseline['query_rows']
 assert not m['query_labels_available'] and not m['query_labels_used'] and not m['extra_sample_weights'] and not m['price_or_ratio_clipping']
 assert m['price_factor_uses_fit_labels'] and not m['price_factor_provenance_verified_in_trainer']
 assert [e['epoch'] for e in m['epochs']]==list(range(1,31))
 for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary',
  'initial_training_median_ratio','initial_head_bias','price_factor_formula','train_price_factor_construction',
  'query_price_factor_construction','output_units','prediction_units'):
  assert m[key]==baseline[key],key
 return old.vector(out/'predictions_twd.npy',m['query_rows']),m

def run():
 tick=time.monotonic()
 assert sha(WRAPPER)==WRAPPER_SHA and sha(previous.TRAINER)==previous.TRAINER_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
 p48=read(ROOT/'outputs/round48/plan.json');p52=read(ROOT/'outputs/round52/plan.json')
 records={r:read(ROOT/f'outputs/round{r}/results.json') for r in (48,49,50,52,53,54,60)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 Xall,yall=old.previous.load_training_rows(p48)
 protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 source={rel(p):sha(p) for p in (Path(__file__),WRAPPER,previous.TRAINER,old.ARCHITECTURE,Path(previous.__file__),Path(old.__file__))}
 result={'round':61,'candidate':'R61','other_fixed_candidate':'R61_ordinary','status':'running','formula':FORMULA,
  'seeds':list(SEEDS),'new_seeds':list(NEW_SEEDS),'methods':METHODS,'new_NN_fits':0,'checked_stages':[],'stages':{},
  'source_sha256':source,'protected_before':protected,'new_submission_generated':False,'fresh_blind_validation':False}
 write(OUT/'plan.json',{'round':61,'formula':FORMULA,'other_candidate':'same formula with ordinary R52 priors',
  'seeds':list(SEEDS),'new_seeds':list(NEW_SEEDS),'new_NN_fits':12,'seed42_retrained':False,'area_NN_retrained':False,
  'fit_order':'stage then seed137/2026 then ordinary/recency','seed_selection':False,'fixed_arithmetic_mean':True,
  'stage_order':list(STAGES),'source_sha256':source,'text_plan_sha256':sha(ROOT/'reports/round61_plan.md'),
  'split_contract':p48['split_contract'],'query_labels_in_trainer':False,'three_periods_regardless_of_scores':True,
  'prior_fold_seed_remains_42':True,'encoder_and_prior_refit':False,'epochs':30,'batch_size':2048,'new_gate':None})
 inputs={}
 def checked(name,digest):
  assert name not in inputs or inputs[name]==digest
  assert sha(ROOT/name)==digest,name
  inputs[name]=digest;return ROOT/name
 def vector(name,digest,n):return old.vector(checked(name,digest),n)
 for s in STAGES:
  start=time.monotonic();folder=OUT/s;folder.mkdir();b=p48['bindings'][s];bs=p52['bindings'][s]
  a48=records[48]['stages'][s];a54=records[54]['stages'][s];ind={}
  for k in ('fit','query'):
   name=f'outputs/round48/{s}/{k}_indices.npy';ind[k]=np.load(checked(name,a48['artifact_sha256'][name]),allow_pickle=False)
   assert old.original.value_sha(ind[k],'<i8')==p48['split_contract'][s][k+'_indices_content_sha256']
   old.previous.save_array(folder/f'{k}_indices.npy',ind[k])
  fi,qi=ind['fit'],ind['query'];n=len(qi);X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True)
  assert not np.intersect1d(fi,qi).size
  assert spatial.frame_sha(X)==a48['original_frame_sha256']['fit_X'] and spatial.frame_sha(Xq)==a48['original_frame_sha256']['query_X']
  data={};baselines={};native={method:{} for method in METHODS};native_paths={method:{} for method in METHODS}
  six={};prior_folds={};provenance={}
  for method,r in METHODS.items():
   a=records[r]['stages'][s]
   for k in ('fit','query'):
    name=f'outputs/round{r}/{s}/{k}_indices.npy'
    assert np.array_equal(ind[k],np.load(checked(name,a['artifact_sha256'][name]),allow_pickle=False))
   name=f'outputs/round{r}/{s}/model/training.json';baseline=json.loads(checked(name,a['artifact_sha256'][name]).read_text(encoding='utf8'))
   assert baseline['status']=='completed' and baseline['configuration']['seed']==42 and baseline['saved_model_reload_exact']
   assert baseline['source_sha256']==previous.TRAINER_SHA and baseline['architecture_source_sha256']==old.ARCHITECTURE_SHA
   name=f'outputs/round{r}/{s}/prepared.npz';data[method]=checked(name,a['artifact_sha256'][name]);assert baseline['input_sha256']==sha(data[method])
   baselines[method]=baseline
   name=f'outputs/round{r}/{s}/prior_preparation.json';prep=read(checked(name,a['artifact_sha256'][name]));provenance[method]=prep
   assert prep['fit_labels_used_for_prior'] and not prep['query_labels_used'] and not prep['encoder_refit']
   for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary'):
    assert prep[key]==baseline[key]
   name=f'outputs/round{r}/{s}/prior_fold_ids.npy';prior_folds[method]=np.load(checked(name,a['artifact_sha256'][name]),allow_pickle=False)
   names=('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities')
   with np.load(data[method],allow_pickle=False) as z:
    assert set(z.files)==set(names)|{'train_price_factor_million','query_price_factor_million'}
    assert z['train_num'].shape==(len(fi),144) and z['query_num'].shape==(n,144)
    assert z['train_y_million'].tobytes()==np.asarray(yall[fi]/1e6,dtype=np.float32).tobytes()
    six[method]={name:old.original.value_sha(z[name],'<i8' if z[name].dtype==np.int64 else '<f4') for name in names}
   name=f'outputs/round{r}/{s}/model/predictions_twd.npy'
   native[method][42]=vector(name,a['artifact_sha256'][name],n);native_paths[method]['42']=name
  assert six['ordinary']==six['recency'] and np.array_equal(prior_folds['ordinary'],prior_folds['recency'])
  assert baselines['ordinary']['configuration']==baselines['recency']['configuration']
  for key in ('fit_target_sha256','effective_area_sha256','fit_unit_sha256','source_indices_sha256','fold_ids_sha256','group_keys_sha256'):
   assert provenance['ordinary']['prior_evidence'][key]==provenance['recency']['prior_evidence'][key]
  fit_records=[]
  for seed in NEW_SEEDS:
   for method in METHODS:
    out=folder/method/f'seed{seed}'/'model'
    pred,m=train(data[method],out,seed,baselines[method]);native[method][seed]=pred;native_paths[method][str(seed)]=rel(out/'predictions_twd.npy')
    fit_records.append({'method':method,'seed':seed,'fit_seconds':m['fit_seconds'],'parameter_count':m['parameter_count'],'saved_model_reload_exact':m['saved_model_reload_exact'],'reload_rows':m['reload_rows']})
    result['new_NN_fits']+=1
    print(json.dumps({'stage':s,'method':method,'seed':seed,'fit_seconds':m['fit_seconds'],'new_NN_fits_completed':result['new_NN_fits']}),flush=True)
  name=f'outputs/round54/{s}/model/predictions_twd.npy';area=vector(name,a54['artifact_sha256'][name],n)
  raw,unit=[vector(b[k],p48['input_sha256'][b[k]],n) for k in ('old_raw','old_unit')]
  with checked(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f:reference=pickle.load(f)
  with checked(b['state'],p48['input_sha256'][b['state']]).open('rb') as f:state=pickle.load(f)
  assert state['state_summary']['fit_X_sha256']==a48['original_frame_sha256']['fit_X'] and not state['state_summary']['query_used_in_build']
  assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
  geo=pd.read_pickle(checked(b['query_geo'],p48['input_sha256'][b['query_geo']]))
  assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
  log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n);ub=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
  def pipeline(prior_native,label):
   B=.75*(.5*raw+.5*(.5*area+.5*prior_native))+.25*unit
   p,diag,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
   branch,detail,sr=spatial.predict(state,Xq,geo,B,p,diag)
   mask=~detail['applied'].to_numpy(dtype=bool);old.exact(branch[mask],p[mask],'spatial fallback')
   value=np.median(np.column_stack([branch,log,ub]),axis=1)
   assert value.shape==(n,) and np.isfinite(value).all() and (value>0).all()
   return value,{'reference_report':rr,'spatial_report':sr}
  paths={};preds={};pipeline_reports={}
  for label,r,key in [('v15',54,'R54'),('v16',60,'R60'),('v13',53,'R53'),('v11',49,'R49'),('v12',50,'R50')]:
   name=f'outputs/round{r}/{s}/predictions.npy';preds[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n);paths[label]=name
  preds['v5']=vector(b['v5'],p48['input_sha256'][b['v5']],n);paths['v5']=b['v5']
  for method in METHODS:
   for seed in SEEDS:
    label=f'{method}_seed{seed}';preds[label],pipeline_reports[label]=pipeline(native[method][seed],label)
    if seed==42:
     control='v15' if method=='ordinary' else 'v16';old.exact(preds[label],preds[control],label+' full pipeline');paths[label]=paths[control]
    else:
     path=folder/(label+'_predictions.npy');old.previous.save_array(path,preds[label]);paths[label]=rel(path)
   mean=(native[method][42]+native[method][137]+native[method][2026])/3.0
   old.previous.save_array(folder/(method+'_native_mean.npy'),mean)
   label='R61' if method=='recency' else 'R61_ordinary';preds[label],pipeline_reports[label]=pipeline(mean,label)
   path=folder/('predictions.npy' if method=='recency' else 'ordinary_predictions.npy');old.previous.save_array(path,preds[label]);paths[label]=rel(path)
  metrics={label:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights']) for label,v in preds.items()}
  for label,r,key in [('v15',54,'R54'),('v16',60,'R60'),('v13',53,'R53'),('v11',49,'R49'),('v12',50,'R50'),('v5',53,'v5')]:
   assert metrics[label]==records[r]['stages'][s]['metrics'][key]
  differences=lambda control,candidate:old.previous.prior.log_run.differences(metrics[control],metrics[candidate])
  item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,
   'differences':{label:differences(label,'R61') for label in ('v16','v15','v13','v11','v12','v5','R61_ordinary')},
   'ordinary_mean_vs_v16':differences('v16','R61_ordinary'),
   'same_seed_method_differences':{str(seed):differences(f'ordinary_seed{seed}',f'recency_seed{seed}') for seed in SEEDS},
   'native_MAE':{method:{str(seed):float(np.abs(pred-yall[qi]).mean()) for seed,pred in members.items()} for method,members in native.items()},
   'new_NN_fits':4,'fits':fit_records,'fit_seconds':sum(x['fit_seconds'] for x in fit_records),
   'numeric_count':144,'original_six_arrays_identical_between_methods':True,'original_six_array_sha256':six['ordinary'],
   'prior_folds_identical_and_unchanged':True,'encoder_and_prior_refit':False,'seed42_full_pipelines_exact':True,
   'native_prediction_paths':native_paths,'prediction_paths':paths,'prediction_sha256':sha(folder/'predictions.npy'),
   'controls_full_metrics_exact':True,'pipeline_reports':pipeline_reports,'reference_predict_calls':8,'spatial_predict_calls':8,
   'elapsed_seconds':time.monotonic()-start,'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
  write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s)
  print(json.dumps({'stage':s,'R61_vs_v16':item['differences']['v16']['overall'],'ordinary_mean_vs_v16':item['ordinary_mean_vs_v16']['overall']}),flush=True)
 assert result['new_NN_fits']==12 and all(sha(ROOT/n)==h for n,h in source.items())
 result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,
  elapsed_seconds=time.monotonic()-tick,protected_after={n:sha(ROOT/n) for n in protected})
 assert result['protected_before']==result['protected_after'];write(OUT/'results.json',result)
 print(json.dumps({'status':result['status'],'new_NN_fits':12,'elapsed_seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':
 assert not OUT.exists(),'Existing output; no overwrite';OUT.mkdir();tick=time.monotonic();code=0;failure=None
 write(OUT/'started.json',{'started_at_utc':old.now(),'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 try:run()
 except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 raise SystemExit(code)
