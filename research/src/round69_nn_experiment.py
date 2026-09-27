"""R69 paired prior-NN seed stability: 12 new fits; three fixed pipelines, no ensemble candidate."""
from pathlib import Path
import json, pickle, subprocess, sys, time, traceback
import numpy as np
import pandas as pd
import round65_nn_experiment as previous

old=previous.old; ROOT=old.ROOT; OUT=ROOT/'outputs/round69'; STAGES=old.STAGES
sha,read,write,spatial=old.sha,old.locked,old.write,old.spatial
rel=lambda p:Path(p).relative_to(ROOT).as_posix()
WRAPPER=ROOT/'src/round69_prior_seed_train.py'
WRAPPER_SHA='b27e5494e596d20cac96847a8929e3145bb9c5233f7c0096dab5be45f6bfd466'
SEEDS=(42,137,2026); NEW_SEEDS=(137,2026); METHODS={'A':63,'B':65}
PIPELINES=('A_oldbase','B_oldbase','B_R66base')
FORMULAS={
 'A_oldbase':'median(spatial(reference(.75*(.5*oldraw+.5*(.5*R54area+.5*A_native))+.25*oldunit)),recency_log,unit_spatial)',
 'B_oldbase':'median(spatial(reference(.75*(.5*oldraw+.5*(.5*R54area+.5*B_native))+.25*oldunit)),recency_log,unit_spatial)',
 'B_R66base':'median(spatial(reference(.75*(.5*R54area+.5*B_native)+.25*oldunit)),recency_log,unit_spatial)'}


def train(path,out,seed,baseline):
 out.mkdir(parents=True)
 command=[str(old.TORCH_PYTHON),'-u','-B','-X','utf8',str(WRAPPER),'--data',str(path),'--out',str(out),'--seed',str(seed)]
 write(out/'fit_started.json',{'command':command,'seed':seed,'started_at_utc':old.now(),'input_sha256':baseline['input_sha256'],'automatic_retry':False})
 tick=time.monotonic()
 with (out/'trainer.log').open('x',encoding='utf8') as f:
  done=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
 write(out/'execution.json',{'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-tick,'command':command,'automatic_retry':False})
 assert done.returncode==0,'Trainer failed; preserve outputs, no automatic retry'
 m=json.loads((out/'training.json').read_text(encoding='utf8'))
 assert m['status']=='completed' and m['configuration']=={**baseline['configuration'],'seed':seed}
 assert m['requested_seed']==m['actual_seed']==m['effective_seed']==seed and m['seed_setup_completed']
 assert m['seeded_configurations']=={'round63_recency_weighted_prior_train':seed,'round63_BASELINE_CONFIG':seed,'round47_nn_train':seed}
 assert m['baseline_seed']==42 and m['seed_changed_configuration_fields']==['seed']
 assert m['seed_entrypoint_source_sha256']==WRAPPER_SHA
 assert m['source_sha256']==m['underlying_trainer_source_sha256']==previous.TRAINER_SHA
 assert m['architecture_source_sha256']==old.ARCHITECTURE_SHA
 assert m['seed_source_sha256']=={rel(p):sha(p) for p in (WRAPPER,previous.TRAINER,old.ARCHITECTURE,previous.PARENT)}
 assert m['input_sha256']==baseline['input_sha256']==sha(path)
 assert m['numeric_count']==baseline['numeric_count'] and m['cardinalities']==baseline['cardinalities']
 assert m['fit_rows']==baseline['fit_rows'] and m['query_rows']==baseline['query_rows']
 assert m['saved_model_reload_exact'] and m['reload_rows']==baseline['query_rows']
 assert not m['query_labels_available'] and not m['query_labels_used'] and m['extra_sample_weights'] and not m['price_or_ratio_clipping']
 assert not m['sample_weights_recomputed_in_trainer'] and not m['sample_weights_batch_renormalized']
 assert m['price_factor_uses_fit_labels'] and not m['price_factor_provenance_verified_in_trainer']
 assert [e['epoch'] for e in m['epochs']]==list(range(1,31))
 for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','train_sample_weight_sha256',
  'initial_training_median_ratio','initial_head_bias','price_factor_formula','train_price_factor_construction',
  'query_price_factor_construction','output_units','prediction_units','changed_configuration_fields',
  'added_configuration_fields','loss_configuration_overrides','parent_trainer_source_sha256'):
  assert m[key]==baseline[key],key
 for key in ('rows','min','max'):assert m['train_sample_weight_summary'][key]==baseline['train_sample_weight_summary'][key]
 for key in ('mean','sum','sum_squares','effective_sample_size'):
  assert np.isclose(m['train_sample_weight_summary'][key],baseline['train_sample_weight_summary'][key],rtol=1e-12,atol=1e-12),key
 for split in ('train','query'):
  for key in ('rows','min','max'):assert m['price_factor_million_summary'][split][key]==baseline['price_factor_million_summary'][split][key]
  assert np.isclose(m['price_factor_million_summary'][split]['mean'],baseline['price_factor_million_summary'][split]['mean'],rtol=1e-12,atol=1e-12)
 return old.vector(out/'predictions_twd.npy',m['query_rows']),m


def run():
 tick=time.monotonic(); inputs={}
 assert sha(WRAPPER)==WRAPPER_SHA and sha(previous.TRAINER)==previous.TRAINER_SHA
 assert sha(previous.PARENT)==previous.PARENT_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
 def checked(name,digest):
  name=name.replace('\\','/'); assert name not in inputs or inputs[name]==digest,name
  if name not in inputs:assert sha(ROOT/name)==digest,name
  inputs[name]=digest;return ROOT/name
 def small(name):
  path=ROOT/name;value=read(path);inputs[name]=sha(path);inputs[rel(path.with_suffix('.sha256'))]=sha(path.with_suffix('.sha256'));return value
 def vector(name,digest,n):return old.vector(checked(name,digest),n)
 p48=small('outputs/round48/plan.json');p52=small('outputs/round52/plan.json')
 records={r:small(f'outputs/round{r}/results.json') for r in (48,54,63,65,66)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 source=dict(records[66]['source_sha256'])
 for p in (Path(__file__),WRAPPER,previous.TRAINER,previous.PARENT,old.ARCHITECTURE,Path(previous.__file__)):
  name,h=rel(p),sha(p);assert name not in source or source[name]==h;source[name]=h
 assert all(sha(ROOT/n)==h for n,h in source.items()),'Inherited source changed'
 document=ROOT/'reports/round69_plan.md';document_sha=sha(document);inputs[rel(document)]=document_sha
 inherited={n:p48['input_sha256'][n] for n in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT)}
 for n in ('reports/feature_schema.json',old.previous.KT):checked(n,inherited[n])
 write(OUT/'plan.json',{'round':69,'formulas':FORMULAS,'seeds':list(SEEDS),'new_seeds':list(NEW_SEEDS),
  'methods':METHODS,'pipelines':list(PIPELINES),'new_NN_fits':12,'seed42_retrained':False,'area_NN_retrained':False,
  'fit_order':'stage then seed137/2026 then A/B','seed_selection':False,'ensemble_candidate':False,
  'stage_order':list(STAGES),'source_sha256':source,'text_plan_sha256':document_sha,'split_contract':p48['split_contract'],
  'inherited_data_sha256':inherited,'data_cache_hashes_inherited_not_recomputed':True,
  'query_labels_in_trainer':False,'three_periods_regardless_of_scores':True,'prior_fold_seed_remains_42':True,
  'encoder_prior_weights_refit':False,'epochs':30,'batch_size':2048,'new_gate':None,'final_fit':False,'CSV':False})
 Xall,yall=old.previous.load_training_rows(p48)
 result={'round':69,'candidate':'R69_seed_stability_only','status':'running','formulas':FORMULAS,
  'seeds':list(SEEDS),'new_seeds':list(NEW_SEEDS),'methods':METHODS,'pipelines':list(PIPELINES),
  'new_NN_fits':0,'checked_stages':[],'stages':{},'source_sha256':source,'protected_before':protected,
  'new_submission_generated':False,'fresh_blind_validation':False,'ensemble_candidate':False,'formal_adoption_decision':None,
  'plan_sha256':document_sha,'machine_plan_sha256':sha(OUT/'plan.json'),'inherited_data_sha256':inherited,
  'data_cache_hashes_inherited_not_recomputed':True}
 for stage in STAGES:
  start=time.monotonic();folder=OUT/stage;folder.mkdir();b=p48['bindings'][stage];bs=p52['bindings'][stage]
  a48=records[48]['stages'][stage];ind={}
  for kind in ('fit','query'):
   name=f'outputs/round48/{stage}/{kind}_indices.npy';ind[kind]=np.load(checked(name,a48['artifact_sha256'][name]),allow_pickle=False)
   c=p48['split_contract'][stage];v=ind[kind]
   assert v.dtype==np.int64 and v.shape==(c[kind+'_rows'],) and (np.diff(v)>0).all() and v.min()>=0 and v.max()<len(Xall)
   assert old.original.value_sha(v,'<i8')==c[kind+'_indices_content_sha256']
   old.previous.save_array(folder/f'{kind}_indices.npy',v)
   for r in (54,63,65,66):
    name=f'outputs/round{r}/{stage}/{kind}_indices.npy';assert np.array_equal(v,np.load(checked(name,records[r]['stages'][stage]['artifact_sha256'][name]),allow_pickle=False))
  fi,qi=ind['fit'],ind['query'];n=len(qi);X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True)
  assert not np.intersect1d(fi,qi).size
  frames={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
  assert all(all(frames[k]==records[r]['stages'][stage]['original_frame_sha256'][k] for k in frames) for r in (48,63,65,66))
  data={};baselines={};native={m:{} for m in METHODS};native_paths={m:{} for m in METHODS};array_hashes={};prefix={};preparations={}
  expected={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million','train_sample_weight'}
  for method,r in METHODS.items():
   a=records[r]['stages'][stage];width=144 if method=='A' else 145
   name=f'outputs/round{r}/{stage}/model/training.json';baseline=json.loads(checked(name,a['artifact_sha256'][name]).read_text(encoding='utf8'))
   assert baseline['status']=='completed' and baseline['configuration']['seed']==42 and baseline['saved_model_reload_exact']
   assert baseline['source_sha256']==previous.TRAINER_SHA and baseline['architecture_source_sha256']==old.ARCHITECTURE_SHA
   assert baseline['numeric_count']==width and baseline['fit_rows']==len(fi) and baseline['query_rows']==n
   name=f'outputs/round{r}/{stage}/prepared.npz';data[method]=checked(name,a['artifact_sha256'][name]);assert baseline['input_sha256']==inputs[name]
   baselines[method]=baseline
   name=f'outputs/round{r}/{stage}/preparation.json';prep=read(checked(name,a['artifact_sha256'][name]));preparations[method]=prep
   assert prep['fit_labels_used_for_prior'] and not prep['query_labels_used'] and not prep['encoder_refit'] and not prep['prior_refit']
   with np.load(data[method],allow_pickle=False) as z:
    assert set(z.files)==expected and z['train_num'].dtype==z['query_num'].dtype==np.float32
    assert z['train_num'].shape==(len(fi),width) and z['query_num'].shape==(n,width)
    assert z['train_y_million'].dtype==np.float32 and z['train_y_million'].tobytes()==np.asarray(yall[fi]/1e6,dtype=np.float32).tobytes()
    assert z['cardinalities'].tolist()==baseline['cardinalities']
    array_hashes[method]={key:old.original.value_sha(z[key],'<i8' if z[key].dtype==np.int64 else '<f4') for key in expected-{'train_num','query_num'}}
    prefix[method]={key:old.original.value_sha(z[key][:,:144],'<f4') for key in ('train_num','query_num')}
    for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
     assert z[key].dtype==np.float32 and old.original.value_sha(z[key],'<f4')==baseline[key+'_sha256']
   name=f'outputs/round{r}/{stage}/model/predictions_twd.npy';native[method][42]=vector(name,a['artifact_sha256'][name],n);native_paths[method]['42']=name
  assert array_hashes['A']==array_hashes['B'] and prefix['A']==prefix['B']
  assert baselines['A']['configuration']==baselines['B']['configuration']
  for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','train_sample_weight_sha256','initial_training_median_ratio','initial_head_bias','cardinalities'):
   assert baselines['A'][key]==baselines['B'][key],key
  assert preparations['A']['weights_use_fit_X_only'] and not preparations['A']['weights_use_targets'] and not preparations['A']['weights_use_query_dates']
  assert preparations['B']['original_144_prefix_bitwise_unchanged'] and preparations['B']['other_seven_arrays_bitwise_unchanged']
  assert not preparations['B']['sample_weights_rebuilt'] and preparations['B']['query_scaling_uses_fit_only']
  assert preparations['B']['R63_preparation_sha256']==sha(ROOT/f'outputs/round63/{stage}/preparation.json')
  fit_records=[]
  for seed in NEW_SEEDS:
   for method in METHODS:
    out=folder/method/f'seed{seed}'/'model';pred,m=train(data[method],out,seed,baselines[method])
    native[method][seed]=pred;native_paths[method][str(seed)]=rel(out/'predictions_twd.npy')
    fit_records.append({'method':method,'seed':seed,'fit_seconds':m['fit_seconds'],'parameter_count':m['parameter_count'],'saved_model_reload_exact':m['saved_model_reload_exact'],'reload_rows':m['reload_rows']})
    result['new_NN_fits']+=1
    print(json.dumps({'stage':stage,'method':method,'seed':seed,'fit_seconds':m['fit_seconds'],'new_NN_fits_completed':result['new_NN_fits']}),flush=True)
  name=f'outputs/round54/{stage}/model/predictions_twd.npy';area=vector(name,records[54]['stages'][stage]['artifact_sha256'][name],n)
  raw,unit=[vector(b[k],p48['input_sha256'][b[k]],n) for k in ('old_raw','old_unit')]
  with checked(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f:reference=pickle.load(f)
  with checked(b['state'],p48['input_sha256'][b['state']]).open('rb') as f:state=pickle.load(f)
  assert state['state_summary']['fit_X_sha256']==frames['fit_X'] and not state['state_summary']['query_used_in_build']
  assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
  geo=pd.read_pickle(checked(b['query_geo'],p48['input_sha256'][b['query_geo']]))
  assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
  log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n);ub=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
  def pipeline(prior_native,mode):
   mean=.5*area+.5*prior_native
   B=.75*mean+.25*unit if mode=='B_R66base' else .75*(.5*raw+.5*mean)+.25*unit
   p,diag,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
   branch,detail,sr=spatial.predict(state,Xq,geo,B,p,diag)
   fallback=~detail['applied'].to_numpy(dtype=bool);old.exact(branch[fallback],p[fallback],'spatial fallback')
   value=np.median(np.column_stack([branch,log,ub]),axis=1)
   assert value.shape==(n,) and np.isfinite(value).all() and (value>0).all()
   return value,{'reference_report':rr,'spatial_report':sr}
  preds={};paths={};pipeline_reports={}
  controls={'v17':(63,'R63'),'R65':(65,'R65'),'v18':(66,'R66')}
  for label,(r,key) in controls.items():
   name=f'outputs/round{r}/{stage}/predictions.npy';preds[label]=vector(name,records[r]['stages'][stage]['prediction_sha256'],n);paths[label]=name
  for seed in SEEDS:
   for mode in PIPELINES:
    method=mode[0];label=f'{mode}_seed{seed}';preds[label],pipeline_reports[label]=pipeline(native[method][seed],mode)
    if seed==42:old.exact(preds[label],preds[{'A_oldbase':'v17','B_oldbase':'R65','B_R66base':'v18'}[mode]],label+' saved control')
    path=folder/(label+'_predictions.npy');old.previous.save_array(path,preds[label]);paths[label]=rel(path)
  metrics={label:old.original.score(yall[qi],v,X,Xq,p48['splits'][stage][0],p48['target_group_weights']) for label,v in preds.items()}
  for label,(r,key) in controls.items():assert metrics[label]==records[r]['stages'][stage]['metrics'][key]
  differences=lambda control,candidate:old.previous.prior.log_run.differences(metrics[control],metrics[candidate])
  paired={str(seed):{
   'v18style_vs_v17style':differences(f'A_oldbase_seed{seed}',f'B_R66base_seed{seed}'),
   'prior_input_at_oldbase':differences(f'A_oldbase_seed{seed}',f'B_oldbase_seed{seed}'),
   'base_change_at_B145':differences(f'B_oldbase_seed{seed}',f'B_R66base_seed{seed}')}
   for seed in SEEDS}
  item={'stage':stage,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][stage],
   'metrics':metrics,'same_seed_method_differences':paired,'new_NN_fits':4,'fits':fit_records,
   'fit_seconds':sum(x['fit_seconds'] for x in fit_records),'numeric_counts':{'A':144,'B':145},
   'original_frame_sha256':frames,'original_144_prefix_identical':True,'original_144_prefix_sha256':prefix['A'],
   'other_seven_arrays_identical':True,'unchanged_seven_array_sha256':array_hashes['A'],
   'encoder_prior_weights_refit':False,'prior_folds_reused_unchanged':True,'prior_fold_seed':42,
   'seed42_full_pipelines_exact':True,'controls_full_metrics_exact':True,'pipeline_reports':pipeline_reports,
   'native_prediction_paths':native_paths,'prediction_paths':paths,
   'prediction_sha256':{label:sha(ROOT/path) for label,path in paths.items()},
   'reference_predict_calls':9,'spatial_predict_calls':9,'query_labels_used_only_by_scorer':True,
   'elapsed_seconds':time.monotonic()-start,'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
  write(folder/'results.json',item);result['stages'][stage]=item;result['checked_stages'].append(stage)
  print(json.dumps({'stage':stage,'paired_overall_pct':{seed:{key:d['overall']['relative_change_pct'] for key,d in v.items()} for seed,v in paired.items()}}),flush=True)
 assert result['new_NN_fits']==12 and all(sha(ROOT/n)==h for n,h in source.items()) and sha(document)==document_sha
 result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,
  elapsed_seconds=time.monotonic()-tick,protected_after={n:sha(ROOT/n) for n in protected})
 assert result['protected_before']==result['protected_after'];write(OUT/'results.json',result)
 print(json.dumps({'status':result['status'],'new_NN_fits':12,'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
 assert sys.argv[1:]==['--run'],'Only --run is supported'
 assert not OUT.exists(),'Existing output; no overwrite';OUT.mkdir();tick=time.monotonic();code=0;failure=None;started=old.now()
 write(OUT/'started.json',{'started_at_utc':started,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 try:run()
 except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 write(OUT/'execution.json',{'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
  'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 raise SystemExit(code)
