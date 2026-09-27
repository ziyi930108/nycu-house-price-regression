"""R71 missing A144/new-base cell; seed42 is the sole candidate, other seeds diagnostics."""
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round66_nn_branch_ablation as parent

old,ROOT,STAGES=parent.old,parent.ROOT,parent.STAGES
OUT=ROOT/'outputs/round71';DOCUMENT=ROOT/'reports/round71_plan.md'
sha,read,write,rel=old.sha,old.locked,old.write,old.rel
SEEDS=(42,137,2026);PRIMARY_SEED=42
NATIVE_FORMULA='.5*saved_R54_area_native_TWD+.5*saved_A144_prior_native_TWD'
BASE_FORMULA='.75*(.5*saved_R54_area_native_TWD+.5*saved_A144_prior_native_TWD)+.25*old_unit_LGB_TWD'
FORMULA='median(spatial(reference('+BASE_FORMULA+')),R46_recency_log_spatial,R40_unit_spatial)'
CONTROLS=('v18','v17','R65','v16','v15','v13','v11','v12','v5','R62')
MODES=('A_oldbase','B_oldbase','B_R66base')


def contrast(minuend,subtrahend):
 """All outputs are MAE differences in TWD: minuend minus subtrahend."""
 assert minuend['rows']==subtrahend['rows']
 out={key:minuend[key]-subtrahend[key] for key in ('overall_mae','weighted_mae')}
 for family in ('strata','forecast_quarters'):
  assert set(minuend[family])==set(subtrahend[family]);out[family]={}
  for key,a in minuend[family].items():
   b=subtrahend[family][key];assert a['rows']==b['rows']
   if a['mae'] is None or b['mae'] is None:
    assert a['mae'] is None and b['mae'] is None;out[family][key]=None
   else:out[family][key]=a['mae']-b['mae']
 return out


def subtract_contrasts(first,second):
 if isinstance(first,dict):
  assert set(first)==set(second);return {key:subtract_contrasts(first[key],second[key]) for key in first}
 if first is None:assert second is None;return None
 assert second is not None;return first-second


def run():
 start=time.monotonic();inputs={}
 def checked(name,digest):
  name=name.replace('\\','/');assert name not in inputs or inputs[name]==digest,name
  if name not in inputs:assert sha(ROOT/name)==digest,name
  inputs[name]=digest;return ROOT/name
 def small(name):
  path=ROOT/name;v=read(path);inputs[name]=sha(path);inputs[rel(path.with_suffix('.sha256'))]=sha(path.with_suffix('.sha256'));return v
 def artifact(record,name):
  m={k.replace('\\','/'):v for k,v in record['artifact_sha256'].items()};return checked(name,m[name])
 def vector(name,digest,n):return old.vector(checked(name,digest),n)
 p48,p52=(small(f'outputs/round{r}/plan.json') for r in (48,52))
 records={r:small(f'outputs/round{r}/results.json') for r in (54,63,66,69)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 r69,r66=records[69],records[66]
 assert r69['status']=='three_periods_completed_no_adoption_decision' and r69['new_NN_fits']==12
 assert r69['seeds']==list(SEEDS) and r69['methods']=={'A':63,'B':65} and not r69['ensemble_candidate']
 assert r66['candidate']=='R66' and r66['formula']==parent.FORMULA
 source=dict(r69['source_sha256']);source[rel(Path(__file__))]=sha(Path(__file__))
 assert all(sha(ROOT/n)==h for n,h in source.items()),'Inherited source changed'
 doc_sha=sha(DOCUMENT);inputs[rel(DOCUMENT)]=doc_sha
 protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 inherited={n:p48['input_sha256'][n] for n in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT)}
 for n in ('reports/feature_schema.json',old.previous.KT):checked(n,inherited[n])
 write(OUT/'plan.json',{'round':71,'candidate':'R71','formula':FORMULA,'base_formula':BASE_FORMULA,
  'native_mean_formula':NATIVE_FORMULA,'seeds':list(SEEDS),'primary_seed':PRIMARY_SEED,'stage_order':list(STAGES),
  'primary_control':'v18','source_sha256':source,'text_plan_sha256':doc_sha,'split_contract':p48['split_contract'],
  'inherited_data_sha256':inherited,'data_cache_hashes_inherited_not_recomputed':True,
  'new_NN_fits':0,'model_predict_calls':0,'reference_predict_calls_per_stage':3,'spatial_predict_calls_per_stage':3,
  'encoder_prior_weights_refit':False,'seed_selection':False,'seed_average':False,'three_periods_regardless_of_scores':True,
  'primary_difference_orientation':'Anew42 minus Bnew42; negative favors R71',
  'paired_input_effect_orientation':'Bnew minus Anew; negative favors B145',
  'interaction_formula':'(Bnew-Anew)-(Bold-Aold) in TWD MAE; not prediction values',
  'new_gate':None,'final_fit':False,'CSV':False})
 Xall,yall=old.previous.load_training_rows(p48)
 result={'round':71,'candidate':'R71','status':'running','formula':FORMULA,'base_formula':BASE_FORMULA,
  'native_mean_formula':NATIVE_FORMULA,'seeds':list(SEEDS),'primary_seed':PRIMARY_SEED,'primary_control':'v18',
  'checked_stages':[],'stages':{},'source_sha256':source,'plan_sha256':doc_sha,'machine_plan_sha256':sha(OUT/'plan.json'),
  'protected_before':protected,'new_NN_fits':0,'model_predict_calls':0,'new_submission_generated':False,
  'fresh_blind_validation':False,'seed_selection':False,'seed_average':False,
  'inherited_data_sha256':inherited,'data_cache_hashes_inherited_not_recomputed':True}
 for stage in STAGES:
  tick=time.monotonic();folder=OUT/stage;folder.mkdir();a69=r69['stages'][stage];a66=r66['stages'][stage]
  b,bs=p48['bindings'][stage],p52['bindings'][stage];indices={}
  for kind in ('fit','query'):
   name=f'outputs/round69/{stage}/{kind}_indices.npy';ind=np.load(artifact(a69,name),allow_pickle=False)
   c=p48['split_contract'][stage]
   assert ind.dtype==np.int64 and ind.shape==(c[kind+'_rows'],) and (np.diff(ind)>0).all()
   assert ind.min()>=0 and ind.max()<len(Xall) and old.original.value_sha(ind,'<i8')==c[kind+'_indices_content_sha256']
   for r in (54,63,66):
    name=f'outputs/round{r}/{stage}/{kind}_indices.npy';assert np.array_equal(ind,np.load(artifact(records[r]['stages'][stage],name),allow_pickle=False))
   indices[kind]=ind;old.previous.save_array(folder/f'{kind}_indices.npy',ind)
  fi,qi=indices['fit'],indices['query'];n=len(qi);assert not np.intersect1d(fi,qi).size
  X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
  frames={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
  assert all(all(frames[key]==record['original_frame_sha256'][key] for key in frames) for record in (a69,a66,records[63]['stages'][stage]))
  assert a69['numeric_counts']['A']==144 and a69['seed42_full_pipelines_exact'] and not a69['encoder_prior_weights_refit']
  paths=a69['native_prediction_paths']['A'];assert set(paths)=={'42','137','2026'}
  natives={};native_bindings={}
  for seed in SEEDS:
   name=paths[str(seed)]
   expected=f'outputs/round63/{stage}/model/predictions_twd.npy' if seed==42 else f'outputs/round69/{stage}/A/seed{seed}/model/predictions_twd.npy'
   assert name==expected,'Unexpected native branch'
   authority=r69['input_sha256'] if seed==42 else a69['artifact_sha256']
   if seed==42:assert authority[name]==records[63]['stages'][stage]['artifact_sha256'][name]
   natives[seed]=vector(name,authority[name],n);native_bindings[str(seed)]={'path':name,'sha256':authority[name]}
  name=f'outputs/round54/{stage}/model/predictions_twd.npy';area=old.vector(artifact(records[54]['stages'][stage],name),n)
  unit=vector(b['old_unit'],p48['input_sha256'][b['old_unit']],n)
  log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n);ub=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
  with checked(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f:reference=pickle.load(f)
  with checked(b['state'],p48['input_sha256'][b['state']]).open('rb') as f:state=pickle.load(f)
  assert state['state_summary']['fit_X_sha256']==frames['fit_X'] and not state['state_summary']['query_used_in_build']
  assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
  geo=pd.read_pickle(checked(b['query_geo'],p48['input_sha256'][b['query_geo']]))
  assert old.spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
  score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][stage][0],p48['target_group_weights'])
  controls={'v18':old.vector(artifact(a66,f'outputs/round66/{stage}/predictions.npy'),n)}
  for label,r,_ in parent.CONTROLS:
   name=f'outputs/round{r}/{stage}/predictions.npy';controls[label]=vector(name,r66['input_sha256'][name],n)
  controls['v5']=vector(b['v5'],p48['input_sha256'][b['v5']],n)
  metrics={label:score(controls[label]) for label in CONTROLS}
  assert all(metrics[label]==a66['metrics']['R66' if label=='v18' else label] for label in CONTROLS),'Full control metrics differ'
  old_metrics={};old_bindings={}
  for seed in SEEDS:
   for mode in MODES:
    label=f'{mode}_seed{seed}';name=a69['prediction_paths'][label];digest=a69['prediction_sha256'][label]
    assert digest==a69['artifact_sha256'][name],'Saved R69 prediction authority differs'
    old_metrics[label]=score(vector(name,digest,n));assert old_metrics[label]==a69['metrics'][label],label
    old_bindings[label]={'path':name,'sha256':digest}
  seed_metrics={};seed_paths={};pipeline_reports={};paired={}
  for seed in SEEDS:
   seed_folder=folder/f'seed{seed}';seed_folder.mkdir()
   mean=.5*area+.5*natives[seed];B=.75*mean+.25*unit
   p,diag,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
   branch,detail,sr=old.spatial.predict(state,Xq,geo,B,p,diag)
   fallback=~detail['applied'].to_numpy(dtype=bool);old.exact(branch[fallback],p[fallback],'spatial fallback')
   candidate=np.median(np.column_stack([branch,log,ub]),axis=1)
   assert candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
   seed_metrics[str(seed)]=score(candidate);seed_paths[str(seed)]=rel(seed_folder/'predictions.npy')
   for name,v in (('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)):
    old.previous.save_array(seed_folder/f'{name}.npy',v)
   old.exact(candidate,old.vector(seed_folder/'predictions.npy',n),'Saved seed candidate unchanged')
   for name,v in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
    with (seed_folder/f'{name}.pkl').open('xb') as f:pickle.dump(v,f,protocol=5)
   pipeline_reports[str(seed)]={'reference_report':rr,'spatial_report':sr}
   anew=seed_metrics[str(seed)];aold=old_metrics[f'A_oldbase_seed{seed}'];bold=old_metrics[f'B_oldbase_seed{seed}'];bnew=old_metrics[f'B_R66base_seed{seed}']
   new_effect=contrast(bnew,anew);old_effect=contrast(bold,aold)
   paired[str(seed)]={'input_effect_Bnew_minus_Anew_TWD':new_effect,'base_effect_Anew_minus_Aold_TWD':contrast(anew,aold),
    'input_effect_Bold_minus_Aold_TWD':old_effect,'interaction_TWD':subtract_contrasts(new_effect,old_effect)}
   if seed==PRIMARY_SEED:
    metrics['R71']=anew;old.previous.save_array(folder/'predictions.npy',candidate)
    old.exact(candidate,old.vector(folder/'predictions.npy',n),'Primary R71 is Anew seed42')
  item={'stage':stage,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][stage],
   'metrics':metrics,'differences':{label:old.previous.prior.log_run.differences(metrics[label],metrics['R71']) for label in CONTROLS},
   'primary_seed':42,'primary_difference_orientation':'Anew42 minus Bnew42; negative favors R71',
   'additional_seed_metrics':seed_metrics,'saved_R69_seed_metrics':old_metrics,'paired_seed_differences_TWD':paired,
   'paired_input_effect_orientation':'Bnew minus Anew; negative favors B145',
   'interaction_formula':'(Bnew-Anew)-(Bold-Aold) in TWD MAE; negative means larger B advantage at new base',
   'original_frame_sha256':frames,'native_seed_bindings':native_bindings,'saved_R69_prediction_bindings':old_bindings,
   'seed_prediction_paths':seed_paths,'seed_prediction_sha256':{seed:sha(ROOT/name) for seed,name in seed_paths.items()},
   'controls_full_metrics_exact':True,'R69_saved_full_seed_metrics_exact':True,'saved_candidates_bitwise_exact':True,
   'new_NN_fits':0,'model_predict_calls':0,'reference_predict_calls':3,'spatial_predict_calls':3,
   'old_pipeline_reference_spatial_calls':0,'query_labels_used_only_by_scorer':True,'pipeline_reports':pipeline_reports,
   'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
   'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
  write(folder/'results.json',item);result['stages'][stage]=item;result['checked_stages'].append(stage)
  print(json.dumps({'stage':stage,'R71_minus_v18':{key:item['differences']['v18'][key] for key in ('overall','weighted')},'paired_TWD':paired}),flush=True)
 assert all(sha(ROOT/n)==h for n,h in source.items()) and sha(DOCUMENT)==doc_sha
 result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,reference_predict_calls=9,spatial_predict_calls=9,
  protected_after={n:sha(ROOT/n) for n in protected},elapsed_seconds=time.monotonic()-start)
 assert result['protected_after']==protected;write(OUT/'results.json',result)
 print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if not args.run:parser.error('--run required')
 assert not OUT.exists(),'Existing output; no overwrite';OUT.mkdir();tick=time.monotonic();code=0;failure=None;started=old.now()
 write(OUT/'started.json',{'started_at_utc':started,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 try:run()
 except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 write(OUT/'execution.json',{'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
  'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 raise SystemExit(code)
