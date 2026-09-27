"""R55 fixed full-prediction half v13 plus half R54; CPU scoring only."""
from pathlib import Path
import argparse,contextlib,json,sys,time,traceback
import numpy as np
import round48_nn_experiment as old
ROOT=old.ROOT;OUT=ROOT/'outputs/round55';STAGES=old.STAGES
DOCUMENT=ROOT/'reports/round55_plan.md'
FORMULA='0.5 * saved R53(v13) complete unrounded TWD + 0.5 * saved R54 complete unrounded TWD'
sha,read,write=old.sha,old.locked,old.write
rel=lambda p:Path(p).resolve().relative_to(ROOT).as_posix()


def run():
 start=time.monotonic();inputs={}
 def checked(name,digest):
  name=Path(name).as_posix();path=ROOT/name
  assert name not in inputs or inputs[name]==digest,'Conflicting input digest'
  assert sha(path)==digest,name
  inputs[name]=digest;return path
 def small(name):
  path=ROOT/name;value=read(path);inputs[name]=sha(path);inputs[rel(path.with_suffix('.sha256'))]=sha(path.with_suffix('.sha256'));return value
 p48=small('outputs/round48/plan.json')
 records={r:small(f'outputs/round{r}/results.json') for r in (49,50,53,54)}
 assert all(r['status']=='three_periods_completed_no_adoption_decision' and r['checked_stages']==list(STAGES) for r in records.values())
 protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 source={rel(Path(__file__)):sha(Path(__file__)),rel(Path(old.__file__)):sha(Path(old.__file__))}
 write(OUT/'plan.json',{'round':55,'formula':FORMULA,'primary_control':'v13','stage_order':list(STAGES),
  'source_sha256':source,'text_plan_sha256':sha(DOCUMENT),'new_gate':None,'ratio_search':False,'new_CSV':False})
 Xall,yall=old.previous.load_training_rows(p48)
 for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT):inputs[name]=p48['input_sha256'][name]
 result={'round':55,'candidate':'R55','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},
  'source_sha256':source,'plan_sha256':sha(OUT/'plan.json'),'protected_before':protected,'fits':0,'new_NN_fits':0,
  'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,'new_submission_generated':False,'fresh_blind_validation':False}
 for stage in STAGES:
  tick=time.monotonic();folder=OUT/stage;folder.mkdir();a54=records[54]['stages'][stage];spec=p48['split_contract'][stage]
  assert all(r['stages'][stage]['dates']==p48['splits'][stage] and r['stages'][stage]['fit_rows']==spec['fit_rows'] and r['stages'][stage]['query_rows']==spec['query_rows'] for r in records.values())
  ind={}
  for key in ('fit','query'):
   arrays=[]
   for r in (53,54):
    name=f'outputs/round{r}/{stage}/{key}_indices.npy';artifacts={k.replace(chr(92),'/'):v for k,v in records[r]['stages'][stage]['artifact_sha256'].items()}
    a=np.load(checked(name,artifacts[name]),allow_pickle=False)
    assert a.dtype==np.int64 and a.shape==(spec[key+'_rows'],) and old.original.value_sha(a,'<i8')==spec[key+'_indices_content_sha256']
    arrays.append(a)
   assert np.array_equal(*arrays);ind[key]=arrays[0];old.previous.save_array(folder/(key+'_indices.npy'),arrays[0])
  fi,qi=ind['fit'],ind['query'];assert not np.intersect1d(fi,qi).size
  X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);n=len(qi)
  preds={}
  for label,r in (('v13',53),('R54',54),('v11',49),('v12',50)):
   name=f'outputs/round{r}/{stage}/predictions.npy';preds[label]=old.vector(checked(name,records[r]['stages'][stage]['prediction_sha256']),n)
  name=p48['bindings'][stage]['v5'];preds['v5']=old.vector(checked(name,p48['input_sha256'][name]),n)
  preds['R55']=.5*preds['v13']+.5*preds['R54']
  assert preds['R55'].dtype==np.float64 and preds['R55'].shape==(n,) and np.isfinite(preds['R55']).all() and (preds['R55']>0).all()
  metrics={k:old.original.score(yall[qi],v,X,Xq,p48['splits'][stage][0],p48['target_group_weights']) for k,v in preds.items()}
  for key in ('v13','R54','v11','v12','v5'):assert metrics[key]==a54['metrics'][key],key+' control full metrics differ'
  old.previous.save_array(folder/'predictions.npy',preds['R55'])
  item={'stage':stage,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][stage],
   'metrics':metrics,'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R55']) for k in ('v13','R54','v11','v12','v5')},
   'fits':0,'new_NN_fits':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,'controls_full_metrics_exact':True,
   'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
   'artifact_sha256':{rel(f):sha(f) for f in folder.iterdir() if f.is_file()}}
  write(folder/'results.json',item);result['stages'][stage]=item;result['checked_stages'].append(stage)
  print(json.dumps({'stage':stage,'overall_MAE':{k:m['overall_mae'] for k,m in metrics.items()},'vs_v13':{k:item['differences']['v13'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
 assert all(sha(ROOT/n)==h for n,h in source.items())
 result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,elapsed_seconds=time.monotonic()-start,protected_after={n:sha(ROOT/n) for n in protected})
 assert result['protected_after']==protected;write(OUT/'results.json',result)
 print(json.dumps({'status':result['status'],'results_sha256':sha(OUT/'results.json'),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args()
 assert not OUT.exists(),'Existing output: preserve evidence, no repeat';OUT.mkdir()
 tick=time.monotonic();started=old.now();code=0;failure=None
 write(OUT/'started.json',{'started_at_utc':started,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 with (OUT/'run.log').open('x',encoding='utf-8') as f:
  with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
   try:run()
   except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 event={'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,
  'source_sha256':sha(Path(__file__)),'log_path':rel(OUT/'run.log'),'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
 write(OUT/'execution.json',event);print(json.dumps(event),flush=True);raise SystemExit(code)
