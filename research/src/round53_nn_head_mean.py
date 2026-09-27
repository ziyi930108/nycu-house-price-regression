"""R53 fixed mean of two native NN price heads, then original v11 full median."""
from pathlib import Path
import json,pickle,hashlib,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
ROOT=old.ROOT;OUT=ROOT/'outputs/round53';STAGES=old.STAGES
FORMULA='median(spatial(reference(.75*(.5*oldraw+.5*(.5*R48_native_TWD+.5*R52_native_TWD))+.25*oldunit)),R46_recency_log_spatial,R40_unit_spatial)'
sha,read,write,spatial=old.sha,old.locked,old.write,old.spatial

def run():
 assert not OUT.exists(),'Existing output: no overwrite';OUT.mkdir();start=time.monotonic()
 protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
 write(OUT/'started.json',{'formula':FORMULA,'stage_order':list(STAGES),'source_sha256':sha(Path(__file__)),'plan_sha256':sha(ROOT/'reports/round53_plan.md'),'fits':0,'automatic_retry':False})
 p48=read(ROOT/'outputs/round48/plan.json');p52=read(ROOT/'outputs/round52/plan.json')
 records={r:read(ROOT/f'outputs/round{r}/results.json') for r in (48,49,50,52)}
 assert all(r['checked_stages']==list(STAGES) for r in records.values())
 Xall,yall=old.previous.load_training_rows(p48)
 inputs={};result={'round':53,'candidate':'R53','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},'fits':0,'model_predict_calls':0,'new_submission_generated':False,'fresh_blind_validation':False,'protected_before':protected,'source_sha256':sha(Path(__file__)),'plan_sha256':sha(ROOT/'reports/round53_plan.md')}
 def check(name,expected):
  path=ROOT/name;assert sha(path)==expected,name;inputs[name]=expected;return path
 def vector(name,expected,n):return old.vector(check(name,expected),n)
 for s in STAGES:
  tick=time.monotonic();folder=OUT/s;folder.mkdir();b=p48['bindings'][s];bs=p52['bindings'][s];a48=records[48]['stages'][s];a52=records[52]['stages'][s]
  ind={}
  for k in ('fit','query'):
   name=f'outputs/round48/{s}/{k}_indices.npy';ind[k]=np.load(check(name,a48['artifact_sha256'][name]),allow_pickle=False)
   name2=f'outputs/round52/{s}/{k}_indices.npy';assert np.array_equal(ind[k],np.load(check(name2,a52['artifact_sha256'][name2]),allow_pickle=False))
   assert old.original.value_sha(ind[k],'<i8')==p48['split_contract'][s][k+'_indices_content_sha256']
   old.previous.save_array(folder/f'{k}_indices.npy',ind[k])
  fi,qi=ind['fit'],ind['query'];X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True);n=len(qi)
  assert not np.intersect1d(fi,qi).size
  assert spatial.frame_sha(X)==a48['original_frame_sha256']['fit_X']==a52['original_frame_sha256']['fit_X']
  assert spatial.frame_sha(Xq)==a48['original_frame_sha256']['query_X']==a52['original_frame_sha256']['query_X']
  natives=[]
  for r,a in ((48,a48),(52,a52)):
   name=f'outputs/round{r}/{s}/model/training.json';meta=json.loads(check(name,a['artifact_sha256'][name]).read_text(encoding='utf-8'))
   assert meta['status']=='completed' and meta['prediction_units']=='TWD' and meta['query_rows']==n and meta['fit_rows']==len(fi)
   assert meta['configuration']['seed']==42 and meta['configuration']['epochs']==30 and meta['query_labels_used'] is False
   name=f'outputs/round{r}/{s}/model/predictions_twd.npy';natives.append(vector(name,a['artifact_sha256'][name],n))
  mean=.5*natives[0]+.5*natives[1]
  raw,unit=[vector(b[k],p48['input_sha256'][b[k]],n) for k in ('old_raw','old_unit')]
  B=.75*(.5*raw+.5*mean)+.25*unit
  with check(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f:reference=pickle.load(f)
  with check(b['state'],p48['input_sha256'][b['state']]).open('rb') as f:state=pickle.load(f)
  assert state['state_summary']['fit_X_sha256']==a48['original_frame_sha256']['fit_X'] and not state['state_summary']['query_used_in_build']
  assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
  geo=pd.read_pickle(check(b['query_geo'],p48['input_sha256'][b['query_geo']]))
  assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
  p,diag,reference_report=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
  branch,detail,spatial_report=spatial.predict(state,Xq,geo,B,p,diag)
  fallback=~detail['applied'].to_numpy(dtype=bool);old.exact(branch[fallback],p[fallback],'spatial fallback')
  log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n)
  unit_branch=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
  candidate=np.median(np.column_stack([branch,log,unit_branch]),axis=1)
  assert candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
  predictions={'R53':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
  for label,r in (('v11',49),('v12',50)):
   name=f'outputs/round{r}/{s}/predictions.npy';predictions[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n)
  score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
  metrics={k:score(v) for k,v in predictions.items()}
  for label,r,key in [('v11',49,'R49'),('v12',50,'R50'),('v5',49,'v5')]:assert metrics[label]==records[r]['stages'][s]['metrics'][key]
  for name,v in [('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)]:old.previous.save_array(folder/f'{name}.npy',v)
  for name,v in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
   with (folder/f'{name}.pkl').open('xb') as f:pickle.dump(v,f,protocol=5)
  item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R53']) for k in ('v11','v12','v5')},'reference_predict_calls':1,'spatial_predict_calls':1,'fits':0,'model_predict_calls':0,'controls_full_metrics_exact':True,'reference_report':reference_report,'spatial_report':spatial_report,'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,'artifact_sha256':{str(f.relative_to(ROOT)):sha(f) for f in folder.iterdir() if f.is_file()}}
  write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s)
  print(json.dumps({'stage':s,'MAE':{k:v['overall_mae'] for k,v in metrics.items()},'vs_v11':{k:item['differences']['v11'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
 result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,elapsed_seconds=time.monotonic()-start,protected_after={n:sha(ROOT/n) for n in protected})
 assert result['protected_before']==result['protected_after'];write(OUT/'results.json',result)
 print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__':
 tick=time.monotonic();code=0;failure=None
 try:run()
 except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 if OUT.exists():write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 raise SystemExit(code)
