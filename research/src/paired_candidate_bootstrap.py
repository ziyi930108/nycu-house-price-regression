"""Fixed paired bootstrap for saved candidate/control predictions on existing windows."""
from pathlib import Path
import argparse,json,time
import numpy as np
import round48_nn_experiment as old
import v3_cluster_stability as methods
ROOT=old.ROOT

def run(args):
 out=ROOT/f'outputs/round{args.round}/paired_vs_{args.control_name}';report=ROOT/f'reports/round{args.round}_paired_vs_{args.control_name}.md'
 assert not out.exists() and not report.exists();out.mkdir(parents=True);tick=time.monotonic()
 plan=old.locked(ROOT/'outputs/round48/plan.json')
 records=[old.locked(ROOT/f'outputs/round{n}/results.json') for n in (args.control_round,args.round)]
 X,y=old.previous.load_training_rows(plan);weights=plan['target_group_weights']
 result={'status':'running','candidate':args.candidate,'control':args.control_name,'difference':'candidate minus control','replicates':2000,'seed':20260925,'fresh_blind_validation':False,'fits':0,'model_predict_calls':0,'new_CSV':False,'source_sha256':old.sha(Path(__file__)),'methods_source_sha256':old.sha(Path(methods.__file__)),'stages':{},'caution':'Conditional on saved predictions and reused historical windows. Not Private probability or a p-value; excludes selection, model retraining and future-shift uncertainty.'}
 old.write(out/'started.json',{k:v for k,v in result.items() if k!='stages'})
 lines=[f'# {args.candidate}相對{args.control_name}的直接成對診斷','','固定沿既有建案及3月循環月份區塊抽樣，每項2,000次、seed20260925。負差值代表候選MAE較低。已重用的歷史窗口與保存模型為條件；不是Private勝率、p值或新盲測，不涵蓋多輪選型、重訓與未來市場移轉。全部兩種抽法、兩指標及三期都呈現。','','| 期別 | 抽法 | 指標 | 候選−對照（元） | 2.5% | 97.5% |','|---|---|---|---:|---:|---:|']
 for si,s in enumerate(old.STAGES):
  fi,qi=[np.load(ROOT/f'outputs/round48/{s}/{k}_indices.npy',allow_pickle=False) for k in ('fit','query')]
  for k,ind in [('fit',fi),('query',qi)]:assert old.original.value_sha(ind,'<i8')==plan['split_contract'][s][k+'_indices_content_sha256']
  xf,xq=X.iloc[fi].reset_index(drop=True),X.iloc[qi].reset_index(drop=True)
  vectors=[];hashes={}
  for label,n,r,key in [(args.control_name,args.control_round,records[0],args.control_key),(args.candidate,args.round,records[1],args.candidate)]:
   path=ROOT/f'outputs/round{n}/{s}/predictions.npy';assert old.sha(path)==r['stages'][s]['prediction_sha256']
   vectors.append(old.vector(path,len(qi)));hashes[label]=old.sha(path)
  labels=methods.group_labels(xf,xq);errors=np.abs(y[qi,None]-np.column_stack(vectors));point=methods.point_values(errors,labels,weights)
  for j,(r,key) in enumerate([(records[0],args.control_key),(records[1],args.candidate)]):
   m=r['stages'][s]['metrics'][key];assert abs(point[j]-m['overall_mae'])<=1e-6 and abs(point[j+2]-m['weighted_mae'])<=1e-6
  keys,fallback=methods.cluster_keys(xq,qi);months=methods.relative_months(xq.trade_day_index,plan['splits'][s][0]);stage={'rows':len(qi),'point_metrics_reproduced':True,'prediction_sha256':hashes,'fallback':fallback}
  for fj,(name,fn,groups) in enumerate([('project',methods.bootstrap_projects,keys),('month',methods.bootstrap_months,months)]):
   samples,support=fn(errors,labels,groups,weights,np.random.default_rng(np.random.SeedSequence([20260925,si,fj])),2000)
   summ={}
   for metric,m in methods.summarize(samples,point).items():
    z={k:v for k,v in m.items() if k not in ['v2_mae','v3_mae','point_delta_v3_minus_v2']};z.update(control_mae=m['v2_mae'],candidate_mae=m['v3_mae'],point_delta=m['point_delta_v3_minus_v2']);summ[metric]=z
    lines.append(f"| {s} | {name} | {metric} | {z['point_delta']:,.2f} | {z['percentile_2_5']:,.2f} | {z['percentile_97_5']:,.2f} |")
   stage[name]={'summary':summ,'support':support};np.save(out/f'{s}_{name}_replicates.npy',samples,allow_pickle=False)
  result['stages'][s]=stage;print(json.dumps({'stage':s,'intervals':{name:{metric:[v['point_delta'],v['percentile_2_5'],v['percentile_97_5']] for metric,v in stage[name]['summary'].items()} for name in ('project','month')}}),flush=True)
 result.update(status='completed',elapsed_seconds=time.monotonic()-tick);old.write(out/'results.json',result)
 lines+=['',f'[完整結果](../outputs/round{args.round}/paired_vs_{args.control_name}/results.json) · [可重跑來源](../src/paired_candidate_bootstrap.py)','']
 report.write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({'status':'completed','elapsed_seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);p.add_argument('--candidate',required=True);p.add_argument('--control-round',type=int,required=True);p.add_argument('--control-name',required=True);p.add_argument('--control-key',required=True);run(p.parse_args())
