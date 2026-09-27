"""Direct paired v11-v12 diagnostic on reused historical windows; no model changes."""
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import round48_nn_experiment as old
import v3_cluster_stability as boot
ROOT=old.ROOT; OUT=ROOT/'outputs/private_v11_v12_audit'; REPORT=ROOT/'reports/private_v11_v12_audit.md'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def run():
 assert not OUT.exists() and not REPORT.exists();OUT.mkdir();tick=time.monotonic()
 plan=old.locked(ROOT/'outputs/round48/plan.json')
 records={n:old.locked(ROOT/f'outputs/round{r}/results.json') for n,r in [('v11',49),('v12',50)]}
 X,y=old.previous.load_training_rows(plan); weights=plan['target_group_weights']
 result={'comparison':'candidate v11 minus control v12','status':'running','seed':20260925,'replicates':2000,'stage_order':list(old.STAGES),'fits':0,'model_predictions':0,'new_CSV':False,'fresh_blind_validation':False,'caution':'Post-selection conditional diagnostic on reused windows, not a Private probability or significance claim. Excludes model retraining, selection and future market shift.','source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),Path(boot.__file__).resolve()]},'stages':{}}
 write(OUT/'started.json',{k:v for k,v in result.items() if k!='stages'})
 lines=['# v11與v12的直接成對診斷','','沿既有建案及3月循環月份區塊抽樣，固定每項2,000次、seed20260925。差值為v11−v12，負值支持v11。這是已反覆使用窗口的事後條件性診斷，不是Private勝率、p值或新的盲測；不涵蓋選型、重訓及市場移轉。沒有重新訓練、預測、調權重或產CSV。','','| 期別 | 抽樣 | 指標 | 差值（元） | 2.5% | 97.5% |','|---|---|---|---:|---:|---:|']
 for si,s in enumerate(old.STAGES):
  fi,qi=[np.load(ROOT/f'outputs/round48/{s}/{k}_indices.npy',allow_pickle=False) for k in ('fit','query')]
  xf,xq=X.iloc[fi].reset_index(drop=True),X.iloc[qi].reset_index(drop=True)
  values=[];digests={}
  for n,r in [('v12',50),('v11',49)]:
   p=ROOT/f'outputs/round{r}/{s}/predictions.npy';assert sha(p)==records[n]['stages'][s]['prediction_sha256'];digests[n]=sha(p);values.append(np.load(p,allow_pickle=False))
  pred=np.column_stack(values);assert pred.shape==(len(qi),2) and np.isfinite(pred).all()
  labels=boot.group_labels(xf,xq);errors=np.abs(y[qi,None]-pred);point=boot.point_values(errors,labels,weights)
  for j,(n,k) in enumerate([('v12','R50'),('v11','R49')]):
   m=records[n]['stages'][s]['metrics'][k];assert abs(point[j]-m['overall_mae'])<1e-6 and abs(point[j+2]-m['weighted_mae'])<1e-6
  keys,fallback=boot.cluster_keys(xq,qi);months=boot.relative_months(xq.trade_day_index,plan['splits'][s][0]);d={'rows':len(qi),'dates':plan['splits'][s],'prediction_sha256':digests,'point_metrics_reproduced':True,'fallback':fallback}
  for fj,(name,fn,groups) in enumerate([('project',boot.bootstrap_projects,keys),('month',boot.bootstrap_months,months)]):
   samples,support=fn(errors,labels,groups,weights,np.random.default_rng(np.random.SeedSequence([20260925,si,fj])),2000)
   raw=boot.summarize(samples,point);summ={}
   for metric,m in raw.items():
    z={k:v for k,v in m.items() if k not in ['v2_mae','v3_mae','point_delta_v3_minus_v2']};z.update(v12_mae=m['v2_mae'],v11_mae=m['v3_mae'],point_delta_v11_minus_v12=m['point_delta_v3_minus_v2']);summ[metric]=z
    lines.append(f"| {s} | {name} | {metric} | {z['point_delta_v11_minus_v12']:,.2f} | {z['percentile_2_5']:,.2f} | {z['percentile_97_5']:,.2f} |")
   np.save(OUT/f'{s}_{name}_replicates.npy',samples,allow_pickle=False);d[name]={'summary':summ,'support':support}
  result['stages'][s]=d;print(json.dumps({'stage':s,'intervals':{k:v['summary'] for k,v in d.items() if k in ('project','month')}}),flush=True)
 result.update(status='completed',elapsed_seconds=time.monotonic()-tick);write(OUT/'results.json',result)
 lines+=['','各期兩種抽樣及兩項指標均完整列出；所有點估計與R49/R50保存結果一致。兩個候選曾各自相對v10做過診斷，本表另用v11/v12實際向量直接成對比較，沒有沿用其他候選的區間。','','[機器結果](../outputs/private_v11_v12_audit/results.json) · [來源](../src/private_v11_v12_audit.py)','']
 REPORT.write_text('\n'.join(lines),encoding='utf-8');print('completed',result['elapsed_seconds'],flush=True)
if __name__=='__main__':run()
