"""One fixed paired bootstrap: saved R49 recency median versus saved R46 median/v10. No fitting or prediction."""
from __future__ import annotations
import argparse, copy, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import round38_stability as prior

original=prior.original
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round49/stability'
REPORT=ROOT/'reports/round49_stability_report.md'
STAGES=original.STAGES
CONTROL='recency_log_three_branch_median'
CANDIDATE='R49'
KT='outputs/round6/geography/development/train_source_cache_indices.npy'
RULES=copy.deepcopy(original.RULES)
RULES['delta']='MAE_R49_combined_median - MAE_R46_v10; negative means improvement'
AUTHORITIES={
 'outputs/round38_stability/plan.json':'23e0ae024a9c19a657fcdab3d36e4ece7c4439f86c187517ea45112988a67ab3',
 'outputs/round46/plan.json':'ae866e6cf95b57a28495eda74c40d4326fe4a2fa9a1e00afbdb973e9c6e59c7f',
 'outputs/round46/results.json':'338fd86a01399930d29055be07471823014f6c5ad8da0dd5ad1f7ed1fa96042e',
 'outputs/round49/input_contract.json':'2f40e1c1e1d0849bbf6ab65f3ba9c32474c414abc50ac789a17ea61f934699eb',
 'outputs/round49/results.json':'0e80d21ef7ebea8489da605d733bf4efbcdd61f4eddf6e05d8558d3c836a087a'}

PROTECTED=['outputs/submission_v3.csv','outputs/submission_v10_trial.csv','runs/versions.json']

need,sha,rel=original.need,original.sha,original.rel


def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')


def prepare():
 # Only saved vectors, same canonical inputs/indices, and their small authorities.
 inputs={}
 def bind(name,expected):
  actual=sha(ROOT/name);need(actual==expected,'Input SHA differs: '+name)
  need(name not in inputs or inputs[name]==actual,'Conflicting input SHA');inputs[name]=actual
 def small(name,expected):
  bind(name,expected);value=original.locked(ROOT/name)
  receipt=Path(name).with_suffix('.sha256').as_posix();inputs[receipt]=sha(ROOT/receipt);return value
 authority={n:small(n,h) for n,h in AUTHORITIES.items()}
 old=authority['outputs/round38_stability/plan.json']
 p46=authority['outputs/round46/plan.json'];r46=authority['outputs/round46/results.json']
 contract=authority['outputs/round49/input_contract.json'];r49=authority['outputs/round49/results.json']
 need(prior.runtime()==old['runtime'] and RULES['seed']==20260925 and RULES['replicates']==2000,'Fixed method/runtime changed')
 need(sha(Path(prior.__file__))=='d9ec7acac4f0e7cc6b507fca283c019ad4f00c21516917e6916e66c2e93f4fb5','R38 helper changed')
 need(sha(Path(original.__file__))==prior.ORIGINAL_SOURCE_SHA,'Original bootstrap helper changed')
 need(p46['splits']==old['splits'] and p46['split_contract']==old['split_contract'],'Window contracts differ')
 need(p46['target_group_weights']==old['fixed_target_group_weights'],'Fixed scoring weights differ')
 for number,result in [(46,r46),(49,r49)]:
  need(result['round']==number and result['status']=='three_periods_completed_no_adoption_decision'
       and result['checked_stages']==list(STAGES),'Incomplete saved experiment')
 need(r46['plan_sha256']==AUTHORITIES['outputs/round46/plan.json'],'R46 plan differs')
 need(r49['input_contract_sha256']==AUTHORITIES['outputs/round49/input_contract.json']
      and contract['stage_order']==list(STAGES),'R49 input contract differs')
 saved={}
 for stage in STAGES:
  a=small(f'outputs/round46/{stage}/results.json',r46['stage_result_sha256'][stage])
  name=f'outputs/round49/{stage}/results.json';b=small(name,sha(ROOT/name))
  need(b==r49['stages'][stage],'R49 stage record differs from bound complete results')
  need(a['stage']==b['stage']==stage and a['dates']==b['dates']==old['splits'][stage],'Stage/date mismatch')
  need(a['plan_sha256']==AUTHORITIES['outputs/round46/plan.json'],'Stage plan differs')
  for which in ('fit','query'):
   need(a[which+'_rows']==b[which+'_rows']==old['split_contract'][stage][which+'_rows'],'Stage rows differ')
   first=p46['bindings'][stage][which+'_indices'];second=contract['bindings'][stage][which+'_indices']
   need(second==f'outputs/round48/{stage}/{which}_indices.npy','R49 saved index path differs')
   bind(first,p46['input_sha256'][first]);bind(second,contract['input_sha256'][second])
   need(inputs[first]==inputs[second],'Saved ordered indices differ across methods')
  control=f'outputs/round46/{stage}/{CONTROL}.npy';candidate=f'outputs/round49/{stage}/predictions.npy'
  bind(control,a['artifact_sha256'][control]);bind(candidate,b['prediction_sha256'])
  need(contract['bindings'][stage]['v10']==control and contract['input_sha256'][control]==inputs[control]
       and a['metrics'][CONTROL]==b['metrics']['v10'],'Completed v10 control differs')
  saved[stage]=(a,b)
 for name in ['data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',KT]:
  expected=old['dependency_sha256'][name]
  need(p46['input_sha256'][name]==contract['input_sha256'][name]==expected,'Original canonical input differs');bind(name,expected)
 sources={rel(Path(__file__)):sha(Path(__file__)),rel(Path(prior.__file__)):sha(Path(prior.__file__)),rel(Path(original.__file__)):sha(Path(original.__file__))}
 protected={n:sha(ROOT/n) for n in PROTECTED}
 return old,saved,inputs,sources,protected


def summarize(boot,point):
 result=original.summarize(boot,point)
 for v in result.values():
  v['v10_mae']=v.pop('v2_mae');v['R49_mae']=v.pop('v3_mae')
  v['delta_R49_minus_v10']=v.pop('point_delta_v3_minus_v2')
 return result


def render(result):
 lines=['# R49 固定組合相對 v10 的成對穩定度診斷','',
  '比較 R49 保存的固定組合（NN area raw＋recency log＋原 unit 完整分支中位數）與 R46／v10 保存的 recency_log_three_branch_median。這是在 R49 三期點估計已知並已選作試交組合後進行的事後診斷，窗口已多次重用。此處使用 R49 實際向量重新抽樣，沒有沿用 R48 的區間。三期完整保留；本輪只對既有未取整預測抽樣，不重算融合、不載樹、不 fit／predict、不改候選設定、CSV、registry 或採用決定。','',
  '**差值 = R49 MAE − v10 MAE，負值較好。** 同一種抽法內兩向量使用完全相同樣本。固定 seed 20260925，每年度／抽法 2,000 次；下列為差值的 2.5%／97.5% 百分位與嚴格小於零的比例，單位新臺幣。','',
  '| 窗口 | 抽法 | 指標 | v10 MAE | R49 MAE | 差值 | 2.5% | 97.5% | 改善抽樣比例 | 有效／無效 |',
  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 def num(x):return 'null' if x is None else f'{x:,.2f}'
 for stage in STAGES:
  for family in ('project','month'):
   for metric,v in result['stages'][stage][family]['summary'].items():
    frac='null' if v['negative_delta_fraction'] is None else f"{v['negative_delta_fraction']:.2%}"
    lines.append(f"| {stage} | {family} | {metric} | {num(v['v10_mae'])} | {num(v['R49_mae'])} | {num(v['delta_R49_minus_v10'])} | {num(v['percentile_2_5'])} | {num(v['percentile_97_5'])} | {frac} | {v['valid_replicates']}/{v['invalid_replicates']} |")
 lines+=['','## 完整解讀','']
 for stage in STAGES:
  s=result['stages'][stage];parts=[]
  for family in ('project','month'):
   for metric,v in s[family]['summary'].items():
    lo,hi=v['percentile_2_5'],v['percentile_97_5']
    label='無法估計' if lo is None else ('完全低於零' if hi<0 else ('完全高於零' if lo>0 else '包含零'))
    parts.append(f'{family}/{metric} 區間{label}')
  lines.append(f"- {stage}："+'；'.join(parts)+'。')
 lines+=['','建案抽法沿原 city|district|project，在 seen／new_nonmissing／missing 分層內整群有放回抽樣；缺名使用原地址、道路、canonical 列號備援，不把所有缺名併成一群。overall 仍按交易列平均，weighted 仍為固定 test-X 分組權重。',
  '月份抽法由預測起日11日定義12個相對月，每次抽四個均勻起點、各連續三月並環繞；它假設這種短期時間區塊能代表相關性，不等於建案抽法。正權重分層分母為零時該次 weighted 記無效、不重抽；無資料則兩項均無效。每期／抽法 RNG 為 SeedSequence([20260925, stage_index, family_index])，順序 development/confirmation/auxiliary、project/month。','',
  '**這些是已保存預測與已用歷史窗口的條件性區間，不是 Private 改善機率或 p 值。** 不涵蓋多輪選型、重訓變異或未來市場移轉；兩種抽法、全部12項都呈現，沒有依區間再選比例。不使用 Public 成績選抽樣方法，也不把負差抽樣比例稱為 Private 勝出機率。','',
  '資料沿保存 canonical K 與同年度 fi/qi，不重算含價格去重。共用 train.pkl 會反序列化完整 train 標籤，但只有各期固定 query y 用於誤差與抽樣；fit X 只定原 seen/new 群組。各期兩向量的 overall／weighted 均與原完成結果核對至1e-6元。未讀 test 標籤、地圖或模型。','',
  f"唯一執行內部耗時 {result['elapsed_seconds']:.3f} 秒；原版本CSV與registry前後SHA一致。來源、實讀檔案SHA、群組支持、月份列數、全部抽樣與valid mask保存在結果；外部exit／耗時／log另存execution。",'',
  '[完整結果](../outputs/round49/stability/results.json) · [唯一執行紀錄](../outputs/round49/stability/execution.json) · [本次來源](../src/round49_stability.py) · [沿用抽樣方法](v3_cluster_stability_plan.md)','']
 return '\n'.join(lines)


def run():
 started=time.monotonic();need(not OUT.exists() and not REPORT.exists(),'No repeat/overwrite')
 OUT.mkdir(parents=True,exist_ok=False)
 write(OUT/'run_claim.json',{'created_at_utc':now(),'rules':RULES,'stage_order':list(STAGES),'source_sha256':sha(Path(__file__)),'automatic_retry':False})
 try:
  old,saved,inputs,sources,before=prepare()
  raw=pd.read_pickle(ROOT/'data_cache/train.pkl');features=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
  schema=original.read(ROOT/'reports/feature_schema.json')
  need(len(raw)==len(features)==455784 and raw.index.equals(pd.RangeIndex(455784)) and features.index.equals(raw.index) and list(features.columns)==schema['columns'],'Cache alignment/schema differs')
  k=original.npy(ROOT/KT);need(k.dtype==np.dtype('int64') and k.shape==(455255,) and np.array_equal(k,np.unique(k)) and ((k>=0)&(k<len(raw))).all(),'Canonical mapping invalid')
  X=features.iloc[k].reset_index(drop=True);y=raw['總價元'].to_numpy(dtype='float64')[k];del raw,features
  result={'round':49,'status':'completed','comparison':'R49 fixed combined median versus R46/v10 saved predictions','rules':RULES,'source_sha256':sources,'input_sha256':inputs,
   'runtime':prior.runtime(),'protected_before':before,'stages':{},'artifact_sha256':{},'fixed_target_group_weights':old['fixed_target_group_weights'],
   'fits':0,'model_loads':0,'predict_calls':0,'new_predictions_generated':False,'registry_changed':False,'selection_changed':False,
   'full_train_labels_deserialized':True,'only_fixed_query_labels_used_for_errors':True,'fresh_blind_validation':False,
   'caution':'Conditional on saved predictions and reused historical windows; not a p-value or Private probability; excludes selection/retraining/future-shift uncertainty.'}
  days=X.trade_day_index.to_numpy();weights=old['fixed_target_group_weights']
  for si,stage in enumerate(STAGES):
   start,end=old['splits'][stage];a,b=saved[stage]
   lo=(pd.Timestamp(start)-pd.Timestamp('2000-01-01')).days;hi=(pd.Timestamp(end)-pd.Timestamp('2000-01-01')).days
   fi=original.npy(ROOT/f'outputs/round48/{stage}/fit_indices.npy');qi=original.npy(ROOT/f'outputs/round48/{stage}/query_indices.npy')
   need(fi.dtype==qi.dtype==np.dtype('int64') and np.array_equal(fi,np.flatnonzero(np.isfinite(days)&(days<lo))) and np.array_equal(qi,np.flatnonzero(np.isfinite(days)&(days>=lo)&(days<=hi))),'Ordered window indices differ')
   for label,idx in [('fit',fi),('query',qi)]:
    need(original.array_sha(idx)==old['split_contract'][stage][label+'_indices_content_sha256'],'Original index content changed')
   pred=np.column_stack([original.npy(ROOT/f'outputs/round46/{stage}/{CONTROL}.npy'),original.npy(ROOT/f'outputs/round49/{stage}/predictions.npy')])
   need(pred.dtype==np.dtype('float64') and pred.shape==(len(qi),2) and np.isfinite(pred).all() and (pred>=0).all(),'Saved predictions invalid')
   need(np.isfinite(y[qi]).all() and (y[qi]>0).all(),'Fixed query labels invalid')
   xf=X.iloc[fi].reset_index(drop=True);xq=X.iloc[qi].reset_index(drop=True);labels=original.group_labels(xf,xq)
   errors=np.abs(y[qi,None]-pred);point=original.point_values(errors,labels,weights)
   for j,(record,name) in enumerate([(a,CONTROL),(b,CANDIDATE)]):
    need(abs(point[j]-record['metrics'][name]['overall_mae'])<=1e-6 and abs(point[j+2]-record['metrics'][name]['weighted_mae'])<=1e-6,'Saved completed MAE differs')
   keys,fallback=original.cluster_keys(xq,qi);months=original.relative_months(xq.trade_day_index,start)
   value={'fit_rows':len(fi),'rows':len(qi),'dates':[start,end],'completed_overall_weighted_replayed':True,'missing_project_fallback_rows':fallback}
   for family,functions,group,fj in [('project',original.bootstrap_projects,keys,0),('month',original.bootstrap_months,months,1)]:
    boot,support=functions(errors,labels,group,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],si,fj])),RULES['replicates'])
    value[family]={'summary':summarize(boot,point),'support':support}
    p=OUT/f'{stage}_{family}_replicates.npy'
    with p.open('xb') as f:np.save(f,boot,allow_pickle=False)
    result['artifact_sha256'][rel(p)]=sha(p);p=OUT/f'{stage}_{family}_deltas_and_valid.npz'
    d=boot[:,1]-boot[:,0];w=boot[:,3]-boot[:,2]
    with p.open('xb') as f:np.savez(f,overall_delta=d,weighted_delta=w,overall_valid=np.isfinite(d),weighted_valid=np.isfinite(w))
    result['artifact_sha256'][rel(p)]=sha(p)
   result['stages'][stage]=value
   print(json.dumps({'event':'stage_complete','stage':stage,'project':value['project']['summary'],'month':value['month']['summary']},ensure_ascii=False),flush=True)
  need(all(sha(ROOT/n)==h for n,h in sources.items()),'Source changed during diagnostic')
  result['protected_after']={n:sha(ROOT/n) for n in PROTECTED};need(result['protected_after']==before,'Protected artifacts changed')
  result.update(protected_unchanged=True,elapsed_seconds=time.monotonic()-started,completed_at_utc=now())
  with REPORT.open('x',encoding='utf-8',newline='\n') as f:f.write(render(result))
  result['artifact_sha256'][rel(REPORT)]=sha(REPORT);write(OUT/'results.json',result)
  print(json.dumps({'status':'completed','results_sha256':sha(OUT/'results.json'),'report_sha256':sha(REPORT),'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False),flush=True)
 except Exception:
  write(OUT/'failure.json',{'status':'technical_failure','elapsed_seconds':time.monotonic()-started,'traceback':traceback.format_exc(),'automatic_retry':False})
  raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args();run()
