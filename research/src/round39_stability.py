"""Fixed paired R39-versus-v6 diagnostics; reuse the unchanged R38 bootstrap helpers."""
from __future__ import annotations
import argparse, copy, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import round38_stability as prior

original=prior.original
ROOT=prior.ROOT
OUT=ROOT/'outputs/round39/stability'
DOC=ROOT/'reports/round39_stability_plan.md'
REPORT=ROOT/'reports/round39_stability_report.md'
BASE=ROOT/'outputs/round39/evaluation'
STAGES=original.STAGES
CONTROL='v6_control'
CANDIDATE='city_road_spatial_half_raw_half_log'
R39_PLAN_SHA='110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13'
R39_RESULTS_SHA='1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27'
PRIOR_PLAN='outputs/round38_stability/plan.json'
PRIOR_PLAN_SHA='23e0ae024a9c19a657fcdab3d36e4ece7c4439f86c187517ea45112988a67ab3'
PRIOR_SOURCE_SHA='d9ec7acac4f0e7cc6b507fca283c019ad4f00c21516917e6916e66c2e93f4fb5'
RULES=copy.deepcopy(original.RULES)
RULES['delta']='MAE_R39_candidate - MAE_v6_control; negative means improvement'
CAUTION=original.CAUTION
need,sha,rel,read,locked,write,lock_write=original.need,original.sha,original.rel,original.read,original.locked,original.write,original.lock_write
PROTECTED=[f'outputs/submission_v{v}.csv' for v in (2,3)]+[f'outputs/submission_v{v}_trial.csv' for v in (4,5,6)]+['runs/versions.json']

def runtime():return prior.runtime()
def protected():return {n:{'sha256':sha(ROOT/n),'bytes':(ROOT/n).stat().st_size} for n in PROTECTED}
def paths(stage):return {role:rel(BASE/stage/(name+'.npy')) for role,name in [('control',CONTROL),('candidate',CANDIDATE),('fit_indices','fit_indices'),('query_indices','query_indices')]}
def checked_path(name):
 p=(ROOT/name).resolve();need(p.is_relative_to(ROOT) and p.is_file(),'Missing/unsafe input: '+name);return p

def freeze():
 need(not OUT.exists() and DOC.is_file() and not REPORT.exists(),'Existing diagnostic or missing plan')
 need(sha(ROOT/PRIOR_PLAN)==PRIOR_PLAN_SHA and sha(Path(prior.__file__))==PRIOR_SOURCE_SHA,'Prior diagnostic source/plan changed')
 old=locked(ROOT/PRIOR_PLAN)
 need(runtime()==old['runtime'] and RULES['seed']==20260925 and RULES['replicates']==2000,'Fixed runtime/rules differ')
 expected=old['dependency_sha256'];deps={}
 def add(name,digest):
  actual=sha(checked_path(name));need(actual==digest,'Historical input identity differs: '+name)
  need(name not in deps or deps[name]==actual,'Conflicting input identity');deps[name]=actual
 def receipt(name,digest):
  add(name,digest);value=locked(ROOT/name);r=Path(name).with_suffix('.sha256').as_posix();add(r,sha(ROOT/r));return value
 receipt(PRIOR_PLAN,PRIOR_PLAN_SHA)
 method=receipt(prior.OLD_PLAN,prior.OLD_PLAN_SHA)
 for name,digest in method['source_closure_sha256'].items():add(name,digest)
 add('src/round38_stability.py',PRIOR_SOURCE_SHA)
 plan=receipt(rel(BASE/'plan.json'),R39_PLAN_SHA);total=receipt(rel(BASE/'results.json'),R39_RESULTS_SHA)
 need(plan['round']==total['round']==39 and total['plan_sha256']==R39_PLAN_SHA and total['status']=='three_periods_completed_no_adoption_decision','R39 authority differs')
 need(plan['stage_order']==total['checked_stages']==list(STAGES) and plan['control']==CONTROL and plan['candidate']==CANDIDATE,'Saved comparison differs')
 need(plan['split_contract']==old['split_contract'] and plan['splits']==old['splits'] and plan['target_group_weights']==old['fixed_target_group_weights'],'Original split/weights differ')
 for name,digest in plan['source_sha256'].items():add(name,digest)
 for name in ['data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json']:
  need(expected[name]==plan['input_sha256'][name],'Canonical input differs');add(name,expected[name])
 for stage in STAGES:
  saved=receipt(rel(BASE/stage/'results.json'),total['stage_result_sha256'][stage])
  need(saved['round']==39 and saved['stage']==stage and saved['status']=='completed' and saved['plan_sha256']==R39_PLAN_SHA and saved['dates']==old['splits'][stage],'Stage contract differs')
  need(saved['metrics']==total['stages'][stage]['metrics'] and saved['original_full_metrics_reproduced'] and saved['all_prior_controls_bitwise_equal'],'Saved full metric/control evidence differs')
  for label in ('fit','query'):need(saved[label+'_rows']==old['split_contract'][stage][label+'_rows'],'Stage row count differs')
  for name in paths(stage).values():add(name,saved['artifact_sha256'][name])
  name=f'outputs/round6/geography/{stage}/train_source_cache_indices.npy';add(name,expected[name])
  base='outputs/round6/baseline/auxiliary' if stage=='auxiliary' else f'outputs/round3/baseline/{stage}'
  for name in [base+'/fit_indices.npy',base+'/check_indices.npy',f'outputs/round6/models/{stage}/geography_catboost/fit_indices.npy',f'outputs/round6/models/{stage}/geography_catboost/query_indices.npy']:add(name,expected[name])
 for name in PROTECTED:add(name,sha(ROOT/name))
 for path in (DOC,Path(__file__)):add(rel(path),sha(path))
 spec={'round':39,'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'paired conditional stability of saved R39 candidate versus saved v6 control',
  'rules':RULES,'reused_functions':prior.REUSED_FUNCTIONS,'R38_stability_plan_sha256':PRIOR_PLAN_SHA,'R38_stability_source_sha256':PRIOR_SOURCE_SHA,
  'original_method_sha256':prior.ORIGINAL_SOURCE_SHA,'R39_plan_sha256':R39_PLAN_SHA,'R39_results_sha256':R39_RESULTS_SHA,
  'fixed_target_group_weights':old['fixed_target_group_weights'],'splits':old['splits'],'split_contract':old['split_contract'],
  'dependency_sha256':deps,'runtime':runtime(),'protected_before':protected(),'caution':CAUTION,
  'plan_creation_read_prediction_or_target_values':False,'model_or_map_operations':False,'no_new_predictions_clipping_or_rounding':True}
 for name,digest in deps.items():need(sha(checked_path(name))==digest,'Input/source changed before freeze: '+name)
 lock_write(OUT/'plan.json',spec)
 print(json.dumps({'status':'frozen','plan_sha256':sha(OUT/'plan.json'),'dependencies':len(deps)},ensure_ascii=False),flush=True)

def checked_plan():
 spec=locked(OUT/'plan.json');need(spec['rules']==RULES and spec['runtime']==runtime(),'Rules/runtime changed')
 for name,digest in spec['dependency_sha256'].items():need(sha(checked_path(name))==digest,'Input/source changed: '+name)
 return spec

def summary(boot,point):
 value=original.summarize(boot,point)
 for v in value.values():
  v['v6_control_mae']=v.pop('v2_mae');v['R39_candidate_mae']=v.pop('v3_mae')
  v['point_delta_candidate_minus_v6']=v.pop('point_delta_v3_minus_v2')
 return value

def render(results):
 lines=['# R39 相對 v6 的固定成對穩定度診斷','',CAUTION,'',
 '比較的是 R39 缺區道路定位方法與 v6 固定平均的完整保存預測，沒有重新產生預測、重播 reference／spatial、訓練或載入樹模型。規約在已知 R39 三期點估計後固定；不是盲測，亦不更改 R39 試交必要條件、CSV 或推薦。','',
 '差值為 **R39 MAE − v6 MAE**，負值較好。直接重用 R38 的原建案及月份 helper，seed 20260925，每年度／抽法 2,000 次，兩方法配對使用相同樣本。百分位區間為 2.5%～97.5%；抽樣改善比例不是 p 值或 Private 改善機率。','',
 '| 窗口 | 抽法 | 指標 | v6 MAE | R39 MAE | 差值 | 2.5% | 97.5% | 抽樣改善比例 | 有效／無效 |',
 '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 def num(v):return '無法估計' if v is None else f'{v:,.2f}'
 for stage in STAGES:
  for family in ('project','month'):
   for metric,v in results['stages'][stage][family]['summary'].items():
    frac='無法估計' if v['negative_delta_fraction'] is None else f"{v['negative_delta_fraction']:.2%}"
    lines.append(f"| {stage} | {family} | {metric} | {num(v['v6_control_mae'])} | {num(v['R39_candidate_mae'])} | {num(v['point_delta_candidate_minus_v6'])} | {num(v['percentile_2_5'])} | {num(v['percentile_97_5'])} | {frac} | {v['valid_replicates']}/{v['invalid_replicates']} |")
 lines+=['','## 預測相同與區間的解讀','', '| 窗口 | 完整query | 預測變動列 | 兩向量逐位相同 |','| --- | ---: | ---: | --- |']
 for stage in STAGES:
  s=results['stages'][stage];lines.append(f"| {stage} | {s['rows']:,} | {s['prediction_changed_rows']:,} | {s['predictions_bitwise_identical']} |")
 lines+=['','若 auxiliary 兩向量逐位相同，所有有效配對差值及區間均為零是相同輸出的必然結果；抽樣嚴格改善比例為0也不代表候選更差，不能將其當作額外泛化證據。',
 '區間跨零表示增量對該種群組／月份構成敏感；未跨零也僅限固定預測及既有窗口，不保證 Private。兩種抽法完整報告，沒有依結果調參、選區間或增加門檻。','',
 '## 原方法與群組支持','',
 '| 窗口 | 分層 | 列數 | 群組數 | 最大群組列占比 | 有效群組數 |','| --- | --- | ---: | ---: | ---: | ---: |']
 for stage in STAGES:
  for group in original.GROUPS:
   v=results['stages'][stage]['project']['support'][group];share='無資料' if v['largest_cluster_row_share'] is None else f"{v['largest_cluster_row_share']:.2%}"
   lines.append(f"| {stage} | {group} | {v['rows']:,} | {v['clusters']:,} | {share} | {num(v['row_weight_effective_cluster_count'])} |")
 lines+=['','建案抽法在原seen／new／missing分層內整群有放回抽樣；缺名沿原地址、道路、canonical列號備援。overall仍按交易列計算，weighted仍沿原test-X權重，沒有改為等權建案MAE。月份沿預測起日11日起算12格，抽四段連續三月並環繞；正權重分層缺席則該次weighted記無效、不重抽。',
 '沿保存canonical K及原fi/qi，未重新含價格去重。載入原train快取會反序列化所有train標籤；只有三期固定query標籤用於完整保存metrics核對及抽樣，未使用fit標籤訓練。未載test標籤、模型、地圖或建立新預測。完整原scorer字典須精確一致；bootstrap聚合點估計沿原1e-6元容差。','',
 f"本次內部耗時 {results['elapsed_seconds']:.3f} 秒；六個既有CSV／registry保護記錄前後相同。外部命令開始／結束／exit／耗時及stdout見execution。來源與實讀输入SHA固定於機器規約，重抽樣與有效mask輸出SHA列於結果。",'',
 '[事前診斷規約](round39_stability_plan.md) · [機器規約](../outputs/round39/stability/plan.json) · [完整結果](../outputs/round39/stability/results.json) · [執行紀錄](../outputs/round39/stability/execution/run.json) · [R38方法](round38_stability_plan.md)','']
 return '\n'.join(lines)

def run():
 started=time.monotonic();need(not (OUT/'run_claim.json').exists() and not REPORT.exists() and not (OUT/'results.json').exists(),'No repeat diagnostic')
 write(OUT/'run_claim.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'one_fixed_run':True})
 spec=checked_plan();raw=pd.read_pickle(ROOT/'data_cache/train.pkl');features=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
 schema=read(ROOT/'reports/feature_schema.json');need(len(raw)==len(features)==455784 and raw.index.equals(pd.RangeIndex(455784)) and features.index.equals(raw.index) and list(features.columns)==schema['columns'],'Raw/cache alignment differs')
 idx=original.npy(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy')
 need(idx.dtype==np.dtype('int64') and idx.shape==(455255,) and np.array_equal(idx,np.unique(idx)) and ((idx>=0)&(idx<len(raw))).all(),'Canonical indices differ')
 X=features.iloc[idx].reset_index(drop=True);y=raw['總價元'].to_numpy(dtype='float64')[idx];del raw,features
 results={'round':39,'status':'completed','purpose':spec['purpose'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),
  'rules':RULES,'caution':CAUTION,'retraining_performed':False,'models_loaded':False,'maps_loaded':False,'selection_changed':False,'registry_changed':False,
  'full_train_labels_deserialized':True,'only_fixed_query_labels_used_for_errors':True,'new_predictions_generated':False,'new_prediction_clipping_or_rounding':False,
  'protected_before':spec['protected_before'],'stages':{},'artifact_sha256':{}}
 days=X.trade_day_index.to_numpy();weights=spec['fixed_target_group_weights']
 for stage_index,stage in enumerate(STAGES):
  start,end=spec['splits'][stage];start_day=(pd.Timestamp(start)-pd.Timestamp('2000-01-01')).days;end_day=(pd.Timestamp(end)-pd.Timestamp('2000-01-01')).days
  fi=np.flatnonzero(np.isfinite(days)&(days<start_day));qi=np.flatnonzero(np.isfinite(days)&(days>=start_day)&(days<=end_day));contract=spec['split_contract'][stage]
  need(len(fi)==contract['fit_rows'] and len(qi)==contract['query_rows'] and not np.intersect1d(fi,qi).size,'Split counts/overlap differs')
  need(original.array_sha(fi)==contract['fit_indices_content_sha256'] and original.array_sha(qi)==contract['query_indices_content_sha256'],'Split ordered content differs')
  need(np.array_equal(idx,original.npy(ROOT/f'outputs/round6/geography/{stage}/train_source_cache_indices.npy')),'Canonical mapping differs')
  p=paths(stage);base='outputs/round6/baseline/auxiliary' if stage=='auxiliary' else f'outputs/round3/baseline/{stage}'
  for name,expected in [(p['fit_indices'],fi),(p['query_indices'],qi),(base+'/fit_indices.npy',fi),(base+'/check_indices.npy',qi),(f'outputs/round6/models/{stage}/geography_catboost/fit_indices.npy',fi),(f'outputs/round6/models/{stage}/geography_catboost/query_indices.npy',qi)]:
   need(np.array_equal(original.npy(ROOT/name),expected),'Saved index alignment differs: '+name)
  saved=locked(BASE/stage/'results.json');predictions=np.column_stack([original.npy(ROOT/p['control']),original.npy(ROOT/p['candidate'])])
  need(predictions.dtype==np.dtype('float64') and predictions.shape==(len(qi),2) and np.isfinite(predictions).all() and (predictions>=0).all(),'Saved prediction validity')
  need(np.isfinite(y[qi]).all() and (y[qi]>0).all(),'Invalid fixed query labels')
  Xfit=X.iloc[fi].reset_index(drop=True);Xq=X.iloc[qi].reset_index(drop=True);labels=original.group_labels(Xfit,Xq)
  errors=np.abs(y[qi,None]-predictions);point=original.point_values(errors,labels,weights)
  for j,name in enumerate((CONTROL,CANDIDATE)):
   scored=original.original_score(y[qi],predictions[:,j],Xfit,Xq,start,weights)
   need(scored==saved['metrics'][name],'Full saved metrics not exactly reproduced: '+name)
   prior_values=saved['metrics'][name];need(abs(point[j]-prior_values['overall_mae'])<=1e-6 and abs(point[j+2]-prior_values['weighted_mae'])<=1e-6,'Point MAE differs')
  keys,fallback=original.cluster_keys(Xq,qi);months=original.relative_months(Xq.trade_day_index,start)
  bp,support=original.bootstrap_projects(errors,labels,keys,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],stage_index,0])),RULES['replicates'])
  bm,month_support=original.bootstrap_months(errors,labels,months,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],stage_index,1])),RULES['replicates'])
  identical=predictions[:,0].tobytes()==predictions[:,1].tobytes()
  if identical:
   for array in (bp,bm):
    for left,right in ((0,1),(2,3)):
     valid=np.isfinite(array[:,left])&np.isfinite(array[:,right]);need(np.array_equal(array[valid,left],array[valid,right]),'Identical prediction bootstrap invariant failed')
  value={'rows':len(qi),'fit_rows':len(fi),'dates':[start,end],'fit_indices_content_sha256':original.array_sha(fi),'query_indices_content_sha256':original.array_sha(qi),
   'saved_result_sha256':sha(BASE/stage/'results.json'),'point_mae_replay_verified':True,'all_original_metrics_exactly_reproduced':True,
   'prediction_changed_rows':int(np.count_nonzero(predictions[:,0]!=predictions[:,1])),'predictions_bitwise_identical':identical,
   'identical_outputs_are_not_extra_generalization_evidence':identical,'missing_project_fallback_rows':fallback,
   'project':{'summary':summary(bp,point),'support':support},'month':{'summary':summary(bm,point),'support':month_support}}
  for family,array in [('project',bp),('month',bm)]:
   path=OUT/f'{stage}_{family}_replicates.npy'
   with path.open('xb') as f:np.save(f,array,allow_pickle=False)
   results['artifact_sha256'][rel(path)]=sha(path);delta=OUT/f'{stage}_{family}_deltas_and_valid.npz'
   with delta.open('xb') as f:np.savez(f,overall_delta=array[:,1]-array[:,0],weighted_delta=array[:,3]-array[:,2],overall_valid=np.isfinite(array[:,1]-array[:,0]),weighted_valid=np.isfinite(array[:,3]-array[:,2]))
   results['artifact_sha256'][rel(delta)]=sha(delta)
  results['stages'][stage]=value
  print(json.dumps({'event':'stage_complete','stage':stage,'identical':identical,'project':value['project']['summary'],'month':value['month']['summary']},ensure_ascii=False),flush=True)
 checked_plan();results['protected_after']=protected();need(results['protected_before']==results['protected_after'],'Protected CSV/registry changed')
 results['protected_unchanged']=True;results['elapsed_seconds']=time.monotonic()-started
 with REPORT.open('x',encoding='utf-8') as f:f.write(render(results))
 results['artifact_sha256'][rel(REPORT)]=sha(REPORT);lock_write(OUT/'results.json',results)
 print(json.dumps({'status':'completed','results_sha256':sha(OUT/'results.json'),'report_sha256':sha(REPORT),'elapsed_seconds':results['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
 {'freeze':freeze,'run':run}[args.command]()
