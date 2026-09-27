"""Fixed paired diagnostics of saved R38 half-combination predictions; no model operations."""
from __future__ import annotations
import argparse, copy, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import v3_cluster_stability as original

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round38_stability'
DOC=ROOT/'reports/round38_stability_plan.md'
REPORT=ROOT/'reports/round38_stability_report.md'
OLD_PLAN='outputs/diagnostics/v3_cluster_stability/plan.json'
OLD_PLAN_SHA='a7464f93713e0de2c2c52a475882e3884b2026dda7d6b12d9dab5533cf535edf'
ORIGINAL_SOURCE_SHA='46622f859660e40bd78fbbd612db53b5a9bd5069f55eb9e7b169e19cc134d1c9'
CANDIDATE='fixed_half_spatial_half_log_spatial'
STAGES=original.STAGES
RULES=copy.deepcopy(original.RULES)
RULES['delta']='MAE_R38_candidate - MAE_v3; negative means improvement'
CAUTION=original.CAUTION
need,sha,rel,read,locked,write,lock_write=original.need,original.sha,original.rel,original.read,original.locked,original.write,original.lock_write
R38_PLAN='outputs/round38/plan.json'
R38_PLAN_SHA='f885d8e2f37ee8655d1ca7e6af448572b847aaaee3b19cd96badaba297759f4b'
R38_RESULTS='outputs/round38/results.json'
R38_RESULTS_SHA='f14d84c7c9163cf894f5302fb6b6f8de0168ff56d2c80aa5bdd0c7f29ae5f947'
PROTECTED=['outputs/submission_v2.csv','outputs/submission_v3.csv','outputs/submission_v4_trial.csv','runs/versions.json']
REUSED_FUNCTIONS=['group_labels','cluster_keys','point_values','bootstrap_projects','relative_months','bootstrap_months','summarize','compare_metrics','original_score']

def paths(stage):
 return {'control':f'outputs/round36/{stage}/v3_control.npy',
  'candidate':f'outputs/round38/{stage}/predictions.npy',
  'fit_indices':f'outputs/round36/{stage}/fit_indices.npy',
  'query_indices':f'outputs/round36/{stage}/query_indices.npy'}

def runtime():return {'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
def protected():return {n:{'sha256':sha(ROOT/n),'bytes':(ROOT/n).stat().st_size} for n in PROTECTED}
def checked_path(name):
 p=(ROOT/name).resolve();need(p.is_relative_to(ROOT) and p.is_file(),'Unsafe/missing input: '+name);return p

def freeze():
 need(not OUT.exists(),'Output directory already exists; no duplicate diagnostic')
 need(DOC.is_file() and not REPORT.exists(),'Plan document missing or result already present')
 need(sha(ROOT/OLD_PLAN)==OLD_PLAN_SHA,'Original diagnostic plan changed')
 old=locked(ROOT/OLD_PLAN);need(sha(Path(original.__file__))==ORIGINAL_SOURCE_SHA,'Original method source changed')
 need(all(callable(getattr(original,n)) for n in REUSED_FUNCTIONS),'Original method API missing')
 need(runtime()==old['runtime'],'Original numpy/pandas/Python diagnostic runtime differs')
 deps={};expected={**old['dependency_sha256'],**old['source_closure_sha256']}
 def add(name,digest=None):
  actual=sha(checked_path(name));need(digest is None or actual==digest,'Expected input SHA differs: '+name)
  need(name not in deps or deps[name]==actual,'Conflicting input identity');deps[name]=actual
 def receipt(name):
  locked(ROOT/name);add(name);add(Path(name).with_suffix('.sha256').as_posix())
 add(OLD_PLAN,OLD_PLAN_SHA);receipt(OLD_PLAN)
 for n,h in old['source_closure_sha256'].items():add(n,h)
 for n in ['data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json']:
  add(n,expected[n])
 add(R38_PLAN,R38_PLAN_SHA);receipt(R38_PLAN)
 add(R38_RESULTS,R38_RESULTS_SHA);receipt(R38_RESULTS)
 candidate_plan=read(ROOT/R38_PLAN);total=read(ROOT/R38_RESULTS)
 need(candidate_plan['round']==38 and total['round']==38 and total['plan_sha256']==R38_PLAN_SHA,'R38 identity mismatch')
 need(total['checked_stages']==list(STAGES) and candidate_plan['stage_order']==list(STAGES),'Stage order differs')
 need(candidate_plan['formula']=='0.5*spatial_v3+0.5*log_spatial_combination','R38 fixed formula differs')
 need(candidate_plan['splits']==old['splits'] and candidate_plan['split_contract']==old['split_contract'],'Original split contract differs')
 need(candidate_plan['target_group_weights']==old['fixed_target_group_weights'],'Original target weights differ')
 for name,digest in candidate_plan['source_sha256'].items():add(name,digest)
 for name in ['data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json']:
  need(deps[name]==candidate_plan['input_sha256'][name],'R38 input differs from original diagnostic')
 for stage in STAGES:
  n=f'outputs/round38/{stage}/results.json';receipt(n);saved=read(ROOT/n)
  need(saved==total['stages'][stage],'Stage result differs from fixed R38 total')
  need(saved['stage']==stage,'Result stage mismatch')
  need(saved['fit_rows']==old['split_contract'][stage]['fit_rows'] and saved['query_rows']==old['split_contract'][stage]['query_rows'],'Result row contract differs')
  p=paths(stage);add(p['candidate'],saved['prediction_sha256'])
  prior_name=f'outputs/round36/{stage}/results.json';add(prior_name,candidate_plan['input_sha256'][prior_name]);receipt(prior_name)
  prior=read(ROOT/prior_name)
  need(prior['dates']==old['splits'][stage],'R36 dates differ')
  need(prior['metrics']['v3_control']==saved['metrics']['v3_control'],'Saved v3 metrics differ between R36 and R38')
  for role in ['control','fit_indices','query_indices']:
   name=p[role];add(name,candidate_plan['input_sha256'][name]);need(deps[name]==prior['artifact_sha256'][name],'R36 artifact identity differs')
  geo=f'outputs/round6/geography/{stage}/train_source_cache_indices.npy';add(geo,expected[geo])
  for item in ['fit_indices.npy','query_indices.npy']:
   name=f'outputs/round6/models/{stage}/geography_catboost/{item}';add(name,expected[name])
  baseline='outputs/round6/baseline/auxiliary' if stage=='auxiliary' else f'outputs/round3/baseline/{stage}'
  for item in ['fit_indices.npy','check_indices.npy']:
   name=baseline+'/'+item;add(name,expected[name])
 for name in PROTECTED:add(name)
 add(rel(DOC));add(rel(Path(__file__)))
 spec={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'paired conditional stability of the fixed R38 half-combination against v3',
  'rules':RULES,'reused_functions':REUSED_FUNCTIONS,'original_diagnostic_source_sha256':ORIGINAL_SOURCE_SHA,
  'original_diagnostic_plan_sha256':OLD_PLAN_SHA,'fixed_target_group_weights':old['fixed_target_group_weights'],
  'splits':old['splits'],'split_contract':old['split_contract'],'R38_plan_sha256':R38_PLAN_SHA,'R38_results_sha256':R38_RESULTS_SHA,
  'dependency_sha256':deps,'runtime':runtime(),'protected_before':protected(),
  'plan_creation_read_prediction_values':False,'plan_creation_read_target_values':False,
  'no_retraining_or_model_reload':True,'no_new_clipping_or_rounding':True,'caution':CAUTION}
 lock_write(OUT/'plan.json',spec)
 print(json.dumps({'status':'fixed_diagnostic_plan_saved','sha256':sha(OUT/'plan.json'),'dependencies':len(deps)},ensure_ascii=False),flush=True)

def checked_plan():
 spec=locked(OUT/'plan.json');need(spec['rules']==RULES and spec['runtime']==runtime(),'Rules/runtime changed')
 for name,digest in spec['dependency_sha256'].items():need(sha(checked_path(name))==digest,'Input/source changed: '+name)
 return spec

def summary(boot,point):
 value=original.summarize(boot,point)
 for v in value.values():
  v['v3_control_mae']=v.pop('v2_mae');v['R38_candidate_mae']=v.pop('v3_mae')
  v['point_delta_candidate_minus_v3']=v.pop('point_delta_v3_minus_v2')
 return value

def render(results):
 lines=['# R38 固定 half 方案相對 v3 的成對穩定度診斷','',CAUTION,'',
 '本次直接讀取 R38 三期已保存的 fixed_half_spatial_half_log_spatial 向量，對照同窗 R36 保存的 v3_control。固定候選為 spatial_v3 與 log_spatial_combination 最終預測各半；本診斷不重新融合、不重新套 reference 或 spatial，也不訓練或載入模型、不新裁切或四捨五入。這是在已知多輪結果後對既定候選的條件診斷，不更改歷史決定、CSV 或 registry。','',
 '差值為 **候選 MAE − v3 MAE**，負值較好；單位為新臺幣元。兩種抽法、三個年度、overall／weighted 全部呈現。每種抽法各 2,000 次，使用原 seed 20260925 及相同的 stage／family SeedSequence；下列區間是第 2.5／97.5 百分位，改善比例不是 p 值。','',
 '| 窗口 | 抽樣方式 | 指標 | v3 MAE | 候選 MAE | 差值 | 2.5% | 97.5% | 抽樣改善比例 | 有效／無效 |',
 '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 def num(v):return '無法估計' if v is None else f'{v:,.2f}'
 for stage in STAGES:
  for family in ['project','month']:
   for metric,v in results['stages'][stage][family]['summary'].items():
    frac='無法估計' if v['negative_delta_fraction'] is None else f"{v['negative_delta_fraction']:.2%}"
    lines.append(f"| {stage} | {family} | {metric} | {num(v['v3_control_mae'])} | {num(v['R38_candidate_mae'])} | {num(v['point_delta_candidate_minus_v3'])} | {num(v['percentile_2_5'])} | {num(v['percentile_97_5'])} | {frac} | {v['valid_replicates']}/{v['invalid_replicates']} |")
 lines += ['', '## 群組支持與限制','',
 '| 窗口 | 分層 | 列數 | 群組數 | 最大群組列占比 | 依列權重的有效群組數 |',
 '| --- | --- | ---: | ---: | ---: | ---: |']
 for stage in STAGES:
  for group in original.GROUPS:
   v=results['stages'][stage]['project']['support'][group]
   share='無資料' if v['largest_cluster_row_share'] is None else f"{v['largest_cluster_row_share']:.2%}"
   lines.append(f"| {stage} | {group} | {v['rows']:,} | {v['clusters']:,} | {share} | {num(v['row_weight_effective_cluster_count'])} |")
 lines += ['', '建案抽法在原 seen／new_nonmissing／missing 分層內，抽回原群組數並整群保留；缺建案名依原地址、道路、canonical query 列號備援。overall 按抽中交易列計算；weighted 使用原 test 分層比例，沒有改成等權建案 MAE。',
 '月份抽法沿各預測起日 11 日起算 12 格，每次抽四段連續三月並環繞。正權重分層缺席時 weighted 記無效，不重抽；各月份列數及缺名備援數保存在結果 JSON。年度分開呈現，沒有把重疊資料當獨立樣本合併。','',
 '若區間跨越零，表示小幅點改善對此種群組／月份構成有敏感性；即使整段低於零，也只是這些固定預測與窗口下的條件結果。本設計不辨認「是哪幾案造成」或證明因果，也不涵蓋多輪選型、重新訓練或未來市場移轉。不能據此估計 Private 改善機率或宣稱正式接受。','',
 '## 實際執行與來源','',
 '共享原載入方式會反序列化整份 train.pkl（含所有 train 標籤）；只有三個固定 query 索引的標籤用於既有預測的誤差核對與抽樣，fit 標籤沒有用於新訓練。原含價格、排除 ID 的 raw 全列去重與已保存 canonical 索引不變；87 欄原特徵只用於原分層／月份／計分。沒有載入 CBM、沒有讀取 test 目標。',
 f"三期完整原計分欄位與點估計均核對至原 0.000001 元容差；來源與四個保護檔前後 hash 相同。診斷程式內計時 {results['elapsed_seconds']:.3f} 秒；命令／stdout／exit／外部耗時另存 execution.json／execution.log。",'',
 '[固定診斷規約](round38_stability_plan.md) · [機器規約](../outputs/round38_stability/plan.json) · [完整結果](../outputs/round38_stability/results.json) · [執行紀錄](../outputs/round38_stability/execution.json) · [原方法](v3_cluster_stability_plan.md)','']
 return '\n'.join(lines)

def run():
 started=time.monotonic();need(not (OUT/'run_claim.json').exists() and not REPORT.exists() and not (OUT/'results.json').exists(),'No repeat diagnostic')
 write(OUT/'run_claim.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'one_fixed_run':True})
 spec=checked_plan();raw=pd.read_pickle(ROOT/'data_cache/train.pkl');features=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
 schema=read(ROOT/'reports/feature_schema.json');need(len(raw)==len(features)==455784 and list(features.columns)==schema['columns'],'Raw/cache alignment or schema differs')
 idx=original.npy(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy')
 need(idx.dtype==np.dtype('int64') and idx.shape==(455255,) and np.array_equal(idx,np.unique(idx)) and ((idx>=0)&(idx<len(raw))).all(),'Invalid canonical indices')
 X=features.iloc[idx].reset_index(drop=True);y=raw['總價元'].to_numpy(dtype='float64')[idx];del raw,features
 need(np.isfinite(y).all() and (y>0).all(),'Invalid original target values')
 results={'status':'completed','purpose':spec['purpose'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),
  'rules':RULES,'caution':CAUTION,'retraining_performed':False,'models_loaded':False,'selection_changed':False,'registry_changed':False,
  'full_train_labels_deserialized':True,'only_fixed_query_labels_used_for_errors':True,'new_prediction_clipping_or_rounding':False,
  'protected_before':spec['protected_before'],'stages':{},'artifact_sha256':{}}
 days=X.trade_day_index.to_numpy();weights=spec['fixed_target_group_weights']
 for stage_index,stage in enumerate(STAGES):
  start,end=spec['splits'][stage];start_day=(pd.Timestamp(start)-pd.Timestamp('2000-01-01')).days;end_day=(pd.Timestamp(end)-pd.Timestamp('2000-01-01')).days
  fi=np.flatnonzero(np.isfinite(days)&(days<start_day));qi=np.flatnonzero(np.isfinite(days)&(days>=start_day)&(days<=end_day));contract=spec['split_contract'][stage]
  need(len(fi)==contract['fit_rows'] and len(qi)==contract['query_rows'] and not np.intersect1d(fi,qi).size,'Split counts/overlap differs')
  need(original.array_sha(fi)==contract['fit_indices_content_sha256'] and original.array_sha(qi)==contract['query_indices_content_sha256'],'Split ordered content differs')
  need(np.array_equal(idx,original.npy(ROOT/f'outputs/round6/geography/{stage}/train_source_cache_indices.npy')),'Canonical stage mapping differs')
  p=paths(stage);base='outputs/round6/baseline/auxiliary' if stage=='auxiliary' else f'outputs/round3/baseline/{stage}'
  for name,expected in [(p['fit_indices'],fi),(p['query_indices'],qi),(base+'/fit_indices.npy',fi),(base+'/check_indices.npy',qi),
   (f'outputs/round6/models/{stage}/geography_catboost/fit_indices.npy',fi),(f'outputs/round6/models/{stage}/geography_catboost/query_indices.npy',qi)]:
   need(np.array_equal(original.npy(ROOT/name),expected),'Saved prediction index alignment differs: '+name)
  saved=locked(ROOT/f'outputs/round38/{stage}/results.json');predictions=np.column_stack([original.npy(ROOT/p['control']),original.npy(ROOT/p['candidate'])])
  need(predictions.dtype==np.dtype('float64') and predictions.shape==(len(qi),2) and np.isfinite(predictions).all() and (predictions>=0).all(),'Saved prediction validity')
  Xfit=X.iloc[fi].reset_index(drop=True);Xq=X.iloc[qi].reset_index(drop=True);labels=original.group_labels(Xfit,Xq)
  errors=np.abs(y[qi,None]-predictions);point=original.point_values(errors,labels,weights)
  for j,name in enumerate(['v3_control',CANDIDATE]):
   original.compare_metrics(original.original_score(y[qi],predictions[:,j],Xfit,Xq,start,weights),saved['metrics'][name])
   prior=saved['metrics'][name];need(abs(point[j]-prior['overall_mae'])<=1e-6 and abs(point[j+2]-prior['weighted_mae'])<=1e-6,'Point MAE differs')
  keys,fallback=original.cluster_keys(Xq,qi);months=original.relative_months(Xq.trade_day_index,start)
  bp,support=original.bootstrap_projects(errors,labels,keys,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],stage_index,0])),RULES['replicates'])
  bm,month_support=original.bootstrap_months(errors,labels,months,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],stage_index,1])),RULES['replicates'])
  value={'rows':len(qi),'fit_rows':len(fi),'dates':[start,end],'fit_indices_content_sha256':original.array_sha(fi),'query_indices_content_sha256':original.array_sha(qi),
   'saved_result_sha256':sha(ROOT/f'outputs/round38/{stage}/results.json'),'point_mae_replay_verified':True,'all_original_metrics_replay_verified':True,'missing_project_fallback_rows':fallback,
   'project':{'summary':summary(bp,point),'support':support},'month':{'summary':summary(bm,point),'support':month_support}}
  for family,array in [('project',bp),('month',bm)]:
   path=OUT/f'{stage}_{family}_replicates.npy'
   with path.open('xb') as f:np.save(f,array,allow_pickle=False)
   results['artifact_sha256'][rel(path)]=sha(path)
   delta=OUT/f'{stage}_{family}_deltas_and_valid.npz'
   with delta.open('xb') as f:np.savez(f,overall_delta=array[:,1]-array[:,0],weighted_delta=array[:,3]-array[:,2],overall_valid=np.isfinite(array[:,1]-array[:,0]),weighted_valid=np.isfinite(array[:,3]-array[:,2]))
   results['artifact_sha256'][rel(delta)]=sha(delta)
  results['stages'][stage]=value
  print(json.dumps({'event':'fixed_stage_complete','stage':stage,'point_metrics_match':True,'project':value['project']['summary'],'month':value['month']['summary']},ensure_ascii=False),flush=True)
 checked_plan();results['protected_after']=protected();need(results['protected_before']==results['protected_after'],'Protected CSV/registry changed')
 results['protected_unchanged']=True;results['elapsed_seconds']=time.monotonic()-started
 with REPORT.open('x',encoding='utf-8') as f:f.write(render(results))
 results['artifact_sha256'][rel(REPORT)]=sha(REPORT);lock_write(OUT/'results.json',results)
 print(json.dumps({'status':'completed','results_sha256':sha(OUT/'results.json'),'report_sha256':sha(REPORT),'elapsed_seconds':results['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
 {'freeze':freeze,'run':run}[args.command]()
