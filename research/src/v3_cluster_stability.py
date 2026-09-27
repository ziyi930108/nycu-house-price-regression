"""Conditional paired-cluster diagnostics for frozen v3; never train or select models."""
from __future__ import annotations
import argparse, ast, hashlib, json, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from round3_evaluate import score as original_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/diagnostics/v3_cluster_stability'
DOC=ROOT/'reports/v3_cluster_stability_plan.md'
REPORT=ROOT/'reports/v3_cluster_stability_report.md'
SNAPSHOT_SHA='77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
STAGES=('development','confirmation','auxiliary')
GROUPS=('seen','new_nonmissing','missing')
RULES={'seed':20260925,'replicates':2000,'stage_order':list(STAGES),'family_order':['project','month'],
 'delta':'MAE_v3 - MAE_v2; negative means improvement','percentiles':[.025,.975],
 'rng':'default_rng(SeedSequence([seed,stage_index,family_index]))',
 'project':'within each stratum sample the original number of clusters with replacement; pair both models',
 'month':'12 origin-relative months; four uniformly drawn circular blocks of three consecutive months',
 'weighted_invalid':'positive-weight stratum absent: weighted replicate NaN, no redraw; report valid count',
 'selection_changed':False,'retraining_performed':False,'confidence_corrected_for_selection':False}
CAUTION=('這是固定預測與既有窗口的條件性診斷。區間未修正多輪選型偏差，不涵蓋未來市場與資料組成移轉、'
 '模型重訓不確定性或所有跨建案／跨時間相關性；重抽樣改善比例不是 p 值，也不是 Private 改善機率。'
 '建案鍵是資料中的地理名稱組合，未認證為實體建案。兩種抽樣結果均完整呈現，不擇優選區間。')

def need(ok,msg):
 if not ok: raise RuntimeError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def locked(p):
 p=Path(p);need(p.with_suffix('.sha256').read_text(encoding='ascii').strip()==sha(p),'Receipt differs: '+str(p));return read(p)
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def lock_write(p,v):
 write(p,v)
 with Path(p).with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
def norm(s):
 v=s.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
 return v.mask(v.isin(['','nan','None','<NA>','NaT']),'__MISSING__')
def project_keys(X):
 p=norm(X.project_name);return norm(X.address_city)+'|'+norm(X.address_district)+'|'+p,p.eq('__MISSING__')
def group_labels(fit,query):
 fk,fm=project_keys(fit);qk,qm=project_keys(query)
 return np.where(qm,'missing',np.where(qk.isin(set(fk[~fm])),'seen','new_nonmissing'))
def cluster_keys(X,source_indices):
 need(len(X)==len(source_indices),'Cluster row alignment')
 fields={k:norm(X[k]).to_numpy() for k in ['project_name','address_city','address_district','address_raw','address_road','address_section']}
 result=[];fallback={'address':0,'road':0,'singleton':0}
 for i in range(len(X)):
  city,district,project=(fields[k][i] for k in ['address_city','address_district','project_name'])
  if project!='__MISSING__':key=['project',city,district,project]
  elif fields['address_raw'][i]!='__MISSING__':key=['missing_address',city,district,fields['address_raw'][i]];fallback['address']+=1
  elif fields['address_road'][i]!='__MISSING__':key=['missing_road',city,district,fields['address_road'][i],fields['address_section'][i]];fallback['road']+=1
  else:key=['missing_row',int(source_indices[i])];fallback['singleton']+=1
  result.append(json.dumps(key,ensure_ascii=False,separators=(',',':')))
 return np.asarray(result,dtype=object),fallback

def aggregate(errors,codes,n):
 return np.bincount(codes,minlength=n).astype('int64'),np.column_stack([np.bincount(codes,weights=errors[:,j],minlength=n) for j in range(2)])
def measures(counts,sums,weights):
 need(np.asarray(counts).shape==(3,) and np.asarray(sums).shape==(3,2),'Aggregate schema')
 need(counts.sum()>0,'Empty resample')
 overall=sums.sum(axis=0)/counts.sum();weighted=np.zeros(2)
 for j,g in enumerate(GROUPS):
  if weights[g]>0:
   if counts[j]==0:return np.r_[overall,[np.nan,np.nan]]
   weighted+=weights[g]*sums[j]/counts[j]
 return np.r_[overall,weighted]
def point_values(errors,labels,weights):
 counts=np.array([(labels==g).sum() for g in GROUPS]);sums=np.array([errors[labels==g].sum(axis=0) for g in GROUPS])
 value=measures(counts,sums,weights);need(np.isfinite(value).all(),'Positive-weight point stratum has no rows')
 return value

def bootstrap_projects(errors,labels,keys,weights,rng,replicates):
 inventories=[];stats={}
 for g in GROUPS:
  mask=labels==g;codes,unique=pd.factorize(pd.Series(keys[mask],dtype=object),sort=True)
  counts,sums=aggregate(errors[mask],codes,len(unique));inventories.append((counts,sums))
  stats[g]={'rows':int(mask.sum()),'clusters':len(unique),'largest_cluster_rows':int(counts.max()) if len(counts) else 0,
   'largest_cluster_row_share':float(counts.max()/counts.sum()) if len(counts) else None,
   'row_weight_effective_cluster_count':float(counts.sum()**2/np.square(counts.astype('float64')).sum()) if len(counts) else None}
 # No cluster may cross scorer strata; sample only whole project groups.
 seen={}
 for k,g in zip(keys,labels):
  need(k not in seen or seen[k]==g,'Cluster crosses strata');seen[k]=g
 result=np.empty((replicates,4))
 for r in range(replicates):
  ns=np.zeros(3,dtype='int64');ss=np.zeros((3,2))
  for j,(counts,sums) in enumerate(inventories):
   if len(counts):
    sampled=rng.integers(0,len(counts),size=len(counts));ns[j]=counts[sampled].sum();ss[j]=sums[sampled].sum(axis=0)
  result[r]=measures(ns,ss,weights)
 return result,stats

def relative_months(days,start):
 dates=pd.Timestamp('2000-01-01')+pd.to_timedelta(np.asarray(days,dtype='float64'),unit='D')
 origin=pd.Timestamp(start);need(origin.day==11,'Expected original day-11 forecast origin')
 result=(dates.year-origin.year)*12+(dates.month-origin.month)-(dates.day<origin.day).astype(int)
 result=np.asarray(result,dtype='int64');need(((result>=0)&(result<12)).all(),'Forecast month outside 12 fixed bins');return result

def bootstrap_months(errors,labels,months,weights,rng,replicates):
 ns=np.zeros((12,3),dtype='int64');ss=np.zeros((12,3,2))
 for m in range(12):
  for j,g in enumerate(GROUPS):
   take=(months==m)&(labels==g);ns[m,j]=take.sum();ss[m,j]=errors[take].sum(axis=0)
 result=np.empty((replicates,4))
 for r in range(replicates):
  starts=rng.integers(0,12,size=4);sample=((starts[:,None]+np.arange(3))%12).ravel()
  counts=ns[sample].sum(axis=0)
  result[r]=measures(counts,ss[sample].sum(axis=0),weights) if counts.sum() else np.full(4,np.nan)
 return result,{'rows_by_relative_month':ns.sum(axis=1).tolist(),'stratum_rows_by_relative_month':{g:ns[:,j].tolist() for j,g in enumerate(GROUPS)},
  'months_with_rows':int((ns.sum(axis=1)>0).sum()),'block_length_months':3,'blocks_per_replicate':4,'circular_boundary_assumption':True}

def summarize(boot,point):
 result={}
 for name,left,right in [('overall',0,1),('weighted',2,3)]:
  deltas=boot[:,right]-boot[:,left];values=deltas[np.isfinite(deltas)]
  bounds=np.quantile(values,[.025,.975],method='linear').tolist() if len(values) else [None,None]
  result[name]={'v2_mae':float(point[left]),'v3_mae':float(point[right]),'point_delta_v3_minus_v2':float(point[right]-point[left]),
   'valid_replicates':len(values),'invalid_replicates':len(boot)-len(values),'percentile_2_5':bounds[0],'percentile_97_5':bounds[1],
   'negative_delta_fraction':float(np.mean(values<0)) if len(values) else None,'bootstrap_mean_delta':float(values.mean()) if len(values) else None}
 return result

def self_check():
 w=dict(zip(GROUPS,[.3,.7,0.]));labels=np.array(['seen','seen','new_nonmissing','new_nonmissing']);keys=np.array(['a','a','b','c'])
 e=np.array([[10.,7.],[50.,47.],[100.,97.],[200.,197.]])
 p=point_values(e,labels,w);need(np.allclose(p[[1,3]]-p[[0,2]],-3),'Point pairing')
 bp,stats=bootstrap_projects(e,labels,keys,w,np.random.default_rng(11),31)
 need(np.allclose(bp[:,1]-bp[:,0],-3) and np.allclose(bp[:,3]-bp[:,2],-3),'Project paired constant difference')
 need(stats['seen']['clusters']==1 and stats['seen']['largest_cluster_rows']==2,'Whole cluster support')
 equal,_=bootstrap_projects(e[:,[0,0]],labels,keys,w,np.random.default_rng(11),31);need(np.array_equal(equal[:,0],equal[:,1]),'Identical model invariant')
 ns=np.array([2,1,0]);ss=np.array([[60.,30.],[100.,70.],[0.,0.]])
 v=measures(ns,ss,w);need(np.allclose(v,[160/3,100/3,79,53.5]),'Counts not equally weighted groups')
 dates=pd.to_datetime(['2022-08-11','2022-09-10','2022-09-11','2023-08-10']);days=(dates-pd.Timestamp('2000-01-01')).days
 need(np.array_equal(relative_months(days,'2022-08-11'),[0,0,1,11]),'Relative month boundaries')
 bm,_=bootstrap_months(e,labels,np.array([0,0,6,6]),w,np.random.default_rng(12),100)
 valid=np.isfinite(bm[:,2:]).all(axis=1);need((~valid).any(),'Invalid replicate remains recorded')
 need(np.allclose((bm[:,1]-bm[:,0])[np.isfinite(bm[:,0])],-3),'Month paired constant difference')
 s=summarize(bp,p);need(s['overall']['negative_delta_fraction']==1.,'Delta sign')
 return {'checks':9,'passed':True}

def old_path(name,base):
 # All four historical provenance maps use project-root-relative paths,
 # including bare filenames such as requirements-round2.txt.
 candidate=(ROOT/name).resolve()
 need(candidate.is_relative_to(ROOT) and candidate.is_file(),'Missing/unsafe recorded input: '+name)
 return candidate

def local_sources():
 pending=[Path(__file__).resolve()];found=set()
 while pending:
  path=pending.pop()
  if path in found:continue
  found.add(path)
  for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
   names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
   for name in names:
    q=ROOT/'src'/(name.split('.')[0]+'.py')
    if q.is_file() and q not in found:pending.append(q)
 return sorted(found-{Path(__file__).resolve()})

def recorded_dependencies():
 result={}
 for stage in STAGES:
  folder=ROOT/'outputs/round6/evaluation'/stage
  d=locked(folder/'results.json')
  for field in ['input_artifact_sha256','output_artifact_sha256','data_sha256','source_sha256']:
   for name,digest in d[field].items():
    key=rel(old_path(name,folder));need(key not in result or result[key]==digest,'Conflicting historical input SHA')
    result[key]=digest
 return result

def compare_metrics(actual,expected,path='metrics'):
 if isinstance(expected,dict):
  need(isinstance(actual,dict) and set(actual)==set(expected),'Metric keys differ: '+path)
  for key,value in expected.items():compare_metrics(actual[key],value,path+'.'+key)
 elif isinstance(expected,list):
  need(isinstance(actual,list) and len(actual)==len(expected),'Metric list differs: '+path)
  for i,value in enumerate(expected):compare_metrics(actual[i],value,path+'.'+str(i))
 elif isinstance(expected,float) and 'mae' in path.rsplit('.',1)[-1].lower():
  need(np.isfinite(actual) and abs(actual-expected)<=1e-6,'MAE replay differs: '+path)
 else:need(actual==expected,'Metric value differs: '+path)


def input_files():
 paths=[ROOT/'runs/v3/snapshot_manifest.json',ROOT/'runs/v3/snapshot_manifest.sha256',ROOT/'runs/versions.json',
  ROOT/'outputs/submission_v2.csv',ROOT/'outputs/submission_v3.csv',ROOT/'outputs/round6/candidate_plan.json',ROOT/'outputs/round6/candidate_plan.sha256',
  ROOT/'outputs/round6/frozen_candidate.json',ROOT/'outputs/round6/frozen_candidate.sha256',ROOT/'outputs/round6/accepted_candidate.json',ROOT/'outputs/round6/accepted_candidate.sha256',
  ROOT/'data_cache/train.pkl',ROOT/'data_cache/train_features.pkl',ROOT/'data_cache/train.json',ROOT/'reports/feature_schema.json']
 paths += [ROOT/'src'/p for p in ['house_features.py','model_common.py','prepare_data.py','round3_evaluate.py','round6_common.py','round6_evaluate.py']]
 for stage in STAGES:
  ev=ROOT/'outputs/round6/evaluation'/stage;geo=ROOT/'outputs/round6/geography'/stage;model=ROOT/'outputs/round6/models'/stage/'geography_catboost'
  base=ROOT/('outputs/round6/baseline/auxiliary' if stage=='auxiliary' else 'outputs/round3/baseline/'+stage)
  paths += [ev/p for p in ['results.json','results.sha256','v2_control.npy','geography_blend.npy']]
  paths += [geo/p for p in ['manifest.json','manifest.sha256','train_source_cache_indices.npy']]
  paths += [model/p for p in ['fit_indices.npy','query_indices.npy','metadata.json']]
  paths += [base/p for p in ['fit_indices.npy','check_indices.npy']]
 paths += local_sources()
 paths += [ROOT/name for name in recorded_dependencies()]
 return sorted(set(paths))

def freeze():
 need(not (OUT/'plan.json').exists(),'Diagnostic plan already exists')
 checks=self_check();need(DOC.is_file(),'Human diagnostic plan missing')
 snap_path=ROOT/'runs/v3/snapshot_manifest.json';need(sha(snap_path)==SNAPSHOT_SHA,'v3 snapshot changed')
 snapshot=locked(snap_path);known={k:v['original_sha256'] for k,v in snapshot['copied_files'].items()}
 known.update({v['original_path']:v['sha256'] for v in snapshot['external_dependencies']})
 current={rel(p):sha(p) for p in input_files()}
 allowed={'runs/v3/snapshot_manifest.json','runs/v3/snapshot_manifest.sha256','runs/versions.json'}
 for name,digest in recorded_dependencies().items():need(current.get(name)==digest,'Historical result dependency SHA changed: '+name)
 for name,digest in current.items():
  if name not in allowed:need(known.get(name)==digest,'Input differs from released snapshot: '+name)
 registry=read(ROOT/'runs/versions.json');need(registry['recommended_version']=='v3','Unexpected recommendation')
 old=locked(ROOT/'outputs/round6/candidate_plan.json')
 weights=old['target_group_weights'];need(set(weights)==set(GROUPS) and abs(sum(weights.values())-1)<1e-12,'Invalid fixed weights')
 need(all(v>=0 for v in weights.values()),'Negative group weights')
 current[rel(DOC)]=sha(DOC);current[rel(Path(__file__))]=sha(Path(__file__))
 value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'conditional diagnostics of already released v3 against same-stage v2',
  'rules':RULES,'source_closure_sha256':{rel(p):sha(p) for p in [Path(__file__).resolve(),*local_sources()]},'fixed_target_group_weights':weights,'splits':old['splits'],'split_contract':old['split_contract'],
  'snapshot_sha256':SNAPSHOT_SHA,'dependency_sha256':current,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},
  'synthetic_checks':checks,'plan_creation_read_prediction_values':False,'plan_creation_read_target_values':False,
  'source_cache_indices_used_without_redefining_dedup':True,'frozen_recommendation':'v3','caution':CAUTION}
 lock_write(OUT/'plan.json',value)
 print(json.dumps({'status':'diagnostic_plan_frozen','plan_sha256':sha(OUT/'plan.json'),'dependencies':len(current),'synthetic_checks':checks}),flush=True)

def checked_plan():
 value=locked(OUT/'plan.json');need(value['rules']==RULES,'Diagnostic rules changed')
 need(value['runtime']=={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'Runtime changed')
 for name,digest in value['dependency_sha256'].items():
  p=(ROOT/name).resolve();need(p.is_relative_to(ROOT) and sha(p)==digest,'Diagnostic input/source changed: '+name)
 return value

def npy(p):return np.load(p,allow_pickle=False)
def array_sha(v):return hashlib.sha256(np.asarray(v,dtype='<i8').tobytes()).hexdigest()
def render(results):
 lines=['# v3 已保存預測的穩定度診斷','',CAUTION,'',
 '模型與既有接受決策沒有變更。此次診斷讀取 train 目標值計算固定預測誤差，與 R10 備註純 X 盤點分開；未用它補詞、調參或改融合比例。','',
 '差值定義為 **v3 MAE − v2 MAE**，負值代表 v3 較好。單位為新臺幣元。區間是 2,000 次抽樣的第 2.5／97.5 百分位，並非校正選型後的信賴保證。','',
 '| 窗口 | 抽樣單位 | 指標 | 原資料差值 | 2.5% | 97.5% | 抽樣改善比例 | 有效次數 |',
 '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
 def number(v):return '無法估計' if v is None else f'{v:,.2f}'
 for stage in STAGES:
  for family in ['project','month']:
   for metric,v in results['stages'][stage][family]['summary'].items():
    frac=v['negative_delta_fraction'];fraction='無法估計' if frac is None else f'{frac:.2%}'
    lines.append(f'| {stage} | {family} | {metric} | {number(v["point_delta_v3_minus_v2"])} | {number(v["percentile_2_5"])} | {number(v["percentile_97_5"])} | {fraction} | {v["valid_replicates"]} |')
 lines += ['', '## 如何閱讀','',
 '- project：每層按縣市、行政區與完整建案名稱分群，整案有放回抽樣；保留同案交易的相關性。缺名列有固定地址／道路／單列備援，實際使用數見結果檔。',
 '- month：相對預測起點的 12 個月份，每次抽 4 段連續 3 月，邊界環繞；僅為時間敏感性檢查。十二個月份資料有限，環繞會把年度末與初相接，並不完整描述景氣相關性。',
 '- overall：抽樣誤差總和除以抽樣資料列總數，沒有把大小建案等權平均。weighted：先計算每層 MAE，再套原已固定的 test 建案新舊比例。',
 '- 正權重層若在某次抽樣無資料，該次 weighted 記為無效且不重抽；結果同時保留有效與無效次數。改善比例不是 p 值。',
 '- 此診斷條件於固定模型，並沒有估計重新訓練或重新選模型的變異；Public／Private 分數仍未知（僅 v2 Public 有已知歷史紀錄）。','',
 '## 支持與集中度','', '| 窗口 | 分層 | 列數 | 建案群組數 | 最大群組列占比 | 依列權重計算的有效群組數 |',
 '| --- | --- | ---: | ---: | ---: | ---: |']
 for stage in STAGES:
  for group in GROUPS:
   v=results['stages'][stage]['project']['support'][group]
   share='無資料' if v['largest_cluster_row_share'] is None else f'{v["largest_cluster_row_share"]:.2%}'
   lines.append(f'| {stage} | {group} | {v["rows"]:,} | {v["clusters"]:,} | {share} | {number(v["row_weight_effective_cluster_count"])} |')
 lines += ['', '[事前方案](v3_cluster_stability_plan.md) · [完整結果](../outputs/diagnostics/v3_cluster_stability/results.json)',
 '', '已核對原 MAE 至 0.000001 元容許誤差，並檢查來源、去重索引、日期切分與預測順序。各階段是分別的重抽樣，沒有將重疊年度當成獨立樣本合併。','']
 return '\n'.join(lines)

def run():
 spec=checked_plan();need(not (OUT/'results.json').exists() and not REPORT.exists(),'Results/report already exist; no repeat diagnostic')
 started=time.monotonic();raw=pd.read_pickle(ROOT/'data_cache/train.pkl');features=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
 schema=read(ROOT/'reports/feature_schema.json');need(len(raw)==len(features)==455784 and list(features.columns)==schema['columns'],'Input alignment/schema changed')
 idx=npy(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy')
 need(idx.shape==(455255,) and idx.dtype==np.dtype('int64') and np.array_equal(idx,np.unique(idx)) and ((idx>=0)&(idx<len(raw))).all(),'Invalid canonical source indices')
 X=features.iloc[idx].reset_index(drop=True);y=raw['總價元'].to_numpy(dtype='float64')[idx];del raw,features
 need(np.isfinite(y).all() and (y>0).all(),'Invalid observed targets')
 results={'purpose':spec['purpose'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),'rules':RULES,
  'caution':CAUTION,'selection_changed':False,'recommended_version':'v3','target_values_read_for_fixed_prediction_diagnostics':True,
  'remark_feature_research_receives_diagnostic_values':False,'stages':{},'artifact_sha256':{}}
 days=X.trade_day_index.to_numpy();weights=spec['fixed_target_group_weights']
 for stage_index,stage in enumerate(STAGES):
  start,end=spec['splits'][stage];start_day=(pd.Timestamp(start)-pd.Timestamp('2000-01-01')).days;end_day=(pd.Timestamp(end)-pd.Timestamp('2000-01-01')).days
  fi=np.flatnonzero(np.isfinite(days)&(days<start_day));qi=np.flatnonzero(np.isfinite(days)&(days>=start_day)&(days<=end_day));contract=spec['split_contract'][stage]
  need(not np.intersect1d(fi,qi).size and len(fi)==contract['fit_rows'] and len(qi)==contract['query_rows'],'Stage row count differs')
  need(array_sha(fi)==contract['fit_indices_content_sha256'] and array_sha(qi)==contract['query_indices_content_sha256'],'Stage content indices differ')
  need(np.array_equal(idx,npy(ROOT/'outputs/round6/geography'/stage/'train_source_cache_indices.npy')),'Canonical source map differs by stage')
  model=ROOT/'outputs/round6/models'/stage/'geography_catboost';base=ROOT/('outputs/round6/baseline/auxiliary' if stage=='auxiliary' else 'outputs/round3/baseline/'+stage)
  for p,expected in [(model/'fit_indices.npy',fi),(model/'query_indices.npy',qi),(base/'fit_indices.npy',fi),(base/'check_indices.npy',qi)]:need(np.array_equal(npy(p),expected),'Prediction order differs: '+str(p))
  ev=ROOT/'outputs/round6/evaluation'/stage;old=locked(ev/'results.json');need(old['stage']==stage and old['target_group_weights']==weights,'Prior result identity differs')
  predictions=np.column_stack([npy(ev/'v2_control.npy'),npy(ev/'geography_blend.npy')]);need(predictions.dtype==np.dtype('float64') and predictions.shape==(len(qi),2) and np.isfinite(predictions).all() and (predictions>=0).all(),'Saved prediction validity')
  Xq=X.iloc[qi].reset_index(drop=True);labels=group_labels(X.iloc[fi],Xq);errors=np.abs(y[qi,None]-predictions);point=point_values(errors,labels,weights)
  for j,name in enumerate(['v2_control','geography_blend']):
   compare_metrics(original_score(y[qi],predictions[:,j],X.iloc[fi].reset_index(drop=True),Xq,start,weights),old['metrics'][name])
   prior=old['metrics'][name];need(abs(point[j]-prior['overall_mae'])<=1e-6 and abs(point[j+2]-prior['weighted_mae'])<=1e-6,'Saved point MAE replay differs')
  keys,fallback=cluster_keys(Xq,qi);months=relative_months(Xq.trade_day_index,start)
  bp,support=bootstrap_projects(errors,labels,keys,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],stage_index,0])),RULES['replicates'])
  bm,month_support=bootstrap_months(errors,labels,months,weights,np.random.default_rng(np.random.SeedSequence([RULES['seed'],stage_index,1])),RULES['replicates'])
  value={'rows':len(qi),'fit_rows':len(fi),'query_indices_content_sha256':array_sha(qi),'point_mae_replay_verified':True,'all_original_metrics_replay_verified':True,'missing_project_fallback_rows':fallback,
   'project':{'summary':summarize(bp,point),'support':support},'month':{'summary':summarize(bm,point),'support':month_support}}
  for name,array in [('project',bp),('month',bm)]:
   path=OUT/(stage+'_'+name+'_replicates.npy');need(not path.exists(),'Replicates already exist')
   np.save(path,array,allow_pickle=False);results['artifact_sha256'][rel(path)]=sha(path)
   delta_path=OUT/(stage+'_'+name+'_deltas_and_valid.npz');need(not delta_path.exists(),'Delta artifact already exists')
   np.savez(delta_path,overall_delta=array[:,1]-array[:,0],weighted_delta=array[:,3]-array[:,2],overall_valid=np.isfinite(array[:,1]-array[:,0]),weighted_valid=np.isfinite(array[:,3]-array[:,2]))
   results['artifact_sha256'][rel(delta_path)]=sha(delta_path)
  results['stages'][stage]=value
  print(json.dumps({'event':'diagnostic_stage_completed','stage':stage,'summary':value['project']['summary']},ensure_ascii=True),flush=True)
 checked_plan();results['elapsed_seconds']=time.monotonic()-started
 with REPORT.open('x',encoding='utf-8') as f:f.write(render(results))
 results['artifact_sha256'][rel(REPORT)]=sha(REPORT);lock_write(OUT/'results.json',results)
 print(json.dumps({'status':'complete','results_sha256':sha(OUT/'results.json'),'elapsed_seconds':results['elapsed_seconds']}),flush=True)

def verify():
 checked_plan();result=locked(OUT/'results.json');need(result['plan_sha256']==sha(OUT/'plan.json'),'Result plan differs')
 for name,digest in result['artifact_sha256'].items():need(sha(ROOT/name)==digest,'Diagnostic output changed: '+name)
 for stage in STAGES:
  for family in ['project','month']:
   boot=npy(OUT/(stage+'_'+family+'_replicates.npy'));saved=result['stages'][stage][family]['summary']
   point=np.array([saved['overall']['v2_mae'],saved['overall']['v3_mae'],saved['weighted']['v2_mae'],saved['weighted']['v3_mae']])
   need(boot.shape==(RULES['replicates'],4) and summarize(boot,point)==saved,'Saved replicate summary differs')
   with np.load(OUT/(stage+'_'+family+'_deltas_and_valid.npz'),allow_pickle=False) as d:
    for metric,left,right in [('overall',0,1),('weighted',2,3)]:
     delta=boot[:,right]-boot[:,left]
     need(np.array_equal(d[metric+'_delta'],delta,equal_nan=True) and np.array_equal(d[metric+'_valid'],np.isfinite(delta)),'Explicit delta/valid record differs')
 need(REPORT.read_text(encoding='utf-8')==render(result),'Human report differs from results')
 print(json.dumps({'status':'verified','results_sha256':sha(OUT/'results.json'),'recommendation':'v3'}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['self-check','freeze','run','verify']);args=parser.parse_args()
 if args.command=='self-check':print(json.dumps(self_check()),flush=True)
 else:{'freeze':freeze,'run':run,'verify':verify}[args.command]()
