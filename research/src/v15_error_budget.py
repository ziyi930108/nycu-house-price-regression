"""One CPU-only error budget of saved R54/v15 predictions; fixed single-field slices."""
from pathlib import Path
from datetime import datetime,timezone
import contextlib,hashlib,json,sys,time,traceback
import numpy as np
import pandas as pd
from round3_evaluate import group_labels,norm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v15_error_budget'
REPORT=ROOT/'reports/v15_error_budget.md'
RESULT=ROOT/'reports/v15_error_budget.json'
STAGES=('development','confirmation','auxiliary')
FIELDS=('building_type','transaction_type','main_use','address_city')
SUPPORT=500
KT='outputs/round6/geography/development/train_source_cache_indices.npy'
R54_SHA='05023065ff34c40d9d6514b77c01d4b9278b3fa34a29665975448976aed9950a'
LABEL={'development':'開發','confirmation':'最近年度','auxiliary':'較早年度'}
TITLE={'building_type':'建物型態','transaction_type':'交易標的','main_use':'主要用途','address_city':'城市'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def write(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')

def summarize(labels,e,test_counts,test_n):
 labels=np.asarray(labels,dtype=str);a=np.abs(e);total=float(a.sum());records=[]
 for label in sorted(set(labels)|set(test_counts)):
  m=labels==label;n=int(m.sum());tn=int(test_counts.get(label,0));z=e[m];az=a[m]
  records.append({'group':str(label),'rows':n,'supported_500':n>=SUPPORT,'row_share':n/len(e),
    'mae_TWD':float(az.mean()) if n else None,'absolute_error_sum_TWD':float(az.sum()),
    'absolute_error_share':float(az.sum()/total),'signed_bias_TWD':float(z.mean()) if n else None,
    'test_rows':tn,'test_share':tn/test_n,'test_minus_query_share_pp':100*(tn/test_n-n/len(e))})
 return sorted(records,key=lambda d:(-d['absolute_error_sum_TWD'],d['group']))

def render(r):
 lines=['# v15：可事先辨認的房屋群與剩餘誤差','',
  '使用 v15 對應的 R54 三期保存預測，固定檢查城市、建物型態、交易標的、主要用途四個單欄，以及原新／舊建案分組；沒有交互組合搜尋、價位挑群、模型呼叫、價格修正或新 CSV。與舊 v5 盤點不同，本次回答目前 v15 的剩餘誤差。',
  '', '以下均為已重用年度的事後描述。正 bias 是平均高估，負值是低估；MAE 與 bias 單位為元。每期至少 500 筆標記「足」，少於 500 筆仍完整保存，不能用高 MAE 小群取代整體誤差占比。test 只讀 X，沒有 test／Private 真價。',
  '', '| 窗口 | Query 筆數 | v15 整體 MAE | 原固定加權 MAE | 平均 bias |','|---|---:|---:|---:|---:|']
 for s in STAGES:
  a=r['stages'][s];lines.append(f"| {LABEL[s]} | {a['query_rows']:,} | {a['mae_TWD']:,.0f} | {a['weighted_mae_TWD']:,.0f} | {a['signed_bias_TWD']:+,.0f} |")
 lines+=['','## 最新年度的主要可辨認群','',
  '每個單欄固定展示三期合計絕對誤差最多的前兩類（只作摘要排序），完整所有類別見 JSON。同一交易同時屬於四個單欄，跨欄誤差占比不可相加。test 占比是完整 47,073 筆 X 的單欄分布；不是 test 誤差估計。','',
  '| 單欄／類別 | 最近年度筆數 | 支持 | Query占比 | 誤差占比 | MAE | bias | Test占比 |',
  '|---|---:|:---:|---:|---:|---:|---:|---:|']
 for field in FIELDS:
  for name in r['summary_groups'][field]:
   a=next(x for x in r['stages']['confirmation']['groups'][field] if x['group']==name)
   lines.append(f"| {TITLE[field]}：{name} | {a['rows']:,} | {'足' if a['supported_500'] else '少'} | {a['row_share']:.2%} | {a['absolute_error_share']:.2%} | {a['mae_TWD']:,.0f} | {a['signed_bias_TWD']:+,.0f} | {a['test_share']:.2%} |")
 lines+=['','## 同群三年是否一致','', '順序為開發／最近／較早年度；括號為各期支持筆數。這裡比較方向與誤差集中，不把不同年度 MAE 的大小當作因果效果。','',
  '| 單欄／類別 | 三期誤差占比 | 三期 bias | 三期支持 |','|---|---:|---:|---:|']
 for field in FIELDS:
  for name in r['summary_groups'][field]:
   aa=[next(x for x in r['stages'][s]['groups'][field] if x['group']==name) for s in STAGES]
   shares=' / '.join(f"{a['absolute_error_share']:.1%}" for a in aa)
   biases=' / '.join('—' if a['signed_bias_TWD'] is None else f"{a['signed_bias_TWD']:+,.0f}" for a in aa)
   support=' / '.join(f"{a['rows']:,}{'*' if not a['supported_500'] else ''}" for a in aa)
   lines.append(f'| {TITLE[field]}：{name} | {shares} | {biases} | {support} |')
 lines+=['','## 新／舊建案','',
  '歷史是否見過依各期 fit 的 city＋district＋project 原規則。Test占比引用原完整 train＋valid 建案目錄的固定權重（seen 36.62%、new 63.38%）；兩者的 fit 時點不同，不能誤稱同一目錄。','',
  '| 窗口／群 | 筆數 | Query占比 | 誤差占比 | MAE | bias | Test占比 |','|---|---:|---:|---:|---:|---:|---:|']
 for s in STAGES:
  for a in r['stages'][s]['groups']['project_group']:
   if not a['rows']:continue
   lines.append(f"| {LABEL[s]}／{a['group']} | {a['rows']:,} | {a['row_share']:.2%} | {a['absolute_error_share']:.2%} | {a['mae_TWD']:,.0f} | {a['signed_bias_TWD']:+,.0f} | {a['test_share']:.2%} |")
 lines+=['','## 解讀與下一步界線','',
  '應優先看同時具有足夠筆數、較大誤差總額、跨年持續現象且 test 仍有支持的群。高 MAE 可能反映金額尺度、異質性或少数大誤差，不能直接推導為固定加減價係數；平均 bias 也可能相互抵消。城市／用途與新案群是重疊描述，沒有因果歸因或可保證改善的新候選。',
  '', '完整 JSON 保存每一類（含未支持或 test-only 類別）、三期 MAE／bias／誤差額／比例與來源 SHA。這次不依真實價格、最大個案或新分數另設群，也不把過去 v5 的結果直接當作 v15 證據。',
  '', f"唯一 CPU 盤點用時 {r['elapsed_seconds']:.3f} 秒；三期整體、固定加權及原新舊建案 MAE 均核對保存值。fit／model load／predict 為 0。",
  '', '[完整 JSON](v15_error_budget.json) · [來源程式](../src/v15_error_budget.py) · [執行收據](../outputs/v15_error_budget/execution.json) · [舊 v5 盤點](error_budget_20260926.md)','']
 with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))

def run():
 tick=time.monotonic();inputs={}
 def small(name,expected=None):
  p=ROOT/name;h=sha(p)
  if expected:assert h==expected,name
  receipt=p.with_suffix('.sha256');assert receipt.read_text().split()[0]==h
  inputs[name]=h;inputs[receipt.relative_to(ROOT).as_posix()]=sha(receipt)
  return json.loads(p.read_text(encoding='utf8'))
 def checked(name,h):
  p=ROOT/name;assert sha(p)==h,name;inputs[name]=h;return p
 p48=small('outputs/round48/plan.json');r48=small('outputs/round48/results.json');r54=small('outputs/round54/results.json',R54_SHA)
 release=small('outputs/round54/final_export/release.json')
 assert release['status']=='experimental_v15_full_R54_generated' and release['historical_results_sha256']==R54_SHA and release['formula']==r54['formula']
 final=small('outputs/round55/final/trial_release.json',release['parent_final_release_sha256'])
 assert r54['checked_stages']==r48['checked_stages']==list(STAGES)
 for name in ('data_cache/train_features.pkl','data_cache/train.pkl',KT):checked(name,p48['input_sha256'][name])
 testpath=checked('data_cache/test_features.pkl',final['input_sha256']['data_cache/test_features.pkl'])
 keep=np.load(ROOT/KT,allow_pickle=False);xf=pd.read_pickle(ROOT/'data_cache/train_features.pkl');raw=pd.read_pickle(ROOT/'data_cache/train.pkl')
 assert len(xf)==len(raw)==455784 and keep.dtype==np.int64 and len(keep)==455255 and len(np.unique(keep))==len(keep)
 X=xf.iloc[keep].reset_index(drop=True);y=raw['總價元'].to_numpy(dtype=np.float64)[keep]
 del xf,raw
 test=pd.read_pickle(testpath);assert len(test)==47073
 tcounts={f:norm(test[f]).value_counts().to_dict() for f in FIELDS}
 project_test={g:int(round(w*len(test))) for g,w in p48['target_group_weights'].items()}
 assert sum(project_test.values())==len(test) and all(project_test[g]/len(test)==w for g,w in p48['target_group_weights'].items())
 result={'status':'running','method':'v15 full R54 saved historical predictions','stage_order':list(STAGES),
  'fixed_single_fields':list(FIELDS),'minimum_supported_rows':SUPPORT,'support_threshold_is_descriptive_not_gate':True,
  'new_model_fits':0,'model_loads':0,'model_prediction_calls':0,'new_price_adjustment':False,'test_labels_available':False,
  'fresh_blind_validation':False,'post_hoc_reused_windows':True,'signed_error_definition':'prediction - observed training price',
  'test_project_reference':'saved R48 full train+valid catalog target_group_weights; historical labels use own fit catalog',
  'target_group_weights':p48['target_group_weights'],'test_rows':len(test),'stages':{},'input_sha256':inputs,
  'source_sha256':{n:sha(ROOT/n) for n in ('src/v15_error_budget.py','src/round3_evaluate.py','src/model_common.py')},
  'prior_report_sha256':sha(ROOT/'reports/error_budget_20260926.md')}
 for s in STAGES:
  rr=r54['stages'][s];fiqi=[]
  for role in ('fit','query'):
   name=f'outputs/round48/{s}/{role}_indices.npy';h=r48['stages'][s]['artifact_sha256'][name]
   assert h==rr['artifact_sha256'][f'outputs/round54/{s}/{role}_indices.npy']
   a=np.load(checked(name,h),allow_pickle=False);c=p48['split_contract'][s]
   assert a.dtype==np.int64 and a.shape==(c[role+'_rows'],) and (np.diff(a)>0).all() and 0<=a.min()<=a.max()<len(X)
   assert hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()==c[role+'_indices_content_sha256'];fiqi.append(a)
  fi,qi=fiqi;assert not np.intersect1d(fi,qi).size
  name=f'outputs/round54/{s}/predictions.npy';pred=np.load(checked(name,rr['prediction_sha256']),allow_pickle=False)
  assert pred.dtype==np.float64 and pred.shape==(len(qi),) and np.isfinite(pred).all() and (pred>0).all()
  yy=y[qi];assert np.isfinite(yy).all() and (yy>0).all();error=pred-yy;q=X.iloc[qi].reset_index(drop=True)
  labels=group_labels(X.iloc[fi].reset_index(drop=True),q);metric=rr['metrics']['R54']
  mae=float(np.abs(error).mean());assert mae==metric['overall_mae']
  means={}
  for g in p48['target_group_weights']:
   mask=labels==g;means[g]=float(np.abs(error[mask]).mean()) if mask.any() else None
   assert int(mask.sum())==metric['strata'][g]['rows'] and means[g]==metric['strata'][g]['mae']
  weighted=sum(p48['target_group_weights'][g]*v for g,v in means.items() if v is not None)
  assert abs(weighted-metric['weighted_mae'])<1e-8
  groups={f:summarize(norm(q[f]),error,tcounts[f],len(test)) for f in FIELDS}
  groups['project_group']=summarize(labels,error,project_test,len(test))
  for records in groups.values():
   assert sum(a['rows'] for a in records)==len(qi) and sum(a['test_rows'] for a in records)==len(test)
   assert abs(sum(a['absolute_error_share'] for a in records)-1)<1e-12
  result['stages'][s]={'fit_rows':len(fi),'query_rows':len(qi),'dates':p48['splits'][s],'mae_TWD':mae,
   'weighted_mae_TWD':metric['weighted_mae'],'signed_bias_TWD':float(error.mean()),'absolute_error_sum_TWD':float(np.abs(error).sum()),
   'saved_overall_weighted_strata_reproduced':True,'groups':groups}
  print(json.dumps({'stage':s,'mae':mae,'groups':{f:len(v) for f,v in groups.items()}},ensure_ascii=False),flush=True)
 result['summary_groups']={}
 for field in FIELDS:
  total={}
  for s in STAGES:
   for a in result['stages'][s]['groups'][field]:total[a['group']]=total.get(a['group'],0)+a['absolute_error_sum_TWD']
  result['summary_groups'][field]=sorted(total,key=lambda k:(-total[k],k))[:2]
 result.update(status='completed_saved_predictions_only',elapsed_seconds=time.monotonic()-tick)
 assert all(sha(ROOT/n)==h for n,h in result['source_sha256'].items())
 write(RESULT,result);render(result)

def execute():
 assert not OUT.exists() and not RESULT.exists() and not REPORT.exists(),'No overwrite/retry'
 OUT.mkdir(parents=True);tick=time.monotonic();start=datetime.now(timezone.utc).isoformat();code=1;failure=None
 with (OUT/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
  try:run();code=0
  except Exception:failure=traceback.format_exc();print(failure,flush=True)
 event={'exit_code':code,'failure':failure,'started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
  'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'log_sha256':sha(OUT/'run.log'),
  'GPU_calls':0,'automatic_retry':False}
 write(OUT/'execution.json',event);print(json.dumps(event),flush=True);return code

if __name__=='__main__':raise SystemExit(execute())
