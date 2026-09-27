"""One post-hoc X-only city/project mixture check. No fit or blend tuning.
Use exact test city x seen/new/missing proportions, retain cells with >=200
historical validation rows, report unmatched mass explicitly. Compare saved
v5/v7/R40(v8 method); this does not identify Public or Private row membership.
"""
from pathlib import Path
import hashlib, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from round3_evaluate import group_labels, norm
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/error_budget_20260926/test_mix_review.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
started=time.monotonic()
assert not OUT.exists()
keep=np.load(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy',allow_pickle=False)
xraw=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
raw=pd.read_pickle(ROOT/'data_cache/train.pkl')
X=xraw.iloc[keep].reset_index(drop=True)
y=raw['總價元'].to_numpy(dtype='float64')[keep]
del xraw,raw
valid=pd.read_pickle(ROOT/'data_cache/valid_features.pkl')
fullkeep=np.load(ROOT/'outputs/round6/geography/final/full_source_cache_indices.npy',allow_pickle=False)
full=pd.concat([X,valid],ignore_index=True).iloc[fullkeep].reset_index(drop=True)
test=pd.read_pickle(ROOT/'data_cache/test_features.pkl')
assert len(full)==467221 and len(test)==47073
labels=group_labels(full,test)
keys=norm(test.address_city)+'|'+pd.Series(labels)
counts=keys.value_counts()
weights=counts/len(test)
result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'method':'Post-hoc city x project-history target composition; minimum 200 historical query rows per cell; no fitting, clipping, smoothing, coefficient search or public/private membership.',
 'source_sha256':sha(Path(__file__)),'private_labels_available':False,'minimum_historical_cell_rows':200,'target_weights_from_X_only':True,'target_cells':{str(k):{'rows':int(counts[k]),'weight':float(v)} for k,v in weights.items()},'stages':{}}
for stage in ['development','confirmation','auxiliary']:
 f39=ROOT/'outputs/round39/evaluation'/stage
 fi,qi=[np.load(f39/(x+'_indices.npy'),allow_pickle=False) for x in ['fit','query']]
 q=X.iloc[qi].reset_index(drop=True)
 group=norm(q.address_city)+'|'+pd.Series(group_labels(X.iloc[fi].reset_index(drop=True),q))
 history=group.value_counts()
 supported=history.reindex(weights.index,fill_value=0)>=200
 coverage=float(weights[supported].sum())
 supported_weights=weights[supported]/coverage
 paths={'v5':ROOT/'outputs/round36'/stage/'log_spatial_combination.npy',
        'v7':f39/'city_road_spatial_half_raw_half_log.npy',
        'v8_method':ROOT/'outputs/round40'/stage/'three_branch_equal_mean.npy'}
 records={}
 for name,path in paths.items():
  pred=np.load(path,allow_pickle=False)
  assert pred.shape==(len(qi),) and np.isfinite(pred).all()
  abs_error=np.abs(pred-y[qi])
  bycell=pd.Series(abs_error).groupby(group).mean()
  records[name]={'original_mae':float(abs_error.mean()),'target_mix_supported_mae':float((bycell.reindex(supported_weights.index)*supported_weights).sum()),'prediction_sha256':sha(path)}
 details=[]
 for cell,w in weights.items():
  details.append({'cell':cell,'test_rows':int(counts[cell]),'test_weight':float(w),'historical_rows':int(history.get(cell,0)),'supported':bool(supported[cell])})
 result['stages'][stage]={'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'full_test_expected_mae_identifiable_from_this_table':bool(supported.all()),'models':records,'cells':details}
result['elapsed_seconds']=time.monotonic()-started
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
p=ROOT/'reports/error_budget_20260926.md'
s=p.read_text(encoding='utf-8')
s+='\n## 測試資料組成的補充比較\n\n測試資料的臺北市占比為7.92%，最近歷史驗證年度僅3.71%。因此另用完整test的無標籤城市×新舊建案比例，重加權已保存驗證誤差；不改任何模型、提交或融合比例。只保留歷史驗證至少200筆的分組，未支援的test比例另列，不以零誤差補齊。這是重用歷史資料的事後診斷，不是Private分數估計保證。\n\n| 年度 | test組成覆蓋 | v5匹配MAE | v7匹配MAE | v8方法匹配MAE |\n|---|---:|---:|---:|---:|\n'
for stage,d in result['stages'].items():
 vals=[d['models'][k]['target_mix_supported_mae'] for k in ['v5','v7','v8_method']]
 s+=f'| {stage} | {100*d["test_mass_covered"]:.2f}% | {vals[0]:,.0f} | {vals[1]:,.0f} | {vals[2]:,.0f} |\n'
s+='\n各欄仅比較同一年度、同一支援範圍。v8方法在此指已完成R40三分支回測，並不表示最終CSV已完成或已上傳。[完整分組與限制](../outputs/error_budget_20260926/test_mix_review.json)。\n'
p.write_text(s,encoding='utf-8')
print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'stages':{k:{x:v for x,v in d.items() if x!='cells'} for k,d in result['stages'].items()}},ensure_ascii=False))
