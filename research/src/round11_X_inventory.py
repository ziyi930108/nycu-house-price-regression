"""Development-only pure-X overlap inventory; no domain or price model fitting."""
from __future__ import annotations
import argparse, ast, hashlib, json, platform, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round11/research'
REPORT=ROOT/'reports/round11_X_inventory.md'
SNAPSHOT_SHA='77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
CACHE='data_cache/train_features.pkl'
KEEP='outputs/round6/geography/development/train_source_cache_indices.npy'
FIT='outputs/round3/baseline/development/fit_indices.npy'
QUERY='outputs/round3/baseline/development/check_indices.npy'
GEO='outputs/round6/geography/development/manifest.json'
SCHEMA='reports/feature_schema.json'
MISSING='__MISSING__'
CAT=['address_city','city_original_district','building_type','main_use','transaction_type','parking_type']
NUM=['building_area','land_area','parking_area','rooms','living_rooms','bathrooms','floor_min','floor_max','total_floors','parking_count']
Q=[0,.01,.05,.25,.5,.75,.95,.99,1]
FOLD_RULE='SHA256(R11|20260925| + compact Unicode JSON group key), first 8 bytes little-endian modulo 5; formal pre-diagnostic rule, no balance search'


def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def locked(path):
 path=Path(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip().split()[0],'Receipt mismatch: '+str(path));return read(path)
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def norm(series):
 values=series.fillna(MISSING).astype(str).str.strip().reset_index(drop=True)
 return values.mask(values.isin(['','nan','None','<NA>','NaT']),MISSING)
def array_sha(array,dtype='<i8'):return hashlib.sha256(np.asarray(array,dtype=dtype).tobytes()).hexdigest()
def frame_sha(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()


def make_z(X):
 z=pd.DataFrame(index=X.index)
 for col in CAT:
  if col=='city_original_district':
   city,district=norm(X['address_city']),norm(X['district']);z[col]=pd.Series([json.dumps([a,b],ensure_ascii=False,separators=(',',':')) if a!=MISSING and b!=MISSING else MISSING for a,b in zip(city,district)])
  else:z[col]=norm(X[col])
 for col in NUM:z[col]=X[col].copy()
 need(z.index.equals(X.index),'Z row alignment differs')
 return z


def group_keys(X,source_rows):
 need(len(X)==len(source_rows),'Group source rows misaligned')
 fields={c:norm(X[c]).to_numpy() for c in ['address_city','address_district','project_name','address_raw','address_road','address_section']}
 keys=[];kinds=[]
 for i in range(len(X)):
  city,district,project=(fields[c][i] for c in ['address_city','address_district','project_name'])
  if project!=MISSING:kind='named';key=['project',city,district,project]
  elif fields['address_road'][i]!=MISSING:kind='missing_road';key=[kind,city,district,fields['address_road'][i],fields['address_section'][i]]
  elif fields['address_raw'][i]!=MISSING:kind='missing_address';key=[kind,city,district,fields['address_raw'][i]]
  else:kind='unresolved';key=['unresolved_missing_project']
  keys.append(json.dumps(key,ensure_ascii=False,separators=(',',':')));kinds.append(kind)
 hashes=pd.Series([hashlib.sha256(k.encode('utf-8')).hexdigest() for k in keys],index=X.index)
 folds=np.asarray([int.from_bytes(hashlib.sha256(('R11|20260925|'+k).encode('utf-8')).digest()[:8],'little')%5 for k in keys],dtype='uint8')
 return hashes,pd.Series(kinds,index=X.index),folds


def numeric_summary(fit,query,col):
 a=pd.to_numeric(fit[col],errors='raise').to_numpy(dtype='float64');b=pd.to_numeric(query[col],errors='raise').to_numpy(dtype='float64');out={}
 for name,v in [('fit',a),('query',b)]:
  finite=v[np.isfinite(v)];out[name]={'rows':len(v),'finite_rows':len(finite),'missing_rows':int(np.isnan(v).sum()),'infinite_rows':int(np.isinf(v).sum()),'zero_rows':int(np.sum(v==0)),'negative_rows':int(np.sum(v<0)),'quantiles':{str(q):float(np.quantile(finite,q)) if len(finite) else None for q in Q}}
 finite=a[np.isfinite(a)];bfinite=np.isfinite(b)
 out['query_outside_fit_finite_range_rows']=int(np.sum(bfinite & ((b<finite.min())|(b>finite.max())))) if len(finite) else None
 return out


def categorical_summary(fit,query,col):
 f=fit[col].value_counts(dropna=False);q=query[col].value_counts(dropna=False);known=set(f.index)-{MISSING};notmissing=query[col].ne(MISSING);unseen=notmissing & ~query[col].isin(known)
 levels=sorted(set(f.index)|set(q.index));distribution=[{'column':col,'category':k,'fit_rows':int(f.get(k,0)),'query_rows':int(q.get(k,0)),'fit_fraction':float(f.get(k,0)/len(fit)),'query_fraction':float(q.get(k,0)/len(query))} for k in levels]
 return {'fit_distinct_nonmissing':len(known),'query_distinct_nonmissing':len(set(q.index)-{MISSING}),'fit_missing_rows':int(f.get(MISSING,0)),'query_missing_rows':int(q.get(MISSING,0)),'query_unseen_nonmissing_rows':int(unseen.sum()),'query_unseen_fraction_all_rows':float(unseen.mean()),'query_unseen_fraction_nonmissing':float(unseen.sum()/notmissing.sum()) if notmissing.any() else None,'total_variation_categorical':float(sum(abs(d['fit_fraction']-d['query_fraction']) for d in distribution)/2),'unseen_categories':[{'category':k,'query_rows':int(q[k])} for k in q.index if k!=MISSING and k not in known]},distribution


def self_check():
 checks=[]
 def test(name,ok):need(ok,'Synthetic failure '+name);checks.append(name)
 X=pd.DataFrame({'address_city':['臺北市']*5,'address_district':['中正區']*5,'project_name':['案A','案A',MISSING,MISSING,MISSING],'address_raw':['甲','乙','同址',MISSING,MISSING],'address_road':['路']*4+[MISSING],'address_section':['一段']*5})
 h,k,f=group_keys(X,np.arange(5));test('named_project_all_rows_same_fold',h[0]==h[1] and f[0]==f[1]);test('fallback_priority',k.tolist()==['named','named','missing_road','missing_road','unresolved'])
 h2,k2,f2=group_keys(X.iloc[[1]].reset_index(drop=True),np.array([1]));test('cross_domain_same_named_project_fold',h2[0]==h[1] and f2[0]==f[1])
 changed=X.copy();changed.loc[4,'project_name']='另案';hc,kc,fc=group_keys(changed,np.arange(5));test('named_overrides_fallback',kc[4]=='named')
 unresolved=pd.concat([X.iloc[[4]],X.iloc[[4]]],ignore_index=True);unresolved.loc[1,'address_city']='另一城市';hu,ku,fu=group_keys(unresolved,np.array([4,1000]));test('global_unresolved_held_together',hu[0]==hu[1] and fu[0]==fu[1]);test('road_first_ignores_different_addresses',h[2]==h[3] and f[2]==f[3])
 test('norm_matches_missing_semantics',norm(pd.Series([None,' nan ','','NaT','甲'])).tolist()==[MISSING]*4+['甲'])
 s,rows=categorical_summary(pd.DataFrame({'x':['A','A',MISSING]}),pd.DataFrame({'x':['A','B',MISSING]}),'x');test('missing_not_called_unseen',s['query_unseen_nonmissing_rows']==1 and s['query_missing_rows']==1)
 n=numeric_summary(pd.DataFrame({'x':[0.,2.,np.nan]}),pd.DataFrame({'x':[-1.,1.,np.nan]}),'x');test('numeric_range_missing_zero_distinct',n['query_outside_fit_finite_range_rows']==1 and n['fit']['zero_rows']==1 and n['fit']['missing_rows']==1)
 return checks


def run():
 need(not OUT.exists() and not REPORT.exists(),'Refusing to overwrite research output')
 start=time.monotonic();source_digest=sha(__file__);checks=self_check()
 sp=ROOT/'runs/v3/snapshot_manifest.json';need(sha(sp)==SNAPSHOT_SHA,'Released v3 identity changed');snapshot=locked(sp)
 bindings={}
 for item in snapshot['external_dependencies']:bindings[item['original_path']]=(item['sha256'],item['bytes'])
 for name,item in snapshot['copied_files'].items():
  pair=(item['original_sha256'],item['original_bytes']);key=item['original_path'];need(key not in bindings or bindings[key]==pair,'Conflicting release binding');bindings[key]=pair
 files=[CACHE,KEEP,FIT,QUERY,GEO,SCHEMA,'src/house_features.py','src/round3_evaluate.py','reports/round11_covariate_shift_feasibility.md']
 inputs={};snapshot_bound={}
 for name in files:
  path=ROOT/name;digest=sha(path);inputs[name]=digest;snapshot_bound[name]=name in bindings
  if name in bindings:need(bindings[name]==(digest,path.stat().st_size),'Released input changed: '+name)
 for name in [CACHE,KEEP,FIT,QUERY,GEO,'src/house_features.py','src/round3_evaluate.py']:need(snapshot_bound[name],'Required input lacks release binding: '+name)
 manifest=locked(ROOT/GEO);need(manifest['stage']=='development' and manifest['index_files_sha256'][KEEP]==inputs[KEEP],'Wrong R6 mapping identity');need(manifest['source_sha256'][CACHE]==inputs[CACHE],'R6 train X identity differs')
 schema=read(ROOT/SCHEMA);need(schema['feature_source_sha256']==inputs['src/house_features.py'],'Schema feature source changed')
 keep=np.load(ROOT/KEEP,allow_pickle=False);fi=np.load(ROOT/FIT,allow_pickle=False);qi=np.load(ROOT/QUERY,allow_pickle=False)
 for name,a,limit in [('keep',keep,455784),('fit',fi,len(keep)),('query',qi,len(keep))]:need(a.ndim==1 and np.issubdtype(a.dtype,np.integer) and len(np.unique(a))==len(a) and (a>=0).all() and (a<limit).all(),'Bad '+name+' indices')
 need(len(keep)==455255 and len(fi)==240316 and len(qi)==81234 and not np.intersect1d(fi,qi).size,'Development sample contract differs')
 cache=pd.read_pickle(ROOT/CACHE);need(cache.shape==(455784,87) and list(cache)==schema['columns'] and not {'總價元','y','target'}&set(cache),'Unexpected X cache')
 fit_source,query_source=keep[fi],keep[qi];Xf=cache.iloc[fit_source].reset_index(drop=True).copy();Xq=cache.iloc[query_source].reset_index(drop=True).copy();del cache
 dates=lambda X:pd.to_datetime(X.trade_day_index,unit='D',origin='2000-01-01')
 need(dates(Xf).lt(pd.Timestamp('2022-08-11')).all() and dates(Xq).between('2022-08-11','2023-08-10').all(),'Original baseline indices not development dates')
 before=[frame_sha(Xf),frame_sha(Xq)];original_dtypes=[Xf.dtypes.copy(),Xq.dtypes.copy()];Zf,Zq=make_z(Xf),make_z(Xq)
 print(json.dumps({'event':'identity_and_alignment_verified','fit_rows':len(Xf),'query_rows':len(Xq),'categorical_Z':CAT,'numeric_Z':NUM}),flush=True)
 summaries={};rows=[]
 for col in CAT:summaries[col],records=categorical_summary(Zf,Zq,col);rows+=records
 numeric={col:numeric_summary(Zf,Zq,col) for col in NUM}
 jf=Zf[['city_original_district','building_type','main_use','transaction_type']].agg('|'.join,axis=1);jq=Zq[['city_original_district','building_type','main_use','transaction_type']].agg('|'.join,axis=1)
 joint_fit_counts=jf.value_counts();joint_support=jq.map(joint_fit_counts).fillna(0).astype('int64')
 joint={'columns':['city_original_district','building_type','main_use','transaction_type'],'query_exact_categorical_combination_absent_in_fit_rows':int(joint_support.eq(0).sum()),'query_exact_categorical_combination_absent_fraction':float(joint_support.eq(0).mean()),'query_support_count_quantiles':{str(q):float(joint_support.quantile(q)) for q in Q},'interpretation':'Observed empty categorical cells, not proof of population support probability zero; no numeric joint-distance model fitted.'}
 hf,kf,ff=group_keys(Xf,fit_source);hq,kq,fq=group_keys(Xq,query_source);fc=Counter(hf);qc=Counter(hq);fold_map={};kind_map={}
 for hashes,kinds,folds in [(hf,kf,ff),(hq,kq,fq)]:
  for h,k,f in zip(hashes,kinds,folds):need(h not in fold_map or fold_map[h]==int(f),'Same group crosses folds');fold_map[h]=int(f);kind_map[h]=k
 fold_rows=[]
 for fold in range(5):
  fr=int((ff==fold).sum());qr=int((fq==fold).sum());fold_rows.append({'fold':fold,'heldout_fit_rows':fr,'heldout_query_rows':qr,'domain_training_fit_rows':len(Xf)-fr,'domain_training_query_rows':len(Xq)-qr,'heldout_groups':sum(f==fold for f in fold_map.values()),'both_training_domains_present':len(Xf)-fr>0 and len(Xq)-qr>0,'both_heldout_domains_present':fr>0 and qr>0})
 grouping={'rule':FOLD_RULE,'named_project_semantics':'exact existing norm(address_city)|norm(address_district)|norm(project_name); district here differs from original district used by Z','missing_fallback':'compact JSON missing_road city/address_district/address_road/address_section; else missing_address city/address_district/address_raw; else one global unresolved_missing_project group','fit_kind_rows':dict(Counter(kf)),'query_kind_rows':dict(Counter(kq)),'fit_groups':len(fc),'query_groups':len(qc),'shared_fit_query_groups':len(set(fc)&set(qc)),'query_rows_in_groups_not_seen_in_fit':sum(n for h,n in qc.items() if h not in fc),'largest_fit_group_rows':max(fc.values()),'largest_query_group_rows':max(qc.values()),'folds':fold_rows,'fivefold_assignment_usable':all(r['both_training_domains_present'] and r['both_heldout_domains_present'] for r in fold_rows),'same_group_crosses_folds':False,'grouping_keys_or_names_exported':False}
 need([frame_sha(Xf),frame_sha(Xq)]==before and Xf.dtypes.equals(original_dtypes[0]) and Xq.dtypes.equals(original_dtypes[1]),'Original 87 X altered')
 OUT.mkdir(parents=True)
 pd.DataFrame(rows).to_csv(OUT/'categorical_support.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'group_sha256':h,'group_kind':kind_map[h],'fold':fold_map[h],'fit_rows':fc.get(h,0),'query_rows':qc.get(h,0)} for h in sorted(fold_map)]).to_csv(OUT/'group_fold_support.csv',index=False,encoding='utf-8-sig')
 for name,a in [('fit_source_cache_indices',fit_source),('query_source_cache_indices',query_source),('fit_fold',ff),('query_fold',fq)]:np.save(OUT/(name+'.npy'),a,allow_pickle=False)
 dump(OUT/'numeric_summary.json',numeric);dump(OUT/'self_check.json',{'passed':True,'checks':checks,'model_training_or_scoring':False})
 need(all(sha(ROOT/n)==d for n,d in inputs.items()) and sha(__file__)==source_digest and sha(sp)==SNAPSHOT_SHA,'Input/source changed during inventory')
 summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'X_inventory_complete_no_model_decision','source_sha256':source_digest,'input_sha256':inputs,'input_bound_in_v3_snapshot':snapshot_bound,'v3_snapshot_sha256':SNAPSHOT_SHA,'source_manifest_receipt_sha256':sha((ROOT/GEO).with_suffix('.sha256')),'snapshot_receipt_sha256':sha(sp.with_suffix('.sha256')),'analysis_stage':'development_only','fit_rows':len(Xf),'query_rows':len(Xq),'original_feature_columns':list(Xf),'original_feature_count':87,'original_87_values_dtypes_indices_unchanged':True,'categorical_Z':CAT,'numeric_Z':NUM,'Z_columns':list(Zf),'Z_count':len(Zf.columns),'Z_dtypes':{c:str(Zf[c].dtype) for c in Zf},'categorical_support':summaries,'numeric_summary_file':'numeric_summary.json','joint_categorical_support':joint,'grouping':grouping,'alignment':{'loader':'read train_features.pkl; X_fit = cache.iloc[train_source_cache_indices[baseline_fit_indices]].reset_index(drop=True); X_query analogous with baseline_check_indices','source_rows':455784,'canonical_train_rows':455255,'fit_source_cache_indices_content_sha256':array_sha(fit_source),'query_source_cache_indices_content_sha256':array_sha(query_source),'fit_X87_frame_sha256':before[0],'query_X87_frame_sha256':before[1],'fit_Z_frame_sha256':frame_sha(Zf),'query_Z_frame_sha256':frame_sha(Zq),'no_target_dedup_recomputed':True,'full_source_cache_indices_not_used':True},'data_access_limits':{'only_feature_pickle_loaded':CACHE,'pickle_deserializes_all_cached_X_rows':True,'only_selected_development_fit_query_rows_analyzed':True,'target_cache_or_y_read':False,'predictions_or_scores_or_stability_read':False,'confirmation_auxiliary_final_distribution_analyzed':False,'raw_remarks_read':False,'project_and_address_used_only_for_grouping':True},'domain_classifier_trained':False,'house_price_model_trained':False,'importance_weights_or_ESS_computed':False,'hyperparameters_searched_or_selected':False,'formal_candidate_frozen':False,'quantile_grid_descriptive_only':Q,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'elapsed_seconds':time.monotonic()-start,'artifact_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 dump(OUT/'inventory.json',summary);(OUT/'inventory.sha256').write_text(sha(OUT/'inventory.json')+'\n',encoding='utf-8')
 lines=['# R11 development 純 X 支援域盤點','','本次只盤點固定 development 的 240,316 fit／81,234 query，沒有價格、預測、穩定度結果、分類器、重要性權重或模型分數。原 87 欄未改；這不是正式候選或模型接受決定。','','## 精確欄位與缺值支援','','類別：`'+'`、`'.join(CAT)+'`。其中 city_original_district 由 address_city 與原 district 組合；任一缺失則記 missing。數值：`'+'`、`'.join(NUM)+'`。廳數是 living_rooms，不是 halls。', '', '| Z 類別 | fit非缺類別数 | query非缺類別数 | query缺失列 | query非缺但fit未見列 | 占全query |','|---|---:|---:|---:|---:|---:|']
 for c,a in summaries.items():lines.append(f"| {c} | {a['fit_distinct_nonmissing']:,} | {a['query_distinct_nonmissing']:,} | {a['query_missing_rows']:,} | {a['query_unseen_nonmissing_rows']:,} | {a['query_unseen_fraction_all_rows']:.4%} |")
 lines+=['','數值分位、零／負／缺失與超出 fit 有限範圍的 query 列保存於 numeric_summary.json；沒有對價格做任何分位。分類頻率差異的 total variation 只是描述統計，不是分類器或 MAE。', '',f"固定 city_original_district×building_type×main_use×transaction_type 四欄組合在 fit 未出現的 query 有 {joint['query_exact_categorical_combination_absent_in_fit_rows']:,} 列（{joint['query_exact_categorical_combination_absent_fraction']:.4%}）。有限樣本中的空格不等於母體機率真的為零；沒有用全數值距離判定支援域。",'','## 五折建案可用性','',f"fit {grouping['fit_groups']:,} 群、query {grouping['query_groups']:,} 群，共享 {grouping['shared_fit_query_groups']:,} 群。最大fit群 {grouping['largest_fit_group_rows']:,} 列，最大query群 {grouping['largest_query_group_rows']:,} 列。五折訓練／持出兩domain皆非空：{grouping['fivefold_assignment_usable']}。",'', '| 折 | 持出fit | 持出query | domain訓練fit | domain訓練query | 持出群數 |','|---|---:|---:|---:|---:|---:|']
 for r in fold_rows:lines.append('| '+ ' | '.join(str(r[k]) for k in ['fold','heldout_fit_rows','heldout_query_rows','domain_training_fit_rows','domain_training_query_rows','heldout_groups'])+' |')
 lines+=['','具名群沿用 address_city|address_district|project_name（與Z的原district用途不同）；缺名依道路及段→完整地址→單一 global unresolved 隔離群；全部為 compact JSON key，SHA256(R11|20260925|+key)前8 bytes little-endian mod5。這次首次盤點前已對齊正式規約，未先跑舊 address-first／來源列回退再改數字。完整地址與案名只用於分組，不進 Z，也不輸出明文群鍵。道路群是保守代理，可能合併不同真實建案；unresolved 也不代表一個真實建案。只用一個固定hash分折，沒有搜尋平衡較佳的分法。','','## 原始列到固定窗口的對齊','','讀取既有194MB train_features.pkl時會反序列化整份無價格X；只有經已凍結索引選出的development兩側參與分布統計。沒有聲稱其餘X連記憶體都未進入。未讀 train.pkl、valid/test特徵或其他窗口資料分布。','','1. R6 train_source_cache_indices 從去表頭的455,784列快取選出455,255列canonical train。','2. 原baseline fit_indices／check_indices再作用canonical train；本研究以兩次索引合成直接取cache，保留來源row並重置frame index。','3. fit／query來源索引、五折陣列、各群支持與來源SHA已保存；未重算任何需要y的去重，也未誤用full_source_cache_indices。','4. 指定核心輸入均與v3 snapshot綁定SHA／bytes及manifest收據一致。schema與方法文件的實際SHA另記，是否直接被v3綁定逐檔揭露，不冒称全部都有旧release綁定。','','## 解讀範圍','','此盤點可區分單欄/粗分類組合的未見支持與分布比例變化，但不能證明完整結構空間重疊、條件價格不變或可得到低變異的重要性權重。五折可分只表示兩domain可用，不是domain模型已通過校準或ESS要求。正式權重上下限、正則化及停止規則尚未選定。','','[聚合JSON](../outputs/round11/research/inventory.json)、[類別支持](../outputs/round11/research/categorical_support.csv)、[數值摘要](../outputs/round11/research/numeric_summary.json)、[匿名群折支持](../outputs/round11/research/group_fold_support.csv)。執行 `.venv/Scripts/python.exe -B src/round11_X_inventory.py`，拒覆寫原輸出；`--self-check-only`只跑合成。']
 REPORT.write_text('\n'.join(lines).replace('類別数','類別數').replace('冒称','冒稱').replace('旧release','舊release')+'\n',encoding='utf-8')
 print(json.dumps({'event':'X_inventory_complete','categorical_support':summaries,'joint_support':joint,'grouping':grouping,'inventory_sha256':sha(OUT/'inventory.json'),'report_sha256':sha(REPORT),'source_sha256':source_digest},ensure_ascii=False,indent=2),flush=True)
 return summary


def main():
 p=argparse.ArgumentParser();p.add_argument('--self-check-only',action='store_true');args=p.parse_args()
 if args.self_check_only:print(json.dumps({'passed':True,'checks':self_check()},ensure_ascii=False));return
 run()
if __name__=='__main__':main()
