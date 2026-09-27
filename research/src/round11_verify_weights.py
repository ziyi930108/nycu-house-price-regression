"""Independent read-only R11 development weight audit; no price data or model refit."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import hashlib,json,math,time,platform
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
import round11_domain_weights as recipe
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'outputs/round11/weights/development'
PLAN=ROOT/'outputs/round11/domain_plan.json'
MANIFEST=FOLDER/'manifest.json'
PLAN_SHA='14266c423bd547bcb514dc624dbd19bd1f4a7566c1e265d3b0a22894108062e6'
MANIFEST_SHA='d8f23d3943fef55015fc337e1e6f8026e2ab93cde60a24ff48d762cac5ceed1a'
CATS=['address_city','city_original_district','building_type','main_use','transaction_type','parking_type']
NUMS=['building_area','land_area','parking_area','rooms','living_rooms','bathrooms','floor_min','floor_max','total_floors','parking_count']
DOC=ROOT/'reports/round11_weight_result_review.md'
PROOF=ROOT/'outputs/round11/weight_result_review.json'
MISSING='__MISSING__'

def need(value,message):
    if not value:raise AssertionError(message)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def text_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def content(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def values_sha(a,dtype='<f8'):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def frame_sha(X):return hashlib.sha256(pd.util.hash_pandas_object(X,index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()
def relative(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def locked(path,expected=None):
    actual=digest(path);need(expected is None or actual==expected,'Unexpected primary SHA: '+str(path))
    need(Path(path).with_suffix('.sha256').read_text(encoding='ascii').strip()==actual,'Receipt mismatch: '+str(path))
    return text_json(path)
def verify(mapping):
    for name,expected in mapping.items():
        p=(ROOT/name).resolve();need(p.is_relative_to(ROOT.resolve()) and p.is_file(),'Unsafe or missing dependency: '+name)
        need(digest(p)==expected,'Dependency changed: '+name)
def close(a,b,label):
    if isinstance(b,dict):
        need(set(a)==set(b),label+' keys')
        for k in b:close(a[k],b[k],label+'.'+str(k))
    elif isinstance(b,list):
        need(len(a)==len(b),label+' length')
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,label+'.'+str(i))
    elif isinstance(b,float):need(math.isclose(float(a),b,rel_tol=1e-10,abs_tol=1e-10),label+' value')
    else:need(a==b,label+' value')
def normal(series):
    a=series.fillna(MISSING).astype(str).str.strip().reset_index(drop=True)
    return a.mask(a.isin(['','nan','None','<NA>','NaT']),MISSING)
def category(X,c):
    if c!='city_original_district':return normal(X[c])
    return pd.Series([json.dumps([a,b],ensure_ascii=False,separators=(',',':')) if a!=MISSING and b!=MISSING else MISSING for a,b in zip(normal(X.address_city),normal(X.district))])
def numeric(X):
    out=[]
    for c in NUMS:
        a=np.asarray(X[c],dtype=np.float64).copy();a[~np.isfinite(a)]=np.nan
        if c in ('floor_min','floor_max'):a=np.sign(a)*np.log1p(np.abs(a))
        else:a[a<0]=np.nan;a=np.log1p(a)
        out.append(a)
    return np.column_stack(out)
def groups(X):
    fields={k:normal(X[k]).to_numpy() for k in ['address_city','address_district','project_name','address_road','address_section','address_raw']}
    output=[];counts={k:0 for k in ['named','missing_road','missing_address','unresolved']}
    for i in range(len(X)):
        city,dist,project=[fields[k][i] for k in ['address_city','address_district','project_name']]
        if project!=MISSING:kind='named';key=['project',city,dist,project]
        elif fields['address_road'][i]!=MISSING:kind='missing_road';key=[kind,city,dist,fields['address_road'][i],fields['address_section'][i]]
        elif fields['address_raw'][i]!=MISSING:kind='missing_address';key=[kind,city,dist,fields['address_raw'][i]]
        else:kind='unresolved';key=['unresolved_missing_project']
        output.append(json.dumps(key,ensure_ascii=False,separators=(',',':')));counts[kind]+=1
    return np.asarray(output,dtype=object),counts
def folds(g):
    return np.asarray([int.from_bytes(hashlib.sha256(('R11|20260925|'+x).encode()).digest()[:8],'little')%5 for x in g],dtype='int8')
def verify_state(source,state,config):
    need(state['state_sha256']==content({k:v for k,v in state.items() if k!='state_sha256'}),'Preprocessor state receipt')
    a=numeric(source);med=[];all_missing=[]
    for j in range(len(NUMS)):
        v=a[:,j][np.isfinite(a[:,j])];med.append(float(np.median(v)) if len(v) else 0.);all_missing.append(not len(v))
    filled=np.where(np.isnan(a),np.asarray(med),a);mean=np.mean(filled,axis=0);scale=np.std(filled,axis=0);scale[scale==0]=1
    for key,expected in [('median',med),('mean',mean),('scale',scale)]:need(np.allclose(state[key],expected,rtol=1e-12,atol=1e-12),'Source-only '+key)
    need(state['all_missing']==all_missing,'All-missing flags')
    need(state['source_rows']==len(source) and state['fit_X_sha256']==frame_sha(source),'Preprocessor source identity')
    need(state['config_sha256']==content(config),'Preprocessor config')
    names=[]
    for c in CATS:
        vals=category(source,c);counts=Counter(v for v in vals if v!=MISSING);frequent=sorted(k for k,n in counts.items() if n>=50)
        cfg={'seen':sorted(counts),'frequent':frequent,'tokens':['M','U','R']+['V:'+v for v in frequent]}
        need(state['categories'][c]==cfg,'Source-only category vocabulary '+c)
        names.extend('cat:'+c+'='+v for v in cfg['tokens'])
    names.extend('num:'+c for c in NUMS);names.extend('missing:'+c for c in NUMS)
    need(names==state['feature_names'],'Preprocessor encoded schema')
def encode_independent(X,state):
    blocks=[]
    for c in CATS:
        cfg=state['categories'][c];seen=set(cfg['seen']);frequent=set(cfg['frequent']);tokens={v:i for i,v in enumerate(cfg['tokens'])}
        columns=np.asarray([tokens['M' if v==MISSING else ('V:'+v if v in frequent else ('R' if v in seen else 'U'))] for v in category(X,c)])
        blocks.append(sparse.csr_matrix((np.ones(len(X)),(np.arange(len(X)),columns)),shape=(len(X),len(tokens))))
    a=numeric(X);mask=np.isnan(a);filled=np.where(mask,np.asarray(state['median']),a)
    blocks.extend([sparse.csr_matrix((filled-state['mean'])/state['scale']),sparse.csr_matrix(mask.astype(float))])
    return sparse.hstack(blocks,format='csr')
def support_independent(X,Xq,g):
    def keys(frame,kind):
        cols=[category(frame,'city_original_district')] if kind=='region' else [normal(frame[c]) for c in ['building_type','main_use','transaction_type']]
        return [tuple(v) if MISSING not in v else None for v in zip(*cols)]
    result={}
    for kind in ['region','structure']:
        a,b=keys(X,kind),keys(Xq,kind);counts=Counter(a);distinct=defaultdict(set)
        for key,group in zip(a,g):distinct[key].add(group)
        allowed={key for key,n in counts.items() if key is not None and n>=20 and len(distinct[key])>=3};present=set(counts)-{None}
        n=sum(key in allowed for key in b)
        result[kind]={'query_rows':len(b),'adequate_query_rows':n,'adequate_query_fraction':n/len(b),'unseen_or_missing_query_rows':sum(key not in present for key in b),'supported_fit_keys':len(allowed),'observed_fit_nonmissing_keys':len(present)}
    return result
def quantiles(v):return {str(q):float(np.quantile(v,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1.]}
def ess(v):return float(np.sum(v)**2/np.dot(v,v))
def weights_independent(lr,g):
    lo,hi=np.log([.25,4.]);clipped=np.exp(np.clip(lr,lo,hi));w=clipped/clipped.mean()
    table=pd.DataFrame({'group':g,'w':w});mass=table.groupby('group',sort=True).w.sum().to_numpy();counts=table.groupby('group',sort=True).size().to_numpy(dtype=float)
    minimum,maximum=np.log(np.nextafter(np.float64(0),np.float64(1))),np.log(np.finfo(float).max)
    raw_q={k:float(np.exp(v)) if minimum<=v<=maximum else None for k,v in quantiles(lr).items()}
    summary={'raw_log_ratio_quantiles':quantiles(lr),'raw_ratio_quantiles_when_representable':raw_q,'raw_ratio_underflow_or_overflow_rows':int(((lr<minimum)|(lr>maximum)).sum()),'clipped_ratio_quantiles':quantiles(clipped),'clipped_ratio_mean':float(clipped.mean()),'normalized_weight_quantiles':quantiles(w),'normalized_weight_mean':float(w.mean()),'clipped_lower_rows':int((lr<lo).sum()),'clipped_upper_rows':int((lr>hi).sum()),'clipped_fraction':float(((lr<lo)|(lr>hi)).mean()),'row_ess':ess(w),'row_ess_fraction':ess(w)/len(w),'groups':len(mass),'group_mass_ess':ess(mass),'uniform_group_mass_ess':ess(counts),'group_ess_ratio_to_uniform':ess(mass)/ess(counts),'largest_group_weight_share':float(mass.max()/mass.sum()),'largest_uniform_group_row_share':float(counts.max()/counts.sum())}
    return w,clipped,summary

def main():
    start=time.monotonic();self_sha=digest(__file__)
    for f in [DOC,PROOF,PROOF.with_suffix('.sha256')]:need(not f.exists(),'Refusing overwrite '+str(f))
    plan=locked(PLAN,PLAN_SHA);m=locked(MANIFEST,MANIFEST_SHA)
    need(plan['round']==m['round']==11 and m['stage']=='development','Wrong round/stage')
    need(m['domain_plan_sha256']==PLAN_SHA and m['config']==plan['config']==recipe.CONFIG,'Plan/config mismatch')
    need(m['config_sha256']==plan['config_sha256']==content(recipe.CONFIG),'Config content identity')
    need(m['source_sha256']==plan['source_sha256'] and m['input_sha256']==plan['input_sha256'],'Provenance mappings differ')
    current={'python':platform.python_version(),**{n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']}}
    need(m['runtime_versions']==plan['runtime_versions']==current,'Runtime differs')
    verify(plan['source_sha256']);verify(plan['input_sha256']);verify(m['artifact_sha256'])
    expected_artifacts={relative(FOLDER/(name+'.npy')) for name in ['fit_folds','query_folds','fit_oof_probability','query_oof_probability','fit_raw_log_ratio','query_raw_log_ratio','fit_clipped_ratio','fit_weights']}
    expected_artifacts|={relative(FOLDER/('fold_'+str(k))/name) for k in range(5) for name in ['classifier.joblib','preprocessor.json']}
    actual={relative(f) for f in FOLDER.rglob('*') if f.is_file() and f.name not in ['manifest.json','manifest.sha256']}
    need(set(m['artifact_sha256'])==expected_artifacts==actual,'Artifact inventory incomplete')
    snapshot=ROOT/'runs/v3/snapshot_manifest.json';need(digest(snapshot)==plan['reference_snapshot_sha256'],'Historical snapshot changed');snap=text_json(snapshot)
    historical={k:v['original_sha256'] for k,v in snap['copied_files'].items()};historical.update({v['original_path']:v['sha256'] for v in snap['external_dependencies']});historical[relative(snapshot)]=digest(snapshot)
    historical_bindings={k:v for k,v in plan['input_sha256'].items() if not k.startswith('reports/round11_') and k!='outputs/round11/research/inventory.json'}
    need(all(historical.get(k)==v for k,v in historical_bindings.items()),'Historical input mismatch')
    schema=text_json(ROOT/'reports/feature_schema.json');all_X=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
    need(list(all_X)==schema['columns'] and len(all_X.columns)==87 and len(all_X)==schema['splits']['train']['rows'],'Raw feature schema')
    need(not {'總價元','y','target','編號'}.intersection(all_X.columns),'Unexpected target or ID in feature cache')
    rowmap=np.load(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy',allow_pickle=False)
    need(rowmap.ndim==1 and np.issubdtype(rowmap.dtype,np.integer) and len(np.unique(rowmap))==455255 and len(rowmap)==455255 and rowmap.min()>=0 and rowmap.max()<len(all_X),'Canonical row map')
    train=all_X.iloc[rowmap].reset_index(drop=True);del all_X
    base=ROOT/'outputs/round6/models/development/geography_catboost'
    fi=np.load(base/'fit_indices.npy',allow_pickle=False);qi=np.load(base/'query_indices.npy',allow_pickle=False)
    days=train.trade_day_index.to_numpy();origin=(pd.Timestamp('2022-08-11')-pd.Timestamp('2000-01-01')).days;end=(pd.Timestamp('2023-08-10')-pd.Timestamp('2000-01-01')).days
    need(np.array_equal(fi,np.flatnonzero(days<origin)) and np.array_equal(qi,np.flatnonzero((days>=origin)&(days<=end))),'Complete fixed date split')
    need(not np.intersect1d(fi,qi).size,'Fit/query index overlap')
    old=text_json(ROOT/'outputs/round6/candidate_plan.json')['split_contract']['development']
    X=train.iloc[fi].reset_index(drop=True);Xq=train.iloc[qi].reset_index(drop=True);del train
    need((len(X),len(Xq))==(240316,81234)==(m['fit_rows'],m['query_rows']),'Split rows')
    for prefix,frame,ix in [('fit',X,fi),('query',Xq,qi)]:
        need(frame_sha(frame)==m[prefix+'_X_sha256'],'Actual '+prefix+' frame identity')
        need(values_sha(ix,'<i8')==m[prefix+'_indices_content_sha256']==old[prefix+'_indices_content_sha256'],'Actual '+prefix+' index identity')
        need(len(ix)==old[prefix+'_rows'],'Historical split count')
    need(list(X)==m['original_feature_columns'] and {c:str(X[c].dtype) for c in X}==m['fit_feature_dtypes'],'Manifest feature schema')
    gs,cs=groups(X);gq,cq=groups(Xq);fs,fq=folds(gs),folds(gq)
    need(cs==m['fit_group_fallback_counts'] and cq==m['query_group_fallback_counts'],'Group fallback counts')
    arrays={name:np.load(FOLDER/(name+'.npy'),allow_pickle=False) for name in ['fit_folds','query_folds','fit_oof_probability','query_oof_probability','fit_raw_log_ratio','query_raw_log_ratio','fit_clipped_ratio','fit_weights']}
    need(np.array_equal(fs,arrays['fit_folds']) and np.array_equal(fq,arrays['query_folds']),'Group fold mapping')
    source_log=np.full(len(X),np.nan);query_log=np.full(len(Xq),np.nan);fold_proofs=[]
    for k in range(5):
        report=m['fold_reports'][k];need(report['fold']==k,'Fold report order')
        si,qi_train=np.flatnonzero(fs!=k),np.flatnonzero(fq!=k);sv,qv=np.flatnonzero(fs==k),np.flatnonzero(fq==k)
        held=set(gs[sv])|set(gq[qv]);fit=set(gs[si])|set(gq[qi_train]);need(not held&fit,'Shared group leaked into fold train')
        for name,ix in [('source_train',si),('query_train',qi_train),('source_heldout',sv),('query_heldout',qv)]:
            need(len(ix)==report[name+'_rows'] and values_sha(ix,'<i8')==report[name+'_indices_sha256'],'Fold index evidence '+name)
        state=text_json(FOLDER/('fold_'+str(k))/'preprocessor.json');source=X.iloc[si].reset_index(drop=True);verify_state(source,state,m['config'])
        need(report['preprocessor_source_X_sha256']==frame_sha(source) and report['preprocessor_source_train_only'] is True,'Source-only preprocessor report')
        model=joblib.load(FOLDER/('fold_'+str(k))/'classifier.joblib')
        need(all(model.get_params()[key]==value for key,value in m['config']['classifier'].items()),'Classifier parameters changed')
        need(np.array_equal(model.classes_,[0,1]) and np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Classifier state')
        need(model.n_features_in_==report['feature_count']==len(state['feature_names']) and int(model.n_iter_[0])==report['iterations']<1000 and report['converged'] is True and report['warnings']==[],'Classifier convergence/schema')
        with threadpool_limits(limits=4):
            ls=np.asarray(model.decision_function(encode_independent(X.iloc[sv],state)),dtype=float)
            lq=np.asarray(model.decision_function(encode_independent(Xq.iloc[qv],state)),dtype=float)
        prior=float(np.log(len(si)/len(qi_train)));need(math.isclose(prior,report['log_prior_correction'],rel_tol=0,abs_tol=1e-14),'Prior correction')
        source_log[sv]=ls+prior;query_log[qv]=lq+prior
        errs={}
        for name,actual,expected in [('source_probability',expit(ls),arrays['fit_oof_probability'][sv]),('query_probability',expit(lq),arrays['query_oof_probability'][qv]),('source_log_ratio',ls+prior,arrays['fit_raw_log_ratio'][sv]),('query_log_ratio',lq+prior,arrays['query_raw_log_ratio'][qv])]:
            need(np.allclose(actual,expected,rtol=1e-12,atol=1e-12),'Full classifier replay '+name);errs[name]=float(np.max(np.abs(actual-expected)))
        auc=float(roc_auc_score(np.r_[np.zeros(len(sv)),np.ones(len(qv))],np.r_[ls,lq]));close(auc,report['heldout_auc'],'Fold AUC')
        fold_proofs.append({'fold':k,'source_train_rows':len(si),'query_train_rows':len(qi_train),'source_heldout_rows':len(sv),'query_heldout_rows':len(qv),'heldout_groups_in_training':0,'source_only_preprocessor_recomputed':True,'all_heldout_rows_replayed':True,'maximum_absolute_errors':errs,'iterations':int(model.n_iter_[0]),'prior_correction':prior,'auc':auc})
        print(json.dumps({'event':'independent_fold_replay_verified','fold':k,'source_rows':len(sv),'query_rows':len(qv)}),flush=True)
    need(np.isfinite(source_log).all() and np.isfinite(query_log).all(),'OOF coverage incomplete')
    coverage=support_independent(X,Xq,gs);close(coverage,m['coverage'],'Support')
    weights,clipped,summary=weights_independent(arrays['fit_raw_log_ratio'],gs)
    need(arrays['fit_weights'].dtype==np.float64 and np.array_equal(weights,arrays['fit_weights']) and np.array_equal(clipped,arrays['fit_clipped_ratio']),'Exact saved weights/clip replay')
    need(values_sha(weights)==m['fit_weight_values_sha256'],'Ordered weight values SHA');close(summary,m['weight_summary'],'Weight statistics')
    f=m['config']['feasibility'];mass_limit=max(f['max_group_mass_floor'],f['max_group_mass_multiple_of_uniform']*summary['largest_uniform_group_row_share'])
    checks={'converged_all_folds':all(x['iterations']<1000 for x in fold_proofs),'query_region_support':coverage['region']['adequate_query_fraction']>=f['min_region_query_support'],'query_structure_support':coverage['structure']['adequate_query_fraction']>=f['min_structure_query_support'],'clipped_fraction_within_limit':summary['clipped_fraction']<=f['max_clipped_fit_fraction'],'row_ess_at_least_half':summary['row_ess_fraction']>=f['min_row_ess_fraction'],'group_ess_not_halved':summary['group_ess_ratio_to_uniform']>=f['min_group_ess_ratio_to_uniform'],'largest_group_mass_within_limit':summary['largest_group_weight_share']<=mass_limit}
    need(checks==m['feasibility_checks'] and all(checks.values())==m['feasibility_passed'],'Seven fixed gates differ')
    D=np.r_[np.zeros(len(X)),np.ones(len(Xq))];lr=np.r_[source_log,query_log]
    auc=float(roc_auc_score(D,lr));loss=float(log_loss(D,expit(lr+np.log(len(Xq)/len(X)))))
    close(auc,m['diagnostic_domain_oof_auc_prior_corrected'],'Pooled prior-corrected AUC');close(loss,m['diagnostic_domain_oof_log_loss_common_prior'],'Common prior logloss')
    city_s=normal(X.address_city);city_q=normal(Xq.address_city);cities={}
    for city in sorted(set(city_s)|set(city_q)):
        a=city_s.eq(city).to_numpy();b=city_q.eq(city).to_numpy();cities[city]={'source_rows':int(a.sum()),'source_unweighted_share':float(a.mean()),'source_weight_share':float(weights[a].sum()/weights.sum()),'query_rows':int(b.sum()),'query_share':float(b.mean())}
    close(cities,m['city_mass_summary'],'City mass summary')
    need(m['classifier_models_fitted']==5 and m['property_price_models_fitted']==0 and m['query_X_used_for_weights'] is True and m['query_y_or_price_cache_used'] is False and m['row_maps_reused_without_deserializing_price_cache'] is True and m['historically_target_containing_dedup_used_for_alignment'] is True,'Data access declarations')
    need(digest(__file__)==self_sha and digest(PLAN)==PLAN_SHA and digest(MANIFEST)==MANIFEST_SHA,'Audit/source identity changed')
    verify(plan['source_sha256']);verify(plan['input_sha256']);verify(m['artifact_sha256'])
    proof={'round':11,'stage':'development','verified':True,'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'reviewer_source_sha256':self_sha,'domain_plan_sha256':PLAN_SHA,'manifest_sha256':MANIFEST_SHA,'runtime_versions':current,'source_sha256':plan['source_sha256'],'verified_input_sha256':plan['input_sha256'],'verified_artifact_sha256':m['artifact_sha256'],'historical_bindings_verified':len(historical_bindings),'artifact_inventory_exact':True,'fit_rows':len(X),'query_rows':len(Xq),'feature_columns':len(X.columns),'fit_X_sha256':frame_sha(X),'query_X_sha256':frame_sha(Xq),'fold_proofs':fold_proofs,'coverage':coverage,'weight_summary':summary,'feasibility_checks':checks,'feasibility_passed':all(checks.values()),'largest_group_mass_limit':mass_limit,'diagnostic_domain_auc_prior_corrected':auc,'diagnostic_domain_log_loss_common_prior':loss,'no_method_changed_after_observing_results':True,'scope':{'feature_cache_deserialized':['data_cache/train_features.pkl'],'saved_domain_classifiers_loaded':5,'source_oof_rows_replayed':len(X),'query_oof_rows_replayed':len(Xq),'source_preprocessor_recomputed_all_folds':True,'query_X_used_by_original_domain_training':True,'raw_price_cache_deserialized':False,'property_targets_or_property_predictions_read':False,'property_MAE_computed':False,'classifiers_refitted':False,'price_models_fitted':False,'confirmation_auxiliary_final_results_read':False,'other_input_files_bytes_hashed_only_except_schema_plans_snapshot_indices_docs_not_parsed':True},'limits':['Pure-X gates are engineering limits, not property-score or statistical-significance proof.','Missing-project groups are conservative proxies, not verified physical projects.','Marginal region/structure support does not imply full joint support.','Historical windows have been reused; this is not an untouched validation set.']}
    rows='\n'.join('| '+name+' | '+('通過' if passed else '未通過')+' |' for name,passed in checks.items())
    report=f'''# R11 development 純 X 權重結果獨立核對

核對通過：保存的五個 domain classifier、逐折前處理、全部 OOF 機率／log-ratio、權重和七個事前安全閘相互一致。這僅確認固定純 X 方法按契約產生結果，不代表房價 MAE 或 Private 分數改善。本審查沒有改方法、門檻、fold、欄位或 seed，也沒有重 fit 任何 classifier。

## 來源與實際讀取範圍

- domain plan SHA256：`{PLAN_SHA}`。
- development manifest SHA256：`{MANIFEST_SHA}`；兩份原有 receipt 均核對。
- [獨立驗證器](../src/round11_verify_weights.py) SHA256：`{self_sha}`。
- 核對 {len(plan['input_sha256'])} 個 input bindings，其中 {len(historical_bindings)} 個既有資料／索引／schema／舊計畫來源與固定 v3 snapshot 一致；另核 {len(m['artifact_sha256'])} 個 artifact，清單與目錄完全對上，前後 SHA 未改。
- 實際反序列化的特徵快取只有 train_features.pkl（原 87 欄），沿用歷史 529 筆去重映射與 R6 同窗 fi/qi，核對日期和列序後得到 fit {len(X):,}／query {len(Xq):,}。沒有重做含 target 去重，沒有讀 raw price cache、房價 y、房價預測或 MAE。
- 讀取保存的五個 domain joblib、五份 preprocessor、八個 OOF／fold／weight NPY。joblib 僅用 decision_function 重播；前處理統計以 source-train fold X 獨立重算，沒有重新訓練模型。其他年度與 test feature cache 只驗 bytes SHA，沒有反序列化；沒有讀後續年度的 domain 結果。

## 隔離、前處理與全量重播

每折 source 和 query 使用共同 group key／固定 SHA fold；held-out groups 與兩側 training groups 的交集皆為 0。具名建案和缺名 road→address→global unresolved 的代理政策與保存 folds 一致。

五折 source-only transformed median、補值後 mean/std、全缺預設、各類別 seen／頻數至少 50 的詞彙、M/U/R/V: 欄位及 feature names 都核對；十個缺值指標未縮放。重新編碼所有 held-out source {len(X):,} 列與 query {len(Xq):,} 列，saved classifier 全量 replay 與 OOF 機率及先驗修正 log-ratio 在 1e-12 容差內一致。五折迭代數為 {', '.join(str(v['iterations']) for v in fold_proofs)}，各折參數等於固定 CONFIG。

逐折先驗使用該折訓練 nsource/nquery，非 held-out 比例。獨立重算 pooled prior-corrected AUC 為 {auc:.9f}，共同全期先驗 logloss 為 {loss:.9f}，與 manifest 一致；它們只是 domain 診斷，未用來挑選設定。

## 權重與固定安全閘

OOF log-ratio→log(0.25)..log(4) 裁切→exp→完整 source 平均 1 的權重逐值一致，未二次裁切。正規化後範圍 {weights.min():.9f}～{weights.max():.9f}，平均 {weights.mean():.12f}；這個範圍不必仍是 0.25..4。

- region 足夠支持 {coverage['region']['adequate_query_rows']:,}/{len(Xq):,}（{coverage['region']['adequate_query_fraction']:.4%}）；structure {coverage['structure']['adequate_query_rows']:,}/{len(Xq):,}（{coverage['structure']['adequate_query_fraction']:.4%}）。支持以 source 至少 20 列且至少 3 個 group 計算，缺失不算支持。
- source 裁切 {summary['clipped_lower_rows']+summary['clipped_upper_rows']:,} 列（{summary['clipped_fraction']:.4%}）；row ESS {summary['row_ess']:,.3f}（{summary['row_ess_fraction']:.4%}）。
- group-mass ESS {summary['group_mass_ess']:,.3f}，相對原均等逐列權重 ESS 的比例 {summary['group_ess_ratio_to_uniform']:.4%}；最大群權重質量 {summary['largest_group_weight_share']:.4%}，本期固定規則導出的上限 {mass_limit:.4%}。group 不全是可靠物理建案，ESS 也不是獨立樣本數證明。

| 固定安全閘 | 獨立重算 |
|---|---|
{rows}

七個 boolean 與 manifest 完全一致；city mass、完整 quantiles、raw ratio overflow 計數亦核對。兩個邊際 90% 支持閘不表示聯合支持或連續特徵支持已充分。歷史年度已反覆使用，不能稱完全未看資料。

完整可查核結果與各折重播誤差見 [weight_result_review.json](../outputs/round11/weight_result_review.json) 及同名 .sha256。核對實際耗時（含 hashes、讀 X、重播與獨立統計）{proof['elapsed_seconds']:.3f} 秒。此證據只涵蓋 development 純 X 權重；後續價格候選須另行固定計畫並經既有三期門檻。
'''
    DOC.write_text(report,encoding='utf-8');proof['review_document_sha256']=digest(DOC)
    with PROOF.open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    with PROOF.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(digest(PROOF)+'\n')
    print(json.dumps({'verified':True,'review_document_sha256':digest(DOC),'proof_sha256':digest(PROOF),'script_sha256':self_sha,'elapsed_seconds':proof['elapsed_seconds'],'feasibility_checks':checks},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
