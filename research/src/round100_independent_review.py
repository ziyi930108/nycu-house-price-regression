"""Bounded R100 saved-result audit; no y/NPZ/model load, prior build, MAE or resampling.

Author also wrote the pure-X policy and reviewed the helper. This is a
cross-author result check, not a claim of full experimental independence.
"""
from __future__ import annotations
import argparse, hashlib, json, time, traceback
import numpy as np
import pandas as pd
import round100_contract as c
import round52_price_prior as original

MAIN=c.ROOT/'outputs/round100'
OUT=c.ROOT/'outputs/round100_independent_review'
REPORT=c.ROOT/'reports/round100_independent_review.md'

def close(a,b,msg,tol=1e-6):
    c.need(np.isfinite(a) and np.isfinite(b) and abs(a-b)<=tol,msg)

def content_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Evidence:
    def __init__(self):self.hashes={};self.arrays=[];self.documents=[]
    def bind(self,p,h):
        name=c.rel(p);c.need(name not in self.hashes or self.hashes[name]==h,'Conflicting authority '+name)
        if name not in self.hashes:c.need(c.sha(name)==h,'Changed recorded file '+name);self.hashes[name]=h
        return c.safe(name)
    def mapping(self,m):
        for p,h in m.items():self.bind(p,h)
    def document(self,r):
        p=self.bind(r['path'],r['sha256']);c.need(p.suffix=='.json','Only JSON documents')
        self.documents.append(c.rel(p));return c.read(p)
    def current(self,p):return self.document(c.record(p))
    def array(self,r):
        p=self.bind(r['path'],r['sha256'])
        c.need(p.suffix=='.npy' and not any(x in p.stem.lower() for x in ('fit_y','query_y','target','labels')),'Forbidden label/NPZ array')
        a=np.load(p,allow_pickle=False);c.need(c.identity(a)=={k:r[k] for k in ('dtype','shape','values_sha256')},'Array identity')
        self.arrays.append(c.rel(p));return a
    def old_array(self,p,h):
        p=self.bind(p,h);c.need(p.suffix=='.npy' and not any(x in p.stem.lower() for x in ('fit_y','query_y','target','labels')),'Forbidden old array')
        self.arrays.append(c.rel(p));return np.load(p,allow_pickle=False)

def identity(a,r,msg):c.need(c.identity(a)==r,msg)

def prefix_audit(e,st,ax,X,fi):
    arrays={k:e.array(r) for k,r in ax['arrays'].items()}
    c.exact(arrays['fit_indices'],fi,'Pure-X canonical identity')
    days=X.trade_day_index.to_numpy(dtype=np.float64)
    invalid=X.trade_date_invalid.to_numpy(dtype=np.float64);outside=X.trade_date_out_of_range.to_numpy(dtype=np.float64)
    c.need(np.isfinite(invalid).all() and np.isfinite(outside).all(),'Finite date flags')
    valid=np.isfinite(days)&(invalid==0)&(outside==0);c.exact(valid,arrays['valid_date_mask'],'Valid-date mask')
    groups,_=original.grouping.cluster_keys(X,fi)
    folds=np.array([int.from_bytes(hashlib.sha256(('42|'+k).encode()).digest()[:8],'little')%5 for k in groups],dtype=np.int64)
    c.exact(folds,arrays['fold_ids'],'Canonical original folds')
    quarter=np.full(len(X),-1,dtype=np.int64)
    dates=pd.Timestamp('2000-01-01')+pd.to_timedelta(days[valid],unit='D')
    quarter[valid]=np.asarray((dates.to_period('Q-DEC').start_time-pd.Timestamp('2000-01-01')).days,dtype=np.int64)
    c.exact(quarter,arrays['quarter_start_days'],'Calendar-quarter boundary')
    mask=np.zeros(len(X),dtype=bool)
    for start in np.unique(quarter[valid]):
        if min(int((valid&(days<start)&(folds!=f)).sum()) for f in range(5))>=30:mask[valid&(quarter==start)]=True
    c.exact(mask,arrays['eligible_mask'],'Whole-quarter five-fold support')
    kept=e.array(st['eligible_positions']);c.exact(kept,np.flatnonzero(mask).astype(np.int64),'Kept row order')
    p=e.array(st['temporal_prior']);ev=e.document(st['temporal_prior_evidence'])
    c.need(p.dtype==np.float64 and p.shape==kept.shape and np.isfinite(p).all() and (p>0).all(),'Full prior finite positive')
    for value,key in ((p,'temporal_prior'),(fi,'canonical_indices'),(kept,'kept_positions'),(folds,'fold_ids'),(mask,'eligible_mask')):identity(value,ev[key],key)
    c.need(ev['status']=='quarter_prefix_priors_completed' and ev['prefix_states_saved'] and ev['saved_state_records_complete'],'Saved prefix states required')
    c.need(not ev['uses_query_labels'] and not ev['uses_query_features'] and not ev['full_query_prior_rebuilt'],'No new query prior')
    keys=original._keys(X);packed=[pd.factorize(k,sort=True) for k in keys]
    expected={(int(q),int(f)) for q,f in zip(quarter[kept],folds[kept])};seen=set();assigned=np.zeros(len(kept),dtype=bool)
    totals={**{'|'.join(k):0 for k in original.LEVELS},'global':0}
    for r in ev['prefixes']:
        start,f=r['quarter_start_day'],r['held_fold'];pair=(start,f)
        c.need(pair in expected and pair not in seen,'Prefix completeness/uniqueness');seen.add(pair)
        ref=valid&(days<start)&(folds!=f);held=mask&(quarter==start)&(folds==f);held_kept=held[kept]
        c.need(ref.sum()==r['reference_rows']>=30 and held.sum()==r['held_rows'] and not (ref&held).any(),'Legal prefix support')
        c.need(not set(groups[ref]).intersection(groups[held]),'Held group reference leakage')
        for a,k in ((fi[ref],'reference_indices'),(fi[held],'held_indices'),(p[held_kept],'held_prior')):identity(a,r[k],k)
        for a,d in zip((days,invalid,outside),r['reference_dates']):identity(a[ref],d,'Reference date identity')
        c.need(r['reference_groups']==len(np.unique(groups[ref])) and r['reference_min_day']==float(days[ref].min()) and r['reference_max_day']==float(days[ref].max())<start,'Reference groups/dates')
        # Target and unit hashes are provenance only; never reconstruct or load their values.
        for key in ('reference_target','reference_units'):
            c.need(r[key]['dtype']=='float64' and r[key]['shape']==[int(ref.sum())] and len(r[key]['values_sha256'])==64,'Target identity schema only')
        state=e.document(r['saved_state']);c.need(content_sha(state)==r['state_content_sha256'],'Saved state content')
        c.need(state['fit_rows']==state['global_raw_rows']==int(ref.sum()) and state['state_role']=='quarter_prefix_for_held_fold','State role/rows')
        c.need(state['date_support']==r['date_support'] and state['date_support']['max_valid_reference_trade_day_index']==float(days[ref].max()),'Reference-only anchor')
        c.need(np.isfinite(state['global_median']) and state['global_median']>0 and state['global_median']==r['global_median'],'Stored global finite')
        selected=np.full(int(held.sum()),4,dtype=np.int8)
        for j,((codes,vocab),level,cols) in enumerate(zip(packed,state['levels'],original.LEVELS)):
            n=np.bincount(codes[ref],minlength=len(vocab));supported=np.flatnonzero(n>=30)
            c.need(level['columns']==cols and level['cell_count']==len(supported) and set(level['cells'])==set(vocab[supported]),'Saved raw-count cell support')
            for code in supported:
                cell=level['cells'][vocab[code]]
                c.need(cell['rows']==int(n[code]) and cell['valid_date_rows']==cell['rows'] and cell['neutral_date_rows']==0 and np.isfinite(cell['median']) and cell['median']>0,'Cell support/finite value')
            take=(selected==4)&(n[codes[held]]>=30);selected[take]=j
        support={**{'|'.join(cols):int((selected==j).sum()) for j,cols in enumerate(original.LEVELS)},'global':int((selected==4).sum())}
        c.need(support==r['held_prior_level_counts'] and r['JSON_reload_exact'],'Stored prior support/reload assertion')
        for k,v in support.items():totals[k]+=v
        c.need(not assigned[held_kept].any(),'Prior rows assigned twice');assigned[held_kept]=True
    c.need(seen==expected and assigned.all() and len(seen)==ev['prefix_state_count'] and totals==ev['held_prior_level_counts'] and sum(totals.values())==len(kept),'Complete prior coverage')
    return p,kept,{'prefix_states':len(seen),'eligible_rows':len(kept),'level_counts':totals,'full_prior_identity_and_support_verified':True,'reference_target_and_unit_values_recomputed':False}

def metric_check(m,n):
    c.need(m['rows']==n,'Metric rows')
    for key in ('strata','forecast_quarters','months','cities','relative_forecast_months'):
        parts=list(m[key].values());c.need(sum(x['rows'] for x in parts)==n,'Metric partition '+key)
        close(sum(x['rows']*x['mae'] for x in parts if x['rows'])/n,m['overall_mae'],'Saved aggregate '+key)
    bands=m['price_bands'];c.need(sum(x['rows'] for x in bands)==n,'Price-band count')
    close(sum(x['rows']*x['mae'] for x in bands if x['rows'])/n,m['overall_mae'],'Price-band arithmetic')
    close(sum(c.WEIGHTS[k]*m['strata'][k]['mae'] for k in c.WEIGHTS if c.WEIGHTS[k]),m['weighted_mae'],'Fixed weights')
    close(m['tails']['absolute_error_sum']/n,m['overall_mae'],'Saved total error arithmetic')

def comparison_check(metrics,comparison,control='v22',candidate='T'):
    a,b=metrics[control],metrics[candidate];adverse=[]
    def delta(x,y):
        return {'control_mae':x,'candidate_mae':y,'delta':y-x,'relative_change_pct':100*(y-x)/x if x else None}
    for k in ('overall_mae','weighted_mae'):c.need(comparison[k]==delta(a[k],b[k]),'Saved point comparison')
    for section in ('strata','forecast_quarters','months','cities','relative_forecast_months'):
        expected={}
        for key,x in a[section].items():
            y=b[section][key];c.need(x['rows']==y['rows'],'Group counts between models')
            if x['rows']:
                item={'rows':x['rows'],**delta(x['mae'],y['mae'])};expected[key]=item
                if (section=='forecast_quarters' or x['rows']>=500) and y['mae']>1.05*x['mae']:adverse.append({'section':section,'group':key,**item})
        c.need(comparison[section]==expected,'Saved group comparison')
    bands=[]
    for x,y in zip(a['price_bands'],b['price_bands']):
        c.need(x['rows']==y['rows'] and x['band']==y['band'],'Matched band counts')
        item={'band':x['band'],'rows':x['rows'],**(delta(x['mae'],y['mae']) if x['rows'] else {})};bands.append(item)
        if x['rows']>=500 and y['mae']>1.05*x['mae']:adverse.append({'section':'price_bands',**item})
    c.need(bands==comparison['price_bands'] and adverse==comparison['quarter_or_large_group_deterioration_above_5pct'],'Adverse group arithmetic')

def saved_CIs(e,stage,item):
    rows=[]
    for control in ('v22','C'):
        paired=item['paired'][control];c.need(paired['seed']==20260925 and paired['replicates']==2000 and paired['stage_index']==c.STAGES.index(stage),'Fixed bootstrap seed/index')
        for j,family in enumerate(('project','month')):
            a=e.array(item['replicate_artifacts'][control][family]);method=paired['methods'][family]
            c.need(a.shape==(2000,4) and a.dtype==np.float64 and method['family_index']==j,'Saved replicate schema')
            for name,l,r in (('overall',0,1),('weighted',2,3)):
                d=a[:,r]-a[:,l];v=d[np.isfinite(d)];bounds=np.quantile(v,[.025,.975],method='linear').tolist() if len(v) else [None,None]
                s=method['summary'][name]
                c.need(bounds==[s['percentile_2_5'],s['percentile_97_5']] and s['valid_replicates']==len(v) and s['invalid_replicates']==2000-len(v),'Saved CI endpoints/count')
                c.need(s['bootstrap_mean_delta']==(float(v.mean()) if len(v) else None) and s['negative_delta_fraction']==(float((v<0).mean()) if len(v) else None),'Saved CI summary')
                close(s['control_mae'],item['metrics'][control][name+'_mae'],'CI control point')
                close(s['candidate_mae'],item['metrics']['T'][name+'_mae'],'CI candidate point')
                close(s['point_delta'],s['candidate_mae']-s['control_mae'],'CI point delta')
                rows.append({'stage':stage,'control':control,'family':family,'metric':name,'point_delta':s['point_delta'],'lower':bounds[0],'upper':bounds[1],'valid':len(v)})
    return rows

def render(result,ev):
    lines=['# R100 保存結果核驗','', '本作者編寫pure-X policy／audit並審查helper；helper、driver、scorer由其他作者實作。這是跨作者保存證據核驗，不聲稱完全獨立。', '',
      '六NN與評分全部exit 0後才執行。沒有反序列化／重建y、NPZ或模型，没有重新建立prior、計MAE或重抽bootstrap。原始含價格檔僅按來源SHA讀位元組；prefix JSON是保存狀態，不是逐列價格表。','',
      f"核 {result['unique_hashed_files']} 個唯一檔案SHA、9保護檔、三窗相同8欄identity、factor公式、prefix合法性／支持／完整向量與24保存CI端點。六次實際fit合計 {result['fit_seconds']:.3f}秒；不把main時間當fit時間。",'',
      '|窗口|T−v22 overall|T−C overall|T−v22 weighted|T−C weighted|','|---|---:|---:|---:|---:|']
    for s in c.STAGES:
        m=ev['stages'][s]['metrics'];lines.append(f"|{s}|{m['T']['overall_mae']-m['v22']['overall_mae']:+,.2f}|{m['T']['overall_mae']-m['C']['overall_mae']:+,.2f}|{m['T']['weighted_mae']-m['v22']['weighted_mae']:+,.2f}|{m['T']['weighted_mae']-m['C']['weighted_mae']:+,.2f}|")
    lines+=['','固定門檻：'+json.dumps(ev['research_gate'],ensure_ascii=False),'','分群的最強有利／不利證據（描述性、不得據此分流test）：']
    for s in c.STAGES:
        for control,ck in (('v22','comparison'),('C','comparison_vs_C')):
            co=ev['stages'][s][ck];items=[]
            for section in ('strata','forecast_quarters','price_bands'):
                vals=co[section].items() if isinstance(co[section],dict) else ((x['band'],x) for x in co[section])
                items += [{'section':section,'group':k,**v} for k,v in vals if v.get('rows',0) and 'delta' in v]
            items.sort(key=lambda x:x['delta']);lines.append(f"- {s}，T−{control}："+json.dumps({'best':items[0],'worst':items[-1]},ensure_ascii=False))
    lines+=['','|窗口|對照|抽樣|指標|點差|2.5%|97.5%|有效次數|','|---|---|---|---|---:|---:|---:|---:|']
    for r in result['intervals']:lines.append('|'+ '|'.join(str(r[k]) for k in ('stage','control','family','metric','point_delta','lower','upper','valid'))+'|')
    lines+=['','原pure-X計畫與科學計畫分別核SHA；已保留原scorer計畫身分誤比的來源，實際評分來源由保存evaluation_request綁定。',
      '沒有用真價重核reference y／unit SHA或weighted median最優值，僅核來源封存、合法索引／日期、保存state與向量／支持。NN重載精確性採保存trainer收據，未重新載模型；八共享陣列依SHA身份核，未開含y的NPZ。',
      '反覆使用歷史窗口與單seed限制仍在；bootstrap不是Private勝率。固定門檻不等於自動採用，v22唯一提交、Private未知。', '']
    return '\n'.join(lines)

def run(evaluation_dir):
    evaluation_dir=c.safe(evaluation_dir)
    # Completion gate precedes any numeric arrays or pure-X deserialization.
    mx=c.read(MAIN/'execution.json');ex=c.read(evaluation_dir/'execution.json')
    c.need(mx['exit_code']==ex['exit_code']==0 and mx['failure'] is None and ex['failure'] is None,'Wait for successful main AND evaluation')
    c.need(not OUT.exists() and not REPORT.exists(),'Preserve existing audit attempt')
    OUT.mkdir();tick=time.monotonic();code=1;failure=None
    try:
        e=Evidence();seal=e.current(MAIN/'sealed_manifest.json');ev=e.current(evaluation_dir/'results.json')
        c.need(seal['status']=='all_temporal_prior_predictions_sealed' and seal['new_NN_fits']==6 and seal['all_query_reload_exact'] and not seal['query_scores_computed'],'Complete six-fit seal')
        c.need(ev['status']=='completed_three_window_six_fit_evaluation' and ev['checked_stages']==list(c.STAGES),'All three evaluations')
        c.need(ev['sealed_manifest']==c.record(MAIN/'sealed_manifest.json'),'Evaluation seal identity')
        launch=e.document(seal['request']);er=e.document(ev['evaluation_request'])
        c.need(er['research_request']==seal['request'] and er['parameters']==c.evaluation_parameters(),'Actual evaluation request')
        for d in (launch['frozen_source_and_plan'],launch['input_authority_sha256'],ev['input_sha256'],ev['source_sha256'],seal['source_sha256']):e.mapping(d)
        c.need(launch['plan_sha256']==seal['plan_sha256']==c.PLAN_SHA and c.sha(c.PLAN)==c.PLAN_SHA,'Scientific plan')
        c.need(seal['protected_before']==seal['protected_after']==ev['protected_before']==ev['protected_after'] and len(seal['protected_before'])==9,'Nine protected identities');e.mapping(seal['protected_before'])
        audit=e.document(seal['pure_X_audit']);ar=e.current('outputs/round100_x_audit_preflight/request.json')
        c.need(audit['request_sha256']==c.sha('outputs/round100_x_audit_preflight/request.json') and audit['plan_sha256']==ar['plan_sha256']==c.sha('reports/round100_x_audit_plan.md') and ar['scientific_plan_sha256']==c.PLAN_SHA,'Separate pure-X/scientific plans')
        e.mapping(audit['source_sha256']);e.mapping(audit['input_sha256'])
        labels=e.document(seal['fit_label_manifest']);mapping=e.old_array(seal['canonical_mapping']['path'],seal['canonical_mapping']['sha256'])
        e.bind(seal['features']['path'],seal['features']['sha256']);Xraw=pd.read_pickle(c.safe(seal['features']['path']))
        c.need('總價元' not in Xraw and len(Xraw)==455784,'Pure-X only')
        stages={};intervals=[];total_fit=0.
        old79=e.current('outputs/round79/results.json');old83=e.current('outputs/round83/results.json');old60=e.current('outputs/round60/results.json');old63=e.current('outputs/round63/results.json')
        for s in c.STAGES:
            st=e.document(seal['stages'][s]['stage_manifest']);sx=e.document(seal['stages'][s]['execution'])
            c.need(sx['exit_code']==0 and sx['failure'] is None and st['new_NN_fits']==2 and st['all_query_reload_exact'],'Stage completed')
            e.mapping(st['original_input_sha256']);e.mapping(st['artifact_sha256'])
            fi=e.array(st['fit_indices']);qi=e.array(st['query_indices']);rf=e.array(st['fit_source_cache_indices']);rq=e.array(st['query_source_cache_indices'])
            c.exact(rf,mapping[fi],'Fit canonical/raw mapping');c.exact(rq,mapping[qi],'Query canonical/raw mapping')
            X=Xraw.iloc[rf].reset_index(drop=True);p,kept,pr=prefix_audit(e,st,audit['stages'][s],X,fi)
            prep=e.document(st['preparation']);armdata={};fit_seconds=0.
            old_training_path=f'outputs/round63/{s}/model/training.json'
            old_training=e.document({'path':old_training_path,'sha256':old63['stages'][s]['artifact_sha256'][old_training_path]})
            for arm in ('C','T'):
                a=st['arms'][arm];tr=e.document(a['training']);execution=e.document(a['execution'])
                c.need(execution['exit_code']==0 and tr['status']=='completed' and tr['saved_model_reload_exact'] and tr['reload_rows']==st['query_rows'],'Trainer completion/reload')
                c.need(tr['source_sha256']==c.TRAINER_SHA and tr['input_sha256']==a['prepared']['sha256'] and tr['fit_rows']==len(kept),'Trainer identities')
                c.need(tr['configuration']==old_training['configuration'] and tr['architecture_source_sha256']==old_training['architecture_source_sha256'],'Unchanged NN configuration/architecture')
                c.need(not tr['query_labels_available'] and not tr['query_labels_used'] and not tr['sample_weights_batch_renormalized'],'Trainer boundary')
                c.need([v['epoch'] for v in tr['epochs']]==list(range(1,31)),'30 fixed epochs')
                close(tr['fit_seconds'],a['fit_seconds'],'Fit seconds');fit_seconds+=tr['fit_seconds'];armdata[arm]=a['all_input_arrays']
            c.need(set(armdata['C'])==set(armdata['T'])==set(c.ARRAY_NAMES),'Nine arm identities')
            for k in c.ARRAY_NAMES:
                if k!='train_price_factor_million':c.need(armdata['C'][k]==armdata['T'][k],'Shared eight arrays '+k)
                if not k.startswith('train_'):c.need(armdata['C'][k]==prep['original_arrays'][k],'Original query/cardinality '+k)
            area=original.effective_area(X)
            factor=np.asarray(area[kept]*p/1e6,dtype=np.float32);identity(factor,armdata['T']['train_price_factor_million'],'T factor formula')
            oldpath=f'outputs/round60/{s}/train_oof_unit_prior.npy';oldprior=e.old_array(oldpath,old60['stages'][s]['artifact_sha256'][oldpath])
            cf=np.asarray(area*oldprior/1e6,dtype=np.float32)[kept];identity(cf,armdata['C']['train_price_factor_million'],'C original factor subset')
            c.need(armdata['C']['train_price_factor_million']!=armdata['T']['train_price_factor_million'],'Actual factor differs')
            fixed={k:e.array(r) for k,r in st['fixed_branches'].items()};pred={k:e.array(r) for k,r in st['predictions'].items()}
            layers={a:{k:e.array(r) for k,r in rs.items()} for a,rs in st['layers'].items()}
            for arm in ('v22','C','T'):
                c.exact(.75*(.5*fixed['area']+.5*layers[arm]['native'])+.25*fixed['lgb'],layers[arm]['base'],'Base formula')
                c.exact(np.median(np.column_stack([layers[arm]['spatial'],fixed['H'],fixed['U']]),axis=1),pred[arm],'Full median formula')
            for name,r,old in (('v22',79,old79),('v24',83,old83)):
                op=f'outputs/round{r}/{s}/predictions.npy';c.exact(pred[name],e.old_array(op,old['stages'][s]['artifact_sha256'][op]),'Original control vector')
            item=ev['stages'][s]
            for m in list(item['metrics'].values())+[m for a in item['layer_metrics'].values() for m in a.values()]:metric_check(m,len(qi))
            for ck,control,candidate in (('comparison','v22','T'),('comparison_vs_C','C','T'),('comparison_vs_v24','v24','T'),('C_vs_v22','v22','C')):comparison_check(item['metrics'],item[ck],control,candidate)
            for name,old,key in (('v22',old79,'R79'),('v24',old83,'R83')):
                prior_metric=old['stages'][s]['metrics'][key];c.need({k:item['metrics'][name][k] for k in prior_metric}==prior_metric,'Original control aggregate replay')
            cis=saved_CIs(e,s,item);intervals+=cis
            close(fit_seconds,st['fit_seconds'],'Stage fit sum');close(fit_seconds,seal['stages'][s]['fit_seconds'],'Root fit sum');total_fit+=fit_seconds
            stages[s]={**pr,'fit_seconds':fit_seconds,'eight_shared_array_identities_exact':True,'factor_formulas_exact':True,'base_and_median_exact':True,'complete_saved_metric_partitions_verified':True}
        c.need(len(intervals)==24 and ev['conditional_intervals_reported']==24,'All24 CI');close(total_fit,seal['fit_seconds'],'Six fit total')
        m=ev['stages'];latest=m['confirmation']['metrics'];fraction=(latest['v22']['overall_mae']-latest['T']['overall_mae'])/latest['v22']['overall_mae']
        components={'all_three_overall_improve':all(m[s]['metrics']['T']['overall_mae']<m[s]['metrics']['v22']['overall_mae'] for s in c.STAGES),
          'all_three_weighted_improve':all(m[s]['metrics']['T']['weighted_mae']<m[s]['metrics']['v22']['weighted_mae'] for s in c.STAGES),
          'confirmation_at_least_half_percent':fraction>=.005,'no_quarter_or_large_group_deterioration_above_5pct':all(not m[s]['comparison']['quarter_or_large_group_deterioration_above_5pct'] for s in c.STAGES)}
        c.need(components==ev['research_gate']['components'] and all(components.values())==ev['research_gate']['adoption_research_gate_passed'],'Fixed gate arithmetic')
        result={'status':'passed','author_role':'pure-X policy/audit author and helper source reviewer; root wrote helper and other agents wrote driver/scorer',
          'source_sha256':c.sha(__file__),'evaluation_request':ev['evaluation_request'],'sealed_manifest':c.record(MAIN/'sealed_manifest.json'),
          'unique_hashed_files':len(e.hashes),'verified_sha256':e.hashes,'loaded_non_y_arrays':e.arrays,'stages':stages,'intervals':intervals,
          'fit_seconds':total_fit,'main_seconds':mx['elapsed_seconds'],'evaluation_seconds':ex['elapsed_seconds'],'research_gate':ev['research_gate'],
          'labels_NPZ_models_deserialized':False,'new_prior_build_fit_MAE_bootstrap_calls':0,'elapsed_seconds':time.monotonic()-tick}
        c.verify_hashes(e.hashes);c.write(OUT/'results.json',result)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(result,ev))
        code=0;print(json.dumps({'status':'passed','files':len(e.hashes),'prefixes':{s:r['prefix_states'] for s,r in stages.items()},'CIs':24,'elapsed_seconds':result['elapsed_seconds']}))
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'source_sha256':c.sha(__file__),'automatic_retry':False,'labels_deserialized':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evaluation-dir',default='outputs/round100/evaluation');a=ap.parse_args();run(a.evaluation_dir)
