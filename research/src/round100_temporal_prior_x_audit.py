"""Fixed calendar-quarter support audit: pure X only, no prior prices or models."""
from __future__ import annotations
import argparse, ast, hashlib, json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import round60_recency_price_prior as prior

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'reports/round100_x_audit_plan.md'
REQUEST=ROOT/'outputs/round100_x_audit_preflight/request.json'
OUT=ROOT/'outputs/round100_x_audit'
STAGES=('auxiliary','development','confirmation')
CUTOFFS=dict(zip(STAGES,('2021-08-11','2022-08-11','2023-08-11')))
FEATURES='data_cache/train_features.pkl'
MAPPING='outputs/round6/geography/development/train_source_cache_indices.npy'
SCHEMA='reports/feature_schema.json'

def need(ok,msg):
    if not ok:raise ValueError(msg)
def path(p):
    p=Path(p);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    need(p.is_relative_to(ROOT),'Outside workspace');return p
def rel(p):return path(p).relative_to(ROOT).as_posix()
def sha(p):
    h=hashlib.sha256()
    with path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def values(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    p=path(p)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
def locked(p):
    p=path(p);need(p.with_suffix('.sha256').read_text().strip()==sha(p),'Changed JSON: '+rel(p));return read(p)
def sources():
    pending=[path(__file__)];seen=set()
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[x.name for x in n.names] if isinstance(n,ast.Import) else ([n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for name in names:
                q=ROOT/'src'/(name.split('.')[0]+'.py')
                if q.is_file() and q not in seen:pending.append(q)
    return {rel(p):sha(p) for p in sorted(seen)}
def readset():
    names=[FEATURES,MAPPING,SCHEMA,'outputs/round48/plan.json','outputs/round48/plan.sha256','outputs/round60/plan.json','outputs/round60/plan.sha256']
    for s in STAGES:
        names += [f'outputs/round{r}/{s}/fit_indices.npy' for r in (48,60,63)]
        names += [f'outputs/round60/{s}/prior_fold_ids.npy']
        for r,base in ((60,'prior_preparation'),(63,'preparation')):
            names += [f'outputs/round{r}/{s}/{base}.json',f'outputs/round{r}/{s}/{base}.sha256']
    return names
def rules():
    return {'partition':'calendar quarter Q-DEC; Jan/Apr/Jul/Oct 1','minimum_all_five_prefix_reference_rows':30,
            'minimum_cell_rows':30,'reference':'valid date, day < quarter start, original fold != held fold',
            'eligible':'valid date AND all five reference counts >=30','invalid_date_quarter_sentinel':-1,
            'threshold_grid':False,'warmup_policy_is_new_not_original_global_rule':True}
def freeze():
    need(not OUT.exists() and not REQUEST.exists(),'Existing actual audit/request; preserve')
    REQUEST.parent.mkdir(parents=True,exist_ok=True)
    p48=locked('outputs/round48/plan.json')
    for p in (FEATURES,MAPPING,SCHEMA):need(sha(p)==p48['input_sha256'][p],'Old X identity changed')
    checks=self_check()
    write(REQUEST.parent/'selfcheck.json',checks)
    write(REQUEST,{'status':'pure_X_audit_frozen_before_feature_deserialization','plan_sha256':sha(PLAN),
          'scientific_plan_sha256':sha('reports/round100_plan.md'),'source_sha256':sources(),
          'input_sha256':{p:sha(p) for p in readset()},'stage_order':list(STAGES),'rules':rules(),
          'created_at_utc':now(),'labels_or_real_X_deserialized':False})
    print(json.dumps({'status':'frozen','request_sha256':sha(REQUEST),'checks':checks['check_count']}))
def folds_for(X,indices):
    groups,fallback=prior.original.grouping.cluster_keys(X,indices)
    assignments={g:int.from_bytes(hashlib.sha256(('42|'+g).encode('utf-8')).digest()[:8],'little')%5 for g in np.unique(groups)}
    return np.asarray([assignments[g] for g in groups],dtype=np.int64),groups,fallback
def quarters_for(days,valid):
    q=np.full(len(days),-1,dtype=np.int64)
    dates=pd.Timestamp('2000-01-01')+pd.to_timedelta(days[valid],unit='D')
    starts=dates.to_period('Q-DEC').start_time
    q[valid]=np.asarray((starts-pd.Timestamp('2000-01-01')).days,dtype=np.int64)
    return q
def policy(days,valid,folds):
    q=quarters_for(days,valid);eligible=np.zeros(len(days),dtype=bool);records=[]
    for start in np.unique(q[valid]):
        prefix=valid&(days<start)
        refs=[int((prefix&(folds!=f)).sum()) for f in range(5)]
        yes=min(refs)>=30;held=valid&(q==start)
        if yes:eligible[held]=True
        records.append({'quarter_start_day':int(start),'quarter':str((pd.Timestamp('2000-01-01')+pd.Timedelta(days=int(start))).to_period('Q-DEC')),
                        'rows':int(held.sum()),'reference_rows_by_held_fold':refs,'minimum_reference_rows':min(refs),'eligible_quarter':yes})
    return q,eligible,records
def selected_support(codes,sizes,reference,held):
    level=np.full(int(held.sum()),4,dtype=np.int8);support=np.full(len(level),int(reference.sum()),dtype=np.int64)
    if not reference.any():return np.full(len(level),5,dtype=np.int8),np.zeros(len(level),dtype=np.int64)
    for j,(code,size) in enumerate(zip(codes,sizes)):
        counts=np.bincount(code[reference],minlength=size);n=counts[code[held]]
        take=(level==4)&(n>=30);level[take]=j;support[take]=n[take]
    return level,support
def support_summary(level,support):
    return {'rows':len(level),'level_rows':{str(j):int((level==j).sum()) for j in range(6)},
            'support_min':int(support.min()) if len(support) else None,
            'support_median':float(np.median(support)) if len(support) else None,
            'support_max':int(support.max()) if len(support) else None}
def save_array(folder,name,a):
    p=folder/(name+'.npy')
    with p.open('xb') as f:np.save(f,a,allow_pickle=False)
    b=np.load(p,allow_pickle=False);need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'Array reload')
    return {'path':rel(p),'sha256':sha(p),'dtype':str(a.dtype),'shape':list(a.shape),'values_sha256':values(a)}
def weight_summary(w,mask):
    v=w[mask].astype(np.float64);sm=float(v.sum());sq=float(np.square(v).sum())
    return {'rows':len(v),'sum':sm,'fraction_original_sum':sm/float(w.astype(np.float64).sum()),'ESS':sm*sm/sq if sq else 0.}
def inspect_stage(X,indices,saved_folds,p63,p60,folder):
    days,bad,outside=prior._dates(X)
    valid=np.isfinite(days)&(bad==0)&(outside==0)
    need(np.isfinite(bad).all() and np.isfinite(outside).all(),'Nonfinite date flags')
    folds,groups,fallback=folds_for(X,indices)
    need(folds.dtype==saved_folds.dtype and folds.tobytes()==saved_folds.tobytes(),'Saved R60 folds differ')
    ev=p60['prior_evidence']
    need(values(folds)==ev['fold_ids_sha256'] and values(indices)==ev['source_indices_sha256'],'Prior fold/index identity')
    group_sha=hashlib.sha256(json.dumps(groups.tolist(),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    need(group_sha==ev['group_keys_sha256'],'Original grouping differs')
    q,eligible,records=policy(days,valid,folds)
    keys=prior.original._keys(X)
    packed=[pd.factorize(k,sort=True) for k in keys];codes=[x[0] for x in packed];sizes=[len(x[1]) for x in packed]
    original_level=np.empty(len(X),dtype=np.int8);original_support=np.empty(len(X),dtype=np.int64)
    temporal_level=np.full(len(X),6,dtype=np.int8);temporal_support=np.zeros(len(X),dtype=np.int64)
    for f in range(5):
        held=folds==f
        original_level[held],original_support[held]=selected_support(codes,sizes,~held,held)
    for record in records:
        start=record['quarter_start_day'];record['folds']=[]
        for f in range(5):
            ref=valid&(days<start)&(folds!=f);held=valid&(q==start)&(folds==f)
            level,support=selected_support(codes,sizes,ref,held)
            temporal_level[held],temporal_support[held]=level,support
            record['folds'].append({'held_fold':f,'held_rows':int(held.sum()),'reference_rows':int(ref.sum()),
                'reference_groups':len(np.unique(groups[ref])),'reference_day_min':float(days[ref].min()) if ref.any() else None,
                'reference_day_max':float(days[ref].max()) if ref.any() else None,'selected_support':support_summary(level,support)})
    w64,old_valid,dateinfo=prior._weights((days,bad,outside));need(np.array_equal(old_valid,valid),'Date rule differs')
    w32=w64.astype(np.float32);need(values(w32)==p63['train_sample_weight_sha256'],'Original R63 float32 date weights differ')
    warm=valid&~eligible
    masks={'all':np.ones(len(X),dtype=bool),'eligible':eligible,'warmup':warm,'invalid_date':~valid}
    need(eligible.sum()+warm.sum()+(~valid).sum()==len(X),'Mask partition')
    arrays={k:save_array(folder,k,a) for k,a in {'fit_indices':indices,'fold_ids':folds,'valid_date_mask':valid,
             'quarter_start_days':q,'eligible_mask':eligible}.items()}
    yes=[r['quarter'] for r in records if r['eligible_quarter']]
    return {'fit_rows':len(X),'eligible_rows':int(eligible.sum()),'warmup_rows':int(warm.sum()),'invalid_date_rows':int((~valid).sum()),
        'invalid_flag_rows':int((bad!=0).sum()),'out_of_range_flag_rows':int((outside!=0).sum()),'nonfinite_day_rows':int((~np.isfinite(days)).sum()),
        'earliest_eligible_quarter':yes[0] if yes else None,'warmup_policy_not_selected_from_scores':True,'folds_match_R60':True,
        'group_keys_sha256':group_sha,'fallback_group_counts':fallback,'calendar_quarters':records,'arrays':arrays,
        'temporal_support_all_valid':support_summary(temporal_level[valid],temporal_support[valid]),
        'temporal_support_eligible':support_summary(temporal_level[eligible],temporal_support[eligible]),
        'original_group_OOF_support_same_eligible':support_summary(original_level[eligible],original_support[eligible]),
        'original_date_weight32_values_sha256':values(w32),'date_weight_mass':{k:weight_summary(w32,m) for k,m in masks.items()},
        'new_weight_normalization':False,'no_price_prior_built':True}
def render(result):
    lines=['# R100 純 X 時間先驗可行性','', '固定日曆季、五折 prefix 最小30列；沒有讀價格、估prior或訓練。30是新的暖身慣例，不是舊global門檻或已證最優。', '',
           '|窗口|fit列|保留|暖身排除|無效日期|最早可用季|原日期權重保留質量|','|---|---:|---:|---:|---:|---|---:|']
    for s in STAGES:
        r=result['stages'][s];lines.append(f"|{s}|{r['fit_rows']}|{r['eligible_rows']}|{r['warmup_rows']}|{r['invalid_date_rows']}|{r['earliest_eligible_quarter']}|{r['date_weight_mass']['eligible']['fraction_original_sum']:.8%}|")
    for s in STAGES:
        r=result['stages'][s];lines+=['','## '+s,'','|日曆季|列數|五個held-fold reference列數|整季可用|','|---|---:|---|---|']
        for q in r['calendar_quarters']:lines.append(f"|{q['quarter']}|{q['rows']}|{' / '.join(map(str,q['reference_rows_by_held_fold']))}|{q['eligible_quarter']}|")
        lines+=['','相同eligible列之支持層级（0–3依原四層、4 global、5無reference）：',
                '- 時間前置：'+json.dumps(r['temporal_support_eligible'],ensure_ascii=False),
                '- 原分組OOF：'+json.dumps(r['original_group_OOF_support_same_eligible'],ensure_ascii=False)]
    lines+=['','每季各fold的參考群数、最早／最晚日期、完整支持量及掉列權重質量均見results.json。精確mask／fold已保存重載。',
            '無效日期無法確定prefix，另列而不猜日期；原R60中性日期policy與此次時間篩選不同。支持量不是獨立實體建案數、可靠機率或MAE改善證明。',
            '只描述純X可行性；尚未建立價格先驗或新模型。正式matched-sample控制、NN訓練與query評分由另外固定計畫及主執行緒安排。v22／未知Private不變。','']
    return '\n'.join(lines)
def run():
    need(not OUT.exists(),'Audit exists; preserve old attempt');request=locked(REQUEST)
    need(request['rules']==rules() and request['plan_sha256']==sha(PLAN) and request['source_sha256']==sources(),'Frozen audit differs')
    for p,h in request['input_sha256'].items():need(sha(p)==h,'Changed audit input: '+p)
    OUT.mkdir();start=now();tick=time.monotonic();code=1;failure=None;loaded=[]
    try:
        p48=locked('outputs/round48/plan.json');p60plan=locked('outputs/round60/plan.json')
        need(p60plan['prior_rules']==prior.RULES,'Original prior rules changed')
        schema=read(SCHEMA);Xraw=pd.read_pickle(ROOT/FEATURES);loaded.append(FEATURES)
        need(isinstance(Xraw,pd.DataFrame) and Xraw.index.equals(pd.RangeIndex(455784)) and list(Xraw.columns)==schema['columns'],'Pure-X schema')
        need(not {'總價元','y','target','total_price','編號'}.intersection(Xraw.columns),'Forbidden target columns')
        allowed_npy={p for p in request['input_sha256'] if p.endswith('.npy')}
        def array(p):
            need(p in allowed_npy,'Unapproved array');loaded.append(p);return np.load(ROOT/p,allow_pickle=False)
        mapping=array(MAPPING)
        need(mapping.dtype==np.int64 and mapping.shape==(455255,) and (np.diff(mapping)>0).all() and mapping.min()>=0 and mapping.max()<len(Xraw),'Canonical mapping')
        Xall=Xraw.iloc[mapping].reset_index(drop=True);del Xraw
        stages={}
        for s in STAGES:
            folder=OUT/s;folder.mkdir();indices=array(f'outputs/round48/{s}/fit_indices.npy')
            for r in (60,63):need(indices.tobytes()==array(f'outputs/round{r}/{s}/fit_indices.npy').tobytes(),'Saved fit indices differ')
            contract=p48['split_contract'][s]
            need(indices.dtype==np.int64 and indices.shape==(contract['fit_rows'],) and values(indices)==contract['fit_indices_content_sha256'],'Canonical fit contract')
            days=Xall.trade_day_index.to_numpy(dtype=np.float64)
            cut=(pd.Timestamp(CUTOFFS[s])-pd.Timestamp('2000-01-01')).days
            need(np.array_equal(indices,np.flatnonzero(np.isfinite(days)&(days<cut))),'Original fit cutoff mask')
            X=Xall.iloc[indices].reset_index(drop=True)
            p60=locked(f'outputs/round60/{s}/prior_preparation.json');p63=locked(f'outputs/round63/{s}/preparation.json')
            stages[s]=inspect_stage(X,indices,array(f'outputs/round60/{s}/prior_fold_ids.npy'),p63,p60,folder)
        for p,h in request['input_sha256'].items():need(sha(p)==h,'Audit input mutated')
        need(sources()==request['source_sha256'],'Source changed during audit')
        result={'status':'completed_pure_X_support_audit','round':100,'rules':rules(),'stage_order':list(STAGES),'stages':stages,
                'source_sha256':sources(),'input_sha256':request['input_sha256'],'request_sha256':sha(REQUEST),'plan_sha256':sha(PLAN),
                'deserialized_data_files':loaded,'labels_rawpricepickle_NPZ_models_loaded':False,'prior_build_calls':0,'fit_predict_score_calls':0,
                'source_and_inputs_unchanged':True,'full_pure_feature_table_deserialized_before_fit_slicing':True,
                'elapsed_seconds':time.monotonic()-tick,'completed_at_utc':now()}
        write(OUT/'results.json',result)
        with (OUT/'report.md').open('x',encoding='utf8',newline='\n') as f:f.write(render(result))
        code=0;print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'eligible':{s:r['eligible_rows'] for s,r in stages.items()}}))
    except Exception:failure=traceback.format_exc();raise
    finally:write(OUT/'execution.json',{'exit_code':code,'failure':failure,'started_at_utc':start,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-tick,
                'source_sha256':sha(__file__),'labels_loaded':False,'prior_build_or_fit_predict_score_calls':0,'automatic_retry':False})
def self_check():
    jan=(pd.Timestamp('2020-01-01')-pd.Timestamp('2000-01-01')).days
    apr=(pd.Timestamp('2020-04-01')-pd.Timestamp('2000-01-01')).days
    days=np.r_[np.full(40,jan),np.full(5,apr),np.nan];valid=np.isfinite(days);folds=np.arange(46,dtype=np.int64)%5
    q,mask,rs=policy(days,valid,folds)
    need(rs[0]['reference_rows_by_held_fold']==[0]*5 and not rs[0]['eligible_quarter'],'Coldstart')
    need(rs[1]['reference_rows_by_held_fold']==[32]*5 and rs[1]['eligible_quarter'],'Prefix five-fold support')
    need(np.array_equal(np.flatnonzero(mask),np.arange(40,45)) and q[-1]==-1,'Quarter boundary/invalid mask')
    sparse=np.r_[np.full(36,jan),apr];_,m,r=policy(sparse,np.ones(37,dtype=bool),np.arange(37,dtype=np.int64)%5)
    need(min(r[1]['reference_rows_by_held_fold'])==28 and not m.any(),'One insufficient fold blocks whole quarter')
    codes=[np.zeros(35,dtype=np.int64)]*4;sizes=[1]*4;held=np.arange(35)>=30
    level,support=selected_support(codes,sizes,~held,held);need((level==0).all() and (support==30).all(),'Cell exact30')
    ref=np.arange(35)<29;level,support=selected_support(codes,sizes,ref,held);need((level==4).all() and (support==29).all(),'Global has no old30 threshold')
    return {'status':'passed','check_count':6,'synthetic_only':True,'real_X_labels_loaded':False,'prior_or_model_calls':0,'source_sha256':sha(__file__)}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    need(a.freeze!=a.run,'Choose one mode');freeze() if a.freeze else run()
