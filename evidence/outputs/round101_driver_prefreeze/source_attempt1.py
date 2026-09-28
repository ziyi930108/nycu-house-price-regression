"""R101 fixed descriptive prior diagnostics. No model, prior prediction or query labels.

Run each stage in a separate process; the parent reads metadata/aggregate JSON only.
The full pure-X cache is sliced by the frozen canonical mapping. Price arrays are
limited to the current stage's fit-only labels, prepared loss inputs, saved priors
and saved native predictions. No row-level derived values are written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round60_recency_price_prior as prior
import round101_statistics as statistics

PLAN='reports/round101_plan.md'
PLAN_SHA='f95a13d1a78afbd9949d4738403880c855ed54da6618ab453be4eb909447ef7d'
MAIN='outputs/round100/sealed_manifest.json'
MAIN_EXEC='outputs/round100/execution.json'
EVAL_EXEC='outputs/round100/evaluation/execution.json'
OUT='outputs/round101'
STAGES=c.STAGES
LEVEL_NAMES=['city_district_transaction_building','city_transaction_building','transaction_building','transaction','global']

def parameters():
    return {'round':101,'stage_order':list(STAGES),'plan_sha256':PLAN_SHA,
        'fit_groups':['calendar_quarter','age_0_365_730_inf','support_transition_25'],
        'query_groups':['relative_forecast_month_12','full_outer_fit_seen_new_missing','support_level_5'],
        'prior_ratio_bands':['<0.9','[0.9,1.1]','>1.1'],
        'quantile_percentiles':[0,1,10,25,50,75,90,99,100],
        'quantile_unweighted':'numpy linear','quantile_weighted':'stable inverse CDF; no interpolation',
        'ratio_target':'saved train_y_million float32 / saved factor float32, both promoted float64',
        'effective_ratio_loss_weight':'saved raw float32 date weight promoted float64 * factor64',
        'query_ratio':'saved native TWD / (shared factor64 * 1e6); recovered not original bytes',
        'source_support_levels':LEVEL_NAMES,'new_native_fits':0,'prior_builds':0,'model_loads':0,
        'model_predict_calls':0,'query_labels_read':False,'MAE_calls':0,'bootstrap_calls':0,
        'new_CSV':False,'automatic_adoption':False}

def source_hashes():
    return c.source_hashes(__file__)

def doc(rec):
    c.need(c.sha(rec['path'])==rec['sha256'],'Metadata SHA mismatch: '+rec['path'])
    return c.read(rec['path'])

def metadata_readset(stage):
    """Build the exact current-stage readset from frozen JSON authorities only.

    This function never opens array/model/feature/label files. The request freezer
    hashes the returned allowed paths; raw train price pickle is never included.
    """
    c.need(stage in STAGES,'Unknown stage')
    seal=c.read(MAIN)
    c.need(seal['status']=='all_temporal_prior_predictions_sealed'
        and seal['stage_order']==list(STAGES) and seal['new_NN_fits']==6
        and seal['all_query_reload_exact'] and seal['plan_sha256']==c.PLAN_SHA,'R100 main contract')
    sm=doc(seal['stages'][stage]['stage_manifest'])
    c.need(sm['stage']==stage and sm['status']=='temporal_prior_stage_predictions_sealed','R100 stage contract')
    labels=doc(seal['fit_label_manifest']);audit=doc(seal['pure_X_audit'])
    readset={MAIN:c.sha(MAIN),MAIN_EXEC:c.sha(MAIN_EXEC),EVAL_EXEC:c.sha(EVAL_EXEC)}
    def add(rec):
        path=c.rel(rec['path']);digest=rec['sha256']
        c.need(path not in readset or readset[path]==digest,'Authority conflict: '+path)
        readset[path]=digest
    for rec in [seal['stages'][stage]['stage_manifest'],seal['stages'][stage]['execution'],
                seal['features'],seal['canonical_mapping'],seal['fit_label_manifest'],seal['pure_X_audit']]:add(rec)
    for name in ['fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices',
                 'eligible_positions','temporal_prior','temporal_prior_evidence']:add(sm[name])
    for rec in labels['stages'][stage]['arrays'].values():add(rec)
    for rec in audit['stages'][stage]['arrays'].values():add(rec)
    for arm in ['C','T']:
        for name in ['prepared','training','execution']:add(sm['arms'][arm][name])
    for arm in ['v22','C','T']:add(sm['layers'][arm]['native'])
    for name in ['fit_indices.npy','query_indices.npy','prior_fold_ids.npy','train_oof_unit_prior.npy',
                 'query_unit_prior.npy','price_prior_state.json','prior_preparation.json']:
        path=f'outputs/round60/{stage}/{name}';add({'path':path,'sha256':sm['original_input_sha256'][path]})
    for name in ['prepared.npz','model/training.json']:
        path=f'outputs/round63/{stage}/{name}';add({'path':path,'sha256':sm['original_input_sha256'][path]})
    for name,digest in sm['artifact_sha256'].items():
        if name.startswith(f'outputs/round100/{stage}/prefix_states/') and name.endswith('.json'):
            add({'path':name,'sha256':digest})
    c.need(all(path!=c.RAW and not path.endswith(('model.pt','.cbm')) for path in readset),'Forbidden input')
    for path in readset:
        for other in STAGES:
            if other!=stage:
                c.need('/'+other+'/' not in path or path==c.CANONICAL,'Cross-stage input: '+path)
    return dict(sorted(readset.items()))

def check_request(path,stage):
    req=c.read(path)
    c.need(req['schema_version']==1 and req['plan']=={'path':PLAN,'sha256':PLAN_SHA}
        and c.sha(PLAN)==PLAN_SHA and req['parameters']==parameters()
        and req['stage_order']==list(STAGES),'Frozen R101 request/plan differs')
    hashes=source_hashes()
    c.need(all(req['source_sha256'].get(k)==v for k,v in hashes.items()),'Source closure differs')
    c.verify_hashes(req['source_sha256'])
    c.need(req['protected_sha256']==c.protected_hashes(),'Protected source differs')
    expected=metadata_readset(stage)
    c.need(req['stages'][stage]['input_sha256']==expected,'Exact same-stage readset differs')
    c.verify_hashes(expected)
    c.need(c.read(MAIN_EXEC)['exit_code']==0 and c.read(EVAL_EXEC)['exit_code']==0,'R100 not terminal')
    c.verify_hashes(c.read(MAIN)['source_sha256'])
    return req,expected

class Reader:
    def __init__(self,allowed):self.allowed=allowed;self.opened=set()
    def path(self,name):
        name=c.rel(name);c.need(name in self.allowed,'Outside frozen readset: '+name)
        c.need(c.sha(name)==self.allowed[name],'Input changed: '+name)
        self.opened.add(name);return c.safe(name)
    def json(self,rec):
        name=rec['path'] if isinstance(rec,dict) else rec
        if isinstance(rec,dict):c.need(self.allowed.get(c.rel(name))==rec['sha256'],'Record authority mismatch')
        return json.loads(self.path(name).read_text(encoding='utf-8-sig'))
    def array(self,rec):
        name=rec['path'] if isinstance(rec,dict) else rec
        a=np.load(self.path(name),allow_pickle=False)
        c.need(isinstance(a,np.ndarray) and not a.dtype.hasobject,'Array required')
        if isinstance(rec,dict):
            c.need(self.allowed.get(c.rel(name))==rec['sha256'],'Array authority mismatch')
            c.need(c.identity(a)=={k:rec[k] for k in ['dtype','shape','values_sha256']},'Array content differs')
        return a

def positive(a,rows,dtype=None):
    c.need(isinstance(a,np.ndarray) and a.shape==(rows,) and (dtype is None or a.dtype==dtype)
        and np.isfinite(a).all() and (a>0).all(),'Invalid positive numeric vector')

def index(a,rows):
    c.need(a.dtype==np.int64 and a.ndim==1 and len(a)>0 and (a>=0).all() and (a<rows).all()
        and (np.diff(a)>0).all(),'Ordered canonical/position indices required')

def load_prepared(reader,sm,y,kept,stage):
    """Read C/T shared matrices one key at a time; retain only diagnostic vectors."""
    result={'C':{},'T':{}};checks=[]
    with np.load(reader.path(sm['arms']['C']['prepared']['path']),allow_pickle=False) as C, \
         np.load(reader.path(sm['arms']['T']['prepared']['path']),allow_pickle=False) as T:
        c.need(set(C.files)==set(T.files)==set(c.ARRAY_NAMES),'Exactly nine prepared keys, no query targets')
        for key in c.ARRAY_NAMES:
            a,b=C[key],T[key]
            for arm,v in [('C',a),('T',b)]:
                c.need(c.identity(v)==sm['arms'][arm]['all_input_arrays'][key],'Saved prepared identity differs')
            if key!='train_price_factor_million':c.exact(a,b,'Eight shared C/T arrays differ: '+key)
            if key in ['train_y_million','train_price_factor_million','query_price_factor_million','train_sample_weight']:
                result['C'][key]=a;result['T'][key]=b
            del a,b
    with np.load(reader.path(f'outputs/round63/{stage}/prepared.npz'),allow_pickle=False) as z:
        c.need(set(z.files)==set(c.ARRAY_NAMES),'Original prepared keys differ')
        for key in ['train_y_million','train_price_factor_million','query_price_factor_million','train_sample_weight']:
            old=z[key]
            if key.startswith('train_'):old=old[kept]
            c.exact(old,result['C'][key],'Original R63 C subset/query differs: '+key)
    c.exact(np.asarray(y[kept]/1e6,dtype=np.float32),result['C']['train_y_million'],'Full precision fit y mapping differs')
    for arm in ['C','T']:
        for key,v in result[arm].items():positive(v,sm['query_rows'] if key.startswith('query_') else len(kept),np.dtype('float32'))
    return result

def route_saved_support(state,keys):
    """Inspect routing and raw rows only; never calculate or return price priors."""
    c.need(len(state['levels'])==4 and state['rules']==prior.RULES,'Saved prior hierarchy differs')
    n=len(keys[0]);level=np.full(n,4,dtype=np.int8)
    support=np.full(n,int(state['global_raw_rows']),dtype=np.int64)
    c.need((support>0).all(),'Empty global support')
    for j,(cols,key,layer) in enumerate(zip(prior.LEVELS,keys,state['levels'])):
        c.need(layer['columns']==cols,'Support hierarchy order differs')
        for i in np.flatnonzero(level==4):
            cell=layer['cells'].get(key[i])
            if cell is not None:
                rows=cell['rows'];c.need(isinstance(rows,int) and rows>=30,'Invalid stored cell support')
                level[i]=j;support[i]=rows
    return level,support

def grouped_oof_support(keys,folds,kept):
    packed=[pd.factorize(k,sort=True) for k in keys]
    level=np.full(len(folds),4,dtype=np.int8);support=np.zeros(len(folds),dtype=np.int64)
    for fold in range(5):
        held=folds==fold;reference=~held;support[held]=int(reference.sum())
        for j,(codes,uniques) in enumerate(packed):
            counts=np.bincount(codes[reference],minlength=len(uniques))
            take=held&(level==4)&(counts[codes]>=30)
            level[take]=j;support[take]=counts[codes[take]]
    c.need((support[kept]>0).all(),'OOF empty reference')
    return level[kept],support[kept]

def level_counts(a):return {str(i):int((a==i).sum()) for i in range(5)}

def day(s):return int((pd.Timestamp(s)-pd.Timestamp('2000-01-01')).days)

def fit_group_masks(days,cutoff,oldlevel,newlevel):
    dates=pd.Timestamp('2000-01-01')+pd.to_timedelta(days,unit='D')
    quarters=np.asarray(dates.to_period('Q-DEC').astype(str))
    age=day(cutoff)-days;c.need(np.isfinite(age).all() and (age>=0).all(),'Negative/invalid fit age')
    return {'calendar_quarter':{q:quarters==q for q in sorted(set(quarters))},
        'age_days':{'[0,365)':age<365,'[365,730)':(age>=365)&(age<730),'[730,inf)':age>=730},
        'support_transition':{f'{a}->{b}':(oldlevel==a)&(newlevel==b) for a in range(5) for b in range(5)}}

def query_group_masks(X,Xq,cutoff,level):
    months=prior.original.grouping.relative_months(Xq.trade_day_index,cutoff)
    labels=prior.original.grouping.group_labels(X,Xq)
    return {'relative_forecast_month':{str(i):months==i for i in range(12)},
        'project_status':{g:labels==g for g in ['seen','new_nonmissing','missing']},
        'support_level':{str(i):level==i for i in range(5)}}

def ratio_quantities(y32,fc32,ft32,w32):
    for a in [y32,fc32,ft32,w32]:positive(a,len(y32),np.dtype('float32'))
    y,fc,ft,w=[a.astype(np.float64) for a in [y32,fc32,ft32,w32]]
    rc,rt=y/fc,y/ft;factor_ratio=ft/fc
    c.need(np.allclose(rt/rc,1/factor_ratio,rtol=1e-14,atol=0),'Ratio/factor reciprocal differs')
    return {'factor_C':fc,'factor_T':ft,'target_ratio_C':rc,'target_ratio_T':rt,
        'factor_T_over_C':factor_ratio,'log_factor_T_over_C':np.log(factor_ratio),
        'date_weight':w,'effective_ratio_weight_C':w*fc,'effective_ratio_weight_T':w*ft}

def initialize_ratio(y32,f32):
    median=float(np.median(y32.astype(np.float64)/f32.astype(np.float64)))
    return {'median':median,'inverseSoftplus':median+float(np.log(-np.expm1(-median)))}

def ratio_bands(ratio,w,mask):
    denom=float(w.sum(dtype=np.float64));bins={'<0.9':ratio<.9,'[0.9,1.1]':(ratio>=.9)&(ratio<=1.1),'>1.1':ratio>1.1}
    return {name:{'rows':int((mask&b).sum()),'original_w_mass':float(w[mask&b].sum(dtype=np.float64)),
        'fraction_full_eligible_w':float(w[mask&b].sum(dtype=np.float64))/denom} for name,b in bins.items()}

def distributions(values,mask,weights=None,effective=None):
    result={}
    for name,x in values.items():
        row={'unweighted':statistics.describe(x[mask],reference_weight_sum=len(x))}
        if weights is not None:row['original_date_weight']=statistics.describe(x[mask],weights[mask],float(weights.sum(dtype=np.float64)))
        if effective is not None and name in effective:
            w=effective[name];row['effective_ratio_loss_weight']=statistics.describe(x[mask],w[mask],float(w.sum(dtype=np.float64)))
        result[name]=row
    return result

def grouped_descriptions(values,groups,weights=None,effective=None,prior_ratio=None):
    n=len(next(iter(values.values())))
    def one(mask):
        c.need(mask.dtype==bool and mask.shape==(n,),'Group mask shape')
        result={'rows':int(mask.sum()),'distributions':distributions(values,mask,weights,effective)}
        if prior_ratio is not None:result['T_over_C_prior_bands']=ratio_bands(prior_ratio,weights,mask)
        return result
    result={'all':one(np.ones(n,dtype=bool)),'groups':{}}
    for name,masks in groups.items():
        c.need(np.array_equal(sum(m.astype(np.int8) for m in masks.values()),np.ones(n,dtype=np.int8)),'Group partition differs: '+name)
        result['groups'][name]={key:one(mask) for key,mask in masks.items()}
    return result

def support_date_summary(value):
    # Older saved R60 summaries lack sum_squares/ESS; never invent it from rows.
    names=['reference_rows','valid_date_rows','neutral_date_rows','max_valid_reference_trade_day_index',
        'min_valid_reference_trade_day_index','weight_sum','weight_min','weight_max','weight_mean','weights_sha256']
    return {**{k:value[k] for k in names},'reference_date_weight_ESS':None,
        'ESS_status':'not present in immutable saved date summary; no independent-project interpretation'}

def run_stage(stage,request_path):
    tick=time.monotonic();req,readset=check_request(request_path,stage)
    out=c.safe(f'{OUT}/{stage}');c.need(not out.exists(),'Refuse overwrite/retry of stage')
    out.mkdir(parents=True);c.write(out/'started.json',{'stage':stage,'created_at_utc':c.now(),'request':c.record(request_path)})
    failure=None;reader=Reader(readset);protected=c.protected_hashes()
    try:
        main=reader.json(MAIN);sm=reader.json(main['stages'][stage]['stage_manifest'])
        labels=reader.json(main['fit_label_manifest']);audit=reader.json(main['pure_X_audit']);ax=audit['stages'][stage]
        start,end,nf,nq=c.SPLITS[stage]
        c.need((sm['cutoff'],sm['query_end'],sm['fit_rows'],sm['query_rows'])==(start,end,nf,nq),'Split contract')
        fi=reader.array(sm['fit_indices']);qi=reader.array(sm['query_indices'])
        mapping=reader.array(main['canonical_mapping']);index(mapping,455784);c.need(len(mapping)==455255,'Canonical rows')
        index(fi,len(mapping));index(qi,len(mapping));c.need(not np.intersect1d(fi,qi).size,'Fit/query overlap')
        rawfi=reader.array(sm['fit_source_cache_indices']);rawqi=reader.array(sm['query_source_cache_indices'])
        c.exact(mapping[fi],rawfi,'Raw fit mapping');c.exact(mapping[qi],rawqi,'Raw query mapping')
        arrays=labels['stages'][stage]['arrays']
        for name,value in [('fit_indices',fi),('fit_source_cache_indices',rawfi)]:c.exact(reader.array(arrays[name]),value,'Extractor index identity')
        y=reader.array(arrays['fit_y']);positive(y,nf,np.dtype('float64'))
        Xraw=pd.read_pickle(reader.path(main['features']['path']))
        c.need(len(Xraw)==455784 and '總價元' not in Xraw and Xraw.index.equals(pd.RangeIndex(len(Xraw))),'Pure-X cache schema')
        X=Xraw.iloc[rawfi].reset_index(drop=True);Xq=Xraw.iloc[rawqi].reset_index(drop=True);del Xraw
        days,bad,outside=prior._dates(X);qdays,qbad,qoutside=prior._dates(Xq)
        valid=np.isfinite(days)&(bad==0)&(outside==0)
        c.need(valid.all() and (days<day(start)).all() and np.isfinite(qdays).all()
            and (qbad==0).all() and (qoutside==0).all() and ((qdays>=day(start))&(qdays<=day(end))).all(),'Outer date boundary')
        c.exact(fi,reader.array(ax['arrays']['fit_indices']),'Audit fit indices')
        c.exact(valid,reader.array(ax['arrays']['valid_date_mask']),'Valid date mask')
        eligible=reader.array(ax['arrays']['eligible_mask']);folds=reader.array(ax['arrays']['fold_ids'])
        qstarts=reader.array(ax['arrays']['quarter_start_days']);kept=reader.array(sm['eligible_positions'])
        c.need(eligible.dtype==bool and eligible.shape==(nf,) and folds.dtype==np.int64 and folds.shape==(nf,)
            and ((folds>=0)&(folds<5)).all(),'Mask/fold schema');index(kept,nf)
        c.exact(np.flatnonzero(eligible).astype(np.int64),kept,'Eligible order')
        c.exact(folds,reader.array(f'outputs/round60/{stage}/prior_fold_ids.npy'),'Saved R60 folds')
        for k,a in [('fit',fi),('query',qi)]:c.exact(reader.array(f'outputs/round60/{stage}/{k}_indices.npy'),a,'Saved prior indices')
        groups,_=prior.original.grouping.cluster_keys(X,fi)
        foldmap={g:int.from_bytes(hashlib.sha256(('42|'+g).encode()).digest()[:8],'little')%5 for g in np.unique(groups)}
        c.exact(np.asarray([foldmap[g] for g in groups],dtype=np.int64),folds,'Original fullfit grouping')
        prepared=load_prepared(reader,sm,y,kept,stage)
        pcfull=reader.array(f'outputs/round60/{stage}/train_oof_unit_prior.npy');positive(pcfull,nf,np.dtype('float64'))
        pc=pcfull[kept];pt=reader.array(sm['temporal_prior']);positive(pt,len(kept),np.dtype('float64'))
        pq=reader.array(f'outputs/round60/{stage}/query_unit_prior.npy');positive(pq,nq,np.dtype('float64'))
        af=prior.original.effective_area(X)[kept];aq=prior.original.effective_area(Xq)
        for arm,p in [('C',pc),('T',pt)]:
            c.exact(np.asarray(af*p/1e6,dtype=np.float32),prepared[arm]['train_price_factor_million'],'Fit factor formula')
            c.exact(np.asarray(aq*pq/1e6,dtype=np.float32),prepared[arm]['query_price_factor_million'],'Query factor formula')
        keys=prior.original._keys(X);qkeys=prior.original._keys(Xq)
        oldlevel,oldsupport=grouped_oof_support(keys,folds,kept)
        c.need(level_counts(oldlevel)=={str(i):ax['original_group_OOF_support_same_eligible']['level_rows'][str(i)] for i in range(5)},'OOF support levels')
        ev=reader.json(sm['temporal_prior_evidence']);newlevel=np.full(len(kept),-1,dtype=np.int8);newsupport=np.zeros(len(kept),dtype=np.int64)
        prefix_summaries=[]
        for record in ev['prefixes']:
            take=(qstarts[kept]==record['quarter_start_day'])&(folds[kept]==record['held_fold'])
            c.need(int(take.sum())==record['held_rows'] and (newlevel[take]==-1).all(),'Prefix row coverage')
            state=reader.json(record['saved_state'])
            statehash=hashlib.sha256(json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            c.need(statehash==record['state_content_sha256'] and state['state_role']=='quarter_prefix_for_held_fold','Prefix state identity')
            reference=valid&(days<record['quarter_start_day'])&(folds!=record['held_fold'])
            c.need(int(reference.sum())==state['fit_rows']==record['reference_rows'] and reference.sum()>=30,'Prefix support boundary')
            a,b=route_saved_support(state,[k[kept][take] for k in keys]);newlevel[take]=a;newsupport[take]=b
            names=['|'.join(cols) for cols in prior.LEVELS]+['global']
            c.need({name:int((a==j).sum()) for j,name in enumerate(names)}==record['held_prior_level_counts'],'Temporal route evidence')
            prefix_summaries.append({'quarter':record['quarter'],'held_fold':record['held_fold'],'held_rows':record['held_rows'],
                'saved_reference_date_support':support_date_summary(record['date_support'])})
        c.need((newlevel>=0).all() and (newlevel<=4).all() and (newsupport>0).all(),'All eligible temporal support assigned')
        c.need(level_counts(newlevel)=={str(i):ax['temporal_support_eligible']['level_rows'][str(i)] for i in range(5)},'Temporal support levels')
        query_state=reader.json(f'outputs/round60/{stage}/price_prior_state.json')
        c.need(query_state['state_role']=='full_fit_for_query' and query_state['fit_rows']==nf,'Query full fit state')
        qlevel,qsupport=route_saved_support(query_state,qkeys)
        priorprep=reader.json(f'outputs/round60/{stage}/prior_preparation.json')
        names=['|'.join(cols) for cols in prior.LEVELS]+['global']
        c.need({name:int((qlevel==j).sum()) for j,name in enumerate(names)}==priorprep['query_prior_level_counts'],'Query support routes')
        C,T=prepared['C'],prepared['T'];v=ratio_quantities(C['train_y_million'],C['train_price_factor_million'],T['train_price_factor_million'],C['train_sample_weight'])
        init={}
        for arm in ['C','T']:
            init[arm]=initialize_ratio(prepared[arm]['train_y_million'],prepared[arm]['train_price_factor_million'])
            training=reader.json(sm['arms'][arm]['training'])
            c.need(init[arm]['median']==training['initial_training_median_ratio'] and init[arm]['inverseSoftplus']==training['initial_head_bias'],'Initialization exact replay')
        v.update(prior_unit_C=pc,prior_unit_T=pt,prior_T_over_C=pt/pc,log_prior_T_over_C=np.log(pt/pc),
            support_raw_rows_C=oldsupport,support_raw_rows_T=newsupport)
        c.need(np.allclose(v['prior_T_over_C'],v['factor_T_over_C'],rtol=3e-7,atol=0),'Factor rounding differs excessively')
        effective={f'target_ratio_{a}':v[f'effective_ratio_weight_{a}'] for a in ['C','T']}
        fitstats=grouped_descriptions(v,fit_group_masks(days[kept],start,oldlevel,newlevel),v['date_weight'],effective,v['prior_T_over_C'])
        native={a:reader.array(sm['layers'][a]['native']) for a in ['v22','C','T']}
        F=C['query_price_factor_million'].astype(np.float64)
        qv={'prior_unit':pq,'factor_million':F,'support_raw_rows':qsupport}
        for arm,pred in native.items():positive(pred,nq,np.dtype('float64'));qv['recovered_ratio_'+arm]=pred/(F*1e6)
        qv.update(native_T_over_C=native['T']/native['C'],native_T_over_original=native['T']/native['v22'])
        querystats=grouped_descriptions(qv,query_group_masks(X,Xq,start,qlevel))
        c.verify_hashes(readset);c.need(c.protected_hashes()==protected,'Protected files changed')
        result={'schema_version':1,'round':101,'stage':stage,'status':'descriptive_diagnostics_completed',
            'cutoff':start,'query_end':end,'fit_rows':nf,'eligible_rows':len(kept),'query_rows':nq,
            'parameters':parameters(),'request':c.record(request_path),'source_sha256':source_hashes(),'input_sha256':readset,
            'deserialized_readset':sorted(reader.opened),'protected_before':protected,'protected_after':c.protected_hashes(),
            'initialization_replay':init,'fit':fitstats,'query':querystats,
            'reference_support':{'note':'Raw support rows are not independent projects; ESS is weight concentration only.',
                'C_fold_complements':[{'held_fold':r['fold'],'saved_reference_date_support':support_date_summary(r['date_support'])}
                    for r in priorprep['prior_evidence']['folds']],
                'T_quarter_prefixes':prefix_summaries,'query_full_fit':support_date_summary(query_state['date_support'])},
            'formula_checks':{'C_T_eight_shared_arrays_exact':True,'full_precision_fit_y_to_float32_exact':True,
                'prior_to_saved_float32_factor_exact':True,'common_original_query_factor_exact':True,
                'ratio_target_reciprocal_within_tolerance':True,'float64_algebra_not_trainer_bitwise_claim':True,
                'initial_median_inverseSoftplus_exact':True,'all_support_rows_assigned_once':True},
            'query_labels_read':False,'raw_price_pickle_opened':False,'other_stage_arrays_opened':False,
            'model_loads':0,'model_predict_calls':0,'prior_builds':0,'new_native_fits':0,'MAE_calls':0,'bootstrap_calls':0,
            'row_level_derived_arrays_written':False,'new_CSV':False,'elapsed_seconds':time.monotonic()-tick}
        c.write(out/'results.json',result)
        print(json.dumps({'stage':stage,'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except Exception as error:
        failure={'type':type(error).__name__,'message':str(error)};raise
    finally:
        c.write(out/'execution.json',{'stage':stage,'exit_code':int(failure is not None),'failure':failure,
            'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False,'source_sha256':c.sha(__file__)})

def self_check():
    tick=time.monotonic();checks=[]
    def mark(name,ok):c.need(ok,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name)
    y=np.array([1,3,8,9],dtype=np.float32);fc=np.array([1,2,4,8],dtype=np.float32)
    ft=np.array([2,1,8,4],dtype=np.float32);w=np.array([.1,.5,1.,2.],dtype=np.float32)
    v=ratio_quantities(y,fc,ft,w)
    mark('target_factor_reciprocal',np.allclose(v['target_ratio_T']/v['target_ratio_C'],fc/ft))
    mark('effective_ratio_loss_weights',np.array_equal(v['effective_ratio_weight_T'],w.astype('float64')*ft))
    r=np.array([.2,1.5,2,4.]);lhs=w*(ft*r-y);rhs=(w.astype('float64')*ft)*(r-y.astype('float64')/ft)
    mark('weighted_absolute_loss_algebra',np.allclose(abs(lhs),abs(rhs),rtol=1e-14,atol=1e-14))
    init=initialize_ratio(y,fc);mark('float64_ratio_median',init['median']==float(np.median(y.astype('float64')/fc)))
    mark('inverse_softplus',np.isclose(np.logaddexp(0,init['inverseSoftplus']),init['median']))
    reject('reject_float64_saved_labels',lambda:ratio_quantities(y.astype('float64'),fc,ft,w))
    reject('reject_zero_factor',lambda:ratio_quantities(y,fc*0,ft,w))
    reject('reject_nonfinite_weights',lambda:ratio_quantities(y,fc,ft,w*np.inf))
    cutoff='2023-08-11';days=np.array([day(cutoff)-1,day(cutoff)-364,day(cutoff)-365,day(cutoff)-729,day(cutoff)-730],dtype=float)
    masks=fit_group_masks(days,cutoff,np.array([0,1,2,3,4]),np.array([4,3,2,1,0]))
    mark('age_boundaries_365_730',[int(x.sum()) for x in masks['age_days'].values()]==[2,2,1])
    mark('all_25_transitions_include_empty',len(masks['support_transition'])==25 and sum(x.sum() for x in masks['support_transition'].values())==5)
    reject('reject_future_fit',lambda:fit_group_masks(np.array([day(cutoff)+1.]),cutoff,np.array([0]),np.array([0])))
    ratio=np.array([.899,.9,1.1,1.101]);bands=ratio_bands(ratio,np.ones(4),np.ones(4,dtype=bool))
    mark('prior_band_boundaries',[b['rows'] for b in bands.values()]==[1,2,1])
    keys=[np.array(['a']*30+['b']*30+['z'],dtype=object) for _ in range(4)]
    folds=np.r_[np.zeros(30,dtype=np.int64),np.ones(30,dtype=np.int64),np.array([2],dtype=np.int64)]
    lev,sup=grouped_oof_support(keys,folds,np.arange(61,dtype=np.int64))
    mark('held_fold_support_excludes_all_own_rows',(lev==4).all() and np.array_equal(sup,np.r_[np.full(60,31),60]))
    state={'rules':prior.RULES,'global_raw_rows':99,'levels':[{'columns':cols,'cells':{}} for cols in prior.LEVELS]}
    state['levels'][0]['cells']['a']={'rows':30};state['levels'][1]['cells']['b']={'rows':42}
    qkeys=[np.array(['a','b','z'],dtype=object) for _ in range(4)];l,n=route_saved_support(state,qkeys)
    mark('first_supported_level_and_global',np.array_equal(l,[0,1,4]) and np.array_equal(n,[30,42,99]))
    mark('support_router_needs_no_price',all('median' not in cell for layer in state['levels'] for cell in layer['cells'].values()))
    state['levels'][0]['cells']['a']['rows']=29;reject('reject_subminimum_saved_cell',lambda:route_saved_support(state,qkeys))
    raw=np.array([11,14,20,31],dtype=np.int64);keep=np.array([0,3],dtype=np.int64)
    mark('full_y_masked_prior_alignment',np.array_equal(raw[keep],[11,31]))
    reject('reject_unsorted_indices',lambda:index(np.array([2,1],dtype=np.int64),4))
    reject('reject_reader_foreign_path',lambda:Reader({}).path('data_cache/train.pkl'))
    qf=np.array([1.1,2.2],dtype=np.float32);qr=np.array([.9,1.5]);pred=qr*qf.astype('float64')*1e6
    mark('query_ratio_recovery_tolerance',np.allclose(pred/(qf.astype('float64')*1e6),qr,rtol=1e-15,atol=0))
    vals={'target_ratio_C':v['target_ratio_C']};gs={'fixed':{'all':np.ones(4,dtype=bool),'empty':np.zeros(4,dtype=bool)}}
    desc=grouped_descriptions(vals,gs,v['date_weight'],{'target_ratio_C':v['effective_ratio_weight_C']})
    mark('empty_group_preserved',desc['groups']['fixed']['empty']['rows']==0)
    mark('weighted_ratio_three_views',set(desc['all']['distributions']['target_ratio_C'])=={'unweighted','original_date_weight','effective_ratio_loss_weight'})
    fit=pd.DataFrame({'project_name':['old','early'],'address_city':['A','A'],'address_district':['D','D']})
    query=pd.DataFrame({'project_name':['early','new','__MISSING__'],'address_city':['A']*3,'address_district':['D']*3,
        'trade_day_index':[day(cutoff),day('2023-09-10'),day('2024-08-10')]})
    qm=query_group_masks(fit,query,cutoff,np.array([0,1,4]))
    mark('query_12_relative_months',len(qm['relative_forecast_month'])==12 and qm['relative_forecast_month']['0'].sum()==2 and qm['relative_forecast_month']['11'].sum()==1)
    mark('query_fullfit_membership_not_masked',all(m.sum()==1 for m in qm['project_status'].values()))
    return {'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':c.sha(__file__),
        'source_hashes':source_hashes(),'plan_sha256':PLAN_SHA,'parameters':parameters(),
        'real_arrays_or_labels_read':False,'new_native_fits':0,'new_statistics_on_real_data':False,'elapsed_seconds':time.monotonic()-tick}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request');parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--self-check',action='store_true');parser.add_argument('--receipt')
    args=parser.parse_args()
    if args.self_check:
        result=self_check()
        if args.receipt:c.write(args.receipt,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    else:
        if not args.request or not args.stage:parser.error('--request and --stage required')
        run_stage(args.stage,args.request)
