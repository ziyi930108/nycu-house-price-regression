"""R103: frozen minimal-readset diagnostics; no model load, fitting or bootstrap."""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round100_temporal_prior_evaluate_v2 as old
import round3_evaluate as original_metrics
import round103_complementarity_math as helper

PLAN = 'reports/round103_plan.md'
PLAN_SHA = '9d59b91dbb3cc6dd1adef038a4ea68c307da024fb2fe908b20fde20f03a0955c'
ROOT_SEAL = {'path':'outputs/round102/sealed_manifest.json', 'sha256':'fd75ffd27998d008cedb65a13632e48c480f05beb9c69ee9bccf5b2e7a1aa68a'}
EVALUATION = {'path':'outputs/round102/evaluation/results.json', 'sha256':'e385ed630cdf3a2b2b0418867563a4064fbe08c1c979feb004787df2592f2347'}
STAGES = c.STAGES
ARMS = ('v22','C','T')
LAYERS = ('native','base','reference','spatial')
INDEX_NAMES = ('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices')
PROTECTED_LIST = 'outputs/github_publish_r97_20260929/protected_before.json'
CAUTION = ('Post-hoc diagnostics on repeatedly used query windows, not blind validation or causal effects. '
           'No weights learned or predictions changed; Private unknown; only v22 selected.')


def parameters():
    return {'round':103,'stage_order':list(STAGES),'arms':list(ARMS),'comparisons':['T-v22','T-C'],
            'saved_prediction_arrays_per_stage':19,'index_arrays_per_stage':4,
            'families':['all','status','city','relative_quarter','price_band'],
            'relative_quarter_horizon_days':original_metrics.QUARTERS,'helper_parameters':helper.parameters(),
            'fixed_status_weights':c.WEIGHTS,'maximum_identity_residual_TWD':1e-5,'maximum_MAE_replay_difference_TWD':1e-6,
            'new_fits':0,'model_loads':0,'prior_builds':0,'new_bootstrap':0,'new_predictions':0,'new_CSV':False}


def source_hashes():
    return c.source_hashes(__file__)


def check_source_reuse(current,original):
    overlap={name:digest for name,digest in current.items() if name in original}
    c.need(bool(overlap) and all(original[name]==digest for name,digest in overlap.items()),'Reused R102 formula/group source changed')
    return overlap


def stage_records(sm):
    """An explicit readset; never follow model/prepared/state artifact maps."""
    c.need(set(sm['fixed_branches'])=={'area','lgb','H','U'} and set(sm['layers'])==set(ARMS),'Branch schema')
    records = list(sm['fixed_branches'].values())
    for arm in ARMS:
        c.need(set(sm['layers'][arm])==set(LAYERS),'Five-layer source schema')
        records.extend(sm['layers'][arm][key] for key in LAYERS)
        records.append(sm['predictions'][arm])
    records.extend(sm[key] for key in INDEX_NAMES)
    c.need(len(records)==23 and len({r['path'] for r in records})==23,'Exactly 19 prediction and 4 index arrays')
    for rec in records:
        c.need(rec['path'].startswith(f"outputs/round102/{sm['stage']}/") and Path(rec['path']).suffix=='.npy', 'Wrong-stage or forbidden array path')
        c.need(set(rec)=={'path','sha256','dtype','shape','values_sha256'},'Complete array identity required')
    return records


def freeze_request(path):
    """Read source/JSON and stream file hashes; do not deserialize arrays or labels."""
    target=c.safe(path);c.need(not target.exists() and not target.with_suffix('.sha256').exists(),'Request exists')
    tick=time.monotonic();inputs={}
    def bind(rec):
        name=c.rel(rec['path']);c.need(name not in inputs or inputs[name]==rec['sha256'],'Conflicting input SHA')
        c.need(c.sha(name)==rec['sha256'],'Source artifact differs: '+name);inputs[name]=rec['sha256'];return name
    def doc(rec):return c.read(bind(rec))
    c.need(c.sha(PLAN)==PLAN_SHA,'Fixed R103 plan differs')
    main=doc(ROOT_SEAL);evaluation=doc(EVALUATION)
    c.need(main['status']=='all_prior_input_predictions_sealed' and main['stage_order']==list(STAGES)
           and main['new_NN_fits']==6 and main['all_query_reload_exact'] is True,'Incomplete R102 seal')
    c.need(evaluation['status']=='completed_three_window_six_fit_evaluation' and evaluation['checked_stages']==list(STAGES)
           and evaluation['sealed_manifest']==ROOT_SEAL,'R102 evaluation authority differs')
    sources=source_hashes();reused_sources=check_source_reuse(sources,evaluation['source_sha256'])
    stages={};global_records={k:main[k] for k in ('features','canonical_mapping','raw_train')}
    for rec in global_records.values():bind(rec)
    for name in ('outputs/round102/execution.json','outputs/round102/evaluation/execution.json'):
        ex=doc({'path':name,'sha256':evaluation['input_sha256'][name] if name in evaluation['input_sha256'] else c.sha(name)})
        c.need(ex['exit_code']==0 and ex.get('failure') is None,'R102 process incomplete')
    for stage in STAGES:
        sr=main['stages'][stage];sm=doc(sr['stage_manifest']);ex=doc(sr['execution'])
        c.need(ex['exit_code']==0 and ex.get('failure') is None and sm['stage']==stage
               and sm['status']=='prior_input_stage_predictions_sealed' and sm['all_query_reload_exact'] is True,'Stage incomplete')
        start,end,nf,nq=c.SPLITS[stage]
        c.need((sm['cutoff'],sm['query_end'],sm['fit_rows'],sm['query_rows'])==(start,end,nf,nq),'Original window differs')
        c.need(sm['predictions']==sr['predictions'] and sm['protected_before']==sm['protected_after']==main['protected_after'],'Stage control identity')
        for key in global_records:c.need(sm[key]==global_records[key],'Shared raw/mapping/feature identity')
        records=stage_records(sm)
        for rec in records:
            bind(rec);c.need(sm['artifact_sha256'].get(rec['path'])==rec['sha256'],'Selected file not in original artifact seal')
        stages[stage]={'stage_manifest':sr['stage_manifest'],'execution':sr['execution'],
            'readset':{r['path']:r['sha256'] for r in records},'query_target_values_sha256':evaluation['stages'][stage]['query_target_values_sha256']}
    bind(c.record(PROTECTED_LIST));protected=c.protected_hashes()
    c.need(protected==main['protected_before']==main['protected_after']==evaluation['protected_after'],'Protected identity differs')
    checks={}
    for name,expected in [('reports/round103_math_selfcheck.json',helper.source_hashes()),('reports/round103_driver_selfcheck.json',c.sha(__file__))]:
        rec=c.record(name);r=doc(rec);c.need(r['status']=='passed' and r['source_sha256']==expected,'Selfcheck is not current source')
        checks[name]=rec['sha256']
    request={'schema_version':1,'status':'frozen_before_R103_row_array_or_query_label_deserialization','created_at_utc':c.now(),
        'plan':{'path':PLAN,'sha256':PLAN_SHA},'parameters':parameters(),'source_sha256':sources,'reused_R102_source_sha256':reused_sources,
        'input_sha256':inputs,'protected_sha256':protected,'root_seal':ROOT_SEAL,'evaluation':EVALUATION,
        'global_records':global_records,'stages':stages,'selfcheck_sha256':checks,
        'arrays_labels_models_deserialized':False,'file_hashing_includes_raw_pickle_bytes':True,'elapsed_seconds':time.monotonic()-tick}
    target.parent.mkdir(parents=True,exist_ok=True);c.write(target,request)
    return request


def groups_and_weights(fit,query,y,start):
    n=len(y);c.need(len(query)==n,'Group row axis differs')
    labels=old.methods.group_labels(fit,query);w=np.zeros(n,dtype=np.float64)
    groups=[('all','all',np.ones(n,dtype=bool))]
    for name,mass in c.WEIGHTS.items():
        mask=labels==name;count=int(mask.sum());c.need(mass==0 or count>0,'Positive-weight stratum absent')
        if count:w[mask]=mass/count
        groups.append(('status',name,mask))
    city=old.methods.norm(query['address_city']).to_numpy()
    for name in sorted(set(city)|{'__MISSING__'}):groups.append(('city',name,city==name))
    horizon=query.trade_day_index.to_numpy(dtype=np.float64)-old.day(start)
    c.need(np.isfinite(horizon).all() and (horizon>=0).all(),'Original forecast horizon')
    for name,(low,high) in original_metrics.QUARTERS.items():
        mask=horizon>=low
        if high is not None:mask &= horizon<=high
        groups.append(('relative_quarter',name,mask))
    for name,low,high in old.BANDS:groups.append(('price_band',name,(y>=low)&(y<high)))
    for family in parameters()['families']:
        masks=[mask for f,_,mask in groups if f==family]
        c.need(np.array_equal(np.sum(masks,axis=0),np.ones(n,dtype=int)),'Fixed family is not a partition: '+family)
    c.need(np.isfinite(w).all() and (w>=0).all() and abs(float(w.sum())-1)<1e-12,'Fixed row weights do not sum to one')
    return groups,w


def replay_metrics(analysis,original):
    maximum=0.
    c.need(set(analysis['metrics'])==set(ARMS),'Helper MAE schema')
    for arm in ARMS:
        c.need(set(analysis['metrics'][arm])==set((*LAYERS,'median')),'Helper must report all five layers')
        for layer in (*LAYERS,'median'):
            expected=original['metrics'][arm] if layer=='median' else original['layer_metrics'][arm][layer]
            for key in ('overall_mae','weighted_mae'):
                actual=analysis['metrics'][arm][layer][key];difference=abs(actual-expected[key])
                c.need(np.isfinite(actual) and difference<=1e-6,f'Existing MAE replay differs: {arm}/{layer}/{key}')
                maximum=max(maximum,difference)
    return {'all_30_MAE_values_within_1e_6':True,'maximum_absolute_difference_TWD':maximum}


def verify_request(request):
    c.need(request['schema_version']==1 and request['status']=='frozen_before_R103_row_array_or_query_label_deserialization','Not a frozen diagnostic request')
    c.need(request['plan']=={'path':PLAN,'sha256':PLAN_SHA} and c.sha(PLAN)==PLAN_SHA,'Plan changed')
    c.need(request['parameters']==parameters() and request['source_sha256']==source_hashes(),'Source/parameters changed')
    c.need(request['root_seal']==ROOT_SEAL and request['evaluation']==EVALUATION,'Fixed authorities changed')
    c.verify_hashes(request['input_sha256']);c.verify_hashes(request['source_sha256'])
    c.need(c.protected_hashes()==request['protected_sha256'],'Protected files changed')


def run(request_path,outpath):
    out=c.safe(outpath);c.need(out==c.safe('outputs/round103') and not out.exists(),'New fixed output directory required')
    out.mkdir(parents=True);tick=time.monotonic();failure=None;raw_opened=False;loaded=[]
    try:
        c.need(c.safe(request_path).with_suffix('.sha256').read_text().strip()==c.sha(request_path),'Request sidecar changed')
        request=c.read(request_path);verify_request(request)
        def doc(rec):
            c.need(request['input_sha256'].get(rec['path'])==rec['sha256'],'JSON outside frozen readset');return c.read(rec['path'])
        main=doc(ROOT_SEAL);original=doc(EVALUATION)
        frames=request['global_records'];c.need(frames=={k:main[k] for k in ('features','canonical_mapping','raw_train')},'Shared dataset records differ')
        mapping=np.load(c.safe(frames['canonical_mapping']['path']),allow_pickle=False)
        loaded.append(frames['canonical_mapping']['path']);old.index(mapping,455784,'canonical mapping');c.need(mapping.shape==(455255,),'Canonical count')
        Xraw=pd.read_pickle(c.safe(frames['features']['path']))
        c.need(len(Xraw)==455784 and Xraw.index.equals(pd.RangeIndex(455784)) and '總價元' not in Xraw,'Pure X schema')
        days=Xraw.iloc[mapping].trade_day_index.to_numpy(dtype=np.float64)
        # The raw table is fully deserialized, but only fixed query rows feed analysis.
        raw=pd.read_pickle(c.safe(frames['raw_train']['path']));raw_opened=True;stages={}
        for stage in STAGES:
            st=time.monotonic();manifest=request['stages'][stage];sm=doc(manifest['stage_manifest'])
            c.need(manifest['stage_manifest']==main['stages'][stage]['stage_manifest'],'Original stage authority differs')
            expected={r['path']:r['sha256'] for r in stage_records(sm)}
            c.need(expected==manifest['readset'],'Stage readset changed')
            def array(rec):
                c.need(manifest['readset'].get(rec['path'])==rec['sha256'],'Array outside exact same-stage readset')
                a=c.load_array(rec);loaded.append(rec['path']);return a
            ix={name:array(sm[name]) for name in INDEX_NAMES};fi,qi=ix['fit_indices'],ix['query_indices']
            old.check_temporal(stage,fi,qi,ix['query_source_cache_indices'],days,mapping)
            c.exact(mapping[fi],ix['fit_source_cache_indices'],'Raw fit mapping changed')
            fixed={name:array(rec) for name,rec in sm['fixed_branches'].items()}
            layers={arm:{key:array(sm['layers'][arm][key]) for key in LAYERS} for arm in ARMS}
            predictions={arm:array(sm['predictions'][arm]) for arm in ARMS}
            for a in [*fixed.values(),*[v for arm in layers.values() for v in arm.values()],*predictions.values()]:old.prices(a,(len(qi),),'Saved prediction')
            old.check_formulas(predictions,layers,fixed)
            y=old.select_query_y(raw,ix['query_source_cache_indices'])
            c.need(c.values_sha(y)==manifest['query_target_values_sha256']==original['stages'][stage]['query_target_values_sha256'],'Original query target identity differs')
            fit=Xraw.iloc[mapping[fi]].reset_index(drop=True);query=Xraw.iloc[mapping[qi]].reset_index(drop=True)
            groups,weights=groups_and_weights(fit,query,y,c.SPLITS[stage][0])
            analysis=helper.analyze(y,fixed,layers,predictions,groups,weights)
            replay=replay_metrics(analysis,original['stages'][stage])
            result={'round':103,'stage':stage,'status':'completed_saved_prediction_diagnostics',
                'cutoff':c.SPLITS[stage][0],'query_end':c.SPLITS[stage][1],'fit_rows':len(fi),'query_rows':len(qi),
                'query_target_values_sha256':c.values_sha(y),'query_target_values_persisted':False,
                'full_outer_fit_defines_membership':True,'row_weights_sum':float(weights.sum()),
                'source_stage_manifest':manifest['stage_manifest'],'input_sha256':manifest['readset'],
                'MAE_replay':replay,'analysis':analysis,'elapsed_seconds':time.monotonic()-st}
            folder=out/stage;folder.mkdir();c.write(folder/'results.json',result)
            stages[stage]={'results':c.record(folder/'results.json'),'elapsed_seconds':result['elapsed_seconds'],'MAE_replay':replay}
            print(json.dumps({'stage':stage,'status':'completed','elapsed_seconds':result['elapsed_seconds']}),flush=True)
        del raw,Xraw
        c.need(len(loaded)==70 and len(set(loaded))==70,'Expected canonical plus 3x23 array reads')
        verify_request(request)
        result={'round':103,'status':'three_window_complementarity_diagnostics_completed','stage_order':list(STAGES),'stages':stages,
            'request':c.record(request_path),'parameters':parameters(),'source_sha256':source_hashes(),'input_sha256':request['input_sha256'],
            'protected_before':request['protected_sha256'],'protected_after':c.protected_hashes(),
            'row_arrays_loaded':70,'array_paths_loaded':loaded,'raw_pickle_whole_table_deserialized':True,
            'query_target_values_persisted':False,'new_models':0,'new_predictions':0,'new_bootstrap':0,'new_CSV':False,
            'selected_submission':'v22','Private_score':None,'caution':CAUTION,'elapsed_seconds':time.monotonic()-tick}
        c.write(out/'results.json',result);return result
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        c.write(out/'execution.json',{'exit_code':int(failure is not None),'failure':failure,'elapsed_seconds':time.monotonic()-tick,
            'raw_prices_deserialized':raw_opened,'source_sha256':c.sha(__file__),'automatic_retry':False,
            'new_fit_model_load_predict_prior_bootstrap_calls':0,'saved_row_arrays_loaded':len(loaded)})


def self_check():
    tick=time.monotonic();checks=[]
    def mark(name,ok):c.need(bool(ok),name);checks.append(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError):checks.append(name)
        else:raise AssertionError('Accepted '+name)
    start='2022-08-11';dates=pd.Timestamp(start)+pd.to_timedelta([0,92,184,276,91],unit='D')
    fit=pd.DataFrame({'project_name':['known'],'address_city':['A'],'address_district':['D']})
    query=pd.DataFrame({'project_name':['known','new','new','new',None],'address_city':['A','B','B','B',None],
        'address_district':['D']*5,'trade_day_index':(dates-pd.Timestamp('2000-01-01')).days})
    y=np.array([5e6,1e7,2e7,5e7,1e8],dtype=np.float64);groups,w=groups_and_weights(fit,query,y,start)
    mark('fixed_weight_seen_new_missing',w[0]==c.WEIGHTS['seen'] and w[-1]==0 and np.all(w[1:4]==c.WEIGHTS['new_nonmissing']/3))
    mark('all_fixed_families',set(f for f,_,_ in groups)==set(parameters()['families']))
    mark('missing_city_separate',next(m for f,n,m in groups if f=='city' and n=='__MISSING__').sum()==1)
    mark('five_price_boundaries',all(m.sum()==1 for f,_,m in groups if f=='price_band'))
    mark('relative_quarter_boundaries',[int(m.sum()) for f,_,m in groups if f=='relative_quarter']==[2,1,1,1])
    boundary=query.copy();boundary.loc[0,'trade_day_index']=old.day('2023-05-11')
    boundary_groups,_=groups_and_weights(fit,boundary,y,start)
    mark('day_quarter_not_month_quarter',next(m for f,n,m in boundary_groups if f=='relative_quarter' and n=='Q3')[0]
         and old.methods.relative_months(boundary.trade_day_index.to_numpy(dtype=np.float64),start)[0]//3==3)
    rejects('positive_weight_group_absent',lambda:groups_and_weights(fit.iloc[:0],query,y,start))
    sm={'stage':'development','fixed_branches':{},'layers':{a:{} for a in ARMS},'predictions':{}}
    def rec(path):return {'path':'outputs/round102/development/'+path+'.npy','sha256':'a'*64,'dtype':'float64','shape':[5],'values_sha256':'b'*64}
    sm['fixed_branches']={k:rec(k) for k in ('area','lgb','H','U')}
    for a in ARMS:
        sm['layers'][a]={key:rec(a+'/'+key) for key in LAYERS};sm['predictions'][a]=rec('prediction_'+a)
    sm.update({k:rec(k) for k in INDEX_NAMES});mark('exact_23_record_readset',len(stage_records(sm))==23)
    sm['layers']['T']['base']['path']='outputs/round102/confirmation/T/base.npy'
    rejects('wrong_stage_vector',lambda:stage_records(sm))
    metrics={a:{layer:{'overall_mae':1.,'weighted_mae':2.} for layer in (*LAYERS,'median')} for a in ARMS}
    original={'metrics':{a:metrics[a]['median'] for a in ARMS},'layer_metrics':{a:{l:metrics[a][l] for l in LAYERS} for a in ARMS}}
    mark('thirty_MAE_replay_values',replay_metrics({'metrics':metrics},original)['all_30_MAE_values_within_1e_6'])
    altered=json.loads(json.dumps(metrics));altered['T']['base']['weighted_mae']+=.01
    rejects('weighted_MAE_mismatch',lambda:replay_metrics({'metrics':altered},original))
    bad=np.array([2,1],dtype=np.int64);rejects('unordered_query_mapping',lambda:old.index(bad,5,'synthetic'))
    selected=old.select_query_y(pd.DataFrame({'總價元':[1.,2.,3.,4.]}),np.array([1,3],dtype=np.int64),4)
    mark('raw_source_index_order',np.array_equal(selected,[2.,4.]))
    mark('overlap_source_identity_accept',check_source_reuse({'old.py':'a','new.py':'b'},{'old.py':'a','unused.py':'c'})=={'old.py':'a'})
    rejects('reused_source_drift',lambda:check_source_reuse({'old.py':'b'},{'old.py':'a'}))
    fixed={'area':y+20.,'lgb':y-15.,'H':y-30.,'U':y+30.};layers={};predictions={}
    for arm,delta in [('v22',20.),('C',30.),('T',10.)]:
        native=y+delta;base=.75*(.5*fixed['area']+.5*native)+.25*fixed['lgb']
        layers[arm]={'native':native,'base':base,'reference':base-1.,'spatial':base-2.}
        predictions[arm]=np.median(np.column_stack([base-2.,fixed['H'],fixed['U']]),axis=1)
    analysis=helper.analyze(y,fixed,layers,predictions,groups,w)
    mark('actual_helper_API_all_5_families',set(g['family'] for g in analysis['comparisons']['T-v22']['groups'])==set(parameters()['families']))
    mark('actual_helper_retains_36_routes_and_9_cells',all(len(a['median_routes'])==36 and len(a['error_sign_transitions'])==9 for a in analysis['comparisons'].values()))
    return {'status':'passed','source_sha256':c.sha(__file__),'plan_sha256':PLAN_SHA,'parameters':parameters(),
        'checks':checks,'check_count':len(checks),'real_arrays_labels_models_read':False,'real_fit_score_calls':0,
        'elapsed_seconds':time.monotonic()-tick}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-request');mode.add_argument('--run',action='store_true');mode.add_argument('--self-check',action='store_true')
    p.add_argument('--request');p.add_argument('--out',default='outputs/round103');p.add_argument('--receipt');a=p.parse_args()
    if a.freeze_request:result=freeze_request(a.freeze_request)
    elif a.self_check:
        result=self_check()
        if a.receipt:c.write(a.receipt,result)
    else:
        if not a.request:p.error('--request required')
        result=run(a.request,a.out)
    print(json.dumps({'status':result['status'],'source_sha256':c.sha(__file__)},ensure_ascii=False))
