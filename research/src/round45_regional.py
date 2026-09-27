"""R45 fixed R3 regional reference in three existing R39 spatial branches."""
from __future__ import annotations
import argparse
import contextlib
import json
import pickle
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round43_median as history
import round3_reference as regional
base=history.prior
geo=base.previous
original=geo.prior.log_run.original
ROOT=history.ROOT
OUT=ROOT/'outputs/round45'
DOCUMENT=ROOT/'reports/round45_regional_plan.md'
STAGES=history.STAGES
CANDIDATE='regional_three_branch_median'
FORMULA='median(R8(saved_R39_state,newgeo,B_branch,R3.regional(reference,B_branch),saved_diag_branch) for raw/log/unit)'
AUTHORITIES={
 'outputs/round43/plan.json':'6f42a9f5c5e57c2a25d9c4c5668007c0f824d6827dbaa9b9dd2939c16bff07c5',
 'outputs/round43/results.json':'6b5f0ea14752b9c70b1c6e89df96f4ebd7ac083054518f0d3485451636bd5387',
 'outputs/round40/plan.json':'628b3c94854e02fb5d13ce49ac92f45234f91482a66479dd5cf052df927d1a19',
 'outputs/round40/results.json':'aeb5e2f7d65802ace77d765848725032d89bcd326a3e45a5026627e2cc9cd9b1',
 'outputs/round39/evaluation/plan.json':'110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13',
 'outputs/round39/evaluation/results.json':'1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27'}
PROTECTED=history.PROTECTED+('outputs/submission_v9_trial.csv',)
need,sha,rel,locked,write,merge,now=history.need,history.sha,history.rel,history.locked,history.write,history.merge,history.now


def sources():
    return merge(history.sources(),{rel(Path(__file__)):sha(Path(__file__)),'src/round3_reference.py':sha(ROOT/'src/round3_reference.py')})


def freeze():
    inputs={rel(DOCUMENT):sha(DOCUMENT)}
    def small(name,expected=None):return geo.prior.small(inputs,ROOT/name,expected,True)
    docs={n:small(n,d) for n,d in AUTHORITIES.items()}
    p43,r43,p40,r40,p39,r39=[docs[n] for n in AUTHORITIES]
    need(history.sources()==p43['source_sha256'] and r43['checked_stages']==list(STAGES),'R43 source/periods changed')
    need(p43['split_contract']==p40['split_contract']==p39['split_contract']
         and p43['target_group_weights']==p40['target_group_weights']==p39['target_group_weights'],'Split/weights differ')
    inputs=merge(inputs,p43['input_sha256'])
    def bind(name,digest):inputs.update(merge(inputs,{name:digest}))
    bindings={}
    for stage in STAGES:
        r=small(f'outputs/round43/{stage}/results.json',r43['stage_result_sha256'][stage])
        a=small(f'outputs/round40/{stage}/results.json',r40['stage_result_sha256'][stage])
        b=small(f'outputs/round39/evaluation/{stage}/results.json',r39['stage_result_sha256'][stage])
        need(r==r43['stages'][stage] and a==r40['stages'][stage],'Stage summary differs')
        spec=dict(p43['bindings'][stage]);spec['expected_metrics']={'v5':r['metrics']['v5'],'v9':r['metrics']['three_branch_median']}
        name=f'outputs/round43/{stage}/predictions.npy';spec['v9']=name;bind(name,r['artifact_sha256'][name])
        b40=p40['bindings'][stage]
        for role in ('old_raw','old_unit','unit_component','reference','state','state_summary','query_geo'):
            name=b40[role];spec[role]=name;bind(name,p40['input_sha256'][name])
        spec['manifest']=b40['manifest']
        raw=f'outputs/round6/models/{stage}/geography_catboost/predictions.npy';spec['raw_component']=raw;bind(raw,p39['input_sha256'][raw])
        log=p39['bindings'][stage]['log_component'];spec['log_component']=log;bind(log,p39['input_sha256'][log])
        for branch in ('raw','log'):
            name=f'outputs/round39/evaluation/{stage}/{branch}_diagnostics.pkl';spec[branch+'_diag']=name;bind(name,b['artifact_sha256'][name])
        name=f'outputs/round40/{stage}/unit_reference_diagnostics.pkl';spec['unit_diag']=name;bind(name,a['artifact_sha256'][name])
        bindings[stage]=spec
    old_source='src/round3_reference.py'
    need(sha(ROOT/old_source)==p40['source_sha256'][old_source],'Original R3 regional source changed')
    src=sources();geo.prior.c6.verify_hashes(merge(inputs,src))
    value={'round':45,'created_at_utc':now(),'candidate':CANDIDATE,'formula':FORMULA,'stage_order':list(STAGES),
        'source_sha256':src,'input_sha256':inputs,'bindings':bindings,'split_contract':p43['split_contract'],
        'splits':p43['splits'],'target_group_weights':p43['target_group_weights'],'regional_parameters':regional.REGIONAL_PARAMETERS,
        'spatial_parameters':base.spatial.parameters(),'runtime':geo.average.runtime(),
        'regional_state_fit_rows_only':True,'query_y_only_in_scorer':True,'existing_spatial_global_beta_unchanged':True,
        'saved_diagnostics_reused_source_and_confidence_only':True,'all_three_periods_regardless_of_scores':True,
        'new_tree_fits':0,'automatic_release_allowed':False,'new_threshold_or_gate':False,
        'protected_sha256':{n:sha(ROOT/n) for n in PROTECTED}}
    need(src==sources() and inputs[rel(DOCUMENT)]==sha(DOCUMENT),'Source changed during freeze')
    write(OUT/'plan.json',value);print('R45_frozen',sha(OUT/'plan.json'),flush=True)


def load(path):
    with (ROOT/path).open('rb') as f:return pickle.load(f)


def evaluate(stage,p,Xall,yall):
    start=time.monotonic();b=p['bindings'][stage];spec=p['split_contract'][stage];folder=OUT/stage;folder.mkdir(exist_ok=False)
    fi,qi=[np.load(ROOT/b[n+'_indices'],allow_pickle=False) for n in ('fit','query')]
    for n,a in (('fit',fi),('query',qi)):
        need(a.dtype==np.dtype('int64') and a.shape==(spec[n+'_rows'],) and (np.diff(a)>0).all() and a.min()>=0 and a.max()<len(Xall)
             and original.value_sha(a,'<i8')==spec[n+'_indices_content_sha256'],'Canonical indices differ')
    need(not np.intersect1d(fi,qi).size,'Fit/query overlap')
    X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
    hashes={'fit_X':base.spatial.frame_sha(X),'query_X':base.spatial.frame_sha(Xq)}
    need(hashes==b['input_frame_sha256'],'X/frame alignment differs')
    vectors={n:base.vector(ROOT/b[n],len(qi)) for n in ('raw_spatial','log_spatial','unit_spatial','v5','v9')}
    oldmedian=np.median(np.column_stack([vectors[n] for n in ('raw_spatial','log_spatial','unit_spatial')]),axis=1)
    base.exact(oldmedian,vectors['v9'],'Full v9 median control')
    score=lambda v:original.score(yall[qi],v,X,Xq,p['splits'][stage][0],p['target_group_weights'])
    metrics={n:score(vectors[n]) for n in ('v5','v9')}
    need(metrics==b['expected_metrics'],'Full v5/v9 control metrics differ')
    reference=load(b['reference']);state=load(b['state']);summary=locked(ROOT/b['state_summary'])
    need(reference['fit_report']['fitting_rows']==len(X)
         and reference['fit_report']['hedonic']['maximum_fit_day_index']==float(np.nanmax(X.trade_day_index)), 'Reference fit boundary differs')
    need(state['state_summary']==summary and summary['fit_X_sha256']==hashes['fit_X']
         and summary['fit_geo_sha256']==b['manifest']['features']['fit']['frame_sha256']
         and state['parameters']==p['spatial_parameters']
         and original.value_sha(reference['reference_residual'],'<f8')==summary['existing_reference_residual_sha256'], 'Saved spatial state differs')
    geoq=pd.read_pickle(ROOT/b['query_geo'])
    need(geoq.index.equals(Xq.index) and base.spatial.frame_sha(geoq)==b['manifest']['features']['query']['frame_sha256'],'New query geography differs')
    diagnostics={}
    for branch in ('raw','log','unit'):
        diag=load(b[branch+'_diag'])
        if branch!='unit':diag=diag[['v2_source','v2_confidence']].rename(columns={'v2_source':'source','v2_confidence':'confidence'})
        else:diag=diag[['source','confidence']]
        need(diag.index.equals(Xq.index) and len(diag)==len(qi) and diag['source'].notna().all()
             and np.isfinite(diag['confidence']).all() and diag['confidence'].between(0,1).all(),'Invalid saved diagnostic alignment')
        diagnostics[branch]=diag
    need(diagnostics['raw'].equals(diagnostics['log']) and diagnostics['raw'].equals(diagnostics['unit']),
         'Source/confidence must be branch-independent for the same reference/X')
    t=time.monotonic();reg=regional.fit_regional_trend(X,yall[fi],reference);regional_seconds=time.monotonic()-t
    need(reg['training_rows']==len(fi) and reg['parameters']==p['regional_parameters']
         and reg['fit_targets_sha256']==original.value_sha(yall[fi],'<f8')
         and reg['constraints']['fit_inputs_only'] is True and reg['constraints']['query_labels_used'] is False,'Regional fit contract differs')
    write(folder/'regional_model.json',reg)
    old,oldunit=[base.vector(ROOT/b[n],len(qi)) for n in ('old_raw','old_unit')]
    predictions=[];reports={};branch_metrics={}
    for branch in ('raw','log','unit'):
        component=base.vector(ROOT/b[branch+'_component'],len(qi))
        B=.75*(.5*old+.5*component)+.25*oldunit
        preg=regional.predict_regional_trend(reference,reg,Xq,B)
        # R3 internally uses anchor plus ONE regional future-baseline correction.
        # R8's own saved global-beta Bfuture normalization remains unchanged.
        pred,detail,report=base.spatial.predict(state,Xq,geoq,B,preg,diagnostics[branch])
        fallback=~detail['applied'].to_numpy(dtype=bool);base.exact(pred[fallback],preg[fallback],branch+' fallback')
        geo.save_array(folder/(branch+'_regional_reference.npy'),preg);geo.save_array(folder/(branch+'_regional_spatial.npy'),pred)
        predictions.append(pred);reports[branch]=report;branch_metrics[branch]=score(pred)
    candidate=np.median(np.column_stack(predictions),axis=1)
    need(candidate.shape==(len(qi),) and np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid regional median')
    metrics[CANDIDATE]=score(candidate);geo.save_array(folder/'predictions.npy',candidate)
    result={'round':45,'stage':stage,'status':'completed','plan_sha256':sha(OUT/'plan.json'),'fit_rows':len(fi),'query_rows':len(qi),
        'dates':p['splits'][stage],'metrics':metrics,'branch_metrics':branch_metrics,
        'candidate_vs_control':{n:geo.prior.log_run.differences(metrics[n],metrics[CANDIDATE]) for n in ('v5','v9')},
        'control_full_metrics_exact':True,'v9_median_full_vector_bitwise_equal':True,'input_frame_sha256':hashes,
        'regional_fit_seconds':regional_seconds,'regional_fits':1,'query_y_only_in_scorer':True,
        'regional_reference_predict_calls':3,'spatial_predict_calls':3,'saved_diag_source_confidence_equal':True,
        'existing_R8_global_beta_and_Bfuture_unchanged':True,'tree_model_calls':0,'reports':reports,
        'regional_supported_cities':[k for k,v in reg['cities'].items() if v['supported']],
        'elapsed_seconds':time.monotonic()-start,'source_sha256':p['source_sha256'],
        'artifact_sha256':{rel(f):sha(f) for f in sorted(folder.iterdir()) if f.is_file()}}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'metrics':{n:{k:m[k] for k in ('overall_mae','weighted_mae')} for n,m in metrics.items()}}),flush=True)
    return result


def run():
    p=locked(OUT/'plan.json')
    need(p['round']==45 and p['formula']==FORMULA and p['stage_order']==list(STAGES) and p['source_sha256']==sources()
         and p['regional_parameters']==regional.REGIONAL_PARAMETERS and p['spatial_parameters']==base.spatial.parameters()
         and p['runtime']==geo.average.runtime(),'Frozen contract changed')
    need(not (OUT/'run_claim.json').exists() and not (OUT/'results.json').exists()
         and not any((OUT/s).exists() for s in STAGES),'Previous attempt exists')
    write(OUT/'run_claim.json',{'started_at_utc':now(),'plan_sha256':sha(OUT/'plan.json'),'automatic_retry':False})
    t=time.monotonic();result={'round':45,'status':'incomplete','checked_stages':[],'stages':{},'stage_result_sha256':{},
      'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],'plan_sha256':sha(OUT/'plan.json'),
      'protected_before':p['protected_sha256'],'formal_adoption_decision':None,'new_submission_generated':False,
      'tree_models_trained_loaded_or_predicted':0,'fresh_blind_validation':False,'old_R3_decision_changed':False,
      'label_scope':'Raw train cache deserialized; canonical fi y only to fixed R3 regional fit; fixed qi y only to scorer.'}
    try:
        geo.prior.c6.verify_hashes(merge(p['input_sha256'],p['source_sha256'],p['protected_sha256']))
        Xall,yall=geo.load_training_rows(p)
        for stage in STAGES:
            item=evaluate(stage,p,Xall,yall);result['stages'][stage]=item;result['checked_stages'].append(stage)
            result['stage_result_sha256'][stage]=sha(OUT/stage/'results.json')
        geo.prior.c6.verify_hashes(merge(p['source_sha256'],p['protected_sha256']))
        result['status']='three_periods_completed_no_adoption_decision'
    except Exception:result['status']='incomplete_technical_failure';result['failure']=traceback.format_exc();print(result['failure'],flush=True)
    result.update(elapsed_seconds=time.monotonic()-t,completed_at_utc=now(),protected_after={n:sha(ROOT/n) for n in PROTECTED})
    write(OUT/'results.json',result);print('R45_result',result['status'],sha(OUT/'results.json'),flush=True)
    return 0 if result['status']=='three_periods_completed_no_adoption_decision' else 1


def execute(mode):
    if mode=='freeze':need(not OUT.exists(),'Existing R45 output');OUT.mkdir(parents=True,exist_ok=False)
    else:need(OUT.is_dir(),'Freeze required')
    started=now();t=time.monotonic();code=1;error=None;log=OUT/(mode+'.log')
    with log.open('x',encoding='utf8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:
                if mode=='freeze':freeze();code=0
                else:code=run()
            except Exception:error=traceback.format_exc();print(error,flush=True)
    event={'round':45,'command':sys.argv,'started_at_utc':started,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-t,
        'exit_code':code,'source_sha256':sha(Path(__file__)),'log_path':rel(log),'log_sha256':sha(log),'failure':error,'automatic_retry':False}
    write(OUT/(mode+'_execution.json'),event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    raise SystemExit(execute(parser.parse_args().command))
