"""Fixed R43 row-wise median of three saved TWD spatial branches; no model calls."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import round40_three_branch as prior

ROOT=prior.ROOT
OUT=ROOT/'outputs/round43'
DOCUMENT=ROOT/'reports/round43_median_plan.md'
REPORT=ROOT/'reports/round43_median_report.md'
STAGES=('development','confirmation','auxiliary')
CANDIDATE='three_branch_median'
CONTROLS=('v5','v7','R40_mean')
FORMULA='np.median(np.column_stack([saved_R39_raw_spatial_TWD,saved_R39_log_spatial_TWD,saved_R40_unit_spatial_TWD]),axis=1)'
AUTHORITIES={
 'outputs/round40/plan.json':'628b3c94854e02fb5d13ce49ac92f45234f91482a66479dd5cf052df927d1a19',
 'outputs/round40/results.json':'aeb5e2f7d65802ace77d765848725032d89bcd326a3e45a5026627e2cc9cd9b1',
 'outputs/round39/evaluation/plan.json':'110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13',
 'outputs/round39/evaluation/results.json':'1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27',
 'outputs/round36/results.json':'133d99e9fdca21ddf97e58fa8c7860ac74f7311bfe56f0005244c3a76791ef9e',
}
PROTECTED=tuple('outputs/submission_'+v+('.csv' if v in ('v2','v3') else '_trial.csv') for v in ('v2','v3','v4','v5','v6','v7','v8'))+('runs/versions.json',)
need,sha,rel,locked,write,merge,now=prior.need,prior.sha,prior.rel,prior.locked,prior.write,prior.merge,prior.now


def sources():
    return merge(prior.sources(),{rel(Path(__file__)):sha(Path(__file__))})


def freeze():
    inputs={rel(DOCUMENT):sha(DOCUMENT)}
    def small(name,expected=None):
        return prior.previous.prior.small(inputs,ROOT/name,expected,True)
    authorities={n:small(n,d) for n,d in AUTHORITIES.items()}
    p40=authorities['outputs/round40/plan.json'];r40=authorities['outputs/round40/results.json']
    p39=authorities['outputs/round39/evaluation/plan.json'];r39=authorities['outputs/round39/evaluation/results.json']
    r36=authorities['outputs/round36/results.json']
    need(prior.sources()==p40['source_sha256'] and p40['stage_order']==list(STAGES)
         and r40['checked_stages']==list(STAGES) and r36['checked_stages']==list(STAGES),'Prior source/periods differ')
    need(p40['split_contract']==p39['split_contract'] and p40['target_group_weights']==p39['target_group_weights'],'Split/weights differ')
    def bind(name,digest):inputs.update(merge(inputs,{name:digest}))
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',prior.previous.KT):
        bind(name,p40['input_sha256'][name])
    bindings={}
    for stage in STAGES:
        a=small(f'outputs/round40/{stage}/results.json',r40['stage_result_sha256'][stage])
        b=small(f'outputs/round39/evaluation/{stage}/results.json',r39['stage_result_sha256'][stage])
        need(a==r40['stages'][stage] and a['plan_sha256']==AUTHORITIES['outputs/round40/plan.json'],'Stage R40 authority differs')
        spec={}
        for role in ('fit_indices','query_indices','raw_spatial','log_spatial','v7'):
            name=p40['bindings'][stage][role];spec[role]=name;bind(name,p40['input_sha256'][name])
        for role,name in (('unit_spatial',f'outputs/round40/{stage}/unit_spatial.npy'),
                          ('R40_mean',f'outputs/round40/{stage}/three_branch_equal_mean.npy')):
            spec[role]=name;bind(name,a['artifact_sha256'][name])
        name=f'outputs/round36/{stage}/log_spatial_combination.npy';spec['v5']=name;bind(name,p39['input_sha256'][name])
        need(b['unchanged_prior_metrics']['log_spatial_combination']==r36['stages'][stage]['metrics']['log_spatial_combination'], 'v5 metric authorities differ')
        spec['expected_metrics']={'v5':b['unchanged_prior_metrics']['log_spatial_combination'],
            'v7':a['metrics']['v7_control'],'R40_mean':a['metrics']['three_branch_equal_mean']}
        need(spec['expected_metrics']['v7']==b['metrics'][prior.previous.CANDIDATE],'v7 metric authorities differ')
        spec['input_frame_sha256']=a['input_frame_sha256'];bindings[stage]=spec
    src=sources();prior.previous.prior.c6.verify_hashes(merge(inputs,src))
    value={'round':43,'created_at_utc':now(),'candidate':CANDIDATE,'formula':FORMULA,'stage_order':list(STAGES),
        'controls':list(CONTROLS),'all_three_periods_regardless_of_scores':True,'source_sha256':src,'input_sha256':inputs,
        'bindings':bindings,'split_contract':p40['split_contract'],'splits':p40['splits'],
        'target_group_weights':p40['target_group_weights'],'runtime':prior.previous.average.runtime(),
        'motivation_observed_after_final_X_and_prediction_extremes':True,'test_labels_observed':False,
        'no_ID_area_specific_rule':True,'no_new_threshold_or_gate':True,'new_submission_allowed':False,
        'protected_sha256':{n:sha(ROOT/n) for n in PROTECTED}}
    need(src==sources() and inputs[rel(DOCUMENT)]==sha(DOCUMENT),'Freeze source changed')
    write(OUT/'plan.json',value);print('R43_frozen',sha(OUT/'plan.json'),flush=True)


def evaluate(stage,p,Xall,yall):
    t=time.monotonic();b=p['bindings'][stage];spec=p['split_contract'][stage]
    fi,qi=[np.load(ROOT/b[n+'_indices'],allow_pickle=False) for n in ('fit','query')]
    for name,a in (('fit',fi),('query',qi)):
        need(a.dtype==np.dtype('int64') and a.shape==(spec[name+'_rows'],) and (np.diff(a)>0).all()
             and a.min()>=0 and a.max()<len(Xall)
             and prior.previous.prior.log_run.original.value_sha(a,'<i8')==spec[name+'_indices_content_sha256'],'Indices differ')
    need(not np.intersect1d(fi,qi).size,'Fit/query overlap')
    X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
    framehash={'fit_X':prior.spatial.frame_sha(X),'query_X':prior.spatial.frame_sha(Xq)}
    need(framehash==b['input_frame_sha256'],'Original X alignment differs')
    vectors={name:prior.vector(ROOT/b[name],len(qi)) for name in ('raw_spatial','log_spatial','unit_spatial','v5','v7','R40_mean')}
    prior.exact(.5*vectors['raw_spatial']+.5*vectors['log_spatial'],vectors['v7'],'Full v7 mean')
    prior.exact((vectors['raw_spatial']+vectors['log_spatial']+vectors['unit_spatial'])/3.0,vectors['R40_mean'],'Full R40 mean')
    score=lambda v:prior.previous.prior.log_run.original.score(yall[qi],v,X,Xq,p['splits'][stage][0],p['target_group_weights'])
    metrics={name:score(vectors[name]) for name in CONTROLS}
    need(metrics==b['expected_metrics'],'Full original v5/v7/R40 metrics do not replay')
    candidate=np.median(np.column_stack([vectors['raw_spatial'],vectors['log_spatial'],vectors['unit_spatial']]),axis=1)
    need(candidate.dtype==np.dtype('float64') and candidate.shape==(len(qi),) and np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid median')
    metrics[CANDIDATE]=score(candidate)
    folder=OUT/stage;folder.mkdir(exist_ok=False)
    prior.previous.save_array(folder/'predictions.npy',candidate)
    differences={name:prior.previous.prior.log_run.differences(metrics[name],metrics[CANDIDATE]) for name in CONTROLS}
    result={'round':43,'stage':stage,'status':'completed','plan_sha256':sha(OUT/'plan.json'),
        'fit_rows':len(fi),'query_rows':len(qi),'dates':p['splits'][stage],'metrics':metrics,'median_vs_control':differences,
        'v7_R40_mean_full_vectors_bitwise_equal':True,'all_control_metrics_exact':True,
        'input_frame_sha256':framehash,'source_sha256':p['source_sha256'],
        'artifact_sha256':{rel(folder/'predictions.npy'):sha(folder/'predictions.npy')},
        'changed_rows':{name:int(np.count_nonzero(candidate!=vectors[name])) for name in CONTROLS},
        'elapsed_seconds':time.monotonic()-t,'tree_calls':0,'reference_or_spatial_predict_calls':0}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'MAE':{n:{k:m[k] for k in ('overall_mae','weighted_mae')} for n,m in metrics.items()}}),flush=True)
    return result


def run():
    p=locked(OUT/'plan.json')
    need(p['round']==43 and p['formula']==FORMULA and p['stage_order']==list(STAGES)
         and p['source_sha256']==sources() and p['runtime']==prior.previous.average.runtime(),'Frozen contract changed')
    need(not (OUT/'run_claim.json').exists() and not (OUT/'results.json').exists()
         and not any((OUT/s).exists() for s in STAGES),'Prior run exists; no retry')
    write(OUT/'run_claim.json',{'started_at_utc':now(),'automatic_retry':False,'plan_sha256':sha(OUT/'plan.json')})
    t=time.monotonic();result={'round':43,'status':'incomplete','checked_stages':[],'stages':{},'stage_result_sha256':{},
        'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],'plan_sha256':sha(OUT/'plan.json'),
        'protected_before':p['protected_sha256'],'new_fits':0,'tree_model_load_or_predict':0,'new_reference_or_spatial_predictions':0,
        'formal_adoption_decision':None,'new_submission_generated':False,'registry_updated':False,'fresh_blind_validation':False,
        'label_loading':'Original raw train cache is deserialized; canonical K and fixed qi labels only used by unchanged scorer.'}
    try:
        prior.previous.prior.c6.verify_hashes(merge(p['input_sha256'],p['source_sha256'],p['protected_sha256']))
        Xall,yall=prior.previous.load_training_rows(p)
        for stage in STAGES:
            item=evaluate(stage,p,Xall,yall);result['stages'][stage]=item;result['checked_stages'].append(stage)
            result['stage_result_sha256'][stage]=sha(OUT/stage/'results.json')
        prior.previous.prior.c6.verify_hashes(merge(p['source_sha256'],p['protected_sha256']))
        result['status']='three_periods_completed_no_adoption_decision'
    except Exception:
        result['status']='incomplete_technical_failure';result['failure']=traceback.format_exc();print(result['failure'],flush=True)
    result.update(elapsed_seconds=time.monotonic()-t,completed_at_utc=now(),protected_after={n:sha(ROOT/n) for n in PROTECTED})
    write(OUT/'results.json',result);print('R43_result',result['status'],sha(OUT/'results.json'),flush=True)
    return 0 if result['status']=='three_periods_completed_no_adoption_decision' else 1


def execute(mode):
    if mode=='freeze':need(not OUT.exists(),'Existing R43 output');OUT.mkdir(parents=True,exist_ok=False)
    else:need(OUT.is_dir(),'Freeze required')
    started=now();t=time.monotonic();code=1;error=None;log=OUT/(mode+'.log')
    with log.open('x',encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:
                if mode=='freeze':freeze();code=0
                else:code=run()
            except Exception:error=traceback.format_exc();print(error,flush=True)
    event={'round':43,'command':sys.argv,'started_at_utc':started,'completed_at_utc':now(),
      'elapsed_seconds':time.monotonic()-t,'exit_code':code,'source_sha256':sha(Path(__file__)),
      'log_path':rel(log),'log_sha256':sha(log),'failure':error,'automatic_retry':False}
    write(OUT/(mode+'_execution.json'),event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    raise SystemExit(execute(parser.parse_args().command))
