"""R40 fixed third unit-price branch. Saved predictions only; no tree model calls."""
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
import round39_city_road_experiment as previous

ROOT=previous.ROOT
OUT=ROOT/'outputs/round40'
DOCUMENT=ROOT/'reports/round40_three_branch_plan.md'
DOCUMENT_SHA='0648bcd275116c1153aa23ecda0ed354f1af1606bd797ef067e8a1baaa2a0def'
STAGES=('development','confirmation','auxiliary')
CONTROL='v7_control'
UNIT='unit_spatial'
CANDIDATE='three_branch_equal_mean'
FORMULA='(saved_R39_raw_spatial + saved_R39_log_spatial + R8(saved_R39_state,new_query_geo,B_unit,p_unit,diag_unit))/3.0'
BASE_FORMULA='.75*(.5*old_raw_TWD+.5*R19_unit_geo_TWD)+.25*old_unit_LGB_TWD'
AUTHORITIES={
 'outputs/round19/candidate_plan.json':'1dda509846fcb8841d9a8a333d58da79fc70a625b9f18ae9a91dc8b7303ce251',
 'outputs/round19/accepted_candidate.json':'3709265097a185612ec6bfd7fe58115845402e48b45ebd3c55a3d9bdafeca0d5',
 'outputs/round39/evaluation/plan.json':'110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13',
 'outputs/round39/evaluation/results.json':'1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27',
}
PROOF_SHA={'development':'fa518ed9290348ad505b3afca5596ee93ff5b0f2f067a0404b7ad24cd7dda967',
 'confirmation':'2e866676932259094725a0ff93bfcec4d0322e201c9790e1b4277a05198454c4',
 'auxiliary':'a79cdcae88255143454149b17702d988bfcb9e72a5c0648edbd152ecdd40e21f'}
PROTECTED=tuple(dict.fromkeys(previous.PROTECTED+tuple(AUTHORITIES)+('outputs/submission_v7_trial.csv',)))
need,sha,rel,locked,write=previous.need,previous.sha,previous.rel,previous.locked,previous.write
merge,now,vector,exact,spatial=previous.merge,previous.now,previous.vector,previous.exact,previous.spatial


def sources():
    return merge(previous.sources(),{rel(Path(__file__)):sha(Path(__file__)),
        'src/round19_unit_geo.py':sha(ROOT/'src/round19_unit_geo.py'),
        'src/round19_experiment.py':sha(ROOT/'src/round19_experiment.py')})


def freeze():
    need(sha(DOCUMENT)==DOCUMENT_SHA,'Predeclared method document changed')
    inputs={rel(DOCUMENT):DOCUMENT_SHA}
    def small(name,expected=None,receipt=True):
        return previous.prior.small(inputs,ROOT/name,expected,receipt)
    def bind(name,digest):
        inputs.update(merge(inputs,{name:digest}))
    authorities={n:small(n,d) for n,d in AUTHORITIES.items()}
    p19=authorities['outputs/round19/candidate_plan.json']
    p39=authorities['outputs/round39/evaluation/plan.json']
    r39=authorities['outputs/round39/evaluation/results.json']
    need(previous.sources()==p39['source_sha256'],'R39 source closure changed')
    need(r39['status']=='three_periods_completed_no_adoption_decision'
         and r39['checked_stages']==list(STAGES),'R39 periods incomplete')
    need(p19['split_contract']==p39['split_contract'] and p19['target_group_weights']==p39['target_group_weights'], 'R19/R39 split or weight mismatch')
    for n in ('src/round19_unit_geo.py','src/round19_experiment.py'):
        need(sha(ROOT/n)==p19['source_sha256'][n],'Original unit method source changed')
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',previous.KT):
        bind(name,p39['input_sha256'][name])
    bindings={}
    for stage in STAGES:
        rp=f'outputs/round39/evaluation/{stage}/results.json'
        r=small(rp,r39['stage_result_sha256'][stage])
        epath=f'outputs/round19/evaluation/{stage}/results.json'
        e=small(epath)
        proof=small(f'outputs/round19/audit/{stage}_replay.json',PROOF_SHA[stage])
        need(proof['status']=='passed' and proof['result_sha256']==sha(ROOT/epath)
             and e['plan_sha256']==AUTHORITIES['outputs/round19/candidate_plan.json']
             and e['target_group_weights']==p39['target_group_weights'],'Original R19 evidence differs')
        mp=f'outputs/round19/models/{stage}/metadata.json'
        m=small(mp,e['input_artifact_sha256'][mp],False)
        need(m['component']=='geography_unit_catboost' and m['prediction_units']=='TWD'
             and m['feature_count']==95 and m['categorical_feature_count']==23
             and m['sample_weight']=='effective_area / 100' and m['weight_argument_passed_to_pool'] is True
             and m['target_formula']=='y_TWD / (effective_area(X) * 1e4)'
             and m['inverse_formula']=='max(z * (effective_area(X) * 1e4), 0)'
             and m['parameters']==m['baseline_parameters'] and m['changed_parameter_names']==[]
             and m['reload_all_query_rows_verified'] is True and m['reload_exact_equality'] is True,'R19 component contract differs')
        for k,v in {'iterations':2999,'learning_rate':.065,'depth':8,'random_seed':42,'boosting_type':'Plain','loss_function':'MAE'}.items():
            need(m['parameters'][k]==v,'R19 fixed parameter differs')
        for label,framekey in (('fit','fit_X'),('query','query_X')):
            need(m[label+'_indices_content_sha256']==p39['split_contract'][stage][label+'_indices_content_sha256'], 'R19 index content differs')
            mkey='fit_original_feature_values_sha256' if label=='fit' else 'query_X_sha256'
            need(m[mkey]==r['input_frame_sha256'][framekey],'Original X identity differs')
        manifest=small(f'outputs/round39/geography/{stage}/manifest.json',r['geo_manifest_sha256'])
        need(manifest==p39['prepared_geography'][stage] and m['geo_manifest_sha256']==manifest['original_manifest_sha256'], 'Historical geo lineage differs')
        spec={'R19_result':epath,'R39_result':rp,'unit_metadata':mp,'manifest':manifest,
              'unit_component':f'outputs/round19/models/{stage}/predictions.npy',
              'unit_original_blend':f'outputs/round19/evaluation/{stage}/unit_catboost_geography_blend.npy',
              'old_v3':f'outputs/round19/evaluation/{stage}/v3_control.npy',
              'reference':r['reference_path']}
        bind(spec['unit_component'],m['artifact_sha256']['predictions.npy'])
        for key in ('unit_original_blend','old_v3'):bind(spec[key],e['output_artifact_sha256'][spec[key]])
        base=previous.prior.c6.baseline_folder(stage)
        for role,filename in (('old_raw','catboost_raw.npy'),('old_unit','lightgbm_l1_unit63.npy')):
            name=rel(base/filename);spec[role]=name
            need(e['input_artifact_sha256'][name]==p39['input_sha256'][name],'Baseline component identity differs')
            bind(name,p39['input_sha256'][name])
        need(e['input_artifact_sha256'][spec['reference']]==r['reference_sha256'], 'Saved reference identity differs')
        bind(spec['reference'],r['reference_sha256'])
        for role,filename in (('fit_indices','fit_indices.npy'),('query_indices','query_indices.npy'),
          ('raw_spatial','new_raw_spatial.npy'),('log_spatial','new_log_spatial.npy'),
          ('v7','city_road_spatial_half_raw_half_log.npy'),('state','spatial_state.pkl'),('state_summary','state_summary.json')):
            name=f'outputs/round39/evaluation/{stage}/{filename}';spec[role]=name;bind(name,r['artifact_sha256'][name])
        for label in ('fit','query'):
            old_idx=f'outputs/round19/models/{stage}/{label}_indices.npy'
            need(m['artifact_sha256'][label+'_indices.npy']==inputs[spec[label+'_indices']], 'Saved R19/R39 index file bytes differ')
            bind(old_idx,m['artifact_sha256'][label+'_indices.npy'])
        summary_receipt=rel((ROOT/spec['state_summary']).with_suffix('.sha256'))
        bind(summary_receipt,r['artifact_sha256'][summary_receipt])
        record=manifest['features']['query'];bind(record['path'],record['sha256'])
        spec['query_geo']=record['path'];bindings[stage]=spec
    source_map=sources();previous.prior.c6.verify_hashes(merge(inputs,source_map))
    value={'round':40,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'unit_branch':UNIT,
       'formula':FORMULA,'unit_base_formula':BASE_FORMULA,'stage_order':list(STAGES),
       'all_three_periods_regardless_of_scores':True,'source_sha256':source_map,'input_sha256':inputs,
       'bindings':bindings,'split_contract':p39['split_contract'],'splits':p39['splits'],
       'target_group_weights':p39['target_group_weights'],'spatial_parameters':spatial.parameters(),
       'thresholds':{'overall_and_weighted_max_ratio':1.0,'nonempty_quarter_max_ratio':1.05,'stratum_minimum_rows':500,'large_stratum_max_ratio':1.05},
       'protected_sha256':{n:sha(ROOT/n) for n in PROTECTED},'runtime':previous.average.runtime(),
       'tree_model_calls':0,'raw_z_used':False,'automatic_release_allowed':False,'fresh_blind_validation':False}
    need(source_map==sources() and sha(DOCUMENT)==DOCUMENT_SHA,'Sources changed during freeze')
    write(OUT/'plan.json',value);print('R40_plan_frozen',sha(OUT/'plan.json'),flush=True)


def checks(control,candidate):
    return {'overall_mae_not_worse':candidate['overall_mae']<=control['overall_mae'],
      'weighted_mae_not_worse':candidate['weighted_mae']<=control['weighted_mae'],
      'nonempty_quarters_at_most_5pct_worse':all(candidate['forecast_quarters'][k]['mae']<=1.05*v['mae'] for k,v in control['forecast_quarters'].items() if v['rows']>0),
      'large_strata_at_most_5pct_worse':all(candidate['strata'][k]['mae']<=1.05*v['mae'] for k,v in control['strata'].items() if v['rows']>=500)}


def evaluate(stage,plan,Xall,yall):
    started=time.monotonic();b=plan['bindings'][stage];folder=OUT/stage;folder.mkdir(exist_ok=False)
    fi,qi=[np.load(ROOT/b[k+'_indices'],allow_pickle=False) for k in ('fit','query')]
    spec=plan['split_contract'][stage]
    for label,a in (('fit',fi),('query',qi)):
        need(a.dtype==np.dtype('int64') and a.shape==(spec[label+'_rows'],) and (np.diff(a)>0).all()
             and a.min()>=0 and a.max()<len(Xall)
             and previous.prior.log_run.original.value_sha(a,'<i8')==spec[label+'_indices_content_sha256'],'Canonical indices changed')
    need(not np.intersect1d(fi,qi).size,'Fit/query overlap')
    X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
    yq=yall[qi]  # Used only below in the unchanged scorer.
    r19,r39=locked(ROOT/b['R19_result']),locked(ROOT/b['R39_result'])
    framehash={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
    need(all(framehash[k]==r39['input_frame_sha256'][k] for k in framehash),'Original frame identity differs')
    geoq=pd.read_pickle(ROOT/b['query_geo']);previous.geo_encoder.check_geo(geoq)
    need(geoq.index.equals(Xq.index) and spatial.frame_sha(geoq)==b['manifest']['features']['query']['frame_sha256'], 'New query geography differs')
    with (ROOT/b['state']).open('rb') as stream:state=pickle.load(stream)
    summary=locked(ROOT/b['state_summary'])
    need(state['state_summary']==summary and state['parameters']==plan['spatial_parameters']
         and summary['fit_X_sha256']==framehash['fit_X']
         and summary['fit_geo_sha256']==b['manifest']['features']['fit']['frame_sha256'],'Saved R39 state differs')
    with (ROOT/b['reference']).open('rb') as stream:reference=pickle.load(stream)
    need(reference['fit_report']['fitting_rows']==len(X)
         and previous.prior.log_run.original.value_sha(reference['reference_residual'],'<f8')==summary['existing_reference_residual_sha256'], 'Reference/residual fit identity differs')
    old,unit,component=[vector(ROOT/b[k],len(qi)) for k in ('old_raw','old_unit','unit_component')]
    B=.75*(.5*old+.5*component)+.25*unit
    p,diag,reference_report=previous.prior.log_run.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
    exact(p,vector(ROOT/b['unit_original_blend'],len(qi)),'Original complete R19 blend')
    raw,log=[vector(ROOT/b[k],len(qi)) for k in ('raw_spatial','log_spatial')]
    control=.5*raw+.5*log;exact(control,vector(ROOT/b['v7'],len(qi)),'v7 complete saved control')
    score=lambda v:previous.prior.log_run.original.score(yq,v,X,Xq,plan['splits'][stage][0],plan['target_group_weights'])
    old_metrics={'v3_control':score(vector(ROOT/b['old_v3'],len(qi))), 'unit_catboost_geography_blend':score(p)}
    need(old_metrics==r19['metrics'],'Full original R19 metrics do not replay')
    control_metrics=score(control)
    need(control_metrics==r39['metrics'][previous.CANDIDATE],'Full original v7 metrics do not replay')
    unit_spatial,detail,spatial_report=spatial.predict(state,Xq,geoq,B,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    exact(unit_spatial[fallback],p[fallback],'Unit spatial fallback baseline')
    candidate=(raw+log+unit_spatial)/3.0
    need(np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid three-branch TWD mean')
    predictions={CONTROL:control,'original_R19_blend':p,UNIT:unit_spatial,CANDIDATE:candidate}
    metrics={CONTROL:control_metrics,UNIT:score(unit_spatial),CANDIDATE:score(candidate)}
    decisions=checks(control_metrics,metrics[CANDIDATE])
    for name,values in predictions.items():
        previous.save_array(folder/(name+'.npy'),values)
        exact(values,vector(folder/(name+'.npy'),len(qi)),'Saved '+name)
    with (folder/'unit_reference_diagnostics.pkl').open('xb') as stream:pickle.dump(diag,stream,protocol=5)
    with (folder/'unit_spatial_diagnostics.pkl').open('xb') as stream:pickle.dump(detail,stream,protocol=5)
    need(framehash=={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
         and spatial.frame_sha(geoq)==b['manifest']['features']['query']['frame_sha256'], 'Features mutated')
    result={'round':40,'stage':stage,'status':'completed','created_at_utc':now(),'plan_sha256':sha(OUT/'plan.json'),
      'fit_rows':len(fi),'query_rows':len(qi),'dates':plan['splits'][stage],'input_frame_sha256':framehash,
      'metrics':metrics,'unit_vs_v7':previous.prior.log_run.differences(control_metrics,metrics[UNIT]),
      'candidate_vs_v7':previous.prior.log_run.differences(control_metrics,metrics[CANDIDATE]),
      'engineering_checks':decisions,'necessary_condition_passed':all(decisions.values()),
      'original_R19_metrics':old_metrics,'old_R19_full_vector_bitwise_equal':True,'v7_full_vector_bitwise_equal':True,
      'old_R19_all_metrics_equal':True,'v7_all_metrics_equal':True,'source_sha256':plan['source_sha256'],
      'reference_report':reference_report,'spatial_report':spatial_report,'tree_models_trained_loaded_or_predicted':0,
      'new_reference_fits':0,'new_spatial_state_builds':0,'reference_predict_calls':1,'new_spatial_predict_calls':1,
      'query_labels_used_in_reference_or_spatial_predict':False,'original_frames_unchanged':True,
      'formal_adoption_decision':None,'new_submission_generated':False,'fresh_blind_validation':False,
      'changed_rows_vs_v7':int(np.count_nonzero(candidate!=control)), 'elapsed_seconds':time.monotonic()-started,
      'artifact_sha256':{rel(p):sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'metrics':{k:{n:v[n] for n in ('overall_mae','weighted_mae')} for k,v in metrics.items()},'engineering_checks':decisions}),flush=True)
    return result


def run():
    p=locked(OUT/'plan.json')
    need(p['round']==40 and p['formula']==FORMULA and p['unit_base_formula']==BASE_FORMULA
         and p['stage_order']==list(STAGES) and p['all_three_periods_regardless_of_scores'] is True
         and p['source_sha256']==sources() and p['runtime']==previous.average.runtime()
         and p['spatial_parameters']==spatial.parameters(),'Frozen contract changed')
    need(not (OUT/'run_claim.json').exists() and not (OUT/'results.json').exists()
         and not any((OUT/s).exists() for s in STAGES),'Prior attempt exists; no automatic retry')
    write(OUT/'run_claim.json',{'started_at_utc':now(),'plan_sha256':sha(OUT/'plan.json'),'automatic_retry':False})
    result={'round':40,'status':'incomplete','plan_sha256':sha(OUT/'plan.json'),'stage_order':list(STAGES),
      'checked_stages':[],'stages':{},'stage_result_sha256':{},'source_sha256':p['source_sha256'],
      'input_sha256':p['input_sha256'],'protected_before':p['protected_sha256'],'formal_adoption_decision':None,
      'tree_models_trained_loaded_or_predicted':0,'new_submission_generated':False,'registry_updated':False,
      'all_periods_necessary_condition':None,'fresh_blind_validation':False,
      'label_loading':'shared R39 loader deserializes raw train cache; only fixed qi labels used by scorer; no target in reference/spatial calls'}
    start=time.monotonic();phase='verify_inputs_before_data_load'
    try:
        previous.prior.c6.verify_hashes(merge(p['input_sha256'],p['source_sha256'],p['protected_sha256']))
        Xall,yall=previous.load_training_rows(p)
        for stage in STAGES:
            phase=stage;item=evaluate(stage,p,Xall,yall)
            result['checked_stages'].append(stage);result['stages'][stage]=item
            result['stage_result_sha256'][stage]=sha(OUT/stage/'results.json')
        result['all_periods_necessary_condition']=all(v['necessary_condition_passed'] for v in result['stages'].values())
        previous.prior.c6.verify_hashes(merge(p['source_sha256'],p['protected_sha256']))
        result['status']='three_periods_completed_no_adoption_decision'
    except Exception:
        result.update(status='incomplete_technical_failure',failure_phase=phase,traceback=traceback.format_exc())
        print(result['traceback'],flush=True)
    result.update(elapsed_seconds=time.monotonic()-start,completed_at_utc=now(),protected_after={n:sha(ROOT/n) for n in PROTECTED})
    write(OUT/'results.json',result)
    print('R40_result',result['status'],sha(OUT/'results.json'),flush=True)
    return 0 if result['status']=='three_periods_completed_no_adoption_decision' else 1


def execute(mode):
    if mode=='freeze':
        need(not OUT.exists(),'R40 output already exists');OUT.mkdir(parents=True,exist_ok=False)
    else:need(OUT.is_dir(),'Freeze is required')
    log=OUT/(mode+'.log');started=now();start=time.monotonic();code=1;failure=None
    with log.open('x',encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:
                if mode=='freeze':freeze();code=0
                else:code=run()
            except Exception:
                failure=traceback.format_exc();print(failure,flush=True)
    receipt={'round':40,'command':sys.argv,'started_at_utc':started,'completed_at_utc':now(),
      'elapsed_seconds':time.monotonic()-start,'exit_code':code,'source_sha256':sha(Path(__file__)),
      'stdout_log':rel(log),'stdout_log_sha256':sha(log),'failure':failure,'automatic_retry':False}
    write(OUT/(mode+'_execution.json'),receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    raise SystemExit(execute(parser.parse_args().command))
