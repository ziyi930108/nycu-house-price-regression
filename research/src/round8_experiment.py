"""R8: fixed project-balanced geographic reference; no tree retraining or weight search."""
from __future__ import annotations
import argparse,ast,hashlib,json,pickle,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import round6_common as c6
import round6_evaluate as e6
import round8_spatial_reference as component
from round6_prepare_geo import load_geographic_partition
from round6_geo import frame_sha
from round2_neighborhood import predict_reference
from round3_evaluate import score,target_weights,runtime_versions as prior_runtime_versions
ROOT=c6.ROOT
OUT=ROOT/'outputs/round8'
STAGES=tuple(c6.SPLITS)
CANDIDATE='project_balanced_spatial_reference'
CONTROL='v3_control'
SNAPSHOT_SHA='77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA='52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
THRESHOLDS=dict(e6.THRESHOLDS)
FORMULAS={CONTROL:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once',CANDIDATE:'p3 + (1-c)*a*(Gclip-Bfuture) only on eligible weak sources; a=.30*exp(-project_MAD/.18); Gclip/Bfuture in [.65,1.5]; no extra trend'}
require,read,write,locked,locked_write,sha,rel=c6.require,c6.read,c6.write,c6.read_locked,c6.locked_write,c6.sha256,c6.rel


def runtime_versions():
    import scipy
    return {**prior_runtime_versions(),'scipy':scipy.__version__}

def now():return datetime.now(timezone.utc).isoformat()

def value_sha(a,dtype):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()

def sources():
    pending=[Path(__file__).resolve()];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/'src'/(name.split('.')[0]+'.py')
                if local.is_file() and local not in found:pending.append(local)
    return {rel(p):sha(p) for p in sorted(found)}

def stage_paths(stage):
    base=c6.baseline_folder(stage)
    geo=c6.OUT/'models'/stage/'geography_catboost'
    evaluation=c6.OUT/'evaluation'/stage
    paths=[base/n for n in ['catboost_raw.npy','lightgbm_l1_unit63.npy','reference_v2.pkl','fit_indices.npy','check_indices.npy']]
    paths += [geo/n for n in ['metadata.json','predictions.npy','fit_indices.npy','query_indices.npy']]
    paths += [evaluation/n for n in ['results.json','results.sha256','geography_blend.npy']]
    manifest=c6.OUT/'geography'/stage/'manifest.json'
    paths += [manifest,manifest.with_suffix('.sha256')]
    item=locked(manifest)
    paths += [ROOT/v['path'] for v in item['features'].values()]
    paths += [ROOT/p for p in item['index_files_sha256']]
    return sorted(set(paths))

def plan():
    value=locked(OUT/'candidate_plan.json')
    require(value['round']==8 and value['candidate']==CANDIDATE and value['control']==CONTROL,'Wrong R8 plan')
    require(value['formulas']==FORMULAS and value['thresholds']==THRESHOLDS and value['parameters']==component.parameters(),'Fixed configuration changed')
    require(value['source_sha256']==sources(),'Frozen source closure changed')
    require(value['splits']=={k:list(v) for k,v in c6.SPLITS.items()},'Split definitions changed')
    require(value['runtime']==runtime_versions(),'Runtime versions changed')
    c6.verify_hashes(value['dependency_sha256'])
    return value

def authorize(stage):
    require(stage in STAGES,'Only fixed backtest stages allowed')
    value=plan();c6.verify_hashes(value['stage_dependency_sha256'][stage])
    if stage!='development':
        frozen=locked(OUT/'frozen_candidate.json')
        require(frozen['selected_candidate']==CANDIDATE and frozen['passed'] is True,'Development rejected; later stages prohibited')
        require(frozen['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong development plan')
        c6.verify_hashes(frozen['development_artifact_sha256'])
    return value

def partition(stage):
    data,X,y,Xq,yq,fi,qi=c6.partition(stage)
    spec=locked(OUT/'candidate_plan.json')['split_contract'][stage]
    for label,array in [('fit',fi),('query',qi)]:
        require(spec[label+'_indices_content_sha256']==value_sha(array,'<i8'),'R8 partition index differs')
    base=c6.baseline_folder(stage);old=c6.OUT/'models'/stage/'geography_catboost'
    e6.verify_indices(base,fi,qi,'check_indices.npy');e6.verify_indices(old,fi,qi,'query_indices.npy')
    return data,X,y,Xq,yq,fi,qi

def publish(staging,destination):
    require(staging.resolve().parent==destination.resolve().parent and staging.resolve().is_relative_to(OUT.resolve()),'Unsafe R8 publication path')
    require(not destination.exists(),'Refusing to replace recorded experiment')
    staging.rename(destination)

def decide(stage,metrics):
    require(set(metrics)=={CONTROL,CANDIDATE},'Only the fixed control and candidate may be scored')
    base,value=metrics[CONTROL],metrics[CANDIDATE]
    large=e6.large_strata_ok(value,base)
    improved=[q for q,item in base['forecast_quarters'].items() if item['rows'] and value['forecast_quarters'][q]['mae']<item['mae']]
    if stage=='development':
        checks={'overall_improves_at_least_1pct':value['overall_mae']<=(1-THRESHOLDS['development_min_overall_improvement'])*base['overall_mae'],'weighted_improves_at_least_1pct':value['weighted_mae']<=(1-THRESHOLDS['development_min_weighted_improvement'])*base['weighted_mae'],'quarters_deteriorate_at_most_5pct':all(value['forecast_quarters'][q]['mae']<=(1+THRESHOLDS['development_max_quarter_deterioration'])*item['mae'] for q,item in base['forecast_quarters'].items() if item['rows']),'large_strata_deteriorate_at_most_5pct':large}
    else:
        require(stage in ('confirmation','auxiliary'),'Unknown evaluation stage')
        checks={'overall_within_limit':value['overall_mae']<=(1+THRESHOLDS[stage+'_max_overall_deterioration'])*base['overall_mae'],'weighted_within_limit':value['weighted_mae']<=(1+THRESHOLDS[stage+'_max_weighted_deterioration'])*base['weighted_mae'],'large_strata_deteriorate_at_most_5pct':large}
        if stage=='confirmation':checks['at_least_two_quarters_improve']=len(improved)>=THRESHOLDS['confirmation_min_improved_quarters']
    passed=all(checks.values())
    return {'passed':passed,'selected_candidate':CANDIDATE if passed else CONTROL,'checks':checks,'improved_quarters':improved,'candidate_reselection_performed':False}

def read_result(stage):
    path=OUT/'evaluation'/stage/'results.json';value=locked(path)
    require(value['stage']==stage and value['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong saved result')
    require(value['source_sha256']==sources() and value['formulas']==FORMULAS and value['thresholds']==THRESHOLDS,'Saved result protocol differs')
    require(value['frozen_candidate_sha256']==(None if stage=='development' else sha(OUT/'frozen_candidate.json')),'Saved result freeze differs')
    require(value['decision']==decide(stage,value['metrics']),'Saved gate decision differs')
    for field in ['input_artifact_sha256','output_artifact_sha256']:c6.verify_hashes(value[field])
    return value

def freeze_development(result):
    path=OUT/'frozen_candidate.json'
    if path.exists():
        frozen=locked(path)
        require(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'Existing freeze result differs')
        return
    artifacts={rel(p):sha(p) for folder in [OUT/'models/development',OUT/'evaluation/development'] for p in folder.iterdir() if p.is_file()}
    value={'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'result_sha256':sha(OUT/'evaluation/development/results.json'),'passed':result['decision']['passed'],'selected_candidate':result['decision']['selected_candidate'],'development_artifact_sha256':artifacts,'source_sha256':sources(),'formulas':FORMULAS,'thresholds':THRESHOLDS,'parameters':component.parameters(),'later_stages_allowed':result['decision']['passed'],'historically_untouched':False}
    locked_write(path,value)

def finalize():
    specification=plan();frozen=locked(OUT/'frozen_candidate.json');development=read_result('development')
    require(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json') and frozen['passed']==development['decision']['passed'],'Development freeze differs')
    results={'development':development}
    if frozen['passed']:
        for stage in ('confirmation','auxiliary'):
            authorize(stage);results[stage]=read_result(stage)
    accepted=all(r['decision']['passed'] for r in results.values()) and len(results)==3
    path=OUT/'accepted_candidate.json'
    result={'created_at_utc':now(),'round':8,'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),'accepted':accepted,'candidate':CANDIDATE,'deployment_decision':CANDIDATE if accepted else 'keep_v3','result_sha256':{s:sha(OUT/'evaluation'/s/'results.json') for s in results},'unevaluated_stages':[s for s in STAGES if s not in results],'no_reselection_or_parameter_change':True,'new_submission_created':False,'private_score_known':False,'source_sha256':specification['source_sha256']}
    if path.exists():
        saved=locked(path);require(all(saved[k]==v for k,v in result.items() if k!='created_at_utc'),'Existing final decision differs');result=saved
    else:locked_write(path,result)
    print(json.dumps(result),flush=True)
    return result

def plan_create():
    require(not (OUT/'candidate_plan.json').exists(),'R8 plan exists; never overwrite')
    snapshot_path=ROOT/'runs/v3/snapshot_manifest.json'
    require(sha(snapshot_path)==SNAPSHOT_SHA,'Released v3 snapshot changed')
    snapshot=locked(snapshot_path)
    known={k:v['original_sha256'] for k,v in snapshot['copied_files'].items()}
    known.update({v['original_path']:v['sha256'] for v in snapshot['external_dependencies']})
    old=c6.plan();deps=dict(old['data_sha256'])
    required=[ROOT/'outputs/submission_v3.csv',ROOT/'outputs/submission_v2.csv',snapshot_path,snapshot_path.with_suffix('.sha256'),c6.OUT/'candidate_plan.json',c6.OUT/'candidate_plan.sha256',ROOT/'reports/round8_validation_plan.md',ROOT/'reports/round8_design_review.md',ROOT/'reports/round8_location_feasibility.md',ROOT/'src/round8_location_feasibility.py',OUT/'audit/location_feasibility.json',OUT/'audit/preflight.json',ROOT/'outputs/round7/accepted_candidate.json',ROOT/'outputs/round7/accepted_candidate.sha256']
    for path in required:deps[rel(path)]=sha(path)
    require(deps['outputs/submission_v3.csv']==V3_SHA and deps['outputs/submission_v2.csv']==V2_SHA,'Preserved submissions changed')
    preflight=read(OUT/'audit/preflight.json')
    require(preflight['all_checks_passed'] is True,'Preflight did not pass')
    require(preflight['source_sha256']==sources(),'Source changed after preflight')
    stages={}
    for stage in STAGES:
        mapping={}
        for path in stage_paths(stage):
            name=rel(path);digest=sha(path)
            require(known.get(name)==digest,'Stage dependency differs from released v3 snapshot: '+name)
            mapping[name]=digest
        manifest=locked(c6.OUT/'geography'/stage/'manifest.json')
        for name,digest in manifest['source_sha256'].items():
            require(sha(ROOT/name)==digest,'Geography source differs: '+name);mapping[name]=digest
        stages[stage]=mapping
    final_manifest=c6.OUT/'geography/final/manifest.json'
    final_paths=[final_manifest,final_manifest.with_suffix('.sha256'),c6.OUT/'release/release.json',c6.OUT/'release/release.sha256']
    fm=locked(final_manifest)
    final_paths += [ROOT/v['path'] for v in fm['features'].values()]
    final_dependencies={}
    for path in final_paths:
        name=rel(path);digest=sha(path)
        require(known.get(name)==digest,'Final dependency differs from v3 snapshot: '+name)
        final_dependencies[name]=digest
    for field in ('source_sha256','index_files_sha256'):
        for name,digest in fm[field].items():
            require(sha(ROOT/name)==digest,'Final geography input differs: '+name);final_dependencies[name]=digest
    value={'round':8,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'formulas':FORMULAS,'thresholds':THRESHOLDS,'parameters':component.parameters(),'new_tree_models_trained':0,'original_feature_columns':87,'geography_feature_columns':8,'source_sha256':sources(),'dependency_sha256':deps,'stage_dependency_sha256':stages,'final_dependency_sha256':final_dependencies,'final_contract':old['final_contract'],'splits':old['splits'],'split_contract':old['split_contract'],'target_group_weights':old['target_group_weights'],'quarter_definition':old['quarter_definition'],'old_plan_sha256':sha(c6.OUT/'candidate_plan.json'),'v3_snapshot_sha256':SNAPSHOT_SHA,'single_development_candidate':True,'tuning_or_weight_search':False,'public_score_used_for_selection':False,'historically_untouched':False,'external_property_answers_used':False,'query_labels_passed_to_component':False,'query_X_used_to_fit_reference':False,'runtime':runtime_versions(),'registry_sha256_at_plan':sha(ROOT/'runs/versions.json')}
    c6.verify_hashes(deps)
    require(value['source_sha256']==preflight['source_sha256']==sources(),'Sources changed during plan creation')
    locked_write(OUT/'candidate_plan.json',value)
    print(json.dumps({'status':'plan_frozen','plan_sha256':sha(OUT/'candidate_plan.json'),'candidate':CANDIDATE}),flush=True)

def controls(stage,Xq,n):
    folder=c6.baseline_folder(stage)
    old=e6.vector(folder/'catboost_raw.npy',n)
    unit=e6.vector(folder/'lightgbm_l1_unit63.npy',n)
    geographic=e6.vector(c6.OUT/'models'/stage/'geography_catboost/predictions.npy',n)
    original=e6.vector(c6.OUT/'evaluation'/stage/'geography_blend.npy',n)
    base=.75*(.5*old+.5*geographic)+.25*unit
    return base,original

def replay_control(reference,Xq,base,original):
    prediction,diagnostics,report=predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
    require(np.allclose(original,prediction,rtol=1e-12,atol=1e-5),'v3 control does not replay')
    require(diagnostics.shape[0]==len(Xq),'Reference diagnostics length differs')
    return diagnostics,report

def build(stage):
    specification=authorize(stage);destination=OUT/'models'/stage
    require(not destination.exists(),'Reference already exists; do not rebuild')
    data,X,y,Xq,yq,fi,qi=partition(stage)
    del data,y,yq
    geo,geoq,manifest=load_geographic_partition(stage,fi,qi)
    reference,reference_path=c6.load_reference(stage,X)
    base,original=controls(stage,Xq,len(qi))
    existing_diag,control_report=replay_control(reference,Xq,base,original)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    started=time.monotonic()
    original_fingerprints={key:frame_sha(frame) for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]}
    print(json.dumps({'event':'reference_build_started','stage':stage,'fit_rows':len(X),'query_rows':len(Xq),'new_tree_models':0}),flush=True)
    model,state=component.build(reference,X,geo)
    predictions,diagnostics,report=component.predict(model,Xq,geoq,base,original,existing_diag)
    with (staging/'reference.pkl').open('wb') as handle:pickle.dump(model,handle,protocol=5)
    write(staging/'state_summary.json',state);write(staging/'prediction_report.json',report)
    diagnostics.to_pickle(staging/'diagnostics.pkl')
    with (staging/'reference.pkl').open('rb') as handle:reloaded=pickle.load(handle)
    small,_,_=component.predict(reloaded,Xq.iloc[:64],geoq.iloc[:64],base[:64],original[:64],existing_diag.iloc[:64])
    require(np.allclose(small,predictions[:64],rtol=1e-12,atol=1e-5),'Reference reload differs')
    require(predictions.shape==(len(Xq),) and np.isfinite(predictions).all() and (predictions>=0).all(),'Invalid TWD predictions')
    after={key:frame_sha(frame) for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]}
    require(after==original_fingerprints,'Feature frames mutated')
    for name,values in [('fit_indices',fi),('query_indices',qi),('predictions',predictions)]:np.save(staging/(name+'.npy'),values,allow_pickle=False)
    metadata={'round':8,'stage':stage,'component':CANDIDATE,'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'source_sha256':specification['source_sha256'],'parameters':component.parameters(),'outer_fit_rows':len(X),'query_rows':len(Xq),**original_fingerprints,'reference_path':rel(reference_path),'reference_sha256':sha(reference_path),'geo_manifest_sha256':sha(c6.OUT/'geography'/stage/'manifest.json'),'artifact_sha256':{p.name:sha(p) for p in staging.iterdir() if p.is_file()},'reload_64_rows_verified':True,'v3_control_replayed_all_rows':True,'feature_frames_unchanged':True,'query_labels_used':False,'new_tree_models_trained':0,'prediction_units':'TWD','reference_state_uses_query':False,'runner_elapsed_seconds':time.monotonic()-started,'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json')}
    locked_write(staging/'metadata.json',metadata)
    authorize(stage);publish(staging,destination)
    print(json.dumps({'event':'reference_build_completed','stage':stage,'elapsed_seconds':time.monotonic()-started,'coverage':report}),flush=True)

def model_check(stage,X,Xq,fi,qi,geo,geoq,specification,base,original,diagnostics,reference_path):
    folder=OUT/'models'/stage;meta=locked(folder/'metadata.json')
    require(meta['round']==8 and meta['stage']==stage and meta['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong model identity')
    require(meta['source_sha256']==specification['source_sha256'] and meta['parameters']==specification['parameters'],'Model configuration differs')
    require(meta['outer_fit_rows']==len(X) and meta['query_rows']==len(Xq),'Model sample counts differ')
    for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]:require(meta[key]==frame_sha(frame),'Model feature identity differs: '+key)
    require(meta['geo_manifest_sha256']==sha(c6.OUT/'geography'/stage/'manifest.json'),'Model map version differs')
    require(meta['reference_path']==rel(reference_path) and meta['reference_sha256']==sha(reference_path),'Reference fit identity differs')
    require(meta['reload_64_rows_verified'] is True and meta['query_labels_used'] is False and meta['reference_state_uses_query'] is False and meta['new_tree_models_trained']==0,'Model construction contract differs')
    require(meta['frozen_candidate_sha256']==(None if stage=='development' else sha(OUT/'frozen_candidate.json')),'Model selection identity differs')
    for name,digest in meta['artifact_sha256'].items():require(Path(name).name==name and sha(folder/name)==digest,'Saved reference artifact changed')
    e6.verify_indices(folder,fi,qi,'query_indices.npy')
    saved=e6.vector(folder/'predictions.npy',len(qi))
    with (folder/'reference.pkl').open('rb') as handle:model=pickle.load(handle)
    replay,replay_diag,replay_report=component.predict(model,Xq,geoq,base,original,diagnostics)
    require(np.allclose(saved,replay,rtol=1e-12,atol=1e-5),'All-query reference replay differs')
    import pandas as pd
    require(replay_diag.equals(pd.read_pickle(folder/'diagnostics.pkl')),'All-query diagnostics replay differs')
    # Runtime timings, when present, are observational rather than fitted state.
    expected_report=read(folder/'prediction_report.json')
    for key,value in expected_report.items():
        if 'seconds' not in key:require(replay_report[key]==value,'Prediction report replay differs: '+key)
    strong=~diagnostics['source'].isin(['district_style','base_fallback']).to_numpy()
    require(np.array_equal(saved[strong],original[strong]),'Strong reference rows changed')
    return saved,{rel(p):sha(p) for p in folder.iterdir() if p.is_file()}

def evaluate(stage):
    specification=authorize(stage);destination=OUT/'evaluation'/stage
    if destination.exists():
        result=read_result(stage)
        if stage=='development':freeze_development(result)
        print(json.dumps({'status':'existing_result_verified','stage':stage,'decision':result['decision']}),flush=True)
        return result
    data,X,y,Xq,yq,fi,qi=partition(stage)
    weights=target_weights(data)
    require(weights==specification['target_group_weights'],'Fixed test-X grouping weights differ')
    geo,geoq,_=load_geographic_partition(stage,fi,qi)
    reference,reference_path=c6.load_reference(stage,X)
    base,original=controls(stage,Xq,len(qi));diagnostics,control_report=replay_control(reference,Xq,base,original)
    new,inputs=model_check(stage,X,Xq,fi,qi,geo,geoq,specification,base,original,diagnostics,reference_path)
    inputs.update(specification['stage_dependency_sha256'][stage])
    predictions={CONTROL:original,CANDIDATE:new}
    metrics={name:score(yq,pred,X,Xq,c6.SPLITS[stage][0],weights) for name,pred in predictions.items()}
    previous=locked(c6.OUT/'evaluation'/stage/'results.json')
    require(metrics[CONTROL]==previous['metrics']['geography_blend'],'Recomputed v3 metrics differ from archived result')
    decision=decide(stage,metrics)
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    for name,values in predictions.items():np.save(staging/(name+'.npy'),values,allow_pickle=False)
    outputs={rel(destination/p.name):sha(p) for p in staging.iterdir() if p.is_file()}
    result={'round':8,'stage':stage,'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json'),'source_sha256':specification['source_sha256'],'input_artifact_sha256':inputs,'output_artifact_sha256':outputs,'formulas':FORMULAS,'thresholds':THRESHOLDS,'target_group_weights':weights,'metrics':metrics,'decision':decision,'v3_control_replayed_all_rows':True,'new_reference_replayed_all_rows':True,'fit_rows':len(fi),'query_rows':len(qi),'evaluated_candidates':[CONTROL,CANDIDATE],'query_labels_used_for_training':False,'public_score_used':False,'historically_untouched':False,'new_public_score_known':False,'private_score_known':False}
    locked_write(staging/'results.json',result)
    authorize(stage);publish(staging,destination)
    if stage=='development':freeze_development(result)
    print(json.dumps({'stage':stage,'metrics':{k:{n:v[n] for n in ['overall_mae','weighted_mae']} for k,v in metrics.items()},'decision':decision}),flush=True)
    return result

def self_check():
    import copy
    base={'overall_mae':100.,'weighted_mae':100.,'strata':{'seen':{'rows':500,'mae':100.},'missing':{'rows':0,'mae':None}},'forecast_quarters':{q:{'rows':10,'mae':100.} for q in ['Q1','Q2','Q3','Q4']}}
    checks=[]
    component_checks=component.self_check()
    def run(label,truth):require(truth,label);checks.append(label)
    def metrics(overall,weighted=None):
        value=copy.deepcopy(base);value['overall_mae']=overall;value['weighted_mae']=overall if weighted is None else weighted
        return {CONTROL:copy.deepcopy(base),CANDIDATE:value}
    run('development_exact_1pct_passes',decide('development',metrics(99))['passed'])
    run('development_below_1pct_rejected',not decide('development',metrics(99.00001))['passed'])
    run('weighted_also_required',not decide('development',metrics(98,100))['passed'])
    m=metrics(98);m[CANDIDATE]['strata']['seen']['mae']=105.00001
    run('500_row_stratum_gate',not decide('development',m)['passed'])
    m=metrics(98);m[CANDIDATE]['forecast_quarters']['Q1']['mae']=105.00001
    run('quarter_gate',not decide('development',m)['passed'])
    m=metrics(99);m[CANDIDATE]['forecast_quarters']['Q1']['mae']=99
    run('confirmation_one_quarter_insufficient',not decide('confirmation',m)['passed'])
    m[CANDIDATE]['forecast_quarters']['Q2']['mae']=99
    run('confirmation_two_quarters_pass',decide('confirmation',m)['passed'])
    run('auxiliary_exact_3pct_passes',decide('auxiliary',metrics(103))['passed'])
    run('auxiliary_over_3pct_rejected',not decide('auxiliary',metrics(103.00001))['passed'])
    proof={'created_at_utc':now(),'all_checks_passed':True,'synthetic_checks':checks,'component_synthetic_checks':component_checks,'source_sha256':sources(),'real_model_training_or_scoring':False,'parameters':component.parameters()}
    path=OUT/'audit/preflight.json';require(not path.exists(),'Preflight exists; preserve audit')
    write(path,proof);print(json.dumps(proof),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check','plan','build','evaluate','finalize'])
    parser.add_argument('--stage',choices=STAGES,default='development')
    args=parser.parse_args()
    if args.command=='self-check':self_check()
    elif args.command=='plan':plan_create()
    elif args.command=='build':build(args.stage)
    elif args.command=='evaluate':evaluate(args.stage)
    else:finalize()
