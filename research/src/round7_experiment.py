"""R7: one fixed geographic unit-LightGBM replacement, compared with v3.
Commands never modify R6, previous submissions, shared caches or the registry.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import lightgbm as lgb
import round6_common as c6
import round6_evaluate as e6
import round7_geo_unit as component
from round6_prepare_geo import load_geographic_partition
from round6_geo import frame_sha
from train_lightgbm import effective_area,price_factor
from round2_neighborhood import predict_reference
from round3_evaluate import score,target_weights,runtime_versions
ROOT=c6.ROOT
OUT=ROOT/'outputs/round7'
STAGES=tuple(c6.SPLITS)
CANDIDATE='geography_unit_replacement'
CONTROL='v3_control'
SNAPSHOT_SHA='77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA='52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
THRESHOLDS=dict(e6.THRESHOLDS)
FORMULAS={CONTROL:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once',CANDIDATE:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*geo_unit_lgb; then v2 trend once'}
require,read,write,locked,locked_write,sha,rel=c6.require,c6.read,c6.write,c6.read_locked,c6.locked_write,c6.sha256,c6.rel

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

def plan_create():
    require(not (OUT/'candidate_plan.json').exists(),'R7 plan exists; never overwrite')
    snapshot_path=ROOT/'runs/v3/snapshot_manifest.json'
    require(sha(snapshot_path)==SNAPSHOT_SHA,'Released v3 snapshot changed')
    snapshot=locked(snapshot_path)
    known={k:v['original_sha256'] for k,v in snapshot['copied_files'].items()}
    known.update({v['original_path']:v['sha256'] for v in snapshot['external_dependencies']})
    old=c6.plan()
    deps=dict(old['data_sha256'])
    required=[ROOT/'outputs/submission_v3.csv',ROOT/'outputs/submission_v2.csv',snapshot_path,snapshot_path.with_suffix('.sha256'),c6.OUT/'candidate_plan.json',c6.OUT/'candidate_plan.sha256',ROOT/'runs/v1/models/final_config.json',ROOT/'reports/round7_validation_plan.md',ROOT/'reports/round7_feature_audit.md',OUT/'audit/preflight.json']
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
        stages[stage]=mapping
    final_manifest=c6.OUT/'geography/final/manifest.json'
    final_paths=[final_manifest,final_manifest.with_suffix('.sha256'),c6.OUT/'release/release.json',c6.OUT/'release/release.sha256']
    final_paths += [ROOT/v['path'] for v in locked(final_manifest)['features'].values()]
    final_dependencies={}
    for path in final_paths:
        name=rel(path);digest=sha(path)
        require(known.get(name)==digest,'Final dependency differs from v3 snapshot: '+name)
        final_dependencies[name]=digest
    value={'round':7,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'formulas':FORMULAS,'thresholds':THRESHOLDS,'parameters':component.parameters(),'feature_count':95,'categorical_count':23,'source_sha256':sources(),'dependency_sha256':deps,'stage_dependency_sha256':stages,'final_dependency_sha256':final_dependencies,'final_contract':old['final_contract'],'splits':old['splits'],'split_contract':old['split_contract'],'target_group_weights':old['target_group_weights'],'quarter_definition':old['quarter_definition'],'old_plan_sha256':sha(c6.OUT/'candidate_plan.json'),'v3_snapshot_sha256':SNAPSHOT_SHA,'single_development_candidate':True,'tuning_or_weight_search':False,'public_score_used_for_selection':False,'historically_untouched':False,'external_property_answers_used':False,'fit_uses_eval_set':False,'early_stopping_used':False,'runtime':runtime_versions(),'registry_sha256_at_plan':sha(ROOT/'runs/versions.json')}
    c6.verify_hashes(deps)
    locked_write(OUT/'candidate_plan.json',value)
    print(json.dumps({'status':'plan_frozen','plan_sha256':sha(OUT/'candidate_plan.json'),'candidate':CANDIDATE}),flush=True)

def plan():
    value=locked(OUT/'candidate_plan.json')
    require(value['round']==7 and value['candidate']==CANDIDATE and value['control']==CONTROL,'Wrong R7 plan')
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
        require(spec[label+'_indices_content_sha256']==value_sha(array,'<i8'),'R7 partition index differs')
    base=c6.baseline_folder(stage);old=c6.OUT/'models'/stage/'geography_catboost'
    e6.verify_indices(base,fi,qi,'check_indices.npy');e6.verify_indices(old,fi,qi,'query_indices.npy')
    return data,X,y,Xq,yq,fi,qi

def publish(staging,destination):
    require(staging.resolve().parent==destination.resolve().parent and staging.resolve().is_relative_to(OUT.resolve()),'Unsafe R7 publication path')
    require(not destination.exists(),'Refusing to replace recorded experiment')
    staging.rename(destination)

def fit(stage):
    specification=authorize(stage);destination=OUT/'models'/stage
    require(not destination.exists(),'Model already exists; do not retrain')
    data,X,y,Xq,yq,fi,qi=partition(stage);del yq
    geo,geoq,manifest=load_geographic_partition(stage,fi,qi)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    started=time.monotonic();cats=data['categorical_columns']
    print(json.dumps({'event':'fit_started','stage':stage,'fit_rows':len(X),'query_rows':len(Xq),'features':95,'parameters':specification['parameters']}),flush=True)
    model,state,metadata=component.fit(X,y,geo,cats)
    predictions=component.predict(model,state,Xq,geoq,cats)
    model.save_model(str(staging/'model.txt'));write(staging/'feature_state.json',state)
    reloaded=lgb.Booster(model_file=str(staging/'model.txt'))
    replay=component.predict(reloaded,read(staging/'feature_state.json'),Xq.iloc[:64],geoq.iloc[:64],cats)
    require(np.allclose(replay,predictions[:64],rtol=1e-12,atol=1e-5),'Saved model reload differs')
    require(reloaded.current_iteration()==3999,'Unexpected LightGBM iteration count')
    require(predictions.shape==(len(Xq),) and np.isfinite(predictions).all() and (predictions>=0).all(),'Invalid TWD predictions')
    for name,values in [('fit_indices',fi),('query_indices',qi),('predictions',predictions)]:np.save(staging/(name+'.npy'),values,allow_pickle=False)
    metadata.update({'round':7,'stage':stage,'component':'geography_unit_lightgbm','created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'source_sha256':specification['source_sha256'],'outer_fit_rows':len(X),'query_rows':len(Xq),'fit_target_sha256':value_sha(y,'<f8'),'fit_X_sha256':frame_sha(X),'query_X_sha256':frame_sha(Xq),'fit_geo_sha256':frame_sha(geo),'query_geo_sha256':frame_sha(geoq),'geo_manifest_sha256':sha(c6.OUT/'geography'/stage/'manifest.json'),'artifact_sha256':{p.name:sha(p) for p in staging.iterdir() if p.is_file()},'reload_64_rows_verified':True,'query_labels_used':False,'prediction_units':'TWD','direct_component_score_omitted':True,'runner_elapsed_seconds':time.monotonic()-started,'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json')})
    locked_write(staging/'metadata.json',metadata)
    authorize(stage);publish(staging,destination)
    print(json.dumps({'event':'fit_completed','stage':stage,'elapsed_seconds':time.monotonic()-started}),flush=True)

def blend(old,geo,unit,reference,Xq,apply_reference=predict_reference):
    base=.75*(.5*np.asarray(old)+.5*np.asarray(geo))+.25*np.asarray(unit)
    prediction=np.asarray(apply_reference(reference,Xq,base,variant='trend'),dtype='float64')
    require(prediction.shape==(len(Xq),) and np.isfinite(prediction).all() and (prediction>=0).all(),'Invalid blended TWD predictions')
    return prediction

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

def model_check(stage,X,y,Xq,fi,qi,geo,geoq,cats,specification):
    folder=OUT/'models'/stage;meta=locked(folder/'metadata.json')
    require(meta['round']==7 and meta['stage']==stage and meta['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong model identity')
    require(meta['source_sha256']==specification['source_sha256'],'Model source closure differs')
    require(meta['parameters']==specification['parameters'] and meta['feature_count']==95 and meta['categorical_feature_count']==23,'Model configuration differs')
    require(meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False and meta['category_maps_fit_only'] is True,'Fit contract differs')
    require(meta['fit_transformed_target_values_sha256']==value_sha(y/price_factor(X,'unit'),'<f8'),'Transformed fit target differs')
    require(meta['fit_sample_weight_values_sha256']==value_sha(effective_area(X)/100,'<f8'),'Fit sample weights differ')
    require(meta['outer_fit_rows']==len(X) and meta['query_rows']==len(Xq),'Model sample counts differ')
    require(meta['fit_target_sha256']==value_sha(y,'<f8'),'Fit target identity differs')
    for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]:require(meta[key]==frame_sha(frame),'Model feature identity differs: '+key)
    require(meta['geo_manifest_sha256']==sha(c6.OUT/'geography'/stage/'manifest.json'),'Model map version differs')
    require(meta['reload_64_rows_verified'] is True and meta['query_labels_used'] is False and meta['prediction_units']=='TWD','Model training/prediction contract differs')
    expected_freeze=None if stage=='development' else sha(OUT/'frozen_candidate.json')
    require(meta['frozen_candidate_sha256']==expected_freeze,'Model selection identity differs')
    for name,digest in meta['artifact_sha256'].items():
        require(Path(name).name==name and sha(folder/name)==digest,'Saved model artifact changed')
    e6.verify_indices(folder,fi,qi,'query_indices.npy')
    saved=e6.vector(folder/'predictions.npy',len(qi))
    model=lgb.Booster(model_file=str(folder/'model.txt'))
    require(model.current_iteration()==3999,'Reloaded model iteration count differs')
    state=read(folder/'feature_state.json')
    require(component.fit_feature_state(X,geo,cats)==state,'Saved category maps differ from exact fit-only state')
    replay=component.predict(model,state,Xq,geoq,cats)
    require(np.allclose(saved,replay,rtol=1e-12,atol=1e-5),'All-query model replay differs')
    return saved,{rel(p):sha(p) for p in folder.iterdir() if p.is_file()}

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
    new,inputs=model_check(stage,X,y,Xq,fi,qi,geo,geoq,data['categorical_columns'],specification)
    inputs.update(specification['stage_dependency_sha256'][stage])
    reference,_=c6.load_reference(stage,X);base=c6.baseline_folder(stage)
    old=e6.vector(base/'catboost_raw.npy',len(qi));unit=e6.vector(base/'lightgbm_l1_unit63.npy',len(qi))
    geographic=e6.vector(c6.OUT/'models'/stage/'geography_catboost/predictions.npy',len(qi))
    original=e6.vector(c6.OUT/'evaluation'/stage/'geography_blend.npy',len(qi))
    replay=blend(old,geographic,unit,reference,Xq)
    require(np.allclose(original,replay,rtol=1e-12,atol=1e-5),'v3 control does not replay')
    predictions={CONTROL:original,CANDIDATE:blend(old,geographic,new,reference,Xq)}
    metrics={name:score(yq,pred,X,Xq,c6.SPLITS[stage][0],weights) for name,pred in predictions.items()}
    previous=locked(c6.OUT/'evaluation'/stage/'results.json')
    require(metrics[CONTROL]==previous['metrics']['geography_blend'],'Recomputed v3 metrics differ from archived result')
    decision=decide(stage,metrics)
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    for name,values in predictions.items():np.save(staging/(name+'.npy'),values,allow_pickle=False)
    outputs={rel(destination/p.name):sha(p) for p in staging.iterdir() if p.is_file()}
    result={'round':7,'stage':stage,'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json'),'source_sha256':specification['source_sha256'],'input_artifact_sha256':inputs,'output_artifact_sha256':outputs,'formulas':FORMULAS,'thresholds':THRESHOLDS,'target_group_weights':weights,'metrics':metrics,'decision':decision,'v3_control_replayed_all_rows':True,'new_model_replayed_all_rows':True,'fit_rows':len(fi),'query_rows':len(qi),'evaluated_candidates':[CONTROL,CANDIDATE],'candidate_component_separately_scored':False,'query_labels_used_for_training':False,'public_score_used':False,'historically_untouched':False,'new_public_score_known':False,'private_score_known':False}
    locked_write(staging/'results.json',result)
    authorize(stage);publish(staging,destination)
    if stage=='development':freeze_development(result)
    print(json.dumps({'stage':stage,'metrics':{k:{n:v[n] for n in ['overall_mae','weighted_mae']} for k,v in metrics.items()},'decision':decision}),flush=True)
    return result

def finalize():
    specification=plan();frozen=locked(OUT/'frozen_candidate.json');development=read_result('development')
    require(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json') and frozen['passed']==development['decision']['passed'],'Development freeze differs')
    results={'development':development}
    if frozen['passed']:
        for stage in ('confirmation','auxiliary'):
            authorize(stage);results[stage]=read_result(stage)
    accepted=all(r['decision']['passed'] for r in results.values()) and len(results)==3
    path=OUT/'accepted_candidate.json'
    result={'created_at_utc':now(),'round':7,'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),'accepted':accepted,'candidate':CANDIDATE,'deployment_decision':CANDIDATE if accepted else 'keep_v3','result_sha256':{s:sha(OUT/'evaluation'/s/'results.json') for s in results},'unevaluated_stages':[s for s in STAGES if s not in results],'no_reselection_or_parameter_change':True,'new_submission_created':False,'private_score_known':False,'source_sha256':specification['source_sha256']}
    if path.exists():
        saved=locked(path);require(all(saved[k]==v for k,v in result.items() if k!='created_at_utc'),'Existing final decision differs');result=saved
    else:locked_write(path,result)
    print(json.dumps(result),flush=True)
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
    calls=[];query=[object(),object()]
    def fake(reference,X,values,variant):
        require(X is query and reference=='fit_only' and variant=='trend','Reference call changed');calls.append(values.copy());return values*1.1
    result=blend([1,2],[3,4],[5,6],'fit_only',query,fake)
    run('fixed_weights_and_single_trend',len(calls)==1 and np.allclose(result,1.1*(.375*np.array([1,2])+.375*np.array([3,4])+.25*np.array([5,6]))))
    proof={'created_at_utc':now(),'all_checks_passed':True,'synthetic_checks':checks,'component_synthetic_checks':component_checks,'source_sha256':sources(),'real_model_training_or_scoring':False,'parameters':component.parameters()}
    path=OUT/'audit/preflight.json';require(not path.exists(),'Preflight exists; preserve audit')
    write(path,proof);print(json.dumps(proof),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check','plan','fit','evaluate','finalize'])
    parser.add_argument('--stage',choices=STAGES,default='development')
    args=parser.parse_args()
    if args.command=='self-check':self_check()
    elif args.command=='plan':plan_create()
    elif args.command=='fit':fit(args.stage)
    elif args.command=='evaluate':evaluate(args.stage)
    else:finalize()
