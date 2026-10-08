"""Independent CPU four-origin R116 postfit audit, only after formal evaluate0.

--self-check reads only artificial metrics. --evaluation-complete is invoked
once by the coordinator after confirmation. No fit/GPU/CSV/selection; never
overwrite or automatically retry. All four complete native folders and layers
are verified before raw query prices are deserialized.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import round100_contract as c
import hashlib
import round116_depthwise_unit as helper

FOLDER='outputs/research_round116_20261008/'
PRIMITIVE='outputs/research_continue_20261008/pipeline_review/audit_primitives.py'
DRIVER='src/round116_research.py'
PLAN='reports/round116_plan.md'
LAYERS={'native','base','reference','spatial','median'}


def same(a,b,name):c.need(a==b,name)


def locked(name):
    path=c.safe(name)
    same(path.with_suffix('.sha256').read_text(encoding='ascii').strip(),c.sha(path),'Receipt sidecar '+name)
    return c.read(path),c.record(path)


def arithmetic():
    spec=importlib.util.spec_from_file_location('r116_independent_math',c.safe(PRIMITIVE))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def comparison(control,candidate,is_offset,secondary=False):
    name='v24' if secondary else 'v22'
    result=arithmetic().manual_comparison(control,candidate,name)
    if is_offset:
        result.pop('stage_research_gate_passed',None);result.pop('used_for_research_gate',None)
        result['automatic_adoption']=False
    result.update(candidate='R116',used_for_release_gate=not secondary)
    return result


def extra_gate(metrics,compared):
    parts={'overall_regression_at_most_0_25pct':metrics['T']['overall_mae']<=1.0025*metrics['v22']['overall_mae'],
           'weighted_regression_at_most_0_25pct':metrics['T']['weighted_mae']<=1.0025*metrics['v22']['weighted_mae'],
           'no_quarter_or_large_group_above_5pct':not compared['quarter_or_large_group_deterioration_above_5pct']}
    return {'components':parts,'passed':all(parts.values()),
            'both_point_metrics_improve':all(metrics['T'][k]<metrics['v22'][k] for k in ('overall_mae','weighted_mae'))}


def effective_area(X):
    building=np.asarray(X['building_area'],dtype=np.float64)
    parking=np.asarray(X['parking_area'],dtype=np.float64)
    area=np.where(np.isfinite(building)&(building>0),building,np.where(np.isfinite(parking)&(parking>0),parking,1.))
    c.need(area.shape==(len(X),) and np.isfinite(area).all() and (area>0).all(),'Independent effective-area fallback')
    return area


def unit_inverse(raw,native,X):
    c.need(raw.dtype==native.dtype==np.float64 and raw.shape==native.shape==(len(X),) and len(X)>0
           and np.isfinite(raw).all() and np.isfinite(native).all(),'Raw unit/native schema')
    restored=raw*(effective_area(X)*1e4)
    c.need(np.isfinite(restored).all(),'Independent unit inverse finite')
    c.exact(native,np.maximum(restored,0.),'Independent exactly one unit-to-TWD inverse and projection')
    return {'raw_units':'unit','raw_negative_unit_rows':int((raw<0).sum()),
            'raw_unit_min':float(raw.min()),'raw_unit_max':float(raw.max()),
            'native_TWD_min':float(native.min()),'native_TWD_max':float(native.max()),
            'native_zero_rows':int((native==0).sum()),'inverse_applications_per_prediction':1,
            'nonnegative_projection_after_TWD_reconstruction':True,
            'query_y_used_for_diagnostics':False,'extra_price_cap_applied':False}


def leaf_identity(leaves):
    leaves=np.asarray(leaves)
    c.need(leaves.dtype.kind in 'iu' and leaves.shape==(2999,) and (leaves>0).all() and (leaves<=256).all(),
           'Independent depth-8 nonsymmetric leaf bounds, no power-of-two requirement')
    values,counts=np.unique(leaves,return_counts=True)
    return {'minimum_tree_leaf_count':int(leaves.min()),'maximum_tree_leaf_count':int(leaves.max()),
            'tree_leaf_counts_count':len(leaves),'tree_leaf_counts_distribution':{str(int(v)):int(n) for v,n in zip(values,counts)},
            'tree_leaf_counts_sha256':hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest()}


def check_stage(origin,record,request,driver):
    folder=c.safe(f'outputs/round116/{origin}');model=folder/'model';nf,nq=driver.COUNTS[origin]
    same(record,c.record(folder/'stage.json'),'Fixed stage record')
    item=c.read(record['path']);same((item['stage'],item['candidate_round'],item['fit_rows'],item['query_rows']),
                                  (origin,116,nf,nq),'Fixed stage/counts')
    c.need(item['query_labels_available'] is False and item['query_scores_computed'] is False
           and item['original_v22_v24_and_unit42_slot_replayed_exact'] is True,'Unscored stage/slot scope')
    same(item['training_log'],c.record(folder/'train.log'),'Actual stage training log')
    files=sorted(model.iterdir())
    required={'model.cbm','started.json','metadata.json','feature_state.json','raw.npy','native.npy'}
    c.need({p.name for p in files}==required and all(p.is_file() and not p.is_symlink() for p in files),
           'Exact complete flat six-file native artifact directory')
    same(item['sealed_models'],[c.record(p) for p in files],'Exact complete native artifact membership/SHA')
    same(item['model_metadata'],c.record(model/'metadata.json'),'Fixed actual metadata record')
    meta=c.read(model/'metadata.json')
    driver.check_metadata(meta,request,origin,{'fit_input_identity':item['fit_input_identity']})
    same(item['context_before'],item['context_after'],'Saved fit/query state unchanged')
    for name,key in (('model.cbm','model_sha256'),('feature_state.json','feature_state_sha256'),
                     ('raw.npy','raw_prediction_sha256'),('native.npy','prediction_sha256')):
        same(c.sha(model/name),meta[key],'Direct native artifact SHA '+name)
    started=c.read(model/'started.json');c.need(all(meta.get(k)==v for k,v in started.items()),'Started/completed metadata identity')
    state=c.read(model/'feature_state.json')
    restored=helper.CatBoostRegressor();restored.load_model(str(model/'model.cbm'))
    actual=helper.check_model(restored,state)
    same(actual,meta['reloaded_native_diagnostics'],'Actual saved native CBM diagnostics')
    same(helper.compare_reload_diagnostics(meta['fit_native_diagnostics'],actual),meta['reload_diagnostics_comparison'],
         'Recorded fit/reload serialization comparison')
    for key,value in meta['fit_native_diagnostics'].items():same(meta[key],value,'Top-level fitted native diagnostic '+key)
    for key,value in leaf_identity(restored.get_tree_leaf_counts()).items():same(actual[key],value,'Independent saved native leaf '+key)
    raw=np.load(model/'raw.npy',allow_pickle=False);native=np.load(model/'native.npy',allow_pickle=False)
    c.need(raw.dtype==native.dtype==np.float64 and raw.shape==native.shape==(nq,)
           and np.isfinite(raw).all() and np.isfinite(native).all() and (native>=0).all(),'Raw/native schema before labels')
    same(item['raw_unit_prediction'],{**c.record(model/'raw.npy'),**c.identity(raw)},'Raw unit full-array identity')
    same(set(item['layers']),LAYERS,'All five layers')
    layers={key:c.load_array(rec) for key,rec in item['layers'].items()}
    for key,a in layers.items():
        c.need(a.dtype==np.float64 and a.shape==(nq,) and np.isfinite(a).all() and (a>=0).all(),'Complete TWD layer '+key)
    c.exact(layers['native'],native,'Model/native-layer identity')
    return item,layers,raw,meta


def exact_weight_hash(values,expected,name):
    same(c.values_sha(values),expected,'Weight vector SHA remains exact '+name)


def summary_comparison(actual,saved,name):
    # Values/dtype/shape SHA are exact before this descriptive comparison.
    # np.dot reduction can round differently across BLAS contexts. Permit only
    # sum_squares/its derived ESS, capped at 128 symmetric float64 ULP.
    same(set(actual),set(saved),'Weight summary exact keys '+name)
    differences=[]
    for key in actual:
        left,right=actual[key],saved[key]
        same(type(left),type(right),'Weight summary scalar type '+name+'.'+key)
        if left==right:continue
        c.need(key in {'sum_squares','effective_sample_size'} and type(left) is float
               and np.isfinite(left) and np.isfinite(right) and left>0 and right>0,
               'Only descriptive dot/derived ESS may round '+name+'.'+key)
        spacing=max(abs(float(np.spacing(left))),abs(float(np.spacing(right))))
        ulp=abs(left-right)/spacing
        c.need(ulp<=128.,'Weight summary exceeds 128 float64 ULP '+name+'.'+key)
        differences.append({'field':key,'saved':right,'recomputed':left,'absolute_difference':abs(left-right),
                            'relative_difference':abs(left-right)/max(abs(left),abs(right)),
                            'symmetric_float64_ULP_difference':ulp})
    return {'status':'passed','allowed_fields':['sum_squares','effective_sample_size'],
            'maximum_allowed_symmetric_float64_ULP':128,'all_other_fields_exact':True,
            'full_actual_summary':actual,'full_saved_summary':saved,'rounding_differences':differences}


def check_native(origin,meta,loaded,request,driver,stage,raw,native):
    prefix=f'outputs/round116/{origin}/model/'
    driver.check_metadata(meta,request,origin,loaded)
    same(stage['fit_input_identity'],loaded['fit_input_identity'],'Actual fit/query frame, target and weight identities')
    same(stage['prepared_array_identities'],loaded['array_identities'],'Actual saved date/target inputs')
    X,Xq,G,Gq=(loaded[k] for k in ('fit_X','query_X','fit_geo','query_geo'))
    state=c.read(prefix+'feature_state.json')
    helper._check_state(state,X,loaded['cats']);helper._check_state(state,Xq,loaded['cats'])
    area=effective_area(X);c.exact(area,helper.original.effective_area(X),'Independent effective fit area')
    dates=loaded['arrays']['train_sample_weight'];y=loaded['fit_y']
    c.need(dates.dtype==np.float32 and dates.shape==y.shape==(len(X),) and np.isfinite(dates).all() and (dates>0).all(),'Actual original saved R63 dates')
    weights=(area/100.)*dates.astype(np.float64)
    target=y/(area*1e4)
    exact_weight_hash(dates,meta['fit_date_weights_sha256'],'date')
    exact_weight_hash(weights,meta['fit_sample_weights_sha256'],'pool')
    same(helper.original.values_sha256(target),meta['fit_transformed_target_values_sha256'],'Independent unit fit target')
    same(helper.original.values_sha256(area/100.),meta['fit_area_weights_sha256'],'Independent original area weights')
    same(helper.original.values_sha256(weights),meta['fit_sample_weights_sha256'],'Independent unnormalized date-times-area pool weights')
    summary_checks={'date':summary_comparison(helper.baseline._summary(dates),meta['fit_date_weights_summary'],'date'),
                    'pool':summary_comparison(helper.baseline._summary(weights),meta['fit_sample_weights_summary'],'pool')}
    same(meta['feature_columns'],state['feature_columns'],'Actual feature schema')
    same(meta['categorical_columns'],state['categorical_columns'],'Actual augmented category schema')
    frame=helper.original.augment_features(Xq,Gq)
    same(list(frame),state['feature_columns'],'Actual full-query feature order')
    same({k:str(frame[k].dtype) for k in frame},state['feature_dtypes'],'Actual full-query feature dtypes')
    model=helper.CatBoostRegressor();model.load_model(str(c.safe(prefix+'model.cbm')))
    actual=helper.check_model(model,state);same(actual,meta['reloaded_native_diagnostics'],'CPU-loaded actual native configuration')
    leaves=leaf_identity(model.get_tree_leaf_counts())
    for key,value in leaves.items():same(actual[key],value,'Independent native leaves '+key)
    parameters=model.get_all_params()
    same(actual['native_parameters'],parameters,'Actual full native/default parameters retained')
    same(actual['native_derived_parameter_names'],sorted(set(parameters)-set(request['recipe']['parameters'])),'Actual derived defaults exact names')
    same(actual['native_checked_passed_parameter_names'],sorted(set(request['recipe']['parameters'])-set(helper.NATIVE_RUNTIME_ONLY_PARAMETER_NAMES)),
         'All recoverable passed parameters checked')
    actual_raw=np.asarray(model.predict(helper.Pool(frame,cat_features=state['categorical_columns'],thread_count=4),
                                        thread_count=4,task_type='CPU'),dtype=np.float64)
    c.exact(actual_raw,raw,'Independent CPU full-query native raw unit predictions')
    diagnostics=unit_inverse(actual_raw,native,Xq)
    same(diagnostics,stage['unit_inverse_diagnostics'],'Independent effective-area inverse-once diagnostics')
    c.exact(helper.original.inverse_transform(actual_raw,Xq),native,'Original helper agrees with independent inverse')
    driver.verify_context(loaded,origin)
    return {'model':c.record(prefix+'model.cbm'),'feature_state':c.record(prefix+'feature_state.json'),
            'CPU_native_model_loaded':True,'full_query_CPU_raw_unit_prediction_exact':True,
            'full_query_TWD_inverse_and_projection_exact':True,'inverse_applications_per_prediction':1,
            'leaf_identity':leaves,'native_parameters':parameters,'weight_summary_rounding_comparison':summary_checks,
            'runtime_only_parameters_not_native_recoverable':actual['runtime_parameters_not_recovered_from_native_model'],
            'derived_parameter_names':actual['native_derived_parameter_names']}


def check_bootstrap(origin,item,point,stage_index):
    pair=item['paired'];same((pair['control'],pair['candidate'],pair['difference'],pair['seed'],pair['replicates'],pair['stage_index']),
        ('v22','R116','R116 minus v22',20260925,2000,stage_index),'Correct paired labels/origin seed')
    c.need('not Private probability' in pair['caution'] and 'not matched refits' in pair['caution'],'Uncertainty caution')
    records=[];same(set(item['paired_samples']),{'project','month'},'Both conditional bootstrap families')
    for family in ('project','month'):
        rec=item['paired_samples'][family]
        same(rec['path'],f'outputs/round116/{origin}/evaluation/{family}_samples.npy','Fixed paired replicate path')
        samples=c.load_array(rec)
        c.need(samples.shape==(2000,4) and samples.dtype==np.float64 and not np.isinf(samples).any(),'Saved replicates schema')
        for metric,left,right in (('overall',0,1),('weighted',2,3)):
            delta=samples[:,right]-samples[:,left];valid=delta[np.isfinite(delta)]
            quant=np.quantile(valid,[.025,.975],method='linear') if len(valid) else [None,None]
            expected={'valid_replicates':len(valid),'invalid_replicates':2000-len(valid),'control_mae':float(point[left]),
                      'candidate_mae':float(point[right]),'point_delta':float(point[right]-point[left]),
                      'percentile_2_5':quant[0],'percentile_97_5':quant[1],
                      'negative_delta_fraction':float(np.mean(valid<0)) if len(valid) else None,
                      'bootstrap_mean_delta':float(valid.mean()) if len(valid) else None}
            for key,v in expected.items():same(pair['methods'][family]['summary'][metric][key],v,'Independent bootstrap '+key)
        records.append(c.record(rec['path']))
    return records


def run():
    tick=time.monotonic();output=FOLDER+'postfit_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(),'Preserve old audit attempts')
    deps={name:c.sha(name) for name in (c.rel(__file__),PRIMITIVE)}
    import round116_research as driver
    request,rr=locked(driver.REQUEST);seal,sr=locked(driver.SEAL)
    training,tr=locked(FOLDER+'training_execution.json');evaluation,er=locked(FOLDER+'evaluation_execution.json')
    same((training['exit_code'],training['failure'],training['request']),(0,None,rr),'Successful separate four-fit execution')
    same((evaluation['exit_code'],evaluation['failure'],evaluation['request'],evaluation['seal']),(0,None,rr,sr),'Successful separate evaluation')
    same(evaluation['training_execution'],tr,'Evaluation/training exact binding')
    c.need(evaluation['query_labels_used_for_scoring_only'] and evaluation['training_already_finished_before_scoring']
           and evaluation['new_fits']==0 and evaluation['new_CSV'] is False,'Evaluation scope')
    same(request['status'],'frozen_before_four_fixed_origin_fits','Fixed request state')
    same(request['new_fits'],4,'Exactly four unconditional fits')
    same(request['origin_order'],list(driver.ORIGINS),'Four-origin fixed order')
    same(request['origin_counts'],{s:list(v) for s,v in driver.COUNTS.items()},'Four-origin actual row contracts')
    c.need(request['selected_submission']=='v22' and not request['new_CSV'] and not request['full_data_fit']
           and not request['query_scores_computed'] and not request['query_labels_used_for_fit']
           and request['Public_score'] is None and request['Private_score'] is None,'Request selection/scores/labels scope')
    same(request['plan'],c.record(PLAN),'Frozen scientific plan')
    same(driver.sources(),request['source_sha256'],'Exact complete recursive source closure')
    merged={**request['source_sha256'],PLAN:request['plan']['sha256']}
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(request['protected_before'])
    same(driver.protected(),request['protected_before'],'Protected source/model/CSV full membership/SHA')
    preflight,pfr=locked(FOLDER+'preflight_review.json')
    same(preflight['status'],'passed','Independent source preflight passed');same(preflight['source_sha256'],merged,'Exact preflight source+plan')
    same(seal['status'],'four_models_all_raw_unit_TWD_and_layers_sealed_unscored','Every raw unit/native/layer sealed')
    same(seal['request'],rr,'Seal/request binding');same(seal['query_scores_computed'],False,'Seal unscored')
    same(set(seal['records']),set(driver.ORIGINS),'Every origin sealed')
    same(seal['origin_order'],list(driver.ORIGINS),'Sealed fixed order');same(seal['new_fits'],4,'All four sealed fits')
    for name,(entry,environment,flag) in driver.formal_checks().items():
        formal,_=locked(FOLDER+name)
        same(formal['exit_code'],0,'CPU selfcheck exit0')
        same(formal['command'],[str(ROOT/environment/'Scripts/python.exe'),'-B','-X','utf8',str(c.safe(entry)),flag],'CPU exact command')
        same(formal['source_sha256'],c.source_hashes(entry),'CPU exact source closure')
    checked={origin:check_stage(origin,seal['records'][origin],request,driver) for origin in driver.ORIGINS}
    # All four completion/native CBM/prediction/layer identities pass
    # before raw query labels are deserialized below.
    results,rs=locked(driver.RESULTS)
    same(results['status'],'four_reused_fixed_origins_evaluated','Completed fixed-origin results')
    for key,value in (('request',rr),('seal',sr),('plan',request['plan']),('source_sha256',request['source_sha256'])):
        same(results[key],value,'Results frozen identity '+key)
    same(results['origin_order'],list(driver.ORIGINS),'Results four origins')
    same(set(results['stages']),set(driver.ORIGINS),'Exactly four results stages')
    c.need(results['selected_submission']=='v22' and not results['new_CSV'] and not results['full_data_fit']
           and results['Public_score'] is None and results['Private_score'] is None and not results['fresh_blind_validation'],'Result role/scope')
    import pandas as pd
    raw=pd.read_pickle(c.safe(c.RAW))
    artifacts,trainmeta,native_facts,summary,boots={},{},{},{},[]
    score_count=0
    for origin in driver.ORIGINS:
        loaded=driver.load_context(request,origin);st,layers,signed,meta=checked[origin]
        same(st['prepared_array_identities'],loaded['array_identities'],'Actual original date-weight/fit-target array identities')
        same(st['context_before'],loaded['before'],'Actual saved origin state/frames/indices/fixed predictions')
        native_facts[origin]=check_native(origin,meta,loaded,request,driver,st,signed,layers['native'])
        ctx=loaded['context'];is_offset=origin==driver.OFFSET
        replay=driver.offset_comp.chain_unit(ctx,layers['native']) if is_offset else driver.history.unit_chain(ctx,layers['native'])
        for key in LAYERS:c.exact(replay[key],layers[key],'Actual complete saved candidate chain '+key)
        metric=driver.offset_metric if is_offset else driver.metric
        y=metric.select_query_y(raw,loaded['indices']['query_source_cache_indices'],455784) if is_offset else metric.select_query_y(raw,loaded['indices']['query_source_cache_indices'])
        descriptive=(lambda a:metric.descriptive_metrics(y,a,loaded['fit_X'],loaded['query_X'])) if is_offset else (lambda a:metric.descriptive_metrics(y,a,loaded['fit_X'],loaded['query_X'],c.SPLITS[origin][0]))
        item=results['stages'][origin];stage_eval,stage_record=locked(f'outputs/round116/{origin}/evaluation/results.json')
        same(stage_eval,item,'Stage/aggregate evaluation identity');same(item['sealed_stage'],seal['records'][origin],'Evaluation/stage exact binding')
        labels=metric.methods.group_labels(loaded['fit_X'],loaded['query_X']);weights=metric.WEIGHTS if is_offset else c.WEIGHTS
        vectors={'v22':loaded['v22'],'v24':loaded['v24'],'T':layers['median']}
        for key,prediction in vectors.items():
            target=item['metrics'][key];errors=np.abs(prediction-y)
            same(float(errors.mean()),target['overall_mae'],'Independent origin MAE')
            weighted=0.
            for group,weight in weights.items():
                take=labels==group;same(int(take.sum()),target['strata'][group]['rows'],'Fixed group support')
                if take.any():same(float(errors[take].mean()),target['strata'][group]['mae'],'Independent group MAE')
                if weight>0:
                    c.need(take.any(),'Positive-weight group absent');weighted+=weight*float(errors[take].mean())
            same(weighted,target['weighted_mae'],'Independent fixed weighted MAE')
            same(descriptive(prediction),target,'Every descriptive group/band/tail')
            if key!='T':
                old=(c.read('outputs/round97/evaluation/results.json')['metrics'][key] if is_offset else
                     c.read(f"outputs/round{79 if key=='v22' else 83}/results.json")['stages'][origin]['metrics']['R79' if key=='v22' else 'R83'])
                same({k:target[k] for k in old},old,'Original v22/v24 baseline metric authority')
            score_count+=1
        primary=comparison(item['metrics']['v22'],item['metrics']['T'],is_offset)
        secondary=comparison(item['metrics']['v24'],item['metrics']['T'],is_offset,True)
        same(primary,item['comparison'],'Independent complete primary group comparison')
        same(secondary,item['comparison_vs_v24'],'Independent complete v24 counterevidence')
        for key,a in layers.items():
            if key!='median':same(descriptive(a),item['branch_metrics'][key],'All four branch metrics')
        unprojected=signed*(effective_area(loaded['query_X'])*1e4)
        same(item['raw_unit_diagnostic_only'],{**unit_inverse(signed,layers['native'],loaded['query_X']),
             'unprojected_TWD_overall_mae':float(np.abs(unprojected-y).mean())},'Independent raw-unit inverse/unprojected TWD MAE')
        c.need((np.abs(layers['native']-y)<=np.abs(unprojected-y)).all(),'Positive truth: nonnegative projection cannot increase native absolute error')
        same(item['both_point_metrics_improve'],all(item['metrics']['T'][k]<item['metrics']['v22'][k] for k in ('overall_mae','weighted_mae')),
             'Strict both-point-improvement label')
        point=metric.methods.point_values(np.abs(y[:,None]-np.column_stack([vectors['v22'],vectors['T']])),labels,weights)
        boots.extend(check_bootstrap(origin,item,point,0 if is_offset else c.STAGES.index(origin)))
        artifacts[origin]={'stage':seal['records'][origin],'training_metadata':st['model_metadata'],
                           'evaluation':stage_record,'sealed_models':st['sealed_models'],'raw_unit_prediction':st['raw_unit_prediction']}
        trainmeta[origin]=meta
        summary[origin]={'overall_mae':item['metrics']['T']['overall_mae'],'weighted_mae':item['metrics']['T']['weighted_mae'],
                         'overall_delta_vs_v22':primary['overall_mae']['delta'],'weighted_delta_vs_v22':primary['weighted_mae']['delta'],
                         'overall_delta_vs_v24':secondary['overall_mae']['delta'],'weighted_delta_vs_v24':secondary['weighted_mae']['delta'],
                         'adverse_groups':primary['quarter_or_large_group_deterioration_above_5pct'],
                         'v24_adverse_groups':secondary['quarter_or_large_group_deterioration_above_5pct'],
                         'unit_inverse_diagnostics':unit_inverse(signed,layers['native'],loaded['query_X']),'fit_seconds':meta['fit_seconds']}
        driver.verify_context(loaded,origin)
    del raw
    manual=arithmetic().manual_gates({s:results['stages'][s] for s in c.STAGES})
    augmented=results['gates']['august']
    same(manual['components'],augmented['exploratory_submission']['components'],'Independent August exploratory components')
    same(manual['exploratory_submission_eligible'],augmented['exploratory_submission']['passed'],'Independent August exploratory pass')
    same(manual['strong_components'],augmented['promotion']['components'],'Independent August strong components')
    same(manual['strong_research_gate_passed'],augmented['promotion']['adoption_research_gate_passed'],'Independent August strong pass')
    same(manual['confirmation_fractional_improvement'],augmented['promotion']['confirmation_fractional_improvement'],'Latest improvement')
    offset_item=results['stages'][driver.OFFSET];extra=extra_gate(offset_item['metrics'],offset_item['comparison'])
    same({k:extra[k] for k in ('components','passed')},results['gates']['extra_origin'],'Independent extra-origin gate')
    verdicts={'august':manual,'extra_origin':extra}
    for name,first in (('combined_exploratory_submission',manual['exploratory_submission_eligible']),
                       ('combined_promotion',manual['strong_research_gate_passed'])):
        parts={'original_three_windows':first,'fixed_extra_origin':extra['passed']}
        verdicts[name]={'components':parts,'passed':all(parts.values())}
        same(verdicts[name],results['gates'][name],'Independent final combined gate')
    c.need(results['gates']['selected_submission']=='v22' and not results['gates']['automatic_selection']
           and results['gates']['private_score'] is None and not results['gates']['new_CSV'],'Gate selection scope')
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(deps)
    same(driver.protected(),request['protected_before'],'Protected full membership/content unchanged after audit')
    for origin in driver.ORIGINS:check_stage(origin,seal['records'][origin],request,driver)
    payload={'status':'passed','reviewer':'research_preflight_20261008','created_at_utc':c.now(),'round':116,
             'scope':'Independent CPU post-evaluation full-query CatBoost/raw-unit/inverse/complete-composition/metrics/gates audit; accuracy gate recorded separately.',
             'source_sha256':merged,'plan':request['plan'],'request':rr,'seal':sr,'training_execution':tr,
             'evaluation_execution':er,'results':rs,'preflight_review':pfr,'review_tool':c.record(__file__),
             'prior_failed_audit_attempt':c.record(FOLDER+'independent_postfit_execution.json'),
             'summary_discrepancy_diagnosis':c.record(FOLDER+'attempt1_summary_discrepancy.json'),
             'pool_summary_discrepancy_diagnosis':c.record(FOLDER+'attempt1_pool_summary_discrepancy.json'),
             'review_dependency_source_sha256':deps,'artifacts':artifacts,'trainmeta':trainmeta,
             'manual_gate_verdicts':verdicts,'stage_summary':summary,'native_identity_diagnostics':native_facts,
             'identity_checks':{'frozen_sources_including_plan':len(merged),'frozen_inputs':len(request['input_sha256']),
                                'protected_files':len(request['protected_before']),'sealed_origins':4,
                                'native_CBM_models_CPU_loaded_and_full_query_inferred':4,'complete_chain_layers_replayed':20,
                                'unit_inverse_and_complete_native_metadata_sets_verified':4},
             'scoring_checks':{'overall_and_weighted_MAE_independently_recomputed':score_count,
                               'primary_and_v24_comparisons_independently_reconstructed':8,
                               'branch_metrics_checked':16,'raw_unit_unprojected_TWD_MAE_and_projection_checks':4,
                               'bootstrap_files_summary_checked':len(boots)},'bootstrap_artifacts':boots,
             'protected_unchanged':True,'query_labels_used_for_postseal_review_only':True,
             'new_model_fits':0,'GPU_work':False,'Public_score':None,'Private_score':None,
             'selected_submission':'v22','new_CSV':False,'blocking_findings_remaining':[],
             'limits':['All four origins are reused and overlapping, not fresh blind validation or four independent votes.',
                       'All four saved CBM models were reloaded on CPU and full-query raw unit predictions independently matched byte-for-byte, then original effective-area inverse-once and projection were independently reconstructed.',
                       'Runtime-only allow_writing_files/thread_count/verbose are source/start metadata controls, not recoverable native model parameters.',
                       'Only descriptive weight sum_squares and derived ESS permit at most 128 symmetric float64 ULP of reduction rounding; all weights dtype/shape/values SHA, predictions, metrics and gates remain exact.',
                       'query_y_used_for_diagnostics=false applies to frozen raw-unit/projection counts only; unprojected_TWD_overall_mae was recomputed using postseal query labels and is not label-free.',
                       'Composition and descriptive definitions reuse frozen helpers; independent MAE/group/gate arithmetic supplements replay.',
                       'Bootstrap summaries use saved replicates without resampling, selection/refit/future-shift correction; not Private probability.',
                       'No fullfit/release/CSV/Kaggle upload or v22 replacement is authorized by audit pass.'],
             'review_elapsed_seconds':time.monotonic()-tick}
    c.write(output,payload)
    c.write(FOLDER+'postfit_interpretation_review.json',{
        'status':'passed','round':116,'created_at_utc':c.now(),'request':rr,'seal':sr,'results':rs,
        'evaluation_execution':er,'postfit_review':c.record(output),'review_tool':c.record(__file__),
        'source_sha256':merged,
        'nested_flag_path':'stages.*.raw_unit_diagnostic_only.query_y_used_for_diagnostics',
        'preserved_original_flag_value':False,
        'flag_false_applies_only_to':'Frozen raw-unit/extrema/negative counts and nonnegative projection statistics.',
        'postseal_query_y_used_for':'stages.*.raw_unit_diagnostic_only.unprojected_TWD_overall_mae',
        'unprojected_TWD_MAE_uses_query_y':True,
        'scoring_authorized_only_after_all_four_models_sealed_and_training_exit0':True,
        'original_source_results_gates_receipts_unmodified':True,'scientific_metrics_and_verdicts_unchanged':True,
        'new_model_fits':0,'GPU_work':False,'Public_score':None,'Private_score':None,
        'selected_submission':'v22','new_CSV':False})
    print(json.dumps({'status':'passed','review':c.record(output),'manual_gate_verdicts':verdicts},ensure_ascii=False))


def self_check():
    import pandas as pd
    checks=[]
    artificial_weights=np.array([.5,1.,2.],np.float32)
    weight_sha=c.values_sha(artificial_weights)
    exact_weight_hash(artificial_weights,weight_sha,'synthetic');checks.append('weight_SHA_exact')
    for bad,expected in ((artificial_weights,'0'*64),(artificial_weights*2,weight_sha)):
        try:exact_weight_hash(bad,expected,'synthetic')
        except Exception:checks.append('wrong_weight_SHA_still_rejected')
        else:raise AssertionError('Weight identity relaxed')
    base={'rows':3,'min':1.,'max':2.,'mean':1.5,'sum':4.5,'sum_squares':8.,'effective_sample_size':2.53125}
    for steps in (0,1,128):
        changed=dict(base)
        for _ in range(steps):changed['sum_squares']=float(np.nextafter(changed['sum_squares'],np.inf))
        summary_comparison(base,changed,'synthetic');checks.append('summary_128_ULP_boundary')
    for key,steps in (('sum_squares',129),('effective_sample_size',129),('sum',1),('mean',1),('min',1),('max',1)):
        changed=dict(base)
        for _ in range(steps):changed[key]=float(np.nextafter(changed[key],np.inf))
        try:summary_comparison(base,changed,'synthetic')
        except Exception:checks.append('summary_outside_declared_scope_rejected')
        else:raise AssertionError('Weight summary tolerance accepted invalid change')
    for family in ('date','pool'):
        for key in ('sum_squares','effective_sample_size'):
            for steps in (128,129):
                changed=dict(base)
                for _ in range(steps):changed[key]=float(np.nextafter(changed[key],np.inf))
                if steps==128:
                    summary_comparison(base,changed,family);checks.append(family+'_'+key+'_128_ULP_pass')
                else:
                    try:summary_comparison(base,changed,family)
                    except Exception:checks.append(family+'_'+key+'_129_ULP_rejected')
                    else:raise AssertionError('129 ULP accepted '+family)
        try:exact_weight_hash(artificial_weights,'0'*64,family)
        except Exception:checks.append(family+'_wrong_weight_SHA_rejected')
        else:raise AssertionError('Wrong SHA accepted '+family)
    X=pd.DataFrame({'building_area':[2.,np.nan,-1.,0.],'parking_area':[9.,3.,np.nan,-2.]})
    area=effective_area(X);c.exact(area,np.array([2.,3.,1.,1.],np.float64),'Area fallback fixture');checks.append('area_fallback')
    raw=np.array([-3.,0.,2.,4.],np.float64);native=np.maximum(raw*(area*1e4),0.)
    d=unit_inverse(raw,native,X);same(d['raw_negative_unit_rows'],1,'Unit diagnostic');checks.append('unit_inverse_once')
    for bad in (native/1e4,native*1e4):
        try:unit_inverse(raw,bad,X)
        except Exception:checks.append('wrong_unit_inverse_rejected')
        else:raise AssertionError('Invalid inverse accepted')
    leaves=np.array(([1,3,256]*1000)[:2999],dtype=np.uint32)
    summary=leaf_identity(leaves);same(summary['tree_leaf_counts_count'],2999,'Leaf count');checks.append('nonsymmetric_nonpower2_leafs_accepted')
    for bad in (leaves.astype(float),leaves[:-1],np.full(2999,0,np.int64),np.full(2999,257,np.int64)):
        try:leaf_identity(bad)
        except Exception:checks.append('invalid_leaf_schema_rejected')
        else:raise AssertionError('Invalid leaf counts accepted')
    def metric(v=100.):
        item={'overall_mae':v,'weighted_mae':v,'price_bands':[{'band':'x','rows':500,'mae':100.}]}
        for k in ('strata','forecast_quarters','months','cities','relative_forecast_months'):
            item[k]={'x':{'rows':500,'mae':100.}}
        return item
    for offset in (False,True):
        for secondary in (False,True):
            comp=comparison(metric(),metric(99.),offset,secondary)
            same(comp['candidate'],'R116','Candidate label');same(comp['control'],'v24' if secondary else 'v22','Control label')
            same(comp['used_for_release_gate'],not secondary,'Gate/counterevidence scope');checks.append('origin_comparison_wrapper')
    for value,want in ((99.,True),(100.25,True),(100.25001,False)):
        m={'v22':metric(),'T':metric(value)}
        same(extra_gate(m,comparison(m['v22'],m['T'],True))['passed'],want,'Extra boundary');checks.append('extra_boundary')
    compile(Path(__file__).read_text(encoding='utf-8-sig'),str(Path(__file__)),'exec')
    print(json.dumps({'status':'passed','check_count':len(checks),'checks':checks,'review_tool':c.record(__file__),
                      'new_fits':0,'GPU_work':False,'real_arrays_or_query_y_read':False},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-check',action='store_true');g.add_argument('--evaluation-complete',action='store_true');a=p.parse_args()
    if a.self_check:self_check()
    else:run()
