"""Independent CPU four-origin R114 postfit audit, only after formal evaluate0.

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
import audit_model_identity as native_audit

FOLDER='outputs/research_round114_20261008/'
PRIMITIVE='outputs/research_continue_20261008/pipeline_review/audit_primitives.py'
DRIVER='src/round114_research.py'
PLAN='reports/round114_plan.md'
LAYERS={'native','base','reference','spatial','median'}


def same(a,b,name):c.need(a==b,name)


def locked(name):
    path=c.safe(name)
    same(path.with_suffix('.sha256').read_text(encoding='ascii').strip(),c.sha(path),'Receipt sidecar '+name)
    return c.read(path),c.record(path)


def arithmetic():
    spec=importlib.util.spec_from_file_location('r114_independent_math',c.safe(PRIMITIVE))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def comparison(control,candidate,is_offset,secondary=False):
    name='v24' if secondary else 'v22'
    result=arithmetic().manual_comparison(control,candidate,name)
    if is_offset:
        result.pop('stage_research_gate_passed',None);result.pop('used_for_research_gate',None)
        result['automatic_adoption']=False
    result.update(candidate='R114',used_for_release_gate=not secondary)
    return result


def extra_gate(metrics,compared):
    parts={'overall_regression_at_most_0_25pct':metrics['T']['overall_mae']<=1.0025*metrics['v22']['overall_mae'],
           'weighted_regression_at_most_0_25pct':metrics['T']['weighted_mae']<=1.0025*metrics['v22']['weighted_mae'],
           'no_quarter_or_large_group_above_5pct':not compared['quarter_or_large_group_deterioration_above_5pct']}
    return {'components':parts,'passed':all(parts.values()),
            'both_point_metrics_improve':all(metrics['T'][k]<metrics['v22'][k] for k in ('overall_mae','weighted_mae'))}


def projection(raw,native):
    c.need(raw.dtype==native.dtype==np.float64 and raw.ndim==1 and raw.shape==native.shape and len(raw)>0
           and np.isfinite(raw).all() and np.isfinite(native).all(),'Signed/native schema')
    c.exact(native,np.maximum(raw,0.),'Exactly one inference total projection')
    correction=native-raw
    return {'raw_negative_total_rows':int((raw<0).sum()),'raw_negative_total_fraction':float((raw<0).mean()),
            'raw_zero_total_rows':int((raw==0).sum()),'raw_min_TWD':float(raw.min()),'raw_max_TWD':float(raw.max()),
            'projected_min_TWD':float(native.min()),'projected_max_TWD':float(native.max()),
            'projected_zero_rows':int((native==0).sum()),'projected_zero_fraction':float((native==0).mean()),
            'projection_mean_correction_TWD':float(correction.mean()),'projection_max_correction_TWD':float(correction.max()),
            'projection_changes_count':int(np.count_nonzero(correction)),'query_y_used_for_diagnostics':False,
            'negative_residual_not_equivalent_to_negative_total':True}


def check_stage(origin,record,request,driver):
    folder=c.safe(f'outputs/round114/{origin}');model=folder/'model';nf,nq=driver.COUNTS[origin]
    same(record,c.record(folder/'stage.json'),'Fixed stage record')
    item=c.read(record['path']);same((item['stage'],item['candidate_round'],item['fit_rows'],item['query_rows']),
                                  (origin,114,nf,nq),'Fixed stage/counts')
    c.need(item['query_labels_available'] is False and item['query_scores_computed'] is False,'Unscored stage')
    files=sorted(model.iterdir())
    c.need(files and all(p.is_file() and not p.is_symlink() and p.suffix in {'.pt','.json','.npy'} for p in files),
           'Complete flat native artifact directory')
    same(item['sealed_models'],[c.record(p) for p in files],'Exact complete native artifact membership/SHA')
    same(item['model_metadata'],c.record(model/'training.json'),'Fixed actual metadata record')
    meta=c.read(model/'training.json');driver.check_metadata(meta,request,origin,item['prepared_array_identities'])
    same(item['context_before'],item['context_after'],'Saved fit/query state unchanged')
    same(c.sha(model/'model.pt'),meta['model_sha256'],'Actual saved checkpoint SHA')
    run=c.read(model/'run.json');same(run['status'],'started','Original run status')
    c.need(all(meta.get(k)==v for k,v in run.items() if k!='status'),'Before/after metadata content')
    signed=np.load(model/'predictions_raw_twd.npy',allow_pickle=False)
    native=np.load(model/'predictions_twd.npy',allow_pickle=False)
    same(signed.shape,(nq,),'Signed raw total count')
    diag=projection(signed,native)
    same(diag,item['projection_diagnostics'],'Independent saved stage projection diagnostics')
    same(diag,meta['projection_diagnostics'],'Independent native projection diagnostics')
    same(item['raw_signed_total'],{**c.record(model/'predictions_raw_twd.npy'),**c.identity(signed)},'Signed raw array identity')
    for name,a in (('predictions_raw_twd.npy',signed),('predictions_twd.npy',native)):
        same(meta['prediction_artifacts'][name],{'sha256':c.sha(model/name),**c.identity(a)},'Actual native artifact identity')
    same(set(item['layers']),LAYERS,'All five layers')
    layers={key:c.load_array(rec) for key,rec in item['layers'].items()}
    for key,a in layers.items():
        c.need(a.dtype==np.float64 and a.shape==(nq,) and np.isfinite(a).all() and (a>=0).all(),'Complete TWD layer '+key)
    c.exact(layers['native'],native,'Model/native-layer identity')
    return item,layers,signed,meta


def check_native(origin,meta,loaded,request):
    prefix=f'outputs/round114/{origin}/model/'
    arrays=loaded['arrays'];recipe=request['recipe']
    same(meta['input_arrays'],{k:c.identity(v) for k,v in arrays.items()},'All nine actual NN inputs')
    same(meta['configuration'],recipe['configuration'],'Fixed additive parameterization')
    same(meta['baseline_configuration'],recipe['baseline_configuration'],'Original complete R63 config')
    same(meta['parameterization_overrides'],recipe['parameterization_overrides'],'Only named parameterization fields')
    c.need(meta['original_globals_modified'] is False and meta['hidden_architecture_embedding_dropouts_optimizer_schedule_unchanged'] is True
           and meta['parameterization_initial_bias_and_gradient_also_change'] is True
           and meta['loss_only_change_claimed'] is False and meta['gradient_magnitude_matching_to_ratio_claimed'] is False
           and meta['head_weight_zeroed'] is False and meta['initial_prediction_equals_prior_claimed'] is False
           and meta['epoch_selection_uses_training_metrics'] is False
           and meta['inference_nonnegative_projection_after_total_restoration'] is True
           and meta['price_factor_used_as_additive_offset'] is True,'Additive noncausal/objective/init scope')
    same(meta['training_objective_name'],'weighted MAE of unprojected additive total million TWD','Actual objective')
    for key in ('train_sample_weight','train_price_factor_million','query_price_factor_million'):
        same(meta[key+'_sha256'],c.values_sha(arrays[key]),'Actual original date/factor '+key)
    bias=float(np.median(arrays['train_y_million'].astype(np.float64)-arrays['train_price_factor_million'].astype(np.float64)))
    same(meta['initial_training_median_residual_million'],bias,'Original fit-only signed median bias')
    same(meta['initial_head_bias'],float(np.float32(bias)),'Actual bias parameter F32 representation')
    old_path=('outputs/round97/nn_models/N_prior/training.json' if origin=='offset_2023_02_11'
              else f'outputs/round63/{origin}/model/training.json')
    old=c.read(old_path)
    same([v['learning_rate'] for v in meta['epochs']],[v['learning_rate'] for v in old['epochs']],'Original actual learning-rate schedule')
    saved=native_audit.checkpoint_cpu(prefix+'model.pt');payload=saved['payload']
    for key in ('configuration','baseline_configuration','numeric_count','cardinalities'):
        same(payload[key],meta[key],'Actual checkpoint '+key)
    same(payload['model_class'],meta['actual_model_class'],'Actual saved model class')
    same(saved['parameter_count'],meta['parameter_count'],'Actual saved parameter count')
    same(saved['embedding_dimensions'],meta['embedding_dimensions'],'Actual saved embedding dimensions')
    same(payload['parent_trainer_source_sha256'],c.sha('src/round63_recency_weighted_prior_train.py'),'Checkpoint original R63 source')
    same(payload['original_architecture_source_sha256'],c.sha('src/round47_nn_train.py'),'Checkpoint original R47 source')
    same(meta['cardinalities'],arrays['cardinalities'].tolist(),'Fit-only vocabulary identity')
    return {'checkpoint':c.record(prefix+'model.pt'),'CPU_checkpoint':saved,
            'full_query_NN_inference_independently_performed':False,'saved_full_query_reload_is_trainer_evidence':True}


def check_bootstrap(origin,item,point,stage_index):
    pair=item['paired'];same((pair['control'],pair['candidate'],pair['difference'],pair['seed'],pair['replicates'],pair['stage_index']),
        ('v22','R114','R114 minus v22',20260925,2000,stage_index),'Correct paired labels/origin seed')
    c.need('not Private probability' in pair['caution'] and 'not matched refits' in pair['caution'],'Uncertainty caution')
    records=[];same(set(item['paired_samples']),{'project','month'},'Both conditional bootstrap families')
    for family in ('project','month'):
        rec=item['paired_samples'][family]
        same(rec['path'],f'outputs/round114/{origin}/evaluation/{family}_samples.npy','Fixed paired replicate path')
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
    deps={name:c.sha(name) for name in (c.rel(__file__),FOLDER+'audit_model_identity.py',PRIMITIVE)}
    import round114_research as driver
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
    same(seal['status'],'four_models_all_native_signed_and_layers_sealed_unscored','Every signed/native/layer sealed')
    same(seal['request'],rr,'Seal/request binding');same(seal['query_scores_computed'],False,'Seal unscored')
    same(set(seal['records']),set(driver.ORIGINS),'Every origin sealed')
    same(seal['origin_order'],list(driver.ORIGINS),'Sealed fixed order');same(seal['new_fits'],4,'All four sealed fits')
    for name,(entry,environment,flag) in driver.formal_checks().items():
        formal,_=locked(FOLDER+name)
        same(formal['exit_code'],0,'CPU selfcheck exit0')
        same(formal['command'],[str(ROOT/environment/'Scripts/python.exe'),'-B','-X','utf8',str(c.safe(entry)),flag],'CPU exact command')
        same(formal['source_sha256'],c.source_hashes(entry),'CPU exact source closure')
    checked={origin:check_stage(origin,seal['records'][origin],request,driver) for origin in driver.ORIGINS}
    # All four completion/native/checkpoint/prediction/layer identities pass
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
        same(st['prepared_array_identities'],loaded['array_identities'],'Actual original nine-array identities')
        same(st['context_before'],loaded['before'],'Actual saved origin state/frames/indices/fixed predictions')
        native_facts[origin]=check_native(origin,meta,loaded,request)
        ctx=loaded['context'];is_offset=origin==driver.OFFSET
        replay=driver.offset_comp.chain_prior(ctx,layers['native']) if is_offset else driver.comp.apply(ctx,layers['native'])[0]
        for key in LAYERS:c.exact(replay[key],layers[key],'Actual complete saved candidate chain '+key)
        metric=driver.offset_metric if is_offset else driver.metric
        y=metric.select_query_y(raw,loaded['indices']['query_source_cache_indices'],455784) if is_offset else metric.select_query_y(raw,loaded['indices']['query_source_cache_indices'])
        descriptive=(lambda a:metric.descriptive_metrics(y,a,loaded['fit_X'],loaded['query_X'])) if is_offset else (lambda a:metric.descriptive_metrics(y,a,loaded['fit_X'],loaded['query_X'],c.SPLITS[origin][0]))
        item=results['stages'][origin];stage_eval,stage_record=locked(f'outputs/round114/{origin}/evaluation/results.json')
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
        same(item['raw_signed_total_diagnostic_only'],{'overall_mae':float(np.abs(signed-y).mean()),**projection(signed,layers['native'])},
             'Independent signed raw MAE/projection diagnostics')
        c.need((np.abs(layers['native']-y)<=np.abs(signed-y)).all(),'Positive truth: nonnegative projection cannot increase native absolute error')
        same(item['both_point_metrics_improve'],all(item['metrics']['T'][k]<item['metrics']['v22'][k] for k in ('overall_mae','weighted_mae')),
             'Strict both-point-improvement label')
        point=metric.methods.point_values(np.abs(y[:,None]-np.column_stack([vectors['v22'],vectors['T']])),labels,weights)
        boots.extend(check_bootstrap(origin,item,point,0 if is_offset else c.STAGES.index(origin)))
        artifacts[origin]={'stage':seal['records'][origin],'training_metadata':st['model_metadata'],
                           'evaluation':stage_record,'sealed_models':st['sealed_models'],'raw_signed_total':st['raw_signed_total']}
        trainmeta[origin]=meta
        summary[origin]={'overall_mae':item['metrics']['T']['overall_mae'],'weighted_mae':item['metrics']['T']['weighted_mae'],
                         'overall_delta_vs_v22':primary['overall_mae']['delta'],'weighted_delta_vs_v22':primary['weighted_mae']['delta'],
                         'overall_delta_vs_v24':secondary['overall_mae']['delta'],'weighted_delta_vs_v24':secondary['weighted_mae']['delta'],
                         'adverse_groups':primary['quarter_or_large_group_deterioration_above_5pct'],
                         'v24_adverse_groups':secondary['quarter_or_large_group_deterioration_above_5pct'],
                         'projection_diagnostics':projection(signed,layers['native']),'fit_seconds':meta['fit_seconds']}
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
    payload={'status':'passed','reviewer':'research_preflight_20261008','created_at_utc':c.now(),'round':114,
             'scope':'Independent CPU post-evaluation native/checkpoint/signed projection/composition/metrics/gates audit; accuracy gate recorded separately.',
             'source_sha256':merged,'plan':request['plan'],'request':rr,'seal':sr,'training_execution':tr,
             'evaluation_execution':er,'results':rs,'preflight_review':pfr,'review_tool':c.record(__file__),
             'review_dependency_source_sha256':deps,'artifacts':artifacts,'trainmeta':trainmeta,
             'manual_gate_verdicts':verdicts,'stage_summary':summary,'native_identity_diagnostics':native_facts,
             'identity_checks':{'frozen_sources_including_plan':len(merged),'frozen_inputs':len(request['input_sha256']),
                                'protected_files':len(request['protected_before']),'sealed_origins':4,
                                'NN_checkpoints_CPU_inspected':4,'complete_chain_layers_replayed':20,
                                'signed_native_projection_and_metadata_sets_verified':4},
             'scoring_checks':{'overall_and_weighted_MAE_independently_recomputed':score_count,
                               'primary_and_v24_comparisons_independently_reconstructed':8,
                               'branch_metrics_checked':16,'signed_raw_MAE_and_projection_checks':4,
                               'bootstrap_files_summary_checked':len(boots)},'bootstrap_artifacts':boots,
             'protected_unchanged':True,'query_labels_used_for_postseal_review_only':True,
             'new_model_fits':0,'GPU_work':False,'Public_score':None,'Private_score':None,
             'selected_submission':'v22','new_CSV':False,'blocking_findings_remaining':[],
             'limits':['All four origins are reused and overlapping, not fresh blind validation or four independent votes.',
                       'CPU checkpoint constructors/tensors/config were inspected without independent full-query NN inference; exact saved reload is frozen training evidence.',
                       'Saved factors, signed total/projection and all complete-chain layers were verified. No separately saved residual output exists, so raw total inverse relies on frozen trainer code/reload.',
                       'Composition and descriptive definitions reuse frozen helpers; independent MAE/group/gate arithmetic supplements replay.',
                       'Bootstrap summaries use saved replicates without resampling, selection/refit/future-shift correction; not Private probability.',
                       'No fullfit/release/CSV/Kaggle upload or v22 replacement is authorized by audit pass.'],
             'review_elapsed_seconds':time.monotonic()-tick}
    c.write(output,payload)
    print(json.dumps({'status':'passed','review':c.record(output),'manual_gate_verdicts':verdicts},ensure_ascii=False))


def self_check():
    checks=[]
    raw=np.array([-3.,0.,2.],np.float64);native=np.maximum(raw,0.)
    d=projection(raw,native);same(d['raw_negative_total_rows'],1,'Signed diagnostic');checks.append('signed_projection')
    def metric(v=100.):
        item={'overall_mae':v,'weighted_mae':v,'price_bands':[{'band':'x','rows':500,'mae':100.}]}
        for k in ('strata','forecast_quarters','months','cities','relative_forecast_months'):
            item[k]={'x':{'rows':500,'mae':100.}}
        return item
    for offset in (False,True):
        for secondary in (False,True):
            comp=comparison(metric(),metric(99.),offset,secondary)
            same(comp['candidate'],'R114','Candidate label');same(comp['control'],'v24' if secondary else 'v22','Control label')
            same(comp['used_for_release_gate'],not secondary,'Gate/counterevidence scope');checks.append('origin_comparison_wrapper')
    for value,want in ((99.,True),(100.25,True),(100.25001,False)):
        m={'v22':metric(),'T':metric(value)}
        same(extra_gate(m,comparison(m['v22'],m['T'],True))['passed'],want,'Extra boundary');checks.append('extra_boundary')
    for path in (Path(__file__),Path(native_audit.__file__)):
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    print(json.dumps({'status':'passed','check_count':len(checks),'checks':checks,'review_tool':c.record(__file__),
                      'native_audit_tool':c.record(native_audit.__file__),'new_fits':0,'GPU_work':False,
                      'real_arrays_or_query_y_read':False},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-check',action='store_true');g.add_argument('--evaluation-complete',action='store_true');a=p.parse_args()
    if a.self_check:self_check()
    else:run()
