"""Independent CPU four-origin R117 engineering-v2 postfit audit, only after formal evaluate0.

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
import audit_model_identity_v2 as native_audit

FOLDER='outputs/research_round117_v2_20261008/'
BASE='outputs/round117_v2/'
ORACLE_REVIEW='outputs/research_round117_20261008/independent_prior_oracle_review.json'
PRIMITIVE='outputs/research_continue_20261008/pipeline_review/audit_primitives.py'
DRIVER='src/round117_research_v2.py'
PLAN='reports/round117_plan.md'
RECOVERY_PLAN='reports/round117_engineering_recovery_v2.md'
LAYERS={'native','base','reference','spatial','median'}


def same(a,b,name):c.need(a==b,name)


def locked(name):
    path=c.safe(name)
    same(path.with_suffix('.sha256').read_text(encoding='ascii').strip(),c.sha(path),'Receipt sidecar '+name)
    return c.read(path),c.record(path)


def arithmetic():
    spec=importlib.util.spec_from_file_location('r117_independent_math',c.safe(PRIMITIVE))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def comparison(control,candidate,is_offset,secondary=False):
    name='v24' if secondary else 'v22'
    result=arithmetic().manual_comparison(control,candidate,name)
    if is_offset:
        result.pop('stage_research_gate_passed',None);result.pop('used_for_research_gate',None)
        result['automatic_adoption']=False
    result.update(candidate='R117',used_for_release_gate=not secondary)
    return result


def extra_gate(metrics,compared):
    parts={'overall_regression_at_most_0_25pct':metrics['T']['overall_mae']<=1.0025*metrics['v22']['overall_mae'],
           'weighted_regression_at_most_0_25pct':metrics['T']['weighted_mae']<=1.0025*metrics['v22']['weighted_mae'],
           'no_quarter_or_large_group_above_5pct':not compared['quarter_or_large_group_deterioration_above_5pct']}
    return {'components':parts,'passed':all(parts.values()),
            'both_point_metrics_improve':all(metrics['T'][k]<metrics['v22'][k] for k in ('overall_mae','weighted_mae'))}


def archive_arrays(archive):
    same(len(archive.files),len(c.ARRAY_NAMES),'Exact NPZ key count before values')
    same(set(archive.files),set(c.ARRAY_NAMES),'Exact NPZ keyset before values; no query y')
    return {k:archive[k].copy() for k in c.ARRAY_NAMES}


def effective_area(X):
    building=np.asarray(X['building_area'],dtype=np.float64);parking=np.asarray(X['parking_area'],dtype=np.float64)
    result=np.where(np.isfinite(building)&(building>0),building,np.where(np.isfinite(parking)&(parking>0),parking,1.))
    c.need(result.shape==(len(X),) and np.isfinite(result).all() and (result>0).all(),'Independent area fallback')
    return result


def ratio_restore(ratio,native,factor):
    c.need(ratio.dtype==native.dtype==np.float64 and factor.dtype==np.float32 and len(ratio)>0
           and ratio.shape==native.shape==factor.shape and ratio.ndim==1
           and np.isfinite(ratio).all() and (ratio>0).all() and np.isfinite(native).all() and (native>0).all()
           and np.isfinite(factor).all() and (factor>0).all(),'Positive raw ratio/native/F32 factor contract')
    c.exact(ratio*factor.astype(np.float64)*1e6,native,'Independent exactly one ratio-to-TWD inverse, no cap/clip')


def ratio_bias(y,factor):
    c.need(y.dtype==factor.dtype==np.float32 and y.ndim==1 and y.shape==factor.shape
           and len(y)>0 and np.isfinite(y).all() and (y>0).all() and np.isfinite(factor).all() and (factor>0).all(),
           'Independent fit ratio inputs')
    median=float(np.median(y.astype(np.float64)/factor.astype(np.float64)))
    bias=median+float(np.log(-np.expm1(-median)))
    return {'initial_training_median_ratio':median,'initial_head_bias_formula_float64':bias,'initial_head_bias_float32':float(np.float32(bias))}


def fixed_folds(n):
    c.need(type(n) is int and n>=5,'Five-fold canonical integer row count')
    result=np.empty(n,np.int64);result[np.random.default_rng(42).permutation(n)]=np.arange(n,dtype=np.int64)%5
    return result


def manual_weighted_median(values,weights):
    c.need(values.shape==weights.shape and values.ndim==1 and len(values)>0
           and np.isfinite(values).all() and (values>0).all() and np.isfinite(weights).all() and (weights>0).all(),'Independent weighted median inputs')
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]/weights.max()
    cumulative=np.cumsum(w,dtype=np.float64);half=float(cumulative[-1])/2.
    j=int(np.searchsorted(cumulative,half,side='left'));value=float(v[j])
    return float(v[j]+(v[j+1]-v[j])/2.) if cumulative[j]==half and j+1<len(v) else value


def date_summary(X,mask,weight_rules):
    days,invalid,outside=[X[key].to_numpy(dtype=np.float64,copy=True)[mask] for key in
                         ('trade_day_index','trade_date_invalid','trade_date_out_of_range')]
    c.need(len(days)>0 and np.isfinite(invalid).all() and np.isfinite(outside).all(),'Independent reference date schema')
    valid=np.isfinite(days)&(invalid==0)&(outside==0);c.need(valid.any(),'Independent reference needs valid dates')
    anchor=float(days[valid].max());u=np.exp2(-(anchor-days[valid])/365.)
    denominator=float(u.mean(dtype=np.float64));weights=np.ones(len(days),np.float64);weights[valid]=u/denominator
    c.need(np.isfinite(weights).all() and (weights>0).all(),'Independent positive date weights')
    return weights,{'reference_rows':len(weights),'valid_date_rows':int(valid.sum()),'neutral_date_rows':int((~valid).sum()),
        'invalid_flag_rows':int((invalid!=0).sum()),'out_of_range_flag_rows':int((outside!=0).sum()),
        'max_valid_reference_trade_day_index':anchor,'min_valid_reference_trade_day_index':float(days[valid].min()),
        'normalization_mean_valid_reference_only':denominator,'weight_min':float(weights.min()),'weight_max':float(weights.max()),
        'weight_mean':float(weights.mean()),'weight_sum':float(weights.sum(dtype=np.float64)),
        'weights_sha256':c.values_sha(weights),'date_values_sha256':c.values_sha(days),'invalid_flags_sha256':c.values_sha(invalid),
        'out_of_range_flags_sha256':c.values_sha(outside),'valid_mask_sha256':c.values_sha(valid.astype('u1')),'weight_rules':weight_rules}


def check_stage(origin,record,request,driver,inspect_native=False):
    folder=c.safe(BASE+origin);model=folder/'model';prior=folder/'prior';nf,nq=driver.COUNTS[origin]
    same(record,c.record(folder/'stage.json'),'Fixed stage record')
    item=c.read(record['path']);same((item['stage'],item['candidate_round'],item['fit_rows'],item['query_rows']),
                                  (origin,117,nf,nq),'Fixed stage/counts')
    c.need(item['query_labels_available'] is False and item['query_scores_computed'] is False
           and item['original_v22_v24_and_prior_slot_replayed_exact'] is True,'Unscored stage/original slot scope')
    for dirname,records,required in ((model,item['sealed_models'],{'model.pt','run.json','training.json','ratio_predictions.npy','predictions_twd.npy'}),
        (prior,item['sealed_prior_artifacts'],{'prepared.npz','preparation.json','preparation.sha256','price_prior_state.json','prior_evidence.json',
            'prior_evidence.sha256','prior_fold_ids.npy','train_oof_unit_prior.npy','query_unit_prior.npy',
            'original_train_factor_million.npy','transaction_train_factor_million.npy'})):
        files=sorted(dirname.iterdir())
        c.need({p.name for p in files}==required and all(p.is_file() and not p.is_symlink() for p in files),'Exact flat native/prior membership')
        same(records,[c.record(p) for p in files],'Every complete native/prior artifact SHA')
    same(item['model_metadata'],c.record(model/'training.json'),'Actual saved metadata identity')
    same(item['prior_preparation'],c.record(prior/'preparation.json'),'Actual saved preparation identity')
    same(item['train_log'],c.record(folder/'train.log'),'Actual stage train log')
    prep=c.read(prior/'preparation.json');meta=c.read(model/'training.json');driver.check_metadata(meta,request,origin,prep)
    same(item['context_before'],item['context_after'],'Fit/query state unchanged')
    same(prep['origin'],origin,'Actual prior origin');same((prep['fit_rows'],prep['query_rows']),(nf,nq),'Actual preparation rows')
    same(prep['prepared'],c.record(prior/'prepared.npz'),'New prepared archive exact SHA')
    same(prep['source_prepared'],request['origins'][origin]['prepared'],'Original prepared exact identity')
    same(prep['original_prior'],request['origins'][origin]['original_prior'],'Original prior authority')
    same(prep['array_identities']['original'],item['original_prepared_array_identities'],'Original nine arrays seal')
    same(prep['array_identities']['revised'],item['prepared_array_identities'],'Revised nine arrays seal')
    with np.load(prior/'prepared.npz',allow_pickle=False) as z:
        same(len(z.files),9,'Nine keys before array reads');same(set(z.files),set(c.ARRAY_NAMES),'No query labels before values')
        for key in c.ARRAY_NAMES:
            value=z[key];same(c.identity(value),item['prepared_array_identities'][key],'Every actual prepared array identity')
            if key!=driver.CHANGED_ARRAY:same(c.identity(value),item['original_prepared_array_identities'][key],'Original eight arrays immutable')
        factor=z['query_price_factor_million']
    same(c.sha(prior/'price_prior_state.json'),request['origins'][origin]['original_prior']['state']['sha256'],'Unchanged fullfit state file bytes')
    same(meta['model_sha256'],c.sha(model/'model.pt'),'Actual saved checkpoint SHA')
    run=c.read(model/'run.json');c.need(all(meta.get(k)==v for k,v in run.items() if k!='status'),'Original run/completed metadata identity')
    ratio=np.load(model/'ratio_predictions.npy',allow_pickle=False);native=np.load(model/'predictions_twd.npy',allow_pickle=False)
    same(ratio.shape,(nq,),'Full ratio query count');ratio_restore(ratio,native,factor)
    for key,name,value in (('ratio','ratio_predictions.npy',ratio),('TWD','predictions_twd.npy',native)):
        same(meta['prediction_artifacts'][key],{**c.record(model/name),**c.identity(value)},'Native prediction saved-array identity')
    same(set(item['layers']),LAYERS,'All five layers')
    layers={key:c.load_array(rec) for key,rec in item['layers'].items()}
    for key,a in layers.items():
        c.need(a.dtype==np.float64 and a.shape==(nq,) and np.isfinite(a).all() and (a>=0).all(),'Complete TWD layer '+key)
    c.exact(layers['native'],native,'Native file/five-layer alignment')
    checkpoint=native_audit.checkpoint_cpu(c.rel(model/'model.pt')) if inspect_native else None
    return item,layers,ratio,meta,prep,checkpoint


def check_native(origin,meta,loaded,request,driver,stage,prep,ratio,native,checkpoint):
    prefix=BASE+origin+'/';nf,nq=driver.COUNTS[origin]
    driver.check_metadata(meta,request,origin,prep)
    with np.load(c.safe(prep['prepared']['path']),allow_pickle=False) as z:revised=archive_arrays(z)
    same(driver.check_changed_arrays(loaded['arrays'],revised),prep['array_identities'],'Actual original eight arrays and new F only')
    same(meta['input_arrays'],{k:c.identity(v) for k,v in revised.items()},'Actual trainer exact nine input arrays')
    same(meta['cardinalities'],revised['cardinalities'].tolist(),'Original fit-only vocabulary cardinalities')
    old_bias=ratio_bias(revised['train_y_million'],loaded['arrays'][driver.CHANGED_ARRAY])
    new_bias=ratio_bias(revised['train_y_million'],revised[driver.CHANGED_ARRAY])
    same(prep['original_initialization'],old_bias,'Independent original fit-only median/inverseSoftplus bias')
    same(prep['transaction_initialization'],new_bias,'Independent linked new fit-only median/inverseSoftplus bias')
    same(stage['original_initialization'],old_bias,'Stage original bias identity');same(stage['transaction_initialization'],new_bias,'Stage new bias identity')
    old=c.read(loaded['original_training']['path'])
    same(old['initial_training_median_ratio'],old_bias['initial_training_median_ratio'],'Original actual fit ratio median authority')
    same(old['initial_head_bias'],old_bias['initial_head_bias_formula_float64'],'Original actual head bias authority')
    same([v['learning_rate'] for v in meta['epochs']],[v['learning_rate'] for v in old['epochs']],'Original actual complete LR schedule')
    payload=checkpoint['payload']
    for key in ('configuration','numeric_count','cardinalities','actual_model_class','architecture_source_sha256'):
        same(payload[key],meta[key],'Actual CPU checkpoint '+key)
    same(checkpoint['parameter_count'],meta['parameter_count'],'Actual saved original parameter count')
    c.need(checkpoint['positive_ratio_Softplus_head'] and not checkpoint['model_inference_performed'],'CPU original checkpoint structure scope')
    ratio_restore(ratio,native,revised['query_price_factor_million'])
    # Replay actual fit-only prior after all models were sealed/evaluation finished.
    oof,state,folds,evidence=driver.prior.fit_oof_prior(loaded['fit_X'],loaded['fit_y'],loaded['source_indices'])
    expected_folds=fixed_folds(nf);c.exact(folds,expected_folds,'Independent fixed transaction fold assignment')
    actual_oof=c.load_array(prep['prior_artifacts']['train_oof_unit_prior'])
    actual_folds=c.load_array(prep['prior_artifacts']['prior_fold_ids'])
    c.exact(oof,actual_oof,'Full fit-only transaction OOF prior replay');c.exact(folds,actual_folds,'Actual sealed transaction folds')
    same(evidence,c.read(prep['prior_evidence']['path']),'Every actual replayed transaction prior evidence field')
    old_state=c.read(loaded['original_prior']['state']['path'])
    same(state,old_state,'Rebuilt fullfit state exact original');same(state,c.read(prep['full_fit_state']['path']),'Saved fullfit state exact rebuilt')
    driver.check_prior_evidence(evidence,loaded,state,old_state,folds,oof)
    area=effective_area(loaded['fit_X']);c.exact(area,driver.prior.effective_area(loaded['fit_X']),'Independent prior effective area')
    units=loaded['fit_y']/area;dates=[]
    for fold in range(5):
        mask=expected_folds!=fold;weights,summary=date_summary(loaded['fit_X'],mask,driver.prior.WEIGHT_RULES)
        same(summary,evidence['folds'][fold]['date_support'],'Independent complete complement-specific date weight summary/hash')
        same(manual_weighted_median(units[mask],weights),evidence['folds'][fold]['reference_global_median'],'Independent complement-only global median')
        dates.append({'fold':fold,'anchor':summary['max_valid_reference_trade_day_index'],
            'normalization_mean_valid_reference_only':summary['normalization_mean_valid_reference_only'],'weights_sha256':summary['weights_sha256']})
    weights,summary=date_summary(loaded['fit_X'],np.ones(nf,bool),driver.prior.WEIGHT_RULES)
    same(summary,evidence['full_fit_date_support'],'Independent fullfit date weights');same(manual_weighted_median(units,weights),state['global_median'],'Independent fullfit global median')
    query_prior,query_routes=driver.prior.predict_prior(state,loaded['query_X'],return_diagnostics=True)
    saved_query=c.load_array(prep['prior_artifacts']['query_unit_prior'])
    original_query=np.load(c.safe(loaded['original_prior']['query_prior']['path']),allow_pickle=False)
    c.exact(query_prior,saved_query,'Rebuilt unchanged fullfit query prior exact sealed');c.exact(query_prior,original_query,'Rebuilt query prior exact original')
    same(query_routes,prep['query_prior_level_counts'],'All actual query hierarchy routes')
    fit_factor=np.asarray(area*oof/1e6,np.float32)
    query_factor=np.asarray(effective_area(loaded['query_X'])*query_prior/1e6,np.float32)
    c.exact(fit_factor,revised[driver.CHANGED_ARRAY],'Independent new F from area times OOF prior once')
    c.exact(query_factor,revised['query_price_factor_million'],'Independent query F exact original')
    changed=fit_factor!=loaded['arrays'][driver.CHANGED_ARRAY]
    same(prep['fit_factor_change_summary'],{'changed_rows':int(changed.sum()),'changed_fraction':float(changed.mean()),
        'mean_absolute_change_million':float(np.abs(fit_factor.astype(np.float64)-loaded['arrays'][driver.CHANGED_ARRAY].astype(np.float64)).mean()),
        'this_is_not_query_MAE':True},'Actual fit-factor difference, never a price score')
    driver.baseline.verify_context(loaded,origin)
    return {'checkpoint':c.record(prefix+'model/model.pt'),'CPU_checkpoint':checkpoint,
        'full_query_NN_inference_independently_performed':False,'saved_full_query_ratio_and_TWD_reload_is_trainer_evidence':True,
        'full_real_transaction_OOF_prior_replayed_exact':True,'complement_and_fullfit_date_weights_and_global_medians_independently_recomputed':True,
        'complement_date_diagnostics':dates,'eight_original_arrays_byte_exact':True,'fullfit_query_state_prior_and_factor_exact_original':True,
        'original_initialization':old_bias,'transaction_initialization':new_bias,'prior_replay_uses_frozen_R60_estimator_with_independent_artificial_oracle':True}


def check_bootstrap(origin,item,point,stage_index):
    pair=item['paired'];same((pair['control'],pair['candidate'],pair['difference'],pair['seed'],pair['replicates'],pair['stage_index']),
        ('v22','R117','R117 minus v22',20260925,2000,stage_index),'Correct paired labels/origin seed')
    c.need('not Private probability' in pair['caution'] and 'retraining' in pair['caution'],'Uncertainty caution')
    records=[];same(set(item['paired_samples']),{'project','month'},'Both conditional bootstrap families')
    for family in ('project','month'):
        rec=item['paired_samples'][family]
        same(rec['path'],BASE+origin+f'/evaluation/{family}_samples.npy','Fixed paired replicate path')
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


def original_array_binding(stage,original,changed_key):
    same(stage['original_prepared_array_identities'],original,'Actual original nine-array identities; revised F is separate')
    same(set(stage['prepared_array_identities']),set(original),'Revised complete nine-array identity keys')
    for key,value in original.items():
        if key!=changed_key:
            same(stage['prepared_array_identities'][key],value,'Every original immutable array identity '+key)


def recovery_bindings(request,results,driver):
    same(request['recovery_plan'],c.record(RECOVERY_PLAN),'Exact separate engineering amendment')
    same(request['recovery_plan']['sha256'],driver.RECOVERY_PLAN_SHA,'Fixed recovery amendment SHA')
    same(request['engineering_recovery'],driver.recovery_specification({}),'Every preserved aborted attempt hash/metadata binding')
    same((request['historical_aborted_batch_completed_NN_fits'],request['planned_fresh_NN_fits'],request['maximum_R117_total_actual_NN_fits']),
         (1,4,5),'Historical aborted native plus four fresh native fits explicitly counted')
    recovery=request['engineering_recovery']
    c.need(recovery['old_partial_models_or_priors_loaded_or_reused'] is False
           and recovery['explicit_engineering_recovery_not_automatic_retry'] is True
           and recovery['scientific_plan_unchanged'] is True and recovery['query_prices_or_new_scores_read'] is False,
           'Engineering metadata is not a model reuse or scoring authorization')
    actual=driver.actual_trainer_lineage()
    c.need(set(actual)<=set(request['source_sha256']),'Actual complete parent lineage in frozen request')
    same(actual,{k:request['source_sha256'][k] for k in actual},'Actual complete parent lineage exact SHAs')
    if results is not None:
        same(results['recovery_plan'],request['recovery_plan'],'Results engineering amendment binding')
        same(results['engineering_recovery'],recovery,'Results complete aborted attempt provenance')
        same((results['historical_aborted_batch_completed_NN_fits'],results['fresh_NN_fits_in_this_batch'],results['total_R117_actual_NN_fits_including_aborted']),
             (1,4,5),'Results exact historical and fresh native counts')


def run():
    tick=time.monotonic();output=FOLDER+'postfit_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(),'Preserve old audit attempts')
    deps={name:c.sha(name) for name in (c.rel(__file__),FOLDER+'audit_model_identity.py',PRIMITIVE)}
    import round117_research_v2 as driver
    request,rr=locked(driver.REQUEST);seal,sr=locked(driver.SEAL)
    training,tr=locked(FOLDER+'training_execution.json');evaluation,er=locked(FOLDER+'evaluation_execution.json')
    same((training['exit_code'],training['failure'],training['request']),(0,None,rr),'Successful separate four-fit execution')
    same((evaluation['exit_code'],evaluation['failure'],evaluation['request'],evaluation['seal']),(0,None,rr,sr),'Successful separate evaluation')
    same(evaluation['training_execution'],tr,'Evaluation/training exact binding')
    c.need(evaluation['query_labels_used_for_scoring_only'] and evaluation['training_already_finished_before_scoring']
           and evaluation['new_fits']==0 and evaluation['new_CSV'] is False,'Evaluation scope')
    same(request['status'],'frozen_before_four_fixed_origin_priors_and_NN_fits','Fixed request state')
    same(request['new_fits'],4,'Exactly four unconditional fits');same(request['new_prior_builds'],4,'Exactly four fit-only prior builds')
    same(request['origin_order'],list(driver.ORIGINS),'Four-origin fixed order')
    same(request['origin_counts'],{s:list(v) for s,v in driver.COUNTS.items()},'Four-origin actual row contracts')
    c.need(request['selected_submission']=='v22' and not request['new_CSV'] and not request['full_data_fit']
           and not request['query_scores_computed'] and not request['query_labels_used_for_fit']
           and request['Public_score'] is None and request['Private_score'] is None,'Request selection/scores/labels scope')
    same(request['plan'],c.record(PLAN),'Frozen scientific plan')
    same(driver.sources(),request['source_sha256'],'Exact complete recursive source closure')
    merged={**request['source_sha256'],PLAN:request['plan']['sha256'],RECOVERY_PLAN:request['recovery_plan']['sha256']}
    recovery_bindings(request,None,driver)
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(request['protected_before'])
    same(driver.protected(),request['protected_before'],'Protected source/model/CSV full membership/SHA')
    preflight,pfr=locked(FOLDER+'preflight_review.json')
    same(preflight['status'],'passed','Independent source preflight passed');same(preflight['source_sha256'],merged,'Exact preflight source+plan')
    same(seal['status'],'four_priors_models_and_layers_sealed_unscored','Every prior/native/layer sealed')
    same(seal['request'],rr,'Seal/request binding');same(seal['query_scores_computed'],False,'Seal unscored')
    same(set(seal['records']),set(driver.ORIGINS),'Every origin sealed')
    same(seal['origin_order'],list(driver.ORIGINS),'Sealed fixed order');same(seal['new_fits'],4,'All four sealed fits')
    for name,(entry,environment,flag) in driver.formal_checks().items():
        formal,_=locked(FOLDER+name)
        same(formal['exit_code'],0,'CPU selfcheck exit0')
        same(formal['command'],[str(ROOT/environment/'Scripts/python.exe'),'-B','-X','utf8',str(c.safe(entry)),flag],'CPU exact command')
        same(formal['source_sha256'],c.source_hashes(entry),'CPU exact source closure')
    checked={origin:check_stage(origin,seal['records'][origin],request,driver,inspect_native=True) for origin in driver.ORIGINS}
    # All four completion/native/checkpoint/prior/prediction/layer identities pass
    # before raw query labels are deserialized below.
    results,rs=locked(driver.RESULTS)
    same(results['status'],'four_reused_fixed_origins_evaluated','Completed fixed-origin results')
    for key,value in (('request',rr),('seal',sr),('plan',request['plan']),('source_sha256',request['source_sha256'])):
        same(results[key],value,'Results frozen identity '+key)
    same(results['origin_order'],list(driver.ORIGINS),'Results four origins')
    recovery_bindings(request,results,driver)
    oracle,oracle_record=locked(ORACLE_REVIEW)
    same(oracle['status'],'passed','Prior independent artificial oracle passed')
    c.need(oracle['check_count']==27 and oracle['oracle_uses_original_key_weight_or_median_helpers'] is False
           and oracle['real_arrays_or_query_y_read'] is False and oracle['real_prior_builds']==0,'Artificial oracle actual scope')
    c.verify_hashes(oracle['source_sha256'])
    same(set(results['stages']),set(driver.ORIGINS),'Exactly four results stages')
    c.need(results['selected_submission']=='v22' and not results['new_CSV'] and not results['full_data_fit']
           and results['Public_score'] is None and results['Private_score'] is None and not results['fresh_blind_validation'],'Result role/scope')
    import pandas as pd
    raw=pd.read_pickle(c.safe(c.RAW))
    artifacts,trainmeta,native_facts,summary,boots={},{},{},{},[]
    score_count=0
    for origin in driver.ORIGINS:
        loaded=driver.load_context(request,origin);st,layers,ratio,meta,prep,checkpoint=checked[origin]
        original_array_binding(st,loaded['array_identities'],driver.CHANGED_ARRAY)
        same(st['context_before'],loaded['before'],'Actual saved origin state/frames/indices/fixed predictions')
        native_facts[origin]=check_native(origin,meta,loaded,request,driver,st,prep,ratio,layers['native'],checkpoint)
        ctx=loaded['context'];is_offset=origin==driver.OFFSET
        replay=driver.offset_comp.chain_prior(ctx,layers['native']) if is_offset else driver.comp.apply(ctx,layers['native'])[0]
        for key in LAYERS:c.exact(replay[key],layers[key],'Actual complete saved candidate chain '+key)
        metric=driver.offset_metric if is_offset else driver.metric
        y=metric.select_query_y(raw,loaded['indices']['query_source_cache_indices'],455784) if is_offset else metric.select_query_y(raw,loaded['indices']['query_source_cache_indices'])
        descriptive=(lambda a:metric.descriptive_metrics(y,a,loaded['fit_X'],loaded['query_X'])) if is_offset else (lambda a:metric.descriptive_metrics(y,a,loaded['fit_X'],loaded['query_X'],c.SPLITS[origin][0]))
        item=results['stages'][origin];stage_eval,stage_record=locked(BASE+origin+'/evaluation/results.json')
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
        same(item['paired']['caution'],driver.CAUTION,'Exact transaction and reused-window bootstrap limitation')
        same(item['both_point_metrics_improve'],all(item['metrics']['T'][k]<item['metrics']['v22'][k] for k in ('overall_mae','weighted_mae')),
             'Strict both-point-improvement label')
        point=metric.methods.point_values(np.abs(y[:,None]-np.column_stack([vectors['v22'],vectors['T']])),labels,weights)
        boots.extend(check_bootstrap(origin,item,point,0 if is_offset else c.STAGES.index(origin)))
        artifacts[origin]={'stage':seal['records'][origin],'training_metadata':st['model_metadata'],
                           'evaluation':stage_record,'sealed_models':st['sealed_models'],
                           'prior_preparation':st['prior_preparation'],'sealed_prior_artifacts':st['sealed_prior_artifacts']}
        trainmeta[origin]=meta
        summary[origin]={'overall_mae':item['metrics']['T']['overall_mae'],'weighted_mae':item['metrics']['T']['weighted_mae'],
                         'overall_delta_vs_v22':primary['overall_mae']['delta'],'weighted_delta_vs_v22':primary['weighted_mae']['delta'],
                         'overall_delta_vs_v24':secondary['overall_mae']['delta'],'weighted_delta_vs_v24':secondary['weighted_mae']['delta'],
                         'adverse_groups':primary['quarter_or_large_group_deterioration_above_5pct'],
                         'v24_adverse_groups':secondary['quarter_or_large_group_deterioration_above_5pct'],
                         'fit_factor_change_summary':prep['fit_factor_change_summary'],
                         'original_initialization':prep['original_initialization'],'transaction_initialization':prep['transaction_initialization'],'fit_seconds':meta['fit_seconds']}
        driver.baseline.verify_context(loaded,origin)
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
    payload={'status':'passed','reviewer':'research_preflight_20261008','created_at_utc':c.now(),'round':117,
             'scope':'Independent CPU post-evaluation real fit-only prior replay/checkpoint/ratio restoration/composition/metrics/gates audit; accuracy gate recorded separately.',
             'source_sha256':merged,'plan':request['plan'],'request':rr,'seal':sr,'training_execution':tr,
             'evaluation_execution':er,'results':rs,'preflight_review':pfr,'review_tool':c.record(__file__),
             'independent_artificial_prior_oracle':oracle_record,
             'recovery_plan':request['recovery_plan'],'engineering_recovery':request['engineering_recovery'],
             'historical_aborted_batch_completed_NN_fits':1,'fresh_NN_fits_in_this_batch':4,
             'total_R117_actual_NN_fits_including_aborted':5,
             'old_partial_models_or_priors_loaded_or_reused':False,
             'review_dependency_source_sha256':deps,'artifacts':artifacts,'trainmeta':trainmeta,
             'manual_gate_verdicts':verdicts,'stage_summary':summary,'native_identity_diagnostics':native_facts,
             'identity_checks':{'frozen_sources_including_science_and_recovery_plans':len(merged),'frozen_inputs':len(request['input_sha256']),
                                'protected_files':len(request['protected_before']),'sealed_origins':4,
                                'NN_checkpoints_CPU_inspected':4,'complete_chain_layers_replayed':20,
                                'actual_real_transaction_prior_folds_and_fullfit_states_replayed':4,
                                'original_eight_arrays_per_origin_byte_verified':32,'complement_date_summaries_independently_recomputed':20},
             'scoring_checks':{'overall_and_weighted_MAE_independently_recomputed':score_count,
                               'primary_and_v24_comparisons_independently_reconstructed':8,
                               'branch_metrics_checked':16,'ratio_factor_TWD_inverse_once_checked':4,
                               'bootstrap_files_summary_checked':len(boots)},'bootstrap_artifacts':boots,
             'protected_unchanged':True,'query_labels_used_for_postseal_review_only':True,
             'new_model_fits':0,'GPU_work':False,'new_learned_prior_states_persisted':0,
             'CPU_fit_only_prior_replay_calls':4,'Public_score':None,'Private_score':None,
             'selected_submission':'v22','new_CSV':False,'blocking_findings_remaining':[],
             'limits':['All four origins are reused and overlapping, not fresh blind validation or four independent votes.',
                       'This numerical postfit reviewer authored only the engineering-v2 driver; final source preflight/minimal-diff review is performed by root/another agent and bound separately here.',
                       'CPU checkpoint constructors/tensors/config and saved ratio-to-TWD inverse were inspected without independent full-query NN inference.',
                       'Full actual transaction OOF priors were replayed using frozen R60 estimator/key functions; independent full complement date/global median arithmetic and a 27-case artificial oracle supplement this shared-estimator replay.',
                       'No fresh neural network retraining, GPU inference or separately independent full-query NN forward pass was performed; saved ratio/TWD reload exactness is original frozen trainer evidence.',
                       'Composition and descriptive definitions reuse frozen helpers; independent MAE/group/gate arithmetic supplements replay.',
                       'Bootstrap summaries use saved replicates without resampling, selection/refit/future-shift correction; not Private probability.',
                       'Original R117 aborted batch retained one completed native before metadata failure; this engineering recovery trained four fresh priors/NNs, and all five historical native fits are counted.',
                       'No fullfit/release/CSV/Kaggle upload or v22 replacement is authorized by audit pass.'],
             'review_elapsed_seconds':time.monotonic()-tick}
    c.write(output,payload)
    print(json.dumps({'status':'passed','review':c.record(output),'manual_gate_verdicts':verdicts},ensure_ascii=False))


def self_check():
    import pandas as pd
    checks=[]
    X=pd.DataFrame({'building_area':[2.,np.nan,-1.],'parking_area':[9.,3.,np.nan],
        'trade_day_index':[1000.,1365.,np.nan],'trade_date_invalid':[0.,0.,1.],'trade_date_out_of_range':[0.,0.,0.]})
    c.exact(effective_area(X),np.array([2.,3.,1.],np.float64),'Area fallback fixture');checks.append('area_fallback')
    ratio=np.array([.5,2.,3.],np.float64);factor=np.array([2.,3.,4.],np.float32);native=ratio*factor.astype(np.float64)*1e6
    ratio_restore(ratio,native,factor);checks.append('ratio_inverse_once')
    for bad in (native/1e6,native*1e6):
        try:ratio_restore(ratio,bad,factor)
        except ValueError:checks.append('wrong_inverse_rejected')
        else:raise AssertionError('Extra/missing inverse accepted')
    y=np.array([1.,2.,3.],np.float32);f=np.full(3,2.,np.float32)
    same(ratio_bias(y,f)['initial_training_median_ratio'],1.,'Original median fixture');checks.append('bias_median')
    c.need(ratio_bias(y,f)!=ratio_bias(y,f*2),'Linked bias changes');checks.append('bias_linked_to_factor')
    weights,summary=date_summary(X,np.ones(3,bool),{'artificial':'rules'})
    c.exact(weights,np.array([2/3,4/3,1.],np.float64),'Manual neutral1/date mean-valid fixture');checks.append('complement_date_weight_arithmetic')
    same(summary['max_valid_reference_trade_day_index'],1365.,'Reference-only anchor');checks.append('date_anchor')
    same(manual_weighted_median(np.array([1.,9.]),np.array([1.,1.])),5.,'Exact-half midpoint');checks.append('manual_exact_half')
    same(manual_weighted_median(np.array([1.,9.]),np.array([1.,2.])),9.,'Weighted recency direction');checks.append('manual_weighted_median')
    folds=fixed_folds(13);same(sorted(np.bincount(folds).tolist()),[2,2,3,3,3],'Balanced folds');checks.append('fixed_transaction_folds')
    class Archive:
        def __init__(self,files):self.files,self.reads=files,[]
        def __getitem__(self,key):self.reads.append(key);raise AssertionError('No reads before key guard')
    for names in (list(c.ARRAY_NAMES)+['query_y'],list(c.ARRAY_NAMES)+['train_num'],list(c.ARRAY_NAMES[:-1])):
        archive=Archive(names)
        try:archive_arrays(archive)
        except ValueError:c.need(not archive.reads,'Bad archive read values before guard');checks.append('query_y_duplicate_missing_keys_rejected_before_read')
        else:raise AssertionError('Bad archive accepted')
    def metric(v=100.):
        item={'overall_mae':v,'weighted_mae':v,'price_bands':[{'band':'x','rows':500,'mae':100.}]}
        for k in ('strata','forecast_quarters','months','cities','relative_forecast_months'):item[k]={'x':{'rows':500,'mae':100.}}
        return item
    for offset in (False,True):
        for secondary in (False,True):
            comp=comparison(metric(),metric(99.),offset,secondary)
            same(comp['candidate'],'R117','Candidate label');same(comp['control'],'v24' if secondary else 'v22','Control label')
            same(comp['used_for_release_gate'],not secondary,'Gate/counterevidence scope');checks.append('origin_comparison_wrapper')
    for value,want in ((99.,True),(100.25,True),(100.25001,False)):
        m={'v22':metric(),'T':metric(value)}
        same(extra_gate(m,comparison(m['v22'],m['T'],True))['passed'],want,'Extra boundary');checks.append('extra_boundary')
    import copy
    from types import SimpleNamespace
    original={key:{'values_sha256':'original_'+key} for key in c.ARRAY_NAMES}
    revised=copy.deepcopy(original);revised['train_price_factor_million']={'values_sha256':'changed_F'}
    fake_stage={'original_prepared_array_identities':original,'prepared_array_identities':revised}
    original_array_binding(fake_stage,original,'train_price_factor_million');checks.append('changed_F_is_not_mislabelled_original_array_identity')
    for bad in ('original','immutable'):
        altered=copy.deepcopy(fake_stage)
        altered['original_prepared_array_identities' if bad=='original' else 'prepared_array_identities']['train_num']={'values_sha256':'bad'}
        try:original_array_binding(altered,original,'train_price_factor_million')
        except ValueError:checks.append('wrong_original_or_immutable_identity_rejected')
        else:raise AssertionError('Wrong original immutable array accepted')
    actual_names=('src/round117_transaction_prior_train.py','src/round63_recency_weighted_prior_train.py',
                  'src/round47_nn_train.py','src/round52_nn_train.py')
    actual={key:'sha_'+key for key in actual_names}
    recovery={'old_partial_models_or_priors_loaded_or_reused':False,'explicit_engineering_recovery_not_automatic_retry':True,
              'scientific_plan_unchanged':True,'query_prices_or_new_scores_read':False}
    record={'path':RECOVERY_PLAN,'sha256':'fixed_amendment_sha'}
    fake_request={'recovery_plan':record,'engineering_recovery':recovery,'source_sha256':actual,
                  'historical_aborted_batch_completed_NN_fits':1,'planned_fresh_NN_fits':4,'maximum_R117_total_actual_NN_fits':5}
    fake_results={'recovery_plan':record,'engineering_recovery':recovery,
                  'historical_aborted_batch_completed_NN_fits':1,'fresh_NN_fits_in_this_batch':4,'total_R117_actual_NN_fits_including_aborted':5}
    fake_driver=SimpleNamespace(RECOVERY_PLAN_SHA='fixed_amendment_sha',recovery_specification=lambda unused:recovery,
                                actual_trainer_lineage=lambda:actual)
    old_record=c.record
    try:
        c.record=lambda name:record if name==RECOVERY_PLAN else (_ for _ in ()).throw(AssertionError('No real record read in artificial recovery guard'))
        recovery_bindings(fake_request,fake_results,fake_driver);checks.append('engineering_and_historical_five_native_count_binding')
        for case in ('missing_actual_R52','wrong_actual_R52_SHA','missing_historical_native','false_fresh_count','wrong_results_amendment'):
            r=copy.deepcopy(fake_request);out=copy.deepcopy(fake_results)
            if case=='missing_actual_R52':r['source_sha256'].pop(actual_names[-1])
            elif case=='wrong_actual_R52_SHA':r['source_sha256'][actual_names[-1]]='wrong'
            elif case=='missing_historical_native':r['historical_aborted_batch_completed_NN_fits']=0
            elif case=='false_fresh_count':out['fresh_NN_fits_in_this_batch']=3
            else:out['recovery_plan']={'path':RECOVERY_PLAN,'sha256':'wrong'}
            try:recovery_bindings(r,out,fake_driver)
            except ValueError:checks.append(case+'_rejected')
            else:raise AssertionError('Bad recovery binding accepted '+case)
    finally:c.record=old_record
    for path in (Path(__file__),Path(native_audit.__file__)):compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    print(json.dumps({'status':'passed','check_count':len(checks),'checks':checks,'review_tool':c.record(__file__),
                      'native_audit_tool':c.record(native_audit.__file__),'new_fits':0,'GPU_work':False,
                      'real_arrays_or_query_y_read':False},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-check',action='store_true');g.add_argument('--evaluation-complete',action='store_true');a=p.parse_args()
    if a.self_check:self_check()
    else:run()
