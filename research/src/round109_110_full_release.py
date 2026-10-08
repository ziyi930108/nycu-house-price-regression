"""Separate gated R109/v26 or R110/v27 full-data fit and exploratory release.

Implementation alone grants no fit: each round needs completed independent
postfit review, its own final plan and an independent source preflight. R109
uses fixed R63 weighted MAE with epochs 21--30 parameter averaging. R110 uses
the original R79 area-date unit model plus global target-border count eight.
Neither candidate enters the other's branch. No automatic Kaggle action.
"""
from pathlib import Path
import argparse
import contextlib
import csv
import json
import subprocess
import time
import traceback
import numpy as np
import round100_contract as c
import round107_final_candidate as final
import round107_full_release as shared
import round109_110_research as history

FOLDER='outputs/research_round109_110_20261008/'
TRAINER='src/round109_tail_parameter_average_train.py'
HELPER='src/round110_target_borders_unit.py'
BUILDER=shared.BUILDER
NODE=shared.NODE
VERSIONS={109:26,110:27}


def paths(round_number,version):
    c.need(type(round_number) is int and type(version) is int
        and VERSIONS.get(round_number)==version,'Explicit mapping is R109/v26 or R110/v27')
    base=f'outputs/round{round_number}/final'
    return {'round':round_number,'version':version,'base':base,'request':base+'/request.json',
        'seal':base+'/seal.json','CSV':f'outputs/submission_v{version}_trial.csv',
        'rows':f'outputs/research_csv_intermediate_20261008/v{version}_rows.json',
        'plan':f'reports/round{round_number}_final_plan.md',
        'preflight':FOLDER+f'round{round_number}_final_release_preflight.json'}


def gate(review,results,n):
    c.need(review['status']=='passed' and results['status']=='three_reused_windows_evaluated',
        'Independent completed postfit review required')
    c.need(results['gates']['exploratory_submission']['passed'] is True
        and review['manual_gate_verdicts'][str(n)]['exploratory_submission_eligible'] is True,
        'This candidate must independently pass its fixed exploratory gate')
    c.need(results['Private_score'] is None and results['gates']['private_score'] is None
        and results['gates']['selected_submission']=='v22'
        and results['gates']['automatic_selection'] is False,'Research selection/scores unchanged')


def bind_records(value,inputs):
    """Bind actual records from the completed reviewer, without reading prices."""
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            name=c.rel(value['path']);digest=value['sha256']
            c.need(name not in inputs or inputs[name]==digest,'Conflicting reviewed record')
            c.need(c.sha(name)==digest,'Reviewed record changed: '+name);inputs[name]=digest
        for child in value.values():bind_records(child,inputs)
    elif isinstance(value,list):
        for child in value:bind_records(child,inputs)


def sources():
    result=c.source_hashes(__file__,TRAINER,HELPER)
    result[BUILDER]=c.sha(BUILDER)
    return result


def freeze(p):
    c.need(not c.safe(p['base']).exists() and not c.safe(p['CSV']).exists()
        and not c.safe(p['rows']).exists(),'Preserve any previous final attempt/CSV')
    review_path=FOLDER+'postfit_review.json';review=c.read(review_path)
    result_path=f"outputs/round{p['round']}/results.json";results=c.read(result_path)
    gate(review,results,p['round'])
    c.need(review['results'][str(p['round'])]==c.record(result_path),'Reviewed results identity')
    for key in ('request','seal','training_execution','evaluation_execution','review_tool','plan'):
        c.need(review[key]==c.record(review[key]['path']),'Completed review identity: '+key)
    c.need(c.read(review['training_execution']['path'])['exit_code']==0
        and c.read(review['evaluation_execution']['path'])['exit_code']==0,'Six fits and evaluation must complete')
    c.verify_hashes(review['source_sha256'])
    expected=[]
    for stage in c.STAGES:
        rec=review['artifacts'][f"{p['round']}/{stage}"]['training_metadata']
        c.need(c.record(rec['path'])==rec,'Reviewed native metadata identity')
        metadata=c.read(rec['path'])
        c.need(metadata==review['trainmeta'][f"{p['round']}/{stage}"],'Reviewer metadata projection differs')
        expected.append(metadata['configuration' if p['round']==109 else 'parameters'])
    c.need(all(item==expected[0] for item in expected),'All three windows used the same fixed recipe')
    preflight=c.read(p['preflight']);current_sources=sources()
    c.need(preflight['status']=='passed' and preflight['round']==p['round']
        and preflight['version']==p['version'],'Independent round/version final preflight required')
    c.verify_hashes(preflight['source_sha256'])
    c.need(all(preflight['source_sha256'].get(name)==digest for name,digest in current_sources.items())
        and preflight['source_sha256'].get(p['plan'])==c.sha(p['plan']),
        'Preflight must cover complete final source closure, builder and new plan')
    spec=final.specification();inputs=dict(spec['input_sha256'])
    inputs.update(review['source_sha256']);bind_records(review,inputs)
    for name in (review_path,result_path,p['plan'],p['preflight'],
                 'outputs/research_resume_20261008/final_context_preflight.json'):
        inputs[name]=c.sha(name)
    c.verify_hashes(current_sources);c.verify_hashes(inputs)
    c.safe(p['base']).mkdir(parents=True,exist_ok=False)
    c.write(p['request'],{'status':'frozen_before_one_exploratory_final_fit','created_at_utc':c.now(),
        'round':p['round'],'version':p['version'],'paths':p,'sources':current_sources,'inputs':inputs,
        'context':spec,'expected_recipe':expected[0],'protected_before':history.protected(),
        'postfit_review':c.record(review_path),'gates':results['gates'],'new_fits':1,
        'query_labels_available':False,'selected_submission_preserved':'v22',
        'Public_score':None,'Private_score':None})


def request(p):
    r=c.read(p['request'])
    c.need(r['status']=='frozen_before_one_exploratory_final_fit' and r['paths']==p
        and r['round']==p['round'] and r['version']==p['version'] and r['new_fits']==1,
        'Correct immutable final request required')
    c.verify_hashes(r['sources']);c.verify_hashes(r['inputs']);c.verify_hashes(r['protected_before'])
    return r


def prepared(ctx):
    original=c.read(ctx['specification']['original_training']['path'])
    prep=c.read(ctx['specification']['preparation']['path'])
    with np.load(c.safe(ctx['prepared']['path']),allow_pickle=False) as z:
        c.need(set(z.files)==set(c.ARRAY_NAMES),'Original nine arrays only; no query labels')
        c.exact(z['train_y_million'],np.asarray(ctx['y']/1e6,dtype=np.float32),'Canonical final fit y alignment')
        c.need(z['train_num'].shape==(467221,144) and z['train_cat'].shape==(467221,23)
            and z['query_num'].shape==(47073,144) and z['query_cat'].shape==(47073,23),'Final NN dimensions')
        ids={key:c.identity(z[key]) for key in c.ARRAY_NAMES}
        for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
            c.need(ids[key]['values_sha256']==original[key+'_sha256'],'Original '+key+' identity')
        c.need(ids['train_sample_weight']['values_sha256']==prep['train_sample_weight_sha256']
            and z['cardinalities'].tolist()==original['cardinalities'],'Original weights/cardinalities')
        weights=z['train_sample_weight'].copy()
    return original,ids,weights


def unit_chain(ctx,native):
    c.need(native.dtype==np.float64 and native.shape==(47073,) and np.isfinite(native).all()
        and (native>=0).all(),'Valid unit-model TWD query vector')
    fixed=ctx['vectors'];base=.75*native+.25*fixed['lgb']
    reference,diagnostics,_=final.f79.exp.original.predict_reference(
        ctx['reference'],ctx['Xq'],base,variant='trend',return_diagnostics=True)
    spatial,detail,_=final.f79.spatial.predict(ctx['spatial_state'],ctx['Xq'],ctx['query_geo'],
        base,reference,diagnostics)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback],reference[fallback],'Unit spatial fallback exact')
    median=np.median(np.column_stack([fixed['nn'],fixed['log'],spatial]),axis=1)
    return {'native':native,'base':base,'reference':reference,'spatial':spatial,'median':median}


def check_nn(meta,r,original,model):
    c.need(meta['status']=='completed' and meta['source_sha256']==r['sources'][TRAINER]
        and meta['configuration']==r['expected_recipe']==original['configuration']
        and meta['configuration']['seed']==42 and meta['configuration']['epochs']==30
        and meta['complete_R63_training_configuration_unchanged']
        and meta['changed_configuration_fields']==[] and meta['added_configuration_fields']==[]
        and meta['training_objective_name']=='weighted MAE' and not meta['epoch_selection_uses_training_metrics'],
        'Actual original R63 recipe with fixed parameter averaging')
    c.need(meta['input_sha256']==r['context']['prepared']['sha256'] and meta['fit_rows']==467221
        and meta['query_rows']==47073 and meta['numeric_count']==144
        and meta['cardinalities']==original['cardinalities'] and not meta['query_labels_used']
        and not meta['query_labels_available'] and not meta['sample_weights_batch_renormalized']
        and meta['saved_model_reload_exact'] and meta['saved_epoch30_model_reload_exact']
        and meta['reload_rows']==47073,'Actual R109 input/scope/reload contract')
    for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256',
                'train_sample_weight_sha256','initial_training_median_ratio','initial_head_bias'):
        c.need(meta[key]==original[key],'Original final preparation: '+key)
    average=meta['tail_parameter_average']
    c.need(average['epochs']==average['collected_epochs']==list(range(21,31))
        and average['count']==average['collected_count']==10 and average['weights']==[.1]*10
        and average['accumulator_device']=='cpu' and average['accumulator_dtype']=='float64'
        and average['input_parameter_dtype']==average['output_parameter_dtype']=='float32'
        and average['module_buffer_count']==0 and average['parameters_averaged_before_inference']
        and not average['predictions_averaged'] and not average['optimizer_or_scheduler_changed']
        and average['one_training_trajectory_for_candidate_and_control'],'Fixed ten-state parameter mean')
    c.need([e['epoch'] for e in meta['epochs']]==list(range(1,31))
        and all(e['tail_parameter_state_collected']==(e['epoch']>=21)
            and e['tail_parameter_average_count']==max(0,e['epoch']-20) for e in meta['epochs']),
        'Actual collection timeline')
    records={}
    for key,model_name,prediction_name,epochs in (
        ('averaged_candidate','model.pt','predictions_twd.npy',list(range(21,31))),
        ('epoch30_control','model_epoch30.pt','predictions_epoch30_twd.npy',[30])):
        item=meta[key]
        c.need(item['model_file']==model_name and item['predictions_file']==prediction_name
            and item['parameter_state_epochs']==epochs and item['saved_model_reload_exact']
            and item['reload_rows']==47073,'Both actual NN checkpoint reload identities')
        c.need(c.sha(model/model_name)==item['model_sha256']
            and c.sha(model/prediction_name)==item['predictions_file_sha256'],'Both NN artifact SHAs')
        records[key]={'model':c.record(model/model_name),'native_predictions':c.record(model/prediction_name)}
    return records


def check_ctr(meta,r,ctx,weights,model):
    import round110_target_borders_unit as helper
    c.need(meta['status']=='completed' and meta['parameters']==r['expected_recipe']==helper.parameters()
        and meta['source_sha256']==helper.source_hashes() and meta['fit_rows']==467221
        and meta['query_rows']==47073 and meta['new_native_fits']==1
        and meta['feature_count']==95 and meta['categorical_feature_count']==23
        and meta['actual_trees']==2999 and meta['actual_max_ctr_complexity']==2
        and meta['actual_ctr_target_border_count']==8 and meta['actual_seed']==42
        and meta['actual_loss_function']=='MAE','Actual R110 native global-eight recipe')
    c.need(meta['saved_model_reload_exact'] and meta['raw_and_TWD_reload_exact'] and meta['reload_rows']==47073
        and not meta['query_labels_used'] and not meta['query_labels_available']
        and not meta['fit_uses_eval_set'] and not meta['early_stopping_used']
        and not meta['date_weights_recomputed'] and not meta['date_weights_renormalized']
        and not meta['weights_normalized_to_mean_one'] and not meta['weights_use_query']
        and not meta['reference_or_trend_applied_here'] and not meta['spatial_applied_here'],
        'Original R110 weights/query/saved-reload scope')
    dates=helper.baseline._date_values(weights,ctx['X'])
    for key,value in {'fit_target_values_sha256':helper.original.values_sha256(ctx['y']),
        'fit_transformed_target_values_sha256':helper.original.values_sha256(helper.original.transform_target(ctx['y'],ctx['X'])),
        'fit_date_weights_sha256':c.values_sha(dates),
        'fit_sample_weights_sha256':helper.original.values_sha256(helper.sample_weights(ctx['X'],weights))}.items():
        c.need(meta[key]==value,'Actual R110 aligned labels/weights: '+key)
    state=c.read(model/'feature_state.json')
    helper._check_state(state,ctx['X'],ctx['cats']);helper._check_state(state,ctx['Xq'],ctx['cats'])
    helper.compare_reload_diagnostics(meta['fit_native_diagnostics'],meta['reloaded_native_diagnostics'])
    for diagnostics in (meta['fit_native_diagnostics'],meta['reloaded_native_diagnostics']):
        c.need(diagnostics['actual_ctr_target_border_count']==8 and diagnostics['actual_max_ctr_complexity']==2
            and diagnostics['actual_trees']==2999 and diagnostics['native_feature_count']==95
            and diagnostics['native_categorical_feature_count']==23,'Fitted/reloaded native structure')
    records={}
    for name,key in (('model.cbm','model_sha256'),('feature_state.json','feature_state_sha256'),
        ('raw.npy','raw_prediction_sha256'),('native.npy','prediction_sha256'),
        ('target_border_summary.json','target_border_summary_sha256')):
        c.need(c.sha(model/name)==meta[key],'Native saved artifact identity: '+name)
        records[name]=c.record(model/name)
    summary=c.read(model/'target_border_summary.json')
    c.need(summary==meta['target_border_summary'] and summary['global_ctr_target_border_count']==8
        and summary['tree_count']==2999 and summary['all_eight_border_indices_required'] is False
        and summary['embedded_descriptor_TargetBorderCount_8_required'] is False,
        'Observed CTR usage must not be confused with the global recipe')
    raw=np.load(model/'raw.npy',allow_pickle=False);native=np.load(model/'native.npy',allow_pickle=False)
    c.exact(helper.original.inverse_transform(raw,ctx['Xq']),native,'Saved unit-to-TWD inverse exact')
    return records


def fit(p):
    r=request(p)
    c.need(history.protected()==r['protected_before'] and not c.safe(p['CSV']).exists(),
        'Serialize final candidates; protected inventory must still match')
    c.need(not c.safe(p['base']+'/fit_started.json').exists(),'One final fit invocation only; preserve failures')
    c.write(p['base']+'/fit_started.json',{'created_at_utc':c.now(),'request':c.record(p['request']),
        'new_fits_authorized':1,'automatic_retry':False})
    tick=time.monotonic();failure=None
    try:
        ctx=final.load_context(r['context'])
        c.need(set(ctx['replay_checks'])==shared.REPLAY_KEYS and all(v is True for v in ctx['replay_checks'].values()),
            'Original eleven v22 replay checks')
        original,array_ids,weights=prepared(ctx);model=c.safe(p['base'])/'model';controls=None
        if p['round']==109:
            cmd=[str(c.ROOT/'.venv_torch/Scripts/python.exe'),'-u','-B','-X','utf8',str(c.safe(TRAINER)),
                '--data',str(c.safe(ctx['prepared']['path'])),'--out',str(model)]
            with (c.safe(p['base'])/'train.log').open('x',encoding='utf8') as log:
                process=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            c.need(process.returncode==0,'R109 fit failed; no automatic retry')
            metadata_path=model/'training.json';meta=c.read(metadata_path)
            model_records=check_nn(meta,r,original,model)
            native=np.load(model/'predictions_twd.npy',allow_pickle=False)
            chain=final.apply_prior(ctx,native)[0]
            control_native=np.load(model/'predictions_epoch30_twd.npy',allow_pickle=False)
            controls=final.apply_prior(ctx,control_native)[0]
        else:
            import round110_target_borders_unit as helper
            with (c.safe(p['base'])/'train.log').open('x',encoding='utf8') as log:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    native,meta=helper.fit_predict(ctx['X'],ctx['y'],ctx['G'],ctx['Xq'],ctx['Gq'],
                        ctx['cats'],weights,model)
            metadata_path=model/'metadata.json';c.need(c.read(metadata_path)==meta,'Saved R110 metadata equality')
            model_records=check_ctr(meta,r,ctx,weights,model)
            original_unit=unit_chain(ctx,ctx['vectors']['unit_native'])
            for key,saved in [('base','unit_base'),('reference','unit_reference'),('spatial','unit'),('median','v22')]:
                c.exact(original_unit[key],ctx['vectors'][saved],'Original unit final replay '+key)
            chain=unit_chain(ctx,native)
        c.need(set(chain)==shared.LAYERS,'Five final layers');c.exact(chain['native'],native,'Native slot unchanged')
        rounded=shared.rounded_prices(chain['median'])
        records={key:c.save_array(c.safe(p['base'])/(key+'.npy'),value) for key,value in chain.items()}
        control_records=None if controls is None else {
            key:c.save_array(c.safe(p['base'])/('epoch30_'+key+'.npy'),value) for key,value in controls.items()}
        final.verify_unchanged(ctx);request(p)
        c.need(history.protected()==r['protected_before'],'Old protected files unchanged before CSV')
        c.write(p['seal'],{'status':'full_model_prediction_sealed','request':c.record(p['request']),
            'round':p['round'],'version':p['version'],'model_artifacts':model_records,
            'training':c.record(metadata_path),'layers':records,'same_trajectory_epoch30_layers':control_records,
            'v22_replay_checks':ctx['replay_checks'],'prepared':ctx['prepared'],'prepared_array_identities':array_ids,
            'new_fits':1,'fit_seconds':meta['fit_seconds'],'elapsed_seconds':time.monotonic()-tick,
            'query_labels_used':False,'new_CSV':False})
        c.safe(p['rows']).parent.mkdir(parents=True,exist_ok=True)
        c.write(p['rows'],{'status':'full_model_prediction_sealed','seal':c.record(p['seal']),
            'rounding':'maximum(rint(final_TWD),1).astype(int64)',
            'rows':[[key,int(value)] for key,value in zip(ctx['ids'],rounded,strict=True)]})
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(p['base']+'/fit_execution.json',{'exit_code':1 if failure else 0,'failure':failure,
        'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False})


def release(p):
    r=request(p);tick=time.monotonic()
    c.need(history.protected()==r['protected_before'] and not c.safe(p['CSV']).exists(),
        'Serialize authoring; preserve any previous CSV')
    seal=c.read(p['seal'])
    c.need(seal['status']=='full_model_prediction_sealed' and seal['request']==c.record(p['request'])
        and seal['round']==p['round'] and seal['version']==p['version'],'Final prediction seal identity')
    execution=c.read(p['base']+'/fit_execution.json')
    c.need(execution['exit_code']==0 and execution['failure'] is None,'Final fit completed successfully')
    records={};bind_records(seal['model_artifacts'],records);bind_records(seal['training'],records)
    c.need(set(seal['layers'])==shared.LAYERS,'Five sealed layers required')
    layers={key:c.load_array(rec) for key,rec in seal['layers'].items()}
    controls=seal['same_trajectory_epoch30_layers']
    c.need((controls is not None)==(p['round']==109),'Epoch30 control belongs to R109 only')
    if controls is not None:
        c.need(set(controls)==shared.LAYERS,'Five epoch30 control layers')
        for rec in controls.values():c.load_array(rec)
    c.need(set(seal['v22_replay_checks'])==shared.REPLAY_KEYS
        and all(v is True for v in seal['v22_replay_checks'].values()),'Original replay evidence')
    rows=c.read(p['rows']);c.need(rows['status']=='full_model_prediction_sealed'
        and rows['seal']==c.record(p['seal']),'Authoring input seal')
    prediction=layers['median'];rounded=shared.rounded_prices(prediction)
    with c.safe(r['context']['sample_path']).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);c.need(next(reader)==['編號','總價元'],'Sample headers');sample=list(reader)
    ids=[row[0] for row in sample]
    c.need(len(ids)==47073 and len(set(ids))==47073 and all(len(row)==2 for row in sample)
        and rows['rows']==[[key,int(value)] for key,value in zip(ids,rounded,strict=True)],'All authored IDs/rounded prices')
    cmd=[str(NODE),str(c.safe(BUILDER)),str(c.safe(p['rows'])),str(c.safe(p['CSV'])),str(c.safe(p['base']))]
    with (c.safe(p['base'])/'csv_authoring.log').open('x',encoding='utf8') as log:
        process=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    c.need(process.returncode==0,'CSV authoring failed; preserve evidence')
    with c.safe(p['CSV']).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);c.need(next(reader)==['編號','總價元'],'Actual CSV headers');values=list(reader)
    c.need(values==[[key,str(value)] for key,value in zip(ids,rounded,strict=True)]
        and c.safe(p['CSV']).read_bytes().startswith(b'\xef\xbb\xbf'),'Independent all CSV values/order/BOM exact')
    author=c.read(p['base']+'/csv_authoring_receipt.json')
    c.need(author['status']=='artifact_typed_range_exported_to_csv' and author['input_json_sha256']==c.sha(p['rows'])
        and author['csv_sha256']==c.sha(p['CSV']) and author['rows']==47073 and author['columns']==['編號','總價元']
        and author['full_workbook_values_exact'] and author['formula_count']==0 and author['utf8_bom'],
        'Artifact typed-range export identity')
    request(p);before=r['protected_before'];after=history.protected()
    c.need(set(after)==set(before)|{p['CSV']} and all(after[key]==value for key,value in before.items()),
        'Exactly one new candidate CSV; all old protected files unchanged')
    original=np.load(c.safe(final.VECTORS['v22']),allow_pickle=False);diff=prediction-original
    receipt={'status':'exploratory_submission_ready_not_selected','created_at_utc':c.now(),
        'round':p['round'],'version':p['version'],'CSV':c.record(p['CSV']),'rows':47073,'columns':['編號','總價元'],
        'request':c.record(p['request']),'seal':c.record(p['seal']),
        'authoring':c.record(p['base']+'/csv_authoring_receipt.json'),
        'all_sample_ids_and_rounded_values_exact':True,'positive_integer_prices':True,
        'same_order_as_sample':True,'old_protected_unchanged':True,'query_labels_used':False,
        'Public_score':None,'Private_score':None,'automatic_Kaggle_upload_or_selection':False,
        'selected_submission_preserved':'v22','gate':r['gates'],
        'test_prediction_difference_not_MAE':{'mean_signed_TWD':float(diff.mean()),
            'mean_absolute_TWD':float(np.mean(np.abs(diff))),'max_absolute_TWD':float(np.max(np.abs(diff))),
            'changed_rounded_rows':int(np.count_nonzero(rounded!=shared.rounded_prices(original)))},
        'prediction_min_TWD':float(prediction.min()),'prediction_max_TWD':float(prediction.max()),
        'elapsed_seconds':time.monotonic()-tick,'clean_environment_end_to_end_retraining_verified':False}
    c.write(p['base']+'/release.json',receipt)
    print(json.dumps({'CSV':p['CSV'],'sha256':receipt['CSV']['sha256'],'rows':47073,'status':receipt['status']}),flush=True)


def self_check():
    """Pure synthetic contract checks; no real data, model, fit or GPU calls."""
    import copy
    checks=[]
    def ok(name,condition):c.need(condition,name);checks.append(name)
    def rejects(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError('Accepted invalid contract: '+name)
    ok('separate_round_paths',paths(109,26)['base']!=paths(110,27)['base'])
    ok('explicit_version_mapping',paths(109,26)['CSV'].endswith('v26_trial.csv')
        and paths(110,27)['CSV'].endswith('v27_trial.csv'))
    rejects('no_swapped_versions',lambda:paths(109,27));rejects('no_unknown_round',lambda:paths(111,28))
    review={'status':'passed','manual_gate_verdicts':{'109':{'exploratory_submission_eligible':True},
        '110':{'exploratory_submission_eligible':False}}}
    results={'status':'three_reused_windows_evaluated','Private_score':None,
        'gates':{'exploratory_submission':{'passed':True},'private_score':None,
            'selected_submission':'v22','automatic_selection':False}}
    gate(review,results,109);ok('eligible_only_gate',True)
    rejects('ineligible_round_rejected',lambda:gate(review,results,110))
    bad=copy.deepcopy(review);bad['status']='pending'
    rejects('pending_review_rejected',lambda:gate(bad,results,109))
    bad=copy.deepcopy(results);bad['gates']['exploratory_submission']['passed']=False
    rejects('failed_gate_rejected',lambda:gate(review,bad,109))
    bad=copy.deepcopy(results);bad['Private_score']=1.
    rejects('unknown_Private_not_claimed',lambda:gate(review,bad,109))
    values=np.full(47073,2.5,dtype=np.float64);values[:5]=[0,.5,1.5,2.5,3.5]
    ok('banker_rounding_and_minimum_one',shared.rounded_prices(values)[:5].tolist()==[1,1,2,2,4])
    for label,value in [('NaN',np.nan),('negative',-1.),('unsafe',float(2**53))]:
        rejects('rounding_rejects_'+label,lambda value=value:shared.rounded_prices(np.full(47073,value)))
    return {'status':'passed','checks':checks,'check_count':len(checks),'real_data_read':False,
        'query_prices_or_scores_read':False,'new_fits':0,'GPU_calls':0,'new_CSV':False,
        'final_freeze_executed':False,'source_sha256':sources(),
        'future_requirements':'Completed postfit_review plus separately approved round final plan/preflight before freeze.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',dest='round_number',type=int,choices=(109,110))
    parser.add_argument('--version',type=int,choices=(26,27))
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--fit',action='store_true')
    group.add_argument('--release',action='store_true');group.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)
    else:
        if args.round_number is None or args.version is None:parser.error('--round and --version are explicitly required')
        selected=paths(args.round_number,args.version)
        if args.freeze:freeze(selected)
        elif args.fit:fit(selected)
        else:release(selected)
