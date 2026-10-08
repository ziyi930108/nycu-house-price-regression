"""Postseal CPU native identity audit helpers; imports never read data or fit."""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import round100_contract as c


def same(a,b,name):c.need(a==b,name)


def inspect_checkpoint(path):
    import torch
    import round111_residual_prior_train as trainer
    torch.set_num_threads(4)
    trainer.check_configuration()
    payload=torch.load(c.safe(path),weights_only=True,map_location='cpu')
    restored=trainer.restore_checkpoint(payload,torch.device('cpu'))
    states=payload['state_dict']
    c.need(not list(restored.named_buffers()),'R111 must have no running-statistic buffers')
    parameters={}
    for name,value in states.items():
        c.need(isinstance(value,torch.Tensor) and value.device.type=='cpu'
               and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Invalid NN parameter')
        parameters[name]={'shape':list(value.shape),'dtype':str(value.dtype),'values_sha256':c.values_sha(value.numpy())}
    record={k:v for k,v in payload.items() if k!='state_dict'}
    return {'payload':record,'parameters':parameters,'named_parameter_order':list(states),
            'parameter_count':sum(v.numel() for v in restored.parameters()),
            'embedding_dimensions':restored.dimensions,'buffer_count':len(list(restored.named_buffers())),
            'CPU_only':True,'model_inference_performed':False,'model_fits':0,'GPU_calls':0}


def checkpoint_cpu(path):
    cmd=[str(ROOT/'.venv_torch/Scripts/python.exe'),'-B','-X','utf8',str(Path(__file__).resolve()),'--checkpoint',str(c.safe(path))]
    result=subprocess.run(cmd,cwd=ROOT,text=True,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    c.need(result.returncode==0,'CPU checkpoint inspect failed: '+result.stderr)
    return json.loads(result.stdout)


def check_nn(stage,st,meta,spec,arrays):
    prefix=f'outputs/round111/{stage}/model/'
    old_path=f'outputs/round63/{stage}/model/training.json'
    old=c.read(old_path)
    authority=c.read('outputs/round63/results.json')['stages'][stage]['artifact_sha256'][old_path]
    same(c.sha(old_path),authority,'Original R63 metadata source')
    expected=c.read('outputs/research_continue_20261008/pipeline_review/independent_R111_cpu_checks_v2.json')['result']
    same(meta['configuration'],expected['configuration'],'R111 fixed config')
    same(meta['baseline_configuration'],old['configuration'],'R111 complete baseline recipe')
    same(meta['source_sha256'],c.sha('src/round111_residual_prior_train.py'),'R111 actual trainer source')
    same(meta['architecture_source_sha256'],meta['source_sha256'],'R111 architecture source')
    same(meta['parent_trainer_source_sha256'],c.sha('src/round63_recency_weighted_prior_train.py'),'R111 original R63 source')
    same(meta['original_architecture_source_sha256'],c.sha('src/round47_nn_train.py'),'Original R47 source')
    same(meta['input_sha256'],spec['prepared']['sha256'],'R111 NPZ identity')
    same(meta['input_arrays'],{k:c.identity(v) for k,v in arrays.items()},'R111 all nine input array identities')
    same((meta['fit_rows'],meta['query_rows']),c.SPLITS[stage][2:],'R111 fit/query rows')
    same((meta['numeric_count'],meta['categorical_count']),(144,23),'R111 input width')
    same(meta['cardinalities'],arrays['cardinalities'].tolist(),'R111 fitted cardinalities')
    same(meta['actual_model_class'],'round111_residual_prior_train.TabularResidualPrior','R111 actual class')
    c.need(meta['complete_R63_nonarchitecture_training_configuration_unchanged']
           and meta['original_configuration_and_source_verified'] and not meta['original_globals_modified']
           and meta['capacity_initialization_and_rng_consumption_also_change']
           and not meta['skip_connection_only_causal_claimed'] and not meta['official_paper_replication_claimed'],
           'R111 changed-method scope differs')
    c.need(meta['status']=='completed' and meta['training_objective_name']=='weighted MAE'
           and not meta['epoch_selection_uses_training_metrics'] and not meta['query_labels_available']
           and not meta['query_labels_used'] and not meta['sample_weights_recomputed_in_trainer']
           and not meta['sample_weights_batch_renormalized'] and not meta['price_or_ratio_clipping']
           and not meta['BatchNorm_statistics_present_or_recomputed'],'R111 objective/scope')
    c.need(meta['saved_model_reload_exact'] and meta['saved_native_ratio_reload_exact']
           and meta['reload_rows']==c.SPLITS[stage][3],'R111 saved full-query exact reload')
    for key in ('train_sample_weight','train_price_factor_million','query_price_factor_million'):
        same(meta[key+'_sha256'],c.values_sha(arrays[key]),'R111 factor/date weight '+key)
    same([e['epoch'] for e in meta['epochs']],list(range(1,31)),'Thirty fixed epochs')
    same([e['learning_rate'] for e in meta['epochs']],[e['learning_rate'] for e in old['epochs']],'Original learning-rate schedule')
    changed=sorted(k for k in old['configuration'] if meta['configuration'][k]!=old['configuration'][k])
    added=sorted(set(meta['configuration'])-set(old['configuration']))
    same(meta['changed_configuration_fields'],changed,'Actual changed config names')
    same(meta['added_configuration_fields'],added,'Actual added config names')
    same(changed,['hidden'],'Only named baseline hidden config overridden')
    median=float(np.median(arrays['train_y_million'].astype(np.float64)/arrays['train_price_factor_million'].astype(np.float64)))
    same(meta['initial_training_median_ratio'],median,'Original unweighted fit ratio median')
    same(meta['initial_head_bias'],median+float(np.log(-np.expm1(-median))),'Original inverseSoftplus bias')
    same(c.sha(prefix+'model.pt'),meta['model_sha256'],'R111 model file SHA')
    saved=checkpoint_cpu(prefix+'model.pt')
    same(saved['payload']['configuration'],meta['configuration'],'Checkpoint actual config')
    same(saved['payload']['baseline_configuration'],meta['baseline_configuration'],'Checkpoint original config')
    same(saved['payload']['numeric_count'],144,'Checkpoint width')
    same(saved['payload']['cardinalities'],meta['cardinalities'],'Checkpoint categorical state')
    same(saved['parameter_count'],meta['parameter_count'],'Actual saved parameter count')
    same(saved['embedding_dimensions'],meta['embedding_dimensions'],'Actual embedding dimensions')
    predictions={}
    same(set(meta['prediction_artifacts']),{'predictions_ratio.npy','predictions_twd.npy'},'Both NN predictions required')
    for filename,record in meta['prediction_artifacts'].items():
        same(c.sha(prefix+filename),record['sha256'],'Saved NN prediction file SHA')
        value=np.load(c.safe(prefix+filename),allow_pickle=False)
        same(c.identity(value),{k:record[k] for k in ('dtype','shape','values_sha256')},'Saved NN array identity')
        c.need(value.dtype==np.float64 and value.shape==(c.SPLITS[stage][3],)
               and np.isfinite(value).all() and (value>0).all(),'NN saved prediction schema')
        predictions[filename]=value
    ratio=predictions['predictions_ratio.npy']
    c.exact(ratio,ratio.astype(np.float32).astype(np.float64),'Ratio remains original float32 output representation')
    c.exact(ratio*arrays['query_price_factor_million'].astype(np.float64)*1e6,
            predictions['predictions_twd.npy'],'Exactly one total-TWD inverse')
    c.exact(predictions['predictions_twd.npy'],c.load_array(st['layers']['native']),'NN native/layer identity')
    return {'model':c.record(prefix+'model.pt'),'CPU_checkpoint':saved,
            'ratio_predictions':c.record(prefix+'predictions_ratio.npy'),
            'TWD_predictions':c.record(prefix+'predictions_twd.npy'),'all_nine_input_array_identities_verified':True}


def check_unit(stage,st,meta,spec,ctx,weights):
    import pandas as pd
    from catboost import CatBoostRegressor
    import round112_ordered_unit as helper
    prefix=f'outputs/round112/{stage}/model/'
    state=c.read(prefix+'feature_state.json')
    cats=c.read('reports/feature_schema.json')['categorical_columns']
    same(len(cats),21,'Raw categorical schema must have21 fields')
    helper._check_state(state,ctx['fit_X'],cats);helper._check_state(state,ctx['query_X'],cats)
    same(meta['parameters'],helper.parameters(),'Actual R112 fixed Ordered recipe')
    same(meta['baseline_parameters'],helper.baseline.parameters(),'Actual original R79 baseline')
    same(meta['changed_parameter_names'],['boosting_type'],'Only Ordered override')
    same(meta['added_parameter_names'],[],'No CTR override')
    same(meta['source_sha256'],helper.source_hashes(),'R112 actual helper source')
    same((meta['fit_rows'],meta['query_rows']),c.SPLITS[stage][2:],'R112 fit/query rows')
    same(meta['new_native_fits'],1,'One actual native fit')
    for filename,key in (('model.cbm','model_sha256'),('feature_state.json','feature_state_sha256'),
                         ('raw.npy','raw_prediction_sha256'),('native.npy','prediction_sha256')):
        same(c.sha(prefix+filename),meta[key],'Saved unit artifact SHA '+filename)
    c.need(meta['status']=='completed' and meta['saved_model_reload_exact'] and meta['raw_and_TWD_reload_exact']
           and meta['reload_rows']==len(ctx['query_X']) and not meta['query_labels_available']
           and not meta['query_labels_used'] and not meta['fit_uses_eval_set'] and not meta['early_stopping_used']
           and not meta['date_weights_recomputed'] and not meta['date_weights_renormalized']
           and not meta['weights_normalized_to_mean_one'] and not meta['extra_price_cap_applied']
           and not meta['reference_or_trend_applied_here'] and not meta['spatial_applied_here'],'R112 scope/reload')
    y=c.load_array(spec['fit_labels']['fit_y'])
    same(meta['fit_target_values_sha256'],helper.original.values_sha256(y),'R112 exact fit y')
    same(meta['fit_transformed_target_values_sha256'],helper.original.values_sha256(helper.original.transform_target(y,ctx['fit_X'])),'R112 unit target')
    dates=helper.baseline._date_values(weights,ctx['fit_X'])
    same(meta['fit_date_weights_sha256'],c.values_sha(dates),'R112 date weights')
    same(meta['fit_area_weights_sha256'],helper.original.values_sha256(helper.original.sample_weights(ctx['fit_X'])),'R112 area weights')
    same(meta['fit_sample_weights_sha256'],helper.original.values_sha256(helper.sample_weights(ctx['fit_X'],weights)),'R112 area/date combined weights')
    tree=pd.read_pickle(c.safe(spec['tree_geo']['path']))
    G=tree.iloc[ctx['indices']['fit_indices']].reset_index(drop=True)
    Gq=tree.iloc[ctx['indices']['query_indices']].reset_index(drop=True)
    frames={'fit_X':ctx['fit_X'],'query_X':ctx['query_X'],'fit_geo':G,'query_geo':Gq,
            'fit_augmented':helper.original.augment_features(ctx['fit_X'],G),
            'query_augmented':helper.original.augment_features(ctx['query_X'],Gq)}
    same({k:helper.base.frame_sha(v) for k,v in frames.items()},meta['frame_sha256'],'R112 fit/query X/geo/augmented values')
    same(meta['feature_columns'],state['feature_columns'],'R112 ordered actual features')
    same(meta['categorical_columns'],state['categorical_columns'],'R112 augmented cats')
    same(len(meta['categorical_columns']),23,'Unit model augmented categories')
    model=CatBoostRegressor();model.load_model(str(c.safe(prefix+'model.cbm')))
    actual=helper.check_model(model,state)
    same(actual,meta['reloaded_native_diagnostics'],'Actual loaded CBM native diagnostics')
    same(helper.compare_reload_diagnostics(meta['fit_native_diagnostics'],actual),meta['reload_diagnostics_comparison'],'Native serialization evidence')
    raw=helper._raw_predict(model,state,ctx['query_X'],Gq,cats)
    c.exact(raw,np.load(c.safe(prefix+'raw.npy'),allow_pickle=False),'CPU saved CBM full-query raw replay')
    native=helper.original.inverse_transform(raw,ctx['query_X'])
    c.exact(native,np.load(c.safe(prefix+'native.npy'),allow_pickle=False),'CPU saved CBM full-query TWD replay')
    c.exact(native,c.load_array(st['layers']['native']),'R112 native/layer alignment')
    return {'model':c.record(prefix+'model.cbm'),'loaded_native_diagnostics':actual,
            'CPU_full_query_raw_and_TWD_exact':True,'GPU_work':False,'new_fits':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args();print(json.dumps(inspect_checkpoint(args.checkpoint),ensure_ascii=False,allow_nan=False))
