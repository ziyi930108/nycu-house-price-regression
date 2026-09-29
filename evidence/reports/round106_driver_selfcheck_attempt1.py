"""Six serial R106 NN fits on unchanged original R63 nine-array inputs.

Only training-mask dependence differs. No raw price pickle or query labels;
the parent/metadata preflight never deserialize NPZ. Root launches true work.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import round100_contract as c
import round104_composition as composition
import round47_nn_encode as encoder

PLAN='reports/round106_plan.md'
PLAN_SHA='2748f8f5cd76df02e99717a10a13f88186b1547d1196affb54bd603cb0351f51'
TRAINER='src/round106_joint_dropout_train.py'
TRAINER_SOURCES=(TRAINER,'src/round63_recency_weighted_prior_train.py','src/round52_nn_train.py','src/round47_nn_train.py')
OUT=c.ROOT/'outputs/round106'
R63='outputs/round63/results.json'
NAMES=['project_name','address_raw','building_and_unit','building_block']
INDEX_KEYS=('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices')
CHECKS={'trainer':'reports/round106_trainer_selfcheck.json','driver':'reports/round106_driver_selfcheck.json',
        'metadata':'reports/round106_driver_metadata_preflight.json'}
FORMULA='median(S(R(.75*(.5*R54_area+.5*R106_prior_NN)+.25*unit_LGB)),H,U)'


def source_hashes():return c.source_hashes(__file__,*TRAINER_SOURCES)


def parameters():
    return {'round':106,'stage_order':list(c.STAGES),'arm_order':['C','T'],'identity_columns':NAMES,
        'C_role':'new independent categorical dropout retraining','T_role':'project_name mask shared by four identity categories',
        'category_dropout':.1,'numeric_count':144,'categorical_count':23,'seed':42,'epochs':30,'batch_size':2048,
        'same_original_R63_NPZ_for_both_arms':True,'all_original_fit_rows':True,'new_NN_fits':6,'new_native_tree_fits':0,
        'prior_or_R_S_build_calls':0,'encoder_build_calls':0,'C_equals_saved_v22_required':False,
        'bitwise_retraining_determinism_claimed':False,'query_labels_available':False,'new_CSV':False,
        'geography_historically_asof':False,'formula':FORMULA}


def merge_records(dst,src):
    for name,h in src.items():c.need(name not in dst or dst[name]==h,'Conflicting input identity');dst[name]=h


def identity_contract(enc,encoder_record,prepared_record):
    columns=enc['categorical_columns']
    c.need(len(columns)==23 and len(set(columns))==23 and all(n in columns for n in NAMES),'Encoder categorical order differs')
    c.need(enc['numeric_output_width']==144 and enc['categorical_output_width']==23
           and not enc['query_used_for_state'] and not enc['target_used_for_state'],'Encoder fit-only scope differs')
    c.need(enc['state_content_sha256']==encoder._content_sha({k:v for k,v in enc.items() if k!='state_content_sha256'}),'Encoder content identity differs')
    return {'schema_version':1,'categorical_columns':columns,'identity_columns':NAMES,
            'identity_indices':[columns.index(n) for n in NAMES],'encoder_source':encoder_record,'prepared_source':prepared_record}


def check_readset(stage,records):
    for name in records:
        c.need(name!=c.RAW and not name.startswith(('data_cache/test','data_cache/valid')) and '/fit_y' not in name,'Forbidden raw price input')
        if name.endswith('.npz'):c.need(name==f'outputs/round63/{stage}/prepared.npz','Only own-stage original R63 NPZ allowed')


def specification(stage):
    """Read original metadata/encoder JSON, never true NPZ or numeric arrays."""
    comp=composition.specification(stage);inputs=dict(comp['input_sha256']);sm=c.read(comp['source_stage_seal']['path'])
    def doc(rec):
        c.need(c.sha(rec['path'])==rec['sha256'],'Metadata identity differs: '+rec['path'])
        merge_records(inputs,{rec['path']:rec['sha256']});return c.read(rec['path'])
    original=doc(sm['source_R100_stage_seal'])
    r63rec={'path':R63,'sha256':original['original_input_sha256'][R63]};history=doc(r63rec)
    c.need(history['checked_stages']==list(c.STAGES),'Original R63 stages incomplete')
    h=history['stages'][stage];art=h['artifact_sha256'];nf,nq=c.SPLITS[stage][2:]
    def historical(suffix):
        name=f'outputs/round63/{stage}/{suffix}';return {'path':name,'sha256':art[name]}
    prep_record=historical('preparation.json');prep=doc(prep_record)
    train_record=historical('model/training.json');training=doc(train_record)
    prepared=historical('prepared.npz');inputs[prepared['path']]=prepared['sha256']
    prior_record={'path':prep['prior_preparation_path'],'sha256':prep['prior_preparation_sha256']};prior=doc(prior_record)
    p48=doc({'path':composition.P48,'sha256':comp['input_sha256'][composition.P48]})
    ep=p48['bindings'][stage]['R47_encoder'];enc_record={'path':ep,'sha256':p48['input_sha256'][ep]};enc=doc(enc_record)
    contract=identity_contract(enc,enc_record,prepared)
    c.need(prep==h['preparation'] and prep['prepared_path']==prepared['path'] and prep['prepared_sha256']==prepared['sha256'],'Original preparation binding differs')
    c.need(training['status']=='completed' and training['source_sha256']==c.TRAINER_SHA and training['input_sha256']==prepared['sha256'],'Original R63 trainer identity differs')
    c.need((prep['fit_rows'],prep['query_rows'],prep['numeric_count'])==(nf,nq,144)
           and (training['fit_rows'],training['query_rows'],training['numeric_count'])==(nf,nq,144)
           and enc['fit_rows']==nf and training['cardinalities']==enc['cardinalities'],'Full original fit/encoder dimensions differ')
    c.need(prep['exact_NPZ_keys']==sorted(c.ARRAY_NAMES) and prep['weights_use_fit_X_only'] and not prep['weights_use_query_dates']
           and not prep['weights_use_targets'] and not prep['query_labels_used'] and not prep['float32_weights_renormalized'],'Original weighting boundary differs')
    c.need(prior['fit_labels_used_for_prior'] and not prior['query_labels_used'] and not prior['encoder_refit'],'Original prior provenance differs')
    c.need(prep['train_sample_weight_sha256']==training['train_sample_weight_sha256'],'Original weights binding differs')
    for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256'):
        c.need(training[key]==prior[key],'Original factor binding differs')
    c.need(training['initial_training_median_ratio']==prior['initial_training_median_ratio'] and training['initial_head_bias']==prior['initial_head_bias'],'Original initialization binding differs')
    records={name:historical(name+'.npy') for name in ('fit_indices','query_indices')}
    native=historical('model/predictions_twd.npy')
    for rec in [*records.values(),native]:inputs[rec['path']]=rec['sha256']
    check_readset(stage,inputs)
    return {'stage':stage,'composition':comp,'source_R102_stage':comp['source_stage_seal'],
        'source_R100_stage':sm['source_R100_stage_seal'],'source_R63_results':r63rec,'source_R63_prepared':prepared,
        'source_R63_preparation':prep_record,'source_R63_training':train_record,'source_R63_prior_preparation':prior_record,
        'source_encoder':enc_record,'source_R63_indices':records,'source_R63_native':native,
        'identity_contract':contract,'cardinalities':enc['cardinalities'],'input_sha256':inputs}


def freeze_request(path):
    path=c.safe(path);c.need(not path.exists() and not path.with_suffix('.sha256').exists(),'Preserve existing request')
    c.need(c.sha(PLAN)==PLAN_SHA,'Plan changed');receipts={k:c.read(p) for k,p in CHECKS.items()}
    for name,r in receipts.items():
        c.need(r['status']=='passed' and r['plan_sha256']==PLAN_SHA,'Required current check differs: '+name)
        c.verify_hashes(r['source_hashes'])
    c.need(receipts['driver']['source_hashes']==source_hashes()==receipts['metadata']['source_hashes'],'Driver check closure differs')
    c.need(receipts['trainer']['source_sha256']==c.sha(TRAINER),'Trainer check source differs')
    specs={s:specification(s) for s in c.STAGES};inputs={}
    for spec in specs.values():c.verify_hashes(spec['input_sha256']);merge_records(inputs,spec['input_sha256'])
    old=c.read(composition.ROOT_SEAL['path']);protected=c.protected_hashes()
    c.need(old['protected_before']==old['protected_after']==protected,'Original protected files changed');c.verify_hashes(old['source_sha256'])
    sources=source_hashes();request={'schema_version':1,'status':'frozen_joint_dropout_before_six_fits','created_at_utc':c.now(),
        'plan':{'path':PLAN,'sha256':PLAN_SHA},'plan_sha256':PLAN_SHA,'parameters':parameters(),
        'source_sha256':sources,'frozen_source_and_plan':{**sources,PLAN:PLAN_SHA},'input_sha256':inputs,
        'stage_specifications':specs,'protected_before':protected,'self_checks':{k:c.record(p) for k,p in CHECKS.items()},
        'trainer_configurations':receipts['trainer']['configurations'],'trainer_parameters':receipts['trainer']['parameters'],
        'freeze_loaded_real_arrays_or_labels':False,'query_labels_available':False,'new_CSV':False}
    path.parent.mkdir(parents=True,exist_ok=True);c.write(path,request);return request


def check_launch(path):
    r=c.read(path);c.need(r['status']=='frozen_joint_dropout_before_six_fits' and r['parameters']==parameters(),'Fixed request differs')
    c.need(r['plan']=={'path':PLAN,'sha256':PLAN_SHA} and c.sha(PLAN)==PLAN_SHA,'Plan differs')
    c.need(r['source_sha256']==source_hashes() and r['frozen_source_and_plan']=={**r['source_sha256'],PLAN:PLAN_SHA},'Frozen source differs')
    c.verify_hashes(r['frozen_source_and_plan']);c.need(r['protected_before']==c.protected_hashes(),'Protected files changed')
    for rec in r['self_checks'].values():c.need(c.sha(rec['path'])==rec['sha256'],'Check receipt changed')
    return r


def validate_arrays(arrays,nf,nq,cardinalities,expected):
    c.need(set(arrays)==set(c.ARRAY_NAMES),'Exactly nine original arrays required')
    card=arrays['cardinalities'];c.need(card.dtype==np.int64 and card.shape==(23,) and card.tolist()==cardinalities,'Cardinality/encoder order differs')
    for prefix,rows in (('train',nf),('query',nq)):
        num,cat=arrays[prefix+'_num'],arrays[prefix+'_cat']
        c.need(num.dtype==np.float32 and num.shape==(rows,144) and np.isfinite(num).all(),'Original numeric matrix differs')
        c.need(cat.dtype==np.int64 and cat.shape==(rows,23) and (cat>=0).all() and (cat<card).all(),'Original category matrix differs')
    for name,rows in (('train_y_million',nf),('train_price_factor_million',nf),('query_price_factor_million',nq),('train_sample_weight',nf)):
        a=arrays[name];c.need(a.dtype==np.float32 and a.shape==(rows,) and np.isfinite(a).all() and (a>0).all(),'Original positive float32 vector differs: '+name)
    identities={k:c.identity(v) for k,v in arrays.items()}
    c.need(set(expected)==set(c.ARRAY_NAMES),'Nine saved value identities required')
    c.need(all(identities[k]['values_sha256']==expected[k] for k in c.ARRAY_NAMES),'Original nine-array value binding differs')
    return identities


def validate_training(meta,stage_info,mode,request):
    c.need(meta['status']=='completed' and meta['mask_mode']==mode and meta['configuration']==request['trainer_configurations'][mode],'Trainer mode/configuration differs')
    c.need(meta['source_sha256']==request['source_sha256'][TRAINER] and meta['plan_sha256']==PLAN_SHA,'Trainer source/plan differs')
    c.need(meta['source_hashes']=={p:request['source_sha256'][p] for p in TRAINER_SOURCES},'Trainer exact source closure differs')
    c.need(meta['input_sha256']==stage_info['prepared']['sha256'] and meta['input_array_identities']==stage_info['all_input_arrays'],'Trainer shared inputs differ')
    c.need(meta['identity_contract']==stage_info['identity_contract'] and meta['identity_indices']==stage_info['identity_indices']
           and meta['identity_columns']==NAMES,'Trainer identity-mask contract differs')
    c.need((meta['fit_rows'],meta['query_rows'],meta['numeric_count'])==(stage_info['fit_rows'],stage_info['query_rows'],144),'Trainer full fit dimensions differ')
    c.need(meta['cardinalities']==stage_info['cardinalities'] and meta['initial_training_median_ratio']==stage_info['initial_training_median_ratio']
           and meta['initial_head_bias']==stage_info['initial_head_bias'],'Matched cardinality/initialization differs')
    c.need(meta['saved_model_reload_exact'] is True and meta['reload_rows']==stage_info['query_rows'],'Whole-query model reload required')
    c.need([x['epoch'] for x in meta['epochs']]==list(range(1,31)),'Fixed 30 epochs required')
    for key in ('query_labels_available','query_labels_used','sample_weights_recomputed_in_trainer','sample_weights_batch_renormalized','price_or_ratio_clipping'):
        c.need(meta[key] is False,'Trainer boundary changed: '+key)


def train_arm(folder,info,mode,request):
    folder.mkdir(parents=True,exist_ok=False);out=folder/'model';start=time.monotonic();code=1;failure=None;process_code=None
    command=[str(c.ROOT/'.venv_torch/Scripts/python.exe'),'-u','-B','-X','utf8',str(c.safe(TRAINER)),
        '--data',str(c.safe(info['prepared']['path'])),'--out',str(out),'--mask-mode',mode,
        '--identity-contract',str(c.safe(info['identity_contract']['path']))]
    try:
        with (folder/'train.log').open('x',encoding='utf8') as log:
            process=subprocess.run(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        process_code=process.returncode;c.need(process_code==0,'Trainer failed; preserve outputs and stop')
        meta=c.read(out/'training.json');validate_training(meta,info,mode,request)
        prediction=np.load(out/'predictions_twd.npy',allow_pickle=False)
        c.need(prediction.dtype==np.float64 and prediction.shape==(info['query_rows'],) and np.isfinite(prediction).all() and (prediction>0).all(),'Native output differs')
        arm={'mask_mode':mode,'prepared':info['prepared'],'identity_contract':info['identity_contract'],'run':c.record(out/'run.json'),
            'training':c.record(out/'training.json'),'model':c.record(out/'model.pt'),
            'native_prediction':{**c.record(out/'predictions_twd.npy'),**c.identity(prediction)},'all_input_arrays':meta['input_array_identities'],
            'fit_seconds':meta['fit_seconds'],'saved_model_reload_exact':True,'reload_rows':len(prediction)}
        code=0
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(folder/'execution.json',{'exit_code':code,'process_returncode':process_code,'failure':failure,
        'mask_mode':mode,'command':command,'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})
    return prediction,{**arm,'execution':c.record(folder/'execution.json')}


def run_stage(stage,request_path):
    request=check_launch(request_path);c.need(stage in c.STAGES and OUT.is_dir() and c.read(OUT/'started.json')['request']==c.record(request_path),'Serial parent required')
    folder=OUT/stage;c.need(not folder.exists(),'Preserve existing stage');folder.mkdir();tick=time.monotonic();code=1;failure=None
    try:
        spec=request['stage_specifications'][stage];c.need(spec==specification(stage),'Stage specification changed')
        check_readset(stage,spec['input_sha256']);c.verify_hashes(spec['input_sha256'])
        context=composition.load_context(spec['composition']);sm=context['source'];nf,nq=len(context['fit_X']),len(context['query_X'])
        for name,rec in spec['source_R63_indices'].items():
            a=np.load(c.safe(rec['path']),allow_pickle=False);c.exact(a,context['indices'][name],'R63 canonical '+name)
        c.exact(np.load(c.safe(spec['source_R63_native']['path']),allow_pickle=False),context['original']['native'],'Original prior NN identity')
        prior=c.read(spec['source_R63_prior_preparation']['path']);prep=c.read(spec['source_R63_preparation']['path']);old=c.read(spec['source_R63_training']['path'])
        expected={**prior['unchanged_six_array_sha256'],
            'train_price_factor_million':prior['train_price_factor_million_sha256'],'query_price_factor_million':prior['query_price_factor_million_sha256'],
            'train_sample_weight':prep['train_sample_weight_sha256']}
        with np.load(c.safe(spec['source_R63_prepared']['path']),allow_pickle=False) as z:
            c.need(set(z.files)==set(c.ARRAY_NAMES),'Validate nine NPZ keys before materializing')
            arrays={k:z[k] for k in c.ARRAY_NAMES}
        identities=validate_arrays(arrays,nf,nq,spec['cardinalities'],expected)
        c.write(folder/'identity_contract.json',spec['identity_contract']);contract_record=c.record(folder/'identity_contract.json')
        info={'fit_rows':nf,'query_rows':nq,'prepared':spec['source_R63_prepared'],'all_input_arrays':identities,
            'identity_contract':contract_record,'identity_indices':spec['identity_contract']['identity_indices'],
            'cardinalities':spec['cardinalities'],'initial_training_median_ratio':old['initial_training_median_ratio'],'initial_head_bias':old['initial_head_bias']}
        del arrays
        arms={};layers={'v22':sm['layers']['v22']};predictions={'v22':sm['predictions']['v22']};diagnostics={};drift={}
        for mode in ('C','T'):
            native,arm=train_arm(folder/mode,info,mode,request)
            chain=composition.save_chain(context,native,folder/(mode+'_chain'))
            arms[mode]=arm;layers[mode]=chain['layers'];predictions[mode]=chain['prediction'];diagnostics[mode]=chain['diagnostics']
            delta=native-context['original']['native']
            drift[mode]={'equal_rows':int(np.count_nonzero(native==context['original']['native'])),
                'rows':nq,'max_absolute_prediction_difference_TWD':float(np.max(np.abs(delta))),
                'mean_signed_prediction_difference_TWD':float(np.mean(delta)),'not_a_target_error_metric':True}
        c.need(arms['C']['prepared']==arms['T']['prepared'] and arms['C']['all_input_arrays']==arms['T']['all_input_arrays'],'C/T input mismatch')
        composition.verify_unchanged(context);c.verify_hashes(spec['input_sha256']);c.verify_hashes(request['source_sha256'])
        c.need(c.protected_hashes()==request['protected_before'],'Protected files changed')
        seal={'schema_version':1,'round':106,'status':'joint_dropout_stage_predictions_sealed','stage':stage,
            'cutoff':sm['cutoff'],'query_end':sm['query_end'],'outer_fit_rows':nf,'fit_rows':nf,'query_rows':nq,'numeric_count':144,'categorical_count':23,
            'plan_sha256':PLAN_SHA,'request':c.record(request_path),'source_sha256':request['source_sha256'],
            'source_R102_stage':spec['source_R102_stage'],'source_R102_seal':composition.ROOT_SEAL,
            **{k:spec[k] for k in ('source_R63_results','source_R63_prepared','source_R63_preparation','source_R63_training','source_R63_prior_preparation','source_encoder')},
            'prepared':info['prepared'],'all_input_arrays':identities,'identity_contract':contract_record,
            **{k:spec['identity_contract'][k] for k in ('categorical_columns','identity_columns','identity_indices')},
            'original_input_sha256':spec['input_sha256'],'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),
            'arms':arms,'layers':layers,'predictions':predictions,'diagnostics':diagnostics,'fixed_branches':sm['fixed_branches'],
            'formula':FORMULA,'features':sm['features'],'canonical_mapping':sm['canonical_mapping'],'raw_train':sm['raw_train'],
            'raw_price_record_declared_only':True,'raw_price_pickle_opened':False,'query_labels_available':False,'query_scores_computed':False,
            'other_stage_NPZ_opened':False,'new_NN_fits':2,'new_native_tree_fits':0,'prior_or_R_S_build_calls':0,'encoder_build_calls':0,
            'all_query_reload_exact':True,'original_v22_all_five_layers_replay_exact':True,'R_S_state_and_frames_unchanged':True,
            'same_prepared_for_C_and_T':True,'C_equals_saved_v22_required':False,'native_prediction_differences_from_saved_v22':drift,
            'fit_seconds':sum(a['fit_seconds'] for a in arms.values()),'elapsed_seconds':time.monotonic()-tick,'geography_historically_asof':False}
        for key in INDEX_KEYS:seal[key]=sm[key]
        seal['artifact_sha256']={c.rel(p):c.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        c.write(folder/'stage_seal.json',seal);code=0
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(folder/'execution.json',{'exit_code':code,'failure':failure,'stage':stage,'elapsed_seconds':time.monotonic()-tick,
        'query_scores_computed':False,'automatic_retry':False})


def pipeline(request_path):
    request=check_launch(request_path);c.need(not OUT.exists(),'Preserve existing run');OUT.mkdir();tick=time.monotonic();code=1;failure=None
    try:
        c.write(OUT/'started.json',{'request':c.record(request_path),'created_at_utc':c.now(),'stage_order':list(c.STAGES)})
        stages={}
        for stage in c.STAGES:
            command=[sys.executable,'-u','-B','-X','utf8',str(Path(__file__).resolve()),'--stage',stage,'--request',str(c.safe(request_path))]
            with (OUT/(stage+'.log')).open('x',encoding='utf8') as log:
                child=subprocess.run(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            c.need(child.returncode==0,'Stage failed; preserve artifacts, no retry: '+stage)
            ex=c.read(OUT/stage/'execution.json');sm=c.read(OUT/stage/'stage_seal.json')
            c.need(ex['exit_code']==0 and ex['failure'] is None and sm['all_query_reload_exact'],'Stage terminal/reload differs')
            stages[stage]={'stage_manifest':c.record(OUT/stage/'stage_seal.json'),'execution':c.record(OUT/stage/'execution.json'),
                'predictions':sm['predictions'],'fit_seconds':sm['fit_seconds']}
        c.verify_hashes(request['source_sha256']);c.verify_hashes(request['input_sha256']);c.need(c.protected_hashes()==request['protected_before'],'Protected files changed')
        old=c.read(composition.ROOT_SEAL['path'])
        seal={'schema_version':1,'round':106,'status':'all_joint_dropout_predictions_sealed','stage_order':list(c.STAGES),
            'stages':stages,'plan_sha256':PLAN_SHA,'parameters':parameters(),'request':c.record(request_path),'source_sha256':request['source_sha256'],
            'new_NN_fits':6,'new_native_tree_fits':0,'prior_or_R_S_build_calls':0,'encoder_build_calls':0,
            'all_query_reload_exact':True,'original_v22_all_five_layers_replay_exact':True,'R_S_state_and_frames_unchanged':True,
            'query_labels_available':False,'query_scores_computed':False,'raw_price_pickle_opened':False,'other_stage_NPZ_opened':False,
            'features':old['features'],'canonical_mapping':old['canonical_mapping'],'raw_train':old['raw_train'],'raw_price_record_declared_only':True,
            'original_input_sha256':request['input_sha256'],'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),
            'fit_seconds':sum(v['fit_seconds'] for v in stages.values()),'elapsed_seconds':time.monotonic()-tick,
            'geography_historically_asof':False,'new_CSV':False,'selected_submission':'v22','Private_score':None}
        c.write(OUT/'sealed_manifest.json',seal);code=0
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
        'source_sha256':c.sha(__file__),'query_scores_computed':False,'automatic_retry':False})


def self_check():
    checks=[]
    def mark(name,ok):c.need(ok,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):checks.append(name)
        else:raise AssertionError('Did not reject '+name)
    columns=['c'+str(i) for i in range(23)]
    for i,name in zip((10,13,11,20),NAMES):columns[i]=name
    e={'categorical_columns':columns,'numeric_output_width':144,'categorical_output_width':23,'query_used_for_state':False,'target_used_for_state':False}
    e['state_content_sha256']=encoder._content_sha(e)
    contract=identity_contract(e,{'path':'encoder.json','sha256':'x'},{'path':'prepared.npz','sha256':'y'})
    mark('indices_derived_from_frozen_order',contract['identity_indices']==[10,13,11,20])
    reject('reject_encoder_content_mutation',lambda:identity_contract({**e,'query_used_for_state':True},{},{}))
    nf,nq=3,2;a={'train_num':np.zeros((nf,144),np.float32),'query_num':np.zeros((nq,144),np.float32),
        'train_cat':np.zeros((nf,23),np.int64),'query_cat':np.zeros((nq,23),np.int64),'cardinalities':np.full(23,2,np.int64),
        'train_y_million':np.ones(nf,np.float32),'train_price_factor_million':np.ones(nf,np.float32),
        'query_price_factor_million':np.ones(nq,np.float32),'train_sample_weight':np.array([.5,1.,1.5],np.float32)}
    hashes={k:c.values_sha(v) for k,v in a.items()};before={k:c.identity(v) for k,v in a.items()}
    mark('exact_original_nine_array_contract',validate_arrays(a,nf,nq,[2]*23,hashes)==before)
    mark('caller_not_mutated',before=={k:c.identity(v) for k,v in a.items()})
    for name,value in [('train_num',np.zeros((nf,145),np.float32)),('train_cat',np.zeros((nf,23),np.float32)),
                       ('train_sample_weight',np.zeros(nf,np.float32)),('query_price_factor_million',np.full(nq,np.nan,np.float32))]:
        reject('reject_'+name,lambda name=name,value=value:validate_arrays({**a,name:value},nf,nq,[2]*23,hashes))
    reject('reject_label_payload_extra_key',lambda:validate_arrays({**a,'query_y':np.ones(nq)},nf,nq,[2]*23,hashes))
    reject('reject_changed_source_values',lambda:validate_arrays(a,nf,nq,[2]*23,{**hashes,'train_y_million':'changed'}))
    reject('reject_foreign_stage_NPZ',lambda:check_readset('development',{'outputs/round63/confirmation/prepared.npz':'x'}))
    reject('reject_raw_prices',lambda:check_readset('development',{c.RAW:'x'}))
    mark('driver_does_not_import_torch','torch' not in sys.modules)
    return {'status':'passed','check_count':len(checks),'checks':checks,'plan_sha256':PLAN_SHA,
        'source_hashes':source_hashes(),'parameters':parameters(),'real_arrays_or_labels_read':False,'new_fit_predict_score_calls':0}


def metadata_preflight():
    c.need(c.sha(PLAN)==PLAN_SHA,'Plan differs');specs={s:specification(s) for s in c.STAGES}
    return {'status':'passed','plan_sha256':PLAN_SHA,'source_hashes':source_hashes(),'parameters':parameters(),
        'stages':{s:{'identity_indices':v['identity_contract']['identity_indices'],'categorical_columns':v['identity_contract']['categorical_columns'],
                     'source_R63_prepared':v['source_R63_prepared'],'source_encoder':v['source_encoder'],'input_records':len(v['input_sha256'])} for s,v in specs.items()},
        'real_arrays_or_labels_read':False,'new_fit_predict_score_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze-request');g.add_argument('--run',action='store_true');g.add_argument('--stage',choices=c.STAGES)
    g.add_argument('--self-check',action='store_true');g.add_argument('--metadata-preflight',action='store_true')
    p.add_argument('--request');p.add_argument('--receipt');a=p.parse_args()
    if a.self_check or a.metadata_preflight:
        c.need(a.receipt is not None,'Receipt required');start=time.monotonic()
        try:r=self_check() if a.self_check else metadata_preflight();r['elapsed_seconds']=time.monotonic()-start
        except Exception:
            c.write(a.receipt,{'status':'failed','plan_sha256':PLAN_SHA,'source_hashes':source_hashes(),'failure':traceback.format_exc(),
                'elapsed_seconds':time.monotonic()-start,'real_arrays_or_labels_read':False});raise
        c.write(a.receipt,r);print(json.dumps({'status':r['status'],'elapsed_seconds':r['elapsed_seconds']}))
    elif a.freeze_request:freeze_request(a.freeze_request)
    elif a.stage:run_stage(a.stage,a.request)
    else:pipeline(a.request)
