"""R104 six serial student fits after two earlier teachers successfully seal.

Freeze reads metadata and file identities only. Each stage prepares its own
matched 10-key NPZ; no raw price table, query labels, or foreign NPZ is opened.
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
import round104_student_preparation as student
import round104_composition as composition

PLAN = 'reports/round104_plan.md'
PLAN_SHA = '106fda0c56cc7c5660c6255bb572fd78c59ccac9c13d25604bbe592204e43778'
TRAINER = 'src/round104_base_aware_train.py'
OUT = c.ROOT/'outputs/round104'
TEACHER_SEAL = 'outputs/round104_teacher/models/sealed_manifest.json'
CHECKS = {'trainer': 'reports/round104_trainer_selfcheck.json',
          'student': 'reports/round104_student_selfcheck.json',
          'driver': 'reports/round104_driver_selfcheck.json'}
FORMULA = 'median(S(R(.75*(.5*R54_area+.5*R104_prior_NN)+.25*unit_LGB)),H,U)'


def source_hashes():
    return c.source_hashes(__file__, TRAINER)


def parameters():
    return {'round': 104, 'stage_order': list(c.STAGES), 'arm_order': ['C','T'],
            'C_role': 'same-subset gradient-scale-matched native objective',
            'T_role': 'same-subset fixed-base objective', 'same_prepared_for_both_arms': True,
            'student_new_NN_fits': 6, 'teacher_new_native_fits': 2, 'total_new_native_fits': 8,
            'teacher_origins': {s: list(v) for s,v in student.TEACHERS.items()},
            'seed': 42, 'epochs': 30, 'numeric_count': 145,
            'sample_weights_renormalized': False, 'geography_historically_asof': False,
            'price_label_temporal_OOT': True, 'query_labels_available': False,
            'prior_or_R_S_build_calls': 0, 'new_CSV': False}


def teacher_header(seal, execution):
    c.need(execution['exit_code'] == 0 and execution.get('failure') is None, 'Teachers must finish successfully before freeze')
    c.need(seal['status'] == 'pre_auxiliary_teachers_sealed' and seal['stage'] == 'pre_auxiliary'
           and seal['plan_sha256'] == PLAN_SHA
           and seal['cutoff'] == '2020-08-11' and seal['query_end'] == '2021-08-10'
           and seal['new_native_fits'] == 2 and seal['all_query_reload_exact'] is True
           and seal['query_labels_available'] is False, 'Teacher identity/time/reload differs')
    c.need({'area','lgb'} <= set(seal['native_predictions']), 'Two native teacher predictions required')


def merge_records(target, values):
    for path,digest in values.items():
        c.need(path not in target or target[path] == digest, 'Conflicting frozen input: '+path)
        target[path] = digest


def check_readset(stage, records):
    for name in records:
        c.need(name != c.RAW and not name.startswith(('data_cache/test','data_cache/valid'))
               and not name.endswith('/fit_y.npy'), 'Forbidden price input')
        if name.endswith('.npz'):
            c.need(name.startswith(f'outputs/round102/{stage}/T/'), 'Only own-stage R102T NPZ is permitted')


def freeze_request(path, teacher_path):
    """No arrays/NPZ/models are deserialized here; source/metadata and byte SHA only."""
    path=c.safe(path);c.need(not path.exists() and not path.with_suffix('.sha256').exists(),'Preserve existing freeze')
    c.need(c.sha(PLAN)==PLAN_SHA and c.rel(teacher_path)==TEACHER_SEAL,'Fixed plan/teacher path')
    teacher=c.read(teacher_path);teacher_exec=c.safe(teacher_path).with_name('execution.json')
    teacher_header(teacher,c.read(teacher_exec))
    c.verify_hashes(teacher['source_sha256'])
    c.need(c.sha(teacher['request']['path'])==teacher['request']['sha256'],'Teacher fit request changed')
    c.need(teacher['protected_before']==teacher['protected_after']==c.protected_hashes(),'Teacher protected identities differ')
    receipts={name:c.read(p) for name,p in CHECKS.items()}
    for name,receipt in receipts.items():
        c.need(receipt['status']=='passed' and receipt['plan_sha256']==PLAN_SHA,'Required self-check differs: '+name)
        sources=receipt['source_hashes'] if 'source_hashes' in receipt else receipt['source_sha256']
        c.need(isinstance(sources,dict),'Self-check source mapping missing: '+name);c.verify_hashes(sources)
    c.need(receipts['trainer']['source_sha256']==c.sha(TRAINER),'Trainer receipt is not current source')
    stages={}; all_inputs={TEACHER_SEAL:c.sha(teacher_path),c.rel(teacher_exec):c.sha(teacher_exec),
                          teacher['request']['path']:teacher['request']['sha256']}
    for stage in c.STAGES:
        spec=composition.specification(stage);sm=c.read(spec['source_stage_seal']['path'])
        sources={'stage_seal':spec['source_stage_seal'],'prepared':sm['arms']['T']['prepared'],
            **{k:sm[k] for k in ('fit_indices','query_indices','eligible_positions','features','canonical_mapping')}}
        teachers={origin:{'stage_seal':c.record(teacher_path) if origin=='pre_auxiliary'
                    else composition.specification(origin)['source_stage_seal']} for origin in student.TEACHERS[stage]}
        inputs=dict(spec['input_sha256']);merge_records(inputs,student.collect_input_records(stage,sources,teachers))
        check_readset(stage,inputs);c.verify_hashes(inputs);merge_records(all_inputs,inputs)
        stages[stage]={'source_records':sources,'teacher_records':teachers,'composition':spec,'input_sha256':inputs}
    sources=source_hashes();protected=c.protected_hashes()
    old=c.read(composition.ROOT_SEAL['path'])
    c.need(old['protected_before']==old['protected_after']==protected,'Original protected identities differ')
    c.verify_hashes(old['source_sha256'])
    request={'schema_version':1,'status':'frozen_after_teacher_seal_before_student_prepare_or_fit',
        'created_at_utc':c.now(),'plan':{'path':PLAN,'sha256':PLAN_SHA},'plan_sha256':PLAN_SHA,
        'parameters':parameters(),'source_sha256':sources,'frozen_source_and_plan':{**sources,PLAN:PLAN_SHA},
        'teacher_seal':c.record(teacher_path),'teacher_execution':c.record(teacher_exec),
        'self_checks':{k:c.record(v) for k,v in CHECKS.items()},
        'trainer_configurations':receipts['trainer']['configurations'],'trainer_parameters':receipts['trainer']['parameters'],
        'stage_specifications':stages,'input_sha256':all_inputs,'protected_before':protected,
        'approval_for_six_student_fits':True,'new_native_teacher_fits_already_completed':2,
        'freeze_loaded_real_arrays_or_labels':False,'query_labels_available':False,'new_CSV':False}
    path.parent.mkdir(parents=True,exist_ok=True);c.write(path,request)
    return request


def check_launch(path):
    request=c.read(path)
    c.need(request['status']=='frozen_after_teacher_seal_before_student_prepare_or_fit'
           and request['approval_for_six_student_fits'] is True,'Frozen six-student request required')
    c.need(request['plan']=={'path':PLAN,'sha256':PLAN_SHA} and c.sha(PLAN)==PLAN_SHA
           and request['parameters']==parameters(),'Fixed plan/parameters differ')
    c.need(request['source_sha256']==source_hashes(),'Frozen source closure differs')
    c.need(request['frozen_source_and_plan']=={**request['source_sha256'],PLAN:PLAN_SHA},'Incomplete source/plan freeze')
    c.verify_hashes(request['frozen_source_and_plan'])
    c.need(request['protected_before']==c.protected_hashes(),'Protected files changed')
    for record in [request['teacher_seal'],request['teacher_execution'],*request['self_checks'].values()]:
        c.need(c.sha(record['path'])==record['sha256'],'Frozen authority changed')
    teacher_header(c.read(request['teacher_seal']['path']),c.read(request['teacher_execution']['path']))
    return request


def validate_training(meta, preparation, objective, request):
    cfg=request['trainer_configurations'][objective];n=preparation['fit_rows'];q=preparation['query_rows']
    c.need(meta['status']=='completed' and meta['objective']==objective and meta['configuration']==cfg,'Trainer objective/configuration differs')
    c.need(meta['source_sha256']==request['source_sha256'][TRAINER] and meta['plan_sha256']==PLAN_SHA,'Trainer source/plan differs')
    c.need(meta['input_sha256']==preparation['prepared']['sha256'] and meta['input_array_identities']==preparation['all_input_arrays'],'Trainer input identity differs')
    c.need(meta['fit_rows']==n and meta['query_rows']==q and meta['numeric_count']==145,'Trainer row/feature dimensions differ')
    c.need(meta['initial_training_median_ratio']==preparation['initial_training_median_ratio']
           and meta['initial_head_bias']==preparation['initial_head_bias'] and meta['initial_bias_uses_offset'] is False,'Matched initialization differs')
    c.need(meta['saved_model_reload_exact'] is True and meta['reload_rows']==q and meta['reload_batch_size']==4096,'Full query model reload required')
    c.need([v['epoch'] for v in meta['epochs']]==list(range(1,31)) and all(v['objective']==objective for v in meta['epochs']),'Fixed complete training curve required')
    for key in ('query_labels_available','query_labels_used','query_offset_used','sample_weights_recomputed_in_trainer',
                'sample_weights_subset_renormalized','sample_weights_batch_renormalized','price_or_ratio_clipping',
                'residual_target_clipping','query_early_stopping','automatic_retry'):
        c.need(meta[key] is False,'Trainer boundary changed: '+key)


def train_arm(folder, preparation, objective, request):
    folder.mkdir(parents=True,exist_ok=False);modelout=folder/'model';start=time.monotonic();failure=None;code=1;process_code=None
    cmd=[str(c.ROOT/'.venv_torch/Scripts/python.exe'),'-u','-B','-X','utf8',str(c.safe(TRAINER)),
         '--data',str(c.safe(preparation['prepared']['path'])),'--out',str(modelout),'--objective',objective]
    try:
        with (folder/'train.log').open('x',encoding='utf8') as log:
            process=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        process_code=process.returncode;c.need(process_code==0,'Student trainer failed; preserve logs and stop')
        meta=c.read(modelout/'training.json');validate_training(meta,preparation,objective,request)
        c.need(c.sha(modelout/'model.pt')==meta['model_sha256'] and c.sha(modelout/'predictions_twd.npy')==meta['predictions_sha256'],'Saved trainer artifacts changed')
        prediction=np.load(modelout/'predictions_twd.npy',allow_pickle=False)
        c.need(prediction.dtype==np.float64 and prediction.shape==(preparation['query_rows'],)
               and np.isfinite(prediction).all() and (prediction>0).all(),'Invalid native total predictions')
        result={'objective':objective,'prepared':preparation['prepared'],'run':c.record(modelout/'run.json'),
                'training':c.record(modelout/'training.json'),'model':c.record(modelout/'model.pt'),
                'native_prediction':{**c.record(modelout/'predictions_twd.npy'),**c.identity(prediction)},
                'all_input_arrays':meta['input_array_identities'],'fit_seconds':meta['fit_seconds'],
                'saved_model_reload_exact':True,'reload_rows':len(prediction)}
        code=0
    except Exception:failure=traceback.format_exc();raise
    finally:
        c.write(folder/'execution.json',{'exit_code':code,'process_returncode':process_code,'failure':failure,
            'objective':objective,'command':cmd,'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})
    return prediction,{**result,'execution':c.record(folder/'execution.json')}


def stage_run(stage,request_path):
    request=check_launch(request_path)
    c.need(stage in c.STAGES and OUT.is_dir() and c.read(OUT/'started.json')['request']==c.record(request_path),'Serial parent required')
    folder=OUT/stage;c.need(not folder.exists(),'Preserve existing stage');folder.mkdir()
    start=time.monotonic();code=1;failure=None
    try:
        spec=request['stage_specifications'][stage];check_readset(stage,spec['input_sha256']);c.verify_hashes(spec['input_sha256'])
        context=composition.load_context(spec['composition']);sm=context['source']
        preparation=student.prepare(stage,spec['source_records'],spec['teacher_records'],folder/'preparation')
        c.need(all(spec['input_sha256'].get(p)==h for p,h in preparation['input_sha256'].items()),'Preparation read outside frozen stage inputs')
        c.need(preparation['same_prepared_for_C_and_T'] and preparation['query_four_arrays_bitwise_unchanged']
               and preparation['all_five_train_arrays_exact_source_subset'] and not preparation['sample_weights_renormalized'],'Matched preparation differs')
        arms={};layers={'v22':sm['layers']['v22']};predictions={'v22':sm['predictions']['v22']};diagnostics={}
        for objective in ('C','T'):
            native,arm=train_arm(folder/objective,preparation,objective,request)
            chain=composition.save_chain(context,native,folder/(objective+'_chain'))
            arms[objective]=arm;layers[objective]=chain['layers'];predictions[objective]=chain['prediction'];diagnostics[objective]=chain['diagnostics']
        c.need(arms['C']['prepared']==arms['T']['prepared'] and arms['C']['all_input_arrays']==arms['T']['all_input_arrays'],'C/T input mismatch')
        composition.verify_unchanged(context);c.verify_hashes(spec['input_sha256']);c.verify_hashes(request['source_sha256'])
        c.need(c.protected_hashes()==request['protected_before'],'Protected files changed')
        seal={'schema_version':1,'round':104,'status':'base_aware_stage_predictions_sealed','stage':stage,
            'cutoff':sm['cutoff'],'query_end':sm['query_end'],'outer_fit_rows':len(context['fit_X']),
            'student_fit_rows':preparation['fit_rows'],'query_rows':len(context['query_X']),'numeric_count':145,
            'plan_sha256':PLAN_SHA,'request':c.record(request_path),'source_sha256':request['source_sha256'],
            'source_R102_stage':spec['composition']['source_stage_seal'],'source_R102_seal':composition.ROOT_SEAL,
            'preparation':preparation['preparation'],'prepared':preparation['prepared'],
            'all_input_arrays':preparation['all_input_arrays'],
            'teacher_seal':request['teacher_seal'],'teacher_records':spec['teacher_records'],
            'original_input_sha256':spec['input_sha256'],'preparation_input_sha256':preparation['input_sha256'],
            'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),
            'arms':arms,'layers':layers,'predictions':predictions,'diagnostics':diagnostics,
            'fixed_branches':sm['fixed_branches'],'formula':FORMULA,'features':sm['features'],
            'canonical_mapping':sm['canonical_mapping'],'raw_train':sm['raw_train'],
            'raw_price_record_declared_only':True,'raw_price_pickle_opened':False,'query_labels_available':False,
            'other_stage_NPZ_opened':False,'query_scores_computed':False,'new_NN_fits':2,
            'new_native_tree_fits':0,'prior_or_R_S_build_calls':0,'all_query_reload_exact':True,
            'original_v22_all_five_layers_replay_exact':True,'R_S_state_and_frames_unchanged':True,
            'same_prepared_for_C_and_T':True,'fit_seconds':sum(v['fit_seconds'] for v in arms.values()),
            'elapsed_seconds':time.monotonic()-start,'geography_historically_asof':False}
        for key in ('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices'):
            seal[key]=sm[key]
        for key in ('student_fit_indices','source_npz_row_positions','student_outer_fit_positions','teacher_origin_codes','teacher_query_positions'):
            seal[key]=preparation[key]
        seal['artifact_sha256']={c.rel(p):c.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        c.write(folder/'stage_seal.json',seal);code=0
    except Exception:failure=traceback.format_exc();raise
    finally:
        c.write(folder/'execution.json',{'exit_code':code,'failure':failure,'stage':stage,
            'elapsed_seconds':time.monotonic()-start,'query_scores_computed':False,'automatic_retry':False})


def pipeline(request_path):
    request=check_launch(request_path);c.need(not OUT.exists(),'Preserve existing run');OUT.mkdir()
    start=time.monotonic();code=1;failure=None
    try:
        c.write(OUT/'started.json',{'request':c.record(request_path),'created_at_utc':c.now(),'stage_order':list(c.STAGES)})
        stages={}
        for stage in c.STAGES:
            command=[sys.executable,'-u','-B','-X','utf8',str(Path(__file__).resolve()),'--stage',stage,'--request',str(c.safe(request_path))]
            with (OUT/(stage+'.log')).open('x',encoding='utf8') as log:
                child=subprocess.run(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            c.need(child.returncode==0,'Stage failed; no retries: '+stage)
            ex=c.read(OUT/stage/'execution.json');sm=c.read(OUT/stage/'stage_seal.json')
            c.need(ex['exit_code']==0 and ex['failure'] is None and sm['all_query_reload_exact'],'Stage terminal/reload differs')
            stages[stage]={'stage_manifest':c.record(OUT/stage/'stage_seal.json'),'execution':c.record(OUT/stage/'execution.json'),
                'predictions':sm['predictions'],'fit_seconds':sm['fit_seconds']}
        c.verify_hashes(request['source_sha256']);c.verify_hashes(request['input_sha256'])
        c.need(c.protected_hashes()==request['protected_before'],'Protected files changed')
        original=c.read(composition.ROOT_SEAL['path'])
        seal={'schema_version':1,'round':104,'status':'all_base_aware_predictions_sealed','stage_order':list(c.STAGES),
            'stages':stages,'plan_sha256':PLAN_SHA,'parameters':parameters(),'request':c.record(request_path),
            'source_sha256':request['source_sha256'],'teacher_seal':request['teacher_seal'],'teacher_execution':request['teacher_execution'],
            'new_NN_fits':6,'new_native_tree_fits':0,'new_teacher_native_fits':2,'new_native_fits_total':8,
            'all_query_reload_exact':True,'query_labels_available':False,'query_scores_computed':False,
            'raw_price_pickle_opened':False,'other_stage_NPZ_opened':False,'prior_or_R_S_build_calls':0,
            'features':original['features'],'canonical_mapping':original['canonical_mapping'],'raw_train':original['raw_train'],
            'raw_price_record_declared_only':True,'original_input_sha256':request['input_sha256'],
            'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),
            'student_fit_seconds':sum(v['fit_seconds'] for v in stages.values()),'elapsed_seconds':time.monotonic()-start,
            'geography_historically_asof':False,'new_CSV':False,'selected_submission':'v22','Private_score':None}
        c.write(OUT/'sealed_manifest.json',seal);code=0
    except Exception:failure=traceback.format_exc();raise
    finally:
        c.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-start,
            'source_sha256':c.sha(__file__),'query_scores_computed':False,'automatic_retry':False})


def self_check():
    checks=[]
    def reject(name,fn):
        try:fn()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    valid={'status':'pre_auxiliary_teachers_sealed','stage':'pre_auxiliary','plan_sha256':PLAN_SHA,'cutoff':'2020-08-11','query_end':'2021-08-10',
           'new_native_fits':2,'all_query_reload_exact':True,'query_labels_available':False,'native_predictions':{'area':{},'lgb':{}}}
    teacher_header(valid,{'exit_code':0,'failure':None});checks.append('successful_teacher_header')
    reject('unfinished_teacher_rejected',lambda:teacher_header(valid,{'exit_code':1,'failure':'synthetic'}))
    reject('wrong_teacher_origin_rejected',lambda:teacher_header({**valid,'cutoff':'2021-08-11'},{'exit_code':0}))
    reject('wrong_teacher_plan_rejected',lambda:teacher_header({**valid,'plan_sha256':'wrong'},{'exit_code':0}))
    reject('query_labels_teacher_rejected',lambda:teacher_header({**valid,'query_labels_available':True},{'exit_code':0}))
    target={'a':'sha'};merge_records(target,{'a':'sha','b':'sha2'});checks.append('same_identity_merge')
    reject('conflicting_source_rejected',lambda:merge_records(target,{'a':'other'}))
    check_readset('development',{'outputs/round102/development/T/prepared.npz':'x'});checks.append('own_stage_NPZ_allowed')
    for name in (c.RAW,'outputs/round102/confirmation/T/prepared.npz','outputs/round104_teacher/fit_y.npy','data_cache/test.pkl'):
        reject('forbidden_readset_'+name,lambda name=name:check_readset('development',{name:'x'}))
    # A tiny metadata fixture verifies the trainer/driver API without fitting.
    prep={'fit_rows':4,'query_rows':3,'prepared':{'sha256':'i'},'all_input_arrays':{'synthetic':'identity'},
          'initial_training_median_ratio':2.,'initial_head_bias':1.854586542131141}
    req={'trainer_configurations':{'C':{'synthetic':'configuration'}},'source_sha256':{TRAINER:'s'}}
    meta={'status':'completed','objective':'C','configuration':req['trainer_configurations']['C'],
          'source_sha256':'s','plan_sha256':PLAN_SHA,'input_sha256':'i','input_array_identities':prep['all_input_arrays'],
          'fit_rows':4,'query_rows':3,'numeric_count':145,'initial_training_median_ratio':2.,'initial_head_bias':prep['initial_head_bias'],
          'initial_bias_uses_offset':False,'saved_model_reload_exact':True,'reload_rows':3,'reload_batch_size':4096,
          'epochs':[{'epoch':e,'objective':'C'} for e in range(1,31)]}
    for key in ('query_labels_available','query_labels_used','query_offset_used','sample_weights_recomputed_in_trainer',
                'sample_weights_subset_renormalized','sample_weights_batch_renormalized','price_or_ratio_clipping',
                'residual_target_clipping','query_early_stopping','automatic_retry'):meta[key]=False
    validate_training(meta,prep,'C',req);checks.append('trainer_metadata_contract')
    reject('unmatched_initialization_rejected',lambda:validate_training({**meta,'initial_bias_uses_offset':True},prep,'C',req))
    reject('training_input_mismatch_rejected',lambda:validate_training({**meta,'input_sha256':'wrong'},prep,'C',req))
    reject('wrong_objective_rejected',lambda:validate_training({**meta,'objective':'T'},prep,'C',req))
    return {'status':'passed','exit_code':0,'check_count':len(checks),'checks':checks,'plan_sha256':PLAN_SHA,
        'source_sha256':source_hashes(),'parameters':parameters(),'synthetic_only':True,'real_data_read':False,
        'new_fit_predict_score_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-request');p.add_argument('--teacher-seal',default=TEACHER_SEAL)
    p.add_argument('--run',action='store_true');p.add_argument('--stage',choices=c.STAGES);p.add_argument('--request')
    p.add_argument('--self-check',action='store_true');p.add_argument('--receipt',default=CHECKS['driver']);a=p.parse_args()
    if a.self_check:
        start=time.monotonic()
        try:r=self_check();r['elapsed_seconds']=time.monotonic()-start
        except Exception:
            c.write(a.receipt,{'status':'failed','exit_code':1,'failure':traceback.format_exc(),'source_sha256':source_hashes(),'real_data_read':False});raise
        c.write(a.receipt,r);print(json.dumps({'status':r['status'],'checks':r['check_count']}),flush=True)
    elif a.freeze_request:freeze_request(a.freeze_request,a.teacher_seal)
    elif a.stage and a.request:stage_run(a.stage,a.request)
    elif a.run and a.request:pipeline(a.request)
    else:p.error('Use --self-check, --freeze-request, --run --request, or --stage --request')
