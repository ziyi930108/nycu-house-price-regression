"""One fixed R105 native average, using saved vectors and immutable same-cutoff R/S.

No training, native-model or NPZ load, raw price table, test input or scoring.
Freeze/preflight read metadata and file hashes only; true application is root-run.
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

PLAN='reports/round105_plan.md'
PLAN_SHA='8eed243d95b7c76c583b42e17b10d95aaaefc166245aaafd2c9af11dffb90643'
OUT=c.ROOT/'outputs/round105'
CHECK='reports/round105_driver_selfcheck.json'
METADATA_CHECK='reports/round105_driver_metadata_preflight.json'
OLD_EXEC='outputs/round102/execution.json'
LAYERS=('native','base','reference','spatial','median')
INDEX_KEYS=('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices')
FORMULA='median(S(R(.75*(.5*R54_area+.5*(.5*v22_native+.5*R102T_native))+.25*unit_LGB)),H,U)'


def source_hashes():return c.source_hashes(__file__)


def parameters():
    return {'round':105,'stage_order':list(c.STAGES),'candidate':'T','primary_control':'v22',
        'supplementary_endpoint':'C','C_role':'saved complete R102T endpoint',
        'native_formula':'.5*P+.5*Q','native_coefficients':[.5,.5],'native_units':'TWD','native_dtype':'float64',
        'base_formula':'.75*(.5*area+.5*native)+.25*lgb','formula':FORMULA,
        'new_fit_calls':0,'new_native_fits':0,'new_NN_fits':0,'new_native_tree_fits':0,'prior_or_R_S_build_calls':0,
        'same_cutoff_R_S_reused':True,'native_models_loaded':False,'native_model_load_calls':0,
        'R_S_states_loaded':True,'learned_R_S_states_loaded':True,
        'NPZ_opened':False,'query_labels_available':False,'test_read':False,'new_CSV':False,
        'geography_historically_asof':False,'automatic_retry':False}


def native_average(P,Q):
    for a in (P,Q):
        c.need(isinstance(a,np.ndarray) and a.dtype==np.float64 and a.ndim==1 and len(a)>0,'Float64 native vector required')
        c.need(np.isfinite(a).all() and (a>=0).all(),'Finite nonnegative native total required')
    c.need(P.shape==Q.shape,'Native row alignment/shape differs')
    result=.5*P+.5*Q
    c.need(np.isfinite(result).all(),'Native average overflowed')
    return result


def merge_records(target,values):
    for name,digest in values.items():
        c.need(name not in target or target[name]==digest,'Conflicting input identity: '+name)
        target[name]=digest


def check_readset(records):
    for name in records:
        p=Path(name)
        c.need(name!=c.RAW and not name.startswith(('data_cache/test','data_cache/valid'))
               and p.suffix.lower() in ('.json','.npy','.pkl') and '/fit_y' not in name
               and 'prepared' not in name and not p.name.startswith(('train_y','query_y')),
               'Forbidden native model/NPZ/price input: '+name)


def specification(stage):
    """Only documents and their identities; no arrays/pickle deserialization."""
    spec=composition.specification(stage);sm=c.read(spec['source_stage_seal']['path'])
    c.need(sm['stage']==stage and sm['status']=='prior_input_stage_predictions_sealed','Original R102 stage identity')
    endpoints={**sm['layers']['T'],'median':sm['predictions']['T']}
    c.need(set(endpoints)==set(LAYERS),'Complete R102T endpoint required')
    inputs=dict(spec['input_sha256'])
    for rec in endpoints.values():merge_records(inputs,{rec['path']:rec['sha256']})
    check_readset(inputs)
    return {'stage':stage,'composition':spec,'source_R102_stage':spec['source_stage_seal'],
            'R102T_endpoint':endpoints,'input_sha256':inputs}


def old_authority():
    c.need(c.sha(composition.ROOT_SEAL['path'])==composition.ROOT_SEAL['sha256'],'Original R102 root identity')
    old=c.read(composition.ROOT_SEAL['path']);ex=c.read(OLD_EXEC)
    c.need(ex['exit_code']==0 and ex['failure'] is None and old['all_query_reload_exact'],'Original run not complete')
    c.need(old['protected_before']==old['protected_after']==c.protected_hashes(),'Original protected identities differ')
    c.verify_hashes(old['source_sha256'])
    return old


def freeze_request(path):
    path=c.safe(path);c.need(not path.exists() and not path.with_suffix('.sha256').exists(),'Preserve existing freeze')
    c.need(c.sha(PLAN)==PLAN_SHA,'Plan changed');old_authority()
    checks={name:c.read(name) for name in (CHECK,METADATA_CHECK)}
    for name,check in checks.items():
        c.need(check['status']=='passed' and check['plan_sha256']==PLAN_SHA
               and check['source_hashes']==source_hashes(),'Required current preflight differs: '+name)
    inputs={composition.ROOT_SEAL['path']:composition.ROOT_SEAL['sha256'],OLD_EXEC:c.sha(OLD_EXEC)}
    stages={s:specification(s) for s in c.STAGES}
    for spec in stages.values():c.verify_hashes(spec['input_sha256']);merge_records(inputs,spec['input_sha256'])
    source=source_hashes()
    request={'schema_version':1,'status':'frozen_native_average_before_application',
        'created_at_utc':c.now(),'plan':{'path':PLAN,'sha256':PLAN_SHA},'plan_sha256':PLAN_SHA,
        'parameters':parameters(),'source_sha256':source,'frozen_source_and_plan':{**source,PLAN:PLAN_SHA},
        'stage_specifications':stages,'input_sha256':inputs,'protected_before':c.protected_hashes(),
        'self_checks':{name:c.record(name) for name in checks},'source_R102_seal':composition.ROOT_SEAL,
        'source_R102_execution':c.record(OLD_EXEC),'freeze_loaded_real_arrays_or_labels':False,
        'query_labels_available':False,'new_CSV':False}
    path.parent.mkdir(parents=True,exist_ok=True);c.write(path,request);return request


def check_launch(path):
    r=c.read(path)
    c.need(r['status']=='frozen_native_average_before_application' and r['parameters']==parameters(),'Fixed request differs')
    c.need(r['plan']=={'path':PLAN,'sha256':PLAN_SHA} and c.sha(PLAN)==PLAN_SHA,'Plan differs')
    c.need(r['source_sha256']==source_hashes() and r['frozen_source_and_plan']=={**r['source_sha256'],PLAN:PLAN_SHA},'Source freeze differs')
    c.verify_hashes(r['frozen_source_and_plan']);c.verify_hashes(r['input_sha256'])
    c.need(r['protected_before']==c.protected_hashes(),'Protected files changed')
    for rec in r['self_checks'].values():c.need(c.sha(rec['path'])==rec['sha256'],'Preflight receipt changed')
    old_authority();return r


def endpoint_exact(actual,saved,name):
    c.need(set(actual)==set(saved)==set(LAYERS),'Five endpoint layers required')
    for layer in LAYERS:c.exact(actual[layer],saved[layer],name+' '+layer+' endpoint replay differs')


def run_stage(stage,request_path):
    request=check_launch(request_path)
    c.need(stage in c.STAGES and OUT.is_dir() and c.read(OUT/'started.json')['request']==c.record(request_path),'Serial parent required')
    folder=OUT/stage;c.need(not folder.exists(),'Preserve existing stage');folder.mkdir()
    tick=time.monotonic();code=1;failure=None
    try:
        spec=request['stage_specifications'][stage];c.need(spec==specification(stage),'Stage specification changed')
        context=composition.load_context(spec['composition']);sm=context['source']
        # load_context already replayed all five v22 layers before returning.
        control={k:c.load_array(rec) for k,rec in spec['R102T_endpoint'].items()}
        actual,_,_=composition.apply(context,control['native']);endpoint_exact(actual,control,'R102T')
        composition.verify_unchanged(context)
        candidate=native_average(context['original']['native'],control['native'])
        chain=composition.save_chain(context,candidate,folder/'T_chain')
        c.exact(c.load_array(chain['layers']['native']),candidate,'Candidate native full reload')
        composition.verify_unchanged(context);c.verify_hashes(spec['input_sha256']);c.verify_hashes(request['source_sha256'])
        c.need(c.protected_hashes()==request['protected_before'],'Protected files changed')
        seal={'schema_version':1,'round':105,'status':'native_average_stage_predictions_sealed','stage':stage,
            'cutoff':sm['cutoff'],'query_end':sm['query_end'],'outer_fit_rows':len(context['fit_X']),'query_rows':len(context['query_X']),
            'plan_sha256':PLAN_SHA,'request':c.record(request_path),'source_sha256':request['source_sha256'],
            'source_R102_stage':spec['source_R102_stage'],'source_R102_seal':composition.ROOT_SEAL,
            'original_input_sha256':spec['input_sha256'],'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),
            'layers':{'v22':sm['layers']['v22'],'C':sm['layers']['T'],'T':chain['layers']},
            'predictions':{'v22':sm['predictions']['v22'],'C':sm['predictions']['T'],'T':chain['prediction']},
            'diagnostics':{'T':chain['diagnostics']},'fixed_branches':sm['fixed_branches'],'formula':FORMULA,
            'parameters':parameters(),'features':sm['features'],'canonical_mapping':sm['canonical_mapping'],'raw_train':sm['raw_train'],
            'raw_price_record_declared_only':True,'raw_price_pickle_opened':False,'query_labels_available':False,'query_scores_computed':False,
            'NPZ_opened':False,'native_models_loaded':False,'native_model_load_calls':0,
            'R_S_states_loaded':True,'learned_R_S_states_loaded':True,'test_read':False,
            'new_fit_calls':0,'new_native_fits':0,'new_NN_fits':0,'new_native_tree_fits':0,'prior_or_R_S_build_calls':0,
            'fit_seconds':0.,'all_query_reload_exact':True,'all_saved_vectors_reload_exact':True,
            'reload_scope':'saved NPY vectors and diagnostics, no native model reload or new fit',
            'original_v22_all_five_layers_replay_exact':True,'original_R102T_all_five_layers_replay_exact':True,
            'all_endpoints_replay_exact':True,'R_S_state_and_frames_unchanged':True,
            'elapsed_seconds':time.monotonic()-tick,'geography_historically_asof':False}
        for key in INDEX_KEYS:seal[key]=sm[key]
        seal['artifact_sha256']={c.rel(p):c.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        c.write(folder/'stage_seal.json',seal);code=0
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(folder/'execution.json',{'exit_code':code,'failure':failure,'stage':stage,
        'elapsed_seconds':time.monotonic()-tick,'query_scores_computed':False,'new_fit_calls':0,'automatic_retry':False})


def pipeline(request_path):
    request=check_launch(request_path);c.need(not OUT.exists(),'Preserve existing run');OUT.mkdir()
    tick=time.monotonic();code=1;failure=None
    try:
        c.write(OUT/'started.json',{'request':c.record(request_path),'created_at_utc':c.now(),'stage_order':list(c.STAGES)})
        stages={}
        for stage in c.STAGES:
            command=[sys.executable,'-u','-B','-X','utf8',str(Path(__file__).resolve()),'--stage',stage,'--request',str(c.safe(request_path))]
            with (OUT/(stage+'.log')).open('x',encoding='utf8') as log:
                p=subprocess.run(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            c.need(p.returncode==0,'Stage failed; no automatic retry: '+stage)
            ex=c.read(OUT/stage/'execution.json');sm=c.read(OUT/stage/'stage_seal.json')
            c.need(ex['exit_code']==0 and ex['failure'] is None and sm['all_endpoints_replay_exact'] and sm['all_query_reload_exact'],'Incomplete stage')
            stages[stage]={'stage_manifest':c.record(OUT/stage/'stage_seal.json'),'execution':c.record(OUT/stage/'execution.json'),'predictions':sm['predictions']}
        c.verify_hashes(request['source_sha256']);c.verify_hashes(request['input_sha256'])
        c.need(c.protected_hashes()==request['protected_before'],'Protected files changed')
        old=c.read(composition.ROOT_SEAL['path'])
        seal={'schema_version':1,'round':105,'status':'all_native_average_predictions_sealed','stage_order':list(c.STAGES),
            'stages':stages,'plan_sha256':PLAN_SHA,'parameters':parameters(),'request':c.record(request_path),'source_sha256':request['source_sha256'],
            'source_R102_seal':composition.ROOT_SEAL,'source_R102_execution':c.record(OLD_EXEC),
            'new_fit_calls':0,'new_native_fits':0,'new_NN_fits':0,'new_native_tree_fits':0,'prior_or_R_S_build_calls':0,'fit_seconds':0.,
            'all_query_reload_exact':True,'all_saved_vectors_reload_exact':True,'all_endpoints_replay_exact':True,
            'original_v22_all_five_layers_replay_exact':True,'original_R102T_all_five_layers_replay_exact':True,
            'R_S_state_and_frames_unchanged':True,'query_labels_available':False,'query_scores_computed':False,
            'raw_price_pickle_opened':False,'NPZ_opened':False,'native_models_loaded':False,'native_model_load_calls':0,
            'R_S_states_loaded':True,'learned_R_S_states_loaded':True,'test_read':False,
            'features':old['features'],'canonical_mapping':old['canonical_mapping'],'raw_train':old['raw_train'],'raw_price_record_declared_only':True,
            'original_input_sha256':request['input_sha256'],'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),
            'elapsed_seconds':time.monotonic()-tick,'geography_historically_asof':False,'new_CSV':False,'selected_submission':'v22','Private_score':None}
        c.write(OUT/'sealed_manifest.json',seal);code=0
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
        'source_sha256':c.sha(__file__),'query_scores_computed':False,'new_fit_calls':0,'automatic_retry':False})


def self_check():
    checks=[]
    def mark(name,value):c.need(value,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError,KeyError):checks.append(name)
        else:raise AssertionError('Did not reject '+name)
    P=np.array([0.,2.,14.,1.e9],dtype=np.float64);Q=np.array([4.,10.,2.,3.e9],dtype=np.float64)
    beforeP=P.copy();beforeQ=Q.copy();M=native_average(P,Q)
    mark('fixed_manual_average',np.array_equal(M,np.array([2.,6.,8.,2.e9])))
    mark('caller_inputs_not_mutated',P.tobytes()==beforeP.tobytes() and Q.tobytes()==beforeQ.tobytes())
    mark('same_endpoint_is_identity',native_average(P,P).tobytes()==P.tobytes())
    A=np.array([4.,8.,12.,2.e9]);L=np.array([8.,16.,24.,4.e9])
    mark('fixed_base_formula',np.array_equal(.75*(.5*A+.5*M)+.25*L,np.array([4.25,9.25,13.5,2.5e9])))
    reject('reject_float32',lambda:native_average(P.astype(np.float32),Q))
    reject('reject_misaligned_shape',lambda:native_average(P,Q[:-1]))
    reject('reject_matrix',lambda:native_average(P[:,None],Q[:,None]))
    reject('reject_empty',lambda:native_average(P[:0],Q[:0]))
    reject('reject_nonfinite',lambda:native_average(np.array([np.nan]),np.array([2.])))
    reject('reject_negative',lambda:native_average(np.array([-1.]),np.array([2.])))
    saved={k:np.array([1.,2.]) for k in LAYERS};endpoint_exact(saved,{k:v.copy() for k,v in saved.items()},'synthetic')
    mark('five_layers_exact',True)
    changed={k:v.copy() for k,v in saved.items()};changed['median'][0]=np.nextafter(1.,2.)
    reject('reject_one_bit_endpoint_change',lambda:endpoint_exact(changed,saved,'synthetic'))
    reject('reject_missing_endpoint_layer',lambda:endpoint_exact({k:v for k,v in saved.items() if k!='base'},saved,'synthetic'))
    for name in ('data_cache/train.pkl','x/prepared.npz','x/model.pt','x/fit_y.npy'):
        reject('forbid_'+name,lambda name=name:check_readset({name:'unused'}))
    mark('no_torch_import','torch' not in sys.modules)
    mark('zero_fit_contract',all(parameters()[k]==0 for k in ('new_fit_calls','new_native_fits','new_NN_fits','new_native_tree_fits','prior_or_R_S_build_calls')))
    return {'status':'passed','check_count':len(checks),'checks':checks,'plan_sha256':PLAN_SHA,
            'source_hashes':source_hashes(),'parameters':parameters(),'real_data_read':False,'real_composition_calls':0,'new_fit_score_calls':0}


def metadata_preflight():
    c.need(c.sha(PLAN)==PLAN_SHA,'Plan differs');old_authority()
    specs={s:specification(s) for s in c.STAGES}
    return {'status':'passed','plan_sha256':PLAN_SHA,'source_hashes':source_hashes(),'parameters':parameters(),
        'stages':{s:{'source_R102_stage':v['source_R102_stage'],'input_record_count':len(v['input_sha256']),
                     'endpoint_layer_names':list(v['R102T_endpoint'])} for s,v in specs.items()},
        'real_arrays_or_labels_read':False,'real_composition_calls':0,'new_fit_score_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze-request');g.add_argument('--run',action='store_true');g.add_argument('--stage',choices=c.STAGES)
    g.add_argument('--self-check',action='store_true');g.add_argument('--metadata-preflight',action='store_true')
    p.add_argument('--request');p.add_argument('--receipt');a=p.parse_args()
    if a.self_check or a.metadata_preflight:
        c.need(a.receipt is not None,'Receipt path required');t=time.monotonic()
        try:r=self_check() if a.self_check else metadata_preflight();r['elapsed_seconds']=time.monotonic()-t
        except Exception:
            c.write(a.receipt,{'status':'failed','failure':traceback.format_exc(),'source_hashes':source_hashes(),
                'plan_sha256':PLAN_SHA,'elapsed_seconds':time.monotonic()-t,'real_arrays_or_labels_read':False});raise
        c.write(a.receipt,r);print(json.dumps({'status':r['status'],'elapsed_seconds':r['elapsed_seconds']}))
    elif a.freeze_request:freeze_request(a.freeze_request)
    elif a.stage:run_stage(a.stage,a.request)
    else:pipeline(a.request)
