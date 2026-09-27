"""R51 fixed seeds42/137/2026 native-TWD mean, then fixed half-v10 median; six NN fits."""
from __future__ import annotations
import argparse,contextlib,copy,json,pickle,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round48_nn_experiment as prior

ROOT=prior.ROOT;OUT=ROOT/'outputs/round51';DOCUMENT=ROOT/'reports/round51_plan.md'
TRAINER=ROOT/'src/round51_nn_seed_train.py'
TRAINER_SHA='5d944b9415797c1397240e0be1600fb1d4d8858035c3d596ad627dc5f967369c'
STAGES=prior.STAGES;SEEDS=(42,137,2026);NEW_SEEDS=(137,2026)
CANDIDATE='nn_seed_mean_half_v10'
FORMULA='.5*v10+.5*median(spatial(reference(base(mean_TWD(seed42,seed137,seed2026)))),recency_log_spatial,unit_spatial)'
MEAN_FORMULA='(seed42_TWD + seed137_TWD + seed2026_TWD) / 3.0, float64 in this order'
AUTHORITIES={'outputs/round48/plan.json':'ece3c43bae1d2f4c5f471d623a0a1082dcd8152dc3b0d20e91e7bc10c1a184af',
 'outputs/round48/results.json':'20d08c8442985054e9065ef2f280e3c9ff3febe054a059f53c17120bcb90f0b9',
 'outputs/round49/results.json':'0e80d21ef7ebea8489da605d733bf4efbcdd61f4eddf6e05d8558d3c836a087a',
 'outputs/round50/results.json':'1383385cb7f389d932a8524499e244db25453b410a2b819f123ff3c1f1ac8802'}
need,sha,rel,write,locked,merge,now=prior.need,prior.sha,prior.rel,prior.write,prior.locked,prior.merge,prior.now
previous=prior.previous;original=prior.original;spatial=prior.spatial;geo_encoder=prior.geo_encoder
vector,exact=prior.vector,prior.exact


def capture():
    inputs={}
    def bind(name,digest):
        need(name not in inputs or inputs[name]==digest,'Conflicting input: '+name)
        previous.prior.c6.verify_hashes({name:digest});inputs[name]=digest
    docs={}
    for name,h in AUTHORITIES.items():
        bind(name,h);docs[name]=locked(ROOT/name)
        receipt=rel((ROOT/name).with_suffix('.sha256'));inputs[receipt]=sha(ROOT/receipt)
    p48,r48,r49,r50=[docs[n] for n in AUTHORITIES]
    need(all(r['checked_stages']==list(STAGES) and r['status']=='three_periods_completed_no_adoption_decision' for r in (r48,r49,r50)),
         'All original comparison stages must be complete')
    need(r48['plan_sha256']==AUTHORITIES['outputs/round48/plan.json'] and p48['source_sha256']==r48['source_sha256']==prior.sources(),
         'Original R48 source/plan changed')
    need(sha(TRAINER)==TRAINER_SHA,'Seed wrapper changed')
    src=merge(p48['source_sha256'],{rel(Path(prior.__file__)):sha(Path(prior.__file__)),rel(Path(__file__)):sha(Path(__file__)),rel(TRAINER):TRAINER_SHA})
    previous.prior.c6.verify_hashes(src)
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',previous.KT):bind(name,p48['input_sha256'][name])
    bindings={}
    for stage in STAGES:
        a=r48['stages'][stage];spec=p48['split_contract'][stage];old=p48['bindings'][stage]
        need(all(r['stages'][stage]['fit_rows']==spec['fit_rows'] and r['stages'][stage]['query_rows']==spec['query_rows']
                 and r['stages'][stage]['dates']==p48['splits'][stage] for r in (r48,r49,r50)),'Comparison rows/dates differ')
        roles=('original_geo','query_geo','state','state_summary','reference','old_raw','old_unit','unit_spatial','v5','v10','R39_result','R46_result')
        b={'manifest':copy.deepcopy(old['manifest'])}
        for role in roles:bind(old[role],p48['input_sha256'][old[role]]);b[role]=old[role]
        receipt=rel((ROOT/b['state_summary']).with_suffix('.sha256'));bind(receipt,p48['input_sha256'][receipt])
        r46=locked(ROOT/b['R46_result']);recency=f'outputs/round46/{stage}/recency_log_spatial.npy'
        bind(recency,r46['artifact_sha256'][recency]);b['recency_log']=recency
        for role,suffix in [('fit_indices','fit_indices.npy'),('query_indices','query_indices.npy'),('prepared','prepared.npz'),
            ('encoder_state','encoder_state.json'),('seed42_predictions','model/predictions_twd.npy'),('seed42_metadata','model/training.json')]:
            name=f'outputs/round48/{stage}/{suffix}';bind(name,a['artifact_sha256'][name]);b[role]=name
        for role,number,r in [('v11',49,r49),('v12',50,r50)]:
            name=f'outputs/round{number}/{stage}/predictions.npy';bind(name,r['stages'][stage]['prediction_sha256']);b[role]=name
        bindings[stage]=b
    protected={rel(f):sha(f) for f in sorted((ROOT/'outputs').glob('submission_v*.csv'))};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    inputs[rel(DOCUMENT)]=sha(DOCUMENT);inputs[rel(prior.TORCH_PYTHON)]=sha(prior.TORCH_PYTHON)
    plan={'round':51,'formula':FORMULA,'native_mean_formula':MEAN_FORMULA,'seeds':list(SEEDS),'new_seeds':list(NEW_SEEDS),
          'source_sha256':src,'input_sha256':inputs,'bindings':bindings,'stage_order':list(STAGES),
          'split_contract':p48['split_contract'],'splits':p48['splits'],'target_group_weights':p48['target_group_weights'],
          'spatial_parameters':p48['spatial_parameters'],'protected_sha256':protected,'epochs':30,
          'query_labels_in_trainer':False,'seed42_retrained':False,'new_fit_count':6,'new_gate':None,
          'automatic_adoption':False,'new_CSV':False,'three_periods_regardless_of_scores':True}
    write(OUT/'plan.json',plan)
    return plan,r48,r49,r50


def train_seed(stage,seed,folder,b,plan,baseline):
    out=folder/'seeds'/str(seed);out.mkdir(parents=True,exist_ok=False)
    command=[str(prior.TORCH_PYTHON),'-u','-B','-X','utf8',str(TRAINER),'--seed',str(seed),
             '--data',str(ROOT/b['prepared']),'--out',str(out)]
    tick=time.monotonic();started=now();log=out/'trainer.log'
    write(out/'fit_started.json',{'round':51,'stage':stage,'seed':seed,'started_at_utc':started,'command':command,
          'prepared_sha256':plan['input_sha256'][b['prepared']],'query_labels_available':False,'automatic_retry':False})
    with log.open('x',encoding='utf8') as f:process=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
    elapsed=time.monotonic()-tick
    execution={'seed':seed,'command':command,'started_at_utc':started,'completed_at_utc':now(),'exit_code':process.returncode,
               'elapsed_seconds':elapsed,'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(out/'execution.json',execution);need(process.returncode==0,'Seed trainer failed; no automatic retry')
    m=json.loads((out/'training.json').read_text(encoding='utf8'))
    seed_sources={rel(p):sha(p) for p in (TRAINER,prior.TRAINER,prior.ARCHITECTURE)}
    need(m['status']=='completed' and m['configuration']=={**baseline['configuration'],'seed':seed}
         and m['requested_seed']==m['actual_seed']==seed and m['seed_setup_completed'] is True
         and m['seeded_configurations']=={'round48_nn_train':seed,'round47_nn_train':seed}
         and m['baseline_seed']==42 and m['changed_configuration_fields']==['seed']
         and m['seed_entrypoint_source_sha256']==TRAINER_SHA and m['seed_source_sha256']==seed_sources
         and m['source_sha256']==baseline['source_sha256'] and m['architecture_source_sha256']==baseline['architecture_source_sha256']
         and m['input_sha256']==baseline['input_sha256']==plan['input_sha256'][b['prepared']]
         and m['fit_rows']==baseline['fit_rows'] and m['query_rows']==baseline['query_rows']
         and m['numeric_count']==144 and m['cardinalities']==baseline['cardinalities']
         and m['query_labels_available'] is False and m['query_labels_used'] is False
         and m['extra_sample_weights'] is False and m['price_or_unit_clipping'] is False
         and all(m[k]==baseline[k] for k in ('output_units','prediction_units','train_area_factor_sha256','query_area_factor_sha256',
                     'area_formula','area_factor_formula','initial_training_median_unit_1e4_twd_per_m2','initial_head_bias'))
         and m['saved_model_reload_exact'] is True and m['reload_rows']==baseline['query_rows']
         and [e['epoch'] for e in m['epochs']]==list(range(1,31)) and (out/'model.pt').is_file(),'Seed-only trainer contract differs')
    values=vector(out/'predictions_twd.npy',m['query_rows']);need((values>0).all(),'Invalid seed total-price vector')
    print(json.dumps({'stage':stage,'seed':seed,'fit_seconds':m['fit_seconds'],'full_reload_rows':m['reload_rows']}),flush=True)
    return values,m,rel(out/'predictions_twd.npy'),rel(out/'training.json')

def evaluate(stage,plan,Xall,yall,r48,r49,r50):
    tick=time.monotonic();b=plan['bindings'][stage];spec=plan['split_contract'][stage]
    folder=OUT/stage;folder.mkdir(exist_ok=False)
    fi,qi=[np.load(ROOT/b[n+'_indices'],allow_pickle=False) for n in ('fit','query')]
    for n,a in (('fit',fi),('query',qi)):
        need(a.dtype==np.dtype('int64') and a.shape==(spec[n+'_rows'],) and (np.diff(a)>0).all()
             and a.min()>=0 and a.max()<len(Xall) and original.value_sha(a,'<i8')==spec[n+'_indices_content_sha256'],'Canonical indices differ')
        previous.save_array(folder/(n+'_indices.npy'),a)
    need(not np.intersect1d(fi,qi).size,'Fit/query overlap')
    X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
    y=yall[fi];need(np.isfinite(y).all() and (y>0).all(),'Invalid fit labels')
    allgeo=pd.read_pickle(ROOT/b['original_geo']);geo_encoder.check_geo(allgeo)
    need(len(allgeo)==len(Xall) and allgeo.index.equals(Xall.index),'Original canonical geo differs')
    geo,geoq=allgeo.iloc[fi].reset_index(drop=True),allgeo.iloc[qi].reset_index(drop=True);del allgeo
    newgeo=pd.read_pickle(ROOT/b['query_geo']);geo_encoder.check_geo(newgeo)
    need(newgeo.index.equals(Xq.index) and spatial.frame_sha(newgeo)==b['manifest']['features']['query']['frame_sha256'],'Spatial query geo differs')
    frames={'fit_X':X,'query_X':Xq,'original_fit_geo':geo,'original_query_geo':geoq}
    before={k:spatial.frame_sha(v) for k,v in frames.items()}
    old=locked(ROOT/b['R39_result'])
    need(all(old['input_frame_sha256'][k]==v for k,v in before.items()),'Original frame/row alignment differs')
    with (ROOT/b['state']).open('rb') as f:state=pickle.load(f)
    summary=locked(ROOT/b['state_summary'])
    with (ROOT/b['reference']).open('rb') as f:reference=pickle.load(f)
    need(state['state_summary']==summary and state['parameters']==plan['spatial_parameters']
         and summary['fit_X_sha256']==before['fit_X'] and summary['query_used_in_build'] is False
         and summary['fit_geo_sha256']==b['manifest']['features']['fit']['frame_sha256']
         and reference['fit_report']['fitting_rows']==len(X)
         and summary['existing_reference_residual_sha256']==original.value_sha(reference['reference_residual'],'<f8'),
         'Fit-only reference/spatial state differs')
    old_stage=r48['stages'][stage]
    need(before==old_stage['original_frame_sha256'],'R48 input row frames differ')
    v12,v11,v10,v5,log,unit=[vector(ROOT/b[n],len(qi)) for n in ('v12','v11','v10','v5','recency_log','unit_spatial')]
    exact(.5*v11+.5*v10,v12,'v12 fixed saved half average')
    score=lambda p:original.score(yall[qi],p,X,Xq,plan['splits'][stage][0],plan['target_group_weights'])
    metrics={n:score(v) for n,v in [('v12_control',v12),('v11_control',v11),('v10_control',v10),('v5_log_spatial',v5)]}
    need(metrics['v12_control']==r50['stages'][stage]['metrics']['R50']
         and metrics['v11_control']==r49['stages'][stage]['metrics']['R49']
         and metrics['v10_control']==r49['stages'][stage]['metrics']['v10']
         and metrics['v5_log_spatial']==r49['stages'][stage]['metrics']['v5'],'Saved control full metrics differ')
    baseline=json.loads((ROOT/b['seed42_metadata']).read_text(encoding='utf8'))
    need(baseline['status']=='completed' and baseline['configuration']['seed']==42
         and baseline['configuration']['epochs']==30 and baseline['input_sha256']==plan['input_sha256'][b['prepared']]
         and baseline['fit_rows']==len(fi) and baseline['query_rows']==len(qi)
         and baseline['saved_model_reload_exact'] is True and baseline['reload_rows']==len(qi)
         and baseline['source_sha256']==prior.TRAINER_SHA and baseline['architecture_source_sha256']==prior.ARCHITECTURE_SHA
         and baseline['prediction_units']=='TWD','Saved seed42 model/array contract differs')
    with np.load(ROOT/b['prepared'],allow_pickle=False) as saved:
        need(set(saved.files)=={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_area_factor','query_area_factor'},
             'Expected same R48 eight arrays with no query y')
        target=saved['train_y_million'];expected=np.asarray(y/1e6,dtype='float32')
        need(target.dtype==expected.dtype and target.shape==expected.shape and target.tobytes()==expected.tobytes(),'R48 fit target order differs')
    preds={42:vector(ROOT/b['seed42_predictions'],len(qi))}
    need((preds[42]>0).all(),'Saved seed42 TWD invalid')
    seed_predictions={'42':{'path':b['seed42_predictions'],'sha256':plan['input_sha256'][b['seed42_predictions']],'reused':True}}
    seed_training_metadata={'42':b['seed42_metadata']};training={}
    for seed in NEW_SEEDS:
        preds[seed],training[str(seed)],path,meta_path=train_seed(stage,seed,folder,b,plan,baseline)
        seed_predictions[str(seed)]={'path':path,'sha256':sha(ROOT/path),'reused':False};seed_training_metadata[str(seed)]=meta_path
    prices=(preds[42]+preds[137]+preds[2026])/3.0
    need(prices.dtype==np.float64 and np.isfinite(prices).all() and (prices>0).all(),'Invalid native mean TWD')
    oldraw,oldunit=vector(ROOT/b['old_raw'],len(qi)),vector(ROOT/b['old_unit'],len(qi))
    base=.75*(.5*oldraw+.5*prices)+.25*oldunit
    p,diag,reference_report=original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state,Xq,newgeo,base,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool);exact(branch[fallback],p[fallback],'Seed mean spatial fallback')
    median=np.median(np.column_stack([branch,log,unit]),axis=1)
    candidate=.5*v10+.5*median
    need(candidate.dtype==np.float64 and candidate.shape==(len(qi),) and np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid final TWD')
    metrics[CANDIDATE]=score(candidate)
    for name,a in [('nn_mean_twd',prices),('base_nn',base),('reference_nn',p),('nn_mean_spatial',branch),('median_recency_log',median),('predictions',candidate)]:
        previous.save_array(folder/(name+'.npy'),a)
    for name,value in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
        with (folder/(name+'.pkl')).open('xb') as f:pickle.dump(value,f,protocol=5)
    need(before=={k:spatial.frame_sha(v) for k,v in frames.items()},'Original frames mutated')
    result={'round':51,'stage':stage,'status':'completed','fit_rows':len(fi),'query_rows':len(qi),'dates':plan['splits'][stage],
       'metrics':metrics,'differences':{n:previous.prior.log_run.differences(metrics[n],metrics[CANDIDATE])
                        for n in ('v12_control','v11_control','v10_control','v5_log_spatial')},
       'seed_predictions':seed_predictions,'seed_training_metadata':seed_training_metadata,'native_mean_formula':MEAN_FORMULA,
       'new_NN_fits':2,'seed42_retrained':False,'new_tree_fits':0,'reference_predict_calls':1,'spatial_predict_calls':1,
       'spatial_state_builds':0,'all_four_controls_full_metrics_exact':True,'v12_control_bitwise_equal':True,
       'query_labels_in_trainer':False,'epochs_each_seed':30,'full_query_reload_each_new_seed':True,
       'fit_seconds_by_new_seed':{n:m['fit_seconds'] for n,m in training.items()},'original_frame_sha256':before,
       'reference_report':reference_report,'spatial_report':spatial_report,'elapsed_seconds':time.monotonic()-tick,
       'artifact_sha256':{rel(f):sha(f) for f in sorted(folder.rglob('*')) if f.is_file()}}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'differences':{n:{k:d[k]['relative_change_pct'] for k in ('overall','weighted')}
           for n,d in result['differences'].items()}}),flush=True)
    return result


def run():
    p,r48,r49,r50=capture();tick=time.monotonic()
    result={'round':51,'status':'incomplete','checked_stages':[],'stages':{},'stage_result_sha256':{},'formula':FORMULA,
       'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],'bindings':p['bindings'],'plan_sha256':sha(OUT/'plan.json'),
       'protected_before':p['protected_sha256'],'new_gate':None,'formal_adoption_decision':None,'new_submission_generated':False,
       'seed42_retrained':False,'fresh_blind_validation':False}
    try:
        Xall,yall=previous.load_training_rows(p)
        for stage in STAGES:
            result['stages'][stage]=evaluate(stage,p,Xall,yall,r48,r49,r50);result['checked_stages'].append(stage)
            result['stage_result_sha256'][stage]=sha(OUT/stage/'results.json')
        previous.prior.c6.verify_hashes(merge(p['source_sha256'],p['protected_sha256']))
        result['status']='three_periods_completed_no_adoption_decision'
    except Exception:result.update(status='incomplete_technical_failure',traceback=traceback.format_exc());print(result['traceback'],flush=True)
    result.update(elapsed_seconds=time.monotonic()-tick,completed_at_utc=now(),protected_after={n:sha(ROOT/n) for n in p['protected_sha256']})
    write(OUT/'results.json',result);return 0 if result['status']=='three_periods_completed_no_adoption_decision' else 1


def execute():
    need(not OUT.exists(),'Existing R51 output; no overwrite or retry');OUT.mkdir(parents=True,exist_ok=False)
    stamp=now();tick=time.monotonic();code=1;failure=None;log=OUT/'run.log'
    write(OUT/'run_claim.json',{'round':51,'started_at_utc':stamp,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    with log.open('x',encoding='utf8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            try:code=run()
            except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':51,'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-tick,
       'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(OUT/'execution.json',event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();raise SystemExit(execute())
