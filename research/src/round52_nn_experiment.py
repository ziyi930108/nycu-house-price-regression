"""R52 group-OOF unit-price prior linked NN ratio; three fixed historical fits."""
from __future__ import annotations
import argparse,contextlib,copy,json,pickle,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round48_nn_experiment as prior
import round52_price_prior as price_prior

ROOT=prior.ROOT;OUT=ROOT/'outputs/round52';DOCUMENT=ROOT/'reports/round52_plan.md'
TRAINER=ROOT/'src/round52_nn_train.py'
TRAINER_SHA='0d27f93d9c48c7dc13cf0252a16eadd1a2aa8a7f53b0740021fc3d8b37c0740f'
PRIOR_SHA='3617c88689976aa27e0a6430c75bb005cbe0c283746d32eddbc1755417e0bb07'
STAGES=prior.STAGES
CANDIDATE='nn_price_prior_half_v10'
FORMULA='.5*v10+.5*median(spatial(reference(base(new_price_prior_NN_TWD))),recency_log_spatial,unit_spatial)'
PRICE_FACTOR_FORMULA='effective_area_A * group_prior_TWD_per_m2 / 1e6'
CONFIG_KEYS=('seed','epochs','batch_size','hidden','dropout','category_dropout','embedding_dimension','optimizer','learning_rate','weight_decay','scheduler','minimum_learning_rate','loss','positive_output','epoch_selection','float_precision','thread_count')
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
    need(sha(TRAINER)==TRAINER_SHA and sha(Path(price_prior.__file__))==PRIOR_SHA,'Final trainer/prior not ready or changed')
    src=merge(p48['source_sha256'],{rel(Path(prior.__file__)):sha(Path(prior.__file__)),rel(Path(__file__)):sha(Path(__file__)),rel(TRAINER):TRAINER_SHA,rel(Path(price_prior.__file__)):PRIOR_SHA,
        'src/v3_cluster_stability.py':sha(ROOT/'src/v3_cluster_stability.py')})
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
            ('encoder_state','encoder_state.json'),('baseline_metadata','model/training.json')]:
            name=f'outputs/round48/{stage}/{suffix}';bind(name,a['artifact_sha256'][name]);b[role]=name
        for role,number,r in [('v11',49,r49),('v12',50,r50)]:
            name=f'outputs/round{number}/{stage}/predictions.npy';bind(name,r['stages'][stage]['prediction_sha256']);b[role]=name
        bindings[stage]=b
    protected={rel(f):sha(f) for f in sorted((ROOT/'outputs').glob('submission_v*.csv'))};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    inputs[rel(DOCUMENT)]=sha(DOCUMENT);inputs[rel(prior.TORCH_PYTHON)]=sha(prior.TORCH_PYTHON)
    plan={'round':52,'formula':FORMULA,'price_factor_formula':PRICE_FACTOR_FORMULA,'seed':42,'group_prior_rules':price_prior.RULES,
          'source_sha256':src,'input_sha256':inputs,'bindings':bindings,'stage_order':list(STAGES),
          'split_contract':p48['split_contract'],'splits':p48['splits'],'target_group_weights':p48['target_group_weights'],
          'spatial_parameters':p48['spatial_parameters'],'protected_sha256':protected,'epochs':30,
          'query_labels_in_trainer':False,'prior_uses_only_fit_labels':True,'new_fit_count':3,'new_gate':None,
          'automatic_adoption':False,'new_CSV':False,'three_periods_regardless_of_scores':True}
    write(OUT/'plan.json',plan)
    return plan,r48,r49,r50


def prepare_data(folder,X,Xq,y,fi,b,plan):
    tick=time.monotonic()
    names=('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities')
    with np.load(ROOT/b['prepared'],allow_pickle=False) as saved:
        need(set(saved.files)==set(names)|{'train_area_factor','query_area_factor'},'Unexpected original eight arrays')
        arrays={n:saved[n] for n in names}
    target=arrays['train_y_million'];expected=np.asarray(y/1e6,dtype='float32')
    need(target.dtype==expected.dtype and target.shape==expected.shape and target.tobytes()==expected.tobytes(),'Original fit target order differs')
    # Canonical fi is passed explicitly for the existing missing-project row fallback.
    train_unit,state,folds,evidence=price_prior.fit_oof_prior(X,y,fi)
    need(np.asarray(train_unit).dtype==np.float64 and train_unit.shape==(len(X),)
         and np.isfinite(train_unit).all() and (train_unit>0).all()
         and folds.dtype==np.int64 and folds.shape==(len(X),) and set(np.unique(folds))==set(range(5)),
         'Invalid group-OOF prior/folds')
    write(folder/'price_prior_state.json',state);restored=locked(folder/'price_prior_state.json')
    need(restored==state,'Prior state JSON roundtrip differs')
    query_unit,query_counts=price_prior.predict_prior(restored,Xq,return_diagnostics=True)
    need(sum(query_counts.values())==len(Xq),'Query prior diagnostics row count differs')
    need(query_unit.dtype==np.float64 and query_unit.shape==(len(Xq),) and np.isfinite(query_unit).all() and (query_unit>0).all(),'Invalid query prior')
    factors={name:np.asarray(prior.area_helper.effective_area(frame)*unit/1e6,dtype='float32')
             for name,frame,unit in [('train_price_factor_million',X,train_unit),('query_price_factor_million',Xq,query_unit)]}
    for name,f in factors.items():need(np.isfinite(f).all() and (f>0).all(),'Price factor overflow/underflow; no clipping')
    for name,a in [('train_oof_unit_prior',train_unit),('query_unit_prior',query_unit),('prior_fold_ids',folds)]:
        previous.save_array(folder/(name+'.npy'),a)
    prepared=folder/'prepared.npz'
    with prepared.open('xb') as f:np.savez(f,**arrays,**factors)
    prep={'price_factor_formula':PRICE_FACTOR_FORMULA,'fit_labels_used_for_prior':True,'query_labels_used':False,
          'train_prior':'group-held-fold OOF; no own-fold labels','query_prior':'full fit labels only',
          'prior_evidence':evidence,'query_prior_level_counts':query_counts,'prior_state_sha256':sha(folder/'price_prior_state.json'),
          'original_six_R48_arrays_reused':True,'encoder_refit':False,'prior_prepare_seconds':time.monotonic()-tick,
          **{n+'_sha256':original.value_sha(a,'<f4') for n,a in factors.items()},
          'price_factor_million_summary':{n.replace('_price_factor_million',''):{'rows':len(a),'min':float(a.min()),
             'max':float(a.max()),'mean':float(a.mean(dtype=np.float64))} for n,a in factors.items()}}
    write(folder/'prior_preparation.json',prep)
    return prepared,prep


def train_model(stage,folder,data_path,baseline,prep):
    out=folder/'model';out.mkdir(exist_ok=False)
    command=[str(prior.TORCH_PYTHON),'-u','-B','-X','utf8',str(TRAINER),'--data',str(data_path),'--out',str(out)]
    tick=time.monotonic();started=now();log=out/'trainer.log'
    write(out/'fit_started.json',{'round':52,'stage':stage,'seed':42,'started_at_utc':started,'command':command,
          'prepared_sha256':sha(data_path),'query_labels_available':False,'automatic_retry':False})
    with log.open('x',encoding='utf8') as f:process=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
    execution={'command':command,'started_at_utc':started,'completed_at_utc':now(),'exit_code':process.returncode,
               'elapsed_seconds':time.monotonic()-tick,'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(out/'execution.json',execution);need(process.returncode==0,'Trainer failed; no automatic retry')
    m=json.loads((out/'training.json').read_text(encoding='utf8'))
    need(m['status']=='completed' and all(m['configuration'][k]==baseline['configuration'][k] for k in CONFIG_KEYS)
         and m['source_sha256']==TRAINER_SHA and m['architecture_source_sha256']==prior.ARCHITECTURE_SHA
         and m['input_sha256']==sha(data_path) and m['fit_rows']==baseline['fit_rows'] and m['query_rows']==baseline['query_rows']
         and m['numeric_count']==144 and m['cardinalities']==baseline['cardinalities']
         and m['query_labels_available'] is False and m['query_labels_used'] is False
         and m['extra_sample_weights'] is False and m['price_or_ratio_clipping'] is False
         and m['prediction_units']=='TWD' and m['output_units']=='dimensionless positive price ratio'
         and m['price_factor_formula']==PRICE_FACTOR_FORMULA and m['price_factor_uses_fit_labels'] is True
         and m['price_factor_provenance_verified_in_trainer'] is False
         and all(m[k]==prep[k] for k in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary'))
         and m['saved_model_reload_exact'] is True and m['reload_rows']==baseline['query_rows']
         and [e['epoch'] for e in m['epochs']]==list(range(1,31)) and (out/'model.pt').is_file(),'Ratio model/training contract differs')
    values=vector(out/'predictions_twd.npy',m['query_rows']);need((values>0).all(),'Invalid total-price output')
    print(json.dumps({'stage':stage,'fit_seconds':m['fit_seconds'],'full_reload_rows':m['reload_rows']}),flush=True)
    return values,m


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
    baseline=json.loads((ROOT/b['baseline_metadata']).read_text(encoding='utf8'))
    need(baseline['status']=='completed' and baseline['configuration']['seed']==42
         and baseline['configuration']['epochs']==30 and baseline['input_sha256']==plan['input_sha256'][b['prepared']]
         and baseline['fit_rows']==len(fi) and baseline['query_rows']==len(qi)
         and baseline['saved_model_reload_exact'] is True and baseline['reload_rows']==len(qi)
         and baseline['source_sha256']==prior.TRAINER_SHA and baseline['architecture_source_sha256']==prior.ARCHITECTURE_SHA
         and baseline['prediction_units']=='TWD','Saved seed42 model/array contract differs')
    data_path,prep=prepare_data(folder,X,Xq,y,fi,b,plan)
    prices,training=train_model(stage,folder,data_path,baseline,prep)
    oldraw,oldunit=vector(ROOT/b['old_raw'],len(qi)),vector(ROOT/b['old_unit'],len(qi))
    base=.75*(.5*oldraw+.5*prices)+.25*oldunit
    p,diag,reference_report=original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state,Xq,newgeo,base,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool);exact(branch[fallback],p[fallback],'Price-prior NN spatial fallback')
    median=np.median(np.column_stack([branch,log,unit]),axis=1)
    candidate=.5*v10+.5*median
    need(candidate.dtype==np.float64 and candidate.shape==(len(qi),) and np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid final TWD')
    metrics[CANDIDATE]=score(candidate)
    for name,a in [('nn_prior_twd',prices),('base_nn',base),('reference_nn',p),('nn_prior_spatial',branch),('median_recency_log',median),('predictions',candidate)]:
        previous.save_array(folder/(name+'.npy'),a)
    for name,value in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
        with (folder/(name+'.pkl')).open('xb') as f:pickle.dump(value,f,protocol=5)
    need(before=={k:spatial.frame_sha(v) for k,v in frames.items()},'Original frames mutated')
    result={'round':52,'stage':stage,'status':'completed','fit_rows':len(fi),'query_rows':len(qi),'dates':plan['splits'][stage],
       'metrics':metrics,'differences':{n:previous.prior.log_run.differences(metrics[n],metrics[CANDIDATE])
                        for n in ('v12_control','v11_control','v10_control','v5_log_spatial')},
       'prior_preparation':prep,'native_predictions_path':rel(folder/'model/predictions_twd.npy'),
       'training_metadata_path':rel(folder/'model/training.json'),
       'new_NN_fits':1,'seed':42,'new_tree_fits':0,'reference_predict_calls':1,'spatial_predict_calls':1,
       'spatial_state_builds':0,'all_four_controls_full_metrics_exact':True,'v12_control_bitwise_equal':True,
       'query_labels_in_trainer':False,'epochs':30,'full_query_reload_exact':True,
       'fit_seconds':training['fit_seconds'],'original_frame_sha256':before,
       'reference_report':reference_report,'spatial_report':spatial_report,'elapsed_seconds':time.monotonic()-tick,
       'artifact_sha256':{rel(f):sha(f) for f in sorted(folder.rglob('*')) if f.is_file()}}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'differences':{n:{k:d[k]['relative_change_pct'] for k in ('overall','weighted')}
           for n,d in result['differences'].items()}}),flush=True)
    return result


def run():
    p,r48,r49,r50=capture();tick=time.monotonic()
    result={'round':52,'status':'incomplete','checked_stages':[],'stages':{},'stage_result_sha256':{},'formula':FORMULA,
       'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],'bindings':p['bindings'],'plan_sha256':sha(OUT/'plan.json'),
       'protected_before':p['protected_sha256'],'new_gate':None,'formal_adoption_decision':None,'new_submission_generated':False,
       'fit_label_prior_cross_fitted':True,'fresh_blind_validation':False}
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
    need(not OUT.exists(),'Existing R52 output; no overwrite or retry');OUT.mkdir(parents=True,exist_ok=False)
    stamp=now();tick=time.monotonic();code=1;failure=None;log=OUT/'run.log'
    write(OUT/'run_claim.json',{'round':52,'started_at_utc':stamp,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    with log.open('x',encoding='utf8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            try:code=run()
            except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':52,'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-tick,
       'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(OUT/'execution.json',event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();raise SystemExit(execute())
