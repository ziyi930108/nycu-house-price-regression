"""R47 fixed embedding-MLP raw geography branch; isolated trainer, three fixed periods."""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round39_city_road_experiment as previous
import round6_geo as geo_encoder
import round47_nn_encode as encoder

ROOT=previous.ROOT
OUT=ROOT/'outputs/round47'
DOCUMENT=ROOT/'reports/round47_plan.md'
TORCH_PYTHON=ROOT/'.venv_torch/Scripts/python.exe'
TRAINER=ROOT/'src/round47_nn_train.py'
PRIOR_PLAN='outputs/round46/plan.json'
PRIOR_PLAN_SHA='ae866e6cf95b57a28495eda74c40d4326fe4a2fa9a1e00afbdb973e9c6e59c7f'
ENCODER_SHA='6bee3297f761d2cda0a0b9224aebc61e6748c9e56d8738d1fe344ea0abda4c47'
STAGES=('development','confirmation','auxiliary')
CANDIDATE='nn_raw_three_branch_median'
CONTROL,COMPARATOR='v9_control','v5_log_spatial'
BASE_FORMULA='.75*(.5*old_raw_TWD+.5*new_NN_raw_TWD)+.25*old_unit_LGB_TWD'
FORMULA='median(new_NN_raw_spatial,saved_R39_log_spatial,saved_R40_unit_spatial)'
EPOCH_SELECTION='fixed last epoch; no query labels or early stopping'
need,sha,rel,locked,write=previous.need,previous.sha,previous.rel,previous.locked,previous.write
merge,now,vector,exact,spatial=previous.merge,previous.now,previous.vector,previous.exact,previous.spatial
original=previous.prior.log_run.original


def sources():
    return merge(previous.sources(),{rel(p):sha(p) for p in (Path(__file__),Path(encoder.__file__),TRAINER)})


def protected_paths():
    return [rel(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))]+['runs/versions.json']


def freeze():
    need(sha(Path(encoder.__file__))==ENCODER_SHA and TORCH_PYTHON.is_file() and TRAINER.is_file(),'Encoder/trainer/runtime not ready')
    inputs={rel(DOCUMENT):sha(DOCUMENT),rel(TORCH_PYTHON):sha(TORCH_PYTHON)}
    old=previous.prior.small(inputs,ROOT/PRIOR_PLAN,PRIOR_PLAN_SHA)
    need(old['round']==46 and old['stage_order']==list(STAGES),'Prior alignment contract differs')
    need(all(old['source_sha256'].get(n)==h for n,h in previous.sources().items()),'Inherited loader/scorer/source differs')
    bindings=copy.deepcopy(old['bindings'])
    def bind(name):
        need(name in old['input_sha256'],'Unbound prior input: '+name)
        inputs.update(merge(inputs,{name:old['input_sha256'][name]}))
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',previous.KT):bind(name)
    for stage,b in bindings.items():
        need(stage in STAGES,'Unexpected stage')
        for role,name in b.items():
            if role!='manifest':bind(name)
        bind(rel((ROOT/b['state_summary']).with_suffix('.sha256')))
        r,v,c=[locked(ROOT/b[k]) for k in ('R39_result','v5_result','v9_result')]
        need(all(x['stage']==stage and x['status']=='completed' and x['fit_rows']==old['split_contract'][stage]['fit_rows']
                 and x['query_rows']==old['split_contract'][stage]['query_rows'] and x['dates']==old['splits'][stage] for x in (r,v,c)),
             'Inherited control windows differ')
    src=sources();previous.prior.c6.verify_hashes(merge(inputs,src))
    value={'round':47,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'comparator':COMPARATOR,
        'formula':FORMULA,'base_formula':BASE_FORMULA,'stage_order':list(STAGES),'bindings':bindings,
        'source_sha256':src,'input_sha256':inputs,'split_contract':old['split_contract'],'splits':old['splits'],
        'target_group_weights':old['target_group_weights'],'spatial_parameters':spatial.parameters(),
        'driver_runtime':previous.average.runtime(),'trainer_python':rel(TORCH_PYTHON),
        'feature_count':95,'original_categorical_count':23,'numeric_input_width':144,'categorical_input_width':23,
        'epochs':30,'epoch_selection':EPOCH_SELECTION,'target':'raw TWD / 1e6','sample_weight':'uniform',
        'all_three_periods_regardless_of_scores':True,'query_labels_in_trainer_input':False,
        'automatic_release_allowed':False,'new_acceptance_thresholds':None,'formal_adoption_decision':None,
        'protected_sha256':{n:sha(ROOT/n) for n in protected_paths()}}
    need(src==sources() and sha(DOCUMENT)==inputs[rel(DOCUMENT)],'Freeze source/document changed')
    write(OUT/'plan.json',value);print('R47_frozen',sha(OUT/'plan.json'),flush=True)


def prepare_npz(folder,X,Xq,geo,geoq,y,cats):
    tick=time.monotonic()
    X95,Xq95=geo_encoder.augment_features(X,geo),geo_encoder.augment_features(Xq,geoq)
    category_names=geo_encoder.categories(X,cats)
    state=encoder.fit_encoder(X95,category_names)
    write(folder/'encoder_state.json',state)
    restored=locked(folder/'encoder_state.json');need(restored==state,'Encoder JSON roundtrip differs')
    tn,tc=encoder.encode(restored,X95);qn,qc=encoder.encode(restored,Xq95)
    cards=np.asarray(state['cardinalities'],dtype='int64')
    for n,c,rows in ((tn,tc,len(X)),(qn,qc,len(Xq))):
        need(n.dtype==np.dtype('float32') and n.shape==(rows,144) and np.isfinite(n).all()
             and c.dtype==np.dtype('int64') and c.shape==(rows,23) and (c>=0).all()
             and cards.shape==(23,) and (cards>0).all() and (c<cards).all(),'Encoder arrays/schema invalid')
    target=np.asarray(y/1e6,dtype='float32')
    need(target.shape==(len(X),) and np.isfinite(target).all() and (target>0).all(),'Invalid fit-only million-TWD target')
    path=folder/'prepared.npz'
    with path.open('xb') as f:
        np.savez(f,train_num=tn,train_cat=tc,train_y_million=target,query_num=qn,query_cat=qc,cardinalities=cards)
    return path,cards.tolist(),time.monotonic()-tick


def train_subprocess(folder,data_path,fit_rows,query_rows,cards,plan):
    command=[str(TORCH_PYTHON),'-u','-B','-X','utf8',str(TRAINER),'--data',str(data_path),'--out',str(folder/'model')]
    started=now();tick=time.monotonic();log=folder/'trainer.log'
    write(folder/'fit_started.json',{'round':47,'started_at_utc':started,'command':command,
        'fit_rows':fit_rows,'query_rows':query_rows,'data_sha256':sha(data_path),'query_labels_available':False,
        'trainer_source_sha256':plan['source_sha256'][rel(TRAINER)],'automatic_retry':False})
    with log.open('x',encoding='utf8') as f:
        execution=subprocess.run(command,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=False)
    elapsed=time.monotonic()-tick
    record={'command':command,'started_at_utc':started,'completed_at_utc':now(),'elapsed_seconds':elapsed,
        'exit_code':execution.returncode,'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(folder/'trainer_execution.json',record)
    need(execution.returncode==0,'Isolated trainer failed; preserve original attempt/log')
    metadata=json.loads((folder/'model/training.json').read_text(encoding='utf8'))
    need(metadata['status']=='completed' and metadata['configuration']['epochs']==30
         and metadata['configuration']['epoch_selection']==EPOCH_SELECTION
         and len(metadata['epochs'])==30 and [x['epoch'] for x in metadata['epochs']]==list(range(1,31))
         and metadata['fit_rows']==fit_rows and metadata['query_rows']==query_rows
         and metadata['numeric_count']==144 and metadata['cardinalities']==cards
         and metadata['source_sha256']==plan['source_sha256'][rel(TRAINER)]
         and metadata['input_sha256']==sha(data_path) and metadata['query_labels_available'] is False
         and metadata['saved_model_reload_exact'] is True and metadata['reload_rows']==query_rows
         and metadata['prediction_units']=='TWD' and (folder/'model/model.pt').is_file(),
         'Trainer fixed recipe/isolation/reload contract differs')
    prices=vector(folder/'model/predictions_twd.npy',query_rows)
    need((prices>0).all(),'Softplus TWD must be positive; no extra inverse or clipping')
    return prices,metadata,record


def evaluate(stage,plan,Xall,yall):
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
    old,v5r,v9r=[locked(ROOT/b[k]) for k in ('R39_result','v5_result','v9_result')]
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
    raw,log,unit,v9,v5=[vector(ROOT/b[n],len(qi)) for n in ('raw_spatial','old_log_spatial','unit_spatial','v9','v5')]
    exact(np.median(np.column_stack([raw,log,unit]),axis=1),v9,'Original v9 median')
    score=lambda p:original.score(yall[qi],p,X,Xq,plan['splits'][stage][0],plan['target_group_weights'])
    metrics={CONTROL:score(v9),COMPARATOR:score(v5)}
    need(metrics[CONTROL]==v9r['metrics']['three_branch_median'] and metrics[COMPARATOR]==v5r['metrics']['log_spatial_combination'],'Full control metrics differ')
    cats=previous.prior.c6.read(ROOT/'reports/feature_schema.json')['categorical_columns']
    data_path,cards,prep_seconds=prepare_npz(folder,X,Xq,geo,geoq,y,cats)
    prices,training,execution=train_subprocess(folder,data_path,len(fi),len(qi),cards,plan)
    oldraw,oldunit=vector(ROOT/b['old_raw'],len(qi)),vector(ROOT/b['old_unit'],len(qi))
    base=.75*(.5*oldraw+.5*prices)+.25*oldunit
    p,diag,reference_report=original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state,Xq,newgeo,base,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool);exact(branch[fallback],p[fallback],'NN spatial fallback')
    candidate=np.median(np.column_stack([branch,log,unit]),axis=1)
    need(candidate.shape==(len(qi),) and np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid median TWD')
    metrics[CANDIDATE]=score(candidate)
    for name,a in (('base_nn',base),('reference_nn',p),('nn_raw_spatial',branch),('predictions',candidate)):
        previous.save_array(folder/(name+'.npy'),a)
    for name,value in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
        with (folder/(name+'.pkl')).open('xb') as f:pickle.dump(value,f,protocol=5)
    need(before=={k:spatial.frame_sha(v) for k,v in frames.items()},'Original frames mutated')
    result={'round':47,'stage':stage,'status':'completed','plan_sha256':sha(OUT/'plan.json'),
        'fit_rows':len(fi),'query_rows':len(qi),'dates':plan['splits'][stage],'metrics':metrics,
        'candidate_vs_v9':previous.prior.log_run.differences(metrics[CONTROL],metrics[CANDIDATE]),
        'candidate_vs_v5':previous.prior.log_run.differences(metrics[COMPARATOR],metrics[CANDIDATE]),
        'v9_control_bitwise_equal':True,'v9_and_v5_all_metrics_exact':True,'original_frame_sha256':before,
        'new_nn_fits':1,'new_tree_fits':0,'reference_predict_calls':1,'spatial_predict_calls':1,'spatial_state_builds':0,
        'query_labels_in_prepared_data_or_trainer':False,'full_query_reload_verified_by_trainer':True,
        'epochs_completed':30,'encoder_prepare_seconds':prep_seconds,'trainer_elapsed_seconds':execution['elapsed_seconds'],
        'elapsed_seconds':time.monotonic()-tick,'training_metadata_path':rel(folder/'model/training.json'),
        'reference_report':reference_report,'spatial_report':spatial_report,'formal_adoption_decision':None,
        'artifact_sha256':{rel(f):sha(f) for f in sorted(folder.rglob('*')) if f.is_file()}}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'candidate_vs_v9':result['candidate_vs_v9'],'candidate_vs_v5':result['candidate_vs_v5']}),flush=True)
    return result


def run():
    p=locked(OUT/'plan.json')
    need(p['round']==47 and p['source_sha256']==sources() and p['stage_order']==list(STAGES)
         and p['formula']==FORMULA and p['base_formula']==BASE_FORMULA and p['epochs']==30
         and p['epoch_selection']==EPOCH_SELECTION and p['driver_runtime']==previous.average.runtime()
         and p['spatial_parameters']==spatial.parameters(),'Frozen method/source changed')
    need(not (OUT/'run_claim.json').exists() and not (OUT/'results.json').exists()
         and not any((OUT/s).exists() for s in STAGES),'Prior attempt exists; preserve it')
    write(OUT/'run_claim.json',{'started_at_utc':now(),'plan_sha256':sha(OUT/'plan.json'),'automatic_retry':False})
    tick=time.monotonic();result={'round':47,'status':'incomplete','checked_stages':[],'stages':{},'stage_result_sha256':{},
        'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],'plan_sha256':sha(OUT/'plan.json'),
        'protected_before':p['protected_sha256'],'formal_adoption_decision':None,'new_acceptance_thresholds':None,
        'new_submission_generated':False,'registry_updated':False,'fresh_blind_validation':False,
        'label_scope':'Shared canonical loader deserializes train targets; only fi y exported to trainer NPZ, qi y only used by scorer.'}
    try:
        previous.prior.c6.verify_hashes(merge(p['source_sha256'],p['input_sha256'],p['protected_sha256']))
        Xall,yall=previous.load_training_rows(p)
        for stage in STAGES:
            item=evaluate(stage,p,Xall,yall);result['stages'][stage]=item;result['checked_stages'].append(stage)
            result['stage_result_sha256'][stage]=sha(OUT/stage/'results.json')
        previous.prior.c6.verify_hashes(merge(p['source_sha256'],p['protected_sha256']))
        result['status']='three_periods_completed_no_adoption_decision'
    except Exception:result.update(status='incomplete_technical_failure',traceback=traceback.format_exc());print(result['traceback'],flush=True)
    result.update(elapsed_seconds=time.monotonic()-tick,completed_at_utc=now(),protected_after={n:sha(ROOT/n) for n in p['protected_sha256']})
    write(OUT/'results.json',result);return 0 if result['status']=='three_periods_completed_no_adoption_decision' else 1


def execute(mode):
    if mode=='freeze':need(not OUT.exists(),'Existing R47 output');OUT.mkdir(parents=True,exist_ok=False)
    else:need(OUT.is_dir(),'Freeze required')
    started=now();tick=time.monotonic();code=1;failure=None;log=OUT/(mode+'.log')
    with log.open('x',encoding='utf8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            try:
                if mode=='freeze':freeze();code=0
                else:code=run()
            except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':47,'command':sys.argv,'started_at_utc':started,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-tick,
        'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(OUT/(mode+'_execution.json'),event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    raise SystemExit(execute(parser.parse_args().command))
