"""R48 fixed area-linked embedding-MLP geography branch; isolated trainer, three fixed periods."""
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
import round19_unit_geo as area_helper

ROOT=previous.ROOT
OUT=ROOT/'outputs/round48'
DOCUMENT=ROOT/'reports/round48_plan.md'
TORCH_PYTHON=ROOT/'.venv_torch/Scripts/python.exe'
TRAINER=ROOT/'src/round48_nn_train.py'
TRAINER_SHA='38fb91e01eabb2ed3163f1260e16c826ca55592951c22ce1107ad68c24b37082'
ARCHITECTURE=ROOT/'src/round47_nn_train.py'
ARCHITECTURE_SHA='68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
AREA_SHA='fac832e6c81871833a136d05759c7f0926fb749d2320a9c39b581e3986dc75cf'
R47_RESULT='outputs/round47/results.json'
R47_RESULT_SHA='06cc1ce2ac9df856d65a3c6df2a097891841e0929de11be3d01ebb9a784416ae'
R46_RESULT='outputs/round46/results.json'
R46_RESULT_SHA='338fd86a01399930d29055be07471823014f6c5ad8da0dd5ad1f7ed1fa96042e'
OUTPUT_UNITS='1e4 TWD per square metre of effective area A'
AREA_FORMULA='finite positive building_area else finite positive parking_area else 1'
AREA_FACTOR_FORMULA='A / 100'
PRIOR_PLAN='outputs/round47/plan.json'
PRIOR_PLAN_SHA='cfd122ad44cfe3e9cd3b520cb00f026b9a729e0e20b802628ca4274db0ecc1b6'
ENCODER_SHA='6bee3297f761d2cda0a0b9224aebc61e6748c9e56d8738d1fe344ea0abda4c47'
STAGES=('development','confirmation','auxiliary')
CANDIDATE='nn_area_three_branch_median'
R47_CONTROL='R47_raw_NN'
V10_CONTROL='v10_recency_log'
CONTROL,COMPARATOR='v9_control','v5_log_spatial'
BASE_FORMULA='.75*(.5*old_raw_TWD+.5*new_NN_area_TWD)+.25*old_unit_LGB_TWD'
FORMULA='median(new_NN_area_spatial,saved_R39_log_spatial,saved_R40_unit_spatial)'
EPOCH_SELECTION='fixed last epoch; no query labels or early stopping'
need,sha,rel,locked,write=previous.need,previous.sha,previous.rel,previous.locked,previous.write
merge,now,vector,exact,spatial=previous.merge,previous.now,previous.vector,previous.exact,previous.spatial
original=previous.prior.log_run.original


def sources():
    return merge(previous.sources(),{rel(p):sha(p) for p in (Path(__file__),Path(encoder.__file__),TRAINER,ARCHITECTURE,Path(area_helper.__file__))})


def protected_paths():
    return [rel(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))]+['runs/versions.json']


def freeze():
    need(sha(Path(encoder.__file__))==ENCODER_SHA and sha(TRAINER)==TRAINER_SHA
         and sha(ARCHITECTURE)==ARCHITECTURE_SHA and sha(Path(area_helper.__file__))==AREA_SHA
         and TORCH_PYTHON.is_file(),'Encoder/trainer/area/runtime not ready')
    inputs={rel(DOCUMENT):sha(DOCUMENT),rel(TORCH_PYTHON):sha(TORCH_PYTHON)}
    old=previous.prior.small(inputs,ROOT/PRIOR_PLAN,PRIOR_PLAN_SHA)
    need(old['round']==47 and old['stage_order']==list(STAGES),'Prior alignment contract differs')
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
    r47=previous.prior.small(inputs,ROOT/R47_RESULT,R47_RESULT_SHA)
    r46=previous.prior.small(inputs,ROOT/R46_RESULT,R46_RESULT_SHA)
    need(all(r['status']=='three_periods_completed_no_adoption_decision' and r['checked_stages']==list(STAGES) for r in (r47,r46)),
         'Prior R47/R46 results incomplete')
    for stage,b in bindings.items():
        spec=old['split_contract'][stage]
        for tag,result,round_number in (('R47',r47,47),('R46',r46,46)):
            name=f'outputs/round{round_number}/{stage}/results.json'
            item=previous.prior.small(inputs,ROOT/name,result['stage_result_sha256'][stage])
            need(item==result['stages'][stage] and item['stage']==stage and item['status']=='completed'
                 and item['fit_rows']==spec['fit_rows'] and item['query_rows']==spec['query_rows']
                 and item['dates']==old['splits'][stage],'Prior comparison windows differ')
            b[tag+'_result']=name
            if tag=='R47':
                choices={'R47_prepared':'prepared.npz','R47_encoder':'encoder_state.json',
                         'R47_training':'model/training.json','R47_prediction':'predictions.npy',
                         'R47_fit_indices':'fit_indices.npy','R47_query_indices':'query_indices.npy'}
            else:choices={'v10':'recency_log_three_branch_median.npy'}
            for role,suffix in choices.items():
                path=f'outputs/round{round_number}/{stage}/{suffix}'
                need(path in item['artifact_sha256'],'Prior artifact not bound: '+path)
                inputs.update(merge(inputs,{path:item['artifact_sha256'][path]}));b[role]=path
            if tag=='R47':
                receipt=rel((ROOT/b['R47_encoder']).with_suffix('.sha256'))
                inputs.update(merge(inputs,{receipt:item['artifact_sha256'][receipt]}))
    src=sources();previous.prior.c6.verify_hashes(merge(inputs,src))
    value={'round':48,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'comparator':COMPARATOR,
        'formula':FORMULA,'base_formula':BASE_FORMULA,'stage_order':list(STAGES),'bindings':bindings,
        'source_sha256':src,'input_sha256':inputs,'split_contract':old['split_contract'],'splits':old['splits'],
        'target_group_weights':old['target_group_weights'],'spatial_parameters':spatial.parameters(),
        'driver_runtime':previous.average.runtime(),'trainer_python':rel(TORCH_PYTHON),
        'feature_count':95,'original_categorical_count':23,'numeric_input_width':144,'categorical_input_width':23,
        'epochs':30,'epoch_selection':EPOCH_SELECTION,'target':'total TWD / 1e6','sample_weight':'uniform',
        'head_output_units':OUTPUT_UNITS,'area_formula':AREA_FORMULA,'area_factor_formula':AREA_FACTOR_FORMULA,
        'loss_formula':'abs(unit_output * train_area_factor - train_y_million).mean()',
        'encoder_policy':'reuse exact saved R47 fit-only state and six NPZ arrays; no refit or reencode',
        'extra_sample_weights':False,'price_or_unit_clipping':False,
        'comparators':[R47_CONTROL,CONTROL,V10_CONTROL,COMPARATOR],
        'all_three_periods_regardless_of_scores':True,'query_labels_in_trainer_input':False,
        'automatic_release_allowed':False,'new_acceptance_thresholds':None,'formal_adoption_decision':None,
        'protected_sha256':{n:sha(ROOT/n) for n in protected_paths()}}
    need(src==sources() and sha(DOCUMENT)==inputs[rel(DOCUMENT)],'Freeze source/document changed')
    write(OUT/'plan.json',value);print('R48_frozen',sha(OUT/'plan.json'),flush=True)


def prepare_npz(folder,X,Xq,geo,geoq,y,cats,b):
    tick=time.monotonic()
    state=locked(ROOT/b['R47_encoder'])
    old=locked(ROOT/b['R47_result'])
    X95=geo_encoder.augment_features(X,geo)
    need(state['fit_rows']==len(X) and state['feature_columns']==list(X95.columns)
         and state['input_dtypes']=={c:str(X95[c].dtype) for c in X95.columns}
         and state['categorical_columns']==geo_encoder.categories(X,cats)
         and state['target_used_for_state'] is False and state['query_used_for_state'] is False,
         'Saved fit-only encoder feature/state contract differs')
    for role,a in (('fit',np.load(ROOT/b['fit_indices'],allow_pickle=False)),('query',np.load(ROOT/b['query_indices'],allow_pickle=False))):
        prior_index=np.load(ROOT/b['R47_'+role+'_indices'],allow_pickle=False)
        need(a.dtype==prior_index.dtype and a.tobytes()==prior_index.tobytes(),'R47 prepared row indices differ')
    need(old['original_frame_sha256']=={k:spatial.frame_sha(v) for k,v in
        {'fit_X':X,'query_X':Xq,'original_fit_geo':geo,'original_query_geo':geoq}.items()},'Saved R47 feature rows differ')
    names=('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities')
    with np.load(ROOT/b['R47_prepared'],allow_pickle=False) as saved:
        need(set(saved.files)==set(names),'Unexpected saved R47 arrays')
        arrays={k:saved[k] for k in names}
    tn,tc,target,qn,qc,cards=[arrays[k] for k in names]
    for n,c,rows in ((tn,tc,len(X)),(qn,qc,len(Xq))):
        need(n.dtype==np.dtype('float32') and n.shape==(rows,144) and np.isfinite(n).all()
             and c.dtype==np.dtype('int64') and c.shape==(rows,23) and (c>=0).all()
             and cards.dtype==np.dtype('int64') and cards.shape==(23,) and (cards>0).all()
             and (c<cards).all(),'Saved encoder arrays/schema invalid')
    expected=np.asarray(y/1e6,dtype='float32')
    need(target.dtype==np.dtype('float32') and target.shape==expected.shape and target.tobytes()==expected.tobytes()
         and np.isfinite(target).all() and (target>0).all(),'Saved fit-only target alignment differs')
    need(cards.tolist()==state['cardinalities'],'Saved cardinalities differ')
    # R19 A is a pure-X output scale, not a sample-weight vector in this model.
    factors={name:np.asarray(area_helper.effective_area(frame)/100,dtype='float32')
             for name,frame in (('train_area_factor',X),('query_area_factor',Xq))}
    for name,a in factors.items():
        rows=len(X) if name.startswith('train') else len(Xq)
        need(a.shape==(rows,) and np.isfinite(a).all() and (a>0).all(),'Invalid float32 area output scale')
    evidence={name+'_sha256':original.value_sha(a,'<f4') for name,a in factors.items()}
    evidence.update(original_six_arrays_reused=True,encoder_refit=False,query_labels_available=False,
                    source_prepared_path=b['R47_prepared'],source_prepared_sha256=sha(ROOT/b['R47_prepared']),
                    source_encoder_path=b['R47_encoder'],source_encoder_sha256=sha(ROOT/b['R47_encoder']),
                    area_formula=AREA_FORMULA,area_factor_formula=AREA_FACTOR_FORMULA,
                    area_factor_summary={name.replace('_area_factor',''):{'rows':len(a),'min':float(a.min()),
                        'max':float(a.max()),'mean':float(a.mean(dtype=np.float64))} for name,a in factors.items()})
    with (folder/'encoder_state.json').open('xb') as f:f.write((ROOT/b['R47_encoder']).read_bytes())
    need(sha(folder/'encoder_state.json')==evidence['source_encoder_sha256'],'Encoder copy differs')
    write(folder/'area_preparation.json',evidence)
    path=folder/'prepared.npz'
    with path.open('xb') as f:np.savez(f,**arrays,**factors)
    return path,cards.tolist(),time.monotonic()-tick,evidence


def train_subprocess(folder,data_path,fit_rows,query_rows,cards,plan,b,area_evidence):
    command=[str(TORCH_PYTHON),'-u','-B','-X','utf8',str(TRAINER),'--data',str(data_path),'--out',str(folder/'model')]
    started=now();tick=time.monotonic();log=folder/'trainer.log'
    write(folder/'fit_started.json',{'round':48,'started_at_utc':started,'command':command,
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
    old_config=json.loads((ROOT/b['R47_training']).read_text(encoding='utf8'))['configuration']
    need(all(metadata['configuration'].get(k)==v for k,v in old_config.items())
         and metadata['architecture_source_sha256']==ARCHITECTURE_SHA
         and metadata['output_units']==OUTPUT_UNITS and metadata['area_formula']==AREA_FORMULA
         and metadata['area_factor_formula']==AREA_FACTOR_FORMULA
         and metadata['query_labels_used'] is False and metadata['extra_sample_weights'] is False
         and metadata['price_or_unit_clipping'] is False
         and all(metadata[k]==area_evidence[k] for k in ('train_area_factor_sha256','query_area_factor_sha256','area_factor_summary')),
         'Area-linked output/fixed original architecture contract differs')
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
    old,v5r,v9r,r47r,r46r=[locked(ROOT/b[k]) for k in ('R39_result','v5_result','v9_result','R47_result','R46_result')]
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
    r47p,v10=[vector(ROOT/b[k],len(qi)) for k in ('R47_prediction','v10')]
    metrics={CONTROL:score(v9),COMPARATOR:score(v5),R47_CONTROL:score(r47p),V10_CONTROL:score(v10)}
    need(metrics[CONTROL]==v9r['metrics']['three_branch_median'] and metrics[COMPARATOR]==v5r['metrics']['log_spatial_combination'],'Full control metrics differ')
    need(metrics[R47_CONTROL]==r47r['metrics']['nn_raw_three_branch_median']
         and metrics[V10_CONTROL]==r46r['metrics']['recency_log_three_branch_median'],'R47/v10 full metrics differ')
    cats=previous.prior.c6.read(ROOT/'reports/feature_schema.json')['categorical_columns']
    data_path,cards,prep_seconds,area_evidence=prepare_npz(folder,X,Xq,geo,geoq,y,cats,b)
    prices,training,execution=train_subprocess(folder,data_path,len(fi),len(qi),cards,plan,b,area_evidence)
    oldraw,oldunit=vector(ROOT/b['old_raw'],len(qi)),vector(ROOT/b['old_unit'],len(qi))
    base=.75*(.5*oldraw+.5*prices)+.25*oldunit
    p,diag,reference_report=original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state,Xq,newgeo,base,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool);exact(branch[fallback],p[fallback],'NN spatial fallback')
    candidate=np.median(np.column_stack([branch,log,unit]),axis=1)
    need(candidate.shape==(len(qi),) and np.isfinite(candidate).all() and (candidate>=0).all(),'Invalid median TWD')
    metrics[CANDIDATE]=score(candidate)
    for name,a in (('base_nn',base),('reference_nn',p),('nn_area_spatial',branch),('predictions',candidate)):
        previous.save_array(folder/(name+'.npy'),a)
    for name,value in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
        with (folder/(name+'.pkl')).open('xb') as f:pickle.dump(value,f,protocol=5)
    need(before=={k:spatial.frame_sha(v) for k,v in frames.items()},'Original frames mutated')
    result={'round':48,'stage':stage,'status':'completed','plan_sha256':sha(OUT/'plan.json'),
        'fit_rows':len(fi),'query_rows':len(qi),'dates':plan['splits'][stage],'metrics':metrics,
        'candidate_vs_v9':previous.prior.log_run.differences(metrics[CONTROL],metrics[CANDIDATE]),
        'candidate_vs_v5':previous.prior.log_run.differences(metrics[COMPARATOR],metrics[CANDIDATE]),
        'candidate_vs_R47':previous.prior.log_run.differences(metrics[R47_CONTROL],metrics[CANDIDATE]),
        'candidate_vs_v10':previous.prior.log_run.differences(metrics[V10_CONTROL],metrics[CANDIDATE]),
        'area_preparation':area_evidence,'R47_v9_v10_v5_all_metrics_exact':True,
        'v9_control_bitwise_equal':True,'v9_and_v5_all_metrics_exact':True,'original_frame_sha256':before,
        'new_nn_fits':1,'new_tree_fits':0,'reference_predict_calls':1,'spatial_predict_calls':1,'spatial_state_builds':0,
        'query_labels_in_prepared_data_or_trainer':False,'full_query_reload_verified_by_trainer':True,
        'epochs_completed':30,'encoder_prepare_seconds':prep_seconds,'trainer_elapsed_seconds':execution['elapsed_seconds'],
        'elapsed_seconds':time.monotonic()-tick,'training_metadata_path':rel(folder/'model/training.json'),
        'reference_report':reference_report,'spatial_report':spatial_report,'formal_adoption_decision':None,
        'artifact_sha256':{rel(f):sha(f) for f in sorted(folder.rglob('*')) if f.is_file()}}
    write(folder/'results.json',result)
    print(json.dumps({'stage':stage,'candidate_vs_v10':result['candidate_vs_v10'],'candidate_vs_R47':result['candidate_vs_R47'],'candidate_vs_v9':result['candidate_vs_v9'],'candidate_vs_v5':result['candidate_vs_v5']}),flush=True)
    return result


def run():
    p=locked(OUT/'plan.json')
    need(p['round']==48 and p['source_sha256']==sources() and p['stage_order']==list(STAGES)
         and p['formula']==FORMULA and p['base_formula']==BASE_FORMULA and p['epochs']==30
         and p['epoch_selection']==EPOCH_SELECTION and p['driver_runtime']==previous.average.runtime()
         and p['area_formula']==AREA_FORMULA and p['area_factor_formula']==AREA_FACTOR_FORMULA
         and p['head_output_units']==OUTPUT_UNITS and p['extra_sample_weights'] is False
         and p['price_or_unit_clipping'] is False
         and p['spatial_parameters']==spatial.parameters(),'Frozen method/source changed')
    need(not (OUT/'run_claim.json').exists() and not (OUT/'results.json').exists()
         and not any((OUT/s).exists() for s in STAGES),'Prior attempt exists; preserve it')
    write(OUT/'run_claim.json',{'started_at_utc':now(),'plan_sha256':sha(OUT/'plan.json'),'automatic_retry':False})
    tick=time.monotonic();result={'round':48,'status':'incomplete','checked_stages':[],'stages':{},'stage_result_sha256':{},
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
    if mode=='freeze':need(not OUT.exists(),'Existing R48 output');OUT.mkdir(parents=True,exist_ok=False)
    else:need(OUT.is_dir(),'Freeze required')
    started=now();tick=time.monotonic();code=1;failure=None;log=OUT/(mode+'.log')
    with log.open('x',encoding='utf8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            try:
                if mode=='freeze':freeze();code=0
                else:code=run()
            except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':48,'command':sys.argv,'started_at_utc':started,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-tick,
        'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_path':rel(log),'log_sha256':sha(log),'automatic_retry':False}
    write(OUT/(mode+'_execution.json'),event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    raise SystemExit(execute(parser.parse_args().command))
