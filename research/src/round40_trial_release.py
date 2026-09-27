"""R40 final unit component plus fixed R39 raw/log branches; one trial-only fit."""
from __future__ import annotations
import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round19_unit_geo as unit
import round34_trial_release as old

ROOT = old.ROOT
OUT = ROOT / 'outputs/round40/trial'
CSV = ROOT / 'outputs/submission_v8_trial.csv'
DOC = ROOT / 'reports/round40_trial_plan.md'
FINAL = 'outputs/round39/trial/artifacts/'
GEO_MANIFEST = 'outputs/round39/geography/final/manifest.json'
AUTHORITIES = {
 'outputs/round40/results.json':'aeb5e2f7d65802ace77d765848725032d89bcd326a3e45a5026627e2cc9cd9b1',
 'outputs/round19/candidate_plan.json':'1dda509846fcb8841d9a8a333d58da79fc70a625b9f18ae9a91dc8b7303ce251',
 'outputs/round34/trial/trial_release.json':'67069dd953999d56c6e1e1fab70a5d7e0a0f98272a1bcde48a3bb538c5b64033',
 'outputs/round37/trial/trial_release.json':'e7794550b976eca72fe3e6fb0bcdbdf02a24ce0b8e008139d328a2422bb827fe',
 'outputs/round39/trial/trial_release.json':'5e385afdf721e6c4925745c7120ee6a98b6284dac0bf7e93d8f4e9a6d31d560c',
 GEO_MANIFEST:'1fb450e986a8f72bd8fd0a85ba52c195b6021592328f680621e499e22ed3f35b',
}
CSVS = tuple('outputs/submission_'+v+('.csv' if v in ('v2','v3') else '_trial.csv') for v in ('v2','v3','v4','v5','v6','v7'))
PROTECTED = CSVS + ('runs/versions.json',)
need, sha, rel, read, write, spatial = old.need, old.sha, old.rel, old.read, old.write_json, old.spatial
FORMULA = '(saved_R39_raw_spatial_TWD + saved_R39_log_spatial_TWD + new_unit_spatial_TWD) / 3.0'


def sources():
    return old.merge(old.sources(), unit.source_hashes(), {rel(Path(__file__)):sha(Path(__file__))})


def capture():
    inputs, docs, known = {}, {}, {}
    def bind(name, digest):
        need(name not in inputs or inputs[name]==digest, 'Conflicting identity: '+name)
        old.verify({name:digest}); inputs[name]=digest
    for name,digest in AUTHORITIES.items():
        bind(name,digest); doc=old.locked(ROOT/name); docs[name]=doc
        receipt=rel((ROOT/name).with_suffix('.sha256')); bind(receipt,sha(ROOT/receipt))
        for key in ('input_sha256','artifact_sha256','source_sha256'):
            known=old.merge(known,doc.get(key,{}))
    result=docs['outputs/round40/results.json']
    need(result['round']==40 and result['all_periods_necessary_condition'] is True
         and result['checked_stages']==['development','confirmation','auxiliary'], 'R40 prerequisite differs')
    baseline=docs['outputs/round19/candidate_plan.json']
    need(unit.parameters()==baseline['parameters']==baseline['baseline_parameters']
         and baseline['target_formula']==unit.TARGET_FORMULA and baseline['inverse_formula']==unit.INVERSE_FORMULA
         and baseline['sample_weight']==unit.WEIGHT_FORMULA, 'Fixed R19 recipe differs')
    p34,p37,p39=[docs[f'outputs/round{r}/trial/trial_release.json'] for r in (34,37,39)]
    need(p39['round']==39 and p39['query_rows']==47073 and p39['fit_rows']==467221, 'Wrong v7 authority')
    known[p39['path']]=p39['sha256']
    names=[f'data_cache/{s}_features.pkl' for s in ('train','valid','test')]
    names += [f'data_cache/{s}.{ext}' for s in ('train','valid') for ext in ('pkl','json')]
    names += ['reports/feature_schema.json','runs/v1/models/final_config.json',rel(old.SAMPLE),old.REFERENCE,old.OLD_RAW,old.OLD_UNIT]
    names += [old.GEO+n for n in ('train_source_cache_indices.npy','valid_source_cache_indices.npy','full_source_cache_indices.npy','test_source_cache_indices.npy','full.pkl','test.pkl')]
    names += [old.MODEL+n for n in ('metadata.json','fit_indices.npy','query_indices.npy')]
    names += [FINAL+n for n in ('raw_predictions.npy','log_predictions.npy','predictions.npy','spatial_state.pkl','state_summary.json')]
    for name in names:
        need(name in known,'Historical input not bound: '+name); bind(name,known[name])
    manifest=docs[GEO_MANIFEST]
    need(manifest['round']==39 and manifest['stage']=='final' and manifest['snapshot_year']==2024
         and manifest['spatial_only'] is True and manifest['status']=='prepared_spatial_only','Wrong geography')
    for record in manifest['features'].values(): bind(record['path'],record['sha256'])
    src=sources()
    for name,digest in src.items():
        if name!=rel(Path(__file__)):
            need(known.get(name)==digest,'Historical source changed: '+name)
        bind(name,digest)
    bind(rel(DOC),sha(DOC))
    protected={n:sha(ROOT/n) for n in PROTECTED}
    for name,digest in protected.items():
        if name in known: need(known[name]==digest,'Old CSV changed: '+name)
        bind(name,digest)
    return inputs,src,protected,p34,p37,manifest


def array(path,value):
    with path.open('xb') as stream: np.save(stream,value,allow_pickle=False)


def dump(path,value):
    with path.open('xb') as stream: pickle.dump(value,stream,protocol=5)


def load(name):
    with (ROOT/name).open('rb') as stream: return pickle.load(stream)


def exact(a,b,label):
    need(a.dtype==b.dtype==np.dtype('float64') and a.shape==b.shape==(47073,)
         and a.tobytes()==b.tobytes(),'Full vector differs: '+label)
    return {'rows':47073,'bitwise_equal':True,'maximum_absolute_difference':0.0}


def csv_check(path,ids,prices):
    f=pd.read_csv(path,encoding='utf-8-sig',dtype={old.ID:str},keep_default_na=False)
    need(list(f.columns)==[old.ID,old.TARGET] and len(f)==47073 and f[old.ID].is_unique
         and f[old.ID].tolist()==ids and f[old.TARGET].dtype==np.dtype('int64')
         and (f[old.TARGET]>0).all() and np.array_equal(f[old.TARGET].to_numpy(),prices),'CSV format/order differs')


def raw_predict(model,state,X,geo,cats):
    unit.check_state(state,X,cats); capacity=unit.check_model(model,state)
    frame=unit.augment_features(X,geo)
    need(list(frame.columns)==state['all_feature_names'],'Query feature order differs')
    z=np.asarray(model.predict(Pool(frame,cat_features=state['categorical_columns'],thread_count=4),thread_count=4),dtype='float64')
    need(z.shape==(len(X),) and np.isfinite(z).all(),'Invalid raw unit prediction')
    return z,capacity


def prepare():
    started=time.monotonic()
    inputs,src,protected,p34,p37,manifest=capture()
    write(OUT/'input_contract.json',{'round':40,'stage':'final','trial_only':True,'formula':FORMULA,
      'parameters':unit.parameters(),'target_formula':unit.TARGET_FORMULA,'inverse_formula':unit.INVERSE_FORMULA,
      'sample_weight':unit.WEIGHT_FORMULA,'source_sha256':src,'input_sha256':inputs,
      'protected_sha256':protected,'runtime':old.runtime(),'catboost_version':old.importlib.metadata.version('catboost')})
    schema=read(ROOT/'reports/feature_schema.json')
    frames={s:pd.read_pickle(ROOT/f'data_cache/{s}_features.pkl') for s in ('train','valid','test')}
    for split,count in (('train',455784),('valid',11994),('test',47073)):
        f=frames[split]
        need(len(f)==count and f.index.equals(pd.RangeIndex(count)) and list(f)==schema['columns']
             and len(f.columns)==87 and old.TARGET not in f,'Original feature schema differs')
    kt=old.index(old.GEO+'train_source_cache_indices.npy',455784,455255)
    kv=old.index(old.GEO+'valid_source_cache_indices.npy',11994,11994,True)
    kf=old.index(old.GEO+'full_source_cache_indices.npy',467249,467221)
    kq=old.index(old.GEO+'test_source_cache_indices.npy',47073,47073,True)
    X=pd.concat([frames['train'].iloc[kt],frames['valid'].iloc[kv]],ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq=frames['test'].iloc[kq].reset_index(drop=True)
    labels={}
    for split,count in (('train',455784),('valid',11994)):
        raw=pd.read_pickle(ROOT/f'data_cache/{split}.pkl'); info=read(ROOT/f'data_cache/{split}.json')
        need(len(raw)==count and raw.index.equals(pd.RangeIndex(count)) and list(raw)==info['columns']
             and info['source_sha256']==schema['splits'][split]['input_source_sha256'],'Target/X cache mapping differs')
        labels[split]=raw[old.TARGET].to_numpy(dtype='float64',copy=True); del raw
    y=np.concatenate([labels['train'][kt],labels['valid'][kv]])[kf]; del labels,frames
    need(y.shape==(467221,) and np.isfinite(y).all() and (y>0).all(),'Invalid final fit labels')
    oldmeta=read(ROOT/(old.MODEL+'metadata.json')); target_sha=unit.values_sha256(y)
    need(target_sha==oldmeta['fit_target_values_sha256']==p37['fit_target_values_sha256'],'Final target alignment differs')
    fi=old.index(old.MODEL+'fit_indices.npy',467221,467221,True)
    qi=old.index(old.MODEL+'query_indices.npy',47073,47073,True)
    geo,geoq=[pd.read_pickle(ROOT/(old.GEO+n)) for n in ('full.pkl','test.pkl')]
    before={n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}
    need(before==p34['input_frame_sha256']==p37['input_frame_sha256'],'Final original X/geo alignment differs')
    newquery=pd.read_pickle(ROOT/manifest['features']['query']['path'])
    need(spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'] and newquery.index.equals(Xq.index)
         and newquery.dtypes.equals(geoq.dtypes) and list(newquery)==list(geoq),'New spatial query geography differs')
    state_spatial=load(FINAL+'spatial_state.pkl'); summary=read(ROOT/(FINAL+'state_summary.json'))
    reference=load(old.REFERENCE)
    need(state_spatial['state_summary']==summary and summary['fit_rows']==len(X)
         and summary['fit_X_sha256']==before['X'] and summary['fit_geo_sha256']==manifest['features']['fit']['frame_sha256']
         and summary['query_used_in_build'] is False and summary['parameters']==spatial.parameters()
         and summary['existing_reference_residual_sha256']==unit.values_sha256(reference['reference_residual']),'R39 saved fit-only spatial state differs')
    raw_branch,log_branch,v7=[old.vector(FINAL+n) for n in ('raw_predictions.npy','log_predictions.npy','predictions.npy')]
    control=exact(.5*raw_branch+.5*log_branch,v7,'v7 saved branch mean')
    sample=pd.read_csv(old.SAMPLE,encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str},keep_default_na=False)
    ids=sample[old.ID].tolist(); need(len(ids)==47073 and sample[old.ID].is_unique and all(ids),'Sample IDs invalid')
    csv_check(ROOT/'outputs/submission_v7_trial.csv',ids,old.rounded(v7))
    write(OUT/'control_replay.json',{'v7_vector':control,'v7_CSV_exact':True,'saved_branches_reused':True,'no_tree_replay':True})
    modeldir=OUT/'model'; modeldir.mkdir(exist_ok=False)
    array(modeldir/'fit_indices.npy',fi); array(modeldir/'query_indices.npy',qi)
    cats=schema['categorical_columns']; need(len(cats)==21 and len(unit.categories(X,cats))==23,'Category schema differs')
    write(OUT/'fit_started.json',{'started_at_utc':old.stamp(),'fit_rows':len(X),'query_rows':len(Xq),
        'fit_target_values_sha256':target_sha,'parameters':unit.parameters(),'fits_started':1,'automatic_retry':False})
    print(json.dumps({'event':'final_unit_fit_started','fit_rows':len(X),'query_rows':len(Xq)}),flush=True)
    fit_start=time.monotonic(); model,state,metadata=unit.fit_model(X,y,geo,cats); fit_seconds=time.monotonic()-fit_start
    model.save_model(str(modeldir/'model.cbm')); write(modeldir/'feature_state.json',state)
    fitted_native=model.get_all_params()
    need(metadata['parameters']==unit.parameters() and metadata['fit_rows']==467221 and metadata['feature_count']==95
         and metadata['categorical_feature_count']==23 and metadata['fit_target_values_sha256']==target_sha
         and metadata['sample_weight']==unit.WEIGHT_FORMULA and metadata['weight_argument_passed_to_pool'] is True
         and not metadata['query_labels_used'] and not metadata['fit_uses_eval_set'] and not metadata['early_stopping_used'],'R19 fitted contract differs')
    need(unit.values_sha256(y)==target_sha,'Fit target changed'); del y
    write(modeldir/'fit_metadata.json',{**metadata,'fitted_native_parameters':fitted_native,'fit_elapsed_seconds':fit_seconds})
    z,capacity=raw_predict(model,state,Xq,geoq,cats); component=unit.inverse_transform(z,Xq)
    array(modeldir/'raw_z.npy',z); array(modeldir/'predictions.npy',component); del model
    restored=CatBoostRegressor(); restored.load_model(str(modeldir/'model.cbm'))
    savedstate=old.locked(modeldir/'feature_state.json'); need(savedstate==state,'State JSON roundtrip differs')
    zr,loaded_capacity=raw_predict(restored,savedstate,Xq,geoq,cats); pr=unit.inverse_transform(zr,Xq)
    raw_reload=exact(z,zr,'saved CBM raw'); TWD_reload=exact(component,pr,'saved CBM TWD')
    loaded_native=restored.get_all_params(); need(capacity==loaded_capacity,'Native capacity changed on reload')
    for config in (fitted_native,loaded_native):
        need(config['random_seed']==42 and config['loss_function']==config['eval_metric']=='MAE'
             and np.isclose(config['learning_rate'],.065,rtol=1e-7,atol=0),'Native fixed recipe differs')
    del restored
    metadata.update({'round':40,'stage':'final','source_sha256':src,'input_contract_sha256':sha(OUT/'input_contract.json'),
       'fitted_native_parameters':fitted_native,'saved_native_parameters':loaded_native,'native_capacity':loaded_capacity,
       'fit_elapsed_seconds':fit_seconds,'full_query_raw_TWD_reload_bitwise_equal':True,'reload_rows':47073,
       'model_sha256':sha(modeldir/'model.cbm'),'feature_state_sha256':sha(modeldir/'feature_state.json'),
       'raw_z_sha256':sha(modeldir/'raw_z.npy'),'predictions_sha256':sha(modeldir/'predictions.npy'),
       'raw_z_units':'ten_thousand_TWD_per_effective_square_metre','test_labels_loaded':False})
    write(modeldir/'metadata.json',metadata)
    print(json.dumps({'event':'final_unit_fit_and_reload_complete','fit_elapsed_seconds':fit_seconds}),flush=True)
    B=.75*(.5*old.vector(old.OLD_RAW)+.5*component)+.25*old.vector(old.OLD_UNIT)
    p,diag,reference_report=old.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state_spatial,Xq,newquery,B,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    need(branch[fallback].tobytes()==p[fallback].tobytes(),'Unit spatial fallback changed')
    values=(raw_branch+log_branch+branch)/3.0
    need(values.dtype==np.dtype('float64') and values.shape==(47073,) and np.isfinite(values).all() and (values>0).all(),'Invalid trial mean')
    for name,arr in (('base_unit',B),('reference_unit',p),('unit_spatial',branch),('predictions',values)):
        array(OUT/(name+'.npy'),arr)
    dump(OUT/'reference_diagnostics.pkl',diag); dump(OUT/'spatial_diagnostics.pkl',detail)
    prices=old.rounded(values); staged=OUT/CSV.name
    pd.DataFrame({old.ID:ids,old.TARGET:prices}).to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
    with staged.open('rb') as stream: need(stream.read(3)==b'\xef\xbb\xbf','CSV BOM absent')
    csv_check(staged,ids,prices)
    need({n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}==before,'Original features mutated')
    need(spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'],'Spatial query frame mutated')
    need(sources()==src,'Source changed during run'); old.verify(protected)
    need(not CSV.exists(),'Refusing existing CSV'); os.link(staged,CSV)
    proof={'round':40,'status':'experimental_trial_generated','trial_only':True,'formal_acceptance':False,
      'path':rel(CSV),'sha256':sha(CSV),'fit_rows':467221,'query_rows':47073,'feature_count':95,'categorical_count':23,
      'formula':FORMULA,'parameters':unit.parameters(),'source_sha256':src,'input_sha256':inputs,
      'input_contract_sha256':sha(OUT/'input_contract.json'),'new_tree_fits':1,'old_tree_models_loaded':0,
      'fit_elapsed_seconds':fit_seconds,'elapsed_seconds':time.monotonic()-started,
      'v7_control':control,'v7_CSV_exact':True,'raw_reload':raw_reload,'TWD_reload':TWD_reload,
      'reference_predictions_new_unit_branch':1,'reference_refitted':False,'new_spatial_state_builds':0,
      'new_spatial_predictions':1,'spatial_state_reused':FINAL+'spatial_state.pkl','test_labels_loaded':False,
      'reference_report':reference_report,'spatial_report':spatial_report,'fallback_bitwise_preserved':True,
      'original_frames_unchanged':True,'rounding':'max(rint((raw+log+unit)/3),1), once',
      'ID_order_verified':True,'UTF8_BOM':True,'positive_int64_roundtrip':True,
      'changed_unrounded_rows_vs_v7':int(np.count_nonzero(values!=v7)),
      'changed_rounded_rows_vs_v7':int(np.count_nonzero(prices!=old.rounded(v7))),
      'mean_absolute_prediction_change_TWD':float(np.abs(values-v7).mean()),
      'maximum_absolute_prediction_change_TWD':float(np.abs(values-v7).max()),
      'prediction_changes_are_not_error_metrics':True,'new_public_mae':None,'private_mae':None,
      'registry_updated':False,'uploaded':False,'old_decisions_changed':False,
      'artifact_sha256':{rel(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='execution.log'},
      'protected_before':protected,'protected_after':{n:sha(ROOT/n) for n in PROTECTED},'completed_at_utc':old.stamp()}
    need(proof['protected_before']==proof['protected_after'],'Protected outputs changed')
    write(OUT/'trial_release.json',proof)
    print(json.dumps({'event':'trial_generated','path':rel(CSV),'sha256':proof['sha256']}),flush=True)


def run():
    need(not OUT.exists() and not CSV.exists(),'Existing attempt; no overwrite or automatic retry')
    OUT.mkdir(parents=True,exist_ok=False); start=time.monotonic(); stamp=old.stamp(); code=1; error=None
    write(OUT/'run_claim.json',{'source_sha256':sha(Path(__file__)),'started_at_utc':stamp,'automatic_retry':False})
    with (OUT/'execution.log').open('x',encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try: prepare(); code=0
            except Exception: error=traceback.format_exc(); print(error,flush=True)
    event={'round':40,'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':old.stamp(),
       'elapsed_seconds':time.monotonic()-start,'exit_code':code,'failure':error,'automatic_retry':False,
       'source_sha256':sha(Path(__file__)),'log_path':rel(OUT/'execution.log'),'log_sha256':sha(OUT/'execution.log')}
    write(OUT/'execution.json',event); print(json.dumps(event,ensure_ascii=False),flush=True); return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args(); raise SystemExit(run())
