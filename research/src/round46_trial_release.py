"""R46 final recency-log component and saved raw/unit median; one trial-only fit."""
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
import round46_recency_log as learner
import round46_experiment as experiment
import round34_trial_release as old

ROOT = old.ROOT
OUT = ROOT / 'outputs/round46/trial'
CSV = ROOT / 'outputs/submission_v10_trial.csv'
DOC = ROOT / 'reports/round46_trial_plan.md'
FINAL = 'outputs/round39/trial/artifacts/'
GEO_MANIFEST = 'outputs/round39/geography/final/manifest.json'
AUTHORITIES = {
 'outputs/round46/results.json':'338fd86a01399930d29055be07471823014f6c5ad8da0dd5ad1f7ed1fa96042e',
 'outputs/round46/plan.json':'ae866e6cf95b57a28495eda74c40d4326fe4a2fa9a1e00afbdb973e9c6e59c7f',
 'outputs/round46/final_date_support.json':'54e3e95c0ede403d512a40c5ed2983fde1d4223c0b9730f8f2e3a4901ae98e2c',
 'outputs/round43/trial/release.json':'e8643d9e977d1b429001359ddf936a5e06a938f2bc3caf4454777a0d75f1d4ea',
 'outputs/round40/trial/trial_release.json':'c5f33956319569f951b99c7a79de089a4c85ae464803d18fb25d128730e5ca2d',
 'outputs/round34/trial/trial_release.json':'67069dd953999d56c6e1e1fab70a5d7e0a0f98272a1bcde48a3bb538c5b64033',
 'outputs/round37/trial/trial_release.json':'e7794550b976eca72fe3e6fb0bcdbdf02a24ce0b8e008139d328a2422bb827fe',
 'outputs/round39/trial/trial_release.json':'5e385afdf721e6c4925745c7120ee6a98b6284dac0bf7e93d8f4e9a6d31d560c',
 GEO_MANIFEST:'1fb450e986a8f72bd8fd0a85ba52c195b6021592328f680621e499e22ed3f35b',
}
V9 = 'outputs/round43/trial/predictions.npy'
V5 = 'outputs/round37/trial/predictions.npy'
UNIT_BRANCH = 'outputs/round40/trial/unit_spatial.npy'
FINAL_DATE_POLICY = 'valid fit dates: exp2(-age/365) normalized over valid fit rows; invalid/missing dates: neutral weight 1; retain all 467221 rows'

CSVS = tuple('outputs/submission_'+v+('.csv' if v in ('v2','v3') else '_trial.csv') for v in ('v2','v3','v4','v5','v6','v7','v8','v9'))
PROTECTED = CSVS + ('runs/versions.json',)
need, sha, rel, read, write, spatial = old.need, old.sha, old.rel, old.read, old.write_json, old.spatial
FORMULA = 'median([saved_R39_raw_spatial_TWD,new_recency_log_spatial_TWD,saved_R40_unit_spatial_TWD],axis=1)'


def sources():
    return old.merge(old.sources(), experiment.sources(), {rel(Path(__file__)):sha(Path(__file__))})


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
    result=docs['outputs/round46/results.json']; plan=docs['outputs/round46/plan.json']
    need(result['round']==46 and result['status']=='three_periods_completed_no_adoption_decision'
         and result['checked_stages']==['development','confirmation','auxiliary']
         and result['plan_sha256']==AUTHORITIES['outputs/round46/plan.json'], 'R46 prerequisite differs')
    need(learner.parameters()==plan['parameters'] and learner.WEIGHT_FORMULA==plan['weight_formula']
         and experiment.sources()==result['source_sha256']==plan['source_sha256'],'Fixed R46 recipe differs')
    for number,name in ((43,'outputs/round43/trial/release.json'),(40,'outputs/round40/trial/trial_release.json')):
        d=docs[name]; need(d['round']==number and d['status']=='experimental_trial_generated' and d['query_rows']==47073,'Wrong saved trial authority')
        known[d['path']]=d['sha256']
    known[V9]=docs['outputs/round43/trial/release.json']['predictions_sha256']
    p34,p37,p39=[docs[f'outputs/round{r}/trial/trial_release.json'] for r in (34,37,39)]
    need(p39['round']==39 and p39['query_rows']==47073 and p39['fit_rows']==467221, 'Wrong v7 authority')
    known[p39['path']]=p39['sha256']
    names=[f'data_cache/{s}_features.pkl' for s in ('train','valid','test')]
    names += [f'data_cache/{s}.{ext}' for s in ('train','valid') for ext in ('pkl','json')]
    names += ['reports/feature_schema.json','runs/v1/models/final_config.json',rel(old.SAMPLE),old.REFERENCE,old.OLD_RAW,old.OLD_UNIT]
    names += [old.GEO+n for n in ('train_source_cache_indices.npy','valid_source_cache_indices.npy','full_source_cache_indices.npy','test_source_cache_indices.npy','full.pkl','test.pkl')]
    names += [old.MODEL+n for n in ('metadata.json','fit_indices.npy','query_indices.npy')]
    names += [FINAL+n for n in ('raw_predictions.npy','log_predictions.npy','predictions.npy','spatial_state.pkl','state_summary.json')]
    names += [V9,V5,UNIT_BRANCH]
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


def final_weights(X):
    days=X['trade_day_index'].to_numpy(dtype='float64')
    invalid=X['trade_date_invalid'].to_numpy(dtype='float64'); outside=X['trade_date_out_of_range'].to_numpy(dtype='float64')
    need(len(X)==467221 and np.isfinite(invalid).all() and np.isfinite(outside).all(),'Final date flag schema differs')
    valid=np.isfinite(days)&(invalid==0)&(outside==0)
    expected=read(ROOT/'outputs/round46/final_date_support.json')['invalid_canonical_positions']
    need(np.flatnonzero(~valid).tolist()==expected and int(valid.sum())==467213,'Observed missing-date identities differ')
    weights=np.ones(len(X),dtype='float64'); anchor=float(days[valid].max()); age=anchor-days[valid]
    need(np.isfinite(age).all() and (age>=0).all(),'Invalid finite-date age')
    unnormalized=np.exp2(-age/365.0); denominator=float(np.mean(unnormalized,dtype='float64'))
    need(np.isfinite(unnormalized).all() and (unnormalized>0).all() and denominator>0,'Invalid recency weights')
    weights[valid]=unnormalized/denominator
    need(np.isfinite(weights).all() and (weights>0).all() and abs(float(weights.mean())-1)<1e-12,'Invalid final normalized weights')
    return weights,{'policy':FINAL_DATE_POLICY,'rows':len(X),'valid_date_rows':int(valid.sum()),'neutral_date_rows':int((~valid).sum()),
      'neutral_canonical_positions':expected,'invalid_date_flag_rows':int((invalid!=0).sum()),'out_of_range_flag_rows':int((outside!=0).sum()),
      'max_valid_fit_trade_day_index':anchor,'min_valid_fit_trade_day_index':float(days[valid].min()),
      'normalization_mean_valid_fit_only':denominator,'min':float(weights.min()),'max':float(weights.max()),
      'mean':float(weights.mean()),'sum':float(weights.sum()),'valid_date_mean':float(weights[valid].mean()),
      'weights_use_target':False,'weights_use_query':False,'weight_floor':None,'weight_half_life_days':365}


def prepare():
    started=time.monotonic()
    inputs,src,protected,p34,p37,manifest=capture()
    write(OUT/'input_contract.json',{'round':46,'stage':'final','trial_only':True,'formula':FORMULA,
      'parameters':learner.parameters(),'target_formula':learner.TARGET_FORMULA,'inverse_formula':learner.INVERSE_FORMULA,
      'sample_weight':learner.WEIGHT_FORMULA,'final_date_policy':FINAL_DATE_POLICY,'source_sha256':src,'input_sha256':inputs,
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
    weights,weight_summary=final_weights(X); weight_sha=learner.values_sha256(weights)
    labels={}
    for split,count in (('train',455784),('valid',11994)):
        raw=pd.read_pickle(ROOT/f'data_cache/{split}.pkl'); info=read(ROOT/f'data_cache/{split}.json')
        need(len(raw)==count and raw.index.equals(pd.RangeIndex(count)) and list(raw)==info['columns']
             and info['source_sha256']==schema['splits'][split]['input_source_sha256'],'Target/X cache mapping differs')
        labels[split]=raw[old.TARGET].to_numpy(dtype='float64',copy=True); del raw
    y=np.concatenate([labels['train'][kt],labels['valid'][kv]])[kf]; del labels,frames
    need(y.shape==(467221,) and np.isfinite(y).all() and (y>0).all(),'Invalid final fit labels')
    oldmeta=read(ROOT/(old.MODEL+'metadata.json')); target_sha=learner.values_sha256(y)
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
         and summary['existing_reference_residual_sha256']==learner.values_sha256(reference['reference_residual']),'R39 saved fit-only spatial state differs')
    raw_branch,log_branch,v7=[old.vector(FINAL+n) for n in ('raw_predictions.npy','log_predictions.npy','predictions.npy')]
    unit_branch,v9,v5=[old.vector(n) for n in (UNIT_BRANCH,V9,V5)]
    v7_control=exact(.5*raw_branch+.5*log_branch,v7,'v7 saved branch mean')
    control=exact(np.median(np.column_stack([raw_branch,log_branch,unit_branch]),axis=1),v9,'v9 saved branch median')
    sample=pd.read_csv(old.SAMPLE,encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str},keep_default_na=False)
    ids=sample[old.ID].tolist(); need(len(ids)==47073 and sample[old.ID].is_unique and all(ids),'Sample IDs invalid')
    csv_check(ROOT/'outputs/submission_v7_trial.csv',ids,old.rounded(v7))
    csv_check(ROOT/'outputs/submission_v9_trial.csv',ids,old.rounded(v9))
    csv_check(ROOT/'outputs/submission_v5_trial.csv',ids,old.rounded(v5))
    write(OUT/'control_replay.json',{'v9_vector':control,'v7_vector':v7_control,'v9_v5_v7_CSV_exact':True,'saved_branches_reused':True,'no_tree_replay':True})
    modeldir=OUT/'model'; modeldir.mkdir(exist_ok=False)
    array(modeldir/'fit_indices.npy',fi); array(modeldir/'query_indices.npy',qi)
    cats=schema['categorical_columns']; need(len(cats)==21 and len(learner.categories(X,cats))==23,'Category schema differs')
    write(OUT/'fit_started.json',{'started_at_utc':old.stamp(),'fit_rows':len(X),'query_rows':len(Xq),
        'fit_target_values_sha256':target_sha,'fit_weights_sha256':weight_sha,'fit_weight_summary':weight_summary,
        'parameters':learner.parameters(),'fits_started':1,'automatic_retry':False})
    print(json.dumps({'event':'final_recency_log_fit_started','fit_rows':len(X),'query_rows':len(Xq)}),flush=True)
    fit_start=time.monotonic()
    state=learner.fit_feature_state(X,geo,cats); frame=learner.augment_features(X,geo)
    transformed=learner.transform_target(y); transformed_sha=learner.values_sha256(transformed)
    model=CatBoostRegressor(**learner.parameters())
    model.fit(Pool(frame,transformed,weight=weights,cat_features=learner.categories(X,cats),thread_count=4))
    fit_seconds=time.monotonic()-fit_start
    model.save_model(str(modeldir/'model.cbm')); write(modeldir/'feature_state.json',state)
    fitted_native=experiment.native(model,state,cats)
    need(state['fit_rows']==len(X) and state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
         and state['geo_state']['fit_original_feature_values_sha256']==before['X']
         and learner.values_sha256(y)==target_sha and learner.values_sha256(transformed)==transformed_sha
         and learner.values_sha256(weights)==weight_sha,'Final fit state/values changed')
    metadata={'round':46,'stage':'final','component':'geography_recency_log_catboost_with_neutral_missing_date_final_policy',
        'parameters':learner.parameters(),'target_formula':learner.TARGET_FORMULA,'inverse_formula':learner.INVERSE_FORMULA,
        'feature_count':95,'categorical_feature_count':23,'fit_rows':467221,'sample_weight':learner.WEIGHT_FORMULA,
        'weight_argument_passed_to_pool':True,'final_date_policy':FINAL_DATE_POLICY,'fit_weight_summary':weight_summary,
        'fit_weights_sha256':weight_sha,'fit_weight_date_values_sha256':learner.values_sha256(X['trade_day_index']),
        'fit_original_target_values_sha256':target_sha,'fit_transformed_target_values_sha256':transformed_sha,
        'fit_uses_eval_set':False,'early_stopping_used':False,'query_labels_used':False,'test_labels_loaded':False,
        'source_sha256':src,'input_contract_sha256':sha(OUT/'input_contract.json'),'original_frame_sha256':before,
        'fitted_native_parameters':fitted_native,'fit_elapsed_seconds':fit_seconds,
        'legacy_state_training_fields_are_provenance_only':True,'neutral_missing_policy_applied_to_historical_evaluation':False}
    write(modeldir/'fit_metadata.json',metadata)
    z,component=experiment.raw_predict(model,state,Xq,geoq,cats)
    array(modeldir/'raw_z.npy',z); array(modeldir/'predictions.npy',component); del model,y,weights,transformed,frame
    restored=CatBoostRegressor(); restored.load_model(str(modeldir/'model.cbm'))
    savedstate=old.locked(modeldir/'feature_state.json'); need(savedstate==state,'State JSON roundtrip differs')
    zr,pr=experiment.raw_predict(restored,savedstate,Xq,geoq,cats)
    raw_reload=exact(z,zr,'saved CBM raw'); TWD_reload=exact(component,pr,'saved CBM TWD')
    loaded_native=experiment.native(restored,savedstate,cats); del restored
    metadata.update(saved_native_parameters=loaded_native,full_query_raw_TWD_reload_bitwise_equal=True,reload_rows=47073,
        model_sha256=sha(modeldir/'model.cbm'),feature_state_sha256=sha(modeldir/'feature_state.json'),
        raw_z_sha256=sha(modeldir/'raw_z.npy'),predictions_sha256=sha(modeldir/'predictions.npy'),
        raw_z_units='log1p_TWD_per_million',raw_z_negative_count=int((z<0).sum()))
    write(modeldir/'metadata.json',metadata)
    print(json.dumps({'event':'final_recency_log_fit_and_reload_complete','fit_elapsed_seconds':fit_seconds}),flush=True)
    B=.75*(.5*old.vector(old.OLD_RAW)+.5*component)+.25*old.vector(old.OLD_UNIT)
    p,diag,reference_report=old.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state_spatial,Xq,newquery,B,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    need(branch[fallback].tobytes()==p[fallback].tobytes(),'Recency log spatial fallback changed')
    values=np.median(np.column_stack([raw_branch,branch,unit_branch]),axis=1)
    need(values.dtype==np.dtype('float64') and values.shape==(47073,) and np.isfinite(values).all() and (values>0).all(),'Invalid trial median')
    for name,arr in (('base_recency_log',B),('reference_recency_log',p),('recency_log_spatial',branch),('predictions',values)):
        array(OUT/(name+'.npy'),arr)
    dump(OUT/'reference_diagnostics.pkl',diag); dump(OUT/'spatial_diagnostics.pkl',detail)
    def scalar(v):
        if pd.isna(v):return None
        return v.item() if isinstance(v,np.generic) else v
    fields=['address_city','address_district','transaction_type','building_type','main_use','building_area','land_area','parking_area','parking_count']
    changes={}; top12={}
    for name,baseline in (('v9',v9),('v5',v5)):
        diff=values-baseline
        changes[name]={'changed_unrounded_rows':int(np.count_nonzero(diff)),
          'changed_rounded_rows':int(np.count_nonzero(old.rounded(values)!=old.rounded(baseline))),
          'mean_absolute_prediction_change_TWD':float(np.abs(diff).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(diff).max()),
          'minimum_signed_change_TWD':float(diff.min()),'maximum_signed_change_TWD':float(diff.max())}
        top12[name]=[{'position':int(i),'ID':ids[i],'pure_X':{c:scalar(Xq.iloc[i][c]) for c in fields},
          'v9_TWD':float(v9[i]),'v5_TWD':float(v5[i]),'v10_TWD':float(values[i]),
          'new_log_spatial_TWD':float(branch[i]),'saved_raw_spatial_TWD':float(raw_branch[i]),'saved_unit_spatial_TWD':float(unit_branch[i]),
          'signed_prediction_change_TWD':float(diff[i]),'true_price_unknown':True}
          for i in np.argsort(-np.abs(diff),kind='stable')[:12]]
    write(OUT/'prediction_changes.json',{'prediction_changes_are_not_MAE':True,'changes':changes,'top12_absolute_change':top12,
          'test_labels_loaded':False,'no_top_row_adjustment_or_price_cap':True})
    prices=old.rounded(values); staged=OUT/CSV.name
    pd.DataFrame({old.ID:ids,old.TARGET:prices}).to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
    with staged.open('rb') as stream: need(stream.read(3)==b'\xef\xbb\xbf','CSV BOM absent')
    csv_check(staged,ids,prices)
    need({n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}==before,'Original features mutated')
    need(spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'],'Spatial query frame mutated')
    need(sources()==src,'Source changed during run'); old.verify(protected)
    need(not CSV.exists(),'Refusing existing CSV'); os.link(staged,CSV)
    proof={'round':46,'status':'experimental_trial_generated','trial_only':True,'formal_acceptance':False,
      'path':rel(CSV),'sha256':sha(CSV),'fit_rows':467221,'query_rows':47073,'feature_count':95,'categorical_count':23,
      'formula':FORMULA,'parameters':learner.parameters(),'source_sha256':src,'input_sha256':inputs,
      'input_contract_sha256':sha(OUT/'input_contract.json'),'new_tree_fits':1,'old_tree_models_loaded':0,
      'fit_elapsed_seconds':fit_seconds,'elapsed_seconds':time.monotonic()-started,
      'v9_control':control,'v7_control':v7_control,'v9_v5_v7_CSV_exact':True,'raw_reload':raw_reload,'TWD_reload':TWD_reload,
      'reference_predictions_new_recency_log_branch':1,'reference_refitted':False,'new_spatial_state_builds':0,
      'new_spatial_predictions':1,'spatial_state_reused':FINAL+'spatial_state.pkl','test_labels_loaded':False,
      'reference_report':reference_report,'spatial_report':spatial_report,'fallback_bitwise_preserved':True,
      'original_frames_unchanged':True,'rounding':'max(rint(median([raw,new_log,unit])),1), once',
      'ID_order_verified':True,'UTF8_BOM':True,'positive_int64_roundtrip':True,
      'prediction_changes_vs_controls':changes,'top12_pure_X_diagnostics_path':rel(OUT/'prediction_changes.json'),
      'final_date_policy':FINAL_DATE_POLICY,'fit_weight_summary':weight_summary,'fit_weights_sha256':weight_sha,
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
    event={'round':46,'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':old.stamp(),
       'elapsed_seconds':time.monotonic()-start,'exit_code':code,'failure':error,'automatic_retry':False,
       'source_sha256':sha(Path(__file__)),'log_path':rel(OUT/'execution.log'),'log_sha256':sha(OUT/'execution.log')}
    write(OUT/'execution.json',event); print(json.dumps(event,ensure_ascii=False),flush=True); return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args(); raise SystemExit(run())
