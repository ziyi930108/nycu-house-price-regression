"""Train fixed round-five components; no scoring, tuning, or shared-cache changes.

The immutable candidate plan is required before development training. Later
stages also require the frozen development choice, and final refit requires
successful confirmation and auxiliary evidence. No eval_set is ever supplied.
"""
from __future__ import annotations
import argparse,time,pickle,uuid
from datetime import datetime,timezone
import numpy as np
from catboost import CatBoostRegressor,Pool
from round5_common import *
from train_lightgbm import encode_features,price_factor,effective_area

def publish(stage,target):
    require(stage.resolve().parent==target.resolve().parent and stage.resolve().is_relative_to(OUT.resolve()),'Unsafe artifact move')
    require(not target.exists(),'Target already exists')
    stage.rename(target)

def save_common(folder,stage,component,data,X,Xq,fi,qi,pred,metadata,started):
    np.save(folder/'fit_indices.npy',fi,allow_pickle=False);np.save(folder/'query_indices.npy',qi,allow_pickle=False)
    pred=np.asarray(pred,dtype='float64')
    require(pred.shape==(len(Xq),) and np.isfinite(pred).all() and (pred>=0).all(),'Invalid component predictions')
    np.save(folder/'predictions.npy',pred,allow_pickle=False)
    if 'source_sha256' in metadata:metadata['component_api_source_sha256']=metadata.pop('source_sha256')
    if 'elapsed_seconds' in metadata:metadata['component_api_fit_elapsed_seconds']=metadata['elapsed_seconds']
    metadata.update({'stage':stage,'component':component,'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'outer_fit_rows':len(X),'query_rows':len(Xq),'dates':None if stage=='final' else list(SPLITS[stage]),
       'plan_sha256':sha256(OUT/'candidate_plan.json'),'source_sha256':data['metadata']['source_sha256'],
       'fit_source_sha256':fit_source_hashes(),'fit_indices_sha256':sha256(folder/'fit_indices.npy'),
       'query_indices_sha256':sha256(folder/'query_indices.npy'),'prediction_sha256':sha256(folder/'predictions.npy'),
       'fit_uses_eval_set':False,'early_stopping_used':False,'query_labels_used':False,
       'prediction_units':'TWD','direct_component_mae_omitted':True,'elapsed_seconds':time.monotonic()-started,
       'frozen_candidate_sha256':None if stage=='development' else sha256(OUT/'frozen_candidate.json')})
    write(folder/'metadata.json',metadata)

def component_run(stage,component):
    authorize(stage,component)
    target=OUT/'models'/stage/component
    require(not target.exists(),'Component already exists; never overwrite recorded experiments')
    data,X,y,Xq,yq,fi,qi=partition(stage)
    reference,reference_path=load_reference(stage,X)
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.parent/f'.{component}-staging-{uuid.uuid4().hex}';staging.mkdir()
    started=time.monotonic();cats=data['categorical_columns']
    print(__import__('json').dumps({'event':'fit_started','stage':stage,'component':component,'outer_fit_rows':len(X),'query_rows':len(Xq)}),flush=True)
    require(component=='project_mask_catboost','Only the prespecified masked component is permitted')
    from round5_project_mask import fit_model,predict_model,query_mask
    model,state,metadata=fit_model(X,y,cats)
    pred=predict_model(model,state,Xq,cats)
    model.save_model(str(staging/'model.cbm'));write(staging/'feature_state.json',state)
    reload=CatBoostRegressor();reload.load_model(str(staging/'model.cbm'))
    replay=predict_model(reload,read(staging/'feature_state.json'),Xq.iloc[:64],cats)
    require(int(reload.tree_count_)==2999,'Reloaded model tree count differs')
    metadata['feature_state_sha256']=sha256(staging/'feature_state.json')
    query_flags=query_mask(Xq,state)
    metadata['query_masked_rows']=int(query_flags.sum())
    metadata['query_masked_fraction']=float(query_flags.mean())
    require(np.allclose(pred[:64],replay,rtol=1e-12,atol=1e-5),'Reloaded component predictions differ')
    metadata.update({'model_sha256':sha256(staging/'model.cbm'),
       'reference_path':rel(reference_path),'reference_sha256':sha256(reference_path),'reload_64_rows_verified':True})
    save_common(staging,stage,component,data,X,Xq,fi,qi,pred,metadata,started)
    authorize(stage,component);publish(staging,target)
    print(__import__('json').dumps({'event':'fit_completed','stage':stage,'component':component,'output':rel(target),'elapsed_seconds':time.monotonic()-started}),flush=True)

def auxiliary_baseline(component):
    authorize('auxiliary','control')
    data,X,y,Xq,yq,fi,qi=partition('auxiliary')
    folder=baseline_folder('auxiliary');folder.mkdir(parents=True,exist_ok=True)
    for name,values in [('fit_indices.npy',fi),('check_indices.npy',qi)]:
        path=folder/name
        if path.exists():require(np.array_equal(np.load(path,allow_pickle=False),values),'Baseline indices differ')
        else:np.save(path,values,allow_pickle=False)
    cfg=read(ROOT/'runs/v1/models/final_config.json');cats=data['categorical_columns'];started=time.monotonic()
    name={'catboost':'catboost_raw','lightgbm':'lightgbm_l1_unit63','reference':'v2_control'}[component]
    require(not (folder/f'{name}.json').exists(),'Auxiliary baseline already completed')
    print(__import__('json').dumps({'event':'baseline_started','component':component,'fit_rows':len(X),'query_rows':len(Xq)}),flush=True)
    if component=='catboost':
        chosen=next(c for c in cfg['components'] if c['family']=='catboost')
        config={**chosen['parameters'],'iterations':chosen['iterations'],'thread_count':4,'verbose':300}
        model=CatBoostRegressor(**config);model.fit(Pool(X,y/1e6,cat_features=cats,thread_count=4))
        pred=np.maximum(model.predict(Pool(Xq,cat_features=cats,thread_count=4),thread_count=4)*1e6,0)
        path=folder/'catboost_raw.cbm';model.save_model(str(path))
    elif component=='lightgbm':
        import lightgbm as lgb
        chosen=next(c for c in cfg['components'] if c['family']=='lightgbm')
        config={**chosen['parameters'],'n_estimators':chosen['iterations'],'n_jobs':4}
        encoded,maps,_,categorical=encode_features({'train':X,'query':Xq},cats)
        model=lgb.LGBMRegressor(**config)
        model.fit(encoded['train'],y/price_factor(X,'unit'),sample_weight=effective_area(X)/100,categorical_feature=categorical)
        pred=np.maximum(model.predict(encoded['query'])*price_factor(Xq,'unit'),0)
        path=folder/'lightgbm_unit63.txt';model.booster_.save_model(str(path));write(folder/'category_maps.json',maps)
    else:
        from round2_neighborhood import fit_final_reference,predict_reference
        for c in ['catboost_raw','lightgbm_l1_unit63']:
            meta=read(folder/f'{c}.json')
            require(meta['plan_sha256']==sha256(OUT/'candidate_plan.json'),'Auxiliary component plan differs')
            require(meta['prediction_sha256']==sha256(folder/f'{c}.npy'),'Auxiliary baseline prediction differs')
        reference=fit_final_reference(X,y);path=folder/'reference_v2.pkl'
        with path.open('wb') as handle:pickle.dump(reference,handle,protocol=pickle.HIGHEST_PROTOCOL)
        # Always use the persisted representation for subsequent identity guards.
        with path.open('rb') as handle:reference=pickle.load(handle)
        base=.75*np.load(folder/'catboost_raw.npy')+.25*np.load(folder/'lightgbm_l1_unit63.npy')
        pred=predict_reference(reference,Xq,base,variant='trend');config=reference['parameters']
    pred=np.asarray(pred,dtype='float64')
    require(pred.shape==(len(Xq),) and np.isfinite(pred).all() and (pred>=0).all(),'Invalid baseline predictions')
    np.save(folder/f'{name}.npy',pred,allow_pickle=False)
    metadata={'stage':'auxiliary','component':name,'created_at_utc':datetime.now(timezone.utc).isoformat(),'parameters':config,'outer_fit_rows':len(X),'query_rows':len(Xq),
       'dates':list(SPLITS['auxiliary']),'source_sha256':data['metadata']['source_sha256'],
       'fit_source_sha256':fit_source_hashes(),'plan_sha256':sha256(OUT/'candidate_plan.json'),
       'frozen_candidate_sha256':sha256(OUT/'frozen_candidate.json'),
       'model_path':rel(path),'model_sha256':sha256(path),'prediction_sha256':sha256(folder/f'{name}.npy'),
       'fit_indices_sha256':sha256(folder/'fit_indices.npy'),'query_indices_sha256':sha256(folder/'check_indices.npy'),
       'fit_uses_eval_set':False,'early_stopping_used':False,'query_labels_used':False,'elapsed_seconds':time.monotonic()-started}
    authorize('auxiliary','control');write(folder/f'{name}.json',metadata)
    print(__import__('json').dumps({'event':'baseline_completed','component':component,'output':rel(folder),'elapsed_seconds':time.monotonic()-started}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=[*SPLITS,'final'],required=True)
    p.add_argument('--component',choices=[*COMPONENTS,'control_catboost','control_lightgbm','control_reference'],required=True)
    args=p.parse_args()
    if args.component.startswith('control_'):
        require(args.stage=='auxiliary','Other controls are frozen and reused, not retrained')
        auxiliary_baseline(args.component.removeprefix('control_'))
    else:component_run(args.stage,args.component)
if __name__=='__main__':main()
