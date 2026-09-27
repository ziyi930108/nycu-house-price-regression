"""Frozen-iteration v2 controls for long-horizon, chronological backtests."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from model_common import ROOT,load_data,temporal_indices,mae
from prepare_data import sha256

SPLITS={'development':('2022-08-11','2023-08-10'),'confirmation':('2023-08-11','2024-08-10')}
OUT=ROOT/'outputs/round3/baseline'

def run(split,component):
    if split=='confirmation' and not (ROOT/'outputs/round3/frozen_candidates.json').is_file():
        raise RuntimeError('Freeze candidates and acceptance rules before confirmation')
    d=load_data();start,end=SPLITS[split]
    fi,vi=temporal_indices(d,start,end=end)
    assert len(fi)>10000 and len(vi)>10000 and not np.intersect1d(fi,vi).size
    xf=d['X_train'].iloc[fi].reset_index(drop=True); xv=d['X_train'].iloc[vi].reset_index(drop=True)
    yf=d['y_train'][fi]; yv=d['y_train'][vi]
    folder=OUT/split;folder.mkdir(parents=True,exist_ok=True)
    for name,index in [('fit_indices',fi),('check_indices',vi)]:
        path=folder/f'{name}.npy'
        if path.exists(): assert np.array_equal(np.load(path),index)
        else: np.save(path,index)
    config=json.loads((ROOT/'runs/v1/models/final_config.json').read_text(encoding='utf-8'))
    chosen=next(c for c in config['components'] if c['family']==component) if component!='reference' else None
    began=time.perf_counter()
    print(json.dumps({"event":"started","split":split,"component":component,"fit_rows":len(fi),"check_rows":len(vi)}),flush=True)
    if component=='catboost':
        from catboost import CatBoostRegressor,Pool
        params={**chosen['parameters'],'iterations':chosen['iterations'],'thread_count':4,'verbose':250}
        model=CatBoostRegressor(**params)
        model.fit(Pool(xf,yf/1e6,cat_features=d['categorical_columns']))
        model_path=folder/'catboost_raw.cbm';model.save_model(str(model_path))
        prediction=np.maximum(model.predict(Pool(xv,cat_features=d['categorical_columns']),thread_count=4)*1e6,0)
        name='catboost_raw'
    elif component=='lightgbm':
        import lightgbm as lgb
        from train_lightgbm import encode_features,price_factor
        encoded,maps,safe,cats=encode_features({'train':xf,'check':xv},d['categorical_columns'])
        factor=price_factor(xf,'unit')
        params={**chosen['parameters'],'n_estimators':chosen['iterations'],'n_jobs':4}
        model=lgb.LGBMRegressor(**params)
        model.fit(encoded['train'],yf/factor,sample_weight=factor/1e6,categorical_feature=cats)
        model_path=folder/'lightgbm_unit63.txt';model.booster_.save_model(str(model_path))
        (folder/'category_maps.json').write_text(json.dumps(maps,ensure_ascii=False),encoding='utf-8')
        prediction=np.maximum(model.predict(encoded['check'],num_threads=4)*price_factor(xv,'unit'),0)
        name='lightgbm_l1_unit63'
    else:
        import pickle
        from round2_neighborhood import fit_final_reference,predict_reference
        base=.75*np.load(folder/'catboost_raw.npy')+.25*np.load(folder/'lightgbm_l1_unit63.npy')
        reference=fit_final_reference(xf,yf)
        model_path=folder/'reference_v2.pkl'
        with model_path.open('wb') as handle: pickle.dump(reference,handle,pickle.HIGHEST_PROTOCOL)
        prediction=predict_reference(reference,xv,base,variant='trend')
        name='v2_control';params=reference['parameters'] if 'parameters' in reference else {}
        np.save(folder/'v2_anchor.npy',predict_reference(reference,xv,base,variant='anchor'))
    assert prediction.shape==yv.shape and np.isfinite(prediction).all()
    np.save(folder/f'{name}.npy',prediction)
    known=set(xf.project_name)-{'__MISSING__'};seen=xv.project_name.isin(known).to_numpy()
    meta={'split':split,'component':name,'fit_rows':len(fi),'check_rows':len(vi),'start':start,'end_inclusive':end,
          'mae':mae(yv,prediction),'seen_mae':mae(yv[seen],prediction[seen]),'unseen_mae':mae(yv[~seen],prediction[~seen]),
          'seen_rows':int(seen.sum()),'unseen_rows':int((~seen).sum()),
          'params':params,'fit_uses_eval_set':False,'iterations_selected_from_confirmation':False,
          'source_sha256':d['metadata']['source_sha256'],'feature_source_sha256':sha256(ROOT/'src/house_features.py'),
          'script_sha256':sha256(Path(__file__)),'model_path':model_path.relative_to(ROOT).as_posix(),
          'model_sha256':sha256(model_path),'prediction_sha256':sha256(folder/f'{name}.npy'),
          'elapsed_seconds':time.perf_counter()-began,'external_property_data':False}
    (folder/f'{name}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ['params','source_sha256']},ensure_ascii=True),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=SPLITS,required=True)
    p.add_argument('--component',choices=['catboost','lightgbm','reference'],required=True)
    args=p.parse_args();run(args.split,args.component)
