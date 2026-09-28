"""Rebuild the four fixed v22 tree components on a new, caller-verified fit window.

No loader, query labels, early stopping, parameter search or reference operation.
All outputs are new; the old recipe modules and their SHA checks remain intact.
"""
from __future__ import annotations
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
import lightgbm as lgb
import round46_recency_log as log
import round79_recency_unit as unit
import train_lightgbm as light

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'runs/v1/models/final_config.json'
COMPONENTS=('raw','light','log','unit')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def save_array(path,value):
    with Path(path).open('xb') as f:np.save(f,value,allow_pickle=False)

def exact(a,b,label):
    a=np.asarray(a);b=np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label

def parameters():
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    raw=next(v for v in config['components'] if v['family']=='catboost')
    lightc=next(v for v in config['components'] if v['family']=='lightgbm')
    return dict(raw={**raw['parameters'],'iterations':raw['iterations'],'thread_count':4,'verbose':250},
                light={**lightc['parameters'],'n_estimators':lightc['iterations'],'n_jobs':4},
                log=log.parameters(),unit=unit.parameters())

def source_hashes():
    return {**log._sources(),**unit.source_hashes(),
            'src/round90_offset_trees.py':sha(Path(__file__)),
            'src/train_lightgbm.py':sha(Path(light.__file__)),
            'runs/v1/models/final_config.json':sha(CONFIG)}

def fit_trees(X,y,G,Xq,Gq,cats,date_weights,out):
    """One fit per component, sequential. Reload verifies every query row."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    y=np.asarray(y,dtype=np.float64)
    assert y.shape==(len(X),) and np.isfinite(y).all() and (y>=0).all()
    assert X.index.equals(G.index) and Xq.index.equals(Gq.index)
    assert list(X.columns)==list(Xq.columns) and list(G.columns)==list(Gq.columns)
    w=np.asarray(date_weights)
    assert w.dtype==np.float32 and w.shape==y.shape and np.isfinite(w).all() and (w>0).all()
    assert set(cats)==set(X.select_dtypes(include=['object','string','category']).columns)
    params=parameters();sources=source_hashes()
    before=dict(X=log.base.frame_sha(X),Xq=log.base.frame_sha(Xq),
                G=log.base.frame_sha(G),Gq=log.base.frame_sha(Gq),
                y=unit.values_sha256(y),
                date_weights=hashlib.sha256(w.astype('<f4').tobytes()).hexdigest())
    allpred={};allmeta={}
    for component in COMPONENTS:
        folder=out/component;folder.mkdir()
        start=time.monotonic()
        write(folder/'started.json',dict(component=component,parameters=params[component],
            fit_rows=len(X),query_rows=len(Xq),query_labels_available=False,
            source_sha256=sources,input_value_sha256=before))
        if component=='raw':
            model=CatBoostRegressor(**params['raw'])
            fit_start=time.monotonic()
            model.fit(Pool(X,y/1e6,cat_features=cats))
            fit_seconds=time.monotonic()-fit_start
            native=folder/'model.cbm';model.save_model(str(native))
            prediction=np.maximum(model.predict(Pool(Xq,cat_features=cats),thread_count=4)*1e6,0)
            restored=CatBoostRegressor();restored.load_model(str(native))
            reloaded=np.maximum(restored.predict(Pool(Xq,cat_features=cats),thread_count=4)*1e6,0)
            assert int(model.tree_count_)==int(restored.tree_count_)==2999
            meta=dict(fit_seconds=fit_seconds,iterations=2999,target='TWD / 1e6',
                      weights=None,feature_count=len(X.columns))
        elif component=='light':
            # Mapping fit on X only; query cannot add a vocabulary entry.
            encoded,maps,safe,catcols=light.encode_features({'train':X,'query':Xq},cats)
            factor=light.price_factor(X,'unit')
            model=lgb.LGBMRegressor(**params['light'])
            fit_start=time.monotonic()
            model.fit(encoded['train'],y/factor,sample_weight=factor/1e6,categorical_feature=catcols)
            fit_seconds=time.monotonic()-fit_start
            native=folder/'model.txt';model.booster_.save_model(str(native))
            write(folder/'category_state.json',dict(category_maps=maps,safe_names=safe,categorical_columns=catcols))
            state=json.loads((folder/'category_state.json').read_text(encoding='utf-8'))
            encoded_reload,_,safe_reload,cat_reload=light.encode_features({'train':X,'query':Xq},cats,state['category_maps'])
            assert safe_reload==state['safe_names'] and cat_reload==state['categorical_columns']
            pd.testing.assert_frame_equal(encoded['query'],encoded_reload['query'],check_exact=True)
            prediction=np.maximum(model.predict(encoded['query'],num_threads=4)*light.price_factor(Xq,'unit'),0)
            restored=lgb.Booster(model_file=str(native))
            reloaded=np.maximum(restored.predict(encoded_reload['query'],num_threads=4)*light.price_factor(Xq,'unit'),0)
            assert model.booster_.current_iteration()==restored.current_iteration()==3999
            meta=dict(fit_seconds=fit_seconds,iterations=3999,target='TWD / (effective_area * 1e4)',
                      weights='effective_area / 100',feature_count=len(X.columns),
                      categories_fit_only=True,category_state_sha256=sha(folder/'category_state.json'))
        else:
            helper=log if component=='log' else unit
            args=(X,y,G,cats) if component=='log' else (X,y,G,cats,w)
            fit_start=time.monotonic()
            model,state,fitmeta=helper.fit_model(*args)
            fit_seconds=time.monotonic()-fit_start
            native=folder/'model.cbm';model.save_model(str(native))
            write(folder/'feature_state.json',state)
            write(folder/'fit_metadata.json',fitmeta)
            restored=CatBoostRegressor();restored.load_model(str(native))
            savedstate=json.loads((folder/'feature_state.json').read_text(encoding='utf-8'))
            prediction=helper.predict_model(model,state,Xq,Gq,cats)
            reloaded=helper.predict_model(restored,savedstate,Xq,Gq,cats)
            assert int(model.tree_count_)==int(restored.tree_count_)==2999
            meta=dict(fit_seconds=fit_seconds,iterations=2999,
                      feature_state_sha256=sha(folder/'feature_state.json'),
                      fit_metadata_sha256=sha(folder/'fit_metadata.json'),
                      feature_count=95)
        prediction=np.asarray(prediction,dtype=np.float64)
        reloaded=np.asarray(reloaded,dtype=np.float64)
        assert prediction.shape==(len(Xq),) and np.isfinite(prediction).all() and (prediction>=0).all()
        exact(prediction,reloaded,component+' full-query model reload')
        save_array(folder/'predictions_twd.npy',prediction)
        meta.update(component=component,status='completed',fit_rows=len(X),query_rows=len(Xq),
                    new_native_fits=1,query_labels_available=False,query_labels_used=False,
                    eval_set_used=False,early_stopping_used=False,parameters=params[component],
                    model_sha256=sha(native),prediction_sha256=sha(folder/'predictions_twd.npy'),
                    source_sha256=sources,input_value_sha256=before,
                    saved_model_reload_exact=True,reload_rows=len(Xq),
                    elapsed_seconds=time.monotonic()-start)
        write(folder/'metadata.json',meta)
        allpred[component]=prediction;allmeta[component]=meta
        print(json.dumps(dict(event='tree_complete',component=component,fit_seconds=meta['fit_seconds'],
                              elapsed_seconds=meta['elapsed_seconds'])),flush=True)
    after=dict(X=log.base.frame_sha(X),Xq=log.base.frame_sha(Xq),
               G=log.base.frame_sha(G),Gq=log.base.frame_sha(Gq),y=unit.values_sha256(y),
               date_weights=hashlib.sha256(w.astype('<f4').tobytes()).hexdigest())
    assert before==after and sources==source_hashes() and params==parameters()
    write(out/'results.json',dict(status='four_components_completed',new_native_fits=4,
          query_labels_used=False,source_sha256=sources,fit_inputs_unchanged=True,
          query_features_unchanged=True,components=allmeta))
    return allpred,allmeta
