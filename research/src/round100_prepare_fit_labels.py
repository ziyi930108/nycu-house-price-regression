"""R100 stage-specific full-precision fit labels; no model or query scoring.
The raw pickle is fully materialized only in this extractor, then only verified
outer-fit indices are exported. Stage model processes must not read this pickle.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, time, traceback
import numpy as np
import pandas as pd
import round98_contract as io

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round100_inputs'
PLAN='reports/round100_plan.md'
PLAN_SHA='bbb29bcc4fb83eaa3fdac9f15ade8221ced3437abacb04e891038ffb3462e107'
ORIGINAL='outputs/round79/plan.json'
ORIGINAL_SHA='7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f'
MAPPING='outputs/round6/geography/development/train_source_cache_indices.npy'
MAPPING_SHA='7694e4fcc14989221c3f82abac1210b5258339c8ebd2b84261df4b2f74941ee4'
RAW='data_cache/train.pkl'
FEATURES='data_cache/train_features.pkl'
STAGES=('development','confirmation','auxiliary')

def select_fit_y(raw, raw_ids):
    io.need(isinstance(raw,pd.DataFrame) and raw.index.equals(pd.RangeIndex(len(raw))), 'Raw frame axis')
    io.need(raw_ids.dtype==np.int64 and raw_ids.ndim==1 and len(raw_ids)>0, 'Raw index shape/dtype')
    io.need((np.diff(raw_ids)>0).all() and raw_ids[0]>=0 and raw_ids[-1]<len(raw),'Raw index bounds/order')
    y=raw.iloc[raw_ids]['總價元'].to_numpy(dtype=np.float64,copy=True)
    io.need(y.shape==raw_ids.shape and np.isfinite(y).all() and (y>0).all(),'Fit target finite positive')
    return y

def run():
    tick=time.monotonic();start=io.now()
    io.need(not OUT.exists(),'Existing extraction is immutable')
    OUT.mkdir();code=1;failure=None
    source=io.source_hashes(__file__); protected=io.protected_hashes()
    io.write(OUT/'started.json',{'started_at_utc':start,'source_sha256':source,'raw_pickle_scope':'extractor only; entire raw frame materialized'})
    try:
        io.need(io.sha(PLAN)==PLAN_SHA and io.sha(ORIGINAL)==ORIGINAL_SHA,'Plan identity')
        plan=io.read(ORIGINAL); inputs={PLAN:PLAN_SHA,ORIGINAL:ORIGINAL_SHA,MAPPING:MAPPING_SHA}
        for name in (RAW,FEATURES): inputs[name]=plan['input_sha256'][name]
        io.verify_hashes(inputs)
        mapping=np.load(ROOT/MAPPING,allow_pickle=False)
        io.need(mapping.dtype==np.int64 and mapping.shape==(455255,) and (np.diff(mapping)>0).all() and mapping[0]>=0 and mapping[-1]<455784,'Canonical mapping')
        rawX=pd.read_pickle(ROOT/FEATURES)
        io.need(len(rawX)==455784 and rawX.index.equals(pd.RangeIndex(455784)) and not {'總價元','編號','target','y'}.intersection(rawX.columns),'Pure-X source')
        X=rawX.iloc[mapping].reset_index(drop=True);del rawX
        days=X.trade_day_index.to_numpy(dtype=np.float64)
        selected={}
        for stage in STAGES:
            path=f'outputs/round63/{stage}/fit_indices.npy'
            inputs[path]=plan['input_sha256'][path]
            io.need(io.sha(path)==inputs[path],'Fit index identity')
            fi=np.load(ROOT/path,allow_pickle=False);spec=plan['split_contract'][stage]
            io.need(fi.dtype==np.int64 and fi.shape==(spec['fit_rows'],) and (np.diff(fi)>0).all(),'Fit index schema')
            io.need(io.values_sha(fi)==spec['fit_indices_content_sha256'],'Canonical fit values')
            cutoff=plan['splits'][stage][0]
            startday=(pd.Timestamp(cutoff)-pd.Timestamp('2000-01-01')).days
            expected=np.flatnonzero(np.isfinite(days)&(days<startday)).astype(np.int64)
            io.exact(fi,expected,'Original fit date mask')
            io.need((X.iloc[fi].trade_date_invalid==0).all() and (X.iloc[fi].trade_date_out_of_range==0).all(),'Outer-fit valid dates')
            selected[stage]=(fi,mapping[fi],cutoff)
        raw=pd.read_pickle(ROOT/RAW)
        io.need(len(raw)==455784,'Raw table length')
        stages={}
        for stage in STAGES:
            fi,raw_ids,cutoff=selected[stage];y=select_fit_y(raw,raw_ids)
            folder=OUT/stage;folder.mkdir()
            arrays={name:io.save_array(folder/(name+'.npy'),a) for name,a in
                    {'fit_indices':fi,'fit_source_cache_indices':raw_ids,'fit_y':y}.items()}
            stages[stage]={'fit_rows':len(fi),'cutoff':cutoff,'query_end':plan['splits'][stage][1],
                'query_rows':plan['split_contract'][stage]['query_rows'],'arrays':arrays,
                'query_y_exported_to_this_stage':False,'dtype':'float64','units':'TWD',
                'fit_canonical_values_sha256':io.values_sha(fi),'fit_raw_cache_values_sha256':io.values_sha(raw_ids)}
        del raw
        io.verify_hashes(inputs);io.need(io.source_hashes(__file__)==source and io.protected_hashes()==protected,'Input/source/protected changed')
        manifest={'schema_version':1,'status':'fit_only_labels_sealed','plan_sha256':PLAN_SHA,
            'stage_order':list(STAGES),'stages':stages,'raw_train':{'path':RAW,'sha256':inputs[RAW],'rows':455784},
            'features':{'path':FEATURES,'sha256':inputs[FEATURES]},'canonical_mapping':{'path':MAPPING,'sha256':MAPPING_SHA},
            'input_sha256':inputs,'source_sha256':source,'raw_price_table_deserialized_in_extractor':True,
            'cross_stage_role_overlap':'Earlier query may be legal later-stage fit labels; isolated stage processes required',
            'query_statistics_computed':False,'new_native_fits':0,'new_prior_builds':0,
            'test_read':False,'valid_read':False,'all_arrays_saved_reload_exact':True,
            'protected_before':protected,'protected_after':io.protected_hashes(),
            'started_at_utc':start,'completed_at_utc':io.now(),'elapsed_seconds':time.monotonic()-tick}
        io.write(OUT/'manifest.json',manifest);code=0
        print(json.dumps({'status':manifest['status'],'fit_rows':{k:v['fit_rows'] for k,v in stages.items()},'elapsed_seconds':manifest['elapsed_seconds']}))
    except Exception:
        failure=traceback.format_exc();print(failure,flush=True)
    finally:
        io.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':source,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False})
    return code

def self_check():
    tick=time.monotonic(); checks=[]
    f=pd.DataFrame({'總價元':[10.,20.,30.,40.,50.],'noise':['a','b','c','d','e']})
    before=f.copy(deep=True)
    ids=np.array([0,2,4],dtype=np.int64);v=select_fit_y(f,ids)
    io.exact(v,np.array([10.,30.,50.],dtype=np.float64),'Nonidentity raw selection');checks.append('raw_index_selection_not_reset_index')
    io.need(f.equals(before),'Input mutation');checks.append('caller_unchanged')
    for bad in (np.array([0,0],dtype=np.int64),np.array([2,1],dtype=np.int64),np.array([-1,1],dtype=np.int64),np.array([0,5],dtype=np.int64)):
        try:select_fit_y(f,bad)
        except ValueError:checks.append('invalid_indices_rejected')
        else:raise AssertionError('Invalid index accepted')
    r={'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':io.sha(__file__),
       'real_data_read':False,'new_fits':0,'elapsed_seconds':time.monotonic()-tick}
    io.write('reports/round100_fit_label_selfcheck.json',r);print(json.dumps(r));return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--self-check',action='store_true');a=ap.parse_args()
    if a.run==a.self_check:ap.error('Choose exactly one mode')
    raise SystemExit(self_check() if a.self_check else run())
