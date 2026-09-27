"""One R39 spatial-only final trial. No tree or reference fit/predict, no scoring."""
from __future__ import annotations
import argparse
import contextlib
import os
import pickle
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round34_trial_release as old

ROOT = old.ROOT
OUT = ROOT / 'outputs/round39/trial'
CSV = ROOT / 'outputs/submission_v7_trial.csv'
DOCUMENT = ROOT / 'reports/round39_trial_plan.md'
GEO_MANIFEST = 'outputs/round39/geography/final/manifest.json'
RESULT = 'outputs/round39/evaluation/results.json'
FORMULA = '.5*R8(new_state,new_geo,B_raw,p_raw,diag_raw)+.5*R8(new_state,new_geo,B_log,p_log,diag_log)'
AUTHORITIES = {
    RESULT: '1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27',
    GEO_MANIFEST: '1fb450e986a8f72bd8fd0a85ba52c195b6021592328f680621e499e22ed3f35b',
    'outputs/round34/trial/trial_release.json': '67069dd953999d56c6e1e1fab70a5d7e0a0f98272a1bcde48a3bb538c5b64033',
    'outputs/round37/trial/trial_release.json': 'e7794550b976eca72fe3e6fb0bcdbdf02a24ce0b8e008139d328a2422bb827fe',
    'outputs/round38/trial_release/artifacts/release.json': 'e4cd76d8e6dcf4102038d917217c09ec5d822fa23b7645461bbc8e574fac0980',
}
CSVS = {v: 'outputs/submission_' + v + ('.csv' if v in ('v2', 'v3') else '_trial.csv') for v in ('v2','v3','v4','v5','v6')}
PROTECTED = tuple(CSVS.values()) + ('runs/versions.json',) + tuple(AUTHORITIES)
need, sha, rel, read, write, spatial = old.need, old.sha, old.rel, old.read, old.write_json, old.spatial


def sources():
    return old.merge(old.sources(), {rel(Path(__file__)): sha(Path(__file__))})


def capture():
    inputs = {}
    def bind(name, digest):
        need(name not in inputs or inputs[name] == digest, 'Conflicting input: ' + name)
        old.verify({name: digest})
        inputs[name] = digest
    authorities = {}
    for name, digest in AUTHORITIES.items():
        bind(name, digest)
        authorities[name] = old.locked(ROOT / name)
        receipt = rel((ROOT / name).with_suffix('.sha256'))
        bind(receipt, sha(ROOT / receipt))
    result = authorities[RESULT]
    need(result['round'] == 39 and result['status'] == 'three_periods_completed_no_adoption_decision'
         and result['checked_stages'] == ['development','confirmation','auxiliary']
         and result['all_periods_not_worse_necessary_condition'] is True, 'R39 prerequisite differs')
    for stage in result['checked_stages']:
        name = 'outputs/round39/evaluation/' + stage + '/results.json'
        bind(name, result['stage_result_sha256'][stage])
        item = old.locked(ROOT / name)
        need(all(item['consider_trial_necessary_checks'].values()), 'Necessary stage check failed')
        bind(rel((ROOT/name).with_suffix('.sha256')), sha((ROOT/name).with_suffix('.sha256')))
    p34 = authorities['outputs/round34/trial/trial_release.json']
    p37 = authorities['outputs/round37/trial/trial_release.json']
    p38 = authorities['outputs/round38/trial_release/artifacts/release.json']
    known = {}
    for proof in (p34,p37,p38):
        for key in ('input_sha256','artifact_sha256','source_sha256'):
            for name,digest in proof.get(key,{}).items():
                need(name not in known or known[name] == digest, 'Historical identity conflict')
                known[name] = digest
    known[p38['prediction_path']] = p38['prediction_sha256']
    known[p38['path']] = p38['sha256']
    names = [f'data_cache/{s}_features.pkl' for s in ('train','valid','test')]
    names += ['reports/feature_schema.json', rel(old.SAMPLE), old.REFERENCE, old.OLD_RAW, old.OLD_UNIT, old.V3_VECTOR]
    names += [old.GEO+n for n in ('train_source_cache_indices.npy','valid_source_cache_indices.npy','full_source_cache_indices.npy','test_source_cache_indices.npy','full.pkl','test.pkl')]
    names += [old.MODEL+n for n in ('predictions.npy','fit_indices.npy','query_indices.npy')]
    names += ['outputs/round34/trial/'+n for n in ('spatial_state.pkl','state_summary.json','diagnostics.pkl','predictions.npy')]
    names += ['outputs/round37/trial/'+n for n in ('base_log.npy','p_log.npy','reference_diagnostics.pkl','predictions.npy')]
    names += [p38['prediction_path']]
    for name in names:
        need(name in known, 'Unbound historical input: ' + name)
        bind(name, known[name])
    geo = authorities[GEO_MANIFEST]
    need(geo['round']==39 and geo['stage']=='final' and geo['snapshot_year']==2024
         and geo['status']=='prepared_spatial_only' and geo['spatial_only'] is True, 'Wrong final geography')
    for kind in ('features','added_positions','provenance','indices'):
        for record in geo[kind].values():
            bind(record['path'], record['sha256'])
    for name,digest in geo['source_sha256'].items(): bind(name,digest)
    current_sources = sources()
    for name,digest in current_sources.items():
        if name != rel(Path(__file__)):
            need(p34['source_sha256'].get(name)==digest, 'Original spatial dependency differs: '+name)
        bind(name,digest)
    bind(rel(DOCUMENT), sha(DOCUMENT))
    protected = {n: sha(ROOT/n) for n in PROTECTED}
    for name,digest in protected.items():
        if name in known: need(known[name]==digest, 'Released CSV identity differs')
        bind(name,digest)
    return inputs, current_sources, protected, p34, p37, p38, geo


def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    need(a.dtype==b.dtype==np.dtype('float64') and a.shape==b.shape==(47073,)
         and a.tobytes()==b.tobytes(), 'Full vector differs: '+label)
    return {'rows':47073,'bitwise_equal':True,'maximum_absolute_difference':0.0}


def dump(path,value):
    with path.open('xb') as stream: pickle.dump(value,stream,protocol=5)


def array(path,value):
    with path.open('xb') as stream: np.save(stream,value,allow_pickle=False)


def load_pickle(name):
    with (ROOT/name).open('rb') as stream: return pickle.load(stream)


def check_csv(path,ids,prices):
    frame=pd.read_csv(path,encoding='utf-8-sig',dtype={old.ID:str},keep_default_na=False)
    need(list(frame.columns)==[old.ID,old.TARGET] and len(frame)==47073 and frame[old.ID].is_unique
         and frame[old.ID].tolist()==ids and frame[old.TARGET].dtype.kind in 'iu'
         and (frame[old.TARGET]>0).all() and np.array_equal(frame[old.TARGET].to_numpy(),prices), 'CSV content/order differs: '+str(path))


def prepare_trial():
    start=time.monotonic()
    inputs, source_map, protected, p34, p37, p38, manifest = capture()
    write(OUT/'input_contract.json', {'round':39,'formula':FORMULA,'source_sha256':source_map,
          'input_sha256':inputs,'protected_sha256':protected,'runtime':old.runtime(),
          'spatial_parameters':spatial.parameters(),'trial_only':True,'formal_acceptance':False,
          'historical_geo_dependencies_inherited_not_rehashed':manifest['inherited_geography_source_sha256']})
    schema=read(ROOT/'reports/feature_schema.json')
    frames={s:pd.read_pickle(ROOT/f'data_cache/{s}_features.pkl') for s in ('train','valid','test')}
    for split,count in (('train',455784),('valid',11994),('test',47073)):
        f=frames[split]
        need(len(f)==count and list(f.columns)==schema['columns'] and len(f.columns)==87
             and f.index.equals(pd.RangeIndex(count)) and old.TARGET not in f, 'X-only cache schema differs')
    kt=old.index(old.GEO+'train_source_cache_indices.npy',455784,455255)
    kv=old.index(old.GEO+'valid_source_cache_indices.npy',11994,11994,True)
    kf=old.index(old.GEO+'full_source_cache_indices.npy',467249,467221)
    kq=old.index(old.GEO+'test_source_cache_indices.npy',47073,47073,True)
    X=pd.concat([frames['train'].iloc[kt],frames['valid'].iloc[kv]],ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq=frames['test'].iloc[kq].reset_index(drop=True)
    old.index(old.MODEL+'fit_indices.npy',467221,467221,True)
    old.index(old.MODEL+'query_indices.npy',47073,47073,True)
    geo,geoq=[pd.read_pickle(ROOT/(old.GEO+n)) for n in ('full.pkl','test.pkl')]
    originals={'X':X,'Xq':Xq,'geo':geo,'geoq':geoq}
    before={n:spatial.frame_sha(f) for n,f in originals.items()}
    need(before==p34['input_frame_sha256']==p37['input_frame_sha256'], 'Final frame alignment differs')
    newgeo=[]
    for label,original in (('fit',geo),('query',geoq)):
        spec=manifest['features'][label]; new=pd.read_pickle(ROOT/spec['path'])
        positions=np.load(ROOT/manifest['added_positions'][label]['path'],allow_pickle=False)
        need(len(new)==spec['rows'] and spatial.frame_sha(new)==spec['frame_sha256']
             and new.index.equals(original.index) and new.dtypes.equals(original.dtypes)
             and list(new.columns)==list(original.columns), 'Prepared geo schema differs')
        need(positions.dtype==np.dtype('int64') and positions.ndim==1
             and np.array_equal(positions,np.unique(positions)) and (positions>=0).all() and (positions<len(new)).all(), 'Added positions differ')
        keep=np.ones(len(new),dtype=bool); keep[positions]=False
        for col in original:
            a,b=original.loc[keep,col],new.loc[keep,col]
            need(a.to_numpy().tobytes()==b.to_numpy().tobytes() if a.dtype.kind=='f' else a.equals(b), 'Unchanged geography altered')
        need(original.iloc[positions].geo_has_position.eq(0).all()
             and original.iloc[positions][['geo_latitude','geo_longitude']].isna().all().all()
             and new.iloc[positions].geo_has_position.eq(1).all(), 'Added position admission differs')
        newgeo.append(new)
    newfit,newquery=newgeo
    Braw=.75*(.5*old.vector(old.OLD_RAW)+.5*old.vector(old.MODEL+'predictions.npy'))+.25*old.vector(old.OLD_UNIT)
    praw=old.vector(old.V3_VECTOR)
    draw=load_pickle('outputs/round34/trial/diagnostics.pkl')[['v2_source','v2_confidence']].rename(columns={'v2_source':'source','v2_confidence':'confidence'})
    Blog=old.vector('outputs/round37/trial/base_log.npy'); plog=old.vector('outputs/round37/trial/p_log.npy')
    dlog=load_pickle('outputs/round37/trial/reference_diagnostics.pkl')
    need(draw.index.equals(Xq.index) and dlog.index.equals(Xq.index), 'Saved diagnostic index differs')
    oldstate=load_pickle('outputs/round34/trial/spatial_state.pkl')
    need(oldstate['state_summary']==read(ROOT/'outputs/round34/trial/state_summary.json')
         and oldstate['state_summary']['fit_X_sha256']==before['X']
         and oldstate['state_summary']['fit_geo_sha256']==before['geo'], 'Original spatial state differs')
    sample=pd.read_csv(old.SAMPLE,encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str},keep_default_na=False)
    ids=sample[old.ID].tolist()
    need(len(ids)==47073 and sample[old.ID].is_unique and all(ids), 'Sample IDs invalid')
    control=[]; replays={}
    for label,B,p,diag,version in (('raw',Braw,praw,draw,'v4'),('log',Blog,plog,dlog,'v5')):
        value,_,_=spatial.predict(oldstate,Xq,geoq,B,p,diag)
        saved=old.vector('outputs/round'+('34' if label=='raw' else '37')+'/trial/predictions.npy')
        replays[version]=exact(value,saved,version)
        check_csv(ROOT/CSVS[version],ids,old.rounded(value)); control.append(value)
    oldmean=.5*control[0]+.5*control[1]
    replays['v6']=exact(oldmean,old.vector(p38['prediction_path']),'v6')
    check_csv(ROOT/CSVS['v6'],ids,old.rounded(oldmean)); check_csv(ROOT/CSVS['v3'],ids,old.rounded(praw))
    v2ids=pd.read_csv(ROOT/CSVS['v2'],encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str})
    need(v2ids[old.ID].tolist()==ids,'v2 IDs differ')
    print('Original v4/v5/v6 vectors and CSVs reproduced exactly',flush=True)
    reference=load_pickle(old.REFERENCE)
    state,summary=spatial.build(reference,X,newfit)
    need(summary['existing_reference_residual_sha256']==oldstate['state_summary']['existing_reference_residual_sha256']
         and summary['saved_beta_sha256']==oldstate['state_summary']['saved_beta_sha256']
         and summary['saved_fill_values_sha256']==oldstate['state_summary']['saved_fill_values_sha256'], 'Saved reference normalization differs')
    artifact=OUT/'artifacts'; artifact.mkdir(exist_ok=False)
    dump(artifact/'spatial_state.pkl',state); write(artifact/'state_summary.json',summary)
    predictions=[]; branch_reports={}
    for label,B,p,diag in (('raw',Braw,praw,draw),('log',Blog,plog,dlog)):
        value,detail,report=spatial.predict(state,Xq,newquery,B,p,diag)
        array(artifact/(label+'_predictions.npy'),value); dump(artifact/(label+'_diagnostics.pkl'),detail)
        branch_reports[label]=report; predictions.append(value)
    values=.5*predictions[0]+.5*predictions[1]
    need(values.shape==(47073,) and values.dtype==np.dtype('float64') and np.isfinite(values).all() and (values>=0).all(), 'Invalid final vector')
    prices=old.rounded(values)
    array(artifact/'predictions.npy',values)
    staged=artifact/CSV.name
    pd.DataFrame({old.ID:ids,old.TARGET:prices}).to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
    with staged.open('rb') as stream: need(stream.read(3)==b'\xef\xbb\xbf','Missing UTF-8 BOM')
    check_csv(staged,ids,prices)
    exact(np.load(artifact/'predictions.npy',allow_pickle=False),values,'saved new mean')
    need({n:spatial.frame_sha(f) for n,f in originals.items()}==before,'Original frames mutated')
    need(spatial.frame_sha(newfit)==manifest['features']['fit']['frame_sha256']
         and spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'],'New geo frames mutated')
    need(sources()==source_map,'Sources changed during run')
    old.verify(protected)
    need(not CSV.exists(),'Refusing existing CSV')
    os.link(staged,CSV)  # Exclusive publication; preserves staging bytes on any later failure.
    check_csv(CSV,ids,prices)
    proof={'round':39,'status':'experimental_trial_generated','trial_only':True,'formal_acceptance':False,
        'recommended_version':'v3','registry_updated':False,'old_decisions_changed':False,'formula':FORMULA,
        'path':rel(CSV),'sha256':sha(CSV),'query_rows':47073,'fit_rows':467221,'geography_year':2024,
        'source_sha256':source_map,'input_sha256':inputs,'input_contract_sha256':sha(OUT/'input_contract.json'),
        'control_replays':replays,'control_CSVs_exact':True,'old_tree_models_trained_loaded_or_predicted':0,
        'reference_predictions_recomputed':0,'reference_refitted':False,'raw_labels_loaded':False,
        'saved_reference_contains_train_valid_target_derived_state':True,'new_spatial_state_builds':1,
        'new_spatial_predictions_per_branch':1,'independent_model_replay_performed':False,
        'reference_trend_not_reapplied':True,'branch_reports':branch_reports,'added_fit_positions':4017,'added_query_positions':327,
        'changed_unrounded_rows_vs_v6':int(np.count_nonzero(values!=oldmean)),
        'changed_rounded_rows_vs_v6':int(np.count_nonzero(prices!=old.rounded(oldmean))),
        'mean_absolute_TWD_change_vs_v6':float(np.abs(values-oldmean).mean()),
        'maximum_absolute_TWD_change_vs_v6':float(np.abs(values-oldmean).max()),
        'prediction_changes_are_not_error_metrics':True,'private_mae':None,'new_public_mae':None,
        'rounding':'max(rint(mean of two unrounded TWD spatial branches),1)',
        'ID_order_verified':True,'UTF8_BOM':True,'positive_int64_roundtrip':True,'original_frames_unchanged':True,
        'artifact_sha256':{rel(p):sha(p) for p in sorted(artifact.iterdir()) if p.is_file()},
        'protected_before':protected,'protected_after':{n:sha(ROOT/n) for n in PROTECTED},
        'elapsed_seconds':time.monotonic()-start,'completed_at_utc':old.stamp(),'runtime':old.runtime()}
    need(proof['protected_after']==protected,'Protected artifact changed')
    write(OUT/'trial_release.json',proof)
    print('experimental_trial_generated',rel(CSV),proof['sha256'],flush=True)
    return proof


def run():
    need(not OUT.exists() and not CSV.exists(),'Existing trial attempt/output; no automatic retry')
    OUT.mkdir(parents=True,exist_ok=False)
    started=old.stamp(); start=time.monotonic(); code=1; error=None
    write(OUT/'run_claim.json',{'started_at_utc':started,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    with (OUT/'execution.log').open('x',encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:
                prepare_trial(); code=0
            except Exception:
                error=traceback.format_exc(); print(error,flush=True)
    execution={'round':39,'command':sys.argv,'started_at_utc':started,'completed_at_utc':old.stamp(),
        'elapsed_seconds':time.monotonic()-start,'exit_code':code,'source_sha256':sha(Path(__file__)),
        'stdout_log':rel(OUT/'execution.log'),'stdout_log_sha256':sha(OUT/'execution.log'),
        'failure':error,'automatic_retry':False}
    write(OUT/'execution.json',execution)
    print(old.read(OUT/'execution.json'),flush=True)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args(); raise SystemExit(run())
