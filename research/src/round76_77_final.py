"""Frozen R76/R77 final composition; one separately saved area fit, no CSV authoring."""
from pathlib import Path
import argparse, contextlib, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round48_final_branch as f48

ROOT=f48.ROOT; old=f48.old; exp=f48.experiment; spatial=f48.spatial
sha,rel,write=f48.sha,f48.rel,f48.write
OUT=ROOT/'outputs/round76_77/final'
PLAN=ROOT/'reports/round76_77_final_plan.md'
AUTHORITIES={
 'outputs/round71/final/trial_release.json':'5a9a1fa7c481992a839f7da94bc38ce8ee418ce1ff334abbc1e76fd23a94d2b3',
 'outputs/round40/trial/trial_release.json':'c5f33956319569f951b99c7a79de089a4c85ae464803d18fb25d128730e5ca2d',
 'outputs/round77/results.json':'6e9b837a9bfe5918cb5200831ae31268d3ad204aec21430da1b0577de2be8337'}


def run(plan_sha):
    tick=time.monotonic(); inputs={}; known={}; sources={}
    def bind(name,digest):
        name=name.replace(chr(92),'/'); path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(),name
        assert name not in inputs or inputs[name]==digest,name
        if name not in inputs: assert sha(path)==digest,name
        inputs[name]=digest; return path
    def locked(name,digest):
        path=bind(name,digest); receipt=path.with_suffix('.sha256')
        assert receipt.read_text(encoding='ascii').strip()==digest,name
        bind(rel(receipt),sha(receipt)); return json.loads(path.read_text(encoding='utf8'))
    bind(rel(PLAN),plan_sha)
    p71,p40,h77=[locked(n,h) for n,h in AUTHORITIES.items()]
    area_path='outputs/round77/final_area/results.json'
    area=locked(area_path,sha(ROOT/area_path))
    assert area['status']=='completed' and area['plan_sha256']==plan_sha
    assert area['fit_rows']==467221 and area['query_rows']==47073 and area['saved_model_reload_exact']
    assert h77['checked_stages']==list(exp.STAGES)
    for doc in (p71,p40,area,h77):
        for n,h in doc['source_sha256'].items():
            assert n not in sources or sources[n]==h,n
            sources[n]=h
    sources[rel(Path(__file__))]=sha(Path(__file__))
    for n,h in sources.items():bind(n,h)
    for doc in (p71,p40,area):
        for family in ('input_sha256','artifact_sha256'):
            for n,h in doc[family].items():
                n=n.replace(chr(92),'/'); assert n not in known or known[n]==h,n
                known[n]=h
    def checked(n): return bind(n,known[n])
    def small(n): return json.loads(checked(n).read_text(encoding='utf8'))
    def vec(n,zero=False):
        a=np.load(checked(n),allow_pickle=False)
        assert a.dtype==np.float64 and a.shape==(47073,) and np.isfinite(a).all()
        assert (a>=0).all() if zero else (a>0).all()
        return a
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    assert len(protected)==20
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    assert protected['outputs/submission_v19_trial.csv']=='287943c5238b67a1e5a2a5313089f766a5b014d82fea386a5c26117b79e7edb8'
    for n,h in p71['protected_after'].items():assert protected[n]==h,n
    selector=ROOT/'outputs/current_private_candidate.json'; selector_sha=sha(selector)
    assert selector_sha=='f5af20180e2ee77f12eb90058930e139c0256f29b0e943581ee7c44d293ee900'
    frames={s:pd.read_pickle(checked(f'data_cache/{s}_features.pkl')) for s in ('train','valid','test')}
    schema=small('reports/feature_schema.json')
    for s,n in (('train',455784),('valid',11994),('test',47073)):
        assert len(frames[s])==n and frames[s].index.equals(pd.RangeIndex(n))
        assert list(frames[s])==schema['columns'] and len(frames[s].columns)==87 and old.TARGET not in frames[s]
    def index(n,bound,count,identity=False):
        a=np.load(checked(n),allow_pickle=False)
        assert a.dtype==np.int64 and a.shape==(count,) and (np.diff(a)>0).all() and a.min()>=0 and a.max()<bound,n
        if identity: assert np.array_equal(a,np.arange(count,dtype=np.int64)),n
        return a
    kt=index(old.GEO+'train_source_cache_indices.npy',455784,455255)
    kv=index(old.GEO+'valid_source_cache_indices.npy',11994,11994,True)
    kf=index(old.GEO+'full_source_cache_indices.npy',467249,467221)
    kq=index(old.GEO+'test_source_cache_indices.npy',47073,47073,True)
    X=pd.concat([frames['train'].iloc[kt],frames['valid'].iloc[kv]],ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq=frames['test'].iloc[kq].reset_index(drop=True); del frames
    geo,geoq=[pd.read_pickle(checked(old.GEO+n)) for n in ('full.pkl','test.pkl')]
    before={n:spatial.frame_sha(v) for n,v in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}
    assert before==p71['input_frame_sha256']==area['input_frame_sha256']
    manifest=small('outputs/round39/geography/final/manifest.json')
    newquery=pd.read_pickle(checked(manifest['features']['query']['path']))
    assert spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256']
    assert newquery.index.equals(Xq.index) and newquery.dtypes.equals(geoq.dtypes) and list(newquery)==list(geoq)
    with checked(old.REFERENCE).open('rb') as f:reference=pickle.load(f)
    with checked('outputs/round39/trial/artifacts/spatial_state.pkl').open('rb') as f:state=pickle.load(f)
    summary=small('outputs/round39/trial/artifacts/state_summary.json')
    assert state['state_summary']==summary and summary['fit_rows']==len(X)==467221
    assert not summary['query_used_in_build'] and summary['fit_X_sha256']==before['X']
    assert summary['fit_geo_sha256']==manifest['features']['fit']['frame_sha256'] and summary['parameters']==spatial.parameters()
    assert reference['fit_report']['fitting_rows']==467221
    assert summary['existing_reference_residual_sha256']==exp.original.value_sha(reference['reference_residual'],'<f8')
    oldunit=vec(old.OLD_UNIT); raw=vec(old.OLD_RAW,True)
    native=vec('outputs/round40/trial/model/predictions.npy',True)
    unit0=vec('outputs/round40/trial/unit_spatial.npy')
    nn0=vec('outputs/round71/final/nn_spatial.npy'); log=vec('outputs/round46/trial/recency_log_spatial.npy')
    v19=vec('outputs/round71/final/predictions.npy')
    controls={'v19':f48.exact(np.median(np.column_stack([nn0,log,unit0]),axis=1),v19,'v19 saved composition')}
    sample_name=rel(old.SAMPLE)
    sample=pd.read_csv(checked(sample_name),encoding='utf-8-sig',dtype={old.ID:str},keep_default_na=False)
    ids=sample[old.ID].tolist(); assert len(ids)==len(set(ids))==47073
    f48.csv_check(ROOT/'outputs/submission_v19_trial.csv',ids,old.rounded(v19))
    write(OUT/'input_contract.json',{'plan_sha256':plan_sha,'source_sha256':sources,'input_sha256':inputs,
        'protected_before':protected,'selector_sha256_before':selector_sha,'labels_loaded':False,
        'area_result_sha256':inputs[area_path],'new_fits_in_this_composer':0})
    calls=[]
    def branch(folder,B):
        folder.mkdir(parents=True,exist_ok=False)
        p,diag,rr=exp.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
        a,detail,sr=spatial.predict(state,Xq,newquery,B,p,diag)
        fallback=~detail['applied'].to_numpy(dtype=bool)
        assert a[fallback].dtype==p[fallback].dtype and a[fallback].shape==p[fallback].shape and a[fallback].tobytes()==p[fallback].tobytes(), 'spatial fallback'
        for n,v in (('base',B),('reference',p),('spatial',a)):f48.array(folder/(n+'.npy'),v)
        for n,v in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
            with (folder/(n+'.pkl')).open('xb') as f:pickle.dump(v,f,protocol=5)
        calls.append({'folder':rel(folder),'reference_report':rr,'spatial_report':sr,'fallback_bitwise_equal':True})
        return p,a
    B0=.75*(.5*raw+.5*native)+.25*oldunit
    controls['R40_base']=f48.exact(B0,vec('outputs/round40/trial/base_unit.npy'),'R40 base')
    p0,u0=branch(OUT/'control_unit',B0)
    controls['R40_reference']=f48.exact(p0,vec('outputs/round40/trial/reference_unit.npy'),'R40 reference')
    controls['R40_spatial']=f48.exact(u0,unit0,'R40 spatial')
    _,unit1=branch(OUT/'new_unit',.75*native+.25*oldunit)
    weighted=vec(area['prediction_path']); assert sha(ROOT/area['prediction_path'])==area['prediction_sha256']
    prior=vec('outputs/round63/final/model/predictions_twd.npy')
    _,nn1=branch(OUT/'weighted_area_nn',.75*(.5*weighted+.5*prior)+.25*oldunit)
    candidates={76:np.median(np.column_stack([nn0,log,unit1]),axis=1),77:np.median(np.column_stack([nn1,log,unit1]),axis=1)}
    proofs={}
    for r,pred in candidates.items():
        folder=ROOT/f'outputs/round{r}/final';folder.mkdir(exist_ok=False)
        assert pred.dtype==np.float64 and pred.shape==(47073,) and np.isfinite(pred).all() and (pred>0).all()
        f48.array(folder/'predictions.npy',pred)
        f48.exact(pred,np.load(folder/'predictions.npy',allow_pickle=False),'saved candidate')
        prices=old.rounded(pred); assert prices.dtype==np.int64 and (prices>0).all()
        payload={'headers':[old.ID,old.TARGET],'ids':ids,'prices':prices.tolist(),'round':r,'rows':47073,
            'prediction_path':rel(folder/'predictions.npy'),'prediction_sha256':sha(folder/'predictions.npy'),
            'sample_ID_path':sample_name,'sample_ID_sha256':inputs[sample_name],
            'rounding':'maximum(rint(unrounded candidate TWD),1).astype(int64), once'}
        write(folder/'export_payload.json',payload)
        proof={k:v for k,v in payload.items() if k not in ('ids','prices','headers')}
        proof.update(status='final_vector_and_export_payload_ready_no_CSV',fit_rows=467221,query_rows=47073,
            plan_sha256=plan_sha,export_payload_path=rel(folder/'export_payload.json'),export_payload_sha256=sha(folder/'export_payload.json'),
            prediction_min_TWD=float(pred.min()),prediction_max_TWD=float(pred.max()),
            changed_unrounded_rows_vs_v19=int(np.count_nonzero(pred-v19)),
            mean_absolute_prediction_change_vs_v19_TWD=float(np.abs(pred-v19).mean()),
            prediction_changes_are_not_errors=True,public_mae=None,private_mae=None,new_CSV=False,
            candidate=f'R{r}',intended_filename=f'outputs/submission_v{r-56}_trial.csv',
            historical_results=f'outputs/round{r}/results.json',historical_results_sha256=sha(ROOT/f'outputs/round{r}/results.json'),
            new_NN_fits_for_this_candidate=int(r==77),saved_area_model_reload_exact=bool(area['saved_model_reload_exact']),
            final_version_chosen=False)
        write(folder/'trial_release.json',proof);proofs[str(r)]=proof
    assert before=={n:spatial.frame_sha(v) for n,v in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}
    assert all(sha(ROOT/n)==h for n,h in sources.items()) and sha(PLAN)==plan_sha
    assert all(sha(ROOT/n)==h for n,h in protected.items()) and sha(selector)==selector_sha
    assert {rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}==set(protected)-{'runs/versions.json'}
    artifact_paths=[p for d in (OUT,ROOT/'outputs/round76/final',ROOT/'outputs/round77/final') for p in d.rglob('*') if p.is_file() and p.name!='execution.log']
    proof={'status':'final_vectors_and_export_payloads_ready_no_CSV','rounds':[76,77],'query_rows':47073,'fit_rows':467221,
        'plan_sha256':plan_sha,'source_sha256':sources,'input_sha256':inputs,'artifact_sha256':{rel(p):sha(p) for p in artifact_paths},
        'protected_before':protected,'protected_after':{n:sha(ROOT/n) for n in protected},
        'selector_sha256_before':selector_sha,'selector_sha256_after':sha(selector),'controls':controls,
        'old_v19_CSV_exact':True,'new_fits_in_this_composer':0,'separately_completed_new_NN_fits':1,
        'new_tree_fits':0,'models_loaded_in_this_composer':0,'tree_NN_model_predict_calls_in_this_composer':0,
        'reference_predict_calls':3,'spatial_predict_calls':3,'reference_or_spatial_refits':0,
        'canonical_frames_sha256':before,'call_reports':calls,'candidates':proofs,'new_CSV':False,
        'selector_updated':False,'registry_updated':False,'test_labels_loaded':False,'uploaded':False,
        'elapsed_seconds':time.monotonic()-tick,'completed_at_utc':old.stamp()}
    write(OUT/'trial_release.json',proof)
    print(json.dumps({'status':proof['status'],'elapsed_seconds':proof['elapsed_seconds'],'rows':47073,'controls':controls}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true',required=True);p.add_argument('--plan-sha',required=True);a=p.parse_args()
    assert sha(PLAN)==a.plan_sha and not OUT.exists()
    assert not any((ROOT/f'outputs/round{r}/final').exists() for r in (76,77))
    OUT.mkdir(parents=True,exist_ok=False);started=old.stamp();tick=time.monotonic();code=1;failure=None
    write(OUT/'started.json',{'command':sys.argv,'started_at_utc':started,'plan_sha256':a.plan_sha,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    with (OUT/'execution.log').open('x',encoding='utf8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:run(a.plan_sha);code=0
            except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'started_at_utc':started,'completed_at_utc':old.stamp(),
        'source_sha256':sha(Path(__file__)),'plan_sha256':a.plan_sha,'automatic_retry':False,'log_path':rel(OUT/'execution.log'),'log_sha256':sha(OUT/'execution.log')}
    write(OUT/'execution.json',event);print(json.dumps(event),flush=True);return code

if __name__=='__main__':raise SystemExit(main())
