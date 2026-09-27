"""R55 one final ramp-area NN fit; fixed half v13 / half R54-complete trial."""
from __future__ import annotations
import argparse,contextlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round48_final_branch as final48
import round54_nn_experiment as experiment

ROOT=experiment.ROOT;OUT=ROOT/'outputs/round55/final';CSV=ROOT/'outputs/submission_v14_trial.csv'
DOCUMENT=ROOT/'reports/round55_plan.md'
R55_RESULT_SHA='b5d0148ff25345f0a32cf5b95a3bb24274b39fe2770f231ef69589e3fb52196a'
PRIOR_FINAL='outputs/round53/final/'
old=final48.old;spatial=final48.spatial
need,sha,rel,read,write=final48.need,final48.sha,final48.rel,final48.read,final48.write
array,dump,load,exact,csv_check=final48.array,final48.dump,final48.load,final48.exact,final48.csv_check
FINAL=final48.FINAL;SAVED='outputs/round48/final/'
FORMULA='0.5 * saved_v13_unrounded_TWD + 0.5 * R54_final_unrounded_TWD'
AUTHORITIES={
 'outputs/round48/final/trial_release.json':'0a3d2854bb66188db1e79e8a9b300d7c6aafb98dd57e7796a442e932d0f19a1f',
 'outputs/round54/results.json':'05023065ff34c40d9d6514b77c01d4b9278b3fa34a29665975448976aed9950a',
 'outputs/round53/final/trial_release.json':'89bf546437d5e11eead362241677c2c84eff3e210a353c177cf1fd6d42701092',
 'outputs/round50/trial/release.json':'2b5718f4c01bdb6465fb155cde4875e1b2752b4ce598e5d5f6bc7996d2b0f537',
 'outputs/round55/results.json':R55_RESULT_SHA}


def capture():
    need(not CSV.exists(),'Existing v14 CSV; no overwrite')
    inputs={};docs={}
    def bind(name,h):
        name=Path(name).as_posix()
        need(name not in inputs or inputs[name]==h,'Conflicting input identity')
        old.verify({name:h});inputs[name]=h
    for n,h in AUTHORITIES.items():
        bind(n,h);docs[n]=read(ROOT/n)
    p48,r54,p53,p50,r55=[docs[n] for n in AUTHORITIES]
    need(p48['fit_rows']==p53['fit_rows']==467221 and p48['query_rows']==p53['query_rows']==47073
         and p48['saved_model_reload_exact'] is True and p53['saved_model_reload_exact'] is True
         and p53['status']=='experimental_v13_trial_generated','Saved final authority differs')
    need(r54['checked_stages']==r55['checked_stages']==list(experiment.STAGES)
         and r54['status']=='three_periods_completed_no_adoption_decision'
         and r54['formula']==experiment.FORMULA
         and r55['status']=='three_periods_completed_no_adoption_decision'
         and r55['formula']=='0.5 * saved R53(v13) complete unrounded TWD + 0.5 * saved R54 complete unrounded TWD'
         and all(r55['stages'][stage]['status']=='completed' for stage in experiment.STAGES),
         'Three-period authority differs')
    need(r54['source_sha256']['src/round54_nn_experiment.py']==sha(Path(experiment.__file__))
         and sha(experiment.old.TRAINER)==experiment.old.TRAINER_SHA
         and sha(Path(experiment.encoder.__file__))==experiment.ENCODER_SHA,'Fixed R54 recipe changed')
    known=old.merge(p48['input_sha256'],p48['artifact_sha256'],p53['artifact_sha256'])
    names=[f'data_cache/{part}_features.pkl' for part in ('train','valid','test')]
    names += [f'data_cache/{part}.{ext}' for part in ('train','valid') for ext in ('pkl','json')]
    names += ['reports/feature_schema.json',rel(old.SAMPLE),old.REFERENCE,old.OLD_RAW,old.OLD_UNIT]
    names += [old.GEO+n for n in ('train_source_cache_indices.npy','valid_source_cache_indices.npy','full_source_cache_indices.npy','test_source_cache_indices.npy','full.pkl','test.pkl')]
    names += [old.MODEL+n for n in ('metadata.json','fit_indices.npy','query_indices.npy')]
    names += [FINAL+n for n in ('spatial_state.pkl','state_summary.json')]
    names += [final48.inherited.GEO_MANIFEST,'outputs/round39/geography/final/query.pkl',final48.RECENCY,
              final48.inherited.UNIT_BRANCH,final48.inherited.V5,final48.V10]
    names += [SAVED+n for n in ('prepared.npz','encoder_state.json','model/training.json',
                              'fit_indices.npy','query_indices.npy','nn_area_spatial.npy','median_recency_log.npy')]
    names += [PRIOR_FINAL+n for n in ('model/training.json','model/predictions_twd.npy','nn_mean_spatial.npy',
                                     'fit_indices.npy','query_indices.npy','predictions.npy')]
    for n in names:bind(n,known[n])
    for proof in (p48,p53,p50):bind(proof['path'],proof['sha256'])
    # The old v12 proof has no vector SHA; its fixed formula and CSV are checked below.
    bind('outputs/round50/trial/predictions.npy',sha(ROOT/'outputs/round50/trial/predictions.npy'))
    src=old.merge(p48['source_sha256'],p53['source_sha256'],r54['source_sha256'],r55['source_sha256'],
                  {rel(Path(__file__)):sha(Path(__file__))})
    old.verify(src);bind('outputs/round55/plan.json',r55['plan_sha256'])
    machine_plan=read(ROOT/'outputs/round55/plan.json')
    need(machine_plan['formula']==r55['formula'] and machine_plan['stage_order']==list(experiment.STAGES),'R55 machine plan differs')
    bind(rel(DOCUMENT),machine_plan['text_plan_sha256'])
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    manifest=read(ROOT/final48.inherited.GEO_MANIFEST)
    need(manifest['stage']=='final' and manifest['spatial_only'] is True,'Final geography authority differs')
    write(OUT/'input_contract.json',{'round':55,'formula':FORMULA,'R54_branch_formula':experiment.FORMULA,
       'source_sha256':src,'input_sha256':inputs,'protected_sha256':protected,'new_NN_fits':1,
       'reused_R52_prior_native':True,'query_labels_used':False,'base_encoder_refit':False,
       'numeric_ramp_state_fit_on_full_X':True,'epochs':30})
    return inputs,src,protected,p48,p53,manifest

def prepare():
    tick=time.monotonic();inputs,src,protected,p48,p53,manifest=capture()
    schema=read(ROOT/'reports/feature_schema.json')
    frames={s:pd.read_pickle(ROOT/f'data_cache/{s}_features.pkl') for s in ('train','valid','test')}
    for split,n in (('train',455784),('valid',11994),('test',47073)):
        f=frames[split]
        need(len(f)==n and f.index.equals(pd.RangeIndex(n)) and list(f)==schema['columns']
             and len(f.columns)==87 and old.TARGET not in f,'Original feature schema differs')
    kt=old.index(old.GEO+'train_source_cache_indices.npy',455784,455255)
    kv=old.index(old.GEO+'valid_source_cache_indices.npy',11994,11994,True)
    kf=old.index(old.GEO+'full_source_cache_indices.npy',467249,467221)
    kq=old.index(old.GEO+'test_source_cache_indices.npy',47073,47073,True)
    X=pd.concat([frames['train'].iloc[kt],frames['valid'].iloc[kv]],ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq=frames['test'].iloc[kq].reset_index(drop=True)
    labels={}
    for split,n in (('train',455784),('valid',11994)):
        raw=pd.read_pickle(ROOT/f'data_cache/{split}.pkl');info=read(ROOT/f'data_cache/{split}.json')
        need(len(raw)==n and raw.index.equals(pd.RangeIndex(n)) and list(raw)==info['columns']
             and info['source_sha256']==schema['splits'][split]['input_source_sha256'],'Target/X cache mapping differs')
        labels[split]=raw[old.TARGET].to_numpy(dtype='float64',copy=True);del raw
    y=np.concatenate([labels['train'][kt],labels['valid'][kv]])[kf];del labels,frames
    need(y.shape==(467221,) and np.isfinite(y).all() and (y>0).all(),'Invalid final fit labels')
    target_sha=experiment.old.original.value_sha(y,'<f8')
    need(target_sha==p48['fit_target_values_sha256']==p53['fit_target_values_sha256']==read(ROOT/(old.MODEL+'metadata.json'))['fit_target_values_sha256'],
         'Final target alignment differs')
    fi=old.index(old.MODEL+'fit_indices.npy',467221,467221,True)
    qi=old.index(old.MODEL+'query_indices.npy',47073,47073,True)
    for n,a in [('fit',fi),('query',qi)]:
        for prefix in (SAVED,PRIOR_FINAL):
            b=np.load(ROOT/(prefix+n+'_indices.npy'),allow_pickle=False)
            need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'Saved final indices differ')
        array(OUT/(n+'_indices.npy'),a)
    geo,geoq=[pd.read_pickle(ROOT/(old.GEO+n)) for n in ('full.pkl','test.pkl')]
    before={n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}
    need(before==p48['input_frame_sha256']==p53['input_frame_sha256'],'R48 final original X/geo alignment differs')
    newquery=pd.read_pickle(ROOT/manifest['features']['query']['path'])
    need(spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'] and newquery.index.equals(Xq.index)
         and newquery.dtypes.equals(geoq.dtypes) and list(newquery)==list(geoq),'Spatial query geography differs')
    state_spatial=load(FINAL+'spatial_state.pkl');summary=read(ROOT/(FINAL+'state_summary.json'));reference=load(old.REFERENCE)
    need(state_spatial['state_summary']==summary and summary['fit_rows']==len(X)
         and summary['fit_X_sha256']==before['X'] and summary['fit_geo_sha256']==manifest['features']['fit']['frame_sha256']
         and summary['query_used_in_build'] is False and summary['parameters']==spatial.parameters()
         and summary['existing_reference_residual_sha256']==experiment.old.original.value_sha(reference['reference_residual'],'<f8'),
         'R39 saved fit-only spatial state differs')
    encoder=read(ROOT/(SAVED+'encoder_state.json'));baseline=read(ROOT/(SAVED+'model/training.json'))
    need(encoder['fit_rows']==467221 and encoder['target_used_for_state'] is False and encoder['query_used_for_state'] is False
         and baseline['status']=='completed' and baseline['configuration']['seed']==42 and baseline['configuration']['epochs']==30
         and baseline['source_sha256']==experiment.old.TRAINER_SHA and baseline['architecture_source_sha256']==experiment.old.ARCHITECTURE_SHA
         and baseline['input_sha256']==inputs[SAVED+'prepared.npz'] and baseline['fit_rows']==467221 and baseline['query_rows']==47073
         and baseline['saved_model_reload_exact'] is True and baseline['reload_rows']==47073,'Saved final encoder/NN recipe differs')
    prior_meta=read(ROOT/(PRIOR_FINAL+'model/training.json'))
    need(prior_meta['status']=='completed' and prior_meta['fit_rows']==467221 and prior_meta['query_rows']==47073
         and prior_meta['saved_model_reload_exact'] is True and prior_meta['reload_rows']==47073
         and prior_meta['source_sha256']==p53['source_sha256']['src/round52_nn_train.py']
         and prior_meta['input_sha256']==p53['artifact_sha256'][PRIOR_FINAL+'prepared.npz']
         and prior_meta['prediction_units']=='TWD','Saved final prior NN differs')
    prior_native=old.vector(PRIOR_FINAL+'model/predictions_twd.npy')
    area_branch=old.vector(SAVED+'nn_area_spatial.npy')
    prior_mean_branch=old.vector(PRIOR_FINAL+'nn_mean_spatial.npy');v13=old.vector(PRIOR_FINAL+'predictions.npy')
    recency,unit,v11,v12,v10,v5=[old.vector(n) for n in (final48.RECENCY,final48.inherited.UNIT_BRANCH,
          SAVED+'median_recency_log.npy','outputs/round50/trial/predictions.npy',final48.V10,final48.inherited.V5)]
    controls={'v11':exact(np.median(np.column_stack([area_branch,recency,unit]),axis=1),v11,'v11 saved median'),
              'v12':exact(.5*v10+.5*v11,v12,'v12 saved half average'),
              'v13':exact(np.median(np.column_stack([prior_mean_branch,recency,unit]),axis=1),v13,'v13 saved median')}
    sample=pd.read_csv(old.SAMPLE,encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str},keep_default_na=False)
    ids=sample[old.ID].tolist();need(len(ids)==47073 and sample[old.ID].is_unique and all(ids),'Sample IDs invalid')
    for n,a in [('v13',v13),('v11',v11),('v12',v12),('v5',v5)]:csv_check(ROOT/f'outputs/submission_{n}_trial.csv',ids,old.rounded(a))
    write(OUT/'control_replay.json',{'vectors':controls,'v13_v11_v12_v5_CSV_exact':True,'no_old_model_replay':True})
    # Only the R54 ramp-area NN is newly fitted; the full-fit R52 native vector is reused.
    data_path,prep=experiment.prepare(OUT,ROOT/(SAVED+'prepared.npz'),y)
    area_native,training=experiment.train(OUT,data_path,baseline,prep)
    mean_native=.5*area_native+.5*prior_native
    B=.75*(.5*old.vector(old.OLD_RAW)+.5*mean_native)+.25*old.vector(old.OLD_UNIT)
    p,diag,reference_report=old.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state_spatial,Xq,newquery,B,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    need(branch[fallback].tobytes()==p[fallback].tobytes(),'Mean-NN spatial fallback differs')
    r54_final=np.median(np.column_stack([branch,recency,unit]),axis=1)
    candidate=.5*v13+.5*r54_final
    for n,a in [('nn_ramp_area_twd',area_native),('nn_mean_twd',mean_native),('base_nn',B),('reference_nn',p),
                ('nn_mean_spatial',branch),('r54_predictions',r54_final),('predictions',candidate)]:
        need(a.dtype==np.float64 and a.shape==(47073,) and np.isfinite(a).all() and (a>0).all(),'Invalid final TWD vector')
        array(OUT/(n+'.npy'),a)
    exact(np.load(OUT/'predictions.npy',allow_pickle=False),candidate,'Saved candidate vector')
    dump(OUT/'reference_diagnostics.pkl',diag);dump(OUT/'spatial_diagnostics.pkl',detail)
    def scalar(v):
        if pd.isna(v):return None
        return v.item() if isinstance(v,np.generic) else v
    fields=['address_city','address_district','transaction_type','building_type','main_use','building_area','land_area','parking_area','parking_count']
    def row(i):
        return {'position':int(i),'ID':ids[i],'pure_X':{k:scalar(Xq.iloc[i][k]) for k in fields},
          'candidate_TWD':float(candidate[i]),'v13_TWD':float(v13[i]),'R54_final_TWD':float(r54_final[i]),'v11_TWD':float(v11[i]),'v12_TWD':float(v12[i]),'v5_TWD':float(v5[i]),
          'ramp_area_NN_native_TWD':float(area_native[i]),'prior_NN_native_TWD':float(prior_native[i]),
          'mean_NN_spatial_TWD':float(branch[i]),'recency_log_spatial_TWD':float(recency[i]),'unit_spatial_TWD':float(unit[i]),
          'true_price_unknown':True}
    changes={};tops={}
    for n,a in [('v13',v13),('v11',v11),('v12',v12),('v5',v5)]:
        d=candidate-a;changes[n]={'changed_unrounded_rows':int(np.count_nonzero(d)),
          'changed_rounded_rows':int(np.count_nonzero(old.rounded(candidate)!=old.rounded(a))),
          'mean_absolute_prediction_change_TWD':float(np.abs(d).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(d).max())}
        tops[n]=[{**row(i),'signed_prediction_change_TWD':float(d[i])} for i in np.argsort(-np.abs(d),kind='stable')[:12]]
    areas=Xq['building_area'].to_numpy(dtype=float);large=np.flatnonzero(np.isfinite(areas)&(np.rint(areas)==94345))
    write(OUT/'prediction_changes.json',{'prediction_changes_are_not_errors':True,'test_labels_loaded':False,
      'changes':changes,'top12_absolute_change':tops,'requested_94345_m2_rows':[row(i) for i in large],
      'no_row_adjustment_or_price_cap':True})
    need(before=={n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))},'Original features mutated')
    old.verify(src);old.verify(protected)
    prices=old.rounded(candidate);staged=OUT/CSV.name
    pd.DataFrame({old.ID:ids,old.TARGET:prices}).to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
    with staged.open('rb') as f:need(f.read(3)==b'\xef\xbb\xbf','CSV BOM absent')
    csv_check(staged,ids,prices);need(not CSV.exists(),'Refusing existing v14 CSV');os.link(staged,CSV)
    proof={'round':55,'status':'experimental_v14_trial_generated','formula':FORMULA,'fit_rows':467221,'query_rows':47073,
      'source_sha256':src,'input_sha256':inputs,'input_frame_sha256':before,'fit_target_values_sha256':target_sha,
      'new_NN_fits':1,'reused_R52_prior_native_predictions':True,'base_encoder_refit':False,'numeric_ramp_state_fit_on_full_X':True,'epochs':30,'seed':42,
      'numeric_preparation':prep,'numeric_count':training['numeric_count'],'fit_seconds':training['fit_seconds'],'saved_model_reload_exact':training['saved_model_reload_exact'],
      'reload_rows':training['reload_rows'],'reference_predict_calls':1,'spatial_predict_calls':1,'new_spatial_state_builds':0,
      'reference_report':reference_report,'spatial_report':spatial_report,'fallback_bitwise_equal':True,
      'test_labels_loaded':False,'v13_v11_v12_v5_CSV_exact':True,'control_replay':controls,'new_CSV':True,
      'path':rel(CSV),'sha256':sha(CSV),'prediction_path':rel(OUT/'predictions.npy'),'prediction_sha256':sha(OUT/'predictions.npy'),
      'sample_ID_order_verified':True,'positive_int64_roundtrip':True,'UTF8_BOM':True,'rounding':'max(rint(0.5*v13+0.5*R54_final),1), once',
      'prediction_changes_vs_controls':changes,'prediction_changes_are_not_errors':True,
      'public_mae':None,'private_mae':None,'uploaded':False,'registry_updated':False,'formal_adoption_decision':None,
      'artifact_sha256':{rel(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='execution.log'},
      'protected_before':protected,'protected_after':{n:sha(ROOT/n) for n in protected},
      'elapsed_seconds':time.monotonic()-tick,'completed_at_utc':old.stamp()}
    need(proof['protected_before']==proof['protected_after'],'Old outputs changed')
    write(OUT/'trial_release.json',proof);print(json.dumps({'status':proof['status'],'fit_seconds':training['fit_seconds']}),flush=True)


def run():
    need(len(R55_RESULT_SHA)==64 and all(c in '0123456789abcdef' for c in R55_RESULT_SHA),'R55 result authority not fixed yet; do not run')
    need(not OUT.exists() and not CSV.exists(),'Existing final attempt or CSV; no overwrite')
    OUT.mkdir(parents=True,exist_ok=False);tick=time.monotonic();stamp=old.stamp();code=1;error=None
    write(OUT/'run_claim.json',{'round':55,'source_sha256':sha(Path(__file__)),'started_at_utc':stamp,'automatic_retry':False})
    with (OUT/'execution.log').open('x',encoding='utf8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:prepare();code=0
            except Exception:error=traceback.format_exc();print(error,flush=True)
    event={'round':55,'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':old.stamp(),
       'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':error,'automatic_retry':False,
       'source_sha256':sha(Path(__file__)),'log_path':rel(OUT/'execution.log'),'log_sha256':sha(OUT/'execution.log')}
    write(OUT/'execution.json',event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();raise SystemExit(run())
