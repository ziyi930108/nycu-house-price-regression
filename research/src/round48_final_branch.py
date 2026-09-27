"""R48 one final area-linked NN fit; save two medians and selected R49 v11 trial CSV."""
from __future__ import annotations
import argparse,contextlib,json,os,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round46_trial_release as inherited
import round48_nn_experiment as experiment

old=inherited.old
ROOT=old.ROOT
OUT=ROOT/'outputs/round48/final'
DOC=ROOT/'reports/round48_final_plan.md'
CSV=ROOT/'outputs/submission_v11_trial.csv'
FINAL=inherited.FINAL
need,sha,rel,read,write,spatial=inherited.need,inherited.sha,inherited.rel,inherited.read,inherited.write,inherited.spatial
array,dump,load,exact,csv_check=inherited.array,inherited.dump,inherited.load,inherited.exact,inherited.csv_check
V10='outputs/round46/trial/predictions.npy'
RECENCY='outputs/round46/trial/recency_log_spatial.npy'
AUTHORITIES={'outputs/round48/results.json':'20d08c8442985054e9065ef2f280e3c9ff3febe054a059f53c17120bcb90f0b9',
 'outputs/round48/plan.json':'ece3c43bae1d2f4c5f471d623a0a1082dcd8152dc3b0d20e91e7bc10c1a184af',
 'outputs/round46/trial/trial_release.json':'23926d1145d920dac12b6c87b4b43299aded6aa81b524fb3007ece93042a6b2f',
 'outputs/round49/results.json':'0e80d21ef7ebea8489da605d733bf4efbcdd61f4eddf6e05d8558d3c836a087a'}


def sources():
    return old.merge(inherited.sources(),experiment.sources(),{rel(Path(inherited.__file__)):sha(Path(inherited.__file__)),rel(Path(__file__)):sha(Path(__file__))})


def capture():
    inputs,_,protected,p34,p37,manifest=inherited.capture()
    docs={}
    def bind(name,digest):
        need(name not in inputs or inputs[name]==digest,'Conflicting final input: '+name)
        old.verify({name:digest});inputs[name]=digest
    for name,digest in AUTHORITIES.items():
        bind(name,digest);docs[name]=old.locked(ROOT/name)
        receipt=rel((ROOT/name).with_suffix('.sha256'));bind(receipt,sha(ROOT/receipt))
    result,plan,trial,selection=[docs[n] for n in AUTHORITIES]
    need(selection['round']==49 and selection['status']=='three_periods_completed_no_adoption_decision'
         and selection['checked_stages']==list(experiment.STAGES)
         and selection['formula']=='median(saved_R48_nn_area_spatial_TWD,saved_R46_recency_log_spatial_TWD,saved_R40_unit_spatial_TWD)',
         'R49 fixed three-period selection authority differs')
    need(result['round']==48 and result['status']=='three_periods_completed_no_adoption_decision'
         and result['checked_stages']==list(experiment.STAGES) and result['plan_sha256']==AUTHORITIES['outputs/round48/plan.json']
         and result['source_sha256']==plan['source_sha256']==experiment.sources(),'R48 completed method differs')
    need(trial['round']==46 and trial['status']=='experimental_trial_generated' and trial['query_rows']==47073,'v10 authority differs')
    for name in (V10,RECENCY):bind(name,trial['artifact_sha256'][name])
    bind(trial['path'],trial['sha256'])
    recipe='outputs/round47/development/model/training.json';bind(recipe,plan['input_sha256'][recipe])
    src=sources()
    for name,digest in src.items():bind(name,digest)
    bind(rel(DOC),sha(DOC))
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    for n,h in protected.items():bind(n,h)
    need(not (ROOT/'outputs/submission_v11_trial.csv').exists(),'v11 already exists; no implicit replacement')
    return inputs,src,protected,p34,p37,manifest,recipe


def prepare():
    started=time.monotonic();inputs,src,protected,p34,p37,manifest,recipe=capture()
    write(OUT/'input_contract.json',{'round':48,'stage':'final','source_sha256':src,'input_sha256':inputs,
      'protected_sha256':protected,'area_formula':experiment.AREA_FORMULA,'area_factor_formula':experiment.AREA_FACTOR_FORMULA,
      'epochs':30,'selected_median':'recency_log','CSV':rel(CSV),'encoder_fit_only':True,'sample_weight':'uniform'})
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
        raw=pd.read_pickle(ROOT/f'data_cache/{split}.pkl');info=read(ROOT/f'data_cache/{split}.json')
        need(len(raw)==count and raw.index.equals(pd.RangeIndex(count)) and list(raw)==info['columns']
             and info['source_sha256']==schema['splits'][split]['input_source_sha256'],'Target/X cache mapping differs')
        labels[split]=raw[old.TARGET].to_numpy(dtype='float64',copy=True);del raw
    y=np.concatenate([labels['train'][kt],labels['valid'][kv]])[kf];del labels,frames
    need(y.shape==(467221,) and np.isfinite(y).all() and (y>0).all(),'Invalid final fit labels')
    oldmeta=read(ROOT/(old.MODEL+'metadata.json'));target_sha=experiment.original.value_sha(y,'<f8')
    need(target_sha==oldmeta['fit_target_values_sha256']==p37['fit_target_values_sha256'],'Final target alignment differs')
    fi=old.index(old.MODEL+'fit_indices.npy',467221,467221,True)
    qi=old.index(old.MODEL+'query_indices.npy',47073,47073,True)
    geo,geoq=[pd.read_pickle(ROOT/(old.GEO+n)) for n in ('full.pkl','test.pkl')]
    before={n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))}
    need(before==p34['input_frame_sha256']==p37['input_frame_sha256'],'Final original X/geo alignment differs')
    newquery=pd.read_pickle(ROOT/manifest['features']['query']['path'])
    need(spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'] and newquery.index.equals(Xq.index)
         and newquery.dtypes.equals(geoq.dtypes) and list(newquery)==list(geoq),'New spatial query geography differs')
    state_spatial=load(FINAL+'spatial_state.pkl');summary=read(ROOT/(FINAL+'state_summary.json'))
    reference=load(old.REFERENCE)
    need(state_spatial['state_summary']==summary and summary['fit_rows']==len(X)
         and summary['fit_X_sha256']==before['X'] and summary['fit_geo_sha256']==manifest['features']['fit']['frame_sha256']
         and summary['query_used_in_build'] is False and summary['parameters']==spatial.parameters()
         and summary['existing_reference_residual_sha256']==experiment.original.value_sha(reference['reference_residual'],'<f8'),
         'R39 saved fit-only spatial state differs')
    raw_branch,log_branch,v7=[old.vector(FINAL+n) for n in ('raw_predictions.npy','log_predictions.npy','predictions.npy')]
    unit_branch,v9,v5,v10,recency=[old.vector(n) for n in (inherited.UNIT_BRANCH,inherited.V9,inherited.V5,V10,RECENCY)]
    controls={'v7':exact(.5*raw_branch+.5*log_branch,v7,'v7 saved branch mean'),
      'v9':exact(np.median(np.column_stack([raw_branch,log_branch,unit_branch]),axis=1),v9,'v9 saved median'),
      'v10':exact(np.median(np.column_stack([raw_branch,recency,unit_branch]),axis=1),v10,'v10 saved median')}
    sample=pd.read_csv(old.SAMPLE,encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str},keep_default_na=False)
    ids=sample[old.ID].tolist();need(len(ids)==47073 and sample[old.ID].is_unique and all(ids),'Sample IDs invalid')
    for name,values in (('v10',v10),('v9',v9),('v5',v5)):
        csv_check(ROOT/f'outputs/submission_{name}_trial.csv',ids,old.rounded(values))
    write(OUT/'control_replay.json',{'vectors':controls,'v10_v9_v5_CSV_exact':True,'no_tree_replay':True})
    array(OUT/'fit_indices.npy',fi);array(OUT/'query_indices.npy',qi)
    X95=experiment.geo_encoder.augment_features(X,geo);Xq95=experiment.geo_encoder.augment_features(Xq,geoq)
    cats=experiment.geo_encoder.categories(X,schema['categorical_columns'])
    state=experiment.encoder.fit_encoder(X95,cats);write(OUT/'encoder_state.json',state)
    restored=old.locked(OUT/'encoder_state.json');need(restored==state,'Encoder JSON roundtrip differs')
    tn,tc=experiment.encoder.encode(restored,X95);qn,qc=experiment.encoder.encode(restored,Xq95)
    cards=np.asarray(state['cardinalities'],dtype='int64');target=np.asarray(y/1e6,dtype='float32')
    need(state['fit_rows']==467221 and state['target_used_for_state'] is False and state['query_used_for_state'] is False
         and target.shape==(467221,) and np.isfinite(target).all() and (target>0).all(),'Final encoder/labels invalid')
    factors={name:np.asarray(experiment.area_helper.effective_area(frame)/100,dtype='float32')
             for name,frame in (('train_area_factor',X),('query_area_factor',Xq))}
    for name,a in factors.items():need(np.isfinite(a).all() and (a>0).all(),'Invalid final area factor')
    evidence={name+'_sha256':experiment.original.value_sha(a,'<f4') for name,a in factors.items()}
    evidence['area_factor_summary']={n.replace('_area_factor',''):{'rows':len(a),'min':float(a.min()),'max':float(a.max()),
       'mean':float(a.mean(dtype=np.float64))} for n,a in factors.items()}
    write(OUT/'area_preparation.json',evidence)
    data_path=OUT/'prepared.npz'
    with data_path.open('xb') as f:np.savez(f,train_num=tn,train_cat=tc,train_y_million=target,query_num=qn,query_cat=qc,cardinalities=cards,**factors)
    del tn,tc,target,qn,qc,X95,Xq95,y
    component,training,execution=experiment.train_subprocess(OUT,data_path,len(X),len(Xq),cards.tolist(),
       {'source_sha256':src},{'R47_training':recipe},evidence)
    B=.75*(.5*old.vector(old.OLD_RAW)+.5*component)+.25*old.vector(old.OLD_UNIT)
    p,diag,reference_report=old.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
    branch,detail,spatial_report=spatial.predict(state_spatial,Xq,newquery,B,p,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    need(branch[fallback].tobytes()==p[fallback].tobytes(),'NN spatial fallback differs')
    candidates={'old_log':np.median(np.column_stack([branch,log_branch,unit_branch]),axis=1),
                'recency_log':np.median(np.column_stack([branch,recency,unit_branch]),axis=1)}
    for name,a in [('base_nn',B),('reference_nn',p),('nn_area_spatial',branch)]+[('median_'+n,a) for n,a in candidates.items()]:
        need(a.dtype==np.float64 and a.shape==(47073,) and np.isfinite(a).all() and (a>0).all(),'Invalid final TWD vector')
        array(OUT/(name+'.npy'),a)
    dump(OUT/'reference_diagnostics.pkl',diag);dump(OUT/'spatial_diagnostics.pkl',detail)
    def scalar(v):
        if pd.isna(v):return None
        return v.item() if isinstance(v,np.generic) else v
    fields=['address_city','address_district','transaction_type','building_type','main_use','building_area','land_area','parking_area','parking_count']
    changes={};top12={}
    def row(i,values,diff,log_values):
        return {'position':int(i),'ID':ids[i],'pure_X':{c:scalar(Xq.iloc[i][c]) for c in fields},
          'candidate_TWD':float(values[i]),'v10_TWD':float(v10[i]),'v5_TWD':float(v5[i]),'v9_TWD':float(v9[i]),
          'new_NN_raw_spatial_TWD':float(branch[i]),'selected_log_spatial_TWD':float(log_values[i]),
          'old_log_spatial_TWD':float(log_branch[i]),'recency_log_spatial_TWD':float(recency[i]),
          'saved_unit_spatial_TWD':float(unit_branch[i]),'signed_prediction_change_TWD':float(diff[i]),'true_price_unknown':True}
    for choice,values in candidates.items():
        changes[choice]={};top12[choice]={}
        log_values=log_branch if choice=='old_log' else recency
        for label,baseline in (('v10',v10),('v5',v5)):
            diff=values-baseline
            changes[choice][label]={'changed_unrounded_rows':int(np.count_nonzero(diff)),
              'changed_rounded_rows':int(np.count_nonzero(old.rounded(values)!=old.rounded(baseline))),
              'mean_absolute_prediction_change_TWD':float(np.abs(diff).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(diff).max())}
            top12[choice][label]=[row(i,values,diff,log_values) for i in np.argsort(-np.abs(diff),kind='stable')[:12]]
    # Requested diagnostic only: retain every row whose building area rounds to 94,345 m2.
    area=Xq['building_area'].to_numpy(dtype=float);large=np.flatnonzero(np.isfinite(area)&(np.rint(area)==94345))
    large_rows={choice:[row(i,values,values-v10,log_branch if choice=='old_log' else recency) for i in large] for choice,values in candidates.items()}
    write(OUT/'prediction_changes.json',{'prediction_changes_are_not_MAE':True,'changes':changes,'top12_absolute_change':top12,
       'requested_94345_m2_row_selection':'round(building_area) == 94345; diagnostic only','requested_94345_m2_rows':large_rows,
       'test_labels_loaded':False,'no_row_adjustment_or_price_cap':True})
    need(before=={n:spatial.frame_sha(f) for n,f in (('X',X),('Xq',Xq),('geo',geo),('geoq',geoq))},'Original features mutated')
    need(spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'],'Spatial query mutated')
    need(sources()==src,'Source changed');old.verify(protected)
    values=candidates['recency_log'];prices=old.rounded(values);staged=OUT/CSV.name
    pd.DataFrame({old.ID:ids,old.TARGET:prices}).to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
    with staged.open('rb') as f:need(f.read(3)==b'\xef\xbb\xbf','CSV BOM absent')
    csv_check(staged,ids,prices)
    need(not CSV.exists(),'Refusing existing v11 CSV');os.link(staged,CSV)
    proof={'round':48,'status':'experimental_v11_R49_median_generated','fit_rows':467221,'query_rows':47073,
      'feature_count':95,'categorical_count':23,'source_sha256':src,'input_sha256':inputs,'fit_target_values_sha256':target_sha,
      'input_frame_sha256':before,'input_contract_sha256':sha(OUT/'input_contract.json'),'new_NN_fits':1,'new_tree_fits':0,
      'epochs':30,'sample_weight':'uniform','area_formula':experiment.AREA_FORMULA,'area_factor_formula':experiment.AREA_FACTOR_FORMULA,
      'area_evidence':evidence,'training_metadata_path':rel(OUT/'model/training.json'),'fit_seconds':training['fit_seconds'],
      'saved_model_reload_exact':training['saved_model_reload_exact'],'reload_rows':training['reload_rows'],
      'v7_v9_v10_controls':controls,'v10_v9_v5_CSV_exact':True,'reference_predict_calls':1,'spatial_predict_calls':1,
      'new_spatial_state_builds':0,'reference_report':reference_report,'spatial_report':spatial_report,'fallback_bitwise_equal':True,
      'test_labels_loaded':False,'new_CSV':True,'registry_updated':False,
      'path':rel(CSV),'sha256':sha(CSV),'selected_median':'recency_log','selected_prediction_path':rel(OUT/'median_recency_log.npy'),
      'selected_prediction_sha256':sha(OUT/'median_recency_log.npy'),'R49_result_sha256':AUTHORITIES['outputs/round49/results.json'],
      'rounding':'max(rint(median(new_NN_raw_spatial,recency_log_spatial,unit_spatial)),1), once',
      'sample_ID_order_verified':True,'positive_int64_roundtrip':True,'UTF8_BOM':True,
      'new_public_mae':None,'private_mae':None,'uploaded':False,'formal_adoption_decision':None,
      'prediction_changes_vs_controls':changes,'prediction_changes_are_not_errors':True,
      'artifact_sha256':{rel(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='execution.log'},
      'protected_before':protected,'protected_after':{n:sha(ROOT/n) for n in protected},
      'elapsed_seconds':time.monotonic()-started,'completed_at_utc':old.stamp()}
    need(proof['protected_before']==proof['protected_after'],'Old outputs changed')
    write(OUT/'trial_release.json',proof);print(json.dumps({'status':proof['status'],'fit_seconds':training['fit_seconds']}),flush=True)


def run():
    need(not OUT.exists(),'Existing final attempt; do not overwrite')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();stamp=old.stamp();code=1;error=None
    write(OUT/'run_claim.json',{'source_sha256':sha(Path(__file__)),'started_at_utc':stamp,'automatic_retry':False})
    with (OUT/'execution.log').open('x',encoding='utf8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:prepare();code=0
            except Exception:error=traceback.format_exc();print(error,flush=True)
    event={'round':48,'command':sys.argv,'started_at_utc':stamp,'completed_at_utc':old.stamp(),
       'elapsed_seconds':time.monotonic()-start,'exit_code':code,'failure':error,'automatic_retry':False,
       'source_sha256':sha(Path(__file__)),'log_path':rel(OUT/'execution.log'),'log_sha256':sha(OUT/'execution.log')}
    write(OUT/'execution.json',event);print(json.dumps(event,ensure_ascii=False),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();raise SystemExit(run())
