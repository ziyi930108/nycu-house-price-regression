"""R74: saved R54 area inputs with saved R63 fit-date weights; fixed v19 pipeline."""
from pathlib import Path
import argparse, json, pickle, subprocess, sys, time, traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old

ROOT,STAGES=old.ROOT,old.STAGES
OUT=ROOT/'outputs/round74'; DOCUMENT=ROOT/'reports/round74_plan.md'
TRAINER=ROOT/'src/round74_recency_weighted_area_train.py'
MATH_CHECK=ROOT/'reports/round74_trainer_math_check.json'
sha,read,write,spatial,rel=old.sha,old.locked,old.write,old.spatial,old.rel
PLAN_SHA='e36596652bdfe0ee7396f69ad0fbbe22ed5195d5d9741cfcd411ee37063b08e7'
TRAINER_SHA='f760d8a264c5c84c9a612a55ba77be4c0a3bafac655b1f21d0de99b2951f7b4d'
FORMULA='median(spatial(reference(.75*(.5*R74_recency_area_native_TWD+.5*R63_prior_native_TWD)+.25*old_unit_LGB_TWD)),R46_recency_log_spatial,R40_unit_spatial)'
CONTROLS=(('v19',71,'R71'),('v18',66,'R66'),('v17',63,'R63'),('v16',60,'R60'),
          ('v15',54,'R54'),('v13',53,'R53'),('v11',49,'R49'),('v12',50,'R50'),('R62',62,'R62'),('R65',65,'R65'))
AREA_KEYS={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_area_factor','query_area_factor'}
PRIOR_KEYS={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million','train_sample_weight'}
LOSS_OVERRIDES={
    'loss':'fit-recency weighted MAE in million TWD',
    'loss_formula':'mean(train_sample_weight * abs(unit_output * train_area_factor - train_y_million))',
    'extra_sample_weights':True,
    'sample_weight_normalization':'driver-supplied fit-global normalization; no batch renormalization',
}


def prepare(folder,path54,path63,yfit,meta54,meta63):
    with np.load(path54,allow_pickle=False) as z:
        assert set(z.files)==AREA_KEYS
        arrays={k:z[k].copy() for k in z.files}
    original={k:sha_array(v) for k,v in arrays.items()}
    with np.load(path63,allow_pickle=False) as z:
        assert set(z.files)==PRIOR_KEYS
        for k in ('train_cat','query_cat','cardinalities','train_y_million'):
            assert arrays[k].dtype==z[k].dtype and arrays[k].shape==z[k].shape and arrays[k].tobytes()==z[k].tobytes(),k
        arrays['train_sample_weight']=z['train_sample_weight'].copy()
    w=arrays['train_sample_weight'];nr=meta54['fit_rows'];nq=meta54['query_rows']
    assert w.dtype==np.float32 and w.shape==(nr,) and np.isfinite(w).all() and (w>0).all()
    assert sha_array(w)==meta63['train_sample_weight_sha256']
    assert arrays['train_y_million'].tobytes()==np.asarray(yfit/1e6,dtype=np.float32).tobytes()
    assert arrays['train_num'].shape==(nr,meta54['numeric_count']) and arrays['query_num'].shape==(nq,meta54['numeric_count'])
    assert arrays['cardinalities'].tolist()==meta54['cardinalities']==meta63['cardinalities']
    for k in ('train_area_factor','query_area_factor'):assert original[k]==meta54[k+'_sha256'],k
    path=folder/'prepared.npz'
    with path.open('xb') as f:np.savez(f,**arrays)
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==AREA_KEYS|{'train_sample_weight'}
        for k,v in arrays.items():assert z[k].dtype==v.dtype and z[k].shape==v.shape and z[k].tobytes()==v.tobytes(),k
    prep={'fit_rows':nr,'query_rows':nq,'numeric_count':meta54['numeric_count'],'prepared_path':rel(path),'prepared_sha256':sha(path),
          'R54_prepared_path':rel(path54),'R54_prepared_sha256':sha(path54),'R63_prepared_path':rel(path63),'R63_prepared_sha256':sha(path63),
          'original_eight_arrays_value_sha256':original,'original_eight_arrays_bitwise_unchanged':True,
          'exact_NPZ_keys':sorted(arrays),'train_sample_weight_sha256':sha_array(w),
          'train_sample_weight_summary':meta63['train_sample_weight_summary'],
          'weight_provenance':'Exact saved R63 float32 fit-date weights, canonical fit indices checked before preparation',
          'feature_encoder_refit':False,'sample_weights_recomputed':False,'sample_weights_renormalized':False,
          'query_labels_used':False,'query_dates_used_for_weights':False}
    write(folder/'preparation.json',prep);return path,prep


def sha_array(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()


def train(folder,path,baseline,prep):
    out=folder/'model';out.mkdir()
    cmd=[str(old.TORCH_PYTHON),'-u','-B','-X','utf8',str(TRAINER),'--data',str(path),'--out',str(out)]
    write(out/'fit_started.json',{'command':cmd,'started_at_utc':old.now(),'input_sha256':sha(path),'automatic_retry':False})
    tick=time.monotonic()
    with (out/'trainer.log').open('x',encoding='utf8') as f:done=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
    write(out/'execution.json',{'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-tick,'command':cmd,'automatic_retry':False})
    assert done.returncode==0,'Trainer failed; preserve existing outputs'
    m=json.loads((out/'training.json').read_text(encoding='utf8'))
    assert m['status']=='completed' and m['configuration']=={**baseline['configuration'],**LOSS_OVERRIDES}
    assert m['source_sha256']==TRAINER_SHA and m['architecture_source_sha256']==old.ARCHITECTURE_SHA and m['input_sha256']==sha(path)
    for k in ('torch','numpy','cuda','gpu','numeric_count','cardinalities','parameter_count','fit_rows','query_rows',
              'output_units','area_formula','area_factor_formula','train_area_factor_sha256','query_area_factor_sha256',
              'initial_training_median_unit_1e4_twd_per_m2','initial_head_bias'):
        assert m[k]==baseline[k],k
    assert m['numeric_count']==prep['numeric_count'] and m['fit_rows']==prep['fit_rows'] and m['query_rows']==prep['query_rows']
    assert m['saved_model_reload_exact'] and m['reload_rows']==prep['query_rows'] and m['prediction_units']=='TWD'
    assert not m['query_labels_available'] and not m['query_labels_used'] and not m['price_or_unit_clipping']
    assert m['extra_sample_weights'] and not m['sample_weights_batch_renormalized'] and not m['sample_weights_recomputed_in_trainer']
    assert not m['epoch_selection_uses_training_metrics'] and not m['bitwise_retraining_determinism_claimed']
    assert m['train_sample_weight_sha256']==prep['train_sample_weight_sha256']
    for k in ('rows','min','max'):assert m['train_sample_weight_summary'][k]==prep['train_sample_weight_summary'][k],k
    for k in ('mean','sum','sum_squares','effective_sample_size'):
        assert np.isclose(m['train_sample_weight_summary'][k],prep['train_sample_weight_summary'][k],rtol=1e-12,atol=1e-12),k
    assert [e['epoch'] for e in m['epochs']]==list(range(1,31))
    assert all(np.isfinite(e[k]) for e in m['epochs'] for k in ('training_mae_million_with_dropout','training_weighted_mae_million_with_dropout'))
    return old.vector(out/'predictions_twd.npy',prep['query_rows']),m


def run():
    tick_all=time.monotonic();inputs={}
    assert sha(DOCUMENT)==PLAN_SHA and sha(TRAINER)==TRAINER_SHA
    assert sha(old.TRAINER)==old.TRAINER_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
    def checked(name,digest):
        name=name.replace(chr(92),'/');p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT.resolve()) and p.is_file(),name
        assert name not in inputs or inputs[name]==digest,name
        if name not in inputs:assert sha(p)==digest,name
        inputs[name]=digest;return p
    def small(name):
        p=ROOT/name;v=read(p);checked(name,sha(p));checked(rel(p.with_suffix('.sha256')),sha(p.with_suffix('.sha256')));return v
    def artifact(a,name):
        mapping={k.replace(chr(92),'/'):v for k,v in a['artifact_sha256'].items()};return checked(name,mapping[name])
    def vector(name,digest,n):return old.vector(checked(name,digest),n)
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};assert len(protected)==20
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json');assert len(protected)==21
    p48=small('outputs/round48/plan.json');p52=small('outputs/round52/plan.json')
    records={r:small(f'outputs/round{r}/results.json') for r in (48,49,50,53,54,60,62,63,65,66,71)}
    assert all(r['checked_stages']==list(STAGES) for r in records.values())
    source={}
    for mapping in (records[71]['source_sha256'],records[54]['source_sha256'],records[63]['source_sha256'],old.previous.sources(),
                    {rel(p):sha(p) for p in (Path(__file__),TRAINER,old.TRAINER,old.ARCHITECTURE,Path(old.__file__))}):
        for name,digest in mapping.items():
            assert name not in source or source[name]==digest,name
            source[name]=digest
    for name,digest in source.items():checked(name,digest)
    checked(rel(DOCUMENT),PLAN_SHA);checked(rel(MATH_CHECK),sha(MATH_CHECK))
    canonical={name:p48['input_sha256'][name] for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT)}
    write(OUT/'plan.json',{'round':74,'candidate':'R74','formula':FORMULA,'source_sha256':source,'text_plan_sha256':PLAN_SHA,
          'trainer_math_check_sha256':sha(MATH_CHECK),'stage_order':list(STAGES),'split_contract':p48['split_contract'],
          'primary_control':'v19=R71','native_control':'R54 saved area model','loss_configuration_overrides':LOSS_OVERRIDES,
          'seed':42,'epochs':30,'batch_size':2048,'planned_new_NN_fits':3,'three_periods_regardless_of_scores':True,
          'half_life_days':365,'half_life_or_weight_search':False,'saved_weights_recomputed':False,
          'numeric_widths':{'development':599,'confirmation':599,'auxiliary':590},'new_gate':None,
          'no_CSV_or_final_fit_in_this_run':True,'fresh_blind_validation':False,'bitwise_retraining_determinism_claimed':False,
          'input_sha256_before_data':dict(inputs),'canonical_input_sha256':canonical,'protected_before':protected})
    for name,digest in canonical.items():checked(name,digest)
    Xall,yall=old.previous.load_training_rows(p48)
    result={'round':74,'candidate':'R74','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},'new_NN_fits':0,
            'new_submission_generated':False,'fresh_blind_validation':False,'source_sha256':source,'plan_sha256':PLAN_SHA,
            'machine_plan_sha256':sha(OUT/'plan.json'),'protected_before':protected,'primary_comparison_direction':'R74 minus v19'}
    for s in STAGES:
        tick=time.monotonic();folder=OUT/s;folder.mkdir();b=p48['bindings'][s];bs=p52['bindings'][s]
        a48,a54,a63,a71=(records[r]['stages'][s] for r in (48,54,63,71));ind={}
        for kind in ('fit','query'):
            name=f'outputs/round54/{s}/{kind}_indices.npy';v=np.load(artifact(a54,name),allow_pickle=False);c=p48['split_contract'][s]
            assert v.dtype==np.int64 and v.shape==(c[kind+'_rows'],) and (np.diff(v)>0).all() and v.min()>=0 and v.max()<len(Xall)
            assert old.original.value_sha(v,'<i8')==c[kind+'_indices_content_sha256']
            for r in (48,63,71):assert np.array_equal(v,np.load(artifact(records[r]['stages'][s],f'outputs/round{r}/{s}/{kind}_indices.npy'),allow_pickle=False))
            ind[kind]=v;old.previous.save_array(folder/f'{kind}_indices.npy',v)
        fi,qi=ind['fit'],ind['query'];assert not np.intersect1d(fi,qi).size
        X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);n=len(qi)
        frames={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        for k,v in frames.items():assert all(a['original_frame_sha256'][k]==v for a in (a48,a63,a71)),k
        metas={r:json.loads(artifact(records[r]['stages'][s],f'outputs/round{r}/{s}/model/training.json').read_text(encoding='utf8')) for r in (54,63)}
        meta54,meta63=metas[54],metas[63]
        assert meta54['status']==meta63['status']=='completed' and meta54['source_sha256']==old.TRAINER_SHA
        assert meta54['numeric_count']=={'development':599,'confirmation':599,'auxiliary':590}[s]
        assert not meta54['extra_sample_weights'] and meta63['extra_sample_weights']
        assert meta54['configuration']['seed']==meta63['configuration']['seed']==42
        assert meta54['fit_rows']==meta63['fit_rows']==len(fi) and meta54['query_rows']==meta63['query_rows']==n
        paths={r:artifact(records[r]['stages'][s],f'outputs/round{r}/{s}/prepared.npz') for r in (54,63)}
        for r in (54,63):assert sha(paths[r])==metas[r]['input_sha256']
        weight_prep=read(artifact(a63,f'outputs/round63/{s}/preparation.json'))
        assert weight_prep['prepared_sha256']==meta63['input_sha256']
        assert weight_prep['weights_use_fit_X_only'] and not weight_prep['weights_use_targets'] and not weight_prep['weights_use_query_dates']
        assert not weight_prep['query_labels_used'] and not weight_prep['encoder_refit'] and not weight_prep['prior_refit']
        assert weight_prep['train_sample_weight_sha256']==meta63['train_sample_weight_sha256']
        prepared,prep=prepare(folder,paths[54],paths[63],yall[fi],meta54,meta63)
        new_native,training=train(folder,prepared,meta54,prep)
        area=old.vector(artifact(a54,f'outputs/round54/{s}/model/predictions_twd.npy'),n)
        prior=old.vector(artifact(a63,f'outputs/round63/{s}/model/predictions_twd.npy'),n)
        mean=.5*new_native+.5*prior;unit=vector(b['old_unit'],p48['input_sha256'][b['old_unit']],n);B=.75*mean+.25*unit
        with checked(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f:reference=pickle.load(f)
        with checked(b['state'],p48['input_sha256'][b['state']]).open('rb') as f:state=pickle.load(f)
        assert state['state_summary']['fit_X_sha256']==frames['fit_X'] and not state['state_summary']['query_used_in_build']
        assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
        geo=pd.read_pickle(checked(b['query_geo'],p48['input_sha256'][b['query_geo']]))
        assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
        p,diag,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
        branch,detail,sr=spatial.predict(state,Xq,geo,B,p,diag)
        fallback=~detail['applied'].to_numpy(dtype=bool);old.exact(branch[fallback],p[fallback],'spatial fallback')
        log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n);ub=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
        candidate=np.median(np.column_stack([branch,log,ub]),axis=1)
        assert candidate.dtype==np.float64 and candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
        preds={'R74':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
        for label,r,key in CONTROLS:preds[label]=vector(f'outputs/round{r}/{s}/predictions.npy',records[r]['stages'][s]['prediction_sha256'],n)
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={k:score(v) for k,v in preds.items()}
        for label,r,key in (*CONTROLS,('v5',53,'v5')):assert metrics[label]==records[r]['stages'][s]['metrics'][key],label
        native={'R74_area':float(np.abs(new_native-yall[qi]).mean()),'R54_area':float(np.abs(area-yall[qi]).mean())}
        assert native['R54_area']==a54['native_MAE_diagnostic']['R54']
        for name,v in (('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)):old.previous.save_array(folder/f'{name}.npy',v)
        old.exact(candidate,old.vector(folder/'predictions.npy',n),'Saved complete candidate')
        for name,v in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
            with (folder/f'{name}.pkl').open('xb') as f:pickle.dump(v,f,protocol=5)
        assert frames=={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,
              'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R74']) for k in preds if k!='R74'},
              'native_MAE_diagnostic':native,'native_area_MAE_change_TWD':native['R74_area']-native['R54_area'],
              'original_frame_sha256':frames,'preparation':prep,'new_NN_fits':1,'fit_seconds':training['fit_seconds'],
              'numeric_count':training['numeric_count'],'parameter_count':training['parameter_count'],
              'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],'reference_predict_calls':1,'spatial_predict_calls':1,
              'controls_full_metrics_exact':True,'full_query_reload_exact':True,'query_labels_used_only_by_scorer':True,
              'old_NN_retrained':False,'encoder_prior_weights_refit':False,'reference_report':rr,'spatial_report':sr,
              'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
              'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
        write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s);result['new_NN_fits']+=1
        print(json.dumps({'stage':s,'fit_seconds':training['fit_seconds'],'vs_v19':{k:item['differences']['v19'][k]['relative_change_pct'] for k in ('overall','weighted')},'native_area_delta_TWD':item['native_area_MAE_change_TWD']}),flush=True)
    assert result['new_NN_fits']==3 and all(sha(ROOT/n)==h for n,h in source.items()) and sha(DOCUMENT)==PLAN_SHA
    assert set(protected)-{'runs/versions.json'}=={rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,elapsed_seconds=time.monotonic()-tick_all,
                  protected_after={n:sha(ROOT/n) for n in protected})
    assert result['protected_before']==result['protected_after'];write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args()
    assert not OUT.exists(),'Existing output: no overwrite';OUT.mkdir();tick=time.monotonic();code=0;failure=None;started=old.now()
    write(OUT/'started.json',{'round':74,'started_at_utc':started,'command':sys.argv,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    try:run()
    except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    write(OUT/'execution.json',{'round':74,'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
          'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
