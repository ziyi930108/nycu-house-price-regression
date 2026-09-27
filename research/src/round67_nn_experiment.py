"""R67 append fit-scaled log1p prior-cell support to R65 inputs; retain the R66 pipeline."""
from pathlib import Path
import json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round65_nn_experiment as baseline
import round67_prior_support as support

old = baseline.old
ROOT = old.ROOT; OUT = ROOT/'outputs/round67'; STAGES = old.STAGES
sha, read, write, spatial = old.sha, old.locked, old.write, old.spatial
rel = lambda p: Path(p).relative_to(ROOT).as_posix()
HELPER_SHA = '92ab5c561ba9f40b1dda49c4a76c98cbab64fc89d66b130d01edcffe49662721'
BASELINE_SOURCE_SHA = '3854f5170170f1d0bdfc3707c8e0e04832a430cdbcd780fb7753bef18c96898c'
PRIOR_SOURCE_SHA = '4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f'
FORMULA = 'median(spatial(reference(.75*(.5*R54_area_native_TWD+.5*R67_prior_support_native_TWD)+.25*old_unit_LGB_TWD)),R46_recency_log_spatial,R40_unit_spatial)'
CONTROLS = (('v18',66,'R66'),('v17',63,'R63'),('v16',60,'R60'),('v15',54,'R54'),
            ('v13',53,'R53'),('v11',49,'R49'),('v12',50,'R50'),('R62',62,'R62'),('R65',65,'R65'))


def prepare(folder, path, X, Xq, yfit, fi, folds, state, evidence, saved_meta, lineage):
    expected = {'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities',
                'train_price_factor_million','query_price_factor_million','train_sample_weight'}
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == expected
        arrays = {k:z[k].copy() for k in z.files}
    assert arrays['train_num'].shape == (len(X),145) and arrays['query_num'].shape == (len(Xq),145)
    assert arrays['train_num'].dtype == arrays['query_num'].dtype == np.float32
    assert arrays['train_y_million'].tobytes() == np.asarray(yfit/1e6,dtype=np.float32).tobytes()
    assert arrays['cardinalities'].tolist() == saved_meta['cardinalities']
    for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
        assert old.original.value_sha(arrays[key],'<f4') == saved_meta[key+'_sha256'], key
    counts_train, counts_query, evidence_summary = support.build_support(X,Xq,fi,folds,state,evidence)
    counts = {'train':counts_train,'query':counts_query}
    for split, rows in (('train',len(X)),('query',len(Xq))):
        v = counts[split]
        assert isinstance(v,np.ndarray) and v.dtype == np.int64 and v.shape == (rows,) and (v>=0).all()
        old.previous.save_array(folder/f'{split}_prior_support_count.npy',v)
    train_log = np.log1p(counts_train.astype(np.float64))
    query_log = np.log1p(counts_query.astype(np.float64))
    mean, std = float(train_log.mean(dtype=np.float64)), float(train_log.std(ddof=0,dtype=np.float64))
    assert np.isfinite(train_log).all() and np.isfinite(query_log).all() and np.isfinite(mean) and np.isfinite(std) and std>=0
    scale = 1.0 if std == 0 else std
    added = {'train':((train_log-mean)/scale).astype(np.float32),
             'query':((query_log-mean)/scale).astype(np.float32)}
    assert all(np.isfinite(v).all() for v in added.values())
    new = {**arrays,'train_num':np.column_stack([arrays['train_num'],added['train']]),
                   'query_num':np.column_stack([arrays['query_num'],added['query']])}
    prepared = folder/'prepared.npz'
    with prepared.open('xb') as f: np.savez(f,**new)
    with np.load(prepared,allow_pickle=False) as z:
        assert set(z.files) == expected
        for key in expected-{'train_num','query_num'}:
            assert z[key].dtype == arrays[key].dtype and z[key].shape == arrays[key].shape and z[key].tobytes() == arrays[key].tobytes(), key
        for split,rows in (('train',len(X)),('query',len(Xq))):
            key = split+'_num'
            assert z[key].dtype == np.float32 and z[key].shape == (rows,146)
            assert z[key][:,:145].tobytes() == arrays[key].tobytes() and z[key][:,145].tobytes() == added[split].tobytes()
    count_summary = {k:{'rows':len(v),'min':int(v.min()),'max':int(v.max()),'mean':float(v.mean(dtype=np.float64)),
                       'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),
                       'p99':float(np.quantile(v,.99)),'zero_rows':int((v==0).sum())} for k,v in counts.items()}
    prep = {'fit_rows':len(X),'query_rows':len(Xq),'numeric_count':146,'prepared_path':rel(prepared),'prepared_sha256':sha(prepared),
        'source_prepared_path':rel(path),'source_prepared_sha256':sha(path),'exact_NPZ_keys':sorted(expected),
        'support_input_lineage':lineage,'support_helper_sha256':HELPER_SHA,'support_evidence':evidence_summary,
        'raw_count_summary':count_summary,'raw_count_sha256':{k:old.original.value_sha(v,'<i8') for k,v in counts.items()},
        'added_feature':'standardized_log1p_selected_prior_cell_raw_support','numeric_column_position_zero_based':145,
        'fit_support_log1p_mean_float64':mean,'fit_support_log1p_std_ddof0_float64':std,'scale_float64':scale,'zero_std_uses_one':std==0,
        'query_scaling_uses_fit_only':True,'added_feature_sha256':{k:old.original.value_sha(v,'<f4') for k,v in added.items()},
        'original_145_prefix_sha256':{k:old.original.value_sha(arrays[k],'<f4') for k in ('train_num','query_num')},
        'model_numeric_values_sha256':{k:old.original.value_sha(new[k],'<f4') for k in ('train_num','query_num')},
        'other_seven_arrays_bitwise_unchanged':True,'original_145_prefix_bitwise_unchanged':True,
        'encoder_refit':False,'prior_refit':False,'sample_weights_rebuilt':False,'support_uses_query_labels':False,
        'fit_labels_used_for_saved_prior':True,'query_labels_used':False,'input_width_changes_parameter_count_and_initialization':True,
        'head_bias_formula_unchanged':True,'train_sample_weight_sha256':saved_meta['train_sample_weight_sha256'],
        'train_sample_weight_summary':saved_meta['train_sample_weight_summary']}
    write(folder/'preparation.json',prep)
    return prepared,prep


def run():
    start = time.monotonic(); inputs = {}
    assert sha(Path(support.__file__)) == HELPER_SHA and sha(Path(baseline.__file__)) == BASELINE_SOURCE_SHA
    assert sha(baseline.TRAINER) == baseline.TRAINER_SHA and sha(baseline.PARENT) == baseline.PARENT_SHA
    assert sha(old.TRAINER) == old.TRAINER_SHA and sha(old.ARCHITECTURE) == old.ARCHITECTURE_SHA
    def checked(name,digest):
        name = name.replace('\\','/'); path = ROOT/name
        assert sha(path) == digest, name
        assert name not in inputs or inputs[name] == digest, name
        inputs[name] = digest; return path
    def small(name):
        path=ROOT/name; value=read(path); inputs[name]=sha(path)
        inputs[rel(path.with_suffix('.sha256'))]=sha(path.with_suffix('.sha256')); return value
    def artifact(record,name):
        mapping={k.replace('\\','/'):v for k,v in record['artifact_sha256'].items()}
        return checked(name,mapping[name])
    def vector(name,digest,n): return old.vector(checked(name,digest),n)
    protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    p48=small('outputs/round48/plan.json'); p52=small('outputs/round52/plan.json')
    records={r:small(f'outputs/round{r}/results.json') for r in (48,49,50,53,54,60,62,63,65,66)}
    assert all(r['checked_stages'] == list(STAGES) for r in records.values())
    prior_name='src/round60_recency_price_prior.py'
    assert records[60]['source_sha256'][prior_name] == PRIOR_SOURCE_SHA
    checked(prior_name,PRIOR_SOURCE_SHA)
    source=old.previous.merge(old.previous.sources(),{rel(p):sha(p) for p in
        (Path(__file__),Path(old.__file__),Path(baseline.__file__),Path(support.__file__),baseline.TRAINER,baseline.PARENT,
         old.TRAINER,old.ARCHITECTURE,ROOT/prior_name,ROOT/'src/round52_price_prior.py')})
    plan_sha=sha(ROOT/'reports/round67_plan.md')
    write(OUT/'plan.json',{'round':67,'candidate':'R67','formula':FORMULA,'source_sha256':source,'text_plan_sha256':plan_sha,
        'stage_order':list(STAGES),'split_contract':p48['split_contract'],'primary_control':'v18=R66','native_control':'R65',
        'numeric_representation':'R65 original 145 columns plus fit-scaled log1p selected prior-cell raw support',
        'numeric_count':146,'configuration_unchanged_from_R65':True,'other_seven_arrays_bitwise_unchanged':True,
        'scaling':'float64 log1p; fit mean/std(ddof=0); exact std==0 uses 1; appended column float32',
        'query_labels_in_trainer':False,'encoder_prior_weights_rebuilt':False,'seed':42,'epochs':30,
        'input_width_changes_initialization':True,'three_periods_regardless_of_scores':True,'new_gate':None})
    Xall,yall=old.previous.load_training_rows(p48)
    result={'round':67,'candidate':'R67','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},
        'new_NN_fits':0,'new_submission_generated':False,'fresh_blind_validation':False,'protected_before':protected,
        'source_sha256':source,'plan_sha256':plan_sha}
    for s in STAGES:
        tick=time.monotonic(); folder=OUT/s; folder.mkdir(); b=p48['bindings'][s]; bs=p52['bindings'][s]
        a48,a54,a60,a65,a66=(records[r]['stages'][s] for r in (48,54,60,65,66)); indices={}
        for kind in ('fit','query'):
            name=f'outputs/round48/{s}/{kind}_indices.npy'; indices[kind]=np.load(artifact(a48,name),allow_pickle=False)
            assert old.original.value_sha(indices[kind],'<i8') == p48['split_contract'][s][kind+'_indices_content_sha256']
            for r,a in ((54,a54),(60,a60),(65,a65),(66,a66)):
                name=f'outputs/round{r}/{s}/{kind}_indices.npy'
                assert np.array_equal(indices[kind],np.load(artifact(a,name),allow_pickle=False))
            old.previous.save_array(folder/f'{kind}_indices.npy',indices[kind])
        fi,qi=indices['fit'],indices['query']; X=Xall.iloc[fi].reset_index(drop=True); Xq=Xall.iloc[qi].reset_index(drop=True); n=len(qi)
        assert not np.intersect1d(fi,qi).size
        frames={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        for key,value in frames.items():
            assert all(a['original_frame_sha256'][key] == value for a in (a48,a60,a65,a66)), key
        name=f'outputs/round65/{s}/model/training.json'; saved_meta=json.loads(artifact(a65,name).read_text(encoding='utf8'))
        assert saved_meta['status']=='completed' and saved_meta['source_sha256']==baseline.TRAINER_SHA
        assert saved_meta['configuration']['seed']==42 and saved_meta['configuration']['epochs']==30 and saved_meta['numeric_count']==145
        saved_prep=read(artifact(a65,f'outputs/round65/{s}/preparation.json'))
        assert not saved_prep['query_labels_used'] and not saved_prep['prior_refit'] and not saved_prep['encoder_refit'] and not saved_prep['sample_weights_rebuilt']
        path=artifact(a65,f'outputs/round65/{s}/prepared.npz')
        assert sha(path)==saved_meta['input_sha256']==saved_prep['prepared_sha256']
        evidence_path=artifact(a60,f'outputs/round60/{s}/prior_preparation.json'); prior_prep=read(evidence_path)
        assert prior_prep['fit_labels_used_for_prior'] and not prior_prep['query_labels_used'] and not prior_prep['encoder_refit']
        state_path=artifact(a60,f'outputs/round60/{s}/price_prior_state.json')
        assert sha(state_path)==prior_prep['prior_state_sha256']
        prior_state=read(state_path); fold_path=artifact(a60,f'outputs/round60/{s}/prior_fold_ids.npy'); folds=np.load(fold_path,allow_pickle=False)
        lineage={rel(p):sha(p) for p in (evidence_path,state_path,fold_path)}
        prepared,prep=prepare(folder,path,X,Xq,yall[fi],fi,folds,prior_state,prior_prep['prior_evidence'],saved_meta,lineage)
        assert prep['support_evidence']['query_selected_level_rows']==prior_prep['query_prior_level_counts']
        new_native,training=baseline.train(folder,prepared,saved_meta,prep)
        area=old.vector(artifact(a54,f'outputs/round54/{s}/model/predictions_twd.npy'),n)
        prior_control=old.vector(artifact(a65,f'outputs/round65/{s}/model/predictions_twd.npy'),n)
        mean=.5*area+.5*new_native; unit=vector(b['old_unit'],p48['input_sha256'][b['old_unit']],n); B=.75*mean+.25*unit
        with checked(b['reference'],p48['input_sha256'][b['reference']]).open('rb') as f: reference=pickle.load(f)
        with checked(b['state'],p48['input_sha256'][b['state']]).open('rb') as f: state=pickle.load(f)
        assert state['state_summary']['fit_X_sha256']==frames['fit_X'] and not state['state_summary']['query_used_in_build']
        assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
        geo=pd.read_pickle(checked(b['query_geo'],p48['input_sha256'][b['query_geo']]))
        assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
        p,diag,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
        branch,detail,sr=spatial.predict(state,Xq,geo,B,p,diag)
        fallback=~detail['applied'].to_numpy(dtype=bool); old.exact(branch[fallback],p[fallback],'spatial fallback')
        log=vector(bs['recency_log'],p52['input_sha256'][bs['recency_log']],n); ub=vector(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']],n)
        candidate=np.median(np.column_stack([branch,log,ub]),axis=1)
        assert candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
        preds={'R67':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
        for label,r,key in CONTROLS:
            name=f'outputs/round{r}/{s}/predictions.npy'; preds[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n)
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={k:score(v) for k,v in preds.items()}
        for label,r,key in (*CONTROLS,('v5',53,'v5')): assert metrics[label]==records[r]['stages'][s]['metrics'][key],label
        for name,v in (('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)):
            old.previous.save_array(folder/f'{name}.npy',v)
        for name,v in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
            with (folder/f'{name}.pkl').open('xb') as f: pickle.dump(v,f,protocol=5)
        item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,
            'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R67']) for k in preds if k!='R67'},
            'original_frame_sha256':frames,'preparation':prep,'new_NN_fits':1,'fit_seconds':training['fit_seconds'],
            'numeric_count':training['numeric_count'],'parameter_count':training['parameter_count'],'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],
            'reference_predict_calls':1,'spatial_predict_calls':1,'controls_full_metrics_exact':True,'full_query_reload_exact':True,
            'native_MAE_diagnostic':{'R67_prior':float(np.abs(new_native-yall[qi]).mean()),'R65_prior':float(np.abs(prior_control-yall[qi]).mean()),'R54_area':float(np.abs(area-yall[qi]).mean())},
            'reference_report':rr,'spatial_report':sr,'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
            'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
        write(folder/'results.json',item); result['stages'][s]=item; result['checked_stages'].append(s); result['new_NN_fits']+=1
        print(json.dumps({'stage':s,'fit_seconds':training['fit_seconds'],'vs_v18':{k:item['differences']['v18'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
    assert all(sha(ROOT/name)==digest for name,digest in source.items())
    result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,elapsed_seconds=time.monotonic()-start,
                  protected_after={name:sha(ROOT/name) for name in protected})
    assert result['protected_before']==result['protected_after']; write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    assert not OUT.exists(),'Existing output: no overwrite'
    OUT.mkdir(); tick=time.monotonic(); code=0; failure=None
    write(OUT/'started.json',{'started_at_utc':old.now(),'command':sys.argv,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    try: run()
    except Exception: code=1; failure=traceback.format_exc(); print(failure,flush=True)
    write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
