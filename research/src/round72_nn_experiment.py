"""R72: append one OOF road-relative prior delta to A144; three fixed seed42 fits."""
from pathlib import Path
import argparse, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round65_nn_experiment as baseline
import round72_road_prior as road

old = baseline.old
ROOT, STAGES = old.ROOT, old.STAGES
OUT = ROOT/'outputs/round72'; DOCUMENT = ROOT/'reports/round72_plan.md'
sha, read, write, spatial, rel = old.sha, old.locked, old.write, old.spatial, old.rel
PLAN_SHA = '7f47f5a25b3c849fde471094c9a4b01d88caf8910f4479429cfbeaf6fc4ad609'
HELPER_SHA = '3c81fbe5e6a86a15f59447535a4dc982a2e4d22b9f2793487f15522b26ce78ff'
BASELINE_SOURCE_SHA = '3854f5170170f1d0bdfc3707c8e0e04832a430cdbcd780fb7753bef18c96898c'
PRIOR_SOURCE_SHA = '4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f'
R71_SHA = '8c9ddd281fb31df324b7aba4febd40c06cbee073ab45704b03b30176752ca833'
FORMULA = 'median(spatial(reference(.75*(.5*R54_area_native_TWD+.5*R72_road_prior_native_TWD)+.25*old_unit_LGB_TWD)),R46_recency_log_spatial,R40_unit_spatial)'
CONTROLS = (('v19',71,'R71'),('v18',66,'R66'),('v17',63,'R63'),('v16',60,'R60'),('v15',54,'R54'),
            ('v13',53,'R53'),('v11',49,'R49'),('v12',50,'R50'),('R62',62,'R62'),('R65',65,'R65'))


def prepare(folder, path, X, Xq, yfit, fi, saved_folds, unit_prior, saved_meta, lineage, base_evidence):
    expected = {'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities',
                'train_price_factor_million','query_price_factor_million','train_sample_weight'}
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == expected
        arrays = {k:z[k].copy() for k in z.files}
    assert arrays['train_num'].shape == (len(X),144) and arrays['query_num'].shape == (len(Xq),144)
    assert arrays['train_num'].dtype == arrays['query_num'].dtype == np.float32
    assert arrays['train_y_million'].dtype == np.float32 and arrays['train_y_million'].shape == (len(X),)
    assert arrays['train_y_million'].tobytes() == np.asarray(yfit/1e6,dtype=np.float32).tobytes()
    assert arrays['cardinalities'].tolist() == saved_meta['cardinalities']
    for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
        assert old.original.value_sha(arrays[key],'<f4') == saved_meta[key+'_sha256'], key
    # Only fit labels enter the supervised road helper; existing base priors are not refitted.
    train_delta, train_support, state, folds, evidence = road.fit_oof_road_prior(X,yfit,fi,saved_folds,unit_prior['train'])
    assert folds.dtype == np.int64 and folds.shape == (len(X),) and np.array_equal(folds,saved_folds)
    for key in ('source_indices_sha256','fold_ids_sha256','fit_target_sha256','effective_area_sha256'):
        assert evidence[key] == base_evidence[key], key
    assert evidence['fit_base_unit_prior_sha256'] == base_evidence['oof_prior_sha256']
    write(folder/'road_prior_state.json',state)
    restored = read(folder/'road_prior_state.json'); assert restored == state
    query_delta, query_support, query_diagnostics = road.predict_road_prior(restored,Xq,unit_prior['query'],return_diagnostics=True)
    deltas, supports = {'train':train_delta,'query':query_delta}, {'train':train_support,'query':query_support}
    for split, rows in (('train',len(X)),('query',len(Xq))):
        delta, support = deltas[split], supports[split]
        assert delta.dtype == np.float32 and delta.shape == (rows,) and np.isfinite(delta).all()
        assert support.dtype == np.int64 and support.shape == (rows,) and (support>=0).all()
        old.previous.save_array(folder/f'{split}_road_delta.npy',delta)
        old.previous.save_array(folder/f'{split}_road_support.npy',support)
    old.previous.save_array(folder/'road_fold_ids.npy',folds)
    write(folder/'road_prior_evidence.json',evidence)
    write(folder/'road_query_diagnostics.json',query_diagnostics)
    new = {**arrays,'train_num':np.column_stack([arrays['train_num'],train_delta]),
                   'query_num':np.column_stack([arrays['query_num'],query_delta])}
    prepared = folder/'prepared.npz'
    with prepared.open('xb') as f: np.savez(f,**new)
    with np.load(prepared,allow_pickle=False) as z:
        assert set(z.files) == expected
        for key in expected-{'train_num','query_num'}:
            assert z[key].dtype == arrays[key].dtype and z[key].shape == arrays[key].shape and z[key].tobytes() == arrays[key].tobytes(), key
        for split,rows in (('train',len(X)),('query',len(Xq))):
            key = split+'_num'
            assert z[key].dtype == np.float32 and z[key].shape == (rows,145)
            assert z[key][:,:144].tobytes() == arrays[key].tobytes() and z[key][:,144].tobytes() == deltas[split].tobytes()
    summaries = {}
    for split in ('train','query'):
        d,c = deltas[split],supports[split]
        summaries[split] = {'rows':len(d),'delta_min':float(d.min()),'delta_max':float(d.max()),'delta_mean':float(d.mean(dtype=np.float64)),
            'delta_quantiles':np.quantile(d.astype(np.float64),[0,.1,.5,.9,1]).tolist(),'zero_delta_rows':int((d==0).sum()),
            'selected_support_positive_rows':int((c>0).sum()),'support_min':int(c.min()),'support_max':int(c.max()),
            'support_mean':float(c.mean(dtype=np.float64)),'support_quantiles':np.quantile(c,[0,.5,.9,1]).tolist()}
    prep = {'fit_rows':len(X),'query_rows':len(Xq),'numeric_count':145,'prepared_path':rel(prepared),'prepared_sha256':sha(prepared),
        'source_prepared_path':rel(path),'source_prepared_sha256':sha(path),'exact_NPZ_keys':sorted(expected),
        'road_input_lineage':lineage,'road_helper_sha256':HELPER_SHA,'road_rules':road.RULES,
        'road_prior_state_path':rel(folder/'road_prior_state.json'),'road_prior_state_sha256':sha(folder/'road_prior_state.json'),
        'road_prior_evidence_path':rel(folder/'road_prior_evidence.json'),'road_prior_evidence_sha256':sha(folder/'road_prior_evidence.json'),
        'road_query_diagnostics_path':rel(folder/'road_query_diagnostics.json'),'road_query_diagnostics_sha256':sha(folder/'road_query_diagnostics.json'),
        'road_feature_summary':summaries,'added_feature':'road_relative_log_unit_prior','numeric_column_position_zero_based':144,
        'formula':'float32(log(road_unit_prior)-log(saved_R60_base_unit_prior)); unavailable road cell -> 0',
        'additional_scaling_or_clipping':False,'factor_inversion_used':False,
        'train_feature_role':'same saved group-fold OOF; complete held fold excluded','query_feature_role':'full-fit road state and saved full-fit base prior',
        'added_feature_sha256':{k:old.original.value_sha(v,'<f4') for k,v in deltas.items()},
        'selected_support_sha256':{k:old.original.value_sha(v,'<i8') for k,v in supports.items()},
        'fold_ids_sha256':old.original.value_sha(folds,'<i8'),'fit_source_indices_sha256':old.original.value_sha(fi,'<i8'),
        'fit_target_values_sha256':old.original.value_sha(yfit,'<f8'),
        'original_144_prefix_sha256':{k:old.original.value_sha(arrays[k],'<f4') for k in ('train_num','query_num')},
        'model_numeric_values_sha256':{k:old.original.value_sha(new[k],'<f4') for k in ('train_num','query_num')},
        'other_seven_arrays_bitwise_unchanged':True,'original_144_prefix_bitwise_unchanged':True,
        'encoder_refit':False,'base_prior_refit':False,'sample_weights_rebuilt':False,'new_road_prior_uses_fit_labels':True,
        'query_labels_used':False,'input_width_changes_parameter_count_and_initialization':True,'head_bias_formula_unchanged':True,
        'train_sample_weight_sha256':saved_meta['train_sample_weight_sha256'],'train_sample_weight_summary':saved_meta['train_sample_weight_summary']}
    write(folder/'preparation.json',prep)
    return prepared,prep


def run():
    start = time.monotonic(); inputs = {}
    assert sha(DOCUMENT) == PLAN_SHA and sha(Path(road.__file__)) == HELPER_SHA
    assert sha(Path(baseline.__file__)) == BASELINE_SOURCE_SHA
    assert sha(baseline.TRAINER) == baseline.TRAINER_SHA and sha(baseline.PARENT) == baseline.PARENT_SHA
    assert sha(old.TRAINER) == old.TRAINER_SHA and sha(old.ARCHITECTURE) == old.ARCHITECTURE_SHA
    def checked(name,digest):
        name = name.replace('\\','/'); path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs: assert sha(path) == digest, name
        inputs[name] = digest; return path
    def small(name):
        path=ROOT/name; value=read(path); checked(name,sha(path))
        checked(rel(path.with_suffix('.sha256')),sha(path.with_suffix('.sha256'))); return value
    def artifact(record,name):
        mapping={k.replace('\\','/'):v for k,v in record['artifact_sha256'].items()}
        return checked(name,mapping[name])
    def vector(name,digest,n): return old.vector(checked(name,digest),n)
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 20
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json'); assert len(protected) == 21
    p48=small('outputs/round48/plan.json'); p52=small('outputs/round52/plan.json')
    records={r:small(f'outputs/round{r}/results.json') for r in (48,49,50,53,54,60,62,63,65,66,71)}
    assert inputs['outputs/round71/results.json'] == R71_SHA
    assert all(r['checked_stages'] == list(STAGES) for r in records.values())
    assert records[71]['primary_seed'] == 42 and not records[71]['seed_average']
    prior_name='src/round60_recency_price_prior.py'
    assert records[60]['source_sha256'][prior_name] == PRIOR_SOURCE_SHA
    source=dict(records[71]['source_sha256'])
    for mapping in (old.previous.sources(),records[60]['source_sha256'],{rel(p):sha(p) for p in
        (Path(__file__),Path(old.__file__),Path(baseline.__file__),Path(road.__file__),baseline.TRAINER,baseline.PARENT,old.TRAINER,old.ARCHITECTURE)}):
        for name,digest in mapping.items():
            assert name not in source or source[name] == digest, name
            source[name]=digest
    for name,digest in source.items(): checked(name,digest)
    checked(rel(DOCUMENT),PLAN_SHA)
    canonical_inputs={name:p48['input_sha256'][name] for name in
        ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT)}
    write(OUT/'plan.json',{'round':72,'candidate':'R72','formula':FORMULA,'source_sha256':source,'text_plan_sha256':PLAN_SHA,
        'stage_order':list(STAGES),'split_contract':p48['split_contract'],'primary_control':'v19=R71 seed42','native_control':'R63 A144',
        'numeric_representation':'R63 original 144 columns plus one unscaled signed road-relative log unit-prior delta',
        'numeric_count':145,'configuration_unchanged_from_R63':True,'other_seven_arrays_bitwise_unchanged':True,
        'road_rules':road.RULES,'additional_feature_scaling_or_clipping':False,'query_labels_in_helper_or_trainer':False,
        'encoder_base_prior_weights_rebuilt':False,'seed':42,'epochs':30,'planned_new_NN_fits':3,
        'input_width_changes_initialization':True,'three_periods_regardless_of_scores':True,'new_gate':None,
        'input_sha256_before_real_data_statistics':dict(inputs),'canonical_input_sha256':canonical_inputs,'protected_before':protected})
    # The complete fixed machine plan precedes real-X feature statistics and any fit.
    for name,digest in canonical_inputs.items(): checked(name,digest)
    Xall,yall=old.previous.load_training_rows(p48)
    result={'round':72,'candidate':'R72','primary_control':'v19','status':'running','formula':FORMULA,'checked_stages':[],'stages':{},
        'new_NN_fits':0,'new_submission_generated':False,'fresh_blind_validation':False,'protected_before':protected,
        'source_sha256':source,'plan_sha256':PLAN_SHA,'machine_plan_sha256':sha(OUT/'plan.json')}
    for s in STAGES:
        tick=time.monotonic(); folder=OUT/s; folder.mkdir(); b=p48['bindings'][s]; bs=p52['bindings'][s]
        a48,a54,a60,a63,a71=(records[r]['stages'][s] for r in (48,54,60,63,71)); indices={}
        for kind in ('fit','query'):
            name=f'outputs/round48/{s}/{kind}_indices.npy'; indices[kind]=np.load(artifact(a48,name),allow_pickle=False)
            ind=indices[kind]; contract=p48['split_contract'][s]
            assert ind.dtype == np.int64 and ind.shape == (contract[kind+'_rows'],) and (np.diff(ind)>0).all()
            assert ind.min()>=0 and ind.max()<len(Xall) and old.original.value_sha(ind,'<i8') == contract[kind+'_indices_content_sha256']
            for r in (54,60,63,65,66,71):
                assert np.array_equal(ind,np.load(artifact(records[r]['stages'][s],f'outputs/round{r}/{s}/{kind}_indices.npy'),allow_pickle=False))
            old.previous.save_array(folder/f'{kind}_indices.npy',ind)
        fi,qi=indices['fit'],indices['query']; X=Xall.iloc[fi].reset_index(drop=True); Xq=Xall.iloc[qi].reset_index(drop=True); n=len(qi)
        assert not np.intersect1d(fi,qi).size
        frames={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        for key,value in frames.items(): assert all(a['original_frame_sha256'][key] == value for a in (a48,a60,a63,a71)), key
        saved_meta=json.loads(artifact(a63,f'outputs/round63/{s}/model/training.json').read_text(encoding='utf8'))
        assert saved_meta['status']=='completed' and saved_meta['source_sha256']==baseline.TRAINER_SHA
        assert saved_meta['configuration']['seed']==42 and saved_meta['configuration']['epochs']==30 and saved_meta['numeric_count']==144
        saved_prep=read(artifact(a63,f'outputs/round63/{s}/preparation.json'))
        assert not saved_prep['query_labels_used'] and not saved_prep['prior_refit'] and not saved_prep['encoder_refit']
        assert saved_prep['all_eight_R60_arrays_reused_without_changes'] and saved_prep['weights_use_fit_X_only']
        assert not saved_prep['weights_use_targets'] and not saved_prep['weights_use_query_dates']
        path=artifact(a63,f'outputs/round63/{s}/prepared.npz')
        assert sha(path)==saved_meta['input_sha256']==saved_prep['prepared_sha256']
        ep=artifact(a60,f'outputs/round60/{s}/prior_preparation.json'); prior_prep=read(ep); e=prior_prep['prior_evidence']
        assert prior_prep['fit_labels_used_for_prior'] and not prior_prep['query_labels_used'] and not prior_prep['encoder_refit']
        assert saved_prep['prior_preparation_sha256']==sha(ep)
        assert e['source_indices_sha256']==old.original.value_sha(fi,'<i8') and e['fit_target_sha256']==old.original.value_sha(yall[fi],'<f8')
        assert e['effective_area_sha256']==old.original.value_sha(old.area_helper.effective_area(X),'<f8')
        assert e['training_prior_role']=='group-fold OOF; complement labels and dates only' and e['query_prior_role']=='full fit state' and not e['uses_query_labels']
        fp=artifact(a60,f'outputs/round60/{s}/prior_fold_ids.npy'); folds=np.load(fp,allow_pickle=False)
        assert folds.dtype==np.int64 and folds.shape==(len(fi),) and old.original.value_sha(folds,'<i8')==e['fold_ids_sha256']
        lineage={rel(ep):sha(ep),rel(fp):sha(fp)}; values={}
        for split,stem,rows in (('train','train_oof_unit_prior',len(fi)),('query','query_unit_prior',n)):
            ap=artifact(a60,f'outputs/round60/{s}/{stem}.npy'); value=np.load(ap,allow_pickle=False)
            assert value.dtype==np.float64 and value.shape==(rows,) and np.isfinite(value).all() and (value>0).all()
            values[split]=value; lineage[rel(ap)]=sha(ap)
        assert old.original.value_sha(values['train'],'<f8')==e['oof_prior_sha256']
        prepared,prep=prepare(folder,path,X,Xq,yall[fi],fi,folds,values,saved_meta,lineage,e)
        assert frames=={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        new_native,training=baseline.train(folder,prepared,saved_meta,prep)
        area=old.vector(artifact(a54,f'outputs/round54/{s}/model/predictions_twd.npy'),n)
        prior_control=old.vector(artifact(a63,f'outputs/round63/{s}/model/predictions_twd.npy'),n)
        assert all(v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>0).all() for v in (area,new_native,prior_control))
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
        assert candidate.dtype==np.float64 and candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
        preds={'R72':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
        for label,r,key in CONTROLS:
            name=f'outputs/round{r}/{s}/predictions.npy'; preds[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n)
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={k:score(v) for k,v in preds.items()}
        for label,r,key in (*CONTROLS,('v5',53,'v5')): assert metrics[label]==records[r]['stages'][s]['metrics'][key],label
        for name,v in (('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)):
            old.previous.save_array(folder/f'{name}.npy',v)
        old.exact(candidate,old.vector(folder/'predictions.npy',n),'Saved complete candidate')
        for name,v in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
            with (folder/f'{name}.pkl').open('xb') as f: pickle.dump(v,f,protocol=5)
        assert frames=={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],'metrics':metrics,
            'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R72']) for k in preds if k!='R72'},
            'original_frame_sha256':frames,'preparation':prep,'new_NN_fits':1,'fit_seconds':training['fit_seconds'],
            'numeric_count':training['numeric_count'],'parameter_count':training['parameter_count'],'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],
            'reference_predict_calls':1,'spatial_predict_calls':1,'controls_full_metrics_exact':True,'full_query_reload_exact':True,
            'query_labels_used_only_by_scorer':True,'old_NN_retrained':False,'encoder_base_prior_weights_refit':False,
            'native_MAE_diagnostic':{'R72_prior':float(np.abs(new_native-yall[qi]).mean()),'R63_prior':float(np.abs(prior_control-yall[qi]).mean()),'R54_area':float(np.abs(area-yall[qi]).mean())},
            'reference_report':rr,'spatial_report':sr,'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
            'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
        write(folder/'results.json',item); result['stages'][s]=item; result['checked_stages'].append(s); result['new_NN_fits']+=1
        print(json.dumps({'stage':s,'fit_seconds':training['fit_seconds'],'vs_v19':{k:item['differences']['v19'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
    assert result['new_NN_fits']==3 and all(sha(ROOT/name)==digest for name,digest in source.items()) and sha(DOCUMENT)==PLAN_SHA
    assert set(protected)-{'runs/versions.json'}=={rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    result.update(status='three_periods_completed_no_adoption_decision',input_sha256=inputs,elapsed_seconds=time.monotonic()-start,
                  protected_after={name:sha(ROOT/name) for name in protected})
    assert result['protected_before']==result['protected_after']; write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true',required=True); parser.parse_args()
    assert not OUT.exists(),'Existing output: no overwrite'; OUT.mkdir(); tick=time.monotonic(); code=0; failure=None; started=old.now()
    write(OUT/'started.json',{'round':72,'started_at_utc':started,'command':sys.argv,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    try: run()
    except Exception: code=1; failure=traceback.format_exc(); print(failure,flush=True)
    write(OUT/'execution.json',{'round':72,'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
