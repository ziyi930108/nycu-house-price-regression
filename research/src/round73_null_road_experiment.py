"""R73 diagnostic-only null145: zero R72 road delta, unchanged trainer and pipeline."""
from pathlib import Path
import argparse, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round65_nn_experiment as baseline

old=baseline.old
ROOT,STAGES=old.ROOT,old.STAGES
OUT=ROOT/'outputs/round73'; DOCUMENT=ROOT/'reports/round73_plan.md'
sha,read,write,spatial,rel=old.sha,old.locked,old.write,old.spatial,old.rel
PLAN_SHA='cb175588face578799887251f883d043e4f888b4dac009448befc3dc269627f0'
R72_SOURCE_SHA='0513d2246ec9f5572f6eab398f4657ff20a5e0833f5c960f541b71abec14cf93'
BASELINE_SOURCE_SHA='3854f5170170f1d0bdfc3707c8e0e04832a430cdbcd780fb7753bef18c96898c'
FORMULA='median(spatial(reference(.75*(.5*R54_area_native_TWD+.5*R73_null145_native_TWD)+.25*old_unit_LGB_TWD)),R46_recency_log_spatial,R40_unit_spatial)'
CONTROLS=(('R72',72,'R72'),('v19',71,'R71'),('v18',66,'R66'),('v17',63,'R63'),('v16',60,'R60'),
          ('v15',54,'R54'),('v13',53,'R53'),('v11',49,'R49'),('v12',50,'R50'),('R62',62,'R62'),('R65',65,'R65'))
EXPECTED={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities',
          'train_price_factor_million','query_price_factor_million','train_sample_weight'}


def prepare(folder,path72,path63,yfit,meta):
    with np.load(path72,allow_pickle=False) as z:
        assert set(z.files)==EXPECTED
        arrays={k:z[k].copy() for k in z.files}
    original_last={}
    with np.load(path63,allow_pickle=False) as z:
        assert set(z.files)==EXPECTED
        for key in EXPECTED-{'train_num','query_num'}:
            assert arrays[key].dtype==z[key].dtype and arrays[key].shape==z[key].shape and arrays[key].tobytes()==z[key].tobytes(),key
        for split,rows in (('train',meta['fit_rows']),('query',meta['query_rows'])):
            key=split+'_num'; a=arrays[key]
            assert a.dtype==z[key].dtype==np.float32 and a.shape==(rows,145) and z[key].shape==(rows,144)
            assert np.isfinite(a).all() and a[:,:144].tobytes()==z[key].tobytes()
            original_last[split]={'sha256':old.original.value_sha(a[:,144],'<f4'),'nonzero_rows':int((a[:,144]!=0).sum())}
    assert arrays['train_y_million'].dtype==np.float32 and arrays['train_y_million'].shape==(len(yfit),)
    assert arrays['train_y_million'].tobytes()==np.asarray(yfit/1e6,dtype=np.float32).tobytes()
    assert arrays['cardinalities'].tolist()==meta['cardinalities']
    for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
        assert old.original.value_sha(arrays[key],'<f4')==meta[key+'_sha256'],key
    for key in ('train_num','query_num'):
        arrays[key][:,144]=np.float32(0.0)
        assert (arrays[key][:,144]==0).all() and not np.signbit(arrays[key][:,144]).any()
    prepared=folder/'prepared.npz'
    with prepared.open('xb') as f: np.savez(f,**arrays)
    with np.load(prepared,allow_pickle=False) as z, np.load(path72,allow_pickle=False) as original:
        assert set(z.files)==EXPECTED
        for key in EXPECTED:
            assert z[key].dtype==arrays[key].dtype and z[key].shape==arrays[key].shape and z[key].tobytes()==arrays[key].tobytes()
            if key in ('train_num','query_num'):
                assert z[key][:,:144].tobytes()==original[key][:,:144].tobytes()
                assert z[key][:,144].tobytes()==np.zeros(len(z[key]),dtype=np.float32).tobytes()
            else: assert z[key].tobytes()==original[key].tobytes(),key
    prep={'fit_rows':meta['fit_rows'],'query_rows':meta['query_rows'],'numeric_count':145,
          'prepared_path':rel(prepared),'prepared_sha256':sha(prepared),'exact_NPZ_keys':sorted(EXPECTED),
          'R72_prepared_path':rel(path72),'R72_prepared_sha256':sha(path72),
          'R63_prepared_path':rel(path63),'R63_prepared_sha256':sha(path63),
          'only_changed_column_zero_based':144,'replacement':'float32 positive zero for every fit/query row',
          'original_last_column':original_last,
          'original_144_prefix_sha256':{k:old.original.value_sha(arrays[k][:,:144],'<f4') for k in ('train_num','query_num')},
          'model_numeric_values_sha256':{k:old.original.value_sha(arrays[k],'<f4') for k in ('train_num','query_num')},
          'original_144_prefix_bitwise_same_as_R72_and_R63':True,'other_seven_arrays_bitwise_same_as_R72_and_R63':True,
          'train_sample_weight_sha256':meta['train_sample_weight_sha256'],'train_sample_weight_summary':meta['train_sample_weight_summary'],
          'encoder_prior_road_prior_weights_refit':False,'query_labels_used':False,'diagnostic_only':True,'eligible_for_export':False,
          'initialization_and_random_draw_shape_inferred_from_same_source_seed_width':True,
          'R72_initial_weights_or_RNG_trace_available':False,'bitwise_retraining_determinism_claimed':False}
    write(folder/'preparation.json',prep)
    return prepared,prep


def run():
    tick_all=time.monotonic();inputs={}
    assert sha(DOCUMENT)==PLAN_SHA and sha(Path(baseline.__file__))==BASELINE_SOURCE_SHA
    assert sha(baseline.TRAINER)==baseline.TRAINER_SHA and sha(baseline.PARENT)==baseline.PARENT_SHA
    assert sha(old.TRAINER)==old.TRAINER_SHA and sha(old.ARCHITECTURE)==old.ARCHITECTURE_SHA
    def checked(name,digest):
        name=name.replace(chr(92),'/');p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT.resolve()) and p.is_file(),name
        assert name not in inputs or inputs[name]==digest,name
        if name not in inputs: assert sha(p)==digest,name
        inputs[name]=digest;return p
    def small(name):
        p=ROOT/name;v=read(p);checked(name,sha(p));checked(rel(p.with_suffix('.sha256')),sha(p.with_suffix('.sha256')));return v
    def artifact(a,name):
        mapping={k.replace(chr(92),'/'):v for k,v in a['artifact_sha256'].items()}
        return checked(name,mapping[name])
    def vector(name,digest,n):return old.vector(checked(name,digest),n)
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};assert len(protected)==20
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json');assert len(protected)==21
    p48=small('outputs/round48/plan.json');p52=small('outputs/round52/plan.json')
    records={r:small(f'outputs/round{r}/results.json') for r in (48,49,50,53,54,60,62,63,65,66,71,72)}
    assert all(r['checked_stages']==list(STAGES) for r in records.values())
    assert records[72]['new_NN_fits']==3 and records[72]['source_sha256']['src/round72_nn_experiment.py']==R72_SOURCE_SHA
    source=dict(records[72]['source_sha256'])
    for mapping in (old.previous.sources(),{rel(p):sha(p) for p in
        (Path(__file__),Path(baseline.__file__),baseline.TRAINER,baseline.PARENT,old.TRAINER,old.ARCHITECTURE)}):
        for name,digest in mapping.items():
            assert name not in source or source[name]==digest,name
            source[name]=digest
    for name,digest in source.items():checked(name,digest)
    checked(rel(DOCUMENT),PLAN_SHA)
    canonical={name:p48['input_sha256'][name] for name in
        ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT)}
    write(OUT/'plan.json',{'round':73,'candidate':'R73','diagnostic_only':True,'eligible_for_export':False,
        'formula':FORMULA,'source_sha256':source,'text_plan_sha256':PLAN_SHA,'stage_order':list(STAGES),
        'split_contract':p48['split_contract'],'numeric_count':145,'replacement_column':144,'replacement_value':'float32 +0.0',
        'primary_comparison':'R72 minus null145 R73; native and complete','secondary_comparison':'null145 R73 minus v19',
        'seed':42,'epochs':30,'batch_size':2048,'planned_new_NN_fits':3,'three_periods_regardless_of_scores':True,
        'same_original_trainer_no_RNG_wrapper':True,'initialization_draw_shape_source_contract_only':True,
        'bitwise_retraining_determinism_claimed':False,'no_new_gate':True,'no_CSV_or_export':True,
        'input_sha256_before_data':dict(inputs),'canonical_input_sha256':canonical,'protected_before':protected})
    for name,digest in canonical.items():checked(name,digest)
    Xall,yall=old.previous.load_training_rows(p48)
    result={'round':73,'candidate':'R73','diagnostic_only':True,'eligible_for_export':False,'status':'running',
        'formula':FORMULA,'checked_stages':[],'stages':{},'new_NN_fits':0,'new_submission_generated':False,
        'source_sha256':source,'plan_sha256':PLAN_SHA,'machine_plan_sha256':sha(OUT/'plan.json'),'protected_before':protected,
        'primary_comparison_direction':'R72 minus R73 null145','secondary_comparison_direction':'R73 null145 minus v19',
        'initialization_draw_shape_source_contract_only':True,'bitwise_retraining_determinism_claimed':False}
    for s in STAGES:
        tick=time.monotonic();folder=OUT/s;folder.mkdir();b=p48['bindings'][s];bs=p52['bindings'][s]
        a48,a54,a63,a71,a72=(records[r]['stages'][s] for r in (48,54,63,71,72));ind={}
        for kind in ('fit','query'):
            name=f'outputs/round72/{s}/{kind}_indices.npy';v=np.load(artifact(a72,name),allow_pickle=False);c=p48['split_contract'][s]
            assert v.dtype==np.int64 and v.shape==(c[kind+'_rows'],) and (np.diff(v)>0).all() and v.min()>=0 and v.max()<len(Xall)
            assert old.original.value_sha(v,'<i8')==c[kind+'_indices_content_sha256']
            for r in (48,54,63,71):assert np.array_equal(v,np.load(artifact(records[r]['stages'][s],f'outputs/round{r}/{s}/{kind}_indices.npy'),allow_pickle=False))
            ind[kind]=v;old.previous.save_array(folder/f'{kind}_indices.npy',v)
        fi,qi=ind['fit'],ind['query'];assert not np.intersect1d(fi,qi).size
        X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);n=len(qi)
        frames={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        for k,v in frames.items():assert all(a['original_frame_sha256'][k]==v for a in (a48,a63,a71,a72)),k
        meta=json.loads(artifact(a72,f'outputs/round72/{s}/model/training.json').read_text(encoding='utf8'))
        assert meta['status']=='completed' and meta['source_sha256']==baseline.TRAINER_SHA and meta['numeric_count']==145
        assert meta['configuration']['seed']==42 and meta['configuration']['epochs']==30 and meta['configuration']['batch_size']==2048
        path72=artifact(a72,f'outputs/round72/{s}/prepared.npz');path63=artifact(a63,f'outputs/round63/{s}/prepared.npz')
        saved_prep=read(artifact(a72,f'outputs/round72/{s}/preparation.json'))
        assert sha(path72)==meta['input_sha256']==saved_prep['prepared_sha256']
        assert sha(path63)==saved_prep['source_prepared_sha256']
        assert saved_prep['other_seven_arrays_bitwise_unchanged'] and saved_prep['original_144_prefix_bitwise_unchanged'] and not saved_prep['query_labels_used']
        prepared,prep=prepare(folder,path72,path63,yall[fi],meta)
        new_native,training=baseline.train(folder,prepared,meta,prep)
        for k in ('torch','numpy','cuda','gpu','configuration','numeric_count','cardinalities','parameter_count',
                  'train_price_factor_million_sha256','query_price_factor_million_sha256','train_sample_weight_sha256','initial_head_bias'):
            assert training[k]==meta[k],k
        assert not training['bitwise_retraining_determinism_claimed'] and not meta['bitwise_retraining_determinism_claimed']
        area=old.vector(artifact(a54,f'outputs/round54/{s}/model/predictions_twd.npy'),n)
        road_native=old.vector(artifact(a72,f'outputs/round72/{s}/model/predictions_twd.npy'),n)
        assert all(v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>0).all() for v in (area,new_native,road_native))
        mean=.5*area+.5*new_native;unit=vector(b['old_unit'],p48['input_sha256'][b['old_unit']],n);B=.75*mean+.25*unit
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
        preds={'R73':candidate,'v5':vector(b['v5'],p48['input_sha256'][b['v5']],n)}
        for label,r,key in CONTROLS:
            name=f'outputs/round{r}/{s}/predictions.npy';preds[label]=vector(name,records[r]['stages'][s]['prediction_sha256'],n)
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={k:score(v) for k,v in preds.items()}
        for label,r,key in (*CONTROLS,('v5',53,'v5')):assert metrics[label]==records[r]['stages'][s]['metrics'][key],label
        native={'R73_null145':float(np.abs(new_native-yall[qi]).mean()),'R72_road145':float(np.abs(road_native-yall[qi]).mean())}
        assert native['R72_road145']==a72['native_MAE_diagnostic']['R72_prior']
        for name,v in (('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)):old.previous.save_array(folder/f'{name}.npy',v)
        old.exact(candidate,old.vector(folder/'predictions.npy',n),'Saved complete null145')
        for name,v in (('reference_diagnostics',diag),('spatial_diagnostics',detail)):
            with (folder/f'{name}.pkl').open('xb') as f:pickle.dump(v,f,protocol=5)
        assert frames=={'fit_X':spatial.frame_sha(X),'query_X':spatial.frame_sha(Xq)}
        item={'stage':s,'status':'completed','diagnostic_only':True,'eligible_for_export':False,'fit_rows':len(fi),'query_rows':n,
            'dates':p48['splits'][s],'metrics':metrics,'differences':{k:old.previous.prior.log_run.differences(metrics[k],metrics['R73']) for k in preds if k!='R73'},
            'R72_minus_null145':old.previous.prior.log_run.differences(metrics['R73'],metrics['R72']),
            'R72_minus_null145_native_MAE_TWD':native['R72_road145']-native['R73_null145'],'native_MAE_diagnostic':native,
            'original_frame_sha256':frames,'preparation':prep,'new_NN_fits':1,'fit_seconds':training['fit_seconds'],
            'numeric_count':145,'parameter_count':training['parameter_count'],'runtime_Torch_numpy_CUDA_GPU_exact_to_R72':True,
            'configuration_factor_weights_bias_exact_to_R72':True,'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],
            'reference_predict_calls':1,'spatial_predict_calls':1,'controls_full_metrics_exact':True,'full_query_reload_exact':True,
            'query_labels_used_only_by_scorer':True,'old_NN_retrained':False,'encoder_prior_weights_refit':False,
            'reference_report':rr,'spatial_report':sr,'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-tick,
            'artifact_sha256':{rel(f):sha(f) for f in folder.rglob('*') if f.is_file()}}
        write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s);result['new_NN_fits']+=1
        print(json.dumps({'stage':s,'fit_seconds':training['fit_seconds'],'R72_minus_null145':{k:item['R72_minus_null145'][k]['relative_change_pct'] for k in ('overall','weighted')}}),flush=True)
    assert result['new_NN_fits']==3 and all(sha(ROOT/n)==h for n,h in source.items()) and sha(DOCUMENT)==PLAN_SHA
    assert set(protected)-{'runs/versions.json'}=={rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    result.update(status='three_periods_completed_diagnostic_only',input_sha256=inputs,elapsed_seconds=time.monotonic()-tick_all,
                  protected_after={n:sha(ROOT/n) for n in protected})
    assert result['protected_before']==result['protected_after'];write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args()
    assert not OUT.exists(),'Existing output: no overwrite';OUT.mkdir();tick=time.monotonic();code=0;failure=None;started=old.now()
    write(OUT/'started.json',{'round':73,'started_at_utc':started,'command':sys.argv,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    try:run()
    except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    write(OUT/'execution.json',{'round':73,'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
