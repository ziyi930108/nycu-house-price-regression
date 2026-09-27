"""R77 final area preparation plus one fixed R74 weighted-area NN fit.

Uses the saved R55 eight-array final data and only R63's saved fit weights.
No encoder/prior/weight refit, reference/spatial prediction, CSV or selection.
A root-authored final-plan SHA is mandatory; existing attempts are never reused.
"""
from pathlib import Path
import argparse, contextlib, hashlib, json, re, sys, time, traceback
import numpy as np
import round74_nn_experiment as historical

ROOT=historical.ROOT
OUT=ROOT/'outputs/round77/final_area'
DOCUMENT=ROOT/'reports/round76_77_final_plan.md'
P55='outputs/round55/final/'
P63='outputs/round63/final/'
AUTHORITIES={P55+'trial_release.json':'e3b3f4333a4d8aa056728819cb2853e8c296f5d2dd5f03e1a2f067e612badb0d',
             P63+'trial_release.json':'55e8fcffc1aef16dd056652191aff5363fa99332db97bcfd013956e88ce5b9a1'}
SOURCE_PINS={'src/round74_nn_experiment.py':'f81cc1519e8d74c226afb40bb0bb4b834dd6dc3657c14bccdb4c6f0c70f34a81',
 'src/round74_recency_weighted_area_train.py':'f760d8a264c5c84c9a612a55ba77be4c0a3bafac655b1f21d0de99b2951f7b4d',
 'src/round48_nn_train.py':'38fb91e01eabb2ed3163f1260e16c826ca55592951c22ce1107ad68c24b37082',
 'src/round47_nn_train.py':'68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7',
 'src/round63_recency_weighted_prior_train.py':'26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'}
sha,rel,write=historical.sha,historical.rel,historical.write


def run(plan_sha):
    tick=time.monotonic();inputs={};sources={}
    def bind(name,digest):
        name=name.replace('\\','/');path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(),name
        assert re.fullmatch('[0-9a-f]{64}',digest),name
        assert name not in inputs or inputs[name]==digest,name
        if name not in inputs:assert sha(path)==digest,name
        inputs[name]=digest;return path
    def small(name,digest,receipt=True):
        path=bind(name,digest)
        if receipt:
            value=historical.read(path);rp=path.with_suffix('.sha256');bind(rel(rp),sha(rp))
            return value
        return json.loads(path.read_text(encoding='utf8'))
    def artifact(proof,name,as_json=False,receipt=True):
        mapping={k.replace('\\','/'):v for k,v in proof['artifact_sha256'].items()}
        path=bind(name,mapping[name])
        return small(name,mapping[name],receipt) if as_json else path
    bind(rel(DOCUMENT),plan_sha)
    p55,p63=[small(name,digest) for name,digest in AUTHORITIES.items()]
    for proof in (p55,p63):
        assert proof['fit_rows']==467221 and proof['query_rows']==47073
        assert proof['saved_model_reload_exact'] and proof['reload_rows']==47073
        for name,digest in proof['source_sha256'].items():
            assert name not in sources or sources[name]==digest,name
            sources[name]=digest
    for name,digest in {**SOURCE_PINS,rel(Path(__file__)):sha(Path(__file__))}.items():
        assert name not in sources or sources[name]==digest,name
        sources[name]=digest
    for name,digest in sources.items():bind(name,digest)
    assert p55['round']==55 and p63['round']==63
    assert p55['input_frame_sha256']==p63['input_frame_sha256']
    assert p55['fit_target_values_sha256']==p63['fit_target_values_sha256']
    target_sha=p55['fit_target_values_sha256'];frames=p55['input_frame_sha256']
    assert set(frames)=={'X','Xq','geo','geoq'}
    # The saved full row index applies after canonical train K + valid K, not raw train.
    for name in ('train','valid','full','test'):
        path=f'outputs/round6/geography/final/{name}_source_cache_indices.npy'
        assert p55['input_sha256'][path]==p63['input_sha256'][path]
        bind(path,p55['input_sha256'][path])
    for kind,rows in (('fit',467221),('query',47073)):
        paths=[artifact(p,prefix+kind+'_indices.npy') for p,prefix in ((p55,P55),(p63,P63))]
        assert sha(paths[0])==sha(paths[1])
        a,b=[np.load(path,allow_pickle=False) for path in paths]
        assert a.dtype==b.dtype==np.int64 and a.shape==b.shape==(rows,)
        assert np.array_equal(a,np.arange(rows,dtype=np.int64)) and a.tobytes()==b.tobytes()
        historical.old.previous.save_array(OUT/f'{kind}_indices.npy',a)
    meta55=artifact(p55,P55+'model/training.json',True,False)
    meta63=artifact(p63,P63+'model/training.json',True,False)
    prep55=artifact(p55,P55+'preparation.json',True)
    prep63=artifact(p63,P63+'preparation.json',True)
    state=artifact(p55,P55+'numeric_state.json',True)
    assert prep55==p55['numeric_preparation'] and prep63==p63['sample_weight_preparation']
    assert state['fit_rows']==467221 and state['input_width']==144 and state['output_width']==603
    assert state['fit_uses_query'] is False and state['fit_uses_target'] is False and state['original_144_preserved']
    assert prep55['state_fit_only'] and prep55['original_numeric_prefix_bitwise_equal'] and not prep55['query_labels_used']
    for meta in (meta55,meta63):
        assert meta['status']=='completed' and meta['fit_rows']==467221 and meta['query_rows']==47073
        assert meta['saved_model_reload_exact'] and meta['reload_rows']==47073
        assert not meta['query_labels_used'] and not meta['query_labels_available']
        assert meta['configuration']['seed']==42 and meta['configuration']['epochs']==30 and meta['configuration']['batch_size']==2048
    assert meta55['numeric_count']==603 and meta63['numeric_count']==144
    assert meta55['source_sha256']==SOURCE_PINS['src/round48_nn_train.py']
    assert meta63['source_sha256']==SOURCE_PINS['src/round63_recency_weighted_prior_train.py']
    assert not meta55['extra_sample_weights'] and meta63['extra_sample_weights']
    assert prep63['weights_use_fit_X_only'] and not prep63['weights_use_targets'] and not prep63['weights_use_query_dates']
    assert not prep63['query_labels_used'] and not prep63['float32_weights_renormalized'] and not prep63['batch_weights_renormalized']
    assert prep63['neutral_weights_exactly_one'] and len(prep63['neutral_date_positions'])==8
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    protected['outputs/current_private_candidate.json']=sha(ROOT/'outputs/current_private_candidate.json')
    paths=[artifact(p,prefix+'prepared.npz') for p,prefix in ((p55,P55),(p63,P63))]
    for path,meta in zip(paths,(meta55,meta63)):assert inputs[rel(path)]==meta['input_sha256']
    write(OUT/'input_contract.json',{'round':77,'component':'R74_recency_weighted_area_final',
        'plan_path':rel(DOCUMENT),'plan_sha256':plan_sha,'source_sha256':sources,'input_sha256':dict(inputs),
        'input_frame_sha256':frames,'fit_target_values_sha256':target_sha,'protected_before':protected,
        'fit_rows':467221,'query_rows':47073,'planned_new_NN_fits':1,'query_labels_used':False,
        'raw_feature_or_label_caches_reloaded':False,'encoder_prior_ramp_or_weights_refit':False})
    with np.load(paths[0],allow_pickle=False) as z:
        assert set(z.files)==historical.AREA_KEYS
        arrays={key:z[key].copy() for key in z.files}
    value_hash={key:historical.sha_array(value) for key,value in arrays.items()}
    for key,digest in prep55['unchanged_six_array_sha256'].items():assert value_hash[key]==digest,key
    with np.load(paths[1],allow_pickle=False) as z:
        assert set(z.files)==historical.PRIOR_KEYS
        for key in ('train_cat','query_cat','cardinalities','train_y_million'):
            other=z[key];a=arrays[key]
            assert a.dtype==other.dtype and a.shape==other.shape and a.tobytes()==other.tobytes(),key
        weights=z['train_sample_weight'].copy()
    assert arrays['train_num'].dtype==arrays['query_num'].dtype==np.float32
    assert arrays['train_num'].shape==(467221,603) and arrays['query_num'].shape==(47073,603)
    assert arrays['train_y_million'].dtype==np.float32 and arrays['train_y_million'].shape==(467221,)
    assert arrays['train_cat'].dtype==arrays['query_cat'].dtype==arrays['cardinalities'].dtype==np.int64
    assert arrays['train_cat'].shape==(467221,23) and arrays['query_cat'].shape==(47073,23)
    assert arrays['cardinalities'].tolist()==meta55['cardinalities']==meta63['cardinalities']
    assert weights.dtype==np.float32 and weights.shape==(467221,) and np.isfinite(weights).all() and (weights>0).all()
    weight_sha=historical.sha_array(weights)
    assert weight_sha==meta63['train_sample_weight_sha256']==prep63['train_sample_weight_sha256']
    neutral=np.asarray(prep63['neutral_date_positions'],dtype=np.int64)
    assert len(np.unique(neutral))==8 and neutral.min()>=0 and neutral.max()<467221 and np.all(weights[neutral]==1)
    for key,rows in (('train_area_factor',467221),('query_area_factor',47073)):
        a=arrays[key];assert a.dtype==np.float32 and a.shape==(rows,) and np.isfinite(a).all() and (a>0).all()
        assert value_hash[key]==meta55[key+'_sha256']
    data=OUT/'prepared.npz'
    with data.open('xb') as f:np.savez(f,**arrays,train_sample_weight=weights)
    with np.load(data,allow_pickle=False) as saved:
        assert set(saved.files)==historical.AREA_KEYS|{'train_sample_weight'}
        for key,a in arrays.items():
            b=saved[key];assert b.dtype==a.dtype and b.shape==a.shape and b.tobytes()==a.tobytes(),key
            assert historical.sha_array(b)==value_hash[key]
        assert saved['train_sample_weight'].tobytes()==weights.tobytes()
    prep={'fit_rows':467221,'query_rows':47073,'numeric_count':603,'prepared_path':rel(data),'prepared_sha256':sha(data),
        'R55_prepared_path':rel(paths[0]),'R55_prepared_sha256':inputs[rel(paths[0])],
        'R63_prepared_path':rel(paths[1]),'R63_prepared_sha256':inputs[rel(paths[1])],
        'numeric_state_sha256':inputs[P55+'numeric_state.json'],'input_frame_sha256':frames,'fit_target_values_sha256':target_sha,
        'original_eight_arrays_value_sha256':value_hash,'original_eight_arrays_bitwise_unchanged':True,
        'shared_categories_cardinalities_target_bitwise_equal':True,'exact_NPZ_keys':sorted(historical.AREA_KEYS|{'train_sample_weight'}),
        'train_sample_weight_sha256':weight_sha,'train_sample_weight_summary':meta63['train_sample_weight_summary'],
        'saved_date_weight_summary_float64':prep63['date_weight_summary_float64'],'neutral_weight_rows':8,
        'weight_provenance':'Exact saved R63 final float32 fit-date weights; invalid dates neutral 1',
        'feature_encoder_refit':False,'sample_weights_recomputed':False,'sample_weights_renormalized':False,
        'query_labels_used':False,'query_dates_used_for_weights':False,'raw_labels_reloaded':False}
    write(OUT/'preparation.json',prep)
    del arrays,weights
    predictions,training=historical.train(OUT,data,meta55,prep)
    assert predictions.dtype==np.float64 and predictions.shape==(47073,) and np.isfinite(predictions).all() and (predictions>0).all()
    assert sha(DOCUMENT)==plan_sha and all(sha(ROOT/name)==digest for name,digest in sources.items())
    assert set(protected)-{'runs/versions.json','outputs/current_private_candidate.json'}=={rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    after={name:sha(ROOT/name) for name in protected};assert after==protected
    proof={'round':77,'status':'completed','component':'R74_recency_weighted_area_final','plan_path':rel(DOCUMENT),'plan_sha256':plan_sha,
        'fit_rows':467221,'query_rows':47073,'numeric_count':603,'seed':42,'epochs':30,'new_fits':1,'new_NN_fits':1,
        'source_sha256':sources,'input_sha256':inputs,'input_frame_sha256':frames,'fit_target_values_sha256':target_sha,
        'preparation':prep,'prepared_sha256':prep['prepared_sha256'],'training_configuration':training['configuration'],
        'fit_seconds':training['fit_seconds'],'saved_model_reload_exact':training['saved_model_reload_exact'],'reload_rows':training['reload_rows'],
        'prediction_path':rel(OUT/'model/predictions_twd.npy'),'prediction_sha256':sha(OUT/'model/predictions_twd.npy'),
        'prediction_units':'TWD','area_applied_in_trainer_only':True,'unweighted_fit_median_initialization_preserved':True,
        'query_labels_used':False,'encoder_prior_ramp_or_weights_refit':False,'reference_predict_calls':0,'spatial_predict_calls':0,
        'new_CSV':False,'selector_updated':False,'registry_updated':False,'protected_before':protected,'protected_after':after,
        'artifact_sha256':{rel(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='run.log'},
        'elapsed_seconds':time.monotonic()-tick,'completed_at_utc':historical.old.now()}
    write(OUT/'results.json',proof)
    print(json.dumps({'status':'completed','new_NN_fits':1,'fit_seconds':training['fit_seconds'],'prediction_sha256':proof['prediction_sha256']}),flush=True)


def execute(plan_sha):
    assert re.fullmatch('[0-9a-f]{64}',plan_sha) and DOCUMENT.is_file() and sha(DOCUMENT)==plan_sha,'Root final-plan SHA required'
    assert not OUT.exists(),'Existing attempt: no overwrite or automatic retry'
    OUT.mkdir(parents=True);tick=time.monotonic();started=historical.old.now();code=0;failure=None
    write(OUT/'attempt.json',{'round':77,'started_at_utc':started,'command':sys.argv,'plan_path':rel(DOCUMENT),
        'plan_sha256':plan_sha,'source_sha256':sha(Path(__file__)),'planned_new_NN_fits':1,'automatic_retry':False})
    with (OUT/'run.log').open('x',encoding='utf8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:run(plan_sha)
        except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    event={'round':77,'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':historical.old.now(),
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'plan_sha256':plan_sha,
        'fit_attempt_record_exists':(OUT/'model/fit_started.json').exists(),'model_saved':(OUT/'model/model.pt').exists(),
        'training_metadata_exists':(OUT/'model/training.json').exists(),'automatic_retry':False,'log_sha256':sha(OUT/'run.log')}
    write(OUT/'execution.json',event);print(json.dumps(event),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--plan-sha',required=True);args=parser.parse_args();raise SystemExit(execute(args.plan_sha))
