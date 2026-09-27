"""R77: fixed saved-vector area/unit four-cell comparison; no fitting or prediction calls."""
from pathlib import Path
import argparse,json,sys,time,traceback
import numpy as np
import round48_nn_experiment as old
import round40_three_branch as unit_run

ROOT=old.ROOT
OUT=ROOT/'outputs/round77'
DOCUMENT=ROOT/'reports/round77_plan.md'
STAGES=old.STAGES
sha,read,write=old.sha,old.locked,old.write
rel=lambda p:Path(p).relative_to(ROOT).as_posix()
FORMULA='median(saved_R74_nn_spatial,saved_R46_recency_log_spatial,saved_R76_candidate_unit_spatial)'


def run(plan_sha):
    tick=time.monotonic(); inputs={}
    def checked(name,digest):
        name=name.replace('\\','/')
        assert name not in inputs or inputs[name]==digest,name
        if name not in inputs:assert sha(ROOT/name)==digest,name
        inputs[name]=digest;return ROOT/name
    def small(name):
        p=ROOT/name;v=read(p);checked(name,sha(p));checked(rel(p.with_suffix('.sha256')),sha(p.with_suffix('.sha256')));return v
    def artifact(a,name):
        m={k.replace('\\','/'):v for k,v in a['artifact_sha256'].items()};return checked(name,m[name])
    checked(rel(DOCUMENT),plan_sha)
    p48,p52=(small(f'outputs/round{r}/plan.json') for r in (48,52))
    records={r:small(f'outputs/round{r}/results.json') for r in (71,74,76)}
    assert sha(ROOT/'outputs/round76/results.json')=='a7ea0219deec6aacaf43be9abd9b62e41b0cd720d09274d6fa81c5fa34d81631'
    assert all(r['checked_stages']==list(STAGES) and r['status']=='three_periods_completed_no_adoption_decision' for r in records.values())
    source=old.previous.merge(records[76]['source_sha256'],records[74]['source_sha256'],
        {rel(Path(__file__)):sha(Path(__file__)),rel(Path(old.__file__)):sha(Path(old.__file__))})
    assert all(sha(ROOT/n)==h for n,h in source.items())
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    selector=ROOT/'outputs/current_private_candidate.json';selector_sha=sha(selector)
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT):
        assert p48['input_sha256'][name]==records[76]['input_sha256'][name]
        checked(name,p48['input_sha256'][name])
    bindings={}
    defaults={'v11':49,'v12':50,'v5':36,'v18':66}
    for s in STAGES:
        a71,a74,a76=(records[r]['stages'][s] for r in (71,74,76))
        b={'A0':f'outputs/round71/{s}/seed42/nn_spatial.npy',
           'A1':f'outputs/round74/{s}/nn_spatial.npy',
           'U1':f'outputs/round76/{s}/candidate_unit_spatial.npy',
           'U0':p48['bindings'][s]['unit_spatial'],'L':p52['bindings'][s]['recency_log']}
        for key,a in (('A0',a71),('A1',a74),('U1',a76)):artifact(a,b[key])
        for key,p in (('U0',p48),('L',p52)):
            checked(b[key],p['input_sha256'][b[key]])
            assert records[76]['input_sha256'][b[key]]==inputs[b[key]]
        for label,r in (('v19',71),('R74',74),('R76',76)):
            name=f'outputs/round{r}/{s}/predictions.npy';b[label]=name;artifact(records[r]['stages'][s],name)
            assert records[r]['stages'][s]['prediction_sha256']==inputs[name]
        for label,r in defaults.items():
            name=f'outputs/round{r}/{s}/'+('log_spatial_combination.npy' if label=='v5' else 'predictions.npy')
            b[label]=name;checked(name,records[71]['input_sha256'][name]);assert label in a71['metrics']
        for kind in ('fit','query'):
            name=f'outputs/round71/{s}/{kind}_indices.npy';artifact(a71,name);b[kind+'_indices']=name
            for r,a in ((74,a74),(76,a76)):
                other=f'outputs/round{r}/{s}/{kind}_indices.npy';artifact(a,other);assert inputs[other]==inputs[name]
        bindings[s]=b
    write(OUT/'plan.json',{'round':77,'candidate':'R77','formula':FORMULA,'primary_control':'R76',
        'four_cells':{'A0_U0':'v19=R71','A1_U0':'R74','A0_U1':'R76','A1_U1':'R77'},
        'stage_order':list(STAGES),'split_contract':p48['split_contract'],'splits':p48['splits'],
        'target_group_weights':p48['target_group_weights'],'source_sha256':source,'input_sha256':dict(inputs),
        'bindings':bindings,'text_plan_sha256':plan_sha,'protected_before':protected,'selector_sha256_before':selector_sha,
        'new_fits':0,'model_loads':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,
        'all_periods_regardless_of_scores':True,'seed_weight_threshold_search':False,'new_CSV':False,
        'selector_update_allowed':False,'automatic_adoption':False,'fresh_blind_validation':False})
    Xall,yall=old.previous.load_training_rows(p48)
    result={'round':77,'candidate':'R77','formula':FORMULA,'primary_control':'R76','status':'running',
        'checked_stages':[],'stages':{},'source_sha256':source,'input_sha256':inputs,
        'plan_sha256':sha(OUT/'plan.json'),'text_plan_sha256':plan_sha,
        'protected_before':protected,'selector_sha256_before':selector_sha,'new_fits':0,'new_NN_fits':0,
        'model_loads':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,
        'new_CSV':False,'new_submission_generated':False,'fresh_blind_validation':False,
        'seed_weight_threshold_search':False,'Public_feedback_used_for_tuning':False}
    for s in STAGES:
        start=time.monotonic();folder=OUT/s;folder.mkdir();b=bindings[s];c=p48['split_contract'][s];indices={}
        for kind in ('fit','query'):
            a=np.load(ROOT/b[kind+'_indices'],allow_pickle=False)
            assert a.dtype==np.int64 and a.shape==(c[kind+'_rows'],) and (np.diff(a)>0).all()
            assert a.min()>=0 and a.max()<len(Xall) and old.original.value_sha(a,'<i8')==c[kind+'_indices_content_sha256']
            for r in (74,76):
                other=np.load(ROOT/f'outputs/round{r}/{s}/{kind}_indices.npy',allow_pickle=False)
                assert other.dtype==a.dtype and other.shape==a.shape and other.tobytes()==a.tobytes()
            indices[kind]=a;old.previous.save_array(folder/f'{kind}_indices.npy',a)
        fi,qi=indices['fit'],indices['query'];assert not np.intersect1d(fi,qi).size;n=len(qi)
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
        frames={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        assert all(frames==records[r]['stages'][s]['original_frame_sha256'] for r in records)
        values={k:unit_run.vector(ROOT/name,n) for k,name in b.items() if not k.endswith('_indices')}
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={}
        for label,a,u,r,key in (('v19','A0','U0',71,'R71'),('R74','A1','U0',74,'R74'),('R76','A0','U1',76,'R76')):
            reconstructed=np.median(np.column_stack([values[a],values['L'],values[u]]),axis=1)
            old.exact(reconstructed,values[label],label+' saved cell')
            metrics[label]=score(reconstructed)
            assert metrics[label]==records[r]['stages'][s]['metrics'][key],label
        for label in defaults:
            metrics[label]=score(values[label]);assert metrics[label]==records[71]['stages'][s]['metrics'][label],label
        candidate=np.median(np.column_stack([values['A1'],values['L'],values['U1']]),axis=1)
        assert candidate.dtype==np.float64 and candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>0).all()
        metrics['R77']=score(candidate);old.previous.save_array(folder/'predictions.npy',candidate)
        old.exact(candidate,unit_run.vector(folder/'predictions.npy',n),'Saved R77 unchanged')
        diff=old.previous.prior.log_run.differences
        interaction={k:(metrics['R77'][k+'_mae']-metrics['R76'][k+'_mae'])-(metrics['R74'][k+'_mae']-metrics['v19'][k+'_mae']) for k in ('overall','weighted')}
        assert frames=={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        item={'stage':s,'status':'completed','fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],
            'original_frame_sha256':frames,'query_target_sha256':old.original.value_sha(yall[qi],'<f8'),
            'metrics':metrics,'differences':{label:diff(metrics[label],metrics['R77']) for label in ('R76','v19','R74','v18')},
            'interaction_TWD':interaction,'interaction_is_descriptive_not_causal':True,
            'changed_final_rows_vs_R76':int(np.count_nonzero(candidate!=values['R76'])),
            'three_saved_cells_bitwise_exact':True,'controls_full_metrics_exact':True,
            'new_NN_fits':0,'model_loads':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,
            'prediction_sha256':sha(folder/'predictions.npy'),'elapsed_seconds':time.monotonic()-start,
            'artifact_sha256':{rel(p):sha(p) for p in folder.iterdir() if p.is_file()}}
        write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'R77_minus_R76':item['differences']['R76']['overall'],
            'R77_minus_v19':item['differences']['v19']['overall'],'interaction_TWD':interaction}),flush=True)
    assert all(sha(ROOT/n)==h for n,h in source.items()) and sha(DOCUMENT)==plan_sha
    result.update(status='three_periods_completed_no_adoption_decision',elapsed_seconds=time.monotonic()-tick,
        protected_after={n:sha(ROOT/n) for n in protected},selector_sha256_after=sha(selector))
    assert result['protected_after']==protected and result['selector_sha256_after']==selector_sha
    write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'result_sha256':sha(OUT/'results.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true',required=True);p.add_argument('--plan-sha',required=True);a=p.parse_args()
    assert sha(DOCUMENT)==a.plan_sha and not OUT.exists(),'Plan changed or prior attempt exists'
    OUT.mkdir();tick=time.monotonic();started=old.now();code=0;failure=None
    write(OUT/'started.json',{'command':sys.argv,'started_at_utc':started,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    try:run(a.plan_sha)
    except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    write(OUT/'execution.json',{'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
