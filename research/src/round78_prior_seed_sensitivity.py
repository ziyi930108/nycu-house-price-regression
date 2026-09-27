"""R78 prior-seed sensitivity; area seed42 fixed, no model fitting or seed selection."""
from pathlib import Path
import argparse,contextlib,json,pickle,sys,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import candidate_diagnostics_v3 as diag

ROOT=old.ROOT;OUT=ROOT/'outputs/round78';DOC=ROOT/'reports/round78_plan.md'
STAGES=old.STAGES;SEEDS=(42,137,2026)
sha,rel,write=old.sha,old.rel,old.write
PINS={71:'8c9ddd281fb31df324b7aba4febd40c06cbee073ab45704b03b30176752ca833',
      74:'428f497383872490dfcc60051f1d25538ce69f845ec79f47c5017fcd50ddb717',
      76:'a7ea0219deec6aacaf43be9abd9b62e41b0cd720d09274d6fa81c5fa34d81631',
      77:'6e9b837a9bfe5918cb5200831ae31268d3ad204aec21430da1b0577de2be8337'}


def run(plan_sha):
    tick=time.monotonic();inputs={}
    def checked(n,h):
        n=n.replace(chr(92),'/');p=(ROOT/n).resolve();assert p.is_relative_to(ROOT.resolve()) and p.is_file(),n
        assert n not in inputs or inputs[n]==h,n
        if n not in inputs:assert sha(p)==h,n
        inputs[n]=h;return p
    def small(n,h=None,receipt=True):
        p=ROOT/n;checked(n,sha(p) if h is None else h)
        if receipt:
            obj=old.locked(p);checked(rel(p.with_suffix('.sha256')),sha(p.with_suffix('.sha256')));return obj
        return json.loads(p.read_text(encoding='utf8'))
    def artifact(a,n):return checked(n,a['artifact_sha256'][n])
    checked(rel(DOC),plan_sha)
    p48,p52=[small(f'outputs/round{r}/plan.json') for r in (48,52)]
    records={r:small(f'outputs/round{r}/results.json',h) for r,h in PINS.items()}
    diagnostics={r:small(f'reports/round{r}_diagnostics.json') for r in (76,77)}
    assert all(a['checked_stages']==list(STAGES) for a in records.values())
    source=old.previous.merge(*(r['source_sha256'] for r in records.values()),
        {rel(p):sha(p) for p in (Path(__file__),Path(diag.__file__),Path(diag.common.__file__),Path(diag.SCORER.__file__))})
    for n,h in source.items():checked(n,h)
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};assert len(protected)==22
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    selector=ROOT/'outputs/current_private_candidate.json';selector_sha=sha(selector)
    selection=json.loads(selector.read_text(encoding='utf8'));assert selection['private_priority']=='outputs/submission_v20_trial.csv'
    assert protected[selection['private_priority']]==selection['sha256'] and selection['private_mae'] is None
    mix=small('outputs/error_budget_20260926/round46_diagnostics.json',receipt=False)
    assert mix['minimum_historical_cell_rows']==200
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float)
    counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    assert counts.sum()==47073 and np.isclose(weights.sum(),1)
    for n in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT):checked(n,p48['input_sha256'][n])
    bindings={}
    for s in STAGES:
        a71,a74,a76,a77=[records[r]['stages'][s] for r in (71,74,76,77)]
        b={'A0':{},'v19':{},'prior':{},'A1_42':f'outputs/round74/{s}/nn_spatial.npy',
           'area':f'outputs/round74/{s}/model/predictions_twd.npy','U1':f'outputs/round76/{s}/candidate_unit_spatial.npy',
           'U0':p48['bindings'][s]['unit_spatial'],'L':p52['bindings'][s]['recency_log']}
        for seed in SEEDS:
            k=str(seed);b['A0'][k]=f'outputs/round71/{s}/seed{seed}/nn_spatial.npy';artifact(a71,b['A0'][k])
            b['v19'][k]=a71['seed_prediction_paths'][k];artifact(a71,b['v19'][k]);assert inputs[b['v19'][k]]==a71['seed_prediction_sha256'][k]
            b['prior'][k]=a71['native_seed_bindings'][k]['path'];checked(b['prior'][k],a71['native_seed_bindings'][k]['sha256'])
        for key,a in (('A1_42',a74),('area',a74),('U1',a76)):artifact(a,b[key])
        for key,p in (('U0',p48),('L',p52)):checked(b[key],p['input_sha256'][b[key]])
        for key in ('old_unit','reference','state','query_geo'):
            b[key]=p48['bindings'][s][key];checked(b[key],p48['input_sha256'][b[key]])
        for r in (76,77):
            n=f'outputs/round{r}/{s}/predictions.npy';b[f'v{r-56}_42']=n;artifact(records[r]['stages'][s],n)
        for kind in ('fit','query'):
            n=f'outputs/round71/{s}/{kind}_indices.npy';b[kind+'_indices']=n;artifact(a71,n)
            for r in (74,76,77):
                other=f'outputs/round{r}/{s}/{kind}_indices.npy';artifact(records[r]['stages'][s],other);assert inputs[other]==inputs[n]
        bindings[s]=b
    write(OUT/'plan.json',{'round':78,'kind':'conditional_prior_seed_sensitivity','seeds':list(SEEDS),'fixed_area_seed':42,
        'stage_order':list(STAGES),'text_plan_sha256':plan_sha,'split_contract':p48['split_contract'],
        'source_sha256':source,'input_sha256_before_scoring':dict(inputs),'bindings':bindings,'protected_before':protected,
        'selector_sha256_before':selector_sha,'planned_reference_predict_calls':6,'planned_spatial_predict_calls':6,
        'new_model_fits':0,'model_predict_calls':0,'seed_selection':False,'seed_average':False,'new_CSV':False})
    Xall,yall=old.previous.load_training_rows(p48)
    result={'round':78,'kind':'conditional_prior_seed_sensitivity','status':'running','seeds':list(SEEDS),'fixed_area_seed':42,
        'checked_stages':[],'stages':{},'source_sha256':source,'input_sha256':inputs,'text_plan_sha256':plan_sha,
        'protected_before':protected,'selector_sha256_before':selector_sha,'new_model_fits':0,'model_loads':0,
        'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,'seed_selection':False,'seed_average':False,
        'fresh_blind_validation':False,'full_model_multiseed_training':False,'new_CSV':False,'new_bootstrap':False}
    for s in STAGES:
        start=time.monotonic();folder=OUT/s;folder.mkdir();b=bindings[s];c=p48['split_contract'][s]
        fi,qi=[np.load(ROOT/b[k+'_indices'],allow_pickle=False) for k in ('fit','query')]
        for k,v in (('fit',fi),('query',qi)):
            assert v.dtype==np.int64 and v.shape==(c[k+'_rows'],) and (np.diff(v)>0).all() and v.min()>=0 and v.max()<len(Xall)
            assert old.original.value_sha(v,'<i8')==c[k+'_indices_content_sha256'];old.previous.save_array(folder/f'{k}_indices.npy',v)
        assert not np.intersect1d(fi,qi).size;n=len(qi);y=yall[qi]
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
        frames={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        assert all(frames==records[r]['stages'][s]['original_frame_sha256'] for r in PINS)
        score=lambda v:old.original.score(y,v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        vec=lambda name:old.vector(ROOT/name,n)
        L,U0,U1,area,unit=[vec(b[k]) for k in ('L','U0','U1','area','old_unit')]
        with (ROOT/b['reference']).open('rb') as f:reference=pickle.load(f)
        with (ROOT/b['state']).open('rb') as f:state=pickle.load(f)
        assert state['state_summary']['fit_X_sha256']==frames['fit_X'] and not state['state_summary']['query_used_in_build']
        assert state['parameters']==p48['spatial_parameters'] and reference['fit_report']['fitting_rows']==len(fi)
        geo=pd.read_pickle(ROOT/b['query_geo'])
        assert old.spatial.frame_sha(geo)==p48['bindings'][s]['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
        groups=diag.common.norm(Xq.address_city)+'|'+pd.Series(diag.common.group_labels(X,Xq));hist=groups.value_counts()
        supported=hist.reindex(weights.index,fill_value=0)>=200;coverage=float(weights[supported].sum());sw=weights[supported]/coverage
        seeds={}
        for seed in SEEDS:
            k=str(seed);sf=folder/f'seed{seed}';sf.mkdir();A0=vec(b['A0'][k]);v19=np.median(np.column_stack([A0,L,U0]),axis=1)
            old.exact(v19,vec(b['v19'][k]),'saved v19 seed '+k)
            m19=score(v19);assert m19==records[71]['stages'][s]['additional_seed_metrics'][k]
            if seed==42:
                A1=vec(b['A1_42']);calls=None
            else:
                B=.75*(.5*area+.5*vec(b['prior'][k]))+.25*unit
                p,diagnostic,rr=old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
                A1,detail,sr=old.spatial.predict(state,Xq,geo,B,p,diagnostic)
                fallback=~detail['applied'].to_numpy(dtype=bool);assert A1[fallback].tobytes()==p[fallback].tobytes()
                for name,a in (('base',B),('reference',p),('nn_spatial',A1)):old.previous.save_array(sf/(name+'.npy'),a)
                for name,a in (('reference_diagnostics',diagnostic),('spatial_diagnostics',detail)):
                    with (sf/(name+'.pkl')).open('xb') as f:pickle.dump(a,f,protocol=5)
                calls={'reference_report':rr,'spatial_report':sr};result['reference_predict_calls']+=1;result['spatial_predict_calls']+=1
            vectors={'v19':v19,'v20':np.median(np.column_stack([A0,L,U1]),axis=1),'v21':np.median(np.column_stack([A1,L,U1]),axis=1)}
            metrics={'v19':m19,**{label:score(vectors[label]) for label in ('v20','v21')}}
            if seed==42:
                for label,r in (('v20',76),('v21',77)):
                    old.exact(vectors[label],vec(b[label+'_42']),label+' seed42 control');assert metrics[label]==records[r]['stages'][s]['metrics'][f'R{r}']
            condition={label:float((pd.Series(np.abs(v-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for label,v in vectors.items()}
            bands=[]
            for name,lo,hi in diag.BANDS:
                mask=(y>=lo)&(y<hi);bands.append({'band':name,'rows':int(mask.sum()),**{label:float(np.abs(v[mask]-y[mask]).mean()) if mask.any() else None for label,v in vectors.items()}})
            if seed==42:
                for r,label,key in ((76,'v19','v19'),(76,'v20','R76'),(77,'v21','R77')):
                    saved=diagnostics[r]['stages'][s];mix0=saved['test_composition']
                    assert condition[label]==mix0['models'][key] and coverage==mix0['test_mass_covered']
                    assert int(counts[~supported].sum())==mix0['unsupported_test_rows']
                    for actual,prior in zip(bands,saved['price_bands']):assert actual['band']==prior['band'] and actual['rows']==prior['rows'] and actual[label]==prior[key]
            diff=old.previous.prior.log_run.differences
            differences={'v20_minus_v19':diff(metrics['v19'],metrics['v20']),'v21_minus_v20':diff(metrics['v20'],metrics['v21']),'v21_minus_v19':diff(metrics['v19'],metrics['v21'])}
            for label,v in vectors.items():
                assert v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>0).all();old.previous.save_array(sf/(label+'.npy'),v)
            seeds[k]={'prior_seed':seed,'area_seed':42,'metrics':metrics,'differences':differences,'test_composition_MAE':condition,
                'test_composition_differences_TWD':{'v20_minus_v19':condition['v20']-condition['v19'],'v21_minus_v20':condition['v21']-condition['v20'],'v21_minus_v19':condition['v21']-condition['v19']},
                'price_bands':bands,'saved_v19_full_metrics_exact':True,'seed42_R76_R77_controls_exact':seed==42,
                'new_reference_spatial_calls':int(seed!=42),'pipeline_reports':calls,
                'artifact_sha256':{rel(p):sha(p) for p in sf.iterdir() if p.is_file()}}
        assert frames=={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        item={'stage':s,'fit_rows':len(fi),'query_rows':n,'seeds':seeds,'original_frame_sha256':frames,
            'query_target_sha256':old.original.value_sha(y,'<f8'),'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),
            'all_prior_seeds_evaluated':True,'elapsed_seconds':time.monotonic()-start}
        write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'overall_differences':{k:{c:d['overall']['delta_TWD'] for c,d in v['differences'].items()} for k,v in seeds.items()}}),flush=True)
    assert result['reference_predict_calls']==result['spatial_predict_calls']==6 and all(sha(ROOT/n)==h for n,h in source.items())
    assert sha(DOC)==plan_sha and sha(selector)==selector_sha
    assert {rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}==set(protected)-{'runs/versions.json'}
    after={n:sha(ROOT/n) for n in protected};assert after==protected
    result.update(status='completed_conditional_prior_seed_sensitivity',elapsed_seconds=time.monotonic()-tick,
        protected_after=after,selector_sha256_after=sha(selector))
    write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.add_argument('--plan-sha',required=True);args=parser.parse_args()
    assert not OUT.exists() and sha(DOC)==args.plan_sha
    OUT.mkdir();started=old.now();tick=time.monotonic();code=0;failure=None
    write(OUT/'started.json',{'command':sys.argv,'source_sha256':sha(Path(__file__)),'plan_sha256':args.plan_sha,'started_at_utc':started,'automatic_retry':False})
    with (OUT/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:run(args.plan_sha)
        except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    event={'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'started_at_utc':started,'completed_at_utc':old.now(),
        'source_sha256':sha(Path(__file__)),'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
    write(OUT/'execution.json',event);print(json.dumps(event),flush=True);raise SystemExit(code)
