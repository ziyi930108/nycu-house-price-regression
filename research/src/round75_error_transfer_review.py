"""R75: saved-vector error transfer from area NN to the fixed complete pipeline."""
from pathlib import Path
import argparse, json, sys, time, traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round3_evaluate as scorer
import round75_projection_diagnostics as projection

ROOT,STAGES=old.ROOT,old.STAGES
OUT=ROOT/'outputs/round75';DOCUMENT=ROOT/'reports/round75_plan.md'
LAYERS=('native_area','native_mean','base','reference','nn_spatial','predictions')
BANDS=(('<10m',0.,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,None))
sha,read,write,rel=old.sha,old.locked,old.write,old.rel
RECORD_SHA={71:'8c9ddd281fb31df324b7aba4febd40c06cbee073ab45704b03b30176752ca833',
            74:'428f497383872490dfcc60051f1d25538ce69f845ec79f47c5017fcd50ddb717'}


def close(a,b):
    return bool(np.isclose(a,b,rtol=1e-11,atol=1e-6))


def run(plan_sha,helper_sha):
    tick_all=time.monotonic();inputs={}
    assert sha(DOCUMENT)==plan_sha and sha(Path(projection.__file__))==helper_sha
    def checked(name,digest):
        name=name.replace(chr(92),'/');p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT.resolve()) and p.is_file(),name
        assert name not in inputs or inputs[name]==digest,name
        if name not in inputs:assert sha(p)==digest,name
        inputs[name]=digest;return p
    def small(name,expected=None):
        p=ROOT/name;v=read(p);checked(name,sha(p));checked(rel(p.with_suffix('.sha256')),sha(p.with_suffix('.sha256')))
        assert expected is None or sha(p)==expected,name
        return v
    def array(a,name):
        mapping={k.replace(chr(92),'/'):v for k,v in a['artifact_sha256'].items()}
        return np.load(checked(name,mapping[name]),allow_pickle=False)
    def vector(a,name,n):
        v=array(a,name)
        assert v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>0).all(),name
        return v
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};assert len(protected)==20
    protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    selector_sha=sha(ROOT/'outputs/current_private_candidate.json')
    p48=small('outputs/round48/plan.json');p52=small('outputs/round52/plan.json')
    records={r:small(f'outputs/round{r}/results.json',RECORD_SHA.get(r)) for r in (54,63,71,74)}
    assert all(r['checked_stages']==list(STAGES) for r in records.values())
    sources={}
    for mapping in (old.previous.sources(),
            {rel(p):sha(p) for p in (Path(__file__),Path(projection.__file__),Path(old.__file__),Path(old.original.__file__),Path(scorer.__file__))}):
        for name,digest in mapping.items():
            assert name not in sources or sources[name]==digest,name
            sources[name]=digest
    for name,digest in sources.items():checked(name,digest)
    for name in ('src/round3_evaluate.py','src/round39_city_road_experiment.py'):
        assert sha(ROOT/name)==records[74]['source_sha256'][name]==records[71]['source_sha256'][name]
    math_name='reports/round75_projection_selfcheck.json';math_path=checked(math_name,sha(ROOT/math_name))
    math=json.loads(math_path.read_text(encoding='utf8'))
    assert math['exit_code']==0 and math['source_sha256']==helper_sha and math['model_fits']==0 and not math['real_arrays_read']
    checked(rel(DOCUMENT),plan_sha)
    canonical={name:p48['input_sha256'][name] for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.previous.KT)}
    plan={'round':75,'diagnostic_only':True,'posthoc_reused_windows':True,'source_sha256':sources,'text_plan_sha256':plan_sha,
          'stage_order':list(STAGES),'layers':list(LAYERS),'split_contract':p48['split_contract'],'primary_comparison':'R74 minus R71 seed42 / v19',
          'native_area_control':'R54 saved native total-price prediction','all_predictions_in_TWD':True,
          'scope':'saved vectors and canonical X/y only; no models, fitted states or geographic objects loaded',
          'new_fits':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,'new_CSV':False,
          'median_identity':'median(spatial,log,unit)=clip(spatial,min(log,unit),max(log,unit)); same saved bounds for both candidates',
          'transition_cells':'below/within-including-boundaries/above, nine cells when lower<upper; equal-bounds separate tenth cell',
          'paired_error_sign_groups':'spatial improved / unchanged / worsened, threshold exactly zero',
          'cohorts':{'strata':list(scorer.GROUPS),'forecast_quarters':scorer.QUARTERS,'price_bands':BANDS},
          'empty_cohorts_retained':True,'threshold_seed_weight_or_new_candidate_search':False,
          'canonical_input_sha256':canonical,'input_sha256_before_arrays':dict(inputs),
          'protected_before':protected,'selector_sha256_before':selector_sha}
    write(OUT/'plan.json',plan)
    for name,digest in canonical.items():checked(name,digest)
    Xall,yall=old.previous.load_training_rows(p48)
    assert yall.dtype==np.float64 and np.isfinite(yall).all() and (yall>0).all()
    result={'round':75,'status':'running','diagnostic_only':True,'posthoc_reused_windows':True,'checked_stages':[],'stages':{},
            'source_sha256':sources,'text_plan_sha256':plan_sha,'plan_sha256':sha(OUT/'plan.json'),
            'new_fits':0,'model_loads':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,
            'new_CSV':False,'new_candidate_created':False,'protected_before':protected,'selector_sha256_before':selector_sha,
            'primary_comparison':'R74 minus R71 seed42 / v19','layers':list(LAYERS),
            'interpretation':'Paired MAE differences across saved processing stages; not a causal decomposition or new prediction rule'}
    for s in STAGES:
        tick=time.monotonic();folder=OUT/s;folder.mkdir();a54,a63,a71,a74=(records[r]['stages'][s] for r in (54,63,71,74))
        contract=p48['split_contract'][s];ind={};b=p48['bindings'][s];bs=p52['bindings'][s]
        for kind in ('fit','query'):
            assert a74['artifact_sha256'][f'outputs/round74/{s}/{kind}_indices.npy']==a71['artifact_sha256'][f'outputs/round71/{s}/{kind}_indices.npy']
            v=array(a74,f'outputs/round74/{s}/{kind}_indices.npy')
            assert v.dtype==np.int64 and v.shape==(contract[kind+'_rows'],) and (np.diff(v)>0).all() and v.min()>=0 and v.max()<len(Xall)
            assert old.original.value_sha(v,'<i8')==contract[kind+'_indices_content_sha256']
            for r in (54,63,71):assert np.array_equal(v,array(records[r]['stages'][s],f'outputs/round{r}/{s}/{kind}_indices.npy'))
            ind[kind]=v
        fi,qi=ind['fit'],ind['query'];assert not np.intersect1d(fi,qi).size
        X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);y=yall[qi];n=len(y)
        frames={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        assert frames==a71['original_frame_sha256']==a74['original_frame_sha256']
        prior_name=f'outputs/round63/{s}/model/predictions_twd.npy'
        assert a71['primary_seed']==42 and a71['native_seed_bindings']['42']=={'path':prior_name,'sha256':a63['artifact_sha256'][prior_name]}
        prior=vector(a63,prior_name,n)
        controls={'native_area':vector(a54,f'outputs/round54/{s}/model/predictions_twd.npy',n)}
        candidates={'native_area':vector(a74,f'outputs/round74/{s}/model/predictions_twd.npy',n)}
        for layer in LAYERS[1:]:
            controls[layer]=vector(a71,f'outputs/round71/{s}/seed42/{layer}.npy',n)
            candidates[layer]=vector(a74,f'outputs/round74/{s}/{layer}.npy',n)
        original_top=vector(a71,f'outputs/round71/{s}/predictions.npy',n)
        old.exact(original_top,controls['predictions'],'R71 top-level vs fixed seed42')
        assert a71['prediction_sha256']==a71['artifact_sha256'][f'outputs/round71/{s}/seed42/predictions.npy']
        shared_names=[prior_name,f'outputs/round54/{s}/model/predictions_twd.npy',b['old_unit'],b['reference'],b['state'],b['query_geo'],bs['recency_log'],b['unit_spatial']]
        shared={}
        for name in shared_names:
            assert records[71]['input_sha256'][name]==records[74]['input_sha256'][name],name
            shared[name]=records[74]['input_sha256'][name]
        unit=old.vector(checked(b['old_unit'],p48['input_sha256'][b['old_unit']]),n)
        log=old.vector(checked(bs['recency_log'],p52['input_sha256'][bs['recency_log']]),n)
        unit_branch=old.vector(checked(b['unit_spatial'],p48['input_sha256'][b['unit_spatial']]),n)
        lower,upper=np.minimum(log,unit_branch),np.maximum(log,unit_branch)
        for label,vectors in (('control',controls),('candidate',candidates)):
            old.exact(vectors['native_mean'],.5*vectors['native_area']+.5*prior,label+' native mean')
            old.exact(vectors['base'],.75*vectors['native_mean']+.25*unit,label+' fixed base')
            old.exact(vectors['predictions'],np.median(np.column_stack([vectors['nn_spatial'],log,unit_branch]),axis=1),label+' median identity')
            old.exact(vectors['predictions'],np.clip(vectors['nn_spatial'],lower,upper),label+' clip identity')
        score=lambda v:old.original.score(y,v,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        layer_metrics={layer:{'control':score(controls[layer]),'candidate':score(candidates[layer])} for layer in LAYERS}
        for layer,m in layer_metrics.items():m['differences']=old.previous.prior.log_run.differences(m['control'],m['candidate'])
        assert layer_metrics['predictions']['control']==a71['metrics']['R71']==a74['metrics']['v19']
        assert layer_metrics['predictions']['candidate']==a74['metrics']['R74']
        assert layer_metrics['native_area']['control']['overall_mae']==a54['native_MAE_diagnostic']['R54']==a74['native_MAE_diagnostic']['R54_area']
        assert layer_metrics['native_area']['candidate']['overall_mae']==a74['native_MAE_diagnostic']['R74_area']
        errors={layer:(np.abs(controls[layer]-y),np.abs(candidates[layer]-y)) for layer in LAYERS}
        def scope(mask):
            count=int(mask.sum());summary={}
            for layer,(e0,e1) in errors.items():
                total0=float(e0[mask].sum(dtype=np.float64));total1=float(e1[mask].sum(dtype=np.float64))
                paired=float((e1[mask]-e0[mask]).sum(dtype=np.float64))
                summary[layer]={'rows':count,'control_MAE_TWD':None if not count else float(e0[mask].mean()),
                    'candidate_MAE_TWD':None if not count else float(e1[mask].mean()),
                    'paired_delta_TWD':None if not count else paired/count,
                    'control_error_sum_TWD':total0,'candidate_error_sum_TWD':total1,'paired_error_sum_TWD':paired,
                    'contribution_to_stage_delta_TWD':paired/n}
            changes={f'{before}->{after}':None if not count else summary[after]['paired_delta_TWD']-summary[before]['paired_delta_TWD']
                     for before,after in zip(LAYERS[:-1],LAYERS[1:])}
            if count:assert close(sum(changes.values()),summary['predictions']['paired_delta_TWD']-summary['native_area']['paired_delta_TWD'])
            projected=None if not count else projection.projection_review(y[mask],controls['nn_spatial'][mask],candidates['nn_spatial'][mask],
                      lower[mask],upper[mask],controls['predictions'][mask],candidates['predictions'][mask])
            if projected is not None:
                projected['input_scope_rows']=count;projected['parent_stage_rows']=n
                projected['contribution_denominator_note']='Helper overall contributions divide by this scope rows; added stage contributions divide by the full historical period rows'
                for cell in [projected['overall'],*projected['transition_cells'].values(),*projected['error_sign_groups'].values()]:
                    cell['row_fraction_of_scope']=cell['rows']/count;cell['row_fraction_of_stage']=cell['rows']/n
                    for layer in ('spatial','final'):
                        cell[layer]['contribution_to_stage_delta_TWD']=cell[layer]['error_delta_sum_TWD']/n
                    cell['transfer_contribution_to_stage_TWD']=cell['transfer_error_sum_TWD']/n
            return {'rows':count,'layers':summary,'adjacent_paired_delta_changes_TWD':changes,'projection':projected}
        labels=scorer.group_labels(X,Xq)
        day0=(pd.Timestamp(p48['splits'][s][0])-pd.Timestamp('2000-01-01')).days
        horizon=Xq['trade_day_index'].to_numpy(dtype=np.float64)-day0
        assert np.isfinite(horizon).all() and (horizon>=0).all()
        masks={'strata':{g:labels==g for g in scorer.GROUPS},
               'forecast_quarters':{q:(horizon>=lo)&(True if hi is None else horizon<=hi) for q,(lo,hi) in scorer.QUARTERS.items()},
               'price_bands':{name:(y>=lo)&(True if hi is None else y<hi) for name,lo,hi in BANDS}}
        full=scope(np.ones(n,dtype=bool));cohorts={}
        for family,groups in masks.items():
            assert np.all(sum(mask.astype(np.int8) for mask in groups.values())==1),family
            cohorts[family]={name:scope(mask) for name,mask in groups.items()}
            assert sum(v['rows'] for v in cohorts[family].values())==n
            for layer in LAYERS:
                for field in ('control_error_sum_TWD','candidate_error_sum_TWD','paired_error_sum_TWD','contribution_to_stage_delta_TWD'):
                    assert close(sum(v['layers'][layer][field] for v in cohorts[family].values()),full['layers'][layer][field]),(family,layer,field)
        for layer,m in layer_metrics.items():
            for side,field in (('control','control_MAE_TWD'),('candidate','candidate_MAE_TWD')):
                assert close(full['layers'][layer][field],m[side]['overall_mae'])
                for family in ('strata','forecast_quarters'):
                    for name,v in cohorts[family].items():
                        saved=m[side][family][name]
                        assert v['rows']==saved['rows']
                        assert (v['rows']==0 and saved['mae'] is None) or close(v['layers'][layer][field],saved['mae'])
        assert frames=={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        item={'stage':s,'fit_rows':len(fi),'query_rows':n,'dates':p48['splits'][s],
              'query_indices_content_sha256':old.original.value_sha(qi,'<i8'),'query_target_sha256':old.original.value_sha(y,'<f8'),
              'original_frame_sha256':frames,'shared_pipeline_recorded_input_sha256':shared,
              'shared_reference_state_geo_identity_compared_from_records_not_loaded':True,
              'layer_metrics':layer_metrics,'full':full,'cohorts':cohorts,
              'native_overall_and_complete_full_saved_metrics_exact':True,'all_cohort_rows_and_error_contributions_reconciled':True,
              'same_fixed_bounds_median_clip_identity_exact':True,'native_mean_and_base_identity_exact':True,
              'new_fits':0,'model_loads':0,'predict_calls':0,'elapsed_seconds':time.monotonic()-tick}
        write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'paired_MAE_deltas_TWD':{layer:m['differences']['overall']['delta_TWD'] for layer,m in layer_metrics.items()},
                          'elapsed_seconds':item['elapsed_seconds']}),flush=True)
    assert result['checked_stages']==list(STAGES) and all(sha(ROOT/n)==h for n,h in sources.items())
    assert sha(DOCUMENT)==plan_sha and sha(ROOT/'outputs/current_private_candidate.json')==selector_sha
    assert set(protected)-{'runs/versions.json'}=={rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    result.update(status='completed_saved_vector_error_transfer_diagnostic',elapsed_seconds=time.monotonic()-tick_all,
                  protected_after={n:sha(ROOT/n) for n in protected},selector_sha256_after=sha(ROOT/'outputs/current_private_candidate.json'),input_sha256=inputs)
    assert result['protected_before']==result['protected_after']
    write(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'new_fits':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--plan-sha',required=True);parser.add_argument('--helper-sha',required=True);args=parser.parse_args()
    for digest in (args.plan_sha,args.helper_sha):assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
    assert not OUT.exists(),'Existing evidence: no overwrite';OUT.mkdir();tick=time.monotonic();started=old.now();code=0;failure=None
    write(OUT/'started.json',{'round':75,'started_at_utc':started,'command':sys.argv,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    try:run(args.plan_sha,args.helper_sha)
    except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    write(OUT/'execution.json',{'round':75,'exit_code':code,'failure':failure,'started_at_utc':started,'completed_at_utc':old.now(),
          'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
