"""Two preregistered serial experiments; fit/query state and scoring separated."""
from __future__ import annotations
import argparse
import json
import pickle
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round104_composition as comp
import round100_temporal_prior_evaluate_v2 as metric

PLAN='reports/round109_110_plan.md'
REQUEST='outputs/research_round109_110_20261008/request.json'
SEAL='outputs/research_round109_110_20261008/seal.json'
TRAINER='src/round109_tail_parameter_average_train.py'
HELPER='src/round110_target_borders_unit.py'


def protected():
    names=set(c.protected_hashes())
    names.update(c.rel(p) for p in (c.ROOT/'outputs').glob('submission_v*.csv'))
    return {p:c.sha(p) for p in sorted(names)}


def unit_chain(ctx,native):
    metric.prices(native,(len(ctx['query_X']),),'unit native')
    base=.75*native+.25*ctx['fixed']['lgb']
    ref,diag,_=comp.old.original.predict_reference(ctx['reference'],ctx['query_X'],base,variant='trend',return_diagnostics=True)
    spatial,detail,_=comp.old.spatial.predict(ctx['spatial_state'],ctx['query_X'],ctx['geo'],base,ref,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback],ref[fallback],'unit spatial fallback')
    pred=np.median(np.column_stack([ctx['original']['spatial'],ctx['fixed']['H'],spatial]),axis=1)
    comp.verify_unchanged(ctx)
    return {'native':native,'base':base,'reference':ref,'spatial':spatial,'median':pred}


def freeze():
    c.need(not c.safe(REQUEST).exists(),'Preserve existing request')
    review=c.read('outputs/research_round109_110_20261008/preflight_review.json')
    c.need(review['status']=='passed','Independent preflight required')
    c.verify_hashes(review['source_sha256'])
    p79=c.read('outputs/round79/plan.json');r79=c.read('outputs/round79/results.json')
    r63=c.read('outputs/round63/results.json');r83=c.read('outputs/round83/results.json')
    labels=c.read('outputs/round100_inputs/manifest.json')
    inputs={p:c.sha(p) for p in ('outputs/round79/plan.json','outputs/round79/results.json',
                                'outputs/round63/results.json','outputs/round83/results.json',
                                'outputs/round100_inputs/manifest.json','reports/feature_schema.json')}
    # Read only in evaluate(); bind the original label identity before any fit.
    inputs[c.RAW]=p79['input_sha256'][c.RAW]
    for name in ('preflight_review.json','R109_average_selfcheck_execution.json','R110_helper_selfcheck_execution.json'):
        path='outputs/research_round109_110_20261008/'+name;inputs[path]=c.sha(path)
        if name!='preflight_review.json':c.need(c.read(path)['exit_code']==0,'Formal selfcheck must pass')
    specs={}
    for s in c.STAGES:
        cs=comp.specification(s);inputs.update(cs['input_sha256'])
        prepared=f'outputs/round63/{s}/prepared.npz'
        inputs[prepared]=r63['stages'][s]['artifact_sha256'][prepared]
        native=f'outputs/round79/{s}/model/native_predictions.npy'
        inputs[native]=r79['stages'][s]['artifact_sha256'][native]
        v24=f'outputs/round83/{s}/predictions.npy'
        inputs[v24]=r83['stages'][s]['artifact_sha256'][v24]
        treegeo=p79['bindings'][s]['tree_geo'];inputs[treegeo]=p79['input_sha256'][treegeo]
        for rec in labels['stages'][s]['arrays'].values():inputs[rec['path']]=rec['sha256']
        specs[s]={'composition':cs,'prepared':c.record(prepared),'native_unit':c.record(native),
                  'tree_geo':c.record(treegeo),'v24':c.record(v24),'fit_labels':labels['stages'][s]['arrays'],
                  'date_weight_sha256':p79['bindings'][s]['preparation']['train_sample_weight_sha256']}
    c.verify_hashes(inputs)
    sources=c.source_hashes(__file__,TRAINER,HELPER)
    c.safe(REQUEST).parent.mkdir(exist_ok=True)
    req={'status':'frozen_before_six_fits','created_at_utc':c.now(),'plan':c.record(PLAN),'stages':specs,
         'source_sha256':sources,'input_sha256':inputs,'protected_before':protected(),
         'new_fits':6,'query_scores_computed':False,'public_tuning':False,'private_score':None}
    c.write(REQUEST,req)
    return req


def request():
    r=c.read(REQUEST);c.need(r['status']=='frozen_before_six_fits','Request status')
    c.need(c.record(PLAN)==r['plan'],'Scientific plan changed')
    c.verify_hashes(r['source_sha256']);c.need(protected()==r['protected_before'],'Protected source changed')
    return r


def stage(s):
    r=request();spec=r['stages'][s];tick=time.monotonic()
    ctx=comp.load_context(spec['composition']);nf,nq=len(ctx['fit_X']),len(ctx['query_X'])
    fit_arrays=spec['fit_labels']
    c.exact(c.load_array(fit_arrays['fit_indices']),ctx['indices']['fit_indices'],'Fit y canonical indices')
    c.exact(c.load_array(fit_arrays['fit_source_cache_indices']),ctx['indices']['fit_source_cache_indices'],'Fit y raw indices')
    y=c.load_array(fit_arrays['fit_y'])
    c.need(y.dtype==np.float64 and y.shape==(nf,) and (y>0).all(),'Fit-only y')
    for key in ('prepared','native_unit','tree_geo'):
        c.need(c.sha(spec[key]['path'])==spec[key]['sha256'],'Input identity '+key)
    with np.load(c.safe(spec['prepared']['path']),allow_pickle=False) as z:
        c.need(set(z.files)==set(c.ARRAY_NAMES),'Exactly nine NN keys; no query labels')
        w=z['train_sample_weight'].copy()
        c.exact(z['train_y_million'],np.asarray(y/1e6,dtype=np.float32),'NN/tree fit target alignment')
    c.need(w.dtype==np.float32 and w.shape==(nf,) and (w>0).all()
           and c.values_sha(w)==spec['date_weight_sha256'],'Saved fit-date weights')
    native_control=np.load(c.safe(spec['native_unit']['path']),allow_pickle=False)
    original_unit=unit_chain(ctx,native_control)
    c.exact(original_unit['spatial'],ctx['fixed']['U'],'Original v22 unit replay')
    c.exact(original_unit['median'],ctx['original']['median'],'Original v22 final replay')
    # R109 and R110 are independent. No candidate results enter the other fit.
    folder=c.ROOT/'outputs/round109'/s;folder.mkdir(parents=True,exist_ok=False)
    cmd=[str(c.ROOT/'.venv_torch/Scripts/python.exe'),'-u','-B','-X','utf8',str(c.safe(TRAINER)),
         '--data',str(c.safe(spec['prepared']['path'])),'--out',str(folder/'model')]
    with (folder/'train.log').open('x',encoding='utf8') as log:
        p=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    c.need(p.returncode==0,'R109 trainer failed; preserve run')
    meta=c.read(folder/'model/training.json')
    c.need(meta['saved_model_reload_exact'] and meta['reload_rows']==nq and not meta['query_labels_used'],'R109 reload/scope')
    c.need(meta['input_sha256']==spec['prepared']['sha256'],'R109 inputs changed')
    native=np.load(folder/'model/predictions_twd.npy',allow_pickle=False)
    chain,_,_=comp.apply(ctx,native)
    rec109={k:c.save_array(folder/(k+'.npy'),v) for k,v in chain.items()}
    epoch30=np.load(folder/'model/predictions_epoch30_twd.npy',allow_pickle=False)
    same_trajectory,_,_=comp.apply(ctx,epoch30)
    controls={k:c.save_array(folder/('epoch30_'+k+'.npy'),v) for k,v in same_trajectory.items()}
    c.write(folder/'stage.json',{'stage':s,'layers':rec109,'model_metadata':c.record(folder/'model/training.json'),
                             'same_trajectory_epoch30_layers':controls,
                             'sealed_models':[c.record(folder/'model/model.pt'),c.record(folder/'model/model_epoch30.pt')],
                             'fit_rows':nf,'query_rows':nq,'fit_seconds':meta['fit_seconds'],'query_labels_available':False})
    folder=c.ROOT/'outputs/round110'/s;folder.mkdir(parents=True,exist_ok=False)
    tree=pd.read_pickle(c.safe(spec['tree_geo']['path']))
    fi,qi=ctx['indices']['fit_indices'],ctx['indices']['query_indices']
    G=tree.iloc[fi].reset_index(drop=True);Gq=tree.iloc[qi].reset_index(drop=True);del tree
    cats=c.read('reports/feature_schema.json')['categorical_columns']
    b=c.read('outputs/round79/plan.json')['bindings'][s]['R19_metadata']
    c.need(comp.old.spatial.frame_sha(G)==b['fit_geo_sha256'] and comp.old.spatial.frame_sha(Gq)==b['query_geo_sha256'],'Cutoff tree geography')
    import round110_target_borders_unit as helper
    with (folder/'train.log').open('x',encoding='utf8') as log:
        import contextlib
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            native,meta=helper.fit_predict(ctx['fit_X'],y,G,ctx['query_X'],Gq,cats,w,folder/'model')
    chain=unit_chain(ctx,native)
    rec110={k:c.save_array(folder/(k+'.npy'),v) for k,v in chain.items()}
    c.write(folder/'stage.json',{'stage':s,'layers':rec110,'model_metadata':c.record(folder/'model/metadata.json'),
                             'sealed_models':[c.record(folder/'model/model.cbm')],
                             'fit_rows':nf,'query_rows':nq,'fit_seconds':meta['fit_seconds'],'query_labels_available':False})
    comp.verify_unchanged(ctx);c.need(protected()==r['protected_before'],'Protected artifacts changed')
    print(json.dumps({'stage':s,'status':'both_candidates_sealed_unscored','elapsed_seconds':time.monotonic()-tick}),flush=True)


def train():
    r=request();tick=time.monotonic();failure=None
    try:
        for s in c.STAGES:
            cmd=[str(c.ROOT/'.venv/Scripts/python.exe'),'-u','-B','-X','utf8',__file__,'--stage',s]
            p=subprocess.run(cmd,cwd=c.ROOT,check=False)
            c.need(p.returncode==0,'Stage failed, no automatic retry: '+s)
        records={str(n):{s:c.record(f'outputs/round{n}/{s}/stage.json') for s in c.STAGES} for n in (109,110)}
        c.verify_hashes(r['source_sha256']);c.verify_hashes(r['input_sha256'])
        c.need(protected()==r['protected_before'],'Protected artifacts changed')
        c.write(SEAL,{'status':'six_models_all_query_predictions_sealed_unscored','request':c.record(REQUEST),
                      'records':records,'elapsed_seconds':time.monotonic()-tick,'query_scores_computed':False})
    except Exception:failure=traceback.format_exc();raise
    finally:c.write('outputs/research_round109_110_20261008/training_execution.json',
                    {'exit_code':1 if failure else 0,'failure':failure,'elapsed_seconds':time.monotonic()-tick})


def gates(stages):
    # Conditions frozen before model fits; trial status is distinct from promotion.
    promotion=metric.research_gate({s:{'metrics':{'T':v['metrics']['T'],'v22':v['metrics']['v22']},
                                      'comparison':v['comparison']} for s,v in stages.items()})
    promotion.pop('C_is_mechanism_control_not_alternative_candidate',None)
    promotion.pop('new_CSV',None)
    latest=stages['confirmation']['metrics']
    parts={'latest_both_no_worse':all(latest['T'][k]<=latest['v22'][k] for k in ('overall_mae','weighted_mae')),
           'each_metric_each_window_within_quarter_percent':all(v['metrics']['T'][k]<=1.0025*v['metrics']['v22'][k]
               for v in stages.values() for k in ('overall_mae','weighted_mae')),
           'at_least_two_windows_overall_improve':sum(v['metrics']['T']['overall_mae']<v['metrics']['v22']['overall_mae'] for v in stages.values())>=2,
           'no_large_group_or_quarter_above_five_percent':all(not v['comparison']['quarter_or_large_group_deterioration_above_5pct'] for v in stages.values())}
    return {'promotion':promotion,'exploratory_submission':{'components':parts,'passed':all(parts.values())},
            'private_score':None,'selected_submission':'v22','automatic_selection':False}


def evaluate():
    r=request();seal=c.read(SEAL);c.need(seal['request']==c.record(REQUEST),'Sealed request identity')
    c.verify_hashes(r['input_sha256'])
    tick=time.monotonic();results={str(n):{} for n in (109,110)}
    c.need(c.sha(c.RAW)==r['input_sha256'][c.RAW],'Evaluation label source changed')
    raw=pd.read_pickle(c.safe(c.RAW))
    for s in c.STAGES:
        ctx=comp.load_context(r['stages'][s]['composition'])
        y=metric.select_query_y(raw,ctx['indices']['query_source_cache_indices'])
        baseline=metric.descriptive_metrics(y,ctx['original']['median'],ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0])
        old=c.read('outputs/round79/results.json')['stages'][s]['metrics']['R79']
        c.need(all(baseline[k]==old[k] for k in old),'Original v22 metrics replay')
        vr=r['stages'][s]['v24'];c.need(c.record(vr['path'])==vr,'v24 input identity')
        v24=np.load(c.safe(vr['path']),allow_pickle=False)
        m24=metric.descriptive_metrics(y,v24,ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0])
        old24=c.read('outputs/round83/results.json')['stages'][s]['metrics']['R83']
        c.need(all(m24[k]==old24[k] for k in old24),'v24 metrics replay')
        for n in (109,110):
            sr=seal['records'][str(n)][s];c.need(c.record(sr['path'])==sr,'Stage seal identity')
            st=c.read(sr['path']);layers={k:c.load_array(v) for k,v in st['layers'].items()}
            c.need(all(c.record(v['path'])==v for v in st['sealed_models']),'Sealed model identity')
            m=metric.descriptive_metrics(y,layers['median'],ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0])
            metrics={'v22':baseline,'v24':m24,'T':m};comparison=metric.comparisons(metrics)
            pair,samples=metric.paired(y,{'v22':ctx['original']['median'],'T':layers['median']},ctx['fit_X'],ctx['query_X'],
                                       ctx['indices']['query_indices'],s,'v22')
            pair['caution']='Reused annual windows; saved v22 is not a matched retraining control. Conditional bootstrap does not include cumulative selection, refitting, or future shift, and is not Private probability.'
            folder=c.ROOT/f'outputs/round{n}/{s}/evaluation';folder.mkdir()
            for k,v in samples.items():c.save_array(folder/(k+'_samples.npy'),v)
            item={'metrics':metrics,'comparison':comparison,'comparison_vs_v24':metric.comparisons(metrics,'v24'),'paired':pair,
                  'branch_metrics':{k:metric.descriptive_metrics(y,v,ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0])
                                    for k,v in layers.items() if k!='median'}}
            if n==109:
                control={k:c.load_array(v) for k,v in st['same_trajectory_epoch30_layers'].items()}
                cm=metric.descriptive_metrics(y,control['median'],ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0])
                item['same_trajectory_epoch30_metrics']=cm
                item['comparison_vs_same_trajectory_epoch30']=metric.comparisons({'epoch30':cm,'T':m},'epoch30')
                item['epoch30_vs_saved_v22_max_abs_prediction_difference_TWD']=float(np.max(np.abs(control['median']-ctx['original']['median'])))
            c.write(folder/'results.json',item);results[str(n)][s]=item
            print(json.dumps({'round':n,'stage':s,'mae':m['overall_mae'],'delta':m['overall_mae']-baseline['overall_mae']}),flush=True)
    for n in (109,110):
        c.write(f'outputs/round{n}/results.json',{'status':'three_reused_windows_evaluated','request':c.record(REQUEST),
                'seal':c.record(SEAL),'stages':results[str(n)],'gates':gates(results[str(n)]),'Private_score':None})
    c.need(protected()==r['protected_before'],'Protected artifacts changed')
    c.write('outputs/research_round109_110_20261008/evaluation_execution.json',{'exit_code':0,'elapsed_seconds':time.monotonic()-tick,
            'query_labels_used_for_scoring_only':True,'training_already_finished_before_scoring':True,'protected_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--train',action='store_true')
    g.add_argument('--stage',choices=c.STAGES);g.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.train:train()
    elif a.stage:stage(a.stage)
    else:evaluate()
