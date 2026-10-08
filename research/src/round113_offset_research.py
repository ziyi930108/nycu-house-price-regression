"""R113: fixed R107/R111/R112 at the saved February 2023 origin.

Fit-only preparation, all three model seals, then a separate scoring command.
No mutation of the August split constants; no CSV or full-data fitting here.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round113_offset_composition as comp
import round97_offset_evaluate as old_metric

PLAN='reports/round113_plan.md'
PLAN_SHA='a21116136190aadf94d89329dd1b9dd98afb62dd493ef6863b356f50ea01dc2c'
REQUEST='outputs/round113/request.json'
SEAL='outputs/round113/seal.json'
RESULTS='outputs/round113/results.json'
EVIDENCE='outputs/research_round113_20261008'
GATE=EVIDENCE+'/round113_release_gate.json'
ORIGIN='offset_2023_02_11'
CANDIDATES=(107,111,112)
TRAINERS={107:'src/round107_weighted_smooth_train.py',111:'src/round111_residual_prior_train.py'}
HELPER='src/round112_ordered_unit.py'
LAYERS=('native','base','reference','spatial','median')
CAUTION=('The February origin was previously used for R97 and overlaps the August windows. '
         'This is not fresh blind validation or an independent fourth vote. Saved controls are '
         'not matched refits. Single-seed conditional bootstrap omits cumulative selection, '
         'refitting and future shift; it is not Private probability.')


def protected():
    names=set(c.protected_hashes())
    names.update(c.rel(p) for p in (c.ROOT/'outputs').glob('submission_v*.csv'))
    return {p:c.sha(p) for p in sorted(names)}


def sources():
    return c.source_hashes(__file__,*TRAINERS.values(),HELPER)


def formal_checks():
    return {'R107_loss_selfcheck_execution.json':(107,TRAINERS[107],'--cpu-unit-check'),
            'R111_residual_selfcheck_execution.json':(111,TRAINERS[111],'--cpu-unit-check'),
            'R112_helper_selfcheck_execution.json':(112,HELPER,'--self-check'),
            'R113_driver_selfcheck_execution.json':(113,c.rel(__file__),'--self-check')}


def freeze():
    c.need(not c.safe(REQUEST).exists(),'Preserve existing R113 request')
    c.need(c.sha(PLAN)==PLAN_SHA,'Fixed scientific plan changed')
    own=sources();review_path=EVIDENCE+'/preflight_review.json';review=c.read(review_path)
    c.need(review['status']=='passed' and review['source_sha256']=={**own,PLAN:PLAN_SHA},
           'Independent R113 preflight must bind the complete current closure and fixed plan')
    c.verify_hashes(review['source_sha256'])
    spec=comp.specification();inputs=dict(spec['input_sha256']);inputs[review_path]=c.sha(review_path)
    # The context excludes the raw-price pickle. The coordinator binds its
    # historical identity as a scoring-only input; hashing does not decode y.
    original_seal=c.read(spec['original_seal']['path'])
    inputs[c.RAW]=original_seal['input_sha256'][c.RAW]
    inputs['outputs/round97/evaluation/results.json']=c.sha('outputs/round97/evaluation/results.json')
    for name,(number,entry,flag) in formal_checks().items():
        path=EVIDENCE+'/'+name;check=c.read(path)
        python=c.ROOT/('.venv_torch/Scripts/python.exe' if number in (107,111) else '.venv/Scripts/python.exe')
        command=[str(python),'-B','-X','utf8',str(c.safe(entry)),flag]
        c.need(check['exit_code']==0 and check['command']==command
               and check['source_sha256']==c.source_hashes(entry),'Formal selfcheck command/closure differs: '+name)
        inputs[path]=c.sha(path)
    c.need(inputs[c.RAW]=='9acf8c65bf00f7067e9ea9612cabe9a1524b020a7596c8da3ce3d620131e5c00',
           'Original query label source identity must be frozen without deserializing it')
    c.verify_hashes(inputs)
    c.safe(REQUEST).parent.mkdir(exist_ok=False)
    c.write(REQUEST,{'status':'frozen_before_three_offset_fits','created_at_utc':c.now(),
        'plan':c.record(PLAN),'source_sha256':own,'input_sha256':inputs,'composition':spec,
        'protected_before':protected(),'origin':ORIGIN,'cutoff':'2023-02-11','query_end':'2024-02-10',
        'fit_rows':272117,'query_rows':107169,'candidates':list(CANDIDATES),'new_fits':3,
        'query_scores_computed':False,'query_labels_used_for_fit':False,'Public_score':None,
        'Private_score':None,'selected_submission':'v22','new_CSV':False})


def request():
    r=c.read(REQUEST)
    c.need(r['status']=='frozen_before_three_offset_fits' and r['origin']==ORIGIN
           and r['candidates']==list(CANDIDATES),'Request scope differs')
    c.need(c.record(PLAN)==r['plan'] and r['plan']['sha256']==PLAN_SHA,'Plan identity differs')
    c.need(r['source_sha256']==sources(),'Request must bind exact current source closure')
    c.verify_hashes(r['source_sha256'])
    c.need(protected()==r['protected_before'],'Protected artifacts changed')
    return r


def model_records(folder):
    files=sorted(folder.iterdir())
    c.need(files and all(p.is_file() and p.suffix in {'.json','.npy','.pt','.cbm'} for p in files),
           'All flat model artifacts must be directly bound')
    return [c.record(p) for p in files]


def validate(number,record):
    c.need(number in CANDIDATES and c.record(record['path'])==record,'Stage identity differs')
    item=c.read(record['path'])
    c.need(item['stage']==ORIGIN and item['candidate_round']==number
           and item['fit_rows']==272117 and item['query_rows']==107169
           and item['query_labels_available'] is False,'Stage scope differs')
    c.need(c.record(item['model_metadata']['path'])==item['model_metadata'],'Model metadata identity')
    folder=c.ROOT/f'outputs/round113/models/R{number}'
    c.need(model_records(folder)==item['sealed_models'],'Native artifact identities differ')
    c.need(set(item['layers'])==set(LAYERS),'All five layers required')
    layers={k:c.load_array(v) for k,v in item['layers'].items()}
    for k,a in layers.items():
        c.need(a.dtype==np.float64 and a.shape==(107169,) and np.isfinite(a).all() and (a>=0).all(),
               'Invalid '+k+' price vector')
    name='native.npy' if number==112 else 'predictions_twd.npy'
    c.exact(layers['native'],np.load(folder/name,allow_pickle=False),'Native model/layer alignment')
    meta=c.read(item['model_metadata']['path'])
    if number!=112:
        c.need(meta['saved_model_reload_exact'] and meta['reload_rows']==107169
               and not meta['query_labels_used'],'NN full-query native reload/scope')
    else:
        c.need(meta['saved_model_reload_exact'] and meta['query_rows']==107169
               and not meta['query_labels_used'],'Ordered native reload/scope')
    return item,layers


def stage(number):
    r=request();c.verify_hashes(r['input_sha256']);ctx=comp.load_context(r['composition']);data=ctx['data']
    folder=c.ROOT/f'outputs/round113/R{number}';folder.mkdir(exist_ok=False)
    model=c.ROOT/f'outputs/round113/models/R{number}';model.parent.mkdir(exist_ok=True)
    if number in TRAINERS:
        command=[str(c.ROOT/'.venv_torch/Scripts/python.exe'),'-u','-B','-X','utf8',
                 str(c.safe(TRAINERS[number])),'--data',str(ctx['prior_npz']),'--out',str(model)]
        with (folder/'train.log').open('x',encoding='utf8') as log:
            p=subprocess.run(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        c.need(p.returncode==0,'NN failed; preserve evidence, no automatic retry')
        metadata=model/'training.json';meta=c.read(metadata)
        c.need(meta['input_sha256']==c.sha(ctx['prior_npz']),'NN input identity changed')
        native=np.load(model/'predictions_twd.npy',allow_pickle=False);chain=comp.chain_prior(ctx,native)
    else:
        import round112_ordered_unit as helper
        with (folder/'train.log').open('x',encoding='utf8') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                native,meta=helper.fit_predict(data.Xfit,ctx['yfit'],data.tree_fit,data.Xquery,
                    data.tree_query,data.categorical_columns,ctx['weights'],model)
        metadata=model/'metadata.json';chain=comp.chain_unit(ctx,native)
    comp.verify_unchanged(ctx)
    records={k:c.save_array(folder/(k+'.npy'),chain[k]) for k in LAYERS}
    c.write(folder/'stage.json',{'stage':ORIGIN,'candidate_round':number,'fit_rows':272117,
        'query_rows':107169,'query_labels_available':False,'model_metadata':c.record(metadata),
        'sealed_models':model_records(model),'layers':records,'fit_seconds':meta['fit_seconds'],
        'baseline_replayed_exact':True,'composition_inputs':ctx['input_sha256']})
    validate(number,c.record(folder/'stage.json'))
    c.need(protected()==r['protected_before'],'Protected artifacts changed')
    print(json.dumps({'round':number,'origin':ORIGIN,'status':'saved_and_sealed_unscored'}),flush=True)


def train():
    r=request();tick=time.monotonic();failure=None
    try:
        for number in CANDIDATES:
            command=[str(c.ROOT/'.venv/Scripts/python.exe'),'-u','-B','-X','utf8',__file__,'--stage',str(number)]
            p=subprocess.run(command,cwd=c.ROOT,check=False)
            c.need(p.returncode==0,'Stage failed; no automatic retry: R'+str(number))
        records={str(n):c.record(f'outputs/round113/R{n}/stage.json') for n in CANDIDATES}
        for n in CANDIDATES:validate(n,records[str(n)])
        c.verify_hashes(r['source_sha256']);c.verify_hashes(r['input_sha256'])
        c.need(protected()==r['protected_before'],'Protected artifacts changed')
        c.write(SEAL,{'status':'three_offset_models_all_query_predictions_sealed_unscored',
            'request':c.record(REQUEST),'records':records,'query_scores_computed':False,
            'elapsed_seconds':time.monotonic()-tick})
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        c.write(EVIDENCE+'/training_execution.json',{'exit_code':1 if failure else 0,'failure':failure,
            'elapsed_seconds':time.monotonic()-tick})


def gates(metrics,comparison):
    parts={'overall_regression_at_most_0_25pct':metrics['T']['overall_mae']<=1.0025*metrics['v22']['overall_mae'],
           'weighted_regression_at_most_0_25pct':metrics['T']['weighted_mae']<=1.0025*metrics['v22']['weighted_mae'],
           'no_quarter_or_large_group_above_5pct':not comparison['quarter_or_large_group_deterioration_above_5pct']}
    return {'components':parts,'passed':all(parts.values())}


def evaluate():
    r=request();seal=c.read(SEAL);tick=time.monotonic()
    c.need(seal['status']=='three_offset_models_all_query_predictions_sealed_unscored'
           and seal['request']==c.record(REQUEST) and seal['query_scores_computed'] is False,
           'All three models must be sealed before query price access')
    execution=c.read(EVIDENCE+'/training_execution.json')
    c.need(execution['exit_code']==0 and execution['failure'] is None,'Successful separate training required')
    checked={n:validate(n,seal['records'][str(n)]) for n in CANDIDATES}
    c.verify_hashes(r['input_sha256']);ctx=comp.load_context(r['composition']);d=ctx['data']
    c.need(c.sha(c.RAW)==r['input_sha256'][c.RAW],'Query label source identity differs')
    raw=pd.read_pickle(c.safe(c.RAW))
    y=old_metric.select_query_y(raw,d.query_source_cache_indices,455784);del raw
    baseline={k:old_metric.descriptive_metrics(y,ctx['baseline'][k],d.Xfit,d.Xquery) for k in ('v22','v24')}
    saved=c.read('outputs/round97/evaluation/results.json')['metrics']
    c.need(all(baseline[k]==saved[k] for k in baseline),'Both R97 baseline metrics must replay exactly')
    candidates={}
    for n in CANDIDATES:
        item,layers=checked[n]
        m=old_metric.descriptive_metrics(y,layers['median'],d.Xfit,d.Xquery)
        metrics={**baseline,'T':m}
        comparison=old_metric.comparisons({'v22':baseline['v22'],'v24':m})
        comparison24=old_metric.comparisons({'v22':baseline['v24'],'v24':m})
        comparison.update(control='R97/v22',candidate='R'+str(n),used_for_release_gate=True)
        comparison24.update(control='R97/v24',candidate='R'+str(n),used_for_release_gate=False)
        pair,samples=old_metric.paired(y,{'v22':ctx['baseline']['v22'],'v24':layers['median']},
                                     d.Xfit,d.Xquery,d.query_indices)
        pair.update(candidate='R'+str(n),difference='R'+str(n)+' minus v22',caution=CAUTION)
        folder=c.ROOT/f'outputs/round113/R{n}/evaluation';folder.mkdir(exist_ok=False)
        sample_records={k:c.save_array(folder/(k+'_samples.npy'),v) for k,v in samples.items()}
        result={'metrics':metrics,'comparison':comparison,'comparison_vs_v24':comparison24,
            'paired':pair,'paired_samples':sample_records,'gate':gates(metrics,comparison),
            'both_point_metrics_improve':all(m[k]<baseline['v22'][k] for k in ('overall_mae','weighted_mae')),
            'branch_metrics':{k:old_metric.descriptive_metrics(y,v,d.Xfit,d.Xquery) for k,v in layers.items() if k!='median'},
            'sealed_stage':seal['records'][str(n)]}
        c.write(folder/'results.json',result);candidates[str(n)]=result
        print(json.dumps({'round':n,'origin':ORIGIN,'mae':m['overall_mae'],
            'delta':m['overall_mae']-baseline['v22']['overall_mae'],'gate':result['gate']}),flush=True)
    comp.verify_unchanged(ctx);c.need(protected()==r['protected_before'],'Protected artifacts changed')
    c.verify_hashes(r['source_sha256']);c.verify_hashes(r['input_sha256'])
    for n in CANDIDATES:validate(n,seal['records'][str(n)])
    c.write(RESULTS,{'status':'fixed_extra_origin_evaluated','created_at_utc':c.now(),'origin':ORIGIN,
        'request':c.record(REQUEST),'seal':c.record(SEAL),'plan':c.record(PLAN),'candidates':candidates,
        'source_sha256':sources(),'query_y_values_sha256':c.values_sha(y),'query_y_persisted':False,
        'query_labels_used_for_fit':False,'fresh_blind_validation':False,'caution':CAUTION,
        'Public_score':None,'Private_score':None,'selected_submission':'v22','new_CSV':False})
    c.write(GATE,{'status':'fixed_extra_origin_evaluated','candidates':list(CANDIDATES),
        'candidate_gates':{n:v['gate'] for n,v in candidates.items()},'results':c.record(RESULTS),
        'plan':c.record(PLAN),'Public_score':None,'Private_score':None,'selected_submission':'v22','new_CSV':False})
    c.write(EVIDENCE+'/evaluation_execution.json',{'exit_code':0,'elapsed_seconds':time.monotonic()-tick,
        'query_labels_used_for_scoring_only':True,'training_already_finished_before_scoring':True,
        'protected_unchanged':True,'results':c.record(RESULTS),'release_gate':c.record(GATE)})


def self_check():
    """Synthetic gate boundaries only; no model/data/GPU access."""
    count=0
    def case(t,w,bad,want):
        nonlocal count
        m={'v22':{'overall_mae':100.,'weighted_mae':200.},'T':{'overall_mae':t,'weighted_mae':w}}
        g=gates(m,{'quarter_or_large_group_deterioration_above_5pct':bad})
        c.need(g['passed'] is want,'Synthetic extra gate differs');count+=1
    case(99,199,[],True);case(100,200,[],True);case(100.25,200.5,[],True)
    case(100.25001,200,[],False);case(100,200.50001,[],False)
    case(99,199,[{'section':'forecast_quarters'}],False)
    case(99,199,[{'rows':500,'section':'cities'}],False)
    c.need(CANDIDATES==(107,111,112) and ORIGIN=='offset_2023_02_11' and len(LAYERS)==5,'Fixed scope');count+=1
    print(json.dumps({'status':'passed','synthetic_checks':count,'true_fits':0,'GPU_calls':0,
        'query_labels_read':False,'source_sha256':sources()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--train',action='store_true')
    group.add_argument('--stage',type=int,choices=CANDIDATES);group.add_argument('--evaluate',action='store_true')
    group.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.train:train()
    elif args.stage:stage(args.stage)
    elif args.evaluate:evaluate()
    else:self_check()
