"""Independent R113 CPU postseal audit. No fit, GPU, CSV, or selection.

--self-check uses artificial in-memory metrics only. --evaluation-complete is
permitted only after the parent confirms the separate formal evaluate exit 0.
All three native folders, metadata, layers and completion receipts are checked
before opening the raw query price pickle. No automatic retry or overwrite.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import round100_contract as c
import audit_model_identity as native_audit

FOLDER='outputs/research_round113_20261008/'
DRIVER='src/round113_offset_research.py'
PLAN='reports/round113_plan.md'
PLAN_SHA='a21116136190aadf94d89329dd1b9dd98afb62dd493ef6863b356f50ea01dc2c'
PRIMITIVES='outputs/research_continue_20261008/pipeline_review/audit_primitives.py'
LAYERS={'native','base','reference','spatial','median'}
CANDIDATES=(107,111,112)


def same(a,b,label):c.need(a==b,label)


def locked(name):
    path=c.safe(name);side=path.with_suffix('.sha256')
    c.need(side.read_text(encoding='ascii').strip()==c.sha(path),'JSON sidecar identity '+name)
    return c.read(path),c.record(path)


def arithmetic():
    spec=importlib.util.spec_from_file_location('r113_independent_arithmetic',c.safe(PRIMITIVES))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def comparison(control,candidate,number,secondary=False):
    value=arithmetic().manual_comparison(control,candidate,'v24' if secondary else 'v22')
    for k in ('stage_research_gate_passed','used_for_research_gate'):
        value.pop(k,None)
    value.update(control='R97/v24' if secondary else 'R97/v22',candidate='R'+str(number),
                 used_for_release_gate=not secondary,automatic_adoption=False)
    return value


def manual_gate(metrics,compared):
    control,candidate=metrics['v22'],metrics['T']
    components={'overall_regression_at_most_0_25pct':candidate['overall_mae']<=control['overall_mae']*1.0025,
                'weighted_regression_at_most_0_25pct':candidate['weighted_mae']<=control['weighted_mae']*1.0025,
                'no_quarter_or_large_group_above_5pct':not compared['quarter_or_large_group_deterioration_above_5pct']}
    return {'components':components,'passed':all(components.values()),
            'both_point_metrics_improve':all(candidate[k]<control[k] for k in ('overall_mae','weighted_mae'))}


def stage_checks(number,record):
    path=f'outputs/round113/R{number}/stage.json'
    same(record,c.record(path),'Fixed candidate stage path/SHA')
    item=c.read(path)
    same((item['candidate_round'],item['stage'],item['fit_rows'],item['query_rows']),
         (number,'offset_2023_02_11',272117,107169),'Fixed stage/row scope')
    same(item['query_labels_available'],False,'Unscored stage label scope')
    folder=c.safe(f'outputs/round113/models/R{number}')
    files=sorted(folder.iterdir())
    c.need(files and all(p.is_file() and not p.is_symlink() and p.suffix in {'.json','.npy','.pt','.cbm'} for p in files),
           'Complete flat native model folder required')
    same(item['sealed_models'],[c.record(p) for p in files],'Exact complete native folder membership/SHA')
    name='metadata.json' if number==112 else 'training.json'
    same(item['model_metadata'],c.record(folder/name),'Fixed model metadata path/SHA')
    same(set(item['layers']),LAYERS,'Exactly five saved layers')
    layers={k:c.load_array(rec) for k,rec in item['layers'].items()}
    for key,a in layers.items():
        c.need(a.dtype==np.float64 and a.shape==(107169,) and np.isfinite(a).all() and (a>=0).all(),
               'Layer schema/values '+key)
    expected=folder/('native.npy' if number==112 else 'predictions_twd.npy')
    c.exact(layers['native'],np.load(expected,allow_pickle=False),'Actual native/stage vector identity')
    meta=c.read(folder/name)
    c.need(meta['status']=='completed' and meta['saved_model_reload_exact'] and meta['reload_rows']==107169
           and meta['fit_rows']==272117 and meta['query_rows']==107169
           and not meta['query_labels_available'] and not meta['query_labels_used'],'Native fit/reload/label scope')
    return item,layers,meta


def bootstrap(number,item,point):
    pair=item['paired']
    same((pair['control'],pair['candidate'],pair['difference'],pair['seed'],pair['replicates'],pair['stage_index']),
         ('v22','R'+str(number),'R'+str(number)+' minus v22',20260925,2000,0),'Offset paired identity')
    c.need('not fresh blind validation' in pair['caution'] and 'not Private probability' in pair['caution'],
           'Fixed offset bootstrap caution')
    records=[]
    same(set(item['paired_samples']),{'project','month'},'Both bootstrap families')
    for family in ('project','month'):
        path=f'outputs/round113/R{number}/evaluation/{family}_samples.npy'
        record=item['paired_samples'][family]
        same(record['path'],path,'Fixed saved bootstrap path')
        samples=c.load_array(record)
        c.need(samples.dtype==np.float64 and samples.shape==(2000,4),'Paired replicate shape/dtype')
        # Missing positive-weight strata intentionally leave weighted NaN.
        c.need(not np.isinf(samples).any(),'Infinite bootstrap value')
        for metric,left,right in (('overall',0,1),('weighted',2,3)):
            delta=samples[:,right]-samples[:,left];valid=delta[np.isfinite(delta)]
            bounds=np.quantile(valid,[.025,.975],method='linear') if len(valid) else [None,None]
            expected={'valid_replicates':len(valid),'invalid_replicates':2000-len(valid),
                      'control_mae':float(point[left]),'candidate_mae':float(point[right]),
                      'point_delta':float(point[right]-point[left]),
                      'percentile_2_5':bounds[0],'percentile_97_5':bounds[1],
                      'negative_delta_fraction':float(np.mean(valid<0)) if len(valid) else None,
                      'bootstrap_mean_delta':float(valid.mean()) if len(valid) else None}
            summary=pair['methods'][family]['summary'][metric]
            for key,value in expected.items():same(summary[key],value,'Independent bootstrap summary '+key)
        records.append(c.record(path))
    return records


def run():
    tick=time.monotonic();output=FOLDER+'postfit_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(),'Preserve prior audit attempt')
    dependencies={path:c.sha(path) for path in (c.rel(__file__),FOLDER+'audit_model_identity.py',PRIMITIVES)}
    import round113_offset_research as driver
    request,rr=locked(driver.REQUEST);seal,sr=locked(driver.SEAL)
    training,tr=locked(FOLDER+'training_execution.json');evaluation,er=locked(FOLDER+'evaluation_execution.json')
    same((training['exit_code'],training['failure']),(0,None),'Separate successful three-model training')
    same(evaluation['exit_code'],0,'Separate formal evaluation completion required')
    c.need(evaluation['query_labels_used_for_scoring_only'] and evaluation['training_already_finished_before_scoring']
           and evaluation['protected_unchanged'],'Evaluation scope receipt')
    same(request['status'],'frozen_before_three_offset_fits','Fixed prefit request')
    same((request['origin'],request['cutoff'],request['query_end'],request['fit_rows'],request['query_rows'],request['candidates'],request['new_fits']),
         ('offset_2023_02_11','2023-02-11','2024-02-10',272117,107169,[107,111,112],3),'All fixed offset candidates')
    c.need(not request['query_scores_computed'] and not request['query_labels_used_for_fit']
           and request['selected_submission']=='v22' and request['new_CSV'] is False
           and request['Public_score'] is None and request['Private_score'] is None,'Frozen label/selection scope')
    same(request['plan'],c.record(PLAN),'Frozen separate scientific plan')
    same(c.sha(PLAN),PLAN_SHA,'Prefixed R113 scientific plan authority')
    same(driver.sources(),request['source_sha256'],'Exact recursive source closure')
    merged={**request['source_sha256'],PLAN:PLAN_SHA}
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(request['protected_before'])
    same(driver.protected(),request['protected_before'],'Protected originals complete membership/content')
    preflight,pfr=locked(FOLDER+'preflight_review.json')
    same(preflight['status'],'passed','Independent source preflight passed')
    same(preflight['source_sha256'],merged,'Exact independent source closure/plan')
    for filename,(number,entry,flag) in driver.formal_checks().items():
        check,_=locked(FOLDER+filename)
        python=ROOT/('.venv_torch/Scripts/python.exe' if number in (107,111) else '.venv/Scripts/python.exe')
        same(check['exit_code'],0,'Formal CPU selfcheck exit0')
        same(check['command'],[str(python),'-B','-X','utf8',str(c.safe(entry)),flag],'Formal CPU command')
        same(check['source_sha256'],c.source_hashes(entry),'Formal CPU exact closure')
    same(seal['status'],'three_offset_models_all_query_predictions_sealed_unscored','All three pre-score seals')
    same(seal['request'],rr,'Seal/request exact binding');same(seal['query_scores_computed'],False,'No preseal score')
    same(set(seal['records']),{'107','111','112'},'All candidate stages sealed')
    checked={n:stage_checks(n,seal['records'][str(n)]) for n in CANDIDATES}
    # Completion, all models and layers have passed before raw y access.
    results,rs=locked(driver.RESULTS);gate,gr=locked(driver.GATE)
    same((results['status'],results['origin']),('fixed_extra_origin_evaluated','offset_2023_02_11'),'Evaluated fixed origin')
    for field,expected in (('request',rr),('seal',sr),('plan',c.record(PLAN)),('source_sha256',request['source_sha256'])):
        same(results[field],expected,'Result frozen identity '+field)
    same(set(results['candidates']),{'107','111','112'},'Exactly all evaluated candidates')
    c.need(results['Public_score'] is None and results['Private_score'] is None and results['selected_submission']=='v22'
           and not results['new_CSV'] and not results['query_labels_used_for_fit'] and not results['fresh_blind_validation'],
           'Result scores/selection/overlap scope')
    same(evaluation['results'],rs,'Evaluation/results exact identity')
    same(evaluation['release_gate'],gr,'Evaluation/gate exact identity')
    same(gate['results'],rs,'Gate/results binding');same(gate['plan'],c.record(PLAN),'Gate/plan binding')
    same(gate['candidates'],[107,111,112],'Gate candidate list')
    c.need(gate['status']=='fixed_extra_origin_evaluated' and gate['Public_score'] is None
           and gate['Private_score'] is None and gate['selected_submission']=='v22' and not gate['new_CSV'],
           'Gate scope')
    import round113_offset_composition as comp
    import round97_offset_evaluate as metric
    ctx=comp.load_context(request['composition']);data=ctx['data']
    with np.load(ctx['prior_npz'],allow_pickle=False) as z:
        c.need(set(z.files)==set(c.ARRAY_NAMES) and len(z.files)==len(c.ARRAY_NAMES),'Exactly nine prior keys before reading values')
        arrays={key:z[key] for key in c.ARRAY_NAMES}
    same({key:c.identity(value) for key,value in arrays.items()},ctx['prepared']['prior_arrays'],'Original offset NPZ array identities')
    native_facts={}
    for n in CANDIDATES:
        st,layers,meta=checked[n]
        same(st['composition_inputs'],ctx['input_sha256'],'Actual stage fit-only composition inputs')
        replay=comp.chain_unit(ctx,layers['native']) if n==112 else comp.chain_prior(ctx,layers['native'])
        for key in LAYERS:c.exact(replay[key],layers[key],'All complete candidate chain layers '+key)
        native_facts[str(n)]=native_audit.check_unit(st,meta,ctx) if n==112 else native_audit.check_nn(n,st,meta,ctx,arrays)
    import pandas as pd
    raw=pd.read_pickle(c.safe(c.RAW));y=metric.select_query_y(raw,data.query_source_cache_indices,455784);del raw
    same(c.values_sha(y),results['query_y_values_sha256'],'Query y cached raw positional mapping')
    old=c.read('outputs/round97/evaluation/results.json')
    same(c.values_sha(y),old['query_target_values_sha256'],'Original R97 query-target authority')
    labels=metric.methods.group_labels(data.Xfit,data.Xquery)
    artifacts,trainmeta,verdicts,summary,boots={},{},{},{},[]
    score_count=0
    for n in CANDIDATES:
        st,layers,meta=checked[n];item=results['candidates'][str(n)]
        stage_eval,stage_record=locked(f'outputs/round113/R{n}/evaluation/results.json')
        same(stage_eval,item,'Candidate/aggregate evaluation identity')
        same(item['sealed_stage'],seal['records'][str(n)],'Candidate/sealed stage binding')
        vectors={'v22':ctx['baseline']['v22'],'v24':ctx['baseline']['v24'],'T':layers['median']}
        for key,prediction in vectors.items():
            target=item['metrics'][key];errors=np.abs(prediction-y)
            same(float(errors.mean()),target['overall_mae'],'Independent offset MAE')
            weighted=0.
            for group,weight in metric.WEIGHTS.items():
                take=labels==group;same(int(take.sum()),target['strata'][group]['rows'],'Fixed group support')
                if take.any():same(float(errors[take].mean()),target['strata'][group]['mae'],'Independent stratum MAE')
                if weight>0:
                    c.need(take.any(),'Positive-weight stratum absent');weighted+=weight*float(errors[take].mean())
            same(weighted,target['weighted_mae'],'Independent fixed offset weighted MAE')
            same(metric.descriptive_metrics(y,prediction,data.Xfit,data.Xquery),target,'Every offset descriptive group/tail')
            if key!='T':same(target,old['metrics'][key],'Original same-origin saved baseline metrics exact')
            score_count+=1
        primary=comparison(item['metrics']['v22'],item['metrics']['T'],n)
        secondary=comparison(item['metrics']['v24'],item['metrics']['T'],n,True)
        same(primary,item['comparison'],'Independent primary all-group arithmetic')
        same(secondary,item['comparison_vs_v24'],'Independent v24 counterevidence arithmetic')
        for key,prediction in layers.items():
            if key!='median':same(metric.descriptive_metrics(y,prediction,data.Xfit,data.Xquery),item['branch_metrics'][key],'Every branch metric')
        verdict=manual_gate(item['metrics'],primary)
        same({key:verdict[key] for key in ('components','passed')},item['gate'],'Manual candidate gate')
        same(item['gate'],gate['candidate_gates'][str(n)],'Manual/aggregate gate identity')
        same(verdict['both_point_metrics_improve'],item['both_point_metrics_improve'],'Strict two-point improvement')
        point=metric.methods.point_values(np.abs(y[:,None]-np.column_stack([vectors['v22'],vectors['T']])),labels,metric.WEIGHTS)
        boots.extend(bootstrap(n,item,point))
        verdicts[str(n)]=verdict;trainmeta[str(n)]=meta
        artifacts[str(n)]={'stage':seal['records'][str(n)],'training_metadata':st['model_metadata'],
                           'evaluation':stage_record,'sealed_models':st['sealed_models']}
        summary[str(n)]={'overall_mae':item['metrics']['T']['overall_mae'],'weighted_mae':item['metrics']['T']['weighted_mae'],
                        'overall_delta_vs_v22':primary['overall_mae']['delta'],'weighted_delta_vs_v22':primary['weighted_mae']['delta'],
                        'overall_delta_vs_v24':secondary['overall_mae']['delta'],'weighted_delta_vs_v24':secondary['weighted_mae']['delta'],
                        'adverse_groups':primary['quarter_or_large_group_deterioration_above_5pct'],
                        'v24_adverse_groups':secondary['quarter_or_large_group_deterioration_above_5pct'],
                        'fit_seconds':meta['fit_seconds'],'gate':verdict}
    comp.verify_unchanged(ctx)
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(dependencies)
    same(driver.protected(),request['protected_before'],'Protected original membership/SHA unchanged after audit')
    for n in CANDIDATES:stage_checks(n,seal['records'][str(n)])
    payload={'status':'passed','reviewer':'research_preflight_20261008','created_at_utc':c.now(),
             'scope':'Independent CPU post-evaluation audit of every fixed R113 candidate; audit pass is separate from accuracy gates.',
             'source_sha256':merged,'plan':c.record(PLAN),'request':rr,'seal':sr,'training_execution':tr,
             'evaluation_execution':er,'results':rs,'release_gate':gr,'preflight_review':pfr,
             'review_tool':c.record(__file__),'review_dependency_source_sha256':dependencies,
             'artifacts':artifacts,'trainmeta':trainmeta,'manual_gate_verdicts':verdicts,'stage_summary':summary,
             'native_identity_diagnostics':native_facts,'protected_unchanged':True,
             'identity_checks':{'sealed_stages':3,'candidate_layers_replayed':15,'original_context_replay_checks':ctx['replay_checks'],
                                'frozen_sources_including_plan':len(merged),'frozen_inputs':len(request['input_sha256']),
                                'protected_files':len(request['protected_before']),'NN_checkpoints_CPU_inspected':2,
                                'native_CBM_full_query_CPU_raw_and_TWD_replayed':1},
             'scoring_checks':{'overall_and_weighted_MAE_independently_recomputed':score_count,
                               'primary_and_v24_comparisons_independently_reconstructed':6,
                               'branch_metrics_checked':12,'bootstrap_files_summary_checked':len(boots)},
             'bootstrap_artifacts':boots,'query_labels_used_for_postseal_review_only':True,
             'new_model_fits':0,'GPU_work':False,'Public_score':None,'Private_score':None,
             'selected_submission':'v22','new_CSV':False,'blocking_findings_remaining':[],
             'limits':['R97 was previously scored and overlaps August windows; this is not blind validation or an independent fourth vote.',
                       'NN checkpoint tensors and constructors were independently inspected without CPU NN inference. Native full-query exact reload is frozen-trainer evidence.',
                       'The saved CBM full-query raw/TWD inference and complete composition were independently replayed on CPU.',
                       'Descriptive group/composition definitions reuse frozen project helpers; independent MAE/comparison/gate arithmetic supplements replay.',
                       'Saved bootstrap summaries were independently recomputed without resampling; they omit refit/selection/future-shift uncertainty and are not Private probabilities.',
                       'No fullfit/release/CSV/upload or replacement of selected v22 is authorized by this review.'],
             'review_elapsed_seconds':time.monotonic()-tick}
    c.write(output,payload)
    print(json.dumps({'status':'passed','review':c.record(output),'manual_gate_verdicts':verdicts},ensure_ascii=False))


def self_check():
    checks=[]
    def metric(overall=100.,weighted=200.):
        value={'overall_mae':overall,'weighted_mae':weighted,'price_bands':[{'band':'x','rows':500,'mae':100.}]}
        for group in ('strata','forecast_quarters','months','cities','relative_forecast_months'):
            value[group]={'x':{'rows':500,'mae':100.}}
        return value
    for overall,weighted,want in ((99,199,True),(100,200,True),(100.25,200.5,True),(100.25001,200,False),(100,200.50001,False)):
        metrics={'v22':metric(),'T':metric(overall,weighted)}
        item=comparison(metrics['v22'],metrics['T'],107)
        same(manual_gate(metrics,item)['passed'],want,'Boundary gate')
        checks.append('offset_gate_boundary')
    for group,rows,want in (('cities',499,True),('cities',500,False),('forecast_quarters',1,False)):
        base=metric();candidate=metric()
        base[group]['x']={'rows':rows,'mae':100.};candidate[group]['x']={'rows':rows,'mae':105.00001}
        metrics={'v22':base,'T':candidate};item=comparison(base,candidate,111)
        same(manual_gate(metrics,item)['passed'],want,'Group support gate')
        checks.append('offset_group_boundary')
    for n in CANDIDATES:
        primary=comparison(metric(),metric(),n);secondary=comparison(metric(),metric(),n,True)
        same((primary['control'],primary['candidate'],primary['used_for_release_gate']),('R97/v22','R'+str(n),True),'Primary labels')
        same((secondary['control'],secondary['candidate'],secondary['used_for_release_gate']),('R97/v24','R'+str(n),False),'Counterevidence labels')
        same(primary['automatic_adoption'],False,'No automatic selection')
        checks.append('fixed_candidate_comparison_labels')
    for path in (Path(__file__),Path(native_audit.__file__)):
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    result={'status':'passed','synthetic_checks':len(checks),'checks':checks,'review_tool':c.record(__file__),
            'native_audit_tool':c.record(native_audit.__file__),'query_labels_read':False,'real_arrays_read':False,
            'new_model_fits':0,'GPU_work':False,'selection_changed':False}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-check',action='store_true');g.add_argument('--evaluation-complete',action='store_true')
    a=p.parse_args()
    if a.self_check:self_check()
    else:run()
