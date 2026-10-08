"""Independent CPU post-evaluation audit. Never fit; refuse premature query labels.

Run --self-check freely. Run --evaluation-complete only after the root confirms
formal evaluation exit 0. Completion/seals/source/input/protected identities
are verified before any raw query labels are deserialized.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import round100_contract as c
import audit_model_identity as model_audit

FOLDER='outputs/research_round111_112_20261008/'
DRIVER='src/round111_112_research.py'
TRAINER='src/round111_residual_prior_train.py'
HELPER='src/round112_ordered_unit.py'
PLAN='reports/round111_112_plan.md'
PRIMITIVES='outputs/research_continue_20261008/pipeline_review/audit_primitives.py'
PRIOR_AUDIT='outputs/research_round109_110_20261008/verify_completed_v2.py'
PRIOR_AUDIT_SHA='018264e8ac779771f6c8804c540b192ebbff2a07ac0214a29b9554a68bc62692'
LAYERS={'native','base','reference','spatial','median'}


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,c.safe(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


c.need(c.sha(PRIOR_AUDIT)==PRIOR_AUDIT_SHA,'Preserved prior auditor source changed')
prior=load_module('preserved_audit_primitives',PRIOR_AUDIT)
manual=load_module('independent_arithmetic',PRIMITIVES)
same,locked,artifact=prior.same,prior.locked,prior.artifact


def run():
    start=time.monotonic();output=FOLDER+'postfit_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(),'Preserve previous audit')
    dependencies={p:c.sha(p) for p in (PRIOR_AUDIT,PRIMITIVES,FOLDER+'audit_model_identity.py',c.rel(__file__))}
    request,rr=locked(FOLDER+'request.json');seal,sr=locked(FOLDER+'seal.json')
    training,tr=locked(FOLDER+'training_execution.json');evaluation,er=locked(FOLDER+'evaluation_execution.json')
    same(training['exit_code'],0,'Training not successful');same(training['failure'],None,'Training failed')
    same(evaluation['exit_code'],0,'Evaluation not successful')
    c.need(evaluation['training_already_finished_before_scoring'] and evaluation['query_labels_used_for_scoring_only']
           and evaluation['protected_unchanged'],'Evaluation scope receipt')
    same(seal['status'],'six_models_all_query_predictions_sealed_unscored','Six models not sealed')
    same(seal['request'],rr,'Seal/request binding');same(seal['query_scores_computed'],False,'Premature scoring')
    same(request['status'],'frozen_before_six_fits','Request not frozen');same(request['new_fits'],6,'Exactly six fits')
    same(request['plan'],c.record(PLAN),'Plan changed')
    merged={**request['source_sha256'],PLAN:request['plan']['sha256']}
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(request['protected_before'])
    same(c.source_hashes(DRIVER,TRAINER,HELPER),request['source_sha256'],'Actual recursive source closure')
    preflight,pfr=locked(FOLDER+'preflight_review.json')
    same(preflight['status'],'passed','Preflight not passed');same(preflight['source_sha256'],merged,'Exact preflight closure+plan')
    artifact(preflight['checks']);facts=c.read(preflight['checks']['path'])
    same(facts['source_sha256'],merged,'Preflight facts source binding')
    for rec in facts['author_CPU_checks_independently_executed'].values():artifact(rec['receipt'])
    for entry,name,venv,flag in ((TRAINER,'R111_residual_selfcheck_execution.json','.venv_torch','--cpu-unit-check'),
                                (HELPER,'R112_helper_selfcheck_execution.json','.venv','--self-check')):
        check,record=locked(FOLDER+name)
        same(check['exit_code'],0,'Formal CPU selfcheck failure')
        same(check['source_sha256'],c.source_hashes(entry),'Formal CPU exact source closure')
        same(check['command'],[str(ROOT/(venv+'/Scripts/python.exe')),'-B','-X','utf8',str(c.safe(entry)),flag],
             'Formal CPU command identity')
    import round111_112_research as driver
    same(driver.protected(),request['protected_before'],'Protected original file set/content')
    results,result_records={},{}
    for n in (111,112):
        results[str(n)],result_records[str(n)]=locked(f'outputs/round{n}/results.json')
        same(results[str(n)]['status'],'three_reused_windows_evaluated','Incomplete aggregate results')
        same(results[str(n)]['request'],rr,'Results/request binding');same(results[str(n)]['seal'],sr,'Results/seal binding')
        same(results[str(n)]['Private_score'],None,'Unknown Private')
    stage_records,trainmeta={},{}
    for n in (111,112):
        for s in c.STAGES:
            record=seal['records'][str(n)][s]
            st,layers=driver.validate_stage_artifacts(n,s,record)
            artifact(st['model_metadata'])
            same(set(st['layers']),LAYERS,'Five layers required')
            for member in st['sealed_models']:artifact(member)
            same((st['fit_rows'],st['query_rows']),c.SPLITS[s][2:],'Stage row counts')
            same(st['candidate_round'],n,'Stage candidate identity')
            same(st['query_labels_available'],False,'Stage label scope')
            stage_records[n,s]=st;trainmeta[f'{n}/{s}']=c.read(st['model_metadata']['path'])
    # All completion and model/artifact bindings passed before any query price.
    import pandas as pd
    import round104_composition as comp
    import round100_temporal_prior_evaluate_v2 as metric
    raw=pd.read_pickle(c.safe(c.RAW))
    artifacts,summary,configs,bootstraps={},{'111':{},'112':{}},{},[]
    score_count=0
    old22=c.read('outputs/round79/results.json');old24=c.read('outputs/round83/results.json')
    for s in c.STAGES:
        spec=request['stages'][s];ctx=comp.load_context(spec['composition'])
        y=metric.select_query_y(raw,ctx['indices']['query_source_cache_indices'])
        labels=metric.methods.group_labels(ctx['fit_X'],ctx['query_X'])
        with np.load(c.safe(spec['prepared']['path']),allow_pickle=False) as npz:
            same(set(npz.files),set(c.ARRAY_NAMES),'Nine original keys, no query y')
            same(len(npz.files),len(c.ARRAY_NAMES),'No duplicate NPZ keys')
            arrays={k:npz[k].copy() for k in c.ARRAY_NAMES}
        c.exact(c.load_array(spec['fit_labels']['fit_indices']),ctx['indices']['fit_indices'],'Canonical fit indices')
        c.exact(c.load_array(spec['fit_labels']['fit_source_cache_indices']),ctx['indices']['fit_source_cache_indices'],'Raw fit indices')
        c.exact(arrays['train_y_million'],np.asarray(c.load_array(spec['fit_labels']['fit_y'])/1e6,dtype=np.float32),'Original fit target')
        same(c.values_sha(arrays['train_sample_weight']),spec['date_weight_sha256'],'Original saved date weights')
        control=driver.unit_chain(ctx,np.load(c.safe(spec['native_unit']['path']),allow_pickle=False))
        c.exact(control['spatial'],ctx['fixed']['U'],'Original unit spatial replay')
        c.exact(control['median'],ctx['original']['median'],'Original unit final replay')
        for n in (111,112):
            st=stage_records[n,s];meta=trainmeta[f'{n}/{s}'];layers={k:c.load_array(v) for k,v in st['layers'].items()}
            replay=comp.apply(ctx,layers['native'])[0] if n==111 else driver.unit_chain(ctx,layers['native'])
            for k in LAYERS:c.exact(replay[k],layers[k],'Complete saved candidate chain '+k)
            configs[f'{n}/{s}']=model_audit.check_nn(s,st,meta,spec,arrays) if n==111 else model_audit.check_unit(s,st,meta,spec,ctx,arrays['train_sample_weight'])
            item,evaluation_record=locked(f'outputs/round{n}/{s}/evaluation/results.json')
            same(item,results[str(n)]['stages'][s],'Per-stage/aggregate evaluation identity')
            predictions={'v22':ctx['original']['median'],
                         'v24':np.load(c.safe(spec['v24']['path']),allow_pickle=False),'T':layers['median']}
            artifact(spec['v24'])
            for name,prediction in predictions.items():
                target=item['metrics'][name];error=np.abs(prediction-y)
                same(float(error.mean()),target['overall_mae'],'Independent total-price MAE')
                weighted=0.
                for group,weight in c.WEIGHTS.items():
                    take=labels==group
                    same(int(take.sum()),target['strata'][group]['rows'],'Fixed stratum support')
                    if take.any():same(float(error[take].mean()),target['strata'][group]['mae'],'Independent stratum MAE')
                    if weight>0:
                        c.need(take.any(),'Positive-weight stratum absent');weighted+=weight*float(error[take].mean())
                same(weighted,target['weighted_mae'],'Independent fixed-weight MAE')
                same(metric.descriptive_metrics(y,prediction,ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0]),target,'All descriptive groups')
                if name=='v22':
                    old=old22['stages'][s]['metrics']['R79'];same({k:target[k] for k in old},old,'Original v22 metric authority')
                elif name=='v24':
                    old=old24['stages'][s]['metrics']['R83'];same({k:target[k] for k in old},old,'Original v24 metric authority')
                score_count+=1
            primary=manual.manual_comparison(item['metrics']['v22'],item['metrics']['T'],'v22')
            secondary=manual.manual_comparison(item['metrics']['v24'],item['metrics']['T'],'v24')
            same(primary,item['comparison'],'Independent primary group comparisons')
            same(secondary,item['comparison_vs_v24'],'Independent v24 group counterevidence')
            for key,prediction in layers.items():
                if key!='median':same(metric.descriptive_metrics(y,prediction,ctx['fit_X'],ctx['query_X'],c.SPLITS[s][0]),item['branch_metrics'][key],'All branch metrics')
            errors=np.abs(y[:,None]-np.column_stack([predictions['v22'],predictions['T']]))
            point=metric.methods.point_values(errors,labels,c.WEIGHTS)
            bootstraps.extend(prior.check_bootstrap(item,f'outputs/round{n}/{s}/evaluation/',point))
            artifacts[f'{n}/{s}']={'stage':c.record(f'outputs/round{n}/{s}/stage.json'),'training_metadata':st['model_metadata'],
                                  'evaluation':evaluation_record,'sealed_models':st['sealed_models']}
            summary[str(n)][s]={'overall_mae':item['metrics']['T']['overall_mae'],'weighted_mae':item['metrics']['T']['weighted_mae'],
                'overall_delta_vs_v22':primary['overall_mae']['delta'],'weighted_delta_vs_v22':primary['weighted_mae']['delta'],
                'overall_delta_vs_v24':secondary['overall_mae']['delta'],'weighted_delta_vs_v24':secondary['weighted_mae']['delta'],
                'adverse_groups':primary['quarter_or_large_group_deterioration_above_5pct'],
                'v24_adverse_groups':secondary['quarter_or_large_group_deterioration_above_5pct'],
                'project_overall_95pct':item['paired']['methods']['project']['summary']['overall'],'fit_seconds':meta['fit_seconds']}
            comp.verify_unchanged(ctx)
    verdicts={}
    for n in (111,112):
        verdict=manual.manual_gates(results[str(n)]['stages']);gates=results[str(n)]['gates']
        same(verdict['components'],gates['exploratory_submission']['components'],'Manual exploration components')
        same(verdict['exploratory_submission_eligible'],gates['exploratory_submission']['passed'],'Manual exploratory verdict')
        same(verdict['strong_components'],gates['promotion']['components'],'Manual strong components')
        same(verdict['strong_research_gate_passed'],gates['promotion']['adoption_research_gate_passed'],'Manual strong verdict')
        same(verdict['confirmation_fractional_improvement'],gates['promotion']['confirmation_fractional_improvement'],'Exact latest improvement')
        c.need(not gates['automatic_selection'] and gates['selected_submission']=='v22' and gates['private_score'] is None,'Selection/Private semantics')
        verdicts[str(n)]=verdict
    c.verify_hashes(merged);c.verify_hashes(request['input_sha256']);c.verify_hashes(dependencies)
    same(driver.protected(),request['protected_before'],'Protected originals changed during audit')
    for n in (111,112):
        for s in c.STAGES:driver.validate_stage_artifacts(n,s,seal['records'][str(n)][s])
    artifact(preflight['checks'])
    for rec in facts['author_CPU_checks_independently_executed'].values():artifact(rec['receipt'])
    payload={'status':'passed','reviewer':'research_preflight_20261008','created_at_utc':c.now(),
        'scope':'Independent post-evaluation CPU identity, full composition, fixed metrics/groups/gates and native configuration audit; no fit/GPU.',
        'request':rr,'seal':sr,'training_execution':tr,'evaluation_execution':er,'preflight_review':pfr,
        'results':result_records,'artifacts':artifacts,'trainmeta':trainmeta,'manual_gate_verdicts':verdicts,
        'stage_summary':summary,'model_identity_diagnostics':configs,'source_sha256':merged,'plan':request['plan'],
        'review_tool':c.record(__file__),'review_dependency_source_sha256':dependencies,'protected_unchanged':True,
        'preflight_contract_checks':preflight['checks'],
        'preflight_artifact_validator_case_count':facts['artifact_validator_checks']['check_count'],
        'identity_checks':{'frozen_sources_including_plan':len(merged),'frozen_inputs':len(request['input_sha256']),
            'protected_files':len(request['protected_before']),'sealed_stages':6,'candidate_layers_replayed':30,
            'NN_checkpoints_CPU_inspected':3,'native_CBM_loaded_CPU':3,'native_CBM_full_query_CPU_raw_and_TWD_replayed':3,
            'model_metadata_and_complete_native_artifact_sets_verified':6},
        'scoring_checks':{'overall_and_weighted_MAE_independently_recomputed':score_count,
            'primary_and_v24_comparisons_independently_reconstructed':12,'bootstrap_files_summary_checked':len(bootstraps)},
        'bootstrap_artifacts':bootstraps,'query_labels_used_for_postseal_review_only':True,
        'new_model_fits':0,'GPU_work':False,'Private_score':None,'blocking_findings_remaining':[],
        'limits':['NN checkpoint tensors/constructor/config were inspected on CPU without independent NN inference; full-query exact reload comes from frozen training.',
                  'Native CBM full-query CPU predictions and all R/S/median layers were independently replayed.',
                  'Composition/descriptive definitions reuse frozen project helpers; independent MAE/group comparison/gate arithmetic supplements replay.',
                  'Bootstrap summaries use saved replicates, without resampling or correcting cumulative selection, refit uncertainty or future shift; not Private probability.',
                  'This receipt does not certify R113 stability, a subsequent full-data fit, CSV release, Kaggle upload or replacement of selected v22.'],
        'conclusion':'Accuracy eligibility is recorded separately for each round. Any fullfit needs its separate plan/freeze/review and required R113 stability; v22 remains selected.',
        'review_elapsed_seconds':time.monotonic()-start}
    c.write(output,payload)
    print(json.dumps({'status':'passed','review':c.record(output),'manual_gate_verdicts':verdicts},ensure_ascii=False))


def self_check():
    result=manual.self_check()
    for path in (Path(__file__),Path(model_audit.__file__)):
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    result.update(review_tool=c.record(__file__),model_audit_tool=c.record(model_audit.__file__),
                  preserved_auditor=c.record(PRIOR_AUDIT),model_fits=0,GPU_work=False)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-check',action='store_true');group.add_argument('--evaluation-complete',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    else:run()
