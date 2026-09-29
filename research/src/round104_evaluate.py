"""R104 independent, all-students-sealed scorer; no training/model prediction."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round100_contract as c
import round100_temporal_prior_evaluate_v2 as metric

PLAN='reports/round104_plan.md'
PLAN_SHA='106fda0c56cc7c5660c6255bb572fd78c59ccac9c13d25604bbe592204e43778'
OLD={'path':'outputs/round102/sealed_manifest.json','sha256':'fd75ffd27998d008cedb65a13632e48c480f05beb9c69ee9bccf5b2e7a1aa68a'}
OLD_RESULTS={'path':'outputs/round102/evaluation/results.json','sha256':'e385ed630cdf3a2b2b0418867563a4064fbe08c1c979feb004787df2592f2347'}
TRAINER='src/round104_base_aware_train.py'
TRAINER_SOURCES=(TRAINER,'src/round63_recency_weighted_prior_train.py','src/round52_nn_train.py','src/round47_nn_train.py')
CAUTION=('Adaptive reused-window research, not blind or fully historically-as-of validation. '
         'Static 2023 geography postdates the earlier origins. Teacher and query model distributions differ. '
         'C is a same-subset, matched-gradient-scale native loss control; T optimizes fixed-base loss. '
         'Conditional bootstrap is not Private probability; v22 remains the sole selected submission.')

def parameters():
    p=c.evaluation_parameters()
    p.pop('point_only_control')
    p.update(round=104, candidate_role='fixed_base_L1', mechanism_control_role='matched_scale_native_L1_same_subset',
             mechanism_required_native_improvement=False,
             mechanism_required_layers=['base','median'], mechanism_required_metrics=['overall_mae','weighted_mae'],
             historical_point_background='R102T', fully_historically_as_of=False)
    return p

def source_hashes():return c.source_hashes(__file__,TRAINER)

def research_gates(stages):
    primary=metric.research_gate(stages)
    parts={f'{layer}_all_three_{name}_improve':all(
        stages[s]['layer_metrics']['T'][layer][name]<stages[s]['layer_metrics']['C'][layer][name]
        for s in c.STAGES) for layer in ('base','median') for name in ('overall_mae','weighted_mae')}
    mechanism={'components':parts,'mechanism_gate_passed':all(parts.values()),
               'native_improvement_required':False,'causal_identification_claimed':False}
    return {'primary':primary,'mechanism':mechanism,
            'both_research_gates_passed':primary['adoption_research_gate_passed'] and mechanism['mechanism_gate_passed'],
            'automatic_adoption':False,'selected_submission':'v22','Private_score':None,'new_CSV':False}

def check_training(stage,preparation,training,launch):
    """Compare certificates without loading weights or training NPZ payloads."""
    n,q=stage['student_fit_rows'],stage['query_rows']
    c.need(0<n<=stage['outer_fit_rows'],'Student fit subset bounds')
    shared=stage['all_input_arrays']
    c.need(set(shared)==set(c.ARRAY_NAMES)|{'train_base_offset_million'},'Exactly ten shared arrays')
    for name in ('train_num','query_num'):
        c.need(shared[name]['dtype']=='float32' and shared[name]['shape']==[n if name.startswith('train') else q,145],'145 numeric columns')
    for arm in ('C','T'):
        ar=stage['arms'][arm];tr=training[arm]
        c.need(tr['status']=='completed' and tr['objective']==arm and tr['plan_sha256']==PLAN_SHA,'Trainer status/objective')
        c.need(tr['configuration']==launch['trainer_configurations'][arm] and tr['parameters']==launch['trainer_parameters'],'Fixed objective/configuration')
        c.need(tr['source_hashes']=={k:c.sha(k) for k in TRAINER_SOURCES} and tr['source_sha256']==c.sha(TRAINER),'Trainer source changed')
        c.need(tr['input_array_identities']==shared and tr['input_sha256']==ar['prepared']['sha256'],'Matched input identity')
        c.need(ar['prepared']==stage['arms']['C']['prepared'],'C/T must use identical prepared file')
        c.need(tr['fit_rows']==n and tr['query_rows']==q and tr['numeric_count']==145,'Training row/width certificate')
        c.need(tr['saved_model_reload_exact'] is True and tr['reload_rows']==q,'Full query saved-model reload')
        for flag in ('query_labels_available','query_labels_used','sample_weights_recomputed_in_trainer',
                     'sample_weights_subset_renormalized','sample_weights_batch_renormalized',
                     'price_or_ratio_clipping','residual_target_clipping','initial_bias_uses_offset'):
            c.need(tr[flag] is False,'Forbidden trainer flag: '+flag)
        c.need([e['epoch'] for e in tr['epochs']]==list(range(1,31)),'Thirty fixed epochs required')
        for key in ('initial_head_bias','initial_training_median_ratio','cardinalities'):
            c.need(tr[key]==training['C'][key],'C/T initialization/categories differ')
        for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight','train_base_offset_million'):
            c.need(tr[key+'_sha256']==shared[key]['values_sha256'],'Factor/weight/offset identity differs')
    c.need(preparation['fit_rows']==n and preparation['query_rows']==q,'Preparation rows differ')

def check_header(seal):
    c.need(seal['schema_version']==1 and seal['status']=='all_base_aware_predictions_sealed'
           and seal['stage_order']==list(c.STAGES) and set(seal['stages'])==set(c.STAGES),'All R104 students must seal')
    c.need(seal['plan_sha256']==PLAN_SHA and seal['new_NN_fits']==6 and seal['new_native_tree_fits']==0,'Student plan/count differs')
    c.need(seal['all_query_reload_exact'] is True and seal['query_scores_computed'] is False,'Unscored exact reload required')
    c.need(len(seal['protected_before'])==9 and seal['protected_before']==seal['protected_after'],'Protected files changed')

def preflight(sealed_path,request_path):
    checked={}
    def bind(path,digest):
        path=c.rel(path);c.need(path not in checked or checked[path]==digest,'Conflicting SHA authority')
        c.need(c.sha(path)==digest,'Input SHA differs: '+path);checked[path]=digest;return c.safe(path)
    def document(rec):return c.read(bind(rec['path'],rec['sha256']))
    def array(rec,semantic=True):
        a=np.load(bind(rec['path'],rec['sha256']),allow_pickle=False)
        c.need(isinstance(a,np.ndarray) and not a.dtype.hasobject,'Numeric array required')
        if semantic:c.need(c.identity(a)=={k:rec[k] for k in ('dtype','shape','values_sha256')},'Array identity differs')
        return a
    def terminal(path):
        p=c.safe(path);v=document({'path':c.rel(p),'sha256':p.with_suffix('.sha256').read_text().strip()})
        c.need(v['exit_code']==0 and v.get('failure') is None,'Execution not successful');return v
    request=document(c.record(request_path))
    c.need(request['schema_version']==1 and request['plan']=={'path':PLAN,'sha256':PLAN_SHA}
           and request['parameters']==parameters(),'Evaluation request differs')
    bind(PLAN,PLAN_SHA);own=source_hashes()
    c.need(all(request['source_sha256'].get(k)==v for k,v in own.items()),'Evaluator closure not frozen')
    c.verify_hashes(request['source_sha256'])
    launch=document(request['launch_request']);c.need(launch['plan_sha256']==PLAN_SHA,'Research launch plan differs')
    c.verify_hashes(launch['source_sha256'])
    c.need(request['protected_sha256']==c.protected_hashes(),'Protected files changed')
    seal=document(c.record(sealed_path));check_header(seal)
    c.need(seal['request']==request['launch_request'] and seal['protected_after']==request['protected_sha256'],'Root provenance differs')
    c.need(c.rel(sealed_path)=='outputs/round104/sealed_manifest.json','Unexpected main seal location')
    terminal('outputs/round104/execution.json')
    c.verify_hashes(seal['source_sha256'])
    c.need(all(launch['source_sha256'].get(k)==v for k,v in seal['source_sha256'].items()),'Driver source not frozen')
    teacher=document(request['teacher_seal'])
    c.need(teacher['stage']=='pre_auxiliary' and teacher['status']=='pre_auxiliary_teachers_sealed'
           and teacher['new_native_fits']==2 and teacher['all_query_reload_exact'] is True
           and teacher['query_labels_available'] is False,'Teacher seal differs')
    c.need((teacher['cutoff'],teacher['query_end'])==('2020-08-11','2021-08-10'),'Teacher chronology differs')
    terminal(c.safe(request['teacher_seal']['path']).parent/'execution.json')
    oldmain=document(OLD);oldresults=document(OLD_RESULTS)
    for key in ('features','canonical_mapping','raw_train'):
        c.need(seal[key]==oldmain[key],'Dataset identity differs');bind(seal[key]['path'],seal[key]['sha256'])
    mapping=array(seal['canonical_mapping'],False);metric.index(mapping,455784,'canonical mapping');c.need(len(mapping)==455255,'Canonical row count')
    Xraw=pd.read_pickle(c.safe(c.FEATURES))
    c.need(len(Xraw)==455784 and '總價元' not in Xraw and Xraw.index.equals(pd.RangeIndex(len(Xraw))),'Pure X schema')
    days=Xraw.iloc[mapping].trade_day_index.to_numpy(dtype=np.float64);stages={}
    for s in c.STAGES:
        sr=seal['stages'][s];sm=document(sr['stage_manifest']);oldstage=document(oldmain['stages'][s]['stage_manifest'])
        ex=document(sr['execution']);c.need(ex['exit_code']==0 and ex.get('failure') is None,'Stage execution incomplete')
        c.need(sm['stage']==s and sm['status']=='base_aware_stage_predictions_sealed' and sm['plan_sha256']==PLAN_SHA,'Stage role differs')
        c.need(sm['request']==seal['request'] and sm['protected_before']==sm['protected_after']==request['protected_sha256'],'Stage provenance')
        c.need(sm['all_query_reload_exact'] is True and sm['query_scores_computed'] is False
               and sm['raw_price_pickle_opened'] is False,'Stage labels/reload boundary')
        for name,digest in sm['artifact_sha256'].items():bind(name,digest)
        start,end,nf,nq=c.SPLITS[s]
        c.need((sm['cutoff'],sm['query_end'],sm['outer_fit_rows'],sm['query_rows'])==(start,end,nf,nq),'Outer window changed')
        c.need(sm['source_R102_stage']==oldmain['stages'][s]['stage_manifest'],'Source R102 authority differs')
        ids={k:array(sm[k]) for k in ('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices')}
        for k,a in ids.items():c.exact(a,array(oldstage[k]),'Outer index changed: '+k)
        fi,qi=ids['fit_indices'],ids['query_indices']
        metric.check_temporal(s,fi,qi,ids['query_source_cache_indices'],days,mapping)
        c.exact(mapping[fi],ids['fit_source_cache_indices'],'Raw fit mapping')
        student=array(sm['student_fit_indices']);metric.index(student,len(mapping),'Student indices')
        c.need(len(student)==sm['student_fit_rows'] and np.isin(student,fi).all(),'Student subset differs')
        preparation=document(sm['preparation']);training={}
        for arm in ('C','T'):
            ar=sm['arms'][arm]
            for key in ('model','prepared'):bind(ar[key]['path'],ar[key]['sha256'])
            training[arm]=document(ar['training']);ax=document(ar['execution'])
            c.need(ax['exit_code']==0 and ax.get('failure') is None,'Arm incomplete')
        check_training(sm,preparation,training,launch)
        fixed={k:array(r) for k,r in sm['fixed_branches'].items()}
        c.need(set(fixed)=={'area','lgb','H','U'},'Fixed branches differ')
        for k,a in fixed.items():c.exact(a,array(oldstage['fixed_branches'][k]),'Fixed branch changed')
        predictions={k:array(r) for k,r in sm['predictions'].items()}
        layers={arm:{k:array(r) for k,r in records.items()} for arm,records in sm['layers'].items()}
        c.need(set(predictions)==set(layers)=={'v22','C','T'} and sr['predictions']==sm['predictions'],'Three arms required')
        c.need(all(set(a)=={'native','base','reference','spatial'} for a in layers.values()),'Four premedian layers required')
        for a in [*fixed.values(),*predictions.values(),*[a for layer in layers.values() for a in layer.values()]]:metric.prices(a,(nq,),'Query vector')
        metric.check_formulas(predictions,layers,fixed)
        c.exact(predictions['v22'],array(oldstage['predictions']['v22']),'v22 changed')
        for k,a in layers['v22'].items():c.exact(a,array(oldstage['layers']['v22'][k]),'v22 layer changed')
        for arm in ('C','T'):c.exact(layers[arm]['native'],array(sm['arms'][arm]['native_prediction']),'Native trainer vector differs')
        stages[s]={'fit':Xraw.iloc[mapping[fi]].reset_index(drop=True),'query':Xraw.iloc[mapping[qi]].reset_index(drop=True),
                   'query_indices':qi,'query_source_cache_indices':ids['query_source_cache_indices'],
                   'student_fit_rows':len(student),'predictions':predictions,'layers':layers}
    c.need(request['protected_sha256']==c.protected_hashes(),'Protected changed during gate')
    return {'stages':stages,'input_sha256':checked,'source_sha256':own,'protected_before':request['protected_sha256'],
            'raw_train':seal['raw_train'],'old_results':oldresults,'sealed_manifest':c.record(sealed_path),
            'evaluation_request':c.record(request_path),'teacher_seal':request['teacher_seal']}

def run(sealed_path,request_path,outpath):
    out=c.safe(outpath);out.mkdir(parents=True,exist_ok=False);tick=time.monotonic();failure=None;opened=False
    try:
        checked=preflight(sealed_path,request_path)
        c.write(out/'prelabel_gate.json',{'status':'all_eight_native_fits_sealed_before_student_query_scoring',
            'input_sha256':checked['input_sha256'],'source_sha256':checked['source_sha256'],
            'sealed_manifest':checked['sealed_manifest'],'teacher_seal':checked['teacher_seal'],
            'models_and_NPZ_hashed_not_deserialized':True,'completed_at_utc':c.now()})
        c.need(c.sha(checked['raw_train']['path'])==checked['raw_train']['sha256'],'Raw prices changed')
        raw=pd.read_pickle(c.safe(checked['raw_train']['path']));opened=True;stages={}
        for s in c.STAGES:
            a=checked['stages'][s];y=metric.select_query_y(raw,a['query_source_cache_indices'])
            score=lambda p:metric.descriptive_metrics(y,p,a['fit'],a['query'],c.SPLITS[s][0])
            metrics={k:score(v) for k,v in a['predictions'].items()}
            layers={arm:{k:score(v) for k,v in values.items()} for arm,values in a['layers'].items()}
            for arm in layers:layers[arm]['median']=metrics[arm]
            old=checked['old_results']['stages'][s]
            c.need(metrics['v22']==old['metrics']['v22'],'Original full v22 metrics replay differs')
            c.need(c.values_sha(y)==old['query_target_values_sha256'],'Query truth mapping differs')
            folder=out/s;folder.mkdir();boots={};replicates={}
            for control in ('v22','C'):
                item,samples=metric.paired(y,a['predictions'],a['fit'],a['query'],a['query_indices'],s,control);item['caution']=CAUTION
                for family in item['methods'].values():
                    for short,key in (('overall','overall_mae'),('weighted','weighted_mae')):
                        z=family['summary'][short]
                        c.need(abs(z['control_mae']-metrics[control][key])<=1e-6 and abs(z['candidate_mae']-metrics['T'][key])<=1e-6,'Bootstrap point differs')
                boots[control]=item;replicates[control]={f:c.save_array(folder/f'T_vs_{control}_{f}_replicates.npy',v) for f,v in samples.items()}
            item={'stage':s,'cutoff':c.SPLITS[s][0],'query_end':c.SPLITS[s][1],
                  'outer_fit_rows':len(a['fit']),'student_fit_rows':a['student_fit_rows'],'query_rows':len(y),
                  'metrics':metrics,'layer_metrics':layers,'comparison':metric.comparisons(metrics),
                  'comparison_vs_C':metric.comparisons(metrics,'C'),'C_vs_v22':metric.comparisons(metrics,'v22','C'),
                  'historical_R102T_point_background':{'metrics':old['metrics']['T'],'layer_metrics':old['layer_metrics']['T'],
                      'T_minus_R102T':{k:metrics['T'][k]-old['metrics']['T'][k] for k in ('overall_mae','weighted_mae')}},
                  'paired':boots,'replicate_artifacts':replicates,'query_target_values_sha256':c.values_sha(y),
                  'query_target_values_persisted':False,'group_membership_uses_original_outer_fit':True,
                  'full_original_v22_metrics_replay_exact':True}
            c.write(folder/'results.json',item);stages[s]=item
            print(json.dumps({'stage':s,'T_minus_v22':item['comparison']['overall_mae']['delta'],
                              'T_minus_C':item['comparison_vs_C']['overall_mae']['delta']}),flush=True)
        del raw
        c.verify_hashes(checked['input_sha256']);c.need(source_hashes()==checked['source_sha256'] and c.protected_hashes()==checked['protected_before'],'Inputs/source/protected changed')
        result={'round':104,'status':'completed_three_window_evaluation','created_at_utc':c.now(),'parameters':parameters(),
                'stages':stages,'research_gates':research_gates(stages),
                **{k:checked[k] for k in ('input_sha256','source_sha256','protected_before','sealed_manifest','evaluation_request','teacher_seal')},
                'protected_after':c.protected_hashes(),'raw_pickle_whole_table_deserialized_after_all_seals':True,
                'models_and_NPZ_hashed_not_deserialized':True,'query_labels_used_in_training_or_predictions':False,
                'conditional_intervals_reported':24,'additional_background_bootstrap_calls':0,'new_fits':0,
                'model_load_predict_calls':0,'test_files_read':0,'new_CSV':False,'selected_submission':'v22',
                'Private_score':None,'fresh_blind_validation':False,'fully_historically_as_of':False,'caution':CAUTION,
                'elapsed_seconds':time.monotonic()-tick}
        c.write(out/'results.json',result);return result
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(out/'execution.json',{'exit_code':int(failure is not None),'failure':failure,
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':c.sha(__file__),'raw_prices_deserialized':opened,
        'new_fits':0,'model_load_predict_calls':0,'automatic_retry':False})

def self_check():
    tick=time.monotonic();checks=[]
    def mark(name,ok):c.need(ok,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name)
    stages={s:{'metrics':{a:{'overall_mae':v,'weighted_mae':v} for a,v in [('v22',100.),('C',95.),('T',90.)]},
               'comparison':{'quarter_or_large_group_deterioration_above_5pct':[]},
               'layer_metrics':{a:{l:{'overall_mae':v,'weighted_mae':v} for l in ('native','base','median')}
                                for a,v in [('v22',100.),('C',95.),('T',90.)]}} for s in c.STAGES}
    mark('both_gates_pass',research_gates(stages)['both_research_gates_passed'])
    stages['development']['layer_metrics']['T']['native']['overall_mae']=200.
    mark('native_worse_does_not_fail_new_objective_gate',research_gates(stages)['both_research_gates_passed'])
    for layer in ('base','median'):
        for name in ('overall_mae','weighted_mae'):
            z=copy.deepcopy(stages);z['auxiliary']['layer_metrics']['T'][layer][name]=95.
            mark('reject_non_improvement_'+layer+'_'+name,not research_gates(z)['mechanism']['mechanism_gate_passed'])
    z=copy.deepcopy(stages);z['confirmation']['metrics']['T']['overall_mae']=99.6
    mark('half_percent_primary_required',not research_gates(z)['primary']['adoption_research_gate_passed'])
    z=copy.deepcopy(stages);z['auxiliary']['comparison']['quarter_or_large_group_deterioration_above_5pct']=[{}]
    mark('group_risk_primary_required',not research_gates(z)['primary']['adoption_research_gate_passed'])
    header={'schema_version':1,'status':'all_base_aware_predictions_sealed','stage_order':list(c.STAGES),
            'stages':dict.fromkeys(c.STAGES),'plan_sha256':PLAN_SHA,'new_NN_fits':6,'new_native_tree_fits':0,
            'all_query_reload_exact':True,'query_scores_computed':False,'protected_before':dict.fromkeys(range(9),'x'),
            'protected_after':dict.fromkeys(range(9),'x')}
    check_header(header);checks.append('all_student_seals_accept')
    for k,v in [('new_NN_fits',5),('query_scores_computed',True),('all_query_reload_exact',False),('plan_sha256',c.PLAN_SHA)]:
        z=copy.deepcopy(header);z[k]=v;reject('reject_header_'+k,lambda z=z:check_header(z))
    mark('replicates_fixed',parameters()['replicates']==2000 and parameters()['seed']==20260925)
    return {'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':c.sha(__file__),
            'source_hashes':source_hashes(),'plan_sha256':PLAN_SHA,'parameters':parameters(),
            'real_data_read':False,'new_fit_score_calls':0,'elapsed_seconds':time.monotonic()-tick}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true');p.add_argument('--receipt')
    p.add_argument('--sealed-manifest');p.add_argument('--evaluation-request');p.add_argument('--out');a=p.parse_args()
    if a.self_check:
        r=self_check()
        if a.receipt:c.write(a.receipt,r)
        print(json.dumps(r,ensure_ascii=False),flush=True)
    else:
        if not all((a.sealed_manifest,a.evaluation_request,a.out)):p.error('sealed-manifest/evaluation-request/out required')
        run(a.sealed_manifest,a.evaluation_request,a.out)
