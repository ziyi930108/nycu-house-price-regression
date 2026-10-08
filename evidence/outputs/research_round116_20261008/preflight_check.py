"""Independent R116 CPU contracts, real saved-cut contexts; never fit/score."""
from pathlib import Path
import copy
import json
import sys
import time
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import catboost
import round116_research as d
c = d.c

def main():
    tick = time.monotonic()
    checks = []
    def mark(name, condition):
        if not bool(condition): raise AssertionError(name)
        checks.append(name)
    def reject(name, call):
        try: call()
        except (ValueError, KeyError, TypeError): checks.append(name)
        else: raise AssertionError('Invalid contract accepted: ' + name)
    sources = d.sources(); plan = c.record(d.PLAN)
    protected = d.protected(); splits = copy.deepcopy(c.SPLITS)
    mark('fixed_plan_pre_fit_identity', plan['sha256'] == d.PLAN_SHA == 'f9a4e0c8b7250d1fad5335d6fd02a92729d62bb8662d0376eeed286b8dd1cee6')
    mark('four_origins_exact_order_counts', d.ORIGINS == ('development','confirmation','auxiliary','offset_2023_02_11')
         and d.COUNTS == {'development':(240316,81234),'confirmation':(321550,133697),
                         'auxiliary':(134805,105511),'offset_2023_02_11':(272117,107169)})
    recipe = d.trainer_recipe(); old,new = recipe['baseline_parameters'],recipe['parameters']
    mark('only_Depthwise_added_with_unmodified_R79', set(new)-set(old)=={'grow_policy'} and set(old)<=set(new)
         and all(new[k]==old[k] for k in old) and new['grow_policy']=='Depthwise'
         and recipe['changed_parameter_names']==[] and recipe['added_parameter_names']==['grow_policy'])
    def stages():
        return {s:{'metrics':{'v22':{'overall_mae':100.,'weighted_mae':200.},
                             'T':{'overall_mae':99.,'weighted_mae':198.}},
                   'comparison':{'quarter_or_large_group_deterioration_above_5pct':[]}} for s in d.ORIGINS}
    def gate_case(name, edit, exploration, promotion):
        v=stages();edit(v);g=d.combined_gates(v)
        mark(name, g['combined_exploratory_submission']['passed'] is exploration
             and g['combined_promotion']['passed'] is promotion)
    gate_case('all_four_improve_combined_both_pass',lambda v:None,True,True)
    gate_case('latest_half_percent_inclusive',lambda v:v['confirmation']['metrics']['T'].__setitem__('overall_mae',99.5),True,True)
    gate_case('latest_less_than_half_percent_strong_fails',lambda v:v['confirmation']['metrics']['T'].__setitem__('overall_mae',99.50001),True,False)
    gate_case('latest_weight_no_worse_explore_only',lambda v:v['confirmation']['metrics']['T'].__setitem__('weighted_mae',200.),True,False)
    gate_case('latest_weight_worse_explore_fails',lambda v:v['confirmation']['metrics']['T'].__setitem__('weighted_mae',200.00001),False,False)
    gate_case('development_quarter_percent_inclusive',lambda v:v['development']['metrics'].__setitem__('T',{'overall_mae':100.25,'weighted_mae':200.5}),True,False)
    gate_case('development_over_quarter_percent_fails',lambda v:v['development']['metrics']['T'].__setitem__('overall_mae',100.250001),False,False)
    gate_case('only_one_August_improvement_fails',lambda v:[v[s]['metrics'].__setitem__('T',{'overall_mae':100.,'weighted_mae':200.}) for s in ('development','auxiliary')],False,False)
    gate_case('offset_quarter_percent_both_inclusive',lambda v:v[d.OFFSET]['metrics'].__setitem__('T',{'overall_mae':100.25,'weighted_mae':200.5}),True,True)
    gate_case('offset_overall_next_ULP_fails_both',lambda v:v[d.OFFSET]['metrics']['T'].__setitem__('overall_mae',np.nextafter(100.25,np.inf)),False,False)
    gate_case('offset_weighted_next_ULP_fails_both',lambda v:v[d.OFFSET]['metrics']['T'].__setitem__('weighted_mae',np.nextafter(200.5,np.inf)),False,False)
    for stage in d.ORIGINS:
        gate_case(stage+'_adverse_blocks_both',lambda v,stage=stage:v[stage]['comparison'].__setitem__('quarter_or_large_group_deterioration_above_5pct',[{'section':'cities','rows':500}]),False,False)
    reject('missing_offset_must_not_downgrade_to_three_windows',lambda:d.combined_gates({k:v for k,v in stages().items() if k!=d.OFFSET}))
    reject('extra_origin_must_not_be_ignored',lambda:d.combined_gates({**stages(),'extra':{}}))
    def fake_metrics(rows,section,mae):
        result={'overall_mae':100.,'weighted_mae':200.,'price_bands':[]}
        for k in ('strata','forecast_quarters','months','cities','relative_forecast_months'): result[k]={}
        result[section]={'fixture':{'rows':rows,'mae':mae}};return result
    for section,rows,amount,want in [('cities',499,106.,False),('cities',500,105.,False),('cities',500,105.000001,True),
                                    ('forecast_quarters',1,106.,True),('forecast_quarters',1,105.,False)]:
        m={'v22':fake_metrics(rows,section,100.),'T':fake_metrics(rows,section,amount)}
        r=d.metric.comparisons(m)
        mark('actual_comparison_boundary_'+section+'_'+str(rows)+'_'+str(amount),bool(r['quarter_or_large_group_deterioration_above_5pct']) is want)
    class Archive:
        def __init__(self,files,arrays):self.files,self.arrays,self.reads=files,arrays,[]
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):self.reads.append(key);return self.arrays[key]
    artificial={'train_sample_weight':np.array([.2,.5,1.],np.float32),'train_y_million':np.array([1.,2.,3.],np.float32)}
    yf=np.array([1e6,2e6,3e6],np.float64)
    a=Archive(list(c.ARRAY_NAMES),artificial)
    with patch.object(np,'load',return_value=a):arr=d.prepared_arrays('synthetic.npz',3,2,yf)
    mark('nine_keys_check_then_only_fit_y_and_dates_read',a.reads==['train_sample_weight','train_y_million'] and set(arr)==set(artificial))
    for label,files in [('extra_y',list(c.ARRAY_NAMES)+['query_y']),('missing',list(c.ARRAY_NAMES[:-1])),
                        ('replace_y',list(c.ARRAY_NAMES[:-1])+['query_y']),('duplicate',list(c.ARRAY_NAMES)+['train_num'])]:
        a=Archive(files,artificial)
        with patch.object(np,'load',return_value=a):reject('archive_'+label,lambda:d.prepared_arrays('synthetic.npz',3,2,yf))
        mark('archive_'+label+'_no_array_materialized',not a.reads)
    for name,key,value in [('bad_weight_dtype','train_sample_weight',np.ones(3,np.float64)),
                           ('negative_weight','train_sample_weight',-np.ones(3,np.float32)),
                           ('infinite_weight','train_sample_weight',np.full(3,np.inf,np.float32)),
                           ('wrong_weight_shape','train_sample_weight',np.ones((3,1),np.float32)),
                           ('target_dtype','train_y_million',np.ones(3,np.float64)),
                           ('target_alignment','train_y_million',np.ones(3,np.float32)),
                           ('target_negative','train_y_million',-np.ones(3,np.float32))]:
        a=Archive(list(c.ARRAY_NAMES),{**artificial,key:value})
        with patch.object(np,'load',return_value=a):reject(name,lambda:d.prepared_arrays('synthetic.npz',3,2,yf))
    record={};request={'input_sha256':{c.RAW:d.RAW_SHA}}
    seal={'status':'four_models_all_raw_unit_TWD_and_layers_sealed_unscored','request':record,'origin_order':list(d.ORIGINS),
          'records':{s:{} for s in d.ORIGINS},'new_fits':4,'query_scores_computed':False}
    execution={'exit_code':0,'failure':None,'request':record}
    calls=[]
    def checked(*args):calls.append(args[0]);return ({},{},np.zeros(1))
    with patch.object(d,'request',return_value=request),patch.object(c,'record',return_value=record), \
         patch.object(c,'read',side_effect=lambda p:seal if p==d.SEAL else execution), \
         patch.object(c,'sha',side_effect=lambda p:(calls.append('raw_sha') or d.RAW_SHA)),patch.object(d,'validate',side_effect=checked), \
         patch.object(pd,'read_pickle',side_effect=AssertionError('Forbidden query price read')):
        d.scoring_preflight()
        mark('all_four_validated_before_scoring_only_raw_hash',calls==[*d.ORIGINS,'raw_sha'])
        for name,field,value in [('unsealed','status','bad'),('already_scored','query_scores_computed',True),
                                 ('missing_fourth','records',{s:{} for s in c.STAGES}),('wrong_order','origin_order',list(reversed(d.ORIGINS)))]:
            good=seal[field];seal[field]=value;calls.clear()
            reject('scoring_'+name, d.scoring_preflight);mark('scoring_'+name+'_no_raw_hash_or_validation',not calls);seal[field]=good
        for name,field,value in [('failed_exit','exit_code',1),('failure_trace','failure','bad'),('request_identity','request',{'wrong':1})]:
            good=execution[field];execution[field]=value;calls.clear()
            reject('scoring_'+name,d.scoring_preflight);mark('scoring_'+name+'_no_raw_hash_or_validation',not calls);execution[field]=good
    def forbidden(*args,**kwargs):raise AssertionError('Forbidden real fit/model/scorer/raw label access')
    inputs={};d.check_formal_checks(inputs)
    mark('both_formal_execution_receipts_exact',set(inputs)=={d.EVIDENCE+'/'+n for n in d.formal_checks()})
    inputs['reports/feature_schema.json']=c.sha('reports/feature_schema.json')
    specs=d.august_specifications(inputs)
    extra=d.offset_comp.specification();d.merge_hashes(inputs,extra['input_sha256'])
    inputs['outputs/round97/evaluation/results.json']=c.sha('outputs/round97/evaluation/results.json')
    specs[d.OFFSET]={'composition':extra,'prepared':extra['prior_npz']}
    mark('scoring_only_raw_source_exact_identity',inputs[c.RAW]==d.RAW_SHA)
    for name in sources:compile(c.safe(name).read_text(encoding='utf-8-sig'),name,'exec')
    c.verify_hashes(inputs);c.verify_hashes({**sources,d.PLAN:d.PLAN_SHA})
    print(json.dumps({'phase':'metadata_and_synthetic_guards_passed','checks':len(checks),'source_files':len(sources),
                      'input_files':len(inputs),'real_contexts_next':True}),flush=True)
    read_pickles=set();real_reader=pd.read_pickle
    def guarded_pickle(path,*args,**kwargs):
        name=c.rel(path)
        if name==c.RAW or name not in inputs:raise AssertionError('Unapproved/price pickle: '+name)
        read_pickles.add(name);return real_reader(path,*args,**kwargs)
    run_request={'origins':specs,'recipe':recipe,'source_sha256':sources}
    contexts={}
    with patch.object(pd,'read_pickle',side_effect=guarded_pickle),patch.object(catboost.CatBoostRegressor,'fit',side_effect=forbidden), \
         patch.object(catboost.CatBoostRegressor,'load_model',side_effect=forbidden), \
         patch.object(d.metric,'select_query_y',side_effect=forbidden),patch.object(d.offset_metric,'select_query_y',side_effect=forbidden), \
         patch.object(d.metric,'descriptive_metrics',side_effect=forbidden),patch.object(d.offset_metric,'descriptive_metrics',side_effect=forbidden):
        for origin in d.ORIGINS:
            start=time.monotonic();loaded=d.load_context(run_request,origin)
            d.verify_context(loaded,origin)
            contexts[origin]={'status':'passed','fit_rows':len(loaded['fit_X']),'query_rows':len(loaded['query_X']),
                'fit_input_identity':loaded['fit_input_identity'],'materialized_prepared_array_identities':loaded['array_identities'],
                'indices':{k:c.identity(v) for k,v in loaded['indices'].items()},'v22':c.identity(loaded['v22']),'v24':c.identity(loaded['v24']),
                'original_v22_v24_and_unit_slot_exact':True,'context_unchanged':True,'query_labels_read':False,
                'elapsed_seconds':time.monotonic()-start}
            if origin==d.OFFSET:contexts[origin]['original_adapter_replay_checks']=loaded['context']['replay_checks']
            del loaded
            print(json.dumps({'phase':'actual_CPU_original_context_passed','origin':origin,'elapsed_seconds':contexts[origin]['elapsed_seconds']}),flush=True)
    c.verify_hashes(inputs);c.verify_hashes({**sources,d.PLAN:d.PLAN_SHA})
    mark('full_source_input_plan_and_protected_after_exact',d.sources()==sources and d.protected()==protected and c.SPLITS==splits)
    helper_review=c.read(d.EVIDENCE+'/helper_review.json')
    mark('independent_helper_review_current_sources',helper_review['status']=='passed' and helper_review['source_sha256']==d.helper.source_hashes())
    author_paths=['outputs/research_continue_20261008/direction_review_a/R116_helper_author_selfcheck.json',
                  'outputs/research_continue_20261008/direction_review_a/R116_driver_author_selfcheck.json']
    author_receipts={p:c.record(p) for p in author_paths}
    report={'status':'passed','role':'independent_source_plan_CPU_preflight_before_four_fixed_fits',
        'source_sha256':{**sources,d.PLAN:d.PLAN_SHA},'input_sha256':inputs,'plan':plan,'recipe':recipe,
        'review_tool':c.record(__file__),'helper_independent_review':c.record(d.EVIDENCE+'/helper_review.json'),
        'author_receipts':author_receipts,'formal_execution_receipts':{n:c.record(d.EVIDENCE+'/'+n) for n in d.formal_checks()},
        'origin_order':list(d.ORIGINS),'origin_counts':d.COUNTS,'CPU_context_checks':contexts,
        'independent_synthetic_check_count':len(checks),'independent_synthetic_checks':checks,
        'compiled_Python_sources':len(sources),'protected_before':protected,'old_protected_unchanged':True,
        'read_pickle_paths':sorted(read_pickles),'raw_price_pickle_opened':False,'raw_price_bytes_hashed_only':True,
        'query_labels_read':False,'new_fits':0,'GPU_calls':0,'native_model_loads':0,'native_model_inference_calls':0,
        'new_query_scores':False,'new_CSV':False,'full_data_fit':False,'Public_score':None,'Private_score':None,
        'selected_submission':'v22','global_SPLITS_modified':False,'blocking_findings':[],
        'elapsed_seconds':time.monotonic()-tick,
        'limits':['No actual Depthwise GPU fit/kernel/memory or native saved-CBM verification before training.',
                  'Original saved controls are replayed; not matched new control refits.',
                  'Four reused overlapping windows and repeated research are not independent blind validation.',
                  'Postfit independent source/model/prediction/metric/gate review remains mandatory before a final plan.']}
    c.write(d.EVIDENCE+'/preflight_review.json',report)
    print(json.dumps({'status':'passed','checks':len(checks),'receipt':c.record(d.EVIDENCE+'/preflight_review.json'),
                      'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
