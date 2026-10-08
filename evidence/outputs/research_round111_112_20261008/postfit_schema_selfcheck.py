"""Synthetic checks for the new auditor's metadata schema; no housing data/fit."""
from __future__ import annotations
import ast
import copy
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import round100_contract as c
import audit_model_identity as audit


def fixture():
    prefix='outputs/round111/development/model/'
    original='outputs/round63/development/model/training.json'
    result=c.read('outputs/research_continue_20261008/pipeline_review/independent_R111_cpu_checks_v2.json')['result']
    arrays={'train_num':np.zeros((4,144),np.float32),'train_cat':np.zeros((4,23),np.int64),
            'train_y_million':np.array([1,2,3,4],np.float32),'query_num':np.zeros((2,144),np.float32),
            'query_cat':np.zeros((2,23),np.int64),'cardinalities':np.full(23,5,np.int64),
            'train_price_factor_million':np.array([1,2,3,4],np.float32),
            'query_price_factor_million':np.array([2,3],np.float32),
            'train_sample_weight':np.array([1,2,.5,1.5],np.float32)}
    ratio=np.array([.5,1.],np.float64);pred=ratio*arrays['query_price_factor_million'].astype(np.float64)*1e6
    predictions={prefix+'predictions_ratio.npy':ratio,prefix+'predictions_twd.npy':pred}
    meta={'configuration':copy.deepcopy(result['configuration']),'baseline_configuration':copy.deepcopy(result['baseline_configuration']),
          'source_sha256':c.sha('src/round111_residual_prior_train.py'),'input_sha256':'b'*64,
          'parent_trainer_source_sha256':c.sha('src/round63_recency_weighted_prior_train.py'),
          'original_architecture_source_sha256':c.sha('src/round47_nn_train.py'),
          'input_arrays':{k:c.identity(v) for k,v in arrays.items()},'fit_rows':4,'query_rows':2,
          'numeric_count':144,'categorical_count':23,'cardinalities':[5]*23,
          'actual_model_class':'round111_residual_prior_train.TabularResidualPrior',
          'complete_R63_nonarchitecture_training_configuration_unchanged':True,
          'original_configuration_and_source_verified':True,'original_globals_modified':False,
          'capacity_initialization_and_rng_consumption_also_change':True,
          'skip_connection_only_causal_claimed':False,'official_paper_replication_claimed':False,
          'status':'completed','training_objective_name':'weighted MAE','epoch_selection_uses_training_metrics':False,
          'query_labels_available':False,'query_labels_used':False,'sample_weights_recomputed_in_trainer':False,
          'sample_weights_batch_renormalized':False,'price_or_ratio_clipping':False,
          'BatchNorm_statistics_present_or_recomputed':False,'saved_model_reload_exact':True,
          'saved_native_ratio_reload_exact':True,'reload_rows':2,
          'epochs':[{'epoch':n,'learning_rate':.001} for n in range(1,31)],
          'changed_configuration_fields':['hidden'],
          'added_configuration_fields':sorted(set(result['configuration'])-set(result['baseline_configuration'])),
          'initial_training_median_ratio':1.,'initial_head_bias':1.+float(np.log(-np.expm1(-1.))),
          'model_sha256':'a'*64,'parameter_count':123,'embedding_dimensions':[4]*23}
    meta['architecture_source_sha256']=meta['source_sha256']
    for k in ('train_sample_weight','train_price_factor_million','query_price_factor_million'):
        meta[k+'_sha256']=c.values_sha(arrays[k])
    meta['prediction_artifacts']={Path(p).name:{'sha256':'a'*64,**c.identity(v)} for p,v in predictions.items()}
    saved={'payload':{'configuration':copy.deepcopy(meta['configuration']),
                     'baseline_configuration':copy.deepcopy(meta['baseline_configuration']),
                     'numeric_count':144,'cardinalities':[5]*23},
           'parameter_count':123,'embedding_dimensions':[4]*23}
    old={'configuration':result['baseline_configuration'],'epochs':copy.deepcopy(meta['epochs'])}
    spec={'prepared':{'sha256':'b'*64}}
    state={'meta':meta,'arrays':arrays,'saved':saved,'predictions':predictions,'native':pred.copy()}
    real_read,real_sha=c.read,c.sha
    def read(path):
        if str(path)==original:return old
        if str(path)=='outputs/round63/results.json':return {'stages':{'development':{'artifact_sha256':{original:'a'*64}}}}
        return real_read(path)
    def sha(path):
        if str(path)==original or str(path).startswith(prefix):return 'a'*64
        return real_sha(path)
    def load(path,**kwargs):return state['predictions'][c.rel(path)].copy()
    return state,spec,read,sha,load


def run():
    cases={
        'valid_new_R111_schema':None,
        'reject_query_label_flag':lambda s:s['meta'].__setitem__('query_labels_used',True),
        'reject_ratio_loss':lambda s:s['meta'].__setitem__('training_objective_name','ratio MAE'),
        'reject_wrong_factor_SHA':lambda s:s['meta'].__setitem__('query_price_factor_million_sha256','c'*64),
        'reject_changed_input_identity':lambda s:s['meta']['input_arrays']['train_y_million'].__setitem__('values_sha256','c'*64),
        'reject_changed_schedule':lambda s:s['meta']['epochs'][4].__setitem__('learning_rate',.2),
        'reject_changed_actual_checkpoint_config':lambda s:s['saved']['payload']['configuration'].__setitem__('learning_rate',.2),
        'reject_changed_checkpoint_parameter_count':lambda s:s['saved'].__setitem__('parameter_count',124),
        'reject_native_layer_mismatch':lambda s:s.__setitem__('native',s['native']+1),
        'reject_changed_ratio_values':lambda s:s['predictions'].__setitem__('outputs/round111/development/model/predictions_ratio.npy',np.array([.2,1.])),
    }
    checked=[]
    for label,mutation in cases.items():
        state,spec,reader,hasher,loader=fixture()
        if mutation:mutation(state)
        with patch.object(c,'read',reader),patch.object(c,'sha',hasher),patch.object(np,'load',loader),\
             patch.object(c,'load_array',lambda rec:state['native'].copy()),\
             patch.object(audit,'checkpoint_cpu',lambda p:copy.deepcopy(state['saved'])),\
             patch.dict(c.SPLITS,{'development':('unused','unused',4,2)}):
            try:audit.check_nn('development',{'layers':{'native':{}}},state['meta'],spec,state['arrays'])
            except ValueError:
                c.need(mutation is not None,'Valid R111 schema rejected')
            else:c.need(mutation is None,'Auditor accepted invalid synthetic schema '+label)
        checked.append(label)
    tree=ast.parse(Path(audit.__file__).read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_unit')
    assignments=[n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cats' for t in n.targets)]
    c.need(len(assignments)==1 and isinstance(assignments[0].value,ast.Subscript)
           and isinstance(assignments[0].value.slice,ast.Constant) and assignments[0].value.slice.value=='categorical_columns',
           'Unit auditor must use original feature-schema cats')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_check_state']
    c.need(len(calls)==2 and all(isinstance(n.args[-1],ast.Name) and n.args[-1].id=='cats' for n in calls),
           'Unit auditor must pass raw cats to both fit/query state checks')
    checked.extend(['original_schema_cats_source','both_fit_and_query_use_raw_cats'])
    return {'status':'passed','checks':checked,'check_count':len(checked),'synthetic_metadata_only':True,
            'actual_new_models_inspected':False,'query_labels_loaded':False,'real_NPZ_payloads_loaded':False,
            'new_fits':0,'GPU_calls':0,'model_audit_tool':c.record(audit.__file__),'test_tool':c.record(__file__)}


if __name__=='__main__':
    import json
    print(json.dumps(run(),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
