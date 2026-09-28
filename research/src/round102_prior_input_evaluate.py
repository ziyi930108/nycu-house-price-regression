"""R102 isolated scorer: all six 145-column NN fits seal before query labels.

C is a same-width positive-zero-column control; T receives standardized log prior.
Both retain the identical R100T factors, loss rows, labels and raw date weights.
Only mature metric/comparison/bootstrap functions are reused from R100 v2;
the R102 authority, seven-shared-array and scale contracts are separate here.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round100_temporal_prior_evaluate_v2 as metric

PLAN='reports/round102_plan.md'
PLAN_SHA='ee59dba0da8abb79c88cb6e1b73a6417a91944de784f3c7f77e61755a68bc355'
STAGES=c.STAGES
R100_SEAL={'path':'outputs/round100/sealed_manifest.json','sha256':'60f33aff318b5ec85e0cc17cadb001ec1064814960c3feff63d2e9dc55ba8d7f'}
R100_EVALUATION={'path':'outputs/round100/evaluation/results.json','sha256':'7b7a8bf55aee0878dea3802c3996aebf30263bc00cc3b0c15d7b9e93f99b0369'}
NUMERIC={'train_num','query_num'}
SHARED=set(c.ARRAY_NAMES)-NUMERIC
CAUTION=('Adaptive study on reused windows, not blind validation. C is a 145-column zero-input control, '
    'not the old R100 mask control. C and T both retain R100T factors and loss rows. '
    'Conditional bootstrap is not Private probability; only v22 remains selected.')

def parameters():
    p=c.evaluation_parameters()
    p.update(round=102,candidate='T',candidate_role='145-column standardized_log_prior',
        mechanism_control_role='C: 145-column positive_zero_input; same R100T factors and loss rows',
        numeric_count=145,shared_non_numeric_arrays=sorted(SHARED),
        mechanism_required_native_all_three_improve=True,
        mechanism_required_full_all_three_overall_weighted_improve=True)
    return p

def source_hashes():return c.source_hashes(__file__)

def check_header(seal):
    c.need(seal['schema_version']==1 and seal['status']=='all_prior_input_predictions_sealed'
        and seal['stage_order']==list(STAGES) and set(seal['stages'])==set(STAGES),'All three R102 stages must seal')
    c.need(seal['plan_sha256']==PLAN_SHA and seal['new_NN_fits']==6 and seal['new_native_tree_fits']==0,'R102 plan/fit count')
    c.need(seal['all_query_reload_exact'] is True and seal['query_scores_computed'] is False,'Unscored full reload required')
    c.need(seal['protected_before']==seal['protected_after'] and len(seal['protected_before'])==9,'Protected files changed')

def check_scale_and_arrays(stage,preparation,scale,oldstage,training):
    """Metadata certificate check; intentionally does not deserialize any NPZ/model."""
    n=stage['eligible_rows'];q=stage['query_rows'];source=oldstage['arms']['T']['all_input_arrays']
    c.need(stage['numeric_count']==145 and stage['prior_build_calls']==0
        and stage['fit_y_separate_file_opened'] is False and stage['same_factors_weights_and_bias'] is True,'New arm roles differ')
    c.need(stage['source_R100_plan_sha256']==c.PLAN_SHA and stage['source_input_arrays']==source,'Original R100 plan/input binding')
    c.need(stage['source_prepared_R100T']==oldstage['arms']['T']['prepared']
        and preparation['source_prepared_R100T']==oldstage['arms']['T']['prepared'],'Original R100T NPZ authority')
    c.need(preparation['source_input_arrays']==source and preparation['other_seven_arrays_bitwise_unchanged'] is True
        and preparation['only_arm_numeric_column_difference']==144 and preparation['date_weights_renormalized'] is False,'Prepared input contract')
    c.need(preparation['original_144_prefix_identity']=={key:source[key] for key in sorted(NUMERIC)},'Original numeric prefix differs')
    c.need(scale['format']=='R102_log_temporal_unit_prior_v1'
        and scale['feature']=='standardized_log_temporal_unit_prior'
        and scale['numeric_column_position_zero_based']==144 and scale['fit_rows']==n and scale['query_rows']==q,'Scale schema differs')
    c.need(np.isfinite(scale['mean_log_float64']) and np.isfinite(scale['std_log_ddof0_float64'])
        and scale['std_log_ddof0_float64']>0 and scale['query_scaling_uses_fit_only'] is True
        and scale['clipping'] is False,'Scale must use fit-only finite mean/positive std')
    for split,rows in [('train',n),('query',q)]:
        for stem,dtype in [('prior','float64'),('feature','float32')]:
            ident=scale[f'{split}_{stem}_identity']
            c.need(ident['dtype']==dtype and ident['shape']==[rows] and len(ident['values_sha256'])==64,'Scale vector identity')
        zero=c.identity(np.zeros(rows,dtype=np.float32))
        c.need(preparation['arm_feature_identity']['C'][split]==zero,'C feature must be positive float32 zero')
        c.need(preparation['arm_feature_identity']['T'][split]==scale[f'{split}_feature_identity'],'T scaled feature identity')
    base_configuration=training['source']['configuration'];bias=training['source']['initial_head_bias']
    median=training['source']['initial_training_median_ratio']
    for arm in ['C','T']:
        ar=stage['arms'][arm];tr=training[arm];a=ar['all_input_arrays']
        c.need(set(a)==set(c.ARRAY_NAMES),'Exactly nine trainer arrays')
        c.need(tr['status']=='completed' and tr['fit_rows']==n and tr['query_rows']==q
            and tr['numeric_count']==145 and tr['saved_model_reload_exact'] is True and tr['reload_rows']==q,'Training terminal/width/reload differs')
        c.need(ar['saved_model_reload_exact'] is True and ar['reload_rows']==q,'Arm reload certificate')
        c.need(tr['configuration']==base_configuration and tr['configuration']['seed']==42
            and tr['configuration']['epochs']==30 and tr['configuration']['batch_size']==2048,'Immutable trainer configuration')
        c.need(tr['source_sha256']==c.TRAINER_SHA and tr['input_sha256']==ar['prepared']['sha256'],'Trainer/prepared authority')
        c.need(tr['query_labels_available'] is False and tr['query_labels_used'] is False
            and tr['sample_weights_recomputed_in_trainer'] is False and tr['sample_weights_batch_renormalized'] is False,'Training boundary/weight changes')
        c.need(tr['price_or_ratio_clipping'] is False and [e['epoch'] for e in tr['epochs']]==list(range(1,31)),'Complete unclipped fixed-epoch training')
        c.need(tr['initial_head_bias']==bias and tr['initial_training_median_ratio']==median,'Same factors/labels must give same initialization')
        c.need(tr['cardinalities']==training['source']['cardinalities']
            and tr['parent_trainer_source_sha256']==training['source']['parent_trainer_source_sha256']
            and tr['architecture_source_sha256']==training['source']['architecture_source_sha256'],'Architecture/category identity')
        for key in SHARED:c.need(a[key]==source[key],'Seven shared arrays changed: '+key)
        for split,rows in [('train',n),('query',q)]:
            rec=a[split+'_num'];c.need(rec['dtype']=='float32' and rec['shape']==[rows,145] and len(rec['values_sha256'])==64,'Numeric matrix width')
        for name,key in [('train_price_factor_million','train_price_factor_million_sha256'),
                         ('query_price_factor_million','query_price_factor_million_sha256'),
                         ('train_sample_weight','train_sample_weight_sha256')]:
            c.need(tr[key]==source[name]['values_sha256'],'Factor/date weight identity differs')
    for name in SHARED:c.need(stage['arms']['C']['all_input_arrays'][name]==stage['arms']['T']['all_input_arrays'][name],'C/T nonnumeric difference')

def research_gates(stages):
    primary=metric.research_gate(stages)
    native=all(stages[s]['layer_metrics']['T']['native']['overall_mae']<stages[s]['layer_metrics']['C']['native']['overall_mae'] for s in STAGES)
    overall=all(stages[s]['metrics']['T']['overall_mae']<stages[s]['metrics']['C']['overall_mae'] for s in STAGES)
    weighted=all(stages[s]['metrics']['T']['weighted_mae']<stages[s]['metrics']['C']['weighted_mae'] for s in STAGES)
    mechanism={'components':{'native_all_three_overall_improve':native,'full_all_three_overall_improve':overall,
        'full_all_three_weighted_improve':weighted},'mechanism_gate_passed':native and overall and weighted,
        'C_role':'same-width zero-input; same R100T factors and loss rows','causal_identification_claimed':False}
    return {'primary':primary,'mechanism':mechanism,
        'both_research_gates_passed':primary['adoption_research_gate_passed'] and mechanism['mechanism_gate_passed'],
        'automatic_adoption':False,'selected_submission':'v22','Private_score':None,'new_CSV':False}

def paired(*args,**kwargs):
    result,arrays=metric.paired(*args,**kwargs)
    result['caution']=CAUTION
    return result,arrays

def preflight(sealed_path,request_path):
    """Entire R102 prelabel gate. It has no raw-price deserialization call."""
    checked={}
    def bind(path,digest):
        name=c.rel(path);c.need(name not in checked or checked[name]==digest,'Conflicting file authority')
        c.need(c.sha(name)==digest,'Input SHA differs: '+name);checked[name]=digest;return c.safe(name)
    def document(rec):return c.read(bind(rec['path'],rec['sha256']))
    def array(rec,semantic=True):
        a=np.load(bind(rec['path'],rec['sha256']),allow_pickle=False)
        c.need(isinstance(a,np.ndarray) and not a.dtype.hasobject,'Numeric array required')
        if semantic:c.need(c.identity(a)=={k:rec[k] for k in ['dtype','shape','values_sha256']},'Array identity differs')
        return a
    def terminal(path):
        p=c.safe(path);digest=p.with_suffix('.sha256').read_text().strip();v=document({'path':c.rel(p),'sha256':digest})
        c.need(v['exit_code']==0 and v.get('failure') is None,'Process has not completed successfully');return v
    request=c.read(request_path)
    c.need(request['schema_version']==1 and request['plan']=={'path':PLAN,'sha256':PLAN_SHA}
        and request['parameters']==parameters(),'Evaluation request/plan/parameters differ')
    bind(PLAN,PLAN_SHA);bind(request_path,c.sha(request_path))
    own_sources=source_hashes()
    c.need(all(request['source_sha256'].get(k)==v for k,v in own_sources.items()),'Frozen evaluator closure differs')
    c.verify_hashes(request['source_sha256'])
    c.need(request['protected_sha256']==c.protected_hashes(),'Protected files changed before evaluation')
    launch=document(request['launch_request']);c.need(launch['plan_sha256']==PLAN_SHA,'Scientific request is not R102')
    c.verify_hashes(launch['source_sha256'])
    seal=c.read(sealed_path);bind(sealed_path,c.sha(sealed_path));check_header(seal)
    c.need(seal['request']==request['launch_request'] and seal['protected_after']==request['protected_sha256'],'Sealed request/protected differs')
    c.verify_hashes(seal['source_sha256'])
    c.need(all(launch['source_sha256'].get(k)==v for k,v in seal['source_sha256'].items()),'Main source not in prefit request')
    c.need(c.rel(sealed_path)=='outputs/round102/sealed_manifest.json','Unexpected main seal path')
    terminal('outputs/round102/execution.json')
    c.need(request['R100_seal']==R100_SEAL,'Original R100 seal authority changed')
    oldmain=document(R100_SEAL);oldresults=document(R100_EVALUATION)
    c.need(oldmain['plan_sha256']==c.PLAN_SHA and oldmain['status']=='all_temporal_prior_predictions_sealed','R100 identity is distinct from R102')
    c.verify_hashes(oldmain['source_sha256'])
    terminal('outputs/round100/execution.json');terminal('outputs/round100/evaluation/execution.json')
    for key in ['features','canonical_mapping','raw_train']:
        c.need(seal[key]==oldmain[key],'Dataset authority differs');bind(seal[key]['path'],seal[key]['sha256'])
    c.need(seal['raw_train']['path']==c.RAW and seal['features']['path']==c.FEATURES and seal['canonical_mapping']['path']==c.CANONICAL,'Fixed data paths')
    mapping=array(seal['canonical_mapping'],semantic=False);metric.index(mapping,455784,'canonical mapping');c.need(len(mapping)==455255,'Canonical rows')
    Xraw=pd.read_pickle(c.safe(c.FEATURES))
    c.need(len(Xraw)==455784 and '總價元' not in Xraw and Xraw.index.equals(pd.RangeIndex(len(Xraw))),'Pure X schema')
    days=Xraw.iloc[mapping].trade_day_index.to_numpy(dtype=np.float64)
    stages={}
    for stage in STAGES:
        sr=seal['stages'][stage];sm=document(sr['stage_manifest']);oldstage=document(oldmain['stages'][stage]['stage_manifest'])
        ex=document(sr['execution']);c.need(ex['exit_code']==0 and ex.get('failure') is None,'Stage not terminal')
        c.need(sm['stage']==stage and sm['status']=='prior_input_stage_predictions_sealed' and sm['plan_sha256']==PLAN_SHA
            and sm['source_R100_plan_sha256']==c.PLAN_SHA and sm['new_NN_fits']==2 and sm['new_native_tree_fits']==0,'Stage roles/plan')
        c.need(sm['source_R100_seal']==R100_SEAL and sm['source_R100_stage_seal']==oldmain['stages'][stage]['stage_manifest'],'Old stage authority')
        c.need(sm['all_query_reload_exact'] is True and sm['query_scores_computed'] is False
            and sm['raw_price_pickle_opened'] is False and sm['other_stage_fit_labels_opened'] is False,'Stage target boundary')
        for name,digest in sm['artifact_sha256'].items():bind(name,digest)
        c.need(sm['request']==seal['request'] and sm['protected_before']==sm['protected_after']==request['protected_sha256'],'Stage provenance')
        start,end,nf,nq=c.SPLITS[stage]
        c.need((sm['cutoff'],sm['query_end'],sm['fit_rows'],sm['query_rows'])==(start,end,nf,nq),'Original outer window changed')
        for key in ['features','canonical_mapping','raw_train']:c.need(sm[key]==seal[key],'Stage data authority')
        ix={name:array(sm[name]) for name in ['fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices','eligible_positions']}
        for name,value in ix.items():c.exact(value,array(oldstage[name]),'R100 indices/mask changed: '+name)
        fi,qi=ix['fit_indices'],ix['query_indices']
        metric.check_temporal(stage,fi,qi,ix['query_source_cache_indices'],days,mapping)
        c.exact(mapping[fi],ix['fit_source_cache_indices'],'Raw fit mapping')
        c.need(sm['eligible_rows']==oldstage['eligible_rows']==len(ix['eligible_positions']),'Loss rows changed')
        prep=document(sm['preparation']);scale=document(sm['scale_state'])
        c.need(prep['scale_state']==sm['scale_state'],'Scale record differs')
        bind(sm['source_prepared_R100T']['path'],sm['source_prepared_R100T']['sha256'])
        train={'source':document(oldstage['arms']['T']['training'])}
        for arm in ['C','T']:
            rec=sm['arms'][arm]
            for name in ['prepared','model']:bind(rec[name]['path'],rec[name]['sha256'])
            train[arm]=document(rec['training']);ax=document(rec['execution'])
            c.need(ax['exit_code']==0 and ax.get('failure') is None,'Arm training not terminal')
        check_scale_and_arrays(sm,prep,scale,oldstage,train)
        c.need(prep['prior_records']['train']==oldstage['temporal_prior'],'Train prior source changed')
        queryprior=f'outputs/round60/{stage}/query_unit_prior.npy'
        c.need(prep['prior_records']['query']['path']==queryprior
            and prep['prior_records']['query']['sha256']==oldstage['original_input_sha256'][queryprior],'Full fit query prior changed')
        for rec in prep['prior_records'].values():bind(rec['path'],rec['sha256'])
        for split,rec in prep['prior_records'].items():
            if all(k in rec for k in ['dtype','shape','values_sha256']):
                c.need(scale[split+'_prior_identity']=={k:rec[k] for k in ['dtype','shape','values_sha256']},'Scale prior identity mismatch')
        fixed={name:array(rec) for name,rec in sm['fixed_branches'].items()}
        c.need(set(fixed)=={'area','lgb','H','U'},'Fixed branch names')
        for name,v in fixed.items():c.exact(v,array(oldstage['fixed_branches'][name]),'Immutable fixed branch differs')
        pred={name:array(rec) for name,rec in sm['predictions'].items()}
        c.need(set(pred)=={'v22','v24','C','T'} and sr['predictions']==sm['predictions'],'Four full predictions required')
        layers={arm:{name:array(rec) for name,rec in records.items()} for arm,records in sm['layers'].items()}
        c.need(set(layers)=={'v22','C','T'} and all(set(v)=={'native','base','reference','spatial'} for v in layers.values()),'Layer schema')
        for v in [*fixed.values(),*pred.values(),*[x for a in layers.values() for x in a.values()]]:metric.prices(v,(nq,),'saved query')
        metric.check_formulas(pred,layers,fixed)
        for arm in ['v22','v24']:c.exact(pred[arm],array(oldstage['predictions'][arm]),'Complete original control differs')
        for name,v in layers['v22'].items():c.exact(v,array(oldstage['layers']['v22'][name]),'Original NN branch differs')
        for arm in ['C','T']:c.exact(layers[arm]['native'],array(sm['arms'][arm]['native_prediction']),'Native trainer vector differs')
        stages[stage]={'fit':Xraw.iloc[mapping[fi]].reset_index(drop=True),'query':Xraw.iloc[mapping[qi]].reset_index(drop=True),
            'query_indices':qi,'query_source_cache_indices':ix['query_source_cache_indices'],'eligible_rows':sm['eligible_rows'],
            'predictions':pred,'layers':layers,'scale_state_sha256':sm['scale_state']['sha256']}
    del Xraw
    c.need(c.protected_hashes()==request['protected_sha256'],'Protected changed during prelabel gate')
    return {'sealed_manifest':c.record(sealed_path),'evaluation_request':c.record(request_path),
        'input_sha256':checked,'source_sha256':own_sources,'protected_before':request['protected_sha256'],
        'stages':stages,'raw_train':seal['raw_train'],'old_results':oldresults}

def render(result):
    rows=['# R102：明示價格先驗的同寬度對照評估','','C=145欄正零對照；T=145欄 log-prior。两組同 R100T factor／loss 列，不是舊 R100 C。','',
        '|窗口|T MAE|C MAE|v22 MAE|T−v22|T−C|native T−C|','|---|---:|---:|---:|---:|---:|---:|']
    for s in STAGES:
        a=result['stages'][s];m=a['metrics'];delta=a['layer_metrics']['T']['native']['overall_mae']-a['layer_metrics']['C']['native']['overall_mae']
        rows.append(f"|{s}|{m['T']['overall_mae']:.2f}|{m['C']['overall_mae']:.2f}|{m['v22']['overall_mae']:.2f}|{m['T']['overall_mae']-m['v22']['overall_mae']:.2f}|{m['T']['overall_mae']-m['C']['overall_mae']:.2f}|{delta:.2f}|")
    rows+=['',f"Primary research gate: {result['research_gates']['primary']['adoption_research_gate_passed']}；mechanism gate: {result['research_gates']['mechanism']['mechanism_gate_passed']}。兩門檻分開保留，不用後處理收益替代原生改善。",'',
        '四完整向量／三支 NN 分支的五層指標、全部城市／月份／季度／價帶／尾部、24 CI 與條件抽樣陣列見 [完整結果](results.json)。', '',CAUTION,
        '', '沒有全量 test 預測或 CSV；Private 未知，v22 唯一提交不變。']
    return '\n'.join(rows)+'\n'

def run(sealed_path,request_path,outpath):
    out=c.safe(outpath);c.need(not out.exists(),'Preserve existing evaluation/failure outputs');out.mkdir(parents=True)
    tick=time.monotonic();failure=None;opened=False
    try:
        checked=preflight(sealed_path,request_path)
        c.write(out/'prelabel_gate.json',{'status':'all_R102_six_fits_and_three_stages_verified_before_price_deserialization',
            'checked_stages':list(STAGES),'sealed_manifest':checked['sealed_manifest'],'source_sha256':checked['source_sha256'],
            'input_sha256':checked['input_sha256'],'NPZ_and_models_hashed_not_deserialized':True,
            'same_width_145_seven_shared_R100T_arrays':True,'completed_at_utc':c.now()})
        c.need(c.sha(checked['raw_train']['path'])==checked['raw_train']['sha256'],'Raw price changed after gate')
        raw=pd.read_pickle(c.safe(checked['raw_train']['path']));opened=True;stages={}
        for s in STAGES:
            a=checked['stages'][s];y=metric.select_query_y(raw,a['query_source_cache_indices'])
            score=lambda p:metric.descriptive_metrics(y,p,a['fit'],a['query'],c.SPLITS[s][0])
            metrics={name:score(v) for name,v in a['predictions'].items()}
            layers={arm:{name:score(v) for name,v in paths.items()} for arm,paths in a['layers'].items()}
            for arm in layers:layers[arm]['median']=metrics[arm]
            for name in ['v22','v24']:c.need(metrics[name]==checked['old_results']['stages'][s]['metrics'][name],'Full R100 control score replay differs')
            folder=out/s;folder.mkdir();boots={};replicates={}
            for control in ['v22','C']:
                item,samples=paired(y,a['predictions'],a['fit'],a['query'],a['query_indices'],s,control)
                for family in item['methods'].values():
                    for m,k in [('overall','overall_mae'),('weighted','weighted_mae')]:
                        z=family['summary'][m];c.need(abs(z['control_mae']-metrics[control][k])<=1e-6 and abs(z['candidate_mae']-metrics['T'][k])<=1e-6,'Paired point differs')
                boots[control]=item;replicates[control]={f:c.save_array(folder/f'T_vs_{control}_{f}_replicates.npy',v) for f,v in samples.items()}
            background={arm:checked['old_results']['stages'][s]['layer_metrics'][arm]['native']['overall_mae'] for arm in ['v22','C','T']}
            item={'stage':s,'cutoff':c.SPLITS[s][0],'query_end':c.SPLITS[s][1],'fit_rows':len(a['fit']),
                'eligible_rows':a['eligible_rows'],'query_rows':len(y),'metrics':metrics,'layer_metrics':layers,
                'comparison':metric.comparisons(metrics),'comparison_vs_C':metric.comparisons(metrics,'C'),
                'comparison_vs_v24':metric.comparisons(metrics,'v24'),'C_vs_v22':metric.comparisons(metrics,'v22','C'),
                'paired':boots,'replicate_artifacts':replicates,'query_target_values_sha256':c.values_sha(y),
                'query_target_values_persisted':False,'group_membership_uses_original_outer_fit':True,
                'full_original_v22_and_v24_metrics_replay_exact':True,
                'historical_R100_native_MAE_background_only':background,'R102_C_is_not_R100_C':True}
            c.write(folder/'results.json',item);stages[s]=item
            print(json.dumps({'stage':s,'T_minus_v22':item['comparison']['overall_mae']['delta'],'T_minus_C':item['comparison_vs_C']['overall_mae']['delta']}),flush=True)
        del raw
        c.verify_hashes(checked['input_sha256']);c.need(source_hashes()==checked['source_sha256'] and c.protected_hashes()==checked['protected_before'],'Source/protected changed')
        result={'round':102,'status':'completed_three_window_six_fit_evaluation','created_at_utc':c.now(),
            'parameters':parameters(),'checked_stages':list(STAGES),'stages':stages,'research_gates':research_gates(stages),
            'sealed_manifest':checked['sealed_manifest'],'evaluation_request':checked['evaluation_request'],
            'input_sha256':checked['input_sha256'],'source_sha256':checked['source_sha256'],
            'protected_before':checked['protected_before'],'protected_after':c.protected_hashes(),
            'NPZ_and_models_hashed_not_deserialized':True,'raw_pickle_whole_table_deserialized_after_all_seals':True,
            'query_labels_used_in_training_or_predictions':False,'query_target_values_persisted':False,
            'query_y_mapping':"raw_train.iloc[canonical_mapping[query_indices]]['總價元']",
            'conditional_intervals_reported':24,'C_vs_v22_and_T_vs_v24_additional_bootstrap_calls':0,
            'model_fit_load_predict_calls':0,'new_fits_in_scorer':0,'test_files_read':0,'new_CSV':False,
            'selected_submission':'v22','Private_score':None,'fresh_blind_validation':False,'caution':CAUTION,
            'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'elapsed_seconds':time.monotonic()-tick}
        with (out/'report.md').open('x',encoding='utf8',newline='\n') as f:f.write(render(result))
        result['artifact_sha256']={c.rel(p):c.sha(p) for p in out.rglob('*') if p.is_file()}
        c.write(out/'results.json',result);return result
    except Exception:failure=traceback.format_exc();raise
    finally:c.write(out/'execution.json',{'exit_code':int(failure is not None),'failure':failure,
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':c.sha(__file__),'raw_prices_deserialized':opened,
        'new_fits':0,'native_model_load_predict_calls':0,'automatic_retry':False})

def synthetic_metadata():
    n,q=4,3
    ident=lambda shape,dtype='float32',v='a':{'dtype':dtype,'shape':shape,'values_sha256':v*64}
    source={k:ident([n] if k.startswith('train_') else [q]) for k in c.ARRAY_NAMES}
    source.update(train_num=ident([n,144]),query_num=ident([q,144]),train_cat=ident([n,23],'int64'),
        query_cat=ident([q,23],'int64'),cardinalities=ident([23],'int64'))
    rec={'path':'synthetic/prepared.npz','sha256':'b'*64};old={'arms':{'T':{'all_input_arrays':source,'prepared':rec}}}
    scale={'format':'R102_log_temporal_unit_prior_v1','feature':'standardized_log_temporal_unit_prior',
        'numeric_column_position_zero_based':144,'fit_rows':n,'query_rows':q,'mean_log_float64':10.,'std_log_ddof0_float64':1.,
        'query_scaling_uses_fit_only':True,'clipping':False}
    for split,rows in [('train',n),('query',q)]:
        scale[split+'_prior_identity']=ident([rows],'float64');scale[split+'_feature_identity']=ident([rows])
    prep={'source_prepared_R100T':rec,'source_input_arrays':source,'other_seven_arrays_bitwise_unchanged':True,
        'only_arm_numeric_column_difference':144,'date_weights_renormalized':False,'original_144_prefix_identity':{k:source[k] for k in NUMERIC},
        'arm_feature_identity':{'C':{s:c.identity(np.zeros(r,dtype=np.float32)) for s,r in [('train',n),('query',q)]},
            'T':{s:scale[s+'_feature_identity'] for s in ['train','query']}}}
    stage={'eligible_rows':n,'query_rows':q,'numeric_count':145,'prior_build_calls':0,'fit_y_separate_file_opened':False,
        'same_factors_weights_and_bias':True,'source_R100_plan_sha256':c.PLAN_SHA,'source_input_arrays':source,
        'source_prepared_R100T':rec,'arms':{}}
    tr={'configuration':{'seed':42,'epochs':30,'batch_size':2048},'initial_head_bias':.2,'initial_training_median_ratio':.8,
        'cardinalities':[1]*23,'parent_trainer_source_sha256':'x','architecture_source_sha256':'y'}
    training={'source':copy.deepcopy(tr)}
    for arm in ['C','T']:
        arrays=copy.deepcopy(source);arrays.update(train_num=ident([n,145]),query_num=ident([q,145]))
        stage['arms'][arm]={'all_input_arrays':arrays,'prepared':rec,'saved_model_reload_exact':True,'reload_rows':q}
        training[arm]={**copy.deepcopy(tr),'status':'completed','fit_rows':n,'query_rows':q,'numeric_count':145,
            'saved_model_reload_exact':True,'reload_rows':q,'source_sha256':c.TRAINER_SHA,'input_sha256':rec['sha256'],
            'query_labels_available':False,'query_labels_used':False,'sample_weights_recomputed_in_trainer':False,
            'price_or_ratio_clipping':False,'epochs':[{'epoch':i} for i in range(1,31)],
            'sample_weights_batch_renormalized':False,'train_price_factor_million_sha256':source['train_price_factor_million']['values_sha256'],
            'query_price_factor_million_sha256':source['query_price_factor_million']['values_sha256'],
            'train_sample_weight_sha256':source['train_sample_weight']['values_sha256']}
    # Real certificates are separately decoded JSON documents, never shared objects.
    return tuple(copy.deepcopy(v) for v in (stage,prep,scale,old,training))

def self_check():
    tick=time.monotonic();checks=[]
    def mark(name,ok):c.need(ok,name);checks.append(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name)
    args=synthetic_metadata();check_scale_and_arrays(*args);checks.append('145_same_R100T_seven_arrays_and_bias_accept')
    for name,mutate in [
        ('width144',lambda a:a[0].update(numeric_count=144)),
        ('old_R100_plan_replaced',lambda a:a[0].update(source_R100_plan_sha256=PLAN_SHA)),
        ('factor_changed',lambda a:a[0]['arms']['T']['all_input_arrays']['train_price_factor_million'].update(values_sha256='f'*64)),
        ('date_renormalized',lambda a:a[1].update(date_weights_renormalized=True)),
        ('query_scale_fit',lambda a:a[2].update(query_scaling_uses_fit_only=False)),
        ('std_zero',lambda a:a[2].update(std_log_ddof0_float64=0)),
        ('clipped_scale',lambda a:a[2].update(clipping=True)),
        ('C_not_positive_zero',lambda a:a[1]['arm_feature_identity']['C'].update(train=c.identity(np.full(4,-0.,dtype=np.float32)))),
        ('bias_differs',lambda a:a[4]['T'].update(initial_head_bias=1.)),
        ('query_labels_used',lambda a:a[4]['T'].update(query_labels_used=True)),
        ('trainer_source',lambda a:a[4]['C'].update(source_sha256='0'*64)),
        ('epoch_changed',lambda a:a[4]['T']['configuration'].update(epochs=31)),
        ('prefix_changed',lambda a:a[1]['original_144_prefix_identity']['train_num'].update(shape=[4,143])),
        ('query_feature_changed',lambda a:a[1]['arm_feature_identity']['T']['query'].update(values_sha256='f'*64))]:
        a=copy.deepcopy(synthetic_metadata());mutate(a);rejects('reject_'+name,lambda a=a:check_scale_and_arrays(*a))
    header={'schema_version':1,'status':'all_prior_input_predictions_sealed','stage_order':list(STAGES),
        'stages':dict.fromkeys(STAGES,{}),'plan_sha256':PLAN_SHA,'new_NN_fits':6,'new_native_tree_fits':0,
        'all_query_reload_exact':True,'query_scores_computed':False,'protected_before':dict.fromkeys(range(9),'x'),
        'protected_after':dict.fromkeys(range(9),'x')}
    check_header(header);checks.append('all_three_six_fit_header')
    for key,val in [('new_NN_fits',5),('query_scores_computed',True),('status','all_temporal_prior_predictions_sealed'),('plan_sha256',c.PLAN_SHA)]:
        h=copy.deepcopy(header);h[key]=val;rejects('reject_header_'+key,lambda h=h:check_header(h))
    stages={s:{'metrics':{'T':{'overall_mae':90.,'weighted_mae':90.},'C':{'overall_mae':100.,'weighted_mae':100.},'v22':{'overall_mae':100.,'weighted_mae':100.}},
        'comparison':{'quarter_or_large_group_deterioration_above_5pct':[]},'layer_metrics':{'T':{'native':{'overall_mae':90.}},'C':{'native':{'overall_mae':100.}}}} for s in STAGES}
    mark('both_gates_positive',research_gates(stages)['both_research_gates_passed'])
    stages['development']['layer_metrics']['T']['native']['overall_mae']=101.
    g=research_gates(stages);mark('native_failure_not_masked_by_full_improvement',g['primary']['adoption_research_gate_passed'] and not g['mechanism']['mechanism_gate_passed'])
    stages['development']['layer_metrics']['T']['native']['overall_mae']=90.
    stages['confirmation']['metrics']['T']['overall_mae']=99.6
    g=research_gates(stages);mark('half_percent_primary_separate_from_mechanism',not g['primary']['adoption_research_gate_passed'] and g['mechanism']['mechanism_gate_passed'])
    stages['confirmation']['metrics']['T']['overall_mae']=90.
    stages['auxiliary']['comparison']['quarter_or_large_group_deterioration_above_5pct']=[{'synthetic':True}]
    mark('large_group_primary_gate',not research_gates(stages)['primary']['adoption_research_gate_passed'])
    saved={'area':np.array([10.,12.,20.]),'lgb':np.array([9.,14.,18.]),'H':np.array([12.,13.,19.]),'U':np.array([9.,15.,21.])}
    pred={};layers={}
    for arm,delta in [('v22',0.),('C',1.),('T',-1.)]:
        native=np.array([11.,12.,21.])+delta;base=.75*(.5*saved['area']+.5*native)+.25*saved['lgb']
        layers[arm]={'native':native,'base':base,'reference':base+.5,'spatial':base+1}
        pred[arm]=np.median(np.vstack([layers[arm]['spatial'],saved['H'],saved['U']]),axis=0)
    metric.check_formulas(pred,layers,saved);checks.append('full_base_and_three_branch_median')
    wrong=copy.deepcopy(pred);wrong['T']=wrong['T']+.01;rejects('reject_final_median_change',lambda:metric.check_formulas(wrong,layers,saved))
    raw=pd.DataFrame({'總價元':[3.,5.,7.,11.]})
    mark('query_raw_mapping_exact',np.array_equal(metric.select_query_y(raw,np.array([1,3],dtype=np.int64),4),[5.,11.]))
    dates=pd.to_datetime(['2023-08-11','2023-09-10','2023-09-11','2024-08-10'])
    mark('relative_month_boundaries',np.array_equal(metric.methods.relative_months((dates-pd.Timestamp('2000-01-01')).days,'2023-08-11'),[0,0,1,11]))
    return {'status':'passed','check_count':len(checks),'checks':checks,'parameters':parameters(),'plan_sha256':PLAN_SHA,
        'source_sha256':c.sha(__file__),'source_hashes':source_hashes(),'real_data_read':False,'real_fit_score_calls':0,
        'elapsed_seconds':time.monotonic()-tick}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed-manifest');parser.add_argument('--evaluation-request');parser.add_argument('--out')
    parser.add_argument('--self-check',action='store_true');parser.add_argument('--receipt')
    a=parser.parse_args()
    if a.self_check:
        result=self_check()
        if a.receipt:c.write(a.receipt,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    else:
        if not all([a.sealed_manifest,a.evaluation_request,a.out]):parser.error('sealed-manifest, evaluation-request and out required')
        run(a.sealed_manifest,a.evaluation_request,a.out)
