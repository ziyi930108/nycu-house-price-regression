"""R95 evidence-only review: JSON/SHA, saved CPU weights and existing CI replicates.

No X/target/prepared arrays, native forward, fit, scoring or resampling is used.
Run only after all five pipeline steps complete. Existing artifacts are immutable.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAGES = ('development', 'confirmation', 'auxiliary')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    records = {}
    expected = {}

    def bind(name, digest):
        assert name not in expected or expected[name] == digest, name
        expected[name] = digest

    def read(name):
        path = ROOT / name
        digest = sha(path)
        records[name] = digest
        side = path.with_suffix('.sha256')
        if side.exists():
            assert side.read_text(encoding='utf-8').strip() == digest, name
            records[side.relative_to(ROOT).as_posix()] = sha(side)
        return json.loads(path.read_text(encoding='utf-8'))

    pipeline = read('outputs/round95_launcher_20260929/pipeline_execution.json')
    assert pipeline['exit_code'] == 0 and not pipeline['failure']
    assert [x['step'] for x in pipeline['steps']] == ['main','diagnostics','paired_v22','paired_v24','review']
    assert all(x['exit_code'] == 0 for x in pipeline['steps'])
    result = read('outputs/round95/results.json')
    plan = read('outputs/round95/plan.json')
    decision = read('outputs/round95/decision_review.json')
    diag = read('reports/round95_diagnostics.json')
    execution = read('outputs/round95/execution.json')
    diag_execution = read('outputs/round95/diagnostics/execution.json')
    assert execution['exit_code'] == diag_execution['exit_code'] == 0
    assert result['status'] == 'three_periods_completed_no_adoption_decision' and result['checked_stages'] == list(STAGES)
    assert result['new_NN_fits'] == 3 and result['new_native_tree_fits'] == 0
    assert result['protected_before'] == result['protected_after'] == plan['protected_before']
    assert result['plan_sha256'] == sha(ROOT/'outputs/round95/plan.json')
    assert result['text_plan_sha256'] == sha(ROOT/'reports/round94_plan.md')
    assert result['input_sha256'] == plan['input_sha256']
    assert result['source_sha256'] == plan['source_sha256']
    assert result['frozen_source_and_plan'] == plan['frozen_source_and_plan']
    assert not any(result[k] for k in ('new_CSV','new_test_predictions','automatic_adoption',
        'new_reference_or_spatial_states','Public_feedback_used_for_tuning','scientific_configuration_changed'))
    assert not decision['new_CSV'] and not decision['new_test_predictions'] and decision['private_score'] is None
    for group in (result['input_sha256'],result['source_sha256'],result['protected_before'],
                  result['frozen_source_and_plan'],diag['input_sha256'],diag['source_sha256'],decision['evidence_sha256']):
        for name,digest in group.items(): bind(name,digest)
    bind('src/round95_review.py',decision['review_source_sha256'])
    trainer='src/round94_attention_prior_train_v2.py'
    tree=ast.parse((ROOT/trainer).read_text(encoding='utf-8'))
    config=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    assert config == plan['configuration']
    summaries = {}
    max_aggregation_difference=0.
    for stage in STAGES:
        row=result['stages'][stage]
        stage_record=read(f'outputs/round95/{stage}/results.json')
        assert row==stage_record
        for name,digest in row['artifact_sha256'].items():bind(name,digest)
        for key in ('R71_seed42_full_chain_and_diagnostics_exact','controls_full_metrics_exact',
                    'full_query_raw_ratio_TWD_reload_exact','prepared_provenance_replay_exact','v22_full_median_exact'):
            assert row[key]
        training=read(f'outputs/round95/{stage}/model/training.json')
        prepared=read(f'outputs/round95/{stage}/prepared_verification.json')
        child=read(f'outputs/round95/{stage}/trainer_process/execution.json')
        te=read(f'outputs/round95/{stage}/model/execution.json')
        assert child['exit_code']==te['exit_code']==0 and not child['automatic_retry']
        assert training['status']=='completed' and training['configuration']==config
        assert training['source_sha256']==training['architecture_source_sha256']==sha(ROOT/trainer)
        assert training['plan_sha256']==result['text_plan_sha256']
        assert training['configuration_sha256']==plan['configuration_sha256']==te['configuration_sha256']
        assert training['all_input_arrays']==prepared['all_input_arrays']
        assert training['input_sha256']==prepared['prepared_sha256']
        bind(prepared['prepared_path'],prepared['prepared_sha256'])
        for name,digest in training['source_hashes'].items():bind(name,digest)
        for name,digest in training['artifact_sha256'].items():bind(f'outputs/round95/{stage}/model/{name}',digest)
        assert training['fit_rows']==row['fit_rows'] and training['query_rows']==row['query_rows']
        assert training['fit_seconds']==row['fit_seconds']
        assert [x['epoch'] for x in training['epochs']]==list(range(1,31))
        assert training['optimizer_steps']==30*math.ceil(row['fit_rows']/2048)
        assert training['numeric_count']==144 and training['categorical_count']==23
        assert all(training[k] for k in ('reload_ratio_exact','reload_TWD_exact','saved_model_reload_exact'))
        assert training['reload_rows']==row['query_rows'] and training['predict_batch_size']==256
        assert not any(training[k] for k in ('query_labels_available','query_labels_used','fit_uses_eval_set',
              'early_stopping_used','epoch_selection_uses_training_metrics','price_or_ratio_clipping'))
        assert all(prepared[k] for k in ('fit_encoder_state_replay_exact','encoded_fit_query_replay_exact',
            'OOF_fold_state_evidence_replay_exact','query_prior_replay_exact','factor_replay_exact','date_weight_replay_exact'))
        assert not prepared['query_labels_used'] and not prepared['chronological_oof']
        assert len(training['all_input_arrays'])==9 and not any('query_y' in k for k in training['all_input_arrays'])
        checkpoint_path=ROOT/f'outputs/round95/{stage}/model/model.pt'
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert checkpoint['configuration']==config and checkpoint['configuration_sha256']==plan['configuration_sha256']
        assert checkpoint['source_sha256']==training['source_sha256']
        assert checkpoint['source_dependencies']==training['source_dependencies']
        assert checkpoint['numeric_count']==144 and checkpoint['cardinalities']==training['cardinalities']
        sd=checkpoint['state_dict']
        assert len(sd)==61 and all(v.device.type=='cpu' and v.dtype==torch.float32 and
                                  bool(torch.isfinite(v).all()) for v in sd.values())
        assert list(sd['tokenizer.numeric_weight'].shape)==[144,64]
        assert list(sd['tokenizer.numeric_bias'].shape)==[144,64]
        assert list(sd['tokenizer.categorical_bias'].shape)==[23,64]
        assert list(sd['tokenizer.cls'].shape)==[64]
        for j,size in enumerate(training['cardinalities']):
            assert list(sd[f'tokenizer.embeddings.{j}.weight'].shape)==[size,64]
        count=sum(v.numel() for v in sd.values())
        assert count==training['parameter_count']==row['parameter_count']==64*sum(training['cardinalities'])+103617
        saved_shapes={k:list(v.shape) for k,v in sd.items()}
        del sd,checkpoint
        d=diag['stages'][stage]
        assert d['all_full_saved_metrics_exact']
        for key,metrics in row['metrics'].items():
            assert d['metrics'][key]==metrics
            for family in ('cities','forecast_quarters','months','strata'):
                groups=metrics[family]
                assert sum(v['rows'] for v in groups.values())==row['query_rows']
                total=sum(v['rows']*v['mae'] for v in groups.values() if v['rows'])/row['query_rows']
                err=abs(total-metrics['overall_mae'])
                max_aggregation_difference=max(max_aggregation_difference,err)
                assert err<1e-6
            weighted=sum(plan['target_group_weights'][g]*metrics['strata'][g]['mae']
                         for g in plan['target_group_weights'] if plan['target_group_weights'][g])
            assert abs(weighted-metrics['weighted_mae'])<1e-6
        assert sum(b['rows'] for b in d['price_bands'])==row['query_rows']
        bands=[]
        for band in d['price_bands']:
            bands.append({**band,'delta_v22':band['R95']-band['v22'],'delta_v24':band['R95']-band['v24'],
                'overall_delta_contribution_v22':(band['R95']-band['v22'])*band['rows']/row['query_rows']})
        delta=row['metrics']['R95']['overall_mae']-row['metrics']['v22']['overall_mae']
        assert abs(sum(b['overall_delta_contribution_v22'] for b in bands)-delta)<1e-6
        summaries[stage]={
            'fit_rows':row['fit_rows'],'query_rows':row['query_rows'],'metrics':row['metrics'],
            'nn_layer_deltas':{k:{'old':row['original_nn_branch_metrics'][k]['overall_mae'],
                'new':row['nn_branch_metrics'][k]['overall_mae'],
                'delta':row['nn_branch_metrics'][k]['overall_mae']-row['original_nn_branch_metrics'][k]['overall_mae']}
                for k in ('native','base','reference','spatial')},
            'price_bands':bands,'test_composition':d['test_composition'],
            'parameter_count':count,'saved_tensor_shapes':saved_shapes,'all_weights_CPU_float32_finite':True,
            'fit_seconds':training['fit_seconds'],'trainer_elapsed_seconds':training['elapsed_seconds'],
            'trainer_process_seconds':child['elapsed_seconds'],'stage_seconds':row['elapsed_seconds'],
            'maximum_gpu_allocated_bytes':training['maximum_gpu_allocated_bytes'],
            'maximum_gpu_reserved_bytes':training['maximum_gpu_reserved_bytes'],
            'first_and_last_epoch_training_loss':[training['epochs'][0],training['epochs'][-1]],
        }
    intervals=[]
    max_point_difference=0.
    for control in ('v22','v24'):
        paired=read(f'outputs/round95/paired_vs_{control}/results.json')
        assert paired['status']=='completed' and paired['replicates']==2000 and paired['seed']==20260925
        assert paired['source_sha256']==sha(ROOT/'src/paired_candidate_bootstrap.py')
        assert paired['methods_source_sha256']==sha(ROOT/'src/v3_cluster_stability.py')
        for stage in STAGES:
            st=paired['stages'][stage]
            assert st['point_metrics_reproduced']
            for method in ('project','month'):
                name=f'outputs/round95/paired_vs_{control}/{stage}_{method}_replicates.npy'
                records[name]=sha(ROOT/name)
                values=np.load(ROOT/name,allow_pickle=False)
                assert values.dtype==np.float64 and values.shape==(2000,4) and np.isfinite(values).all()
                for metric,left,right in (('overall',0,1),('weighted',2,3)):
                    old=st[method]['summary'][metric]
                    samples=values[:,right]-values[:,left]
                    bounds=np.quantile(samples,[.025,.975],method='linear')
                    assert bounds.tolist()==[old['percentile_2_5'],old['percentile_97_5']]
                    assert float(samples.mean())==old['bootstrap_mean_delta']
                    assert float((samples<0).mean())==old['negative_delta_fraction']
                    assert old['valid_replicates']==2000 and old['invalid_replicates']==0
                    metric_key=metric+'_mae'
                    r=result['stages'][stage]['metrics']
                    point=r['R95'][metric_key]-r[control][metric_key]
                    max_point_difference=max(max_point_difference,abs(point-old['point_delta']))
                    assert abs(point-old['point_delta'])<1e-6
                    intervals.append({'control':control,'stage':stage,'method':method,'metric':metric,
                        'point_delta':old['point_delta'],'lower':float(bounds[0]),'upper':float(bounds[1]),
                        'classification':'negative' if bounds[1]<0 else 'positive' if bounds[0]>0 else 'crosses_zero'})
    assert len(intervals)==24
    for i,j in zip(intervals,decision['paired_intervals']):
        assert all(i[k]==j[k] for k in ('control','stage','method','metric','point_delta','lower','upper'))
    gate={
        'all_three_overall_improve':all(summaries[s]['metrics']['R95']['overall_mae']<summaries[s]['metrics']['v22']['overall_mae'] for s in STAGES),
        'all_three_weighted_improve':all(summaries[s]['metrics']['R95']['weighted_mae']<summaries[s]['metrics']['v22']['weighted_mae'] for s in STAGES),
        'latest_at_least_half_percent':summaries['confirmation']['metrics']['R95']['overall_mae']<=.995*summaries['confirmation']['metrics']['v22']['overall_mae']}
    assert gate==decision['gate_components'] and all(gate.values())==decision['adoption_gate_passed']
    fit_sum=sum(summaries[s]['fit_seconds'] for s in STAGES)
    assert fit_sum==decision['fit_seconds']
    for name,digest in expected.items():assert sha(ROOT/name)==digest,name
    for name,digest in records.items():assert sha(ROOT/name)==digest,name
    report_path=ROOT/'reports/round95_report.md'
    records['reports/round95_report.md']=sha(report_path)
    audit={'status':'completed_read_only_independent_review','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(__file__),'read_evidence_sha256':records,'verified_dependency_artifact_sha256':expected,
        'unique_verified_dependency_artifact_files':len(expected),'stage_summary':summaries,
        'predefined_gate':gate,'gate_passed':all(gate.values()),'paired_intervals':intervals,
        'CI_sign_counts':{c:{cl:sum(i['control']==c and i['classification']==cl for i in intervals)
           for cl in ('negative','crosses_zero','positive')} for c in ('v22','v24')},
        'maximum_group_aggregate_difference_TWD':max_aggregation_difference,
        'maximum_paired_point_difference_TWD':max_point_difference,
        'fit_seconds_sum':fit_sum,'main_inner_seconds':execution['elapsed_seconds'],
        'pipeline_outer_steps':pipeline['steps'],'pipeline_seconds':pipeline['elapsed_seconds'],
        'query_or_fit_labels_loaded':False,'prepared_or_X_arrays_loaded':False,
        'saved_model_weights_loaded_CPU_only':True,'model_created_or_forward_called':False,
        'new_fit_predict_score_or_bootstrap':False,'saved_bootstrap_replicates_reaggregated':True,
        'test_features_or_prices_loaded':False,'Private_score_or_win_probability_estimated':False,
        'elapsed_seconds':time.perf_counter()-started}
    with (OUT/'independent_review.json').open('x',encoding='utf-8') as f:
        json.dump(audit,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:audit[k] for k in ('status','unique_verified_dependency_artifact_files',
        'predefined_gate','fit_seconds_sum','CI_sign_counts','elapsed_seconds')},ensure_ascii=False))


if __name__=='__main__':
    main()
