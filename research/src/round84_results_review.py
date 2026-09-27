"""R84 saved small-JSON review only; no arrays, models, scores or adoption actions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
from round81_results_review import STAGES, NAMES, BANDS, sha, near, delta, metric_deltas

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ('v24', 'v22')
OUT, MD = ROOT/'outputs/round84/results_review.json', ROOT/'reports/round84_report.md'
PINS = {
    'src/round81_results_review.py': '685572222f8b07d9a1f8d43e2ed339a970f7412dc41510868129dfdb40c9d875',
    'src/round84_experiment.py': '6b81e9005e3b018e628dccfe692536521576c68fdcce768cfc02f829055a5e71',
    'src/round84_capacity_unit.py': '93cf74c5410c6da782f925879e1e624fb57591c3329dac1a4fcafadec11ba68c',
    'src/round84_diagnostics.ps1': 'abe46df57f41b7a73552998642082cd9b80155591828b7010d1f082644e794b6',
    'src/candidate_diagnostics_v3.py': '49cf3a89e8a1d60defb8f49bc63ef0f8f88ded528af1e3687684d70c8a4ae128',
    'src/paired_candidate_bootstrap.py': '3761c1cb09abe4e26412bf3aea48273ce0a4bad10b5a408e23d8fae488903bb0',
    'reports/round84_plan.md': '3418a21b9b50dcbfa79b96e11d12ea7b7b9b1daec304272caa327d69ca788c7d',
    'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'outputs/round83/results.json': '8db07138c15ef2725d848c8b0ca65a6095bcc183b8550611377b3e165c59583b',
}
CHECKS = ('v24_base_reference_spatial_median_full_vectors_exact', 'controls_full_metrics_exact',
    'original_R79_legacy_feature_state_exact', 'date_weight_provenance_verified',
    'full_query_raw_TWD_reload_exact', 'inherited_2999_seed42_seed137_reload_evidence_verified',
    'spatial_fallback_exact', 'query_labels_used_only_for_evaluation')
WEIGHT_HASH_KEYS = ('fit_target_values_sha256', 'fit_transformed_target_values_sha256',
    'fit_effective_area_sha256', 'fit_factor_sha256', 'fit_area_weights_sha256',
    'fit_date_weights_sha256', 'fit_sample_weights_sha256', 'fit_original_feature_values_sha256',
    'fit_geo_feature_values_sha256', 'fit_model_feature_values_sha256')


def render(r):
    row = lambda xs: '| ' + ' | '.join(str(x) for x in xs) + ' |'
    num = lambda x, sign=False: '—' if x is None else (f'{x:+,.2f}' if sign else f'{x:,.2f}')
    lines = ['# R84：兩個固定種子的單價模型增加至4499樹', '',
        '主比較R84−v24/R83，次比較R84−v22/R79。每個相同seed唯一參數改動是iterations由2999增加到4499；depth8、learning rate .065、原95欄／23類別、單價目標、有效面積逆轉換、保存日期×面積權重不變。三期各訓練seed42、137，共六個模型；兩支分別還原非負TWD後，固定 .5×42＋.5×137，再套原trend reference、R39 spatial，與固定R71 NN及R46 log取median。',
        '不挑seed、輪數或中間checkpoint；沒有重訓NN／log／LightGBM。2999控制使用舊保存模型輸出，沒有同期重訓，因此觀察差異也含GPU重訓變異。以下差值為候選減對照，負值較好。本報告沒有採用決定。', '']
    def table(headers, rows):
        lines.extend([row(headers), row(['---']*len(headers)), *[row(x) for x in rows], ''])
    lines += ['## 三期完整候選', '']
    for metric in ('overall', 'weighted'):
        table(['期別', metric+' v24', 'v22', 'R84', '差vs v24', '差vs v22', '變動% vs v24／v22'],
            [[NAMES[s], *[num(r['stages'][s]['metrics'][k][metric+'_mae']) for k in (*CONTROLS,'R84')],
              *[num(r['stages'][s]['differences'][k][metric]['delta_TWD'],True) for k in CONTROLS],
              ' / '.join(f"{r['stages'][s]['differences'][k][metric]['relative_change_pct']:+.4f}%" for k in CONTROLS)] for s in STAGES])
    lines += ['## 固定test組成', '', '原city×seen/new、歷史cell至少200列；估計僅在有支持的test質量內重正規化，未支持列明列。這不是Private MAE。', '']
    table(['期別','涵蓋率','未支持列','v24','v22','R84','差vs v24','差vs v22'],
        [[NAMES[s], f"{r['stages'][s]['test_composition']['test_mass_covered']:.4%}", r['stages'][s]['test_composition']['unsupported_test_rows'],
          *[num(r['stages'][s]['test_composition']['models'][k]) for k in (*CONTROLS,'R84')],
          *[num(r['stages'][s]['test_composition_differences'][k]['delta_TWD'],True) for k in CONTROLS]] for s in STAGES])
    lines += ['## 相同seed的原生TWD比較', '', '2999控制與4499候選按相同seed比較；mean固定各半，不把raw unit直接平均或再次乘面積。單seed與各層僅是診斷，沒有挑選替代候選。', '']
    table(['期別','輸出','2999整體','4499整體','整體差','2999加權','4499加權','加權差'],
        [[NAMES[s], key, num(r['stages'][s]['native_metrics']['control'][key]['overall_mae']),
          num(r['stages'][s]['native_metrics']['candidate'][key]['overall_mae']), num(r['stages'][s]['native_differences'][key]['overall']['delta_TWD'],True),
          num(r['stages'][s]['native_metrics']['control'][key]['weighted_mae']), num(r['stages'][s]['native_metrics']['candidate'][key]['weighted_mae']),
          num(r['stages'][s]['native_differences'][key]['weighted']['delta_TWD'],True)] for s in STAGES for key in ('seed42','seed137','mean')])
    lines += ['## 逐層傳遞', '']
    rows = []
    for s in STAGES:
        a = r['stages'][s]
        pairs = [(k,a['unit_branch_metrics']['control'][k],a['unit_branch_metrics']['candidate'][k]) for k in ('base','reference','spatial')]
        pairs.append(('final median',a['metrics']['v24'],a['metrics']['R84']))
        rows += [[NAMES[s], layer, num(c['overall_mae']), num(n['overall_mae']), num(n['overall_mae']-c['overall_mae'],True),
                  num(c['weighted_mae']),num(n['weighted_mae']),num(n['weighted_mae']-c['weighted_mae'],True)] for layer,c,n in pairs]
    table(['期別','層','v24整體','R84整體','整體差','v24加權','R84加權','加權差'],rows)
    lines += ['單支／原生改善不保證median改善；逐層MAE差不是可相加的獨立收益或因果分解。', '', '## 全部新舊案、季度與城市', '']
    for family,title in (('strata','建案組'),('forecast_quarters','季度'),('cities','城市')):
        table(['期別',title,'列數','v24','v22','R84','差vs v24','差vs v22'],
            [[NAMES[s],key,c['rows'], *[num(r['stages'][s]['metrics'][k][family][key]['mae']) for k in (*CONTROLS,'R84')],
              *[num(r['stages'][s]['differences'][k][family][key]['delta_TWD'],True) for k in CONTROLS]]
             for s in STAGES for key,c in r['stages'][s]['metrics']['v24'][family].items()])
    lines += ['## 五價帶與有號尾部貢獻', '']
    table(['期別','真價帶','列數','v24','v22','R84','差vs v24','差vs v22'],
        [[NAMES[s],b['band'],b['rows'],*[num(b[k]) for k in (*CONTROLS,'R84')],*[num(b['differences'][k]['delta_TWD'],True) for k in CONTROLS]]
         for s in STAGES for b in r['stages'][s]['price_bands']])
    table(['期別','群','對照','列數','全期MAE差貢獻','其餘價帶貢獻'],
        [[NAMES[s],label,k,r['stages'][s]['tail_contributions'][key][k]['rows'],
          num(r['stages'][s]['tail_contributions'][key][k]['overall_delta_contribution_TWD'],True),
          num(r['stages'][s]['tail_contributions'][key][k]['other_bands_contribution_TWD'],True)]
         for s in STAGES for key,label in (('at_least_50m','≥5000萬'),('at_least_100m','≥1億')) for k in CONTROLS])
    lines += ['有號貢獻＝群MAE差×群列數／全期列數，並非prediction−truth的平均偏差；五帶貢獻合計整體MAE差。≥1億包含於≥5000萬，不可相加。真價分組只作分析，不作模型分流或逐ID規則。', '', '## 全部24個配對區間', '',
        '兩個對照各三期×建案／三月循環區塊×整體／加權，共24區間；各2000次、seed20260925。全部呈現，沒有挑選抽法。', '']
    for control in CONTROLS:
        lines += [f'### R84−{control}', '']
        rows = []
        for s in STAGES:
            for method in ('project','month'):
                for metric in ('overall','weighted'):
                    c = r['stages'][s]['paired'][control][method][metric]
                    where = '全負' if c['percentile_97_5']<0 else '全正' if c['percentile_2_5']>0 else '含／跨零'
                    rows.append([NAMES[s],method,metric,*[num(c[k],True) for k in ('point_delta','percentile_2_5','percentile_97_5')],f"{c['valid_replicates']}／{c['invalid_replicates']}",where])
        table(['期別','抽法','指標','點差','2.5%','97.5%','有效／無效','區間'],rows)
    lines += ['## 真實執行與保存證據', '',
        f"六個新fit合計 {r['total_fit_seconds']:.3f}秒；main {r['main_seconds']:.3f}秒（外層 {r['executions']['main']['elapsed_seconds']:.3f}秒），generic {r['diagnostics_seconds']:.3f}秒。0新NN／reference／spatial state fit。"]
    for k in CONTROLS:
        lines.append(f"paired vs {k}：內層 {r['paired_seconds'][k]:.3f}秒，外層 {r['executions']['paired_'+k]['elapsed_seconds']:.3f}秒；exit0。")
    table(['期別','seed','fit／query列','fit秒','fit＋保存秒','全query重載列','原生零價列','metadata'],
        [[NAMES[s],seed,f"{r['stages'][s]['fit_rows']}／{r['stages'][s]['query_rows']}",f"{m['fit_seconds']:.3f}",f"{m['fit_and_save_seconds']:.3f}",m['reload_rows'],m['native_zero_TWD_rows'],f"[metadata](../{m['metadata_path']})"]
         for s in STAGES for seed,m in r['stages'][s]['model_evidence'].items()])
    for s in STAGES:
        a=r['stages'][s]
        lines.append(f"{NAMES[s]}：完整期 {a['stage_seconds']:.3f}秒，最終median改變 {a['changed_final_rows']}／{a['query_rows']}列。")
    lines += ['', '新模型各自先保存，再重載核全query raw unit及TWD逐位相等；這是主程式的保存證據，報告器没有另載模型或向量。原R19內層state的2999／seed42／area-only欄位只作特徵來源，外層明列本次seed及4499樹。實際Pool權重仍為A/100×float64(保存R63日期float32權重)，沒有重算或重新正規化；本輪沒有測試權重共同縮放的影響。',
        f"主流程記錄v24控制全向量與所有control metrics重現、全部query重載及fallback檢查通過，{r['main_declared_protected_file_count']}份CSV／registry與selector前後不變。reference／spatial預測各6次，沒有新增state。所有fit、主流程、generic及兩個paired execution皆exit0。", '', '## 解讀限制', '',
        '模型增加1500棵樹可能增加擬合能力，也可能過擬合。相同seed只改iterations，但GPU重訓不保證位元決定性，2999對照未同期重訓；兩seed不構成完整隨機性估計，不能把容量改動視為已隔離的因果效果。',
        '本報告呈現三期與所有分組的得失，不能只以原生單支、極高价尾或某一年度改善宣稱普遍收益。所有歷史窗口已反覆用於選型，並非新盲測；bootstrap未校正多輪選擇偏差，也不是Private勝率。Public未用來調整此配置，Private未知。',
        '本報告不自動採用、不新增全期必勝門檻，不挑seed／輪數／權重，不輸出新CSV或改推薦。後续是否採用由獨立決策評估實際收益、群組代價與不確定性。程式與模型保留，GitHub仍待最終版本整理，沒有上傳。', '',
        '[固定計畫](round84_plan.md) · [主結果](../outputs/round84/results.json) · [主執行](../outputs/round84/execution.json) · [分組診斷](round84_diagnostics.md)',
        '[主配對vs v24](../outputs/round84/paired_vs_v24/results.json) · [次配對vs v22](../outputs/round84/paired_vs_v22/results.json)',
        '[報告JSON](../outputs/round84/results_review.json) · [報告器](../src/round84_results_review.py)', '']
    return '\n'.join(lines)


def review():
    tick,started=time.monotonic(),datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT,OUT.with_suffix('.sha256'),MD)), 'No overwrite'
    inputs,receipts={},{}
    def capture(name,expected=None):
        p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json','.sha256','.py','.ps1','.md')
        assert p.stat().st_size<10_000_000, 'Small JSON/source/text only'
        digest,key=sha(p),p.relative_to(ROOT).as_posix()
        assert expected is None or digest==expected,str(name)
        assert key not in inputs or inputs[key]==digest
        inputs[key]=digest
        return p
    def small(name,expected=None):
        p=capture(name,expected or PINS.get(name))
        found=[q for q in (p.with_suffix('.sha256'),Path(str(p)+'.sha256')) if q.exists()]
        assert found,'Missing receipt: '+name
        receipts[name]=[q.relative_to(ROOT).as_posix() for q in found]
        for q in found:
            capture(q)
            assert q.read_text(encoding='utf8').strip().split()[0]==sha(p)
        return json.loads(p.read_text(encoding='utf8'))
    for name,digest in PINS.items(): capture(name,digest)
    r=small('outputs/round84/results.json')
    plan=small('outputs/round84/plan.json',r['plan_sha256'])
    priors={k:small(f'outputs/round{n}/results.json') for k,n in (('v24',83),('v22',79))}
    diag=small('reports/round84_diagnostics.json')
    paired={k:small(f'outputs/round84/paired_vs_{k}/results.json') for k in CONTROLS}
    execution_paths={'main':'outputs/round84/execution.json','diagnostics':'outputs/round84/diagnostics/execution.json',
        'diagnostics_outer':'outputs/round84/diagnostics_wrapper_execution.json',
        **{'paired_'+k:f'outputs/round84/paired_vs_{k}_execution.json' for k in CONTROLS}}
    executions={k:small(p) for k,p in execution_paths.items()}
    for k,e in executions.items():
        assert e['exit_code']==0 and e.get('failure') is None and e['automatic_retry'] is False
        source='src/round84_experiment.py' if k=='main' else 'src/candidate_diagnostics_v3.py' if k.startswith('diagnostics') else 'src/paired_candidate_bootstrap.py'
        assert e['source_sha256']==PINS[source]
        if k=='diagnostics_outer' or k.startswith('paired_'):
            assert e['wrapper_sha256']==PINS['src/round84_diagnostics.ps1']
    assert r['round']==plan['round']==84 and r['candidate']==plan['candidate']=='R84'
    assert r['status']=='three_periods_completed_no_adoption_decision'
    assert r['primary_control']==plan['primary_control']=='v24' and r['secondary_control']==plan['secondary_control']=='v22'
    assert r['checked_stages']==diag['checked_stages']==plan['stage_order']==list(STAGES)
    assert all(p['checked_stages']==list(STAGES) for p in priors.values())
    assert r['formula']==plan['formula'] and r['text_plan_sha256']==plan['text_plan_sha256']==PINS['reports/round84_plan.md']
    assert r['source_sha256']==plan['source_sha256'] and r['input_sha256']==plan['input_sha256']
    for name in ('src/round84_experiment.py','src/round84_capacity_unit.py'): assert r['source_sha256'][name]==PINS[name]
    for name in ('outputs/round79/results.json','outputs/round83/results.json'): assert r['input_sha256'][name]==PINS[name]
    assert r['seeds']==plan['seeds']==r['trained_seeds']==[42,137] and r['reused_seeds']==[]
    assert r['baseline_iterations']==plan['baseline_iterations']==2999 and r['candidate_iterations']==plan['candidate_iterations']==4499
    assert r['changed_parameter_names']==plan['changed_parameter_names']==['iterations']
    for seed in ('42','137'):
        assert plan['baseline_parameters'][seed]['random_seed']==int(seed) and plan['baseline_parameters'][seed]['iterations']==2999
        assert plan['parameters'][seed]=={**plan['baseline_parameters'][seed],'iterations':4499}
    assert r['new_fits']==plan['planned_new_fits']==6 and r['new_NN_fits']==r['new_reference_fits']==r['new_spatial_state_builds']==0
    assert r['reference_predict_calls']==r['spatial_predict_calls']==6
    assert plan['all_periods_regardless_of_scores'] and plan['saved_date_weights_recomputed'] is False
    assert all(r[k] is False for k in ('new_CSV','new_test_predictions','new_submission_generated','automatic_adoption','fresh_blind_validation','Public_feedback_used_for_tuning','individual_seed_selection'))
    assert r['query_labels_used_only_for_evaluation'] is True
    assert r['protected_before']==r['protected_after']==plan['protected_before']
    assert r['selector_sha256_before']==r['selector_sha256_after']==plan['selector_sha256_before']
    assert diag['round']==84 and diag['candidate']=='R84' and diag['status']=='completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls']==0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert diag['input_sha256']['outputs/round84/results.json']==inputs['outputs/round84/results.json']
    for k,p in paired.items():
        assert p['status']=='completed' and p['candidate']=='R84' and p['control']==k
        assert p['difference']=='candidate minus control' and p['replicates']==2000 and p['seed']==20260925
        assert set(p['stages'])==set(STAGES) and p['fits']==p['model_predict_calls']==0 and not p['new_CSV']
    out=dict(round=84,status='completed_JSON_only_review',primary_comparison='R84 minus v24=R83',secondary_comparison='R84 minus v22=R79',
        adoption_decision=None,is_decision_authority=False,new_fits_in_reporter=0,model_or_array_loads=0,new_scores=0,new_bootstrap=False,
        independent_vector_replay_performed=False,new_CSV=False,selector_modified=False,source_sha256=sha(Path(__file__)),input_sha256=inputs,
        observed_receipts=receipts,started_at_utc=started,executions=executions,checked_stages=list(STAGES),stages={},
        main_seconds=r['elapsed_seconds'],diagnostics_seconds=diag['elapsed_seconds'],paired_seconds={k:p['elapsed_seconds'] for k,p in paired.items()},
        main_declared_counts={k:r[k] for k in ('new_fits','new_NN_fits','new_reference_fits','new_spatial_state_builds','reference_predict_calls','spatial_predict_calls')},
        main_declared_protected_file_count=len(r['protected_before']),main_declared_protected_files_equal=True,main_declared_selector_unchanged=True,
        main_evidence_only_no_independent_array_check=True,paired_interval_count=24)
    for s in STAGES:
        a,dg=r['stages'][s],diag['stages'][s]
        before83,before79=priors['v24']['stages'][s],priors['v22']['stages'][s]
        assert a['status']=='completed' and a['fit_rows']==dg['fit_rows']==before83['fit_rows']==before79['fit_rows']
        assert a['query_rows']==dg['query_rows']==before83['query_rows']==before79['query_rows']
        assert a['new_fits']==2 and a['new_NN_fits']==0 and all(a[k] is True for k in CHECKS)
        assert a['baseline_2999_models_reloaded_or_retrained'] is False and a['individual_seed_selection'] is False
        assert dg['all_full_saved_metrics_exact'] and all(a['metrics'][k]==v for k,v in dg['metrics'].items())
        for label,before,key in (('v24',before83,'R83'),('v22',before79,'R79')):
            assert a['metrics'][label]==before['metrics'][key]
            assert dg['extra_control_existing_test_mix_checks'][label]=='exact'
        assert a['original_frame_sha256']==before83['original_frame_sha256']==before79['original_frame_sha256']
        layers,native=a['unit_branch_metrics'],a['native_metrics']
        assert layers['control']==before83['unit_branch_metrics']['candidate']
        assert native['control']==before83['native_metrics']
        assert native['control']['seed42']==before79['unit_branch_metrics']['native']
        assert native['control']['mean']==layers['control']['base'] and native['candidate']['mean']==layers['candidate']['base']
        differences={k:metric_deltas(a['metrics'][k],a['metrics']['R84']) for k in CONTROLS}
        layer_diff={k:metric_deltas(layers['control'][k],layers['candidate'][k]) for k in ('base','reference','spatial')}
        native_diff={k:metric_deltas(native['control'][k],native['candidate'][k]) for k in ('seed42','seed137','mean')}
        for saved,calculated in ((a['differences'],differences),(a['unit_branch_differences'],layer_diff),(a['native_differences'],native_diff)):
            for key in calculated:
                for metric in ('overall','weighted'): assert near(saved[key][metric]['delta_TWD'],calculated[key][metric]['delta_TWD'])
        evidence={}
        for seed in ('42','137'):
            model_path=f'outputs/round84/{s}/model_seed{seed}'
            names={k:model_path+'/'+k+'.json' for k in ('metadata','fit_metadata','feature_state','fit_execution')}
            model_records={k:small(name,a['artifact_sha256'][name]) for k,name in names.items()}
            m,fm,state,fe=(model_records[k] for k in ('metadata','fit_metadata','feature_state','fit_execution'))
            binding=plan['bindings'][s]
            before=binding['baseline_models'][seed]
            legacy=binding['R79_state']
            assert fe['exit_code']==0 and fe['failure'] is None and fe['automatic_retry'] is False
            assert near(fe['elapsed_seconds'],m['fit_and_save_seconds']) and near(a['fit_seconds_by_seed'][seed],m['fit_seconds'])
            assert m['round']==84 and m['stage']==s and m['fit_rows']==a['fit_rows'] and m['query_rows']==a['query_rows']
            assert m['seed']==m['actual_random_seed']==state['seed']==int(seed) and m['prediction_units']=='TWD'
            assert m['feature_count']==95 and m['categorical_feature_count']==23 and m['iterations']==m['actual_iterations']==4499
            assert m['parameters']==fm['parameters']==state['parameters']==plan['parameters'][seed]=={**before['parameters'],'iterations':4499}
            assert m['baseline_parameters']==state['baseline_parameters']==before['parameters']==plan['baseline_parameters'][seed]
            assert m['changed_parameter_names']==state['changed_parameter_names']==['iterations']
            assert state['legacy_feature_state']==legacy
            assert m['legacy_state_parameters']==legacy['parameters']==binding['R79_metadata']['parameters']
            assert state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
            assert m['legacy_feature_state_content_sha256']==legacy['state_content_sha256']==binding['R79_metadata']['feature_state_content_sha256']
            assert m['feature_state_content_sha256']==state['state_content_sha256']
            assert m['matched_2999_metadata_path']==binding['baseline_metadata_paths'][seed]
            for key in WEIGHT_HASH_KEYS: assert m[key]==fm[key]==before[key],key
            assert m['fit_date_weights_sha256']==m['R63_preparation']['train_sample_weight_sha256']
            assert m['original_frame_sha256']==a['original_frame_sha256'] and m['plan_sha256']==r['plan_sha256']
            for kind in ('fit','query'): assert m[kind+'_indices_content_sha256']==plan['split_contract'][s][kind+'_indices_content_sha256']==before[kind+'_indices_content_sha256']
            assert m['source_sha256']['src/round84_capacity_unit.py']==state['source_sha256']['src/round84_capacity_unit.py']==PINS['src/round84_capacity_unit.py']
            assert all(m[k] is True for k in ('reload_all_query_rows_verified','reload_raw_exact','reload_TWD_exact','original_R79_legacy_feature_state_exact','date_weight_provenance_verified_in_driver','legacy_state_training_fields_are_provenance_only','weight_argument_passed_to_pool'))
            assert all(m[k] is False for k in ('query_labels_used','weights_use_fit_target','weights_use_query','weights_normalized_to_mean_one','date_weights_recomputed','date_weights_renormalized','fit_uses_eval_set','early_stopping_used'))
            assert m['reload_rows']==a['query_rows'] and m['weight_dtype']=='float64'
            assert all(before[k] is True for k in ('reload_all_query_rows_verified','reload_raw_exact','reload_TWD_exact'))
            fitted,loaded=m['fitted_native_capacity'],m['loaded_native_capacity']
            assert all(fitted[k]==loaded[k] for k in fitted if k!='actual_learning_rate')
            for c in (fitted,loaded):
                assert c['actual_random_seed']==int(seed) and c['actual_trees']==c['actual_iterations']==4499 and c['actual_model_depth']==8
                assert c['actual_boosting_type']=='Plain' and c['actual_grow_policy']=='SymmetricTree'
                assert c['actual_feature_count']==95 and c['actual_categorical_feature_count']==23 and 0<c['maximum_tree_leaf_count']<=256
                assert c['actual_loss_function']==c['actual_eval_metric']=='MAE' and abs(c['actual_learning_rate']-.065)<=.065*1e-7
            native_config=m['actual_native_configuration']
            assert native_config==fm['actual_native_configuration'] and native_config['iterations']==4499 and native_config['random_seed']==int(seed)
            wsummary=m['fit_sample_weights_summary']
            assert wsummary['rows']==a['fit_rows'] and wsummary['min']>0
            assert all(near(wsummary[k],before['fit_sample_weights_summary'][k]) for k in wsummary)
            evidence[seed]=dict(metadata_path=names['metadata'],metadata_sha256=inputs[names['metadata']],fit_metadata_sha256=inputs[names['fit_metadata']],
                wrapper_state_sha256=inputs[names['feature_state']],fit_execution=fe,fit_seconds=m['fit_seconds'],fit_and_save_seconds=m['fit_and_save_seconds'],
                actual_parameters=m['parameters'],baseline_parameters=m['baseline_parameters'],changed_parameter_names=m['changed_parameter_names'],
                identity_hashes={k:m[k] for k in WEIGHT_HASH_KEYS},fit_weight_summary=wsummary,
                fitted_native_capacity=fitted,loaded_native_capacity=loaded,actual_native_configuration=native_config,
                reload_rows=m['reload_rows'],native_negative_raw_rows=m['native_negative_raw_rows'],native_zero_TWD_rows=m['native_zero_TWD_rows'],
                wrapper_preserves_R79_legacy_state=True,legacy_training_fields_are_provenance_only=True,query_labels_used=False)
        assert near(sum(v['fit_seconds'] for v in evidence.values()),a['fit_seconds'])
        ci,support={},{}
        for k in CONTROLS:
            ps=paired[k]['stages'][s]
            assert ps['rows']==a['query_rows'] and ps['point_metrics_reproduced']
            assert ps['prediction_sha256']['R84']==a['prediction_sha256'] and ps['prediction_sha256'][k]==priors[k]['stages'][s]['prediction_sha256']
            ci[k]={method:ps[method]['summary'] for method in ('project','month')}
            support[k]={method:ps[method]['support'] for method in ('project','month')}
            for method in ci[k]:
                for metric in ('overall','weighted'):
                    c=ci[k][method][metric]
                    assert c['valid_replicates']+c['invalid_replicates']==2000
                    assert near(c['candidate_mae'],a['metrics']['R84'][metric+'_mae']) and near(c['control_mae'],a['metrics'][k][metric+'_mae'])
                    assert near(c['point_delta'],differences[k][metric]['delta_TWD'])
        mix,bands=dg['test_composition'],[]
        assert mix['minimum_historical_cell_rows']==200 and [b['band'] for b in dg['price_bands']]==list(BANDS)
        assert sum(b['rows'] for b in dg['price_bands'])==a['query_rows']
        for b in dg['price_bands']:
            d={k:delta(b[k],b['R84']) for k in CONTROLS}
            contribution={k:0.0 if b['rows']==0 else d[k]['delta_TWD']*b['rows']/a['query_rows'] for k in CONTROLS}
            bands.append(dict(b,differences=d,contribution_to_stage_overall_delta_TWD=contribution))
        tails={}
        for threshold,selected in (('at_least_50m',bands[-2:]),('at_least_100m',bands[-1:])):
            tails[threshold]={}
            for k in CONTROLS:
                total=differences[k]['overall']['delta_TWD']
                assert near(sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands),total)
                tail=sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in selected)
                tails[threshold][k]=dict(rows=sum(b['rows'] for b in selected),overall_delta_contribution_TWD=tail,other_bands_contribution_TWD=total-tail,signed_ratio_to_net_delta=None if total==0 else tail/total)
        out['stages'][s]=dict(fit_rows=a['fit_rows'],query_rows=a['query_rows'],metrics=a['metrics'],differences=differences,
            native_metrics=native,native_differences=native_diff,unit_branch_metrics=layers,unit_branch_differences=layer_diff,
            test_composition=mix,test_composition_differences={k:delta(mix['models'][k],mix['models']['R84']) for k in CONTROLS},
            price_bands=bands,tail_contributions=tails,paired=ci,paired_support=support,main_evidence_checks={k:a[k] for k in CHECKS},
            changed_unit_rows=a['changed_unit_rows'],changed_final_rows=a['changed_final_rows'],fit_seconds=a['fit_seconds'],
            stage_seconds=a['elapsed_seconds'],model_evidence=evidence)
    out['total_fit_seconds']=sum(out['stages'][s]['fit_seconds'] for s in STAGES)
    md=render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick,report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest())
    assert all(sha(ROOT/n)==h for n,h in inputs.items()),'Small evidence changed'
    raw=(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x',encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest()+'\n')
    with MD.open('x',encoding='utf8',newline='\n') as f: f.write(md)
    print(json.dumps(dict(status=out['status'],elapsed_seconds=out['elapsed_seconds'],review_sha256=sha(OUT),report_sha256=sha(MD)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    review()
