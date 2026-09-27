"""R83 small-JSON report only; no arrays/models, rescoring or selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
from round81_results_review import STAGES, NAMES, BANDS, sha, near, delta, metric_deltas

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ('v23', 'v22')
OUT = ROOT / 'outputs/round83/results_review.json'
MD = ROOT / 'reports/round83_report.md'
PINS = {
    'src/round81_results_review.py': '685572222f8b07d9a1f8d43e2ed339a970f7412dc41510868129dfdb40c9d875',
    'src/round83_diagnostics.ps1': '4926df1ec75c7e653217e5f5e7b3b0c1dfeac4c5e38fe7470bd3b83e16b2511c',
    'src/candidate_diagnostics_v3.py': '49cf3a89e8a1d60defb8f49bc63ef0f8f88ded528af1e3687684d70c8a4ae128',
    'src/paired_candidate_bootstrap.py': '3761c1cb09abe4e26412bf3aea48273ce0a4bad10b5a408e23d8fae488903bb0',
    'reports/round83_plan.md': 'c0cf070337ca86ab742479b664628413155f1e69745c7a80ae1f15e152c747c0',
    'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'outputs/round80/results.json': 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6',
}


def render(r):
    row = lambda xs: '| ' + ' | '.join(str(x) for x in xs) + ' |'
    num = lambda x, sign=False: '—' if x is None else (f'{x:+,.2f}' if sign else f'{x:,.2f}')
    lines = ['# R83：兩個種子的日期加權單價 CatBoost', '',
        '主比較R83−v23；v22為次對照。重用R79 seed42，每期只新增seed137，原95欄／23類別、2999樹、depth8、單價目標及面積×保存日期權重不變。各支先以原逆轉換得到非負TWD，再依固定順序各半平均。均值作為純native unit base，經同一trend reference及R39 spatial各一次，最後與固定R71 seed42 NN、R46 log取median。',
        '只訓練三個seed137單價模型，沒有重訓seed42、NN、log或LightGBM；沒有挑選較好的單一seed、加入第三seed或搜尋混合比例。seed137單支只作診斷，不是另外可採用的候選。',
        '以下MAE差均為候選減對照，負值較好；本報告僅彙整保存證據，沒有採用決定、新CSV或推薦更新。', '', '## 三期完整候選', '']
    for metric, title in (('overall', '整體MAE'), ('weighted', '固定加權MAE')):
        lines += [row(['期別', *[k+' '+title for k in (*CONTROLS, 'R83')], *['差vs'+k for k in CONTROLS]]), row(['---']+['---:']*5)]
        for s in STAGES:
            a = r['stages'][s]
            lines.append(row([NAMES[s], *[num(a['metrics'][k][metric+'_mae']) for k in (*CONTROLS, 'R83')], *[num(a['differences'][k][metric]['delta_TWD'], True) for k in CONTROLS]]))
        lines.append('')
    lines += [row(['期別', *[k+' 整體／加權變動' for k in CONTROLS]]), row(['---']+['---:']*2)]
    for s in STAGES:
        a = r['stages'][s]
        lines.append(row([NAMES[s], *[' / '.join(f"{a['differences'][k][m]['relative_change_pct']:+.4f}%" for m in ('overall','weighted')) for k in CONTROLS]]))
    lines += ['', '## 固定test組成', '',
        '原city×seen/new、歷史cell至少200列；只在有支持的test質量內重正規化，並明示未支持列。這是組成診斷，不是Private MAE。', '',
        row(['期別','涵蓋率','未支持列',*CONTROLS,'R83',*['差vs'+k for k in CONTROLS]]), row(['---']+['---:']*7)]
    for s in STAGES:
        a = r['stages'][s]; m = a['test_composition']
        lines.append(row([NAMES[s],f"{m['test_mass_covered']:.4%}",m['unsupported_test_rows'], *[num(m['models'][k]) for k in (*CONTROLS,'R83')], *[num(a['test_composition_differences'][k]['delta_TWD'],True) for k in CONTROLS]]))
    lines += ['', '## 各seed原生總價與固定平均', '',
        '均為逐支依原有效面積逆轉換、非負處理後的TWD；不把raw unit值直接平均或再乘一次面積。', '',
        row(['期別','原生輸出','整體MAE','加權MAE','整體差vs42','加權差vs42']), row(['---','---']+['---:']*4)]
    for s in STAGES:
        a = r['stages'][s]; base = a['native_metrics']['seed42']
        for k in ('seed42','seed137','mean'):
            m = a['native_metrics'][k]
            lines.append(row([NAMES[s],k,num(m['overall_mae']),num(m['weighted_mae']),num(m['overall_mae']-base['overall_mae'],True),num(m['weighted_mae']-base['weighted_mae'],True)]))
    lines += ['', '## base、reference、spatial與最終median', '',
        row(['期別','層','v23 MAE','R83 MAE','整體差','加權差']), row(['---','---']+['---:']*4)]
    for s in STAGES:
        a = r['stages'][s]
        pairs = [(k,a['unit_branch_metrics']['control'][k],a['unit_branch_metrics']['candidate'][k]) for k in ('base','reference','spatial')]
        pairs.append(('final median',a['metrics']['v23'],a['metrics']['R83']))
        for layer,c,n in pairs:
            lines.append(row([NAMES[s],layer,num(c['overall_mae']),num(n['overall_mae']),num(n['overall_mae']-c['overall_mae'],True),num(n['weighted_mae']-c['weighted_mae'],True)]))
    lines.append('')
    for s in STAGES:
        a = r['stages'][s]
        lines.append(f"{NAMES[s]}：最終median改變 {a['changed_final_rows']}／{a['query_rows']}列；新fit {a['fit_seconds']:.3f}秒，完整期 {a['stage_seconds']:.3f}秒。")
    lines += ['', '原生平均或單支改善不保證最終median改善；逐層MAE差不是可相加的獨立收益，也不是因果分解。', '', '## 新舊建案與季度', '',
        row(['期別','群','列數','v23 MAE','R83 MAE','差vs v23','差vs v22']), row(['---','---']+['---:']*5)]
    for s in STAGES:
        a = r['stages'][s]
        for family in ('strata','forecast_quarters'):
            for key,c in a['metrics']['v23'][family].items():
                n = a['metrics']['R83'][family][key]
                lines.append(row([NAMES[s],key,c['rows'],num(c['mae']),num(n['mae']), *[num(a['differences'][k][family][key]['delta_TWD'],True) for k in CONTROLS]]))
    lines += ['', '## 全部城市', '', row(['期別','城市','列數','v23 MAE','R83 MAE','差vs v23','差vs v22']), row(['---','---']+['---:']*5)]
    for s in STAGES:
        a = r['stages'][s]
        for key,c in a['metrics']['v23']['cities'].items():
            n = a['metrics']['R83']['cities'][key]
            lines.append(row([NAMES[s],key,c['rows'],num(c['mae']),num(n['mae']), *[num(a['differences'][k]['cities'][key]['delta_TWD'],True) for k in CONTROLS]]))
    lines += ['', '## 固定五價帶及尾部貢獻', '',
        row(['期別','真價帶','列數',*CONTROLS,'R83',*['差vs'+k for k in CONTROLS]]), row(['---','---']+['---:']*6)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s],b['band'],b['rows'], *[num(b[k]) for k in (*CONTROLS,'R83')], *[num(b['differences'][k]['delta_TWD'],True) for k in CONTROLS]]))
    lines += ['', row(['期別','尾部','列數',*['全期MAE差貢獻vs'+k for k in CONTROLS],'其他價帶vs v23']), row(['---','---']+['---:']*4)]
    for s in STAGES:
        for key,label in (('at_least_50m','≥5000萬'),('at_least_100m','≥1億')):
            d = r['stages'][s]['tail_contributions'][key]
            lines.append(row([NAMES[s],label,d['v23']['rows'], *[num(d[k]['overall_delta_contribution_TWD'],True) for k in CONTROLS],num(d['v23']['other_bands_contribution_TWD'],True)]))
    lines += ['', '有號貢獻＝群MAE差×群列數／全期列數，負為改善、正為拖累；五帶加總為整體MAE差。不是prediction−truth的平均偏差。≥1億已包含在≥5000萬，兩者不可相加；小樣本巨額交易可能影響結論，真價分組僅作分析、不作推論分流或逐ID改價。', '',
        '## R83−v23全部12個配對區間', '',
        '建案／三月循環區塊各2000次、seed20260925。所有區間均呈現；抽樣改善比例不是Private勝率。', '',
        row(['期別','抽法','指標','點差','2.5%','97.5%','有效／無效','位置']),row(['---','---','---']+['---:']*4+['---'])]
    for s in STAGES:
        for method in ('project','month'):
            for metric in ('overall','weighted'):
                c = r['stages'][s]['paired'][method][metric]
                where = '全負' if c['percentile_97_5']<0 else '全正' if c['percentile_2_5']>0 else '含／跨零'
                lines.append(row([NAMES[s],method,metric, *[num(c[k],True) for k in ('point_delta','percentile_2_5','percentile_97_5')],f"{c['valid_replicates']}／{c['invalid_replicates']}",where]))
    lines += ['', '## 執行及模型證據', '',
        f"本輪新CatBoost fit共3次，合計 {r['total_fit_seconds']:.3f}秒；0新NN fit。main {r['main_seconds']:.3f}秒（外層 {r['executions']['main']['elapsed_seconds']:.3f}秒），generic {r['diagnostics_seconds']:.3f}秒（外層 {r['executions']['diagnostics']['elapsed_seconds']:.3f}秒），paired {r['paired_seconds']:.3f}秒（外層 {r['executions']['paired']['elapsed_seconds']:.3f}秒）。三份execution皆exit0；既有seed42未重訓或重新載入模型。",
        '新seed137模型逐期先保存，再以獨立重載核全部query raw unit與TWD逐位一致。原R19 feature state保留在wrapper內；其seed42／area-only欄位僅是舊provenance，外層實際training parameters明列seed137，唯一參數差random_seed。實際權重仍為A/100×float64(保存R63日期float32權重)，不重新計算、重新正規化或使用query日期／價格。',
        '', row(['期別','fit／query列','重載列','權重集中比例','來源']), row(['---','---:','---:','---:','---'])]
    for s in STAGES:
        a = r['stages'][s]; e = a['model_evidence']
        lines.append(row([NAMES[s],f"{a['fit_rows']}／{a['query_rows']}",e['reload_rows'],f"{e['weight_concentration_fraction']:.2%}",f"[metadata](../outputs/round83/{s}/model/metadata.json)"]))
    lines += ['', '權重集中比例是(sum w)²／sum(w²)／fit列數，描述權重集中程度，不是獨立樣本量或有效驗證量。乘積權重的尺度會影響訓練，這次保持原R79規則，未事後追加歸一化變體。',
        f"主程式記錄v23 base/reference/spatial/median控制逐位重現、完整背景metrics一致、三期canonical indices／frame與日期權重來源一致，{r['main_declared_protected_file_count']}份CSV／registry及selector前後不變。每期對照與候選reference及spatial各2次，共各6次，沒有新reference fit或spatial state build。",
        '本報告只讀小JSON、source與收據，核保存摘要的對應及差額算式；沒有載array／CBM、再次模型預測、逐筆重算MAE或重抽樣。全query重載與控制逐位回放是主程式保存的證據，不能當作報告器獨立重演。', '', '## 解讀限制', '',
        'R24曾做不同raw／均勻權重的CatBoost種子組合；本輪是原單價／日期權重的固定42與137。兩個seed不足以估計完整訓練隨機性分布，GPU也不保證全部運算位元決定性。均值可能減少部分波動，但不能消除共同特徵、目標尺度或超大物件外推偏差。',
        '所有歷史窗口已多次使用，並非新盲測；配對區間未校正多輪模型選擇偏差。主比較只固定此種子平均改動，不能把層間變化當作獨立因果證據，也不能保證Private成績。',
        '本報告不預設全期必勝的新門檻，不挑seed或群組，不根據Public調整；實際採用由另份決策審視收益與退步。沒有final／新CSV／上傳／selector變更，Private未知。原程式、模型、執行紀錄保留；GitHub待最終版本確定後整理，尚未發布。', '',
        '[固定計畫](round83_plan.md) · [主程式](../src/round83_experiment.py) · [主結果](../outputs/round83/results.json) · [主執行](../outputs/round83/execution.json)',
        '[完整分組](round83_diagnostics.md) · [配對結果](../outputs/round83/paired_vs_v23/results.json) · [配對執行](../outputs/round83/paired_vs_v23_execution.json)',
        '[本報告JSON](../outputs/round83/results_review.json) · [可重現報告器](../src/round83_results_review.py)', '']
    return '\n'.join(lines)

PINS.update({
    'src/round83_experiment.py': '8672b65b4e7258348a9124d0b20cdfeceadd404aa2a1a6f9bb0dee0a3ad44f5c',
    'src/round83_seed_unit.py': 'c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b',
})
CHECKS = ('v23_base_reference_spatial_median_full_vectors_exact', 'controls_full_metrics_exact',
    'original_R79_legacy_feature_state_exact', 'date_weight_provenance_verified',
    'full_query_raw_TWD_reload_exact', 'inherited_seed42_reload_evidence_verified',
    'spatial_fallback_exact', 'query_labels_used_only_for_evaluation')
WEIGHT_HASH_KEYS = ('fit_target_values_sha256', 'fit_transformed_target_values_sha256',
    'fit_effective_area_sha256', 'fit_factor_sha256', 'fit_area_weights_sha256',
    'fit_date_weights_sha256', 'fit_sample_weights_sha256', 'fit_original_feature_values_sha256',
    'fit_geo_feature_values_sha256', 'fit_model_feature_values_sha256')


def review():
    tick, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs, receipts = {}, {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.ps1', '.md')
        assert p.stat().st_size < 10_000_000, 'Small saved JSON/source/text only'
        digest, key = sha(p), p.relative_to(ROOT).as_posix()
        assert expected is None or digest == expected, str(name)
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return p

    def small(name, expected=None):
        p = capture(name, expected or PINS.get(name))
        found = [q for q in (p.with_suffix('.sha256'), Path(str(p) + '.sha256')) if q.exists()]
        assert found, 'Actual receipt missing: ' + name
        receipts[name] = [q.relative_to(ROOT).as_posix() for q in found]
        for q in found:
            capture(q)
            assert q.read_text(encoding='utf8').strip().split()[0] == sha(p), str(q)
        return json.loads(p.read_text(encoding='utf8'))

    for name, digest in PINS.items():
        capture(name, digest)
    r = small('outputs/round83/results.json')
    plan = small('outputs/round83/plan.json', r['plan_sha256'])
    priors = {k: small(f'outputs/round{n}/results.json') for k,n in (('v23',80),('v22',79))}
    diag = small('reports/round83_diagnostics.json')
    paired = small('outputs/round83/paired_vs_v23/results.json')
    executions = {k: small(p) for k,p in (
        ('main','outputs/round83/execution.json'), ('diagnostics','outputs/round83/diagnostics/execution.json'),
        ('paired','outputs/round83/paired_vs_v23_execution.json'))}
    assert all(e['exit_code'] == 0 and e.get('failure') is None and e['automatic_retry'] is False for e in executions.values())
    for key, source in (('main','src/round83_experiment.py'),('diagnostics','src/candidate_diagnostics_v3.py'),('paired','src/paired_candidate_bootstrap.py')):
        assert executions[key]['source_sha256'] == PINS[source]
    assert executions['paired']['wrapper_sha256'] == PINS['src/round83_diagnostics.ps1']
    assert r['round'] == plan['round'] == 83 and r['candidate'] == plan['candidate'] == 'R83'
    assert r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['primary_control'] == plan['primary_control'] == 'v23' and r['secondary_control'] == plan['secondary_control'] == 'v22'
    assert r['checked_stages'] == diag['checked_stages'] == plan['stage_order'] == list(STAGES)
    assert all(p['checked_stages'] == list(STAGES) for p in priors.values())
    assert r['formula'] == plan['formula'] and r['text_plan_sha256'] == plan['text_plan_sha256'] == PINS['reports/round83_plan.md']
    assert r['source_sha256'] == plan['source_sha256'] and r['input_sha256'] == plan['input_sha256']
    for name in ('src/round83_experiment.py','src/round83_seed_unit.py'):
        assert r['source_sha256'][name] == PINS[name]
    for name in ('outputs/round79/results.json','outputs/round80/results.json'):
        assert r['input_sha256'][name] == PINS[name]
    assert r['seeds'] == plan['seeds'] == [42,137] and r['reused_seeds'] == [42] and r['trained_seeds'] == [137]
    assert plan['parameters'] == {**plan['baseline_parameters'], 'random_seed':137}
    assert plan['baseline_parameters']['random_seed'] == 42 and plan['changed_parameter_names'] == ['random_seed']
    assert r['new_fits'] == plan['planned_new_fits'] == 3
    assert r['new_NN_fits'] == r['new_reference_fits'] == r['new_spatial_state_builds'] == 0
    assert r['reference_predict_calls'] == r['spatial_predict_calls'] == 6
    assert plan['all_periods_regardless_of_scores'] is True and plan['saved_date_weights_recomputed'] is False
    assert all(r[k] is False for k in ('new_CSV','new_test_predictions','new_submission_generated','automatic_adoption',
        'fresh_blind_validation','Public_feedback_used_for_tuning','individual_seed_selection'))
    assert r['query_labels_used_only_for_evaluation'] is True
    assert r['protected_before'] == r['protected_after'] == plan['protected_before']
    assert r['selector_sha256_before'] == r['selector_sha256_after'] == plan['selector_sha256_before']
    assert diag['round'] == 83 and diag['candidate'] == 'R83' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert diag['input_sha256']['outputs/round83/results.json'] == inputs['outputs/round83/results.json']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R83' and paired['control'] == 'v23'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = dict(round=83, status='completed_JSON_only_review', primary_comparison='R83 minus v23=R80',
        secondary_comparison='R83 minus v22=R79', adoption_decision=None, is_decision_authority=False,
        new_fits_in_reporter=0, model_or_array_loads=0, new_scores=0, new_bootstrap=False,
        independent_vector_replay_performed=False, new_CSV=False, selector_modified=False,
        source_sha256=sha(Path(__file__)), input_sha256=inputs, observed_receipts=receipts,
        started_at_utc=started, executions=executions, checked_stages=list(STAGES), stages={},
        main_seconds=r['elapsed_seconds'], diagnostics_seconds=diag['elapsed_seconds'], paired_seconds=paired['elapsed_seconds'],
        main_declared_counts={k:r[k] for k in ('new_fits','new_NN_fits','new_reference_fits','new_spatial_state_builds','reference_predict_calls','spatial_predict_calls')},
        main_declared_protected_file_count=len(r['protected_before']), main_declared_protected_files_equal=True,
        main_declared_selector_unchanged=True, main_evidence_only_no_independent_array_check=True)
    for s in STAGES:
        a, dg, ps = r['stages'][s], diag['stages'][s], paired['stages'][s]
        a79, a80 = priors['v22']['stages'][s], priors['v23']['stages'][s]
        assert a['status'] == 'completed' and a['fit_rows'] == dg['fit_rows'] == a79['fit_rows'] == a80['fit_rows']
        assert a['query_rows'] == dg['query_rows'] == ps['rows'] == a79['query_rows'] == a80['query_rows']
        assert a['new_fits'] == 1 and a['new_NN_fits'] == 0 and a['reference_predict_calls'] == a['spatial_predict_calls'] == 2
        assert all(a[k] is True for k in CHECKS)
        assert a['seed42_retrained_or_model_reloaded'] is False and a['individual_seed_selection'] is False
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        assert all(a['metrics'][k] == v for k,v in dg['metrics'].items())
        for label, before, key in (('v23',a80,'R80'),('v22',a79,'R79')):
            assert a['metrics'][label] == before['metrics'][key]
            assert dg['extra_control_existing_test_mix_checks'][label] == 'exact'
        assert a['original_frame_sha256'] == a79['original_frame_sha256'] == a80['original_frame_sha256']
        layers, native = a['unit_branch_metrics'], a['native_metrics']
        assert layers['control'] == a80['unit_branch_metrics']['candidate']
        assert native['seed42'] == a79['unit_branch_metrics']['native'] == layers['control']['base']
        assert native['mean'] == layers['candidate']['base']
        differences = {k:metric_deltas(a['metrics'][k],a['metrics']['R83']) for k in CONTROLS}
        layer_differences = {k:metric_deltas(layers['control'][k],layers['candidate'][k]) for k in ('base','reference','spatial')}
        native_differences = {k:metric_deltas(native['seed42'],native[k]) for k in ('seed137','mean')}
        for saved, calculated in ((a['differences'],differences),(a['unit_branch_differences'],layer_differences),(a['native_differences_vs_seed42'],native_differences)):
            for key in calculated:
                for metric in ('overall','weighted'):
                    assert near(saved[key][metric]['delta_TWD'],calculated[key][metric]['delta_TWD'])
        model_path = f'outputs/round83/{s}/model'
        meta_name = model_path+'/metadata.json'
        m = small(meta_name,a['artifact_sha256'][meta_name])
        fe_name, state_name, fit_name = (model_path+'/'+k+'.json' for k in ('fit_execution','feature_state','fit_metadata'))
        fe, state, fm = (small(name,a['artifact_sha256'][name]) for name in (fe_name,state_name,fit_name))
        before = plan['bindings'][s]['R79_metadata']
        assert fe['exit_code'] == 0 and fe['failure'] is None and fe['automatic_retry'] is False
        assert near(fe['elapsed_seconds'],m['fit_and_save_seconds']) and near(a['fit_seconds'],m['fit_seconds'])
        assert m['round'] == 83 and m['stage'] == s and m['fit_rows'] == a['fit_rows'] and m['query_rows'] == a['query_rows']
        assert m['seed'] == m['actual_random_seed'] == 137 and m['prediction_units'] == 'TWD'
        assert m['feature_count'] == 95 and m['categorical_feature_count'] == 23 and m['actual_iterations'] == 2999
        assert m['parameters'] == fm['parameters'] == state['parameters'] == plan['parameters'] == {**before['parameters'],'random_seed':137}
        assert m['baseline_parameters'] == state['baseline_parameters'] == before['parameters']
        assert m['changed_parameter_names'] == state['changed_parameter_names'] == ['random_seed']
        assert state['legacy_feature_state'] == plan['bindings'][s]['R79_state']
        assert m['legacy_state_parameters'] == state['legacy_feature_state']['parameters'] == before['parameters']
        assert state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
        assert m['legacy_feature_state_content_sha256'] == state['legacy_feature_state']['state_content_sha256'] == before['feature_state_content_sha256']
        assert m['feature_state_content_sha256'] == state['state_content_sha256']
        for key in WEIGHT_HASH_KEYS:
            assert m[key] == fm[key] == before[key], key
        assert m['fit_date_weights_sha256'] == m['R63_preparation']['train_sample_weight_sha256']
        assert m['original_frame_sha256'] == a['original_frame_sha256'] and m['plan_sha256'] == r['plan_sha256']
        for kind in ('fit','query'):
            assert m[kind+'_indices_content_sha256'] == plan['split_contract'][s][kind+'_indices_content_sha256'] == before[kind+'_indices_content_sha256']
        assert m['source_sha256']['src/round83_seed_unit.py'] == state['source_sha256']['src/round83_seed_unit.py'] == PINS['src/round83_seed_unit.py']
        assert all(m[k] is True for k in ('reload_all_query_rows_verified','reload_raw_exact','reload_TWD_exact',
            'original_R79_legacy_feature_state_exact','date_weight_provenance_verified_in_driver',
            'legacy_state_training_fields_are_provenance_only','weight_argument_passed_to_pool'))
        assert all(m[k] is False for k in ('query_labels_used','weights_use_fit_target','weights_use_query',
            'weights_normalized_to_mean_one','date_weights_recomputed','date_weights_renormalized','fit_uses_eval_set','early_stopping_used'))
        assert m['reload_rows'] == a['query_rows'] and m['weight_dtype'] == 'float64'
        assert all(before[k] is True for k in ('reload_all_query_rows_verified','reload_raw_exact','reload_TWD_exact'))
        fitted, loaded = m['fitted_native_capacity'],m['loaded_native_capacity']
        assert all(fitted[k] == loaded[k] for k in fitted if k != 'actual_learning_rate')
        for capacity in (fitted,loaded):
            assert capacity['actual_random_seed'] == 137 and capacity['actual_trees'] == 2999 and capacity['actual_model_depth'] == 8
            assert capacity['actual_loss_function'] == capacity['actual_eval_metric'] == 'MAE'
            assert abs(capacity['actual_learning_rate']-.065) <= .065*1e-7
        wsummary = m['fit_sample_weights_summary']
        assert wsummary['rows'] == a['fit_rows'] and wsummary['min'] > 0
        for key in wsummary:
            assert near(wsummary[key],before['fit_sample_weights_summary'][key])
        assert ps['prediction_sha256']['R83'] == a['prediction_sha256'] and ps['prediction_sha256']['v23'] == a80['prediction_sha256']
        ci = {method:ps[method]['summary'] for method in ('project','month')}
        for method in ci:
            for metric in ('overall','weighted'):
                c = ci[method][metric]
                assert c['valid_replicates']+c['invalid_replicates'] == 2000
                assert near(c['candidate_mae'],a['metrics']['R83'][metric+'_mae']) and near(c['control_mae'],a['metrics']['v23'][metric+'_mae'])
                assert near(c['point_delta'],differences['v23'][metric]['delta_TWD'])
        mix, bands = dg['test_composition'], []
        assert mix['minimum_historical_cell_rows'] == 200 and [b['band'] for b in dg['price_bands']] == list(BANDS)
        assert sum(b['rows'] for b in dg['price_bands']) == a['query_rows']
        for b in dg['price_bands']:
            d = {k:delta(b[k],b['R83']) for k in CONTROLS}
            contribution = {k:0.0 if b['rows']==0 else d[k]['delta_TWD']*b['rows']/a['query_rows'] for k in CONTROLS}
            bands.append(dict(b,differences=d,contribution_to_stage_overall_delta_TWD=contribution))
        tails = {}
        for threshold,selected in (('at_least_50m',bands[-2:]),('at_least_100m',bands[-1:])):
            tails[threshold] = {}
            for k in CONTROLS:
                total = differences[k]['overall']['delta_TWD']
                assert near(sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands),total)
                tail = sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in selected)
                tails[threshold][k] = dict(rows=sum(b['rows'] for b in selected),overall_delta_contribution_TWD=tail,
                    other_bands_contribution_TWD=total-tail,signed_ratio_to_net_delta=None if total==0 else tail/total)
        out['stages'][s] = dict(fit_rows=a['fit_rows'],query_rows=a['query_rows'],metrics=a['metrics'],differences=differences,
            native_metrics=native,native_differences_vs_seed42=native_differences,unit_branch_metrics=layers,unit_branch_differences=layer_differences,
            test_composition=mix,test_composition_differences={k:delta(mix['models'][k],mix['models']['R83']) for k in CONTROLS},
            price_bands=bands,tail_contributions=tails,paired=ci,paired_support={k:ps[k]['support'] for k in ci},
            main_evidence_checks={k:a[k] for k in CHECKS},changed_unit_rows=a['changed_unit_rows'],changed_final_rows=a['changed_final_rows'],
            fit_seconds=a['fit_seconds'],stage_seconds=a['elapsed_seconds'],fit_execution=fe,
            model_evidence=dict(metadata_path=meta_name,metadata_sha256=inputs[meta_name],fit_metadata_sha256=inputs[fit_name],
                wrapper_state_sha256=inputs[state_name],actual_parameters=m['parameters'],baseline_parameters=m['baseline_parameters'],
                changed_parameter_names=m['changed_parameter_names'],fit_weight_summary=wsummary,
                weight_concentration_fraction=wsummary['effective_sample_size']/a['fit_rows'],
                identity_hashes={k:m[k] for k in WEIGHT_HASH_KEYS},fitted_native_capacity=fitted,loaded_native_capacity=loaded,
                reload_rows=m['reload_rows'],native_negative_raw_rows=m['native_negative_raw_rows'],native_zero_TWD_rows=m['native_zero_TWD_rows'],
                wrapper_preserves_R79_legacy_state=True,legacy_training_fields_are_provenance_only=True,
                seed42_reload_evidence_inherited_not_repeated=True,query_labels_used=False))
    out['total_fit_seconds'] = sum(out['stages'][s]['fit_seconds'] for s in STAGES)
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick,
        report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest())
    assert all(sha(ROOT/n)==h for n,h in inputs.items()), 'Small evidence changed'
    raw = (json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x',encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest()+'\n')
    with MD.open('x',encoding='utf8',newline='\n') as f: f.write(md)
    print(json.dumps(dict(status=out['status'],elapsed_seconds=out['elapsed_seconds'],review_sha256=sha(OUT),report_sha256=sha(MD)),ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    review()
