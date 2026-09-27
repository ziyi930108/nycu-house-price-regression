"""R81 small-JSON report only; no arrays, models, rescoring or selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('development', 'confirmation', 'auxiliary')
NAMES = {'development': '開發／中間年度', 'confirmation': '確認／最新年度', 'auxiliary': '輔助／最舊年度'}
CONTROLS = ('v22', 'v23')
BANDS = ('<10m', '10-20m', '20-50m', '50-100m', '>=100m')
OUT, MD = ROOT / 'outputs/round81/results_review.json', ROOT / 'reports/round81_report.md'
PINS = {
    'src/round81_experiment.py': '19d41f49f736cd5d4ddd960fd6bb0d3f629ab3e59b575e31862d0bb65796cc38',
    'src/round81_recency_lgb.py': '8ce19dbc9800737d8de394a2bf266f6514cadd9d738a3dec1944434803b2ab3c',
    'reports/round81_plan.md': '15ce1debd8bf9433c3038761b57bdffc5d6b9d3fb7520e5408c6205a560ecef5',
    'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'outputs/round80/results.json': 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-5)


def delta(control, candidate):
    if control is None or candidate is None:
        assert control is candidate is None
        return {'delta_TWD': None, 'relative_change_pct': None}
    return {'delta_TWD': candidate - control,
            'relative_change_pct': None if control == 0 else 100 * (candidate / control - 1)}


def metric_deltas(control, candidate):
    assert control['rows'] == candidate['rows']
    d = {k: delta(control[k + '_mae'], candidate[k + '_mae']) for k in ('overall', 'weighted')}
    for family in ('strata', 'forecast_quarters', 'cities'):
        assert set(control[family]) == set(candidate[family])
        d[family] = {}
        for key, c in control[family].items():
            n = candidate[family][key]
            assert c['rows'] == n['rows']
            d[family][key] = dict(rows=c['rows'], **delta(c['mae'], n['mae']))
    return d


def review():
    tick, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs, receipts = {}, {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.ps1', '.md')
        assert p.stat().st_size < 10_000_000, 'Small JSON/source/text only'
        digest, key = sha(p), p.relative_to(ROOT).as_posix()
        assert expected is None or digest == expected, name
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
    r = small('outputs/round81/results.json')
    plan = small('outputs/round81/plan.json', r['plan_sha256'])
    priors = {'v22': small('outputs/round79/results.json'), 'v23': small('outputs/round80/results.json')}
    diag = small('reports/round81_diagnostics.json')
    paired = small('outputs/round81/paired_vs_v22/results.json')
    executions = {k: small(p) for k, p in (
        ('main', 'outputs/round81/execution.json'), ('diagnostics', 'outputs/round81/diagnostics/execution.json'),
        ('paired', 'outputs/round81/paired_vs_v22_execution.json'))}
    assert all(e['exit_code'] == 0 and e.get('failure') is None and e['automatic_retry'] is False for e in executions.values())
    assert executions['main']['source_sha256'] == PINS['src/round81_experiment.py']
    for key, source in (('diagnostics', 'src/candidate_diagnostics_v3.py'), ('paired', 'src/paired_candidate_bootstrap.py')):
        capture(source, executions[key]['source_sha256'])
    assert r['round'] == 81 and r['candidate'] == 'R81' and r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['checked_stages'] == diag['checked_stages'] == list(STAGES)
    assert all(v['checked_stages'] == list(STAGES) for v in priors.values())
    assert r['primary_control'] == 'v22' and r['secondary_control'] == 'v23'
    assert r['formula'] == plan['formula'] and r['text_plan_sha256'] == plan['text_plan_sha256'] == PINS['reports/round81_plan.md']
    assert r['source_sha256'] == plan['source_sha256'] and r['input_sha256'] == plan['input_sha256']
    assert all(r['source_sha256'][p] == PINS[p] for p in ('src/round81_experiment.py', 'src/round81_recency_lgb.py'))
    assert r['new_fits'] == plan['planned_new_fits'] == 3 and r['new_NN_fits'] == r['new_CatBoost_fits'] == 0
    assert r['reference_predict_calls'] == r['spatial_predict_calls'] == 6
    assert all(r[k] is False for k in ('new_CSV', 'new_test_predictions', 'new_submission_generated', 'automatic_adoption', 'fresh_blind_validation', 'Public_feedback_used_for_tuning'))
    assert r['query_labels_used_only_for_evaluation'] is True
    assert r['protected_before'] == r['protected_after'] and r['selector_sha256_before'] == r['selector_sha256_after']
    assert diag['candidate'] == 'R81' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R81' and paired['control'] == 'v22'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = {'round': 81, 'status': 'completed_JSON_only_review', 'primary_comparison': 'R81 minus v22=R79',
        'secondary_comparison': 'R81 minus v23=R80', 'adoption_decision': None, 'is_decision_authority': False,
        'new_fits_in_reporter': 0, 'model_or_array_loads': 0, 'new_scores': 0, 'new_bootstrap': False,
        'independent_vector_replay_performed': False, 'new_CSV': False, 'selector_modified': False,
        'source_sha256': sha(Path(__file__)), 'input_sha256': inputs, 'observed_receipts': receipts,
        'started_at_utc': started, 'executions': executions, 'main_seconds': r['elapsed_seconds'],
        'diagnostics_seconds': diag['elapsed_seconds'], 'paired_seconds': paired['elapsed_seconds'],
        'checked_stages': list(STAGES), 'stages': {}, 'main_declared_new_LightGBM_fits': 3,
        'main_declared_new_NN_fits': 0, 'main_declared_new_CatBoost_fits': 0,
        'main_declared_reference_calls': 6, 'main_declared_spatial_calls': 6,
        'main_declared_protected_file_count': len(r['protected_before']), 'main_declared_protected_files_equal': True,
        'main_declared_selector_unchanged': True, 'main_evidence_only_no_independent_array_check': True}
    checks = ('R79_base_reference_spatial_full_vectors_exact', 'v22_complete_combination_exact',
        'controls_full_metrics_exact', 'full_query_raw_TWD_reload_exact', 'date_weight_provenance_verified', 'query_labels_used_only_for_evaluation')
    for s in STAGES:
        a, dg, ps = r['stages'][s], diag['stages'][s], paired['stages'][s]
        before, secondary = (priors[k]['stages'][s] for k in CONTROLS)
        assert a['status'] == 'completed' and a['fit_rows'] == dg['fit_rows']
        assert a['query_rows'] == dg['query_rows'] == ps['rows'] == before['query_rows'] == secondary['query_rows']
        assert all(a[k] is True for k in checks)
        assert a['new_fits'] == 1 and a['new_NN_fits'] == a['new_CatBoost_fits'] == 0
        assert a['reference_predict_calls'] == a['spatial_predict_calls'] == 2
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        assert all(a['metrics'][k] == v for k, v in dg['metrics'].items())
        assert all(k in dg['metrics'] for k in ('R81', *CONTROLS))
        assert a['metrics']['v22'] == before['metrics']['R79'] and a['metrics']['v23'] == secondary['metrics']['R80']
        assert a['original_frame_sha256'] == before['original_frame_sha256'] == secondary['original_frame_sha256']
        assert all(a['unit_branch_metrics']['control'][k] == before['unit_branch_metrics'][k] for k in ('base', 'reference', 'spatial'))
        assert ps['prediction_sha256']['R81'] == a['prediction_sha256'] and ps['prediction_sha256']['v22'] == before['prediction_sha256']
        differences = {k: metric_deltas(a['metrics'][k], a['metrics']['R81']) for k in CONTROLS}
        for k in CONTROLS:
            for metric in ('overall', 'weighted'):
                assert near(differences[k][metric]['delta_TWD'], a['differences'][k][metric]['delta_TWD'])
        native = metric_deltas(a['original_LGB_native_metrics'], a['new_LGB_native_metrics'])
        for metric in ('overall', 'weighted'):
            assert near(native[metric]['delta_TWD'], a['native_LGB_difference'][metric]['delta_TWD'])
        ci = {m: ps[m]['summary'] for m in ('project', 'month')}
        for method in ci:
            for metric in ('overall', 'weighted'):
                v = ci[method][metric]
                assert v['valid_replicates'] + v['invalid_replicates'] == 2000
                assert near(v['candidate_mae'], a['metrics']['R81'][metric + '_mae']) and near(v['control_mae'], a['metrics']['v22'][metric + '_mae'])
                assert near(v['point_delta'], differences['v22'][metric]['delta_TWD'])
        mix, bands = dg['test_composition'], []
        assert mix['minimum_historical_cell_rows'] == 200 and [b['band'] for b in dg['price_bands']] == list(BANDS)
        assert sum(b['rows'] for b in dg['price_bands']) == a['query_rows']
        for b in dg['price_bands']:
            d = {k: delta(b[k], b['R81']) for k in CONTROLS}
            contribution = {k: 0.0 if b['rows'] == 0 else d[k]['delta_TWD'] * b['rows'] / a['query_rows'] for k in CONTROLS}
            bands.append(dict(b, differences=d, contribution_to_stage_overall_delta_TWD=contribution))
        tails = {}
        for k in CONTROLS:
            total = differences[k]['overall']['delta_TWD']
            assert near(sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands), total)
            tail = sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands[-2:])
            tails[k] = {'rows': sum(b['rows'] for b in bands[-2:]), 'overall_delta_contribution_TWD': tail,
                'other_bands_contribution_TWD': total - tail, 'signed_ratio_to_net_delta': None if total == 0 else tail / total}
        model_path = f'outputs/round81/{s}/model/metadata.json'
        meta = small(model_path, a['artifact_sha256'][model_path])
        fit_path = f'outputs/round81/{s}/model/fit_execution.json'
        fit_execution = small(fit_path, a['artifact_sha256'][fit_path])
        assert fit_execution['exit_code'] == 0 and fit_execution['failure'] is None
        assert meta['family'] == 'lightgbm' and meta['stage'] == s and meta['round'] == 81
        assert meta['parameters'] == plan['parameters'] and meta['feature_count'] == 87 and meta['categorical_feature_count'] == 21
        assert meta['actual_trees'] == meta['actual_iterations'] == 3999 and meta['actual_seed'] == 42
        assert meta['source_sha256']['src/round81_recency_lgb.py'] == PINS['src/round81_recency_lgb.py']
        assert meta['prediction_units'] == 'TWD' and meta['fit_rows'] == a['fit_rows'] and meta['reload_rows'] == meta['query_rows'] == a['query_rows']
        assert meta['original_frame_sha256'] == a['original_frame_sha256']
        assert all(meta[k] is True for k in ('reload_all_query_rows_verified', 'reload_raw_exact', 'reload_TWD_exact', 'date_weight_provenance_verified_in_driver'))
        assert all(meta[k] is False for k in ('query_labels_used', 'fit_uses_eval_set', 'early_stopping_used', 'date_weights_recomputed', 'date_weights_renormalized', 'combined_weights_renormalized', 'weights_use_query', 'weights_use_fit_target'))
        assert meta['fitted_native_capacity'] == meta['loaded_native_capacity']
        assert near(meta['fit_seconds'], a['fit_seconds']) and near(meta['fit_elapsed_seconds'], a['fit_seconds'])
        summaries = {k: meta[k] for k in ('fit_area_weights_summary', 'fit_date_weights_summary', 'fit_sample_weights_summary')}
        assert all(v['rows'] == a['fit_rows'] for v in summaries.values())
        out['stages'][s] = {'fit_rows': a['fit_rows'], 'query_rows': a['query_rows'], 'metrics': a['metrics'],
            'differences': differences, 'native_LGB_difference': native, 'original_LGB_native_metrics': a['original_LGB_native_metrics'],
            'new_LGB_native_metrics': a['new_LGB_native_metrics'], 'unit_branch_metrics': a['unit_branch_metrics'],
            'unit_branch_differences': a['unit_branch_differences'], 'test_composition': mix,
            'test_composition_differences': {k: delta(mix['models'][k], mix['models']['R81']) for k in CONTROLS},
            'price_bands': bands, 'at_least_50m_contributions': tails, 'paired': ci,
            'paired_support': {m: ps[m]['support'] for m in ci}, 'main_evidence_checks': {k: a[k] for k in checks},
            'changed_unit_rows': a['changed_unit_rows'], 'changed_final_rows': a['changed_final_rows'],
            'fit_seconds': a['fit_seconds'], 'stage_seconds': a['elapsed_seconds'], 'fit_execution': fit_execution,
            'weights': summaries, 'parameters': meta['parameters'], 'native_capacity': meta['loaded_native_capacity'],
            'native_negative_raw_rows': meta['native_negative_raw_rows'], 'native_zero_TWD_rows': meta['native_zero_TWD_rows'],
            'main_declared_reload_rows': meta['reload_rows'], 'main_declared_full_query_raw_TWD_reload_exact': True}
    out['total_fit_seconds'] = sum(a['fit_seconds'] for a in out['stages'].values())
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
               report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest())
    assert all(sha(ROOT / n) == h for n, h in inputs.items()), 'Small evidence changed'
    raw = (json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest() + '\n')
    with MD.open('x', encoding='utf8', newline='\n') as f: f.write(md)
    print(json.dumps({'status': out['status'], 'elapsed_seconds': out['elapsed_seconds'], 'review_sha256': sha(OUT), 'report_sha256': sha(MD)}, ensure_ascii=False))


def render(r):
    row = lambda xs: '| ' + ' | '.join(str(x) for x in xs) + ' |'
    num = lambda x, sign=False: '—' if x is None else (f'{x:+,.2f}' if sign else f'{x:,.2f}')
    lines = ['# R81：原單價 LightGBM 加入固定日期權重', '',
        '唯一新訓練變動是原87欄／21類別、3999樹、seed42單價LightGBM的sample weight：各期原面積權重乘保存R63的float32日期權重（精確轉float64），不重算、不重歸一或裁切。原單價target與TWD逆轉換不變。',
        '只替換v22單價base中的25%舊LightGBM：.75×相同R79 native TWD＋.25×新LGB TWD，自己的reference trend及R39 spatial各一次後，再與固定R71 NN、R46 log分支取median。R79已是TWD，不能再乘面積。v23則是移除舊LGB的既存對照。',
        '以下差值一律R81−對照，負值為MAE較小。本報告只彙整已保存證據，不作採用／final fit／CSV或推薦決定。', '',
        '## 三期完整候選', '', row(['期別','v22 MAE','v23 MAE','R81 MAE','差vs22','加權差vs22','差vs23','加權差vs23']),row(['---']+['---:']*7)]
    for s in STAGES:
        a=r['stages'][s]; d=a['differences']
        lines.append(row([NAMES[s],*[num(a['metrics'][k]['overall_mae']) for k in ('v22','v23','R81')],*[num(d[k][m]['delta_TWD'],True) for k in CONTROLS for m in ('overall','weighted')]]))
    lines += ['',row(['期別','v22加權','v23加權','R81加權','v22 testmix','v23 testmix','R81 testmix','mix差vs22','mix差vs23']),row(['---']+['---:']*8)]
    for s in STAGES:
        a=r['stages'][s]
        lines.append(row([NAMES[s],*[num(a['metrics'][k]['weighted_mae']) for k in ('v22','v23','R81')],*[num(a['test_composition']['models'][k]) for k in ('v22','v23','R81')],*[num(a['test_composition_differences'][k]['delta_TWD'],True) for k in CONTROLS]]))
    for s in STAGES:
        a=r['stages'][s]
        for k in CONTROLS:
            d=a['differences'][k]
            lines.append(f"\n{NAMES[s]}對{k}：整體 {d['overall']['relative_change_pct']:+.4f}%，加權 {d['weighted']['relative_change_pct']:+.4f}%。")
    lines += ['', 'testmix沿固定city×seen/new、歷史cell至少200列；僅在有支持的test質量內正規化，不是已知Private MAE。']
    for s in STAGES:
        m=r['stages'][s]['test_composition']; lines.append(f"- {NAMES[s]}涵蓋 {m['test_mass_covered']:.4%}，未支持 {m['unsupported_test_rows']} 列。")
    lines += ['', 'v20/v21/v19/v18與通用背景控制完整指標保留於主結果／分組JSON；不依背景排名選另一候選。', '',
        '## LGB原生與融合傳遞', '',row(['期別','層','v22對照MAE','R81 MAE','整體差','加權差']),row(['---','---']+['---:']*4)]
    for s in STAGES:
        a=r['stages'][s]
        pairs=[('LGB native',a['original_LGB_native_metrics'],a['new_LGB_native_metrics'])]
        pairs += [(k,a['unit_branch_metrics']['control'][k],a['unit_branch_metrics']['candidate'][k]) for k in ('base','reference','spatial')]
        pairs += [('final median',a['metrics']['v22'],a['metrics']['R81'])]
        for layer,c,n in pairs:
            lines.append(row([NAMES[s],layer,num(c['overall_mae']),num(n['overall_mae']),num(n['overall_mae']-c['overall_mae'],True),num(n['weighted_mae']-c['weighted_mae'],True)]))
    for s in STAGES:
        a=r['stages'][s]
        lines.append(f"\n{NAMES[s]}：unit spatial變動 {a['changed_unit_rows']} 列，最終median變動 {a['changed_final_rows']} 列。")
    lines += ['', '單支改善不保證median改善；reference／spatial與median存在交互，逐層差額不是可相加的獨立收益或因果分解。', '',
        '## 新舊建案、季度與退步城市', '',row(['期別','群','列數','v22 MAE','R81 MAE','差vs22','差vs23']),row(['---','---']+['---:']*5)]
    for s in STAGES:
        a=r['stages'][s]
        for family in ('strata','forecast_quarters'):
            for key,c in a['metrics']['v22'][family].items():
                n=a['metrics']['R81'][family][key]
                lines.append(row([NAMES[s],key,c['rows'],num(c['mae']),num(n['mae']),*[num(a['differences'][k][family][key]['delta_TWD'],True) for k in CONTROLS]]))
    for s in STAGES:
        a=r['stages'][s]
        bad=[]
        for key,d in a['differences']['v22']['cities'].items():
            if d['delta_TWD'] is not None and d['delta_TWD']>0: bad.append(f"{key} {num(d['delta_TWD'],True)}元（n={d['rows']}）")
        lines.append(f"\n{NAMES[s]}對v22退步城市："+('；'.join(bad) if bad else '無')+'。')
    lines += ['', '完整城市好壞格見[原分組JSON](round81_diagnostics.json)；小群樣本不應被當成穩健規律。', '',
        '## 五個真價帶與高價尾端', '',row(['期別','真價帶','列數','v22 MAE','R81 MAE','差vs22','差vs23','對全期差vs22貢獻']),row(['---','---']+['---:']*6)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s],b['band'],b['rows'],num(b['v22']),num(b['R81']),*[num(b['differences'][k]['delta_TWD'],True) for k in CONTROLS],num(b['contribution_to_stage_overall_delta_TWD']['v22'],True)]))
    lines += ['',row(['期別','≥5000萬列數','高價尾端貢獻vs22','其他價帶貢獻','有號淨差比']),row(['---']+['---:']*4)]
    for s in STAGES:
        h=r['stages'][s]['at_least_50m_contributions']['v22']; share=h['signed_ratio_to_net_delta']
        lines.append(row([NAMES[s],h['rows'],num(h['overall_delta_contribution_TWD'],True),num(h['other_bands_contribution_TWD'],True),'—' if share is None else f'{share:+.2%}']))
    lines += ['', '貢獻＝群MAE差×群列數／全期列數。正為拖累、負為改善；有號淨差比可能負值或超過100%，不是穩定收益占比。真價帶是事後診斷，推論不可依真價分組或逐ID改價。', '',
        '## R81−v22全部12個配對區間', '', '建案／3月循環區塊各2000次、seed20260925；負向抽樣比例不是Private勝率。', '',
        row(['期別','抽法','指標','點差','2.5%','97.5%','有效／無效','位置']),row(['---','---','---']+['---:']*4+['---'])]
    for s in STAGES:
        for method in ('project','month'):
            for metric in ('overall','weighted'):
                c=r['stages'][s]['paired'][method][metric]
                where='全負' if c['percentile_97_5']<0 else '全正' if c['percentile_2_5']>0 else '含／跨零'
                lines.append(row([NAMES[s],method,metric]+[num(c[k],True) for k in ('point_delta','percentile_2_5','percentile_97_5')]+[f"{c['valid_replicates']}／{c['invalid_replicates']}",where]))
    lines += ['', '## 訓練、重載與解讀限制', '',row(['期別','LGB fit秒','完整期秒','重載列数','raw負值／TWD零值','日期權重集中比例','乘積權重集中比例']),row(['---']+['---:']*6)]
    for s in STAGES:
        a=r['stages'][s]; w=a['weights']; d=w['fit_date_weights_summary']; p=w['fit_sample_weights_summary']
        lines.append(row([NAMES[s],f"{a['fit_seconds']:.3f}",f"{a['stage_seconds']:.3f}",a['main_declared_reload_rows'],f"{a['native_negative_raw_rows']}／{a['native_zero_TWD_rows']}",f"{d['effective_sample_size']/d['rows']:.2%}",f"{p['effective_sample_size']/p['rows']:.2%}"]))
    lines += ['', f"3次新LightGBM fit合計 {r['total_fit_seconds']:.3f}秒；main {r['main_seconds']:.3f}秒／外層 {r['executions']['main']['elapsed_seconds']:.3f}秒；generic {r['diagnostics_seconds']:.3f}秒／外層 {r['executions']['diagnostics']['elapsed_seconds']:.3f}秒；paired {r['paired_seconds']:.3f}秒／外層 {r['executions']['paired']['elapsed_seconds']:.3f}秒。三份實際execution均exit0，無自動重試。",
        '原87欄／21類別、3999樹／seed42／CPU4threads，fit-only詞表；沒有query y進fit、驗證早停或選樹數。主程式記錄先保存模型與state，再完整query重載比對raw單價及TWD逐位相同；raw負值沿原投影至零規則處理，沒有刪列或新增價格上限。',
        f"0新NN／CatBoost fit，reference與spatial無重訓；對照和候選合計reference 6次、spatial 6次。主程式記錄控制全向量與完整metrics一致，{r['main_declared_protected_file_count']}份CSV／registry檔案與selector前後不變。",
        '權重集中比例為Kish量／原fit列數，只描述權重集中度，並非獨立樣本數或驗證有效量。乘上日期權重而不重新正規化會改變權重總量，可能連動LightGBM的有效正則化；本輪不把這另改成新變體。',
        '本報告只讀保存小JSON、來源及實際收據並核摘要算式；沒有獨立載入arrays、重播模型或重算score／bootstrap。模型及canonical／query隔離的確認是主流程所保存的證據，不冒稱再次獨立重演。',
        '三個窗口已多輪用於選型，並非新增盲測；單seed與共用分支可能共同外推，配對區間沒有校正累積選型。沒有Public調參，Private仍未知，最終版本未選。本報告不自動採用、不final fit、不新增CSV或變更selector。程式／原模型／來源／執行紀錄保留，GitHub待定版後整理，尚未發布。', '',
        '[固定計畫](round81_plan.md) · [主程式](../src/round81_experiment.py) · [訓練helper](../src/round81_recency_lgb.py) · [主結果](../outputs/round81/results.json) · [主執行](../outputs/round81/execution.json)',
        '[完整分組](round81_diagnostics.md) · [分組JSON](round81_diagnostics.json) · [配對結果](../outputs/round81/paired_vs_v22/results.json) · [配對執行](../outputs/round81/paired_vs_v22_execution.json)',
        '[本報告JSON](../outputs/round81/results_review.json) · [可重現報告器](../src/round81_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
