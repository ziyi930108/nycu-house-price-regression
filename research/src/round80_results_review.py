"""R80 report from saved small JSON only: no arrays, models, scoring or decision."""
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
CONTROLS = ('v22', 'v20', 'v21', 'v19', 'v18')
BANDS = ('<10m', '10-20m', '20-50m', '50-100m', '>=100m')
OUT, MD = ROOT / 'outputs/round80/results_review.json', ROOT / 'reports/round80_report.md'
PINS = {
    'src/round80_unit_lgb_ablation.py': '9f30fd4ef093b8dbe29c48f256e4831c7381c6e355c54758f7440ad8e9159d8e',
    'reports/round80_plan.md': '92020b607a34d5ded10da23858219b664d24f4f2703b68b258e44fe7ba728cfe',
    'outputs/round80/results.json': 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6',
    'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'reports/round80_diagnostics.json': '61a1a989195317734acc88b7b00a658ff12d1d05ae1b4a032f7f1c18457ca10a',
    'outputs/round80/paired_vs_v22/results.json': '8413a67e96e79626a070eca9e5d7f3e40db6b9e6d023ad97b70aaca9a267c50f',
    'src/round80_diagnostics.ps1': '0d2b6f06b45aa028e7c945e09025a9d98997251aeb88b1e39d26965d7e075716',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-5)


def delta(a, b):
    if a is None or b is None:
        assert a is b is None
        return {'delta_TWD': None, 'relative_change_pct': None}
    return {'delta_TWD': b - a, 'relative_change_pct': None if a == 0 else 100 * (b / a - 1)}


def review():
    tick, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs, receipts = {}, {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.ps1', '.md')
        assert p.stat().st_size < 10_000_000, 'Only small JSON/source/text'
        digest, key = sha(p), p.relative_to(ROOT).as_posix()
        assert expected is None or digest == expected, name
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return p

    def small(name, expected=None):
        p = capture(name, expected or PINS.get(name))
        actual = [q for q in (p.with_suffix('.sha256'), Path(str(p) + '.sha256')) if q.exists()]
        assert actual, 'Actual receipt missing: ' + name
        receipts[name] = [q.relative_to(ROOT).as_posix() for q in actual]
        for q in actual:
            capture(q)
            assert q.read_text(encoding='utf8').strip().split()[0] == sha(p)
        return json.loads(p.read_text(encoding='utf8'))

    for name, digest in PINS.items():
        capture(name, digest)
    r, prior = small('outputs/round80/results.json'), small('outputs/round79/results.json')
    plan = small('outputs/round80/plan.json', r['plan_sha256'])
    diag, paired = small('reports/round80_diagnostics.json'), small('outputs/round80/paired_vs_v22/results.json')
    executions = {k: small(p) for k, p in (
        ('main', 'outputs/round80/execution.json'), ('diagnostics', 'outputs/round80/diagnostics/execution.json'),
        ('paired', 'outputs/round80/paired_vs_v22_execution.json'))}
    assert all(x['exit_code'] == 0 and x.get('failure') is None and x['automatic_retry'] is False for x in executions.values())
    assert executions['main']['source_sha256'] == PINS['src/round80_unit_lgb_ablation.py']
    for k in ('diagnostics', 'paired'):
        path = 'src/candidate_diagnostics_v3.py' if k == 'diagnostics' else 'src/paired_candidate_bootstrap.py'
        capture(path, executions[k]['source_sha256'])
    assert executions['paired']['wrapper_sha256'] == PINS['src/round80_diagnostics.ps1']
    assert r['round'] == 80 and r['candidate'] == 'R80' and r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['checked_stages'] == diag['checked_stages'] == prior['checked_stages'] == list(STAGES)
    assert plan['text_plan_sha256'] == r['text_plan_sha256'] == PINS['reports/round80_plan.md']
    zero_fields = ('new_fits', 'new_NN_fits', 'tree_NN_model_loads', 'model_predict_calls', 'new_reference_fits', 'new_spatial_state_builds')
    assert all(r[k] == 0 for k in zero_fields)
    assert r['reference_predict_calls'] == r['spatial_predict_calls'] == 6
    assert all(r[k] is False for k in ('new_CSV', 'new_test_predictions', 'new_submission_generated', 'automatic_adoption', 'fresh_blind_validation', 'seed_weight_threshold_search', 'Public_feedback_used_for_tuning'))
    assert r['protected_before'] == r['protected_after'] and r['selector_sha256_before'] == r['selector_sha256_after']
    assert diag['candidate'] == 'R80' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R80' and paired['control'] == 'v22'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = {'round': 80, 'status': 'completed_JSON_only_review', 'primary_comparison': 'R80 minus v22=R79',
        'adoption_decision': None, 'is_decision_authority': False, 'new_fits_in_reporter': 0,
        'model_or_array_loads': 0, 'new_scores': 0, 'new_bootstrap': False, 'independent_vector_replay_performed': False,
        'new_CSV': False, 'selector_modified': False, 'source_sha256': sha(Path(__file__)),
        'input_sha256': inputs, 'observed_receipts': receipts, 'started_at_utc': started,
        'executions': executions, 'main_seconds': r['elapsed_seconds'], 'diagnostics_seconds': diag['elapsed_seconds'],
        'paired_seconds': paired['elapsed_seconds'], 'checked_stages': list(STAGES), 'stages': {},
        'main_declared_calls': {k: r[k] for k in (*zero_fields, 'reference_predict_calls', 'spatial_predict_calls')},
        'main_declared_protected_file_count': len(r['protected_before']), 'main_declared_protected_files_equal': True,
        'main_declared_selector_unchanged': True, 'main_evidence_only_no_independent_array_check': True}
    checks = ('R79_base_reference_spatial_full_vectors_exact', 'R79_unit_branch_full_metrics_exact',
        'v22_complete_combination_exact', 'controls_full_metrics_exact', 'saved_R79_native_TWD_used_without_area_conversion',
        'inherited_R79_model_reload_evidence_verified', 'query_labels_used_only_for_evaluation')
    for s in STAGES:
        a, dg, ps, before = r['stages'][s], diag['stages'][s], paired['stages'][s], prior['stages'][s]
        assert a['status'] == 'completed' and a['fit_rows'] == dg['fit_rows']
        assert a['query_rows'] == dg['query_rows'] == ps['rows'] == before['query_rows']
        assert all(a[k] is True for k in checks) and a['independent_model_reload_performed'] is False
        assert all(a[k] == 0 for k in zero_fields[:4]) and a['reference_predict_calls'] == a['spatial_predict_calls'] == 2
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        assert a['metrics'] == dg['metrics'] and a['metrics']['v22'] == before['metrics']['R79']
        assert a['original_frame_sha256'] == before['original_frame_sha256']
        assert a['native_metrics'] == before['unit_branch_metrics']['native']
        assert a['unit_branch_metrics']['candidate']['base'] == a['native_metrics']
        assert all(a['unit_branch_metrics']['control'][k] == before['unit_branch_metrics'][k] for k in ('base', 'reference', 'spatial'))
        assert ps['prediction_sha256']['R80'] == a['prediction_sha256'] and ps['prediction_sha256']['v22'] == before['prediction_sha256']
        for label in CONTROLS:
            for metric in ('overall', 'weighted'):
                d = delta(a['metrics'][label][metric + '_mae'], a['metrics']['R80'][metric + '_mae'])
                assert near(d['delta_TWD'], a['differences'][label][metric]['delta_TWD'])
        ci = {}
        for method in ('project', 'month'):
            ci[method] = ps[method]['summary']
            for metric, v in ci[method].items():
                assert metric in ('overall', 'weighted') and v['valid_replicates'] + v['invalid_replicates'] == 2000
                assert near(v['candidate_mae'], a['metrics']['R80'][metric + '_mae']) and near(v['control_mae'], a['metrics']['v22'][metric + '_mae'])
                assert near(v['point_delta'], a['differences']['v22'][metric]['delta_TWD'])
        mix, bands = dg['test_composition'], []
        assert mix['minimum_historical_cell_rows'] == 200 and [b['band'] for b in dg['price_bands']] == list(BANDS)
        assert sum(b['rows'] for b in dg['price_bands']) == a['query_rows']
        for b in dg['price_bands']:
            d = {k: delta(b[k], b['R80']) for k in CONTROLS}
            c = {k: 0.0 if b['rows'] == 0 else d[k]['delta_TWD'] * b['rows'] / a['query_rows'] for k in CONTROLS}
            bands.append(dict(b, differences=d, contribution_to_stage_overall_delta_TWD=c))
        high = {}
        for k in CONTROLS:
            total = a['differences'][k]['overall']['delta_TWD']
            assert near(sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands), total)
            tail = sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands[-2:])
            high[k] = {'rows': sum(b['rows'] for b in bands[-2:]), 'overall_delta_contribution_TWD': tail,
                'other_bands_contribution_TWD': total - tail, 'signed_ratio_to_net_delta': None if total == 0 else tail / total}
        out['stages'][s] = {'fit_rows': a['fit_rows'], 'query_rows': a['query_rows'], 'metrics': a['metrics'],
            'differences': a['differences'], 'native_metrics': a['native_metrics'], 'unit_branch_metrics': a['unit_branch_metrics'],
            'unit_branch_differences': a['unit_branch_differences'], 'test_composition': mix,
            'test_composition_differences': {k: delta(mix['models'][k], mix['models']['R80']) for k in CONTROLS},
            'price_bands': bands, 'at_least_50m_contributions': high, 'paired': ci,
            'paired_support': {m: ps[m]['support'] for m in ci}, 'main_evidence_checks': {k: a[k] for k in checks},
            'changed_unit_rows': a['changed_unit_rows'], 'changed_final_rows': a['changed_final_rows'],
            'stage_seconds': a['elapsed_seconds']}
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
    lines = ['# R80：移除單價分支舊 LightGBM 混合', '',
        '固定單一消融：v22/R79的unit base為 .75×R79 native TWD＋.25×舊unit LightGBM；R80改成100%相同R79 native TWD。native已是總價元，不再乘面積。各自reference與R39 spatial後，均與原R71 seed42 NN、R46 recency log取median。沒有新fit、樹／NN模型載入或推論；這仍是多分支組合，不是單一模型。',
        '以下差值一律R80−對照，負值為MAE較小。結果屬跨期取捨，本報告不作採用／發布決定，也不改CSV或推薦。', '',
        '## 三期主比較', '', row(['期別','v22整體','R80整體','整體差','加權差','testmix差']), row(['---']+['---:']*5)]
    for s in STAGES:
        a=r['stages'][s]; d=a['differences']['v22']
        lines.append(row([NAMES[s],num(a['metrics']['v22']['overall_mae']),num(a['metrics']['R80']['overall_mae']),num(d['overall']['delta_TWD'],True),num(d['weighted']['delta_TWD'],True),num(a['test_composition_differences']['v22']['delta_TWD'],True)]))
    for s in STAGES:
        a=r['stages'][s]; d=a['differences']['v22']
        lines.append(f"\n{NAMES[s]}：整體 {d['overall']['relative_change_pct']:+.4f}%；加權MAE v22／R80 {num(a['metrics']['v22']['weighted_mae'])}／{num(a['metrics']['R80']['weighted_mae'])}；testmix {num(a['test_composition']['models']['v22'])}／{num(a['test_composition']['models']['R80'])}。")
    lines += ['', 'testmix沿固定city×seen/new、歷史cell至少200列；只在有支持的test質量內正規化，不是已知Private MAE。']
    for s in STAGES:
        m=r['stages'][s]['test_composition']; lines.append(f"- {NAMES[s]}涵蓋test質量 {m['test_mass_covered']:.4%}，未支持 {m['unsupported_test_rows']} 列。")
    lines += ['', '背景v20/v21/v19/v18及原通用控制的完整MAE、加權與分組指標均保留於JSON；主比較固定v22，不依背景排名改選候選。', '',
        '## 單價分支與最外median', '',row(['期別','層','對照MAE','R80 MAE','整體差','加權差']),row(['---','---']+['---:']*4)]
    for s in STAGES:
        a=r['stages'][s]
        for layer in ('base','reference','spatial','final'):
            c,n=(a['metrics']['v22'],a['metrics']['R80']) if layer=='final' else (a['unit_branch_metrics']['control'][layer],a['unit_branch_metrics']['candidate'][layer])
            lines.append(row([NAMES[s],layer,num(c['overall_mae']),num(n['overall_mae']),num(n['overall_mae']-c['overall_mae'],True),num(n['weighted_mae']-c['weighted_mae'],True)]))
    for s in STAGES:
        a=r['stages'][s]
        lines.append(f"\n{NAMES[s]}：unit spatial改變 {a['changed_unit_rows']} 列，完整median改變 {a['changed_final_rows']} 列。")
    lines += ['', '兩方案共用完全相同的原生R79向量；candidate base的指標即該native指標。單支改善不保證median改善，單支退步也不必然令median退步；逐層差額不能相加視為獨立收益或因果效果。', '',
        '## 新舊建案與不利季度／城市', '',row(['期別','分組','列數','v22 MAE','R80 MAE','差']),row(['---','---']+['---:']*4)]
    for s in STAGES:
        a=r['stages'][s]
        for k,c in a['metrics']['v22']['strata'].items():
            n=a['metrics']['R80']['strata'][k]
            lines.append(row([NAMES[s],k,c['rows'],num(c['mae']),num(n['mae']),num(delta(c['mae'],n['mae'])['delta_TWD'],True)]))
    for s in STAGES:
        a=r['stages'][s]
        for family,title in (('forecast_quarters','季度'),('cities','城市')):
            bad=[]
            for k,c in a['metrics']['v22'][family].items():
                n=a['metrics']['R80'][family][k]; d=delta(c['mae'],n['mae'])['delta_TWD']
                if d is not None and d>0: bad.append(f"{k} {num(d,True)}元（n={n['rows']}）")
            lines.append(f"\n{NAMES[s]}退步{title}："+('；'.join(bad) if bad else '無')+'。')
    lines += ['', '季度／城市所有好壞格的完整數值見[分組診斷](round80_diagnostics.md)與[原JSON](round80_diagnostics.json)，不省略未改善群。', '',
        '## 五個真價帶與尾端貢獻', '',row(['期別','真價帶','列數','v22 MAE','R80 MAE','差vs22','對全期差貢獻']),row(['---','---']+['---:']*5)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s],b['band'],b['rows'],num(b['v22']),num(b['R80']),num(b['differences']['v22']['delta_TWD'],True),num(b['contribution_to_stage_overall_delta_TWD']['v22'],True)]))
    lines += ['',row(['期別','≥5000萬列數','≥5000萬貢獻','其他價帶貢獻','有號淨差比']),row(['---']+['---:']*4)]
    for s in STAGES:
        h=r['stages'][s]['at_least_50m_contributions']['v22']; share=h['signed_ratio_to_net_delta']
        lines.append(row([NAMES[s],h['rows'],num(h['overall_delta_contribution_TWD'],True),num(h['other_bands_contribution_TWD'],True),'—' if share is None else f'{share:+.2%}']))
    lines += ['', '貢獻＝群MAE差×群列數／全期列數，僅分解整體MAE差。正為拖累、負為改善；有號淨差比可負或超過100%，不是穩定收益占比。真價帶只供事後診斷，推論不可據此路由或逐ID改價。', '',
        '## R80−v22全部12個配對區間', '', '沿原建案／3月循環區塊抽樣，各2000次、seed20260925；抽樣負向比例不是Private勝率。', '',
        row(['期別','抽法','指標','點差','2.5%','97.5%','有效／無效','位置']),row(['---','---','---']+['---:']*4+['---'])]
    for s in STAGES:
        for method in ('project','month'):
            for metric in ('overall','weighted'):
                c=r['stages'][s]['paired'][method][metric]; where='全負' if c['percentile_97_5']<0 else '全正' if c['percentile_2_5']>0 else '含／跨零'
                lines.append(row([NAMES[s],method,metric]+[num(c[k],True) for k in ('point_delta','percentile_2_5','percentile_97_5')]+[f"{c['valid_replicates']}／{c['invalid_replicates']}",where]))
    lines += ['', '## 執行與限制', '',
        f"主main {r['main_seconds']:.3f}秒／execution {r['executions']['main']['elapsed_seconds']:.3f}秒；generic {r['diagnostics_seconds']:.3f}秒／execution {r['executions']['diagnostics']['elapsed_seconds']:.3f}秒；paired {r['paired_seconds']:.3f}秒／外層 {r['executions']['paired']['elapsed_seconds']:.3f}秒。三份實際execution均exit0，沒有自動重試。",
        f"0新fit、0樹／NN模型載入／推論、0reference或spatial重訓；對照與候選合計reference 6次、spatial 6次。主程式記錄各層控制全向量及完整metrics一致、TWD不再乘面積、canonical列序一致；{r['main_declared_protected_file_count']}份既有CSV／registry保護檔及selector前後不變。reference與spatial state確有讀取，不能把這稱為完全不讀模型狀態。",
        '本報告僅讀保存小JSON／來源／實際收據並核摘要算式；沒有獨立載入arrays、重播模型或重新計算score／bootstrap。保留原生模型既有full-query reload證據，不宣稱本輪另做native模型重載。',
        '結果是跨期與分組取捨：不能只看最新年度整體、單支或高價群決定採用。三個歷史窗口已反覆用於選型，沒有新增盲測；區間未校正多輪選型，單seed与共通分支仍可能共同外推。Public不作本輪比例調整依據，Private仍未知，最終版本未定。本輪沒有test預測、新CSV或採用決定。程式、依賴、輸入與執行紀錄保留；GitHub待定版後整理，尚未發布。', '',
        '[固定計畫](round80_plan.md) · [主程式](../src/round80_unit_lgb_ablation.py) · [主結果](../outputs/round80/results.json) · [主執行](../outputs/round80/execution.json)',
        '[分組JSON](round80_diagnostics.json) · [配對結果](../outputs/round80/paired_vs_v22/results.json) · [配對執行](../outputs/round80/paired_vs_v22_execution.json) · [JSON彙整](../outputs/round80/results_review.json) · [報告程式](../src/round80_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    review()
