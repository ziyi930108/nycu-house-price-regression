"""R76 saved-JSON report. No model/array imports, scoring, bootstrap or adoption decision."""
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
LAYERS = ('base', 'reference', 'spatial')
CONTROLS = ('v19', 'v18')
BANDS = ('<10m', '10-20m', '20-50m', '50-100m', '>=100m')
OUT = ROOT / 'outputs/round76/results_review.json'
MD = ROOT / 'reports/round76_report.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-5)


def measure(m, key):
    if key in ('overall', 'weighted'):
        return m[key + '_mae']
    if key in ('seen', 'new_nonmissing', 'missing'):
        return m['strata'][key]['mae']
    return m['forecast_quarters'][key]['mae']


def difference(control, candidate):
    if control is None or candidate is None:
        assert control is candidate is None
        return {'delta_TWD': None, 'relative_change_pct': None}
    return {'delta_TWD': candidate - control,
            'relative_change_pct': None if control == 0 else 100 * (candidate / control - 1)}


def metric_differences(control, candidate):
    assert control['rows'] == candidate['rows']
    values = {k: difference(measure(control, k), measure(candidate, k)) for k in ('overall', 'weighted')}
    for family in ('strata', 'forecast_quarters', 'cities'):
        assert set(control[family]) == set(candidate[family])
        values[family] = {}
        for k, c in control[family].items():
            assert c['rows'] == candidate[family][k]['rows']
            values[family][k] = difference(c['mae'], candidate[family][k]['mae'])
    return values


def compare_saved_diff(saved, derived):
    for k in ('overall', 'weighted'):
        assert near(saved[k]['delta_TWD'], derived[k]['delta_TWD'])
    for family in ('strata', 'forecast_quarters'):
        for k, v in derived[family].items():
            actual = saved[family][k]['delta_TWD']
            assert (actual is None and v['delta_TWD'] is None) or near(actual, v['delta_TWD'])


def review():
    tick = time.monotonic()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs = {}

    def capture(name, expected=None):
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and path.suffix in ('.json', '.sha256', '.py', '.md')
        assert path.stat().st_size < 30_000_000, 'Only small JSON/text evidence'
        digest = sha(path)
        assert expected is None or digest == expected, name
        key = path.relative_to(ROOT).as_posix()
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return path

    def small(name):
        p = capture(name)
        receipt = capture(p.with_suffix('.sha256').relative_to(ROOT).as_posix())
        assert receipt.read_text(encoding='utf8').strip() == sha(p)
        return json.loads(p.read_text(encoding='utf8'))

    r = small('outputs/round76/results.json')
    execution = small('outputs/round76/execution.json')
    diag = small('reports/round76_diagnostics.json')
    diag_execution = small('outputs/round76/diagnostics/execution.json')
    paired = small('outputs/round76/paired_vs_v19/results.json')
    first_failure = small('outputs/round76_failed_attempt1/execution.json')
    retry_path = capture('outputs/round76_failed_attempt1/retry_reason.json')
    retry_reason = json.loads(retry_path.read_text(encoding='utf8'))  # This preserved record has no receipt.
    assert first_failure['exit_code'] == retry_reason['original_exit_code'] == 1
    assert first_failure['failure'] and not first_failure['automatic_retry']
    assert first_failure['source_sha256'] == retry_reason['original_source_sha256']
    assert retry_reason['new_candidate_results_before_failure'] is False
    assert retry_reason['retry_is_explicit_after_identified_code_fix'] is True
    assert retry_reason['corrected_source_sha256'] == 'ee1fe6451225882cde3b36aef2fce44c0241c5c29a50efdc716e18e929ef8d82'
    assert retry_reason['corrected_source_sha256'] == execution['source_sha256']
    diagnostics_failure = small('outputs/round76/diagnostics_failed_attempt1/execution.json')
    diagnostics_repair = small('outputs/round76/diagnostics_failed_attempt1/repair_note.json')
    assert inputs['outputs/round76/diagnostics_failed_attempt1/repair_note.json'] == '2d910a82f1dda5a15c441c4a1b898a4bd3decc50f4f18c667461f74ca24d99ed'
    assert diagnostics_failure['exit_code'] == diagnostics_repair['original_exit_code'] == 1
    assert diagnostics_failure['failure'] and "KeyError: 'v11'" in diagnostics_failure['failure']
    assert diagnostics_failure['source_sha256'] == diagnostics_repair['original_source_sha256']
    assert diagnostics_repair['preserved_file_sha256']['execution.json'] == inputs['outputs/round76/diagnostics_failed_attempt1/execution.json']
    assert diagnostics_repair['main_results_unchanged_sha256'] == inputs['outputs/round76/results.json']
    assert diagnostics_repair['adapter_source_sha256'] == 'c1e0a89c6eb2e97c2849aea4728d3a82912cf79ec61df541f8e6b80ae9e5af84'
    assert diagnostics_repair['changes_to_scoring_test_mix_price_bands'] is False
    assert diagnostics_repair['original_generic_modified'] is False and diagnostics_repair['paired_outputs_modified'] is False
    assert diagnostics_repair['automatic_retry'] is False
    capture(diagnostics_repair['original_source'], diagnostics_repair['original_source_sha256'])
    capture(diagnostics_repair['adapter_source'], diagnostics_repair['adapter_source_sha256'])
    assert diag_execution['source_sha256'] == diagnostics_repair['adapter_source_sha256']
    capture('src/round76_unit_branch_ablation.py', r['source_sha256']['src/round76_unit_branch_ablation.py'])
    capture('reports/round76_plan.md')
    assert execution['exit_code'] == diag_execution['exit_code'] == 0
    assert execution['failure'] is None and diag_execution['failure'] is None
    assert execution['source_sha256'] == r['source_sha256']['src/round76_unit_branch_ablation.py']
    assert r['round'] == 76 and r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['checked_stages'] == diag['checked_stages'] == list(STAGES)
    for k in ('new_fits', 'new_NN_fits', 'tree_NN_model_loads', 'model_predict_calls', 'new_reference_fits', 'new_spatial_state_builds'):
        assert r[k] == 0, k
    assert r['reference_predict_calls'] == r['spatial_predict_calls'] == 6
    assert r.get('new_submission_generated', r.get('new_CSV')) is False
    assert r['protected_before'] == r['protected_after'] and len(r['protected_before']) == 21
    if 'selector_sha256_before' in r:
        assert r['selector_sha256_before'] == r['selector_sha256_after']
    assert diag['candidate'] == 'R76' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R76' and paired['control'] == 'v19'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = {'round': 76, 'status': 'completed_JSON_only_review', 'candidate': 'R76', 'primary_control': 'v19=R71',
           'secondary_control': 'v18=R66', 'comparison_direction': 'R76 minus control; unit branch candidate minus original R40 branch',
           'is_decision_authority': False, 'adoption_decision': None, 'model_or_array_loads': 0,
           'new_fits': 0, 'new_scores': 0, 'new_bootstrap': False, 'new_CSV': False,
           'source_sha256': sha(Path(__file__)), 'input_sha256': inputs, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
           'checked_stages': [], 'stages': {}, 'main_seconds': r['elapsed_seconds'], 'execution': execution,
           'diagnostics_seconds': diag['elapsed_seconds'], 'diagnostics_execution': diag_execution,
           'paired_seconds': paired['elapsed_seconds'], 'paired_replicates': paired['replicates'], 'paired_seed': paired['seed'],
           'preserved_files_declared_equal': 21, 'independent_vector_replay_performed': False,
           'preserved_first_failure_execution': first_failure, 'preserved_retry_reason': retry_reason,
           'preserved_diagnostics_failure': diagnostics_failure, 'preserved_diagnostics_repair_note': diagnostics_repair,
           'unit_base_before': '.75*(.5*old_raw_TWD+.5*R19_native_TWD)+.25*old_unit_TWD',
           'unit_base_after': '.75*R19_native_TWD+.25*old_unit_TWD',
           'complete_formula': 'median(saved_R71_seed42_nn_spatial,saved_R46_recency_log_spatial,new_unit_spatial)'}
    for s in STAGES:
        a, dg, ps = r['stages'][s], diag['stages'][s], paired['stages'][s]
        assert a['fit_rows'] == dg['fit_rows'] and a['query_rows'] == dg['query_rows'] == ps['rows']
        for k in ('R19_reference_full_vector_exact', 'R40_unit_spatial_full_vector_exact', 'R19_and_R40_full_metrics_exact',
                  'v19_complete_combination_exact', 'controls_full_metrics_exact'):
            assert a[k] is True, (s, k)
        assert a['new_NN_fits'] == a['tree_NN_model_loads'] == a['model_predict_calls'] == 0
        assert a['reference_predict_calls'] == a['spatial_predict_calls'] == 2
        assert 0 <= a['changed_unit_rows'] <= a['query_rows'] and 0 <= a['changed_final_rows'] <= a['query_rows']
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        metrics = {k: a['metrics'][k] for k in ('R76', *CONTROLS)}
        assert all(m == dg['metrics'][k] for k, m in metrics.items())
        assert ps['prediction_sha256']['R76'] == a['prediction_sha256']
        differences = {k: metric_differences(metrics[k], metrics['R76']) for k in CONTROLS}
        for k in CONTROLS:
            compare_saved_diff(a['differences'][k], differences[k])
        ub = a['unit_branch_metrics']
        assert set(ub) == {'control', 'candidate'} and all(set(v) == set(LAYERS) for v in ub.values())
        branch_diff = {}
        for layer in LAYERS:
            assert ub['control'][layer]['rows'] == ub['candidate'][layer]['rows'] == a['query_rows']
            branch_diff[layer] = metric_differences(ub['control'][layer], ub['candidate'][layer])
            compare_saved_diff(a['unit_branch_differences'][layer], branch_diff[layer])
        ci = {}
        for method in ('project', 'month'):
            ci[method] = {}
            for key in ('overall', 'weighted'):
                v = ps[method]['summary'][key]
                assert v['valid_replicates'] + v['invalid_replicates'] == paired['replicates']
                assert near(v['candidate_mae'], measure(metrics['R76'], key)) and near(v['control_mae'], measure(metrics['v19'], key))
                assert near(v['point_delta'], differences['v19'][key]['delta_TWD'])
                ci[method][key] = dict(v)
        mix = dg['test_composition']
        assert mix['minimum_historical_cell_rows'] == 200
        assert [b['band'] for b in dg['price_bands']] == list(BANDS)
        bands = [{k: b[k] for k in ('band', 'rows', 'R76', *CONTROLS)} |
                 {'differences': {k: difference(b[k], b['R76']) for k in CONTROLS}} for b in dg['price_bands']]
        out['stages'][s] = {'fit_rows': a['fit_rows'], 'query_rows': a['query_rows'], 'dates': a['dates'],
            'metrics': metrics, 'differences': differences, 'unit_branch_metrics': ub, 'unit_branch_differences': branch_diff,
            'changed_unit_rows': a['changed_unit_rows'], 'changed_final_rows': a['changed_final_rows'],
            'price_bands': bands, 'test_composition': mix,
            'test_composition_differences': {k: difference(mix['models'][k], mix['models']['R76']) for k in CONTROLS},
            'paired': ci, 'paired_support': {m: ps[m]['support'] for m in ('project', 'month')},
            'all_full_diagnostic_metrics_match_main': True, 'all_recorded_control_replay_checks_passed': True}
        out['checked_stages'].append(s)
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
               rendered_report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest(), human_report_is_decision_authority=False)
    assert all(sha(ROOT / n) == d for n, d in inputs.items()), 'Small evidence changed'
    raw = (json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with OUT.open('xb') as f:
        f.write(raw)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(hashlib.sha256(raw).hexdigest() + '\n')
    with MD.open('x', encoding='utf8', newline='\n') as f:
        f.write(md)
    print(json.dumps({'status': out['status'], 'elapsed_seconds': out['elapsed_seconds'],
                      'review_sha256': sha(OUT), 'report_sha256': sha(MD), 'new_fits': 0, 'new_scores': 0}, ensure_ascii=False))


def render(r):
    row = lambda values: '| ' + ' | '.join(str(v) for v in values) + ' |'
    num = lambda v, sign=False: '—' if v is None else (f'{v:+,.2f}' if sign else f'{v:,.2f}')
    lines = ['# R76：單價分支移除 oldraw 子槽', '',
      '本輪只改單價分支的base：原 .75*(.5*oldraw+.5*R19_native)+.25*oldunit，改成 .75*R19_native+.25*oldunit。R19保存原生預測已為TWD，不再乘面積或倍率；oldunit為原LGB分量，與unit_spatial不同。R71／v19 seed42 NN完整分支、R46 recency-log及最外中位數固定。',
      '沒有新fit、樹／NN模型載入或樹／NN模型推論；有讀取既存reference與spatial state pickle。每期先重現原R19 reference、R40 unit spatial及v19完整控制，再計算新unit的reference／spatial。以下差額皆為候選減對照，負值較好；不以分支改善自動作採用決定。', '',
      '## 完整候選對 v19 與 v18', '',
      row(['期別', '對照', '對照整體', 'R76整體', '整體差／%', '對照加權', 'R76加權', '加權差／%']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        a = r['stages'][s]
        for key in CONTROLS:
            c, p, d = a['metrics'][key], a['metrics']['R76'], a['differences'][key]
            lines.append(row([NAMES[s], key, num(c['overall_mae']), num(p['overall_mae']),
                f"{num(d['overall']['delta_TWD'], True)}／{d['overall']['relative_change_pct']:+.4f}%", num(c['weighted_mae']), num(p['weighted_mae']),
                f"{num(d['weighted']['delta_TWD'], True)}／{d['weighted']['relative_change_pct']:+.4f}%"]))
    lines += ['', '## 單價完整分支的三層', '', row(['期別', '層', '舊MAE', '新MAE', '差', '舊加權', '新加權', '加權差']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        a = r['stages'][s]
        for layer in LAYERS:
            c, p = (a['unit_branch_metrics'][side][layer] for side in ('control', 'candidate'))
            d = a['unit_branch_differences'][layer]
            lines.append(row([NAMES[s], layer, num(c['overall_mae']), num(p['overall_mae']), num(d['overall']['delta_TWD'], True),
                              num(c['weighted_mae']), num(p['weighted_mae']), num(d['weighted']['delta_TWD'], True)]))
    lines.append('')
    for s in STAGES:
        a = r['stages'][s]; b = a['unit_branch_differences']['spatial']['overall']['delta_TWD']; f = a['differences']['v19']['overall']['delta_TWD']
        interpretation = ('改善' if b < 0 else '退步' if b > 0 else '相同') + '的單價分支，對應完整候選' + ('改善' if f < 0 else '退步' if f > 0 else '相同')
        lines.append(f"{NAMES[s]}：unit spatial差 {num(b, True)}元、final差 {num(f, True)}元；{interpretation}。")
    lines += ['', '最外median對變動unit的固定界限是min/max(NN分支,log分支)，不是先前R75的log/unit界限。分支和完整差額的改變不是獨立因果分解，也不能把改善率相加。', '',
      row(['期別', 'query筆數', 'unit改變筆數', 'final改變筆數']), row(['---', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]; lines.append(row([NAMES[s], a['query_rows'], a['changed_unit_rows'], a['changed_final_rows']]))
    lines += ['', '## 新舊建案與全部季度', '', '完整候選MAE差（元）；兩個對照均保留，城市與缺案名組完整指標在JSON。', '',
      row(['期別', '對照', '新案', '舊案', 'Q1', 'Q2', 'Q3', 'Q4']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        for key in CONTROLS:
            d = r['stages'][s]['differences'][key]
            lines.append(row([NAMES[s], key] + [num(d['strata'][k]['delta_TWD'], True) for k in ('new_nonmissing', 'seen')] +
                            [num(d['forecast_quarters'][q]['delta_TWD'], True) for q in ('Q1', 'Q2', 'Q3', 'Q4')]))
    lines += ['', '## 固定test組成與五個真價群', '',
      'test組成沿city×seen/new、歷史cell至少200列，僅在有支持的test質量內正規化；不是全部test／Private的預測MAE。', '',
      row(['期別', '涵蓋test', '未支持列', 'R76', 'v19', '差vs19', 'v18', '差vs18']), row(['---'] + ['---:'] * 7)]
    for s in STAGES:
        a = r['stages'][s]; m = a['test_composition']; d = a['test_composition_differences']
        lines.append(row([NAMES[s], f"{m['test_mass_covered']:.6%}", m['unsupported_test_rows'], num(m['models']['R76']), num(m['models']['v19']),
                          num(d['v19']['delta_TWD'], True), num(m['models']['v18']), num(d['v18']['delta_TWD'], True)]))
    lines += ['', '真價分群是事後診斷，不能用於推論時分流或逐ID改價；完整列出全部五群，不略過極端價格風險。', '',
      row(['期別', '價群', '筆數', 'R76 MAE', 'v19 MAE', '差vs19', 'v18 MAE', '差vs18']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s], b['band'], b['rows'], num(b['R76']), num(b['v19']), num(b['differences']['v19']['delta_TWD'], True), num(b['v18']), num(b['differences']['v18']['delta_TWD'], True)]))
    lines += ['', '## 全部成對區間：R76−v19', '',
      f"固定seed {r['paired_seed']}、每項 {r['paired_replicates']} 次；保存兩種抽法及overall／weighted的2.5%–97.5%區間，正向或負向比例不解讀為Private勝率。", '',
      row(['期別', '抽法', '指標', '點估計', '2.5%', '97.5%', '有效／無效', '區間位置']), row(['---', '---', '---', '---:', '---:', '---:', '---:', '---'])]
    for s in STAGES:
        for family in ('project', 'month'):
            for key in ('overall', 'weighted'):
                c = r['stages'][s]['paired'][family][key]
                position = '全負' if c['percentile_97_5'] < 0 else '全正' if c['percentile_2_5'] > 0 else '含／跨零'
                lines.append(row([NAMES[s], '建案' if family == 'project' else '3月循環區塊', key] +
                    [num(c[k], True) for k in ('point_delta', 'percentile_2_5', 'percentile_97_5')] + [f"{c['valid_replicates']}／{c['invalid_replicates']}", position]))
    lines += ['', '區間以保存模型與反覆使用的窗口為條件，未涵蓋重訓、選型或未來市場移轉。本輪固定原seed42保存部件，沒有新增跨seed證據；以最新confirmation及跨期新舊案／季度／價帶完整證據判斷，不是年度票數，也不因Public結果調參。', '',
      '## 執行與證據範圍', '',
      f"首次執行exit1／{r['preserved_first_failure_execution']['elapsed_seconds']:.3f}秒：把限float64預測的exact helper誤用於int64列索引，發生在任何候選預測或評分之前。原execution、來源及retry_reason保留於round76_failed_attempt1；唯一修正為int64 dtype／shape與bytes相等檢查，固定公式與事前plan未改。這不是首次全通過，也不是看到分數後重做候選。",
      f"分組診斷另有首次exit1／{r['preserved_diagnostics_failure']['elapsed_seconds']:.3f}秒：原generic工具假設主結果含v11／v12／v5，R76主結果只保存R76／v19／v18，因此v11 KeyError。失敗execution／log保存在diagnostics_failed_attempt1；新round76_diagnostics adapter僅將缺省對照的核對來源接回原R49／R50／R36結果，未改原generic、主結果、scorer／testmix／價帶或paired產物。repair_note的adapter尚未執行狀態是修復前保存時點，成功執行另以新execution為準。",
      f"修正後主流程exit0：main {r['main_seconds']:.3f}秒、outer {r['execution']['elapsed_seconds']:.3f}秒。修復版分組診斷execution {r['diagnostics_execution']['elapsed_seconds']:.3f}秒；paired程式 {r['paired_seconds']:.3f}秒。",
      '3期0fit／tree或NN載入／模型推論；原、新unit各reference與spatial一次，因此總計各6次，沒有新增reference fit或spatial state。R19／R40／v19控制向量及完整metrics保存核對通过；21份舊CSV／registry不變。此報告只核小JSON／收據及摘要算式，不載陣列或模型、不重新score／bootstrap，不聲稱另做逐列重播。',
      '這份報告不作採用決定、不製新CSV、不改selector。單價分支即使改善，也不能保證中位數final改善或減少每類房屋的誤差；原面積外推與極端值風險仍須保留。v19仍為有限本機優先候選，v18已回報Public最佳2,388,386.4613；R76的Public／Private未知，最終版本未選。全部程式與紀錄保留，GitHub待定版後整理，現在不發布。', '',
      '[固定計畫](round76_plan.md) · [主程式](../src/round76_unit_branch_ablation.py) · [主結果](../outputs/round76/results.json) · [執行紀錄](../outputs/round76/execution.json)',
      '[首次失敗執行](../outputs/round76_failed_attempt1/execution.json) · [明示修正原因](../outputs/round76_failed_attempt1/retry_reason.json)',
      '[分組工具首次失敗](../outputs/round76/diagnostics_failed_attempt1/execution.json) · [保存修復說明](../outputs/round76/diagnostics_failed_attempt1/repair_note.json) · [本輪分組adapter](../src/round76_diagnostics.py)',
      '[分組診斷](round76_diagnostics.json) · [配對結果](../outputs/round76/paired_vs_v19/results.json) · [JSON彙整](../outputs/round76/results_review.json) · [報告程式](../src/round76_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
