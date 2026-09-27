"""R77 four-cell saved-JSON report: no arrays/models/scoring/bootstrap or adoption decision."""
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
CELLS = ('v19', 'R74', 'R76', 'R77')
CONTROLS = ('R76', 'v19')
BANDS = ('<10m', '10-20m', '20-50m', '50-100m', '>=100m')
PLAN_SHA = '9d71a6929e9d8a528c065a3162b4f95dcf28658c5f8d7f51c3c01c66c74df084'
OUT = ROOT / 'outputs/round77/results_review.json'
MD = ROOT / 'reports/round77_report.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-5)


def difference(control, candidate):
    if control is None or candidate is None:
        assert control is candidate is None
        return {'delta_TWD': None, 'relative_change_pct': None}
    return {'delta_TWD': candidate - control, 'relative_change_pct': None if control == 0 else 100 * (candidate / control - 1)}


def metric_differences(control, candidate):
    assert control['rows'] == candidate['rows']
    d = {k: difference(control[k + '_mae'], candidate[k + '_mae']) for k in ('overall', 'weighted')}
    for family in ('strata', 'forecast_quarters', 'cities'):
        assert set(control[family]) == set(candidate[family])
        d[family] = {}
        for k, c in control[family].items():
            assert c['rows'] == candidate[family][k]['rows']
            d[family][k] = difference(c['mae'], candidate[family][k]['mae'])
    return d


def check_differences(saved, computed):
    for k in ('overall', 'weighted'):
        assert near(saved[k]['delta_TWD'], computed[k]['delta_TWD'])
    for family in ('strata', 'forecast_quarters'):
        for key, item in computed[family].items():
            value = saved[family][key]['delta_TWD']
            assert (value is None and item['delta_TWD'] is None) or near(value, item['delta_TWD'])


def review():
    tick = time.monotonic()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs = {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.md')
        assert p.stat().st_size < 30_000_000, 'Small JSON/text only'
        digest = sha(p)
        assert expected is None or digest == expected, name
        key = p.relative_to(ROOT).as_posix()
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return p

    def small(name):
        p = capture(name)
        q = capture(p.with_suffix('.sha256').relative_to(ROOT).as_posix())
        assert q.read_text(encoding='utf8').strip() == sha(p)
        return json.loads(p.read_text(encoding='utf8'))

    r = small('outputs/round77/results.json')
    execution = small('outputs/round77/execution.json')
    diag = small('reports/round77_diagnostics.json')
    de = small('outputs/round77/diagnostics/execution.json')
    paired = small('outputs/round77/paired_vs_R76/results.json')
    capture('reports/round77_plan.md', PLAN_SHA)
    driver_paths = [n for n, h in r['source_sha256'].items() if h == execution['source_sha256'] and Path(n).name.startswith('round77') and n.endswith('.py')]
    assert len(driver_paths) == 1, 'Unique executed R77 driver source binding required'
    capture(driver_paths[0], execution['source_sha256'])
    assert execution['exit_code'] == de['exit_code'] == 0 and execution['failure'] is None and de['failure'] is None
    assert r['round'] == 77 and r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['checked_stages'] == diag['checked_stages'] == list(STAGES)
    for k in ('new_NN_fits', 'model_loads', 'model_predict_calls', 'reference_predict_calls', 'spatial_predict_calls'):
        assert r[k] == 0, k
    assert r.get('new_submission_generated', r.get('new_CSV')) is False
    assert r['protected_before'] == r['protected_after']
    if 'selector_sha256_before' in r:
        assert r['selector_sha256_before'] == r['selector_sha256_after']
    assert diag['candidate'] == 'R77' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R77' and paired['control'] == 'R76'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = {'round': 77, 'status': 'completed_JSON_only_review', 'primary_comparison': 'R77 minus R76', 'secondary_comparison': 'R77 minus v19',
           'cells': {'v19': 'A0,U0', 'R74': 'A1,U0', 'R76': 'A0,U1', 'R77': 'A1,U1'},
           'interaction_formula': '(MAE_R77-MAE_R76)-(MAE_R74-MAE_v19)',
           'interaction_is_causal_estimate': False, 'is_decision_authority': False, 'adoption_decision': None,
           'new_fits': 0, 'model_or_array_loads': 0, 'new_scores': 0, 'new_bootstrap': False, 'new_CSV': False,
           'source_sha256': sha(Path(__file__)), 'input_sha256': inputs, 'driver_path': driver_paths[0],
           'started_at_utc': datetime.now(timezone.utc).isoformat(), 'checked_stages': [], 'stages': {},
           'main_seconds': r['elapsed_seconds'], 'execution': execution, 'diagnostics_execution': de,
           'diagnostics_seconds': diag['elapsed_seconds'], 'paired_seconds': paired['elapsed_seconds'],
           'paired_replicates': paired['replicates'], 'paired_seed': paired['seed'],
           'preserved_files_declared_equal': len(r['protected_before']), 'independent_vector_replay_performed': False}
    for s in STAGES:
        a, dg, ps = r['stages'][s], diag['stages'][s], paired['stages'][s]
        assert a['fit_rows'] == dg['fit_rows'] and a['query_rows'] == dg['query_rows'] == ps['rows']
        assert a['controls_full_metrics_exact'] and a['three_saved_cells_bitwise_exact']
        for k in ('new_NN_fits', 'model_loads', 'model_predict_calls', 'reference_predict_calls', 'spatial_predict_calls'):
            assert a[k] == 0, (s, k)
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        metrics = a['metrics']
        assert all(metrics[k] == dg['metrics'][k] for k in metrics)
        assert ps['prediction_sha256']['R77'] == a['prediction_sha256']
        differences = {k: metric_differences(metrics[k], metrics['R77']) for k in ('R76', 'v19', 'R74', 'v18')}
        for k in differences:
            check_differences(a['differences'][k], differences[k])
        interaction = {}
        for k in ('overall', 'weighted'):
            f = k + '_mae'
            value = (metrics['R77'][f] - metrics['R76'][f]) - (metrics['R74'][f] - metrics['v19'][f])
            assert near(value, a['interaction_TWD'][k])
            interaction[k] = value
        ci = {}
        for method in ('project', 'month'):
            ci[method] = {}
            for key in ('overall', 'weighted'):
                v = ps[method]['summary'][key]
                assert v['valid_replicates'] + v['invalid_replicates'] == paired['replicates']
                assert near(v['candidate_mae'], metrics['R77'][key + '_mae']) and near(v['control_mae'], metrics['R76'][key + '_mae'])
                assert near(v['point_delta'], differences['R76'][key]['delta_TWD'])
                ci[method][key] = dict(v)
        mix = dg['test_composition']
        assert mix['minimum_historical_cell_rows'] == 200
        assert [v['band'] for v in dg['price_bands']] == list(BANDS)
        bands = [{k: b[k] for k in ('band', 'rows', *CELLS)} |
                 {'differences': {k: difference(b[k], b['R77']) for k in CONTROLS}} for b in dg['price_bands']]
        out['stages'][s] = {'fit_rows': a['fit_rows'], 'query_rows': a['query_rows'], 'dates': a['dates'],
            'original_frame_sha256': a['original_frame_sha256'], 'metrics': metrics, 'differences': differences,
            'interaction_TWD': interaction, 'test_composition': mix,
            'test_composition_differences': {k: difference(mix['models'][k], mix['models']['R77']) for k in CONTROLS},
            'price_bands': bands, 'paired': ci, 'paired_support': {m: ps[m]['support'] for m in ('project', 'month')},
            'saved_three_cell_vector_and_full_metrics_checks_passed': True}
        out['checked_stages'].append(s)
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
               rendered_report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest(), human_report_is_decision_authority=False)
    assert all(sha(ROOT / n) == h for n, h in inputs.items()), 'Small evidence changed'
    raw = (json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest() + '\n')
    with MD.open('x', encoding='utf8', newline='\n') as f: f.write(md)
    print(json.dumps({'status': out['status'], 'elapsed_seconds': out['elapsed_seconds'], 'review_sha256': sha(OUT), 'report_sha256': sha(MD)}, ensure_ascii=False))


def render(r):
    row = lambda values: '| ' + ' | '.join(str(v) for v in values) + ' |'
    num = lambda v, signed=False: '—' if v is None else (f'{v:+,.2f}' if signed else f'{v:,.2f}')
    lines = ['# R77：area日期加權 × unit分支消融的四格比較', '',
      '唯一新格為 median(A1,L,U1)：A0是R71 seed42 NN完整分支，A1是R74日期加權area的NN完整分支，U0是R40 unit，U1是R76消融unit，L固定為R46 recency-log。所有輸入均是已完成reference／spatial、未取整的TWD向量；本輪不重新fit或呼叫模型、reference、spatial。',
      '主差值是R77−R76，次對照是v19／R71；負值表示MAE較小。本報告依保存結果陳述，不自動採用或改推薦。計畫固定時R76分組診斷尚未完成，這是當時狀態，不把後續完成或既有分數改寫成事前盲測。', '',
      '## 四格完整指標', '', row(['期別', '格', '組合', '整體MAE', '加權MAE']), row(['---', '---', '---', '---:', '---:'])]
    for s in STAGES:
        for cell in CELLS:
            m = r['stages'][s]['metrics'][cell]
            lines.append(row([NAMES[s], cell, r['cells'][cell], num(m['overall_mae']), num(m['weighted_mae'])]))
    lines += ['', row(['期別', '比較', '整體差／%', '加權差／%']), row(['---', '---', '---:', '---:'])]
    for s in STAGES:
        for key in CONTROLS:
            d = r['stages'][s]['differences'][key]
            lines.append(row([NAMES[s], 'R77−' + key] + [f"{num(d[k]['delta_TWD'], True)}／{d[k]['relative_change_pct']:+.4f}%" for k in ('overall', 'weighted')]))
    lines += ['', '## 描述性交互差額', '',
      '固定量=(R77−R76)−(R74−v19)，由四個同窗MAE相減；負值表示保存結果中的area加權改動在U1下較在U0下有利。它不是獨立因果估計，也不是兩次改善率相加。', '',
      row(['期別', '整體交互量TWD', '加權交互量TWD']), row(['---', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]
        lines.append(row([NAMES[s], num(a['interaction_TWD']['overall'], True), num(a['interaction_TWD']['weighted'], True)]))
    lines += ['', '## 新舊建案與四季', '', '完整保留主／次對照及所有季度，城市與缺案名組完整指標見JSON。', '',
      row(['期別', '對照', '新案差', '舊案差', 'Q1差', 'Q2差', 'Q3差', 'Q4差']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        for key in CONTROLS:
            d = r['stages'][s]['differences'][key]
            lines.append(row([NAMES[s], key] + [num(d['strata'][k]['delta_TWD'], True) for k in ('new_nonmissing', 'seen')] +
                            [num(d['forecast_quarters'][q]['delta_TWD'], True) for q in ('Q1', 'Q2', 'Q3', 'Q4')]))
    lines += ['', '## 固定test組成與五個真价群', '',
      'test組成沿city×seen/new、mincell200，僅在有支持的test質量內正規化，不是全部test／Private的預測MAE。', '',
      row(['期別', '涵蓋test', '未支持列', 'v19', 'R74', 'R76', 'R77', '差vsR76', '差vsv19']), row(['---'] + ['---:'] * 8)]
    for s in STAGES:
        a = r['stages'][s]; m = a['test_composition']; d = a['test_composition_differences']
        lines.append(row([NAMES[s], f"{m['test_mass_covered']:.6%}", m['unsupported_test_rows']] +
                        [num(m['models'][k]) for k in CELLS] + [num(d[k]['delta_TWD'], True) for k in CONTROLS]))
    lines += ['', '固定真價群僅事後描述，不能在推論時依未知答案分流或逐ID改價。所有五群及不利結果均保留。', '',
      row(['期別', '價帶', '筆數', 'R77 MAE', 'R76 MAE', '差vsR76', 'v19 MAE', '差vsv19']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s], b['band'], b['rows'], num(b['R77']), num(b['R76']), num(b['differences']['R76']['delta_TWD'], True),
                              num(b['v19']), num(b['differences']['v19']['delta_TWD'], True)]))
    lines += ['', '## 主比較 R77−R76 的全部配對區間', '',
      f"固定seed {r['paired_seed']}，每種抽法／指標／年度 {r['paired_replicates']} 次。以下是2.5%–97.5%區間，抽樣負向比例不視為Private勝率。", '',
      row(['期別', '抽法', '指標', '點差', '2.5%', '97.5%', '有效／無效', '位置']), row(['---', '---', '---', '---:', '---:', '---:', '---:', '---'])]
    for s in STAGES:
        for method in ('project', 'month'):
            for key in ('overall', 'weighted'):
                c = r['stages'][s]['paired'][method][key]
                position = '全負' if c['percentile_97_5'] < 0 else '全正' if c['percentile_2_5'] > 0 else '含／跨零'
                lines.append(row([NAMES[s], '建案' if method == 'project' else '3月循環區塊', key] + [num(c[k], True) for k in ('point_delta', 'percentile_2_5', 'percentile_97_5')] +
                                [f"{c['valid_replicates']}／{c['invalid_replicates']}", position]))
    lines += ['', '這些是固定保存模型、單seed與重用窗口的條件診斷，未校正多輪選型，沒有跨seed或新盲測證據。不能由四格交互量、某個單支改善或某年度勝負直接宣稱Private較佳；仍須看最近confirmation、跨期新舊案／季度／價帶及不確定性。', '',
      '## 執行與保留', '',
      f"主流程exit0：main {r['main_seconds']:.3f}秒、outer {r['execution']['elapsed_seconds']:.3f}秒；generic execution {r['diagnostics_execution']['elapsed_seconds']:.3f}秒，paired {r['paired_seconds']:.3f}秒。",
      f"三期0fit／模型載入／模型predict／reference／spatial呼叫。原三格保存向量及完整metrics的主程序核對通過；{r['preserved_files_declared_equal']}份保護檔前後相同。本報告只核小JSON、收據與摘要算式，未另讀陣列、模型或重算score／bootstrap。",
      '沒有新CSV、selector／registry／推薦變更或Public調參；本報告不作採用決定。v19仍是有限本機優先候選，v18已回報Public最佳2,388,386.4613，Private未知、最終版本未選。完整程式與紀錄保留，GitHub待定版後整理，現在不發布。', '',
      f"[固定計畫](round77_plan.md) · [主程式](../{r['driver_path']}) · [主結果](../outputs/round77/results.json) · [執行](../outputs/round77/execution.json)",
      '[分組診斷](round77_diagnostics.json) · [配對結果](../outputs/round77/paired_vs_R76/results.json) · [JSON彙整](../outputs/round77/results_review.json) · [報告程式](../src/round77_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
