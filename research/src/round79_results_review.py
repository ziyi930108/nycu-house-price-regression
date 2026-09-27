"""R79 saved-small-JSON report only; never loads arrays/models or scores predictions."""
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
CONTROLS = ('v20', 'v21', 'v19', 'v18')
BANDS = ('<10m', '10-20m', '20-50m', '50-100m', '>=100m')
PINS = {
    'src/round79_experiment.py': '33e59572785a4500288246b09a2a5aed038d83482f7ac79ac2050846cf530db6',
    'src/round79_recency_unit.py': '9a124322a139e5621c28d26a6e7ab95385c73507f017a2765d9eaaa0eef39cd2',
    'reports/round79_plan.md': '4992740d273f3d0a4655037519fedd0e94c3775fcbf30216c6c3f4afbda079a9',
    'outputs/round76/results.json': 'a7ea0219deec6aacaf43be9abd9b62e41b0cd720d09274d6fa81c5fa34d81631',
}
OUT = ROOT / 'outputs/round79/results_review.json'
MD = ROOT / 'reports/round79_report.md'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-5)


def delta(control, candidate):
    if control is None or candidate is None:
        assert control is candidate is None
        return {'delta_TWD': None, 'relative_change_pct': None}
    return {'delta_TWD': candidate - control,
            'relative_change_pct': None if control == 0 else 100 * (candidate / control - 1)}


def differences(control, candidate):
    assert control['rows'] == candidate['rows']
    d = {k: delta(control[k + '_mae'], candidate[k + '_mae']) for k in ('overall', 'weighted')}
    for family in ('strata', 'forecast_quarters', 'cities'):
        assert set(control[family]) == set(candidate[family])
        d[family] = {}
        for key, item in control[family].items():
            other = candidate[family][key]
            assert item['rows'] == other['rows']
            d[family][key] = delta(item['mae'], other['mae'])
    return d


def check_differences(saved, calculated):
    for key in ('overall', 'weighted'):
        assert near(saved[key]['delta_TWD'], calculated[key]['delta_TWD'])
    for family in ('strata', 'forecast_quarters'):
        for key, item in calculated[family].items():
            a, b = saved[family][key]['delta_TWD'], item['delta_TWD']
            assert (a is None and b is None) or near(a, b)


def review():
    tick, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs, receipts = {}, {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.md')
        assert p.stat().st_size < 30_000_000, 'Only small JSON/source/text'
        h, key = sha(p), p.relative_to(ROOT).as_posix()
        assert expected is None or h == expected, name
        assert key not in inputs or inputs[key] == h
        inputs[key] = h
        return p

    def small(name, required_receipt=True, expected=None):
        p = capture(name, expected)
        found = [q for q in (p.with_suffix('.sha256'), Path(str(p) + '.sha256')) if q.exists()]
        assert found or not required_receipt, 'Missing actual receipt: ' + name
        receipts[name] = [q.relative_to(ROOT).as_posix() for q in found]
        for q in found:
            capture(q)
            words = q.read_text(encoding='utf8').strip().split()
            assert words and words[0] == sha(p), str(q)
        return json.loads(p.read_text(encoding='utf8'))

    for name, digest in PINS.items():
        capture(name, digest)
    r = small('outputs/round79/results.json')
    execution = small('outputs/round79/execution.json', required_receipt=False)
    plan = small('outputs/round79/plan.json', expected=r['plan_sha256'])
    old = small('outputs/round76/results.json', expected=PINS['outputs/round76/results.json'])
    diag = small('reports/round79_diagnostics.json')
    de = small('outputs/round79/diagnostics/execution.json', required_receipt=False)
    paired = small('outputs/round79/paired_vs_v20/results.json')
    pe_name = 'outputs/round79/paired_vs_v20_execution.json'
    pe = small(pe_name, required_receipt=False) if (ROOT / pe_name).exists() else None
    for record in (execution, de, pe):
        if record is not None:
            assert record['exit_code'] == 0 and record.get('failure') is None
    assert execution['source_sha256'] == PINS['src/round79_experiment.py']
    assert r['round'] == 79 and r['candidate'] == 'R79' and r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['checked_stages'] == diag['checked_stages'] == list(STAGES)
    assert plan['text_plan_sha256'] == r['text_plan_sha256'] == PINS['reports/round79_plan.md']
    assert plan['planned_new_fits'] == r['new_fits'] == 3 and r['new_NN_fits'] == 0
    assert not r['new_submission_generated'] and not r['fresh_blind_validation']
    assert r['protected_before'] == r['protected_after'] and len(r['protected_before']) == 23
    assert r['selector_sha256_before'] == r['selector_sha256_after']
    for name in ('src/round79_experiment.py', 'src/round79_recency_unit.py'):
        assert r['source_sha256'][name] == PINS[name]
    assert diag['candidate'] == 'R79' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R79' and paired['control'] == 'v20'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = {'round': 79, 'status': 'completed_JSON_only_review', 'primary_comparison': 'R79 minus v20',
        'secondary_controls': list(CONTROLS[1:]), 'is_decision_authority': False, 'adoption_decision': None,
        'new_fits_in_this_reporter': 0, 'model_or_array_loads': 0, 'new_scores': 0, 'new_bootstrap': False,
        'independent_vector_replay_performed': False, 'new_CSV': False, 'selector_modified': False,
        'source_sha256': sha(Path(__file__)), 'input_sha256': inputs, 'observed_receipts': receipts,
        'started_at_utc': started, 'checked_stages': list(STAGES), 'stages': {},
        'main_seconds': r['elapsed_seconds'], 'main_execution': execution, 'diagnostics_execution': de,
        'paired_execution': pe, 'diagnostics_seconds': diag['elapsed_seconds'], 'paired_seconds': paired['elapsed_seconds'],
        'paired_replicates': paired['replicates'], 'paired_seed': paired['seed'],
        'main_declared_new_CatBoost_fits': 3, 'main_declared_new_NN_fits': 0,
        'main_declared_protected_file_count': 23, 'main_declared_protected_files_equal': True,
        'main_declared_selector_unchanged': True}
    for s in STAGES:
        a, dg, ps = r['stages'][s], diag['stages'][s], paired['stages'][s]
        assert a['status'] == 'completed' and a['fit_rows'] == dg['fit_rows']
        assert a['query_rows'] == dg['query_rows'] == ps['rows']
        assert a['new_fits'] == 1 and a['new_NN_fits'] == 0
        assert a['reference_predict_calls'] == a['spatial_predict_calls'] == 2
        checks = ('controls_full_metrics_exact', 'v19_v20_v21_full_median_exact', 'R76_unit_spatial_replay_exact',
                  'R19_feature_state_unchanged', 'full_query_raw_TWD_reload_exact', 'date_weight_provenance_verified',
                  'query_labels_used_only_for_evaluation')
        assert all(a[k] is True for k in checks)
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        metrics = a['metrics']
        assert all(metrics[k] == dg['metrics'][k] for k in metrics)
        assert metrics['v20'] == old['stages'][s]['metrics']['R76']
        assert ps['prediction_sha256']['R79'] == a['prediction_sha256']
        ds = {k: differences(metrics[k], metrics['R79']) for k in metrics if k != 'R79'}
        for k in ds:
            check_differences(a['differences'][k], ds[k])
        control_layers = dict(old['stages'][s]['unit_branch_metrics']['candidate'])
        control_layers['native'] = a['original_R19_native_metrics']
        transfer = {}
        for layer in ('native', 'base', 'reference', 'spatial'):
            dm = differences(control_layers[layer], a['unit_branch_metrics'][layer])
            check_differences(a['native_difference_vs_R19'] if layer == 'native' else a['unit_branch_differences_vs_R76'][layer], dm)
            transfer[layer] = {'control': control_layers[layer], 'candidate': a['unit_branch_metrics'][layer], 'differences': dm}
        transfer['final'] = {'control': metrics['v20'], 'candidate': metrics['R79'], 'differences': ds['v20']}
        meta_name = f'outputs/round79/{s}/model/metadata.json'
        m = small(meta_name, expected=a['artifact_sha256'][meta_name])
        fit_name = f'outputs/round79/{s}/model/execution.json'
        fe = small(fit_name, required_receipt=False, expected=a['artifact_sha256'][fit_name])
        assert fe['exit_code'] == 0 and fe['failure'] is None and near(fe['elapsed_seconds'], a['fit_seconds'])
        assert m['round'] == 79 and m['stage'] == s and m['fit_rows'] == a['fit_rows'] and m['query_rows'] == a['query_rows']
        assert m['feature_count'] == 95 and m['categorical_feature_count'] == 23 and m['parameters'] == plan['parameters']
        assert m['original_frame_sha256'] == a['original_frame_sha256'] and m['prediction_units'] == 'TWD'
        for key in ('reload_all_query_rows_verified', 'reload_raw_exact', 'reload_TWD_exact', 'original_R19_feature_state_exact',
                    'date_weight_provenance_verified_in_driver', 'legacy_state_training_fields_are_provenance_only', 'weight_argument_passed_to_pool'):
            assert m[key] is True
        assert m['reload_rows'] == a['query_rows'] and near(m['fit_seconds'], a['fit_seconds'])
        for key in ('query_labels_used', 'weights_use_fit_target', 'weights_use_query', 'weights_normalized_to_mean_one',
                    'date_weights_recomputed', 'date_weights_renormalized', 'fit_uses_eval_set', 'early_stopping_used'):
            assert m[key] is False
        assert m['fit_date_weights_sha256'] == m['R63_preparation']['train_sample_weight_sha256']
        assert m['source_sha256']['src/round79_recency_unit.py'] == PINS['src/round79_recency_unit.py']
        assert m['weight_dtype'] == 'float64' and m['plan_sha256'] == r['plan_sha256']
        ci = {}
        for method in ('project', 'month'):
            ci[method] = {}
            for key in ('overall', 'weighted'):
                v = ps[method]['summary'][key]
                assert v['valid_replicates'] + v['invalid_replicates'] == paired['replicates']
                assert near(v['candidate_mae'], metrics['R79'][key + '_mae']) and near(v['control_mae'], metrics['v20'][key + '_mae'])
                assert near(v['point_delta'], ds['v20'][key]['delta_TWD'])
                ci[method][key] = v
        mix, bands = dg['test_composition'], []
        assert mix['minimum_historical_cell_rows'] == 200 and [b['band'] for b in dg['price_bands']] == list(BANDS)
        assert sum(b['rows'] for b in dg['price_bands']) == a['query_rows']
        for b in dg['price_bands']:
            bd = {k: delta(b[k], b['R79']) for k in CONTROLS}
            contribution = {k: 0.0 if b['rows'] == 0 else bd[k]['delta_TWD'] * b['rows'] / a['query_rows'] for k in CONTROLS}
            bands.append(dict(b, differences=bd, contribution_to_stage_overall_delta_TWD=contribution))
        high = {}
        for k in CONTROLS:
            assert near(sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands), ds[k]['overall']['delta_TWD'])
            value = sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands[-2:])
            high[k] = {'rows': sum(b['rows'] for b in bands[-2:]), 'overall_delta_contribution_TWD': value,
                       'other_bands_contribution_TWD': ds[k]['overall']['delta_TWD'] - value,
                       'signed_ratio_to_net_delta': None if ds[k]['overall']['delta_TWD'] == 0 else value / ds[k]['overall']['delta_TWD']}
        out['stages'][s] = {'fit_rows': a['fit_rows'], 'query_rows': a['query_rows'], 'dates': a['dates'],
            'original_frame_sha256': a['original_frame_sha256'], 'metrics': metrics, 'differences': ds,
            'unit_to_final': transfer, 'test_composition': mix,
            'test_composition_differences': {k: delta(mix['models'][k], mix['models']['R79']) for k in CONTROLS},
            'price_bands': bands, 'at_least_50m_contributions': high, 'paired': ci,
            'paired_support': {method: ps[method]['support'] for method in ('project', 'month')},
            'main_evidence_checks': {k: a[k] for k in checks}, 'fit_seconds': a['fit_seconds'],
            'stage_seconds': a['elapsed_seconds'], 'fit_execution': fe, 'model_metadata': m,
            'metadata_evidence_only_no_independent_array_check': True}
    out['total_fit_seconds'] = sum(a['fit_seconds'] for a in out['stages'].values())
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
               report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest(), human_report_is_decision_authority=False)
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items()), 'Small evidence changed'
    raw = (json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest() + '\n')
    with MD.open('x', encoding='utf8', newline='\n') as f: f.write(md)
    print(json.dumps({'status': out['status'], 'elapsed_seconds': out['elapsed_seconds'], 'review_sha256': sha(OUT), 'report_sha256': sha(MD)}, ensure_ascii=False))


def render(r):
    row = lambda xs: '| ' + ' | '.join(str(x) for x in xs) + ' |'
    num = lambda x, sign=False: '—' if x is None else (f'{x:+,.2f}' if sign else f'{x:,.2f}')
    lines = ['# R79：單價樹加入既存日期權重', '',
        '本輪只把原 R19 單價 CatBoost 的 area/100 權重乘上相同 fit 列序的 R63 保存日期權重；不重新計算或再次正規化日期權重，原95欄／23類別、參數、原R6地理、單價目標與TWD逆轉換保持。三期各一個新 CatBoost fit，沒有新 NN fit。舊 R19 feature state 的 area-only 權重欄僅是歷史 provenance，實際乘積權重記於本輪 metadata。',
        '新單價TWD元件→.75元件+.25原unit→原reference一次→R39 spatial一次，再與固定R71 NN42和R46 log取median。對照單價分支另外重現一次，所以主程序各期記錄reference／spatial各2次。主比較R79−v20；負值表示保存MAE較小。分支改善不等於最終median改善，逐層差額不是因果分解。', '',
        '## 完整結果與固定 test 組成', '', row(['期別', '方法', '整體MAE', '加權MAE', 'test組成MAE']), row(['---', '---', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]
        for k in ('R79', *CONTROLS):
            lines.append(row([NAMES[s], k, num(a['metrics'][k]['overall_mae']), num(a['metrics'][k]['weighted_mae']), num(a['test_composition']['models'][k])]))
    lines += ['', row(['期別', 'R79−對照', '整體差／%', '加權差／%', 'test組成差']), row(['---', '---', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]
        for k in CONTROLS:
            d = a['differences'][k]
            lines.append(row([NAMES[s], k] + [f"{num(d[x]['delta_TWD'], True)}／{d[x]['relative_change_pct']:+.4f}%" for x in ('overall', 'weighted')] + [num(a['test_composition_differences'][k]['delta_TWD'], True)]))
    for s in STAGES:
        mix = r['stages'][s]['test_composition']
        lines.append(f"\n{NAMES[s]}：city×seen/new、cell至少200列；涵蓋test質量 {mix['test_mass_covered']:.4%}，未支持 {mix['unsupported_test_rows']} 列。")
    lines += ['', 'test組成指標僅在有歷史支持的test質量內正規化，不是可知的test／Private真實MAE。', '',
        '## 原生單價到最外 median', '', row(['期別', '層', '對照MAE', 'R79 MAE', '整體差', '加權差']), row(['---', '---', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        for layer, a in r['stages'][s]['unit_to_final'].items():
            lines.append(row([NAMES[s], layer, num(a['control']['overall_mae']), num(a['candidate']['overall_mae'])] + [num(a['differences'][k]['delta_TWD'], True) for k in ('overall', 'weighted')]))
    lines += ['', 'native對照原R19；base/reference/spatial對照R76的新unit分支；final對照v20。這些是不同層的保存預測誤差，不能把層差相加當獨立改善或因果效果。', '',
        '## 新舊建案與四季', '', row(['期別', '對照', '新案差', '舊案差', 'Q1差', 'Q2差', 'Q3差', 'Q4差']), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        for k in CONTROLS:
            d = r['stages'][s]['differences'][k]
            lines.append(row([NAMES[s], k] + [num(d['strata'][g]['delta_TWD'], True) for g in ('new_nonmissing', 'seen')] + [num(d['forecast_quarters'][q]['delta_TWD'], True) for q in ('Q1', 'Q2', 'Q3', 'Q4')]))
    lines += ['', '## 城市與五個真價群', '', '城市列數≥500僅標示描述性支持，不是新門檻；其他列也保留。所有對照的完整城市／缺案名指標存於JSON。', '',
        row(['期別', '城市', '列數', '≥500', 'R79 MAE', '差vs v20']), row(['---', '---', '---:', '---', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]
        for city, m in a['metrics']['R79']['cities'].items():
            lines.append(row([NAMES[s], city, m['rows'], '是' if m['rows'] >= 500 else '否', num(m['mae']), num(a['differences']['v20']['cities'][city]['delta_TWD'], True)]))
    lines += ['', row(['期別', '真價帶', '列數', 'R79 MAE', *['差vs ' + k for k in CONTROLS]]), row(['---', '---'] + ['---:'] * 6)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s], b['band'], b['rows'], num(b['R79'])] + [num(b['differences'][k]['delta_TWD'], True) for k in CONTROLS]))
    lines += ['', '真價分帶只是事後診斷，推論時未知真價，不能據此分流或逐ID改價。以下把每群MAE差乘列數／全期列數，得到對全期整體MAE差的有號貢獻；不是加權MAE的分解。', '',
        row(['期別', '≥5000萬列數', '≥5000萬對整體差貢獻', '其他群貢獻', '全部差']), row(['---'] + ['---:'] * 4)]
    for s in STAGES:
        a = r['stages'][s]; h = a['at_least_50m_contributions']['v20']
        lines.append(row([NAMES[s], h['rows'], num(h['overall_delta_contribution_TWD'], True), num(h['other_bands_contribution_TWD'], True), num(a['differences']['v20']['overall']['delta_TWD'], True)]))
    lines += ['', '正貢獻表示拖累、負貢獻表示改善；群間可互相抵銷，不能把有號淨差比例當穩定收益占比。完整五群及各對照貢獻存於JSON。', '',
        '## R79−v20 的全部配對區間', '', f"固定seed {r['paired_seed']}、每項 {r['paired_replicates']} 次；負向抽樣比例不是Private勝率。", '',
        row(['期別', '抽法', '指標', '點差', '2.5%', '97.5%', '有效／無效', '位置']), row(['---', '---', '---', '---:', '---:', '---:', '---:', '---'])]
    for s in STAGES:
        for method in ('project', 'month'):
            for key in ('overall', 'weighted'):
                c = r['stages'][s]['paired'][method][key]
                where = '全負' if c['percentile_97_5'] < 0 else '全正' if c['percentile_2_5'] > 0 else '含／跨零'
                lines.append(row([NAMES[s], '建案' if method == 'project' else '3月循環區塊', key] + [num(c[k], True) for k in ('point_delta', 'percentile_2_5', 'percentile_97_5')] + [f"{c['valid_replicates']}／{c['invalid_replicates']}", where]))
    lines += ['', '## 權重、執行與解讀界線', '', row(['期別', 'fit列', '日期權重均值', '乘積權重均值', '乘積最大值', '日期集中度ESS／列數', '乘積集中度ESS／列數', 'fit秒', 'raw負值列']), row(['---'] + ['---:'] * 8)]
    for s in STAGES:
        a = r['stages'][s]; m = a['model_metadata']; d, w = m['fit_date_weights_summary'], m['fit_sample_weights_summary']
        lines.append(row([NAMES[s], a['fit_rows'], num(d['mean']), num(w['mean']), num(w['max']), f"{d['effective_sample_size']/a['fit_rows']:.2%}", f"{w['effective_sample_size']/a['fit_rows']:.2%}", f"{a['fit_seconds']:.3f}", m['native_negative_raw_rows']]))
    lines += ['', 'ESS僅描述權重集中程度，不是獨立建案數、有效驗證樣本量或信賴程度。乘積權重沒有再次均重正規化；raw負值沿既有逆轉換投影為0，不刪列。',
        f"主流程main {r['main_seconds']:.3f}秒，保存execution {r['main_execution']['elapsed_seconds']:.3f}秒；3個CatBoost fit合計 {r['total_fit_seconds']:.3f}秒；generic {r['diagnostics_seconds']:.3f}秒，paired {r['paired_seconds']:.3f}秒。以上已讀execution皆exit0；是否存在外層收據按JSON observed_receipts／paired_execution實錄，不補造。",
        '主程序記錄原生raw-unit及TWD全query保存／重載逐位相同、原R19純X state不變、canonical indices/frame及R63日期權重身分一致、query y僅評分。這些是主程序與metadata證據，本報告只核小JSON、收據及保存摘要算式，沒有独立載入陣列、樹／NN模型、重播預測、重算score或bootstrap。',
        '23份舊CSV／registry保護檔及selector前後相同。本輪沒有新CSV或推薦變更，本報告不作採用決定。不能因unit單支改善即採用；須同時看最外median、最新confirmation、新舊案／季度、test組成、價格尾端和區間。',
        '單seed42、同一批窗口已反覆用於選型，配對區間未校正多輪選型，也沒有新增盲測。大型物件的面積與多分支可能共同外推，median不能消除共有錯誤；不能把預測變動當真實誤差。Public不作本輪調參依據，Private未知、最終版本未定。程式、模型、依賴與執行紀錄保留；GitHub待定版後整理，尚未發布。', '',
        '[固定計畫](round79_plan.md) · [主程式](../src/round79_experiment.py) · [模型helper](../src/round79_recency_unit.py) · [主結果](../outputs/round79/results.json) · [主執行](../outputs/round79/execution.json)',
        '[分組診斷](round79_diagnostics.json) · [配對結果](../outputs/round79/paired_vs_v20/results.json) · [JSON彙整](../outputs/round79/results_review.json) · [報告程式](../src/round79_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
