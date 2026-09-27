"""R82 JSON-only evidence report; no arrays, models, rescoring or decision."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
from round81_results_review import STAGES, NAMES, BANDS, sha, near, delta, metric_deltas

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ('R81', 'v22', 'v23')
OUT = ROOT / 'outputs/round82/results_review.json'
MD = ROOT / 'reports/round82_report.md'
PINS = {
    'src/round82_unit_anchor.py': '4fc3d1eaca38e83bca6ddeb74a4c52cfef339a2a2b93d3b813632f136fb56d75',
    'src/round81_results_review.py': '685572222f8b07d9a1f8d43e2ed339a970f7412dc41510868129dfdb40c9d875',
    'src/round82_diagnostics.ps1': '0ebe16c7bcb26c16e49fb7dc3ff0ceefbc98a51c21af8224cc0a969cb1909081',
    'src/candidate_diagnostics_v3.py': '49cf3a89e8a1d60defb8f49bc63ef0f8f88ded528af1e3687684d70c8a4ae128',
    'src/paired_candidate_bootstrap.py': '3761c1cb09abe4e26412bf3aea48273ce0a4bad10b5a408e23d8fae488903bb0',
    'reports/round82_plan.md': '0a442808919c9bd15de4b40b074d890bb4d6952a7ef597f780b9071b34a3b26a',
    'outputs/round81/results.json': '0b5dbf01b0d22290af534406a3cf8ce59f2f1200fc9270593838ef3f24f3012c',
    'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'outputs/round80/results.json': 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6',
}
CHECKS = ('R81_base_reference_spatial_median_full_vectors_exact', 'controls_full_metrics_exact',
    'reference_diagnostics_equal', 'fixed_spatial_diagnostics_equal_except_price_adjustment',
    'spatial_fallback_exact', 'inherited_native_model_reload_evidence_verified',
    'query_labels_used_only_for_evaluation')


def review():
    tick, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite'
    inputs, receipts = {}, {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.ps1', '.md')
        assert p.stat().st_size < 10_000_000, 'Small saved evidence only'
        digest, key = sha(p), p.relative_to(ROOT).as_posix()
        assert expected is None or digest == expected, name
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return p

    def small(name, expected=None):
        p = capture(name, expected or PINS.get(name))
        found = [q for q in (p.with_suffix('.sha256'), Path(str(p) + '.sha256')) if q.exists()]
        assert found, 'Receipt missing: ' + name
        receipts[name] = [q.relative_to(ROOT).as_posix() for q in found]
        for q in found:
            capture(q)
            assert q.read_text(encoding='utf8').strip().split()[0] == sha(p), str(q)
        return json.loads(p.read_text(encoding='utf8'))

    for name, digest in PINS.items():
        capture(name, digest)
    r = small('outputs/round82/results.json')
    plan = small('outputs/round82/plan.json', r['plan_sha256'])
    priors = {label: small(f'outputs/round{number}/results.json') for label, number in (('R81', 81), ('v22', 79), ('v23', 80))}
    diag = small('reports/round82_diagnostics.json')
    paired = small('outputs/round82/paired_vs_R81/results.json')
    executions = {key: small(path) for key, path in (
        ('main', 'outputs/round82/execution.json'), ('diagnostics', 'outputs/round82/diagnostics/execution.json'),
        ('paired', 'outputs/round82/paired_vs_R81_execution.json'))}
    assert all(e['exit_code'] == 0 and e.get('failure') is None and e['automatic_retry'] is False for e in executions.values())
    for key, source in (('main', 'src/round82_unit_anchor.py'), ('diagnostics', 'src/candidate_diagnostics_v3.py'), ('paired', 'src/paired_candidate_bootstrap.py')):
        assert executions[key]['source_sha256'] == PINS[source]
    assert executions['paired']['wrapper_sha256'] == PINS['src/round82_diagnostics.ps1']
    assert r['round'] == plan['round'] == 82 and r['candidate'] == plan['candidate'] == 'R82'
    assert r['status'] == 'three_periods_completed_no_adoption_decision' and r['primary_control'] == 'R81'
    assert r['checked_stages'] == diag['checked_stages'] == plan['stage_order'] == list(STAGES)
    assert all(p['checked_stages'] == list(STAGES) for p in priors.values())
    assert r['formula'] == plan['formula'] and r['text_plan_sha256'] == plan['text_plan_sha256'] == PINS['reports/round82_plan.md']
    assert r['source_sha256'] == plan['source_sha256'] and r['input_sha256'] == plan['input_sha256']
    assert r['source_sha256']['src/round82_unit_anchor.py'] == PINS['src/round82_unit_anchor.py']
    assert r['input_sha256']['outputs/round81/results.json'] == PINS['outputs/round81/results.json']
    assert r['control_reference_variant'] == plan['control_reference_variant'] == 'trend'
    assert r['candidate_reference_variant'] == plan['candidate_reference_variant'] == 'anchor'
    assert r['base_and_spatial_future_trend_unchanged'] is True
    counts = ('new_fits', 'new_NN_fits', 'new_CatBoost_fits', 'tree_NN_model_loads', 'model_predict_calls', 'new_reference_fits', 'new_spatial_state_builds')
    assert all(r[k] == plan[k] == 0 for k in counts)
    assert r['reference_predict_calls'] == r['spatial_predict_calls'] == 6
    assert all(r[k] is False for k in ('new_CSV', 'new_test_predictions', 'new_submission_generated', 'automatic_adoption', 'fresh_blind_validation', 'Public_feedback_used_for_tuning', 'seed_weight_threshold_search'))
    assert r['query_labels_used_only_for_evaluation'] is True
    assert r['protected_before'] == r['protected_after'] and r['selector_sha256_before'] == r['selector_sha256_after']
    assert diag['round'] == 82 and diag['candidate'] == 'R82' and diag['status'] == 'completed_saved_predictions_only'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV'] and not diag['new_bootstrap']
    assert diag['input_sha256']['outputs/round82/results.json'] == inputs['outputs/round82/results.json']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R82' and paired['control'] == 'R81'
    assert paired['difference'] == 'candidate minus control' and paired['replicates'] == 2000 and paired['seed'] == 20260925
    assert set(paired['stages']) == set(STAGES) and paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    out = dict(round=82, status='completed_JSON_only_review', primary_comparison='R82 minus R81',
        background_comparisons=['R82 minus v22=R79', 'R82 minus v23=R80'], adoption_decision=None,
        is_decision_authority=False, new_fits_in_reporter=0, model_or_array_loads=0, new_scores=0,
        new_bootstrap=False, independent_vector_replay_performed=False, new_CSV=False, selector_modified=False,
        source_sha256=sha(Path(__file__)), input_sha256=inputs, observed_receipts=receipts,
        started_at_utc=started, executions=executions, checked_stages=list(STAGES), stages={},
        main_seconds=r['elapsed_seconds'], diagnostics_seconds=diag['elapsed_seconds'], paired_seconds=paired['elapsed_seconds'],
        main_declared_counts={k: r[k] for k in (*counts, 'reference_predict_calls', 'spatial_predict_calls')},
        main_declared_protected_file_count=len(r['protected_before']), main_declared_protected_files_equal=True,
        main_declared_selector_unchanged=True, main_evidence_only_no_independent_array_check=True)
    for s in STAGES:
        a, dg, ps = r['stages'][s], diag['stages'][s], paired['stages'][s]
        before = priors['R81']['stages'][s]
        assert a['status'] == 'completed' and a['fit_rows'] == dg['fit_rows'] == before['fit_rows']
        assert a['query_rows'] == dg['query_rows'] == ps['rows'] == before['query_rows']
        assert all(a[k] is True for k in CHECKS) and a['independent_tree_NN_model_reload_performed'] is False
        assert all(a[k] == 0 for k in ('new_fits', 'new_NN_fits', 'tree_NN_model_loads', 'model_predict_calls'))
        assert a['reference_predict_calls'] == a['spatial_predict_calls'] == 2
        assert before['full_query_raw_TWD_reload_exact'] is True  # Inherited R81 evidence, not a new reload.
        assert dg['all_full_saved_metrics_exact'] and ps['point_metrics_reproduced']
        assert all(a['metrics'][k] == v for k, v in dg['metrics'].items())
        for label, key in (('R81', 'R81'), ('v22', 'R79'), ('v23', 'R80')):
            assert a['metrics'][label] == priors[label]['stages'][s]['metrics'][key]
            assert dg['extra_control_existing_test_mix_checks'][label] == 'exact'
        assert a['original_frame_sha256'] == before['original_frame_sha256']
        layers = a['unit_branch_metrics']
        assert layers['control'] == before['unit_branch_metrics']['candidate']
        assert layers['control']['base'] == layers['candidate']['base']
        assert ps['prediction_sha256']['R82'] == a['prediction_sha256'] and ps['prediction_sha256']['R81'] == before['prediction_sha256']
        differences = {k: metric_deltas(a['metrics'][k], a['metrics']['R82']) for k in CONTROLS}
        layer_differences = {k: metric_deltas(layers['control'][k], layers['candidate'][k]) for k in ('base', 'reference', 'spatial')}
        for k in CONTROLS:
            for m in ('overall', 'weighted'):
                assert near(differences[k][m]['delta_TWD'], a['differences'][k][m]['delta_TWD'])
        for k in layer_differences:
            for m in ('overall', 'weighted'):
                assert near(layer_differences[k][m]['delta_TWD'], a['unit_branch_differences'][k][m]['delta_TWD'])
        ci = {m: ps[m]['summary'] for m in ('project', 'month')}
        for method in ci:
            for metric in ('overall', 'weighted'):
                v = ci[method][metric]
                assert v['valid_replicates'] + v['invalid_replicates'] == 2000
                assert near(v['candidate_mae'], a['metrics']['R82'][metric + '_mae']) and near(v['control_mae'], a['metrics']['R81'][metric + '_mae'])
                assert near(v['point_delta'], differences['R81'][metric]['delta_TWD'])
        mix, bands = dg['test_composition'], []
        assert mix['minimum_historical_cell_rows'] == 200 and [b['band'] for b in dg['price_bands']] == list(BANDS)
        assert sum(b['rows'] for b in dg['price_bands']) == a['query_rows']
        for b in dg['price_bands']:
            d = {k: delta(b[k], b['R82']) for k in CONTROLS}
            contribution = {k: 0.0 if b['rows'] == 0 else d[k]['delta_TWD'] * b['rows'] / a['query_rows'] for k in CONTROLS}
            bands.append(dict(b, differences=d, contribution_to_stage_overall_delta_TWD=contribution))
        tails = {}
        for threshold, selected in (('at_least_50m', bands[-2:]), ('at_least_100m', bands[-1:])):
            tails[threshold] = {}
            for k in CONTROLS:
                total = differences[k]['overall']['delta_TWD']
                assert near(sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in bands), total)
                tail = sum(b['contribution_to_stage_overall_delta_TWD'][k] for b in selected)
                tails[threshold][k] = dict(rows=sum(b['rows'] for b in selected), overall_delta_contribution_TWD=tail,
                    other_bands_contribution_TWD=total-tail, signed_ratio_to_net_delta=None if total == 0 else tail/total)
        out['stages'][s] = dict(fit_rows=a['fit_rows'], query_rows=a['query_rows'], metrics=a['metrics'],
            differences=differences, unit_branch_metrics=layers, unit_branch_differences=layer_differences,
            test_composition=mix, test_composition_differences={k: delta(mix['models'][k], mix['models']['R82']) for k in CONTROLS},
            price_bands=bands, tail_contributions=tails, paired=ci, paired_support={m: ps[m]['support'] for m in ci},
            main_evidence_checks={k: a[k] for k in CHECKS}, changed_final_rows=a['changed_final_rows'],
            stage_seconds=a['elapsed_seconds'], inherited_R81_full_query_raw_TWD_reload_exact=True,
            new_model_reload_performed=False, reference_beta=a['reference_beta'],
            reference_and_spatial_anchor_day_index=a['reference_and_spatial_anchor_day_index'])
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-tick,
        report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest())
    assert all(sha(ROOT/n) == h for n, h in inputs.items()), 'Small evidence changed'
    raw = (json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest()+'\n')
    with MD.open('x', encoding='utf8', newline='\n') as f: f.write(md)
    print(json.dumps(dict(status=out['status'], elapsed_seconds=out['elapsed_seconds'], review_sha256=sha(OUT), report_sha256=sha(MD)), ensure_ascii=False))


def render(r):
    row = lambda xs: '| ' + ' | '.join(str(x) for x in xs) + ' |'
    num = lambda x, sign=False: '—' if x is None else (f'{x:+,.2f}' if sign else f'{x:,.2f}')
    lines = ['# R82：只將單價分支的 reference trend 改為 anchor', '',
        '主比較R82−R81；v22與v23是既存背景對照。完全重用R81的unit base＝.75×R79 native TWD＋.25×R81日期加權LightGBM TWD，只將該分支reference variant由trend改為anchor。R39 spatial、R71 seed42 NN、R46 log與外層median不變。',
        '只移除reference base的額外全域未來趨勢；local reference日期調整、R8保存beta／Bfuture仍在。不是整個系統移除時間校正，也不因日期權重與trend並存就認定重複計算。R2已試過anchor但當时組合不同，本輪不是全新方法。',
        '以下MAE差皆為R82−對照，負值較好；本報告只彙整保存證據，不自動採用、不產CSV或更新推薦。', '', '## 三期完整候選', '']
    for metric, title in (('overall', '整體MAE'), ('weighted', '固定加權MAE')):
        lines += [row(['期別', *[k+' '+title for k in (*CONTROLS, 'R82')], *['差vs'+k for k in CONTROLS]]), row(['---']+['---:']*7)]
        for s in STAGES:
            a=r['stages'][s]
            lines.append(row([NAMES[s], *[num(a['metrics'][k][metric+'_mae']) for k in (*CONTROLS, 'R82')], *[num(a['differences'][k][metric]['delta_TWD'], True) for k in CONTROLS]]))
        lines.append('')
    lines += [row(['期別', *[k+' 整體／加權變動' for k in CONTROLS]]), row(['---']+['---:']*3)]
    for s in STAGES:
        a=r['stages'][s]
        lines.append(row([NAMES[s], *[' / '.join(f"{a['differences'][k][m]['relative_change_pct']:+.4f}%" for m in ('overall','weighted')) for k in CONTROLS]]))
    lines += ['', '## 固定test組成', '', '原city×seen/new、歷史cell至少200列；僅在有支持的test質量內重正規化，不是Private MAE。', '',
        row(['期別','涵蓋率','未支持列',*CONTROLS,'R82',*['差vs'+k for k in CONTROLS]]), row(['---']+['---:']*9)]
    for s in STAGES:
        a=r['stages'][s]; m=a['test_composition']
        lines.append(row([NAMES[s], f"{m['test_mass_covered']:.4%}",m['unsupported_test_rows'], *[num(m['models'][k]) for k in (*CONTROLS,'R82')], *[num(a['test_composition_differences'][k]['delta_TWD'],True) for k in CONTROLS]]))
    lines += ['', '## 同一base的reference、spatial與median傳遞', '',
        row(['期別','層','R81 MAE','R82 MAE','整體差','加權差']), row(['---','---']+['---:']*4)]
    for s in STAGES:
        a=r['stages'][s]
        pairs=[(k,a['unit_branch_metrics']['control'][k],a['unit_branch_metrics']['candidate'][k]) for k in ('base','reference','spatial')]
        pairs.append(('final median',a['metrics']['R81'],a['metrics']['R82']))
        for layer,c,n in pairs:
            lines.append(row([NAMES[s],layer,num(c['overall_mae']),num(n['overall_mae']),num(n['overall_mae']-c['overall_mae'],True),num(n['weighted_mae']-c['weighted_mae'],True)]))
        lines.append(f"\n{NAMES[s]}最終median改變 {a['changed_final_rows']}／{a['query_rows']}列；完整期 {a['stage_seconds']:.3f}秒。\n")
    lines += ['單支改善不保證median改善；逐層MAE差不是可相加的獨立收益或因果分解。', '', '## 新舊建案、季度與城市', '',
        row(['期別','群','列數','R81 MAE','R82 MAE',*['差vs'+k for k in CONTROLS]]), row(['---','---']+['---:']*6)]
    for s in STAGES:
        a=r['stages'][s]
        for family in ('strata','forecast_quarters'):
            for key,c in a['metrics']['R81'][family].items():
                n=a['metrics']['R82'][family][key]
                lines.append(row([NAMES[s],key,c['rows'],num(c['mae']),num(n['mae']), *[num(a['differences'][k][family][key]['delta_TWD'],True) for k in CONTROLS]]))
    for s in STAGES:
        for k in CONTROLS:
            bad=[f"{city} {num(d['delta_TWD'],True)}元（n={d['rows']}）" for city,d in r['stages'][s]['differences'][k]['cities'].items() if d['delta_TWD'] is not None and d['delta_TWD']>0]
            lines.append(f"\n{NAMES[s]}對{k}退步城市："+('；'.join(bad) if bad else '無')+'。')
    lines += ['', '全部城市的改善與退步、每月與背景控制完整metrics保留於[分組JSON](round82_diagnostics.json)，沒有依城市或真價修改推論。', '', '## 固定五價帶與尾部貢獻', '',
        row(['期別','真價帶','列數',*CONTROLS,'R82',*['差vs'+k for k in CONTROLS]]), row(['---','---']+['---:']*8)]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s],b['band'],b['rows'], *[num(b[k]) for k in (*CONTROLS,'R82')], *[num(b['differences'][k]['delta_TWD'],True) for k in CONTROLS]]))
    lines += ['', row(['期別','尾部','列數',*['全期MAE差貢獻vs'+k for k in CONTROLS],'其他價帶貢獻vsR81']), row(['---','---']+['---:']*5)]
    for s in STAGES:
        for key,label in (('at_least_50m','≥5000萬'),('at_least_100m','≥1億')):
            d=r['stages'][s]['tail_contributions'][key]
            lines.append(row([NAMES[s],label,d['R81']['rows'], *[num(d[k]['overall_delta_contribution_TWD'],True) for k in CONTROLS],num(d['R81']['other_bands_contribution_TWD'],True)]))
    lines += ['', '有號貢獻＝該群MAE差×群列數／全期列數；正為拖累、負為改善，五帶加總等於整體MAE差。這是絕對誤差差值的有號貢獻，不是prediction−truth的平均signed error。≥1億包含於≥5000萬，兩尾群不能再相加；JSON另存各對照其餘價帶貢獻與有號淨差比。', '',
        '## R82−R81全部12個配對區間', '', '建案／3月循環區塊各2000次、seed20260925；此為保存預測的條件診斷，負向抽樣比例不是Private勝率。', '',
        row(['期別','抽法','指標','點差','2.5%','97.5%','有效／無效','位置']),row(['---','---','---']+['---:']*4+['---'])]
    for s in STAGES:
        for method in ('project','month'):
            for metric in ('overall','weighted'):
                c=r['stages'][s]['paired'][method][metric]
                where='全負' if c['percentile_97_5']<0 else '全正' if c['percentile_2_5']>0 else '含／跨零'
                lines.append(row([NAMES[s],method,metric, *[num(c[k],True) for k in ('point_delta','percentile_2_5','percentile_97_5')],f"{c['valid_replicates']}／{c['invalid_replicates']}",where]))
    lines += ['', '## 執行與解讀界線', '',
        f"本輪0新fit、0 tree/NN模型載入或predict、0新reference fit／spatial state build；對照及候選合計reference 6次、spatial 6次。main {r['main_seconds']:.3f}秒（外層 {r['executions']['main']['elapsed_seconds']:.3f}秒），generic {r['diagnostics_seconds']:.3f}秒（外層 {r['executions']['diagnostics']['elapsed_seconds']:.3f}秒），paired {r['paired_seconds']:.3f}秒（外層 {r['executions']['paired']['elapsed_seconds']:.3f}秒）。三份execution均exit0、沒有自動重試。",
        f"主程式記錄三期R81 base／trend reference／spatial／median完整回放與metrics一致、同base及所有reference diagnostics一致、spatial fallback exact，{r['main_declared_protected_file_count']}份CSV／registry和selector前後不變。原生模型全query raw/TWD重載是承接R81既有證據，本輪沒有再次模型重載。",
        '本報告僅讀來源、小JSON及收據，核完整保存指標／點估計／貢獻等摘要算式；未載arrays、逐列重播或重新score/bootstrap。控制回放與query-label僅評分的確認來自主流程證據，不能冒稱本報告又獨立執行了一遍。',
        '最強反對理由：舊R2的anchor曾較差，未來價格漂移仍可能需要trend；R81層間MAE差本身不能指出是哪個reference項造成退步。主比較只隔離此固定variant操作，仍是多輪重用窗口、單seed及共同分支下的結果，不能推論獨立因果或未來收益保證。',
        '不新增三期必須全贏門檻，也不只靠高價尾改善聲稱普遍收益；採用須另審跨期、新舊案、城市／價带與不確定性。本報告沒有採用／拒絕權威、沒有Public調參、Private仍未知，無CSV／上傳／selector變更。程式、舊模型與紀錄保留；GitHub待定版後整理，尚未發布。', '',
        '[固定計畫](round82_plan.md) · [主程式](../src/round82_unit_anchor.py) · [主結果](../outputs/round82/results.json) · [主執行](../outputs/round82/execution.json)',
        '[完整分組](round82_diagnostics.md) · [配對結果](../outputs/round82/paired_vs_R81/results.json) · [配對執行](../outputs/round82/paired_vs_R81_execution.json)',
        '[本報告JSON](../outputs/round82/results_review.json) · [可重現報告器](../src/round82_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
