"""R74 saved-JSON review and report; no array/model loads or new scoring."""
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
OUT = ROOT / 'outputs/round74/results_review.json'
MD = ROOT / 'reports/round74_report.md'
DRIVER_SHA = 'f81cc1519e8d74c226afb40bb0bb4b834dd6dc3657c14bccdb4c6f0c70f34a81'
TRAINER_SHA = 'f760d8a264c5c84c9a612a55ba77be4c0a3bafac655b1f21d0de99b2951f7b4d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(m, key):
    if key in ('overall', 'weighted'):
        return m[key + '_mae']
    if key in ('new_nonmissing', 'seen'):
        return m['strata'][key]['mae']
    return m['forecast_quarters'][key]['mae']


def delta(control, candidate):
    return {'delta_TWD': candidate - control, 'relative_change_pct': 100 * (candidate - control) / control}


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-5)


def review():
    tick = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'No overwrite or retry'
    inputs = {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file()
        digest = sha(p)
        assert expected is None or digest == expected, name
        key = p.relative_to(ROOT).as_posix()
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return p

    def small(name, expected=None, receipt=True):
        p = capture(name, expected)
        assert p.suffix == '.json'
        if receipt:
            q = capture(p.with_suffix('.sha256').relative_to(ROOT).as_posix())
            assert q.read_text(encoding='utf8').strip() == inputs[p.relative_to(ROOT).as_posix()]
        return json.loads(p.read_text(encoding='utf8'))

    run = small('outputs/round74/execution.json')
    diagnostics_run = small('outputs/round74/diagnostics/execution.json')
    assert run['exit_code'] == diagnostics_run['exit_code'] == 0
    assert run['failure'] is None and diagnostics_run['failure'] is None
    r = small('outputs/round74/results.json')
    plan = small('outputs/round74/plan.json', r['machine_plan_sha256'])
    diag = small('reports/round74_diagnostics.json')
    paired = small('outputs/round74/paired_vs_v19/results.json')
    original = {n: small(f'outputs/round{n}/results.json') for n in (54, 63, 71)}
    capture('src/round74_nn_experiment.py', DRIVER_SHA)
    capture('src/round74_recency_weighted_area_train.py', TRAINER_SHA)
    capture('reports/round74_plan.md', r['plan_sha256'])
    assert run['source_sha256'] == r['source_sha256']['src/round74_nn_experiment.py'] == DRIVER_SHA
    assert r['source_sha256']['src/round74_recency_weighted_area_train.py'] == TRAINER_SHA
    assert all(x['checked_stages'] == list(STAGES) for x in (r, diag, *original.values()))
    assert r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['new_NN_fits'] == 3 and not r['new_submission_generated']
    assert r['protected_before'] == r['protected_after'] and len(r['protected_before']) == 21
    assert plan['formula'] == r['formula'] and plan['text_plan_sha256'] == r['plan_sha256']
    assert plan['stage_order'] == list(STAGES) and plan['three_periods_regardless_of_scores']
    assert plan['seed'] == 42 and plan['epochs'] == 30 and plan['batch_size'] == 2048
    assert plan['no_CSV_or_final_fit_in_this_run'] and plan['planned_new_NN_fits'] == 3
    assert not plan['saved_weights_recomputed'] and not plan['half_life_or_weight_search']
    assert diag['status'] == 'completed_saved_predictions_only' and diag['candidate'] == 'R74'
    assert diag['model_fit_load_predict_calls'] == 0 and not diag['new_CSV']
    assert paired['status'] == 'completed' and paired['candidate'] == 'R74' and paired['control'] == 'v19'
    assert paired['difference'] == 'candidate minus control'
    assert paired['replicates'] == 2000 and paired['seed'] == 20260925 and set(paired['stages']) == set(STAGES)
    result = {'round': 74, 'status': 'completed_JSON_only_review', 'candidate': 'R74', 'primary_control': 'v19=R71',
              'comparison_direction': 'R74 minus v19; native area comparison is R74_area minus R54_area',
              'started_at_utc': started, 'source_sha256': sha(Path(__file__)), 'input_sha256': inputs,
              'checked_stages': [], 'stages': {}, 'formula': r['formula'], 'is_decision_authority': False,
              'current_interpretation': 'valuable_research_direction_not_replacing_v19_no_CSV',
              'model_or_array_loads': 0, 'new_scores': 0, 'new_fits': 0, 'new_bootstrap': False, 'new_CSV': False,
              'R74_fit_count': 3, 'R74_fit_seconds': sum(r['stages'][s]['fit_seconds'] for s in STAGES),
              'R74_main_seconds': r['elapsed_seconds'], 'main_execution': run, 'diagnostics_execution': diagnostics_run,
              'diagnostics_seconds': diag['elapsed_seconds'], 'paired_seconds': paired['elapsed_seconds'],
              'preserved_files_declared_equal': 21, 'independent_binary_replay_performed': False}
    for s in STAGES:
        a = r['stages'][s]
        b54, b63, b71 = (original[n]['stages'][s] for n in (54, 63, 71))
        dg, ps = diag['stages'][s], paired['stages'][s]
        assert a['status'] == 'completed' and a['new_NN_fits'] == 1
        assert a['controls_full_metrics_exact'] and a['full_query_reload_exact'] and a['query_labels_used_only_by_scorer']
        assert a['reference_predict_calls'] == a['spatial_predict_calls'] == 1
        assert not a['old_NN_retrained'] and not a['encoder_prior_weights_refit']
        assert all(a['fit_rows'] == x['fit_rows'] and a['query_rows'] == x['query_rows'] for x in (b54, b63, b71, dg))
        assert a['metrics']['v19'] == b71['metrics']['R71']
        assert all(m == a['metrics'][name] for name, m in dg['metrics'].items()) and dg['all_full_saved_metrics_exact']
        prep = a['preparation']
        assert prep == small(f'outputs/round74/{s}/preparation.json', a['artifact_sha256'][f'outputs/round74/{s}/preparation.json'])
        assert prep['original_eight_arrays_bitwise_unchanged']
        assert not any(prep[k] for k in ('feature_encoder_refit', 'sample_weights_recomputed', 'sample_weights_renormalized', 'query_labels_used', 'query_dates_used_for_weights'))
        assert prep['R54_prepared_sha256'] == b54['artifact_sha256'][prep['R54_prepared_path']]
        assert prep['R63_prepared_sha256'] == b63['artifact_sha256'][prep['R63_prepared_path']]
        metadata = {}
        for n, stage in ((74, a), (54, b54), (63, b63)):
            path = f'outputs/round{n}/{s}/model/training.json'
            metadata[n] = small(path, stage['artifact_sha256'][path], receipt=False)
        m, m54, m63 = (metadata[n] for n in (74, 54, 63))
        assert m['status'] == 'completed' and m['source_sha256'] == TRAINER_SHA
        assert m['configuration'] == {**m54['configuration'], **plan['loss_configuration_overrides']}
        assert m['input_sha256'] == prep['prepared_sha256'] and m['fit_seconds'] == a['fit_seconds']
        assert m['numeric_count'] == a['numeric_count'] == m54['numeric_count'] == {'development': 599, 'confirmation': 599, 'auxiliary': 590}[s]
        assert m['fit_rows'] == a['fit_rows'] and m['query_rows'] == m['reload_rows'] == a['query_rows']
        assert m['saved_model_reload_exact'] and m['extra_sample_weights'] and m['prediction_units'] == 'TWD'
        assert not any(m[k] for k in ('query_labels_available', 'query_labels_used', 'sample_weights_recomputed_in_trainer', 'sample_weights_batch_renormalized', 'price_or_unit_clipping'))
        assert [x['epoch'] for x in m['epochs']] == list(range(1, 31))
        assert all(math.isfinite(x[k]) for x in m['epochs'] for k in ('training_mae_million_with_dropout', 'training_weighted_mae_million_with_dropout'))
        for k in ('train_area_factor_sha256', 'query_area_factor_sha256', 'initial_training_median_unit_1e4_twd_per_m2', 'initial_head_bias', 'area_formula', 'area_factor_formula', 'output_units', 'cardinalities', 'parameter_count'):
            assert m[k] == m54[k], (s, k)
        assert m['train_sample_weight_sha256'] == prep['train_sample_weight_sha256'] == m63['train_sample_weight_sha256']
        for k, value in m['train_sample_weight_summary'].items():
            assert math.isclose(value, m63['train_sample_weight_summary'][k], rel_tol=1e-12, abs_tol=1e-12)
        native = a['native_MAE_diagnostic']
        assert native['R54_area'] == b54['native_MAE_diagnostic']['R54']
        assert native['R74_area'] - native['R54_area'] == a['native_area_MAE_change_TWD']
        metrics = {k: a['metrics'][k] for k in ('R74', 'v19')}
        differences = {k: delta(measure(metrics['v19'], k), measure(metrics['R74'], k))
                       for k in ('overall', 'weighted', 'new_nonmissing', 'seen', 'Q1', 'Q2', 'Q3', 'Q4')}
        for k, v in differences.items():
            saved = a['differences']['v19'][k] if k in ('overall', 'weighted') else a['differences']['v19']['strata' if k in ('new_nonmissing', 'seen') else 'forecast_quarters'][k]
            assert v['delta_TWD'] == saved['delta_TWD'] and close(v['relative_change_pct'], saved['relative_change_pct'])
        assert ps['point_metrics_reproduced'] and ps['rows'] == a['query_rows']
        assert ps['prediction_sha256'] == {'R74': a['prediction_sha256'], 'v19': b71['prediction_sha256']}
        ci = {}
        for method in ('project', 'month'):
            ci[method] = {}
            for k in ('overall', 'weighted'):
                v = ps[method]['summary'][k]
                assert v['valid_replicates'] == 2000 and v['invalid_replicates'] == 0
                assert close(v['candidate_mae'], measure(metrics['R74'], k)) and close(v['control_mae'], measure(metrics['v19'], k))
                assert close(v['point_delta'], differences[k]['delta_TWD'])
                ci[method][k] = {n: v[n] for n in ('point_delta', 'percentile_2_5', 'percentile_97_5', 'valid_replicates', 'invalid_replicates')}
        mix = dg['test_composition']
        assert mix['minimum_historical_cell_rows'] == 200 and dg['extra_control_existing_test_mix_checks']['v19'] == 'exact'
        assert [x['band'] for x in dg['price_bands']] == ['<10m', '10-20m', '20-50m', '50-100m', '>=100m']
        bands = [{k: v[k] for k in ('band', 'rows', 'R74', 'v19')} | delta(v['v19'], v['R74']) for v in dg['price_bands']]
        w = m['train_sample_weight_summary']
        result['stages'][s] = {'dates': a['dates'], 'fit_rows': a['fit_rows'], 'query_rows': a['query_rows'],
            'numeric_count': a['numeric_count'], 'parameter_count': a['parameter_count'], 'fit_seconds': a['fit_seconds'],
            'native_MAE': native, 'native_difference': delta(native['R54_area'], native['R74_area']),
            'metrics': metrics, 'differences': differences, 'price_bands': bands,
            'test_composition': mix, 'test_composition_difference': delta(mix['models']['v19'], mix['models']['R74']),
            'paired': ci, 'weight_summary': w, 'weight_Kish_concentration_fraction': w['effective_sample_size'] / a['fit_rows'],
            'preparation': prep, 'full_query_reload_asserted_in_saved_metadata': True,
            'all_diagnostic_full_metrics_match_main_result': True}
        result['checked_stages'].append(s)
    md = render(result)
    result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
                  rendered_report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest(), human_report_is_decision_authority=False)
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items()), 'Small evidence changed during review'
    data = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with OUT.open('xb') as f:
        f.write(data)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(hashlib.sha256(data).hexdigest() + '\n')
    with MD.open('x', encoding='utf8', newline='\n') as f:
        f.write(md)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'review_sha256': sha(OUT), 'report_sha256': sha(MD), 'new_fits': 0, 'new_scores': 0}, ensure_ascii=False))


def render(r):
    row = lambda xs: '| ' + ' | '.join(str(x) for x in xs) + ' |'
    lines = ['# R74：面積 NN 加入固定日期權重', '',
      '**原生 area NN 三期皆改善，但完整候選在最新年度略退，目前未替換 v19，也未產出新 CSV。** 這是有價值的研究方向，不能把本輪未採用解讀為永久放棄日期權重。以下差額皆為候選減對照，負數較好；原生對照是 R54 area NN，完整對照是 v19／R71。', '',
      '唯一變更是重用 R63 已保存的 fit 日期權重：固定365日半衰期、fit平均正規化、無效日期中性1；不重算權重、不按batch重新正規化。R54 原八陣列及數值折線保持逐位一致，三期數值寬度599／599／590，另加第九權重陣列。seed42、30epochs、batch2048、原架構與未加權 median(y/area_factor) 初始化皆不變。',
      'NN輸出仍為每有效平方米的萬元單價；乘 A/100 得百萬元總價，loss為 mean(w*abs(unit*A/100−y_million))，推論再轉為TWD一次，沒有裁價。新area與保存R63 prior原生預測各半，base=.75*均值+.25*oldunit，reference／spatial各一次，再與原R46 log及R40 unit取逐列中位數；沒有oldraw子分支。query標籤只供離線評分，未進trainer或權重。', '',
      '## 原生模型與完整組合', '',
      row(['期別', 'R54原生MAE', 'R74原生MAE', '差額／變化']), row(['---', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]; d = a['native_difference']; n = a['native_MAE']
        lines.append(row([NAMES[s], f"{n['R54_area']:,.2f}", f"{n['R74_area']:,.2f}", f"{d['delta_TWD']:+,.2f}／{d['relative_change_pct']:+.4f}%"]))
    lines += ['', row(['期別', 'v19整體', 'R74整體', '整體差／%', 'v19加權', 'R74加權', '加權差／%']), row(['---', '---:', '---:', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]; m = a['metrics']; d = a['differences']
        lines.append(row([NAMES[s], f"{m['v19']['overall_mae']:,.2f}", f"{m['R74']['overall_mae']:,.2f}", f"{d['overall']['delta_TWD']:+,.2f}／{d['overall']['relative_change_pct']:+.4f}%", f"{m['v19']['weighted_mae']:,.2f}", f"{m['R74']['weighted_mae']:,.2f}", f"{d['weighted']['delta_TWD']:+,.2f}／{d['weighted']['relative_change_pct']:+.4f}%"]))
    lines += ['', '原生單支改善不保證非線性reference／spatial／中位數組合也改善。最新確認期整體增加926.26元（0.06375%），固定新舊案加權亦退；較早兩期完整組合改善。這是保存結果的層次差異，未據此判定唯一原因或另搜權重。', '',
      '## 新舊建案與全部季度', '', '表內為 R74−v19 的MAE差（元）；缺案名組三期皆0筆，沒有虛構其MAE。完整列數與城市指標保留於JSON。', '',
      row(['期別', '新案', '舊案', 'Q1', 'Q2', 'Q3', 'Q4']), row(['---', '---:', '---:', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        lines.append(row([NAMES[s]] + [f"{r['stages'][s]['differences'][k]['delta_TWD']:+,.2f}" for k in ('new_nonmissing', 'seen', 'Q1', 'Q2', 'Q3', 'Q4')]))
    lines += ['', '開發與輔助期新舊案、四季皆改善；最新期新案退步、舊案略好，Q1／Q4退步，Q2／Q3改善。', '',
      '## 有支持的test組成', '',
      '沿固定 city×seen/new 的 test-X 組成，歷史query cell至少200筆，僅在有支持的test質量內重新正規化；不是全部test或Private的預測MAE。', '',
      row(['期別', '涵蓋test質量', '未支持筆數', 'v19', 'R74', '差额']), row(['---', '---:', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]; x = a['test_composition']
        lines.append(row([NAMES[s], f"{x['test_mass_covered']:.6%}", x['unsupported_test_rows'], f"{x['models']['v19']:,.2f}", f"{x['models']['R74']:,.2f}", f"{a['test_composition_difference']['delta_TWD']:+,.2f}"]))
    lines += ['', '## 五個固定真價群', '',
      '真價分群只能作事後誤差描述，不能在推論時依未知答案改價；五群完整保留。', '',
      row(['期別', '真價', '筆數', 'v19 MAE', 'R74 MAE', '差額／%']), row(['---', '---', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:
            lines.append(row([NAMES[s], b['band'], b['rows'], f"{b['v19']:,.2f}", f"{b['R74']:,.2f}", f"{b['delta_TWD']:+,.2f}／{b['relative_change_pct']:+.4f}%"]))
    lines += ['', '最新期億元以上319筆退步約51萬元MAE，2,000萬–5,000萬元群亦略退；其他三群改善。開發5,000萬–1億元群退步，其餘四群改善；輔助五群皆改善。不能由平均分數說每種房屋都更準。', '',
      '## 配對區間', '', '固定seed20260925，每種抽法／指標／年度2,000次有效重抽樣、0無效。完整候選減v19，表列2.5%–97.5%區間。', '',
      row(['期別', '抽法', '指標', '點估計差', '2.5%', '97.5%']), row(['---', '---', '---', '---:', '---:', '---:'])]
    for s in STAGES:
        for family in ('project', 'month'):
            for key in ('overall', 'weighted'):
                c = r['stages'][s]['paired'][family][key]
                lines.append(row([NAMES[s], '建案' if family == 'project' else '3月循環區塊', key] + [f"{c[k]:+,.2f}" for k in ('point_delta', 'percentile_2_5', 'percentile_97_5')]))
    lines += ['', '輔助兩種抽法、兩項指標皆全負；開發只有月份抽法全負，建案區間跨零；最新確認期兩種抽法、兩項指標皆跨零。本輪只有固定seed42的一次三期比較，沒有跨seed穩定性證據。區間以保存模型及反覆使用的窗口為條件，未涵蓋多輪選型、重訓或未來市場移轉，不是Private勝率或p值。', '',
      '## 日期權重與執行', '', '以下Kish值=(sum w)^2/sum(w^2)，只描述權重集中度；不是獨立樣本數、獨立建案數或有效驗證資料量。float32保存均值僅近似1，未二次正規化。', '',
      row(['期別', 'fit列數', '權重min／max', '權重mean', 'Kish集中度／fit比例', 'fit秒']), row(['---', '---:', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]; w = a['weight_summary']
        lines.append(row([NAMES[s], a['fit_rows'], f"{w['min']:.9f}／{w['max']:.6f}", f"{w['mean']:.12f}", f"{w['effective_sample_size']:,.2f}／{a['weight_Kish_concentration_fraction']:.4%}", f"{a['fit_seconds']:.3f}"]))
    lines += ['', f"主流程exit0：3次新fit合計{r['R74_fit_seconds']:.3f}秒，main {r['R74_main_seconds']:.3f}秒、外層execution {r['main_execution']['elapsed_seconds']:.3f}秒；generic診斷execution {r['diagnostics_execution']['elapsed_seconds']:.3f}秒，配對程式{r['paired_seconds']:.3f}秒。",
      '三期全部完成，沒有重訓舊NN、重估encoder／prior／weights或使用query y選epoch。主程序記錄三期全query保存重載一致，20份舊CSV及registry共21檔前後不變。此報告只核小JSON／收據與原有斷言，不載模型或陣列、不重新評分／bootstrap，也沒有額外独立模型重播。',
      '尚無final fit或新CSV，未在此修改selector或平台勾選。當前保留v19的有限本機優先、v18的已知Public最佳；R74無Public／Private實測，Private未知、最終版未選。程式、模型與紀錄保留，待最終版本選定後再整理GitHub，現在未發布。', '',
      '[事前計畫](round74_plan.md) · [主程式](../src/round74_nn_experiment.py) · [area加權trainer](../src/round74_recency_weighted_area_train.py) · [CPU數學核對](round74_trainer_math_check.json)',
      '[主結果](../outputs/round74/results.json) · [主執行](../outputs/round74/execution.json) · [分組診斷](round74_diagnostics.json) · [配對結果](../outputs/round74/paired_vs_v19/results.json)',
      '[JSON彙整](../outputs/round74/results_review.json) · [本報告產生程式](../src/round74_results_review.py)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
