"""R75 saved-JSON review/report only; no arrays, model imports, scoring or bootstrap."""
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
LAYERS = ('native_area', 'native_mean', 'base', 'reference', 'nn_spatial', 'predictions')
TRANSITIONS = tuple(f'{a}->{b}' for a in ('below', 'within', 'above') for b in ('below', 'within', 'above')) + ('fixed_equal_bounds',)
SIGNS = ('improved', 'unchanged', 'worsened')
BANDS = ('<10m', '10-20m', '20-50m', '50-100m', '>=100m')
OUT = ROOT / 'outputs/round75/results_review.json'
MD = ROOT / 'reports/round75_report.md'
OLD_SHA = {71: '8c9ddd281fb31df324b7aba4febd40c06cbee073ab45704b03b30176752ca833',
           74: '428f497383872490dfcc60051f1d25538ce69f845ec79f47c5017fcd50ddb717'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-5)


def validate_projection(p, scope_rows, stage_rows):
    assert p['rows'] == p['input_scope_rows'] == scope_rows and p['parent_stage_rows'] == stage_rows
    assert set(p['transition_cells']) == set(TRANSITIONS) and set(p['error_sign_groups']) == set(SIGNS)
    rules = p['rules']
    assert rules['within_includes_both_boundaries'] and rules['equal_bounds_separate'] and rules['empty_cells_retained']
    assert rules['sign_group_threshold'] == 0 and rules['contribution_denominator'] == 'all input rows'
    for key in ('final_projection_bitwise_exact', 'prediction_shift_nonexpansive_with_tolerance',
                'transition_rows_and_sums_reconstruct_overall', 'sign_rows_and_sums_reconstruct_overall'):
        assert p['checks'][key] is True, key
    for cell in (p['overall'], *p['transition_cells'].values(), *p['error_sign_groups'].values()):
        n = cell['rows']
        assert 0 <= n <= scope_rows
        assert near(cell['row_fraction_of_scope'], n / scope_rows) and near(cell['row_fraction_of_stage'], n / stage_rows)
        for layer in ('spatial', 'final'):
            v = cell[layer]
            assert near(v['contribution_to_overall_delta_TWD'], v['error_delta_sum_TWD'] / scope_rows)
            assert near(v['contribution_to_stage_delta_TWD'], v['error_delta_sum_TWD'] / stage_rows)
            if n:
                assert near(v['paired_delta_TWD'], v['error_delta_sum_TWD'] / n)
                assert near(v['paired_delta_TWD'], v['candidate_MAE_TWD'] - v['control_MAE_TWD'])
            else:
                assert all(v[k] is None for k in ('paired_delta_TWD', 'control_MAE_TWD', 'candidate_MAE_TWD'))
                assert v['error_delta_sum_TWD'] == v['contribution_to_stage_delta_TWD'] == 0
        assert near(cell['transfer_contribution_to_overall_TWD'], cell['transfer_error_sum_TWD'] / scope_rows)
        assert near(cell['transfer_contribution_to_stage_TWD'], cell['transfer_error_sum_TWD'] / stage_rows)
        if n:
            assert near(cell['transfer_effect_TWD'], cell['final']['paired_delta_TWD'] - cell['spatial']['paired_delta_TWD'])
        else:
            assert cell['transfer_effect_TWD'] is None and cell['transfer_error_sum_TWD'] == 0
    for partition in (p['transition_cells'], p['error_sign_groups']):
        assert sum(v['rows'] for v in partition.values()) == scope_rows
        for layer in ('spatial', 'final'):
            for key in ('error_delta_sum_TWD', 'contribution_to_overall_delta_TWD', 'contribution_to_stage_delta_TWD'):
                assert near(sum(v[layer][key] for v in partition.values()), p['overall'][layer][key])
        assert near(sum(v['transfer_contribution_to_stage_TWD'] for v in partition.values()), p['overall']['transfer_contribution_to_stage_TWD'])


def validate_scope(scope, stage_rows):
    n = scope['rows']
    assert set(scope['layers']) == set(LAYERS)
    for layer in LAYERS:
        v = scope['layers'][layer]
        assert v['rows'] == n and near(v['contribution_to_stage_delta_TWD'], v['paired_error_sum_TWD'] / stage_rows)
        if n:
            assert near(v['paired_delta_TWD'], v['paired_error_sum_TWD'] / n)
            assert near(v['paired_delta_TWD'], v['candidate_MAE_TWD'] - v['control_MAE_TWD'])
        else:
            assert v['paired_delta_TWD'] is None and v['control_MAE_TWD'] is None and v['candidate_MAE_TWD'] is None
            assert v['paired_error_sum_TWD'] == v['contribution_to_stage_delta_TWD'] == 0
    if n:
        validate_projection(scope['projection'], n, stage_rows)
        for a, b in zip(LAYERS[:-1], LAYERS[1:]):
            assert near(scope['adjacent_paired_delta_changes_TWD'][f'{a}->{b}'],
                        scope['layers'][b]['paired_delta_TWD'] - scope['layers'][a]['paired_delta_TWD'])
        for output, layer in (('spatial', 'nn_spatial'), ('final', 'predictions')):
            assert near(scope['projection']['overall'][output]['paired_delta_TWD'], scope['layers'][layer]['paired_delta_TWD'])
    else:
        assert scope['projection'] is None and all(v is None for v in scope['adjacent_paired_delta_changes_TWD'].values())


def review():
    tick = time.monotonic()
    assert not any(p.exists() for p in (OUT, OUT.with_suffix('.sha256'), MD)), 'Preserve prior evidence; no overwrite'
    inputs = {}

    def capture(name, expected=None):
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        assert path.suffix in ('.json', '.sha256', '.py', '.md'), 'Only small JSON/text evidence permitted'
        assert path.stat().st_size < 30_000_000, 'Unexpectedly large JSON/text evidence'
        digest = sha(path)
        assert expected is None or digest == expected, name
        key = path.relative_to(ROOT).as_posix()
        assert key not in inputs or inputs[key] == digest
        inputs[key] = digest
        return path

    def small(name, expected=None):
        path = capture(name, expected)
        receipt = capture(path.with_suffix('.sha256').relative_to(ROOT).as_posix())
        assert receipt.read_text(encoding='utf8').strip() == inputs[path.relative_to(ROOT).as_posix()]
        return json.loads(path.read_text(encoding='utf8'))

    r = small('outputs/round75/results.json')
    execution = small('outputs/round75/execution.json')
    plan = small('outputs/round75/plan.json', r['plan_sha256'])
    old = {k: small(f'outputs/round{k}/results.json', v) for k, v in OLD_SHA.items()}
    capture('reports/round75_plan.md', r['text_plan_sha256'])
    for name in ('src/round75_error_transfer_review.py', 'src/round75_projection_diagnostics.py'):
        capture(name, r['source_sha256'][name])
    assert execution['exit_code'] == 0 and execution['failure'] is None
    assert execution['source_sha256'] == r['source_sha256']['src/round75_error_transfer_review.py']
    assert r['round'] == plan['round'] == execution['round'] == 75
    assert r['status'] == 'completed_saved_vector_error_transfer_diagnostic' and r['diagnostic_only']
    assert r['checked_stages'] == plan['stage_order'] == list(STAGES)
    assert r['layers'] == plan['layers'] == list(LAYERS)
    assert r['text_plan_sha256'] == plan['text_plan_sha256'] and r['source_sha256'] == plan['source_sha256']
    for key in ('new_fits', 'model_loads', 'model_predict_calls', 'reference_predict_calls', 'spatial_predict_calls'):
        assert r[key] == 0, key
    assert r['new_CSV'] is False and r['new_candidate_created'] is False
    assert r['protected_before'] == r['protected_after'] == plan['protected_before'] and len(r['protected_before']) == 21
    assert r['selector_sha256_before'] == r['selector_sha256_after'] == plan['selector_sha256_before']
    assert not plan['threshold_seed_weight_or_new_candidate_search']
    out = {'round': 75, 'status': 'completed_JSON_only_review', 'diagnostic_only': True,
           'is_decision_authority': False, 'new_fits': 0, 'model_or_array_loads': 0, 'new_scores': 0,
           'new_bootstrap': False, 'new_CSV': False, 'independent_vector_replay_performed': False,
           'source_sha256': sha(Path(__file__)), 'input_sha256': inputs,
           'started_at_utc': datetime.now(timezone.utc).isoformat(), 'checked_stages': [], 'stages': {},
           'primary_comparison': r['primary_comparison'], 'layers': list(LAYERS),
           'main_seconds': r['elapsed_seconds'], 'execution': execution,
           'protected_files_declared_equal': 21, 'selector_declared_unchanged': True,
           'denominator_note': 'Helper overall uses input scope rows; contribution_to_stage uses the complete period query rows',
           'interpretation': 'Descriptive transfer of saved paired errors, not causal effects or an adoption decision'}
    for s in STAGES:
        a = r['stages'][s]
        b71, b74 = old[71]['stages'][s], old[74]['stages'][s]
        assert a['stage'] == s and a['query_rows'] == b71['query_rows'] == b74['query_rows']
        assert a['fit_rows'] == b71['fit_rows'] == b74['fit_rows']
        assert a['original_frame_sha256'] == b71['original_frame_sha256'] == b74['original_frame_sha256']
        for k in ('native_overall_and_complete_full_saved_metrics_exact', 'all_cohort_rows_and_error_contributions_reconciled',
                  'same_fixed_bounds_median_clip_identity_exact', 'native_mean_and_base_identity_exact',
                  'shared_reference_state_geo_identity_compared_from_records_not_loaded'):
            assert a[k] is True, (s, k)
        assert a['new_fits'] == a['model_loads'] == a['predict_calls'] == 0
        lm = a['layer_metrics']
        assert lm['predictions']['control'] == b71['metrics']['R71'] == b74['metrics']['v19']
        assert lm['predictions']['candidate'] == b74['metrics']['R74']
        assert lm['native_area']['control']['overall_mae'] == b74['native_MAE_diagnostic']['R54_area']
        assert lm['native_area']['candidate']['overall_mae'] == b74['native_MAE_diagnostic']['R74_area']
        validate_scope(a['full'], a['query_rows'])
        assert a['full']['rows'] == a['query_rows']
        assert set(a['cohorts']) == {'strata', 'forecast_quarters', 'price_bands'}
        assert set(a['cohorts']['price_bands']) == set(BANDS)
        for family, scopes in a['cohorts'].items():
            assert sum(v['rows'] for v in scopes.values()) == a['query_rows']
            for scope in scopes.values():
                validate_scope(scope, a['query_rows'])
            for layer in LAYERS:
                assert near(sum(v['layers'][layer]['contribution_to_stage_delta_TWD'] for v in scopes.values()),
                            a['full']['layers'][layer]['paired_delta_TWD'])
        for layer in LAYERS:
            v = lm[layer]
            for side in ('control', 'candidate'):
                assert v[side]['rows'] == a['query_rows']
                assert near(a['full']['layers'][layer][side + '_MAE_TWD'], v[side]['overall_mae'])
            for k in ('overall', 'weighted'):
                assert near(v['differences'][k]['delta_TWD'], v['candidate'][k + '_mae'] - v['control'][k + '_mae'])
        out['stages'][s] = a
        out['checked_stages'].append(s)
    md = render(out)
    out.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
               rendered_report_sha256=hashlib.sha256(md.encode('utf8')).hexdigest(), human_report_is_decision_authority=False)
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items()), 'Small evidence changed during review'
    data = (json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with OUT.open('xb') as f:
        f.write(data)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(hashlib.sha256(data).hexdigest() + '\n')
    with MD.open('x', encoding='utf8', newline='\n') as f:
        f.write(md)
    print(json.dumps({'status': out['status'], 'elapsed_seconds': out['elapsed_seconds'],
                      'review_sha256': sha(OUT), 'report_sha256': sha(MD), 'new_fits': 0, 'new_scores': 0}, ensure_ascii=False))


def render(r):
    def row(values):
        return '| ' + ' | '.join(str(v) for v in values) + ' |'
    def num(value, signed=False):
        return '—' if value is None else (f'{value:+,.2f}' if signed else f'{value:,.2f}')
    lines = ['# R75：面積 NN 日期加權的逐層誤差傳遞', '',
      '這是 R74 對 R71／v19 的保存向量診斷，沒有新模型或預測規則，也不作採用決定。差額固定為 R74−對照，負數表示該層 MAE 較小；原生 area 層對照為 R54，其餘為 R71 seed42。三期完整列出，不依結果選年度、權重或 seed。', '',
      '六層依序為 native_area → native_mean（新area與共享R63 prior各半）→ base（.75均值+.25oldunit）→ reference → nn_spatial → final中位數。沒有oldraw子槽；共享R46 log、R40 unit外分支不變。reference與spatial本輪均未重跑。', '',
      '## 六層整體與加權指標', '',
      row(['期別', '層', '對照 MAE', 'R74 MAE', '成對差額', '對照加權', 'R74加權', '加權差額']),
      row(['---', '---', '---:', '---:', '---:', '---:', '---:', '---:'])]
    for s in STAGES:
        a = r['stages'][s]
        for layer in LAYERS:
            v = a['layer_metrics'][layer]
            lines.append(row([NAMES[s], layer, num(v['control']['overall_mae']), num(v['candidate']['overall_mae']),
                num(a['full']['layers'][layer]['paired_delta_TWD'], True), num(v['control']['weighted_mae']),
                num(v['candidate']['weighted_mae']), num(v['differences']['weighted']['delta_TWD'], True)]))
    lines += ['', '成對差額以逐列 error(candidate)−error(control) 的保存總和為準；MAE相減的末位可能有加總順序誤差。加權沿原固定test-seen/new權重，並非本輪訓練權重。', '',
      '## 相鄰層差額的變化', '',
      '下表為後一層成對MAE差減前一層成對MAE差。負數表示差額向有利方向移動；只是同批保存結果的算式差，不能視為各層獨立因果效果。', '',
      row(['期別'] + [f'{a}→{b}' for a, b in zip(LAYERS[:-1], LAYERS[1:])]), row(['---'] + ['---:'] * 5)]
    for s in STAGES:
        lines.append(row([NAMES[s]] + [num(r['stages'][s]['full']['adjacent_paired_delta_changes_TWD'][f'{a}->{b}'], True)
                                      for a, b in zip(LAYERS[:-1], LAYERS[1:])]))
    lines += ['', '## 最外中位數／固定區間投影', '',
      'L=min(log,unit)、U=max(log,unit)，保存 final 恰等於 clip(nn_spatial,L,U)。這不是新加的裁價規則。transfer effect=final成對MAE差−spatial成對MAE差；裁切限制預測差幅度，並不保證縮小MAE差或保留改善方向。', '',
      row(['期別', 'spatial差', 'final差', 'transfer effect', 'spatial平均預測差絕對值', 'final平均預測差絕對值']), row(['---'] + ['---:'] * 5)]
    for s in STAGES:
        p = r['stages'][s]['full']['projection']['overall']
        lines.append(row([NAMES[s], num(p['spatial']['paired_delta_TWD'], True), num(p['final']['paired_delta_TWD'], True),
                          num(p['transfer_effect_TWD'], True), num(p['predshift']['spatial']['mean_absolute_TWD']), num(p['predshift']['final']['mean_absolute_TWD'])]))
    lines.append('')
    for s in STAGES:
        p = r['stages'][s]['full']['projection']['overall']
        before, after = p['spatial']['paired_delta_TWD'], p['final']['paired_delta_TWD']
        if before < 0 < after:
            description = '保存spatial層的改善在final層翻為退步'
        elif after < 0 < before:
            description = '保存spatial層的退步在final層翻為改善'
        elif before < 0 and after < 0:
            description = '兩層皆改善，final改善幅度' + ('縮小' if abs(after) < abs(before) else '增加')
        else:
            description = '方向及幅度依表中保存差額解讀，不假定投影必然改善'
        lines.append(f"{NAMES[s]}：{num(before, True)} → {num(after, True)}元，{description}；transfer effect {num(p['transfer_effect_TWD'], True)}元。")
    lines += ['', '以下每格的「差」分母為該格筆數；「全期貢獻」為該格成對誤差總和除完整年度query筆數。十格全期貢獻加總回到上表整體差。within含L/U兩端；L=U另列，不混入九格。空格保留0筆、均值為—、貢獻0。']
    for s in STAGES:
        lines += ['', f'### {NAMES[s]}：十格轉移', '', row(['對照→R74', '筆數', 'spatial差', 'final差', 'spatial全期貢獻', 'final全期貢獻', 'transfer全期貢獻']), row(['---'] + ['---:'] * 6)]
        p = r['stages'][s]['full']['projection']
        for key in TRANSITIONS:
            v = p['transition_cells'][key]
            lines.append(row([key, v['rows'], num(v['spatial']['paired_delta_TWD'], True), num(v['final']['paired_delta_TWD'], True),
                              num(v['spatial']['contribution_to_stage_delta_TWD'], True), num(v['final']['contribution_to_stage_delta_TWD'], True), num(v['transfer_contribution_to_stage_TWD'], True)]))
        lines += ['', row(['spatial誤差符號群', '筆數', 'spatial差', 'final差', 'spatial全期貢獻', 'final全期貢獻', 'transfer全期貢獻']), row(['---'] + ['---:'] * 6)]
        for key in SIGNS:
            v = p['error_sign_groups'][key]
            lines.append(row([key, v['rows'], num(v['spatial']['paired_delta_TWD'], True), num(v['final']['paired_delta_TWD'], True),
                              num(v['spatial']['contribution_to_stage_delta_TWD'], True), num(v['final']['contribution_to_stage_delta_TWD'], True), num(v['transfer_contribution_to_stage_TWD'], True)]))
    lines += ['', '符號群以 spatial 成對絕對誤差差額的0界定義（<0改善、=0不變、>0退步），不是按預測方向或final誤差挑群。', '',
      '## 新舊建案、四季與固定五個真價群', '',
      '下表完整列各期scope內六層成對MAE差（元），包含空組與不利結果；五個價位邊界為1,000萬／2,000萬／5,000萬／1億，下界含、上界不含。真價分群僅事後描述，不能作推論時可知規則。']
    for s in STAGES:
        lines += ['', f'### {NAMES[s]}：完整scope表', '', row(['scope', '筆數'] + list(LAYERS)), row(['---', '---:'] + ['---:'] * 6)]
        a = r['stages'][s]
        for family in ('strata', 'forecast_quarters', 'price_bands'):
            scopes = a['cohorts'][family]
            keys = BANDS if family == 'price_bands' else tuple(scopes)
            for key in keys:
                v = scopes[key]
                lines.append(row([f'{family}/{key}', v['rows']] + [num(v['layers'][layer]['paired_delta_TWD'], True) for layer in LAYERS]))
    lines += ['', 'JSON保留各scope的十格／符號群完整結果、六層control/candidate MAE與原scorer城市指標。**helper的 contribution_to_overall 分母是該次輸入scope筆數**；driver另存的 contribution_to_stage 分母才是整年度筆數。不能把不同scope的組內均值或helper的overall貢獻直接相加成全期效果。', '',
      '## 執行、核對與限制', '',
      f"主流程 exit0，main {r['main_seconds']:.3f} 秒、外層 {r['execution']['elapsed_seconds']:.3f} 秒；三期都是0 fit／model load／predict，reference與spatial重新執行次數均0。",
      '主流程核對R48 canonical indices與原frame、共享分支來源、R71 seed42及頂層final、native_mean/base/median/clip恆等式、原生MAE及最終完整指標；21份舊CSV／registry與selector前後相同。此報告程式只讀小JSON及文字來源／收據，核對保存斷言與摘要算式，不讀陣列、不重新評分或bootstrap，也不冒稱再次獨立重演逐列證據。',
      '單一seed42、事後分析與反覆重用的窗口限制不變；逐層差額與投影表不是因果分解，也不是Private勝率。沒有新gate、Public調參、逐ID修價或替代CSV，不由此自動改變R74未替換v19的決定。v19仍為有限本機優先候選；v18已回報Public最佳2,388,386.4613，Private未知、最終版本未選。完整程式與紀錄保留，GitHub待定版後整理，現在未發布。', '',
      '[固定計畫](round75_plan.md) · [主診斷程式](../src/round75_error_transfer_review.py) · [投影helper](../src/round75_projection_diagnostics.py)',
      '[主結果](../outputs/round75/results.json) · [主執行](../outputs/round75/execution.json) · [JSON彙整](../outputs/round75/results_review.json) · [報告程式](../src/round75_results_review.py)',
      '[R74](round74_report.md) · [R71／v19](round71_report.md)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    review()
