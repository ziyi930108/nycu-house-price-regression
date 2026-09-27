"""Independent R22 saved-report QA. No arrays, models, rescoring or gate calculation.

Run --check only after the finalized report and independent replay receipts exist.
This checks saved aggregate presentation and bounded source identities; the report
and final archiver retain responsibility for the complete large-file hash closure.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/round22'
REPORT = 'reports/round22_report.md'
RESULTS = 'reports/round22_results.json'
DEST = 'outputs/round22/audit/report_artifact_qa.json'
CONTROL = 'v3_control'
CANDIDATE = 'huber_catboost_geography_blend'
STAGES = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
GROUPS = ('seen', 'new_nonmissing', 'missing')
SOURCE_EXPECTED = {
    'src/round22_report.py': '954a23d5e312ce4c185897c13e3cd08e3a392b20f2005b63312425edb67e2ee5',
    'src/round22_experiment.py': '2171dee73ec8cb5036f746f1e8cdc859c3003bf5ea0d6d184d3a7cfce621ae81',
    'src/round22_huber_geo.py': '3fddeb2b9c0d191dfae31538060c5713dbb3728ff1910c1cfe2b06777f0f1808',
    'src/round22_verify_backtest.py': 'df4255a360a9e431ee39f0769276f462a534567c7fd3d91b2c84b795a0b6d061',
}
CAPACITY = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
    'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
    'actual_learning_rate', 'actual_loss_function', 'actual_eval_metric',
    'actual_leaf_estimation_method', 'actual_leaf_estimation_iterations', 'actual_boost_from_average')
OBJECTIVE = {'loss_function': 'Huber:delta=1', 'eval_metric': 'MAE',
    'leaf_estimation_method': 'Newton', 'leaf_estimation_iterations': 1, 'boost_from_average': False}
PROTECTED = ('outputs/submission_v2.csv', 'outputs/submission_v3.csv', 'runs/versions.json')
PROTECTED_CSV_SHA = {
    'outputs/submission_v2.csv': 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b',
    'outputs/submission_v3.csv': '52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3',
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def path(name):
    p = (ROOT / name).resolve()
    need(p.is_relative_to(ROOT) and str(name).replace('\\', '/') == p.relative_to(ROOT).as_posix(),
         'Unsafe/noncanonical path: ' + str(name))
    return p


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(name):
    p = path(name)
    need(p.is_file(), 'Missing file: ' + name)
    h = hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


class Evidence:
    def __init__(self):
        self.files = {}
        self.receipts = {}

    def small(self, name):
        p = path(name)
        need(p.is_file() and p.stat().st_size <= 4 * 1024 * 1024, 'Missing/oversize small input: ' + name)
        need(p.suffix in ('.json', '.md', '.sha256', '.py', '.log'), 'Disallowed content read: ' + name)
        data = p.read_bytes()
        record = {'bytes': len(data), 'sha256': digest(data)}
        need(name not in self.files or self.files[name] == record, 'Input changed: ' + name)
        self.files[name] = record
        return data

    def locked(self, name):
        data = self.small(name)
        receipt = str(Path(name).with_suffix('.sha256')).replace('\\', '/')
        receipt_data = self.small(receipt)
        need(receipt_data.decode('ascii').strip() == digest(data), 'Receipt differs: ' + name)
        self.receipts[name] = dict(self.files[name], receipt_path=receipt,
                                   receipt_sha256=digest(receipt_data))
        return json.loads(data.decode('utf-8-sig'))


def model_fields(source):
    """Extract only literal field inventory from the pinned report, never import it."""
    assignments = {}
    def value(node):
        if isinstance(node, ast.Name):
            return assignments[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        return ast.literal_eval(node)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('CAPACITY_FIELDS', 'MODEL_FIELDS'):
                assignments[name] = value(node.value)
    need(assignments['CAPACITY_FIELDS'] == CAPACITY, 'Report native field inventory changed')
    return assignments['MODEL_FIELDS']


def num(value):
    if value is None:
        return '—'
    need(type(value) in (int, float) and math.isfinite(value), 'Invalid saved display number')
    return f'{value:,.2f}'


def change(a, b):
    # Formatting saved aggregate differences is not price-error recomputation.
    return '—' if a in (None, 0) or b is None else f'{100 * (b - a) / a:+.4f}%'


def section(markdown, title):
    marker = '## ' + title + '分組與季度'
    need(markdown.count(marker) == 1, 'Missing/duplicate stage section: ' + title)
    return markdown.split(marker, 1)[1].split('\n## ', 1)[0]


def check(report_log):
    destination = path(DEST)
    receipt_destination = destination.with_suffix('.sha256')
    need(not destination.exists() and not receipt_destination.exists(), 'Refusing to overwrite QA')
    protected_before = {name: fingerprint(name) for name in PROTECTED}
    e = Evidence()
    checks = []
    def ok(name, condition):
        need(condition, name)
        checks.append(name)
    for name, expected in SOURCE_EXPECTED.items():
        ok('source_identity:' + name, digest(e.small(name)) == expected)
    fields = model_fields(e.small('src/round22_report.py').decode('utf-8-sig'))
    plan_path = OUT + '/candidate_plan.json'
    frozen_path = OUT + '/frozen_candidate.json'
    accepted_path = OUT + '/accepted_candidate.json'
    plan, frozen, accepted = [e.locked(name) for name in (plan_path, frozen_path, accepted_path)]
    report = e.locked(RESULTS)
    markdown = e.small(REPORT).decode('utf-8-sig')
    authority = {'plan_sha256': e.files[plan_path]['sha256'],
                 'frozen_candidate_sha256': e.files[frozen_path]['sha256'],
                 'accepted_candidate_sha256': e.files[accepted_path]['sha256']}
    ok('authority_identity', all(x['round'] == 22 and x['candidate'] == CANDIDATE
       for x in (plan, frozen, accepted, report)) and plan['control'] == CONTROL)
    ok('authority_lineage', frozen['plan_sha256'] == accepted['plan_sha256'] == authority['plan_sha256']
       and accepted['frozen_candidate_sha256'] == authority['frozen_candidate_sha256']
       and all(report[key] == val for key, val in authority.items()))
    ok('authority_source_maps', plan['source_sha256'] == frozen['source_sha256'] == accepted['source_sha256']
       and all(plan['source_sha256'].get(n) == SOURCE_EXPECTED[n]
               for n in ('src/round22_experiment.py', 'src/round22_huber_geo.py')))
    ok('protected_csv_identity', all(protected_before[n]['sha256'] == h for n, h in PROTECTED_CSV_SHA.items()))
    ok('registry_matches_price_plan', protected_before['runs/versions.json']['sha256'] == plan['registry_sha256_at_plan'])
    ok('report_decision_copied', all(report[k] == accepted[k]
       for k in ('accepted', 'stopping_reason', 'deployment_decision', 'new_submission_created', 'currently_recommended_submission')))
    ok('no_new_release_or_claimed_blindness', report['new_submission_created'] is False
       and report['currently_recommended_submission'] == 'outputs/submission_v3.csv'
       and report['historically_untouched'] is False and report['public_score_used'] is False
       and report['private_score_known'] is False
       and report['report_only_reads_saved_JSON_and_hashes_binary_inputs'] is True)
    ok('declared_stage_order', [s['stage'] for s in report['stages']] == list(STAGES))
    completed, saved_passes = [], {}
    metric_rows = gate_count = unexecuted_rows = 0
    for item in report['stages']:
        stage = item['stage']
        ok(stage + ':dates_counts_status', item['title'] == STAGES[stage]
           and item['dates'] == plan['splits'][stage]
           and item['fit_rows'] == plan['split_contract'][stage]['fit_rows']
           and item['query_rows'] == plan['split_contract'][stage]['query_rows']
           and item['status'] == accepted['stage_status'][stage])
        common = f"| {item['title']} | {'～'.join(item['dates'])} | {item['fit_rows']:,}／{item['query_rows']:,} |"
        if stage not in accepted['stage_result_sha256']:
            ok(stage + ':unexecuted_null', stage in accepted['unevaluated_stages']
               and item['status'] == 'not_executed_after_development_rejection'
               and all(item[k] is None for k in ('metrics', 'decision', 'model', 'independent_review'))
               and not path(OUT + '/models/' + stage).exists()
               and not path(OUT + '/evaluation/' + stage).exists())
            row = common + ' — | — | — | — | — | — | 開發期未通過，未執行 |'
            ok(stage + ':display_dashes_not_zeros', markdown.splitlines().count(row) == 1)
            unexecuted_rows += 1
            continue
        completed.append(stage)
        rp = OUT + '/evaluation/' + stage + '/results.json'
        mp = OUT + '/models/' + stage + '/metadata.json'
        pp = OUT + '/audit/' + stage + '_replay.json'
        result, meta, proof = [e.locked(n) for n in (rp, mp, pp)]
        ok(stage + ':result_identity_and_saved_protocol', result['round'] == meta['round'] == proof['round'] == 22
           and result['stage'] == meta['stage'] == proof['stage'] == stage
           and e.files[rp]['sha256'] == accepted['stage_result_sha256'][stage]
           and result['plan_sha256'] == meta['plan_sha256'] == proof['plan_sha256'] == authority['plan_sha256']
           and result['source_sha256'] == meta['source_sha256'] == plan['source_sha256']
           and all(result[k] == plan[k] for k in ('thresholds', 'formulas', 'target_group_weights'))
           and result['fit_rows'] == item['fit_rows'] and result['query_rows'] == item['query_rows'])
        ok(stage + ':all_saved_metrics_and_gates_exact', item['metrics'] == result['metrics'] == proof['independent_metrics']
           and item['decision'] == result['decision'] == proof['independent_decision'])
        ok(stage + ':independent_proof_identity', proof['status'] == 'passed'
           and proof['result_sha256'] == e.files[rp]['sha256']
           and proof['source_sha256'] == SOURCE_EXPECTED['src/round22_verify_backtest.py']
           and proof['frozen_candidate_sha256'] == authority['frozen_candidate_sha256']
           and proof['accepted_candidate_sha256'] == authority['accepted_candidate_sha256']
           and proof['saved_raw_z_bitwise_equal'] is True)
        ok(stage + ':model_summary_exact', item['model'] == {k: meta[k] for k in fields})
        expected_review = {'proof_path': pp, 'proof_sha256': e.files[pp]['sha256'],
            'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal'],
            'native_model_capacity': proof['native_model_capacity'],
            'removed_feature_names': proof['removed_feature_names'], 'added_feature_names': proof['added_feature_names'],
            'native_objective_contract': proof['native_objective_contract'],
            'implicit_native_policy_changes': proof['implicit_native_policy_changes'],
            'independent_fit_state_evidence': proof['independent_fit_state_evidence']}
        ok(stage + ':review_summary_exact', item['independent_review'] == expected_review)
        native = proof['native_model_capacity']
        ok(stage + ':native_twelve_exact', set(native) == set(CAPACITY)
           and native == {k: meta[k] for k in CAPACITY}
           and native['actual_trees'] == 2999 and native['actual_model_depth'] == 8
           and native['actual_grow_policy'] == 'SymmetricTree' and native['actual_boosting_type'] == 'Plain'
           and 0 < native['maximum_tree_leaf_count'] <= 256
           and all(native['actual_' + k] == v for k, v in OBJECTIVE.items()))
        ok(stage + ':raw_uniform_95_23_only_passed_loss_changes', meta['family'] == 'catboost'
           and meta['component'] == 'geography_huber_catboost' and meta['target_mode'] == 'raw'
           and meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23
           and meta['native_categorical_feature_count'] == 23 and meta['raw_z_units'] == 'million_TWD'
           and meta['sample_weight'] == 'uniform' and meta['weight_argument_passed_to_pool'] is False
           and meta['target_formula'] == 'y_TWD / 1e6' and meta['inverse_formula'] == 'max(z * 1e6, 0)'
           and meta['added_feature_names'] == meta['removed_feature_names'] == []
           and meta['changed_parameter_names'] == ['loss_function']
           and meta['parameters'] == dict(meta['baseline_parameters'], loss_function='Huber:delta=1')
           and meta['baseline_parameters']['loss_function'] == 'MAE'
           and meta['parameters'] == plan['parameters'])
        ok(stage + ':saved_learning_rates_in_fixed_tolerance', all(math.isfinite(meta[k])
           and math.isclose(meta[k], .065, rel_tol=1e-7, abs_tol=0.)
           for k in ('actual_learning_rate', 'fitted_actual_learning_rate')))
        a, b = item['metrics'][CONTROL], item['metrics'][CANDIDATE]
        ok(stage + ':saved_group_inventory', set(item['metrics']) == {CONTROL, CANDIDATE}
           and a['rows'] == b['rows'] == item['query_rows']
           and set(a['strata']) == set(b['strata']) == set(GROUPS)
           and set(a['forecast_quarters']) == set(b['forecast_quarters']) == {'Q1', 'Q2', 'Q3', 'Q4'})
        row = common + f" {num(a['overall_mae'])} | {num(b['overall_mae'])} | {change(a['overall_mae'], b['overall_mae'])} | {num(a['weighted_mae'])} | {num(b['weighted_mae'])} | {change(a['weighted_mae'], b['weighted_mae'])} | {'通過' if item['decision']['passed'] else '未通過'} |"
        ok(stage + ':overall_weighted_display', markdown.splitlines().count(row) == 1)
        metric_rows += 1
        body = section(markdown, item['title'])
        for family, keys in (('strata', GROUPS), ('forecast_quarters', ('Q1', 'Q2', 'Q3', 'Q4'))):
            for key in keys:
                x, y = a[family][key], b[family][key]
                row = f"| {key} | {x['rows']:,} | {num(x['mae'])} | {num(y['mae'])} | {change(x['mae'], y['mae'])} |"
                ok(stage + ':display_' + family + '_' + key, x['rows'] == y['rows'] and body.splitlines().count(row) == 1)
                metric_rows += 1
        saved_passes[stage] = item['decision']['passed']
        ok(stage + ':saved_gate_text', all(type(v) is bool and
           body.splitlines().count(f"- `{k}`：{'通過' if v else '未通過'}。") == 1
           for k, v in item['decision']['checks'].items()))
        gate_count += len(item['decision']['checks'])
        d = meta['raw_z_diagnostics']
        ok(stage + ':saved_raw_diagnostic_text', d == proof['raw_z_diagnostics']
           and f"原始 z<0 為 {d['negative_count']:,} 列，z=0 為 {d['zero_count']:,} 列" in body
           and f"重播全部 {item['query_rows']:,} 列" in body)
        ok(stage + ':native_text', f"learning_rate={native['actual_learning_rate']:.12g}" in body
           and f"訓練記憶體 learning_rate={meta['fitted_actual_learning_rate']:.12g}" in body
           and f"{native['actual_trees']:,} 棵樹、最大 {native['maximum_tree_leaf_count']} 葉" in body
           and all((k + '=' + (str(v).lower() if type(v) is bool else str(v))) in body for k, v in OBJECTIVE.items()))
        if stage == 'development':
            ok('development_freeze_result', frozen['result_sha256'] == e.files[rp]['sha256'])
    ok('completed_and_unexecuted_inventory', set(completed) == set(accepted['stage_result_sha256'])
       and accepted['unevaluated_stages'] == [s for s in STAGES if s not in completed])
    dev_pass = saved_passes['development']
    ok('freeze_follows_saved_development_decision', frozen['passed'] is dev_pass
       and frozen['later_stages_allowed'] is dev_pass
       and frozen['selected_candidate'] == (CANDIDATE if dev_pass else CONTROL))
    ok('finalized_branch_from_saved_booleans',
       (not dev_pass and completed == ['development'] and accepted['accepted'] is False
        and accepted['stopping_reason'] == 'development_price_gates_failed')
       or (dev_pass and completed == list(STAGES)
           and accepted['accepted'] is all(saved_passes.values())
           and accepted['stopping_reason'] == ('all_fixed_backtests_passed' if accepted['accepted'] else 'later_backtest_gates_failed')))
    ok('deployment_and_header_consistent', accepted['deployment_decision'] ==
       ('backtest_accepted_pending_final_release' if accepted['accepted'] else 'keep_v3')
       and (('三年度固定回測通過，尚待另行最終訓練與發布。' if accepted['accepted']
             else '本輪未採用，維持 submission_v3.csv。') in markdown))
    required_phrases = ('Huber:delta=1', '95', '23', '2,999', '0.065', '1,000,000',
        '不是對數或單價', '最小化者一般不等於 MAE 的條件中位數',
        '不能由名稱推定比 MAE 更抗大型離群值', '並非未接觸盲測', '不是統計顯著性',
        'v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知',
        '沒有查內政部或外部成交答案', '沒有上傳或發布新提交檔')
    ok('scope_and_limitations_text', all(s in markdown for s in required_phrases))
    visible = re.sub(r'```.*?```', '', markdown, flags=re.S)
    links = []
    for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', visible):
        target = target.strip().strip('<>')
        if re.match(r'^[a-z][a-z0-9+.-]*:', target, flags=re.I) or target.startswith('#'):
            continue
        resolved = (path(REPORT).parent / target.split('#', 1)[0]).resolve()
        need(resolved.is_relative_to(ROOT) and resolved.is_file(), 'Missing/unsafe Markdown link: ' + target)
        links.append({'target': target, 'resolved': resolved.relative_to(ROOT).as_posix()})
    ok('all_ten_local_links_exist', len(links) == 10)
    for document in ('reports/round22_plan.md', 'reports/round22_method_review.md', 'reports/round22_pipeline_review.md'):
        e.small(document)
    report_inputs = report['source_input_sha256']
    ok('bounded_read_sources_bound_to_generated_report', all(report_inputs.get(n) == info['sha256']
       for n, info in e.files.items() if n not in (REPORT, RESULTS, 'reports/round22_results.sha256')))
    ok('research_identity_map_is_saved_plan_subset', all(plan['dependency_sha256'].get(k) == v
       for k, v in plan['research_dependency_sha256'].items())
       and report['research_summary']['dependency_sha256'] == plan['research_dependency_sha256'])
    log_data = e.small(report_log)
    events = []
    for line in log_data.decode('utf-8-sig').splitlines():
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(record, dict) and record.get('status') == 'report_written':
            events.append(record)
    ok('one_matching_report_written_event', len(events) == 1
       and events[0]['accepted'] is report['accepted']
       and events[0]['report_sha256'] == e.files[REPORT]['sha256']
       and events[0]['results_sha256'] == e.files[RESULTS]['sha256'])
    ok('all_small_inputs_unchanged_during_QA', all(fingerprint(n) == info for n, info in e.files.items()))
    protected_after = {name: fingerprint(name) for name in PROTECTED}
    ok('protected_before_equals_after', protected_after == protected_before)
    script_name = Path(__file__).resolve().relative_to(ROOT).as_posix()
    payload = {'round': 22, 'status': 'passed', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_path': script_name, 'script_sha256': fingerprint(script_name)['sha256'],
        'checks': checks, 'check_count': len(checks), 'checked_stages': completed,
        'unexecuted_stages': accepted['unevaluated_stages'], 'candidate_accepted': report['accepted'],
        'checked_JSON_receipts': e.receipts, 'checked_small_files': e.files,
        'displayed_metric_rows_checked': metric_rows, 'saved_gate_count_checked': gate_count,
        'unexecuted_table_rows_checked': unexecuted_rows, 'local_links': links, 'local_link_count': len(links),
        'protected_before': protected_before, 'protected_after': protected_after, 'protected_unchanged': True,
        'report_sha256': e.files[REPORT]['sha256'], 'report_results_sha256': e.files[RESULTS]['sha256'],
        'report_source_sha256': SOURCE_EXPECTED['src/round22_report.py'],
        'report_log_path': report_log, 'report_log_sha256': digest(log_data), 'report_written_events': len(events),
        'arrays_or_models_loaded': False, 'labels_loaded': False, 'MAE_recomputed': False,
        'gates_recalculated': False, 'replay_reexecuted': False, 'full_dependency_hashes_rechecked': False,
        'percentage_formatting_from_saved_summary_only': True,
        'read_scope': 'Small saved JSON/receipts/Markdown/source/log; three protected artifacts byte-hashed only. No model or array contents, labels, or full large-file dependency rehash.',
        **authority}
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    need(not destination.exists() and not receipt_destination.exists(), 'Concurrent QA destination appeared')
    with destination.open('xb') as handle:
        handle.write(encoded)
    with receipt_destination.open('x', encoding='ascii', newline='\n') as handle:
        handle.write(digest(encoded) + '\n')
    print(json.dumps({'status': 'passed', 'path': DEST, 'sha256': digest(encoded),
                      'checks': len(checks), 'metric_rows': metric_rows, 'gates': gate_count,
                      'links': len(links)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.add_argument('--report-log', default='outputs/round22/logs/report.log')
    args = parser.parse_args()
    check(args.report_log)
