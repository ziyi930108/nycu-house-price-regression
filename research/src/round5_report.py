"""Render round-five reports from receipt-verified JSON only.

This standard-library module never imports project modules, fits a model,
loads a dataset/prediction array, scores a component, or selects a candidate.
Only main() writes the two report files. Missing stages remain null.
"""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'round5'
REPORTS = ROOT / 'reports'
CONTROL = 'v2_control'
CANDIDATE = 'project_mask_blend'
COMPONENT = 'project_mask_catboost'
CANDIDATES = (CONTROL, CANDIDATE)
STAGES = ('development', 'confirmation', 'auxiliary')
GROUPS = ('seen', 'new_nonmissing', 'missing')
QUARTERS = ('Q1', 'Q2', 'Q3', 'Q4')
PUBLIC_V2_MAE = 2883250.4138
SPLITS = {'development': ['2022-08-11', '2023-08-10'],
          'confirmation': ['2023-08-11', '2024-08-10'],
          'auxiliary': ['2021-08-11', '2022-08-10']}
FORMULAS = {
    CONTROL: {'base': '0.75 * old_catboost + 0.25 * lightgbm_unit', 'reference': 'v2_trend_once'},
    CANDIDATE: {'base': '0.75 * (0.5 * old_catboost + 0.5 * project_mask_catboost) + 0.25 * lightgbm_unit', 'reference': 'v2_trend_once'},
}
REQUIRED_COMPONENTS = {CONTROL: [], CANDIDATE: [COMPONENT]}
THRESHOLDS = {
    'development_min_overall_improvement': .01, 'development_min_weighted_improvement': .01,
    'development_max_quarter_deterioration': .05, 'large_stratum_min_rows': 500,
    'max_large_stratum_deterioration': .05, 'confirmation_max_overall_deterioration': .0,
    'confirmation_max_weighted_deterioration': .0, 'confirmation_min_improved_quarters': 2,
    'auxiliary_max_overall_deterioration': .03, 'auxiliary_max_weighted_deterioration': .03,
    'auxiliary_quarters_diagnostic_only': True,
}
MASK_CONTRACT = {
    'format_version': 1, 'feature_count': 87, 'categorical_feature_count': 21,
    'mask_columns': ['project_name', 'address_raw', 'building_and_unit', 'building_block'],
    'mask_token': '__R5_PROJECT_MASK__', 'missing_token': '__MISSING__',
    'project_key': 'address_city|address_district|complete_project_name',
    'key_normalization': 'fill missing; convert to string; strip surrounding whitespace; normalize missing sentinels only',
    'hash_prefix': 'round5|42|',
    'hash_function': 'SHA256, first 8 bytes as unsigned big-endian integer',
    'mask_if_integer_less_than': 2**63, 'mask_probability_per_named_project': .5,
    'fit_named_project_mask': 'same deterministic mask for every row of the geographic project',
    'query_seen_project_mask': 'reuse fitted flag', 'query_unseen_named_project_mask': True,
    'fit_and_query_missing_project_mask': True, 'missing_name_is_not_a_physical_project': True,
    'fit_state_uses_y': False, 'fit_state_uses_query': False,
    'other_feature_columns_unchanged': True, 'reference_uses_original_features': True,
    'raw_target_divisor': 1000000.0, 'sample_weight': 'uniform',
    'iterations': 2999, 'random_seed': 42, 'threads': 4, 'task_type': 'GPU',
    'no_eval_set': True, 'no_early_stopping': True,
}
STAGE_LABELS = {'development': 'Development 開發年度', 'confirmation': 'Confirmation 確認年度',
                'auxiliary': 'Auxiliary 輔助年度'}
LABELS = {CONTROL: 'v2 基準', CANDIDATE: '整建案遮罩融合'}
CHECK_LABELS = {
    'overall_improves_at_least_1pct': '整體 MAE 至少改善 1%',
    'weighted_improves_at_least_1pct': '加權 MAE 至少改善 1%',
    'quarters_deteriorate_at_most_5pct': '每個預測季度惡化不超過 5%',
    'large_strata_deteriorate_at_most_5pct': '至少 500 筆的每組惡化不超過 5%',
    'overall_within_limit': '整體 MAE 在該階段上限內',
    'weighted_within_limit': '加權 MAE 在該階段上限內',
    'at_least_two_quarters_improve': '至少兩個預測季度改善',
}
DEVELOPMENT_CHECKS = {'overall_improves_at_least_1pct', 'weighted_improves_at_least_1pct',
                      'quarters_deteriorate_at_most_5pct', 'large_strata_deteriorate_at_most_5pct'}
LATER_CHECKS = {'overall_within_limit', 'weighted_within_limit', 'large_strata_deteriorate_at_most_5pct'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_locked(path, snapshots, optional=False):
    """Verify the JSON's adjacent receipt; track absence as well as read bytes."""
    path = Path(path)
    receipt = path.with_suffix('.sha256')
    if not path.exists():
        require(not receipt.exists(), 'Orphan receipt: ' + rel(receipt))
        if optional:
            snapshots[rel(path)] = None
            snapshots[rel(receipt)] = None
            return None
        raise RuntimeError('Missing required JSON: ' + rel(path))
    require(path.is_file() and receipt.is_file(), 'Missing JSON/receipt: ' + rel(path))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(receipt.read_text(encoding='ascii').strip() == digest, 'Receipt mismatch: ' + rel(path))
    value = json.loads(raw.decode('utf-8-sig'))
    require(isinstance(value, dict), 'Expected JSON object: ' + rel(path))
    snapshots[rel(path)] = digest
    snapshots[rel(receipt)] = sha256(receipt)
    return value


def nonnegative(value, context):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, 'Invalid numeric value: ' + context)


def validate_hash_mapping(mapping, context):
    require(isinstance(mapping, dict) and mapping, 'Missing hash mapping: ' + context)
    for path, digest in mapping.items():
        require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64
                and all(char in '0123456789abcdef' for char in digest), 'Invalid SHA record: ' + context)


def validate_weights(weights):
    require(set(weights) == set(GROUPS), 'Unexpected target weight groups')
    for group, weight in weights.items():
        nonnegative(weight, 'weight/' + group)
    require(math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10), 'Weights do not sum to one')
    require(weights['missing'] == 0, 'Expected zero test weight for missing project names')


def validate_plan(plan):
    require(plan['round'] == 5 and plan['control'] == CONTROL, 'Unexpected round/control')
    require(plan['candidate_formulas'] == FORMULAS and plan['required_components'] == REQUIRED_COMPONENTS,
            'The report supports only the single fixed round-five candidate and control')
    require(plan['thresholds'] == THRESHOLDS and plan['splits'] == SPLITS, 'Changed fixed gates/windows')
    require(set(plan['candidates']) == {CANDIDATE} and set(plan['components']) == {COMPONENT}, 'Changed candidate/component set')
    require(plan['candidates'][CANDIDATE]['components'] == [COMPONENT], 'Changed candidate components')
    component = plan['components'][COMPONENT]
    require(component['mask_rules'] == MASK_CONTRACT, 'Changed mask rules')
    require(component['target'] == 'y/1e6' and component['weight'] == 'uniform', 'Changed training target/weights')
    params = component['parameters']
    for key, value in {'iterations': 2999, 'random_seed': 42, 'thread_count': 4, 'depth': 8,
                       'learning_rate': .065, 'loss_function': 'MAE', 'task_type': 'GPU'}.items():
        require(params[key] == value, 'Changed fixed model parameter: ' + key)
    for key in ('eval_set', 'early_stopping', 'query_labels_for_features_or_training',
                'hyperparameter_sweep', 'seed_sweep', 'blend_weight_sweep'):
        require(plan['training'][key] is False, 'Unexpected training/search setting: ' + key)
    for key in ('new_development_components', 'new_development_candidates'):
        require(plan['training'][key] == 1, 'Unexpected experiment count: ' + key)
    require(plan['historically_untouched'] is False and plan['external_property_answers'] is False,
            'Unexpected historical/external-answer policy')
    validate_weights(plan['target_group_weights'])
    for key in ('fit_source_sha256', 'evaluation_source_sha256', 'data_sha256'):
        validate_hash_mapping(plan[key], 'plan/' + key)
    for path, digest in plan['fit_source_sha256'].items():
        require(plan['evaluation_source_sha256'].get(path) == digest, 'Fit/scoring source closure mismatch')
    require('src/round5_evaluate.py' in plan['evaluation_source_sha256'], 'Evaluator absent from frozen source closure')


def validate_metrics(metrics, expected_rows):
    for name, item in metrics.items():
        require(item['rows'] == expected_rows, 'Metric row count differs: ' + name)
        for key in ('overall_mae', 'weighted_mae'):
            nonnegative(item[key], name + '/' + key)
        for field, keys in (('strata', GROUPS), ('forecast_quarters', QUARTERS)):
            require(set(item[field]) == set(keys), 'Incomplete metric groups: ' + name + '/' + field)
            for key, part in item[field].items():
                require(isinstance(part['rows'], int) and not isinstance(part['rows'], bool)
                        and part['rows'] >= 0, 'Invalid group rows: ' + key)
                if part['rows'] == 0:
                    require(part['mae'] is None, 'Empty group MAE must remain null: ' + key)
                else:
                    nonnegative(part['mae'], name + '/' + field + '/' + key)
            require(sum(part['rows'] for part in item[field].values()) == expected_rows, 'Groups do not cover query rows')
            require({key: part['rows'] for key, part in item[field].items()}
                    == {key: part['rows'] for key, part in metrics[CONTROL][field].items()}, 'Candidate group rows differ')


def validate_decision(stage, decision):
    """Check saved boolean consistency only; never recalculate MAE or gate scores."""
    require(decision['selected_candidate'] in CANDIDATES, 'Unknown selected candidate')
    if stage == 'development':
        require(set(decision['candidate_checks']) == {CANDIDATE}, 'Unexpected development candidate checks')
        item = decision['candidate_checks'][CANDIDATE]
        checks = item['checks']
        require(set(checks) == DEVELOPMENT_CHECKS and all(isinstance(ok, bool) for ok in checks.values()), 'Invalid development gate flags')
        require(isinstance(item['eligible'], bool) and item['eligible'] == all(checks.values()), 'Inconsistent saved eligibility')
        expected_qualified = [CANDIDATE] if item['eligible'] else []
        require(decision['qualified_candidates'] == expected_qualified, 'Saved qualified list differs')
        require(decision['selected_candidate'] == (CANDIDATE if item['eligible'] else CONTROL), 'Saved selection differs from eligibility')
    else:
        checks = decision['checks']
        expected_keys = LATER_CHECKS | ({'at_least_two_quarters_improve'} if stage == 'confirmation' else set())
        require(set(checks) == expected_keys and all(isinstance(ok, bool) for ok in checks.values()), 'Invalid later-stage gate flags')
        require(decision['selected_candidate'] == CANDIDATE and isinstance(decision['passed'], bool)
                and decision['passed'] == all(checks.values()), 'Inconsistent later-stage decision')
        require(decision['candidate_reselection_performed'] is False, 'Later-stage reselection prohibited')
        improved = decision['improved_quarters']
        require(isinstance(improved, list) and len(improved) == len(set(improved))
                and set(improved).issubset(QUARTERS), 'Invalid improved quarter record')
        if stage == 'confirmation':
            require(checks['at_least_two_quarters_improve'] == (len(improved) >= 2), 'Inconsistent saved quarter decision')


def load_snapshot():
    snapshots = {}
    plan_path = OUT / 'candidate_plan.json'
    plan = read_locked(plan_path, snapshots)
    validate_plan(plan)
    plan_sha = snapshots[rel(plan_path)]
    frozen = read_locked(OUT / 'frozen_candidate.json', snapshots, optional=True)
    results = {}
    for stage in STAGES:
        value = read_locked(OUT / 'evaluation' / stage / 'results.json', snapshots, optional=True)
        results[stage] = value
        if value is None:
            continue
        require(value['format_version'] == 1 and value['stage'] == stage, 'Unexpected result schema/stage')
        require(value['plan_sha256'] == plan_sha, 'Result uses another plan: ' + stage)
        require(value['candidate_formulas'] == FORMULAS and value['thresholds'] == THRESHOLDS
                and value['project_mask_rules'] == MASK_CONTRACT, 'Result contract differs: ' + stage)
        require(value['evaluated_candidates'] == list(CANDIDATES) and set(value['metrics']) == set(CANDIDATES), 'Unexpected scored candidates')
        require(value['prediction_units'] == 'TWD' and value['rounded'] is False, 'Unexpected units/rounding')
        for flag in ('query_labels_used_for_prediction', 'component_scored_directly',
                     'historically_untouched', 'public_score_used_as_target'):
            require(value[flag] is False, 'Unexpected result policy: ' + flag)
        for flag in ('fit_X_state_reconstruction_verified', 'query_mask_state_unchanged',
                     'reference_uses_original_features', 'historical_confirmation_control_previously_evaluated'):
            require(value[flag] is True, 'Missing recorded evaluator verification: ' + flag)
        require(value['reference_trend_applications_per_candidate'] == 1, 'Trend must be applied exactly once')
        require(value['source_sha256'] == plan['evaluation_source_sha256'], 'Result source identity differs from plan')
        validate_hash_mapping(value['data_sha256'], 'result/data')
        for path, digest in plan['data_sha256'].items():
            require(value['data_sha256'].get(path) == digest, 'Result input identity differs from plan: ' + path)
        for field in ('input_artifact_sha256', 'output_artifact_sha256'):
            validate_hash_mapping(value[field], 'result/' + field)
        require(value['component_parameters'] == {COMPONENT: plan['components'][COMPONENT]['parameters']}, 'Result parameters differ')
        require(value['target_group_weights'] == plan['target_group_weights'], 'Result target weights differ')
        require(value['runtime_versions'] == plan['runtime_versions'], 'Result runtime differs')
        require(len(value['feature_columns']) == 87 and len(set(value['feature_columns'])) == 87, 'Unexpected feature schema')
        for split_stage in STAGES:
            split, planned = value['splits'][split_stage], plan['split_contract'][split_stage]
            require(split['dates'] == SPLITS[split_stage] and split['fit_rows'] == planned['fit_rows']
                    and split['query_rows'] == planned['query_rows'], 'Result split contract differs')
        validate_metrics(value['metrics'], value['splits'][stage]['query_rows'])
        validate_decision(stage, value['decision'])
        if stage == 'development':
            require(value['frozen_candidate_sha256'] is None, 'Development unexpectedly depends on a later freeze')
        else:
            require(frozen is not None and frozen['selected_candidate'] == CANDIDATE, 'Unauthorized later stage')
            require(value['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Later stage uses another freeze')
            require(value['decision']['selected_candidate'] == frozen['selected_candidate'], 'Later-stage candidate changed')
    development = results['development']
    if frozen is not None:
        require(development is not None and frozen['format_version'] == 1, 'Freeze lacks development evidence')
        require(frozen['plan_sha256'] == plan_sha and frozen['selected_candidate'] == development['decision']['selected_candidate'], 'Freeze decision/plan differs')
        require(frozen['required_components'] == REQUIRED_COMPONENTS[frozen['selected_candidate']], 'Frozen required components differ')
        require(frozen['candidate_formulas'] == FORMULAS and frozen['thresholds'] == THRESHOLDS
                and frozen['project_mask_rules'] == MASK_CONTRACT, 'Frozen formula/gates/mask differ')
        for field in ('source_sha256', 'data_sha256', 'component_parameters', 'target_group_weights',
                      'splits', 'feature_columns', 'runtime_versions'):
            require(frozen[field] == development[field], 'Frozen development contract differs: ' + field)
        require(frozen['later_stage_new_fitting_artifacts_absent_at_freeze'] is True
                and frozen['historically_untouched_confirmation'] is False, 'Unexpected freeze policy')
        evidence = frozen['development_artifact_sha256']
        validate_hash_mapping(evidence, 'freeze/development artifacts')
        expected = {**development['input_artifact_sha256'], **development['output_artifact_sha256']}
        for suffix in ('.json', '.sha256'):
            path = rel(OUT / 'evaluation' / 'development' / ('results' + suffix))
            expected[path] = snapshots[path]
        for path, digest in expected.items():
            require(evidence.get(path) == digest, 'Freeze artifact record differs: ' + path)
        for stage in ('confirmation', 'auxiliary'):
            if results[stage] is not None:
                for field in ('data_sha256', 'splits', 'feature_columns', 'target_group_weights'):
                    require(results[stage][field] == frozen[field], 'Later-stage frozen contract differs: ' + field)
    acceptance = read_locked(OUT / 'accepted_candidate.json', snapshots, optional=True)
    if acceptance is not None:
        require(frozen is not None and acceptance['plan_sha256'] == plan_sha, 'Acceptance lacks matching freeze/plan')
        require(acceptance['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Acceptance uses another freeze')
        require(acceptance['selected_candidate'] == frozen['selected_candidate'], 'Acceptance selected another candidate')
        require(isinstance(acceptance['accepted'], bool) and acceptance['candidate_reselection_performed'] is False
                and acceptance['historically_untouched_confirmation'] is False, 'Invalid acceptance flags')
        if frozen['selected_candidate'] == CONTROL:
            require(acceptance['accepted'] is False and acceptance['stage_decisions'] == {}
                    and all(results[stage] is None for stage in ('confirmation', 'auxiliary')), 'Rejected development cannot have later-stage evaluation')
        else:
            require(set(acceptance['stage_decisions']) == {'confirmation', 'auxiliary'}, 'Missing finalized veto decisions')
            for stage in ('confirmation', 'auxiliary'):
                require(results[stage] is not None and acceptance['stage_decisions'][stage] == results[stage]['decision'], 'Acceptance stage differs')
            require(acceptance['accepted'] == all(results[stage]['decision']['passed'] for stage in ('confirmation', 'auxiliary')), 'Acceptance disagrees with saved veto decisions')
        require(acceptance['deployment_decision'] == (CANDIDATE if acceptance['accepted'] else 'retain_v2'), 'Unexpected deployment decision')
        validate_hash_mapping(acceptance['evidence_sha256'], 'acceptance/evidence')
        for path, digest in snapshots.items():
            if digest is not None and (path.startswith('outputs/round5/evaluation/') or path.startswith('outputs/round5/frozen_candidate.')):
                require(acceptance['evidence_sha256'].get(path) == digest, 'Acceptance evidence receipt differs: ' + path)
    return plan, frozen, results, acceptance, snapshots


def delta_percent(value, base):
    return None if value is None or base is None or base == 0 else 100.0 * (value / base - 1.0)


def amount(value):
    return '—（無樣本）' if value is None else f'{value:,.2f}'


def percent(value):
    return '—' if value is None else f'{value:+.3f}%'


def candidate_label(name):
    return LABELS[name] + ' (`' + name + '`)'


def gate_text(checks):
    failed = [CHECK_LABELS[key] for key, passed in checks.items() if not passed]
    return '未通過：' + '；'.join(failed) if failed else '全部通過'


def link(path, title=None):
    path = Path(path)
    text = title or path.name
    if not path.exists():
        return f'{text}：待產生（`{rel(path)}`）'
    target = path.relative_to(REPORTS).as_posix() if path.is_relative_to(REPORTS) else '../' + rel(path)
    return f'[{text}]({target})'


def summarize(plan, frozen, results, acceptance, snapshots):
    status = 'still_pending' if acceptance is None else 'passed_backtests' if acceptance['accepted'] else 'retain_v2'
    development = results['development']
    selected = frozen['selected_candidate'] if frozen is not None else development['decision']['selected_candidate'] if development is not None else None
    dev_rejected = development is not None and development['decision']['selected_candidate'] == CONTROL
    stages = {}
    for stage, result in results.items():
        not_required = stage != 'development' and result is None and dev_rejected
        stages[stage] = {
            'status': 'complete' if result is not None else 'not_required' if not_required else 'pending',
            'dates': plan['splits'][stage], 'planned_fit_rows': plan['split_contract'][stage]['fit_rows'],
            'planned_query_rows': plan['split_contract'][stage]['query_rows'],
            'metrics': result['metrics'] if result is not None else None,
            'decision': result['decision'] if result is not None else None,
            'target_group_weights': result['target_group_weights'] if result is not None else None,
            'result_path': rel(OUT/'evaluation'/stage/'results.json'),
            'result_sha256': snapshots.get(rel(OUT/'evaluation'/stage/'results.json')),
            'comparison_path': rel(OUT/'evaluation'/stage/'comparison.csv'),
            'relative_changes_percent': None if result is None else {
                name: {key: delta_percent(item[key], result['metrics'][CONTROL][key]) for key in ('overall_mae', 'weighted_mae')}
                for name, item in result['metrics'].items()},
        }
    return {
        'round': 5, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'selected_candidate': selected, 'development_rejected': dev_rejected,
        'acceptance': acceptance, 'plan_sha256': snapshots[rel(OUT/'candidate_plan.json')],
        'public_v2_mae': PUBLIC_V2_MAE, 'public_score_source': 'User-confirmed v2 Kaggle Public result; record only, not a tuning target',
        'private_mae': None, 'private_status': 'unknown', 'stages': stages,
        'candidate_formulas': plan['candidate_formulas'], 'components': plan['components'], 'thresholds': plan['thresholds'],
        'prediction_units': 'TWD', 'lower_is_better': True, 'rounded': False,
        'relative_change_sign': 'negative means improvement versus the same-stage v2 control',
        'historically_untouched': False,
        'limitations': [
            'Repeated development reuse causes selection bias; one prespecified candidate does not erase previous reuse.',
            'Model overfitting and repeated-validation selection overfitting are distinct. Masking does not prove either was reduced.',
            'Masking can discard useful location details; retained features may still identify projects.',
            'Geographic keys are unverified project identities. Missing names are not one physical project.',
            'Historical confirmation was previously evaluated; auxiliary labels occurred in earlier historical training.',
            'A 1% gate is not statistical significance; observations within a project are correlated.',
            'Project-cluster bootstrap is a possible future diagnostic, not performed or added to this protocol.',
            'Passing backtests does not guarantee Public or Private improvement.',
        ],
        'final_refit_and_new_csv': {
            'status': 'pending' if status == 'passed_backtests' else 'not_required' if status == 'retain_v2' or dev_rejected else 'awaiting_finalized_backtest_decision',
            'performed_or_verified_by_this_report': False, 'submission_ready': None,
        },
        'retained_submission': 'outputs/submission_v2.csv' if status == 'retain_v2' else None,
        'reporting_only': True, 'scoring_or_training_performed': False,
        'receipt_verified_inputs': {path: digest for path, digest in snapshots.items() if digest is not None},
        'missing_optional_inputs': [path for path, digest in snapshots.items() if digest is None],
        'verification_scope': 'JSON/receipt integrity and saved contract/evidence consistency only; binary models, data, predictions and submission are not revalidated by this report',
        'report_script_sha256': sha256(Path(__file__)),
    }


def stage_markdown(stage, item):
    lines = ['## ' + STAGE_LABELS[stage], '',
             f"固定區間：{item['dates'][0]}～{item['dates'][1]}；計畫 fit {item['planned_fit_rows']:,} 筆、query {item['planned_query_rows']:,} 筆。", '']
    if item['status'] != 'complete':
        message = ('未執行（not_required）：Development 的唯一新候選不合格，依事前規則停止本輪後續年度；此處沒有分數或通過結論。'
                   if item['status'] == 'not_required' else '尚未完成（pending）：尚無附有效 SHA256 收據的完成結果；分數與決策保持 null。')
        lines.extend([message, ''])
        if item['status'] == 'pending':
            lines.extend([link(OUT/'evaluation'/stage/'results.json', '階段結果 JSON'), ''])
        return lines
    metrics, decision = item['metrics'], item['decision']
    lines.extend(['單位為元，MAE 越低越好；相對變化負值表示優於同年度 v2。', '',
                  '| 候選 | 整體 MAE | 加權 MAE | 整體相對 v2 | 加權相對 v2 | 已記錄門檻決策 |',
                  '|---|---:|---:|---:|---:|---|'])
    for name in CANDIDATES:
        metric, change = metrics[name], item['relative_changes_percent'][name]
        if name == CONTROL:
            gate = '比較基準'
        else:
            saved = decision['candidate_checks'][name] if stage == 'development' else decision
            passed = saved['eligible'] if stage == 'development' else saved['passed']
            gate = ('通過；' if passed else '未通過；') + gate_text(saved['checks'])
        lines.append(f'| {candidate_label(name)} | {amount(metric["overall_mae"])} | {amount(metric["weighted_mae"])} | {percent(change["overall_mae"])} | {percent(change["weighted_mae"])} | {gate} |')
    lines.extend(['', '評估器已記錄的候選選擇：' + candidate_label(decision['selected_candidate']) + '。本報告只轉述，不重新選型。', ''])
    if stage != 'development':
        lines.extend(['已記錄改善季度：' + ('、'.join(decision['improved_quarters']) or '無') + '；未換候選或比例重試。', ''])
    weights = item['target_group_weights']
    lines.extend([f'固定加權比例來自 full train＋valid 對 test X 的建案分組：已見 {weights["seen"]:.4%}、具名新案 {weights["new_nonmissing"]:.4%}、缺名 {weights["missing"]:.4%}。', '',
                  '| 候選 | 已見 MAE | 具名新案 MAE | 缺名 MAE |', '|---|---:|---:|---:|'])
    for name in CANDIDATES:
        lines.append('| ' + candidate_label(name) + ' | ' + ' | '.join(amount(metrics[name]['strata'][group]['mae']) for group in GROUPS) + ' |')
    base = metrics[CONTROL]
    lines.extend(['', '三組 query 筆數依序為 ' + '／'.join(f'{base["strata"][group]["rows"]:,}' for group in GROUPS) + '。缺名組權重為 0；無樣本組的 MAE 保留 null，不填 0 元。', '',
                  '| 預測季度 | 筆數 | v2 MAE | 遮罩融合 MAE | 相對 v2 |', '|---|---:|---:|---:|---:|'])
    for quarter in QUARTERS:
        original, masked = base['forecast_quarters'][quarter], metrics[CANDIDATE]['forecast_quarters'][quarter]
        lines.append(f'| {quarter} | {original["rows"]:,} | {amount(original["mae"])} | {amount(masked["mae"])} | {percent(delta_percent(masked["mae"], original["mae"]))} |')
    lines.extend(['', 'Q1～Q4 是從 query 起日算起第 0～91、92～183、184～275、276 天起四段，不是曆年季度。月份與縣市的完整實際分解保存在結果 JSON。', '',
                  link(OUT/'evaluation'/stage/'comparison.csv', '比較 CSV') + '；' + link(OUT/'evaluation'/stage/'results.json', '完整階段 JSON') + '。', ''])
    return lines


def render(report):
    if report['status'] == 'passed_backtests':
        outcome = '固定回測門檻已通過：' + candidate_label(report['selected_candidate']) + '。全量重訓與新提交 CSV 仍待完成及查核，尚不代表新提交檔就緒。'
    elif report['status'] == 'retain_v2':
        outcome = '最終決策：保留既有 v2。第五輪沒有獲准取代它的新候選；繼續使用既有 submission_v2.csv，已提交者無需因本輪重交。'
    else:
        outcome = '目前仍待完成／定案（pending）。以下只列已有有效收據的實際結果，不填補尚未產生的分數，也不先宣告回測通過。'
    lines = ['# 第五輪整建案遮罩回測報告', '', outcome, '',
             f'既有 v2 的 Kaggle Public MAE：**{PUBLIC_V2_MAE:,.4f} 元**；Private MAE：**未知**。Public 僅記錄使用者已確認的成績，沒有用作本輪目標、係數或門檻。', '',
             f'報告時間：{report["generated_at_utc"]}。凍結計畫 SHA256：`{report["plan_sha256"]}`。', '',
             '本程式僅驗證 JSON 的 SHA256 收據及所存契約／證據之間的一致性，再轉述結果；不載入資料、模型或預測陣列，不訓練、不重新評分。模型與資料二進位檔的完整檢核由引用的評估紀錄負責，本報告不將收據查核宣稱為重新核驗所有檔案。', '']
    for stage in STAGES:
        lines.extend(stage_markdown(stage, report['stages'][stage]))
    lines.extend(['## 唯一固定候選與遮罩方法', '',
                  '| 候選 | 一次 trend 前的固定總價融合公式 |', '|---|---|'])
    for name in CANDIDATES:
        lines.append(f'| {candidate_label(name)} | `{report["candidate_formulas"][name]["base"]}` |')
    lines.extend(['', '`old_catboost` 是舊 raw 總價模型；`lightgbm_unit` 是舊面積加權單價模型還原後的總價；`project_mask_catboost` 是本輪唯一新組件。各完整候選只套用一次既有 v2 reference 的 trend，reference 接收原始、未遮罩的 87 欄 X。只有完整融合候選與 control 計分，不直接以新組件分數選型。', '',
        '- 遮罩單位是城市＋行政區＋完整案名的整個具名建案。固定 `round5|42|` 前綴與 SHA256 門檻，讓每案落在雜湊空間的遮罩機率為 50%；同案所有列一致。這不保證恰好一半建案，更不等於遮掉一半資料列。',
        '- state 僅由當期 fit X 建立。已見 query 建案重用 fit 的遮罩旗標；全新具名 query 建案強制遮罩。缺案名列在 fit／query 都遮罩，但不合併為一個實體建案，也不加入具名建案旗標字典。',
        '- 同時遮罩 `project_name`、`address_raw`、`building_and_unit`、`building_block` 四欄，使用固定 token；其餘 83 欄及原順序、型別保持原樣，共 87 欄、21 個類別欄。計分建案分組及 reference 一律沿用原 X。',
        '- 固定原 raw CatBoost 參數、seed 42、2,999 輪；目標 `y/1e6`、均等權重，不設 eval_set、不 early stopping。遮罩率、seed、輪數與融合比例沒有搜尋或依回測重試。', '',
        '遮罩可能移除有用的地點／棟戶資訊，而保留欄仍可能識別同案，不能預先認定一定降低模型過擬合。地理鍵沒有經過實體建案身分核驗，同名異地或地理欄缺失仍是限制。', '',
        '## 固定接受門檻與限制', '',
        '- Development：整體及固定三組加權 MAE 都至少改善 1%；每一預測季度、每一至少 500 筆的計分組，惡化不得超過 5%。唯一候選失敗便停止本輪，不看後兩階段、不換遮罩率或比例重試；通過才凍結這個候選。',
        '- Confirmation：只比較凍結候選與 v2；整體及加權 MAE 都不得變差，至少兩個預測季度改善，至少 500 筆的每組惡化不超過 5%。',
        '- Auxiliary：只作額外否決；整體及加權 MAE 最多惡化 3%，至少 500 筆的每組最多惡化 5%；季度僅診斷。後兩階段都通過才允許後續全量重訓，不能據此重新選比例或候選。', '',
        '模型對 fit 資料的過擬合，與多輪重複使用相同開發年度所產生的選型過擬合，是不同問題。單一事前固定候選只能限制本輪新增搜尋，不能消除歷次開發集重用的偏差。Development 與歷史 Confirmation 已使用過；Auxiliary 的 query 年度雖不與另兩段重疊，其標籤曾用於過往訓練，不能稱為 untouched。', '',
        '**1% 是事前工程門檻，不是統計顯著性證明。** 同建案交易誤差相關，逐列觀測不獨立；按建案分群的 paired bootstrap 只是尚未執行的未來診斷，未加入這次規則，也沒有由此建立信賴區間。單次回測改善不保證 Kaggle Public 或 Private 改善。', '',
        '只使用使用者提供的本機資料；沒有到內政部成交資料來源對答案。Private 成績仍未知。', '',
        '## 決策與追溯紀錄', ''])
    if report['acceptance'] is None:
        lines.extend(['尚無 finalized 的 accepted_candidate.json。即使開發已拒絕候選，後兩階段無需執行，正式 finalize 紀錄仍須另行保存。', ''])
    else:
        saved = report['acceptance']
        lines.extend([f'已記錄 `accepted={str(saved["accepted"]).lower()}`、`deployment_decision={saved["deployment_decision"]}`。原因：{saved["reason"]}。', ''])
    if report['status'] == 'retain_v2':
        lines.extend([link(ROOT/'outputs/submission_v2.csv', '維持既有 v2 提交檔') + '。本報告不重新驗證或產生該 CSV。', ''])
    for path, title in [
        (REPORTS/'round5_work_log.md', '第五輪工作紀錄'),
        (REPORTS/'round5_preprocessing_record.md', '第五輪處理紀錄'),
        (REPORTS/'round5_validation_plan.md', '第五輪驗證計畫'),
        (OUT/'audit', '第五輪稽核資料'),
        (OUT/'audit/mask_distribution.json', 'fit X 遮罩分布稽核'),
        (OUT/'audit/mask_synthetic_preflight_v2.json', '修正缺值口徑後的合成前置檢查'),
        (OUT/'audit/evaluation_preflight.json', '評估門檻與一次 trend 合成檢查'),
        (OUT/'candidate_plan.json', '訓練前凍結計畫'),
        (OUT/'frozen_candidate.json', 'Development 凍結決策'),
        (OUT/'accepted_candidate.json', '最終回測接受決策'),
    ]:
        lines.append('- ' + link(path, title))
    lines.extend(['- [第五輪完整結果 JSON](round5_results.json)', '',
                  '本報告不修改任何凍結來源、計畫、模型、評估結果或既有提交檔；回測通過也不等於全量模型與新 CSV 已完成或已上傳。', ''])
    return '\n'.join(lines)


def main():
    plan, frozen, results, acceptance, snapshots = load_snapshot()
    report = summarize(plan, frozen, results, acceptance, snapshots)
    markdown = render(report)
    for path, digest in snapshots.items():
        if digest is None:
            require(not (ROOT/path).exists(), 'Optional input appeared while reporting: ' + path)
        else:
            require(sha256(ROOT/path) == digest, 'Input changed while reporting: ' + path)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS/'optimization_round5_report.md').write_text(markdown, encoding='utf-8')
    (REPORTS/'round5_results.json').write_text(encoded, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'selected_candidate': report['selected_candidate'],
                      'outputs': ['reports/optimization_round5_report.md', 'reports/round5_results.json']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
