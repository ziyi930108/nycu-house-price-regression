"""Render R6 reports from verified JSON receipts; never load data or predictions.

No project imports, model training, MAE recomputation, gate recalculation, or
candidate selection. Relative improvements use saved aggregate MAEs only.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, REPORTS = ROOT/'outputs/round6', ROOT/'reports'
PLAN_SHA = '606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8'
CONTROL, CANDIDATE, COMPONENT = 'v2_control', 'geography_blend', 'geography_catboost'
NAMES = (CONTROL, CANDIDATE)
STAGES = ('development', 'confirmation', 'auxiliary')
GROUPS, QUARTERS = ('seen', 'new_nonmissing', 'missing'), ('Q1', 'Q2', 'Q3', 'Q4')
PUBLIC_V2_MAE = 2883250.4138
LABELS = {CONTROL: 'v2 控制', CANDIDATE: '地理融合候選'}
GROUP_LABELS = {'seen': '已見建案', 'new_nonmissing': '具名新建案', 'missing': '缺案名'}
CHECK_LABELS = {
    'overall_improves_at_least_1pct': '整體 MAE 至少改善 1%',
    'weighted_improves_at_least_1pct': '加權 MAE 至少改善 1%',
    'quarters_deteriorate_at_most_5pct': '各預測季退步最多 5%',
    'large_strata_deteriorate_at_most_5pct': 'n≥500 的各建案分組退步最多 5%',
    'overall_within_limit': '整體 MAE 符合該階段上限',
    'weighted_within_limit': '加權 MAE 符合該階段上限',
    'at_least_two_quarters_improve': '至少兩個預測季改善',
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_locked(path, snapshots, optional=False):
    path, receipt = Path(path), Path(path).with_suffix('.sha256')
    if not path.exists():
        need(not receipt.exists(), 'Orphan receipt: ' + rel(receipt))
        if optional:
            snapshots[rel(path)] = snapshots[rel(receipt)] = None
            return None
        raise RuntimeError('Missing required JSON: ' + rel(path))
    need(path.is_file() and receipt.is_file(), 'Missing JSON/receipt: ' + rel(path))
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    need(receipt.read_text(encoding='ascii').strip() == digest, 'Receipt differs: ' + rel(path))
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw.decode('utf-8-sig'), parse_constant=invalid)
    need(isinstance(value, dict), 'Expected JSON object')
    snapshots[rel(path)], snapshots[rel(receipt)] = digest, sha(receipt)
    return value


def number(value):
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, 'Invalid MAE/weight')


def validate_metrics(metrics, rows):
    need(set(metrics) == set(NAMES), 'Unexpected evaluated candidates')
    for name, item in metrics.items():
        need(item['rows'] == rows, 'Metric row count differs')
        for key in ('overall_mae', 'weighted_mae'):
            number(item[key])
        for field, groups in (('strata', GROUPS), ('forecast_quarters', QUARTERS)):
            need(set(item[field]) == set(groups), 'Incomplete metric groups')
            for group, value in item[field].items():
                n = value['rows']
                need(isinstance(n, int) and not isinstance(n, bool) and n >= 0, 'Invalid group rows')
                if n == 0:
                    need(value['mae'] is None, 'Empty-group MAE must remain null')
                else:
                    number(value['mae'])
                need(n == metrics[CONTROL][field][group]['rows'], 'Candidate/control group rows differ')
            need(sum(v['rows'] for v in item[field].values()) == rows, 'Groups do not cover query rows')


def validate_decision(stage, decision):
    """Only saved booleans/identities are checked, never score-based gates."""
    need(decision['selected_candidate'] in NAMES, 'Unknown saved selection')
    if stage == 'development':
        need(set(decision['candidate_checks']) == {CANDIDATE}, 'Unexpected candidate checks')
        item = decision['candidate_checks'][CANDIDATE]
        expected = {'overall_improves_at_least_1pct', 'weighted_improves_at_least_1pct', 'quarters_deteriorate_at_most_5pct', 'large_strata_deteriorate_at_most_5pct'}
        checks = item['checks']
        need(set(checks) == expected and all(isinstance(v, bool) for v in checks.values()), 'Invalid saved development flags')
        need(isinstance(item['eligible'], bool) and item['eligible'] == all(checks.values()), 'Inconsistent saved eligibility')
        need(decision['qualified_candidates'] == ([CANDIDATE] if item['eligible'] else []), 'Saved qualified list differs')
        need(decision['selected_candidate'] == (CANDIDATE if item['eligible'] else CONTROL), 'Saved selected identity differs')
    else:
        checks = decision['checks']
        expected = {'overall_within_limit', 'weighted_within_limit', 'large_strata_deteriorate_at_most_5pct'}
        if stage == 'confirmation':
            expected.add('at_least_two_quarters_improve')
        need(set(checks) == expected and all(isinstance(v, bool) for v in checks.values()), 'Invalid later flags')
        need(decision['selected_candidate'] == CANDIDATE and isinstance(decision['passed'], bool) and decision['passed'] == all(checks.values()), 'Inconsistent saved later decision')
        need(decision['candidate_reselection_performed'] is False, 'Later reselection prohibited')


def load_snapshot():
    snapshots = {}
    plan = read_locked(OUT/'candidate_plan.json', snapshots)
    need(snapshots[rel(OUT/'candidate_plan.json')] == PLAN_SHA, 'Wrong frozen round-six plan')
    need(plan['round'] == 6 and plan['control'] == CONTROL and set(plan['candidates']) == {CANDIDATE} and set(plan['components']) == {COMPONENT}, 'Unexpected experiment')
    rules = plan['components'][COMPONENT]['feature_rules']
    frozen = read_locked(OUT/'frozen_candidate.json', snapshots, optional=True)
    results = {s: read_locked(OUT/'evaluation'/s/'results.json', snapshots, optional=True) for s in STAGES}
    manifests = {}
    for stage in (*STAGES, 'final'):
        path = OUT/'geography'/stage/'manifest.json'
        item = read_locked(path, snapshots, optional=stage != 'development'); manifests[stage] = item
        if item is None:
            continue
        need(item['stage'] == stage and item['snapshot_year'] == plan['geography']['years'][stage], 'Geography stage/year differs')
        need(item['feature_rules'] == rules and item['geographic_values_learned_from_targets'] is False and item['existing_target_containing_dedup_rule_reproduced_for_alignment'] is True, 'Geography/dedup contract differs')
        stamp = datetime.fromisoformat(item['replication_timestamp'].replace('Z', '+00:00'))
        origin = datetime.fromisoformat(item['forecast_origin_utc'].replace('Z', '+00:00'))
        need(stamp.tzinfo is not None and origin.tzinfo is not None and stamp < origin, 'Geography timestamp violates origin')
        if stage == 'development':
            need(snapshots[rel(path)] == plan['geography']['development_manifest_sha256'], 'Development geography differs from plan')
    for stage, item in results.items():
        if item is None:
            continue
        need(item['format_version'] == 1 and item['stage'] == stage and item['plan_sha256'] == PLAN_SHA, 'Result identity differs')
        for key in ('candidate_formulas', 'thresholds', 'target_group_weights', 'runtime_versions'):
            need(item[key] == plan[key], 'Result/plan differs: ' + key)
        need(item['source_sha256'] == plan['evaluation_source_sha256'] and item['geo_feature_rules'] == rules, 'Result source/geography differs')
        need(item['component_parameters'] == {COMPONENT: plan['components'][COMPONENT]['parameters']}, 'Parameters differ')
        need(item['evaluated_candidates'] == list(NAMES) and item['prediction_units'] == 'TWD' and item['rounded'] is False, 'Candidate/units differ')
        for key in ('query_labels_used_for_prediction', 'component_scored_directly', 'historically_untouched', 'public_score_used_as_target'):
            need(item[key] is False, 'Unexpected recorded policy: ' + key)
        for key in ('fit_X_state_reconstruction_verified', 'query_geographic_state_unchanged', 'reference_uses_original_features', 'historical_confirmation_control_previously_evaluated'):
            need(item[key] is True, 'Missing saved verification: ' + key)
        need(item['reference_trend_applications_per_candidate'] == 1 and len(item['feature_columns']) == len(set(item['feature_columns'])) == 87, 'Reference/schema differs')
        for path, digest in plan['data_sha256'].items():
            need(item['data_sha256'].get(path) == digest, 'Result data identity differs: ' + path)
        for name in STAGES:
            actual, expected = item['splits'][name], plan['split_contract'][name]
            need(actual['dates'] == plan['splits'][name] and actual['fit_rows'] == expected['fit_rows'] and actual['query_rows'] == expected['query_rows'], 'Result split differs')
        validate_metrics(item['metrics'], plan['split_contract'][stage]['query_rows'])
        validate_decision(stage, item['decision'])
        manifest_path = rel(OUT/'geography'/stage/'manifest.json')
        need(manifests[stage] is not None and item['input_artifact_sha256'].get(manifest_path) == snapshots[manifest_path], 'Result geography manifest is not bound')
        if stage == 'development':
            need(item['frozen_candidate_sha256'] is None, 'Development depends on later freeze')
        else:
            need(frozen is not None and frozen['selected_candidate'] == CANDIDATE and item['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Later result lacks matching freeze')
    dev = results['development']
    if frozen is not None:
        need(dev is not None and frozen['format_version'] == 1 and frozen['plan_sha256'] == PLAN_SHA, 'Freeze lacks matching development/plan')
        need(frozen['selected_candidate'] == dev['decision']['selected_candidate'] and frozen['required_components'] == plan['required_components'][frozen['selected_candidate']], 'Freeze selected identity differs')
        for key in ('source_sha256', 'data_sha256', 'component_parameters', 'target_group_weights', 'splits', 'feature_columns', 'runtime_versions', 'geo_feature_rules', 'candidate_formulas', 'thresholds'):
            need(frozen[key] == dev[key], 'Frozen contract differs: ' + key)
        need(frozen['later_stage_new_fitting_artifacts_absent_at_freeze'] is True and frozen['historically_untouched_confirmation'] is False, 'Freeze policy differs')
        evidence = {**dev['input_artifact_sha256'], **dev['output_artifact_sha256']}
        for suffix in ('.json', '.sha256'):
            path = rel(OUT/'evaluation/development'/('results' + suffix)); evidence[path] = snapshots[path]
        for path, digest in evidence.items():
            need(frozen['development_artifact_sha256'].get(path) == digest, 'Frozen evidence differs: ' + path)
        for stage in ('confirmation', 'auxiliary'):
            if results[stage] is not None:
                for key in ('data_sha256', 'splits', 'feature_columns', 'target_group_weights'):
                    need(results[stage][key] == frozen[key], 'Later frozen contract differs: ' + key)
    acceptance = read_locked(OUT/'accepted_candidate.json', snapshots, optional=True)
    if acceptance is not None:
        need(frozen is not None and acceptance['plan_sha256'] == PLAN_SHA and acceptance['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Acceptance lacks matching freeze/plan')
        need(acceptance['selected_candidate'] == frozen['selected_candidate'] and isinstance(acceptance['accepted'], bool), 'Acceptance selection differs')
        need(acceptance['candidate_reselection_performed'] is False and acceptance['historically_untouched_confirmation'] is False, 'Acceptance policy differs')
        if frozen['selected_candidate'] == CONTROL:
            need(acceptance['accepted'] is False and acceptance['stage_decisions'] == {} and all(results[s] is None for s in ('confirmation', 'auxiliary')), 'Rejected development has later decisions')
        else:
            need(set(acceptance['stage_decisions']) == {'confirmation', 'auxiliary'}, 'Missing final veto decisions')
            for stage in ('confirmation', 'auxiliary'):
                need(results[stage] is not None and acceptance['stage_decisions'][stage] == results[stage]['decision'], 'Final/stage decision differs')
            need(acceptance['accepted'] == all(results[s]['decision']['passed'] for s in ('confirmation', 'auxiliary')), 'Final flag differs from saved stage flags')
        need(acceptance['deployment_decision'] == (CANDIDATE if acceptance['accepted'] else 'retain_v2'), 'Final deployment decision differs')
        for path, digest in snapshots.items():
            if digest is not None and (path.startswith('outputs/round6/evaluation/') or path.startswith('outputs/round6/frozen_candidate.')):
                need(acceptance['evidence_sha256'].get(path) == digest, 'Final evidence identity differs: ' + path)
    return plan, frozen, results, acceptance, manifests, snapshots


def load_release(acceptance, snapshots):
    """Validate release JSON identity and CSV bytes only; never parse prices."""
    proof = read_locked(OUT/'release/release.json', snapshots, optional=True)
    if proof is None:
        return None
    need(acceptance is not None and acceptance['accepted'] is True, 'Release lacks accepted backtests')
    need(proof['plan_sha256'] == PLAN_SHA and proof['accepted_candidate_sha256'] == snapshots[rel(OUT/'accepted_candidate.json')], 'Release acceptance/plan differs')
    need(proof['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Release freeze differs')
    need(proof['candidate'] == CANDIDATE and proof['release_version'] == 'v3' and proof['experiment_round'] == 6 and proof['rows'] == 47073, 'Unexpected release identity/row metadata')
    need(proof['path'] == 'outputs/submission_v3.csv', 'Unexpected submission path')
    path = ROOT/proof['path']
    need(path.resolve().is_relative_to(ROOT) and path.is_file(), 'Missing or unsafe released CSV')
    digest = sha(path)
    need(digest == proof['sha256'], 'Released CSV SHA differs')
    snapshots[rel(path)] = digest
    return proof


def outcome(dev, frozen, results, acceptance):
    if acceptance is not None:
        return ('backtests_passed_release_pending', '固定回測已接受；全量訓練與新提交檔仍待另行完成及驗證。') if acceptance['accepted'] else ('retain_v2', '正式決策：保留 v2；本輪候選未獲接受。')
    if dev is None:
        return 'awaiting_development', '開發結果待完成，尚待驗證，现行推薦仍為 v2。'
    if dev['decision']['selected_candidate'] == CONTROL:
        return 'retain_v2_finalization_pending', '開發未通過，依固定規則保留 v2；正式 finalize 紀錄待完成。'
    if frozen is None:
        return 'development_passed_freeze_pending', '開發通過，候選凍結紀錄尚待完成；現行推薦仍為 v2。'
    absent = [s for s in ('confirmation', 'auxiliary') if results[s] is None]
    if absent:
        return 'development_passed_validation_pending', '開發通過且候選已凍結；' + '、'.join(absent) + ' 未執行評估，尚待驗證；現行推薦仍為 v2。'
    return 'awaiting_finalization', '後續年度已有評估紀錄，正式接受決策尚待 finalize；現行推薦仍為 v2。'


def improvement(value, base):
    return None if value is None or base is None or base == 0 else 100.0 * (base-value) / base


def summarize(plan, frozen, results, acceptance, manifests, snapshots, release=None):
    status, text = outcome(results['development'], frozen, results, acceptance)
    stages = {}
    rejected = results['development'] is not None and results['development']['decision']['selected_candidate'] == CONTROL
    for stage, result in results.items():
        changes = None
        if result is not None:
            changes = {}; base = result['metrics'][CONTROL]
            for name, metric in result['metrics'].items():
                changes[name] = {k: improvement(metric[k], base[k]) for k in ('overall_mae', 'weighted_mae')}
                for field in ('strata', 'forecast_quarters'):
                    changes[name][field] = {k: improvement(v['mae'], base[field][k]['mae']) for k, v in metric[field].items()}
        path = rel(OUT/'evaluation'/stage/'results.json')
        stages[stage] = {'status': 'evaluated' if result is not None else 'not_executed', 'not_executed_reason': None if result is not None else 'development_rejected' if rejected and stage != 'development' else 'no_completed_evaluation_json', 'dates': plan['splits'][stage], **{k: plan['split_contract'][stage][k] for k in ('fit_rows', 'query_rows')}, 'metrics': None if result is None else result['metrics'], 'decision': None if result is None else result['decision'], 'relative_improvement_percent': changes, 'result_path': path, 'result_sha256': snapshots[path]}
    geography = {}
    for stage, manifest in manifests.items():
        path = rel(OUT/'geography'/stage/'manifest.json')
        geography[stage] = {'status': 'prepared' if manifest is not None else 'not_prepared', 'planned_snapshot_year': plan['geography']['years'][stage], 'manifest_path': path, 'manifest_sha256': snapshots[path], 'replication_timestamp': None if manifest is None else manifest['replication_timestamp'], 'forecast_origin_utc': None if manifest is None else manifest['forecast_origin_utc'], 'alignment': None if manifest is None else manifest['alignment']}
    selected = frozen['selected_candidate'] if frozen is not None else None if results['development'] is None else results['development']['decision']['selected_candidate']
    report = {'round': 6, 'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'metric': 'MAE', 'unit': 'TWD', 'status': status, 'status_text': text.replace('现行', '現行'), 'selected_candidate': selected, 'acceptance': acceptance, 'backtest_accepted': None if acceptance is None else acceptance['accepted'], 'acceptance_finalized': acceptance is not None, 'recommendation': 'v2', 'recommended_submission': 'outputs/submission_v2.csv', 'new_submission': None, 'new_submission_verified_by_this_report': False, 'stages': stages, 'geography': geography, 'plan_sha256': PLAN_SHA, 'frozen_candidate_sha256': snapshots[rel(OUT/'frozen_candidate.json')], 'accepted_candidate_sha256': snapshots[rel(OUT/'accepted_candidate.json')], 'candidate_formulas': plan['candidate_formulas'], 'components': plan['components'], 'thresholds': plan['thresholds'], 'target_group_weights': plan['target_group_weights'], 'quarter_definition': plan['quarter_definition'], 'public_v2_mae': PUBLIC_V2_MAE, 'public_score_source': 'User-confirmed historical v2 Kaggle Public result; record only, never a tuning target', 'public_round6_mae': None, 'private_mae': None, 'private_status': 'unknown', 'relative_improvement_sign': 'positive means improvement versus same-stage v2; undefined for zero baseline or absent group', 'relative_improvement_source': 'arithmetic on receipt-verified aggregate MAEs only; no predictions or labels read', 'historically_untouched': False, 'limitations': plan['limitations'], 'reporting_only': True, 'scoring_or_training_performed': False, 'receipt_verified_inputs': {k: v for k, v in snapshots.items() if v is not None}, 'missing_optional_inputs': [k for k, v in snapshots.items() if v is None], 'verification_scope': 'JSON receipts and saved plan/result/freeze/acceptance/geography identity links only; binary data, models, predictions and CSV contents are not reread', 'report_script_sha256': sha(__file__)}

    report['release'] = release
    if release is not None:
        need(acceptance is not None and acceptance['accepted'] is True, 'Cannot recommend release without acceptance')
        report.update(status='released_v3', status_text='固定回測已接受，release 收據與新 CSV SHA 已核對；推薦本輪 submission_v3.csv。新提交的 Kaggle Public／Private 尚未測得。', recommendation='v3', recommended_submission=release['path'], new_submission={'path': release['path'], 'sha256': release['sha256'], 'rows': release['rows']}, new_submission_verified_by_this_report=True)
        report['verification_scope'] += '; released CSV bytes SHA verified without parsing price values'
    return report


def recommendation_line(report, base=REPORTS):
    if report['recommendation'] == 'v3':
        return '現行推薦為 ' + link(ROOT/report['recommended_submission'], '本輪新提交 submission_v3.csv', base) + '。已核對 release 收據、接受決策與 47,073 列中繼資料及 CSV SHA；本報告不重新計算或解析價格。新提交的 Kaggle Public／Private 尚未測得。'
    return '現行推薦為 ' + link(ROOT/report['recommended_submission'], '第二輪主版 submission_v2.csv', base) + '。本報告未驗證第六輪新 CSV；回測接受與全量推論、提交檔就緒是不同階段。'


def amount(value):
    return '無樣本（null）' if value is None else f'{value:,.2f}'


def percent(value):
    return '—' if value is None else f'{value:+.3f}%'


def link(path, title, base=REPORTS):
    return '[' + title + '](' + Path(os.path.relpath(path, base)).as_posix() + ')'


def stage_lines(stage, item):
    lines = ['## ' + stage, '', f"固定期間 {item['dates'][0]} 至 {item['dates'][1]}；fit {item['fit_rows']:,} 列、query {item['query_rows']:,} 列。", '']
    if item['status'] != 'evaluated':
        reason = '開發未通過，依規則不執行本年度。' if item['not_executed_reason'] == 'development_rejected' else '尚無附有效收據的完成評估 JSON，後續評估仍待執行。'
        return lines + ['**未執行評估。** ' + reason + ' 分數與決策為 null，不推測通過或失敗。', '']
    metrics, decision, changes = item['metrics'], item['decision'], item['relative_improvement_percent']
    lines += ['單位為元，MAE 越低越好；相對改善正值表示優於同年度 v2，負值表示退步。', '', '| 比較項 | 整體 MAE | 加權 MAE | 整體改善 | 加權改善 |', '|---|---:|---:|---:|---:|']
    for name in NAMES:
        m, d = metrics[name], changes[name]
        lines.append(f"| {LABELS[name]} | {amount(m['overall_mae'])} | {amount(m['weighted_mae'])} | {percent(d['overall_mae'])} | {percent(d['weighted_mae'])} |")
    checks = decision['candidate_checks'][CANDIDATE]['checks'] if stage == 'development' else decision['checks']
    lines += ['', '以下直接轉述評估器保存的門檻旗標，不由 MAE 重新判門檻。', '', '| 固定檢查 | 已保存結論 |', '|---|---|']
    for key, passed in checks.items():
        lines.append(f"| {CHECK_LABELS[key]} | {'通過' if passed else '未通過'} |")
    lines += ['', f"已保存選擇：`{decision['selected_candidate']}`。" + ('' if stage == 'development' else f" 本年度 `passed={str(decision['passed']).lower()}`；已保存改善季度：{'、'.join(decision['improved_quarters']) or '無'}。"), '', '| 建案組別 | 列數 | v2 MAE | 地理融合 MAE | 相對改善 |', '|---|---:|---:|---:|---:|']
    for group in GROUPS:
        b, c = metrics[CONTROL]['strata'][group], metrics[CANDIDATE]['strata'][group]
        lines.append(f"| {GROUP_LABELS[group]} | {b['rows']:,} | {amount(b['mae'])} | {amount(c['mae'])} | {percent(changes[CANDIDATE]['strata'][group])} |")
    lines += ['', '| 預測季 | 列數 | v2 MAE | 地理融合 MAE | 相對改善 |', '|---|---:|---:|---:|---:|']
    for quarter in QUARTERS:
        b, c = metrics[CONTROL]['forecast_quarters'][quarter], metrics[CANDIDATE]['forecast_quarters'][quarter]
        lines.append(f"| {quarter} | {b['rows']:,} | {amount(b['mae'])} | {amount(c['mae'])} | {percent(changes[CANDIDATE]['forecast_quarters'][quarter])} |")
    return lines + ['', '預測季為起點後第 0–91、92–183、184–275、276 天以上，並非曆年季度。', '', link(ROOT/item['result_path'], '完整結果 JSON') + f"；SHA256 `{item['result_sha256']}`。", '']


def render_report(report):
    lines = ['# 第六輪地理特徵回測報告', '', '**' + report['status_text'] + '**', '', recommendation_line(report), '', f"既有 v2 Kaggle Public MAE：**{PUBLIC_V2_MAE:,.4f} 元**，僅為歷史提交紀錄；第六輪 Public 未知，Private 未知。Public 不用於調參、地理規則或門檻。", '', f"報告時間：{report['generated_at_utc']}。plan SHA256：`{PLAN_SHA}`。", '', '本程式只讀有收據的 JSON，核對保存的契約與證據連結；不讀 y、模型或預測，不訓練、不重算 MAE、不選模型。相對改善只以已保存彙總 MAE 計算；原始二進位產物查核由評估器紀錄負責。', '']
    for stage in STAGES:
        lines += stage_lines(stage, report['stages'][stage])
    lines += ['## 固定候選、地理欄位及時間來源', '', '| 比較項 | 一次 trend 前的總價融合公式 |', '|---|---|']
    for name in NAMES:
        lines.append(f"| {LABELS[name]} | `{report['candidate_formulas'][name]['base']}` |")
    lines += ['', '`lightgbm_unit` 已還原為總價；唯一新元件為 `geography_catboost`。兩個完整融合各以原 87 欄與同一 fit 側 reference 套一次 trend，不以新元件直接分數選型。', '', '原 87 欄不變，追加經緯度、log 道路長度、log 支持範圍、位置旗標、碎段旗標、匹配狀態與 0.02 度格網，共 95 欄／23 類別。格網先由 float64 座標 floor，再將數值欄轉 float32；無位置時相關數值為 NaN、位置旗標為 0、格網為 `__MISSING__`。', '', 'raw CatBoost 固定 2,999 輪、seed 42、GPU、4 threads、`y/1e6`、均等權重；沒有 eval_set、early stopping、參數／格網／seed／融合比例搜尋。', '', '| 階段 | 指定快照 | 地理準備 | PBF 實際時間（UTC） | manifest |', '|---|---:|---|---|---|']
    for stage, geo in report['geography'].items():
        source = link(ROOT/geo['manifest_path'], '來源 manifest') if geo['manifest_sha256'] else '尚未產生'
        lines.append(f"| {stage} | {geo['planned_snapshot_year']} | {'已準備' if geo['status'] == 'prepared' else '未準備'} | {geo['replication_timestamp'] or '未取得'} | {source} |")
    lines += ['', '使用離線 Geofabrik／OpenStreetMap，PBF 時間早於該預測起點，同窗 fit/query 使用同版。這是預測起點地圖，不證明每笔早期交易當年已有相同道路。2022 版 test 配對僅供工程診斷，正式 test 指定 2024 版。', '', '高雄依同年度 parent/subarea 與原始幾何核對，缺面保留缺失。雙路僅接受明確、無巷弄／地籍／多門牌歧義的實際平面交集，保留 50 公尺點群與任兩點距離限制。位置是粗略代表點，不是門牌，也不證明同高程連通。', '', '對齊重現原 529 筆 train 重複及合併後 28 筆重複的刪除規則；既有完整列去重包含目標，只用於維持模型列序。新座標、配對、格網不從價格學習；沒有外部成交答案或內政部對答案。', '', '## 固定接受門檻與限制', '', '- Development：整體及加權都至少改善 1%；各季及 n≥500 的各建案分組退步最多 5%，單一固定候選失敗即停。', '- Confirmation：整體及加權不得退步，至少兩季改善，大分組退步最多 5%；只確認 freeze 候選。', '- Auxiliary：僅作否決；整體及加權退步最多 3%，大分組最多 5%，季度僅診斷，不重選。', '', '評分組沿用原 87 欄；seen／new_nonmissing／missing 權重為 ' + '／'.join(f"{report['target_group_weights'][g]:.6f}" for g in GROUPS) + '。缺名權重為 0，但 n≥500 仍受分組限制；無樣本 MAE 保留 null，不填 0。', '']
    if report['acceptance'] is None:
        lines += ['正式 acceptance 尚未產生：**尚待驗證／定案，尚無經本報告驗證的新提交 CSV**。開發通過不是已接受 release。', '']
    else:
        a = report['acceptance']; lines += [f"已保存 `accepted={str(a['accepted']).lower()}`、`deployment_decision={a['deployment_decision']}`；原因：{a['reason']}。", '']
    lines += ['歷史 development／confirmation 已使用，auxiliary 年度標籤亦曾進入過往訓練，不能稱為完全未見測試。固定單候選不能消除多輪選型偏差；新增地理欄也不證明過擬合下降。1% 是工程門檻，不是統計顯著性證明；同案誤差相關，回測改善不保證 Public／Private 改善。', '', '## 追溯與重現', '']
    for path, title in [(OUT/'candidate_plan.json', '正式凍結計畫'), (REPORTS/'round6_preprocessing_record.md', '資料處理紀錄'), (REPORTS/'round6_work_log.md', '操作工作紀錄'), (REPORTS/'round6_validation_plan.md', '驗證規約'), (REPORTS/'round6_results.json', '完整結果 JSON'), (ROOT/'README_round6.md', '第六輪說明')]:
        lines.append('- ' + link(path, title))
    for name, title in [('frozen_candidate.json', '開發凍結決策'), ('accepted_candidate.json', '正式回測接受決策'), ('release/release.json', '新提交 release 證據')]:
        path = OUT/name; lines.append('- ' + (link(path, title) if rel(path) in report['receipt_verified_inputs'] else title + '：尚未產生'))
    lines += ['', '專案根目錄執行 `.venv\\Scripts\\python.exe src\\round6_report.py` 可重生本報告、結果 JSON 與 README；只讀既有決策，不訓練或重新評分。', '']
    return '\n'.join(lines).replace('每笔', '每筆')


def render_readme(report):
    lines = ['# 第六輪：離線地理特徵', '', '**' + report['status_text'] + '**', '', recommendation_line(report, ROOT), '', '## 結果與紀錄', '']
    for path, title in [(REPORTS/'optimization_round6_report.md', '完整報告與各組／季度結果'), (REPORTS/'round6_results.json', '機器可讀結果及收據來源'), (REPORTS/'round6_preprocessing_record.md', '資料處理紀錄'), (REPORTS/'round6_work_log.md', '操作紀錄'), (REPORTS/'round6_validation_plan.md', '驗證規約'), (OUT/'candidate_plan.json', '不可變正式計畫')]:
        lines.append('- ' + link(path, title, ROOT))
    lines += ['', '| 階段 | 已保存狀態 |', '|---|---|']
    for stage, item in report['stages'].items():
        if item['status'] != 'evaluated':
            text = '未執行評估；分數為 null'
        elif stage == 'development':
            text = '已評估；' + ('開發通過' if item['decision']['selected_candidate'] == CANDIDATE else '開發拒絕，保留 v2')
        else:
            text = '已評估；保存的門檻' + ('通過' if item['decision']['passed'] else '未通過')
        lines.append(f'| {stage} | {text} |')
    lines += ['', '## 固定方法與報告重現', '', '唯一新增 GPU raw CatBoost 使用 87 原欄＋8 地理欄，95 欄／23 類別；2,999 輪、seed 42、4 threads、總價除以 1e6、均等權重，沒有 eval_set 或 early stopping。固定融合為原 raw 37.5%、地理 raw 37.5%、舊單價 LightGBM 還原總價 25%，再套原 87 欄 reference 一次 trend，沒有搜尋參數、格網或比例。', '', '離線 OSM／Geofabrik 不含外部成交價格。development／confirmation／auxiliary／正式 test 各指定 2022／2023／2021／2024 快照；實際 PBF 時間以 manifest 為準。道路／平面交點是粗略位置。', '', '在專案根目錄重生三份報告（Python 標準函式庫，不載入模型或資料陣列）：', '', '```powershell', '.\\.venv\\Scripts\\python.exe src\\round6_report.py', '```', '', '程式核對 plan、evaluation、freeze、acceptance 與地理 manifest 的 JSON 收據及身分後覆寫報告；沒有評估 JSON 的年度明列未執行，acceptance 缺失時保持尚待驗證。完整實驗另依固定計畫於獨立工作副本重建，不在既有封存輸出上重訓。', '', f'計畫 SHA256：`{PLAN_SHA}`。v2 Public MAE {PUBLIC_V2_MAE:,.4f} 元僅為歷史提交紀錄；第六輪 Public、Private 未知。', '', '歷史標籤已曾使用，不能稱這些窗口完全未見；固定單候選不能消除以前的選型偏差。只轉述已保存決策，Private 改善未獲證明。沒有到內政部成交資料對答案。', '']
    return '\n'.join(lines)


def verify_snapshot(snapshots):
    for name, expected in snapshots.items():
        path = ROOT/name
        if expected is None:
            need(not path.exists(), 'Optional input appeared while rendering; rerun: ' + name)
        else:
            need(sha(path) == expected, 'JSON/receipt changed while rendering: ' + name)


def self_check():
    absent = {s: None for s in STAGES}; passed = {'decision': {'selected_candidate': CANDIDATE}}; rejected = {'decision': {'selected_candidate': CONTROL}}
    need(outcome(None, None, absent, None)[0] == 'awaiting_development', 'Missing status failed')
    need(outcome(rejected, {}, absent, None)[0] == 'retain_v2_finalization_pending', 'Development rejection failed')
    need(outcome(passed, {}, absent, None)[0] == 'development_passed_validation_pending', 'Development pass must remain pending')
    need(outcome(passed, {}, absent, {'accepted': False})[0] == 'retain_v2', 'Final rejection failed')
    need(outcome(passed, {}, absent, {'accepted': True})[0] == 'backtests_passed_release_pending', 'Acceptance must not claim release')
    need(improvement(90, 100) == 10 and improvement(110, 100) == -10 and improvement(None, 100) is None and improvement(10, 0) is None, 'Improvement/null handling failed')
    return {'status': 'passed', 'synthetic_status_cases': 5, 'relative_improvement_cases': 4, 'real_scores_recomputed': False}


def main():
    self_check()
    plan, frozen, results, acceptance, manifests, snapshots = load_snapshot()
    release = load_release(acceptance, snapshots)
    report = summarize(plan, frozen, results, acceptance, manifests, snapshots, release=release)
    contents = {REPORTS/'optimization_round6_report.md': render_report(report), REPORTS/'round6_results.json': json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', ROOT/'README_round6.md': render_readme(report)}
    verify_snapshot(snapshots)
    for path, text in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'selected_candidate': report['selected_candidate'], 'acceptance_finalized': report['acceptance_finalized'], 'recommended_submission': report['recommended_submission'], 'outputs': [rel(p) for p in contents]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
