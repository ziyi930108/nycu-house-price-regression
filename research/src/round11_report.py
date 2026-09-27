"""R11 finalized-artifact reporting; no training, prediction, scoring or release.

The only price metrics read are the already recorded, receipt-verified result JSON.
Binary dependencies are hashed for identity, never deserialized by this report.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import round11_experiment as experiment

ROOT = experiment.ROOT
OUT = experiment.OUT
STAGES = experiment.STAGES
REPORT = ROOT / 'reports/round11_report.md'
RESULTS = ROOT / 'reports/round11_results.json'
STAGE_LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
STATUS_LABELS = {
    'price_evaluated': '已完成價格評分',
    'pure_X_feasibility_failed_price_not_evaluated': '純 X 可行性未通過；未訓練或評分價格候選',
    'weights_prepared_price_not_executed': '已準備權重；未執行價格候選',
    'not_executed': '未執行',
    'not_executed_after_pure_X_stop': '因另一年度純 X 可行性失敗而未執行',
    'not_executed_after_development_rejection': '開發期價格門檻未通過，未執行',
}
STOP_LABELS = {
    'pure_X_feasibility_failed': '後期固定純 X 可行性條件未通過',
    'development_price_gates_failed': '開發期固定價格門檻未通過',
    'later_price_gates_failed': '後續年度固定價格門檻未全部通過',
    'three_stage_price_gates_passed': '三年度固定價格門檻皆通過，待最終訓練與另行發布',
}
require = experiment.require


def number(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def improvement(base, candidate):
    if base is None or candidate is None or base == 0:
        return None
    return 100.0 * (base - candidate) / base


def percent(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}%'


def validate_metrics(metrics, rows):
    require(set(metrics) == {experiment.CONTROL, experiment.CANDIDATE}, 'Wrong scored candidates')
    left, right = metrics[experiment.CONTROL], metrics[experiment.CANDIDATE]
    for item in [left, right]:
        for key in ['overall_mae', 'weighted_mae']:
            require(isinstance(item[key], (int, float)) and math.isfinite(item[key]) and item[key] >= 0,
                    'Invalid recorded MAE: ' + key)
    for family in ['strata', 'forecast_quarters']:
        require(set(left[family]) == set(right[family]), 'Paired recorded group inventory differs')
        total = 0
        for key, a in left[family].items():
            b = right[family][key]
            require(type(a['rows']) is int and a['rows'] >= 0 and a['rows'] == b['rows'], 'Paired group rows differ')
            total += a['rows']
            for value in [a, b]:
                if value['rows'] == 0:
                    require(value['mae'] is None, 'Empty group must retain null MAE')
                else:
                    require(isinstance(value['mae'], (int, float)) and math.isfinite(value['mae'])
                            and value['mae'] >= 0, 'Nonempty group has invalid MAE')
        require(total == rows, 'Recorded group row totals differ from query rows')


def preparation_record(manifest):
    if manifest is None:
        return None
    fields = ['fit_rows', 'query_rows', 'fit_X_sha256', 'query_X_sha256',
              'fit_indices_content_sha256', 'query_indices_content_sha256',
              'fit_weight_values_sha256', 'feasibility_passed', 'feasibility_checks',
              'coverage', 'weight_summary', 'fold_reports',
              'diagnostic_domain_oof_auc_prior_corrected', 'diagnostic_domain_oof_log_loss_common_prior',
              'fit_group_fallback_counts', 'query_group_fallback_counts',
              'classifier_models_fitted', 'property_price_models_fitted',
              'query_X_used_for_weights', 'query_y_or_price_cache_used']
    return {key: copy.deepcopy(manifest[key]) for key in fields}


def make_stage(specification, stage, status, result, metadata, manifest):
    require(status in STATUS_LABELS, 'Unknown finalized stage status: ' + str(status))
    split = specification['split_contract'][stage]
    record = {
        'stage': stage, 'status': status, 'status_text': STATUS_LABELS[status],
        'query_dates': list(specification['splits'][stage]),
        'fit_rows': split['fit_rows'], 'query_rows': split['query_rows'],
        'price_evaluated': result is not None, 'metrics': None, 'decision': None,
        'model_summary': None, 'pure_X': preparation_record(manifest),
        'result_path': None, 'model_metadata_path': None, 'weight_manifest_path': None,
        'result_sha256': None, 'model_metadata_sha256': None, 'weight_manifest_sha256': None,
    }
    if manifest is not None:
        require(manifest['fit_rows'] == split['fit_rows'] and manifest['query_rows'] == split['query_rows'],
                'Prepared sample counts differ from planned rows')
        record['weight_manifest_path'] = f'outputs/round11/weights/{stage}/manifest.json'
    if result is None:
        require(metadata is None and status != 'price_evaluated', 'Missing completed price evidence')
        if status == 'weights_prepared_price_not_executed':
            require(manifest is not None and manifest['feasibility_passed'] is True, 'Missing prepared-but-unscored evidence')
        if status.startswith('not_executed'):
            require(manifest is None, 'An unexecuted preparation cannot have a manifest')
        if status == 'pure_X_feasibility_failed_price_not_evaluated':
            require(manifest is not None and manifest['feasibility_passed'] is False, 'Missing failed pure-X evidence')
        return record
    require(status == 'price_evaluated' and metadata is not None and manifest is not None
            and manifest['feasibility_passed'] is True, 'Price result lacks passed weight preparation')
    require(result['fit_rows'] == split['fit_rows'] and result['query_rows'] == split['query_rows'],
            'Recorded price counts differ from fixed window')
    validate_metrics(result['metrics'], split['query_rows'])
    require(result['decision'] == experiment.decide(stage, result['metrics']), 'Recorded decision differs')
    record.update({
        'metrics': copy.deepcopy(result['metrics']), 'decision': copy.deepcopy(result['decision']),
        'result_path': f'outputs/round11/evaluation/{stage}/results.json',
        'model_metadata_path': f'outputs/round11/models/{stage}/metadata.json',
        'model_summary': {key: copy.deepcopy(metadata[key]) for key in [
            'feature_count', 'categorical_feature_count', 'iterations', 'actual_iterations',
            'target_mode', 'target_formula', 'prediction_formula', 'sample_weight',
            'fit_sample_weight_values_sha256', 'fit_sample_weight_summary',
            'query_X_used_for_training_weights', 'query_labels_used',
            'query_used_for_geographic_state', 'fit_uses_eval_set', 'early_stopping_used',
            'reload_64_rows_verified', 'runner_elapsed_seconds']},
    })
    return record


def build_payload(specification, decision, stages, input_hashes, authority_hashes):
    require(set(stages) == set(STAGES), 'All three annual statuses must be explicit')
    actual_results = {stage: {'decision': value['decision']} for stage, value in stages.items()
                      if value['price_evaluated']}
    failed = [stage for stage in STAGES if stages[stage]['status'] == 'pure_X_feasibility_failed_price_not_evaluated']
    accepted, reason = experiment.final_outcome(actual_results, failed)
    require(decision['accepted'] is accepted and decision['stopping_reason'] == reason, 'Final disposition differs')
    require(decision['pure_X_feasibility_failure_stages'] == failed, 'Pure-X stopping stages differ')
    require(set(decision['stage_result_sha256']) == set(actual_results), 'Acceptance price result inventory differs')
    unevaluated = [stage for stage in STAGES if stage not in actual_results]
    require(decision['unevaluated_stages'] == unevaluated, 'Unevaluated stages differ')
    return {
        'round': 11, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'metric': 'MAE', 'unit': 'TWD',
        'status': 'backtest_accepted_pending_release' if accepted else 'rejected_keep_v3',
        'accepted': accepted, 'candidate': experiment.CANDIDATE, 'control': experiment.CONTROL,
        'selected_candidate': decision['selected_candidate'],
        'stopping_reason': reason, 'stopping_reason_text': STOP_LABELS[reason],
        'current_released_recommendation': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
        'new_submission_created': False, 'release_performed': False, 'kaggle_upload_performed': False,
        'stages': stages, 'unevaluated_stages': unevaluated,
        'formulas': copy.deepcopy(specification['formulas']), 'thresholds': copy.deepcopy(specification['thresholds']),
        'parameters': copy.deepcopy(specification['parameters']), 'feature_rules': copy.deepcopy(specification['feature_rules']),
        'domain_config': copy.deepcopy(specification['domain_config']),
        'target_group_weights': copy.deepcopy(specification['target_group_weights']),
        'query_X_used_for_training_weights': True, 'query_labels_used_for_training': False,
        'price_loader_may_deserialize_query_labels_but_does_not_pass_them_to_fit': True,
        'training_importance_weights_used_for_scoring': False,
        'external_property_answers_used': False, 'public_score_used_for_selection': False,
        'historically_untouched': False, 'private_score_known': False,
        'public_score_record': {'submission_v2.csv': 2883250.4138, 'submission_v3.csv': None,
                                'round11_candidate': None},
        'private_mae': None, 'source_input_sha256': dict(sorted(input_hashes.items())),
        'script_path': 'src/round11_report.py', 'script_sha256': input_hashes.get('src/round11_report.py'),
        **authority_hashes,
    }


def validate_backtest_review(proof, specification, decision, authorities, completed, expected_evidence, md_sha):
    require(proof['round'] == 11 and proof['status'] == 'passed', 'Independent backtest verification did not pass')
    require(proof['checked_stages'] == completed and set(proof['stage_evidence']) == set(completed),
            'Independent review checked a different completed-stage inventory')
    for key in ['plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256']:
        require(proof[key] == authorities[key], 'Independent review authority differs: ' + key)
    require(proof['candidate_accepted'] is decision['accepted'], 'Verification success was confused with candidate acceptance')
    require(proof['only_completed_stages_scored'] is True and proof['training_performed'] is False,
            'Independent review performed unauthorized work')
    require(proof['plan_source_sha256'] == specification['source_sha256'], 'Independent review source plan differs')
    for name, digest in specification['source_sha256'].items():
        require(proof['source_sha256'].get(name) == digest, 'Independent review omitted frozen source: ' + name)
    for stage, fields in expected_evidence.items():
        for key, digest in fields.items():
            require(proof['stage_evidence'][stage].get(key) == digest, 'Independent reviewed artifact differs: ' + stage + '/' + key)
    require(proof['report_sha256'] == md_sha, 'Independent human review differs')


def load_verified():
    specification = experiment.plan()
    frozen = experiment.locked(OUT / 'frozen_candidate.json')
    decision = experiment.locked(OUT / 'accepted_candidate.json')
    plan_sha = experiment.sha(OUT / 'candidate_plan.json')
    freeze_sha = experiment.sha(OUT / 'frozen_candidate.json')
    domain_sha = experiment.sha(OUT / 'domain_plan.json')
    require(decision['round'] == 11 and decision['plan_sha256'] == plan_sha
            and decision['frozen_candidate_sha256'] == freeze_sha
            and decision['domain_plan_sha256'] == domain_sha, 'Acceptance authority changed')
    require(frozen['round'] == 11 and frozen['plan_sha256'] == plan_sha
            and frozen['domain_plan_sha256'] == domain_sha, 'Development freeze authority changed')
    require(decision['candidate'] == experiment.CANDIDATE
            and decision['selected_candidate'] == frozen['selected_candidate'], 'Candidate identity differs')
    require(decision['source_sha256'] == frozen['source_sha256'] == specification['source_sha256'], 'Source closure changed')
    require(frozen['formulas'] == specification['formulas'] and frozen['thresholds'] == specification['thresholds']
            and frozen['parameters'] == specification['parameters'], 'Frozen rules changed')
    require(decision['new_submission_created'] is False and decision['no_reselection_or_parameter_change'] is True,
            'Unexpected release/reselection evidence')
    require(decision['currently_recommended_submission'] == 'outputs/submission_v3.csv', 'Existing release recommendation differs')
    expected_deployment = 'backtest_accepted_pending_final_release' if decision['accepted'] else 'keep_v3'
    require(decision['deployment_decision'] == expected_deployment, 'Acceptance/release distinction changed')
    inputs = {}
    def merge(mapping):
        for name, digest in mapping.items():
            require(name not in inputs or inputs[name] == digest, 'Conflicting source identity: ' + name)
            inputs[name] = digest
    def bind(path, receipt=False):
        path = Path(path)
        if receipt:
            experiment.locked(path)
        merge({experiment.rel(path): experiment.sha(path)})
        if receipt:
            merge({experiment.rel(path.with_suffix('.sha256')): experiment.sha(path.with_suffix('.sha256'))})
    merge(specification['source_sha256']); merge(specification['dependency_sha256'])
    merge(specification['final_dependency_sha256'])
    for mapping in specification['stage_dependency_sha256'].values():
        merge(mapping)
    merge(frozen['development_artifact_sha256']); merge(decision['preparation_evidence_sha256'])
    for name in ['candidate_plan.json', 'domain_plan.json', 'frozen_candidate.json', 'accepted_candidate.json']:
        bind(OUT / name, receipt=True)
    bind(Path(__file__).resolve())
    bind(ROOT / 'runs/versions.json')
    for document in experiment.REQUIRED_DOCUMENTS:
        bind(ROOT / document)
    stages = {}; results = {}; prepared_hashes = {}
    require(set(decision['stage_status']) == set(STAGES), 'Acceptance lacks stage statuses')
    for stage in STAGES:
        # A partial run is an error, not evidence that a stage was never executed.
        for parent, pattern in [(OUT / 'weights', '_' + stage + '_*'),
                                (OUT / 'models', '.' + stage + '-*'),
                                (OUT / 'evaluation', '.' + stage + '-*')]:
            require(not list(parent.glob(pattern)), 'Unresolved partial stage artifacts: ' + stage)
        result = metadata = manifest = None
        weight_path = OUT / 'weights' / stage / 'manifest.json'
        result_path = OUT / 'evaluation' / stage / 'results.json'
        model_path = OUT / 'models' / stage / 'metadata.json'
        if weight_path.exists():
            manifest, weight_inputs = experiment.verify_weight_manifest(stage, specification, require_pass=False)
            merge(weight_inputs); prepared_hashes[stage] = experiment.sha(weight_path)
        else:
            require(not weight_path.parent.exists() and not weight_path.with_suffix('.sha256').exists(), 'Incomplete weight preparation')
        if stage in decision['stage_result_sha256']:
            result = experiment.read_result(stage); results[stage] = result
            require(experiment.sha(result_path) == decision['stage_result_sha256'][stage], 'Accepted result SHA differs')
            require(result['domain_plan_sha256'] == domain_sha and result['v3_control_replayed_all_rows'] is True
                    and result['new_model_replayed_all_rows'] is True
                    and result['training_importance_weights_used_for_scoring'] is False, 'Result evidence differs')
            require(result['target_group_weights'] == specification['target_group_weights']
                    and result['public_score_used'] is False and result['historically_untouched'] is False
                    and result['query_labels_used_for_training'] is False, 'Recorded evaluation/data-use policy differs')
            metadata = experiment.locked(model_path)
            require(manifest is not None, 'Price result has no weight preparation')
            contract = {'phase': stage, **{key: manifest[key] for key in [
                'fit_X_sha256', 'query_X_sha256', 'fit_indices_content_sha256', 'query_indices_content_sha256']},
                'weights_values_sha256': manifest['fit_weight_values_sha256'], 'domain_plan_sha256': domain_sha,
                'preparation_manifest_sha256': experiment.sha(weight_path),
                'query_X_used_for_weights': True, 'query_labels_used_for_weights': False}
            expected = {'stage': stage, 'plan_sha256': plan_sha, 'source_sha256': specification['source_sha256'],
                        'domain_plan_sha256': domain_sha, 'preparation_manifest_sha256': experiment.sha(weight_path),
                        'weight_contract': contract, 'fit_sample_weight_values_sha256': manifest['fit_weight_values_sha256'],
                        'outer_fit_rows': result['fit_rows'], 'query_rows': result['query_rows']}
            for key in ['fit_X_sha256', 'query_X_sha256', 'fit_indices_content_sha256', 'query_indices_content_sha256']:
                expected[key] = manifest[key]
            experiment.validate_model_contract(metadata, specification, expected)
            require(result['weight_contract'] == contract and result['preparation_manifest_sha256'] == experiment.sha(weight_path),
                    'Result weight/whole-query identity differs')
            require(metadata['frozen_candidate_sha256'] == (None if stage == 'development' else freeze_sha), 'Model freeze differs')
            require(metadata['reload_64_rows_verified'] is True and metadata['prediction_units'] == 'TWD', 'Model reload/units evidence differs')
            bind(result_path, receipt=True); bind(model_path, receipt=True)
            merge(result['input_artifact_sha256']); merge(result['output_artifact_sha256'])
            for name, digest in metadata['artifact_sha256'].items():
                require(Path(name).name == name, 'Unsafe model artifact name')
                merge({experiment.rel(model_path.parent / name): digest})
        else:
            require(not result_path.parent.exists() and not model_path.parent.exists(), 'Unreported price artifacts exist')
        record = make_stage(specification, stage, decision['stage_status'][stage], result, metadata, manifest)
        for field, path in [('result', result_path), ('model_metadata', model_path), ('weight_manifest', weight_path)]:
            if record[field + '_path'] is not None:
                record[field + '_sha256'] = experiment.sha(path)
        stages[stage] = record
    require('development' in results and frozen['result_sha256'] == experiment.sha(OUT / 'evaluation/development/results.json'),
            'Development freeze result changed')
    require(frozen['passed'] is results['development']['decision']['passed']
            and frozen['later_stages_allowed'] is frozen['passed']
            and frozen['selected_candidate'] == results['development']['decision']['selected_candidate'], 'Development decision changed')
    require(decision['stage_preparation_manifest_sha256'] == {k: v for k, v in prepared_hashes.items() if k != 'development'},
            'Finalization preparation inventory differs')
    authorities = {'plan_sha256': plan_sha, 'frozen_candidate_sha256': freeze_sha,
                   'accepted_candidate_sha256': experiment.sha(OUT / 'accepted_candidate.json'),
                   'domain_plan_sha256': domain_sha}
    review_path = OUT / 'backtest_review.json'
    review_md = ROOT / 'reports/round11_backtest_review.md'
    proof = experiment.locked(review_path)
    completed = [stage for stage in STAGES if stage in results]
    expected_evidence = {}
    for stage in completed:
        model_folder = OUT / 'models' / stage; result_folder = OUT / 'evaluation' / stage
        paths = {'results_sha256': result_folder / 'results.json',
                 'model_metadata_sha256': model_folder / 'metadata.json',
                 'model_sha256': model_folder / 'model.cbm',
                 'feature_state_sha256': model_folder / 'feature_state.json',
                 'component_predictions_sha256': model_folder / 'predictions.npy',
                 'control_predictions_sha256': result_folder / (experiment.CONTROL + '.npy'),
                 'candidate_predictions_sha256': result_folder / (experiment.CANDIDATE + '.npy')}
        expected_evidence[stage] = {key: experiment.sha(value) for key, value in paths.items()}
    validate_backtest_review(proof, specification, decision, authorities, completed, expected_evidence,
                             experiment.sha(review_md))
    bind(review_path, receipt=True); bind(review_md); merge(proof['source_sha256'])
    experiment.c6.verify_hashes(inputs)
    payload = build_payload(specification, decision, stages, inputs, authorities)
    payload['independent_backtest_review'] = {
        'status': proof['status'], 'candidate_accepted': proof['candidate_accepted'],
        'checked_stages': completed, 'path': experiment.rel(review_path),
        'sha256': experiment.sha(review_path), 'report_path': experiment.rel(review_md),
        'report_sha256': experiment.sha(review_md)}
    return payload


def render(payload):
    accepted = payload['accepted']
    headline = ('唯一候選通過三年度固定回測門檻；尚待最終訓練、完整重播與另行發布，現在仍推薦 v3。'
                if accepted else '本輪未採用；維持已發布的 submission_v3.csv，沒有產生新版提交檔。')
    lines = ['# 第十一輪結果：純 X 分布權重', '', '**' + headline + '**', '',
             '判定原因：' + payload['stopping_reason_text'] + '。', '',
             '本報告只整理已保存並核對收據的決策與數值，沒有重新訓練、重算逐列誤差、選型、上傳 Kaggle 或發布模型。', '',
             '## 唯一候選與實際資料處理', '',
             '- 沿用原始 3 列嵌入表頭清理、train 的 529 列去重及 full 的 28 列去重映射。既有去重使用含 target 的歷史欄位；本輪純 X 權重步驟重用已保存索引，不讀價格重新去重。',
             '- 原 87 個房屋欄加 8 個地理欄，維持 95 欄／23 類別、值、dtype、順序與缺值；沒有新增 POI、備註或價格特徵。',
             '- 權重器只使用事前固定的 6 個結構／地區類別及 10 個數值來源。建案／地址只用於隔離群組，不作 domain 分類器特徵；日期不進權重器。',
             '- 固定五折建案共同隔離：同群 fit 與 query 均在同一折；每折前處理只學該折 source 訓練部分。固定 logistic 估本期 query 對 fit 的分布比，使用折內樣本先驗修正。',
             '- 比率在 log 空間截於 0.25–4，再以完整 source 的平均值正常化為 1，沒有第二次截斷。最終權重可能超出先前裁切界限，以下照實顯示。',
             '- CatBoost Pool 使用準備完成且逐列對齊的 float64 重要性權重；raw y/1e6，2,999 輪、seed 42，其餘 R6 參數不變，不使用 eval_set 或 early stopping。',
             '- 只替換 v3 中 37.5% 的地理 raw CatBoost；另 37.5% 原 raw CatBoost 與 25% 原 unit LightGBM 不變，同一保存的 reference／trend 只套一次。',
             '- 原地理 feature state 仍由 fit X 建立；訓練權重使用整批無標籤 query X，因此屬傳導式方法。價格 loader 可能反序列化 query labels，fit 立即捨棄且不傳給權重器或模型；不能宣稱價格流程從未載入 query labels。',
             '- 價格結果取自未取整 TWD 的既有評分。密度訓練權重不乘到 query MAE；原三組 test-X 評分權重保持不變。', '',
             '## 三年度價格結果', '',
             'MAE 單位為新臺幣元；改善率為正代表誤差下降。「—」代表沒有該項數值，不代表零或通過。', '',
             '| 窗口／query 日期 | fit／query 筆數 | v3 整體 MAE | 候選整體 MAE | 改善 | v3 加權 MAE | 候選加權 MAE | 改善 | 實際狀態 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for stage in STAGES:
        value = payload['stages'][stage]
        dates = '～'.join(value['query_dates'])
        prefix = f"| {STAGE_LABELS[stage]} {dates} | {value['fit_rows']:,}／{value['query_rows']:,} |"
        if not value['price_evaluated']:
            lines.append(prefix + ' — | — | — | — | — | — | ' + value['status_text'] + ' |')
            continue
        a, b = value['metrics'][payload['control']], value['metrics'][payload['candidate']]
        lines.append(prefix + f" {number(a['overall_mae'])} | {number(b['overall_mae'])} | {percent(improvement(a['overall_mae'], b['overall_mae']),4)} | {number(a['weighted_mae'])} | {number(b['weighted_mae'])} | {percent(improvement(a['weighted_mae'], b['weighted_mae']),4)} | {'門檻通過' if value['decision']['passed'] else '門檻未通過'} |")
    for stage in STAGES:
        value = payload['stages'][stage]
        if not value['price_evaluated']:
            continue
        a, b = value['metrics'][payload['control']], value['metrics'][payload['candidate']]
        lines += ['', '### ' + STAGE_LABELS[stage] + '：三組與預測季度', '',
                  '| 群組 | 筆數 | 納入大組限制 | v3 MAE | 候選 MAE | 改善 |', '|---|---:|---|---:|---:|---:|']
        for family in ['strata', 'forecast_quarters']:
            keys = ['seen', 'new_nonmissing', 'missing'] if family == 'strata' else sorted(a[family])
            for key in keys:
                item = a[family][key]
                other = b[family][key]
                large = ('是' if item['rows'] >= payload['thresholds']['large_stratum_min_rows'] else '否') if family == 'strata' else '季度規則另列'
                lines.append(f"| {key} | {item['rows']:,} | {large} | {number(item['mae'])} | {number(other['mae'])} | {percent(improvement(item['mae'],other['mae']),4)} |")
        lines += ['', '該期固定条件：']
        for key, passed in sorted(value['decision']['checks'].items()):
            lines.append(f"- `{key}`：{'通過' if passed else '未通過'}。")
    lines += ['', '## 純 X 可行性與權重診斷', '',
              '下列都是已保存的權重準備摘要，不是房價分數。AUC／log-loss 沒有選型門檻；高 AUC 不證明密度比準確。純 X 失敗時不產生價格分數，也不以缺檔或程式錯誤冒充方法拒絕。', '',
              '| 窗口 | 地區支持 | 結構支持 | 裁切列比例 | 列 ESS／fit 列數 | 群 ESS／原群 ESS | 最大群權重占比 | OOF AUC | 七項條件 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for stage in STAGES:
        pure = payload['stages'][stage]['pure_X']
        if pure is None:
            lines.append(f'| {STAGE_LABELS[stage]} | — | — | — | — | — | — | — | 未執行 |')
            continue
        summary, coverage = pure['weight_summary'], pure['coverage']
        entries = [coverage['region']['adequate_query_fraction'], coverage['structure']['adequate_query_fraction'],
                   summary['clipped_fraction'], summary['row_ess_fraction'], summary['group_ess_ratio_to_uniform'],
                   summary['largest_group_weight_share']]
        lines.append('| ' + STAGE_LABELS[stage] + ' | ' + ' | '.join(percent(100 * value) for value in entries)
                     + ' | ' + number(pure['diagnostic_domain_oof_auc_prior_corrected'], 4)
                     + ' | ' + ('通過' if pure['feasibility_passed'] else '未通過') + ' |')
    for stage in STAGES:
        pure = payload['stages'][stage]['pure_X']
        if pure is None:
            continue
        summary = pure['weight_summary']
        quantiles = summary['normalized_weight_quantiles']
        maximum = quantiles['1.0'] if '1.0' in quantiles else quantiles['1']
        lines += ['', f"{STAGE_LABELS[stage]}最終權重 min／median／max：{number(quantiles['0'],6)}／{number(quantiles['0.5'],6)}／{number(maximum,6)}；均值 {number(summary['normalized_weight_mean'],6)}；OOF common-prior log-loss {number(pure['diagnostic_domain_oof_log_loss_common_prior'],6)}。",
                  '純 X 條件：' + '；'.join(f"`{key}` {'通過' if passed else '未通過'}" for key, passed in sorted(pure['feasibility_checks'].items())) + '。', '']
    review = payload.get('independent_backtest_review')
    if review is not None:
        lines += ['', '## 獨立回測核對', '',
                  '已完成窗口的模型重載、整份 query 重播、既有成績與列索引由 [獨立核對](round11_backtest_review.md) 驗證；[核對收據](../outputs/round11/backtest_review.json) 的 status=passed 表示證據核對成功，並不代表候選被採用。',
                  '實際核對窗口：' + '、'.join(STAGE_LABELS[s] for s in review['checked_stages']) + '。']
    t = payload['thresholds']
    lines += ['', '## 固定接受條件與限制', '',
              f"開發期整體及三組加權 MAE 均須改善至少 {100*t['development_min_overall_improvement']:g}%／{100*t['development_min_weighted_improvement']:g}%；非空季度退步不超過 {100*t['development_max_quarter_deterioration']:g}%。確認期整體／加權退步不超過 {100*t['confirmation_max_overall_deterioration']:g}%／{100*t['confirmation_max_weighted_deterioration']:g}%，且至少 {t['confirmation_min_improved_quarters']} 個非空季度嚴格改善。輔助期整體／加權退步不超過 {100*t['auxiliary_max_overall_deterioration']:g}%／{100*t['auxiliary_max_weighted_deterioration']:g}%；其季度只作診斷。",
              f"三期都限制不少於 {t['large_stratum_min_rows']} 列的既有建案 stratum 退步不超過 {100*t['max_large_stratum_deterioration']:g}%。這裡的大組不是任選城市或 domain 隔離群。零列組的 MAE 保持 null，不以 0 代替。",
              '只有本輪開發期價格門檻通過並凍結，才允許後兩期準備權重與評估；後期只確認或否決，沒有重新選候選、比例或門檻。正式規約保存在來源鏈中。', '',
              '這些年度窗口在歷史輪次已使用，不能稱為盲測或完全未見確認集。單一事前候選、OOF 群隔離、工程閘與固定門檻仍不能消除跨輪選型偏差。訓練權重依 query 清單而變；地區支援、條件房價、時間趨勢與未見建案品質仍可能轉移。裁切不能創造原本不存在的資料支持。', '',
              '既有道路代表位置不等於住宅門牌，地理覆蓋有年度及城市差異。本輪沒有新增外部成交答案、沒有到內政部對答案，也沒有按 Public 分數調參。', '',
              '使用者回報的 submission_v2.csv Public MAE 為 **2,883,250.4138**。v3 與本輪候選沒有已知 Public 成績；Private 未知。本機歷史門檻通過不保證 Private 改善。本報告未上傳 Kaggle、未執行全量 fit 或 release，也未產生新 CSV。', '',
              '## 證據與重現', '',
              '- [事前固定規約](round11_domain_plan.md)、[接線與資料映射](round11_pipeline_design.md)、[獨立價格流程審查](round11_price_pipeline_review.md)。',
              '- [機器可讀結果](round11_results.json)、[價格計畫](../outputs/round11/candidate_plan.json)、[開發凍結](../outputs/round11/frozen_candidate.json)、[最終判定](../outputs/round11/accepted_candidate.json)。',
              '- [現行推薦提交檔](../outputs/submission_v3.csv)。原 v2／v3、模型與版本表保持原樣。',
              '- 報告程式只核對既有收據、來源及數值，命令為 `.venv\\Scripts\\python.exe src\\round11_report.py generate`；已存在的報告禁止覆寫。合成格式檢查為同命令改用 `self-check`。', '',
              '主要契約 SHA256：', '',
              '| 契約 | SHA256 |', '|---|---|']
    for key in ['plan_sha256', 'domain_plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256']:
        lines.append(f"| {key} | `{payload[key]}` |")
    lines += ['', '全部實際讀取與依賴的 root-relative 路徑／SHA 收錄於機器報告 `source_input_sha256`；數值由同一 payload 產生，未另行估分。', '']
    return '\n'.join(lines).replace('条件', '條件')


def generate():
    for path in [REPORT, RESULTS, RESULTS.with_suffix('.sha256')]:
        require(not path.exists(), 'Refusing to overwrite existing report: ' + str(path))
    payload = load_verified()
    markdown = render(payload)
    payload['report_path'] = 'reports/round11_report.md'
    payload['report_markdown_sha256'] = hashlib.sha256(markdown.encode('utf-8')).hexdigest()
    require(render(payload) == markdown, 'Human/machine payload rendering differs')
    experiment.c6.verify_hashes(payload['source_input_sha256'])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(markdown)
    experiment.locked_write(RESULTS, payload)
    require(experiment.sha(REPORT) == payload['report_markdown_sha256'], 'Written human report differs')
    require(experiment.locked(RESULTS) == payload, 'Written machine report differs')
    experiment.c6.verify_hashes(payload['source_input_sha256'])
    print(json.dumps({'status': payload['status'], 'accepted': payload['accepted'],
                      'report_sha256': experiment.sha(REPORT), 'results_sha256': experiment.sha(RESULTS),
                      'recommendation': 'v3', 'new_submission_created': False}, ensure_ascii=False))
    return payload


def self_check():
    """Formatting/absence checks only; no real authority or result files are read."""
    checks = []
    def check(name, condition):
        require(condition, 'Synthetic report check failed: ' + name); checks.append(name)
    def reject(name, operation):
        try:
            operation()
        except (RuntimeError, ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError('Expected report rejection: ' + name)
    specification = {
        'splits': {k: list(v) for k, v in experiment.c6.SPLITS.items()},
        'split_contract': {k: {'fit_rows': 10, 'query_rows': 3} for k in STAGES},
        'formulas': experiment.FORMULAS, 'thresholds': experiment.THRESHOLDS,
        'parameters': {'iterations': 2999}, 'feature_rules': {'feature_count': 95},
        'domain_config': {'synthetic': True}, 'target_group_weights': {'seen': .5, 'new_nonmissing': .5, 'missing': 0.},
    }
    base = {'overall_mae': 100., 'weighted_mae': 100.,
            'strata': {'seen': {'rows': 2, 'mae': 100.}, 'new_nonmissing': {'rows': 1, 'mae': 100.},
                       'missing': {'rows': 0, 'mae': None}},
            'forecast_quarters': {key: {'rows': 1 if key != 'Q4' else 0, 'mae': 100. if key != 'Q4' else None}
                                  for key in ['Q1', 'Q2', 'Q3', 'Q4']}}
    candidate = copy.deepcopy(base)
    candidate['overall_mae'] = candidate['weighted_mae'] = 98.
    for family in ['strata', 'forecast_quarters']:
        for item in candidate[family].values():
            if item['rows']:
                item['mae'] = 98.
    metrics = {experiment.CONTROL: base, experiment.CANDIDATE: candidate}
    pure = {key: {} for key in ['fit_group_fallback_counts', 'query_group_fallback_counts']}
    pure.update({
        'fit_rows': 10, 'query_rows': 3, 'fit_X_sha256': 'a'*64, 'query_X_sha256': 'b'*64,
        'fit_indices_content_sha256': 'c'*64, 'query_indices_content_sha256': 'd'*64,
        'fit_weight_values_sha256': 'e'*64, 'feasibility_passed': True, 'feasibility_checks': {'synthetic': True},
        'coverage': {key: {'adequate_query_fraction': 1.} for key in ['region', 'structure']},
        'weight_summary': {'clipped_fraction': 0., 'row_ess_fraction': 1., 'group_ess_ratio_to_uniform': 1.,
                           'largest_group_weight_share': .1, 'normalized_weight_quantiles': {'0': .1, '0.5': 1., '1': 2.},
                           'normalized_weight_mean': 1.},
        'fold_reports': [], 'diagnostic_domain_oof_auc_prior_corrected': .5,
        'diagnostic_domain_oof_log_loss_common_prior': .693, 'classifier_models_fitted': 5,
        'property_price_models_fitted': 0, 'query_X_used_for_weights': True, 'query_y_or_price_cache_used': False,
    })
    metadata = {'feature_count': 95, 'categorical_feature_count': 23, 'iterations': 2999, 'actual_iterations': 2999,
                'target_mode': 'raw', 'target_formula': 'raw TWD / 1e6', 'prediction_formula': 'max(raw * 1e6, 0)',
                'sample_weight': 'importance_clip_then_mean_one', 'fit_sample_weight_values_sha256': 'e'*64,
                'fit_sample_weight_summary': {}, 'query_X_used_for_training_weights': True, 'query_labels_used': False,
                'query_used_for_geographic_state': False, 'fit_uses_eval_set': False, 'early_stopping_used': False,
                'reload_64_rows_verified': True, 'runner_elapsed_seconds': 1.0}
    authorities = {key: 'f'*64 for key in ['plan_sha256', 'domain_plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256']}
    stages = {}
    for stage in STAGES:
        result = {'fit_rows': 10, 'query_rows': 3, 'metrics': copy.deepcopy(metrics),
                  'decision': experiment.decide(stage, metrics)}
        stages[stage] = make_stage(specification, stage, 'price_evaluated', result, metadata, pure)
    decision = {'accepted': True, 'selected_candidate': experiment.CANDIDATE,
                'stopping_reason': 'three_stage_price_gates_passed', 'pure_X_feasibility_failure_stages': [],
                'stage_result_sha256': {stage: 'x' for stage in STAGES}, 'unevaluated_stages': []}
    payload = build_payload(specification, decision, stages, {'src/round11_report.py': 'f'*64}, authorities)
    text = render(payload)
    check('accepted_history_still_recommends_existing_v3', payload['current_released_recommendation'] == 'v3'
          and payload['new_submission_created'] is False and '尚待最終訓練' in text)
    check('all_three_years_quarters_and_strata_present', all(STAGE_LABELS[s] in text for s in STAGES)
          and 'Q4' in text and 'new_nonmissing' in text)
    check('public_record_and_private_unknown', payload['public_score_record']['submission_v2.csv'] == 2883250.4138
          and payload['public_score_record']['submission_v3.csv'] is None and payload['private_mae'] is None)
    check('empty_stratum_preserves_null', payload['stages']['development']['metrics'][experiment.CONTROL]['strata']['missing']['mae'] is None)
    check('render_JSON_roundtrip_identical', render(json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))) == text)
    check('honest_loader_and_historical_reuse_language', '不能宣稱價格流程從未載入' in text and '不能稱為盲測' in text)
    check('relative_gain_from_recorded_metrics', improvement(100, 98) == 2.0 and improvement(None, None) is None)
    failed = copy.deepcopy(pure); failed['feasibility_passed'] = False; failed['feasibility_checks'] = {'synthetic': False}
    stopped = copy.deepcopy(stages)
    stopped['confirmation'] = make_stage(specification, 'confirmation', 'pure_X_feasibility_failed_price_not_evaluated', None, None, failed)
    stopped['auxiliary'] = make_stage(specification, 'auxiliary', 'not_executed_after_pure_X_stop', None, None, None)
    stop_decision = {'accepted': False, 'selected_candidate': experiment.CANDIDATE,
                     'stopping_reason': 'pure_X_feasibility_failed', 'pure_X_feasibility_failure_stages': ['confirmation'],
                     'stage_result_sha256': {'development': 'x'}, 'unevaluated_stages': ['confirmation', 'auxiliary']}
    stop_payload = build_payload(specification, stop_decision, stopped, {}, authorities)
    stop_text = render(stop_payload)
    check('late_pure_X_failure_has_no_invented_price_metrics', stop_payload['stages']['confirmation']['metrics'] is None
          and stop_payload['stages']['auxiliary']['metrics'] is None and '未訓練或評分價格候選' in stop_text)
    check('unexecuted_weight_stage_is_null', stop_payload['stages']['auxiliary']['pure_X'] is None)
    rejected_stages = copy.deepcopy(stopped)
    rejected_metrics = copy.deepcopy(metrics)
    rejected_metrics[experiment.CANDIDATE]['overall_mae'] = 102.
    rejected_metrics[experiment.CANDIDATE]['weighted_mae'] = 102.
    rejected_result = {'fit_rows': 10, 'query_rows': 3, 'metrics': rejected_metrics,
                       'decision': experiment.decide('development', rejected_metrics)}
    rejected_stages['development'] = make_stage(specification, 'development', 'price_evaluated', rejected_result, metadata, pure)
    for stage in ['confirmation', 'auxiliary']:
        rejected_stages[stage] = make_stage(specification, stage, 'not_executed_after_development_rejection', None, None, None)
    rejected_decision = dict(stop_decision, selected_candidate=experiment.CONTROL,
                             stopping_reason='development_price_gates_failed', pure_X_feasibility_failure_stages=[])
    rejected_payload = build_payload(specification, rejected_decision, rejected_stages, {}, authorities)
    check('development_rejection_keeps_later_metrics_absent', rejected_payload['accepted'] is False
          and rejected_payload['stages']['confirmation']['metrics'] is None and '本輪未採用' in render(rejected_payload))
    reject('prepared_status_cannot_fake_absent_manifest', lambda: make_stage(
        specification, 'confirmation', 'weights_prepared_price_not_executed', None, None, None))
    check('human_report_has_actual_newlines', text.count(chr(10)) > 40 and text.startswith('# 第十一輪結果'))
    table_start = text.index('| 窗口 | 地區支持')
    table_end = text.index('開發期最終權重', table_start)
    table = text[table_start:table_end]
    check('pure_X_table_has_three_contiguous_rows', all('| ' + STAGE_LABELS[s] + ' |' in table for s in STAGES))
    reject('missing_actual_result_cannot_be_accepted', lambda: build_payload(
        specification, dict(stop_decision, accepted=True), stopped, {}, authorities))
    reject('failed_pure_X_status_needs_evidence', lambda: make_stage(
        specification, 'confirmation', 'pure_X_feasibility_failed_price_not_evaluated', None, None, None))
    reject('price_evaluated_status_needs_result', lambda: make_stage(
        specification, 'confirmation', 'price_evaluated', None, None, pure))
    broken = copy.deepcopy(metrics); broken[experiment.CANDIDATE]['strata']['missing']['mae'] = 0.
    reject('empty_group_zero_is_not_valid_MAE', lambda: validate_metrics(broken, 3))
    broken = copy.deepcopy(metrics); broken[experiment.CANDIDATE]['overall_mae'] = float('nan')
    reject('nonfinite_MAE_rejected', lambda: validate_metrics(broken, 3))
    review_spec = {'source_sha256': {'src/frozen_synthetic.py': 'a'*64}}
    review_decision = {'accepted': False}
    review_authorities = {key: 'b'*64 for key in ['plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256']}
    evidence = {'development': {'results_sha256': 'c'*64}}
    proof = {'round': 11, 'status': 'passed', 'checked_stages': ['development'],
             'candidate_accepted': False, 'only_completed_stages_scored': True, 'training_performed': False,
             'source_sha256': dict(review_spec['source_sha256']), 'plan_source_sha256': review_spec['source_sha256'],
             'stage_evidence': evidence, 'report_sha256': 'd'*64, **review_authorities}
    validate_backtest_review(proof, review_spec, review_decision, review_authorities, ['development'], evidence, 'd'*64)
    check('successful_verification_does_not_imply_candidate_acceptance', proof['status'] == 'passed' and not proof['candidate_accepted'])
    reject('review_of_unperformed_stage_rejected', lambda: validate_backtest_review(
        dict(proof, checked_stages=['development','confirmation']), review_spec, review_decision,
        review_authorities, ['development'], evidence, 'd'*64))
    reject('review_cannot_change_acceptance', lambda: validate_backtest_review(
        dict(proof, candidate_accepted=True), review_spec, review_decision, review_authorities,
        ['development'], evidence, 'd'*64))
    reject('review_wrong_report_sha_rejected', lambda: validate_backtest_review(
        proof, review_spec, review_decision, review_authorities, ['development'], evidence, 'e'*64))
    return {'synthetic_only': True, 'all_checks_passed': True, 'checks': checks, 'count': len(checks),
            'real_data_or_new_scores_read': False, 'report_files_written': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'self-check'])
    args = parser.parse_args()
    if args.command == 'self-check':
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))
    else:
        generate()
