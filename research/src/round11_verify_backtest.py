"""Independent read-only replay of the finalized R11 development rejection.
No model fitting, classifier replay, later-window scoring, or artifact mutation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round11'
CONTROL = 'v3_control'
CANDIDATE = 'importance_geo_blend'
DRIVER_SHA = 'f66e0347bb0ccc041c8c3afe1065e0d3314e5c3cc9d7a83b4f7e740468757be8'
PLAN_SHA = 'a65e15819942b2fb579d8cf37e5234dd87017b77b4744066bc2b3fe0bc906fff'
RESULT_SHA = '12eccfa1593ea4654bbefdadbf1140e3cc49132864859784c9b1c1d6f9fa64a0'
FREEZE_SHA = '0b9b67d61d5abd2663d5bdc61d9d922f52ae63f0c74f3dda13f42933c6fad884'
PREVIOUS_INCOMPLETE_AUDIT_SOURCE_SHA = '16377e786eaddabc4243ac0a43e667e99ed2075e82ede704efb84d76973570b9'
GROUPS = ('seen', 'new_nonmissing', 'missing')
QUARTERS = {'Q1': [0, 91], 'Q2': [92, 183], 'Q3': [184, 275], 'Q4': [276, None]}
THRESHOLDS = {
    'development_min_overall_improvement': .01, 'development_min_weighted_improvement': .01,
    'development_max_quarter_deterioration': .05, 'large_stratum_min_rows': 500,
    'max_large_stratum_deterioration': .05, 'confirmation_max_overall_deterioration': .0,
    'confirmation_max_weighted_deterioration': .0, 'confirmation_min_improved_quarters': 2,
    'auxiliary_max_overall_deterioration': .03, 'auxiliary_max_weighted_deterioration': .03,
    'auxiliary_quarters_diagnostic_only': True,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def norm(values):
    values = values.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
    return values.mask(values.isin(['', 'nan', 'None', '<NA>', 'NaT']), '__MISSING__')


def labels(fit, query):
    def keys(frame):
        project = norm(frame['project_name'])
        return norm(frame['address_city']) + '|' + norm(frame['address_district']) + '|' + project, project.eq('__MISSING__')
    fit_key, fit_missing = keys(fit)
    query_key, query_missing = keys(query)
    known = set(fit_key[~fit_missing])
    return np.where(query_missing, 'missing', np.where(query_key.isin(known), 'seen', 'new_nonmissing'))


def scoring_weights(data):
    columns = ['address_city', 'address_district', 'project_name']
    fit = pd.concat([data['X_train'][columns], data['X_valid'][columns]], ignore_index=True)
    group = labels(fit, data['X_test'])
    return {key: float(np.mean(group == key)) for key in GROUPS}


def independent_score(y, prediction, fit, query, start, weights):
    y, prediction = np.asarray(y, dtype='float64'), np.asarray(prediction, dtype='float64')
    require(y.shape == prediction.shape == (len(query),) and len(y) > 0, 'Unaligned score arrays')
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), 'Nonfinite score arrays')
    require(set(weights) == set(GROUPS) and all(np.isfinite(v) and v >= 0 for v in weights.values()), 'Invalid scoring weights')
    require(abs(sum(weights.values()) - 1.) < 1e-12, 'Scoring weights do not sum to one')
    errors = np.abs(y - prediction)
    group = labels(fit, query)
    horizon = query['trade_day_index'].to_numpy(dtype='float64') - (pd.Timestamp(start) - pd.Timestamp('2000-01-01')).days
    require(np.isfinite(horizon).all() and (horizon >= 0).all(), 'Invalid forecast horizon')
    month = pd.to_datetime(query['trade_day_index'], unit='D', origin='2000-01-01').dt.strftime('%Y-%m').to_numpy()
    city = norm(query['address_city']).to_numpy()
    def part(mask):
        n = int(np.sum(mask))
        return {'rows': n, 'mae': float(errors[mask].mean()) if n else None}
    strata = {key: part(group == key) for key in GROUPS}
    weighted = 0.0
    for key, weight in weights.items():
        if weight > 0:
            require(strata[key]['rows'] > 0, 'Positive scoring weight has no stratum rows')
            weighted += weight * strata[key]['mae']
    quarters = {}
    for name, (low, high) in QUARTERS.items():
        mask = horizon >= low
        if high is not None:
            mask &= horizon <= high
        quarters[name] = {**part(mask), 'horizon_days': [low, high]}
    require(sum(v['rows'] for v in quarters.values()) == len(query), 'Quarter masks do not partition rows')
    return {'rows': len(errors), 'overall_mae': float(errors.mean()), 'weighted_mae': float(weighted),
            'strata': strata, 'forecast_quarters': quarters,
            'months': {str(key): part(month == key) for key in sorted(set(month))},
            'cities': {str(key): part(city == key) for key in sorted(set(city))}}


def independent_development_decision(metrics):
    require(set(metrics) == {CONTROL, CANDIDATE}, 'Unexpected candidate set')
    base, value = metrics[CONTROL], metrics[CANDIDATE]
    improved = [q for q, item in base['forecast_quarters'].items()
                if item['rows'] and value['forecast_quarters'][q]['mae'] < item['mae']]
    checks = {
        'overall_improves_at_least_1pct': value['overall_mae'] <= .99 * base['overall_mae'],
        'weighted_improves_at_least_1pct': value['weighted_mae'] <= .99 * base['weighted_mae'],
        'quarters_deteriorate_at_most_5pct': all(value['forecast_quarters'][q]['mae'] <= 1.05 * item['mae']
                                               for q, item in base['forecast_quarters'].items() if item['rows']),
        'large_strata_deteriorate_at_most_5pct': all(value['strata'][g]['mae'] <= 1.05 * item['mae']
                                                  for g, item in base['strata'].items() if item['rows'] >= 500),
    }
    passed = all(checks.values())
    return {'passed': passed, 'selected_candidate': CANDIDATE if passed else CONTROL,
            'checks': checks, 'improved_quarters': improved, 'candidate_reselection_performed': False}


def independent_blend(old, geographic, unit, reference, query, apply_reference):
    # Preserve the frozen arithmetic order as well as its 37.5/37.5/25 meaning.
    base = .75 * (.5 * np.asarray(old, dtype='float64') + .5 * np.asarray(geographic, dtype='float64')) + .25 * np.asarray(unit, dtype='float64')
    result = np.asarray(apply_reference(reference, query, base, variant='trend'), dtype='float64')
    require(result.shape == (len(query),) and np.isfinite(result).all() and (result >= 0).all(), 'Invalid blended prices')
    return result


def compare_vector(saved, replay, name):
    require(saved.shape == replay.shape and np.isfinite(replay).all(), name + ' shape/finite mismatch')
    require(np.allclose(saved, replay, rtol=1e-12, atol=1e-5), name + ' full-query replay mismatch')
    return {'rows': len(saved), 'bitwise_equal': bool(np.array_equal(saved.view('uint64'), replay.view('uint64'))),
            'maximum_absolute_difference_twd': float(np.max(np.abs(saved - replay))),
            'rtol': 1e-12, 'atol_twd': 1e-5}


def validate_category_counts(original_count, augmented_count):
    require(original_count == 21, 'Original 87 features require 21 categorical columns')
    require(augmented_count == 23, 'Augmented 95 features require 23 categorical columns')


def self_check():
    import copy
    checks = []
    def check(name, truth):
        require(truth, name); checks.append(name)
    def rejected(name, operation):
        try:
            operation()
        except RuntimeError:
            checks.append(name)
        else:
            raise AssertionError(name)
    validate_category_counts(21, 23)
    checks.append('original_87_21cats_and_augmented_95_23cats_are_distinct')
    rejected('original_23cats_schema_rejected', lambda: validate_category_counts(23, 23))
    rejected('augmented_21cats_schema_rejected', lambda: validate_category_counts(21, 21))
    start = pd.Timestamp('2022-08-11')
    day = (start - pd.Timestamp('2000-01-01')).days
    fit = pd.DataFrame({'address_city': ['A'], 'address_district': ['D'], 'project_name': ['known']})
    query = pd.DataFrame({'address_city': ['A'] * 4, 'address_district': ['D'] * 4,
                          'project_name': ['known', 'new', None, 'known'], 'trade_day_index': np.array([0, 92, 184, 276]) + day})
    weights = {'seen': .2, 'new_nonmissing': .3, 'missing': .5}
    result = independent_score([11, 12, 13, 14], np.repeat(10., 4), fit, query, str(start.date()), weights)
    check('overall_mae_in_twd', result['overall_mae'] == 2.5)
    check('fixed_group_weighted_mae', result['weighted_mae'] == .2 * 2.5 + .3 * 2 + .5 * 3)
    check('four_nonoverlapping_forecast_quarters', [v['mae'] for v in result['forecast_quarters'].values()] == [1., 2., 3., 4.])
    check('seen_new_missing_semantics', labels(fit, query).tolist() == ['seen', 'new_nonmissing', 'missing', 'seen'])
    subset = query.iloc[:1].copy()
    absent = independent_score([11], [10], fit, subset, str(start.date()), {'seen': 1., 'new_nonmissing': 0., 'missing': 0.})
    check('empty_strata_and_quarters_are_null', absent['strata']['missing']['mae'] is None and absent['forecast_quarters']['Q2']['mae'] is None)
    rejected('empty_positive_weight_stratum_rejected', lambda: independent_score([11], [10], fit, subset, str(start.date()), weights))
    rejected('unmatched_prediction_rows_rejected', lambda: independent_score([11], [10, 12], fit, subset, str(start.date()), weights))
    base = {'overall_mae': 100., 'weighted_mae': 100., 'strata': {'seen': {'rows': 500, 'mae': 100.}, 'missing': {'rows': 0, 'mae': None}},
            'forecast_quarters': {q: {'rows': 1, 'mae': 100.} for q in QUARTERS}}
    metrics = {CONTROL: copy.deepcopy(base), CANDIDATE: copy.deepcopy(base)}
    metrics[CANDIDATE]['overall_mae'] = metrics[CANDIDATE]['weighted_mae'] = 99.
    check('exact_one_percent_passes', independent_development_decision(metrics)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['weighted_mae'] = 99.00001
    check('both_mae_gates_required', not independent_development_decision(bad)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['strata']['seen']['mae'] = 105.00001
    check('large_group_deterioration_rejected', not independent_development_decision(bad)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 105.00001
    check('quarter_deterioration_rejected', not independent_development_decision(bad)['passed'])
    calls = []
    def reference_spy(ref, q, p, variant):
        require(ref == 'fit_reference' and q is query and variant == 'trend', 'Wrong reference identity')
        calls.append(p.copy()); return p * 1.1
    got = independent_blend([1] * 4, [3] * 4, [5] * 4, 'fit_reference', query, reference_spy)
    check('single_trend_fixed_slot_original_query', len(calls) == 1 and np.array_equal(got, np.repeat(1.1 * (.375 + 1.125 + 1.25), 4)))
    return checks


def verify():
    """Only the one finalized, explicitly authorized development window is read."""
    report_path = ROOT / 'reports/round11_backtest_review.md'
    proof_path = OUT / 'backtest_review.json'
    receipt_path = proof_path.with_suffix('.sha256')
    for path in (report_path, proof_path, receipt_path):
        require(not path.exists(), 'Refusing to overwrite independent evidence: ' + rel(path))
    started = datetime.now(timezone.utc).isoformat()
    checks = self_check()
    require(sha(ROOT / 'src/round11_experiment.py') == DRIVER_SHA, 'Reviewed runner changed')
    # Imports alone neither load real data nor train/evaluate models.
    import round11_experiment as driver
    driver.require_no_staging()
    specification = driver.plan()
    require(sha(OUT / 'candidate_plan.json') == PLAN_SHA, 'Unapproved price plan')
    require(specification['thresholds'] == THRESHOLDS, 'Frozen thresholds differ')
    require(isinstance(specification['runtime'], dict) and 'catboost' in specification['runtime'], 'Actual price plan runtime schema differs')
    require(driver.CONTROL == CONTROL and driver.CANDIDATE == CANDIDATE, 'Candidate identity changed')
    frozen = driver.locked(OUT / 'frozen_candidate.json')
    accepted = driver.locked(OUT / 'accepted_candidate.json')
    require(sha(OUT / 'frozen_candidate.json') == FREEZE_SHA, 'Unapproved freeze')
    require(accepted['accepted'] is False and accepted['stopping_reason'] == 'development_price_gates_failed',
            'This bounded verifier is only authorized for the finalized development rejection')
    require(accepted['stage_result_sha256'] == {'development': RESULT_SHA}, 'Unexpected scored window')
    require(accepted['plan_sha256'] == PLAN_SHA and accepted['frozen_candidate_sha256'] == FREEZE_SHA,
            'Acceptance provenance differs')
    require(accepted['source_sha256'] == specification['source_sha256'], 'Acceptance source differs')
    require(accepted['domain_plan_sha256'] == specification['domain_plan_sha256'], 'Acceptance domain authority differs')
    require(accepted['deployment_decision'] == 'keep_v3' and accepted['candidate'] == CANDIDATE
            and accepted['selected_candidate'] == CONTROL and accepted['new_submission_created'] is False
            and accepted['no_reselection_or_parameter_change'] is True, 'Acceptance meaning differs')
    require(accepted['currently_recommended_submission'] == 'outputs/submission_v3.csv'
            and accepted['private_score_known'] is False and accepted['historically_untouched'] is False,
            'Acceptance release/score declaration differs')
    require(accepted['stage_status'] == {'development': 'price_evaluated',
            'confirmation': 'not_executed_after_development_rejection',
            'auxiliary': 'not_executed_after_development_rejection'}, 'Stage status differs')
    require(accepted['unevaluated_stages'] == ['confirmation', 'auxiliary']
            and accepted['pure_X_feasibility_failure_stages'] == []
            and accepted['stage_preparation_manifest_sha256'] == {}
            and accepted['preparation_evidence_sha256'] == {}, 'Later-stage evidence unexpectedly present')
    for stage in ('confirmation', 'auxiliary'):
        for area in ('weights', 'models', 'evaluation'):
            require(not (OUT / area / stage).exists(), 'Unauthorized later-stage artifact')
    result = driver.read_result('development')
    require(sha(OUT / 'evaluation/development/results.json') == RESULT_SHA, 'Unapproved result')
    require(result['fit_rows'] == 240316 and result['query_rows'] == 81234, 'Fixed development counts differ')
    require(result['training_importance_weights_used_for_scoring'] is False
            and result['public_score_used'] is False and result['historically_untouched'] is False
            and result['new_public_score_known'] is False and result['private_score_known'] is False,
            'Result scoring/exposure declarations differ')
    require(result['v3_control_replayed_all_rows'] is True and result['new_model_replayed_all_rows'] is True,
            'Saved full-replay declarations missing')

    captured = {}
    def capture(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe/missing audited dependency')
        digest = sha(path)
        require(expected is None or digest == expected, 'Dependency identity mismatch: ' + rel(path))
        require(rel(path) not in captured or captured[rel(path)] == digest, 'Dependency changed during capture')
        captured[rel(path)] = digest
        return digest
    def capture_nested(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and '/' in key and isinstance(item, str) and len(item) == 64:
                    capture(ROOT / key, item)
                else:
                    capture_nested(item)
        elif isinstance(value, list):
            for item in value:
                capture_nested(item)
    for record in (specification, frozen, accepted, result):
        capture_nested(record)
    for path in (Path(__file__), OUT / 'candidate_plan.json', OUT / 'frozen_candidate.json',
                 OUT / 'accepted_candidate.json', OUT / 'domain_plan.json', OUT / 'evaluation/development/results.json'):
        capture(path)
        if path.suffix == '.json':
            capture(path.with_suffix('.sha256'))
    capture(OUT / 'backtest_review_incomplete_attempts.json')
    capture(OUT / 'backtest_review_incomplete_attempt2.md')
    for split in ('train', 'valid', 'test'):
        for suffix in ('.pkl', '_features.pkl', '.json'):
            capture(ROOT / 'data_cache' / (split + suffix))
    for name in ('valid_clean_mask.npy', 'valid_clean_mask.json'):
        capture(ROOT / 'data_cache' / name)
    # Preflight the canonical loader's only conditional cache writes: they must be reusable.
    schema = read(ROOT / 'reports/feature_schema.json')
    signature = {split: schema['splits'][split]['input_source_sha256'] for split in ('train', 'valid')}
    require(read(ROOT / 'data_cache/valid_clean_mask.json') == signature, 'Canonical mask would be rewritten')
    require((ROOT / 'models').is_dir() and (ROOT / 'outputs/experiments').is_dir(), 'Canonical output directories absent')
    protected = {}
    for path in [ROOT / 'outputs/submission_v2.csv', ROOT / 'outputs/submission_v3.csv']:
        protected[rel(path)] = capture(path)
    for path in (ROOT / 'outputs').glob('*version*.json'):
        protected[rel(path)] = capture(path)
    print(json.dumps({'event': 'audit_authorized', 'stage': 'development', 'query_rows': 81234,
                      'training': False, 'later_window_scoring': False}), flush=True)

    # Shared canonical loading physically deserializes all train/valid targets and raw rows.
    # Only the authorized development query target is used in error calculations below.
    data, X, y, Xq, yq, fi, qi = driver.partition('development')
    require(len(X) == 240316 and len(Xq) == 81234 and len(X.columns) == 87, 'Canonical development shape differs')
    validate_category_counts(len(data['categorical_columns']), 23)
    require(not np.intersect1d(fi, qi).size, 'Fit/query indices overlap')
    cutoff = (pd.Timestamp('2022-08-11') - pd.Timestamp('2000-01-01')).days
    end = (pd.Timestamp('2023-08-10') - pd.Timestamp('2000-01-01')).days
    require(np.isfinite(X['trade_day_index']).all() and (X['trade_day_index'] < cutoff).all(), 'Fit time leakage')
    require(np.isfinite(Xq['trade_day_index']).all() and Xq['trade_day_index'].between(cutoff, end).all(), 'Wrong query dates')
    fixed_weights = scoring_weights(data)
    require(fixed_weights == specification['target_group_weights'] == result['target_group_weights'],
            'Independent target-X group weights differ')
    geo, geoq, _ = driver.load_geographic_partition('development', fi, qi)
    weights, contract, manifest, weight_inputs = driver.load_weights('development', X, Xq, fi, qi, specification)
    capture_nested(weight_inputs)
    require(manifest['feasibility_passed'] is True, 'Weights did not pass engineering gates')
    require(result['preparation_manifest_sha256'] == contract['preparation_manifest_sha256']
            and result['weight_contract'] == contract, 'Result preparation contract differs')
    folder = OUT / 'models/development'
    metadata = driver.locked(folder / 'metadata.json')
    state = read(folder / 'feature_state.json')
    expected = {key: driver.frame_sha(frame) for key, frame in [
        ('fit_X_sha256', X), ('query_X_sha256', Xq), ('fit_geo_sha256', geo), ('query_geo_sha256', geoq)]}
    expected.update({'domain_plan_sha256': specification['domain_plan_sha256'],
        'preparation_manifest_sha256': contract['preparation_manifest_sha256'], 'weight_contract': contract,
        'fit_sample_weight_values_sha256': driver.component.weights_values_sha256(weights),
        'fit_indices_content_sha256': driver.value_sha(fi, '<i8'),
        'query_indices_content_sha256': driver.value_sha(qi, '<i8'),
        'weight_preparation_input_sha256': weight_inputs})
    driver.validate_model_contract(metadata, specification, expected)
    require(metadata['stage'] == 'development' and metadata['plan_sha256'] == PLAN_SHA
            and metadata['source_sha256'] == specification['source_sha256'], 'Model source/stage/plan mismatch')
    require(metadata['fit_target_sha256'] == metadata['fit_target_values_sha256'] == driver.value_sha(y, '<f8'),
            'Model fit-target identity mismatch')
    require(metadata['fit_transformed_target_values_sha256'] == driver.value_sha(np.asarray(y, dtype='float64') / 1e6, '<f8'),
            'Raw-million-TWD target mismatch')
    require(metadata['outer_fit_rows'] == len(X) and metadata['query_rows'] == len(Xq), 'Model row counts mismatch')
    require(metadata['geo_manifest_sha256'] == sha(driver.c6.OUT / 'geography/development/manifest.json')
            and metadata['frozen_candidate_sha256'] is None, 'Model geography/freeze mismatch')
    require(metadata['reload_64_rows_verified'] is True and metadata['prediction_units'] == 'TWD'
            and metadata['direct_component_score_omitted'] is True, 'Model prediction declarations differ')
    required = {'model.cbm', 'feature_state.json', 'predictions.npy', 'fit_indices.npy', 'query_indices.npy'}
    require(set(metadata['artifact_sha256']) == required, 'Model artifact inventory differs')
    for name, digest in metadata['artifact_sha256'].items():
        capture(folder / name, digest)
    capture(folder / 'metadata.json'); capture(folder / 'metadata.sha256')
    driver.e6.verify_indices(folder, fi, qi, 'query_indices.npy')
    reconstructed_state = driver.component.fit_feature_state(X, geo, data['categorical_columns'], weights, contract)
    require(reconstructed_state == state, 'Saved fit-only feature/weight state differs')
    require(metadata['feature_state_content_sha256'] == state['state_content_sha256']
            and metadata['geo_state_content_sha256'] == state['geo_state']['state_content_sha256']
            and metadata['weight_contract_sha256'] == state['weight_contract_sha256']
            and metadata['fit_sample_weight_summary'] == state['fit_sample_weight_summary'], 'Model/state summaries differ')
    model = driver.load_model(folder / 'model.cbm')
    require(int(model.tree_count_) == 2999 and len(model.feature_names_) == 95, 'Saved model structure mismatch')
    validate_category_counts(len(data['categorical_columns']), len(model.get_cat_feature_indices()))
    fresh_component = driver.component.predict_model(model, state, Xq, geoq, data['categorical_columns'])
    saved_component = driver.e6.vector(folder / 'predictions.npy', len(qi))
    component_comparison = compare_vector(saved_component, fresh_component, 'Weighted geographic CatBoost')
    print(json.dumps({'event': 'all_query_model_replayed', **component_comparison}), flush=True)
    for key, frame in [('fit_X_sha256', X), ('query_X_sha256', Xq), ('fit_geo_sha256', geo), ('query_geo_sha256', geoq)]:
        require(driver.frame_sha(frame) == expected[key], 'Original feature frame mutated by replay')

    reference, reference_path = driver.c6.load_reference('development', X)
    capture(reference_path)
    previous = driver.c6.baseline_folder('development')
    old_path, unit_path = previous / 'catboost_raw.npy', previous / 'lightgbm_l1_unit63.npy'
    geographic_path = driver.c6.OUT / 'models/development/geography_catboost/predictions.npy'
    original_path = driver.c6.OUT / 'evaluation/development/geography_blend.npy'
    old = driver.e6.vector(old_path, len(qi)); unit = driver.e6.vector(unit_path, len(qi))
    geographic = driver.e6.vector(geographic_path, len(qi)); original = driver.e6.vector(original_path, len(qi))
    for path in (old_path, unit_path, geographic_path, original_path):
        capture(path)
    fresh_control = independent_blend(old, geographic, unit, reference, Xq, driver.predict_reference)
    fresh_candidate = independent_blend(old, fresh_component, unit, reference, Xq, driver.predict_reference)
    control_path = OUT / 'evaluation/development/v3_control.npy'
    candidate_path = OUT / 'evaluation/development/importance_geo_blend.npy'
    saved_control = driver.e6.vector(control_path, len(qi))
    saved_candidate = driver.e6.vector(candidate_path, len(qi))
    require(np.array_equal(original, saved_control), 'R11 control differs from original v3 vector')
    control_comparison = compare_vector(saved_control, fresh_control, 'v3 control blend')
    candidate_comparison = compare_vector(saved_candidate, fresh_candidate, 'Sole R11 candidate blend')
    metrics = {CONTROL: independent_score(yq, saved_control, X, Xq, '2022-08-11', fixed_weights),
               CANDIDATE: independent_score(yq, saved_candidate, X, Xq, '2022-08-11', fixed_weights)}
    require(metrics == result['metrics'], 'Independent full metrics do not equal saved R11 metrics')
    old_result_path = driver.c6.OUT / 'evaluation/development/results.json'
    old_result = driver.locked(old_result_path)
    capture(old_result_path); capture(old_result_path.with_suffix('.sha256'))
    require(metrics[CONTROL] == old_result['metrics']['geography_blend'], 'Original v3 result metrics differ')
    decision = independent_development_decision(metrics)
    require(decision == result['decision'] and decision['passed'] is False, 'Independent gate decision differs')
    require(frozen['plan_sha256'] == PLAN_SHA and frozen['result_sha256'] == RESULT_SHA
            and frozen['source_sha256'] == specification['source_sha256'] and frozen['round'] == 11
            and frozen['parameters'] == specification['parameters'] and frozen['formulas'] == driver.FORMULAS
            and frozen['thresholds'] == THRESHOLDS and frozen['domain_plan_sha256'] == specification['domain_plan_sha256']
            and frozen['passed'] is False and frozen['later_stages_allowed'] is False
            and frozen['selected_candidate'] == decision['selected_candidate']
            and frozen['historically_untouched'] is False, 'Freeze does not implement independent decision')
    actual_frozen = {rel(p): sha(p) for base in (folder, OUT / 'evaluation/development') for p in base.iterdir() if p.is_file()}
    require(frozen['development_artifact_sha256'] == actual_frozen, 'Frozen development artifact inventory differs')
    driver.require_no_staging()
    for stage in ('confirmation', 'auxiliary'):
        for area in ('weights', 'models', 'evaluation'):
            require(not (OUT / area / stage).exists(), 'Later-stage artifacts appeared during audit')
    for key, frame in [('fit_X_sha256', X), ('query_X_sha256', Xq), ('fit_geo_sha256', geo), ('query_geo_sha256', geoq)]:
        require(driver.frame_sha(frame) == expected[key], 'Feature frame changed during full model/reference replay')
    for key, digest in captured.items():
        require(sha(ROOT / key) == digest, 'Audited dependency changed during replay: ' + key)
    require(driver.sources() == specification['source_sha256'], 'Frozen source changed during audit')
    stage_evidence = {
        'results_sha256': RESULT_SHA, 'model_metadata_sha256': sha(folder / 'metadata.json'),
        'model_sha256': sha(folder / 'model.cbm'), 'feature_state_sha256': sha(folder / 'feature_state.json'),
        'component_predictions_sha256': sha(folder / 'predictions.npy'),
        'control_predictions_sha256': sha(control_path), 'candidate_predictions_sha256': sha(candidate_path),
        'fit_indices_sha256': sha(folder / 'fit_indices.npy'), 'query_indices_sha256': sha(folder / 'query_indices.npy'),
        'fit_indices_content_sha256': driver.value_sha(fi, '<i8'), 'query_indices_content_sha256': driver.value_sha(qi, '<i8'),
        'query_target_content_sha256': driver.value_sha(yq, '<f8'), 'fit_rows': len(fi), 'query_rows': len(qi),
        'dates': ['2022-08-11', '2023-08-10'], 'original_v3_results_sha256': sha(old_result_path),
        'original_v3_predictions_sha256': sha(original_path), 'reference_sha256': sha(reference_path),
        'weight_preparation_manifest_sha256': contract['preparation_manifest_sha256'],
        'component_replay': component_comparison, 'v3_control_replay': control_comparison, 'candidate_replay': candidate_comparison,
        'all_metric_sections_equal': True, 'metrics': metrics, 'decision': decision,
        'feature_state_reconstructed_without_training': True, 'original_X87_and_geo8_unchanged': True,
        'saved_model_trees': int(model.tree_count_), 'features': 95, 'categorical_features': 23,
    }
    return publish_review(started, checks, specification, accepted, captured, protected, stage_evidence)


def publish_review(started, checks, specification, accepted, captured, protected, evidence):
    report_path = ROOT / 'reports/round11_backtest_review.md'
    proof_path = OUT / 'backtest_review.json'
    receipt_path = proof_path.with_suffix('.sha256')
    for path in (report_path, proof_path, receipt_path):
        require(not path.exists(), 'Independent evidence exists; refusing replacement')
    finished = datetime.now(timezone.utc).isoformat()
    metrics = evidence['metrics']; base, candidate = metrics[CONTROL], metrics[CANDIDATE]
    def delta(metric):
        return 100. * (base[metric] - candidate[metric]) / base[metric]
    lines = [
        '# R11 已完成回測的獨立核對', '',
        '核對狀態：passed（保存結果與重播一致）；候選決策：accepted=false，保留 v3。審查通過不代表候選通過價格門檻。', '',
        f'本次實際核對時間：{started} 至 {finished}；僅檢查已正式完成的 development（2022-08-11 至 2023-08-10）。',
        '執行備註：第一次核對在 canonical loader 後、模型重播及 MAE 重算前中止；新 verifier 把原 87 欄的類別欄數誤寫成增強後的 23。已依 frozen schema 修成原始 87/21 與增強後 95/23 分開驗證，補三項合成案例，再完成同一固定核對；未改模型、helper、plan、結果或門檻。第一次沒有產生正式 review／proof，未保存獨立失敗 log；本說明補記已發生的工具退出，不倒填先前時間。', '',
        '第二次執行已完成全部固定模型／融合重播、MAE／decision 及來源前後核對，但 proof 封裝把價格計畫 runtime 鍵误用為 domain 計畫的 runtime_versions，於 JSON 寫入前退出。當時 MD 保留為 outputs/round11/backtest_review_incomplete_attempt2.md，兩次中止均記於 outputs/round11/backtest_review_incomplete_attempts.json。本次修正只涉及審查封裝，並以實際 plan schema 預檢；所有正式證據先在本輪專用暫存組合、驗證後才發布。', '',
        'Fit 240,316 筆；query 81,234 筆。confirmation 與 auxiliary 的 R11 權重、模型、評分目錄均不存在；沒有建立後期分數、模型或 CSV。', '',
        '| 指標（TWD） | v3 control | 唯一候選 | 改善率（正值為改善） |',
        '|---|---:|---:|---:|',
        f"| Overall MAE | {base['overall_mae']:.10f} | {candidate['overall_mae']:.10f} | {delta('overall_mae'):.6f}% |",
        f"| 固定三組重加權 MAE | {base['weighted_mae']:.10f} | {candidate['weighted_mae']:.10f} | {delta('weighted_mae'):.6f}% |", '',
        '完整 overall、三組重加權、seen/new_nonmissing/missing、四個 forecast quarters、每月及各城市 MAE 均獨立重算，與保存的 R11 JSON 完全相等；v3 control 也與原 R6 development geography_blend 結果完全相等。', '',
        '| 事前 development 門檻 | 獨立核對結果 |', '|---|---|',
    ]
    lines.extend(f'| {name} | {str(passed).lower()} |' for name, passed in evidence['decision']['checks'].items())
    lines += ['',
        '保存的 2,999 棵 CatBoost 模型已重新載入，逐一重播全部 81,234 筆 query。95 個模型欄位／23 類別欄與 fit-only feature/weight state 均核對；原 87 個 X 及 8 個 geo 欄的值、型別、索引雜湊在重播前後不變。',
        f"候選元件、v3 control 融合、R11 唯一融合的最大重播絕對差分別為 {evidence['component_replay']['maximum_absolute_difference_twd']:.12g}、{evidence['v3_control_replay']['maximum_absolute_difference_twd']:.12g}、{evidence['candidate_replay']['maximum_absolute_difference_twd']:.12g} TWD。", '',
        '模型元件沿用 raw y / 1e6、prediction × 1e6 並裁切非負。融合獨立重建為 37.5% old raw CatBoost + 37.5% geographic CatBoost + 25% old unit LightGBM，再以原 87 欄 query 呼叫保存的 fit-only reference trend 一次。control 使用原 geographic 預測，唯一候選只替換該 37.5% 槽位；沒有 R7／R8／POI 元件。',
        '舊三個價格元件使用已綁定 SHA 的保存預測，本次沒有重新推論舊模型；重播的是新加權 CatBoost 全 query 以及兩個融合／reference 管線。沒有重訓任何模型，也沒有重播五個 domain classifiers。', '',
        '權重準備沿 mandatory load_prepared 核驗合法收據、runtime、來源、完整 query X、fi/qi、18 個保存檔案與權重順序／內容；它由既存 logits 重算權重算術及固定工程閘，不重新訓練或推論 domain classifier。訓練 importance weights 沒有拿來取代評分用的固定三組 test-X 權重。', '',
        '資料接觸範圍：本審查使用共用 canonical loader，會實際反序列化 train／valid 的完整標籤及 train／valid／test 原始快取，並沿既有去重規則取得一致索引。這與前段 pure-X 準備不同，不能稱為「未讀到其他標籤」。Fit y 僅核對既存模型的 target SHA；誤差計算只使用本次已授權 development query 的 81,234 個 y。其餘窗口、valid 與 test 不計分、不做誤差探索。', '',
        '這些年度窗口已有歷史研究暴露，並非全新盲測；1% 是固定採用門檻，不是顯著性檢定。此核對沒有新增 Public 或 Private 成績，也不使用外部成交答案。', '',
        f'來源與輸入／保存產物共 {len(captured)} 個檔案的 SHA 在核對前後一致，包含 frozen source closure、price/domain plan、model／indices／prepared manifest、基準／reference／結果、freeze／acceptance、原 v2/v3 CSV 及已存在的 version registry。',
        'JSON 證據保存全部指標、decision、各模型／向量／索引雜湊、原 v3 證據及讀取範圍；不輸出每筆價格、誤差或新提交檔。', '',
        f'Price plan SHA: `{PLAN_SHA}`',
        f'Runner SHA: `{DRIVER_SHA}`',
        f'Independent verifier SHA: `{sha(Path(__file__))}`',
        f'Saved development results SHA: `{RESULT_SHA}`',
        f'Freeze SHA: `{FREEZE_SHA}`',
        f'Acceptance SHA: `{sha(OUT / "accepted_candidate.json")}`', '',
    ]
    report_text = '\n'.join(lines)
    report_digest = hashlib.sha256(report_text.encode('utf-8')).hexdigest()
    proof = {
        'round': 11, 'status': 'passed', 'started_at_utc': started, 'recorded_at_utc': finished,
        'checked_stages': ['development'], 'plan_sha256': PLAN_SHA,
        'frozen_candidate_sha256': FREEZE_SHA, 'accepted_candidate_sha256': sha(OUT / 'accepted_candidate.json'),
        'candidate_accepted': False, 'acceptance_stopping_reason': accepted['stopping_reason'],
        'source_sha256': {**specification['source_sha256'], rel(Path(__file__)): sha(Path(__file__))},
        'plan_source_sha256': specification['source_sha256'],
        'runtime_versions': specification['runtime'],
        'stage_evidence': {'development': evidence}, 'only_completed_stages_scored': True,
        'training_performed': False, 'domain_classifier_replay_performed': False,
        'old_price_models_freshly_inferred': False, 'new_saved_model_all_query_replayed': True,
        'later_stages_checked_for_absence': ['confirmation', 'auxiliary'],
        'later_stages_scored': [], 'synthetic_checks': checks,
        'execution_notes': [{'attempt': 1, 'status': 'stopped_before_model_replay_and_metric_recomputation',
            'prior_verifier_source_sha256': PREVIOUS_INCOMPLETE_AUDIT_SOURCE_SHA,
            'reason': 'Verifier incorrectly expected original87 cats=23; actual original87 cats=21, augmented95 cats=23',
            'targets_deserialized': True, 'model_replay_performed': False, 'metrics_recomputed': False,
            'formal_review_files_written': False, 'separate_failure_log_saved': False,
            'correction': 'Separate schema assertions plus three synthetic checks; frozen model/source/plan/results unchanged',
            'recorded_at_utc': finished, 'prior_event_timestamp_not_reconstructed': True},
            {'attempt': 2, 'status': 'all_fixed_checks_completed_proof_serialization_failed',
             'prior_verifier_source_sha256': '7e4dabcdfb227c7c93897595b2262b4c77352bb5bde28fac22207df5a2971b6b',
             'reason': 'Price plan runtime key was incorrectly accessed as runtime_versions at proof construction',
             'model_replay_performed': True, 'metrics_recomputed': True, 'formal_proof_written': False,
             'incomplete_draft_path': 'outputs/round11/backtest_review_incomplete_attempt2.md',
             'incomplete_draft_sha256': sha(OUT / 'backtest_review_incomplete_attempt2.md'),
             'recorded_at_utc': finished, 'prior_event_timestamp_not_reconstructed': True}],
        'incomplete_attempts_log_sha256': sha(OUT / 'backtest_review_incomplete_attempts.json'),
        'actual_plan_publication_schema_preflight_passed': True,
        'input_artifact_sha256': captured, 'source_and_inputs_unchanged_after_execution': True,
        'protected_release_files_sha256': protected, 'new_submission_created': False,
        'currently_recommended_submission': 'outputs/submission_v3.csv',
        'public_score_used': False, 'new_public_score_known': False, 'private_score_known': False,
        'historically_untouched': False, 'thresholds_are_significance_tests': False,
        'report_sha256': report_digest,
        'data_access_scope': {
            'raw_cache_splits_deserialized': ['train', 'valid', 'test'],
            'complete_train_and_valid_target_arrays_deserialized': True,
            'canonical_duplicate_rule_reused': True, 'existing_clean_mask_preverified_to_avoid_write': True,
            'fit_targets_used_for': 'Existing saved model target identity hashes only; no fitting',
            'query_targets_used_for': 'Fixed development metrics only, 81,234 rows',
            'other_window_targets_scored_or_explored': False,
            'query_X_used_by_original_training_weight_method': True,
            'query_y_used_by_training_weight_method': False,
            'target_X_used_for_fixed_scoring_stratum_weights': True,
            'new_per_row_error_analysis': False,
        },
    }
    # Serialize and validate the entire evidence bundle before publishing any formal file.
    needed = {'results_sha256','model_metadata_sha256','model_sha256','feature_state_sha256',
              'component_predictions_sha256','control_predictions_sha256','candidate_predictions_sha256'}
    require(needed <= set(proof['stage_evidence']['development']), 'Missing report-consumer evidence fields')
    require(proof['plan_source_sha256'] == specification['source_sha256'] and proof['checked_stages'] == ['development'], 'Proof plan/stage mismatch')
    encoded = (json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    proof_digest = hashlib.sha256(encoded).hexdigest()
    staging = OUT / ('.backtest-review-' + uuid.uuid4().hex)
    staging.mkdir()
    stage_report, stage_proof, stage_receipt = staging / report_path.name, staging / proof_path.name, staging / receipt_path.name
    stage_report.write_bytes(report_text.encode('utf-8'))
    stage_proof.write_bytes(encoded)
    stage_receipt.write_text(proof_digest + '\n', encoding='utf-8')
    staged = read(stage_proof)
    require(sha(stage_report) == staged['report_sha256'] and sha(stage_proof) == stage_receipt.read_text(encoding='utf-8').strip(), 'Staged bundle receipt mismatch')
    require(staged == proof, 'Staged JSON roundtrip mismatch')
    for path in (report_path, proof_path, receipt_path):
        require(not path.exists(), 'Formal evidence appeared during publication')
    stage_report.rename(report_path)
    stage_proof.rename(proof_path)
    stage_receipt.rename(receipt_path)
    staging.rmdir()
    print(json.dumps({'status': 'passed', 'candidate_accepted': False, 'checked_stages': ['development'],
                      'proof_sha256': sha(proof_path), 'report_sha256': sha(report_path),
                      'source_sha256': sha(Path(__file__)), 'checked_files': len(captured)}, ensure_ascii=False), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['self-check', 'verify'])
    args = parser.parse_args()
    if args.command == 'self-check':
        checks = self_check()
        print(json.dumps({'passed': True, 'checks': checks, 'count': len(checks),
                          'real_data_read': False, 'training_or_real_scoring': False}))
    else:
        verify()
