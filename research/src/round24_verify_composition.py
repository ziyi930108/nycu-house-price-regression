"""R24 saved-output composition audit, not an independent native-model replay.

Only --run performs I/O. It requires a finalized experiment, reads saved float64
outputs, and independently implements clipping, the fixed three-seed mean, and
blending. Original score/decision functions and reference are deliberately reused.
No CatBoost model is deserialized, fitted, or asked to predict by this verifier.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import round24_experiment as driver

ROOT = driver.ROOT
OUT = driver.OUT
DESTINATION = OUT / 'audit/composition_verification.json'
DRIVER_SHA256 = 'e0cee086cfd1349ee1ce46cc1d2b491ff348e3d8939189dbdca3873bca228fbb'
SEEDS = (42, 137, 2026)
require = driver.require


def value_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def vector(path, rows, nonnegative=False):
    values = np.load(path, allow_pickle=False)
    require(values.dtype == np.dtype('float64') and values.shape == (rows,),
            'Expected saved float64 vector: ' + str(path))
    require(np.isfinite(values).all(), 'Nonfinite saved vector: ' + str(path))
    if nonnegative:
        require((values >= 0).all(), 'Negative TWD vector: ' + str(path))
    return values


def same_bits(left, right, label):
    require(left.dtype == right.dtype == np.dtype('float64') and left.shape == right.shape,
            'Vector schema differs: ' + label)
    require(np.array_equal(left.view(np.uint64), right.view(np.uint64)),
            'Saved-output bitwise comparison failed: ' + label)
    return {'rows': len(left), 'bitwise_equal': True, 'maximum_absolute_error': 0.0}


def inverse_raw(raw):
    # Independent of the fitting helper's inverse_transform.
    with np.errstate(over='raise', invalid='raise'):
        result = np.maximum(raw * 1e6, 0.0)
    require(result.dtype == np.dtype('float64') and np.isfinite(result).all(),
            'Inverse transformation overflow')
    return result


def mean_three(predictions):
    require(set(predictions) == set(SEEDS), 'Exactly the three fixed seeds are required')
    with np.errstate(over='raise', invalid='raise'):
        result = (predictions[42] + predictions[137] + predictions[2026]) / 3.0
    require(np.isfinite(result).all() and (result >= 0).all(), 'Invalid fixed mean')
    return result


def compose(old, geography, unit, reference, query):
    # Preserve the frozen arithmetic order, rather than reassociating its terms.
    with np.errstate(over='raise', invalid='raise'):
        base = .75 * (.5 * old + .5 * geography) + .25 * unit
    result = np.asarray(driver.predict_reference(reference, query, base, variant='trend'),
                        dtype='float64')
    require(result.shape == (len(query),) and np.isfinite(result).all()
            and (result >= 0).all(), 'Invalid reference-adjusted composition')
    return result


def same_metrics(actual, saved, path='metrics'):
    """Full structure/count equality; only MAE-valued leaves allow 1e-6 TWD."""
    if isinstance(saved, dict):
        require(isinstance(actual, dict) and set(actual) == set(saved), 'Metric keys differ: ' + path)
        for key in saved:
            same_metrics(actual[key], saved[key], path + '.' + key)
    elif isinstance(saved, list):
        require(isinstance(actual, list) and len(actual) == len(saved), 'Metric list differs: ' + path)
        for i, (a, b) in enumerate(zip(actual, saved)):
            same_metrics(a, b, path + '.' + str(i))
    elif isinstance(saved, float) and 'mae' in path.rsplit('.', 1)[-1].lower():
        require(isinstance(actual, (int, float)) and np.isfinite(actual)
                and np.isfinite(saved) and abs(actual - saved) <= 1e-6,
                'MAE differs by more than 1e-6 TWD: ' + path)
    else:
        require(type(actual) is type(saved) and actual == saved, 'Metric value/type differs: ' + path)


def check_indices(folder, fi, qi, query_name='query_indices.npy'):
    for filename, expected in [('fit_indices.npy', fi), (query_name, qi)]:
        actual = np.load(folder / filename, allow_pickle=False)
        require(actual.dtype == np.dtype('int64') and actual.shape == expected.shape
                and np.array_equal(actual, expected), 'Saved row indices differ: ' + str(folder / filename))


def merge(mapping, more):
    for name, digest in more.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Unsafe/missing dependency: ' + name)
        require(name not in mapping or mapping[name] == digest, 'Conflicting dependency: ' + name)
        mapping[name] = digest


def bind_json(mapping, path):
    value = driver.locked(path)
    merge(mapping, {driver.rel(path): driver.sha(path),
                    driver.rel(path.with_suffix('.sha256')): driver.sha(path.with_suffix('.sha256'))})
    return value


def check_seed42(evidence, stage, spec, folder, fi, qi, raw, prediction, dependencies):
    original = driver.c6.OUT / 'models' / stage / 'geography_catboost'
    names = ('model.cbm', 'feature_state.json', 'metadata.json', 'predictions.npy',
             'fit_indices.npy', 'query_indices.npy')
    expected = {driver.rel(original / name): spec['stage_dependency_sha256'][stage][driver.rel(original / name)]
                for name in names}
    require(evidence['source_artifact_sha256'] == expected, 'Seed42 original source map differs')
    merge(dependencies, expected)
    require(evidence['round'] == 24 and evidence['stage'] == stage and evidence['seed'] == 42
            and evidence['retrained'] is False and evidence['query_rows'] == len(qi), 'Seed42 replay identity differs')
    for key in ('original_state_is_flat_R6', 'fit_state_reconstructed_from_fit_X_geo_only',
                'all_query_rows_replayed', 'original_saved_predictions_bitwise_equal',
                'this_round_full_replay_is_separate_from_original_64_row_check'):
        require(evidence[key] is True, 'Seed42 driver evidence is incomplete: ' + key)
    require(evidence['fit_indices_content_sha256'] == value_sha(fi, '<i8')
            and evidence['query_indices_content_sha256'] == value_sha(qi, '<i8')
            and evidence['raw_values_sha256'] == value_sha(raw, '<f8')
            and evidence['TWD_values_sha256'] == value_sha(prediction, '<f8'), 'Seed42 content fingerprint differs')
    state = driver.read(original / 'feature_state.json')
    require(state['state_content_sha256'] == evidence['flat_state_content_sha256'], 'Seed42 flat state differs')
    check_indices(original, fi, qi)
    return same_bits(prediction, vector(original / 'predictions.npy', len(qi), True),
                     stage + ': seed42 vs original R6 saved prediction')


def verify_stage(stage, result, spec, dependencies):
    # The established loader deserializes targets. Here query y is used only to
    # replay the already declared score, never for fitting or new combinations.
    data, X, y, Xq, yq, fi, qi = driver.partition(stage)
    require(len(X.columns) == len(Xq.columns) == 87 and list(X.columns) == list(Xq.columns),
            'Reference/scoring must receive original 87 features')
    weights = driver.target_weights(data)
    require(weights == spec['target_group_weights'], 'Fixed test-X score weights differ')
    split = spec['split_contract'][stage]
    require(fi.dtype == qi.dtype == np.dtype('int64') and len(fi) == split['fit_rows']
            and len(qi) == split['query_rows'] and not np.intersect1d(fi, qi).size, 'Partition differs')
    folder = OUT / 'models' / stage
    meta = bind_json(dependencies, folder / 'metadata.json')
    check_indices(folder, fi, qi)
    require(meta['fit_target_sha256'] == value_sha(y, '<f8'), 'Original fit target identity differs')
    require(meta['fit_X_sha256'] == driver.frame_sha(X)
            and meta['query_X_sha256'] == driver.frame_sha(Xq), 'Original feature identity differs')
    predictions, members = {}, {}
    for seed in SEEDS:
        branch = folder if seed == 42 else folder / ('seed_' + str(seed))
        raw_path = branch / ('seed42_raw_z.npy' if seed == 42 else 'raw_z.npy')
        price_path = branch / ('seed42_predictions.npy' if seed == 42 else 'predictions.npy')
        raw = vector(raw_path, len(qi))
        prediction = inverse_raw(raw)
        match = same_bits(prediction, vector(price_path, len(qi), True), stage + ': seed ' + str(seed))
        evidence = {'raw_z_sha256': driver.sha(raw_path), 'predictions_sha256': driver.sha(price_path),
                    'raw_values_sha256': value_sha(raw, '<f8'), 'TWD_values_sha256': value_sha(prediction, '<f8'),
                    'inverse_comparison': match}
        if seed == 42:
            replay = bind_json(dependencies, folder / 'seed42_replay.json')
            evidence['original_prediction_comparison'] = check_seed42(
                replay, stage, spec, folder, fi, qi, raw, prediction, dependencies)
            evidence['driver_model_replay_evidence_sha256'] = driver.sha(folder / 'seed42_replay.json')
        else:
            member = bind_json(dependencies, branch / 'metadata.json')
            driver.member_contract(member, seed, spec)
            check_indices(branch, fi, qi)
            require(member['reload_query_rows'] == len(qi), 'Incomplete member driver replay')
            require(member['fit_original_target_values_sha256'] == value_sha(y, '<f8')
                    and member['fit_transformed_target_values_sha256'] == value_sha(np.asarray(y, dtype='float64') / 1e6, '<f8'),
                    'Member target scale identity differs')
            diag = member['raw_z_diagnostics']
            require(diag['rows'] == len(qi) and diag['negative_count'] == int((raw < 0).sum())
                    and diag['zero_count'] == int((raw == 0).sum()) and diag['positive_count'] == int((raw > 0).sum())
                    and diag['min'] == float(raw.min()) and diag['max'] == float(raw.max())
                    and diag['zero_price_count'] == int((prediction == 0).sum())
                    and diag['raw_values_sha256'] == value_sha(raw, '<f8')
                    and diag['TWD_values_sha256'] == value_sha(prediction, '<f8')
                    and diag['inverse_matches_component_all_rows'] is True and diag['used_for_tuning'] is False,
                    'Member saved raw diagnostics differ')
            evidence['driver_model_replay_evidence_sha256'] = driver.sha(branch / 'metadata.json')
        members[str(seed)] = evidence
        predictions[seed] = prediction
    mean = mean_three(predictions)
    mean_match = same_bits(mean, vector(folder / 'predictions.npy', len(qi), True), stage + ': fixed mean')
    baseline = driver.c6.baseline_folder(stage)
    check_indices(baseline, fi, qi, 'check_indices.npy')
    old = vector(baseline / 'catboost_raw.npy', len(qi), True)
    unit = vector(baseline / 'lightgbm_l1_unit63.npy', len(qi), True)
    reference, reference_path = driver.c6.load_reference(stage, X)
    require(spec['stage_dependency_sha256'][stage][driver.rel(reference_path)] == driver.sha(reference_path),
            'Reference not bound to this stage')
    reconstructed_control = compose(old, predictions[42], unit, reference, Xq)
    original_control = vector(driver.c6.OUT / 'evaluation' / stage / 'geography_blend.npy', len(qi), True)
    # The original driver declares this tolerance for the old reference path.
    require(np.allclose(reconstructed_control, original_control, rtol=1e-12, atol=1e-5),
            'Reconstructed original v3 differs beyond the frozen replay tolerance')
    evaluation = OUT / 'evaluation' / stage
    control_match = same_bits(original_control, vector(evaluation / (driver.CONTROL + '.npy'), len(qi), True),
                              stage + ': saved v3 control')
    candidate = compose(old, mean, unit, reference, Xq)
    candidate_match = same_bits(candidate, vector(evaluation / (driver.CANDIDATE + '.npy'), len(qi), True),
                                stage + ': candidate composition')
    metrics = {name: driver.score(yq, values, X, Xq, driver.c6.SPLITS[stage][0], weights)
               for name, values in ((driver.CONTROL, original_control), (driver.CANDIDATE, candidate))}
    same_metrics(metrics, result['metrics'])
    old_result = driver.locked(driver.c6.OUT / 'evaluation' / stage / 'results.json')
    same_metrics(metrics[driver.CONTROL], old_result['metrics']['geography_blend'], 'original_v3.metrics')
    decision = driver.decide(stage, metrics)
    require(decision == result['decision'], 'Recomputed existing decision differs')
    return {'stage': stage, 'fit_rows': len(fi), 'query_rows': len(qi), 'dates': list(driver.c6.SPLITS[stage]),
            'result_sha256': driver.sha(evaluation / 'results.json'),
            'model_metadata_sha256': driver.sha(folder / 'metadata.json'),
            'fit_indices_content_sha256': value_sha(fi, '<i8'),
            'query_indices_content_sha256': value_sha(qi, '<i8'),
            'members': members, 'mean': mean_match, 'saved_control': control_match,
            'candidate': candidate_match, 'reference_sha256': driver.sha(reference_path),
            'original_v3_reference_replay': {'rows': len(qi), 'within_original_tolerance': True,
                'bitwise_equal': bool(np.array_equal(reconstructed_control.view(np.uint64), original_control.view(np.uint64))),
                'maximum_absolute_error': float(np.max(np.abs(reconstructed_control - original_control))),
                'rtol': 1e-12, 'atol_TWD': 1e-5},
            'metrics_all_fields_matched': True, 'MAE_absolute_tolerance_TWD': 1e-6,
            'original_v3_metrics_matched': True, 'decision_matched': True, 'decision': decision,
            'individual_seed_or_component_metrics_computed': False}


def run():
    require(not DESTINATION.exists() and not DESTINATION.with_suffix('.sha256').exists(),
            'Composition proof exists or is partial; refusing overwrite')
    driver.require_no_staging()
    require(driver.sha(Path(driver.__file__)) == DRIVER_SHA256, 'Reviewed driver changed')
    script_sha = driver.sha(Path(__file__))
    spec = driver.plan()  # Hash/config verification only; does not fit or load CBM.
    dependencies = {}
    merge(dependencies, spec['source_sha256'])
    merge(dependencies, spec['dependency_sha256'])
    merge(dependencies, {driver.rel(Path(__file__)): script_sha})
    bind_json(dependencies, OUT / 'candidate_plan.json')
    frozen = bind_json(dependencies, OUT / 'frozen_candidate.json')
    accepted = bind_json(dependencies, OUT / 'accepted_candidate.json')
    require(accepted['round'] == 24 and accepted['candidate'] == driver.CANDIDATE
            and accepted['plan_sha256'] == driver.sha(OUT / 'candidate_plan.json')
            and accepted['frozen_candidate_sha256'] == driver.sha(OUT / 'frozen_candidate.json')
            and accepted['source_sha256'] == spec['source_sha256'], 'Finalized experiment identity differs')
    stages = list(driver.STAGES) if frozen['passed'] else ['development']
    require(set(accepted['stage_result_sha256']) == set(stages), 'Incomplete/unauthorized stage results')
    require(accepted['unevaluated_stages'] == [s for s in driver.STAGES if s not in stages], 'Unevaluated stage list differs')
    results = {}
    for stage in driver.STAGES:
        if stage not in stages:
            require(accepted['stage_status'][stage] == 'not_executed_after_development_rejection'
                    and not (OUT / 'models' / stage).exists() and not (OUT / 'evaluation' / stage).exists(),
                    'Unauthorized future-stage artifact/status')
            continue
        require(accepted['stage_status'][stage] == 'price_evaluated', 'Completed stage status differs')
        if stage != 'development':
            driver.check_later_gate(stage, frozen, driver.sha(OUT / 'candidate_plan.json'))
        results[stage] = driver.read_result(stage, spec)  # JSON/receipts/binary hashes, no model loading.
        require(accepted['stage_result_sha256'][stage] == driver.sha(OUT / 'evaluation' / stage / 'results.json'),
                'Accepted record is not bound to the saved stage')
        bind_json(dependencies, OUT / 'evaluation' / stage / 'results.json')
        for key in ('input_artifact_sha256', 'output_artifact_sha256'):
            merge(dependencies, results[stage][key])
        merge(dependencies, spec['stage_dependency_sha256'][stage])
    require(not (OUT / 'models/final').exists() and not (OUT / 'evaluation/final').exists(), 'Unexpected final artifacts')
    driver.validate_freeze(frozen, results['development'], spec)
    outcome, reason = driver.final_outcome(results)
    require(accepted['accepted'] is outcome and accepted['stopping_reason'] == reason
            and accepted['selected_candidate'] == frozen['selected_candidate']
            and accepted['deployment_decision'] == ('backtest_accepted_pending_final_release' if outcome else 'keep_v3')
            and accepted['no_reselection_or_parameter_change'] is True and accepted['new_submission_created'] is False,
            'Final decision differs from the already saved results')
    protected = {name: driver.sha(ROOT / name) for name in
                 ('outputs/submission_v2.csv', 'outputs/submission_v3.csv', 'runs/versions.json')}
    driver.c6.verify_hashes(dependencies)
    evidence = {stage: verify_stage(stage, results[stage], spec, dependencies) for stage in stages}
    driver.c6.verify_hashes(dependencies)
    after = {name: driver.sha(ROOT / name) for name in protected}
    require(protected == after and driver.sha(Path(__file__)) == script_sha, 'Inputs/protected files changed during verification')
    proof = {'round': 24, 'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'verification_level': 'saved_raw_output_composition_and_existing_metric_replay',
             'script_path': driver.rel(Path(__file__)), 'script_sha256': script_sha,
             'plan_sha256': driver.sha(OUT / 'candidate_plan.json'),
             'frozen_candidate_sha256': driver.sha(OUT / 'frozen_candidate.json'),
             'accepted_candidate_sha256': driver.sha(OUT / 'accepted_candidate.json'),
             'candidate_accepted': outcome, 'checked_stages': stages,
             'stage_status': accepted['stage_status'], 'stage_evidence': evidence,
             'source_sha256': {**spec['source_sha256'], driver.rel(Path(__file__)): script_sha},
             'source_input_sha256': dependencies, 'protected_before': protected, 'protected_after': after,
             'native_CBM_models_loaded': False, 'training_performed': False,
             'independent_native_model_replay_performed': False,
             'driver_native_replay_claims_checked_as_saved_evidence_only': True,
             'helper_inverse_average_and_driver_blend_called': False,
             'reference_implementation_reused': True, 'reference_calls_per_composition': 1,
             'score_and_gate_definitions_reused': True, 'independently_defined_statistical_protocol': False,
             'only_completed_stages_scored': True, 'new_candidate_or_weight_search': False,
             'historically_untouched': False,
             'data_access_scope': 'Existing aligned data loader for original X87/y; yq only for saved control/candidate score replay. Saved float64 arrays and one frozen reference pickle per completed stage; CBM files hashed only.'}
    driver.locked_write(DESTINATION, proof)
    print(json.dumps({'status': 'passed', 'checked_stages': stages,
                      'proof_path': driver.rel(DESTINATION), 'proof_sha256': driver.sha(DESTINATION)}, ensure_ascii=False))
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    run()
