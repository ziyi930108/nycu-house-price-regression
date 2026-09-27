"""R39 method A: fixed trees/reference, newly supported geo for R8 spatial only.

Freeze then evaluate all three original periods once. No tree fit/load/predict,
map scan, parameter search, CSV or automatic adoption. Old geo/X stay unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import round36_combination as prior
import round38_fixed_average as average
import round6_geo as geo_encoder

ROOT = prior.ROOT
OUT = ROOT / 'outputs/round39/evaluation'
GEO = ROOT / 'outputs/round39/geography'
DOCUMENT = ROOT / 'reports/round39_experiment_plan.md'
STAGES = prior.STAGES
CONTROL = 'v6_control'
CANDIDATE = 'city_road_spatial_half_raw_half_log'
FORMULA = '.5*R8(new_state,new_geo,B_raw,p_raw,diag_raw)+.5*R8(new_state,new_geo,B_log,p_log,diag_log)'
TRIAL_DIRECTION = 'all_three_overall_and_weighted_MAE_not_worse_than_v6; necessary_only_no_automatic_release'
P36 = ROOT / 'outputs/round36/combination_plan.json'
R36 = ROOT / 'outputs/round36/results.json'
P38 = ROOT / 'outputs/round38/plan.json'
R38 = ROOT / 'outputs/round38/results.json'
AUTHORITIES = {
    'outputs/round36/combination_plan.json': '26194404d158613f0837e49c4cdb97b4c9c1ef213b1561cfd7ac80dddc4fc642',
    'outputs/round36/results.json': '133d99e9fdca21ddf97e58fa8c7860ac74f7311bfe56f0005244c3a76791ef9e',
    'outputs/round38/plan.json': 'f885d8e2f37ee8655d1ca7e6af448572b847aaaee3b19cd96badaba297759f4b',
    'outputs/round38/results.json': 'f14d84c7c9163cf894f5302fb6b6f8de0168ff56d2c80aa5bdd0c7f29ae5f947',
}
GEO_MANIFEST_SHA = {
    'development': 'da1d3826ef153feaf2e2041ad4731e949f697f8922968340e09461a07414ad11',
    'confirmation': '556ccfa3ad60c2360df3e5b220f737c6cd1d9a157dca5f02a4238f2eba00c6be',
    'auxiliary': '3a009e2b3b9594fdc8c5a44d747be31448f47782c2307b6c05335979a27e896c',
}
KT = 'outputs/round6/geography/development/train_source_cache_indices.npy'
need, sha, rel, locked, write = prior.require, prior.sha, prior.rel, prior.locked, prior.locked_write
merge, now, save_array = prior.merge, prior.now, prior.save_array
spatial = prior.spatial
PROTECTED = tuple(dict.fromkeys(prior.PROTECTED + tuple(AUTHORITIES) + (
    'outputs/submission_v4_trial.csv', 'outputs/submission_v5_trial.csv', 'outputs/submission_v6_trial.csv',
)))


def sources():
    return merge(average.sources(), {rel(Path(__file__)): sha(Path(__file__))})


def bind_json(inputs, path, expected=None):
    return prior.small(inputs, path, expected)


def vector(path, n):
    values = np.load(path, allow_pickle=False)
    need(values.dtype == np.dtype('float64') and values.shape == (n,)
         and np.isfinite(values).all() and (values >= 0).all(), 'Invalid saved TWD vector: ' + rel(path))
    return values


def exact(left, right, label):
    need(left.dtype == right.dtype == np.dtype('float64') and left.shape == right.shape
         and left.tobytes() == right.tobytes(), label + ' is not bitwise equal')


def frozen_geography(stage, inputs):
    path = GEO / stage / 'manifest.json'
    manifest = bind_json(inputs, path, GEO_MANIFEST_SHA[stage])
    expected_year = {'development': 2022, 'confirmation': 2023, 'auxiliary': 2021}[stage]
    need(manifest['round'] == 39 and manifest['status'] == 'prepared_spatial_only'
         and manifest['spatial_only'] is True and manifest['stage'] == stage
         and manifest['snapshot_year'] == expected_year
         and manifest['encoder_feature_rules'] == geo_encoder.FEATURE_RULES, 'Prepared geo stage/schema differs')
    stamp, origin = pd.Timestamp(manifest['replication_timestamp']), pd.Timestamp(manifest['forecast_origin_utc'])
    need(stamp.tzinfo is not None and origin.tzinfo is not None and stamp.year == expected_year
         and stamp < origin, 'Wrong/unavailable annual geography')
    for group, suffix in [('features', '.pkl'), ('provenance', '_provenance.pkl'), ('added_positions', '_added_positions.npy')]:
        for label in ('fit', 'query'):
            record = manifest[group][label]
            need(record['path'] == rel(path.parent / (label + suffix)), 'Prepared artifact outside exact stage namespace')
            inputs.update(merge(inputs, {record['path']: record['sha256']}))
    inputs.update(merge(inputs, manifest['source_sha256']))
    # The prep agent verified maps/PBF/parser lineage. Preserve those declared
    # identities without treating un-read large upstream objects as this run's I/O.
    return manifest


def freeze():
    need(not (OUT / 'plan.json').exists() and not (OUT / 'run_claim.json').exists(), 'Existing R39 attempt')
    inputs = {rel(DOCUMENT): sha(DOCUMENT)}
    for name, digest in AUTHORITIES.items():
        bind_json(inputs, ROOT / name, digest)
    p36, r36, p38, r38 = [locked(p) for p in (P36, R36, P38, R38)]
    need(prior.sources() == p36['source_sha256'] and average.sources() == p38['source_sha256'], 'Prior sources changed')
    need(r36['status'] == r38['status'] == 'three_periods_completed_no_adoption_decision'
         and r36['checked_stages'] == r38['checked_stages'] == list(STAGES), 'Three-period control incomplete')
    need(p36['target_group_weights'] == p38['target_group_weights']
         and p36['split_contract'] == p38['split_contract'] and p36['splits'] == p38['splits'], 'Prior split/weight contracts differ')
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', KT):
        inputs[name] = p36['dependency_sha256'][name]
    bindings, prepared = {}, {}
    for stage in STAGES:
        binding = copy.deepcopy(p36['bindings'][stage])
        known = merge(p36['dependency_sha256'], p36['stage_dependency_sha256'][stage])
        old36path = R36.parent / stage / 'results.json'
        old36 = bind_json(inputs, old36path, r36['stage_result_sha256'][stage])
        need(old36['status'] == 'completed' and old36['plan_sha256'] == sha(P36), 'Wrong R36 stage result')
        old38path = R38.parent / stage / 'results.json'
        old38 = bind_json(inputs, old38path)
        need(old38 == r38['stages'][stage] and old38['stage'] == stage, 'R38 stage differs from completed summary')
        for name in ('fit_indices', 'query_indices', prior.V3, prior.SPATIAL, prior.LOG, prior.COMBINED):
            path = old36path.parent / (name + '.npy')
            inputs[rel(path)] = old36['artifact_sha256'][rel(path)]
        inputs[rel(old38path.parent / 'predictions.npy')] = old38['prediction_sha256']
        paths = [ROOT / binding[name] for name in ('spatial_state', 'spatial_state_summary', 'log_component', 'log_metadata', 'log_prediction')]
        base = prior.c6.baseline_folder(stage)
        paths += [base / name for name in ('catboost_raw.npy', 'lightgbm_l1_unit63.npy', 'reference_v2.pkl', 'fit_indices.npy', 'check_indices.npy')]
        original_model = prior.c6.OUT / 'models' / stage / 'geography_catboost'
        paths += [original_model / name for name in ('predictions.npy', 'fit_indices.npy', 'query_indices.npy')]
        for key in ('spatial_indices_folder', 'log_indices_folder'):
            paths += [ROOT / binding[key] / (name + '_indices.npy') for name in ('fit', 'query')]
        old_geo = prior.c6.OUT / 'geography' / stage / 'train.pkl'
        paths += [old_geo]
        for path in paths:
            inputs.update(merge(inputs, {rel(path): known[rel(path)]}))
        manifest = frozen_geography(stage, inputs)
        need(manifest['original_manifest_sha256'] == known[rel(old_geo.parent / 'manifest.json')], 'Prepared original manifest identity differs')
        for label in ('fit', 'query'):
            idx = manifest['indices'][label]
            expected_idx = original_model / (label + '_indices.npy')
            need(idx['path'] == rel(expected_idx) and idx['sha256'] == known[rel(expected_idx)]
                 and idx['content_sha256'] == p36['split_contract'][stage][label + '_indices_content_sha256']
                 and idx['rows'] == p36['split_contract'][stage][label + '_rows'], 'Prepared period indices differ')
            need(manifest['features'][label]['original_geo_path'] == rel(old_geo)
                 and manifest['features'][label]['original_geo_sha256'] == known[rel(old_geo)]
                 and manifest['features'][label]['rows'] == p36['split_contract'][stage][label + '_rows'], 'Prepared original frame identity differs')
        binding['original_geo'] = rel(old_geo)
        bindings[stage], prepared[stage] = binding, manifest
    value = {'round': 39, 'created_at_utc': now(), 'method': 'spatial_only_city_road_fallback',
             'stage_order': list(STAGES), 'formula': FORMULA, 'control': CONTROL, 'candidate': CANDIDATE,
             'consider_trial_direction': TRIAL_DIRECTION, 'source_sha256': sources(),
             'input_sha256': inputs, 'bindings': bindings, 'prepared_geography': prepared,
             'runtime': average.runtime(), 'spatial_parameters': spatial.parameters(),
             'splits': p36['splits'], 'split_contract': p36['split_contract'], 'target_group_weights': p36['target_group_weights'],
             'protected_sha256': {name: sha(ROOT / name) for name in PROTECTED},
             'tree_models_trained_loaded_or_predicted': 0, 'automatic_release_allowed': False,
             'all_three_periods_regardless_of_scores': True, 'fresh_blind_validation': False,
             'upstream_geographic_maps_rehashed_by_evaluator': False}
    need(value['source_sha256'] == sources() and value['input_sha256'][rel(DOCUMENT)] == sha(DOCUMENT), 'Source/document changed during freeze')
    write(OUT / 'plan.json', value)
    print('R39_plan_frozen', sha(OUT / 'plan.json'), flush=True)


def load_training_rows(plan):
    Xraw = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
    raw = pd.read_pickle(ROOT / 'data_cache/train.pkl')
    kt = np.load(ROOT / KT, allow_pickle=False)
    schema = prior.c6.read(ROOT / 'reports/feature_schema.json')
    need(len(Xraw) == len(raw) == 455784 and Xraw.index.equals(pd.RangeIndex(455784))
         and raw.index.equals(Xraw.index) and list(Xraw.columns) == schema['columns'], 'Original X/y cache alignment differs')
    need(kt.dtype == np.dtype('int64') and kt.shape == (455255,) and (np.diff(kt) > 0).all()
         and kt.min() >= 0 and kt.max() < len(raw), 'Canonical index differs')
    return Xraw.iloc[kt].reset_index(drop=True), raw['總價元'].to_numpy(dtype='float64')[kt]


def load_geo(manifest, label, original):
    record = manifest['features'][label]
    frame = pd.read_pickle(ROOT / record['path'])
    geo_encoder.check_geo(frame)
    need(frame.index.equals(original.index) and frame.dtypes.equals(original.dtypes)
         and spatial.frame_sha(frame) == record['frame_sha256']
         and {c: str(frame[c].dtype) for c in frame} == record['dtypes']
         and spatial.frame_sha(original) == record['original_partition_frame_sha256'], 'New/original geo frame contract differs')
    added_record = manifest['added_positions'][label]
    positions = np.load(ROOT / added_record['path'], allow_pickle=False)
    need(positions.dtype == np.dtype('int64') and positions.ndim == 1 and len(positions) == added_record['rows']
         and (np.diff(positions) > 0).all() and (not len(positions) or (positions.min() >= 0 and positions.max() < len(frame)))
         and prior.log_run.original.value_sha(positions, '<i8') == added_record['content_sha256'], 'Added-position identity differs')
    mask = np.zeros(len(frame), dtype=bool)
    mask[positions] = True
    expected = original.geo_has_position.eq(0).to_numpy() & frame.geo_has_position.eq(1).to_numpy()
    need(np.array_equal(mask, expected), 'Added-position mask differs')
    need(original.loc[mask, ['geo_latitude', 'geo_longitude']].isna().all().all()
         and frame.loc[mask, 'geo_match_status'].eq('road_level').all(), 'Fallback changed an existing coordinate or quality')
    pd.testing.assert_frame_equal(frame.loc[~mask], original.loc[~mask], check_exact=True)
    for column in geo_encoder.NUMERIC:
        need(frame.loc[~mask, column].to_numpy().tobytes() == original.loc[~mask, column].to_numpy().tobytes(), 'Existing numeric geo bits changed')
    return frame, mask


def impact_groups(y, control, candidate, old_geo, added):
    masks = {'new_query_position': added, 'original_query_position': old_geo.geo_has_position.eq(1).to_numpy(),
             'still_without_query_position': old_geo.geo_has_position.eq(0).to_numpy() & ~added}
    need(np.all(sum(mask.astype('int8') for mask in masks.values()) == 1), 'Affected groups must partition full query')
    output = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        a, b = control[mask], candidate[mask]
        ma = None if not n else float(np.mean(np.abs(y[mask] - a)))
        mb = None if not n else float(np.mean(np.abs(y[mask] - b)))
        output[name] = {'rows': n, 'changed_rows': int(np.count_nonzero(a != b)), 'control_mae': ma, 'candidate_mae': mb,
                        'MAE_delta_TWD': None if not n else mb - ma,
                        'mean_absolute_prediction_change_TWD': None if not n else float(np.mean(np.abs(b - a))),
                        'prediction_change_is_not_error': True}
    return output



def evaluate(stage, plan, Xall, yall):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    binding, manifest = plan['bindings'][stage], plan['prepared_geography'][stage]
    old36folder, old38folder = R36.parent / stage, R38.parent / stage
    old36, old38 = locked(old36folder / 'results.json'), locked(old38folder / 'results.json')
    fi, qi = [np.load(old36folder / (name + '_indices.npy'), allow_pickle=False) for name in ('fit', 'query')]
    for label, values in (('fit', fi), ('query', qi)):
        spec = plan['split_contract'][stage]
        need(values.dtype == np.dtype('int64') and values.shape == (spec[label + '_rows'],)
             and (np.diff(values) > 0).all() and values.min() >= 0 and values.max() < len(Xall)
             and prior.log_run.original.value_sha(values, '<i8') == spec[label + '_indices_content_sha256'], 'Original indices differ')
        save_array(folder / (label + '_indices.npy'), values)
    need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
    X, Xq, yq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True), yall[qi]
    base_folder = prior.c6.baseline_folder(stage)
    original_model = prior.c6.OUT / 'models' / stage / 'geography_catboost'
    prior.e6.verify_indices(base_folder, fi, qi, 'check_indices.npy')
    prior.e6.verify_indices(original_model, fi, qi, 'query_indices.npy')
    for key in ('spatial_indices_folder', 'log_indices_folder'):
        prior.e6.verify_indices(ROOT / binding[key], fi, qi, 'query_indices.npy')
    original_geo = pd.read_pickle(ROOT / binding['original_geo'])
    geo_encoder.check_geo(original_geo)
    need(len(original_geo) == len(Xall) and original_geo.index.equals(Xall.index), 'Canonical geo/X alignment differs')
    geo, geoq = original_geo.iloc[fi].reset_index(drop=True), original_geo.iloc[qi].reset_index(drop=True)
    new_geo, fit_added = load_geo(manifest, 'fit', geo)
    new_geoq, query_added = load_geo(manifest, 'query', geoq)
    frames = {'fit_X': X, 'query_X': Xq, 'original_fit_geo': geo, 'original_query_geo': geoq,
              'new_fit_geo': new_geo, 'new_query_geo': new_geoq}
    before = {name: spatial.frame_sha(frame) for name, frame in frames.items()}
    summary = prior.c6.read(ROOT / binding['spatial_state_summary'])
    meta = locked(ROOT / binding['log_metadata'])
    need(summary['fit_X_sha256'] == before['fit_X'] and summary['fit_geo_sha256'] == before['original_fit_geo']
         and meta['fit_original_feature_values_sha256'] == before['fit_X']
         and meta['fit_geo_feature_values_sha256'] == before['original_fit_geo'], 'Original fit state frames differ')
    reference_path = base_folder / 'reference_v2.pkl'
    with reference_path.open('rb') as stream:
        reference = pickle.load(stream)
    need(reference['fit_report']['fitting_rows'] == len(X)
         and reference['fit_report']['hedonic']['maximum_fit_day_index'] == float(np.nanmax(X.trade_day_index)), 'Reference fit/date differs')
    with (ROOT / binding['spatial_state']).open('rb') as stream:
        old_state = pickle.load(stream)
    need(old_state['parameters'] == spatial.parameters() and old_state['state_summary'] == summary
         and summary['existing_reference_residual_sha256'] == hashlib.sha256(np.asarray(reference['reference_residual'], dtype='<f8').tobytes()).hexdigest(), 'Original spatial state/reference differs')
    old = vector(base_folder / 'catboost_raw.npy', len(qi))
    unit = vector(base_folder / 'lightgbm_l1_unit63.npy', len(qi))
    raw_geo = vector(original_model / 'predictions.npy', len(qi))
    log_geo = vector(ROOT / binding['log_component'], len(qi))
    base_raw = .75 * (.5 * old + .5 * raw_geo) + .25 * unit
    base_log = .75 * (.5 * old + .5 * log_geo) + .25 * unit
    p_raw, diag_raw, raw_report = prior.log_run.original.predict_reference(reference, Xq, base_raw, variant='trend', return_diagnostics=True)
    p_log, diag_log, log_report = prior.log_run.original.predict_reference(reference, Xq, base_log, variant='trend', return_diagnostics=True)
    exact(p_raw, vector(old36folder / (prior.V3 + '.npy'), len(qi)), 'Original v3 reference')
    exact(p_log, vector(old36folder / (prior.LOG + '.npy'), len(qi)), 'Original log reference')
    old_raw, _, _ = spatial.predict(old_state, Xq, geoq, base_raw, p_raw, diag_raw)
    old_log, _, _ = spatial.predict(old_state, Xq, geoq, base_log, p_log, diag_log)
    exact(old_raw, vector(old36folder / (prior.SPATIAL + '.npy'), len(qi)), 'Original raw-spatial branch')
    exact(old_log, vector(old36folder / (prior.COMBINED + '.npy'), len(qi)), 'Original log-spatial branch')
    control = .5 * old_raw + .5 * old_log
    exact(control, vector(old38folder / 'predictions.npy', len(qi)), 'Original full-query v6 control')
    score = lambda values: prior.log_run.original.score(yq, values, X, Xq, plan['splits'][stage][0], plan['target_group_weights'])
    old_metrics = {name: score(values) for name, values in ((prior.V3, p_raw), (prior.LOG, p_log), (prior.SPATIAL, old_raw), (prior.COMBINED, old_log))}
    need(all(value == old36['metrics'][name] for name, value in old_metrics.items()), 'R36 original full metrics differ')
    control_metrics = score(control)
    need(control_metrics == old38['metrics'][average.CANDIDATE], 'R38 original full metrics differ')
    new_state, new_summary = spatial.build(reference, X, new_geo)
    need(new_summary['fit_X_sha256'] == before['fit_X'] and new_summary['fit_geo_sha256'] == before['new_fit_geo']
         and new_summary['existing_reference_residual_sha256'] == summary['existing_reference_residual_sha256']
         and new_summary['query_used_in_build'] is False and new_summary['new_reference_fit_performed'] is False, 'New spatial fit-only state differs')
    with (folder / 'spatial_state.pkl').open('xb') as stream:
        pickle.dump(new_state, stream, protocol=5)
    write(folder / 'state_summary.json', new_summary)
    predictions = {CONTROL: control}
    reports = {}
    for branch, raw_base, baseline, diagnostics in (('raw', base_raw, p_raw, diag_raw), ('log', base_log, p_log, diag_log)):
        values, detail, report = spatial.predict(new_state, Xq, new_geoq, raw_base, baseline, diagnostics)
        fallback = ~detail.applied.to_numpy(dtype=bool)
        exact(values[fallback], baseline[fallback], branch + ' fallback baseline')
        need(values.dtype == np.dtype('float64') and values.shape == (len(qi),)
             and np.isfinite(values).all() and (values >= 0).all(), 'Invalid spatial branch')
        predictions['new_' + branch + '_spatial'] = values
        reports[branch] = report
        with (folder / (branch + '_diagnostics.pkl')).open('xb') as stream:
            pickle.dump(detail, stream, protocol=5)
        write(folder / (branch + '_prediction_report.json'), report)
    candidate = .5 * predictions['new_raw_spatial'] + .5 * predictions['new_log_spatial']
    need(np.isfinite(candidate).all() and (candidate >= 0).all(), 'Invalid candidate mean')
    predictions[CANDIDATE] = candidate
    for name, values in predictions.items():
        save_array(folder / (name + '.npy'), values)
        exact(values, vector(folder / (name + '.npy'), len(qi)), 'Saved ' + name)
    metrics = {CONTROL: control_metrics, CANDIDATE: score(candidate)}
    affected = impact_groups(yq, control, candidate, geoq, query_added)
    need(before == {name: spatial.frame_sha(frame) for name, frame in frames.items()}, 'Original/new feature frames mutated')
    checks = {name + '_not_worse': metrics[CANDIDATE][name] <= metrics[CONTROL][name] for name in ('overall_mae', 'weighted_mae')}
    result = {'round': 39, 'stage': stage, 'status': 'completed', 'created_at_utc': now(),
              'plan_sha256': sha(OUT / 'plan.json'), 'fit_rows': len(fi), 'query_rows': len(qi), 'dates': plan['splits'][stage],
              'metrics': metrics, 'candidate_vs_v6': prior.log_run.differences(control_metrics, metrics[CANDIDATE]),
              'unchanged_prior_metrics': old_metrics, 'all_prior_controls_bitwise_equal': True,
              'original_full_metrics_reproduced': True, 'control_spatial_predictions_per_branch': 1,
              'new_spatial_predictions_per_branch': 1, 'new_spatial_state_builds': 1,
              'fit_positions_added': int(fit_added.sum()), 'query_positions_added': int(query_added.sum()), 'affected_groups': affected,
              'reference_reports': {'raw': raw_report, 'log': log_report}, 'prediction_reports': reports,
              'input_frame_sha256': before, 'geo_manifest_sha256': plan['input_sha256'][rel(GEO / stage / 'manifest.json')],
              'source_sha256': plan['source_sha256'], 'reference_path': rel(reference_path), 'reference_sha256': sha(reference_path),
              'unapplied_rows_preserve_their_branch_baseline_bitwise': True, 'reference_trend_applications_per_branch': 1,
              'tree_models_trained_loaded_or_predicted': 0, 'query_labels_used_in_build_or_predict': False,
              'consider_trial_necessary_checks': checks, 'formal_adoption_decision': None, 'new_submission_generated': False,
              'elapsed_seconds': time.monotonic() - started,
              'artifact_sha256': {rel(path): sha(path) for path in sorted(folder.iterdir()) if path.is_file()}}
    write(folder / 'results.json', result)
    print(json.dumps({'event': 'period_completed', 'stage': stage, 'candidate_vs_v6': result['candidate_vs_v6']}, ensure_ascii=False), flush=True)
    return result


def run():
    plan = locked(OUT / 'plan.json')
    need(plan['round'] == 39 and plan['stage_order'] == list(STAGES) and plan['formula'] == FORMULA
         and plan['control'] == CONTROL and plan['candidate'] == CANDIDATE
         and plan['consider_trial_direction'] == TRIAL_DIRECTION and plan['spatial_parameters'] == spatial.parameters()
         and plan['source_sha256'] == sources() and plan['runtime'] == average.runtime()
         and plan['all_three_periods_regardless_of_scores'] is True and plan['automatic_release_allowed'] is False, 'Frozen R39 contract changed')
    need(not (OUT / 'run_claim.json').exists() and not (OUT / 'results.json').exists()
         and not any((OUT / stage).exists() for stage in STAGES), 'Existing attempt; no automatic retry')
    write(OUT / 'run_claim.json', {'round': 39, 'started_at_utc': now(), 'plan_sha256': sha(OUT / 'plan.json'), 'automatic_retry': False})
    start = time.monotonic()
    result = {'round': 39, 'status': 'incomplete', 'plan_sha256': sha(OUT / 'plan.json'),
              'stage_order': list(STAGES), 'checked_stages': [], 'stages': {}, 'stage_result_sha256': {},
              'source_sha256': plan['source_sha256'], 'protected_before': plan['protected_sha256'],
              'tree_models_trained_loaded_or_predicted': 0, 'formal_adoption_decision': None,
              'new_submission_generated': False, 'registry_updated': False, 'recommendation_changed': False,
              'all_periods_not_worse_necessary_condition': None, 'fresh_blind_validation': False}
    phase = 'verify_all_actual_inputs_before_loading'
    try:
        prior.c6.verify_hashes(merge(plan['input_sha256'], plan['source_sha256'], plan['protected_sha256']))
        Xall, yall = load_training_rows(plan)
        for stage in STAGES:  # Never branch on a prior period's score.
            phase = stage
            item = evaluate(stage, plan, Xall, yall)
            result['checked_stages'].append(stage)
            result['stage_result_sha256'][stage] = sha(OUT / stage / 'results.json')
            result['stages'][stage] = {key: item[key] for key in ('metrics', 'candidate_vs_v6', 'affected_groups', 'fit_positions_added', 'query_positions_added', 'consider_trial_necessary_checks')}
        result['all_periods_not_worse_necessary_condition'] = all(all(item['consider_trial_necessary_checks'].values()) for item in result['stages'].values())
        prior.c6.verify_hashes(merge(plan['source_sha256'], plan['protected_sha256']))
        for binding in plan['bindings'].values():
            name = binding['original_geo']
            need(sha(ROOT / name) == plan['input_sha256'][name], 'Original geographic artifact changed')
        result['protected_after'] = {name: sha(ROOT / name) for name in PROTECTED}
        result['status'] = 'three_periods_completed_no_adoption_decision'
    except Exception as exc:
        result.update(status='incomplete_technical_failure', failure_phase=phase, error=str(exc), traceback=traceback.format_exc())
        result['protected_after'] = {name: sha(ROOT / name) for name in PROTECTED}
    result.update(completed_at_utc=now(), elapsed_seconds=time.monotonic() - start)
    write(OUT / 'results.json', result)
    print(result['status'], sha(OUT / 'results.json'), flush=True)
    return 0 if result['status'] == 'three_periods_completed_no_adoption_decision' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        raise SystemExit(run())
