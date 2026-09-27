"""R41 method B: six fixed raw/log geo CatBoost fits using R39 prepared geo.

New namespace and authority only. Freeze, then run all three periods regardless
of scores. No old-tree replay, map scan, tuning, CSV, or automatic adoption.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import pickle
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import round39_city_road_experiment as old
import round12_log_geo as log_component

ROOT = old.ROOT
OUT = ROOT / 'outputs/round41'
DOCUMENT = ROOT / 'reports/round41_retrain_plan.md'
P39 = old.OUT / 'plan.json'
R39 = old.OUT / 'results.json'
P39_SHA = '110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13'
R39_SHA = '1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27'
SOURCE39_SHA = '8c26637712304043ed49aca18f53099d5ec1249f71718458a17f1cbc35ace4fd'
RAW_SHA = 'df1454829b919e2a10ac264dd4e3f597367187174607a6d32abb20f40ef060a0'
LOG_SHA = 'e16fbe33928bd1a635e5de8bd717f223263847b718aef373fb933224b13d6c42'
CONFIG_PATH = ROOT / 'runs/v1/models/final_config.json'
CONFIG_SHA = '2ca4258f363d1937df50aed72f5b11dc32b7e70d9dd5fb57fb6f375673a2a425'
R12_PLAN = ROOT / 'outputs/round12/candidate_plan.json'
R12_PLAN_SHA = '5b37d819e1fffb624b26740815fae10fb6597d5da84613026a6436132df3e25b'
STAGES = ('development', 'confirmation', 'auxiliary')
BRANCHES = ('raw', 'log')
CONTROL = 'v7_control'
CANDIDATE = 'retrained_city_road_half_raw_half_log'
FORMULA = {
    'base_per_branch': '.75*(.5*saved_old_raw+.5*new_geo_TWD)+.25*saved_old_unit',
    'reference_per_branch': 'original87 reference trend once; same-call diagnostics',
    'spatial_per_branch': 'unchanged R8 predict with saved R39 fit-only spatial state and R39 query geo',
    'candidate': '.5*new_raw_spatial+.5*new_log_spatial',
    'raw_target': 'y_TWD/1e6', 'raw_inverse': 'max(z*1e6,0)',
    'log_target': 'log1p(y_TWD/1e6)', 'log_inverse': 'max(expm1(z)*1e6,0)',
}
TRIAL_RULES = {'each_period_overall_and_weighted_not_worse_than_v7': True,
               'each_nonempty_forecast_quarter_max_relative_degradation': .05,
               'each_stratum_min_rows': 500, 'each_large_stratum_max_relative_degradation': .05,
               'necessary_only_no_automatic_adoption': True}
PUBLIC_CONTEXT = {
    'method_selected_before_latest_public_feedback': True,
    'latest_feedback_known_before_source_finalization': True,
    'feedback_path': 'outputs/kaggle_feedback/public_submissions_20260926.json',
    'feedback_sha256': '8e2867570eebff873a16c407b0827ad452eb5a03dddfe7d7520bf3101a8c65fd',
    'reported_public_mae': {'v3': 2832598.6108, 'v4_trial': 2823347.1398,
                           'v5_trial': 2741740.6242, 'v6_trial': 2769962.7545, 'v7_trial': 2770232.8164},
    'provenance': 'user filename-bearing screenshots relayed by root before this source was finalized',
    'public_used_to_change_method_weights_or_periods': False,
    'private_scores_unknown': True,
}
raw_component = old.geo_encoder
spatial = old.spatial
original = old.prior.log_run.original
c6 = old.prior.c6
need, sha, rel, locked, write = old.need, old.sha, old.rel, old.locked, old.write
merge, now, save_array, exact = old.merge, old.now, old.save_array, old.exact
PROTECTED = tuple(dict.fromkeys(old.PROTECTED + (
    'outputs/submission_v7_trial.csv', rel(P39), rel(R39),
)))


def sources():
    return merge(old.sources(), {rel(Path(__file__)): sha(Path(__file__))})


def runtime():
    return old.average.runtime()


def method_contract():
    need(sha(Path(old.__file__)) == SOURCE39_SHA and sha(Path(raw_component.__file__)) == RAW_SHA
         and sha(Path(log_component.__file__)) == LOG_SHA, 'Immutable helper/source differs')
    need(sha(CONFIG_PATH) == CONFIG_SHA, 'Original parameter file differs')
    config = raw_component.parameters()
    need(config == log_component.parameters(), 'Raw/log baseline parameters differ')
    for key, value in {'iterations': 2999, 'depth': 8, 'learning_rate': .065,
                       'random_seed': 42, 'loss_function': 'MAE', 'task_type': 'GPU', 'boosting_type': 'Plain'}.items():
        need(config[key] == value, 'Original parameter differs: ' + key)
    return config


def freeze():
    need(not (OUT / 'plan.json').exists() and not (OUT / 'run_claim.json').exists()
         and not any((OUT / stage).exists() for stage in STAGES), 'Existing R41 attempt')
    need(sha(P39) == P39_SHA and sha(R39) == R39_SHA, 'R39 authority identity differs')
    p39, r39 = locked(P39), locked(R39)
    need(p39['source_sha256'] == old.sources() and p39['runtime'] == runtime(), 'R39 source/runtime changed')
    need(r39['status'] == 'three_periods_completed_no_adoption_decision'
         and r39['plan_sha256'] == P39_SHA and r39['checked_stages'] == list(STAGES)
         and p39['stage_order'] == list(STAGES), 'Incomplete/wrong R39 authority')
    config = method_contract()
    inputs = merge(p39['input_sha256'], {rel(DOCUMENT): sha(DOCUMENT), rel(CONFIG_PATH): CONFIG_SHA})
    r12 = old.bind_json(inputs, R12_PLAN, R12_PLAN_SHA)
    need(r12['parameters'] == config and r12['dependency_sha256'][rel(CONFIG_PATH)] == CONFIG_SHA,
         'Original R12 full parameter contract differs')
    old.bind_json(inputs, ROOT / PUBLIC_CONTEXT['feedback_path'], PUBLIC_CONTEXT['feedback_sha256'])
    for path in (P39, R39):
        old.bind_json(inputs, path)
    stages = {}
    for stage in STAGES:
        folder = old.OUT / stage
        result = old.bind_json(inputs, folder / 'results.json', r39['stage_result_sha256'][stage])
        need(result['round'] == 39 and result['stage'] == stage and result['status'] == 'completed'
             and result['plan_sha256'] == P39_SHA and result['metrics'] == r39['stages'][stage]['metrics'],
             'Wrong R39 stage authority')
        paths = {key: rel(folder / name) for key, name in {
            'control': old.CANDIDATE + '.npy', 'raw_spatial': 'new_raw_spatial.npy',
            'log_spatial': 'new_log_spatial.npy', 'spatial_state': 'spatial_state.pkl',
            'state_summary': 'state_summary.json', 'fit_indices': 'fit_indices.npy',
            'query_indices': 'query_indices.npy'}.items()}
        for name in paths.values():
            inputs = merge(inputs, {name: result['artifact_sha256'][name]})
        receipt = Path(paths['state_summary']).with_suffix('.sha256').as_posix()
        inputs = merge(inputs, {receipt: result['artifact_sha256'][receipt]})
        paths['result'] = rel(folder / 'results.json')
        stages[stage] = paths
    value = {'round': 41, 'kind': 'R39_geo_in_both_raw_and_log_tree_training', 'created_at_utc': now(),
             'stage_order': list(STAGES), 'branch_order': list(BRANCHES), 'new_tree_models_planned': 6,
             'candidate': CANDIDATE, 'control': CONTROL, 'formulas': FORMULA, 'trial_necessary_rules': TRIAL_RULES,
             'parameters': config, 'raw_feature_rules': copy.deepcopy(raw_component.FEATURE_RULES),
             'log_feature_rules': copy.deepcopy(log_component.FEATURE_RULES),
             'source_sha256': sources(), 'input_sha256': inputs, 'runtime': runtime(),
             'R39_plan_sha256': P39_SHA, 'R39_results_sha256': R39_SHA,
             'bindings': p39['bindings'], 'prepared_geography': p39['prepared_geography'],
             'stage_artifacts': stages, 'splits': p39['splits'], 'split_contract': p39['split_contract'],
             'target_group_weights': p39['target_group_weights'], 'spatial_parameters': spatial.parameters(),
             'feature_count': 95, 'categorical_count': 23, 'sample_weight': 'uniform',
             'all_six_fits_regardless_of_scores': True, 'query_labels_passed_to_fit': False,
             'full_train_labels_deserialized': True, 'fit_uses_eval_set': False, 'early_stopping_used': False,
             'fresh_blind_validation': False, 'GPU_retraining_can_change_all_query_predictions': True,
             'difference_not_attributable_only_to_geography': True, 'public_feedback_context': PUBLIC_CONTEXT,
             'old_tree_models_loaded_or_replayed': False, 'map_scan_performed': False,
             'automatic_release_allowed': False, 'formal_adoption_decision': None,
             'protected_sha256': {name: sha(ROOT / name) for name in PROTECTED},
             'large_input_identity_at_freeze': 'inherited immutable expected SHA; actual bytes checked before run loads'}
    need(value['source_sha256'] == sources() and inputs[rel(DOCUMENT)] == sha(DOCUMENT)
         and sha(P39) == P39_SHA and sha(R39) == R39_SHA, 'Inputs/source changed during freeze')
    write(OUT / 'plan.json', value)
    print(json.dumps({'status': 'frozen', 'plan_sha256': sha(OUT / 'plan.json'),
                      'planned_fits': 6, 'input_count': len(inputs)}, ensure_ascii=False), flush=True)


def read_plan():
    p = locked(OUT / 'plan.json')
    need(p['round'] == 41 and p['stage_order'] == list(STAGES) and p['branch_order'] == list(BRANCHES)
         and p['candidate'] == CANDIDATE and p['control'] == CONTROL and p['formulas'] == FORMULA
         and p['trial_necessary_rules'] == TRIAL_RULES and p['parameters'] == method_contract()
         and p['source_sha256'] == sources() and p['runtime'] == runtime()
         and p['spatial_parameters'] == spatial.parameters() and p['public_feedback_context'] == PUBLIC_CONTEXT,
         'Frozen R41 method/source/runtime changed')
    need(p['all_six_fits_regardless_of_scores'] is True and p['new_tree_models_planned'] == 6
         and p['automatic_release_allowed'] is False and p['formal_adoption_decision'] is None,
         'Frozen scope changed')
    return p


def value_sha(values, dtype='<f8'):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def native_check(model, state, cats):
    params = model.get_all_params()
    names = state['all_feature_names']
    need(int(model.tree_count_) == 2999 and list(model.feature_names_) == names and len(names) == 95,
         'Native tree count/features differ')
    need(sorted(model.get_cat_feature_indices()) == sorted(names.index(c) for c in cats), 'Native categorical positions differ')
    for key, val in {'depth': 8, 'random_seed': 42, 'loss_function': 'MAE',
                     'boosting_type': 'Plain', 'grow_policy': 'SymmetricTree'}.items():
        need(params[key] == val, 'Native configuration differs: ' + key)
    need(math.isclose(float(params['learning_rate']), .065, rel_tol=0, abs_tol=1e-8), 'Native learning rate differs')
    leaves = np.asarray(model.get_tree_leaf_counts())
    need(leaves.shape == (2999,) and np.issubdtype(leaves.dtype, np.integer)
         and (leaves > 0).all() and (leaves <= 256).all(), 'Native capacity differs')
    return {'tree_count': 2999, 'feature_count': 95, 'categorical_count': len(cats),
            'depth': 8, 'random_seed': 42, 'loss_function': 'MAE', 'boosting_type': 'Plain',
            'grow_policy': 'SymmetricTree', 'learning_rate': float(params['learning_rate']),
            'tree_leaf_counts_sha256': value_sha(leaves, '<i8')}


def native_predict(branch, model, state, X, geo, cats):
    if branch == 'raw':
        raw_component.check_state(state, X)
    else:
        log_component.check_state(state, X, cats)
    augmented = raw_component.augment_features(X, geo)
    categories = raw_component.categories(X, cats)
    native_check(model, state, categories)
    z = np.asarray(model.predict(raw_component.Pool(augmented, cat_features=categories, thread_count=4),
                                 thread_count=4), dtype='float64')
    need(z.shape == (len(X),) and np.isfinite(z).all(), 'Invalid native query output')
    if branch == 'log':
        pred = log_component.inverse_transform(z)
    else:
        with np.errstate(over='raise', invalid='raise'):
            pred = np.maximum(z * 1e6, 0)
        need(np.isfinite(pred).all(), 'Raw inverse overflow')
    return z, pred


def fit_branch(stage, branch, plan, X, y, geo, Xq, geoq, cats, fi, qi):
    folder = OUT / stage / 'models' / branch
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    component = raw_component if branch == 'raw' else log_component
    before = {'fit_X': raw_component.frame_sha(X), 'fit_geo': raw_component.frame_sha(geo),
              'query_X': raw_component.frame_sha(Xq), 'query_geo': raw_component.frame_sha(geoq)}
    target_sha = value_sha(y)
    write(folder / 'fit_started.json', {'round': 41, 'stage': stage, 'branch': branch,
          'created_at_utc': now(), 'fit_rows': len(X), 'query_rows': len(Xq), 'plan_sha256': sha(OUT / 'plan.json'),
          'fit_target_values_sha256': target_sha, 'fit_indices_content_sha256': value_sha(fi, '<i8'),
          'query_indices_content_sha256': value_sha(qi, '<i8'), 'parameters': plan['parameters'],
          'query_labels_passed_to_fit': False, 'automatic_retry': False})
    print(json.dumps({'event': 'fit_started', 'stage': stage, 'branch': branch}), flush=True)
    model, state, metadata = component.fit_model(X, y, geo, cats)
    # Preserve the expensive completed fit before inference, reload or scoring.
    model.save_model(str(folder / 'model.cbm'))
    write(folder / 'feature_state.json', state)
    write(folder / 'fit_metadata.json', metadata)
    print(json.dumps({'event': 'fit_saved', 'stage': stage, 'branch': branch,
                      'fit_seconds': metadata['elapsed_seconds']}, ensure_ascii=False), flush=True)
    geo_state = state if branch == 'raw' else state['geo_state']
    need(metadata['parameters'] == plan['parameters'] and metadata['sample_weight'] == 'uniform'
         and metadata['feature_count'] == 95 and metadata['categorical_feature_count'] == 23
         and metadata['fit_target_values_sha256'] == target_sha
         and geo_state['fit_rows'] == len(X) and geo_state['fit_original_feature_values_sha256'] == before['fit_X']
         and geo_state['fit_geo_feature_values_sha256'] == before['fit_geo'], 'Actual fit state/target differs')
    need(metadata['fit_uses_eval_set'] is False and metadata['early_stopping_used'] is False
         and geo_state['fit_state_uses_y'] is False and geo_state['fit_state_uses_query'] is False,
         'Forbidden training/state behavior')
    transformed = y / 1e6 if branch == 'raw' else log_component.transform_target(y)
    if branch == 'log':
        need(metadata['fit_transformed_target_values_sha256'] == value_sha(transformed), 'Log target differs')
    native_fit = native_check(model, state, raw_component.categories(X, cats))
    z, pred = native_predict(branch, model, state, Xq, geoq, cats)
    for name, values in [('raw_z', z), ('predictions', pred)]:
        save_array(folder / (name + '.npy'), values)
    del model
    reloaded = raw_component.CatBoostRegressor()
    reloaded.load_model(str(folder / 'model.cbm'))
    restored = locked(folder / 'feature_state.json')
    need(restored == state, 'Saved feature state differs')
    native_loaded = native_check(reloaded, restored, raw_component.categories(X, cats))
    for key in native_fit:
        if key != 'learning_rate':
            need(native_fit[key] == native_loaded[key], 'Reloaded capacity differs: ' + key)
    replay_z, replay = native_predict(branch, reloaded, restored, Xq, geoq, cats)
    exact(z, replay_z, 'Full-query native raw reload')
    exact(pred, replay, 'Full-query TWD reload')
    del reloaded
    for name, values in [('raw_z', z), ('predictions', pred)]:
        exact(values, np.load(folder / (name + '.npy'), allow_pickle=False), 'Saved ' + name)
    need(value_sha(y) == target_sha and before == {
        'fit_X': raw_component.frame_sha(X), 'fit_geo': raw_component.frame_sha(geo),
        'query_X': raw_component.frame_sha(Xq), 'query_geo': raw_component.frame_sha(geoq)}, 'Fit/query inputs changed')
    final = {'round': 41, 'stage': stage, 'branch': branch, 'status': 'fit_saved_full_query_reload_verified',
             'plan_sha256': sha(OUT / 'plan.json'), 'helper_metadata': metadata,
             'source_sha256': plan['source_sha256'], 'fit_rows': len(X), 'query_rows': len(Xq),
             'parameters': plan['parameters'], 'native_fitted': native_fit, 'native_reloaded': native_loaded,
             'frame_sha256': before, 'fit_target_values_sha256': target_sha,
             'fit_transformed_target_values_sha256': value_sha(transformed),
             'fit_indices_content_sha256': value_sha(fi, '<i8'), 'query_indices_content_sha256': value_sha(qi, '<i8'),
             'feature_state_content_sha256': state['state_content_sha256'],
             'geo_manifest_sha256': plan['input_sha256'][rel(old.GEO / stage / 'manifest.json')],
             'all_query_raw_and_TWD_bitwise_reload': True, 'reload_query_rows': len(Xq),
             'query_labels_passed_to_fit': False, 'reference_or_spatial_applied_in_model': False,
             'raw_z_diagnostics': {'negative_rows': int((z < 0).sum()), 'zero_rows': int((z == 0).sum()),
                                   'raw_values_sha256': value_sha(z), 'TWD_values_sha256': value_sha(pred)},
             'elapsed_seconds': time.monotonic() - started,
             'artifact_sha256': {rel(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    write(folder / 'metadata.json', final)
    return pred


def trial_checks(control, candidate):
    checks = {k + '_not_worse': candidate[k] <= control[k] for k in ('overall_mae', 'weighted_mae')}
    detail = {}
    for field, minimum in [('forecast_quarters', 1), ('strata', 500)]:
        need(set(control[field]) == set(candidate[field]), 'Metric group keys differ')
        rows = {}
        for name, old_value in control[field].items():
            new = candidate[field][name]
            need(old_value['rows'] == new['rows'], 'Metric group rows differ')
            if old_value['rows'] >= minimum:
                need(old_value['mae'] is not None and new['mae'] is not None, 'Nonempty group has no MAE')
                rows[name] = {'rows': new['rows'], 'control_mae': old_value['mae'], 'candidate_mae': new['mae'],
                              'at_most_5_percent_worse': new['mae'] <= old_value['mae'] * 1.05}
        detail[field] = rows
        checks[field + '_at_most_5_percent_worse'] = all(v['at_most_5_percent_worse'] for v in rows.values())
    return checks, detail


def evaluate(stage, plan, Xall, yall, cats):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths, spec = plan['stage_artifacts'][stage], plan['split_contract'][stage]
    fi, qi = [np.load(ROOT / paths[k + '_indices'], allow_pickle=False) for k in ('fit', 'query')]
    for label, index in [('fit', fi), ('query', qi)]:
        need(index.dtype == np.dtype('int64') and index.shape == (spec[label + '_rows'],)
             and (np.diff(index) > 0).all() and index.min() >= 0 and index.max() < len(Xall)
             and value_sha(index, '<i8') == spec[label + '_indices_content_sha256'], 'Canonical indices differ')
        save_array(folder / (label + '_indices.npy'), index)
    need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
    X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
    y, yq = np.asarray(yall[fi], dtype='float64'), np.asarray(yall[qi], dtype='float64')
    need(np.isfinite(y).all() and np.isfinite(yq).all() and (y > 0).all() and (yq > 0).all(), 'Invalid labels')
    binding = plan['bindings'][stage]
    original_geo = pd.read_pickle(ROOT / binding['original_geo'])
    need(original_geo.index.equals(Xall.index) and len(original_geo) == len(Xall), 'Canonical geo alignment differs')
    manifest = plan['prepared_geography'][stage]
    old_fit, old_query = original_geo.iloc[fi].reset_index(drop=True), original_geo.iloc[qi].reset_index(drop=True)
    geo, fit_added = old.load_geo(manifest, 'fit', old_fit)
    geoq, query_added = old.load_geo(manifest, 'query', old_query)
    summary = locked(ROOT / paths['state_summary'])
    with (ROOT / paths['spatial_state']).open('rb') as stream:
        state = pickle.load(stream)
    base_folder = c6.baseline_folder(stage)
    with (base_folder / 'reference_v2.pkl').open('rb') as stream:
        reference = pickle.load(stream)
    need(state['parameters'] == spatial.parameters() and state['state_summary'] == summary
         and summary['fit_X_sha256'] == spatial.frame_sha(X)
         and summary['fit_geo_sha256'] == spatial.frame_sha(geo)
         and summary['query_used_in_build'] is False
         and summary['existing_reference_residual_sha256'] == value_sha(reference['reference_residual'])
         and reference['fit_report']['fitting_rows'] == len(X), 'R39 spatial/reference fit state differs')
    score = lambda values: original.score(yq, values, X, Xq, plan['splits'][stage][0], plan['target_group_weights'])
    control = old.vector(ROOT / paths['control'], len(qi))
    # Recompose saved R39 branches, without any old tree or spatial replay.
    exact(control, .5 * old.vector(ROOT / paths['raw_spatial'], len(qi))
          + .5 * old.vector(ROOT / paths['log_spatial'], len(qi)), 'Saved R39 fixed-half control')
    control_metrics = score(control)
    historical = locked(ROOT / paths['result'])
    need(control_metrics == historical['metrics'][old.CANDIDATE], 'Saved full v7 control metrics differ')
    old_raw = old.vector(base_folder / 'catboost_raw.npy', len(qi))
    old_unit = old.vector(base_folder / 'lightgbm_l1_unit63.npy', len(qi))
    branch_values, branch_reports = {}, {}
    for branch in BRANCHES:  # Always both; no gate uses an intermediate score.
        component = fit_branch(stage, branch, plan, X, y, geo, Xq, geoq, cats, fi, qi)
        base = .75 * (.5 * old_raw + .5 * component) + .25 * old_unit
        p, diagnostics, ref_report = original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
        values, detail, report = spatial.predict(state, Xq, geoq, base, p, diagnostics)
        fallback = ~detail.applied.to_numpy(dtype=bool)
        exact(values[fallback], p[fallback], 'Unapplied ' + branch + ' spatial fallback')
        need(values.dtype == np.dtype('float64') and values.shape == (len(qi),)
             and np.isfinite(values).all() and (values >= 0).all(), 'Invalid spatial candidate branch')
        for name, a in [('base', base), ('reference', p), ('spatial', values)]:
            save_array(folder / (branch + '_' + name + '.npy'), a)
        with (folder / (branch + '_diagnostics.pkl')).open('xb') as stream:
            pickle.dump(detail, stream, protocol=5)
        branch_values[branch] = values
        branch_reports[branch] = {'reference': ref_report, 'spatial': report,
                                  'model_metadata_sha256': sha(folder / 'models' / branch / 'metadata.json')}
    candidate = .5 * branch_values['raw'] + .5 * branch_values['log']
    need(np.isfinite(candidate).all() and (candidate >= 0).all(), 'Invalid fixed mean')
    for name, a in [(CONTROL, control), (CANDIDATE, candidate)]:
        save_array(folder / (name + '.npy'), a)
        exact(a, old.vector(folder / (name + '.npy'), len(qi)), 'Saved ' + name)
    metrics = {CONTROL: control_metrics, CANDIDATE: score(candidate)}
    checks, check_details = trial_checks(metrics[CONTROL], metrics[CANDIDATE])
    result = {'round': 41, 'stage': stage, 'status': 'completed', 'plan_sha256': sha(OUT / 'plan.json'),
              'fit_rows': len(fi), 'query_rows': len(qi), 'dates': plan['splits'][stage], 'metrics': metrics,
              'candidate_vs_v7': old.prior.log_run.differences(metrics[CONTROL], metrics[CANDIDATE]),
              'trial_necessary_checks': checks, 'trial_check_details': check_details,
              'affected_groups': old.impact_groups(yq, control, candidate, old_query, query_added),
              'fit_positions_added': int(fit_added.sum()), 'query_positions_added': int(query_added.sum()),
              'new_tree_models_trained': 2, 'old_tree_models_loaded_or_replayed': False,
              'saved_v7_control_bitwise_recomposed': True, 'saved_v7_full_metrics_exact': True,
              'reference_trend_per_candidate_branch': 1, 'spatial_predict_per_candidate_branch': 1,
              'spatial_state_rebuilt': False, 'branch_reports': branch_reports,
              'query_labels_passed_to_training_reference_or_spatial': False,
              'source_sha256': plan['source_sha256'], 'runtime': plan['runtime'],
              'formal_adoption_decision': None, 'new_submission_generated': False,
              'elapsed_seconds': time.monotonic() - started,
              'artifact_sha256': {rel(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
    write(folder / 'results.json', result)
    print(json.dumps({'event': 'period_completed', 'stage': stage, 'changes': result['candidate_vs_v7'],
                      'checks': checks}, ensure_ascii=False), flush=True)
    return result


def run():
    need(not (OUT / 'run_claim.json').exists() and not (OUT / 'results.json').exists()
         and not any((OUT / stage).exists() for stage in STAGES), 'No automatic retry/overwrite')
    write(OUT / 'run_claim.json', {'created_at_utc': now(), 'plan_sha256': sha(OUT / 'plan.json'),
                                 'stage_order': list(STAGES), 'branch_order': list(BRANCHES), 'automatic_retry': False})
    started, phase = time.monotonic(), 'validate_plan_and_inputs'
    record = {'round': 41, 'status': 'incomplete', 'checked_stages': [], 'stage_result_sha256': {}, 'stages': {},
              'new_submission_generated': False, 'registry_updated': False, 'recommendation_changed': False,
              'formal_adoption_decision': None, 'automatic_release_allowed': False,
              'fresh_blind_validation': False, 'public_feedback_context': PUBLIC_CONTEXT}
    try:
        plan = read_plan()
        record.update(plan_sha256=sha(OUT / 'plan.json'), source_sha256=plan['source_sha256'],
                      protected_before=plan['protected_sha256'])
        c6.verify_hashes(merge(plan['input_sha256'], plan['source_sha256'], plan['protected_sha256']))
        Xall, yall = old.load_training_rows(plan)
        cats = c6.read(ROOT / 'reports/feature_schema.json')['categorical_columns']
        need(len(cats) == 21, 'Original category schema differs')
        for stage in STAGES:  # All windows are mandatory, independent of score direction.
            phase = stage
            value = evaluate(stage, plan, Xall, yall, cats)
            record['checked_stages'].append(stage)
            record['stage_result_sha256'][stage] = sha(OUT / stage / 'results.json')
            record['stages'][stage] = {k: value[k] for k in ('metrics', 'candidate_vs_v7', 'trial_necessary_checks', 'affected_groups')}
        need(plan['source_sha256'] == sources(), 'Source changed during run')
        # Large inputs were checked once before loads; do not repeat the full byte scan.
        record['protected_after'] = {name: sha(ROOT / name) for name in PROTECTED}
        need(record['protected_before'] == record['protected_after'], 'Protected old files changed')
        record.update(status='three_periods_six_models_completed_no_adoption_decision', new_tree_models_trained=6,
                      all_trial_necessary_conditions=all(all(s['trial_necessary_checks'].values()) for s in record['stages'].values()))
    except Exception as exc:
        record.update(status='incomplete_technical_failure', failure_phase=phase,
                      error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()},
                      completed_stage_models_only=sum(2 for _ in record['checked_stages']),
                      partial_fit_status='Inspect per-branch fit_started/fit_metadata/metadata; do not infer unrecorded fits or retry')
        record['protected_after'] = {name: sha(ROOT / name) for name in PROTECTED}
    record.update(completed_at_utc=now(), elapsed_seconds=time.monotonic() - started)
    write(OUT / 'results.json', record)
    print(json.dumps({'status': record['status'], 'checked_stages': record['checked_stages'],
                      'results_sha256': sha(OUT / 'results.json')}, ensure_ascii=False), flush=True)
    return 0 if record['status'] == 'three_periods_six_models_completed_no_adoption_decision' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', action='store_true')
    g.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        raise SystemExit(run())
