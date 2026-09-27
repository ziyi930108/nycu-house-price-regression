"""R42: one fixed masked-geographic raw model per period; prepare before run.

Mask only the tree's four identity columns. Original X remains the sole input
to the reference, spatial grouping and scorer. No adoption or CSV operation.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import pickle
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round39_city_road_experiment as previous
import round42_masked_geo as component

ROOT, OUT = previous.ROOT, previous.ROOT / 'outputs/round42'
DOCUMENT = ROOT / 'reports/round42_plan.md'
NOTE = ROOT / 'reports/round42_candidate_note.md'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, COMPARATOR = 'v7_control', 'v5_log_spatial'
CANDIDATE = 'masked_raw_log_spatial_half_mean'
HELPER_SHA = '3ca1da6e539abaeea4774c53b1dc34165e0f2bf29d0b42b23235e686123c8f7b'
MASK_SHA = '78509888884c658510108887a54a99356515ac381569662363db74b4315c56d0'
AUTHORITIES = {
    'outputs/round6/candidate_plan.json': '606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8',
    'outputs/round39/evaluation/plan.json': '110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13',
    'outputs/round39/evaluation/results.json': '1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27',
    'outputs/round36/results.json': '133d99e9fdca21ddf97e58fa8c7860ac74f7311bfe56f0005244c3a76791ef9e',
}
FORMULA = '.5*R8(saved_R39_state,new_query_geo,B_mask,p_mask,diag_mask)+.5*saved_R39_log_spatial'
BASE_FORMULA = '.75*(.5*old_raw_TWD+.5*new_masked_R6_geo_raw_TWD)+.25*old_unit_LGB_TWD'
need, sha, rel, locked, write = previous.need, previous.sha, previous.rel, previous.locked, previous.write
merge, now, vector, exact, spatial = previous.merge, previous.now, previous.vector, previous.exact, previous.spatial
geo_encoder = component.geography


def sources():
    return merge(previous.sources(), {rel(Path(__file__)): sha(Path(__file__)),
        'src/round42_masked_geo.py': sha(ROOT / 'src/round42_masked_geo.py'),
        'src/round5_project_mask.py': sha(ROOT / 'src/round5_project_mask.py')})


def protected_paths():
    names = ['outputs/submission_v2.csv', 'outputs/submission_v3.csv', 'runs/versions.json']
    names += [f'outputs/submission_v{v}_trial.csv' for v in range(4, 9)]
    return [name for name in names if (ROOT / name).exists()]


def freeze():
    need(sha(ROOT / 'src/round42_masked_geo.py') == HELPER_SHA
         and sha(ROOT / 'src/round5_project_mask.py') == MASK_SHA, 'Fixed helper/mask changed')
    inputs = {rel(DOCUMENT): sha(DOCUMENT), rel(NOTE): sha(NOTE)}
    def small(name, expected=None):
        return previous.prior.small(inputs, ROOT / name, expected)
    def bind(name, digest):
        inputs.update(merge(inputs, {name: digest}))
    docs = {n: small(n, digest) for n, digest in AUTHORITIES.items()}
    p6 = docs['outputs/round6/candidate_plan.json']
    p39 = docs['outputs/round39/evaluation/plan.json']
    r39 = docs['outputs/round39/evaluation/results.json']
    r36 = docs['outputs/round36/results.json']
    need(previous.sources() == p39['source_sha256']
         and r39['checked_stages'] == list(STAGES)
         and r39['status'] == 'three_periods_completed_no_adoption_decision', 'R39 authority differs')
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', previous.KT):
        bind(name, p39['input_sha256'][name])
    bind('runs/v1/models/final_config.json', p6['data_sha256']['runs/v1/models/final_config.json'])
    bindings = {}
    for stage in STAGES:
        rp = f'outputs/round39/evaluation/{stage}/results.json'
        r = small(rp, r39['stage_result_sha256'][stage])
        vp = f'outputs/round36/{stage}/results.json'
        v = small(vp, r36['stage_result_sha256'][stage])
        need(r['round'] == 39 and v['round'] == 36 and r['stage'] == v['stage'] == stage
             and r['status'] == v['status'] == 'completed'
             and r['fit_rows'] == v['fit_rows'] == p39['split_contract'][stage]['fit_rows']
             and r['query_rows'] == v['query_rows'] == p39['split_contract'][stage]['query_rows']
             and r['dates'] == v['dates'] == p39['splits'][stage], 'Saved window differs')
        manifest_path = f'outputs/round39/geography/{stage}/manifest.json'
        manifest = small(manifest_path, r['geo_manifest_sha256'])
        need(manifest == p39['prepared_geography'][stage], 'R39 prepared geography differs')
        original_manifest = small(f'outputs/round6/geography/{stage}/manifest.json', manifest['original_manifest_sha256'])
        train_geo = original_manifest['features']['train']
        need(train_geo['rows'] == 455255 and p39['input_sha256'][train_geo['path']] == train_geo['sha256'], 'Original annual geo differs')
        bind(train_geo['path'], train_geo['sha256'])
        spec = {'R39_result': rp, 'v5_result': vp, 'manifest': manifest,
                'original_geo': train_geo['path'], 'reference': r['reference_path']}
        bind(spec['reference'], r['reference_sha256'])
        for role, filename in (('fit_indices', 'fit_indices.npy'), ('query_indices', 'query_indices.npy'),
            ('old_raw_spatial', 'new_raw_spatial.npy'), ('log_spatial', 'new_log_spatial.npy'),
            ('v7', 'city_road_spatial_half_raw_half_log.npy'), ('state', 'spatial_state.pkl'), ('state_summary', 'state_summary.json')):
            name = f'outputs/round39/evaluation/{stage}/{filename}'
            spec[role] = name; bind(name, r['artifact_sha256'][name])
        receipt = rel((ROOT / spec['state_summary']).with_suffix('.sha256'))
        bind(receipt, r['artifact_sha256'][receipt])
        for role, filename in (('old_raw', 'catboost_raw.npy'), ('old_unit', 'lightgbm_l1_unit63.npy')):
            name = rel(previous.prior.c6.baseline_folder(stage) / filename)
            spec[role] = name; bind(name, p39['input_sha256'][name])
        spec['v5'] = f'outputs/round36/{stage}/log_spatial_combination.npy'
        bind(spec['v5'], v['artifact_sha256'][spec['v5']])
        spec['query_geo'] = manifest['features']['query']['path']
        bind(spec['query_geo'], manifest['features']['query']['sha256'])
        bindings[stage] = spec
    source_map = sources()
    previous.prior.c6.verify_hashes(merge(inputs, source_map))
    parameters = component.parameters()
    need(parameters['iterations'] == 2999 and parameters['depth'] == 8
         and parameters['learning_rate'] == .065 and parameters['random_seed'] == 42
         and parameters['boosting_type'] == 'Plain' and parameters['loss_function'] == 'MAE', 'Original R6 recipe differs')
    value = {'round': 42, 'created_at_utc': now(), 'candidate': CANDIDATE, 'control': CONTROL,
        'comparator': COMPARATOR, 'formula': FORMULA, 'base_formula': BASE_FORMULA,
        'source_sha256': source_map, 'input_sha256': inputs, 'bindings': bindings,
        'parameters': parameters, 'rules': component.RULES, 'mask_rules': component.masking.MASK_RULES,
        'stage_order': list(STAGES), 'splits': p39['splits'], 'split_contract': p39['split_contract'],
        'target_group_weights': p39['target_group_weights'], 'spatial_parameters': spatial.parameters(),
        'runtime': previous.average.runtime(), 'feature_count': 95, 'categorical_count': 23,
        'new_tree_fits_planned': 3, 'all_three_periods_regardless_of_scores': True,
        'new_acceptance_thresholds': None, 'formal_adoption_decision': None, 'automatic_release_allowed': False,
        'fresh_blind_validation': False, 'query_labels_passed_to_fit': False,
        'protected_sha256': {n: sha(ROOT / n) for n in protected_paths()}}
    need(source_map == sources() and all(sha(ROOT / n) == h for n, h in AUTHORITIES.items()), 'Authority changed during freeze')
    write(OUT / 'plan.json', value)
    print(json.dumps({'event': 'plan_frozen', 'sha256': sha(OUT / 'plan.json')}), flush=True)


def native(model, state, cats):
    names = state['all_feature_names']; category_names = list(cats) + geo_encoder.CATEGORICAL
    need(len(names) == 95 and len(category_names) == 23 and list(model.feature_names_) == names
         and model.get_cat_feature_indices() == sorted(names.index(c) for c in category_names), 'Native 95/23 order differs')
    p = model.get_all_params(); leaves = np.asarray(model.get_tree_leaf_counts())
    need(model.tree_count_ == 2999 and p['depth'] == 8 and p['grow_policy'] == 'SymmetricTree'
         and p['boosting_type'] == 'Plain' and p['random_seed'] == 42
         and p['loss_function'] == p['eval_metric'] == 'MAE'
         and np.isfinite(p['learning_rate']) and np.isclose(p['learning_rate'], .065, rtol=1e-7, atol=0)
         and leaves.shape == (2999,) and leaves.dtype.kind in 'iu' and (leaves > 0).all() and (leaves <= 256).all(), 'Native recipe differs')
    return p


def raw_predict(model, state, X, geo, cats):
    masked = component.transformed_query(state, X)
    geo_encoder.check_state(state['geo_state'], masked)
    frame = geo_encoder.augment_features(masked, geo)
    need(list(frame) == state['all_feature_names'], 'Masked model order differs')
    native(model, state, cats)
    z = np.asarray(model.predict(Pool(frame, cat_features=list(cats) + geo_encoder.CATEGORICAL, thread_count=4), thread_count=4), dtype='float64')
    need(z.shape == (len(X),) and np.isfinite(z).all(), 'Invalid raw prediction')
    prices = np.maximum(z * 1e6, 0)
    need(np.isfinite(prices).all(), 'Invalid inverse TWD prediction')
    return z, prices


def evaluate(stage, plan, Xall, yall):
    started = time.monotonic(); b = plan['bindings'][stage]; folder = OUT / stage
    folder.mkdir(exist_ok=False); model_folder = folder / 'model'; model_folder.mkdir()
    fi, qi = [np.load(ROOT / b[k + '_indices'], allow_pickle=False) for k in ('fit', 'query')]
    spec = plan['split_contract'][stage]
    for name, a in (('fit', fi), ('query', qi)):
        need(a.dtype == np.dtype('int64') and a.shape == (spec[name + '_rows'],)
             and (np.diff(a) > 0).all() and a.min() >= 0 and a.max() < len(Xall)
             and previous.prior.log_run.original.value_sha(a, '<i8') == spec[name + '_indices_content_sha256'], 'Canonical indices differ')
        previous.save_array(model_folder / (name + '_indices.npy'), a)
    need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
    X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
    y, yq = yall[fi], yall[qi]
    need(y.shape == (len(X),) and np.isfinite(y).all() and (y >= 0).all(), 'Invalid fit target')
    allgeo = pd.read_pickle(ROOT / b['original_geo']); geo_encoder.check_geo(allgeo)
    need(len(allgeo) == len(Xall) and allgeo.index.equals(Xall.index), 'Canonical original geo differs')
    geo, geoq = allgeo.iloc[fi].reset_index(drop=True), allgeo.iloc[qi].reset_index(drop=True)
    del allgeo
    newgeo = pd.read_pickle(ROOT / b['query_geo']); geo_encoder.check_geo(newgeo)
    need(newgeo.index.equals(Xq.index) and spatial.frame_sha(newgeo) == b['manifest']['features']['query']['frame_sha256'], 'New spatial query geo differs')
    before = {k: spatial.frame_sha(v) for k, v in {'fit_X': X, 'query_X': Xq, 'original_fit_geo': geo, 'original_query_geo': geoq}.items()}
    old_result, v5_result = locked(ROOT / b['R39_result']), locked(ROOT / b['v5_result'])
    need(all(old_result['input_frame_sha256'][k] == v for k, v in before.items()), 'Original X/geo identity differs')
    with (ROOT / b['state']).open('rb') as f: spatial_state = pickle.load(f)
    summary = locked(ROOT / b['state_summary'])
    with (ROOT / b['reference']).open('rb') as f: reference = pickle.load(f)
    need(spatial_state['state_summary'] == summary and spatial_state['parameters'] == plan['spatial_parameters']
         and summary['fit_X_sha256'] == before['fit_X'] and summary['query_used_in_build'] is False
         and summary['fit_geo_sha256'] == b['manifest']['features']['fit']['frame_sha256']
         and reference['fit_report']['fitting_rows'] == len(X)
         and summary['existing_reference_residual_sha256'] == previous.prior.log_run.original.value_sha(reference['reference_residual'], '<f8'), 'Fit-only spatial/reference identity differs')
    raw, log, control, v5 = [vector(ROOT / b[k], len(qi)) for k in ('old_raw_spatial', 'log_spatial', 'v7', 'v5')]
    exact(.5 * raw + .5 * log, control, 'v7 saved branch mean')
    score = lambda p: previous.prior.log_run.original.score(yq, p, X, Xq, plan['splits'][stage][0], plan['target_group_weights'])
    metrics = {CONTROL: score(control), COMPARATOR: score(v5)}
    need(metrics[CONTROL] == old_result['metrics'][previous.CANDIDATE]
         and metrics[COMPARATOR] == v5_result['metrics']['log_spatial_combination'], 'Saved full control metrics differ')
    cats = previous.prior.c6.read(ROOT / 'reports/feature_schema.json')['categorical_columns']
    target_sha = previous.prior.log_run.original.value_sha(y, '<f8')
    write(folder / 'fit_started.json', {'round': 42, 'stage': stage, 'started_at_utc': now(),
        'plan_sha256': sha(OUT / 'plan.json'), 'fit_rows': len(X), 'query_rows': len(Xq),
        'parameters': plan['parameters'], 'fit_target_sha256': target_sha, 'query_labels_used': False, 'automatic_retry': False})
    print(json.dumps({'event': 'fit_started', 'stage': stage, 'fit_rows': len(X), 'query_rows': len(Xq)}), flush=True)
    tick = time.monotonic(); model, state, metadata = component.fit_model(X, y, geo, cats)
    fit_seconds = time.monotonic() - tick
    model.save_model(str(model_folder / 'model.cbm')); write(model_folder / 'feature_state.json', state)
    fitted_native = native(model, state, cats)
    need(metadata['parameters'] == plan['parameters'] and metadata['feature_count'] == 95
         and metadata['categorical_feature_count'] == 23 and metadata['sample_weight'] == 'uniform'
         and metadata['fit_target_values_sha256'] == target_sha and metadata['mask_uses_query_or_target'] is False
         and state['original_fit_X_sha256'] == before['fit_X'] and state['mask_state']['fit_rows'] == len(X)
         and state['mask_state']['target_used'] is False and state['mask_state']['query_rows_used'] is False
         and not metadata['fit_uses_eval_set'] and not metadata['early_stopping_used'], 'Masked component contract differs')
    metadata.update(round=42, stage=stage, plan_sha256=sha(OUT / 'plan.json'), source_sha256=plan['source_sha256'],
        fitted_native_parameters=fitted_native, fit_elapsed_seconds=fit_seconds,
        original_frame_sha256=before, query_labels_passed_to_fit=False, prediction_units='TWD')
    write(model_folder / 'fit_metadata.json', metadata)
    z, prices = raw_predict(model, state, Xq, geoq, cats)
    previous.save_array(model_folder / 'raw_z.npy', z); previous.save_array(model_folder / 'predictions.npy', prices)
    del model
    restored = CatBoostRegressor(); restored.load_model(str(model_folder / 'model.cbm'))
    saved_state = locked(model_folder / 'feature_state.json'); need(saved_state == state, 'State JSON roundtrip differs')
    rz, rp = raw_predict(restored, saved_state, Xq, geoq, cats)
    exact(z, rz, 'Full raw saved model replay'); exact(prices, rp, 'Full TWD saved model replay')
    metadata.update(saved_native_parameters=native(restored, saved_state, cats),
        raw_z_diagnostics={'rows': len(z), 'negative_count': int((z < 0).sum()), 'zero_count': int((z == 0).sum()),
                           'min': float(z.min()), 'max': float(z.max()), 'units': 'million_TWD'},
        reload_rows=len(Xq), raw_TWD_reload_bitwise_equal=True,
        query_masked_rows=int(component.masking.query_mask(Xq, state['mask_state']).sum()),
        artifact_sha256={rel(p): sha(p) for p in sorted(model_folder.iterdir()) if p.is_file()})
    del restored
    need(previous.prior.log_run.original.value_sha(y, '<f8') == target_sha, 'Fit target mutated')
    write(model_folder / 'metadata.json', metadata)
    old, unit = vector(ROOT / b['old_raw'], len(qi)), vector(ROOT / b['old_unit'], len(qi))
    base = .75 * (.5 * old + .5 * prices) + .25 * unit
    p, diag, reference_report = previous.prior.log_run.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
    branch, detail, spatial_report = spatial.predict(spatial_state, Xq, newgeo, base, p, diag)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    exact(branch[fallback], p[fallback], 'Masked branch spatial fallback')
    candidate = .5 * branch + .5 * log
    need(np.isfinite(candidate).all() and (candidate >= 0).all(), 'Invalid candidate TWD')
    metrics[CANDIDATE] = score(candidate)
    for name, values in ((CONTROL, control), (COMPARATOR, v5), ('base_masked', base), ('reference_masked', p),
                         ('masked_raw_spatial', branch), (CANDIDATE, candidate)):
        previous.save_array(folder / (name + '.npy'), values)
    for name, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
        with (folder / (name + '.pkl')).open('xb') as f: pickle.dump(value, f, protocol=5)
    need(before == {k: spatial.frame_sha(v) for k, v in {'fit_X': X, 'query_X': Xq, 'original_fit_geo': geo, 'original_query_geo': geoq}.items()}
         and spatial.frame_sha(newgeo) == b['manifest']['features']['query']['frame_sha256'], 'Original/new frames mutated')
    result = {'round': 42, 'stage': stage, 'status': 'completed', 'created_at_utc': now(),
        'plan_sha256': sha(OUT / 'plan.json'), 'fit_rows': len(fi), 'query_rows': len(qi), 'dates': plan['splits'][stage],
        'metrics': metrics, 'candidate_vs_v7': previous.prior.log_run.differences(metrics[CONTROL], metrics[CANDIDATE]),
        'candidate_vs_v5': previous.prior.log_run.differences(metrics[COMPARATOR], metrics[CANDIDATE]),
        'v7_control_bitwise_equal': True, 'v7_and_v5_all_metrics_exact': True,
        'mask_summary': metadata['mask_summary'], 'query_masked_rows': metadata['query_masked_rows'],
        'original_X_used_for_reference_spatial_and_score': True, 'original_frames_unchanged': True,
        'original_geo_used_for_tree': True, 'new_geo_used_only_for_spatial': True,
        'new_tree_fits': 1, 'reference_predict_calls': 1, 'new_spatial_predict_calls': 1, 'new_spatial_state_builds': 0,
        'query_labels_used_for_fit_or_predict': False, 'full_query_raw_TWD_reload_bitwise_equal': True,
        'reference_report': reference_report, 'spatial_report': spatial_report,
        'model_metadata_sha256': sha(model_folder / 'metadata.json'), 'fit_elapsed_seconds': fit_seconds,
        'elapsed_seconds': time.monotonic() - started, 'formal_adoption_decision': None,
        'new_acceptance_thresholds': None, 'new_submission_generated': False, 'fresh_blind_validation': False,
        'artifact_sha256': {rel(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
    write(folder / 'results.json', result)
    print(json.dumps({'event': 'stage_complete', 'stage': stage, 'vs_v7': result['candidate_vs_v7'], 'vs_v5': result['candidate_vs_v5']}), flush=True)
    return result


def run():
    p = locked(OUT / 'plan.json')
    need(p['round'] == 42 and p['stage_order'] == list(STAGES) and p['all_three_periods_regardless_of_scores'] is True
         and p['parameters'] == component.parameters() and p['rules'] == component.RULES
         and p['mask_rules'] == component.masking.MASK_RULES and p['source_sha256'] == sources()
         and p['runtime'] == previous.average.runtime() and p['spatial_parameters'] == spatial.parameters()
         and p['formula'] == FORMULA and p['base_formula'] == BASE_FORMULA
         and p['new_acceptance_thresholds'] is None and p['automatic_release_allowed'] is False, 'Frozen method changed')
    need(not (OUT / 'run_claim.json').exists() and not (OUT / 'results.json').exists()
         and not any((OUT / s).exists() for s in STAGES), 'Prior attempt exists; preserve it')
    write(OUT / 'run_claim.json', {'started_at_utc': now(), 'plan_sha256': sha(OUT / 'plan.json'), 'automatic_retry': False})
    result = {'round': 42, 'status': 'incomplete', 'plan_sha256': sha(OUT / 'plan.json'), 'stage_order': list(STAGES),
        'checked_stages': [], 'stages': {}, 'stage_result_sha256': {}, 'source_sha256': p['source_sha256'],
        'input_sha256': p['input_sha256'], 'protected_before': p['protected_sha256'],
        'formal_adoption_decision': None, 'new_acceptance_thresholds': None, 'new_submission_generated': False,
        'registry_updated': False, 'fresh_blind_validation': False,
        'label_loading': 'shared loader deserializes canonical train targets; fit y only to tree, query y only to scorer'}
    start = time.monotonic(); phase = 'verify_inputs_before_load'
    try:
        previous.prior.c6.verify_hashes(merge(p['input_sha256'], p['source_sha256'], p['protected_sha256']))
        Xall, yall = previous.load_training_rows(p)
        for stage in STAGES:  # No branch depends on earlier scores.
            phase = stage; item = evaluate(stage, p, Xall, yall)
            result['checked_stages'].append(stage); result['stages'][stage] = item
            result['stage_result_sha256'][stage] = sha(OUT / stage / 'results.json')
        previous.prior.c6.verify_hashes(merge(p['source_sha256'], p['protected_sha256']))
        result['status'] = 'three_periods_completed_no_adoption_decision'
    except Exception:
        result.update(status='incomplete_technical_failure', failure_phase=phase, traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    result.update(elapsed_seconds=time.monotonic() - start, completed_at_utc=now(),
        fit_started_stages=[s for s in STAGES if (OUT / s / 'fit_started.json').exists()],
        model_saved_stages=[s for s in STAGES if (OUT / s / 'model/model.cbm').exists()],
        protected_after={n: sha(ROOT / n) for n in p['protected_sha256']})
    write(OUT / 'results.json', result)
    return 0 if result['status'] == 'three_periods_completed_no_adoption_decision' else 1


def execute(mode):
    if mode == 'freeze':
        need(not OUT.exists(), 'R42 output already exists'); OUT.mkdir(parents=True, exist_ok=False)
    else: need(OUT.is_dir(), 'Freeze is required')
    log = OUT / (mode + '.log'); started = now(); tick = time.monotonic(); code = 1; failure = None
    with log.open('x', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                if mode == 'freeze': freeze(); code = 0
                else: code = run()
            except Exception:
                failure = traceback.format_exc(); print(failure, flush=True)
    record = {'round': 42, 'command': sys.argv, 'started_at_utc': started, 'completed_at_utc': now(),
        'elapsed_seconds': time.monotonic() - tick, 'exit_code': code, 'failure': failure,
        'source_sha256': sha(Path(__file__)), 'stdout_log': rel(log), 'stdout_log_sha256': sha(log), 'automatic_retry': False}
    write(OUT / (mode + '_execution.json'), record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=['freeze', 'run'])
    raise SystemExit(execute(parser.parse_args().command))
