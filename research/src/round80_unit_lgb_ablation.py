"""R80: remove old-unit LightGBM from the saved R79 unit base; no fitting."""
from pathlib import Path
import argparse
import json
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd
import round40_three_branch as unit_run
import round48_nn_experiment as old

ROOT, STAGES = old.ROOT, old.STAGES
OUT = ROOT / 'outputs/round80'
DOCUMENT = ROOT / 'reports/round80_plan.md'
PLAN_SHA = '92020b607a34d5ded10da23858219b664d24f4f2703b68b258e44fe7ba728cfe'
R79_RESULT_SHA = '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6'
R79_EXECUTION_SHA = '8b0132b6ad0b374488f02957661fdc1b46ac69c381c0e71717498c02e18f986d'
sha, read, write, spatial, rel = old.sha, old.locked, old.write, old.spatial, old.rel
BACKGROUND = ('v20', 'v21', 'v19', 'v18', 'v11', 'v12', 'v5')
CONTROL_BASE = '.75*saved_R79_native_TWD+.25*saved_old_unit_LGB_TWD'
CANDIDATE_BASE = 'saved_R79_native_TWD'
FORMULA = 'median(saved_R71_seed42_nn_spatial,saved_R46_recency_log_spatial,spatial(reference(saved_R79_native_TWD)))'


def exact(a, b, label):
    # Also accepts empty fallback selections and int64 indices.
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def run(expected_plan_sha):
    tick = time.monotonic()
    assert expected_plan_sha == PLAN_SHA == sha(DOCUMENT)
    inputs = {}

    def checked(name, digest):
        name = name.replace('\\', '/')
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs:
            assert sha(path) == digest, name
        inputs[name] = digest
        return path

    def small(name, digest):
        path = checked(name, digest)
        value = read(path)
        receipt = path.with_suffix('.sha256')
        checked(rel(receipt), sha(receipt))
        return value

    def artifact(record, name):
        mapping = {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}
        return checked(name, mapping[name])

    checked(rel(DOCUMENT), PLAN_SHA)
    r79 = small('outputs/round79/results.json', R79_RESULT_SHA)
    p79 = small('outputs/round79/plan.json', r79['plan_sha256'])
    execution = small('outputs/round79/execution.json', R79_EXECUTION_SHA)
    assert execution['exit_code'] == 0 and execution['failure'] is None
    assert execution['source_sha256'] == r79['source_sha256']['src/round79_experiment.py']
    assert r79['round'] == p79['round'] == 79 and r79['candidate'] == 'R79'
    assert r79['status'] == 'three_periods_completed_no_adoption_decision'
    assert r79['checked_stages'] == p79['stage_order'] == list(STAGES)
    assert r79['new_fits'] == 3 and r79['new_NN_fits'] == 0
    assert r79['source_sha256'] == p79['source_sha256']
    assert r79['input_sha256'] == p79['input_sha256']
    p48 = small('outputs/round48/plan.json', p79['input_sha256']['outputs/round48/plan.json'])
    p40 = small('outputs/round40/plan.json', p79['input_sha256']['outputs/round40/plan.json'])
    assert p79['split_contract'] == p48['split_contract'] == p40['split_contract']
    assert p79['splits'] == p48['splits'] == p40['splits']
    assert p79['target_group_weights'] == p48['target_group_weights'] == p40['target_group_weights']
    assert p40['spatial_parameters'] == spatial.parameters()
    source = old.previous.merge(r79['source_sha256'], {rel(Path(__file__)): sha(Path(__file__))})
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    selector_path = ROOT / 'outputs/current_private_candidate.json'
    selector_sha = sha(selector_path)
    selector = json.loads(selector_path.read_text(encoding='utf8'))
    assert selector['private_mae'] is None and not selector['final_version_chosen']
    assert sha(ROOT / selector['private_priority']) == selector['sha256']
    # The initial selector may already be v22. Its exact snapshot must stay fixed.
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT):
        assert p79['input_sha256'][name] == p48['input_sha256'][name]
        checked(name, p79['input_sha256'][name])
    bindings = {}
    for s in STAGES:
        a, inherited = r79['stages'][s], p79['bindings'][s]
        b = inherited['unit']
        spec = p79['split_contract'][s]
        assert a['status'] == 'completed' and a['new_fits'] == 1 and a['new_NN_fits'] == 0
        assert a['full_query_raw_TWD_reload_exact'] and a['query_labels_used_only_for_evaluation']
        for key in ('fit_indices', 'query_indices', 'old_unit', 'reference', 'state', 'state_summary', 'query_geo'):
            checked(b[key], p79['input_sha256'][b[key]])
        receipt = rel((ROOT / b['state_summary']).with_suffix('.sha256'))
        checked(receipt, p79['input_sha256'][receipt])
        paths = {key: inherited['paths'][key] for key in ('nn', 'log', *BACKGROUND)}
        for path in paths.values():
            checked(path, p79['input_sha256'][path])
        for key, suffix in (('native', 'model/native_predictions.npy'), ('v22', 'predictions.npy'),
                            ('base', 'base.npy'), ('reference', 'reference.npy'), ('spatial', 'unit_spatial.npy'),
                            ('fit', 'fit_indices.npy'), ('query', 'query_indices.npy')):
            paths[key] = f'outputs/round79/{s}/{suffix}'
            artifact(a, paths[key])
        assert inputs[paths['v22']] == a['prediction_sha256']
        for kind in ('fit', 'query'):
            assert inputs[paths[kind]] == inputs[b[kind + '_indices']]
            assert a[kind + '_rows'] == spec[kind + '_rows']
        mp = f'outputs/round79/{s}/model/metadata.json'
        artifact(a, mp)
        meta = small(mp, inputs[mp])
        assert meta['prediction_units'] == 'TWD' and meta['query_rows'] == meta['reload_rows'] == a['query_rows']
        assert meta['fit_rows'] == a['fit_rows'] and meta['plan_sha256'] == r79['plan_sha256']
        assert meta['reload_all_query_rows_verified'] and meta['reload_raw_exact'] and meta['reload_TWD_exact']
        assert meta['artifact_sha256'][paths['native']] == inputs[paths['native']]
        assert meta['original_frame_sha256'] == a['original_frame_sha256']
        for kind in ('fit', 'query'):
            assert meta[kind + '_indices_content_sha256'] == spec[kind + '_indices_content_sha256']
        bindings[s] = {'unit': b, 'paths': paths, 'metadata_path': mp}
    plan = {'round': 80, 'candidate': 'R80', 'primary_control': 'v22=R79',
        'text_plan_sha256': PLAN_SHA, 'source_sha256': source, 'input_sha256': dict(inputs),
        'control_base_formula': CONTROL_BASE, 'candidate_base_formula': CANDIDATE_BASE, 'formula': FORMULA,
        'bindings': bindings, 'stage_order': list(STAGES), 'split_contract': p79['split_contract'],
        'splits': p79['splits'], 'target_group_weights': p79['target_group_weights'],
        'protected_before': protected, 'selector_sha256_before': selector_sha,
        'new_fits': 0, 'new_NN_fits': 0, 'tree_NN_model_loads': 0, 'model_predict_calls': 0,
        'new_reference_fits': 0, 'new_spatial_state_builds': 0,
        'reference_predict_calls_per_stage': 2, 'spatial_predict_calls_per_stage': 2,
        'all_periods_regardless_of_scores': True, 'seed_weight_threshold_search': False,
        'query_labels_used_only_for_evaluation': True, 'Public_feedback_used_for_tuning': False,
        'new_CSV': False, 'new_test_predictions': False, 'automatic_adoption': False, 'fresh_blind_validation': False}
    write(OUT / 'plan.json', plan)
    Xall, yall = old.previous.load_training_rows(p48)
    result = {k: plan[k] for k in ('round', 'candidate', 'formula', 'control_base_formula', 'candidate_base_formula',
        'text_plan_sha256', 'source_sha256', 'input_sha256', 'protected_before', 'selector_sha256_before',
        'new_fits', 'new_NN_fits', 'tree_NN_model_loads', 'model_predict_calls', 'new_reference_fits',
        'new_spatial_state_builds', 'query_labels_used_only_for_evaluation', 'Public_feedback_used_for_tuning',
        'new_CSV', 'new_test_predictions', 'automatic_adoption', 'fresh_blind_validation', 'seed_weight_threshold_search')}
    result.update(status='running', primary_control='v22', checked_stages=[], stages={},
                  plan_sha256=sha(OUT / 'plan.json'), new_submission_generated=False)
    for s in STAGES:
        start = time.monotonic()
        folder = OUT / s
        folder.mkdir()
        a, b, paths = r79['stages'][s], bindings[s]['unit'], bindings[s]['paths']
        spec, indices = p79['split_contract'][s], {}
        for kind in ('fit', 'query'):
            v = np.load(ROOT / paths[kind], allow_pickle=False)
            assert v.dtype == np.int64 and v.shape == (spec[kind + '_rows'],)
            assert (np.diff(v) > 0).all() and v.min() >= 0 and v.max() < len(Xall)
            assert old.original.value_sha(v, '<i8') == spec[kind + '_indices_content_sha256']
            exact(v, np.load(ROOT / b[kind + '_indices'], allow_pickle=False), 'R79 canonical ' + kind)
            indices[kind] = v
            old.previous.save_array(folder / (kind + '_indices.npy'), v)
        fi, qi = indices['fit'], indices['query']
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        n = len(qi)
        frames = {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert frames == a['original_frame_sha256']
        geo = pd.read_pickle(ROOT / b['query_geo'])
        unit_run.previous.geo_encoder.check_geo(geo)
        assert geo.index.equals(Xq.index) and spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        with (ROOT / b['state']).open('rb') as f:
            state = pickle.load(f)
        summary = read(ROOT / b['state_summary'])
        assert state['state_summary'] == summary and state['parameters'] == p40['spatial_parameters']
        assert summary['fit_X_sha256'] == frames['fit_X'] and not summary['query_used_in_build']
        assert summary['fit_geo_sha256'] == b['manifest']['features']['fit']['frame_sha256']
        with (ROOT / b['reference']).open('rb') as f:
            reference = pickle.load(f)
        assert reference['fit_report']['fitting_rows'] == len(fi)
        assert old.original.value_sha(reference['reference_residual'], '<f8') == summary['existing_reference_residual_sha256']
        vector = lambda name: unit_run.vector(ROOT / name, n)  # Already TWD; zero is valid.
        fixed = {key: vector(paths[key]) for key in ('native', 'nn', 'log', 'v22', *BACKGROUND)}
        native, unit = fixed['native'], vector(b['old_unit'])
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p79['splits'][s][0], p79['target_group_weights'])
        metrics = {key: score(fixed[key]) for key in ('v22', *BACKGROUND)}
        assert metrics['v22'] == a['metrics']['R79']
        assert all(metrics[key] == a['metrics'][key] for key in BACKGROUND)
        native_metrics = score(native)
        assert native_metrics == a['unit_branch_metrics']['native']
        branch_values, branch_metrics, reports = {}, {}, {}
        for mode in ('control', 'candidate'):
            base = .75 * native + .25 * unit if mode == 'control' else native.copy()
            if mode == 'control':
                exact(base, vector(paths['base']), 'R79 base exact')
            p, diag, rr = old.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
            branch, detail, sr = spatial.predict(state, Xq, geo, base, p, diag)
            fallback = ~detail['applied'].to_numpy(dtype=bool)
            exact(branch[fallback], p[fallback], mode + ' spatial fallback')
            branch_metrics[mode] = {'base': score(base), 'reference': score(p), 'spatial': score(branch)}
            if mode == 'control':
                exact(p, vector(paths['reference']), 'R79 reference exact')
                exact(branch, vector(paths['spatial']), 'R79 spatial exact')
                exact(np.median(np.column_stack([fixed['nn'], fixed['log'], branch]), axis=1), fixed['v22'], 'R79 median exact')
                assert all(branch_metrics[mode][layer] == a['unit_branch_metrics'][layer] for layer in ('base', 'reference', 'spatial'))
            else:
                exact(base, native, 'Candidate base is native TWD without rescaling')
            branch_values[mode] = branch
            reports[mode] = {'reference': rr, 'spatial': sr}
            for label, value in (('base', base), ('reference', p), ('unit_spatial', branch)):
                old.previous.save_array(folder / (mode + '_' + label + '.npy'), value)
            if mode == 'candidate':
                for label, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
                    with (folder / (label + '.pkl')).open('xb') as f:
                        pickle.dump(value, f, protocol=5)
        candidate = np.median(np.column_stack([fixed['nn'], fixed['log'], branch_values['candidate']]), axis=1)
        assert candidate.dtype == np.float64 and candidate.shape == (n,) and np.isfinite(candidate).all() and (candidate > 0).all()
        metrics['R80'] = score(candidate)
        old.previous.save_array(folder / 'predictions.npy', candidate)
        exact(candidate, vector(rel(folder / 'predictions.npy')), 'Saved R80 vector exact')
        assert frames == {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        diff = old.previous.prior.log_run.differences
        item = {'stage': s, 'status': 'completed', 'fit_rows': len(fi), 'query_rows': n,
            'dates': p79['splits'][s], 'original_frame_sha256': frames, 'metrics': metrics,
            'differences': {label: diff(metrics[label], metrics['R80']) for label in ('v22', *BACKGROUND)},
            'native_metrics': native_metrics, 'unit_branch_metrics': branch_metrics,
            'unit_branch_differences': {layer: diff(branch_metrics['control'][layer], branch_metrics['candidate'][layer]) for layer in ('base', 'reference', 'spatial')},
            'pipeline_reports': reports, 'R79_base_reference_spatial_full_vectors_exact': True,
            'R79_unit_branch_full_metrics_exact': True, 'v22_complete_combination_exact': True,
            'controls_full_metrics_exact': True, 'saved_R79_native_TWD_used_without_area_conversion': True,
            'inherited_R79_model_reload_evidence_verified': True, 'independent_model_reload_performed': False,
            'changed_unit_rows': int(np.count_nonzero(branch_values['candidate'] != branch_values['control'])),
            'changed_final_rows': int(np.count_nonzero(candidate != fixed['v22'])),
            'new_fits': 0, 'new_NN_fits': 0, 'tree_NN_model_loads': 0, 'model_predict_calls': 0,
            'reference_predict_calls': 2, 'spatial_predict_calls': 2, 'query_labels_used_only_for_evaluation': True,
            'prediction_sha256': sha(folder / 'predictions.npy'), 'elapsed_seconds': time.monotonic() - start,
            'artifact_sha256': {rel(p): sha(p) for p in folder.iterdir() if p.is_file()}}
        write(folder / 'results.json', item)
        result['stages'][s] = item
        result['checked_stages'].append(s)
        print(json.dumps({'stage': s, 'R80_vs_v22': item['differences']['v22']['overall'],
            'unit_spatial_vs_R79': item['unit_branch_differences']['spatial']['overall'],
            'changed_final_rows': item['changed_final_rows']}), flush=True)
    assert result['checked_stages'] == list(STAGES)
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items())
    after = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    after['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    assert after == protected and sha(selector_path) == selector_sha and sha(DOCUMENT) == PLAN_SHA
    result.update(status='three_periods_completed_no_adoption_decision', elapsed_seconds=time.monotonic() - tick,
                  reference_predict_calls=6, spatial_predict_calls=6, protected_after=after,
                  selector_sha256_after=sha(selector_path))
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'result_sha256': sha(OUT / 'results.json'),
        'elapsed_seconds': result['elapsed_seconds'], 'new_fits': 0, 'new_CSV': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert args.plan_sha == PLAN_SHA == sha(DOCUMENT) and not OUT.exists(), 'Plan changed or previous attempt exists'
    OUT.mkdir()
    tick, started, code, failure = time.monotonic(), old.now(), 0, None
    write(OUT / 'started.json', {'command': sys.argv, 'started_at_utc': started,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        run(args.plan_sha)
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': code, 'failure': failure,
        'started_at_utc': started, 'completed_at_utc': old.now(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
