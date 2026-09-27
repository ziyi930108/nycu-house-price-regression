"""R76: remove oldraw only from the saved R40 unit branch; no model fitting."""
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

ROOT = old.ROOT
OUT = ROOT / 'outputs/round76'
DOCUMENT = ROOT / 'reports/round76_plan.md'
STAGES = ('development', 'confirmation', 'auxiliary')
sha, read, write, spatial = old.sha, old.locked, old.write, old.spatial
rel = lambda p: Path(p).relative_to(ROOT).as_posix()
OLD_BASE = '.75*(.5*old_raw_TWD+.5*R19_unit_native_TWD)+.25*old_unit_LGB_TWD'
NEW_BASE = '.75*R19_unit_native_TWD+.25*old_unit_LGB_TWD'
FORMULA = 'median(saved_R71_seed42_nn_spatial,saved_R46_recency_log_spatial,spatial(reference(' + NEW_BASE + ')))'


def run(expected_plan_sha):
    tick = time.monotonic()
    inputs = {}

    def checked(name, digest):
        name = name.replace('\\', '/')
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs:
            assert sha(ROOT / name) == digest, name
        inputs[name] = digest
        return ROOT / name

    def small(name):
        path = ROOT / name
        value = read(path)
        checked(name, sha(path))
        receipt = path.with_suffix('.sha256')
        checked(rel(receipt), sha(receipt))
        return value

    def artifact(record, name):
        mapping = {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}
        return checked(name, mapping[name])

    def vector(path, n):
        # Original tree controls can contain zero. Native R19 is already TWD.
        return unit_run.vector(path, n)

    checked(rel(DOCUMENT), expected_plan_sha)
    p40, p48, p52, p71 = (small(f'outputs/round{r}/plan.json') for r in (40, 48, 52, 71))
    r40, r71 = (small(f'outputs/round{r}/results.json') for r in (40, 71))
    assert r40['checked_stages'] == r71['checked_stages'] == list(STAGES)
    assert r40['status'] == r71['status'] == 'three_periods_completed_no_adoption_decision'
    assert r71['primary_seed'] == 42
    assert p40['split_contract'] == p48['split_contract'] == p71['split_contract']
    assert p40['splits'] == p48['splits']
    assert p40['target_group_weights'] == p48['target_group_weights']
    assert p40['spatial_parameters'] == p48['spatial_parameters'] == spatial.parameters()
    assert unit_run.sources() == p40['source_sha256'], 'R40 source closure changed'
    source = old.previous.merge(p40['source_sha256'], r71['source_sha256'],
        {rel(Path(__file__)): sha(Path(__file__)), rel(Path(old.__file__)): sha(Path(old.__file__))})
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    selector_path = ROOT / 'outputs/current_private_candidate.json'
    selector_sha = sha(selector_path)
    selector = json.loads(selector_path.read_text(encoding='utf8'))
    assert selector['private_priority'] == 'outputs/submission_v19_trial.csv'
    assert selector['private_mae'] is None and not selector['final_version_chosen']
    assert sha(ROOT / selector['private_priority']) == selector['sha256']

    # Bind all used files before evaluating the first period.
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT):
        assert p40['input_sha256'][name] == p48['input_sha256'][name]
        checked(name, p40['input_sha256'][name])
    bindings = {}
    for s in STAGES:
        b, a = p40['bindings'][s], r71['stages'][s]
        for key in ('fit_indices', 'query_indices', 'unit_component', 'unit_original_blend', 'old_raw', 'old_unit',
                    'reference', 'state', 'state_summary', 'query_geo'):
            checked(b[key], p40['input_sha256'][b[key]])
        summary_receipt = rel((ROOT / b['state_summary']).with_suffix('.sha256'))
        checked(summary_receipt, p40['input_sha256'][summary_receipt])
        e19 = small(b['R19_result'])
        assert sha(ROOT / b['R19_result']) == p40['input_sha256'][b['R19_result']]
        paths = {'nn': f'outputs/round71/{s}/seed42/nn_spatial.npy',
                 'control': f'outputs/round71/{s}/predictions.npy',
                 'unit_control': f'outputs/round40/{s}/unit_spatial.npy',
                 'log': p52['bindings'][s]['recency_log'],
                 'v18': f'outputs/round66/{s}/predictions.npy'}
        artifact(a, paths['nn']); artifact(a, paths['control'])
        artifact(r40['stages'][s], paths['unit_control'])
        assert a['prediction_sha256'] == inputs[paths['control']]
        assert p48['bindings'][s]['unit_spatial'] == paths['unit_control']
        assert p48['input_sha256'][paths['unit_control']] == inputs[paths['unit_control']]
        checked(paths['log'], p52['input_sha256'][paths['log']])
        assert r71['input_sha256'][paths['log']] == inputs[paths['log']]
        checked(paths['v18'], r71['input_sha256'][paths['v18']])
        for k in ('fit', 'query'):
            path = f'outputs/round71/{s}/{k}_indices.npy'
            artifact(a, path)
            assert inputs[path] == p40['input_sha256'][b[k + '_indices']]
            paths[k + '_R71_indices'] = path
        bindings[s] = {'unit': b, 'fixed': paths, 'R19_metrics': e19['metrics']}
    plan = {'round': 76, 'candidate': 'R76', 'primary_control': 'v19=R71_seed42',
        'old_unit_base_formula': OLD_BASE, 'new_unit_base_formula': NEW_BASE, 'formula': FORMULA,
        'stage_order': list(STAGES), 'split_contract': p40['split_contract'],
        'splits': p40['splits'], 'target_group_weights': p40['target_group_weights'],
        'source_sha256': source, 'input_sha256': dict(inputs), 'text_plan_sha256': expected_plan_sha,
        'bindings': bindings, 'protected_before': protected, 'selector_sha256_before': selector_sha,
        'new_fits': 0, 'tree_NN_model_loads': 0, 'model_predict_calls': 0,
        'new_reference_fits': 0, 'new_spatial_state_builds': 0,
        'reference_predict_calls_per_stage': 2, 'spatial_predict_calls_per_stage': 2,
        'no_R74_change_applied': True, 'seed_weight_threshold_search': False,
        'Public_feedback_used_for_tuning': False, 'all_periods_regardless_of_scores': True,
        'query_labels_used_only_for_evaluation': True, 'automatic_release_allowed': False,
        'new_CSV': False, 'selector_update_allowed': False, 'fresh_blind_validation': False}
    write(OUT / 'plan.json', plan)
    Xall, yall = old.previous.load_training_rows(p40)
    result = {'round': 76, 'candidate': 'R76', 'status': 'running', 'formula': FORMULA,
        'old_unit_base_formula': OLD_BASE, 'new_unit_base_formula': NEW_BASE,
        'primary_control': 'v19', 'checked_stages': [], 'stages': {}, 'source_sha256': source,
        'input_sha256': inputs, 'plan_sha256': sha(OUT / 'plan.json'), 'text_plan_sha256': expected_plan_sha,
        'protected_before': protected, 'selector_sha256_before': selector_sha,
        'new_fits': 0, 'new_NN_fits': 0, 'tree_NN_model_loads': 0, 'model_predict_calls': 0,
        'new_reference_fits': 0, 'new_spatial_state_builds': 0,
        'new_submission_generated': False, 'new_CSV': False, 'fresh_blind_validation': False,
        'seed_weight_threshold_search': False, 'Public_feedback_used_for_tuning': False,
        'query_labels_used_only_for_evaluation': True}
    for s in STAGES:
        start = time.monotonic()
        folder = OUT / s
        folder.mkdir()
        b, fixed = bindings[s]['unit'], bindings[s]['fixed']
        spec = p40['split_contract'][s]
        indices = {}
        for kind in ('fit', 'query'):
            ind = np.load(ROOT / b[kind + '_indices'], allow_pickle=False)
            assert ind.dtype == np.int64 and ind.shape == (spec[kind + '_rows'],)
            assert (np.diff(ind) > 0).all() and ind.min() >= 0 and ind.max() < len(Xall)
            assert old.original.value_sha(ind, '<i8') == spec[kind + '_indices_content_sha256']
            other = np.load(ROOT / fixed[kind + '_R71_indices'], allow_pickle=False)
            assert other.dtype == ind.dtype == np.int64 and other.shape == ind.shape
            assert ind.tobytes() == other.tobytes(), 'R40/R71 index bytes differ'
            indices[kind] = ind
            old.previous.save_array(folder / (kind + '_indices.npy'), ind)
        fi, qi = indices['fit'], indices['query']
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        n = len(qi)
        frames = {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert frames == r40['stages'][s]['input_frame_sha256'] == r71['stages'][s]['original_frame_sha256']
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
        raw, unit, native = [vector(ROOT / b[k], n) for k in ('old_raw', 'old_unit', 'unit_component')]
        nn, log, control, unit_control, v18 = [vector(ROOT / fixed[k], n) for k in ('nn', 'log', 'control', 'unit_control', 'v18')]
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p40['splits'][s][0], p40['target_group_weights'])
        before = .75 * (.5 * raw + .5 * native) + .25 * unit
        branch_values, branch_metrics, reports = {}, {}, {}
        for mode in ('control', 'candidate'):
            base = before if mode == 'control' else .75 * native + .25 * unit
            p, diag, rr = old.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
            if mode == 'control':
                old.exact(p, vector(ROOT / b['unit_original_blend'], n), 'R19 complete reference replay')
            branch, detail, sr = spatial.predict(state, Xq, geo, base, p, diag)
            fallback = ~detail['applied'].to_numpy(dtype=bool)
            old.exact(branch[fallback], p[fallback], mode + ' spatial fallback')
            if mode == 'control':
                old.exact(branch, unit_control, 'Original R40 unit branch replay')
            branch_values[mode] = branch
            branch_metrics[mode] = {'base': score(base), 'reference': score(p), 'spatial': score(branch)}
            if mode == 'control':
                assert branch_metrics[mode]['reference'] == bindings[s]['R19_metrics']['unit_catboost_geography_blend']
                assert branch_metrics[mode]['spatial'] == r40['stages'][s]['metrics']['unit_spatial']
                old.exact(np.median(np.column_stack([nn, log, branch]), axis=1), control, 'v19 full reconstruction')
                control_metrics = score(control)
                assert control_metrics == r71['stages'][s]['metrics']['R71']
            reports[mode] = {'reference': rr, 'spatial': sr}
            for label, value in (('base', base), ('reference', p), ('unit_spatial', branch)):
                old.previous.save_array(folder / (mode + '_' + label + '.npy'), value)
            if mode == 'candidate':
                for label, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
                    with (folder / (label + '.pkl')).open('xb') as f:
                        pickle.dump(value, f, protocol=5)
        candidate = np.median(np.column_stack([nn, log, branch_values['candidate']]), axis=1)
        assert candidate.shape == (n,) and np.isfinite(candidate).all() and (candidate > 0).all()
        metrics = {'v19': control_metrics, 'v18': score(v18), 'R76': score(candidate)}
        assert metrics['v19'] == r71['stages'][s]['metrics']['R71']
        assert metrics['v18'] == r71['stages'][s]['metrics']['v18']
        old.previous.save_array(folder / 'predictions.npy', candidate)
        old.exact(candidate, vector(folder / 'predictions.npy', n), 'Saved R76 unchanged')
        assert frames == {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        diff = old.previous.prior.log_run.differences
        item = {'stage': s, 'status': 'completed', 'fit_rows': len(fi), 'query_rows': n,
            'dates': p40['splits'][s], 'original_frame_sha256': frames, 'metrics': metrics,
            'differences': {label: diff(metrics[label], metrics['R76']) for label in ('v19', 'v18')},
            'unit_branch_metrics': branch_metrics,
            'unit_branch_differences': {layer: diff(branch_metrics['control'][layer], branch_metrics['candidate'][layer]) for layer in ('base', 'reference', 'spatial')},
            'pipeline_reports': reports, 'R19_reference_full_vector_exact': True,
            'R40_unit_spatial_full_vector_exact': True, 'R19_and_R40_full_metrics_exact': True,
            'v19_complete_combination_exact': True, 'controls_full_metrics_exact': True,
            'changed_unit_rows': int(np.count_nonzero(branch_values['candidate'] != unit_control)),
            'changed_final_rows': int(np.count_nonzero(candidate != control)),
            'new_NN_fits': 0, 'tree_NN_model_loads': 0, 'model_predict_calls': 0,
            'reference_predict_calls': 2, 'spatial_predict_calls': 2,
            'query_labels_used_only_for_evaluation': True,
            'prediction_sha256': sha(folder / 'predictions.npy'), 'elapsed_seconds': time.monotonic() - start,
            'artifact_sha256': {rel(p): sha(p) for p in folder.iterdir() if p.is_file()}}
        write(folder / 'results.json', item)
        result['stages'][s] = item
        result['checked_stages'].append(s)
        print(json.dumps({'stage': s, 'unit_spatial_vs_original': item['unit_branch_differences']['spatial']['overall'],
            'R76_vs_v19': item['differences']['v19']['overall'], 'changed_final_rows': item['changed_final_rows']}), flush=True)
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    assert sha(DOCUMENT) == expected_plan_sha and sha(selector_path) == selector_sha
    result.update(status='three_periods_completed_no_adoption_decision', elapsed_seconds=time.monotonic() - tick,
        reference_predict_calls=6, spatial_predict_calls=6,
        protected_after={name: sha(ROOT / name) for name in protected}, selector_sha256_after=sha(selector_path))
    assert result['protected_after'] == protected
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'result_sha256': sha(OUT / 'results.json'), 'new_fits': 0, 'new_CSV': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert sha(DOCUMENT) == args.plan_sha and not OUT.exists(), 'Plan changed or prior attempt exists'
    OUT.mkdir()
    tick = time.monotonic()
    started, code, failure = old.now(), 0, None
    write(OUT / 'started.json', {'started_at_utc': started, 'command': sys.argv,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        run(args.plan_sha)
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'exit_code': code, 'failure': failure, 'started_at_utc': started,
        'completed_at_utc': old.now(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
