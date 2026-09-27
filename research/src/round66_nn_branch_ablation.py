"""R66 saved-output ablation of the old raw branch in the R65 NN base; zero fits."""
from pathlib import Path
import json
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd
import round48_nn_experiment as old

ROOT = old.ROOT
OUT = ROOT / 'outputs/round66'
STAGES = old.STAGES
DOCUMENT = ROOT / 'reports/round66_plan.md'
sha, read, write, spatial = old.sha, old.locked, old.write, old.spatial
rel = lambda p: Path(p).relative_to(ROOT).as_posix()
NATIVE_FORMULA = '.5*R54_area_native_TWD+.5*R65_prior_native_TWD'
BASE_FORMULA = '.75*saved_R65_native_mean_TWD+.25*old_unit_LGB_TWD'
FORMULA = 'median(spatial(reference(.75*saved_R65_native_mean_TWD+.25*old_unit_LGB_TWD)),R46_recency_log_spatial,R40_unit_spatial)'
CONTROLS = (('v17', 63, 'R63'), ('R65', 65, 'R65'), ('v16', 60, 'R60'),
            ('v15', 54, 'R54'), ('v13', 53, 'R53'), ('v11', 49, 'R49'),
            ('v12', 50, 'R50'), ('R62', 62, 'R62'))


def run():
    start = time.monotonic()
    inputs = {}

    def checked(name, digest):
        name = name.replace('\\', '/')
        path = ROOT / name
        assert sha(path) == digest, name
        assert name not in inputs or inputs[name] == digest, name
        inputs[name] = digest
        return path

    def small(name):
        path = ROOT / name
        value = read(path)
        inputs[name] = sha(path)
        inputs[rel(path.with_suffix('.sha256'))] = sha(path.with_suffix('.sha256'))
        return value

    def artifact(record, name):
        mapping = {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}
        return checked(name, mapping[name])

    def vector(name, digest, rows):
        return old.vector(checked(name, digest), rows)

    protected = {rel(p): sha(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')}
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    p48 = small('outputs/round48/plan.json')
    p52 = small('outputs/round52/plan.json')
    records = {r: small(f'outputs/round{r}/results.json') for r in (48, 49, 50, 53, 54, 60, 62, 63, 65)}
    assert all(r['checked_stages'] == list(STAGES) for r in records.values())
    source = old.previous.merge(old.previous.sources(), {rel(p): sha(p) for p in (Path(__file__), Path(old.__file__))})
    text_plan_sha = sha(DOCUMENT)
    inputs[rel(DOCUMENT)] = text_plan_sha
    write(OUT / 'plan.json', {'round': 66, 'candidate': 'R66', 'source_sha256': source,
        'text_plan_sha256': text_plan_sha, 'stage_order': list(STAGES),
        'split_contract': p48['split_contract'], 'native_mean_formula': NATIVE_FORMULA,
        'base_formula': BASE_FORMULA, 'formula': FORMULA, 'primary_control': 'v17=R63',
        'direct_ablation_control': 'R65', 'new_NN_fits': 0, 'model_predict_calls': 0,
        'encoder_prior_weights_rebuilt': False, 'three_periods_regardless_of_scores': True,
        'relative_to_R65_only_oldraw_base_removed': True,
        'relative_to_v17_also_includes_R65_prior_input_change': True,
        'query_labels_used_only_by_scorer': True, 'new_gate': None})
    Xall, yall = old.previous.load_training_rows(p48)
    result = {'round': 66, 'candidate': 'R66', 'status': 'running', 'formula': FORMULA,
        'base_formula': BASE_FORMULA, 'native_mean_formula': NATIVE_FORMULA,
        'checked_stages': [], 'stages': {}, 'new_NN_fits': 0, 'model_predict_calls': 0,
        'new_submission_generated': False, 'fresh_blind_validation': False,
        'protected_before': protected, 'source_sha256': source, 'plan_sha256': text_plan_sha}
    for s in STAGES:
        tick = time.monotonic()
        folder = OUT / s
        folder.mkdir()
        b, bs = p48['bindings'][s], p52['bindings'][s]
        a48, a54, a63, a65 = (records[r]['stages'][s] for r in (48, 54, 63, 65))
        indices = {}
        for kind in ('fit', 'query'):
            name = f'outputs/round48/{s}/{kind}_indices.npy'
            indices[kind] = np.load(artifact(a48, name), allow_pickle=False)
            assert old.original.value_sha(indices[kind], '<i8') == p48['split_contract'][s][kind + '_indices_content_sha256']
            for r, record in ((54, a54), (63, a63), (65, a65)):
                name = f'outputs/round{r}/{s}/{kind}_indices.npy'
                assert np.array_equal(indices[kind], np.load(artifact(record, name), allow_pickle=False))
            old.previous.save_array(folder / f'{kind}_indices.npy', indices[kind])
        fi, qi = indices['fit'], indices['query']
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        n = len(qi)
        frames = {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        for key, digest in frames.items():
            assert digest == a48['original_frame_sha256'][key] == a63['original_frame_sha256'][key] == a65['original_frame_sha256'][key]
        # R54 binds the same canonical indices and native artifacts above; it has no frame map.
        natives = []
        for r, record in ((54, a54), (65, a65)):
            name = f'outputs/round{r}/{s}/model/predictions_twd.npy'
            natives.append(old.vector(artifact(record, name), n))
        mean = old.vector(artifact(a65, f'outputs/round65/{s}/native_mean.npy'), n)
        old.exact(mean, .5 * natives[0] + .5 * natives[1], 'R65 native mean identity')
        raw, unit = [vector(b[k], p48['input_sha256'][b[k]], n) for k in ('old_raw', 'old_unit')]
        old.exact(old.vector(artifact(a65, f'outputs/round65/{s}/base.npy'), n),
                  .75 * (.5 * raw + .5 * mean) + .25 * unit, 'R65 original base identity')
        B = .75 * mean + .25 * unit
        with checked(b['reference'], p48['input_sha256'][b['reference']]).open('rb') as f:
            reference = pickle.load(f)
        with checked(b['state'], p48['input_sha256'][b['state']]).open('rb') as f:
            state = pickle.load(f)
        assert state['state_summary']['fit_X_sha256'] == frames['fit_X'] and not state['state_summary']['query_used_in_build']
        assert state['parameters'] == p48['spatial_parameters'] and reference['fit_report']['fitting_rows'] == len(fi)
        geo = pd.read_pickle(checked(b['query_geo'], p48['input_sha256'][b['query_geo']]))
        assert spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256'] and geo.index.equals(Xq.index)
        p, diag, rr = old.original.predict_reference(reference, Xq, B, variant='trend', return_diagnostics=True)
        branch, detail, sr = spatial.predict(state, Xq, geo, B, p, diag)
        fallback = ~detail['applied'].to_numpy(dtype=bool)
        old.exact(branch[fallback], p[fallback], 'spatial fallback')
        log = vector(bs['recency_log'], p52['input_sha256'][bs['recency_log']], n)
        ub = vector(b['unit_spatial'], p48['input_sha256'][b['unit_spatial']], n)
        candidate = np.median(np.column_stack([branch, log, ub]), axis=1)
        assert candidate.shape == (n,) and np.isfinite(candidate).all() and (candidate > 0).all()
        predictions = {'R66': candidate, 'v5': vector(b['v5'], p48['input_sha256'][b['v5']], n)}
        for label, r, _ in CONTROLS:
            name = f'outputs/round{r}/{s}/predictions.npy'
            predictions[label] = vector(name, records[r]['stages'][s]['prediction_sha256'], n)
        prior_branch = old.vector(artifact(a65, f'outputs/round65/{s}/nn_spatial.npy'), n)
        old.exact(predictions['R65'], np.median(np.column_stack([prior_branch, log, ub]), axis=1), 'R65 complete combination identity')
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p48['splits'][s][0], p48['target_group_weights'])
        metrics = {label: score(v) for label, v in predictions.items()}
        for label, r, key in (*CONTROLS, ('v5', 53, 'v5')):
            assert metrics[label] == records[r]['stages'][s]['metrics'][key], label
        for name, v in (('native_mean', mean), ('base', B), ('reference', p), ('nn_spatial', branch), ('predictions', candidate)):
            old.previous.save_array(folder / f'{name}.npy', v)
        for name, v in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
            with (folder / f'{name}.pkl').open('xb') as f:
                pickle.dump(v, f, protocol=5)
        item = {'stage': s, 'status': 'completed', 'fit_rows': len(fi), 'query_rows': n,
            'dates': p48['splits'][s], 'metrics': metrics,
            'differences': {label: old.previous.prior.log_run.differences(metrics[label], metrics['R66']) for label in predictions if label != 'R66'},
            'original_frame_sha256': frames, 'native_mean_reconstruction_exact': True,
            'R65_base_and_complete_combination_exact': True, 'controls_full_metrics_exact': True,
            'new_NN_fits': 0, 'model_predict_calls': 0, 'query_labels_used_only_by_scorer': True,
            'reference_predict_calls': 1, 'spatial_predict_calls': 1, 'reference_report': rr, 'spatial_report': sr,
            'prediction_sha256': sha(folder / 'predictions.npy'), 'elapsed_seconds': time.monotonic() - tick,
            'artifact_sha256': {rel(f): sha(f) for f in folder.iterdir() if f.is_file()}}
        write(folder / 'results.json', item)
        result['stages'][s] = item
        result['checked_stages'].append(s)
        print(json.dumps({'stage': s, 'new_NN_fits': 0, 'vs_v17': {k: item['differences']['v17'][k]['relative_change_pct'] for k in ('overall', 'weighted')},
                          'vs_R65': {k: item['differences']['R65'][k]['relative_change_pct'] for k in ('overall', 'weighted')}}), flush=True)
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    result.update(status='three_periods_completed_no_adoption_decision', input_sha256=inputs,
                  elapsed_seconds=time.monotonic() - start, protected_after={name: sha(ROOT / name) for name in protected})
    assert result['protected_before'] == result['protected_after']
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    assert not OUT.exists(), 'Existing output: no overwrite'
    OUT.mkdir()
    tick = time.monotonic()
    code, failure = 0, None
    write(OUT / 'started.json', {'started_at_utc': old.now(), 'command': sys.argv,
                               'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        run()
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'exit_code': code, 'failure': failure, 'elapsed_seconds': time.monotonic() - tick,
                                 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
