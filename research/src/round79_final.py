"""R79/v22 final composition from saved TWD vectors; zero fits/model loads."""
from pathlib import Path
import argparse, contextlib, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round76_77_final as previous

ROOT, old, exp, spatial = previous.ROOT, previous.old, previous.exp, previous.spatial
f48 = previous.f48
sha, rel, write = previous.sha, previous.rel, previous.write
OUT = ROOT / 'outputs/round79/final'
PLAN = ROOT / 'reports/round79_final_plan.md'
PLAN_SHA = 'cd3dd3b71fc24e405c8ade92ee89bc928e8ad76a6feca258c79e66e65455ae2d'
DECISION = 'outputs/round79/decision.json'
DECISION_SHA = '12c5c188057ef36704f6e0553d1c9cfbd7d3d345dbbcac083275aa8fe9405ef5'
CONTROL = 'outputs/round76_77/final/trial_release.json'
CONTROL_SHA = 'd77da69bc3fa10fc41a2e811294ec24ab0d39dfb756a9f89951bbc5d8778ac2f'
MODEL_RESULT = 'outputs/round79/final_model/results.json'
NATIVE = 'outputs/round79/final_model/model/predictions_twd.npy'
CSV = 'outputs/submission_v22_trial.csv'
FORMULA = 'median(saved_R71_final_NN42,saved_R46_final_log,spatial(reference(.75*R79_native_TWD+.25*old_unit_LGB_TWD)))'


def run(plan_sha):
    tick = time.monotonic(); inputs = {}; known = {}; sources = {}
    def bind(name, digest):
        name = name.replace(chr(92), '/'); path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs: assert sha(path) == digest, name
        inputs[name] = digest
        return path
    def small(name, digest, receipt=True):
        path = bind(name, digest)
        if receipt:
            receipt_path = path.with_suffix('.sha256')
            assert receipt_path.read_text(encoding='ascii').strip() == digest, name
            bind(rel(receipt_path), sha(receipt_path))
        return json.loads(path.read_text(encoding='utf8'))
    assert plan_sha == PLAN_SHA
    bind(rel(PLAN), PLAN_SHA)
    decision = small(DECISION, DECISION_SHA, receipt=False)
    assert decision['round'] == 79 and decision['status'] == 'advance_to_one_fixed_final_candidate_not_yet_released'
    assert decision['proposed_filename'] == CSV and not decision['final_version_chosen']
    p76 = small(CONTROL, CONTROL_SHA)
    model = small(MODEL_RESULT, sha(ROOT / MODEL_RESULT))
    assert model['status'] == 'final_unit_model_ready_no_CSV'
    assert model['text_plan_sha256'] == PLAN_SHA and model['decision_sha256'] == DECISION_SHA
    assert model['fit_rows'] == 467221 and model['query_rows'] == 47073
    assert model['full_query_raw_TWD_reload_exact'] and model['new_fits'] == 1 and model['new_NN_fits'] == 0
    assert model['new_CSV'] is False and model['native_prediction_path'] == NATIVE
    for doc in (p76, model):
        for name, digest in doc['source_sha256'].items():
            assert name not in sources or sources[name] == digest, name
            sources[name] = digest
        for family in ('input_sha256', 'artifact_sha256'):
            for name, digest in doc[family].items():
                name = name.replace(chr(92), '/')
                assert name not in known or known[name] == digest, name
                known[name] = digest
    sources[rel(Path(__file__))] = sha(Path(__file__))
    assert sources[rel(Path(previous.__file__))] == sha(Path(previous.__file__))
    for name, digest in sources.items(): bind(name, digest)
    def checked(name): return bind(name, known[name])
    def document(name): return json.loads(checked(name).read_text(encoding='utf8'))
    def vec(name, zero=False):
        value = np.load(checked(name), allow_pickle=False)
        assert value.dtype == np.float64 and value.shape == (47073,) and np.isfinite(value).all(), name
        assert (value >= 0).all() if zero else (value > 0).all(), name
        return value
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 22
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    assert protected == decision['protected_before'] == model['protected_before'] == model['protected_after']
    selector = ROOT / 'outputs/current_private_candidate.json'; selector_sha = sha(selector)
    assert selector_sha == decision['selector_sha256_before'] == model['selector_sha256_before'] == model['selector_sha256_after']
    historical_path = 'outputs/round79/results.json'
    bind(historical_path, decision['input_sha256'][historical_path])
    frames = {s: pd.read_pickle(checked(f'data_cache/{s}_features.pkl')) for s in ('train', 'valid', 'test')}
    schema = document('reports/feature_schema.json')
    for s, rows in (('train', 455784), ('valid', 11994), ('test', 47073)):
        assert len(frames[s]) == rows and frames[s].index.equals(pd.RangeIndex(rows))
        assert list(frames[s]) == schema['columns'] and len(frames[s].columns) == 87 and old.TARGET not in frames[s]
    def index(name, bound, rows, identity=False):
        a = np.load(checked(name), allow_pickle=False)
        assert a.dtype == np.int64 and a.shape == (rows,) and (np.diff(a) > 0).all() and a.min() >= 0 and a.max() < bound, name
        if identity: assert np.array_equal(a, np.arange(rows, dtype=np.int64)), name
        return a
    kt = index(old.GEO + 'train_source_cache_indices.npy', 455784, 455255)
    kv = index(old.GEO + 'valid_source_cache_indices.npy', 11994, 11994, True)
    kf = index(old.GEO + 'full_source_cache_indices.npy', 467249, 467221)
    kq = index(old.GEO + 'test_source_cache_indices.npy', 47073, 47073, True)
    X = pd.concat([frames['train'].iloc[kt], frames['valid'].iloc[kv]], ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq = frames['test'].iloc[kq].reset_index(drop=True); del frames
    before = {'X': spatial.frame_sha(X), 'Xq': spatial.frame_sha(Xq)}
    assert set(model['input_frame_sha256']) == {'X', 'Xq', 'geo', 'geoq'}
    assert model['input_frame_sha256'] == p76['canonical_frames_sha256']
    assert all(before[key] == model['input_frame_sha256'][key] for key in before)
    for kind, rows in (('fit', 467221), ('query', 47073)):
        a = index(f'outputs/round79/final_model/{kind}_indices.npy', rows, rows, True)
        assert exp.original.value_sha(a, '<i8') == model[kind + '_indices_content_sha256']
    # The tree's original R6 geo is already verified by the model authority;
    # this composer loads only the frozen R39 query geo for spatial prediction.
    manifest = document('outputs/round39/geography/final/manifest.json')
    newquery = pd.read_pickle(checked(manifest['features']['query']['path']))
    geo_sha = spatial.frame_sha(newquery)
    assert geo_sha == manifest['features']['query']['frame_sha256'] and newquery.index.equals(Xq.index)
    assert newquery.shape == (47073, 8) and newquery.columns.is_unique
    with checked(old.REFERENCE).open('rb') as f: reference = pickle.load(f)
    with checked('outputs/round39/trial/artifacts/spatial_state.pkl').open('rb') as f: state = pickle.load(f)
    summary = document('outputs/round39/trial/artifacts/state_summary.json')
    assert state['state_summary'] == summary and summary['fit_rows'] == len(X) == 467221
    assert not summary['query_used_in_build'] and summary['fit_X_sha256'] == before['X']
    assert summary['fit_geo_sha256'] == manifest['features']['fit']['frame_sha256'] and summary['parameters'] == spatial.parameters()
    assert reference['fit_report']['fitting_rows'] == 467221
    assert summary['existing_reference_residual_sha256'] == exp.original.value_sha(reference['reference_residual'], '<f8')
    oldunit = vec(old.OLD_UNIT, True)
    original_native = vec('outputs/round40/trial/model/predictions.npy', True)
    native = vec(NATIVE, True); assert inputs[NATIVE] == model['native_prediction_sha256']
    nn = vec('outputs/round71/final/nn_spatial.npy')
    log = vec('outputs/round46/trial/recency_log_spatial.npy')
    old_branch = vec('outputs/round76_77/final/new_unit/spatial.npy')
    v20 = vec('outputs/round76/final/predictions.npy')
    controls = {'v20_saved_median': f48.exact(np.median(np.column_stack([nn, log, old_branch]), axis=1), v20, 'v20 saved median')}
    sample_name = rel(old.SAMPLE)
    assert pd.read_csv(checked(sample_name), nrows=0).columns.tolist() == [old.ID, old.TARGET]
    ids = pd.read_csv(checked(sample_name), encoding='utf-8-sig', usecols=[old.ID], dtype={old.ID: str}, keep_default_na=False)[old.ID].tolist()
    assert len(ids) == len(set(ids)) == 47073
    f48.csv_check(ROOT / 'outputs/submission_v20_trial.csv', ids, old.rounded(v20))
    write(OUT / 'input_contract.json', {'plan_sha256': plan_sha, 'decision_sha256': DECISION_SHA,
        'model_result_path': MODEL_RESULT, 'model_result_sha256': inputs[MODEL_RESULT],
        'source_sha256': sources, 'input_sha256': dict(inputs), 'protected_before': protected,
        'selector_sha256_before': selector_sha, 'raw_labels_loaded': False, 'test_labels_loaded': False,
        'reference_contains_fit_label_derived_state': True, 'new_fits_in_this_composer': 0})
    calls = []
    def branch(folder, B):
        folder.mkdir(parents=True, exist_ok=False)
        p, diag, rr = exp.original.predict_reference(reference, Xq, B, variant='trend', return_diagnostics=True)
        a, detail, sr = spatial.predict(state, Xq, newquery, B, p, diag)
        fallback = ~detail['applied'].to_numpy(dtype=bool)
        assert a[fallback].dtype == p[fallback].dtype and a[fallback].shape == p[fallback].shape and a[fallback].tobytes() == p[fallback].tobytes(), 'Spatial fallback'
        for name, value in (('base', B), ('reference', p), ('spatial', a)): f48.array(folder / (name + '.npy'), value)
        for name, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
            with (folder / (name + '.pkl')).open('xb') as f: pickle.dump(value, f, protocol=5)
        calls.append({'folder': rel(folder), 'reference_report': rr, 'spatial_report': sr, 'fallback_bitwise_equal': True})
        return p, a
    control_base = .75 * original_native + .25 * oldunit
    controls['v20_unit_base'] = f48.exact(control_base, vec('outputs/round76_77/final/new_unit/base.npy', True), 'v20 unit base')
    control_ref, control_spatial = branch(OUT / 'control_unit', control_base)
    controls['v20_unit_reference'] = f48.exact(control_ref, vec('outputs/round76_77/final/new_unit/reference.npy'), 'v20 unit reference')
    controls['v20_unit_spatial'] = f48.exact(control_spatial, old_branch, 'v20 unit spatial')
    controls['v20_replayed_median'] = f48.exact(np.median(np.column_stack([nn, log, control_spatial]), axis=1), v20, 'v20 replayed median')
    _, unit = branch(OUT / 'new_unit', .75 * native + .25 * oldunit)
    pred = np.median(np.column_stack([nn, log, unit]), axis=1)
    assert pred.dtype == np.float64 and pred.shape == (47073,) and np.isfinite(pred).all() and (pred > 0).all()
    f48.array(OUT / 'predictions.npy', pred)
    f48.exact(pred, np.load(OUT / 'predictions.npy', allow_pickle=False), 'Saved candidate')
    prices = old.rounded(pred); assert prices.dtype == np.int64 and (prices > 0).all()
    payload = {'headers': [old.ID, old.TARGET], 'ids': ids, 'prices': prices.tolist(), 'round': 79, 'rows': 47073,
        'prediction_path': rel(OUT / 'predictions.npy'), 'prediction_sha256': sha(OUT / 'predictions.npy'),
        'sample_ID_path': sample_name, 'sample_ID_sha256': inputs[sample_name], 'intended_filename': CSV,
        'rounding': 'maximum(rint(unrounded candidate TWD),1).astype(int64), once'}
    write(OUT / 'export_payload.json', payload)
    assert before == {'X': spatial.frame_sha(X), 'Xq': spatial.frame_sha(Xq)} and spatial.frame_sha(newquery) == geo_sha
    assert all(sha(ROOT / name) == digest for name, digest in sources.items()) and sha(PLAN) == PLAN_SHA
    after = {name: sha(ROOT / name) for name in protected}
    assert after == protected and sha(selector) == selector_sha
    assert {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')} == set(protected) - {'runs/versions.json'}
    proof = {key: value for key, value in payload.items() if key not in ('ids', 'prices', 'headers')}
    proof.update(status='final_vector_and_export_payload_ready_no_CSV', candidate='R79', formula=FORMULA,
        fit_rows=467221, query_rows=47073, plan_sha256=plan_sha, decision_sha256=DECISION_SHA,
        export_payload_path=rel(OUT / 'export_payload.json'), export_payload_sha256=sha(OUT / 'export_payload.json'),
        model_result_path=MODEL_RESULT, model_result_sha256=inputs[MODEL_RESULT], native_prediction_path=NATIVE,
        native_prediction_sha256=inputs[NATIVE], native_model_full_query_raw_TWD_reload_exact=True,
        source_sha256=sources, input_sha256=inputs,
        artifact_sha256={rel(p): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'execution.log'},
        protected_before=protected, protected_after=after, selector_sha256_before=selector_sha, selector_sha256_after=sha(selector),
        controls=controls, old_v20_CSV_exact=True, canonical_frames_sha256=before, spatial_query_frame_sha256=geo_sha,
        call_reports=calls, reference_predict_calls=2, spatial_predict_calls=2, reference_or_spatial_refits=0,
        new_fits_in_this_composer=0, new_NN_fits=0, separately_completed_new_tree_fits=1,
        models_loaded_in_this_composer=0, tree_NN_model_predict_calls_in_this_composer=0, saved_prediction_vector_reload_exact=True,
        historical_results=historical_path, historical_results_sha256=inputs[historical_path],
        prediction_min_TWD=float(pred.min()), prediction_max_TWD=float(pred.max()),
        changed_unrounded_rows_vs_v20=int(np.count_nonzero(pred - v20)),
        mean_absolute_prediction_change_vs_v20_TWD=float(np.abs(pred - v20).mean()), prediction_changes_are_not_errors=True,
        public_mae=None, private_mae=None, raw_labels_loaded=False, test_labels_loaded=False,
        reference_contains_fit_label_derived_state=True, new_CSV=False, selector_updated=False, registry_updated=False,
        uploaded=False, final_version_chosen=False, elapsed_seconds=time.monotonic() - tick, completed_at_utc=old.stamp())
    write(OUT / 'trial_release.json', proof)
    print(json.dumps({'status': proof['status'], 'elapsed_seconds': proof['elapsed_seconds'],
        'rows': 47073, 'prediction_sha256': proof['prediction_sha256'], 'controls': controls}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True); parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert args.plan_sha == PLAN_SHA == sha(PLAN) and sha(ROOT / DECISION) == DECISION_SHA
    assert not OUT.exists() and not (ROOT / CSV).exists()
    OUT.mkdir(parents=True, exist_ok=False); started = old.stamp(); tick = time.monotonic(); code = 1; failure = None
    write(OUT / 'started.json', {'command': sys.argv, 'started_at_utc': started, 'plan_sha256': args.plan_sha,
        'decision_sha256': DECISION_SHA, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    with (OUT / 'execution.log').open('x', encoding='utf8') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try: run(args.plan_sha); code = 0
            except Exception: failure = traceback.format_exc(); print(failure, flush=True)
    event = {'exit_code': code, 'failure': failure, 'elapsed_seconds': time.monotonic() - tick,
        'started_at_utc': started, 'completed_at_utc': old.stamp(), 'source_sha256': sha(Path(__file__)),
        'plan_sha256': args.plan_sha, 'automatic_retry': False, 'log_path': rel(OUT / 'execution.log'), 'log_sha256': sha(OUT / 'execution.log')}
    write(OUT / 'execution.json', event); print(json.dumps(event), flush=True)
    return code


if __name__ == '__main__': raise SystemExit(main())
