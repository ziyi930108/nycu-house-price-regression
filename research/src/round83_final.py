"""R83/v24: saved two-seed unit TWD mean, zero model calls in this composer."""
from pathlib import Path
import argparse, contextlib, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round80_final as previous

ROOT, old, exp, spatial, f48 = previous.ROOT, previous.old, previous.exp, previous.spatial, previous.f48
sha, rel, write = previous.sha, previous.rel, previous.write
OUT = ROOT / 'outputs/round83/final'
PLAN = ROOT / 'reports/round83_final_plan.md'
PLAN_SHA = '792631bfde4e83e89134246e55a91fe05ce0196c2bc97c7b4ed360f48a10b4c6'
DECISION = 'outputs/round83/decision.json'
DECISION_SHA = '759d2ec0cfafe3c05575ecc0536e4287b5e006286564b922cea60cb5f34c7b28'
CONTROL = 'outputs/round80/final/trial_release.json'
CONTROL_SHA = '7e79c6060a744366643d96105e3d556497f0db71d687e11b8babcfddad35dcc0'
CONTROL_RELEASE = 'outputs/round80/final_export/release.json'
CONTROL_RELEASE_SHA = '4243f4c19332f7d433d85e1964af2dce2200e90ae871be8df7a1e7c3a512795f'
SELECTOR_SHA = 'c993a861fa6766fb5c9da8c156521bb28886a459587653ee8734b2927b9e87a8'
SEED42_RESULT = 'outputs/round79/final_model/results.json'
SEED42_RESULT_SHA = '4b7172156359f476add69af11f9d2f53c78b3879bb469acf34dbd072c0db7942'
NATIVE42 = 'outputs/round79/final_model/model/predictions_twd.npy'
NATIVE42_SHA = '9df25c674546142bf63a6f7ef5a20dd0fc85067471879f4a5c3d0f9d00970612'
MODEL_RESULT = 'outputs/round83/final_model/results.json'
MODEL_SOURCE = 'src/round83_final_unit.py'
HELPER = 'src/round83_seed_unit.py'
HELPER_SHA = 'c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b'
NATIVE137 = 'outputs/round83/final_model/model/predictions_twd.npy'
CSV = 'outputs/submission_v24_trial.csv'
FORMULA = 'median(saved_R71_final_NN42,saved_R46_final_log,spatial(reference_trend(.5*saved_seed42_TWD+.5*new_seed137_TWD)))'


def run(plan_sha):
    tick = time.monotonic(); inputs = {}; known = {}
    def bind(name, digest):
        name = name.replace(chr(92), '/'); path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs: assert sha(path) == digest, name
        inputs[name] = digest
        return path
    def small(name, digest):
        path = bind(name, digest); receipt = path.with_suffix('.sha256')
        assert receipt.read_text(encoding='ascii').strip() == digest, name
        bind(rel(receipt), sha(receipt))
        return json.loads(path.read_text(encoding='utf8'))
    def merge(record):
        for family in ('input_sha256', 'artifact_sha256'):
            for name, digest in record[family].items():
                name = name.replace(chr(92), '/')
                assert name not in known or known[name] == digest, name
                known[name] = digest
    def checked(name): return bind(name, known[name])
    def document(name): return json.loads(checked(name).read_text(encoding='utf8'))
    def vec(name, zero=False):
        value = np.load(checked(name), allow_pickle=False)
        assert value.dtype == np.float64 and value.shape == (47073,) and np.isfinite(value).all(), name
        assert (value >= 0).all() if zero else (value > 0).all(), name
        return value
    assert plan_sha == PLAN_SHA; bind(rel(PLAN), PLAN_SHA)
    decision = small(DECISION, DECISION_SHA)
    assert decision['round'] == 83 and decision['status'] == 'advance_one_fixed_complementary_candidate_not_yet_released'
    assert decision['proposed_filename'] == CSV and decision['new_fits_allowed'] == 1 and decision['final_fit_seed'] == 137
    assert not decision['final_version_chosen'] and not decision['selector_updated']
    assert decision['priority_retained'] == 'outputs/submission_v22_trial.csv' and decision['backup_retained'] == 'outputs/submission_v20_trial.csv'
    for name, digest in decision['input_sha256'].items(): bind(name, digest)
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 24; protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    assert protected == decision['protected_before']
    selector = ROOT / 'outputs/current_private_candidate.json'; selector_sha = sha(selector)
    assert selector_sha == SELECTOR_SHA == decision['selector_sha256_before']
    p80 = small(CONTROL, CONTROL_SHA); release80 = small(CONTROL_RELEASE, CONTROL_RELEASE_SHA)
    assert p80['round'] == 80 and p80['status'] == 'final_vector_and_export_payload_ready_no_CSV'
    assert p80['fit_rows'] == 467221 and p80['query_rows'] == 47073
    assert p80['native_prediction_path'] == NATIVE42 and p80['native_prediction_sha256'] == NATIVE42_SHA
    assert release80['status'] == 'experimental_v23_trial_generated' and release80['rows'] == 47073
    assert release80['path'] == 'outputs/submission_v23_trial.csv' and release80['sha256'] == protected[release80['path']]
    assert release80['candidate_proof_path'] == CONTROL and release80['candidate_proof_sha256'] == CONTROL_SHA
    seed42 = small(SEED42_RESULT, SEED42_RESULT_SHA)
    model = small(MODEL_RESULT, sha(ROOT / MODEL_RESULT))
    for record in (p80, seed42, model): merge(record)
    sources = dict(p80['source_sha256'])
    for record in (seed42, model):
        for name, digest in record['source_sha256'].items():
            assert name not in sources or sources[name] == digest, name
            sources[name] = digest
    assert sources[rel(Path(previous.__file__))] == sha(Path(previous.__file__))
    assert sources[HELPER] == HELPER_SHA
    sources[rel(Path(__file__))] = sha(Path(__file__))
    for name, digest in sources.items(): bind(name, digest)
    for root_name, expected_source in (('outputs/round80/final', sources['src/round80_final.py']),
            ('outputs/round79/final_model', seed42['source_sha256']['src/round79_final_unit.py']),
            ('outputs/round83/final_model', model['source_sha256'][MODEL_SOURCE])):
        name = root_name + '/execution.json'; event = small(name, sha(ROOT / name))
        assert event['exit_code'] == 0 and event['source_sha256'] == expected_source and event['failure'] is None, name
    export_event = small('outputs/round80/final_export/execution.json', sha(ROOT / 'outputs/round80/final_export/execution.json'))
    assert export_event['exit_code'] == 0
    assert model['status'] == seed42['status'] == 'final_unit_model_ready_no_CSV'
    assert model['fit_rows'] == seed42['fit_rows'] == 467221 and model['query_rows'] == seed42['query_rows'] == 47073
    assert model['text_plan_sha256'] == plan_sha and model['decision_sha256'] == DECISION_SHA
    assert model['new_fits'] == 1 and model['new_NN_fits'] == 0 and model['new_CSV'] is False
    assert model['protected_before'] == model['protected_after'] == protected
    assert model['selector_sha256_before'] == model['selector_sha256_after'] == selector_sha
    assert model['full_query_raw_TWD_reload_exact'] is True and seed42['full_query_raw_TWD_reload_exact'] is True
    assert model['native_prediction_path'] == NATIVE137 and known[NATIVE137] == model['native_prediction_sha256']
    assert seed42['native_prediction_path'] == NATIVE42 and known[NATIVE42] == seed42['native_prediction_sha256'] == NATIVE42_SHA
    assert model['input_frame_sha256'] == seed42['input_frame_sha256']
    for key in ('fit_indices_content_sha256', 'query_indices_content_sha256'):
        assert model[key] == seed42[key], key
    meta = document('outputs/round83/final_model/model/metadata.json')
    oldmeta = document('outputs/round79/final_model/model/metadata.json')
    wrapper = document('outputs/round83/final_model/model/feature_state.json')
    legacy = document('outputs/round79/final_model/model/feature_state.json')
    assert wrapper['legacy_feature_state'] == legacy and meta['legacy_feature_state_content_sha256'] == legacy['state_content_sha256']
    config = dict(oldmeta['parameters']); assert config['random_seed'] == 42; config['random_seed'] = 137
    assert meta['parameters'] == wrapper['parameters'] == config and meta['baseline_parameters'] == oldmeta['parameters']
    assert meta['seed'] == meta['actual_random_seed'] == 137 and meta['prediction_units'] == 'TWD'
    assert meta['fit_rows'] == 467221 and meta['query_rows'] == meta['reload_rows'] == 47073
    assert meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23
    assert meta['full_query_raw_TWD_reload_exact'] is True and meta['original_R79_legacy_feature_state_exact'] is True
    assert meta['date_weight_provenance_verified_in_driver'] is True
    assert not meta['date_weights_recomputed'] and not meta['date_weights_renormalized'] and not meta['query_labels_used']
    same_keys = ('fit_target_values_sha256', 'fit_transformed_target_values_sha256', 'fit_effective_area_sha256',
        'fit_factor_sha256', 'fit_area_weights_sha256', 'fit_date_weights_sha256', 'fit_sample_weights_sha256',
        'fit_original_feature_values_sha256', 'fit_geo_feature_values_sha256', 'fit_model_feature_values_sha256',
        'input_frame_sha256', 'fit_indices_content_sha256', 'query_indices_content_sha256')
    for key in same_keys: assert meta[key] == oldmeta[key], key
    fitted, loaded = meta['fitted_native_capacity'], meta['loaded_native_capacity']
    assert set(fitted) == set(loaded)
    for key in fitted:
        if key == 'actual_learning_rate':
            assert all(np.isclose(float(v), .065, rtol=1e-7, atol=0) for v in (fitted[key], loaded[key]))
        else: assert fitted[key] == loaded[key], key
    assert loaded['actual_random_seed'] == 137 and loaded['actual_trees'] == 2999 and loaded['actual_model_depth'] == 8
    assert loaded['actual_loss_function'] == loaded['actual_eval_metric'] == 'MAE'
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
    assert before == p80['canonical_frames_sha256'] == {k: model['input_frame_sha256'][k] for k in ('X', 'Xq')}
    manifest = document('outputs/round39/geography/final/manifest.json')
    newquery = pd.read_pickle(checked(manifest['features']['query']['path'])); geo_sha = spatial.frame_sha(newquery)
    assert geo_sha == manifest['features']['query']['frame_sha256'] == p80['spatial_query_frame_sha256']
    assert newquery.index.equals(Xq.index) and newquery.shape == (47073, 8) and newquery.columns.is_unique
    with checked(old.REFERENCE).open('rb') as f: reference = pickle.load(f)
    with checked('outputs/round39/trial/artifacts/spatial_state.pkl').open('rb') as f: state = pickle.load(f)
    summary = document('outputs/round39/trial/artifacts/state_summary.json')
    assert state['state_summary'] == summary and summary['fit_rows'] == len(X) == 467221
    assert not summary['query_used_in_build'] and summary['fit_X_sha256'] == before['X']
    assert summary['fit_geo_sha256'] == manifest['features']['fit']['frame_sha256'] and summary['parameters'] == spatial.parameters()
    assert reference['fit_report']['fitting_rows'] == 467221
    assert summary['existing_reference_residual_sha256'] == exp.original.value_sha(reference['reference_residual'], '<f8')
    native42 = vec(NATIVE42, True); native137 = vec(NATIVE137, True)
    nn = vec('outputs/round71/final/nn_spatial.npy'); log = vec('outputs/round46/trial/recency_log_spatial.npy')
    old_branch = vec('outputs/round80/final/new_unit/spatial.npy'); v23 = vec('outputs/round80/final/predictions.npy')
    controls = {'v23_saved_median': f48.exact(np.median(np.column_stack([nn, log, old_branch]), axis=1), v23, 'v23 saved median')}
    sample_name = rel(old.SAMPLE)
    assert pd.read_csv(checked(sample_name), nrows=0).columns.tolist() == [old.ID, old.TARGET]
    ids = pd.read_csv(checked(sample_name), encoding='utf-8-sig', usecols=[old.ID], dtype={old.ID: str}, keep_default_na=False)[old.ID].tolist()
    assert len(ids) == len(set(ids)) == 47073 and all(ids)
    f48.csv_check(ROOT / release80['path'], ids, old.rounded(v23))
    write(OUT / 'input_contract.json', {'plan_sha256': plan_sha, 'decision_sha256': DECISION_SHA,
        'control_proof_path': CONTROL, 'control_proof_sha256': CONTROL_SHA,
        'model_result_path': MODEL_RESULT, 'model_result_sha256': inputs[MODEL_RESULT],
        'seed42_model_result_path': SEED42_RESULT, 'seed42_model_result_sha256': SEED42_RESULT_SHA,
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
    controls['v23_unit_base'] = f48.exact(native42, vec('outputs/round80/final/new_unit/base.npy', True), 'v23 unit base')
    control_ref, control_spatial = branch(OUT / 'control_unit', native42.copy())
    controls['v23_unit_reference'] = f48.exact(control_ref, vec('outputs/round80/final/new_unit/reference.npy'), 'v23 unit reference')
    controls['v23_unit_spatial'] = f48.exact(control_spatial, old_branch, 'v23 unit spatial')
    controls['v23_replayed_median'] = f48.exact(np.median(np.column_stack([nn, log, control_spatial]), axis=1), v23, 'v23 replayed median')
    mean = .5 * native42 + .5 * native137
    _, unit = branch(OUT / 'new_unit', mean)
    pred = np.median(np.column_stack([nn, log, unit]), axis=1)
    assert pred.dtype == np.float64 and pred.shape == (47073,) and np.isfinite(pred).all() and (pred > 0).all()
    f48.array(OUT / 'predictions.npy', pred)
    f48.exact(pred, np.load(OUT / 'predictions.npy', allow_pickle=False), 'Saved candidate')
    prices = old.rounded(pred); assert prices.dtype == np.int64 and (prices > 0).all()
    payload = {'headers': [old.ID, old.TARGET], 'ids': ids, 'prices': prices.tolist(), 'round': 83, 'rows': 47073,
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
    proof.update(status='final_vector_and_export_payload_ready_no_CSV', candidate='R83', formula=FORMULA,
        fit_rows=467221, query_rows=47073, plan_sha256=plan_sha, decision_sha256=DECISION_SHA,
        export_payload_path=rel(OUT / 'export_payload.json'), export_payload_sha256=sha(OUT / 'export_payload.json'),
        control_proof_path=CONTROL, control_proof_sha256=CONTROL_SHA,
        model_result_path=MODEL_RESULT, model_result_sha256=inputs[MODEL_RESULT], native_prediction_path=NATIVE137,
        native_prediction_sha256=inputs[NATIVE137], native_model_full_query_raw_TWD_reload_exact=True,
        seed42_model_result_path=SEED42_RESULT, seed42_model_result_sha256=SEED42_RESULT_SHA,
        seed42_native_prediction_path=NATIVE42, seed42_native_prediction_sha256=inputs[NATIVE42], seed42_full_query_raw_TWD_reload_exact=True,
        native_mean_path=rel(OUT / 'new_unit/base.npy'), native_mean_sha256=sha(OUT / 'new_unit/base.npy'),
        native_mean_formula='.5*seed42_TWD + .5*seed137_TWD',
        independent_native_model_reload_in_this_composer=False, inherited_R79_tree_fits=1,
        source_sha256=sources, input_sha256=inputs,
        artifact_sha256={rel(p): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'execution.log'},
        protected_before=protected, protected_after=after, selector_sha256_before=selector_sha, selector_sha256_after=sha(selector),
        controls=controls, old_v23_CSV_exact=True, canonical_frames_sha256=before, spatial_query_frame_sha256=geo_sha,
        call_reports=calls, reference_predict_calls=2, spatial_predict_calls=2, reference_or_spatial_refits=0,
        new_fits_in_this_composer=0, new_NN_fits=0, separately_completed_new_tree_fits=1,
        models_loaded_in_this_composer=0, tree_NN_model_predict_calls_in_this_composer=0, saved_prediction_vector_reload_exact=True,
        historical_results='outputs/round83/results.json', historical_results_sha256=inputs['outputs/round83/results.json'],
        prediction_min_TWD=float(pred.min()), prediction_max_TWD=float(pred.max()),
        changed_unrounded_rows_vs_v23=int(np.count_nonzero(pred - v23)),
        changed_rounded_rows_vs_v23=int(np.count_nonzero(prices != old.rounded(v23))),
        mean_absolute_prediction_change_vs_v23_TWD=float(np.abs(pred - v23).mean()),
        max_absolute_prediction_change_vs_v23_TWD=float(np.abs(pred - v23).max()), prediction_changes_are_not_errors=True,
        public_mae=None, private_mae=None, raw_labels_loaded=False, test_labels_loaded=False,
        reference_contains_fit_label_derived_state=True, new_CSV=False, selector_updated=False, registry_updated=False,
        priority_retained='outputs/submission_v22_trial.csv', backup_retained='outputs/submission_v20_trial.csv',
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
