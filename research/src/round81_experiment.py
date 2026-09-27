"""R81: three fixed legacy unit-LightGBM fits with saved fit-date weights."""
from pathlib import Path
import argparse
import contextlib
import json
import pickle
import sys
import time
import traceback
import lightgbm as lgb
import numpy as np
import pandas as pd
import round40_three_branch as unit_run
import round48_nn_experiment as old
import round81_recency_lgb as helper

ROOT, STAGES = old.ROOT, old.STAGES
OUT = ROOT / 'outputs/round81'
DOCUMENT = ROOT / 'reports/round81_plan.md'
PLAN_SHA = '15ce1debd8bf9433c3038761b57bdffc5d6b9d3fb7520e5408c6205a560ecef5'
HELPER_SHA = '8ce19dbc9800737d8de394a2bf266f6514cadd9d738a3dec1944434803b2ab3c'
SELECTOR_SHA = '357db63598a1b689a2128f0c53989dd97c2c79133218fb664f73019435f3eed7'
R79_SHA = '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6'
R80_SHA = 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6'
sha, read, write, spatial, rel = old.sha, old.locked, old.write, old.spatial, old.rel
BACKGROUND = ('v20', 'v21', 'v19', 'v18', 'v11', 'v12', 'v5')
CONTROLS = ('v22', 'v23', *BACKGROUND)
FORMULA = 'median(saved_R71_seed42_nn_spatial,saved_R46_log_spatial,spatial(reference(.75*saved_R79_native_TWD+.25*new_LGB_TWD)))'
NPZ_KEYS = {'train_num', 'query_num', 'train_cat', 'query_cat', 'train_y_million',
            'cardinalities', 'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'}


def exact(a, b, label):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def weight_summary(w):
    v = np.asarray(w, dtype=np.float64)
    total, squares = float(v.sum()), float(np.square(v).sum())
    return dict(rows=len(v), min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                sum=total, sum_squares=squares, effective_sample_size=total * total / squares)


def summary_equal(actual, saved):
    for k in ('rows', 'min', 'max'):
        assert actual[k] == saved[k], k
    for k in ('mean', 'sum', 'sum_squares', 'effective_sample_size'):
        assert np.isclose(actual[k], saved[k], rtol=1e-12, atol=1e-12), k


def date_weights(binding, X, yfit, n):
    prep, training = binding['preparation'], binding['R63_training']
    with np.load(ROOT / binding['prepared'], allow_pickle=False) as z:
        assert set(z.files) == NPZ_KEYS
        w = z['train_sample_weight'].copy()
        exact(z['train_y_million'], np.asarray(yfit / 1e6, dtype=np.float32), 'Canonical fit target/order')
    assert w.dtype == np.float32 and w.shape == (len(X),) and np.isfinite(w).all() and (w > 0).all()
    assert old.original.value_sha(w, '<f4') == prep['train_sample_weight_sha256']
    summary_equal(weight_summary(w), prep['train_sample_weight_summary'])
    summary_equal(weight_summary(w), training['train_sample_weight_summary'])
    assert prep['fit_rows'] == len(X) and prep['query_rows'] == n
    dates = prep['date_weight_summary_float64']
    d, invalid, outside = (X[k].to_numpy(dtype=np.float64) for k in ('trade_day_index', 'trade_date_invalid', 'trade_date_out_of_range'))
    assert np.isfinite(invalid).all() and np.isfinite(outside).all()
    for v, key in ((d, 'date_values_sha256'), (invalid, 'invalid_flags_sha256'), (outside, 'out_of_range_flags_sha256')):
        assert helper.values_sha256(v) == dates[key]
    valid = np.isfinite(d) & (invalid == 0) & (outside == 0)
    assert dates['reference_rows'] == len(X) and dates['valid_date_rows'] == int(valid.sum())
    assert dates['neutral_date_rows'] == int((~valid).sum()) and (w[~valid] == np.float32(1)).all()
    rules = dates['weight_rules']
    assert rules['half_life_days'] == 365 and not rules['weights_use_query'] and not rules['weights_use_targets']
    return w  # The saved values, not recomputed or renormalized date weights.


def fit_native(folder, stage, X, yfit, Xq, cats, w, binding, frames, spec, plan_sha, inputs):
    model_dir = folder / 'model'
    model_dir.mkdir()
    started, tick, error = old.now(), time.monotonic(), None
    write(model_dir / 'fit_started.json', {'started_at_utc': started, 'helper_sha256': HELPER_SHA,
        'parameters': helper.parameters(), 'prepared_path': binding['prepared'],
        'prepared_sha256': inputs[binding['prepared']], 'date_weight_sha256': old.original.value_sha(w, '<f4'),
        'query_labels_used': False, 'eval_set_used': False, 'early_stopping': False, 'automatic_retry': False})
    try:
        with (model_dir / 'fit.log').open('x', encoding='utf8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            model, state, meta = helper.fit_model(X, yfit, cats, w, stage)
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        fit_seconds = time.monotonic() - tick
        write(model_dir / 'fit_execution.json', {'exit_code': 1 if error else 0, 'failure': error,
            'started_at_utc': started, 'completed_at_utc': old.now(), 'elapsed_seconds': fit_seconds})
    # Persist the completed fit before later validation/inference can fail.
    model.booster_.save_model(str(model_dir / 'model.txt'))
    write(model_dir / 'feature_state.json', state)
    write(model_dir / 'fit_metadata.json', meta)
    fit_call_seconds, fit_seconds = fit_seconds, meta['fit_elapsed_seconds']
    assert state['fit_original_feature_values_sha256'] == helper._frame_sha(X)
    area_w = helper.area_weights(X, stage)
    product_w = area_w * w.astype(np.float64)
    assert meta['parameters'] == helper.parameters() and meta['fit_rows'] == len(X)
    assert meta['fit_target_values_sha256'] == helper.values_sha256(yfit)
    assert meta['fit_transformed_target_values_sha256'] == helper.values_sha256(yfit / helper.original.price_factor(X, 'unit'))
    assert meta['fit_area_weights_sha256'] == helper.values_sha256(area_w)
    assert meta['fit_date_weights_sha256'] == binding['preparation']['train_sample_weight_sha256']
    assert meta['fit_sample_weights_sha256'] == helper.values_sha256(product_w)
    summary_equal(meta['fit_date_weights_summary'], binding['preparation']['train_sample_weight_summary'])
    fitted_capacity = helper.check_model(model, state)
    raw = helper.predict_raw(model, state, Xq)
    native = helper.inverse_transform(raw, Xq)
    loaded_state = read(model_dir / 'feature_state.json')
    assert loaded_state == state
    loaded = lgb.Booster(model_file=str(model_dir / 'model.txt'))
    loaded_capacity = helper.check_model(loaded, loaded_state)
    assert loaded_capacity == fitted_capacity
    reloaded_raw = helper.predict_raw(loaded, loaded_state, Xq)
    reloaded_native = helper.inverse_transform(reloaded_raw, Xq)
    exact(raw, reloaded_raw, 'All query unit raw LightGBM reload')
    exact(native, reloaded_native, 'All query TWD LightGBM reload')
    assert raw.dtype == native.dtype == np.float64 and native.shape == raw.shape == (len(Xq),)
    assert np.isfinite(raw).all() and np.isfinite(native).all() and (native >= 0).all()
    for name, value in (('raw_z', raw), ('native_predictions', native), ('date_weights', w), ('fit_weights', product_w)):
        old.previous.save_array(model_dir / (name + '.npy'), value)
    meta.update(round=81, stage=stage, query_rows=len(Xq), prediction_units='TWD', fit_seconds=fit_seconds,
        fitted_native_capacity=fitted_capacity, loaded_native_capacity=loaded_capacity, fit_call_elapsed_seconds=fit_call_seconds,
        original_frame_sha256=frames, R63_preparation=binding['preparation'], date_weight_provenance_verified_in_driver=True,
        fit_indices_content_sha256=spec['fit_indices_content_sha256'], query_indices_content_sha256=spec['query_indices_content_sha256'],
        reload_all_query_rows_verified=True, reload_raw_exact=True, reload_TWD_exact=True, reload_rows=len(Xq),
        native_negative_raw_rows=int((raw < 0).sum()), native_zero_TWD_rows=int((native == 0).sum()),
        plan_sha256=plan_sha, input_sha256=dict(inputs), lightgbm_version=lgb.__version__)
    meta['artifact_sha256'] = {rel(p): sha(p) for p in model_dir.iterdir() if p.is_file()}
    write(model_dir / 'metadata.json', meta)
    return native, fit_seconds


def run(expected_plan_sha):
    tick = time.monotonic()
    assert expected_plan_sha == PLAN_SHA == sha(DOCUMENT)
    assert sha(Path(helper.__file__)) == HELPER_SHA
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
    def small(name, digest, receipt=True):
        path = checked(name, digest)
        value = read(path) if receipt else json.loads(path.read_text(encoding='utf8'))
        if receipt:
            checked(rel(path.with_suffix('.sha256')), sha(path.with_suffix('.sha256')))
        return value
    def artifact(record, name):
        mapping = {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}
        return checked(name, mapping[name])
    checked(rel(DOCUMENT), PLAN_SHA)
    r79 = small('outputs/round79/results.json', R79_SHA)
    r80 = small('outputs/round80/results.json', R80_SHA)
    p79 = small('outputs/round79/plan.json', r79['plan_sha256'])
    for number, digest in ((79, '8b0132b6ad0b374488f02957661fdc1b46ac69c381c0e71717498c02e18f986d'),
                           (80, '973fd036c1b9a107a067918be4686a3deb9d7399292974ce35b74f295723b577')):
        execution = small(f'outputs/round{number}/execution.json', digest)
        assert execution['exit_code'] == 0 and execution['failure'] is None
    assert all(r['status'] == 'three_periods_completed_no_adoption_decision' and r['checked_stages'] == list(STAGES) for r in (r79, r80))
    assert r79['source_sha256'] == p79['source_sha256'] and r79['input_sha256'] == p79['input_sha256']
    p48, p40, r63 = (small(name, p79['input_sha256'][name]) for name in ('outputs/round48/plan.json', 'outputs/round40/plan.json', 'outputs/round63/results.json'))
    assert p79['split_contract'] == p48['split_contract'] == p40['split_contract']
    assert p79['splits'] == p48['splits'] == p40['splits'] and p79['target_group_weights'] == p48['target_group_weights'] == p40['target_group_weights']
    assert p40['spatial_parameters'] == spatial.parameters() and r63['checked_stages'] == list(STAGES)
    source = old.previous.merge(r79['source_sha256'], r80['source_sha256'], helper.source_hashes(), {rel(Path(__file__)): sha(Path(__file__))})
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    config_name = 'runs/v1/models/final_config.json'
    config = small(config_name, p79['input_sha256'][config_name], receipt=False)
    chosen = next(v for v in config['components'] if v['name'] == 'lightgbm_l1_unit63')
    assert helper.parameters() == {**chosen['parameters'], 'n_estimators': chosen['iterations'], 'n_jobs': 4}
    assert chosen['iterations'] == 3999 and len(config['feature_columns']) == 87 and len(config['categorical_columns']) == 21
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 24
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    selector_path = ROOT / 'outputs/current_private_candidate.json'
    assert sha(selector_path) == SELECTOR_SHA
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT):
        assert p79['input_sha256'][name] == p48['input_sha256'][name]
        checked(name, p79['input_sha256'][name])
    bindings = {}
    for s in STAGES:
        a, inherited, a63, a80 = r79['stages'][s], p79['bindings'][s], r63['stages'][s], r80['stages'][s]
        b, spec = inherited['unit'], p79['split_contract'][s]
        assert a['status'] == a63['status'] == a80['status'] == 'completed' and a['full_query_raw_TWD_reload_exact']
        for key in ('fit_indices', 'query_indices', 'old_unit', 'reference', 'state', 'state_summary', 'query_geo'):
            checked(b[key], p79['input_sha256'][b[key]])
        receipt = rel((ROOT / b['state_summary']).with_suffix('.sha256'))
        checked(receipt, p79['input_sha256'][receipt])
        paths = {key: inherited['paths'][key] for key in ('nn', 'log', *BACKGROUND)}
        for path in paths.values():
            checked(path, p79['input_sha256'][path])
        for key, suffix in (('native', 'model/native_predictions.npy'), ('v22', 'predictions.npy'), ('base', 'base.npy'),
                            ('reference', 'reference.npy'), ('spatial', 'unit_spatial.npy'), ('fit', 'fit_indices.npy'), ('query', 'query_indices.npy')):
            paths[key] = f'outputs/round79/{s}/{suffix}'
            artifact(a, paths[key])
        paths['v23'] = f'outputs/round80/{s}/predictions.npy'
        artifact(a80, paths['v23'])
        assert inputs[paths['v22']] == a['prediction_sha256'] and inputs[paths['v23']] == a80['prediction_sha256']
        for kind in ('fit', 'query'):
            assert inputs[paths[kind]] == inputs[b[kind + '_indices']]
            assert a[kind + '_rows'] == a63[kind + '_rows'] == a80[kind + '_rows'] == spec[kind + '_rows']
            for record, number in ((a63, 63), (a80, 80)):
                path = f'outputs/round{number}/{s}/{kind}_indices.npy'
                artifact(record, path)
                assert inputs[path] == inputs[paths[kind]]
        prepared, pp, tp = (f'outputs/round63/{s}/{suffix}' for suffix in ('prepared.npz', 'preparation.json', 'model/training.json'))
        for path in (prepared, pp, tp):
            artifact(a63, path)
        prep, training = small(pp, inputs[pp]), small(tp, inputs[tp], receipt=False)
        assert prep == a63['preparation'] and prep['prepared_path'] == prepared and prep['prepared_sha256'] == inputs[prepared]
        assert training['status'] == 'completed' and training['input_sha256'] == inputs[prepared]
        assert training['train_sample_weight_sha256'] == prep['train_sample_weight_sha256']
        assert prep['weights_use_fit_X_only'] and not prep['weights_use_targets'] and not prep['weights_use_query_dates']
        assert not prep['query_labels_used'] and not prep['float32_weights_renormalized'] and prep['neutral_weights_exactly_one']
        assert not prep['encoder_refit'] and not prep['prior_refit'] and prep['all_eight_R60_arrays_reused_without_changes']
        mp = f'outputs/round79/{s}/model/metadata.json'
        artifact(a, mp)
        meta = small(mp, inputs[mp])
        assert meta['prediction_units'] == 'TWD' and meta['query_rows'] == meta['reload_rows'] == spec['query_rows']
        assert meta['reload_all_query_rows_verified'] and meta['reload_raw_exact'] and meta['reload_TWD_exact']
        assert meta['artifact_sha256'][paths['native']] == inputs[paths['native']] and meta['original_frame_sha256'] == a['original_frame_sha256']
        bindings[s] = dict(unit=b, paths=paths, prepared=prepared, preparation=prep, R63_training=training, R79_metadata=meta)
    plan = dict(round=81, candidate='R81', primary_control='v22', secondary_control='v23', text_plan_sha256=PLAN_SHA,
        source_sha256=source, input_sha256=dict(inputs), bindings=bindings, formula=FORMULA, parameters=helper.parameters(),
        stage_order=list(STAGES), split_contract=p79['split_contract'], splits=p79['splits'], target_group_weights=p79['target_group_weights'],
        protected_before=protected, selector_sha256_before=SELECTOR_SHA, planned_new_fits=3, new_NN_fits=0, new_CatBoost_fits=0,
        all_periods_regardless_of_scores=True, saved_date_weights_recomputed=False, query_labels_used_only_for_evaluation=True,
        Public_feedback_used_for_tuning=False, new_CSV=False, new_test_predictions=False, automatic_adoption=False, fresh_blind_validation=False)
    write(OUT / 'plan.json', plan)
    schema = json.loads((ROOT / 'reports/feature_schema.json').read_text(encoding='utf8'))
    assert schema['columns'] == config['feature_columns'] and schema['categorical_columns'] == config['categorical_columns']
    cats = schema['categorical_columns']
    Xall, yall = old.previous.load_training_rows(p48)
    assert len(Xall) == 455255 and list(Xall.columns) == config['feature_columns']
    result = {k: plan[k] for k in ('round', 'candidate', 'primary_control', 'secondary_control', 'formula', 'text_plan_sha256',
        'source_sha256', 'input_sha256', 'protected_before', 'selector_sha256_before', 'new_NN_fits', 'new_CatBoost_fits',
        'query_labels_used_only_for_evaluation', 'Public_feedback_used_for_tuning', 'new_CSV', 'new_test_predictions', 'automatic_adoption', 'fresh_blind_validation')}
    result.update(status='running', checked_stages=[], stages={}, plan_sha256=sha(OUT / 'plan.json'), new_fits=0,
                  new_submission_generated=False, runtime={'lightgbm': lgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    for s in STAGES:
        start = time.monotonic()
        folder = OUT / s
        folder.mkdir()
        a, binding, spec = r79['stages'][s], bindings[s], p79['split_contract'][s]
        b, paths, indices = binding['unit'], binding['paths'], {}
        for kind in ('fit', 'query'):
            v = np.load(ROOT / paths[kind], allow_pickle=False)
            assert v.dtype == np.int64 and v.shape == (spec[kind + '_rows'],)
            assert (np.diff(v) > 0).all() and v.min() >= 0 and v.max() < len(Xall)
            assert old.original.value_sha(v, '<i8') == spec[kind + '_indices_content_sha256']
            indices[kind] = v
            old.previous.save_array(folder / (kind + '_indices.npy'), v)
        fi, qi = indices['fit'], indices['query']
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        yfit, n = yall[fi].copy(), len(qi)
        frames = {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert frames == a['original_frame_sha256'] == r80['stages'][s]['original_frame_sha256']
        assert all(r63['stages'][s]['original_frame_sha256'][k] == v for k, v in frames.items())
        assert helper.values_sha256(yfit) == binding['R79_metadata']['fit_target_values_sha256']
        w = date_weights(binding, X, yfit, n)
        geo = pd.read_pickle(ROOT / b['query_geo'])  # R39 geo is ONLY for spatial prediction, never the LGB model.
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
        assert helper.values_sha256(reference['reference_residual']) == summary['existing_reference_residual_sha256']
        vector = lambda name: unit_run.vector(ROOT / name, n)
        fixed = {key: vector(paths[key]) for key in ('native', 'nn', 'log', *CONTROLS)}
        native, legacy_unit = fixed['native'], vector(b['old_unit'])  # Both already TWD; zero is allowed.
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p79['splits'][s][0], p79['target_group_weights'])
        metrics = {key: score(fixed[key]) for key in CONTROLS}
        assert metrics['v22'] == a['metrics']['R79'] and metrics['v23'] == r80['stages'][s]['metrics']['R80']
        assert all(metrics[key] == a['metrics'][key] for key in BACKGROUND)
        assert score(native) == a['unit_branch_metrics']['native']
        branch_values, branch_metrics, reports = {}, {}, {}
        for mode in ('control', 'candidate'):
            if mode == 'candidate':
                assert all(sha(ROOT / name) == digest for name, digest in source.items())
                new_unit, fit_seconds = fit_native(folder, s, X, yfit, Xq, cats, w, binding, frames, spec, result['plan_sha256'], inputs)
            base = .75 * native + .25 * (legacy_unit if mode == 'control' else new_unit)
            if mode == 'control':
                exact(base, vector(paths['base']), 'R79 base exact')
            p, diag, rr = old.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
            branch, detail, sr = spatial.predict(state, Xq, geo, base, p, diag)
            fallback = ~detail['applied'].to_numpy(dtype=bool)
            exact(branch[fallback], p[fallback], mode + ' spatial fallback')
            branch_metrics[mode] = {key: score(value) for key, value in (('base', base), ('reference', p), ('spatial', branch))}
            if mode == 'control':
                exact(p, vector(paths['reference']), 'R79 reference exact')
                exact(branch, vector(paths['spatial']), 'R79 spatial exact')
                exact(np.median(np.column_stack([fixed['nn'], fixed['log'], branch]), axis=1), fixed['v22'], 'R79 median exact')
                assert all(branch_metrics[mode][layer] == a['unit_branch_metrics'][layer] for layer in ('base', 'reference', 'spatial'))
            branch_values[mode], reports[mode] = branch, {'reference': rr, 'spatial': sr}
            for label, value in (('base', base), ('reference', p), ('unit_spatial', branch)):
                old.previous.save_array(folder / (mode + '_' + label + '.npy'), value)
            if mode == 'candidate':
                for label, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
                    with (folder / (label + '.pkl')).open('xb') as f:
                        pickle.dump(value, f, protocol=5)
        candidate = np.median(np.column_stack([fixed['nn'], fixed['log'], branch_values['candidate']]), axis=1)
        assert candidate.dtype == np.float64 and candidate.shape == (n,) and np.isfinite(candidate).all() and (candidate > 0).all()
        metrics['R81'] = score(candidate)
        old.previous.save_array(folder / 'predictions.npy', candidate)
        exact(candidate, vector(rel(folder / 'predictions.npy')), 'Saved R81 vector exact')
        assert frames == {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        diff = old.previous.prior.log_run.differences
        old_native_metrics, new_native_metrics = score(legacy_unit), score(new_unit)
        item = dict(stage=s, status='completed', fit_rows=len(fi), query_rows=n, dates=p79['splits'][s], original_frame_sha256=frames,
            metrics=metrics, differences={label: diff(metrics[label], metrics['R81']) for label in CONTROLS},
            unit_branch_metrics=branch_metrics, unit_branch_differences={layer: diff(branch_metrics['control'][layer], branch_metrics['candidate'][layer]) for layer in ('base', 'reference', 'spatial')},
            original_LGB_native_metrics=old_native_metrics, new_LGB_native_metrics=new_native_metrics,
            native_LGB_difference=diff(old_native_metrics, new_native_metrics), pipeline_reports=reports,
            R79_base_reference_spatial_full_vectors_exact=True, v22_complete_combination_exact=True, controls_full_metrics_exact=True,
            changed_unit_rows=int(np.count_nonzero(branch_values['candidate'] != branch_values['control'])),
            changed_final_rows=int(np.count_nonzero(candidate != fixed['v22'])), new_fits=1, new_NN_fits=0, new_CatBoost_fits=0,
            full_query_raw_TWD_reload_exact=True, date_weight_provenance_verified=True, fit_seconds=fit_seconds,
            reference_predict_calls=2, spatial_predict_calls=2, query_labels_used_only_for_evaluation=True,
            prediction_sha256=sha(folder / 'predictions.npy'), elapsed_seconds=time.monotonic() - start,
            artifact_sha256={rel(p): sha(p) for p in folder.rglob('*') if p.is_file()})
        write(folder / 'results.json', item)
        result['stages'][s], result['new_fits'] = item, result['new_fits'] + 1
        result['checked_stages'].append(s)
        print(json.dumps({'stage': s, 'fit_seconds': fit_seconds, 'R81_vs_v22': item['differences']['v22']['overall'], 'LGB_native': item['native_LGB_difference']['overall']}), flush=True)
    assert result['new_fits'] == 3 and result['checked_stages'] == list(STAGES)
    assert all(sha(ROOT / name) == digest for name, digest in source.items()) and all(sha(ROOT / name) == digest for name, digest in inputs.items())
    after = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    after['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    assert after == protected and sha(selector_path) == SELECTOR_SHA and sha(DOCUMENT) == PLAN_SHA
    result.update(status='three_periods_completed_no_adoption_decision', elapsed_seconds=time.monotonic() - tick,
        reference_predict_calls=6, spatial_predict_calls=6, protected_after=after, selector_sha256_after=sha(selector_path))
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'result_sha256': sha(OUT / 'results.json'), 'elapsed_seconds': result['elapsed_seconds'], 'new_fits': 3, 'new_CSV': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert args.plan_sha == PLAN_SHA == sha(DOCUMENT) and not OUT.exists(), 'Plan changed or previous attempt exists'
    OUT.mkdir()
    tick, started, code, failure = time.monotonic(), old.now(), 0, None
    write(OUT / 'started.json', {'command': sys.argv, 'started_at_utc': started, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        run(args.plan_sha)
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': code, 'failure': failure,
        'started_at_utc': started, 'completed_at_utc': old.now(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
