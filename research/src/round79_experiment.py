"""R79: saved fit-date weights times R19 area weights; three fixed CatBoost fits."""
from pathlib import Path
import argparse
import contextlib
import json
import pickle
import sys
import time
import traceback
import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round40_three_branch as unit_run
import round48_nn_experiment as old
import round79_recency_unit as helper

ROOT, STAGES = old.ROOT, old.STAGES
OUT = ROOT / 'outputs/round79'
DOCUMENT = ROOT / 'reports/round79_plan.md'
PLAN_SHA = '4992740d273f3d0a4655037519fedd0e94c3775fcbf30216c6c3f4afbda079a9'
HELPER_SHA = '9a124322a139e5621c28d26a6e7ab95385c73507f017a2765d9eaaa0eef39cd2'
DATE_SOURCE = 'src/round60_recency_price_prior.py'
DATE_SHA = '4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f'
sha, read, write, spatial, rel = old.sha, old.locked, old.write, old.spatial, old.rel
CONTROLS = (('v20', 76, 'R76'), ('v21', 77, 'R77'), ('v19', 71, 'R71'),
            ('v18', 66, 'R66'), ('v11', 49, 'R49'), ('v12', 50, 'R50'))
FORMULA = 'median(R71_seed42_nn_spatial,R46_recency_log_spatial,spatial(reference(.75*R79_native_TWD+.25*old_unit_LGB_TWD)))'
NPZ_KEYS = {'train_num', 'query_num', 'train_cat', 'query_cat', 'train_y_million',
            'cardinalities', 'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'}


def summary_equal(actual, saved):
    for key in ('rows', 'min', 'max'):
        assert actual[key] == saved[key], key
    for key in ('mean', 'sum', 'sum_squares', 'effective_sample_size'):
        assert np.isclose(actual[key], saved[key], rtol=1e-12, atol=1e-12), key


def raw_predict(model, state, X, geo, cats):
    helper.check_state(state, X, cats)
    helper.check_model(model, state)
    frame = helper.augment_features(X, geo)
    z = np.asarray(model.predict(Pool(frame, cat_features=state['categorical_columns'], thread_count=4),
                                 thread_count=4), dtype=np.float64)
    assert z.shape == (len(X),) and np.isfinite(z).all()
    return z


def run(expected_plan_sha):
    started = time.monotonic()
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

    def small(name, digest=None, receipt=True):
        path = checked(name, digest or sha(ROOT / name))
        value = read(path) if receipt else json.loads(path.read_text(encoding='utf8'))
        if receipt:
            checked(rel(path.with_suffix('.sha256')), sha(path.with_suffix('.sha256')))
        return value

    def artifact(record, name):
        mapping = {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}
        return checked(name, mapping[name])

    checked(rel(DOCUMENT), PLAN_SHA)
    p19 = small('outputs/round19/candidate_plan.json', unit_run.AUTHORITIES['outputs/round19/candidate_plan.json'])
    p40, p48, p63 = (small(f'outputs/round{n}/plan.json') for n in (40, 48, 63))
    records = {n: small(f'outputs/round{n}/results.json') for n in (40, 49, 50, 63, 66, 71, 76, 77)}
    assert all(r['checked_stages'] == list(STAGES) for r in records.values())
    assert records[71]['primary_seed'] == 42
    assert p19['split_contract'] == p40['split_contract'] == p48['split_contract'] == p63['split_contract']
    assert p19['splits'] == p40['splits'] == p48['splits']
    assert p19['target_group_weights'] == p40['target_group_weights'] == p48['target_group_weights']
    assert p40['spatial_parameters'] == spatial.parameters()
    assert unit_run.sources() == p40['source_sha256']
    source = old.previous.merge(p40['source_sha256'], records[71]['source_sha256'],
        records[63]['source_sha256'], records[77]['source_sha256'], helper.source_hashes(),
        {rel(Path(__file__)): sha(Path(__file__)), rel(Path(old.__file__)): sha(Path(old.__file__)), DATE_SOURCE: DATE_SHA})
    for name, digest in source.items():
        checked(name, digest)
    config_path = 'runs/v1/models/final_config.json'
    checked(config_path, p19['dependency_sha256'][config_path])
    assert helper.parameters() == p19['parameters'] == p19['baseline_parameters']
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 22
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    selector_path = ROOT / 'outputs/current_private_candidate.json'
    selector_sha = sha(selector_path)

    # Freeze the files used by all three periods before any new fit.
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT):
        assert p40['input_sha256'][name] == p48['input_sha256'][name]
        checked(name, p48['input_sha256'][name])
    bindings = {}
    for s in STAGES:
        b = p40['bindings'][s]
        for key in ('fit_indices', 'query_indices', 'unit_component', 'unit_original_blend', 'old_raw', 'old_unit',
                    'reference', 'state', 'state_summary', 'query_geo'):
            checked(b[key], p40['input_sha256'][b[key]])
        checked(rel((ROOT / b['state_summary']).with_suffix('.sha256')),
                p40['input_sha256'][rel((ROOT / b['state_summary']).with_suffix('.sha256'))])
        e19 = small(b['R19_result'], p40['input_sha256'][b['R19_result']])
        model_folder = f'outputs/round19/models/{s}'
        meta_path, state_path = model_folder + '/metadata.json', model_folder + '/feature_state.json'
        m19 = small(meta_path, e19['input_artifact_sha256'][meta_path])
        legacy = small(state_path, m19['artifact_sha256']['feature_state.json'], receipt=False)
        assert e19['input_artifact_sha256'][state_path] == inputs[state_path]
        assert m19['parameters'] == helper.parameters() and m19['prediction_units'] == 'TWD'
        assert m19['sample_weight'] == 'effective_area / 100' and m19['feature_state_content_sha256'] == legacy['state_content_sha256']
        tree_geo = f'outputs/round6/geography/{s}/train.pkl'
        tree_manifest = f'outputs/round6/geography/{s}/manifest.json'
        checked(tree_geo, p19['stage_dependency_sha256'][s][tree_geo])
        gm = small(tree_manifest, p19['stage_dependency_sha256'][s][tree_manifest])
        assert inputs[tree_manifest] == m19['geo_manifest_sha256'] == b['manifest']['original_manifest_sha256']
        assert gm['stage'] == s and not gm['geographic_values_learned_from_targets']
        a63 = records[63]['stages'][s]
        prepared = f'outputs/round63/{s}/prepared.npz'
        prep_path = f'outputs/round63/{s}/preparation.json'
        artifact(a63, prepared); artifact(a63, prep_path)
        prep = small(prep_path, inputs[prep_path])
        assert prep == a63['preparation'] and prep['prepared_path'] == prepared and prep['prepared_sha256'] == inputs[prepared]
        training_path = f'outputs/round63/{s}/model/training.json'
        artifact(a63, training_path)
        training = small(training_path, inputs[training_path], receipt=False)
        assert training['status'] == 'completed' and training['input_sha256'] == inputs[prepared]
        assert training['train_sample_weight_sha256'] == prep['train_sample_weight_sha256']
        assert prep['weights_use_fit_X_only'] and not prep['weights_use_targets'] and not prep['weights_use_query_dates']
        assert not prep['query_labels_used'] and not prep['float32_weights_renormalized'] and prep['neutral_weights_exactly_one']
        assert not prep['encoder_refit'] and not prep['prior_refit'] and prep['all_eight_R60_arrays_reused_without_changes']
        paths = {'nn': f'outputs/round71/{s}/seed42/nn_spatial.npy',
                 'weighted_nn': f'outputs/round74/{s}/nn_spatial.npy',
                 'log': f'outputs/round46/{s}/recency_log_spatial.npy',
                 'original_unit_spatial': f'outputs/round40/{s}/unit_spatial.npy',
                 'v20_unit_spatial': f'outputs/round76/{s}/candidate_unit_spatial.npy',
                 'v5': f'outputs/round36/{s}/log_spatial_combination.npy'}
        artifact(records[71]['stages'][s], paths['nn'])
        artifact(records[40]['stages'][s], paths['original_unit_spatial'])
        artifact(records[76]['stages'][s], paths['v20_unit_spatial'])
        for key in ('weighted_nn', 'log', 'v5'):
            checked(paths[key], records[77]['input_sha256'][paths[key]])
        for label, number, _ in CONTROLS:
            paths[label] = f'outputs/round{number}/{s}/predictions.npy'
            checked(paths[label], records[number]['stages'][s]['prediction_sha256'])
        for kind in ('fit', 'query'):
            original_path = f'{model_folder}/{kind}_indices.npy'
            checked(original_path, m19['artifact_sha256'][kind + '_indices.npy'])
            assert inputs[original_path] == inputs[b[kind + '_indices']]
            for number in (63, 71, 76, 77):
                path = f'outputs/round{number}/{s}/{kind}_indices.npy'
                artifact(records[number]['stages'][s], path)
                assert inputs[path] == inputs[original_path]
            assert m19[kind + '_indices_content_sha256'] == p48['split_contract'][s][kind + '_indices_content_sha256']
        bindings[s] = {'unit': b, 'tree_geo': tree_geo, 'tree_manifest': tree_manifest, 'paths': paths,
            'prepared': prepared, 'preparation': prep, 'R63_training': training,
            'R19_metadata': m19, 'R19_state': legacy, 'R19_metrics': e19['metrics']}
    write(OUT / 'plan.json', {'round': 79, 'candidate': 'R79', 'text_plan_sha256': PLAN_SHA,
        'source_sha256': source, 'input_sha256': dict(inputs), 'bindings': bindings,
        'split_contract': p48['split_contract'], 'splits': p48['splits'], 'target_group_weights': p48['target_group_weights'],
        'stage_order': list(STAGES), 'primary_control': 'v20=R76', 'formula': FORMULA,
        'parameters': helper.parameters(), 'sample_weight': helper.WEIGHT_FORMULA,
        'planned_new_fits': 3, 'new_NN_fits': 0, 'all_periods_regardless_of_scores': True,
        'saved_date_weights_recomputed': False, 'new_reference_or_spatial_states': False,
        'protected_before': protected, 'selector_sha256_before': selector_sha,
        'new_CSV': False, 'automatic_adoption': False, 'fresh_blind_validation': False})
    schema = json.loads((ROOT / 'reports/feature_schema.json').read_text(encoding='utf8'))
    cats = schema['categorical_columns']
    Xall, yall = old.previous.load_training_rows(p48)
    assert len(Xall) == 455255
    result = {'round': 79, 'candidate': 'R79', 'status': 'running', 'primary_control': 'v20',
        'formula': FORMULA, 'checked_stages': [], 'stages': {}, 'source_sha256': source, 'input_sha256': inputs,
        'plan_sha256': sha(OUT / 'plan.json'), 'text_plan_sha256': PLAN_SHA,
        'new_fits': 0, 'new_NN_fits': 0, 'new_submission_generated': False, 'fresh_blind_validation': False,
        'protected_before': protected, 'selector_sha256_before': selector_sha,
        'runtime': {'catboost': catboost.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    for s in STAGES:
        tick = time.monotonic()
        folder = OUT / s
        folder.mkdir()
        binding = bindings[s]
        b, paths, m19, legacy, prep = (binding[k] for k in ('unit', 'paths', 'R19_metadata', 'R19_state', 'preparation'))
        spec, ind = p48['split_contract'][s], {}
        for kind in ('fit', 'query'):
            v = np.load(ROOT / b[kind + '_indices'], allow_pickle=False)
            assert v.dtype == np.int64 and v.shape == (spec[kind + '_rows'],)
            assert (np.diff(v) > 0).all() and v.min() >= 0 and v.max() < len(Xall)
            assert old.original.value_sha(v, '<i8') == spec[kind + '_indices_content_sha256']
            ind[kind] = v
            old.previous.save_array(folder / (kind + '_indices.npy'), v)
        fi, qi = ind['fit'], ind['query']
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        yfit, n = yall[fi].copy(), len(qi)
        frames = {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        for key, digest in frames.items():
            assert m19[key + '_sha256'] == digest
            assert all(records[r]['stages'][s]['original_frame_sha256'][key] == digest for r in (63, 71, 76, 77))
        original_geo = pd.read_pickle(ROOT / binding['tree_geo'])
        assert len(original_geo) == len(Xall) and original_geo.index.equals(Xall.index)
        G, Gq = original_geo.iloc[fi].reset_index(drop=True), original_geo.iloc[qi].reset_index(drop=True)
        del original_geo
        assert spatial.frame_sha(G) == m19['fit_geo_sha256'] and spatial.frame_sha(Gq) == m19['query_geo_sha256']
        assert helper.fit_feature_state(X, G, cats) == legacy, 'Original R19 fit-only state differs'
        assert helper.values_sha256(yfit) == m19['fit_target_values_sha256']
        with np.load(ROOT / binding['prepared'], allow_pickle=False) as z:
            assert set(z.files) == NPZ_KEYS
            weights32 = z['train_sample_weight'].copy()
            assert z['train_y_million'].dtype == np.float32 and z['train_y_million'].shape == (len(fi),)
            assert z['train_y_million'].tobytes() == np.asarray(yfit / 1e6, dtype=np.float32).tobytes()
        assert weights32.dtype == np.float32 and weights32.shape == (len(fi),) and np.isfinite(weights32).all() and (weights32 > 0).all()
        assert old.original.value_sha(weights32, '<f4') == prep['train_sample_weight_sha256']
        summary_equal(helper._summary(weights32), prep['train_sample_weight_summary'])
        summary_equal(helper._summary(weights32), binding['R63_training']['train_sample_weight_summary'])
        assert prep['fit_rows'] == len(fi) and prep['query_rows'] == n
        dates = prep['date_weight_summary_float64']
        d, invalid, outside = (X[k].to_numpy(dtype=np.float64) for k in ('trade_day_index', 'trade_date_invalid', 'trade_date_out_of_range'))
        for values, key in ((d, 'date_values_sha256'), (invalid, 'invalid_flags_sha256'), (outside, 'out_of_range_flags_sha256')):
            assert helper.values_sha256(values) == dates[key]
        valid = np.isfinite(d) & (invalid == 0) & (outside == 0)
        assert dates['reference_rows'] == len(fi) and dates['valid_date_rows'] == int(valid.sum())
        assert dates['neutral_date_rows'] == int((~valid).sum()) and (weights32[~valid] == np.float32(1)).all()
        assert dates['weight_rules']['half_life_days'] == 365 and not dates['weight_rules']['weights_use_query'] and not dates['weight_rules']['weights_use_targets']

        geo = pd.read_pickle(ROOT / b['query_geo'])  # R39 geography is spatial-only.
        unit_run.previous.geo_encoder.check_geo(geo)
        assert geo.index.equals(Xq.index) and spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        with (ROOT / b['state']).open('rb') as f:
            spatial_state = pickle.load(f)
        spatial_summary = read(ROOT / b['state_summary'])
        assert spatial_state['state_summary'] == spatial_summary and spatial_state['parameters'] == p40['spatial_parameters']
        assert spatial_summary['fit_X_sha256'] == frames['fit_X'] and not spatial_summary['query_used_in_build']
        assert spatial_summary['fit_geo_sha256'] == b['manifest']['features']['fit']['frame_sha256']
        with (ROOT / b['reference']).open('rb') as f:
            reference = pickle.load(f)
        assert reference['fit_report']['fitting_rows'] == len(fi)
        assert helper.values_sha256(reference['reference_residual']) == spatial_summary['existing_reference_residual_sha256']
        vector = lambda name: unit_run.vector(ROOT / name, n)  # Legacy raw and native may be zero.
        fixed = {key: vector(path) for key, path in paths.items()}
        old_unit, original_native = vector(b['old_unit']), vector(b['unit_component'])
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p48['splits'][s][0], p48['target_group_weights'])
        metrics = {label: score(fixed[label]) for label, _, _ in CONTROLS}
        metrics['v5'] = score(fixed['v5'])
        for label, number, key in CONTROLS:
            assert metrics[label] == records[number]['stages'][s]['metrics'][key], label
        assert metrics['v5'] == records[50]['stages'][s]['metrics']['v5']
        for label, nn, ub in (('v20', fixed['nn'], fixed['v20_unit_spatial']),
                              ('v21', fixed['weighted_nn'], fixed['v20_unit_spatial']),
                              ('v19', fixed['nn'], fixed['original_unit_spatial'])):
            old.exact(np.median(np.column_stack([nn, fixed['log'], ub]), axis=1), fixed[label], label + ' median replay')
        # Reproduce the primary unit branch with the unchanged state and its own diagnostics.
        control_base = .75 * original_native + .25 * old_unit
        control_ref, control_diag, _ = old.original.predict_reference(reference, Xq, control_base, variant='trend', return_diagnostics=True)
        control_spatial, _, _ = spatial.predict(spatial_state, Xq, geo, control_base, control_ref, control_diag)
        old.exact(control_spatial, fixed['v20_unit_spatial'], 'R76 unit branch replay')
        for layer, value in (('base', control_base), ('reference', control_ref), ('spatial', control_spatial)):
            assert score(value) == records[76]['stages'][s]['unit_branch_metrics']['candidate'][layer]

        assert all(sha(ROOT / name) == digest for name, digest in source.items())
        model_dir = folder / 'model'
        model_dir.mkdir()
        fit_start, fit_time = old.now(), time.monotonic()
        write(model_dir / 'fit_started.json', {'started_at_utc': fit_start, 'helper_sha256': HELPER_SHA,
            'parameters': helper.parameters(), 'prepared_path': binding['prepared'], 'prepared_sha256': inputs[binding['prepared']],
            'date_weight_sha256': prep['train_sample_weight_sha256'], 'query_labels_used': False, 'automatic_retry': False})
        fit_error = None
        try:
            with (model_dir / 'fit.log').open('x', encoding='utf8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model, state, metadata = helper.fit_model(X, yfit, G, cats, weights32)
        except Exception:
            fit_error = traceback.format_exc()
            raise
        finally:
            fit_seconds = time.monotonic() - fit_time
            write(model_dir / 'execution.json', {'exit_code': 1 if fit_error else 0, 'failure': fit_error,
                'started_at_utc': fit_start, 'completed_at_utc': old.now(), 'elapsed_seconds': fit_seconds, 'automatic_retry': False})
        assert state == legacy and metadata['feature_state_content_sha256'] == legacy['state_content_sha256']
        area_weights = helper.original.sample_weights(X)
        actual_weights = area_weights * weights32.astype(np.float64)
        assert metadata['fit_date_weights_sha256'] == prep['train_sample_weight_sha256']
        assert metadata['fit_area_weights_sha256'] == legacy['fit_sample_weights_sha256']
        assert metadata['fit_sample_weights_sha256'] == helper.values_sha256(actual_weights)
        summary_equal(metadata['fit_date_weights_summary'], prep['train_sample_weight_summary'])
        # The legacy state's weight field deliberately remains AREA ONLY.
        fitted_capacity = helper.check_model(model, state)
        raw = raw_predict(model, state, Xq, Gq, cats)
        native = helper.inverse_transform(raw, Xq)
        assert native.dtype == np.float64 and np.isfinite(native).all() and (native >= 0).all()
        model.save_model(str(model_dir / 'model.cbm'), format='cbm')
        write(model_dir / 'feature_state.json', state)
        loaded_state = read(model_dir / 'feature_state.json')
        assert loaded_state == state
        loaded = CatBoostRegressor()
        loaded.load_model(str(model_dir / 'model.cbm'), format='cbm')
        loaded_capacity = helper.check_model(loaded, loaded_state)
        for key in fitted_capacity:
            if key != 'actual_learning_rate':
                assert loaded_capacity[key] == fitted_capacity[key], key
        reloaded_raw = raw_predict(loaded, loaded_state, Xq, Gq, cats)
        reloaded_native = helper.inverse_transform(reloaded_raw, Xq)
        old.exact(raw, reloaded_raw, 'All query raw unit CBM reload')
        old.exact(native, reloaded_native, 'All query TWD CBM reload')
        for name, value in (('raw_z', raw), ('native_predictions', native), ('date_weights', weights32), ('pool_weights', actual_weights)):
            old.previous.save_array(model_dir / (name + '.npy'), value)
        del model, loaded
        metadata.update(round=79, stage=s, query_rows=n, prediction_units='TWD',
            fit_seconds=fit_seconds, fitted_native_capacity=fitted_capacity, loaded_native_capacity=loaded_capacity,
            fitted_actual_learning_rate=fitted_capacity['actual_learning_rate'], **loaded_capacity,
            learning_rate_tolerance={'reference': .065, 'rtol': 1e-7, 'atol': 0, 'fit_reload_exact_required': False},
            original_R19_feature_state_exact=True, original_frame_sha256=frames,
            date_weight_provenance_verified_in_driver=True, R63_preparation=prep,
            fit_indices_content_sha256=spec['fit_indices_content_sha256'], query_indices_content_sha256=spec['query_indices_content_sha256'],
            reload_all_query_rows_verified=True, reload_raw_exact=True, reload_TWD_exact=True, reload_rows=n,
            native_negative_raw_rows=int((raw < 0).sum()), native_zero_TWD_rows=int((native == 0).sum()),
            runtime=result['runtime'], plan_sha256=result['plan_sha256'], input_sha256=dict(inputs))
        metadata['artifact_sha256'] = {rel(p): sha(p) for p in model_dir.iterdir() if p.is_file()}
        write(model_dir / 'metadata.json', metadata)
        base = .75 * native + .25 * old_unit
        p, diag, rr = old.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
        branch, detail, sr = spatial.predict(spatial_state, Xq, geo, base, p, diag)
        fallback = ~detail['applied'].to_numpy(dtype=bool)
        old.exact(branch[fallback], p[fallback], 'New unit spatial fallback')
        candidate = np.median(np.column_stack([fixed['nn'], fixed['log'], branch]), axis=1)
        assert candidate.dtype == np.float64 and candidate.shape == (n,) and np.isfinite(candidate).all() and (candidate > 0).all()
        metrics['R79'] = score(candidate)
        branch_metrics = {label: score(v) for label, v in (('native', native), ('base', base), ('reference', p), ('spatial', branch))}
        original_native_metrics = score(original_native)
        for name, value in (('base', base), ('reference', p), ('unit_spatial', branch), ('predictions', candidate)):
            old.previous.save_array(folder / (name + '.npy'), value)
        for name, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
            with (folder / (name + '.pkl')).open('xb') as f:
                pickle.dump(value, f, protocol=5)
        assert frames == {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert spatial.frame_sha(G) == m19['fit_geo_sha256'] and spatial.frame_sha(Gq) == m19['query_geo_sha256']
        assert spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        differences = old.previous.prior.log_run.differences
        item = {'stage': s, 'status': 'completed', 'fit_rows': len(fi), 'query_rows': n, 'dates': p48['splits'][s],
            'original_frame_sha256': frames, 'metrics': metrics,
            'differences': {label: differences(metrics[label], metrics['R79']) for label in metrics if label != 'R79'},
            'unit_branch_metrics': branch_metrics, 'original_R19_native_metrics': original_native_metrics,
            'unit_branch_differences_vs_R76': {layer: differences(records[76]['stages'][s]['unit_branch_metrics']['candidate'][layer], branch_metrics[layer]) for layer in ('base', 'reference', 'spatial')},
            'native_difference_vs_R19': differences(original_native_metrics, branch_metrics['native']),
            'reference_report': rr, 'spatial_report': sr, 'controls_full_metrics_exact': True,
            'v19_v20_v21_full_median_exact': True, 'R76_unit_spatial_replay_exact': True,
            'R19_feature_state_unchanged': True, 'full_query_raw_TWD_reload_exact': True,
            'date_weight_provenance_verified': True, 'new_fits': 1, 'new_NN_fits': 0, 'fit_seconds': fit_seconds,
            'query_labels_used_only_for_evaluation': True, 'reference_predict_calls': 2, 'spatial_predict_calls': 2,
            'prediction_sha256': sha(folder / 'predictions.npy'), 'elapsed_seconds': time.monotonic() - tick,
            'artifact_sha256': {rel(p): sha(p) for p in folder.rglob('*') if p.is_file()}}
        write(folder / 'results.json', item)
        result['stages'][s] = item
        result['checked_stages'].append(s)
        result['new_fits'] += 1
        print(json.dumps({'stage': s, 'fit_seconds': fit_seconds, 'R79_vs_v20': item['differences']['v20']['overall'],
            'native_vs_R19': item['native_difference_vs_R19']['overall']}), flush=True)
    assert result['new_fits'] == 3 and result['checked_stages'] == list(STAGES)
    assert all(sha(ROOT / name) == digest for name, digest in source.items()) and sha(DOCUMENT) == PLAN_SHA
    result.update(status='three_periods_completed_no_adoption_decision', elapsed_seconds=time.monotonic() - started,
        protected_after={name: sha(ROOT / name) for name in protected}, selector_sha256_after=sha(selector_path))
    assert result['protected_after'] == protected and result['selector_sha256_after'] == selector_sha
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'result_sha256': sha(OUT / 'results.json'), 'new_fits': 3, 'new_CSV': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert args.plan_sha == PLAN_SHA == sha(DOCUMENT) and not OUT.exists(), 'Plan changed or previous attempt exists'
    OUT.mkdir()
    tick, start, exit_code, failure = time.monotonic(), old.now(), 0, None
    write(OUT / 'started.json', {'command': sys.argv, 'started_at_utc': start,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        run(args.plan_sha)
    except Exception:
        exit_code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': exit_code, 'failure': failure,
        'started_at_utc': start, 'completed_at_utc': old.now(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(exit_code)
