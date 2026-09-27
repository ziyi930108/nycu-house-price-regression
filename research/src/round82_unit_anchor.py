"""R82: saved R81 unit base, reference anchor instead of trend; zero fits."""
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
OUT = ROOT / 'outputs/round82'
DOCUMENT = ROOT / 'reports/round82_plan.md'
PLAN_SHA = '0a442808919c9bd15de4b40b074d890bb4d6952a7ef597f780b9071b34a3b26a'
R81_SHA = '0b5dbf01b0d22290af534406a3cf8ce59f2f1200fc9270593838ef3f24f3012c'
EXECUTION_SHA = '27b20f205d6dbe5d72b3fb81c5a7a4b96087f85f912174597db8597cdcffaa09'
SELECTOR_SHA = '9e07fee7910dbe33d94938e2f846849cb43b98eecfa084f4a7169155f2c1488c'
sha, read, write, spatial, rel = old.sha, old.locked, old.write, old.spatial, old.rel
BACKGROUND = ('v22', 'v23', 'v20', 'v21', 'v19', 'v18', 'v11', 'v12', 'v5')
FORMULA = 'median(saved_R71_seed42_nn_spatial,saved_R46_log_spatial,spatial(anchor_reference(.75*saved_R79_native_TWD+.25*saved_R81_LGB_TWD)))'


def exact(a, b, label):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def prices(value, n):
    assert value.dtype == np.float64 and value.shape == (n,) and np.isfinite(value).all() and (value >= 0).all()
    return value  # No new clamp or area conversion.


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
        checked(rel(path.with_suffix('.sha256')), sha(path.with_suffix('.sha256')))
        return value
    def artifact(record, name):
        mapping = {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}
        return checked(name, mapping[name])
    checked(rel(DOCUMENT), PLAN_SHA)
    r81 = small('outputs/round81/results.json', R81_SHA)
    p81 = small('outputs/round81/plan.json', r81['plan_sha256'])
    execution = small('outputs/round81/execution.json', EXECUTION_SHA)
    assert execution['exit_code'] == 0 and execution['failure'] is None
    assert execution['source_sha256'] == r81['source_sha256']['src/round81_experiment.py']
    assert r81['round'] == p81['round'] == 81 and r81['candidate'] == 'R81'
    assert r81['status'] == 'three_periods_completed_no_adoption_decision' and r81['new_fits'] == 3
    assert r81['checked_stages'] == p81['stage_order'] == list(STAGES)
    assert r81['new_NN_fits'] == r81['new_CatBoost_fits'] == 0
    assert r81['source_sha256'] == p81['source_sha256'] and r81['input_sha256'] == p81['input_sha256']
    p48, p40, p79 = (small(name, p81['input_sha256'][name]) for name in ('outputs/round48/plan.json', 'outputs/round40/plan.json', 'outputs/round79/plan.json'))
    assert p81['split_contract'] == p48['split_contract'] == p40['split_contract'] == p79['split_contract']
    assert p81['splits'] == p48['splits'] == p40['splits'] and p81['target_group_weights'] == p48['target_group_weights'] == p40['target_group_weights']
    assert p40['spatial_parameters'] == spatial.parameters()
    source = old.previous.merge(r81['source_sha256'], {rel(Path(__file__)): sha(Path(__file__))})
    assert all(sha(ROOT / name) == digest for name, digest in source.items())
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 24
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    selector_path = ROOT / 'outputs/current_private_candidate.json'
    assert sha(selector_path) == SELECTOR_SHA
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT):
        assert p81['input_sha256'][name] == p48['input_sha256'][name]
        checked(name, p81['input_sha256'][name])
    bindings = {}
    for stage in STAGES:
        a, inherited, spec = r81['stages'][stage], p81['bindings'][stage], p81['split_contract'][stage]
        b = inherited['unit']
        assert a['status'] == 'completed' and a['new_fits'] == 1 and a['full_query_raw_TWD_reload_exact']
        assert a['controls_full_metrics_exact'] and a['query_labels_used_only_for_evaluation']
        for key in ('fit_indices', 'query_indices', 'reference', 'state', 'state_summary', 'query_geo'):
            checked(b[key], p81['input_sha256'][b[key]])
        receipt = rel((ROOT / b['state_summary']).with_suffix('.sha256'))
        checked(receipt, p81['input_sha256'][receipt])
        paths = {key: inherited['paths'][key] for key in ('native', 'nn', 'log', *BACKGROUND)}
        for path in paths.values():
            checked(path, p81['input_sha256'][path])
        for key, suffix in (('lgb_native', 'model/native_predictions.npy'), ('R81', 'predictions.npy'),
                            ('base', 'candidate_base.npy'), ('reference', 'candidate_reference.npy'), ('spatial', 'candidate_unit_spatial.npy'),
                            ('reference_diagnostics', 'reference_diagnostics.pkl'), ('spatial_diagnostics', 'spatial_diagnostics.pkl'),
                            ('fit', 'fit_indices.npy'), ('query', 'query_indices.npy')):
            paths[key] = f'outputs/round81/{stage}/{suffix}'
            artifact(a, paths[key])
        assert inputs[paths['R81']] == a['prediction_sha256']
        for kind in ('fit', 'query'):
            assert inputs[paths[kind]] == inputs[b[kind + '_indices']] and a[kind + '_rows'] == spec[kind + '_rows']
        mp = f'outputs/round81/{stage}/model/metadata.json'
        artifact(a, mp)
        meta = small(mp, inputs[mp])
        assert meta['plan_sha256'] == r81['plan_sha256'] and meta['fit_rows'] == spec['fit_rows']
        assert meta['prediction_units'] == 'TWD' and meta['query_rows'] == meta['reload_rows'] == spec['query_rows']
        assert meta['reload_all_query_rows_verified'] and meta['reload_raw_exact'] and meta['reload_TWD_exact']
        assert meta['artifact_sha256'][paths['lgb_native']] == inputs[paths['lgb_native']]
        assert meta['original_frame_sha256'] == a['original_frame_sha256']
        mp79 = f'outputs/round79/{stage}/model/metadata.json'
        meta79 = small(mp79, p81['input_sha256'][mp79])
        assert meta79 == inherited['R79_metadata'] and meta79['prediction_units'] == 'TWD'
        assert meta79['reload_all_query_rows_verified'] and meta79['reload_raw_exact'] and meta79['reload_TWD_exact']
        assert meta79['artifact_sha256'][paths['native']] == inputs[paths['native']]
        for kind in ('fit', 'query'):
            assert meta[kind + '_indices_content_sha256'] == meta79[kind + '_indices_content_sha256'] == spec[kind + '_indices_content_sha256']
        bindings[stage] = dict(unit=b, paths=paths, metadata_paths=[mp, mp79])
    plan = dict(round=82, candidate='R82', primary_control='R81', text_plan_sha256=PLAN_SHA, formula=FORMULA,
        source_sha256=source, input_sha256=dict(inputs), bindings=bindings, stage_order=list(STAGES),
        split_contract=p81['split_contract'], splits=p81['splits'], target_group_weights=p81['target_group_weights'],
        control_reference_variant='trend', candidate_reference_variant='anchor', base_and_spatial_future_trend_unchanged=True,
        protected_before=protected, selector_sha256_before=SELECTOR_SHA, new_fits=0, new_NN_fits=0, new_CatBoost_fits=0,
        tree_NN_model_loads=0, model_predict_calls=0, new_reference_fits=0, new_spatial_state_builds=0,
        reference_predict_calls_per_stage=2, spatial_predict_calls_per_stage=2, all_periods_regardless_of_scores=True,
        query_labels_used_only_for_evaluation=True, Public_feedback_used_for_tuning=False, seed_weight_threshold_search=False,
        new_CSV=False, new_test_predictions=False, automatic_adoption=False, fresh_blind_validation=False)
    write(OUT / 'plan.json', plan)
    Xall, yall = old.previous.load_training_rows(p48)
    assert len(Xall) == 455255
    result = {k: plan[k] for k in ('round', 'candidate', 'primary_control', 'formula', 'text_plan_sha256', 'source_sha256',
        'input_sha256', 'control_reference_variant', 'candidate_reference_variant', 'base_and_spatial_future_trend_unchanged',
        'protected_before', 'selector_sha256_before', 'new_fits', 'new_NN_fits', 'new_CatBoost_fits', 'tree_NN_model_loads',
        'model_predict_calls', 'new_reference_fits', 'new_spatial_state_builds', 'query_labels_used_only_for_evaluation',
        'Public_feedback_used_for_tuning', 'seed_weight_threshold_search', 'new_CSV', 'new_test_predictions', 'automatic_adoption', 'fresh_blind_validation')}
    result.update(status='running', checked_stages=[], stages={}, plan_sha256=sha(OUT / 'plan.json'), new_submission_generated=False)
    for stage in STAGES:
        started = time.monotonic()
        folder = OUT / stage
        folder.mkdir()
        a, b, paths = r81['stages'][stage], bindings[stage]['unit'], bindings[stage]['paths']
        spec, indices = p81['split_contract'][stage], {}
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
        beta, spatial_beta = np.asarray(reference['beta'], dtype=np.float64), np.asarray(state['beta'], dtype=np.float64)
        assert beta.shape == (4,) and np.isfinite(beta).all() and -.08 <= beta[3] <= .20
        exact(beta, spatial_beta, 'Original reference/spatial beta')
        anchor = float(reference['fit_report']['hedonic']['maximum_fit_day_index'])
        assert np.isfinite(anchor) and anchor == float(state['anchor_day_index'])
        vector = lambda name: unit_run.vector(ROOT / name, n)
        fixed = {key: vector(paths[key]) for key in ('native', 'lgb_native', 'nn', 'log', 'R81', *BACKGROUND)}
        base = prices(.75 * fixed['native'] + .25 * fixed['lgb_native'], n)
        exact(base, vector(paths['base']), 'Saved R81 candidate base bitwise')
        with (ROOT / paths['reference_diagnostics']).open('rb') as f:
            saved_ref_diag = pickle.load(f)
        with (ROOT / paths['spatial_diagnostics']).open('rb') as f:
            saved_spatial_diag = pickle.load(f)
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p81['splits'][stage][0], p81['target_group_weights'])
        metrics = {key: score(fixed[key]) for key in ('R81', *BACKGROUND)}
        assert metrics == a['metrics']
        base_metrics = score(base)
        assert base_metrics == a['unit_branch_metrics']['candidate']['base']
        branch_metrics, values, reports = {}, {}, {}
        for mode, variant in (('control', 'trend'), ('candidate', 'anchor')):
            p, diag, rr = old.original.predict_reference(reference, Xq, base, variant=variant, return_diagnostics=True)
            prices(p, n)
            assert diag.equals(saved_ref_diag), 'Reference diagnostic columns/index unchanged'
            assert diag.index.equals(Xq.index) and list(diag.columns) == ['source', 'confidence', 'nearest_distance', 'reference_dispersion']
            branch, detail, sr = spatial.predict(state, Xq, geo, base, p, diag)
            prices(branch, n)
            fallback = ~detail['applied'].to_numpy(dtype=bool)
            exact(branch[fallback], p[fallback], mode + ' spatial fallback')
            assert detail.drop(columns=['price_adjustment']).equals(saved_spatial_diag.drop(columns=['price_adjustment'])), 'Fixed spatial diagnostics differ'
            complete = prices(np.median(np.column_stack([fixed['nn'], fixed['log'], branch]), axis=1), n)
            assert (complete > 0).all()
            branch_metrics[mode] = {'base': base_metrics, 'reference': score(p), 'spatial': score(branch)}
            if mode == 'control':
                exact(p, vector(paths['reference']), 'R81 trend reference exact')
                exact(branch, vector(paths['spatial']), 'R81 trend spatial exact')
                exact(complete, fixed['R81'], 'R81 complete median exact')
                assert branch_metrics[mode] == a['unit_branch_metrics']['candidate']
            else:
                metrics['R82'] = score(complete)
            values[mode], reports[mode] = complete, {'reference': rr, 'spatial': sr}
            for label, value in (('base', base), ('reference', p), ('unit_spatial', branch), ('predictions', complete)):
                old.previous.save_array(folder / (mode + '_' + label + '.npy'), value)
            for label, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
                with (folder / (mode + '_' + label + '.pkl')).open('xb') as f:
                    pickle.dump(value, f, protocol=5)
        old.previous.save_array(folder / 'predictions.npy', values['candidate'])
        exact(values['candidate'], vector(rel(folder / 'predictions.npy')), 'Saved R82 predictions exact')
        assert frames == {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        diff = old.previous.prior.log_run.differences
        item = dict(stage=stage, status='completed', fit_rows=len(fi), query_rows=n, dates=p81['splits'][stage],
            original_frame_sha256=frames, metrics=metrics, differences={key: diff(metrics[key], metrics['R82']) for key in ('R81', *BACKGROUND)},
            unit_branch_metrics=branch_metrics, unit_branch_differences={key: diff(branch_metrics['control'][key], branch_metrics['candidate'][key]) for key in ('base', 'reference', 'spatial')},
            pipeline_reports=reports, reference_beta=beta.tolist(), reference_and_spatial_anchor_day_index=anchor,
            R81_base_reference_spatial_median_full_vectors_exact=True, controls_full_metrics_exact=True,
            reference_diagnostics_equal=True, fixed_spatial_diagnostics_equal_except_price_adjustment=True, spatial_fallback_exact=True,
            inherited_native_model_reload_evidence_verified=True, independent_tree_NN_model_reload_performed=False,
            changed_final_rows=int(np.count_nonzero(values['candidate'] != fixed['R81'])),
            new_fits=0, new_NN_fits=0, tree_NN_model_loads=0, model_predict_calls=0,
            reference_predict_calls=2, spatial_predict_calls=2, query_labels_used_only_for_evaluation=True,
            prediction_sha256=sha(folder / 'predictions.npy'), elapsed_seconds=time.monotonic() - started,
            artifact_sha256={rel(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(folder / 'results.json', item)
        result['stages'][stage] = item
        result['checked_stages'].append(stage)
        print(json.dumps({'stage': stage, 'R82_vs_R81': item['differences']['R81']['overall'],
            'reference_difference': item['unit_branch_differences']['reference']['overall'], 'changed_final_rows': item['changed_final_rows']}), flush=True)
    assert result['checked_stages'] == list(STAGES)
    assert all(sha(ROOT / name) == digest for name, digest in source.items()) and all(sha(ROOT / name) == digest for name, digest in inputs.items())
    after = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    after['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    assert after == protected and sha(selector_path) == SELECTOR_SHA and sha(DOCUMENT) == PLAN_SHA
    result.update(status='three_periods_completed_no_adoption_decision', elapsed_seconds=time.monotonic() - tick,
        reference_predict_calls=6, spatial_predict_calls=6, protected_after=after, selector_sha256_after=sha(selector_path))
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'result_sha256': sha(OUT / 'results.json'), 'elapsed_seconds': result['elapsed_seconds'], 'new_fits': 0, 'new_CSV': False}), flush=True)


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
