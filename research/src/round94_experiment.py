"""R94 fixed attention prior NN; reuse same-window prepared inputs and v22 branches.

No torch import: only the isolated .venv_torch child can fit the new model.
The old NN control is reconstructed from saved native TWD, never refitted.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import pickle
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round47_nn_encode as encoder
import round6_geo as geo_encoder
import round19_unit_geo as area_helper
import round60_recency_price_prior as price_prior

ROOT = old.ROOT
OUT = ROOT / 'outputs/round94'
DOCUMENT = ROOT / 'reports/round94_plan.md'
TRAINER = ROOT / 'src/round94_attention_prior_train_v2.py'
REQUEST = ROOT / 'outputs/round94_launcher_20260929/research_request.json'
CPU_RECEIPT = ROOT / 'reports/round94_trainer_selfcheck_v2.json'
SMOKE_RECEIPT = ROOT / 'outputs/round94_smoke_v2_20260929/smoke.json'
SMOKE_EXECUTION = ROOT / 'outputs/round94_smoke_v2_20260929/execution.json'
SMOKE_CARDINALITIES = ROOT / 'outputs/round94_launcher_20260929/smoke_cardinalities.json'
STAGES = old.STAGES
sha, locked, write, rel, spatial = old.sha, old.locked, old.write, old.rel, old.spatial
ARRAY_NAMES = ('train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat',
               'cardinalities', 'train_price_factor_million', 'query_price_factor_million',
               'train_sample_weight')
FORMULA = 'median(spatial(reference(.75*(.5*saved_R54_area+.5*R94_prior_TWD)+.25*saved_unit_LGB)),saved_R46_log_spatial,saved_R79_unit_spatial)'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def value_sha(values, dtype=None):
    a = np.asarray(values)
    dtype = dtype or ('<i8' if a.dtype == np.int64 else '<f4' if a.dtype == np.float32 else '<f8')
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def exact(a, b, label):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def literal_configuration():
    tree = ast.parse(TRAINER.read_text(encoding='utf-8-sig'))
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets)]
    assert len(values) == 1 and isinstance(values[0], dict), 'Expected one literal trainer CONFIG'
    return values[0]


def check_launch(plan_sha):
    """Read root's immutable launch decision; no model imports or data arrays."""
    assert sha(DOCUMENT) == plan_sha
    request = read_json(REQUEST)
    frozen = request['frozen_source_and_plan']
    assert request['approval_for_three_fits'] is True and request['resource_gate_passed'] is True
    assert request['plan_sha256'] == plan_sha and isinstance(frozen, dict) and frozen
    assert request.get('cpu_receipt_path', rel(CPU_RECEIPT)) == rel(CPU_RECEIPT)
    assert request.get('smoke_receipt_path', rel(SMOKE_RECEIPT)) == rel(SMOKE_RECEIPT)
    required = [DOCUMENT, TRAINER, Path(__file__).resolve(), CPU_RECEIPT,
                SMOKE_RECEIPT, SMOKE_EXECUTION, SMOKE_CARDINALITIES]
    assert all(rel(p) in frozen for p in required), 'Missing launch source/receipt binding'
    for name, digest in frozen.items():
        assert sha(ROOT / name) == digest, 'Frozen launch dependency changed: ' + name
    config = literal_configuration()
    assert request['trainer_configuration_sha256'] == content_sha(config)
    assert config['seed'] == 42 and config['epochs'] == 30 and config['batch_size'] == 2048
    assert config['microbatch_size'] == config['predict_batch_size'] == 256
    assert config['numeric_count'] == 144 and config['categorical_count'] == 23
    cpu, smoke, execution = map(read_json, [CPU_RECEIPT, SMOKE_RECEIPT, SMOKE_EXECUTION])
    for receipt in (cpu, smoke):
        assert receipt['status'] == 'passed' and receipt['exit_code'] == 0
        assert receipt['synthetic_only'] is True and receipt['real_data_read'] is False
        assert receipt['new_real_native_fits'] == 0
        assert receipt['plan_sha256'] == plan_sha and receipt['configuration'] == config
        assert receipt['configuration_sha256'] == content_sha(config)
        assert receipt['trainer_source_sha256'] == receipt['architecture_source_sha256'] == sha(TRAINER)
        for name, digest in receipt['source_sha256'].items():
            assert sha(ROOT / name) == digest
            assert name in frozen and frozen[name] == digest, 'Receipt dependency absent from freeze'
    assert cpu['GPU_calls'] == 0 and cpu['check_count'] == len(cpu['checks']) > 0
    assert execution['exit_code'] == 0 and execution['mode'] == 'smoke'
    assert execution['source_sha256'] == sha(TRAINER)
    assert smoke['warmup_groups'] == 2 and smoke['timed_groups'] == 3
    assert smoke['synthetic_optimizer_steps'] == 5 and smoke['cost_gate_seconds'] == 5400
    assert smoke['cost_gate_passed'] is True
    assert 0 < smoke['estimated_three_stage_fit_seconds'] <= 5400
    cards = read_json(SMOKE_CARDINALITIES)
    assert smoke['cardinalities_json_sha256'] == sha(SMOKE_CARDINALITIES)
    assert smoke['cardinalities'] == cards['cardinalities'] and smoke['same_train_group_as_true_fit'] is True
    assert smoke['optimizer_steps_per_stage'] == cards['optimizer_steps_per_stage']
    assert smoke['total_optimizer_steps'] == sum(cards['optimizer_steps_per_stage'].values()) == 10260
    assert all(cards['fit_rows_by_stage'][s] > 0 for s in STAGES)
    return request, config, cpu, smoke


def run(plan_sha):
    tick = time.monotonic()
    request, config, cpu, smoke = check_launch(plan_sha)
    frozen = request['frozen_source_and_plan']
    inputs = {rel(REQUEST): sha(REQUEST)}

    def checked(name, digest):
        name = name.replace('\\', '/')
        assert name not in inputs or inputs[name] == digest, 'Conflicting input SHA: ' + name
        if name not in inputs:
            assert sha(ROOT / name) == digest, 'Input changed: ' + name
        inputs[name] = digest
        return ROOT / name

    def small(name):
        path = ROOT / name
        value = locked(path)
        checked(name, sha(path))
        checked(rel(path.with_suffix('.sha256')), sha(path.with_suffix('.sha256')))
        return value

    p48, p79 = small('outputs/round48/plan.json'), small('outputs/round79/plan.json')
    records = {r: small(f'outputs/round{r}/results.json') for r in (48, 54, 60, 63, 71, 79, 83)}
    assert all(r['checked_stages'] == list(STAGES) for r in records.values())
    assert p48['split_contract'] == p79['split_contract'] and p48['splits'] == p79['splits']
    assert records[79]['plan_sha256'] == sha(ROOT / 'outputs/round79/plan.json')
    assert records[71]['primary_seed'] == 42 and records[71]['model_predict_calls'] == 0

    def artifact(number, stage, name):
        authority = {k.replace('\\', '/'): v for k, v in records[number]['stages'][stage]['artifact_sha256'].items()}
        return checked(name, authority[name])

    def vector(path, rows):
        values = np.load(path, allow_pickle=False)
        assert values.dtype == np.float64 and values.shape == (rows,)
        assert np.isfinite(values).all() and (values >= 0).all()
        return values

    for mapping in (p48['input_sha256'], p79['input_sha256']):
        for name, digest in mapping.items():
            checked(name, digest)
    source = {}
    for inherited in (p48['source_sha256'], p79['source_sha256']):
        for name, digest in inherited.items():
            assert name not in source or source[name] == digest, 'Conflicting historical source SHA'
            assert sha(ROOT / name) == digest, 'Historical source changed: ' + name
            source[name] = digest
    for module in (old, encoder, geo_encoder, area_helper, price_prior):
        source[rel(Path(module.__file__).resolve())] = sha(Path(module.__file__))
    source.update(cpu['source_sha256'])
    source[rel(Path(__file__).resolve())] = sha(Path(__file__))
    for name, digest in {**source, **frozen}.items():
        assert sha(ROOT / name) == digest
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    for name in ('runs/versions.json', 'outputs/current_private_candidate.json', 'PRIVATE_SELECTION.md',
                 'CURRENT_STATUS.md', 'outputs/v22_submission_focus_20260928/user_selection.json',
                 'outputs/round84/decision.json', 'outputs/round79/selection/decision.json'):
        protected[name] = sha(ROOT / name)
    # Bind the actual artifacts before fitting; no old .pt model needs loading.
    for s in STAGES:
        for r in (48, 54, 60, 63, 71, 79, 83):
            for kind in ('fit', 'query'):
                artifact(r, s, f'outputs/round{r}/{s}/{kind}_indices.npy')
        for suffix in ('train_oof_unit_prior.npy', 'query_unit_prior.npy', 'prior_fold_ids.npy',
                       'price_prior_state.json', 'prior_preparation.json'):
            artifact(60, s, f'outputs/round60/{s}/{suffix}')
        for suffix in ('prepared.npz', 'preparation.json', 'model/training.json', 'model/predictions_twd.npy'):
            artifact(63, s, f'outputs/round63/{s}/{suffix}')
        artifact(54, s, f'outputs/round54/{s}/model/predictions_twd.npy')
        for suffix in ('native_mean.npy', 'base.npy', 'reference.npy', 'nn_spatial.npy',
                       'reference_diagnostics.pkl', 'spatial_diagnostics.pkl'):
            artifact(71, s, f'outputs/round71/{s}/seed42/{suffix}')
        for r in (79, 83):
            artifact(r, s, f'outputs/round{r}/{s}/predictions.npy')
        artifact(79, s, f'outputs/round79/{s}/unit_spatial.npy')
        for suffix in ('encoder_state.json', 'area_preparation.json'):
            artifact(48, s, f'outputs/round48/{s}/{suffix}')
    plan = {'round': 94, 'candidate': 'R94', 'stage_order': list(STAGES), 'planned_new_NN_fits': 3,
            'primary_control': 'v22', 'secondary_control': 'v24', 'formula': FORMULA,
            'configuration': config, 'configuration_sha256': content_sha(config),
            'text_plan_sha256': plan_sha, 'source_sha256': source, 'input_sha256': inputs,
            'frozen_source_and_plan': frozen, 'launch_request_sha256': sha(REQUEST),
            'splits': p79['splits'], 'split_contract': p79['split_contract'],
            'target_group_weights': p79['target_group_weights'], 'protected_before': protected,
            'fresh_blind_validation': False, 'Public_feedback_used_for_tuning': False,
            'query_labels_in_trainer': False, 'shared_driver_deserializes_canonical_y': True,
            'query_labels_used_only_for_evaluation': True, 'old_NN_model_predict_calls': 0,
            'new_native_tree_fits': 0, 'new_reference_or_spatial_states': 0,
            'new_CSV': False, 'new_test_predictions': False, 'automatic_adoption': False,
            'resource_estimate_seconds': smoke['estimated_three_stage_fit_seconds']}
    write(OUT / 'plan.json', plan)
    Xall, yall = old.previous.load_training_rows(p79)
    assert len(Xall) == 455255
    cats = read_json(ROOT / 'reports/feature_schema.json')['categorical_columns']
    result = {k: v for k, v in plan.items() if k not in ('configuration',)}
    result.update(status='running', checked_stages=[], stages={}, new_NN_fits=0,
                  plan_sha256=sha(OUT / 'plan.json'), runtime={'numpy': np.__version__, 'pandas': pd.__version__})

    for s in STAGES:
        start = time.monotonic()
        folder = OUT / s
        folder.mkdir()
        b = p48['bindings'][s]
        bound79 = p79['bindings'][s]
        contract = p79['split_contract'][s]
        indices = {}
        for kind in ('fit', 'query'):
            v = np.load(ROOT / f'outputs/round63/{s}/{kind}_indices.npy', allow_pickle=False)
            assert v.dtype == np.int64 and v.shape == (contract[kind + '_rows'],)
            assert (np.diff(v) > 0).all() and v.min() >= 0 and v.max() < len(Xall)
            assert value_sha(v) == contract[kind + '_indices_content_sha256']
            for r in (48, 54, 60, 71, 79, 83):
                exact(v, np.load(ROOT / f'outputs/round{r}/{s}/{kind}_indices.npy', allow_pickle=False), 'Canonical indices')
            indices[kind] = v
            old.previous.save_array(folder / (kind + '_indices.npy'), v)
        fi, qi = indices['fit'], indices['query']
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        yfit = yall[fi].copy()
        n = len(qi)
        cutoff = (pd.Timestamp(p79['splits'][s][0]) - pd.Timestamp('2000-01-01')).days
        end = (pd.Timestamp(p79['splits'][s][1]) - pd.Timestamp('2000-01-01')).days
        assert np.isfinite(X.trade_day_index).all() and (X.trade_day_index < cutoff).all()
        assert Xq.trade_day_index.between(cutoff, end).all()
        frames = {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        assert frames == records[79]['stages'][s]['original_frame_sha256']
        assert frames == records[63]['stages'][s]['original_frame_sha256']
        prep_start = time.monotonic()
        prepared = ROOT / f'outputs/round63/{s}/prepared.npz'
        preparation = locked(ROOT / f'outputs/round63/{s}/preparation.json')
        baseline_training = read_json(ROOT / f'outputs/round63/{s}/model/training.json')
        assert sha(prepared) == preparation['prepared_sha256'] == baseline_training['input_sha256']
        assert preparation['all_eight_R60_arrays_reused_without_changes']
        assert preparation['encoder_refit'] is False and preparation['prior_refit'] is False
        with np.load(prepared, allow_pickle=False) as z:
            assert set(z.files) == set(ARRAY_NAMES), 'Exactly nine arrays; query labels forbidden'
            arrays = {k: z[k] for k in ARRAY_NAMES}
        for prefix, rows in (('train', len(fi)), ('query', n)):
            num, cat = arrays[prefix + '_num'], arrays[prefix + '_cat']
            assert num.dtype == np.float32 and num.shape == (rows, 144) and np.isfinite(num).all()
            assert cat.dtype == np.int64 and cat.shape == (rows, 23)
        cards = arrays['cardinalities']
        assert cards.dtype == np.int64 and cards.shape == (23,) and (cards > 0).all()
        for name in ('train_cat', 'query_cat'):
            assert (arrays[name] >= 0).all() and (arrays[name] < cards).all()
        exact(arrays['train_y_million'], np.asarray(yfit / 1e6, dtype=np.float32), 'Prepared fit labels')
        assert (arrays['train_y_million'] > 0).all() and np.isfinite(arrays['train_y_million']).all()
        # Original R6 geography feeds the NN encoder; R39 geography is only for R/S below.
        allgeo = pd.read_pickle(ROOT / b['original_geo'])
        geo_encoder.check_geo(allgeo)
        assert len(allgeo) == len(Xall) and allgeo.index.equals(Xall.index)
        G, Gq = allgeo.iloc[fi].reset_index(drop=True), allgeo.iloc[qi].reset_index(drop=True)
        del allgeo
        r47 = locked(ROOT / b['R47_result'])
        assert r47['original_frame_sha256'] == {**frames, 'original_fit_geo': spatial.frame_sha(G),
                                                'original_query_geo': spatial.frame_sha(Gq)}
        saved_encoder = locked(ROOT / b['R47_encoder'])
        assert saved_encoder == locked(ROOT / f'outputs/round48/{s}/encoder_state.json')
        assert sha(ROOT / b['R47_encoder']) == sha(ROOT / f'outputs/round48/{s}/encoder_state.json')
        X95, Q95 = geo_encoder.augment_features(X, G), geo_encoder.augment_features(Xq, Gq)
        cats95 = geo_encoder.categories(X, cats)
        assert encoder.fit_encoder(X95, cats95) == saved_encoder, 'Fit-only encoder state replay'
        assert saved_encoder['cardinalities'] == cards.tolist() == baseline_training['cardinalities']
        assert saved_encoder['fit_rows'] == len(fi) and not saved_encoder['query_used_for_state']
        assert not saved_encoder['target_used_for_state']
        for prefix, frame in (('train', X95), ('query', Q95)):
            num, cat = encoder.encode(saved_encoder, frame)
            exact(num, arrays[prefix + '_num'], 'Frozen numeric encoding replay')
            exact(cat, arrays[prefix + '_cat'], 'Frozen categorical encoding replay')
        del X95, Q95, G, Gq, num, cat
        prior_path = ROOT / f'outputs/round60/{s}'
        prior_prep = locked(prior_path / 'prior_preparation.json')
        evidence = prior_prep['prior_evidence']
        assert evidence['source_indices_sha256'] == value_sha(fi)
        assert evidence['fit_target_sha256'] == value_sha(yfit)
        assert evidence['uses_fit_labels'] and not evidence['uses_query_labels'] and not evidence['chronological_oof']
        train_prior = vector(prior_path / 'train_oof_unit_prior.npy', len(fi))
        query_prior = vector(prior_path / 'query_unit_prior.npy', n)
        replay_oof, replay_state, replay_folds, replay_evidence = price_prior.fit_oof_prior(X, yfit, fi)
        exact(replay_oof, train_prior, 'Full fit-only OOF replay')
        exact(replay_folds, np.load(prior_path / 'prior_fold_ids.npy', allow_pickle=False), 'Prior fold replay')
        assert replay_evidence == evidence and replay_state == locked(prior_path / 'price_prior_state.json')
        exact(price_prior.predict_prior(replay_state, Xq), query_prior, 'Full-fit query prior replay')
        for prefix, frame, prior_values in (('train', X, train_prior), ('query', Xq, query_prior)):
            factor = np.asarray(area_helper.effective_area(frame) * prior_values / 1e6, dtype=np.float32)
            name = prefix + '_price_factor_million'
            exact(factor, arrays[name], 'Prepared float32 factor replay')
            assert value_sha(factor) == prior_prep[name + '_sha256'] == baseline_training[name + '_sha256']
            assert np.isfinite(factor).all() and (factor > 0).all()
        weights64, valid_dates, date_summary = price_prior._weights(price_prior._dates(X))
        exact(weights64.astype(np.float32), arrays['train_sample_weight'], 'Saved float32 fit dates replay')
        assert date_summary == preparation['date_weight_summary_float64']
        assert value_sha(arrays['train_sample_weight']) == preparation['train_sample_weight_sha256'] == baseline_training['train_sample_weight_sha256']
        assert np.isfinite(arrays['train_sample_weight']).all() and (arrays['train_sample_weight'] > 0).all()
        assert (arrays['train_sample_weight'][~valid_dates] == 1).all()
        original_six = prior_prep['unchanged_six_array_sha256']
        assert all(value_sha(arrays[k]) == h for k, h in original_six.items())
        array_identity = {k: {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': value_sha(a)}
                          for k, a in arrays.items()}
        query_factor = arrays['query_price_factor_million'].copy()
        prep_receipt = {'status': 'passed', 'fit_rows': len(fi), 'query_rows': n,
                        'source_indices_role': 'canonical train positions; not raw cache mapping or reset index',
                        'fit_indices_content_sha256': value_sha(fi), 'query_indices_content_sha256': value_sha(qi),
                        'prepared_path': rel(prepared), 'prepared_sha256': sha(prepared),
                        'all_input_arrays': array_identity, 'encoder_state_path': b['R47_encoder'],
                        'encoder_state_sha256': sha(ROOT / b['R47_encoder']),
                        'fit_encoder_state_replay_exact': True, 'encoded_fit_query_replay_exact': True,
                        'OOF_fold_state_evidence_replay_exact': True, 'query_prior_replay_exact': True,
                        'factor_replay_exact': True, 'date_weight_replay_exact': True,
                        'fit_labels_used_only_for_prior_and_train_target': True, 'query_labels_used': False,
                        'chronological_oof': False, 'elapsed_seconds': time.monotonic() - prep_start}
        write(folder / 'prepared_verification.json', prep_receipt)
        del arrays, replay_oof, replay_state, replay_folds, replay_evidence, train_prior, query_prior, weights64, factor
        # Same fitted reference/spatial correction, not a new correction family.
        with (ROOT / b['reference']).open('rb') as f:
            reference = pickle.load(f)
        with (ROOT / b['state']).open('rb') as f:
            state_spatial = pickle.load(f)
        summary = locked(ROOT / b['state_summary'])
        assert state_spatial['state_summary'] == summary and state_spatial['parameters'] == spatial.parameters()
        assert summary['fit_X_sha256'] == frames['fit_X'] and not summary['query_used_in_build']
        assert summary['fit_geo_sha256'] == b['manifest']['features']['fit']['frame_sha256']
        assert reference['fit_report']['fitting_rows'] == len(fi)
        assert value_sha(reference['reference_residual']) == summary['existing_reference_residual_sha256']
        geo = pd.read_pickle(ROOT / b['query_geo'])
        assert geo.index.equals(Xq.index) and spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        area = vector(ROOT / f'outputs/round54/{s}/model/predictions_twd.npy', n)
        prior_old = vector(ROOT / f'outputs/round63/{s}/model/predictions_twd.npy', n)
        old_binding = records[71]['stages'][s]['native_seed_bindings']['42']
        assert old_binding['path'] == f'outputs/round63/{s}/model/predictions_twd.npy'
        assert old_binding['sha256'] == sha(ROOT / old_binding['path'])
        lgb = vector(ROOT / b['old_unit'], n)
        fixed = {k: vector(ROOT / bound79['paths'][k], n) for k in ('nn', 'log', 'v11', 'v12', 'v5')}
        fixed['unit'] = vector(ROOT / f'outputs/round79/{s}/unit_spatial.npy', n)
        fixed['v22'] = vector(ROOT / f'outputs/round79/{s}/predictions.npy', n)
        fixed['v24'] = vector(ROOT / f'outputs/round83/{s}/predictions.npy', n)
        old_mean = .5 * area + .5 * prior_old
        old_base = .75 * old_mean + .25 * lgb
        old_ref, old_diag, _ = old.original.predict_reference(reference, Xq, old_base, variant='trend', return_diagnostics=True)
        old_nn, old_detail, _ = spatial.predict(state_spatial, Xq, geo, old_base, old_ref, old_diag)
        for name, values in (('native_mean', old_mean), ('base', old_base), ('reference', old_ref), ('nn_spatial', old_nn)):
            exact(values, vector(ROOT / f'outputs/round71/{s}/seed42/{name}.npy', n), 'R71 seed42 ' + name)
        for name, frame in (('reference_diagnostics', old_diag), ('spatial_diagnostics', old_detail)):
            with (ROOT / f'outputs/round71/{s}/seed42/{name}.pkl').open('rb') as f:
                pd.testing.assert_frame_equal(frame, pickle.load(f), check_exact=True)
        exact(old_nn, fixed['nn'], 'R79 selected NN branch')
        exact(np.median(np.column_stack([old_nn, fixed['log'], fixed['unit']]), axis=1), fixed['v22'], 'Complete v22 median')
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p79['splits'][s][0], p79['target_group_weights'])
        metrics = {k: score(fixed[k]) for k in ('v11', 'v12', 'v5', 'v22', 'v24')}
        for k in ('v11', 'v12', 'v5'):
            assert metrics[k] == records[79]['stages'][s]['metrics'][k]
        assert metrics['v22'] == records[79]['stages'][s]['metrics']['R79']
        assert metrics['v24'] == records[83]['stages'][s]['metrics']['R83']
        old_layer_metrics = {k: score(v) for k, v in (('native', prior_old), ('base', old_base),
                                                     ('reference', old_ref), ('spatial', old_nn))}
        write(folder / 'control_verification.json', {'controls_full_metrics_exact': True,
              'R71_seed42_mean_base_reference_spatial_vectors_exact': True, 'R71_diagnostics_exact': True,
              'v22_complete_median_exact': True, 'old_NN_fit_calls': 0, 'old_NN_model_predict_calls': 0})
        # Only after provenance and old full-control replay may the child fit start.
        for name, digest in frozen.items():
            assert sha(ROOT / name) == digest
        model_dir = folder / 'model'
        assert not model_dir.exists(), 'Trainer creates its own new output directory'
        process_dir = folder / 'trainer_process'
        process_dir.mkdir()
        command = [str(old.TORCH_PYTHON), '-u', '-B', '-X', 'utf8', str(TRAINER),
                   '--data', str(prepared), '--out', str(model_dir)]
        write(process_dir / 'fit_started.json', {'started_at_utc': old.now(), 'command': command,
              'input_sha256': sha(prepared), 'configuration': config, 'query_labels_in_trainer': False,
              'automatic_retry': False, 'source_sha256': sha(TRAINER)})
        child_tick = time.monotonic()
        with (process_dir / 'trainer.log').open('x', encoding='utf-8') as log:
            done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        write(process_dir / 'execution.json', {'exit_code': done.returncode, 'command': command,
              'elapsed_seconds': time.monotonic() - child_tick, 'automatic_retry': False})
        assert done.returncode == 0, 'Trainer failed; preserve evidence, no adaptive retry'
        training = read_json(model_dir / 'training.json')
        assert training['status'] == 'completed' and training['configuration'] == config
        assert training['configuration_sha256'] == content_sha(config)
        assert training['source_sha256'] == training['architecture_source_sha256'] == sha(TRAINER)
        assert training['plan_sha256'] == plan_sha and training['upstream_commit'] == 'e3ed46cac38568785289d8fa16b8cfa585bde27e'
        assert training['input_sha256'] == sha(prepared) and training['all_input_arrays'] == array_identity
        assert training['fit_rows'] == len(fi) and training['query_rows'] == n and training['numeric_count'] == 144
        assert training['cardinalities'] == cards.tolist() and [e['epoch'] for e in training['epochs']] == list(range(1, 31))
        assert not training['query_labels_available'] and not training['query_labels_used']
        assert training['saved_model_reload_exact'] and training['reload_ratio_exact'] and training['reload_TWD_exact']
        assert training['reload_rows'] == n and training['predict_batch_size'] == 256
        for name, digest in training['source_dependencies'].items():
            assert sha(ROOT / name) == digest == frozen[name]
        for name, digest in training['artifact_sha256'].items():
            assert sha(model_dir / name) == digest
        ratios = vector(model_dir / 'predictions_ratio.npy', n)
        native = vector(model_dir / 'predictions_twd.npy', n)
        assert (ratios > 0).all() and (native > 0).all()
        exact(native, ratios * query_factor.astype(np.float64) * 1e6, 'Ratio factor applied exactly once')
        mean = .5 * area + .5 * native
        base = .75 * mean + .25 * lgb
        reference_prediction, diag, rr = old.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
        nn_branch, detail, sr = spatial.predict(state_spatial, Xq, geo, base, reference_prediction, diag)
        fallback = ~detail['applied'].to_numpy(dtype=bool)
        exact(nn_branch[fallback], reference_prediction[fallback], 'Spatial fallback')
        predictions = np.median(np.column_stack([nn_branch, fixed['log'], fixed['unit']]), axis=1)
        assert predictions.dtype == np.float64 and predictions.shape == (n,) and np.isfinite(predictions).all() and (predictions > 0).all()
        metrics['R94'] = score(predictions)
        for name, values in (('native_mean', mean), ('base', base), ('reference', reference_prediction),
                             ('nn_spatial', nn_branch), ('predictions', predictions)):
            old.previous.save_array(folder / (name + '.npy'), values)
        for name, frame in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
            with (folder / (name + '.pkl')).open('xb') as f:
                pickle.dump(frame, f, protocol=5)
        assert frames == {'fit_X': spatial.frame_sha(X), 'query_X': spatial.frame_sha(Xq)}
        difference = old.previous.prior.log_run.differences
        item = {'stage': s, 'status': 'completed', 'fit_rows': len(fi), 'query_rows': n,
                'dates': p79['splits'][s], 'original_frame_sha256': frames, 'metrics': metrics,
                'differences': {k: difference(metrics[k], metrics['R94']) for k in metrics if k != 'R94'},
                'nn_branch_metrics': {k: score(v) for k, v in (('native', native), ('base', base),
                                      ('reference', reference_prediction), ('spatial', nn_branch))},
                'original_nn_branch_metrics': old_layer_metrics, 'reference_report': rr, 'spatial_report': sr,
                'controls_full_metrics_exact': True, 'R71_seed42_full_chain_and_diagnostics_exact': True,
                'v22_full_median_exact': True, 'prepared_provenance_replay_exact': True,
                'full_query_raw_ratio_TWD_reload_exact': True, 'query_labels_used_only_for_evaluation': True,
                'new_NN_fits': 1, 'new_native_tree_fits': 0, 'old_NN_model_predict_calls': 0,
                'reference_predict_calls': 2, 'spatial_predict_calls': 2,
                'fit_seconds': training['fit_seconds'], 'trainer_elapsed_seconds': training['elapsed_seconds'],
                'parameter_count': training['parameter_count'], 'elapsed_seconds': time.monotonic() - start,
                'prediction_sha256': sha(folder / 'predictions.npy'),
                'artifact_sha256': {rel(p): sha(p) for p in folder.rglob('*') if p.is_file()}}
        write(folder / 'results.json', item)
        result['stages'][s] = item
        result['checked_stages'].append(s)
        result['new_NN_fits'] += 1
        print(json.dumps({'stage': s, 'fit_seconds': training['fit_seconds'], 'R94': metrics['R94']['overall_mae'],
                          'v22': metrics['v22']['overall_mae'], 'v24': metrics['v24']['overall_mae']}), flush=True)
    assert result['new_NN_fits'] == 3 and result['checked_stages'] == list(STAGES)
    for name, digest in {**inputs, **source, **frozen, **protected}.items():
        assert sha(ROOT / name) == digest, 'Dependency changed after research: ' + name
    assert sha(DOCUMENT) == plan_sha
    result.update(status='three_periods_completed_no_adoption_decision', input_sha256=inputs,
                  protected_after={name: sha(ROOT / name) for name in protected},
                  reference_predict_calls=6, spatial_predict_calls=6, elapsed_seconds=time.monotonic() - tick)
    write(OUT / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error('--run required; no implicit training')
    assert sha(DOCUMENT) == args.plan_sha and not OUT.exists(), 'Plan changed or output exists'
    OUT.mkdir()
    started, tick, failure = old.now(), time.monotonic(), None
    write(OUT / 'started.json', {'started_at_utc': started, 'command': sys.argv,
          'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        run(args.plan_sha)
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'started_at_utc': started, 'completed_at_utc': old.now(),
          'elapsed_seconds': time.monotonic() - tick, 'exit_code': 1 if failure else 0,
          'failure': failure, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(1 if failure else 0)
