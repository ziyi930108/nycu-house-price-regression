"""R83 full-data seed137 unit CatBoost; unchanged R79 inputs, no CSV or old model load."""
from pathlib import Path
import argparse
import contextlib
import json
import sys
import time
import traceback
import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import round79_experiment as history
import round83_seed_unit as helper
import round79_recency_unit as baseline

ROOT = history.ROOT
OUT = ROOT / 'outputs/round83/final_model'
PLAN = ROOT / 'reports/round83_final_plan.md'
PLAN_SHA = '792631bfde4e83e89134246e55a91fe05ce0196c2bc97c7b4ed360f48a10b4c6'
DECISION = 'outputs/round83/decision.json'
DECISION_SHA = '759d2ec0cfafe3c05575ecc0536e4287b5e006286564b922cea60cb5f34c7b28'
HELPER_SHA = 'c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b'
R79_RESULT = 'outputs/round79/final_model/results.json'
R79_SHA = '4b7172156359f476add69af11f9d2f53c78b3879bb469acf34dbd072c0db7942'
R79_EXECUTION_SHA = 'ce662450d58f6cc22d473f4111a09e4b6fa89278b3fcbf8aa87209077d981f43'
SELECTOR_SHA = 'c993a861fa6766fb5c9da8c156521bb28886a459587653ee8734b2927b9e87a8'
AUTHORITIES = {
    'outputs/round40/trial/trial_release.json': 'c5f33956319569f951b99c7a79de089a4c85ae464803d18fb25d128730e5ca2d',
    'outputs/round63/final/trial_release.json': '55e8fcffc1aef16dd056652191aff5363fa99332db97bcfd013956e88ce5b9a1',
}
GEO = 'outputs/round6/geography/final/'
sha, read, write, rel = history.sha, history.read, history.write, history.rel
save_array = history.old.previous.save_array
value_sha = history.old.original.value_sha
TARGET = '總價元'


def run(expected_plan_sha):
    tick = time.monotonic()
    assert expected_plan_sha == PLAN_SHA == sha(PLAN)
    assert sha(Path(helper.__file__)) == HELPER_SHA
    inputs, known = {}, {}

    def bind(name, digest):
        name = name.replace('\\', '/')
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs:
            assert sha(path) == digest, name
        inputs[name] = digest
        return path

    def small(name, digest, receipt=True):
        path = bind(name, digest)
        result = read(path) if receipt else json.loads(path.read_text(encoding='utf8'))
        if receipt:
            bind(rel(path.with_suffix('.sha256')), sha(path.with_suffix('.sha256')))
        return result

    bind(rel(PLAN), PLAN_SHA)
    decision = small(DECISION, DECISION_SHA)
    assert decision['round'] == 83 and decision['candidate'] == 'R83' and decision['final_fit_seed'] == 137
    assert decision['status'] == 'advance_one_fixed_complementary_candidate_not_yet_released'
    for name, digest in decision['input_sha256'].items():
        bind(name, digest)
    historical = small('outputs/round83/results.json', decision['input_sha256']['outputs/round83/results.json'])
    assert historical['checked_stages'] == list(history.STAGES) and historical['new_fits'] == 3
    assert historical['status'] == 'three_periods_completed_no_adoption_decision'
    p40, p63 = [small(name, digest) for name, digest in AUTHORITIES.items()]
    p79 = small(R79_RESULT, R79_SHA)
    e79 = small('outputs/round79/final_model/execution.json', R79_EXECUTION_SHA)
    assert p79['round'] == 79 and p79['status'] == 'final_unit_model_ready_no_CSV'
    assert p79['full_query_raw_TWD_reload_exact'] and p79['new_fits'] == 1 and p79['prediction_units'] == 'TWD'
    assert e79['exit_code'] == 0 and e79['failure'] is None
    assert e79['source_sha256'] == p79['source_sha256']['src/round79_final_unit.py']
    assert e79['text_plan_sha256'] == p79['text_plan_sha256'] and e79['decision_sha256'] == p79['decision_sha256']
    for proof in (p40, p63, p79):
        assert proof['fit_rows'] == 467221 and proof['query_rows'] == 47073
        for field in ('input_sha256', 'artifact_sha256'):
            for name, digest in proof[field].items():
                name = name.replace('\\', '/')
                assert name not in known or known[name] == digest, name
                known[name] = digest
    sources = history.old.previous.merge(historical['source_sha256'], p40['source_sha256'], p63['source_sha256'], p79['source_sha256'],
        helper.source_hashes(), {rel(Path(__file__)): sha(Path(__file__)), rel(Path(history.__file__)): sha(Path(history.__file__))})
    for name, digest in sources.items():
        bind(name, digest)

    def checked(name):
        return bind(name, known[name])

    def record(name, receipt=True):
        return small(name, known[name], receipt)

    old_meta = record('outputs/round40/trial/model/metadata.json')
    legacy = record('outputs/round40/trial/model/feature_state.json')
    assert old_meta['feature_state_sha256'] == known['outputs/round40/trial/model/feature_state.json']
    assert old_meta['feature_state_content_sha256'] == legacy['state_content_sha256']
    r79_meta = record(p79['model_metadata_path'])
    r79_legacy = record('outputs/round79/final_model/model/feature_state.json')
    c79 = record('outputs/round79/final_model/input_contract.json')
    assert inputs[p79['model_metadata_path']] == p79['model_metadata_sha256']
    assert inputs['outputs/round79/final_model/input_contract.json'] == p79['input_contract_sha256']
    assert r79_legacy == legacy and r79_meta['feature_state_content_sha256'] == legacy['state_content_sha256']
    assert r79_meta['parameters'] == baseline.parameters() == c79['parameters'] and r79_meta['seed'] == 42
    assert r79_meta['full_query_raw_TWD_reload_exact'] and r79_meta['reload_rows'] == 47073
    prep = record('outputs/round63/final/preparation.json')
    prior_contract = record('outputs/round63/final/input_contract.json')
    prior_training = record('outputs/round63/final/model/training.json', receipt=False)
    assert prep == p63['sample_weight_preparation'] == c79['R63_weight_preparation'] == r79_meta['R63_preparation']
    assert prep['prepared_path'] == 'outputs/round63/final/prepared.npz'
    assert known[prep['prepared_path']] == prep['prepared_sha256'] == prior_training['input_sha256']
    assert prep['train_sample_weight_sha256'] == p63['train_sample_weight_sha256'] == prior_training['train_sample_weight_sha256']
    assert prep['fit_rows'] == prior_training['fit_rows'] == 467221 and prep['query_rows'] == prior_training['query_rows'] == 47073
    assert prep['weights_use_fit_X_only'] and not prep['weights_use_targets'] and not prep['weights_use_query_dates']
    assert not prep['query_labels_used'] and not prep['float32_weights_renormalized'] and prep['neutral_weights_exactly_one']
    assert not prep['encoder_refit'] and not prep['prior_refit'] and not prior_contract['query_labels_used']
    assert prior_training['status'] == 'completed' and prior_training['extra_sample_weights']
    assert not prior_training['sample_weights_recomputed_in_trainer'] and not prior_training['sample_weights_batch_renormalized']
    assert prior_training['train_sample_weight_dtype'] == 'float32' and prior_training['train_sample_weight_shape'] == [467221]

    paths = [f'data_cache/{s}_features.pkl' for s in ('train', 'valid', 'test')]
    paths += ['data_cache/train.pkl', 'data_cache/valid.pkl', 'reports/feature_schema.json',
              'runs/v1/models/final_config.json', prep['prepared_path']]
    paths += [GEO + s + '_source_cache_indices.npy' for s in ('train', 'valid', 'full', 'test')]
    paths += [GEO + 'full.pkl', GEO + 'test.pkl', 'outputs/round63/final/fit_indices.npy', 'outputs/round63/final/query_indices.npy']
    for name in paths:
        checked(name)
    # The full-final date/feature provenance is inherited from both saved contracts.
    for name in paths:
        if name in prior_contract['input_sha256']:
            assert prior_contract['input_sha256'][name] == inputs[name], name
        assert c79['input_sha256'][name] == p79['input_sha256'][name] == inputs[name], name
    parameters = helper.parameters()
    assert baseline.parameters() == old_meta['parameters'] == legacy['parameters'] == p40['parameters'] == r79_meta['parameters']
    assert parameters == {**r79_meta['parameters'], 'random_seed': 137}
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 24
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    assert protected == decision['protected_before']
    selector = ROOT / 'outputs/current_private_candidate.json'
    selector_sha = sha(selector)
    assert selector_sha == decision['selector_sha256_before'] == SELECTOR_SHA
    contract = {'round': 83, 'stage': 'final', 'text_plan_sha256': PLAN_SHA, 'decision_sha256': DECISION_SHA,
        'source_sha256': sources, 'input_sha256': dict(inputs), 'parameters': parameters,
        'fit_rows': 467221, 'query_rows': 47073, 'planned_new_fits': 1, 'new_NN_fits': 0,
        'target_formula': helper.TARGET_FORMULA, 'inverse_formula': helper.INVERSE_FORMULA,
        'sample_weight': helper.WEIGHT_FORMULA, 'R63_weight_preparation': prep,
        'original_R19_state_content_sha256': legacy['state_content_sha256'],
        'R79_final_model_result_path': R79_RESULT, 'R79_final_model_result_sha256': R79_SHA,
        'baseline_parameters': r79_meta['parameters'], 'changed_parameter_names': ['random_seed'],
        'seed': 137, 'legacy_state_training_fields_are_provenance_only': True,
        'date_weights_recomputed': False, 'date_weights_renormalized': False, 'query_labels_used': False,
        'reference_predict_calls': 0, 'spatial_predict_calls': 0, 'new_CSV': False,
        'protected_before': protected, 'selector_sha256_before': selector_sha}
    write(OUT / 'input_contract.json', contract)

    frames = {s: pd.read_pickle(ROOT / f'data_cache/{s}_features.pkl') for s in ('train', 'valid', 'test')}
    schema = json.loads((ROOT / 'reports/feature_schema.json').read_text(encoding='utf8'))
    for s, rows in (('train', 455784), ('valid', 11994), ('test', 47073)):
        assert len(frames[s]) == rows and frames[s].index.equals(pd.RangeIndex(rows))
        assert list(frames[s]) == schema['columns'] and len(schema['columns']) == 87 and TARGET not in frames[s]
    indices = {}
    for name, bound, rows, identity in (('train', 455784, 455255, False), ('valid', 11994, 11994, True),
                                       ('full', 467249, 467221, False), ('test', 47073, 47073, True)):
        index = np.load(ROOT / (GEO + name + '_source_cache_indices.npy'), allow_pickle=False)
        assert index.dtype == np.int64 and index.shape == (rows,)
        assert (np.diff(index) > 0).all() and index.min() >= 0 and index.max() < bound
        if identity:
            assert np.array_equal(index, np.arange(rows, dtype=np.int64))
        indices[name] = index
    kt, kv, kf, kq = [indices[name] for name in ('train', 'valid', 'full', 'test')]
    X = pd.concat([frames['train'].iloc[kt], frames['valid'].iloc[kv]], ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq = frames['test'].iloc[kq].reset_index(drop=True)
    labels = {s: pd.read_pickle(ROOT / f'data_cache/{s}.pkl')[TARGET].to_numpy(dtype=np.float64, copy=True) for s in ('train', 'valid')}
    assert len(labels['train']) == 455784 and len(labels['valid']) == 11994
    y = np.concatenate([labels['train'][kt], labels['valid'][kv]])[kf]
    assert y.shape == (467221,) and np.isfinite(y).all() and (y > 0).all()
    target_sha = helper.values_sha256(y)
    assert target_sha == p63['fit_target_values_sha256'] == old_meta['fit_target_values_sha256'] == p79['fit_target_values_sha256'] == r79_meta['fit_target_values_sha256']
    del labels, frames
    # These original R6 files are already canonical; never apply KF to them again.
    G, Gq = [pd.read_pickle(ROOT / (GEO + name)) for name in ('full.pkl', 'test.pkl')]
    assert len(G) == len(X) == 467221 and len(Gq) == len(Xq) == 47073
    assert G.index.equals(X.index) and Gq.index.equals(Xq.index)
    before = {name: helper.base.frame_sha(frame) for name, frame in (('X', X), ('Xq', Xq), ('geo', G), ('geoq', Gq))}
    assert before == p63['input_frame_sha256'] == p79['input_frame_sha256'] == r79_meta['input_frame_sha256']
    assert before['X'] == old_meta['fit_original_feature_values_sha256']
    assert before['geo'] == old_meta['fit_geo_feature_values_sha256']
    cats = schema['categorical_columns']
    assert cats == legacy['original_categorical_columns']
    assert helper.fit_feature_state(X, G, cats)['legacy_feature_state'] == legacy == r79_legacy
    canonical = {}
    for name, rows in (('fit', 467221), ('query', 47073)):
        index = np.load(ROOT / f'outputs/round63/final/{name}_indices.npy', allow_pickle=False)
        assert index.dtype == np.int64 and np.array_equal(index, np.arange(rows, dtype=np.int64))
        canonical[name + '_indices_content_sha256'] = value_sha(index, '<i8')
        assert canonical[name + '_indices_content_sha256'] == p79[name + '_indices_content_sha256'] == r79_meta[name + '_indices_content_sha256']
        prior_index = f'outputs/round79/final_model/{name}_indices.npy'
        assert known[prior_index] == inputs[f'outputs/round63/final/{name}_indices.npy']
        save_array(OUT / (name + '_indices.npy'), index)
    with np.load(ROOT / prep['prepared_path'], allow_pickle=False) as data:
        assert set(data.files) == history.NPZ_KEYS
        weights32 = data['train_sample_weight'].copy()
        target32 = data['train_y_million']
        assert target32.dtype == np.float32 and target32.shape == (467221,)
        assert target32.tobytes() == np.asarray(y / 1e6, dtype=np.float32).tobytes()
    assert weights32.dtype == np.float32 and weights32.shape == (467221,) and np.isfinite(weights32).all() and (weights32 > 0).all()
    assert value_sha(weights32, '<f4') == prep['train_sample_weight_sha256']
    history.summary_equal(baseline._summary(weights32), prep['train_sample_weight_summary'])
    history.summary_equal(baseline._summary(weights32), prior_training['train_sample_weight_summary'])
    d, bad, outside = (X[k].to_numpy(dtype=np.float64) for k in ('trade_day_index', 'trade_date_invalid', 'trade_date_out_of_range'))
    dates = prep['date_weight_summary_float64']
    for values, key in ((d, 'date_values_sha256'), (bad, 'invalid_flags_sha256'), (outside, 'out_of_range_flags_sha256')):
        assert helper.values_sha256(values) == dates[key]
    valid = np.isfinite(d) & (bad == 0) & (outside == 0)
    assert dates['reference_rows'] == 467221 and dates['valid_date_rows'] == int(valid.sum()) == 467213
    assert dates['neutral_date_rows'] == int((~valid).sum()) == 8
    assert np.flatnonzero(~valid).tolist() == prep['neutral_date_positions'] and (weights32[~valid] == np.float32(1)).all()
    assert dates['weight_rules']['half_life_days'] == 365 and not dates['weight_rules']['weights_use_query'] and not dates['weight_rules']['weights_use_targets']
    area_weights = helper.original.sample_weights(X)
    pool_weights = area_weights * weights32.astype(np.float64)
    assert helper.values_sha256(area_weights) == legacy['fit_sample_weights_sha256']
    assert np.isfinite(pool_weights).all() and (pool_weights > 0).all()
    assert value_sha(weights32, '<f4') == p79['train_date_weight_sha256'] == r79_meta['fit_date_weights_sha256']
    assert helper.values_sha256(pool_weights) == p79['actual_pool_weight_sha256'] == r79_meta['fit_sample_weights_sha256']
    assert helper.values_sha256(helper.transform_target(y, X)) == r79_meta['fit_transformed_target_values_sha256']
    history.summary_equal(baseline._summary(pool_weights), r79_meta['fit_sample_weights_summary'])
    for name, value in (('date_weights', weights32), ('pool_weights', pool_weights)):
        save_array(OUT / (name + '.npy'), value)
    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    model_dir = OUT / 'model'
    model_dir.mkdir()
    fit_started, fit_tick, fit_failure = history.old.now(), time.monotonic(), None
    write(model_dir / 'fit_started.json', {'started_at_utc': fit_started, 'fit_rows': len(X), 'query_labels_used': False,
        'helper_sha256': HELPER_SHA, 'parameters': parameters, 'date_weights_sha256': prep['train_sample_weight_sha256'],
        'pool_weights_sha256': helper.values_sha256(pool_weights), 'automatic_retry': False})
    try:
        model, state, meta = helper.fit_model(X, y, G, cats, weights32)
        fit_seconds = time.monotonic() - fit_tick
        # Preserve the completed fit before query predictions and later consistency checks.
        model.save_model(str(model_dir / 'model.cbm'), format='cbm')
        write(model_dir / 'feature_state.json', state)
        write(model_dir / 'fit_metadata.json', meta)
    except Exception:
        fit_failure = traceback.format_exc()
        raise
    finally:
        fit_and_save_seconds = time.monotonic() - fit_tick
        write(model_dir / 'fit_execution.json', {'exit_code': 1 if fit_failure else 0, 'failure': fit_failure,
            'started_at_utc': fit_started, 'completed_at_utc': history.old.now(), 'elapsed_seconds': fit_and_save_seconds, 'automatic_retry': False})
    assert state['legacy_feature_state'] == legacy == r79_legacy
    assert meta['parameters'] == parameters and meta['seed'] == meta['actual_random_seed'] == 137
    assert meta['changed_parameter_names'] == ['random_seed']
    assert meta['legacy_feature_state_content_sha256'] == r79_meta['feature_state_content_sha256']
    for key in ('fit_target_values_sha256', 'fit_transformed_target_values_sha256', 'fit_effective_area_sha256',
                'fit_factor_sha256', 'fit_area_weights_sha256', 'fit_date_weights_sha256', 'fit_sample_weights_sha256',
                'fit_original_feature_values_sha256', 'fit_geo_feature_values_sha256', 'fit_model_feature_values_sha256'):
        assert meta[key] == r79_meta[key], key
    assert meta['fit_date_weights_sha256'] == prep['train_sample_weight_sha256']
    assert meta['fit_area_weights_sha256'] == legacy['fit_sample_weights_sha256']
    assert meta['fit_sample_weights_sha256'] == helper.values_sha256(pool_weights)
    history.summary_equal(meta['fit_date_weights_summary'], prep['train_sample_weight_summary'])
    fitted_capacity = helper.check_model(model, state)
    raw = helper.predict_raw(model, state, Xq, Gq, cats)
    native = helper.inverse_transform(raw, Xq)
    assert native.dtype == np.float64 and native.shape == (47073,) and np.isfinite(native).all() and (native >= 0).all()
    loaded_state = read(model_dir / 'feature_state.json')
    assert loaded_state == state
    reloaded = CatBoostRegressor()
    reloaded.load_model(str(model_dir / 'model.cbm'), format='cbm')
    loaded_capacity = helper.check_model(reloaded, loaded_state)
    for key in fitted_capacity:
        if key != 'actual_learning_rate':
            assert loaded_capacity[key] == fitted_capacity[key], key
    raw_again = helper.predict_raw(reloaded, loaded_state, Xq, Gq, cats)
    native_again = helper.inverse_transform(raw_again, Xq)
    history.old.exact(raw, raw_again, 'Full test raw unit CBM reload')
    history.old.exact(native, native_again, 'Full test native TWD CBM reload')
    save_array(model_dir / 'raw_z.npy', raw)
    save_array(model_dir / 'predictions_twd.npy', native)
    assert before == {name: helper.base.frame_sha(frame) for name, frame in (('X', X), ('Xq', Xq), ('geo', G), ('geoq', Gq))}
    assert target_sha == helper.values_sha256(y) and value_sha(weights32, '<f4') == prep['train_sample_weight_sha256']
    meta.update(round=83, stage='final', query_rows=47073, fit_seconds=fit_seconds, fit_and_save_seconds=fit_and_save_seconds, **loaded_capacity,
        fitted_native_capacity=fitted_capacity, loaded_native_capacity=loaded_capacity,
        fitted_actual_learning_rate=fitted_capacity['actual_learning_rate'],
        learning_rate_tolerance={'reference': .065, 'rtol': 1e-7, 'atol': 0, 'fit_reload_exact_required': False},
        full_query_raw_TWD_reload_exact=True, reload_rows=47073,
        original_R19_feature_state_exact=True, original_R79_legacy_feature_state_exact=True,
        R79_final_model_result_path=R79_RESULT, R79_final_model_result_sha256=R79_SHA,
        date_weight_provenance_verified_in_driver=True, R63_preparation=prep,
        input_frame_sha256=before, **canonical, text_plan_sha256=PLAN_SHA, decision_sha256=DECISION_SHA,
        source_sha256=sources, input_sha256=dict(inputs), native_negative_raw_rows=int((raw < 0).sum()),
        native_zero_TWD_rows=int((native == 0).sum()), prediction_min_TWD=float(native.min()),
        prediction_max_TWD=float(native.max()), test_labels_loaded=False,
        runtime={'catboost': catboost.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    write(model_dir / 'metadata.json', meta)
    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items())
    assert sha(PLAN) == PLAN_SHA and sha(ROOT / DECISION) == DECISION_SHA
    protected_after = {name: sha(ROOT / name) for name in protected}
    assert protected_after == protected and sha(selector) == selector_sha
    assert {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')} == set(protected) - {'runs/versions.json'}
    artifacts = {rel(p): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'run.log'}
    proof = {'round': 83, 'status': 'final_unit_model_ready_no_CSV', 'fit_rows': 467221, 'query_rows': 47073,
        'native_prediction_path': rel(model_dir / 'predictions_twd.npy'),
        'native_prediction_sha256': artifacts[rel(model_dir / 'predictions_twd.npy')],
        'raw_prediction_path': rel(model_dir / 'raw_z.npy'), 'raw_prediction_sha256': artifacts[rel(model_dir / 'raw_z.npy')],
        'model_metadata_path': rel(model_dir / 'metadata.json'), 'model_metadata_sha256': artifacts[rel(model_dir / 'metadata.json')],
        'source_sha256': sources, 'input_sha256': inputs, 'artifact_sha256': artifacts,
        'input_contract_sha256': sha(OUT / 'input_contract.json'), 'text_plan_sha256': PLAN_SHA, 'decision_sha256': DECISION_SHA,
        'input_frame_sha256': before, **canonical, 'fit_target_values_sha256': target_sha,
        'full_query_raw_TWD_reload_exact': True, 'reload_rows': 47073, 'new_fits': 1, 'new_NN_fits': 0,
        'new_CSV': False, 'existing_models_loaded': 0, 'new_model_reload_calls': 1,
        'native_model_predict_calls': 2, 'reference_predict_calls': 0, 'spatial_predict_calls': 0,
        'query_labels_used': False, 'test_labels_loaded': False, 'original_R19_feature_state_exact': True,
        'original_R79_legacy_feature_state_exact': True,
        'legacy_feature_state_content_sha256': legacy['state_content_sha256'],
        'R79_final_model_result_path': R79_RESULT, 'R79_final_model_result_sha256': R79_SHA,
        'seed': 137, 'actual_random_seed': loaded_capacity['actual_random_seed'],
        'parameters': parameters, 'baseline_parameters': r79_meta['parameters'],
        'changed_parameter_names': ['random_seed'], 'seed42_model_loaded_or_retrained': False,
        'date_weights_recomputed': False, 'date_weights_renormalized': False,
        'train_date_weight_sha256': prep['train_sample_weight_sha256'], 'actual_pool_weight_sha256': meta['fit_sample_weights_sha256'],
        'prediction_units': 'TWD', 'prediction_min_TWD': float(native.min()), 'prediction_max_TWD': float(native.max()),
        'native_zero_rows': int((native == 0).sum()), 'fit_seconds': fit_seconds,
        'elapsed_seconds': time.monotonic() - tick, 'protected_before': protected, 'protected_after': protected_after,
        'selector_sha256_before': selector_sha, 'selector_sha256_after': sha(selector),
        'selector_updated': False, 'registry_updated': False, 'uploaded': False}
    write(OUT / 'results.json', proof)
    print(json.dumps({'status': proof['status'], 'fit_seconds': fit_seconds, 'elapsed_seconds': proof['elapsed_seconds'],
        'native_prediction_sha256': proof['native_prediction_sha256'], 'results_sha256': sha(OUT / 'results.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert args.plan_sha == PLAN_SHA == sha(PLAN) and not OUT.exists(), 'Plan changed or earlier attempt exists'
    OUT.mkdir(parents=True, exist_ok=False)
    started, tick, code, failure = history.old.now(), time.monotonic(), 1, None
    write(OUT / 'started.json', {'command': sys.argv, 'started_at_utc': started, 'source_sha256': sha(Path(__file__)),
        'text_plan_sha256': PLAN_SHA, 'decision_sha256': DECISION_SHA, 'automatic_retry': False})
    with (OUT / 'run.log').open('x', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                run(args.plan_sha)
                code = 0
            except Exception:
                failure = traceback.format_exc()
                print(failure, flush=True)
    execution = {'command': sys.argv, 'exit_code': code, 'failure': failure,
        'started_at_utc': started, 'completed_at_utc': history.old.now(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'text_plan_sha256': PLAN_SHA, 'decision_sha256': DECISION_SHA,
        'log_path': rel(OUT / 'run.log'), 'log_sha256': sha(OUT / 'run.log'), 'automatic_retry': False}
    write(OUT / 'execution.json', execution)
    print(json.dumps(execution), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

