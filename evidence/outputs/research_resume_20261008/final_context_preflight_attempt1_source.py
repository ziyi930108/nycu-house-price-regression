"""Read-only full-fit context and saved-vector v22 replay for an R107 candidate.

No fitting, native model loading/inference, scoring, CSV writing or selection.
specification() identifies every directly read input before price/array loading.
A later full-fit request must freeze this specification and separate gate/plan.
"""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
import round100_contract as c
import round79_final as f79

ROOT = c.ROOT
ANCHORS = {
    'outputs/round79/final_export/release.json': '56440a97e73fbd5ba29e9235a63982a8b2a4af445383c88d20f973f8deca731d',
    'outputs/round79/final/trial_release.json': '376d33d9a3e5f6d15f8369b70d808b265d83320c11bd85c11b137b3406b4b626',
    'outputs/round79/final_model/results.json': '4b7172156359f476add69af11f9d2f53c78b3879bb469acf34dbd072c0db7942',
    'outputs/round71/final/metadata.json': 'e1474af8a13d573c8fb342bdda7165682f2374493070c9212de717ea86fcacb5',
    'outputs/round63/final/trial_release.json': '55e8fcffc1aef16dd056652191aff5363fa99332db97bcfd013956e88ce5b9a1',
    'outputs/round55/final/trial_release.json': 'e3b3f4333a4d8aa056728819cb2853e8c296f5d2dd5f03e1a2f067e612badb0d'}
GEO = 'outputs/round6/geography/final/'
REFERENCE = 'outputs/round2/neighborhood/final_reference.pkl'
SPATIAL = 'outputs/round39/trial/artifacts/spatial_state.pkl'
SUMMARY = 'outputs/round39/trial/artifacts/state_summary.json'
GEO_MANIFEST = 'outputs/round39/geography/final/manifest.json'
PREPARED = 'outputs/round63/final/prepared.npz'
PREPARATION = 'outputs/round63/final/preparation.json'
TRAINING = 'outputs/round63/final/model/training.json'
CSV = 'outputs/submission_v22_trial.csv'
VECTORS = {
    'area': 'outputs/round55/final/model/predictions_twd.npy',
    'prior': 'outputs/round63/final/model/predictions_twd.npy',
    'lgb': 'outputs/round2/final_predictions/lightgbm_l1_unit63_test.npy',
    'log': 'outputs/round46/trial/recency_log_spatial.npy',
    'unit_native': 'outputs/round79/final_model/model/predictions_twd.npy',
    'unit_base': 'outputs/round79/final/new_unit/base.npy',
    'unit_reference': 'outputs/round79/final/new_unit/reference.npy',
    'unit': 'outputs/round79/final/new_unit/spatial.npy',
    'nn_mean': 'outputs/round71/final/native_mean.npy',
    'nn_base': 'outputs/round71/final/base.npy',
    'nn_reference': 'outputs/round71/final/reference.npy',
    'nn': 'outputs/round71/final/nn_spatial.npy',
    'v22': 'outputs/round79/final/predictions.npy'}


def source_hashes():
    return c.source_hashes(__file__)


def specification():
    """Verify metadata and all direct-input hashes before numerical loading."""
    inputs, known, docs, inherited_sources = {}, {}, {}, {}

    def merge(dst, mapping):
        for name, digest in mapping.items():
            name = name.replace(chr(92), '/')
            c.need(name not in dst or dst[name] == digest, 'Conflicting input identity: ' + name)
            dst[name] = digest

    def bind(name, digest):
        c.need(name not in inputs or inputs[name] == digest, 'Conflicting bound identity: ' + name)
        if name not in inputs:
            c.need(c.sha(name) == digest, 'Immutable final input changed: ' + name)
        inputs[name] = digest
        return c.safe(name)

    for name, digest in ANCHORS.items():
        docs[name] = c.read(bind(name, digest))
        merge(known, docs[name].get('input_sha256', {}))
        merge(known, docs[name].get('artifact_sha256', {}))
        merge(inherited_sources, docs[name].get('source_sha256', {}))
    release = docs['outputs/round79/final_export/release.json']
    p79 = docs['outputs/round79/final/trial_release.json']
    model = docs['outputs/round79/final_model/results.json']
    p63 = docs['outputs/round63/final/trial_release.json']
    p55 = docs['outputs/round55/final/trial_release.json']
    p71 = docs['outputs/round71/final/metadata.json']
    c.need(release['candidate_proof_sha256'] == ANCHORS[release['candidate_proof_path']]
           and p79['model_result_sha256'] == ANCHORS[p79['model_result_path']], 'Final authority chain differs')
    c.need(release['path'] == CSV and release['rows'] == 47073 and release['version'] == 22,
           'v22 CSV authority differs')
    c.need(all(d['fit_rows'] == 467221 and d['query_rows'] == 47073
               for d in (p79, model, p63, p55, p71)), 'Final row authority differs')
    c.need(model['input_frame_sha256'] == p63['input_frame_sha256'] == p55['input_frame_sha256']
           == p71['input_frame_sha256'], 'Final feature-frame authorities disagree')
    c.need(model['fit_target_values_sha256'] == p63['fit_target_values_sha256']
           == p55['fit_target_values_sha256'] == p71['fit_target_values_sha256_inherited'],
           'Final target authorities disagree')
    merge(known, ANCHORS)
    merge(known, {CSV: release['sha256'], release['sample_ID_path']: release['sample_ID_sha256']})
    required = [f'data_cache/{s}_features.pkl' for s in ('train', 'valid', 'test')]
    required += ['data_cache/train.pkl', 'data_cache/valid.pkl', 'reports/feature_schema.json']
    required += [GEO + s + '_source_cache_indices.npy' for s in ('train', 'valid', 'full', 'test')]
    required += [GEO + 'full.pkl', GEO + 'test.pkl', GEO_MANIFEST, REFERENCE, SPATIAL, SUMMARY,
                 PREPARED, PREPARATION, TRAINING, CSV, release['sample_ID_path']]
    required += [f'outputs/round63/final/{s}_indices.npy' for s in ('fit', 'query')]
    required += list(VECTORS.values())
    for name in required:
        c.need(name in known, 'Direct final input not in immutable authority: ' + name)
        bind(name, known[name])
    manifest = c.read(GEO_MANIFEST)
    query_geo = manifest['features']['query']['path']
    bind(query_geo, manifest['features']['query']['sha256'])
    prep, training = c.read(PREPARATION), c.read(TRAINING)
    c.need(prep['prepared_path'] == PREPARED and prep['prepared_sha256'] == inputs[PREPARED]
           == training['input_sha256'] and prep == p63['sample_weight_preparation'],
           'Original final NN archive/preparation identity differs')
    c.need(training['status'] == 'completed' and prep['fit_rows'] == 467221
           and prep['query_rows'] == 47073 and prep['numeric_count'] == 144
           and not prep['query_labels_used'] and not prep['encoder_refit'] and not prep['prior_refit']
           and prep['train_sample_weight_sha256'] == training['train_sample_weight_sha256'],
           'Original final prior/encoder/date-weight scope differs')
    c.verify_hashes(inherited_sources)
    return {'format': 'R107_readonly_final_context_v1', 'fit_rows': 467221, 'query_rows': 47073,
            'input_sha256': inputs, 'source_sha256': source_hashes(),
            'inherited_source_sha256': inherited_sources,
            'sample_path': release['sample_ID_path'], 'query_geo_path': query_geo,
            'fit_target_values_sha256': model['fit_target_values_sha256'],
            'input_frame_sha256': model['input_frame_sha256'],
            'prepared': {'path': PREPARED, 'sha256': inputs[PREPARED]},
            'preparation': {'path': PREPARATION, 'sha256': inputs[PREPARATION]},
            'original_training': {'path': TRAINING, 'sha256': inputs[TRAINING]},
            'final_CSV': {'path': CSV, 'sha256': release['sha256']},
            'metadata_only_specification': True, 'NPZ_arrays_or_prices_deserialized': False,
            'new_fits': 0, 'new_CSV': False, 'Private_score': None}


def _vector(name):
    value = np.load(c.safe(name), allow_pickle=False)
    c.need(value.dtype == np.float64 and value.shape == (47073,)
           and np.isfinite(value).all() and (value >= 0).all(), 'Invalid saved TWD vector: ' + name)
    return value


def _index(name, bound, rows, identity=False):
    value = np.load(c.safe(name), allow_pickle=False)
    c.need(value.dtype == np.int64 and value.shape == (rows,)
           and (np.diff(value) > 0).all() and value.min() >= 0 and value.max() < bound,
           'Canonical index schema differs: ' + name)
    if identity:
        c.exact(value, np.arange(rows, dtype=np.int64), 'Expected identity indices: ' + name)
    return value


def _state_identity(reference, spatial_state):
    return hashlib.sha256(pickle.dumps((reference, spatial_state), protocol=5)).hexdigest()


def _frame_identities(context):
    return {k: f79.spatial.frame_sha(context[k]) for k in ('X', 'Xq', 'G', 'Gq', 'query_geo')}


def apply_prior(context, native):
    """Apply original v22 NN-slot formula/R/S/median; zero fits or writes."""
    c.need(native.dtype == np.float64 and native.shape == (47073,)
           and np.isfinite(native).all() and (native >= 0).all(), 'Invalid prior NN total-price vector')
    fixed = context['vectors']
    base = .75 * (.5 * fixed['area'] + .5 * native) + .25 * fixed['lgb']
    reference, diagnostics, _ = f79.exp.original.predict_reference(
        context['reference'], context['Xq'], base, variant='trend', return_diagnostics=True)
    spatial, detail, _ = f79.spatial.predict(context['spatial_state'], context['Xq'],
        context['query_geo'], base, reference, diagnostics)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback], reference[fallback], 'NN-slot spatial fallback differs')
    median = np.median(np.column_stack([spatial, fixed['log'], fixed['unit']]), axis=1)
    return {'native': native, 'base': base, 'reference': reference, 'spatial': spatial,
            'median': median}, diagnostics, detail


def verify_unchanged(context, rehash=True):
    c.need(_frame_identities(context) == context['frame_identities'], 'Final X/geography mutated')
    c.need(_state_identity(context['reference'], context['spatial_state']) == context['state_identity'],
           'Final reference/spatial state mutated')
    c.need(c.values_sha(context['y']) == context['target_identity'], 'Final fit targets mutated')
    c.need({k: c.values_sha(v) for k, v in context['vectors'].items()} == context['vector_identities'],
           'Saved final vectors mutated')
    if rehash:
        c.verify_hashes(context['specification']['input_sha256'])
        c.verify_hashes(context['specification']['source_sha256'])
        c.verify_hashes(context['specification']['inherited_source_sha256'])


def load_context(spec=None):
    """Load only a fully bound original full-fit context and replay v22 exactly."""
    spec = specification() if spec is None else spec
    c.need(spec == specification(), 'Caller-bound final specification differs')
    c.verify_hashes(spec['input_sha256'])
    c.verify_hashes(spec['source_sha256'])
    frames = {s: pd.read_pickle(c.safe(f'data_cache/{s}_features.pkl')) for s in ('train', 'valid', 'test')}
    schema = c.read('reports/feature_schema.json')
    for split, rows in [('train', 455784), ('valid', 11994), ('test', 47073)]:
        frame = frames[split]
        c.need(len(frame) == rows and frame.index.equals(pd.RangeIndex(rows))
               and list(frame) == schema['columns'] and len(frame.columns) == 87
               and '總價元' not in frame, 'Original final feature-cache schema differs')
    kt = _index(GEO + 'train_source_cache_indices.npy', 455784, 455255)
    kv = _index(GEO + 'valid_source_cache_indices.npy', 11994, 11994, True)
    kf = _index(GEO + 'full_source_cache_indices.npy', 467249, 467221)
    kq = _index(GEO + 'test_source_cache_indices.npy', 47073, 47073, True)
    X = pd.concat([frames['train'].iloc[kt], frames['valid'].iloc[kv]], ignore_index=True).iloc[kf].reset_index(drop=True)
    Xq = frames['test'].iloc[kq].reset_index(drop=True)
    del frames
    labels = {s: pd.read_pickle(c.safe(f'data_cache/{s}.pkl')) for s in ('train', 'valid')}
    for split, rows in [('train', 455784), ('valid', 11994)]:
        c.need(len(labels[split]) == rows and labels[split].index.equals(pd.RangeIndex(rows)),
               'Raw fit-label cache row axis differs')
    y = np.concatenate([labels['train'].iloc[kt]['總價元'].to_numpy(dtype=np.float64),
                        labels['valid'].iloc[kv]['總價元'].to_numpy(dtype=np.float64)])[kf]
    del labels
    c.need(y.shape == (467221,) and np.isfinite(y).all() and (y > 0).all()
           and f79.exp.original.value_sha(y, '<f8') == spec['fit_target_values_sha256'],
           'Canonical final fit target identity differs')
    G, Gq = [pd.read_pickle(c.safe(GEO + name)) for name in ('full.pkl', 'test.pkl')]
    c.need(len(G) == len(X) and len(Gq) == len(Xq) and G.index.equals(X.index)
           and Gq.index.equals(Xq.index), 'Already-canonical R6 geography alignment differs')
    expected_frames = {'X': f79.spatial.frame_sha(X), 'Xq': f79.spatial.frame_sha(Xq),
                       'geo': f79.spatial.frame_sha(G), 'geoq': f79.spatial.frame_sha(Gq)}
    c.need(expected_frames == spec['input_frame_sha256'], 'Canonical final frames differ from v22')
    fi = _index('outputs/round63/final/fit_indices.npy', 467221, 467221, True)
    qi = _index('outputs/round63/final/query_indices.npy', 47073, 47073, True)
    manifest = c.read(GEO_MANIFEST)
    query_geo = pd.read_pickle(c.safe(spec['query_geo_path']))
    c.need(query_geo.index.equals(Xq.index) and query_geo.shape == (47073, 8)
           and f79.spatial.frame_sha(query_geo) == manifest['features']['query']['frame_sha256']
           and query_geo.dtypes.equals(Gq.dtypes) and list(query_geo) == list(Gq),
           'Frozen R39 query geography differs')
    with c.safe(REFERENCE).open('rb') as f:
        reference = pickle.load(f)
    with c.safe(SPATIAL).open('rb') as f:
        spatial_state = pickle.load(f)
    summary = c.read(SUMMARY)
    c.need(spatial_state['state_summary'] == summary and summary['fit_rows'] == len(X)
           and summary['fit_X_sha256'] == expected_frames['X'] and not summary['query_used_in_build']
           and summary['fit_geo_sha256'] == manifest['features']['fit']['frame_sha256']
           and summary['parameters'] == f79.spatial.parameters()
           and reference['fit_report']['fitting_rows'] == len(X), 'Final fit-only R/S state differs')
    c.need(summary['existing_reference_residual_sha256'] ==
           f79.exp.original.value_sha(reference['reference_residual'], '<f8'), 'Final R/S residual binding differs')
    vectors = {name: _vector(path) for name, path in VECTORS.items()}
    sample = pd.read_csv(c.safe(spec['sample_path']), encoding='utf-8-sig',
                         dtype={'編號': str}, keep_default_na=False)
    c.need(list(sample.columns) == ['編號', '總價元'] and len(sample) == 47073
           and sample['編號'].is_unique and sample['編號'].map(bool).all(), 'Sample ID/schema differs')
    context = {'specification': spec, 'X': X, 'Xq': Xq, 'y': y, 'G': G, 'Gq': Gq,
               'cats': schema['categorical_columns'], 'query_geo': query_geo,
               'reference': reference, 'spatial_state': spatial_state, 'spatial_summary': summary,
               'vectors': vectors, 'ids': sample['編號'].tolist(),
               'fit_indices': fi, 'query_indices': qi,
               'source_indices': {'train': kt, 'valid': kv, 'full': kf, 'test': kq},
               'prepared': spec['prepared'], 'target_identity': c.values_sha(y),
               'vector_identities': {k: c.values_sha(v) for k, v in vectors.items()},
               'state_identity': _state_identity(reference, spatial_state)}
    context['frame_identities'] = _frame_identities(context)
    replay, _, _ = apply_prior(context, vectors['prior'])
    checks = {}

    def exact(name, left, right):
        c.exact(left, right, name + ' replay differs')
        checks[name] = True

    exact('NN_native_mean', .5 * vectors['area'] + .5 * vectors['prior'], vectors['nn_mean'])
    for layer, saved in [('base', 'nn_base'), ('reference', 'nn_reference'), ('spatial', 'nn'), ('median', 'v22')]:
        exact('v22_NN_' + layer, replay[layer], vectors[saved])
    base = .75 * vectors['unit_native'] + .25 * vectors['lgb']
    ref, diagnostics, _ = f79.exp.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
    unit, detail, _ = f79.spatial.predict(spatial_state, Xq, query_geo, base, ref, diagnostics)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    exact('unit_spatial_fallback', unit[fallback], ref[fallback])
    for name, value, saved in [('base', base, 'unit_base'), ('reference', ref, 'unit_reference'), ('spatial', unit, 'unit')]:
        exact('v22_unit_' + name, value, vectors[saved])
    exact('v22_full_unrounded_median', np.median(np.column_stack([replay['spatial'], vectors['log'], unit]), axis=1), vectors['v22'])
    rounded = np.maximum(np.rint(vectors['v22']), 1).astype(np.int64)
    with c.safe(CSV).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        c.need(next(reader) == ['編號', '總價元'], 'v22 CSV headers differ')
        rows = list(reader)
    c.need(len(rows) == len(context['ids']) and all(len(row) == 2 for row in rows)
           and [row[0] for row in rows] == context['ids']
           and [row[1] for row in rows] == [str(value) for value in rounded],
           'v22 full sample-ID/rounded-price CSV replay differs')
    c.need(c.sha(CSV) == spec['final_CSV']['sha256'], 'v22 whole CSV SHA differs')
    checks['v22_CSV_all_IDs_prices_order_and_whole_SHA'] = True
    context['replay_checks'] = checks
    verify_unchanged(context)
    return context


def run_preflight(out):
    tick = time.monotonic()
    out = c.safe(out)
    c.need(not out.exists() and not out.with_suffix('.sha256').exists(), 'Preserve existing final preflight receipt')
    spec = specification()
    context = load_context(spec)
    receipt = {'status': 'passed_readonly_final_context_and_original_v22_replay',
        'created_at_utc': c.now(), 'specification': spec, 'input_sha256': spec['input_sha256'],
        'source_sha256': spec['source_sha256'], 'fit_rows': len(context['X']), 'query_rows': len(context['Xq']),
        'frame_sha256': context['frame_identities'], 'fit_target_values_sha256': context['target_identity'],
        'canonical_indices': {k: c.identity(v) for k, v in context['source_indices'].items()},
        'replay_checks': context['replay_checks'], 'prepared_NPZ_hashed_not_deserialized': True,
        'train_and_valid_whole_label_pickles_deserialized_for_final_fit_labels': True,
        'test_true_prices_loaded': False, 'query_or_test_labels_used': False,
        'saved_native_vector_composition_replay_not_six_model_native_inference': True,
        'new_fits': 0, 'native_models_loaded_or_predicted': 0,
        'reference_predict_calls': 2, 'spatial_predict_calls': 2, 'new_R_S_states': 0,
        'new_CSV': False, 'Kaggle_uploaded_or_selection_changed': False,
        'Public_score': 1998568.5706, 'Private_score': None,
        'elapsed_seconds': time.monotonic() - tick}
    out.parent.mkdir(parents=True, exist_ok=True)
    c.write(out, receipt)
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.preflight)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'checks': result['replay_checks'], 'new_fits': 0, 'new_CSV': False}, ensure_ascii=False))
