"""Read-only R97 offset context and v22 native-slot composition for R113.

Only saved fit labels are loaded. No raw price pickle, query labels, fitting,
native-model inference, scoring, CSV authoring or old-source modification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round97_offset_experiment as old
import round90_offset_data as data_helper
import round90_offset_nn as neural
import round2_neighborhood as reference
import round8_spatial_reference as spatial

PURE = 'outputs/round90/preparation/manifest.json'
PURE_SHA = '400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7'
SEAL = 'outputs/round97/sealed_manifest.json'
SEAL_SHA = '429e5405652b4e9d54308d6f09d6722ab540694b13f92e2ad8577277b85893cb'
LABELS = 'outputs/round97_inputs/manifest.json'
LABELS_SHA = 'e1250558e16b463ad8eca37eda324b3b65deaa3fe014fa6668d7aba55e098a4b'
REF = 'outputs/round97/reference_spatial/reference.pkl'
SPATIAL = 'outputs/round97/reference_spatial/spatial.pkl'
ROWS, QUERY_ROWS = 272117, 107169
LAYERS = {'native', 'base', 'reference', 'spatial', 'median'}


def source_hashes():
    return c.source_hashes(__file__)


def specification():
    """Bind metadata and byte hashes before any feature/label/NPZ/state loading."""
    inputs = {}
    def bind(path, digest):
        path = c.rel(path)
        c.need(path != 'data_cache/train.pkl' and not path.startswith('data_cache/valid')
               and not path.startswith('data_cache/test'), 'No raw/query price dependency')
        c.need(path not in inputs or inputs[path] == digest, 'Conflicting R97 input authority: ' + path)
        c.need(c.sha(path) == digest, 'Immutable R97 artifact changed: ' + path)
        inputs[path] = digest
    for name, digest in ((PURE, PURE_SHA), (SEAL, SEAL_SHA), (LABELS, LABELS_SHA)):
        bind(name, digest)
    pure, seal, labels = c.read(PURE), c.read(SEAL), c.read(LABELS)
    c.need(pure['status'] == 'prepared_target_free' and pure['parameters'] == data_helper.PARAMETERS
           and pure['fit_rows'] == seal['fit_rows'] == ROWS and pure['query_rows'] == seal['query_rows'] == QUERY_ROWS,
           'Fixed R97 origin/row preparation')
    c.need(seal['status'] == 'sealed_predictions_without_query_labels' and seal['all_fits_complete']
           and seal['predictions_sealed'] and seal['preparation_manifest'] == {'path': PURE, 'sha256': PURE_SHA}
           and seal['fit_label_manifest'] == {'path': LABELS, 'sha256': LABELS_SHA}
           and seal['formulas'] == old.FORMULAS and seal['cutoff'] == '2023-02-11'
           and seal['query_end'] == '2024-02-10' and seal['scores'] is None
           and seal['query_labels_available'] is False and seal['new_query_metrics_computed'] is False,
           'Original R97 sealed unscored prediction authority')
    c.need(labels['status'] == 'fit_labels_sealed_no_training' and labels['fit_rows'] == ROWS
           and labels['pure_X_manifest_sha256'] == PURE_SHA and labels['query_labels_exported'] is False,
           'Fit-only label export authority')
    for name, digest in pure['input_sha256'].items():
        bind(name, digest)
    for item in pure['artifacts'].values():
        bind(item['path'], item['sha256'])
    for name, digest in seal['artifact_sha256'].items():
        bind(name, digest)
    # The old seal declares raw train solely for its later scorer. Do not hash/open it.
    c.need(seal['declared_only_input_paths_not_opened_by_driver'] == ['data_cache/train.pkl'], 'Raw declaration boundary')
    for name, digest in seal['input_sha256'].items():
        if name != 'data_cache/train.pkl':
            bind(name, digest)
    bind(labels['fit_y_path'], labels['fit_y_sha256'])
    for name, digest in labels['indices_sha256'].items():
        bind(name, digest)
    prepared = seal['NN_preparation']
    c.need(prepared == c.read(prepared['prepared_metadata']) and prepared['prior_numeric_width'] == 144
           and prepared['area_numeric_width'] == 599 and len(prepared['cardinalities']) == 23
           and prepared['source_indices_contract'] == neural.SOURCE_INDEX_CONTRACT
           and prepared['chronological_oof'] is False and prepared['query_labels_available'] is False,
           'Original prior 144/23, not area 599; fixed canonical group-OOF preparation')
    for mapping in (pure['source_sha256'], seal['source_sha256'], prepared['source_sha256']):
        for name, digest in mapping.items():
            bind(name, digest)
    c.need(REF in inputs and SPATIAL in inputs, 'Saved R97 reference/spatial states must be sealed')
    return {'format': 'R113_readonly_offset_context_v1', 'input_sha256': inputs,
            'source_sha256': source_hashes(), 'pure_manifest': {'path': PURE, 'sha256': PURE_SHA},
            'original_seal': {'path': SEAL, 'sha256': SEAL_SHA}, 'fit_labels': {'path': LABELS, 'sha256': LABELS_SHA},
            'prior_npz': {'path': prepared['prior_npz'], 'sha256': inputs[prepared['prior_npz']]},
            'fit_rows': ROWS, 'query_rows': QUERY_ROWS, 'origin': '2023-02-11', 'query_end': '2024-02-10',
            'metadata_and_hashes_only': True, 'raw_price_table_opened_or_rehashed': False,
            'query_prices_read': False, 'new_fits': 0, 'new_CSV': False}


def _states_identity(ctx):
    return hashlib.sha256(pickle.dumps((ctx['reference'], ctx['spatial_state']), protocol=5)).hexdigest()


def _data_identity(data):
    return {**{name: old.frame_sha(getattr(data, name)) for name in data_helper.FRAME_NAMES},
            **{name: c.values_sha(getattr(data, name)) for name in data_helper.ARRAY_NAMES}}


def verify_unchanged(ctx, rehash=True):
    c.need(_states_identity(ctx) == ctx['state_identity'], 'R97 fit reference/spatial state mutated')
    c.need(_data_identity(ctx['data']) == ctx['data_identity'], 'Offset pure-X frames/indices mutated')
    c.need(c.identity(ctx['yfit']) == ctx['fit_label_identity'] and c.identity(ctx['weights']) == ctx['weight_identity'],
           'Offset fit labels or date weights mutated')
    actual = {key: c.identity(value) for key, value in ctx['baseline']['native'].items()}
    actual.update({key: c.identity(ctx['baseline'][key]) for key in ('v22', 'v24')})
    for branch, layers in ctx['baseline']['branches'].items():
        actual.update({branch + '/' + key: c.identity(value) for key, value in layers.items()})
    c.need(actual == ctx['vector_identity'], 'Saved R97 baseline vectors mutated')
    if rehash:
        c.verify_hashes(ctx['specification']['input_sha256'])
        c.verify_hashes(ctx['specification']['source_sha256'])


def _apply(ctx, native, base, others):
    data = ctx['data']
    old.prices(native, len(data.Xquery), 'candidate native')
    old.prices(base, len(data.Xquery), 'candidate base')
    before = _states_identity(ctx)
    ref, diagnostics, _ = reference.predict_reference(ctx['reference'], data.Xquery, base,
                                                     variant='trend', return_diagnostics=True)
    result, detail, _ = spatial.predict(ctx['spatial_state'], data.Xquery, data.spatial_query,
                                       base, ref, diagnostics)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    c.exact(result[fallback], ref[fallback], 'Offset spatial fallback exact')
    median = np.median(np.column_stack([result, *others]), axis=1)
    c.need(_states_identity(ctx) == before == ctx['state_identity'], 'Prediction mutated saved fit states')
    values = {'native': native, 'base': base, 'reference': ref, 'spatial': result, 'median': median}
    for key, value in values.items():
        old.prices(value, len(data.Xquery), 'candidate ' + key)
    return values


def chain_prior(ctx, new_native):
    fixed = ctx['baseline']
    base = .75 * (.5 * fixed['native']['N_area'] + .5 * new_native) + .25 * fixed['native']['light']
    return _apply(ctx, new_native, base, [fixed['branches']['H']['spatial'], fixed['branches']['U22']['spatial']])


def chain_unit(ctx, new_native):
    fixed = ctx['baseline']
    base = .75 * new_native + .25 * fixed['native']['light']
    return _apply(ctx, new_native, base, [fixed['branches']['N']['spatial'], fixed['branches']['H']['spatial']])


def load_context(spec=None):
    spec = specification() if spec is None else spec
    c.need(spec == specification(), 'Caller-frozen offset specification differs')
    seal = c.read(SEAL)
    data = data_helper.load_prepared(c.safe(PURE).parent, PURE_SHA)
    series, label_metadata = old.load_fit_labels(data, LABELS_SHA)
    yfit = series.to_numpy(dtype=np.float64, copy=True)
    prepared = seal['NN_preparation']
    weights = old.verify_nn_preparation(prepared, data, series, neural)
    for name in data_helper.ARRAY_NAMES:
        c.exact(c.load_array(seal['arrays'][name]), getattr(data, name), 'R90/R97 offset axis ' + name)
    with np.load(c.safe(prepared['prior_npz']), allow_pickle=False) as z:
        c.need(set(z.files) == set(c.ARRAY_NAMES) and len(z.files) == len(c.ARRAY_NAMES), 'Only original nine prior arrays')
        arrays = {key: z[key] for key in c.ARRAY_NAMES}
    c.need({key: c.identity(value) for key, value in arrays.items()} == prepared['prior_arrays'], 'All nine prior array identities')
    c.need(arrays['train_num'].shape == (ROWS, 144) and arrays['query_num'].shape == (QUERY_ROWS, 144)
           and arrays['train_cat'].shape == (ROWS, 23) and arrays['query_cat'].shape == (QUERY_ROWS, 23)
           and arrays['train_num'].dtype == arrays['query_num'].dtype == np.float32
           and arrays['train_cat'].dtype == arrays['query_cat'].dtype == np.int64
           and arrays['cardinalities'].dtype == np.int64, 'Original 144/23 prior contract')
    for key in ('train_num', 'query_num'):
        c.need(np.isfinite(arrays[key]).all(), 'Finite encoded prior numeric input')
    c.exact(arrays['train_y_million'], np.asarray(yfit / 1e6, dtype=np.float32), 'Fit-only NN target alignment')
    c.exact(arrays['train_sample_weight'], weights, 'Original saved float32 date weights')
    encoder = c.read(prepared['encoder_state'])
    prior_state = c.read(prepared['prior_state'])
    evidence = c.read(prepared['prior_evidence'])
    date_meta = c.read(prepared['date_weights_metadata'])
    c.need(encoder['fit_rows'] == ROWS and encoder['numeric_output_width'] == 144
           and encoder['categorical_output_width'] == 23 and not encoder['query_used_for_state']
           and not encoder['target_used_for_state'] and encoder['cardinalities'] == prepared['cardinalities'],
           'Fit-only original encoder scope')
    c.need(prior_state['fit_rows'] == ROWS and prior_state['uses_fit_labels'] and not prior_state['uses_query_labels']
           and evidence['uses_fit_labels'] and not evidence['uses_query_labels'] and not evidence['chronological_oof']
           and evidence['fit_target_sha256'] == c.values_sha(yfit)
           and evidence['source_indices_sha256'] == c.values_sha(data.fit_indices), 'Original fit group-OOF prior scope')
    c.need(date_meta['reference_rows'] == ROWS and date_meta['saved_dtype'] == 'float32'
           and date_meta['saved_values_sha256'] == c.values_sha(weights)
           and date_meta['renormalized_after_cast'] is False, 'Saved fit-only recency metadata')
    # Pure transform/replay only: do not refit any encoder, prior or date state.
    xf = neural.geo.augment_features(data.Xfit, data.tree_fit)
    xq = neural.geo.augment_features(data.Xquery, data.tree_query)
    for role, frame in [('train', xf), ('query', xq)]:
        numeric, categorical = neural.encoder.encode(encoder, frame)
        c.exact(numeric, arrays[role + '_num'], 'Frozen encoder ' + role + ' numeric replay')
        c.exact(categorical, arrays[role + '_cat'], 'Frozen encoder ' + role + ' category replay')
    oof = np.load(c.safe(prepared['train_oof_unit_prior']), allow_pickle=False)
    query_prior = np.load(c.safe(prepared['query_unit_prior']), allow_pickle=False)
    c.need(c.values_sha(oof) == evidence['oof_prior_sha256'], 'Original OOF prior value identity')
    c.exact(neural.prior.predict_prior(prior_state, data.Xquery), query_prior, 'Frozen full-fit query prior replay')
    c.exact(np.asarray(neural.effective_area(data.Xfit) * oof / 1e6, dtype=np.float32),
            arrays['train_price_factor_million'], 'OOF fit price factor replay')
    c.exact(np.asarray(neural.effective_area(data.Xquery) * query_prior / 1e6, dtype=np.float32),
            arrays['query_price_factor_million'], 'Full-fit query price factor replay')
    folds = np.load(c.safe(prepared['fold_ids']), allow_pickle=False)
    c.need(c.values_sha(folds) == evidence['fold_ids_sha256'] and set(np.unique(folds)) == set(range(5)), 'Original five fold identities')
    native = {key: c.load_array(rec) for key, rec in seal['native_predictions'].items()}
    branches = {key: {layer: c.load_array(rec) for layer, rec in value.items()}
                for key, value in seal['branches'].items()}
    baseline = {key: c.load_array(rec) for key, rec in seal['predictions'].items()}
    baseline.update(native=native, branches=branches)
    with c.safe(REF).open('rb') as f:
        ref = pickle.load(f)
    with c.safe(SPATIAL).open('rb') as f:
        sp = pickle.load(f)
    c.need(ref['fit_report']['fitting_rows'] == ROWS and not ref['fit_report']['validation_targets_used_for_references']
           and sp['state_summary']['query_used_in_build'] is False, 'Saved fit-only reference/spatial scope')
    ctx = {'data': data, 'yfit': yfit, 'weights': weights, 'prior_npz': c.safe(prepared['prior_npz']),
           'baseline': baseline, 'reference': ref, 'spatial_state': sp, 'input_sha256': spec['input_sha256'],
           'specification': spec, 'spec': spec, 'prepared': prepared, 'prepared_array_identities': prepared['prior_arrays'],
           'fit_label_metadata': label_metadata, 'replay_checks': {}}
    ctx['state_identity'], ctx['data_identity'] = _states_identity(ctx), _data_identity(data)
    ctx['fit_label_identity'], ctx['weight_identity'] = c.identity(yfit), c.identity(weights)
    ctx['vector_identity'] = {key: c.identity(value) for key, value in native.items()}
    ctx['vector_identity'].update({key: c.identity(baseline[key]) for key in ('v22', 'v24')})
    for branch, layers in branches.items():
        ctx['vector_identity'].update({branch + '/' + key: c.identity(value) for key, value in layers.items()})
    bases, outputs = old.compose_bases(native), {}
    for branch in old.BRANCH_KEYS:
        replay = _apply(ctx, native['N_prior'], bases[branch], [])
        for layer in ('base', 'reference', 'spatial'):
            c.exact(replay[layer], branches[branch][layer], 'Original R97 ' + branch + '/' + layer)
            ctx['replay_checks'][branch + '/' + layer] = True
        outputs[branch] = replay['spatial']
    for key, prediction in old.compose_final(outputs).items():
        c.exact(prediction, baseline[key], 'Original R97 final median ' + key)
        ctx['replay_checks'][key] = True
    c.exact(chain_prior(ctx, native['N_prior'])['median'], baseline['v22'], 'Original prior substitution replays v22')
    c.exact(chain_unit(ctx, native['unit42'])['median'], baseline['v22'], 'Original unit substitution replays v22')
    ctx['replay_checks'].update(prior_slot_v22=True, unit_slot_v22=True)
    verify_unchanged(ctx)
    return ctx


def context():
    return load_context()


def self_check():
    native = {key: np.array([10., 20., 30.], dtype=np.float64) + 10 * i for i, key in enumerate(old.NATIVE_KEYS)}
    bases = old.compose_bases(native)
    c.exact(bases['N'], .75 * (.5 * native['N_area'] + .5 * native['N_prior']) + .25 * native['light'], 'Synthetic prior base')
    c.exact(bases['U22'], .75 * native['unit42'] + .25 * native['light'], 'Synthetic unit base')
    finals = old.compose_final(bases)
    c.exact(finals['v22'], np.median(np.column_stack([bases[k] for k in ('N', 'H', 'U22')]), axis=1), 'Synthetic v22')
    c.exact(finals['v24'], np.median(np.column_stack([bases[k] for k in ('N', 'H', 'U24')]), axis=1), 'Synthetic v24')
    return {'status': 'passed', 'checks': ['prior_base', 'unit_base', 'v22_three_branch_median', 'v24_separate_unit137_median'],
            'real_data_read': False, 'query_y_read': False, 'new_fits': 0, 'GPU_calls': 0, 'new_CSV': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-check', action='store_true')
    group.add_argument('--preflight', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
    else:
        tick = time.monotonic()
        ctx = load_context()
        result = {'status': 'passed', 'specification': ctx['specification'], 'replay_checks': ctx['replay_checks'],
                  'prepared_array_identities': ctx['prepared_array_identities'], 'fit_rows': ROWS, 'query_rows': QUERY_ROWS,
                  'fit_label_identity': ctx['fit_label_identity'], 'weight_identity': ctx['weight_identity'],
                  'elapsed_seconds': time.monotonic() - tick, 'raw_price_table_opened_or_rehashed': False,
                  'query_y_read': False, 'new_fits': 0, 'GPU_calls': 0, 'native_model_inference_calls': 0, 'new_CSV': False,
                  'author_preflight_not_independent_source_review': True}
    if args.receipt is not None:
        c.safe(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        c.write(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
