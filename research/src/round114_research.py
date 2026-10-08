"""R114: one signed additive prior NN at four fixed origins, then scoring.

Source preparation is not fit authorization. The coordinator alone invokes
freeze/train/evaluate after independent preflight. All four native models,
signed totals, projected totals and complete five-layer chains are sealed and
training exits successfully before the separate scorer can read query prices.
No CSV, full-data fit, parameter search or shared split mutation is provided.
"""
from __future__ import annotations
import argparse
import ast
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round104_composition as comp
import round100_temporal_prior_evaluate_v2 as metric
import round111_112_research as history
import round113_offset_composition as offset_comp
import round113_offset_research as offset
import round97_offset_evaluate as offset_metric

PLAN = 'reports/round114_plan.md'
PLAN_SHA = '45b4095430f6adffab7977b37e60eb0e30f958e1a4634e6dd251739acd5b126a'
TRAINER = 'src/round114_additive_prior_train.py'
BASE = 'outputs/round114'
EVIDENCE = 'outputs/research_round114_20261008'
REQUEST, SEAL, RESULTS = (BASE + '/' + name + '.json' for name in ('request', 'seal', 'results'))
OFFSET = 'offset_2023_02_11'
ORIGINS = (*c.STAGES, OFFSET)
LAYERS = ('native', 'base', 'reference', 'spatial', 'median')
COUNTS = {**{s: c.SPLITS[s][2:] for s in c.STAGES}, OFFSET: (272117, 107169)}
RAW_SHA = '9acf8c65bf00f7067e9ea9612cabe9a1524b020a7596c8da3ce3d620131e5c00'
CAUTION = ('Four reused overlapping origins are not independent votes or fresh blind validation. '
           'Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap '
           'omits cumulative selection, retraining variation and future shift; not Private probability. '
           'R91 additive CatBoost and R104 offset NN supplied strong contrary evidence; '
           'native improvement alone cannot establish complete-composition improvement.')


def sources():
    return c.source_hashes(__file__, TRAINER)


def protected():
    return history.protected()


def merge_hashes(target, incoming):
    for name, digest in incoming.items():
        c.need(name not in target or target[name] == digest, 'Conflicting input authority: ' + name)
        target[name] = digest


def trainer_recipe():
    """Read fixed literal overrides, without importing Torch or invoking trainer."""
    tree = ast.parse(c.safe(TRAINER).read_text(encoding='utf-8-sig'))
    found = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'PARAMETERIZATION_OVERRIDES' for t in n.targets)]
    c.need(len(found) == 1, 'One literal additive parameterization required')
    overrides = ast.literal_eval(found[0])
    old_name = 'outputs/round63/development/model/training.json'
    old = c.read(old_name)['configuration']
    c.need(isinstance(overrides, dict) and overrides['training_total_clipped'] is False
           and overrides['loss_formula'] == 'mean(train_sample_weight * abs(train_price_factor_million + residual_million - train_y_million))',
           'Fixed additive total loss and unclipped training required')
    return {'baseline_configuration': old, 'parameterization_overrides': overrides,
            'configuration': {**old, **overrides}, 'baseline_training': c.record(old_name)}


def formal_checks():
    return {'R114_trainer_selfcheck_execution.json': (TRAINER, '.venv_torch', '--cpu-unit-check'),
            'R114_driver_selfcheck_execution.json': (c.rel(__file__), '.venv', '--self-check')}


def check_formal_checks(inputs):
    for name, (entry, environment, flag) in formal_checks().items():
        path = EVIDENCE + '/' + name
        check = c.read(path)
        command = [str(c.ROOT / environment / 'Scripts/python.exe'), '-B', '-X', 'utf8', str(c.safe(entry)), flag]
        c.need(check['exit_code'] == 0 and check['command'] == command
               and check['source_sha256'] == c.source_hashes(entry), 'Formal source/command check differs: ' + name)
        inputs[path] = c.sha(path)


def august_specifications(inputs):
    names = ('outputs/round79/plan.json', 'outputs/round79/results.json', 'outputs/round63/results.json',
             'outputs/round83/results.json', 'outputs/round100_inputs/manifest.json')
    for name in names: inputs[name] = c.sha(name)
    p79, r79, r63, r83, labels = [c.read(name) for name in names]
    # Metadata/opaque identities only here. No raw price deserialization.
    merge_hashes(inputs, r63['input_sha256'])
    c.need(p79['input_sha256'][c.RAW] == RAW_SHA, 'Original scoring label identity')
    inputs[c.RAW] = RAW_SHA
    specs = {}
    for origin in c.STAGES:
        cs = comp.specification(origin); merge_hashes(inputs, cs['input_sha256'])
        old = r63['stages'][origin]
        prepared = f'outputs/round63/{origin}/prepared.npz'
        prep = f'outputs/round63/{origin}/preparation.json'
        training = f'outputs/round63/{origin}/model/training.json'
        for path in (prepared, prep, training): inputs[path] = old['artifact_sha256'][path]
        prior = c.read(prep)
        for key in ('prior_preparation', 'source_prepared'):
            path = prior[key + '_path']; digest = prior[key + '_sha256']
            merge_hashes(inputs, {path: digest})
        native = f'outputs/round79/{origin}/model/native_predictions.npy'
        inputs[native] = r79['stages'][origin]['artifact_sha256'][native]
        v24_names = {key: f'outputs/round83/{origin}/{name}' for key, name in (
            ('seed137', 'model/native_predictions.npy'), ('base', 'candidate_base.npy'),
            ('reference', 'candidate_reference.npy'), ('spatial', 'candidate_unit_spatial.npy'),
            ('median', 'predictions.npy'))}
        for path in v24_names.values(): inputs[path] = r83['stages'][origin]['artifact_sha256'][path]
        for record in labels['stages'][origin]['arrays'].values(): inputs[record['path']] = record['sha256']
        specs[origin] = {'composition': cs, 'prepared': c.record(prepared), 'preparation': c.record(prep),
                        'original_training': c.record(training), 'native_unit': c.record(native),
                        'v24_layers': {k: c.record(v) for k, v in v24_names.items()},
                        'fit_labels': labels['stages'][origin]['arrays'],
                        'date_weight_sha256': p79['bindings'][origin]['preparation']['train_sample_weight_sha256']}
    return specs


def freeze():
    c.need(not c.safe(BASE).exists(), 'Preserve any previous R114 attempt')
    c.need(c.sha(PLAN) == PLAN_SHA, 'Fixed pre-scoring four-origin plan changed')
    own = sources(); review_path = EVIDENCE + '/preflight_review.json'; review = c.read(review_path)
    c.need(review['status'] == 'passed' and review['source_sha256'] == {**own, PLAN: PLAN_SHA},
           'Independent complete source/plan preflight required')
    c.verify_hashes(review['source_sha256'])
    inputs = {review_path: c.sha(review_path)}; check_formal_checks(inputs)
    recipe = trainer_recipe(); inputs[recipe['baseline_training']['path']] = recipe['baseline_training']['sha256']
    specs = august_specifications(inputs)
    extra = offset_comp.specification(); merge_hashes(inputs, extra['input_sha256'])
    inputs['outputs/round97/evaluation/results.json'] = c.sha('outputs/round97/evaluation/results.json')
    old_seal = c.read(extra['original_seal']['path'])
    c.need(old_seal['input_sha256'][c.RAW] == inputs[c.RAW] == RAW_SHA, 'All scoring-only raw axes agree')
    specs[OFFSET] = {'composition': extra, 'prepared': extra['prior_npz']}
    c.verify_hashes(inputs)
    c.safe(BASE).mkdir(exist_ok=False)
    c.write(REQUEST, {'status': 'frozen_before_four_fixed_origin_fits', 'created_at_utc': c.now(),
        'round': 114, 'candidate': 'R114', 'origin_order': list(ORIGINS), 'origin_counts': COUNTS,
        'plan': c.record(PLAN), 'origins': specs, 'source_sha256': own, 'input_sha256': inputs,
        'recipe': recipe, 'protected_before': protected(), 'new_fits': 4,
        'query_scores_computed': False, 'query_labels_used_for_fit': False, 'public_tuning': False,
        'Public_score': None, 'Private_score': None, 'selected_submission': 'v22',
        'reserved_exploratory_version': 30, 'new_CSV': False, 'full_data_fit': False})


def request():
    r = c.read(REQUEST)
    c.need(r['status'] == 'frozen_before_four_fixed_origin_fits' and r['round'] == 114
           and r['origin_order'] == list(ORIGINS) and set(r['origins']) == set(ORIGINS)
           and r['new_fits'] == 4 and r['new_CSV'] is False and r['full_data_fit'] is False,
           'Immutable four-origin request scope differs')
    c.need(r['plan'] == c.record(PLAN) and r['plan']['sha256'] == PLAN_SHA, 'Plan changed')
    c.need(r['source_sha256'] == sources() and r['recipe'] == trainer_recipe(), 'Frozen source/recipe changed')
    c.verify_hashes(r['source_sha256']); c.verify_hashes(r['input_sha256'])
    c.need(protected() == r['protected_before'], 'Protected original artifacts changed')
    return r


def prepared_arrays(path, nf, nq, fit_y):
    with np.load(c.safe(path), allow_pickle=False) as archive:
        c.need(len(archive.files) == len(c.ARRAY_NAMES) and set(archive.files) == set(c.ARRAY_NAMES),
               'Exactly nine prepared keys before materializing; no query y')
        arrays = {key: archive[key].copy() for key in c.ARRAY_NAMES}
    c.need(arrays['train_num'].dtype == arrays['query_num'].dtype == np.float32
           and arrays['train_num'].shape == (nf, 144) and arrays['query_num'].shape == (nq, 144)
           and arrays['train_cat'].dtype == arrays['query_cat'].dtype == np.int64
           and arrays['train_cat'].shape == (nf, 23) and arrays['query_cat'].shape == (nq, 23)
           and arrays['cardinalities'].dtype == np.int64 and arrays['cardinalities'].shape == (23,),
           'Original prior numeric/categorical dimensions and dtypes')
    for key, rows in (('train_price_factor_million', nf), ('query_price_factor_million', nq), ('train_sample_weight', nf)):
        value = arrays[key]
        c.need(value.dtype == np.float32 and value.shape == (rows,)
               and np.isfinite(value).all() and (value > 0).all(), 'Positive original factor/date weight: ' + key)
    for key in ('train_num', 'query_num'): c.need(np.isfinite(arrays[key]).all(), 'Finite numeric inputs')
    c.exact(arrays['train_y_million'], np.asarray(fit_y / 1e6, dtype=np.float32), 'Fit-only canonical y alignment')
    return arrays


def replay_august_v24(ctx, spec):
    """R83 mean of native42/native137, with its original R/S; no LGB added."""
    native42 = np.load(c.safe(spec['native_unit']['path']), allow_pickle=False)
    seed137 = np.load(c.safe(spec['v24_layers']['seed137']['path']), allow_pickle=False)
    control = history.unit_chain(ctx, native42)
    c.exact(control['spatial'], ctx['fixed']['U'], 'Original v22 unit spatial replay')
    c.exact(control['median'], ctx['original']['median'], 'Original v22 full median replay')
    base = .5 * native42 + .5 * seed137
    ref, diag, _ = comp.old.original.predict_reference(ctx['reference'], ctx['query_X'], base, variant='trend', return_diagnostics=True)
    spatial, detail, _ = comp.old.spatial.predict(ctx['spatial_state'], ctx['query_X'], ctx['geo'], base, ref, diag)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback], ref[fallback], 'Original v24 spatial fallback')
    median = np.median(np.column_stack([ctx['original']['spatial'], ctx['fixed']['H'], spatial]), axis=1)
    for key, value in {'base': base, 'reference': ref, 'spatial': spatial, 'median': median}.items():
        c.exact(value, np.load(c.safe(spec['v24_layers'][key]['path']), allow_pickle=False), 'Original v24 ' + key + ' replay')
    comp.verify_unchanged(ctx)
    return median


def context_identity(ctx, origin):
    if origin == OFFSET:
        return {'state': ctx['state_identity'], 'frames_and_indices': ctx['data_identity'],
                'fit_labels': ctx['fit_label_identity'], 'weights': ctx['weight_identity'],
                'baseline_vectors': ctx['vector_identity']}
    return {'state': ctx['state_identity'], 'frames': ctx['frame_identity'],
            'indices': {k: c.identity(v) for k, v in ctx['indices'].items()},
            'fixed': {k: c.identity(v) for k, v in ctx['fixed'].items()},
            'original': {k: c.identity(v) for k, v in ctx['original'].items()}}


def load_context(r, origin):
    c.need(origin in ORIGINS, 'Unknown origin')
    spec = r['origins'][origin]; nf, nq = COUNTS[origin]
    if origin == OFFSET:
        ctx = offset_comp.load_context(spec['composition'])
        y = ctx['yfit']; v24 = ctx['baseline']['v24']
        X, Xq = ctx['data'].Xfit, ctx['data'].Xquery
        indices = {'query_indices': ctx['data'].query_indices,
                   'query_source_cache_indices': ctx['data'].query_source_cache_indices}
    else:
        ctx = comp.load_context(spec['composition']); labels = spec['fit_labels']
        for key in ('fit_indices', 'fit_source_cache_indices'):
            c.exact(c.load_array(labels[key]), ctx['indices'][key], 'Fit label axis: ' + key)
        y = c.load_array(labels['fit_y'])
        v24 = replay_august_v24(ctx, spec)
        X, Xq, indices = ctx['fit_X'], ctx['query_X'], ctx['indices']
    c.need(len(X) == nf and len(Xq) == nq and y.dtype == np.float64 and y.shape == (nf,)
           and np.isfinite(y).all() and (y > 0).all(), 'Fixed origin fit labels/row counts')
    arrays = prepared_arrays(spec['prepared']['path'], nf, nq, y)
    identities = {key: c.identity(value) for key, value in arrays.items()}
    if origin == OFFSET:
        c.need(identities == ctx['prepared_array_identities'], 'Original offset nine arrays changed')
        c.exact(arrays['train_sample_weight'], ctx['weights'], 'Offset saved dates')
    else:
        old = c.read(spec['original_training']['path'])
        for key in ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
            c.need(identities[key]['values_sha256'] == old[key + '_sha256'], 'Original ' + key)
        c.need(identities['train_sample_weight']['values_sha256'] == spec['date_weight_sha256'], 'Original date weight authority')
    return {'context': ctx, 'fit_y': y, 'arrays': arrays, 'array_identities': identities,
            'fit_X': X, 'query_X': Xq, 'indices': indices, 'v24': v24,
            'v22': ctx['baseline']['v22'] if origin == OFFSET else ctx['original']['median'],
            'before': context_identity(ctx, origin)}


def verify_context(loaded, origin):
    ctx = loaded['context']
    (offset_comp.verify_unchanged if origin == OFFSET else comp.verify_unchanged)(ctx)
    c.need(context_identity(ctx, origin) == loaded['before'], 'Fit/query state or original vectors mutated')
    c.need({k: c.identity(v) for k, v in loaded['arrays'].items()} == loaded['array_identities'], 'Prepared inputs mutated')


def signed_projection(raw, native, rows):
    c.need(type(rows) is int and rows > 0 and raw.dtype == native.dtype == np.float64 and raw.shape == native.shape == (rows,)
           and np.isfinite(raw).all() and np.isfinite(native).all(), 'Finite float64 signed/native totals')
    c.exact(np.maximum(raw, 0), native, 'Exactly one nonnegative inference-total projection')
    correction = native - raw
    return {'raw_negative_total_rows': int((raw < 0).sum()), 'raw_negative_total_fraction': float((raw < 0).mean()),
            'raw_zero_total_rows': int((raw == 0).sum()), 'raw_min_TWD': float(raw.min()), 'raw_max_TWD': float(raw.max()),
            'projected_min_TWD': float(native.min()), 'projected_max_TWD': float(native.max()),
            'projected_zero_rows': int((native == 0).sum()), 'projected_zero_fraction': float((native == 0).mean()),
            'projection_mean_correction_TWD': float(correction.mean()), 'projection_max_correction_TWD': float(correction.max()),
            'projection_changes_count': int(np.count_nonzero(correction)), 'query_y_used_for_diagnostics': False,
            'negative_residual_not_equivalent_to_negative_total': True}


def check_metadata(meta, r, origin, identities):
    nf, nq = COUNTS[origin]; recipe = r['recipe']
    c.need(meta['status'] == 'completed' and meta['source_sha256'] == r['source_sha256'][TRAINER]
           and meta['input_sha256'] == r['origins'][origin]['prepared']['sha256']
           and meta['input_arrays'] == identities and meta['fit_rows'] == nf and meta['query_rows'] == nq
           and meta['configuration'] == recipe['configuration'] and meta['baseline_configuration'] == recipe['baseline_configuration']
           and meta['parameterization_overrides'] == recipe['parameterization_overrides']
           and meta['actual_model_class'] == 'round114_additive_prior_train.AdditivePriorMLP'
           and meta['numeric_count'] == 144 and meta['categorical_count'] == 23
           and meta['saved_raw_total_reload_exact'] is True and meta['saved_model_reload_exact'] is True
           and meta['reload_rows'] == nq and meta['query_labels_used'] is False and meta['query_labels_available'] is False
           and meta['training_total_clipped'] is False and meta['other_price_cap_applied'] is False
           and meta['sample_weights_recomputed_in_trainer'] is False and meta['sample_weights_batch_renormalized'] is False
           and meta['head_weight_zeroed'] is False and meta['loss_only_change_claimed'] is False
           and [v['epoch'] for v in meta['epochs']] == list(range(1, 31)), 'Actual fixed additive NN metadata differs')


def model_records(folder):
    files = sorted(folder.iterdir())
    c.need(files and all(p.is_file() and p.suffix in {'.json', '.npy', '.pt'} for p in files), 'Bind every flat model artifact')
    required = {'model.pt', 'run.json', 'training.json', 'predictions_raw_twd.npy', 'predictions_twd.npy'}
    c.need(required <= {p.name for p in files}, 'Missing required native signed/total artifacts')
    return [c.record(path) for path in files]


def validate(origin, record, r):
    folder = c.safe(BASE + '/' + origin); model = folder / 'model'; nf, nq = COUNTS[origin]
    c.need(record == c.record(folder / 'stage.json'), 'Stage exact identity')
    item = c.read(record['path'])
    c.need(item['stage'] == origin and item['candidate_round'] == 114 and item['fit_rows'] == nf and item['query_rows'] == nq
           and item['query_labels_available'] is False and item['model_metadata'] == c.record(model / 'training.json')
           and item['sealed_models'] == model_records(model) and item['context_before'] == item['context_after'], 'Stage scope/models/state seal differs')
    meta = c.read(model / 'training.json'); check_metadata(meta, r, origin, item['prepared_array_identities'])
    run = c.read(model / 'run.json')
    c.need(run['status'] == 'started' and all(meta.get(k) == v for k, v in run.items() if k != 'status'),
           'Native before/after metadata contract differs')
    c.need(c.sha(model / 'model.pt') == meta['model_sha256'], 'Saved model checkpoint identity')
    raw, native = [np.load(model / name, allow_pickle=False) for name in ('predictions_raw_twd.npy', 'predictions_twd.npy')]
    projection = signed_projection(raw, native, nq)
    for name, values in (('predictions_raw_twd.npy', raw), ('predictions_twd.npy', native)):
        c.need(meta['prediction_artifacts'][name] == {'sha256': c.sha(model / name), **c.identity(values)}, 'Native prediction metadata identity')
    c.need(item['projection_diagnostics'] == meta['projection_diagnostics'] == projection
           and item['raw_signed_total'] == {**c.record(model / 'predictions_raw_twd.npy'), **c.identity(raw)}, 'Signed native projection seal')
    layers = {k: c.load_array(v) for k, v in item['layers'].items()}
    c.need(set(layers) == set(LAYERS), 'Exactly five layers')
    for key, values in layers.items(): metric.prices(values, (nq,), key + ' complete layer')
    c.exact(layers['native'], native, 'Native model/five-layer alignment')
    return item, layers, raw


def stage(origin):
    r = request(); started = c.read(EVIDENCE + '/training_started.json')
    c.need(started['request'] == c.record(REQUEST) and started['new_fits_authorized'] == 4, 'Coordinated serial training marker required')
    loaded = load_context(r, origin); nf, nq = COUNTS[origin]
    folder = c.safe(BASE + '/' + origin); folder.mkdir(exist_ok=False)
    command = [str(c.ROOT / '.venv_torch/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(c.safe(TRAINER)),
               '--data', str(c.safe(r['origins'][origin]['prepared']['path'])), '--out', str(folder / 'model')]
    with (folder / 'train.log').open('x', encoding='utf8') as log:
        process = subprocess.run(command, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    c.need(process.returncode == 0, 'Trainer failed; preserve evidence, no retry')
    meta = c.read(folder / 'model/training.json'); check_metadata(meta, r, origin, loaded['array_identities'])
    expected_bias = float(np.median(loaded['arrays']['train_y_million'].astype(np.float64)
                                    - loaded['arrays']['train_price_factor_million'].astype(np.float64)))
    c.need(meta['initial_training_median_residual_million'] == expected_bias
           and meta['initial_head_bias'] == float(np.float32(expected_bias)), 'Actual fixed fit-median additive head bias')
    raw, native = [np.load(folder / 'model' / name, allow_pickle=False) for name in ('predictions_raw_twd.npy', 'predictions_twd.npy')]
    projection = signed_projection(raw, native, nq)
    ctx = loaded['context']; chain = offset_comp.chain_prior(ctx, native) if origin == OFFSET else comp.apply(ctx, native)[0]
    c.need(set(chain) == set(LAYERS), 'Exactly one prior-slot replacement and complete composition')
    records = {key: c.save_array(folder / (key + '.npy'), value) for key, value in chain.items()}
    verify_context(loaded, origin)
    c.write(folder / 'stage.json', {'stage': origin, 'candidate_round': 114, 'fit_rows': nf, 'query_rows': nq,
        'model_metadata': c.record(folder / 'model/training.json'), 'sealed_models': model_records(folder / 'model'),
        'raw_signed_total': {**c.record(folder / 'model/predictions_raw_twd.npy'), **c.identity(raw)},
        'layers': records, 'prepared_array_identities': loaded['array_identities'], 'fit_seconds': meta['fit_seconds'],
        'context_before': loaded['before'], 'context_after': context_identity(ctx, origin),
        'projection_diagnostics': projection, 'original_v22_v24_and_prior_slot_replayed_exact': True,
        'query_labels_available': False, 'query_scores_computed': False})
    validate(origin, c.record(folder / 'stage.json'), r); request()
    print(json.dumps({'round': 114, 'origin': origin, 'status': 'saved_and_sealed_unscored'}), flush=True)


def train():
    r = request(); tick = time.monotonic(); failure = None
    c.need(not c.safe(EVIDENCE + '/training_started.json').exists()
           and not c.safe(SEAL).exists() and not c.safe(EVIDENCE + '/training_execution.json').exists(), 'One serial attempt only')
    c.write(EVIDENCE + '/training_started.json', {'request': c.record(REQUEST), 'created_at_utc': c.now(),
            'new_fits_authorized': 4, 'automatic_retry': False})
    try:
        for origin in ORIGINS:
            command = [str(c.ROOT / '.venv/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--stage', origin]
            process = subprocess.run(command, cwd=c.ROOT, check=False)
            c.need(process.returncode == 0, 'Origin failed, no automatic retry: ' + origin)
        records = {s: c.record(BASE + '/' + s + '/stage.json') for s in ORIGINS}
        for origin, record in records.items(): validate(origin, record, r)
        request()
        c.write(SEAL, {'status': 'four_models_all_native_signed_and_layers_sealed_unscored',
            'request': c.record(REQUEST), 'origin_order': list(ORIGINS), 'records': records, 'new_fits': 4,
            'query_scores_computed': False, 'elapsed_seconds': time.monotonic() - tick})
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(EVIDENCE + '/training_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
            'request': c.record(REQUEST), 'elapsed_seconds': time.monotonic() - tick, 'automatic_retry': False})


def combined_gates(stages):
    c.need(set(stages) == set(ORIGINS), 'All four fixed origins required')
    august = history.gates({s: stages[s] for s in c.STAGES})
    extra = offset.gates(stages[OFFSET]['metrics'], stages[OFFSET]['comparison'])
    result = {'august': august, 'extra_origin': extra}
    for name, old in (('combined_exploratory_submission', august['exploratory_submission']['passed']),
                      ('combined_promotion', august['promotion']['adoption_research_gate_passed'])):
        parts = {'original_three_windows': old, 'fixed_extra_origin': extra['passed']}
        result[name] = {'components': parts, 'passed': all(parts.values())}
    result.update(private_score=None, selected_submission='v22', automatic_selection=False,
                  new_CSV=False, full_data_fit=False, reserved_exploratory_version=30)
    return result


def scoring_preflight():
    r = request(); seal = c.read(SEAL); execution = c.read(EVIDENCE + '/training_execution.json')
    c.need(seal['status'] == 'four_models_all_native_signed_and_layers_sealed_unscored'
           and seal['request'] == c.record(REQUEST) and seal['origin_order'] == list(ORIGINS)
           and set(seal['records']) == set(ORIGINS) and seal['new_fits'] == 4 and seal['query_scores_computed'] is False,
           'All four models/layers must be sealed before any query price access')
    c.need(execution['exit_code'] == 0 and execution['failure'] is None and execution['request'] == c.record(REQUEST),
           'Serial training must exit successfully before separate scoring')
    checked = {s: validate(s, seal['records'][s], r) for s in ORIGINS}
    c.need(r['input_sha256'][c.RAW] == RAW_SHA and c.sha(c.RAW) == RAW_SHA, 'Scoring-only raw label source changed')
    return r, seal, checked


def evaluate():
    c.need(not c.safe(RESULTS).exists() and not c.safe(EVIDENCE + '/evaluation_execution.json').exists(), 'Preserve any previous scoring attempt')
    r, seal, checked = scoring_preflight(); tick = time.monotonic(); failure = None
    try:
        # First and only raw-price deserialization: after every model/metadata/layer and successful exit guard.
        raw = pd.read_pickle(c.safe(c.RAW)); stages = {}
        for origin in ORIGINS:
            loaded = load_context(r, origin); item, layers, signed = checked[origin]
            if origin == OFFSET:
                descriptive = lambda v: offset_metric.descriptive_metrics(y, v, loaded['fit_X'], loaded['query_X'])
                y = offset_metric.select_query_y(raw, loaded['indices']['query_source_cache_indices'], 455784)
            else:
                y = metric.select_query_y(raw, loaded['indices']['query_source_cache_indices'])
                descriptive = lambda v: metric.descriptive_metrics(y, v, loaded['fit_X'], loaded['query_X'], c.SPLITS[origin][0])
            metrics = {'v22': descriptive(loaded['v22']), 'v24': descriptive(loaded['v24']), 'T': descriptive(layers['median'])}
            if origin == OFFSET:
                old = c.read('outputs/round97/evaluation/results.json')['metrics']
                c.need(all(metrics[k] == old[k] for k in ('v22', 'v24')), 'Original offset control metrics replay')
                comparison = offset_metric.comparisons({'v22': metrics['v22'], 'v24': metrics['T']})
                comparison24 = offset_metric.comparisons({'v22': metrics['v24'], 'v24': metrics['T']})
                paired, samples = offset_metric.paired(y, {'v22': loaded['v22'], 'v24': layers['median']},
                    loaded['fit_X'], loaded['query_X'], loaded['indices']['query_indices'])
            else:
                for key, number, label in (('v22', 79, 'R79'), ('v24', 83, 'R83')):
                    old = c.read(f'outputs/round{number}/results.json')['stages'][origin]['metrics'][label]
                    c.need(all(metrics[key][k] == old[k] for k in old), 'Original August control metrics replay: ' + key)
                comparison, comparison24 = metric.comparisons(metrics), metric.comparisons(metrics, 'v24')
                paired, samples = metric.paired(y, {'v22': loaded['v22'], 'T': layers['median']},
                    loaded['fit_X'], loaded['query_X'], loaded['indices']['query_indices'], origin, 'v22')
            comparison.update(control='v22', candidate='R114', used_for_release_gate=True)
            comparison24.update(control='v24', candidate='R114', used_for_release_gate=False)
            paired.update(candidate='R114', difference='R114 minus v22', caution=CAUTION)
            folder = c.safe(BASE + '/' + origin + '/evaluation'); folder.mkdir(exist_ok=False)
            sample_records = {k: c.save_array(folder / (k + '_samples.npy'), v) for k, v in samples.items()}
            result = {'metrics': metrics, 'comparison': comparison, 'comparison_vs_v24': comparison24,
                'paired': paired, 'paired_samples': sample_records, 'sealed_stage': seal['records'][origin],
                'branch_metrics': {k: descriptive(v) for k, v in layers.items() if k != 'median'},
                'raw_signed_total_diagnostic_only': {'overall_mae': float(np.abs(signed - y).mean()),
                                                   **item['projection_diagnostics']},
                'both_point_metrics_improve': all(metrics['T'][k] < metrics['v22'][k] for k in ('overall_mae', 'weighted_mae'))}
            verify_context(loaded, origin); c.write(folder / 'results.json', result); stages[origin] = result
            print(json.dumps({'round': 114, 'origin': origin, 'mae': metrics['T']['overall_mae'],
                  'delta': metrics['T']['overall_mae'] - metrics['v22']['overall_mae']}), flush=True)
        del raw
        scoring_preflight()
        c.write(RESULTS, {'status': 'four_reused_fixed_origins_evaluated', 'created_at_utc': c.now(),
            'round': 114, 'request': c.record(REQUEST), 'seal': c.record(SEAL), 'plan': c.record(PLAN),
            'source_sha256': sources(), 'origin_order': list(ORIGINS), 'stages': stages, 'gates': combined_gates(stages),
            'new_fits': 4, 'new_CSV': False, 'full_data_fit': False, 'Public_score': None, 'Private_score': None,
            'selected_submission': 'v22', 'fresh_blind_validation': False, 'caution': CAUTION})
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(EVIDENCE + '/evaluation_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
            'elapsed_seconds': time.monotonic() - tick, 'request': c.record(REQUEST), 'seal': c.record(SEAL),
            'training_execution': c.record(EVIDENCE + '/training_execution.json'),
            'query_labels_used_for_scoring_only': True, 'training_already_finished_before_scoring': True,
            'new_fits': 0, 'new_CSV': False})


def self_check():
    """Synthetic boundary checks only; no data/model load, subprocess or GPU."""
    from unittest.mock import patch
    checks = []
    def ok(name, condition): c.need(condition, name); checks.append(name)
    def rejects(name, call):
        try: call()
        except (ValueError, KeyError): checks.append(name)
        else: raise AssertionError('Guard accepted: ' + name)
    raw = np.array([-1., 0., 2.], np.float64); native = np.maximum(raw, 0)
    ok('signed_total_single_projection', signed_projection(raw, native, 3)['raw_negative_total_rows'] == 1)
    rejects('native_second_offset_or_cap_rejected', lambda: signed_projection(raw, native + 1, 3))
    rejects('signed_nan_rejected', lambda: signed_projection(np.array([np.nan], np.float64), np.zeros(1), 1))
    rejects('signed_float32_rejected', lambda: signed_projection(raw.astype(np.float32), native, 3))
    rejects('empty_signed_vector_rejected', lambda: signed_projection(np.empty(0), np.empty(0), 0))
    ok('origins_separate_no_global_split_mutation', ORIGINS == ('development', 'confirmation', 'auxiliary', OFFSET)
       and tuple(c.SPLITS) == c.STAGES and COUNTS[OFFSET] == (272117, 107169))
    def sample(a=True, e=True, strong=True):
        old={'exploratory_submission':{'passed':a}, 'promotion':{'adoption_research_gate_passed':strong}}
        data={s:{'metrics':{},'comparison':{}} for s in ORIGINS}
        with patch.object(history, 'gates', return_value=old), patch.object(offset, 'gates', return_value={'passed':e}):
            return combined_gates(data)
    for a in (False, True):
        for e in (False, True):
            for strong in (False, True):
                result=sample(a,e,strong)
                ok('combined_gate_boolean_matrix', result['combined_exploratory_submission']['passed'] == (a and e)
                   and result['combined_promotion']['passed'] == (strong and e) and result['new_CSV'] is False)
    rejects('missing_fourth_origin_rejected', lambda: combined_gates({s:{} for s in c.STAGES}))
    class FakeArchive:
        def __init__(self, arrays, files=None): self.arrays, self.files, self.reads = arrays, list(arrays) if files is None else files, 0
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key): self.reads += 1; return self.arrays[key]
    arrays={'train_num':np.zeros((3,144),np.float32),'query_num':np.zeros((3,144),np.float32),
            'train_cat':np.zeros((3,23),np.int64),'query_cat':np.zeros((3,23),np.int64),
            'cardinalities':np.full(23,2,np.int64),'train_y_million':np.ones(3,np.float32),
            'train_price_factor_million':np.ones(3,np.float32),'query_price_factor_million':np.ones(3,np.float32),
            'train_sample_weight':np.ones(3,np.float32)}
    truth=np.full(3,1e6,np.float64)
    archive=FakeArchive(arrays)
    with patch.object(np,'load',return_value=archive):
        actual=prepared_arrays('synthetic.npz',3,3,truth)
        ok('synthetic_original_nine_arrays',set(actual)==set(c.ARRAY_NAMES) and archive.reads==9)
    for files in (list(c.ARRAY_NAMES)+['query_y'],list(c.ARRAY_NAMES[:-1])+['query_y'],list(c.ARRAY_NAMES)+['train_num']):
        archive=FakeArchive(arrays,files)
        with patch.object(np,'load',return_value=archive):
            rejects('invalid_archive_keys_before_values',lambda:prepared_arrays('synthetic.npz',3,3,truth))
        ok('invalid_archive_zero_reads',archive.reads==0)
    for key,bad in (('train_sample_weight',np.zeros(3,np.float32)),
                    ('query_price_factor_million',np.full(3,np.nan,np.float32)),
                    ('query_num',np.zeros((3,143),np.float32)),
                    ('query_cat',np.zeros((3,23),np.float32)),
                    ('train_y_million',np.zeros(3,np.float32))):
        archive=FakeArchive({**arrays,key:bad})
        with patch.object(np,'load',return_value=archive):
            rejects('invalid_prepared_'+key,lambda:prepared_arrays('synthetic.npz',3,3,truth))
    # No raw-price read can occur when successful training/seal guards fail.
    dummy={'request':{},'status':'bad'}
    with patch.dict(globals(), {'request': lambda:{}, 'c':c}), patch.object(c, 'read', return_value=dummy), patch.object(c, 'record', return_value={}):
        rejects('incomplete_seal_rejected_before_price_read', scoring_preflight)
    sealed={'status':'four_models_all_native_signed_and_layers_sealed_unscored','request':{},
            'origin_order':list(ORIGINS),'records':{s:{} for s in ORIGINS},'new_fits':4,'query_scores_computed':False}
    execution={'exit_code':0,'failure':None,'request':{}}
    with patch.dict(globals(),{'request':lambda:{'input_sha256':{c.RAW:RAW_SHA}},'validate':lambda *args:({}, {}, raw)}), \
         patch.object(c,'read',side_effect=lambda name:sealed if name==SEAL else execution), \
         patch.object(c,'record',return_value={}),patch.object(c,'sha',return_value=RAW_SHA):
        ok('all_four_validation_before_label_access',len(scoring_preflight()[2])==4)
        execution['exit_code']=1
        rejects('failed_training_exit_refuses_scoring',scoring_preflight);execution['exit_code']=0
        sealed['records'].pop(OFFSET)
        rejects('missing_offset_seal_refuses_scoring',scoring_preflight)
    return {'status':'passed','checks':checks,'check_count':len(checks),
        'source_sha256':c.source_hashes(__file__), 'complete_research_source_sha256':sources(),
        'query_labels_read':False,'real_data_read':False,'new_fits':0,'GPU_calls':0,'new_CSV':False,
        'global_SPLITS_modified':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true'); group.add_argument('--train',action='store_true')
    group.add_argument('--stage',choices=ORIGINS); group.add_argument('--evaluate',action='store_true')
    group.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    if args.freeze: freeze()
    elif args.train: train()
    elif args.stage: stage(args.stage)
    elif args.evaluate: evaluate()
    else: print(json.dumps(self_check(),ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)
