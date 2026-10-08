"""Prepare one independently gated R111/v28 or R112/v29 exploratory release.

This file grants no fit. Completed independent batch postfit review, the fixed
exploratory gate, a separate final plan and independent final source preflight
are required. --fit is invoked only by the coordinating root, once. It writes
sealed unrounded vectors and typed integer rows. Root separately invokes the
frozen Artifact builder; --release verifies its existing CSV and receipt and
never creates a CSV, uploads to Kaggle or changes the user's v22 selection.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import subprocess
import time
import traceback
import numpy as np
import round100_contract as c
import round107_final_candidate as final
import round107_full_release as shared
import round111_112_research as history

FOLDER = 'outputs/research_round111_112_20261008/'
TRAINER = 'src/round111_residual_prior_train.py'
HELPER = 'src/round112_ordered_unit.py'
CHECKPOINT_REVIEW = 'src/round111_checkpoint_review.py'
BUILDER = shared.BUILDER
VERSIONS = {111: 28, 112: 29}
EXTRA_PLAN = 'reports/round113_plan.md'
EXTRA_PLAN_SHA = 'a21116136190aadf94d89329dd1b9dd98afb62dd493ef6863b356f50ea01dc2c'
EXTRA_RESULTS = 'outputs/round113/results.json'
EXTRA_GATE = 'outputs/research_round113_20261008/round113_release_gate.json'
EXTRA_REVIEW = 'outputs/research_round113_20261008/postfit_review.json'
EXTRA_COMPONENTS = {'overall_regression_at_most_0_25pct', 'weighted_regression_at_most_0_25pct',
                    'no_quarter_or_large_group_above_5pct'}


def paths(round_number, version):
    c.need(type(round_number) is int and type(version) is int and VERSIONS.get(round_number) == version,
           'Explicit mapping is R111/v28 or R112/v29')
    base = f'outputs/round{round_number}/final'
    return {'round': round_number, 'version': version, 'base': base, 'request': base + '/request.json',
            'seal': base + '/seal.json', 'release': base + '/release.json',
            'CSV': f'outputs/submission_v{version}_trial.csv',
            'rows': f'outputs/research_csv_intermediate_20261008/v{version}_rows.json',
            'plan': f'reports/round{round_number}_final_plan.md',
            'preflight': FOLDER + f'round{round_number}_final_release_preflight.json'}


def gate(review, results, number):
    c.need(review['status'] == 'passed' and results['status'] == 'three_reused_windows_evaluated',
           'Independent completed batch postfit review required')
    c.need(results['gates']['exploratory_submission']['passed'] is True
           and review['manual_gate_verdicts'][str(number)]['exploratory_submission_eligible'] is True,
           'This candidate must independently pass its original exploratory gate')
    c.need(results['Private_score'] is None and results['gates']['private_score'] is None
           and results['gates']['selected_submission'] == 'v22'
           and results['gates']['automatic_selection'] is False, 'Preserve unknown Private and v22 selection')


def extra_gate(review, evidence, number):
    """Required R113 gate; independent review binds evidence without a SHA cycle."""
    c.need(review['status'] == 'passed' and evidence['status'] == 'fixed_extra_origin_evaluated',
           'Independent completed R113 gate review is mandatory')
    c.need(evidence['candidates'] == [107, 111, 112] and set(evidence['candidate_gates']) == {'107', '111', '112'},
           'All three R113 methods must have been fixed before any new scoring')
    item = evidence['candidate_gates'][str(number)]
    c.need(set(item['components']) == EXTRA_COMPONENTS and all(v is True for v in item['components'].values())
           and item['passed'] is True, 'Required extra-origin regression and adverse-group gate must pass')
    c.need(evidence['Private_score'] is None and evidence['Public_score'] is None
           and evidence['selected_submission'] == 'v22' and evidence['new_CSV'] is False,
           'R113 has no CSV or claimed Public/Private score')


def required_extra_records(number):
    review, evidence = c.read(EXTRA_REVIEW), c.read(EXTRA_GATE)
    extra_gate(review, evidence, number)
    c.need(c.sha(EXTRA_PLAN) == EXTRA_PLAN_SHA, 'Fixed pre-scoring R113 plan identity')
    c.need(review['release_gate'] == c.record(EXTRA_GATE) and review['results'] == c.record(EXTRA_RESULTS)
           and review['plan'] == c.record(EXTRA_PLAN), 'Independent R113 review exact gate/result/plan records')
    c.need(evidence['results'] == review['results'] and evidence['plan'] == review['plan'],
           'R113 gate/result/plan authority agrees with independent review')
    c.need(isinstance(review['source_sha256'], dict) and bool(review['source_sha256']),
           'R113 independent review must bind actual source identities')
    c.verify_hashes(review['source_sha256'])
    inputs = dict(review['source_sha256'])
    bind_records(review, inputs)
    bind_records(evidence, inputs)
    for name in (EXTRA_REVIEW, EXTRA_GATE, EXTRA_RESULTS, EXTRA_PLAN):
        inputs[name] = c.sha(name)
    return inputs, {'gate': c.record(EXTRA_GATE), 'review': c.record(EXTRA_REVIEW),
                    'results': c.record(EXTRA_RESULTS), 'plan': c.record(EXTRA_PLAN),
                    'candidate_gate': evidence['candidate_gates'][str(number)]}


def bind_records(value, inputs):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            name, digest = c.rel(value['path']), value['sha256']
            c.need(name not in inputs or inputs[name] == digest, 'Conflicting reviewed record')
            c.need(c.sha(name) == digest, 'Reviewed record changed: ' + name)
            inputs[name] = digest
        for child in value.values():
            bind_records(child, inputs)
    elif isinstance(value, list):
        for child in value:
            bind_records(child, inputs)


def sources():
    result = c.source_hashes(__file__, TRAINER, HELPER, CHECKPOINT_REVIEW)
    result[BUILDER] = c.sha(BUILDER)
    return result


def freeze(p):
    c.need(not c.safe(p['base']).exists() and not c.safe(p['CSV']).exists()
           and not c.safe(p['rows']).exists(), 'Preserve any previous final attempt, rows or CSV')
    review_path, result_path = FOLDER + 'postfit_review.json', f"outputs/round{p['round']}/results.json"
    review, results = c.read(review_path), c.read(result_path)
    gate(review, results, p['round'])
    extra_inputs, extra_records = required_extra_records(p['round'])
    c.need(review['results'][str(p['round'])] == c.record(result_path), 'Reviewed result identity')
    for key in ('request', 'seal', 'training_execution', 'evaluation_execution', 'review_tool', 'plan'):
        c.need(review[key] == c.record(review[key]['path']), 'Completed independent review identity: ' + key)
    c.need(c.read(review['training_execution']['path'])['exit_code'] == 0
           and c.read(review['evaluation_execution']['path'])['exit_code'] == 0,
           'All six fits and formal evaluation must complete before final freeze')
    batch = c.read(review['request']['path'])
    c.need(batch['status'] == 'frozen_before_six_fits'
           and review['request'] == results['request'] and review['seal'] == results['seal'],
           'Original batch request and prediction seal')
    c.need(review['source_sha256'] == {**batch['source_sha256'], history.PLAN: c.sha(history.PLAN)},
           'Independent batch review must bind exact frozen source closure and original plan')
    c.verify_hashes(review['source_sha256'])
    recipes = []
    for stage in c.STAGES:
        key = f"{p['round']}/{stage}"
        rec = review['artifacts'][key]['training_metadata']
        c.need(c.record(rec['path']) == rec, 'Reviewed native training metadata identity')
        metadata = c.read(rec['path'])
        c.need(metadata == review['trainmeta'][key], 'Reviewer training metadata projection differs')
        if p['round'] == 111:
            c.need(metadata['source_sha256'] == c.sha(TRAINER), 'Reviewed R111 trainer source')
        else:
            import round112_ordered_unit as helper
            c.need(metadata['source_sha256'] == helper.source_hashes(), 'Reviewed R112 helper sources')
        recipes.append(metadata['configuration' if p['round'] == 111 else 'parameters'])
    c.need(all(recipe == recipes[0] for recipe in recipes), 'Same preregistered recipe in all three windows')
    preflight, current_sources = c.read(p['preflight']), sources()
    c.need(preflight['status'] == 'passed' and preflight['round'] == p['round']
           and preflight['version'] == p['version'], 'Independent round/version final source preflight required')
    c.need(preflight['source_sha256'] == {**current_sources, p['plan']: c.sha(p['plan'])},
           'Final preflight must bind exact source closure, Artifact builder and separate final plan')
    c.verify_hashes(preflight['source_sha256'])
    context = final.specification()
    inputs = dict(context['input_sha256'])
    inputs.update(review['source_sha256'])
    for name, digest in extra_inputs.items():
        c.need(name not in inputs or inputs[name] == digest, 'Conflicting R113 source/input binding')
        inputs[name] = digest
    bind_records(review, inputs)
    for name in (review_path, result_path, p['plan'], p['preflight'],
                 'outputs/research_resume_20261008/final_context_preflight.json'):
        inputs[name] = c.sha(name)
    c.verify_hashes(current_sources)
    c.verify_hashes(inputs)
    c.safe(p['base']).mkdir(parents=True, exist_ok=False)
    c.write(p['request'], {'status': 'frozen_before_one_exploratory_final_fit', 'created_at_utc': c.now(),
            'round': p['round'], 'version': p['version'], 'paths': p, 'sources': current_sources, 'inputs': inputs,
            'context': context, 'expected_recipe': recipes[0], 'protected_before': history.protected(),
            'postfit_review': c.record(review_path), 'gates': results['gates'], 'new_fits': 1,
            'required_extra_origin': extra_records,
            'query_labels_available': False, 'selected_submission_preserved': 'v22',
            'Public_score': None, 'Private_score': None})


def request(p):
    r = c.read(p['request'])
    c.need(r['status'] == 'frozen_before_one_exploratory_final_fit' and r['paths'] == p
           and r['round'] == p['round'] and r['version'] == p['version'] and r['new_fits'] == 1,
           'Correct immutable final request required')
    extra_inputs, extra_records = required_extra_records(p['round'])
    c.need(r['required_extra_origin'] == extra_records
           and all(r['inputs'].get(name) == digest for name, digest in extra_inputs.items()),
           'Required independent R113 gate remains exactly frozen')
    c.verify_hashes(r['sources'])
    c.verify_hashes(r['inputs'])
    c.verify_hashes(r['protected_before'])
    return r


def prepared(ctx):
    original = c.read(ctx['specification']['original_training']['path'])
    preparation = c.read(ctx['specification']['preparation']['path'])
    with np.load(c.safe(ctx['prepared']['path']), allow_pickle=False) as z:
        c.need(set(z.files) == set(c.ARRAY_NAMES) and len(z.files) == len(c.ARRAY_NAMES),
               'Original nine keys required before reading values; no query labels')
        c.exact(z['train_y_million'], np.asarray(ctx['y'] / 1e6, dtype=np.float32), 'Canonical full fit y alignment')
        c.need(z['train_num'].shape == (467221, 144) and z['train_cat'].shape == (467221, 23)
               and z['query_num'].shape == (47073, 144) and z['query_cat'].shape == (47073, 23), 'Final NN dimensions')
        identities = {key: c.identity(z[key]) for key in c.ARRAY_NAMES}
        for key in ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
            c.need(identities[key]['values_sha256'] == original[key + '_sha256'], 'Original final ' + key)
        c.need(identities['train_sample_weight']['values_sha256'] == preparation['train_sample_weight_sha256']
               and z['cardinalities'].tolist() == original['cardinalities'], 'Original date weights/cardinalities')
        weights, query_factors = z['train_sample_weight'].copy(), z['query_price_factor_million'].copy()
    return original, identities, weights, query_factors


def unit_chain(ctx, native):
    c.need(native.dtype == np.float64 and native.shape == (47073,) and np.isfinite(native).all()
           and (native >= 0).all(), 'Finite nonnegative unit-model TWD query vector')
    fixed = ctx['vectors']
    base = .75 * native + .25 * fixed['lgb']
    reference, diagnostics, _ = final.f79.exp.original.predict_reference(
        ctx['reference'], ctx['Xq'], base, variant='trend', return_diagnostics=True)
    spatial, detail, _ = final.f79.spatial.predict(
        ctx['spatial_state'], ctx['Xq'], ctx['query_geo'], base, reference, diagnostics)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback], reference[fallback], 'Unit spatial fallback exact')
    median = np.median(np.column_stack([fixed['nn'], fixed['log'], spatial]), axis=1)
    return {'native': native, 'base': base, 'reference': reference, 'spatial': spatial, 'median': median}


def check_nn(meta, r, original, identities, factors, model):
    c.need(meta['status'] == 'completed' and meta['source_sha256'] == r['sources'][TRAINER]
           and meta['configuration'] == r['expected_recipe'] and meta['baseline_configuration'] == original['configuration']
           and meta['actual_model_class'] == 'round111_residual_prior_train.TabularResidualPrior'
           and meta['complete_R63_nonarchitecture_training_configuration_unchanged']
           and meta['original_configuration_and_source_verified'] and not meta['original_globals_modified']
           and meta['training_objective_name'] == 'weighted MAE' and not meta['epoch_selection_uses_training_metrics'],
           'Actual fixed residual architecture with original R63 objective')
    c.need(meta['input_sha256'] == r['context']['prepared']['sha256'] and meta['input_arrays'] == identities
           and meta['fit_rows'] == 467221 and meta['query_rows'] == meta['reload_rows'] == 47073
           and meta['numeric_count'] == 144 and meta['categorical_count'] == 23
           and meta['cardinalities'] == original['cardinalities'] and not meta['query_labels_used']
           and not meta['query_labels_available'] and not meta['sample_weights_batch_renormalized']
           and not meta['price_or_ratio_clipping'] and meta['saved_model_reload_exact']
           and meta['saved_native_ratio_reload_exact'], 'Actual R111 nine-input/scope/full-reload contract')
    c.need(meta['parent_trainer_source_sha256'] == original['source_sha256']
           and meta['original_architecture_source_sha256'] == original['architecture_source_sha256'],
           'Original R63/R47 source provenance')
    for key in ('train_price_factor_million_sha256', 'query_price_factor_million_sha256',
                'train_sample_weight_sha256', 'initial_training_median_ratio', 'initial_head_bias'):
        c.need(meta[key] == original[key], 'Original final preparation: ' + key)
    c.need([e['epoch'] for e in meta['epochs']] == list(range(1, 31))
           and [e['learning_rate'] for e in meta['epochs']] == [e['learning_rate'] for e in original['epochs']]
           and all(np.isfinite(e['training_mae_million_with_dropout'])
                   and np.isfinite(e['training_weighted_mae_million_with_dropout']) for e in meta['epochs']),
           'Fixed complete epoch/schedule timeline')
    started = c.read(model / 'run.json')
    c.need(started['status'] == 'started' and all(meta[k] == v for k, v in started.items() if k != 'status'),
           'Immutable started metadata differs from actual completed recipe')
    c.need(c.sha(model / 'model.pt') == meta['model_sha256'], 'Actual R111 model SHA')
    records = {'model': c.record(model / 'model.pt'), 'run': c.record(model / 'run.json')}
    c.need(set(meta['prediction_artifacts']) == {'predictions_ratio.npy', 'predictions_twd.npy'}, 'Both R111 native artifacts')
    for name, item in meta['prediction_artifacts'].items():
        a = np.load(model / name, allow_pickle=False)
        c.need(c.sha(model / name) == item['sha256'] and c.identity(a) == {k: item[k] for k in c.identity(a)}
               and a.dtype == np.float64 and a.shape == (47073,) and np.isfinite(a).all() and (a > 0).all(),
               'Actual R111 saved ratio/TWD identity')
        records[name] = c.record(model / name)
    ratios = np.load(model / 'predictions_ratio.npy', allow_pickle=False)
    native = np.load(model / 'predictions_twd.npy', allow_pickle=False)
    c.exact(ratios * factors.astype(np.float64) * 1e6, native, 'Actual native-ratio to TWD inverse exactly once')
    command = [str(c.ROOT / '.venv_torch/Scripts/python.exe'), '-B', '-X', 'utf8', str(c.safe(CHECKPOINT_REVIEW)),
               '--model', str(model / 'model.pt'), '--training-meta', str(model / 'training.json')]
    process = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True, encoding='utf8', check=False)
    with (model / 'checkpoint_cpu_review_execution.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump({'command': command, 'exit_code': process.returncode, 'stdout': process.stdout,
                   'stderr': process.stderr}, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    c.need(process.returncode == 0, 'CPU checkpoint structure check failed; preserve evidence')
    review = json.loads(process.stdout)
    c.need(review['status'] == 'passed' and review['model']['sha256'] == meta['model_sha256']
           and review['training']['sha256'] == c.sha(model / 'training.json')
           and review['review_tool_sha256'] == r['sources'][CHECKPOINT_REVIEW]
           and review['trainer_source_sha256'] == r['sources'][TRAINER]
           and review['new_fits'] == review['GPU_calls'] == review['native_model_inference_calls'] == 0,
           'Isolated CPU actual-checkpoint evidence')
    c.write(model / 'checkpoint_cpu_review.json', review)
    records['checkpoint_cpu_review'] = c.record(model / 'checkpoint_cpu_review.json')
    records['checkpoint_cpu_review_execution'] = c.record(model / 'checkpoint_cpu_review_execution.json')
    return records


def check_ordered(meta, r, ctx, weights, model):
    import round112_ordered_unit as helper
    c.need(meta['status'] == 'completed' and meta['parameters'] == r['expected_recipe'] == helper.parameters()
           and meta['baseline_parameters'] == helper.baseline.parameters()
           and meta['changed_parameter_names'] == ['boosting_type'] and meta['added_parameter_names'] == []
           and meta['source_sha256'] == helper.source_hashes() and meta['fit_rows'] == 467221
           and meta['query_rows'] == 47073 and meta['new_native_fits'] == 1
           and meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23
           and meta['actual_trees'] == 2999 and meta['actual_max_ctr_complexity'] == 2
           and meta['actual_ctr_target_border_count'] == 1 and meta['actual_seed'] == 42
           and meta['actual_boosting_type'] == 'Ordered' and meta['actual_task_type'] == 'GPU'
           and meta['actual_loss_function'] == 'MAE', 'Actual R79 recipe with only Ordered boosting changed')
    c.need(meta['saved_model_reload_exact'] and meta['raw_and_TWD_reload_exact'] and meta['reload_rows'] == 47073
           and not meta['query_labels_used'] and not meta['query_labels_available']
           and not meta['fit_uses_eval_set'] and not meta['early_stopping_used']
           and not meta['date_weights_recomputed'] and not meta['date_weights_renormalized']
           and not meta['weights_normalized_to_mean_one'] and not meta['weights_use_query']
           and not meta['weights_use_fit_target'] and not meta['reference_or_trend_applied_here']
           and not meta['spatial_applied_here'] and not meta['extra_price_cap_applied'], 'Original unit weight/query/reload scope')
    c.need(meta['target_mode'] == helper.TARGET_MODE and meta['target_formula'] == helper.TARGET_FORMULA
           and meta['inverse_formula'] == helper.INVERSE_FORMULA and meta['sample_weight'] == helper.WEIGHT_FORMULA
           and meta['weight_dtype'] == 'float64' and meta['weight_argument_passed_to_pool'],
           'Original unit target, inverse and date-area Pool semantics')
    for name, key in [('fit_X', 'X'), ('query_X', 'Xq'), ('fit_geo', 'G'), ('query_geo', 'Gq')]:
        c.need(meta['frame_sha256'][name] == ctx['frame_identities'][key], 'Actual R112 canonical frame: ' + name)
    dates = helper.baseline._date_values(weights, ctx['X'])
    expected = {'fit_target_values_sha256': helper.original.values_sha256(ctx['y']),
                'fit_transformed_target_values_sha256': helper.original.values_sha256(helper.original.transform_target(ctx['y'], ctx['X'])),
                'fit_date_weights_sha256': c.values_sha(dates),
                'fit_area_weights_sha256': helper.original.values_sha256(helper.original.sample_weights(ctx['X'])),
                'fit_sample_weights_sha256': helper.original.values_sha256(helper.sample_weights(ctx['X'], weights))}
    for key, value in expected.items():
        c.need(meta[key] == value, 'Actual aligned R112 target/weights: ' + key)
    state = c.read(model / 'feature_state.json')
    helper._check_state(state, ctx['X'], ctx['cats'])
    helper._check_state(state, ctx['Xq'], ctx['cats'])
    comparison = helper.compare_reload_diagnostics(meta['fit_native_diagnostics'], meta['reloaded_native_diagnostics'])
    c.need(comparison == meta['reload_diagnostics_comparison'], 'Recorded native serialization tolerance evidence')
    restored = helper.CatBoostRegressor()
    restored.load_model(str(model / 'model.cbm'))
    diagnostics = helper.check_model(restored, state)
    c.need(diagnostics == meta['reloaded_native_diagnostics'], 'Actual CPU-loaded CBM complete native recipe')
    records = {}
    for name, key in (('model.cbm', 'model_sha256'), ('feature_state.json', 'feature_state_sha256'),
                      ('raw.npy', 'raw_prediction_sha256'), ('native.npy', 'prediction_sha256')):
        c.need(c.sha(model / name) == meta[key], 'Actual R112 saved artifact identity: ' + name)
        records[name] = c.record(model / name)
    records['started'] = c.record(model / 'started.json')
    raw, native = [np.load(model / name, allow_pickle=False) for name in ('raw.npy', 'native.npy')]
    c.exact(helper.original.inverse_transform(raw, ctx['Xq']), native, 'Saved unit-to-TWD inverse exact')
    return records


def fit(p):
    r = request(p)
    c.need(history.protected() == r['protected_before'] and not c.safe(p['CSV']).exists(),
           'Serialize final candidates; protected inventory must still match')
    c.need(not c.safe(p['base'] + '/fit_started.json').exists(), 'One final fit invocation only; preserve failures')
    c.write(p['base'] + '/fit_started.json', {'created_at_utc': c.now(), 'request': c.record(p['request']),
            'new_fits_authorized': 1, 'automatic_retry': False})
    tick, failure = time.monotonic(), None
    try:
        ctx = final.load_context(r['context'])
        c.need(set(ctx['replay_checks']) == shared.REPLAY_KEYS and all(v is True for v in ctx['replay_checks'].values()),
               'All eleven original v22 replay checks required')
        original, identities, weights, factors = prepared(ctx)
        model = c.safe(p['base']) / 'model'
        if p['round'] == 111:
            command = [str(c.ROOT / '.venv_torch/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(c.safe(TRAINER)),
                       '--data', str(c.safe(ctx['prepared']['path'])), '--out', str(model)]
            with (c.safe(p['base']) / 'train.log').open('x', encoding='utf8') as log:
                process = subprocess.run(command, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            c.need(process.returncode == 0, 'R111 single final fit failed; no retry')
            metadata_path = model / 'training.json'
            meta = c.read(metadata_path)
            artifacts = check_nn(meta, r, original, identities, factors, model)
            native = np.load(model / 'predictions_twd.npy', allow_pickle=False)
            chain = final.apply_prior(ctx, native)[0]
        else:
            import round112_ordered_unit as helper
            with (c.safe(p['base']) / 'train.log').open('x', encoding='utf8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    native, meta = helper.fit_predict(ctx['X'], ctx['y'], ctx['G'], ctx['Xq'], ctx['Gq'],
                                                    ctx['cats'], weights, model)
            metadata_path = model / 'metadata.json'
            c.need(c.read(metadata_path) == meta, 'Saved R112 metadata equality')
            artifacts = check_ordered(meta, r, ctx, weights, model)
            original_unit = unit_chain(ctx, ctx['vectors']['unit_native'])
            for key, saved in [('base', 'unit_base'), ('reference', 'unit_reference'), ('spatial', 'unit'), ('median', 'v22')]:
                c.exact(original_unit[key], ctx['vectors'][saved], 'Original unit final replay: ' + key)
            chain = unit_chain(ctx, native)
        c.need(set(chain) == shared.LAYERS, 'All five final layers')
        c.exact(chain['native'], native, 'One corresponding native slot replaced')
        for key, value in chain.items():
            c.need(value.dtype == np.float64 and value.shape == (47073,) and np.isfinite(value).all()
                   and (value >= 0).all(), 'Five-layer TWD vector: ' + key)
        records = {key: c.save_array(c.safe(p['base']) / (key + '.npy'), value) for key, value in chain.items()}
        final.verify_unchanged(ctx)
        request(p)
        c.need(history.protected() == r['protected_before'], 'All old protected files unchanged before CSV')
        c.write(p['seal'], {'status': 'full_model_prediction_sealed', 'request': c.record(p['request']),
                'round': p['round'], 'version': p['version'], 'model_artifacts': artifacts,
                'training': c.record(metadata_path), 'layers': records, 'v22_replay_checks': ctx['replay_checks'],
                'prepared': ctx['prepared'], 'prepared_array_identities': identities, 'new_fits': 1,
                'fit_seconds': meta['fit_seconds'], 'elapsed_seconds': time.monotonic() - tick,
                'query_labels_used': False, 'new_CSV': False})
        rounded = shared.rounded_prices(chain['median'])
        c.safe(p['rows']).parent.mkdir(parents=True, exist_ok=True)
        c.write(p['rows'], {'status': 'full_model_prediction_sealed', 'seal': c.record(p['seal']),
                'rounding': 'maximum(rint(final_unrounded_median_TWD),1).astype(int64)',
                'rows': [[key, int(value)] for key, value in zip(ctx['ids'], rounded, strict=True)]})
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        c.write(p['base'] + '/fit_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
                'elapsed_seconds': time.monotonic() - tick, 'automatic_retry': False})


def release(p):
    """Validate root-authored Artifact CSV; no fit, builder invocation or CSV writes."""
    r, tick = request(p), time.monotonic()
    c.need(not c.safe(p['release']).exists(), 'Preserve immutable release; no second authoring attempt')
    before, after = r['protected_before'], history.protected()
    c.need(set(after) == set(before) | {p['CSV']} and all(after[k] == v for k, v in before.items()),
           'Root must author exactly one candidate CSV with all previous protected files unchanged')
    seal = c.read(p['seal'])
    c.need(seal['status'] == 'full_model_prediction_sealed' and seal['request'] == c.record(p['request'])
           and seal['round'] == p['round'] and seal['version'] == p['version'] and seal['new_fits'] == 1
           and not seal['query_labels_used'] and seal['new_CSV'] is False, 'Immutable final prediction seal')
    execution = c.read(p['base'] + '/fit_execution.json')
    c.need(execution['exit_code'] == 0 and execution['failure'] is None, 'Final fit completed successfully')
    artifacts = {}
    bind_records(seal['model_artifacts'], artifacts)
    bind_records(seal['training'], artifacts)
    c.need(set(seal['layers']) == shared.LAYERS, 'Five sealed layers required')
    layers = {key: c.load_array(rec) for key, rec in seal['layers'].items()}
    c.need(set(seal['v22_replay_checks']) == shared.REPLAY_KEYS
           and all(v is True for v in seal['v22_replay_checks'].values()), 'Original v22 replay evidence')
    rows = c.read(p['rows'])
    c.need(rows['status'] == 'full_model_prediction_sealed' and rows['seal'] == c.record(p['seal']), 'Authoring rows seal')
    prediction, rounded = layers['median'], shared.rounded_prices(layers['median'])
    with c.safe(r['context']['sample_path']).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        c.need(next(reader) == ['編號', '總價元'], 'Original sample headers')
        sample = list(reader)
    c.need(len(sample) == 47073 and all(len(row) == 2 for row in sample), 'All sample rows/schema')
    ids = [row[0] for row in sample]
    c.need(len(set(ids)) == 47073 and rows['rows'] == [[key, int(value)] for key, value in zip(ids, rounded, strict=True)],
           'Every ID and once-rounded median value in authoring rows')
    expected_bytes = ('\ufeff' + '\r\n'.join(['編號,總價元'] +
                      [key + ',' + str(int(value)) for key, value in zip(ids, rounded, strict=True)]) + '\r\n').encode('utf8')
    c.need(c.safe(p['CSV']).read_bytes() == expected_bytes, 'Every CSV byte, header, BOM, ID/order and rounded value')
    authoring_path = p['base'] + '/csv_authoring_receipt.json'
    authoring = c.read(authoring_path)
    c.need(authoring['status'] == 'artifact_typed_range_exported_to_csv' and authoring['input_json_sha256'] == c.sha(p['rows'])
           and authoring['csv_sha256'] == c.sha(p['CSV']) and authoring['rows'] == 47073
           and authoring['columns'] == ['編號', '總價元'] and authoring['full_workbook_values_exact']
           and authoring['formula_count'] == 0 and authoring['utf8_bom'], 'Existing Artifact typed-range export evidence')
    c.need(c.safe(p['base'] + '/csv_preview.png').is_file(), 'Root builder preview evidence')
    request(p)
    c.need(history.protected() == after, 'No protected change during release validation')
    original = np.load(c.safe(final.VECTORS['v22']), allow_pickle=False)
    difference = prediction - original
    receipt = {'status': 'exploratory_submission_ready_not_selected', 'created_at_utc': c.now(),
               'round': p['round'], 'version': p['version'], 'CSV': c.record(p['CSV']), 'rows': 47073,
               'columns': ['編號', '總價元'], 'request': c.record(p['request']), 'seal': c.record(p['seal']),
               'training': seal['training'], 'authoring': c.record(authoring_path),
               'all_sample_ids_and_rounded_values_exact': True, 'all_CSV_bytes_exact': True,
               'CSV_once_rounded_from_unrounded_median': True, 'positive_integer_prices': True,
               'positive_safe_integer_prices': True, 'same_order_as_sample': True, 'old_protected_unchanged': True,
               'query_labels_used': False, 'Public_score': None, 'Private_score': None,
               'automatic_Kaggle_upload_or_selection': False, 'selected_submission_preserved': 'v22', 'gate': r['gates'],
               'required_extra_origin': r['required_extra_origin'],
               'CSV_authored_by_root_Artifact_builder': True, 'CSV_builder_invoked_in_release': False,
               'test_prediction_difference_not_MAE': {'mean_signed_TWD': float(difference.mean()),
                   'mean_absolute_TWD': float(np.mean(np.abs(difference))), 'max_absolute_TWD': float(np.max(np.abs(difference))),
                   'changed_rounded_rows': int(np.count_nonzero(rounded != shared.rounded_prices(original)))},
               'prediction_min_TWD': float(prediction.min()), 'prediction_max_TWD': float(prediction.max()),
               'elapsed_seconds': time.monotonic() - tick, 'clean_environment_end_to_end_retraining_verified': False}
    c.write(p['release'], receipt)
    print(json.dumps({'CSV': p['CSV'], 'sha256': receipt['CSV']['sha256'], 'rows': 47073, 'status': receipt['status']}), flush=True)


def self_check():
    import copy
    checks = []
    def ok(name, condition):
        c.need(condition, name)
        checks.append(name)
    def rejects(name, call):
        try:
            call()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError('Accepted invalid contract: ' + name)
    ok('separate_fixed_round_version_paths', paths(111, 28)['base'] != paths(112, 29)['base'])
    rejects('swapped_version_rejected', lambda: paths(111, 29))
    rejects('old_round_rejected', lambda: paths(109, 26))
    review = {'status': 'passed', 'manual_gate_verdicts': {'111': {'exploratory_submission_eligible': True},
                                                       '112': {'exploratory_submission_eligible': False}}}
    results = {'status': 'three_reused_windows_evaluated', 'Private_score': None,
               'gates': {'exploratory_submission': {'passed': True}, 'private_score': None,
                         'selected_submission': 'v22', 'automatic_selection': False}}
    gate(review, results, 111)
    ok('eligible_review_and_original_gate', True)
    rejects('ineligible_round_rejected', lambda: gate(review, results, 112))
    bad = copy.deepcopy(review)
    bad['status'] = 'pending'
    rejects('pending_review_rejected', lambda: gate(bad, results, 111))
    bad = copy.deepcopy(results)
    bad['gates']['exploratory_submission']['passed'] = False
    rejects('failed_exploratory_gate_rejected', lambda: gate(review, bad, 111))
    bad = copy.deepcopy(results)
    bad['Private_score'] = 1.
    rejects('unknown_Private_not_claimed', lambda: gate(review, bad, 111))
    bad = copy.deepcopy(results)
    bad['gates']['automatic_selection'] = True
    rejects('automatic_selection_rejected', lambda: gate(review, bad, 111))
    extra = {'status': 'fixed_extra_origin_evaluated', 'candidates': [107, 111, 112],
             'candidate_gates': {str(n): {'passed': True, 'components': {key: True for key in EXTRA_COMPONENTS}}
                                 for n in (107, 111, 112)},
             'Private_score': None, 'Public_score': None, 'selected_submission': 'v22', 'new_CSV': False}
    extra_gate({'status': 'passed'}, extra, 111)
    ok('required_R113_extra_origin_gate', True)
    bad = copy.deepcopy(extra)
    bad['candidate_gates']['111']['components']['weighted_regression_at_most_0_25pct'] = False
    rejects('R113_weighted_failure_rejected', lambda: extra_gate({'status': 'passed'}, bad, 111))
    rejects('R113_pending_review_rejected', lambda: extra_gate({'status': 'pending'}, extra, 111))
    bad = copy.deepcopy(extra)
    bad['candidates'] = [111]
    rejects('R113_selected_after_scoring_subset_rejected', lambda: extra_gate({'status': 'passed'}, bad, 111))
    values = np.full(47073, 2.5, dtype=np.float64)
    values[:5] = [0., .5, 1.5, 2.5, 3.5]
    ok('round_once_from_unrounded_median_and_minimum_one', shared.rounded_prices(values)[:5].tolist() == [1, 1, 2, 2, 4])
    for label, value in [('NaN', np.nan), ('negative', -1.), ('unsafe_integer', float(2**53))]:
        rejects('rounding_rejects_' + label, lambda value=value: shared.rounded_prices(np.full(47073, value)))
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks),
            'real_data_read': False, 'query_prices_or_scores_read': False, 'new_fits': 0, 'GPU_calls': 0,
            'new_CSV': False, 'builder_invocations': 0, 'final_freeze_executed': False,
            'source_sha256': sources(), 'author_selfcheck_not_independent_final_preflight': True,
            'future_requirements': 'Independent completed batch postfit, original exploratory gate, mandatory reviewed R113 extra gate and separate round final plan/preflight.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', dest='round_number', type=int, choices=(111, 112))
    parser.add_argument('--version', type=int, choices=(28, 29))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--fit', action='store_true')
    group.add_argument('--release', action='store_true')
    group.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        if args.round_number is not None or args.version is not None:
            parser.error('Synthetic self-check does not accept real round/version operations')
        print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    else:
        if args.round_number is None or args.version is None:
            parser.error('--round and --version explicitly required')
        selected = paths(args.round_number, args.version)
        if args.freeze:
            freeze(selected)
        elif args.fit:
            fit(selected)
        else:
            release(selected)
