"""R117: fixed transaction-fold fit priors, unchanged R63 ratio NN, four origins.

Only the coordinator invokes freeze/train/evaluate after R115 postfit and the
complete independent preflight. Source/CPU preparation is not authorization.
Every new private prior/prepared/model artifact and five-layer composition is
sealed before the separate scorer may deserialize raw query prices. No CSV,
full-data fit, automatic retry, parameter grid or shared split mutation exists.
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
import round114_research as baseline
import round115_prior_support_x_audit as support
import round117_transaction_prior as prior

PLAN = 'reports/round117_plan.md'
PLAN_SHA = 'c4084bee6bbe0f9ed97e019a88f48cbeee992f3a297a1b6ce251709f5377e8eb'
TRAINER = 'src/round117_transaction_prior_train.py'
BASE = 'outputs/round117'
EVIDENCE = 'outputs/research_round117_20261008'
REQUEST, SEAL, RESULTS = (BASE + '/' + n + '.json' for n in ('request', 'seal', 'results'))
R115_REVIEW = 'outputs/research_round115_20261008/postfit_review.json'
OFFSET = baseline.OFFSET
ORIGINS = baseline.ORIGINS
COUNTS = baseline.COUNTS
LAYERS = baseline.LAYERS
RAW_SHA = baseline.RAW_SHA
CHANGED_ARRAY = 'train_price_factor_million'
UNCHANGED_ARRAYS = tuple(k for k in c.ARRAY_NAMES if k != CHANGED_ARRAY)
CAUTION = ('Four reused overlapping origins are not independent votes or fresh blind validation. '
           'Transaction OOF excludes each whole transaction fold, not all same-project peers; '
           'it is neither chronological OOF nor per-row as-of prediction. Prior factors and the '
           'linked initial ratio/head bias change together. Original group OOF may generalize '
           'better to new projects. Conditional single-seed bootstrap omits cumulative selection, '
           'retraining and future shift; not Private probability. R67/R100/R65/R102 contrary '
           'evidence and v24 comparisons remain; native improvement alone is insufficient.')


def sources():
    return c.source_hashes(__file__, TRAINER)


def protected():
    return history.protected()


def trainer_recipe():
    """AST-only provenance overrides; do not import Torch or run the trainer."""
    tree = ast.parse(c.safe(TRAINER).read_text(encoding='utf-8-sig'))
    found = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'PROVENANCE_CONFIG_OVERRIDES' for t in n.targets)]
    c.need(len(found) == 1, 'One literal descriptive-only R63 override required')
    value = found[0]
    c.need(isinstance(value, ast.Dict) and len(value.keys) == len(value.values) == 1
           and ast.literal_eval(value.keys[0]) == 'train_price_factor_construction'
           and isinstance(value.values[0], ast.Name) and value.values[0].id == 'TRAIN_FACTOR_CONSTRUCTION',
           'Only the known provenance literal constant may be referenced')
    constants = [n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'TRAIN_FACTOR_CONSTRUCTION' for t in n.targets)]
    c.need(len(constants) == 1, 'One literal transaction provenance string required')
    overrides = {'train_price_factor_construction': ast.literal_eval(constants[0])}
    c.need(isinstance(overrides, dict) and set(overrides) == {'train_price_factor_construction'}
           and 'transaction' in overrides['train_price_factor_construction']
           and 'group-fold' not in overrides['train_price_factor_construction'],
           'Only transaction-fold provenance description may differ from R63 CONFIG')
    name = 'outputs/round63/development/model/training.json'
    old = c.read(name)['configuration']
    c.need(old['seed'] == 42 and old['epochs'] == 30 and old['batch_size'] == 2048
           and old['loss_formula'] == 'mean(train_sample_weight * abs(ratio_output * train_price_factor_million - train_y_million))'
           and old['price_or_ratio_clipping'] is False, 'Original R63 recipe differs')
    return {'baseline_configuration': old, 'provenance_config_overrides': overrides,
            'configuration': {**old, **overrides}, 'baseline_training': c.record(name)}


def formal_checks():
    return {'R117_prior_selfcheck_execution.json': ('src/round117_transaction_prior.py', '.venv', '--self-check'),
            'R117_trainer_selfcheck_execution.json': (TRAINER, '.venv_torch', '--cpu-unit-check'),
            'R117_driver_selfcheck_execution.json': (c.rel(__file__), '.venv', '--self-check')}


def check_formal_checks(inputs):
    for name, (entry, environment, flag) in formal_checks().items():
        path = EVIDENCE + '/' + name
        check = c.read(path)
        command = [str(c.ROOT / environment / 'Scripts/python.exe'), '-B', '-X', 'utf8', str(c.safe(entry)), flag]
        c.need(check['exit_code'] == 0 and check['command'] == command
               and check['source_sha256'] == c.source_hashes(entry), 'Formal source/command check differs: ' + name)
        inputs[path] = c.sha(path)


def check_r115(inputs):
    """Require exact independent pure-X completion, not merely a passed label."""
    spec = support.specification()
    c.need(not spec['blockers'], 'R115 immutable typed readset blocked')
    r = c.read(R115_REVIEW)
    expected = {**spec['source_sha256'], support.PLAN: spec['plan']['sha256']}
    c.need(r['status'] == 'passed' and r['round'] == 115 and r['source_sha256'] == expected
           and r['plan'] == spec['plan'] and r['new_model_fits'] == 0 and r['GPU_work'] is False
           and r['new_CSV'] is False and r['prices_or_query_y_read_by_reviewer'] is False
           and r['Public_score'] is None and r['Private_score'] is None
           and r['blocking_findings_remaining'] == [] and r['source_and_input_before_after_verified'] is True,
           'R115 independent completed scope/identity required')
    for key, name in (('request', support.OUT + '/request.json'), ('started', support.OUT + '/started.json'),
                      ('results', support.OUT + '/results.json'), ('preflight_review', support.PREFLIGHT),
                      ('freeze_execution', 'outputs/research_round115_20261008/freeze_execution.json'),
                      ('run_execution', 'outputs/research_round115_20261008/run_execution.json')):
        c.need(r[key] == c.record(name), 'R115 independent exact binding: ' + key)
        inputs[name] = r[key]['sha256']
    request = c.read(r['request']['path']); result = c.read(r['results']['path'])
    c.need(request['specification'] == spec and request['input_sha256'] == spec['input_sha256']
           and result['source_sha256'] == spec['source_sha256'] and result['plan'] == spec['plan']
           and result['input_sha256'] == spec['input_sha256']
           and result['status'] == 'all_four_origins_pure_X_audit_completed'
           and set(result['stages']) == set(ORIGINS), 'R115 exact frozen/readset/result contract')
    for key in ('new_fits', 'GPU_calls', 'MAE_calls', 'bootstrap_calls', 'model_loads', 'price_prior_builds'):
        c.need(result[key] == 0, 'R115 pure-X scope: ' + key)
    for key in ('query_labels_read', 'raw_price_pickle_opened', 'NPZ_opened', 'new_CSV', 'automatic_model_experiment_authorized'):
        c.need(result[key] is False, 'R115 forbidden action: ' + key)
    c.need(result['Public_score'] is None and result['Private_score'] is None, 'R115 no price score')
    c.need(set(r['artifacts']) == set(ORIGINS) and r['artifacts'] == result['stages'], 'All four R115 aggregate bindings')
    for origin, rec in result['stages'].items():
        c.need(rec == c.record(support.OUT + '/' + origin + '/aggregates.json'), 'R115 stage exact identity')
        inputs[rec['path']] = rec['sha256']
    for key in ('freeze_execution', 'run_execution'):
        c.need(c.read(r[key]['path'])['exit_code'] == 0, 'R115 formal execution failed')
    c.verify_hashes(expected); c.verify_hashes(spec['input_sha256'])
    baseline.merge_hashes(inputs, expected); baseline.merge_hashes(inputs, spec['input_sha256'])
    inputs[R115_REVIEW] = c.sha(R115_REVIEW)
    return c.record(R115_REVIEW)


def specifications():
    """Metadata and streaming hashes only; no prior build, NPZ or raw price load."""
    inputs = {}; r115 = check_r115(inputs); check_formal_checks(inputs)
    recipe = trainer_recipe(); inputs[recipe['baseline_training']['path']] = recipe['baseline_training']['sha256']
    specs = baseline.august_specifications(inputs)
    old_name = 'outputs/round60/results.json'; inputs[old_name] = c.sha(old_name); old = c.read(old_name)
    for origin in c.STAGES:
        paths = {k: f'outputs/round60/{origin}/{name}' for k, name in (
            ('state', 'price_prior_state.json'), ('oof', 'train_oof_unit_prior.npy'),
            ('query_prior', 'query_unit_prior.npy'), ('folds', 'prior_fold_ids.npy'),
            ('evidence', 'prior_preparation.json'))}
        authority = old['stages'][origin]['artifact_sha256']
        for name in paths.values(): baseline.merge_hashes(inputs, {name: authority[name]})
        specs[origin]['original_prior'] = {k: {'path': p, 'sha256': inputs[p]} for k, p in paths.items()}
    extra = offset_comp.specification(); baseline.merge_hashes(inputs, extra['input_sha256'])
    old_seal = c.read(extra['original_seal']['path']); prepared = old_seal['NN_preparation']
    c.need(old_seal['input_sha256'][c.RAW] == inputs[c.RAW] == RAW_SHA, 'All scoring-only raw axes agree')
    paths = {k: prepared[name] for k, name in (
        ('state', 'prior_state'), ('oof', 'train_oof_unit_prior'), ('query_prior', 'query_unit_prior'),
        ('folds', 'fold_ids'), ('evidence', 'prior_evidence'))}
    old_training = 'outputs/round97/nn_models/N_prior/training.json'
    c.need(old_training in inputs, 'Original offset prior NN metadata must be bound')
    specs[OFFSET] = {'composition': extra, 'prepared': extra['prior_npz'], 'original_training': c.record(old_training),
                    'original_prior': {k: {'path': p, 'sha256': inputs[p]} for k, p in paths.items()}}
    name = 'outputs/round97/evaluation/results.json'; inputs[name] = c.sha(name)
    c.verify_hashes(inputs)
    return {'origins': specs, 'input_sha256': inputs, 'recipe': recipe, 'R115_postfit': r115}


def freeze():
    c.need(not c.safe(BASE).exists(), 'Preserve any previous R117 attempt')
    c.need(c.sha(PLAN) == PLAN_SHA, 'Fixed pre-scoring transaction-fold plan changed')
    own = sources(); spec = specifications(); path = EVIDENCE + '/preflight_review.json'; review = c.read(path)
    c.need(review['status'] == 'passed' and review['source_sha256'] == {**own, PLAN: PLAN_SHA}
           and review['input_sha256'] == spec['input_sha256'], 'Independent complete source/plan/input preflight required')
    c.verify_hashes(review['source_sha256']); inputs = {**spec['input_sha256'], path: c.sha(path)}
    c.safe(BASE).mkdir(exist_ok=False)
    c.write(REQUEST, {'status': 'frozen_before_four_fixed_origin_priors_and_NN_fits', 'created_at_utc': c.now(),
        'round': 117, 'candidate': 'R117', 'origin_order': list(ORIGINS), 'origin_counts': COUNTS,
        'plan': c.record(PLAN), 'origins': spec['origins'], 'source_sha256': own, 'input_sha256': inputs,
        'recipe': spec['recipe'], 'R115_postfit': spec['R115_postfit'], 'protected_before': protected(),
        'new_fits': 4, 'new_prior_builds': 4, 'query_scores_computed': False,
        'query_labels_used_for_fit': False, 'public_tuning': False, 'Public_score': None, 'Private_score': None,
        'selected_submission': 'v22', 'reserved_exploratory_version': 32, 'new_CSV': False, 'full_data_fit': False})


def request():
    r = c.read(REQUEST)
    c.need(r['status'] == 'frozen_before_four_fixed_origin_priors_and_NN_fits' and r['round'] == 117
           and r['origin_order'] == list(ORIGINS) and set(r['origins']) == set(ORIGINS)
           and r['new_fits'] == r['new_prior_builds'] == 4 and r['new_CSV'] is False
           and r['full_data_fit'] is False, 'Immutable four-origin scope differs')
    c.need(r['plan'] == c.record(PLAN) and r['plan']['sha256'] == PLAN_SHA, 'Plan changed')
    c.need(r['source_sha256'] == sources() and r['recipe'] == trainer_recipe(), 'Frozen source/recipe changed')
    c.verify_hashes(r['source_sha256']); c.verify_hashes(r['input_sha256'])
    check = {}; c.need(check_r115(check) == r['R115_postfit'], 'R115 prerequisite changed')
    c.need(protected() == r['protected_before'], 'Protected original artifacts changed')
    return r


def fixed_folds(rows):
    c.need(type(rows) is int and rows >= 5, 'At least five ordered fit rows required')
    permutation = np.random.default_rng(42).permutation(rows)
    folds = np.empty(rows, dtype=np.int64); folds[permutation] = np.arange(rows, dtype=np.int64) % 5
    return folds


def ratio_bias(y, factor):
    c.need(y.dtype == factor.dtype == np.float32 and y.ndim == 1 and y.shape == factor.shape
           and len(y) > 0 and np.isfinite(y).all() and (y > 0).all()
           and np.isfinite(factor).all() and (factor > 0).all(), 'Positive finite fit ratio inputs')
    median = float(np.median(y.astype(np.float64) / factor.astype(np.float64)))
    bias = median + float(np.log(-np.expm1(-median)))
    c.need(np.isfinite(median) and median > 0 and np.isfinite(np.float32(bias)), 'Finite inverseSoftplus bias')
    return {'initial_training_median_ratio': median, 'initial_head_bias_formula_float64': bias,
            'initial_head_bias_float32': float(np.float32(bias))}


def check_changed_arrays(original, revised):
    c.need(set(original) == set(revised) == set(c.ARRAY_NAMES), 'Exactly nine arrays, no query y')
    for key in UNCHANGED_ARRAYS: c.exact(original[key], revised[key], 'Unchanged eight arrays: ' + key)
    a = revised[CHANGED_ARRAY]
    c.need(a.dtype == np.float32 and a.shape == original[CHANGED_ARRAY].shape
           and np.isfinite(a).all() and (a > 0).all(), 'New fit factor must be positive finite float32')
    return {'original': {k: c.identity(v) for k, v in original.items()},
            'revised': {k: c.identity(v) for k, v in revised.items()},
            'unchanged_eight': list(UNCHANGED_ARRAYS), 'only_array_allowed_to_change': CHANGED_ARRAY}


def load_context(r, origin):
    loaded = baseline.load_context(r, origin)
    loaded['source_indices'] = (loaded['context']['data'].fit_indices if origin == OFFSET
                                else loaded['context']['indices']['fit_indices'])
    loaded['original_prior'] = r['origins'][origin]['original_prior']
    loaded['original_training'] = r['origins'][origin]['original_training']
    return loaded


def prior_records(folder):
    files = sorted(folder.iterdir())
    required = {'prepared.npz', 'preparation.json', 'price_prior_state.json', 'prior_evidence.json',
                'prior_fold_ids.npy', 'train_oof_unit_prior.npy', 'query_unit_prior.npy',
                'original_train_factor_million.npy', 'transaction_train_factor_million.npy'}
    c.need(files and all(p.is_file() and p.suffix in {'.json', '.sha256', '.npy', '.npz'} for p in files)
           and required <= {p.name for p in files}, 'Bind complete private prior/prepared artifacts')
    return [c.record(p) for p in files]


def check_prior_evidence(evidence, loaded, state, old_state, folds, oof):
    """Actual fold/fit/state identity, independent of legacy group-rule prose."""
    nf = len(loaded['fit_y'])
    c.exact(folds, fixed_folds(nf), 'Fixed transaction permutation assignment')
    c.need(state == old_state and evidence['uses_fit_labels'] is True and evidence['uses_query_labels'] is False
           and evidence['chronological_oof'] is False and evidence['fit_rows'] == nf
           and evidence['fold_policy'] == 'transaction-permutation-5-fold'
           and evidence['fold_assignment'] == 'per origin: p=np.random.default_rng(42).permutation(n); folds[p]=np.arange(n)%5'
           and evidence['fold_assignment_uses_labels_dates_groups_or_query'] is False
           and evidence['fold_assignment_identical_to_R52_rule'] is False
           and evidence['legacy_full_fit_state_unmodified'] is True
           and evidence['date_weights_rebuilt_in_each_complement_by_original_R60'] is True
           and evidence['R115_full_outer_dateweights_used_for_prior_median'] is False
           and evidence['state_content_sha256'] == prior.state_content_sha(old_state)
           and evidence['source_sha256'] == prior.source_hashes()
           and evidence['fit_target_sha256'] == c.values_sha(loaded['fit_y'])
           and evidence['source_indices_sha256'] == c.values_sha(loaded['source_indices'])
           and evidence['fold_ids_sha256'] == c.values_sha(folds)
           and evidence['oof_prior_sha256'] == c.values_sha(oof), 'Rebuilt prior fold/evidence/full-state identity')
    c.need(len(evidence['folds']) == 5 and [v['fold'] for v in evidence['folds']] == list(range(5)), 'Five exact fold evidence records')
    area = prior.effective_area(loaded['fit_X']); units = loaded['fit_y'] / area
    c.need(evidence['effective_area_sha256'] == c.values_sha(area)
           and evidence['fit_unit_sha256'] == c.values_sha(units), 'Fit unit/effective-area identities')
    for k, detail in enumerate(evidence['folds']):
        mask = folds != k
        c.need(detail['held_rows'] == int((~mask).sum()) and detail['query_rows'] == int((~mask).sum())
               and detail['fit_rows'] == detail['reference_rows'] == int(mask.sum())
               and detail['reference_indices_sha256'] == c.values_sha(loaded['source_indices'][mask])
               and detail['reference_target_sha256'] == c.values_sha(loaded['fit_y'][mask])
               and detail['reference_unit_sha256'] == c.values_sha(units[mask])
               and detail['same_groups_may_cross_transaction_folds'] is True,
               'Canonical complete transaction-fold complement evidence')


def prepare(loaded, origin, folder):
    """True fit-label prior build, invoked only in a frozen coordinated stage."""
    tick = time.monotonic(); folder.mkdir(exist_ok=False)
    original = loaded['arrays']; nf, nq = COUNTS[origin]; old = loaded['original_prior']
    old_state = c.read(old['state']['path'])
    old_oof, old_query, old_folds = [np.load(c.safe(old[k]['path']), allow_pickle=False) for k in ('oof', 'query_prior', 'folds')]
    c.need(old_oof.dtype == old_query.dtype == np.float64 and old_oof.shape == (nf,) and old_query.shape == (nq,)
           and old_folds.dtype == np.int64 and old_folds.shape == (nf,), 'Original prior artifact axes')
    for key, frame, units in (('train', loaded['fit_X'], old_oof), ('query', loaded['query_X'], old_query)):
        factor = np.asarray(prior.effective_area(frame) * units / 1e6, dtype=np.float32)
        c.exact(factor, original[key + '_price_factor_million'], 'Original factor replay: ' + key)
    c.exact(prior.predict_prior(old_state, loaded['query_X']), old_query, 'Original full-fit query prior replay')
    oof, state, folds, evidence = prior.fit_oof_prior(loaded['fit_X'], loaded['fit_y'], loaded['source_indices'])
    c.need(oof.dtype == np.float64 and oof.shape == (nf,) and np.isfinite(oof).all() and (oof > 0).all(), 'New OOF unit prices')
    check_prior_evidence(evidence, loaded, state, old_state, folds, oof)
    query_prior, counts = prior.predict_prior(state, loaded['query_X'], return_diagnostics=True)
    c.exact(query_prior, old_query, 'Full-fit query prior must be exactly unchanged')
    c.need(sum(counts.values()) == nq, 'All query prior hierarchy routes')
    fit_factor = np.asarray(prior.effective_area(loaded['fit_X']) * oof / 1e6, dtype=np.float32)
    query_factor = np.asarray(prior.effective_area(loaded['query_X']) * query_prior / 1e6, dtype=np.float32)
    c.exact(query_factor, original['query_price_factor_million'], 'Original query factor bytes unchanged')
    revised = {k: v.copy() for k, v in original.items()}; revised[CHANGED_ARRAY] = fit_factor
    identities = check_changed_arrays(original, revised)
    # Preserve the old full-fit state's complete file bytes after rebuilt dict equality.
    with (folder / 'price_prior_state.json').open('xb') as f: f.write(c.safe(old['state']['path']).read_bytes())
    c.need(c.sha(folder / 'price_prior_state.json') == old['state']['sha256'], 'Full-fit state file bytes unchanged')
    c.write(folder / 'prior_evidence.json', evidence)
    artifacts = {name: c.save_array(folder / (name + '.npy'), values) for name, values in (
        ('prior_fold_ids', folds), ('train_oof_unit_prior', oof), ('query_unit_prior', query_prior),
        ('original_train_factor_million', original[CHANGED_ARRAY]), ('transaction_train_factor_million', fit_factor))}
    with (folder / 'prepared.npz').open('xb') as f: np.savez(f, **{k: revised[k] for k in c.ARRAY_NAMES})
    reread = baseline.prepared_arrays(c.rel(folder / 'prepared.npz'), nf, nq, loaded['fit_y'])
    for key in c.ARRAY_NAMES: c.exact(revised[key], reread[key], 'New prepared archive reload: ' + key)
    old_bias = ratio_bias(original['train_y_million'], original[CHANGED_ARRAY])
    new_bias = ratio_bias(revised['train_y_million'], revised[CHANGED_ARRAY])
    old_training = c.read(loaded['original_training']['path'])
    c.need(old_training['initial_training_median_ratio'] == old_bias['initial_training_median_ratio']
           and old_training['initial_head_bias'] == old_bias['initial_head_bias_formula_float64'],
           'Original initialization must match actual saved original trainer metadata')
    preparation = {'status': 'transaction_fold_prior_prepared_without_query_labels', 'origin': origin,
        'fit_rows': nf, 'query_rows': nq, 'prepared': c.record(folder / 'prepared.npz'),
        'source_prepared': c.record(loaded['source_prepared_path']), 'array_identities': identities,
        'fold_policy': 'transaction-permutation-5-fold', 'fold_seed': 42, 'folds': 5,
        'fold_rows': [int((folds == k).sum()) for k in range(5)], 'fold_assignment': c.identity(folds),
        'original_prior': old, 'prior_artifacts': artifacts,
        'full_fit_state': c.record(folder / 'price_prior_state.json'), 'prior_evidence': c.record(folder / 'prior_evidence.json'),
        'full_fit_state_and_query_prior_and_factor_exact_original': True,
        'query_prior_identity': c.identity(query_prior), 'query_prior_level_counts': counts,
        'original_initialization': old_bias, 'transaction_initialization': new_bias,
        'original_training_metadata': loaded['original_training'], 'original_initialization_matches_saved_training': True,
        'fit_factor_change_summary': {'changed_rows': int(np.count_nonzero(fit_factor != original[CHANGED_ARRAY])),
            'changed_fraction': float(np.mean(fit_factor != original[CHANGED_ARRAY])),
            'mean_absolute_change_million': float(np.abs(fit_factor.astype(np.float64) - original[CHANGED_ARRAY].astype(np.float64)).mean()),
            'this_is_not_query_MAE': True},
        'initial_bias_changes_linked_to_fit_factor': True, 'encoder_refit': False,
        'numeric_input_width': 144, 'categorical_input_width': 23, 'numeric_inputs_pure_X_original': True,
        'cardinalities': revised['cardinalities'].tolist(),
        'prior_uses_fit_prices': True, 'uses_query_labels': False, 'chronological_oof': False,
        'legacy_state_group_rule_string_is_not_transaction_fold_evidence': True,
        'private_learned_states_and_per_row_factors_not_for_GitHub': True,
        'prior_prepare_seconds': time.monotonic() - tick}
    c.write(folder / 'preparation.json', preparation)
    baseline.verify_context(loaded, origin)
    return revised, preparation


def ratio_total(ratio, native, factor, rows):
    c.need(type(rows) is int and rows > 0 and ratio.dtype == native.dtype == np.float64
           and ratio.shape == native.shape == (rows,) and factor.dtype == np.float32 and factor.shape == (rows,)
           and np.isfinite(ratio).all() and (ratio > 0).all() and np.isfinite(native).all() and (native > 0).all()
           and np.isfinite(factor).all() and (factor > 0).all(), 'Positive unclipped ratio/native contracts')
    c.exact(ratio * factor.astype(np.float64) * 1e6, native, 'Exactly one original ratio-to-TWD inverse')


def check_metadata(meta, r, origin, prep):
    nf, nq = COUNTS[origin]; recipe = r['recipe']; identities = prep['array_identities']['revised']
    c.need(meta['status'] == 'completed' and meta['source_sha256'] == r['source_sha256'][TRAINER]
           and meta['input_sha256'] == prep['prepared']['sha256'] and meta['input_arrays'] == identities
           and meta['configuration'] == recipe['configuration']
           and meta['provenance_configuration_overrides'] == recipe['provenance_config_overrides']
           and meta['changed_configuration_fields'] == ['train_price_factor_construction']
           and meta['added_configuration_fields'] == [] and meta['ML_configuration_changes'] == []
           and meta['loss_configuration_changes'] == []
           and meta['actual_model_class'] == 'round47_nn_train.EmbeddingMLP'
           and meta['fit_rows'] == nf and meta['query_rows'] == nq and meta['numeric_count'] == 144
           and meta['cardinalities'] == prep['cardinalities'] and len(meta['cardinalities']) == 23
           and meta['saved_model_reload_exact'] is True
           and meta['saved_ratio_reload_exact'] is True and meta['reload_rows'] == nq
           and meta['query_labels_available'] is False and meta['query_labels_used'] is False
           and meta['price_or_ratio_clipping'] is False and meta['extra_sample_weights'] is True
           and meta['sample_weights_recomputed_in_trainer'] is False and meta['sample_weights_batch_renormalized'] is False
           and meta['fit_prior_exclusion_policy'] == 'transaction-permutation-5-fold'
           and meta['transaction_fold_provenance_verified_in_trainer'] is False
           and meta['price_factor_provenance_verified_in_trainer'] is False
           and meta['initial_bias_formula_unchanged_from_R63'] is True
           and meta['training_objective_name'] == 'weighted MAE' and meta['epoch_selection_uses_training_metrics'] is False
           and meta['automatic_retry_used'] is False and meta['automatic_fallback_used'] is False and meta['new_CSV'] is False
           and meta['train_price_factor_construction'] == recipe['configuration']['train_price_factor_construction']
           and meta['query_price_factor_construction'] == recipe['configuration']['query_price_factor_construction']
           and [v['epoch'] for v in meta['epochs']] == list(range(1, 31)), 'Actual fixed ratio NN contract differs')
    init = prep['transaction_initialization']
    c.need(meta['initial_training_median_ratio'] == init['initial_training_median_ratio']
           and meta['initial_head_bias'] == init['initial_head_bias_formula_float64']
           and meta['initial_head_bias_float32'] == init['initial_head_bias_float32'], 'Linked transaction fit initialization')
    parent_names = (TRAINER, 'src/round63_recency_weighted_prior_train.py', 'src/round47_nn_train.py', 'src/round52_nn_train.py')
    c.need(meta['complete_source_sha256'] == {k: r['source_sha256'][k] for k in parent_names}
           and meta['parent_trainer_source_sha256'] == r['source_sha256'][parent_names[1]]
           and meta['architecture_source_sha256'] == r['source_sha256'][parent_names[2]], 'Exact trainer parent lineage')
    for key in ('train_sample_weight', CHANGED_ARRAY, 'query_price_factor_million'):
        c.need(meta[key + '_sha256'] == identities[key]['values_sha256'], 'Trainer factor/dateweight identity: ' + key)
    c.need(all(np.isfinite(e[k]) for e in meta['epochs'] for k in
               ('training_mae_million_with_dropout', 'training_weighted_mae_million_with_dropout')), 'Finite epoch metrics')


def model_records(folder):
    files = sorted(folder.iterdir())
    c.need(files and all(p.is_file() and p.suffix in {'.json', '.npy', '.pt'} for p in files)
           and {'model.pt', 'run.json', 'training.json', 'ratio_predictions.npy', 'predictions_twd.npy'} <= {p.name for p in files}
           and 'failed.json' not in {p.name for p in files},
           'Bind every ratio NN checkpoint/metadata/prediction artifact')
    return [c.record(path) for path in files]


def validate(origin, record, r):
    folder = c.safe(BASE + '/' + origin); model = folder / 'model'; nf, nq = COUNTS[origin]
    c.need(record == c.record(folder / 'stage.json'), 'Stage exact identity')
    item = c.read(record['path']); preparation = c.read(folder / 'prior/preparation.json')
    c.need(item['stage'] == origin and item['candidate_round'] == 117 and item['fit_rows'] == nf and item['query_rows'] == nq
           and item['query_labels_available'] is False and item['model_metadata'] == c.record(model / 'training.json')
           and item['prior_preparation'] == c.record(folder / 'prior/preparation.json')
           and item['sealed_models'] == model_records(model) and item['sealed_prior_artifacts'] == prior_records(folder / 'prior')
           and item['train_log'] == c.record(folder / 'train.log') and item['context_before'] == item['context_after'],
           'Stage scope/models/prior/state seal differs')
    c.need(preparation['origin'] == origin and preparation['fit_rows'] == nf and preparation['query_rows'] == nq
           and preparation['prepared'] == c.record(folder / 'prior/prepared.npz')
           and preparation['source_prepared'] == r['origins'][origin]['prepared']
           and preparation['original_prior'] == r['origins'][origin]['original_prior']
           and preparation['original_training_metadata'] == r['origins'][origin]['original_training']
           and preparation['original_initialization_matches_saved_training'] is True, 'Prepared origin/source identity')
    c.need(preparation['array_identities']['original'] == item['original_prepared_array_identities']
           and preparation['array_identities']['revised'] == item['prepared_array_identities'], 'Original/revised nine array seals')
    with np.load(folder / 'prior/prepared.npz', allow_pickle=False) as z:
        c.need(len(z.files) == 9 and set(z.files) == set(c.ARRAY_NAMES), 'Exactly nine sealed prepared keys before reads')
        arrays = {k: z[k].copy() for k in c.ARRAY_NAMES}
    c.need({k: c.identity(v) for k, v in arrays.items()} == item['prepared_array_identities'], 'Sealed prepared bytes')
    c.need(preparation['cardinalities'] == arrays['cardinalities'].tolist(), 'Exact original categorical vocabulary sizes')
    for key in UNCHANGED_ARRAYS:
        c.need(item['prepared_array_identities'][key] == item['original_prepared_array_identities'][key], 'Eight arrays immutable: ' + key)
    c.need(c.sha(folder / 'prior/price_prior_state.json') == r['origins'][origin]['original_prior']['state']['sha256'], 'Original full-fit state bytes')
    c.need(preparation['full_fit_state_and_query_prior_and_factor_exact_original'] is True
           and preparation['fold_policy'] == 'transaction-permutation-5-fold'
           and preparation['uses_query_labels'] is False, 'Transaction preparation scope')
    fold = c.load_array(preparation['prior_artifacts']['prior_fold_ids']); c.exact(fold, fixed_folds(nf), 'Sealed actual transaction folds')
    c.need(preparation['fold_assignment'] == c.identity(fold)
           and preparation['fold_rows'] == [int((fold == k).sum()) for k in range(5)], 'Sealed actual balanced fold counts')
    original_factor = c.load_array(preparation['prior_artifacts']['original_train_factor_million'])
    new_factor = c.load_array(preparation['prior_artifacts']['transaction_train_factor_million'])
    c.need(c.identity(original_factor) == item['original_prepared_array_identities'][CHANGED_ARRAY], 'Sealed original fit factor')
    c.exact(new_factor, arrays[CHANGED_ARRAY], 'Sealed transaction factor/archive identity')
    old_init = ratio_bias(arrays['train_y_million'], original_factor)
    new_init = ratio_bias(arrays['train_y_million'], new_factor)
    c.need(preparation['original_initialization'] == item['original_initialization'] == old_init
           and preparation['transaction_initialization'] == item['transaction_initialization'] == new_init,
           'Original/new initialization independently recomputed from sealed factors')
    query_prior = c.load_array(preparation['prior_artifacts']['query_unit_prior'])
    c.exact(query_prior, np.load(c.safe(r['origins'][origin]['original_prior']['query_prior']['path']), allow_pickle=False),
            'Sealed full-fit query prior remains original')
    c.need(preparation['full_fit_state'] == c.record(folder / 'prior/price_prior_state.json')
           and preparation['prior_evidence'] == c.record(folder / 'prior/prior_evidence.json'), 'Private state/evidence exact records')
    meta = c.read(model / 'training.json'); check_metadata(meta, r, origin, preparation)
    run = c.read(model / 'run.json')
    c.need(all(meta.get(k) == v for k, v in run.items() if k != 'status'), 'Trainer before/after immutable metadata')
    c.need(c.sha(model / 'model.pt') == meta['model_sha256'], 'Saved checkpoint identity')
    ratio, native = [np.load(model / n, allow_pickle=False) for n in ('ratio_predictions.npy', 'predictions_twd.npy')]
    ratio_total(ratio, native, arrays['query_price_factor_million'], nq)
    c.need(set(meta['prediction_artifacts']) == {'ratio', 'TWD'}, 'Exactly ratio and TWD metadata records')
    for key, name, values in (('ratio', 'ratio_predictions.npy', ratio), ('TWD', 'predictions_twd.npy', native)):
        c.need(meta['prediction_artifacts'][key] == {**c.record(model / name), **c.identity(values)}, 'Prediction metadata identity')
    layers = {k: c.load_array(v) for k, v in item['layers'].items()}
    c.need(set(layers) == set(LAYERS), 'Exactly five layers')
    for key, value in layers.items(): metric.prices(value, (nq,), key + ' complete layer')
    c.exact(layers['native'], native, 'Model native/five-layer alignment')
    return item, layers


def stage(origin):
    r = request(); started = c.read(EVIDENCE + '/training_started.json')
    c.need(started['request'] == c.record(REQUEST) and started['new_fits_authorized'] == 4
           and started['new_prior_builds_authorized'] == 4, 'Coordinated serial prior/NN marker required')
    loaded = load_context(r, origin); nf, nq = COUNTS[origin]
    loaded['source_prepared_path'] = r['origins'][origin]['prepared']['path']
    folder = c.safe(BASE + '/' + origin); folder.mkdir(exist_ok=False)
    arrays, prep = prepare(loaded, origin, folder / 'prior')
    command = [str(c.ROOT / '.venv_torch/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(c.safe(TRAINER)),
               '--data', str(folder / 'prior/prepared.npz'), '--out', str(folder / 'model')]
    with (folder / 'train.log').open('x', encoding='utf8') as log:
        process = subprocess.run(command, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    c.need(process.returncode == 0, 'Trainer failed; preserve evidence, no retry')
    meta = c.read(folder / 'model/training.json'); check_metadata(meta, r, origin, prep)
    ratio, native = [np.load(folder / 'model' / n, allow_pickle=False) for n in ('ratio_predictions.npy', 'predictions_twd.npy')]
    ratio_total(ratio, native, arrays['query_price_factor_million'], nq)
    ctx = loaded['context']; chain = offset_comp.chain_prior(ctx, native) if origin == OFFSET else comp.apply(ctx, native)[0]
    c.need(set(chain) == set(LAYERS), 'One prior-slot replacement, complete composition')
    records = {k: c.save_array(folder / (k + '.npy'), v) for k, v in chain.items()}
    baseline.verify_context(loaded, origin)
    c.write(folder / 'stage.json', {'stage': origin, 'candidate_round': 117, 'fit_rows': nf, 'query_rows': nq,
        'model_metadata': c.record(folder / 'model/training.json'), 'sealed_models': model_records(folder / 'model'),
        'prior_preparation': c.record(folder / 'prior/preparation.json'), 'sealed_prior_artifacts': prior_records(folder / 'prior'),
        'layers': records, 'prepared_array_identities': prep['array_identities']['revised'],
        'original_prepared_array_identities': loaded['array_identities'], 'train_log': c.record(folder / 'train.log'),
        'fit_seconds': meta['fit_seconds'], 'prior_prepare_seconds': prep['prior_prepare_seconds'],
        'original_initialization': prep['original_initialization'], 'transaction_initialization': prep['transaction_initialization'],
        'context_before': loaded['before'], 'context_after': baseline.context_identity(ctx, origin),
        'original_v22_v24_and_prior_slot_replayed_exact': True,
        'query_labels_available': False, 'query_scores_computed': False})
    validate(origin, c.record(folder / 'stage.json'), r); request()
    print(json.dumps({'round': 117, 'origin': origin, 'status': 'prior_model_and_layers_saved_sealed_unscored'}), flush=True)


def train():
    r = request(); tick = time.monotonic(); failure = None
    c.need(not c.safe(EVIDENCE + '/training_started.json').exists()
           and not c.safe(SEAL).exists() and not c.safe(EVIDENCE + '/training_execution.json').exists(), 'One serial attempt only')
    c.write(EVIDENCE + '/training_started.json', {'request': c.record(REQUEST), 'created_at_utc': c.now(),
            'new_fits_authorized': 4, 'new_prior_builds_authorized': 4, 'automatic_retry': False})
    try:
        for origin in ORIGINS:
            command = [str(c.ROOT / '.venv/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--stage', origin]
            process = subprocess.run(command, cwd=c.ROOT, check=False)
            c.need(process.returncode == 0, 'Origin failed, no automatic retry: ' + origin)
        records = {s: c.record(BASE + '/' + s + '/stage.json') for s in ORIGINS}
        for origin, rec in records.items(): validate(origin, rec, r)
        request()
        c.write(SEAL, {'status': 'four_priors_models_and_layers_sealed_unscored', 'request': c.record(REQUEST),
            'origin_order': list(ORIGINS), 'records': records, 'new_fits': 4, 'new_prior_builds': 4,
            'query_scores_computed': False, 'elapsed_seconds': time.monotonic() - tick})
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(EVIDENCE + '/training_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
            'request': c.record(REQUEST), 'elapsed_seconds': time.monotonic() - tick, 'automatic_retry': False})


def combined_gates(stages):
    c.need(set(stages) == set(ORIGINS), 'All four fixed origins required')
    august = history.gates({s: stages[s] for s in c.STAGES}); extra = offset.gates(stages[OFFSET]['metrics'], stages[OFFSET]['comparison'])
    result = {'august': august, 'extra_origin': extra}
    for name, old in (('combined_exploratory_submission', august['exploratory_submission']['passed']),
                      ('combined_promotion', august['promotion']['adoption_research_gate_passed'])):
        parts = {'original_three_windows': old, 'fixed_extra_origin': extra['passed']}
        result[name] = {'components': parts, 'passed': all(parts.values())}
    result.update(private_score=None, selected_submission='v22', automatic_selection=False,
                  new_CSV=False, full_data_fit=False, reserved_exploratory_version=32)
    return result


def scoring_preflight():
    r = request(); seal = c.read(SEAL); execution = c.read(EVIDENCE + '/training_execution.json')
    c.need(seal['status'] == 'four_priors_models_and_layers_sealed_unscored' and seal['request'] == c.record(REQUEST)
           and seal['origin_order'] == list(ORIGINS) and set(seal['records']) == set(ORIGINS)
           and seal['new_fits'] == seal['new_prior_builds'] == 4 and seal['query_scores_computed'] is False,
           'All four priors/models/layers must be sealed before any query price access')
    c.need(execution['exit_code'] == 0 and execution['failure'] is None and execution['request'] == c.record(REQUEST),
           'Serial training must exit successfully before separate scoring')
    checked = {s: validate(s, seal['records'][s], r) for s in ORIGINS}
    c.need(r['input_sha256'][c.RAW] == RAW_SHA and c.sha(c.RAW) == RAW_SHA, 'Scoring-only label source changed')
    return r, seal, checked


def evaluate():
    c.need(not c.safe(RESULTS).exists() and not c.safe(EVIDENCE + '/evaluation_execution.json').exists(), 'Preserve any previous scoring attempt')
    r, seal, checked = scoring_preflight(); tick = time.monotonic(); failure = None
    try:
        # First raw-price deserialization, after all four native/prior/layer/exit guards.
        raw = pd.read_pickle(c.safe(c.RAW)); stages = {}
        for origin in ORIGINS:
            loaded = load_context(r, origin); item, layers = checked[origin]
            if origin == OFFSET:
                y = offset_metric.select_query_y(raw, loaded['indices']['query_source_cache_indices'], 455784)
                descriptive = lambda v: offset_metric.descriptive_metrics(y, v, loaded['fit_X'], loaded['query_X'])
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
                    c.need(all(metrics[key][k] == old[k] for k in old), 'Original August control replay: ' + key)
                comparison, comparison24 = metric.comparisons(metrics), metric.comparisons(metrics, 'v24')
                paired, samples = metric.paired(y, {'v22': loaded['v22'], 'T': layers['median']},
                    loaded['fit_X'], loaded['query_X'], loaded['indices']['query_indices'], origin, 'v22')
            comparison.update(control='v22', candidate='R117', used_for_release_gate=True)
            comparison24.update(control='v24', candidate='R117', used_for_release_gate=False)
            paired.update(candidate='R117', difference='R117 minus v22', caution=CAUTION)
            folder = c.safe(BASE + '/' + origin + '/evaluation'); folder.mkdir(exist_ok=False)
            sample_records = {k: c.save_array(folder / (k + '_samples.npy'), v) for k, v in samples.items()}
            result = {'metrics': metrics, 'comparison': comparison, 'comparison_vs_v24': comparison24,
                'paired': paired, 'paired_samples': sample_records, 'sealed_stage': seal['records'][origin],
                'branch_metrics': {k: descriptive(v) for k, v in layers.items() if k != 'median'},
                'both_point_metrics_improve': all(metrics['T'][k] < metrics['v22'][k] for k in ('overall_mae', 'weighted_mae'))}
            baseline.verify_context(loaded, origin); c.write(folder / 'results.json', result); stages[origin] = result
            print(json.dumps({'round': 117, 'origin': origin, 'mae': metrics['T']['overall_mae'],
                             'delta': metrics['T']['overall_mae'] - metrics['v22']['overall_mae']}), flush=True)
        del raw; scoring_preflight()
        c.write(RESULTS, {'status': 'four_reused_fixed_origins_evaluated', 'created_at_utc': c.now(), 'round': 117,
            'request': c.record(REQUEST), 'seal': c.record(SEAL), 'plan': c.record(PLAN), 'source_sha256': sources(),
            'origin_order': list(ORIGINS), 'stages': stages, 'gates': combined_gates(stages), 'new_fits': 4,
            'new_prior_builds': 4, 'new_CSV': False, 'full_data_fit': False, 'Public_score': None, 'Private_score': None,
            'selected_submission': 'v22', 'fresh_blind_validation': False, 'caution': CAUTION})
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(EVIDENCE + '/evaluation_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
            'elapsed_seconds': time.monotonic() - tick, 'request': c.record(REQUEST), 'seal': c.record(SEAL),
            'training_execution': c.record(EVIDENCE + '/training_execution.json'),
            'query_labels_used_for_scoring_only': True, 'training_already_finished_before_scoring': True,
            'new_fits': 0, 'new_prior_builds': 0, 'new_CSV': False})


def self_check():
    """Synthetic formula/identity/guard checks; no actual prior build or array preparation."""
    from unittest.mock import patch
    checks = []
    def ok(name, value): c.need(value, name); checks.append(name)
    def rejects(name, call):
        try: call()
        except (ValueError, KeyError): checks.append(name)
        else: raise AssertionError('Guard accepted: ' + name)
    old_rng = np.random.get_state(); a = fixed_folds(13); b = fixed_folds(13); new_rng = np.random.get_state()
    ok('fixed_seed42_repeatable_balanced_folds', a.tobytes() == b.tobytes() and sorted(np.bincount(a).tolist()) == [2, 2, 3, 3, 3])
    ok('local_default_rng_does_not_touch_global_RNG', all(np.array_equal(x, y) for x, y in zip(old_rng, new_rng)))
    rejects('fewer_than_five_rows_rejected', lambda: fixed_folds(4))
    rejects('boolean_row_count_rejected', lambda: fixed_folds(True))
    y = np.array([2., 4., 6.], np.float32); f = np.ones(3, np.float32)
    init = ratio_bias(y, f)
    ok('inverseSoftplus_median_hand_calculation', init['initial_training_median_ratio'] == 4.
       and init['initial_head_bias_formula_float64'] == 4. + float(np.log(-np.expm1(-4.))))
    ok('fit_factor_linked_initialization_changes', ratio_bias(y, f * 2)['initial_training_median_ratio'] == 2.)
    for key, bad in (('zero', np.zeros(3, np.float32)), ('nan', np.full(3, np.nan, np.float32)), ('dtype', f.astype(np.float64))):
        rejects('bad_ratio_factor_' + key, lambda bad=bad: ratio_bias(y, bad))
    arrays = {'train_num': np.zeros((3, 144), np.float32), 'query_num': np.zeros((3, 144), np.float32),
              'train_cat': np.zeros((3, 23), np.int64), 'query_cat': np.zeros((3, 23), np.int64),
              'cardinalities': np.full(23, 2, np.int64), 'train_y_million': y,
              'train_price_factor_million': f, 'query_price_factor_million': f.copy(), 'train_sample_weight': f.copy()}
    revised = {k: v.copy() for k, v in arrays.items()}; revised[CHANGED_ARRAY] *= 2
    ok('only_fit_factor_may_change', len(check_changed_arrays(arrays, revised)['unchanged_eight']) == 8)
    for key in UNCHANGED_ARRAYS:
        corrupt = {k: v.copy() for k, v in revised.items()}; corrupt[key].flat[0] += 1
        rejects('changed_immutable_array_' + key, lambda corrupt=corrupt: check_changed_arrays(arrays, corrupt))
    rejects('query_y_key_rejected', lambda: check_changed_arrays(arrays, {**revised, 'query_y': y}))
    rejects('duplicate_missing_key_rejected', lambda: check_changed_arrays(arrays, {k: v for k, v in revised.items() if k != 'train_cat'}))
    ratio = np.array([1., 2., 3.], np.float64); native = ratio * f.astype(np.float64) * 1e6
    ratio_total(ratio, native, f, 3); ok('one_ratio_to_TWD_inverse', True)
    rejects('extra_area_or_cap_rejected', lambda: ratio_total(ratio, native * 2, f, 3))
    rejects('zero_native_rejected', lambda: ratio_total(ratio, np.zeros(3, np.float64), f, 3))
    rejects('ratio_float32_rejected', lambda: ratio_total(ratio.astype(np.float32), native, f, 3))
    # Verify the actual fold evidence guard without constructing any estimator.
    import copy
    fit_y = np.arange(1., 14., dtype=np.float64); ids = np.arange(13, dtype=np.int64)
    folds = fixed_folds(13); artificial_oof = np.ones(13, np.float64); state = {'artificial': 'full-state'}
    loaded = {'fit_y': fit_y, 'fit_X': object(), 'source_indices': ids}
    ev = {'uses_fit_labels': True, 'uses_query_labels': False, 'chronological_oof': False, 'fit_rows': 13,
          'fold_policy': 'transaction-permutation-5-fold', 'fold_assignment': prior.FOLD_RULE,
          'fold_assignment_uses_labels_dates_groups_or_query': False, 'fold_assignment_identical_to_R52_rule': False,
          'legacy_full_fit_state_unmodified': True, 'date_weights_rebuilt_in_each_complement_by_original_R60': True,
          'R115_full_outer_dateweights_used_for_prior_median': False, 'state_content_sha256': prior.state_content_sha(state),
          'source_sha256': prior.source_hashes(), 'fit_target_sha256': c.values_sha(fit_y), 'source_indices_sha256': c.values_sha(ids),
          'fold_ids_sha256': c.values_sha(folds), 'oof_prior_sha256': c.values_sha(artificial_oof),
          'effective_area_sha256': c.values_sha(np.ones(13, np.float64)), 'fit_unit_sha256': c.values_sha(fit_y), 'folds': []}
    for k in range(5):
        mask = folds != k
        ev['folds'].append({'fold': k, 'held_rows': int((~mask).sum()), 'query_rows': int((~mask).sum()),
            'fit_rows': int(mask.sum()), 'reference_rows': int(mask.sum()), 'same_groups_may_cross_transaction_folds': True,
            'reference_indices_sha256': c.values_sha(ids[mask]), 'reference_target_sha256': c.values_sha(fit_y[mask]),
            'reference_unit_sha256': c.values_sha(fit_y[mask])})
    with patch.object(prior, 'effective_area', return_value=np.ones(13, np.float64)):
        check_prior_evidence(ev, loaded, state, state.copy(), folds, artificial_oof); ok('synthetic_transaction_fold_identity_guard', True)
        for key, bad in (('fold_policy', 'group-hash'), ('uses_query_labels', True), ('chronological_oof', True),
                         ('fold_assignment_uses_labels_dates_groups_or_query', True), ('fold_assignment_identical_to_R52_rule', True),
                         ('date_weights_rebuilt_in_each_complement_by_original_R60', False), ('R115_full_outer_dateweights_used_for_prior_median', True),
                         ('fit_target_sha256', 'wrong'), ('state_content_sha256', 'wrong'), ('source_indices_sha256', 'wrong')):
            corrupt = copy.deepcopy(ev); corrupt[key] = bad
            rejects('prior_evidence_' + key, lambda corrupt=corrupt: check_prior_evidence(corrupt, loaded, state, state, folds, artificial_oof))
        corrupt = copy.deepcopy(ev); corrupt['folds'][0]['reference_target_sha256'] = 'wrong'
        rejects('prior_fold_reference_label_identity', lambda: check_prior_evidence(corrupt, loaded, state, state, folds, artificial_oof))
        corrupt = copy.deepcopy(ev); corrupt['folds'] = corrupt['folds'][:-1]
        rejects('prior_missing_complete_fold', lambda: check_prior_evidence(corrupt, loaded, state, state, folds, artificial_oof))
        rejects('changed_full_fit_state_rejected', lambda: check_prior_evidence(ev, loaded, {'changed': True}, state, folds, artificial_oof))
    # Actual metadata names are checked against artificial identities and no ML imports.
    init = ratio_bias(y, revised[CHANGED_ARRAY]); identities = {k: c.identity(v) for k, v in revised.items()}
    parent_names = (TRAINER, 'src/round63_recency_weighted_prior_train.py', 'src/round47_nn_train.py', 'src/round52_nn_train.py')
    lineage = {k: c.sha(k) for k in parent_names}
    configuration = {'train_price_factor_construction': 'artificial transaction provenance',
                     'query_price_factor_construction': 'artificial full-fit provenance'}
    overrides = {'train_price_factor_construction': configuration['train_price_factor_construction']}
    recipe = {'configuration': configuration, 'provenance_config_overrides': overrides}
    r = {'source_sha256': lineage, 'recipe': recipe}
    prep = {'prepared': {'sha256': 'artificial-npz'}, 'array_identities': {'revised': identities}, 'transaction_initialization': init,
            'cardinalities': [2] * 23}
    meta = {'status': 'completed', 'source_sha256': lineage[TRAINER], 'input_sha256': 'artificial-npz', 'input_arrays': identities,
        'configuration': configuration, 'provenance_configuration_overrides': overrides,
        'changed_configuration_fields': ['train_price_factor_construction'], 'added_configuration_fields': [],
        'ML_configuration_changes': [], 'loss_configuration_changes': [], 'actual_model_class': 'round47_nn_train.EmbeddingMLP',
        'fit_rows': 3, 'query_rows': 3, 'numeric_count': 144, 'cardinalities': [2] * 23,
        'saved_model_reload_exact': True, 'saved_ratio_reload_exact': True, 'reload_rows': 3,
        'query_labels_available': False, 'query_labels_used': False, 'price_or_ratio_clipping': False, 'extra_sample_weights': True,
        'sample_weights_recomputed_in_trainer': False, 'sample_weights_batch_renormalized': False,
        'fit_prior_exclusion_policy': 'transaction-permutation-5-fold', 'transaction_fold_provenance_verified_in_trainer': False,
        'price_factor_provenance_verified_in_trainer': False, 'initial_bias_formula_unchanged_from_R63': True,
        'training_objective_name': 'weighted MAE', 'epoch_selection_uses_training_metrics': False,
        'automatic_retry_used': False, 'automatic_fallback_used': False, 'new_CSV': False,
        'train_price_factor_construction': configuration['train_price_factor_construction'],
        'query_price_factor_construction': configuration['query_price_factor_construction'],
        'initial_training_median_ratio': init['initial_training_median_ratio'], 'initial_head_bias': init['initial_head_bias_formula_float64'],
        'initial_head_bias_float32': init['initial_head_bias_float32'], 'complete_source_sha256': lineage,
        'parent_trainer_source_sha256': lineage[parent_names[1]], 'architecture_source_sha256': lineage[parent_names[2]],
        'epochs': [{'epoch': i, 'training_mae_million_with_dropout': 1., 'training_weighted_mae_million_with_dropout': 2.} for i in range(1, 31)]}
    meta.update({key + '_sha256': identities[key]['values_sha256'] for key in ('train_sample_weight', CHANGED_ARRAY, 'query_price_factor_million')})
    with patch.dict(globals(), {'COUNTS': {'development': (3, 3)}}):
        check_metadata(meta, r, 'development', prep); ok('actual_ratio_trainer_metadata_schema_guard', True)
        for key, bad in (('actual_model_class', 'changed'), ('saved_ratio_reload_exact', False), ('query_labels_used', True),
                         ('price_or_ratio_clipping', True), ('sample_weights_batch_renormalized', True),
                         ('fit_prior_exclusion_policy', 'group-fold'), ('ML_configuration_changes', ['loss']),
                         ('initial_head_bias_float32', init['initial_head_bias_float32'] + 1), ('input_arrays', {}),
                         ('query_price_factor_million_sha256', 'changed'), ('complete_source_sha256', {})):
            corrupt = copy.deepcopy(meta); corrupt[key] = bad
            rejects('trainer_metadata_' + key, lambda corrupt=corrupt: check_metadata(corrupt, r, 'development', prep))
    with patch.object(support, 'specification', return_value={'blockers': ['missing audit']}):
        rejects('blocked_R115_prerequisite_rejected', lambda: check_r115({}))
    with patch.object(support, 'specification', return_value={'blockers': [], 'source_sha256': {}, 'plan': {'sha256': 'p'}}), \
         patch.object(c, 'read', return_value={'status': 'not-passed'}):
        rejects('nonpassed_R115_postfit_rejected', lambda: check_r115({}))
    ok('four_fixed_origins_no_shared_split_mutation', ORIGINS == ('development', 'confirmation', 'auxiliary', OFFSET)
       and tuple(c.SPLITS) == c.STAGES and COUNTS[OFFSET] == (272117, 107169))
    for aug in (False, True):
        for extra in (False, True):
            for strong in (False, True):
                data = {s: {'metrics': {}, 'comparison': {}} for s in ORIGINS}
                with patch.object(history, 'gates', return_value={'exploratory_submission': {'passed': aug}, 'promotion': {'adoption_research_gate_passed': strong}}), \
                     patch.object(offset, 'gates', return_value={'passed': extra}): result = combined_gates(data)
                ok('combined_gate_boolean_matrix', result['combined_exploratory_submission']['passed'] == (aug and extra)
                   and result['combined_promotion']['passed'] == (strong and extra) and result['reserved_exploratory_version'] == 32)
    rejects('missing_fourth_origin_rejected', lambda: combined_gates({s: {} for s in c.STAGES}))
    sealed = {'status': 'four_priors_models_and_layers_sealed_unscored', 'request': {}, 'origin_order': list(ORIGINS),
              'records': {s: {} for s in ORIGINS}, 'new_fits': 4, 'new_prior_builds': 4, 'query_scores_computed': False}
    execution = {'exit_code': 0, 'failure': None, 'request': {}}
    with patch.dict(globals(), {'request': lambda: {'input_sha256': {c.RAW: RAW_SHA}}, 'validate': lambda *args: ({}, {})}), \
         patch.object(c, 'read', side_effect=lambda name: sealed if name == SEAL else execution), \
         patch.object(c, 'record', return_value={}), patch.object(c, 'sha', return_value=RAW_SHA):
        ok('all_four_models_and_priors_validated_before_label_access', len(scoring_preflight()[2]) == 4)
        execution['exit_code'] = 1; rejects('failed_training_exit_refuses_scoring', scoring_preflight); execution['exit_code'] = 0
        sealed['records'].pop(OFFSET); rejects('missing_offset_seal_refuses_scoring', scoring_preflight)
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks), 'source_sha256': c.source_hashes(__file__),
            'complete_research_source_sha256': sources(), 'real_data_read': False, 'query_labels_read': False,
            'new_fits': 0, 'new_prior_builds': 0, 'GPU_calls': 0, 'new_CSV': False, 'global_SPLITS_modified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--train', action='store_true')
    group.add_argument('--stage', choices=ORIGINS); group.add_argument('--evaluate', action='store_true')
    group.add_argument('--self-check', action='store_true'); args = parser.parse_args()
    if args.freeze: freeze()
    elif args.train: train()
    elif args.stage: stage(args.stage)
    elif args.evaluate: evaluate()
    else: print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
