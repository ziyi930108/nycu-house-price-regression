"""One preregistered R107 diagnostic refit; seal 30 snapshots before query y.

Public entry points: --freeze, --train, --evaluate. Internal Torch workers only
inspect already-trained checkpoints or score already-sealed checkpoints.
--self-check uses source text and artificial arrays, never true data or CUDA.
"""
from __future__ import annotations
import argparse
from datetime import date
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import round100_contract as c

ROOT = c.ROOT
BASE = 'outputs/v25_training_curve_20261010'
AUTHOR = 'outputs/v25_training_curve_author_20261010'
PLAN = 'reports/v25_training_curve_plan_20261010.md'
PLAN_SHA = '9fcafe852041782da3d274e91e2a9b6a34899824f5e477f5890a9ab3da5ac2be'
ORIGINAL = 'src/round107_weighted_smooth_train.py'
ORIGINAL_SHA = 'f2330726165b1a102c893211268734c89e981b751f45167b82ae1c940917b36e'
TRAINER = 'src/round107_training_curve_train_20261010.py'
TRAINER_SHA = 'da03e0bbe5fa22789da6345b0bdd7f80a57e256a40ea113438c62bcd93a240fd'
OLD_REQUEST = 'outputs/research_resume_20261008/request.json'
OLD_REQUEST_SHA = '1a0eb13f48c22f026be5551c8ecfd1810ca6299c24529282b3d9f937d433e321'
OLD_TRAINING = 'outputs/round107/confirmation/model/training.json'
OLD_TRAINING_SHA = '24fff85c1140e4abba01dd2c22c31163eb66400cf010a8e29f48c3121cdacf3b'
PREPARED = 'outputs/round63/confirmation/prepared.npz'
PREPARED_SHA = '4d4b7866f4f94068e60585b6351fd1a669be184d9e1f8d14cc4a5eac652a0aff'
PREPARATION = 'outputs/round63/confirmation/preparation.json'
PREPARATION_SHA = '33dc5c22da3ed66879ee738aedd5b8d1a6a645f5f1a9c3dd184ee2a8e7d530de'
PRIOR_PREPARATION = 'outputs/round60/confirmation/prior_preparation.json'
PRIOR_PREPARATION_SHA = '9210b247b219fde6e9a70c20d4fd6c0b656f76cc6782bc266e76150af4c32d20'
RAW_SHA = '9acf8c65bf00f7067e9ea9612cabe9a1524b020a7596c8da3ce3d620131e5c00'
STAGE = 'confirmation'
FIT_ROWS, QUERY_ROWS = 321550, 133697
START, END = '2023-08-11', '2024-08-10'
EPOCHS = list(range(1, 31))
INDEX_KEYS = ('fit_indices', 'query_indices', 'fit_source_cache_indices', 'query_source_cache_indices')
SNAPSHOT_BLOCK = """        checkpoint_dir=out/'epoch_checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path=checkpoint_dir/f'epoch_{epoch:02d}.pt'
        with checkpoint_path.open('xb') as checkpoint_file:
            torch.save({'numeric_count':meta['numeric_count'],'cardinalities':card,'configuration':CONFIG,
                        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},checkpoint_file)
"""


def source_difference():
    c.need(c.sha(PLAN) == PLAN_SHA, 'Diagnostic plan changed')
    c.need(c.sha(ORIGINAL) == ORIGINAL_SHA and c.sha(TRAINER) == TRAINER_SHA, 'Trainer source changed')
    original = c.safe(ORIGINAL).read_text(encoding='utf8')
    diagnostic = c.safe(TRAINER).read_text(encoding='utf8')
    needle = '        schedule.step()\n'
    c.need(original.count(needle) == 1 and diagnostic == original.replace(needle, needle + SNAPSHOT_BLOCK),
           'Only the six-line epoch checkpoint insertion is allowed')
    return ''.join(difflib.unified_diff(original.splitlines(keepends=True), diagnostic.splitlines(keepends=True),
        fromfile=ORIGINAL, tofile=TRAINER))


def protected():
    names = [ORIGINAL, OLD_TRAINING, 'outputs/submission_v22_trial.csv', 'outputs/submission_v25_trial.csv',
             'outputs/v25_submission_focus_20261010/user_selection.json',
             'outputs/kaggle_feedback/public_v25_user_report_20261010.json']
    return {name: c.sha(name) for name in names}


def frozen_specification():
    """Metadata and hashes only: no numerical arrays, pickle or model loads."""
    source_difference()
    pinned = {OLD_REQUEST: OLD_REQUEST_SHA, OLD_TRAINING: OLD_TRAINING_SHA,
              PREPARED: PREPARED_SHA, PREPARATION: PREPARATION_SHA,
              PRIOR_PREPARATION: PRIOR_PREPARATION_SHA, c.RAW: RAW_SHA}
    c.verify_hashes(pinned)
    old, training = c.read(OLD_REQUEST), c.read(OLD_TRAINING)
    prep, prior_prep = c.read(PREPARATION), c.read(PRIOR_PREPARATION)
    spec = old['stages'][STAGE]
    c.need(old['status'] == 'frozen_before_six_fits' and spec['composition']['stage'] == STAGE,
           'Original R107 confirmation authority differs')
    c.need(spec['prepared'] == {'path': PREPARED, 'sha256': PREPARED_SHA}, 'Original nine-array archive differs')
    stage_rec = spec['composition']['source_stage_seal']
    c.need(spec['composition']['input_sha256'][stage_rec['path']] == stage_rec['sha256'], 'Stage authority differs')
    c.need(c.record(stage_rec['path']) == stage_rec, 'Original stage seal changed')
    stage = c.read(stage_rec['path'])
    # R102 source_input_arrays belong to filtered R100T, NOT to the R107 archive.
    axes = {key: stage[key] for key in INDEX_KEYS}
    references = {key: stage[key] for key in ('features', 'canonical_mapping')}
    for rec in [stage_rec, *axes.values(), *references.values(), *spec['fit_labels'].values()]:
        c.need(old['input_sha256'].get(rec['path']) == rec['sha256'], 'Original request did not bind input')
        c.need(rec['path'] not in pinned or pinned[rec['path']] == rec['sha256'], 'Conflicting input SHA')
        pinned[rec['path']] = rec['sha256']
    c.need(training['status'] == 'completed' and training['source_sha256'] == ORIGINAL_SHA
           and training['input_sha256'] == PREPARED_SHA and training['fit_rows'] == FIT_ROWS
           and training['query_rows'] == QUERY_ROWS and training['numeric_count'] == 144
           and not training['query_labels_used'] and not training['query_labels_available']
           and training['saved_model_reload_exact'] and training['reload_rows'] == QUERY_ROWS,
           'Original R107 training provenance differs')
    config = training['configuration']
    c.need(config['epochs'] == 30 and config['seed'] == 42 and config['batch_size'] == 2048
           and config['smooth_l1_beta_million'] == .1
           and config['epoch_selection'] == 'fixed last epoch; no query labels or early stopping',
           'Fixed R107 recipe differs')
    c.need(prep['prepared_path'] == PREPARED and prep['prepared_sha256'] == PREPARED_SHA
           and prep['prior_preparation_path'] == PRIOR_PREPARATION
           and prep['prior_preparation_sha256'] == PRIOR_PREPARATION_SHA
           and prep['fit_rows'] == prior_prep['fit_rows'] == FIT_ROWS
           and prep['query_rows'] == prior_prep['query_rows'] == QUERY_ROWS
           and not prep['encoder_refit'] and not prep['prior_refit'] and not prep['query_labels_used']
           and not prior_prep['encoder_refit'] and not prior_prep['query_labels_used']
           and prior_prep['prior_evidence']['chronological_oof'] is False
           and prior_prep['prior_evidence']['fit_target_sha256'] == spec['fit_labels']['fit_y']['values_sha256'],
           'Original own-cut encoder/group-OOF prior metadata differs')
    prior_sources = prior_prep['prior_evidence']['source_sha256']
    for path, digest in prior_sources.items():
        c.need(path not in pinned or pinned[path] == digest, 'Conflicting prior source identity')
        pinned[path] = digest
    c.verify_hashes(pinned)
    return {'stage': STAGE, 'start': START, 'end': END, 'fit_rows': FIT_ROWS, 'query_rows': QUERY_ROWS,
            'prepared': spec['prepared'], 'axes': axes, **references, 'fit_labels': spec['fit_labels'],
            'original_training': {'path': OLD_TRAINING, 'sha256': OLD_TRAINING_SHA},
            'expected_training_configuration': config, 'input_sha256': pinned,
            'original_six_encoding_values_sha256': prior_prep['unchanged_six_array_sha256'],
            'factor_values_sha256': {k: training[k + '_sha256'] for k in
                ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight')},
            'expected_cardinalities': training['cardinalities']}


def freeze():
    spec = frozen_specification()
    sources = c.source_hashes(__file__, TRAINER, ORIGINAL)
    c.verify_hashes(sources)
    diff = source_difference()
    with c.safe(BASE + '/trainer_source.diff').open('x', encoding='utf8', newline='\n') as f:
        f.write(diff)
    c.write(BASE + '/request.json', {'status': 'frozen_before_one_diagnostic_fit', 'created_at_utc': c.now(),
        'plan': c.record(PLAN), 'source_sha256': sources, 'specification': spec,
        'source_difference': c.record(BASE + '/trainer_source.diff'), 'protected_before': protected(),
        'model_fits': 1, 'checkpoint_epochs': EPOCHS, 'query_labels_available_for_fitting': False,
        'query_scores_computed': False, 'new_CSV': False, 'selection_changed': False,
        'metadata_only_freeze': True, 'NPZ_features_prices_models_deserialized': False})


def request():
    r = c.read(BASE + '/request.json')
    c.need(r['status'] == 'frozen_before_one_diagnostic_fit' and r['checkpoint_epochs'] == EPOCHS,
           'Diagnostic request differs')
    c.need(r['plan'] == c.record(PLAN), 'Frozen plan changed')
    c.verify_hashes(r['source_sha256'])
    c.verify_hashes(r['specification']['input_sha256'])
    c.need(protected() == r['protected_before'], 'Protected original artifacts changed')
    c.need(source_difference() == c.safe(BASE + '/trainer_source.diff').read_text(encoding='utf8'),
           'Recorded trainer difference changed')
    c.need(c.record(BASE + '/trainer_source.diff') == r['source_difference'], 'Trainer diff SHA changed')
    return r


def index_contract(a, bound, count=None):
    c.need(isinstance(a, np.ndarray) and a.dtype == np.int64 and a.ndim == 1 and len(a) > 0
           and (a >= 0).all() and (a < bound).all() and (np.diff(a) > 0).all(), 'Index axis differs')
    c.need(count is None or len(a) == count, 'Index row count differs')


def temporal_contract(days, mapping, axes, start=START, end=END, fit_rows=FIT_ROWS, query_rows=QUERY_ROWS):
    index_contract(mapping, 455784)
    c.need(days.dtype == np.float64 and days.shape == mapping.shape, 'Canonical date axis differs')
    lo = (date.fromisoformat(start) - date(2000, 1, 1)).days
    hi = (date.fromisoformat(end) - date(2000, 1, 1)).days
    fi = np.flatnonzero(np.isfinite(days) & (days < lo)).astype(np.int64)
    qi = np.flatnonzero(np.isfinite(days) & (days >= lo) & (days <= hi)).astype(np.int64)
    index_contract(axes['fit_indices'], len(mapping), fit_rows)
    index_contract(axes['query_indices'], len(mapping), query_rows)
    c.exact(fi, axes['fit_indices'], 'Own-cut fit date mask differs')
    c.exact(qi, axes['query_indices'], 'Own-cut query date mask differs')
    c.need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
    c.exact(mapping[fi], axes['fit_source_cache_indices'], 'Fit canonical/raw axis differs')
    c.exact(mapping[qi], axes['query_source_cache_indices'], 'Query canonical/raw axis differs')
    return fi, qi


def load_axes(spec):
    """Pure-X dates and indices only; no raw price table or NPZ is opened here."""
    import pandas as pd
    axes = {key: c.load_array(rec) for key, rec in spec['axes'].items()}
    mapping = np.load(c.safe(spec['canonical_mapping']['path']), allow_pickle=False)
    index_contract(mapping, 455784, 455255)
    raw_x = pd.read_pickle(c.safe(spec['features']['path']))
    c.need(len(raw_x) == 455784 and raw_x.index.equals(pd.RangeIndex(455784))
           and len(raw_x.columns) == 87 and '總價元' not in raw_x, 'Pure-X frame schema differs')
    days = raw_x.iloc[mapping]['trade_day_index'].to_numpy(dtype=np.float64)
    fi, qi = temporal_contract(days, mapping, axes)
    Xfit = raw_x.iloc[mapping[fi]].reset_index(drop=True)
    del raw_x
    for key in ('fit_indices', 'fit_source_cache_indices'):
        c.exact(c.load_array(spec['fit_labels'][key]), axes[key], 'Saved fit-label axis differs')
    return axes, Xfit


def input_contract(arrays, y, spec):
    nf, nq = spec['fit_rows'], spec['query_rows']
    c.need(set(arrays) == set(c.ARRAY_NAMES), 'Exactly the original nine NPZ keys required')
    shapes = {'train_num': (nf, 144), 'query_num': (nq, 144), 'train_cat': (nf, 23),
              'query_cat': (nq, 23), 'train_y_million': (nf,), 'train_price_factor_million': (nf,),
              'query_price_factor_million': (nq,), 'train_sample_weight': (nf,), 'cardinalities': (23,)}
    for key, shape in shapes.items():
        a = arrays[key]
        dtype = np.int64 if key in ('train_cat', 'query_cat', 'cardinalities') else np.float32
        c.need(a.dtype == dtype and a.shape == shape and np.isfinite(a).all(), 'NPZ schema: ' + key)
    c.need(y.dtype == np.float64 and y.shape == (nf,) and np.isfinite(y).all() and (y > 0).all(), 'Fit TWD labels differ')
    c.exact(arrays['train_y_million'], np.asarray(y / 1e6, dtype=np.float32), 'Original fit targets/NPZ differ')
    for key, digest in spec['original_six_encoding_values_sha256'].items():
        c.need(c.values_sha(arrays[key]) == digest, 'Original R60 encoding identity: ' + key)
    for key, digest in spec['factor_values_sha256'].items():
        c.need(c.values_sha(arrays[key]) == digest and (arrays[key] > 0).all(), 'Original factor/weight identity: ' + key)
    card = arrays['cardinalities'].tolist()
    c.need(card == spec['expected_cardinalities'] and all(k >= 1 for k in card), 'Original cardinalities differ')
    for key in ('train_cat', 'query_cat'):
        c.need(all(((arrays[key][:, j] >= 0) & (arrays[key][:, j] < k)).all() for j, k in enumerate(card)),
               'Categorical code bounds differ')
    return {key: c.identity(value) for key, value in arrays.items()}


def load_training_arrays(spec, Xfit):
    import round60_recency_price_prior as recency
    y = c.load_array(spec['fit_labels']['fit_y'])
    with np.load(c.safe(spec['prepared']['path']), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    identities = input_contract(arrays, y, spec)
    weights, _, summary = recency._weights(recency._dates(Xfit))
    c.exact(weights.astype(np.float32), arrays['train_sample_weight'], 'Own-fit date weights differ')
    c.need(summary == c.read(PREPARATION)['date_weight_summary_float64'], 'Original date support differs')
    return identities


def torch_command(flag):
    return [str(ROOT / '.venv_torch/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(Path(__file__).resolve()), flag]


def child(cmd, log_name):
    tick = time.monotonic()
    with c.safe(BASE + '/' + log_name).open('x', encoding='utf8') as log:
        process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {'exit_code': process.returncode, 'elapsed_seconds': time.monotonic() - tick,
            'command': cmd, 'automatic_retry': False}


def train():
    c.need(c.read(BASE + '/freeze_execution.json')['exit_code'] == 0, 'Successful freeze required')
    r = request()
    spec = r['specification']
    axes, Xfit = load_axes(spec)
    identities = load_training_arrays(spec, Xfit)
    del Xfit
    c.write(BASE + '/training_input_review.json', {'status': 'passed_own_cut_before_fit',
        'array_identities': identities, 'axis_identities': {k: c.identity(v) for k, v in axes.items()},
        'fit_rows': FIT_ROWS, 'query_rows': QUERY_ROWS, 'query_labels_loaded': False,
        'raw_price_table_deserialized': False, 'encoder_or_prior_refit': False})
    cmd = [str(ROOT / '.venv_torch/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(c.safe(TRAINER)),
           '--data', str(c.safe(PREPARED)), '--out', str(c.safe(BASE + '/model'))]
    execution = child(cmd, 'train.log')
    c.write(BASE + '/native_training_execution.json', execution)
    c.need(execution['exit_code'] == 0, 'Diagnostic trainer failed; preserve without automatic retry')
    metadata = c.read(BASE + '/model/training.json')
    c.need(metadata['status'] == 'completed' and metadata['configuration'] == spec['expected_training_configuration']
           and metadata['source_sha256'] == TRAINER_SHA and metadata['input_sha256'] == PREPARED_SHA
           and metadata['fit_rows'] == FIT_ROWS and metadata['query_rows'] == QUERY_ROWS
           and [e['epoch'] for e in metadata['epochs']] == EPOCHS
           and metadata['saved_model_reload_exact'] and metadata['reload_rows'] == QUERY_ROWS
           and not metadata['query_labels_used'] and not metadata['query_labels_available']
           and not metadata['epoch_selection_uses_training_metrics'], 'Diagnostic training contract differs')
    original = c.read(OLD_TRAINING)
    for key in ('cardinalities', 'numeric_count', 'initial_training_median_ratio', 'initial_head_bias',
                'train_price_factor_million_sha256', 'query_price_factor_million_sha256', 'train_sample_weight_sha256'):
        c.need(metadata[key] == original[key], 'Original input/init contract changed: ' + key)
    snapshots = snapshot_records()
    c.write(BASE + '/unscored_snapshots.json', {'status': 'thirty_saved_unscored_snapshots',
        'snapshots': snapshots, 'training': c.record(BASE + '/model/training.json'),
        'native_training_execution': c.record(BASE + '/native_training_execution.json'),
        'final_model': c.record(BASE + '/model/model.pt'), 'query_labels_available': False})
    inspection = child(torch_command('--inspect-training'), 'checkpoint_inspection.log')
    c.write(BASE + '/checkpoint_inspection_execution.json', inspection)
    c.need(inspection['exit_code'] == 0, 'Checkpoint inspection failed; preserve evidence')
    review = c.read(BASE + '/checkpoints_review.json')
    c.need(review['status'] == 'passed_thirty_checkpoint_structure_and_final_state'
           and review['last_snapshot_equals_final_model'] is True, 'Checkpoint review differs')
    request()
    c.write(BASE + '/seal.json', {'status': 'thirty_snapshots_sealed_before_query_labels',
        'request': c.record(BASE + '/request.json'), 'snapshots': snapshots,
        'training': c.record(BASE + '/model/training.json'), 'checkpoint_review': c.record(BASE + '/checkpoints_review.json'),
        'native_training_execution': c.record(BASE + '/native_training_execution.json'),
        'inspection_execution': c.record(BASE + '/checkpoint_inspection_execution.json'),
        'array_identities': identities, 'axis_identities': {k: c.identity(v) for k, v in axes.items()},
        'checkpoint_epochs': EPOCHS, 'query_labels_used_for_fitting': False,
        'query_scores_computed': False, 'new_CSV': False, 'selection_changed': False})


def snapshot_records():
    folder = c.safe(BASE + '/model/epoch_checkpoints')
    expected = [f'epoch_{e:02d}.pt' for e in EPOCHS]
    c.need(folder.is_dir() and sorted(p.name for p in folder.iterdir()) == expected,
           'Exactly 30 expected checkpoint files required')
    return [{'epoch': e, **c.record(folder / f'epoch_{e:02d}.pt')} for e in EPOCHS]


def checkpoint_contract(saved, spec):
    c.need(isinstance(saved, dict) and set(saved) == {'numeric_count', 'cardinalities', 'configuration', 'state_dict'},
           'Checkpoint schema differs')
    c.need(saved['numeric_count'] == 144 and saved['cardinalities'] == spec['expected_cardinalities']
           and saved['configuration'] == spec['expected_training_configuration']
           and isinstance(saved['state_dict'], dict) and bool(saved['state_dict']), 'Checkpoint configuration differs')


def inspect_training():
    import torch
    import round107_training_curve_train_20261010 as trainer
    r = request()
    c.need(c.read(BASE + '/native_training_execution.json')['exit_code'] == 0, 'Completed native fit required')
    raw = c.read(BASE + '/unscored_snapshots.json')
    c.need(raw['status'] == 'thirty_saved_unscored_snapshots' and raw['snapshots'] == snapshot_records(),
           'Unscored snapshot identities differ')
    for key in ('training', 'native_training_execution', 'final_model'):
        c.need(c.record(raw[key]['path']) == raw[key], 'Unscored training artifact changed')
    spec = r['specification']
    trainer.check_configuration()
    final = torch.load(c.safe(raw['final_model']['path']), map_location='cpu', weights_only=True)
    checkpoint_contract(final, spec)
    model = trainer.EmbeddingMLP(144, spec['expected_cardinalities'])
    model.load_state_dict(final['state_dict'], strict=True)
    parameter_shapes = {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in final['state_dict'].items()}
    for rec in raw['snapshots']:
        c.need(c.record(rec['path']) == {k: rec[k] for k in ('path', 'sha256')}, 'Checkpoint file changed')
        saved = torch.load(c.safe(rec['path']), map_location='cpu', weights_only=True)
        checkpoint_contract(saved, spec)
        c.need(set(saved['state_dict']) == set(parameter_shapes), 'Checkpoint parameter names differ')
        for name, tensor in saved['state_dict'].items():
            shape, dtype = parameter_shapes[name]
            c.need(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape and tensor.dtype == dtype
                   and torch.isfinite(tensor).all().item(), 'Checkpoint tensor contract differs')
        model.load_state_dict(saved['state_dict'], strict=True)
        if rec['epoch'] == 30:
            c.need(all(torch.equal(saved['state_dict'][name], tensor) for name, tensor in final['state_dict'].items()),
                   'Last snapshot must equal fixed final model state')
    c.write(BASE + '/checkpoints_review.json', {'status': 'passed_thirty_checkpoint_structure_and_final_state',
        'snapshots': raw['snapshots'], 'last_snapshot_equals_final_model': True,
        'checkpoint_epochs': EPOCHS, 'query_labels_loaded': False, 'new_fits': 0, 'GPU_calls': 0,
        'native_forward_calls': 0, 'strict_state_dict_shape_dtype_finite_checks': True})


def completion_contract(execution, seal):
    c.need(execution['exit_code'] == 0 and execution['failure'] is None, 'Root train must exit 0 before query labels')
    c.need(seal['status'] == 'thirty_snapshots_sealed_before_query_labels'
           and seal['checkpoint_epochs'] == EPOCHS and len(seal['snapshots']) == 30
           and [rec['epoch'] for rec in seal['snapshots']] == EPOCHS
           and seal['query_labels_used_for_fitting'] is False
           and seal['query_scores_computed'] is False, 'Complete unscored 30-snapshot seal required')


def completed_training(r):
    execution = c.read(BASE + '/train_execution.json')
    seal = c.read(BASE + '/seal.json')
    completion_contract(execution, seal)
    c.need(seal['request'] == c.record(BASE + '/request.json') and seal['snapshots'] == snapshot_records(),
           'Completed train identities differ')
    for key in ('training', 'checkpoint_review', 'native_training_execution', 'inspection_execution'):
        rec = seal[key]
        c.need(c.record(rec['path']) == rec, 'Training evidence changed')
    c.need(c.read(seal['native_training_execution']['path'])['exit_code'] == 0
           and c.read(seal['inspection_execution']['path'])['exit_code'] == 0, 'Fit/inspection must succeed')
    c.need(r['specification']['prepared'] == {'path': PREPARED, 'sha256': PREPARED_SHA}, 'Only own-cut archive allowed')
    return seal


def evaluate():
    import pandas as pd
    r = request()
    seal = completed_training(r)  # The raw-price deserialization occurs strictly below this gate.
    spec = r['specification']
    axes, Xfit = load_axes(spec)
    identities = load_training_arrays(spec, Xfit)
    del Xfit
    c.need(identities == seal['array_identities']
           and {k: c.identity(v) for k, v in axes.items()} == seal['axis_identities'], 'Scoring input axes changed')
    raw = pd.read_pickle(c.safe(c.RAW))
    c.need(len(raw) == 455784 and raw.index.equals(pd.RangeIndex(455784)), 'Raw price row axis differs')
    yfit = raw.iloc[axes['fit_source_cache_indices']]['總價元'].to_numpy(dtype=np.float64, copy=True)
    yquery = raw.iloc[axes['query_source_cache_indices']]['總價元'].to_numpy(dtype=np.float64, copy=True)
    del raw
    c.exact(yfit, c.load_array(spec['fit_labels']['fit_y']), 'Saved float64 fit y differs from own raw fit rows')
    c.need(yquery.shape == (QUERY_ROWS,) and np.isfinite(yquery).all() and (yquery > 0).all(), 'Own-query TWD labels differ')
    labels = c.save_array(c.safe(BASE + '/query_y.npy'), yquery)
    c.write(BASE + '/evaluation_labels.json', {'status': 'own_query_labels_after_all_snapshots_sealed',
        'seal': c.record(BASE + '/seal.json'), 'root_train_execution': c.record(BASE + '/train_execution.json'),
        'fit_y': spec['fit_labels']['fit_y'], 'query_y': labels,
        'fit_raw_y_values_sha256': c.values_sha(yfit), 'query_raw_y_values_sha256': c.values_sha(yquery),
        'axes': {k: c.identity(v) for k, v in axes.items()}, 'raw_price_pickle_materialized_once': True,
        'raw_price_pickle_contains_all_train_prices': True, 'only_confirmation_rows_used_for_curve': True,
        'truth_precision_for_both_curves': 'original float64 TWD from canonical raw rows',
        'query_labels_used_for_fitting': False})
    del yfit, yquery, axes
    command = torch_command('--score-sealed') + ['--labels-sha256', c.sha(BASE + '/evaluation_labels.json')]
    execution = child(command, 'curve_scoring.log')
    c.write(BASE + '/scoring_execution.json', execution)
    c.need(execution['exit_code'] == 0, 'Sealed curve scoring failed; preserve evidence')
    curve = c.read(BASE + '/curve.json')
    curve_contract(curve)
    request()


def mae_twd(prediction, truth):
    c.need(prediction.dtype == truth.dtype == np.float64 and prediction.ndim == 1
           and prediction.shape == truth.shape and len(truth) > 0 and np.isfinite(prediction).all()
           and np.isfinite(truth).all() and (prediction >= 0).all() and (truth > 0).all(), 'MAE input differs')
    value = float(np.mean(np.abs(prediction - truth), dtype=np.float64))
    c.need(np.isfinite(value) and value >= 0, 'MAE must be finite nonnegative TWD')
    return value


def curve_contract(curve):
    c.need(curve['format'] == 'v25_prior_nn_learning_curve_v1'
           and curve['component'] == 'R107_prior_NN_native'
           and curve['metric'] == 'unweighted_total_price_MAE' and curve['unit'] == 'TWD'
           and curve['evaluation_mode'] == 'eval' and curve['epoch_selection'] == 'fixed_last_epoch'
           and curve['query_labels_used_for_fitting'] is False and curve['epochs'] == EPOCHS
           and curve['fit_rows'] == FIT_ROWS and curve['query_rows'] == QUERY_ROWS
           and bool(curve['fit_period_label']) and bool(curve['query_period_label'])
           and bool(curve['sources']) and 'best_epoch' not in curve, 'Curve schema differs')
    c.need(all(isinstance(rec, dict) and isinstance(rec.get('path'), str) and bool(rec['path'])
               and isinstance(rec.get('sha256'), str) and len(rec['sha256']) == 64
               and all(ch in '0123456789abcdef' for ch in rec['sha256']) for rec in curve['sources']),
           'Curve sources must contain paths and SHA256 identities')
    for key in ('train_mae_twd', 'validation_mae_twd'):
        a = np.asarray(curve[key], dtype=np.float64)
        c.need(a.shape == (30,) and np.isfinite(a).all() and (a >= 0).all(), 'Thirty finite MAE points required')


def score_sealed(labels_sha256):
    import torch
    import round107_training_curve_train_20261010 as trainer
    r = request()
    seal = completed_training(r)
    c.need(isinstance(labels_sha256, str) and len(labels_sha256) == 64
           and c.sha(BASE + '/evaluation_labels.json') == labels_sha256,
           'Parent-bound post-seal label manifest changed')
    labels = c.read(BASE + '/evaluation_labels.json')
    c.need(labels['status'] == 'own_query_labels_after_all_snapshots_sealed'
           and labels['seal'] == c.record(BASE + '/seal.json')
           and labels['root_train_execution'] == c.record(BASE + '/train_execution.json')
           and labels['fit_y'] == r['specification']['fit_labels']['fit_y']
           and labels['axes'] == seal['axis_identities'], 'Post-seal label authority differs')
    yfit, yquery = c.load_array(labels['fit_y']), c.load_array(labels['query_y'])
    with np.load(c.safe(PREPARED), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    c.need(input_contract(arrays, yfit, r['specification']) == seal['array_identities'], 'Scorer NPZ identities differ')
    c.need(yquery.dtype == np.float64 and yquery.shape == (QUERY_ROWS,)
           and c.values_sha(yquery) == labels['query_raw_y_values_sha256'], 'Scorer own-query label axis differs')
    trainer.check_configuration()
    c.need(trainer.CONFIG == r['specification']['expected_training_configuration'], 'Fixed diagnostic config differs')
    device = trainer.prior.setup()
    tn = torch.as_tensor(arrays['train_num'], device=device)
    tc = torch.as_tensor(arrays['train_cat'], device=device)
    qn = torch.as_tensor(arrays['query_num'], device=device)
    qc = torch.as_tensor(arrays['query_cat'], device=device)
    model = trainer.EmbeddingMLP(144, r['specification']['expected_cardinalities']).to(device)
    train_mae, query_mae, vectors = [], [], []
    pred_folder = c.safe(BASE + '/private_predictions')
    pred_folder.mkdir(exist_ok=False)
    for rec in seal['snapshots']:
        saved = torch.load(c.safe(rec['path']), map_location='cpu', weights_only=True)
        checkpoint_contract(saved, r['specification'])
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        with torch.inference_mode():
            train_pred = trainer.predict(model, tn, tc, arrays['train_price_factor_million'])
            query_pred = trainer.predict(model, qn, qc, arrays['query_price_factor_million'])
        train_mae.append(mae_twd(train_pred, yfit))
        query_mae.append(mae_twd(query_pred, yquery))
        epoch = rec['epoch']
        vectors.append({'epoch': epoch,
            'train': c.save_array(pred_folder / f'epoch_{epoch:02d}_train.npy', train_pred),
            'query': c.save_array(pred_folder / f'epoch_{epoch:02d}_query.npy', query_pred)})
        if epoch == 30:
            original_final = np.load(c.safe(BASE + '/model/predictions_twd.npy'), allow_pickle=False)
            c.exact(query_pred, original_final, 'Final epoch query output differs from diagnostic trainer reload')
        print(json.dumps({'epoch': epoch, 'train_mae_twd': train_mae[-1],
                          'validation_mae_twd': query_mae[-1]}), flush=True)
    c.write(BASE + '/private_prediction_manifest.json', {'epochs': vectors, 'new_fits': 0,
        'all_predictions_eval_mode': True, 'query_labels_used_for_fitting': False,
        'all_saved_vectors_reload_exact': True, 'last_query_equals_diagnostic_final_reload': True})
    curve = {'format': 'v25_prior_nn_learning_curve_v1', 'created_at_utc': c.now(),
        'component': 'R107_prior_NN_native', 'metric': 'unweighted_total_price_MAE', 'unit': 'TWD',
        'evaluation_mode': 'eval', 'epoch_selection': 'fixed_last_epoch',
        'query_labels_used_for_fitting': False,
        'fit_period_label': '成交日期 < 2023-08-11', 'query_period_label': '2023-08-11～2024-08-10',
        'fit_rows': FIT_ROWS, 'query_rows': QUERY_ROWS, 'epochs': EPOCHS,
        'train_mae_twd': train_mae, 'validation_mae_twd': query_mae,
        'sources': [c.record(PLAN), c.record(BASE + '/request.json'), c.record(BASE + '/seal.json'),
                    c.record(BASE + '/model/training.json'), c.record(BASE + '/evaluation_labels.json'),
                    c.record(BASE + '/private_prediction_manifest.json')],
        'original_v25_training_trajectory': False, 'diagnostic_refit_only': True,
        'model_capacity_varied': False, 'x_axis': 'epoch at fixed architecture',
        'full_v25_median_curve': False, 'Private_score': None,
        'train_truth_and_query_truth': 'original canonical raw float64 TWD; verified saved fit_y exact',
        'fit_prior': 'group-fold OOF', 'query_prior': 'full outer-fit',
        'gap_also_reflects_prior_construction_and_time_shift': True,
        'historical_query_reused_not_blind': True, 'new_CSV': False, 'selection_changed': False}
    curve_contract(curve)
    c.write(BASE + '/curve.json', curve)


def self_check():
    """Source-only and artificial contracts. Never read true arrays/prices/models."""
    checks = []
    def ok(name, value):
        c.need(bool(value), 'Artificial check failed: ' + name)
        checks.append(name)
    def rejects(name, function):
        try:
            function()
        except (ValueError, KeyError, FileExistsError):
            checks.append(name)
        else:
            raise AssertionError('Artificial rejection failed: ' + name)
    diff = source_difference()
    ok('exact_source_delta_only_checkpoint_save', diff.count('@@') == 2 and '+        checkpoint_dir=' in diff)
    ok('no_query_label_in_trainer', 'query_y' not in c.safe(TRAINER).read_text(encoding='utf8'))
    ok('no_rng_or_eval_in_snapshot_insertion', all(v not in SNAPSHOT_BLOCK for v in ('rand', 'predict(', '.eval(', 'query_y')))
    ok('exclusive_snapshot_writes', "checkpoint_path.open('xb')" in SNAPSHOT_BLOCK)
    days = np.array([0., 1., 2., 3., 4., np.nan], dtype=np.float64)
    mapping = np.array([2, 3, 5, 7, 9, 11], dtype=np.int64)
    axes = {'fit_indices': np.array([0, 1], dtype=np.int64), 'query_indices': np.array([2, 3], dtype=np.int64),
            'fit_source_cache_indices': mapping[[0, 1]], 'query_source_cache_indices': mapping[[2, 3]]}
    fi, qi = temporal_contract(days, mapping, axes, '2000-01-03', '2000-01-04', 2, 2)
    ok('strict_fit_and_inclusive_query_date_boundary', fi.tolist() == [0, 1] and qi.tolist() == [2, 3])
    for key in INDEX_KEYS:
        bad = {k: v.copy() for k, v in axes.items()}
        bad[key][0] += 1
        rejects('misaligned_' + key, lambda bad=bad: temporal_contract(days, mapping, bad, '2000-01-03', '2000-01-04', 2, 2))
    rejects('index_wrong_dtype', lambda: index_contract(np.array([0., 1.]), 3))
    rejects('index_duplicate', lambda: index_contract(np.array([0, 0], dtype=np.int64), 3))
    rejects('index_descending', lambda: index_contract(np.array([1, 0], dtype=np.int64), 3))
    y = np.array([1e6, 2e6, 3e6, 4e6], dtype=np.float64)
    arrays = {'train_num': np.zeros((4, 144), dtype=np.float32), 'query_num': np.zeros((2, 144), dtype=np.float32),
        'train_cat': np.zeros((4, 23), dtype=np.int64), 'query_cat': np.zeros((2, 23), dtype=np.int64),
        'cardinalities': np.full(23, 5, dtype=np.int64), 'train_y_million': (y / 1e6).astype(np.float32),
        'train_price_factor_million': np.ones(4, dtype=np.float32), 'query_price_factor_million': np.ones(2, dtype=np.float32),
        'train_sample_weight': np.ones(4, dtype=np.float32)}
    six = ('train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat', 'cardinalities')
    factors = ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight')
    spec = {'fit_rows': 4, 'query_rows': 2, 'original_six_encoding_values_sha256': {k: c.values_sha(arrays[k]) for k in six},
        'factor_values_sha256': {k: c.values_sha(arrays[k]) for k in factors}, 'expected_cardinalities': [5] * 23,
        'expected_training_configuration': {'epochs': 30, 'seed': 42}}
    ok('original_nine_array_schema', len(input_contract(arrays, y, spec)) == 9)
    bad = dict(arrays, query_y=np.ones(2))
    rejects('query_labels_in_NPZ', lambda: input_contract(bad, y, spec))
    bad = dict(arrays, train_y_million=np.zeros(4, dtype=np.float32))
    rejects('fit_target_mismatch', lambda: input_contract(bad, y, spec))
    for key in six + factors:
        bad = {k: v.copy() for k, v in arrays.items()}
        bad[key].flat[0] += 1
        rejects('changed_original_array_' + key, lambda bad=bad: input_contract(bad, y, spec))
    ok('MAE_in_TWD_not_million_or_training_loss', mae_twd(np.array([1e6 + 10., 2e6 - 30.]), y[:2]) == 20.)
    rejects('MAE_truth_precision_mismatch', lambda: mae_twd(y.astype(np.float32), y))
    rejects('MAE_shape_mismatch', lambda: mae_twd(y[:2], y))
    rejects('MAE_nonfinite', lambda: mae_twd(np.array([np.nan]), y[:1]))
    seal = {'status': 'thirty_snapshots_sealed_before_query_labels', 'checkpoint_epochs': EPOCHS,
        'snapshots': [{'epoch': e} for e in EPOCHS], 'query_labels_used_for_fitting': False, 'query_scores_computed': False}
    completion_contract({'exit_code': 0, 'failure': None}, seal)
    ok('all_30_seal_and_root_exit0_required', True)
    rejects('failed_training_before_labels', lambda: completion_contract({'exit_code': 1, 'failure': 'error'}, seal))
    bad_seal = dict(seal, snapshots=seal['snapshots'][:-1])
    rejects('missing_snapshot_before_labels', lambda: completion_contract({'exit_code': 0, 'failure': None}, bad_seal))
    bad_seal = dict(seal, checkpoint_epochs=list(range(1, 30)))
    rejects('epoch_schedule_changed', lambda: completion_contract({'exit_code': 0, 'failure': None}, bad_seal))
    saved = {'numeric_count': 144, 'cardinalities': [5] * 23,
        'configuration': spec['expected_training_configuration'], 'state_dict': {'dummy': np.ones(1)}}
    checkpoint_contract(saved, spec)
    ok('checkpoint_config_contract', True)
    rejects('checkpoint_unknown_key', lambda: checkpoint_contract(dict(saved, query_y=[]), spec))
    rejects('checkpoint_wrong_cardinality', lambda: checkpoint_contract(dict(saved, cardinalities=[6] * 23), spec))
    curve = {'format': 'v25_prior_nn_learning_curve_v1', 'component': 'R107_prior_NN_native',
        'metric': 'unweighted_total_price_MAE', 'unit': 'TWD', 'evaluation_mode': 'eval',
        'epoch_selection': 'fixed_last_epoch', 'query_labels_used_for_fitting': False, 'epochs': EPOCHS,
        'fit_rows': FIT_ROWS, 'query_rows': QUERY_ROWS, 'fit_period_label': 'fit', 'query_period_label': 'query',
        'sources': [{'path': 'synthetic', 'sha256': '0' * 64}], 'train_mae_twd': [1.] * 30, 'validation_mae_twd': [2.] * 30}
    curve_contract(curve)
    ok('exact_30point_public_curve_contract', True)
    rejects('no_best_epoch_selection', lambda: curve_contract(dict(curve, best_epoch=7)))
    rejects('no_mixed_training_loss_units', lambda: curve_contract(dict(curve, unit='million TWD')))
    rejects('no_missing_curve_epoch', lambda: curve_contract(dict(curve, train_mae_twd=[1.] * 29)))
    rejects('no_nonfinite_curve_point', lambda: curve_contract(dict(curve, validation_mae_twd=[np.inf] * 30)))
    return {'status': 'source_and_artificial_contracts_passed', 'checks': checks, 'check_count': len(checks),
        'source': c.record(__file__), 'trainer': c.record(TRAINER), 'original_trainer': c.record(ORIGINAL),
        'plan': c.record(PLAN), 'true_NPZ_features_prices_indices_models_loaded': False,
        'model_fits': 0, 'native_models_created_or_loaded': 0, 'GPU_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('freeze', 'train', 'evaluate', 'inspect-training', 'score-sealed', 'self-check'):
        group.add_argument('--' + flag, action='store_true')
    parser.add_argument('--labels-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, allow_nan=False), flush=True)
        return
    if args.inspect_training:
        inspect_training()
        return
    if args.score_sealed:
        score_sealed(args.labels_sha256)
        return
    action = 'freeze' if args.freeze else 'train' if args.train else 'evaluate'
    if action == 'freeze':
        c.safe(BASE).mkdir(parents=True, exist_ok=False)
    else:
        c.need(c.safe(BASE).is_dir(), 'Freeze first; do not create a replacement run')
    receipt = BASE + '/' + action + '_execution.json'
    c.need(not c.safe(receipt).exists() and not c.safe(receipt).with_suffix('.sha256').exists(),
           'Preserve previous execution; no automatic retry')
    tick, failure = time.monotonic(), None
    try:
        globals()[action]()
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        c.write(receipt, {'exit_code': 1 if failure else 0, 'failure': failure,
            'elapsed_seconds': time.monotonic() - tick, 'automatic_retry': False,
            'source': c.record(__file__), 'plan': c.record(PLAN), 'action': action,
            'new_CSV': False, 'selection_changed': False})


if __name__ == '__main__':
    main()
