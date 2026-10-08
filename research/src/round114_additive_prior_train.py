"""Prepared R114 signed additive prior-residual MLP; not authorization to fit.

The immutable R63 nine inputs and R47 hidden MLP are retained. The signed
linear head learns residual million TWD, added to the saved fit-OOF/query-full
price factor. Weighted total MAE uses the unprojected total. Inference alone
projects the restored total to >=0. Parameterization, head activation, bias
and gradient scaling change together; this is not a loss-only intervention.
Root must freeze a four-origin plan and obtain independent preflight first.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import round47_nn_train as original
import round63_recency_weighted_prior_train as baseline

PARENT = Path(__file__).resolve().with_name('round63_recency_weighted_prior_train.py')
PARENT_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
ARCHITECTURE_SHA = '68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
MODEL_CLASS = 'round114_additive_prior_train.AdditivePriorMLP'
ARRAY_NAMES = ('train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat', 'cardinalities',
               'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight')
BASELINE_CONFIG = copy.deepcopy(baseline.CONFIG)
PARAMETERIZATION_OVERRIDES = {
    'positive_output': 'Identity signed residual; inference projection only after total restoration',
    'prediction_head': 'signed linear residual million TWD; total_million = saved_price_factor_million + residual_million',
    'output_units': 'signed million TWD residual',
    'loss_formula': 'mean(train_sample_weight * abs(train_price_factor_million + residual_million - train_y_million))',
    'initial_bias': 'median_fit(train_y_million - train_price_factor_million)',
    'head_weight_initialization': 'original R47 Linear weight initialization; not zeroed',
    'price_or_ratio_clipping': True,
    'price_or_ratio_clipping_scope': 'only one nonnegative projection of inference total; no training or ratio clipping',
    'training_total_clipped': False,
    'inference_total_projection': 'max((saved_factor_float64 + signed_residual_float64) * 1e6, 0)',
    'loss_only_change_claimed': False,
    'gradient_magnitude_matching_to_ratio_claimed': False,
}
CONFIG = {**copy.deepcopy(BASELINE_CONFIG), **copy.deepcopy(PARAMETERIZATION_OVERRIDES)}
sample_weight_values = baseline.sample_weight_values
sample_weight_summary = baseline.sample_weight_summary
price_factor_values = baseline.price_factor_values
write = baseline.write


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(a):
    return {'dtype': str(a.dtype), 'shape': list(a.shape),
            'values_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def check_configuration():
    need(sha(PARENT) == PARENT_SHA and sha(original.__file__) == ARCHITECTURE_SHA, 'Original R63/R47 source changed')
    baseline.check_configuration()
    need(BASELINE_CONFIG == baseline.CONFIG and CONFIG == {**BASELINE_CONFIG, **PARAMETERIZATION_OVERRIDES},
         'Only the fixed additive parameterization fields may change')
    for key in ('seed', 'epochs', 'batch_size', 'hidden', 'dropout', 'category_dropout', 'embedding_dimension',
                'optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'minimum_learning_rate', 'thread_count'):
        need(CONFIG[key] == baseline.CONFIG[key] == original.CONFIG[key], 'Original fixed training/hidden recipe: ' + key)


class AdditivePriorMLP(original.EmbeddingMLP):
    """Construct original weights in original RNG order, replacing only Softplus."""
    def __init__(self, numeric_count, cardinalities):
        need(type(numeric_count) is int and numeric_count == 144, 'Exactly original 144 numeric inputs')
        need(len(cardinalities) == 23 and all(type(k) is int and k >= 1 for k in cardinalities), '23 fitted cardinalities')
        super().__init__(numeric_count, cardinalities)
        self.positive = nn.Identity()  # Module-instance replacement; no original globals or source mutations.


def materialize_archive(z):
    need(set(z.files) == set(ARRAY_NAMES) and len(z.files) == len(ARRAY_NAMES),
         'Exactly original nine NPZ key names required before reading any values; no query y')
    return {key: z[key] for key in ARRAY_NAMES}


def validate_inputs(a):
    need(set(a) == set(ARRAY_NAMES) and all(isinstance(v, np.ndarray) and not v.dtype.hasobject for v in a.values()),
         'Exactly nine non-object arrays required')
    xn, xc, y, qn, qc, cardinalities, factor, query_factor, weight = [a[key] for key in ARRAY_NAMES]
    need(xn.ndim == qn.ndim == xc.ndim == qc.ndim == 2 and len(xn) > 0 and len(qn) > 0, 'Nonempty matrices')
    need(xn.shape[1] == qn.shape[1] == 144 and xc.shape[1] == qc.shape[1] == 23, 'Original numeric/category widths')
    need(len(xn) == len(xc) == len(y) and len(qn) == len(qc) and y.ndim == 1, 'Aligned input rows')
    need(xn.dtype == qn.dtype == y.dtype == np.float32 and xc.dtype == qc.dtype == np.int64,
         'Original float32 numeric/target and int64 categories')
    need(cardinalities.dtype == np.int64 and cardinalities.shape == (23,), '23 int64 cardinalities')
    need(np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y > 0).all(), 'Finite inputs/positive fit y')
    factor = price_factor_values(factor, len(xn))
    query_factor = price_factor_values(query_factor, len(qn))
    weight = sample_weight_values(weight, len(xn))
    cards = [int(k) for k in cardinalities]
    for i, k in enumerate(cards):
        need(k >= 1 and ((xc[:, i] >= 0) & (xc[:, i] < k)).all()
             and ((qc[:, i] >= 0) & (qc[:, i] < k)).all(), 'Category code outside fit vocabulary')
    return xn, xc, y, qn, qc, cards, factor, query_factor, weight


def initial_bias(y, factor):
    price_factor_values(factor, len(y))
    need(y.dtype == np.float32 and y.shape == factor.shape and np.isfinite(y).all() and (y > 0).all(), 'Fit-only positive target')
    bias = float(np.median(y.astype(np.float64) - factor.astype(np.float64)))
    need(np.isfinite(bias) and np.isfinite(np.float32(bias)), 'Finite signed fit residual median')
    return bias


def objective_loss(residual_million, factor_million, target_million, sample_weight):
    need(residual_million.ndim == 1 and residual_million.numel() > 0
         and factor_million.shape == target_million.shape == sample_weight.shape == residual_million.shape,
         'Aligned one-dimensional additive loss inputs')
    need(all(v.dtype == residual_million.dtype and v.device == residual_million.device
             for v in (factor_million, target_million, sample_weight)), 'Loss dtype/device alignment')
    # Deliberately no projection here, including when restored total is negative.
    return baseline.weighted_loss(factor_million + residual_million, target_million, sample_weight)


def to_twd(residual_million, factor_million):
    residual = np.asarray(residual_million)
    need(residual.dtype == np.float64 and residual.ndim == 1 and len(residual) > 0 and np.isfinite(residual).all(),
         'Finite signed float64 residual predictions')
    factor = price_factor_values(factor_million, len(residual))
    with np.errstate(over='raise', invalid='raise'):
        raw_total = (factor.astype(np.float64) + residual) * 1e6
    need(np.isfinite(raw_total).all(), 'Finite raw signed total TWD; no replacement or cap')
    positive = np.maximum(raw_total, 0.)
    return raw_total, positive


@torch.inference_mode()
def predict(model, numeric, categorical, factor_million):
    model.eval()
    chunks = [model(numeric[start:start + 4096], categorical[start:start + 4096]).cpu().numpy()
              for start in range(0, len(numeric), 4096)]
    residual = np.concatenate(chunks).astype(np.float64)
    return to_twd(residual, factor_million)


def checkpoint_payload(model, cards):
    return {'model_class': MODEL_CLASS, 'numeric_count': 144, 'cardinalities': list(cards),
            'configuration': copy.deepcopy(CONFIG), 'baseline_configuration': copy.deepcopy(BASELINE_CONFIG),
            'parent_trainer_source_sha256': PARENT_SHA, 'original_architecture_source_sha256': ARCHITECTURE_SHA,
            'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}


def restore_checkpoint(p, device):
    need(set(p) == {'model_class', 'numeric_count', 'cardinalities', 'configuration', 'baseline_configuration',
                    'parent_trainer_source_sha256', 'original_architecture_source_sha256', 'state_dict'}, 'Exact checkpoint schema')
    need(p['model_class'] == MODEL_CLASS and p['configuration'] == CONFIG and p['baseline_configuration'] == BASELINE_CONFIG
         and p['parent_trainer_source_sha256'] == PARENT_SHA and p['original_architecture_source_sha256'] == ARCHITECTURE_SHA,
         'Actual additive checkpoint recipe/provenance')
    model = AdditivePriorMLP(p['numeric_count'], p['cardinalities'])
    expected = model.state_dict()
    need(set(expected) == set(p['state_dict']), 'Checkpoint keys differ')
    for key, value in p['state_dict'].items():
        need(isinstance(value, torch.Tensor) and value.dtype == expected[key].dtype
             and value.shape == expected[key].shape and bool(torch.isfinite(value).all()), 'Invalid checkpoint tensor: ' + key)
    model.load_state_dict(p['state_dict'], strict=True)
    return model.to(device)


def projection_diagnostics(raw, native):
    need(raw.dtype == native.dtype == np.float64 and raw.ndim == 1 and raw.shape == native.shape and len(raw) > 0
         and np.isfinite(raw).all() and np.isfinite(native).all(), 'Finite aligned inference diagnostic vectors')
    need(np.array_equal(native, np.maximum(raw, 0.)), 'Exactly the fixed nonnegative projection after total restoration')
    correction = native - raw
    return {'raw_negative_total_rows': int(np.count_nonzero(raw < 0)),
            'raw_negative_total_fraction': float(np.count_nonzero(raw < 0) / len(raw)),
            'raw_zero_total_rows': int(np.count_nonzero(raw == 0)),
            'projected_zero_rows': int(np.count_nonzero(native == 0)),
            'projected_zero_fraction': float(np.count_nonzero(native == 0) / len(native)),
            'raw_min_TWD': float(raw.min()), 'raw_max_TWD': float(raw.max()),
            'projected_min_TWD': float(native.min()), 'projected_max_TWD': float(native.max()),
            'projection_mean_correction_TWD': float(correction.mean()),
            'projection_max_correction_TWD': float(correction.max()), 'projection_changes_count': int(np.count_nonzero(correction)),
            'query_y_used_for_diagnostics': False, 'negative_residual_not_equivalent_to_negative_total': True}


def run(data_path, out):
    tick = time.monotonic()
    data_path, out = Path(data_path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ('run.json', 'training.json', 'model.pt', 'predictions_raw_twd.npy', 'predictions_twd.npy')
    if any((out / name).exists() for name in names):
        raise FileExistsError('Preserve existing attempt/artifacts; no overwrite or adaptive retry')
    check_configuration()
    source_sha, input_sha = sha(__file__), sha(data_path)
    with np.load(data_path, allow_pickle=False) as z:
        arrays = materialize_archive(z)
    xn, xc, y, qn, qc, cards, factor, query_factor, weight = validate_inputs(arrays)
    identities = {key: identity(value) for key, value in arrays.items()}
    need(sha(data_path) == input_sha, 'Input changed during read')
    bias = initial_bias(y, factor)
    # Original setup only, after checked source/config and fit-only input validation.
    device = original.setup()
    meta = {'status': 'started', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': source_sha, 'input_sha256': input_sha, 'input_arrays': identities,
            'configuration': copy.deepcopy(CONFIG), 'baseline_configuration': copy.deepcopy(BASELINE_CONFIG),
            'parameterization_overrides': copy.deepcopy(PARAMETERIZATION_OVERRIDES),
            'changed_configuration_fields': sorted(k for k in BASELINE_CONFIG if CONFIG[k] != BASELINE_CONFIG[k]),
            'added_configuration_fields': sorted(set(CONFIG) - set(BASELINE_CONFIG)),
            'parent_trainer_source': 'src/round63_recency_weighted_prior_train.py', 'parent_trainer_source_sha256': PARENT_SHA,
            'architecture_source_sha256': ARCHITECTURE_SHA, 'original_architecture_source_sha256': ARCHITECTURE_SHA,
            'actual_model_class': MODEL_CLASS, 'original_globals_modified': False,
            'hidden_architecture_embedding_dropouts_optimizer_schedule_unchanged': True,
            'parameterization_initial_bias_and_gradient_also_change': True,
            'loss_only_change_claimed': False, 'gradient_magnitude_matching_to_ratio_claimed': False,
            'head_weight_zeroed': False, 'initial_prediction_equals_prior_claimed': False,
            'initial_training_median_residual_million': bias, 'initial_head_bias': float(np.float32(bias)),
            'initial_head_bias_dtype': 'float32', 'initial_head_bias_rounding': 'original parameter dtype float32',
            'training_objective_name': 'weighted MAE of unprojected additive total million TWD',
            'epoch_selection_uses_training_metrics': False, 'training_total_clipped': False,
            'inference_nonnegative_projection_after_total_restoration': True, 'other_price_cap_applied': False,
            'query_labels_available': False, 'query_labels_used': False,
            'fit_rows': len(xn), 'query_rows': len(qn), 'numeric_count': 144, 'categorical_count': 23, 'cardinalities': cards,
            'prediction_units': 'TWD', 'output_units': CONFIG['output_units'],
            'train_price_factor_construction': baseline.TRAIN_FACTOR_CONSTRUCTION,
            'query_price_factor_construction': baseline.QUERY_FACTOR_CONSTRUCTION,
            'price_factor_formula': baseline.PRICE_FACTOR_FORMULA, 'price_factor_used_as_additive_offset': True,
            'price_factor_uses_fit_labels': True, 'price_factor_provenance_verified_in_trainer': False,
            'train_price_factor_million_sha256': identities['train_price_factor_million']['values_sha256'],
            'query_price_factor_million_sha256': identities['query_price_factor_million']['values_sha256'],
            'train_sample_weight_sha256': identities['train_sample_weight']['values_sha256'],
            'train_sample_weight_summary': sample_weight_summary(weight),
            'sample_weight_provenance_verified_in_trainer': False, 'sample_weights_recomputed_in_trainer': False,
            'sample_weights_batch_renormalized': False,
            'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'bitwise_retraining_determinism_claimed': False}
    write(out / 'run.json', meta)
    model = AdditivePriorMLP(144, cards).to(device)
    with torch.no_grad():
        model.head.bias.fill_(bias)
    need(float(model.head.bias.detach().cpu().item()) == meta['initial_head_bias'], 'Actual initialized head bias float32')
    tn, tc, ty, tf, tw, qtn, qtc = [torch.as_tensor(a, device=device) for a in (xn, xc, y, factor, weight, qn, qc)]
    del arrays, xn, xc, y, qn, qc, factor, weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['minimum_learning_rate'])
    epochs, fit_started = [], time.monotonic()
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        order = torch.randperm(len(tn), device=device)
        total, total_mae, negative = [torch.zeros((), device=device) for _ in range(3)]
        lr = optimizer.param_groups[0]['lr']
        for first in range(0, len(tn), CONFIG['batch_size']):
            index = order[first:first + CONFIG['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            residual = model(tn[index], tc[index])
            raw_total = tf[index] + residual
            loss = objective_loss(residual, tf[index], ty[index], tw[index])
            mae = (raw_total.detach() - ty[index]).abs().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(index)
            total_mae += mae * len(index)
            negative += (raw_total.detach() < 0).sum()
        value, mae_value = float((total / len(tn)).item()), float((total_mae / len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):
            raise FloatingPointError('Nonfinite training objective; preserve failure without adaptive retry')
        epochs.append({'epoch': epoch, 'learning_rate': lr, 'training_mae_million_with_dropout': mae_value,
                       'training_weighted_mae_million_with_dropout': value,
                       'training_negative_raw_total_rows_with_dropout': int(negative.item())})
        schedule.step()
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'event': 'epoch_complete', **epochs[-1], 'elapsed_seconds': time.monotonic() - fit_started}), flush=True)
    torch.cuda.synchronize()
    fit_seconds = time.monotonic() - fit_started
    with (out / 'model.pt').open('xb') as f:
        torch.save(checkpoint_payload(model, cards), f)
    raw, native = predict(model, qtn, qtc, query_factor)
    payload = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    need(payload['cardinalities'] == cards and payload['numeric_count'] == 144, 'Actual saved input schema')
    restored = restore_checkpoint(payload, device)
    replay_raw, replay_native = predict(restored, qtn, qtc, query_factor)
    need(identity(raw) == identity(replay_raw) and raw.tobytes() == replay_raw.tobytes(), 'Full query raw signed total reload exact')
    need(identity(native) == identity(replay_native) and native.tobytes() == replay_native.tobytes(), 'Full query projected TWD reload exact')
    records = {}
    for name, values in [('predictions_raw_twd.npy', raw), ('predictions_twd.npy', native)]:
        with (out / name).open('xb') as f:
            np.save(f, values, allow_pickle=False)
        need(identity(np.load(out / name, allow_pickle=False)) == identity(values), 'Saved NPY identity')
        records[name] = {'sha256': sha(out / name), **identity(values)}
    check_configuration()
    need(sha(__file__) == source_sha and sha(data_path) == input_sha, 'Source/input changed during fit')
    meta.update(status='completed', epochs=epochs, parameter_count=sum(p.numel() for p in model.parameters()),
                embedding_dimensions=model.dimensions, fit_seconds=fit_seconds, model_sha256=sha(out / 'model.pt'),
                prediction_artifacts=records, saved_model_reload_exact=True, saved_raw_total_reload_exact=True,
                reload_rows=len(native), projection_diagnostics=projection_diagnostics(raw, native),
                prediction_min_twd=float(native.min()), prediction_max_twd=float(native.max()),
                elapsed_seconds=time.monotonic() - tick, maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out / 'training.json', meta)
    print(json.dumps({k: meta[k] for k in ('status', 'fit_rows', 'query_rows', 'fit_seconds', 'saved_model_reload_exact')}), flush=True)


def self_check():
    check_configuration()
    originals = copy.deepcopy((original.CONFIG, baseline.CONFIG))
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(114)
    checks = []
    def ok(name, condition):
        need(bool(condition), name)
        checks.append(name)
    def rejects(name, call):
        try:
            call()
        except (ValueError, TypeError, RuntimeError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    cards = [5] * 23
    rng = torch.get_rng_state()
    control = original.EmbeddingMLP(144, cards)
    torch.set_rng_state(rng)
    model = AdditivePriorMLP(144, cards)
    ok('original_parameter_initialization_and_rng_order_exact', all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items()))
    ok('original_hidden_body_and_linear_head_preserved', repr(model.body) == repr(control.body)
       and isinstance(model.head, nn.Linear) and isinstance(model.positive, nn.Identity)
       and model.dimensions == control.dimensions and model.head.weight.abs().sum() > 0)
    y_np, factor_np = np.array([1., 5., 10.], np.float32), np.array([4., 4., 4.], np.float32)
    ok('fit_signed_additive_median_bias', initial_bias(y_np, factor_np) == 1.)
    ok('negative_initial_bias_allowed', initial_bias(np.ones(3, np.float32), factor_np) == -3.)
    residual = torch.tensor([-4., -2., 0., 1.], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor([1., 1., 2., 3.], dtype=torch.float64)
    target, weight = torch.full_like(factor, 2.), torch.tensor([.5, .5, 1., 2.], dtype=torch.float64)
    loss = objective_loss(residual, factor, target, weight)
    loss.backward()
    torch.testing.assert_close(loss, (weight * (factor + residual - target).abs()).mean(), rtol=0, atol=0)
    torch.testing.assert_close(residual.grad, torch.tensor([-.125, -.125, 0., .5], dtype=torch.float64), rtol=0, atol=0)
    checks.append('unclipped_additive_MAE_value_and_autograd')
    negative = torch.tensor([-2.], dtype=torch.float64, requires_grad=True)
    negative_loss = objective_loss(negative, torch.ones_like(negative), torch.full_like(negative, 2.), torch.ones_like(negative))
    negative_loss.backward()
    ok('negative_total_trains_without_projection_dead_zone', negative_loss.item() == 3. and negative.grad.item() == -1.)
    for mode in ('additive', 'ratio'):
        z = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        f = torch.tensor([1., 10.], dtype=torch.float64)
        y, w = f + 5., torch.ones_like(f)
        objective = objective_loss(z, f, y, w) if mode == 'additive' else (w * (torch.nn.functional.softplus(z) * f - y).abs()).mean()
        objective.backward()
        expected = [-.5, -.5] if mode == 'additive' else [-.25, -2.5]
        torch.testing.assert_close(z.grad, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
        checks.append(mode + '_direct_head_gradient_semantics')
    raw, projected = to_twd(np.array([-2., -1., 0., 1.], np.float64), np.ones(4, np.float32))
    ok('add_factor_once_then_TWD_then_projection', raw.tolist() == [-1e6, 0., 1e6, 2e6] and projected.tolist() == [0., 0., 1e6, 2e6])
    diagnostics = projection_diagnostics(raw, projected)
    ok('negative_zero_projection_diagnostics', diagnostics['raw_negative_total_rows'] == 1
       and diagnostics['raw_zero_total_rows'] == 1 and diagnostics['projected_zero_rows'] == 2
       and diagnostics['raw_negative_total_fraction'] == .25 and diagnostics['projected_zero_fraction'] == .5)
    rejects('wrong_factor_dtype_rejected', lambda: to_twd(np.ones(4, np.float64), np.ones(4, np.float64)))
    rejects('nonfinite_signed_residual_rejected', lambda: to_twd(np.array([np.nan], np.float64), np.ones(1, np.float32)))
    class FakeArchive:
        def __init__(self, files):
            self.files, self.reads = files, 0
        def __getitem__(self, key):
            self.reads += 1
            raise AssertionError('Forbidden archive values read')
    archive = FakeArchive(list(ARRAY_NAMES[:-1]) + ['query_y'])
    rejects('query_y_replacement_key_refused_before_materializing', lambda: materialize_archive(archive))
    ok('invalid_archive_zero_value_reads', archive.reads == 0)
    archive = FakeArchive(list(ARRAY_NAMES) + ['train_num'])
    rejects('duplicate_archive_key_refused_before_materializing', lambda: materialize_archive(archive))
    numeric = torch.randn(4, 144)
    categorical = torch.ones((4, 23), dtype=torch.int64)
    arrays = {'train_num': numeric.numpy().copy(), 'query_num': numeric.numpy().copy(),
              'train_cat': categorical.numpy().copy(), 'query_cat': categorical.numpy().copy(),
              'train_y_million': np.ones(4, np.float32), 'cardinalities': np.array(cards, np.int64),
              'train_price_factor_million': np.ones(4, np.float32), 'query_price_factor_million': np.ones(4, np.float32),
              'train_sample_weight': np.ones(4, np.float32)}
    validate_inputs(arrays)
    checks.append('valid_original_nine_array_contract')
    for key in ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
        for value in (0., -1., np.nan, np.inf):
            bad = {**arrays, key: arrays[key].copy()}
            bad[key][0] = value
            rejects('invalid_' + key + '_' + str(value), lambda bad=bad: validate_inputs(bad))
    rejects('numeric_width_change_rejected', lambda: validate_inputs({**arrays, 'query_num': arrays['query_num'][:, :-1]}))
    model.eval()
    with torch.no_grad():
        model.head.bias.fill_(1.)
    model.zero_grad(set_to_none=True)
    r = model(numeric, categorical)
    objective_loss(r, torch.ones_like(r), r.detach() + 2., torch.ones_like(r)).backward()
    ok('finite_gradients_through_original_hidden_model', all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    buffer = io.BytesIO()
    torch.save(checkpoint_payload(model, cards), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = restore_checkpoint(payload, torch.device('cpu'))
    raw, projected = predict(model, numeric, categorical, np.ones(4, np.float32))
    replay_raw, replay_projected = predict(restored, numeric, categorical, np.ones(4, np.float32))
    ok('full_synthetic_raw_and_projected_checkpoint_reload_exact', raw.tobytes() == replay_raw.tobytes()
       and projected.tobytes() == replay_projected.tobytes())
    bad = copy.deepcopy(payload)
    key = next(iter(bad['state_dict']))
    bad['state_dict'][key] = bad['state_dict'][key].double()
    rejects('checkpoint_silent_dtype_cast_rejected', lambda: restore_checkpoint(bad, torch.device('cpu')))
    bad = copy.deepcopy(payload)
    bad['configuration']['training_total_clipped'] = True
    rejects('checkpoint_projection_policy_change_rejected', lambda: restore_checkpoint(bad, torch.device('cpu')))
    ok('original_module_configurations_unchanged', originals == (original.CONFIG, baseline.CONFIG))
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks), 'source_sha256': sha(__file__),
            'configuration': copy.deepcopy(CONFIG), 'baseline_configuration': copy.deepcopy(BASELINE_CONFIG),
            'author_selfcheck_not_independent_preflight': True, 'prepared_source_not_fit_authorization': True,
            'new_fits': 0, 'optimizer_steps': 0, 'GPU_calls': 0, 'housing_data_read': False, 'new_CSV': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-check', '--cpu-unit-check', dest='self_check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        if args.data is not None or args.out is not None:
            parser.error('Synthetic CPU self-check cannot accept housing input/output')
        print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    else:
        if args.data is None or args.out is None:
            parser.error('--data and --out explicitly required; caller owns frozen plan/preflight authorization')
        run(args.data, args.out)
