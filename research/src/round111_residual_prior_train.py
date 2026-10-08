"""R111: one fixed tabular residual architecture with the immutable R63 objective.

This is a task-specific adaptation, not an exact implementation of the RTDL
paper. Architecture/capacity/initialization RNG consumption change together.
The nine R63 arrays, independent categorical dropout, ratio semantics,
fit-date weighted total MAE, optimizer and fixed 30 epochs are retained.
No query labels, adaptive epoch selection, parameter averaging or SmoothL1.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import round47_nn_train as original_architecture
import round63_recency_weighted_prior_train as baseline

PARENT_TRAINER = Path(__file__).resolve().with_name('round63_recency_weighted_prior_train.py')
PARENT_TRAINER_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
ORIGINAL_ARCHITECTURE_SHA = '68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
MODEL_CLASS = 'round111_residual_prior_train.TabularResidualPrior'
NUMERIC_COUNT, CATEGORY_COUNT = 144, 23
ARRAY_NAMES = ('train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat',
               'cardinalities', 'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight')
BASELINE_CONFIG = copy.deepcopy(baseline.CONFIG)
ARCHITECTURE_OVERRIDES = {
    'hidden': [256, 256],
    'architecture': 'fixed task-adapted pre-LayerNorm tabular residual network',
    'input_projection': 'Linear(144 + sum(embedding_dimensions), 256)',
    'residual_width': 256,
    'residual_expansion': 512,
    'residual_block_count': 2,
    'residual_block_formula': 'x + Dropout(.1)(Linear256(Dropout(.1)(ReLU(Linear512(LayerNorm256(x))))))',
    'head_formula': 'Softplus(Linear1(ReLU(LayerNorm256(x))))',
    'official_paper_replication_claimed': False,
    'skip_connection_only_causal_claimed': False,
}
CONFIG = {**copy.deepcopy(BASELINE_CONFIG), **copy.deepcopy(ARCHITECTURE_OVERRIDES)}
PRICE_FACTOR_FORMULA = baseline.PRICE_FACTOR_FORMULA
OUTPUT_UNITS = baseline.OUTPUT_UNITS
TRAIN_FACTOR_CONSTRUCTION = baseline.TRAIN_FACTOR_CONSTRUCTION
QUERY_FACTOR_CONSTRUCTION = baseline.QUERY_FACTOR_CONSTRUCTION
# These utilities are imported without modifying any original module state.
sample_weight_values = baseline.sample_weight_values
sample_weight_summary = baseline.sample_weight_summary
price_factor_values = baseline.price_factor_values
weighted_loss = baseline.weighted_loss
to_twd = baseline.to_twd
predict_ratio = baseline.predict_ratio
write = baseline.write


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_configuration():
    need(sha(PARENT_TRAINER) == PARENT_TRAINER_SHA, 'Original R63 source changed')
    need(sha(original_architecture.__file__) == ORIGINAL_ARCHITECTURE_SHA, 'Original R47 source changed')
    baseline.check_configuration()  # Also checks immutable R52 and original R47 configuration.
    need(BASELINE_CONFIG == baseline.CONFIG, 'Complete original R63 configuration changed')
    need(CONFIG == {**BASELINE_CONFIG, **ARCHITECTURE_OVERRIDES}, 'Only the fixed architecture may change')
    need(CONFIG['seed'] == 42 and CONFIG['epochs'] == 30 and CONFIG['batch_size'] == 2048,
         'Fixed seed, epochs and batch are required')
    need(CONFIG['dropout'] == CONFIG['category_dropout'] == .1, 'Original dropout recipe required')
    need(CONFIG['residual_width'] == 256 and CONFIG['residual_expansion'] == 512
         and CONFIG['residual_block_count'] == 2, 'Fixed residual recipe required')


class PreNormResidualBlock(nn.Module):
    """Per-row LayerNorm followed by a 256 -> 512 -> 256 residual branch."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.expand = nn.Linear(256, 512)
        self.activation = nn.ReLU()
        self.dropout_inner = nn.Dropout(.1)
        self.project = nn.Linear(512, 256)
        self.dropout_output = nn.Dropout(.1)

    def forward(self, x):
        branch = self.expand(self.norm(x))
        branch = self.dropout_inner(self.activation(branch))
        return x + self.dropout_output(self.project(branch))


class TabularResidualPrior(nn.Module):
    def __init__(self, numeric_count, cardinalities):
        super().__init__()
        need(type(numeric_count) is int and numeric_count == NUMERIC_COUNT, 'Exactly 144 numeric inputs required')
        need(len(cardinalities) == CATEGORY_COUNT and all(type(k) is int and k >= 1 for k in cardinalities),
             'Exactly 23 positive integer cardinalities required')
        self.dimensions = [min(32, max(4, math.ceil(math.sqrt(k)))) for k in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(k, d) for k, d in zip(cardinalities, self.dimensions)])
        self.input_projection = nn.Linear(numeric_count + sum(self.dimensions), 256)
        self.blocks = nn.ModuleList([PreNormResidualBlock(), PreNormResidualBlock()])
        self.head_norm = nn.LayerNorm(256)
        self.head_activation = nn.ReLU()
        self.head = nn.Linear(256, 1)
        self.positive = nn.Softplus()

    def masked_categories(self, categorical):
        if self.training:
            drop = torch.rand(categorical.shape, device=categorical.device) < CONFIG['category_dropout']
            return categorical.masked_fill(drop, 0)
        return categorical

    def forward(self, numeric, categorical):
        categorical = self.masked_categories(categorical)
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        x = self.input_projection(torch.cat([numeric, *embedded], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.positive(self.head(self.head_activation(self.head_norm(x)))).squeeze(1)


def validate_arrays(arrays):
    """Accept only frozen nine-array X/fit-y/factor/weight inputs, never query y."""
    need(set(arrays) == set(ARRAY_NAMES), 'Exactly nine NPZ fields required; no query labels')
    need(all(isinstance(arrays[k], np.ndarray) and not arrays[k].dtype.hasobject for k in ARRAY_NAMES),
         'Only non-object numpy arrays are permitted')
    xn, xc, y, qn, qc, card, af, qaf, sw = [arrays[k] for k in ARRAY_NAMES]
    need(xn.ndim == qn.ndim == xc.ndim == qc.ndim == 2 and len(xn) > 0 and len(qn) > 0,
         'Nonempty aligned matrices required')
    need(card.dtype == np.int64 and card.shape == (CATEGORY_COUNT,), '23 int64 cardinalities required')
    need(xn.shape[1] == qn.shape[1] == NUMERIC_COUNT and xc.shape[1] == qc.shape[1] == CATEGORY_COUNT,
         'Frozen 144 numeric / 23 category input width required')
    need(len(xn) == len(xc) == len(y) and len(qn) == len(qc) and y.ndim == 1, 'Row alignment differs')
    need(xn.dtype == qn.dtype == y.dtype == np.float32 and xc.dtype == qc.dtype == np.int64,
         'Original float32 numeric/target and int64 category dtypes required')
    need(np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y > 0).all(),
         'Numeric inputs and positive targets must be finite')
    af, qaf = price_factor_values(af, len(xn)), price_factor_values(qaf, len(qn))
    sw = sample_weight_values(sw, len(xn))
    cards = [int(k) for k in card]
    for i, k in enumerate(cards):
        need(k >= 1 and ((xc[:, i] >= 0) & (xc[:, i] < k)).all()
             and ((qc[:, i] >= 0) & (qc[:, i] < k)).all(), 'Categorical code outside fitted cardinality')
    return xn, xc, y, qn, qc, cards, af, qaf, sw


def array_identity(array):
    return {'dtype': str(array.dtype), 'shape': list(array.shape),
            'values_sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def setup():
    # Same original R47 setup choices, read from this checked, independent CONFIG.
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['thread_count'])
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded experiment requires CUDA; no silent CPU fit or adaptive retry')
    torch.cuda.manual_seed_all(CONFIG['seed'])
    return torch.device('cuda:0')


def checkpoint_payload(model, numeric_count, cardinalities):
    return {'model_class': MODEL_CLASS, 'numeric_count': numeric_count, 'cardinalities': list(cardinalities),
            'configuration': copy.deepcopy(CONFIG), 'baseline_configuration': copy.deepcopy(BASELINE_CONFIG),
            'parent_trainer_source_sha256': PARENT_TRAINER_SHA,
            'original_architecture_source_sha256': ORIGINAL_ARCHITECTURE_SHA,
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def restore_checkpoint(payload, device):
    need(set(payload) == {'model_class', 'numeric_count', 'cardinalities', 'configuration', 'baseline_configuration',
                         'parent_trainer_source_sha256', 'original_architecture_source_sha256', 'state_dict'},
         'Checkpoint schema differs')
    need(payload['model_class'] == MODEL_CLASS and payload['configuration'] == CONFIG
         and payload['baseline_configuration'] == BASELINE_CONFIG
         and payload['parent_trainer_source_sha256'] == PARENT_TRAINER_SHA
         and payload['original_architecture_source_sha256'] == ORIGINAL_ARCHITECTURE_SHA,
         'Checkpoint actual architecture or original provenance differs')
    restored = TabularResidualPrior(payload['numeric_count'], payload['cardinalities'])
    expected = restored.state_dict()
    need(set(payload['state_dict']) == set(expected), 'Checkpoint parameter keys differ')
    for key, value in payload['state_dict'].items():
        need(isinstance(value, torch.Tensor) and value.dtype == expected[key].dtype
             and value.shape == expected[key].shape and bool(torch.isfinite(value).all()),
             'Checkpoint parameter dtype, shape or finiteness differs: ' + key)
    restored.load_state_dict(payload['state_dict'], strict=True)
    return restored.to(device)


def run(data_path, out):
    start = time.monotonic()
    data_path, out = Path(data_path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ('run.json', 'model.pt', 'training.json', 'predictions_twd.npy', 'predictions_ratio.npy')
    if any((out / name).exists() for name in names):
        raise FileExistsError('Preserve existing run/model/predictions; no overwrite or automatic retry')
    check_configuration()
    source_sha, input_sha = sha(__file__), sha(data_path)
    with np.load(data_path, allow_pickle=False) as z:
        need(set(z.files) == set(ARRAY_NAMES) and len(z.files) == len(ARRAY_NAMES),
             'Exactly nine original NPZ fields required before reading values; no duplicate or query-label fields')
        arrays = {key: z[key] for key in z.files}
    xn, xc, y, qn, qc, card, af, qaf, sw = validate_arrays(arrays)
    identities = {key: array_identity(value) for key, value in arrays.items()}
    need(sha(data_path) == input_sha, 'Input changed during loading')
    median = float(np.median(y.astype(np.float64) / af.astype(np.float64)))
    need(np.isfinite(median) and median > 0, 'Invalid fit ratio-target median')
    inverse_median = median + float(np.log(-np.expm1(-median)))
    need(np.isfinite(np.float32(inverse_median)), 'Nonfinite initial ratio head bias')
    device = setup()
    changed = sorted(k for k in BASELINE_CONFIG if CONFIG[k] != BASELINE_CONFIG[k])
    added = sorted(set(CONFIG) - set(BASELINE_CONFIG))
    meta = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'started',
        'source_sha256': source_sha, 'input_sha256': input_sha, 'input_arrays': identities,
        'configuration': copy.deepcopy(CONFIG), 'baseline_configuration': copy.deepcopy(BASELINE_CONFIG),
        'architecture_configuration_overrides': copy.deepcopy(ARCHITECTURE_OVERRIDES),
        'changed_configuration_fields': changed, 'added_configuration_fields': added,
        'parent_trainer_source': 'src/round63_recency_weighted_prior_train.py',
        'parent_trainer_source_sha256': PARENT_TRAINER_SHA,
        'original_architecture_source': 'src/round47_nn_train.py',
        'original_architecture_source_sha256': ORIGINAL_ARCHITECTURE_SHA,
        'architecture_source_sha256': source_sha, 'actual_model_class': MODEL_CLASS,
        'original_configuration_and_source_verified': True, 'original_globals_modified': False,
        'complete_R63_nonarchitecture_training_configuration_unchanged': True,
        'capacity_initialization_and_rng_consumption_also_change': True,
        'skip_connection_only_causal_claimed': False, 'official_paper_replication_claimed': False,
        'BatchNorm_statistics_present_or_recomputed': False,
        'training_objective_name': 'weighted MAE', 'epoch_selection_uses_training_metrics': False,
        'query_labels_available': False, 'query_labels_used': False,
        'fit_rows': len(xn), 'query_rows': len(qn), 'numeric_count': NUMERIC_COUNT,
        'categorical_count': CATEGORY_COUNT, 'cardinalities': card,
        'prediction_units': 'TWD', 'output_units': OUTPUT_UNITS, 'price_factor_formula': PRICE_FACTOR_FORMULA,
        'price_factor_uses_fit_labels': True, 'train_price_factor_construction': TRAIN_FACTOR_CONSTRUCTION,
        'query_price_factor_construction': QUERY_FACTOR_CONSTRUCTION,
        'price_factor_provenance_verified_in_trainer': False,
        'train_price_factor_million_sha256': identities['train_price_factor_million']['values_sha256'],
        'query_price_factor_million_sha256': identities['query_price_factor_million']['values_sha256'],
        'train_sample_weight_sha256': identities['train_sample_weight']['values_sha256'],
        'train_sample_weight_shape': list(sw.shape), 'train_sample_weight_dtype': str(sw.dtype),
        'train_sample_weight_summary': sample_weight_summary(sw),
        'sample_weight_provenance_verified_in_trainer': False, 'sample_weights_recomputed_in_trainer': False,
        'sample_weights_batch_renormalized': False, 'price_or_ratio_clipping': False,
        'price_factor_million_summary': {name: {'rows': len(a), 'min': float(a.min()), 'max': float(a.max()),
            'mean': float(a.mean(dtype=np.float64))} for name, a in [('train', af), ('query', qaf)]},
        'initial_training_median_ratio': median, 'initial_head_bias': inverse_median,
        'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'bitwise_retraining_determinism_claimed': False,
    }
    write(out / 'run.json', meta)
    model = TabularResidualPrior(NUMERIC_COUNT, card).to(device)
    with torch.no_grad():
        model.head.bias.fill_(inverse_median)
    tn, tc, ty = [torch.as_tensor(a, device=device) for a in (xn, xc, y)]
    ta, tw = [torch.as_tensor(a, device=device) for a in (af, sw)]
    qtn, qtc = [torch.as_tensor(a, device=device) for a in (qn, qc)]
    del arrays, xn, xc, y, qn, qc, af, sw
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['minimum_learning_rate'])
    epochs, fit_started = [], time.monotonic()
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        order = torch.randperm(len(tn), device=device)
        total, total_mae = torch.zeros((), device=device), torch.zeros((), device=device)
        lr = optimizer.param_groups[0]['lr']
        for first in range(0, len(tn), CONFIG['batch_size']):
            index = order[first:first + CONFIG['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tn[index], tc[index]) * ta[index]
            loss = weighted_loss(prediction, ty[index], tw[index])
            mae = (prediction.detach() - ty[index]).abs().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(index)
            total_mae += mae * len(index)
        value, mae_value = float((total / len(tn)).item()), float((total_mae / len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):
            raise FloatingPointError('Nonfinite objective/MAE; preserve failure without adaptive retry')
        epochs.append({'epoch': epoch, 'learning_rate': lr, 'training_mae_million_with_dropout': mae_value,
                       'training_weighted_mae_million_with_dropout': value})
        schedule.step()
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'event': 'epoch_complete', **epochs[-1],
                              'elapsed_seconds': time.monotonic() - fit_started}), flush=True)
    torch.cuda.synchronize()
    fit_seconds = time.monotonic() - fit_started
    with (out / 'model.pt').open('xb') as stream:
        torch.save(checkpoint_payload(model, NUMERIC_COUNT, card), stream)
    ratios = predict_ratio(model, qtn, qtc)
    predictions = to_twd(ratios, qaf)
    payload = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    need(payload['numeric_count'] == NUMERIC_COUNT and payload['cardinalities'] == card, 'Saved input identity differs')
    restored = restore_checkpoint(payload, device)
    reloaded_ratios = predict_ratio(restored, qtn, qtc)
    reloaded = to_twd(reloaded_ratios, qaf)
    need(ratios.dtype == reloaded_ratios.dtype and ratios.shape == reloaded_ratios.shape
         and ratios.tobytes() == reloaded_ratios.tobytes(), 'Saved full-query native-ratio reload differs')
    need(predictions.dtype == reloaded.dtype and predictions.shape == reloaded.shape
         and predictions.tobytes() == reloaded.tobytes(), 'Saved full-query TWD reload differs')
    records = {}
    for name, values in [('predictions_ratio.npy', ratios), ('predictions_twd.npy', predictions)]:
        with (out / name).open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
        saved = np.load(out / name, allow_pickle=False)
        need(array_identity(saved) == array_identity(values), 'Saved prediction array identity differs')
        records[name] = {'sha256': sha(out / name), **array_identity(values)}
    check_configuration()
    need(sha(__file__) == source_sha and sha(data_path) == input_sha, 'Source/input changed during fit')
    meta.update(status='completed', epochs=epochs, embedding_dimensions=model.dimensions,
                parameter_count=sum(p.numel() for p in model.parameters()), fit_seconds=fit_seconds,
                model_sha256=sha(out / 'model.pt'), prediction_artifacts=records,
                saved_model_reload_exact=True, saved_native_ratio_reload_exact=True, reload_rows=len(predictions),
                prediction_min_twd=float(predictions.min()), prediction_max_twd=float(predictions.max()),
                elapsed_seconds=time.monotonic() - start, maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out / 'training.json', meta)
    print(json.dumps({k: meta[k] for k in ('status', 'fit_rows', 'query_rows', 'fit_seconds',
                                         'elapsed_seconds', 'saved_model_reload_exact')}), flush=True)


def self_check():
    """Synthetic CPU-only contract/algebra/gradient/serialization checks; no fit or CUDA API."""
    checks = []
    original_configs = copy.deepcopy((original_architecture.CONFIG, baseline.CONFIG))
    check_configuration()
    torch.set_num_threads(1)
    # Seed the CPU generator directly; torch.manual_seed also seeds CUDA lazily.
    torch.random.default_generator.manual_seed(111)

    def ok(name, condition):
        need(bool(condition), 'Self-check failed: ' + name)
        checks.append(name)

    def rejects(name, operation):
        try:
            operation()
        except (ValueError, TypeError, RuntimeError):
            checks.append(name)
        else:
            raise AssertionError('Self-check unexpectedly accepted: ' + name)

    cards = [5] * CATEGORY_COUNT
    numeric = torch.randn(4, NUMERIC_COUNT, device='cpu')
    categorical = torch.ones((4, CATEGORY_COUNT), dtype=torch.int64, device='cpu')
    model = TabularResidualPrior(NUMERIC_COUNT, cards).eval()
    with torch.no_grad():
        output = model(numeric, categorical)
    ok('actual_shape_finite_positive', output.shape == (4,) and torch.isfinite(output).all() and (output > 0).all())
    ok('embedding_dimensions_equal_original', model.dimensions == original_architecture.EmbeddingMLP(NUMERIC_COUNT, cards).dimensions)
    ok('exactly_two_fixed_blocks', len(model.blocks) == 2 and all(isinstance(b, PreNormResidualBlock) for b in model.blocks))
    ok('no_BatchNorm_running_statistics', not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
       and not list(model.named_buffers()))
    altered = numeric.clone()
    altered[1:] += 17
    with torch.no_grad():
        independent = model(altered, categorical)
        perm = torch.tensor([3, 0, 2, 1])
        permuted = model(numeric[perm], categorical[perm])
    ok('query_other_rows_do_not_change_first_prediction', torch.equal(output[0], independent[0]))
    torch.testing.assert_close(permuted, output[perm], rtol=1e-6, atol=1e-7)
    checks.append('row_permutation_equivariance_with_CPU_arithmetic_tolerance')
    block = PreNormResidualBlock().eval()
    x = torch.randn(4, 256)
    with torch.no_grad():
        block.project.weight.zero_()
        block.project.bias.zero_()
    ok('zero_residual_branch_exact_identity', torch.equal(block(x), x))
    with torch.no_grad():
        block.project.bias.fill_(.25)
    ok('constant_residual_branch_exact_skip_addition', torch.equal(block(x), x + .25))
    model.train()
    state = torch.get_rng_state()
    expected_mask = torch.rand(categorical.shape) < .1
    torch.set_rng_state(state)
    dropped = model.masked_categories(categorical)
    ok('independent_category_mask_matches_original_formula', torch.equal(dropped, categorical.masked_fill(expected_mask, 0)))
    model.eval()
    ok('eval_categories_unchanged', torch.equal(model.masked_categories(categorical), categorical))
    model.zero_grad(set_to_none=True)
    unknown = torch.zeros_like(categorical)
    ratio = model(numeric, unknown)
    factor = torch.tensor([2., 3., 5., 7.])
    target = (ratio.detach() * factor) + 1
    weights = torch.tensor([.5, .5, 1., 2.])
    loss = weighted_loss(ratio * factor, target, weights)
    loss.backward()
    ok('weighted_total_MAE_exact_objective', torch.equal(loss.detach(), (weights * (ratio.detach() * factor - target).abs()).mean()))
    ok('all_model_gradients_finite', all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    ok('unknown_embedding_receives_gradient', all(e.weight.grad[0].abs().sum() > 0 for e in model.embeddings))
    gradient_input = torch.tensor([-2., -.5, 0., .5, 2.], dtype=torch.float64, requires_grad=True)
    gradient_weights = torch.tensor([.5, .5, .5, .5, 3.], dtype=torch.float64)
    exact_loss = weighted_loss(gradient_input, torch.zeros_like(gradient_input), gradient_weights)
    exact_loss.backward()
    torch.testing.assert_close(exact_loss, torch.tensor(1.5, dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(gradient_input.grad, torch.tensor([-.1, -.1, 0., .1, .6], dtype=torch.float64), rtol=0, atol=1e-15)
    checks.append('nonuniform_MAE_gradient_without_batch_renormalization')
    ratios = np.array([.5, 1., 2., 3.], dtype=np.float64)
    factors = np.array([2., 3., 5., 7.], dtype=np.float32)
    ok('native_ratio_to_TWD_exactly_once', np.array_equal(to_twd(ratios, factors), ratios * factors.astype(np.float64) * 1e6))
    arrays = {'train_num': numeric.numpy().copy(), 'train_cat': categorical.numpy().copy(),
              'train_y_million': np.ones(4, dtype=np.float32), 'query_num': numeric.numpy().copy(),
              'query_cat': categorical.numpy().copy(), 'cardinalities': np.array(cards, dtype=np.int64),
              'train_price_factor_million': factors.copy(), 'query_price_factor_million': factors.copy(),
              'train_sample_weight': np.ones(4, dtype=np.float32)}
    validate_arrays(arrays)
    checks.append('valid_nine_array_contract')
    rejects('reject_query_labels', lambda: validate_arrays({**arrays, 'query_y': np.ones(4)}))
    for key in ('train_num', 'train_y_million', 'train_price_factor_million', 'train_sample_weight'):
        bad = {**arrays, key: arrays[key].astype(np.float64)}
        rejects('reject_wrong_dtype_' + key, lambda bad=bad: validate_arrays(bad))
    rejects('reject_numeric_width_change', lambda: validate_arrays({**arrays, 'query_num': arrays['query_num'][:, :-1]}))
    rejects('reject_category_count_change', lambda: validate_arrays({**arrays, 'cardinalities': arrays['cardinalities'][:-1]}))
    for key in ('train_num', 'query_num', 'train_y_million', 'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
        bad = {**arrays, key: arrays[key].copy()}
        bad[key].flat[0] = np.nan
        rejects('reject_nonfinite_' + key, lambda bad=bad: validate_arrays(bad))
    for key in ('train_y_million', 'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
        bad = {**arrays, key: arrays[key].copy()}
        bad[key].flat[0] = 0
        rejects('reject_nonpositive_' + key, lambda bad=bad: validate_arrays(bad))
    for key in ('train_cat', 'query_cat'):
        bad = {**arrays, key: arrays[key].copy()}
        bad[key][0, 0] = 5
        rejects('reject_out_of_range_' + key, lambda bad=bad: validate_arrays(bad))
    model.eval()
    buffer = io.BytesIO()
    torch.save(checkpoint_payload(model, NUMERIC_COUNT, cards), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = restore_checkpoint(payload, torch.device('cpu'))
    raw = predict_ratio(model, numeric, categorical)
    replay = predict_ratio(restored, numeric, categorical)
    ok('in_memory_full_synthetic_native_reload_exact', raw.tobytes() == replay.tobytes())
    ok('in_memory_full_synthetic_TWD_reload_exact', to_twd(raw, factors).tobytes() == to_twd(replay, factors).tobytes())
    bad_payload = copy.deepcopy(payload)
    bad_payload['configuration']['residual_block_count'] = 3
    rejects('reject_checkpoint_recipe_change', lambda: restore_checkpoint(bad_payload, torch.device('cpu')))
    bad_payload = copy.deepcopy(payload)
    key = next(iter(bad_payload['state_dict']))
    bad_payload['state_dict'][key] = bad_payload['state_dict'][key].double()
    rejects('reject_checkpoint_silent_dtype_cast', lambda: restore_checkpoint(bad_payload, torch.device('cpu')))
    ok('original_global_configurations_unchanged', original_configs == (original_architecture.CONFIG, baseline.CONFIG))
    check_configuration()
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks),
            'configuration': copy.deepcopy(CONFIG), 'baseline_configuration': copy.deepcopy(BASELINE_CONFIG),
            'source_sha256': sha(__file__), 'parent_trainer_source_sha256': PARENT_TRAINER_SHA,
            'original_architecture_source_sha256': ORIGINAL_ARCHITECTURE_SHA,
            'parameter_count_synthetic_cardinality5': sum(p.numel() for p in model.parameters()),
            'model_fits': 0, 'optimizer_steps': 0, 'GPU_calls': 0, 'real_data_read': False,
            'housing_model_accuracy_checked': False, 'original_globals_modified': False,
            'author_selfcheck_not_independent_source_review': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-check', '--cpu-unit-check', dest='self_check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        if args.data is not None or args.out is not None:
            parser.error('Synthetic self-check does not accept housing input or output')
        print(json.dumps(self_check(), ensure_ascii=False, allow_nan=False), flush=True)
    else:
        if args.data is None or args.out is None:
            parser.error('--data and --out required for an explicitly launched housing fit')
        run(args.data, args.out)
