"""R109: fixed R63 training with equal parameter averaging over epochs 21--30.

The R63 nine-array contract, weighted-MAE loss, seed, initialization, architecture,
optimizer and all 30 cosine-schedule epochs are unchanged. After each of epochs
21--30, float32 model parameters are copied in fixed named-state order to a CPU
float64 accumulator. The ten equally weighted states are averaged once, then
converted to float32 for inference. This is parameter averaging, not prediction
averaging or standard SWA with a changed learning-rate schedule.

model.pt / predictions_twd.npy contain the averaged candidate. The same training
trajectory's epoch-30 control is preserved separately as model_epoch30.pt /
predictions_epoch30_twd.npy. Both models must pass full-query exact reload checks.
No query labels, adaptive epoch selection, extra fits or BatchNorm updates occur.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import round47_nn_train as prior
import round63_recency_weighted_prior_train as baseline

EmbeddingMLP = prior.EmbeddingMLP
ARCHITECTURE_SHA = '68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
PARENT_TRAINER = Path(__file__).resolve().with_name('round63_recency_weighted_prior_train.py')
PARENT_TRAINER_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
CONFIG = copy.deepcopy(baseline.CONFIG)
AVERAGE_EPOCHS = tuple(range(21, 31))
AVERAGE_COUNT = 10
PARAMETER_SELECTION = 'equal parameter mean of fixed end-of-epoch states 21--30; CPU float64 sum then float32'
CONTROL_PARAMETER_SELECTION = 'same-trajectory epoch 30; no parameter averaging'
PRICE_FACTOR_FORMULA = baseline.PRICE_FACTOR_FORMULA
OUTPUT_UNITS = baseline.OUTPUT_UNITS
TRAIN_FACTOR_CONSTRUCTION = baseline.TRAIN_FACTOR_CONSTRUCTION
QUERY_FACTOR_CONSTRUCTION = baseline.QUERY_FACTOR_CONSTRUCTION


def check_configuration():
    if hashlib.sha256(PARENT_TRAINER.read_bytes()).hexdigest() != PARENT_TRAINER_SHA:
        raise ValueError('Parent R63 trainer source changed')
    baseline.check_configuration()
    if CONFIG != baseline.CONFIG:
        raise ValueError('R109 must preserve the complete immutable R63 training configuration')
    if CONFIG['seed'] != 42 or CONFIG['epochs'] != 30 or CONFIG['batch_size'] != 2048:
        raise ValueError('Fixed seed/epochs/batch differ')
    if AVERAGE_EPOCHS != tuple(range(21, 31)) or AVERAGE_COUNT != 10:
        raise ValueError('Exactly ten fixed epoch-end states 21--30 are required')
    if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() != ARCHITECTURE_SHA:
        raise ValueError('R47 setup/architecture source changed')


class TailParameterAverage:
    """Ordered float32 states -> equal CPU float64 parameter mean; no RNG/optimizer."""

    def __init__(self):
        self.epochs = []
        self.names = None
        self.shapes = None
        self._sums = {}

    @torch.no_grad()
    def add(self, epoch, state):
        if len(self.epochs) >= AVERAGE_COUNT or epoch != AVERAGE_EPOCHS[len(self.epochs)]:
            raise ValueError('Parameter states must arrive exactly once in epoch order 21--30')
        if not isinstance(state, Mapping) or not state:
            raise ValueError('A nonempty named parameter state is required')
        names = tuple(state)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Parameter names must be nonempty strings')
        if self.names is not None and names != self.names:
            raise ValueError('Parameter names or their fixed accumulation order changed')
        values = {}
        shapes = {}
        for name in names:
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.layout != torch.strided:
                raise ValueError('Every averaged parameter must be a dense float32 tensor')
            shapes[name] = tuple(value.shape)
            if self.shapes is not None and shapes[name] != self.shapes[name]:
                raise ValueError('Parameter shape changed between epoch states')
            value = value.detach().to(device='cpu', dtype=torch.float64, copy=True)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite parameter state; no dropping or replacing epochs')
            values[name] = value
        # Validate every key before committing anything, including on error paths.
        if self.names is None:
            self.names, self.shapes = names, shapes
            self._sums = values
        else:
            for name in self.names:
                self._sums[name].add_(values[name])
        self.epochs.append(epoch)

    @torch.no_grad()
    def finish(self):
        if tuple(self.epochs) != AVERAGE_EPOCHS or self.names is None:
            raise ValueError('Cannot average fewer than the ten fixed epoch states')
        result = {}
        for name in self.names:
            value = (self._sums[name] / AVERAGE_COUNT).to(dtype=torch.float32)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite float32 averaged parameter; no clipping')
            result[name] = value
        return result


# Reuse the SHA-guarded parent's exact numeric contracts, objective and inference.
sample_weight_values = baseline.sample_weight_values
sample_weight_summary = baseline.sample_weight_summary
weighted_loss = baseline.weighted_loss
write = baseline.write
price_factor_values = baseline.price_factor_values
to_twd = baseline.to_twd
predict_ratio = baseline.predict_ratio
predict = baseline.predict


def save_predict_reload(model, numeric, categorical, factors, out, model_name, prediction_name, meta, selection, epochs):
    """Save an actual parameter state, predict, then check every saved query row."""
    model_path = out / model_name
    prediction_path = out / prediction_name
    if model_path.exists() or prediction_path.exists():
        raise FileExistsError('Refuse to replace either model/prediction artifact')
    payload = {
        'numeric_count': meta['numeric_count'], 'cardinalities': meta['cardinalities'],
        'configuration': copy.deepcopy(CONFIG), 'parameter_selection': selection,
        'parameter_state_epochs': list(epochs),
        'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    with model_path.open('xb') as handle:
        torch.save(payload, handle)
    predictions = predict(model, numeric, categorical, factors)
    loaded = torch.load(model_path, map_location='cpu', weights_only=True)
    if (loaded['configuration'] != CONFIG or loaded['cardinalities'] != meta['cardinalities']
            or loaded['numeric_count'] != meta['numeric_count'] or loaded['parameter_selection'] != selection
            or loaded['parameter_state_epochs'] != list(epochs)):
        raise ValueError('Saved model identity/configuration differs')
    # A deepcopy, unlike a fresh randomly initialized module, does not draw RNG.
    restored = copy.deepcopy(model)
    restored.load_state_dict(loaded['state_dict'], strict=True)
    reloaded = predict(restored, numeric, categorical, factors)
    if predictions.tobytes() != reloaded.tobytes():
        raise ValueError('Saved model full-query TWD reload differs')
    with prediction_path.open('xb') as handle:
        np.save(handle, predictions, allow_pickle=False)
    return {
        'model_file': model_name, 'predictions_file': prediction_name,
        'parameter_selection': selection, 'parameter_state_epochs': list(epochs),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'predictions_file_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        'saved_model_reload_exact': True, 'reload_rows': len(predictions),
        'prediction_min_twd': float(predictions.min()), 'prediction_max_twd': float(predictions.max()),
    }


def run(data_path, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    artifact_names = ('run.json', 'model.pt', 'training.json', 'predictions_twd.npy',
                      'model_epoch30.pt', 'predictions_epoch30_twd.npy')
    if any((out / name).exists() for name in artifact_names):
        raise FileExistsError('Run/output already exists; preserve evidence instead of retrying')
    expected = {'train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat', 'cardinalities',
                'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'}
    with np.load(data_path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected:
            raise ValueError('Expected exactly nine arrays including fit weights, no query labels')
        xn, xc, y, qn, qc, card, af, qaf, sw = [arrays[key] for key in (
            'train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat', 'cardinalities',
            'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight')]
    assert xn.ndim == qn.ndim == xc.ndim == qc.ndim == 2 and len(xn) > 0 and len(qn) > 0
    assert card.dtype == np.int64 and card.ndim == 1
    card = [int(k) for k in card]
    assert xn.shape[1] == qn.shape[1] and xc.shape[1] == qc.shape[1] == len(card)
    assert len(xn) == len(xc) == len(y) and len(qn) == len(qc) and y.ndim == 1
    assert xn.dtype == qn.dtype == y.dtype == np.float32 and xc.dtype == qc.dtype == np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y > 0).all()
    af = price_factor_values(af, len(xn))
    qaf = price_factor_values(qaf, len(qn))
    sw = sample_weight_values(sw, len(xn))
    for i, k in enumerate(card):
        assert k >= 1 and ((xc[:, i] >= 0) & (xc[:, i] < k)).all() and ((qc[:, i] >= 0) & (qc[:, i] < k)).all()
    check_configuration()
    device = prior.setup()
    median = float(np.median(y.astype(np.float64) / af.astype(np.float64)))
    if not np.isfinite(median) or median <= 0:
        raise ValueError('Invalid fit ratio-target median')
    inverse_median = median + float(np.log(-np.expm1(-median)))
    if not np.isfinite(np.float32(inverse_median)):
        raise ValueError('Nonfinite initial ratio head bias')
    meta = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'configuration': copy.deepcopy(CONFIG),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parent_trainer_source': 'src/round63_recency_weighted_prior_train.py',
        'parent_trainer_source_sha256': PARENT_TRAINER_SHA,
        'changed_configuration_fields': [], 'added_configuration_fields': [],
        'complete_R63_training_configuration_unchanged': True,
        'training_objective_name': 'weighted MAE', 'epoch_selection_uses_training_metrics': False,
        'configuration_epoch_selection_describes_baseline_training': True,
        'inference_parameter_selection': PARAMETER_SELECTION,
        'control_parameter_selection': CONTROL_PARAMETER_SELECTION,
        'tail_parameter_average': {
            'epochs': list(AVERAGE_EPOCHS), 'count': AVERAGE_COUNT,
            'snapshot_timing': 'after all optimizer steps of each listed epoch',
            'weights': [1 / AVERAGE_COUNT] * AVERAGE_COUNT,
            'accumulator_device': 'cpu', 'accumulator_dtype': 'float64',
            'input_parameter_dtype': 'float32', 'output_parameter_dtype': 'float32',
            'reduction_order': 'epochs 21 through 30, then fixed model.state_dict named order',
            'parameters_averaged_before_inference': True, 'predictions_averaged': False,
            'optimizer_or_scheduler_changed': False, 'averaging_draws_random_numbers': False,
            'standard_SWA_claimed': False, 'BatchNorm_statistics_recomputed': False,
            'one_training_trajectory_for_candidate_and_control': True,
        },
        'train_sample_weight_sha256': hashlib.sha256(sw.astype('<f4').tobytes()).hexdigest(),
        'train_sample_weight_shape': list(sw.shape), 'train_sample_weight_dtype': 'float32',
        'train_sample_weight_summary': sample_weight_summary(sw),
        'sample_weight_provenance_verified_in_trainer': False, 'sample_weights_recomputed_in_trainer': False,
        'sample_weights_batch_renormalized': False,
        'architecture_source_sha256': hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'fit_rows': len(xn), 'query_rows': len(qn), 'numeric_count': xn.shape[1], 'cardinalities': card,
        'query_labels_available': False, 'query_labels_used': False, 'prediction_units': 'TWD',
        'output_units': OUTPUT_UNITS, 'price_factor_formula': PRICE_FACTOR_FORMULA,
        'price_factor_uses_fit_labels': True, 'train_price_factor_construction': TRAIN_FACTOR_CONSTRUCTION,
        'query_price_factor_construction': QUERY_FACTOR_CONSTRUCTION,
        'price_factor_provenance_verified_in_trainer': False,
        'train_price_factor_million_sha256': hashlib.sha256(af.astype('<f4').tobytes()).hexdigest(),
        'query_price_factor_million_sha256': hashlib.sha256(qaf.astype('<f4').tobytes()).hexdigest(),
        'price_factor_million_summary': {name: {
            'rows': len(a), 'min': float(a.min()), 'max': float(a.max()), 'mean': float(a.mean(dtype=np.float64))
        } for name, a in [('train', af), ('query', qaf)]},
        'initial_training_median_ratio': median, 'initial_head_bias': inverse_median,
        'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'bitwise_retraining_determinism_claimed': False,
        'extra_sample_weights': True, 'price_or_ratio_clipping': False,
    }
    write(out / 'run.json', meta)
    model = EmbeddingMLP(xn.shape[1], card).to(device)
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError('Unexpected BatchNorm; no BN statistic recalibration is authorized')
    if list(model.named_buffers()) or tuple(model.state_dict()) != tuple(name for name, _ in model.named_parameters()):
        raise ValueError('Expected exactly named float32 parameters and no module buffers')
    with torch.no_grad():
        model.head.bias.fill_(inverse_median)
    tn = torch.as_tensor(xn, device=device)
    tc = torch.as_tensor(xc, device=device)
    ty = torch.as_tensor(y, device=device)
    ta = torch.as_tensor(af, device=device)
    tw = torch.as_tensor(sw, device=device)
    qtn = torch.as_tensor(qn, device=device)
    qtc = torch.as_tensor(qc, device=device)
    del xn, xc, y, qn, qc, af, sw
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['minimum_learning_rate'])
    average = TailParameterAverage()
    epochs = []
    fit_started = time.monotonic()
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        order = torch.randperm(len(tn), device=device)
        total = torch.zeros((), device=device)
        total_mae = torch.zeros((), device=device)
        lr = optimizer.param_groups[0]['lr']
        for first in range(0, len(tn), CONFIG['batch_size']):
            index = order[first:first + CONFIG['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            ratio = model(tn[index], tc[index])
            prediction = ratio * ta[index]
            loss = weighted_loss(prediction, ty[index], tw[index])
            mae = (prediction.detach() - ty[index]).abs().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(index)
            total_mae += mae * len(index)
        value = float((total / len(tn)).item())
        mae_value = float((total_mae / len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):
            raise FloatingPointError('Nonfinite weighted objective/unweighted MAE; no adaptive retry')
        if epoch in AVERAGE_EPOCHS:
            average.add(epoch, model.state_dict())
        epochs.append({
            'epoch': epoch, 'learning_rate': lr, 'training_mae_million_with_dropout': mae_value,
            'training_weighted_mae_million_with_dropout': value,
            'tail_parameter_state_collected': epoch in AVERAGE_EPOCHS,
            'tail_parameter_average_count': len(average.epochs),
        })
        schedule.step()
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'event': 'epoch_complete', **epochs[-1],
                              'elapsed_seconds': time.monotonic() - fit_started}), flush=True)
    torch.cuda.synchronize()
    fit_seconds = time.monotonic() - fit_started
    averaged_state = average.finish()
    control = save_predict_reload(
        model, qtn, qtc, qaf, out, 'model_epoch30.pt', 'predictions_epoch30_twd.npy',
        meta, CONTROL_PARAMETER_SELECTION, [30])
    model.load_state_dict(averaged_state, strict=True)
    candidate = save_predict_reload(
        model, qtn, qtc, qaf, out, 'model.pt', 'predictions_twd.npy',
        meta, PARAMETER_SELECTION, AVERAGE_EPOCHS)
    meta['tail_parameter_average'].update(
        collected_epochs=list(average.epochs), collected_count=len(average.epochs),
        named_parameter_order=list(average.names),
        parameter_shapes={name: list(average.shapes[name]) for name in average.names},
        module_buffer_count=0, collection_included_in_fit_seconds=True)
    meta.update(
        status='completed', epochs=epochs, parameter_count=sum(p.numel() for p in model.parameters()),
        fit_seconds=fit_seconds, saved_model_reload_exact=candidate['saved_model_reload_exact'],
        saved_epoch30_model_reload_exact=control['saved_model_reload_exact'], reload_rows=candidate['reload_rows'],
        prediction_min_twd=candidate['prediction_min_twd'], prediction_max_twd=candidate['prediction_max_twd'],
        averaged_candidate=candidate, epoch30_control=control,
        elapsed_seconds=time.monotonic() - start, maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out / 'training.json', meta)
    print(json.dumps({key: meta[key] for key in (
        'status', 'fit_rows', 'query_rows', 'fit_seconds', 'elapsed_seconds',
        'saved_model_reload_exact', 'saved_epoch30_model_reload_exact')}), flush=True)


def cpu_unit_check():
    """Tiny deterministic parameter algebra only; no model, optimizer, fit or data."""
    from unittest.mock import patch
    check_configuration()
    before_config = copy.deepcopy(baseline.CONFIG)
    before_rng = torch.get_rng_state().clone()
    failures = []

    def prohibited(*args, **kwargs):
        raise AssertionError('A CPU parameter-algebra check must not draw RNG or create an optimizer')

    with patch.object(torch, 'rand', prohibited), patch.object(torch, 'randn', prohibited), \
            patch.object(torch, 'randperm', prohibited), patch.object(torch, 'manual_seed', prohibited), \
            patch.object(torch.optim, 'AdamW', prohibited):
        parent_check = baseline.cpu_unit_check()
        states = [{
            'linear.weight': torch.tensor([[float(i), float(i + 2)]], dtype=torch.float32),
            'linear.bias': torch.tensor([float(2 * i)], dtype=torch.float32),
        } for i in range(AVERAGE_COUNT)]
        average = TailParameterAverage()
        for epoch, state in zip(AVERAGE_EPOCHS, states):
            before = {name: value.clone() for name, value in state.items()}
            average.add(epoch, state)
            assert all(torch.equal(state[name], before[name]) for name in state)
        result = average.finish()
        assert tuple(result) == ('linear.weight', 'linear.bias')
        assert all(value.device.type == 'cpu' and value.dtype == torch.float32 for value in result.values())
        assert all(value.device.type == 'cpu' and value.dtype == torch.float64 for value in average._sums.values())
        torch.testing.assert_close(result['linear.weight'], torch.tensor([[4.5, 6.5]]), rtol=0, atol=0)
        torch.testing.assert_close(result['linear.bias'], torch.tensor([9.]), rtol=0, atol=0)
        x = torch.tensor([[1., 2.], [-1., 3.]])
        linear_prediction = torch.nn.functional.linear(x, result['linear.weight'], result['linear.bias'])
        torch.testing.assert_close(linear_prediction, torch.tensor([[26.5], [24.]]), rtol=0, atol=0)
        # Nonlinear counterexample: average(parameters) -> prediction differs from
        # first predicting each snapshot and then averaging those predictions.
        nonlinear = TailParameterAverage()
        individual_predictions = []
        for epoch, i in zip(AVERAGE_EPOCHS, range(AVERAGE_COUNT)):
            state = {'weight': torch.tensor([float(i - 4)], dtype=torch.float32)}
            nonlinear.add(epoch, state)
            individual_predictions.append(torch.nn.functional.softplus(state['weight']))
        averaged_parameter_prediction = torch.nn.functional.softplus(nonlinear.finish()['weight'])
        averaged_predictions = torch.stack(individual_predictions).mean(dim=0)
        assert abs(float(averaged_parameter_prediction - averaged_predictions)) > .1
        # Snapshot storage must not alias later mutations of the live model.
        detached = TailParameterAverage()
        for epoch, i in zip(AVERAGE_EPOCHS, range(AVERAGE_COUNT)):
            state = {'weight': torch.tensor([float(i)], dtype=torch.float32, requires_grad=True)}
            detached.add(epoch, state)
            with torch.no_grad():
                state['weight'].fill_(999.)
            assert state['weight'].grad is None
        torch.testing.assert_close(detached.finish()['weight'], torch.tensor([4.5]), rtol=0, atol=0)
        # Failed additions must leave the accumulator and epoch count unchanged.
        bad_states = {
            'shape': {'linear.weight': torch.zeros(2, 2), 'linear.bias': torch.zeros(1)},
            'dtype': {'linear.weight': torch.zeros(1, 2, dtype=torch.float64), 'linear.bias': torch.zeros(1)},
            'nan': {'linear.weight': torch.tensor([[float('nan'), 0.]]), 'linear.bias': torch.zeros(1)},
            'inf': {'linear.weight': torch.zeros(1, 2), 'linear.bias': torch.tensor([float('inf')])},
            'names': {'unexpected.weight': torch.zeros(1, 2), 'linear.bias': torch.zeros(1)},
            'order': {'linear.bias': torch.zeros(1), 'linear.weight': torch.zeros(1, 2)},
        }
        for name, state in bad_states.items():
            checker = TailParameterAverage()
            checker.add(21, states[0])
            old_sums = {key: value.clone() for key, value in checker._sums.items()}
            try:
                checker.add(22, state)
            except ValueError:
                failures.append(name)
            else:
                raise AssertionError('Invalid parameter state accepted: ' + name)
            assert checker.epochs == [21]
            assert all(torch.equal(checker._sums[key], value) for key, value in old_sums.items())
        for count in (0, 9):
            checker = TailParameterAverage()
            for epoch, state in zip(AVERAGE_EPOCHS[:count], states[:count]):
                checker.add(epoch, state)
            try:
                checker.finish()
            except ValueError:
                failures.append('count_' + str(count))
            else:
                raise AssertionError('An incomplete parameter mean was accepted')
        for name, checker, epoch in (
                ('wrong_epoch', TailParameterAverage(), 22), ('eleventh_state', average, 31)):
            try:
                checker.add(epoch, states[0])
            except ValueError:
                failures.append(name)
            else:
                raise AssertionError('Invalid averaging epoch accepted')
        assert torch.equal(before_rng, torch.get_rng_state())
        assert CONFIG == baseline.CONFIG == before_config
    return {
        'status': 'CPU_fixed_tail_parameter_average_passed', 'epochs': list(AVERAGE_EPOCHS),
        'equal_ten_state_mean_checked': True, 'CPU_float64_accumulation_float32_output_checked': True,
        'linear_hand_calculation_checked': True, 'nonlinear_prediction_average_counterexample_checked': True,
        'input_states_unmodified': True, 'detached_nonaliasing_snapshots_checked': True,
        'invalid_state_rejections': failures, 'failed_additions_leave_accumulator_unchanged': True,
        'CPU_RNG_state_unchanged': True, 'optimizer_created': False, 'optimizer_steps': 0,
        'complete_R63_training_configuration_unchanged': True, 'parent_weighted_MAE_unit_check': parent_check,
        'model_fits': 0, 'GPU_calls': 0, 'real_data_read': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cpu-unit-check', action='store_true')
    args = parser.parse_args()
    if args.cpu_unit_check:
        print(json.dumps(cpu_unit_check()), flush=True)
    else:
        if args.data is None or args.out is None:
            parser.error('--data and --out are required')
        run(args.data, args.out)
