"""R104 matched-scale native/base-objective prior NN; no query labels.

The immutable R63 architecture, optimizer schedule, positive ratio prediction,
and inference are reused. This is a new objective, not an unchanged R63 loss.
The driver owns temporal teacher/factor provenance and canonical row alignment.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
import round63_recency_weighted_prior_train as baseline

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'reports/round104_plan.md'
PLAN_SHA = '106fda0c56cc7c5660c6255bb572fd78c59ccac9c13d25604bbe592204e43778'
R63 = 'src/round63_recency_weighted_prior_train.py'
R63_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
ARRAY_NAMES = ('train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat',
               'cardinalities', 'train_price_factor_million', 'query_price_factor_million',
               'train_sample_weight', 'train_base_offset_million')
COEFFICIENT = .375
LOSS_FORMULAS = {
    'C': '.375 * mean(train_sample_weight * abs(ratio_output * train_price_factor_million - train_y_million))',
    'T': 'mean(train_sample_weight * abs(.375 * ratio_output * train_price_factor_million + train_base_offset_million - train_y_million))',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def identity(a):
    return {'dtype': str(a.dtype), 'shape': list(a.shape),
            'values_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def source_hashes():
    names = [Path(__file__).resolve().relative_to(ROOT).as_posix(), R63,
             'src/round52_nn_train.py', 'src/round47_nn_train.py']
    return {p: sha(ROOT/p) for p in names}


def parameters():
    return {'round': 104, 'objectives': list(LOSS_FORMULAS), 'loss_formulas': LOSS_FORMULAS.copy(),
            'input_keys': list(ARRAY_NAMES), 'numeric_count': 145, 'categorical_count': 23,
            'native_coefficient': COEFFICIENT, 'offset_used_in_query': False,
            'offset_used_in_C_loss': False, 'seed': 42, 'epochs': 30, 'batch_size': 2048,
            'query_batch_size': 4096, 'sample_weights_renormalized': False,
            'residual_target_clipping': False, 'query_labels_available': False}


def configuration(objective):
    need(objective in LOSS_FORMULAS, 'Objective must be C or T')
    result = copy.deepcopy(baseline.CONFIG)
    result.update(loss='R104 matched-scale native weighted L1' if objective == 'C' else 'R104 fixed-base weighted L1',
        loss_formula=LOSS_FORMULAS[objective],
        train_price_factor_construction='driver-supplied matched subset of saved R102T temporal fit prior factors',
        sample_weight_normalization='saved outer-fit date weights; no subset or batch renormalization')
    return result


def check_configuration():
    need(sha(ROOT/PLAN) == PLAN_SHA, 'R104 fixed plan changed')
    need(sha(ROOT/R63) == R63_SHA, 'Immutable R63 source changed')
    baseline.check_configuration()
    need(sha(Path(baseline.prior.__file__)) == baseline.ARCHITECTURE_SHA, 'R47 architecture changed')
    for objective in LOSS_FORMULAS:
        cfg = configuration(objective)
        changed = {k for k in cfg if cfg[k] != baseline.CONFIG[k]}
        need(changed == {'loss', 'loss_formula', 'train_price_factor_construction', 'sample_weight_normalization'},
             'Unexpected R104 configuration changes')
        need((cfg['seed'], cfg['epochs'], cfg['batch_size']) == (42, 30, 2048), 'Fixed schedule changed')


def positive_vector(a, n, name):
    need(isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == (n,)
         and np.isfinite(a).all() and (a > 0).all(), name + ' must be positive finite aligned float32')
    return a


def materialize_archive(archive):
    # Reject extra/query-label members before accessing any array payload.
    need(len(archive.files) == len(ARRAY_NAMES) and set(archive.files) == set(ARRAY_NAMES),
         'Expected exactly ten arrays, no query labels')
    return {name: archive[name] for name in ARRAY_NAMES}


def validate_inputs(a):
    need(set(a) == set(ARRAY_NAMES), 'Exactly ten input arrays required')
    xn, xc, y, qn, qc, card = (a[k] for k in ARRAY_NAMES[:6])
    need(all(isinstance(v, np.ndarray) for v in a.values()), 'NumPy arrays required')
    need(xn.ndim == qn.ndim == xc.ndim == qc.ndim == 2 and len(xn) > 0 and len(qn) > 0, 'Input dimensions/empty data')
    n, q = len(xn), len(qn)
    need(xn.dtype == qn.dtype == np.float32 and xn.shape == (n, 145) and qn.shape == (q, 145)
         and np.isfinite(xn).all() and np.isfinite(qn).all(), 'Exactly 145 finite float32 numeric columns')
    need(card.dtype == np.int64 and card.shape == (23,) and (card >= 1).all(), '23 positive int64 cardinalities')
    need(xc.dtype == qc.dtype == np.int64 and xc.shape == (n, 23) and qc.shape == (q, 23), 'Categorical shape/dtype')
    for i, size in enumerate(card):
        need(((xc[:, i] >= 0) & (xc[:, i] < size)).all() and ((qc[:, i] >= 0) & (qc[:, i] < size)).all(), 'Categorical value outside cardinality')
    for name in ('train_y_million', 'train_price_factor_million', 'train_sample_weight', 'train_base_offset_million'):
        positive_vector(a[name], n, name)
    positive_vector(a['query_price_factor_million'], q, 'query_price_factor_million')
    return n, q


def initial_bias(y, factor):
    median = float(np.median(y.astype(np.float64)/factor.astype(np.float64)))
    need(np.isfinite(median) and median > 0, 'Invalid same-subset ratio median')
    value = median + float(np.log(-np.expm1(-median)))
    need(np.isfinite(np.float32(value)), 'Nonfinite initial bias')
    return median, value


def objective_loss(p, y, weights, offset, objective):
    need(objective in LOSS_FORMULAS, 'Objective must be C or T')
    need(p.ndim == 1 and p.numel() > 0 and y.shape == weights.shape == offset.shape == p.shape,
         'Loss requires aligned nonempty one-dimensional tensors')
    need(all(v.dtype == p.dtype and v.device == p.device for v in (y, weights, offset))
         and p.dtype in (torch.float32, torch.float64), 'Loss dtype/device mismatch')
    if objective == 'C':
        return COEFFICIENT * (weights * (p-y).abs()).mean()
    return (weights * (COEFFICIENT*p + offset-y).abs()).mean()


def run(data_path, out, objective):
    start = time.monotonic(); check_configuration(); cfg = configuration(objective)
    sources = source_hashes(); input_sha = sha(data_path)
    need(not out.exists() or not any(out.iterdir()), 'Preserve existing output/failure; no retry')
    out.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = materialize_archive(archive)
    n, q = validate_inputs(arrays); identities = {k: identity(v) for k, v in arrays.items()}
    xn, xc, y, qn, qc, card, af, qaf, sw, offset = (arrays[k] for k in ARRAY_NAMES)
    card = [int(v) for v in card]; median, bias = initial_bias(y, af)
    device = baseline.prior.setup()
    meta = {'round': 104, 'objective': objective, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': cfg, 'parameters': parameters(), 'source_sha256': sha(__file__), 'source_hashes': sources,
        'plan_sha256': PLAN_SHA, 'parent_trainer_source': R63, 'parent_trainer_source_sha256': R63_SHA,
        'architecture_source_sha256': baseline.ARCHITECTURE_SHA, 'original_architecture_and_optimizer_schedule': True,
        'loss_equals_R63': False, 'changed_configuration_fields': sorted(k for k in cfg if cfg[k] != baseline.CONFIG[k]),
        'input_sha256': input_sha, 'input_array_identities': identities, 'fit_rows': n, 'query_rows': q,
        'numeric_count': 145, 'cardinalities': card, 'initial_training_median_ratio': median, 'initial_head_bias': bias,
        'initial_bias_uses_offset': False, 'training_objective_name': cfg['loss'],
        'query_labels_available': False, 'query_labels_used': False, 'query_offset_used': False,
        'C_offset_used_in_loss': False, 'offset_provenance_verified_in_trainer': False,
        'price_factor_provenance_verified_in_trainer': False, 'sample_weight_provenance_verified_in_trainer': False,
        'canonical_row_alignment_verified_in_trainer': False,
        'train_price_factor_construction': cfg['train_price_factor_construction'],
        'query_price_factor_construction': baseline.QUERY_FACTOR_CONSTRUCTION,
        'train_base_offset_formula': 'float32((.375*A_OOT_TWD+.25*L_OOT_TWD)/1e6)',
        'train_base_offset_million_sha256': identities['train_base_offset_million']['values_sha256'],
        'train_price_factor_million_sha256': identities['train_price_factor_million']['values_sha256'],
        'query_price_factor_million_sha256': identities['query_price_factor_million']['values_sha256'],
        'train_sample_weight_sha256': identities['train_sample_weight']['values_sha256'],
        'train_sample_weight_summary': baseline.sample_weight_summary(sw),
        'sample_weights_recomputed_in_trainer': False, 'sample_weights_subset_renormalized': False,
        'sample_weights_batch_renormalized': False, 'price_or_ratio_clipping': False, 'residual_target_clipping': False,
        'negative_equivalent_residual_target_rows': int(np.count_nonzero(y.astype(np.float64)-offset.astype(np.float64) < 0)),
        'prediction_units': 'TWD', 'output_units': baseline.OUTPUT_UNITS,
        'loss_precision': 'float32 Torch; offset saved float32',
        'query_restore_precision': 'ratio float64 * saved factor float64 * 1e6; no offset',
        'epoch_selection_uses_training_metrics': False, 'query_early_stopping': False, 'automatic_retry': False,
        'bitwise_retraining_determinism_claimed': False, 'torch': torch.__version__, 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    baseline.write(out/'run.json', meta)
    model = baseline.EmbeddingMLP(145, card).to(device)
    with torch.no_grad(): model.head.bias.fill_(bias)
    tn = torch.as_tensor(xn, device=device); tc = torch.as_tensor(xc, device=device); ty = torch.as_tensor(y, device=device)
    ta = torch.as_tensor(af, device=device); tw = torch.as_tensor(sw, device=device); to = torch.as_tensor(offset, device=device)
    qtn = torch.as_tensor(qn, device=device); qtc = torch.as_tensor(qc, device=device)
    del xn, xc, y, qn, qc, af, sw, offset, arrays
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['minimum_learning_rate'])
    epochs = []; fit_started = time.monotonic()
    for epoch in range(1, cfg['epochs']+1):
        model.train(); order = torch.randperm(len(tn), device=device)
        total = torch.zeros((), device=device); native_mae = torch.zeros((), device=device)
        lr = optimizer.param_groups[0]['lr']
        for first in range(0, len(tn), cfg['batch_size']):
            index = order[first:first+cfg['batch_size']]; optimizer.zero_grad(set_to_none=True)
            prediction = model(tn[index], tc[index])*ta[index]
            loss = objective_loss(prediction, ty[index], tw[index], to[index], objective)
            mae = (prediction.detach()-ty[index]).abs().mean()
            loss.backward(); optimizer.step(); total += loss.detach()*len(index); native_mae += mae*len(index)
        value = float((total/len(tn)).item()); mae_value = float((native_mae/len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):
            raise FloatingPointError('Nonfinite objective/native MAE; no clipping or retry')
        epochs.append({'epoch': epoch, 'learning_rate': lr, 'objective': objective,
            'training_objective_million_with_dropout': value, 'training_native_mae_million_with_dropout': mae_value})
        schedule.step()
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'event': 'epoch_complete', **epochs[-1], 'elapsed_seconds': time.monotonic()-fit_started}), flush=True)
    torch.cuda.synchronize(); fit_seconds = time.monotonic()-fit_started
    model_path = out/'model.pt'
    torch.save({'numeric_count': 145, 'cardinalities': card, 'configuration': cfg, 'objective': objective,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}, model_path)
    predictions = baseline.predict(model, qtn, qtc, qaf)
    loaded = torch.load(model_path, map_location='cpu', weights_only=True)
    need(loaded['configuration'] == cfg and loaded['objective'] == objective and loaded['cardinalities'] == card
         and loaded['numeric_count'] == 145, 'Model reload metadata differs')
    restored = baseline.EmbeddingMLP(145, card).to(device); restored.load_state_dict(loaded['state_dict'])
    reloaded = baseline.predict(restored, qtn, qtc, qaf)
    need(predictions.tobytes() == reloaded.tobytes(), 'Saved model full-query TWD reload differs')
    need(sha(data_path) == input_sha and source_hashes() == sources and sha(ROOT/PLAN) == PLAN_SHA, 'Input/source/plan changed during training')
    with (out/'predictions_twd.npy').open('xb') as f: np.save(f, predictions, allow_pickle=False)
    meta.update(status='completed', epochs=epochs, fit_seconds=fit_seconds,
        parameter_count=sum(p.numel() for p in model.parameters()), saved_model_reload_exact=True,
        reload_rows=len(predictions), reload_batch_size=4096, query_offset_used=False,
        prediction_min_twd=float(predictions.min()), prediction_max_twd=float(predictions.max()),
        model_sha256=sha(model_path), predictions_sha256=sha(out/'predictions_twd.npy'),
        elapsed_seconds=time.monotonic()-start, maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    baseline.write(out/'training.json', meta)
    print(json.dumps({k: meta[k] for k in ('status', 'objective', 'fit_rows', 'query_rows', 'fit_seconds', 'saved_model_reload_exact')}), flush=True)


def cpu_self_check():
    tick = time.monotonic(); check_configuration(); before = copy.deepcopy(baseline.CONFIG); checks = []
    def mark(name, ok): need(bool(ok), name); checks.append(name)
    def rejects(name, fn):
        try: fn()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    p = torch.tensor([2., 4., 8., 10.], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([3., 3., 10., 2.], dtype=torch.float64)
    w = torch.tensor([.5, 2., 1., .5], dtype=torch.float64)
    off = torch.tensor([.75, 1.5, 2., 5.], dtype=torch.float64)
    lc = objective_loss(p, y, w, off, 'C'); lc.backward()
    mark('C_hand_loss', abs(lc.item()-.796875) < 1e-15)
    mark('C_hand_gradient_matched_scale', torch.equal(p.grad, torch.tensor([-.046875, .1875, -.09375, .046875], dtype=torch.float64)))
    p.grad = None; lt = objective_loss(p, y, w, off, 'T'); lt.backward()
    mark('T_hand_loss_negative_equivalent_target', abs(lt.item()-2.28125) < 1e-15 and (y[-1]-off[-1]).item() < 0)
    mark('T_hand_gradient_including_exact_zero', torch.equal(p.grad, torch.tensor([-.046875, 0., -.09375, .046875], dtype=torch.float64)))
    mark('C_ignores_positive_offset', torch.equal(lc.detach(), objective_loss(p, y, w, off*8, 'C').detach()))
    shifted = objective_loss(p, y, w, off*8, 'T')
    mark('T_offset_affects_objective', shifted.item() != lt.item())
    for objective in ('C', 'T'):
        full = objective_loss(p, y, w, off, objective)
        chunks = (objective_loss(p[:1], y[:1], w[:1], off[:1], objective)+3*objective_loss(p[1:], y[1:], w[1:], off[1:], objective))/4
        mark(objective+'_tail_batch_mean_no_renormalization', torch.equal(full, chunks))
        pp = p.detach().float().requires_grad_(True)
        z = objective_loss(pp, y.float(), w.float(), off.float(), objective); z.backward()
        mark(objective+'_float32_finite_loss_gradient', torch.isfinite(z).item() and torch.isfinite(pp.grad).all().item())
    rejects('loss_shape_rejected', lambda: objective_loss(p, y[:, None], w, off, 'T'))
    rejects('loss_dtype_rejected', lambda: objective_loss(p, y.float(), w, off, 'T'))
    rejects('loss_objective_rejected', lambda: objective_loss(p, y, w, off, 'X'))
    rejects('empty_loss_rejected', lambda: objective_loss(p[:0], y[:0], w[:0], off[:0], 'C'))
    a = {'train_num': np.zeros((4,145),np.float32), 'query_num': np.zeros((3,145),np.float32),
         'train_cat': np.zeros((4,23),np.int64), 'query_cat': np.zeros((3,23),np.int64),
         'cardinalities': np.full(23,2,np.int64), 'train_y_million': y.numpy().astype(np.float32),
         'train_price_factor_million': np.ones(4,np.float32), 'query_price_factor_million': np.ones(3,np.float32),
         'train_sample_weight': w.numpy().astype(np.float32), 'train_base_offset_million': off.numpy().astype(np.float32)}
    ids = {k:identity(v) for k,v in a.items()}; mark('ten_array_contract', validate_inputs(a)==(4,3))
    class Archive:
        def __init__(self, names): self.files=names
        def __getitem__(self, key): raise AssertionError('Payload was touched before invalid key rejection')
    rejects('query_label_key_rejected_before_materializing', lambda: materialize_archive(Archive([*ARRAY_NAMES,'query_y'])))
    rejects('missing_offset_key_rejected_before_materializing', lambda: materialize_archive(Archive(list(ARRAY_NAMES[:-1]))))
    for name in ('train_y_million','train_price_factor_million','query_price_factor_million','train_sample_weight','train_base_offset_million'):
        for bad in (0., -1., np.nan, np.inf):
            b={k:v.copy() for k,v in a.items()}; b[name][0]=bad
            rejects(name+'_reject_'+str(bad), lambda b=b:validate_inputs(b))
    for mutation, val in [('shape', np.ones((4,1),np.float32)), ('dtype', np.ones(4,np.float64)), ('length', np.ones(3,np.float32))]:
        b={**a,'train_base_offset_million':val}; rejects('offset_'+mutation+'_rejected',lambda b=b:validate_inputs(b))
    b={**a,'query_cat':a['query_cat'].copy()}; b['query_cat'][0,0]=2
    rejects('category_range_rejected',lambda:validate_inputs(b))
    b={**a,'train_num':a['train_num'][:,:144]};rejects('numeric_width_rejected',lambda:validate_inputs(b))
    med,bias=initial_bias(a['train_y_million'],a['train_price_factor_million'])
    mark('same_subset_bias_independent_of_offset', med==3. and np.isfinite(bias))
    mark('input_arrays_unchanged',ids=={k:identity(v) for k,v in a.items()})
    # Tiny CPU evaluation/save-reload only, no optimizer or real data.
    model=baseline.EmbeddingMLP(145,[2]*23); model.eval()
    with torch.no_grad(): model.head.bias.fill_(bias)
    numeric=torch.from_numpy(a['query_num']); categorical=torch.from_numpy(a['query_cat'])
    pred=baseline.predict(model,numeric,categorical,a['query_price_factor_million'])
    buffer=io.BytesIO(); torch.save({'state_dict':model.state_dict()},buffer); buffer.seek(0)
    restored=baseline.EmbeddingMLP(145,[2]*23);restored.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=True)['state_dict'])
    reload=baseline.predict(restored,numeric,categorical,a['query_price_factor_million'])
    mark('CPU_saved_state_full_query_TWD_exact',pred.tobytes()==reload.tobytes() and np.isfinite(pred).all() and (pred>0).all())
    ratio=np.array([.25,1.,2.],np.float64);f=np.array([2.,3.,4.],np.float32)
    mark('original_to_TWD_float64_no_offset',np.array_equal(baseline.to_twd(ratio,f),ratio*f.astype(np.float64)*1e6))
    mark('immutable_R63_configuration_unchanged',baseline.CONFIG==before)
    return {'status':'passed','exit_code':0,'check_count':len(checks),'checks':checks,
        'source_sha256':sha(__file__),'source_hashes':source_hashes(),'plan_sha256':PLAN_SHA,
        'parameters':parameters(),'configurations':{o:configuration(o) for o in LOSS_FORMULAS},
        'synthetic_only':True,'real_data_read':False,'true_fit_calls':0,'GPU_calls':0,
        'synthetic_CPU_model_eval_and_save_reload':True,'elapsed_seconds':time.monotonic()-tick}


def write_receipt(path, result):
    path=Path(path);side=path.with_suffix('.sha256')
    need(not path.exists() and not side.exists(),'Preserve existing self-check receipt')
    baseline.write(path,result)
    with side.open('x',encoding='ascii') as f:f.write(sha(path)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--objective',choices=['C','T']);parser.add_argument('--cpu-self-check',action='store_true')
    parser.add_argument('--receipt',type=Path,default=ROOT/'reports/round104_trainer_selfcheck.json')
    args=parser.parse_args()
    if args.cpu_self_check:
        try: result=cpu_self_check()
        except Exception:
            write_receipt(args.receipt,{'status':'failed','exit_code':1,'failure':traceback.format_exc(),
                'source_sha256':sha(__file__),'source_hashes':source_hashes(),'real_data_read':False,'synthetic_only':True});raise
        write_receipt(args.receipt,result)
        print(json.dumps({'status':result['status'],'checks':result['check_count'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    else:
        if args.data is None or args.out is None or args.objective is None:parser.error('--data, --out and --objective required')
        run(args.data,args.out,args.objective)
