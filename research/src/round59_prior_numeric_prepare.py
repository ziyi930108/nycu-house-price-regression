"""R59 immutable R52 prior factors + saved R54 numerical ramps; CPU preparation only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np

NUMERIC = ('train_num', 'query_num')
SHARED = ('train_cat', 'train_y_million', 'query_cat', 'cardinalities')
FACTORS = ('train_price_factor_million', 'query_price_factor_million')
AREA = ('train_area_factor', 'query_area_factor')
PRIOR_KEYS = set(NUMERIC + SHARED + FACTORS)
RAMP_KEYS = set(NUMERIC + SHARED + AREA)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def exact(a, b, name):
    need(a.dtype == b.dtype and a.shape == b.shape and
         np.array_equal(a.view(np.uint8), b.view(np.uint8)), name + ' differs bitwise')


def finite32(a, shape, name, positive=False):
    need(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), name + ' dtype/shape/finite')
    if positive:
        need((a > 0).all(), name + ' must be positive')


def check_state(state, rows):
    need(state['version'] == 1 and state['input_width'] == 144 and state['numeric_columns'] == 72
         and state['fit_rows'] == rows and state['fit_uses_query'] is False and state['fit_uses_target'] is False
         and state['original_144_preserved'] is True and state['output_dtype'] == 'float32'
         and state['quantile_method'] == 'linear' and state['quantiles'] == np.linspace(0, 1, 17).tolist()
         and state['ramp_standardization'] is False and len(state['columns']) == 72, 'R54 frozen state contract')
    offset = 144
    for column, entry in enumerate(state['columns']):
        knots = np.asarray(entry['knots'], dtype=np.float64)
        need(knots.ndim == 1 and 1 <= len(knots) <= 17 and np.isfinite(knots).all()
             and (np.diff(knots) > 0).all() and entry['column'] == column
             and entry['offset'] == offset and entry['width'] == len(knots) - 1, 'R54 knots/offsets')
        offset += entry['width']
    need(state['output_width'] == offset and state['ramp_width'] == offset - 144, 'R54 output width')
    return offset


def prepare(folder, prior_path, ramp_path, numeric_state_path, expected_fit_y_million=None):
    """Return (prepared.npz Path, evidence); no fitting, transformation or query labels.

    folder must already exist. Optional expected fit targets are float32/float64
    million TWD; compare after the established float32 training-target conversion.
    The driver owns canonical indices and trusted artifact identities.
    """
    tick = time.monotonic()
    folder = Path(folder)
    need(folder.is_dir(), 'Caller must create the output folder')
    output, proof = folder / 'prepared.npz', folder / 'preparation.json'
    if output.exists() or proof.exists():
        raise FileExistsError('Preserve existing prepared data/evidence; no overwrite')
    paths = {k: Path(v) for k, v in [('prior_prepared', prior_path), ('ramp_prepared', ramp_path),
                                    ('numeric_state', numeric_state_path)]}
    inputs = {k: sha(v) for k, v in paths.items()}
    state = json.loads(paths['numeric_state'].read_text(encoding='utf-8'))
    with np.load(paths['prior_prepared'], allow_pickle=False) as z:
        need(len(z.files) == 8 and set(z.files) == PRIOR_KEYS, 'R52 exactly eight keys')
        arrays = {k: z[k] for k in z.files}
    n, q = len(arrays['train_num']), len(arrays['query_num'])
    need(n > 0 and q > 0, 'Nonempty fit/query required')
    width = check_state(state, n)
    for key, rows in zip(NUMERIC, (n, q)):
        finite32(arrays[key], (rows, 144), key)
    finite32(arrays['train_y_million'], (n,), 'fit target', positive=True)
    for key, rows in zip(FACTORS, (n, q)):
        finite32(arrays[key], (rows,), key, positive=True)
    cards = arrays['cardinalities']
    need(cards.dtype == np.int64 and cards.shape == (23,) and (cards >= 1).all(), 'Cardinality contract')
    for key, rows in [('train_cat', n), ('query_cat', q)]:
        a = arrays[key]
        need(a.dtype == np.int64 and a.shape == (rows, 23) and (a >= 0).all()
             and (a < cards[None, :]).all(), key + ' dtype/shape/range')
    if expected_fit_y_million is not None:
        expected = np.asarray(expected_fit_y_million)
        need(expected.dtype in (np.dtype('float32'), np.dtype('float64')) and expected.shape == (n,)
             and np.isfinite(expected).all(), 'Expected fit targets must be finite float32/64 million TWD')
        exact(arrays['train_y_million'], expected.astype(np.float32), 'Expected fit target order')
    unchanged = {k: array_sha(arrays[k]) for k in SHARED + FACTORS}
    with np.load(paths['ramp_prepared'], allow_pickle=False) as z:
        need(len(z.files) == 8 and set(z.files) == RAMP_KEYS, 'R54 exactly eight area keys')
        for key in SHARED:
            exact(arrays[key], z[key], 'Shared ' + key)
        for key, rows in zip(AREA, (n, q)):
            finite32(z[key], (rows,), key, positive=True)
        for key, rows in zip(NUMERIC, (n, q)):
            ramp = z[key]
            finite32(ramp, (rows, width), 'R54 ' + key)
            exact(arrays[key], ramp[:, :144], 'Original 144 ' + key)
            need(((ramp[:, 144:] >= 0) & (ramp[:, 144:] <= 1)).all(), 'Saved ramps outside [0,1]')
            arrays[key] = ramp
    need(all(array_sha(arrays[k]) == digest for k, digest in unchanged.items()), 'Preserved six arrays changed')
    with output.open('xb') as f:
        np.savez(f, **arrays)
    evidence = {'input_sha256': inputs, 'input_paths': {k: p.resolve().as_posix() for k, p in paths.items()},
        'source_sha256': {'src/round59_prior_numeric_prepare.py': sha(__file__)},
        'unchanged_six_array_sha256': unchanged, 'original_numeric_prefix_bitwise_equal': True,
        'shared_categories_cardinalities_target_bitwise_equal': True, 'fit_rows': n, 'query_rows': q,
        'numeric_count': width, 'query_labels_used': False, 'encoder_refit': False, 'prior_refit': False,
        'numeric_transform_recomputed': False, 'state_fit_only': True, 'numeric_state_sha256': inputs['numeric_state'],
        'price_factor_formula': 'effective_area_A * group_prior_TWD_per_m2 / 1e6',
        'train_price_factor_million_sha256': unchanged[FACTORS[0]],
        'query_price_factor_million_sha256': unchanged[FACTORS[1]],
        'area_factors_copied_to_output': False, 'elapsed_seconds': time.monotonic() - tick}
    with proof.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return output, evidence


def self_check():
    """One small synthetic save/reload and misordered-input rejection; no real data."""
    with tempfile.TemporaryDirectory(prefix='round59_prepare_') as tmp:
        root = Path(tmp)
        x = np.arange(4 * 144, dtype=np.float32).reshape(4, 144) / np.float32(144)
        x[0, 3] = -0.0
        prior = {'train_num': x, 'query_num': x[[3, 1]].copy(), 'train_cat': np.zeros((4, 23), dtype=np.int64),
                 'query_cat': np.zeros((2, 23), dtype=np.int64), 'cardinalities': np.ones(23, dtype=np.int64),
                 'train_y_million': np.arange(1, 5, dtype=np.float32),
                 FACTORS[0]: np.arange(2, 6, dtype=np.float32), FACTORS[1]: np.array([9, 7], dtype=np.float32)}
        columns = [{'column': j, 'knots': [0., 1., 2.] if j == 0 else [0.],
                    'offset': 144 if j == 0 else 146, 'width': 2 if j == 0 else 0} for j in range(72)]
        state = {'version': 1, 'input_width': 144, 'numeric_columns': 72, 'fit_rows': 4, 'fit_uses_query': False,
                 'fit_uses_target': False, 'original_144_preserved': True, 'output_dtype': 'float32',
                 'quantile_method': 'linear', 'quantiles': np.linspace(0, 1, 17).tolist(),
                 'ramp_standardization': False, 'columns': columns, 'output_width': 146, 'ramp_width': 2}
        ramp = {k: prior[k].copy() for k in SHARED}
        for key in NUMERIC:
            a = prior[key]
            ramp[key] = np.column_stack((a, np.clip(a[:, 0], 0, 1), np.clip(a[:, 0] - 1, 0, 1))).astype(np.float32)
        ramp.update(train_area_factor=np.ones(4, dtype=np.float32), query_area_factor=np.ones(2, dtype=np.float32))
        np.savez(root / 'prior.npz', **prior); np.savez(root / 'ramp.npz', **ramp)
        (root / 'state.json').write_text(json.dumps(state), encoding='utf-8')
        out = root / 'out'; out.mkdir()
        path, proof = prepare(out, root / 'prior.npz', root / 'ramp.npz', root / 'state.json', prior['train_y_million'].astype(np.float64))
        with np.load(path, allow_pickle=False) as saved:
            need(set(saved.files) == PRIOR_KEYS and saved['train_num'].shape == (4, 146), 'Output keys/width')
            for key in PRIOR_KEYS:
                exact(saved[key], ramp[key] if key in NUMERIC else prior[key], 'Saved ' + key)
        need(json.loads((out / 'preparation.json').read_text(encoding='utf-8')) == proof, 'Evidence JSON reload')
        ramp['query_num'] = ramp['query_num'][::-1].copy()
        np.savez(root / 'misordered.npz', **ramp)
        bad = root / 'bad'; bad.mkdir()
        try:
            prepare(bad, root / 'prior.npz', root / 'misordered.npz', root / 'state.json')
        except ValueError as e:
            need('Original 144 query_num differs bitwise' in str(e), 'Wrong rejection reason')
        else:
            raise AssertionError('Misordered rows accepted')
        need(not list(bad.iterdir()), 'Rejected preparation wrote outputs')
    return {'status': 'passed', 'numeric_count': 146, 'saved_arrays_bitwise_equal': True,
            'price_factors_preserved': True, 'row_misalignment_rejected': True, 'real_data_read': False, 'GPU_calls': 0}


if __name__ == '__main__':
    print(json.dumps(self_check(), ensure_ascii=False))
