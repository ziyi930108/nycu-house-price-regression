"""R100 file contracts and constants; no data access or fitting on import."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'reports/round100_plan.md'
PLAN_SHA = 'bbb29bcc4fb83eaa3fdac9f15ade8221ced3437abacb04e891038ffb3462e107'
STAGES = ('development', 'confirmation', 'auxiliary')
SPLITS = {'development': ('2022-08-11', '2023-08-10', 240316, 81234),
          'confirmation': ('2023-08-11', '2024-08-10', 321550, 133697),
          'auxiliary': ('2021-08-11', '2022-08-10', 134805, 105511)}
WEIGHTS = {'seen': .36617593949822613, 'new_nonmissing': .6338240605017739, 'missing': 0.}
FEATURES = 'data_cache/train_features.pkl'
CANONICAL = 'outputs/round6/geography/development/train_source_cache_indices.npy'
RAW = 'data_cache/train.pkl'
X_AUDIT = 'outputs/round100_x_audit/results.json'
FIT_LABELS = 'outputs/round100_inputs/manifest.json'
TRAINER = 'src/round63_recency_weighted_prior_train.py'
TRAINER_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
ARRAY_NAMES = ('train_num', 'train_cat', 'train_y_million', 'query_num', 'query_cat',
               'cardinalities', 'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight')
CAUTION = ('Adaptive research on reused annual windows; matched training rows do not remove single-seed '
           'variation. Bootstrap is conditional on saved predictions, not Private probability. '
           'Only fit-label priors are chronological; encoder and query prior retain full outer-fit scope.')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(name):
    p = Path(name)
    p = ((ROOT / p) if not p.is_absolute() else p).resolve()
    need(p.is_relative_to(ROOT), 'Path outside workspace')
    return p


def rel(name):
    return safe(name).relative_to(ROOT).as_posix()


def sha(name):
    h = hashlib.sha256()
    with safe(name).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads(safe(name).read_text(encoding='utf-8-sig'))


def write(name, value):
    p = safe(name)
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(p) + '\n')


def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def identity(a):
    return {'dtype': str(a.dtype), 'shape': list(a.shape), 'values_sha256': values_sha(a)}


def record(path):
    return {'path': rel(path), 'sha256': sha(path)}


def exact(a, b, name):
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name)


def save_array(path, a):
    p = safe(path)
    need(isinstance(a, np.ndarray) and not a.dtype.hasobject, 'Numeric array required')
    with p.open('xb') as f:
        np.save(f, a, allow_pickle=False)
    exact(a, np.load(p, allow_pickle=False), 'Saved array reload differs')
    return {**record(p), **identity(a)}


def load_array(rec):
    need(sha(rec['path']) == rec['sha256'], 'Array file SHA differs')
    a = np.load(safe(rec['path']), allow_pickle=False)
    need(not a.dtype.hasobject and identity(a) == {k: rec[k] for k in ('dtype', 'shape', 'values_sha256')},
         'Array identity differs')
    return a


def source_hashes(*entries):
    pending, found = [safe(p) for p in entries], set()
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names = [v.name for v in n.names] if isinstance(n, ast.Import) else ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
            for name in names:
                q = ROOT / 'src' / (name.split('.')[0] + '.py')
                if q.is_file() and q not in found:
                    pending.append(q)
    return {rel(p): sha(p) for p in sorted(found)}


def verify_hashes(mapping):
    for name, expected in mapping.items():
        need(sha(name) == expected, 'Immutable file changed: ' + name)


def protected_hashes():
    names = read('outputs/github_publish_r97_20260929/protected_before.json')
    need(len(names) == 9, 'Nine protected files required')
    return {name: sha(name) for name in names}


def evaluation_parameters():
    return {'round': 100, 'stage_order': list(STAGES), 'candidate': 'T',
            'primary_control': 'v22', 'mechanism_control': 'C', 'point_only_control': 'v24',
            'seed': 20260925, 'replicates': 2000, 'stage_indices': {s: i for i, s in enumerate(STAGES)},
            'family_order': ['project', 'month'], 'family_indices': {'project': 0, 'month': 1},
            'month_block_length': 3, 'month_blocks': 4, 'fixed_group_weights': WEIGHTS,
            'large_group_minimum_rows': 500, 'max_adverse_fraction': .05,
            'confirmation_minimum_overall_improvement_fraction': .005,
            'automatic_adoption': False, 'new_CSV': False}
