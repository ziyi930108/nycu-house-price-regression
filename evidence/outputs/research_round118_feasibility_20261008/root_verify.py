"""Independent root dates/metadata audit, with price/prediction reader traps."""
from pathlib import Path
import sys
import importlib.util
import json
import hashlib
import time
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import round100_contract as c

def read(name):
    return json.loads((BASE / name).read_text(encoding='utf-8-sig'))

def record(path):
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'sha256': c.sha(path)}

def main():
    tick = time.monotonic()
    manifest = read('manifest.json')
    before = read('readset.json')['input_sha256']
    c.need(len(before) == 77 and len(manifest['artifacts']) == 17, 'Fixed bounded inventory')
    c.verify_hashes(before); c.verify_hashes(manifest['artifacts'])
    for relative, digest in manifest['artifacts'].items():
        p = ROOT / relative
        sidecar = p.with_suffix('.sha256') if p.suffix == '.json' else Path(str(p) + '.sha256')
        if sidecar.is_file():
            c.need(sidecar.read_text(encoding='ascii').strip() == digest, 'Existing artifact sidecar')
    audit = read('feasibility.json')
    oracle = read('author_synthetic_checks.json')
    captured = []
    spec = importlib.util.spec_from_file_location('root_R118_author_checks', BASE / 'seal_evidence.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class ChecksComplete(Exception): pass
    def capture(name, value):
        c.need(name == 'author_synthetic_checks.json', 'Stop before any author evidence mutation')
        captured.append(value)
        raise ChecksComplete()
    module.write = capture
    def deny(*args, **kwargs):
        raise AssertionError('No real pickle/array read in synthetic checks')
    with patch.object(pd, 'read_pickle', deny), patch.object(np, 'load', deny):
        try: module.run()
        except ChecksComplete: pass
    c.need(len(captured) == 1 and captured[0]['checks'] == oracle['checks']
           and captured[0]['check_count'] == 30, 'All author checks independently reexecuted')

    modes = read('readset.json')['actual_deserialization_or_text_read_modes']
    allowed_arrays = {p for p, mode in modes.items() if mode == 'standalone_target_free_dates_or_indices'}
    c.need(len(allowed_arrays) == 15, 'Only fifteen standalone target-free vectors allowed')
    old_array, old_pickle = np.load, pd.read_pickle
    array_reads = []
    def array(path, **kwargs):
        relative = Path(path).resolve().relative_to(ROOT).as_posix()
        c.need(relative in allowed_arrays and kwargs == {'allow_pickle': False}, 'Array readset restriction')
        array_reads.append(relative)
        return old_array(path, **kwargs)
    def pure_pickle(path, **kwargs):
        c.need(Path(path).resolve() == ROOT / 'data_cache/train_features.pkl' and not kwargs,
               'Only inherited pure-X pickle')
        return old_pickle(path)
    with patch.object(np, 'load', array), patch.object(pd, 'read_pickle', pure_pickle):
        mapping = np.load(ROOT / 'outputs/round6/geography/development/train_source_cache_indices.npy', allow_pickle=False)
        frame = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
        c.need(frame.shape == (455784, 87) and '總價元' not in frame, 'Target-free original X')
        dates = frame.trade_day_index.iloc[mapping].to_numpy(dtype=np.float64, copy=True)
        del frame
        epoch = pd.Timestamp('2000-01-01')
        valid = np.isfinite(dates)
        c.need(int(valid.sum()) == 455247 and len(dates) == 455255, 'Canonical date count')
        def day(text): return int((pd.Timestamp(text) - epoch).days)
        def mask(start, end=None):
            keep = valid & (dates < day(start)) if end is None else valid & (dates >= day(start)) & (dates <= day(end))
            return np.flatnonzero(keep).astype(np.int64)
        pairs = {}
        independent_coverage = {}
        for origin, aggregate in audit['four_origin_forward_meta_coverage'].items():
            fi, qi = mask(aggregate['cutoff']), mask(aggregate['cutoff'], aggregate['query_end_inclusive'])
            c.need((len(fi), len(qi)) == (aggregate['fit_rows'], aggregate['query_rows']), 'Original four full masks')
            pairs[origin] = (fi, qi)
        for origin, aggregate in audit['four_origin_forward_meta_coverage'].items():
            ids = np.concatenate([pairs[b][1][dates[pairs[b][1]] < day(aggregate['cutoff'])]
                                  for b in aggregate['existing_R98_source_order']]).astype(np.int64) if aggregate['existing_R98_source_order'] else np.empty(0, dtype=np.int64)
            c.need(len(ids) == aggregate['existing_forward_meta_rows']
                   and c.values_sha(ids) == aggregate['existing_forward_meta_indices_sha256'], 'Meta full IDs independently recomputed')
            c.need(len(np.unique(ids)) == len(ids) and np.isin(ids, pairs[origin][0]).all()
                   and not np.intersect1d(ids, pairs[origin][1]).size, 'Temporal meta role')
            independent_coverage[origin] = len(ids)
        for overlap in audit['query_window_overlap_counts']:
            c.need(len(np.intersect1d(pairs[overlap['a']][1], pairs[overlap['b']][1])) == overlap['query_shared_canonical_rows'], 'All overlap counts')
        geo = pd.Timestamp('2021-01-01T21:42:03Z')
        expected = {'2021-01-02': (126926, 7879, False), '2021-01-03': (126957, 7848, True), '2021-02-11': (128614, 6191, True)}
        for start, (nfit, nquery, legal) in expected.items():
            fi, qi = mask(start), mask(start, '2021-08-10')
            origin = pd.Timestamp(start).tz_localize('Asia/Taipei').tz_convert('UTC')
            c.need((len(fi), len(qi), geo < origin) == (nfit, nquery, legal), 'Independent dates and UTC legality')
            c.exact(np.sort(np.r_[fi, qi]), pairs['auxiliary'][0], 'New partitions union original auxfit')
        early = read('earliest_boundary_feasibility.json')
        c.need(early['seconds_origin_after_header'] == 65877.0
               and early['prefix_rows'] == 126957 and early['query_rows'] == 7848, 'Early aggregate exact')
    c.verify_hashes(before); c.verify_hashes(manifest['artifacts'])
    result = {'status': 'passed_bounded_independent_root_review', 'created_at_utc': c.now(),
        'review_source': record(Path(__file__)), 'manifest': record(BASE / 'manifest.json'),
        'artifact_sha256': manifest['artifacts'], 'input_sha256': before,
        'input_files_before_after_exact': True, 'author_checks_reexecuted_with_reader_traps': 30,
        'all_four_meta_masks_and_ID_hashes_recomputed': True,
        'all_overlap_counts_recomputed': True, 'three_partial_cutoffs_dates_and_UTC_recomputed': True,
        'meta_rows': independent_coverage, 'array_deserialization_calls': array_reads,
        'feature_pickle_scope': 'All87 pure-X columns transiently materialized; only canonical dates retained.',
        'price_or_prediction_values_read': False, 'models_private_states_or_NPZ_loaded': False,
        'new_fits': 0, 'GPU_work': False, 'new_CSV': False, 'new_MAE_or_bootstrap': False,
        'Public_score': None, 'Private_score': None, 'elapsed_seconds': time.monotonic() - tick,
        'limits': ['Not a full independent binary geography replay or PBF parse.',
                   'Six native plus all learned states not yet rebuilt; this audit selects no model/cutoff.',
                   'Known reused windows and short early meta are not new blind validation.',
                   'Hash-only prediction checks do not inspect their values.']}
    c.write(BASE / 'root_independent_review.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'review': record(BASE / 'root_independent_review.json')}, ensure_ascii=False))

if __name__ == '__main__': main()
