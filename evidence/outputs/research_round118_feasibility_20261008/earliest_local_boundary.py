"""Final root-requested date-only coverage: earliest legal Taipei midnight."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FILES = ['data_cache/train_features.pkl', 'reports/feature_schema.json',
         'outputs/round6/geography/development/train_source_cache_indices.npy',
         'outputs/round98_artifact_inventory/results.json',
         'outputs/geography/osm_2021/audit.json']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write(name, value):
    p = OUT / name
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(p) + '\n')
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)}


def run():
    tick = time.perf_counter()
    before = {p: sha(ROOT / p) for p in FILES}
    request = write('earliest_boundary_request.json', {
        'root_explicit_final_date_coverage_request': '2021-01-03 through2021-08-10, derived from timestamp not prices/scores; no new experiment selected',
        'input_sha256': before, 'allowed_deserialization': 'original pureX87frame then only trade_day_index; canonical standaloneint64 mapping; aggregateJSON',
        'no_price_prediction_model_private_state_or_NPZ_reads': True})
    def js(p):
        need(p in FILES and p.endswith('.json'), 'Only fixed aggregate metadata')
        return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
    schema = js('reports/feature_schema.json')
    inv = js('outputs/round98_artifact_inventory/results.json')
    geo = js('outputs/geography/osm_2021/audit.json')
    rec = inv['canonical_mapping']
    need(before[rec['path']] == rec['file_sha256'] and before['data_cache/train_features.pkl'] == inv['blocks']['auxiliary']['feature_file_sha256'], 'Feature/mapping identity')
    mapping = np.load(ROOT / rec['path'], allow_pickle=False)
    need(mapping.dtype == np.dtype('int64') and list(mapping.shape) == rec['shape'] and
         hashlib.sha256(mapping.tobytes()).hexdigest() == rec['values_sha256'], 'Mapping schema/value')
    frame = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
    need(len(frame) == 455784 and list(frame.columns) == schema['columns'] and
         not {'總價元', 'y', 'target', 'ID', 'id', '編號'}.intersection(frame.columns), 'Pure X')
    days = frame['trade_day_index'].iloc[mapping].to_numpy(dtype='float64', copy=True)
    del frame
    stamp = pd.Timestamp(geo['source']['header']['replication_timestamp'])
    earliest = (stamp.tz_convert('Asia/Taipei').normalize() + pd.Timedelta(days=1))
    need(earliest.strftime('%Y-%m-%d') == '2021-01-03' and earliest.tz_convert('UTC') > stamp and
         earliest - pd.Timedelta(days=1) < stamp, 'Earliest strictly later local midnight')
    start = (earliest.tz_localize(None) - pd.Timestamp('2000-01-01')).days
    end = (pd.Timestamp('2021-08-10') - pd.Timestamp('2000-01-01')).days
    finite = np.isfinite(days)
    fi = np.flatnonzero(finite & (days < start))
    qi = np.flatnonzero(finite & (days >= start) & (days <= end))
    need(len(fi) + len(qi) == 134805 and not np.intersect1d(fi, qi).size, 'Exact preaux union')
    need(all(sha(ROOT / p) == h for p, h in before.items()), 'Inputs changed')
    record = write('earliest_boundary_feasibility.json', {
        'status': 'passed_timestamp_derived_date_inventory', 'request': request,
        'source_sha256': sha(Path(__file__)), 'input_sha256': before,
        'unselected_start': '2021-01-03', 'query_end_inclusive': '2021-08-10',
        'prefix_rows': len(fi), 'query_rows': len(qi),
        'origin_utc': earliest.tz_convert('UTC').isoformat(),
        'geography_header': stamp.isoformat(), 'header_strictly_before_origin': True,
        'seconds_origin_after_header': float((earliest - stamp).total_seconds()),
        'fit_query_disjoint_union_exact_preauxiliary_fit': True,
        'seen_new_support_or_price_tail_measured_here': False,
        'small_meta_sample_and_potential_tail_dominance_are_counterevidence_not_quantified': True,
        'new_model_or_cutoff_experiment_selected': False, 'new_training_arrays_saved': False,
        'price_values_loaded': 0, 'prediction_values_loaded': 0, 'models_loaded': 0,
        'private_states_loaded': 0, 'new_fits': 0, 'GPU_fits': 0, 'MAE_computed': False,
        'NPZ_opened': False, 'new_CSV': False, 'Public_score': None, 'Private_score': None,
        'all_inputs_unchanged': True, 'elapsed_seconds': time.perf_counter() - tick,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'result': record, 'prefix_rows': len(fi), 'query_rows': len(qi),
        'earliest_local_midnight': earliest.isoformat()}, ensure_ascii=False))


if __name__ == '__main__':
    run()
