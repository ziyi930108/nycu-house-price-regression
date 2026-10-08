"""Root-requested R118 partial forward-band date inventory; no model selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FILES = ['reports/round118_feasibility_plan.md',
         'data_cache/train_features.pkl', 'reports/feature_schema.json',
         'outputs/round6/geography/development/train_source_cache_indices.npy',
         'outputs/round79/auxiliary/fit_indices.npy',
         'outputs/round98_artifact_inventory/results.json',
         'outputs/round6/geography/auxiliary/manifest.json',
         'outputs/round39/geography/auxiliary/manifest.json',
         'outputs/geography/osm_2021/audit.json',
         'src/round6_geo.py', 'src/round6_prepare_geo.py',
         'src/round39_prepare_geo.py', 'src/geography_city_road_support.py']
CANDIDATES = ('2021-01-02', '2021-02-11')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    p = OUT / name
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    digest = sha(p)
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(digest + '\n')
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': digest}


def js(name):
    need(name in FILES and name.endswith('.json'), 'Only fixed metadata JSON')
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def day(s):
    return (pd.Timestamp(s) - pd.Timestamp('2000-01-01')).days


def run():
    tick = time.perf_counter()
    before = {p: sha(ROOT / p) for p in FILES}
    request = write('supplemental_request.json', {
        'source': 'Root explicit task message on 2026-10-08; expansion of R118 pure-X feasibility inventory only',
        'candidate_dates_not_selected': list(CANDIDATES), 'query_end_inclusive': '2021-08-10',
        'full_forward_block_means_all_six_natives_and_reference_spatial_not_necessarily_12_months': True,
        'input_sha256': before, 'deserialization': {
            'target_free_feature_pickle': 'materialize87 then retain trade_day_index only',
            'standalone_indices': ['canonical_mapping', 'auxiliary_fit_indices'],
            'other_binary_files': 'not opened'},
        'no_fit_labels_or_query_labels_or_predictions_models_private_states': True,
        'no_new_cutoff_or_model_selected': True})
    inv = js('outputs/round98_artifact_inventory/results.json')
    schema = js('reports/feature_schema.json')
    rm = inv['canonical_mapping']
    mapping = np.load(ROOT / rm['path'], allow_pickle=False)
    need(before[rm['path']] == rm['file_sha256'] and mapping.dtype == np.dtype('int64')
         and list(mapping.shape) == rm['shape'] and
         hashlib.sha256(mapping.tobytes()).hexdigest() == rm['values_sha256'], 'Mapping')
    ri = inv['blocks']['auxiliary']['fit_indices']
    auxfi = np.load(ROOT / ri['path'], allow_pickle=False)
    need(before[ri['path']] == ri['file_sha256'] and
         hashlib.sha256(auxfi.tobytes()).hexdigest() == ri['values_sha256'], 'Auxiliary fit index')
    frame = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
    need(list(frame.columns) == schema['columns'] and len(frame) == 455784 and len(frame.columns) == 87
         and not {'總價元', 'y', 'target', '編號', 'ID', 'id'}.intersection(frame.columns), 'Pure X schema')
    days = frame['trade_day_index'].iloc[mapping].to_numpy(dtype='float64', copy=True)
    del frame
    valid = np.isfinite(days)
    need(valid.sum() == 455247 and np.array_equal(auxfi, np.flatnonzero(valid & (days < day('2021-08-11')))), 'Canonical date mask')
    r6 = js('outputs/round6/geography/auxiliary/manifest.json')
    r39 = js('outputs/round39/geography/auxiliary/manifest.json')
    audit = js('outputs/geography/osm_2021/audit.json')
    stamp = pd.Timestamp(audit['source']['header']['replication_timestamp'])
    need(r6['replication_timestamp'] == r39['replication_timestamp'] == audit['source']['header']['replication_timestamp'], 'Snapshot metadata identity')
    need(r6['source_pbf_sha256'] == audit['source']['sha256'] and
         r6['index_files_sha256']['outputs/round6/geography/auxiliary/train_source_cache_indices.npy'] == rm['file_sha256'], 'Annual geography mapping/source binding')
    need(r6['geographic_values_learned_from_targets'] is False and
         r6['feature_rules']['query_or_target_fitted_state'] is False and
         r6['features']['train']['rows'] == 455255 and
         r39['features']['fit']['rows'] == len(auxfi), 'Target-free geography contract')
    cache = {'tree_2021_canonical_train': r6['features']['train'],
             'spatial_2021_auxiliary_fit': r39['features']['fit']}
    for record in cache.values():
        need((ROOT / record['path']).is_file(), 'Declared geography cache missing')
    candidates = []
    for start in CANDIDATES:
        fi = np.flatnonzero(valid & (days < day(start)))
        qi = np.flatnonzero(valid & (days >= day(start)) & (days <= day('2021-08-10')))
        origin = pd.Timestamp(start).tz_localize('Asia/Taipei').tz_convert('UTC')
        need(np.isin(fi, auxfi).all() and np.isin(qi, auxfi).all()
             and not np.intersect1d(fi, qi).size and len(fi) + len(qi) == len(auxfi), 'Exact auxiliary-fit partition')
        candidates.append({'candidate_start_not_selected': start,
            'origin_utc_using_original_Taipei_midnight_convention': origin.isoformat(),
            'query_end_inclusive': '2021-08-10', 'prefix_rows': len(fi), 'query_rows': len(qi),
            'fit_query_disjoint_and_union_exact_auxiliary_fit': True,
            'fit_and_query_both_subset_R39_auxiliary_fit': True,
            '2021_geography_snapshot_strictly_before_origin': bool(stamp < origin),
            'snapshot_minus_origin_seconds': float((stamp - origin).total_seconds()),
            'six_native_models_encoder_prior_reference_spatial_for_this_cutoff_exist_in_bounded_authority': False})
    unchanged = {p: sha(ROOT / p) == h for p, h in before.items()}
    need(all(unchanged.values()), 'Input changed')
    result = {'status': 'passed_partial_band_feasibility_inventory', 'request': request,
        'source_sha256': sha(Path(__file__)), 'input_sha256': before,
        'candidates': candidates, 'geography_replication_timestamp': stamp.isoformat(),
        'geography_cache_metadata': cache,
        'cache_binary_loaded_or_rehashed': False,
        'cache_existence_only_checked_metadata_SHA_not_actual_binary_reverified': True,
        'source_findings': {
            'R6_encode_geo_features': 'Deterministic rowwise 8-column transformation from matched geometry/status; no fitted statistics, y or query population.',
            'R6_existing_cache_alignment': 'Canonical455255 train row axis recorded against unchanged inherited mapping; original alignment used price-containing dedup once, which must not be recomputed now.',
            'R39_fallback': 'Row address predicate and unique usable city-road record in full2021 index. classify(rows) only uses rows in aggregate counts, not uniqueness/coordinates. The old aux fit geometry covers both candidate partitions.',
            'new_pure_X_adapter_required': 'Reindex R6 canonical train and R39 auxiliary-fit row axis through saved auxfi; preserve identical eight columns/dtypes/float32 bytes and new fi/qi hashes. No existing fixed-stage driver may be relabelled.',
            'learned_states_must_rebuild': 'LGB category maps, R47 numeric/vocab encoder, R54 knots, R60 OOF/full prior/dateweights, all six native models, reference and spatial fit states must use each new prefix.',
            'geometry_not_private_spatial_state': 'R39 feature cache is pure geometry; R8 spatial model contains label-derived residuals and cannot be reused.'},
        'all_inputs_unchanged': True, 'new_fits': 0, 'GPU_fits': 0,
        'price_values_loaded': 0, 'prediction_values_loaded': 0, 'query_y_loaded': 0,
        'models_loaded': 0, 'private_states_loaded': 0, 'NPZ_opened': False,
        'MAE_computed': False, 'bootstrap_computed': False,
        'new_training_arrays_saved': False, 'new_CSV': False,
        'new_cutoff_selected': False, 'new_model_selected': False,
        'automatic_Kaggle_upload_or_selection': False, 'Private_score': None, 'Public_score': None,
        'elapsed_seconds': time.perf_counter() - tick,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    record = write('partial_band_feasibility.json', result)
    print(json.dumps({'result': record, 'candidates': candidates,
                      'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    run()
