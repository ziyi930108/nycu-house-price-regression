"""R118 bounded dates/indices feasibility audit. No price/prediction/model loading.

Only this new audit folder is written. Existing models and preparers are never
imported or called. JSON metadata may declare price/model paths but those paths
are not opened. The target-free feature pickle is necessarily materialized as
its original 87-column frame, then only three date columns are retained.
"""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLAN = 'reports/round118_feasibility_plan.md'
INV = 'outputs/round98_artifact_inventory/results.json'
PREP = 'outputs/round98_inputs/manifest.json'
FEATURE = 'data_cache/train_features.pkl'
MAP = 'outputs/round6/geography/development/train_source_cache_indices.npy'
SCHEMA = 'reports/feature_schema.json'
ORIGINS = ('auxiliary', 'development', 'offset', 'confirmation')
SPLITS = {'auxiliary': ('2021-08-11', '2022-08-10'),
          'development': ('2022-08-11', '2023-08-10'),
          'offset': ('2023-02-11', '2024-02-10'),
          'confirmation': ('2023-08-11', '2024-08-10')}
DATE_COLUMNS = ('trade_day_index', 'trade_date_invalid', 'trade_date_out_of_range')
FORBIDDEN_COLUMNS = {'總價元', '編號', 'target', 'y', 'ID', 'id'}
TEXT = [PLAN, 'outputs/research_continue_20261008/direction_review_a/conditional_meta_direction.md',
        'outputs/research_continue_20261008/direction_review_a/conditional_meta_direction.json',
        'reports/regression_direction_after_r97_inventory.md',
        'reports/validation_design_audit_r84.md', 'reports/r89_offset_validation_inventory.md',
        'src/round98_contract.py', 'src/round98_prepare_inputs.py',
        'src/round98_temporal_stack.py', 'src/round99_temporal_stack.py',
        'src/round6_prepare_geo.py', 'src/round6_geo.py',
        'src/round39_prepare_geo.py', 'src/geography_city_road_support.py',
        'src/round90_offset_data.py', 'src/round90_offset_nn.py',
        'src/round90_offset_trees.py', 'src/round97_offset_experiment.py',
        'src/round3_backtest.py', 'src/round79_recency_unit.py',
        'src/model_common.py', 'src/house_features.py', 'src/round47_nn_encode.py',
        'src/round60_recency_price_prior.py', 'src/round2_neighborhood.py',
        'src/round8_spatial_reference.py']
META = [INV, PREP, SCHEMA, 'outputs/round79/plan.json',
        'outputs/round79/results.json', 'outputs/round71/results.json',
        'outputs/round46/results.json', 'outputs/round90/preparation/manifest.json',
        'outputs/round97/sealed_manifest.json'] + [
        f'outputs/round{r}/geography/{s}/manifest.json'
        for r in (6, 39) for s in ('auxiliary', 'development', 'confirmation')] + [
        f'outputs/geography/osm_{y}/audit.json' for y in (2021, 2022, 2023, 2024)]
INDEX_FILES = [MAP] + [f'outputs/round79/{s}/{k}_indices.npy'
        for s in ('auxiliary', 'development', 'confirmation') for k in ('fit', 'query')] + [
        f'outputs/round97/{k}_indices.npy' for k in ('fit', 'query')] + [
        f'outputs/round98_inputs/{s}/{k}.npy'
        for s in ('development', 'offset', 'confirmation')
        for k in ('meta_indices', 'meta_trade_days')]
PRED_FILES = [f'outputs/round71/{s}/seed42/nn_spatial.npy'
        for s in ('auxiliary', 'development', 'confirmation')] + [
        f'outputs/round46/{s}/recency_log_spatial.npy'
        for s in ('auxiliary', 'development', 'confirmation')] + [
        f'outputs/round79/{s}/{k}.npy'
        for s in ('auxiliary', 'development', 'confirmation')
        for k in ('unit_spatial', 'predictions')] + [
        f'outputs/round97/reference_spatial/{k}/spatial.npy' for k in ('N', 'H', 'U22')] + [
        'outputs/round97/v22_predictions.npy']
ALLOWED = set(TEXT + META + INDEX_FILES + PRED_FILES + [FEATURE])
OPENED = {}


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def safe(name):
    need(name in ALLOWED, 'Not in fixed R118 readset: ' + str(name))
    p = (ROOT / name).resolve()
    need(p.is_relative_to(ROOT) and p.is_file(), 'Missing or unsafe input: ' + name)
    return p


def sha(name):
    h = hashlib.sha256()
    with safe(name).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def vsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(name, value):
    p = OUT / name
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(digest + '\n')
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': digest}


def metadata(name):
    need(name in META, 'JSON metadata read forbidden')
    OPENED[name] = 'json_aggregate_metadata'
    return json.loads(safe(name).read_text(encoding='utf-8-sig'))


def array(name, expected):
    need(name in INDEX_FILES, 'Only standalone date/index arrays may be deserialized')
    need(sha(name) == expected.get('sha256', expected.get('file_sha256')), 'File SHA: ' + name)
    a = np.load(safe(name), allow_pickle=False)
    need(str(a.dtype) == expected['dtype'] and list(a.shape) == expected['shape'], 'Array schema: ' + name)
    need(vsha(a) == expected['values_sha256'], 'Array values SHA: ' + name)
    need(a.ndim == 1 and not a.dtype.hasobject, 'Only 1D date/index values')
    if not name.endswith('trade_days.npy'):
        need(a.dtype == np.dtype('int64') and len(a) == len(np.unique(a)), 'Unique int64 index')
        if not name.endswith('meta_indices.npy'):
            need(np.array_equal(a, np.unique(a)), 'Sorted canonical index')
    OPENED[name] = 'standalone_target_free_dates_or_indices'
    return a


def day(s):
    return (pd.Timestamp(s) - pd.Timestamp('2000-01-01')).days


def ds(v):
    return (pd.Timestamp('2000-01-01') + pd.Timedelta(days=int(v))).strftime('%Y-%m-%d')


def run():
    tick = time.perf_counter()
    # Discovery text/metadata was read interactively before this file was made.
    # This exact list is sealed BEFORE any R118 real dates/indices are loaded.
    before = {p: sha(p) for p in sorted(ALLOWED)}
    listings = {folder: sorted(p.name for p in (ROOT / folder).iterdir()) for folder in
                ('data_external/geography', 'outputs/geography')}
    planned = write('planned_readset.json', {
        'role': 'Exact binary readset sealed before real R118 date/index analysis',
        'text_metadata_discovery_preceded_this_seal': True,
        'immutable_input_sha256': before, 'metadata_files': META, 'source_and_report_files': TEXT,
        'standalone_date_index_arrays': INDEX_FILES, 'opaque_prediction_hash_only': PRED_FILES,
        'feature_pickle': {'path': FEATURE, 'target_free': True,
            'materializes_original_87_columns_then_retains_only': list(DATE_COLUMNS)},
        'directory_names_only': listings,
        'forbidden': ['raw train/valid/test price tables', 'any NPZ', 'query_y',
            'prediction array deserialization', 'model files', 'private prior/reference/spatial state'],
        'no_new_cutoff_selected': True, 'no_training_authorized_by_this_audit': True})
    inv = metadata(INV)
    prep = metadata(PREP)
    schema = metadata(SCHEMA)
    need(inv['status'] == 'passed' and inv['canonical_rows'] == 455255
         and inv['invalid_date_rows'] == 8, 'Inventory contract')
    need(before[INV] == prep['inventory']['sha256'], 'Original R98 inventory binding')
    for p in ('outputs/round79/plan.json', 'outputs/round79/results.json',
              'outputs/round71/results.json', 'outputs/round46/results.json',
              'outputs/round90/preparation/manifest.json', 'outputs/round97/sealed_manifest.json', SCHEMA):
        need(before[p] == inv['authority_sha256'][p], 'Authority changed: ' + p)
    need(before[FEATURE] == prep['features']['sha256'] and
         before[MAP] == prep['canonical_mapping']['sha256'], 'Feature/mapping contract')
    need(before['src/house_features.py'] == schema['feature_source_sha256'], 'Feature source')
    mapping = array(MAP, inv['canonical_mapping'])
    need(len(mapping) == 455255 and mapping.min() >= 0 and mapping.max() < 455784, 'Canonical bounds')
    frame = pd.read_pickle(safe(FEATURE))
    OPENED[FEATURE] = 'target_free_87_column_frame_materialized_then_date_columns_only_retained'
    need(len(frame) == 455784 and list(frame.columns) == schema['columns'] and
         len(frame.columns) == 87 and not FORBIDDEN_COLUMNS.intersection(frame.columns), 'Pure X schema')
    dates = frame.loc[:, list(DATE_COLUMNS)].iloc[mapping].reset_index(drop=True).copy()
    del frame
    days = dates['trade_day_index'].to_numpy(dtype='float64', copy=True)
    invalid = dates['trade_date_invalid'].to_numpy()
    outside = dates['trade_date_out_of_range'].to_numpy()
    need(np.array_equal(~np.isfinite(days), invalid.astype(bool)) and
         int(invalid.sum()) == 8 and int(outside.sum()) == 5, 'Original invalid date flags')
    del dates
    valid = np.isfinite(days)
    blocks = {}
    authority_vectors = {}
    for s in ORIGINS:
        b = inv['blocks'][s]
        need((b['cutoff'], b['query_end']) == SPLITS[s], 'Fixed origin')
        fi, qi = (array(b[k]['path'], b[k]) for k in ('fit_indices', 'query_indices'))
        expected_fi = np.flatnonzero(valid & (days < day(b['cutoff'])))
        expected_qi = np.flatnonzero(valid & (days >= day(b['cutoff'])) & (days <= day(b['query_end'])))
        need(fi.tobytes() == expected_fi.tobytes() and qi.tobytes() == expected_qi.tobytes(), 'Complete canonical date masks: ' + s)
        need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
        hashes = {}
        for role, record in {**b['branches'], 'v22': b['control_v22']}.items():
            p = record['path']
            need(p in PRED_FILES and before[p] == record['file_sha256'], 'Opaque branch authority hash: ' + p)
            hashes[role] = {'path': p, 'sha256': before[p], 'deserialized': False,
                            'authority': record['authority']}
        authority_vectors[s] = hashes
        blocks[s] = {'fi': fi, 'qi': qi}
    coverage = {}
    for s in ORIGINS:
        cutoff = SPLITS[s][0]
        source_order = [] if s == 'auxiliary' else (['auxiliary'] if s == 'development' else ['auxiliary', 'development'])
        pieces = [blocks[k]['qi'][days[blocks[k]['qi']] < day(cutoff)] for k in source_order]
        ids = np.concatenate(pieces) if pieces else np.empty(0, dtype='int64')
        need(len(ids) == len(np.unique(ids)) and np.isin(ids, blocks[s]['fi']).all()
             and not np.intersect1d(ids, blocks[s]['qi']).size, 'Legal forward meta membership')
        original_verified = False
        if s != 'auxiliary':
            q = prep['stages'][s]
            saved_ids = array(q['arrays']['meta_indices']['path'], q['arrays']['meta_indices'])
            saved_days = array(q['arrays']['meta_trade_days']['path'], q['arrays']['meta_trade_days'])
            need(ids.tobytes() == saved_ids.tobytes() and days[ids].tobytes() == saved_days.tobytes()
                 and len(ids) == q['meta_rows'], 'Saved R98 meta exact replay')
            original_verified = True
        eligible = {k: int(np.count_nonzero(days[blocks[k]['qi']] < day(cutoff))) for k in ORIGINS}
        all_ids = np.concatenate([blocks[k]['qi'][days[blocks[k]['qi']] < day(cutoff)] for k in ORIGINS])
        extra = int(len(np.setdiff1d(np.unique(all_ids), ids)))
        coverage[s] = {'cutoff': cutoff, 'query_end_inclusive': SPLITS[s][1],
            'fit_rows': len(blocks[s]['fi']), 'query_rows': len(blocks[s]['qi']),
            'existing_R98_source_order': source_order, 'existing_forward_meta_rows': len(ids),
            'existing_forward_meta_indices_sha256': vsha(ids),
            'meta_first_date': ds(days[ids].min()) if len(ids) else None,
            'meta_last_date': ds(days[ids].max()) if len(ids) else None,
            'original_R98_meta_indices_and_dates_exact': original_verified,
            'all_four_saved_blocks_eligible_rows_before_origin': eligible,
            'additional_unique_meta_rows_from_other_saved_blocks': extra,
            'existing_origin_forward_meta_nonempty': bool(len(ids))}
    overlaps = []
    for i, a in enumerate(ORIGINS):
        for b in ORIGINS[i + 1:]:
            overlaps.append({'a': a, 'b': b,
                'query_shared_canonical_rows': int(len(np.intersect1d(blocks[a]['qi'], blocks[b]['qi'])))})
    # Exhaustive anniversary bands ending before auxiliary, not a chosen cutoff.
    earliest, latest = ds(days[valid].min()), ds(days[valid].max())
    annual = []
    for year in range(int(earliest[:4]) - 1, 2021):
        start, end = f'{year}-08-11', f'{year + 1}-08-10'
        fi = valid & (days < day(start))
        qi = valid & (days >= day(start)) & (days <= day(end))
        if not np.any(qi):
            continue
        annual.append({'candidate_start_not_selected': start, 'query_end_inclusive': end,
            'prefix_rows': int(fi.sum()), 'query_rows': int(qi.sum()),
            'first_observed_date': ds(days[qi].min()), 'last_observed_date': ds(days[qi].max()),
            'full_anniversary_band_after_earliest_observation': start >= earliest,
            'has_nonempty_prefix': bool(fi.any()),
            'complete_v22_forward_predictions_found_in_bounded_authority': False})
    geography = {}
    for year in (2021, 2022, 2023, 2024):
        p = f'outputs/geography/osm_{year}/audit.json'
        d = metadata(p)
        src = d['source']
        geography[str(year)] = {'audit': {'path': p, 'sha256': before[p]},
            'PBF_path_recorded_not_opened': src['path'], 'recorded_PBF_sha256': src['sha256'],
            'replication_timestamp': src['header']['replication_timestamp'],
            'timestamp_semantics': src['header']['timestamp_interpretation'],
            'audit_created_at_utc_not_historical_availability': d['created_at_utc']}
    aligned_geo = {}
    for s in ('auxiliary', 'development', 'confirmation'):
        records = {}
        for r in (6, 39):
            p = f'outputs/round{r}/geography/{s}/manifest.json'
            d = metadata(p)
            records[str(r)] = {k: d[k] for k in ('snapshot_year', 'replication_timestamp', 'forecast_origin_utc')}
            records[str(r)]['path'] = p
            records[str(r)]['sha256'] = before[p]
        need(records['6']['replication_timestamp'] == records['39']['replication_timestamp'], 'Mixed geography')
        need(pd.Timestamp(records['6']['replication_timestamp']) < pd.Timestamp(records['6']['forecast_origin_utc']), 'Geography after origin')
        aligned_geo[s] = records
    offset = metadata('outputs/round90/preparation/manifest.json')
    aligned_geo['offset'] = {k: offset[k] for k in ('geography_replication_timestamp', 'new_origin_utc')}
    need(offset['geography_replication_timestamp'] == geography['2023']['replication_timestamp'], 'Offset geography')
    earliest_geo = min(pd.Timestamp(v['replication_timestamp']) for v in geography.values())
    for a in annual:
        origin = pd.Timestamp(a['candidate_start_not_selected']).tz_localize('Asia/Taipei').tz_convert('UTC')
        a['earliest_existing_geography_precedes_origin'] = bool(earliest_geo < origin)
    for p in TEXT:
        OPENED[p] = 'source_or_report_text_read_interactively_only'
    unchanged = {p: sha(p) == digest for p, digest in before.items()}
    need(all(unchanged.values()), 'Existing readset changed')
    result = {'schema_version': 1, 'status': 'passed_bounded_feasibility_audit',
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan': {'path': PLAN, 'sha256': before[PLAN]}, 'planned_readset': planned,
        'canonical_date_summary': {'raw_feature_rows': 455784, 'canonical_rows': len(days),
            'removed_rows_by_existing_mapping_only': 529, 'valid_date_rows': int(valid.sum()),
            'invalid_date_rows': int(invalid.sum()), 'out_of_range_date_rows': int(outside.sum()),
            'earliest_valid_date': earliest, 'latest_valid_date': latest,
            'canonical_trade_day_values_sha256': vsha(days),
            'canonical_mapping_recomputed_or_raw_dedup_repeated': False},
        'four_origin_forward_meta_coverage': coverage, 'saved_forward_branch_opaque_hash_checks': authority_vectors,
        'query_window_overlap_counts': overlaps, 'all_anniversary_bands_before_auxiliary': annual,
        'geographic_snapshot_metadata': geography, 'aligned_original_geography': aligned_geo,
        'geography_directory_names_only': listings,
        'conclusion': 'Four-origin complete meta coverage is not ready: auxiliary has no earlier full forward predictions, and all enumerated pre-auxiliary August origins precede the earliest existing 2021 geography snapshot. No new cutoff or model is selected.',
        'complete_four_origin_meta_ready': False, 'early_model_and_state_rebuild_required': True,
        'historical_geography_before_early_origin_missing_in_bounded_inventory': True,
        'input_files_unchanged': all(unchanged.values()), 'unchanged_checks': unchanged,
        'new_fits': 0, 'GPU_fits': 0, 'models_loaded': 0, 'private_states_loaded': 0,
        'price_values_loaded': 0, 'prediction_values_loaded': 0, 'query_y_loaded': 0,
        'raw_price_pickle_opened': False, 'NPZ_opened': False, 'valid_test_read': False,
        'MAE_computed': False, 'bootstrap_computed': False, 'new_training_arrays_saved': False,
        'new_CSV': False, 'new_cutoff_selected': False, 'new_model_selected': False,
        'automatic_Kaggle_upload_or_selection': False, 'Public_score': None, 'Private_score': None,
        'limitations': ['Text/metadata discovery preceded the exact binary readset seal; chronology is disclosed.',
            'The trusted pure-X pickle materialized all 87 feature columns; only three date columns were retained and used. No raw price table was opened.',
            'Existing aggregate reports may contain old scores; no score or row prediction was newly computed.',
            'Geography PBF/header values are authenticated metadata claims, not re-parsed here; PBFs were not opened.',
            'Directory absence is a bounded inventory, not a global filesystem absence proof.',
            'Complete annual band means date-range coverage, not independent labels or unbiased holdout.',
            'Original source text supports isolated rebuild recipes, not a completed clean-environment end-to-end reproduction.'],
        'elapsed_seconds': time.perf_counter() - tick,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    record = write('feasibility.json', result)
    write('readset.json', {'planned_readset': planned, 'actual_deserialization_or_text_read_modes': OPENED,
        'opaque_hash_only': PRED_FILES, 'input_sha256': before, 'all_unchanged': all(unchanged.values()),
        'shared_handoff_context_read_before_audit_not_frozen': ['HANDOFF.md:first18', 'HANDOFF_LOG.md:tail14']})
    print(json.dumps({'status': result['status'], 'result': record,
        'canonical_dates': result['canonical_date_summary'], 'meta_rows': {
            s: coverage[s]['existing_forward_meta_rows'] for s in ORIGINS},
        'annual_bands': annual, 'overlaps': overlaps,
        'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true', required=True)
    p.parse_args()
    run()
