"""Finalize only new R118 documents after preserving a sidecar-name collision."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def js(name):
    return json.loads((OUT / name).read_text(encoding='utf8'))


def write(name, value):
    p = OUT / name
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(p) + '\n')


def record(name):
    p = OUT / name
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)}


def run():
    checks = js('author_synthetic_checks.json')
    need(checks['status'] == 'passed' and checks['check_count'] == 30, 'Author checks')
    need(checks['source_sha256'] == sha(OUT / 'seal_evidence.py'), 'Preserved failed sealer source identity')
    earliest = js('earliest_boundary_feasibility.json')
    need(earliest['prefix_rows'] == 126957 and earliest['query_rows'] == 7848
         and earliest['seconds_origin_after_header'] == 65877
         and earliest['source_sha256'] == sha(OUT / 'earliest_local_boundary.py'), 'Final date evidence')
    for name in ('audit_feasibility.py', 'supplemental_date_candidates.py', 'seal_evidence.py',
                 'earliest_local_boundary.py', 'finalize_evidence.py'):
        compile((OUT / name).read_text(encoding='utf8'), name, 'exec')
        with (OUT / (name + '.sha256')).open('x', encoding='ascii') as f:
            f.write(sha(OUT / name) + '\n')
    write('packaging_failure_attempt1.json', {
        'role': 'Preserved author document packaging failure, not model/data failure',
        'entry': record('seal_evidence.py'), 'exit_code': 1,
        'failure': "FileExistsError: feasibility.sha256 already belonged to feasibility.json; using Path.with_suffix for feasibility.md would collide.",
        'completed_before_failure': [record('author_synthetic_checks.json'), record('design_limits.json')],
        'no_existing_sidecar_overwritten': True, 'no_old_source_changed': True,
        'resolution': 'Separate finalize tool uses feasibility.md.sha256, preserves original JSON sidecar and both successfully written receipts; no real data rerun.'})
    write('final_design_limits.json', {
        'status': 'bounded_inventory_complete_no_model_or_cutoff_selected',
        'initial_limits': record('design_limits.json'),
        'final_timestamp_derived_addition': record('earliest_boundary_feasibility.json'),
        'complete_forward_block_means_six_native_plus_reference_spatial_not_fixed_duration': True,
        'earliest_existing_map_legal_Taipei_midnight': '2021-01-03',
        'candidate_2021_01_03_prefix_query_rows': [126957, 7848],
        'candidate_2021_02_11_prefix_query_rows': [128614, 6191],
        'candidate_2021_01_02_fails_original_timezone_contract': True,
        'small_meta_sample_may_be_dominated_by_price_tail': True,
        'seen_new_group_support_and_price_tail_not_measured': True,
        'formal_constructibility_not_strong_validation': True,
        'geo_cache_rowwise_source_and_metadata_supported_only_not_binary_replayed': True,
        'all_new_prefix_models_encoders_priors_reference_spatial_still_required': True,
        'single_recommendation': 'May separately fix one full-forward-meta completion protocol; independent preflight required before fit. No conditional model or cutoff has been chosen here.',
        'original_inputs_unique_count': 77, 'initial_prediction_files_opaque_hash_only': 16,
        'original_array_deserializations_dates_indices_only': 15,
        'new_fits': 0, 'GPU_fits': 0, 'price_values_loaded': 0,
        'prediction_values_loaded': 0, 'models_loaded': 0, 'private_states_loaded': 0,
        'new_scores_or_bootstrap': False, 'new_CSV': False,
        'automatic_Kaggle_upload_or_selection': False,
        'Public_score': None, 'Private_score': None})
    md = OUT / 'feasibility.md'
    with (OUT / 'feasibility.md.sha256').open('x', encoding='ascii') as f:
        f.write(sha(md) + '\n')
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    write('manifest.json', {
        'status': 'sealed_bounded_feasibility_inventory',
        'artifacts': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        'original_inputs_unique_count': 77,
        'source_and_report_scope': 'Dates/indices, source/aggregate metadata, geo filenames/timestamps. No price/model/private-state/prediction deserialization.',
        'new_sources_compiled_in_memory': True, 'author_synthetic_check_count': 30,
        'author_checks_not_independent_review': True, 'independent_review_pending_root': True,
        'no_existing_project_source_or_artifact_mutated': True,
        'no_model_or_cutoff_selected': True, 'no_training_or_CSV_or_Kaggle_changes': True,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    for p in OUT.glob('*.json'):
        need(p.with_suffix('.sha256').read_text(encoding='ascii').strip() == sha(p), 'JSON sidecar: ' + p.name)
    need((OUT / 'feasibility.md.sha256').read_text(encoding='ascii').strip() == sha(md), 'MD sidecar')
    print(json.dumps({'status': 'passed', 'report': record('feasibility.md'),
        'manifest': record('manifest.json'), 'checks': checks['check_count'],
        'initial_result': record('feasibility.json'),
        'partial_result': record('partial_band_feasibility.json'),
        'earliest_result': record('earliest_boundary_feasibility.json')}, ensure_ascii=False))


if __name__ == '__main__':
    run()
