"""Seal R118 bounded inventory and synthetic reader/temporal-role checks only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
checks = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, value):
    p = OUT / name
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(p) + '\n')


def passed(name, ok):
    if not ok:
        raise AssertionError(name)
    checks.append(name)


def rejects(name, fn):
    try:
        fn()
    except ValueError:
        checks.append(name)
        return
    raise AssertionError(name)


def run():
    spec = importlib.util.spec_from_file_location('r118_bounded_author_audit', OUT / 'audit_feasibility.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for path in ['data_cache/train.pkl', 'data_cache/valid.pkl', 'data_cache/test.pkl',
                 'outputs/round63/development/prepared.npz',
                 'outputs/round97/reference_spatial/reference.pkl',
                 'outputs/round97/NN/prior_state.json', 'outputs/round79/development/model/model.cbm']:
        rejects('fixed_readset_rejects_' + path, lambda p=path: m.safe(p))
    rejects('metadata_rejects_feature_pickle', lambda: m.metadata(m.FEATURE))
    rejects('array_reader_rejects_prediction_before_open',
            lambda: m.array('outputs/round79/development/predictions.npy', {}))
    rejects('array_reader_rejects_NPZ_before_open',
            lambda: m.array('outputs/round63/development/prepared.npz', {}))
    days = np.array([m.day('2021-01-01'), m.day('2021-01-02'), m.day('2021-02-10'),
                     m.day('2021-02-11'), m.day('2021-08-10'), m.day('2021-08-11'), np.nan], dtype='float64')
    f = np.flatnonzero(np.isfinite(days) & (days < m.day('2021-02-11')))
    q = np.flatnonzero(np.isfinite(days) & (days >= m.day('2021-02-11')) & (days <= m.day('2021-08-10')))
    passed('synthetic_prefix_strict_boundary', f.tolist() == [0, 1, 2])
    passed('synthetic_query_includes_both_endpoints', q.tolist() == [3, 4])
    passed('synthetic_invalid_date_excluded', 6 not in f and 6 not in q)
    passed('synthetic_fit_query_disjoint', np.intersect1d(f, q).size == 0)
    later = np.flatnonzero(np.isfinite(days) & (days < m.day('2021-08-11')))
    passed('synthetic_forward_meta_belongs_outer_fit', np.isin(q, later).all())
    query_current = np.array([3, 4, 5], dtype='int64')
    passed('synthetic_future_query_part_not_legal_meta', query_current[days[query_current] < m.day('2021-08-11')].tolist() == [3, 4])
    stamp = pd.Timestamp('2021-01-01T21:42:03Z')
    jan = pd.Timestamp('2021-01-02').tz_localize('Asia/Taipei').tz_convert('UTC')
    feb = pd.Timestamp('2021-02-11').tz_localize('Asia/Taipei').tz_convert('UTC')
    passed('January2_local_midnight_precedes_header', jan < stamp)
    passed('January2_exact_20523_seconds', (stamp - jan).total_seconds() == 20523)
    passed('February11_header_strictly_precedes_origin', stamp < feb)
    # AST-extract this standalone pure geometry classification; no legacy module
    # import or geographic/cache/price/model values are loaded.
    p = ROOT / 'src/geography_city_road_support.py'
    node = next(n for n in ast.parse(p.read_text(encoding='utf-8-sig')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'classify')
    space = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), space)
    classify = space['classify']
    for records in [[], [{'key': 'a', 'original_coordinate_admissible': True, 'quality': 'road_level'}],
                    [{'key': 'a', 'original_coordinate_admissible': False, 'quality': 'broad_road'}],
                    [{'key': 'a', 'original_coordinate_admissible': True, 'quality': 'road_level'},
                     {'key': 'b', 'original_coordinate_admissible': True, 'quality': 'road_level'}]]:
        a = classify(('city', 'road'), 1, records)
        b = classify(('city', 'road'), 999, records)
        passed('R39_row_frequency_changes_only_aggregate_' + str(len(checks)),
               {k: v for k, v in a.items() if k != 'candidate_rows'} ==
               {k: v for k, v in b.items() if k != 'candidate_rows'})
    for name in ('audit_feasibility.py', 'supplemental_date_candidates.py', 'seal_evidence.py'):
        source = (OUT / name).read_text(encoding='utf8')
        compile(source, name, 'exec')
        tree = ast.parse(source)
        called = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        passed('no_fit_predict_loadmodel_call_' + name,
               not {'fit', 'predict', 'load_model', 'torch_load', 'score', 'mae'}.intersection(called))
    original = json.loads((OUT / 'feasibility.json').read_text(encoding='utf8'))
    supplement = json.loads((OUT / 'partial_band_feasibility.json').read_text(encoding='utf8'))
    passed('original_receipt_source_exact', original['audit_source_sha256'] == sha(OUT / 'audit_feasibility.py'))
    passed('supplemental_receipt_source_exact', supplement['source_sha256'] == sha(OUT / 'supplemental_date_candidates.py'))
    passed('all_four_origin_extra_unique_meta_zero', all(
        v['additional_unique_meta_rows_from_other_saved_blocks'] == 0
        for v in original['four_origin_forward_meta_coverage'].values()))
    passed('auxiliary_meta_unavailable', original['four_origin_forward_meta_coverage']['auxiliary']['existing_forward_meta_rows'] == 0)
    write('author_synthetic_checks.json', {'status': 'passed', 'checks': checks, 'check_count': len(checks),
        'source_sha256': sha(Path(__file__)), 'author_review_not_independent': True,
        'new_actual_data_deserializations_by_selfcheck': 0, 'new_fits': 0, 'GPU_fits': 0,
        'price_values_loaded': 0, 'prediction_values_loaded': 0, 'models_loaded': 0,
        'private_states_loaded': 0, 'new_MAE_or_bootstrap': False, 'new_CSV': False})
    write('design_limits.json', {
        'status': 'bounded_inventory_no_model_or_cutoff_selected',
        'original_annual_date_inventory': {'path': 'outputs/research_round118_feasibility_20261008/feasibility.json',
                                          'sha256': sha(OUT / 'feasibility.json')},
        'later_root_authorized_partial_band_inventory': {'path': 'outputs/research_round118_feasibility_20261008/partial_band_feasibility.json',
                                                        'sha256': sha(OUT / 'partial_band_feasibility.json')},
        'initial_annual_only_conclusion_is_not_proof_all_partial_preaux_bands_lack_geography': True,
        'nonempty_annual_list_omits_zero_2008_band': {'start': '2008-08-11', 'end_inclusive': '2009-08-10',
            'prefix_rows': 1, 'query_rows': 0, 'derivation': 'prior2007band contains1 row;2009prefix still1'},
        'complete_four_origin_meta_available_now': False,
        'partial_forward_block_need_not_have_12_months': True,
        'January2_timezone_admissible': False, 'February11_geography_temporally_admissible': True,
        'February11_model_cutoff_selected': False,
        'geo_cache_reuse_source_and_metadata_supported_not_binary_replayed': True,
        'price_containing_learned_states_or_later_pureX_encoders_reusable': False,
        'prefix_relative_roles': {
            'pure_row_X': 'own transaction values, no price',
            'seen_support_count_and_date_ESS': 'own prediction-producing prefix, never later outer fit',
            'encoder_vocabulary_stats_ramps': 'own prediction-producing prefix pure X fit',
            'geography': 'row X and fixed map snapshot before own origin, exact canonical reindex',
            'prior_price_reference_spatial_actual_diagnostics': 'new prefix fit labels and state required',
            'branch_disagreement': 'own saved full forward branches; opaque hashes verified only here'},
        'strongest_counterevidence': 'R99 improved metaMAE15k-26k but worsened next-period2of3; conditional capacity and short earlier block do not remove selection bias.',
        'recommendation': 'A single future fixed full-forward-meta completion protocol may be designed, but no training or selection is authorized by this feasibility inventory.',
        'new_fits': 0, 'GPU_fits': 0, 'new_scores': 0, 'new_CSV': False,
        'automatic_Kaggle_upload_or_selection': False, 'Private_score': None, 'Public_score': None})
    md = OUT / 'feasibility.md'
    with md.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(md) + '\n')
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    write('manifest.json', {'status': 'sealed_bounded_feasibility_inventory',
        'artifacts': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        'no_source_or_existing_artifact_mutation': True,
        'independent_review_pending_root': True,
        'source_and_report_scopes': 'Dates/indices and text/aggregate metadata only; no models/private states/predictions/price values.',
        'CSV_or_final_selection_changes': False})
    print(json.dumps({'status': 'passed', 'checks': len(checks),
        'manifest_sha256': sha(OUT / 'manifest.json'),
        'report_sha256': sha(md)}, ensure_ascii=False))


if __name__ == '__main__':
    run()
