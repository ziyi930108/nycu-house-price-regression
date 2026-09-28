"""Read-only R91 evidence audit; saved-array descriptions, no labels/model/score.

Run only after the three-period main/generic/two-paired jobs have completed.
New output is exclusive-created; the audited experiment is never overwritten.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAGES = ('development', 'confirmation', 'auxiliary')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def values_sha(a):
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def describe(a):
    a = np.asarray(a, dtype=np.float64)
    assert a.ndim == 1 and len(a) and np.isfinite(a).all()
    q = np.quantile(a, [0, .01, .5, .9, .99, 1])
    return {'rows': len(a), 'mean': float(a.mean()),
            **{k: float(v) for k, v in zip(('min', 'p01', 'median', 'p90', 'p99', 'max'), q)}}


def main():
    begin = time.perf_counter()
    read_hashes = {}

    def read(name):
        p = ROOT / name
        h = sha(p)
        read_hashes[name] = h
        side = p.with_suffix('.sha256')
        if side.exists():
            assert side.read_text('utf8').strip() == h, name
            read_hashes[side.relative_to(ROOT).as_posix()] = sha(side)
        return json.loads(p.read_text('utf8'))

    result = read('outputs/round91/results.json')
    plan = read('outputs/round91/plan.json')
    control = read('outputs/round79/results.json')
    diag = read('reports/round91_diagnostics.json')
    paired = {c: read(f'outputs/round91/paired_vs_{c}/results.json') for c in ('v22', 'v24')}
    execution = read('outputs/round91/execution.json')
    diag_execution = read('outputs/round91/diagnostics/execution.json')
    assert execution['exit_code'] == diag_execution['exit_code'] == 0
    assert result['checked_stages'] == list(STAGES) and result['new_fits'] == 3
    assert result['protected_before'] == result['protected_after'] == plan['protected_before']
    assert result['plan_sha256'] == sha(ROOT / 'outputs/round91/plan.json')
    assert result['text_plan_sha256'] == sha(ROOT / 'reports/round91_plan.md')
    assert result['input_sha256'] == plan['input_sha256'] and result['source_sha256'] == plan['source_sha256']
    expected = {}
    for group in (result['input_sha256'], result['source_sha256'], result['protected_before']):
        for name, h in group.items():
            assert name not in expected or expected[name] == h
            expected[name] = h
    for s in STAGES:
        for name, h in result['stages'][s]['artifact_sha256'].items():
            assert name not in expected or expected[name] == h
            expected[name] = h
    for name, h in expected.items():
        assert sha(ROOT / name) == h, name
    assert not any(result[k] for k in ('new_CSV', 'new_test_predictions', 'automatic_adoption',
                                      'new_NN_fits', 'new_reference_or_spatial_states', 'Public_feedback_used_for_tuning'))
    intervals = []
    maximum_paired_point_difference = 0.
    for c, p in paired.items():
        assert p['status'] == 'completed' and p['replicates'] == 2000 and p['seed'] == 20260925
        for s in STAGES:
            assert p['stages'][s]['point_metrics_reproduced']
            for method in ('project', 'month'):
                for metric in ('overall', 'weighted'):
                    v = p['stages'][s][method]['summary'][metric]
                    assert v['valid_replicates'] == 2000 and v['invalid_replicates'] == 0
                    lo, hi = v['percentile_2_5'], v['percentile_97_5']
                    delta = result['stages'][s]['metrics']['R91'][metric + '_mae'] - result['stages'][s]['metrics'][c][metric + '_mae']
                    maximum_paired_point_difference = max(maximum_paired_point_difference, abs(delta - v['point_delta']))
                    intervals.append({'control': c, 'stage': s, 'method': method, 'metric': metric,
                                      'point_delta': v['point_delta'], 'lower': lo, 'upper': hi,
                                      'classification': 'positive' if lo > 0 else 'negative' if hi < 0 else 'crosses_zero'})

    # Only cached X is read here. No fit/query y is read and no predictor is called.
    Xraw = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
    kt = np.load(ROOT / 'outputs/round6/geography/development/train_source_cache_indices.npy', allow_pickle=False)
    assert len(Xraw) == 455784 and kt.shape == (455255,) and kt.dtype == np.int64
    X = Xraw.iloc[kt].reset_index(drop=True)
    building = X['building_area'].to_numpy(dtype=np.float64)
    parking = X['parking_area'].to_numpy(dtype=np.float64)
    A = np.where(np.isfinite(building) & (building > 0), building,
                 np.where(np.isfinite(parking) & (parking > 0), parking, 1.))
    summary = {}
    for s in STAGES:
        st = result['stages'][s]
        m = read(f'outputs/round91/{s}/model/metadata.json')
        state = read(f'outputs/round91/{s}/model/feature_state.json')
        pv = read(f'outputs/round91/{s}/prior_verification.json')
        fe = read(f'outputs/round91/{s}/model/execution.json')
        assert fe['exit_code'] == 0 and fe['elapsed_seconds'] == st['fit_seconds']
        assert state['legacy_feature_state'] == plan['bindings'][s]['R19_state']
        assert m['feature_count'] == 95 and m['categorical_feature_count'] == 23 and m['actual_trees'] == 2999
        assert m['added_feature_names'] == m['removed_feature_names'] == m['changed_parameter_names'] == []
        assert m['target_mode'] == 'signed_additive_total_residual_million'
        assert all(m[k] for k in ('reload_all_query_rows_verified', 'reload_raw_exact', 'reload_TWD_exact',
                                 'date_weight_provenance_verified', 'nonnegative_projection_after_offset'))
        assert not any(m[k] for k in ('area_multiplied_into_weights', 'signed_z_clipped_before_offset',
                                     'prior_as_feature', 'fit_uses_eval_set', 'early_stopping_used', 'query_labels_used'))
        assert all(pv[k] for k in ('fit_only_OOF_rebuild_exact', 'full_fit_state_exact', 'query_prediction_exact', 'R63_factor_values_exact'))
        assert not pv['chronological_oof'] and not pv['query_labels_used']
        assert all(st[k] for k in ('controls_full_metrics_exact', 'v22_base_reference_spatial_median_full_vectors_exact',
                                  'full_query_raw_TWD_reload_exact', 'date_weight_provenance_verified'))
        d = diag['stages'][s]
        assert d['all_full_saved_metrics_exact']
        for key in ('R91', 'v22', 'v24'):
            assert d['metrics'][key] == st['metrics'][key]
        fi = np.load(OUT / s / 'fit_indices.npy', allow_pickle=False)
        qi = np.load(OUT / s / 'query_indices.npy', allow_pickle=False)
        dates = np.load(OUT / s / 'model/date_weights.npy', allow_pickle=False)
        weights = np.load(OUT / s / 'model/pool_weights.npy', allow_pickle=False)
        assert dates.dtype == np.float32 and weights.dtype == np.float64
        assert dates.astype(np.float64).tobytes() == weights.tobytes()
        train_prior = np.load(ROOT / f'outputs/round60/{s}/train_oof_unit_prior.npy', allow_pickle=False)
        query_prior = np.load(ROOT / f'outputs/round60/{s}/query_unit_prior.npy', allow_pickle=False)
        fit_offset, query_offset = A[fi] * train_prior, A[qi] * query_prior
        assert values_sha(A[fi]) == state['fit_effective_area_sha256']
        assert values_sha(fit_offset) == state['fit_offset_TWD_sha256']
        raw = np.load(OUT / s / 'model/raw_z.npy', allow_pickle=False)
        native = np.load(OUT / s / 'model/native_predictions.npy', allow_pickle=False)
        before_projection = query_offset + raw * 1e6
        assert np.maximum(before_projection, 0.).tobytes() == native.tobytes()
        old_native = np.load(ROOT / f'outputs/round79/{s}/model/native_predictions.npy', allow_pickle=False)
        prediction = np.load(OUT / s / 'predictions.npy', allow_pickle=False)
        final_old = np.load(ROOT / f'outputs/round79/{s}/predictions.npy', allow_pickle=False)
        contributions = {}
        for c in ('v22', 'v24'):
            contributions[c] = {b['band']: (b['R91'] - b[c]) * b['rows'] / st['query_rows'] for b in d['price_bands']}
            assert abs(sum(contributions[c].values()) - (st['metrics']['R91']['overall_mae'] - st['metrics'][c]['overall_mae'])) < 1e-7
        summary[s] = {'fit_rows': len(fi), 'query_rows': len(qi), 'fit_seconds': st['fit_seconds'],
            'phase_seconds': st['elapsed_seconds'], 'overall': st['metrics']['R91']['overall_mae'],
            'weighted': st['metrics']['R91']['weighted_mae'], 'differences': st['differences'],
            'unit_layer_deltas_vs_v22': {k: st['unit_branch_metrics'][k]['overall_mae'] - control['stages'][s]['unit_branch_metrics'][k]['overall_mae'] for k in ('native', 'base', 'reference', 'spatial')},
            'price_band_contributions_TWD': contributions,
            'prior_values_TWD_per_m2': {'fit_group_OOF': describe(train_prior), 'query_fullfit': describe(query_prior)},
            'area_m2': {'fit': describe(A[fi]), 'query': describe(A[qi])},
            'offset_TWD': {'fit_group_OOF': describe(fit_offset), 'query_fullfit': describe(query_offset)},
            'fit_signed_target_counts': {k: m[k] for k in ('signed_residual_negative_rows', 'signed_residual_zero_rows', 'signed_residual_positive_rows')},
            'query_raw_signed_million': describe(raw),
            'query_native_TWD': describe(native), 'old_query_native_TWD': describe(old_native),
            'query_zero_native_rows': int((native == 0).sum()), 'query_negative_raw_rows': int((raw < 0).sum()),
            'query_offset_ge_100m_rows': int((query_offset >= 1e8).sum()),
            'query_offset_ge_1b_rows': int((query_offset >= 1e9).sum()),
            'query_native_ge_1b_rows': int((native >= 1e9).sum()),
            'query_area_ge_1000m2_rows': int((A[qi] >= 1000).sum()),
            'median_changed_rows': int((prediction != final_old).sum()),
            'median_equal_unit_rows': int((prediction == np.load(OUT / s / 'unit_spatial.npy', allow_pickle=False)).sum()),
            'prior_replay_seconds': pv['elapsed_seconds']}
    gate = {
        'all_three_overall_improve_v22': all(result['stages'][s]['metrics']['R91']['overall_mae'] < result['stages'][s]['metrics']['v22']['overall_mae'] for s in STAGES),
        'all_three_weighted_improve_v22': all(result['stages'][s]['metrics']['R91']['weighted_mae'] < result['stages'][s]['metrics']['v22']['weighted_mae'] for s in STAGES),
        'latest_overall_improvement_at_least_half_percent': result['stages']['confirmation']['metrics']['R91']['overall_mae'] <= .995 * result['stages']['confirmation']['metrics']['v22']['overall_mae']}
    audit = {'status': 'completed_read_only_evidence_and_saved_vector_descriptions',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_source_sha256': sha(Path(__file__)),
        'read_json_sha256': read_hashes, 'verified_dependency_and_artifact_sha256': expected,
        'unique_verified_dependency_and_artifact_files': len(expected), 'three_stages': summary,
        'predefined_gate': gate, 'all_gate_checks_pass': all(gate.values()), 'paired_intervals': intervals,
        'maximum_paired_point_difference_TWD': maximum_paired_point_difference,
        'CI_sign_counts': {c: {cl: sum(x['control'] == c and x['classification'] == cl for x in intervals)
                         for cl in ('positive', 'negative', 'crosses_zero')} for c in paired},
        'fit_seconds_sum': sum(x['fit_seconds'] for x in summary.values()), 'main_outer_seconds': execution['elapsed_seconds'],
        'generic_outer_seconds': diag_execution['elapsed_seconds'],
        'query_or_fit_labels_read': False, 'new_fit_predict_score_or_bootstrap': False,
        'test_data_read': False, 'prior_summary_comparison_is_not_OOF_effect_isolation': True,
        'private_score_or_win_probability_estimated': False, 'elapsed_seconds': time.perf_counter() - begin}
    with (OUT / 'independent_review.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: audit[k] for k in ('status', 'unique_verified_dependency_and_artifact_files', 'predefined_gate', 'fit_seconds_sum', 'CI_sign_counts', 'elapsed_seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
