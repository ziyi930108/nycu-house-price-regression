"""R78: target-free inspection of saved final tails; no fitting or prediction calls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / 'reports/round78_final_tail_inspection.json'
MD_OUT = ROOT / 'reports/round78_final_tail_inspection.md'
AUTHORITIES = {
    'outputs/round76_77/final/trial_release.json': 'd77da69bc3fa10fc41a2e811294ec24ab0d39dfb756a9f89951bbc5d8778ac2f',
    'outputs/round76_77/final_export/release.json': 'bc700d294d431805e9e5ce040e64524aad43c5284ec481c09bc90920bd5f66d2',
}
VECTORS = {
    'v19': 'outputs/round71/final/predictions.npy',
    'v20': 'outputs/round76/final/predictions.npy',
    'v21': 'outputs/round77/final/predictions.npy',
    'NN': 'outputs/round71/final/nn_spatial.npy',
    'weighted_NN': 'outputs/round76_77/final/weighted_area_nn/spatial.npy',
    'log': 'outputs/round46/trial/recency_log_spatial.npy',
    'old_unit': 'outputs/round40/trial/unit_spatial.npy',
    'new_unit': 'outputs/round76_77/final/new_unit/spatial.npy',
    'new_unit_reference': 'outputs/round76_77/final/new_unit/reference.npy',
    'new_unit_base': 'outputs/round76_77/final/new_unit/base.npy',
    'unit_native_TWD': 'outputs/round40/trial/model/predictions.npy',
    'old_unit_LGB_TWD': 'outputs/round2/final_predictions/lightgbm_l1_unit63_test.npy',
}
FEATURES = ['address_city', 'district', 'project_name', 'transaction_type', 'building_type',
    'main_use', 'main_material', 'parking_type', 'land_area', 'building_area', 'parking_area',
    'land_count', 'building_count', 'parking_count', 'rooms', 'living_rooms', 'bathrooms',
    'transfer_floor_raw', 'total_floors', 'floor_min', 'floor_max', 'floor_count',
    'is_entire_building', 'is_basement', 'indoor_area', 'area_per_building', 'parking_fraction',
    'building_area_negative', 'parking_area_negative', 'building_area_zero', 'large_area']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scalar(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def summary(a):
    a = np.asarray(a, dtype=np.float64)
    return {'rows': len(a), **dict(zip(['min', 'p50', 'p90', 'p99', 'p999', 'max'],
        map(float, np.quantile(a, [0, .5, .9, .99, .999, 1]))))}


def median_matches(v, final, names, positions):
    return {name: int(np.count_nonzero(v[name][positions] == v[final][positions])) for name in names}


def run():
    started = datetime.now(timezone.utc).isoformat(); tick = time.monotonic()
    assert not JSON_OUT.exists() and not MD_OUT.exists(), 'Refuse to overwrite inspection outputs'
    inputs = {}; known = {}; authorities = {}
    def bind(name, digest):
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs:
            assert sha(path) == digest, name
        inputs[name] = digest
        return path
    for name, digest in AUTHORITIES.items():
        doc = json.loads(bind(name, digest).read_text(encoding='utf8')); authorities[name] = doc
    proof, release = authorities.values()
    assert release['joint_vector_evidence_path'] == 'outputs/round76_77/final/trial_release.json'
    assert release['joint_vector_evidence_sha256'] == AUTHORITIES[release['joint_vector_evidence_path']]
    for family in ('input_sha256', 'artifact_sha256'):
        for name, digest in proof[family].items():
            assert name not in known or known[name] == digest, name
            known[name] = digest
    def checked(name):
        return bind(name, known[name])
    sources = {str(Path(__file__).relative_to(ROOT)).replace('\\', '/'): sha(__file__)}
    for name in ('src/house_features.py', 'src/round19_unit_geo.py', 'src/round76_77_final.py'):
        sources[name] = sha(bind(name, proof['source_sha256'][name]))
    protect = ['outputs/current_private_candidate.json', 'runs/versions.json',
        'outputs/submission_v19_trial.csv', 'outputs/submission_v20_trial.csv', 'outputs/submission_v21_trial.csv']
    before = {name: sha(ROOT / name) for name in protect}
    schema = json.loads(checked('reports/feature_schema.json').read_text(encoding='utf8'))
    X = pd.read_pickle(checked('data_cache/test_features.pkl'))
    assert X.shape == (47073, 87) and X.index.equals(pd.RangeIndex(47073))
    assert list(X.columns) == schema['columns'] and set(FEATURES).issubset(X.columns)
    k = np.load(checked('outputs/round6/geography/final/test_source_cache_indices.npy'), allow_pickle=False)
    assert k.dtype == np.int64 and np.array_equal(k, np.arange(47073, dtype=np.int64))
    X = X.iloc[k].reset_index(drop=True)
    sample = proof['candidates']['76']['sample_ID_path']
    assert known[sample] == proof['candidates']['76']['sample_ID_sha256']
    payload = json.loads(checked(proof['candidates']['76']['export_payload_path']).read_text(encoding='utf8'))
    headers = pd.read_csv(checked(sample), nrows=0).columns.tolist()
    assert headers == payload['headers'] and len(headers) == 2
    id_column = headers[0]
    ids = pd.read_csv(checked(sample), usecols=[id_column])[id_column]
    assert len(ids) == 47073 and ids.notna().all() and ids.is_unique
    v = {name: np.load(checked(path), allow_pickle=False) for name, path in VECTORS.items()}
    for name, a in v.items():
        assert a.shape == (47073,) and a.dtype == np.float64 and np.isfinite(a).all() and (a >= 0).all(), name
    checks = {
        'v19_median_exact': np.array_equal(v['v19'], np.median(np.column_stack([v['NN'], v['log'], v['old_unit']]), axis=1)),
        'v20_median_exact': np.array_equal(v['v20'], np.median(np.column_stack([v['NN'], v['log'], v['new_unit']]), axis=1)),
        'v21_median_exact': np.array_equal(v['v21'], np.median(np.column_stack([v['weighted_NN'], v['log'], v['new_unit']]), axis=1)),
        'new_unit_base_exact_TWD_only': np.array_equal(v['new_unit_base'], .75 * v['unit_native_TWD'] + .25 * v['old_unit_LGB_TWD']),
        'canonical_test_indices_are_identity': True, 'all_vectors_finite_nonnegative_float64': True,
    }
    assert all(checks.values()), checks
    building = X['building_area'].to_numpy(dtype=np.float64)
    parking = X['parking_area'].to_numpy(dtype=np.float64)
    good_b = np.isfinite(building) & (building > 0); good_p = np.isfinite(parking) & (parking > 0)
    area = np.where(good_b, building, np.where(good_p, parking, 1.0))
    area_source = np.where(good_b, 'building_area', np.where(good_p, 'parking_area', 'default_1'))
    change = v['v20'] - v['v19']
    groups = {'highest_v20': np.argsort(-v['v20'], kind='stable')[:10],
              'largest_absolute_v20_minus_v19': np.argsort(-np.abs(change), kind='stable')[:10]}
    def record(i):
        i = int(i); branches = ['NN', 'log', 'new_unit']
        return {'canonical_test_position': i, 'ID': scalar(ids.iloc[i]),
            'features': {name: scalar(X.iloc[i][name]) for name in FEATURES},
            'predictions_TWD': {name: float(a[i]) for name, a in v.items()},
            'v20_minus_v19_TWD': float(change[i]), 'v20_minus_v19_percent': float(100 * change[i] / v['v19'][i]),
            'median_v19_matches': [name for name in ['NN', 'log', 'old_unit'] if v[name][i] == v['v19'][i]],
            'median_v20_matches': [name for name in branches if v[name][i] == v['v20'][i]],
            'median_v21_matches': [name for name in ['weighted_NN', 'log', 'new_unit'] if v[name][i] == v['v21'][i]],
            'v20_branch_order_ascending': sorted(branches, key=lambda name: (v[name][i], name)),
            'effective_area_m2': float(area[i]), 'effective_area_source': str(area_source[i]),
            'implied_TWD_per_effective_m2_not_observed_prices': {name: float(v[name][i] / area[i]) for name in ['v19', 'v20', 'v21', 'NN', 'log', 'new_unit', 'unit_native_TWD']},
            'new_unit_spatial_minus_reference_TWD': float(v['new_unit'][i] - v['new_unit_reference'][i])}
    records = {name: [record(i) for i in pos] for name, pos in groups.items()}
    all_groups = {'all_test': np.arange(47073), **groups}
    distributions = {name: {'rows': int(len(pos)),
        'predictions_TWD': {key: summary(v[key][pos]) for key in ['v19', 'v20', 'v21', 'NN', 'log', 'new_unit']},
        'effective_area_m2': summary(area[pos]),
        'v20_median_match_counts_ties_in_each_branch': median_matches(v, 'v20', ['NN', 'log', 'new_unit'], pos),
        'v19_median_match_counts_ties_in_each_branch': median_matches(v, 'v19', ['NN', 'log', 'old_unit'], pos),
        'feature_counts': {col: {str(key): int(value) for key, value in X.iloc[pos][col].value_counts(dropna=False).items()}
                           for col in ['transaction_type', 'building_type', 'main_use']},
        'area_source_counts': {str(key): int(value) for key, value in zip(*np.unique(area_source[pos], return_counts=True))}}
        for name, pos in all_groups.items()}
    after = {name: sha(ROOT / name) for name in protect}; assert after == before
    assert sha(__file__) == sources[str(Path(__file__).relative_to(ROOT)).replace('\\', '/')]
    report = {'round': 78, 'status': 'completed_target_free_tail_inspection',
        'plan': {'selection': 'Top 10 v20; top 10 absolute v20-v19 unrounded changes; stable row order resolves ties',
                 'threshold_search': False, 'ID_specific_changes': False, 'selection_updates': False},
        'rows': 47073, 'new_fits': 0, 'model_loads': 0, 'model_predict_calls': 0,
        'reference_predict_calls': 0, 'spatial_predict_calls': 0, 'target_labels_read': False,
        'sample_read_columns': [id_column], 'raw_addresses_or_remarks_exported': False,
        'MAE_or_error_estimated': False, 'prediction_changes_are_not_errors': True,
        'source_sha256': sources, 'input_sha256': inputs, 'vector_paths': VECTORS,
        'checks': checks, 'protected_before': before, 'protected_after': after,
        'area_rule': 'finite positive building_area; else finite positive parking_area; else 1',
        'unit_scale_source': 'R19 inverse already returns max(z*effective_area*1e4,0) TWD. R76/R77 base is .75 native_TWD+.25 old_unit_TWD; no further area multiplication.',
        'feature_parse_scope': 'Read saved canonical features and parser source only; no raw worksheet reparse or measurement validation. Numeric source area is copied, not multiplied by transaction counts.',
        'known_feature_schema_corrections': schema.get('verified_feature_corrections', []),
        'distributions': distributions, 'selected_records': records,
        'selected_union_rows': int(len(np.union1d(*groups.values()))),
        'changed_v20_vs_v19_rows': int(np.count_nonzero(change)),
        'limitations': ['Actual test prices are unknown: this is neither MAE nor evidence of over/underpricing.',
            'Large aggregate area and transaction counts can explain model scale without proving prediction accuracy.',
            'Area-linked branches can share extrapolation errors; median cannot remove shared errors.',
            'Arithmetic consistency excludes an extra area multiply in this final composition, not all model or upstream data errors.'],
        'prior_attempt': {'exit_code': 1, 'elapsed_seconds': 1.0273147,
            'source_sha256': '0e09e8f2d71230d4fbf256037ea63416a975d9a870d1dd07b3b8001ef2b97cb4',
            'error': 'Sample ID column was incorrectly assumed to be English ID; usecols did not match',
            'prediction_vectors_loaded': False, 'statistics_computed': False, 'reports_created': False,
            'correction': 'Read the official ID column from the SHA-bound export payload headers, compare sample header, then load that ID column only'},
        'execution': {'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                      'elapsed_seconds': time.monotonic() - tick, 'exit_code': 0}}
    lines = ['# R78：v20／v21 最終高價尾端描述性檢查', '',
        '只讀 47,073 列 test 特徵、sample ID 與保存的未取整 TWD 向量。沒有真價、模型載入、訓練或推論；沒有改價、上限、CSV 或選型更新。', '',
        f"v20 最大預測為 {v['v20'].max():,.2f} 元，v19 最大為 {v['v19'].max():,.2f} 元。固定兩份前 10 名名單合計 {report['selected_union_rows']} 個不同資料列；變動指預測差，不是誤差。", '',
        '全部列的 v19／v20／v21 保存中位數組成逐位一致；新 unit base 也逐位等於 0.75×已換算總價的 native TWD＋0.25×old unit TWD。R19 已在 inverse 乘過一次有效面積×10,000，這段 final 組成沒有再乘面積。', '',
        '原始面積欄由數值解析直接取得，交易土地／建物／車位數另外解析，沒有用數量乘面積。這次未重新解析原試算表，不能據此證明來源面積或所有樓層語法正確；既存 schema 記錄 ID 30528 的 floor_count 已由 8 修正為 7。', '']
    for name, label in [('highest_v20', 'v20 最高 10 列'), ('largest_absolute_v20_minus_v19', '相對 v19 絕對變動最大 10 列')]:
        lines += [f'## {label}', '', '金額單位為億元；面積為 m²。NN／log／新 unit 均是各自完成後處理的總價分支。完整建案、用途、樓層及三分支明細保存在 JSON，未輸出地址或備註。', '',
            '| ID | 建物面積 | 建物／車位數 | 類型／用途 | NN | log | 新 unit | v19 | v20 | 變動 | v20 中位來源 |',
            '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
        for row in records[name]:
            f = row['features']; p = row['predictions_TWD']
            typ = str(f['building_type']) + '／' + str(f['main_use'])
            lines += [f"| {row['ID']} | {f['building_area']} | {f['building_count']}／{f['parking_count']} | {typ.replace('|', '/')} | {p['NN']/1e8:.3f} | {p['log']/1e8:.3f} | {p['new_unit']/1e8:.3f} | {p['v19']/1e8:.3f} | {p['v20']/1e8:.3f} | {row['v20_minus_v19_TWD']/1e8:+.3f} | {','.join(row['median_v20_matches'])} |"]
        lines += ['', '中位來源計數（平手分別計入）：' + json.dumps(distributions[name]['v20_median_match_counts_ties_in_each_branch'], ensure_ascii=False) + '。', '']
    lines += ['## 解讀界線', '',
        '沒有在本次 final 算式找到再次乘面積的證據。大面積、整棟或多筆交易與高總價可以同時出現，但本機沒有 test 真價，不能斷言這些預測正確、過高或過低。若多個面積相關分支共同外推，最外中位數仍可能保留巨額預測；較小的舊預測也不是已知答案。', '',
        '本次只檢查事前指定兩組尾端，不由個案建立 cap、ID 規則或改模型；原 v19／v20／v21 CSV、registry 與本機 selector 前後 SHA 不變。', '',
        f"執行 {report['execution']['elapsed_seconds']:.3f} 秒、exit 0。可重跑來源：[round78_final_tail_inspection.py](C:/NYCU_ML/src/round78_final_tail_inspection.py)；完整記錄／來源 SHA：[JSON](C:/NYCU_ML/reports/round78_final_tail_inspection.json)。", '',
        f"Source SHA256：`{sources['src/round78_final_tail_inspection.py']}`。", '']
    with MD_OUT.open('x', encoding='utf8', newline='\n') as f:
        f.write('\n'.join(lines))
    report['report_sha256_at_generation'] = sha(MD_OUT)
    with JSON_OUT.open('x', encoding='utf8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': report['status'], 'exit_code': 0, 'elapsed_seconds': report['execution']['elapsed_seconds'],
        'source_sha256': sources['src/round78_final_tail_inspection.py'], 'json_sha256': sha(JSON_OUT),
        'report_sha256': sha(MD_OUT), 'top_record': records['highest_v20'][0],
        'top10_median': distributions['highest_v20']['v20_median_match_counts_ties_in_each_branch']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    assert args.run, 'Use --run; existing reports are never overwritten'
    run()
