"""Target-free Taoyuan literal-name inventory on the frozen R6 development split."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round13_research/alias_inventory.json'
REPORT = ROOT / 'reports/round13_alias_inventory.md'
PLAN = 'outputs/round6/candidate_plan.json'
GEO = 'outputs/round6/geography/development/manifest.json'
CACHE = 'data_cache/train_features.pkl'
SOURCE_INDEX = 'outputs/round6/geography/development/train_source_cache_indices.npy'
FIT_INDEX = 'outputs/round6/models/development/geography_catboost/fit_indices.npy'
QUERY_INDEX = 'outputs/round6/models/development/geography_catboost/query_indices.npy'
SCHEMA = 'reports/feature_schema.json'
DOCS = ['reports/round13_city_alias_feasibility.md', 'reports/round13_official_name_evidence.md']
CITIES = ['桃園縣', '桃園市']
MISSING = '__MISSING__'
FIELDS = ['address_city', 'district', 'address_district', 'address_road', 'location_road', 'trade_day_index']
PLAN_SHA = '606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8'
GEO_SHA = '4b21f0620aedb8e9527044e9a44a4e5d6f8cae49e71210252c56ba614a5151c6'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def content_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes(order='C')).hexdigest()


def locked_json(relative, expected):
    path = ROOT / relative
    digest = sha(path)
    require(digest == expected, 'Frozen JSON changed: ' + relative)
    receipt = path.with_suffix('.sha256')
    require(receipt.read_text(encoding='utf-8').strip().split()[0] == digest,
            'Receipt mismatch: ' + relative)
    return json.loads(path.read_text(encoding='utf-8'))


def check_index(values, upper, expected_rows, name):
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype('int64') and values.ndim == 1,
            name + ': expected one-dimensional int64')
    require(len(values) == expected_rows, name + ': row count differs')
    require(len(values) > 0 and values[0] >= 0 and values[-1] < upper, name + ': out of range')
    require(bool(np.all(np.diff(values) > 0)), name + ': indices must be unique and increasing')


def align(frame, source_index, fit_index, query_index, sizes):
    raw, canonical, fit_rows, query_rows = sizes
    require(len(frame) == raw, 'Feature cache row count differs')
    check_index(source_index, raw, canonical, 'source cache mapping')
    check_index(fit_index, canonical, fit_rows, 'fit indices')
    check_index(query_index, canonical, query_rows, 'query indices')
    require(len(np.intersect1d(fit_index, query_index)) == 0, 'Fit/query overlap')
    canonical_frame = frame.iloc[source_index]
    return canonical_frame, canonical_frame.iloc[fit_index], canonical_frame.iloc[query_index]


def text_values(series):
    # Missingness only; no administrative alias, numeral or spelling normalization.
    result = series.fillna(MISSING).astype(str)
    return result.mask(result.str.strip().eq(''), MISSING)


def summarize(frame):
    values = frame.loc[:, FIELDS[:-1]].apply(text_values)
    scoped = values.loc[values.address_city.isin(CITIES)].copy()
    pairs = []
    for key, count in scoped.groupby(['address_city', 'district', 'address_district'], sort=True, dropna=False).size().items():
        pairs.append(dict(zip(['address_city', 'district', 'address_district'], key), rows=int(count)))
    by_city = {}
    for city in CITIES:
        part = scoped.loc[scoped.address_city.eq(city)]
        both = part.district.ne(MISSING) & part.address_district.ne(MISSING)
        tokens = part.location_road.str.split('|', regex=False)
        well_formed = tokens.map(len).eq(3) & part.location_road.ne(MISSING)
        usable = tokens.loc[well_formed]
        components = {name: usable.str[pos] for pos, name in enumerate(['address_city', 'address_district', 'address_road'])}
        matching = {name: component.eq(part.loc[well_formed, name]) for name, component in components.items()}
        all_match = pd.Series(True, index=usable.index)
        for mask in matching.values():
            all_match &= mask
        by_city[city] = {
            'rows': len(part),
            'missing': {name: int(part[name].eq(MISSING).sum()) for name in ['district', 'address_district', 'address_road', 'location_road']},
            'district_both_present_rows': int(both.sum()),
            'district_literal_disagreement_rows': int((both & part.district.ne(part.address_district)).sum()),
            'child_literal_taoyuan_city': {
                'original_district_rows': int(part.district.eq('桃園市').sum()),
                'address_district_rows': int(part.address_district.eq('桃園市').sum()),
                'both_rows': int((part.district.eq('桃園市') & part.address_district.eq('桃園市')).sum()),
                'one_taoyuan_city_other_taoyuan_district_rows': int(((part.district.eq('桃園市') & part.address_district.eq('桃園區')) | (part.district.eq('桃園區') & part.address_district.eq('桃園市'))).sum())},
            'location_road_consistency': {
                'three_token_rows': int(well_formed.sum()),
                'missing_rows': int(part.location_road.eq(MISSING).sum()),
                'nonmissing_non_three_token_rows': int((~well_formed & part.location_road.ne(MISSING)).sum()),
                'component_mismatch_rows': {name: int((~mask).sum()) for name, mask in matching.items()},
                'all_three_components_match_rows': int(all_match.sum()),
                'district_token_differs_from_original_district_rows': int(components['address_district'].ne(part.loc[well_formed, 'district']).sum())}}
    literal_pairs = []
    for key, group in scoped.groupby(['district', 'address_district'], sort=True, dropna=False):
        counts = {city: int(group.address_city.eq(city).sum()) for city in CITIES}
        literal_pairs.append({'district': key[0], 'address_district': key[1], 'rows_by_parent_city': counts,
                              'observed_under_both_parent_spellings': all(counts.values())})
    return {'all_phase_rows': len(frame), 'scoped_rows': len(scoped), 'outside_scope_rows': len(frame) - len(scoped),
            'by_parent_city': by_city, 'city_original_address_district_counts': pairs,
            'identical_child_literal_pairs_across_parent_spellings': literal_pairs,
            'alias_mapping_inferred_or_applied': False}


def self_check():
    checks = []
    frame = pd.DataFrame({'value': np.arange(7)})
    source = np.array([0, 2, 3, 5, 6], dtype=np.int64)
    fi = np.array([0, 2], dtype=np.int64); qi = np.array([3, 4], dtype=np.int64)
    canonical, fit, query = align(frame, source, fi, qi, (7, 5, 2, 2))
    require(fit.value.tolist() == [0, 3] and query.value.tolist() == [5, 6] and canonical.index.tolist() == source.tolist(), 'Mapping order test')
    checks.append('canonical_then_phase_mapping_preserves_original_row_order')
    bad = [('duplicate', np.array([0, 2, 2, 5, 6], dtype=np.int64)),
           ('out_of_range', np.array([0, 2, 3, 5, 7], dtype=np.int64)),
           ('unordered', np.array([0, 3, 2, 5, 6], dtype=np.int64)), ('float_dtype', source.astype(float))]
    for name, value in bad:
        try:
            align(frame, value, fi, qi, (7, 5, 2, 2))
        except ValueError:
            checks.append('reject_' + name)
        else:
            raise AssertionError('Did not reject ' + name)
    try:
        align(frame, source, fi, np.array([2, 4], dtype=np.int64), (7, 5, 2, 2))
    except ValueError:
        checks.append('reject_fit_query_overlap')
    else:
        raise AssertionError('Did not reject overlapping phases')
    fixture = pd.DataFrame([
        ['桃園縣', '桃園市', '桃園市', '甲路', '桃園縣|桃園市|甲路', 1],
        ['桃園市', '桃園區', '桃園市', '乙路', '桃園市|桃園市|乙路', 2],
        ['桃園市', None, '桃園區', '丙路', '桃園縣|桃園市|丙路', 3],
        ['桃園市', '桃園區', '桃園區', '丁路', 'malformed', 4],
        ['臺北市', '桃園市', '桃園市', '戊路', '臺北市|桃園市|戊路', 5],
    ], columns=FIELDS)
    before = fixture.copy(deep=True)
    s = summarize(fixture); old = s['by_parent_city']['桃園縣']; new = s['by_parent_city']['桃園市']
    require(s['scoped_rows'] == 4 and old['rows'] == 1 and new['rows'] == 3, 'Parent filtering')
    checks.append('exact_parent_scope_prevents_child_name_collision')
    require(new['district_literal_disagreement_rows'] == 1 and new['missing']['district'] == 1, 'Missing versus mismatch')
    checks.append('missing_and_nonmissing_literal_conflict_separated')
    loc = new['location_road_consistency']
    require(loc['nonmissing_non_three_token_rows'] == 1 and loc['component_mismatch_rows'] == {'address_city': 1, 'address_district': 1, 'address_road': 0}, 'Location component check')
    checks.append('location_parent_district_mismatch_and_malformed_checked')
    require(old['child_literal_taoyuan_city']['both_rows'] == 1 and new['child_literal_taoyuan_city']['both_rows'] == 0, 'Child hierarchy test')
    checks.append('same_literal_child_kept_separate_by_parent')
    pd.testing.assert_frame_equal(fixture, before)
    serialized = json.dumps(s, ensure_ascii=False)
    require('甲路' not in serialized and 'address_raw' not in serialized and s['alias_mapping_inferred_or_applied'] is False, 'Output data minimization')
    checks.append('input_unchanged_no_raw_address_or_road_names_exported_no_alias')
    return checks


def render(result):
    lines = ['# R13 桃園名稱純 X 盤點', '',
             '本次是名稱證據盤點，尚未建立第 13 輪候選或訓練模型。只讀既有 87 欄特徵快取與固定索引，不讀原始試算表、目標價格、預測或分數，也不讀測試集。', '',
             '沿用 R6 的索引：455,787 筆原始列去除 3 筆夾入表頭後為 455,784；既有索引再排除 529 筆重複，得到 455,255 列。歷史去重曾包含目標欄，本盤點只重用保存的列映射，沒有重新讀價格或去重。訓練為 2022-08-11 前的 240,316 列；開發驗證為 2022-08-11 至 2023-08-10 的 81,234 列。', '',
             '本次允許讀開發驗證 X，因此它不是完全未接觸的盲測。以下僅保留 address_city 字面為「桃園縣」或「桃園市」的列，不做別名轉換。district 是原提供行政區欄的特徵表示；address_district 是既有地址解析結果。兩者字面不同不直接代表其中一者錯誤。', '',
             '| 時段 | 父層字面 | 列數 | 原 district 缺值 | 地址行政區缺值 | 兩者非缺且不同 | 子層兩欄同為桃園市 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for phase, title in [('development_fit', '開發訓練'), ('development_query', '開發驗證')]:
        for city, item in result['phases'][phase]['by_parent_city'].items():
            lines.append(f"| {title} | {city} | {item['rows']:,} | {item['missing']['district']:,} | {item['missing']['address_district']:,} | {item['district_literal_disagreement_rows']:,} | {item['child_literal_taoyuan_city']['both_rows']:,} |")
    lines += ['', '下表列出全部父層 × 原行政區 × 地址行政區組合，按字面排序；缺值以 __MISSING__ 表示。這是共現計數，不是歷史 13 區對照表。', '',
              '| 時段 | address_city | district | address_district | 列數 |', '|---|---|---|---|---:|']
    for phase, title in [('development_fit', '開發訓練'), ('development_query', '開發驗證')]:
        for item in result['phases'][phase]['city_original_address_district_counts']:
            lines.append(f"| {title} | {item['address_city']} | {item['district']} | {item['address_district']} | {item['rows']:,} |")
    lines += ['', 'location_road 檢查只拆既有的三個 `|` 分隔 token，核對父層、地址行政區及道路是否與相應特徵欄一致；不輸出地址或道路名稱。行政區 token 與原 district 不同另列，避免把既有解析差異誤記成組合欄損壞。', '',
              '| 時段 | 父層 | 三段格式列數 | 非缺但非三段 | 父層不符 | 地址行政區不符 | 道路不符 | 與原 district 不符 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for phase, title in [('development_fit', '開發訓練'), ('development_query', '開發驗證')]:
        for city, item in result['phases'][phase]['by_parent_city'].items():
            loc = item['location_road_consistency']; m = loc['component_mismatch_rows']
            lines.append(f"| {title} | {city} | {loc['three_token_rows']:,} | {loc['nonmissing_non_three_token_rows']:,} | {m['address_city']:,} | {m['address_district']:,} | {m['address_road']:,} | {loc['district_token_differs_from_original_district_rows']:,} |")
    lines += ['', '官方名稱文件只支持「桃園縣→桃園市」以及舊縣轄子層「桃園市→桃園區」的指定行政層級事實。本表中的字面共現不能證明任意 13 區別名或行政邊界等價；尤其父層「桃園市」與舊子層同名，若日後處理必須保留層級條件。本次沒有生成或套用任何別名表，也沒有提出分數門檻或推定價格效果。', '',
              f"合成檢查 {len(result['synthetic_checks'])} 項通過。真實映射已核對 int64、唯一遞增、範圍、列數、互斥、日期與凍結索引內容雜湊；輸入及來源雜湊在讀取前後一致。快取 schema 為 87 欄／21 類別／66 數值，且不含目標或 ID 欄。", '',
              '完整字面組合、缺值、子層歧義和來源 SHA 保存在 [盤點 JSON](../outputs/round13_research/alias_inventory.json)。重現方式是 `.venv\\Scripts\\python.exe -B -X utf8 src\\round13_alias_inventory.py --run`；程式拒絕覆寫既有結果，應在具相同來源的全新工作副本重現。單獨 `--self-check` 僅跑合成資料。', '',
              '參考：[可行性研究](round13_city_alias_feasibility.md)、[官方名稱證據](round13_official_name_evidence.md)、[稽核程式](../src/round13_alias_inventory.py)、[R6 地理對齊來源](../src/round6_prepare_geo.py)。', '']
    return '\n'.join(lines)


def run():
    targets = [OUT, REPORT, OUT.with_suffix('.sha256')]
    require(not any(path.exists() for path in targets), 'Refusing to overwrite existing or partial inventory')
    checks = self_check()
    plan = locked_json(PLAN, PLAN_SHA); geo = locked_json(GEO, GEO_SHA)
    require(plan['splits']['development'] == ['2022-08-11', '2023-08-10'], 'Development dates changed')
    contract = plan['split_contract']['development']
    require((contract['fit_rows'], contract['query_rows']) == (240316, 81234), 'Development counts changed')
    schema = json.loads((ROOT / SCHEMA).read_text(encoding='utf-8'))
    require(len(schema['columns']) == 87 and len(schema['categorical_columns']) == 21, 'Original schema changed')
    require(not set(schema['columns']) & {'編號', '總價元', 'target', 'y', 'ID'}, 'Target/ID in feature schema')
    inputs = [PLAN, str(Path(PLAN).with_suffix('.sha256')).replace('\\', '/'), GEO,
              str(Path(GEO).with_suffix('.sha256')).replace('\\', '/'), CACHE, SCHEMA, SOURCE_INDEX, FIT_INDEX, QUERY_INDEX] + DOCS
    sources = ['src/round13_alias_inventory.py', 'src/house_features.py', 'src/round6_prepare_geo.py']
    digests = {name: sha(ROOT / name) for name in inputs + sources}
    require(digests[CACHE] == plan['data_sha256'][CACHE], 'Pure-X cache differs from frozen R6')
    require(digests[SCHEMA] == plan['data_sha256'][SCHEMA], 'Feature schema differs from frozen R6')
    require(digests[SOURCE_INDEX] == geo['index_files_sha256'][SOURCE_INDEX], 'Canonical source mapping changed')
    require(digests['src/house_features.py'] == schema['feature_source_sha256'], 'Feature builder changed')
    require(digests['src/round6_prepare_geo.py'] == geo['source_sha256']['src/round6_prepare_geo.py'], 'Mapping source changed')
    for name in DOCS:
        (ROOT / name).read_text(encoding='utf-8')
    frame = pd.read_pickle(ROOT / CACHE)  # Frozen target-free feature cache only.
    require(isinstance(frame, pd.DataFrame) and list(frame.columns) == schema['columns'], 'Actual feature schema differs')
    dtype_counts = frame.dtypes.astype(str).value_counts().to_dict()
    frame = frame.loc[:, FIELDS]
    source_idx = np.load(ROOT / SOURCE_INDEX, allow_pickle=False)
    fi = np.load(ROOT / FIT_INDEX, allow_pickle=False); qi = np.load(ROOT / QUERY_INDEX, allow_pickle=False)
    canonical, fit, query = align(frame, source_idx, fi, qi, (455784, 455255, 240316, 81234))
    for values, key in [(fi, 'fit_indices_content_sha256'), (qi, 'query_indices_content_sha256')]:
        require(content_sha(values) == contract[key], 'Frozen index content differs: ' + key)
    days = canonical.trade_day_index.to_numpy(dtype=np.float64)
    origin = (pd.Timestamp('2022-08-11') - pd.Timestamp('2000-01-01')).days
    end = (pd.Timestamp('2023-08-10') - pd.Timestamp('2000-01-01')).days
    require(np.array_equal(fi, np.flatnonzero(np.isfinite(days) & (days < origin))), 'Fit/date selection differs')
    require(np.array_equal(qi, np.flatnonzero(np.isfinite(days) & (days >= origin) & (days <= end))), 'Query/date selection differs')
    result = {'purpose': 'target-free literal administrative name inventory; no formal candidate', 'round': 13,
              'read_boundaries': {'pure_X_cache_only': True, 'raw_workbooks_read': False, 'target_cache_read': False,
                                  'labels_predictions_or_scores_read': False, 'test_X_read': False, 'development_query_X_read': True,
                                  'development_query_is_untouched_blind': False, 'historical_target_containing_dedup_mapping_reused': True,
                                  'model_fit_or_scoring_performed': False, 'alias_mapping_inferred_or_applied': False},
              'scope': {'exact_parent_city_values': CITIES, 'fields_analyzed': FIELDS,
                        'missing_rule': 'pandas missing, empty/whitespace-only string, or literal __MISSING__; other text unchanged',
                        'conflict_definition': 'original and parsed districts both present but literal values differ; not a correctness judgment'},
              'alignment': {'raw_workbook_rows_from_existing_manifest': geo['alignment']['train']['original_workbook_rows'],
                            'header_removed_cache_rows': len(frame), 'canonical_rows': len(canonical), 'historical_duplicates_excluded': 529,
                            'fit_rows': len(fit), 'query_rows': len(query), 'fit_before': '2022-08-11',
                            'query_inclusive_dates': ['2022-08-11', '2023-08-10'], 'fit_query_overlap_rows': 0,
                            'indices_dtype_unique_increasing_range_order_and_dates_verified': True,
                            'source_cache_indices_content_sha256': content_sha(source_idx), **contract},
              'schema': {'feature_columns': 87, 'categorical_columns': 21, 'numeric_columns': 66, 'dtype_counts': dtype_counts},
              'phases': {'development_fit': summarize(fit), 'development_query': summarize(query)},
              'synthetic_checks': checks, 'source_sha256': {name: digests[name] for name in sources},
              'input_sha256': {name: digests[name] for name in inputs},
              'runtime_versions': {'numpy': np.__version__, 'pandas': pd.__version__}}
    report_bytes = render(result).encode('utf-8')
    result['report'] = {'path': REPORT.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(report_bytes).hexdigest()}
    for name, digest in digests.items():
        require(sha(ROOT / name) == digest, 'Read source changed during inventory: ' + name)
    result['input_and_source_hashes_unchanged_after_read'] = True
    payload = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stage = OUT.parent / ('.alias_inventory-' + uuid.uuid4().hex)
    stage.mkdir()
    blobs = [(OUT, payload), (REPORT, report_bytes), (OUT.with_suffix('.sha256'), (hashlib.sha256(payload).hexdigest() + '\n').encode('ascii'))]
    for i, (target, body) in enumerate(blobs):
        temp = stage / str(i)
        with temp.open('xb') as f:
            f.write(body)
    # Exclusive hard links publish complete individual files; receipt is published last.
    # A partial publication is retained and blocks a rerun rather than being overwritten.
    for i, (target, _) in enumerate(blobs):
        os.link(stage / str(i), target)
    for i in range(len(blobs)):
        (stage / str(i)).unlink()
    stage.rmdir()
    print(json.dumps({'status': 'complete', 'json_sha256': sha(OUT), 'report_sha256': sha(REPORT),
                      'script_sha256': digests['src/round13_alias_inventory.py'],
                      'counts': {phase: {city: item['rows'] for city, item in value['by_parent_city'].items()} for phase, value in result['phases'].items()},
                      'synthetic_checks': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true'); mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps({'status': 'passed', 'checks': self_check()}, ensure_ascii=False))
    else:
        run()
